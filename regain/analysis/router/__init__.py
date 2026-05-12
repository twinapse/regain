"""
Repair router analysis outputs.

This module implements the repair router post-processing analysis described in `PLAN.md`. It
consumes the staged frontier artifact (`frontier/selection.csv`) and writes a
set of router-specific tables, summary metrics, and a viability gate. Router logic
must not modify experiment orchestration, controller fitting, model training, repair
fitting, or MLflow execution.
"""

import json
from pathlib import Path
from typing import Any

from regain.analysis.router.actions import ACTION_ALIASES
from regain.analysis.router.actions import action_family_map
from regain.analysis.router.actions import collect_action_ids
from regain.analysis.router.actions import FORBIDDEN_FEATURE_PATTERNS
from regain.analysis.router.actions import resolve_action_family
from regain.analysis.router.actions import validate_router_feature_schema
from regain.analysis.router.constants import COVERAGE_THRESHOLD
from regain.analysis.router.constants import ROUTER_ALLOWED_CATEGORICAL_FEATURES
from regain.analysis.router.constants import ROUTER_ALLOWED_NUMERIC_FEATURES
from regain.analysis.router.constants import ROUTER_ID_COLUMNS
from regain.analysis.router.constants import ROUTER_POLICY_NAMES
from regain.analysis.router.constants import STATIC_LOW_COST_BASELINES
from regain.analysis.router.data import build_feature_row
from regain.analysis.router.data import build_label_row
from regain.analysis.router.data import collect_optional_drift_columns
from regain.analysis.router.data import FeaturePreprocessor
from regain.analysis.router.data import OPTIONAL_PRE_REPAIR_DRIFT_COLUMNS
from regain.analysis.router.data import read_selection_rows
from regain.analysis.router.evaluation import aggregate_policy_summary
from regain.analysis.router.evaluation import build_decision_gate
from regain.analysis.router.evaluation import evaluate_policies
from regain.analysis.router.folds import all_folds
from regain.analysis.router.folds import RouterFold
from regain.analysis.router.policies import build_policies
from regain.analysis.router.policies import RouterPolicy
from regain.analysis.utils import write_csv
from regain.utils import get_logger

__all__ = [
    'ACTION_ALIASES',
    'FORBIDDEN_FEATURE_PATTERNS',
    'FeaturePreprocessor',
    'OPTIONAL_PRE_REPAIR_DRIFT_COLUMNS',
    'ROUTER_ALLOWED_CATEGORICAL_FEATURES',
    'ROUTER_ALLOWED_NUMERIC_FEATURES',
    'ROUTER_ID_COLUMNS',
    'ROUTER_POLICY_NAMES',
    'STATIC_LOW_COST_BASELINES',
    'RouterFold',
    'RouterPolicy',
    'resolve_action_family',
    'validate_router_feature_schema',
    'write_repair_router_outputs',
]


def write_repair_router_outputs(
    *,
    analysis_dir: Path,
    out_dir: Path,
    random_state: int = 0,
) -> dict[str, Path]:
    """
    Write all router artifacts derived from `frontier/selection.csv`.

    Args:
        analysis_dir: Staged experiment analysis directory containing `frontier/`.
        out_dir: Output directory for router artifacts.
        random_state: Deterministic random seed used by learned policies.

    Returns:
        dict[str, Path]: Paths to the written router artifacts.

    Raises:
        FileNotFoundError: If `frontier/selection.csv` is missing.
        ValueError: If router feature schema validation fails.
    """
    logger = get_logger()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selection_path = Path(analysis_dir) / 'frontier' / 'selection.csv'
    selection_rows = read_selection_rows(path=selection_path)

    action_ids = collect_action_ids(rows=selection_rows)
    action_family_by_id = action_family_map(action_ids=action_ids)
    optional_drift_columns = collect_optional_drift_columns(rows=selection_rows)

    feature_rows = [build_feature_row(row=row, optional_drift_columns=optional_drift_columns) for row in selection_rows]
    label_rows = [
        build_label_row(
            row=row,
            action_ids=action_ids,
            action_family_by_id=action_family_by_id,
        ) for row in selection_rows
    ]
    feature_columns = list(feature_rows[0].keys()) if feature_rows else []
    invalid_columns = validate_router_feature_schema(feature_columns=feature_columns)
    if invalid_columns:
        raise ValueError('Router feature schema validation failed; invalid feature columns '
                         f'detected: {sorted(invalid_columns)}.')

    manifest_warnings: list[dict[str, Any]] = []
    folds, fold_metadata = all_folds(
        feature_rows=feature_rows,
        manifest_warnings=manifest_warnings,
    )
    prediction_rows = evaluate_policies(
        folds=folds,
        feature_rows=feature_rows,
        label_rows=label_rows,
        action_ids=action_ids,
        action_family_by_id=action_family_by_id,
        random_state=random_state,
        manifest_warnings=manifest_warnings,
    )
    summary_rows = aggregate_policy_summary(
        prediction_rows=prediction_rows,
        action_family_by_id=action_family_by_id,
        manifest_warnings=manifest_warnings,
    )
    decision_gate = build_decision_gate(summary_rows=summary_rows)

    features_path = out_dir / 'features.csv'
    labels_path = out_dir / 'labels.csv'
    predictions_path = out_dir / 'predictions.csv'
    summary_path = out_dir / 'policy_summary.csv'
    decision_gate_path = out_dir / 'decision_gate.json'
    manifest_path = out_dir / 'manifest.json'

    write_csv(features_path, feature_rows)
    write_csv(labels_path, label_rows)
    write_csv(predictions_path, prediction_rows)
    write_csv(summary_path, summary_rows)
    decision_gate_path.write_text(json.dumps(decision_gate, indent=2), encoding='utf-8')

    preprocessor = FeaturePreprocessor.fit(
        feature_rows,
        numeric_columns=ROUTER_ALLOWED_NUMERIC_FEATURES,
        categorical_columns=ROUTER_ALLOWED_CATEGORICAL_FEATURES,
    )
    manifest_payload: dict[str, Any] = {
        'schema': {
            'name': 'regain.analysis.router',
            'version': 1,
        },
        'random_state': int(random_state),
        'id_columns': list(ROUTER_ID_COLUMNS),
        'predictor_columns': {
            'categorical': list(ROUTER_ALLOWED_CATEGORICAL_FEATURES),
            'numeric': list(ROUTER_ALLOWED_NUMERIC_FEATURES),
            'optional_drift': list(optional_drift_columns),
            'expanded': preprocessor.feature_names(),
        },
        'label_columns': sorted(label_rows[0].keys()) if label_rows else [],
        'budget_treatment': {
            'role':
                'externally_fixed_input',
            'reason': ('repair budget is fixed before repair-selection policy choice; '
                       'the action set contains controller families only, not (family, budget) pairs'),
        },
        'cheap_pilot_features_included': False,
        'coverage_threshold': COVERAGE_THRESHOLD,
        'available_action_ids': list(action_ids),
        'action_family_by_id': action_family_by_id,
        'policies': [policy.name for policy in build_policies()],
        'validation_levels': fold_metadata,
        'leakage_check': {
            'passed': not invalid_columns,
            'forbidden_feature_columns': invalid_columns,
        },
        'warnings': manifest_warnings,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, default=str), encoding='utf-8')

    logger.info('Repair router outputs written: '
                f'{features_path}, {labels_path}, {predictions_path}, {summary_path}, '
                f'{decision_gate_path}, {manifest_path}')
    return {
        'features': features_path,
        'labels': labels_path,
        'predictions': predictions_path,
        'policy_summary': summary_path,
        'decision_gate': decision_gate_path,
        'manifest': manifest_path,
    }
