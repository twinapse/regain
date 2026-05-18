"""
Tests for the repair router analysis.
"""

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from regain.analysis.router import FORBIDDEN_FEATURE_PATTERNS
from regain.analysis.router import resolve_action_family
from regain.analysis.router import ROUTER_ALLOWED_CATEGORICAL_FEATURES
from regain.analysis.router import ROUTER_ALLOWED_NUMERIC_FEATURES
from regain.analysis.router import ROUTER_ID_COLUMNS
from regain.analysis.router import validate_router_feature_schema
from regain.analysis.router import write_repair_router_outputs


def _selection_row(
    *,
    seed: int = 1,
    budget: float = 0.5,
    no_op_utility_conservative: float = 0.0,
    bic_utility_conservative: float = 0.05,
    weight_aligning_utility_conservative: float = 0.02,
    bic_worst_harm: float = 0.02,
    bic_pareto: bool = True,
    no_op_worst_harm: float = 0.0,
    weight_aligning_worst_harm: float = 0.03,
) -> dict[str, Any]:
    """
    Construct a synthetic `selection.csv` row.

    Args:
        seed: Seed identifier.
        budget: Budget fraction.
        no_op_utility_conservative: Conservative utility for the no-op action.
        bic_utility_conservative: Conservative utility for the BiC action.
        weight_aligning_utility_conservative: Conservative utility for the weight-aligning action.
        bic_worst_harm: Worst-task harm for the BiC action.
        bic_pareto: Pareto flag for the BiC action.
        no_op_worst_harm: Worst-task harm for the no-op action.
        weight_aligning_worst_harm: Worst-task harm for the weight-aligning action.

    Returns:
        dict[str, Any]: Synthetic selection row.
    """
    row: dict[str, Any] = {
        'experiment_id': 'exp_1',
        'scenario': 'cifar100',
        'backbone_name': 'vit_small',
        'strategy_name': 'er',
        'seed': seed,
        'b': budget,
        'repair_budget_fraction': budget,
        'repair_budget_total': int(budget * 10),
        'repair_set_total': 10,
        'repair_split_fraction': 0.2,
        'num_classes': 10,
        'replay_mem_size': 100,
        'replay_batch_size_mem': 32,
        'mean_A_ref': 0.85,
        'mean_A_post': 0.65,
        'mean_forgetting': 0.2,
        'task_age_mean': 2.0,
        'task_age_min': 0,
        'task_age_max': 4,
        'task_age_std': 1.0,
        'oldest_task_forgetting': 0.25,
        'newest_task_forgetting': 0.05,
        'age_weighted_forgetting': 0.18,
        'run.calibration.max_ece': 0.1,
        'mean_run.calibration.ece': 0.08,
        'mean_run.calibration.aece': 0.09,
        'mean_run.calibration.nll': 0.3,
        'mean_run.diagnostics.out_of_task_rate': 0.15,
        'mean_run.diagnostics.avg_conf': 0.8,
        'mean_run.diagnostics.avg_entropy': 0.4,
        'mean_run.diagnostics.logit_avg_drift': 0.05,
        'utility_primary__no_op': no_op_utility_conservative,
        'utility_conservative__no_op': no_op_utility_conservative,
        'utility_cost_aware__no_op': no_op_utility_conservative,
        'mean_harmed_task_fraction__no_op': 0.0,
        'worst_task_harm__no_op': no_op_worst_harm,
        'is_pareto__no_op': False,
        'is_no_op_action__no_op': True,
        'action_repair_budget_fraction__no_op': 0.0,
        'action_repair_budget_total__no_op': 0,
        'utility_primary__bic': bic_utility_conservative,
        'utility_conservative__bic': bic_utility_conservative,
        'utility_cost_aware__bic': bic_utility_conservative,
        'mean_harmed_task_fraction__bic': 0.05,
        'worst_task_harm__bic': bic_worst_harm,
        'is_pareto__bic': bic_pareto,
        'is_no_op_action__bic': False,
        'action_repair_budget_fraction__bic': budget,
        'action_repair_budget_total__bic': int(budget * 10),
        'utility_primary__weight_aligning': weight_aligning_utility_conservative,
        'utility_conservative__weight_aligning': weight_aligning_utility_conservative,
        'utility_cost_aware__weight_aligning': weight_aligning_utility_conservative,
        'mean_harmed_task_fraction__weight_aligning': 0.04,
        'worst_task_harm__weight_aligning': weight_aligning_worst_harm,
        'is_pareto__weight_aligning': False,
        'is_no_op_action__weight_aligning': False,
        'action_repair_budget_fraction__weight_aligning': budget,
        'action_repair_budget_total__weight_aligning': int(budget * 10),
        'best_controller_by_utility_conservative': 'bic',
        'oracle_margin_vs_best_static_controller': 0.04,
    }
    return row


def _write_selection_csv(*, rows: list[dict[str, Any]], path: Path) -> None:
    """
    Write selection rows to CSV with a stable header.

    Args:
        rows: Synthetic selection rows.
        path: Output CSV path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """
    Read CSV rows back as dicts.

    Args:
        path: Path to a CSV file.

    Returns:
        list[dict[str, str]]: Parsed rows.
    """
    with path.open('r', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


class TestResolveActionFamily:
    """
    Tests for action family resolution.
    """

    def test_handles_normalized_aliases(self) -> None:
        assert resolve_action_family(action_id='no-op') == 'no_op'
        assert resolve_action_family(action_id='NoOp') == 'no_op'
        assert resolve_action_family(action_id='temperature') == 'temperature_scaling'
        assert resolve_action_family(action_id='ts_classifier') == 'temperature_scaling'
        assert resolve_action_family(action_id='wa') == 'weight_aligning'
        assert resolve_action_family(action_id='bias_correction_v2') == 'bic'
        assert resolve_action_family(action_id='custom_unknown_action') == 'custom_unknown_action'


class TestValidateRouterFeatureSchema:
    """
    Tests for router feature schema validation.
    """

    def test_rejects_outcome_columns(self) -> None:
        invalid = validate_router_feature_schema(feature_columns=[
            'scenario',
            'backbone_name',
            'utility_primary__bic',
            'best_controller_by_utility_primary',
            'mean_A_ctrl',
        ])
        assert 'utility_primary__bic' in invalid
        assert 'best_controller_by_utility_primary' in invalid
        assert 'mean_A_ctrl' in invalid
        assert 'backbone_name' not in invalid
        assert 'scenario' not in invalid

    def test_forbidden_patterns_include_double_underscore(self) -> None:
        assert '__' in FORBIDDEN_FEATURE_PATTERNS
        assert 'oracle' in FORBIDDEN_FEATURE_PATTERNS


class TestWriteRepairRouterOutputs:
    """
    Tests for repair router output writing.
    """

    def test_router_features_exclude_outcome_columns(self, tmp_path: Path) -> None:
        rows = [_selection_row(seed=seed, budget=0.5) for seed in (1, 2, 3)]
        for row in rows:
            row['mean_A_ctrl__bic'] = 0.7
        selection_path = tmp_path / 'analysis' / 'frontier' / 'selection.csv'
        _write_selection_csv(rows=rows, path=selection_path)

        paths = write_repair_router_outputs(
            analysis_dir=tmp_path / 'analysis',
            out_dir=tmp_path / 'router_out',
        )

        assert paths['features'].name == 'features.csv'
        assert paths['labels'].name == 'labels.csv'
        assert paths['predictions'].name == 'predictions.csv'
        assert paths['policy_summary'].name == 'policy_summary.csv'
        assert paths['decision_gate'].name == 'decision_gate.json'
        assert paths['manifest'].name == 'manifest.json'
        assert set(paths) == {
            'features',
            'labels',
            'predictions',
            'policy_summary',
            'decision_gate',
            'manifest',
        }
        assert {path.name for path in (tmp_path / 'router_out').iterdir()} == {
            'features.csv',
            'labels.csv',
            'predictions.csv',
            'policy_summary.csv',
            'decision_gate.json',
            'manifest.json',
        }

        feature_rows = _read_csv_rows(paths['features'])
        assert feature_rows
        feature_columns = list(feature_rows[0].keys())
        assert not any('__' in column for column in feature_columns)
        assert 'best_controller_by_utility_conservative' not in feature_columns
        assert 'mean_A_ctrl__bic' not in feature_columns

        label_rows = _read_csv_rows(paths['labels'])
        label_columns = set(label_rows[0].keys()) if label_rows else set()
        assert 'utility_conservative__bic' in label_columns
        assert 'mean_harmed_task_fraction__bic' in label_columns

    def test_router_features_raise_on_injected_outcome_columns(self, tmp_path: Path) -> None:
        rows = [_selection_row(seed=seed) for seed in (1, 2)]
        for row in rows:
            row['utility_primary__bic'] = 0.05
            row['best_controller_by_utility_primary'] = 'bic'
            row['mean_A_ctrl'] = 0.7
        selection_path = tmp_path / 'analysis' / 'frontier' / 'selection.csv'
        _write_selection_csv(rows=rows, path=selection_path)

        # End-to-end the writer still works (the allowlist filters the injected outcomes out before validation).
        paths = write_repair_router_outputs(
            analysis_dir=tmp_path / 'analysis',
            out_dir=tmp_path / 'router_out',
        )
        feature_rows = _read_csv_rows(paths['features'])
        feature_columns = list(feature_rows[0].keys())
        assert 'mean_A_ctrl' not in feature_columns
        assert 'utility_primary__bic' not in feature_columns

    def test_held_seed_folds_appear_in_manifest(self, tmp_path: Path) -> None:
        rows = [_selection_row(seed=seed, budget=0.5) for seed in (1, 2, 3)]
        selection_path = tmp_path / 'analysis' / 'frontier' / 'selection.csv'
        _write_selection_csv(rows=rows, path=selection_path)

        paths = write_repair_router_outputs(
            analysis_dir=tmp_path / 'analysis',
            out_dir=tmp_path / 'router_out',
        )
        manifest = json.loads(paths['manifest'].read_text(encoding='utf-8'))
        assert manifest['schema']['name'] == 'regain.analysis.router'
        assert manifest['schema']['version'] == 1
        held_seed = manifest['validation_levels']['held_seed']
        assert held_seed['num_folds'] == 3

    def test_manifest_separates_id_predictor_label_columns(self, tmp_path: Path) -> None:
        rows = [_selection_row(seed=seed, budget=0.5) for seed in (1, 2, 3)]
        selection_path = tmp_path / 'analysis' / 'frontier' / 'selection.csv'
        _write_selection_csv(rows=rows, path=selection_path)

        paths = write_repair_router_outputs(
            analysis_dir=tmp_path / 'analysis',
            out_dir=tmp_path / 'router_out',
        )
        manifest = json.loads(paths['manifest'].read_text(encoding='utf-8'))

        assert set(manifest['id_columns']) == set(ROUTER_ID_COLUMNS)

        predictor_columns = manifest['predictor_columns']
        assert set(predictor_columns['categorical']) == set(ROUTER_ALLOWED_CATEGORICAL_FEATURES)
        assert set(predictor_columns['numeric']) == set(ROUTER_ALLOWED_NUMERIC_FEATURES)
        assert isinstance(predictor_columns['optional_drift'], list)
        assert isinstance(predictor_columns['expanded'], list)
        assert predictor_columns['expanded']

        feature_rows = _read_csv_rows(paths['features'])
        features_columns = set(feature_rows[0].keys())
        predictor_union = (set(predictor_columns['categorical']) | set(predictor_columns['numeric']) |
                           set(predictor_columns['optional_drift']))
        assert predictor_union.issubset(features_columns)
        assert predictor_union.isdisjoint({'experiment_id', 'seed', 'b'})

        label_rows = _read_csv_rows(paths['labels'])
        assert label_rows
        assert set(manifest['label_columns']) == set(label_rows[0].keys())

        assert manifest['budget_treatment']['role'] == 'externally_fixed_input'
        assert 'reason' in manifest['budget_treatment']

    def test_best_static_low_cost_does_not_inspect_test_rows(self, tmp_path: Path) -> None:
        train_rows = []
        for seed in (1, 2):
            # Train rows favor BiC: bic utility > no-op utility.
            train_rows.append(
                _selection_row(
                    seed=seed,
                    budget=0.5,
                    bic_utility_conservative=0.10,
                    no_op_utility_conservative=0.00,
                    weight_aligning_utility_conservative=0.02,
                ))
        # Held-out test row strongly favors no-op (negative bic utility), but the policy
        # must not look at this row when choosing the static action.
        test_row = _selection_row(
            seed=3,
            budget=0.5,
            bic_utility_conservative=-0.20,
            no_op_utility_conservative=0.00,
            weight_aligning_utility_conservative=-0.10,
        )
        selection_path = tmp_path / 'analysis' / 'frontier' / 'selection.csv'
        _write_selection_csv(rows=[*train_rows, test_row], path=selection_path)

        paths = write_repair_router_outputs(
            analysis_dir=tmp_path / 'analysis',
            out_dir=tmp_path / 'router_out',
        )
        prediction_rows = _read_csv_rows(paths['predictions'])
        held_seed_predictions = [
            row for row in prediction_rows if row['validation_level'] == 'held_seed' and
            row['policy_name'] == 'best_static_low_cost_conservative' and int(row['seed']) == 3
        ]
        assert held_seed_predictions
        assert all(row['selected_action_id'] == 'bic' for row in held_seed_predictions)

    def test_two_stage_router_selects_no_op_when_all_actions_harmful(self, tmp_path: Path) -> None:
        # Construct rows where every active action has lower conservative utility than no-op AND positive harm.
        rows = []
        for seed in (1, 2, 3):
            rows.append(
                _selection_row(
                    seed=seed,
                    budget=0.5,
                    no_op_utility_conservative=0.05,
                    no_op_worst_harm=0.0,
                    bic_utility_conservative=-0.20,
                    bic_worst_harm=0.20,
                    bic_pareto=False,
                    weight_aligning_utility_conservative=-0.30,
                    weight_aligning_worst_harm=0.30,
                ))
        selection_path = tmp_path / 'analysis' / 'frontier' / 'selection.csv'
        _write_selection_csv(rows=rows, path=selection_path)

        paths = write_repair_router_outputs(
            analysis_dir=tmp_path / 'analysis',
            out_dir=tmp_path / 'router_out',
        )
        prediction_rows = _read_csv_rows(paths['predictions'])
        two_stage_predictions = [
            row for row in prediction_rows if row['policy_name'] == 'two_stage_expected_utility_router'
        ]
        assert two_stage_predictions
        assert all(row['selected_action_family'] == 'no_op' for row in two_stage_predictions)

        summary_rows = _read_csv_rows(paths['policy_summary'])
        two_stage_summary = [
            row for row in summary_rows
            if row['policy_name'] == 'two_stage_expected_utility_router' and row['validation_level'] == 'held_seed'
        ]
        assert two_stage_summary
        for row in two_stage_summary:
            recall = row.get('noop_recall')
            if recall and recall.strip():
                assert pytest.approx(float(recall)) == 1.0

    def test_metric_summary_aggregates_regret_and_action_distribution(self, tmp_path: Path) -> None:
        rows = [_selection_row(seed=seed, budget=0.5) for seed in (1, 2, 3)]
        selection_path = tmp_path / 'analysis' / 'frontier' / 'selection.csv'
        _write_selection_csv(rows=rows, path=selection_path)

        paths = write_repair_router_outputs(
            analysis_dir=tmp_path / 'analysis',
            out_dir=tmp_path / 'router_out',
        )

        summary_rows = _read_csv_rows(paths['policy_summary'])
        always_bic = next(row for row in summary_rows
                          if row['validation_level'] == 'held_seed' and row['policy_name'] == 'always_bic')
        # always_bic conservative utility for the synthetic dataset is 0.05.
        assert float(always_bic['mean_utility_conservative']) == pytest.approx(0.05)
        # Regret for the no-op policy is the gap to the oracle.
        always_no_op = next(row for row in summary_rows
                            if row['validation_level'] == 'held_seed' and row['policy_name'] == 'always_no_op')
        assert float(always_no_op['mean_utility_conservative']) == pytest.approx(0.0)
        assert float(always_no_op['mean_regret_conservative']) == pytest.approx(0.05)
        distribution = json.loads(always_no_op['selected_action_distribution_json'])
        assert distribution == {'no_op': 3}

    def test_decision_gate_writes_per_validation_level_payload(self, tmp_path: Path) -> None:
        rows = [_selection_row(seed=seed, budget=0.5) for seed in (1, 2, 3)]
        selection_path = tmp_path / 'analysis' / 'frontier' / 'selection.csv'
        _write_selection_csv(rows=rows, path=selection_path)

        paths = write_repair_router_outputs(
            analysis_dir=tmp_path / 'analysis',
            out_dir=tmp_path / 'router_out',
        )
        payload = json.loads(paths['decision_gate'].read_text(encoding='utf-8'))
        assert payload['schema']['name'] == 'regain.analysis.router.decision_gate'
        assert payload['schema']['version'] == 1
        assert 'held_seed' in payload['levels']
        level_payload = payload['levels']['held_seed']
        assert 'success' in level_payload
        assert 'recommended_next_step' in level_payload
        assert level_payload['recommended_next_step'] in {
            'router_viable_continue_refinement',
            'focus_on_safer_controllers_before_more_complex_router',
        }
