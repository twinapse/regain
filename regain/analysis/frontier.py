"""
Repairability frontier analysis outputs.
"""

import json
import math
from pathlib import Path
import re
from typing import Any
import uuid

from regain.analysis.artifacts import ARTIFACT_ACC_EXP_BASE
from regain.analysis.artifacts import ARTIFACT_ACC_FINAL_BASE
from regain.analysis.artifacts import ARTIFACT_ACC_FINAL_CTRL
from regain.analysis.utils import mean
from regain.analysis.utils import stdev
from regain.analysis.utils import to_float
from regain.analysis.utils import write_csv
from regain.constants import COLUMN_B
from regain.constants import COLUMN_CONTROLLER_MODEL_PARAM_COUNT
from regain.constants import COLUMN_CONTROLLER_NAME
from regain.constants import COLUMN_CONTROLLER_TYPE
from regain.constants import COLUMN_EXP_IDX
from regain.constants import COLUMN_EXPERIMENT_ID
from regain.constants import COLUMN_NUM_CLASSES
from regain.constants import COLUMN_REPAIR_BUDGET_FRACTION
from regain.constants import COLUMN_REPAIR_BUDGET_TOTAL
from regain.constants import COLUMN_REPAIR_SET_TOTAL
from regain.constants import COLUMN_REPAIR_SPLIT_FRACTION
from regain.constants import COLUMN_RUN_ID
from regain.constants import COLUMN_RUN_NAME
from regain.constants import COLUMN_SEED
from regain.constants import COLUMN_TASK_AGE
from regain.constants import RUN_CALIB_AECE
from regain.constants import RUN_CALIB_BRIER
from regain.constants import RUN_CALIB_ECE
from regain.constants import RUN_CALIB_MAX_ECE
from regain.constants import RUN_CALIB_MCE
from regain.constants import RUN_CALIB_NLL
from regain.constants import RUN_DIAG_AVG_CONF
from regain.constants import RUN_DIAG_AVG_ENTROPY
from regain.constants import RUN_DIAG_LOGIT_AVG_DRIFT
from regain.constants import RUN_DIAG_OUT_OF_TASK_RATE
from regain.constants import RUN_LATENCY_MS_PER_SAMPLE_BASE
from regain.constants import RUN_LATENCY_MS_PER_SAMPLE_CTRL
from regain.constants import RUN_LATENCY_MS_RATIO
from regain.constants import RUN_REPAIR_SECONDS
from regain.constants import RUN_REPAIR_STEPS
from regain.utils import get_logger

__all__ = [
    'write_repairability_frontier_outputs',
]

_COLUMN_A_REF = 'A_ref'
_COLUMN_A_POST = 'A_post'
_COLUMN_A_CTRL = 'A_ctrl'
_COLUMN_ABSOLUTE_RECOVERY = 'absolute_recovery'
_COLUMN_ACTION_REPAIR_BUDGET_FRACTION = 'action_repair_budget_fraction'
_COLUMN_ACTION_REPAIR_BUDGET_TOTAL = 'action_repair_budget_total'
_COLUMN_BACKBONE_NAME = 'backbone_name'
_COLUMN_CONTROLLER_ID = 'controller_id'
_COLUMN_FORGETTING = 'forgetting'
_COLUMN_HARM_MAGNITUDE = 'harm_magnitude'
_COLUMN_HELPED = 'helped'
_COLUMN_HARMED = 'harmed'
_COLUMN_IS_NO_OP_ACTION = 'is_no_op_action'
_COLUMN_IS_PARETO = 'is_pareto'
_COLUMN_MEAN_ABSOLUTE_RECOVERY = 'mean_absolute_recovery'
_COLUMN_MEAN_ACC_CTRL = 'mean_A_ctrl'
_COLUMN_MEAN_ACC_POST = 'mean_A_post'
_COLUMN_MEAN_ACC_REF = 'mean_A_ref'
_COLUMN_MEAN_FORGETTING = 'mean_forgetting'
_COLUMN_MEAN_HARMED_TASK_FRACTION = 'mean_harmed_task_fraction'
_COLUMN_MEAN_HARM_MAGNITUDE = 'mean_harm_magnitude'
_COLUMN_MEAN_HELPED_TASK_FRACTION = 'mean_helped_task_fraction'
_COLUMN_MEAN_RESIDUAL_FORGETTING = 'mean_residual_forgetting'
_COLUMN_MEAN_RHO = 'mean_rho'
_COLUMN_MEAN_TASK_DELTA = 'mean_task_delta'
_COLUMN_MEAN_LATENCY_MS_RATIO = 'mean_latency_ms_ratio'
_COLUMN_MEAN_REPAIR_SECONDS = 'mean_repair_seconds'
_COLUMN_MEAN_CONTROLLER_MODEL_PARAM_COUNT = 'mean_controller_model_param_count'
_COLUMN_NUM_PARETO = 'num_pareto'
_COLUMN_NUM_ROWS = 'num_rows'
_COLUMN_NUM_RUNS = 'num_runs'
_COLUMN_NUM_SEEDS = 'num_seeds'
_COLUMN_NUM_SETTINGS = 'num_settings'
_COLUMN_ORACLE_MARGIN = 'oracle_margin_vs_best_static_controller'
_COLUMN_PARETO_RATE = 'pareto_rate'
_COLUMN_RESIDUAL_FORGETTING = 'residual_forgetting'
_COLUMN_RHO = 'rho'
_COLUMN_RHO_VALID = 'rho_valid'
_COLUMN_RHO_VALID_FRACTION = 'rho_valid_fraction'
_COLUMN_SCENARIO = 'scenario'
_COLUMN_SEM_ABSOLUTE_RECOVERY = 'sem_absolute_recovery'
_COLUMN_SEM_RHO = 'sem_rho'
_COLUMN_SEM_UTILITY_CONSERVATIVE = 'sem_utility_conservative'
_COLUMN_SEM_UTILITY_PRIMARY = 'sem_utility_primary'
_COLUMN_SEM_WORST_TASK_HARM = 'sem_worst_task_harm'
_COLUMN_SOURCE_STAGE = 'source_stage'
_COLUMN_STRATEGY_NAME = 'strategy_name'
_COLUMN_TASK_DELTA = 'task_delta'
_COLUMN_UTILITY_CONSERVATIVE = 'utility_conservative'
_COLUMN_UTILITY_COST_AWARE = 'utility_cost_aware'
_COLUMN_UTILITY_PRIMARY = 'utility_primary'
_COLUMN_WORST_TASK_HARM = 'worst_task_harm'
_NO_OP_CONTROLLER_ID = 'no_op'
_NO_OP_CONTROLLER_NAME = 'no-op'
_NO_OP_SOURCE_STAGE = 'no_op'
_NO_OP_RUN_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL,
    'regain.analysis.frontier.no_op_run',
)

_FRONTIER_MAXIMIZE_COLUMNS = (
    _COLUMN_MEAN_ABSOLUTE_RECOVERY,
    _COLUMN_UTILITY_PRIMARY,
    _COLUMN_UTILITY_CONSERVATIVE,
)
_FRONTIER_MINIMIZE_COLUMNS = (
    _COLUMN_MEAN_HARMED_TASK_FRACTION,
    _COLUMN_WORST_TASK_HARM,
    RUN_LATENCY_MS_RATIO,
    RUN_REPAIR_SECONDS,
    COLUMN_CONTROLLER_MODEL_PARAM_COUNT,
    _COLUMN_ACTION_REPAIR_BUDGET_TOTAL,
)
_PAIRWISE_UTILITY_METRICS = (
    _COLUMN_UTILITY_PRIMARY,
    _COLUMN_UTILITY_CONSERVATIVE,
    _COLUMN_UTILITY_COST_AWARE,
    _COLUMN_MEAN_ABSOLUTE_RECOVERY,
    _COLUMN_MEAN_HARMED_TASK_FRACTION,
    _COLUMN_WORST_TASK_HARM,
    _COLUMN_IS_PARETO,
    _COLUMN_ACTION_REPAIR_BUDGET_FRACTION,
    _COLUMN_ACTION_REPAIR_BUDGET_TOTAL,
    _COLUMN_IS_NO_OP_ACTION,
)
_COLUMN_REPLAY_BATCH_SIZE_MEM = 'replay_batch_size_mem'
_COLUMN_REPLAY_MEM_SIZE = 'replay_mem_size'
_COLUMN_TASK_AGE_MEAN = 'task_age_mean'
_COLUMN_TASK_AGE_MIN = 'task_age_min'
_COLUMN_TASK_AGE_MAX = 'task_age_max'
_COLUMN_TASK_AGE_STD = 'task_age_std'
_COLUMN_OLDEST_TASK_FORGETTING = 'oldest_task_forgetting'
_COLUMN_NEWEST_TASK_FORGETTING = 'newest_task_forgetting'
_COLUMN_AGE_WEIGHTED_FORGETTING = 'age_weighted_forgetting'

_PRE_REPAIR_SELECTION_COLUMNS = (
    COLUMN_EXPERIMENT_ID,
    _COLUMN_SCENARIO,
    _COLUMN_BACKBONE_NAME,
    _COLUMN_STRATEGY_NAME,
    COLUMN_SEED,
    COLUMN_B,
    COLUMN_REPAIR_BUDGET_FRACTION,
    COLUMN_REPAIR_BUDGET_TOTAL,
    COLUMN_REPAIR_SET_TOTAL,
    COLUMN_REPAIR_SPLIT_FRACTION,
    COLUMN_NUM_CLASSES,
    _COLUMN_REPLAY_MEM_SIZE,
    _COLUMN_REPLAY_BATCH_SIZE_MEM,
    _COLUMN_MEAN_ACC_REF,
    _COLUMN_MEAN_ACC_POST,
    _COLUMN_MEAN_FORGETTING,
    _COLUMN_TASK_AGE_MEAN,
    _COLUMN_TASK_AGE_MIN,
    _COLUMN_TASK_AGE_MAX,
    _COLUMN_TASK_AGE_STD,
    _COLUMN_OLDEST_TASK_FORGETTING,
    _COLUMN_NEWEST_TASK_FORGETTING,
    _COLUMN_AGE_WEIGHTED_FORGETTING,
    RUN_CALIB_MAX_ECE,
    f'mean_{RUN_CALIB_ECE}',
    f'mean_{RUN_CALIB_AECE}',
    f'mean_{RUN_CALIB_NLL}',
    f'mean_{RUN_DIAG_OUT_OF_TASK_RATE}',
    f'mean_{RUN_DIAG_AVG_CONF}',
    f'mean_{RUN_DIAG_AVG_ENTROPY}',
    f'mean_{RUN_DIAG_LOGIT_AVG_DRIFT}',
)


def _sort_key(value: Any) -> tuple[int, str]:
    """
    Build a deterministic sort key for heterogeneous values.

    Args:
        value: Arbitrary sortable-ish value.

    Returns:
        tuple[int, str]: Tuple suitable for stable sorting.
    """
    return (0 if value is None else 1, '' if value is None else str(value))


def _frontier_setting_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """
    Build the canonical frontier-setting key.

    Args:
        row: Repair outcome row.

    Returns:
        tuple[Any, ...]: Canonical frontier-setting identity.
    """
    return (
        row.get(COLUMN_EXPERIMENT_ID),
        row.get(_COLUMN_SCENARIO),
        row.get(_COLUMN_BACKBONE_NAME),
        row.get(_COLUMN_STRATEGY_NAME),
        row.get(COLUMN_SEED),
        row.get(COLUMN_CONTROLLER_NAME),
        row.get(_COLUMN_CONTROLLER_ID),
        row.get(COLUMN_B),
        row.get(COLUMN_REPAIR_BUDGET_FRACTION),
        row.get(COLUMN_REPAIR_BUDGET_TOTAL),
    )


def _selection_setting_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """
    Build the canonical repair-selection setting key.

    Args:
        row: Frontier row.

    Returns:
        tuple[Any, ...]: Canonical selection-setting identity.
    """
    return (
        row.get(COLUMN_EXPERIMENT_ID),
        row.get(_COLUMN_SCENARIO),
        row.get(_COLUMN_BACKBONE_NAME),
        row.get(_COLUMN_STRATEGY_NAME),
        row.get(COLUMN_SEED),
        row.get(COLUMN_B),
        row.get(COLUMN_REPAIR_BUDGET_FRACTION),
        row.get(COLUMN_REPAIR_BUDGET_TOTAL),
    )


def _static_selection_slice_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """
    Build the canonical static-selection lookup slice key.

    Args:
        row: Frontier or selection row.

    Returns:
        tuple[Any, ...]: Canonical static-selection slice identity.
    """
    return (
        row.get(COLUMN_EXPERIMENT_ID),
        row.get(_COLUMN_SCENARIO),
        row.get(_COLUMN_BACKBONE_NAME),
        row.get(_COLUMN_STRATEGY_NAME),
        row.get(COLUMN_B),
        row.get(COLUMN_REPAIR_BUDGET_FRACTION),
        row.get(COLUMN_REPAIR_BUDGET_TOTAL),
    )


def _pareto_group_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """
    Build the canonical Pareto-group key.

    Args:
        row: Frontier row.

    Returns:
        tuple[Any, ...]: Canonical Pareto-group identity.
    """
    return (
        row.get(COLUMN_EXPERIMENT_ID),
        row.get(_COLUMN_SCENARIO),
        row.get(_COLUMN_BACKBONE_NAME),
        row.get(_COLUMN_STRATEGY_NAME),
        row.get(COLUMN_SEED),
    )


def _normalize_no_op_token(*, value: Any, fallback: str) -> str:
    """
    Normalize a no-op run-name token.

    Args:
        value: Candidate token value.
        fallback: Fallback token when the input is empty.

    Returns:
        str: Lowercase token with non-alphanumeric characters replaced by underscores.
    """
    normalized_value = re.sub(r'[^a-z0-9]+', '_', str(value or '').strip().lower()).strip('_')
    if normalized_value:
        return normalized_value
    return fallback


def _format_no_op_budget_token(*, repair_budget_fraction: Any) -> str:
    """
    Format the budget token for no-op run names.

    Args:
        repair_budget_fraction: Budget fraction value in [0, 1].

    Returns:
        str: `budget_<percent>` token using underscores for decimal separators.
    """
    budget_fraction = to_float(repair_budget_fraction)
    if budget_fraction is None:
        return 'budget_unknown'
    budget_percent = max(0.0, min(100.0, float(budget_fraction) * 100.0))
    if math.isclose(budget_percent, round(budget_percent), abs_tol=1e-9):
        formatted_percent = str(int(round(budget_percent)))
    else:
        formatted_percent = f'{budget_percent:.6f}'.rstrip('0').rstrip('.').replace('.', '_')
    return f'budget_{formatted_percent}'


def _build_no_op_run_name(*, exemplar: dict[str, Any]) -> str:
    """
    Build the canonical no-op run name.

    Args:
        exemplar: Representative repair-outcome row.

    Returns:
        str: `no_op-<scenario>-<backbone>-<strategy>-<budget>-<seed>` run name.
    """
    tokens = [
        _normalize_no_op_token(value=exemplar.get(_COLUMN_SCENARIO), fallback='unknown'),
        _normalize_no_op_token(value=exemplar.get(_COLUMN_BACKBONE_NAME), fallback='unknown'),
        _normalize_no_op_token(value=exemplar.get(_COLUMN_STRATEGY_NAME), fallback='unknown'),
        _format_no_op_budget_token(repair_budget_fraction=exemplar.get(COLUMN_REPAIR_BUDGET_FRACTION)),
        f"seed_{_normalize_no_op_token(value=exemplar.get(COLUMN_SEED), fallback='unknown')}",
    ]
    return f"no_op-{'-'.join(tokens)}"


def _normalize_no_op_identity_text(*, value: Any) -> str | None:
    """
    Normalize a text field for no-op run identity generation.

    Args:
        value: Candidate text value.

    Returns:
        str | None: Trimmed text value, or None when empty.
    """
    normalized_value = str(value or '').strip()
    if normalized_value:
        return normalized_value
    return None


def _normalize_no_op_identity_number(*, value: Any) -> int | float | None:
    """
    Normalize a numeric field for no-op run identity generation.

    Args:
        value: Candidate numeric value.

    Returns:
        int | float | None: Integer-like values as ints, finite numerics as floats, or None.
    """
    normalized_value = to_float(value)
    if normalized_value is None:
        return None
    if math.isclose(normalized_value, round(normalized_value), abs_tol=1e-9):
        return int(round(normalized_value))
    return float(normalized_value)


def _build_no_op_setting_payload(*, row: dict[str, Any]) -> dict[str, Any]:
    """
    Build the canonical no-op setting payload used for deterministic identity.

    Args:
        row: Representative repair-outcome row.

    Returns:
        dict[str, Any]: JSON-serializable setting payload.
    """
    return {
        'action': 'no_op',
        'experiment_id': _normalize_no_op_identity_text(value=row.get(COLUMN_EXPERIMENT_ID)),
        'scenario': _normalize_no_op_token(value=row.get(_COLUMN_SCENARIO), fallback='unknown'),
        'backbone_name': _normalize_no_op_token(value=row.get(_COLUMN_BACKBONE_NAME), fallback='unknown'),
        'strategy_name': _normalize_no_op_token(value=row.get(_COLUMN_STRATEGY_NAME), fallback='unknown'),
        'seed': _normalize_no_op_identity_number(value=row.get(COLUMN_SEED)),
        'b': _normalize_no_op_identity_number(value=row.get(COLUMN_B)),
        'repair_budget_fraction': _normalize_no_op_identity_number(value=row.get(COLUMN_REPAIR_BUDGET_FRACTION)),
        'repair_budget_total': _normalize_no_op_identity_number(value=row.get(COLUMN_REPAIR_BUDGET_TOTAL)),
    }


def _build_no_op_setting_tuple(*, row: dict[str, Any]) -> tuple[Any, ...]:
    """
    Build the normalized grouping tuple for one no-op repair setting.

    Args:
        row: Representative repair-outcome row.

    Returns:
        tuple[Any, ...]: Normalized setting values excluding task index.
    """
    payload = _build_no_op_setting_payload(row=row)
    return (
        payload['experiment_id'],
        payload['scenario'],
        payload['backbone_name'],
        payload['strategy_name'],
        payload['seed'],
        payload['b'],
        payload['repair_budget_fraction'],
        payload['repair_budget_total'],
    )


def _build_no_op_setting_key(*, row: dict[str, Any]) -> str:
    """
    Build the canonical string key for one no-op repair setting.

    Args:
        row: Representative repair-outcome row.

    Returns:
        str: Deterministic JSON key with sorted fields.
    """
    return json.dumps(
        _build_no_op_setting_payload(row=row),
        sort_keys=True,
        separators=(',', ':'),
    )


def _generate_no_op_run_id(*, row: dict[str, Any]) -> str:
    """
    Generate a deterministic MLflow-compatible no-op run id.

    The generated id is stable across reruns for the same no-op setting and
    shared by all synthesized task rows in that setting.

    Args:
        row: Representative repair-outcome row.

    Returns:
        str: 32-character lowercase hexadecimal run id.
    """
    return uuid.uuid5(
        _NO_OP_RUN_NAMESPACE,
        _build_no_op_setting_key(row=row),
    ).hex


def _sem(values: list[float | None]) -> float | None:
    """
    Compute the standard error of the mean for finite values.

    Args:
        values: Optional numeric values.

    Returns:
        float | None: Standard error of the mean, or None if fewer than 2 values are present.
    """
    finite = [value for value in values if value is not None]
    if len(finite) < 2:
        return None
    std_value = stdev(finite)
    if std_value is None:
        return None
    return float(std_value / math.sqrt(len(finite)))


def _append_warning(
    *,
    manifest: dict[str, Any],
    code: str,
    message: str,
    context: dict[str, Any] | None = None,
) -> None:
    """
    Append a structured warning to the manifest.

    Args:
        manifest: Mutable manifest payload.
        code: Stable warning code.
        message: Human-readable warning message.
        context: Optional structured context.
    """
    manifest.setdefault('warnings', []).append({
        'code': code,
        'message': message,
        'context': context or {},
    })


def _normalize_controller_slug(*, controller_name: str) -> str:
    """
    Normalize a controller name into a filesystem-safe identifier stem.

    Args:
        controller_name: Raw controller name.

    Returns:
        str: Normalized slug.
    """
    slug = re.sub(r'[^a-z0-9]+', '_', str(controller_name).lower()).strip('_')
    return slug or 'controller'


def _build_controller_id_map(
    *,
    controller_names: set[str],
    manifest: dict[str, Any],
) -> dict[str, str]:
    """
    Build stable controller ids with deterministic collision handling.

    Args:
        controller_names: Unique controller names present in repair outcomes.
        manifest: Mutable manifest payload.

    Returns:
        dict[str, str]: Mapping from controller name to normalized controller id.
    """
    grouped_names: dict[str, list[str]] = {}
    reserved_ids: set[str] = set()
    controller_id_map: dict[str, str] = {}
    controller_ids: list[dict[str, str]] = []
    if _NO_OP_CONTROLLER_NAME in controller_names:
        reserved_ids.add(_NO_OP_CONTROLLER_ID)
        controller_id_map[_NO_OP_CONTROLLER_NAME] = _NO_OP_CONTROLLER_ID
        controller_ids.append({
            COLUMN_CONTROLLER_NAME: _NO_OP_CONTROLLER_NAME,
            _COLUMN_CONTROLLER_ID: _NO_OP_CONTROLLER_ID,
        })

    for controller_name in sorted(controller_names):
        if controller_name == _NO_OP_CONTROLLER_NAME:
            continue
        grouped_names.setdefault(
            _normalize_controller_slug(controller_name=controller_name),
            [],
        ).append(controller_name)

    collisions: list[dict[str, Any]] = []
    for base_slug, names in sorted(grouped_names.items()):
        assigned_rows: list[dict[str, str]] = []
        start_index = 2 if base_slug in reserved_ids else 1
        for index, controller_name in enumerate(sorted(names), start=1):
            suffix = index + start_index - 1
            controller_id = base_slug if start_index == 1 and index == 1 else f'{base_slug}__{suffix}'
            controller_id_map[controller_name] = controller_id
            controller_ids.append({
                COLUMN_CONTROLLER_NAME: controller_name,
                _COLUMN_CONTROLLER_ID: controller_id,
            })
            assigned_rows.append({
                COLUMN_CONTROLLER_NAME: controller_name,
                _COLUMN_CONTROLLER_ID: controller_id,
            })
        if len(names) > 1 or base_slug in reserved_ids:
            collisions.append({
                'normalized_base': base_slug,
                'assignments': assigned_rows,
            })
            _append_warning(
                manifest=manifest,
                code='controller_id_collision',
                message=(f'Normalized controller id collision for `{base_slug}`. '
                         'Deterministic numeric suffixes were assigned.'),
                context={
                    'normalized_base': base_slug,
                    'controller_names': sorted(names),
                    'reserved_controller_ids': sorted(reserved_ids),
                },
            )

    manifest['controller_ids'] = controller_ids
    manifest['controller_id_collisions'] = collisions
    return controller_id_map


def _normalize_accuracy_value(
    *,
    raw_value: Any,
    field_name: str,
    run_id: str,
    exp_idx: Any,
    manifest: dict[str, Any],
) -> float | None:
    """
    Validate an accuracy-like field is already in the `[0, 1]` range.

    Args:
        raw_value: Original value from a collected table.
        field_name: Field name (`A_ref`, `A_post`, or `A_ctrl`).
        run_id: Run identifier for diagnostics.
        exp_idx: Experience index for diagnostics.
        manifest: Mutable manifest payload.

    Returns:
        float | None: Value in `[0, 1]`, or None if the input is invalid.
    """
    value = to_float(raw_value)
    if value is None:
        return None
    if 0.0 <= value <= 1.0:
        return value

    suspicious_rows = manifest.setdefault('normalization', {}).setdefault('suspicious_values', [])
    suspicious_rows.append({
        COLUMN_RUN_ID: run_id,
        COLUMN_EXP_IDX: exp_idx,
        'field': field_name,
        'raw_value': raw_value,
    })
    return None


def _derive_outcome_metrics(*, a_ref: float | None, a_post: float | None, a_ctrl: float | None) -> dict[str, Any]:
    """
    Derive row-level repair metrics from normalized accuracy values.

    Args:
        a_ref: Reference accuracy.
        a_post: Post-sequence base accuracy.
        a_ctrl: Controller-on accuracy.

    Returns:
        dict[str, Any]: Derived row metrics.
    """
    if a_ref is None or a_post is None or a_ctrl is None:
        return {
            _COLUMN_FORGETTING: None,
            _COLUMN_ABSOLUTE_RECOVERY: None,
            _COLUMN_RESIDUAL_FORGETTING: None,
            _COLUMN_RHO: None,
            _COLUMN_RHO_VALID: False,
            _COLUMN_TASK_DELTA: None,
            _COLUMN_HELPED: None,
            _COLUMN_HARMED: None,
            _COLUMN_HARM_MAGNITUDE: None,
        }

    forgetting = float(a_ref - a_post)
    absolute_recovery = float(a_ctrl - a_post)
    residual_forgetting = float(a_ref - a_ctrl)
    rho = None
    rho_valid = False
    if forgetting > 1e-6:
        rho = float(absolute_recovery / forgetting)
        rho_valid = True
    task_delta = float(a_ctrl - a_post)
    helped = task_delta > 1e-6
    harmed = task_delta < -1e-6
    harm_magnitude = float(max(0.0, a_post - a_ctrl))

    return {
        _COLUMN_FORGETTING: forgetting,
        _COLUMN_ABSOLUTE_RECOVERY: absolute_recovery,
        _COLUMN_RESIDUAL_FORGETTING: residual_forgetting,
        _COLUMN_RHO: rho,
        _COLUMN_RHO_VALID: rho_valid,
        _COLUMN_TASK_DELTA: task_delta,
        _COLUMN_HELPED: helped,
        _COLUMN_HARMED: harmed,
        _COLUMN_HARM_MAGNITUDE: harm_magnitude,
    }


def _is_valid_controller_on_outcome(
    *,
    controller_name: Any,
    controller_type: Any,
    a_ctrl: float | None,
) -> bool:
    """
    Determine whether a row belongs in repair-outcome analysis.

    Args:
        controller_name: Controller name from the collected tables.
        controller_type: Controller type from the collected tables.
        a_ctrl: Normalized controller-on accuracy.

    Returns:
        bool: True when the row has a valid controller-on outcome.
    """
    if a_ctrl is None:
        return False
    if str(controller_type or '').strip().lower() != 'repair':
        return False
    name = str(controller_name or '').strip().lower()
    if name in {'', 'none', 'null', 'backbone'}:
        return False
    if re.fullmatch(r'no[-_ ]*op', name):
        return False
    return True


def _build_repair_outcome_candidates(
    *,
    runs_table: list[dict[str, Any]],
    experiences_table: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Join collected run and experience tables into normalized repair outcomes.

    Args:
        runs_table: Collected run-level table rows.
        experiences_table: Collected experience-level table rows.
        manifest: Mutable manifest payload.

    Returns:
        list[dict[str, Any]]: Repair-outcome rows without controller ids applied.
    """
    runs_by_id = {str(row.get(COLUMN_RUN_ID)): row for row in runs_table if row.get(COLUMN_RUN_ID) is not None}
    excluded_rows = manifest.setdefault('excluded_rows', {
        'missing_run': 0,
        'invalid_controller_outcome': 0,
    })
    candidates: list[dict[str, Any]] = []
    for experience_row in experiences_table:
        run_id = str(experience_row.get(COLUMN_RUN_ID) or '')
        run_row = runs_by_id.get(run_id)
        if run_row is None:
            excluded_rows['missing_run'] += 1
            _append_warning(
                manifest=manifest,
                code='missing_run_for_experience',
                message='Skipped an experience row because its run-level row was missing.',
                context={
                    COLUMN_RUN_ID: run_id,
                    COLUMN_EXP_IDX: experience_row.get(COLUMN_EXP_IDX),
                },
            )
            continue

        exp_idx = experience_row.get(COLUMN_EXP_IDX)
        a_ref = _normalize_accuracy_value(
            raw_value=experience_row.get(ARTIFACT_ACC_EXP_BASE),
            field_name=_COLUMN_A_REF,
            run_id=run_id,
            exp_idx=exp_idx,
            manifest=manifest,
        )
        a_post = _normalize_accuracy_value(
            raw_value=experience_row.get(ARTIFACT_ACC_FINAL_BASE),
            field_name=_COLUMN_A_POST,
            run_id=run_id,
            exp_idx=exp_idx,
            manifest=manifest,
        )
        a_ctrl = _normalize_accuracy_value(
            raw_value=experience_row.get(ARTIFACT_ACC_FINAL_CTRL),
            field_name=_COLUMN_A_CTRL,
            run_id=run_id,
            exp_idx=exp_idx,
            manifest=manifest,
        )
        controller_name = experience_row.get(COLUMN_CONTROLLER_NAME)
        controller_type = experience_row.get(COLUMN_CONTROLLER_TYPE)
        if not _is_valid_controller_on_outcome(
                controller_name=controller_name,
                controller_type=controller_type,
                a_ctrl=a_ctrl,
        ):
            excluded_rows['invalid_controller_outcome'] += 1
            continue

        candidate = {
            COLUMN_EXPERIMENT_ID: run_row.get(COLUMN_EXPERIMENT_ID),
            _COLUMN_SCENARIO: run_row.get(_COLUMN_SCENARIO),
            _COLUMN_STRATEGY_NAME: run_row.get(_COLUMN_STRATEGY_NAME),
            COLUMN_RUN_ID: run_id,
            COLUMN_RUN_NAME: run_row.get(COLUMN_RUN_NAME),
            COLUMN_SEED: run_row.get(COLUMN_SEED),
            COLUMN_CONTROLLER_NAME: controller_name,
            COLUMN_CONTROLLER_TYPE: controller_type,
            COLUMN_REPAIR_BUDGET_FRACTION: run_row.get(COLUMN_REPAIR_BUDGET_FRACTION),
            COLUMN_REPAIR_BUDGET_TOTAL: run_row.get(COLUMN_REPAIR_BUDGET_TOTAL),
            _COLUMN_ACTION_REPAIR_BUDGET_FRACTION: run_row.get(COLUMN_REPAIR_BUDGET_FRACTION),
            _COLUMN_ACTION_REPAIR_BUDGET_TOTAL: run_row.get(COLUMN_REPAIR_BUDGET_TOTAL),
            COLUMN_REPAIR_SET_TOTAL: run_row.get(COLUMN_REPAIR_SET_TOTAL),
            COLUMN_REPAIR_SPLIT_FRACTION: run_row.get(COLUMN_REPAIR_SPLIT_FRACTION),
            COLUMN_NUM_CLASSES: run_row.get(COLUMN_NUM_CLASSES),
            COLUMN_B: run_row.get(COLUMN_B),
            COLUMN_CONTROLLER_MODEL_PARAM_COUNT: run_row.get(COLUMN_CONTROLLER_MODEL_PARAM_COUNT),
            COLUMN_EXP_IDX: exp_idx,
            COLUMN_TASK_AGE: experience_row.get(COLUMN_TASK_AGE),
            _COLUMN_A_REF: a_ref,
            _COLUMN_A_POST: a_post,
            _COLUMN_A_CTRL: a_ctrl,
            RUN_CALIB_MAX_ECE: run_row.get(RUN_CALIB_MAX_ECE),
            RUN_CALIB_ECE: experience_row.get(RUN_CALIB_ECE),
            RUN_CALIB_AECE: experience_row.get(RUN_CALIB_AECE),
            RUN_CALIB_MCE: experience_row.get(RUN_CALIB_MCE),
            RUN_CALIB_NLL: experience_row.get(RUN_CALIB_NLL),
            RUN_CALIB_BRIER: experience_row.get(RUN_CALIB_BRIER),
            RUN_DIAG_OUT_OF_TASK_RATE: experience_row.get(RUN_DIAG_OUT_OF_TASK_RATE),
            RUN_DIAG_AVG_CONF: experience_row.get(RUN_DIAG_AVG_CONF),
            RUN_DIAG_AVG_ENTROPY: experience_row.get(RUN_DIAG_AVG_ENTROPY),
            RUN_DIAG_LOGIT_AVG_DRIFT: experience_row.get(RUN_DIAG_LOGIT_AVG_DRIFT),
            RUN_LATENCY_MS_PER_SAMPLE_BASE: run_row.get(RUN_LATENCY_MS_PER_SAMPLE_BASE),
            RUN_LATENCY_MS_PER_SAMPLE_CTRL: run_row.get(RUN_LATENCY_MS_PER_SAMPLE_CTRL),
            RUN_LATENCY_MS_RATIO: run_row.get(RUN_LATENCY_MS_RATIO),
            RUN_REPAIR_SECONDS: run_row.get(RUN_REPAIR_SECONDS),
            RUN_REPAIR_STEPS: run_row.get(RUN_REPAIR_STEPS),
            _COLUMN_SOURCE_STAGE: 'collect',
            _COLUMN_IS_NO_OP_ACTION: False,
            _COLUMN_BACKBONE_NAME: run_row.get(_COLUMN_BACKBONE_NAME),
            _COLUMN_REPLAY_MEM_SIZE: run_row.get(_COLUMN_REPLAY_MEM_SIZE),
            _COLUMN_REPLAY_BATCH_SIZE_MEM: run_row.get(_COLUMN_REPLAY_BATCH_SIZE_MEM),
        }
        candidate.update(_derive_outcome_metrics(
            a_ref=a_ref,
            a_post=a_post,
            a_ctrl=a_ctrl,
        ))
        candidates.append(candidate)

    normalization = manifest.setdefault('normalization', {})
    suspicious_values = normalization.setdefault('suspicious_values', [])
    normalization['suspicious_value_count'] = len(suspicious_values)
    if suspicious_values:
        _append_warning(
            manifest=manifest,
            code='suspicious_accuracy_values',
            message='Some accuracy-like values were outside the expected [0, 1] range and were dropped.',
            context={
                'count': len(suspicious_values),
            },
        )
    return candidates


def _has_numeric_mismatch(*, values: list[Any], tolerance: float = 1e-6) -> bool:
    """
    Determine whether a list of optional numeric values disagrees.

    Args:
        values: Candidate values.
        tolerance: Allowed absolute spread.

    Returns:
        bool: True when the values are inconsistent.
    """
    normalized_values = [to_float(value) for value in values]
    finite_values = [value for value in normalized_values if value is not None]
    if finite_values and len(finite_values) != len(normalized_values):
        return True
    if len(finite_values) <= 1:
        return False
    return bool(max(finite_values) - min(finite_values) > tolerance)


def _append_no_op_candidates(
    *,
    candidates: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Append one no-op action per observed repair-comparison setting.

    Args:
        candidates: Real repair-outcome candidates.
        manifest: Mutable manifest payload.

    Returns:
        list[dict[str, Any]]: Real and no-op repair-outcome candidates.
    """
    grouped_rows: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in candidates:
        key = (
            *_build_no_op_setting_tuple(row=row),
            row.get(COLUMN_EXP_IDX),
        )
        grouped_rows.setdefault(key, []).append(row)

    no_op_rows: list[dict[str, Any]] = []
    no_op_run_ids: dict[str, str] = {}
    no_op_run_names: dict[str, str] = {}
    for key, rows in sorted(grouped_rows.items(), key=lambda item: tuple(_sort_key(value) for value in item[0])):
        exemplar = rows[0]
        setting_key = _build_no_op_setting_key(row=exemplar)
        no_op_run_id = no_op_run_ids.setdefault(setting_key, _generate_no_op_run_id(row=exemplar))
        no_op_run_name = no_op_run_names.setdefault(setting_key, _build_no_op_run_name(exemplar=exemplar))
        a_ref_values = [row.get(_COLUMN_A_REF) for row in rows]
        a_post_values = [row.get(_COLUMN_A_POST) for row in rows]
        if _has_numeric_mismatch(values=a_ref_values) or _has_numeric_mismatch(values=a_post_values):
            _append_warning(
                manifest=manifest,
                code='no_op_baseline_mismatch',
                message=('No-op baseline inputs disagreed across repair controllers. '
                         'The first repair row was used as the source of truth.'),
                context={
                    COLUMN_EXPERIMENT_ID: key[0],
                    _COLUMN_SCENARIO: key[1],
                    _COLUMN_BACKBONE_NAME: key[2],
                    _COLUMN_STRATEGY_NAME: key[3],
                    COLUMN_SEED: key[4],
                    COLUMN_B: key[5],
                    COLUMN_REPAIR_BUDGET_FRACTION: key[6],
                    COLUMN_REPAIR_BUDGET_TOTAL: key[7],
                    COLUMN_EXP_IDX: key[8],
                    'a_ref_values': a_ref_values,
                    'a_post_values': a_post_values,
                    'source_controllers': [row.get(COLUMN_CONTROLLER_NAME) for row in rows],
                },
            )

        no_op_row = dict(exemplar)
        no_op_row.update({
            COLUMN_RUN_ID: no_op_run_id,
            COLUMN_RUN_NAME: no_op_run_name,
            COLUMN_CONTROLLER_NAME: _NO_OP_CONTROLLER_NAME,
            COLUMN_CONTROLLER_TYPE: 'none',
            COLUMN_CONTROLLER_MODEL_PARAM_COUNT: 0,
            _COLUMN_A_CTRL: exemplar.get(_COLUMN_A_POST),
            _COLUMN_ACTION_REPAIR_BUDGET_FRACTION: 0.0,
            _COLUMN_ACTION_REPAIR_BUDGET_TOTAL: 0,
            RUN_LATENCY_MS_PER_SAMPLE_CTRL: exemplar.get(RUN_LATENCY_MS_PER_SAMPLE_BASE),
            RUN_LATENCY_MS_RATIO: 1.0,
            RUN_REPAIR_SECONDS: 0.0,
            RUN_REPAIR_STEPS: 0,
            _COLUMN_SOURCE_STAGE: _NO_OP_SOURCE_STAGE,
            _COLUMN_IS_NO_OP_ACTION: True,
        })
        no_op_row.update(
            _derive_outcome_metrics(
                a_ref=to_float(exemplar.get(_COLUMN_A_REF)),
                a_post=to_float(exemplar.get(_COLUMN_A_POST)),
                a_ctrl=to_float(exemplar.get(_COLUMN_A_POST)),
            ))
        no_op_rows.append(no_op_row)

    return [*candidates, *no_op_rows]


def _mean_bool(rows: list[dict[str, Any]], *, key: str) -> float | None:
    """
    Compute the mean of a nullable boolean column.

    Args:
        rows: Input rows.
        key: Boolean-like field name.

    Returns:
        float | None: Mean fraction or None when no values are present.
    """
    values: list[float | None] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            values.append(None)
        else:
            values.append(1.0 if bool(value) else 0.0)
    return mean(values)


def _max_value(rows: list[dict[str, Any]], *, key: str) -> float | None:
    """
    Compute the max of a nullable numeric field.

    Args:
        rows: Input rows.
        key: Field name.

    Returns:
        float | None: Max finite value, or None when no values are present.
    """
    values = [to_float(row.get(key)) for row in rows]
    finite = [value for value in values if value is not None]
    if not finite:
        return None
    return max(finite)


def _first_non_null(rows: list[dict[str, Any]], *, key: str) -> Any:
    """
    Return the first non-null value for a key across a row group.

    Args:
        rows: Input rows.
        key: Field name.

    Returns:
        Any: First non-null value, or None when missing.
    """
    for row in rows:
        value = row.get(key)
        if value is not None:
            return value
    return None


def _compute_task_age_summaries(*, rows: list[dict[str, Any]]) -> dict[str, float | None]:
    """
    Compute controller-independent task-age summary fields.

    Args:
        rows: Per-task repair-outcome rows for a single setting.

    Returns:
        dict[str, float | None]: Task-age summary values keyed by column name.
    """
    task_age_values: list[float] = []
    forgetting_by_age: list[tuple[float, float]] = []
    for item in rows:
        task_age_value = to_float(item.get(COLUMN_TASK_AGE))
        if task_age_value is None:
            continue
        task_age_values.append(task_age_value)
        forgetting_value = to_float(item.get(_COLUMN_FORGETTING))
        if forgetting_value is not None:
            forgetting_by_age.append((task_age_value, forgetting_value))

    if not task_age_values:
        return {
            _COLUMN_TASK_AGE_MEAN: None,
            _COLUMN_TASK_AGE_MIN: None,
            _COLUMN_TASK_AGE_MAX: None,
            _COLUMN_TASK_AGE_STD: None,
            _COLUMN_OLDEST_TASK_FORGETTING: None,
            _COLUMN_NEWEST_TASK_FORGETTING: None,
            _COLUMN_AGE_WEIGHTED_FORGETTING: None,
        }

    task_age_min = min(task_age_values)
    task_age_max = max(task_age_values)

    oldest_forgetting: float | None = None
    newest_forgetting: float | None = None
    weighted_sum = 0.0
    weight_sum = 0.0
    for age_value, forgetting_value in forgetting_by_age:
        weight = age_value + 1.0
        weighted_sum += weight * forgetting_value
        weight_sum += weight
        if age_value == task_age_max and oldest_forgetting is None:
            oldest_forgetting = forgetting_value
        if age_value == task_age_min and newest_forgetting is None:
            newest_forgetting = forgetting_value

    age_weighted_forgetting: float | None
    if weight_sum > 0.0 and forgetting_by_age:
        age_weighted_forgetting = float(weighted_sum / weight_sum)
    else:
        age_weighted_forgetting = None

    return {
        _COLUMN_TASK_AGE_MEAN: mean(task_age_values),
        _COLUMN_TASK_AGE_MIN: float(task_age_min),
        _COLUMN_TASK_AGE_MAX: float(task_age_max),
        _COLUMN_TASK_AGE_STD: stdev(task_age_values),
        _COLUMN_OLDEST_TASK_FORGETTING: oldest_forgetting,
        _COLUMN_NEWEST_TASK_FORGETTING: newest_forgetting,
        _COLUMN_AGE_WEIGHTED_FORGETTING: age_weighted_forgetting,
    }


def _aggregate_frontier_rows(*, repair_outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Aggregate repair outcomes into per-setting frontier rows.

    Args:
        repair_outcomes: Row-level repair outcomes.

    Returns:
        list[dict[str, Any]]: Aggregated frontier rows.
    """
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in repair_outcomes:
        key = _frontier_setting_key(row)
        groups.setdefault(key, []).append(row)

    frontier_rows: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(_sort_key(value) for value in item[0])):
        (
            experiment_id,
            scenario,
            backbone_name,
            strategy_name,
            seed,
            controller_name,
            controller_id,
            b_value,
            repair_budget_fraction,
            repair_budget_total,
        ) = key
        mean_repair_budget_total = mean([to_float(item.get(COLUMN_REPAIR_BUDGET_TOTAL)) for item in rows])
        row = {
            COLUMN_EXPERIMENT_ID:
                experiment_id,
            _COLUMN_SCENARIO:
                scenario,
            _COLUMN_BACKBONE_NAME:
                backbone_name,
            _COLUMN_STRATEGY_NAME:
                strategy_name,
            COLUMN_SEED:
                seed,
            COLUMN_CONTROLLER_NAME:
                controller_name,
            _COLUMN_CONTROLLER_ID:
                controller_id,
            COLUMN_B:
                b_value,
            COLUMN_REPAIR_BUDGET_FRACTION:
                repair_budget_fraction,
            COLUMN_REPAIR_BUDGET_TOTAL:
                (mean_repair_budget_total if mean_repair_budget_total is not None else to_float(repair_budget_total)),
            _COLUMN_ACTION_REPAIR_BUDGET_FRACTION:
                mean([to_float(item.get(_COLUMN_ACTION_REPAIR_BUDGET_FRACTION)) for item in rows]),
            _COLUMN_ACTION_REPAIR_BUDGET_TOTAL:
                mean([to_float(item.get(_COLUMN_ACTION_REPAIR_BUDGET_TOTAL)) for item in rows]),
            COLUMN_REPAIR_SET_TOTAL:
                mean([to_float(item.get(COLUMN_REPAIR_SET_TOTAL)) for item in rows]),
            COLUMN_REPAIR_SPLIT_FRACTION:
                mean([to_float(item.get(COLUMN_REPAIR_SPLIT_FRACTION)) for item in rows]),
            COLUMN_NUM_CLASSES:
                mean([to_float(item.get(COLUMN_NUM_CLASSES)) for item in rows]),
            COLUMN_CONTROLLER_MODEL_PARAM_COUNT:
                mean([to_float(item.get(COLUMN_CONTROLLER_MODEL_PARAM_COUNT)) for item in rows]),
            _COLUMN_IS_NO_OP_ACTION:
                _first_non_null(rows, key=_COLUMN_IS_NO_OP_ACTION),
            _COLUMN_NUM_ROWS:
                len(rows),
            _COLUMN_NUM_RUNS:
                len({row.get(COLUMN_RUN_ID) for row in rows}),
            _COLUMN_MEAN_ACC_REF:
                mean([to_float(item.get(_COLUMN_A_REF)) for item in rows]),
            _COLUMN_MEAN_ACC_POST:
                mean([to_float(item.get(_COLUMN_A_POST)) for item in rows]),
            _COLUMN_MEAN_ACC_CTRL:
                mean([to_float(item.get(_COLUMN_A_CTRL)) for item in rows]),
            _COLUMN_MEAN_FORGETTING:
                mean([to_float(item.get(_COLUMN_FORGETTING)) for item in rows]),
            _COLUMN_MEAN_ABSOLUTE_RECOVERY:
                mean([to_float(item.get(_COLUMN_ABSOLUTE_RECOVERY)) for item in rows]),
            _COLUMN_MEAN_RESIDUAL_FORGETTING:
                mean([to_float(item.get(_COLUMN_RESIDUAL_FORGETTING)) for item in rows]),
            _COLUMN_MEAN_RHO:
                mean([to_float(item.get(_COLUMN_RHO)) for item in rows]),
            _COLUMN_RHO_VALID_FRACTION:
                _mean_bool(rows, key=_COLUMN_RHO_VALID),
            _COLUMN_MEAN_TASK_DELTA:
                mean([to_float(item.get(_COLUMN_TASK_DELTA)) for item in rows]),
            _COLUMN_MEAN_HELPED_TASK_FRACTION:
                _mean_bool(rows, key=_COLUMN_HELPED),
            _COLUMN_MEAN_HARMED_TASK_FRACTION:
                _mean_bool(rows, key=_COLUMN_HARMED),
            _COLUMN_MEAN_HARM_MAGNITUDE:
                mean([to_float(item.get(_COLUMN_HARM_MAGNITUDE)) for item in rows]),
            _COLUMN_WORST_TASK_HARM:
                _max_value(rows, key=_COLUMN_HARM_MAGNITUDE),
            f'mean_{RUN_CALIB_ECE}':
                mean([to_float(item.get(RUN_CALIB_ECE)) for item in rows]),
            f'mean_{RUN_CALIB_AECE}':
                mean([to_float(item.get(RUN_CALIB_AECE)) for item in rows]),
            f'mean_{RUN_CALIB_NLL}':
                mean([to_float(item.get(RUN_CALIB_NLL)) for item in rows]),
            f'mean_{RUN_DIAG_OUT_OF_TASK_RATE}':
                mean([to_float(item.get(RUN_DIAG_OUT_OF_TASK_RATE)) for item in rows]),
            f'mean_{RUN_DIAG_AVG_CONF}':
                mean([to_float(item.get(RUN_DIAG_AVG_CONF)) for item in rows]),
            f'mean_{RUN_DIAG_AVG_ENTROPY}':
                mean([to_float(item.get(RUN_DIAG_AVG_ENTROPY)) for item in rows]),
            f'mean_{RUN_DIAG_LOGIT_AVG_DRIFT}':
                mean([to_float(item.get(RUN_DIAG_LOGIT_AVG_DRIFT)) for item in rows]),
            RUN_CALIB_MAX_ECE:
                mean([to_float(item.get(RUN_CALIB_MAX_ECE)) for item in rows]),
            RUN_LATENCY_MS_PER_SAMPLE_BASE:
                mean([to_float(item.get(RUN_LATENCY_MS_PER_SAMPLE_BASE)) for item in rows]),
            RUN_LATENCY_MS_PER_SAMPLE_CTRL:
                mean([to_float(item.get(RUN_LATENCY_MS_PER_SAMPLE_CTRL)) for item in rows]),
            RUN_LATENCY_MS_RATIO:
                mean([to_float(item.get(RUN_LATENCY_MS_RATIO)) for item in rows]),
            RUN_REPAIR_SECONDS:
                mean([to_float(item.get(RUN_REPAIR_SECONDS)) for item in rows]),
            RUN_REPAIR_STEPS:
                mean([to_float(item.get(RUN_REPAIR_STEPS)) for item in rows]),
            _COLUMN_REPLAY_MEM_SIZE:
                _first_non_null(rows, key=_COLUMN_REPLAY_MEM_SIZE),
            _COLUMN_REPLAY_BATCH_SIZE_MEM:
                _first_non_null(rows, key=_COLUMN_REPLAY_BATCH_SIZE_MEM),
        }
        row.update(_compute_task_age_summaries(rows=rows))
        row[_COLUMN_UTILITY_PRIMARY] = _compute_utility_primary(row=row)
        row[_COLUMN_UTILITY_CONSERVATIVE] = _compute_utility_conservative(row=row)
        row[_COLUMN_UTILITY_COST_AWARE] = _compute_utility_cost_aware(row=row)
        frontier_rows.append(row)

    return frontier_rows


def _compute_utility_primary(*, row: dict[str, Any]) -> float | None:
    """
    Compute `utility_primary` for a frontier row.

    Args:
        row: Frontier row.

    Returns:
        float | None: Utility value.
    """
    mean_absolute_recovery = to_float(row.get(_COLUMN_MEAN_ABSOLUTE_RECOVERY))
    mean_harm_magnitude = to_float(row.get(_COLUMN_MEAN_HARM_MAGNITUDE))
    if mean_absolute_recovery is None or mean_harm_magnitude is None:
        return None
    return float(mean_absolute_recovery - mean_harm_magnitude)


def _compute_utility_conservative(*, row: dict[str, Any]) -> float | None:
    """
    Compute `utility_conservative` for a frontier row.

    Args:
        row: Frontier row.

    Returns:
        float | None: Utility value.
    """
    mean_absolute_recovery = to_float(row.get(_COLUMN_MEAN_ABSOLUTE_RECOVERY))
    worst_task_harm = to_float(row.get(_COLUMN_WORST_TASK_HARM))
    harmed_task_fraction = to_float(row.get(_COLUMN_MEAN_HARMED_TASK_FRACTION))
    if mean_absolute_recovery is None or worst_task_harm is None or harmed_task_fraction is None:
        return None
    return float(mean_absolute_recovery - 0.5 * worst_task_harm - 0.01 * harmed_task_fraction)


def _compute_utility_cost_aware(*, row: dict[str, Any]) -> float | None:
    """
    Compute `utility_cost_aware` for a frontier row.

    Args:
        row: Frontier row.

    Returns:
        float | None: Utility value.
    """
    utility_conservative = to_float(row.get(_COLUMN_UTILITY_CONSERVATIVE))
    latency_ms_ratio = to_float(row.get(RUN_LATENCY_MS_RATIO))
    repair_seconds = to_float(row.get(RUN_REPAIR_SECONDS))
    if utility_conservative is None or latency_ms_ratio is None or repair_seconds is None:
        return None
    return float(utility_conservative - 0.001 * max(0.0, latency_ms_ratio - 1.0) -
                 0.0001 * math.log1p(max(0.0, repair_seconds)))


def _dominates(*, left: dict[str, Any], right: dict[str, Any], dimensions: dict[str, tuple[str, ...]]) -> bool:
    """
    Test whether one frontier row Pareto-dominates another.

    Args:
        left: Candidate dominating row.
        right: Candidate dominated row.
        dimensions: Retained maximize/minimize dimensions.

    Returns:
        bool: True if `left` dominates `right`.
    """
    no_worse = True
    strictly_better = False

    for key in dimensions['maximize']:
        left_value = to_float(left.get(key))
        right_value = to_float(right.get(key))
        if left_value is None or right_value is None:
            return False
        if left_value < right_value:
            no_worse = False
            break
        if left_value > right_value:
            strictly_better = True

    if not no_worse:
        return False

    for key in dimensions['minimize']:
        left_value = to_float(left.get(key))
        right_value = to_float(right.get(key))
        if left_value is None or right_value is None:
            return False
        if left_value > right_value:
            no_worse = False
            break
        if left_value < right_value:
            strictly_better = True

    return bool(no_worse and strictly_better)


def _annotate_pareto_membership(*, frontier_rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    """
    Annotate frontier rows with Pareto membership.

    Args:
        frontier_rows: Mutable frontier rows.
        manifest: Mutable manifest payload.
    """
    grouped_rows: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in frontier_rows:
        key = _pareto_group_key(row)
        grouped_rows.setdefault(key, []).append(row)

    pareto_summary: list[dict[str, Any]] = []
    for key, rows in sorted(grouped_rows.items(), key=lambda item: tuple(_sort_key(value) for value in item[0])):
        retained_maximize = tuple(column for column in _FRONTIER_MAXIMIZE_COLUMNS if any(
            to_float(row.get(column)) is not None for row in rows))
        retained_minimize = tuple(column for column in _FRONTIER_MINIMIZE_COLUMNS if any(
            to_float(row.get(column)) is not None for row in rows))
        dimensions = {
            'maximize': retained_maximize,
            'minimize': retained_minimize,
        }
        if not retained_maximize and not retained_minimize:
            for row in rows:
                row[_COLUMN_IS_PARETO] = None
            _append_warning(
                manifest=manifest,
                code='pareto_no_dimensions',
                message='A Pareto group had no usable dimensions and was skipped.',
                context={
                    COLUMN_EXPERIMENT_ID: key[0],
                    _COLUMN_SCENARIO: key[1],
                    _COLUMN_BACKBONE_NAME: key[2],
                    _COLUMN_STRATEGY_NAME: key[3],
                    COLUMN_SEED: key[4],
                },
            )
            pareto_summary.append({
                COLUMN_EXPERIMENT_ID: key[0],
                _COLUMN_SCENARIO: key[1],
                _COLUMN_BACKBONE_NAME: key[2],
                _COLUMN_STRATEGY_NAME: key[3],
                COLUMN_SEED: key[4],
                'retained_dimensions': [],
                'num_evaluable_rows': 0,
            })
            continue

        evaluable_rows: list[dict[str, Any]] = []
        retained_dimensions = [*retained_maximize, *retained_minimize]
        for row in rows:
            if any(to_float(row.get(column)) is None for column in retained_dimensions):
                row[_COLUMN_IS_PARETO] = None
                _append_warning(
                    manifest=manifest,
                    code='pareto_missing_dimension',
                    message=(
                        'A frontier row was excluded from Pareto checks because it was missing a retained dimension.'),
                    context={
                        COLUMN_EXPERIMENT_ID: row.get(COLUMN_EXPERIMENT_ID),
                        _COLUMN_SCENARIO: row.get(_COLUMN_SCENARIO),
                        _COLUMN_BACKBONE_NAME: row.get(_COLUMN_BACKBONE_NAME),
                        _COLUMN_STRATEGY_NAME: row.get(_COLUMN_STRATEGY_NAME),
                        COLUMN_SEED: row.get(COLUMN_SEED),
                        COLUMN_CONTROLLER_NAME: row.get(COLUMN_CONTROLLER_NAME),
                        _COLUMN_CONTROLLER_ID: row.get(_COLUMN_CONTROLLER_ID),
                        'retained_dimensions': retained_dimensions,
                    },
                )
                continue
            evaluable_rows.append(row)

        for row in evaluable_rows:
            dominated = any(
                _dominates(left=other, right=row, dimensions=dimensions)
                for other in evaluable_rows
                if other is not row)
            row[_COLUMN_IS_PARETO] = not dominated

        pareto_summary.append({
            COLUMN_EXPERIMENT_ID: key[0],
            _COLUMN_SCENARIO: key[1],
            _COLUMN_BACKBONE_NAME: key[2],
            _COLUMN_STRATEGY_NAME: key[3],
            COLUMN_SEED: key[4],
            'retained_dimensions': retained_dimensions,
            'num_evaluable_rows': len(evaluable_rows),
        })

    manifest['pareto_groups'] = pareto_summary


def _aggregate_repair_pareto_rows(*, frontier_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Aggregate frontier rows by controller and budget into Pareto-rate summaries.

    Args:
        frontier_rows: Frontier rows with Pareto annotations.

    Returns:
        list[dict[str, Any]]: Repair Pareto summary rows.
    """
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in frontier_rows:
        key = (
            row.get(COLUMN_CONTROLLER_NAME),
            row.get(_COLUMN_CONTROLLER_ID),
            row.get(COLUMN_B),
            row.get(COLUMN_REPAIR_BUDGET_FRACTION),
        )
        groups.setdefault(key, []).append(row)

    result_rows: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(_sort_key(value) for value in item[0])):
        controller_name, controller_id, b_value, repair_budget_fraction = key
        num_pareto = sum(1 for row in rows if row.get(_COLUMN_IS_PARETO) is True)
        num_settings = len(rows)
        result_rows.append({
            COLUMN_CONTROLLER_NAME:
                controller_name,
            _COLUMN_CONTROLLER_ID:
                controller_id,
            COLUMN_B:
                b_value,
            COLUMN_REPAIR_BUDGET_FRACTION:
                repair_budget_fraction,
            _COLUMN_ACTION_REPAIR_BUDGET_FRACTION:
                mean([to_float(row.get(_COLUMN_ACTION_REPAIR_BUDGET_FRACTION)) for row in rows]),
            _COLUMN_ACTION_REPAIR_BUDGET_TOTAL:
                mean([to_float(row.get(_COLUMN_ACTION_REPAIR_BUDGET_TOTAL)) for row in rows]),
            _COLUMN_IS_NO_OP_ACTION:
                _first_non_null(rows, key=_COLUMN_IS_NO_OP_ACTION),
            _COLUMN_NUM_SETTINGS:
                num_settings,
            _COLUMN_NUM_PARETO:
                num_pareto,
            _COLUMN_PARETO_RATE:
                float(num_pareto / num_settings) if num_settings > 0 else None,
            f'mean_{_COLUMN_UTILITY_PRIMARY}':
                mean([to_float(row.get(_COLUMN_UTILITY_PRIMARY)) for row in rows]),
            f'mean_{_COLUMN_UTILITY_CONSERVATIVE}':
                mean([to_float(row.get(_COLUMN_UTILITY_CONSERVATIVE)) for row in rows]),
            _COLUMN_MEAN_ABSOLUTE_RECOVERY:
                mean([to_float(row.get(_COLUMN_MEAN_ABSOLUTE_RECOVERY)) for row in rows]),
            _COLUMN_MEAN_HARMED_TASK_FRACTION:
                mean([to_float(row.get(_COLUMN_MEAN_HARMED_TASK_FRACTION)) for row in rows]),
            f'mean_{_COLUMN_WORST_TASK_HARM}':
                mean([to_float(row.get(_COLUMN_WORST_TASK_HARM)) for row in rows]),
            _COLUMN_MEAN_LATENCY_MS_RATIO:
                mean([to_float(row.get(RUN_LATENCY_MS_RATIO)) for row in rows]),
            _COLUMN_MEAN_REPAIR_SECONDS:
                mean([to_float(row.get(RUN_REPAIR_SECONDS)) for row in rows]),
            _COLUMN_MEAN_CONTROLLER_MODEL_PARAM_COUNT:
                mean([to_float(row.get(COLUMN_CONTROLLER_MODEL_PARAM_COUNT)) for row in rows]),
        })

    return result_rows


def _aggregate_repair_impact_rows(*, frontier_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Aggregate frontier rows across seeds for scenario-level impact summaries.

    Args:
        frontier_rows: Frontier rows with one row per setting.

    Returns:
        list[dict[str, Any]]: Repair impact rows.
    """
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in frontier_rows:
        key = (
            row.get(_COLUMN_SCENARIO),
            row.get(_COLUMN_BACKBONE_NAME),
            row.get(_COLUMN_STRATEGY_NAME),
            row.get(COLUMN_CONTROLLER_NAME),
            row.get(_COLUMN_CONTROLLER_ID),
            row.get(COLUMN_B),
            row.get(COLUMN_REPAIR_BUDGET_FRACTION),
            row.get(COLUMN_REPAIR_BUDGET_TOTAL),
        )
        groups.setdefault(key, []).append(row)

    result_rows: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(_sort_key(value) for value in item[0])):
        (
            scenario,
            backbone_name,
            strategy_name,
            controller_name,
            controller_id,
            b_value,
            repair_budget_fraction,
            repair_budget_total,
        ) = key
        mean_absolute_recovery_values = [to_float(row.get(_COLUMN_MEAN_ABSOLUTE_RECOVERY)) for row in rows]
        mean_rho_values = [to_float(row.get(_COLUMN_MEAN_RHO)) for row in rows]
        helped_task_fraction_values = [to_float(row.get(_COLUMN_MEAN_HELPED_TASK_FRACTION)) for row in rows]
        harmed_task_fraction_values = [to_float(row.get(_COLUMN_MEAN_HARMED_TASK_FRACTION)) for row in rows]
        worst_task_harm_values = [to_float(row.get(_COLUMN_WORST_TASK_HARM)) for row in rows]
        utility_primary_values = [to_float(row.get(_COLUMN_UTILITY_PRIMARY)) for row in rows]
        utility_conservative_values = [to_float(row.get(_COLUMN_UTILITY_CONSERVATIVE)) for row in rows]
        result_rows.append({
            _COLUMN_SCENARIO:
                scenario,
            _COLUMN_BACKBONE_NAME:
                backbone_name,
            _COLUMN_STRATEGY_NAME:
                strategy_name,
            COLUMN_CONTROLLER_NAME:
                controller_name,
            _COLUMN_CONTROLLER_ID:
                controller_id,
            COLUMN_B:
                b_value,
            COLUMN_REPAIR_BUDGET_FRACTION:
                repair_budget_fraction,
            COLUMN_REPAIR_BUDGET_TOTAL:
                repair_budget_total,
            _COLUMN_ACTION_REPAIR_BUDGET_FRACTION:
                mean([to_float(row.get(_COLUMN_ACTION_REPAIR_BUDGET_FRACTION)) for row in rows]),
            _COLUMN_ACTION_REPAIR_BUDGET_TOTAL:
                mean([to_float(row.get(_COLUMN_ACTION_REPAIR_BUDGET_TOTAL)) for row in rows]),
            _COLUMN_IS_NO_OP_ACTION:
                _first_non_null(rows, key=_COLUMN_IS_NO_OP_ACTION),
            _COLUMN_NUM_SEEDS:
                len({row.get(COLUMN_SEED) for row in rows}),
            _COLUMN_MEAN_ABSOLUTE_RECOVERY:
                mean(mean_absolute_recovery_values),
            _COLUMN_SEM_ABSOLUTE_RECOVERY:
                _sem(mean_absolute_recovery_values),
            _COLUMN_MEAN_RHO:
                mean(mean_rho_values),
            _COLUMN_SEM_RHO:
                _sem(mean_rho_values),
            _COLUMN_MEAN_HELPED_TASK_FRACTION:
                mean(helped_task_fraction_values),
            _COLUMN_MEAN_HARMED_TASK_FRACTION:
                mean(harmed_task_fraction_values),
            f'mean_{_COLUMN_WORST_TASK_HARM}':
                mean(worst_task_harm_values),
            _COLUMN_SEM_WORST_TASK_HARM:
                _sem(worst_task_harm_values),
            f'mean_{_COLUMN_UTILITY_PRIMARY}':
                mean(utility_primary_values),
            _COLUMN_SEM_UTILITY_PRIMARY:
                _sem(utility_primary_values),
            f'mean_{_COLUMN_UTILITY_CONSERVATIVE}':
                mean(utility_conservative_values),
            _COLUMN_SEM_UTILITY_CONSERVATIVE:
                _sem(utility_conservative_values),
        })

    return result_rows


def _best_controller_for_metric(
    *,
    frontier_rows: list[dict[str, Any]],
    metric_key: str,
) -> tuple[str | None, float | None]:
    """
    Select the best controller id for a metric in one repair-selection row.

    Args:
        frontier_rows: Frontier rows in one repair-selection group.
        metric_key: Metric to maximize.

    Returns:
        tuple[str | None, float | None]: Best controller id and best metric value.
    """
    best_controller_id: str | None = None
    best_value: float | None = None
    for row in sorted(frontier_rows, key=lambda item: _sort_key(item.get(_COLUMN_CONTROLLER_ID))):
        value = to_float(row.get(metric_key))
        if value is None:
            continue
        controller_id = str(row.get(_COLUMN_CONTROLLER_ID))
        if best_value is None or value > best_value:
            best_controller_id = controller_id
            best_value = value
    return best_controller_id, best_value


def _build_best_static_controller_lookup(
    *,
    frontier_rows: list[dict[str, Any]],
) -> dict[tuple[Any, ...], str]:
    """
    Build the best static controller lookup for selection-margin calculations.

    Args:
        frontier_rows: Frontier rows.

    Returns:
        dict[tuple[Any, ...], str]: Slice -> best static controller id.
    """
    slice_groups: dict[tuple[Any, ...], dict[str, list[float | None]]] = {}
    for row in frontier_rows:
        slice_key = _static_selection_slice_key(row)
        controller_id = str(row.get(_COLUMN_CONTROLLER_ID))
        slice_groups.setdefault(slice_key, {}).setdefault(controller_id,
                                                          []).append(to_float(row.get(_COLUMN_UTILITY_CONSERVATIVE)))

    best_static_lookup: dict[tuple[Any, ...], str] = {}
    for slice_key, controller_values in slice_groups.items():
        best_controller_id: str | None = None
        best_value: float | None = None
        for controller_id, values in sorted(controller_values.items()):
            mean_value = mean(values)
            if mean_value is None:
                continue
            if best_value is None or mean_value > best_value:
                best_controller_id = controller_id
                best_value = mean_value
        if best_controller_id is not None:
            best_static_lookup[slice_key] = best_controller_id

    return best_static_lookup


def _build_repair_selection_rows(*, frontier_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Build the repair-selection dataset from frontier rows.

    Args:
        frontier_rows: Frontier rows with Pareto annotations.

    Returns:
        list[dict[str, Any]]: Repair-selection rows.
    """
    best_static_lookup = _build_best_static_controller_lookup(frontier_rows=frontier_rows)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in frontier_rows:
        key = _selection_setting_key(row)
        groups.setdefault(key, []).append(row)

    selection_rows: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(_sort_key(value) for value in item[0])):
        (
            experiment_id,
            scenario,
            backbone_name,
            strategy_name,
            seed,
            b_value,
            repair_budget_fraction,
            repair_budget_total,
        ) = key
        selection_row: dict[str, Any] = {
            COLUMN_EXPERIMENT_ID: experiment_id,
            _COLUMN_SCENARIO: scenario,
            _COLUMN_BACKBONE_NAME: backbone_name,
            _COLUMN_STRATEGY_NAME: strategy_name,
            COLUMN_SEED: seed,
            COLUMN_B: b_value,
            COLUMN_REPAIR_BUDGET_FRACTION: repair_budget_fraction,
            COLUMN_REPAIR_BUDGET_TOTAL: repair_budget_total,
        }
        for column in _PRE_REPAIR_SELECTION_COLUMNS:
            if column in selection_row:
                continue
            selection_row[column] = mean([to_float(row.get(column)) for row in rows])
            if selection_row[column] is None:
                selection_row[column] = _first_non_null(rows, key=column)

        for row in sorted(rows, key=lambda item: _sort_key(item.get(_COLUMN_CONTROLLER_ID))):
            controller_id = str(row.get(_COLUMN_CONTROLLER_ID))
            for metric_key in _PAIRWISE_UTILITY_METRICS:
                selection_row[f'{metric_key}__{controller_id}'] = row.get(metric_key)

        best_primary_controller, best_primary_value = _best_controller_for_metric(
            frontier_rows=rows,
            metric_key=_COLUMN_UTILITY_PRIMARY,
        )
        best_conservative_controller, best_conservative_value = _best_controller_for_metric(
            frontier_rows=rows,
            metric_key=_COLUMN_UTILITY_CONSERVATIVE,
        )
        best_cost_aware_controller, best_cost_aware_value = _best_controller_for_metric(
            frontier_rows=rows,
            metric_key=_COLUMN_UTILITY_COST_AWARE,
        )
        selection_row['best_controller_by_utility_primary'] = best_primary_controller
        selection_row['best_controller_by_utility_conservative'] = best_conservative_controller
        selection_row['best_controller_by_utility_cost_aware'] = best_cost_aware_controller
        selection_row['best_utility_primary'] = best_primary_value
        selection_row['best_utility_conservative'] = best_conservative_value
        selection_row['best_utility_cost_aware'] = best_cost_aware_value

        static_controller_id = best_static_lookup.get(_static_selection_slice_key(selection_row))
        selection_row['best_static_controller_by_utility_conservative'] = static_controller_id
        if static_controller_id is None or best_conservative_value is None:
            selection_row[_COLUMN_ORACLE_MARGIN] = None
        else:
            static_utility = to_float(selection_row.get(f'{_COLUMN_UTILITY_CONSERVATIVE}__{static_controller_id}'))
            if static_utility is None:
                selection_row[_COLUMN_ORACLE_MARGIN] = None
            else:
                selection_row[_COLUMN_ORACLE_MARGIN] = float(best_conservative_value - static_utility)
        selection_rows.append(selection_row)

    return selection_rows


def write_repairability_frontier_outputs(
    *,
    runs_table: list[dict[str, Any]],
    experiences_table: list[dict[str, Any]],
    out_dir: str | Path,
) -> dict[str, Path]:
    """
    Write repairability-frontier outputs under the canonical analysis layout.

    Args:
        runs_table: Collected run-level rows.
        experiences_table: Collected experience-level rows.
        out_dir: Root analysis directory for a single experiment.

    Returns:
        dict[str, Path]: Written artifact paths keyed by artifact stem.
    """
    logger = get_logger()
    root_dir = Path(out_dir)
    tables_dir = root_dir / 'tables'
    frontier_dir = root_dir / 'frontier'
    tables_dir.mkdir(parents=True, exist_ok=True)
    frontier_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        'schema': {
            'name': 'regain.analysis.frontier.manifest',
            'version': 1,
        },
        'plots': {
            'saved': [],
            'skipped': [],
        },
    }

    repair_outcomes = _build_repair_outcome_candidates(
        runs_table=runs_table,
        experiences_table=experiences_table,
        manifest=manifest,
    )
    repair_outcomes = _append_no_op_candidates(
        candidates=repair_outcomes,
        manifest=manifest,
    )
    controller_id_map = _build_controller_id_map(
        controller_names={str(row.get(COLUMN_CONTROLLER_NAME)) for row in repair_outcomes},
        manifest=manifest,
    )
    for row in repair_outcomes:
        controller_name = str(row.get(COLUMN_CONTROLLER_NAME))
        if controller_name == _NO_OP_CONTROLLER_NAME:
            row[_COLUMN_CONTROLLER_ID] = _NO_OP_CONTROLLER_ID
            continue
        row[_COLUMN_CONTROLLER_ID] = controller_id_map[controller_name]

    frontier_rows = _aggregate_frontier_rows(repair_outcomes=repair_outcomes)
    _annotate_pareto_membership(frontier_rows=frontier_rows, manifest=manifest)
    repair_pareto_rows = _aggregate_repair_pareto_rows(frontier_rows=frontier_rows)
    repair_impact_rows = _aggregate_repair_impact_rows(frontier_rows=frontier_rows)
    repair_selection_rows = _build_repair_selection_rows(frontier_rows=frontier_rows)

    repair_outcomes_path = tables_dir / 'repair_outcomes.jsonl'
    repair_outcomes_path.write_text(
        '\n'.join(json.dumps(row, default=str) for row in repair_outcomes) + ('\n' if repair_outcomes else ''),
        encoding='utf-8',
    )
    candidates_path = frontier_dir / 'candidates.csv'
    pareto_path = frontier_dir / 'pareto.csv'
    impact_path = frontier_dir / 'impact.csv'
    selection_path = frontier_dir / 'selection.csv'
    manifest_path = frontier_dir / 'manifest.json'

    write_csv(candidates_path, frontier_rows)
    write_csv(pareto_path, repair_pareto_rows)
    write_csv(impact_path, repair_impact_rows)
    write_csv(selection_path, repair_selection_rows)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    logger.warning('Wrote %s', repair_outcomes_path)
    logger.warning('Wrote %s', candidates_path)
    logger.warning('Wrote %s', pareto_path)
    logger.warning('Wrote %s', impact_path)
    logger.warning('Wrote %s', selection_path)
    logger.warning('Wrote %s', manifest_path)

    return {
        'repair_outcomes': repair_outcomes_path,
        'candidates': candidates_path,
        'pareto': pareto_path,
        'impact': impact_path,
        'selection': selection_path,
        'manifest': manifest_path,
    }
