"""
Action-identifier helpers for the repair router analysis package.
"""

import re
from typing import Any, Optional, Sequence

from regain.analysis.router.constants import NO_OP_CANONICAL_ID
from regain.analysis.router.constants import UTILITY_METRICS
from regain.analysis.utils import to_float

__all__ = [
    'ACTION_ALIASES',
    'FORBIDDEN_FEATURE_PATTERNS',
    'get_action_metric',
    'resolve_action_family',
    'validate_router_feature_schema',
]

ACTION_ALIASES = {
    'no_op': ('no_op', 'no-op', 'noop'),
    'weight_aligning': ('weight_aligning', 'weight_align', 'wa', 'weight_aligner'),
    'bic': ('bic', 'bias_correction'),
    'temperature_scaling': ('temperature_scaling', 'temp_scaling', 'temperature', 'ts'),
    'prototype_mean_shift': ('prototype', 'mean_shift', 'prototype_mean_shift', 'moment', 'moment_shift'),
    'linear_probe': ('linear_probe', 'linear', 'probe', 'ridge_probe', 'ridge'),
}

FORBIDDEN_FEATURE_PATTERNS = (
    '__',
    'A_ctrl',
    'mean_A_ctrl',
    'rho',
    'absolute_recovery',
    'residual_forgetting',
    'task_delta',
    'helped',
    'harmed',
    'harm_magnitude',
    'utility_',
    'oracle',
    'best_controller',
    'best_action',
    'selected_action',
    'pareto',
    'controller_id',
    'controller_name',
    'run_id',
    'source_stage',
)


def _normalize_action_token(*, action_id: str) -> str:
    """
    Normalize an action identifier into a lowercase underscore token form.

    Args:
        action_id: Raw action identifier.

    Returns:
        str: Normalized action identifier.
    """
    normalized = re.sub(r'[^a-z0-9]+', '_', str(action_id).lower()).strip('_')
    return normalized


def resolve_action_family(*, action_id: str) -> str:
    """
    Resolve an action identifier to its family name.

    Args:
        action_id: Raw action identifier.

    Returns:
        str: Family identifier; falls back to the normalized action id when no alias matches.
    """
    normalized = _normalize_action_token(action_id=action_id)
    if not normalized:
        return normalized
    tokens = normalized.split('_')
    for family, aliases in ACTION_ALIASES.items():
        sorted_aliases = sorted(aliases, key=lambda value: -len(value))
        for alias in sorted_aliases:
            alias_tokens = _normalize_action_token(action_id=alias).split('_')
            if not alias_tokens:
                continue
            if normalized == '_'.join(alias_tokens):
                return family
            for start in range(0, len(tokens) - len(alias_tokens) + 1):
                if tokens[start:start + len(alias_tokens)] == alias_tokens:
                    return family
    return normalized


def validate_router_feature_schema(*, feature_columns: list[str]) -> list[str]:
    """
    Validate the router feature column names against the forbidden-pattern list.

    Args:
        feature_columns: Candidate feature column names from `router/features.csv`.

    Returns:
        list[str]: Names that violate the forbidden-pattern rule.
    """
    invalid: list[str] = []
    for column in feature_columns:
        if column == 'backbone_name':
            continue
        for pattern in FORBIDDEN_FEATURE_PATTERNS:
            if pattern in column:
                invalid.append(column)
                break
    return invalid


def collect_action_ids(*, rows: list[dict[str, Any]]) -> list[str]:
    """
    Collect all action identifiers present in the selection rows.

    Args:
        rows: Selection rows.

    Returns:
        list[str]: Sorted unique action identifiers.
    """
    action_ids: set[str] = set()
    metric_prefixes = tuple(f'{metric}__' for metric in UTILITY_METRICS)
    for row in rows:
        for key in row.keys():
            for prefix in metric_prefixes:
                if key.startswith(prefix):
                    action_ids.add(key[len(prefix):])
    return sorted(action_ids)


def action_family_map(*, action_ids: Sequence[str]) -> dict[str, str]:
    """
    Build a stable mapping from action identifier to family identifier.

    Args:
        action_ids: Action identifiers.

    Returns:
        dict[str, str]: Mapping from action id to family.
    """
    return {action_id: resolve_action_family(action_id=action_id) for action_id in action_ids}


def resolve_no_op_action_id(*, action_family_by_id: dict[str, str]) -> Optional[str]:
    """
    Resolve the canonical no-op action identifier when available.

    Args:
        action_family_by_id: Mapping of action ids to families.

    Returns:
        Optional[str]: The first action id mapped to the `no_op` family, or None.
    """
    no_op_ids = sorted(action_id for action_id, family in action_family_by_id.items() if family == NO_OP_CANONICAL_ID)
    if not no_op_ids:
        return None
    if NO_OP_CANONICAL_ID in no_op_ids:
        return NO_OP_CANONICAL_ID
    return no_op_ids[0]


def get_action_metric(row: dict[str, Any], *, metric: str, action_id: str) -> Optional[float]:
    """
    Look up a per-action metric value on a row.

    Args:
        row: Source row.
        metric: Base metric key (e.g., `utility_conservative`).
        action_id: Action identifier suffix.

    Returns:
        Optional[float]: Numeric value, or None when absent or non-finite.
    """
    return to_float(row.get(f'{metric}__{action_id}'))


def is_action_evaluable(*, row: dict[str, Any], action_id: str) -> bool:
    """
    Determine whether an action's utility metrics are all numeric for a row.

    Args:
        row: Selection row.
        action_id: Action identifier.

    Returns:
        bool: True when all three utility columns are numeric.
    """
    for metric in UTILITY_METRICS:
        if get_action_metric(row, metric=metric, action_id=action_id) is None:
            return False
    return True


def is_pareto_value(value: Any) -> Optional[bool]:
    """
    Normalize a Pareto-membership value.

    Args:
        value: Raw value from selection columns.

    Returns:
        Optional[bool]: Boolean representation, or None when absent.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if not text or text in {'none', 'null', 'nan'}:
        return None
    if text in {'true', '1'}:
        return True
    if text in {'false', '0'}:
        return False
    return None


def row_oracle_action(
    *,
    row: dict[str, Any],
    metric: str,
    action_ids: Sequence[str],
) -> tuple[Optional[str], Optional[float]]:
    """
    Choose the action id with the largest finite metric value for a row.

    Args:
        row: Selection row.
        metric: Base metric key.
        action_ids: Candidate action identifiers.

    Returns:
        tuple[Optional[str], Optional[float]]: Best action id and value, or (None, None).
    """
    best_action: Optional[str] = None
    best_value: Optional[float] = None
    for action_id in sorted(action_ids):
        value = get_action_metric(row, metric=metric, action_id=action_id)
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_action = action_id
            best_value = value
    return best_action, best_value
