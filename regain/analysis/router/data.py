"""
Feature/label row construction and preprocessing for the router analysis.
"""

import csv
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from regain.analysis.router.actions import get_action_metric
from regain.analysis.router.actions import is_action_evaluable
from regain.analysis.router.actions import is_pareto_value
from regain.analysis.router.actions import row_oracle_action
from regain.analysis.router.constants import ROUTER_ALLOWED_CATEGORICAL_FEATURES
from regain.analysis.router.constants import ROUTER_ALLOWED_NUMERIC_FEATURES
from regain.analysis.router.constants import ROUTER_ID_COLUMNS
from regain.analysis.utils import to_float

__all__ = [
    'FeaturePreprocessor',
    'OPTIONAL_PRE_REPAIR_DRIFT_COLUMNS',
]

OPTIONAL_PRE_REPAIR_DRIFT_COLUMNS = (
    'mean_run.diagnostics.feature_drift',
    'mean_run.diagnostics.prototype_distance',
    'mean_run.diagnostics.prototype_l2',
    'mean_run.diagnostics.feature_centroid_shift',
)


def _coerce_csv_value(value: Any) -> Any:
    """
    Coerce a string-typed CSV value into a Python scalar.

    Args:
        value: CSV value (often `str`).

    Returns:
        Any: Coerced value as int, float, bool, None, or str.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    raw = value.strip()
    if not raw:
        return None
    lowered = raw.lower()
    if lowered in {'none', 'null', 'nan'}:
        return None
    if lowered == 'true':
        return True
    if lowered == 'false':
        return False
    if raw.lstrip('+-').isdigit():
        try:
            return int(raw)
        except ValueError:
            return raw
    try:
        return float(raw)
    except ValueError:
        return raw


def read_selection_rows(*, path: Path) -> list[dict[str, Any]]:
    """
    Read `frontier/selection.csv` rows with coerced scalar values.

    Args:
        path: Path to `frontier/selection.csv`.

    Returns:
        list[dict[str, Any]]: Coerced selection rows.

    Raises:
        FileNotFoundError: If the selection CSV is missing.
    """
    if not path.exists():
        raise FileNotFoundError(f'Repair-selection CSV not found at {path}.')
    with path.open('r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = [{key: _coerce_csv_value(raw) for key, raw in row.items()} for row in reader]
    return rows


def build_label_row(
    *,
    row: dict[str, Any],
    action_ids: Sequence[str],
    action_family_by_id: dict[str, str],
) -> dict[str, Any]:
    """
    Build a labels-table row containing offline outcomes and oracle labels.

    Args:
        row: Selection row.
        action_ids: Action identifiers in this dataset.
        action_family_by_id: Mapping from action id to family.

    Returns:
        dict[str, Any]: Labels-table row.
    """
    label_row: dict[str, Any] = {key: row.get(key) for key in ROUTER_ID_COLUMNS}
    evaluable_actions = [action_id for action_id in action_ids if is_action_evaluable(row=row, action_id=action_id)]
    label_row['available_actions'] = ','.join(sorted(evaluable_actions))

    primary_action, primary_value = row_oracle_action(
        row=row,
        metric='utility_primary',
        action_ids=evaluable_actions,
    )
    conservative_action, conservative_value = row_oracle_action(
        row=row,
        metric='utility_conservative',
        action_ids=evaluable_actions,
    )
    cost_action, cost_value = row_oracle_action(
        row=row,
        metric='utility_cost_aware',
        action_ids=evaluable_actions,
    )
    label_row['oracle_action_primary'] = primary_action
    label_row['oracle_action_conservative'] = conservative_action
    label_row['oracle_action_cost_aware'] = cost_action
    label_row['oracle_utility_primary'] = primary_value
    label_row['oracle_utility_conservative'] = conservative_value
    label_row['oracle_utility_cost_aware'] = cost_value

    pareto_primary_action: Optional[str] = None
    pareto_primary_value: Optional[float] = None
    for action_id in sorted(evaluable_actions):
        if is_pareto_value(row.get(f'is_pareto__{action_id}')) is not True:
            continue
        value = get_action_metric(row, metric='utility_primary', action_id=action_id)
        if value is None:
            continue
        if pareto_primary_value is None or value > pareto_primary_value:
            pareto_primary_action = action_id
            pareto_primary_value = value
    label_row['oracle_pareto_action_primary'] = pareto_primary_action

    metric_keys = (
        'utility_primary',
        'utility_conservative',
        'utility_cost_aware',
        'mean_harmed_task_fraction',
        'worst_task_harm',
        'is_pareto',
        'is_no_op_action',
        'action_repair_budget_fraction',
        'action_repair_budget_total',
    )
    for action_id in action_ids:
        for metric_key in metric_keys:
            label_row[f'{metric_key}__{action_id}'] = row.get(f'{metric_key}__{action_id}')
        label_row[f'action_family__{action_id}'] = action_family_by_id.get(action_id)
    return label_row


def build_feature_row(
    *,
    row: dict[str, Any],
    optional_drift_columns: Sequence[str],
) -> dict[str, Any]:
    """
    Build a feature-table row containing only allowlisted pre-repair fields.

    Args:
        row: Selection row.
        optional_drift_columns: Optional drift columns to retain when present.

    Returns:
        dict[str, Any]: Feature-table row.
    """
    mean_a_ref = to_float(row.get('mean_A_ref'))
    mean_a_post = to_float(row.get('mean_A_post'))
    base_final_accuracy: Optional[float] = mean_a_post
    reference_accuracy: Optional[float] = mean_a_ref
    headroom: Optional[float] = None
    if mean_a_ref is not None and mean_a_post is not None:
        headroom = max(0.0, mean_a_ref - mean_a_post)

    feature_row: dict[str, Any] = {key: row.get(key) for key in ROUTER_ID_COLUMNS}
    for column in ROUTER_ALLOWED_CATEGORICAL_FEATURES:
        if column in feature_row:
            continue
        feature_row[column] = row.get(column)
    for column in ROUTER_ALLOWED_NUMERIC_FEATURES:
        if column == 'base_final_accuracy':
            feature_row[column] = base_final_accuracy
        elif column == 'reference_accuracy':
            feature_row[column] = reference_accuracy
        elif column == 'headroom':
            feature_row[column] = headroom
        else:
            feature_row[column] = row.get(column)
    for column in optional_drift_columns:
        feature_row[column] = row.get(column)
    return feature_row


def collect_optional_drift_columns(*, rows: list[dict[str, Any]]) -> list[str]:
    """
    Determine which optional drift columns are present in the selection rows.

    Args:
        rows: Selection rows.

    Returns:
        list[str]: Optional drift columns present in any row.
    """
    available: list[str] = []
    if not rows:
        return available
    columns = set()
    for row in rows:
        columns.update(row.keys())
    for column in OPTIONAL_PRE_REPAIR_DRIFT_COLUMNS:
        if column in columns:
            available.append(column)
    return available


@dataclass
class FeaturePreprocessor:
    """
    Shared numeric/categorical preprocessor for router policies.

    Attributes:
        numeric_columns: Numeric feature column names in stable order.
        categorical_columns: Categorical feature column names in stable order.
        numeric_medians: Train-set medians for numeric columns.
        numeric_means: Train-set means for numeric columns.
        numeric_stds: Train-set standard deviations for numeric columns.
        categorical_levels: Train-set categorical levels per categorical column.
    """

    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    numeric_medians: dict[str, float] = field(default_factory=dict)
    numeric_means: dict[str, float] = field(default_factory=dict)
    numeric_stds: dict[str, float] = field(default_factory=dict)
    categorical_levels: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        rows: list[dict[str, Any]],
        *,
        numeric_columns: Sequence[str],
        categorical_columns: Sequence[str],
    ) -> 'FeaturePreprocessor':
        """
        Fit the preprocessor on training rows.

        Args:
            rows: Training rows.
            numeric_columns: Numeric feature names.
            categorical_columns: Categorical feature names.

        Returns:
            FeaturePreprocessor: Fitted preprocessor.
        """
        numeric_medians: dict[str, float] = {}
        numeric_means: dict[str, float] = {}
        numeric_stds: dict[str, float] = {}
        for column in numeric_columns:
            values = [to_float(row.get(column)) for row in rows if to_float(row.get(column)) is not None]
            finite_values = [value for value in values if value is not None]
            if finite_values:
                arr = np.asarray(finite_values, dtype=float)
                numeric_medians[column] = float(np.median(arr))
                numeric_means[column] = float(np.mean(arr))
                std_value = float(np.std(arr, ddof=0))
                numeric_stds[column] = std_value if std_value > 0.0 else 1.0
            else:
                numeric_medians[column] = 0.0
                numeric_means[column] = 0.0
                numeric_stds[column] = 1.0

        categorical_levels: dict[str, tuple[str, ...]] = {}
        for column in categorical_columns:
            seen = sorted({str(row.get(column)) if row.get(column) is not None else '<missing>' for row in rows})
            categorical_levels[column] = tuple(seen)

        return cls(
            numeric_columns=tuple(numeric_columns),
            categorical_columns=tuple(categorical_columns),
            numeric_medians=numeric_medians,
            numeric_means=numeric_means,
            numeric_stds=numeric_stds,
            categorical_levels=categorical_levels,
        )

    def feature_names(self) -> list[str]:
        """
        Build the expanded feature-name list for the transformed matrix.

        Returns:
            list[str]: Ordered expanded feature names.
        """
        names = list(self.numeric_columns)
        for column in self.categorical_columns:
            for level in self.categorical_levels.get(column, ()):
                names.append(f'{column}={level}')
            names.append(f'{column}=<other>')
        return names

    def transform(self, rows: list[dict[str, Any]], *, scale_numeric: bool) -> np.ndarray:
        """
        Transform rows into a deterministic feature matrix.

        Args:
            rows: Rows to transform.
            scale_numeric: Whether to apply train mean/std scaling.

        Returns:
            np.ndarray: Feature matrix of shape `(len(rows), expanded_dim)`.
        """
        if not rows:
            return np.zeros((0, len(self.feature_names())), dtype=float)
        n_rows = len(rows)
        numeric_dim = len(self.numeric_columns)
        n_columns = len(self.feature_names())
        matrix = np.zeros((n_rows, n_columns), dtype=float)
        for index, row in enumerate(rows):
            for col_index, column in enumerate(self.numeric_columns):
                value = to_float(row.get(column))
                if value is None:
                    value = self.numeric_medians.get(column, 0.0)
                if scale_numeric:
                    mean_value = self.numeric_means.get(column, 0.0)
                    std_value = self.numeric_stds.get(column, 1.0)
                    matrix[index, col_index] = (value - mean_value) / std_value
                else:
                    matrix[index, col_index] = value
            offset = numeric_dim
            for column in self.categorical_columns:
                raw_value = row.get(column)
                value_text = str(raw_value) if raw_value is not None else '<missing>'
                levels = self.categorical_levels.get(column, ())
                set_value = False
                for level_index, level in enumerate(levels):
                    if value_text == level:
                        matrix[index, offset + level_index] = 1.0
                        set_value = True
                        break
                offset += len(levels)
                if not set_value:
                    matrix[index, offset] = 1.0
                offset += 1
        return matrix
