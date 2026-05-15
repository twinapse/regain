"""
Router policy classes and training helpers.
"""

import math
from typing import Any, Optional, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from regain.analysis.router.actions import action_family_map
from regain.analysis.router.actions import get_action_metric
from regain.analysis.router.actions import is_action_evaluable
from regain.analysis.router.actions import resolve_no_op_action_id
from regain.analysis.router.constants import NO_OP_CANONICAL_ID
from regain.analysis.router.constants import ROUTER_ALLOWED_CATEGORICAL_FEATURES
from regain.analysis.router.constants import ROUTER_ALLOWED_NUMERIC_FEATURES
from regain.analysis.router.data import FeaturePreprocessor
from regain.analysis.utils import to_float

__all__ = [
    'RouterPolicy',
]

LOW_COST_ACTION_FAMILIES = (
    'no_op',
    'weight_aligning',
    'bic',
    'temperature_scaling',
    'prototype_mean_shift',
)

THRESHOLD_FEATURES = (
    'mean_forgetting',
    'headroom',
    'base_final_accuracy',
    'mean_run.diagnostics.out_of_task_rate',
    'mean_run.calibration.nll',
    'mean_run.calibration.ece',
    'mean_run.calibration.aece',
)

THRESHOLD_HIGH_ACTION_FAMILIES = (
    'bic',
    'temperature_scaling',
    'prototype_mean_shift',
    'linear_probe',
)

THRESHOLD_LOW_ACTION_FAMILIES = (
    'no_op',
    'temperature_scaling',
)

_FAMILY_PRIORITY_ORDER = (
    'no_op',
    'temperature_scaling',
    'bic',
    'weight_aligning',
    'prototype_mean_shift',
    'linear_probe',
)


def _action_mean_utility(
    *,
    label_rows: list[dict[str, Any]],
    indices: Sequence[int],
    action_id: str,
    metric: str,
) -> Optional[float]:
    """
    Compute the mean of a per-action metric over the given indices.

    Args:
        label_rows: Labels-table rows.
        indices: Row indices to consider.
        action_id: Action identifier.
        metric: Base metric key.

    Returns:
        Optional[float]: Mean value, or None when no finite samples exist.
    """
    values: list[float] = []
    for index in indices:
        value = get_action_metric(label_rows[index], metric=metric, action_id=action_id)
        if value is not None:
            values.append(value)
    if not values:
        return None
    return float(np.mean(values))


def _family_priority(family: str) -> int:
    """
    Return the family priority order index used in tie breaks.

    Args:
        family: Action family identifier.

    Returns:
        int: Lower values indicate higher priority.
    """
    if family in _FAMILY_PRIORITY_ORDER:
        return _FAMILY_PRIORITY_ORDER.index(family)
    return len(_FAMILY_PRIORITY_ORDER)


def _pick_best_action_for_family(
    *,
    family: str,
    action_ids: Sequence[str],
    action_family_by_id: dict[str, str],
    label_rows: list[dict[str, Any]],
    indices: Sequence[int],
    metric: str,
    fallback_indices: Optional[Sequence[int]] = None,
) -> Optional[str]:
    """
    Select the best train-time action id for a family.

    Args:
        family: Target family.
        action_ids: Action ids in the dataset.
        action_family_by_id: Mapping from action id to family.
        label_rows: Labels-table rows.
        indices: Training indices.
        metric: Metric to maximize.
        fallback_indices: Optional broader index pool used if the primary scope has no action ids.

    Returns:
        Optional[str]: Selected action id, or None if the family is absent.
    """
    family_action_ids = [action_id for action_id in sorted(action_ids) if action_family_by_id.get(action_id) == family]
    if not family_action_ids:
        return None
    best_action: Optional[str] = None
    best_value: Optional[float] = None
    for action_id in family_action_ids:
        value = _action_mean_utility(
            label_rows=label_rows,
            indices=indices,
            action_id=action_id,
            metric=metric,
        )
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_action = action_id
            best_value = value
    if best_action is not None:
        return best_action
    if fallback_indices is None:
        return family_action_ids[0]
    for action_id in family_action_ids:
        value = _action_mean_utility(
            label_rows=label_rows,
            indices=fallback_indices,
            action_id=action_id,
            metric=metric,
        )
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_action = action_id
            best_value = value
    return best_action or family_action_ids[0]


def _percentile(values: list[float], *, percentile: float) -> Optional[float]:
    """
    Compute a percentile over finite values.

    Args:
        values: Finite numeric values.
        percentile: Percentile in [0, 100].

    Returns:
        Optional[float]: Percentile value, or None when empty.
    """
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def _finite_values(*, rows: list[dict[str, Any]], indices: Sequence[int], key: str) -> list[float]:
    """
    Extract finite values for a key over a row subset.

    Args:
        rows: Source rows.
        indices: Row indices.
        key: Column name.

    Returns:
        list[float]: Finite numeric values.
    """
    values: list[float] = []
    for index in indices:
        value = to_float(rows[index].get(key))
        if value is not None:
            values.append(value)
    return values


def _selection_outcome_score(
    *,
    label_rows: list[dict[str, Any]],
    indices: Sequence[int],
    selections: dict[int, Optional[str]],
) -> float:
    """
    Score a routed selection on the training fold.

    Args:
        label_rows: Labels-table rows.
        indices: Training indices.
        selections: Mapping from row index to selected action id.

    Returns:
        float: Penalized mean conservative utility score.
    """
    utilities: list[float] = []
    worst_harms: list[float] = []
    harmed_fractions: list[float] = []
    for index in indices:
        action_id = selections.get(index)
        if action_id is None:
            continue
        utility = get_action_metric(label_rows[index], metric='utility_conservative', action_id=action_id)
        worst = get_action_metric(label_rows[index], metric='worst_task_harm', action_id=action_id)
        fraction = get_action_metric(label_rows[index], metric='mean_harmed_task_fraction', action_id=action_id)
        if utility is None:
            continue
        utilities.append(utility)
        if worst is not None:
            worst_harms.append(worst)
        if fraction is not None:
            harmed_fractions.append(fraction)
    if not utilities:
        return -math.inf
    score = float(np.mean(utilities))
    if worst_harms:
        score -= 0.25 * float(np.mean(worst_harms))
    if harmed_fractions:
        score -= 0.10 * float(np.mean(harmed_fractions))
    return score


def _resolve_family_action(
    *,
    family: str,
    action_ids: Sequence[str],
    action_family_by_id: dict[str, str],
    label_rows: list[dict[str, Any]],
    train_indices: Sequence[int],
) -> Optional[str]:
    """
    Helper to resolve a family to an action id at fit time.

    Args:
        family: Target family.
        action_ids: Action identifiers.
        action_family_by_id: Mapping from action id to family.
        label_rows: Labels-table rows.
        train_indices: Training indices.

    Returns:
        Optional[str]: Selected action id, or None when absent.
    """
    return _pick_best_action_for_family(
        family=family,
        action_ids=action_ids,
        action_family_by_id=action_family_by_id,
        label_rows=label_rows,
        indices=train_indices,
        metric='utility_conservative',
        fallback_indices=list(range(len(label_rows))),
    )


def _fit_per_action_utility_regressors(
    *,
    preprocessor: FeaturePreprocessor,
    feature_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    train_indices: Sequence[int],
    action_ids: Sequence[str],
    random_state: int,
) -> dict[str, tuple[Any, float]]:
    """
    Fit per-action expected-utility regressors with constant fallbacks.

    Args:
        preprocessor: Fitted preprocessor.
        feature_rows: Feature-table rows.
        label_rows: Labels-table rows.
        train_indices: Training row indices.
        action_ids: Action identifiers.
        random_state: Deterministic seed.

    Returns:
        dict[str, tuple[Any, float]]: Mapping from action id to (model, constant fallback).
    """
    train_feature_rows = [feature_rows[index] for index in train_indices]
    matrix = preprocessor.transform(train_feature_rows, scale_numeric=True)
    regressors: dict[str, tuple[Any, float]] = {}
    for action_id in action_ids:
        indices_with_outcome: list[int] = []
        targets: list[float] = []
        for offset, train_index in enumerate(train_indices):
            value = get_action_metric(
                label_rows[train_index],
                metric='utility_conservative',
                action_id=action_id,
            )
            if value is None:
                continue
            indices_with_outcome.append(offset)
            targets.append(float(value))
        if not targets:
            regressors[action_id] = (None, 0.0)
            continue
        constant_value = float(np.mean(targets))
        if len(targets) < 5:
            regressors[action_id] = (None, constant_value)
            continue
        subset = matrix[indices_with_outcome]
        model = HistGradientBoostingRegressor(
            random_state=random_state,
            max_iter=100,
            max_leaf_nodes=15,
        )
        model.fit(subset, np.asarray(targets, dtype=float))
        regressors[action_id] = (model, constant_value)
    return regressors


def _predict_per_action_utility(
    *,
    preprocessor: FeaturePreprocessor,
    feature_rows: list[dict[str, Any]],
    test_indices: Sequence[int],
    regressors: dict[str, tuple[Any, float]],
) -> dict[int, dict[str, float]]:
    """
    Predict per-action utility for each test row.

    Args:
        preprocessor: Fitted preprocessor.
        feature_rows: Feature-table rows.
        test_indices: Test row indices.
        regressors: Per-action `(model, constant)` pairs.

    Returns:
        dict[int, dict[str, float]]: Per-index per-action predicted utility.
    """
    if not test_indices:
        return {}
    test_feature_rows = [feature_rows[index] for index in test_indices]
    matrix = preprocessor.transform(test_feature_rows, scale_numeric=True)
    predictions: dict[int, dict[str, float]] = {index: {} for index in test_indices}
    for action_id, (model, constant_value) in regressors.items():
        if model is None:
            values = np.full(len(test_indices), constant_value, dtype=float)
        else:
            values = model.predict(matrix)
        for offset, index in enumerate(test_indices):
            predictions[index][action_id] = float(values[offset])
    return predictions


class RouterPolicy:
    """
    Base class for router policies operating on pre-repair features only.

    Attributes:
        name: Policy identifier.
        decision_time_controller_fits: Count of candidate controllers fitted at decision time.
    """

    name: str = 'router_policy'
    decision_time_controller_fits: int = 0

    def fit(
        self,
        *,
        feature_rows: list[dict[str, Any]],
        label_rows: list[dict[str, Any]],
        train_indices: list[int],
        action_ids: list[str],
        random_state: int,
    ) -> None:
        """
        Fit the policy on the training rows.

        Args:
            feature_rows: Feature-table rows.
            label_rows: Labels-table rows.
            train_indices: Training row indices.
            action_ids: Available action identifiers.
            random_state: Deterministic seed.
        """
        raise NotImplementedError

    def predict(
        self,
        *,
        feature_rows: list[dict[str, Any]],
        test_indices: list[int],
    ) -> dict[int, Optional[str]]:
        """
        Predict the selected action id for each test row.

        Args:
            feature_rows: Feature-table rows.
            test_indices: Test row indices.

        Returns:
            dict[int, Optional[str]]: Mapping from row index to action id (None if unavailable).
        """
        raise NotImplementedError


class _AlwaysAction(RouterPolicy):
    """
    Policy that always selects the train-best action in a fixed family.
    """

    def __init__(
        self,
        *,
        name: str,
        family: str,
    ) -> None:
        self.name = name
        self.family = family
        self._selected_action: Optional[str] = None

    def fit(
        self,
        *,
        feature_rows: list[dict[str, Any]],
        label_rows: list[dict[str, Any]],
        train_indices: list[int],
        action_ids: list[str],
        random_state: int,
    ) -> None:
        del feature_rows, random_state
        action_family_by_id = action_family_map(action_ids=action_ids)
        self._selected_action = _pick_best_action_for_family(
            family=self.family,
            action_ids=action_ids,
            action_family_by_id=action_family_by_id,
            label_rows=label_rows,
            indices=train_indices,
            metric='utility_conservative',
            fallback_indices=list(range(len(label_rows))),
        )

    def predict(
        self,
        *,
        feature_rows: list[dict[str, Any]],
        test_indices: list[int],
    ) -> dict[int, Optional[str]]:
        del feature_rows
        return {index: self._selected_action for index in test_indices}


class _BestStaticAction(RouterPolicy):
    """
    Policy that selects a single action id chosen on training data.
    """

    def __init__(
        self,
        *,
        name: str,
        family_pool: Optional[Sequence[str]],
    ) -> None:
        self.name = name
        self.family_pool = tuple(family_pool) if family_pool is not None else None
        self._selected_action: Optional[str] = None

    def fit(
        self,
        *,
        feature_rows: list[dict[str, Any]],
        label_rows: list[dict[str, Any]],
        train_indices: list[int],
        action_ids: list[str],
        random_state: int,
    ) -> None:
        del feature_rows, random_state
        action_family_by_id = action_family_map(action_ids=action_ids)
        if self.family_pool is None:
            candidate_action_ids = list(action_ids)
        else:
            candidate_action_ids = [
                action_id for action_id in action_ids if action_family_by_id.get(action_id) in self.family_pool
            ]
        best_action: Optional[str] = None
        best_score: Optional[tuple[float, float, float, int, str]] = None
        for action_id in sorted(candidate_action_ids):
            utility = _action_mean_utility(
                label_rows=label_rows,
                indices=train_indices,
                action_id=action_id,
                metric='utility_conservative',
            )
            if utility is None:
                continue
            worst_task_harm = _action_mean_utility(
                label_rows=label_rows,
                indices=train_indices,
                action_id=action_id,
                metric='worst_task_harm',
            ) or 0.0
            budget_total = _action_mean_utility(
                label_rows=label_rows,
                indices=train_indices,
                action_id=action_id,
                metric='action_repair_budget_total',
            ) or 0.0
            family = action_family_by_id.get(action_id, '')
            family_priority = _family_priority(family)
            score = (-utility, worst_task_harm, budget_total, family_priority, action_id)
            if best_score is None or score < best_score:
                best_score = score
                best_action = action_id
        self._selected_action = best_action

    def predict(
        self,
        *,
        feature_rows: list[dict[str, Any]],
        test_indices: list[int],
    ) -> dict[int, Optional[str]]:
        del feature_rows
        return {index: self._selected_action for index in test_indices}


class _ThresholdSingleFeatureConservative(RouterPolicy):
    """
    Train-fitted single-feature threshold policy.
    """

    name = 'threshold_single_feature_conservative'

    def __init__(self) -> None:
        self._feature: Optional[str] = None
        self._threshold: Optional[float] = None
        self._low_action: Optional[str] = None
        self._high_action: Optional[str] = None

    def fit(
        self,
        *,
        feature_rows: list[dict[str, Any]],
        label_rows: list[dict[str, Any]],
        train_indices: list[int],
        action_ids: list[str],
        random_state: int,
    ) -> None:
        del random_state
        action_family_by_id = action_family_map(action_ids=action_ids)
        best_score = -math.inf
        best_rule: Optional[tuple[str, float, str, str]] = None
        for feature in THRESHOLD_FEATURES:
            values = _finite_values(rows=feature_rows, indices=train_indices, key=feature)
            if not values:
                continue
            candidate_thresholds = sorted({
                _percentile(values, percentile=percentile)
                for percentile in (10, 20, 30, 40, 50, 60, 70, 80, 90)
                if _percentile(values, percentile=percentile) is not None
            })
            for threshold in candidate_thresholds:
                for low_family in THRESHOLD_LOW_ACTION_FAMILIES:
                    low_action = _pick_best_action_for_family(
                        family=low_family,
                        action_ids=action_ids,
                        action_family_by_id=action_family_by_id,
                        label_rows=label_rows,
                        indices=train_indices,
                        metric='utility_conservative',
                        fallback_indices=list(range(len(label_rows))),
                    )
                    if low_action is None:
                        continue
                    for high_family in THRESHOLD_HIGH_ACTION_FAMILIES:
                        high_action = _pick_best_action_for_family(
                            family=high_family,
                            action_ids=action_ids,
                            action_family_by_id=action_family_by_id,
                            label_rows=label_rows,
                            indices=train_indices,
                            metric='utility_conservative',
                            fallback_indices=list(range(len(label_rows))),
                        )
                        if high_action is None:
                            continue
                        selections: dict[int, Optional[str]] = {}
                        for index in train_indices:
                            value = to_float(feature_rows[index].get(feature))
                            if value is None or value <= threshold:
                                selections[index] = low_action
                            else:
                                selections[index] = high_action
                        score = _selection_outcome_score(
                            label_rows=label_rows,
                            indices=train_indices,
                            selections=selections,
                        )
                        if score > best_score:
                            best_score = score
                            best_rule = (feature, threshold, low_action, high_action)
        if best_rule is not None:
            self._feature, self._threshold, self._low_action, self._high_action = best_rule

    def predict(
        self,
        *,
        feature_rows: list[dict[str, Any]],
        test_indices: list[int],
    ) -> dict[int, Optional[str]]:
        if self._feature is None or self._threshold is None:
            return {index: None for index in test_indices}
        selections: dict[int, Optional[str]] = {}
        for index in test_indices:
            value = to_float(feature_rows[index].get(self._feature))
            if value is None or value <= self._threshold:
                selections[index] = self._low_action
            else:
                selections[index] = self._high_action
        return selections


class _ThresholdDiagnosticCascade(RouterPolicy):
    """
    Fixed diagnostic cascade threshold policy.
    """

    name = 'threshold_diagnostic_cascade'

    def __init__(self) -> None:
        self._thresholds: dict[str, Optional[float]] = {}
        self._actions: dict[str, Optional[str]] = {}

    def fit(
        self,
        *,
        feature_rows: list[dict[str, Any]],
        label_rows: list[dict[str, Any]],
        train_indices: list[int],
        action_ids: list[str],
        random_state: int,
    ) -> None:
        del random_state
        action_family_by_id = action_family_map(action_ids=action_ids)
        headroom_values = _finite_values(rows=feature_rows, indices=train_indices, key='headroom')
        forgetting_values = _finite_values(rows=feature_rows, indices=train_indices, key='mean_forgetting')
        out_of_task_values = _finite_values(
            rows=feature_rows,
            indices=train_indices,
            key='mean_run.diagnostics.out_of_task_rate',
        )
        nll_values = _finite_values(
            rows=feature_rows,
            indices=train_indices,
            key='mean_run.calibration.nll',
        )
        ece_values: list[float] = []
        for index in train_indices:
            ece = to_float(feature_rows[index].get('mean_run.calibration.ece'))
            aece = to_float(feature_rows[index].get('mean_run.calibration.aece'))
            candidates = [value for value in (ece, aece) if value is not None]
            if candidates:
                ece_values.append(max(candidates))
        base_accuracy_values = _finite_values(
            rows=feature_rows,
            indices=train_indices,
            key='base_final_accuracy',
        )
        self._thresholds = {
            'low_headroom_threshold': _percentile(headroom_values, percentile=25),
            'low_forgetting_threshold': _percentile(forgetting_values, percentile=25),
            'high_out_of_task_threshold': _percentile(out_of_task_values, percentile=75),
            'high_nll_threshold': _percentile(nll_values, percentile=75),
            'high_ece_threshold': _percentile(ece_values, percentile=75),
            'high_headroom_threshold': _percentile(headroom_values, percentile=75),
            'high_base_accuracy_threshold': _percentile(base_accuracy_values, percentile=75),
        }
        self._actions = {
            'no_op':
                _resolve_family_action(
                    family='no_op',
                    action_ids=action_ids,
                    action_family_by_id=action_family_by_id,
                    label_rows=label_rows,
                    train_indices=train_indices,
                ),
            'bic':
                _resolve_family_action(
                    family='bic',
                    action_ids=action_ids,
                    action_family_by_id=action_family_by_id,
                    label_rows=label_rows,
                    train_indices=train_indices,
                ),
            'temperature_scaling':
                _resolve_family_action(
                    family='temperature_scaling',
                    action_ids=action_ids,
                    action_family_by_id=action_family_by_id,
                    label_rows=label_rows,
                    train_indices=train_indices,
                ),
            'prototype_mean_shift':
                _resolve_family_action(
                    family='prototype_mean_shift',
                    action_ids=action_ids,
                    action_family_by_id=action_family_by_id,
                    label_rows=label_rows,
                    train_indices=train_indices,
                ),
        }

    def predict(
        self,
        *,
        feature_rows: list[dict[str, Any]],
        test_indices: list[int],
    ) -> dict[int, Optional[str]]:
        selections: dict[int, Optional[str]] = {}
        low_headroom = self._thresholds.get('low_headroom_threshold')
        low_forgetting = self._thresholds.get('low_forgetting_threshold')
        high_out_of_task = self._thresholds.get('high_out_of_task_threshold')
        high_nll = self._thresholds.get('high_nll_threshold')
        high_ece = self._thresholds.get('high_ece_threshold')
        high_headroom = self._thresholds.get('high_headroom_threshold')
        high_base_accuracy = self._thresholds.get('high_base_accuracy_threshold')
        no_op_action = self._actions.get('no_op')
        for index in test_indices:
            row = feature_rows[index]
            headroom = to_float(row.get('headroom'))
            forgetting = to_float(row.get('mean_forgetting'))
            base_accuracy = to_float(row.get('base_final_accuracy'))
            out_of_task = to_float(row.get('mean_run.diagnostics.out_of_task_rate'))
            ece = to_float(row.get('mean_run.calibration.ece'))
            aece = to_float(row.get('mean_run.calibration.aece'))
            nll = to_float(row.get('mean_run.calibration.nll'))
            ece_candidates = [value for value in (ece, aece) if value is not None]
            ece_value = max(ece_candidates) if ece_candidates else None

            if low_headroom is not None and headroom is not None and headroom <= low_headroom:
                selections[index] = no_op_action
                continue
            if (high_base_accuracy is not None and base_accuracy is not None and base_accuracy >= high_base_accuracy and
                    low_forgetting is not None and forgetting is not None and forgetting <= low_forgetting):
                selections[index] = no_op_action
                continue
            if (high_out_of_task is not None and out_of_task is not None and out_of_task > high_out_of_task):
                selections[index] = self._actions.get('bic') or no_op_action
                continue
            if ((high_nll is not None and nll is not None and nll > high_nll) or
                (high_ece is not None and ece_value is not None and ece_value > high_ece)):
                selections[index] = self._actions.get('temperature_scaling') or no_op_action
                continue
            if high_headroom is not None and headroom is not None and headroom > high_headroom:
                selections[index] = self._actions.get('prototype_mean_shift') or no_op_action
                continue
            selections[index] = no_op_action
        return selections


class _TwoStageExpectedUtilityRouter(RouterPolicy):
    """
    Two-stage safety-gated expected-utility router.
    """

    name = 'two_stage_expected_utility_router'

    def __init__(self) -> None:
        self._preprocessor: Optional[FeaturePreprocessor] = None
        self._gate_model: Any = None
        self._gate_constant: Optional[bool] = None
        self._regressors: dict[str, tuple[Any, float]] = {}
        self._no_op_action: Optional[str] = None
        self._action_ids: list[str] = []
        self._gate_threshold: float = 0.5
        self._action_family_by_id: dict[str, str] = {}

    def fit(
        self,
        *,
        feature_rows: list[dict[str, Any]],
        label_rows: list[dict[str, Any]],
        train_indices: list[int],
        action_ids: list[str],
        random_state: int,
    ) -> None:
        self._action_ids = list(action_ids)
        self._action_family_by_id = action_family_map(action_ids=action_ids)
        self._no_op_action = resolve_no_op_action_id(action_family_by_id=self._action_family_by_id)
        self._preprocessor = FeaturePreprocessor.fit(
            [feature_rows[index] for index in train_indices],
            numeric_columns=ROUTER_ALLOWED_NUMERIC_FEATURES,
            categorical_columns=ROUTER_ALLOWED_CATEGORICAL_FEATURES,
        )
        active_action_ids = [
            action_id for action_id in self._action_ids
            if self._action_family_by_id.get(action_id) != NO_OP_CANONICAL_ID
        ]
        gate_labels: list[int] = []
        gate_indices: list[int] = []
        for train_index in train_indices:
            row = label_rows[train_index]
            no_op_utility = get_action_metric(
                row,
                metric='utility_conservative',
                action_id=self._no_op_action or '',
            ) if self._no_op_action is not None else 0.0
            no_op_worst = get_action_metric(
                row,
                metric='worst_task_harm',
                action_id=self._no_op_action or '',
            ) if self._no_op_action is not None else 0.0
            evaluable_actions = [
                action_id for action_id in active_action_ids if is_action_evaluable(row=row, action_id=action_id)
            ]
            if not evaluable_actions:
                continue
            best_non_noop_utility = max(
                (get_action_metric(row, metric='utility_conservative', action_id=action_id)
                 for action_id in evaluable_actions),
                default=None,
            )
            best_non_noop_harm = min(
                (get_action_metric(row, metric='worst_task_harm', action_id=action_id)
                 for action_id in evaluable_actions),
                default=None,
            )
            if best_non_noop_utility is None or best_non_noop_harm is None:
                continue
            no_op_utility_value = no_op_utility if no_op_utility is not None else 0.0
            no_op_harm_value = no_op_worst if no_op_worst is not None else 0.0
            active_safe = (best_non_noop_utility > no_op_utility_value and
                           best_non_noop_harm <= max(0.01, no_op_harm_value + 0.01))
            gate_labels.append(1 if active_safe else 0)
            gate_indices.append(train_index)

        if not gate_labels:
            self._gate_constant = False
        else:
            unique_labels = set(gate_labels)
            if len(unique_labels) <= 1:
                self._gate_constant = bool(next(iter(unique_labels)))
            else:
                gate_feature_matrix = self._preprocessor.transform(
                    [feature_rows[index] for index in gate_indices],
                    scale_numeric=True,
                )
                gate_model = LogisticRegression(
                    random_state=random_state,
                    max_iter=1000,
                    class_weight='balanced',
                )
                gate_model.fit(gate_feature_matrix, np.asarray(gate_labels, dtype=int))
                self._gate_model = gate_model

        self._regressors = _fit_per_action_utility_regressors(
            preprocessor=self._preprocessor,
            feature_rows=feature_rows,
            label_rows=label_rows,
            train_indices=train_indices,
            action_ids=self._action_ids,
            random_state=random_state,
        )

        if self._gate_model is not None:
            train_gate_matrix = self._preprocessor.transform(
                [feature_rows[index] for index in train_indices],
                scale_numeric=True,
            )
            try:
                probabilities = self._gate_model.predict_proba(train_gate_matrix)
                classes = list(self._gate_model.classes_)
                positive_index = classes.index(1) if 1 in classes else 0
                positive_probs = probabilities[:, positive_index]
            except Exception:  # pylint: disable=broad-exception-caught
                positive_probs = np.zeros(len(train_indices), dtype=float)
            best_threshold = 0.5
            best_score = -math.inf
            train_predictions = _predict_per_action_utility(
                preprocessor=self._preprocessor,
                feature_rows=feature_rows,
                test_indices=list(train_indices),
                regressors=self._regressors,
            )
            for threshold in (0.30, 0.40, 0.50, 0.60, 0.70):
                selections: dict[int, Optional[str]] = {}
                for offset, train_index in enumerate(train_indices):
                    if positive_probs[offset] < threshold:
                        selections[train_index] = self._no_op_action
                        continue
                    action_utilities = train_predictions.get(train_index, {})
                    best_action = self._choose_best_active_action(
                        utilities=action_utilities,
                        active_action_ids=active_action_ids,
                    )
                    selections[train_index] = best_action or self._no_op_action
                score = _selection_outcome_score(
                    label_rows=label_rows,
                    indices=train_indices,
                    selections=selections,
                )
                if score > best_score or (math.isclose(score, best_score, abs_tol=1e-12) and
                                          threshold > best_threshold):
                    best_score = score
                    best_threshold = threshold
            self._gate_threshold = best_threshold

    def _choose_best_active_action(
        self,
        *,
        utilities: dict[str, float],
        active_action_ids: Sequence[str],
    ) -> Optional[str]:
        """
        Select the highest-utility active action.

        Args:
            utilities: Per-action predicted utilities.
            active_action_ids: Candidate non-noop action ids.

        Returns:
            Optional[str]: Selected action id.
        """
        best_action: Optional[str] = None
        best_score: Optional[tuple[float, int, str]] = None
        for action_id in sorted(active_action_ids):
            if action_id not in utilities:
                continue
            family = self._action_family_by_id.get(action_id, '')
            score = (-utilities[action_id], _family_priority(family), action_id)
            if best_score is None or score < best_score:
                best_score = score
                best_action = action_id
        return best_action

    def predict(
        self,
        *,
        feature_rows: list[dict[str, Any]],
        test_indices: list[int],
    ) -> dict[int, Optional[str]]:
        if self._preprocessor is None:
            return {index: self._no_op_action for index in test_indices}
        active_action_ids = [
            action_id for action_id in self._action_ids
            if self._action_family_by_id.get(action_id) != NO_OP_CANONICAL_ID
        ]
        test_predictions = _predict_per_action_utility(
            preprocessor=self._preprocessor,
            feature_rows=feature_rows,
            test_indices=test_indices,
            regressors=self._regressors,
        )
        selections: dict[int, Optional[str]] = {}
        gate_matrix: Optional[np.ndarray] = None
        if self._gate_model is not None:
            gate_matrix = self._preprocessor.transform(
                [feature_rows[index] for index in test_indices],
                scale_numeric=True,
            )
        for offset, index in enumerate(test_indices):
            if self._gate_constant is False:
                selections[index] = self._no_op_action
                continue
            if self._gate_constant is True:
                gate_pass = True
            elif gate_matrix is not None:
                try:
                    probabilities = self._gate_model.predict_proba(gate_matrix[offset:offset + 1])
                    classes = list(self._gate_model.classes_)
                    positive_index = classes.index(1) if 1 in classes else 0
                    gate_pass = float(probabilities[0, positive_index]) >= self._gate_threshold
                except Exception:  # pylint: disable=broad-exception-caught
                    gate_pass = False
            else:
                gate_pass = False
            if not gate_pass:
                selections[index] = self._no_op_action
                continue
            utilities = test_predictions.get(index, {})
            chosen = self._choose_best_active_action(
                utilities=utilities,
                active_action_ids=active_action_ids,
            )
            if chosen is None:
                selections[index] = self._no_op_action
                continue
            chosen_utility = utilities.get(chosen)
            noop_utility = utilities.get(self._no_op_action) if self._no_op_action else None
            if chosen_utility is None or (noop_utility is not None and chosen_utility < noop_utility):
                selections[index] = self._no_op_action
            else:
                selections[index] = chosen
        return selections


class _DecisionStumpConservative(RouterPolicy):
    """
    Decision-stump conservative oracle classifier.
    """

    name = 'decision_stump_conservative'

    def __init__(self, *, max_depth: int = 1) -> None:
        self._max_depth = max_depth
        self._preprocessor: Optional[FeaturePreprocessor] = None
        self._model: Any = None
        self._constant_label: Optional[str] = None

    def fit(
        self,
        *,
        feature_rows: list[dict[str, Any]],
        label_rows: list[dict[str, Any]],
        train_indices: list[int],
        action_ids: list[str],
        random_state: int,
    ) -> None:
        del action_ids
        self._preprocessor = FeaturePreprocessor.fit(
            [feature_rows[index] for index in train_indices],
            numeric_columns=ROUTER_ALLOWED_NUMERIC_FEATURES,
            categorical_columns=ROUTER_ALLOWED_CATEGORICAL_FEATURES,
        )
        targets: list[str] = []
        weights: list[float] = []
        kept_indices: list[int] = []
        for train_index in train_indices:
            label = label_rows[train_index].get('oracle_action_conservative')
            if label is None:
                continue
            no_op_utility = get_action_metric(
                label_rows[train_index],
                metric='utility_conservative',
                action_id=NO_OP_CANONICAL_ID,
            ) or 0.0
            oracle_utility = to_float(label_rows[train_index].get('oracle_utility_conservative')) or 0.0
            weight = abs(oracle_utility - no_op_utility) + 1e-3
            targets.append(str(label))
            weights.append(weight)
            kept_indices.append(train_index)
        if not targets:
            self._constant_label = None
            return
        unique_labels = sorted(set(targets))
        if len(unique_labels) <= 1:
            self._constant_label = unique_labels[0] if unique_labels else None
            return
        matrix = self._preprocessor.transform(
            [feature_rows[index] for index in kept_indices],
            scale_numeric=True,
        )
        self._model = DecisionTreeClassifier(
            random_state=random_state,
            max_depth=self._max_depth,
        )
        self._model.fit(matrix, np.asarray(targets), sample_weight=np.asarray(weights, dtype=float))

    def predict(
        self,
        *,
        feature_rows: list[dict[str, Any]],
        test_indices: list[int],
    ) -> dict[int, Optional[str]]:
        if self._model is None:
            return {index: self._constant_label for index in test_indices}
        matrix = self._preprocessor.transform(
            [feature_rows[index] for index in test_indices],
            scale_numeric=True,
        )
        predictions = self._model.predict(matrix)
        return {index: str(predictions[offset]) for offset, index in enumerate(test_indices)}


class _ShallowTreeConservative(_DecisionStumpConservative):
    """
    Shallow decision-tree conservative oracle classifier.
    """

    name = 'shallow_tree_conservative'

    def __init__(self) -> None:
        super().__init__(max_depth=3)

    def fit(
        self,
        *,
        feature_rows: list[dict[str, Any]],
        label_rows: list[dict[str, Any]],
        train_indices: list[int],
        action_ids: list[str],
        random_state: int,
    ) -> None:
        del action_ids
        self._preprocessor = FeaturePreprocessor.fit(
            [feature_rows[index] for index in train_indices],
            numeric_columns=ROUTER_ALLOWED_NUMERIC_FEATURES,
            categorical_columns=ROUTER_ALLOWED_CATEGORICAL_FEATURES,
        )
        targets: list[str] = []
        weights: list[float] = []
        kept_indices: list[int] = []
        for train_index in train_indices:
            label = label_rows[train_index].get('oracle_action_conservative')
            if label is None:
                continue
            no_op_utility = get_action_metric(
                label_rows[train_index],
                metric='utility_conservative',
                action_id=NO_OP_CANONICAL_ID,
            ) or 0.0
            oracle_utility = to_float(label_rows[train_index].get('oracle_utility_conservative')) or 0.0
            weight = abs(oracle_utility - no_op_utility) + 1e-3
            targets.append(str(label))
            weights.append(weight)
            kept_indices.append(train_index)
        if not targets:
            self._constant_label = None
            return
        unique_labels = sorted(set(targets))
        if len(unique_labels) <= 1:
            self._constant_label = unique_labels[0] if unique_labels else None
            return
        matrix = self._preprocessor.transform(
            [feature_rows[index] for index in kept_indices],
            scale_numeric=True,
        )
        min_samples_leaf = max(1, int(0.05 * len(train_indices)))
        self._model = DecisionTreeClassifier(
            random_state=random_state,
            max_depth=3,
            min_samples_leaf=min_samples_leaf,
        )
        self._model.fit(matrix, np.asarray(targets), sample_weight=np.asarray(weights, dtype=float))


class _LogisticConservative(RouterPolicy):
    """
    Logistic-regression conservative oracle classifier.
    """

    name = 'logistic_conservative'

    def __init__(self) -> None:
        self._preprocessor: Optional[FeaturePreprocessor] = None
        self._model: Any = None
        self._constant_label: Optional[str] = None

    def fit(
        self,
        *,
        feature_rows: list[dict[str, Any]],
        label_rows: list[dict[str, Any]],
        train_indices: list[int],
        action_ids: list[str],
        random_state: int,
    ) -> None:
        del action_ids
        self._preprocessor = FeaturePreprocessor.fit(
            [feature_rows[index] for index in train_indices],
            numeric_columns=ROUTER_ALLOWED_NUMERIC_FEATURES,
            categorical_columns=ROUTER_ALLOWED_CATEGORICAL_FEATURES,
        )
        targets: list[str] = []
        kept_indices: list[int] = []
        for train_index in train_indices:
            label = label_rows[train_index].get('oracle_action_conservative')
            if label is None:
                continue
            targets.append(str(label))
            kept_indices.append(train_index)
        if not targets:
            self._constant_label = None
            return
        unique_labels = sorted(set(targets))
        if len(unique_labels) <= 1:
            self._constant_label = unique_labels[0] if unique_labels else None
            return
        matrix = self._preprocessor.transform(
            [feature_rows[index] for index in kept_indices],
            scale_numeric=True,
        )
        self._model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            class_weight='balanced',
        )
        self._model.fit(matrix, np.asarray(targets))

    def predict(
        self,
        *,
        feature_rows: list[dict[str, Any]],
        test_indices: list[int],
    ) -> dict[int, Optional[str]]:
        if self._model is None:
            return {index: self._constant_label for index in test_indices}
        matrix = self._preprocessor.transform(
            [feature_rows[index] for index in test_indices],
            scale_numeric=True,
        )
        predictions = self._model.predict(matrix)
        return {index: str(predictions[offset]) for offset, index in enumerate(test_indices)}


class _GradientBoostedUtility(RouterPolicy):
    """
    Per-action gradient-boosted expected-utility router (un-gated).
    """

    name = 'gradient_boosted_utility'

    def __init__(self) -> None:
        self._preprocessor: Optional[FeaturePreprocessor] = None
        self._regressors: dict[str, tuple[Any, float]] = {}
        self._action_ids: list[str] = []
        self._action_family_by_id: dict[str, str] = {}
        self._no_op_action: Optional[str] = None

    def fit(
        self,
        *,
        feature_rows: list[dict[str, Any]],
        label_rows: list[dict[str, Any]],
        train_indices: list[int],
        action_ids: list[str],
        random_state: int,
    ) -> None:
        self._action_ids = list(action_ids)
        self._action_family_by_id = action_family_map(action_ids=action_ids)
        self._no_op_action = resolve_no_op_action_id(action_family_by_id=self._action_family_by_id)
        self._preprocessor = FeaturePreprocessor.fit(
            [feature_rows[index] for index in train_indices],
            numeric_columns=ROUTER_ALLOWED_NUMERIC_FEATURES,
            categorical_columns=ROUTER_ALLOWED_CATEGORICAL_FEATURES,
        )
        self._regressors = _fit_per_action_utility_regressors(
            preprocessor=self._preprocessor,
            feature_rows=feature_rows,
            label_rows=label_rows,
            train_indices=train_indices,
            action_ids=action_ids,
            random_state=random_state,
        )

    def predict(
        self,
        *,
        feature_rows: list[dict[str, Any]],
        test_indices: list[int],
    ) -> dict[int, Optional[str]]:
        if self._preprocessor is None:
            return {index: self._no_op_action for index in test_indices}
        predictions = _predict_per_action_utility(
            preprocessor=self._preprocessor,
            feature_rows=feature_rows,
            test_indices=test_indices,
            regressors=self._regressors,
        )
        selections: dict[int, Optional[str]] = {}
        for index in test_indices:
            row_predictions = predictions.get(index, {})
            best_action: Optional[str] = None
            best_score: Optional[tuple[float, int, str]] = None
            for action_id in sorted(self._action_ids):
                value = row_predictions.get(action_id)
                if value is None:
                    continue
                family = self._action_family_by_id.get(action_id, '')
                score = (-value, _family_priority(family), action_id)
                if best_score is None or score < best_score:
                    best_score = score
                    best_action = action_id
            selections[index] = best_action or self._no_op_action
        return selections


class _MonotoneThresholdPolicy(RouterPolicy):
    """
    Train-fitted interpretable monotone-threshold policy.
    """

    name = 'monotone_threshold_policy'

    def __init__(self) -> None:
        self._thresholds: dict[str, Optional[float]] = {}
        self._actions: dict[str, Optional[str]] = {}

    def fit(
        self,
        *,
        feature_rows: list[dict[str, Any]],
        label_rows: list[dict[str, Any]],
        train_indices: list[int],
        action_ids: list[str],
        random_state: int,
    ) -> None:
        del random_state
        action_family_by_id = action_family_map(action_ids=action_ids)
        self._actions = {
            'no_op':
                _resolve_family_action(
                    family='no_op',
                    action_ids=action_ids,
                    action_family_by_id=action_family_by_id,
                    label_rows=label_rows,
                    train_indices=train_indices,
                ),
            'bic':
                _resolve_family_action(
                    family='bic',
                    action_ids=action_ids,
                    action_family_by_id=action_family_by_id,
                    label_rows=label_rows,
                    train_indices=train_indices,
                ),
            'temperature_scaling':
                _resolve_family_action(
                    family='temperature_scaling',
                    action_ids=action_ids,
                    action_family_by_id=action_family_by_id,
                    label_rows=label_rows,
                    train_indices=train_indices,
                ),
            'prototype_mean_shift':
                _resolve_family_action(
                    family='prototype_mean_shift',
                    action_ids=action_ids,
                    action_family_by_id=action_family_by_id,
                    label_rows=label_rows,
                    train_indices=train_indices,
                ),
        }

        feature_definitions = {
            'forgetting_active_threshold': 'mean_forgetting',
            'headroom_active_threshold': 'headroom',
            'base_accuracy_noop_threshold': 'base_final_accuracy',
            'out_of_task_bic_threshold': 'mean_run.diagnostics.out_of_task_rate',
            'calibration_temperature_threshold': None,
            'nll_temperature_threshold': 'mean_run.calibration.nll',
        }
        candidate_grid: dict[str, list[Optional[float]]] = {}
        for threshold_name, source in feature_definitions.items():
            if threshold_name == 'calibration_temperature_threshold':
                values: list[float] = []
                for index in train_indices:
                    ece = to_float(feature_rows[index].get('mean_run.calibration.ece'))
                    aece = to_float(feature_rows[index].get('mean_run.calibration.aece'))
                    candidates = [value for value in (ece, aece) if value is not None]
                    if candidates:
                        values.append(max(candidates))
            else:
                values = _finite_values(rows=feature_rows, indices=train_indices, key=source)
            grid = [_percentile(values, percentile=percentile) for percentile in (20, 40, 60, 80)]
            candidate_grid[threshold_name] = [value for value in grid if value is not None] or [None]

        best_score = -math.inf
        best_thresholds: dict[str, Optional[float]] = {key: None for key in feature_definitions}
        for forgetting_threshold in candidate_grid['forgetting_active_threshold']:
            for headroom_threshold in candidate_grid['headroom_active_threshold']:
                for base_accuracy_threshold in candidate_grid['base_accuracy_noop_threshold']:
                    for out_of_task_threshold in candidate_grid['out_of_task_bic_threshold']:
                        for calibration_threshold in candidate_grid['calibration_temperature_threshold']:
                            for nll_threshold in candidate_grid['nll_temperature_threshold']:
                                thresholds = {
                                    'forgetting_active_threshold': forgetting_threshold,
                                    'headroom_active_threshold': headroom_threshold,
                                    'base_accuracy_noop_threshold': base_accuracy_threshold,
                                    'out_of_task_bic_threshold': out_of_task_threshold,
                                    'calibration_temperature_threshold': calibration_threshold,
                                    'nll_temperature_threshold': nll_threshold,
                                }
                                selections: dict[int, Optional[str]] = {}
                                for index in train_indices:
                                    selections[index] = self._select_action(
                                        feature_row=feature_rows[index],
                                        thresholds=thresholds,
                                    )
                                score = _selection_outcome_score(
                                    label_rows=label_rows,
                                    indices=train_indices,
                                    selections=selections,
                                )
                                if score > best_score:
                                    best_score = score
                                    best_thresholds = thresholds
        self._thresholds = best_thresholds

    def _select_action(
        self,
        *,
        feature_row: dict[str, Any],
        thresholds: dict[str, Optional[float]],
    ) -> Optional[str]:
        """
        Apply the policy to a single feature row.

        Args:
            feature_row: Source row.
            thresholds: Threshold values.

        Returns:
            Optional[str]: Selected action id.
        """
        base_accuracy = to_float(feature_row.get('base_final_accuracy'))
        headroom = to_float(feature_row.get('headroom'))
        forgetting = to_float(feature_row.get('mean_forgetting'))
        out_of_task = to_float(feature_row.get('mean_run.diagnostics.out_of_task_rate'))
        nll = to_float(feature_row.get('mean_run.calibration.nll'))
        ece = to_float(feature_row.get('mean_run.calibration.ece'))
        aece = to_float(feature_row.get('mean_run.calibration.aece'))
        ece_candidates = [value for value in (ece, aece) if value is not None]
        ece_value = max(ece_candidates) if ece_candidates else None
        no_op_action = self._actions.get('no_op')

        base_threshold = thresholds.get('base_accuracy_noop_threshold')
        headroom_threshold = thresholds.get('headroom_active_threshold')
        if (base_threshold is not None and base_accuracy is not None and base_accuracy >= base_threshold and
                headroom_threshold is not None and headroom is not None and headroom < headroom_threshold):
            return no_op_action
        forgetting_threshold = thresholds.get('forgetting_active_threshold')
        if (forgetting_threshold is not None and forgetting is not None and forgetting < forgetting_threshold):
            return no_op_action
        out_of_task_threshold = thresholds.get('out_of_task_bic_threshold')
        if (out_of_task_threshold is not None and out_of_task is not None and out_of_task > out_of_task_threshold):
            return self._actions.get('bic') or no_op_action
        calibration_threshold = thresholds.get('calibration_temperature_threshold')
        nll_threshold = thresholds.get('nll_temperature_threshold')
        calibration_high = (calibration_threshold is not None and ece_value is not None and
                            ece_value > calibration_threshold)
        nll_high = nll_threshold is not None and nll is not None and nll > nll_threshold
        if calibration_high or nll_high:
            return self._actions.get('temperature_scaling') or no_op_action
        if (headroom_threshold is not None and headroom is not None and headroom > headroom_threshold):
            return self._actions.get('prototype_mean_shift') or no_op_action
        return no_op_action

    def predict(
        self,
        *,
        feature_rows: list[dict[str, Any]],
        test_indices: list[int],
    ) -> dict[int, Optional[str]]:
        return {
            index: self._select_action(
                feature_row=feature_rows[index],
                thresholds=self._thresholds,
            ) for index in test_indices
        }


def build_policies() -> list[RouterPolicy]:
    """
    Construct the full policy set used for routing.

    Returns:
        list[RouterPolicy]: Newly constructed policy instances.
    """
    return [
        _AlwaysAction(name='always_no_op', family='no_op'),
        _AlwaysAction(name='always_weight_aligning', family='weight_aligning'),
        _AlwaysAction(name='always_bic', family='bic'),
        _AlwaysAction(name='always_temperature_scaling', family='temperature_scaling'),
        _AlwaysAction(name='always_prototype_mean_shift', family='prototype_mean_shift'),
        _AlwaysAction(name='always_linear_probe', family='linear_probe'),
        _BestStaticAction(
            name='best_static_low_cost_conservative',
            family_pool=LOW_COST_ACTION_FAMILIES,
        ),
        _ThresholdSingleFeatureConservative(),
        _ThresholdDiagnosticCascade(),
        _TwoStageExpectedUtilityRouter(),
        _DecisionStumpConservative(),
        _ShallowTreeConservative(),
        _LogisticConservative(),
        _GradientBoostedUtility(),
        _MonotoneThresholdPolicy(),
    ]
