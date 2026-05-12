"""
Scoring, summary aggregation, and decision-gate logic for the router analysis.
"""

import json
import math
from typing import Any, Optional

import numpy as np

from regain.analysis.router.actions import get_action_metric
from regain.analysis.router.actions import is_action_evaluable
from regain.analysis.router.actions import is_pareto_value
from regain.analysis.router.constants import COVERAGE_THRESHOLD
from regain.analysis.router.constants import NO_OP_CANONICAL_ID
from regain.analysis.router.constants import ROUTER_POLICY_NAMES
from regain.analysis.router.constants import STATIC_LOW_COST_BASELINES
from regain.analysis.router.folds import RouterFold
from regain.analysis.router.policies import build_policies
from regain.analysis.utils import to_float


def _safe_mean(values: list[Optional[float]]) -> Optional[float]:
    """
    Compute mean of finite values, returning None when no finite values exist.

    Args:
        values: Optional numeric values.

    Returns:
        Optional[float]: Mean value or None.
    """
    finite = [value for value in values if value is not None]
    if not finite:
        return None
    return float(np.mean(finite))


def _safe_max(values: list[Optional[float]]) -> Optional[float]:
    """
    Compute max of finite values.

    Args:
        values: Optional numeric values.

    Returns:
        Optional[float]: Max value or None.
    """
    finite = [value for value in values if value is not None]
    if not finite:
        return None
    return float(max(finite))


def _build_prediction_rows(
    *,
    fold: RouterFold,
    policy_name: str,
    selections: dict[int, Optional[str]],
    feature_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    action_family_by_id: dict[str, str],
    decision_time_controller_fits: int,
) -> list[dict[str, Any]]:
    """
    Build per-row prediction rows for a fold.

    Args:
        fold: Validation fold.
        policy_name: Policy identifier.
        selections: Mapping from row index to selected action id.
        feature_rows: Feature-table rows.
        label_rows: Labels-table rows.
        action_family_by_id: Mapping from action id to family.
        decision_time_controller_fits: Count of candidate controllers fitted at decision time.

    Returns:
        list[dict[str, Any]]: Prediction rows.
    """
    rows: list[dict[str, Any]] = []
    for index in fold.test_indices:
        selection = selections.get(index)
        label_row = label_rows[index]
        feature_row = feature_rows[index]
        oracle_primary = to_float(label_row.get('oracle_utility_primary'))
        oracle_conservative = to_float(label_row.get('oracle_utility_conservative'))
        oracle_cost_aware = to_float(label_row.get('oracle_utility_cost_aware'))
        selected_primary = (get_action_metric(label_row, metric='utility_primary', action_id=selection)
                            if selection is not None else None)
        selected_conservative = (get_action_metric(label_row, metric='utility_conservative', action_id=selection)
                                 if selection is not None else None)
        selected_cost_aware = (get_action_metric(label_row, metric='utility_cost_aware', action_id=selection)
                               if selection is not None else None)
        selected_harmed = (get_action_metric(label_row, metric='mean_harmed_task_fraction', action_id=selection)
                           if selection is not None else None)
        selected_worst_harm = (get_action_metric(label_row, metric='worst_task_harm', action_id=selection)
                               if selection is not None else None)
        selected_pareto = (is_pareto_value(label_row.get(f'is_pareto__{selection}')) if selection is not None else None)
        evaluable = selection is not None and is_action_evaluable(row=label_row, action_id=selection)
        regret_primary = (oracle_primary -
                          selected_primary if oracle_primary is not None and selected_primary is not None else None)
        regret_conservative = (oracle_conservative - selected_conservative
                               if oracle_conservative is not None and selected_conservative is not None else None)
        regret_cost_aware = (oracle_cost_aware - selected_cost_aware
                             if oracle_cost_aware is not None and selected_cost_aware is not None else None)
        family = action_family_by_id.get(selection) if selection is not None else None
        rows.append({
            'validation_level': fold.validation_level,
            'fold_id': fold.fold_id,
            'heldout_group': fold.heldout_group,
            'policy_name': policy_name,
            'experiment_id': feature_row.get('experiment_id'),
            'scenario': feature_row.get('scenario'),
            'backbone_name': feature_row.get('backbone_name'),
            'strategy_name': feature_row.get('strategy_name'),
            'seed': feature_row.get('seed'),
            'b': feature_row.get('b'),
            'repair_budget_fraction': feature_row.get('repair_budget_fraction'),
            'repair_budget_total': feature_row.get('repair_budget_total'),
            'selected_action_id': selection,
            'selected_action_family': family,
            'selected_action_available': evaluable,
            'decision_time_controller_fits': decision_time_controller_fits,
            'utility_conservative': selected_conservative,
            'utility_primary': selected_primary,
            'utility_cost_aware': selected_cost_aware,
            'regret_conservative': regret_conservative,
            'regret_primary': regret_primary,
            'regret_cost_aware': regret_cost_aware,
            'mean_harmed_task_fraction': selected_harmed,
            'worst_task_harm': selected_worst_harm,
            'is_pareto': selected_pareto,
            'oracle_action_conservative': label_row.get('oracle_action_conservative'),
            'oracle_action_primary': label_row.get('oracle_action_primary'),
            'oracle_action_cost_aware': label_row.get('oracle_action_cost_aware'),
        })
    return rows


def aggregate_policy_summary(
    *,
    prediction_rows: list[dict[str, Any]],
    action_family_by_id: dict[str, str],
    manifest_warnings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Compute the per-policy summary metrics grouped by validation level.

    Args:
        prediction_rows: All policy prediction rows.
        action_family_by_id: Mapping from action id to family.
        manifest_warnings: Mutable manifest warnings.

    Returns:
        list[dict[str, Any]]: Summary rows.
    """
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in prediction_rows:
        key = (str(row['validation_level']), str(row['policy_name']))
        grouped.setdefault(key, []).append(row)

    reference_means: dict[tuple[str, str], dict[str, Optional[float]]] = {}
    for (validation_level, policy_name), rows in grouped.items():
        reference_means[(validation_level, policy_name)] = {
            'utility_conservative':
                _safe_mean([
                    to_float(row.get('utility_conservative')) for row in rows if row.get('selected_action_available')
                ]),
            'mean_harmed_task_fraction':
                _safe_mean([
                    to_float(row.get('mean_harmed_task_fraction'))
                    for row in rows
                    if row.get('selected_action_available')
                ]),
            'worst_task_harm':
                _safe_max(
                    [to_float(row.get('worst_task_harm')) for row in rows if row.get('selected_action_available')]),
        }

    summary_rows: list[dict[str, Any]] = []
    for (validation_level, policy_name), rows in sorted(grouped.items()):
        evaluable_rows = [row for row in rows if row.get('selected_action_available')]
        num_rows = len(rows)
        num_evaluable = len(evaluable_rows)
        coverage_fraction = (num_evaluable / num_rows) if num_rows else None
        utility_conservative_values = [to_float(row.get('utility_conservative')) for row in evaluable_rows]
        utility_primary_values = [to_float(row.get('utility_primary')) for row in evaluable_rows]
        utility_cost_aware_values = [to_float(row.get('utility_cost_aware')) for row in evaluable_rows]
        regret_conservative_values = [to_float(row.get('regret_conservative')) for row in evaluable_rows]
        regret_primary_values = [to_float(row.get('regret_primary')) for row in evaluable_rows]
        regret_cost_aware_values = [to_float(row.get('regret_cost_aware')) for row in evaluable_rows]
        harmed_fraction_values = [to_float(row.get('mean_harmed_task_fraction')) for row in evaluable_rows]
        worst_harm_values = [to_float(row.get('worst_task_harm')) for row in evaluable_rows]
        pareto_values = [
            1.0 if row.get('is_pareto') is True else 0.0 if row.get('is_pareto') is False else None
            for row in evaluable_rows
        ]
        controller_fit_values = [to_float(row.get('decision_time_controller_fits')) for row in evaluable_rows]

        always_bic_mean = reference_means.get((validation_level, 'always_bic'), {}).get('utility_conservative')
        always_wa_mean = reference_means.get((validation_level, 'always_weight_aligning'),
                                             {}).get('utility_conservative')
        always_linear_harmed = reference_means.get((validation_level, 'always_linear_probe'),
                                                   {}).get('mean_harmed_task_fraction')
        always_linear_worst = reference_means.get((validation_level, 'always_linear_probe'), {}).get('worst_task_harm')

        mean_utility_conservative = _safe_mean(utility_conservative_values)
        mean_utility_primary = _safe_mean(utility_primary_values)
        mean_utility_cost_aware = _safe_mean(utility_cost_aware_values)
        mean_regret_conservative = _safe_mean(regret_conservative_values)
        mean_regret_primary = _safe_mean(regret_primary_values)
        mean_regret_cost_aware = _safe_mean(regret_cost_aware_values)
        mean_harmed = _safe_mean(harmed_fraction_values)
        max_worst_harm = _safe_max(worst_harm_values)

        improvement_bic: Optional[float]
        if mean_utility_conservative is not None and always_bic_mean is not None:
            improvement_bic = float(mean_utility_conservative - always_bic_mean)
        else:
            improvement_bic = None
        improvement_wa: Optional[float]
        if mean_utility_conservative is not None and always_wa_mean is not None:
            improvement_wa = float(mean_utility_conservative - always_wa_mean)
        else:
            improvement_wa = None
        harm_reduction_fraction: Optional[float]
        if always_linear_harmed is not None and mean_harmed is not None:
            harm_reduction_fraction = float(always_linear_harmed - mean_harmed)
        else:
            harm_reduction_fraction = None
        harm_reduction_worst: Optional[float]
        if always_linear_worst is not None and max_worst_harm is not None:
            harm_reduction_worst = float(always_linear_worst - max_worst_harm)
        else:
            harm_reduction_worst = None

        no_op_predicted = [
            row for row in evaluable_rows if str(row.get('selected_action_family') or '') == NO_OP_CANONICAL_ID
        ]
        no_op_oracle = [
            row for row in evaluable_rows
            if action_family_by_id.get(str(row.get('oracle_action_conservative') or '')) == NO_OP_CANONICAL_ID
        ]
        no_op_correct = [
            row for row in no_op_predicted
            if action_family_by_id.get(str(row.get('oracle_action_conservative') or '')) == NO_OP_CANONICAL_ID
        ]
        if no_op_predicted:
            no_op_precision = float(len(no_op_correct) / len(no_op_predicted))
        else:
            no_op_precision = None
            manifest_warnings.append({
                'code': 'noop_precision_zero_denominator',
                'message': 'No predictions selected the no-op family.',
                'context': {
                    'validation_level': validation_level,
                    'policy_name': policy_name
                },
            })
        if no_op_oracle:
            no_op_recall = float(len(no_op_correct) / len(no_op_oracle))
        else:
            no_op_recall = None
            manifest_warnings.append({
                'code': 'noop_recall_zero_denominator',
                'message': 'No oracle labels mapped to the no-op family.',
                'context': {
                    'validation_level': validation_level,
                    'policy_name': policy_name
                },
            })

        action_distribution: dict[str, int] = {}
        for row in evaluable_rows:
            family = str(row.get('selected_action_family') or '')
            action_distribution[family] = action_distribution.get(family, 0) + 1

        summary_rows.append({
            'validation_level': validation_level,
            'policy_name': policy_name,
            'num_rows': num_rows,
            'num_evaluable_rows': num_evaluable,
            'coverage_fraction': coverage_fraction,
            'mean_utility_conservative': mean_utility_conservative,
            'mean_utility_primary': mean_utility_primary,
            'mean_utility_cost_aware': mean_utility_cost_aware,
            'mean_regret_conservative': mean_regret_conservative,
            'mean_regret_primary': mean_regret_primary,
            'mean_regret_cost_aware': mean_regret_cost_aware,
            'mean_improvement_over_always_bic_conservative': improvement_bic,
            'mean_improvement_over_always_weight_aligning_conservative': improvement_wa,
            'harm_reduction_vs_always_linear_probe_fraction': harm_reduction_fraction,
            'harm_reduction_vs_always_linear_probe_worst': harm_reduction_worst,
            'mean_harmed_task_fraction': mean_harmed,
            'max_worst_task_harm': max_worst_harm,
            'mean_decision_time_controller_fits': _safe_mean(controller_fit_values),
            'pareto_hit_rate': _safe_mean(pareto_values),
            'noop_precision': no_op_precision,
            'noop_recall': no_op_recall,
            'selected_action_distribution_json': json.dumps(
                dict(sorted(action_distribution.items())),
                sort_keys=True,
            ),
        })
    return summary_rows


def build_decision_gate(
    *,
    summary_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute the per-validation-level decision gate payload.

    Args:
        summary_rows: Aggregated policy summary rows.

    Returns:
        dict[str, Any]: Decision-gate payload.
    """
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in summary_rows:
        validation_level = str(row['validation_level'])
        grouped.setdefault(validation_level, {})[str(row['policy_name'])] = row

    gate_payload: dict[str, Any] = {
        'schema': {
            'name': 'regain.analysis.router.decision_gate',
            'version': 1,
        },
        'coverage_threshold': COVERAGE_THRESHOLD,
        'levels': {},
    }
    for validation_level, by_policy in grouped.items():
        candidate_routers = [
            by_policy[name]
            for name in ROUTER_POLICY_NAMES
            if name in by_policy and (by_policy[name].get('coverage_fraction') or 0.0) >= COVERAGE_THRESHOLD and
            by_policy[name].get('mean_utility_conservative') is not None
        ]
        candidate_static = [
            by_policy[name]
            for name in STATIC_LOW_COST_BASELINES
            if name in by_policy and (by_policy[name].get('coverage_fraction') or 0.0) >= COVERAGE_THRESHOLD and
            by_policy[name].get('mean_utility_conservative') is not None
        ]
        linear_reference = by_policy.get('always_linear_probe')
        if linear_reference is not None and (linear_reference.get('coverage_fraction') or 0.0) < COVERAGE_THRESHOLD:
            linear_reference = None

        strongest_router = max(
            candidate_routers,
            key=lambda item: item.get('mean_utility_conservative') or -math.inf,
            default=None,
        )
        strongest_static = max(
            candidate_static,
            key=lambda item: item.get('mean_utility_conservative') or -math.inf,
            default=None,
        )

        router_beats_static = (strongest_router is not None and strongest_static is not None and
                               (strongest_router.get('mean_utility_conservative') or
                                -math.inf) > (strongest_static.get('mean_utility_conservative') or -math.inf))
        router_reduces_harm = False
        if strongest_router is not None and linear_reference is not None:
            router_harm = strongest_router.get('mean_harmed_task_fraction')
            linear_harm = linear_reference.get('mean_harmed_task_fraction')
            router_worst = strongest_router.get('max_worst_task_harm')
            linear_worst = linear_reference.get('max_worst_task_harm')
            if router_harm is not None and linear_harm is not None and router_harm < linear_harm:
                router_reduces_harm = True
            elif router_worst is not None and linear_worst is not None and router_worst < linear_worst:
                router_reduces_harm = True
        success = bool(router_beats_static and router_reduces_harm)

        simple_policy_success = False
        if strongest_static is not None and linear_reference is not None:
            for simple_name in (
                    'threshold_single_feature_conservative',
                    'threshold_diagnostic_cascade',
                    'monotone_threshold_policy',
            ):
                policy_row = by_policy.get(simple_name)
                if policy_row is None:
                    continue
                if (policy_row.get('coverage_fraction') or 0.0) < COVERAGE_THRESHOLD:
                    continue
                if policy_row.get('mean_utility_conservative') is None:
                    continue
                if (policy_row.get('mean_utility_conservative') or
                        -math.inf) <= (strongest_static.get('mean_utility_conservative') or -math.inf):
                    continue
                policy_harm = policy_row.get('mean_harmed_task_fraction')
                policy_worst = policy_row.get('max_worst_task_harm')
                linear_harm = linear_reference.get('mean_harmed_task_fraction')
                linear_worst = linear_reference.get('max_worst_task_harm')
                if (policy_harm is not None and linear_harm is not None and
                        policy_harm < linear_harm) or (policy_worst is not None and linear_worst is not None and
                                                       policy_worst < linear_worst):
                    simple_policy_success = True
                    break

        recommended_next_step = ('router_viable_continue_refinement'
                                 if simple_policy_success else 'focus_on_safer_controllers_before_more_complex_router')

        gate_payload['levels'][validation_level] = {
            'router_beats_static': bool(router_beats_static),
            'router_reduces_harm': bool(router_reduces_harm),
            'success': success,
            'recommended_next_step': recommended_next_step,
            'strongest_router': {
                'policy_name': strongest_router['policy_name'] if strongest_router else None,
                'mean_utility_conservative':
                    (strongest_router.get('mean_utility_conservative') if strongest_router else None),
                'mean_harmed_task_fraction':
                    (strongest_router.get('mean_harmed_task_fraction') if strongest_router else None),
                'max_worst_task_harm': (strongest_router.get('max_worst_task_harm') if strongest_router else None),
                'coverage_fraction': (strongest_router.get('coverage_fraction') if strongest_router else None),
            },
            'strongest_static_low_cost': {
                'policy_name': strongest_static['policy_name'] if strongest_static else None,
                'mean_utility_conservative':
                    (strongest_static.get('mean_utility_conservative') if strongest_static else None),
                'coverage_fraction': (strongest_static.get('coverage_fraction') if strongest_static else None),
            },
            'always_linear_probe_reference': {
                'mean_harmed_task_fraction':
                    (linear_reference.get('mean_harmed_task_fraction') if linear_reference else None),
                'max_worst_task_harm': (linear_reference.get('max_worst_task_harm') if linear_reference else None),
                'coverage_fraction': (linear_reference.get('coverage_fraction') if linear_reference else None),
            },
        }
    return gate_payload


def evaluate_policies(
    *,
    folds: list[RouterFold],
    feature_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    action_ids: list[str],
    action_family_by_id: dict[str, str],
    random_state: int,
    manifest_warnings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Evaluate every policy on every fold.

    Args:
        folds: Validation folds.
        feature_rows: Feature-table rows.
        label_rows: Labels-table rows.
        action_ids: Action identifiers.
        action_family_by_id: Mapping from action id to family.
        random_state: Deterministic seed.
        manifest_warnings: Mutable manifest warnings.

    Returns:
        list[dict[str, Any]]: Prediction rows across all (fold, policy) pairs.
    """
    prediction_rows: list[dict[str, Any]] = []
    for fold in folds:
        for policy in build_policies():
            try:
                policy.fit(
                    feature_rows=feature_rows,
                    label_rows=label_rows,
                    train_indices=list(fold.train_indices),
                    action_ids=action_ids,
                    random_state=random_state,
                )
                selections = policy.predict(
                    feature_rows=feature_rows,
                    test_indices=list(fold.test_indices),
                )
            except Exception as exc:
                manifest_warnings.append({
                    'code': 'policy_evaluation_failed',
                    'message': f'Policy `{policy.name}` failed on fold `{fold.fold_id}`: {exc}',
                    'context': {
                        'validation_level': fold.validation_level,
                        'fold_id': fold.fold_id,
                        'policy_name': policy.name,
                    },
                })
                continue
            prediction_rows.extend(
                _build_prediction_rows(
                    fold=fold,
                    policy_name=policy.name,
                    selections=selections,
                    feature_rows=feature_rows,
                    label_rows=label_rows,
                    action_family_by_id=action_family_by_id,
                    decision_time_controller_fits=policy.decision_time_controller_fits,
                ))
    return prediction_rows
