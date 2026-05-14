"""
Shared constants for the repair router analysis package.
"""

__all__ = [
    'ROUTER_ALLOWED_CATEGORICAL_FEATURES',
    'ROUTER_ALLOWED_NUMERIC_FEATURES',
    'ROUTER_ID_COLUMNS',
    'ROUTER_POLICY_NAMES',
    'STATIC_LOW_COST_BASELINES',
    'UTILITY_METRICS',
]

ROUTER_ID_COLUMNS = (
    'experiment_id',
    'scenario',
    'backbone_name',
    'strategy_name',
    'seed',
    'b',
    'repair_budget_fraction',
    'repair_budget_total',
)

ROUTER_ALLOWED_CATEGORICAL_FEATURES = (
    'scenario',
    'backbone_name',
    'strategy_name',
)

ROUTER_ALLOWED_NUMERIC_FEATURES = (
    'replay_mem_size',
    'replay_batch_size_mem',
    'repair_budget_fraction',
    'repair_budget_total',
    'repair_set_total',
    'repair_split_fraction',
    'num_classes',
    'mean_A_ref',
    'mean_A_post',
    'base_final_accuracy',
    'reference_accuracy',
    'mean_forgetting',
    'headroom',
    'task_age_mean',
    'task_age_min',
    'task_age_max',
    'task_age_std',
    'oldest_task_forgetting',
    'newest_task_forgetting',
    'age_weighted_forgetting',
    'run.calibration.max_ece',
    'mean_run.calibration.ece',
    'mean_run.calibration.aece',
    'mean_run.calibration.nll',
    'mean_run.diagnostics.out_of_task_rate',
    'mean_run.diagnostics.avg_conf',
    'mean_run.diagnostics.avg_entropy',
    'mean_run.diagnostics.logit_avg_drift',
)

UTILITY_METRICS = (
    'utility_conservative',
    'utility_primary',
    'utility_cost_aware',
)

ROUTER_POLICY_NAMES = (
    'two_stage_expected_utility_router',
    'decision_stump_conservative',
    'shallow_tree_conservative',
    'logistic_conservative',
    'gradient_boosted_utility',
    'monotone_threshold_policy',
)

STATIC_LOW_COST_BASELINES = (
    'always_no_op',
    'always_weight_aligning',
    'always_bic',
    'always_temperature_scaling',
    'always_prototype_mean_shift',
    'best_static_low_cost_conservative',
)

HELD_SEED = 'held_seed'
HELD_SETTING = 'held_setting'
HELD_DATASET = 'held_dataset'

NO_OP_CANONICAL_ID = 'no_op'
COVERAGE_THRESHOLD = 0.8
