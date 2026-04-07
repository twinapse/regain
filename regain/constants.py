"""
Project-wide constants.

Naming conventions
------------------

**Constant prefixes:**
- ``RUN_*``       — per-run metric keys logged to MLflow (prefixed with ``run.`` in string values).
- ``ANALYSIS_*``  — cross-run aggregated metric keys (CSV columns, prefixed with ``analysis.``).
- ``COLUMN_*``    — table metadata column keys (run ID, seed, controller name, …).
- ``PARAM_*``     — MLflow run parameter paths.

**Namespace hierarchy (dot-separated):**
``run.<namespace>.[…dimensions…].<base|ctrl>``   for per-run metrics.
``analysis.<namespace>.[…dimensions…].<base|ctrl>`` for cross-run aggregates.

``base`` = without controller corrections; ``ctrl`` = with controller corrections.
``base`` / ``ctrl`` is **always the last segment** of a metric key.

**Temporal markers:**
- ``exp``   — measured at end of each experience (reference accuracy).
- ``final`` — measured at end of all training (post-sequence accuracy).

**Aggregation suffixes:**
- ``avg`` — average value.
- ``std`` — standard deviation (cross-run only).

**Avalanche native metrics** also carry the ``run.`` prefix via namespace constants
(``NAMESPACE_TRAIN``, ``NAMESPACE_EVAL``).
"""

__all__ = [
    # ---- Namespace constants ----
    'NAMESPACE_ANALYSIS',
    'NAMESPACE_DEBUG',
    'NAMESPACE_EVAL',
    'NAMESPACE_RUN',
    'NAMESPACE_TRAIN',
    'NS_SEP',
    # ---- Per-run metric keys (MLflow) ----
    'RUN_ACC_FINAL_TEST',
    'RUN_ACC_FINAL_TEST_AVG_BASE',
    'RUN_ACC_FINAL_TEST_AVG_CTRL',
    'RUN_ACC_FINAL_TRAIN',
    'RUN_ACC_FINAL_TRAIN_AVG_BASE',
    'RUN_ACC_FINAL_TRAIN_AVG_CTRL',
    'RUN_ACC_REF_TEST',
    'RUN_ACC_REF_TRAIN',
    'RUN_CALIB_AECE',
    'RUN_CALIB_BRIER',
    'RUN_CALIB_ECE',
    'RUN_CALIB_MAX_ECE',
    'RUN_CALIB_MCE',
    'RUN_CALIB_NLL',
    'RUN_DIAG_AVG_CONF',
    'RUN_DIAG_AVG_ENTROPY',
    'RUN_DIAG_LOGIT_AVG_DRIFT',
    'RUN_DIAG_OUT_OF_TASK_RATE',
    'RUN_EPS',
    'RUN_LATENCY_MS_PER_SAMPLE_BASE',
    'RUN_LATENCY_MS_PER_SAMPLE_CTRL',
    'RUN_LATENCY_MS_RATIO',
    'RUN_LATENCY_SAMPLES_PER_SEC_BASE',
    'RUN_LATENCY_SAMPLES_PER_SEC_CTRL',
    'RUN_REPAIR_SECONDS',
    'RUN_REPAIR_STEPS',
    'RUN_RHO',
    'RUN_RHO_AVG',
    # ---- Cross-run metric keys (CSV / analysis) ----
    'ANALYSIS_ACC_FINAL_AVG_CTRL',
    'ANALYSIS_RHO_AVG',
    # ---- Diagnostic vector tuple ----
    'DIAG_VECTOR_KEYS',
    # ---- Table column keys ----
    'COLUMN_B',
    'COLUMN_CONTROLLER_MODEL_PARAM_COUNT',
    'COLUMN_CONTROLLER_NAME',
    'COLUMN_END_TIME',
    'COLUMN_EXPERIMENT_ID',
    'COLUMN_EXP_IDX',
    'COLUMN_NUM_CLASSES',
    'COLUMN_PERFORMANCE',
    'COLUMN_REPAIR_BUDGET_FRACTION',
    'COLUMN_REPAIR_BUDGET_TOTAL',
    'COLUMN_REPAIR_SET_TOTAL',
    'COLUMN_REPAIR_SPLIT_FRACTION',
    'COLUMN_RUN_ID',
    'COLUMN_RUN_NAME',
    'COLUMN_SEED',
    'COLUMN_START_TIME',
    'COLUMN_STATUS',
    'COLUMN_TASK_AGE',
    'COLUMN_TOTAL_COST',
    # ---- MLflow parameters ----
    'PARAM_AVALANCHE_VERSION',
    'PARAM_BACKBONE',
    'PARAM_BACKBONE_REPLAY_BATCH_SIZE_MEM',
    'PARAM_BACKBONE_REPLAY_MEM_SIZE',
    'PARAM_CONTROLLER',
    'PARAM_CONTROLLER_MODEL_PARAM_COUNT',
    'PARAM_CONTROLLER_PATH',
    'PARAM_CONTROLLER_TYPE',
    'PARAM_DEBUG_SKIP_REASON',
    'PARAM_NUM_CLASSES',
    'PARAM_REPAIR_BUDGET_FRACTION',
    'PARAM_REPAIR_SPLIT_FRACTION',
    'PARAM_RUN_NAME',
    'PARAM_SCENARIO',
    'PARAM_SEED',
    'PARAM_TORCH_DETERMINISTIC_ALGORITHMS',
    # ---- Misc ----
    'EXPERIENCE_KEY_PREFIX',
    'MLFLOW_ARTIFACT_ANALYSIS_FILE',
    'MLFLOW_ARTIFACT_BACKBONE_CHECKPOINTS_DIR',
    'MLFLOW_ARTIFACT_CONFIG_FILE',
    'MLFLOW_ARTIFACT_ERROR_FILE',
    'MLFLOW_ARTIFACT_PREDICTIONS_DIR',
    'MLFLOW_ARTIFACT_SPLITS_FILE',
    'RUN_NAME_BACKBONE',
    'STREAMS',
    'STREAM_REPAIR',
    'STREAM_TEST',
    'STREAM_TRAIN',
]

###########################
# Namespace hierarchy     #
###########################

NS_SEP = '.'  # Namespace separator

NAMESPACE_RUN = 'run'
NAMESPACE_ANALYSIS = 'analysis'
NAMESPACE_TRAIN = f'{NAMESPACE_RUN}{NS_SEP}train'
NAMESPACE_EVAL = f'{NAMESPACE_RUN}{NS_SEP}eval'
NAMESPACE_DEBUG = f'{NAMESPACE_RUN}{NS_SEP}debug'

########################
# Predefined run names #
########################

RUN_NAME_BACKBONE = 'backbone'

#################################################################
# Predefined run parameter paths                                #
#################################################################
# Used for:                                                     #
#   1. Configuring runs (pydantic keys)                         #
#   2. Logging parameters to MLflow                             #
#   3. Building experiment components (e.g. models, strategies) #
#################################################################
# TODO: Use a helper that resolves the parameter path from the Pydantic schema field instead of hardcoding.

PARAM_AVALANCHE_VERSION = 'avalanche_version'
PARAM_BACKBONE = 'backbone'
PARAM_BACKBONE_REPLAY_BATCH_SIZE_MEM = f'{PARAM_BACKBONE}{NS_SEP}training{NS_SEP}strategy{NS_SEP}batch_size_mem'
PARAM_BACKBONE_REPLAY_MEM_SIZE = f'{PARAM_BACKBONE}{NS_SEP}training{NS_SEP}strategy{NS_SEP}mem_size'
PARAM_CONTROLLER = 'controller'
PARAM_CONTROLLER_MODEL_PARAM_COUNT = f'{PARAM_CONTROLLER}{NS_SEP}model{NS_SEP}param_count'
PARAM_CONTROLLER_PATH = f'{PARAM_CONTROLLER}{NS_SEP}path'
PARAM_CONTROLLER_TYPE = f'{PARAM_CONTROLLER}{NS_SEP}type'
PARAM_DEBUG_SKIP_REASON = f'debug{NS_SEP}skip_reason'
PARAM_NUM_CLASSES = 'num_classes'
PARAM_REPAIR_BUDGET_FRACTION = 'repair.budget_fraction'
PARAM_REPAIR_SPLIT_FRACTION = 'repair.split_fraction'
PARAM_RUN_NAME = 'run_name'
PARAM_SCENARIO = 'scenario'
PARAM_SEED = 'seed'
PARAM_TORCH_DETERMINISTIC_ALGORITHMS = 'torch_deterministic_algorithms'

###################################################################
# Tabular export / analysis column keys                           #
###################################################################
# Used for:                                                       #
#   1. Analysis and visualization (plotting)                      #
#   2. Generating tabular exports (pandas DataFrames, CSVs)       #
#   3. Reading back flattened metrics/params from analysis tables #
###################################################################

COLUMN_B = 'b'
COLUMN_CONTROLLER_MODEL_PARAM_COUNT = 'controller_model_param_count'
COLUMN_CONTROLLER_NAME = 'controller_name'
COLUMN_END_TIME = 'end_time'
COLUMN_EXPERIMENT_ID = 'experiment_id'
COLUMN_EXP_IDX = 'exp_idx'
COLUMN_NUM_CLASSES = PARAM_NUM_CLASSES
COLUMN_PERFORMANCE = 'performance'
COLUMN_REPAIR_BUDGET_FRACTION = 'repair_budget_fraction'
COLUMN_REPAIR_BUDGET_TOTAL = 'repair_budget_total'
COLUMN_REPAIR_SET_TOTAL = 'repair_set_total'
COLUMN_REPAIR_SPLIT_FRACTION = 'repair_split_fraction'
COLUMN_RUN_ID = 'run_id'
COLUMN_RUN_NAME = PARAM_RUN_NAME
COLUMN_SEED = PARAM_SEED
COLUMN_START_TIME = 'start_time'
COLUMN_STATUS = 'status'
COLUMN_TASK_AGE = 'task_age'
COLUMN_TOTAL_COST = 'total_cost'

#########################################
# Per-run metric keys (logged to MLflow) #
#########################################
# All string values carry the ``run.`` prefix.
# base/ctrl variant is always at the very end.

_RUN = NAMESPACE_RUN  # shorthand for building metric strings

# Unified accuracy metrics under `run.eval.acc.*`.
RUN_ACC_REF_TEST = f'{NAMESPACE_EVAL}{NS_SEP}acc{NS_SEP}ref{NS_SEP}test'
RUN_ACC_REF_TRAIN = f'{NAMESPACE_EVAL}{NS_SEP}acc{NS_SEP}ref{NS_SEP}train'
RUN_ACC_FINAL_TEST = f'{NAMESPACE_EVAL}{NS_SEP}acc{NS_SEP}final{NS_SEP}test'
RUN_ACC_FINAL_TRAIN = f'{NAMESPACE_EVAL}{NS_SEP}acc{NS_SEP}final{NS_SEP}train'
RUN_ACC_FINAL_TEST_AVG_BASE = (
    f'{NAMESPACE_EVAL}{NS_SEP}acc{NS_SEP}final{NS_SEP}test{NS_SEP}avg{NS_SEP}base'
)
RUN_ACC_FINAL_TEST_AVG_CTRL = (
    f'{NAMESPACE_EVAL}{NS_SEP}acc{NS_SEP}final{NS_SEP}test{NS_SEP}avg{NS_SEP}ctrl'
)
RUN_ACC_FINAL_TRAIN_AVG_BASE = (
    f'{NAMESPACE_EVAL}{NS_SEP}acc{NS_SEP}final{NS_SEP}train{NS_SEP}avg{NS_SEP}base'
)
RUN_ACC_FINAL_TRAIN_AVG_CTRL = (
    f'{NAMESPACE_EVAL}{NS_SEP}acc{NS_SEP}final{NS_SEP}train{NS_SEP}avg{NS_SEP}ctrl'
)

# Rho (correctable fraction)
RUN_RHO = f'{_RUN}{NS_SEP}repair{NS_SEP}rho'                          # run.repair.rho (+ .exp###)
RUN_RHO_AVG = f'{_RUN}{NS_SEP}repair{NS_SEP}rho{NS_SEP}avg'           # run.repair.rho.avg

# Calibration
RUN_CALIB_AECE = f'{_RUN}{NS_SEP}calibration{NS_SEP}aece'
RUN_CALIB_BRIER = f'{_RUN}{NS_SEP}calibration{NS_SEP}brier'
RUN_CALIB_ECE = f'{_RUN}{NS_SEP}calibration{NS_SEP}ece'
RUN_CALIB_MAX_ECE = f'{_RUN}{NS_SEP}calibration{NS_SEP}max_ece'
RUN_CALIB_MCE = f'{_RUN}{NS_SEP}calibration{NS_SEP}mce'
RUN_CALIB_NLL = f'{_RUN}{NS_SEP}calibration{NS_SEP}nll'

# Diagnostics
RUN_DIAG_AVG_CONF = f'{_RUN}{NS_SEP}diagnostics{NS_SEP}avg_conf'
RUN_DIAG_AVG_ENTROPY = f'{_RUN}{NS_SEP}diagnostics{NS_SEP}avg_entropy'
RUN_DIAG_LOGIT_AVG_DRIFT = f'{_RUN}{NS_SEP}diagnostics{NS_SEP}logit_avg_drift'
RUN_DIAG_OUT_OF_TASK_RATE = f'{_RUN}{NS_SEP}diagnostics{NS_SEP}out_of_task_rate'

DIAG_VECTOR_KEYS = (
    RUN_DIAG_OUT_OF_TASK_RATE,
    RUN_DIAG_AVG_CONF,
    RUN_DIAG_AVG_ENTROPY,
    RUN_CALIB_ECE,
    RUN_CALIB_AECE,
    RUN_CALIB_NLL,
    RUN_DIAG_LOGIT_AVG_DRIFT,
)

# Latency metrics
RUN_LATENCY_MS_PER_SAMPLE_BASE = f'{_RUN}{NS_SEP}latency{NS_SEP}ms_per_sample{NS_SEP}base'
RUN_LATENCY_MS_PER_SAMPLE_CTRL = f'{_RUN}{NS_SEP}latency{NS_SEP}ms_per_sample{NS_SEP}ctrl'
RUN_LATENCY_MS_RATIO = f'{_RUN}{NS_SEP}latency{NS_SEP}ms_ratio'
RUN_LATENCY_SAMPLES_PER_SEC_BASE = f'{_RUN}{NS_SEP}latency{NS_SEP}samples_per_sec{NS_SEP}base'
RUN_LATENCY_SAMPLES_PER_SEC_CTRL = f'{_RUN}{NS_SEP}latency{NS_SEP}samples_per_sec{NS_SEP}ctrl'

# Repair resources
RUN_REPAIR_SECONDS = f'{_RUN}{NS_SEP}repair{NS_SEP}seconds'
RUN_REPAIR_STEPS = f'{_RUN}{NS_SEP}repair{NS_SEP}steps'

# Misc per-run
RUN_EPS = f'{_RUN}{NS_SEP}eps'

##############################################
# Cross-run metric keys (CSV / analysis)     #
##############################################
# All string values carry the ``analysis.`` prefix.

_ANA = NAMESPACE_ANALYSIS  # shorthand

ANALYSIS_ACC_FINAL_AVG_CTRL = f'{_ANA}{NS_SEP}acc{NS_SEP}final{NS_SEP}avg{NS_SEP}ctrl'
ANALYSIS_RHO_AVG = f'{_ANA}{NS_SEP}repair{NS_SEP}rho{NS_SEP}avg'

################################################
# Debug metric fragments (internal)            #
################################################
# Name fragments used to build run.debug.repair.* keys.
# Only consumed by regain/debug/; kept here for central reference.

_DEBUG_CE = 'ce'
_DEBUG_ENTROPY = 'entropy'
_DEBUG_LOGIT_L2 = 'logit_l2'
_DEBUG_NUM_CLASSES = 'num_classes'
_DEBUG_N_SAMPLES = 'n_samples'
_DEBUG_PRED_ENTROPY = 'pred_entropy'
_DEBUG_PRED_HIST = 'pred_hist'
_DEBUG_PRED_MAX_FRAC = 'pred_max_frac'
_DEBUG_PRED_UNIQUE = 'pred_unique'
_DEBUG_TOP1 = 'top1'

_DEBUG_HEALTH = 'health'
_DEBUG_HEALTH_DELTA = 'health_delta'
_DEBUG_HEALTH_D_ACC = 'd_acc'
_DEBUG_HEALTH_D_ENT = 'd_ent'
_DEBUG_HEALTH_D_MAXFRAC = 'd_maxfrac'
_DEBUG_HEALTH_D_PREDENT = 'd_predent'
_DEBUG_HEALTH_D_UNIQUE = 'd_unique'
_DEBUG_HEALTH_NEUTRAL = 'health_neutral'
_DEBUG_HEALTH_R_CE = 'r_ce'
_DEBUG_HEALTH_R_NORM = 'r_norm'
_DEBUG_HEALTH_S1_PERF = 's1_perf'
_DEBUG_HEALTH_S2_CONF = 's2_conf'
_DEBUG_HEALTH_S3_DIV = 's3_div'

####################
# MLFlow artifacts #
####################
# Note: some constants in this section are used only in one module,
#       but we keep them to keep track of all the MLflow artifacts in one place.

MLFLOW_ARTIFACT_ANALYSIS_FILE = 'analysis_artifacts.json'
MLFLOW_ARTIFACT_BACKBONE_CHECKPOINTS_DIR = 'checkpoints'
MLFLOW_ARTIFACT_CONFIG_FILE = 'config.yaml'
MLFLOW_ARTIFACT_ERROR_FILE = 'error.txt'
MLFLOW_ARTIFACT_PREDICTIONS_DIR = 'predictions'
MLFLOW_ARTIFACT_SPLITS_FILE = 'splits.tar.gz'

########
# Misc #
########

EXPERIENCE_KEY_PREFIX = 'exp'

STREAM_TRAIN = 'train'
STREAM_TEST = 'test'  # Currently unused, but kept for consistency with the rest of the streams
STREAM_REPAIR = 'repair'

STREAMS = (STREAM_TRAIN, STREAM_TEST, STREAM_REPAIR)
