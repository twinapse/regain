"""
Project-wide constants.
"""

__all__ = [
    'COLUMN_B',
    'COLUMN_CONTROLLER_MODEL_PARAM_COUNT',
    'COLUMN_CONTROLLER_NAME',
    'COLUMN_END_TIME',
    'COLUMN_EXPERIMENT_ID',
    'COLUMN_EXP_IDX',
    'COLUMN_NUM_CLASSES',
    'COLUMN_PARENT_RUN_ID',
    'COLUMN_PERFORMANCE',
    'COLUMN_REPAIR_BUDGET_PER_CLASS',
    'COLUMN_REPAIR_BUDGET_TOTAL',
    'COLUMN_RUN_ID',
    'COLUMN_RUN_NAME',
    'COLUMN_SEED',
    'COLUMN_START_TIME',
    'COLUMN_STATUS',
    'COLUMN_TASK_AGE',
    'COLUMN_TOTAL_COST',
    'EXPERIENCE_KEY_PREFIX',
    'METRIC_A_CTRL',
    'METRIC_A_CTRL_MEAN',
    'METRIC_A_CTRL_MEAN_AVG',
    'METRIC_A_POST',
    'METRIC_A_POST_MEAN',
    'METRIC_A_REF',
    'METRIC_DIAG_CE',
    'METRIC_DIAG_ENTROPY',
    'METRIC_DIAG_LOGIT_L2',
    'METRIC_DIAG_NUM_CLASSES',
    'METRIC_DIAG_N_SAMPLES',
    'METRIC_DIAG_PRED_ENTROPY',
    'METRIC_DIAG_PRED_HIST',
    'METRIC_DIAG_PRED_MAX_FRAC',
    'METRIC_DIAG_PRED_UNIQUE',
    'METRIC_DIAG_TOP1',
    'METRIC_EPS',
    'METRIC_HEALTH',
    'METRIC_HEALTH_DELTA',
    'METRIC_HEALTH_D_ACC',
    'METRIC_HEALTH_D_ENT',
    'METRIC_HEALTH_D_MAXFRAC',
    'METRIC_HEALTH_D_PREDENT',
    'METRIC_HEALTH_D_UNIQUE',
    'METRIC_HEALTH_NEUTRAL',
    'METRIC_HEALTH_R_CE',
    'METRIC_HEALTH_R_NORM',
    'METRIC_HEALTH_S1_PERF',
    'METRIC_HEALTH_S2_CONF',
    'METRIC_HEALTH_S3_DIV',
    'METRIC_PREFIX_ANALYSIS',
    'METRIC_PREFIX_SUMMARY',
    'METRIC_RHO',
    'METRIC_RHO_MEAN',
    'METRIC_RHO_MEAN_AVG',
    'MLFLOW_ARTIFACT_BACKBONE_CHECKPOINTS_DIR',
    'MLFLOW_ARTIFACT_CONFIG_FILE',
    'MLFLOW_ARTIFACT_SPLITS_FILE',
    'NAMESPACE_EVAL',
    'NAMESPACE_TRAIN',
    'NS_SEP',
    'PARAM_AVALANCHE_VERSION',
    'PARAM_BACKBONE',
    'PARAM_BACKBONE_REPLAY_BATCH_SIZE_MEM',
    'PARAM_BACKBONE_REPLAY_MEM_SIZE',
    'PARAM_CONTROLLER',
    'PARAM_CONTROLLER_MODEL_PARAM_COUNT',
    'PARAM_CONTROLLER_PATH',
    'PARAM_DEBUG_SKIP_REASON',
    'PARAM_NUM_CLASSES',
    'PARAM_RUN_NAME',
    'PARAM_SCENARIO',
    'PARAM_SEED',
    'PARAM_TORCH_DETERMINISTIC_ALGORITHMS',
    'RUN_NAME_BACKBONE',
    'STREAMS',
    'STREAM_REPAIR',
    'STREAM_TEST',
    'STREAM_TRAIN',
]

##############
# Namespaces #
##############

NS_SEP = '.'  # Namespace separator

NAMESPACE_EVAL = 'eval'
NAMESPACE_TRAIN = 'train'

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
PARAM_DEBUG_SKIP_REASON = f'debug{NS_SEP}skip_reason'
PARAM_NUM_CLASSES = 'num_classes'
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
COLUMN_PARENT_RUN_ID = 'parent_run_id'
COLUMN_PERFORMANCE = 'performance'
COLUMN_REPAIR_BUDGET_PER_CLASS = 'repair_budget_per_class'
COLUMN_REPAIR_BUDGET_TOTAL = 'repair_budget_total'
COLUMN_RUN_ID = 'run_id'
COLUMN_RUN_NAME = PARAM_RUN_NAME
COLUMN_SEED = PARAM_SEED
COLUMN_START_TIME = 'start_time'
COLUMN_STATUS = 'status'
COLUMN_TASK_AGE = 'task_age'
COLUMN_TOTAL_COST = 'total_cost'

##############################
# Predefined metric key names #
##############################

METRIC_A_CTRL = 'a_ctrl'
METRIC_A_CTRL_MEAN = 'a_ctrl_mean'
METRIC_A_CTRL_MEAN_AVG = 'a_ctrl_mean_avg'
METRIC_A_POST = 'a_post'
METRIC_A_POST_MEAN = 'a_post_mean'
METRIC_A_REF = 'a_ref'
METRIC_PREFIX_ANALYSIS = f'analysis{NS_SEP}'
METRIC_PREFIX_SUMMARY = f'summary{NS_SEP}'
METRIC_RHO = 'rho'
METRIC_RHO_MEAN = 'rho_mean'
METRIC_RHO_MEAN_AVG = 'rho_mean_avg'

################################################
# Avalanche/debug metric and payload key names #
################################################

METRIC_DIAG_CE = 'ce'
METRIC_DIAG_ENTROPY = 'entropy'
METRIC_DIAG_LOGIT_L2 = 'logit_l2'
METRIC_DIAG_NUM_CLASSES = 'num_classes'
METRIC_DIAG_N_SAMPLES = 'n_samples'
METRIC_DIAG_PRED_ENTROPY = 'pred_entropy'
METRIC_DIAG_PRED_HIST = 'pred_hist'
METRIC_DIAG_PRED_MAX_FRAC = 'pred_max_frac'
METRIC_DIAG_PRED_UNIQUE = 'pred_unique'
METRIC_DIAG_TOP1 = 'top1'
METRIC_EPS = 'eps'

METRIC_HEALTH = 'health'
METRIC_HEALTH_DELTA = 'health_delta'
METRIC_HEALTH_D_ACC = 'd_acc'
METRIC_HEALTH_D_ENT = 'd_ent'
METRIC_HEALTH_D_MAXFRAC = 'd_maxfrac'
METRIC_HEALTH_D_PREDENT = 'd_predent'
METRIC_HEALTH_D_UNIQUE = 'd_unique'
METRIC_HEALTH_NEUTRAL = 'health_neutral'
METRIC_HEALTH_R_CE = 'r_ce'
METRIC_HEALTH_R_NORM = 'r_norm'
METRIC_HEALTH_S1_PERF = 's1_perf'
METRIC_HEALTH_S2_CONF = 's2_conf'
METRIC_HEALTH_S3_DIV = 's3_div'

####################
# MLFlow artifacts #
####################
# Note: some constants in this section are used only in one module,
#       but we keep them to keep track of all the MLflow artifacts in one place.

MLFLOW_ARTIFACT_BACKBONE_CHECKPOINTS_DIR = 'checkpoints'
MLFLOW_ARTIFACT_CONFIG_FILE = 'config.yaml'
MLFLOW_ARTIFACT_SPLITS_FILE = 'splits.tar.gz'

########
# Misc #
########

EXPERIENCE_KEY_PREFIX = 'exp'

STREAM_TRAIN = 'train'
STREAM_TEST = 'test'  # Currently unused, but kept for consistency with the rest of the streams
STREAM_REPAIR = 'repair'

STREAMS = (STREAM_TRAIN, STREAM_TEST, STREAM_REPAIR)
