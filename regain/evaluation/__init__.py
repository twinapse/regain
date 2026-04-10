"""
Evaluation primitives used by the custom REGAIN evaluation loop.
"""

from regain.evaluation.calibration import CalibrationCollector
from regain.evaluation.forgetting import ForgettingTracker
from regain.evaluation.forgetting import ForwardTransferTracker
from regain.evaluation.guards import check_eval_batch
from regain.evaluation.guards import frozen_model_state
from regain.evaluation.masking import ClassMask
from regain.evaluation.predictions import PredictionRecorder
from regain.evaluation.results import derive_masked_ref_accuracy
from regain.evaluation.results import EvaluationPassResult

__all__ = [
    'CalibrationCollector',
    'check_eval_batch',
    'ClassMask',
    'derive_masked_ref_accuracy',
    'EvaluationPassResult',
    'ForgettingTracker',
    'ForwardTransferTracker',
    'frozen_model_state',
    'PredictionRecorder',
]
