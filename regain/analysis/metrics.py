"""
Metrics for evaluating retrieval-based repair effectiveness.
"""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable, List, Optional, Tuple

from regain.constants import NAMESPACE_EVAL
from regain.constants import NAMESPACE_TRAIN

__all__ = [
    'MetricContext',
    'MetricPhase',
    'mean_ignore_invalid',
    'retrieval_correctable_fraction',
    'retrieval_correctable_fractions',
]


class MetricPhase(Enum):
    TRAIN = NAMESPACE_TRAIN
    EVAL = NAMESPACE_EVAL


@dataclass
class MetricContext:
    """
    Contextual information for metric computation during continual learning.

    Attributes:
        phase (MetricPhase): Current phase ('train' or 'eval').
        experience_index (int): Current experience index.
        epoch_index (int): Current epoch index within the experience.
        train_step (int): Current training step (only advanced by training).
        log_step (int): Current step to use when logging.
        log_namespace (str): Namespace to prefix metric names for logging.
        log_enabled (bool): Whether logging is enabled.
    """
    phase: MetricPhase = MetricPhase.TRAIN
    experience_index: int = 0
    epoch_index: int = 0
    train_step: int = 0
    log_step: int = 0
    log_namespace: str = NAMESPACE_TRAIN
    log_enabled: bool = False

    _epoch_in_experience: int = 0  # Internal counter for deterministic epoch tracking

    def set_phase(self, phase: MetricPhase) -> None:
        self.phase = phase

    def set_experience(self, exp_idx: int) -> None:
        self.experience_index = int(max(0, exp_idx))
        self.epoch_index = 0

    def set_epoch(self, epoch_idx: int) -> None:
        self.epoch_index = int(max(0, epoch_idx))

    def set_train_step(self, step: int) -> None:
        self.train_step = int(max(0, step))

    def set_log_step(self, step: int) -> None:
        self.log_step = int(max(0, step))

    def set_log_namespace(self, name: str) -> None:
        self.log_namespace = str(name)

    def set_log_enabled(self, enabled: bool) -> None:
        self.log_enabled = enabled

    def reset_training_counters(self) -> None:
        """
        Reset counters used for deterministic training step and epoch tracking at training start.

        Returns:
            None.
        """
        self._epoch_in_experience = 0
        self.set_epoch(0)
        self.set_train_step(0)

    def reset_experience_counters(self) -> None:
        """
        Reset epoch counters at the start of a new experience.

        Returns:
            None.
        """
        self._epoch_in_experience = 0
        self.set_epoch(0)

    def advance_training_epoch(self) -> None:
        """
        Advance counters for a new training epoch deterministically, updating the public
        epoch index and then incrementing the internal epoch counter.

        Returns:
            None.
        """
        self.set_epoch(self._epoch_in_experience)
        self.set_train_step(self.train_step + 1)
        self._epoch_in_experience += 1


def retrieval_correctable_fraction(
    a_ref: float,
    a_post: float,
    a_ctrl: float,
    eps: float = 1e-6,
) -> Optional[float]:
    """
    Compute the retrieval-correctable fraction for a single task.

    Args:
        a_ref: Accuracy immediately after training the task.
        a_post: Accuracy after completing all subsequent tasks (forgetting applied).
        a_ctrl: Accuracy after applying the retrieval-based controller.
        eps: Minimum magnitude of total forgetting to consider the task valid.

    Returns:
        Retrieval-correctable fraction; None when total forgetting is negligible.

    Notes:
        Definitions (per proposal):

        - F_total = A_ref - A_post
        - F_res = A_ref - A_ctrl
        - ρ = (F_total - F_res) / F_total

        Tasks with non-positive or negligible forgetting (F_total <= eps) return None.
    """
    f_total = a_ref - a_post
    if f_total <= eps:
        return None
    f_res = a_ref - a_ctrl
    return (f_total - f_res) / f_total


def retrieval_correctable_fractions(
    triples: Iterable[Tuple[float, float, float]],
    eps: float = 1e-6,
) -> List[Optional[float]]:
    """
    Vectorized helper over (a_ref, a_post, a_ctrl) triples.

    Args:
        triples: Iterable of accuracy triples per task.
        eps: Minimum magnitude of total forgetting to consider a task valid.

    Returns:
        List of retrieval-correctable fractions, one per input triple.
    """
    return [retrieval_correctable_fraction(a_ref, a_post, a_ctrl, eps) for a_ref, a_post, a_ctrl in triples]


def mean_ignore_invalid(values: Iterable[Optional[float]]) -> Optional[float]:
    """
    Compute the mean of non-None values, returning None when no valid entries exist.

    Args:
        values: Iterable of optional float values.

    Returns:
        Mean of valid values or None when no valid values are present.
    """
    valid_values = [
        v for v in values
        if v is not None and isinstance(v, (int, float)) and math.isfinite(float(v))
    ]
    if not valid_values:
        return None
    return sum(valid_values) / len(valid_values)
