"""
Containers and pure post-processing helpers for evaluation results.
"""

from dataclasses import dataclass
from typing import Mapping

import numpy as np

__all__ = ['derive_masked_ref_accuracy', 'EvaluationPassResult']


@dataclass(frozen=True)
class EvaluationPassResult:
    """
    Aggregated outputs from one evaluation pass.

    Attributes:
        label (str): Human-readable pass label.
        per_exp_acc (dict[int, float]): Top-1 accuracy by experience.
        per_exp_loss (dict[int, float]): Mean loss by experience.
        per_exp_logits (dict[int, np.ndarray] | None): Post-controller logits by experience.
        per_exp_backbone_logits (dict[int, np.ndarray] | None): Pre-correction backbone logits by
            experience when available.
        per_exp_targets (dict[int, np.ndarray] | None): Integer class targets by experience.
        per_exp_class_ids (dict[int, list[int]]): Sorted class ids for each experience.
        timing_ms (float): Total wall-clock pass time in milliseconds.
    """

    label: str
    per_exp_acc: dict[int, float]
    per_exp_loss: dict[int, float]
    per_exp_logits: dict[int, np.ndarray] | None
    per_exp_backbone_logits: dict[int, np.ndarray] | None
    per_exp_targets: dict[int, np.ndarray] | None
    per_exp_class_ids: dict[int, list[int]]
    timing_ms: float


def _mask_logits(
    *,
    logits: np.ndarray,
    seen_class_ids: set[int],
    mask_value: float,
) -> np.ndarray:
    """
    Apply seen-class masking to a logit array.

    Args:
        logits (np.ndarray): Logits shaped `(batch, num_classes)`.
        seen_class_ids (set[int]): Class ids that remain unmasked.
        mask_value (float): Value written into masked columns.

    Returns:
        np.ndarray: Masked logits array.
    """
    if logits.ndim != 2:
        raise ValueError('Reference-accuracy derivation requires 2D logits. '
                         f'observed_shape={tuple(logits.shape)}')

    num_classes = int(logits.shape[1])
    unseen_class_ids = [class_id for class_id in range(num_classes) if class_id not in seen_class_ids]
    if not unseen_class_ids:
        return logits

    masked_logits = np.array(logits, copy=True)
    masked_logits[:, unseen_class_ids] = float(mask_value)
    return masked_logits


def derive_masked_ref_accuracy(
    result: EvaluationPassResult,
    *,
    exp_idx: int,
    seen_class_ids: set[int] | list[int] | tuple[int, ...],
    mask_value: float = -1e9,
) -> float | None:
    """
    Derive masked current-task reference accuracy from captured logits.

    Args:
        result (EvaluationPassResult): Pass result containing logits and targets.
        exp_idx (int): Experience index to evaluate.
        seen_class_ids (set[int] | list[int] | tuple[int, ...]): Class ids that remain unmasked.
        mask_value (float): Value written into masked columns.

    Returns:
        float | None: Masked accuracy for the requested experience, or `None`
            when logits or targets are unavailable.
    """
    if result.per_exp_targets is None:
        return None

    targets = result.per_exp_targets.get(int(exp_idx))
    if targets is None:
        return None

    logits_map: Mapping[int, np.ndarray] | None = result.per_exp_backbone_logits
    if logits_map is None:
        logits_map = result.per_exp_logits
    if logits_map is None:
        return None

    logits = logits_map.get(int(exp_idx))
    if logits is None:
        return None

    targets_array = np.asarray(targets, dtype=np.int64).reshape(-1)
    if targets_array.size <= 0:
        return None
    if logits.shape[0] != targets_array.shape[0]:
        raise ValueError('Logits and targets must have the same batch dimension. '
                         f'logits_batch={int(logits.shape[0])}, targets_batch={int(targets_array.shape[0])}')

    seen_set = {int(class_id) for class_id in seen_class_ids}
    masked_logits = _mask_logits(
        logits=np.asarray(logits, dtype=np.float32),
        seen_class_ids=seen_set,
        mask_value=float(mask_value),
    )
    predictions = np.argmax(masked_logits, axis=1)
    return float(np.mean(predictions == targets_array))
