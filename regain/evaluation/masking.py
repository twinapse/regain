"""
Class masking helpers for evaluation passes.
"""

from dataclasses import dataclass
from typing import Iterable

import torch

__all__ = ['ClassMask']


@dataclass(frozen=True)
class ClassMask:
    """
    Value object describing the seen-class masking policy for one pass.

    Attributes:
        seen_class_ids (frozenset[int]): Class ids that remain unmasked.
        mask_value (float): Value written into masked logit columns.
    """

    seen_class_ids: frozenset[int]
    mask_value: float = -1e9

    @classmethod
    def from_seen_classes(
        cls,
        seen_class_ids: Iterable[int],
        *,
        mask_value: float = -1e9,
    ) -> 'ClassMask':
        """
        Build a mask from any iterable of seen class ids.

        Args:
            seen_class_ids (Iterable[int]): Seen class ids.
            mask_value (float): Value written into masked columns.

        Returns:
            ClassMask: Resolved mask value object.
        """
        return cls(
            seen_class_ids=frozenset(int(class_id) for class_id in seen_class_ids),
            mask_value=float(mask_value),
        )

    def apply(self, *, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply the unseen-class mask to a 2D logit tensor.

        Args:
            logits (torch.Tensor): Batch logits shaped `(batch, num_classes)`.

        Returns:
            torch.Tensor: Masked logits. When nothing is masked, the input tensor
                is returned unchanged.
        """
        if logits.ndim != 2:
            raise ValueError(
                'Class masking requires a 2D logits tensor. '
                f'observed_shape={tuple(logits.shape)}'
            )

        num_classes = int(logits.shape[1])
        unseen_class_ids = [
            class_id
            for class_id in range(num_classes)
            if class_id not in self.seen_class_ids
        ]
        if not unseen_class_ids:
            return logits

        masked_logits = logits.clone()
        masked_logits[:, unseen_class_ids] = float(self.mask_value)
        return masked_logits

    @property
    def seen(self) -> list[int]:
        """
        Return the sorted seen-class ids.

        Returns:
            list[int]: Sorted seen-class ids.
        """
        return sorted(int(class_id) for class_id in self.seen_class_ids)
