"""
Learning-rate schedulers used by experiment builders.
"""

import math

import torch

__all__ = [
    'WarmupCosineLR',
]


class WarmupCosineLR(torch.optim.lr_scheduler.LRScheduler):
    """
    Epoch-based linear warmup followed by cosine decay.

    Args:
        optimizer: Optimizer whose learning rate is scheduled.
        total_epochs: Total epochs in the current experience.
        warmup_epochs: Warmup epochs with linear ramp-up.
        min_lr: Minimum learning rate reached by the cosine phase.
        last_epoch: Last completed epoch index.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        total_epochs: int,
        warmup_epochs: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.total_epochs = int(total_epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.min_lr = float(min_lr)
        if self.total_epochs <= 0:
            raise ValueError('total_epochs must be > 0.')
        if self.warmup_epochs < 0:
            raise ValueError('warmup_epochs must be >= 0.')
        if self.warmup_epochs >= self.total_epochs:
            raise ValueError('warmup_epochs must be < total_epochs.')
        if self.min_lr < 0.0:
            raise ValueError('min_lr must be >= 0.')
        super().__init__(optimizer=optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list[float]:
        """
        Compute the learning rate for the current epoch.

        Returns:
            list[float]: One learning rate per optimizer parameter group.
        """
        epoch_idx = min(max(int(self.last_epoch), 0), self.total_epochs - 1)

        if self.warmup_epochs > 0 and epoch_idx < self.warmup_epochs:
            scale = float(epoch_idx + 1) / float(self.warmup_epochs)
            return [float(base_lr) * scale for base_lr in self.base_lrs]

        if self.warmup_epochs == 0:
            cosine_epoch = epoch_idx
            cosine_total = self.total_epochs
        else:
            cosine_epoch = epoch_idx - self.warmup_epochs
            cosine_total = self.total_epochs - self.warmup_epochs

        if cosine_total <= 1:
            cosine_progress = 0.0
        else:
            cosine_progress = float(cosine_epoch) / float(cosine_total - 1)
        cosine_progress = min(max(cosine_progress, 0.0), 1.0)
        cosine_scale = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
        return [
            self.min_lr + (float(base_lr) - self.min_lr) * cosine_scale
            for base_lr in self.base_lrs
        ]
