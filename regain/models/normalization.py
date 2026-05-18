"""
Normalization layers and utilities.
"""

from collections.abc import Callable
import math

import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

__all__ = [
    'replace_batchnorm2d',
    'ContinualNormalization4',
    'ContinualNormalization8',
    'ContinualNormalization16',
    'ContinualNormalization32',
    'ContinualNormalization64',
    'TaskBalancedBatchNorm',
]

#############
# Utilities #
#############


def replace_batchnorm2d(module: nn.Module, nl: Callable[[nn.BatchNorm2d], nn.Module]) -> None:
    """
    Replace BatchNorm2d modules in-place with a provided normalization layer factory.

    The traversal matches the reference implementation by inspecting both direct attributes
    and named children recursively.

    Args:
        module: Root module to rewrite.
        nl: Callable that takes a `BatchNorm2d` instance and returns a replacement module.

    Returns:
        None.
    """
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str, None)
        if target_attr.__class__ is nn.BatchNorm2d:
            new_bn = nl(target_attr)
            setattr(module, attr_str, new_bn)

    for name, child in module.named_children():
        if child.__class__ is nn.BatchNorm2d:
            new_bn = nl(child)
            setattr(module, name, new_bn)
        replace_batchnorm2d(child, nl)


################################
# Continual Normalization (CN) #
################################


class _ContinualNormalization(_BatchNorm):
    """
    Base Continual Normalization (CN) layer wrapping an existing `BatchNorm2d` module.
    CN replaces `BatchNorm2d` with a composition: `GroupNorm (no affine) -> BatchNorm (with affine + running stats)`.
    Subclasses represent CN variants differing by the number of groups used for the `GroupNorm` stage.

    Args:
        target: `BatchNorm2d` module to wrap.
        eps: Numerical stability constant.
        momentum: Running-stat momentum.
        affine: Unused; kept for API consistency.
    """

    def __init__(
            self,
            target: nn.BatchNorm2d,
            eps: float = 1e-5,
            momentum: float = 0.1,
            affine: bool = True,  # Keep for API consistency  # pylint: disable=unused-argument
    ) -> None:
        num_features = int(target.num_features)
        super().__init__(num_features=num_features, eps=eps, momentum=momentum, affine=True)

        self.running_mean = target.running_mean
        self.running_var = target.running_var
        self.weight = target.weight
        self.bias = target.bias

        # `num_features_internal` stores the wrapped BN channel count for CN bookkeeping.
        self.num_features_internal = num_features
        # `set_num_groups` is the descriptive replacement for original notation `setG`.
        # Implementations set `self.num_groups`, which is consumed by GroupNorm in `forward`.
        self.set_num_groups()

    def set_num_groups(self) -> None:
        """Set the number of groups for the GroupNorm stage."""
        raise NotImplementedError

    def _check_input_dim(
            self,
            input: torch.Tensor,  # pylint: disable=redefined-builtin
    ) -> None:
        del input

    def forward(
            self,
            input: torch.Tensor,  # pylint: disable=redefined-builtin
    ) -> torch.Tensor:
        """
        Apply CN: GroupNorm without affine, followed by BatchNorm with affine.

        Args:
            input (torch.Tensor): Feature map shaped `(B, C, H, W)`.

        Returns:
            torch.Tensor: Normalized feature map.
        """
        out_gn = F.group_norm(input, self.num_groups, None, None, self.eps)
        out = F.batch_norm(
            out_gn,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training,
            self.momentum,
            self.eps,
        )
        return out


class ContinualNormalization4(_ContinualNormalization):
    """CN variant with 4 GroupNorm groups."""

    def set_num_groups(self) -> None:
        self.num_groups = 4


class ContinualNormalization8(_ContinualNormalization):
    """CN variant with 8 GroupNorm groups."""

    def set_num_groups(self) -> None:
        self.num_groups = 8


class ContinualNormalization16(_ContinualNormalization):
    """CN variant with 16 GroupNorm groups."""

    def set_num_groups(self) -> None:
        self.num_groups = 16


class ContinualNormalization32(_ContinualNormalization):
    """CN variant with 32 GroupNorm groups."""

    def set_num_groups(self) -> None:
        self.num_groups = 32


class ContinualNormalization64(_ContinualNormalization):
    """CN variant with 64 GroupNorm groups."""

    def set_num_groups(self) -> None:
        self.num_groups = 64


############################################
# Task-Balanced Batch Normalization (TBBN) #
############################################


class TaskBalancedBatchNorm(nn.BatchNorm2d):
    """
    Task-Balanced Batch Normalization (TBBN), matching the official implementation.

    Notes:
        - This layer assumes training minibatches are formed as:
          [current-task samples | replay/memory samples] with a fixed, integer batch ratio.
        - `current_batch_size` is the expected current-task minibatch size; `replay_batch_size` is the
          expected replay minibatch size. The ratio is derived as `current_batch_size // replay_batch_size`
          and must satisfy `current_batch_size == ratio * replay_batch_size`.
        - Call `set_number_of_task(t)` (0-indexed) at the beginning of each task/experience.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        batch_ratio: int | None = None,
        current_batch_size: int = 48,
        replay_batch_size: int = 16,
    ) -> None:
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

        # The forward pass assumes every training minibatch follows the [current-task | replay] split.
        # `current_batch_size` is the expected current-task minibatch size (`B_c` in TBBN original notation).
        self.current_batch_size = int(current_batch_size)
        # `replay_batch_size` is the expected replay minibatch size (`B_p` in TBBN original notation).
        self.replay_batch_size = int(replay_batch_size)
        if self.current_batch_size <= 0 or self.replay_batch_size <= 0:
            raise ValueError('current_batch_size and replay_batch_size must be positive integers.')

        implied_ratio = self.current_batch_size // self.replay_batch_size
        if implied_ratio * self.replay_batch_size != self.current_batch_size:
            raise ValueError('current_batch_size must be an integer multiple of replay_batch_size for TBBN.')

        if batch_ratio is None:
            self.batch_ratio = implied_ratio
        else:
            self.batch_ratio = int(batch_ratio)
            if self.batch_ratio <= 0:
                raise ValueError('batch_ratio must be a positive integer.')
            if self.batch_ratio != implied_ratio:
                raise ValueError(f'Inconsistent batch partition: batch_ratio={self.batch_ratio}, '
                                 f'but current_batch_size={self.current_batch_size} and '
                                 f'replay_batch_size={self.replay_batch_size} imply {implied_ratio}.')

        # The official code expects this to be set via `set_number_of_task`.
        # We default to 0 so the layer is usable even if the hook is missed.
        # `task_index` is the 0-based experience id (`T` in TBBN original notation).
        self.task_index: int = 0

    def set_number_of_task(self, task_index: int) -> None:
        """
        Set the task number (0-indexed).

        Args:
            task_index: Task index (0-indexed).
        """
        self.task_index = int(task_index)

    def forward(
            self,
            input: torch.Tensor,  # pylint: disable=redefined-builtin
    ) -> torch.Tensor:
        """
        Forward pass for TBBN.

        Args:
            input (torch.Tensor): Feature map shaped `(B, C, H, W)`.

        Returns:
            torch.Tensor: Normalized feature map shaped `(B, C, H, W)`.
        """
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = float(self.momentum)

        # If task_index == 0: "general BN" branch (as in official code).
        if self.task_index == 0:
            splits = 1
            if self.training:
                running_mean_split = self.running_mean.repeat(splits)
                running_var_split = self.running_var.repeat(splits)

                batch_size, num_channels, height, width = input.shape
                reshaped = input.view(-1, num_channels * splits, height, width)

                mean = reshaped.mean([0, 2, 3])
                var = reshaped.var([0, 2, 3], unbiased=False)

                n = reshaped.numel() / reshaped.size(1)

                with torch.no_grad():
                    running_mean_split = (exponential_average_factor * mean +
                                          (1 - exponential_average_factor) * running_mean_split)
                    running_var_split = (exponential_average_factor * var * n / (n - 1) +
                                         (1 - exponential_average_factor) * running_var_split)

                reshaped = (reshaped - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)
                if self.affine:
                    reshaped = (reshaped * self.weight.repeat(splits)[None, :, None, None] +
                                self.bias.repeat(splits)[None, :, None, None])

                out = reshaped.view(batch_size, num_channels, height, width)

                with torch.no_grad():
                    self.running_mean.copy_(running_mean_split.view(splits, num_channels).mean(dim=0))
                    self.running_var.copy_(running_var_split.view(splits, num_channels).mean(dim=0))

                return out

            mean = self.running_mean
            var = self.running_var
            out = (input - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)
            if self.affine:
                out = out * self.weight[None, :, None, None] + self.bias[None, :, None, None]
            return out

        if self.training:
            batch_size, num_channels, height, width = input.shape
            mem_count = self.replay_batch_size
            curr_count = batch_size - mem_count

            if mem_count <= 0 or curr_count <= 0:
                raise ValueError(f'TaskBalancedBatchNorm expects minibatches arranged as [current | replay] '
                                 f'with replay_batch_size={self.replay_batch_size}, got batch size {batch_size}.')

            mem_batch = input[curr_count:, :, :, :]
            if mem_batch.shape[0] != mem_count:
                raise ValueError(f'TaskBalancedBatchNorm expects minibatches arranged as [current | replay] '
                                 f'with replay_batch_size={self.replay_batch_size}, '
                                 f'got replay slice of size {mem_batch.shape[0]}.')

            r = self.batch_ratio * self.task_index
            splits = max(1, math.gcd(int(curr_count), int(r)))

            running_mean_split = self.running_mean.repeat(splits)
            running_var_split = self.running_var.repeat(splits)

            curr_batch = input[:curr_count, :, :, :]
            mem_batch = input[curr_count:, :, :, :]

            curr_batch = curr_batch.view(-1, num_channels * splits, height, width)
            mem_batch_repeat = mem_batch.repeat(1, splits, 1, 1)

            concat_batch = torch.cat([curr_batch, mem_batch_repeat], dim=0)

            repeat_mean = concat_batch.mean([0, 2, 3])
            repeat_var = concat_batch.var([0, 2, 3], unbiased=False)

            n = concat_batch.numel() / concat_batch.size(1)

            with torch.no_grad():
                running_mean_split = (exponential_average_factor * repeat_mean +
                                      (1 - exponential_average_factor) * running_mean_split)
                running_var_split = (exponential_average_factor * repeat_var * n / (n - 1) +
                                     (1 - exponential_average_factor) * running_var_split)

                self.running_mean.copy_(running_mean_split.view(splits, num_channels).mean(dim=0))
                self.running_var.copy_(running_var_split.view(splits, num_channels).mean(dim=0))

            concat_batch = (concat_batch -
                            repeat_mean[None, :, None, None]) / torch.sqrt(repeat_var[None, :, None, None] + self.eps)
            if self.affine:
                concat_batch = (concat_batch * self.weight.repeat(splits)[None, :, None, None] +
                                self.bias.repeat(splits)[None, :, None, None])

            reshaped_curr_batch = concat_batch[:curr_batch.shape[0]].view(-1, num_channels, height, width)
            reshaped_mem_batch = torch.mean(
                concat_batch[curr_batch.shape[0]:].view(-1, splits, num_channels, height, width),
                dim=1,
            )

            return torch.cat([reshaped_curr_batch, reshaped_mem_batch], dim=0)

        mean = self.running_mean
        var = self.running_var

        out = (input - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)
        if self.affine:
            out = out * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return out
