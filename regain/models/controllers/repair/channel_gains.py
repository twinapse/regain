"""
Grouped channel gain controllers.

These controllers learn per-unit multiplicative gains at the level of channel groups (e.g., groups of 32
channels). Gains are applied to unit outputs via temporary forward hooks.

The implementation includes probing/caching to keep channel counts aligned with the current backbone and to
avoid stale or mismatched parameters when unit structure changes.
"""

import math
from typing import Any, Callable

import torch
from torch import nn

from regain.models.controllers.repair.common import BaseUnitGainController
from regain.models.controllers.repair.common import bounded_positive_gain
from regain.models.controllers.repair.common import build_unit_gain_hooks
from regain.models.controllers.repair.common import mean_l2_distance_to_one
from regain.models.controllers.repair.common import resolve_block_units
from regain.models.controllers.repair.common import resolve_stage_units
from regain.models.controllers.repair.common import run_model_with_hooks
from regain.utils import preserve_model_mode_after_eval

__all__ = [
    'ChannelBlockGainController',
    'ChannelStageGainController',
]


class _GroupedChannelUnitGainController(BaseUnitGainController):
    """
    Shared implementation for grouped channel gains controllers.

    Args:
        group_size (int): Channels per group.
        unit_resolver (Callable[..., list[tuple[str, nn.Module]]]): Unit resolver for this controller.
        See `BaseUnitGainController` for other args.

    Returns:
        None.

    Raises:
        ValueError: If `group_size` is non-positive.
    """

    def __init__(
        self,
        *,
        group_size: int = 32,
        unit_resolver: Callable[..., list[tuple[str, nn.Module]]],
        gain_max: float = 2.0,
        **kwargs,
    ) -> None:
        # Validate group size before initializing.
        if int(group_size) <= 0:
            raise ValueError(f'{type(self).__name__} requires group_size > 0.')
        if float(gain_max) <= 1.0:
            raise ValueError(f'{type(self).__name__} requires gain_max > 1.0.')
        super().__init__(**kwargs)

        self.group_size = int(group_size)
        self._unit_resolver = unit_resolver
        self._raw_gains = nn.ParameterDict()
        self._log_gain_max = float(math.log(float(gain_max)))
        self._channels: dict[str, int] = {}

    def _ensure_initialized(self, *, model: nn.Module, device: torch.device, sample_inputs: torch.Tensor) -> None:
        """
        Ensure units exist, channel counts are current, and grouped-gain parameters match.

        This re-probes when:
          - unit set changes, or
          - any unit module identity changes, or
          - a unit is missing a cached channel count.

        Args:
            model (nn.Module): Model to inspect and probe.
            device (torch.device): Device for controller parameters and probing.
            sample_inputs (torch.Tensor): Representative input batch used to probe unit outputs.

        Returns:
            None.
        """
        units, unit_ids = self._refresh_units(model=model)
        keys = [k for k, _ in units]
        active_keys = set(keys)

        # Purge any stale entries from previous unit sets.
        for k in list(self._channels.keys()):
            # Drop cached channel counts for removed units.
            if k not in active_keys:
                self._channels.pop(k, None)
        for k in list(self._raw_gains.keys()):
            # Drop gain parameters for removed units.
            if k not in active_keys:
                del self._raw_gains[k]

        needs_probe = False
        # Re-probe when unit membership or module identities change.
        if set(keys) != set(self._unit_module_ids.keys()):
            needs_probe = True
        else:
            for k in keys:
                # Re-probe when channel counts are missing.
                if k not in self._channels:
                    needs_probe = True
                    break
                # Re-probe when the underlying unit module has changed.
                if int(self._unit_module_ids.get(k, -1)) != int(unit_ids.get(k, -2)):
                    needs_probe = True
                    break

        # Cache the latest units and module identity map.
        self._units = units
        self._unit_module_ids = unit_ids

        # Recompute channel counts when required.
        if needs_probe:
            # Probe the current model to refresh channel counts per unit.
            probed = self._probe_unit_channels(
                model=model,
                units=units,
                inputs=sample_inputs,
                device=device,
            )

            # Overwrite channels; un-probed keys become 0 (no stale counts).
            for key, _module in units:
                c = int(probed.get(key, 0))
                self._channels[key] = c
                if c <= 0 and key in self._raw_gains:
                    # Drop stale params if the unit didn't produce a probeable output.
                    del self._raw_gains[key]

        # Ensure parameters exist and have correct group length.
        for key, _module in units:
            c = int(self._channels.get(key, 0))
            # Skip units without a valid channel count.
            if c <= 0:
                continue

            # Compute number of channel groups for this unit.
            num_groups = (c + self.group_size - 1) // self.group_size

            # Create parameters for units that are newly added.
            if key not in self._raw_gains:
                # Initialize missing raw group gains to zeros (gain == 1.0).
                self._raw_gains[key] = nn.Parameter(torch.zeros(int(num_groups), device=device))
                continue

            cur = self._raw_gains[key]
            # Resize when the group count changes.
            if int(cur.numel()) != int(num_groups):
                # Resize while preserving the overlapping group gains.
                new_p = torch.zeros(int(num_groups), device=device, dtype=cur.dtype)
                with torch.no_grad():
                    copy_n = min(int(cur.numel()), int(num_groups))
                    new_p[:copy_n].copy_(cur.detach()[:copy_n])
                self._raw_gains[key] = nn.Parameter(new_p)

    @staticmethod
    def _probe_unit_channels(
        *,
        model: nn.Module,
        units: list[tuple[str, nn.Module]],
        inputs: torch.Tensor,
        device: torch.device,
    ) -> dict[str, int]:
        """
        Probe unit output channel sizes using a single forward pass.

        Args:
            model (nn.Module): Model to run.
            units (list[tuple[str, nn.Module]]): Units to hook.
            inputs (torch.Tensor): Representative input batch.
            device (torch.device): Device used to run the model.

        Returns:
            dict[str, int]: Mapping unit_key -> inferred channel count.
        """
        captured: dict[str, int] = {}
        hooks: list[tuple[nn.Module, Callable[[nn.Module, tuple[Any, ...], Any], Any]]] = []

        def _make_probe_hook(key: str):
            def _hook(_module: nn.Module, _inp: tuple[Any, ...], out: Any) -> Any:
                # Capture channel dimension from 2D/4D tensors only.
                if torch.is_tensor(out) and out.ndim in (2, 4) and int(out.shape[1]) > 0:
                    captured[key] = int(out.shape[1])
                return out
            return _hook

        for key, module in units:
            hooks.append((module, _make_probe_hook(key)))

        with preserve_model_mode_after_eval(model):
            with torch.inference_mode():
                # Run a single forward pass with probes attached.
                _ = run_model_with_hooks(
                    model=model,
                    inputs=inputs,
                    device=device,
                    hooks=hooks,
                )

        return captured

    def _forward_with_gains(self, *, model: nn.Module, inputs: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Apply grouped channel gains via forward hooks.

        This hook logic is robust to channel-count mismatches:
        - If the current output implies more groups than learned, missing groups default to 1.0.

        Args:
            model (nn.Module): Model to run.
            inputs (torch.Tensor): Input batch.
            device (torch.device): Device used to run the model.

        Returns:
            torch.Tensor: Logits shaped (B, C).

        Raises:
            ValueError: If model forward does not return tensor logits.
        """
        effective_gains = {
            k: bounded_positive_gain(raw=p, log_gain_max=self._log_gain_max)
            for k, p in self._raw_gains.items()
        }

        def _make_hook(gain_groups: torch.Tensor):
            def _hook(_module: nn.Module, _inp: tuple[Any, ...], out: Any) -> Any:
                # Skip non-tensor outputs or unsupported shapes.
                if not torch.is_tensor(out) or out.ndim not in (2, 4):
                    return out

                c = int(out.shape[1])
                # Skip empty channel outputs.
                if c <= 0:
                    return out

                needed_groups = (c + self.group_size - 1) // self.group_size
                if gain_groups.device != out.device or gain_groups.dtype != out.dtype:
                    g = gain_groups.to(device=out.device, dtype=out.dtype)
                else:
                    g = gain_groups

                # Pad/truncate to match the current number of channel groups.
                if int(g.numel()) < int(needed_groups):
                    pad_n = int(needed_groups) - int(g.numel())
                    pad = torch.ones(pad_n, device=out.device, dtype=g.dtype)
                    g = torch.cat([g, pad], dim=0)
                elif int(g.numel()) > int(needed_groups):
                    g = g[: int(needed_groups)]

                # Expand grouped gains across channels.
                expanded = g.repeat_interleave(self.group_size)[:c]

                # Broadcast gains across spatial dimensions when needed.
                if out.ndim == 4:
                    return out * expanded.view(1, c, 1, 1)
                return out * expanded.view(1, c)
            return _hook

        # Build hook list matching unit keys to grouped gains.
        hooks = build_unit_gain_hooks(
            units=self._units,
            gains=effective_gains,
            device=device,
            hook_factory=_make_hook,
        )

        # Run the model with temporary gain hooks applied.
        logits = run_model_with_hooks(
            model=model,
            inputs=inputs,
            device=device,
            hooks=hooks,
            logits_error=f'{type(self).__name__} expects tensor logits from the model.',
        )
        return logits

    def _l2_reg_term(self, *, device: torch.device) -> torch.Tensor:
        """
        L2 penalty that keeps grouped channel gains close to 1.0.

        Args:
            device (torch.device): Device for the returned scalar.

        Returns:
            torch.Tensor: Scalar regularization term.
        """
        effective_gains = {
            k: bounded_positive_gain(raw=p, log_gain_max=self._log_gain_max)
            for k, p in self._raw_gains.items()
        }
        return mean_l2_distance_to_one(gains=effective_gains, device=device)


class ChannelStageGainController(_GroupedChannelUnitGainController):
    """
    Grouped channel gain controller at stage granularity (coarse).

    Args:
        lr (float): Learning rate for gain fitting.
        group_size (int): Channels per group.
        momentum (float): SGD momentum.
        weight_decay (float): SGD weight_decay.
        l2_reg (float): L2 penalty strength that keeps gains close to 1.0.
        gain_max (float): Maximum gain value; gains are in [1 / gain_max, gain_max].
        max_stages (int | None): Maximum number of stages to include. None means all stages.
        device (str | None): Device used for controller parameters and fitting.
        seed (int): Random seed for dataloader shuffling.
        lr_milestones (tuple[int, ...] | None): Optional LR schedule milestones.
        lr_gamma (float): LR decay factor used when `lr_milestones` is provided.

    Returns:
        None.

    Raises:
        ValueError: If `group_size` is non-positive.
        ValueError: If `gain_max` <= 1.0.
    """

    def __init__(
        self,
        *,
        lr: float,
        group_size: int = 32,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        l2_reg: float = 0.0,
        gain_max: float = 2.0,
        max_stages: int | None = None,
        device: str | None = None,
        seed: int = 1,
        lr_milestones: tuple[int, ...] | None = None,
        lr_gamma: float = 0.1,
    ) -> None:
        super().__init__(
            lr=lr,
            group_size=group_size,
            momentum=momentum,
            weight_decay=weight_decay,
            l2_reg=l2_reg,
            gain_max=gain_max,
            max_units=max_stages,
            unit_resolver=resolve_stage_units,
            device=device,
            seed=seed,
            lr_milestones=lr_milestones,
            lr_gamma=lr_gamma,
        )


class ChannelBlockGainController(_GroupedChannelUnitGainController):
    """
    Grouped channel gain controller at block granularity (proposal-aligned per-layer).

    Args:
        lr (float): Learning rate for gain fitting.
        group_size (int): Channels per group.
        momentum (float): SGD momentum.
        weight_decay (float): SGD weight_decay.
        l2_reg (float): L2 penalty strength that keeps gains close to 1.0.
        gain_max (float): Maximum gain value; gains are in [1 / gain_max, gain_max].
        max_blocks (int | None): Maximum number of blocks to include. None means all blocks.
        device (str | None): Device used for controller parameters and fitting.
        seed (int): Random seed for dataloader shuffling.
        lr_milestones (tuple[int, ...] | None): Optional LR schedule milestones.
        lr_gamma (float): LR decay factor used when `lr_milestones` is provided.

    Returns:
        None.

    Raises:
        ValueError: If `group_size` is non-positive.
        ValueError: If `gain_max` <= 1.0.
    """

    def __init__(
        self,
        *,
        lr: float,
        group_size: int = 32,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        l2_reg: float = 0.0,
        gain_max: float = 2.0,
        max_blocks: int | None = None,
        device: str | None = None,
        seed: int = 1,
        lr_milestones: tuple[int, ...] | None = None,
        lr_gamma: float = 0.1,
    ) -> None:
        super().__init__(
            lr=lr,
            group_size=group_size,
            momentum=momentum,
            weight_decay=weight_decay,
            l2_reg=l2_reg,
            gain_max=gain_max,
            max_units=max_blocks,
            unit_resolver=resolve_block_units,
            device=device,
            seed=seed,
            lr_milestones=lr_milestones,
            lr_gamma=lr_gamma,
        )
