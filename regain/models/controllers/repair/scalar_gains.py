"""
Static unit-level gain controllers.

These controllers learn one scalar gain per resolved backbone unit (e.g., stage-level or block-level units for
ResNet-like backbones). Gains are applied multiplicatively to unit outputs via temporary forward hooks.

This provides a compact intermediate-capacity controller between logit calibration and per-channel gains.
"""

from typing import Any, Callable

import torch
from torch import nn

from regain.models.controllers.repair.common import BaseUnitGainController
from regain.models.controllers.repair.common import build_unit_gain_hooks
from regain.models.controllers.repair.common import mean_l2_distance_to_one
from regain.models.controllers.repair.common import resolve_block_units
from regain.models.controllers.repair.common import resolve_stage_units
from regain.models.controllers.repair.common import run_model_with_hooks

__all__ = [
    'ScalarBlockGainController',
    'ScalarStageGainController',
]


class _ScalarUnitGainController(BaseUnitGainController):
    """
    Shared implementation for scalar gain controllers.

    Args:
        unit_resolver (Callable[..., list[tuple[str, nn.Module]]]): Unit resolver for this controller.
        See `BaseUnitGainController`.

    Returns:
        None.
    """

    def __init__(
        self,
        *,
        unit_resolver: Callable[..., list[tuple[str, nn.Module]]],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # Initialize gain parameters and resolver.
        self._gains = nn.ParameterDict()
        self._unit_resolver = unit_resolver

    def _ensure_initialized(self, *, model: nn.Module, device: torch.device, sample_inputs: torch.Tensor) -> None:
        """
        Ensure units and scalar gains exist.

        Args:
            model (nn.Module): Model to inspect.
            device (torch.device): Device for controller parameters.
            sample_inputs (torch.Tensor): Representative input batch (unused).

        Returns:
            None.
        """
        del sample_inputs

        units, unit_ids = self._refresh_units(model=model)
        active_keys = {k for k, _ in units}

        # Purge stale entries from previous unit sets.
        for k in list(self._gains.keys()):
            # Remove gains for units that no longer exist.
            if k not in active_keys:
                del self._gains[k]

        # Cache the latest units and module identity map.
        self._units = units
        self._unit_module_ids = unit_ids

        for key, _module in units:
            # Initialize gains for newly discovered units.
            if key not in self._gains:
                # Initialize missing scalar gains to ones.
                self._gains[key] = nn.Parameter(torch.ones((), device=device))

    def _forward_with_gains(self, *, model: nn.Module, inputs: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Apply scalar gains to each unit output via forward hooks.

        Args:
            model (nn.Module): Model to run.
            inputs (torch.Tensor): Input batch.
            device (torch.device): Device used to run the model.

        Returns:
            torch.Tensor: Logits shaped (B, C).

        Raises:
            ValueError: If model forward does not return tensor logits.
        """
        def _make_hook(gain: torch.Tensor):
            def _hook(_module: nn.Module, _inp: tuple[Any, ...], out: Any) -> Any:
                # Skip non-tensor outputs.
                if not torch.is_tensor(out):
                    return out
                # Apply the scalar gain to any tensor output.
                return out * gain
            return _hook

        # Build hook list for each unit gain parameter.
        hooks = build_unit_gain_hooks(
            units=self._units,
            gains=self._gains,
            device=device,
            hook_factory=_make_hook,
        )

        # Run the model with scalar gain hooks attached.
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
        L2 penalty that keeps scalar gains close to 1.0.

        Args:
            device (torch.device): Device for the returned scalar.

        Returns:
            torch.Tensor: Scalar regularization term.
        """
        return mean_l2_distance_to_one(gains=self._gains, device=device)


class ScalarStageGainController(_ScalarUnitGainController):
    """
    Scalar gain controller at stage granularity (coarse, ultra-low capacity).

    Args:
        lr (float): Learning rate for gain fitting.
        momentum (float): SGD momentum.
        weight_decay (float): SGD weight decay.
        l2_reg (float): L2 penalty strength that keeps gains close to 1.0.
        max_stages (int | None): Maximum number of stages to include. None means all stages.
        device (str | None): Device used for controller parameters and fitting.
        seed (int): Random seed for dataloader shuffling.
        lr_milestones (tuple[int, ...] | None): Optional LR schedule milestones.
        lr_gamma (float): LR decay factor used when `lr_milestones` is provided.

    Returns:
        None.

    Raises:
        ValueError: If required hyperparameters are invalid.
    """

    def __init__(
        self,
        *,
        lr: float,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        l2_reg: float = 0.0,
        max_stages: int | None = None,
        device: str | None = None,
        seed: int = 1,
        lr_milestones: tuple[int, ...] | None = None,
        lr_gamma: float = 0.1,
    ) -> None:
        super().__init__(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            l2_reg=l2_reg,
            max_units=max_stages,
            unit_resolver=resolve_stage_units,
            device=device,
            seed=seed,
            lr_milestones=lr_milestones,
            lr_gamma=lr_gamma,
        )


class ScalarBlockGainController(_ScalarUnitGainController):
    """
    Scalar gain controller at block granularity (proposal-aligned per-layer).

    Args:
        lr (float): Learning rate for gain fitting.
        momentum (float): SGD momentum.
        weight_decay (float): SGD weight decay.
        l2_reg (float): L2 penalty strength that keeps gains close to 1.0.
        max_blocks (int | None): Maximum number of blocks to include. None means all blocks.
        device (str | None): Device used for controller parameters and fitting.
        seed (int): Random seed for dataloader shuffling.
        lr_milestones (tuple[int, ...] | None): Optional LR schedule milestones.
        lr_gamma (float): LR decay factor used when `lr_milestones` is provided.

    Returns:
        None.

    Raises:
        ValueError: If required hyperparameters are invalid.
    """

    def __init__(
        self,
        *,
        lr: float,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        l2_reg: float = 0.0,
        max_blocks: int | None = None,
        device: str | None = None,
        seed: int = 1,
        lr_milestones: tuple[int, ...] | None = None,
        lr_gamma: float = 0.1,
    ) -> None:
        super().__init__(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            l2_reg=l2_reg,
            max_units=max_blocks,
            unit_resolver=resolve_block_units,
            device=device,
            seed=seed,
            lr_milestones=lr_milestones,
            lr_gamma=lr_gamma,
        )
