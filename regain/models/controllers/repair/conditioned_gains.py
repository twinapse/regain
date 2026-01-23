"""
Input-conditioned gain controllers.

These controllers use a small gating MLP to predict per-unit gains from the current input's backbone features,
then apply the predicted gains through temporary forward hooks during a gain-conditioned forward pass.
"""

from typing import Any, Callable

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset

from regain.models.controllers.base import RepairController
from regain.models.controllers.repair.common import build_sgd_optimizer_and_scheduler
from regain.models.controllers.repair.common import fit_repair_controller
from regain.models.controllers.repair.common import maybe_correct_outputs
from regain.models.controllers.repair.common import prepare_repair_fit_context
from regain.models.controllers.repair.common import resolve_backbone_or_raise
from regain.models.controllers.repair.common import resolve_block_units
from regain.models.controllers.repair.common import resolve_stage_units
from regain.models.controllers.repair.common import run_model_with_hooks
from regain.utils import preserve_model_mode_after_eval

__all__ = [
    'InputConditionedBlockGainController',
    'InputConditionedStageGainController',
]


class _GainGatingMLP(nn.Module):
    """
    Tiny MLP that maps backbone features to per-unit scalar gains.

    The gain parameterization is bounded and centered at 1.0:
        gains = gain_max * sigmoid(raw)
    so when raw == 0, gains == gain_max / 2. With gain_max=2.0, this yields gains=1.0.
    """

    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        unit_keys: list[str],
        gain_max: float,
    ) -> None:
        """
        Initialize the gating MLP.

        Args:
            in_dim (int): Input feature dimension.
            hidden_dim (int): Hidden layer width.
            num_hidden_layers (int): Number of hidden layers (>= 0).
            unit_keys (list[str]): Ordered unit keys; output dimension is len(unit_keys).
            gain_max (float): Upper bound for gains (gains are in (0, gain_max)).

        Raises:
            ValueError: If `in_dim`, `hidden_dim`, `num_hidden_layers` or `gain_max` are invalid.
        """
        super().__init__()

        # Validate dimensions and bounds for the gating network.
        # Require a positive input feature dimension.
        if int(in_dim) <= 0:
            raise ValueError('in_dim must be > 0.')
        # Require a positive hidden dimension.
        if int(hidden_dim) <= 0:
            raise ValueError('hidden_dim must be > 0.')
        # Require a non-negative number of hidden layers.
        if int(num_hidden_layers) < 0:
            raise ValueError('num_hidden_layers must be >= 0.')
        # Require a positive gain upper bound.
        if float(gain_max) <= 0.0:
            raise ValueError('gain_max must be > 0.')

        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_hidden_layers = int(num_hidden_layers)
        self.gain_max = float(gain_max)

        self.unit_keys: list[str] = list(unit_keys)

        # Build the feature extractor and output head.
        layers: list[nn.Module] = []
        # Use a linear head directly when no hidden layers are requested.
        if self.num_hidden_layers == 0:
            self.feature = nn.Identity()
            feat_out_dim = self.in_dim
        else:
            layers.append(nn.Linear(self.in_dim, self.hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            for _ in range(self.num_hidden_layers - 1):
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                layers.append(nn.ReLU(inplace=True))
            self.feature = nn.Sequential(*layers)
            feat_out_dim = self.hidden_dim

        self.out = nn.Linear(int(feat_out_dim), len(self.unit_keys))

        # Initialize to produce raw == 0 => gains == 1 when gain_max == 2.
        with torch.no_grad():
            self.out.weight.zero_()
            self.out.bias.zero_()

    def resize_output(self, *, unit_keys: list[str]) -> None:
        """
        Resize the output layer to match a new ordered `unit_keys`, preserving overlap.

        Args:
            unit_keys (list[str]): New ordered unit keys.

        Returns:
            None.
        """
        unit_keys = list(unit_keys)
        if unit_keys == self.unit_keys:
            # No resize needed if the unit set is unchanged.
            return

        # Preserve overlapping weights/biases while resizing the output head.
        device = self.out.weight.device
        dtype = self.out.weight.dtype

        old_keys = list(self.unit_keys)
        old_w = self.out.weight.detach()
        old_b = self.out.bias.detach()

        # Allocate a new output head and copy overlapping parameters.
        new_out = nn.Linear(self.out.in_features, len(unit_keys)).to(device=device, dtype=dtype)
        with torch.no_grad():
            new_out.weight.zero_()
            new_out.bias.zero_()

            # Map old unit keys to indices for overlap copying.
            old_pos = {k: i for i, k in enumerate(old_keys)}
            for new_i, k in enumerate(unit_keys):
                j = old_pos.get(k)
                # Skip units that are new in the resized head.
                if j is None:
                    continue
                new_out.weight[new_i].copy_(old_w[j])
                new_out.bias[new_i].copy_(old_b[j])

        self.out = new_out
        self.unit_keys = unit_keys

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Compute per-unit gains from backbone feature vectors.

        Args:
            feats (torch.Tensor): Feature vectors shaped `(batch, in_dim)`.

        Returns:
            torch.Tensor: Gains shaped `(batch, num_units)`.
        """
        # Predict bounded gains with a sigmoid projection.
        h = self.feature(feats)
        raw = self.out(h)
        return self.gain_max * torch.sigmoid(raw)


class _InputConditionedUnitGainController(RepairController):
    """
    Shared implementation for input-conditioned gain controllers.

    Subclasses configure the unit resolver (stage/block) and set the public constructor signature.
    """

    def __init__(
        self,
        *,
        lr: float,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        l2_reg: float = 0.0,
        max_units: int,
        max_units_name: str,
        unit_resolver: Callable[..., list[tuple[str, nn.Module]]],
        hidden_dim: int = 128,
        num_hidden_layers: int = 1,
        gain_max: float = 2.0,
        device: str | None = None,
        seed: int = 1,
        lr_milestones: tuple[int, ...] | None = None,
        lr_gamma: float = 0.1,
    ) -> None:
        """
        Initialize the shared input-conditioned gain controller.

        Args:
            lr (float): Learning rate for fitting.
            momentum (float): SGD momentum.
            weight_decay (float): SGD weight decay.
            l2_reg (float): Strength of gain regularization toward 1.0 (applied to predicted gains).
            max_units (int): Maximum number of units to include.
            max_units_name (str): Public-facing name for `max_units` in error messages.
            unit_resolver (Callable[..., list[tuple[str, nn.Module]]]): Unit resolver for the controller.
            hidden_dim (int): Hidden width of the gating MLP.
            num_hidden_layers (int): Number of hidden layers of the gating MLP.
            gain_max (float): Maximum gain value (gains in (0, gain_max)).
            device (str | None): Device used for controller parameters and fitting.
            seed (int): Random seed.
            lr_milestones (tuple[int, ...] | None): Optional LR schedule milestones.
            lr_gamma (float): LR decay factor used when `lr_milestones` is provided.

        Raises:
            ValueError: If required hyperparameters are invalid.
        """
        super().__init__()

        # Validate required hyperparameters.
        # Require a positive learning rate.
        if float(lr) <= 0.0:
            raise ValueError('lr must be > 0.')
        # Require a positive unit budget.
        if int(max_units) <= 0:
            raise ValueError(f'{max_units_name} must be > 0.')
        # Require a positive hidden dimension.
        if int(hidden_dim) <= 0:
            raise ValueError('hidden_dim must be > 0.')
        # Require a non-negative hidden layer count.
        if int(num_hidden_layers) < 0:
            raise ValueError('num_hidden_layers must be >= 0.')
        # Require a positive gain upper bound.
        if float(gain_max) <= 0.0:
            raise ValueError('gain_max must be > 0.')

        # Store hyperparameters and configure the controller device.
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.l2_reg = float(l2_reg)
        self.max_units = int(max_units)
        self._unit_resolver = unit_resolver
        self.hidden_dim = int(hidden_dim)
        self.num_hidden_layers = int(num_hidden_layers)
        self.gain_max = float(gain_max)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = int(seed)
        self.lr_milestones = tuple(int(m) for m in lr_milestones) if lr_milestones is not None else None
        self.lr_gamma = float(lr_gamma)

        # Initialize cached units, gating network, and feature dimension.
        self._units: list[tuple[str, nn.Module]] = []
        self._unit_module_ids: dict[str, int] = {}
        self._gating: _GainGatingMLP | None = None
        self._feat_dim: int | None = None

        # Move parameters to the configured device.
        self.to(self.device)

    def fit_on_repair_data(
        self,
        *,
        model: nn.Module,
        repair_dataset: Dataset | None,
        new_classes: list[int],
        num_epochs: int,
        batch_size: int,
    ) -> None:
        """
        Fit the gating MLP using repair data.

        Args:
            model (nn.Module): Model used to compute features/logits for fitting.
            repair_dataset (Dataset | None): Repair dataset for fitting.
            new_classes (list[int]): Newly introduced classes (unused).
            num_epochs (int): Number of epochs used for fitting.
            batch_size (int): Batch size used for fitting.

        Returns:
            None.

        Raises:
            ValueError: If model forward does not return logits shaped `(B, C)`.
        """
        del new_classes

        # Prepare dataloader/device/params for fitting the gating network.
        fit_context = prepare_repair_fit_context(
            controller=self,
            model=model,
            repair_dataset=repair_dataset,
            seed=self.seed,
            batch_size=batch_size,
            device=self.device,
            ensure_initialized_fn=self._ensure_initialized,
        )
        # Skip fitting when there is no data or no trainable parameters.
        if fit_context is None:
            return

        dataloader, model_device, trainable_params = fit_context

        # Configure optimizer/scheduler and the classification loss.
        optimizer, scheduler = build_sgd_optimizer_and_scheduler(
            params=trainable_params,
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            lr_milestones=self.lr_milestones,
            lr_gamma=self.lr_gamma,
        )

        # Use cross-entropy loss on logits.
        criterion = CrossEntropyLoss()

        # Wrap the controller forward for the generic training loop.
        def _forward(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
            return self._forward_with_input_conditioning(
                model=model,
                inputs=x,
                device=model_device,
            )

        def _reg_term(aux: Any) -> torch.Tensor | None:
            gains = aux if torch.is_tensor(aux) else None
            # Skip regularization when disabled or gains are missing.
            if self.l2_reg <= 0.0 or gains is None:
                return None
            # Regularize predicted gains toward 1.0.
            one = torch.ones((), device=model_device, dtype=gains.dtype)
            return self.l2_reg * (gains - one).pow(2).mean()

        fit_repair_controller(
            controller=self,
            model=model,
            dataloader=dataloader,
            device=model_device,
            num_epochs=int(num_epochs),
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            forward_fn=_forward,
            reg_term_fn=_reg_term,
            logits_error='Expected logits shaped (B, C).',
        )

    def correct_outputs(self, *, outputs: Any, model: nn.Module | None = None, inputs: Any | None = None) -> Any:
        """
        Recompute logits with input-conditioned gains when enabled.

        Args:
            outputs (Any): Raw model outputs (ignored when correction is applied).
            model (nn.Module | None): Model used to recompute corrected logits.
            inputs (Any | None): Inputs for the forward pass.

        Returns:
            Any: Corrected logits when enabled and possible; otherwise the original `outputs`.
        """
        # Wrap the controller forward to match the correction helper signature.
        def _forward(x: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
            return self._forward_with_input_conditioning(model=model, inputs=x, device=device)

        # Extract logits when the forward returns (logits, gains).
        def _extract_logits(result: Any) -> Any:
            # Normalize to logits when the forward returns (logits, gains).
            if isinstance(result, tuple):
                return result[0]
            return result

        # Defer to the shared helper for conditional recomputation.
        return maybe_correct_outputs(
            controller=self,
            outputs=outputs,
            model=model,
            inputs=inputs,
            device=self.device,
            ensure_initialized_fn=self._ensure_initialized,
            forward_fn=_forward,
            extract_logits_fn=_extract_logits,
        )

    def _ensure_initialized(self, *, model: nn.Module, device: torch.device, sample_inputs: torch.Tensor) -> None:
        """
        Ensure units and gating MLP are initialized and compatible with the current model.

        Args:
            model (nn.Module): Model used for unit discovery/probing.
            device (torch.device): Device for controller parameters and probing.
            sample_inputs (torch.Tensor): Representative input batch.

        Returns:
            None.

        Raises:
            ValueError: If backbone features are not 2D.
        """
        backbone = resolve_backbone_or_raise(model=model)
        # Discover hookable units and cache their identities.
        units = self._unit_resolver(backbone=backbone, max_units=int(self.max_units))
        unit_ids = {k: int(id(m)) for k, m in units}

        self._units = units
        self._unit_module_ids = unit_ids

        with preserve_model_mode_after_eval(model):
            with torch.inference_mode():
                # Probe backbone features to determine the gating input dimension.
                feats = backbone(sample_inputs.to(device))
        # Validate that backbone features are 2D with nonzero width.
        if not torch.is_tensor(feats) or feats.ndim != 2 or int(feats.shape[1]) <= 0:
            raise ValueError('Expected 2D backbone features shaped (B, D).')

        # Cache the feature dimension and unit ordering for gating.
        feat_dim = int(feats.shape[1])
        unit_keys = [k for k, _m in units]

        # Decide whether to rebuild the gating network.
        needs_new = False
        # Rebuild when no gating network exists yet.
        if self._gating is None:
            needs_new = True
        elif self._feat_dim is None or int(self._feat_dim) != int(feat_dim):
            needs_new = True

        if needs_new:
            self._feat_dim = int(feat_dim)
            # Build a fresh gating MLP when feature dimensions change.
            self._gating = _GainGatingMLP(
                in_dim=int(feat_dim),
                hidden_dim=int(self.hidden_dim),
                num_hidden_layers=int(self.num_hidden_layers),
                unit_keys=list(unit_keys),
                gain_max=float(self.gain_max),
            ).to(device=device)
            return

        if self._gating is not None:
            # Keep existing weights but resize for changed unit lists.
            self._gating = self._gating.to(device=device)
            self._gating.resize_output(unit_keys=list(unit_keys))

    def _forward_with_input_conditioning(
        self,
        *,
        model: nn.Module,
        inputs: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Run a two-pass forward: features -> gains -> hooked logits.

        Args:
            model (nn.Module): Model to run.
            inputs (torch.Tensor): Input batch.
            device (torch.device): Device used to run the model.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: (logits, gains) where gains has shape `(B, U)` if available.
        """
        backbone = resolve_backbone_or_raise(model=model)
        gating = self._gating
        if gating is None:
            logits = model(inputs.to(device))
            return logits, None

        with torch.no_grad():
            # Compute backbone features without tracking gradients.
            feats = backbone(inputs.to(device))
        feats = feats.detach()

        # Predict per-unit gains from the current features.
        gains = gating(feats)

        # Build hook list for scaling each unit output.
        hooks: list[tuple[nn.Module, Callable[[nn.Module, tuple[Any, ...], Any], Any]]] = []

        def _make_hook(unit_index: int):
            def _hook(_module: nn.Module, _inp: tuple[Any, ...], out: Any) -> Any:
                # Skip non-tensor outputs.
                if not torch.is_tensor(out):
                    return out
                # Skip when batch sizes do not match the gain predictions.
                if out.shape[0] != gains.shape[0]:
                    return out

                # Broadcast the per-example gain to the unit output shape.
                g = gains[:, unit_index].to(device=out.device, dtype=out.dtype)
                # Scale spatial outputs.
                if out.ndim == 4:
                    return out * g.view(-1, 1, 1, 1)
                # Scale vector outputs.
                if out.ndim == 2:
                    return out * g.view(-1, 1)
                return out
            return _hook

        for i, (_key, module) in enumerate(self._units):
            # Guard against shorter gain vectors than unit lists.
            if i >= int(gains.shape[1]):
                break
            # Attach a hook per unit to scale its outputs.
            hooks.append((module, _make_hook(i)))

        logits = run_model_with_hooks(
            model=model,
            inputs=inputs,
            device=device,
            hooks=hooks,
        )

        return logits, gains


class InputConditionedStageGainController(_InputConditionedUnitGainController):
    """
    Input-conditioned controller: per-stage gains conditioned on the current input.

    It extracts backbone features, predicts per-stage gains with a small MLP, then re-runs the model with
    temporary forward hooks that apply those gains to each stage output.
    """

    def __init__(
        self,
        *,
        lr: float,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        l2_reg: float = 0.0,
        max_stages: int = 8,
        hidden_dim: int = 128,
        num_hidden_layers: int = 1,
        gain_max: float = 2.0,
        device: str | None = None,
        seed: int = 1,
        lr_milestones: tuple[int, ...] | None = None,
        lr_gamma: float = 0.1,
    ) -> None:
        """
        Initialize the stage-level input-conditioned gain controller.

        Args:
            lr (float): Learning rate for fitting.
            momentum (float): SGD momentum.
            weight_decay (float): SGD weight decay.
            l2_reg (float): Strength of gain regularization toward 1.0 (applied to predicted gains).
            max_stages (int): Maximum number of stages to include.
            hidden_dim (int): Hidden width of the gating MLP.
            num_hidden_layers (int): Number of hidden layers of the gating MLP.
            gain_max (float): Maximum gain value (gains in (0, gain_max)).
            device (str | None): Device used for controller parameters and fitting.
            seed (int): Random seed.
            lr_milestones (tuple[int, ...] | None): Optional LR schedule milestones.
            lr_gamma (float): LR decay factor used when `lr_milestones` is provided.

        Raises:
            ValueError: If required hyperparameters are invalid.
        """
        super().__init__(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            l2_reg=l2_reg,
            max_units=int(max_stages),
            max_units_name='max_stages',
            unit_resolver=resolve_stage_units,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            gain_max=gain_max,
            device=device,
            seed=seed,
            lr_milestones=lr_milestones,
            lr_gamma=lr_gamma,
        )
        self.max_stages = int(self.max_units)


class InputConditionedBlockGainController(_InputConditionedUnitGainController):
    """
    Input-conditioned controller: per-block gains conditioned on the current input.

    It extracts backbone features, predicts per-block gains with a small MLP, then re-runs the model with
    temporary forward hooks that apply those gains to each block output.
    """

    def __init__(
        self,
        *,
        lr: float,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        l2_reg: float = 0.0,
        max_blocks: int = 64,
        hidden_dim: int = 128,
        num_hidden_layers: int = 1,
        gain_max: float = 2.0,
        device: str | None = None,
        seed: int = 1,
        lr_milestones: tuple[int, ...] | None = None,
        lr_gamma: float = 0.1,
    ) -> None:
        """
        Initialize the block-level input-conditioned gain controller.

        Args:
            lr (float): Learning rate for fitting.
            momentum (float): SGD momentum.
            weight_decay (float): SGD weight decay.
            l2_reg (float): Strength of gain regularization toward 1.0 (applied to predicted gains).
            max_blocks (int): Maximum number of blocks to include.
            hidden_dim (int): Hidden width of the gating MLP.
            num_hidden_layers (int): Number of hidden layers of the gating MLP.
            gain_max (float): Maximum gain value (gains in (0, gain_max)).
            device (str | None): Device used for controller parameters and fitting.
            seed (int): Random seed.
            lr_milestones (tuple[int, ...] | None): Optional LR schedule milestones.
            lr_gamma (float): LR decay factor used when `lr_milestones` is provided.

        Raises:
            ValueError: If required hyperparameters are invalid.
        """
        super().__init__(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            l2_reg=l2_reg,
            max_units=int(max_blocks),
            max_units_name='max_blocks',
            unit_resolver=resolve_block_units,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            gain_max=gain_max,
            device=device,
            seed=seed,
            lr_milestones=lr_milestones,
            lr_gamma=lr_gamma,
        )
        self.max_blocks = int(self.max_units)
