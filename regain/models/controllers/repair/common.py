"""
Shared utilities for repair controllers.

The helpers support both ResNet-like and ViT-like backbones while keeping the public controller interfaces
generic.
"""

from abc import ABC
from abc import abstractmethod
from contextlib import contextmanager
import math
from typing import Any, Callable, Iterator, Mapping

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from regain.constants import PARAM_BACKBONE
from regain.models.controllers.base import RepairController
from regain.models.controllers.utils import resolve_backbone_or_raise
from regain.utils import get_logger
from regain.utils import module_device
from regain.utils import preserve_model_mode_after_eval

__all__ = [
    'as_batch_tensors',
    'BaseUnitGainController',
    'bounded_positive_gain',
    'build_repair_dataloader',
    'build_unit_gain_hooks',
    'build_sgd_optimizer_and_scheduler',
    'effective_gains_from_raw',
    'extract_probe_inputs',
    'fit_repair_controller',
    'log_gain_max',
    'mean_l2_distance_to_one',
    'apply_repair_correction',
    'maybe_correct_outputs',
    'model_logits',
    'prepare_repair_fit_context',
    'resolve_backbone_or_raise',
    'resolve_block_units',
    'resolve_classifier_weight',
    'resolve_stage_units',
    'run_model_with_hooks',
]

_DEFAULT_VIT_STAGE_UNITS = 5


def build_repair_dataloader(
    *,
    repair_dataset: Dataset | None,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader | None:
    """
    Build a dataloader from the repair dataset.

    Args:
        repair_dataset (Dataset | None): Repair dataset to load.
        batch_size (int): Batch size for the dataloader.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        DataLoader | None: Dataloader over repair data, or None if dataset is empty.
    """
    if repair_dataset is None:
        return None
    if len(repair_dataset) <= 0:
        return None

    return DataLoader(
        repair_dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
    )


def as_batch_tensors(batch: object, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    Extract input and target tensors from a repair dataloader batch.

    Args:
        batch (object): Dataloader batch, expected to start with `(x, y, ...)`.
        device (torch.device): Device for returned inputs.

    Returns:
        tuple[torch.Tensor, torch.Tensor] | None: Inputs and targets, or None if the batch is invalid.
    """
    try:
        x, y, *_ = batch
    except (TypeError, ValueError):
        return None
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if not torch.is_tensor(y):
        y = torch.as_tensor(y)
    return x.to(device), y.reshape(-1).to(device=device, dtype=torch.long)


def model_logits(*, model: nn.Module, inputs: torch.Tensor) -> torch.Tensor | Any:
    """
    Run the model and return its logits-like output.

    Args:
        model (nn.Module): Model to evaluate.
        inputs (torch.Tensor): Input batch.

    Returns:
        torch.Tensor | Any: Model output.
    """
    with preserve_model_mode_after_eval(model):
        with torch.inference_mode():
            return model(inputs)


def resolve_classifier_weight(model: nn.Module) -> torch.Tensor | None:
    """
    Resolve the linear classifier weight tensor from supported classifier layouts.

    Args:
        model (nn.Module): Model exposing a classifier head.

    Returns:
        torch.Tensor | None: Classifier weight shaped `(num_classes, feat_dim)`, if found.
    """
    classifier = getattr(model, 'classifier', None)
    if isinstance(classifier, nn.Linear):
        return classifier.weight.detach()
    fc = getattr(classifier, 'fc', None)
    if isinstance(fc, nn.Linear):
        return fc.weight.detach()
    return None


@contextmanager
def _frozen_model_parameters(*, model: nn.Module) -> Iterator[None]:
    """
    Temporarily freeze model parameters by disabling gradients.

    Args:
        model (nn.Module): Model whose parameters should be frozen.

    Yields:
        None.
    """
    # Track previous requires_grad state so we can restore it.
    prev_requires_grad = [bool(p.requires_grad) for p in model.parameters()]
    try:
        # Disable gradient tracking on all model parameters.
        for p in model.parameters():
            p.requires_grad = False
        yield
    finally:
        # Restore the original requires_grad flags.
        for p, req in zip(model.parameters(), prev_requires_grad, strict=False):
            p.requires_grad = bool(req)


@contextmanager
def _temporary_forward_hooks(
    *,
    hooks: list[tuple[nn.Module, Callable[[nn.Module, tuple[Any, ...], Any], Any]]],
) -> Iterator[None]:
    """
    Register forward hooks for the duration of the context.

    Args:
        hooks (list[tuple[nn.Module, Callable]]): (module, hook_fn) pairs to register.

    Yields:
        None.
    """
    handles: list[Any] = []
    try:
        # Register hooks and keep handles for later removal.
        for module, hook in hooks:
            handles.append(module.register_forward_hook(hook))
        yield
    finally:
        # Ensure hooks are cleaned up even if the forward pass fails.
        for h in handles:
            try:
                h.remove()
            except Exception:  # pylint: disable=broad-exception-caught
                get_logger().warning('Failed to remove temporary forward hook: %s', h, exc_info=True)


def run_model_with_hooks(
    *,
    model: nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
    hooks: list[tuple[nn.Module, Callable[[nn.Module, tuple[Any, ...], Any], Any]]],
    logits_error: str | None = None,
) -> Any:
    """
    Run a model forward pass under a set of temporary forward hooks.

    Args:
        model (nn.Module): Model to run.
        inputs (torch.Tensor): Input batch.
        device (torch.device): Device used to run the model.
        hooks (list[tuple[nn.Module, Callable]]): (module, hook_fn) pairs to register.
        logits_error (str | None): Optional error message if output is not a tensor.

    Returns:
        Any: Model outputs.

    Raises:
        ValueError: If `logits_error` is provided and outputs are not tensor logits.
    """
    # Run the model while the temporary hooks are active.
    with _temporary_forward_hooks(hooks=hooks):
        outputs = model(inputs.to(device))

    # Validate output shape when the caller expects logits.
    if logits_error is not None and not torch.is_tensor(outputs):
        raise ValueError(logits_error)
    return outputs


def build_unit_gain_hooks(
    *,
    units: list[tuple[str, nn.Module]],
    gains: Mapping[str, torch.Tensor],
    device: torch.device,
    hook_factory: Callable[[torch.Tensor], Callable[[nn.Module, tuple[Any, ...], Any], Any]],
) -> list[tuple[nn.Module, Callable[[nn.Module, tuple[Any, ...], Any], Any]]]:
    """
    Build forward hooks for per-unit gain tensors.

    Args:
        units (list[tuple[str, nn.Module]]): (unit_key, unit_module) pairs.
        gains (Mapping[str, torch.Tensor]): Mapping from unit keys to gain tensors.
        device (torch.device): Device for gain tensors.
        hook_factory (Callable[[torch.Tensor], Callable]]): Factory for per-unit hook functions.

    Returns:
        list[tuple[nn.Module, Callable]]: Forward hook list.
    """
    hooks: list[tuple[nn.Module, Callable[[nn.Module, tuple[Any, ...], Any], Any]]] = []
    for key, module in units:
        # Skip units that do not have a configured gain tensor.
        if key not in gains:
            continue
        # Ensure gains live on the same device as the forward pass.
        gain = gains[key]
        # Move gains to the target device when needed.
        if gain.device != device:
            gain = gain.to(device=device)
        hooks.append((module, hook_factory(gain)))
    return hooks


def bounded_positive_gain(*, raw: torch.Tensor, log_gain_max_value: float) -> torch.Tensor:
    """
    Map unconstrained raw gains to a positive bounded range around 1.0.

    Args:
        raw (torch.Tensor): Unconstrained raw parameters.
        log_gain_max_value (float): Natural log of the maximum gain (> 0).

    Returns:
        torch.Tensor: Gains in [1 / gain_max, gain_max].
    """
    return torch.exp(float(log_gain_max_value) * torch.tanh(raw))


def log_gain_max(*, gain_max: float) -> float:
    """
    Compute log(gain_max) with validation.

    Args:
        gain_max (float): Maximum gain value (> 1.0).

    Returns:
        float: log(gain_max).

    Raises:
        ValueError: If `gain_max` <= 1.0.
    """
    gain_max = float(gain_max)
    if gain_max <= 1.0:
        raise ValueError('gain_max must be > 1.0.')
    return math.log(gain_max)


def effective_gains_from_raw(
    *,
    raw_gains: Mapping[str, torch.Tensor],
    log_gain_max_value: float,
) -> dict[str, torch.Tensor]:
    """
    Convert raw gains to effective bounded gains.

    Args:
        raw_gains (Mapping[str, torch.Tensor]): Mapping from unit keys to raw gain parameters.
        log_gain_max_value (float): log(gain_max) for the bounded transform.

    Returns:
        dict[str, torch.Tensor]: Mapping from unit keys to effective gains.
    """
    return {k: bounded_positive_gain(raw=v, log_gain_max_value=log_gain_max_value) for k, v in raw_gains.items()}


def mean_l2_distance_to_one(*, gains: Mapping[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    """
    Compute the mean L2 distance to 1.0 across gain tensors.

    Args:
        gains (Mapping[str, torch.Tensor]): Mapping from unit keys to gain tensors.
        device (torch.device): Device for the returned scalar.

    Returns:
        torch.Tensor: Mean L2 distance to 1.0.
    """
    if not gains:
        # Return a scalar on the target device for empty collections.
        return torch.zeros((), device=device)

    # Compute average squared distance of each gain tensor to ones.
    terms: list[torch.Tensor] = []
    for p in gains.values():
        p = p.to(device)
        one = torch.ones_like(p)
        terms.append((p - one).pow(2).mean())
    return torch.stack(terms).mean()


def extract_probe_inputs(*, dataloader: DataLoader, device: torch.device) -> torch.Tensor | None:
    """
    Extract a representative input batch from a dataloader.

    Args:
        dataloader (DataLoader): Dataloader yielding `(x, y, ...)` batches.
        device (torch.device): Device used to move the inputs.

    Returns:
        torch.Tensor | None: Input batch moved to `device`, or None if no batches are available.
    """
    # Return the first batch as a representative probe input.
    for batch in dataloader:
        x, *_ = batch
        # Ensure inputs are tensors before moving to the device.
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        return x.to(device)
    return None


def prepare_repair_fit_context(
    *,
    controller: nn.Module,
    model: nn.Module,
    repair_dataset: Dataset | None,
    batch_size: int,
    device: str | torch.device,
    ensure_initialized_fn: Callable[[nn.Module, torch.device, torch.Tensor], None],
) -> tuple[DataLoader, torch.device, list[nn.Parameter]] | None:
    """
    Prepare the dataloader, device, and trainable params for fitting a repair controller.

    Args:
        controller (nn.Module): Controller (or module) to train.
        model (nn.Module): Model used for forward passes.
        repair_dataset (Dataset | None): Repair dataset for fitting.
        batch_size (int): Batch size for the repair dataloader.
        device (str | torch.device): Fallback device for the controller.
        ensure_initialized_fn (Callable[[nn.Module, torch.device, torch.Tensor], None]): Init hook.

    Returns:
        tuple[DataLoader, torch.device, list[nn.Parameter]] | None: Prepared context, or None if no data/params.
    """
    # Build the dataloader from the repair dataset.
    dataloader = build_repair_dataloader(
        repair_dataset=repair_dataset,
        batch_size=batch_size,
    )
    if dataloader is None:
        return None

    # Align controller to the model's device.
    model_device = module_device(model, device)
    controller.to(model_device)

    # Use one batch to initialize gain parameters and unit caches.
    probe_x = extract_probe_inputs(dataloader=dataloader, device=model_device)
    if probe_x is None:
        # Cannot initialize without a probe batch.
        return None

    with preserve_model_mode_after_eval(model):
        ensure_initialized_fn(model=model, device=model_device, sample_inputs=probe_x)

    # Train only parameters that require gradients.
    trainable_params = [p for p in controller.parameters() if p.requires_grad]
    if not trainable_params:
        return None

    return dataloader, model_device, trainable_params


def build_sgd_optimizer_and_scheduler(
    *,
    params: list[nn.Parameter],
    lr: float,
    momentum: float,
    weight_decay: float,
    lr_milestones: tuple[int, ...] | None,
    lr_gamma: float,
) -> tuple[torch.optim.Optimizer, Any | None]:
    """
    Build an SGD optimizer with an optional MultiStepLR scheduler.

    Args:
        params (list[nn.Parameter]): Parameters to optimize.
        lr (float): Learning rate.
        momentum (float): SGD momentum.
        weight_decay (float): SGD weight decay.
        lr_milestones (tuple[int, ...] | None): Optional LR schedule milestones.
        lr_gamma (float): LR decay factor used when `lr_milestones` is provided.

    Returns:
        tuple[torch.optim.Optimizer, Any | None]: (optimizer, scheduler).
    """
    # Configure SGD for controller parameters.
    optimizer = SGD(
        params,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    scheduler = None
    if lr_milestones is not None and len(lr_milestones) > 0:
        # Apply a multistep LR schedule when milestones are provided.
        scheduler = MultiStepLR(optimizer, milestones=list(lr_milestones), gamma=lr_gamma)
    return optimizer, scheduler


def fit_repair_controller(
    *,
    controller: nn.Module,
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    forward_fn: Callable[[torch.Tensor], Any],
    reg_term_fn: Callable[[Any], torch.Tensor | None] | None,
    logits_error: str,
    set_to_none: bool = True,
) -> None:
    """
    Run a standard training loop over repair data with a frozen model.

    Args:
        controller (nn.Module): Controller (or module) to train.
        model (nn.Module): Model used for forward passes.
        dataloader (DataLoader): Repair dataloader.
        device (torch.device): Device used for training.
        num_epochs (int): Number of epochs.
        optimizer (torch.optim.Optimizer): Optimizer for controller parameters.
        scheduler (Any | None): Optional learning rate scheduler.
        criterion (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): Loss criterion.
        forward_fn (Callable[[torch.Tensor], Any]): Forward function returning logits or (logits, aux).
        reg_term_fn (Callable[[Any], torch.Tensor | None] | None): Optional regularization term builder.
        logits_error (str): Error message if logits are not shaped (B, C).
        set_to_none (bool): Whether to set gradients to None in zero_grad.

    Returns:
        None.

    Raises:
        ValueError: If the model does not return logits shaped (B, C).
    """
    with _frozen_model_parameters(model=model):
        with preserve_model_mode_after_eval(model):
            # Train the controller while the backbone remains frozen.
            controller.train()
            for _ in range(int(num_epochs)):
                for batch in dataloader:
                    x, y, *_ = batch
                    # Convert input batches to tensors when needed.
                    if not torch.is_tensor(x):
                        x = torch.as_tensor(x)
                    # Convert labels to tensors when needed.
                    if not torch.is_tensor(y):
                        y = torch.as_tensor(y)

                    # Move data to the target device.
                    x = x.to(device)
                    y = y.to(device)

                    # Forward through the controller/model wrapper.
                    output = forward_fn(x)
                    # Unpack auxiliary outputs when provided.
                    if isinstance(output, tuple):
                        logits, aux = output
                    else:
                        logits, aux = output, None

                    # Validate logits and compute classification loss.
                    if not torch.is_tensor(logits) or logits.ndim != 2:
                        raise ValueError(logits_error)

                    loss = criterion(logits, y)
                    # Apply optional regularization if configured.
                    if reg_term_fn is not None:
                        # Add optional regularization term (e.g., L2 to ones).
                        reg_term = reg_term_fn(aux)
                        # Only add the term when the callback provides one.
                        if reg_term is not None:
                            loss = loss + reg_term

                    # Standard SGD step.
                    optimizer.zero_grad(set_to_none=bool(set_to_none))
                    loss.backward()
                    optimizer.step()

                if scheduler is not None:
                    # Step the LR schedule once per epoch.
                    scheduler.step()

            # Return the controller to eval mode after fitting.
            controller.eval()


def maybe_correct_outputs(
    *,
    controller: RepairController,
    outputs: Any,
    model: nn.Module | None,
    inputs: Any | None,
    device: str | torch.device,
    ensure_initialized_fn: Callable[[nn.Module, torch.device, torch.Tensor], None],
    forward_fn: Callable[[torch.Tensor, torch.device], Any],
    extract_logits_fn: Callable[[Any], Any] | None = None,
) -> Any:
    """
    Conditionally recompute outputs with a controller, falling back to original outputs if not possible.

    Args:
        controller (RepairController): Controller used to compute corrected outputs.
        outputs (Any): Raw model outputs.
        model (nn.Module | None): Model used for recomputing outputs.
        inputs (Any | None): Input batch for the forward pass.
        device (str | torch.device): Fallback device for the controller.
        ensure_initialized_fn (Callable[[nn.Module, torch.device, torch.Tensor], None]): Init hook.
        forward_fn (Callable[[torch.Tensor, torch.device], Any]): Forward function for corrected outputs.
        extract_logits_fn (Callable[[Any], Any] | None): Optional extractor for logits from forward results.

    Returns:
        Any: Corrected outputs when possible; otherwise `outputs`.
    """
    # Fast-path exits when we cannot recompute outputs.
    # Require both model and inputs to recompute outputs.
    if model is None or inputs is None:
        return outputs
    # Only support recomputation for tensor inputs.
    if not torch.is_tensor(inputs):
        return outputs

    # Ensure controller and inputs are on the same device as the model.
    model_device = module_device(model, device)
    controller.to(model_device)

    # Initialize controller parameters based on the current model/inputs.
    inputs_on_device = inputs.to(model_device)
    with preserve_model_mode_after_eval(model):
        ensure_initialized_fn(model=model, device=model_device, sample_inputs=inputs_on_device)
        with torch.inference_mode():
            # Recompute outputs using the controller-aware forward pass.
            result = forward_fn(inputs_on_device, model_device)

    if extract_logits_fn is not None:
        # Optionally strip auxiliary outputs from the controller forward.
        return extract_logits_fn(result)
    return result


def apply_repair_correction(
    *,
    controller: RepairController,
    model: nn.Module,
    inputs: torch.Tensor,
    backbone_outputs: torch.Tensor,
    train_seen_classes: set[int],
) -> torch.Tensor:
    """
    Apply controller correction while enforcing anti-cheat invariants.

    Args:
        controller (RepairController): Repair controller to apply.
        model (nn.Module): Backbone model.
        inputs (torch.Tensor): Batch inputs.
        backbone_outputs (torch.Tensor): Backbone outputs.
        train_seen_classes (set[int]): Class ids seen during training so far.

    Returns:
        torch.Tensor: Corrected outputs.
    """
    corrected_outputs = controller.correct_outputs(
        outputs=backbone_outputs,
        model=model,
        inputs=inputs,
    )
    if not (torch.is_tensor(backbone_outputs) and torch.is_tensor(corrected_outputs)):
        return corrected_outputs
    if backbone_outputs.ndim != 2 or corrected_outputs.ndim != 2:
        return corrected_outputs

    # Normalize device/dtype and validate batch compatibility.
    if int(corrected_outputs.shape[0]) != int(backbone_outputs.shape[0]):
        raise RuntimeError('Repair controller output batch dimension mismatch. '
                           f'backbone_batch={int(backbone_outputs.shape[0])}, '
                           f'controller_batch={int(corrected_outputs.shape[0])}')
    if corrected_outputs.device != backbone_outputs.device:
        corrected_outputs = corrected_outputs.to(device=backbone_outputs.device)
    if corrected_outputs.dtype != backbone_outputs.dtype:
        corrected_outputs = corrected_outputs.to(dtype=backbone_outputs.dtype)

    # Enforce anti-cheat constraints for output width and seen-class coverage.
    backbone_width = int(backbone_outputs.shape[1])
    corrected_width = int(corrected_outputs.shape[1])
    if corrected_width > backbone_width:
        raise RuntimeError('Repair controller output width exceeds backbone output width. '
                           f'backbone_width={backbone_width}, controller_width={corrected_width}')

    if train_seen_classes:
        max_seen_class = int(max(train_seen_classes))
        if max_seen_class >= backbone_width:
            raise RuntimeError('Seen class ID exceeds backbone output width. '
                               f'max_seen_class={max_seen_class}, backbone_width={backbone_width}')
        if max_seen_class >= corrected_width:
            raise RuntimeError('Repair controller output does not cover all seen classes. '
                               f'max_seen_class={max_seen_class}, controller_width={corrected_width}')

    # Merge only seen-class columns so unseen classes remain backbone-owned.
    merged_outputs = backbone_outputs.clone()
    seen_class_ids = sorted(
        cls for cls in train_seen_classes if 0 <= int(cls) < backbone_width and int(cls) < corrected_width)
    if seen_class_ids:
        merged_outputs[:, seen_class_ids] = corrected_outputs[:, seen_class_ids]
    return merged_outputs


def _is_resnet_block(module: nn.Module) -> bool:
    """
    Heuristic predicate for identifying ResNet-like residual blocks.

    Args:
        module (nn.Module): Module to test.

    Returns:
        bool: True if the module looks like a ResNet BasicBlock/Bottleneck.
    """
    # Use conv1/conv2 presence as a lightweight ResNet-block heuristic.
    conv1 = getattr(module, 'conv1', None)
    conv2 = getattr(module, 'conv2', None)
    return isinstance(conv1, nn.Conv2d) and isinstance(conv2, nn.Conv2d)


def _is_residual_stage_container(module: nn.Module) -> bool:
    """
    Identify residual-stage containers that hold ResNet-like blocks.

    Args:
        module (nn.Module): Module to test.

    Returns:
        bool: True if the module looks like a residual stage container.
    """
    if not isinstance(module, nn.Sequential):
        return False
    return any(_is_resnet_block(child) for child in module.modules())


def _resolve_residual_stage_units(*, backbone: nn.Module) -> list[tuple[str, nn.Module]]:
    """
    Resolve residual-stage units from common ResNet-like backbone layouts.

    Args:
        backbone (nn.Module): Backbone/encoder module to inspect.

    Returns:
        list[tuple[str, nn.Module]]: List of (unit_key, unit_module) pairs.
    """
    layer_names = ('layer1', 'layer2', 'layer3', 'layer4')
    layers = [getattr(backbone, name, None) for name in layer_names]
    if all(isinstance(layer, nn.Module) for layer in layers):
        return [(name, layer) for name, layer in zip(layer_names, layers)]

    features = getattr(backbone, 'features', None)
    units: list[tuple[str, nn.Module]] = []
    if isinstance(features, nn.Sequential):
        for i, child in enumerate(features):
            if _is_residual_stage_container(child):
                units.append((f'features_{i}', child))
    return units


def _resolve_vit_model_root(*, backbone: nn.Module) -> nn.Module | None:
    """
    Resolve the ViT module that exposes patch embedding and encoder blocks.

    Args:
        backbone (nn.Module): Backbone/encoder module to inspect.

    Returns:
        nn.Module | None: The ViT root module, or None when the backbone does not look ViT-like.
    """
    candidates = [backbone]
    model = getattr(backbone, 'model', None)
    if isinstance(model, nn.Module):
        candidates.append(model)

    for candidate in candidates:
        conv_proj = getattr(candidate, 'conv_proj', None)
        encoder = getattr(candidate, 'encoder', None)
        if isinstance(conv_proj, nn.Module) and isinstance(encoder, nn.Module):
            return candidate
        patch_embed = getattr(candidate, 'patch_embed', None)
        blocks = getattr(candidate, 'blocks', None)
        if (isinstance(patch_embed, nn.Module) and isinstance(blocks, (nn.Sequential, nn.ModuleList))):
            return candidate
    return None


def _resolve_vit_block_sequence(*, vit_root: nn.Module) -> list[nn.Module]:
    """
    Resolve the ordered transformer blocks for a ViT root module.

    Args:
        vit_root (nn.Module): ViT root module.

    Returns:
        list[nn.Module]: Ordered transformer blocks.
    """
    encoder = getattr(vit_root, 'encoder', None)
    layers = getattr(encoder, 'layers', None)
    if isinstance(layers, nn.Sequential):
        return list(layers)

    blocks = getattr(vit_root, 'blocks', None)
    if isinstance(blocks, (nn.Sequential, nn.ModuleList)):
        return list(blocks)
    return []


def _split_contiguous_ranges(*, num_items: int, num_groups: int) -> list[tuple[int, int]]:
    """
    Split `num_items` into near-even contiguous half-open index ranges.

    Args:
        num_items (int): Number of items to split.
        num_groups (int): Number of contiguous groups to create.

    Returns:
        list[tuple[int, int]]: List of `(start, end)` ranges with `end` exclusive.
    """
    if int(num_items) <= 0 or int(num_groups) <= 0:
        return []

    num_groups = min(int(num_groups), int(num_items))
    base = int(num_items) // int(num_groups)
    remainder = int(num_items) % int(num_groups)

    ranges: list[tuple[int, int]] = []
    start = 0
    for group_index in range(num_groups):
        group_size = base + (1 if group_index < remainder else 0)
        end = start + group_size
        ranges.append((start, end))
        start = end
    return ranges


def _resolve_vit_stage_units(
    *,
    backbone: nn.Module,
    max_units: int | None,
) -> list[tuple[str, nn.Module]]:
    """
    Resolve coarse stage-like units from common ViT backbones.

    The patch projection is kept as its own stage. Encoder stages are represented
    by contiguous groups of transformer blocks, hooked at the last block of each
    group.

    Args:
        backbone (nn.Module): Backbone/encoder module to inspect.
        max_units (int | None): Maximum number of stage units to include. None means default stage grouping.

    Returns:
        list[tuple[str, nn.Module]]: List of (unit_key, unit_module) pairs.
    """
    vit_root = _resolve_vit_model_root(backbone=backbone)
    if vit_root is None:
        return []

    units: list[tuple[str, nn.Module]] = []
    conv_proj = getattr(vit_root, 'conv_proj', None)
    if isinstance(conv_proj, nn.Module):
        units.append(('conv_proj', conv_proj))
    else:
        patch_embed = getattr(vit_root, 'patch_embed', None)
        if isinstance(patch_embed, nn.Module):
            patch_projection = getattr(patch_embed, 'proj', None)
            units.append((
                'patch_embed',
                patch_projection if isinstance(patch_projection, nn.Module) else patch_embed,
            ))

    encoder = getattr(vit_root, 'encoder', None)
    block_layers = _resolve_vit_block_sequence(vit_root=vit_root)
    if not block_layers:
        if isinstance(encoder, nn.Module):
            units.append(('encoder', encoder))
        return units

    if max_units is None or int(max_units) <= 0:
        target_total_units = min(_DEFAULT_VIT_STAGE_UNITS, len(block_layers) + 1)
    else:
        target_total_units = int(max_units)

    if target_total_units <= 1:
        return units[:1]

    num_encoder_groups = min(len(block_layers), max(target_total_units - 1, 1))
    for group_index, (_, end) in enumerate(
            _split_contiguous_ranges(num_items=len(block_layers), num_groups=num_encoder_groups),):
        # Hook the last block in each contiguous group, which is the group boundary seen by downstream blocks.
        group_last_block = block_layers[end - 1]
        units.append((f'encoder_group_{group_index}', group_last_block))

    return units


def _resolve_vit_block_units(*, backbone: nn.Module) -> list[tuple[str, nn.Module]]:
    """
    Resolve transformer block units from common ViT backbones.

    Args:
        backbone (nn.Module): Backbone/encoder module to inspect.

    Returns:
        list[tuple[str, nn.Module]]: List of (unit_key, unit_module) pairs.
    """
    vit_root = _resolve_vit_model_root(backbone=backbone)
    if vit_root is None:
        return []

    block_layers = _resolve_vit_block_sequence(vit_root=vit_root)
    if not block_layers:
        return []

    return [(f'encoder_layer_{layer_index}', layer) for layer_index, layer in enumerate(block_layers)]


def resolve_stage_units(*, backbone: nn.Module, max_units: int | None) -> list[tuple[str, nn.Module]]:
    """
    Resolve coarse stage-like units under a backbone.

    This is intentionally low-capacity: it prefers residual stages when available,
    then common ViT stages, and otherwise falls back to Conv-like containers.

    Args:
        backbone (nn.Module): Backbone/encoder module to inspect.
        max_units (int | None): Maximum number of units to include. None means all.

    Returns:
        list[tuple[str, nn.Module]]: List of (unit_key, unit_module) pairs.
    """
    units = _resolve_residual_stage_units(backbone=backbone)
    if not units:
        units = _resolve_vit_stage_units(backbone=backbone, max_units=max_units)

    if not units:
        features = getattr(backbone, 'features', None)
        # Handle backbones that expose a Sequential `features` stack.
        if isinstance(features, nn.Sequential):
            # Prefer explicit feature stacks with Conv2d content.
            for i, child in enumerate(features):
                key = f'features_{i}'
                # Accept Conv2d stems as stage-like units.
                if isinstance(child, nn.Conv2d):
                    units.append((key, child))
                    continue
                # Accept Sequential containers that include Conv2d modules.
                if isinstance(child, nn.Sequential) and any(isinstance(m, nn.Conv2d) for m in child.modules()):
                    units.append((key, child))
                    continue
        else:
            # Fall back to direct children when no `features` attribute is present.
            for name, child in backbone.named_children():
                # Treat Conv2d children as stage-like units.
                if isinstance(child, nn.Conv2d):
                    units.append((name, child))
                    continue
                # Include Sequential containers that hold Conv2d modules.
                if isinstance(child, nn.Sequential) and any(isinstance(m, nn.Conv2d) for m in child.modules()):
                    units.append((name, child))
                    continue
                # Include other modules that embed Conv2d blocks.
                if any(isinstance(m, nn.Conv2d) for m in child.modules()):
                    units.append((name, child))

    # Enforce the maximum unit budget when requested.
    if max_units is not None and max_units > 0 and len(units) > max_units:
        units = units[:max_units]

    if not units:
        # Last resort: treat the entire backbone as a single unit.
        units = [(PARAM_BACKBONE, backbone)]

    return units


def resolve_block_units(*, backbone: nn.Module, max_units: int | None) -> list[tuple[str, nn.Module]]:
    """
    Resolve finer block-like units under a backbone.

    For ResNet-like backbones, this selects residual blocks (BasicBlock/Bottleneck)
    in forward order. For ViT-like backbones, this selects encoder blocks in forward
    order. This is the proposal-aligned interpretation of "per-layer" granularity.

    Args:
        backbone (nn.Module): Backbone/encoder module to inspect.
        max_units (int | None): Maximum number of units to include. None means all.

    Returns:
        list[tuple[str, nn.Module]]: List of (unit_key, unit_module) pairs.
    """
    units: list[tuple[str, nn.Module]] = []

    stage_units = _resolve_residual_stage_units(backbone=backbone)
    if stage_units:
        # Flatten blocks within residual-stage containers.
        for stage_key, stage in stage_units:
            if not isinstance(stage, nn.Sequential):
                continue
            for bi, block in enumerate(stage):
                if _is_resnet_block(block):
                    units.append((f'{stage_key}_{bi}', block))

    if not units:
        units = _resolve_vit_block_units(backbone=backbone)

    # Fall back to layer/feature scanning when no blocks are found.
    if not units:
        # Case 1: torchvision-style ResNet backbone with layer1..layer4 attributes.
        for lname in ('layer1', 'layer2', 'layer3', 'layer4'):
            layer = getattr(backbone, lname, None)
            # Only iterate blocks for Sequential layers.
            if isinstance(layer, nn.Sequential):
                for bi, block in enumerate(layer):
                    # Keep blocks that match ResNet-like structure.
                    if _is_resnet_block(block):
                        units.append((f'{lname}_{bi}', block))

    # Case 2: wrapper backbones exposing `features` (e.g., ResNet18Backbone.features).
    # Fall back to feature-stack discovery when no blocks are found.
    if not units:
        features = getattr(backbone, 'features', None)
        # Scan Sequential feature stacks for block-like containers.
        if isinstance(features, nn.Sequential):
            for i, child in enumerate(features):
                # Skip non-sequential children in the feature stack.
                if not isinstance(child, nn.Sequential):
                    continue
                # Treat children that look like resnet layers as stage containers of blocks.
                for j, maybe_block in enumerate(child):
                    # Keep only ResNet-like blocks.
                    if _is_resnet_block(maybe_block):
                        units.append((f'features_{i}_{j}', maybe_block))

    # Respect max_units when provided.
    if max_units is not None and max_units > 0 and len(units) > max_units:
        units = units[:max_units]

    # Fallback: if we can't find blocks, fall back to stage units.
    if not units:
        units = resolve_stage_units(backbone=backbone, max_units=max_units)

    return units


class BaseUnitGainController(RepairController, ABC):
    """
    Shared implementation for gain-based repair controllers over a set of hookable units.

    Subclasses define:
      - how to resolve units (block vs stage),
      - the parameterization (scalar vs grouped channel gains),
      - the forward-hook logic.

    Args:
        lr (float): Learning rate for fitting.
        momentum (float): SGD momentum.
        weight_decay (float): SGD weight decay.
        l2_reg (float): L2 penalty strength that keeps gains close to 1.0.
        max_units (int | None): Maximum number of units (blocks/stages) to include. None means all.
        device (str | None): Device used for controller parameters and fitting.
        lr_milestones (tuple[int, ...] | None): Optional LR schedule milestones.
        lr_gamma (float): LR decay factor used when `lr_milestones` is provided.

    Returns:
        None.
    """

    def __init__(
        self,
        *,
        lr: float,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        l2_reg: float = 0.0,
        max_units: int | None,
        device: str | None = None,
        lr_milestones: tuple[int, ...] | None = None,
        lr_gamma: float = 0.1,
    ) -> None:
        super().__init__()

        # Store hyperparameters and resolve the default device.
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.l2_reg = float(l2_reg)
        self.max_units = max_units
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr_milestones = tuple(int(m) for m in lr_milestones) if lr_milestones is not None else None
        self.lr_gamma = float(lr_gamma)

        # Cached unit list and module identities (to detect changes).
        self._units: list[tuple[str, nn.Module]] = []
        self._unit_module_ids: dict[str, int] = {}
        self._unit_resolver: Callable[..., list[tuple[str, nn.Module]]] | None = None

        # Move controller parameters to the configured device.
        self.to(self.device)

    def initialize_parameters(self, *, model: nn.Module, sample_inputs: Any | None = None) -> None:
        """
        Initialize unit lists and gain parameters for the current model.

        Args:
            model (nn.Module): Model used to discover units.
            sample_inputs (Any | None): Representative input batch for probing.

        Returns:
            None.
        """
        if sample_inputs is None:
            return
        if not torch.is_tensor(sample_inputs):
            sample_inputs = torch.as_tensor(sample_inputs)

        model_device = module_device(model, self.device)
        self.to(model_device)
        inputs_on_device = sample_inputs.to(model_device)
        with preserve_model_mode_after_eval(model):
            self._ensure_initialized(model=model, device=model_device, sample_inputs=inputs_on_device)

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
        Fit gain parameters on the repair dataset.

        Args:
            model (nn.Module): Model used to compute logits during fitting.
            repair_dataset (Dataset | None): Repair dataset for fitting.
            new_classes (list[int]): Newly introduced classes (unused).
            num_epochs (int): Number of epochs used for fitting.
            batch_size (int): Batch size used for fitting.

        Returns:
            None.

        Raises:
            ValueError: If the model does not return logits shaped (B, C).
        """
        del new_classes

        # Prepare dataloader, device placement, and initial parameter setup.
        fit_context = prepare_repair_fit_context(
            controller=self,
            model=model,
            repair_dataset=repair_dataset,
            batch_size=batch_size,
            device=self.device,
            ensure_initialized_fn=self._ensure_initialized,
        )
        # Nothing to fit when there is no data or trainable params.
        if fit_context is None:
            return

        dataloader, model_device, trainable_params = fit_context

        # Build optimizer and optional scheduler for gain parameters.
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
        def _forward(x: torch.Tensor) -> torch.Tensor:
            return self._forward_with_gains(model=model, inputs=x, device=model_device)

        # Build a regularization term when requested.
        def _reg_term(aux: Any) -> torch.Tensor | None:
            del aux
            # Skip regularization when disabled.
            if self.l2_reg <= 0.0:
                return None
            return self.l2_reg * self._l2_reg_term(device=model_device)

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
            logits_error=f'{type(self).__name__} expects logits shaped (B, C).',
        )

    def correct_outputs(self, *, outputs: Any, model: nn.Module | None = None, inputs: Any | None = None) -> Any:
        """
        Recompute logits with fitted gains when enabled.

        Args:
            outputs (Any): Raw model outputs (ignored when correction is applied).
            model (nn.Module | None): Model used to recompute corrected logits.
            inputs (Any | None): Inputs for the forward pass.

        Returns:
            Any: Corrected logits when enabled and possible; otherwise the original `outputs`.
        """

        # Wrap the controller forward to match the correction helper signature.
        def _forward(x: torch.Tensor, device: torch.device) -> torch.Tensor:
            return self._forward_with_gains(model=model, inputs=x, device=device)

        # Defer to the common correction helper (handles device/init/recompute).
        return maybe_correct_outputs(
            controller=self,
            outputs=outputs,
            model=model,
            inputs=inputs,
            device=self.device,
            ensure_initialized_fn=self._ensure_initialized,
            forward_fn=_forward,
        )

    def _refresh_units(self, *, model: nn.Module) -> tuple[list[tuple[str, nn.Module]], dict[str, int]]:
        """
        Resolve units and compute identity map for change detection.

        Args:
            model (nn.Module): Model to inspect.

        Returns:
            tuple[list[tuple[str, nn.Module]], dict[str, int]]: (units, unit_module_ids).
        """
        # Resolve the unit list and record module identities for change detection.
        backbone = resolve_backbone_or_raise(model=model)
        units = self._resolve_units(backbone=backbone, max_units=self.max_units)
        unit_ids = {k: int(id(m)) for k, m in units}
        return units, unit_ids

    def _resolve_units(self, *, backbone: nn.Module, max_units: int | None) -> list[tuple[str, nn.Module]]:
        """
        Resolve the hookable unit list (block-level or stage-level).

        Args:
            backbone (nn.Module): Backbone/encoder module.
            max_units (int | None): Maximum number of units to include. None means all.

        Returns:
            list[tuple[str, nn.Module]]: List of (unit_key, unit_module) pairs.
        """
        # Require a resolver to translate backbone structure into hookable units.
        if self._unit_resolver is None:
            raise ValueError(f'{type(self).__name__} requires a unit resolver.')
        return self._unit_resolver(backbone=backbone, max_units=max_units)  # pylint: disable=not-callable

    @abstractmethod
    def _ensure_initialized(self, *, model: nn.Module, device: torch.device, sample_inputs: torch.Tensor) -> None:
        """
        Ensure unit list and parameters are initialized (and probed if needed).

        Args:
            model (nn.Module): Model used for unit discovery/probing.
            device (torch.device): Device for controller parameters and probing.
            sample_inputs (torch.Tensor): Representative input batch.

        Returns:
            None.
        """

    @abstractmethod
    def _forward_with_gains(self, *, model: nn.Module, inputs: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Forward pass with temporary gain hooks installed.

        Args:
            model (nn.Module): Model to run.
            inputs (torch.Tensor): Input batch.
            device (torch.device): Device used to run the model.

        Returns:
            torch.Tensor: Logits shaped (B, C).

        Raises:
            ValueError: If model forward does not return tensor logits.
        """

    @abstractmethod
    def _l2_reg_term(self, *, device: torch.device) -> torch.Tensor:
        """
        L2 regularization term that keeps gains close to 1.0.

        Args:
            device (torch.device): Device for the returned scalar.

        Returns:
            torch.Tensor: Scalar regularization term.
        """
