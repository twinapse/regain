"""
Validation and frozen-state guards for evaluation passes.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn

__all__ = ['check_eval_batch', 'frozen_model_state']


@dataclass(frozen=True)
class _TensorStateSignature:
    """
    Low-overhead state signature for one parameter or buffer.

    Attributes:
        tensor_id (int): Python object identifier.
        data_ptr (int): Storage pointer.
        shape (tuple[int, ...]): Tensor shape.
        dtype (torch.dtype): Tensor dtype.
        device (torch.device): Tensor device.
        version (int): PyTorch version counter.
    """

    tensor_id: int
    data_ptr: int
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    version: int


def check_eval_batch(
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> None:
    """
    Validate the per-batch evaluation tensor contract.

    Args:
        logits (torch.Tensor): Output logits.
        targets (torch.Tensor): Integer class targets.
        num_classes (int): Expected output width.

    Raises:
        RuntimeError: If the batch contract is violated.
    """
    if not torch.is_tensor(logits):
        raise RuntimeError('Evaluation integrity violation: logits must be a tensor.')
    if logits.ndim != 2:
        raise RuntimeError('Evaluation integrity violation: logits must be 2D. '
                           f'observed_shape={tuple(logits.shape)}')
    if int(logits.shape[1]) != int(num_classes):
        raise RuntimeError('Evaluation integrity violation: logits width mismatch. '
                           f'expected={int(num_classes)}, observed={int(logits.shape[1])}')

    non_finite_count = int(torch.sum(~torch.isfinite(logits)).item())
    if non_finite_count > 0:
        raise RuntimeError('Evaluation integrity violation: logits contain non-finite values. '
                           f'non_finite_count={non_finite_count}')

    if not torch.is_tensor(targets):
        raise RuntimeError('Evaluation integrity violation: targets must be a tensor of class indices.')
    target_vector = targets.reshape(-1) if targets.ndim > 0 else targets.view(1)
    if torch.is_floating_point(target_vector) or torch.is_complex(target_vector):
        raise RuntimeError('Evaluation integrity violation: targets must use integer class indices. '
                           f'observed_dtype={targets.dtype}')
    if int(target_vector.shape[0]) != int(logits.shape[0]):
        raise RuntimeError('Evaluation integrity violation: target batch size must match logits batch size. '
                           f'logits_batch={int(logits.shape[0])}, target_batch={int(target_vector.shape[0])}')
    if target_vector.numel() <= 0:
        return

    invalid_mask = (target_vector < 0) | (target_vector >= int(num_classes))
    invalid_count = int(torch.sum(invalid_mask).item())
    if invalid_count > 0:
        min_target = int(torch.min(target_vector).item())
        max_target = int(torch.max(target_vector).item())
        raise RuntimeError('Evaluation integrity violation: target class indices are out of range. '
                           f'invalid_count={invalid_count}, target_min={min_target}, '
                           f'target_max={max_target}, num_classes={int(num_classes)}')


def _named_state_tensors(*, module: nn.Module) -> dict[str, torch.Tensor]:
    """
    Enumerate parameters and buffers for one module.

    Args:
        module (nn.Module): Module to inspect.

    Returns:
        dict[str, torch.Tensor]: Named state tensors.
    """
    named_tensors: dict[str, torch.Tensor] = {}
    for name, parameter in module.named_parameters():
        named_tensors[f'parameter:{name}'] = parameter
    for name, buffer in module.named_buffers():
        named_tensors[f'buffer:{name}'] = buffer
    return named_tensors


def _build_fast_signature(*, module: nn.Module) -> dict[str, _TensorStateSignature]:
    """
    Build a fast mutation-detection signature for one module.

    Args:
        module (nn.Module): Module to snapshot.

    Returns:
        dict[str, _TensorStateSignature]: Signature map keyed by state tensor.
    """
    signatures: dict[str, _TensorStateSignature] = {}
    for name, tensor in _named_state_tensors(module=module).items():
        signatures[name] = _TensorStateSignature(
            tensor_id=int(id(tensor)),
            data_ptr=int(tensor.data_ptr()),
            shape=tuple(int(dim) for dim in tensor.shape),
            dtype=tensor.dtype,
            device=tensor.device,
            version=int(getattr(tensor, '_version', -1)),
        )
    return signatures


def _build_exact_snapshot(*, module: nn.Module) -> dict[str, torch.Tensor]:
    """
    Build an exact CPU snapshot for one module.

    Args:
        module (nn.Module): Module to snapshot.

    Returns:
        dict[str, torch.Tensor]: Exact CPU clones.
    """
    return {name: tensor.detach().cpu().clone() for name, tensor in _named_state_tensors(module=module).items()}


def _tensors_equal_for_snapshot(
    *,
    current_value: torch.Tensor,
    baseline_value: torch.Tensor,
) -> bool:
    """
    Compare tensors exactly while treating aligned NaNs as equal.

    Args:
        current_value (torch.Tensor): Current tensor value.
        baseline_value (torch.Tensor): Baseline tensor value.

    Returns:
        bool: True when the values are unchanged under snapshot semantics.
    """
    if current_value.shape != baseline_value.shape:
        return False
    if current_value.dtype != baseline_value.dtype:
        return False

    if torch.is_floating_point(current_value) or torch.is_complex(current_value):
        return bool(torch.allclose(
            current_value,
            baseline_value,
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        ))
    return bool(torch.equal(current_value, baseline_value))


def _compute_snapshot_max_abs_delta(
    *,
    current_value: torch.Tensor,
    baseline_value: torch.Tensor,
) -> float | None:
    """
    Compute a stable max absolute delta for diagnostics.

    Args:
        current_value (torch.Tensor): Current tensor value.
        baseline_value (torch.Tensor): Baseline tensor value.

    Returns:
        float | None: Maximum absolute difference or `None` for non-numeric tensors.
    """
    if not (torch.is_floating_point(current_value) or torch.is_complex(current_value)):
        return None

    target_dtype = torch.complex128 if torch.is_complex(current_value) else torch.float64
    current_cpu = current_value.detach().to(device='cpu', dtype=target_dtype)
    baseline_cpu = baseline_value.detach().to(device='cpu', dtype=target_dtype)
    delta = torch.abs(current_cpu - baseline_cpu)
    delta = torch.nan_to_num(delta, nan=float('inf'))
    if delta.numel() <= 0:
        return 0.0
    return float(torch.max(delta).item())


@contextmanager
def frozen_model_state(
    *,
    model: nn.Module,
    controller_module: nn.Module | None = None,
) -> Iterator[None]:
    """
    Assert that model and optional controller state do not change inside the block.

    Args:
        model (nn.Module): Backbone model.
        controller_module (nn.Module | None): Optional controller module.

    Yields:
        Iterator[None]: Context manager body.

    Raises:
        RuntimeError: If parameters or buffers change.
    """
    tracked_modules: dict[str, nn.Module] = {'model': model}
    if controller_module is not None:
        tracked_modules['controller'] = controller_module

    fast_signatures = {
        module_name: _build_fast_signature(module=module) for module_name, module in tracked_modules.items()
    }
    exact_snapshots = {
        module_name: _build_exact_snapshot(module=module) for module_name, module in tracked_modules.items()
    }

    try:
        yield
    finally:
        for module_name, module in tracked_modules.items():
            current_signatures = _build_fast_signature(module=module)
            baseline_signatures = fast_signatures[module_name]

            baseline_keys = set(baseline_signatures.keys())
            current_keys = set(current_signatures.keys())
            if baseline_keys != current_keys:
                missing_keys = sorted(baseline_keys - current_keys)
                new_keys = sorted(current_keys - baseline_keys)
                raise RuntimeError('Evaluation integrity violation: state tensor membership changed during evaluation. '
                                   f'module={module_name}, missing_keys={missing_keys}, new_keys={new_keys}')

            for tensor_name, current_signature in current_signatures.items():
                baseline_signature = baseline_signatures[tensor_name]
                if current_signature == baseline_signature:
                    continue

                changed_fields: list[str] = []
                if current_signature.tensor_id != baseline_signature.tensor_id:
                    changed_fields.append('tensor_id')
                if current_signature.data_ptr != baseline_signature.data_ptr:
                    changed_fields.append('data_ptr')
                if current_signature.shape != baseline_signature.shape:
                    changed_fields.append('shape')
                if current_signature.dtype != baseline_signature.dtype:
                    changed_fields.append('dtype')
                if current_signature.device != baseline_signature.device:
                    changed_fields.append('device')
                if current_signature.version != baseline_signature.version:
                    changed_fields.append('version')

                raise RuntimeError('Evaluation integrity violation: state tensor signature changed during evaluation. '
                                   f'module={module_name}, tensor={tensor_name}, changed_fields={changed_fields}')

            current_snapshot = _build_exact_snapshot(module=module)
            baseline_snapshot = exact_snapshots[module_name]
            for tensor_name, current_value in current_snapshot.items():
                baseline_value = baseline_snapshot[tensor_name]
                if torch.equal(current_value, baseline_value):
                    continue
                if _tensors_equal_for_snapshot(
                        current_value=current_value,
                        baseline_value=baseline_value,
                ):
                    continue

                max_abs_delta = _compute_snapshot_max_abs_delta(
                    current_value=current_value,
                    baseline_value=baseline_value,
                )
                raise RuntimeError('Evaluation integrity violation: state tensor values changed during evaluation. '
                                   f'module={module_name}, tensor={tensor_name}, max_abs_delta={max_abs_delta}')
