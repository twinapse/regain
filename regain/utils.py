"""
General-purpose utilities shared across the regain package.
"""
from contextlib import contextmanager
import logging
import random
from typing import Iterator, Protocol, runtime_checkable, TypeVar

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import Subset

__all__ = [
    'RegainDataset',
    'cast_tensor',
    'extract_targets',
    'get_logger',
    'get_targets',
    'module_device',
    'preserve_model_mode_after_eval',
    'preserve_rng_state',
]

logging.basicConfig(
    level=logging.WARNING,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

_logger = logging.getLogger(__name__)


def get_logger() -> logging.Logger:
    return _logger


T_co = TypeVar('T_co', covariant=True)


@runtime_checkable
class RegainDataset(Protocol[T_co]):
    """
    Protocol for datasets that support `__len__` and `__getitem__`.
    """

    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int) -> T_co:
        ...


def extract_targets(dataset: RegainDataset | None) -> list[int]:
    """
    Extract integer targets from a dataset without expensive iteration when possible.

    Args:
        dataset: Dataset to inspect.

    Returns:
        List of integer class labels.
    """
    if dataset is None:
        return []

    if isinstance(dataset, Subset):
        base = dataset.dataset
        if hasattr(base, 'targets'):
            base_targets = np.asarray(base.targets)
            return base_targets[np.asarray(dataset.indices)].astype(int).tolist()

    if hasattr(dataset, 'targets'):
        return [int(target) for target in dataset.targets]

    targets: list[int] = []
    for sample in dataset:
        if isinstance(sample, tuple) and len(sample) >= 2:
            targets.append(int(sample[1]))
    return targets


def get_targets(dataset: Dataset) -> np.ndarray:
    """
    Extract class labels from a dataset.

    Args:
        dataset: Dataset exposing a `targets` attribute or returning label in index 1.

    Returns:
        Array of integer class labels.
    """
    if isinstance(dataset, Subset):
        base = dataset.dataset
        if hasattr(base, 'targets'):
            base_targets = np.asarray(base.targets)
            return base_targets[np.asarray(dataset.indices)]
    if hasattr(dataset, 'targets'):
        return np.asarray(dataset.targets)
    labels: list[int] = []
    for _, y, *_ in dataset:
        labels.append(int(y))
    return np.asarray(labels)


def module_device(module: nn.Module, fallback: str) -> torch.device:
    """
    Infer the device of a module from its first parameter.

    Args:
        module: Module to inspect.
        fallback: Fallback device string when the module has no parameters.

    Returns:
        torch.device for the module.
    """
    first_param = next(module.parameters(), None)
    return first_param.device if first_param is not None else torch.device(fallback)


def cast_tensor(*, tensor: torch.Tensor, ref_tensor: torch.Tensor) -> torch.Tensor:
    """
    Cast a tensor to match the device and dtype of a reference tensor.

    Args:
        tensor (torch.Tensor): Tensor to cast.
        ref_tensor (torch.Tensor): Reference tensor that provides device and dtype.

    Returns:
        torch.Tensor: Tensor on the same device and dtype as `ref_tensor`.
    """
    if tensor.device == ref_tensor.device and tensor.dtype == ref_tensor.dtype:
        return tensor
    return tensor.to(device=ref_tensor.device, dtype=ref_tensor.dtype)


@contextmanager
def preserve_model_mode_after_eval(model: nn.Module) -> Iterator[None]:
    """
    Context manager to preserve the training/evaluation mode of a model after temporarily setting it to evaluation mode.

    Args:
        model (nn.Module): Model to manage.

    Yields:
        None
    """
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


@contextmanager
def preserve_rng_state() -> Iterator[None]:
    """
    Preserve Python, NumPy, and Torch RNG states within a temporary seeded block.

    Yields:
        None.
    """
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            try:
                torch.cuda.set_rng_state_all(cuda_states)
            except RuntimeError:
                pass
