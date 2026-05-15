"""
Experiment-level utility helpers for determinism, backbone, and controller resolution.
"""
import inspect
from typing import Any

import torch
from torch import nn

__all__ = [
    'enable_determinism',
    'resolve_backbone_training_config',
    'resolve_controller_type',
    'resolve_avalanche_eval_every',
    'to_scalar',
    'extract_scalar_metrics',
    'count_parameters',
]

from regain.experiments.config import ControllerConfig
from regain.experiments.config import ExperimentConfig
from regain.experiments.config import OptimizerConfig
from regain.experiments.config import StrategyConfig
from regain.experiments.config import TrainingConfig
from regain.models.controllers import PreventionController
from regain.models.controllers import RepairController
from regain.registry import get_controller_path
from regain.registry import import_symbol


def enable_determinism() -> bool:
    """
    Enable deterministic PyTorch behavior where supported.

    Returns:
        True if deterministic algorithms were fully enabled, False otherwise.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        return False
    return True


def resolve_backbone_training_config(
    *,
    experiment_config: ExperimentConfig,
    use_backbone_checkpoints: bool,
) -> TrainingConfig:
    """
    Resolve backbone training settings for strategy construction.

    Args:
        experiment_config (ExperimentConfig): Experiment configuration.
        use_backbone_checkpoints (bool): Whether this run consumes precomputed checkpoints.

    Returns:
        TrainingConfig: Backbone training configuration for this run.
    """
    backbone_config = experiment_config.backbone
    training_config = (backbone_config.training if backbone_config is not None else None)
    if training_config is not None:
        return training_config
    if not use_backbone_checkpoints:
        raise ValueError('`backbone.training` must be provided for runs that train backbone '
                         'weights.')
    return TrainingConfig(
        num_epochs=1,
        strategy=StrategyConfig(name='naive'),
        optimizer=OptimizerConfig(),
        batch_size=128,
        lr_scheduler=None,
        grad_clip_max_norm=None,
    )


def resolve_controller_type(controller_config: ControllerConfig) -> str:
    """
    Resolve controller type from configuration.

    Args:
        controller_config (ControllerConfig): Controller config.

    Returns:
        str: One of `repair` or `prevention`.
    """
    controller_path = get_controller_path(controller_config.name)
    controller_cls = import_symbol(controller_path)
    if not inspect.isclass(controller_cls):
        raise TypeError(f'Controller symbol is not a class: {controller_path}')
    if issubclass(controller_cls, RepairController):
        return 'repair'
    if issubclass(controller_cls, PreventionController):
        return 'prevention'
    raise ValueError(f'{controller_path} is not a prevention or repair controller.')


def resolve_avalanche_eval_every(
    *,
    avalanche_schedule: str,
) -> int:
    """
    Convert the config-level Avalanche evaluation schedule to `eval_every`.

    Args:
        avalanche_schedule (str): Config schedule (`per_experience` or `final_only`).

    Returns:
        int: Avalanche `eval_every` value.
    """
    if avalanche_schedule == 'per_experience':
        return 0
    if avalanche_schedule == 'final_only':
        return -1
    raise ValueError(f'Unsupported eval schedule: {avalanche_schedule}')


def to_scalar(value: object) -> float | None:
    """
    Safely convert a metric value to a scalar float if possible.

    Args:
        value: Metric value to convert.

    Returns:
        Float representation if conversion succeeds, otherwise None.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, 'item'):
        try:
            return float(value.item())
        except Exception:  # pylint: disable=broad-exception-caught
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_scalar_metrics(metrics: dict[str, Any] | None) -> dict[str, float]:
    """
    Filter a metrics dictionary down to scalar float entries.

    Args:
        metrics: Raw metric scores.

    Returns:
        Dictionary of scalar metrics keyed by metric name.
    """
    if metrics is None:
        return {}
    scalar_results: dict[str, float] = {}
    for key, value in metrics.items():
        scalar_value = to_scalar(value)
        if scalar_value is not None:
            scalar_results[key] = scalar_value
    return scalar_results


def count_parameters(module: nn.Module | None, *, trainable_only: bool = False) -> int:
    """
    Count parameters in a module.

    Args:
        module (nn.Module | None): Module to inspect.
        trainable_only (bool): If True, count only parameters with `requires_grad=True`.

    Returns:
        int: Number of parameters (0 if `module` is None).
    """
    if module is None:
        return 0

    if trainable_only:
        return int(sum(p.numel() for p in module.parameters() if getattr(p, 'requires_grad', False)))

    return int(sum(p.numel() for p in module.parameters()))
