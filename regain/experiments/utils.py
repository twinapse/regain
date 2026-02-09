from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
import yaml

from regain.registry import get_controller_path
from regain.registry import get_scenario_builder_path

__all__ = [
    'EvalMode',
    'ControllerConfig',
    'StrategyConfig',
    'OptimizerConfig',
    'RunConfig',
    'ExperimentConfig',
    'load_experiment_config',
    'guard_experiment_config_overrides',
    'enable_determinism',
    'to_scalar',
    'extract_scalar_metrics',
    'count_parameters',
]

_CONFIG_PARAM_OVERRIDE_MAP: list[tuple[str, list[str]]] = [
    ('num_epochs', ['num_epochs', 'train_epochs', 'n_epochs', 'epochs']),
    ('num_classes', ['num_classes', 'n_classes']),
    ('train_batch_size', ['train_batch_size', 'train_mb_size', 'batch_size']),
    ('eval_batch_size', ['eval_batch_size', 'eval_mb_size']),
    ('replay_batch_size', ['replay_batch_size', 'mem_mb_size', 'mem_batch_size', 'batch_size_mem']),
    ('replay_memory_size', ['replay_memory_size', 'replay_mem_size', 'mem_size']),
    ('eval_every', ['eval_every']),
    ('device', ['device']),
]


EvalMode = Literal['single', 'compare']


@dataclass
class ControllerConfig:
    """
    Configuration for a retrieval controller to construct dynamically.

    Attributes:
        name: Controller identifier (e.g., `logit_bias`, `linear_probe`).
        repair_epochs (int): Number of epochs to use for repair fitting (only for repair controllers).
                             If not provided, defaults to the value used for the training strategy.
        repair_batch_size (int): Batch size to use for repair fitting (only for repair controllers).
                                 If not provided, defaults to the value used for the training strategy.
        params: Keyword arguments to pass to the controller constructor.
    """

    name: str
    repair_epochs: int | None = None
    repair_batch_size: int | None = None
    params: dict[str, object] = field(default_factory=dict)


@dataclass
class StrategyConfig:
    """
    Configuration for the Avalanche training strategy.

    Attributes:
        name: Strategy identifier.
        params: Strategy-specific parameters.
    """

    name: Literal['naive', 'replay', 'bic', 'il2m']
    params: dict[str, object] = field(default_factory=dict)


@dataclass
class OptimizerConfig:
    """
    Optimizer configuration for a run.

    Attributes:
        name: Optimizer identifier.
        params: Keyword arguments passed to the optimizer constructor.
    """

    name: str = 'sgd'
    params: dict[str, object] = field(default_factory=dict)


@dataclass
class RunConfig:
    """
    Configuration for an individual run within an experiment.

    Attributes:
        run_name: Identifier for the run (used as the MLflow run name).
        strategy: Avalanche strategy configuration.
        optimizer: Optimizer configuration.
        controller: Optional controller configuration.
        eval_mode: Evaluation mode to use for metric reporting.
    """

    run_name: str
    strategy: StrategyConfig
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    controller: ControllerConfig | None = None
    eval_mode: EvalMode = 'single'


@dataclass
class ExperimentConfig:
    """
    Experiment-level configuration shared across all runs.

    Attributes:
        experiment_name: MLflow experiment name.
        scenario: Scenario name registered in `regain.registry`.
        num_experiences: Number of experiences in which the dataset is split.
        num_epochs: Number of epochs per experience.
        runs_config: Sequence of run configurations to execute.
        repair_budget_per_class: Number of repair examples per class (per experience).
        repair_after_experience: Whether to perform repair after each experience.
        replay_memory_size: Replay memory size for replay-style strategies.
        replay_batch_size: Replay mini-batch size for replay strategies (Avalanche `mem_mb_size`).
        train_batch_size: Training mini-batch size for current data (Avalanche `train_mb_size`).
        eval_batch_size: Evaluation mini-batch size.
        eval_every: Evaluation frequency for Avalanche built-in evaluation (-1 disables, 0=end of experience).
        seed: Random seed.
        deterministic: Whether to enforce deterministic PyTorch behavior.
        device: Device identifier for training.
        mlflow_tracking_uri: Optional MLflow tracking URI or filesystem path (SQLite only).
        mlflow_artifact_uri: Optional MLflow artifact URI or filesystem path.
        dataset_path: Optional dataset root to pass to the scenario builder.
        debug: Whether to enable debug instrumentation for repair controllers.
    """

    experiment_name: str
    scenario: str
    num_experiences: int
    num_epochs: int
    runs_config: list[RunConfig]
    repair_budget_per_class: int = 0
    repair_after_experience: bool = True
    replay_memory_size: int = 2000
    replay_batch_size: int = 128
    train_batch_size: int = 128
    eval_batch_size: int = 128
    eval_every: int = -1
    seed: int = 1
    deterministic: bool = False
    debug: bool = False
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    mlflow_tracking_uri: str | None = None
    mlflow_artifact_uri: str | None = None
    dataset_path: str | Path | None = None


def _resolve_device(device: str | None) -> str:
    """
    Resolve a device string, defaulting to available CUDA if requested.

    Args:
        device: Device identifier or `auto`/None to infer.

    Returns:
        Concrete device string.
    """
    if device is None or device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


# TODO: Decompose this function. It is too long.
def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    """
    Load an experiment configuration from a YAML file.

    Args:
        config_path: Path to a YAML file matching ExperimentConfig/RunConfig fields.

    Returns:
        Parsed ExperimentConfig instance.
    """
    # Load raw YAML payload
    path = Path(config_path)
    with path.open('r', encoding='utf-8') as f:
        payload = yaml.safe_load(f) or {}

    # Validate top-level structure
    if 'runs_config' not in payload:
        raise ValueError('Configuration file must include a "runs_config" list.')

    scenario = payload.get('scenario')
    if scenario is not None:
        get_scenario_builder_path(scenario)

    # Get optional dataset path
    dataset_path_value = payload.get('dataset_path')
    dataset_path = Path(dataset_path_value) if dataset_path_value is not None else None

    # Parse run configurations
    guard_experiment_config_overrides(payload)

    def _parse_strategy_config(run_config: dict[str, Any]) -> StrategyConfig:
        strategy_config = run_config.get('strategy')
        if not isinstance(strategy_config, dict):
            raise ValueError('Each run must define a strategy configuration.')
        if 'name' not in strategy_config:
            raise ValueError('Each run strategy must provide a `name`.')
        return StrategyConfig(
            name=strategy_config['name'],
            params=dict(strategy_config.get('params') or {}),
        )

    def _parse_optimizer_config(run_config: dict[str, Any]) -> OptimizerConfig:
        optimizer_config = run_config.get('optimizer')
        if not isinstance(optimizer_config, dict):
            raise ValueError('Each run must define an optimizer configuration.')
        if 'name' not in optimizer_config:
            raise ValueError('Optimizer configuration must include a `name`.')
        return OptimizerConfig(
            name=optimizer_config['name'],
            params=dict(optimizer_config.get('params') or {}),
        )

    def _parse_controller_config(run_config: dict[str, Any]) -> ControllerConfig | None:
        # Controller is optional
        if 'controller' not in run_config:
            return None
        # Get and validate controller config
        config = run_config['controller']
        if not isinstance(config, dict):
            raise ValueError('Controller configuration must be a mapping.')
        if 'name' not in config:
            raise ValueError('Controller configuration must include a `name`.')
        get_controller_path(config['name'])
        # Return parsed controller config
        return ControllerConfig(
            name=config['name'],
            repair_epochs=config.get('repair_epochs'),
            repair_batch_size=config.get('repair_batch_size'),
            params=dict(config.get('params') or {}),
        )

    def _parse_eval_mode(run_config: dict[str, Any]) -> EvalMode:
        eval_mode = run_config.get('eval_mode', 'single')
        if eval_mode not in ('single', 'compare'):
            raise ValueError('`eval_mode` must be "single" or "compare".')
        return eval_mode

    runs_config = []
    for run_cfg in payload['runs_config']:
        if 'run_name' not in run_cfg:
            raise ValueError('Each run configuration must include a "run_name".')
        runs_config.append(
            RunConfig(
                run_name=run_cfg['run_name'],
                strategy=_parse_strategy_config(run_cfg),
                optimizer=_parse_optimizer_config(run_cfg),
                controller=_parse_controller_config(run_cfg),
                eval_mode=_parse_eval_mode(run_cfg),
            )
        )

    # Build and return experiment config
    return ExperimentConfig(
        experiment_name=payload['experiment_name'],
        scenario=payload['scenario'],
        num_experiences=payload['num_experiences'],
        num_epochs=payload['num_epochs'],
        runs_config=runs_config,
        repair_budget_per_class=payload.get('repair_budget_per_class', 0),
        repair_after_experience=payload.get('repair_after_experience', True),
        replay_memory_size=payload.get('replay_memory_size', 2000),
        replay_batch_size=payload.get('replay_batch_size', 128),
        train_batch_size=payload.get('train_batch_size', 128),
        eval_batch_size=payload.get('eval_batch_size', 128),
        eval_every=payload.get('eval_every', 0),
        seed=payload.get('seed', 1),
        deterministic=payload.get('deterministic', False),
        debug=payload.get('debug', False),
        device=_resolve_device(payload.get('device')),
        mlflow_tracking_uri=payload.get('mlflow_tracking_uri'),
        mlflow_artifact_uri=payload.get('mlflow_artifact_uri'),
        dataset_path=dataset_path,
    )


def guard_experiment_config_overrides(experiment_config: dict[str, Any]) -> None:
    """
    Validate that nested run config parameters do not override experiment config parameters.

    Args:
        experiment_config (dict[str, Any]): Raw YAML payload.
    """
    runs = experiment_config.get('runs_config') or []
    if not isinstance(runs, list):
        raise ValueError('Configuration file must include a "runs_config" list.')

    for run_config in runs:
        if not isinstance(run_config, dict):
            raise ValueError('Each run configuration must be a mapping.')

        _guard_config_param_overrides(parent_config=experiment_config, child_config=run_config)


def _guard_config_param_overrides(
    *,
    parent_config: Mapping[str, object],
    child_config: Mapping[str, object],
) -> None:
    """
    Raise if a child config subtree attempts to override parent-owned values.

    Args:
        parent_config (Mapping[str, object]): Parent config mapping.
        child_config (Mapping[str, object]): Config subtree to validate.
    """
    if not isinstance(parent_config, Mapping):
        raise TypeError('parent_config must be a mapping.')
    if not isinstance(child_config, Mapping):
        raise TypeError('child_config must be a mapping.')
    parent_params = _resolve_parent_config_params(parent_config)
    _guard_config_tree(parent_params=parent_params, child_config=child_config, context='config')


def _resolve_parent_config_params(parent_config: Mapping[str, object]) -> dict[str, object]:
    """
    Extract parent-owned parameter values from a config object.

    Args:
        parent_config (Mapping[str, object]): Parent config mapping.

    Returns:
        dict[str, object]: Mapping of parent-owned parameters to values (missing keys use None).
    """
    parent_params: dict[str, object] = {}
    for parent_key, _ in _CONFIG_PARAM_OVERRIDE_MAP:
        parent_params[parent_key] = parent_config.get(parent_key)
    return parent_params


def _guard_config_tree(*, parent_params: dict[str, object], child_config: object, context: str) -> None:
    """
    Recursively validate a config subtree against parent-owned parameters.

    Args:
        parent_params (dict[str, object]): Parent-owned parameter values.
        child_config (object): Config subtree to validate.
        context (str): Label for error messages.
    """
    if isinstance(child_config, Mapping):
        for parent_key, child_keys in _CONFIG_PARAM_OVERRIDE_MAP:
            overrides = [key for key in child_keys if key in child_config]
            if overrides:
                raise ValueError(
                    f'{context} params should not override `{parent_key}`; '
                    f'use experiment config instead (remove: {sorted(overrides)}).'
                )
        for key, value in child_config.items():
            if isinstance(value, Mapping) or isinstance(value, (list, tuple)):
                _guard_config_tree(parent_params=parent_params, child_config=value, context=f'{context}.{key}')
        return

    if isinstance(child_config, (list, tuple)):
        for index, value in enumerate(child_config):
            if isinstance(value, Mapping) or isinstance(value, (list, tuple)):
                _guard_config_tree(parent_params=parent_params, child_config=value, context=f'{context}[{index}]')


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
        except Exception:
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
