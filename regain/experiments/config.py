from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any, Literal

import torch
import yaml

from regain.constants import NS_SEP
from regain.constants import PARAM_AVALANCHE_VERSION
from regain.constants import PARAM_BACKBONE
from regain.constants import PARAM_CONTROLLER
from regain.constants import PARAM_CONTROLLER_MODEL_PARAM_COUNT
from regain.constants import PARAM_CONTROLLER_PATH
from regain.constants import PARAM_DEBUG_SKIP_REASON
from regain.constants import PARAM_NUM_CLASSES
from regain.constants import PARAM_SCENARIO
from regain.constants import PARAM_SEED
from regain.constants import PARAM_TORCH_DETERMINISTIC_ALGORITHMS
from regain.constants import RUN_NAME_BACKBONE
from regain.registry import get_backbone_path
from regain.registry import get_controller_path
from regain.registry import get_scenario_builder_path

__all__ = [
    'ControllerConfig',
    'StrategyConfig',
    'OptimizerConfig',
    'TrainingConfig',
    'BackboneConfig',
    'RepairConfig',
    'RunConfig',
    'ExperimentConfig',
    'load_experiment_config',
    'guard_experiment_config_overrides',
]

_CONFIG_PARAM_OVERRIDE_MAP: list[tuple[str, list[str]]] = [
    ('num_epochs', ['num_epochs', 'train_epochs', 'n_epochs', 'epochs']),
    (PARAM_NUM_CLASSES, ['n_classes']),
    ('batch_size', ['batch_size', 'train_mb_size', 'train_batch_size']),
    ('eval_batch_size', ['eval_batch_size', 'eval_mb_size']),
    ('batch_size_mem', ['batch_size_mem', 'mem_mb_size', 'mem_batch_size']),
    ('mem_size', ['mem_size']),
    ('eval_schedule', ['eval_schedule']),
    ('device', ['device']),
    ('budget_per_class', ['budget_per_class']),
    ('fit_schedule', ['fit_schedule']),
    ('checkpoints_enabled', ['checkpoints_enabled']),
]

_RESERVED_LOG_PARAMS: set[str] = {
    PARAM_AVALANCHE_VERSION,
    PARAM_CONTROLLER_MODEL_PARAM_COUNT,
    PARAM_CONTROLLER_PATH,
    PARAM_DEBUG_SKIP_REASON,
    PARAM_TORCH_DETERMINISTIC_ALGORITHMS,
}


@dataclass
class ControllerConfig:
    """
    Configuration for a retrieval controller to construct dynamically.

    Attributes:
        name: Controller identifier (e.g., `logit_bias`, `linear_probe`).
        kwargs: Keyword arguments to pass to the controller constructor.
    """

    name: str
    kwargs: dict[str, object] = field(default_factory=dict)


@dataclass
class StrategyConfig:
    """
    Configuration for the Avalanche training strategy.

    Attributes:
        name: Strategy identifier.
        kwargs: Strategy-specific keyword arguments.
    """

    name: Literal['naive', 'replay', 'bic', 'il2m']
    kwargs: dict[str, object] = field(default_factory=dict)


@dataclass
class OptimizerConfig:
    """
    Optimizer configuration for a run.

    Attributes:
        name: Optimizer identifier.
        kwargs: Keyword arguments passed to the optimizer constructor.
    """

    name: str = 'sgd'
    kwargs: dict[str, object] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """
    Backbone training configuration for learning from scratch.

    Attributes:
        num_epochs: Number of backbone training epochs per experience.
        strategy: Avalanche strategy configuration for backbone training.
        optimizer: Optimizer configuration for backbone training.
        batch_size: Training mini-batch size for current data (Avalanche `train_mb_size`).
    """

    num_epochs: int
    strategy: StrategyConfig
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    batch_size: int = 128


@dataclass
class BackboneConfig:
    """
    Backbone configuration shared by all runs.

    Attributes:
        name: Optional backbone registry name used to resolve the model class.
              It may be `None` when a source experiment is configured, in which
              case the name is resolved from the source `backbone` run.
        training: Optional backbone training configuration for learning from scratch.
        source_experiment: Optional source experiment id/name from which to load the reserved `backbone` run
                           checkpoints and baseline artifacts.
    """

    name: str | None = None
    training: TrainingConfig | None = None
    source_experiment: str | None = None


@dataclass
class RepairConfig:
    """
    Config shared by all repair controllers.

    Attributes:
        budget_per_class: Number of repair examples per class (per experience).
        fit_schedule: Repair fitting schedule (`per_experience` or `final_only`).
        num_epochs: Number of epochs used by all repair controllers.
        batch_size: Batch size used by all repair controllers. If omitted, the backbone training batch size is used.
    """

    budget_per_class: int = 0
    fit_schedule: Literal['per_experience', 'final_only'] = 'per_experience'
    num_epochs: int | None = None
    batch_size: int | None = None


@dataclass
class RunConfig:
    """
    Configuration for an individual run within an experiment.

    Attributes:
        name: Identifier for the run (used as the MLflow run name).
        controller: Controller configuration.
    """

    name: str
    controller: ControllerConfig


@dataclass
class ExperimentConfig:
    """
    Experiment-level configuration shared across all runs.

    Attributes:
        experiment_name: MLflow experiment name.
        scenario: Scenario name registered in `regain.registry`.
        num_experiences: Number of experiences in which the dataset is split.
        backbone: Optional backbone configuration shared by all runs.
                  May be omitted/null only when reusing an existing local reserved `backbone` run.
        repair: Config shared by all repair controllers.
                This is mandatory because the backbone must be trained with the same data splitting
                logic as the repair controllers to prevent data leakage.
        runs: Sequence of run configurations to execute. May be empty for backbone-only pretraining.
        eval_batch_size: Evaluation mini-batch size.
        eval_schedule: Evaluation schedule for Avalanche built-in evaluation (`per_experience` or `final_only`).
        checkpoints_enabled: Whether to persist checkpoints to MLflow artifacts.
                             Checkpoint artifact logging applies to backbone checkpoints.
        device: Device identifier for training.
        seed: Random seed.
        deterministic: Whether to enforce deterministic PyTorch behavior.
        mlflow_tracking_uri: Optional MLflow tracking URI or filesystem path (SQLite only).
        mlflow_artifact_uri: Optional MLflow artifact URI or filesystem path.
        dataset_path: Optional dataset root to pass to the scenario builder.
        debug: Whether to enable debug instrumentation for repair controllers.
    """

    ###############
    # Core config #
    ###############
    experiment_name: str
    scenario: str
    num_experiences: int
    backbone: BackboneConfig | None
    repair: RepairConfig
    runs: list[RunConfig] | None

    #################
    # Training/eval #
    #################
    eval_batch_size: int = 128
    eval_schedule: Literal['per_experience', 'final_only'] = 'per_experience'
    checkpoints_enabled: bool = False

    ###############################
    # Runtime and reproducibility #
    ###############################
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 1
    deterministic: bool = False

    ###########################
    # Tracking and data paths #
    ###########################
    mlflow_tracking_uri: str | None = None
    mlflow_artifact_uri: str | None = None
    dataset_path: str | Path | None = None

    ###################
    # Debug/diagnosis #
    ###################
    debug: bool = False


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


def _has_kwargs_segment(*, dotted_path: str) -> bool:
    """
    Check whether a dotted path includes a `kwargs` segment.

    Args:
        dotted_path (str): Candidate dotted path.

    Returns:
        bool: True when the path includes a `kwargs` segment.
    """
    return 'kwargs' in dotted_path.split(NS_SEP)


def _normalize_logged_param_path(*, dotted_path: str) -> str:
    """
    Normalize a dotted path to match MLflow logging key normalization.

    Args:
        dotted_path (str): Candidate dotted path.

    Returns:
        str: Normalized path with `kwargs` segments removed.
    """
    segments = dotted_path.split(NS_SEP)
    if len(segments) == 1:
        return dotted_path
    return NS_SEP.join(segment for segment in segments if segment != 'kwargs')


def _collect_logged_leaf_paths(
    *,
    config_tree: Mapping[str, object],
    prefix: str | None = None,
    within_kwargs: bool = False,
) -> list[tuple[str, str, bool]]:
    """
    Collect raw and normalized dotted leaf param paths for logging conflict checks.

    Args:
        config_tree (Mapping[str, object]): Config mapping to inspect.
        prefix (str | None): Optional path prefix.
        within_kwargs (bool): Whether the current subtree is within a `kwargs` namespace.

    Returns:
        list[tuple[str, str, bool]]: Tuples of raw path, normalized path, and `kwargs` membership.
    """
    leaf_paths: list[tuple[str, str, bool]] = []
    for key, value in config_tree.items():
        key_str = str(key)
        raw_path = f'{prefix}{NS_SEP}{key_str}' if prefix else key_str
        next_within_kwargs = within_kwargs or key_str == 'kwargs'
        if isinstance(value, Mapping):
            leaf_paths.extend(
                _collect_logged_leaf_paths(
                    config_tree=value,
                    prefix=raw_path,
                    within_kwargs=next_within_kwargs,
                )
            )
            continue
        if value is None:
            continue
        normalized_path = _normalize_logged_param_path(dotted_path=raw_path)
        leaf_paths.append((raw_path, normalized_path, next_within_kwargs))
    return leaf_paths


def _guard_kwargs_conflicts_in_scope(
    *,
    scope_config: Mapping[str, object],
    scope_context: str,
) -> None:
    """
    Reject param paths in `kwargs` that collide with logged param paths in a config scope.

    Args:
        scope_config (Mapping[str, object]): Config scope to validate.
        scope_context (str): Prefix for error message paths.

    Returns:
        None

    Raises:
        ValueError: If a param path in `kwargs` conflicts with a normalized or reserved logged param.
    """
    leaf_paths = _collect_logged_leaf_paths(config_tree=scope_config)
    normalized_to_raw_paths: dict[str, set[str]] = {}
    for raw_path, normalized_path, _ in leaf_paths:
        normalized_to_raw_paths.setdefault(normalized_path, set()).add(raw_path)

    for normalized_path, raw_paths in normalized_to_raw_paths.items():
        if len(raw_paths) < 2:
            continue
        param_paths_in_kwargs = sorted(path for path in raw_paths if _has_kwargs_segment(dotted_path=path))
        if not param_paths_in_kwargs:
            continue
        conflicting_paths = sorted(path for path in raw_paths if path not in param_paths_in_kwargs)
        param_path_in_kwargs = param_paths_in_kwargs[0]
        conflicting_path = conflicting_paths[0] if conflicting_paths else param_paths_in_kwargs[1]
        raise ValueError(
            f'Invalid param `{scope_context}{NS_SEP}{param_path_in_kwargs}`: '
            f'resolves to `{normalized_path}` and conflicts with '
            f'`{scope_context}{NS_SEP}{conflicting_path}` after removing `.kwargs`.'
        )

    for raw_path, normalized_path, is_within_kwargs in leaf_paths:
        if not is_within_kwargs:
            continue
        if normalized_path in _RESERVED_LOG_PARAMS:
            raise ValueError(
                f'Invalid param `{scope_context}{NS_SEP}{raw_path}`: '
                f'resolves to reserved logged parameter `{normalized_path}`. '
                f'`{normalized_path}` cannot be set under `kwargs`.'
            )


def _guard_kwargs_conflicts(*, payload: Mapping[str, object]) -> None:
    """
    Reject param paths in `kwargs` that would collide with normalized or reserved logged params.

    Args:
        payload (Mapping[str, object]): Raw experiment config payload.

    Returns:
        None
    """
    experiment_scope = {
        key: value
        for key, value in payload.items()
        if key != 'runs'
    }
    _guard_kwargs_conflicts_in_scope(
        scope_config=experiment_scope,
        scope_context='config',
    )

    runs_payload = payload.get('runs')
    if runs_payload is None:
        return
    if not isinstance(runs_payload, list):
        raise ValueError('Configuration field `runs` must be a list when provided.')

    for run_index, run_cfg in enumerate(runs_payload):
        if not isinstance(run_cfg, Mapping):
            raise ValueError('Each run configuration must be a mapping.')
        _guard_kwargs_conflicts_in_scope(
            scope_config=run_cfg,
            scope_context=f'runs[{run_index}]',
        )


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

    _guard_kwargs_conflicts(payload=payload)

    scenario = payload.get(PARAM_SCENARIO)
    if scenario is not None:
        get_scenario_builder_path(scenario)

    # Get optional dataset path
    dataset_path_value = payload.get('dataset_path')
    dataset_path = Path(dataset_path_value) if dataset_path_value is not None else None

    # Parse run configurations
    guard_experiment_config_overrides(payload)

    def _parse_strategy_config(config: dict[str, Any], *, config_name: str) -> StrategyConfig:
        strategy_config = config.get('strategy')
        if not isinstance(strategy_config, dict):
            raise ValueError(f'{config_name} must define a strategy configuration.')
        if 'name' not in strategy_config:
            raise ValueError(f'{config_name} strategy must provide a `name`.')
        kwargs_payload = strategy_config.get('kwargs')
        if kwargs_payload is None:
            kwargs_payload = {}
        if not isinstance(kwargs_payload, Mapping):
            raise ValueError(f'{config_name} strategy `kwargs` must be a mapping.')
        return StrategyConfig(
            name=strategy_config['name'],
            kwargs=dict(kwargs_payload),
        )

    def _parse_optimizer_config(config: dict[str, Any], *, config_name: str) -> OptimizerConfig:
        optimizer_config = config.get('optimizer')
        if optimizer_config is None:
            return OptimizerConfig()
        if not isinstance(optimizer_config, dict):
            raise ValueError(f'{config_name} must define an optimizer configuration.')
        if 'name' not in optimizer_config:
            raise ValueError(f'{config_name} optimizer configuration must include a `name`.')
        kwargs_payload = optimizer_config.get('kwargs')
        if kwargs_payload is None:
            kwargs_payload = {}
        if not isinstance(kwargs_payload, Mapping):
            raise ValueError(f'{config_name} optimizer `kwargs` must be a mapping.')
        return OptimizerConfig(
            name=optimizer_config['name'],
            kwargs=dict(kwargs_payload),
        )

    def _parse_controller_config(run_config: dict[str, Any]) -> ControllerConfig:
        if PARAM_CONTROLLER not in run_config:
            raise ValueError('Each run configuration must include a `controller` mapping.')
        # Get and validate controller config
        config = run_config[PARAM_CONTROLLER]
        if not isinstance(config, dict):
            raise ValueError('Controller configuration must be a mapping.')
        if 'name' not in config:
            raise ValueError('Controller configuration must include a `name`.')
        kwargs_payload = config.get('kwargs')
        if kwargs_payload is None:
            kwargs_payload = {}
        if not isinstance(kwargs_payload, Mapping):
            raise ValueError('Controller config `kwargs` must be a mapping.')
        get_controller_path(config['name'])
        # Return parsed controller config
        return ControllerConfig(
            name=config['name'],
            kwargs=dict(kwargs_payload),
        )

    def _parse_backbone_config(config: dict[str, Any]) -> BackboneConfig | None:
        backbone_config = config.get(PARAM_BACKBONE)
        if backbone_config is None:
            return None
        if not isinstance(backbone_config, dict):
            raise ValueError('Experiment config `backbone` must be a mapping when provided.')

        source_experiment_raw = backbone_config.get('source_experiment')
        if source_experiment_raw not in (None, ''):
            if not isinstance(source_experiment_raw, (str, int)):
                raise ValueError('Backbone config `source_experiment` must be a string or integer experiment id.')
            source_experiment = str(source_experiment_raw)
            invalid_keys = sorted(
                key
                for key in backbone_config
                if key != 'source_experiment'
            )
            if invalid_keys:
                raise ValueError(
                    'When `backbone.source_experiment` is provided, it must be the only field under `backbone`. '
                    f'Remove: {invalid_keys}'
                )
            experiment_name_raw = config.get('experiment_name')
            if experiment_name_raw is not None:
                if source_experiment.strip() == str(experiment_name_raw).strip():
                    raise ValueError(
                        'Backbone config `source_experiment` must be different from `experiment_name`.'
                    )
            return BackboneConfig(
                name=None,
                training=None,
                source_experiment=source_experiment,
            )

        backbone_name = backbone_config.get('name', 'resnet18')
        if not isinstance(backbone_name, str):
            raise ValueError('Backbone config `name` must be a string.')
        backbone_name = backbone_name.strip()
        get_backbone_path(backbone_name)

        training_config = backbone_config.get('training')
        if training_config is None:
            raise ValueError('Backbone config must define exactly one of `training` or `source_experiment`.')
        if not isinstance(training_config, dict):
            raise ValueError('Backbone config `training` must be a mapping when provided.')
        if 'num_epochs' not in training_config:
            raise ValueError('Backbone config `training` must include `num_epochs`.')
        training = TrainingConfig(
            num_epochs=training_config['num_epochs'],
            strategy=_parse_strategy_config(training_config, config_name='Backbone training config'),
            optimizer=_parse_optimizer_config(training_config, config_name='Backbone training config'),
            batch_size=training_config.get('batch_size', 128),
        )

        return BackboneConfig(
            name=backbone_name,
            training=training,
            source_experiment=None,
        )

    def _parse_repair_config(config: dict[str, Any]) -> RepairConfig:
        repair_config = config.get('repair')
        if repair_config is None:
            raise ValueError(
                'Experiment config must include a `repair` section to define data splitting. '
                'Use `budget_per_class: 0` if using the full dataset (no repair).'
            )
        if not isinstance(repair_config, dict):
            raise ValueError('`repair` must be a mapping when provided.')
        fit_schedule = repair_config.get('fit_schedule', 'per_experience')
        if fit_schedule not in {'per_experience', 'final_only'}:
            raise ValueError('`repair.fit_schedule` must be one of: `per_experience`, `final_only`.')
        return RepairConfig(
            budget_per_class=repair_config.get('budget_per_class', 0),
            fit_schedule=fit_schedule,
            num_epochs=repair_config.get('num_epochs'),
            batch_size=repair_config.get('batch_size'),
        )

    runs_payload = payload.get('runs')
    runs: list[RunConfig] | None = None
    if runs_payload is not None:
        if not isinstance(runs_payload, list):
            raise ValueError('Configuration field `runs` must be a list when provided.')

        runs = []
        for run_cfg in runs_payload:
            if not isinstance(run_cfg, dict):
                raise ValueError('Each run configuration must be a mapping.')
            if 'name' not in run_cfg:
                raise ValueError('Each run configuration must include a `name`.')
            run_name = run_cfg['name']
            if not isinstance(run_name, str):
                raise ValueError('Run config `name` must be a string.')
            if run_name == RUN_NAME_BACKBONE:
                raise ValueError(
                    f"Run name '{RUN_NAME_BACKBONE}' is reserved and cannot be used in runs."
                )
            runs.append(
                RunConfig(
                    name=run_name,
                    controller=_parse_controller_config(run_cfg),
                )
            )

    checkpoints_enabled = payload.get('checkpoints_enabled', False)
    if not isinstance(checkpoints_enabled, bool):
        raise ValueError('`checkpoints_enabled` must be a boolean.')

    eval_schedule = payload.get('eval_schedule', 'per_experience')
    if eval_schedule not in {'per_experience', 'final_only'}:
        raise ValueError('`eval_schedule` must be one of: `per_experience`, `final_only`.')

    # Build and return experiment config
    return ExperimentConfig(
        experiment_name=payload['experiment_name'],
        scenario=payload[PARAM_SCENARIO],
        num_experiences=payload['num_experiences'],
        backbone=_parse_backbone_config(payload),
        repair=_parse_repair_config(payload),
        runs=runs,
        eval_batch_size=payload.get('eval_batch_size', 128),
        eval_schedule=eval_schedule,
        checkpoints_enabled=checkpoints_enabled,
        device=_resolve_device(payload.get('device')),
        seed=payload.get(PARAM_SEED, 1),
        deterministic=payload.get('deterministic', False),
        mlflow_tracking_uri=payload.get('mlflow_tracking_uri'),
        mlflow_artifact_uri=payload.get('mlflow_artifact_uri'),
        dataset_path=dataset_path,
        debug=payload.get('debug', False),
    )


def guard_experiment_config_overrides(experiment_config: dict[str, Any]) -> None:
    """
    Validate that nested run config parameters do not override experiment config parameters.

    Args:
        experiment_config (dict[str, Any]): Raw YAML payload.
    """
    runs_payload = experiment_config.get('runs')
    runs = [] if runs_payload is None else runs_payload
    if not isinstance(runs, list):
        raise ValueError('Configuration field `runs` must be a list when provided.')

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
    _guard_config_tree(child_config=child_config, context='config')


def _guard_config_tree(*, child_config: object, context: str) -> None:
    """
    Recursively validate a config subtree against parent-owned parameters.

    Args:
        child_config (object): Config subtree to validate.
        context (str): Label for error messages.
    """
    if isinstance(child_config, Mapping):
        for parent_key, child_keys in _CONFIG_PARAM_OVERRIDE_MAP:
            overrides = [key for key in child_keys if key in child_config]
            if overrides:
                raise ValueError(
                    f'{context} keys should not override `{parent_key}`; '
                    f'use experiment config instead (remove: {sorted(overrides)}).'
                )
        for key, value in child_config.items():
            if isinstance(value, Mapping) or isinstance(value, (list, tuple)):
                _guard_config_tree(child_config=value, context=f'{context}.{key}')
        return

    if isinstance(child_config, (list, tuple)):
        for index, value in enumerate(child_config):
            if isinstance(value, Mapping) or isinstance(value, (list, tuple)):
                _guard_config_tree(child_config=value, context=f'{context}[{index}]')
