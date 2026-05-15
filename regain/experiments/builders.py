"""
Builder utilities for experiment benchmarks, models, strategies, and controllers.
"""

from collections.abc import Mapping, Sequence
import inspect

from avalanche.benchmarks.scenarios import NCScenario
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import Naive
from avalanche.training.supervised import Replay
from avalanche.training.templates import BaseTemplate
import torch
from torch.optim import AdamW
from torch.optim import SGD

from regain.avalanche_utils.plugins import ControllerPlugin
from regain.avalanche_utils.plugins import GradientClippingPlugin
from regain.avalanche_utils.plugins import LRSchedulerPlugin
from regain.avalanche_utils.plugins import PreventionControllerPlugin
from regain.avalanche_utils.plugins import RepairControllerPlugin
from regain.avalanche_utils.scenarios import get_scenario_builder
from regain.avalanche_utils.scenarios import ScenarioBuilder
from regain.constants import NS_SEP
from regain.constants import PARAM_BACKBONE_REPLAY_BATCH_SIZE_MEM
from regain.constants import PARAM_BACKBONE_REPLAY_MEM_SIZE
from regain.constants import PARAM_NUM_CLASSES
from regain.debug.avalanche_utils import DebugRepairControllerPlugin
from regain.experiments.config import BackboneConfig
from regain.experiments.config import ControllerConfig
from regain.experiments.config import ExperimentConfig
from regain.experiments.config import OptimizerConfig
from regain.experiments.config import StrategyConfig
from regain.experiments.config import TrainingConfig
from regain.experiments.utils import resolve_avalanche_eval_every
from regain.models.controllers import Controller
from regain.models.controllers import PreventionController
from regain.models.controllers import RepairController
from regain.registry import get_backbone_path
from regain.registry import get_controller_path
from regain.registry import get_lr_scheduler_path
from regain.registry import import_symbol

__all__ = [
    'build_backbone',
    'build_benchmark',
    'build_controller',
    'build_controller_plugin',
    'build_gradient_clipping_plugin',
    'build_lr_scheduler_plugin',
    'build_optimizer',
    'make_strategy',
]

_PARAM_CONTROLLER_CLASSES = 'classes'
_PARAM_CONTROLLER_REPLAY_BATCH_SIZE = 'replay_batch_size'
_PARAM_CONTROLLER_REPLAY_MEMORY_SIZE = 'replay_memory_size'
_PARAM_CONTROLLER_TRAIN_BATCH_SIZE = 'train_batch_size'

_PARAM_NAME_REPLAY_BATCH_SIZE_MEM = PARAM_BACKBONE_REPLAY_BATCH_SIZE_MEM.rsplit(NS_SEP, 1)[-1]
_PARAM_NAME_REPLAY_MEM_SIZE = PARAM_BACKBONE_REPLAY_MEM_SIZE.rsplit(NS_SEP, 1)[-1]


##########################
# Benchmarks (scenarios) #
##########################


def _resolve_backbone_image_size(*, backbone_config: BackboneConfig) -> int | None:
    """
    Resolve an optional image size from backbone constructor kwargs.

    Args:
        backbone_config (BackboneConfig): Backbone configuration.

    Returns:
        int | None: Parsed image size from `backbone_config.kwargs.image_size`, if present.
    """
    if backbone_config is None:
        return None

    image_size = backbone_config.kwargs.get('image_size')
    if image_size is None:
        return None

    return int(image_size)


def build_benchmark(
    *,
    experiment_config: ExperimentConfig,
    repair_split_fraction: float,
) -> NCScenario:
    """
    Build the benchmark scenario.

    Args:
        experiment_config (ExperimentConfig): Experiment configuration.
        repair_split_fraction (float): Fraction of each training experience excluded into the repair stream.

    Returns:
        NCScenario: Built benchmark.

    Raises:
        ValueError: If the scenario name is empty or not in the registered builder map.
        RuntimeError: If the scenario builder fails to create a valid NCScenario.
    """
    scenario_builder: ScenarioBuilder = get_scenario_builder(
        scenario=experiment_config.scenario
    )
    benchmark = scenario_builder(
        num_experiences=experiment_config.num_experiences,
        return_task_id=False,
        repair_split_fraction=repair_split_fraction,
        dataset_path=experiment_config.dataset_path,
        seed=experiment_config.seed,
        transform_random_resized_crop=experiment_config.transforms.random_resized_crop,
        transform_horizontal_flip=experiment_config.transforms.horizontal_flip,
        transform_image_size=_resolve_backbone_image_size(backbone_config=experiment_config.backbone),
    )
    return benchmark


#############
# Backbones #
#############


def build_backbone(
    *,
    name: str,
    num_classes: int,
    backbone_kwargs: Mapping[str, object] | None = None,
) -> torch.nn.Module:
    """
    Build the backbone model.

    Args:
        name (str): Backbone registry name.
        num_classes (int): Total number of target classes.
        backbone_kwargs (Mapping[str, object] | None): Optional constructor kwargs for the selected backbone.

    Returns:
        torch.nn.Module: Instantiated backbone model.
    """
    backbone_path = get_backbone_path(name)
    model_cls = import_symbol(backbone_path)
    if not inspect.isclass(model_cls):
        raise TypeError(f'Backbone symbol is not a class: {backbone_path}')

    constructor_kwargs: dict[str, object] = (
        dict(backbone_kwargs)
        if backbone_kwargs is not None
        else {}
    )
    model_cls_sig = inspect.signature(model_cls.__init__)
    if 'n_classes' in model_cls_sig.parameters:
        if 'n_classes' in constructor_kwargs or PARAM_NUM_CLASSES in constructor_kwargs:
            raise ValueError(
                f'Backbone `{backbone_path}` constructor kwargs must not include '
                '`n_classes` or `num_classes`.'
            )
        constructor_kwargs['n_classes'] = int(num_classes)
    elif PARAM_NUM_CLASSES in model_cls_sig.parameters:
        if 'n_classes' in constructor_kwargs or PARAM_NUM_CLASSES in constructor_kwargs:
            raise ValueError(
                f'Backbone `{backbone_path}` constructor kwargs must not include '
                '`n_classes` or `num_classes`.'
            )
        constructor_kwargs[PARAM_NUM_CLASSES] = int(num_classes)
    else:
        raise ValueError(
            f'Backbone `{backbone_path}` must accept either `n_classes` or '
            '`num_classes`.'
        )
    try:
        backbone = model_cls(**constructor_kwargs)
    except TypeError as exc:
        raise ValueError(
            f'Backbone `{backbone_path}` could not be initialized with kwargs: '
            f'{sorted(constructor_kwargs.keys())}.'
        ) from exc

    if not isinstance(backbone, torch.nn.Module):
        raise TypeError(f'Backbone `{backbone_path}` did not produce a torch.nn.Module.')
    return backbone


##############
# Strategies #
##############


def make_strategy(
    experiment_config: ExperimentConfig,
    training_config: TrainingConfig,
    strategy_config: StrategyConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    evaluator: EvaluationPlugin,
    plugins: Sequence[object],
    *,
    train_epochs_override: int | None = None,
    eval_every_override: int | None = None,
) -> BaseTemplate:
    """
    Instantiate an Avalanche strategy for the requested configuration.

    Args:
        experiment_config (ExperimentConfig): Experiment configuration shared across runs.
        training_config (TrainingConfig): Backbone training configuration.
        strategy_config (StrategyConfig): Strategy configuration for the run.
        model (torch.nn.Module): Model to train.
        optimizer (torch.optim.Optimizer): Optimizer.
        criterion (torch.nn.Module): Training loss function.
        evaluator (EvaluationPlugin): Evaluation plugin.
        plugins (Sequence[object]): Additional Avalanche plugins (e.g., checkpointing).
        train_epochs_override (int | None): Optional override for strategy `train_epochs`.
        eval_every_override (int | None): Optional override for strategy `eval_every`.

    Returns:
        BaseTemplate: Concrete Avalanche strategy ready to train.
    """
    common_kwargs = dict(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=training_config.batch_size,
        train_epochs=(
            int(train_epochs_override)
            if train_epochs_override is not None
            else training_config.num_epochs
        ),
        eval_mb_size=experiment_config.evaluation.batch_size,
        device=experiment_config.device,
        plugins=list(plugins),
        evaluator=evaluator,
        eval_every=(
            int(eval_every_override)
            if eval_every_override is not None
            else resolve_avalanche_eval_every(
                avalanche_schedule=experiment_config.evaluation.avalanche_schedule
            )
        ),
    )
    common_kwarg_names = set(common_kwargs)

    if not isinstance(strategy_config.kwargs, dict):
        raise ValueError('Strategy kwargs must be a mapping.')
    strategy_kwargs = dict(strategy_config.kwargs)

    if strategy_config.name == 'naive':
        reserved_overlap = common_kwarg_names.intersection(strategy_kwargs)
        if reserved_overlap:
            raise ValueError(
                f'Strategy kwargs should not override {sorted(reserved_overlap)}.'
            )
        return Naive(**common_kwargs, **strategy_kwargs)

    if strategy_config.name == 'replay':
        reserved_overlap = common_kwarg_names.intersection(strategy_kwargs)
        if reserved_overlap:
            raise ValueError(
                f'Strategy kwargs should not override {sorted(reserved_overlap)}.'
            )

        replay_batch_size = strategy_kwargs.pop(_PARAM_NAME_REPLAY_BATCH_SIZE_MEM, None)
        if (
            _PARAM_NAME_REPLAY_MEM_SIZE in strategy_kwargs
            and strategy_kwargs[_PARAM_NAME_REPLAY_MEM_SIZE] is not None
        ):
            strategy_kwargs[_PARAM_NAME_REPLAY_MEM_SIZE] = int(
                strategy_kwargs[_PARAM_NAME_REPLAY_MEM_SIZE]
            )
        replay_kwargs = dict(**common_kwargs, **strategy_kwargs)
        strategy = Replay(**replay_kwargs)

        replay_plugins = getattr(strategy, 'plugins', None)
        if not replay_plugins:
            raise ValueError('Replay strategy did not expose any plugins.')

        replay_plugin = next(
            (
                plugin
                for plugin in replay_plugins
                if plugin.__class__.__name__ == 'ReplayPlugin'
            ),
            None,
        )
        if replay_plugin is None or not hasattr(
            replay_plugin,
            _PARAM_NAME_REPLAY_BATCH_SIZE_MEM,
        ):
            raise ValueError(
                'Replay strategy did not expose a ReplayPlugin with batch_size_mem.'
            )

        if replay_batch_size is None:
            setattr(replay_plugin, _PARAM_NAME_REPLAY_BATCH_SIZE_MEM, None)
        else:
            setattr(
                replay_plugin,
                _PARAM_NAME_REPLAY_BATCH_SIZE_MEM,
                int(replay_batch_size),
            )
        return strategy

    if strategy_config.name == 'bic':
        raise ValueError(
            'BiC is implemented as a post-hoc repair controller in this project'
        )
    if strategy_config.name == 'il2m':
        raise ValueError(
            'IL2M is implemented as a post-hoc repair controller in this project'
        )
    raise ValueError(f'Unsupported strategy: {strategy_config.name}')


##############
# Optimizers #
##############


def build_optimizer(
    model: torch.nn.Module,
    optimizer_config: OptimizerConfig,
) -> tuple[torch.optim.Optimizer, dict[str, object]]:
    """
    Construct an optimizer instance from configuration.

    Args:
        model (torch.nn.Module): Model providing parameters.
        optimizer_config (OptimizerConfig): Optimizer configuration.

    Returns:
        tuple[torch.optim.Optimizer, dict[str, object]]:
            Tuple of optimizer instance and the keyword arguments used.
    """
    if not isinstance(optimizer_config.name, str):
        raise ValueError('Optimizer name must be a string.')
    optimizer_name = optimizer_config.name.lower()
    if not isinstance(optimizer_config.kwargs, dict):
        raise ValueError('Optimizer kwargs must be a mapping.')
    optimizer_kwargs_payload = optimizer_config.kwargs
    if optimizer_name == 'sgd':
        default_kwargs = {
            'lr': 0.1,
            'momentum': 0.9,
            'weight_decay': 5e-4,
        }
        optimizer_kwargs = {
            kwarg: float(value)
            for kwarg, value in {**default_kwargs, **optimizer_kwargs_payload}.items()
        }
        optimizer = SGD(model.parameters(), **optimizer_kwargs)
        return optimizer, optimizer_kwargs
    if optimizer_name == 'adamw':
        default_kwargs: dict[str, object] = {
            'lr': 1e-3,
            'betas': [0.9, 0.999],
            'eps': 1e-8,
            'weight_decay': 0.0,
        }
        merged_kwargs = {**default_kwargs, **optimizer_kwargs_payload}
        betas_raw = merged_kwargs.get('betas', [0.9, 0.999])
        if isinstance(betas_raw, str):
            raise ValueError(
                'AdamW optimizer `betas` must be provided as a YAML sequence like '
                '`[0.9, 0.999]`.'
            )
        if not isinstance(betas_raw, (list, tuple)):
            raise ValueError('AdamW optimizer `betas` must be a sequence of two floats.')
        betas_values = list(betas_raw)
        if len(betas_values) != 2:
            raise ValueError('AdamW optimizer `betas` must contain exactly two values.')
        betas = (float(betas_values[0]), float(betas_values[1]))
        optimizer_kwargs = {
            'lr': float(merged_kwargs['lr']),
            'betas': [betas[0], betas[1]],
            'eps': float(merged_kwargs['eps']),
            'weight_decay': float(merged_kwargs['weight_decay']),
        }
        optimizer = AdamW(
            model.parameters(),
            lr=float(optimizer_kwargs['lr']),
            betas=betas,
            eps=float(optimizer_kwargs['eps']),
            weight_decay=float(optimizer_kwargs['weight_decay']),
        )
        return optimizer, optimizer_kwargs
    raise ValueError(f'Unsupported optimizer: {optimizer_config.name}')


##################
# LR Schedulers  #
##################


def build_lr_scheduler_plugin(
    *,
    name: str,
    scheduler_kwargs: dict[str, object],
    initial_lr: float = 0.1,
    total_epochs: int | None = None,
) -> LRSchedulerPlugin:
    """
    Build an LR scheduler plugin from a registry name.

    Args:
        name (str): LR scheduler registry name.
        scheduler_kwargs (dict[str, object]): Keyword arguments for the scheduler.
        initial_lr (float): Initial learning rate to reset to each experience.
        total_epochs (int | None): Training epochs per experience when required by the scheduler.

    Returns:
        LRSchedulerPlugin: Configured LR scheduler plugin.
    """
    scheduler_path = get_lr_scheduler_path(name)
    scheduler_cls = import_symbol(scheduler_path)
    if not inspect.isclass(scheduler_cls):
        raise TypeError(f'LR scheduler symbol is not a class: {scheduler_path}')
    effective_scheduler_kwargs = dict(scheduler_kwargs)
    if str(name).strip().lower() == 'warmup_cosine':
        if total_epochs is None:
            raise ValueError('`total_epochs` is required for `warmup_cosine`.')
        effective_scheduler_kwargs['total_epochs'] = int(total_epochs)
    return LRSchedulerPlugin(
        scheduler_cls=scheduler_cls,
        scheduler_kwargs=effective_scheduler_kwargs,
        initial_lr=initial_lr,
    )


def build_gradient_clipping_plugin(
    *,
    max_norm: float,
) -> GradientClippingPlugin:
    """
    Build a gradient clipping plugin.

    Args:
        max_norm (float): Maximum gradient norm.

    Returns:
        GradientClippingPlugin: Configured gradient clipping plugin.
    """
    return GradientClippingPlugin(max_norm=max_norm)


###############
# Controllers #
###############


def build_controller(
    *,
    controller_config: ControllerConfig | None,
    train_batch_size: int,
    replay_batch_size: int | None,
    replay_memory_size: int | None,
) -> Controller | None:
    """
    Build a controller.

    Args:
        controller_config (ControllerConfig | None): Controller configuration.
        train_batch_size (int): Training batch size.
        replay_batch_size (int | None): Replay batch size.
        replay_memory_size (int | None): Replay memory size.

    Returns:
        Controller | None: Controller instance or None if no controller is configured.
    """
    if controller_config is None:
        return None

    controller_path = get_controller_path(controller_config.name)
    controller_cls = import_symbol(controller_path)
    _validate_controller_replay_requirements(
        controller_cls=controller_cls,
        replay_batch_size=replay_batch_size,
        controller_path=controller_path,
    )
    controller_cls_sig = inspect.signature(controller_cls.__init__)

    controller_kwargs = dict(controller_config.kwargs)
    injectable_kwargs = {
        _PARAM_CONTROLLER_TRAIN_BATCH_SIZE: train_batch_size,
        _PARAM_CONTROLLER_REPLAY_BATCH_SIZE: replay_batch_size,
        _PARAM_CONTROLLER_REPLAY_MEMORY_SIZE: replay_memory_size,
    }
    for param_name, param_value in injectable_kwargs.items():
        if param_value is None:
            continue
        if (
            param_name in controller_cls_sig.parameters
            and param_name not in controller_kwargs
        ):
            controller_kwargs[param_name] = param_value

    forbidden_kwargs = (
        PARAM_NUM_CLASSES,
        'n_classes',
        _PARAM_CONTROLLER_CLASSES,
        'seed',
    )
    found_forbidden_kwargs = [key for key in forbidden_kwargs if key in controller_kwargs]
    if found_forbidden_kwargs:
        raise ValueError(
            f'Controller `{controller_path}` should not receive the following '
            f'keyword arguments: {", ".join(found_forbidden_kwargs)}'
        )

    controller: Controller = controller_cls(**controller_kwargs)
    if not isinstance(controller, (PreventionController, RepairController)):
        raise ValueError(f'{controller_path} is not a prevention or repair controller.')
    return controller


def _validate_controller_replay_requirements(
    *,
    controller_cls: object,
    replay_batch_size: int | None,
    controller_path: str,
) -> None:
    """
    Validate replay requirements before controller instantiation.

    Args:
        controller_cls (object): Imported controller class candidate.
        replay_batch_size (int | None): Replay batch size resolved from strategy config.
        controller_path (str): Fully qualified controller path used for error reporting.

    Returns:
        None.

    Raises:
        ValueError: If a replay-required controller is configured without replay batch metadata.
    """
    if not inspect.isclass(controller_cls):
        return
    if not issubclass(controller_cls, PreventionController):
        return
    if not controller_cls.requires_replay():
        return
    if replay_batch_size is not None:
        return

    raise ValueError(
        f'Controller `{controller_path}` requires a replay-based strategy and replay batch size '
        '(`training.strategy.kwargs.batch_size_mem`).'
    )


def build_controller_plugin(
    *,
    controller: Controller,
    fit_after_experience: bool | None = None,
    num_epochs: int | None = None,
    batch_size: int | None = None,
    budget_fraction: float = 1.0,
    seed: int,
    debug: bool = False,
    debug_epochs: int | None = None,
    debug_experiences: int | None = None,
) -> ControllerPlugin:
    """
    Build a controller plugin.

    Args:
        controller (Controller): Controller instance.
        fit_after_experience (bool | None): Whether to fit after each experience.
        num_epochs (int | None): Number of repair epochs.
        batch_size (int | None): Repair batch size.
        budget_fraction (float): Fraction of each fixed repair set used for controller fitting.
        seed (int): Global experiment seed used for deterministic repair subset selection.
        debug (bool): Whether to use the debug repair controller plugin.
        debug_epochs (int | None): Epochs per experience for debug metric steps.
        debug_experiences (int | None): Total experiences for debug metric steps.

    Returns:
        ControllerPlugin: Controller plugin instance.
    """
    if isinstance(controller, PreventionController):
        return PreventionControllerPlugin(controller=controller)

    if isinstance(controller, RepairController):
        if fit_after_experience is None or num_epochs is None or batch_size is None:
            raise ValueError(
                'Repair controllers require `fit_after_experience`, `num_epochs`, '
                'and `batch_size` to be specified.'
            )
        if debug:
            if (
                debug_epochs is None
                or debug_experiences is None
            ):
                raise ValueError(
                    'Debug controller plugin requires debug_epochs and '
                    'debug_experiences.'
                )
            return DebugRepairControllerPlugin(
                controller=controller,
                fit_after_experience=fit_after_experience,
                repair_epochs=num_epochs,
                repair_batch_size=batch_size,
                budget_fraction=budget_fraction,
                seed=seed,
                debug_epochs=debug_epochs,
                debug_experiences=debug_experiences,
            )
        return RepairControllerPlugin(
            controller=controller,
            fit_after_experience=fit_after_experience,
            repair_epochs=num_epochs,
            repair_batch_size=batch_size,
            budget_fraction=budget_fraction,
            seed=seed,
        )

    raise ValueError(
        f'Unsupported controller plugin type: {type(controller).__name__}.'
    )
