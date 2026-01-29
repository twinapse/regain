"""
Unified experiment runner for Avalanche continual learning strategies.
"""
from dataclasses import asdict
import inspect
import json
from logging import Logger
from pathlib import Path
import tempfile
from typing import Sequence

import avalanche
from avalanche.benchmarks.scenarios import NCScenario
from avalanche.training.determinism.rng_manager import RNGManager
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import Naive
from avalanche.training.supervised import Replay
from avalanche.training.templates import BaseTemplate
import mlflow
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
import yaml

from regain.analysis.metrics import METRIC_NAMESPACE_SEPARATOR
from regain.analysis.metrics import MetricContext
from regain.avalanche_utils.logging import normalize_metric_name
from regain.avalanche_utils.plugins import ControllerPlugin
from regain.avalanche_utils.plugins import make_evaluation_plugin
from regain.avalanche_utils.plugins import MetricContextPlugin
from regain.avalanche_utils.plugins import PreventionControllerPlugin
from regain.avalanche_utils.plugins import RegainEvaluationPlugin
from regain.avalanche_utils.plugins import RepairControllerPlugin
from regain.avalanche_utils.plugins import SeenClassesMaskPlugin
from regain.avalanche_utils.scenarios import get_scenario_builder
from regain.avalanche_utils.scenarios import ScenarioBuilder
from regain.debug.avalanche_utils import DebugRepairControllerPlugin
from regain.experiments.utils import ControllerConfig
from regain.experiments.utils import count_trainable_parameters
from regain.experiments.utils import enable_determinism
from regain.experiments.utils import ExperimentConfig
from regain.experiments.utils import guard_experiment_config_overrides
from regain.experiments.utils import init_mlflow
from regain.experiments.utils import OptimizerConfig
from regain.experiments.utils import RunConfig
from regain.experiments.utils import StrategyConfig
from regain.models.classifiers import ResNet18Classifier
from regain.models.controllers import Controller
from regain.models.controllers import PreventionController
from regain.models.controllers import RepairController
from regain.registry import get_controller_path
from regain.registry import import_symbol
from regain.utils import get_logger

__all__ = [
    'run_experiment',
]


def _verify_classes(benchmark: NCScenario) -> None:
    """
    Verify that all class IDs in the benchmark are contiguous starting from zero
    (`class_ids_from_zero_from_first_exp`=True).

    Args:
        benchmark (NCScenario): Benchmark scenario.

    Returns:
        None
    """
    all_class_ids = []
    for exp in benchmark.train_stream:
        all_class_ids.extend(exp.classes_in_this_experience)

    expected = set(range(benchmark.n_classes))
    seen = set(all_class_ids)

    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(
            'Class IDs in the benchmark are not contiguous starting from zero. '
            f'missing={missing[:20]}{"..." if len(missing) > 20 else ""} '
            f'extra={extra[:20]}{"..." if len(extra) > 20 else ""}'
        )

    if len(seen) != benchmark.n_classes:
        raise ValueError(
            f'Expected {benchmark.n_classes} unique class IDs, got {len(seen)}.'
        )


def _make_strategy(
    experiment_config: ExperimentConfig,
    strategy_config: StrategyConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    evaluator: EvaluationPlugin,
    plugins: Sequence[object],
) -> BaseTemplate:
    """
    Instantiate an Avalanche strategy for the requested configuration.

    Args:
        experiment_config: Experiment configuration shared across runs.
        strategy_config: Strategy configuration for the run.
        model: Model to train.
        optimizer: Optimizer.
        criterion: Training loss function.
        evaluator: Evaluation plugin.
        plugins: Additional Avalanche plugins (e.g., checkpointing).

    Returns:
        Concrete Avalanche strategy ready to train.
    """
    # Set common params for all strategies
    common_params = dict(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=experiment_config.train_batch_size,
        train_epochs=experiment_config.num_epochs,
        eval_mb_size=experiment_config.eval_batch_size,
        device=experiment_config.device,
        plugins=list(plugins),
        evaluator=evaluator,
        eval_every=experiment_config.eval_every,
    )
    common_param_names = set(common_params)

    # Get strategy-specific parameters
    if not isinstance(strategy_config.params, dict):
        raise ValueError('Strategy params must be a mapping.')
    params = dict(strategy_config.params)

    # Instantiate requested strategy
    if strategy_config.name == 'naive':
        # Check for reserved param overrides
        reserved_overlap = common_param_names.intersection(params)
        if reserved_overlap:
            raise ValueError(f'Strategy params should not override {sorted(reserved_overlap)}.')

        # Instantiate Naive strategy
        return Naive(**common_params, **params)

    if strategy_config.name == 'replay':
        # Check for reserved param overrides
        reserved_overlap = common_param_names.union({'mem_size'}).intersection(params)
        if reserved_overlap:
            raise ValueError(f'Strategy params should not override {sorted(reserved_overlap)}.')

        # Set replay-specific params
        if experiment_config.replay_batch_size is None:
            raise ValueError('Replay strategy requires experiment_config.replay_batch_size to be set.')
        replay_kwargs = dict(mem_size=experiment_config.replay_memory_size, **common_params, **params)

        # Instantiate Replay strategy
        strategy = Replay(**replay_kwargs)

        # Adjust the replay plugin's batch size
        replay_plugins = getattr(strategy, 'plugins', None)
        if not replay_plugins:
            raise ValueError('Replay strategy did not expose any plugins.')

        replay_plugin = next(
            (plugin for plugin in replay_plugins if plugin.__class__.__name__ == 'ReplayPlugin'),
            None,
        )
        if replay_plugin is None or not hasattr(replay_plugin, 'batch_size_mem'):
            raise ValueError('Replay strategy did not expose a ReplayPlugin with batch_size_mem.')

        replay_plugin.batch_size_mem = experiment_config.replay_batch_size

        return strategy

    if strategy_config.name == 'bic':
        raise ValueError('BiC is implemented as a post-hoc repair controller in this project')

    if strategy_config.name == 'il2m':
        raise ValueError('IL2M is implemented as a post-hoc repair controller in this project')

    raise ValueError(f'Unsupported strategy: {strategy_config.name}')


def _build_optimizer(
    model: torch.nn.Module,
    optimizer_config: OptimizerConfig,
) -> tuple[torch.optim.Optimizer, dict[str, object]]:
    """
    Construct an optimizer instance from configuration.

    Args:
        model: Model providing parameters.
        optimizer_config: Optimizer configuration.

    Returns:
        Tuple of optimizer instance and the keyword arguments used.
    """
    if not isinstance(optimizer_config.name, str):
        raise ValueError('Optimizer name must be a string.')
    optimizer_name = optimizer_config.name.lower()
    if not isinstance(optimizer_config.params, dict):
        raise ValueError('Optimizer params must be a mapping.')
    params = optimizer_config.params
    if optimizer_name == 'sgd':
        default_kwargs = {'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4}
        optimizer_kwargs = {kwarg: float(value) for kwarg, value in {**default_kwargs, **params}.items()}
        optimizer = SGD(model.parameters(), **optimizer_kwargs)
        return optimizer, optimizer_kwargs
    raise ValueError(f'Unsupported optimizer: {optimizer_config.name}')


def _build_controller(
    *,
    controller_config: ControllerConfig,
    train_batch_size: int,
    replay_batch_size: int,
    replay_memory_size: int,
) -> Controller | None:
    """
    Build a controller.

    Args:
        controller_config (ControllerConfig): Controller configuration.
        train_batch_size (int): Training batch size.
        replay_batch_size (int): Replay batch size.
        replay_memory_size (int): Replay memory size.

    Returns:
        Controller | None: Controller instance or None if no controller is configured.
    """
    # Early exit if no controller is configured
    if controller_config is None:
        return None

    # Import and inspect the controller class
    controller_path = get_controller_path(controller_config.name)
    controller_cls = import_symbol(controller_path)
    controller_cls_sig = inspect.signature(controller_cls.__init__)

    # Set controller parameters
    controller_params = dict(controller_config.params)  # Make a copy to avoid mutating the original
    injectable_params = {
        'train_batch_size': train_batch_size,
        'replay_batch_size': replay_batch_size,
        'replay_memory_size': replay_memory_size,
    }
    for param_name, param_value in injectable_params.items():
        if param_value is None:
            continue
        if param_name in controller_cls_sig.parameters and param_name not in controller_params:
            controller_params[param_name] = param_value

    # Check for invalid controller parameters
    invalid_params = ('num_classes', 'n_classes', 'classes')
    found_invalid_params = [p for p in invalid_params if p in controller_params]
    if found_invalid_params:
        raise ValueError(
            f'Controller `{controller_path}` should not receive the following parameters: '
            f'{", ".join(found_invalid_params)}'
        )

    # Instantiate and validate the controller
    controller: Controller = controller_cls(**controller_params)
    if not isinstance(controller, (PreventionController, RepairController)):
        raise ValueError(f'{controller_path} is not a prevention or repair controller.')

    return controller


def _build_controller_plugin(
    *,
    controller: Controller,
    fit_after_experience: bool | None = None,
    repair_epochs: int | None = None,
    repair_batch_size: int | None = None,
    debug: bool = False,
    debug_epochs: int | None = None,
    debug_experiences: int | None = None,
    debug_seed: int | None = None,
) -> ControllerPlugin:
    """
    Build a controller plugin.

    Args:
        controller (Controller): Controller instance.
        fit_after_experience (bool | None): Whether to fit after each experience (only for repair controllers).
        repair_epochs (int | None): Number of repair epochs (only for repair controllers).
        repair_batch_size (int | None): Repair batch size (only for repair controllers).
        debug (bool): Whether to use the debug repair controller plugin.
        debug_epochs (int | None): Epochs per experience used only to compute debug metric step values.
        debug_experiences (int | None): Total number of experiences used only to compute debug metric step values.
        debug_seed (int | None): Seed used for debug dataloader ordering (debug-only).

    Returns:
        ControllerPlugin: Controller plugin instance.
    """
    if isinstance(controller, PreventionController):
        controller_plugin = PreventionControllerPlugin(controller=controller)
    elif isinstance(controller, RepairController):
        # Validate required parameters for repair controllers
        if fit_after_experience is None or repair_epochs is None or repair_batch_size is None:
            raise ValueError(
                'Repair controllers require `fit_after_experience`, `repair_epochs`, '
                'and `repair_batch_size` to be specified.'
            )
        # Build the repair controller plugin
        if debug:
            if debug_epochs is None or debug_experiences is None or debug_seed is None:
                raise ValueError('Debug controller plugin requires debug_epochs, debug_experiences, debug_seed.')
            controller_plugin = DebugRepairControllerPlugin(
                controller=controller,
                fit_after_experience=fit_after_experience,
                repair_epochs=repair_epochs,
                repair_batch_size=repair_batch_size,
                debug_epochs=debug_epochs,
                debug_experiences=debug_experiences,
                debug_seed=debug_seed,
            )
        else:
            controller_plugin = RepairControllerPlugin(
                controller=controller,
                fit_after_experience=fit_after_experience,
                repair_epochs=repair_epochs,
                repair_batch_size=repair_batch_size,
            )
    else:
        controller_plugin = None
    return controller_plugin


def _log_run_params(
    *,
    experiment_config: ExperimentConfig,
    run_config: RunConfig,
    deterministic_algorithms_enabled: bool,
    optimizer_params: dict[str, object],
    controller_model_param_count: int | None = None,
    num_classes: int | None = None,
    debug_skip_reason: str | None = None,
) -> None:
    """
    Log common parameters to MLflow for a run.

    Args:
        experiment_config: Experiment configuration.
        run_config: Run configuration.
        deterministic_algorithms_enabled: Whether deterministic algorithms were enabled.
        optimizer_params: Effective optimizer parameters.
        controller_model_param_count: Number of trainable parameters in the controller model (if any).
        num_classes: Total number of benchmark classes (optional).
        debug_skip_reason: Optional debug skip reason (debug-only).
    """
    # Basic run params
    run_params = {
        'run_name': run_config.run_name,
        'scenario': experiment_config.scenario,
        'num_experiences': experiment_config.num_experiences,
        'num_epochs': experiment_config.num_epochs,
        'replay_memory_size': experiment_config.replay_memory_size,
        'replay_batch_size': experiment_config.replay_batch_size,
        'train_batch_size': experiment_config.train_batch_size,
        'eval_batch_size': experiment_config.eval_batch_size,
        'eval_every': experiment_config.eval_every,
        'eval_mode': run_config.eval_mode,
        'strategy_name': run_config.strategy.name,
        'strategy_params': json.dumps(run_config.strategy.params, default=str),
        'optimizer_name': run_config.optimizer.name,
        'optimizer_params': json.dumps(optimizer_params, default=str),
        'seed': experiment_config.seed,
        'deterministic': experiment_config.deterministic,
        'debug': experiment_config.debug,
        'device': experiment_config.device,
    }

    # Optional dataset path
    if experiment_config.dataset_path is not None:
        run_params['dataset_path'] = str(experiment_config.dataset_path)

    # Repair config (if present)
    for key in [
        'repair_after_experience',
        'repair_budget_per_class',
        'repair_epochs',
        'repair_batch_size',
    ]:
        if hasattr(experiment_config, key):
            v = getattr(experiment_config, key)
            if v is not None:
                run_params[key] = v

    # Benchmark metadata
    if num_classes is not None:
        run_params['num_classes'] = int(num_classes)

    # Debug metadata
    if debug_skip_reason is not None:
        run_params['debug_skip_reason'] = debug_skip_reason

    # Controller config
    if run_config.controller is not None:
        controller_name = run_config.controller.name
        controller_path = get_controller_path(controller_name)
        run_params['controller_name'] = controller_name
        run_params['controller_path'] = controller_path
        run_params['controller_params'] = json.dumps(run_config.controller.params, default=str)

    # Controller model parameter count
    if controller_model_param_count is not None:
        run_params['controller_model_param_count'] = int(controller_model_param_count)

    # Log all params to MLflow
    mlflow.log_params(run_params)
    mlflow.log_param('avalanche_version', avalanche.__version__)
    mlflow.log_param('torch_deterministic_algorithms', deterministic_algorithms_enabled)


# TODO: This function is too long and does too many things. Divide it into smaller functions.
def _train_and_evaluate_strategy(
    experiment_config: ExperimentConfig,
    run_config: RunConfig,
) -> tuple[BaseTemplate, NCScenario, dict[str, float]]:
    """
    Train and evaluate a single strategy run.

    Args:
        experiment_config: Experiment configuration shared across runs.
        run_config: Run-specific configuration.

    Returns:
        Tuple of (strategy, benchmark, final evaluation results).
    """
    # Guard against unsupported configuration overrides
    guard_experiment_config_overrides(asdict(experiment_config))

    # Set random seeds and enable determinism if requested
    RNGManager.set_random_seeds(experiment_config.seed)
    deterministic_algorithms_enabled = False
    if experiment_config.deterministic:
        deterministic_algorithms_enabled = enable_determinism()

    # Initialize MLflow experiment and run
    with init_mlflow(
        experiment_name=experiment_config.experiment_name,
        run_name=run_config.run_name,
        tracking_uri=experiment_config.mlflow_tracking_uri,
    ):
        with tempfile.TemporaryDirectory() as artifacts_dir:
            # Instantiate and validate the scenario builder
            scenario_builder: ScenarioBuilder = get_scenario_builder(scenario=experiment_config.scenario)

            # Instantiate and validate the controller
            controller_config = run_config.controller
            controller: Controller | None = _build_controller(
                controller_config=controller_config,
                train_batch_size=experiment_config.train_batch_size,
                replay_batch_size=experiment_config.replay_batch_size,
                replay_memory_size=experiment_config.replay_memory_size,
            )

            # Check debug mode (supported only for repair controllers)
            debug_skip_reason: str | None = None
            if experiment_config.debug and not isinstance(controller, RepairController):
                debug_skip_reason = 'no_repair_controller'

            # Validate strategy-controller compatibility
            if isinstance(controller, PreventionController):
                # Check if the controller requires a replay-based strategy
                if controller.requires_replay() and run_config.strategy.name != 'replay':
                    raise ValueError(
                        f'Controller `{type(controller).__name__}` requires a replay-based strategy.'
                    )
                # Add additional validations here if needed

            # Validate evaluation mode
            if run_config.eval_mode == 'compare':
                if controller is None:
                    raise ValueError('eval_mode="compare" requires a controller.')
                if not isinstance(controller, RepairController):
                    raise ValueError('eval_mode="compare" requires a toggleable repair controller.')

            # Get the per-class repair budget (only for controllers using dedicated repair data)
            repair_budget_per_class = (
                experiment_config.repair_budget_per_class if isinstance(controller, RepairController) else 0
            )

            # Build the benchmark scenario
            benchmark = scenario_builder(
                num_experiences=experiment_config.num_experiences,
                return_task_id=False,  # Class-incremental learning
                repair_budget_per_class=repair_budget_per_class,
                dataset_path=experiment_config.dataset_path,
                seed=experiment_config.seed,
            )

            # Verify that all class IDs are contiguous (`class_ids_from_zero_from_first_exp=True`)
            _verify_classes(benchmark)

            # Initialize the metric context and build its plugin
            context = MetricContext()
            context_plugin = MetricContextPlugin(context=context)

            # Build the evaluation plugin
            eval_plugin = make_evaluation_plugin(
                context=context,
                keep_timestep_results=True,
                log_to_console=True,
                log_to_mlflow=True,
            )

            # Build the seen classes mask plugin
            seen_mask_plugin = SeenClassesMaskPlugin()

            # Build the controller plugin
            if controller_config is not None:
                controller_plugin: ControllerPlugin = _build_controller_plugin(
                    controller=controller,
                    fit_after_experience=experiment_config.repair_after_experience,
                    repair_epochs=controller_config.repair_epochs or experiment_config.num_epochs,
                    repair_batch_size=controller_config.repair_batch_size or experiment_config.train_batch_size,
                    debug=experiment_config.debug,
                    debug_epochs=experiment_config.num_epochs,
                    debug_experiences=experiment_config.num_experiences,
                    debug_seed=experiment_config.seed,
                )
            else:
                controller_plugin: ControllerPlugin | None = None

            # Build the evaluation plugin
            regain_evaluation_plugin = RegainEvaluationPlugin(
                benchmark=benchmark,
                controller_plugin=controller_plugin,
                eval_mode=run_config.eval_mode,
                repair_after_experience=experiment_config.repair_after_experience,
                seen_mask_plugin=seen_mask_plugin,
                num_epochs_per_experience=experiment_config.num_epochs,
                context=context,
                eps=1e-4,
            )

            # Save the plugins that will be used in the strategy
            strategy_plugins = [context_plugin, seen_mask_plugin]
            if controller_plugin is not None:
                strategy_plugins.append(controller_plugin)
            strategy_plugins.append(regain_evaluation_plugin)

            # Build the model
            model = ResNet18Classifier(n_classes=benchmark.n_classes)
            model.to(experiment_config.device)

            # Initialize the controller's parameters (only for repair controllers)
            if isinstance(controller_plugin, RepairControllerPlugin):
                if len(benchmark.train_stream) > 0:
                    probe_dataset = benchmark.train_stream[0].dataset
                else:
                    probe_dataset = None
                controller_plugin.initialize_parameters(model=model, dataset=probe_dataset)

            # Count controller model parameters
            controller_model_param_count = count_trainable_parameters(controller)

            # Build the optimizer and criterion
            optimizer, optimizer_params = _build_optimizer(model=model, optimizer_config=run_config.optimizer)
            criterion = CrossEntropyLoss()

            # Build the strategy
            strategy = _make_strategy(
                experiment_config=experiment_config,
                strategy_config=run_config.strategy,
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                evaluator=eval_plugin,
                plugins=strategy_plugins,
            )

            setattr(strategy, '_regain_metric_context', context)

            # Log run parameters to MLflow
            _log_run_params(
                experiment_config=experiment_config,
                run_config=run_config,
                deterministic_algorithms_enabled=deterministic_algorithms_enabled,
                optimizer_params=optimizer_params,
                controller_model_param_count=controller_model_param_count,
                num_classes=benchmark.n_classes,
                debug_skip_reason=debug_skip_reason,
            )

            # Train and evaluate the strategy
            strategy.train(
                experiences=benchmark.train_stream,
                eval_streams=[benchmark.test_stream],
            )

            eval_scalar_results = regain_evaluation_plugin.last_posthoc_scalar_results
            if eval_scalar_results is None:
                raise RuntimeError('Posthoc evaluation results missing from evaluation plugin.')

            # Log the final evaluation metrics to MLflow
            final_step = int(experiment_config.num_experiences * experiment_config.num_epochs)
            if mlflow.active_run() is not None:
                for metric_name, value in (eval_scalar_results or {}).items():
                    safe_name = normalize_metric_name(metric_name)
                    mlflow.log_metric(f'summary{METRIC_NAMESPACE_SEPARATOR}{safe_name}', float(value), step=final_step)

            # Log the configuration used for the run to MLflow
            config_path = Path(artifacts_dir) / 'config.yaml'

            with config_path.open('w', encoding='utf-8') as f:
                dumped_config = asdict(experiment_config)

                # Note: We exclude `dataset_path` and `mlflow_tracking_uri` because they are environment-specific
                dumped_config.pop('dataset_path')
                dumped_config.pop('mlflow_tracking_uri')
                dumped_config.pop('runs_config')

                dumped_config['run'] = asdict(run_config)

                yaml.safe_dump(data=dumped_config, stream=f, sort_keys=False)

            mlflow.log_artifact(str(config_path))

            # Return the trained strategy, benchmark, and evaluation results
            return strategy, benchmark, eval_scalar_results


def run_experiment(experiment_config: ExperimentConfig) -> dict[str, dict[str, float]]:
    """
    Execute one or more runs defined in the experiment configuration.

    Args:
        experiment_config (ExperimentConfig): Experiment configuration describing all runs.

    Returns:
        dict: Mapping from run name to scalar evaluation metrics.
    """
    logger = get_logger()

    if experiment_config.deterministic:
        # When enabled, we get the following error:
        #
        # RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or
        # `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses
        # CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment
        # variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or
        # CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to
        # https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
        raise ValueError('Deterministic mode is not supported yet')

    results: dict[str, dict[str, float]] = {}

    for run_config in experiment_config.runs_config:
        try:
            _, _, eval_results = _train_and_evaluate_strategy(
                experiment_config=experiment_config,
                run_config=run_config,
            )
        except Exception:
            if logger is not None:
                logger.exception('Error during run: %s', run_config.run_name)
            continue
        results[run_config.run_name] = eval_results

    return results
