"""
Unified experiment runner for Avalanche continual learning strategies.
"""
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile

from avalanche.benchmarks.scenarios import NCScenario
from avalanche.training.determinism.rng_manager import RNGManager
from avalanche.training.templates import BaseTemplate
import mlflow
from mlflow.tracking import MlflowClient
from torch.nn import CrossEntropyLoss
import yaml

from regain.analysis.metrics import MetricContext
from regain.avalanche_utils.plugins import BackboneCheckpointLoaderPlugin
from regain.avalanche_utils.plugins import BackboneCheckpointWriterPlugin
from regain.avalanche_utils.plugins import ControllerPlugin
from regain.avalanche_utils.plugins import make_evaluation_plugin
from regain.avalanche_utils.plugins import MetricContextPlugin
from regain.avalanche_utils.plugins import RegainEvaluationPlugin
from regain.avalanche_utils.plugins import RepairControllerPlugin
from regain.avalanche_utils.plugins import SeenClassesMaskPlugin
from regain.constants import MLFLOW_ARTIFACT_BACKBONE_CHECKPOINTS_DIR
from regain.constants import MLFLOW_ARTIFACT_CONFIG_FILE
from regain.constants import NS_SEP
from regain.constants import PARAM_BACKBONE_REPLAY_BATCH_SIZE_MEM
from regain.constants import PARAM_BACKBONE_REPLAY_MEM_SIZE
from regain.constants import PARAM_CONTROLLER_MODEL_PARAM_COUNT
from regain.constants import RUN_NAME_BACKBONE
from regain.experiments.backbone import extract_backbone_analysis_baseline
from regain.experiments.backbone import extract_backbone_name_from_run
from regain.experiments.backbone import extract_backbone_training_config_from_run
from regain.experiments.backbone import load_backbone_from_existing_run
from regain.experiments.backbone import load_backbone_from_source_experiment
from regain.experiments.backbone import resolve_local_backbone_run
from regain.experiments.builders import build_backbone
from regain.experiments.builders import build_benchmark
from regain.experiments.builders import build_controller
from regain.experiments.builders import build_controller_plugin
from regain.experiments.builders import build_optimizer
from regain.experiments.builders import make_strategy
from regain.experiments.config import BackboneConfig
from regain.experiments.config import ControllerConfig
from regain.experiments.config import ExperimentConfig
from regain.experiments.config import guard_experiment_config_overrides
from regain.experiments.config import RunConfig
from regain.experiments.config import TrainingConfig
from regain.experiments.logging import log_dataset_indices
from regain.experiments.logging import log_run_params
from regain.experiments.logging import log_summary_metrics
from regain.experiments.utils import count_parameters
from regain.experiments.utils import enable_determinism
from regain.experiments.utils import resolve_backbone_training_config
from regain.experiments.utils import resolve_controller_type
from regain.mlflow_utils import init_mlflow
from regain.mlflow_utils import resolve_experiment_id
from regain.mlflow_utils import set_tracking_uri
from regain.models.controllers import Controller
from regain.models.controllers import PreventionController
from regain.models.controllers import RepairController
from regain.utils import get_logger

__all__ = [
    'run_experiment',
]


@dataclass
class _InternalRunConfig:
    """
    Internal run configuration for reserved system-managed runs.

    Attributes:
        name: Identifier for the run.
        controller: Optional controller configuration.
    """

    name: str
    controller: ControllerConfig | None = None


# TODO: This function is too long and does too many things. Divide it into smaller functions.
def _train_and_evaluate_strategy(
    experiment_config: ExperimentConfig,
    run_config: RunConfig | _InternalRunConfig,
    *,
    backbone_checkpoint_paths: Sequence[Path] | None = None,
    backbone_analysis_baseline: Mapping[str, Sequence[float]] | None = None,
    checkpoint_dir: Path | None = None,
    log_checkpoint_artifacts: bool = False,
    budget_per_class_override: int | None = None,
    backbone_source_experiment_id: str | None = None,
    backbone_source_experiment_name: str | None = None,
) -> tuple[BaseTemplate, NCScenario, dict[str, float], list[Path] | None]:
    """
    Train and evaluate a single strategy run.

    Args:
        experiment_config: Experiment configuration shared across runs.
        run_config: Run-specific configuration.
        backbone_checkpoint_paths: Optional backbone checkpoint paths for repair-controller runs.
        backbone_analysis_baseline: Optional controller-off baseline vectors from the reserved backbone run.
        checkpoint_dir: Optional output directory for one checkpoint per experience.
        log_checkpoint_artifacts: Whether to log checkpoints to MLflow artifacts.
        budget_per_class_override: Optional per-class repair budget override for scenario creation.
        backbone_source_experiment_id: Optional source experiment id to log for controller runs.
        backbone_source_experiment_name: Optional source experiment name snapshot to log for controller runs.

    Returns:
        Tuple of (strategy, benchmark, final evaluation results, checkpoint paths).
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
        run_name=run_config.name,
        tracking_uri=experiment_config.mlflow_tracking_uri,
        artifact_uri=experiment_config.mlflow_artifact_uri,
    ):
        with tempfile.TemporaryDirectory() as artifacts_dir:
            # Validate backbone checkpoint usage
            use_backbone_checkpoints = backbone_checkpoint_paths is not None
            backbone_training = resolve_backbone_training_config(
                experiment_config=experiment_config,
                use_backbone_checkpoints=use_backbone_checkpoints,
            )

            # Get replay strategy parameters if applicable (used for controller construction)
            replay_batch_size: int | None = None
            replay_memory_size: int | None = None
            if backbone_training.strategy.name == 'replay':
                strategy_kwargs = backbone_training.strategy.kwargs
                replay_memory_value = strategy_kwargs.get(
                    PARAM_BACKBONE_REPLAY_MEM_SIZE.rsplit(NS_SEP, 1)[-1],
                    200,
                )
                replay_batch_value = strategy_kwargs.get(
                    PARAM_BACKBONE_REPLAY_BATCH_SIZE_MEM.rsplit(NS_SEP, 1)[-1]
                )
                replay_memory_size = (
                    int(replay_memory_value)
                    if replay_memory_value is not None
                    else None
                )
                replay_batch_size = (
                    int(replay_batch_value)
                    if replay_batch_value is not None
                    else int(backbone_training.batch_size)
                )

            # Instantiate and validate the controller
            controller_config = run_config.controller
            controller: Controller | None = build_controller(
                controller_config=controller_config,
                train_batch_size=backbone_training.batch_size,
                replay_batch_size=replay_batch_size,
                replay_memory_size=replay_memory_size,
            )

            # Check debug mode (supported only for repair controllers)
            debug_skip_reason: str | None = None
            if experiment_config.debug and not isinstance(controller, RepairController):
                debug_skip_reason = 'no_repair_controller'

            backbone_checkpoint_paths_local: list[Path] | None = None
            if use_backbone_checkpoints:
                if not isinstance(controller, RepairController):
                    raise ValueError(
                        'Backbone checkpoints can only be used with repair-controller runs.'
                    )
                backbone_checkpoint_paths_local = [Path(path) for path in backbone_checkpoint_paths]

            # Validate strategy-controller compatibility
            if isinstance(controller, PreventionController):
                # Check if the controller requires a replay-based strategy
                if controller.requires_replay() and backbone_training.strategy.name != 'replay':
                    raise ValueError(
                        f'Controller `{type(controller).__name__}` requires a replay-based strategy.'
                    )
                # Add additional validations here if needed

            # Get the per-class repair budget
            if budget_per_class_override is not None:
                budget_per_class = int(budget_per_class_override)
            else:
                budget_per_class = experiment_config.repair.budget_per_class

            # Build the benchmark scenario
            benchmark = build_benchmark(
                experiment_config=experiment_config,
                budget_per_class=budget_per_class,
            )

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
                repair_num_epochs = experiment_config.repair.num_epochs
                repair_batch_size = experiment_config.repair.batch_size
                fit_after_experience = experiment_config.repair.fit_schedule == 'per_experience'

                controller_plugin: ControllerPlugin = build_controller_plugin(
                    controller=controller,
                    fit_after_experience=fit_after_experience,
                    num_epochs=(
                        int(repair_num_epochs)
                        if repair_num_epochs is not None
                        else int(backbone_training.num_epochs)
                    ),
                    batch_size=(
                        int(repair_batch_size)
                        if repair_batch_size is not None
                        else int(backbone_training.batch_size)
                    ),
                    debug=experiment_config.debug,
                    debug_epochs=backbone_training.num_epochs,
                    debug_experiences=experiment_config.num_experiences,
                    debug_seed=experiment_config.seed,
                )
            else:
                controller_plugin: ControllerPlugin | None = None

            # Build the evaluation plugin
            regain_evaluation_plugin = RegainEvaluationPlugin(
                benchmark=benchmark,
                controller_plugin=controller_plugin,
                repair_after_experience=experiment_config.repair.fit_schedule == 'per_experience',
                seen_mask_plugin=seen_mask_plugin,
                num_epochs_per_experience=backbone_training.num_epochs,
                context=context,
                backbone_analysis_baseline=backbone_analysis_baseline,
                eps=1e-4,
            )

            # Save the plugins that will be used in the strategy
            strategy_plugins = [context_plugin, seen_mask_plugin]
            checkpoint_writer_plugin: BackboneCheckpointWriterPlugin | None = None
            if checkpoint_dir is not None:
                checkpoint_writer_plugin = BackboneCheckpointWriterPlugin(checkpoint_dir=checkpoint_dir)
                strategy_plugins.append(checkpoint_writer_plugin)
            if use_backbone_checkpoints and backbone_checkpoint_paths_local is not None:
                strategy_plugins.append(
                    BackboneCheckpointLoaderPlugin(
                        checkpoint_paths=backbone_checkpoint_paths_local,
                    )
                )
            if controller_plugin is not None:
                strategy_plugins.append(controller_plugin)
            strategy_plugins.append(regain_evaluation_plugin)

            # Build the backbone model
            backbone_name = (
                experiment_config.backbone.name
                if experiment_config.backbone is not None
                else None
            )
            if not isinstance(backbone_name, str) or backbone_name.strip() == '':
                raise RuntimeError(
                    'Backbone name must be resolved before strategy construction.'
                )
            backbone = build_backbone(
                name=backbone_name,
                num_classes=benchmark.n_classes,
            )
            backbone.to(experiment_config.device)

            # Initialize the controller's parameters (only for repair controllers)
            if isinstance(controller_plugin, RepairControllerPlugin):
                if len(benchmark.train_stream) > 0:
                    probe_dataset = benchmark.train_stream[0].dataset
                else:
                    probe_dataset = None
                controller_plugin.initialize_parameters(model=backbone, dataset=probe_dataset)

            # Build the optimizer and criterion
            optimizer, optimizer_kwargs = build_optimizer(
                model=backbone,
                optimizer_config=backbone_training.optimizer,
            )
            criterion = CrossEntropyLoss()

            # Build the strategy
            strategy = make_strategy(
                experiment_config=experiment_config,
                training_config=backbone_training,
                strategy_config=backbone_training.strategy,
                model=backbone,
                optimizer=optimizer,
                criterion=criterion,
                evaluator=eval_plugin,
                plugins=strategy_plugins,
                train_epochs_override=0 if use_backbone_checkpoints else None,
            )

            setattr(strategy, '_regain_metric_context', context)

            # Log run parameters to MLflow
            log_run_params(
                experiment_config=experiment_config,
                run_config_payload=asdict(run_config),
                controller_name=(
                    run_config.controller.name
                    if run_config.controller is not None
                    else None
                ),
                deterministic_algorithms_enabled=deterministic_algorithms_enabled,
                optimizer_kwargs=optimizer_kwargs,
                include_backbone_params=run_config.controller is None,
                backbone_source_experiment_id=backbone_source_experiment_id,
                backbone_source_experiment_name=backbone_source_experiment_name,
                num_classes=benchmark.n_classes,
                debug_skip_reason=debug_skip_reason,
            )

            # Train and evaluate the strategy
            strategy.train(
                experiences=benchmark.train_stream,
                eval_streams=[benchmark.test_stream],
            )

            checkpoint_paths: list[Path] | None = None
            if checkpoint_writer_plugin is not None:
                checkpoint_paths = checkpoint_writer_plugin.checkpoint_paths(
                    expected_count=len(benchmark.train_stream)
                )

            if log_checkpoint_artifacts:
                if checkpoint_dir is None:
                    raise ValueError('Checkpoint artifacts requested but no checkpoint directory was provided.')
                mlflow.log_artifacts(str(checkpoint_dir), artifact_path=MLFLOW_ARTIFACT_BACKBONE_CHECKPOINTS_DIR)

            # Count and log controller model parameters after training because some controllers
            # may materialize additional parameters dynamically during fitting.
            if controller is not None:
                controller_model_param_count = count_parameters(controller)
                mlflow.log_param(PARAM_CONTROLLER_MODEL_PARAM_COUNT, int(controller_model_param_count))

            # Log the final evaluation metrics to MLflow
            eval_scalar_results = regain_evaluation_plugin.last_posthoc_scalar_results
            if eval_scalar_results is None:
                raise RuntimeError('Posthoc evaluation results missing from evaluation plugin.')

            final_step = int(experiment_config.num_experiences * backbone_training.num_epochs)
            log_summary_metrics(
                summary_metrics=eval_scalar_results,
                step=final_step,
            )

            # Log the configuration used for the run to MLflow
            config_path = Path(artifacts_dir) / MLFLOW_ARTIFACT_CONFIG_FILE

            with config_path.open('w', encoding='utf-8') as f:
                dumped_config = asdict(experiment_config)

                # Note: We exclude `dataset_path`, `mlflow_tracking_uri`, and `mlflow_artifact_uri`
                #       because they are environment-specific
                dumped_config.pop('dataset_path')
                dumped_config.pop('mlflow_tracking_uri')
                dumped_config.pop('mlflow_artifact_uri')
                dumped_config.pop('runs')

                dumped_config['run'] = asdict(run_config)

                yaml.safe_dump(data=dumped_config, stream=f, sort_keys=False)

            mlflow.log_artifact(str(config_path))

            # Log per-experience dataset indices for reproducibility checks
            log_dataset_indices(benchmark=benchmark, artifacts_dir=Path(artifacts_dir))

            # Return the trained strategy, benchmark, and evaluation results
            return strategy, benchmark, eval_scalar_results, checkpoint_paths


def _run_backbone_pretraining_run(
    *,
    experiment_config: ExperimentConfig,
    checkpoint_dir: Path,
) -> tuple[list[Path], dict[str, float], dict[str, list[float]]]:
    """
    Execute the dedicated backbone pretraining run.

    Args:
        experiment_config (ExperimentConfig): Experiment configuration.
        checkpoint_dir (Path): Directory where one checkpoint per experience will be saved.

    Returns:
        tuple[list[Path], dict[str, float], dict[str, list[float]]]:
            (checkpoint paths, scalar evaluation metrics, backbone analysis baseline vectors).
    """
    # Internal-only run: this is the single controller-off run in the pipeline.
    backbone_run_config = _InternalRunConfig(name=RUN_NAME_BACKBONE, controller=None)
    strategy, _, eval_results, checkpoint_paths = _train_and_evaluate_strategy(
        experiment_config=experiment_config,
        run_config=backbone_run_config,
        checkpoint_dir=checkpoint_dir,
        log_checkpoint_artifacts=bool(experiment_config.checkpoints_enabled),
        budget_per_class_override=experiment_config.repair.budget_per_class,
    )
    if checkpoint_paths is None:
        raise RuntimeError('Backbone run did not produce checkpoints.')
    analysis_baseline = extract_backbone_analysis_baseline(
        strategy=strategy,
        expected_num_experiences=experiment_config.num_experiences,
    )
    return checkpoint_paths, eval_results, analysis_baseline


# TODO: This function is too long and does too many things. Divide it into smaller functions.
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

    run_configs = list(experiment_config.runs) if experiment_config.runs is not None else []
    results: dict[str, dict[str, float]] = {}
    run_entries: list[tuple[RunConfig, str]] = []
    has_repair_runs = False
    for run_config in run_configs:
        if run_config.name == RUN_NAME_BACKBONE:
            raise ValueError(
                f"Run name '{RUN_NAME_BACKBONE}' is reserved and cannot be used in runs."
            )
        if run_config.controller is None:
            raise ValueError(f'Run `{run_config.name}` is missing `controller`.')
        try:
            controller_type = resolve_controller_type(run_config.controller)
        except Exception:
            if logger is not None:
                logger.exception('Error resolving controller type for run: %s', run_config.name)
            continue
        run_entries.append((run_config, controller_type))
        has_repair_runs = has_repair_runs or controller_type == 'repair'

    set_tracking_uri(tracking_uri=experiment_config.mlflow_tracking_uri)
    mlflow_client = MlflowClient()

    local_backbone_run = resolve_local_backbone_run(
        client=mlflow_client,
        experiment_name=experiment_config.experiment_name,
    )
    backbone_config = experiment_config.backbone
    if backbone_config is None and local_backbone_run is None:
        raise ValueError(
            'Experiment config must define a non-null `backbone` when no local `backbone` run exists.'
        )

    source_experiment = (
        backbone_config.source_experiment
        if backbone_config is not None
        else None
    )
    source_experiment_id_for_logging: str | None = None
    source_experiment_name_for_logging: str | None = None
    backbone_training = (
        backbone_config.training
        if backbone_config is not None
        else None
    )
    non_repair_run_names = [
        run_config.name
        for run_config, controller_type in run_entries
        if controller_type != 'repair'
    ]

    backbone_checkpoint_dir = Path(tempfile.mkdtemp(prefix='regain_backbone_'))
    backbone_checkpoint_paths: list[Path] | None = None
    backbone_eval_results: dict[str, float] | None = None
    backbone_analysis_baseline: dict[str, list[float]] | None = None
    local_backbone_name: str | None = None
    local_backbone_training: TrainingConfig | None = None
    try:
        if source_experiment:
            try:
                current_experiment_id = resolve_experiment_id(
                    client=mlflow_client,
                    experiment=experiment_config.experiment_name,
                )
                source_experiment_id = resolve_experiment_id(
                    client=mlflow_client,
                    experiment=source_experiment,
                )
            except ValueError:
                pass
            else:
                if source_experiment_id == current_experiment_id:
                    raise ValueError(
                        '`backbone.source_experiment` must be different from the current experiment.'
                    )
        if local_backbone_run is not None and backbone_config is not None:
            raise RuntimeError(
                f'Experiment `{experiment_config.experiment_name}` already has a local `backbone` run. '
                'Providing a non-null `backbone` config is not allowed when a local `backbone` run exists.'
            )

        if source_experiment:
            (
                backbone_checkpoint_paths,
                backbone_eval_results,
                backbone_analysis_baseline,
                source_backbone_run,
            ) = load_backbone_from_source_experiment(
                client=mlflow_client,
                source_experiment=source_experiment,
                checkpoint_dir=backbone_checkpoint_dir,
                expected_num_experiences=experiment_config.num_experiences,
            )
            local_backbone_name = extract_backbone_name_from_run(
                run=source_backbone_run
            )
            source_experiment_id_for_logging = str(source_backbone_run.info.experiment_id)
            source_experiment_entity = mlflow_client.get_experiment(
                experiment_id=source_experiment_id_for_logging
            )
            source_experiment_name_for_logging = (
                str(source_experiment_entity.name)
                if source_experiment_entity is not None
                else None
            )
            if logger is not None:
                logger.info(
                    'Using backbone run `%s` from source experiment `%s`.',
                    source_backbone_run.info.run_id,
                    source_experiment,
                )
                logger.info(
                    'Resolved backbone name from source run: %s',
                    local_backbone_name,
                )
        elif backbone_config is None:
            local_backbone_name = extract_backbone_name_from_run(run=local_backbone_run)
            if non_repair_run_names:
                local_backbone_training = extract_backbone_training_config_from_run(
                    run=local_backbone_run
                )
            (
                backbone_checkpoint_paths,
                backbone_eval_results,
                backbone_analysis_baseline,
            ) = load_backbone_from_existing_run(
                client=mlflow_client,
                backbone_run=local_backbone_run,
                checkpoint_dir=backbone_checkpoint_dir,
                expected_num_experiences=experiment_config.num_experiences,
                include_checkpoints_and_baseline=has_repair_runs,
            )
            if logger is not None:
                logger.info(
                    'Using existing local backbone run `%s` from experiment `%s`.',
                    local_backbone_run.info.run_id,
                    experiment_config.experiment_name,
                )
                logger.info('Resolved backbone name from local run: %s', local_backbone_name)
        else:
            if backbone_training is None:
                raise ValueError(
                    '`backbone.training` is required when `backbone.source_experiment` is not set '
                    'and no local `backbone` run exists.'
                )
            (
                backbone_checkpoint_paths,
                backbone_eval_results,
                backbone_analysis_baseline,
            ) = _run_backbone_pretraining_run(
                experiment_config=experiment_config,
                checkpoint_dir=backbone_checkpoint_dir,
            )

        if backbone_eval_results is None:
            raise RuntimeError('Backbone evaluation results are unavailable.')
        results[RUN_NAME_BACKBONE] = backbone_eval_results

        if not run_entries:
            return results

        if local_backbone_name is not None:
            if experiment_config.backbone is None:
                experiment_config.backbone = BackboneConfig(
                    name=local_backbone_name,
                    training=(
                        local_backbone_training
                        if local_backbone_training is not None
                        else None
                    ),
                )
            else:
                experiment_config.backbone.name = local_backbone_name

        resolved_backbone_name = (
            experiment_config.backbone.name
            if experiment_config.backbone is not None
            else None
        )
        if not isinstance(resolved_backbone_name, str) or resolved_backbone_name.strip() == '':
            raise RuntimeError(
                'Backbone name resolution failed. Ensure a valid `backbone` run is available.'
            )

        resolved_backbone_training = (
            experiment_config.backbone.training
            if experiment_config.backbone is not None
            else None
        )
        if non_repair_run_names and resolved_backbone_training is None:
            raise ValueError(
                '`backbone.training` is required for non-repair runs. '
                f'Invalid runs: {non_repair_run_names}'
            )

        if has_repair_runs and (
            backbone_checkpoint_paths is None or backbone_analysis_baseline is None
        ):
            raise RuntimeError('If repair controllers are configured, a `backbone` run must always exist.')

        for run_config, controller_type in run_entries:
            try:
                _, _, eval_results, _ = _train_and_evaluate_strategy(
                    experiment_config=experiment_config,
                    run_config=run_config,
                    backbone_checkpoint_paths=(
                        backbone_checkpoint_paths if controller_type == 'repair' else None
                    ),
                    backbone_analysis_baseline=(
                        backbone_analysis_baseline if controller_type == 'repair' else None
                    ),
                    backbone_source_experiment_id=source_experiment_id_for_logging,
                    backbone_source_experiment_name=source_experiment_name_for_logging,
                )
            except Exception:
                if logger is not None:
                    logger.exception('Error during run: %s', run_config.name)
                continue
            results[run_config.name] = eval_results
    finally:
        # Safe to delete local temp files after `mlflow.log_artifacts`; MLflow persists artifact copies.
        if backbone_checkpoint_dir.exists():
            shutil.rmtree(path=backbone_checkpoint_dir, ignore_errors=True)

    return results
