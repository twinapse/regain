"""
Utilities for experiment logging.
"""

from collections.abc import Mapping
from dataclasses import asdict
import io
from pathlib import Path
import tarfile

import avalanche
from avalanche.benchmarks import AvalancheDataset
from avalanche.benchmarks.scenarios import NCScenario
import mlflow

from regain.constants import EXPERIENCE_KEY_PREFIX
from regain.constants import MLFLOW_ARTIFACT_SPLITS_FILE
from regain.constants import NAMESPACE_SUMMARY
from regain.constants import NS_SEP
from regain.constants import PARAM_AVALANCHE_VERSION
from regain.constants import PARAM_BACKBONE
from regain.constants import PARAM_CONTROLLER
from regain.constants import PARAM_CONTROLLER_PATH
from regain.constants import PARAM_DEBUG_SKIP_REASON
from regain.constants import PARAM_NUM_CLASSES
from regain.constants import PARAM_RUN_NAME
from regain.constants import PARAM_TORCH_DETERMINISTIC_ALGORITHMS
from regain.constants import STREAMS
from regain.experiments.config import ExperimentConfig
from regain.mlflow_utils import log_scalar_metrics_to_namespace
from regain.registry import get_controller_path

__all__ = [
    'drop_non_leaf_run_params',
    'extract_dataset_indices',
    'flatten_experiment_config_params',
    'flatten_prefixed_params',
    'is_param_parent_present',
    'log_dataset_indices',
    'log_run_params',
    'log_summary_metrics',
    'update_run_params_with_prefixed_children',
]

_PARAM_BACKBONE_OPTIMIZER = 'backbone.training.optimizer'
_PARAM_BACKBONE_SOURCE_EXPERIMENT = f'{PARAM_BACKBONE}{NS_SEP}source_experiment'
_PARAM_BACKBONE_SOURCE_EXPERIMENT_ID = f'{_PARAM_BACKBONE_SOURCE_EXPERIMENT}{NS_SEP}id'
_PARAM_BACKBONE_SOURCE_EXPERIMENT_NAME = f'{_PARAM_BACKBONE_SOURCE_EXPERIMENT}{NS_SEP}name'
_PARAM_BACKBONE_TRAINING = 'backbone.training'


def _remove_kwargs_segment_from_path(*, path: str | None) -> str | None:
    """
    Remove nested `.kwargs` path segments from dotted parameter keys.

    Args:
        path (str | None): Candidate dotted parameter key.

    Returns:
        str | None: Normalized key with nested `.kwargs` segments removed.
    """
    if path is None:
        return None
    segments = path.split(NS_SEP)
    if len(segments) == 1:
        return path
    return NS_SEP.join(segment for segment in segments if segment != 'kwargs')


def flatten_prefixed_params(*, prefix: str | None, params: Mapping[str, object]) -> dict[str, object]:
    """
    Flatten a nested mapping into MLflow param-compatible key-value pairs.

    Args:
        prefix (str | None): Key prefix for flattened fields.
        params (Mapping[str, object]): Mapping to flatten.

    Returns:
        dict[str, object]: Flattened key-value pairs.

    """
    flattened: dict[str, object] = {}
    normalized_prefix = _remove_kwargs_segment_from_path(path=prefix)
    for key, value in params.items():
        key_str = str(key)
        raw_key_name = (
            f'{normalized_prefix}{NS_SEP}{key_str}'
            if normalized_prefix
            else key_str
        )
        key_name = _remove_kwargs_segment_from_path(path=raw_key_name) or raw_key_name
        if isinstance(value, Mapping):
            flattened.update(flatten_prefixed_params(prefix=raw_key_name, params=value))
            continue
        if isinstance(value, (bool, int, float, str)):
            flattened[key_name] = value
            continue
        if value is None:
            continue
        flattened[key_name] = str(value)
    return flattened


def is_param_parent_present(
    *,
    run_params: Mapping[str, object],
    parent_key: str,
) -> bool:
    """
    Check whether a parent param is present for child-param logging.

    Args:
        run_params (Mapping[str, object]): Candidate MLflow params.
        parent_key (str): Parent namespace key.

    Returns:
        bool: True when the parent param is present.
    """
    parent_value = run_params.get(parent_key)
    if parent_value is not None and str(parent_value).strip().lower() not in {'', 'none'}:
        return True
    child_prefix = f'{parent_key}{NS_SEP}'
    return any(key.startswith(child_prefix) for key in run_params)


def update_run_params_with_prefixed_children(
    *,
    run_params: Mapping[str, object],
    required_parent_key: str,
    child_prefix: str,
    child_params: Mapping[str, object],
) -> dict[str, object]:
    """
    Add prefixed child params only when their parent namespace is present.

    Args:
        run_params (Mapping[str, object]): Candidate MLflow params.
        required_parent_key (str): Parent namespace that must be present.
        child_prefix (str): Prefix to apply to `child_params` keys.
        child_params (Mapping[str, object]): Child params to flatten and merge.

    Returns:
        dict[str, object]: Updated params.
    """
    merged_params = dict(run_params)
    if not is_param_parent_present(run_params=merged_params, parent_key=required_parent_key):
        return merged_params
    merged_params.update(flatten_prefixed_params(prefix=child_prefix, params=child_params))
    return merged_params


def drop_non_leaf_run_params(*, run_params: Mapping[str, object]) -> dict[str, object]:
    """
    Drop all non-leaf params from a flattened parameter map.

    Args:
        run_params (Mapping[str, object]): Candidate MLflow params.

    Returns:
        dict[str, object]: Leaf-only params.
    """
    all_keys = set(run_params.keys())
    leaf_only_params: dict[str, object] = {}
    for key, value in run_params.items():
        child_prefix = f'{key}{NS_SEP}'
        if any(other_key.startswith(child_prefix) for other_key in all_keys):
            continue
        leaf_only_params[key] = value
    return leaf_only_params


def flatten_experiment_config_params(
    *,
    experiment_config: ExperimentConfig,
    include_backbone_params: bool = True,
) -> dict[str, object]:
    """
    Flatten all experiment config fields for MLflow param logging.

    Args:
        experiment_config (ExperimentConfig): Experiment configuration to flatten.

    Returns:
        dict[str, object]: Flattened config fields excluding run-specific and environment-specific fields.
    """
    config_payload = asdict(experiment_config)
    config_payload.pop('runs', None)
    if not include_backbone_params:
        config_payload.pop(PARAM_BACKBONE, None)
    config_payload.pop('dataset_path', None)
    return flatten_prefixed_params(prefix=None, params=config_payload)


def log_run_params(
    *,
    experiment_config: ExperimentConfig,
    run_config_payload: Mapping[str, object],
    controller_name: str | None,
    deterministic_algorithms_enabled: bool,
    optimizer_kwargs: Mapping[str, object],
    include_backbone_params: bool,
    backbone_source_experiment_id: str | None = None,
    backbone_source_experiment_name: str | None = None,
    num_classes: int | None = None,
    debug_skip_reason: str | None = None,
) -> None:
    """
    Log common run parameters to MLflow.

    Args:
        experiment_config (ExperimentConfig): Experiment configuration.
        run_config_payload (Mapping[str, object]): Run configuration as a mapping.
        controller_name (str | None): Optional controller name.
        deterministic_algorithms_enabled (bool): Whether deterministic algorithms are enabled.
        optimizer_kwargs (Mapping[str, object]): Effective optimizer keyword arguments.
        include_backbone_params (bool): Whether to log `backbone.*` parameters.
        backbone_source_experiment_id (str | None): Optional source experiment id to log at
                                                    `backbone.source_experiment.id`.
        backbone_source_experiment_name (str | None): Optional source experiment name snapshot to log at
                                                      `backbone.source_experiment.name`. This value is not
                                                      synchronized after logging; if the source
                                                      experiment is renamed later, this logged name can differ from
                                                      the current MLflow experiment name.
        num_classes (int | None): Total number of benchmark classes (optional).
        debug_skip_reason (str | None): Optional debug skip reason.

    Returns:
        None
    """
    run_params = {}

    # Add flatten experiment config
    run_params.update(
        flatten_experiment_config_params(
            experiment_config=experiment_config,
            include_backbone_params=include_backbone_params,
        )
    )

    # Add flatten run config
    flattened_run_config = flatten_prefixed_params(prefix=None, params=run_config_payload)
    if 'name' in flattened_run_config:
        flattened_run_config[PARAM_RUN_NAME] = flattened_run_config.pop('name')
    run_params.update(flattened_run_config)

    # Add controller path if controller is specified, otherwise drop any controller-related params
    if controller_name is None:
        run_params = {
            key: value
            for key, value in run_params.items()
            if key != PARAM_CONTROLLER and not key.startswith(f'{PARAM_CONTROLLER}{NS_SEP}')
        }
    else:
        run_params[PARAM_CONTROLLER_PATH] = get_controller_path(controller_name)
        if not include_backbone_params:
            run_params = {
                key: value
                for key, value in run_params.items()
                if key != PARAM_BACKBONE and not key.startswith(f'{PARAM_BACKBONE}{NS_SEP}')
            }
        if (
            backbone_source_experiment_id is not None
            and str(backbone_source_experiment_id).strip() != ''
        ):
            run_params[_PARAM_BACKBONE_SOURCE_EXPERIMENT_ID] = str(backbone_source_experiment_id)
        if (
            backbone_source_experiment_name is not None
            and str(backbone_source_experiment_name).strip() != ''
        ):
            run_params[_PARAM_BACKBONE_SOURCE_EXPERIMENT_NAME] = str(backbone_source_experiment_name)

    # Add optimizer params with appropriate prefix only when the backbone training namespace is present
    run_params = update_run_params_with_prefixed_children(
        run_params=run_params,
        required_parent_key=_PARAM_BACKBONE_TRAINING,
        child_prefix=_PARAM_BACKBONE_OPTIMIZER,
        child_params=optimizer_kwargs,
    )

    # Add runtime params
    run_params[PARAM_AVALANCHE_VERSION] = avalanche.__version__
    run_params[PARAM_TORCH_DETERMINISTIC_ALGORITHMS] = deterministic_algorithms_enabled

    # Add optional params
    if num_classes is not None:
        run_params[PARAM_NUM_CLASSES] = int(num_classes)
    if debug_skip_reason is not None:
        run_params[PARAM_DEBUG_SKIP_REASON] = debug_skip_reason

    # Drop non-leaf params to avoid logging redundant parent namespaces
    run_params = drop_non_leaf_run_params(run_params=run_params)

    # Log all params to MLflow (sorted by key for consistent ordering)
    mlflow.log_params({k: run_params[k] for k in sorted(run_params)})


def log_summary_metrics(
    *,
    summary_metrics: Mapping[str, float],
    step: int,
) -> None:
    """
    Log summary scalar metrics to MLflow.

    Args:
        summary_metrics (Mapping[str, float]): Summary metric mapping.
        step (int): Global metric step.

    Returns:
        None
    """
    log_scalar_metrics_to_namespace(
        scalar_metrics=summary_metrics,
        namespace=NAMESPACE_SUMMARY,
        step=int(step),
    )


def extract_dataset_indices(experience_dataset: AvalancheDataset) -> list[int]:
    """
    Extract the index of each example in the given experience dataset w.r.t. the original dataset.

    Args:
        experience_dataset (AvalancheDataset): The experience dataset to extract indices from.

    Returns:
        list[int]: List of integer indices in the original dataset.

    Raises:
        AttributeError: If the experience dataset does not have an `original_indices` data attribute.
        RuntimeError: If indices cannot be extracted.
    """
    if not hasattr(experience_dataset, 'original_indices'):
        raise AttributeError('The experience dataset does not have an `original_indices` data attribute')

    try:
        indices = list(experience_dataset.original_indices)
        return [int(idx) for idx in indices]
    except Exception as exc:
        raise RuntimeError(f'Failed to extract indices from `original_indices` data attribute: {exc}') from exc


def log_dataset_indices(
    *,
    benchmark: NCScenario,
    artifacts_dir: Path,
) -> None:
    """
    Log per-experience dataset indices for every stream as a compressed MLflow artifact.

    Each experience produces a plain-text file with one global index per line.
    All files are packed into a single gzip-compressed tar archive using the naming scheme
    ``{stream}/exp_{idx:03d}.txt``.

    Args:
        benchmark (NCScenario): Benchmark scenario containing the streams.
        artifacts_dir (Path): Temporary directory where the archive is written before being logged to MLflow.

    Returns:
        None

    Raises:
        RuntimeError: If indices cannot be extracted for any experience dataset.
    """
    archive_path = Path(artifacts_dir) / MLFLOW_ARTIFACT_SPLITS_FILE

    with tarfile.open(archive_path, 'w:gz') as tar:
        for stream_name in STREAMS:
            stream_attr = f'{stream_name}_stream'
            stream = getattr(benchmark, stream_attr, None)
            if stream is None:
                continue

            for experience in stream:
                exp_idx = int(getattr(experience, 'current_experience', 0))
                dataset = getattr(experience, 'dataset', None)
                if dataset is None:
                    raise RuntimeError(
                        'Experience dataset is missing while logging dataset indices: '
                        f'stream={stream_name} exp={exp_idx}'
                    )

                indices = extract_dataset_indices(dataset)
                content = '\n'.join(str(idx) for idx in indices)
                if content:
                    content += '\n'
                data = content.encode('utf-8')

                entry_name = f'{stream_name}/{EXPERIENCE_KEY_PREFIX}_{exp_idx:03d}.txt'
                info = tarfile.TarInfo(name=entry_name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))

    mlflow.log_artifact(str(archive_path))
