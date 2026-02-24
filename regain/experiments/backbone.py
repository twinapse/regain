"""
Backbone utilities for experiment execution.
"""

from collections.abc import Mapping
from pathlib import Path

from avalanche.training.templates import BaseTemplate
from mlflow.entities import Run
from mlflow.tracking import MlflowClient
import yaml

from regain.analysis.artifacts import ARTIFACT_ACC_EXP_BASE
from regain.analysis.artifacts import ARTIFACT_ACC_FINAL_BASE
from regain.avalanche_utils.plugins import RegainEvaluationPlugin
from regain.constants import DIAG_VECTOR_KEYS
from regain.constants import EXPERIENCE_KEY_PREFIX
from regain.constants import MLFLOW_ARTIFACT_BACKBONE_CHECKPOINTS_DIR
from regain.constants import NAMESPACE_SUMMARY
from regain.constants import NS_SEP
from regain.constants import PARAM_BACKBONE
from regain.constants import RUN_ACC_EXP
from regain.constants import RUN_ACC_FINAL
from regain.constants import RUN_NAME_BACKBONE
from regain.experiments.config import OptimizerConfig
from regain.experiments.config import StrategyConfig
from regain.experiments.config import TrainingConfig
from regain.mlflow_utils import download_json_artifact
from regain.mlflow_utils import resolve_experiment_id
from regain.mlflow_utils import resolve_mlflow_run_name
from regain.mlflow_utils import search_runs_paginated

__all__ = [
    'collect_backbone_checkpoint_paths',
    'download_backbone_checkpoints_from_run',
    'extract_backbone_analysis_baseline',
    'extract_backbone_analysis_baseline_from_metrics',
    'extract_backbone_name_from_run',
    'extract_backbone_training_config_from_run',
    'extract_required_float_vector',
    'extract_summary_metrics_from_run',
    'find_backbone_runs',
    'load_backbone_from_existing_run',
    'load_backbone_analysis_baseline_from_run',
    'load_backbone_from_source_experiment',
    'resolve_local_backbone_run',
]


def load_backbone_from_source_experiment(
    *,
    client: MlflowClient,
    source_experiment: str,
    checkpoint_dir: Path,
    expected_num_experiences: int,
) -> tuple[list[Path], dict[str, float], dict[str, list[float | None]], Run]:
    """
    Resolve and load backbone artifacts from `backbone.source_experiment`.

    Args:
        client (MlflowClient): MLflow client.
        source_experiment (str): Source experiment id or name.
        checkpoint_dir (Path): Local directory to download checkpoints into.
        expected_num_experiences (int): Expected number of experiences.

    Returns:
        tuple[list[Path], dict[str, float], dict[str, list[float | None]], Run]:
            (checkpoint paths, summary metrics, baseline vectors, source run).
    """
    try:
        backbone_runs = find_backbone_runs(
            client=client,
            experiment=source_experiment,
            allow_missing_experiment=False,
        )
    except ValueError as exc:
        raise RuntimeError(
            f'Source experiment `{source_experiment}` was not found. '
            'If repair controllers are configured, a `backbone` run must always '
            'exist.'
        ) from exc
    if not backbone_runs:
        raise RuntimeError(
            'If repair controllers are configured, a `backbone` run must always '
            f'exist. No `backbone` run was found in source experiment '
            f'`{source_experiment}`.'
        )
    if len(backbone_runs) > 1:
        raise RuntimeError(
            f'Source experiment `{source_experiment}` has multiple `backbone` '
            'runs. An experiment cannot have multiple `backbone` runs.'
        )

    backbone_run = backbone_runs[0]
    backbone_run_id = str(backbone_run.info.run_id)
    checkpoint_paths = download_backbone_checkpoints_from_run(
        client=client,
        run_id=backbone_run_id,
        checkpoint_dir=checkpoint_dir,
        expected_count=expected_num_experiences,
    )
    analysis_baseline = load_backbone_analysis_baseline_from_run(
        client=client,
        run=backbone_run,
        expected_num_experiences=expected_num_experiences,
    )
    eval_results = extract_summary_metrics_from_run(run=backbone_run)
    return checkpoint_paths, eval_results, analysis_baseline, backbone_run


def _deserialize_mlflow_param_value(*, raw_value: str) -> object:
    """
    Deserialize an MLflow parameter string into a scalar value when possible.

    Args:
        raw_value (str): Raw MLflow parameter value.

    Returns:
        object: Parsed scalar value or the original string.
    """
    try:
        parsed_value = yaml.safe_load(raw_value)
    except yaml.YAMLError:
        return raw_value
    if isinstance(parsed_value, (Mapping, list, tuple, set)):
        return raw_value
    return parsed_value


def _extract_required_run_param_str(
    *,
    params: Mapping[str, str],
    run_id: str,
    key: str,
) -> str:
    """
    Extract a required non-empty string parameter from a backbone run.

    Args:
        params (Mapping[str, str]): Run parameter payload.
        run_id (str): Source run id.
        key (str): Parameter key.

    Returns:
        str: Parameter value as a stripped string.
    """
    raw_value = params.get(key)
    if raw_value is None:
        raise RuntimeError(
            f'Backbone run `{run_id}` is missing required param `{key}`.'
        )
    value = str(raw_value).strip()
    if not value:
        raise RuntimeError(
            f'Backbone run `{run_id}` has invalid param `{key}`.'
        )
    return value


def _extract_optional_run_param_str(
    *,
    params: Mapping[str, str],
    key: str,
    default: str,
) -> str:
    """
    Extract an optional string parameter from a run, returning a default when unset.

    Args:
        params (Mapping[str, str]): Run parameter payload.
        key (str): Parameter key.
        default (str): Default value if unset/empty.

    Returns:
        str: Extracted or default string value.
    """
    raw_value = params.get(key)
    if raw_value is None:
        return default
    value = str(raw_value).strip()
    if not value:
        return default
    return value


def _extract_required_run_param_int(
    *,
    params: Mapping[str, str],
    run_id: str,
    key: str,
) -> int:
    """
    Extract a required integer-like parameter from a backbone run.

    Args:
        params (Mapping[str, str]): Run parameter payload.
        run_id (str): Source run id.
        key (str): Parameter key.

    Returns:
        int: Parsed integer value.
    """
    value = _extract_required_run_param_str(params=params, run_id=run_id, key=key)
    parsed_value = _deserialize_mlflow_param_value(raw_value=value)
    if isinstance(parsed_value, bool):
        raise RuntimeError(
            f'Backbone run `{run_id}` has non-integer param `{key}`: {parsed_value}'
        )
    try:
        return int(parsed_value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f'Backbone run `{run_id}` has non-integer param `{key}`: {parsed_value}'
        ) from exc


def _extract_optional_run_param_int(
    *,
    params: Mapping[str, str],
    run_id: str,
    key: str,
    default: int,
) -> int:
    """
    Extract an optional integer-like parameter from a backbone run.

    Args:
        params (Mapping[str, str]): Run parameter payload.
        run_id (str): Source run id.
        key (str): Parameter key.
        default (int): Default value when the parameter is missing.

    Returns:
        int: Parsed integer value.
    """
    if key not in params:
        return int(default)
    return _extract_required_run_param_int(params=params, run_id=run_id, key=key)


def _extract_prefixed_params(
    *,
    params: Mapping[str, str],
    prefix: str,
    excluded_leaf_keys: set[str],
) -> dict[str, object]:
    """
    Extract scalar parameters under a dotted prefix.

    Args:
        params (Mapping[str, str]): Run parameter payload.
        prefix (str): Dotted prefix.
        excluded_leaf_keys (set[str]): Leaf names to skip.

    Returns:
        dict[str, object]: Parsed suffix-keyed scalar values.
    """
    extracted: dict[str, object] = {}
    child_prefix = f'{prefix}{NS_SEP}'
    for key, raw_value in params.items():
        if not key.startswith(child_prefix):
            continue
        suffix = key.removeprefix(child_prefix).strip()
        if not suffix or suffix in excluded_leaf_keys:
            continue
        extracted[suffix] = _deserialize_mlflow_param_value(raw_value=str(raw_value))
    return extracted


def extract_backbone_name_from_run(*, run: Run) -> str:
    """
    Extract the logged backbone architecture name from a reserved `backbone` run.

    Args:
        run (Run): Source backbone run.

    Returns:
        str: Backbone name (e.g. `resnet18`).

    Raises:
        RuntimeError: If the run does not log a valid `backbone.name` parameter.
    """
    run_id = str(run.info.run_id)
    key = f'{PARAM_BACKBONE}{NS_SEP}name'
    params_payload = dict(run.data.params or {})
    return _extract_required_run_param_str(params=params_payload, run_id=run_id, key=key)


def extract_backbone_training_config_from_run(*, run: Run) -> TrainingConfig:
    """
    Extract backbone training settings from a reserved `backbone` run.

    Args:
        run (Run): Source backbone run.

    Returns:
        TrainingConfig: Reconstructed backbone training configuration.

    Raises:
        RuntimeError: If required training parameters are missing/invalid in the run.
    """
    run_id = str(run.info.run_id)
    params_payload = dict(run.data.params or {})
    training_prefix = f'{PARAM_BACKBONE}{NS_SEP}training'
    strategy_prefix = f'{training_prefix}{NS_SEP}strategy'
    optimizer_prefix = f'{training_prefix}{NS_SEP}optimizer'

    num_epochs = _extract_required_run_param_int(
        params=params_payload,
        run_id=run_id,
        key=f'{training_prefix}{NS_SEP}num_epochs',
    )
    batch_size = _extract_optional_run_param_int(
        params=params_payload,
        run_id=run_id,
        key=f'{training_prefix}{NS_SEP}batch_size',
        default=128,
    )
    strategy_name = _extract_required_run_param_str(
        params=params_payload,
        run_id=run_id,
        key=f'{strategy_prefix}{NS_SEP}name',
    )
    strategy_kwargs = _extract_prefixed_params(
        params=params_payload,
        prefix=strategy_prefix,
        excluded_leaf_keys={'name'},
    )
    optimizer_name = _extract_optional_run_param_str(
        params=params_payload,
        key=f'{optimizer_prefix}{NS_SEP}name',
        default='sgd',
    )
    optimizer_kwargs = _extract_prefixed_params(
        params=params_payload,
        prefix=optimizer_prefix,
        excluded_leaf_keys={'name'},
    )
    return TrainingConfig(
        num_epochs=num_epochs,
        strategy=StrategyConfig(
            name=strategy_name,
            kwargs=strategy_kwargs,
        ),
        optimizer=OptimizerConfig(
            name=optimizer_name,
            kwargs=optimizer_kwargs,
        ),
        batch_size=batch_size,
    )


def extract_required_float_vector(
    *,
    payload: Mapping[str, object],
    key: str,
    expected_len: int,
) -> list[float]:
    """
    Extract a required float vector from a mapping and validate its length.

    Args:
        payload (Mapping[str, object]): Source mapping.
        key (str): Key to extract.
        expected_len (int): Expected vector length.

    Returns:
        list[float]: Extracted vector as floats.
    """
    raw_values = payload.get(key)
    if not isinstance(raw_values, list):
        raise RuntimeError(
            f'Missing or invalid `{key}` vector in backbone analysis artifacts.'
        )
    values = [float(value) for value in raw_values]
    if len(values) != int(expected_len):
        raise RuntimeError(
            f'Backbone `{key}` length mismatch. '
            f'expected={int(expected_len)}, observed={len(values)}'
        )
    return values


def extract_required_nullable_float_vector(
    *,
    payload: Mapping[str, object],
    key: str,
    expected_len: int,
) -> list[float | None]:
    """
    Extract a required float vector (allowing missing entries) from a mapping.

    Args:
        payload (Mapping[str, object]): Source mapping.
        key (str): Key to extract.
        expected_len (int): Expected vector length.

    Returns:
        list[float | None]: Extracted vector.
    """
    raw_values = payload.get(key)
    if not isinstance(raw_values, list):
        raise RuntimeError(
            f'Missing or invalid `{key}` vector in backbone analysis artifacts.'
        )
    values: list[float | None] = []
    for value in raw_values:
        if value is None:
            values.append(None)
            continue
        values.append(float(value))
    if len(values) != int(expected_len):
        raise RuntimeError(
            f'Backbone `{key}` length mismatch. '
            f'expected={int(expected_len)}, observed={len(values)}'
        )
    return values


def extract_backbone_analysis_baseline(
    *,
    strategy: BaseTemplate,
    expected_num_experiences: int,
) -> dict[str, list[float | None]]:
    """
    Extract controller-off baseline vectors from the strategy evaluation plugin.

    Args:
        strategy (BaseTemplate): Strategy used in the backbone run.
        expected_num_experiences (int): Expected number of experiences.

    Returns:
        dict[str, list[float | None]]: Baseline vectors keyed by artifact keys and diagnostic vectors.
    """
    plugins = getattr(strategy, 'plugins', [])
    evaluation_plugin = next(
        (
            plugin
            for plugin in plugins
            if isinstance(plugin, RegainEvaluationPlugin)
        ),
        None,
    )
    if evaluation_plugin is None:
        raise RuntimeError('Backbone run is missing RegainEvaluationPlugin.')
    if not isinstance(evaluation_plugin.artifacts, dict):
        raise RuntimeError('Backbone analysis artifacts are unavailable.')

    artifacts = evaluation_plugin.artifacts
    baseline: dict[str, list[float | None]] = {
        ARTIFACT_ACC_EXP_BASE: extract_required_float_vector(
            payload=artifacts,
            key=ARTIFACT_ACC_EXP_BASE,
            expected_len=expected_num_experiences,
        ),
        ARTIFACT_ACC_FINAL_BASE: extract_required_float_vector(
            payload=artifacts,
            key=ARTIFACT_ACC_FINAL_BASE,
            expected_len=expected_num_experiences,
        ),
    }
    for diag_key in DIAG_VECTOR_KEYS:
        vector = extract_required_nullable_float_vector(
            payload=artifacts,
            key=diag_key,
            expected_len=expected_num_experiences,
        )
        baseline[diag_key] = vector
    return baseline


def find_backbone_runs(
    *,
    client: MlflowClient,
    experiment: str,
    allow_missing_experiment: bool = False,
) -> list[Run]:
    """
    Find runs named `backbone` for an experiment.

    Args:
        client (MlflowClient): MLflow client.
        experiment (str): Experiment id or name.
        allow_missing_experiment (bool): Whether missing experiments return an empty list.

    Returns:
        list[Run]: Matching backbone runs.
    """
    try:
        experiment_id = resolve_experiment_id(client=client, experiment=experiment)
    except ValueError:
        if allow_missing_experiment:
            return []
        raise

    runs = search_runs_paginated(
        client=client,
        experiment_ids=[experiment_id],
        filter_string='',
    )
    return [
        run
        for run in runs
        if resolve_mlflow_run_name(run=run) == RUN_NAME_BACKBONE
    ]


def resolve_local_backbone_run(
    *,
    client: MlflowClient,
    experiment_name: str,
) -> Run | None:
    """
    Resolve the single local reserved `backbone` run for an experiment.

    Args:
        client (MlflowClient): MLflow client.
        experiment_name (str): Local experiment name.

    Returns:
        Run | None: Existing local `backbone` run, if present.
    """
    local_backbone_runs = find_backbone_runs(
        client=client,
        experiment=experiment_name,
        allow_missing_experiment=True,
    )
    if not local_backbone_runs:
        return None
    if len(local_backbone_runs) > 1:
        raise RuntimeError(
            f'Experiment `{experiment_name}` has multiple `backbone` runs. '
            'An experiment cannot have multiple `backbone` runs.'
        )
    return local_backbone_runs[0]


def load_backbone_from_existing_run(
    *,
    client: MlflowClient,
    backbone_run: Run,
    checkpoint_dir: Path,
    expected_num_experiences: int,
    include_checkpoints_and_baseline: bool,
) -> tuple[list[Path] | None, dict[str, float], dict[str, list[float | None]] | None]:
    """
    Load backbone outputs from an existing local `backbone` run.

    Args:
        client (MlflowClient): MLflow client.
        backbone_run (Run): Existing local `backbone` run.
        checkpoint_dir (Path): Local directory where checkpoints are downloaded.
        expected_num_experiences (int): Expected number of experiences.
        include_checkpoints_and_baseline (bool): Whether to load checkpoints and analysis baseline vectors.

    Returns:
        tuple[list[Path] | None, dict[str, float], dict[str, list[float | None]] | None]:
            (checkpoint paths, scalar evaluation metrics, backbone analysis baseline vectors).
    """
    eval_results = extract_summary_metrics_from_run(run=backbone_run)
    if not include_checkpoints_and_baseline:
        return None, eval_results, None

    run_id = str(backbone_run.info.run_id)
    checkpoint_paths = download_backbone_checkpoints_from_run(
        client=client,
        run_id=run_id,
        checkpoint_dir=checkpoint_dir,
        expected_count=expected_num_experiences,
    )
    analysis_baseline = load_backbone_analysis_baseline_from_run(
        client=client,
        run=backbone_run,
        expected_num_experiences=expected_num_experiences,
    )
    return checkpoint_paths, eval_results, analysis_baseline


def collect_backbone_checkpoint_paths(
    *,
    checkpoint_dir: Path,
    expected_count: int,
) -> list[Path]:
    """
    Collect and validate per-experience backbone checkpoints from a directory.

    Args:
        checkpoint_dir (Path): Directory containing `exp_###.pt` files.
        expected_count (int): Expected number of checkpoints.

    Returns:
        list[Path]: Checkpoint paths ordered by experience index.
    """
    indexed_paths: dict[int, Path] = {}
    for checkpoint_path in sorted(checkpoint_dir.glob('exp_*.pt')):
        filename = checkpoint_path.name
        if not filename.startswith('exp_') or not filename.endswith('.pt'):
            continue
        idx_token = filename[len('exp_'):-len('.pt')]
        if not idx_token.isdigit():
            continue
        exp_idx = int(idx_token)
        if exp_idx in indexed_paths:
            raise RuntimeError(
                f'Duplicate backbone checkpoint index {exp_idx}: {checkpoint_path}'
            )
        indexed_paths[exp_idx] = checkpoint_path

    if not indexed_paths:
        raise RuntimeError(
            f'No backbone checkpoints were found under: {checkpoint_dir}'
        )

    observed_indices = sorted(indexed_paths)
    expected_indices = list(range(int(expected_count)))
    if observed_indices != expected_indices:
        raise RuntimeError(
            'Backbone checkpoints are incomplete or out of order. '
            f'expected={expected_indices}, observed={observed_indices}'
        )
    return [indexed_paths[idx] for idx in observed_indices]


def download_backbone_checkpoints_from_run(
    *,
    client: MlflowClient,
    run_id: str,
    checkpoint_dir: Path,
    expected_count: int,
) -> list[Path]:
    """
    Download backbone checkpoints from MLflow artifacts and validate them.

    Args:
        client (MlflowClient): MLflow client.
        run_id (str): Source run id.
        checkpoint_dir (Path): Local download directory.
        expected_count (int): Expected number of checkpoints.

    Returns:
        list[Path]: Ordered checkpoint paths.
    """
    try:
        downloaded_path = client.download_artifacts(
            run_id,
            MLFLOW_ARTIFACT_BACKBONE_CHECKPOINTS_DIR,
            str(checkpoint_dir),
        )
    except Exception as exc:
        raise RuntimeError(
            f'Backbone run `{run_id}` is missing checkpoint artifacts. '
            'Provide `backbone.source_experiment` pointing to an experiment with '
            'saved backbone checkpoints.'
        ) from exc

    artifacts_dir = Path(downloaded_path)
    if not artifacts_dir.exists() or not artifacts_dir.is_dir():
        raise RuntimeError(
            f'Backbone checkpoints artifact is invalid for run `{run_id}`: '
            f'{artifacts_dir}'
        )

    return collect_backbone_checkpoint_paths(
        checkpoint_dir=artifacts_dir,
        expected_count=expected_count,
    )


def extract_backbone_analysis_baseline_from_metrics(
    *,
    metrics: Mapping[str, float],
    expected_num_experiences: int,
) -> dict[str, list[float]] | None:
    """
    Extract baseline vectors from run metrics.

    Args:
        metrics (Mapping[str, float]): Run metrics mapping.
        expected_num_experiences (int): Expected number of experiences.

    Returns:
        dict[str, list[float]] | None: Baseline vectors when complete.
    """
    baseline: dict[str, list[float]] = {}
    key_to_prefix = {
        ARTIFACT_ACC_EXP_BASE: RUN_ACC_EXP,
        ARTIFACT_ACC_FINAL_BASE: RUN_ACC_FINAL,
    }
    for key, metric_prefix in key_to_prefix.items():
        values: list[float] = []
        for exp_idx in range(int(expected_num_experiences)):
            metric_key = (
                f'{metric_prefix}'
                f'{NS_SEP}{EXPERIENCE_KEY_PREFIX}{exp_idx:03d}'
                f'{NS_SEP}base'
            )
            raw_value = metrics.get(metric_key)
            if raw_value is None:
                return None
            values.append(float(raw_value))
        baseline[key] = values
    return baseline


def load_backbone_analysis_baseline_from_run(
    *,
    client: MlflowClient,
    run: Run,
    expected_num_experiences: int,
) -> dict[str, list[float | None]]:
    """
    Load backbone baseline vectors from analysis artifact.

    Args:
        client (MlflowClient): MLflow client.
        run (Run): Source backbone run.
        expected_num_experiences (int): Expected number of experiences.

    Returns:
        dict[str, list[float | None]]: Baseline vectors keyed by artifact keys and diagnostic vectors.
    """
    run_id = str(run.info.run_id)
    artifacts_payload = download_json_artifact(
        client=client,
        run_id=run_id,
        artifact_path='analysis_artifacts.json',
    )
    if not isinstance(artifacts_payload, Mapping):
        raise RuntimeError(
            f'Backbone run `{run_id}` is missing required `analysis_artifacts.json`.'
        )

    baseline: dict[str, list[float | None]] = {
        ARTIFACT_ACC_EXP_BASE: extract_required_float_vector(
            payload=artifacts_payload,
            key=ARTIFACT_ACC_EXP_BASE,
            expected_len=expected_num_experiences,
        ),
        ARTIFACT_ACC_FINAL_BASE: extract_required_float_vector(
            payload=artifacts_payload,
            key=ARTIFACT_ACC_FINAL_BASE,
            expected_len=expected_num_experiences,
        ),
    }
    for diag_key in DIAG_VECTOR_KEYS:
        vector = extract_required_nullable_float_vector(
            payload=artifacts_payload,
            key=diag_key,
            expected_len=expected_num_experiences,
        )
        baseline[diag_key] = vector

    return baseline


def extract_summary_metrics_from_run(*, run: Run) -> dict[str, float]:
    """
    Extract summary-namespace scalar metrics from an MLflow run.

    Args:
        run (Run): Source run.

    Returns:
        dict[str, float]: Summary metrics without the summary namespace prefix.
    """
    metrics_payload = dict(run.data.metrics or {})
    prefix = f'{NAMESPACE_SUMMARY}{NS_SEP}'
    summary_metrics: dict[str, float] = {}
    for key, value in metrics_payload.items():
        key_str = str(key)
        if not key_str.startswith(prefix):
            continue
        summary_key = key_str[len(prefix):]
        summary_metrics[summary_key] = float(value)
    return summary_metrics
