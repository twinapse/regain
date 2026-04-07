"""
MLflow utilities.
"""

from collections.abc import Mapping
import contextlib
from datetime import datetime
from datetime import timezone
import json
from pathlib import Path
import re
import tempfile
import traceback
from typing import Any, Iterator, Sequence
from urllib.parse import urlparse

import mlflow
from mlflow.entities import Experiment
from mlflow.entities import Run
from mlflow.tracking import MlflowClient
from mlflow.utils.yaml_utils import write_yaml

from regain.constants import COLUMN_END_TIME
from regain.constants import COLUMN_EXPERIMENT_ID
from regain.constants import COLUMN_RUN_ID
from regain.constants import COLUMN_RUN_NAME
from regain.constants import COLUMN_START_TIME
from regain.constants import COLUMN_STATUS
from regain.constants import MLFLOW_ARTIFACT_ERROR_FILE
from regain.constants import NS_SEP
from regain.constants import PARAM_RUN_NAME

__all__ = [
    'build_mlflow_run_columns',
    'delete_mlflow_runs',
    'download_json_artifact',
    'ensure_experiment',
    'format_timestamp_ms',
    'init_mlflow',
    'log_scalar_metrics_to_namespace',
    'log_fatal_error_context',
    'normalize_metric_name',
    'normalize_tracking_uri',
    'resolve_active_runs_by_name',
    'resolve_artifact_location',
    'resolve_experiment_id',
    'resolve_latest_active_runs_by_name',
    'resolve_mlflow_run_name',
    'resolve_tracking_uri',
    'search_runs_paginated',
    'set_tracking_uri',
    'to_scalar_metric_value',
    'write_experiment_meta_yaml',
]

_NS_SEP_ESCAPED = re.escape(NS_SEP)
_NON_ALNUM_SEP = re.compile(rf'[^a-zA-Z0-9_{_NS_SEP_ESCAPED}]+')
_MULTI_UNDERSCORE = re.compile(r'_+')
_MULTI_NAMESPACE_SEP = re.compile(rf'{_NS_SEP_ESCAPED}+')


############################
# Metric logging utilities #
############################


def normalize_metric_name(raw: str) -> str:
    """
    Normalize a raw metric name into a stable MLflow-safe token.

    Args:
        raw (str): Raw metric name.

    Returns:
        str: Normalized metric token.
    """
    raw = '' if raw is None else str(raw)
    norm = raw.replace('/', NS_SEP)
    norm = _NON_ALNUM_SEP.sub('_', norm)
    norm = _MULTI_UNDERSCORE.sub('_', norm).strip('_')
    norm = _MULTI_NAMESPACE_SEP.sub(NS_SEP, norm).strip(NS_SEP)
    return norm.lower() or 'unnamed_metric'


def to_scalar_metric_value(value: Any) -> float | None:
    """
    Convert metric-like values to scalar floats when possible.

    Args:
        value (Any): Candidate metric value.

    Returns:
        float | None: Scalar metric value, or None when conversion fails.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if hasattr(value, 'item'):
        try:
            item_value = value.item()
            if isinstance(item_value, (int, float)) and not isinstance(item_value, bool):
                return float(item_value)
            return float(item_value)
        except Exception:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def log_scalar_metrics_to_namespace(
    *,
    scalar_metrics: Mapping[str, float],
    namespace: str,
    step: int,
) -> None:
    """
    Log scalar metrics under a namespace after normalizing metric names.

    Args:
        scalar_metrics (Mapping[str, float]): Scalar metrics keyed by raw names.
        namespace (str): Metric namespace prefix.
        step (int): Metric step.

    Returns:
        None
    """
    if mlflow.active_run() is None:
        return

    namespace_prefix = str(namespace).strip()
    for metric_name, metric_value in scalar_metrics.items():
        scalar_value = to_scalar_metric_value(metric_value)
        if scalar_value is None:
            continue
        normalized_name = normalize_metric_name(metric_name)
        metric_key = (
            f'{namespace_prefix}{NS_SEP}{normalized_name}'
            if namespace_prefix != ''
            else normalized_name
        )
        mlflow.log_metric(
            key=metric_key,
            value=float(scalar_value),
            step=int(step),
        )


##########################
# URI/path normalization #
##########################


def _normalize_artifact_location(raw_uri: str) -> str:
    """
    Normalize an artifact location to a stable representation.

    Args:
        raw_uri (str): Artifact location or filesystem path.

    Returns:
        str: Normalized artifact location.
    """
    parsed = urlparse(raw_uri)
    if parsed.scheme:
        if parsed.scheme == 'file':
            resolved = Path(parsed.path).expanduser().resolve()
            return f'file:///{resolved.as_posix().lstrip("/")}'
        if len(parsed.scheme) == 1 and raw_uri[1:3] in {':\\', ':/'}:
            resolved = Path(raw_uri).expanduser().resolve()
            return f'file:///{resolved.as_posix().lstrip("/")}'
        return raw_uri
    resolved = Path(raw_uri).expanduser().resolve()
    return f'file:///{resolved.as_posix().lstrip("/")}'


def resolve_artifact_location(*, artifact_location: str | None) -> str | None:
    """
    Normalize an optional artifact location.

    Args:
        artifact_location (str | None): Artifact location or filesystem path supplied by the user.

    Returns:
        str | None: Normalized artifact location or None when unset.
    """
    raw_uri = str(artifact_location).strip() if artifact_location is not None else ''
    if not raw_uri:
        return None
    return _normalize_artifact_location(raw_uri)


def resolve_tracking_uri(
    *,
    tracking_uri: str | None,
) -> str:
    """
    Resolve a tracking URI using MLflow-native semantics.

    Args:
        tracking_uri (str | None): Tracking URI supplied by the user.

    Returns:
        str: Effective tracking URI.
    """
    if tracking_uri is not None:
        return tracking_uri
    return mlflow.get_tracking_uri()


def normalize_tracking_uri(*, tracking_uri: str | None) -> str | None:
    """
    Normalize a tracking URI value for equality checks.

    Args:
        tracking_uri (str | None): Raw tracking URI.

    Returns:
        str | None: Stripped URI or None when unset/blank.
    """
    if tracking_uri is None:
        return None
    normalized = str(tracking_uri).strip()
    return normalized if normalized else None


def set_tracking_uri(
    *,
    tracking_uri: str | None,
) -> str:
    """
    Set the MLflow tracking URI using MLflow-native semantics.

    Args:
        tracking_uri (str | None): Tracking URI supplied by the user.

    Returns:
        str: Effective tracking URI.
    """
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.get_tracking_uri()


####################################
# Experiment/run lifecycle helpers #
####################################


def ensure_experiment(
    *,
    experiment_name: str,
    artifact_location: str | None,
) -> str:
    """
    Ensure an MLflow experiment exists, optionally enforcing artifact location.

    Args:
        experiment_name (str): Experiment name.
        artifact_location (str | None): Optional artifact location or filesystem path.

    Returns:
        str: Experiment id.

    Raises:
        ValueError: If the experiment exists with a different artifact location.
    """
    client = MlflowClient()
    existing = client.get_experiment_by_name(experiment_name)
    normalized_artifact_location = resolve_artifact_location(artifact_location=artifact_location)
    if existing is None:
        if normalized_artifact_location is not None:
            return client.create_experiment(name=experiment_name, artifact_location=normalized_artifact_location)
        return client.create_experiment(name=experiment_name)

    if normalized_artifact_location is not None:
        existing_location = resolve_artifact_location(artifact_location=existing.artifact_location)
        if existing_location is not None and existing_location != normalized_artifact_location:
            raise ValueError(
                'MLflow experiment already exists with a different artifact location. '
                f'Experiment={experiment_name}, existing={existing_location}, requested={normalized_artifact_location}. '
                'Use a new experiment name or delete the existing experiment to change artifact storage.'
            )

    return str(existing.experiment_id)


@contextlib.contextmanager
def init_mlflow(
    experiment_name: str = 'regain_experiments',
    run_name: str | None = None,
    tracking_uri: str | None = None,
    artifact_location: str | None = None,
) -> Iterator[mlflow.ActiveRun]:
    """
    Initialize an MLflow experiment and yield an active run context.

    Args:
        experiment_name: Name of the MLflow experiment.
        run_name: Optional run name.
        tracking_uri: Optional tracking URI.
        artifact_location: Optional artifact location or filesystem path.

    Yields:
        Active MLflow run object.
    """
    set_tracking_uri(tracking_uri=tracking_uri)
    if artifact_location is not None:
        experiment_id = ensure_experiment(experiment_name=experiment_name, artifact_location=artifact_location)
        mlflow.set_experiment(experiment_id=experiment_id)
    else:
        mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        yield run


def _log_fatal_error_artifact(
    *,
    run_name: str,
    exc: Exception,
    traceback_text: str,
) -> None:
    """
    Log a fatal error artifact for the active run.

    Args:
        run_name (str): Name of the run that failed.
        exc (Exception): Uncaught exception that caused the run failure.
        traceback_text (str): Formatted traceback text for the exception.

    Returns:
        None
    """
    if mlflow.active_run() is None:
        return
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    payload = (
        f'timestamp_utc: {timestamp_utc}\n'
        f'run_name: {run_name}\n'
        f'exception_type: {type(exc).__name__}\n'
        f'exception_message: {exc}\n'
        'traceback:\n'
        f'{traceback_text.rstrip()}\n'
    )
    try:
        mlflow.log_text(payload, MLFLOW_ARTIFACT_ERROR_FILE)
    except Exception:
        return


@contextlib.contextmanager
def log_fatal_error_context(
    *,
    run_name: str,
) -> Iterator[None]:
    """
    Capture uncaught exceptions and log a fatal error artifact for the active run.

    Args:
        run_name (str): Name of the run that failed.

    Yields:
        None
    """
    try:
        yield
    except Exception as exc:
        _log_fatal_error_artifact(
            run_name=run_name,
            exc=exc,
            traceback_text=traceback.format_exc(),
        )
        raise


################################
# Experiment search/query APIs #
################################


def resolve_experiment_id(
    *,
    client: MlflowClient,
    experiment: str,
) -> str:
    """
    Resolve an MLflow experiment id from a name or id.

    Args:
        client (MlflowClient): MLflow client instance.
        experiment (str): Experiment name or id.

    Returns:
        str: Experiment id.

    Raises:
        ValueError: If the experiment cannot be resolved.
    """
    def _try_name() -> str | None:
        exp = client.get_experiment_by_name(experiment)
        if exp is not None:
            return str(exp.experiment_id)
        return None

    def _try_id() -> str | None:
        try:
            exp = client.get_experiment(experiment_id=str(experiment))
        except Exception:
            return None
        if exp is not None:
            return str(exp.experiment_id)
        return None

    experiment_id: str | None = None
    if str(experiment).isdigit():
        experiment_id = _try_id() or _try_name()
    else:
        experiment_id = _try_name() or _try_id()

    if experiment_id is None:
        raise ValueError(f'No MLflow experiment found for: {experiment}')
    return experiment_id


def search_runs_paginated(
    *,
    client: MlflowClient,
    experiment_ids: Sequence[str],
    filter_string: str,
    run_view_type: mlflow.entities.ViewType | None = None,
    max_results: int = 1000,
    order_by: list[str] | None = None,
    max_runs: int | None = None,
) -> list[Run]:
    """
    Search MLflow runs with pagination.

    Args:
        client (MlflowClient): MLflow client instance.
        experiment_ids (Sequence[str]): Experiment IDs to search.
        filter_string (str): MLflow filter string.
        run_view_type (mlflow.entities.ViewType | None): Optional run view type filter.
        max_results (int): Max results per page.
        order_by (list[str] | None): Optional ordering clauses.
        max_runs (int | None): Optional total run limit.

    Returns:
        list[Run]: Runs matching the query.
    """
    all_runs: list[Run] = []
    page_token: str | None = None
    while True:
        kwargs: dict[str, object] = {
            'experiment_ids': list(experiment_ids),
            'filter_string': filter_string,
            'max_results': max_results,
            'page_token': page_token,
        }
        if run_view_type is not None:
            kwargs['run_view_type'] = run_view_type
        if order_by is not None:
            kwargs['order_by'] = order_by
        runs = client.search_runs(**kwargs)
        all_runs.extend(list(runs))
        if max_runs is not None and len(all_runs) >= int(max_runs):
            return all_runs[:int(max_runs)]
        page_token = getattr(runs, 'token', None) or getattr(runs, 'next_page_token', None)
        if not page_token:
            break
    return all_runs


######################
# Run info resolvers #
######################


def resolve_active_runs_by_name(
    *,
    experiment_name: str,
    tracking_uri: str | None,
) -> dict[str, list[object]]:
    """
    Resolve active MLflow runs for an experiment grouped by run name.

    Args:
        experiment_name (str): MLflow experiment name.
        tracking_uri (str | None): Optional MLflow tracking URI override.

    Returns:
        dict[str, list[object]]: Active runs grouped by resolved run name.
    """
    set_tracking_uri(tracking_uri=tracking_uri)
    client = MlflowClient()
    try:
        experiment_id = resolve_experiment_id(
            client=client,
            experiment=experiment_name,
        )
    except ValueError:
        return {}

    runs = search_runs_paginated(
        client=client,
        experiment_ids=[experiment_id],
        filter_string='',
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
    )
    grouped_runs: dict[str, list[object]] = {}
    for run in runs:
        run_name = str(resolve_mlflow_run_name(run=run)).strip()
        if not run_name:
            continue
        if run_name not in grouped_runs:
            grouped_runs[run_name] = []
        grouped_runs[run_name].append(run)
    return grouped_runs


def resolve_latest_active_runs_by_name(
    *,
    active_runs_by_name: dict[str, list[object]],
) -> dict[str, object]:
    """
    Resolve the latest active run per run name.

    Args:
        active_runs_by_name (dict[str, list[object]]): Active runs grouped by run name.

    Returns:
        dict[str, object]: Latest active run object per name.
    """
    latest_runs_by_name: dict[str, object] = {}
    for run_name, grouped_runs in active_runs_by_name.items():
        if not grouped_runs:
            continue
        sorted_runs = sorted(
            grouped_runs,
            key=lambda run: (
                int(getattr(getattr(run, 'info', None), 'start_time', 0) or 0),
                str(getattr(getattr(run, 'info', None), 'run_id', '') or ''),
            ),
            reverse=True,
        )
        latest_runs_by_name[run_name] = sorted_runs[0]
    return latest_runs_by_name


def delete_mlflow_runs(
    *,
    runs: list[object],
    tracking_uri: str | None,
) -> None:
    """
    Delete MLflow runs by id, deduplicating repeated ids.

    Args:
        runs (list[object]): Run-like objects with `.info.run_id`.
        tracking_uri (str | None): Optional MLflow tracking URI override.

    Returns:
        None
    """
    if not runs:
        return

    set_tracking_uri(tracking_uri=tracking_uri)
    client = MlflowClient()
    deleted_run_ids: set[str] = set()
    for run in runs:
        run_id = getattr(getattr(run, 'info', None), 'run_id', '')
        resolved_run_id = str(run_id) if run_id is not None else ''
        if resolved_run_id == '' or resolved_run_id in deleted_run_ids:
            continue
        client.delete_run(run_id=resolved_run_id)
        deleted_run_ids.add(resolved_run_id)


def resolve_mlflow_run_name(*, run: Run) -> str:
    """
    Resolve a stable run name from MLflow run data.

    Args:
        run (Run): MLflow run.

    Returns:
        str: Resolved run name (empty string when missing).
    """
    param_run_name = run.data.params.get(PARAM_RUN_NAME)
    if param_run_name:
        return str(param_run_name)

    info_name = getattr(run.info, PARAM_RUN_NAME, None)
    if info_name:
        return str(info_name)

    tags = dict(run.data.tags or {})
    tag_name = tags.get('mlflow.runName')
    if tag_name:
        return str(tag_name)
    return ''


########################
# Export/table helpers #
########################


def format_timestamp_ms(timestamp_ms: int | None) -> str:
    """
    Format a millisecond timestamp as an ISO-8601 UTC string.

    Args:
        timestamp_ms (int | None): Millisecond timestamp since epoch.

    Returns:
        str: ISO-8601 UTC timestamp string or empty string if unavailable.
    """
    if timestamp_ms is None:
        return ''
    return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc).isoformat()


def build_mlflow_run_columns(
    *,
    run: Run,
    include_params: bool = True,
    include_metrics: bool = True,
) -> dict[str, object]:
    """
    Build a flattened column map for a single MLflow run.

    Args:
        run (Run): MLflow run to flatten.
        include_params (bool): Whether to include parameter columns.
        include_metrics (bool): Whether to include metric columns.

    Returns:
        dict[str, object]: Flattened columns for the run.
    """
    columns: dict[str, object] = {}
    run_name = resolve_mlflow_run_name(run=run)

    columns[COLUMN_RUN_ID] = run.info.run_id
    columns[COLUMN_RUN_NAME] = run_name
    columns[COLUMN_STATUS] = run.info.status
    columns[COLUMN_START_TIME] = format_timestamp_ms(run.info.start_time)
    columns[COLUMN_END_TIME] = format_timestamp_ms(run.info.end_time)

    reserved_keys = {
        COLUMN_RUN_ID,
        COLUMN_RUN_NAME,
        COLUMN_STATUS,
        COLUMN_START_TIME,
        COLUMN_END_TIME,
    }
    if include_params:
        for param_key, param_value in run.data.params.items():
            if param_key in reserved_keys:
                continue
            columns[param_key] = param_value
    if include_metrics:
        for metric_key, metric_value in run.data.metrics.items():
            if metric_key in reserved_keys:
                continue
            columns[metric_key] = metric_value
    return columns


def write_experiment_meta_yaml(*, experiment: Experiment, output_dir: Path) -> None:
    """
    Write experiment metadata to a meta.yaml file.

    Args:
        experiment (Experiment): MLflow experiment metadata.
        output_dir (Path): Directory where meta.yaml should be written.

    Returns:
        None
    """
    experiment_dict = dict(experiment)
    experiment_dict[COLUMN_EXPERIMENT_ID] = str(experiment.experiment_id)
    write_yaml(str(output_dir), 'meta.yaml', experiment_dict, overwrite=True)


####################
# Artifact helpers #
####################


def download_json_artifact(
    *,
    client: MlflowClient,
    run_id: str,
    artifact_path: str,
) -> dict[str, object] | None:
    """
    Download and parse a JSON artifact from an MLflow run.

    Args:
        client (MlflowClient): MLflow client.
        run_id (str): Source run id.
        artifact_path (str): Artifact path in the run.

    Returns:
        dict[str, object] | None: Parsed artifact payload, if available.
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            downloaded_path = client.download_artifacts(
                run_id,
                artifact_path,
                temp_dir,
            )
            local_path = Path(downloaded_path)
            if local_path.is_dir():
                local_path = local_path / Path(artifact_path).name
            if not local_path.exists():
                return None
            return json.loads(local_path.read_text(encoding='utf-8'))
    except Exception:
        try:
            local_path = Path(
                mlflow.artifacts.download_artifacts(
                    run_id=run_id,
                    artifact_path=artifact_path,
                )
            )
            if local_path.is_dir():
                local_path = local_path / Path(artifact_path).name
            if not local_path.exists():
                return None
            return json.loads(local_path.read_text(encoding='utf-8'))
        except Exception:
            return None
