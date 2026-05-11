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
from mlflow.utils.git_utils import get_git_commit
from mlflow.utils.mlflow_tags import MLFLOW_GIT_COMMIT
from mlflow.utils.yaml_utils import write_yaml

from regain.constants import COLUMN_END_TIME
from regain.constants import COLUMN_EXPERIMENT_ID
from regain.constants import COLUMN_GIT_COMMIT
from regain.constants import COLUMN_RUN_ID
from regain.constants import COLUMN_RUN_NAME
from regain.constants import COLUMN_START_TIME
from regain.constants import COLUMN_STATUS
from regain.constants import EXPERIENCE_KEY_PREFIX
from regain.constants import MLFLOW_ARTIFACT_ERROR_FILE
from regain.constants import NAMESPACE_EVAL
from regain.constants import NS_SEP
from regain.constants import PARAM_RUN_NAME
from regain.constants import RUN_ACC_REF

__all__ = [
    'build_mlflow_run_columns',
    'delete_mlflow_runs',
    'download_json_artifact',
    'ensure_experiment',
    'extract_mlflow_run_git_commit',
    'format_timestamp_ms',
    'init_mlflow',
    'log_fatal_error_context',
    'normalize_metric_name',
    'normalize_tracking_uri',
    'resolve_active_runs_by_name',
    'resolve_artifact_location',
    'resolve_experiment_id',
    'resolve_git_commit',
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
_HISTORY_BEARING_EVAL_METRIC_PREFIXES = (
    f'{NAMESPACE_EVAL}{NS_SEP}forgetting{NS_SEP}',
    f'{NAMESPACE_EVAL}{NS_SEP}transfer{NS_SEP}',
)
_REF_METRIC_RE = re.compile(
    rf'^{re.escape(RUN_ACC_REF)}'
    rf'{_NS_SEP_ESCAPED}{re.escape(EXPERIENCE_KEY_PREFIX)}(?P<idx>\d+)'
    rf'{_NS_SEP_ESCAPED}base$'
)


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


def extract_mlflow_run_git_commit(run: Run) -> str:
    """
    Return the git commit tag for an MLflow run, or '' if absent.

    Args:
        run (Run): MLflow run whose tags should be inspected.

    Returns:
        str: Git commit tag value, or an empty string when unavailable.
    """
    tags = dict(getattr(run.data, 'tags', {}) or {})
    value = tags.get(MLFLOW_GIT_COMMIT)
    return str(value) if value else ''


def resolve_git_commit(repo_path: str | Path | None = None) -> str | None:
    """
    Return the current repository git commit at call time.

    Args:
        repo_path (str | Path | None): Optional repository path override.

    Returns:
        str | None: Git commit hash, or None when no git repo can be found.
    """
    target = str(Path(repo_path) if repo_path is not None else Path(__file__).parent)
    return get_git_commit(target)


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

    tags = dict(getattr(run.data, 'tags', {}) or {})
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


def _is_history_bearing_eval_metric(*, metric_key: str) -> bool:
    """
    Check whether an eval metric must be materialized from MLflow step history.

    Args:
        metric_key (str): Metric key stored in MLflow.

    Returns:
        bool: True when the exporter should inject an `after_exp###` token.
    """
    metric_key_str = str(metric_key)
    return any(
        metric_key_str.startswith(prefix)
        for prefix in _HISTORY_BEARING_EVAL_METRIC_PREFIXES
    )


def _insert_after_experience_token(
    *,
    metric_key: str,
    after_exp_idx: int,
) -> str:
    """
    Insert an `after_exp###` token into a retained eval metric key.

    Args:
        metric_key (str): Canonical metric key such as `run.eval.forgetting.exp000`.
        after_exp_idx (int): History ordinal derived from MLflow steps.

    Returns:
        str: Flattened export key with checkpoint identity materialized.
    """
    parts = str(metric_key).split(NS_SEP)
    if len(parts) < 4:
        return str(metric_key)
    after_token = f'after_exp{int(after_exp_idx):03d}'
    return NS_SEP.join(parts[:3] + [after_token] + parts[3:])


def _require_metric_history(
    *,
    client: MlflowClient | None,
    run_id: str,
    metric_key: str,
) -> list[Any]:
    """
    Fetch metric history required for strict export materialization.

    Args:
        client (MlflowClient | None): MLflow client used to fetch history.
        run_id (str): Source run identifier.
        metric_key (str): Metric key stored in MLflow.

    Returns:
        list[Any]: Raw MLflow metric history entries.

    Raises:
        ValueError: If the client is missing, the lookup fails, or no history exists.
    """
    if client is None:
        raise ValueError(
            'History-bearing eval metric export requires an MLflow client. '
            f'run_id={run_id}, metric_key={metric_key}'
        )

    try:
        history = list(client.get_metric_history(str(run_id), str(metric_key)))
    except Exception as exc:
        raise ValueError(
            'Failed to fetch required MLflow metric history. '
            f'run_id={run_id}, metric_key={metric_key}'
        ) from exc

    if not history:
        raise ValueError(
            'Missing required MLflow metric history. '
            f'run_id={run_id}, metric_key={metric_key}'
        )
    return history


def _ref_checkpoint_step_map(
    *,
    run: Run,
    client: MlflowClient | None,
) -> dict[int, int]:
    """
    Resolve checkpoint identity from reference test-accuracy histories.

    Args:
        run (Run): MLflow run whose metrics are being exported.
        client (MlflowClient | None): MLflow client used to fetch history.

    Returns:
        dict[int, int]: Mapping from MLflow step to `after_exp###` index.

    Raises:
        ValueError: If required reference histories are missing or ambiguous.
    """
    run_id = str(run.info.run_id)
    metrics = dict(getattr(run.data, 'metrics', {}) or {})
    ref_metric_pairs: list[tuple[int, str]] = []
    for metric_key in metrics:
        match = _REF_METRIC_RE.match(str(metric_key))
        if match is None:
            continue
        ref_metric_pairs.append((int(match.group('idx')), str(metric_key)))

    if not ref_metric_pairs:
        raise ValueError(
            'Missing required reference accuracy metrics for history-bearing export. '
            f'run_id={run_id}, required_prefix={RUN_ACC_REF}'
        )

    observed_exp_indices = sorted(exp_idx for exp_idx, _ in ref_metric_pairs)
    expected_exp_indices = list(range(observed_exp_indices[-1] + 1))
    if observed_exp_indices != expected_exp_indices:
        missing_exp_indices = sorted(set(expected_exp_indices) - set(observed_exp_indices))
        missing_tokens = ', '.join(f'exp{exp_idx:03d}' for exp_idx in missing_exp_indices)
        raise ValueError(
            'Missing required reference accuracy metrics for history-bearing export. '
            f'run_id={run_id}, missing={missing_tokens}'
        )

    step_map: dict[int, int] = {}
    for exp_idx, metric_key in sorted(ref_metric_pairs):
        history = _require_metric_history(
            client=client,
            run_id=run_id,
            metric_key=metric_key,
        )
        distinct_steps = {
            int(getattr(metric, 'step', 0) or 0)
            for metric in history
        }
        if len(distinct_steps) != 1:
            raise ValueError(
                'Reference accuracy history must map to exactly one checkpoint step. '
                f'run_id={run_id}, metric_key={metric_key}, steps={sorted(distinct_steps)}'
            )
        step = next(iter(distinct_steps))
        if step in step_map:
            raise ValueError(
                'Reference accuracy histories map multiple experiences to the same checkpoint step. '
                f'run_id={run_id}, step={step}, after_exp_existing={step_map[step]:03d}, '
                f'after_exp_new={exp_idx:03d}'
            )
        step_map[step] = exp_idx
    return step_map


def _materialize_metric_history_columns(
    *,
    client: MlflowClient | None,
    run_id: str,
    metric_key: str,
    checkpoint_step_map: Mapping[int, int],
) -> dict[str, float]:
    """
    Expand one history-bearing eval metric into `after_exp###` export columns.

    Args:
        client (MlflowClient | None): MLflow client used to fetch metric history.
        run_id (str): Source run identifier.
        metric_key (str): Metric key stored in MLflow.
        checkpoint_step_map (Mapping[int, int]): Mapping from MLflow step to checkpoint identity.

    Returns:
        dict[str, float]: Export columns keyed by materialized metric names.

    Raises:
        ValueError: If the metric history is missing or cannot be mapped to checkpoints.
    """
    history = _require_metric_history(
        client=client,
        run_id=run_id,
        metric_key=metric_key,
    )

    latest_value_by_step: dict[int, float] = {}
    for metric in history:
        step = int(getattr(metric, 'step', 0) or 0)
        latest_value_by_step[step] = float(metric.value)

    flattened_columns: dict[str, float] = {}
    ordered_steps = sorted(latest_value_by_step)
    for step in ordered_steps:
        if step not in checkpoint_step_map:
            # Avalanche emits a pre-training bootstrap history point at step=0 for both
            # stream- and per-experience forgetting/transfer metrics, because their reporter
            # uses `strategy.clock.train_iterations` which is 0 before the first training
            # iteration. Reference test accuracies are only logged after each training
            # experience completes (step >= num_epochs), so step=0 can never appear in the
            # checkpoint step map. Drop the bootstrap point — there is no recoverable signal
            # before training begins — but keep raising on any other unmapped step.
            if step == 0:
                continue
            raise ValueError(
                'History-bearing eval metric uses a step with no matching checkpoint identity. '
                f'run_id={run_id}, metric_key={metric_key}, step={step}'
            )
        flattened_key = _insert_after_experience_token(
            metric_key=str(metric_key),
            after_exp_idx=int(checkpoint_step_map[step]),
        )
        flattened_columns[flattened_key] = float(latest_value_by_step[step])
    return flattened_columns


def build_mlflow_run_columns(
    *,
    run: Run,
    client: MlflowClient | None = None,
    include_params: bool = True,
    include_metrics: bool = True,
) -> dict[str, object]:
    """
    Build a flattened column map for a single MLflow run.

    Args:
        run (Run): MLflow run to flatten.
        client (MlflowClient | None): Optional MLflow client for metric history expansion.
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
    columns[COLUMN_GIT_COMMIT] = extract_mlflow_run_git_commit(run)

    reserved_keys = {
        COLUMN_GIT_COMMIT,
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
        checkpoint_step_map: dict[int, int] | None = None
        for metric_key, metric_value in run.data.metrics.items():
            if metric_key in reserved_keys:
                continue
            if _is_history_bearing_eval_metric(metric_key=metric_key):
                if checkpoint_step_map is None:
                    checkpoint_step_map = _ref_checkpoint_step_map(
                        run=run,
                        client=client,
                    )
                history_columns = _materialize_metric_history_columns(
                    client=client,
                    run_id=str(run.info.run_id),
                    metric_key=str(metric_key),
                    checkpoint_step_map=checkpoint_step_map,
                )
                columns.update(history_columns)
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
