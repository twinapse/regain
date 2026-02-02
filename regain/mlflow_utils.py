"""
MLflow utilities.
"""

import contextlib
from pathlib import Path
from typing import Final
from typing import Iterator
from typing import Sequence
from urllib.parse import urlparse

import mlflow
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

__all__ = [
    'resolve_tracking_uri',
    'resolve_artifact_uri',
    'set_tracking_uri',
    'ensure_experiment',
    'init_mlflow',
    'resolve_experiment_id',
    'search_runs_paginated',
]


_DEFAULT_SQLITE_DB_NAME: Final[str] = 'mlflow.db'


def _path_to_sqlite_uri(path: Path) -> str:
    """
    Convert a filesystem path into a SQLite tracking URI.

    Args:
        path (Path): SQLite database path.

    Returns:
        str: SQLite tracking URI.
    """
    resolved = path.expanduser()
    if resolved.exists() and resolved.is_dir():
        raise ValueError(
            f'MLflow tracking URI must point to a SQLite database file, not a directory: {resolved}'
        )
    return f'sqlite:///{resolved.as_posix()}'


def _normalize_artifact_uri(raw_uri: str) -> str:
    """
    Normalize an artifact URI to a stable representation.

    Args:
        raw_uri (str): Artifact URI or filesystem path.

    Returns:
        str: Normalized artifact URI.
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


def resolve_artifact_uri(*, artifact_uri: str | None) -> str | None:
    """
    Normalize an optional artifact URI.

    Args:
        artifact_uri (str | None): Artifact URI or filesystem path supplied by the user.

    Returns:
        str | None: Normalized artifact URI or None when unset.
    """
    raw_uri = str(artifact_uri).strip() if artifact_uri is not None else ''
    if not raw_uri:
        return None
    return _normalize_artifact_uri(raw_uri)


def resolve_tracking_uri(
    *,
    tracking_uri: str | None,
) -> str:
    """
    Normalize a tracking URI to a SQLite backend.

    Falls back to `./mlflow.db` when no URI is provided.

    Args:
        tracking_uri (str | None): Tracking URI or filesystem path supplied by the user.

    Returns:
        str: SQLite tracking URI.

    Raises:
        ValueError: If the tracking URI uses a non-SQLite scheme.
    """
    raw_uri = str(tracking_uri).strip() if tracking_uri is not None else ''
    if not raw_uri:
        return _path_to_sqlite_uri(Path.cwd() / _DEFAULT_SQLITE_DB_NAME)
    parsed = urlparse(raw_uri)
    if parsed.scheme:
        if parsed.scheme != 'sqlite':
            if len(parsed.scheme) == 1 and raw_uri[1:3] in {':\\', ':/'}:
                return _path_to_sqlite_uri(Path(raw_uri))
            raise ValueError(
                'MLflow tracking URI must use a SQLite backend '
                f"(e.g., 'sqlite:///path/to/mlflow.db'). Got: {raw_uri}"
            )
        return raw_uri

    return _path_to_sqlite_uri(Path(raw_uri))


def set_tracking_uri(
    *,
    tracking_uri: str | None,
) -> str:
    """
    Resolve and set the MLflow tracking URI, forcing SQLite as the backend.

    Args:
        tracking_uri (str | None): Tracking URI or filesystem path supplied by the user.

    Returns:
        str: Normalized SQLite tracking URI.
    """
    resolved = resolve_tracking_uri(tracking_uri=tracking_uri)
    mlflow.set_tracking_uri(resolved)
    return resolved


def ensure_experiment(
    *,
    experiment_name: str,
    artifact_uri: str | None,
) -> str:
    """
    Ensure an MLflow experiment exists, optionally enforcing artifact location.

    Args:
        experiment_name (str): Experiment name.
        artifact_uri (str | None): Optional artifact URI or filesystem path.

    Returns:
        str: Experiment id.

    Raises:
        ValueError: If the experiment exists with a different artifact location.
    """
    client = MlflowClient()
    existing = client.get_experiment_by_name(experiment_name)
    normalized_artifact_uri = resolve_artifact_uri(artifact_uri=artifact_uri)
    if existing is None:
        if normalized_artifact_uri is not None:
            return client.create_experiment(name=experiment_name, artifact_location=normalized_artifact_uri)
        return client.create_experiment(name=experiment_name)

    if normalized_artifact_uri is not None:
        existing_location = resolve_artifact_uri(artifact_uri=existing.artifact_location)
        if existing_location is not None and existing_location != normalized_artifact_uri:
            raise ValueError(
                'MLflow experiment already exists with a different artifact location. '
                f'Experiment={experiment_name}, existing={existing_location}, requested={normalized_artifact_uri}. '
                'Use a new experiment name or delete the existing experiment to change artifact storage.'
            )

    return str(existing.experiment_id)


@contextlib.contextmanager
def init_mlflow(
    experiment_name: str = 'regain_experiments',
    run_name: str | None = None,
    tracking_uri: str | None = None,
    artifact_uri: str | None = None,
) -> Iterator[mlflow.ActiveRun]:
    """
    Initialize an MLflow experiment and yield an active run context.

    Args:
        experiment_name: Name of the MLflow experiment.
        run_name: Optional run name.
        tracking_uri: Optional tracking URI or filesystem path (SQLite only).
        artifact_uri: Optional artifact URI or filesystem path.

    Yields:
        Active MLflow run object.
    """
    set_tracking_uri(tracking_uri=tracking_uri)
    if artifact_uri is not None:
        experiment_id = ensure_experiment(experiment_name=experiment_name, artifact_uri=artifact_uri)
        mlflow.set_experiment(experiment_id=experiment_id)
    else:
        mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        yield run


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
