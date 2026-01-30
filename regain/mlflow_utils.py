"""
MLflow utilities.
"""

import os
from pathlib import Path
from typing import Final
from typing import Sequence
from urllib.parse import urlparse

import mlflow
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

__all__ = [
    'resolve_sqlite_tracking_uri',
    'set_sqlite_tracking_uri',
    'resolve_experiment_id',
    'search_runs_paginated',
]


_DEFAULT_SQLITE_DB_NAME: Final[str] = 'mlflow.db'


def _default_sqlite_path(default_dir: Path | None) -> Path:
    """
    Resolve the default SQLite database path.

    Args:
        default_dir (Path | None): Optional base directory for the SQLite DB file.

    Returns:
        Path: Path to the default SQLite database file.
    """
    base_dir = Path.cwd() if default_dir is None else Path(default_dir)
    return base_dir / _DEFAULT_SQLITE_DB_NAME


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


def resolve_sqlite_tracking_uri(
    *,
    tracking_uri: str | None,
    default_dir: Path | None = None,
) -> str:
    """
    Normalize a tracking URI to a SQLite backend.

    Falls back to `MLFLOW_TRACKING_URI` (if set) or `./mlflow.db` when no URI is provided.

    Args:
        tracking_uri (str | None): Tracking URI or filesystem path supplied by the user.
        default_dir (Path | None): Optional base directory for the default SQLite DB.

    Returns:
        str: SQLite tracking URI.

    Raises:
        ValueError: If the tracking URI uses a non-SQLite scheme.
    """
    raw_uri = str(tracking_uri).strip() if tracking_uri is not None else ''
    if not raw_uri:
        env_uri = os.environ.get('MLFLOW_TRACKING_URI', '').strip()
        raw_uri = env_uri
    if not raw_uri:
        return _path_to_sqlite_uri(_default_sqlite_path(default_dir=default_dir))
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


def set_sqlite_tracking_uri(
    *,
    tracking_uri: str | None,
    default_dir: Path | None = None,
) -> str:
    """
    Resolve and set the MLflow tracking URI, forcing SQLite as the backend.

    Args:
        tracking_uri (str | None): Tracking URI or filesystem path supplied by the user.
        default_dir (Path | None): Optional base directory for the default SQLite DB.

    Returns:
        str: Normalized SQLite tracking URI.
    """
    resolved = resolve_sqlite_tracking_uri(tracking_uri=tracking_uri, default_dir=default_dir)
    mlflow.set_tracking_uri(resolved)
    return resolved


def resolve_experiment_id(
    *,
    client: MlflowClient,
    experiment: str,
    prefer_name: bool = True,
    raise_on_missing: bool = True,
) -> str | None:
    """
    Resolve an MLflow experiment id from a name or id.

    Args:
        client (MlflowClient): MLflow client instance.
        experiment (str): Experiment name or id.
        prefer_name (bool): Prefer resolving by name before id when possible.
        raise_on_missing (bool): Whether to raise if the experiment cannot be resolved.

    Returns:
        str | None: Experiment id if found.

    Raises:
        ValueError: If the experiment cannot be resolved and raise_on_missing is True.
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
    if prefer_name:
        experiment_id = _try_name() or _try_id()
    else:
        if str(experiment).isdigit():
            experiment_id = _try_id() or _try_name()
        else:
            experiment_id = _try_name()

    if experiment_id is None and raise_on_missing:
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
