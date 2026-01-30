"""
Helpers for exporting MLflow runs.
"""

import csv
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

from mlflow.entities import Experiment
from mlflow.entities import Run
from mlflow.tracking import MlflowClient
from mlflow.utils.yaml_utils import write_yaml

from regain.mlflow_utils import resolve_experiment_id
from regain.mlflow_utils import search_runs_paginated
from regain.mlflow_utils import set_sqlite_tracking_uri

__all__ = [
    'export_runs_csv',
]


def _format_timestamp_ms(timestamp_ms: int | None) -> str:
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


def _resolve_run_name(run: Run) -> str:
    """
    Resolve a human-readable MLflow run name.

    Args:
        run (Run): MLflow run to inspect.

    Returns:
        str: Resolved run name (empty string if missing).
    """
    if run.info.run_name:
        return str(run.info.run_name)
    tag_name = run.data.tags.get('mlflow.runName')
    if tag_name:
        return str(tag_name)
    return ''


def _build_run_columns(
    run: Run,
    *,
    include_params: bool = True,
    include_metrics: bool = True,
) -> dict[str, Any]:
    """
    Build a flattened column map for a single MLflow run.

    Args:
        run (Run): MLflow run to flatten.
        include_params (bool): Whether to include parameter columns.
        include_metrics (bool): Whether to include metric columns.

    Returns:
        dict[str, Any]: Flattened columns for the run.
    """
    columns: dict[str, Any] = {}

    columns['run_id'] = run.info.run_id
    columns['run_name'] = _resolve_run_name(run)
    columns['parent_run_id'] = run.data.tags.get('mlflow.parentRunId', '')
    columns['status'] = run.info.status
    columns['start_time'] = _format_timestamp_ms(run.info.start_time)
    columns['end_time'] = _format_timestamp_ms(run.info.end_time)

    reserved_keys = {'run_id', 'run_name', 'parent_run_id', 'status', 'start_time', 'end_time'}
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


def _write_experiment_meta(*, experiment: Experiment, output_dir: Path) -> None:
    """
    Write experiment metadata to a meta.yaml file.

    Args:
        experiment (Experiment): MLflow experiment metadata.
        output_dir (Path): Directory where meta.yaml should be written.

    Returns:
        None
    """
    experiment_dict = dict(experiment)
    experiment_dict['experiment_id'] = str(experiment.experiment_id)
    write_yaml(str(output_dir), 'meta.yaml', experiment_dict)


def export_runs_csv(
    *,
    experiment: str,
    metadata_path: Path,
    params_path: Path,
    metrics_path: Path,
    tracking_uri: str | None,
) -> None:
    """
    Export all MLflow runs for an experiment into CSV files and write meta.yaml.

    Args:
        experiment (str): MLflow experiment name or id.
        metadata_path (Path): Output CSV path for metadata.
        params_path (Path): Output CSV path for params.
        metrics_path (Path): Output CSV path for metrics.
        tracking_uri (str | None): Optional MLflow tracking URI or filesystem path (SQLite only).

    Returns:
        None

    Raises:
        FileExistsError: If an export path already exists.
        OSError: If writing a CSV fails.
        ValueError: If the tracking URI is not SQLite or the experiment cannot be resolved.
    """
    set_sqlite_tracking_uri(tracking_uri=tracking_uri)

    client = MlflowClient()
    experiment_id = resolve_experiment_id(
        client=client,
        experiment=experiment,
        prefer_name=False,
        raise_on_missing=True,
    )

    experiment_meta = client.get_experiment(experiment_id)
    if experiment_meta is None:
        raise ValueError(f'No MLflow experiment found for: {experiment}')

    all_runs = search_runs_paginated(
        client=client,
        experiment_ids=[experiment_id],
        filter_string='',
    )
    rows: list[dict[str, Any]] = []
    parent_param_keys: set[str] = set()
    parent_metric_keys: set[str] = set()

    for run in all_runs:
        rows.append(_build_run_columns(run=run))
        parent_param_keys.update(run.data.params.keys())
        parent_metric_keys.update(run.data.metrics.keys())

    parent_metadata = ['run_id', 'run_name', 'parent_run_id', 'status', 'start_time', 'end_time']
    reserved_keys = {'run_id', 'run_name', 'parent_run_id', 'status', 'start_time', 'end_time'}
    param_columns = sorted(key for key in parent_param_keys if key not in reserved_keys)
    metric_columns = sorted(key for key in parent_metric_keys if key not in reserved_keys)

    meta_path = metadata_path.parent / 'meta.yaml'
    existing_paths = [
        path for path in (metadata_path, params_path, metrics_path, meta_path) if path.exists()
    ]
    if existing_paths:
        existing_list = ', '.join(str(path) for path in existing_paths)
        raise FileExistsError(f'Export artifacts already exist: {existing_list}')

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    _write_experiment_meta(experiment=experiment_meta, output_dir=metadata_path.parent)

    metadata_fieldnames = parent_metadata
    params_fieldnames = ['run_id', 'run_name', 'parent_run_id'] + param_columns
    metrics_fieldnames = ['run_id', 'run_name', 'parent_run_id'] + metric_columns

    with metadata_path.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=metadata_fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in metadata_fieldnames})

    with params_path.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=params_fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            if not any(row.get(key, '') != '' for key in param_columns):
                continue
            writer.writerow({key: row.get(key, '') for key in params_fieldnames})

    with metrics_path.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=metrics_fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            if not any(row.get(key, '') != '' for key in metric_columns):
                continue
            writer.writerow({key: row.get(key, '') for key in metrics_fieldnames})
