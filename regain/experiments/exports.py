"""
Helpers for exporting MLflow runs.
"""

import csv
from pathlib import Path
from typing import Any

from mlflow.tracking import MlflowClient

from regain.constants import COLUMN_END_TIME
from regain.constants import COLUMN_GIT_COMMIT
from regain.constants import COLUMN_RUN_ID
from regain.constants import COLUMN_RUN_NAME
from regain.constants import COLUMN_START_TIME
from regain.constants import COLUMN_STATUS
from regain.mlflow_utils import build_mlflow_run_columns
from regain.mlflow_utils import resolve_experiment_id
from regain.mlflow_utils import search_runs_paginated
from regain.mlflow_utils import set_tracking_uri
from regain.mlflow_utils import write_experiment_meta_yaml

__all__ = [
    'export_runs_to_csvs',
]


def export_runs_to_csvs(
    *,
    experiment: str,
    metadata_path: Path,
    params_path: Path,
    metrics_path: Path,
    tracking_uri: str | None,
) -> None:
    """
    Export all MLflow runs for an experiment into CSV files and write experiment metadata (`meta.yaml`).

    If export files already exist, they are overwritten to capture the latest snapshot/state.

    Args:
        experiment (str): MLflow experiment name or id.
        metadata_path (Path): Output CSV path for metadata.
        params_path (Path): Output CSV path for params.
        metrics_path (Path): Output CSV path for metrics.
        tracking_uri (str | None): Optional MLflow tracking URI.

    Returns:
        None

    Raises:
        OSError: If writing a CSV fails.
        ValueError: If the experiment cannot be resolved.
    """
    set_tracking_uri(tracking_uri=tracking_uri)

    client = MlflowClient()
    experiment_id = resolve_experiment_id(
        client=client,
        experiment=experiment,
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

    for run in all_runs:
        row = build_mlflow_run_columns(run=run, client=client)
        rows.append(row)
        parent_param_keys.update(run.data.params.keys())

    metadata_columns = [
        COLUMN_RUN_ID,
        COLUMN_RUN_NAME,
        COLUMN_STATUS,
        COLUMN_START_TIME,
        COLUMN_END_TIME,
        COLUMN_GIT_COMMIT,
    ]
    reserved_keys = set(metadata_columns)
    param_columns = sorted(key for key in parent_param_keys if key not in reserved_keys)
    metric_columns = sorted({
        key
        for row in rows
        for key in row.keys()
        if key not in reserved_keys and key not in parent_param_keys
    })

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    write_experiment_meta_yaml(
        experiment=experiment_meta,
        output_dir=metadata_path.parent,
    )

    metadata_fieldnames = metadata_columns
    row_identity_columns = [
        COLUMN_RUN_ID,
        COLUMN_RUN_NAME,
    ]
    params_fieldnames = row_identity_columns + param_columns
    metrics_fieldnames = row_identity_columns + metric_columns

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
