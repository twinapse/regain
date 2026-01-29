"""
Helpers for exporting analysis outputs.
"""

import csv
from datetime import datetime
from datetime import timezone
import json
from pathlib import Path
from typing import Any

__all__ = [
    'export_analysis_json',
]


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    """
    Read a CSV into a list of dict rows.

    Args:
        path (Path): Path to CSV file.

    Returns:
        list[dict[str, Any]]: List of rows as dicts.
    """
    if not path.exists():
        return []
    with path.open('r', newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]


def _coerce_csv_value(value: str | None) -> Any:
    """
    Coerce a CSV string value into a JSON-friendly scalar type.

    Args:
        value (str | None): CSV value.

    Returns:
        Any: Coerced value as int, float, bool, None, or str.
    """
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    lowered = raw.lower()
    if lowered in {'none', 'null', 'nan', 'inf', '-inf', 'infinity', '-infinity'}:
        return None
    if lowered == 'true':
        return True
    if lowered == 'false':
        return False
    if raw.lstrip('+-').isdigit():
        try:
            return int(raw)
        except ValueError:
            return raw
    try:
        return float(raw)
    except ValueError:
        return raw


def _coerce_csv_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Coerce all values in a list of CSV row dictionaries.

    Args:
        rows (list[dict[str, Any]]): Raw CSV rows.

    Returns:
        list[dict[str, Any]]: Rows with values coerced to scalar types.
    """
    coerced_rows: list[dict[str, Any]] = []
    for row in rows:
        coerced_rows.append({key: _coerce_csv_value(value) for key, value in row.items()})
    return coerced_rows


def export_analysis_json(
    *,
    experiment: str,
    experiment_dir: Path,
    export_path: Path,
    tracking_uri: str | None,
    runs_table: list[dict[str, Any]],
    experiences_table: list[dict[str, Any]],
    include_controllers: list[str] | None,
    exclude_controllers: list[str] | None,
    max_runs: int | None,
    default_num_classes: int | None,
    require_finished: bool,
) -> None:
    """
    Write a self-contained JSON bundle of analysis outputs and inputs.

    Args:
        experiment (str): MLflow experiment name or id.
        experiment_dir (Path): Analysis output directory for a single experiment.
        export_path (Path): Output JSON path.
        tracking_uri (str | None): Optional MLflow tracking URI.
        runs_table (list[dict[str, Any]]): Table rows for runs.
        experiences_table (list[dict[str, Any]]): Table rows for experiences.
        include_controllers (list[str] | None): Parsed controller allowlist.
        exclude_controllers (list[str] | None): Parsed controller denylist.
        max_runs (int | None): Optional maximum number of parent runs.
        default_num_classes (int | None): Optional default class count.
        require_finished (bool): Whether to require finished runs.

    Returns:
        None

    Raises:
        FileExistsError: If the export path already exists.
        OSError: If writing the export file fails.
        ValueError: If the export payload cannot be serialized.
    """
    curves_dir = experiment_dir / 'curves'
    frontier_dir = experiment_dir / 'frontier'
    recoverability_path = curves_dir / 'recoverability_curve.csv'
    task_age_path = curves_dir / 'task_age_rho.csv'
    frontier_points_path = frontier_dir / 'frontier_points.csv'
    frontier_pareto_path = frontier_dir / 'frontier_pareto.csv'

    missing_sections: list[str] = []

    def _read_section(path: Path, name: str) -> list[dict[str, Any]]:
        if path.exists():
            return _coerce_csv_rows(_read_csv_rows(path))
        missing_sections.append(name)
        return []

    export_payload: dict[str, Any] = {
        'schema': {
            'name': 'regain.analysis.export',
            'version': 1,
        },
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'mlflow': {
            'experiment': str(experiment),
            'tracking_uri': tracking_uri,
            'include_controllers': include_controllers,
            'exclude_controllers': exclude_controllers,
            'max_runs': max_runs,
            'default_num_classes': default_num_classes,
            'require_finished': require_finished,
        },
        'tables': {
            'runs_table': runs_table,
            'experiences_table': experiences_table,
        },
        'curves': {
            'recoverability_curve': _read_section(
                recoverability_path,
                'curves.recoverability_curve',
            ),
            'task_age_rho': _read_section(task_age_path, 'curves.task_age_rho'),
        },
        'frontier': {
            'points': _read_section(frontier_points_path, 'frontier.points'),
            'pareto': _read_section(frontier_pareto_path, 'frontier.pareto'),
        },
        'notes': {
            'plots_embedded': False,
        },
    }

    if missing_sections:
        export_payload['missing_sections'] = missing_sections

    if export_path.exists():
        raise FileExistsError(f'Export JSON already exists: {export_path}')

    export_path.parent.mkdir(parents=True, exist_ok=True)
    with export_path.open('w', encoding='utf-8') as f:
        json.dump(export_payload, f, indent=2)
