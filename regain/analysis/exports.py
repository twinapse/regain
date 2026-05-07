"""
Helpers for exporting analysis outputs.
"""

import csv
from datetime import datetime
from datetime import timezone
import json
from pathlib import Path
from typing import Any

from regain.mlflow_utils import resolve_artifact_location
from regain.mlflow_utils import resolve_tracking_uri

__all__ = [
    'export_analysis_to_json',
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


def _read_json_payload(path: Path) -> dict[str, Any] | list[Any] | None:
    """
    Read a JSON file into a Python payload.

    Args:
        path: JSON path.

    Returns:
        dict[str, Any] | list[Any] | None: Parsed JSON payload, or None when missing.
    """
    if not path.exists():
        return None
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """
    Read newline-delimited JSON rows from disk.

    Args:
        path: JSONL file path.

    Returns:
        list[dict[str, Any]]: Parsed object rows.
    """
    rows: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            payload = line.strip()
            if not payload:
                continue
            row = json.loads(payload)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def export_analysis_to_json(
    *,
    experiment: str,
    experiment_dir: Path,
    export_path: Path,
    tracking_uri: str | None,
    artifact_location: str | None,
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

    If the export path already exists, it is overwritten to capture the latest snapshot/state.

    Args:
        experiment (str): MLflow experiment name or id.
        experiment_dir (Path): Analysis output directory for a single experiment.
        export_path (Path): Output JSON path.
        tracking_uri (str | None): Optional MLflow tracking URI.
        artifact_location (str | None): Optional MLflow artifact location or filesystem path.
        runs_table (list[dict[str, Any]]): Table rows for runs.
        experiences_table (list[dict[str, Any]]): Table rows for experiences.
        include_controllers (list[str] | None): Parsed controller allowlist.
        exclude_controllers (list[str] | None): Parsed controller denylist.
        max_runs (int | None): Optional maximum number of runs.
        default_num_classes (int | None): Optional default class count.
        require_finished (bool): Whether to require finished runs.

    Returns:
        None

    Raises:
        OSError: If writing the export file fails.
        ValueError: If the export payload cannot be serialized.
    """
    curves_dir = experiment_dir / 'curves'
    frontier_dir = experiment_dir / 'frontier'
    recoverability_path = curves_dir / 'recoverability_curve.csv'
    task_age_path = curves_dir / 'task_age_rho.csv'
    calibration_budget_path = curves_dir / 'calibration_vs_budget.csv'
    latency_budget_path = curves_dir / 'latency_vs_budget.csv'
    repair_outcomes_path = experiment_dir / 'tables' / 'repair_outcomes.jsonl'
    repair_frontier_path = frontier_dir / 'repair_frontier.csv'
    repair_pareto_path = frontier_dir / 'repair_pareto.csv'
    repair_impact_path = frontier_dir / 'repair_impact.csv'
    repair_selection_path = frontier_dir / 'repair_selection.csv'
    manifest_path = frontier_dir / 'manifest.json'
    predictive_corr_path = experiment_dir / 'predictive' / 'predictive_correlations.csv'

    missing_sections: list[str] = []
    resolved_tracking_uri = resolve_tracking_uri(tracking_uri=tracking_uri)
    resolved_artifact_location = resolve_artifact_location(artifact_location=artifact_location)

    def _read_section(path: Path, name: str) -> list[dict[str, Any]]:
        if path.exists():
            return _coerce_csv_rows(_read_csv_rows(path))
        missing_sections.append(name)
        return []

    def _read_json_section(path: Path, name: str) -> dict[str, Any] | list[Any]:
        payload = _read_json_payload(path)
        if payload is not None:
            return payload
        missing_sections.append(name)
        return {}

    def _read_jsonl_section(path: Path, name: str) -> list[dict[str, Any]]:
        if path.exists():
            return _read_jsonl_rows(path)
        missing_sections.append(name)
        return []

    export_payload: dict[str, Any] = {
        'schema': {
            'name': 'regain.analysis.export',
            'version': 2,
        },
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'mlflow': {
            'experiment': str(experiment),
            'tracking_uri': resolved_tracking_uri,
            'artifact_location': resolved_artifact_location,
            'include_controllers': include_controllers,
            'exclude_controllers': exclude_controllers,
            'max_runs': max_runs,
            'default_num_classes': default_num_classes,
            'require_finished': require_finished,
        },
        'tables': {
            'run_metrics': runs_table,
            'experience_metrics': experiences_table,
            'repair_outcomes': _read_jsonl_section(
                repair_outcomes_path,
                'tables.repair_outcomes',
            ),
        },
        'curves': {
            'recoverability_curve': _read_section(
                recoverability_path,
                'curves.recoverability_curve',
            ),
            'task_age_rho': _read_section(task_age_path, 'curves.task_age_rho'),
            'calibration_vs_budget': _read_section(
                calibration_budget_path,
                'curves.calibration_vs_budget',
            ),
            'latency_vs_budget': _read_section(
                latency_budget_path,
                'curves.latency_vs_budget',
            ),
        },
        'frontier': {
            'repair_frontier': _read_section(
                repair_frontier_path,
                'frontier.repair_frontier',
            ),
            'repair_pareto': _read_section(
                repair_pareto_path,
                'frontier.repair_pareto',
            ),
            'repair_impact': _read_section(
                repair_impact_path,
                'frontier.repair_impact',
            ),
            'repair_selection': _read_section(
                repair_selection_path,
                'frontier.repair_selection',
            ),
            'manifest': _read_json_section(
                manifest_path,
                'frontier.manifest',
            ),
        },
        'predictive': {
            'correlations': _read_section(
                predictive_corr_path,
                'predictive.correlations',
            ),
        },
        'notes': {
            'plots_embedded': False,
        },
    }

    if missing_sections:
        export_payload['missing_sections'] = missing_sections

    export_path.parent.mkdir(parents=True, exist_ok=True)
    with export_path.open('w', encoding='utf-8') as f:
        json.dump(export_payload, f, indent=2)
