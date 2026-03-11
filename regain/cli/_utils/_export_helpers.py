"""
Shared CLI helpers for exporting run and analysis outputs.
"""

from pathlib import Path
from typing import Any

from regain.analysis.exports import export_analysis_to_json
from regain.experiments.exports import export_runs_to_csvs

__all__ = [
    'export_analysis_bundle',
    'export_runs_for_experiment',
]


def export_runs_for_experiment(
    *,
    experiment_name: str,
    export_dir: str | Path,
    tracking_uri: str | None,
) -> tuple[Path, Path, Path]:
    """
    Export run data for one experiment to CSV files.

    Args:
        experiment_name (str): MLflow experiment name.
        export_dir (str | Path): Directory for exportable run outputs.
        tracking_uri (str | None): Optional MLflow tracking URI.

    Returns:
        tuple[Path, Path, Path]: Paths to metadata, params, and metrics CSV exports.

    """
    export_root = Path(export_dir) / experiment_name
    metadata_path = export_root / 'run_metadata.csv'
    params_path = export_root / 'run_params.csv'
    metrics_path = export_root / 'run_metrics.csv'

    export_runs_to_csvs(
        experiment=experiment_name,
        metadata_path=metadata_path,
        params_path=params_path,
        metrics_path=metrics_path,
        tracking_uri=tracking_uri,
    )

    return metadata_path, params_path, metrics_path


def export_analysis_bundle(
    *,
    experiment: str,
    experiment_dir: Path,
    export_dir: str | Path,
    tracking_uri: str | None,
    artifact_uri: str | None,
    runs_table: list[dict[str, Any]],
    experiences_table: list[dict[str, Any]],
    include_controllers: list[str] | None,
    exclude_controllers: list[str] | None,
    max_runs: int | None,
    default_num_classes: int | None,
    require_finished: bool,
) -> Path:
    """
    Export analysis outputs for one experiment to a self-contained JSON bundle.

    Args:
        experiment (str): MLflow experiment name or id.
        experiment_dir (Path): Analysis output directory for one experiment.
        export_dir (str | Path): Export directory root.
        tracking_uri (str | None): Optional MLflow tracking URI.
        artifact_uri (str | None): Optional MLflow artifact URI or filesystem path.
        runs_table (list[dict[str, Any]]): Run-level analysis rows.
        experiences_table (list[dict[str, Any]]): Experience-level analysis rows.
        include_controllers (list[str] | None): Optional controller allowlist.
        exclude_controllers (list[str] | None): Optional controller denylist.
        max_runs (int | None): Optional maximum number of runs included.
        default_num_classes (int | None): Optional fallback number of classes.
        require_finished (bool): Whether only finished runs are required.

    Returns:
        Path: Path to the written analysis export JSON.
    """
    export_path = Path(export_dir) / str(experiment) / 'analysis.json'
    export_analysis_to_json(
        experiment=str(experiment),
        experiment_dir=experiment_dir,
        export_path=export_path,
        tracking_uri=tracking_uri,
        artifact_uri=artifact_uri,
        runs_table=runs_table,
        experiences_table=experiences_table,
        include_controllers=include_controllers,
        exclude_controllers=exclude_controllers,
        max_runs=max_runs,
        default_num_classes=default_num_classes,
        require_finished=require_finished,
    )
    return export_path
