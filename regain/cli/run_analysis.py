"""
CLI entrypoint for running analysis.

It consumes metrics logged by `regain/cli/run_experiment.py`.

Examples:
  python -m regain.cli.run_analysis all --experiment experiment_1 --output-dir ./analysis_results
  python -m regain.cli.run_analysis curves --experiment experiment_1 --output-dir ./analysis_results
  python -m regain.cli.run_analysis frontier --experiment experiment_1 --output-dir ./analysis_results --perf-key rho_mean_avg
"""

import argparse
import csv
from pathlib import Path
from typing import Any

from regain.analysis.collectors import collect_experiment_tables
from regain.analysis.curves import write_recoverability_curves
from regain.analysis.exports import export_analysis_to_json
from regain.analysis.frontier import write_efficiency_frontiers
from regain.analysis.plotting import plot_analysis_outputs
from regain.utils import get_logger

__all__ = [
    'main',
]


def _parse_list(value: str | None) -> list[str] | None:
    """
    Parse a comma-separated list.

    Args:
        value (str | None): Input string.

    Returns:
        list[str] | None: Parsed list or None.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    return [t.strip() for t in s.split(',') if t.strip()]


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


def _plot_mode(*, show: bool, save: bool) -> str:
    """
    Determine plotting mode from flags.

    Args:
        show (bool): Whether to show plots.
        save (bool): Whether to save plots.

    Returns:
        str: One of 'none', 'show', 'save', 'both'.
    """
    if show and save:
        return 'both'
    if save:
        return 'save'
    if show:
        return 'show'
    return 'none'


def main() -> None:
    """
    Run analysis subcommands for collected experiment metrics.

    Returns:
        None
    """
    logger = get_logger()

    p = argparse.ArgumentParser(prog='regain-analysis-tool')
    p.add_argument('--experiment', type=str, required=True, help='MLflow experiment name or id.')
    p.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Root output directory (experiment subdirectory will be created under this path).',
    )
    p.add_argument('--export-dir', type=str, default=None, help='Path to the directory for exportable analysis outputs.')
    p.add_argument(
        '--tracking-uri',
        type=str,
        default=None,
        help='MLflow tracking URI or filesystem path (SQLite only).',
    )
    p.add_argument(
        '--artifact-uri',
        type=str,
        default=None,
        help='MLflow artifact URI or filesystem path.',
    )
    p.add_argument('--include-controllers', type=str, default=None, help='Comma-separated allowlist for controller_name.')
    p.add_argument('--exclude-controllers', type=str, default=None, help='Comma-separated denylist for controller_name.')
    p.add_argument('--max-runs', type=int, default=None, help='Max number of parent runs.')
    p.add_argument('--default-num-classes', type=int, default=None, help='Fallback num classes when not logged.')
    p.add_argument('--show-plots', action='store_true', help='Show plots.')
    p.add_argument('--save-plots', action='store_true', help='Save plots.')
    p.add_argument('--perf-key', type=str, default='rho_mean_avg', help='Performance key to maximize.')

    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('collect', help='Collect MLflow runs into tidy tables.')
    sub.add_parser('curves', help='Compute recoverability curves.')
    sub.add_parser('frontier', help='Compute efficiency frontier.')
    sub.add_parser('all', help='Run collect + curves + frontier.')

    args = p.parse_args()

    output_root = Path(args.output_dir)
    experiment_dir = output_root / str(args.experiment)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    include_controllers = _parse_list(args.include_controllers)
    exclude_controllers = _parse_list(args.exclude_controllers)

    export_path: Path | None = None
    if args.export_dir is not None:
        export_path = Path(args.export_dir) / str(args.experiment) / 'analysis.json'

    runs_table: list[dict[str, Any]] = []
    experiences_table: list[dict[str, Any]] = []

    # Collect (always required for downstream steps).
    if args.cmd in ['collect', 'all', 'curves', 'frontier']:
        tables_dir = experiment_dir / 'tables'
        runs_table, experiences_table = collect_experiment_tables(
            experiment=str(args.experiment),
            out_dir=tables_dir,
            tracking_uri=args.tracking_uri,
            include_controllers=include_controllers,
            exclude_controllers=exclude_controllers,
            max_runs=args.max_runs,
            require_finished=True,
            default_num_classes=args.default_num_classes,
        )
        logger.info(f'Collected tables under: {experiment_dir / "tables"}')

    # Recoverability curves.
    if args.cmd in ['curves', 'all']:
        curves_dir = experiment_dir / 'curves'
        curve_path, task_path = write_recoverability_curves(
            runs_table=runs_table,
            experiences_table=experiences_table,
            out_dir=curves_dir,
        )
        logger.info(f'Curves written: {curve_path}, {task_path}')

    # Efficiency frontier.
    if args.cmd in ['frontier', 'all']:
        # Frontier is computed from aggregated curve points.
        curves_dir = experiment_dir / 'curves'
        curve_csv = curves_dir / 'recoverability_curve.csv'
        curve_rows = _read_csv_rows(curve_csv)

        frontier_dir = experiment_dir / 'frontier'
        points_path, pareto_path = write_efficiency_frontiers(
            curve_rows=curve_rows,
            out_dir=frontier_dir,
            perf_key=str(getattr(args, 'perf_key', 'rho_mean_avg')),
        )
        logger.info(f'Frontier written: {points_path}, {pareto_path}')

    # Optional visualization / plot export.
    mode = _plot_mode(show=bool(args.show_plots), save=bool(args.save_plots))
    if mode != 'none' and args.cmd in ['curves', 'frontier', 'all']:
        plot_perf_key = str(getattr(args, 'perf_key', 'rho_mean_avg'))
        saved = plot_analysis_outputs(
            analysis_out=experiment_dir,
            perf_key=plot_perf_key,
            mode=mode,
        )
        if saved:
            logger.info(f'Plots written under: {experiment_dir / "plots"}')

    if args.export_dir is not None:
        if export_path is None:
            export_path = Path(args.export_dir) / str(args.experiment) / 'analysis.json'
        export_analysis_to_json(
            experiment=str(args.experiment),
            experiment_dir=experiment_dir,
            export_path=export_path,
            tracking_uri=args.tracking_uri,
            artifact_uri=args.artifact_uri,
            runs_table=runs_table,
            experiences_table=experiences_table,
            include_controllers=include_controllers,
            exclude_controllers=exclude_controllers,
            max_runs=args.max_runs,
            default_num_classes=args.default_num_classes,
            require_finished=True,
        )
        print(f'Analysis export written to: {export_path}')


if __name__ == '__main__':
    main()
