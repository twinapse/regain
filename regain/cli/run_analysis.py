"""
CLI entrypoint for running analysis.

It consumes metrics logged by `regain/cli/run_experiment.py`.

Examples:
  python -m regain.cli.run_analysis --experiments experiment_1 --output-dir ./analysis_results all
  python -m regain.cli.run_analysis --experiments experiment_1 --output-dir ./analysis_results curves
  python -m regain.cli.run_analysis --experiments experiment_1 --output-dir ./analysis_results --perf-key analysis.repair.rho.avg frontier
"""

import argparse
import csv
from pathlib import Path
import sys
import tempfile
from typing import Any

from regain.analysis.collectors import collect_experiment_tables
from regain.analysis.curves import write_recoverability_curves
from regain.analysis.frontier import write_efficiency_frontiers
from regain.analysis.plotting import plot_analysis_outputs
from regain.analysis.predictive import write_predictive_correlations
from regain.cli._utils._output_helpers import add_failure
from regain.cli._utils._output_helpers import CliFailure
from regain.cli._utils._output_helpers import finalize_staged_outputs
from regain.cli._utils._output_helpers import print_failure_summary
from regain.cli._utils._output_helpers import resolve_exit_code
from regain.cli._utils._output_helpers import StagedOutput
from regain.cli._utils._selector_helpers import add_experiment_selector_arguments
from regain.cli._utils._selector_helpers import resolve_experiment_targets
from regain.constants import ANALYSIS_RHO_AVG
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


def _register_run_collection_failures(
    *,
    experiment_name: str,
    failures: list[CliFailure],
    run_failures: list[dict[str, str]],
) -> None:
    """
    Register run-level collection failures into the CLI failure list.

    Args:
        experiment_name (str): Experiment name for failure scoping.
        failures (list[CliFailure]): Mutable CLI failure list.
        run_failures (list[dict[str, str]]): Run-level failure payloads from collection.

    Returns:
        None
    """
    for run_failure in run_failures:
        run_id = str(run_failure.get('run_id') or '')
        run_name = str(run_failure.get('run_name') or '')
        scope = f'experiment={experiment_name} stage=collect run={run_id}'
        if run_name:
            scope = f'{scope} ({run_name})'
        add_failure(
            failures=failures,
            scope=scope,
            error=str(run_failure.get('error') or 'Unknown run collection failure.'),
        )


def main() -> None:
    """
    Run analysis subcommands for collected experiment metrics.
    """
    logger = get_logger()

    parser = argparse.ArgumentParser(prog='regain-analysis-tool')
    add_experiment_selector_arguments(parser=parser)
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Root output directory (experiment subdirectory is created under this path).',
    )
    parser.add_argument(
        '--tracking-uri',
        type=str,
        default=None,
        help='Optional MLflow tracking URI override.',
    )
    parser.add_argument('--include-controllers', type=str, default=None, help='Comma-separated allowlist for controller_name.')
    parser.add_argument('--exclude-controllers', type=str, default=None, help='Comma-separated denylist for controller_name.')
    parser.add_argument('--max-runs', type=int, default=None, help='Max number of runs.')
    parser.add_argument('--default-num-classes', type=int, default=None, help='Fallback num classes when not logged.')
    parser.add_argument('--show-plots', action='store_true', help='Show plots.')
    parser.add_argument('--save-plots', action='store_true', help='Save plots.')
    parser.add_argument('--perf-key', type=str, default=ANALYSIS_RHO_AVG, help='Performance key to maximize.')
    parser.add_argument(
        '--allow-partial',
        action='store_true',
        help='Allow partial outputs when some analysis stages fail.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing target outputs.',
    )

    sub = parser.add_subparsers(dest='cmd', required=True)
    sub.add_parser('collect', help='Collect MLflow runs into tidy tables.')
    sub.add_parser('curves', help='Compute recoverability curves.')
    sub.add_parser('frontier', help='Compute efficiency frontier.')
    sub.add_parser('predictive', help='Compute predictive correlations.')
    sub.add_parser('all', help='Run collect + curves + frontier + predictive.')

    args = parser.parse_args()

    include_controllers = _parse_list(args.include_controllers)
    exclude_controllers = _parse_list(args.exclude_controllers)
    failures: list[CliFailure] = []
    staged_outputs: list[StagedOutput] = []
    targets = resolve_experiment_targets(
        parser=parser,
        config_files=args.config_files,
        config_dir=args.config_dir,
        experiments=args.experiments,
        tracking_uri=args.tracking_uri,
        failures=failures,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        staged_root = Path(temp_dir)
        staged_analysis_root = staged_root / 'analysis_outputs'

        for target in targets:
            experiment_name = target.experiment_name
            tracking_uri = target.tracking_uri
            destination_analysis_dir = Path(args.output_dir) / experiment_name
            staged_experiment_dir = staged_analysis_root / experiment_name
            analysis_output_names: set[str] = set()
            runs_table: list[dict[str, Any]] = []
            experiences_table: list[dict[str, Any]] = []
            collect_completed = False
            plot_curve_rows: list[dict[str, Any]] | None = None
            plot_frontier_rows: list[dict[str, Any]] | None = None

            if args.cmd in ['collect', 'all', 'curves', 'frontier', 'predictive']:
                tables_dir = staged_experiment_dir / 'tables'
                try:
                    runs_table, experiences_table, run_failures = collect_experiment_tables(
                        experiment=experiment_name,
                        out_dir=tables_dir,
                        tracking_uri=tracking_uri,
                        include_controllers=include_controllers,
                        exclude_controllers=exclude_controllers,
                        max_runs=args.max_runs,
                        require_finished=True,
                        default_num_classes=args.default_num_classes,
                    )
                    _register_run_collection_failures(
                        experiment_name=experiment_name,
                        failures=failures,
                        run_failures=run_failures,
                    )
                    if not runs_table:
                        add_failure(
                            failures=failures,
                            scope=f'experiment={experiment_name} stage=collect',
                            error='No successful runs were collected. Refusing to publish empty analysis outputs.',
                        )
                    else:
                        collect_completed = True
                        analysis_output_names.add('tables')
                        logger.info(f'Collected tables under: {tables_dir}')
                except Exception as exc:
                    add_failure(
                        failures=failures,
                        scope=f'experiment={experiment_name} stage=collect',
                        error=exc,
                    )

            if args.cmd in ['curves', 'all']:
                if not collect_completed:
                    add_failure(
                        failures=failures,
                        scope=f'experiment={experiment_name} stage=curves',
                        error='Skipped because collect stage failed.',
                    )
                else:
                    curves_dir = staged_experiment_dir / 'curves'
                    try:
                        curve_path, task_path, calib_path, latency_path = write_recoverability_curves(
                            runs_table=runs_table,
                            experiences_table=experiences_table,
                            out_dir=curves_dir,
                        )
                        analysis_output_names.add('curves')
                        logger.info(f'Curves written: {curve_path}, {task_path}, {calib_path}, {latency_path}')
                    except Exception as exc:
                        add_failure(
                            failures=failures,
                            scope=f'experiment={experiment_name} stage=curves',
                            error=exc,
                        )

            if args.cmd in ['frontier', 'all']:
                if not collect_completed:
                    add_failure(
                        failures=failures,
                        scope=f'experiment={experiment_name} stage=frontier',
                        error='Skipped because collect stage failed.',
                    )
                else:
                    try:
                        staged_curves_dir = staged_experiment_dir / 'curves'
                        curve_csv = staged_curves_dir / 'recoverability_curve.csv'
                        if not curve_csv.exists():
                            existing_curve_csv = destination_analysis_dir / 'curves' / 'recoverability_curve.csv'
                            if existing_curve_csv.exists():
                                curve_csv = existing_curve_csv
                        curve_rows = _read_csv_rows(curve_csv)
                        frontier_dir = staged_experiment_dir / 'frontier'
                        points_path, pareto_path = write_efficiency_frontiers(
                            curve_rows=curve_rows,
                            out_dir=frontier_dir,
                            perf_key=str(getattr(args, 'perf_key', ANALYSIS_RHO_AVG)),
                        )
                        analysis_output_names.add('frontier')
                        plot_curve_rows = curve_rows
                        plot_frontier_rows = _read_csv_rows(points_path)
                        logger.info(f'Frontier written: {points_path}, {pareto_path}')
                    except Exception as exc:
                        add_failure(
                            failures=failures,
                            scope=f'experiment={experiment_name} stage=frontier',
                            error=exc,
                        )

            if args.cmd in ['predictive', 'all']:
                if not collect_completed:
                    add_failure(
                        failures=failures,
                        scope=f'experiment={experiment_name} stage=predictive',
                        error='Skipped because collect stage failed.',
                    )
                else:
                    predictive_dir = staged_experiment_dir / 'predictive'
                    try:
                        predictive_path = write_predictive_correlations(
                            experiences_table=experiences_table,
                            out_dir=predictive_dir,
                        )
                        analysis_output_names.add('predictive')
                        logger.info(f'Predictive correlations written: {predictive_path}')
                    except Exception as exc:
                        add_failure(
                            failures=failures,
                            scope=f'experiment={experiment_name} stage=predictive',
                            error=exc,
                        )

            mode = _plot_mode(show=bool(args.show_plots), save=bool(args.save_plots))
            if mode != 'none' and args.cmd in ['curves', 'frontier', 'all']:
                if not collect_completed:
                    add_failure(
                        failures=failures,
                        scope=f'experiment={experiment_name} stage=plots',
                        error='Skipped because collect stage failed.',
                    )
                else:
                    try:
                        plot_perf_key = str(getattr(args, 'perf_key', ANALYSIS_RHO_AVG))
                        saved = plot_analysis_outputs(
                            curve_rows=plot_curve_rows,
                            frontier_rows=plot_frontier_rows,
                            analysis_out=staged_experiment_dir,
                            perf_key=plot_perf_key,
                            mode=mode,
                        )
                        if saved:
                            analysis_output_names.add('plots')
                            logger.info(f'Plots written under: {staged_experiment_dir / "plots"}')
                    except Exception as exc:
                        add_failure(
                            failures=failures,
                            scope=f'experiment={experiment_name} stage=plots',
                            error=exc,
                        )

            for output_name in sorted(analysis_output_names):
                staged_item = staged_experiment_dir / output_name
                if not staged_item.exists():
                    continue
                staged_outputs.append(
                    StagedOutput(
                        scope=f'experiment={experiment_name} analysis-output={output_name}',
                        source=staged_item,
                        destination=destination_analysis_dir / output_name,
                    )
                )

        published_count = finalize_staged_outputs(
            outputs=staged_outputs,
            failures=failures,
            allow_partial=bool(args.allow_partial),
            overwrite=bool(args.overwrite),
        )

    if published_count > 0:
        print(f'Published {published_count} output item(s).')

    print_failure_summary(
        command_name='regain-analysis-tool',
        failures=failures,
    )
    sys.exit(
        resolve_exit_code(
            failures=failures,
            allow_partial=bool(args.allow_partial),
            published_count=published_count,
        )
    )


if __name__ == '__main__':
    main()
