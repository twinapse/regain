"""
CLI entrypoint for exporting MLflow run data to CSV files.
"""

import argparse
from pathlib import Path
import sys
import tempfile

from regain.cli._utils.output_helpers import add_failure
from regain.cli._utils.output_helpers import CliFailure
from regain.cli._utils.output_helpers import finalize_staged_outputs
from regain.cli._utils.output_helpers import print_failure_summary
from regain.cli._utils.output_helpers import resolve_exit_code
from regain.cli._utils.output_helpers import StagedOutput
from regain.cli._utils.selector_helpers import add_experiment_selector_arguments
from regain.cli._utils.selector_helpers import resolve_experiment_targets
from regain.experiments.exports import export_runs_to_csvs

__all__ = [
    'main',
]


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser for standalone execution.

    Returns:
        argparse.ArgumentParser: Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(description='Export REGAIN experiment runs to CSVs')
    add_experiment_selector_arguments(parser=parser)
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Path to the directory for exported run outputs.',
    )
    parser.add_argument(
        '--tracking-uri',
        type=str,
        default=None,
        help='Optional MLflow tracking URI override.',
    )
    parser.add_argument(
        '--allow-partial',
        action='store_true',
        help='Allow partial outputs when some experiment exports fail.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing target outputs.',
    )
    return parser


def main() -> None:
    """
    CLI entrypoint to export runs for one or more experiments.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()
    destination_root = Path(args.output_dir)
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
        staged_root = Path(temp_dir) / 'run_exports'
        for target in targets:
            experiment_name = target.experiment_name
            tracking_uri = target.tracking_uri
            experiment_scope = f'experiment={experiment_name}'
            staged_experiment_root = staged_root / experiment_name
            metadata_path = staged_experiment_root / 'run_metadata.csv'
            params_path = staged_experiment_root / 'run_params.csv'
            metrics_path = staged_experiment_root / 'run_metrics.csv'
            try:
                export_runs_to_csvs(
                    experiment=experiment_name,
                    metadata_path=metadata_path,
                    params_path=params_path,
                    metrics_path=metrics_path,
                    tracking_uri=tracking_uri,
                )
            except Exception as exc:
                add_failure(
                    failures=failures,
                    scope=experiment_scope,
                    error=exc,
                )
                continue

            staged_outputs.append(
                StagedOutput(
                    scope=experiment_scope,
                    source=staged_experiment_root,
                    destination=destination_root / experiment_name,
                )
            )

        published_count = finalize_staged_outputs(
            outputs=staged_outputs,
            failures=failures,
            allow_partial=bool(args.allow_partial),
            overwrite=bool(args.overwrite),
        )

    if published_count > 0:
        print(f'Published run exports for {published_count} experiment(s).')

    print_failure_summary(
        command_name='regain-export-runs',
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
