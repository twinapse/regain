"""
CLI entrypoint for exporting MLflow run data to CSV files.
"""

import argparse
from pathlib import Path
import sys
import tempfile

from regain.cli._utils._export_helpers import export_runs_for_experiment
from regain.cli._utils._output_helpers import add_failure
from regain.cli._utils._output_helpers import CliFailure
from regain.cli._utils._output_helpers import finalize_staged_outputs
from regain.cli._utils._output_helpers import print_failure_summary
from regain.cli._utils._output_helpers import resolve_exit_code
from regain.cli._utils._output_helpers import StagedOutput
from regain.experiments.config import load_experiment_config
from regain.mlflow_utils import normalize_tracking_uri

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
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        '--config-files',
        help='Comma-separated list of paths to experiment config YAML files',
    )
    config_group.add_argument(
        '--config-dir',
        type=str,
        help='Path to a directory recursively searched for experiment config YAML files',
    )
    parser.add_argument(
        '--export-dir',
        type=str,
        required=True,
        help='Path to the directory for exportable run outputs.',
    )
    parser.add_argument(
        '--allow-partial',
        action='store_true',
        help='Allow partial outputs when some config files fail.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing target outputs.',
    )
    return parser


def _find_config_files(*, config_dir: str) -> list[str]:
    """
    Recursively find experiment config YAML files in a directory.

    Args:
        config_dir (str): Root directory path.

    Returns:
        list[str]: Sorted config file paths.

    Raises:
        ValueError: If `config_dir` does not exist or is not a directory.
    """
    root_dir = Path(config_dir)
    if not root_dir.exists():
        raise ValueError(f'Config directory does not exist: {config_dir}')
    if not root_dir.is_dir():
        raise ValueError(f'Config directory is not a directory: {config_dir}')

    config_paths = sorted(
        [
            path
            for path in root_dir.rglob('*')
            if path.is_file() and path.suffix.lower() in ['.yaml', '.yml']
        ]
    )
    return [str(path) for path in config_paths]


def _resolve_config_files(
    *,
    parser: argparse.ArgumentParser,
    config_files: str | None,
    config_dir: str | None,
) -> list[str]:
    """
    Resolve config paths from CLI arguments.

    Args:
        parser (argparse.ArgumentParser): Parser used for user-facing argument errors.
        config_files (str | None): Comma-separated config files value.
        config_dir (str | None): Config directory value.

    Returns:
        list[str]: Resolved config file paths.
    """
    resolved_config_files: list[str] = []
    if config_files is not None:
        resolved_config_files = [config_file.strip() for config_file in config_files.split(',') if config_file.strip()]
        if not resolved_config_files:
            parser.error('At least one config file must be provided via --config-files.')
    elif config_dir is not None:
        try:
            resolved_config_files = _find_config_files(config_dir=config_dir)
        except ValueError as exc:
            parser.error(str(exc))
        if not resolved_config_files:
            parser.error(f'No config YAML files found in --config-dir: {config_dir}')
    return resolved_config_files


def _resolve_experiment_tracking_uris(
    *,
    config_files: list[str],
    failures: list[CliFailure],
) -> dict[str, str | None]:
    """
    Resolve one tracking URI per experiment from config files.

    Args:
        config_files (list[str]): Resolved config file paths.
        failures (list[CliFailure]): Mutable failure collector.

    Returns:
        dict[str, str | None]: Mapping experiment name -> normalized tracking URI.
    """
    experiment_tracking_uris: dict[str, str | None] = {}
    invalid_experiments: set[str] = set()

    for config_file in config_files:
        config_scope = f'config={config_file}'
        try:
            experiment_config = load_experiment_config(config_file)
        except Exception as exc:
            add_failure(
                failures=failures,
                scope=config_scope,
                error=exc,
            )
            continue

        experiment_name = str(experiment_config.experiment_name)
        tracking_uri = normalize_tracking_uri(tracking_uri=experiment_config.mlflow_tracking_uri)
        experiment_scope = f'experiment={experiment_name}'

        if experiment_name in invalid_experiments:
            continue

        if experiment_name not in experiment_tracking_uris:
            experiment_tracking_uris[experiment_name] = tracking_uri
            continue

        existing_tracking_uri = experiment_tracking_uris[experiment_name]
        if existing_tracking_uri == tracking_uri:
            continue

        add_failure(
            failures=failures,
            scope=experiment_scope,
            error=(
                'Conflicting tracking URIs for the same experiment. '
                f'Expected one URI per experiment but found `{existing_tracking_uri}` and `{tracking_uri}`.'
            ),
        )
        invalid_experiments.add(experiment_name)
        experiment_tracking_uris.pop(experiment_name, None)

    return experiment_tracking_uris


def main() -> None:
    """
    CLI entrypoint to export runs for one or more experiment config files.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()
    config_files = _resolve_config_files(
        parser=parser,
        config_files=args.config_files,
        config_dir=args.config_dir,
    )
    destination_root = Path(args.export_dir)
    failures: list[CliFailure] = []
    staged_outputs: list[StagedOutput] = []
    experiment_tracking_uris = _resolve_experiment_tracking_uris(
        config_files=config_files,
        failures=failures,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        staged_root = Path(temp_dir) / 'run_exports'
        for experiment_name, tracking_uri in sorted(experiment_tracking_uris.items()):
            experiment_scope = f'experiment={experiment_name}'
            try:
                export_runs_for_experiment(
                    experiment_name=experiment_name,
                    export_dir=staged_root,
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
                    source=staged_root / experiment_name,
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
