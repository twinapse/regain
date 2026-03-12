"""
CLI entrypoint for exporting analysis outputs to a JSON bundle.
"""

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

from regain.analysis.exports import export_analysis_to_json
from regain.cli._utils._output_helpers import add_failure
from regain.cli._utils._output_helpers import CliFailure
from regain.cli._utils._output_helpers import finalize_staged_outputs
from regain.cli._utils._output_helpers import print_failure_summary
from regain.cli._utils._output_helpers import resolve_exit_code
from regain.cli._utils._output_helpers import StagedOutput
from regain.cli._utils._selector_helpers import add_experiment_selector_arguments
from regain.cli._utils._selector_helpers import resolve_experiment_targets

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
    stripped = str(value).strip()
    if not stripped:
        return None
    return [token.strip() for token in stripped.split(',') if token.strip()]


def _read_jsonl_table(*, path: Path) -> list[dict[str, Any]]:
    """
    Read a JSONL table into a list of dictionary rows.

    Args:
        path (Path): JSONL table path.

    Returns:
        list[dict[str, Any]]: Parsed rows.

    Raises:
        ValueError: If a JSON line is invalid or not an object.
    """
    rows: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for line_index, line in enumerate(f, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                row = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSON in {path} at line {line_index}: {exc.msg}') from exc
            if not isinstance(row, dict):
                raise ValueError(f'Invalid row in {path} at line {line_index}: expected a JSON object.')
            rows.append(dict(row))
    return rows


def _load_analysis_tables(*, experiment_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Load required analysis tables from an experiment output directory.

    Args:
        experiment_dir (Path): Experiment analysis output directory.

    Returns:
        tuple[list[dict[str, Any]], list[dict[str, Any]]]: Run-level and experience-level tables.

    Raises:
        FileNotFoundError: If one or more required table files are missing.
        ValueError: If table contents are invalid.
    """
    tables_dir = experiment_dir / 'tables'
    runs_table_path = tables_dir / 'runs_table.jsonl'
    experiences_table_path = tables_dir / 'experiences_table.jsonl'

    missing_paths = [path for path in [runs_table_path, experiences_table_path] if not path.exists()]
    if missing_paths:
        missing_str = ', '.join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            f'Missing required analysis tables: {missing_str}. '
            'Run `python -m regain.cli.run_analysis --experiments <experiment> --output-dir <analysis_dir> collect` first.'
        )

    runs_table = _read_jsonl_table(path=runs_table_path)
    experiences_table = _read_jsonl_table(path=experiences_table_path)
    return runs_table, experiences_table


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser for standalone execution.

    Returns:
        argparse.ArgumentParser: Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(prog='regain-export-analysis')
    add_experiment_selector_arguments(parser=parser)
    parser.add_argument(
        '--analysis-dir',
        type=str,
        required=True,
        help='Root directory containing per-experiment analysis outputs.',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Path to the directory for exported analysis bundles.',
    )
    parser.add_argument(
        '--tracking-uri',
        type=str,
        default=None,
        help='MLflow tracking URI (overrides config-derived values).',
    )
    parser.add_argument(
        '--artifact-uri',
        type=str,
        default=None,
        help='MLflow artifact URI or filesystem path.',
    )
    parser.add_argument(
        '--include-controllers',
        type=str,
        default=None,
        help='Comma-separated allowlist for controller_name.',
    )
    parser.add_argument(
        '--exclude-controllers',
        type=str,
        default=None,
        help='Comma-separated denylist for controller_name.',
    )
    parser.add_argument('--max-runs', type=int, default=None, help='Max number of runs.')
    parser.add_argument('--default-num-classes', type=int, default=None, help='Fallback num classes when not logged.')
    parser.add_argument(
        '--allow-partial',
        action='store_true',
        help='Allow partial outputs when some required resources fail.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing target outputs.',
    )
    return parser


def main() -> None:
    """
    CLI entrypoint to export analysis bundles from existing analysis outputs.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    include_controllers = _parse_list(args.include_controllers)
    exclude_controllers = _parse_list(args.exclude_controllers)
    analysis_root = Path(args.analysis_dir)
    destination_root = Path(args.output_dir)
    failures: list[CliFailure] = []
    staged_outputs: list[StagedOutput] = []
    targets = resolve_experiment_targets(
        parser=parser,
        config_files=args.config_files,
        config_dir=args.config_dir,
        experiments=args.experiments,
        tracking_uri_override=args.tracking_uri,
        failures=failures,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        staged_export_root = Path(temp_dir) / 'analysis_exports'
        for target in targets:
            experiment_name = target.experiment_name
            experiment_dir = analysis_root / experiment_name
            destination_path = destination_root / experiment_name / 'analysis.json'
            scope = f'experiment={experiment_name}'
            try:
                runs_table, experiences_table = _load_analysis_tables(experiment_dir=experiment_dir)
                staged_export_path = staged_export_root / experiment_name / 'analysis.json'
                export_analysis_to_json(
                    experiment=experiment_name,
                    experiment_dir=experiment_dir,
                    export_path=staged_export_path,
                    tracking_uri=target.tracking_uri,
                    artifact_uri=args.artifact_uri,
                    runs_table=runs_table,
                    experiences_table=experiences_table,
                    include_controllers=include_controllers,
                    exclude_controllers=exclude_controllers,
                    max_runs=args.max_runs,
                    default_num_classes=args.default_num_classes,
                    require_finished=True,
                )
                staged_outputs.append(
                    StagedOutput(
                        scope=scope,
                        source=staged_export_path,
                        destination=destination_path,
                    )
                )
            except Exception as exc:
                add_failure(
                    failures=failures,
                    scope=scope,
                    error=exc,
                )

        published_count = finalize_staged_outputs(
            outputs=staged_outputs,
            failures=failures,
            allow_partial=bool(args.allow_partial),
            overwrite=bool(args.overwrite),
        )

    if published_count > 0:
        print(f'Published analysis export(s) for {published_count} experiment(s).')

    print_failure_summary(
        command_name='regain-export-analysis',
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
