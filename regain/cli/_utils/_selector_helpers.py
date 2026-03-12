"""
Shared CLI helpers for experiment selector arguments and target resolution.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

from regain.cli._utils._output_helpers import add_failure
from regain.cli._utils._output_helpers import CliFailure
from regain.experiments.config import load_experiment_config
from regain.mlflow_utils import normalize_tracking_uri

__all__ = [
    'ExperimentTarget',
    'add_experiment_selector_arguments',
    'find_config_files',
    'resolve_experiment_targets',
]


@dataclass(frozen=True)
class ExperimentTarget:
    """
    Resolved experiment target for CLI processing.

    Attributes:
        experiment_name (str): Experiment name or id.
        tracking_uri (str | None): Optional effective tracking URI for this experiment.
    """

    experiment_name: str
    tracking_uri: str | None


def add_experiment_selector_arguments(*, parser: argparse.ArgumentParser) -> None:
    """
    Add standard experiment-selector arguments to a parser.

    Args:
        parser (argparse.ArgumentParser): Target parser.

    Returns:
        None
    """
    selector_group = parser.add_mutually_exclusive_group(required=True)
    selector_group.add_argument(
        '--config-files',
        type=str,
        help='Comma-separated list of experiment config YAML files.',
    )
    selector_group.add_argument(
        '--config-dir',
        type=str,
        help='Directory recursively searched for experiment config YAML files.',
    )
    selector_group.add_argument(
        '--experiments',
        type=str,
        help='Comma-separated list of MLflow experiment names or ids.',
    )


def _parse_csv_argument(
    *,
    parser: argparse.ArgumentParser,
    argument_name: str,
    raw_value: str,
) -> list[str]:
    """
    Parse a non-empty comma-separated CLI argument value.

    Args:
        parser (argparse.ArgumentParser): Parser used for user-facing errors.
        argument_name (str): Argument name to include in error messages.
        raw_value (str): Raw argument value.

    Returns:
        list[str]: Parsed non-empty tokens.
    """
    tokens = [token.strip() for token in str(raw_value).split(',') if token.strip()]
    if not tokens:
        parser.error(f'At least one value must be provided via {argument_name}.')
    return tokens


def find_config_files(*, config_dir: str) -> list[str]:
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


def _resolve_config_file_paths(
    *,
    parser: argparse.ArgumentParser,
    config_files: str | None,
    config_dir: str | None,
) -> list[str]:
    """
    Resolve config paths from selector arguments.

    Args:
        parser (argparse.ArgumentParser): Parser used for user-facing errors.
        config_files (str | None): Raw comma-separated config files argument.
        config_dir (str | None): Config directory argument.

    Returns:
        list[str]: Resolved config file paths.
    """
    if config_files is not None:
        return _parse_csv_argument(
            parser=parser,
            argument_name='--config-files',
            raw_value=config_files,
        )

    try:
        resolved_config_files = find_config_files(config_dir=str(config_dir))
    except ValueError as exc:
        parser.error(str(exc))

    if not resolved_config_files:
        parser.error(f'No config YAML files found in --config-dir: {config_dir}')

    return resolved_config_files


def _resolve_experiment_targets_from_configs(
    *,
    config_paths: list[str],
    tracking_uri_override: str | None,
    failures: list[CliFailure],
) -> list[ExperimentTarget]:
    """
    Resolve deduplicated experiment targets from config files.

    Args:
        config_paths (list[str]): Resolved config paths.
        tracking_uri_override (str | None): Optional global tracking URI override.
        failures (list[CliFailure]): Mutable failure collector.

    Returns:
        list[ExperimentTarget]: Resolved targets in deterministic first-seen order.
    """
    targets: list[ExperimentTarget] = []
    target_indexes: dict[str, int] = {}
    invalid_experiments: set[str] = set()

    for config_path in config_paths:
        config_scope = f'config={config_path}'
        try:
            experiment_config = load_experiment_config(config_path)
        except Exception as exc:
            add_failure(
                failures=failures,
                scope=config_scope,
                error=exc,
            )
            continue

        experiment_name = str(experiment_config.experiment_name)
        if experiment_name in invalid_experiments:
            continue

        tracking_uri = tracking_uri_override
        if tracking_uri is None:
            tracking_uri = normalize_tracking_uri(
                tracking_uri=experiment_config.mlflow_tracking_uri,
            )
        current_index = target_indexes.get(experiment_name)
        if current_index is None:
            target_indexes[experiment_name] = len(targets)
            targets.append(
                ExperimentTarget(
                    experiment_name=experiment_name,
                    tracking_uri=tracking_uri,
                )
            )
            continue

        current_target = targets[current_index]
        if current_target.tracking_uri == tracking_uri:
            continue

        add_failure(
            failures=failures,
            scope=f'experiment={experiment_name}',
            error=(
                'Conflicting tracking URIs for the same experiment. '
                f'Expected one URI per experiment but found `{current_target.tracking_uri}` and `{tracking_uri}`.'
            ),
        )
        invalid_experiments.add(experiment_name)
        targets.pop(current_index)
        target_indexes = {target.experiment_name: index for index, target in enumerate(targets)}

    return targets


def _resolve_experiment_targets_from_names(
    *,
    experiment_names: list[str],
    tracking_uri: str | None,
) -> list[ExperimentTarget]:
    """
    Resolve deduplicated experiment targets from explicit names.

    Args:
        experiment_names (list[str]): Parsed experiment names.
        tracking_uri (str | None): Optional global tracking URI.

    Returns:
        list[ExperimentTarget]: Resolved targets in deterministic first-seen order.
    """
    seen_names: set[str] = set()
    targets: list[ExperimentTarget] = []
    for experiment_name in experiment_names:
        if experiment_name in seen_names:
            continue
        seen_names.add(experiment_name)
        targets.append(
            ExperimentTarget(
                experiment_name=experiment_name,
                tracking_uri=tracking_uri,
            )
        )
    return targets


def resolve_experiment_targets(
    *,
    parser: argparse.ArgumentParser,
    config_files: str | None,
    config_dir: str | None,
    experiments: str | None,
    tracking_uri_override: str | None,
    failures: list[CliFailure],
) -> list[ExperimentTarget]:
    """
    Resolve experiment targets from standard selector arguments.

    Args:
        parser (argparse.ArgumentParser): Parser used for user-facing errors.
        config_files (str | None): Raw comma-separated config paths argument.
        config_dir (str | None): Root config directory argument.
        experiments (str | None): Raw comma-separated experiment names argument.
        tracking_uri_override (str | None): Optional global tracking URI override.
        failures (list[CliFailure]): Mutable failure collector.

    Returns:
        list[ExperimentTarget]: Resolved targets.
    """
    normalized_tracking_uri_override = normalize_tracking_uri(
        tracking_uri=tracking_uri_override,
    )

    if experiments is not None:
        experiment_names = _parse_csv_argument(
            parser=parser,
            argument_name='--experiments',
            raw_value=experiments,
        )
        return _resolve_experiment_targets_from_names(
            experiment_names=experiment_names,
            tracking_uri=normalized_tracking_uri_override,
        )

    config_paths = _resolve_config_file_paths(
        parser=parser,
        config_files=config_files,
        config_dir=config_dir,
    )
    return _resolve_experiment_targets_from_configs(
        config_paths=config_paths,
        tracking_uri_override=normalized_tracking_uri_override,
        failures=failures,
    )
