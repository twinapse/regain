"""
CLI entrypoint for running experiments from YAML configuration and logging metrics.
"""

import argparse
import inspect
from pathlib import Path
import sys

# WARNING: Don't import non-standard modules at the top-level to ensure prerequisites first.

__all__ = [
    'main',
]


def _patch_avalanche_bic_multistep_lr() -> None:
    """
    Avalanche `BiCPlugin` passes `verbose=...` to `MultiStepLR`, but newer PyTorch versions removed that argument.
    This function patches Avalanche to ignore `verbose`.
    """
    import torch.optim.lr_scheduler as lr_scheduler

    # If this PyTorch still supports `verbose`, nothing to do.
    if 'verbose' in inspect.signature(lr_scheduler.MultiStepLR).parameters:
        return

    # If `bic` is not available, nothing to do.
    try:
        import avalanche.training.plugins.bic as bic
    except Exception:
        return

    # Idempotency guard
    if getattr(bic, '_regain_multistep_lr_patched', False):
        return

    # Save original `MultiStepLR`
    OriginalMultiStepLR = bic.MultiStepLR

    # Define patched `MultiStepLR`
    def _MultiStepLR(*args, **kwargs):
        kwargs.pop('verbose', None)
        return OriginalMultiStepLR(*args, **kwargs)

    # Apply patch
    bic.MultiStepLR = _MultiStepLR
    bic._regain_multistep_lr_patched = True


def _patch_avalanche_il2m_initial_eval() -> None:
    """
    `IL2MPlugin` can crash during Avalanche's initial (pre-training) eval because `classes2exp` is still an empty list.
    This function skips rectification until initialized.
    """
    # If `il2m` is not available, nothing to do.
    try:
        import avalanche.training.plugins.il2m as il2m
    except Exception:
        return

    # Save original `IL2MPlugin.after_eval_forward`
    original_after_eval_forward = il2m.IL2MPlugin.after_eval_forward

    # Idempotency guard
    if getattr(original_after_eval_forward, '_regain_patched', False):
        return

    # Define patched `after_eval_forward`
    def _after_eval_forward(self, strategy, **kwargs):
        # Not initialized yet (typical during do_initial pre-eval)
        if not getattr(self, 'classes2exp', None):
            return
        return original_after_eval_forward(self, strategy, **kwargs)

    # Apply patch
    _after_eval_forward._regain_patched = True
    il2m.IL2MPlugin.after_eval_forward = _after_eval_forward


def _ensure_prerequisites() -> None:
    """
    Ensure all prerequisites for running experiments are met.
    """
    _patch_avalanche_bic_multistep_lr()
    _patch_avalanche_il2m_initial_eval()


def _export_runs_to_csvs(
    *,
    experiment_name: str,
    export_dir: str,
    tracking_uri: str | None,
) -> None:
    """
    Export run data for one experiment to CSV files.

    Args:
        experiment_name (str): MLflow experiment name.
        export_dir (str): Directory for exportable run outputs.
        tracking_uri (str | None): MLflow tracking URI.

    Returns:
        None

    """
    # Import `regain` modules after prerequisites are ensured
    from regain.cli._utils._export_helpers import export_runs_for_experiment

    export_runs_for_experiment(
        experiment_name=experiment_name,
        export_dir=export_dir,
        tracking_uri=tracking_uri,
    )


def _run_experiment(config_file: str) -> tuple[str, str | None]:
    """
    Load and run a single experiment configuration.

    Args:
        config_file (str): Path to a single experiment config YAML file.

    Returns:
        tuple[str, str | None]: Experiment name and MLflow tracking URI.
    """
    # Import `regain` modules after prerequisites are ensured
    from regain.experiments.config import load_experiment_config
    from regain.experiments.orchestrator import run_experiment

    # Load experiment config
    experiment_config = load_experiment_config(config_file)
    # Run experiment
    run_experiment(experiment_config)
    # Return experiment name and tracking URI
    return experiment_config.experiment_name, experiment_config.mlflow_tracking_uri


def _resolve_export_targets(
    *,
    experiment_runs: list[tuple[str, str | None]],
) -> list[tuple[str, str | None]]:
    """
    Resolve export targets grouped by experiment.

    Args:
        experiment_runs (list[tuple[str, str | None]]): Completed run results.

    Returns:
        list[tuple[str, str | None]]: One export target per experiment.

    Raises:
        ValueError: If one experiment is associated with multiple tracking URIs.
    """
    # Import `regain` modules after prerequisites are ensured
    from regain.mlflow_utils import normalize_tracking_uri

    experiment_tracking_uris: dict[str, str | None] = {}
    for experiment_name, tracking_uri_raw in experiment_runs:
        tracking_uri = normalize_tracking_uri(tracking_uri=tracking_uri_raw)
        if experiment_name not in experiment_tracking_uris:
            experiment_tracking_uris[experiment_name] = tracking_uri
            continue

        existing_tracking_uri = experiment_tracking_uris[experiment_name]
        if existing_tracking_uri == tracking_uri:
            continue

        raise ValueError(
            'Conflicting tracking URIs for the same experiment. '
            f'Experiment `{experiment_name}` uses both `{existing_tracking_uri}` and `{tracking_uri}`.'
        )

    return [(name, uri) for name, uri in experiment_tracking_uris.items()]


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser for standalone execution.

    Returns:
        argparse.ArgumentParser: Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(description='Run a REGAIN experiment')
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
        default=None,
        help='Path to the directory for exportable run outputs.',
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


def main() -> None:
    """
    CLI entrypoint to run experiments from YAML config files.
    """
    # Ensure prerequisites
    _ensure_prerequisites()

    # Parse CLI arguments
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Get config files
    config_files: list[str] = []
    if args.config_files is not None:
        config_files = [config_file.strip() for config_file in args.config_files.split(',') if config_file.strip()]
        if not config_files:
            parser.error('At least one config file must be provided via --config-files.')
    elif args.config_dir is not None:
        try:
            config_files = _find_config_files(config_dir=args.config_dir)
        except ValueError as exc:
            parser.error(str(exc))
        if not config_files:
            parser.error(f'No config YAML files found in --config-dir: {args.config_dir}')

    # Run each experiment config file
    experiment_runs: list[tuple[str, str | None]] = []
    for config_file in config_files:
        experiment_runs.append(_run_experiment(config_file))

    # Export runs to CSV once per experiment if requested.
    if args.export_dir is not None:
        try:
            export_targets = _resolve_export_targets(
                experiment_runs=experiment_runs,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(1) from exc
        for experiment_name, tracking_uri in export_targets:
            _export_runs_to_csvs(
                experiment_name=experiment_name,
                export_dir=args.export_dir,
                tracking_uri=tracking_uri,
            )


if __name__ == '__main__':
    main()
