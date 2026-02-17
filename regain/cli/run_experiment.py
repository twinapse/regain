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

    Raises:
        SystemExit: If one or more target export files already exist.
    """
    # Import `regain` modules after prerequisites are ensured
    from regain.experiments.exports import export_runs_to_csvs

    # Build export paths
    export_root = Path(export_dir) / experiment_name
    metadata_path = export_root / 'run_metadata.csv'
    params_path = export_root / 'run_params.csv'
    metrics_path = export_root / 'run_metrics.csv'

    # Export runs to CSVs, handling existing files gracefully
    try:
        export_runs_to_csvs(
            experiment=experiment_name,
            metadata_path=metadata_path,
            params_path=params_path,
            metrics_path=metrics_path,
            tracking_uri=tracking_uri,
        )
        print(
            'Run exports written to: '
            f'{metadata_path}, {params_path}, {metrics_path}'
        )
    except FileExistsError as exc:
        message = f'{exc}. Remove it or choose a different --export-dir.'
        print(message, file=sys.stderr)
        sys.exit(1)


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


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser for standalone execution.

    Returns:
        argparse.ArgumentParser: Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(description='Run a REGAIN experiment')
    parser.add_argument(
        '--config-files',
        required=True,
        help='Comma-separated list of paths to experiment config YAML files',
    )
    parser.add_argument(
        '--export-dir',
        type=str,
        default=None,
        help='Path to the directory for exportable run outputs.',
    )
    return parser


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
    config_files = [config_file.strip() for config_file in args.config_files.split(',') if config_file.strip()]
    if not config_files:
        parser.error('At least one config file must be provided via --config_files.')

    # Run each experiment config file and optionally export runs to CSVs
    for config_file in config_files:
        # Run experiment
        experiment_name, tracking_uri = _run_experiment(config_file)
        # Export runs to CSV if requested
        if args.export_dir is not None:
            _export_runs_to_csvs(
                experiment_name=experiment_name,
                export_dir=args.export_dir,
                tracking_uri=tracking_uri,
            )


if __name__ == '__main__':
    main()
