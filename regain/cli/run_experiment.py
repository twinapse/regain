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


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser for standalone execution.

    Returns:
        argparse.ArgumentParser: Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(description='Run a REGAIN experiment')
    parser.add_argument('--config_file', required=True, help='Path to experiment config YAML')
    parser.add_argument('--export-dir', type=str, default=None, help='Path to the directory for exportable run outputs.')
    return parser


def main() -> None:
    """
    CLI entrypoint to run experiments from a YAML config file.
    """
    # Ensure prerequisites
    _ensure_prerequisites()

    # Import `regain` modules after prerequisites are ensured
    from regain.experiments.core import run_experiment
    from regain.experiments.exports import export_runs_csv
    from regain.experiments.utils import load_experiment_config

    # Parse CLI arguments
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Load the experiment config
    experiment_config = load_experiment_config(args.config_file)

    # Prepare export paths if requested
    metadata_path: Path | None = None
    params_path: Path | None = None
    metrics_path: Path | None = None
    if args.export_dir is not None:
        export_root = Path(args.export_dir) / experiment_config.experiment_name
        metadata_path = export_root / 'run_metadata.csv'
        params_path = export_root / 'run_params.csv'
        metrics_path = export_root / 'run_metrics.csv'

    # Run the experiment
    run_experiment(experiment_config=experiment_config)

    # Export runs to CSV if requested
    if args.export_dir is not None:
        if metadata_path is None or params_path is None or metrics_path is None:
            export_root = Path(args.export_dir) / experiment_config.experiment_name
            metadata_path = export_root / 'run_metadata.csv'
            params_path = export_root / 'run_params.csv'
            metrics_path = export_root / 'run_metrics.csv'
        try:
            export_runs_csv(
                experiment=experiment_config.experiment_name,
                metadata_path=metadata_path,
                params_path=params_path,
                metrics_path=metrics_path,
                tracking_uri=experiment_config.mlflow_tracking_uri,
            )
            print(
                'Run exports written to: '
                f'{metadata_path}, {params_path}, {metrics_path}'
            )
        except FileExistsError as exc:
            message = f'{exc}. Remove it or choose a different --export-dir.'
            print(message, file=sys.stderr)
            sys.exit(1)


if __name__ == '__main__':
    main()
