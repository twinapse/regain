"""
CLI entrypoint for running experiments from YAML configuration and logging metrics.
"""

import argparse
import inspect

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
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(description='Run a REGAIN experiment')
    parser.add_argument('--config_file', required=True, help='Path to experiment config YAML')
    return parser


def main() -> None:
    """
    CLI entrypoint to run experiments from a YAML config file.
    """
    # Ensure prerequisites
    _ensure_prerequisites()

    # Import `regain` modules after prerequisites are ensured
    from regain.experiments.core import run_experiment
    from regain.experiments.utils import load_experiment_config

    # Parse CLI arguments
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Load the experiment config
    experiment_config = load_experiment_config(args.config_file)

    # Run the experiment
    run_experiment(experiment_config=experiment_config)


if __name__ == '__main__':
    main()
