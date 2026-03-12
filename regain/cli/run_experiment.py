"""
CLI entrypoint for running experiments from YAML configuration and logging metrics.
"""

import argparse
import inspect
from pathlib import Path

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


def _run_experiment(config_file: str) -> None:
    """
    Load and run a single experiment configuration.

    Args:
        config_file (str): Path to a single experiment config YAML file.

    Returns:
        None
    """
    # Import `regain` modules after prerequisites are ensured
    from regain.experiments.config import load_experiment_config
    from regain.experiments.orchestrator import run_experiment

    # Load experiment config
    experiment_config = load_experiment_config(config_file)
    # Run experiment
    run_experiment(experiment_config)


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
    for config_file in config_files:
        _run_experiment(config_file)


if __name__ == '__main__':
    main()
