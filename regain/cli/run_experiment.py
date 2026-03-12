"""
CLI entrypoint for running experiments from YAML configuration and logging metrics.
"""

import argparse
import inspect
from pathlib import Path
import traceback

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


def _is_policy_enabled(
    *,
    resume: bool,
    retry: bool,
    overwrite: bool,
) -> bool:
    """
    Check whether any run-selection policy flag is enabled.

    Args:
        resume (bool): Whether resume-only mode is enabled.
        retry (bool): Whether retry-failed mode is enabled.
        overwrite (bool): Whether overwrite mode is enabled.

    Returns:
        bool: True when at least one policy flag is enabled.
    """
    return bool(resume or retry or overwrite)


def _resolve_run_status(*, run: object) -> str:
    """
    Resolve the MLflow run status from a run-like object.

    Args:
        run (object): MLflow run-like object with `.info.status`.

    Returns:
        str: Run status, or empty string when unavailable.
    """
    info = getattr(run, 'info', None)
    status = getattr(info, 'status', '')
    return str(status) if status is not None else ''


def _select_run_names_for_policy(
    *,
    candidate_run_names: list[str],
    latest_active_runs_by_name: dict[str, object],
    resume: bool,
    retry: bool,
    overwrite: bool,
) -> set[str]:
    """
    Select run names to launch for the current policy mode.

    Args:
        candidate_run_names (list[str]): Candidate run names from config and optional reserved backbone run.
        latest_active_runs_by_name (dict[str, object]): Latest active run per run name.
        resume (bool): Resume-only mode.
        retry (bool): Retry-failed-only mode.
        overwrite (bool): Overwrite mode.

    Returns:
        set[str]: Selected run names to launch.
    """
    if overwrite:
        return set(candidate_run_names)

    if resume:
        return {
            run_name
            for run_name in candidate_run_names
            if run_name not in latest_active_runs_by_name
        }

    if retry:
        selected_run_names: set[str] = set()
        for run_name in candidate_run_names:
            latest_run = latest_active_runs_by_name.get(run_name)
            if latest_run is None:
                continue
            if _resolve_run_status(run=latest_run) == 'FAILED':
                selected_run_names.add(run_name)
        return selected_run_names

    return set(candidate_run_names)


def _run_experiment(
    config_file: str,
    *,
    tracking_uri: str | None = None,
    artifact_location: str | None = None,
    resume: bool = False,
    retry: bool = False,
    overwrite: bool = False,
) -> None:
    """
    Load and run a single experiment configuration.

    Args:
        config_file (str): Path to a single experiment config YAML file.
        tracking_uri (str | None): Optional MLflow tracking URI override.
        artifact_location (str | None): Optional MLflow artifact location override.
        resume (bool): If true, launch only runs that do not already exist.
        retry (bool): If true, launch only runs whose latest active run failed.
        overwrite (bool): If true, relaunch selected runs after deleting active runs.

    Returns:
        None
    """
    # Import `regain` modules after prerequisites are ensured
    from regain.experiments.config import load_experiment_config
    from regain.experiments.orchestrator import run_experiment

    # Load experiment config
    experiment_config = load_experiment_config(config_file)
    if _is_policy_enabled(
        resume=resume,
        retry=retry,
        overwrite=overwrite,
    ):
        # Import non-stdlib modules lazily to keep top-level imports minimal until prerequisites are ensured.
        from regain.constants import RUN_NAME_BACKBONE
        from regain.mlflow_utils import delete_mlflow_runs
        from regain.mlflow_utils import resolve_active_runs_by_name
        from regain.mlflow_utils import resolve_latest_active_runs_by_name

        run_configs = list(experiment_config.runs) if experiment_config.runs is not None else []
        manages_local_backbone = (
            experiment_config.backbone is not None
            and experiment_config.backbone.source_experiment is None
        )
        candidate_run_names = [run_config.name for run_config in run_configs]
        if manages_local_backbone:
            candidate_run_names.append(RUN_NAME_BACKBONE)

        active_runs_by_name = resolve_active_runs_by_name(
            experiment_name=experiment_config.experiment_name,
            tracking_uri=tracking_uri,
        )
        latest_active_runs_by_name = resolve_latest_active_runs_by_name(
            active_runs_by_name=active_runs_by_name
        )
        selected_run_names = _select_run_names_for_policy(
            candidate_run_names=candidate_run_names,
            latest_active_runs_by_name=latest_active_runs_by_name,
            resume=resume,
            retry=retry,
            overwrite=overwrite,
        )
        selected_run_configs = [
            run_config
            for run_config in run_configs
            if run_config.name in selected_run_names
        ]
        selected_backbone = (
            manages_local_backbone
            and RUN_NAME_BACKBONE in selected_run_names
        )

        runs_to_delete: list[object] = []
        if overwrite:
            for run_name in selected_run_names:
                runs_to_delete.extend(active_runs_by_name.get(run_name, []))
        elif retry:
            for run_config in selected_run_configs:
                for existing_run in active_runs_by_name.get(run_config.name, []):
                    if _resolve_run_status(run=existing_run) == 'FAILED':
                        runs_to_delete.append(existing_run)
            if selected_backbone:
                runs_to_delete.extend(
                    active_runs_by_name.get(RUN_NAME_BACKBONE, [])
                )
        delete_mlflow_runs(
            runs=runs_to_delete,
            tracking_uri=tracking_uri,
        )

        if manages_local_backbone and not selected_backbone:
            local_backbone_exists = bool(active_runs_by_name.get(RUN_NAME_BACKBONE, []))
            if local_backbone_exists:
                experiment_config.backbone = None

        if not selected_run_configs and not selected_backbone:
            return

        experiment_config.runs = selected_run_configs

    # Run experiment
    run_experiment(
        experiment_config,
        tracking_uri=tracking_uri,
        artifact_location=artifact_location,
    )


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
        '--tracking-uri',
        type=str,
        default=None,
        help='Optional MLflow tracking URI override.',
    )
    parser.add_argument(
        '--artifact-location',
        type=str,
        default=None,
        help='Optional MLflow artifact location or filesystem path override.',
    )
    launch_mode_group = parser.add_mutually_exclusive_group(required=False)
    launch_mode_group.add_argument(
        '--resume',
        action='store_true',
        help='Launch only runs that do not already exist.',
    )
    launch_mode_group.add_argument(
        '--retry',
        action='store_true',
        help='Launch only failed runs (failed runs are deleted before relaunch).',
    )
    launch_mode_group.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing runs before relaunching selected runs.',
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
        config_files = [
            config_file.strip()
            for config_file in args.config_files.split(',')
            if config_file.strip()
        ]
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
    failed_config_files: list[str] = []
    for config_file in config_files:
        try:
            _run_experiment(
                config_file,
                tracking_uri=args.tracking_uri,
                artifact_location=args.artifact_location,
                resume=bool(args.resume),
                retry=bool(args.retry),
                overwrite=bool(args.overwrite),
            )
        except Exception as exc:
            failed_config_files.append(config_file)
            print(f'Error while running config file `{config_file}`: {exc}')
            traceback.print_exc()
            continue
    if failed_config_files:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
