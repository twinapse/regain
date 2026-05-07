"""
CLI tool to render plots from analysis artifacts.

This script reads analysis CSV artifacts produced by `python -m regain.cli.run_analysis ...`
and renders (and/or saves) `matplotlib` plots.

Examples:
  python -m regain.cli.generate_plots --analysis-dir ./analysis_results --experiments experiment_1 --show
  python -m regain.cli.generate_plots --analysis-dir ./analysis_results --experiments experiment_1 --save
  python -m regain.cli.generate_plots --analysis-dir ./analysis_results --experiments experiment_1 --show --save
  --output-dir ./plots
"""

import argparse
from pathlib import Path
import sys
import tempfile

from regain.analysis.plotting import plot_analysis_outputs
from regain.analysis.plotting import write_plot_manifest_update
from regain.cli._utils.output_helpers import add_failure
from regain.cli._utils.output_helpers import CliFailure
from regain.cli._utils.output_helpers import finalize_staged_outputs
from regain.cli._utils.output_helpers import print_failure_summary
from regain.cli._utils.output_helpers import resolve_exit_code
from regain.cli._utils.output_helpers import StagedOutput
from regain.cli._utils.selector_helpers import add_experiment_selector_arguments
from regain.cli._utils.selector_helpers import resolve_experiment_targets
from regain.utils import get_logger

__all__ = [
    'main',
]


def _mode_from_flags(*, show: bool, save: bool) -> str:
    """
    Infer plotting mode from CLI flags.

    Args:
        show (bool): Whether to show plots interactively.
        save (bool): Whether to save plots.

    Returns:
        str: One of 'show', 'save', or 'both'.
    """
    if show and save:
        return 'both'
    if save:
        return 'save'
    if show:
        return 'show'
    # Default behavior: interactive usage typically expects a window.
    return 'show'


def main() -> None:
    """
    Render plots from existing analysis outputs.

    Returns:
        None
    """
    logger = get_logger()

    parser = argparse.ArgumentParser(prog='regain-generate-plots')
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
        default=None,
        help='Optional root directory for saved plots (uses <output-dir>/<experiment>/plots).',
    )
    parser.add_argument('--show', action='store_true', help='Show plots interactively.')
    parser.add_argument('--save', action='store_true', help='Save plots as PNGs.')
    parser.add_argument(
        '--allow-partial',
        action='store_true',
        help='Allow partial outputs when plotting fails.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing target outputs.',
    )
    args = parser.parse_args()

    analysis_root = Path(args.analysis_dir)
    mode = _mode_from_flags(show=bool(args.show), save=bool(args.save))
    failures: list[CliFailure] = []
    staged_outputs: list[StagedOutput] = []
    targets = resolve_experiment_targets(
        parser=parser,
        config_files=args.config_files,
        config_dir=args.config_dir,
        experiments=args.experiments,
        tracking_uri=None,
        failures=failures,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        staged_root = Path(temp_dir) / 'plots'
        for target in targets:
            experiment_name = target.experiment_name
            analysis_out = analysis_root / experiment_name
            destination_dir = (
                Path(args.output_dir) / experiment_name / 'plots'
                if args.output_dir is not None
                else (analysis_out / 'plots')
            )
            scope = f'experiment={experiment_name}'

            if not analysis_out.exists():
                add_failure(
                    failures=failures,
                    scope=scope,
                    error=f'analysis_dir does not exist: {analysis_out}',
                )
                continue

            try:
                staged_save_dir = staged_root / experiment_name / 'plots'
                save_dir = staged_save_dir if mode in ['save', 'both'] else None
                plot_result = plot_analysis_outputs(
                    analysis_out=analysis_out,
                    mode=mode,
                    save_dir=save_dir,
                )
                saved_plot_files = bool(plot_result.saved_paths)
                has_plot_manifest_metadata = bool(plot_result.saved_filenames or plot_result.skipped)
                if saved_plot_files and mode in ['save', 'both']:
                    staged_outputs.append(
                        StagedOutput(
                            scope=scope,
                            source=staged_save_dir,
                            destination=destination_dir,
                        )
                    )
                if args.output_dir is None and mode in ['save', 'both'] and has_plot_manifest_metadata:
                    plot_destination_publishable = (
                        not saved_plot_files
                        or not destination_dir.exists()
                        or bool(args.overwrite)
                    )
                    if plot_destination_publishable:
                        staged_manifest_path = staged_root / experiment_name / 'frontier' / 'manifest.json'
                        write_plot_manifest_update(
                            source_manifest_path=analysis_out / 'frontier' / 'manifest.json',
                            destination_manifest_path=staged_manifest_path,
                            saved_filenames=plot_result.saved_filenames,
                            skipped=plot_result.skipped,
                        )
                        staged_outputs.append(
                            StagedOutput(
                                scope=f'{scope} analysis-output=frontier-manifest',
                                source=staged_manifest_path,
                                destination=analysis_out / 'frontier' / 'manifest.json',
                                overwrite_destination=True,
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
        logger.warning(f'Plots written for {published_count} experiment(s).')

    print_failure_summary(
        command_name='regain-generate-plots',
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
