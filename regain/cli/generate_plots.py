"""
CLI tool to render plots from analysis artifacts.

This script reads the CSV artifacts produced by `python -m regain.cli.run_analysis ...`
and renders (and/or saves) `matplotlib` plots.

Examples:
  python -m regain.cli.generate_plots --analysis-dir ./analysis_results/experiment_1 --show
  python -m regain.cli.generate_plots --analysis-dir ./analysis_results/experiment_1 --save
  python -m regain.cli.generate_plots --analysis-dir ./analysis_results/experiment_1 --show --save --save-dir ./plots
"""
import argparse
from pathlib import Path
import sys
import tempfile

from regain.analysis.plotting import plot_analysis_outputs
from regain.cli._utils._output_helpers import add_failure
from regain.cli._utils._output_helpers import CliFailure
from regain.cli._utils._output_helpers import finalize_staged_outputs
from regain.cli._utils._output_helpers import print_failure_summary
from regain.cli._utils._output_helpers import resolve_exit_code
from regain.cli._utils._output_helpers import StagedOutput
from regain.constants import ANALYSIS_RHO_AVG
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

    p = argparse.ArgumentParser(prog='regain-generate-plots')
    p.add_argument('--analysis-dir', type=str, required=True, help='Path to the analysis directory for a single experiment.')
    p.add_argument('--show', action='store_true', help='Show plots interactively.')
    p.add_argument('--save', action='store_true', help='Save plots as PNGs.')
    p.add_argument(
        '--perf-key',
        type=str,
        default=ANALYSIS_RHO_AVG,
        help='Which performance key to plot for the recoverability curve.',
    )
    p.add_argument(
        '--save-dir',
        type=str,
        default=None,
        help='Optional directory to write plot PNGs (defaults to <analysis-dir>/plots when saving).',
    )
    p.add_argument(
        '--allow-partial',
        action='store_true',
        help='Allow partial outputs when plotting fails.',
    )
    p.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing target outputs.',
    )
    args = p.parse_args()

    analysis_out = Path(args.analysis_dir)
    mode = _mode_from_flags(show=bool(args.show), save=bool(args.save))
    failures: list[CliFailure] = []
    staged_outputs: list[StagedOutput] = []
    destination_dir = Path(args.save_dir) if args.save_dir is not None else (analysis_out / 'plots')

    with tempfile.TemporaryDirectory() as temp_dir:
        staged_save_dir = Path(temp_dir) / 'plots'
        if not analysis_out.exists():
            add_failure(
                failures=failures,
                scope=f'analysis_dir={analysis_out}',
                error=f'analysis_dir does not exist: {analysis_out}',
            )
        else:
            try:
                save_dir = staged_save_dir if mode in ['save', 'both'] else None
                saved = plot_analysis_outputs(
                    analysis_out=analysis_out,
                    perf_key=str(args.perf_key),
                    mode=mode,
                    save_dir=save_dir,
                )
                if saved and mode in ['save', 'both']:
                    staged_outputs.append(
                        StagedOutput(
                            scope=f'analysis_dir={analysis_out}',
                            source=staged_save_dir,
                            destination=destination_dir,
                        )
                    )
            except Exception as exc:
                add_failure(
                    failures=failures,
                    scope=f'analysis_dir={analysis_out}',
                    error=exc,
                )

        published_count = finalize_staged_outputs(
            outputs=staged_outputs,
            failures=failures,
            allow_partial=bool(args.allow_partial),
            overwrite=bool(args.overwrite),
        )

    if published_count > 0:
        logger.warning(f'Plots written under: {destination_dir}')

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
