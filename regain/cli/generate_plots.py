"""
CLI tool to render plots from analysis artifacts.

This script reads the CSV artifacts produced by `python -m regain.cli.run_analysis ...`
and renders (and/or saves) `matplotlib` plots.

Examples:
  python -m regain.cli.generate_plots --analysis-out ./analysis_out --show
  python -m regain.cli.generate_plots --analysis-out ./analysis_out --save
  python -m regain.cli.generate_plots --analysis-out ./analysis_out --show --save --save-dir ./plots
"""
import argparse
from pathlib import Path

from regain.analysis.plotting import plot_analysis_outputs
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
        None: Does not return a value.
    """
    logger = get_logger()

    p = argparse.ArgumentParser(prog='regain-generate-plots')
    p.add_argument('--analysis-out', type=str, required=True, help='Root analysis output directory.')
    p.add_argument('--show', action='store_true', help='Show plots interactively.')
    p.add_argument('--save', action='store_true', help='Save plots as PNGs.')
    p.add_argument(
        '--perf-key',
        type=str,
        default='rho_mean_avg',
        help='Which performance key to plot for the recoverability curve.',
    )
    p.add_argument(
        '--save-dir',
        type=str,
        default=None,
        help='Optional directory to write plot PNGs (defaults to <analysis_out>/plots when saving).',
    )
    args = p.parse_args()

    analysis_out = Path(args.analysis_out)
    if not analysis_out.exists():
        raise FileNotFoundError(f'analysis_out does not exist: {analysis_out}')

    mode = _mode_from_flags(show=bool(args.show), save=bool(args.save))

    saved = plot_analysis_outputs(
        analysis_out=analysis_out,
        perf_key=str(args.perf_key),
        mode=mode,
        save_dir=args.save_dir,
    )

    if saved:
        out_dir = Path(args.save_dir) if args.save_dir is not None else (analysis_out / 'plots')
        logger.warning(f'Plots written under: {out_dir}')


if __name__ == '__main__':
    main()
