"""
Plotting helpers for analysis outputs.

The analysis pipeline is intentionally table-first (CSV/JSON artifacts).
This module provides optional visualization of those artifacts using `matplotlib`.

Typical usage:
  - `python -m regain.cli.run_analysis all --experiment experiment_1 --output-dir ./analysis_results --show-plots`
  - `python -m regain.cli.generate_plots --analysis-dir ./analysis_results/experiment_1 --save`
"""
from pathlib import Path
from typing import Any, Iterable, Optional

from regain.analysis.utils import to_float
from regain.analysis.utils import to_int
from regain.constants import COLUMN_B
from regain.constants import COLUMN_CONTROLLER_NAME
from regain.constants import COLUMN_PERFORMANCE
from regain.constants import COLUMN_TOTAL_COST
from regain.constants import METRIC_RHO_MEAN_AVG


def _sorted_unique(values: Iterable[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[Any] = set()
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def plot_analysis_outputs(
    *,
    curve_rows: list[dict[str, Any]] | None = None,
    frontier_rows: list[dict[str, Any]] | None = None,
    analysis_out: str | Path | None = None,
    perf_key: str = METRIC_RHO_MEAN_AVG,
    mode: str = 'show',
    save_dir: str | Path | None = None,
) -> list[Path]:
    """
    Plot analysis artifacts.

    You can pass rows directly (e.g., loaded from CSV)
    or point the function at `analysis_out` to load the standard artifacts.

    Expected artifacts (relative to `analysis_out`):
      - curves/recoverability_curve.csv
      - frontier/frontier_points.csv

    Args:
        curve_rows (list[dict], optional): Rows from `recoverability_curve.csv`.
        frontier_rows (list[dict], optional): Rows from `frontier_points.csv`.
        analysis_out (str | Path, optional): Root analysis output directory.
        perf_key (str): Performance key to plot for the recoverability curve.
        mode (str): One of: `none`, `show`, `save`, `both`.
        save_dir (str | Path, optional): Directory to save PNG files.
                                         Defaults to `<analysis_dir>/plots` when an analysis directory is provided.

    Returns:
        list: List of saved plot paths (empty if not saving).
    """

    mode = str(mode).lower().strip()
    if mode not in {'none', 'show', 'save', 'both'}:
        raise ValueError(f"Invalid plotting mode: {mode}. Expected one of: none/show/save/both")

    import csv

    def read_csv_rows(p: Path) -> list[dict[str, Any]]:
        if not p.exists():
            return []
        with p.open('r', newline='', encoding='utf-8') as f:
            r = csv.DictReader(f)
            return [dict(row) for row in r]

    analysis_out_p: Optional[Path] = Path(analysis_out) if analysis_out is not None else None
    if curve_rows is None and analysis_out_p is not None:
        curve_rows = read_csv_rows(analysis_out_p / 'curves' / 'recoverability_curve.csv')
    if frontier_rows is None and analysis_out_p is not None:
        frontier_rows = read_csv_rows(analysis_out_p / 'frontier' / 'frontier_points.csv')

    curve_rows = curve_rows or []
    frontier_rows = frontier_rows or []

    # Nothing to do.
    if not curve_rows and not frontier_rows:
        return []

    save_paths: list[Path] = []

    if mode in {'save', 'both'}:
        if save_dir is not None:
            save_dir_p = Path(save_dir)
        elif analysis_out_p is not None:
            save_dir_p = analysis_out_p / 'plots'
        else:
            save_dir_p = Path('plots')
        save_dir_p.mkdir(parents=True, exist_ok=True)
    else:
        save_dir_p = None

    # Lazy import to keep matplotlib optional for users who only want tables.
    import matplotlib.pyplot as plt

    def std_key_for(perf: str) -> str:
        # Convention used by `regain.analysis.curves`:
        #   rho_mean_avg <-> rho_mean_std
        #   a_ctrl_mean_avg <-> a_ctrl_mean_std
        if perf.endswith('_avg'):
            return perf[:-4] + '_std'
        return perf + '_std'

    # -----------------
    # 1) Recoverability curves (budget -> performance)
    # -----------------
    if curve_rows:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        controllers = _sorted_unique([r.get(COLUMN_CONTROLLER_NAME) for r in curve_rows])
        for c in controllers:
            rows_c = [r for r in curve_rows if r.get(COLUMN_CONTROLLER_NAME) == c]
            rows_c = sorted(rows_c, key=lambda r: (to_float(r.get(COLUMN_B)) or float('inf')))

            xs = [to_float(r.get(COLUMN_B)) for r in rows_c]
            ys = [to_float(r.get(perf_key)) for r in rows_c]
            yerr = [to_float(r.get(std_key_for(perf_key))) for r in rows_c]

            pts = [(x, y, e) for x, y, e in zip(xs, ys, yerr) if x is not None and y is not None]
            if not pts:
                continue
            xs2, ys2, es2 = zip(*pts)

            has_err = any(e is not None for e in es2)
            if has_err:
                ax.errorbar(xs2, ys2, yerr=[e or 0.0 for e in es2], marker='o', linestyle='-', label=str(c))
            else:
                ax.plot(xs2, ys2, marker='o', linestyle='-', label=str(c))

        ax.set_xlabel('b (shots / class)')
        ax.set_ylabel(perf_key)
        ax.set_title('Recoverability curve')
        ax.legend()

        if save_dir_p is not None:
            out = save_dir_p / f'recoverability_curve__{perf_key}.png'
            fig.savefig(out, bbox_inches='tight')
            save_paths.append(out)

    # -----------------
    # 2) Efficiency frontier (cost -> performance)
    # -----------------
    if frontier_rows:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        # Decide x-axis: use total_cost when present for at least one point.
        any_total_cost = any(
            to_int(r.get(COLUMN_TOTAL_COST), coerce_float=True) is not None for r in frontier_rows
        )
        x_key = COLUMN_TOTAL_COST if any_total_cost else COLUMN_B

        # Group by controller for readability.
        controllers = _sorted_unique([r.get(COLUMN_CONTROLLER_NAME) for r in frontier_rows])
        for c in controllers:
            rows_c = [r for r in frontier_rows if r.get(COLUMN_CONTROLLER_NAME) == c]
            xs = [to_float(r.get(x_key)) for r in rows_c]
            ys = [to_float(r.get(COLUMN_PERFORMANCE)) for r in rows_c]

            pts = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
            if not pts:
                continue
            xs2, ys2 = zip(*pts)
            ax.scatter(xs2, ys2, label=str(c))

        ax.set_xlabel(x_key)
        ax.set_ylabel(COLUMN_PERFORMANCE)
        ax.set_title('Efficiency frontier (points)')
        ax.legend()

        if save_dir_p is not None:
            out = save_dir_p / 'efficiency_frontier_points.png'
            fig.savefig(out, bbox_inches='tight')
            save_paths.append(out)

    if mode in {'show', 'both'}:
        plt.show()
    else:
        plt.close('all')

    return save_paths
