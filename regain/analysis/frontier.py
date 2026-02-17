"""
Efficiency frontier automation.

Computes Pareto frontiers over:
  - data cost (b = shots per class),
  - parameter cost (controller_model_param_count),
  - performance (rho_mean or a_ctrl_mean).

Also emits an optional scalar total-cost view when repair_budget_total is known:
  total_cost = repair_budget_total + controller_model_param_count
"""

from pathlib import Path
from typing import Any

from regain.analysis.utils import to_float
from regain.analysis.utils import to_int
from regain.analysis.utils import write_csv
from regain.constants import COLUMN_B
from regain.constants import COLUMN_CONTROLLER_MODEL_PARAM_COUNT
from regain.constants import COLUMN_CONTROLLER_NAME
from regain.constants import COLUMN_NUM_CLASSES
from regain.constants import COLUMN_PERFORMANCE
from regain.constants import COLUMN_REPAIR_BUDGET_PER_CLASS
from regain.constants import COLUMN_REPAIR_BUDGET_TOTAL
from regain.constants import COLUMN_TOTAL_COST
from regain.constants import METRIC_A_CTRL_MEAN_AVG
from regain.constants import METRIC_RHO_MEAN_AVG
from regain.utils import get_logger

__all__ = [
    'write_efficiency_frontiers',
]

_COLUMN_PERFORMANCE_KEY = 'performance_key'


def _dominates(a: dict[str, Any], b: dict[str, Any], *, perf_key: str) -> bool:
    """
    Check 3D dominance:
      - minimize (b, controller_model_param_count)
      - maximize (perf_key)

    Args:
        a: Candidate dominating point.
        b: Candidate dominated point.
        perf_key: Performance key (maximize).

    Returns:
        True if a dominates b, else False.
    """
    a_b = to_float(a.get(COLUMN_B))
    b_b = to_float(b.get(COLUMN_B))
    a_p = to_int(a.get(COLUMN_CONTROLLER_MODEL_PARAM_COUNT))
    b_p = to_int(b.get(COLUMN_CONTROLLER_MODEL_PARAM_COUNT))
    a_perf = to_float(a.get(perf_key))
    b_perf = to_float(b.get(perf_key))

    if a_b is None or b_b is None or a_p is None or b_p is None or a_perf is None or b_perf is None:
        return False

    no_worse = (a_b <= b_b) and (a_p <= b_p) and (a_perf >= b_perf)
    strictly_better = (a_b < b_b) or (a_p < b_p) or (a_perf > b_perf)
    return bool(no_worse and strictly_better)


def _pareto_set(points: list[dict[str, Any]], *, perf_key: str) -> list[dict[str, Any]]:
    """
    Compute the non-dominated subset (Pareto set) under the dominance rule in _dominates().

    Args:
        points: Input points.
        perf_key: Performance key to maximize.

    Returns:
        List of non-dominated points.
    """
    keep: list[dict[str, Any]] = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            if _dominates(q, p, perf_key=perf_key):
                dominated = True
                break
        if not dominated:
            keep.append(p)
    return keep


def write_efficiency_frontiers(
    *,
    curve_rows: list[dict[str, Any]],
    out_dir: str | Path,
    perf_key: str = METRIC_RHO_MEAN_AVG,
) -> tuple[Path, Path]:
    """
    Compute and write frontier tables from aggregated curve points.

    Args:
        curve_rows: Rows from recoverability_curve.csv (already aggregated across seeds).
        out_dir: Output directory.
        perf_key: Which performance column to use (maximize), usually one of the aggregated curve metric columns.

    Returns:
        (frontier_points_path, frontier_pareto_path)
    """
    logger = get_logger()
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    # Normalize points: rename perf_key -> performance for a stable output schema.
    points: list[dict[str, Any]] = []
    missing_param_count_controllers: set[str] = set()
    for r in curve_rows:
        perf = to_float(r.get(perf_key))
        pc = r.get(COLUMN_CONTROLLER_MODEL_PARAM_COUNT)
        if to_int(pc) is None:
            cn = r.get(COLUMN_CONTROLLER_NAME)
            if cn is not None:
                missing_param_count_controllers.add(str(cn))
        points.append({
            COLUMN_CONTROLLER_NAME: r.get(COLUMN_CONTROLLER_NAME),
            COLUMN_B: r.get(COLUMN_B),
            COLUMN_REPAIR_BUDGET_PER_CLASS: r.get(COLUMN_REPAIR_BUDGET_PER_CLASS),
            COLUMN_REPAIR_BUDGET_TOTAL: r.get(COLUMN_REPAIR_BUDGET_TOTAL),
            COLUMN_NUM_CLASSES: r.get(COLUMN_NUM_CLASSES),
            COLUMN_CONTROLLER_MODEL_PARAM_COUNT: pc,
            COLUMN_PERFORMANCE: perf,
            _COLUMN_PERFORMANCE_KEY: perf_key,
            METRIC_RHO_MEAN_AVG: r.get(METRIC_RHO_MEAN_AVG),
            METRIC_A_CTRL_MEAN_AVG: r.get(METRIC_A_CTRL_MEAN_AVG),
        })

    if missing_param_count_controllers:
        # Pareto dominance and total_cost need controller_model_param_count.
        missing_sorted = ', '.join(sorted(missing_param_count_controllers))
        logger.warning(
            'Missing controller_model_param_count for some points (controllers: '
            f'{missing_sorted}). Pareto filtering and total_cost will be less informative.'
        )

    # Total-cost view (optional).
    for p in points:
        rb = to_int(p.get(COLUMN_REPAIR_BUDGET_TOTAL))
        pc = to_int(p.get(COLUMN_CONTROLLER_MODEL_PARAM_COUNT))
        if rb is not None and pc is not None:
            p[COLUMN_TOTAL_COST] = int(rb + pc)
        else:
            p[COLUMN_TOTAL_COST] = None

    points_path = outp / 'frontier_points.csv'
    write_csv(points_path, points)
    logger.warning(f'Wrote {points_path}')

    pareto = _pareto_set(points, perf_key=COLUMN_PERFORMANCE)
    pareto_path = outp / 'frontier_pareto.csv'
    write_csv(pareto_path, pareto)
    logger.warning(f'Wrote {pareto_path}')

    return points_path, pareto_path
