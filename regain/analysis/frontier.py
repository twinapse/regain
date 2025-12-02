"""
Efficiency frontier automation.

Computes Pareto frontiers over:
  - data cost (b = shots per class),
  - parameter cost (controller_model_param_count),
  - performance (rho_mean or a_ctrl_mean).

Also emits an optional scalar 'total_cost' view when repair_budget_total is known:
  total_cost = repair_budget_total + controller_model_param_count
"""

from pathlib import Path
from typing import Any

from regain.analysis.utils import to_float
from regain.analysis.utils import to_int
from regain.analysis.utils import write_csv
from regain.utils import get_logger

__all__ = [
    'write_efficiency_frontiers',
]


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
    a_b = to_float(a.get('b'))
    b_b = to_float(b.get('b'))
    a_p = to_int(a.get('controller_model_param_count'))
    b_p = to_int(b.get('controller_model_param_count'))
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
    perf_key: str = 'rho_mean_avg',
) -> tuple[Path, Path]:
    """
    Compute and write frontier tables from aggregated curve points.

    Args:
        curve_rows: Rows from recoverability_curve.csv (already aggregated across seeds).
        out_dir: Output directory.
        perf_key: Which performance column to use (maximize). Typical:
            - 'rho_mean_avg'
            - 'a_ctrl_mean_avg'

    Returns:
        (frontier_points_path, frontier_pareto_path)
    """
    logger = get_logger()
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    # Normalize points: rename perf_key -> 'performance' for a stable output schema.
    points: list[dict[str, Any]] = []
    missing_param_count_controllers: set[str] = set()
    for r in curve_rows:
        perf = to_float(r.get(perf_key))
        pc = r.get('controller_model_param_count')
        if to_int(pc) is None:
            cn = r.get('controller_name')
            if cn is not None:
                missing_param_count_controllers.add(str(cn))
        points.append({
            'controller_name': r.get('controller_name'),
            'b': r.get('b'),
            'repair_budget_per_class': r.get('repair_budget_per_class'),
            'repair_budget_total': r.get('repair_budget_total'),
            'num_classes': r.get('num_classes'),
            'controller_model_param_count': pc,
            'performance': perf,
            'performance_key': perf_key,
            'rho_mean_avg': r.get('rho_mean_avg'),
            'a_ctrl_mean_avg': r.get('a_ctrl_mean_avg'),
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
        rb = to_int(p.get('repair_budget_total'))
        pc = to_int(p.get('controller_model_param_count'))
        if rb is not None and pc is not None:
            p['total_cost'] = int(rb + pc)
        else:
            p['total_cost'] = None

    points_path = outp / 'frontier_points.csv'
    write_csv(points_path, points)
    logger.warning(f'Wrote {points_path}')

    pareto = _pareto_set(points, perf_key='performance')
    pareto_path = outp / 'frontier_pareto.csv'
    write_csv(pareto_path, pareto)
    logger.warning(f'Wrote {pareto_path}')

    return points_path, pareto_path
