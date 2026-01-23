"""
Recoverability curve automation.

This module aggregates MLflow runs (collected via `regain.analysis.collectors`) into:
  - budget curves: ρ_mean and/or A_ctrl_mean vs b (shots per class), and
  - task-age curves: ρ vs exp_idx / task_age.
"""

from pathlib import Path
from typing import Any

from regain.analysis.utils import mean
from regain.analysis.utils import stdev
from regain.analysis.utils import to_float
from regain.analysis.utils import write_csv
from regain.utils import get_logger

__all__ = [
    'write_recoverability_curves',
]


def write_recoverability_curves(
    *,
    runs_table: list[dict[str, Any]],
    experiences_table: list[dict[str, Any]],
    out_dir: str | Path,
) -> tuple[Path, Path]:
    """
    Compute and write recoverability curves and task-age curves.

    Outputs:
      - recoverability_curve.csv
      - task_age_rho.csv

    Args:
        runs_table: One row per run (from `regain.analysis.collectors`).
        experiences_table: One row per experience (run, exp_idx) (from `regain.analysis.collectors`).
        out_dir: Output directory.

    Returns:
        (recoverability_curve_path, task_age_rho_path)
    """
    logger = get_logger()
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    # ---- A) Budget curves: aggregate across seeds for (controller_name, b) ----
    groups: dict[tuple[str, Any], list[dict[str, Any]]] = {}
    for r in runs_table:
        controller = str(r.get('controller_name') or 'none')
        b = r.get('b')
        key = (controller, b)
        groups.setdefault(key, []).append(r)

    curve_rows: list[dict[str, Any]] = []
    for (controller, b), rows in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        rho_vals = [to_float(x.get('rho_mean')) for x in rows]
        a_ctrl_vals = [to_float(x.get('a_ctrl_mean')) for x in rows]
        a_post_vals = [to_float(x.get('a_post_mean')) for x in rows]

        curve_rows.append({
            'controller_name': controller,
            'b': b,
            'repair_budget_per_class': rows[0].get('repair_budget_per_class'),
            'repair_budget_total': rows[0].get('repair_budget_total'),
            'num_classes': rows[0].get('num_classes'),
            'controller_model_param_count': rows[0].get('controller_model_param_count'),
            'rho_mean_avg': mean(rho_vals),
            'rho_mean_std': stdev(rho_vals),
            'a_ctrl_mean_avg': mean(a_ctrl_vals),
            'a_ctrl_mean_std': stdev(a_ctrl_vals),
            'a_post_mean_avg': mean(a_post_vals),
            'a_post_mean_std': stdev(a_post_vals),
            'n_seeds': len({r.get('seed') for r in rows}),
            'n_runs': len(rows),
        })

    curve_path = outp / 'recoverability_curve.csv'
    write_csv(curve_path, curve_rows)
    logger.warning(f'Wrote {curve_path}')

    # ---- B) Task-age curves: aggregate ρ per exp_idx / task_age for (controller_name, b) ----
    experience_groups: dict[tuple[str, Any, int, Any], list[dict[str, Any]]] = {}
    for row in experiences_table:
        controller = str(row.get('controller_name') or 'none')
        b = row.get('b')
        exp_idx = row.get('exp_idx')
        task_age = row.get('task_age')
        if exp_idx is None:
            continue
        key = (controller, b, int(exp_idx), task_age)
        experience_groups.setdefault(key, []).append(row)

    task_rows: list[dict[str, Any]] = []
    for (controller, b, exp_idx, task_age), rows in sorted(experience_groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        rho_vals = [to_float(x.get('rho')) for x in rows]
        task_rows.append({
            'controller_name': controller,
            'b': b,
            'exp_idx': int(exp_idx),
            'task_age': task_age,
            'rho_avg': mean(rho_vals),
            'rho_std': stdev(rho_vals),
            'n_seeds': len({r.get('seed') for r in rows}),
            'n_rows': len(rows),
        })

    task_path = outp / 'task_age_rho.csv'
    write_csv(task_path, task_rows)
    logger.warning(f'Wrote {task_path}')

    return curve_path, task_path
