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
from regain.constants import COLUMN_B
from regain.constants import COLUMN_CONTROLLER_MODEL_PARAM_COUNT
from regain.constants import COLUMN_CONTROLLER_NAME
from regain.constants import COLUMN_EXP_IDX
from regain.constants import COLUMN_NUM_CLASSES
from regain.constants import COLUMN_REPAIR_BUDGET_PER_CLASS
from regain.constants import COLUMN_REPAIR_BUDGET_TOTAL
from regain.constants import COLUMN_SEED
from regain.constants import COLUMN_TASK_AGE
from regain.constants import METRIC_A_CTRL_MEAN
from regain.constants import METRIC_A_CTRL_MEAN_AVG
from regain.constants import METRIC_A_POST_MEAN
from regain.constants import METRIC_RHO
from regain.constants import METRIC_RHO_MEAN
from regain.constants import METRIC_RHO_MEAN_AVG
from regain.utils import get_logger

__all__ = [
    'write_recoverability_curves',
]

_COLUMN_N_ROWS = 'n_rows'
_COLUMN_N_RUNS = 'n_runs'
_COLUMN_N_SEEDS = 'n_seeds'
_METRIC_A_CTRL_MEAN_STD = 'a_ctrl_mean_std'
_METRIC_A_POST_MEAN_AVG = 'a_post_mean_avg'
_METRIC_A_POST_MEAN_STD = 'a_post_mean_std'
_METRIC_RHO_AVG = 'rho_avg'
_METRIC_RHO_MEAN_STD = 'rho_mean_std'
_METRIC_RHO_STD = 'rho_std'


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
        controller = str(r.get(COLUMN_CONTROLLER_NAME) or 'none')
        b = r.get(COLUMN_B)
        key = (controller, b)
        groups.setdefault(key, []).append(r)

    curve_rows: list[dict[str, Any]] = []
    for (controller, b), rows in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        rho_vals = [to_float(x.get(METRIC_RHO_MEAN)) for x in rows]
        a_ctrl_vals = [to_float(x.get(METRIC_A_CTRL_MEAN)) for x in rows]
        a_post_vals = [to_float(x.get(METRIC_A_POST_MEAN)) for x in rows]

        curve_rows.append({
            COLUMN_CONTROLLER_NAME: controller,
            COLUMN_B: b,
            COLUMN_REPAIR_BUDGET_PER_CLASS: rows[0].get(COLUMN_REPAIR_BUDGET_PER_CLASS),
            COLUMN_REPAIR_BUDGET_TOTAL: rows[0].get(COLUMN_REPAIR_BUDGET_TOTAL),
            COLUMN_NUM_CLASSES: rows[0].get(COLUMN_NUM_CLASSES),
            COLUMN_CONTROLLER_MODEL_PARAM_COUNT: rows[0].get(COLUMN_CONTROLLER_MODEL_PARAM_COUNT),
            METRIC_RHO_MEAN_AVG: mean(rho_vals),
            _METRIC_RHO_MEAN_STD: stdev(rho_vals),
            METRIC_A_CTRL_MEAN_AVG: mean(a_ctrl_vals),
            _METRIC_A_CTRL_MEAN_STD: stdev(a_ctrl_vals),
            _METRIC_A_POST_MEAN_AVG: mean(a_post_vals),
            _METRIC_A_POST_MEAN_STD: stdev(a_post_vals),
            _COLUMN_N_SEEDS: len({r.get(COLUMN_SEED) for r in rows}),
            _COLUMN_N_RUNS: len(rows),
        })

    curve_path = outp / 'recoverability_curve.csv'
    write_csv(curve_path, curve_rows)
    logger.warning(f'Wrote {curve_path}')

    # ---- B) Task-age curves: aggregate ρ per exp_idx / task_age for (controller_name, b) ----
    experience_groups: dict[tuple[str, Any, int, Any], list[dict[str, Any]]] = {}
    for row in experiences_table:
        controller = str(row.get(COLUMN_CONTROLLER_NAME) or 'none')
        b = row.get(COLUMN_B)
        exp_idx = row.get(COLUMN_EXP_IDX)
        task_age = row.get(COLUMN_TASK_AGE)
        if exp_idx is None:
            continue
        key = (controller, b, int(exp_idx), task_age)
        experience_groups.setdefault(key, []).append(row)

    task_rows: list[dict[str, Any]] = []
    for (controller, b, exp_idx, task_age), rows in sorted(experience_groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        rho_vals = [to_float(x.get(METRIC_RHO)) for x in rows]
        task_rows.append({
            COLUMN_CONTROLLER_NAME: controller,
            COLUMN_B: b,
            COLUMN_EXP_IDX: int(exp_idx),
            COLUMN_TASK_AGE: task_age,
            _METRIC_RHO_AVG: mean(rho_vals),
            _METRIC_RHO_STD: stdev(rho_vals),
            _COLUMN_N_SEEDS: len({r.get(COLUMN_SEED) for r in rows}),
            _COLUMN_N_ROWS: len(rows),
        })

    task_path = outp / 'task_age_rho.csv'
    write_csv(task_path, task_rows)
    logger.warning(f'Wrote {task_path}')

    return curve_path, task_path
