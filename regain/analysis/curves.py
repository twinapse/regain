"""
Recoverability curve automation.

This module aggregates run-level and experience-level analysis tables into:
  - budget curves (recoverability and final accuracy vs repair budget),
  - task-age curves (recoverability vs task age),
  - calibration-vs-budget summaries, and
  - latency-vs-budget summaries.
"""

from pathlib import Path
from typing import Any

from regain.analysis.artifacts import ARTIFACT_RHO
from regain.analysis.utils import mean
from regain.analysis.utils import stdev
from regain.analysis.utils import to_float
from regain.analysis.utils import write_csv
from regain.constants import ANALYSIS_ACC_FINAL_AVG_CTRL
from regain.constants import ANALYSIS_RHO_AVG
from regain.constants import COLUMN_B
from regain.constants import COLUMN_CONTROLLER_MODEL_PARAM_COUNT
from regain.constants import COLUMN_CONTROLLER_NAME
from regain.constants import COLUMN_EXP_IDX
from regain.constants import COLUMN_NUM_CLASSES
from regain.constants import COLUMN_REPAIR_BUDGET_FRACTION
from regain.constants import COLUMN_REPAIR_BUDGET_TOTAL
from regain.constants import COLUMN_REPAIR_SET_TOTAL
from regain.constants import COLUMN_REPAIR_SPLIT_FRACTION
from regain.constants import COLUMN_SEED
from regain.constants import COLUMN_TASK_AGE
from regain.constants import RUN_ACC_FINAL_AVG_BASE
from regain.constants import RUN_ACC_FINAL_AVG_CTRL
from regain.constants import RUN_CALIB_MAX_ECE
from regain.constants import RUN_LATENCY_MS_PER_SAMPLE_BASE
from regain.constants import RUN_LATENCY_MS_PER_SAMPLE_CTRL
from regain.constants import RUN_LATENCY_MS_RATIO
from regain.constants import RUN_RHO_AVG
from regain.utils import get_logger

__all__ = [
    'write_recoverability_curves',
]

_COLUMN_N_ROWS = 'n_rows'
_COLUMN_N_RUNS = 'n_runs'
_COLUMN_N_SEEDS = 'n_seeds'

_ANALYSIS_ACC_FINAL_AVG_BASE = 'analysis.acc.final.avg.base'
_ANALYSIS_ACC_FINAL_STD_BASE = 'analysis.acc.final.std.base'
_ANALYSIS_ACC_FINAL_STD_CTRL = 'analysis.acc.final.std.ctrl'
_ANALYSIS_CALIB_MAX_ECE_AVG = 'analysis.calibration.max_ece.avg'
_ANALYSIS_CALIB_MAX_ECE_STD = 'analysis.calibration.max_ece.std'
_ANALYSIS_LATENCY_MS_PER_SAMPLE_AVG_BASE = 'analysis.latency.ms_per_sample.avg.base'
_ANALYSIS_LATENCY_MS_PER_SAMPLE_AVG_CTRL = 'analysis.latency.ms_per_sample.avg.ctrl'
_ANALYSIS_LATENCY_MS_PER_SAMPLE_STD_BASE = 'analysis.latency.ms_per_sample.std.base'
_ANALYSIS_LATENCY_MS_PER_SAMPLE_STD_CTRL = 'analysis.latency.ms_per_sample.std.ctrl'
_ANALYSIS_LATENCY_MS_RATIO_AVG = 'analysis.latency.ms_ratio.avg'
_ANALYSIS_LATENCY_MS_RATIO_STD = 'analysis.latency.ms_ratio.std'
_ANALYSIS_RHO_STD = 'analysis.repair.rho.std'


def write_recoverability_curves(
    *,
    runs_table: list[dict[str, Any]],
    experiences_table: list[dict[str, Any]],
    out_dir: str | Path,
) -> tuple[Path, Path, Path, Path]:
    """
    Compute and write recoverability, calibration, task-age, and latency curves.

    Outputs:
      - recoverability_curve.csv
      - task_age_rho.csv
      - calibration_vs_budget.csv
      - latency_vs_budget.csv

    Args:
        runs_table: One row per run (from `regain.analysis.collectors`).
        experiences_table: One row per experience (run, exp_idx) (from `regain.analysis.collectors`).
        out_dir: Output directory.

    Returns:
        tuple[Path, Path, Path, Path]: Written file paths
            `(recoverability_curve, task_age_rho, calibration_vs_budget, latency_vs_budget)`.
    """
    logger = get_logger()
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    groups: dict[tuple[str, Any], list[dict[str, Any]]] = {}
    for row in runs_table:
        controller = str(row.get(COLUMN_CONTROLLER_NAME) or 'none')
        budget = row.get(COLUMN_B)
        groups.setdefault((controller, budget), []).append(row)

    curve_rows: list[dict[str, Any]] = []
    for (controller, budget), rows in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        rho_vals = [to_float(row.get(RUN_RHO_AVG)) for row in rows]
        acc_ctrl_vals = [to_float(row.get(RUN_ACC_FINAL_AVG_CTRL)) for row in rows]
        acc_base_vals = [to_float(row.get(RUN_ACC_FINAL_AVG_BASE)) for row in rows]

        curve_rows.append({
            COLUMN_CONTROLLER_NAME: controller,
            COLUMN_B: budget,
            COLUMN_REPAIR_BUDGET_FRACTION: rows[0].get(COLUMN_REPAIR_BUDGET_FRACTION),
            COLUMN_REPAIR_BUDGET_TOTAL: rows[0].get(COLUMN_REPAIR_BUDGET_TOTAL),
            COLUMN_REPAIR_SET_TOTAL: rows[0].get(COLUMN_REPAIR_SET_TOTAL),
            COLUMN_REPAIR_SPLIT_FRACTION: rows[0].get(COLUMN_REPAIR_SPLIT_FRACTION),
            COLUMN_NUM_CLASSES: rows[0].get(COLUMN_NUM_CLASSES),
            COLUMN_CONTROLLER_MODEL_PARAM_COUNT: rows[0].get(COLUMN_CONTROLLER_MODEL_PARAM_COUNT),
            ANALYSIS_RHO_AVG: mean(rho_vals),
            _ANALYSIS_RHO_STD: stdev(rho_vals),
            ANALYSIS_ACC_FINAL_AVG_CTRL: mean(acc_ctrl_vals),
            _ANALYSIS_ACC_FINAL_STD_CTRL: stdev(acc_ctrl_vals),
            _ANALYSIS_ACC_FINAL_AVG_BASE: mean(acc_base_vals),
            _ANALYSIS_ACC_FINAL_STD_BASE: stdev(acc_base_vals),
            _COLUMN_N_SEEDS: len({row.get(COLUMN_SEED) for row in rows}),
            _COLUMN_N_RUNS: len(rows),
        })

    curve_path = outp / 'recoverability_curve.csv'
    write_csv(curve_path, curve_rows)
    logger.warning(f'Wrote {curve_path}')

    experience_groups: dict[tuple[str, Any, int, Any], list[dict[str, Any]]] = {}
    for row in experiences_table:
        controller = str(row.get(COLUMN_CONTROLLER_NAME) or 'none')
        budget = row.get(COLUMN_B)
        exp_idx = row.get(COLUMN_EXP_IDX)
        task_age = row.get(COLUMN_TASK_AGE)
        if exp_idx is None:
            continue
        experience_groups.setdefault((controller, budget, int(exp_idx), task_age), []).append(row)

    task_rows: list[dict[str, Any]] = []
    for (controller, budget, exp_idx, task_age), rows in sorted(
        experience_groups.items(),
        key=lambda kv: (kv[0][0], kv[0][1], kv[0][2]),
    ):
        rho_vals = [to_float(row.get(ARTIFACT_RHO)) for row in rows]
        task_rows.append({
            COLUMN_CONTROLLER_NAME: controller,
            COLUMN_B: budget,
            COLUMN_EXP_IDX: int(exp_idx),
            COLUMN_TASK_AGE: task_age,
            ANALYSIS_RHO_AVG: mean(rho_vals),
            _ANALYSIS_RHO_STD: stdev(rho_vals),
            _COLUMN_N_SEEDS: len({row.get(COLUMN_SEED) for row in rows}),
            _COLUMN_N_ROWS: len(rows),
        })

    task_path = outp / 'task_age_rho.csv'
    write_csv(task_path, task_rows)
    logger.warning(f'Wrote {task_path}')

    calibration_rows: list[dict[str, Any]] = []
    for (controller, budget), rows in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        calib_vals = [to_float(row.get(RUN_CALIB_MAX_ECE)) for row in rows]
        if not any(value is not None for value in calib_vals):
            continue
        calibration_rows.append({
            COLUMN_CONTROLLER_NAME: controller,
            COLUMN_B: budget,
            COLUMN_REPAIR_BUDGET_FRACTION: rows[0].get(COLUMN_REPAIR_BUDGET_FRACTION),
            COLUMN_REPAIR_SPLIT_FRACTION: rows[0].get(COLUMN_REPAIR_SPLIT_FRACTION),
            _ANALYSIS_CALIB_MAX_ECE_AVG: mean(calib_vals),
            _ANALYSIS_CALIB_MAX_ECE_STD: stdev(calib_vals),
            _COLUMN_N_SEEDS: len({row.get(COLUMN_SEED) for row in rows}),
            _COLUMN_N_RUNS: len(rows),
        })

    calib_path = outp / 'calibration_vs_budget.csv'
    write_csv(calib_path, calibration_rows)
    logger.warning(f'Wrote {calib_path}')

    latency_rows: list[dict[str, Any]] = []
    for (controller, budget), rows in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        base_ms_vals = [to_float(row.get(RUN_LATENCY_MS_PER_SAMPLE_BASE)) for row in rows]
        ctrl_ms_vals = [to_float(row.get(RUN_LATENCY_MS_PER_SAMPLE_CTRL)) for row in rows]
        ratio_vals = [to_float(row.get(RUN_LATENCY_MS_RATIO)) for row in rows]
        if not any(value is not None for value in base_ms_vals + ctrl_ms_vals + ratio_vals):
            continue
        latency_rows.append({
            COLUMN_CONTROLLER_NAME: controller,
            COLUMN_B: budget,
            COLUMN_REPAIR_BUDGET_FRACTION: rows[0].get(COLUMN_REPAIR_BUDGET_FRACTION),
            COLUMN_REPAIR_SPLIT_FRACTION: rows[0].get(COLUMN_REPAIR_SPLIT_FRACTION),
            _ANALYSIS_LATENCY_MS_PER_SAMPLE_AVG_BASE: mean(base_ms_vals),
            _ANALYSIS_LATENCY_MS_PER_SAMPLE_STD_BASE: stdev(base_ms_vals),
            _ANALYSIS_LATENCY_MS_PER_SAMPLE_AVG_CTRL: mean(ctrl_ms_vals),
            _ANALYSIS_LATENCY_MS_PER_SAMPLE_STD_CTRL: stdev(ctrl_ms_vals),
            _ANALYSIS_LATENCY_MS_RATIO_AVG: mean(ratio_vals),
            _ANALYSIS_LATENCY_MS_RATIO_STD: stdev(ratio_vals),
            _COLUMN_N_SEEDS: len({row.get(COLUMN_SEED) for row in rows}),
            _COLUMN_N_RUNS: len(rows),
        })

    latency_path = outp / 'latency_vs_budget.csv'
    write_csv(latency_path, latency_rows)
    logger.warning(f'Wrote {latency_path}')

    return curve_path, task_path, calib_path, latency_path
