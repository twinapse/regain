"""
Tests for analysis writer modules.
"""

import csv
from pathlib import Path

import pytest

from regain.analysis.artifacts import ARTIFACT_RHO
from regain.analysis.curves import write_recoverability_curves
from regain.analysis.frontier import write_efficiency_frontiers
from regain.analysis.predictive import write_predictive_correlations
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
from regain.constants import RUN_CALIB_ECE
from regain.constants import RUN_CALIB_MAX_ECE
from regain.constants import RUN_LATENCY_MS_PER_SAMPLE_BASE
from regain.constants import RUN_LATENCY_MS_PER_SAMPLE_CTRL
from regain.constants import RUN_LATENCY_MS_RATIO
from regain.constants import RUN_RHO_AVG


def _read_rows(path: Path) -> list[dict[str, str]]:
    """
    Read one CSV file into a list of dict rows.

    Args:
        path (Path): CSV path.

    Returns:
        list[dict[str, str]]: Parsed rows.
    """
    with path.open('r', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


class TestWriteRecoverabilityCurves:
    def test_writes_recoverability_and_related_analysis_outputs(self, tmp_path: Path) -> None:
        runs_table = [
            {
                COLUMN_CONTROLLER_NAME: 'repair_a',
                COLUMN_B: 0.5,
                COLUMN_REPAIR_BUDGET_FRACTION: 0.5,
                COLUMN_REPAIR_BUDGET_TOTAL: 5,
                COLUMN_REPAIR_SET_TOTAL: 10,
                COLUMN_REPAIR_SPLIT_FRACTION: 0.2,
                COLUMN_NUM_CLASSES: 4,
                COLUMN_CONTROLLER_MODEL_PARAM_COUNT: 100,
                COLUMN_SEED: 1,
                RUN_RHO_AVG: 0.20,
                RUN_ACC_FINAL_AVG_CTRL: 0.60,
                RUN_ACC_FINAL_AVG_BASE: 0.50,
                RUN_CALIB_MAX_ECE: 0.30,
                RUN_LATENCY_MS_PER_SAMPLE_BASE: 1.00,
                RUN_LATENCY_MS_PER_SAMPLE_CTRL: 2.00,
                RUN_LATENCY_MS_RATIO: 2.00,
            },
            {
                COLUMN_CONTROLLER_NAME: 'repair_a',
                COLUMN_B: 0.5,
                COLUMN_REPAIR_BUDGET_FRACTION: 0.5,
                COLUMN_REPAIR_BUDGET_TOTAL: 5,
                COLUMN_REPAIR_SET_TOTAL: 10,
                COLUMN_REPAIR_SPLIT_FRACTION: 0.2,
                COLUMN_NUM_CLASSES: 4,
                COLUMN_CONTROLLER_MODEL_PARAM_COUNT: 100,
                COLUMN_SEED: 2,
                RUN_RHO_AVG: 0.40,
                RUN_ACC_FINAL_AVG_CTRL: 0.80,
                RUN_ACC_FINAL_AVG_BASE: 0.70,
                RUN_CALIB_MAX_ECE: 0.50,
                RUN_LATENCY_MS_PER_SAMPLE_BASE: 1.20,
                RUN_LATENCY_MS_PER_SAMPLE_CTRL: 2.40,
                RUN_LATENCY_MS_RATIO: 2.00,
            },
        ]
        experiences_table = [
            {
                COLUMN_CONTROLLER_NAME: 'repair_a',
                COLUMN_B: 0.5,
                COLUMN_SEED: 1,
                COLUMN_EXP_IDX: 0,
                COLUMN_TASK_AGE: 1,
                ARTIFACT_RHO: 0.10,
            },
            {
                COLUMN_CONTROLLER_NAME: 'repair_a',
                COLUMN_B: 0.5,
                COLUMN_SEED: 2,
                COLUMN_EXP_IDX: 0,
                COLUMN_TASK_AGE: 1,
                ARTIFACT_RHO: 0.30,
            },
            {
                COLUMN_CONTROLLER_NAME: 'repair_a',
                COLUMN_B: 0.5,
                COLUMN_SEED: 1,
                COLUMN_EXP_IDX: 1,
                COLUMN_TASK_AGE: 0,
                ARTIFACT_RHO: 0.20,
            },
            {
                COLUMN_CONTROLLER_NAME: 'repair_a',
                COLUMN_B: 0.5,
                COLUMN_SEED: 2,
                COLUMN_EXP_IDX: 1,
                COLUMN_TASK_AGE: 0,
                ARTIFACT_RHO: 0.40,
            },
        ]

        curve_path, task_path, calib_path, latency_path = write_recoverability_curves(
            runs_table=runs_table,
            experiences_table=experiences_table,
            out_dir=tmp_path,
        )

        curve_rows = _read_rows(curve_path)
        assert len(curve_rows) == 1
        assert float(curve_rows[0][ANALYSIS_RHO_AVG]) == pytest.approx(0.30)
        assert float(curve_rows[0][ANALYSIS_ACC_FINAL_AVG_CTRL]) == pytest.approx(0.70)
        assert float(curve_rows[0]['analysis.acc.final.avg.base']) == pytest.approx(0.60)
        assert int(curve_rows[0]['n_seeds']) == 2
        assert int(curve_rows[0]['n_runs']) == 2

        task_rows = _read_rows(task_path)
        assert len(task_rows) == 2
        task_row_by_age = {
            int(row[COLUMN_TASK_AGE]): row
            for row in task_rows
        }
        assert float(task_row_by_age[1][ANALYSIS_RHO_AVG]) == pytest.approx(0.20)
        assert float(task_row_by_age[0][ANALYSIS_RHO_AVG]) == pytest.approx(0.30)

        calib_rows = _read_rows(calib_path)
        assert len(calib_rows) == 1
        assert float(calib_rows[0]['analysis.calibration.max_ece.avg']) == pytest.approx(0.40)

        latency_rows = _read_rows(latency_path)
        assert len(latency_rows) == 1
        assert float(latency_rows[0]['analysis.latency.ms_per_sample.avg.base']) == pytest.approx(1.10)
        assert float(latency_rows[0]['analysis.latency.ms_per_sample.avg.ctrl']) == pytest.approx(2.20)
        assert float(latency_rows[0]['analysis.latency.ms_ratio.avg']) == pytest.approx(2.00)


class TestWriteEfficiencyFrontiers:
    def test_writes_points_and_pareto_frontier(self, tmp_path: Path) -> None:
        curve_rows = [
            {
                COLUMN_CONTROLLER_NAME: 'controller_a',
                COLUMN_B: 0.50,
                COLUMN_REPAIR_BUDGET_FRACTION: 0.50,
                COLUMN_REPAIR_BUDGET_TOTAL: 5,
                COLUMN_NUM_CLASSES: 4,
                COLUMN_CONTROLLER_MODEL_PARAM_COUNT: 10,
                ANALYSIS_RHO_AVG: 0.80,
                ANALYSIS_ACC_FINAL_AVG_CTRL: 0.85,
            },
            {
                COLUMN_CONTROLLER_NAME: 'controller_b',
                COLUMN_B: 0.60,
                COLUMN_REPAIR_BUDGET_FRACTION: 0.60,
                COLUMN_REPAIR_BUDGET_TOTAL: 6,
                COLUMN_NUM_CLASSES: 4,
                COLUMN_CONTROLLER_MODEL_PARAM_COUNT: 10,
                ANALYSIS_RHO_AVG: 0.70,
                ANALYSIS_ACC_FINAL_AVG_CTRL: 0.75,
            },
            {
                COLUMN_CONTROLLER_NAME: 'controller_c',
                COLUMN_B: 0.40,
                COLUMN_REPAIR_BUDGET_FRACTION: 0.40,
                COLUMN_REPAIR_BUDGET_TOTAL: 4,
                COLUMN_NUM_CLASSES: 4,
                COLUMN_CONTROLLER_MODEL_PARAM_COUNT: 20,
                ANALYSIS_RHO_AVG: 0.90,
                ANALYSIS_ACC_FINAL_AVG_CTRL: 0.95,
            },
        ]

        points_path, pareto_path = write_efficiency_frontiers(
            curve_rows=curve_rows,
            out_dir=tmp_path,
            perf_key=ANALYSIS_RHO_AVG,
        )

        point_rows = _read_rows(points_path)
        assert len(point_rows) == 3
        point_row_by_controller = {
            row[COLUMN_CONTROLLER_NAME]: row
            for row in point_rows
        }
        assert float(point_row_by_controller['controller_a']['performance']) == pytest.approx(0.80)
        assert int(point_row_by_controller['controller_a']['total_cost']) == 15

        pareto_rows = _read_rows(pareto_path)
        pareto_controllers = {row[COLUMN_CONTROLLER_NAME] for row in pareto_rows}
        assert pareto_controllers == {'controller_a', 'controller_c'}


class TestWritePredictiveCorrelations:
    def test_writes_predictive_correlations_from_experience_rows(self, tmp_path: Path) -> None:
        experiences_table = [
            {
                COLUMN_CONTROLLER_NAME: 'repair_a',
                COLUMN_B: 0.5,
                RUN_CALIB_ECE: 1.0,
                ARTIFACT_RHO: 2.0,
            },
            {
                COLUMN_CONTROLLER_NAME: 'repair_a',
                COLUMN_B: 0.5,
                RUN_CALIB_ECE: 2.0,
                ARTIFACT_RHO: 4.0,
            },
            {
                COLUMN_CONTROLLER_NAME: 'repair_a',
                COLUMN_B: 0.5,
                RUN_CALIB_ECE: 3.0,
                ARTIFACT_RHO: 6.0,
            },
        ]

        output_path = write_predictive_correlations(
            experiences_table=experiences_table,
            out_dir=tmp_path,
        )

        rows = _read_rows(output_path)
        assert len(rows) == 1
        assert rows[0][COLUMN_CONTROLLER_NAME] == 'repair_a'
        assert rows[0]['diagnostic'] == RUN_CALIB_ECE
        assert int(rows[0]['n_valid_tasks']) == 3
        assert float(rows[0]['pearson_r']) == pytest.approx(1.0)
        assert float(rows[0]['spearman_r']) == pytest.approx(1.0)
        assert float(rows[0]['r2']) == pytest.approx(1.0)
