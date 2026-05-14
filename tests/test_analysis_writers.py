"""
Tests for analysis writer modules.
"""

import csv
import json
import re
from pathlib import Path

import pytest

from regain.analysis.artifacts import ARTIFACT_RHO
from regain.analysis.curves import write_recoverability_curves
from regain.analysis.frontier import write_repairability_frontier_outputs
from regain.analysis.plotting import plot_analysis_outputs
from regain.analysis.predictive import write_predictive_correlations
from regain.constants import ANALYSIS_ACC_FINAL_AVG_CTRL
from regain.constants import ANALYSIS_RHO_AVG
from regain.constants import COLUMN_B
from regain.constants import COLUMN_CONTROLLER_MODEL_PARAM_COUNT
from regain.constants import COLUMN_CONTROLLER_NAME
from regain.constants import COLUMN_CONTROLLER_TYPE
from regain.constants import COLUMN_EXP_IDX
from regain.constants import COLUMN_EXPERIMENT_ID
from regain.constants import COLUMN_NUM_CLASSES
from regain.constants import COLUMN_REPAIR_BUDGET_FRACTION
from regain.constants import COLUMN_REPAIR_BUDGET_TOTAL
from regain.constants import COLUMN_REPAIR_SET_TOTAL
from regain.constants import COLUMN_REPAIR_SPLIT_FRACTION
from regain.constants import COLUMN_RUN_ID
from regain.constants import COLUMN_RUN_NAME
from regain.constants import COLUMN_SEED
from regain.constants import COLUMN_TASK_AGE
from regain.constants import RUN_ACC_FINAL_AVG_BASE
from regain.constants import RUN_ACC_FINAL_AVG_CTRL
from regain.constants import RUN_CALIB_ECE
from regain.constants import RUN_CALIB_MAX_ECE
from regain.constants import RUN_CALIB_NLL
from regain.constants import RUN_DIAG_AVG_CONF
from regain.constants import RUN_DIAG_AVG_ENTROPY
from regain.constants import RUN_DIAG_LOGIT_AVG_DRIFT
from regain.constants import RUN_DIAG_OUT_OF_TASK_RATE
from regain.constants import RUN_LATENCY_MS_PER_SAMPLE_BASE
from regain.constants import RUN_LATENCY_MS_PER_SAMPLE_CTRL
from regain.constants import RUN_LATENCY_MS_RATIO
from regain.constants import RUN_REPAIR_SECONDS
from regain.constants import RUN_RHO_AVG

_COLUMN_CONTROLLER_ID = 'controller_id'
_COLUMN_ACTION_REPAIR_BUDGET_FRACTION = 'action_repair_budget_fraction'
_COLUMN_ACTION_REPAIR_BUDGET_TOTAL = 'action_repair_budget_total'
_COLUMN_BACKBONE_NAME = 'backbone_name'
_COLUMN_IS_NO_OP_ACTION = 'is_no_op_action'
_COLUMN_ORACLE_MARGIN = 'oracle_margin_vs_best_static_controller'
_COLUMN_SCENARIO = 'scenario'
_COLUMN_STRATEGY_NAME = 'strategy_name'
_NO_OP_CONTROLLER_NAME = 'no-op'


def _read_rows(path: Path) -> list[dict[str, str]]:
    """
    Read one CSV file into a list of dict rows.

    Args:
        path: CSV path.

    Returns:
        list[dict[str, str]]: Parsed rows.
    """
    with path.open('r', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    """
    Read a JSONL file into a list of dictionary rows.

    Args:
        path: JSONL path.

    Returns:
        list[dict[str, object]]: Parsed rows.
    """
    rows: list[dict[str, object]] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            payload = line.strip()
            if not payload:
                continue
            rows.append(json.loads(payload))
    return rows


def _run_row(
    *,
    run_id: str,
    controller_name: str,
    controller_type: str = 'repair',
    experiment_id: str = 'exp_1',
    scenario: str = 'cifar100',
    backbone_name: str = 'vit_small',
    strategy_name: str = 'er',
    seed: int = 1,
    budget_fraction: float = 0.5,
    budget_total: int = 5,
    repair_set_total: int = 10,
    split_fraction: float = 0.2,
    num_classes: int = 4,
    param_count: int = 100,
    calib_max_ece: float = 0.3,
    latency_ratio: float | None = 1.2,
    repair_seconds: float | None = 10.0,
    replay_mem_size: int | None = None,
    replay_batch_size_mem: int | None = None,
) -> dict[str, object]:
    return {
        COLUMN_RUN_ID: run_id,
        COLUMN_EXPERIMENT_ID: experiment_id,
        COLUMN_RUN_NAME: run_id,
        _COLUMN_SCENARIO: scenario,
        'backbone_name': backbone_name,
        _COLUMN_STRATEGY_NAME: strategy_name,
        COLUMN_SEED: seed,
        COLUMN_CONTROLLER_NAME: controller_name,
        COLUMN_CONTROLLER_TYPE: controller_type,
        COLUMN_REPAIR_BUDGET_FRACTION: budget_fraction,
        COLUMN_REPAIR_BUDGET_TOTAL: budget_total,
        COLUMN_REPAIR_SET_TOTAL: repair_set_total,
        COLUMN_REPAIR_SPLIT_FRACTION: split_fraction,
        COLUMN_NUM_CLASSES: num_classes,
        COLUMN_B: budget_fraction,
        COLUMN_CONTROLLER_MODEL_PARAM_COUNT: param_count,
        'replay_mem_size': replay_mem_size,
        'replay_batch_size_mem': replay_batch_size_mem,
        RUN_RHO_AVG: 0.3,
        RUN_ACC_FINAL_AVG_CTRL: 0.7,
        RUN_ACC_FINAL_AVG_BASE: 0.6,
        RUN_CALIB_MAX_ECE: calib_max_ece,
        RUN_LATENCY_MS_PER_SAMPLE_BASE: 1.0,
        RUN_LATENCY_MS_PER_SAMPLE_CTRL: 1.2 if latency_ratio is not None else None,
        RUN_LATENCY_MS_RATIO: latency_ratio,
        RUN_REPAIR_SECONDS: repair_seconds,
    }


def _experience_row(
    *,
    run_id: str,
    controller_name: str,
    controller_type: str = 'repair',
    exp_idx: int,
    a_ref: float | int,
    a_post: float | int,
    a_ctrl: float | int | None,
    seed: int = 1,
    budget_fraction: float = 0.5,
    budget_total: int = 5,
    repair_set_total: int = 10,
    split_fraction: float = 0.2,
    num_classes: int = 4,
    param_count: int = 100,
    task_age: int | None = None,
    calib_ece: float = 0.05,
    calib_nll: float = 0.2,
    drift: float = 0.1,
) -> dict[str, object]:
    return {
        COLUMN_RUN_ID: run_id,
        COLUMN_SEED: seed,
        COLUMN_CONTROLLER_NAME: controller_name,
        COLUMN_CONTROLLER_TYPE: controller_type,
        COLUMN_REPAIR_BUDGET_FRACTION: budget_fraction,
        COLUMN_REPAIR_BUDGET_TOTAL: budget_total,
        COLUMN_REPAIR_SET_TOTAL: repair_set_total,
        COLUMN_REPAIR_SPLIT_FRACTION: split_fraction,
        COLUMN_NUM_CLASSES: num_classes,
        COLUMN_B: budget_fraction,
        COLUMN_CONTROLLER_MODEL_PARAM_COUNT: param_count,
        COLUMN_EXP_IDX: exp_idx,
        COLUMN_TASK_AGE: task_age if task_age is not None else exp_idx,
        'acc.exp.base': a_ref,
        'acc.final.base': a_post,
        'acc.final.ctrl': a_ctrl,
        RUN_CALIB_ECE: calib_ece,
        RUN_CALIB_NLL: calib_nll,
        RUN_DIAG_OUT_OF_TASK_RATE: 0.2,
        RUN_DIAG_AVG_CONF: 0.7,
        RUN_DIAG_AVG_ENTROPY: 0.4,
        RUN_DIAG_LOGIT_AVG_DRIFT: drift,
    }


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


class TestWriteRepairabilityFrontierOutputs:
    def test_writes_repair_prefixed_artifacts_and_omits_legacy_files(self, tmp_path: Path) -> None:
        runs_table = [
            _run_row(run_id='run_a', controller_name='repair_a'),
            _run_row(
                run_id='run_b',
                controller_name='repair_b',
                param_count=200,
                latency_ratio=1.5,
                repair_seconds=15.0,
            ),
            _run_row(run_id='run_none', controller_name='none'),
            _run_row(
                run_id='run_prev',
                controller_name='prevent_a',
                controller_type='prevention',
            ),
        ]
        experiences_table = [
            _experience_row(run_id='run_a', controller_name='repair_a', exp_idx=0, a_ref=0.9, a_post=0.6, a_ctrl=0.8),
            _experience_row(run_id='run_a', controller_name='repair_a', exp_idx=1, a_ref=0.8, a_post=0.75, a_ctrl=0.7),
            _experience_row(run_id='run_b', controller_name='repair_b', exp_idx=0, a_ref=0.9, a_post=0.6, a_ctrl=0.9),
            _experience_row(run_id='run_b', controller_name='repair_b', exp_idx=1, a_ref=0.8, a_post=0.75, a_ctrl=0.78),
            _experience_row(run_id='run_none', controller_name='none', exp_idx=0, a_ref=0.9, a_post=0.6, a_ctrl=None),
            _experience_row(
                run_id='run_prev',
                controller_name='prevent_a',
                controller_type='prevention',
                exp_idx=0,
                a_ref=0.9,
                a_post=0.6,
                a_ctrl=0.88,
            ),
        ]

        paths = write_repairability_frontier_outputs(
            runs_table=runs_table,
            experiences_table=experiences_table,
            out_dir=tmp_path,
        )

        assert paths['repair_outcomes'].exists()
        assert paths['candidates'].exists()
        assert paths['pareto'].exists()
        assert paths['impact'].exists()
        assert paths['selection'].exists()
        assert paths['manifest'].exists()
        assert paths['candidates'].name == 'candidates.csv'
        assert paths['pareto'].name == 'pareto.csv'
        assert paths['impact'].name == 'impact.csv'
        assert paths['selection'].name == 'selection.csv'
        assert {path.name for path in (tmp_path / 'frontier').iterdir()} == {
            'candidates.csv',
            'pareto.csv',
            'impact.csv',
            'selection.csv',
            'manifest.json',
        }
        assert not (tmp_path / 'frontier' / 'frontier_points.csv').exists()
        assert not (tmp_path / 'frontier' / 'frontier_pareto.csv').exists()

        repair_outcomes = _read_jsonl(paths['repair_outcomes'])
        assert len(repair_outcomes) == 6
        assert {row[COLUMN_CONTROLLER_NAME] for row in repair_outcomes} == {
            'repair_a',
            'repair_b',
            _NO_OP_CONTROLLER_NAME,
        }
        assert {row[COLUMN_CONTROLLER_TYPE] for row in repair_outcomes} == {'repair', 'none'}

    def test_rejects_percentage_scale_accuracy_values(self, tmp_path: Path) -> None:
        runs_table = [_run_row(run_id='run_a', controller_name='repair_a')]
        experiences_table = [
            _experience_row(run_id='run_a', controller_name='repair_a', exp_idx=0, a_ref=90, a_post=60, a_ctrl=75),
            _experience_row(
                run_id='run_a',
                controller_name='repair_a',
                exp_idx=1,
                a_ref=0.75,
                a_post=0.75,
                a_ctrl=0.80,
            ),
            _experience_row(
                run_id='run_a',
                controller_name='repair_a',
                exp_idx=2,
                a_ref=0.80,
                a_post=0.75,
                a_ctrl=0.70,
            ),
        ]

        paths = write_repairability_frontier_outputs(
            runs_table=runs_table,
            experiences_table=experiences_table,
            out_dir=tmp_path,
        )

        repair_outcomes = _read_jsonl(paths['repair_outcomes'])
        by_exp_idx = {
            int(row[COLUMN_EXP_IDX]): row
            for row in repair_outcomes
            if row[COLUMN_CONTROLLER_NAME] == 'repair_a'
        }
        assert sorted(by_exp_idx) == [1, 2]
        assert by_exp_idx[1]['rho'] is None
        assert by_exp_idx[1]['rho_valid'] is False
        assert by_exp_idx[2]['harmed'] is True
        assert by_exp_idx[2]['harm_magnitude'] == pytest.approx(0.05)
        assert by_exp_idx[2]['helped'] is False
        assert by_exp_idx[1]['source_stage'] == 'collect'

        with paths['manifest'].open('r', encoding='utf-8') as f:
            manifest = json.load(f)
        assert manifest['normalization']['suspicious_value_count'] == 3
        suspicious_rows = manifest['normalization']['suspicious_values']
        assert len(suspicious_rows) == 3
        assert {
            (row[COLUMN_RUN_ID], row[COLUMN_EXP_IDX], row['field'], row['raw_value'])
            for row in suspicious_rows
        } == {
            ('run_a', 0, 'A_ref', 90),
            ('run_a', 0, 'A_post', 60),
            ('run_a', 0, 'A_ctrl', 75),
        }

    def test_emits_no_op_rows_once_per_task_and_budget(self, tmp_path: Path) -> None:
        runs_table = [
            _run_row(run_id='run_a', controller_name='repair_a'),
            _run_row(run_id='run_b', controller_name='repair_b', latency_ratio=1.4, repair_seconds=12.0),
        ]
        experiences_table = [
            _experience_row(
                run_id='run_a',
                controller_name='repair_a',
                exp_idx=0,
                a_ref=0.90,
                a_post=0.60,
                a_ctrl=0.82,
            ),
            _experience_row(
                run_id='run_a',
                controller_name='repair_a',
                exp_idx=1,
                a_ref=0.80,
                a_post=0.70,
                a_ctrl=0.78,
            ),
            _experience_row(
                run_id='run_b',
                controller_name='repair_b',
                exp_idx=0,
                a_ref=0.90,
                a_post=0.60,
                a_ctrl=0.84,
            ),
            _experience_row(
                run_id='run_b',
                controller_name='repair_b',
                exp_idx=1,
                a_ref=0.80,
                a_post=0.70,
                a_ctrl=0.76,
            ),
        ]

        paths = write_repairability_frontier_outputs(
            runs_table=runs_table,
            experiences_table=experiences_table,
            out_dir=tmp_path,
        )

        repair_outcomes = _read_jsonl(paths['repair_outcomes'])
        repair_a_rows = [row for row in repair_outcomes if row[COLUMN_CONTROLLER_NAME] == 'repair_a']
        repair_b_rows = [row for row in repair_outcomes if row[COLUMN_CONTROLLER_NAME] == 'repair_b']
        no_op_rows = [row for row in repair_outcomes if row[COLUMN_CONTROLLER_NAME] == _NO_OP_CONTROLLER_NAME]

        assert len(repair_a_rows) == 2
        assert len(repair_b_rows) == 2
        assert len(no_op_rows) == 2
        assert {int(row[COLUMN_EXP_IDX]) for row in no_op_rows} == {0, 1}
        assert all(row[_COLUMN_IS_NO_OP_ACTION] is True for row in no_op_rows)
        assert all(row['source_stage'] == 'no_op' for row in no_op_rows)
        assert all(bool(re.fullmatch(r'[0-9a-f]{32}', str(row[COLUMN_RUN_ID]))) for row in no_op_rows)
        assert len({str(row[COLUMN_RUN_ID]) for row in no_op_rows}) == 1
        assert len({str(row[COLUMN_RUN_NAME]) for row in no_op_rows}) == 1
        assert all(
            row[COLUMN_RUN_NAME] == 'no_op-cifar100-vit_small-er-budget_50-seed_1'
            for row in no_op_rows
        )

        frontier_rows = _read_rows(paths['candidates'])
        no_op_frontier_row = next(row for row in frontier_rows if row[COLUMN_CONTROLLER_NAME] == _NO_OP_CONTROLLER_NAME)
        assert int(no_op_frontier_row['num_runs']) == 1

    def test_no_op_metrics_are_exact(self, tmp_path: Path) -> None:
        runs_table = [_run_row(run_id='run_a', controller_name='repair_a')]
        experiences_table = [
            _experience_row(
                run_id='run_a',
                controller_name='repair_a',
                exp_idx=0,
                a_ref=0.90,
                a_post=0.60,
                a_ctrl=0.78,
            ),
            _experience_row(
                run_id='run_a',
                controller_name='repair_a',
                exp_idx=1,
                a_ref=0.75,
                a_post=0.75,
                a_ctrl=0.80,
            ),
        ]

        paths = write_repairability_frontier_outputs(
            runs_table=runs_table,
            experiences_table=experiences_table,
            out_dir=tmp_path,
        )

        repair_outcomes = _read_jsonl(paths['repair_outcomes'])
        no_op_by_exp_idx = {
            int(row[COLUMN_EXP_IDX]): row
            for row in repair_outcomes
            if row[COLUMN_CONTROLLER_NAME] == _NO_OP_CONTROLLER_NAME
        }

        assert no_op_by_exp_idx[0]['A_ctrl'] == pytest.approx(0.60)
        assert no_op_by_exp_idx[0]['forgetting'] == pytest.approx(0.30)
        assert no_op_by_exp_idx[0]['absolute_recovery'] == pytest.approx(0.0)
        assert no_op_by_exp_idx[0]['residual_forgetting'] == pytest.approx(0.30)
        assert no_op_by_exp_idx[0]['rho'] == pytest.approx(0.0)
        assert no_op_by_exp_idx[0]['rho_valid'] is True
        assert no_op_by_exp_idx[0]['task_delta'] == pytest.approx(0.0)
        assert no_op_by_exp_idx[0]['helped'] is False
        assert no_op_by_exp_idx[0]['harmed'] is False
        assert no_op_by_exp_idx[0]['harm_magnitude'] == pytest.approx(0.0)
        assert no_op_by_exp_idx[0]['source_stage'] == 'no_op'
        assert no_op_by_exp_idx[0][_COLUMN_IS_NO_OP_ACTION] is True

        assert no_op_by_exp_idx[1]['A_ctrl'] == pytest.approx(0.75)
        assert no_op_by_exp_idx[1]['rho'] is None
        assert no_op_by_exp_idx[1]['rho_valid'] is False

    def test_no_op_wins_oracle_when_repairs_are_harmful(self, tmp_path: Path) -> None:
        runs_table = [
            _run_row(run_id='run_a', controller_name='repair_a'),
            _run_row(run_id='run_b', controller_name='repair_b', latency_ratio=1.5, repair_seconds=15.0),
        ]
        experiences_table = [
            _experience_row(
                run_id='run_a',
                controller_name='repair_a',
                exp_idx=0,
                a_ref=0.90,
                a_post=0.60,
                a_ctrl=0.55,
            ),
            _experience_row(
                run_id='run_a',
                controller_name='repair_a',
                exp_idx=1,
                a_ref=0.80,
                a_post=0.75,
                a_ctrl=0.70,
            ),
            _experience_row(
                run_id='run_b',
                controller_name='repair_b',
                exp_idx=0,
                a_ref=0.90,
                a_post=0.60,
                a_ctrl=0.50,
            ),
            _experience_row(
                run_id='run_b',
                controller_name='repair_b',
                exp_idx=1,
                a_ref=0.80,
                a_post=0.75,
                a_ctrl=0.65,
            ),
        ]

        paths = write_repairability_frontier_outputs(
            runs_table=runs_table,
            experiences_table=experiences_table,
            out_dir=tmp_path,
        )

        selection_rows = _read_rows(paths['selection'])
        assert len(selection_rows) == 1
        selection_row = selection_rows[0]

        assert float(selection_row['utility_primary__no_op']) == pytest.approx(0.0)
        assert float(selection_row['utility_conservative__no_op']) == pytest.approx(0.0)
        assert selection_row['best_controller_by_utility_primary'] == 'no_op'
        assert selection_row['best_controller_by_utility_conservative'] == 'no_op'
        assert selection_row['best_controller_by_utility_cost_aware'] == 'no_op'

    def test_no_op_has_zero_action_cost_and_zero_utility(self, tmp_path: Path) -> None:
        runs_table = [
            _run_row(
                run_id='run_a',
                controller_name='repair_a',
                budget_fraction=0.4,
                budget_total=4,
                param_count=150,
                latency_ratio=1.8,
                repair_seconds=20.0,
            )
        ]
        experiences_table = [
            _experience_row(
                run_id='run_a',
                controller_name='repair_a',
                exp_idx=0,
                a_ref=0.90,
                a_post=0.60,
                a_ctrl=0.80,
            ),
            _experience_row(
                run_id='run_a',
                controller_name='repair_a',
                exp_idx=1,
                a_ref=0.85,
                a_post=0.70,
                a_ctrl=0.82,
            ),
        ]

        paths = write_repairability_frontier_outputs(
            runs_table=runs_table,
            experiences_table=experiences_table,
            out_dir=tmp_path,
        )

        frontier_rows = _read_rows(paths['candidates'])
        no_op_row = next(row for row in frontier_rows if row[COLUMN_CONTROLLER_NAME] == _NO_OP_CONTROLLER_NAME)

        assert float(no_op_row[_COLUMN_ACTION_REPAIR_BUDGET_FRACTION]) == pytest.approx(0.0)
        assert float(no_op_row[_COLUMN_ACTION_REPAIR_BUDGET_TOTAL]) == pytest.approx(0.0)
        assert float(no_op_row[COLUMN_CONTROLLER_MODEL_PARAM_COUNT]) == pytest.approx(0.0)
        assert float(no_op_row[RUN_LATENCY_MS_RATIO]) == pytest.approx(1.0)
        assert float(no_op_row[RUN_REPAIR_SECONDS]) == pytest.approx(0.0)
        assert float(no_op_row['utility_primary']) == pytest.approx(0.0)
        assert float(no_op_row['utility_conservative']) == pytest.approx(0.0)
        assert float(no_op_row['utility_cost_aware']) == pytest.approx(0.0)
        assert no_op_row[_COLUMN_IS_NO_OP_ACTION] == 'True'

    def test_deduplicates_no_op_rows_across_multiple_repair_controllers(self, tmp_path: Path) -> None:
        runs_table = [
            _run_row(run_id='run_a', controller_name='repair_a'),
            _run_row(run_id='run_b', controller_name='repair_b'),
            _run_row(run_id='run_c', controller_name='repair_c'),
        ]
        experiences_table = [
            _experience_row(
                run_id='run_a',
                controller_name='repair_a',
                exp_idx=0,
                a_ref=0.90,
                a_post=0.60,
                a_ctrl=0.81,
            ),
            _experience_row(
                run_id='run_b',
                controller_name='repair_b',
                exp_idx=0,
                a_ref=0.90,
                a_post=0.60,
                a_ctrl=0.82,
            ),
            _experience_row(
                run_id='run_c',
                controller_name='repair_c',
                exp_idx=0,
                a_ref=0.90,
                a_post=0.60,
                a_ctrl=0.83,
            ),
        ]

        paths = write_repairability_frontier_outputs(
            runs_table=runs_table,
            experiences_table=experiences_table,
            out_dir=tmp_path,
        )

        repair_outcomes = _read_jsonl(paths['repair_outcomes'])
        no_op_rows = [row for row in repair_outcomes if row[COLUMN_CONTROLLER_NAME] == _NO_OP_CONTROLLER_NAME]
        assert len(no_op_rows) == 1
        assert int(no_op_rows[0][COLUMN_EXP_IDX]) == 0

    def test_no_op_run_ids_are_deterministic_across_reruns(self, tmp_path: Path) -> None:
        runs_table = [
            _run_row(run_id='run_a', controller_name='repair_a'),
            _run_row(run_id='run_b', controller_name='repair_b'),
        ]
        experiences_table = [
            _experience_row(
                run_id='run_a',
                controller_name='repair_a',
                exp_idx=0,
                a_ref=0.90,
                a_post=0.60,
                a_ctrl=0.82,
            ),
            _experience_row(
                run_id='run_a',
                controller_name='repair_a',
                exp_idx=1,
                a_ref=0.80,
                a_post=0.70,
                a_ctrl=0.78,
            ),
            _experience_row(
                run_id='run_b',
                controller_name='repair_b',
                exp_idx=0,
                a_ref=0.90,
                a_post=0.60,
                a_ctrl=0.84,
            ),
            _experience_row(
                run_id='run_b',
                controller_name='repair_b',
                exp_idx=1,
                a_ref=0.80,
                a_post=0.70,
                a_ctrl=0.76,
            ),
        ]

        first_paths = write_repairability_frontier_outputs(
            runs_table=runs_table,
            experiences_table=experiences_table,
            out_dir=tmp_path / 'first',
        )
        second_paths = write_repairability_frontier_outputs(
            runs_table=runs_table,
            experiences_table=experiences_table,
            out_dir=tmp_path / 'second',
        )

        first_no_op_rows = [
            row
            for row in _read_jsonl(first_paths['repair_outcomes'])
            if row[COLUMN_CONTROLLER_NAME] == _NO_OP_CONTROLLER_NAME
        ]
        second_no_op_rows = [
            row
            for row in _read_jsonl(second_paths['repair_outcomes'])
            if row[COLUMN_CONTROLLER_NAME] == _NO_OP_CONTROLLER_NAME
        ]

        assert [row[COLUMN_RUN_ID] for row in first_no_op_rows] == [row[COLUMN_RUN_ID] for row in second_no_op_rows]

    def test_no_op_run_identity_includes_backbone_name(self, tmp_path: Path) -> None:
        runs_table = [
            _run_row(run_id='run_a', controller_name='repair_a', backbone_name='vit_small'),
            _run_row(run_id='run_b', controller_name='repair_b', backbone_name='resnet18'),
        ]
        experiences_table = [
            _experience_row(
                run_id='run_a',
                controller_name='repair_a',
                exp_idx=0,
                a_ref=0.90,
                a_post=0.60,
                a_ctrl=0.82,
            ),
            _experience_row(
                run_id='run_b',
                controller_name='repair_b',
                exp_idx=0,
                a_ref=0.90,
                a_post=0.60,
                a_ctrl=0.84,
            ),
        ]

        paths = write_repairability_frontier_outputs(
            runs_table=runs_table,
            experiences_table=experiences_table,
            out_dir=tmp_path,
        )

        no_op_rows = [
            row
            for row in _read_jsonl(paths['repair_outcomes'])
            if row[COLUMN_CONTROLLER_NAME] == _NO_OP_CONTROLLER_NAME
        ]

        assert len(no_op_rows) == 2
        assert len({str(row[COLUMN_RUN_ID]) for row in no_op_rows}) == 2
        assert {
            str(row[COLUMN_RUN_NAME])
            for row in no_op_rows
        } == {
            'no_op-cifar100-resnet18-er-budget_50-seed_1',
            'no_op-cifar100-vit_small-er-budget_50-seed_1',
        }

    def test_no_op_identity_includes_repair_budget_total(self, tmp_path: Path) -> None:
        run_low_total = _run_row(
            run_id='run_a',
            controller_name='repair_a',
            budget_fraction=0.5,
            budget_total=5,
        )
        run_high_total = _run_row(
            run_id='run_b',
            controller_name='repair_a',
            budget_fraction=0.5,
            budget_total=10,
        )
        run_low_total[COLUMN_B] = 0.5
        run_high_total[COLUMN_B] = 0.5
        runs_table = [run_low_total, run_high_total]
        experiences_table = [
            _experience_row(
                run_id='run_a',
                controller_name='repair_a',
                exp_idx=0,
                a_ref=0.90,
                a_post=0.60,
                a_ctrl=0.85,
            ),
            _experience_row(
                run_id='run_b',
                controller_name='repair_a',
                exp_idx=0,
                a_ref=0.90,
                a_post=0.60,
                a_ctrl=0.82,
            ),
        ]

        paths = write_repairability_frontier_outputs(
            runs_table=runs_table,
            experiences_table=experiences_table,
            out_dir=tmp_path,
        )

        repair_outcomes = _read_jsonl(paths['repair_outcomes'])
        no_op_rows = [
            row for row in repair_outcomes
            if row[COLUMN_CONTROLLER_NAME] == _NO_OP_CONTROLLER_NAME
        ]
        assert len(no_op_rows) == 2
        assert {
            int(float(row[COLUMN_REPAIR_BUDGET_TOTAL]))
            for row in no_op_rows
        } == {5, 10}
        assert len({str(row[COLUMN_RUN_ID]) for row in no_op_rows}) == 2

        frontier_rows = _read_rows(paths['candidates'])
        no_op_frontier_rows = [
            row for row in frontier_rows
            if row[COLUMN_CONTROLLER_NAME] == _NO_OP_CONTROLLER_NAME
        ]
        assert len(no_op_frontier_rows) == 2
        assert {
            int(float(row[COLUMN_REPAIR_BUDGET_TOTAL]))
            for row in no_op_frontier_rows
        } == {5, 10}

        selection_rows = _read_rows(paths['selection'])
        assert len(selection_rows) == 2
        assert {
            int(float(row[COLUMN_REPAIR_BUDGET_TOTAL]))
            for row in selection_rows
        } == {5, 10}
        for row in selection_rows:
            assert 'utility_primary__no_op' in row
            assert 'utility_conservative__no_op' in row
            assert 'utility_cost_aware__no_op' in row

    def test_records_manifest_warning_when_no_op_baselines_disagree(self, tmp_path: Path) -> None:
        runs_table = [
            _run_row(run_id='run_a', controller_name='repair_a'),
            _run_row(run_id='run_b', controller_name='repair_b'),
        ]
        experiences_table = [
            _experience_row(
                run_id='run_a',
                controller_name='repair_a',
                exp_idx=0,
                a_ref=0.90,
                a_post=0.60,
                a_ctrl=0.82,
            ),
            _experience_row(
                run_id='run_b',
                controller_name='repair_b',
                exp_idx=0,
                a_ref=0.88,
                a_post=0.58,
                a_ctrl=0.84,
            ),
        ]

        paths = write_repairability_frontier_outputs(
            runs_table=runs_table,
            experiences_table=experiences_table,
            out_dir=tmp_path,
        )

        repair_outcomes = _read_jsonl(paths['repair_outcomes'])
        no_op_rows = [row for row in repair_outcomes if row[COLUMN_CONTROLLER_NAME] == _NO_OP_CONTROLLER_NAME]
        assert len(no_op_rows) == 1

        with paths['manifest'].open('r', encoding='utf-8') as f:
            manifest = json.load(f)
        warning_codes = {warning['code'] for warning in manifest['warnings']}
        assert 'no_op_baseline_mismatch' in warning_codes

    def test_controller_id_collisions_pareto_and_selection_pivots(self, tmp_path: Path) -> None:
        runs_table = [
            _run_row(
                run_id='run_a',
                controller_name='Repair A',
                param_count=100,
                latency_ratio=1.1,
                repair_seconds=8.0,
            ),
            _run_row(
                run_id='run_b',
                controller_name='repair_a',
                param_count=120,
                latency_ratio=1.3,
                repair_seconds=12.0,
            ),
        ]
        experiences_table = [
            _experience_row(
                run_id='run_a',
                controller_name='Repair A',
                exp_idx=0,
                a_ref=0.90,
                a_post=0.60,
                a_ctrl=0.85,
            ),
            _experience_row(
                run_id='run_a',
                controller_name='Repair A',
                exp_idx=1,
                a_ref=0.80,
                a_post=0.70,
                a_ctrl=0.79,
            ),
            _experience_row(
                run_id='run_b',
                controller_name='repair_a',
                exp_idx=0,
                a_ref=0.90,
                a_post=0.60,
                a_ctrl=0.75,
            ),
            _experience_row(
                run_id='run_b',
                controller_name='repair_a',
                exp_idx=1,
                a_ref=0.80,
                a_post=0.70,
                a_ctrl=0.68,
            ),
        ]

        paths = write_repairability_frontier_outputs(
            runs_table=runs_table,
            experiences_table=experiences_table,
            out_dir=tmp_path,
        )

        frontier_rows = _read_rows(paths['candidates'])
        frontier_by_controller = {
            row[COLUMN_CONTROLLER_NAME]: row
            for row in frontier_rows
        }
        assert frontier_by_controller['Repair A']['controller_id'] == 'repair_a'
        assert frontier_by_controller['repair_a']['controller_id'] == 'repair_a__2'
        assert frontier_by_controller['Repair A']['is_pareto'] == 'True'
        assert frontier_by_controller['repair_a']['is_pareto'] == 'False'

        selection_rows = _read_rows(paths['selection'])
        assert len(selection_rows) == 1
        selection_row = selection_rows[0]
        assert 'utility_primary__repair_a' in selection_row
        assert 'utility_primary__repair_a__2' in selection_row
        assert selection_row['best_controller_by_utility_primary'] == 'repair_a'
        assert selection_row['best_controller_by_utility_conservative'] == 'repair_a'
        assert float(selection_row[_COLUMN_ORACLE_MARGIN]) == pytest.approx(0.0)

        with paths['manifest'].open('r', encoding='utf-8') as f:
            manifest = json.load(f)
        assert manifest['controller_id_collisions']

    def test_sets_is_pareto_null_when_retained_dimension_is_missing(self, tmp_path: Path) -> None:
        runs_table = [
            _run_row(run_id='run_a', controller_name='repair_a', latency_ratio=None),
            _run_row(run_id='run_b', controller_name='repair_b', latency_ratio=1.5),
        ]
        experiences_table = [
            _experience_row(run_id='run_a', controller_name='repair_a', exp_idx=0, a_ref=0.9, a_post=0.6, a_ctrl=0.8),
            _experience_row(run_id='run_b', controller_name='repair_b', exp_idx=0, a_ref=0.9, a_post=0.6, a_ctrl=0.8),
        ]

        paths = write_repairability_frontier_outputs(
            runs_table=runs_table,
            experiences_table=experiences_table,
            out_dir=tmp_path,
        )

        frontier_rows = _read_rows(paths['candidates'])
        frontier_by_controller = {
            row[COLUMN_CONTROLLER_NAME]: row
            for row in frontier_rows
        }
        assert frontier_by_controller['repair_a']['is_pareto'] == ''
        assert frontier_by_controller['repair_b']['is_pareto'] == 'True'

        with paths['manifest'].open('r', encoding='utf-8') as f:
            manifest = json.load(f)
        warning_codes = {warning['code'] for warning in manifest['warnings']}
        assert 'pareto_missing_dimension' in warning_codes
        pareto_warning = next(
            warning
            for warning in manifest['warnings']
            if warning['code'] == 'pareto_missing_dimension'
        )
        assert pareto_warning['context'][_COLUMN_BACKBONE_NAME] == 'vit_small'
        assert all(_COLUMN_BACKBONE_NAME in group for group in manifest['pareto_groups'])

    def test_frontier_does_not_merge_different_backbones(self, tmp_path: Path) -> None:
        runs_table = [
            _run_row(run_id='run_a', controller_name='repair_a', backbone_name='vit_small'),
            _run_row(run_id='run_b', controller_name='repair_a', backbone_name='resnet18'),
        ]
        experiences_table = [
            _experience_row(run_id='run_a', controller_name='repair_a', exp_idx=0, a_ref=0.90, a_post=0.60, a_ctrl=0.84),
            _experience_row(run_id='run_b', controller_name='repair_a', exp_idx=0, a_ref=0.90, a_post=0.60, a_ctrl=0.74),
        ]

        paths = write_repairability_frontier_outputs(
            runs_table=runs_table,
            experiences_table=experiences_table,
            out_dir=tmp_path,
        )

        frontier_rows = _read_rows(paths['candidates'])
        repair_rows = [row for row in frontier_rows if row[COLUMN_CONTROLLER_NAME] == 'repair_a']
        assert len(repair_rows) == 2
        assert _COLUMN_BACKBONE_NAME in frontier_rows[0]
        assert {row[_COLUMN_BACKBONE_NAME] for row in repair_rows} == {'vit_small', 'resnet18'}

        selection_rows = _read_rows(paths['selection'])
        assert len(selection_rows) == 2
        assert _COLUMN_BACKBONE_NAME in selection_rows[0]
        assert {row[_COLUMN_BACKBONE_NAME] for row in selection_rows} == {'vit_small', 'resnet18'}

    def test_selection_does_not_merge_same_b_with_different_budgets(self, tmp_path: Path) -> None:
        run_low_budget = _run_row(
            run_id='run_a',
            controller_name='repair_a',
            budget_fraction=0.25,
            budget_total=5,
        )
        run_high_budget = _run_row(
            run_id='run_b',
            controller_name='repair_a',
            budget_fraction=0.5,
            budget_total=10,
        )
        run_low_budget[COLUMN_B] = 0.5
        run_high_budget[COLUMN_B] = 0.5
        runs_table = [run_low_budget, run_high_budget]
        experiences_table = [
            _experience_row(run_id='run_a', controller_name='repair_a', exp_idx=0, a_ref=0.90, a_post=0.60, a_ctrl=0.85),
            _experience_row(run_id='run_b', controller_name='repair_a', exp_idx=0, a_ref=0.90, a_post=0.60, a_ctrl=0.82),
        ]

        paths = write_repairability_frontier_outputs(
            runs_table=runs_table,
            experiences_table=experiences_table,
            out_dir=tmp_path,
        )

        frontier_rows = _read_rows(paths['candidates'])
        repair_rows = [row for row in frontier_rows if row[COLUMN_CONTROLLER_NAME] == 'repair_a']
        assert len(repair_rows) == 2

        selection_rows = _read_rows(paths['selection'])
        assert len(selection_rows) == 2
        selection_keys = {
            (
                float(row[COLUMN_B]),
                float(row[COLUMN_REPAIR_BUDGET_FRACTION]),
                int(float(row[COLUMN_REPAIR_BUDGET_TOTAL])),
            )
            for row in selection_rows
        }
        assert selection_keys == {(0.5, 0.25, 5), (0.5, 0.5, 10)}
        assert all(float(row[_COLUMN_ORACLE_MARGIN]) == pytest.approx(0.0) for row in selection_rows)

    def test_pareto_is_scoped_by_backbone(self, tmp_path: Path) -> None:
        runs_table = [
            _run_row(
                run_id='run_a',
                controller_name='repair_a',
                backbone_name='vit_small',
                param_count=100,
                latency_ratio=1.1,
                repair_seconds=5.0,
            ),
            _run_row(
                run_id='run_b',
                controller_name='repair_b',
                backbone_name='resnet18',
                param_count=180,
                latency_ratio=1.5,
                repair_seconds=15.0,
            ),
        ]
        experiences_table = [
            _experience_row(run_id='run_a', controller_name='repair_a', exp_idx=0, a_ref=0.90, a_post=0.60, a_ctrl=0.88),
            _experience_row(run_id='run_b', controller_name='repair_b', exp_idx=0, a_ref=0.90, a_post=0.60, a_ctrl=0.78),
        ]

        paths = write_repairability_frontier_outputs(
            runs_table=runs_table,
            experiences_table=experiences_table,
            out_dir=tmp_path,
        )

        frontier_rows = _read_rows(paths['candidates'])
        repair_rows = [row for row in frontier_rows if row[COLUMN_CONTROLLER_NAME] in {'repair_a', 'repair_b'}]
        assert len(repair_rows) == 2
        assert all(row['is_pareto'] == 'True' for row in repair_rows)

        with paths['manifest'].open('r', encoding='utf-8') as f:
            manifest = json.load(f)
        pareto_backbones = {
            group[_COLUMN_BACKBONE_NAME]
            for group in manifest['pareto_groups']
        }
        assert pareto_backbones == {'vit_small', 'resnet18'}

    def test_best_static_lookup_is_scoped_by_full_setting(self, tmp_path: Path) -> None:
        run_low_budget = _run_row(
            run_id='run_a',
            controller_name='repair_a',
            budget_fraction=0.25,
            budget_total=5,
        )
        run_high_budget = _run_row(
            run_id='run_b',
            controller_name='repair_a',
            budget_fraction=0.5,
            budget_total=10,
        )
        run_low_budget[COLUMN_B] = 0.5
        run_high_budget[COLUMN_B] = 0.5
        runs_table = [run_low_budget, run_high_budget]
        experiences_table = [
            _experience_row(run_id='run_a', controller_name='repair_a', exp_idx=0, a_ref=0.90, a_post=0.60, a_ctrl=0.85),
            _experience_row(run_id='run_b', controller_name='repair_a', exp_idx=0, a_ref=0.90, a_post=0.60, a_ctrl=0.55),
        ]

        paths = write_repairability_frontier_outputs(
            runs_table=runs_table,
            experiences_table=experiences_table,
            out_dir=tmp_path,
        )

        selection_rows = _read_rows(paths['selection'])
        assert len(selection_rows) == 2
        rows_by_budget = {
            int(float(row[COLUMN_REPAIR_BUDGET_TOTAL])): row
            for row in selection_rows
        }

        low_budget_row = rows_by_budget[5]
        high_budget_row = rows_by_budget[10]
        assert low_budget_row['best_controller_by_utility_conservative'] == 'repair_a'
        assert low_budget_row['best_static_controller_by_utility_conservative'] == 'repair_a'
        assert float(low_budget_row[_COLUMN_ORACLE_MARGIN]) == pytest.approx(0.0)

        assert high_budget_row['best_controller_by_utility_conservative'] == 'no_op'
        assert high_budget_row['best_static_controller_by_utility_conservative'] == 'no_op'
        assert float(high_budget_row[_COLUMN_ORACLE_MARGIN]) == pytest.approx(0.0)

    def test_plot_grouping_and_filenames_do_not_collide_across_backbones(self, tmp_path: Path) -> None:
        plot_result = plot_analysis_outputs(
            frontier_rows=[
                {
                    _COLUMN_SCENARIO: 'cifar100',
                    _COLUMN_BACKBONE_NAME: 'vit_small',
                    _COLUMN_STRATEGY_NAME: 'er',
                    COLUMN_CONTROLLER_NAME: 'repair_a',
                    _COLUMN_CONTROLLER_ID: 'repair_a',
                    COLUMN_SEED: 1,
                    COLUMN_B: 0.5,
                    COLUMN_REPAIR_BUDGET_FRACTION: 0.5,
                    COLUMN_REPAIR_BUDGET_TOTAL: 5,
                    'mean_absolute_recovery': 0.20,
                    'mean_harmed_task_fraction': 0.10,
                    'utility_conservative': 0.15,
                },
                {
                    _COLUMN_SCENARIO: 'cifar100',
                    _COLUMN_BACKBONE_NAME: 'resnet18',
                    _COLUMN_STRATEGY_NAME: 'er',
                    COLUMN_CONTROLLER_NAME: 'repair_b',
                    _COLUMN_CONTROLLER_ID: 'repair_b',
                    COLUMN_SEED: 1,
                    COLUMN_B: 0.5,
                    COLUMN_REPAIR_BUDGET_FRACTION: 0.5,
                    COLUMN_REPAIR_BUDGET_TOTAL: 5,
                    'mean_absolute_recovery': 0.18,
                    'mean_harmed_task_fraction': 0.12,
                    'utility_conservative': 0.12,
                },
            ],
            impact_rows=[],
            mode='save',
            save_dir=tmp_path / 'plots',
        )

        filenames = {path.name for path in plot_result.saved_paths}
        assert 'recovery_vs_budget__cifar100__vit_small__er.png' in filenames
        assert 'recovery_vs_budget__cifar100__resnet18__er.png' in filenames
        assert 'harm_vs_budget__cifar100__vit_small.png' in filenames
        assert 'harm_vs_budget__cifar100__resnet18.png' in filenames
        assert not any(name.startswith('utility_delta__') for name in filenames)

    def test_plot_analysis_outputs_skips_utility_vs_cost_without_action_cost(self, tmp_path: Path) -> None:
        plot_result = plot_analysis_outputs(
            frontier_rows=[
                {
                    _COLUMN_SCENARIO: 'cifar100',
                    _COLUMN_BACKBONE_NAME: 'vit_small',
                    _COLUMN_STRATEGY_NAME: 'er',
                    COLUMN_CONTROLLER_NAME: 'repair_a',
                    _COLUMN_CONTROLLER_ID: 'repair_a',
                    COLUMN_SEED: 1,
                    COLUMN_B: 0.5,
                    COLUMN_REPAIR_BUDGET_FRACTION: 0.5,
                    COLUMN_REPAIR_BUDGET_TOTAL: 5,
                    'mean_absolute_recovery': 0.20,
                    'mean_harmed_task_fraction': 0.10,
                    'utility_conservative': 0.15,
                }
            ],
            impact_rows=[],
            mode='save',
            save_dir=tmp_path / 'plots',
        )

        saved_filenames = {path.name for path in plot_result.saved_paths}
        assert 'utility_vs_cost.png' not in saved_filenames
        assert any(skip['filename'] == 'utility_vs_cost.png' for skip in plot_result.skipped)

    def test_plot_analysis_outputs_has_no_manifest_side_effect(self, tmp_path: Path) -> None:
        analysis_out = tmp_path / 'analysis'
        frontier_dir = analysis_out / 'frontier'
        frontier_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = frontier_dir / 'manifest.json'
        manifest_payload = {
            'schema': {
                'name': 'regain.analysis.frontier.manifest',
                'version': 1,
            },
            'plots': {
                'saved': ['old_plot.png'],
                'skipped': [
                    {
                        'filename': 'old_missing.png',
                        'reason': 'old skip reason',
                        'context': {},
                    }
                ],
            },
        }
        manifest_path.write_text(json.dumps(manifest_payload), encoding='utf-8')

        plot_result = plot_analysis_outputs(
            analysis_out=analysis_out,
            frontier_rows=[
                {
                    _COLUMN_SCENARIO: 'cifar100',
                    _COLUMN_BACKBONE_NAME: 'vit_small',
                    _COLUMN_STRATEGY_NAME: 'er',
                    COLUMN_CONTROLLER_NAME: 'repair_a',
                    _COLUMN_CONTROLLER_ID: 'repair_a',
                    COLUMN_SEED: 1,
                    COLUMN_B: 0.5,
                    COLUMN_REPAIR_BUDGET_FRACTION: 0.5,
                    COLUMN_REPAIR_BUDGET_TOTAL: 5,
                    'mean_absolute_recovery': 0.20,
                    'mean_harmed_task_fraction': 0.10,
                    'utility_conservative': 0.15,
                }
            ],
            impact_rows=[],
            mode='save',
            save_dir=tmp_path / 'plots',
        )

        assert plot_result.saved_paths
        with manifest_path.open('r', encoding='utf-8') as f:
            after_payload = json.load(f)
        assert after_payload == manifest_payload

    def test_selection_outputs_include_replay_metadata_and_task_age_summaries(
        self,
        tmp_path: Path,
    ) -> None:
        runs_table = [
            _run_row(
                run_id='run_a',
                controller_name='repair_a',
                replay_mem_size=200,
                replay_batch_size_mem=64,
            )
        ]
        experiences_table = [
            _experience_row(
                run_id='run_a',
                controller_name='repair_a',
                exp_idx=0,
                a_ref=0.90,
                a_post=0.60,
                a_ctrl=0.82,
                task_age=2,
            ),
            _experience_row(
                run_id='run_a',
                controller_name='repair_a',
                exp_idx=1,
                a_ref=0.80,
                a_post=0.70,
                a_ctrl=0.78,
                task_age=1,
            ),
            _experience_row(
                run_id='run_a',
                controller_name='repair_a',
                exp_idx=2,
                a_ref=0.75,
                a_post=0.72,
                a_ctrl=0.74,
                task_age=0,
            ),
        ]

        paths = write_repairability_frontier_outputs(
            runs_table=runs_table,
            experiences_table=experiences_table,
            out_dir=tmp_path,
        )

        selection_rows = _read_rows(paths['selection'])
        assert selection_rows
        first = selection_rows[0]
        assert int(float(first['replay_mem_size'])) == 200
        assert int(float(first['replay_batch_size_mem'])) == 64
        assert float(first['task_age_mean']) == pytest.approx(1.0)
        assert float(first['task_age_min']) == pytest.approx(0.0)
        assert float(first['task_age_max']) == pytest.approx(2.0)
        assert float(first['oldest_task_forgetting']) == pytest.approx(0.30)
        assert float(first['newest_task_forgetting']) == pytest.approx(0.03)


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
