"""
Tests for strict backbone baseline loading.
"""

from types import SimpleNamespace
from typing import Any

import pytest

import regain.experiments.backbone as backbone_module
from regain.analysis.artifacts import ARTIFACT_ACC_EXP_BASE
from regain.analysis.artifacts import ARTIFACT_ACC_FINAL_BASE
from regain.constants import RUN_CALIB_AECE
from regain.constants import RUN_CALIB_ECE
from regain.constants import RUN_CALIB_NLL
from regain.constants import RUN_DIAG_AVG_CONF
from regain.constants import RUN_DIAG_AVG_ENTROPY
from regain.constants import RUN_DIAG_LOGIT_AVG_DRIFT
from regain.constants import RUN_DIAG_OUT_OF_TASK_RATE
from regain.experiments.backbone import load_backbone_analysis_baseline_from_run


def _make_run(*, metrics: dict[str, float]) -> SimpleNamespace:
    return SimpleNamespace(
        info=SimpleNamespace(run_id='run_1'),
        data=SimpleNamespace(metrics=metrics),
    )


class TestLoadBackboneAnalysisBaselineFromRun:
    def test_requires_analysis_artifact_even_when_metrics_have_baselines(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run = _make_run(
            metrics={
                'run.accuracy.exp.exp000.base': 0.80,
                'run.accuracy.final.exp000.base': 0.55,
            },
        )
        monkeypatch.setattr(
            backbone_module,
            'download_json_artifact',
            lambda **kwargs: None,
        )

        with pytest.raises(RuntimeError, match='analysis_artifacts.json'):
            load_backbone_analysis_baseline_from_run(
                client=object(),
                run=run,
                expected_num_experiences=1,
            )

    def test_requires_all_diagnostic_vectors_in_analysis_artifact(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run = _make_run(metrics={})
        monkeypatch.setattr(
            backbone_module,
            'download_json_artifact',
            lambda **kwargs: {
                ARTIFACT_ACC_EXP_BASE: [0.80],
                ARTIFACT_ACC_FINAL_BASE: [0.55],
            },
        )

        with pytest.raises(RuntimeError, match=RUN_DIAG_OUT_OF_TASK_RATE):
            load_backbone_analysis_baseline_from_run(
                client=object(),
                run=run,
                expected_num_experiences=1,
            )

    def test_loads_baselines_and_diagnostic_vectors_from_artifact(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run = _make_run(metrics={})
        artifact_payload: dict[str, Any] = {
            ARTIFACT_ACC_EXP_BASE: [0.80],
            ARTIFACT_ACC_FINAL_BASE: [0.55],
            RUN_DIAG_OUT_OF_TASK_RATE: [0.20],
            RUN_DIAG_AVG_CONF: [0.30],
            RUN_DIAG_AVG_ENTROPY: [0.40],
            RUN_CALIB_ECE: [0.10],
            RUN_CALIB_AECE: [0.11],
            RUN_CALIB_NLL: [0.12],
            RUN_DIAG_LOGIT_AVG_DRIFT: [0.13],
        }
        monkeypatch.setattr(
            backbone_module,
            'download_json_artifact',
            lambda **kwargs: artifact_payload,
        )

        baseline = load_backbone_analysis_baseline_from_run(
            client=object(),
            run=run,
            expected_num_experiences=1,
        )

        assert baseline[ARTIFACT_ACC_EXP_BASE] == pytest.approx([0.80])
        assert baseline[ARTIFACT_ACC_FINAL_BASE] == pytest.approx([0.55])
        assert baseline[RUN_DIAG_OUT_OF_TASK_RATE] == pytest.approx([0.20])
