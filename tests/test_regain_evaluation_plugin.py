"""
Tests for RegainEvaluationPlugin baseline-only calibration scalar policy.
"""

from types import SimpleNamespace
from typing import Any

import mlflow
import pytest
from torch import nn
from torch.utils.data import Dataset

# Ensure a stable import order for plugin module initialization.
import regain.experiments.orchestrator  # noqa: F401
import regain.avalanche_utils.plugins as plugins_module
from regain.avalanche_utils.plugins import CalibrationDiagnosticsPlugin
from regain.avalanche_utils.plugins import RegainEvaluationPlugin
from regain.avalanche_utils.plugins import RepairControllerPlugin
from regain.analysis.artifacts import ARTIFACT_ACC_EXP_BASE
from regain.analysis.artifacts import ARTIFACT_ACC_FINAL_BASE
from regain.constants import MLFLOW_ARTIFACT_ANALYSIS_FILE
from regain.constants import RUN_CALIB_ECE
from regain.constants import RUN_CALIB_MAX_ECE
from regain.models.controllers import RepairController


class _DummyRepairController(RepairController):
    def fit_on_repair_data(
        self,
        *,
        model: nn.Module,
        repair_dataset: Dataset | None,
        new_classes: list[int],
        num_epochs: int,
        batch_size: int,
    ) -> None:
        del model, repair_dataset, new_classes, num_epochs, batch_size
        return

    def correct_outputs(
        self,
        *,
        outputs: Any,
        model: nn.Module | None = None,
        inputs: Any | None = None,
    ) -> Any:
        del model, inputs
        return outputs


class _StubCalibrationPlugin:
    def __init__(self, *, latest_max_ece: float | None) -> None:
        self._latest_max_ece = latest_max_ece

    def latest_max_ece(self) -> float | None:
        return self._latest_max_ece


def _make_repair_controller_plugin() -> RepairControllerPlugin:
    return RepairControllerPlugin(
        controller=_DummyRepairController(),
        fit_after_experience=False,
        repair_epochs=1,
        repair_batch_size=1,
        budget_fraction=1.0,
        seed=1,
    )


class TestRegainEvaluationPluginCalibrationScalarPolicy:
    def test_repair_run_uses_backbone_baseline_max_ece(self) -> None:
        plugin = object.__new__(RegainEvaluationPlugin)
        plugin.controller_plugin = _make_repair_controller_plugin()
        plugin.calibration_plugin = _StubCalibrationPlugin(latest_max_ece=0.91)
        plugin._backbone_diag_vectors = {
            RUN_CALIB_ECE: [0.10, None, 0.45, 0.30],
        }

        max_ece = plugin._calibration_max_ece_for_artifacts()

        assert max_ece == pytest.approx(0.45)

    def test_repair_run_raises_without_backbone_calibration_vector(self) -> None:
        plugin = object.__new__(RegainEvaluationPlugin)
        plugin.controller_plugin = _make_repair_controller_plugin()
        plugin.calibration_plugin = _StubCalibrationPlugin(latest_max_ece=0.91)
        plugin._backbone_diag_vectors = None

        with pytest.raises(RuntimeError, match='backbone calibration vectors'):
            plugin._calibration_max_ece_for_artifacts()

    def test_non_repair_run_uses_latest_eval_max_ece(self) -> None:
        plugin = object.__new__(RegainEvaluationPlugin)
        plugin.controller_plugin = None
        plugin.calibration_plugin = _StubCalibrationPlugin(latest_max_ece=0.33)
        plugin._backbone_diag_vectors = {
            RUN_CALIB_ECE: [0.10, 0.95, 0.90],
        }

        max_ece = plugin._calibration_max_ece_for_artifacts()

        assert max_ece == pytest.approx(0.33)


class TestRegainEvaluationPluginArtifactsLogging:
    def test_after_training_logs_full_analysis_artifacts_json(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plugin = object.__new__(RegainEvaluationPlugin)
        plugin._num_experiences = 1
        plugin.num_epochs_per_experience = 50
        plugin.last_posthoc_scalar_results = {}
        plugin.last_posthoc_exp_idx = 0
        plugin.controller_plugin = None
        plugin.calibration_plugin = _StubCalibrationPlugin(latest_max_ece=0.12)
        plugin.a_exp_base = [0.81]
        plugin.eps = 0.0
        plugin.last_base_eval_results = {'ignored': 0.0}
        plugin.last_ctrl_eval_results = None
        plugin.benchmark = SimpleNamespace(test_stream=[object()])
        plugin._log_analysis_metric = lambda **kwargs: None
        plugin._log_summary_metric = lambda **kwargs: None
        plugin._log_latency_overhead = lambda **kwargs: None
        plugin._run_posthoc_eval = lambda **kwargs: None
        plugin._run_eval_with_state = lambda *args, **kwargs: {'ignored': 0.0}
        plugin._diagnostic_vectors_for_artifacts = lambda: {RUN_CALIB_ECE: [0.12]}
        plugin._calibration_max_ece_for_artifacts = lambda: 0.12

        expected_artifacts = {
            ARTIFACT_ACC_EXP_BASE: [0.81],
            ARTIFACT_ACC_FINAL_BASE: [0.73],
            RUN_CALIB_ECE: [0.12],
            RUN_CALIB_MAX_ECE: 0.12,
        }
        logged: list[tuple[dict[str, object], str]] = []

        monkeypatch.setattr(
            plugins_module,
            'ordered_accuracies',
            lambda eval_results, num_experiences: [0.73],
        )
        monkeypatch.setattr(
            plugins_module,
            'build_analysis_artifacts',
            lambda **kwargs: expected_artifacts,
        )
        monkeypatch.setattr(
            mlflow,
            'log_dict',
            lambda artifacts, artifact_file: logged.append((artifacts, artifact_file)),
        )

        plugin.after_training(strategy=object())

        assert plugin.artifacts == expected_artifacts
        assert logged == [(expected_artifacts, MLFLOW_ARTIFACT_ANALYSIS_FILE)]


class TestCalibrationDiagnosticsPluginState:
    def test_latest_max_ece_resets_when_eval_pass_has_no_metrics(self) -> None:
        plugin = CalibrationDiagnosticsPlugin(num_bins=10)
        plugin._latest_eval_metrics = {0: {RUN_CALIB_ECE: 0.99}}
        plugin._current_eval_metrics = {}

        plugin.after_eval(strategy=object())

        assert plugin.latest_max_ece() is None
