"""
Tests for RegainEvaluationPlugin baseline-only calibration scalar policy.
"""

from typing import Any

import pytest
from torch import nn
from torch.utils.data import Dataset

# Ensure a stable import order for plugin module initialization.
import regain.experiments.orchestrator  # noqa: F401
from regain.avalanche_utils.plugins import CalibrationDiagnosticsPlugin
from regain.avalanche_utils.plugins import RegainEvaluationPlugin
from regain.avalanche_utils.plugins import RepairControllerPlugin
from regain.constants import RUN_CALIB_ECE
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
        budget_per_class=1,
        max_repair_samples_per_class=1,
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


class TestCalibrationDiagnosticsPluginState:
    def test_latest_max_ece_resets_when_eval_pass_has_no_metrics(self) -> None:
        plugin = CalibrationDiagnosticsPlugin(num_bins=10)
        plugin._latest_eval_metrics = {0: {RUN_CALIB_ECE: 0.99}}
        plugin._current_eval_metrics = {}

        plugin.after_eval(strategy=object())

        assert plugin.latest_max_ece() is None
