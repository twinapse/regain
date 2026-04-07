"""
Tests for RegainEvaluationPlugin baseline-only calibration scalar policy.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlflow
import numpy as np
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

    def test_rejects_non_finite_backbone_nullable_vector_values(self) -> None:
        with pytest.raises(ValueError, match='non-finite value'):
            RegainEvaluationPlugin._coerce_required_nullable_backbone_vector(
                baseline={RUN_CALIB_ECE: [float('nan')]},
                key=RUN_CALIB_ECE,
                expected_len=1,
            )


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


class TestRegainEvaluationPluginCheckpointEval:
    def test_run_checkpoint_eval_uses_full_stream_and_caches_results(self) -> None:
        plugin = object.__new__(RegainEvaluationPlugin)
        plugin._num_experiences = 3
        plugin.benchmark = SimpleNamespace(test_stream=['exp0', 'exp1', 'exp2'])
        plugin.controller_plugin = None
        plugin.last_posthoc_scalar_results = None
        plugin.last_base_eval_results = None
        plugin.last_ctrl_eval_results = None
        plugin.last_posthoc_exp_idx = None
        captured: dict[str, object] = {}

        def _run_eval_with_logging(
            *,
            strategy: object,
            stream: list[object],
            mask_enabled: bool,
            log_namespace: str,
            log_step: int,
            eval_tag: str,
            checkpoint_exp_idx: int,
        ) -> dict[str, float]:
            del strategy
            captured['stream'] = list(stream)
            captured['mask_enabled'] = bool(mask_enabled)
            captured['log_namespace'] = str(log_namespace)
            captured['log_step'] = int(log_step)
            captured['eval_tag'] = str(eval_tag)
            captured['checkpoint_exp_idx'] = int(checkpoint_exp_idx)
            return {'Top1_Acc_Stream/eval_phase/test_stream': 0.73}

        plugin._run_eval_with_logging = _run_eval_with_logging  # type: ignore[method-assign]

        scalar_results = plugin._run_checkpoint_eval(
            strategy=object(),
            checkpoint_exp_idx=1,
            log_step=20,
        )

        assert captured['stream'] == ['exp0', 'exp1', 'exp2']
        assert captured['mask_enabled'] is False
        assert captured['log_namespace'] == 'run.eval'
        assert captured['log_step'] == 20
        assert captured['eval_tag'] == 'base'
        assert captured['checkpoint_exp_idx'] == 1
        assert scalar_results == {'Top1_Acc_Stream/eval_phase/test_stream': 0.73}
        assert plugin.last_posthoc_scalar_results == scalar_results
        assert plugin.last_base_eval_results == {'Top1_Acc_Stream/eval_phase/test_stream': 0.73}
        assert plugin.last_ctrl_eval_results is None
        assert plugin.last_posthoc_exp_idx == 1

    def test_acc_exp_base_from_prediction_artifact_masks_unseen_classes(
        self,
        tmp_path: Path,
    ) -> None:
        plugin = object.__new__(RegainEvaluationPlugin)
        plugin.controller_plugin = None
        plugin.prediction_logging_plugin = SimpleNamespace(
            artifact_root=tmp_path / 'predictions'
        )
        plugin._seen_class_ids_by_experience = [[0, 1]]
        artifact_path = plugin._prediction_artifact_path(
            checkpoint_exp_idx=0,
            test_exp_idx=0,
        )
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            artifact_path,
            logits=np.asarray(
                [
                    [0.0, 1.0, 9.0],
                    [2.0, 1.5, 8.0],
                ],
                dtype=np.float32,
            ),
            targets=np.asarray([1, 0], dtype=np.int32),
            class_ids=np.asarray([0, 1], dtype=np.int32),
        )

        acc_exp_base = plugin._acc_exp_base_from_prediction_artifact(exp_idx=0)

        assert acc_exp_base == pytest.approx(1.0)

    def test_after_training_exp_logs_repair_posthoc_metrics_from_single_eval_pass(self) -> None:
        plugin = object.__new__(RegainEvaluationPlugin)
        plugin.controller_plugin = _make_repair_controller_plugin()
        plugin.repair_after_experience = True
        plugin.num_epochs_per_experience = 50
        plugin._backbone_a_exp_base = [0.81]
        plugin.a_exp_base = []
        run_checkpoint_calls: list[int] = []
        mirrored_namespaces: list[str] = []

        def _run_checkpoint_eval(
            *,
            strategy: object,
            checkpoint_exp_idx: int,
            log_step: int,
        ) -> dict[str, float]:
            del strategy, log_step
            run_checkpoint_calls.append(int(checkpoint_exp_idx))
            return {'Top1_Acc_Stream/eval_phase/test_stream': 0.73}

        def _log_scalar_metrics_to_namespace(
            *,
            scalar_metrics: dict[str, float],
            namespace: str,
            step: int,
        ) -> None:
            del scalar_metrics, step
            mirrored_namespaces.append(str(namespace))

        plugin._run_checkpoint_eval = _run_checkpoint_eval  # type: ignore[method-assign]
        plugin._log_scalar_metrics_to_namespace = _log_scalar_metrics_to_namespace  # type: ignore[method-assign]
        plugin._log_analysis_metric = lambda **kwargs: None

        strategy = SimpleNamespace(
            experience=SimpleNamespace(
                current_experience=0,
            )
        )

        plugin.after_training_exp(strategy=strategy)

        assert run_checkpoint_calls == [0]
        assert mirrored_namespaces == ['run.exp000']
        assert plugin.a_exp_base == pytest.approx([0.81])

    def test_after_training_logs_run_final_metrics_from_cached_checkpoint(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plugin = object.__new__(RegainEvaluationPlugin)
        plugin._num_experiences = 1
        plugin.num_epochs_per_experience = 50
        plugin.last_posthoc_scalar_results = {'Top1_Acc_Stream/eval_phase/test_stream': 0.73}
        plugin.last_posthoc_exp_idx = 0
        plugin.controller_plugin = None
        plugin.calibration_plugin = _StubCalibrationPlugin(latest_max_ece=0.12)
        plugin.a_exp_base = [0.81]
        plugin.eps = 0.0
        plugin.last_base_eval_results = {'ignored': 0.0}
        plugin.last_ctrl_eval_results = None
        plugin.benchmark = SimpleNamespace(test_stream=[object()])
        plugin._run_checkpoint_eval = lambda **kwargs: (_ for _ in ()).throw(
            AssertionError('Unexpected extra final evaluation pass.')
        )
        mirrored_namespaces: list[str] = []
        plugin._log_scalar_metrics_to_namespace = (  # type: ignore[method-assign]
            lambda **kwargs: mirrored_namespaces.append(kwargs['namespace'])
        )
        plugin._log_analysis_metric = lambda **kwargs: None
        plugin._log_summary_metric = lambda **kwargs: None
        plugin._log_latency_overhead = lambda **kwargs: None
        plugin._run_eval_with_state = lambda *args, **kwargs: {'ignored': 0.0}
        plugin._diagnostic_vectors_for_artifacts = lambda: {RUN_CALIB_ECE: [0.12]}
        plugin._calibration_max_ece_for_artifacts = lambda: 0.12

        expected_artifacts = {
            ARTIFACT_ACC_EXP_BASE: [0.81],
            ARTIFACT_ACC_FINAL_BASE: [0.73],
            RUN_CALIB_ECE: [0.12],
            RUN_CALIB_MAX_ECE: 0.12,
        }

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
            lambda artifacts, artifact_file: None,
        )

        plugin.after_training(strategy=object())

        assert mirrored_namespaces == ['run.final']


class TestCalibrationDiagnosticsPluginState:
    def test_latest_max_ece_resets_when_eval_pass_has_no_metrics(self) -> None:
        plugin = CalibrationDiagnosticsPlugin(num_bins=10)
        plugin._latest_eval_metrics = {0: {RUN_CALIB_ECE: 0.99}}
        plugin._current_eval_metrics = {}

        plugin.after_eval(strategy=object())

        assert plugin.latest_max_ece() is None
