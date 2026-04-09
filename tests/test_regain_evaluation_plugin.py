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
from regain.avalanche_utils.plugins import PredictionLoggingPlugin
from regain.avalanche_utils.plugins import RegainEvaluationPlugin
from regain.avalanche_utils.plugins import RepairControllerPlugin
from regain.analysis.artifacts import ARTIFACT_ACC_EXP_BASE
from regain.analysis.artifacts import ARTIFACT_ACC_FINAL_BASE
from regain.constants import MLFLOW_ARTIFACT_ANALYSIS_FILE
from regain.constants import RUN_ACC_FINAL_TEST
from regain.constants import RUN_ACC_FINAL_TEST_AVG_BASE
from regain.constants import RUN_ACC_FINAL_TRAIN
from regain.constants import RUN_ACC_FINAL_TRAIN_AVG_BASE
from regain.constants import RUN_ACC_REF_TEST
from regain.constants import RUN_ACC_REF_TRAIN
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


class _StubPredictionLoggingPlugin:
    def __init__(self, *, derived_ref_accuracy: float | None) -> None:
        self._derived_ref_accuracy = derived_ref_accuracy
        self.pop_calls: list[tuple[str, int]] = []

    def pop_derived_ref_test_accuracy(
        self,
        *,
        eval_tag: str,
        checkpoint_exp_idx: int,
    ) -> float | None:
        self.pop_calls.append((str(eval_tag), int(checkpoint_exp_idx)))
        value = self._derived_ref_accuracy
        self._derived_ref_accuracy = None
        return value


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
        plugin.benchmark = SimpleNamespace(test_stream=[object()], train_stream=[object()])
        plugin._log_analysis_metric = lambda **kwargs: None
        plugin._log_latency_overhead = lambda **kwargs: None
        plugin._run_eval_with_state = lambda *args, **kwargs: {'ignored': 0.0}
        plugin._evaluate_stream_accuracies = lambda **kwargs: [0.69]  # type: ignore[method-assign]
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
            ref_test_exp_idx: int | None = None,
            ref_seen_class_ids: list[int] | None = None,
            ref_use_backbone_logits: bool = False,
            ref_mask_value: float | None = None,
        ) -> dict[str, float]:
            del strategy
            captured['stream'] = list(stream)
            captured['mask_enabled'] = bool(mask_enabled)
            captured['log_namespace'] = str(log_namespace)
            captured['log_step'] = int(log_step)
            captured['eval_tag'] = str(eval_tag)
            captured['checkpoint_exp_idx'] = int(checkpoint_exp_idx)
            captured['ref_test_exp_idx'] = ref_test_exp_idx
            captured['ref_seen_class_ids'] = ref_seen_class_ids
            captured['ref_use_backbone_logits'] = bool(ref_use_backbone_logits)
            captured['ref_mask_value'] = ref_mask_value
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
        assert captured['ref_test_exp_idx'] is None
        assert captured['ref_seen_class_ids'] is None
        assert captured['ref_use_backbone_logits'] is False
        assert captured['ref_mask_value'] is None
        assert scalar_results == {'Top1_Acc_Stream/eval_phase/test_stream': 0.73}
        assert plugin.last_posthoc_scalar_results == scalar_results
        assert plugin.last_base_eval_results == {'Top1_Acc_Stream/eval_phase/test_stream': 0.73}
        assert plugin.last_ctrl_eval_results is None
        assert plugin.last_posthoc_exp_idx == 1

    def test_after_training_exp_logs_repair_posthoc_metrics_from_single_eval_pass(
        self,
    ) -> None:
        plugin = object.__new__(RegainEvaluationPlugin)
        plugin.controller_plugin = _make_repair_controller_plugin()
        plugin.prediction_logging_plugin = _StubPredictionLoggingPlugin(
            derived_ref_accuracy=0.81,
        )
        plugin.repair_after_experience = True
        plugin.num_epochs_per_experience = 50
        plugin.a_exp_base = []
        run_checkpoint_calls: list[tuple[int, bool]] = []
        train_eval_calls: list[int] = []
        logged_analysis_metrics: list[dict[str, object]] = []

        def _run_checkpoint_eval(
            *,
            strategy: object,
            checkpoint_exp_idx: int,
            log_step: int,
            derive_current_test_ref: bool = False,
        ) -> dict[str, float]:
            del strategy, log_step
            run_checkpoint_calls.append(
                (int(checkpoint_exp_idx), bool(derive_current_test_ref))
            )
            return {'Top1_Acc_Stream/eval_phase/test_stream': 0.73}

        plugin._run_checkpoint_eval = _run_checkpoint_eval  # type: ignore[method-assign]
        plugin._run_current_train_ref_eval = (
            lambda *, strategy, exp_idx: (
                train_eval_calls.append(int(exp_idx)),
                0.79,
            )[1]
        )
        plugin._run_current_test_ref_eval = lambda **kwargs: (_ for _ in ()).throw(
            AssertionError('Unexpected extra current-test reference evaluation.')
        )
        plugin._log_analysis_metric = lambda **kwargs: logged_analysis_metrics.append(kwargs)

        strategy = SimpleNamespace(
            experience=SimpleNamespace(
                current_experience=0,
            )
        )

        plugin.after_training_exp(strategy=strategy)

        assert run_checkpoint_calls == [(0, True)]
        assert plugin.prediction_logging_plugin.pop_calls == [('ctrl', 0)]
        assert train_eval_calls == [0]
        assert plugin.a_exp_base == pytest.approx([0.81])
        assert len(logged_analysis_metrics) == 2
        assert logged_analysis_metrics[0]['key'] == RUN_ACC_REF_TEST
        assert logged_analysis_metrics[0]['experience'] == 0
        assert logged_analysis_metrics[0]['variant'] == 'base'
        assert logged_analysis_metrics[0]['step'] == 50
        assert logged_analysis_metrics[0]['value'] == pytest.approx(0.81)
        assert logged_analysis_metrics[1]['key'] == RUN_ACC_REF_TRAIN
        assert logged_analysis_metrics[1]['experience'] == 0
        assert logged_analysis_metrics[1]['variant'] == 'base'
        assert logged_analysis_metrics[1]['step'] == 50
        assert logged_analysis_metrics[1]['value'] == pytest.approx(0.79)

    def test_after_training_exp_raises_when_derived_current_test_ref_is_missing(
        self,
    ) -> None:
        plugin = object.__new__(RegainEvaluationPlugin)
        plugin.controller_plugin = None
        plugin.prediction_logging_plugin = _StubPredictionLoggingPlugin(
            derived_ref_accuracy=None,
        )
        plugin.num_epochs_per_experience = 50
        plugin.a_exp_base = []
        run_checkpoint_calls: list[tuple[int, bool]] = []
        train_eval_calls: list[int] = []
        plugin._run_checkpoint_eval = (
            lambda *,
            strategy,
            checkpoint_exp_idx,
            log_step,
            derive_current_test_ref=False: (
                run_checkpoint_calls.append(
                    (int(checkpoint_exp_idx), bool(derive_current_test_ref))
                ),
                {'Top1_Acc_Stream/eval_phase/test_stream': 0.73},
            )[1]
        )
        plugin._run_current_train_ref_eval = (
            lambda *, strategy, exp_idx: (
                train_eval_calls.append(int(exp_idx)),
                0.79,
            )[1]
        )
        plugin._log_analysis_metric = lambda **kwargs: None

        strategy = SimpleNamespace(
            experience=SimpleNamespace(
                current_experience=1,
            )
        )

        with pytest.raises(RuntimeError, match='Missing derived current-test reference accuracy'):
            plugin.after_training_exp(strategy=strategy)

        assert run_checkpoint_calls == [(1, True)]
        assert plugin.prediction_logging_plugin.pop_calls == [('base', 1)]
        assert train_eval_calls == []
        assert plugin.a_exp_base == []

    def test_run_current_train_ref_eval_returns_ref_train_accuracy(
        self,
    ) -> None:
        plugin = object.__new__(RegainEvaluationPlugin)
        plugin.benchmark = SimpleNamespace(train_stream=['exp0', 'exp1'])
        plugin.controller_plugin = None
        captured_eval: dict[str, object] = {}

        def _run_eval_with_state(
            *,
            strategy: object,
            stream: list[object],
            mask_enabled: bool,
            eval_tag: str,
            checkpoint_exp_idx: int,
            capture_predictions: bool,
            capture_auxiliary_metrics: bool,
            controller_enabled: bool,
        ) -> dict[str, float]:
            del strategy
            captured_eval['stream'] = list(stream)
            captured_eval['mask_enabled'] = bool(mask_enabled)
            captured_eval['eval_tag'] = str(eval_tag)
            captured_eval['checkpoint_exp_idx'] = int(checkpoint_exp_idx)
            captured_eval['capture_predictions'] = bool(capture_predictions)
            captured_eval['capture_auxiliary_metrics'] = bool(capture_auxiliary_metrics)
            captured_eval['controller_enabled'] = bool(controller_enabled)
            return {
                'Top1_Acc_Exp/eval_phase/train_stream/Exp001': 0.77,
                'Loss_Exp/eval_phase/train_stream/Exp001': 0.23,
                'Top1_Acc_Stream/eval_phase/train_stream': 0.77,
                'Top1_Acc_Exp/eval_phase/test_stream/Exp001': 0.66,
            }

        plugin._run_eval_with_state = _run_eval_with_state  # type: ignore[method-assign]

        train_eval_accuracy = plugin._run_current_train_ref_eval(
            strategy=object(),
            exp_idx=1,
        )

        assert captured_eval == {
            'stream': ['exp1'],
            'mask_enabled': True,
            'eval_tag': 'ref',
            'checkpoint_exp_idx': 1,
            'capture_predictions': False,
            'capture_auxiliary_metrics': False,
            'controller_enabled': True,
        }
        assert train_eval_accuracy == pytest.approx(0.77)

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
        plugin.benchmark = SimpleNamespace(test_stream=[object()], train_stream=[object()])
        plugin._run_checkpoint_eval = lambda **kwargs: (_ for _ in ()).throw(
            AssertionError('Unexpected extra final evaluation pass.')
        )
        logged_analysis_metrics: list[dict[str, object]] = []
        plugin._log_analysis_metric = lambda **kwargs: logged_analysis_metrics.append(kwargs)
        plugin._log_latency_overhead = lambda **kwargs: None
        plugin._run_eval_with_state = lambda *args, **kwargs: {'ignored': 0.0}
        plugin._evaluate_stream_accuracies = lambda **kwargs: [0.69]  # type: ignore[method-assign]
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

        logged_keys = [entry['key'] for entry in logged_analysis_metrics]
        assert RUN_ACC_FINAL_TEST in logged_keys
        assert RUN_ACC_FINAL_TEST_AVG_BASE in logged_keys
        assert RUN_ACC_FINAL_TRAIN in logged_keys
        assert RUN_ACC_FINAL_TRAIN_AVG_BASE in logged_keys


class TestCalibrationDiagnosticsPluginState:
    def test_latest_max_ece_resets_when_eval_pass_has_no_metrics(self) -> None:
        plugin = CalibrationDiagnosticsPlugin(num_bins=10)
        plugin._latest_eval_metrics = {0: {RUN_CALIB_ECE: 0.99}}
        plugin._current_eval_metrics = {}

        plugin.after_eval(strategy=object())

        assert plugin.latest_max_ece() is None

    def test_train_probe_does_not_override_latest_eval_metrics(self) -> None:
        plugin = CalibrationDiagnosticsPlugin(num_bins=10)
        plugin._latest_eval_metrics = {0: {RUN_CALIB_ECE: 0.55}}

        strategy = SimpleNamespace(
            _regain_eval_tag='base',
            _regain_prediction_capture_context={
                'eval_tag': 'base',
                'checkpoint_exp_idx': 0,
                'capture_auxiliary_metrics': False,
            },
            _regain_metric_context=SimpleNamespace(log_step=50),
        )

        plugin.before_eval(strategy=strategy)
        plugin.after_eval(strategy=strategy)

        assert plugin.latest_max_ece() == pytest.approx(0.55)


class TestPredictionLoggingPluginCaptureContext:
    def test_capture_context_can_disable_prediction_artifacts(self) -> None:
        strategy = SimpleNamespace(
            _regain_prediction_capture_context={
                'eval_tag': 'base',
                'checkpoint_exp_idx': 0,
                'capture_predictions': False,
            }
        )

        capture_context = PredictionLoggingPlugin._coerce_capture_context(strategy)

        assert capture_context is None
