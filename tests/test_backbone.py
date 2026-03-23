"""
Tests for strict backbone baseline loading.
"""

from types import SimpleNamespace
from typing import Any

import pytest

import regain.experiments.backbone as backbone_module
import regain.experiments.logging as logging_module
from regain.analysis.artifacts import ARTIFACT_ACC_EXP_BASE
from regain.analysis.artifacts import ARTIFACT_ACC_FINAL_BASE
from regain.constants import RUN_CALIB_AECE
from regain.constants import RUN_CALIB_ECE
from regain.constants import RUN_CALIB_NLL
from regain.constants import RUN_DIAG_AVG_CONF
from regain.constants import RUN_DIAG_AVG_ENTROPY
from regain.constants import RUN_DIAG_LOGIT_AVG_DRIFT
from regain.constants import RUN_DIAG_OUT_OF_TASK_RATE
from regain.experiments.backbone import extract_backbone_kwargs_from_run
from regain.experiments.backbone import extract_backbone_training_config_from_run
from regain.experiments.config import BackboneConfig
from regain.experiments.config import EvaluationConfig
from regain.experiments.config import ExperimentConfig
from regain.experiments.config import OptimizerConfig
from regain.experiments.config import RepairConfig
from regain.experiments.config import StrategyConfig
from regain.experiments.config import TrainingConfig
from regain.experiments.config import LRSchedulerConfig
from regain.experiments.logging import log_run_params
from regain.experiments.backbone import load_backbone_analysis_baseline_from_run


def _make_run(
    *,
    metrics: dict[str, float],
    params: dict[str, str] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        info=SimpleNamespace(run_id='run_1'),
        data=SimpleNamespace(
            metrics=metrics,
            params=(params if params is not None else {}),
        ),
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

    def test_rejects_nan_in_required_diagnostic_vector(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run = _make_run(metrics={})
        artifact_payload: dict[str, Any] = {
            ARTIFACT_ACC_EXP_BASE: [0.80],
            ARTIFACT_ACC_FINAL_BASE: [0.55],
            RUN_DIAG_OUT_OF_TASK_RATE: [0.20],
            RUN_DIAG_AVG_CONF: [float('nan')],
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

        with pytest.raises(RuntimeError) as exc_info:
            load_backbone_analysis_baseline_from_run(
                client=object(),
                run=run,
                expected_num_experiences=1,
            )
        error_msg = str(exc_info.value)
        assert RUN_DIAG_AVG_CONF in error_msg
        assert 'non-finite value at index 0' in error_msg

    def test_rejects_inf_in_required_diagnostic_vector(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run = _make_run(metrics={})
        artifact_payload: dict[str, Any] = {
            ARTIFACT_ACC_EXP_BASE: [0.80],
            ARTIFACT_ACC_FINAL_BASE: [0.55],
            RUN_DIAG_OUT_OF_TASK_RATE: [0.20],
            RUN_DIAG_AVG_CONF: [float('inf')],
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

        with pytest.raises(RuntimeError) as exc_info:
            load_backbone_analysis_baseline_from_run(
                client=object(),
                run=run,
                expected_num_experiences=1,
            )
        error_msg = str(exc_info.value)
        assert RUN_DIAG_AVG_CONF in error_msg
        assert 'non-finite value at index 0' in error_msg


class TestExtractBackboneKwargsFromRun:
    def test_extracts_non_training_backbone_params(self) -> None:
        run = _make_run(
            metrics={},
            params={
                'backbone.name': 'vit_small',
                'backbone.patch_size': '4',
                'backbone.image_size': '32',
                'backbone.dropout': '0.1',
                'backbone.training.num_epochs': '50',
                'backbone.training.strategy.name': 'replay',
                'backbone.source_experiment.id': '123',
            },
        )

        kwargs = extract_backbone_kwargs_from_run(run=run)

        assert kwargs == {
            'patch_size': 4,
            'image_size': 32,
            'dropout': 0.1,
        }


class TestExtractBackboneTrainingConfigFromRun:
    def test_extracts_optimizer_scheduler_and_grad_clip_params(self) -> None:
        run = _make_run(
            metrics={},
            params={
                'backbone.training.num_epochs': '100',
                'backbone.training.batch_size': '64',
                'backbone.training.grad_clip_max_norm': '1.0',
                'backbone.training.strategy.name': 'naive',
                'backbone.training.optimizer.name': 'adamw',
                'backbone.training.optimizer.lr': '0.0005',
                'backbone.training.optimizer.betas': '[0.9, 0.999]',
                'backbone.training.optimizer.eps': '1e-08',
                'backbone.training.optimizer.weight_decay': '0.0001',
                'backbone.training.lr_scheduler.name': 'warmup_cosine',
                'backbone.training.lr_scheduler.warmup_epochs': '5',
                'backbone.training.lr_scheduler.min_lr': '0.0',
            },
        )

        training = extract_backbone_training_config_from_run(run=run)

        assert training == TrainingConfig(
            num_epochs=100,
            strategy=StrategyConfig(name='naive', kwargs={}),
            optimizer=OptimizerConfig(
                name='adamw',
                kwargs={
                    'lr': 5e-4,
                    'betas': [0.9, 0.999],
                    'eps': 1e-8,
                    'weight_decay': 1e-4,
                },
            ),
            batch_size=64,
            lr_scheduler=LRSchedulerConfig(
                name='warmup_cosine',
                kwargs={
                    'warmup_epochs': 5,
                    'min_lr': 0.0,
                },
            ),
            grad_clip_max_norm=1.0,
        )


class TestBackboneTrainingLoggingRoundTrip:
    def test_round_trips_adamw_scheduler_and_grad_clip_through_logged_params(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, object] = {}

        def _log_params(params: dict[str, object]) -> None:
            captured.update(params)

        monkeypatch.setattr(logging_module.mlflow, 'log_params', _log_params)

        experiment_config = ExperimentConfig(
            experiment_name='unit_test_experiment',
            scenario='split_cifar100',
            num_experiences=2,
            backbone=BackboneConfig(
                name='vit_small',
                kwargs={
                    'image_size': 32,
                    'patch_size': 4,
                },
                training=TrainingConfig(
                    num_epochs=100,
                    strategy=StrategyConfig(name='naive', kwargs={}),
                    optimizer=OptimizerConfig(
                        name='adamw',
                        kwargs={
                            'lr': 5e-4,
                            'betas': [0.9, 0.999],
                            'eps': 1e-8,
                            'weight_decay': 1e-4,
                        },
                    ),
                    batch_size=64,
                    lr_scheduler=LRSchedulerConfig(
                        name='warmup_cosine',
                        kwargs={
                            'warmup_epochs': 5,
                            'min_lr': 0.0,
                        },
                    ),
                    grad_clip_max_norm=1.0,
                ),
            ),
            repair=RepairConfig(split_fraction=0.0),
            evaluation=EvaluationConfig(),
            runs=[],
        )

        log_run_params(
            experiment_config=experiment_config,
            run_config_payload={'name': 'backbone', 'controller': None},
            controller_name=None,
            deterministic_algorithms_enabled=False,
            optimizer_kwargs={
                'lr': 5e-4,
                'betas': [0.9, 0.999],
                'eps': 1e-8,
                'weight_decay': 1e-4,
            },
            include_backbone_params=True,
            num_classes=100,
        )

        run = _make_run(
            metrics={},
            params={key: str(value) for key, value in captured.items()},
        )
        training = extract_backbone_training_config_from_run(run=run)

        assert training.optimizer.kwargs['betas'] == [0.9, 0.999]
        assert training.lr_scheduler is not None
        assert training.lr_scheduler.name == 'warmup_cosine'
        assert training.grad_clip_max_norm == pytest.approx(1.0)
