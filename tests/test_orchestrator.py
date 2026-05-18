"""
Tests for the experiment orchestrator.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import yaml

from regain.experiments.config import BackboneConfig
from regain.experiments.config import ControllerConfig
from regain.experiments.config import EvaluationConfig
from regain.experiments.config import ExperimentConfig
from regain.experiments.config import OptimizerConfig
from regain.experiments.config import RepairConfig
from regain.experiments.config import RunConfig
from regain.experiments.config import StrategyConfig
from regain.experiments.config import TrainingConfig
from regain.experiments.logging import log_run_params
import regain.experiments.logging as logging_module
import regain.experiments.orchestrator as orchestrator_module


class _FakeStrategy:
    """
    Fake training strategy stub.
    """

    def __init__(self) -> None:
        self.train_kwargs: dict[str, object] | None = None

    def train(self, *, experiences: object, **kwargs) -> None:
        self.train_kwargs = dict(kwargs)
        self.train_kwargs['experiences'] = experiences


class _FakeRegainEvaluationPlugin:
    """
    Fake RegainEvaluationPlugin stub.
    """

    def __init__(self, **kwargs) -> None:
        del kwargs
        self.last_posthoc_scalar_results = {
            'run.eval.acc.final.avg.base': 0.73,
        }


class _FakeRegainEvaluator:
    """
    Fake RegainEvaluator stub.
    """

    def __init__(self, **kwargs) -> None:
        del kwargs


class _FakeRepairController:
    """
    Fake repair controller stub.
    """
    pass


class _FakeRepairControllerPlugin:
    """
    Fake RepairControllerPlugin stub.
    """

    def __init__(self, controller: _FakeRepairController) -> None:
        self.controller = controller

    def initialize_parameters(self, model: nn.Module, dataset: object) -> None:
        del model, dataset


class _FakeSeenClassesObserver:
    """
    Fake SeenClassesObserver stub.
    """

    def __init__(self) -> None:
        self.seen_classes: set[int] = set()


class _FakePredictionRecorder:
    """
    Fake PredictionRecorder stub.
    """

    def __init__(self, *, artifact_root: Path, num_classes: int) -> None:
        del num_classes
        self.artifact_root = artifact_root

    def has_artifacts(self) -> bool:
        return False


class _FakeMlflowClient:
    """
    Fake MLflow client stub.
    """

    def __init__(self, experiment_name: str | None = None) -> None:
        self._experiment_name = experiment_name

    def get_experiment(self, experiment_id: str) -> SimpleNamespace | None:
        del experiment_id
        if self._experiment_name is None:
            return None
        return SimpleNamespace(name=self._experiment_name)


class _DummyContextManager:
    """
    Dummy context manager for testing.
    """

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return False


class TestTrainInvocation:
    """
    Tests for training invocation behavior.
    """

    def test_train_call_omits_eval_streams(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
    ) -> None:
        del tmp_path

        strategy = _FakeStrategy()

        monkeypatch.setattr(orchestrator_module, 'init_mlflow', lambda **kwargs: _DummyContextManager())
        monkeypatch.setattr(
            orchestrator_module,
            'log_fatal_error_context',
            lambda **kwargs: _DummyContextManager(),
        )
        monkeypatch.setattr(orchestrator_module, 'build_controller', lambda **kwargs: None)
        monkeypatch.setattr(
            orchestrator_module,
            'build_benchmark',
            lambda **kwargs: SimpleNamespace(
                n_classes=2,
                train_stream=[SimpleNamespace()],
                test_stream=[SimpleNamespace(dataset=[0, 1])],
            ),
        )
        monkeypatch.setattr(
            orchestrator_module,
            'make_training_evaluation_plugin',
            lambda **kwargs: object(),
        )
        monkeypatch.setattr(
            orchestrator_module,
            'RegainEvaluator',
            _FakeRegainEvaluator,
        )
        monkeypatch.setattr(orchestrator_module, 'RegainEvaluationPlugin', _FakeRegainEvaluationPlugin)
        monkeypatch.setattr(orchestrator_module, 'build_backbone', lambda **kwargs: nn.Linear(1, 2))
        monkeypatch.setattr(
            orchestrator_module,
            'build_optimizer',
            lambda **kwargs: (
                torch.optim.SGD(nn.Linear(1, 2).parameters(), lr=0.1),
                {
                    'lr': 0.1
                },
            ),
        )
        monkeypatch.setattr(orchestrator_module, 'make_strategy', lambda **kwargs: strategy)
        monkeypatch.setattr(orchestrator_module, 'log_run_params', lambda **kwargs: None)
        monkeypatch.setattr(orchestrator_module, 'log_dataset_indices', lambda **kwargs: None)
        monkeypatch.setattr(orchestrator_module.mlflow, 'log_param', lambda *args, **kwargs: None)
        monkeypatch.setattr(orchestrator_module.mlflow, 'log_artifacts', lambda *args, **kwargs: None)
        monkeypatch.setattr(orchestrator_module.mlflow, 'log_artifact', lambda *args, **kwargs: None)

        experiment_config = ExperimentConfig(
            experiment_name='unit_test_experiment',
            scenario='cifar100',
            num_experiences=1,
            backbone=BackboneConfig(
                name='resnet18',
                training=TrainingConfig(
                    num_epochs=1,
                    strategy=StrategyConfig(name='naive', kwargs={}),
                    optimizer=OptimizerConfig(name='sgd', kwargs={'lr': 0.1}),
                    batch_size=2,
                ),
            ),
            repair=RepairConfig(
                split_fraction=0.2,
                budget_fraction=1.0,
                fit_schedule='final_only',
                num_epochs=1,
                batch_size=2,
            ),
            evaluation=EvaluationConfig(batch_size=2, avalanche_schedule='final_only'),
            runs=[],
            dataset_path=None,
        )
        run_config = orchestrator_module._InternalRunConfig(
            name='backbone',
            controller=None,
        )

        orchestrator_module._train_and_evaluate_strategy(
            experiment_config=experiment_config,
            run_config=run_config,
        )

        assert strategy.train_kwargs is not None
        assert 'eval_streams' not in strategy.train_kwargs


def _make_backbone_run(*, params: dict[str, str], experiment_id: str = 'source_exp_id') -> SimpleNamespace:
    return SimpleNamespace(
        info=SimpleNamespace(
            run_id='backbone_run_id',
            experiment_id=experiment_id,
        ),
        data=SimpleNamespace(
            metrics={},
            params=params,
        ),
    )


def _make_backbone_run_params(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    captured: dict[str, object] = {}
    training_config = TrainingConfig(
        num_epochs=7,
        strategy=StrategyConfig(name='naive', kwargs={}),
        optimizer=OptimizerConfig(name='adamw', kwargs={'lr': 0.01}),
        batch_size=16,
    )
    experiment_config = ExperimentConfig(
        experiment_name='backbone_source_experiment',
        scenario='cifar100',
        num_experiences=1,
        backbone=BackboneConfig(
            name='resnet18',
            training=training_config,
        ),
        repair=RepairConfig(split_fraction=0.2),
        evaluation=EvaluationConfig(batch_size=4, avalanche_schedule='final_only'),
        runs=[],
    )

    monkeypatch.setattr(logging_module.mlflow, 'log_params', captured.update)

    log_run_params(
        experiment_config=experiment_config,
        run_config_payload={
            'name': 'backbone',
            'controller': None
        },
        controller_name=None,
        deterministic_algorithms_enabled=False,
        optimizer_kwargs={'lr': 0.01},
        include_backbone_params=True,
        num_classes=2,
    )
    return {key: str(value) for key, value in captured.items()}


def _make_repair_experiment_config(*, backbone: BackboneConfig | None) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name='unit_test_experiment',
        scenario='cifar100',
        num_experiences=1,
        backbone=backbone,
        repair=RepairConfig(
            split_fraction=0.2,
            budget_fraction=1.0,
            fit_schedule='final_only',
            num_epochs=1,
            batch_size=2,
        ),
        evaluation=EvaluationConfig(batch_size=2, avalanche_schedule='final_only'),
        runs=[
            RunConfig(
                name='repair_run',
                controller=ControllerConfig(name='fake_repair'),
            ),
        ],
        dataset_path=None,
    )


def _patch_execution_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    captured_manifests: list[dict[str, object]],
) -> None:
    strategy = _FakeStrategy()

    monkeypatch.setattr(orchestrator_module, 'init_mlflow', lambda **kwargs: _DummyContextManager())
    monkeypatch.setattr(
        orchestrator_module,
        'log_fatal_error_context',
        lambda **kwargs: _DummyContextManager(),
    )
    monkeypatch.setattr(orchestrator_module, 'set_tracking_uri', lambda **kwargs: None)
    monkeypatch.setattr(orchestrator_module, 'resolve_controller_type', lambda controller_config: 'repair')
    monkeypatch.setattr(orchestrator_module, 'MetricContextPlugin', lambda **kwargs: object())
    monkeypatch.setattr(orchestrator_module, 'SeenClassesObserver', _FakeSeenClassesObserver)
    monkeypatch.setattr(orchestrator_module, 'CalibrationCollector', lambda **kwargs: object())
    monkeypatch.setattr(orchestrator_module, 'PredictionRecorder', _FakePredictionRecorder)
    monkeypatch.setattr(orchestrator_module, 'NumericalStabilityGuardPlugin', lambda **kwargs: object())
    monkeypatch.setattr(orchestrator_module, 'BackboneCheckpointLoaderPlugin', lambda **kwargs: object())
    monkeypatch.setattr(orchestrator_module, 'RepairController', _FakeRepairController)
    monkeypatch.setattr(orchestrator_module, 'RepairControllerPlugin', _FakeRepairControllerPlugin)
    monkeypatch.setattr(orchestrator_module, 'build_controller', lambda **kwargs: _FakeRepairController())
    monkeypatch.setattr(
        orchestrator_module,
        'build_controller_plugin',
        lambda **kwargs: _FakeRepairControllerPlugin(kwargs['controller']),
    )
    monkeypatch.setattr(
        orchestrator_module,
        'build_benchmark',
        lambda **kwargs: SimpleNamespace(
            n_classes=2,
            train_stream=[SimpleNamespace(dataset=[0, 1])],
            test_stream=[SimpleNamespace(dataset=[0, 1])],
        ),
    )
    monkeypatch.setattr(
        orchestrator_module,
        'make_training_evaluation_plugin',
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        orchestrator_module,
        'RegainEvaluator',
        _FakeRegainEvaluator,
    )
    monkeypatch.setattr(orchestrator_module, 'RegainEvaluationPlugin', _FakeRegainEvaluationPlugin)
    monkeypatch.setattr(orchestrator_module, 'build_backbone', lambda **kwargs: nn.Linear(1, 2))
    monkeypatch.setattr(
        orchestrator_module,
        'build_optimizer',
        lambda **kwargs: (
            torch.optim.SGD(kwargs['model'].parameters(), lr=0.1),
            {
                'lr': 0.1
            },
        ),
    )
    monkeypatch.setattr(orchestrator_module, 'make_strategy', lambda **kwargs: strategy)
    monkeypatch.setattr(orchestrator_module, 'count_parameters', lambda *args, **kwargs: 0)
    monkeypatch.setattr(orchestrator_module, 'log_run_params', lambda **kwargs: None)
    monkeypatch.setattr(orchestrator_module, 'log_dataset_indices', lambda **kwargs: None)
    monkeypatch.setattr(orchestrator_module.mlflow, 'log_param', lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestrator_module.mlflow, 'log_artifacts', lambda *args, **kwargs: None)

    def _log_artifact(path: str) -> None:
        with Path(path).open('r', encoding='utf-8') as stream:
            captured_manifests.append(yaml.safe_load(stream))

    monkeypatch.setattr(orchestrator_module.mlflow, 'log_artifact', _log_artifact)


class TestBackboneReuseManifestArtifact:
    """
    Regression tests for reused-backbone manifest artifact logging.
    """

    def test_local_backbone_reuse_logs_full_training_in_manifest_artifact(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        backbone_run = _make_backbone_run(params=_make_backbone_run_params(monkeypatch))
        captured_manifests: list[dict[str, object]] = []

        _patch_execution_dependencies(
            monkeypatch,
            captured_manifests=captured_manifests,
        )
        monkeypatch.setattr(orchestrator_module, 'MlflowClient', _FakeMlflowClient)
        monkeypatch.setattr(
            orchestrator_module,
            'resolve_local_backbone_run',
            lambda **kwargs: backbone_run,
        )
        monkeypatch.setattr(
            orchestrator_module,
            'load_backbone_from_existing_run',
            lambda **kwargs: ([], {
                'backbone_metric': 1.0
            }, {}),
        )

        orchestrator_module.run_experiment(_make_repair_experiment_config(backbone=None),)

        assert len(captured_manifests) == 1
        backbone_config = captured_manifests[0]['backbone']
        assert backbone_config['name'] == 'resnet18'
        assert backbone_config['training']['strategy']['name'] == 'naive'
        assert backbone_config['training']['num_epochs'] == 7
        assert backbone_config['source_experiment_id'] is None

    def test_source_experiment_backbone_reuse_logs_full_training_in_manifest_artifact(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        backbone_run = _make_backbone_run(
            params=_make_backbone_run_params(monkeypatch),
            experiment_id='source_exp_id',
        )
        captured_manifests: list[dict[str, object]] = []

        _patch_execution_dependencies(
            monkeypatch,
            captured_manifests=captured_manifests,
        )
        monkeypatch.setattr(
            orchestrator_module,
            'MlflowClient',
            lambda: _FakeMlflowClient(experiment_name='source_backbone_experiment'),
        )
        monkeypatch.setattr(
            orchestrator_module,
            'resolve_local_backbone_run',
            lambda **kwargs: None,
        )
        monkeypatch.setattr(
            orchestrator_module,
            'resolve_experiment_id',
            lambda **kwargs: ('current_exp_id' if kwargs['experiment'] == 'unit_test_experiment' else 'source_exp_id'),
        )
        monkeypatch.setattr(
            orchestrator_module,
            'load_backbone_from_source_experiment',
            lambda **kwargs: ([], {
                'backbone_metric': 1.0
            }, {}, backbone_run),
        )

        orchestrator_module.run_experiment(
            _make_repair_experiment_config(backbone=BackboneConfig(
                source_experiment='other_exp',
                training=None,
            ),),)

        assert len(captured_manifests) == 1
        backbone_config = captured_manifests[0]['backbone']
        assert backbone_config['name'] == 'resnet18'
        assert backbone_config['training']['strategy']['name'] == 'naive'
        assert backbone_config['training']['num_epochs'] == 7
        assert backbone_config['source_experiment'] == 'other_exp'
        assert backbone_config['source_experiment_id'] == 'source_exp_id'
