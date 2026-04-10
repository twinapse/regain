"""
Tests for the experiment orchestrator.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from regain.experiments.config import BackboneConfig
from regain.experiments.config import EvaluationConfig
from regain.experiments.config import ExperimentConfig
from regain.experiments.config import OptimizerConfig
from regain.experiments.config import RepairConfig
from regain.experiments.config import StrategyConfig
from regain.experiments.config import TrainingConfig
import regain.experiments.orchestrator as orchestrator_module


class _FakeStrategy:
    def __init__(self) -> None:
        self.train_kwargs: dict[str, object] | None = None

    def train(self, *, experiences: object, **kwargs) -> None:
        self.train_kwargs = dict(kwargs)
        self.train_kwargs['experiences'] = experiences


class _FakeRegainEvaluationPlugin:
    def __init__(self, **kwargs) -> None:
        del kwargs
        self.last_posthoc_scalar_results = {
            'run.eval.acc.final.avg.base': 0.73,
        }


class _FakeRegainEvaluator:
    def __init__(self, **kwargs) -> None:
        del kwargs


class _DummyContextManager:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return False


class TestTrainInvocation:
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
                {'lr': 0.1},
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
