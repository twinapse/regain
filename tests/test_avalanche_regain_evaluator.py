"""
Tests for the custom Avalanche-side evaluator.
"""

from dataclasses import dataclass
from types import SimpleNamespace

from avalanche.benchmarks import CLScenario
from avalanche.benchmarks.scenarios.deprecated.generators import nc_benchmark
from avalanche.benchmarks.utils.classification_dataset import _make_taskaware_tensor_classification_dataset
import mlflow
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from regain.analysis import MetricContext
from regain.avalanche_utils.evaluation import RegainEvaluator
from regain.evaluation import CalibrationCollector
from regain.evaluation import ClassMask
from regain.evaluation import PredictionRecorder
from regain.models.controllers import RepairController


class _IdentityModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _LogitDataset(Dataset):
    def __init__(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        self._logits = logits
        self._targets = targets

    def __len__(self) -> int:
        return int(self._targets.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        return self._logits[idx], self._targets[idx], 0


@dataclass
class _Experience:
    current_experience: int
    dataset: Dataset
    classes_in_this_experience: list[int]


@dataclass
class _Benchmark:
    train_stream: list[object]
    test_stream: list[object]
    n_classes: int


class _StubRepairController(RepairController):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

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

    def correct_outputs(
        self,
        *,
        outputs,
        model: nn.Module | None = None,
        inputs=None,
    ):
        del model, inputs
        return outputs + torch.tensor([[0.0, 5.0]], dtype=outputs.dtype)

    def on_eval_begin(self, *args, **kwargs) -> None:
        del args, kwargs
        self.calls.append('before_eval')

    def on_eval_end(self, *args, **kwargs) -> None:
        del args, kwargs
        self.calls.append('after_eval')

    def on_eval_experience_begin(self, *args, **kwargs) -> None:
        del args, kwargs
        self.calls.append('before_eval_exp')

    def on_eval_experience_end(self, *args, **kwargs) -> None:
        del args, kwargs
        self.calls.append('after_eval_exp')


def _make_benchmark() -> _Benchmark:
    exp0 = _Experience(
        current_experience=0,
        dataset=_LogitDataset(
            logits=torch.tensor([[3.0, 0.0], [0.0, 3.0]], dtype=torch.float32),
            targets=torch.tensor([0, 1], dtype=torch.long),
        ),
        classes_in_this_experience=[0, 1],
    )
    exp1 = _Experience(
        current_experience=1,
        dataset=_LogitDataset(
            logits=torch.tensor([[0.2, 0.8]], dtype=torch.float32),
            targets=torch.tensor([1], dtype=torch.long),
        ),
        classes_in_this_experience=[1],
    )
    return _Benchmark(
        train_stream=[exp0, exp1],
        test_stream=[exp0, exp1],
        n_classes=2,
    )


def _make_avalanche_benchmark() -> CLScenario:
    datasets = []
    for class_ids in ([0, 1], [2, 3]):
        logits = torch.eye(4, dtype=torch.float32)[list(class_ids)]
        targets = torch.tensor(class_ids, dtype=torch.long)
        datasets.append(
            _make_taskaware_tensor_classification_dataset(
                logits,
                targets,
                targets=targets.tolist(),
                task_labels=0,
            )
        )
    benchmark = nc_benchmark(
        train_dataset=datasets,
        test_dataset=datasets,
        n_experiences=2,
        task_labels=False,
        shuffle=False,
        one_dataset_per_exp=True,
    )
    benchmark.n_classes = 4
    return benchmark


def _make_evaluator(
    *,
    controller: RepairController | None,
    tmp_path,
) -> RegainEvaluator:
    benchmark = _make_benchmark()
    seen_classes: set[int] = {0, 1} if controller is not None else set()
    return RegainEvaluator(
        benchmark=benchmark,
        model=_IdentityModel(),
        controller=controller,
        seen_classes=seen_classes,
        device=torch.device('cpu'),
        criterion=nn.CrossEntropyLoss(),
        num_classes=2,
        calibration=CalibrationCollector(num_bins=2),
        prediction_recorder=PredictionRecorder(
            artifact_root=tmp_path / 'predictions',
            num_classes=2,
        ),
        context=MetricContext(),
        batch_size=2,
        num_epochs_per_experience=5,
        repair_after_experience=True,
        include_forward_transfer=True,
        backbone_analysis_baseline={
            'acc.exp.base': [1.0, 1.0],
            'acc.final.base': [1.0, 1.0],
            'run.diagnostics.out_of_task_rate': [0.0, 0.0],
            'run.diagnostics.avg_conf': [0.0, 0.0],
            'run.diagnostics.avg_entropy': [0.0, 0.0],
            'run.calibration.ece': [0.0, 0.0],
            'run.calibration.aece': [0.0, 0.0],
            'run.calibration.nll': [0.0, 0.0],
            'run.diagnostics.logit_avg_drift': [0.0, 0.0],
        }
        if controller is not None
        else None,
        eps=1e-4,
    )


class TestRegainEvaluator:
    def test_eval_pass_collects_accuracy_loss_and_logits(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
    ) -> None:
        monkeypatch.setattr(mlflow, 'active_run', lambda: None)
        evaluator = _make_evaluator(controller=None, tmp_path=tmp_path)

        result = evaluator.eval_pass(
            evaluator.benchmark.test_stream,
            label='ckpt',
            eval_tag='base',
            checkpoint_exp_idx=0,
            capture_logits=True,
            capture_predictions=True,
            capture_auxiliary_metrics=True,
            log_step=5,
        )

        assert result.per_exp_acc[0] == pytest.approx(1.0)
        assert result.per_exp_acc[1] == pytest.approx(1.0)
        assert result.per_exp_loss[0] >= 0.0
        assert result.per_exp_logits is not None
        assert result.per_exp_targets is not None
        assert result.per_exp_logits[0].shape == (2, 2)
        assert result.per_exp_targets[0].shape == (2,)

    def test_eval_pass_applies_mask(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setattr(mlflow, 'active_run', lambda: None)
        evaluator = _make_evaluator(controller=None, tmp_path=tmp_path)

        result = evaluator.eval_pass(
            [evaluator.benchmark.test_stream[1]],
            label='ref',
            eval_tag='ref',
            checkpoint_exp_idx=0,
            mask=ClassMask.from_seen_classes([0], mask_value=-1e9),
            capture_logits=False,
            capture_predictions=False,
            capture_auxiliary_metrics=False,
        )

        assert result.per_exp_acc[1] == pytest.approx(0.0)

    def test_eval_pass_applies_repair_controller_and_captures_backbone_logits(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
    ) -> None:
        monkeypatch.setattr(mlflow, 'active_run', lambda: None)
        controller = _StubRepairController()
        evaluator = _make_evaluator(controller=controller, tmp_path=tmp_path)

        evaluator.eval_pass(
            [evaluator.benchmark.test_stream[0]],
            label='ckpt',
            eval_tag='ctrl',
            checkpoint_exp_idx=0,
            capture_logits=True,
            capture_backbone_logits=True,
            capture_predictions=False,
            capture_auxiliary_metrics=False,
        )

        assert controller.calls == [
            'before_eval',
            'before_eval_exp',
            'after_eval_exp',
            'after_eval',
        ]

    def test_eval_pass_accepts_avalanche_classification_datasets(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
    ) -> None:
        monkeypatch.setattr(mlflow, 'active_run', lambda: None)
        benchmark = _make_avalanche_benchmark()
        evaluator = RegainEvaluator(
            benchmark=benchmark,
            model=_IdentityModel(),
            controller=None,
            seen_classes=set(),
            device=torch.device('cpu'),
            criterion=nn.CrossEntropyLoss(),
            num_classes=4,
            calibration=CalibrationCollector(num_bins=2),
            prediction_recorder=PredictionRecorder(
                artifact_root=tmp_path / 'predictions',
                num_classes=4,
            ),
            context=MetricContext(),
            batch_size=2,
            num_epochs_per_experience=1,
            repair_after_experience=True,
            include_forward_transfer=True,
            backbone_analysis_baseline=None,
            eps=1e-4,
        )

        result = evaluator.eval_pass(
            benchmark.test_stream,
            label='ckpt',
            eval_tag='base',
            checkpoint_exp_idx=0,
            capture_logits=True,
            capture_predictions=True,
            capture_auxiliary_metrics=True,
            log_step=1,
        )

        assert result.per_exp_acc[0] == pytest.approx(1.0)
        assert result.per_exp_acc[1] == pytest.approx(1.0)
        assert result.per_exp_logits is not None
        assert result.per_exp_targets is not None
        assert result.per_exp_logits[0].shape == (2, 4)
        assert result.per_exp_targets[0].shape == (2,)

    def test_run_after_training_exp_handles_single_experience_ref_pass(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
    ) -> None:
        monkeypatch.setattr(mlflow, 'active_run', lambda: None)
        evaluator = _make_evaluator(controller=None, tmp_path=tmp_path)
        strategy = SimpleNamespace(experience=SimpleNamespace(current_experience=0))

        evaluator.run_before_training()
        evaluator.run_after_training_exp(strategy=strategy, seen_classes={0, 1})

        assert evaluator.acc_exp_base == pytest.approx([1.0])
        assert evaluator.last_posthoc_scalar_results is not None

    def test_run_after_training_exposes_canonical_final_accuracy_keys(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
    ) -> None:
        monkeypatch.setattr(mlflow, 'active_run', lambda: None)
        evaluator = _make_evaluator(controller=None, tmp_path=tmp_path)
        strategy = SimpleNamespace(experience=SimpleNamespace(current_experience=0))

        evaluator.run_before_training()
        evaluator.run_after_training_exp(strategy=strategy, seen_classes={0, 1})
        strategy.experience.current_experience = 1
        evaluator.run_after_training_exp(strategy=strategy, seen_classes={0, 1})
        evaluator.run_after_training(strategy=strategy, seen_classes={0, 1})

        assert evaluator.last_posthoc_scalar_results == {
            'run.eval.acc.final.exp000.base': pytest.approx(1.0),
            'run.eval.acc.final.exp001.base': pytest.approx(1.0),
            'run.eval.acc.final.avg.base': pytest.approx(1.0),
        }

    def test_run_after_training_exp_uses_train_step_for_eval_style_metrics(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
    ) -> None:
        logged_metrics: list[tuple[str, float, int]] = []

        monkeypatch.setattr(mlflow, 'active_run', lambda: object())
        monkeypatch.setattr(
            mlflow,
            'log_metric',
            lambda key, value, step: logged_metrics.append(
                (str(key), float(value), int(step))
            ),
        )

        evaluator = _make_evaluator(controller=_StubRepairController(), tmp_path=tmp_path)
        strategy = SimpleNamespace(experience=SimpleNamespace(current_experience=0))

        evaluator.run_before_training()
        evaluator.run_after_training_exp(strategy=strategy, seen_classes={0, 1})

        logged_steps: dict[str, list[int]] = {}
        for key, _value, step in logged_metrics:
            logged_steps.setdefault(key, []).append(step)

        assert logged_steps['run.calibration.ece.exp000'] == [0]
        assert logged_steps['run.eval.forgetting.stream'] == [0, 0]
        assert logged_steps['run.eval.transfer.stream'] == [0, 0]
        assert logged_steps['run.train.loss.exp000.train'] == [0]
        assert logged_steps['run.train.loss.exp000.test'] == [0]
        assert logged_steps['run.eval.acc.ref.exp000.base'] == [5]
        assert not any(key.startswith('run.eval.loss.') for key in logged_steps)
        assert not any(
            key.startswith('run.train.') and f'.stream' in key
            for key in logged_steps
        )
        assert not any(
            key.startswith('run.eval.') and (
                '.train.' in f'.{key}.'
                or '.test.' in f'.{key}.'
            )
            for key in logged_steps
        )
