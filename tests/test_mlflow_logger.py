"""
Tests for the Avalanche MLflow logger canonicalization policy.
"""

import mlflow
import pytest

from regain.analysis import MetricContext
from regain.analysis.metrics import MetricPhase
from regain.avalanche_utils.logging import MLflowLogger
from regain.constants import NAMESPACE_EVAL
from regain.constants import NAMESPACE_TRAIN


def _make_context(*, namespace: str, phase: MetricPhase, step: int) -> MetricContext:
    """
    Build a metric context configured for one logger test.

    Args:
        namespace (str): Logging namespace.
        phase (MetricPhase): Active metric phase.
        step (int): Step to expose through the context.

    Returns:
        MetricContext: Configured context.
    """
    context = MetricContext()
    context.set_phase(phase=phase)
    context.set_log_namespace(name=namespace)
    context.set_log_enabled(True)
    context.set_log_step(step=step)
    context.set_train_step(step=step)
    return context


class TestMlflowLogger:
    def test_canonicalizes_eval_forgetting_metrics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        logged_metrics: list[tuple[str, float, int]] = []
        logger = MLflowLogger(
            context=_make_context(
                namespace=NAMESPACE_EVAL,
                phase=MetricPhase.EVAL,
                step=20,
            )
        )

        monkeypatch.setattr(mlflow, 'active_run', lambda: object())
        monkeypatch.setattr(
            mlflow,
            'log_metric',
            lambda key, value, step: logged_metrics.append((str(key), float(value), int(step))),
        )

        logger.log_single_metric(
            name='ExperienceForgetting/eval_phase/test_stream/Exp002',
            value=0.31,
            x_plot=0,
        )

        assert logged_metrics == [('run.eval.forgetting.exp002', pytest.approx(0.31), 20)]

    def test_drops_eval_accuracy_metrics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        logger = MLflowLogger(
            context=_make_context(
                namespace=NAMESPACE_EVAL,
                phase=MetricPhase.EVAL,
                step=30,
            )
        )
        called = False

        monkeypatch.setattr(mlflow, 'active_run', lambda: object())

        def _log_metric(*args, **kwargs) -> None:
            del args, kwargs
            nonlocal called
            called = True

        monkeypatch.setattr(mlflow, 'log_metric', _log_metric)

        logger.log_single_metric(
            name='Top1_Acc_Exp/eval_phase/test_stream/Exp001',
            value=0.73,
            x_plot=0,
        )

        assert called is False

    def test_canonicalizes_train_loss_and_time_metrics(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        logged_metrics: list[tuple[str, float, int]] = []
        logger = MLflowLogger(
            context=_make_context(
                namespace=NAMESPACE_TRAIN,
                phase=MetricPhase.TRAIN,
                step=7,
            )
        )

        monkeypatch.setattr(mlflow, 'active_run', lambda: object())
        monkeypatch.setattr(
            mlflow,
            'log_metric',
            lambda key, value, step: logged_metrics.append((str(key), float(value), int(step))),
        )

        logger.log_single_metric(
            name='Loss_Exp/train_phase/train_stream/Exp001',
            value=0.22,
            x_plot=0,
        )
        logger.log_single_metric(
            name='Time_Epoch/train_phase/train_stream',
            value=1.5,
            x_plot=0,
        )

        assert logged_metrics == [
            ('run.train.loss.exp.exp001', pytest.approx(0.22), 7),
            ('run.train.time.epoch', pytest.approx(1.5), 7),
        ]

    def test_canonicalizes_eval_phase_loss_to_train_namespace(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        logged_metrics: list[tuple[str, float, int]] = []
        logger = MLflowLogger(
            context=_make_context(
                namespace=NAMESPACE_EVAL,
                phase=MetricPhase.EVAL,
                step=5,
            )
        )

        monkeypatch.setattr(mlflow, 'active_run', lambda: object())
        monkeypatch.setattr(
            mlflow,
            'log_metric',
            lambda key, value, step: logged_metrics.append((str(key), float(value), int(step))),
        )

        logger.log_single_metric(
            name='Loss_Exp/eval_phase/test_stream/Exp003',
            value=0.45,
            x_plot=0,
        )

        assert logged_metrics == [('run.train.loss.exp.exp003', pytest.approx(0.45), 5)]
