"""
Tests for the retained Avalanche training logger.
"""

import mlflow
import pytest

from regain.analysis import MetricContext
from regain.analysis.metrics import MetricPhase
from regain.avalanche_utils.logging import MLflowTrainingLogger
from regain.constants import NAMESPACE_EVAL
from regain.constants import NAMESPACE_TRAIN


def _make_context(*, namespace: str, phase: MetricPhase, step: int) -> MetricContext:
    """
    Build a metric context configured for one logger test.

    Args:
        namespace (str): Logging namespace.
        phase (MetricPhase): Active phase.
        step (int): Training and log step.

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


class TestMlflowTrainingLogger:
    """
    Tests for MLflowTrainingLogger.
    """

    def test_canonicalizes_train_loss_and_time_metrics(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        logged_metrics: list[tuple[str, float, int]] = []
        logger = MLflowTrainingLogger(context=_make_context(
            namespace=NAMESPACE_TRAIN,
            phase=MetricPhase.TRAIN,
            step=7,
        ))

        monkeypatch.setattr(mlflow, 'active_run', object)
        monkeypatch.setattr(
            mlflow,
            'log_metric',
            lambda key, value, step: logged_metrics.append((str(key), float(value), int(step))),
        )

        logger.log_single_metric(
            name='Loss_Epoch/train_phase/train_stream',
            value=0.22,
            x_plot=0,
        )
        logger.log_single_metric(
            name='Time_Epoch/train_phase/train_stream',
            value=1.5,
            x_plot=0,
        )
        logger.log_single_metric(
            name='Loss_Exp/train_phase/train_stream/Exp001',
            value=0.33,
            x_plot=0,
        )
        logger.log_single_metric(
            name='Loss_Stream/train_phase/train_stream',
            value=0.44,
            x_plot=0,
        )

        assert logged_metrics == [
            ('run.train.loss.epoch', pytest.approx(0.22), 7),
            ('run.train.time.epoch', pytest.approx(1.5), 7),
        ]

    def test_ignores_eval_namespace_metrics(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        logger = MLflowTrainingLogger(context=_make_context(
            namespace=NAMESPACE_EVAL,
            phase=MetricPhase.EVAL,
            step=5,
        ))
        logged_metrics: list[tuple[str, float, int]] = []

        monkeypatch.setattr(mlflow, 'active_run', object)
        monkeypatch.setattr(
            mlflow,
            'log_metric',
            lambda key, value, step: logged_metrics.append((str(key), float(value), int(step))),
        )

        logger.log_single_metric(
            name='ExperienceForgetting',
            value=0.31,
            x_plot=0,
        )
        logger.log_single_metric(
            name='ExperienceForwardTransfer',
            value=0.12,
            x_plot=0,
        )
        logger.log_single_metric(
            name='Loss_Stream',
            value=1.75,
            x_plot=0,
        )
        logger.log_single_metric(
            name='Top1_Acc_Stream',
            value=0.9,
            x_plot=0,
        )

        assert not logged_metrics
