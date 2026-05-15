"""
Tests for the slim Avalanche strategy-side evaluator factory.
"""

from regain.analysis import MetricContext
from regain.avalanche_utils.plugins import make_training_evaluation_plugin


class TestTrainingEvaluationPluginFactory:
    """
    Tests for training evaluation plugin factory.
    """

    def test_keeps_only_loss_and_timing_metrics(self) -> None:
        plugin = make_training_evaluation_plugin(
            context=MetricContext(),
            keep_timestep_results=True,
            log_to_console=True,
            log_to_mlflow=False,
        )

        metric_names = {type(metric).__name__ for metric in plugin.metrics}
        assert 'EpochLoss' in metric_names
        assert 'ExperienceLoss' not in metric_names
        assert 'StreamLoss' not in metric_names
        assert 'EpochTime' in metric_names
        assert 'ExperienceForgetting' not in metric_names
        assert 'StreamForgetting' not in metric_names
        assert 'ExperienceForwardTransfer' not in metric_names
        assert 'StreamForwardTransfer' not in metric_names

    def test_uses_avalanche_logger_when_enabled(self) -> None:
        plugin = make_training_evaluation_plugin(
            context=MetricContext(),
            keep_timestep_results=True,
            log_to_console=False,
            log_to_mlflow=True,
        )

        logger_names = {type(logger).__name__ for logger in plugin.loggers}
        assert logger_names == {'MLflowTrainingLogger'}
