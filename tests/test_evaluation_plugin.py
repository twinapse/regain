"""
Tests for evaluation plugin.
"""

# Ensure a stable import order for plugin module initialization.
import regain.experiments.orchestrator  # noqa: F401
from regain.analysis import MetricContext
from regain.avalanche_utils.plugins import make_evaluation_plugin


###############################
# Evaluation factory coverage #
###############################

class TestEvaluationPluginFactory:
    def test_includes_forward_transfer_metrics_when_enabled(self) -> None:
        plugin = make_evaluation_plugin(
            context=MetricContext(),
            keep_timestep_results=True,
            log_to_console=True,
            log_to_mlflow=False,
            include_forward_transfer=True,
        )

        metric_names = {type(metric).__name__ for metric in plugin.metrics}
        assert 'ExperienceForwardTransfer' in metric_names
        assert 'StreamForwardTransfer' in metric_names

    def test_excludes_forward_transfer_metrics_when_disabled(self) -> None:
        plugin = make_evaluation_plugin(
            context=MetricContext(),
            keep_timestep_results=True,
            log_to_console=True,
            log_to_mlflow=False,
            include_forward_transfer=False,
        )

        metric_names = {type(metric).__name__ for metric in plugin.metrics}
        assert 'ExperienceForwardTransfer' not in metric_names
        assert 'StreamForwardTransfer' not in metric_names
