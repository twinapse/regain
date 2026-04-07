from typing import Any

from avalanche.logging import BaseLogger
import mlflow

from regain.analysis.metrics import MetricContext
from regain.analysis.metrics import MetricPhase
from regain.constants import NS_SEP
from regain.mlflow_utils import normalize_metric_name
from regain.mlflow_utils import to_scalar_metric_value

__all__ = ['MLflowLogger']


class MLflowLogger(BaseLogger):
    """
    MLflow logger.
    """

    def __init__(self, *, context: MetricContext) -> None:
        super().__init__()
        self.context = context

    def _compute_step(self) -> int:
        if self.context.phase == MetricPhase.EVAL:
            return int(self.context.log_step)
        return int(self.context.train_step)

    def log_single_metric(self, name: str, value: Any, x_plot: int, **kwargs) -> None:
        if mlflow.active_run() is None:
            return
        if not self.context.log_enabled:
            return

        scalar = to_scalar_metric_value(value)
        if scalar is None:
            return

        normalized = normalize_metric_name(name)
        log_namespace = str(self.context.log_namespace or '').strip()
        if log_namespace:
            metric_key = f'{log_namespace}{NS_SEP}{normalized}'
        else:
            metric_key = normalized

        try:
            step = self._compute_step()
        except Exception:
            try:
                step = int(x_plot)
            except Exception:
                step = 0

        mlflow.log_metric(metric_key, scalar, step=step)
