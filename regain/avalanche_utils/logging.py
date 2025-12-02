import re
from typing import Any

from avalanche.logging import BaseLogger
import mlflow

from regain.analysis.metrics import METRIC_NAMESPACE_SEPARATOR
from regain.analysis.metrics import MetricContext
from regain.analysis.metrics import MetricPhase

__all__ = ['MLflowLogger', 'normalize_metric_name']

_NON_ALNUM_SEP = re.compile(r'[^a-zA-Z0-9_-]+')
_MULTI_UNDERSCORE = re.compile(r'_+')
_MULTI_HYPHEN = re.compile(r'-+')


def normalize_metric_name(raw: str) -> str:
    """
    Normalize a raw Avalanche metric name into a stable MLflow-safe token.
    """
    raw = '' if raw is None else str(raw)
    norm = raw.replace('/', METRIC_NAMESPACE_SEPARATOR)  # Avalanche uses '/' in metric names
    # TODO: If `METRIC_NAMESPACE_SEPARATOR` is not "-" or "_", the line below would replace it too
    norm = _NON_ALNUM_SEP.sub('_', norm)
    norm = _MULTI_UNDERSCORE.sub('_', norm).strip('_')
    norm = _MULTI_HYPHEN.sub('-', norm).strip('-')
    return norm.lower() or 'unnamed_metric'


def _to_scalar(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if hasattr(value, 'item'):
        try:
            v = value.item()
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
            return float(v)
        except Exception:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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

        scalar = _to_scalar(value)
        if scalar is None:
            return

        normalized = normalize_metric_name(name)
        metric_key = f'{self.context.log_namespace}{METRIC_NAMESPACE_SEPARATOR}{normalized}'

        try:
            step = self._compute_step()
        except Exception:
            try:
                step = int(x_plot)
            except Exception:
                step = 0

        mlflow.log_metric(metric_key, scalar, step=step)
