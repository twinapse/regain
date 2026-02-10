import re
from typing import Any

from avalanche.logging import BaseLogger
import mlflow

from regain.analysis.metrics import MetricContext
from regain.analysis.metrics import MetricPhase
from regain.constants import NS_SEP

__all__ = ['MLflowLogger', 'normalize_metric_name']

_NS_SEP_ESCAPED = re.escape(NS_SEP)
_NON_ALNUM_SEP = re.compile(rf'[^a-zA-Z0-9_{_NS_SEP_ESCAPED}]+')
_MULTI_UNDERSCORE = re.compile(r'_+')
_MULTI_NAMESPACE_SEP = re.compile(rf'{_NS_SEP_ESCAPED}+')


def normalize_metric_name(raw: str) -> str:
    """
    Normalize a raw Avalanche metric name into a stable MLflow-safe token.
    """
    raw = '' if raw is None else str(raw)
    norm = raw.replace('/', NS_SEP)  # Avalanche uses '/' in metric names
    norm = _NON_ALNUM_SEP.sub('_', norm)
    norm = _MULTI_UNDERSCORE.sub('_', norm).strip('_')
    norm = _MULTI_NAMESPACE_SEP.sub(NS_SEP, norm).strip(NS_SEP)
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
