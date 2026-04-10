from typing import Any

from avalanche.logging import BaseLogger
import mlflow

from regain.analysis.metrics import MetricContext
from regain.analysis.metrics import MetricPhase
from regain.constants import NAMESPACE_TRAIN
from regain.constants import NS_SEP
from regain.mlflow_utils import normalize_metric_name
from regain.mlflow_utils import to_scalar_metric_value

__all__ = [
    'MLflowTrainingLogger',
]


_IGNORED_METRIC_TOKENS = {
    'eval_phase',
    'test_stream',
    'train_phase',
    'train_stream',
}


def _simplify_metric_tokens(*, normalized_name: str) -> list[str]:
    """
    Drop phase/stream/task tokens from a normalized Avalanche metric name.

    Args:
        normalized_name (str): Normalized metric token from Avalanche.

    Returns:
        list[str]: Simplified tokens used for canonicalization.
    """
    tokens: list[str] = []
    for token in str(normalized_name).split(NS_SEP):
        token_str = str(token).strip()
        if token_str == '':
            continue
        if token_str in _IGNORED_METRIC_TOKENS:
            continue
        if token_str.startswith('task'):
            continue
        tokens.append(token_str)
    return tokens


def _canonicalize_train_metric(*, normalized_name: str) -> str | None:
    """
    Canonicalize retained Avalanche training metrics.

    Args:
        normalized_name (str): Normalized Avalanche metric token.

    Returns:
        str | None: Canonical MLflow key or `None` when the metric is dropped.
    """
    tokens = _simplify_metric_tokens(normalized_name=normalized_name)
    if not tokens:
        return None

    head = tokens[0]
    tail = tokens[1:]
    if head.startswith('loss_'):
        family_tokens = ['loss', head[len('loss_'):]]
        family_tokens.extend(tail)
        return f'{NAMESPACE_TRAIN}{NS_SEP}{NS_SEP.join(family_tokens)}'

    if head.startswith('time_'):
        family_tokens = ['time', head[len('time_'):]]
        family_tokens.extend(tail)
        return f'{NAMESPACE_TRAIN}{NS_SEP}{NS_SEP.join(family_tokens)}'

    return None


def _canonicalize_metric_key(
    *,
    normalized_name: str,
    log_namespace: str,
) -> str | None:
    """
    Canonicalize retained Avalanche metrics into stable MLflow namespaces.

    Args:
        normalized_name (str): Normalized Avalanche metric token.
        log_namespace (str): Active logging namespace.

    Returns:
        str | None: Canonical metric key or `None` when the metric should be dropped.
    """
    namespace = str(log_namespace).strip()
    if namespace == NAMESPACE_TRAIN:
        return _canonicalize_train_metric(normalized_name=normalized_name)
    return None


class MLflowTrainingLogger(BaseLogger):
    """
    MLflow logger for retained Avalanche training metrics.
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
        metric_key = _canonicalize_metric_key(
            normalized_name=normalized,
            log_namespace=log_namespace,
        )
        if metric_key is None:
            return

        try:
            step = self._compute_step()
        except Exception:
            try:
                step = int(x_plot)
            except Exception:
                step = 0

        mlflow.log_metric(metric_key, scalar, step=step)
