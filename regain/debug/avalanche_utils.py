"""
Avalanche-facing debug utilities for repair controllers.
"""

from collections.abc import Mapping
import copy
from typing import Any

from avalanche.training.templates import BaseTemplate
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from regain.avalanche_utils.plugins import RepairControllerPlugin
from regain.constants import _DEBUG_CE
from regain.constants import _DEBUG_ENTROPY
from regain.constants import _DEBUG_HEALTH
from regain.constants import _DEBUG_HEALTH_D_ACC
from regain.constants import _DEBUG_HEALTH_D_ENT
from regain.constants import _DEBUG_HEALTH_D_MAXFRAC
from regain.constants import _DEBUG_HEALTH_D_PREDENT
from regain.constants import _DEBUG_HEALTH_D_UNIQUE
from regain.constants import _DEBUG_HEALTH_DELTA
from regain.constants import _DEBUG_HEALTH_NEUTRAL
from regain.constants import _DEBUG_HEALTH_R_CE
from regain.constants import _DEBUG_HEALTH_R_NORM
from regain.constants import _DEBUG_HEALTH_S1_PERF
from regain.constants import _DEBUG_HEALTH_S2_CONF
from regain.constants import _DEBUG_HEALTH_S3_DIV
from regain.constants import _DEBUG_LOGIT_L2
from regain.constants import _DEBUG_N_SAMPLES
from regain.constants import _DEBUG_NUM_CLASSES
from regain.constants import _DEBUG_PRED_ENTROPY
from regain.constants import _DEBUG_PRED_HIST
from regain.constants import _DEBUG_PRED_MAX_FRAC
from regain.constants import _DEBUG_PRED_UNIQUE
from regain.constants import _DEBUG_TOP1
from regain.constants import EXPERIENCE_KEY_PREFIX
from regain.constants import NAMESPACE_DEBUG
from regain.constants import NS_SEP
from regain.debug.metrics import compute_repair_diagnostics
from regain.debug.metrics import compute_repair_health_score
from regain.models.controllers import RepairController
from regain.utils import get_logger

__all__ = [
    'DebugRepairControllerPlugin',
    'format_debug_metric_name',
    'log_debug_metric',
    'log_debug_metrics',
]

_DEBUG_PRED_HIST_EXP_FILENAME_TEMPLATE = 'debug_pred_hist_{tag}_exp{exp_idx:03d}.json'
_DEBUG_PRED_HIST_FILENAME_TEMPLATE = 'debug_pred_hist_{tag}.json'
_DEBUG_REPAIR_DELTA_TEMPLATE = 'repair.{metric}.delta'
_DEBUG_REPAIR_DIAGNOSTICS_SKIPPED = 'repair.diagnostics_skipped'
_DEBUG_REPAIR_HEALTH = 'repair.health'
_DEBUG_REPAIR_HEALTH_COMPONENT_D_ACC = 'repair.health.d_acc'
_DEBUG_REPAIR_HEALTH_COMPONENT_D_ENT = 'repair.health.d_ent'
_DEBUG_REPAIR_HEALTH_COMPONENT_D_MAXFRAC = 'repair.health.d_maxfrac'
_DEBUG_REPAIR_HEALTH_COMPONENT_R_CE = 'repair.health.r_ce'
_DEBUG_REPAIR_HEALTH_COMPONENT_R_NORM = 'repair.health.r_norm'
_DEBUG_REPAIR_HEALTH_COMPONENT_S1_PERF = 'repair.health.s1_perf'
_DEBUG_REPAIR_HEALTH_COMPONENT_S2_CONF = 'repair.health.s2_conf'
_DEBUG_REPAIR_HEALTH_COMPONENT_S3_DIV = 'repair.health.s3_div'
_DEBUG_REPAIR_HEALTH_DELTA = 'repair.health.delta'
_DEBUG_REPAIR_HEALTH_D_PREDENT = 'repair.health.d_predent'
_DEBUG_REPAIR_HEALTH_D_UNIQUE = 'repair.health.d_unique'
_DEBUG_REPAIR_HEALTH_FINAL = 'repair.health.final'
_DEBUG_REPAIR_HEALTH_AVG = 'repair.health.avg'
_DEBUG_REPAIR_HEALTH_MIN = 'repair.health.min'
_DEBUG_REPAIR_HEALTH_NEUTRAL = 'repair.health.neutral'
_DEBUG_REPAIR_HEALTH_SKIPPED = 'repair.health.skipped'
_DEBUG_REPAIR_N_SAMPLES_TEMPLATE = 'repair.n_samples.{stage}'
_DEBUG_REPAIR_TEMPLATE = 'repair.{metric}.{stage}'


_REPAIR_DIAG_METRIC_KEYS = (
    _DEBUG_CE,
    _DEBUG_TOP1,
    _DEBUG_LOGIT_L2,
    _DEBUG_ENTROPY,
    _DEBUG_PRED_UNIQUE,
    _DEBUG_PRED_MAX_FRAC,
    _DEBUG_PRED_ENTROPY,
)

_HEALTH_COMPONENT_KEY_MAP = {
    _DEBUG_REPAIR_HEALTH_COMPONENT_S1_PERF: _DEBUG_HEALTH_S1_PERF,
    _DEBUG_REPAIR_HEALTH_COMPONENT_S2_CONF: _DEBUG_HEALTH_S2_CONF,
    _DEBUG_REPAIR_HEALTH_COMPONENT_S3_DIV: _DEBUG_HEALTH_S3_DIV,
    _DEBUG_REPAIR_HEALTH_COMPONENT_R_CE: _DEBUG_HEALTH_R_CE,
    _DEBUG_REPAIR_HEALTH_COMPONENT_D_ACC: _DEBUG_HEALTH_D_ACC,
    _DEBUG_REPAIR_HEALTH_COMPONENT_R_NORM: _DEBUG_HEALTH_R_NORM,
    _DEBUG_REPAIR_HEALTH_COMPONENT_D_ENT: _DEBUG_HEALTH_D_ENT,
    _DEBUG_REPAIR_HEALTH_COMPONENT_D_MAXFRAC: _DEBUG_HEALTH_D_MAXFRAC,
}


def format_debug_metric_name(name: str, exp_idx: int | None, mode: str | None = None) -> str:
    """
    Format a debug metric name with the required namespace and suffix.

    Args:
        name (str): Metric base name.
        exp_idx (int | None): Optional experience index.
        mode (str | None): Optional variant mode (`base` or `ctrl`) appended at the end.

    Returns:
        str: Formatted debug metric name.
    """
    key = f'{NAMESPACE_DEBUG}{NS_SEP}{name}'
    if exp_idx is not None:
        key = f'{key}{NS_SEP}{EXPERIENCE_KEY_PREFIX}{int(exp_idx):03d}'
    if mode is not None:
        key = f'{key}{NS_SEP}{mode}'
    return key


def log_debug_metric(
    *,
    name: str,
    value: float,
    step: int,
    exp_idx: int | None,
    mode: str | None = None,
) -> None:
    """
    Log a single debug metric to MLflow.

    Args:
        name (str): Metric base name.
        value (float): Metric value.
        step (int): Logging step.
        exp_idx (int | None): Optional experience index.
        mode (str | None): Optional variant mode (`base` or `ctrl`).
    """
    if mlflow.active_run() is None:
        return
    mlflow.log_metric(
        key=format_debug_metric_name(name, exp_idx, mode),
        value=float(value),
        step=int(step),
    )


def log_debug_metrics(
    *,
    metrics: Mapping[str, float],
    step: int,
    exp_idx: int | None,
    mode: str | None = None,
) -> None:
    """
    Log multiple debug metrics to MLflow.

    Args:
        metrics (Mapping[str, float]): Metric values keyed by base name.
        step (int): Logging step.
        exp_idx (int | None): Optional experience index.
        mode (str | None): Optional variant mode (`base` or `ctrl`).
    """
    if mlflow.active_run() is None:
        return
    for name, value in metrics.items():
        if value is None:
            continue
        log_debug_metric(name=name, value=float(value), step=step, exp_idx=exp_idx, mode=mode)


class DebugRepairControllerPlugin(RepairControllerPlugin):
    """
    Repair controller plugin that logs pre/post-fit debug diagnostics.
    """

    def __init__(
        self,
        controller: RepairController,
        *,
        fit_after_experience: bool,
        repair_epochs: int,
        repair_batch_size: int,
        budget_per_class: int,
        max_repair_samples_per_class: int,
        seed: int,
        debug_epochs: int,
        debug_experiences: int,
        debug_seed: int,
        debug_max_samples: int | None = 2048,
        log_pred_histograms: bool = True,
    ) -> None:
        """
        Initialize the debug repair controller plugin.

        Args:
            controller (RepairController): Controller to wire into Avalanche.
            fit_after_experience (bool): Whether to fit after each experience.
            repair_epochs (int): Number of epochs for repair fitting.
            repair_batch_size (int): Batch size for repair fitting.
            budget_per_class (int): Repair budget `b` used from each fixed repair set.
            max_repair_samples_per_class (int): Upper bound on per-class repair samples available in the scenario.
            seed (int): Global seed used for deterministic budget selection.
            debug_epochs (int): Epochs per experience used only to compute debug metric step values.
            debug_experiences (int): Total experiences used only to compute debug metric step values.
            debug_seed (int): Random seed for debug dataloading.
            debug_max_samples (int | None): Max number of samples for diagnostics.
            log_pred_histograms (bool): Whether to log prediction histograms.
        """
        super().__init__(
            controller=controller,
            fit_after_experience=fit_after_experience,
            repair_epochs=repair_epochs,
            repair_batch_size=repair_batch_size,
            budget_per_class=budget_per_class,
            max_repair_samples_per_class=max_repair_samples_per_class,
            seed=seed,
        )
        self._debug_epochs = int(debug_epochs)
        self._debug_experiences = int(debug_experiences)
        self._debug_seed = int(debug_seed)
        self._debug_max_samples = debug_max_samples
        self._log_pred_histograms = bool(log_pred_histograms)
        self._health_scores: list[float] = []

    def _compute_step(self, exp_idx: int | None) -> int:
        if exp_idx is None:
            return int(self._debug_experiences * self._debug_epochs)
        return int((exp_idx + 1) * self._debug_epochs)

    def _log_pred_histogram(
        self,
        *,
        pred_hist: list[int],
        n_samples: int,
        exp_idx: int | None,
        tag: str,
    ) -> None:
        if not self._log_pred_histograms:
            return
        if mlflow.active_run() is None:
            return
        payload = {
            'counts': pred_hist,
            'total_samples': int(n_samples),
        }
        if exp_idx is None:
            filename = _DEBUG_PRED_HIST_FILENAME_TEMPLATE.format(tag=tag)
        else:
            filename = _DEBUG_PRED_HIST_EXP_FILENAME_TEMPLATE.format(tag=tag, exp_idx=int(exp_idx))
        mlflow.log_dict(payload, filename)

    def _log_metrics_block(
        self,
        *,
        metrics: Mapping[str, Any],
        mode: str,
        stage: str,
        exp_idx: int | None,
        step: int,
    ) -> None:
        mapped: dict[str, float] = {}
        for metric in _REPAIR_DIAG_METRIC_KEYS:
            value = metrics.get(metric)
            if value is None:
                continue
            mapped[_DEBUG_REPAIR_TEMPLATE.format(metric=metric, stage=stage)] = float(value)
        log_debug_metrics(metrics=mapped, step=step, exp_idx=exp_idx, mode=mode)

        n_samples = metrics.get(_DEBUG_N_SAMPLES)
        if n_samples is not None:
            log_debug_metric(
                name=_DEBUG_REPAIR_N_SAMPLES_TEMPLATE.format(stage=stage),
                value=float(n_samples),
                step=step,
                exp_idx=exp_idx,
                mode=mode,
            )

    def _log_deltas(
        self,
        *,
        pre_metrics: Mapping[str, Any],
        post_metrics: Mapping[str, Any],
        mode: str,
        exp_idx: int | None,
        step: int,
    ) -> None:
        deltas: dict[str, float] = {}
        for metric in _REPAIR_DIAG_METRIC_KEYS:
            pre_value = pre_metrics.get(metric)
            post_value = post_metrics.get(metric)
            if pre_value is None or post_value is None:
                continue
            deltas[_DEBUG_REPAIR_DELTA_TEMPLATE.format(metric=metric)] = (
                float(post_value) - float(pre_value)
            )
        log_debug_metrics(metrics=deltas, step=step, exp_idx=exp_idx, mode=mode)

    def _record_health_score(self, *, health_payload: Mapping[str, float], exp_idx: int | None, step: int) -> None:
        log_debug_metric(
            name=_DEBUG_REPAIR_HEALTH,
            value=health_payload[_DEBUG_HEALTH],
            step=step,
            exp_idx=exp_idx,
        )
        log_debug_metric(
            name=_DEBUG_REPAIR_HEALTH_DELTA,
            value=health_payload[_DEBUG_HEALTH_DELTA],
            step=step,
            exp_idx=exp_idx,
        )
        log_debug_metric(
            name=_DEBUG_REPAIR_HEALTH_NEUTRAL,
            value=health_payload[_DEBUG_HEALTH_NEUTRAL],
            step=step,
            exp_idx=exp_idx,
        )
        log_debug_metrics(
            metrics={
                debug_key: float(health_payload[payload_key])
                for debug_key, payload_key in _HEALTH_COMPONENT_KEY_MAP.items()
            },
            step=step,
            exp_idx=exp_idx,
        )
        if _DEBUG_HEALTH_D_UNIQUE in health_payload:
            log_debug_metric(
                name=_DEBUG_REPAIR_HEALTH_D_UNIQUE,
                value=float(health_payload[_DEBUG_HEALTH_D_UNIQUE]),
                step=step,
                exp_idx=exp_idx,
            )
        if _DEBUG_HEALTH_D_PREDENT in health_payload:
            log_debug_metric(
                name=_DEBUG_REPAIR_HEALTH_D_PREDENT,
                value=float(health_payload[_DEBUG_HEALTH_D_PREDENT]),
                step=step,
                exp_idx=exp_idx,
            )
        self._health_scores.append(float(health_payload[_DEBUG_HEALTH]))

    def _run_debug_fit(
        self,
        *,
        model: nn.Module,
        repair_dataset: Dataset,
        new_classes: list[int],
        exp_idx: int | None,
    ) -> None:
        step = self._compute_step(exp_idx)

        # Raw pre-fit diagnostics to infer class width.
        ctrl_pre_raw = compute_repair_diagnostics(
            model=model,
            controller=self.controller,
            dataset=repair_dataset,
            batch_size=self.repair_batch_size,
            seed=self._debug_seed,
            apply_controller=True,
            class_cap=None,
            max_samples=self._debug_max_samples,
        )

        if not ctrl_pre_raw:
            log_debug_metric(name=_DEBUG_REPAIR_DIAGNOSTICS_SKIPPED, value=1.0, step=step, exp_idx=exp_idx)
            log_debug_metric(name=_DEBUG_REPAIR_HEALTH_SKIPPED, value=1.0, step=step, exp_idx=exp_idx)
            return

        # Snapshot controller object pre-fit.
        # Some controllers have a dynamic module structure and create/replace submodules during fit
        # (e.g., LinearProbeController sets self._probe), so load_state_dict(pre_state) fails because
        # pre-fit state lacks _probe.* keys.
        try:
            pre_controller = copy.deepcopy(self.controller)
        except Exception:
            pre_controller = None
        with torch.enable_grad():
            self._fit_controller_on_repair_dataset(
                model=model,
                repair_dataset=repair_dataset,
                new_classes=new_classes,
                exp_idx=exp_idx,
            )

        # Raw post-fit diagnostics to infer class width.
        ctrl_post_raw = compute_repair_diagnostics(
            model=model,
            controller=self.controller,
            dataset=repair_dataset,
            batch_size=self.repair_batch_size,
            seed=self._debug_seed,
            apply_controller=True,
            class_cap=None,
            max_samples=self._debug_max_samples,
        )

        if not ctrl_post_raw:
            log_debug_metric(name=_DEBUG_REPAIR_DIAGNOSTICS_SKIPPED, value=1.0, step=step, exp_idx=exp_idx)
            log_debug_metric(name=_DEBUG_REPAIR_HEALTH_SKIPPED, value=1.0, step=step, exp_idx=exp_idx)
            return

        # Shared label space for comparable pre/post metrics.
        pre_classes = int(ctrl_pre_raw.get(_DEBUG_NUM_CLASSES) or 0)
        post_classes = int(ctrl_post_raw.get(_DEBUG_NUM_CLASSES) or 0)
        shared_classes = min(pre_classes, post_classes)
        if shared_classes <= 0:
            log_debug_metric(name=_DEBUG_REPAIR_DIAGNOSTICS_SKIPPED, value=1.0, step=step, exp_idx=exp_idx)
            log_debug_metric(name=_DEBUG_REPAIR_HEALTH_SKIPPED, value=1.0, step=step, exp_idx=exp_idx)
            return

        # Recompute post-fit diagnostics in the shared label space.
        ctrl_post = compute_repair_diagnostics(
            model=model,
            controller=self.controller,
            dataset=repair_dataset,
            batch_size=self.repair_batch_size,
            seed=self._debug_seed,
            apply_controller=True,
            class_cap=shared_classes,
            max_samples=self._debug_max_samples,
        )
        if not ctrl_post:
            log_debug_metric(name=_DEBUG_REPAIR_DIAGNOSTICS_SKIPPED, value=1.0, step=step, exp_idx=exp_idx)
            log_debug_metric(name=_DEBUG_REPAIR_HEALTH_SKIPPED, value=1.0, step=step, exp_idx=exp_idx)
            return

        # Recompute pre-fit diagnostics in the shared label space using the snapshot.
        if pre_controller is None:
            log_debug_metric(name=_DEBUG_REPAIR_DIAGNOSTICS_SKIPPED, value=1.0, step=step, exp_idx=exp_idx)
            log_debug_metric(name=_DEBUG_REPAIR_HEALTH_SKIPPED, value=1.0, step=step, exp_idx=exp_idx)
            return

        ctrl_pre = compute_repair_diagnostics(
            model=model,
            controller=pre_controller,
            dataset=repair_dataset,
            batch_size=self.repair_batch_size,
            seed=self._debug_seed,
            apply_controller=True,
            class_cap=shared_classes,
            max_samples=self._debug_max_samples,
        )
        base_pre = compute_repair_diagnostics(
            model=model,
            controller=pre_controller,
            dataset=repair_dataset,
            batch_size=self.repair_batch_size,
            seed=self._debug_seed,
            apply_controller=False,
            class_cap=shared_classes,
            max_samples=self._debug_max_samples,
        )

        if not ctrl_pre:
            log_debug_metric(name=_DEBUG_REPAIR_DIAGNOSTICS_SKIPPED, value=1.0, step=step, exp_idx=exp_idx)
            log_debug_metric(name=_DEBUG_REPAIR_HEALTH_SKIPPED, value=1.0, step=step, exp_idx=exp_idx)
            return

        self._log_metrics_block(metrics=ctrl_pre, mode='ctrl', stage='pre', exp_idx=exp_idx, step=step)
        if base_pre:
            self._log_metrics_block(metrics=base_pre, mode='base', stage='pre', exp_idx=exp_idx, step=step)

        if ctrl_pre.get(_DEBUG_PRED_HIST):
            self._log_pred_histogram(
                pred_hist=list(ctrl_pre[_DEBUG_PRED_HIST]),
                n_samples=int(ctrl_pre.get(_DEBUG_N_SAMPLES, 0)),
                exp_idx=exp_idx,
                tag='pre',
            )

        self._log_metrics_block(metrics=ctrl_post, mode='ctrl', stage='post', exp_idx=exp_idx, step=step)
        if (
            ctrl_pre.get(_DEBUG_NUM_CLASSES) == ctrl_post.get(_DEBUG_NUM_CLASSES)
            and ctrl_pre.get(_DEBUG_N_SAMPLES) == ctrl_post.get(_DEBUG_N_SAMPLES)
        ):
            self._log_deltas(pre_metrics=ctrl_pre, post_metrics=ctrl_post, mode='ctrl', exp_idx=exp_idx, step=step)

        if ctrl_post.get(_DEBUG_PRED_HIST):
            self._log_pred_histogram(
                pred_hist=list(ctrl_post[_DEBUG_PRED_HIST]),
                n_samples=int(ctrl_post.get(_DEBUG_N_SAMPLES, 0)),
                exp_idx=exp_idx,
                tag='post',
            )

        try:
            health_payload = compute_repair_health_score(pre_metrics=ctrl_pre, post_metrics=ctrl_post)
        except Exception:
            get_logger().warning('Failed to compute repair health score', exc_info=True)
            log_debug_metric(name=_DEBUG_REPAIR_HEALTH_SKIPPED, value=1.0, step=step, exp_idx=exp_idx)
            return

        self._record_health_score(health_payload=health_payload, exp_idx=exp_idx, step=step)

    def after_training_exp(self, strategy: BaseTemplate, **kwargs) -> None:
        del kwargs
        experience = strategy.experience
        if experience is None:
            return
        model = strategy.model
        if not isinstance(model, nn.Module):
            raise TypeError('Strategy.model must be an nn.Module.')

        repair_set_ds = self._ingest_repair_dataset(experience=experience)

        new_classes = self._resolve_new_classes(experience, repair_set_ds)
        self._seen_classes.update(new_classes)

        self.controller.on_train_experience_end(model)

        if not self.fit_after_experience:
            return

        combined_dataset = self._combined_repair_dataset()
        if combined_dataset is None:
            exp_idx = int(getattr(experience, 'current_experience', 0))
            step = self._compute_step(exp_idx)
            log_debug_metric(name=_DEBUG_REPAIR_DIAGNOSTICS_SKIPPED, value=1.0, step=step, exp_idx=exp_idx)
            log_debug_metric(name=_DEBUG_REPAIR_HEALTH_SKIPPED, value=1.0, step=step, exp_idx=exp_idx)
            return

        exp_idx = int(getattr(experience, 'current_experience', 0))
        self._run_debug_fit(
            model=model,
            repair_dataset=combined_dataset,
            new_classes=new_classes,
            exp_idx=exp_idx,
        )

    def after_training(self, strategy: BaseTemplate, **kwargs) -> None:
        del kwargs
        model = strategy.model
        if not isinstance(model, nn.Module):
            raise TypeError('Strategy.model must be an nn.Module.')

        self.controller.on_train_end(model)

        if not self.fit_after_experience:
            combined_dataset = self._combined_repair_dataset()
            if combined_dataset is None:
                step = self._compute_step(None)
                log_debug_metric(name=_DEBUG_REPAIR_DIAGNOSTICS_SKIPPED, value=1.0, step=step, exp_idx=None)
                log_debug_metric(name=_DEBUG_REPAIR_HEALTH_SKIPPED, value=1.0, step=step, exp_idx=None)
                return
            self._run_debug_fit(
                model=model,
                repair_dataset=combined_dataset,
                new_classes=sorted(self._seen_classes),
                exp_idx=None,
            )

        self._log_health_score_summary()

    def _log_health_score_summary(self) -> None:
        if not self._health_scores:
            step = self._compute_step(None)
            log_debug_metric(name=_DEBUG_REPAIR_HEALTH_SKIPPED, value=1.0, step=step, exp_idx=None)
            return

        final_step = self._compute_step(None)
        mean_score = float(sum(self._health_scores) / max(1, len(self._health_scores)))
        min_score = float(min(self._health_scores))
        final_score = float(self._health_scores[-1])

        log_debug_metric(name=_DEBUG_REPAIR_HEALTH_AVG, value=mean_score, step=final_step, exp_idx=None)
        log_debug_metric(name=_DEBUG_REPAIR_HEALTH_MIN, value=min_score, step=final_step, exp_idx=None)
        log_debug_metric(name=_DEBUG_REPAIR_HEALTH_FINAL, value=final_score, step=final_step, exp_idx=None)
