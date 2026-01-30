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


def format_debug_metric_name(name: str, exp_idx: int | None) -> str:
    """
    Format a debug metric name with the required namespace and suffix.

    Args:
        name (str): Metric base name.
        exp_idx (int | None): Optional experience index.

    Returns:
        str: Formatted debug metric name.
    """
    key = f'debug-{name}'
    if exp_idx is not None:
        key = f'{key}-exp{int(exp_idx):03d}'
    return key


def log_debug_metric(
    *,
    name: str,
    value: float,
    step: int,
    exp_idx: int | None,
) -> None:
    """
    Log a single debug metric to MLflow.

    Args:
        name (str): Metric base name.
        value (float): Metric value.
        step (int): Logging step.
        exp_idx (int | None): Optional experience index.
    """
    if mlflow.active_run() is None:
        return
    mlflow.log_metric(
        key=format_debug_metric_name(name, exp_idx),
        value=float(value),
        step=int(step),
    )


def log_debug_metrics(
    *,
    metrics: Mapping[str, float],
    step: int,
    exp_idx: int | None,
) -> None:
    """
    Log multiple debug metrics to MLflow.

    Args:
        metrics (Mapping[str, float]): Metric values keyed by base name.
        step (int): Logging step.
        exp_idx (int | None): Optional experience index.
    """
    if mlflow.active_run() is None:
        return
    for name, value in metrics.items():
        if value is None:
            continue
        log_debug_metric(name=name, value=float(value), step=step, exp_idx=exp_idx)


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
            filename = f'debug_pred_hist_{tag}.json'
        else:
            filename = f'debug_pred_hist_{tag}_exp{int(exp_idx):03d}.json'
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
        prefix = f'repair_{{metric}}_{mode}_{stage}'
        mapped: dict[str, float] = {}
        for metric in (
            'ce',
            'top1',
            'logit_l2',
            'entropy',
            'pred_unique',
            'pred_max_frac',
            'pred_entropy',
        ):
            value = metrics.get(metric)
            if value is None:
                continue
            mapped[prefix.format(metric=metric)] = float(value)
        log_debug_metrics(metrics=mapped, step=step, exp_idx=exp_idx)

        n_samples = metrics.get('n_samples')
        if n_samples is not None:
            log_debug_metric(
                name=f'repair_n_samples_{mode}_{stage}',
                value=float(n_samples),
                step=step,
                exp_idx=exp_idx,
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
        for metric in (
            'ce',
            'top1',
            'logit_l2',
            'entropy',
            'pred_unique',
            'pred_max_frac',
            'pred_entropy',
        ):
            pre_value = pre_metrics.get(metric)
            post_value = post_metrics.get(metric)
            if pre_value is None or post_value is None:
                continue
            deltas[f'repair_{metric}_{mode}_delta'] = float(post_value) - float(pre_value)
        log_debug_metrics(metrics=deltas, step=step, exp_idx=exp_idx)

    def _record_health_score(self, *, health_payload: Mapping[str, float], exp_idx: int | None, step: int) -> None:
        log_debug_metric(
            name='repair_health',
            value=health_payload['health'],
            step=step,
            exp_idx=exp_idx,
        )
        log_debug_metric(
            name='repair_health_delta',
            value=health_payload['health_delta'],
            step=step,
            exp_idx=exp_idx,
        )
        log_debug_metric(
            name='repair_health_neutral',
            value=health_payload['health_neutral'],
            step=step,
            exp_idx=exp_idx,
        )
        log_debug_metrics(
            metrics={
                'repair_health_s1_perf': float(health_payload['s1_perf']),
                'repair_health_s2_conf': float(health_payload['s2_conf']),
                'repair_health_s3_div': float(health_payload['s3_div']),
                'repair_health_r_ce': float(health_payload['r_ce']),
                'repair_health_d_acc': float(health_payload['d_acc']),
                'repair_health_r_norm': float(health_payload['r_norm']),
                'repair_health_d_ent': float(health_payload['d_ent']),
                'repair_health_d_maxfrac': float(health_payload['d_maxfrac']),
            },
            step=step,
            exp_idx=exp_idx,
        )
        if 'd_unique' in health_payload:
            log_debug_metric(
                name='repair_health_d_unique',
                value=float(health_payload['d_unique']),
                step=step,
                exp_idx=exp_idx,
            )
        if 'd_predent' in health_payload:
            log_debug_metric(
                name='repair_health_d_predent',
                value=float(health_payload['d_predent']),
                step=step,
                exp_idx=exp_idx,
            )
        self._health_scores.append(float(health_payload['health']))

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
            log_debug_metric(name='repair_diagnostics_skipped', value=1.0, step=step, exp_idx=exp_idx)
            log_debug_metric(name='repair_health_skipped', value=1.0, step=step, exp_idx=exp_idx)
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
            self.controller.fit_on_repair_data(
                model=model,
                repair_dataset=repair_dataset,
                new_classes=new_classes,
                num_epochs=self.repair_epochs,
                batch_size=self.repair_batch_size,
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
            log_debug_metric(name='repair_diagnostics_skipped', value=1.0, step=step, exp_idx=exp_idx)
            log_debug_metric(name='repair_health_skipped', value=1.0, step=step, exp_idx=exp_idx)
            return

        # Shared label space for comparable pre/post metrics.
        pre_classes = int(ctrl_pre_raw.get('num_classes') or 0)
        post_classes = int(ctrl_post_raw.get('num_classes') or 0)
        shared_classes = min(pre_classes, post_classes)
        if shared_classes <= 0:
            log_debug_metric(name='repair_diagnostics_skipped', value=1.0, step=step, exp_idx=exp_idx)
            log_debug_metric(name='repair_health_skipped', value=1.0, step=step, exp_idx=exp_idx)
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
            log_debug_metric(name='repair_diagnostics_skipped', value=1.0, step=step, exp_idx=exp_idx)
            log_debug_metric(name='repair_health_skipped', value=1.0, step=step, exp_idx=exp_idx)
            return

        # Recompute pre-fit diagnostics in the shared label space using the snapshot.
        if pre_controller is None:
            log_debug_metric(name='repair_diagnostics_skipped', value=1.0, step=step, exp_idx=exp_idx)
            log_debug_metric(name='repair_health_skipped', value=1.0, step=step, exp_idx=exp_idx)
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
            log_debug_metric(name='repair_diagnostics_skipped', value=1.0, step=step, exp_idx=exp_idx)
            log_debug_metric(name='repair_health_skipped', value=1.0, step=step, exp_idx=exp_idx)
            return

        self._log_metrics_block(metrics=ctrl_pre, mode='ctrl', stage='pre', exp_idx=exp_idx, step=step)
        if base_pre:
            self._log_metrics_block(metrics=base_pre, mode='base', stage='pre', exp_idx=exp_idx, step=step)

        if ctrl_pre.get('pred_hist'):
            self._log_pred_histogram(
                pred_hist=list(ctrl_pre['pred_hist']),
                n_samples=int(ctrl_pre.get('n_samples', 0)),
                exp_idx=exp_idx,
                tag='pre',
            )

        self._log_metrics_block(metrics=ctrl_post, mode='ctrl', stage='post', exp_idx=exp_idx, step=step)
        if (
            ctrl_pre.get('num_classes') == ctrl_post.get('num_classes')
            and ctrl_pre.get('n_samples') == ctrl_post.get('n_samples')
        ):
            self._log_deltas(pre_metrics=ctrl_pre, post_metrics=ctrl_post, mode='ctrl', exp_idx=exp_idx, step=step)

        if ctrl_post.get('pred_hist'):
            self._log_pred_histogram(
                pred_hist=list(ctrl_post['pred_hist']),
                n_samples=int(ctrl_post.get('n_samples', 0)),
                exp_idx=exp_idx,
                tag='post',
            )

        try:
            health_payload = compute_repair_health_score(pre_metrics=ctrl_pre, post_metrics=ctrl_post)
        except Exception:
            get_logger().warning('Failed to compute repair health score', exc_info=True)
            log_debug_metric(name='repair_health_skipped', value=1.0, step=step, exp_idx=exp_idx)
            return

        self._record_health_score(health_payload=health_payload, exp_idx=exp_idx, step=step)

    def after_training_exp(self, strategy: BaseTemplate, **kwargs) -> None:
        experience = strategy.experience
        model = strategy.model
        if not isinstance(model, nn.Module):
            raise TypeError('Strategy.model must be an nn.Module.')

        repair_ds = self._resolve_repair_dataset(experience)
        if repair_ds is not None:
            self._repair_datasets.append(repair_ds)

        new_classes = self._resolve_new_classes(experience, repair_ds)
        self._seen_classes.update(new_classes)

        self.controller.on_train_experience_end(model)

        if not self.fit_after_experience:
            return

        combined_dataset = self._combined_repair_dataset()
        if combined_dataset is None:
            exp_idx = int(getattr(experience, 'current_experience', 0))
            step = self._compute_step(exp_idx)
            log_debug_metric(name='repair_diagnostics_skipped', value=1.0, step=step, exp_idx=exp_idx)
            log_debug_metric(name='repair_health_skipped', value=1.0, step=step, exp_idx=exp_idx)
            return

        exp_idx = int(getattr(experience, 'current_experience', 0))
        self._run_debug_fit(
            model=model,
            repair_dataset=combined_dataset,
            new_classes=new_classes,
            exp_idx=exp_idx,
        )

    def after_training(self, strategy: BaseTemplate, **kwargs) -> None:
        model = strategy.model
        if not isinstance(model, nn.Module):
            raise TypeError('Strategy.model must be an nn.Module.')

        self.controller.on_train_end(model)

        if not self.fit_after_experience:
            combined_dataset = self._combined_repair_dataset()
            if combined_dataset is None:
                step = self._compute_step(None)
                log_debug_metric(name='repair_diagnostics_skipped', value=1.0, step=step, exp_idx=None)
                log_debug_metric(name='repair_health_skipped', value=1.0, step=step, exp_idx=None)
                return
            self._run_debug_fit(
                model=model,
                repair_dataset=combined_dataset,
                new_classes=sorted(self._seen_classes),
                exp_idx=None,
            )

        self._log_health_score_summary()

    def _log_health_score_summary(self) -> None:
        if mlflow.active_run() is None:
            return
        if not self._health_scores:
            step = self._compute_step(None)
            log_debug_metric(name='repair_health_skipped', value=1.0, step=step, exp_idx=None)
            return

        final_step = self._compute_step(None)
        mean_score = float(sum(self._health_scores) / max(1, len(self._health_scores)))
        min_score = float(min(self._health_scores))
        final_score = float(self._health_scores[-1])

        log_debug_metric(name='repair_health_mean', value=mean_score, step=final_step, exp_idx=None)
        log_debug_metric(name='repair_health_min', value=min_score, step=final_step, exp_idx=None)
        log_debug_metric(name='repair_health_final', value=final_score, step=final_step, exp_idx=None)
