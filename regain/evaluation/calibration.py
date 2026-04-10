"""
Calibration and diagnostic metric collection for evaluation passes.
"""

from dataclasses import dataclass
from dataclasses import field

import mlflow
import numpy as np
import torch

from regain.constants import DIAG_VECTOR_KEYS
from regain.constants import EXPERIENCE_KEY_PREFIX
from regain.constants import NS_SEP
from regain.constants import RUN_CALIB_AECE
from regain.constants import RUN_CALIB_BRIER
from regain.constants import RUN_CALIB_ECE
from regain.constants import RUN_CALIB_MAX_ECE
from regain.constants import RUN_CALIB_MCE
from regain.constants import RUN_CALIB_NLL
from regain.constants import RUN_DIAG_AVG_CONF
from regain.constants import RUN_DIAG_AVG_ENTROPY
from regain.constants import RUN_DIAG_LOGIT_AVG_DRIFT
from regain.constants import RUN_DIAG_OUT_OF_TASK_RATE

__all__ = ['CalibrationCollector']


@dataclass
class _ExperienceStats:
    """
    Mutable accumulator for one evaluated experience.
    """

    class_ids: set[int]
    n: int = 0
    nll_sum: float = 0.0
    brier_sum: float = 0.0
    conf_sum: float = 0.0
    entropy_sum: float = 0.0
    in_task_sum: float = 0.0
    conf_chunks: list[torch.Tensor] = field(default_factory=list)
    corr_chunks: list[torch.Tensor] = field(default_factory=list)
    logit_sum: torch.Tensor | None = None


@dataclass(frozen=True)
class _EvalMetrics:
    """
    Finalized calibration and diagnostic metrics for one experience.
    """

    ece: float
    aece: float
    mce: float
    nll: float
    brier: float
    avg_conf: float
    avg_entropy: float
    logit_mean: np.ndarray | None
    out_of_task_rate: float | None = None


class CalibrationCollector:
    """
    Collect per-experience calibration and diagnostic metrics.
    """

    def __init__(self, *, num_bins: int = 15) -> None:
        """
        Initialize the collector.

        Args:
            num_bins (int): Number of bins for ECE-like metrics.
        """
        self.num_bins = int(num_bins)
        if self.num_bins <= 0:
            raise ValueError('`num_bins` must be positive.')

        self._eval_tag: str = ''
        self._checkpoint_exp_idx: int | None = None
        self._capture_auxiliary_metrics: bool = True
        self._current_eval_metrics: dict[int, _EvalMetrics] = {}
        self._current_exp_stats: _ExperienceStats | None = None
        self._current_exp_idx: int | None = None

        self._latest_eval_metrics: dict[int, _EvalMetrics] = {}
        self._ref_logit_means: dict[int, np.ndarray] = {}
        self._base_logit_means: dict[int, np.ndarray] = {}
        self._base_diagnostics: dict[int, dict[str, float]] = {}

    def begin_pass(
        self,
        *,
        eval_tag: str,
        checkpoint_exp_idx: int | None,
        capture_auxiliary_metrics: bool,
    ) -> None:
        """
        Start one evaluation pass.

        Args:
            eval_tag (str): Evaluation tag such as `base`, `ctrl`, or `ref`.
            checkpoint_exp_idx (int | None): Checkpoint experience index represented by the pass.
            capture_auxiliary_metrics (bool): Whether to collect metrics for the pass.
        """
        self._eval_tag = str(eval_tag)
        self._checkpoint_exp_idx = (
            int(checkpoint_exp_idx)
            if checkpoint_exp_idx is not None
            else None
        )
        self._capture_auxiliary_metrics = bool(capture_auxiliary_metrics)
        self._current_eval_metrics = {}
        self._current_exp_stats = None
        self._current_exp_idx = None

    def end_pass(self, *, log_step: int | None) -> None:
        """
        Finalize one evaluation pass.

        Args:
            log_step (int | None): MLflow step used for pass-level logging.
        """
        if not self._capture_auxiliary_metrics:
            self._checkpoint_exp_idx = None
            return

        self._latest_eval_metrics = {
            int(exp_idx): values
            for exp_idx, values in self._current_eval_metrics.items()
        }
        ece_values = [
            float(values.ece)
            for values in self._current_eval_metrics.values()
        ]
        if ece_values and log_step is not None and mlflow.active_run() is not None:
            mlflow.log_metric(
                key=RUN_CALIB_MAX_ECE,
                value=float(max(ece_values)),
                step=int(log_step),
            )

        if self._eval_tag == 'base' and log_step is not None and mlflow.active_run() is not None:
            common_idxs = sorted(set(self._ref_logit_means).intersection(self._base_logit_means))
            for exp_idx in common_idxs:
                ref_mean = self._ref_logit_means[exp_idx]
                base_mean = self._base_logit_means[exp_idx]
                drift = float(np.linalg.norm(ref_mean - base_mean, ord=2))
                self._base_diagnostics.setdefault(exp_idx, {})[RUN_DIAG_LOGIT_AVG_DRIFT] = drift
                mlflow.log_metric(
                    key=self._exp_metric_key(base_key=RUN_DIAG_LOGIT_AVG_DRIFT, exp_idx=exp_idx),
                    value=drift,
                    step=int(log_step),
                )
        self._checkpoint_exp_idx = None

    def begin_experience(
        self,
        *,
        exp_idx: int,
        class_ids: list[int],
    ) -> None:
        """
        Start collecting one evaluated experience.

        Args:
            exp_idx (int): Experience index.
            class_ids (list[int]): Class ids present in the experience.
        """
        self._current_exp_stats = None
        self._current_exp_idx = None
        if not self._capture_auxiliary_metrics:
            return

        self._current_exp_idx = int(exp_idx)
        self._current_exp_stats = _ExperienceStats(class_ids=set(class_ids))

    def observe_batch(
        self,
        *,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """
        Observe one minibatch for the current experience.

        Args:
            logits (torch.Tensor): Batch logits.
            targets (torch.Tensor): Integer class targets aligned to `logits`.
        """
        if not self._capture_auxiliary_metrics:
            return
        if self._current_exp_stats is None:
            return

        targets_vec = targets.reshape(-1).to(device=logits.device, dtype=torch.long)
        if int(targets_vec.shape[0]) != int(logits.shape[0]):
            raise ValueError(
                'Calibration collector batch mismatch. '
                f'logits_batch={int(logits.shape[0])}, target_batch={int(targets_vec.shape[0])}'
            )
        if targets_vec.numel() <= 0:
            return

        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            conf, preds = torch.max(probs, dim=1)
            corr = preds.eq(targets_vec).to(dtype=torch.float32)
            p_true = probs.gather(1, targets_vec.unsqueeze(1)).squeeze(1).clamp(min=1e-12)
            nll_sum = float(torch.sum(-torch.log(p_true)).item())
            one_hot = torch.nn.functional.one_hot(
                targets_vec,
                num_classes=int(probs.shape[1]),
            ).to(dtype=probs.dtype)
            brier_sum = float(torch.sum(torch.sum((probs - one_hot) ** 2, dim=1)).item())
            entropy = -torch.sum(probs * torch.log(probs.clamp(min=1e-12)), dim=1)

            class_ids = self._current_exp_stats.class_ids
            in_task_sum = 0.0
            if class_ids:
                in_task_mask = torch.zeros_like(preds, dtype=torch.bool)
                for class_id in class_ids:
                    in_task_mask |= preds.eq(int(class_id))
                in_task_sum = float(torch.sum(in_task_mask).item())

            logit_sum_tensor = torch.sum(logits.detach(), dim=0).to(device='cpu', dtype=torch.float64)
            existing_logit_sum = self._current_exp_stats.logit_sum
            if existing_logit_sum is None:
                self._current_exp_stats.logit_sum = logit_sum_tensor
            else:
                self._current_exp_stats.logit_sum = existing_logit_sum + logit_sum_tensor

            self._current_exp_stats.n += int(targets_vec.shape[0])
            self._current_exp_stats.nll_sum += nll_sum
            self._current_exp_stats.brier_sum += brier_sum
            self._current_exp_stats.conf_sum += float(torch.sum(conf).item())
            self._current_exp_stats.entropy_sum += float(torch.sum(entropy).item())
            self._current_exp_stats.in_task_sum += in_task_sum
            self._current_exp_stats.conf_chunks.append(conf.detach().cpu())
            self._current_exp_stats.corr_chunks.append(corr.detach().cpu())

    def end_experience(self, *, log_step: int | None) -> None:
        """
        Finalize metrics for the current experience.

        Args:
            log_step (int | None): MLflow step used for logging.
        """
        if not self._capture_auxiliary_metrics:
            return
        if self._current_exp_stats is None or self._current_exp_idx is None:
            return

        n = int(self._current_exp_stats.n)
        if n <= 0:
            return

        conf_chunks = self._current_exp_stats.conf_chunks
        corr_chunks = self._current_exp_stats.corr_chunks
        confidences = torch.cat(conf_chunks).numpy() if conf_chunks else np.asarray([], dtype=np.float64)
        correctness = torch.cat(corr_chunks).numpy() if corr_chunks else np.asarray([], dtype=np.float64)

        ece, mce = self._ece_and_mce(
            confidences=confidences,
            correctness=correctness,
            num_bins=self.num_bins,
        )
        aece = self._adaptive_ece(
            confidences=confidences,
            correctness=correctness,
            num_bins=self.num_bins,
        )

        nll = float(self._current_exp_stats.nll_sum) / float(n)
        brier = float(self._current_exp_stats.brier_sum) / float(n)
        mean_conf = float(self._current_exp_stats.conf_sum) / float(n)
        mean_entropy = float(self._current_exp_stats.entropy_sum) / float(n)
        class_ids = self._current_exp_stats.class_ids
        out_of_task_rate = None
        if class_ids:
            out_of_task_rate = 1.0 - (float(self._current_exp_stats.in_task_sum) / float(n))
        logit_sum_tensor = self._current_exp_stats.logit_sum
        logit_mean = (
            logit_sum_tensor / float(n)
            if isinstance(logit_sum_tensor, torch.Tensor)
            else None
        )

        exp_idx = int(self._current_exp_idx)
        metrics_payload = _EvalMetrics(
            ece=float(ece),
            aece=float(aece),
            mce=float(mce),
            nll=float(nll),
            brier=float(brier),
            avg_conf=float(mean_conf),
            avg_entropy=float(mean_entropy),
            logit_mean=(
                logit_mean.detach().cpu().numpy()
                if isinstance(logit_mean, torch.Tensor)
                else None
            ),
            out_of_task_rate=(
                float(out_of_task_rate)
                if out_of_task_rate is not None
                else None
            ),
        )

        self._current_eval_metrics[exp_idx] = metrics_payload

        if log_step is not None and mlflow.active_run() is not None:
            for key, value in (
                (RUN_CALIB_ECE, metrics_payload.ece),
                (RUN_CALIB_AECE, metrics_payload.aece),
                (RUN_CALIB_MCE, metrics_payload.mce),
                (RUN_CALIB_NLL, metrics_payload.nll),
                (RUN_CALIB_BRIER, metrics_payload.brier),
            ):
                mlflow.log_metric(
                    key=self._exp_metric_key(base_key=key, exp_idx=exp_idx),
                    value=float(value),
                    step=int(log_step),
                )

            if self._eval_tag == 'base':
                for pred_key, pred_value in (
                    (RUN_DIAG_OUT_OF_TASK_RATE, metrics_payload.out_of_task_rate),
                    (RUN_DIAG_AVG_CONF, metrics_payload.avg_conf),
                    (RUN_DIAG_AVG_ENTROPY, metrics_payload.avg_entropy),
                ):
                    if pred_value is None:
                        continue
                    mlflow.log_metric(
                        key=self._exp_metric_key(base_key=pred_key, exp_idx=exp_idx),
                        value=float(pred_value),
                        step=int(log_step),
                    )

        if self._eval_tag == 'ref' and metrics_payload.logit_mean is not None:
            self._ref_logit_means[exp_idx] = np.asarray(metrics_payload.logit_mean, dtype=np.float64)
        if (
            self._eval_tag == 'base'
            and self._checkpoint_exp_idx is not None
            and int(exp_idx) == int(self._checkpoint_exp_idx)
            and metrics_payload.logit_mean is not None
        ):
            self._ref_logit_means[exp_idx] = np.asarray(metrics_payload.logit_mean, dtype=np.float64)
        if self._eval_tag == 'base':
            if metrics_payload.logit_mean is not None:
                self._base_logit_means[exp_idx] = np.asarray(metrics_payload.logit_mean, dtype=np.float64)
            self._base_diagnostics[exp_idx] = {
                RUN_DIAG_AVG_CONF: float(metrics_payload.avg_conf),
                RUN_DIAG_AVG_ENTROPY: float(metrics_payload.avg_entropy),
                RUN_CALIB_ECE: float(metrics_payload.ece),
                RUN_CALIB_AECE: float(metrics_payload.aece),
                RUN_CALIB_NLL: float(metrics_payload.nll),
            }
            if metrics_payload.out_of_task_rate is not None:
                self._base_diagnostics[exp_idx][RUN_DIAG_OUT_OF_TASK_RATE] = float(
                    metrics_payload.out_of_task_rate
                )

    def latest_max_ece(self) -> float | None:
        """
        Return the maximum ECE from the latest completed pass.

        Returns:
            float | None: Maximum ECE or `None` when unavailable.
        """
        ece_values = [
            float(values.ece)
            for values in self._latest_eval_metrics.values()
        ]
        if not ece_values:
            return None
        return float(max(ece_values))

    def base_diagnostic_vectors(self, *, expected_len: int) -> dict[str, list[float | None]]:
        """
        Build diagnostic vectors for `analysis_artifacts.json`.

        Args:
            expected_len (int): Expected number of experiences.

        Returns:
            dict[str, list[float | None]]: Diagnostic vectors keyed by run metric name.
        """
        vectors: dict[str, list[float | None]] = {
            key: [None for _ in range(int(expected_len))]
            for key in DIAG_VECTOR_KEYS
        }

        for exp_idx, payload in self._base_diagnostics.items():
            if exp_idx < 0 or exp_idx >= int(expected_len):
                continue
            for key in DIAG_VECTOR_KEYS:
                value = payload.get(key)
                vectors[key][exp_idx] = float(value) if value is not None else None

        for exp_idx in range(int(expected_len)):
            if vectors[RUN_DIAG_LOGIT_AVG_DRIFT][exp_idx] is not None:
                continue
            ref_mean = self._ref_logit_means.get(exp_idx)
            base_mean = self._base_logit_means.get(exp_idx)
            if ref_mean is None or base_mean is None:
                continue
            vectors[RUN_DIAG_LOGIT_AVG_DRIFT][exp_idx] = float(
                np.linalg.norm(ref_mean - base_mean, ord=2)
            )

        return vectors

    @staticmethod
    def _exp_metric_key(*, base_key: str, exp_idx: int) -> str:
        return f'{base_key}{NS_SEP}{EXPERIENCE_KEY_PREFIX}{int(exp_idx):03d}'

    @staticmethod
    def _ece_and_mce(
        *,
        confidences: np.ndarray,
        correctness: np.ndarray,
        num_bins: int,
    ) -> tuple[float, float]:
        """
        Compute fixed-width ECE and MCE.

        Args:
            confidences (np.ndarray): Max predicted probabilities.
            correctness (np.ndarray): Binary correctness indicators.
            num_bins (int): Number of equal-width bins.

        Returns:
            tuple[float, float]: `(ece, mce)`.
        """
        if confidences.size == 0:
            return 0.0, 0.0

        ece = 0.0
        mce = 0.0
        n = float(confidences.size)
        for bin_idx in range(int(num_bins)):
            lower = float(bin_idx) / float(num_bins)
            upper = float(bin_idx + 1) / float(num_bins)
            if bin_idx == 0:
                mask = (confidences >= lower) & (confidences <= upper)
            else:
                mask = (confidences > lower) & (confidences <= upper)
            if not np.any(mask):
                continue
            bin_acc = float(np.mean(correctness[mask]))
            bin_conf = float(np.mean(confidences[mask]))
            gap = abs(bin_acc - bin_conf)
            ece += (float(np.sum(mask)) / n) * gap
            mce = max(mce, gap)
        return float(ece), float(mce)

    @staticmethod
    def _adaptive_ece(
        *,
        confidences: np.ndarray,
        correctness: np.ndarray,
        num_bins: int,
    ) -> float:
        """
        Compute adaptive ECE using approximately equal-count bins.

        Args:
            confidences (np.ndarray): Max predicted probabilities.
            correctness (np.ndarray): Binary correctness indicators.
            num_bins (int): Number of equal-count bins.

        Returns:
            float: Adaptive ECE.
        """
        if confidences.size == 0:
            return 0.0
        order = np.argsort(confidences)
        bins = np.array_split(order, int(num_bins))
        n = float(confidences.size)
        aece = 0.0
        for idxs in bins:
            if idxs.size == 0:
                continue
            bin_acc = float(np.mean(correctness[idxs]))
            bin_conf = float(np.mean(confidences[idxs]))
            aece += (float(idxs.size) / n) * abs(bin_acc - bin_conf)
        return float(aece)
