"""
Diagnostics and health score utilities for repair controller debugging.
"""

import math
import random
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

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
from regain.models.controllers import RepairController
from regain.models.controllers.repair.common import build_repair_dataloader
from regain.utils import module_device
from regain.utils import preserve_model_mode_after_eval
from regain.utils import preserve_rng_state

__all__ = [
    'clamp01',
    'sigmoid01',
    'compute_repair_diagnostics',
    'compute_repair_health_score',
]

_DEBUG_PRED_UNIQUE_FRAC = 'pred_unique_frac'
_DEBUG_HEALTH_UNIQUE_FRAC_POST = 'unique_frac_post'
_DEBUG_HEALTH_UNIQUE_FRAC_PRE = 'unique_frac_pre'


_DEFAULT_EPS = 1e-12


def clamp01(value: float) -> float:
    """
    Clamp a scalar into [0, 1].

    Args:
        value (float): Input value.

    Returns:
        float: Clamped value in [0, 1].
    """
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def sigmoid01(z: float) -> float:
    """
    Compute a sigmoid in [0, 1].

    Args:
        z (float): Input scalar.

    Returns:
        float: Sigmoid output in [0, 1].
    """
    z = float(z)
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def _extract_logits(outputs: Any) -> torch.Tensor:
    """
    Extract logits from a model/controller output.

    Args:
        outputs (Any): Raw model output.

    Returns:
        torch.Tensor: Logits tensor.
    """
    if torch.is_tensor(outputs):
        return outputs
    if isinstance(outputs, (tuple, list)) and outputs:
        for item in outputs:
            if torch.is_tensor(item):
                return item
    if isinstance(outputs, dict):
        for key in ('logits', 'outputs', 'output'):
            if key in outputs and torch.is_tensor(outputs[key]):
                return outputs[key]
    raise ValueError('Unable to extract logits from model outputs.')


def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract (inputs, targets) from a batch.

    Args:
        batch (Any): Batch payload from a dataloader.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Input and target tensors.
    """
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise ValueError('Repair diagnostics batch must yield (inputs, targets, ...).')


def _hist_entropy(counts: np.ndarray) -> float:
    """
    Compute entropy from a histogram of counts.

    Args:
        counts (np.ndarray): Histogram counts.

    Returns:
        float: Entropy of the normalized histogram.
    """
    total = float(np.sum(counts))
    if total <= 0.0:
        return 0.0
    probs = counts.astype(np.float64) / total
    return float(-np.sum(probs * np.log(probs + _DEFAULT_EPS)))

def compute_repair_diagnostics(
    *,
    model: nn.Module,
    controller: RepairController,
    dataset: Dataset,
    batch_size: int,
    debug_seed: int,
    apply_controller: bool,
    class_cap: int | None = None,
    max_samples: int | None = 2048,
) -> dict[str, object]:
    """
    Evaluate a repair dataset and compute diagnostic metrics.

    Invariant: class label == logit index. Logit column ``c`` corresponds to global class ID ``c`` and controllers must
    not permute the class axis.

    Args:
        model (nn.Module): Model producing logits.
        controller (RepairController): Repair controller used for correction.
        dataset (Dataset): Repair dataset to evaluate.
        batch_size (int): Batch size for evaluation.
        debug_seed (int): Seed for deterministic debug evaluation.
        apply_controller (bool): Whether to apply controller corrections.
        class_cap (int | None): Optional shared label-space size ``K``. Diagnostics are computed on logits sliced to
            ``logits[:, :K]`` and only samples with labels in ``[0, K-1]`` are evaluated.
        max_samples (int | None): Optional cap on number of samples evaluated.

    Returns:
        dict[str, object]: Diagnostic metrics and auxiliary info.
    """
    dataloader = build_repair_dataloader(
        repair_dataset=dataset,
        batch_size=int(batch_size),
        shuffle=False,
    )
    if dataloader is None:
        return {}

    device = module_device(model, 'cpu')

    # Track raw vs valid counts to keep max_samples deterministic.
    total_loss = 0.0
    total_correct = 0.0
    total_logit_l2 = 0.0
    total_entropy = 0.0
    total_raw = 0
    total_valid = 0
    pred_hist: list[int] | None = None
    num_classes = None

    with preserve_rng_state():
        random.seed(int(debug_seed))
        np.random.seed(int(debug_seed))
        torch.manual_seed(int(debug_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(debug_seed))
        prev_training = bool(controller.training)
        controller.eval()
        try:
            with preserve_model_mode_after_eval(model):
                with torch.no_grad():
                    for batch in dataloader:
                        x, y = _unpack_batch(batch)
                        x = x.to(device)
                        y = y.to(device)
                        if y.ndim > 1:
                            y = y.view(-1)
                        y = y.long()

                        outputs = model(x)
                        logits = _extract_logits(outputs)

                        if apply_controller:
                            corrected = controller.correct_outputs(outputs=logits, model=model, inputs=x)
                            logits = _extract_logits(corrected)

                        logits = logits.float()
                        if logits.ndim != 2:
                            raise ValueError('Expected logits with shape (batch, classes).')

                        # Determine effective label space and valid targets.
                        batch_size_local = int(logits.shape[0])
                        total_raw += batch_size_local
                        k_eff = int(logits.shape[1])
                        if class_cap is not None:
                            k_eff = min(k_eff, int(class_cap))
                        if k_eff < 0:
                            k_eff = 0
                        if num_classes is None:
                            num_classes = k_eff

                        logits = logits[:, :k_eff]
                        valid = (y >= 0) & (y < k_eff)
                        if not torch.any(valid):
                            if max_samples is not None and total_raw >= int(max_samples):
                                break
                            continue

                        logits_v = logits[valid]
                        y_v = y[valid]
                        batch_valid = int(logits_v.shape[0])

                        ce = F.cross_entropy(logits_v, y_v, reduction='sum')
                        preds = torch.argmax(logits_v, dim=1)
                        correct = torch.sum(preds == y_v).item()
                        logit_l2 = torch.linalg.norm(logits_v, ord=2, dim=1).sum().item()
                        log_probs = F.log_softmax(logits_v, dim=1)
                        probs = torch.exp(log_probs)
                        entropy = -(probs * log_probs).sum(dim=1).sum().item()

                        if pred_hist is None:
                            pred_hist = [0 for _ in range(k_eff)]
                        pred_counts = torch.bincount(preds, minlength=k_eff).cpu().tolist()
                        pred_hist = [prev + int(add) for prev, add in zip(pred_hist, pred_counts, strict=False)]

                        total_loss += float(ce)
                        total_correct += float(correct)
                        total_logit_l2 += float(logit_l2)
                        total_entropy += float(entropy)
                        total_valid += batch_valid

                        if max_samples is not None and total_raw >= int(max_samples):
                            break
        finally:
            if prev_training:
                controller.train(True)

    if total_valid <= 0:
        return {}

    mean_loss = total_loss / float(total_valid)
    mean_top1 = total_correct / float(total_valid)
    mean_logit_l2 = total_logit_l2 / float(total_valid)
    mean_entropy = total_entropy / float(total_valid)

    hist_counts = np.asarray(pred_hist or [], dtype=np.int64)
    pred_unique = int(np.sum(hist_counts > 0)) if hist_counts.size > 0 else 0
    pred_max_frac = float(hist_counts.max() / total_valid) if hist_counts.size > 0 else 0.0
    pred_entropy = _hist_entropy(hist_counts) if hist_counts.size > 0 else 0.0
    pred_unique_frac = (
        float(pred_unique / num_classes) if num_classes is not None and num_classes > 0 else None
    )

    return {
        _DEBUG_CE: float(mean_loss),
        _DEBUG_TOP1: float(mean_top1),
        _DEBUG_LOGIT_L2: float(mean_logit_l2),
        _DEBUG_ENTROPY: float(mean_entropy),
        _DEBUG_PRED_UNIQUE: float(pred_unique),
        _DEBUG_PRED_MAX_FRAC: float(pred_max_frac),
        _DEBUG_PRED_ENTROPY: float(pred_entropy),
        _DEBUG_PRED_UNIQUE_FRAC: pred_unique_frac,
        _DEBUG_NUM_CLASSES: int(num_classes) if num_classes is not None else None,
        _DEBUG_N_SAMPLES: int(total_valid),
        _DEBUG_PRED_HIST: pred_hist or [],
    }


def compute_repair_health_score(
    *,
    pre_metrics: Mapping[str, object],
    post_metrics: Mapping[str, object],
    ce_scale: float = 0.02,
    acc_scale: float = 0.01,
    norm_scale: float = 0.10,
    ent_scale: float = 0.10,
    maxfrac_scale: float = 0.10,
    unique_scale: float = 0.10,
    eps: float = _DEFAULT_EPS,
) -> dict[str, float]:
    """
    Compute a repair health score in [0, 1] from pre/post diagnostics.

    Invariant: class label == logit index. Logit column ``c`` corresponds to global class ID ``c`` and controllers must
    not permute the class axis. Health score assumes pre/post diagnostics were computed over the same effective label
    space and sample subset.

    Args:
        pre_metrics (Mapping[str, object]): Controller pre-fit diagnostics.
        post_metrics (Mapping[str, object]): Controller post-fit diagnostics.
        ce_scale (float): Scale for CE improvement sigmoid.
        acc_scale (float): Scale for accuracy improvement sigmoid.
        norm_scale (float): Scale for logit norm stability term.
        ent_scale (float): Scale for entropy stability term.
        maxfrac_scale (float): Scale for max fraction stability term.
        unique_scale (float): Scale for unique-class stability term.
        eps (float): Numerical epsilon.

    Returns:
        dict[str, float]: Health score and intermediate components.
    """
    pre_num_classes = pre_metrics.get(_DEBUG_NUM_CLASSES)
    post_num_classes = post_metrics.get(_DEBUG_NUM_CLASSES)
    pre_samples = pre_metrics.get(_DEBUG_N_SAMPLES)
    post_samples = post_metrics.get(_DEBUG_N_SAMPLES)
    if pre_num_classes != post_num_classes or pre_samples != post_samples:
        raise ValueError(
            'Repair health score requires matching num_classes and n_samples in pre/post diagnostics.'
        )

    ce_pre = float(pre_metrics[_DEBUG_CE])
    ce_post = float(post_metrics[_DEBUG_CE])
    top1_pre = float(pre_metrics[_DEBUG_TOP1])
    top1_post = float(post_metrics[_DEBUG_TOP1])
    logit_l2_pre = float(pre_metrics[_DEBUG_LOGIT_L2])
    logit_l2_post = float(post_metrics[_DEBUG_LOGIT_L2])
    entropy_pre = float(pre_metrics[_DEBUG_ENTROPY])
    entropy_post = float(post_metrics[_DEBUG_ENTROPY])
    pred_unique_pre = float(pre_metrics[_DEBUG_PRED_UNIQUE])
    pred_unique_post = float(post_metrics[_DEBUG_PRED_UNIQUE])
    pred_max_frac_pre = float(pre_metrics[_DEBUG_PRED_MAX_FRAC])
    pred_max_frac_post = float(post_metrics[_DEBUG_PRED_MAX_FRAC])

    num_classes = int(pre_num_classes or 0)

    r_ce = (ce_pre - ce_post) / max(abs(ce_pre), eps)
    d_acc = top1_post - top1_pre
    s_ce = sigmoid01(r_ce / ce_scale)
    s_acc = sigmoid01(d_acc / acc_scale)
    s1 = 0.5 * s_ce + 0.5 * s_acc

    r_norm = logit_l2_post / max(logit_l2_pre, eps)
    s_norm = math.exp(-abs(r_norm - 1.0) / norm_scale)
    d_ent = entropy_post - entropy_pre
    s_ent = math.exp(-max(0.0, d_ent) / ent_scale)
    s2 = 0.5 * s_norm + 0.5 * s_ent

    d_maxfrac = pred_max_frac_post - pred_max_frac_pre
    s_maxfrac = math.exp(-max(0.0, d_maxfrac) / maxfrac_scale)

    if num_classes > 0:
        unique_frac_pre = pred_unique_pre / float(num_classes)
        unique_frac_post = pred_unique_post / float(num_classes)
        d_unique = unique_frac_pre - unique_frac_post
        s_unique = math.exp(-max(0.0, d_unique) / unique_scale)
    else:
        unique_frac_pre = None
        unique_frac_post = None
        d_unique = 0.0
        s_unique = 0.5

    pred_entropy_pre = pre_metrics.get(_DEBUG_PRED_ENTROPY)
    pred_entropy_post = post_metrics.get(_DEBUG_PRED_ENTROPY)
    use_pred_entropy = pred_entropy_pre is not None and pred_entropy_post is not None
    if use_pred_entropy:
        d_predent = float(pred_entropy_pre) - float(pred_entropy_post)
        s_predent = math.exp(-max(0.0, d_predent) / ent_scale)
        s3 = (s_maxfrac + s_unique + s_predent) / 3.0
    else:
        d_predent = 0.0
        s_predent = 0.0
        s3 = 0.5 * s_maxfrac + 0.5 * s_unique

    health = clamp01(0.50 * s1 + 0.25 * s2 + 0.25 * s3)

    s1_neutral = 0.5
    s2_neutral = 1.0
    if use_pred_entropy:
        s_unique_neutral = 1.0 if num_classes > 0 else 0.5
        s3_neutral = (1.0 + s_unique_neutral + 1.0) / 3.0
    else:
        s_unique_neutral = 1.0 if num_classes > 0 else 0.5
        s3_neutral = 0.5 * 1.0 + 0.5 * s_unique_neutral
    health_neutral = clamp01(0.50 * s1_neutral + 0.25 * s2_neutral + 0.25 * s3_neutral)
    health_delta = float(health) - float(health_neutral)

    payload = {
        _DEBUG_HEALTH: float(health),
        _DEBUG_HEALTH_NEUTRAL: float(health_neutral),
        _DEBUG_HEALTH_DELTA: float(health_delta),
        _DEBUG_HEALTH_S1_PERF: float(s1),
        _DEBUG_HEALTH_S2_CONF: float(s2),
        _DEBUG_HEALTH_S3_DIV: float(s3),
        _DEBUG_HEALTH_R_CE: float(r_ce),
        _DEBUG_HEALTH_D_ACC: float(d_acc),
        _DEBUG_HEALTH_R_NORM: float(r_norm),
        _DEBUG_HEALTH_D_ENT: float(d_ent),
        _DEBUG_HEALTH_D_MAXFRAC: float(d_maxfrac),
    }

    if num_classes > 0:
        payload[_DEBUG_HEALTH_D_UNIQUE] = float(d_unique)
    if use_pred_entropy:
        payload[_DEBUG_HEALTH_D_PREDENT] = float(d_predent)
    if unique_frac_pre is not None:
        payload[_DEBUG_HEALTH_UNIQUE_FRAC_PRE] = float(unique_frac_pre)
    if unique_frac_post is not None:
        payload[_DEBUG_HEALTH_UNIQUE_FRAC_POST] = float(unique_frac_post)

    return payload
