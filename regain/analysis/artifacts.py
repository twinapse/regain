"""
Helpers for extracting analysis accuracies from evaluation results.
"""

import re
from typing import Dict, List, Sequence

from regain.analysis.metrics import mean_ignore_invalid
from regain.analysis.metrics import retrieval_correctable_fractions
from regain.analysis.utils import to_float
from regain.constants import METRIC_A_CTRL
from regain.constants import METRIC_A_POST
from regain.constants import METRIC_A_REF
from regain.constants import METRIC_EPS
from regain.constants import METRIC_RHO
from regain.constants import METRIC_RHO_MEAN

__all__ = [
    'build_analysis_artifacts',
    'extract_top1_by_experience',
    'ordered_accuracies',
]

_METRIC_DELTA_A = 'delta_a'
_METRIC_F_RES = 'f_res'
_METRIC_F_TOTAL = 'f_total'
_METRIC_TOKEN_AVALANCHE_EVAL_PHASE = 'eval_phase'
_METRIC_TOKEN_AVALANCHE_TEST_STREAM = 'test_stream'
_METRIC_TOKEN_AVALANCHE_TOP1_ACC_EXP = 'Top1_Acc_Exp'


_EXP_RE = re.compile(r'(?:^|/|\b)Exp(\d+)(?:\b|/|$)')


def _parse_exp_idx(key: str) -> int | None:
    m = _EXP_RE.search(key)
    return int(m.group(1)) if m else None


# TODO: Make it Avalanche-agnostic by passing expected metric name patterns
def extract_top1_by_experience(
    eval_results: dict[str, object],
    num_experiences: int,
) -> Dict[int, float]:
    """
    Extract Top1 accuracy per experience from Avalanche eval results.

    Args:
        eval_results: Evaluation results mapping metric names to values.
        num_experiences: Number of experiences to expect.

    Returns:
        Mapping from experience index to Top1 accuracy.
    """
    acc_by_exp: dict[int, float] = {}
    def _maybe_record(items: dict[str, object]) -> None:
        for key, value in items.items():
            exp_idx = _parse_exp_idx(key)
            if exp_idx is None or exp_idx < 0 or exp_idx >= num_experiences:
                continue
            score = to_float(value, allow_bool=True, require_finite=False)
            if score is None:
                continue
            acc_by_exp[exp_idx] = score

    preferred_keys = {
        key: value
        for key, value in eval_results.items()
        if _METRIC_TOKEN_AVALANCHE_TOP1_ACC_EXP in key
        and (_METRIC_TOKEN_AVALANCHE_EVAL_PHASE in key)
        and (_METRIC_TOKEN_AVALANCHE_TEST_STREAM in key)
    }
    if preferred_keys:
        _maybe_record(preferred_keys)
    else:
        candidates = {
            key: value
            for key, value in eval_results.items()
            if _METRIC_TOKEN_AVALANCHE_TOP1_ACC_EXP in key
        }
        _maybe_record(candidates)
    return acc_by_exp


def ordered_accuracies(
    eval_results: dict[str, object],
    num_experiences: int,
) -> List[float]:
    """
    Produce an ordered list of Top1 accuracies across experiences.

    Args:
        eval_results: Evaluation results mapping metric names to values.
        num_experiences: Number of experiences to expect.

    Returns:
        List of accuracies ordered by experience index.

    Raises:
        ValueError: If an experience accuracy is missing.
    """
    acc_map = extract_top1_by_experience(eval_results, num_experiences)
    accuracies: list[float] = []
    for exp_idx in range(num_experiences):
        if exp_idx not in acc_map:
            raise ValueError(f'Missing Top1 accuracy for experience {exp_idx}.')
        accuracies.append(acc_map[exp_idx])
    return accuracies


def build_analysis_artifacts(
    a_ref: Sequence[float],
    a_post: Sequence[float],
    a_ctrl: Sequence[float],
    eps: float = 1e-4,
) -> dict[str, object]:
    """
    Construct a JSON-serializable bundle of analysis metrics.

    Args:
        a_ref: Reference accuracies measured after each experience.
        a_post: Post-sequence accuracies without repair.
        a_ctrl: Post-sequence accuracies with controller applied.
        eps: Minimum magnitude of total forgetting to consider a task valid.

    Returns:
        Dictionary containing per-task vectors, aggregate rho mean, and eps used.
    """

    lengths = {len(array) for array in (a_ref, a_post, a_ctrl)}
    if len(lengths) > 1:
        raise ValueError('Inputs must have the same length.')

    a_ref_list = [float(value) for value in a_ref]
    a_post_list = [float(value) for value in a_post]
    a_ctrl_list = [float(value) for value in a_ctrl]

    f_total = [ref - post for ref, post in zip(a_ref_list, a_post_list)]
    f_res = [ref - ctrl for ref, ctrl in zip(a_ref_list, a_ctrl_list)]
    delta_a = [ctrl - post for ctrl, post in zip(a_ctrl_list, a_post_list)]

    rho = retrieval_correctable_fractions(zip(a_ref_list, a_post_list, a_ctrl_list), eps)
    rho_mean = mean_ignore_invalid(rho)

    return {
        METRIC_A_REF: a_ref_list,
        METRIC_A_POST: a_post_list,
        METRIC_A_CTRL: a_ctrl_list,
        _METRIC_F_TOTAL: f_total,
        _METRIC_F_RES: f_res,
        _METRIC_DELTA_A: delta_a,
        METRIC_RHO: rho,
        METRIC_RHO_MEAN: rho_mean,
        METRIC_EPS: eps,
    }
