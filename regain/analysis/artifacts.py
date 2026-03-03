"""
Helpers for extracting analysis accuracies from evaluation results.
"""

from collections.abc import Mapping
import re
from typing import Dict, List, Sequence

from regain.analysis.metrics import mean_ignore_invalid
from regain.analysis.metrics import retrieval_correctable_fractions
from regain.analysis.utils import to_float

__all__ = [
    'ARTIFACT_ACC_EXP_BASE',
    'ARTIFACT_ACC_FINAL_BASE',
    'ARTIFACT_ACC_FINAL_CTRL',
    'ARTIFACT_DELTA_A',
    'ARTIFACT_EPS',
    'ARTIFACT_F_RES',
    'ARTIFACT_F_TOTAL',
    'ARTIFACT_RHO',
    'ARTIFACT_RHO_AVG',
    'build_analysis_artifacts',
    'extract_top1_by_experience',
    'ordered_accuracies',
]

# Artifact JSON keys (not MLflow metric keys — no run. prefix).
# These follow the same naming convention but live inside analysis_artifacts.json.
# Public because collectors.py and backbone.py read them back from the JSON.
ARTIFACT_ACC_EXP_BASE = 'acc.exp.base'
ARTIFACT_ACC_FINAL_BASE = 'acc.final.base'
ARTIFACT_ACC_FINAL_CTRL = 'acc.final.ctrl'
ARTIFACT_RHO = 'rho'
ARTIFACT_RHO_AVG = 'rho.avg'
ARTIFACT_EPS = 'eps'
ARTIFACT_DELTA_A = 'delta_a'
ARTIFACT_F_RES = 'f_res'
ARTIFACT_F_TOTAL = 'f_total'

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
    a_exp_base: Sequence[float],
    a_base: Sequence[float],
    a_final_ctrl: Sequence[float],
    eps: float = 1e-4,
    extra_vectors: Mapping[str, Sequence[float | None]] | None = None,
    extra_scalars: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """
    Construct a JSON-serializable bundle of analysis metrics.

    Args:
        a_exp_base: Base accuracies measured after each experience (end-of-experience).
        a_base: Final accuracies without controller (base, post-sequence).
        a_final_ctrl: Final accuracies with controller applied (ctrl, post-sequence).
        eps: Minimum magnitude of total forgetting to consider a task valid.
        extra_vectors: Optional additional per-task vectors to embed in the artifact.
        extra_scalars: Optional additional scalar metrics to embed in the artifact.

    Returns:
        Dictionary containing per-task vectors, aggregate rho avg, and eps used.
    """

    lengths = {len(array) for array in (a_exp_base, a_base, a_final_ctrl)}
    if len(lengths) > 1:
        raise ValueError('Inputs must have the same length.')

    a_exp_base_list = [float(value) for value in a_exp_base]
    a_base_list = [float(value) for value in a_base]
    a_final_ctrl_list = [float(value) for value in a_final_ctrl]

    f_total = [exp_base - base for exp_base, base in zip(a_exp_base_list, a_base_list)]
    f_res = [exp_base - ctrl for exp_base, ctrl in zip(a_exp_base_list, a_final_ctrl_list)]
    delta_a = [ctrl - base for ctrl, base in zip(a_final_ctrl_list, a_base_list)]

    rho = retrieval_correctable_fractions(zip(a_exp_base_list, a_base_list, a_final_ctrl_list), eps)
    rho_avg = mean_ignore_invalid(rho)

    payload: dict[str, object] = {
        ARTIFACT_ACC_EXP_BASE: a_exp_base_list,
        ARTIFACT_ACC_FINAL_BASE: a_base_list,
        ARTIFACT_ACC_FINAL_CTRL: a_final_ctrl_list,
        ARTIFACT_F_TOTAL: f_total,
        ARTIFACT_F_RES: f_res,
        ARTIFACT_DELTA_A: delta_a,
        ARTIFACT_RHO: rho,
        ARTIFACT_RHO_AVG: rho_avg,
        ARTIFACT_EPS: eps,
    }

    if extra_vectors is not None:
        for key, vector in extra_vectors.items():
            vector_values: list[float | None] = []
            for value in vector:
                if value is None:
                    vector_values.append(None)
                else:
                    vector_values.append(float(value))
            if len(vector_values) != len(a_exp_base_list):
                raise ValueError(
                    f'Extra vector `{key}` length mismatch. '
                    f'expected={len(a_exp_base_list)}, observed={len(vector_values)}'
                )
            payload[str(key)] = vector_values

    if extra_scalars is not None:
        for key, value in extra_scalars.items():
            payload[str(key)] = float(value)

    return payload
