"""
Helpers for assembling `analysis_artifacts.json`.
"""

from collections.abc import Mapping
from typing import Sequence, TypeAlias

from regain.analysis.metrics import mean_ignore_invalid
from regain.analysis.metrics import retrieval_correctable_fractions

__all__ = [
    'AnalysisArtifacts',
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

AnalysisArtifactScalar: TypeAlias = str | int | float | None
AnalysisArtifactVector: TypeAlias = list[float] | list[float | None]
AnalysisArtifactValue: TypeAlias = AnalysisArtifactScalar | AnalysisArtifactVector
AnalysisArtifacts: TypeAlias = dict[str, AnalysisArtifactValue]


def build_analysis_artifacts(
    a_exp_base: Sequence[float],
    a_base: Sequence[float],
    a_final_ctrl: Sequence[float],
    eps: float = 1e-4,
    extra_vectors: Mapping[str, Sequence[float | None]] | None = None,
    extra_scalars: Mapping[str, float] | None = None,
) -> AnalysisArtifacts:
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

    payload: AnalysisArtifacts = {
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
