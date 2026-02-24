"""
Predictive analysis of repairability.

Assesses the predictive power of diagnostic signals (confidence, entropy, calibration, drift) for repairability (rho)
using correlation-based measures (Pearson, Spearman, R²).
"""

from pathlib import Path
from typing import Any

import numpy as np

from regain.analysis.artifacts import ARTIFACT_RHO
from regain.analysis.utils import to_float
from regain.analysis.utils import write_csv
from regain.constants import COLUMN_B
from regain.constants import COLUMN_CONTROLLER_NAME
from regain.constants import RUN_CALIB_AECE
from regain.constants import RUN_CALIB_ECE
from regain.constants import RUN_CALIB_NLL
from regain.constants import RUN_DIAG_AVG_CONF
from regain.constants import RUN_DIAG_AVG_ENTROPY
from regain.constants import RUN_DIAG_LOGIT_AVG_DRIFT
from regain.constants import RUN_DIAG_OUT_OF_TASK_RATE
from regain.utils import get_logger

__all__ = [
    'write_predictive_correlations',
]

_COLUMN_N_VALID_TASKS = 'n_valid_tasks'
_COLUMN_PEARSON_R = 'pearson_r'
_COLUMN_DIAGNOSTIC = 'diagnostic'
_COLUMN_R2 = 'r2'
_COLUMN_SPEARMAN_R = 'spearman_r'

_DIAGNOSTIC_KEYS = (
    RUN_DIAG_OUT_OF_TASK_RATE,
    RUN_DIAG_AVG_CONF,
    RUN_DIAG_AVG_ENTROPY,
    RUN_CALIB_ECE,
    RUN_CALIB_AECE,
    RUN_CALIB_NLL,
    RUN_DIAG_LOGIT_AVG_DRIFT,
)


def _pearson(*, x: np.ndarray, y: np.ndarray) -> float | None:
    """
    Compute Pearson correlation between diagnostic values and repairability.

    Args:
        x (np.ndarray): Predictor values.
        y (np.ndarray): Repairability values (`rho`).

    Returns:
        float | None: Pearson correlation in `[-1, 1]`, or None when undefined.
    """
    if x.size < 2 or y.size < 2:
        return None
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    x_norm = float(np.sqrt(np.sum(x_centered ** 2)))
    y_norm = float(np.sqrt(np.sum(y_centered ** 2)))
    if x_norm <= 0.0 or y_norm <= 0.0:
        return None
    numerator = float(np.sum(x_centered * y_centered))
    return float(numerator / (x_norm * y_norm))


def _rankdata(values: np.ndarray) -> np.ndarray:
    """
    Compute average ranks (ties get mean rank).

    Args:
        values (np.ndarray): Input values.

    Returns:
        np.ndarray: Rank vector.
    """
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.zeros(values.shape[0], dtype=np.float64)
    i = 0
    while i < sorted_values.size:
        j = i + 1
        while j < sorted_values.size and sorted_values[j] == sorted_values[i]:
            j += 1
        avg_rank = 0.5 * (float(i) + float(j - 1)) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _spearman(*, x: np.ndarray, y: np.ndarray) -> float | None:
    """
    Compute Spearman rank correlation between diagnostic signal and repairability.

    Args:
        x (np.ndarray): Predictor values.
        y (np.ndarray): Repairability values (`rho`).

    Returns:
        float | None: Spearman correlation in `[-1, 1]`, or None when undefined.
    """
    if x.size < 2 or y.size < 2:
        return None
    return _pearson(x=_rankdata(x), y=_rankdata(y))


def _r2_linear(*, x: np.ndarray, y: np.ndarray) -> float | None:
    """
    Compute univariate linear-regression coefficient of determination.

    Args:
        x (np.ndarray): Diagnostic values.
        y (np.ndarray): Repairability values (`rho`).

    Returns:
        float | None: `R^2` for `y ~ a + b*x`, or None when undefined.
    """
    if x.size < 2 or y.size < 2:
        return None
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    var_x = float(np.sum((x - x_mean) ** 2))
    if var_x <= 0.0:
        return None
    cov_xy = float(np.sum((x - x_mean) * (y - y_mean)))
    slope = cov_xy / var_x
    intercept = y_mean - slope * x_mean
    y_hat = intercept + slope * x
    denom = float(np.sum((y - y_mean) ** 2))
    if denom <= 0.0:
        return None
    sse = float(np.sum((y - y_hat) ** 2))
    return float(1.0 - (sse / denom))


def _valid_xy(
    *,
    rows: list[dict[str, Any]],
    diagnostic_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract aligned diagnostic/rho arrays with missing values removed.

    Args:
        rows (list[dict[str, Any]]): Per-task rows.
        diagnostic_key (str): Diagnostic metric key.

    Returns:
        tuple[np.ndarray, np.ndarray]: `(x, y)` arrays for valid tasks where both
            diagnostic and `rho` are defined.
    """
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        y = to_float(row.get(ARTIFACT_RHO))
        x = to_float(row.get(diagnostic_key))
        if x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def write_predictive_correlations(
    *,
    experiences_table: list[dict[str, Any]],
    out_dir: str | Path,
) -> Path:
    """
    Compute diagnostic-vs-repairability correlations and write them to CSV.

    Args:
        experiences_table (list[dict[str, Any]]): Per-experience table rows.
        out_dir (str | Path): Output directory.

    Returns:
        Path: Written CSV path.

    Notes:
        "Predictive" here refers to assessing whether a diagnostic signal has statistical predictive power for
        repairability (rho), not to building a prediction model.

        For each `(controller, budget, diagnostic)` group, rows contain:
            - `pearson_r`: linear correlation between diagnostic value and `rho`.
            - `spearman_r`: rank-based monotonic correlation.
            - `r2`: univariate linear-fit coefficient of determination.
            - `n_valid_tasks`: tasks where diagnostic and `rho` are both defined.
    """
    logger = get_logger()
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    grouped_rows: dict[tuple[str, Any], list[dict[str, Any]]] = {}
    for row in experiences_table:
        controller = str(row.get(COLUMN_CONTROLLER_NAME) or 'none')
        b = row.get(COLUMN_B)
        grouped_rows.setdefault((controller, b), []).append(row)

    result_rows: list[dict[str, Any]] = []
    for (controller, b), rows in sorted(grouped_rows.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        for diagnostic_key in _DIAGNOSTIC_KEYS:
            x, y = _valid_xy(rows=rows, diagnostic_key=diagnostic_key)
            n_valid = int(x.size)
            if n_valid == 0:
                continue
            result_rows.append({
                COLUMN_CONTROLLER_NAME: controller,
                COLUMN_B: b,
                _COLUMN_DIAGNOSTIC: diagnostic_key,
                _COLUMN_PEARSON_R: _pearson(x=x, y=y),
                _COLUMN_SPEARMAN_R: _spearman(x=x, y=y),
                _COLUMN_R2: _r2_linear(x=x, y=y),
                _COLUMN_N_VALID_TASKS: n_valid,
            })

    output_path = outp / 'predictive_correlations.csv'
    write_csv(output_path, result_rows)
    logger.warning(f'Wrote {output_path}')
    return output_path
