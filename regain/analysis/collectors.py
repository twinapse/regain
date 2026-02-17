"""
MLflow collectors for the REGAIN analysis tool.

This module converts MLflow runs into tidy tables suitable for automation of:
  - recoverability curves (ρ / recovered accuracy vs repair budget), and
  - efficiency frontiers (Pareto sets over data cost, parameter cost, performance).
"""

import json
from pathlib import Path
import re
from typing import Any, Optional, Sequence

import mlflow
from mlflow.tracking import MlflowClient

from regain.analysis.utils import mean
from regain.analysis.utils import to_float
from regain.analysis.utils import to_int
from regain.constants import COLUMN_B
from regain.constants import COLUMN_CONTROLLER_MODEL_PARAM_COUNT
from regain.constants import COLUMN_CONTROLLER_NAME
from regain.constants import COLUMN_EXP_IDX
from regain.constants import COLUMN_EXPERIMENT_ID
from regain.constants import COLUMN_NUM_CLASSES
from regain.constants import COLUMN_REPAIR_BUDGET_PER_CLASS
from regain.constants import COLUMN_REPAIR_BUDGET_TOTAL
from regain.constants import COLUMN_RUN_ID
from regain.constants import COLUMN_RUN_NAME
from regain.constants import COLUMN_SEED
from regain.constants import COLUMN_STATUS
from regain.constants import COLUMN_TASK_AGE
from regain.constants import EXPERIENCE_KEY_PREFIX
from regain.constants import METRIC_A_CTRL
from regain.constants import METRIC_A_CTRL_MEAN
from regain.constants import METRIC_A_POST
from regain.constants import METRIC_A_POST_MEAN
from regain.constants import METRIC_A_REF
from regain.constants import METRIC_RHO
from regain.constants import METRIC_RHO_MEAN
from regain.constants import NS_SEP
from regain.constants import PARAM_CONTROLLER_MODEL_PARAM_COUNT
from regain.constants import PARAM_CONTROLLER_PATH
from regain.constants import PARAM_NUM_CLASSES
from regain.constants import PARAM_SEED
from regain.mlflow_utils import download_json_artifact
from regain.mlflow_utils import is_parent_mlflow_run
from regain.mlflow_utils import resolve_experiment_id
from regain.mlflow_utils import resolve_mlflow_run_name
from regain.mlflow_utils import search_runs_paginated
from regain.mlflow_utils import set_tracking_uri
from regain.utils import get_logger

__all__ = [
    'collect_experiment_tables',
]

_COLUMN_SCENARIO = 'scenario'
_COLUMN_STRATEGY_NAME = 'strategy_name'
_METRIC_ANALYSIS_A_CTRL = 'analysis.a_ctrl'
_METRIC_ANALYSIS_A_POST = 'analysis.a_post'
_METRIC_ANALYSIS_A_REF = 'analysis.a_ref'
_METRIC_ANALYSIS_RHO = 'analysis.rho'
_METRIC_A_REF_MEAN = 'a_ref_mean'
_METRIC_SUMMARY_FINAL_A_CTRL_MEAN = 'summary.final_a_ctrl_mean'
_METRIC_SUMMARY_FINAL_A_POST_MEAN = 'summary.final_a_post_mean'
_METRIC_SUMMARY_FINAL_A_REF_MEAN = 'summary.final_a_ref_mean'
_METRIC_SUMMARY_FINAL_RHO_MEAN = 'summary.final_rho_mean'
_PARAM_BACKBONE_STRATEGY_NAME = 'backbone.training.strategy.name'
_PARAM_CONTROLLER_NAME = 'controller.name'
_PARAM_REPAIR_BUDGET_PER_CLASS = 'repair.budget_per_class'


_NS_SEP_ESCAPED = re.escape(NS_SEP)

_A_REF_RE = re.compile(
    rf'^{re.escape(_METRIC_ANALYSIS_A_REF)}'
    rf'{_NS_SEP_ESCAPED}{EXPERIENCE_KEY_PREFIX}(?P<idx>\d+)$'
)
_A_POST_RE = re.compile(
    rf'^{re.escape(_METRIC_ANALYSIS_A_POST)}'
    rf'{_NS_SEP_ESCAPED}{EXPERIENCE_KEY_PREFIX}(?P<idx>\d+)$'
)
_A_CTRL_RE = re.compile(
    rf'^{re.escape(_METRIC_ANALYSIS_A_CTRL)}'
    rf'{_NS_SEP_ESCAPED}{EXPERIENCE_KEY_PREFIX}(?P<idx>\d+)$'
)
_RHO_RE = re.compile(
    rf'^{re.escape(_METRIC_ANALYSIS_RHO)}'
    rf'{_NS_SEP_ESCAPED}{EXPERIENCE_KEY_PREFIX}(?P<idx>\d+)$'
)


def _extract_experience_metrics(metrics: dict[str, float]) -> dict[int, dict[str, Optional[float]]]:
    """
    Extract per-experience analysis metrics from run metrics.

    Args:
        metrics: Run metrics dict.

    Returns:
        Dict mapping experience index -> dict with keys: a_ref, a_post, a_ctrl, rho.
    """
    exp_metrics: dict[int, dict[str, Optional[float]]] = {}

    def _ensure(idx: int) -> dict[str, Optional[float]]:
        if idx not in exp_metrics:
            exp_metrics[idx] = {
                METRIC_A_REF: None,
                METRIC_A_POST: None,
                METRIC_A_CTRL: None,
                METRIC_RHO: None,
            }
        return exp_metrics[idx]

    for k, v in (metrics or {}).items():
        m = _A_REF_RE.match(k)
        if m:
            idx = int(m.group('idx'))
            _ensure(idx)[METRIC_A_REF] = to_float(v)
            continue

        m = _A_POST_RE.match(k)
        if m:
            idx = int(m.group('idx'))
            _ensure(idx)[METRIC_A_POST] = to_float(v)
            continue

        m = _A_CTRL_RE.match(k)
        if m:
            idx = int(m.group('idx'))
            _ensure(idx)[METRIC_A_CTRL] = to_float(v)
            continue

        m = _RHO_RE.match(k)
        if m:
            idx = int(m.group('idx'))
            _ensure(idx)[METRIC_RHO] = to_float(v)
            continue

    return exp_metrics


def _merge_experience_artifacts(
    exp_metrics: dict[int, dict[str, Optional[float]]],
    artifacts: dict[str, Any],
    *,
    include_keys: Sequence[str] | None = None,
) -> dict[int, dict[str, Optional[float]]]:
    """
    Merge analysis_artifacts.json vectors into a per-experience metrics dict.

    Expected artifact shape (best-effort):
        {
          "<analysis_metric_vector>": [...],
          "<analysis_metric_vector>": [...],
          "<analysis_metric_vector>": [...],
          "<analysis_metric_vector>": [...],
          "<summary_metric>": float,
          ...
        }

    Args:
        exp_metrics: Existing per-experience metrics (possibly sparse).
        artifacts: Parsed analysis artifacts.

    Returns:
        Updated per-experience metrics dict.
    """
    if not artifacts:
        return exp_metrics

    artifact_keys = (
        list(include_keys)
        if include_keys is not None
        else [METRIC_A_REF, METRIC_A_POST, METRIC_A_CTRL, METRIC_RHO]
    )

    def _ingest_vector(key: str) -> None:
        vec = artifacts.get(key)
        if not isinstance(vec, list):
            return
        for i, raw in enumerate(vec):
            idx = int(i)
            if idx not in exp_metrics:
                exp_metrics[idx] = {
                    METRIC_A_REF: None,
                    METRIC_A_POST: None,
                    METRIC_A_CTRL: None,
                    METRIC_RHO: None,
                }
            exp_metrics[idx][key] = to_float(raw)

    for k in artifact_keys:
        _ingest_vector(k)

    return exp_metrics


def _has_logged_ctrl_metrics(
    *,
    metrics: dict[str, float],
    experience_metrics: dict[int, dict[str, Optional[float]]],
) -> bool:
    """
    Check whether repair-controller analysis metrics were logged for this run.

    Args:
        metrics: Run-level metrics.
        experience_metrics: Extracted per-experience metrics from the run.

    Returns:
        bool: True if repair-controller metrics are available, else False.
    """
    if (
        _METRIC_SUMMARY_FINAL_A_CTRL_MEAN in metrics
        or _METRIC_SUMMARY_FINAL_RHO_MEAN in metrics
    ):
        return True

    for values in experience_metrics.values():
        if values.get(METRIC_A_CTRL) is not None or values.get(METRIC_RHO) is not None:
            return True
    return False


def _controller_expects_ctrl_metrics(*, params: dict[str, Any]) -> bool | None:
    """
    Infer whether a run should expose repair-controller analysis metrics.

    Args:
        params: Run params dictionary.

    Returns:
        bool | None: True for repair controllers, False for known non-repair runs, None when unknown.
    """
    controller_name = str(params.get(_PARAM_CONTROLLER_NAME) or 'none').strip().lower()
    if controller_name in {'', 'none'}:
        return False

    controller_path = str(params.get(PARAM_CONTROLLER_PATH) or '').strip().lower()
    if '.repair.' in controller_path:
        return True
    if '.prevention.' in controller_path:
        return False

    return None


def collect_experiment_tables(
    *,
    experiment: str,
    out_dir: str | Path | None = None,
    tracking_uri: str | None = None,
    include_controllers: list[str] | None = None,
    exclude_controllers: list[str] | None = None,
    max_runs: int | None = None,
    require_finished: bool = True,
    default_num_classes: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Collect parent-run tables for the analysis tool.

    Produces:
      - runs_table: one row per parent run
      - experiences_table: one row per experience (run, experience index)

    Args:
        experiment: MLflow experiment name or id.
        out_dir: Optional directory to also write 'runs_table.jsonl' and 'experiences_table.jsonl'.
        tracking_uri: Optional MLflow tracking URI or filesystem path (SQLite only).
        include_controllers: Optional allowlist of controller_name values.
        exclude_controllers: Optional denylist of controller_name values.
        max_runs: Optional limit on number of parent runs to load.
        require_finished: If True, keep only runs with status FINISHED.
        default_num_classes: Fallback number of classes when not logged.

    Returns:
        (runs_table, experiences_table)

    Raises:
        ValueError: If the experiment cannot be resolved.
    """
    logger = get_logger()

    set_tracking_uri(tracking_uri=tracking_uri)

    client = MlflowClient()
    experiment_id = resolve_experiment_id(
        client=client,
        experiment=experiment,
    )

    if max_runs is not None and int(max_runs) <= 0:
        return [], []

    # Fetch runs with pagination (avoid brittle filter-string dependency on tags presence).
    # Apply `max_runs` only after filtering nested runs so the limit refers to parent runs.
    candidate_runs = search_runs_paginated(
        client=client,
        experiment_ids=[experiment_id],
        filter_string='',
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
        order_by=['attributes.start_time DESC'],
    )

    runs_table: list[dict[str, Any]] = []
    experiences_table: list[dict[str, Any]] = []

    for run in candidate_runs:
        if not is_parent_mlflow_run(run=run):
            continue

        info = run.info
        if require_finished and str(getattr(info, COLUMN_STATUS, '')) != 'FINISHED':
            continue

        params = dict(getattr(run.data, 'params', {}) or {})
        metrics = dict(getattr(run.data, 'metrics', {}) or {})

        run_name = resolve_mlflow_run_name(run=run)
        controller_name = str(params.get(_PARAM_CONTROLLER_NAME) or 'none')
        if include_controllers is not None and controller_name not in include_controllers:
            continue
        if exclude_controllers is not None and controller_name in exclude_controllers:
            continue

        seed = to_int(params.get(PARAM_SEED))
        repair_budget_per_class = to_int(params.get(_PARAM_REPAIR_BUDGET_PER_CLASS))
        num_classes = to_int(params.get(PARAM_NUM_CLASSES))
        if num_classes is None:
            num_classes = int(default_num_classes) if default_num_classes is not None else None

        ctrl_param_count = to_int(params.get(PARAM_CONTROLLER_MODEL_PARAM_COUNT))

        b = float(repair_budget_per_class) if repair_budget_per_class is not None else None
        repair_budget_total = None
        if b is not None and num_classes is not None and num_classes > 0:
            repair_budget_total = int(float(b) * float(num_classes))

        # Prefer canonical summary metrics, then compute from per-task values.
        rho_mean = to_float(metrics.get(_METRIC_SUMMARY_FINAL_RHO_MEAN))
        a_ctrl_mean = to_float(metrics.get(_METRIC_SUMMARY_FINAL_A_CTRL_MEAN))
        a_post_mean = to_float(metrics.get(_METRIC_SUMMARY_FINAL_A_POST_MEAN))
        a_ref_mean = to_float(metrics.get(_METRIC_SUMMARY_FINAL_A_REF_MEAN))

        strategy_name = str(params.get(_PARAM_BACKBONE_STRATEGY_NAME) or '')

        experience_metrics = _extract_experience_metrics(metrics)
        has_logged_ctrl_metrics = _has_logged_ctrl_metrics(
            metrics=metrics,
            experience_metrics=experience_metrics,
        )
        ctrl_metrics_hint = _controller_expects_ctrl_metrics(params=params)
        expects_ctrl_metrics = bool(has_logged_ctrl_metrics)
        if ctrl_metrics_hint is True:
            expects_ctrl_metrics = True

        required_experience_keys = [METRIC_A_REF, METRIC_A_POST]
        if expects_ctrl_metrics:
            required_experience_keys.extend([METRIC_A_CTRL, METRIC_RHO])

        # Fallback to artifact if required per-experience metrics are empty or sparse.
        if not experience_metrics or any(
            row.get(key) is None
            for row in experience_metrics.values()
            for key in required_experience_keys
        ):
            artifacts = download_json_artifact(
                client=client,
                run_id=str(info.run_id),
                artifact_path='analysis_artifacts.json',
            )
            if artifacts:
                experience_metrics = _merge_experience_artifacts(
                    experience_metrics,
                    artifacts,
                    include_keys=required_experience_keys,
                )
                # Artifact may also include rho_mean directly
                if rho_mean is None and expects_ctrl_metrics:
                    rho_mean = to_float(artifacts.get(METRIC_RHO_MEAN))

        # Compute missing summary fields from per-experience metrics if needed.
        if rho_mean is None:
            if expects_ctrl_metrics:
                rho_mean = mean([row.get(METRIC_RHO) for row in experience_metrics.values()])
        if a_ctrl_mean is None and expects_ctrl_metrics:
            a_ctrl_mean = mean([row.get(METRIC_A_CTRL) for row in experience_metrics.values()])
        if a_post_mean is None:
            a_post_mean = mean([row.get(METRIC_A_POST) for row in experience_metrics.values()])
        if a_ref_mean is None:
            a_ref_mean = mean([row.get(METRIC_A_REF) for row in experience_metrics.values()])

        run_row: dict[str, Any] = {
            COLUMN_RUN_ID: str(info.run_id),
            COLUMN_EXPERIMENT_ID: str(experiment_id),
            COLUMN_RUN_NAME: run_name,
            _COLUMN_SCENARIO: str(params.get(_COLUMN_SCENARIO) or ''),
            _COLUMN_STRATEGY_NAME: strategy_name,
            COLUMN_SEED: seed,
            COLUMN_CONTROLLER_NAME: controller_name,
            COLUMN_REPAIR_BUDGET_PER_CLASS: repair_budget_per_class,
            COLUMN_REPAIR_BUDGET_TOTAL: repair_budget_total,
            COLUMN_NUM_CLASSES: num_classes,
            COLUMN_B: b,
            COLUMN_CONTROLLER_MODEL_PARAM_COUNT: ctrl_param_count,
            METRIC_RHO_MEAN: rho_mean,
            METRIC_A_CTRL_MEAN: a_ctrl_mean,
            METRIC_A_POST_MEAN: a_post_mean,
            _METRIC_A_REF_MEAN: a_ref_mean,
        }
        runs_table.append(run_row)

        # Build per-experience table rows and add task_age per run.
        if experience_metrics:
            max_idx = max(experience_metrics.keys())
        else:
            max_idx = -1

        for exp_idx in sorted(experience_metrics.keys()):
            row = experience_metrics[exp_idx]
            experiences_table.append({
                COLUMN_RUN_ID: str(info.run_id),
                COLUMN_SEED: seed,
                COLUMN_CONTROLLER_NAME: controller_name,
                COLUMN_REPAIR_BUDGET_PER_CLASS: repair_budget_per_class,
                COLUMN_REPAIR_BUDGET_TOTAL: repair_budget_total,
                COLUMN_NUM_CLASSES: num_classes,
                COLUMN_B: b,
                COLUMN_CONTROLLER_MODEL_PARAM_COUNT: ctrl_param_count,
                COLUMN_EXP_IDX: int(exp_idx),
                COLUMN_TASK_AGE: int(max_idx - exp_idx) if max_idx >= 0 else None,
                METRIC_A_REF: row.get(METRIC_A_REF),
                METRIC_A_POST: row.get(METRIC_A_POST),
                METRIC_A_CTRL: row.get(METRIC_A_CTRL),
                METRIC_RHO: row.get(METRIC_RHO),
            })

        if max_runs is not None and len(runs_table) >= int(max_runs):
            break

    # Optional writeout (JSONL for robustness; analysis tool scripts write CSV).
    if out_dir is not None:
        outp = Path(out_dir)
        outp.mkdir(parents=True, exist_ok=True)

        (outp / 'runs_table.jsonl').write_text(
            '\n'.join(json.dumps(r, default=str) for r in runs_table) + ('\n' if runs_table else ''),
            encoding='utf-8',
        )
        (outp / 'experiences_table.jsonl').write_text(
            '\n'.join(json.dumps(r, default=str) for r in experiences_table) + ('\n' if experiences_table else ''),
            encoding='utf-8',
        )
        logger.warning(f'Wrote tables to {outp}')

    return runs_table, experiences_table
