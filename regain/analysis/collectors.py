"""
MLflow collectors for the REGAIN analysis tool.

This module converts MLflow runs into tidy tables suitable for automation of:
  - recoverability curves (ρ / recovered accuracy vs repair budget), and
  - efficiency frontiers (Pareto sets over data cost, parameter cost, performance).
"""

import json
from pathlib import Path
import re
import tempfile
from typing import Any, Optional

import mlflow
from mlflow.tracking import MlflowClient

from regain.analysis.utils import mean
from regain.analysis.utils import to_float
from regain.analysis.utils import to_int
from regain.mlflow_utils import resolve_experiment_id
from regain.mlflow_utils import search_runs_paginated
from regain.mlflow_utils import set_sqlite_tracking_uri
from regain.utils import get_logger

__all__ = [
    'collect_experiment_tables',
]


_A_REF_RE = re.compile(r'^analysis-a_ref-exp(?P<idx>\d+)$')
_A_POST_RE = re.compile(r'^analysis-a_post-exp(?P<idx>\d+)$')
_A_CTRL_RE = re.compile(r'^analysis-a_ctrl-exp(?P<idx>\d+)$')
_RHO_RE = re.compile(r'^analysis-rho-exp(?P<idx>\d+)$')


def _is_parent_run(run_tags: dict[str, str]) -> bool:
    """
    Determine whether a run is a parent run (not a nested eval run).

    Args:
        run_tags: Run tags.

    Returns:
        True if run is a parent run, False otherwise.
    """
    # MLflow nested runs have a 'mlflow.parentRunId' tag.
    return 'mlflow.parentRunId' not in (run_tags or {})


def _download_json_artifact(
    client: MlflowClient,
    *,
    run_id: str,
    artifact_path: str,
) -> Optional[dict[str, Any]]:
    """
    Download and parse a JSON artifact from an MLflow run.

    Args:
        client: MLflow client.
        run_id: Run id.
        artifact_path: Artifact path in the run.

    Returns:
        Parsed JSON dict, or None if not available.
    """
    try:
        with tempfile.TemporaryDirectory() as tmp:
            dst = client.download_artifacts(run_id, artifact_path, tmp)
            p = Path(dst)
            if p.is_dir():
                p = p / Path(artifact_path).name
            if not p.exists():
                return None
            return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        # Fallback: mlflow.artifacts API (varies across MLflow versions)
        try:
            local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
            p = Path(local_path)
            if p.is_dir():
                p = p / Path(artifact_path).name
            if not p.exists():
                return None
            return json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            return None


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
            exp_metrics[idx] = {'a_ref': None, 'a_post': None, 'a_ctrl': None, 'rho': None}
        return exp_metrics[idx]

    for k, v in (metrics or {}).items():
        m = _A_REF_RE.match(k)
        if m:
            idx = int(m.group('idx'))
            _ensure(idx)['a_ref'] = to_float(v)
            continue

        m = _A_POST_RE.match(k)
        if m:
            idx = int(m.group('idx'))
            _ensure(idx)['a_post'] = to_float(v)
            continue

        m = _A_CTRL_RE.match(k)
        if m:
            idx = int(m.group('idx'))
            _ensure(idx)['a_ctrl'] = to_float(v)
            continue

        m = _RHO_RE.match(k)
        if m:
            idx = int(m.group('idx'))
            _ensure(idx)['rho'] = to_float(v)
            continue

    return exp_metrics


def _merge_experience_artifacts(
    exp_metrics: dict[int, dict[str, Optional[float]]],
    artifacts: dict[str, Any],
) -> dict[int, dict[str, Optional[float]]]:
    """
    Merge analysis_artifacts.json vectors into a per-experience metrics dict.

    Expected artifact shape (best-effort):
        {
          "a_ref": [...],
          "a_post": [...],
          "a_ctrl": [...],
          "rho": [...],
          "rho_mean": float,
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

    def _ingest_vector(key: str) -> None:
        vec = artifacts.get(key)
        if not isinstance(vec, list):
            return
        for i, raw in enumerate(vec):
            idx = int(i)
            if idx not in exp_metrics:
                exp_metrics[idx] = {'a_ref': None, 'a_post': None, 'a_ctrl': None, 'rho': None}
            exp_metrics[idx][key] = to_float(raw)

    for k in ['a_ref', 'a_post', 'a_ctrl', 'rho']:
        _ingest_vector(k)

    return exp_metrics


def _first_present(metrics: dict[str, float], keys: list[str]) -> Optional[float]:
    """
    Return the first present metric value among candidates.

    Args:
        metrics: Run metrics.
        keys: Candidate metric keys in priority order.

    Returns:
        First present float or None.
    """
    for k in keys:
        if k in (metrics or {}):
            v = to_float(metrics.get(k))
            if v is not None:
                return v
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

    set_sqlite_tracking_uri(tracking_uri=tracking_uri)

    client = MlflowClient()
    experiment_id = resolve_experiment_id(
        client=client,
        experiment=experiment,
        prefer_name=True,
        raise_on_missing=True,
    )

    # Fetch runs with pagination (avoid brittle filter-string dependency on tags presence).
    parent_runs = search_runs_paginated(
        client=client,
        experiment_ids=[experiment_id],
        filter_string='',
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
        order_by=['attributes.start_time DESC'],
        max_runs=max_runs,
    )

    runs_table: list[dict[str, Any]] = []
    experiences_table: list[dict[str, Any]] = []

    for run in parent_runs:
        tags = dict(getattr(run.data, 'tags', {}) or {})
        if not _is_parent_run(tags):
            continue

        info = run.info
        if require_finished and str(getattr(info, 'status', '')) != 'FINISHED':
            continue

        params = dict(getattr(run.data, 'params', {}) or {})
        metrics = dict(getattr(run.data, 'metrics', {}) or {})

        controller_name = str(params.get('controller_name') or 'none')
        if include_controllers is not None and controller_name not in include_controllers:
            continue
        if exclude_controllers is not None and controller_name in exclude_controllers:
            continue

        seed = to_int(params.get('seed'))
        repair_budget_per_class = to_int(params.get('repair_budget_per_class'))
        num_classes = to_int(params.get('num_classes'))
        if num_classes is None:
            num_classes = int(default_num_classes) if default_num_classes is not None else None

        ctrl_param_count = to_int(params.get('controller_model_param_count'))

        b = float(repair_budget_per_class) if repair_budget_per_class is not None else None
        repair_budget_total = None
        if b is not None and num_classes is not None and num_classes > 0:
            repair_budget_total = int(float(b) * float(num_classes))

        # Prefer logged summary metrics; fallback to analysis summary; then compute from per-task.
        rho_mean = _first_present(metrics, ['summary-final_rho_mean', 'analysis-rho_mean'])
        a_ctrl_mean = _first_present(metrics, ['summary-final_a_ctrl_mean', 'analysis-a_ctrl_mean'])
        a_post_mean = _first_present(metrics, ['summary-final_a_post_mean', 'analysis-a_post_mean'])
        a_ref_mean = _first_present(metrics, ['summary-final_a_ref_mean', 'analysis-a_ref_mean'])

        experience_metrics = _extract_experience_metrics(metrics)

        # Fallback to artifact if per-experience metrics are empty or sparse.
        if not experience_metrics or any(v is None for row in experience_metrics.values() for v in row.values()):
            artifacts = _download_json_artifact(client, run_id=str(info.run_id), artifact_path='analysis_artifacts.json')
            if artifacts:
                experience_metrics = _merge_experience_artifacts(experience_metrics, artifacts)
                # Artifact may also include rho_mean directly
                if rho_mean is None:
                    rho_mean = to_float(artifacts.get('rho_mean'))

        # Compute missing summary fields from per-experience metrics if needed.
        if rho_mean is None:
            rho_mean = mean([row.get('rho') for row in experience_metrics.values()])
        if a_ctrl_mean is None:
            a_ctrl_mean = mean([row.get('a_ctrl') for row in experience_metrics.values()])
        if a_post_mean is None:
            a_post_mean = mean([row.get('a_post') for row in experience_metrics.values()])
        if a_ref_mean is None:
            a_ref_mean = mean([row.get('a_ref') for row in experience_metrics.values()])

        run_row: dict[str, Any] = {
            'run_id': str(info.run_id),
            'experiment_id': str(experiment_id),
            'run_name': str(params.get('run_name') or getattr(info, 'run_name', '') or ''),
            'scenario': str(params.get('scenario') or ''),
            'strategy_name': str(params.get('strategy_name') or ''),
            'eval_mode': str(params.get('eval_mode') or ''),
            'seed': seed,
            'controller_name': controller_name,
            'repair_budget_per_class': repair_budget_per_class,
            'repair_budget_total': repair_budget_total,
            'num_classes': num_classes,
            'b': b,
            'controller_model_param_count': ctrl_param_count,
            'rho_mean': rho_mean,
            'a_ctrl_mean': a_ctrl_mean,
            'a_post_mean': a_post_mean,
            'a_ref_mean': a_ref_mean,
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
                'run_id': str(info.run_id),
                'seed': seed,
                'controller_name': controller_name,
                'repair_budget_per_class': repair_budget_per_class,
                'repair_budget_total': repair_budget_total,
                'num_classes': num_classes,
                'b': b,
                'controller_model_param_count': ctrl_param_count,
                'exp_idx': int(exp_idx),
                'task_age': int(max_idx - exp_idx) if max_idx >= 0 else None,
                'a_ref': row.get('a_ref'),
                'a_post': row.get('a_post'),
                'a_ctrl': row.get('a_ctrl'),
                'rho': row.get('rho'),
            })

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
