"""
MLflow collectors for the REGAIN analysis tool.

This module converts MLflow runs into tidy tables suitable for automation of:
  - recoverability curves (ρ / recovered accuracy vs repair budget), and
  - repairability frontier summaries and selection datasets.
"""

import json
import math
from pathlib import Path
import re
import tarfile
import tempfile
from typing import Any, Optional, Sequence

import mlflow
from mlflow.tracking import MlflowClient

from regain.analysis.artifacts import ARTIFACT_ACC_EXP_BASE
from regain.analysis.artifacts import ARTIFACT_ACC_FINAL_BASE
from regain.analysis.artifacts import ARTIFACT_ACC_FINAL_CTRL
from regain.analysis.artifacts import ARTIFACT_RHO
from regain.analysis.utils import mean
from regain.analysis.utils import to_float
from regain.analysis.utils import to_int
from regain.constants import COLUMN_B
from regain.constants import COLUMN_CONTROLLER_MODEL_PARAM_COUNT
from regain.constants import COLUMN_CONTROLLER_NAME
from regain.constants import COLUMN_CONTROLLER_TYPE
from regain.constants import COLUMN_EXP_IDX
from regain.constants import COLUMN_EXPERIMENT_ID
from regain.constants import COLUMN_NUM_CLASSES
from regain.constants import COLUMN_REPAIR_BUDGET_FRACTION
from regain.constants import COLUMN_REPAIR_BUDGET_TOTAL
from regain.constants import COLUMN_REPAIR_SET_TOTAL
from regain.constants import COLUMN_REPAIR_SPLIT_FRACTION
from regain.constants import COLUMN_RUN_ID
from regain.constants import COLUMN_RUN_NAME
from regain.constants import COLUMN_SEED
from regain.constants import COLUMN_STATUS
from regain.constants import COLUMN_TASK_AGE
from regain.constants import DIAG_VECTOR_KEYS
from regain.constants import EXPERIENCE_KEY_PREFIX
from regain.constants import MLFLOW_ARTIFACT_ANALYSIS_FILE
from regain.constants import MLFLOW_ARTIFACT_CONFIG_FILE
from regain.constants import MLFLOW_ARTIFACT_SPLITS_FILE
from regain.constants import NS_SEP
from regain.constants import PARAM_BACKBONE_REPLAY_BATCH_SIZE_MEM
from regain.constants import PARAM_BACKBONE_REPLAY_MEM_SIZE
from regain.constants import PARAM_CONTROLLER_MODEL_PARAM_COUNT
from regain.constants import PARAM_CONTROLLER_TYPE
from regain.constants import PARAM_NUM_CLASSES
from regain.constants import PARAM_REPAIR_BUDGET_FRACTION
from regain.constants import PARAM_REPAIR_SPLIT_FRACTION
from regain.constants import PARAM_SEED
from regain.constants import RUN_ACC_FINAL
from regain.constants import RUN_ACC_FINAL_AVG_BASE
from regain.constants import RUN_ACC_FINAL_AVG_CTRL
from regain.constants import RUN_ACC_REF
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
from regain.constants import RUN_LATENCY_MS_PER_SAMPLE_BASE
from regain.constants import RUN_LATENCY_MS_PER_SAMPLE_CTRL
from regain.constants import RUN_LATENCY_MS_RATIO
from regain.constants import RUN_LATENCY_SAMPLES_PER_SEC_BASE
from regain.constants import RUN_LATENCY_SAMPLES_PER_SEC_CTRL
from regain.constants import RUN_REPAIR_SECONDS
from regain.constants import RUN_REPAIR_STEPS
from regain.constants import RUN_RHO
from regain.constants import RUN_RHO_AVG
from regain.constants import STREAM_REPAIR
from regain.experiments.config import load_experiment_config
from regain.mlflow_utils import download_json_artifact
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
_COLUMN_BACKBONE_NAME = 'backbone_name'

_PARAM_CONTROLLER_NAME = 'controller.name'
_CONTROLLER_TYPE_NONE = 'none'
_CONTROLLER_TYPE_PREVENTION = 'prevention'
_CONTROLLER_TYPE_REPAIR = 'repair'

# Per-experience metric keys used as dict keys in experience_metrics.
# These use the artifact JSON key names for core metrics (accuracy, rho)
# and RUN_* constants for extra metrics (calibration, diagnostics).
_EXPERIENCE_METRIC_KEYS = (
    ARTIFACT_ACC_EXP_BASE,
    ARTIFACT_ACC_FINAL_BASE,
    ARTIFACT_ACC_FINAL_CTRL,
    ARTIFACT_RHO,
    RUN_CALIB_ECE,
    RUN_CALIB_AECE,
    RUN_CALIB_MCE,
    RUN_CALIB_NLL,
    RUN_CALIB_BRIER,
    RUN_DIAG_OUT_OF_TASK_RATE,
    RUN_DIAG_AVG_CONF,
    RUN_DIAG_AVG_ENTROPY,
    RUN_DIAG_LOGIT_AVG_DRIFT,
)


_NS_SEP_ESCAPED = re.escape(NS_SEP)

# Regex patterns for parsing per-experience MLflow metric keys.
# Example keys:
# `run.eval.acc.ref.exp000.base`, `run.eval.acc.final.exp000.ctrl`,
# `run.repair.rho.exp000`, etc.
_ACC_EXP_BASE_RE = re.compile(
    rf'^{re.escape(RUN_ACC_REF)}'
    rf'{_NS_SEP_ESCAPED}{EXPERIENCE_KEY_PREFIX}(?P<idx>\d+)'
    rf'{_NS_SEP_ESCAPED}base$'
)
_ACC_FINAL_BASE_RE = re.compile(
    rf'^{re.escape(RUN_ACC_FINAL)}'
    rf'{_NS_SEP_ESCAPED}{EXPERIENCE_KEY_PREFIX}(?P<idx>\d+)'
    rf'{_NS_SEP_ESCAPED}base$'
)
_ACC_FINAL_CTRL_RE = re.compile(
    rf'^{re.escape(RUN_ACC_FINAL)}'
    rf'{_NS_SEP_ESCAPED}{EXPERIENCE_KEY_PREFIX}(?P<idx>\d+)'
    rf'{_NS_SEP_ESCAPED}ctrl$'
)
_RHO_RE = re.compile(
    rf'^{re.escape(RUN_RHO)}'
    rf'{_NS_SEP_ESCAPED}{EXPERIENCE_KEY_PREFIX}(?P<idx>\d+)$'
)
_EXTRA_EXPERIENCE_REGEXES: dict[str, re.Pattern[str]] = {
    metric_key: re.compile(
        rf'^{re.escape(metric_key)}'
        rf'{_NS_SEP_ESCAPED}{EXPERIENCE_KEY_PREFIX}(?P<idx>\d+)$'
    )
    for metric_key in (
        RUN_CALIB_ECE,
        RUN_CALIB_AECE,
        RUN_CALIB_MCE,
        RUN_CALIB_NLL,
        RUN_CALIB_BRIER,
        RUN_DIAG_OUT_OF_TASK_RATE,
        RUN_DIAG_AVG_CONF,
        RUN_DIAG_AVG_ENTROPY,
        RUN_DIAG_LOGIT_AVG_DRIFT,
    )
}
_REPAIR_SPLIT_FILE_RE = re.compile(
    rf'^{re.escape(STREAM_REPAIR)}'
    rf'/'
    rf'{re.escape(EXPERIENCE_KEY_PREFIX)}_\d+'
    rf'\.txt$',
)


def _extract_repair_set_total_from_splits_artifact(
    *,
    client: MlflowClient,
    run_id: str,
) -> int:
    """
    Extract exact repair set total size from the logged split archive.

    Args:
        client (MlflowClient): MLflow client.
        run_id (str): Run identifier.

    Returns:
        int: Exact total repair set size across repair experiences.

    Raises:
        ValueError: If the archive is missing, invalid, or has no repair split files.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        downloaded_path = client.download_artifacts(
            run_id,
            MLFLOW_ARTIFACT_SPLITS_FILE,
            temp_dir,
        )
        local_path = Path(downloaded_path)
        if local_path.is_dir():
            local_path = local_path / MLFLOW_ARTIFACT_SPLITS_FILE
        if not local_path.exists():
            raise ValueError(
                f'Run `{run_id}` is missing required split archive `{MLFLOW_ARTIFACT_SPLITS_FILE}`.'
            )

        total = 0
        found_any = False
        with tarfile.open(local_path, 'r:gz') as archive:
            for member in archive.getmembers():
                member_name = str(member.name)
                if not member.isfile():
                    continue
                if not _REPAIR_SPLIT_FILE_RE.match(member_name):
                    continue
                found_any = True
                extracted_file = archive.extractfile(member)
                if extracted_file is None:
                    raise ValueError(
                        f'Run `{run_id}` has unreadable repair split file `{member_name}`.'
                    )
                payload = extracted_file.read().decode('utf-8')
                total += sum(1 for line in payload.splitlines() if line.strip() != '')
        if not found_any:
            raise ValueError(
                f'Run `{run_id}` split archive `{MLFLOW_ARTIFACT_SPLITS_FILE}` '
                'contains no repair split files.'
            )
        return int(total)


def _extract_analysis_identity_from_config_artifact(
    *,
    client: MlflowClient,
    run_id: str,
) -> tuple[str, str, Optional[int], Optional[int]]:
    """
    Extract analysis identity from a run's logged experiment config.

    Args:
        client (MlflowClient): MLflow client.
        run_id (str): Run identifier.

    Returns:
        tuple[str, str, Optional[int], Optional[int]]: Backbone name, backbone training
            strategy name, replay memory size, and replay memory batch size. The replay
            fields are `None` when the configured strategy is not a replay-based one.

    Raises:
        ValueError: If `config.yaml` is missing, cannot be parsed by the official
            experiment config parser, or lacks resolved backbone/strategy identity.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        downloaded_path = client.download_artifacts(
            run_id,
            MLFLOW_ARTIFACT_CONFIG_FILE,
            temp_dir,
        )
        config_path = Path(downloaded_path)
        if config_path.is_dir():
            config_path = config_path / MLFLOW_ARTIFACT_CONFIG_FILE
        if not config_path.exists():
            raise ValueError(
                f'Run `{run_id}` is missing required config artifact `{MLFLOW_ARTIFACT_CONFIG_FILE}`.'
            )

        try:
            experiment_config = load_experiment_config(config_path)
        except Exception as exc:
            raise ValueError(
                f'Run `{run_id}` config artifact `{MLFLOW_ARTIFACT_CONFIG_FILE}` '
                f'could not be parsed by `load_experiment_config`: {exc}'
            ) from exc

        if experiment_config.backbone is None:
            raise ValueError(
                f'Run `{run_id}` config artifact `{MLFLOW_ARTIFACT_CONFIG_FILE}` '
                'is missing required `backbone`.'
            )
        backbone_name = str(experiment_config.backbone.name or '').strip()
        if not backbone_name:
            raise ValueError(
                f'Run `{run_id}` config artifact `{MLFLOW_ARTIFACT_CONFIG_FILE}` '
                'is missing required `backbone.name`.'
            )

        if experiment_config.backbone.training is None:
            raise ValueError(
                f'Run `{run_id}` config artifact `{MLFLOW_ARTIFACT_CONFIG_FILE}` '
                'is missing required `backbone.training`.'
            )
        strategy = getattr(experiment_config.backbone.training, 'strategy', None)
        strategy_name = str(getattr(strategy, 'name', '') or '').strip()
        if not strategy_name:
            raise ValueError(
                f'Run `{run_id}` config artifact `{MLFLOW_ARTIFACT_CONFIG_FILE}` '
                'is missing required `backbone.training.strategy.name`.'
            )

        strategy_kwargs = getattr(strategy, 'kwargs', None) or {}
        replay_mem_size = to_int(strategy_kwargs.get('mem_size'))
        replay_batch_size_mem = to_int(strategy_kwargs.get('batch_size_mem'))

        return backbone_name, strategy_name, replay_mem_size, replay_batch_size_mem


def _extract_experience_metrics(metrics: dict[str, float]) -> dict[int, dict[str, Optional[float]]]:
    """
    Extract per-experience analysis metrics from run metrics.

    Args:
        metrics: Run metrics dict.

    Returns:
        Dict mapping experience index -> dict with per-experience metric values.
    """
    exp_metrics: dict[int, dict[str, Optional[float]]] = {}

    def _ensure(idx: int) -> dict[str, Optional[float]]:
        if idx not in exp_metrics:
            exp_metrics[idx] = {metric_key: None for metric_key in _EXPERIENCE_METRIC_KEYS}
        return exp_metrics[idx]

    for k, v in (metrics or {}).items():
        m = _ACC_EXP_BASE_RE.match(k)
        if m:
            idx = int(m.group('idx'))
            _ensure(idx)[ARTIFACT_ACC_EXP_BASE] = to_float(v)
            continue

        m = _ACC_FINAL_BASE_RE.match(k)
        if m:
            idx = int(m.group('idx'))
            _ensure(idx)[ARTIFACT_ACC_FINAL_BASE] = to_float(v)
            continue

        m = _ACC_FINAL_CTRL_RE.match(k)
        if m:
            idx = int(m.group('idx'))
            _ensure(idx)[ARTIFACT_ACC_FINAL_CTRL] = to_float(v)
            continue

        m = _RHO_RE.match(k)
        if m:
            idx = int(m.group('idx'))
            _ensure(idx)[ARTIFACT_RHO] = to_float(v)
            continue

        matched = False
        for metric_key, regex in _EXTRA_EXPERIENCE_REGEXES.items():
            m = regex.match(k)
            if m:
                idx = int(m.group('idx'))
                _ensure(idx)[metric_key] = to_float(v)
                matched = True
                break
        if matched:
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
          "<artifact_vector_key>": [...],
          "<artifact_scalar_key>": float,
          ...
        }

    Args:
        exp_metrics: Existing per-experience metrics (possibly sparse).
        artifacts: Parsed analysis artifacts.
        include_keys: Optional list of artifact keys to ingest. Defaults to all experience keys.

    Returns:
        Updated per-experience metrics dict.
    """
    if not artifacts:
        return exp_metrics

    artifact_keys = (
        list(include_keys)
        if include_keys is not None
        else list(_EXPERIENCE_METRIC_KEYS)
    )

    def _ingest_vector(key: str) -> None:
        vec = artifacts.get(key)
        if not isinstance(vec, list):
            return
        for i, raw in enumerate(vec):
            idx = int(i)
            if idx not in exp_metrics:
                exp_metrics[idx] = {metric_key: None for metric_key in _EXPERIENCE_METRIC_KEYS}
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
        RUN_ACC_FINAL_AVG_CTRL in metrics
        or RUN_RHO_AVG in metrics
    ):
        return True

    for values in experience_metrics.values():
        if values.get(ARTIFACT_ACC_FINAL_CTRL) is not None or values.get(ARTIFACT_RHO) is not None:
            return True
    return False


def _validate_required_repair_diagnostics(
    *,
    run_id: str,
    experience_metrics: dict[int, dict[str, Optional[float]]],
) -> None:
    """
    Validate that repair runs include complete baseline diagnostic vectors.

    Args:
        run_id (str): Run identifier.
        experience_metrics (dict[int, dict[str, Optional[float]]]): Per-experience metrics.

    Raises:
        ValueError: If any diagnostic vector is missing for one or more experiences.
    """
    if not experience_metrics:
        raise ValueError(
            f'Repair run `{run_id}` is missing per-experience metrics required for diagnostic validation.'
        )

    missing_by_key: dict[str, list[int]] = {}
    for exp_idx, values in sorted(experience_metrics.items()):
        for diag_key in DIAG_VECTOR_KEYS:
            if values.get(diag_key) is None:
                missing_by_key.setdefault(diag_key, []).append(int(exp_idx))

    if missing_by_key:
        details = ', '.join(
            (
                f'{metric_key}: '
                f'{"/".join(f"exp{idx:03d}" for idx in missing_indices)}'
            )
            for metric_key, missing_indices in sorted(missing_by_key.items())
        )
        raise ValueError(
            f'Repair run `{run_id}` is missing required baseline diagnostic metrics '
            f'in `analysis_artifacts.json`: {details}.'
        )


def _controller_expects_ctrl_metrics(*, params: dict[str, Any]) -> bool:
    """
    Infer whether a run should expose repair-controller analysis metrics.

    Args:
        params: Run params dictionary.

    Returns:
        bool: True for repair controllers, False otherwise.

    Raises:
        ValueError: If `controller.type` is missing or invalid.
    """
    controller_type = str(params.get(PARAM_CONTROLLER_TYPE) or '').strip().lower()
    if controller_type == _CONTROLLER_TYPE_REPAIR:
        return True
    if controller_type in {_CONTROLLER_TYPE_PREVENTION, _CONTROLLER_TYPE_NONE}:
        return False
    raise ValueError(
        'Run is missing a valid `controller.type` param. '
        'Expected one of: `repair`, `prevention`, `none`.'
    )


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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    """
    Collect run tables for the analysis tool.

    Produces:
      - runs_table: one row per run
      - experiences_table: one row per experience (run, experience index)
      - run_failures: one row per skipped run with run-level failure details

    Args:
        experiment: MLflow experiment name or id.
        out_dir: Optional directory to also write `run_metrics.jsonl` and `experience_metrics.jsonl`.
        tracking_uri: Optional MLflow tracking URI.
        include_controllers: Optional allowlist of controller_name values.
        exclude_controllers: Optional denylist of controller_name values.
        max_runs: Optional limit on number of runs to load.
        require_finished: If True, keep only runs with status FINISHED.
        default_num_classes: Fallback number of classes when not logged.

    Returns:
        tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
            Run table, experience table, and run-level failure summaries.

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
        return [], [], []

    # Fetch runs with pagination (avoid brittle filter-string dependency on tags presence).
    candidate_runs = search_runs_paginated(
        client=client,
        experiment_ids=[experiment_id],
        filter_string='',
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
        order_by=['attributes.start_time DESC'],
    )

    runs_table: list[dict[str, Any]] = []
    experiences_table: list[dict[str, Any]] = []
    run_failures: list[dict[str, str]] = []
    repair_set_total_cache: dict[
        tuple[str, int | None, float | None, int | None, float | None],
        int,
    ] = {}

    for run in candidate_runs:
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
        try:
            seed = to_int(params.get(PARAM_SEED))
            repair_budget_fraction = to_float(params.get(PARAM_REPAIR_BUDGET_FRACTION))
            repair_split_fraction = to_float(params.get(PARAM_REPAIR_SPLIT_FRACTION))
            num_classes = to_int(params.get(PARAM_NUM_CLASSES))
            if num_classes is None:
                num_classes = int(default_num_classes) if default_num_classes is not None else None

            ctrl_param_count = to_int(params.get(PARAM_CONTROLLER_MODEL_PARAM_COUNT))

            b = float(repair_budget_fraction) if repair_budget_fraction is not None else None
            repair_budget_total: int | None = None

            # Prefer direct aggregate metrics, then compute from per-task values.
            rho_avg = to_float(metrics.get(RUN_RHO_AVG))
            a_ctrl_avg = to_float(metrics.get(RUN_ACC_FINAL_AVG_CTRL))
            a_base_avg = to_float(metrics.get(RUN_ACC_FINAL_AVG_BASE))
            calib_max_ece = to_float(metrics.get(RUN_CALIB_MAX_ECE))
            latency_base_ms = to_float(metrics.get(RUN_LATENCY_MS_PER_SAMPLE_BASE))
            latency_base_sps = to_float(metrics.get(RUN_LATENCY_SAMPLES_PER_SEC_BASE))
            latency_ctrl_ms = to_float(metrics.get(RUN_LATENCY_MS_PER_SAMPLE_CTRL))
            latency_ctrl_sps = to_float(metrics.get(RUN_LATENCY_SAMPLES_PER_SEC_CTRL))
            latency_ms_ratio = to_float(metrics.get(RUN_LATENCY_MS_RATIO))
            repair_seconds = to_float(metrics.get(RUN_REPAIR_SECONDS))
            repair_steps = to_float(metrics.get(RUN_REPAIR_STEPS))

            (
                backbone_name,
                strategy_name,
                config_replay_mem_size,
                config_replay_batch_size_mem,
            ) = _extract_analysis_identity_from_config_artifact(
                client=client,
                run_id=str(info.run_id),
            )
            replay_mem_size = to_int(params.get(PARAM_BACKBONE_REPLAY_MEM_SIZE))
            if replay_mem_size is None:
                replay_mem_size = config_replay_mem_size
            replay_batch_size_mem = to_int(params.get(PARAM_BACKBONE_REPLAY_BATCH_SIZE_MEM))
            if replay_batch_size_mem is None:
                replay_batch_size_mem = config_replay_batch_size_mem

            experience_metrics = _extract_experience_metrics(metrics)
            has_logged_ctrl_metrics = _has_logged_ctrl_metrics(
                metrics=metrics,
                experience_metrics=experience_metrics,
            )
            is_repair_controller = _controller_expects_ctrl_metrics(params=params)
            if is_repair_controller and b is None:
                raise ValueError(
                    'Repair run is missing required `repair.budget_fraction` param.'
                )
            expects_ctrl_metrics = bool(has_logged_ctrl_metrics)
            if is_repair_controller:
                expects_ctrl_metrics = True

            repair_set_total: int | None
            if repair_split_fraction is None:
                raise ValueError(
                    'Run is missing required `repair.split_fraction` param. '
                    'This parameter is required for analysis table derivations.'
                )
            if repair_split_fraction < 0.0:
                raise ValueError('`repair.split_fraction` must be non-negative.')
            if repair_split_fraction <= 0.0:
                repair_set_total = 0
            else:
                cache_key = (
                    str(info.run_id),
                    seed,
                    repair_split_fraction,
                    num_classes,
                    repair_budget_fraction,
                )
                if cache_key in repair_set_total_cache:
                    repair_set_total = repair_set_total_cache[cache_key]
                else:
                    repair_set_total = _extract_repair_set_total_from_splits_artifact(
                        client=client,
                        run_id=str(info.run_id),
                    )
                    repair_set_total_cache[cache_key] = int(repair_set_total)
            if b is not None and repair_set_total is not None:
                repair_budget_total = int(math.floor(float(repair_set_total) * float(b)))

            artifacts: dict[str, Any] | None = None
            if is_repair_controller:
                # Baseline-only policy: repair-run diagnostic features must come from
                # controller-off artifact vectors, not controller-on eval metrics.
                for values in experience_metrics.values():
                    for diag_key in DIAG_VECTOR_KEYS:
                        values[diag_key] = None
                artifacts = download_json_artifact(
                    client=client,
                    run_id=str(info.run_id),
                    artifact_path=MLFLOW_ARTIFACT_ANALYSIS_FILE,
                )
                if not isinstance(artifacts, dict):
                    raise ValueError(
                        f'Repair run `{info.run_id}` is missing required `{MLFLOW_ARTIFACT_ANALYSIS_FILE}`.'
                    )
                calib_max_ece = to_float(artifacts.get(RUN_CALIB_MAX_ECE))
                if calib_max_ece is None:
                    raise ValueError(
                        f'Repair run `{info.run_id}` is missing required '
                        f'`{RUN_CALIB_MAX_ECE}` in `{MLFLOW_ARTIFACT_ANALYSIS_FILE}`.'
                    )
                experience_metrics = _merge_experience_artifacts(
                    experience_metrics,
                    artifacts,
                    include_keys=list(DIAG_VECTOR_KEYS),
                )

            required_experience_keys = [ARTIFACT_ACC_EXP_BASE, ARTIFACT_ACC_FINAL_BASE]
            if expects_ctrl_metrics:
                required_experience_keys.append(ARTIFACT_ACC_FINAL_CTRL)

            if not experience_metrics:
                raise ValueError(
                    f'Run `{info.run_id}` is missing required per-experience analysis metrics.'
                )

            missing_experience_metrics: list[str] = []
            for exp_idx, row in sorted(experience_metrics.items()):
                for key in required_experience_keys:
                    if row.get(key) is None:
                        missing_experience_metrics.append(f'exp{exp_idx:03d}:{key}')
            if missing_experience_metrics:
                details = ', '.join(missing_experience_metrics)
                raise ValueError(
                    f'Run `{info.run_id}` is missing required per-experience metrics: {details}.'
                )

            if is_repair_controller:
                _validate_required_repair_diagnostics(
                    run_id=str(info.run_id),
                    experience_metrics=experience_metrics,
                )

            if calib_max_ece is None:
                raise ValueError(
                    f'Run `{info.run_id}` is missing required `{RUN_CALIB_MAX_ECE}` metric.'
                )

            # Compute missing summary fields from per-experience metrics if needed.
            if rho_avg is None and expects_ctrl_metrics:
                rho_avg = mean([row.get(ARTIFACT_RHO) for row in experience_metrics.values()])
            if a_ctrl_avg is None and expects_ctrl_metrics:
                a_ctrl_avg = mean([row.get(ARTIFACT_ACC_FINAL_CTRL) for row in experience_metrics.values()])
            if a_base_avg is None:
                a_base_avg = mean([row.get(ARTIFACT_ACC_FINAL_BASE) for row in experience_metrics.values()])

            run_row: dict[str, Any] = {
                COLUMN_RUN_ID: str(info.run_id),
                COLUMN_EXPERIMENT_ID: str(experiment_id),
                COLUMN_RUN_NAME: run_name,
                _COLUMN_SCENARIO: str(params.get(_COLUMN_SCENARIO) or ''),
                _COLUMN_BACKBONE_NAME: backbone_name,
                _COLUMN_STRATEGY_NAME: strategy_name,
                COLUMN_SEED: seed,
                COLUMN_CONTROLLER_NAME: controller_name,
                COLUMN_CONTROLLER_TYPE: str(params.get(PARAM_CONTROLLER_TYPE) or '').strip().lower(),
                COLUMN_REPAIR_BUDGET_FRACTION: repair_budget_fraction,
                COLUMN_REPAIR_BUDGET_TOTAL: repair_budget_total,
                COLUMN_REPAIR_SET_TOTAL: repair_set_total,
                COLUMN_REPAIR_SPLIT_FRACTION: repair_split_fraction,
                COLUMN_NUM_CLASSES: num_classes,
                COLUMN_B: b,
                COLUMN_CONTROLLER_MODEL_PARAM_COUNT: ctrl_param_count,
                'replay_mem_size': replay_mem_size,
                'replay_batch_size_mem': replay_batch_size_mem,
                RUN_RHO_AVG: rho_avg,
                RUN_ACC_FINAL_AVG_CTRL: a_ctrl_avg,
                RUN_ACC_FINAL_AVG_BASE: a_base_avg,
                RUN_CALIB_MAX_ECE: calib_max_ece,
                RUN_LATENCY_MS_PER_SAMPLE_BASE: latency_base_ms,
                RUN_LATENCY_SAMPLES_PER_SEC_BASE: latency_base_sps,
                RUN_LATENCY_MS_PER_SAMPLE_CTRL: latency_ctrl_ms,
                RUN_LATENCY_SAMPLES_PER_SEC_CTRL: latency_ctrl_sps,
                RUN_LATENCY_MS_RATIO: latency_ms_ratio,
                RUN_REPAIR_SECONDS: repair_seconds,
                RUN_REPAIR_STEPS: repair_steps,
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
                    COLUMN_CONTROLLER_TYPE: str(params.get(PARAM_CONTROLLER_TYPE) or '').strip().lower(),
                    COLUMN_REPAIR_BUDGET_FRACTION: repair_budget_fraction,
                    COLUMN_REPAIR_BUDGET_TOTAL: repair_budget_total,
                    COLUMN_REPAIR_SET_TOTAL: repair_set_total,
                    COLUMN_REPAIR_SPLIT_FRACTION: repair_split_fraction,
                    COLUMN_NUM_CLASSES: num_classes,
                    COLUMN_B: b,
                    COLUMN_CONTROLLER_MODEL_PARAM_COUNT: ctrl_param_count,
                    COLUMN_EXP_IDX: int(exp_idx),
                    COLUMN_TASK_AGE: int(max_idx - exp_idx) if max_idx >= 0 else None,
                    ARTIFACT_ACC_EXP_BASE: row.get(ARTIFACT_ACC_EXP_BASE),
                    ARTIFACT_ACC_FINAL_BASE: row.get(ARTIFACT_ACC_FINAL_BASE),
                    ARTIFACT_ACC_FINAL_CTRL: row.get(ARTIFACT_ACC_FINAL_CTRL),
                    ARTIFACT_RHO: row.get(ARTIFACT_RHO),
                    RUN_CALIB_ECE: row.get(RUN_CALIB_ECE),
                    RUN_CALIB_AECE: row.get(RUN_CALIB_AECE),
                    RUN_CALIB_MCE: row.get(RUN_CALIB_MCE),
                    RUN_CALIB_NLL: row.get(RUN_CALIB_NLL),
                    RUN_CALIB_BRIER: row.get(RUN_CALIB_BRIER),
                    RUN_DIAG_OUT_OF_TASK_RATE: row.get(RUN_DIAG_OUT_OF_TASK_RATE),
                    RUN_DIAG_AVG_CONF: row.get(RUN_DIAG_AVG_CONF),
                    RUN_DIAG_AVG_ENTROPY: row.get(RUN_DIAG_AVG_ENTROPY),
                    RUN_DIAG_LOGIT_AVG_DRIFT: row.get(RUN_DIAG_LOGIT_AVG_DRIFT),
                })

            if max_runs is not None and len(runs_table) >= int(max_runs):
                break
        except Exception as exc:
            run_failures.append({
                'run_id': str(info.run_id),
                'run_name': run_name,
                'error': str(exc),
            })
            logger.warning(f'Skipping run `{info.run_id}` due to analysis collection error: {exc}')
            continue

    # Optional writeout (JSONL for robustness; analysis tool scripts write CSV).
    if out_dir is not None:
        outp = Path(out_dir)
        outp.mkdir(parents=True, exist_ok=True)

        (outp / 'run_metrics.jsonl').write_text(
            '\n'.join(json.dumps(r, default=str) for r in runs_table) + ('\n' if runs_table else ''),
            encoding='utf-8',
        )
        (outp / 'experience_metrics.jsonl').write_text(
            '\n'.join(json.dumps(r, default=str) for r in experiences_table) + ('\n' if experiences_table else ''),
            encoding='utf-8',
        )
        logger.warning(f'Wrote tables to {outp}')

    return runs_table, experiences_table, run_failures
