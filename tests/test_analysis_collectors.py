"""
Tests for analysis collectors.
"""

import io
from pathlib import Path
import tarfile
from types import SimpleNamespace
from typing import Any

import pytest

import regain.analysis.collectors as collectors_module
from regain.analysis.collectors import _extract_repair_set_total_from_splits_artifact
from regain.analysis.collectors import collect_experiment_tables
from regain.constants import COLUMN_REPAIR_SET_TOTAL
from regain.constants import RUN_CALIB_AECE
from regain.constants import RUN_CALIB_ECE
from regain.constants import RUN_CALIB_MAX_ECE
from regain.constants import RUN_CALIB_NLL
from regain.constants import RUN_DIAG_LOGIT_AVG_DRIFT
from regain.constants import RUN_DIAG_AVG_CONF
from regain.constants import RUN_DIAG_AVG_ENTROPY
from regain.constants import RUN_DIAG_OUT_OF_TASK_RATE
from regain.constants import PARAM_CONTROLLER_TYPE


def _make_run(
    *,
    params: dict[str, str],
    metrics: dict[str, float],
) -> SimpleNamespace:
    return SimpleNamespace(
        info=SimpleNamespace(
            run_id='run_1',
            status='FINISHED',
        ),
        data=SimpleNamespace(
            params=params,
            metrics=metrics,
        ),
    )


def _patch_collectors(
    *,
    monkeypatch: pytest.MonkeyPatch,
    runs: list[SimpleNamespace],
    artifact_payload: dict[str, Any] | None,
    client_factory: Any = None,
) -> None:
    monkeypatch.setattr(
        collectors_module,
        'set_tracking_uri',
        lambda *, tracking_uri: None,
    )
    monkeypatch.setattr(
        collectors_module,
        'MlflowClient',
        client_factory if client_factory is not None else (lambda: object()),
    )
    monkeypatch.setattr(
        collectors_module,
        'resolve_experiment_id',
        lambda *, client, experiment: 'exp_1',
    )
    monkeypatch.setattr(
        collectors_module,
        'search_runs_paginated',
        lambda **kwargs: runs,
    )
    monkeypatch.setattr(
        collectors_module,
        'resolve_mlflow_run_name',
        lambda *, run: 'mock_run',
    )
    monkeypatch.setattr(
        collectors_module,
        'download_json_artifact',
        lambda **kwargs: artifact_payload,
    )


def _base_metrics_with_exp000() -> dict[str, float]:
    return {
        'run.accuracy.exp.exp000.base': 0.80,
        'run.accuracy.final.exp000.base': 0.55,
        RUN_CALIB_MAX_ECE: 0.50,
    }


def _write_splits_archive(
    *,
    tmp_path: Path,
    repair_indices_by_exp: dict[int, list[int]],
    extra_files: dict[str, str] | None = None,
) -> Path:
    archive_path = tmp_path / 'splits.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as archive:
        for exp_idx, indices in repair_indices_by_exp.items():
            payload = '\n'.join(str(index) for index in indices)
            if payload:
                payload += '\n'
            payload_bytes = payload.encode('utf-8')
            info = tarfile.TarInfo(name=f'repair/exp_{exp_idx:03d}.txt')
            info.size = len(payload_bytes)
            archive.addfile(info, io.BytesIO(payload_bytes))
        for entry_name, payload in (extra_files or {}).items():
            payload_bytes = payload.encode('utf-8')
            info = tarfile.TarInfo(name=entry_name)
            info.size = len(payload_bytes)
            archive.addfile(info, io.BytesIO(payload_bytes))
    return archive_path


class _FakeMlflowClient:
    def __init__(self, *, splits_archive_path: Path) -> None:
        self._splits_archive_path = Path(splits_archive_path)

    def download_artifacts(
        self,
        run_id: str,
        artifact_path: str,
        dst_path: str,
    ) -> str:
        del run_id, artifact_path, dst_path
        return str(self._splits_archive_path)


class _MissingSplitsMlflowClient:
    def download_artifacts(
        self,
        run_id: str,
        artifact_path: str,
        dst_path: str,
    ) -> str:
        del run_id, artifact_path, dst_path
        return '/tmp/nonexistent_splits_archive.tar.gz'


class TestRepairSetTotalExtraction:
    def test_extracts_exact_total_from_splits_archive(self, tmp_path: Path) -> None:
        archive_path = _write_splits_archive(
            tmp_path=tmp_path,
            repair_indices_by_exp={
                0: [0, 1, 2],
                1: [10, 11],
            },
            extra_files={
                'repair/readme.txt': '999\n',
                'repair/exp_abc.txt': '9\n',
                'train/exp_000.txt': '8\n',
            },
        )
        client = _FakeMlflowClient(splits_archive_path=archive_path)

        repair_set_total = _extract_repair_set_total_from_splits_artifact(
            client=client,
            run_id='run_1',
        )

        assert repair_set_total == 5

    def test_raises_when_splits_archive_is_missing(self) -> None:
        client = _MissingSplitsMlflowClient()

        with pytest.raises(ValueError, match='split archive'):
            _extract_repair_set_total_from_splits_artifact(
                client=client,
                run_id='run_1',
            )


class TestCollectExperimentTablesPredictiveBaselinePolicy:
    def test_repair_run_policy_uses_logged_controller_type(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        params = {
            'controller.name': 'my_custom_controller',
            'controller.path': 'vendor.controllers.custom_controller',
            PARAM_CONTROLLER_TYPE: 'repair',
            'seed': '1',
            'repair.split_fraction': '0.0',
            'repair.budget_per_class': '5',
            'num_classes': '2',
        }
        metrics = _base_metrics_with_exp000()
        metrics.update({
            'run.accuracy.final.exp000.ctrl': 0.60,
            'run.repair.rho.exp000': 0.20,
            RUN_CALIB_MAX_ECE: 0.95,
            RUN_CALIB_ECE + '.exp000': 0.99,
            RUN_CALIB_AECE + '.exp000': 0.88,
            RUN_CALIB_NLL + '.exp000': 3.33,
        })
        artifact_payload = {
            RUN_CALIB_MAX_ECE: 0.31,
            RUN_CALIB_ECE: [0.11],
            RUN_CALIB_AECE: [0.12],
            RUN_CALIB_NLL: [0.13],
            RUN_DIAG_OUT_OF_TASK_RATE: [0.21],
            RUN_DIAG_AVG_CONF: [0.22],
            RUN_DIAG_AVG_ENTROPY: [0.23],
            RUN_DIAG_LOGIT_AVG_DRIFT: [0.24],
        }
        run = _make_run(params=params, metrics=metrics)
        _patch_collectors(
            monkeypatch=monkeypatch,
            runs=[run],
            artifact_payload=artifact_payload,
        )

        runs_table, experiences_table, run_failures = collect_experiment_tables(experiment='exp_name')

        assert len(runs_table) == 1
        assert runs_table[0][RUN_CALIB_MAX_ECE] == pytest.approx(0.31)
        assert len(experiences_table) == 1
        assert not run_failures
        row = experiences_table[0]
        assert row[RUN_CALIB_ECE] == pytest.approx(0.11)
        assert row[RUN_CALIB_AECE] == pytest.approx(0.12)
        assert row[RUN_CALIB_NLL] == pytest.approx(0.13)
        assert row[RUN_DIAG_OUT_OF_TASK_RATE] == pytest.approx(0.21)
        assert row[RUN_DIAG_AVG_CONF] == pytest.approx(0.22)
        assert row[RUN_DIAG_AVG_ENTROPY] == pytest.approx(0.23)
        assert row[RUN_DIAG_LOGIT_AVG_DRIFT] == pytest.approx(0.24)

    def test_repair_run_diagnostics_are_loaded_and_calib_max_comes_from_artifact_scalar(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        params = {
            'controller.name': 'my_repair_controller',
            'controller.path': 'regain.controllers.repair.mock_controller',
            PARAM_CONTROLLER_TYPE: 'repair',
            'seed': '1',
            'repair.split_fraction': '0.0',
            'repair.budget_per_class': '5',
            'num_classes': '2',
        }
        metrics = _base_metrics_with_exp000()
        metrics.update({
            'run.accuracy.final.exp000.ctrl': 0.60,
            'run.repair.rho.exp000': 0.20,
            RUN_CALIB_MAX_ECE: 0.95,
            RUN_CALIB_ECE + '.exp000': 0.99,
            RUN_CALIB_AECE + '.exp000': 0.88,
            RUN_CALIB_NLL + '.exp000': 3.33,
        })
        artifact_payload = {
            RUN_CALIB_MAX_ECE: 0.10,
            RUN_CALIB_ECE: [0.11],
            RUN_CALIB_AECE: [0.12],
            RUN_CALIB_NLL: [0.13],
            RUN_DIAG_OUT_OF_TASK_RATE: [0.21],
            RUN_DIAG_AVG_CONF: [0.22],
            RUN_DIAG_AVG_ENTROPY: [0.23],
            RUN_DIAG_LOGIT_AVG_DRIFT: [0.24],
        }
        run = _make_run(params=params, metrics=metrics)
        _patch_collectors(
            monkeypatch=monkeypatch,
            runs=[run],
            artifact_payload=artifact_payload,
        )

        runs_table, experiences_table, run_failures = collect_experiment_tables(experiment='exp_name')

        assert len(runs_table) == 1
        assert runs_table[0][RUN_CALIB_MAX_ECE] == pytest.approx(0.10)
        assert len(experiences_table) == 1
        assert not run_failures
        row = experiences_table[0]
        assert row[RUN_CALIB_ECE] == pytest.approx(0.11)
        assert row[RUN_CALIB_AECE] == pytest.approx(0.12)
        assert row[RUN_CALIB_NLL] == pytest.approx(0.13)
        assert row[RUN_DIAG_OUT_OF_TASK_RATE] == pytest.approx(0.21)
        assert row[RUN_DIAG_AVG_CONF] == pytest.approx(0.22)
        assert row[RUN_DIAG_AVG_ENTROPY] == pytest.approx(0.23)
        assert row[RUN_DIAG_LOGIT_AVG_DRIFT] == pytest.approx(0.24)

    def test_repair_run_missing_analysis_artifacts_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        params = {
            'controller.name': 'my_repair_controller',
            'controller.path': 'regain.controllers.repair.mock_controller',
            PARAM_CONTROLLER_TYPE: 'repair',
            'seed': '1',
            'repair.split_fraction': '0.0',
            'repair.budget_per_class': '5',
            'num_classes': '2',
        }
        metrics = _base_metrics_with_exp000()
        metrics.update({
            'run.accuracy.final.exp000.ctrl': 0.60,
            'run.repair.rho.exp000': 0.20,
            RUN_CALIB_MAX_ECE: 0.95,
            RUN_CALIB_ECE + '.exp000': 0.99,
            RUN_CALIB_AECE + '.exp000': 0.88,
            RUN_CALIB_NLL + '.exp000': 3.33,
        })
        run = _make_run(params=params, metrics=metrics)
        _patch_collectors(
            monkeypatch=monkeypatch,
            runs=[run],
            artifact_payload=None,
        )

        runs_table, experiences_table, run_failures = collect_experiment_tables(experiment='exp_name')
        assert not runs_table
        assert not experiences_table
        assert len(run_failures) == 1
        assert 'analysis_artifacts.json' in run_failures[0]['error']

    def test_repair_run_missing_diagnostic_vectors_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        params = {
            'controller.name': 'my_repair_controller',
            'controller.path': 'regain.controllers.repair.mock_controller',
            PARAM_CONTROLLER_TYPE: 'repair',
            'seed': '1',
            'repair.split_fraction': '0.0',
            'repair.budget_per_class': '5',
            'num_classes': '2',
        }
        metrics = _base_metrics_with_exp000()
        metrics.update({
            'run.accuracy.final.exp000.ctrl': 0.60,
            'run.repair.rho.exp000': 0.20,
            RUN_CALIB_MAX_ECE: 0.95,
            RUN_CALIB_ECE + '.exp000': 0.99,
            RUN_CALIB_AECE + '.exp000': 0.88,
            RUN_CALIB_NLL + '.exp000': 3.33,
        })
        _patch_collectors(
            monkeypatch=monkeypatch,
            runs=[_make_run(params=params, metrics=metrics)],
            artifact_payload={
                RUN_CALIB_MAX_ECE: 0.61,
                RUN_CALIB_ECE: [0.11],
            },
        )

        runs_table, experiences_table, run_failures = collect_experiment_tables(experiment='exp_name')
        assert not runs_table
        assert not experiences_table
        assert len(run_failures) == 1
        assert 'required baseline diagnostic metrics' in run_failures[0]['error']

    def test_non_repair_run_keeps_logged_diagnostics(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        params = {
            'controller.name': 'my_prevention_controller',
            'controller.path': 'regain.controllers.prevention.mock_controller',
            PARAM_CONTROLLER_TYPE: 'prevention',
            'seed': '1',
            'repair.split_fraction': '0.0',
            'repair.budget_per_class': '5',
            'num_classes': '2',
        }
        metrics = _base_metrics_with_exp000()
        metrics.update({
            RUN_CALIB_MAX_ECE: 0.44,
            RUN_CALIB_ECE + '.exp000': 0.77,
            RUN_CALIB_AECE + '.exp000': 0.66,
            RUN_CALIB_NLL + '.exp000': 0.55,
        })
        artifact_payload = {
            RUN_CALIB_ECE: [0.11],
            RUN_CALIB_AECE: [0.12],
            RUN_CALIB_NLL: [0.13],
            RUN_DIAG_OUT_OF_TASK_RATE: [0.21],
            RUN_DIAG_AVG_CONF: [0.22],
            RUN_DIAG_AVG_ENTROPY: [0.23],
            RUN_DIAG_LOGIT_AVG_DRIFT: [0.24],
        }
        run = _make_run(params=params, metrics=metrics)
        _patch_collectors(
            monkeypatch=monkeypatch,
            runs=[run],
            artifact_payload=artifact_payload,
        )

        _, experiences_table, run_failures = collect_experiment_tables(experiment='exp_name')

        assert len(experiences_table) == 1
        assert not run_failures
        row = experiences_table[0]
        assert row[RUN_CALIB_ECE] == pytest.approx(0.77)
        assert row[RUN_CALIB_AECE] == pytest.approx(0.66)
        assert row[RUN_CALIB_NLL] == pytest.approx(0.55)

    def test_collect_uses_exact_repair_set_total_from_splits(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        params = {
            'controller.name': 'my_repair_controller',
            'controller.path': 'regain.controllers.repair.mock_controller',
            PARAM_CONTROLLER_TYPE: 'repair',
            'seed': '1',
            'repair.split_fraction': '0.2',
            'repair.budget_per_class': '5',
            'repair.max_samples_per_class': '99',
            'num_classes': '2',
        }
        metrics = _base_metrics_with_exp000()
        metrics.update({
            'run.accuracy.final.exp000.ctrl': 0.60,
            'run.repair.rho.exp000': 0.20,
        })
        archive_path = _write_splits_archive(
            tmp_path=tmp_path,
            repair_indices_by_exp={
                0: [0, 1, 2],
                1: [10, 11],
            },
        )
        run = _make_run(params=params, metrics=metrics)
        _patch_collectors(
            monkeypatch=monkeypatch,
            runs=[run],
            artifact_payload={
                RUN_CALIB_MAX_ECE: 0.14,
                RUN_CALIB_ECE: [0.11],
                RUN_CALIB_AECE: [0.12],
                RUN_CALIB_NLL: [0.13],
                RUN_DIAG_OUT_OF_TASK_RATE: [0.21],
                RUN_DIAG_AVG_CONF: [0.22],
                RUN_DIAG_AVG_ENTROPY: [0.23],
                RUN_DIAG_LOGIT_AVG_DRIFT: [0.24],
            },
            client_factory=lambda: _FakeMlflowClient(splits_archive_path=archive_path),
        )

        runs_table, experiences_table, run_failures = collect_experiment_tables(experiment='exp_name')

        assert len(runs_table) == 1
        assert runs_table[0][COLUMN_REPAIR_SET_TOTAL] == 5
        assert len(experiences_table) == 1
        assert experiences_table[0][COLUMN_REPAIR_SET_TOTAL] == 5
        assert not run_failures

    def test_repair_run_missing_calib_max_ece_scalar_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        params = {
            'controller.name': 'my_repair_controller',
            'controller.path': 'regain.controllers.repair.mock_controller',
            PARAM_CONTROLLER_TYPE: 'repair',
            'seed': '1',
            'repair.split_fraction': '0.0',
            'repair.budget_per_class': '5',
            'num_classes': '2',
        }
        metrics = _base_metrics_with_exp000()
        metrics.update({
            'run.accuracy.final.exp000.ctrl': 0.60,
            'run.repair.rho.exp000': 0.20,
            RUN_CALIB_MAX_ECE: 0.95,
            RUN_CALIB_ECE + '.exp000': 0.99,
        })
        artifact_payload = {
            RUN_CALIB_ECE: [0.14],
            RUN_CALIB_AECE: [0.15],
            RUN_CALIB_NLL: [0.16],
            RUN_DIAG_OUT_OF_TASK_RATE: [0.21],
            RUN_DIAG_AVG_CONF: [0.22],
            RUN_DIAG_AVG_ENTROPY: [0.23],
            RUN_DIAG_LOGIT_AVG_DRIFT: [0.24],
        }
        run = _make_run(params=params, metrics=metrics)
        _patch_collectors(
            monkeypatch=monkeypatch,
            runs=[run],
            artifact_payload=artifact_payload,
        )

        runs_table, experiences_table, run_failures = collect_experiment_tables(experiment='exp_name')
        assert not runs_table
        assert not experiences_table
        assert len(run_failures) == 1
        assert RUN_CALIB_MAX_ECE in run_failures[0]['error']

    def test_collect_skips_splits_download_when_split_fraction_is_zero(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        params = {
            'controller.name': 'my_prevention_controller',
            'controller.path': 'regain.controllers.prevention.mock_controller',
            PARAM_CONTROLLER_TYPE: 'prevention',
            'seed': '7',
            'repair.split_fraction': '0.0',
            'repair.budget_per_class': '0',
            'num_classes': '2',
        }
        metrics = _base_metrics_with_exp000()
        run = _make_run(params=params, metrics=metrics)
        _patch_collectors(
            monkeypatch=monkeypatch,
            runs=[run],
            artifact_payload=None,
        )

        def _raise_unexpected_download(**kwargs: Any) -> int | None:
            del kwargs
            raise AssertionError('unexpected splits download')

        monkeypatch.setattr(
            collectors_module,
            '_extract_repair_set_total_from_splits_artifact',
            _raise_unexpected_download,
        )

        runs_table, experiences_table, run_failures = collect_experiment_tables(experiment='exp_name')

        assert len(runs_table) == 1
        assert runs_table[0][COLUMN_REPAIR_SET_TOTAL] == 0
        assert len(experiences_table) == 1
        assert experiences_table[0][COLUMN_REPAIR_SET_TOTAL] == 0
        assert not run_failures

    def test_collect_does_not_share_repair_set_cache_across_runs(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        params = {
            'controller.name': 'my_repair_controller',
            PARAM_CONTROLLER_TYPE: 'repair',
            'seed': '1',
            'repair.split_fraction': '0.2',
            'repair.budget_per_class': '5',
            'num_classes': '2',
        }
        metrics = _base_metrics_with_exp000()
        metrics.update({
            'run.accuracy.final.exp000.ctrl': 0.60,
            'run.repair.rho.exp000': 0.20,
        })

        run_1 = SimpleNamespace(
            info=SimpleNamespace(
                run_id='run_1',
                status='FINISHED',
            ),
            data=SimpleNamespace(
                params=params,
                metrics=metrics,
            ),
        )
        run_2 = SimpleNamespace(
            info=SimpleNamespace(
                run_id='run_2',
                status='FINISHED',
            ),
            data=SimpleNamespace(
                params=params,
                metrics=metrics,
            ),
        )

        _patch_collectors(
            monkeypatch=monkeypatch,
            runs=[run_1, run_2],
            artifact_payload={
                RUN_CALIB_MAX_ECE: 0.17,
                RUN_CALIB_ECE: [0.11],
                RUN_CALIB_AECE: [0.12],
                RUN_CALIB_NLL: [0.13],
                RUN_DIAG_OUT_OF_TASK_RATE: [0.21],
                RUN_DIAG_AVG_CONF: [0.22],
                RUN_DIAG_AVG_ENTROPY: [0.23],
                RUN_DIAG_LOGIT_AVG_DRIFT: [0.24],
            },
        )

        call_count = {'value': 0}

        def _extract_per_run_total(*, client: Any, run_id: str) -> int:
            del client
            call_count['value'] += 1
            if run_id == 'run_1':
                return 4
            return 5

        monkeypatch.setattr(
            collectors_module,
            '_extract_repair_set_total_from_splits_artifact',
            _extract_per_run_total,
        )

        runs_table, experiences_table, run_failures = collect_experiment_tables(experiment='exp_name')

        assert call_count['value'] == 2
        assert len(runs_table) == 2
        assert runs_table[0][COLUMN_REPAIR_SET_TOTAL] == 4
        assert runs_table[1][COLUMN_REPAIR_SET_TOTAL] == 5
        assert len(experiences_table) == 2
        assert experiences_table[0][COLUMN_REPAIR_SET_TOTAL] == 4
        assert experiences_table[1][COLUMN_REPAIR_SET_TOTAL] == 5
        assert not run_failures

    def test_collect_requires_controller_type_param(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        params = {
            'controller.name': 'my_prevention_controller',
            'seed': '1',
            'repair.split_fraction': '0.2',
            'repair.budget_per_class': '1',
            'num_classes': '2',
        }
        run = _make_run(params=params, metrics=_base_metrics_with_exp000())
        _patch_collectors(
            monkeypatch=monkeypatch,
            runs=[run],
            artifact_payload=None,
        )

        runs_table, experiences_table, run_failures = collect_experiment_tables(experiment='exp_name')
        assert not runs_table
        assert not experiences_table
        assert len(run_failures) == 1
        assert 'controller.type' in run_failures[0]['error']

    def test_collect_requires_repair_split_fraction_param(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        params = {
            'controller.name': 'my_prevention_controller',
            PARAM_CONTROLLER_TYPE: 'prevention',
            'seed': '1',
            'repair.budget_per_class': '1',
            'num_classes': '2',
        }
        run = _make_run(params=params, metrics=_base_metrics_with_exp000())
        _patch_collectors(
            monkeypatch=monkeypatch,
            runs=[run],
            artifact_payload=None,
        )

        runs_table, experiences_table, run_failures = collect_experiment_tables(experiment='exp_name')
        assert not runs_table
        assert not experiences_table
        assert len(run_failures) == 1
        assert 'repair.split_fraction' in run_failures[0]['error']
