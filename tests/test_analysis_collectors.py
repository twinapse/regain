"""
Tests for analysis collectors.
"""

import csv
import io
from pathlib import Path
import tarfile
from types import SimpleNamespace
from typing import Any

import pytest

from regain.analysis.artifacts import ARTIFACT_RHO
from regain.analysis.collectors import _extract_repair_set_total_from_splits_artifact
from regain.analysis.collectors import collect_experiment_tables
from regain.analysis.frontier import write_repairability_frontier_outputs
import regain.analysis.collectors as collectors_module
from regain.constants import COLUMN_REPAIR_SET_TOTAL
from regain.constants import PARAM_CONTROLLER_TYPE
from regain.constants import RUN_ACC_FINAL
from regain.constants import RUN_ACC_REF
from regain.constants import RUN_CALIB_AECE
from regain.constants import RUN_CALIB_ECE
from regain.constants import RUN_CALIB_MAX_ECE
from regain.constants import RUN_CALIB_NLL
from regain.constants import RUN_DIAG_AVG_CONF
from regain.constants import RUN_DIAG_AVG_ENTROPY
from regain.constants import RUN_DIAG_LOGIT_AVG_DRIFT
from regain.constants import RUN_DIAG_OUT_OF_TASK_RATE


def _make_run(
    *,
    run_id: str = 'run_1',
    params: dict[str, str],
    metrics: dict[str, float],
) -> SimpleNamespace:
    return SimpleNamespace(
        info=SimpleNamespace(
            run_id=run_id,
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
    analysis_identity: tuple[str, str, int | None, int | None] | None = ('vit_small', 'er', None, None),
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
    if analysis_identity is not None:
        monkeypatch.setattr(
            collectors_module,
            '_extract_analysis_identity_from_config_artifact',
            lambda **kwargs: analysis_identity,
        )


def _base_metrics_with_exp000() -> dict[str, float]:
    return {
        f'{RUN_ACC_REF}.exp000.base': 0.80,
        f'{RUN_ACC_FINAL}.exp000.base': 0.55,
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


class _ConfigArtifactMlflowClient:
    def __init__(
        self,
        *,
        config_path_by_run_id: dict[str, Path] | None = None,
        default_config_path: Path | None = None,
    ) -> None:
        self._config_path_by_run_id = config_path_by_run_id or {}
        self._default_config_path = default_config_path

    def download_artifacts(
        self,
        run_id: str,
        artifact_path: str,
        dst_path: str,
    ) -> str:
        del artifact_path, dst_path
        if run_id in self._config_path_by_run_id:
            return str(self._config_path_by_run_id[run_id])
        if self._default_config_path is not None:
            return str(self._default_config_path)
        return '/tmp/nonexistent_config_artifact.yaml'


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
    def test_repair_run_identity_comes_from_config_parser(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        params = {
            'controller.name': 'my_repair_controller',
            PARAM_CONTROLLER_TYPE: 'repair',
            'seed': '1',
            'repair.split_fraction': '0.0',
            'repair.budget_fraction': '0.5',
            'num_classes': '2',
        }
        metrics = _base_metrics_with_exp000()
        metrics.update({
            f'{RUN_ACC_FINAL}.exp000.ctrl': 0.60,
        })
        run = _make_run(params=params, metrics=metrics)
        config_path = tmp_path / 'config.yaml'
        config_path.write_text('experiment_name: exp\n', encoding='utf-8')
        _patch_collectors(
            monkeypatch=monkeypatch,
            runs=[run],
            artifact_payload={
                RUN_CALIB_MAX_ECE: 0.10,
                RUN_CALIB_ECE: [0.11],
                RUN_CALIB_AECE: [0.12],
                RUN_CALIB_NLL: [0.13],
                RUN_DIAG_OUT_OF_TASK_RATE: [0.21],
                RUN_DIAG_AVG_CONF: [0.22],
                RUN_DIAG_AVG_ENTROPY: [0.23],
                RUN_DIAG_LOGIT_AVG_DRIFT: [0.24],
            },
            client_factory=lambda: _ConfigArtifactMlflowClient(default_config_path=config_path),
            analysis_identity=None,
        )

        captured_paths: list[Path] = []

        def _fake_load_experiment_config(path: str | Path) -> SimpleNamespace:
            captured_paths.append(Path(path))
            return SimpleNamespace(
                backbone=SimpleNamespace(
                    name='resnet18',
                    training=SimpleNamespace(
                        strategy=SimpleNamespace(name='bic'),
                    ),
                )
            )

        monkeypatch.setattr(
            collectors_module,
            'load_experiment_config',
            _fake_load_experiment_config,
        )

        runs_table, _, run_failures = collect_experiment_tables(experiment='exp_name')

        assert len(runs_table) == 1
        assert runs_table[0]['backbone_name'] == 'resnet18'
        assert runs_table[0]['strategy_name'] == 'bic'
        assert not run_failures
        assert captured_paths == [config_path]

    def test_missing_config_artifact_skips_run(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        params = {
            'controller.name': 'my_prevention_controller',
            PARAM_CONTROLLER_TYPE: 'prevention',
            'seed': '1',
            'repair.split_fraction': '0.0',
            'repair.budget_fraction': '0.5',
            'num_classes': '2',
        }
        run = _make_run(params=params, metrics=_base_metrics_with_exp000())
        _patch_collectors(
            monkeypatch=monkeypatch,
            runs=[run],
            artifact_payload=None,
            client_factory=lambda: _ConfigArtifactMlflowClient(),
            analysis_identity=None,
        )

        runs_table, experiences_table, run_failures = collect_experiment_tables(experiment='exp_name')

        assert not runs_table
        assert not experiences_table
        assert len(run_failures) == 1
        assert 'run_1' in run_failures[0]['error']
        assert 'config.yaml' in run_failures[0]['error']

    @pytest.mark.parametrize(
        'parsed_config,expected_field',
        [
            (
                SimpleNamespace(backbone=None),
                'backbone',
            ),
            (
                SimpleNamespace(
                    backbone=SimpleNamespace(
                        name='',
                        training=SimpleNamespace(strategy=SimpleNamespace(name='er')),
                    )
                ),
                'backbone.name',
            ),
            (
                SimpleNamespace(
                    backbone=SimpleNamespace(
                        name='resnet18',
                        training=None,
                    )
                ),
                'backbone.training',
            ),
            (
                SimpleNamespace(
                    backbone=SimpleNamespace(
                        name='resnet18',
                        training=SimpleNamespace(strategy=SimpleNamespace(name='')),
                    )
                ),
                'backbone.training.strategy.name',
            ),
        ],
    )
    def test_unresolved_config_identity_skips_run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        parsed_config: SimpleNamespace,
        expected_field: str,
    ) -> None:
        params = {
            'controller.name': 'my_prevention_controller',
            PARAM_CONTROLLER_TYPE: 'prevention',
            'seed': '1',
            'repair.split_fraction': '0.0',
            'repair.budget_fraction': '0.5',
            'num_classes': '2',
        }
        run = _make_run(params=params, metrics=_base_metrics_with_exp000())
        config_path = tmp_path / 'config.yaml'
        config_path.write_text('experiment_name: exp\n', encoding='utf-8')
        _patch_collectors(
            monkeypatch=monkeypatch,
            runs=[run],
            artifact_payload=None,
            client_factory=lambda: _ConfigArtifactMlflowClient(default_config_path=config_path),
            analysis_identity=None,
        )
        monkeypatch.setattr(
            collectors_module,
            'load_experiment_config',
            lambda path: parsed_config,
        )

        runs_table, experiences_table, run_failures = collect_experiment_tables(experiment='exp_name')

        assert not runs_table
        assert not experiences_table
        assert len(run_failures) == 1
        assert 'run_1' in run_failures[0]['error']
        assert 'config.yaml' in run_failures[0]['error']
        assert expected_field in run_failures[0]['error']

    def test_frontier_grouping_preserves_distinct_config_identity(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        params = {
            'controller.name': 'my_repair_controller',
            PARAM_CONTROLLER_TYPE: 'repair',
            'seed': '1',
            'repair.split_fraction': '0.0',
            'repair.budget_fraction': '0.5',
            'num_classes': '2',
            'scenario': 'cifar100',
        }
        metrics = _base_metrics_with_exp000()
        metrics.update({
            f'{RUN_ACC_FINAL}.exp000.ctrl': 0.60,
            'run.repair.rho.exp000': 0.20,
        })
        run_1 = _make_run(run_id='run_1', params=params, metrics=metrics)
        run_2 = _make_run(run_id='run_2', params=params, metrics=metrics)
        config_path_1 = tmp_path / 'run_1_config.yaml'
        config_path_2 = tmp_path / 'run_2_config.yaml'
        config_path_1.write_text('experiment_name: exp\n', encoding='utf-8')
        config_path_2.write_text('experiment_name: exp\n', encoding='utf-8')
        _patch_collectors(
            monkeypatch=monkeypatch,
            runs=[run_1, run_2],
            artifact_payload={
                RUN_CALIB_MAX_ECE: 0.10,
                RUN_CALIB_ECE: [0.11],
                RUN_CALIB_AECE: [0.12],
                RUN_CALIB_NLL: [0.13],
                RUN_DIAG_OUT_OF_TASK_RATE: [0.21],
                RUN_DIAG_AVG_CONF: [0.22],
                RUN_DIAG_AVG_ENTROPY: [0.23],
                RUN_DIAG_LOGIT_AVG_DRIFT: [0.24],
            },
            client_factory=lambda: _ConfigArtifactMlflowClient(
                config_path_by_run_id={
                    'run_1': config_path_1,
                    'run_2': config_path_2,
                }
            ),
            analysis_identity=None,
        )

        def _fake_load_experiment_config(path: str | Path) -> SimpleNamespace:
            if Path(path) == config_path_1:
                return SimpleNamespace(
                    backbone=SimpleNamespace(
                        name='vit_small',
                        training=SimpleNamespace(strategy=SimpleNamespace(name='er')),
                    )
                )
            return SimpleNamespace(
                backbone=SimpleNamespace(
                    name='resnet18',
                    training=SimpleNamespace(strategy=SimpleNamespace(name='er')),
                )
            )

        monkeypatch.setattr(
            collectors_module,
            'load_experiment_config',
            _fake_load_experiment_config,
        )

        runs_table, experiences_table, run_failures = collect_experiment_tables(experiment='exp_name')
        assert len(runs_table) == 2
        assert len(experiences_table) == 2
        assert not run_failures

        output_paths = write_repairability_frontier_outputs(
            runs_table=runs_table,
            experiences_table=experiences_table,
            out_dir=tmp_path / 'frontier_out',
        )

        with output_paths['candidates'].open('r', newline='', encoding='utf-8') as f:
            frontier_rows = list(csv.DictReader(f))
        repair_frontier_rows = [row for row in frontier_rows if row['controller_name'] == 'my_repair_controller']
        assert len(repair_frontier_rows) == 2
        assert {
            (row['backbone_name'], row['strategy_name'])
            for row in repair_frontier_rows
        } == {('vit_small', 'er'), ('resnet18', 'er')}

        with output_paths['selection'].open('r', newline='', encoding='utf-8') as f:
            selection_rows = list(csv.DictReader(f))
        assert len(selection_rows) == 2
        assert {
            (row['backbone_name'], row['strategy_name'])
            for row in selection_rows
        } == {('vit_small', 'er'), ('resnet18', 'er')}

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
            'repair.budget_fraction': '0.5',
            'num_classes': '2',
        }
        metrics = _base_metrics_with_exp000()
        metrics.update({
            f'{RUN_ACC_FINAL}.exp000.ctrl': 0.60,
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
            'repair.budget_fraction': '0.5',
            'num_classes': '2',
        }
        metrics = _base_metrics_with_exp000()
        metrics.update({
            f'{RUN_ACC_FINAL}.exp000.ctrl': 0.60,
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

    def test_repair_run_allows_missing_per_experience_rho(self, monkeypatch: pytest.MonkeyPatch) -> None:
        params = {
            'controller.name': 'my_repair_controller',
            'controller.path': 'regain.controllers.repair.mock_controller',
            PARAM_CONTROLLER_TYPE: 'repair',
            'seed': '1',
            'repair.split_fraction': '0.0',
            'repair.budget_fraction': '0.5',
            'num_classes': '2',
        }
        metrics = _base_metrics_with_exp000()
        metrics.update({
            f'{RUN_ACC_FINAL}.exp000.ctrl': 0.60,
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
        assert len(experiences_table) == 1
        assert not run_failures
        assert experiences_table[0][ARTIFACT_RHO] is None

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
            'repair.budget_fraction': '0.5',
            'num_classes': '2',
        }
        metrics = _base_metrics_with_exp000()
        metrics.update({
            f'{RUN_ACC_FINAL}.exp000.ctrl': 0.60,
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
            'repair.budget_fraction': '0.5',
            'num_classes': '2',
        }
        metrics = _base_metrics_with_exp000()
        metrics.update({
            f'{RUN_ACC_FINAL}.exp000.ctrl': 0.60,
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
            'repair.budget_fraction': '0.5',
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
            'repair.budget_fraction': '0.5',
            'num_classes': '2',
        }
        metrics = _base_metrics_with_exp000()
        metrics.update({
            f'{RUN_ACC_FINAL}.exp000.ctrl': 0.60,
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
            'repair.budget_fraction': '0.5',
            'num_classes': '2',
        }
        metrics = _base_metrics_with_exp000()
        metrics.update({
            f'{RUN_ACC_FINAL}.exp000.ctrl': 0.60,
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
            'repair.budget_fraction': '1.0',
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
            'repair.budget_fraction': '0.5',
            'num_classes': '2',
        }
        metrics = _base_metrics_with_exp000()
        metrics.update({
            f'{RUN_ACC_FINAL}.exp000.ctrl': 0.60,
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
            'repair.budget_fraction': '1.0',
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
            'repair.budget_fraction': '1.0',
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

    def test_replay_metadata_is_collected_from_params(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        params = {
            'controller.name': 'my_prevention_controller',
            PARAM_CONTROLLER_TYPE: 'prevention',
            'seed': '1',
            'repair.split_fraction': '0.0',
            'repair.budget_fraction': '0.5',
            'num_classes': '2',
            'backbone.training.strategy.mem_size': '500',
            'backbone.training.strategy.batch_size_mem': '32',
        }
        run = _make_run(params=params, metrics=_base_metrics_with_exp000())
        _patch_collectors(
            monkeypatch=monkeypatch,
            runs=[run],
            artifact_payload=None,
        )

        runs_table, _, run_failures = collect_experiment_tables(experiment='exp_name')
        assert not run_failures
        assert len(runs_table) == 1
        assert runs_table[0]['replay_mem_size'] == 500
        assert runs_table[0]['replay_batch_size_mem'] == 32

    def test_replay_metadata_falls_back_to_config_strategy_kwargs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        params = {
            'controller.name': 'my_prevention_controller',
            PARAM_CONTROLLER_TYPE: 'prevention',
            'seed': '1',
            'repair.split_fraction': '0.0',
            'repair.budget_fraction': '0.5',
            'num_classes': '2',
        }
        run = _make_run(params=params, metrics=_base_metrics_with_exp000())
        config_path = tmp_path / 'config.yaml'
        config_path.write_text('experiment_name: exp\n', encoding='utf-8')
        _patch_collectors(
            monkeypatch=monkeypatch,
            runs=[run],
            artifact_payload=None,
            client_factory=lambda: _ConfigArtifactMlflowClient(default_config_path=config_path),
            analysis_identity=None,
        )
        monkeypatch.setattr(
            collectors_module,
            'load_experiment_config',
            lambda path: SimpleNamespace(
                backbone=SimpleNamespace(
                    name='resnet18',
                    training=SimpleNamespace(
                        strategy=SimpleNamespace(
                            name='replay',
                            kwargs={'mem_size': 1000, 'batch_size_mem': 64},
                        )
                    ),
                )
            ),
        )

        runs_table, _, run_failures = collect_experiment_tables(experiment='exp_name')
        assert not run_failures
        assert len(runs_table) == 1
        assert runs_table[0]['replay_mem_size'] == 1000
        assert runs_table[0]['replay_batch_size_mem'] == 64

    def test_replay_metadata_is_none_when_strategy_lacks_kwargs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        params = {
            'controller.name': 'my_prevention_controller',
            PARAM_CONTROLLER_TYPE: 'prevention',
            'seed': '1',
            'repair.split_fraction': '0.0',
            'repair.budget_fraction': '0.5',
            'num_classes': '2',
        }
        run = _make_run(params=params, metrics=_base_metrics_with_exp000())
        config_path = tmp_path / 'config.yaml'
        config_path.write_text('experiment_name: exp\n', encoding='utf-8')
        _patch_collectors(
            monkeypatch=monkeypatch,
            runs=[run],
            artifact_payload=None,
            client_factory=lambda: _ConfigArtifactMlflowClient(default_config_path=config_path),
            analysis_identity=None,
        )
        monkeypatch.setattr(
            collectors_module,
            'load_experiment_config',
            lambda path: SimpleNamespace(
                backbone=SimpleNamespace(
                    name='resnet18',
                    training=SimpleNamespace(
                        strategy=SimpleNamespace(name='naive', kwargs={}),
                    ),
                )
            ),
        )

        runs_table, _, run_failures = collect_experiment_tables(experiment='exp_name')
        assert not run_failures
        assert runs_table[0]['replay_mem_size'] is None
        assert runs_table[0]['replay_batch_size_mem'] is None
