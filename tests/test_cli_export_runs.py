"""
Tests for run export CLI behavior.
"""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import regain.cli.export_runs as export_runs_cli
from regain.cli._utils._output_helpers import CliFailure


def test_export_runs_groups_same_experiment_and_exports_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_map = {
        'a.yaml': SimpleNamespace(experiment_name='exp_shared', mlflow_tracking_uri='mlflow://tracking'),
        'b.yaml': SimpleNamespace(experiment_name='exp_shared', mlflow_tracking_uri='mlflow://tracking'),
    }
    export_calls: list[tuple[str, str | None]] = []
    captured_outputs: list[export_runs_cli.StagedOutput] = []

    monkeypatch.setattr(
        export_runs_cli,
        '_resolve_config_files',
        lambda **kwargs: ['a.yaml', 'b.yaml'],
    )
    monkeypatch.setattr(
        export_runs_cli,
        'load_experiment_config',
        lambda config_file: config_map[config_file],
    )

    def _fake_export_runs_for_experiment(
        *,
        experiment_name: str,
        export_dir: str | Path,
        tracking_uri: str | None,
    ) -> tuple[Path, Path, Path]:
        export_calls.append((experiment_name, tracking_uri))
        export_root = Path(export_dir) / experiment_name
        export_root.mkdir(parents=True, exist_ok=True)
        metadata = export_root / 'run_metadata.csv'
        params = export_root / 'run_params.csv'
        metrics = export_root / 'run_metrics.csv'
        metadata.write_text('metadata', encoding='utf-8')
        params.write_text('params', encoding='utf-8')
        metrics.write_text('metrics', encoding='utf-8')
        return metadata, params, metrics

    def _fake_finalize_staged_outputs(
        *,
        outputs: list[export_runs_cli.StagedOutput],
        failures: list[CliFailure],
        allow_partial: bool,
        overwrite: bool,
    ) -> int:
        captured_outputs.extend(list(outputs))
        return len(outputs)

    monkeypatch.setattr(export_runs_cli, 'export_runs_for_experiment', _fake_export_runs_for_experiment)
    monkeypatch.setattr(export_runs_cli, 'finalize_staged_outputs', _fake_finalize_staged_outputs)
    monkeypatch.setattr(export_runs_cli, 'print_failure_summary', lambda **kwargs: None)
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'regain-export-runs',
            '--config-files',
            'a.yaml,b.yaml',
            '--export-dir',
            str(tmp_path / 'exports'),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        export_runs_cli.main()

    assert int(exc_info.value.code) == 0
    assert export_calls == [('exp_shared', 'mlflow://tracking')]
    assert len(captured_outputs) == 1
    assert captured_outputs[0].destination == tmp_path / 'exports' / 'exp_shared'


def test_export_runs_rejects_mixed_tracking_uris_for_same_experiment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_map = {
        'a.yaml': SimpleNamespace(experiment_name='exp_shared', mlflow_tracking_uri='mlflow://one'),
        'b.yaml': SimpleNamespace(experiment_name='exp_shared', mlflow_tracking_uri='mlflow://two'),
    }
    export_calls: list[tuple[str, str | None]] = []
    captured_failures: list[CliFailure] = []

    monkeypatch.setattr(
        export_runs_cli,
        '_resolve_config_files',
        lambda **kwargs: ['a.yaml', 'b.yaml'],
    )
    monkeypatch.setattr(
        export_runs_cli,
        'load_experiment_config',
        lambda config_file: config_map[config_file],
    )

    def _fake_export_runs_for_experiment(
        *,
        experiment_name: str,
        export_dir: str | Path,
        tracking_uri: str | None,
    ) -> tuple[Path, Path, Path]:
        export_calls.append((experiment_name, tracking_uri))
        raise AssertionError('export should not be called for mixed URI group')

    def _fake_finalize_staged_outputs(
        *,
        outputs: list[export_runs_cli.StagedOutput],
        failures: list[CliFailure],
        allow_partial: bool,
        overwrite: bool,
    ) -> int:
        captured_failures.extend(list(failures))
        return 0

    monkeypatch.setattr(export_runs_cli, 'export_runs_for_experiment', _fake_export_runs_for_experiment)
    monkeypatch.setattr(export_runs_cli, 'finalize_staged_outputs', _fake_finalize_staged_outputs)
    monkeypatch.setattr(export_runs_cli, 'print_failure_summary', lambda **kwargs: None)
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'regain-export-runs',
            '--config-files',
            'a.yaml,b.yaml',
            '--export-dir',
            str(tmp_path / 'exports'),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        export_runs_cli.main()

    assert int(exc_info.value.code) == 1
    assert export_calls == []
    assert any('Conflicting tracking URIs' in failure.message for failure in captured_failures)

