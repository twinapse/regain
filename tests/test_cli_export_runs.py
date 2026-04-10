"""
Tests for run export CLI behavior.
"""

from pathlib import Path
import sys

import pytest

from regain.cli._utils.output_helpers import CliFailure
import regain.cli.export_runs as export_runs_cli


def test_export_runs_uses_output_dir_for_multiple_experiments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    export_calls: list[tuple[str, str | None]] = []
    captured_outputs: list[export_runs_cli.StagedOutput] = []

    def _fake_export_runs_to_csvs(
        *,
        experiment: str,
        metadata_path: Path,
        params_path: Path,
        metrics_path: Path,
        tracking_uri: str | None,
    ) -> None:
        export_calls.append((experiment, tracking_uri))
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text('metadata', encoding='utf-8')
        params_path.write_text('params', encoding='utf-8')
        metrics_path.write_text('metrics', encoding='utf-8')

    def _fake_finalize_staged_outputs(
        *,
        outputs: list[export_runs_cli.StagedOutput],
        failures: list[CliFailure],
        allow_partial: bool,
        overwrite: bool,
    ) -> int:
        captured_outputs.extend(list(outputs))
        return len(outputs)

    monkeypatch.setattr(export_runs_cli, 'export_runs_to_csvs', _fake_export_runs_to_csvs)
    monkeypatch.setattr(export_runs_cli, 'finalize_staged_outputs', _fake_finalize_staged_outputs)
    monkeypatch.setattr(export_runs_cli, 'print_failure_summary', lambda **kwargs: None)
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'regain-export-runs',
            '--experiments',
            'exp_a,exp_b',
            '--output-dir',
            str(tmp_path / 'exports'),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        export_runs_cli.main()

    assert int(exc_info.value.code) == 0
    assert export_calls == [('exp_a', None), ('exp_b', None)]
    assert len(captured_outputs) == 2
    assert captured_outputs[0].destination == tmp_path / 'exports' / 'exp_a'
    assert captured_outputs[1].destination == tmp_path / 'exports' / 'exp_b'


def test_export_runs_passes_tracking_uri(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tracking_uris: list[str | None] = []

    def _fake_export_runs_to_csvs(
        *,
        experiment: str,
        metadata_path: Path,
        params_path: Path,
        metrics_path: Path,
        tracking_uri: str | None,
    ) -> None:
        tracking_uris.append(tracking_uri)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text('metadata', encoding='utf-8')
        params_path.write_text('params', encoding='utf-8')
        metrics_path.write_text('metrics', encoding='utf-8')

    monkeypatch.setattr(export_runs_cli, 'export_runs_to_csvs', _fake_export_runs_to_csvs)
    monkeypatch.setattr(export_runs_cli, 'print_failure_summary', lambda **kwargs: None)
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'regain-export-runs',
            '--experiments',
            'exp_shared',
            '--output-dir',
            str(tmp_path / 'exports'),
            '--tracking-uri',
            'mlflow://override',
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        export_runs_cli.main()

    assert int(exc_info.value.code) == 0
    assert tracking_uris == ['mlflow://override']


def test_parser_rejects_export_dir_flag() -> None:
    parser = export_runs_cli._build_arg_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                '--experiments',
                'exp_1',
                '--output-dir',
                '/tmp/exports',
                '--export-dir',
                '/tmp/legacy',
            ]
        )
