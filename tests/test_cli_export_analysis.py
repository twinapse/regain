"""
Tests for analysis export CLI behavior.
"""

from pathlib import Path
import sys

import pytest

from regain.cli._utils.output_helpers import CliFailure
import regain.cli.export_analysis as export_analysis_cli


def test_export_analysis_uses_analysis_and_output_roots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_outputs: list[export_analysis_cli.StagedOutput] = []
    export_paths: list[Path] = []
    load_dirs: list[Path] = []

    def _fake_load_analysis_tables(*, experiment_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        load_dirs.append(experiment_dir)
        return ([{'run_id': 'run_1'}], [{'task': '0'}])

    def _fake_export_analysis_to_json(
        *,
        experiment: str,
        experiment_dir: Path,
        export_path: Path,
        tracking_uri: str | None,
        artifact_location: str | None,
        runs_table: list[dict[str, object]],
        experiences_table: list[dict[str, object]],
        include_controllers: list[str] | None,
        exclude_controllers: list[str] | None,
        max_runs: int | None,
        default_num_classes: int | None,
        require_finished: bool,
    ) -> None:
        export_paths.append(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text('{}', encoding='utf-8')

    def _fake_finalize_staged_outputs(
        *,
        outputs: list[export_analysis_cli.StagedOutput],
        failures: list[CliFailure],
        allow_partial: bool,
        overwrite: bool,
    ) -> int:
        captured_outputs.extend(list(outputs))
        return len(outputs)

    monkeypatch.setattr(export_analysis_cli, '_load_analysis_tables', _fake_load_analysis_tables)
    monkeypatch.setattr(export_analysis_cli, 'export_analysis_to_json', _fake_export_analysis_to_json)
    monkeypatch.setattr(export_analysis_cli, 'finalize_staged_outputs', _fake_finalize_staged_outputs)
    monkeypatch.setattr(export_analysis_cli, 'print_failure_summary', lambda **kwargs: None)
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'regain-export-analysis',
            '--experiments',
            'exp_a,exp_b',
            '--analysis-dir',
            str(tmp_path / 'analysis'),
            '--output-dir',
            str(tmp_path / 'exports'),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        export_analysis_cli.main()

    assert int(exc_info.value.code) == 0
    assert load_dirs == [tmp_path / 'analysis' / 'exp_a', tmp_path / 'analysis' / 'exp_b']
    assert len(export_paths) == 2
    assert len(captured_outputs) == 2
    assert captured_outputs[0].destination == tmp_path / 'exports' / 'exp_a' / 'analysis.json'
    assert captured_outputs[1].destination == tmp_path / 'exports' / 'exp_b' / 'analysis.json'


def test_parser_rejects_legacy_export_dir_flag() -> None:
    parser = export_analysis_cli._build_arg_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                '--experiments',
                'exp_1',
                '--analysis-dir',
                '/tmp/analysis',
                '--output-dir',
                '/tmp/exports',
                '--export-dir',
                '/tmp/legacy',
            ]
        )
