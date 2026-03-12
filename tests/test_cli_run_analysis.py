"""
Tests for analysis runner CLI behavior.
"""

from pathlib import Path
import sys

import pytest

import regain.cli.run_analysis as run_analysis_cli
from regain.cli._utils._output_helpers import CliFailure


def _run_collect_main(
    *,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    collect_result: tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, str]]],
    experiments: str,
    allow_partial: bool,
) -> tuple[int, list[run_analysis_cli.StagedOutput], list[CliFailure]]:
    """
    Execute `run_analysis collect` with patched dependencies.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
        tmp_path (Path): Temporary output directory.
        collect_result (tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, str]]]):
            Return payload for `collect_experiment_tables`.
        experiments (str): Value passed to `--experiments`.
        allow_partial (bool): Whether to pass `--allow-partial`.

    Returns:
        tuple[int, list[run_analysis_cli.StagedOutput], list[CliFailure]]:
            Exit code, staged outputs, and failures passed to finalize.
    """
    captured_outputs: list[run_analysis_cli.StagedOutput] = []
    captured_failures: list[CliFailure] = []

    def _fake_collect_experiment_tables(
        **kwargs,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, str]]]:
        return collect_result

    def _fake_finalize_staged_outputs(
        *,
        outputs: list[run_analysis_cli.StagedOutput],
        failures: list[CliFailure],
        allow_partial: bool,
        overwrite: bool,
    ) -> int:
        captured_outputs.extend(list(outputs))
        captured_failures.extend(list(failures))
        return len(outputs)

    monkeypatch.setattr(run_analysis_cli, 'collect_experiment_tables', _fake_collect_experiment_tables)
    monkeypatch.setattr(run_analysis_cli, 'finalize_staged_outputs', _fake_finalize_staged_outputs)
    monkeypatch.setattr(run_analysis_cli, 'print_failure_summary', lambda **kwargs: None)

    argv = [
        'regain-analysis-tool',
        '--experiments',
        experiments,
        '--output-dir',
        str(tmp_path),
    ]
    if allow_partial:
        argv.append('--allow-partial')
    argv.append('collect')
    monkeypatch.setattr(sys, 'argv', argv)

    with pytest.raises(SystemExit) as exc_info:
        run_analysis_cli.main()

    return int(exc_info.value.code), captured_outputs, captured_failures


def test_collect_zero_success_runs_fails_without_allow_partial(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    exit_code, staged_outputs, failures = _run_collect_main(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        collect_result=([], [], []),
        experiments='exp',
        allow_partial=False,
    )

    assert exit_code == 1
    assert staged_outputs == []
    assert any('No successful runs were collected' in failure.message for failure in failures)


def test_collect_zero_success_runs_fails_with_allow_partial(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    exit_code, staged_outputs, failures = _run_collect_main(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        collect_result=(
            [],
            [],
            [
                {
                    'run_id': 'run_1',
                    'run_name': 'bad_run',
                    'error': 'invalid payload',
                }
            ],
        ),
        experiments='exp',
        allow_partial=True,
    )

    assert exit_code == 1
    assert staged_outputs == []
    assert any('run=run_1' in failure.scope for failure in failures)
    assert any('No successful runs were collected' in failure.message for failure in failures)


def test_collect_multiple_experiments_stages_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    collect_calls: list[str] = []
    captured_outputs: list[run_analysis_cli.StagedOutput] = []

    def _fake_collect_experiment_tables(
        *,
        experiment: str,
        **kwargs,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, str]]]:
        collect_calls.append(experiment)
        out_dir = Path(kwargs['out_dir'])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'runs_table.jsonl').write_text('{}\n', encoding='utf-8')
        (out_dir / 'experiences_table.jsonl').write_text('{}\n', encoding='utf-8')
        return ([{'run_id': f'run_{experiment}'}], [{'task': '0'}], [])

    def _fake_finalize_staged_outputs(
        *,
        outputs: list[run_analysis_cli.StagedOutput],
        failures: list[CliFailure],
        allow_partial: bool,
        overwrite: bool,
    ) -> int:
        captured_outputs.extend(list(outputs))
        return len(outputs)

    monkeypatch.setattr(run_analysis_cli, 'collect_experiment_tables', _fake_collect_experiment_tables)
    monkeypatch.setattr(run_analysis_cli, 'finalize_staged_outputs', _fake_finalize_staged_outputs)
    monkeypatch.setattr(run_analysis_cli, 'print_failure_summary', lambda **kwargs: None)
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'regain-analysis-tool',
            '--experiments',
            'exp_a,exp_b',
            '--output-dir',
            str(tmp_path),
            'collect',
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        run_analysis_cli.main()

    assert int(exc_info.value.code) == 0
    assert collect_calls == ['exp_a', 'exp_b']
    assert len(captured_outputs) == 2
    assert captured_outputs[0].destination == tmp_path / 'exp_a' / 'tables'
    assert captured_outputs[1].destination == tmp_path / 'exp_b' / 'tables'
