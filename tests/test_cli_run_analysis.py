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
    allow_partial: bool,
) -> tuple[int, list[run_analysis_cli.StagedOutput], list[CliFailure]]:
    """
    Execute `run_analysis collect` with patched dependencies.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
        tmp_path (Path): Temporary output directory.
        collect_result (tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, str]]]):
            Return payload for `collect_experiment_tables`.
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
        return 0

    monkeypatch.setattr(run_analysis_cli, 'collect_experiment_tables', _fake_collect_experiment_tables)
    monkeypatch.setattr(run_analysis_cli, 'finalize_staged_outputs', _fake_finalize_staged_outputs)
    monkeypatch.setattr(run_analysis_cli, 'print_failure_summary', lambda **kwargs: None)

    argv = [
        'regain-analysis-tool',
        '--experiment',
        'exp',
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
        allow_partial=True,
    )

    assert exit_code == 1
    assert staged_outputs == []
    assert any('run=run_1' in failure.scope for failure in failures)
    assert any('No successful runs were collected' in failure.message for failure in failures)
