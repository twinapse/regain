"""
Tests for plot generation CLI behavior.
"""

from pathlib import Path
import sys

import pytest

import regain.cli.generate_plots as generate_plots_cli
from regain.cli._utils._output_helpers import CliFailure


def test_generate_plots_uses_output_root_per_experiment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    analysis_root = tmp_path / 'analysis'
    (analysis_root / 'exp_a').mkdir(parents=True)
    (analysis_root / 'exp_b').mkdir(parents=True)
    captured_outputs: list[generate_plots_cli.StagedOutput] = []

    def _fake_plot_analysis_outputs(
        *,
        curve_rows: list[dict[str, object]] | None = None,
        frontier_rows: list[dict[str, object]] | None = None,
        analysis_out: str | Path | None = None,
        perf_key: str,
        mode: str,
        save_dir: str | Path | None = None,
    ) -> list[Path]:
        if mode in ['save', 'both'] and save_dir is not None:
            save_path = Path(save_dir) / 'plot.png'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text('plot', encoding='utf-8')
            return [save_path]
        return []

    def _fake_finalize_staged_outputs(
        *,
        outputs: list[generate_plots_cli.StagedOutput],
        failures: list[CliFailure],
        allow_partial: bool,
        overwrite: bool,
    ) -> int:
        captured_outputs.extend(list(outputs))
        return len(outputs)

    monkeypatch.setattr(generate_plots_cli, 'plot_analysis_outputs', _fake_plot_analysis_outputs)
    monkeypatch.setattr(generate_plots_cli, 'finalize_staged_outputs', _fake_finalize_staged_outputs)
    monkeypatch.setattr(generate_plots_cli, 'print_failure_summary', lambda **kwargs: None)
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'regain-generate-plots',
            '--experiments',
            'exp_a,exp_b',
            '--analysis-dir',
            str(analysis_root),
            '--save',
            '--output-dir',
            str(tmp_path / 'plots'),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        generate_plots_cli.main()

    assert int(exc_info.value.code) == 0
    assert len(captured_outputs) == 2
    assert captured_outputs[0].destination == tmp_path / 'plots' / 'exp_a' / 'plots'
    assert captured_outputs[1].destination == tmp_path / 'plots' / 'exp_b' / 'plots'


def test_generate_plots_defaults_output_dir_to_analysis_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    analysis_root = tmp_path / 'analysis'
    (analysis_root / 'exp_1').mkdir(parents=True)
    captured_outputs: list[generate_plots_cli.StagedOutput] = []

    def _fake_plot_analysis_outputs(
        *,
        curve_rows: list[dict[str, object]] | None = None,
        frontier_rows: list[dict[str, object]] | None = None,
        analysis_out: str | Path | None = None,
        perf_key: str,
        mode: str,
        save_dir: str | Path | None = None,
    ) -> list[Path]:
        save_path = Path(save_dir) / 'plot.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text('plot', encoding='utf-8')
        return [save_path]

    def _fake_finalize_staged_outputs(
        *,
        outputs: list[generate_plots_cli.StagedOutput],
        failures: list[CliFailure],
        allow_partial: bool,
        overwrite: bool,
    ) -> int:
        captured_outputs.extend(list(outputs))
        return len(outputs)

    monkeypatch.setattr(generate_plots_cli, 'plot_analysis_outputs', _fake_plot_analysis_outputs)
    monkeypatch.setattr(generate_plots_cli, 'finalize_staged_outputs', _fake_finalize_staged_outputs)
    monkeypatch.setattr(generate_plots_cli, 'print_failure_summary', lambda **kwargs: None)
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'regain-generate-plots',
            '--experiments',
            'exp_1',
            '--analysis-dir',
            str(analysis_root),
            '--save',
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        generate_plots_cli.main()

    assert int(exc_info.value.code) == 0
    assert len(captured_outputs) == 1
    assert captured_outputs[0].destination == analysis_root / 'exp_1' / 'plots'


def test_generate_plots_rejects_legacy_save_dir_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    analysis_root = tmp_path / 'analysis'
    (analysis_root / 'exp_1').mkdir(parents=True)

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'regain-generate-plots',
            '--experiments',
            'exp_1',
            '--analysis-dir',
            str(analysis_root),
            '--save-dir',
            str(tmp_path / 'plots'),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        generate_plots_cli.main()

    assert int(exc_info.value.code) == 2
