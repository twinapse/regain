"""
Tests for plot generation CLI behavior.
"""

import json
from pathlib import Path
import sys

import pytest

import regain.analysis.plotting as plotting_module
from regain.analysis.plotting import PlotAnalysisResult
from regain.cli._utils.output_helpers import CliFailure
import regain.cli.generate_plots as generate_plots_cli


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
        frontier_rows: list[dict[str, object]] | None = None,
        impact_rows: list[dict[str, object]] | None = None,
        analysis_out: str | Path | None = None,
        mode: str,
        save_dir: str | Path | None = None,
    ) -> PlotAnalysisResult:
        del frontier_rows, impact_rows, analysis_out
        if mode in ['save', 'both'] and save_dir is not None:
            save_path = Path(save_dir) / 'plot.png'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text('plot', encoding='utf-8')
            return PlotAnalysisResult(
                saved_paths=[save_path],
                saved_filenames=[save_path.name],
                skipped=[],
            )
        return PlotAnalysisResult(
            saved_paths=[],
            saved_filenames=[],
            skipped=[],
        )

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
    frontier_dir = analysis_root / 'exp_1' / 'frontier'
    frontier_dir.mkdir(parents=True, exist_ok=True)
    (frontier_dir / 'manifest.json').write_text('{"plots": {"saved": [], "skipped": []}}', encoding='utf-8')
    captured_outputs: list[generate_plots_cli.StagedOutput] = []

    def _fake_plot_analysis_outputs(
        *,
        frontier_rows: list[dict[str, object]] | None = None,
        impact_rows: list[dict[str, object]] | None = None,
        analysis_out: str | Path | None = None,
        mode: str,
        save_dir: str | Path | None = None,
    ) -> PlotAnalysisResult:
        del frontier_rows, impact_rows, analysis_out, mode
        save_path = Path(save_dir) / 'plot.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text('plot', encoding='utf-8')
        return PlotAnalysisResult(
            saved_paths=[save_path],
            saved_filenames=[save_path.name],
            skipped=[],
        )

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
    assert len(captured_outputs) == 2
    destinations = {output.destination for output in captured_outputs}
    assert destinations == {
        analysis_root / 'exp_1' / 'plots',
        analysis_root / 'exp_1' / 'frontier' / 'manifest.json',
    }


def test_generate_plots_save_requires_existing_frontier_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    analysis_root = tmp_path / 'analysis'
    analysis_out = analysis_root / 'exp_1'
    analysis_out.mkdir(parents=True, exist_ok=True)

    def _fake_plot_analysis_outputs(
        *,
        frontier_rows: list[dict[str, object]] | None = None,
        impact_rows: list[dict[str, object]] | None = None,
        analysis_out: str | Path | None = None,
        mode: str,
        save_dir: str | Path | None = None,
    ) -> PlotAnalysisResult:
        del frontier_rows, impact_rows, analysis_out, mode
        save_path = Path(save_dir) / 'plot.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text('plot', encoding='utf-8')
        return PlotAnalysisResult(
            saved_paths=[save_path],
            saved_filenames=[save_path.name],
            skipped=[],
        )

    monkeypatch.setattr(generate_plots_cli, 'plot_analysis_outputs', _fake_plot_analysis_outputs)
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

    assert int(exc_info.value.code) == 1
    assert not (analysis_out / 'plots' / 'plot.png').exists()


def test_generate_plots_save_updates_manifest_when_publish_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    analysis_root = tmp_path / 'analysis'
    analysis_out = analysis_root / 'exp_1'
    frontier_dir = analysis_out / 'frontier'
    frontier_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = frontier_dir / 'manifest.json'
    manifest_path.write_text(
        '{"plots": {"saved": ["old.png"], "skipped": []}}',
        encoding='utf-8',
    )

    def _fake_plot_analysis_outputs(
        *,
        frontier_rows: list[dict[str, object]] | None = None,
        impact_rows: list[dict[str, object]] | None = None,
        analysis_out: str | Path | None = None,
        mode: str,
        save_dir: str | Path | None = None,
    ) -> PlotAnalysisResult:
        del frontier_rows, impact_rows, analysis_out, mode
        save_path = Path(save_dir) / 'plot.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text('plot', encoding='utf-8')
        return PlotAnalysisResult(
            saved_paths=[save_path],
            saved_filenames=[save_path.name],
            skipped=[{
                'filename': 'utility_delta__a__b.png',
                'reason': 'No overlapping utility values.',
                'context': {},
            }],
        )

    monkeypatch.setattr(generate_plots_cli, 'plot_analysis_outputs', _fake_plot_analysis_outputs)
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
    assert (analysis_out / 'plots' / 'plot.png').exists()
    updated_manifest = manifest_path.read_text(encoding='utf-8')
    assert '"saved": [' in updated_manifest
    assert '"plot.png"' in updated_manifest
    assert '"utility_delta__a__b.png"' in updated_manifest


def test_generate_plots_save_records_skipped_manifest_metadata_when_all_plots_are_skipped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    analysis_root = tmp_path / 'analysis'
    analysis_out = analysis_root / 'exp_1'
    frontier_dir = analysis_out / 'frontier'
    frontier_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = frontier_dir / 'manifest.json'
    manifest_path.write_text(
        '{"plots": {"saved": ["old.png"], "skipped": []}}',
        encoding='utf-8',
    )

    def _fake_plot_analysis_outputs(
        *,
        frontier_rows: list[dict[str, object]] | None = None,
        impact_rows: list[dict[str, object]] | None = None,
        analysis_out: str | Path | None = None,
        mode: str,
        save_dir: str | Path | None = None,
    ) -> PlotAnalysisResult:
        del frontier_rows, impact_rows, analysis_out, mode, save_dir
        return PlotAnalysisResult(
            saved_paths=[],
            saved_filenames=[],
            skipped=[{
                'filename': 'harm_vs_recovery.png',
                'reason': 'No valid frontier rows.',
                'context': {},
            }],
        )

    monkeypatch.setattr(generate_plots_cli, 'plot_analysis_outputs', _fake_plot_analysis_outputs)
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
    updated_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert updated_manifest['plots']['saved'] == []
    assert updated_manifest['plots']['skipped'] == [{
        'filename': 'harm_vs_recovery.png',
        'reason': 'No valid frontier rows.',
        'context': {},
    }]
    assert not (analysis_out / 'plots').exists()


def test_generate_plots_save_does_not_update_manifest_when_publish_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    analysis_root = tmp_path / 'analysis'
    analysis_out = analysis_root / 'exp_1'
    frontier_dir = analysis_out / 'frontier'
    frontier_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = frontier_dir / 'manifest.json'
    original_manifest = '{"plots": {"saved": ["before.png"], "skipped": []}}'
    manifest_path.write_text(original_manifest, encoding='utf-8')
    existing_plots_dir = analysis_out / 'plots'
    existing_plots_dir.mkdir(parents=True, exist_ok=True)
    (existing_plots_dir / 'existing.png').write_text('existing', encoding='utf-8')

    def _fake_plot_analysis_outputs(
        *,
        frontier_rows: list[dict[str, object]] | None = None,
        impact_rows: list[dict[str, object]] | None = None,
        analysis_out: str | Path | None = None,
        mode: str,
        save_dir: str | Path | None = None,
    ) -> PlotAnalysisResult:
        del frontier_rows, impact_rows, analysis_out, mode
        save_path = Path(save_dir) / 'plot.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text('plot', encoding='utf-8')
        return PlotAnalysisResult(
            saved_paths=[save_path],
            saved_filenames=[save_path.name],
            skipped=[],
        )

    monkeypatch.setattr(generate_plots_cli, 'plot_analysis_outputs', _fake_plot_analysis_outputs)
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

    assert int(exc_info.value.code) == 1
    assert manifest_path.read_text(encoding='utf-8') == original_manifest


def test_generate_plots_allow_partial_does_not_update_manifest_when_plots_destination_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    analysis_root = tmp_path / 'analysis'
    analysis_out = analysis_root / 'exp_1'
    frontier_dir = analysis_out / 'frontier'
    frontier_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = frontier_dir / 'manifest.json'
    original_manifest = '{"plots": {"saved": ["before.png"], "skipped": []}}'
    manifest_path.write_text(original_manifest, encoding='utf-8')

    existing_plots_dir = analysis_out / 'plots'
    existing_plots_dir.mkdir(parents=True, exist_ok=True)
    (existing_plots_dir / 'existing.png').write_text('existing', encoding='utf-8')

    def _fake_plot_analysis_outputs(
        *,
        frontier_rows: list[dict[str, object]] | None = None,
        impact_rows: list[dict[str, object]] | None = None,
        analysis_out: str | Path | None = None,
        mode: str,
        save_dir: str | Path | None = None,
    ) -> PlotAnalysisResult:
        del frontier_rows, impact_rows, analysis_out, mode
        save_path = Path(save_dir) / 'plot.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text('plot', encoding='utf-8')
        return PlotAnalysisResult(
            saved_paths=[save_path],
            saved_filenames=[save_path.name],
            skipped=[],
        )

    monkeypatch.setattr(generate_plots_cli, 'plot_analysis_outputs', _fake_plot_analysis_outputs)
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
            '--allow-partial',
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        generate_plots_cli.main()

    assert int(exc_info.value.code) == 1
    assert manifest_path.read_text(encoding='utf-8') == original_manifest


def test_generate_plots_output_dir_does_not_mutate_source_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    analysis_root = tmp_path / 'analysis'
    analysis_out = analysis_root / 'exp_1'
    frontier_dir = analysis_out / 'frontier'
    frontier_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = frontier_dir / 'manifest.json'
    original_manifest = '{"plots": {"saved": ["source.png"], "skipped": []}}'
    manifest_path.write_text(original_manifest, encoding='utf-8')
    output_root = tmp_path / 'external_plots'

    def _fake_plot_analysis_outputs(
        *,
        frontier_rows: list[dict[str, object]] | None = None,
        impact_rows: list[dict[str, object]] | None = None,
        analysis_out: str | Path | None = None,
        mode: str,
        save_dir: str | Path | None = None,
    ) -> PlotAnalysisResult:
        del frontier_rows, impact_rows, analysis_out, mode
        save_path = Path(save_dir) / 'plot.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text('plot', encoding='utf-8')
        return PlotAnalysisResult(
            saved_paths=[save_path],
            saved_filenames=[save_path.name],
            skipped=[{
                'filename': 'harm_vs_recovery.png',
                'reason': 'No valid frontier rows.',
                'context': {},
            }],
        )

    monkeypatch.setattr(generate_plots_cli, 'plot_analysis_outputs', _fake_plot_analysis_outputs)
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
            '--output-dir',
            str(output_root),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        generate_plots_cli.main()

    assert int(exc_info.value.code) == 0
    assert (output_root / 'exp_1' / 'plots' / 'plot.png').exists()
    assert manifest_path.read_text(encoding='utf-8') == original_manifest


def test_generate_plots_output_dir_with_all_skipped_does_not_mutate_source_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    analysis_root = tmp_path / 'analysis'
    analysis_out = analysis_root / 'exp_1'
    frontier_dir = analysis_out / 'frontier'
    frontier_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = frontier_dir / 'manifest.json'
    original_manifest = '{"plots": {"saved": ["source.png"], "skipped": []}}'
    manifest_path.write_text(original_manifest, encoding='utf-8')
    output_root = tmp_path / 'external_plots'

    def _fake_plot_analysis_outputs(
        *,
        frontier_rows: list[dict[str, object]] | None = None,
        impact_rows: list[dict[str, object]] | None = None,
        analysis_out: str | Path | None = None,
        mode: str,
        save_dir: str | Path | None = None,
    ) -> PlotAnalysisResult:
        del frontier_rows, impact_rows, analysis_out, mode, save_dir
        return PlotAnalysisResult(
            saved_paths=[],
            saved_filenames=[],
            skipped=[{
                'filename': 'harm_vs_recovery.png',
                'reason': 'No valid frontier rows.',
                'context': {},
            }],
        )

    monkeypatch.setattr(generate_plots_cli, 'plot_analysis_outputs', _fake_plot_analysis_outputs)
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
            '--output-dir',
            str(output_root),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        generate_plots_cli.main()

    assert int(exc_info.value.code) == 0
    assert manifest_path.read_text(encoding='utf-8') == original_manifest
    assert not (output_root / 'exp_1' / 'plots').exists()


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


def test_generate_plots_rejects_removed_perf_key_flag() -> None:
    parser = generate_plots_cli.argparse.ArgumentParser(prog='regain-generate-plots')
    generate_plots_cli.add_experiment_selector_arguments(parser=parser)
    parser.add_argument('--analysis-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--overwrite', action='store_true')

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                '--experiments',
                'exp_1',
                '--analysis-dir',
                '/tmp/analysis',
                '--perf-key',
                'analysis.repair.rho.avg',
            ]
        )


def test_plot_analysis_outputs_uses_action_cost_for_utility_plot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import matplotlib.pyplot as plt

    figures: list[object] = []

    class _FakeAxis:
        def __init__(self) -> None:
            self.xlabel: str | None = None
            self.ylabel: str | None = None
            self.title: str | None = None
            self.scatter_calls: list[dict[str, object]] = []

        def plot(self, *args, **kwargs) -> None:
            del args, kwargs

        def scatter(self, xs, ys, label=None) -> None:
            self.scatter_calls.append({
                'xs': list(xs),
                'ys': list(ys),
                'label': label,
            })

        def set_xlabel(self, value: str) -> None:
            self.xlabel = value

        def set_ylabel(self, value: str) -> None:
            self.ylabel = value

        def set_title(self, value: str) -> None:
            self.title = value

        def legend(self) -> None:
            return None

        def axhline(self, *args, **kwargs) -> None:
            del args, kwargs

    class _FakeFigure:
        def __init__(self) -> None:
            self.axis = _FakeAxis()
            figures.append(self)

        def add_subplot(self, *_args, **_kwargs) -> _FakeAxis:
            return self.axis

        def savefig(self, *args, **kwargs) -> None:
            del args, kwargs

    monkeypatch.setattr(plt, 'figure', lambda: _FakeFigure())
    monkeypatch.setattr(plt, 'close', lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, 'show', lambda: None)

    plotting_module.plot_analysis_outputs(
        frontier_rows=[
            {
                'scenario': 'cifar100',
                'strategy_name': 'er',
                'controller_name': 'repair_a',
                'controller_id': 'repair_a',
                'seed': 1,
                'b': 0.5,
                'repair_budget_fraction': 0.5,
                'action_repair_budget_fraction': 0.5,
                'mean_absolute_recovery': 0.2,
                'mean_harmed_task_fraction': 0.1,
                'utility_conservative': 0.15,
            },
            {
                'scenario': 'cifar100',
                'strategy_name': 'er',
                'controller_name': 'no-op',
                'controller_id': 'no_op',
                'seed': 1,
                'b': 0.5,
                'repair_budget_fraction': 0.5,
                'action_repair_budget_fraction': 0.0,
                'mean_absolute_recovery': 0.0,
                'mean_harmed_task_fraction': 0.0,
                'utility_conservative': 0.0,
            },
            {
                'scenario': 'cifar100',
                'strategy_name': 'er',
                'controller_name': 'legacy-only',
                'controller_id': 'legacy_only',
                'seed': 1,
                'b': 0.5,
                'repair_budget_fraction': 0.25,
                'mean_absolute_recovery': 0.05,
                'mean_harmed_task_fraction': 0.05,
                'utility_conservative': 0.03,
            },
        ],
        impact_rows=[],
        mode='none',
    )

    utility_axes = [
        figure.axis
        for figure in figures
        if figure.axis.title == 'Utility vs cost'
    ]
    assert len(utility_axes) == 1
    utility_axis = utility_axes[0]
    assert utility_axis.xlabel == 'action_repair_budget_fraction'
    no_op_call = next(
        call for call in utility_axis.scatter_calls
        if call['label'] == 'no-op'
    )
    assert no_op_call['xs'] == [0.0]
    assert {call['label'] for call in utility_axis.scatter_calls} == {'repair_a', 'no-op'}
