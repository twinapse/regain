"""
Tests for analysis runner CLI behavior.
"""

import json
from pathlib import Path
import sys

import pytest

from regain.analysis.plotting import PlotAnalysisResult
from regain.cli._utils.output_helpers import CliFailure
import regain.cli.run_analysis as run_analysis_cli


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
        (out_dir / 'run_metrics.jsonl').write_text('{}\n', encoding='utf-8')
        (out_dir / 'experience_metrics.jsonl').write_text('{}\n', encoding='utf-8')
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


def test_frontier_uses_collect_outputs_directly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_frontier_calls: list[dict[str, object]] = []
    captured_outputs: list[run_analysis_cli.StagedOutput] = []

    def _fake_collect_experiment_tables(
        **kwargs,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, str]]]:
        out_dir = Path(kwargs['out_dir'])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'run_metrics.jsonl').write_text('{}\n', encoding='utf-8')
        (out_dir / 'experience_metrics.jsonl').write_text('{}\n', encoding='utf-8')
        return (
            [{'run_id': 'run_1', 'controller_name': 'repair_a', 'b': 0.5}],
            [{'run_id': 'run_1', 'controller_name': 'repair_a', 'exp_idx': 0}],
            [],
        )

    def _fake_write_repairability_frontier_outputs(
        *,
        runs_table: list[dict[str, object]],
        experiences_table: list[dict[str, object]],
        out_dir: str | Path,
    ) -> dict[str, Path]:
        captured_frontier_calls.append({
            'runs_table': runs_table,
            'experiences_table': experiences_table,
            'out_dir': Path(out_dir),
        })
        frontier_dir = Path(out_dir) / 'frontier'
        tables_dir = Path(out_dir) / 'tables'
        frontier_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)
        repair_frontier_path = frontier_dir / 'candidates.csv'
        repair_frontier_path.write_text('controller_name\nrepair_a\n', encoding='utf-8')
        repair_pareto_path = frontier_dir / 'pareto.csv'
        repair_pareto_path.write_text('controller_name\nrepair_a\n', encoding='utf-8')
        repair_selection_path = frontier_dir / 'selection.csv'
        repair_selection_path.write_text('best_controller_by_utility_primary\nrepair_a\n', encoding='utf-8')
        repair_impact_path = frontier_dir / 'impact.csv'
        repair_impact_path.write_text('scenario\ncifar100\n', encoding='utf-8')
        manifest_path = frontier_dir / 'manifest.json'
        manifest_path.write_text('{}', encoding='utf-8')
        repair_outcomes_path = tables_dir / 'repair_outcomes.jsonl'
        repair_outcomes_path.write_text('{}\n', encoding='utf-8')
        return {
            'repair_outcomes': repair_outcomes_path,
            'candidates': repair_frontier_path,
            'pareto': repair_pareto_path,
            'impact': repair_impact_path,
            'selection': repair_selection_path,
            'manifest': manifest_path,
        }

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
    monkeypatch.setattr(
        run_analysis_cli,
        'write_repairability_frontier_outputs',
        _fake_write_repairability_frontier_outputs,
    )
    monkeypatch.setattr(run_analysis_cli, 'finalize_staged_outputs', _fake_finalize_staged_outputs)
    monkeypatch.setattr(run_analysis_cli, 'print_failure_summary', lambda **kwargs: None)
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'regain-analysis-tool',
            '--experiments',
            'exp_1',
            '--output-dir',
            str(tmp_path),
            'frontier',
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        run_analysis_cli.main()

    assert int(exc_info.value.code) == 0
    assert len(captured_frontier_calls) == 1
    assert captured_frontier_calls[0]['runs_table'] == [{'run_id': 'run_1', 'controller_name': 'repair_a', 'b': 0.5}]
    assert captured_frontier_calls[0]['experiences_table'] == [
        {'run_id': 'run_1', 'controller_name': 'repair_a', 'exp_idx': 0}
    ]
    destinations = {output.destination for output in captured_outputs}
    assert destinations == {
        tmp_path / 'exp_1' / 'tables',
        tmp_path / 'exp_1' / 'frontier',
    }


def test_run_analysis_rejects_removed_perf_key_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'regain-analysis-tool',
            '--experiments',
            'exp_1',
            '--output-dir',
            str(tmp_path),
            '--perf-key',
            'analysis.repair.rho.avg',
            'frontier',
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        run_analysis_cli.main()

    assert int(exc_info.value.code) == 2


def test_run_analysis_save_plots_writes_manifest_plot_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_root = tmp_path / 'analysis_outputs'

    def _fake_collect_experiment_tables(
        **kwargs,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, str]]]:
        out_dir = Path(kwargs['out_dir'])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'run_metrics.jsonl').write_text('{}\n', encoding='utf-8')
        (out_dir / 'experience_metrics.jsonl').write_text('{}\n', encoding='utf-8')
        return ([{'run_id': 'run_1'}], [{'run_id': 'run_1', 'exp_idx': 0}], [])

    def _fake_write_repairability_frontier_outputs(
        *,
        runs_table: list[dict[str, object]],
        experiences_table: list[dict[str, object]],
        out_dir: str | Path,
    ) -> dict[str, Path]:
        del runs_table, experiences_table
        frontier_dir = Path(out_dir) / 'frontier'
        tables_dir = Path(out_dir) / 'tables'
        frontier_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)
        repair_frontier_path = frontier_dir / 'candidates.csv'
        repair_frontier_path.write_text('controller_name\nrepair_a\n', encoding='utf-8')
        repair_pareto_path = frontier_dir / 'pareto.csv'
        repair_pareto_path.write_text('controller_name\nrepair_a\n', encoding='utf-8')
        repair_selection_path = frontier_dir / 'selection.csv'
        repair_selection_path.write_text('best_controller_by_utility_primary\nrepair_a\n', encoding='utf-8')
        repair_impact_path = frontier_dir / 'impact.csv'
        repair_impact_path.write_text('scenario\ncifar100\n', encoding='utf-8')
        manifest_path = frontier_dir / 'manifest.json'
        manifest_path.write_text(
            '{"plots": {"saved": [], "skipped": []}}',
            encoding='utf-8',
        )
        repair_outcomes_path = tables_dir / 'repair_outcomes.jsonl'
        repair_outcomes_path.write_text('{}\n', encoding='utf-8')
        return {
            'repair_outcomes': repair_outcomes_path,
            'candidates': repair_frontier_path,
            'pareto': repair_pareto_path,
            'impact': repair_impact_path,
            'selection': repair_selection_path,
            'manifest': manifest_path,
        }

    def _fake_plot_analysis_outputs(
        *,
        frontier_rows: list[dict[str, object]] | None = None,
        impact_rows: list[dict[str, object]] | None = None,
        analysis_out: str | Path | None = None,
        mode: str,
        save_dir: str | Path | None = None,
    ) -> PlotAnalysisResult:
        del frontier_rows, impact_rows, mode
        save_root = Path(save_dir) if save_dir is not None else (Path(analysis_out) / 'plots')
        save_path = save_root / 'plot.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text('plot', encoding='utf-8')
        return PlotAnalysisResult(
            saved_paths=[save_path],
            saved_filenames=[save_path.name],
            skipped=[{
                'filename': 'utility_delta__a__b.png',
                'reason': 'No overlapping settings.',
                'context': {},
            }],
        )

    monkeypatch.setattr(run_analysis_cli, 'collect_experiment_tables', _fake_collect_experiment_tables)
    monkeypatch.setattr(
        run_analysis_cli,
        'write_repairability_frontier_outputs',
        _fake_write_repairability_frontier_outputs,
    )
    monkeypatch.setattr(run_analysis_cli, 'plot_analysis_outputs', _fake_plot_analysis_outputs)
    monkeypatch.setattr(run_analysis_cli, 'print_failure_summary', lambda **kwargs: None)
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'regain-analysis-tool',
            '--experiments',
            'exp_1',
            '--output-dir',
            str(output_root),
            '--save-plots',
            'frontier',
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        run_analysis_cli.main()

    assert int(exc_info.value.code) == 0
    manifest_path = output_root / 'exp_1' / 'frontier' / 'manifest.json'
    manifest_text = manifest_path.read_text(encoding='utf-8')
    assert '"saved": [' in manifest_text
    assert '"plot.png"' in manifest_text
    assert '"utility_delta__a__b.png"' in manifest_text
    assert (output_root / 'exp_1' / 'plots' / 'plot.png').exists()


def test_run_analysis_save_plots_requires_existing_frontier_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_root = tmp_path / 'analysis_outputs'

    def _fake_collect_experiment_tables(
        **kwargs,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, str]]]:
        out_dir = Path(kwargs['out_dir'])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'run_metrics.jsonl').write_text('{}\n', encoding='utf-8')
        (out_dir / 'experience_metrics.jsonl').write_text('{}\n', encoding='utf-8')
        return ([{'run_id': 'run_1'}], [{'run_id': 'run_1', 'exp_idx': 0}], [])

    def _fake_write_repairability_frontier_outputs(
        *,
        runs_table: list[dict[str, object]],
        experiences_table: list[dict[str, object]],
        out_dir: str | Path,
    ) -> dict[str, Path]:
        del runs_table, experiences_table
        frontier_dir = Path(out_dir) / 'frontier'
        tables_dir = Path(out_dir) / 'tables'
        frontier_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)
        repair_frontier_path = frontier_dir / 'candidates.csv'
        repair_frontier_path.write_text('controller_name\nrepair_a\n', encoding='utf-8')
        repair_pareto_path = frontier_dir / 'pareto.csv'
        repair_pareto_path.write_text('controller_name\nrepair_a\n', encoding='utf-8')
        repair_selection_path = frontier_dir / 'selection.csv'
        repair_selection_path.write_text('best_controller_by_utility_primary\nrepair_a\n', encoding='utf-8')
        repair_impact_path = frontier_dir / 'impact.csv'
        repair_impact_path.write_text('scenario\ncifar100\n', encoding='utf-8')
        repair_outcomes_path = tables_dir / 'repair_outcomes.jsonl'
        repair_outcomes_path.write_text('{}\n', encoding='utf-8')
        return {
            'repair_outcomes': repair_outcomes_path,
            'candidates': repair_frontier_path,
            'pareto': repair_pareto_path,
            'impact': repair_impact_path,
            'selection': repair_selection_path,
            'manifest': frontier_dir / 'manifest.json',
        }

    def _fake_plot_analysis_outputs(
        *,
        frontier_rows: list[dict[str, object]] | None = None,
        impact_rows: list[dict[str, object]] | None = None,
        analysis_out: str | Path | None = None,
        mode: str,
        save_dir: str | Path | None = None,
    ) -> PlotAnalysisResult:
        del frontier_rows, impact_rows, mode
        save_root = Path(save_dir) if save_dir is not None else (Path(analysis_out) / 'plots')
        save_path = save_root / 'plot.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text('plot', encoding='utf-8')
        return PlotAnalysisResult(
            saved_paths=[save_path],
            saved_filenames=[save_path.name],
            skipped=[],
        )

    monkeypatch.setattr(run_analysis_cli, 'collect_experiment_tables', _fake_collect_experiment_tables)
    monkeypatch.setattr(
        run_analysis_cli,
        'write_repairability_frontier_outputs',
        _fake_write_repairability_frontier_outputs,
    )
    monkeypatch.setattr(run_analysis_cli, 'plot_analysis_outputs', _fake_plot_analysis_outputs)
    monkeypatch.setattr(run_analysis_cli, 'print_failure_summary', lambda **kwargs: None)
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'regain-analysis-tool',
            '--experiments',
            'exp_1',
            '--output-dir',
            str(output_root),
            '--save-plots',
            'frontier',
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        run_analysis_cli.main()

    assert int(exc_info.value.code) == 1
    assert not (output_root / 'exp_1' / 'plots' / 'plot.png').exists()


def test_run_analysis_save_plots_allow_partial_does_not_publish_unpublishable_plot_manifest_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_root = tmp_path / 'analysis_outputs'
    existing_plots_dir = output_root / 'exp_1' / 'plots'
    existing_plots_dir.mkdir(parents=True, exist_ok=True)
    (existing_plots_dir / 'existing.png').write_text('existing', encoding='utf-8')

    def _fake_collect_experiment_tables(
        **kwargs,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, str]]]:
        out_dir = Path(kwargs['out_dir'])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'run_metrics.jsonl').write_text('{}\n', encoding='utf-8')
        (out_dir / 'experience_metrics.jsonl').write_text('{}\n', encoding='utf-8')
        return ([{'run_id': 'run_1'}], [{'run_id': 'run_1', 'exp_idx': 0}], [])

    def _fake_write_repairability_frontier_outputs(
        *,
        runs_table: list[dict[str, object]],
        experiences_table: list[dict[str, object]],
        out_dir: str | Path,
    ) -> dict[str, Path]:
        del runs_table, experiences_table
        frontier_dir = Path(out_dir) / 'frontier'
        tables_dir = Path(out_dir) / 'tables'
        frontier_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)
        repair_frontier_path = frontier_dir / 'candidates.csv'
        repair_frontier_path.write_text('controller_name\nrepair_a\n', encoding='utf-8')
        repair_pareto_path = frontier_dir / 'pareto.csv'
        repair_pareto_path.write_text('controller_name\nrepair_a\n', encoding='utf-8')
        repair_selection_path = frontier_dir / 'selection.csv'
        repair_selection_path.write_text('best_controller_by_utility_primary\nrepair_a\n', encoding='utf-8')
        repair_impact_path = frontier_dir / 'impact.csv'
        repair_impact_path.write_text('scenario\ncifar100\n', encoding='utf-8')
        manifest_path = frontier_dir / 'manifest.json'
        manifest_path.write_text(
            '{"plots": {"saved": [], "skipped": []}}',
            encoding='utf-8',
        )
        repair_outcomes_path = tables_dir / 'repair_outcomes.jsonl'
        repair_outcomes_path.write_text('{}\n', encoding='utf-8')
        return {
            'repair_outcomes': repair_outcomes_path,
            'candidates': repair_frontier_path,
            'pareto': repair_pareto_path,
            'impact': repair_impact_path,
            'selection': repair_selection_path,
            'manifest': manifest_path,
        }

    def _fake_plot_analysis_outputs(
        *,
        frontier_rows: list[dict[str, object]] | None = None,
        impact_rows: list[dict[str, object]] | None = None,
        analysis_out: str | Path | None = None,
        mode: str,
        save_dir: str | Path | None = None,
    ) -> PlotAnalysisResult:
        del frontier_rows, impact_rows, mode
        save_root = Path(save_dir) if save_dir is not None else (Path(analysis_out) / 'plots')
        save_path = save_root / 'plot.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text('plot', encoding='utf-8')
        return PlotAnalysisResult(
            saved_paths=[save_path],
            saved_filenames=[save_path.name],
            skipped=[],
        )

    monkeypatch.setattr(run_analysis_cli, 'collect_experiment_tables', _fake_collect_experiment_tables)
    monkeypatch.setattr(
        run_analysis_cli,
        'write_repairability_frontier_outputs',
        _fake_write_repairability_frontier_outputs,
    )
    monkeypatch.setattr(run_analysis_cli, 'plot_analysis_outputs', _fake_plot_analysis_outputs)
    monkeypatch.setattr(run_analysis_cli, 'print_failure_summary', lambda **kwargs: None)
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'regain-analysis-tool',
            '--experiments',
            'exp_1',
            '--output-dir',
            str(output_root),
            '--save-plots',
            '--allow-partial',
            'frontier',
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        run_analysis_cli.main()

    assert int(exc_info.value.code) == 0
    manifest_path = output_root / 'exp_1' / 'frontier' / 'manifest.json'
    manifest_payload = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest_payload['plots']['saved'] == []
    assert manifest_payload['plots']['skipped'] == []
    assert not (output_root / 'exp_1' / 'plots' / 'plot.png').exists()


def test_run_analysis_save_plots_records_skipped_manifest_metadata_when_all_plots_are_skipped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_root = tmp_path / 'analysis_outputs'

    def _fake_collect_experiment_tables(
        **kwargs,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, str]]]:
        out_dir = Path(kwargs['out_dir'])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'run_metrics.jsonl').write_text('{}\n', encoding='utf-8')
        (out_dir / 'experience_metrics.jsonl').write_text('{}\n', encoding='utf-8')
        return ([{'run_id': 'run_1'}], [{'run_id': 'run_1', 'exp_idx': 0}], [])

    def _fake_write_repairability_frontier_outputs(
        *,
        runs_table: list[dict[str, object]],
        experiences_table: list[dict[str, object]],
        out_dir: str | Path,
    ) -> dict[str, Path]:
        del runs_table, experiences_table
        frontier_dir = Path(out_dir) / 'frontier'
        tables_dir = Path(out_dir) / 'tables'
        frontier_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)
        repair_frontier_path = frontier_dir / 'candidates.csv'
        repair_frontier_path.write_text('controller_name\nrepair_a\n', encoding='utf-8')
        repair_pareto_path = frontier_dir / 'pareto.csv'
        repair_pareto_path.write_text('controller_name\nrepair_a\n', encoding='utf-8')
        repair_selection_path = frontier_dir / 'selection.csv'
        repair_selection_path.write_text('best_controller_by_utility_primary\nrepair_a\n', encoding='utf-8')
        repair_impact_path = frontier_dir / 'impact.csv'
        repair_impact_path.write_text('scenario\ncifar100\n', encoding='utf-8')
        manifest_path = frontier_dir / 'manifest.json'
        manifest_path.write_text(
            '{"plots": {"saved": [], "skipped": []}}',
            encoding='utf-8',
        )
        repair_outcomes_path = tables_dir / 'repair_outcomes.jsonl'
        repair_outcomes_path.write_text('{}\n', encoding='utf-8')
        return {
            'repair_outcomes': repair_outcomes_path,
            'candidates': repair_frontier_path,
            'pareto': repair_pareto_path,
            'impact': repair_impact_path,
            'selection': repair_selection_path,
            'manifest': manifest_path,
        }

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

    monkeypatch.setattr(run_analysis_cli, 'collect_experiment_tables', _fake_collect_experiment_tables)
    monkeypatch.setattr(
        run_analysis_cli,
        'write_repairability_frontier_outputs',
        _fake_write_repairability_frontier_outputs,
    )
    monkeypatch.setattr(run_analysis_cli, 'plot_analysis_outputs', _fake_plot_analysis_outputs)
    monkeypatch.setattr(run_analysis_cli, 'print_failure_summary', lambda **kwargs: None)
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'regain-analysis-tool',
            '--experiments',
            'exp_1',
            '--output-dir',
            str(output_root),
            '--save-plots',
            'frontier',
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        run_analysis_cli.main()

    assert int(exc_info.value.code) == 0
    manifest_path = output_root / 'exp_1' / 'frontier' / 'manifest.json'
    manifest_payload = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest_payload['plots']['saved'] == []
    assert manifest_payload['plots']['skipped'] == [{
        'filename': 'harm_vs_recovery.png',
        'reason': 'No valid frontier rows.',
        'context': {},
    }]
    assert not (output_root / 'exp_1' / 'plots').exists()


def _stub_frontier_and_router(
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Install fake collect, frontier, and router writers for CLI tests.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """

    def _fake_collect_experiment_tables(
        **kwargs,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, str]]]:
        out_dir = Path(kwargs['out_dir'])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'run_metrics.jsonl').write_text('{}\n', encoding='utf-8')
        (out_dir / 'experience_metrics.jsonl').write_text('{}\n', encoding='utf-8')
        return ([{'run_id': 'run_1'}], [{'run_id': 'run_1', 'exp_idx': 0}], [])

    def _fake_write_repairability_frontier_outputs(
        *,
        runs_table: list[dict[str, object]],
        experiences_table: list[dict[str, object]],
        out_dir: str | Path,
    ) -> dict[str, Path]:
        del runs_table, experiences_table
        frontier_dir = Path(out_dir) / 'frontier'
        tables_dir = Path(out_dir) / 'tables'
        frontier_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)
        repair_frontier_path = frontier_dir / 'candidates.csv'
        repair_frontier_path.write_text('controller_name\nrepair_a\n', encoding='utf-8')
        repair_pareto_path = frontier_dir / 'pareto.csv'
        repair_pareto_path.write_text('controller_name\nrepair_a\n', encoding='utf-8')
        repair_selection_path = frontier_dir / 'selection.csv'
        repair_selection_path.write_text(
            'best_controller_by_utility_primary\nrepair_a\n',
            encoding='utf-8',
        )
        repair_impact_path = frontier_dir / 'impact.csv'
        repair_impact_path.write_text('scenario\ncifar100\n', encoding='utf-8')
        manifest_path = frontier_dir / 'manifest.json'
        manifest_path.write_text('{}', encoding='utf-8')
        repair_outcomes_path = tables_dir / 'repair_outcomes.jsonl'
        repair_outcomes_path.write_text('{}\n', encoding='utf-8')
        return {
            'repair_outcomes': repair_outcomes_path,
            'candidates': repair_frontier_path,
            'pareto': repair_pareto_path,
            'impact': repair_impact_path,
            'selection': repair_selection_path,
            'manifest': manifest_path,
        }

    def _fake_write_repair_router_outputs(
        *,
        analysis_dir: Path,
        out_dir: Path,
        random_state: int = 0,
    ) -> dict[str, Path]:
        del analysis_dir, random_state
        router_dir = Path(out_dir)
        router_dir.mkdir(parents=True, exist_ok=True)
        features_path = router_dir / 'features.csv'
        features_path.write_text('scenario\ncifar100\n', encoding='utf-8')
        labels_path = router_dir / 'labels.csv'
        labels_path.write_text('oracle_action_conservative\nno_op\n', encoding='utf-8')
        predictions_path = router_dir / 'predictions.csv'
        predictions_path.write_text('policy_name\nalways_no_op\n', encoding='utf-8')
        summary_path = router_dir / 'policy_summary.csv'
        summary_path.write_text('policy_name\nalways_no_op\n', encoding='utf-8')
        decision_gate_path = router_dir / 'decision_gate.json'
        decision_gate_path.write_text('{}', encoding='utf-8')
        manifest_path = router_dir / 'manifest.json'
        manifest_path.write_text('{}', encoding='utf-8')
        return {
            'features': features_path,
            'labels': labels_path,
            'predictions': predictions_path,
            'policy_summary': summary_path,
            'decision_gate': decision_gate_path,
            'manifest': manifest_path,
        }

    monkeypatch.setattr(run_analysis_cli, 'collect_experiment_tables', _fake_collect_experiment_tables)
    monkeypatch.setattr(
        run_analysis_cli,
        'write_repairability_frontier_outputs',
        _fake_write_repairability_frontier_outputs,
    )
    monkeypatch.setattr(
        run_analysis_cli,
        'write_repair_router_outputs',
        _fake_write_repair_router_outputs,
    )
    monkeypatch.setattr(run_analysis_cli, 'print_failure_summary', lambda **kwargs: None)


def test_router_command_publishes_frontier_and_router_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_outputs: list[run_analysis_cli.StagedOutput] = []

    def _fake_finalize_staged_outputs(
        *,
        outputs: list[run_analysis_cli.StagedOutput],
        failures: list[CliFailure],
        allow_partial: bool,
        overwrite: bool,
    ) -> int:
        captured_outputs.extend(list(outputs))
        return len(outputs)

    _stub_frontier_and_router(monkeypatch=monkeypatch)
    monkeypatch.setattr(run_analysis_cli, 'finalize_staged_outputs', _fake_finalize_staged_outputs)
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'regain-analysis-tool',
            '--experiments',
            'exp_1',
            '--output-dir',
            str(tmp_path),
            'router',
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        run_analysis_cli.main()

    assert int(exc_info.value.code) == 0
    destinations = {output.destination for output in captured_outputs}
    assert destinations == {
        tmp_path / 'exp_1' / 'tables',
        tmp_path / 'exp_1' / 'frontier',
        tmp_path / 'exp_1' / 'router',
    }


def test_all_command_includes_router_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_outputs: list[run_analysis_cli.StagedOutput] = []

    def _fake_finalize_staged_outputs(
        *,
        outputs: list[run_analysis_cli.StagedOutput],
        failures: list[CliFailure],
        allow_partial: bool,
        overwrite: bool,
    ) -> int:
        captured_outputs.extend(list(outputs))
        return len(outputs)

    _stub_frontier_and_router(monkeypatch=monkeypatch)
    monkeypatch.setattr(run_analysis_cli, 'finalize_staged_outputs', _fake_finalize_staged_outputs)
    monkeypatch.setattr(
        run_analysis_cli,
        'write_recoverability_curves',
        lambda **kwargs: (
            Path(kwargs['out_dir']) / 'recoverability_curve.csv',
            Path(kwargs['out_dir']) / 'task_age_rho.csv',
            Path(kwargs['out_dir']) / 'calibration_vs_budget.csv',
            Path(kwargs['out_dir']) / 'latency_vs_budget.csv',
        ),
    )
    monkeypatch.setattr(
        run_analysis_cli,
        'write_predictive_correlations',
        lambda **kwargs: Path(kwargs['out_dir']) / 'predictive_correlations.csv',
    )
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'regain-analysis-tool',
            '--experiments',
            'exp_1',
            '--output-dir',
            str(tmp_path),
            'all',
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        run_analysis_cli.main()

    assert int(exc_info.value.code) == 0
    destinations = {output.destination for output in captured_outputs}
    assert tmp_path / 'exp_1' / 'router' in destinations
    assert tmp_path / 'exp_1' / 'frontier' in destinations


def test_router_command_skips_when_frontier_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_failures: list[CliFailure] = []

    def _fake_collect_experiment_tables(
        **kwargs,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, str]]]:
        out_dir = Path(kwargs['out_dir'])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'run_metrics.jsonl').write_text('{}\n', encoding='utf-8')
        (out_dir / 'experience_metrics.jsonl').write_text('{}\n', encoding='utf-8')
        return ([{'run_id': 'run_1'}], [{'run_id': 'run_1', 'exp_idx': 0}], [])

    def _failing_frontier(**kwargs) -> dict[str, Path]:
        raise ValueError('forced frontier failure')

    def _fake_finalize_staged_outputs(
        *,
        outputs: list[run_analysis_cli.StagedOutput],
        failures: list[CliFailure],
        allow_partial: bool,
        overwrite: bool,
    ) -> int:
        del outputs, allow_partial, overwrite
        captured_failures.extend(list(failures))
        return 0

    monkeypatch.setattr(run_analysis_cli, 'collect_experiment_tables', _fake_collect_experiment_tables)
    monkeypatch.setattr(run_analysis_cli, 'write_repairability_frontier_outputs', _failing_frontier)
    monkeypatch.setattr(
        run_analysis_cli,
        'write_repair_router_outputs',
        lambda **kwargs: (_ for _ in ()).throw(AssertionError('router must not run')),
    )
    monkeypatch.setattr(run_analysis_cli, 'finalize_staged_outputs', _fake_finalize_staged_outputs)
    monkeypatch.setattr(run_analysis_cli, 'print_failure_summary', lambda **kwargs: None)
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'regain-analysis-tool',
            '--experiments',
            'exp_1',
            '--output-dir',
            str(tmp_path),
            'router',
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        run_analysis_cli.main()

    assert int(exc_info.value.code) == 1
    assert any('Skipped because frontier stage failed' in failure.message for failure in captured_failures)
