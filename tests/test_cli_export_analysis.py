"""
Tests for analysis export CLI behavior.
"""

import json
from pathlib import Path
import sys

import pytest

from regain.analysis.exports import export_analysis_to_json
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


def test_load_analysis_tables_uses_renamed_jsonl_inputs(tmp_path: Path) -> None:
    experiment_dir = tmp_path / 'analysis' / 'exp_1'
    tables_dir = experiment_dir / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / 'run_metrics.jsonl').write_text('{"run_id":"run_1"}\n', encoding='utf-8')
    (tables_dir / 'experience_metrics.jsonl').write_text('{"run_id":"run_1","exp_idx":0}\n', encoding='utf-8')

    runs_table, experiences_table = export_analysis_cli._load_analysis_tables(
        experiment_dir=experiment_dir,
    )

    assert runs_table == [{'run_id': 'run_1'}]
    assert experiences_table == [{'run_id': 'run_1', 'exp_idx': 0}]


def test_export_analysis_includes_no_op_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    experiment_dir = tmp_path / 'analysis' / 'exp_1'
    tables_dir = experiment_dir / 'tables'
    frontier_dir = experiment_dir / 'frontier'
    tables_dir.mkdir(parents=True, exist_ok=True)
    frontier_dir.mkdir(parents=True, exist_ok=True)

    (tables_dir / 'repair_outcomes.jsonl').write_text(
        '\n'.join([
            json.dumps({
                'controller_name': 'no-op',
                'controller_id': 'no_op',
                'source_stage': 'no_op',
                'is_no_op_action': True,
            }),
            '',
        ]),
        encoding='utf-8',
    )
    (frontier_dir / 'candidates.csv').write_text(
        (
            'controller_name,controller_id,is_no_op_action,action_repair_budget_fraction,utility_primary\n'
            'no-op,no_op,True,0.0,0.0\n'
        ),
        encoding='utf-8',
    )
    (frontier_dir / 'pareto.csv').write_text(
        'controller_name,controller_id\nno-op,no_op\n',
        encoding='utf-8',
    )
    (frontier_dir / 'impact.csv').write_text(
        'controller_name,controller_id\nno-op,no_op\n',
        encoding='utf-8',
    )
    (frontier_dir / 'selection.csv').write_text(
        'best_controller_by_utility_primary,utility_primary__no_op\nno_op,0.0\n',
        encoding='utf-8',
    )
    (frontier_dir / 'manifest.json').write_text('{}', encoding='utf-8')

    export_path = tmp_path / 'exports' / 'analysis.json'
    monkeypatch.setattr(
        'regain.analysis.exports.resolve_git_commit',
        lambda: 'd' * 40,
    )
    export_analysis_to_json(
        experiment='exp_1',
        experiment_dir=experiment_dir,
        export_path=export_path,
        tracking_uri=None,
        artifact_location=None,
        runs_table=[{'run_id': 'run_1'}],
        experiences_table=[{'run_id': 'run_1', 'exp_idx': 0}],
        include_controllers=None,
        exclude_controllers=None,
        max_runs=None,
        default_num_classes=None,
        require_finished=False,
    )

    payload = json.loads(export_path.read_text(encoding='utf-8'))
    assert payload['schema']['version'] == 4
    assert payload['mlflow']['git_commit'] == 'd' * 40
    assert payload['tables']['repair_outcomes'][0]['controller_name'] == 'no-op'
    assert payload['tables']['repair_outcomes'][0]['is_no_op_action'] is True
    assert payload['frontier']['candidates'][0]['controller_id'] == 'no_op'
    assert payload['frontier']['selection'][0]['best_controller_by_utility_primary'] == 'no_op'


def test_export_analysis_includes_router_section(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    experiment_dir = tmp_path / 'analysis' / 'exp_1'
    tables_dir = experiment_dir / 'tables'
    frontier_dir = experiment_dir / 'frontier'
    router_dir = experiment_dir / 'router'
    tables_dir.mkdir(parents=True, exist_ok=True)
    frontier_dir.mkdir(parents=True, exist_ok=True)
    router_dir.mkdir(parents=True, exist_ok=True)

    (tables_dir / 'repair_outcomes.jsonl').write_text('{}\n', encoding='utf-8')
    (frontier_dir / 'candidates.csv').write_text('controller_name\nno-op\n', encoding='utf-8')
    (frontier_dir / 'pareto.csv').write_text('controller_name\nno-op\n', encoding='utf-8')
    (frontier_dir / 'impact.csv').write_text('controller_name\nno-op\n', encoding='utf-8')
    (frontier_dir / 'selection.csv').write_text(
        'best_controller_by_utility_primary\nno_op\n',
        encoding='utf-8',
    )
    (frontier_dir / 'manifest.json').write_text('{}', encoding='utf-8')
    (router_dir / 'features.csv').write_text(
        'scenario,backbone_name\ncifar100,vit_small\n',
        encoding='utf-8',
    )
    (router_dir / 'labels.csv').write_text(
        'oracle_action_conservative\nno_op\n',
        encoding='utf-8',
    )
    (router_dir / 'predictions.csv').write_text(
        'policy_name,selected_action_id\nalways_no_op,no_op\n',
        encoding='utf-8',
    )
    (router_dir / 'policy_summary.csv').write_text(
        'policy_name,validation_level\nalways_no_op,held_seed\n',
        encoding='utf-8',
    )
    (router_dir / 'decision_gate.json').write_text(
        '{"levels": {"held_seed": {"success": false}}}',
        encoding='utf-8',
    )
    (router_dir / 'manifest.json').write_text(
        '{"schema": {"name": "regain.analysis.router", "version": 1}}',
        encoding='utf-8',
    )

    export_path = tmp_path / 'exports' / 'analysis.json'
    monkeypatch.setattr(
        'regain.analysis.exports.resolve_git_commit',
        lambda: 'e' * 40,
    )
    export_analysis_to_json(
        experiment='exp_1',
        experiment_dir=experiment_dir,
        export_path=export_path,
        tracking_uri=None,
        artifact_location=None,
        runs_table=[{'run_id': 'run_1'}],
        experiences_table=[{'run_id': 'run_1', 'exp_idx': 0}],
        include_controllers=None,
        exclude_controllers=None,
        max_runs=None,
        default_num_classes=None,
        require_finished=False,
    )

    payload = json.loads(export_path.read_text(encoding='utf-8'))
    assert payload['schema']['version'] == 4
    router_payload = payload['router']
    assert router_payload['features'][0]['scenario'] == 'cifar100'
    assert router_payload['labels'][0]['oracle_action_conservative'] == 'no_op'
    assert router_payload['predictions'][0]['policy_name'] == 'always_no_op'
    assert router_payload['policy_summary'][0]['policy_name'] == 'always_no_op'
    assert router_payload['decision_gate']['levels']['held_seed']['success'] is False
    assert router_payload['manifest']['schema']['version'] == 1
