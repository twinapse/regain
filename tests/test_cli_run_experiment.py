"""
Tests for experiment runner CLI.
"""

from pathlib import Path
import sys

import pytest

import regain.cli.run_experiment as run_experiment_cli
from regain.cli.run_experiment import _build_arg_parser
from regain.cli.run_experiment import _find_config_files


####################
# Parser validation #
####################


class TestRunExperimentParser:
    def test_rejects_config_files_and_config_dir_together(self) -> None:
        parser = _build_arg_parser()

        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    '--config-files',
                    'a.yaml,b.yaml',
                    '--config-dir',
                    'configs',
                ]
            )


#######################
# Config dir discovery #
#######################


class TestFindConfigFiles:
    def test_finds_yaml_files_recursively(self, tmp_path: Path) -> None:
        configs_dir = tmp_path / 'configs'
        nested_dir = configs_dir / 'nested'
        nested_dir.mkdir(parents=True)

        file_a = configs_dir / 'root.yaml'
        file_b = nested_dir / 'sub.yml'
        file_c = nested_dir / 'ignore.txt'

        file_a.write_text('experiment_name: root\n', encoding='utf-8')
        file_b.write_text('experiment_name: sub\n', encoding='utf-8')
        file_c.write_text('not a config\n', encoding='utf-8')

        config_files = _find_config_files(config_dir=str(configs_dir))

        assert config_files == sorted([str(file_a), str(file_b)])

    def test_raises_for_missing_directory(self, tmp_path: Path) -> None:
        missing_dir = tmp_path / 'missing'

        with pytest.raises(ValueError, match='does not exist'):
            _find_config_files(config_dir=str(missing_dir))

    def test_raises_for_non_directory_path(self, tmp_path: Path) -> None:
        file_path = tmp_path / 'single.yaml'
        file_path.write_text('experiment_name: only\n', encoding='utf-8')

        with pytest.raises(ValueError, match='is not a directory'):
            _find_config_files(config_dir=str(file_path))


##############################
# Grouped run export behavior #
##############################


class TestGroupedRunExports:
    def test_exports_once_per_experiment_group(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run_results = [
            ('shared_exp', 'mlflow://tracking'),
            ('shared_exp', 'mlflow://tracking'),
            ('other_exp', None),
        ]
        run_calls: list[str] = []
        export_calls: list[tuple[str, str | None]] = []

        def _fake_run_experiment(config_file: str) -> tuple[str, str | None]:
            run_calls.append(config_file)
            return run_results[len(run_calls) - 1]

        def _fake_export_runs_to_csvs(
            *,
            experiment_name: str,
            export_dir: str,
            tracking_uri: str | None,
        ) -> None:
            export_calls.append((experiment_name, tracking_uri))

        monkeypatch.setattr(run_experiment_cli, '_ensure_prerequisites', lambda: None)
        monkeypatch.setattr(run_experiment_cli, '_run_experiment', _fake_run_experiment)
        monkeypatch.setattr(run_experiment_cli, '_export_runs_to_csvs', _fake_export_runs_to_csvs)
        monkeypatch.setattr(
            sys,
            'argv',
            [
                'regain-run-experiment',
                '--config-files',
                'a.yaml,b.yaml,c.yaml',
                '--export-dir',
                '/tmp/exports',
            ],
        )

        run_experiment_cli.main()

        assert run_calls == ['a.yaml', 'b.yaml', 'c.yaml']
        assert export_calls == [
            ('shared_exp', 'mlflow://tracking'),
            ('other_exp', None),
        ]

    def test_rejects_mixed_tracking_uris_within_same_experiment(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run_results = [
            ('shared_exp', 'mlflow://one'),
            ('shared_exp', 'mlflow://two'),
        ]
        run_calls: list[str] = []
        export_calls: list[tuple[str, str | None]] = []

        def _fake_run_experiment(config_file: str) -> tuple[str, str | None]:
            run_calls.append(config_file)
            return run_results[len(run_calls) - 1]

        def _fake_export_runs_to_csvs(
            *,
            experiment_name: str,
            export_dir: str,
            tracking_uri: str | None,
        ) -> None:
            export_calls.append((experiment_name, tracking_uri))

        monkeypatch.setattr(run_experiment_cli, '_ensure_prerequisites', lambda: None)
        monkeypatch.setattr(run_experiment_cli, '_run_experiment', _fake_run_experiment)
        monkeypatch.setattr(run_experiment_cli, '_export_runs_to_csvs', _fake_export_runs_to_csvs)
        monkeypatch.setattr(
            sys,
            'argv',
            [
                'regain-run-experiment',
                '--config-files',
                'a.yaml,b.yaml',
                '--export-dir',
                '/tmp/exports',
            ],
        )

        with pytest.raises(SystemExit) as exc_info:
            run_experiment_cli.main()

        assert int(exc_info.value.code) == 1
        assert run_calls == ['a.yaml', 'b.yaml']
        assert export_calls == []
