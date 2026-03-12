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

    def test_rejects_export_dir_flag(self) -> None:
        parser = _build_arg_parser()

        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    '--config-files',
                    'a.yaml',
                    '--export-dir',
                    '/tmp/exports',
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


########################
# Execution behavior   #
########################


class TestRunExecution:
    def test_runs_all_config_files(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run_calls: list[str] = []

        def _fake_run_experiment(config_file: str) -> None:
            run_calls.append(config_file)

        monkeypatch.setattr(run_experiment_cli, '_ensure_prerequisites', lambda: None)
        monkeypatch.setattr(run_experiment_cli, '_run_experiment', _fake_run_experiment)
        monkeypatch.setattr(
            sys,
            'argv',
            [
                'regain-run-experiment',
                '--config-files',
                'a.yaml,b.yaml,c.yaml',
            ],
        )

        run_experiment_cli.main()

        assert run_calls == ['a.yaml', 'b.yaml', 'c.yaml']

    def test_runs_config_dir_entries(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run_calls: list[str] = []

        def _fake_run_experiment(config_file: str) -> None:
            run_calls.append(config_file)

        monkeypatch.setattr(run_experiment_cli, '_ensure_prerequisites', lambda: None)
        monkeypatch.setattr(run_experiment_cli, '_run_experiment', _fake_run_experiment)
        monkeypatch.setattr(run_experiment_cli, '_find_config_files', lambda **kwargs: ['a.yaml', 'b.yaml'])
        monkeypatch.setattr(
            sys,
            'argv',
            [
                'regain-run-experiment',
                '--config-dir',
                '/tmp/configs',
            ],
        )

        run_experiment_cli.main()

        assert run_calls == ['a.yaml', 'b.yaml']
