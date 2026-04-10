"""
Tests for experiment runner CLI.
"""

from dataclasses import dataclass
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest

from regain.cli.run_experiment import _build_arg_parser
from regain.cli.run_experiment import _find_config_files
import regain.cli.run_experiment as run_experiment_cli
import regain.mlflow_utils as mlflow_utils


@dataclass
class _DummyBackboneConfig:
    source_experiment: str | None = None


@dataclass
class _DummyRunConfig:
    name: str


@dataclass
class _DummyExperimentConfig:
    experiment_name: str
    backbone: _DummyBackboneConfig | None
    runs: list[_DummyRunConfig] | None


def _make_active_run(
    *,
    run_id: str,
    status: str,
    start_time: int,
) -> object:
    return SimpleNamespace(
        info=SimpleNamespace(
            run_id=run_id,
            status=status,
            start_time=start_time,
        )
    )


def _patch_loader_and_orchestrator(
    *,
    monkeypatch: pytest.MonkeyPatch,
    experiment_config: _DummyExperimentConfig,
    run_calls: list[tuple[_DummyExperimentConfig, str | None, str | None]],
) -> None:
    def _fake_load_experiment_config(config_file: str) -> _DummyExperimentConfig:
        del config_file
        return experiment_config

    def _fake_orchestrator_run_experiment(
        config: _DummyExperimentConfig,
        *,
        tracking_uri: str | None = None,
        artifact_location: str | None = None,
    ) -> None:
        run_calls.append((config, tracking_uri, artifact_location))

    config_module = ModuleType('regain.experiments.config')
    config_module.load_experiment_config = _fake_load_experiment_config  # type: ignore[attr-defined]
    orchestrator_module = ModuleType('regain.experiments.orchestrator')
    orchestrator_module.run_experiment = _fake_orchestrator_run_experiment  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, 'regain.experiments.config', config_module)
    monkeypatch.setitem(sys.modules, 'regain.experiments.orchestrator', orchestrator_module)


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

    def test_rejects_resume_and_retry_together(self) -> None:
        parser = _build_arg_parser()

        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    '--config-files',
                    'a.yaml',
                    '--resume',
                    '--retry',
                ]
            )

    def test_rejects_resume_and_overwrite_together(self) -> None:
        parser = _build_arg_parser()

        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    '--config-files',
                    'a.yaml',
                    '--resume',
                    '--overwrite',
                ]
            )

    def test_rejects_retry_and_overwrite_together(self) -> None:
        parser = _build_arg_parser()

        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    '--config-files',
                    'a.yaml',
                    '--retry',
                    '--overwrite',
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
        run_calls: list[tuple[str, bool, bool, bool]] = []

        def _fake_run_experiment(
            config_file: str,
            *,
            tracking_uri: str | None = None,
            artifact_location: str | None = None,
            resume: bool = False,
            retry: bool = False,
            overwrite: bool = False,
        ) -> None:
            del tracking_uri
            del artifact_location
            run_calls.append((config_file, resume, retry, overwrite))

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

        assert run_calls == [
            ('a.yaml', False, False, False),
            ('b.yaml', False, False, False),
            ('c.yaml', False, False, False),
        ]

    def test_runs_config_dir_entries(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run_calls: list[tuple[str, bool, bool, bool]] = []

        def _fake_run_experiment(
            config_file: str,
            *,
            tracking_uri: str | None = None,
            artifact_location: str | None = None,
            resume: bool = False,
            retry: bool = False,
            overwrite: bool = False,
        ) -> None:
            del tracking_uri
            del artifact_location
            run_calls.append((config_file, resume, retry, overwrite))

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

        assert run_calls == [
            ('a.yaml', False, False, False),
            ('b.yaml', False, False, False),
        ]

    def test_forwards_retry_flag_to_run_experiment(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run_calls: list[tuple[str, bool, bool, bool]] = []

        def _fake_run_experiment(
            config_file: str,
            *,
            tracking_uri: str | None = None,
            artifact_location: str | None = None,
            resume: bool = False,
            retry: bool = False,
            overwrite: bool = False,
        ) -> None:
            del tracking_uri
            del artifact_location
            run_calls.append((config_file, resume, retry, overwrite))

        monkeypatch.setattr(run_experiment_cli, '_ensure_prerequisites', lambda: None)
        monkeypatch.setattr(run_experiment_cli, '_run_experiment', _fake_run_experiment)
        monkeypatch.setattr(
            sys,
            'argv',
            [
                'regain-run-experiment',
                '--config-files',
                'a.yaml',
                '--retry',
            ],
        )

        run_experiment_cli.main()

        assert run_calls == [('a.yaml', False, True, False)]

    def test_forwards_resume_flag_to_run_experiment(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run_calls: list[tuple[str, bool, bool, bool]] = []

        def _fake_run_experiment(
            config_file: str,
            *,
            tracking_uri: str | None = None,
            artifact_location: str | None = None,
            resume: bool = False,
            retry: bool = False,
            overwrite: bool = False,
        ) -> None:
            del tracking_uri
            del artifact_location
            run_calls.append((config_file, resume, retry, overwrite))

        monkeypatch.setattr(run_experiment_cli, '_ensure_prerequisites', lambda: None)
        monkeypatch.setattr(run_experiment_cli, '_run_experiment', _fake_run_experiment)
        monkeypatch.setattr(
            sys,
            'argv',
            [
                'regain-run-experiment',
                '--config-files',
                'a.yaml',
                '--resume',
            ],
        )

        run_experiment_cli.main()

        assert run_calls == [('a.yaml', True, False, False)]

    def test_forwards_overwrite_flag_to_run_experiment(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run_calls: list[tuple[str, bool, bool, bool]] = []

        def _fake_run_experiment(
            config_file: str,
            *,
            tracking_uri: str | None = None,
            artifact_location: str | None = None,
            resume: bool = False,
            retry: bool = False,
            overwrite: bool = False,
        ) -> None:
            del tracking_uri
            del artifact_location
            run_calls.append((config_file, resume, retry, overwrite))

        monkeypatch.setattr(run_experiment_cli, '_ensure_prerequisites', lambda: None)
        monkeypatch.setattr(run_experiment_cli, '_run_experiment', _fake_run_experiment)
        monkeypatch.setattr(
            sys,
            'argv',
            [
                'regain-run-experiment',
                '--config-files',
                'a.yaml',
                '--overwrite',
            ],
        )

        run_experiment_cli.main()

        assert run_calls == [('a.yaml', False, False, True)]

    def test_main_continues_after_failed_config_and_exits_nonzero(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run_calls: list[str] = []

        def _fake_run_experiment(
            config_file: str,
            *,
            tracking_uri: str | None = None,
            artifact_location: str | None = None,
            resume: bool = False,
            retry: bool = False,
            overwrite: bool = False,
        ) -> None:
            del tracking_uri
            del artifact_location
            del resume
            del retry
            del overwrite
            run_calls.append(config_file)
            if config_file == 'bad.yaml':
                raise RuntimeError('broken config')

        monkeypatch.setattr(run_experiment_cli, '_ensure_prerequisites', lambda: None)
        monkeypatch.setattr(run_experiment_cli, '_run_experiment', _fake_run_experiment)
        monkeypatch.setattr(
            sys,
            'argv',
            [
                'regain-run-experiment',
                '--config-files',
                'bad.yaml,good.yaml',
            ],
        )

        with pytest.raises(SystemExit) as exc_info:
            run_experiment_cli.main()

        assert int(exc_info.value.code) == 1
        assert run_calls == ['bad.yaml', 'good.yaml']


###########################
# Launch policy behavior  #
###########################


class TestRunExperimentPolicies:
    def test_resume_runs_only_missing_configs_and_reuses_backbone(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run_calls: list[tuple[_DummyExperimentConfig, str | None, str | None]] = []
        experiment_config = _DummyExperimentConfig(
            experiment_name='exp',
            backbone=_DummyBackboneConfig(source_experiment=None),
            runs=[
                _DummyRunConfig(name='run_a'),
                _DummyRunConfig(name='run_b'),
            ],
        )
        _patch_loader_and_orchestrator(
            monkeypatch=monkeypatch,
            experiment_config=experiment_config,
            run_calls=run_calls,
        )
        monkeypatch.setattr(
            mlflow_utils,
            'resolve_active_runs_by_name',
            lambda **kwargs: {
                'backbone': [_make_active_run(run_id='b1', status='FINISHED', start_time=50)],
                'run_a': [_make_active_run(run_id='a1', status='FINISHED', start_time=60)],
            },
        )
        deleted_runs: list[list[str]] = []
        monkeypatch.setattr(
            mlflow_utils,
            'delete_mlflow_runs',
            lambda **kwargs: deleted_runs.append(
                [str(run.info.run_id) for run in kwargs['runs']]
            ),
        )

        run_experiment_cli._run_experiment(
            'dummy.yaml',
            resume=True,
        )

        assert deleted_runs == [[]]
        assert len(run_calls) == 1
        launched_config = run_calls[0][0]
        assert launched_config.backbone is None
        assert [run_config.name for run_config in launched_config.runs or []] == ['run_b']

    def test_retry_runs_latest_failed_and_deletes_expected_runs(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run_calls: list[tuple[_DummyExperimentConfig, str | None, str | None]] = []
        experiment_config = _DummyExperimentConfig(
            experiment_name='exp',
            backbone=_DummyBackboneConfig(source_experiment=None),
            runs=[
                _DummyRunConfig(name='run_a'),
                _DummyRunConfig(name='run_b'),
            ],
        )
        _patch_loader_and_orchestrator(
            monkeypatch=monkeypatch,
            experiment_config=experiment_config,
            run_calls=run_calls,
        )
        monkeypatch.setattr(
            mlflow_utils,
            'resolve_active_runs_by_name',
            lambda **kwargs: {
                'backbone': [
                    _make_active_run(run_id='b_failed', status='FAILED', start_time=300),
                    _make_active_run(run_id='b_finished', status='FINISHED', start_time=100),
                ],
                'run_a': [
                    _make_active_run(run_id='a_failed', status='FAILED', start_time=200),
                    _make_active_run(run_id='a_finished', status='FINISHED', start_time=50),
                ],
                'run_b': [
                    _make_active_run(run_id='b_run', status='FINISHED', start_time=210),
                ],
            },
        )
        deleted_runs: list[list[str]] = []
        monkeypatch.setattr(
            mlflow_utils,
            'delete_mlflow_runs',
            lambda **kwargs: deleted_runs.append(
                [str(run.info.run_id) for run in kwargs['runs']]
            ),
        )

        run_experiment_cli._run_experiment(
            'dummy.yaml',
            retry=True,
        )

        assert deleted_runs == [['a_failed', 'b_failed', 'b_finished']]
        assert len(run_calls) == 1
        launched_config = run_calls[0][0]
        assert isinstance(launched_config.backbone, _DummyBackboneConfig)
        assert [run_config.name for run_config in launched_config.runs or []] == ['run_a']

    def test_overwrite_runs_all_selected_and_deletes_all_active(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run_calls: list[tuple[_DummyExperimentConfig, str | None, str | None]] = []
        experiment_config = _DummyExperimentConfig(
            experiment_name='exp',
            backbone=_DummyBackboneConfig(source_experiment=None),
            runs=[
                _DummyRunConfig(name='run_a'),
                _DummyRunConfig(name='run_b'),
            ],
        )
        _patch_loader_and_orchestrator(
            monkeypatch=monkeypatch,
            experiment_config=experiment_config,
            run_calls=run_calls,
        )
        monkeypatch.setattr(
            mlflow_utils,
            'resolve_active_runs_by_name',
            lambda **kwargs: {
                'backbone': [_make_active_run(run_id='bone1', status='FINISHED', start_time=100)],
                'run_a': [_make_active_run(run_id='a1', status='FINISHED', start_time=110)],
                'run_b': [_make_active_run(run_id='b1', status='FAILED', start_time=120)],
            },
        )
        deleted_runs: list[list[str]] = []
        monkeypatch.setattr(
            mlflow_utils,
            'delete_mlflow_runs',
            lambda **kwargs: deleted_runs.append(
                sorted([str(run.info.run_id) for run in kwargs['runs']])
            ),
        )

        run_experiment_cli._run_experiment(
            'dummy.yaml',
            overwrite=True,
        )

        assert deleted_runs == [['a1', 'b1', 'bone1']]
        assert len(run_calls) == 1
        launched_config = run_calls[0][0]
        assert isinstance(launched_config.backbone, _DummyBackboneConfig)
        assert [run_config.name for run_config in launched_config.runs or []] == ['run_a', 'run_b']

    def test_retry_with_no_failed_runs_skips_orchestration(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run_calls: list[tuple[_DummyExperimentConfig, str | None, str | None]] = []
        experiment_config = _DummyExperimentConfig(
            experiment_name='exp',
            backbone=_DummyBackboneConfig(source_experiment=None),
            runs=[_DummyRunConfig(name='run_a')],
        )
        _patch_loader_and_orchestrator(
            monkeypatch=monkeypatch,
            experiment_config=experiment_config,
            run_calls=run_calls,
        )
        monkeypatch.setattr(
            mlflow_utils,
            'resolve_active_runs_by_name',
            lambda **kwargs: {
                'backbone': [_make_active_run(run_id='bone', status='FINISHED', start_time=100)],
                'run_a': [_make_active_run(run_id='a1', status='FINISHED', start_time=101)],
            },
        )
        deleted_runs: list[list[str]] = []
        monkeypatch.setattr(
            mlflow_utils,
            'delete_mlflow_runs',
            lambda **kwargs: deleted_runs.append(
                [str(run.info.run_id) for run in kwargs['runs']]
            ),
        )

        run_experiment_cli._run_experiment(
            'dummy.yaml',
            retry=True,
        )

        assert deleted_runs == [[]]
        assert run_calls == []

    def test_overwrite_with_source_backbone_does_not_select_backbone(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run_calls: list[tuple[_DummyExperimentConfig, str | None, str | None]] = []
        experiment_config = _DummyExperimentConfig(
            experiment_name='exp',
            backbone=_DummyBackboneConfig(source_experiment='source_exp'),
            runs=[_DummyRunConfig(name='run_a')],
        )
        _patch_loader_and_orchestrator(
            monkeypatch=monkeypatch,
            experiment_config=experiment_config,
            run_calls=run_calls,
        )
        monkeypatch.setattr(
            mlflow_utils,
            'resolve_active_runs_by_name',
            lambda **kwargs: {
                'backbone': [_make_active_run(run_id='bone1', status='FINISHED', start_time=100)],
                'run_a': [_make_active_run(run_id='a1', status='FINISHED', start_time=110)],
            },
        )
        deleted_runs: list[list[str]] = []
        monkeypatch.setattr(
            mlflow_utils,
            'delete_mlflow_runs',
            lambda **kwargs: deleted_runs.append(
                sorted([str(run.info.run_id) for run in kwargs['runs']])
            ),
        )

        run_experiment_cli._run_experiment(
            'dummy.yaml',
            overwrite=True,
        )

        assert deleted_runs == [['a1']]
        assert len(run_calls) == 1
        launched_config = run_calls[0][0]
        assert isinstance(launched_config.backbone, _DummyBackboneConfig)
        assert launched_config.backbone.source_experiment == 'source_exp'
        assert [run_config.name for run_config in launched_config.runs or []] == ['run_a']
