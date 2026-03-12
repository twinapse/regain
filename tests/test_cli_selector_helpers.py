"""
Tests for shared CLI selector helpers.
"""

import argparse
from types import SimpleNamespace

import pytest

from regain.cli._utils.output_helpers import CliFailure
import regain.cli._utils.selector_helpers as selector_helpers


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    selector_helpers.add_experiment_selector_arguments(parser=parser)
    return parser


def test_resolve_experiment_targets_from_experiments_dedupes_order() -> None:
    parser = _build_parser()
    failures: list[CliFailure] = []

    targets = selector_helpers.resolve_experiment_targets(
        parser=parser,
        config_files=None,
        config_dir=None,
        experiments='exp_b,exp_a,exp_b',
        tracking_uri=None,
        failures=failures,
    )

    assert [target.experiment_name for target in targets] == ['exp_b', 'exp_a']
    assert all(target.tracking_uri is None for target in targets)
    assert failures == []


def test_resolve_experiment_targets_from_configs_dedupes_experiment_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = _build_parser()
    failures: list[CliFailure] = []
    config_map = {
        'a.yaml': SimpleNamespace(experiment_name='exp_shared'),
        'b.yaml': SimpleNamespace(experiment_name='exp_shared'),
    }

    monkeypatch.setattr(
        selector_helpers,
        'load_experiment_config',
        lambda config_path: config_map[config_path],
    )

    targets = selector_helpers.resolve_experiment_targets(
        parser=parser,
        config_files='a.yaml,b.yaml',
        config_dir=None,
        experiments=None,
        tracking_uri=None,
        failures=failures,
    )

    assert len(targets) == 1
    assert targets[0].experiment_name == 'exp_shared'
    assert targets[0].tracking_uri is None
    assert failures == []


def test_resolve_experiment_targets_tracking_uri_applies_to_configs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = _build_parser()
    failures: list[CliFailure] = []
    config_map = {
        'a.yaml': SimpleNamespace(experiment_name='exp_shared'),
        'b.yaml': SimpleNamespace(experiment_name='exp_shared'),
    }

    monkeypatch.setattr(
        selector_helpers,
        'load_experiment_config',
        lambda config_path: config_map[config_path],
    )

    targets = selector_helpers.resolve_experiment_targets(
        parser=parser,
        config_files='a.yaml,b.yaml',
        config_dir=None,
        experiments=None,
        tracking_uri='mlflow://override',
        failures=failures,
    )

    assert len(targets) == 1
    assert targets[0].experiment_name == 'exp_shared'
    assert targets[0].tracking_uri == 'mlflow://override'
    assert failures == []


def test_resolve_experiment_targets_invalid_config_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = _build_parser()
    failures: list[CliFailure] = []

    def _fake_load_config(config_path: str) -> SimpleNamespace:
        if config_path == 'bad.yaml':
            raise ValueError('bad config')
        return SimpleNamespace(experiment_name='exp_ok')

    monkeypatch.setattr(selector_helpers, 'load_experiment_config', _fake_load_config)

    targets = selector_helpers.resolve_experiment_targets(
        parser=parser,
        config_files='bad.yaml,ok.yaml',
        config_dir=None,
        experiments=None,
        tracking_uri=None,
        failures=failures,
    )

    assert [target.experiment_name for target in targets] == ['exp_ok']
    assert any('bad config' in failure.message for failure in failures)
