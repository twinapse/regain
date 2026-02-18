"""
Tests for experiment configuration parsing.
"""

from pathlib import Path

import pytest
import yaml

from regain.experiments.config import EvaluationConfig
from regain.experiments.config import load_experiment_config


################
# Test helpers #
################


def _build_base_payload() -> dict[str, object]:
    return {
        'experiment_name': 'unit_test_experiment',
        'scenario': 'split_cifar100',
        'num_experiences': 2,
        'backbone': {
            'name': 'resnet18',
            'training': {
                'num_epochs': 1,
                'strategy': {
                    'name': 'naive',
                    'kwargs': {},
                },
            },
        },
        'repair': {
            'budget_per_class': 0,
            'fit_schedule': 'per_experience',
        },
        'runs': [],
    }


def _write_payload(*, tmp_path: Path, payload: dict[str, object]) -> Path:
    config_path = tmp_path / 'config.yaml'
    with config_path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return config_path


##################
# Parsing checks #
##################


class TestEvaluationConfigParsing:
    def test_parses_nested_evaluation_config(self, tmp_path: Path) -> None:
        payload = _build_base_payload()
        payload['evaluation'] = {
            'batch_size': 64,
            'avalanche_schedule': 'final_only',
        }
        config_path = _write_payload(tmp_path=tmp_path, payload=payload)

        config = load_experiment_config(config_path)

        assert config.evaluation == EvaluationConfig(
            batch_size=64,
            avalanche_schedule='final_only',
        )

    def test_uses_evaluation_defaults_when_fields_are_missing(self, tmp_path: Path) -> None:
        payload = _build_base_payload()
        payload['evaluation'] = {}
        config_path = _write_payload(tmp_path=tmp_path, payload=payload)

        config = load_experiment_config(config_path)

        assert config.evaluation == EvaluationConfig()

    def test_raises_when_evaluation_section_is_missing(self, tmp_path: Path) -> None:
        payload = _build_base_payload()
        config_path = _write_payload(tmp_path=tmp_path, payload=payload)

        with pytest.raises(ValueError, match='must include an `evaluation` mapping'):
            load_experiment_config(config_path)

    def test_raises_on_invalid_evaluation_schedule(self, tmp_path: Path) -> None:
        payload = _build_base_payload()
        payload['evaluation'] = {
            'batch_size': 64,
            'avalanche_schedule': 'epochly',
        }
        config_path = _write_payload(tmp_path=tmp_path, payload=payload)

        with pytest.raises(ValueError, match='`evaluation.avalanche_schedule`'):
            load_experiment_config(config_path)

    def test_rejects_run_level_evaluation_override(self, tmp_path: Path) -> None:
        payload = _build_base_payload()
        payload['evaluation'] = {
            'batch_size': 128,
            'avalanche_schedule': 'per_experience',
        }
        payload['runs'] = [
            {
                'name': 'invalid_run',
                'evaluation': {
                    'batch_size': 64,
                },
            },
        ]
        config_path = _write_payload(tmp_path=tmp_path, payload=payload)

        with pytest.raises(ValueError, match='keys should not override `evaluation`'):
            load_experiment_config(config_path)
