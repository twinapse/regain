"""
Tests for experiment configuration parsing.
"""

from pathlib import Path

import pytest
import yaml

from regain.experiments.config import EvaluationConfig
from regain.experiments.config import TransformsConfig
from regain.experiments.config import load_experiment_config


################
# Test helpers #
################


def _build_base_payload() -> dict[str, object]:
    return {
        'experiment_name': 'unit_test_experiment',
        'scenario': 'cifar100',
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
            'split_fraction': 0.0,
            'budget_fraction': 1.0,
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


class TestTransformsConfigParsing:
    def test_parses_nested_transforms_config(self, tmp_path: Path) -> None:
        payload = _build_base_payload()
        payload['evaluation'] = {}
        payload['transforms'] = {
            'random_resized_crop': True,
            'horizontal_flip': False,
        }
        config_path = _write_payload(tmp_path=tmp_path, payload=payload)

        config = load_experiment_config(config_path)

        assert config.transforms == TransformsConfig(
            random_resized_crop=True,
            horizontal_flip=False,
        )

    def test_uses_transforms_defaults_when_section_is_missing(self, tmp_path: Path) -> None:
        payload = _build_base_payload()
        payload['evaluation'] = {}
        config_path = _write_payload(tmp_path=tmp_path, payload=payload)

        config = load_experiment_config(config_path)

        assert config.transforms == TransformsConfig()

    def test_rejects_non_mapping_transforms_section(self, tmp_path: Path) -> None:
        payload = _build_base_payload()
        payload['evaluation'] = {}
        payload['transforms'] = ['invalid']
        config_path = _write_payload(tmp_path=tmp_path, payload=payload)

        with pytest.raises(ValueError, match='`transforms` must be a mapping when provided'):
            load_experiment_config(config_path)

    def test_rejects_non_boolean_random_resized_crop(self, tmp_path: Path) -> None:
        payload = _build_base_payload()
        payload['evaluation'] = {}
        payload['transforms'] = {
            'random_resized_crop': 'true',
        }
        config_path = _write_payload(tmp_path=tmp_path, payload=payload)

        with pytest.raises(ValueError, match='`transforms.random_resized_crop` must be a boolean'):
            load_experiment_config(config_path)

    def test_rejects_non_boolean_horizontal_flip(self, tmp_path: Path) -> None:
        payload = _build_base_payload()
        payload['evaluation'] = {}
        payload['transforms'] = {
            'horizontal_flip': 'false',
        }
        config_path = _write_payload(tmp_path=tmp_path, payload=payload)

        with pytest.raises(ValueError, match='`transforms.horizontal_flip` must be a boolean'):
            load_experiment_config(config_path)

    def test_rejects_run_level_transforms_override(self, tmp_path: Path) -> None:
        payload = _build_base_payload()
        payload['evaluation'] = {}
        payload['transforms'] = {
            'random_resized_crop': None,
            'horizontal_flip': None,
        }
        payload['runs'] = [
            {
                'name': 'invalid_run',
                'transforms': {
                    'horizontal_flip': False,
                },
            },
        ]
        config_path = _write_payload(tmp_path=tmp_path, payload=payload)

        with pytest.raises(ValueError, match='keys should not override `transforms`'):
            load_experiment_config(config_path)

class TestRepairConfigParsing:
    def test_parses_repair_split_fraction_field(self, tmp_path: Path) -> None:
        payload = _build_base_payload()
        payload['evaluation'] = {}
        payload['repair'] = {
            'split_fraction': 0.25,
            'budget_fraction': 0.5,
            'fit_schedule': 'per_experience',
        }
        config_path = _write_payload(tmp_path=tmp_path, payload=payload)

        config = load_experiment_config(config_path)

        assert config.repair.split_fraction == pytest.approx(0.25)
        assert config.repair.budget_fraction == pytest.approx(0.5)
        assert config.repair.fit_schedule == 'per_experience'

    def test_requires_repair_split_fraction_field(self, tmp_path: Path) -> None:
        payload = _build_base_payload()
        payload['evaluation'] = {}
        payload['repair'] = {
            'budget_fraction': 0.5,
            'fit_schedule': 'per_experience',
        }
        config_path = _write_payload(tmp_path=tmp_path, payload=payload)

        with pytest.raises(ValueError, match='repair.split_fraction'):
            load_experiment_config(config_path)

    def test_defaults_non_split_repair_fields_to_none_without_repair_runs(
        self,
        tmp_path: Path,
    ) -> None:
        payload = _build_base_payload()
        payload['evaluation'] = {}
        payload['repair'] = {
            'split_fraction': 0.2,
        }
        config_path = _write_payload(tmp_path=tmp_path, payload=payload)

        config = load_experiment_config(config_path)

        assert config.repair.budget_fraction is None
        assert config.repair.fit_schedule is None
        assert config.repair.num_epochs is None
        assert config.repair.batch_size is None

    def test_requires_all_non_split_repair_fields_for_repair_runs(self, tmp_path: Path) -> None:
        payload = _build_base_payload()
        payload['evaluation'] = {}
        payload['repair'] = {
            'split_fraction': 0.2,
        }
        payload['runs'] = [
            {
                'name': 'repair_run',
                'controller': {
                    'name': 'logit_bias',
                    'kwargs': {},
                },
            },
        ]
        config_path = _write_payload(tmp_path=tmp_path, payload=payload)

        with pytest.raises(ValueError) as exc_info:
            load_experiment_config(config_path)

        message = str(exc_info.value)
        assert 'repair.budget_fraction' in message
        assert 'repair.fit_schedule' in message
        assert 'repair.num_epochs' in message
        assert 'repair.batch_size' in message

    def test_rejects_repair_split_fraction_out_of_range(self, tmp_path: Path) -> None:
        payload = _build_base_payload()
        payload['evaluation'] = {}
        payload['repair'] = {
            'split_fraction': 1.5,
            'budget_fraction': 0.5,
            'fit_schedule': 'per_experience',
        }
        config_path = _write_payload(tmp_path=tmp_path, payload=payload)

        with pytest.raises(ValueError, match='repair.split_fraction'):
            load_experiment_config(config_path)


class TestBackboneConfigParsing:
    def test_accepts_registered_vit_backbone_name(self, tmp_path: Path) -> None:
        payload = _build_base_payload()
        payload['evaluation'] = {}
        payload['backbone']['name'] = 'vit_small'
        config_path = _write_payload(tmp_path=tmp_path, payload=payload)

        config = load_experiment_config(config_path)

        assert config.backbone is not None
        assert config.backbone.name == 'vit_small'

    def test_parses_backbone_kwargs_mapping(self, tmp_path: Path) -> None:
        payload = _build_base_payload()
        payload['evaluation'] = {}
        payload['backbone']['name'] = 'vit_small'
        payload['backbone']['kwargs'] = {
            'image_size': 32,
            'patch_size': 4,
            'dropout': 0.1,
        }
        config_path = _write_payload(tmp_path=tmp_path, payload=payload)

        config = load_experiment_config(config_path)

        assert config.backbone is not None
        assert config.backbone.kwargs == {
            'image_size': 32,
            'patch_size': 4,
            'dropout': 0.1,
        }

    def test_rejects_non_mapping_backbone_kwargs(self, tmp_path: Path) -> None:
        payload = _build_base_payload()
        payload['evaluation'] = {}
        payload['backbone']['kwargs'] = ['invalid']
        config_path = _write_payload(tmp_path=tmp_path, payload=payload)

        with pytest.raises(ValueError, match='Backbone config `kwargs` must be a mapping'):
            load_experiment_config(config_path)

    def test_rejects_backbone_kwargs_with_source_experiment(self, tmp_path: Path) -> None:
        payload = _build_base_payload()
        payload['evaluation'] = {}
        payload['backbone'] = {
            'source_experiment': 'other_experiment',
            'kwargs': {
                'patch_size': 4,
            },
        }
        config_path = _write_payload(tmp_path=tmp_path, payload=payload)

        with pytest.raises(
            ValueError,
            match='must be the only field under `backbone`',
        ):
            load_experiment_config(config_path)

    def test_parses_adamw_optimizer_and_warmup_cosine_scheduler(self, tmp_path: Path) -> None:
        payload = _build_base_payload()
        payload['evaluation'] = {}
        payload['backbone']['training']['optimizer'] = {
            'name': 'adamw',
            'kwargs': {
                'lr': 5e-4,
                'betas': [0.9, 0.999],
                'eps': 1e-8,
                'weight_decay': 1e-4,
            },
        }
        payload['backbone']['training']['lr_scheduler'] = {
            'name': 'warmup_cosine',
            'kwargs': {
                'warmup_epochs': 0,
                'min_lr': 0.0,
            },
        }
        payload['backbone']['training']['grad_clip_max_norm'] = 1.0
        config_path = _write_payload(tmp_path=tmp_path, payload=payload)

        config = load_experiment_config(config_path)

        assert config.backbone is not None
        assert config.backbone.training is not None
        assert config.backbone.training.optimizer.name == 'adamw'
        assert config.backbone.training.optimizer.kwargs['betas'] == [0.9, 0.999]
        assert config.backbone.training.lr_scheduler is not None
        assert config.backbone.training.lr_scheduler.name == 'warmup_cosine'
        assert config.backbone.training.grad_clip_max_norm == pytest.approx(1.0)

    def test_rejects_string_literal_betas_for_adamw(self, tmp_path: Path) -> None:
        payload = _build_base_payload()
        payload['evaluation'] = {}
        payload['backbone']['training']['optimizer'] = {
            'name': 'adamw',
            'kwargs': {
                'betas': '(0.9, 0.999)',
            },
        }
        config_path = _write_payload(tmp_path=tmp_path, payload=payload)

        with pytest.raises(ValueError, match='YAML sequence'):
            load_experiment_config(config_path)

    def test_rejects_non_positive_grad_clip_max_norm(self, tmp_path: Path) -> None:
        payload = _build_base_payload()
        payload['evaluation'] = {}
        payload['backbone']['training']['grad_clip_max_norm'] = -1.0
        config_path = _write_payload(tmp_path=tmp_path, payload=payload)

        with pytest.raises(ValueError, match='grad_clip_max_norm'):
            load_experiment_config(config_path)

    def test_rejects_negative_warmup_epochs(self, tmp_path: Path) -> None:
        payload = _build_base_payload()
        payload['evaluation'] = {}
        payload['backbone']['training']['lr_scheduler'] = {
            'name': 'warmup_cosine',
            'kwargs': {
                'warmup_epochs': -1,
            },
        }
        config_path = _write_payload(tmp_path=tmp_path, payload=payload)

        with pytest.raises(ValueError, match='warmup_epochs'):
            load_experiment_config(config_path)

    def test_rejects_warmup_epochs_not_less_than_num_epochs(self, tmp_path: Path) -> None:
        payload = _build_base_payload()
        payload['evaluation'] = {}
        payload['backbone']['training']['num_epochs'] = 1
        payload['backbone']['training']['lr_scheduler'] = {
            'name': 'warmup_cosine',
            'kwargs': {
                'warmup_epochs': 1,
            },
        }
        config_path = _write_payload(tmp_path=tmp_path, payload=payload)

        with pytest.raises(ValueError, match='warmup_epochs'):
            load_experiment_config(config_path)
