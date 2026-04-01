"""
Tests for experiment builders.
"""

import pytest
import torch

import regain.experiments.builders as builders_module
from regain.experiments.builders import build_backbone
from regain.experiments.builders import build_benchmark
from regain.experiments.builders import build_controller
from regain.experiments.builders import build_lr_scheduler_plugin
from regain.experiments.builders import build_optimizer
from regain.experiments.config import BackboneConfig
from regain.experiments.config import ControllerConfig
from regain.experiments.config import EvaluationConfig
from regain.experiments.config import ExperimentConfig
from regain.experiments.config import OptimizerConfig
from regain.experiments.config import RepairConfig
from regain.experiments.config import TransformsConfig
from regain.models.controllers import PreventionController


###############################
# Controller replay contracts #
###############################


class TestBuildControllerReplayRequirements:
    def test_raises_clear_error_when_replay_required_controller_has_no_replay_batch_size(self) -> None:
        controller_config = ControllerConfig(name='tbbn', kwargs={})

        with pytest.raises(ValueError, match='requires a replay-based strategy'):
            build_controller(
                controller_config=controller_config,
                train_batch_size=32,
                replay_batch_size=None,
                replay_memory_size=None,
            )

    def test_builds_replay_required_controller_when_replay_batch_size_is_available(self) -> None:
        controller_config = ControllerConfig(name='tbbn', kwargs={})

        controller = build_controller(
            controller_config=controller_config,
            train_batch_size=32,
            replay_batch_size=16,
            replay_memory_size=200,
        )

        assert isinstance(controller, PreventionController)


class TestBuildBenchmark:
    def test_forwards_none_transform_overrides_and_backbone_image_size(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured_kwargs: dict[str, object] = {}

        def _fake_scenario_builder(**kwargs: object) -> dict[str, object]:
            captured_kwargs.update(kwargs)
            return {'ok': True}

        monkeypatch.setattr(
            builders_module,
            'get_scenario_builder',
            lambda *, scenario: _fake_scenario_builder,
        )
        experiment_config = ExperimentConfig(
            experiment_name='unit_test_experiment',
            scenario='split_cifar100',
            num_experiences=20,
            backbone=BackboneConfig(
                name='vit_small',
                kwargs={
                    'image_size': 384,
                },
            ),
            repair=RepairConfig(split_fraction=0.2),
            transforms=TransformsConfig(),
            evaluation=EvaluationConfig(),
            runs=[],
            dataset_path='/tmp/datasets',
            seed=7,
        )

        benchmark = build_benchmark(
            experiment_config=experiment_config,
            repair_split_fraction=0.2,
        )

        assert benchmark == {'ok': True}
        assert captured_kwargs['num_experiences'] == 20
        assert captured_kwargs['return_task_id'] is False
        assert captured_kwargs['repair_split_fraction'] == pytest.approx(0.2)
        assert captured_kwargs['dataset_path'] == '/tmp/datasets'
        assert captured_kwargs['seed'] == 7
        assert captured_kwargs['transform_random_resized_crop'] is None
        assert captured_kwargs['transform_horizontal_flip'] is None
        assert captured_kwargs['transform_image_size'] == 384

    def test_forwards_explicit_transform_overrides_without_bool_coercion(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured_kwargs: dict[str, object] = {}

        def _fake_scenario_builder(**kwargs: object) -> dict[str, object]:
            captured_kwargs.update(kwargs)
            return {'ok': True}

        monkeypatch.setattr(
            builders_module,
            'get_scenario_builder',
            lambda *, scenario: _fake_scenario_builder,
        )
        experiment_config = ExperimentConfig(
            experiment_name='unit_test_experiment',
            scenario='split_tinyimagenet',
            num_experiences=10,
            backbone=BackboneConfig(name='resnet18', kwargs={}),
            repair=RepairConfig(split_fraction=0.1),
            transforms=TransformsConfig(
                random_resized_crop=True,
                horizontal_flip=False,
            ),
            evaluation=EvaluationConfig(),
            runs=[],
        )

        benchmark = build_benchmark(
            experiment_config=experiment_config,
            repair_split_fraction=0.1,
        )

        assert benchmark == {'ok': True}
        assert captured_kwargs['transform_random_resized_crop'] is True
        assert captured_kwargs['transform_horizontal_flip'] is False
        assert captured_kwargs['transform_image_size'] is None


class TestBuildBackbone:
    def test_builds_vit_backbone_with_constructor_kwargs(self) -> None:
        backbone = build_backbone(
            name='vit_small',
            num_classes=11,
            backbone_kwargs={
                'image_size': 32,
                'patch_size': 4,
                'dropout': 0.1,
            },
        )
        x = torch.randn(2, 3, 32, 32)

        logits = backbone(x)

        assert isinstance(backbone, torch.nn.Module)
        assert logits.shape == (2, 11)


class TestBuildOptimizer:
    def test_builds_adamw_optimizer_with_beta_sequence(self) -> None:
        model = torch.nn.Linear(4, 2)

        optimizer, optimizer_kwargs = build_optimizer(
            model=model,
            optimizer_config=OptimizerConfig(
                name='adamw',
                kwargs={
                    'lr': 5e-4,
                    'betas': [0.9, 0.999],
                    'eps': 1e-8,
                    'weight_decay': 1e-4,
                },
            ),
        )

        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.defaults['lr'] == pytest.approx(5e-4)
        assert optimizer.defaults['betas'] == pytest.approx((0.9, 0.999))
        assert optimizer_kwargs['betas'] == [0.9, 0.999]

    def test_rejects_string_beta_literal_for_adamw(self) -> None:
        model = torch.nn.Linear(4, 2)

        with pytest.raises(ValueError, match='YAML sequence'):
            build_optimizer(
                model=model,
                optimizer_config=OptimizerConfig(
                    name='adamw',
                    kwargs={
                        'betas': '(0.9, 0.999)',
                    },
                ),
            )


class TestBuildLRSchedulerPlugin:
    def test_rejects_missing_total_epochs_for_warmup_cosine(self) -> None:
        with pytest.raises(ValueError, match='total_epochs'):
            build_lr_scheduler_plugin(
                name='warmup_cosine',
                scheduler_kwargs={
                    'warmup_epochs': 5,
                },
            )
