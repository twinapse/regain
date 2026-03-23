"""
Tests for experiment builders.
"""

import pytest
import torch

from regain.experiments.builders import build_backbone
from regain.experiments.builders import build_controller
from regain.experiments.builders import build_lr_scheduler_plugin
from regain.experiments.builders import build_optimizer
from regain.experiments.config import ControllerConfig
from regain.experiments.config import OptimizerConfig
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
    def test_builds_adam_optimizer_with_beta_sequence(self) -> None:
        model = torch.nn.Linear(4, 2)

        optimizer, optimizer_kwargs = build_optimizer(
            model=model,
            optimizer_config=OptimizerConfig(
                name='adam',
                kwargs={
                    'lr': 5e-4,
                    'betas': [0.9, 0.999],
                    'eps': 1e-8,
                    'weight_decay': 1e-4,
                },
            ),
        )

        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.defaults['lr'] == pytest.approx(5e-4)
        assert optimizer.defaults['betas'] == pytest.approx((0.9, 0.999))
        assert optimizer_kwargs['betas'] == [0.9, 0.999]

    def test_rejects_string_beta_literal_for_adam(self) -> None:
        model = torch.nn.Linear(4, 2)

        with pytest.raises(ValueError, match='YAML sequence'):
            build_optimizer(
                model=model,
                optimizer_config=OptimizerConfig(
                    name='adam',
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
