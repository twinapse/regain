"""
Tests for experiment builders.
"""

import pytest
import torch

from regain.experiments.builders import build_backbone
from regain.experiments.builders import build_controller
from regain.experiments.config import ControllerConfig
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
