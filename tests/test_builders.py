"""
Tests for experiment builders.
"""

import pytest

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
