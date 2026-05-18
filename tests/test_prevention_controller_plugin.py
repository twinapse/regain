"""
Tests for prevention controller plugin.
"""

from typing import Any

import torch
from torch import nn

from regain.avalanche_utils.plugins import PreventionControllerPlugin
# Ensure a stable import order for plugin module initialization.
import regain.experiments.orchestrator  # noqa: F401  # pylint: disable=unused-import
from regain.models.controllers import PreventionController

################
# Test helpers #
################


class _IdentityModel(nn.Module):
    """
    Identity model for testing.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _DummyStrategy:
    """
    Dummy strategy for testing.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        mb_output: torch.Tensor,
        mb_x: torch.Tensor,
        experience: object | None = None,
    ) -> None:
        self.model = model
        self.mb_output = mb_output
        self.mb_x = mb_x
        self.experience = experience


class _SpyPreventionController(PreventionController):
    """
    Spy wrapper for PreventionController.
    """

    def __init__(self) -> None:
        super().__init__()
        self.eval_begin_args: tuple[Any, ...] | None = None
        self.eval_begin_kwargs: dict[str, Any] | None = None
        self.eval_end_args: tuple[Any, ...] | None = None
        self.eval_end_kwargs: dict[str, Any] | None = None
        self.eval_exp_begin_args: tuple[Any, ...] | None = None
        self.eval_exp_begin_kwargs: dict[str, Any] | None = None
        self.eval_exp_end_args: tuple[Any, ...] | None = None
        self.eval_exp_end_kwargs: dict[str, Any] | None = None

    def on_eval_begin(self, *args, **kwargs) -> None:
        self.eval_begin_args = args
        self.eval_begin_kwargs = dict(kwargs)

    def on_eval_end(self, *args, **kwargs) -> None:
        self.eval_end_args = args
        self.eval_end_kwargs = dict(kwargs)

    def on_eval_experience_begin(self, *args, **kwargs) -> None:
        self.eval_exp_begin_args = args
        self.eval_exp_begin_kwargs = dict(kwargs)

    def on_eval_experience_end(self, *args, **kwargs) -> None:
        self.eval_exp_end_args = args
        self.eval_exp_end_kwargs = dict(kwargs)


########################
# Eval hook boundaries #
########################


class TestPreventionControllerPluginEvalHooks:
    """
    Tests for PreventionControllerPlugin eval hooks.
    """

    def test_prevention_eval_hooks_do_not_receive_strategy_or_experience(self) -> None:
        controller = _SpyPreventionController()
        plugin = PreventionControllerPlugin(controller=controller)
        strategy = _DummyStrategy(
            model=_IdentityModel(),
            mb_output=torch.zeros((1, 2), dtype=torch.float32),
            mb_x=torch.zeros((1, 1), dtype=torch.float32),
        )

        plugin.before_eval(strategy)
        plugin.before_eval_exp(strategy)
        plugin.after_eval_exp(strategy)
        plugin.after_eval(strategy)

        assert controller.eval_begin_args == tuple()
        assert controller.eval_begin_kwargs == {}
        assert controller.eval_exp_begin_args == tuple()
        assert controller.eval_exp_begin_kwargs == {}
        assert controller.eval_exp_end_args == tuple()
        assert controller.eval_exp_end_kwargs == {}
        assert controller.eval_end_args == tuple()
        assert controller.eval_end_kwargs == {}
