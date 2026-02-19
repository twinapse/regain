"""
Tests for repair controller plugin.
"""

import types
from typing import Any

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

# Ensure a stable import order for plugin module initialization.
import regain.experiments.orchestrator  # noqa: F401
from regain.avalanche_utils.plugins import RepairControllerPlugin
from regain.models.controllers import RepairController


################
# Test helpers #
################

class _IdentityModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _DummyStrategy:
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


class _ScriptedRepairController(RepairController):
    def __init__(self, *, corrected_outputs: torch.Tensor | None = None) -> None:
        super().__init__()
        self.corrected_outputs = corrected_outputs
        self.eval_begin_args: tuple[Any, ...] | None = None
        self.eval_begin_kwargs: dict[str, Any] | None = None
        self.eval_end_args: tuple[Any, ...] | None = None
        self.eval_end_kwargs: dict[str, Any] | None = None
        self.eval_exp_begin_args: tuple[Any, ...] | None = None
        self.eval_exp_begin_kwargs: dict[str, Any] | None = None
        self.eval_exp_end_args: tuple[Any, ...] | None = None
        self.eval_exp_end_kwargs: dict[str, Any] | None = None

    def fit_on_repair_data(
        self,
        *,
        model: nn.Module,
        repair_dataset: Dataset | None,
        new_classes: list[int],
        num_epochs: int,
        batch_size: int,
    ) -> None:
        del model, repair_dataset, new_classes, num_epochs, batch_size
        return

    def correct_outputs(
        self,
        *,
        outputs: Any,
        model: nn.Module | None = None,
        inputs: Any | None = None,
    ) -> Any:
        del model, inputs
        if self.corrected_outputs is None:
            return outputs
        return self.corrected_outputs.clone()

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


def _make_plugin(
    *,
    corrected_outputs: torch.Tensor | None,
    seen_classes: list[int],
) -> RepairControllerPlugin:
    controller = _ScriptedRepairController(corrected_outputs=corrected_outputs)
    plugin = RepairControllerPlugin(
        controller=controller,
        fit_after_experience=False,
        repair_epochs=1,
        repair_batch_size=1,
    )
    experience = types.SimpleNamespace(
        classes_in_this_experience=seen_classes,
        dataset=None,
    )
    strategy = _DummyStrategy(
        model=_IdentityModel(),
        mb_output=torch.zeros((1, max(seen_classes) + 1), dtype=torch.float32),
        mb_x=torch.zeros((1, 1), dtype=torch.float32),
        experience=experience,
    )
    plugin.before_training_exp(strategy)
    return plugin


###########################
# Output contract coverage #
###########################

class TestRepairControllerPluginOutputContract:
    def test_allows_seen_class_modification_only(self) -> None:
        base_logits = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]],
            dtype=torch.float32,
        )
        corrected_logits = torch.tensor(
            [[10.0, 20.0], [30.0, 40.0]],
            dtype=torch.float32,
        )
        plugin = _make_plugin(
            corrected_outputs=corrected_logits,
            seen_classes=[0, 1],
        )

        strategy = _DummyStrategy(
            model=_IdentityModel(),
            mb_output=base_logits.clone(),
            mb_x=torch.zeros((2, 1), dtype=torch.float32),
        )
        plugin.after_eval_forward(strategy)

        expected = base_logits.clone()
        expected[:, :2] = corrected_logits
        assert torch.equal(strategy.mb_output, expected)

    def test_raises_when_controller_width_exceeds_backbone_width(self) -> None:
        base_logits = torch.zeros((2, 4), dtype=torch.float32)
        corrected_logits = torch.zeros((2, 5), dtype=torch.float32)
        plugin = _make_plugin(
            corrected_outputs=corrected_logits,
            seen_classes=[0, 1],
        )

        strategy = _DummyStrategy(
            model=_IdentityModel(),
            mb_output=base_logits,
            mb_x=torch.zeros((2, 1), dtype=torch.float32),
        )
        with pytest.raises(RuntimeError, match='exceeds backbone output width'):
            plugin.after_eval_forward(strategy)


#######################
# Anti-cheat coverage #
#######################

class TestRepairControllerPluginAntiCheat:
    def test_unseen_logits_not_modified_by_controller(self) -> None:
        base_logits = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]],
            dtype=torch.float32,
        )
        corrected_logits = base_logits.clone()
        corrected_logits[:, 2] = corrected_logits[:, 2] + 1.0
        plugin = _make_plugin(
            corrected_outputs=corrected_logits,
            seen_classes=[0, 1],
        )

        strategy = _DummyStrategy(
            model=_IdentityModel(),
            mb_output=base_logits,
            mb_x=torch.zeros((2, 1), dtype=torch.float32),
        )
        plugin.after_eval_forward(strategy)

        expected = base_logits.clone()
        expected[:, [0, 1]] = corrected_logits[:, [0, 1]]
        assert torch.equal(strategy.mb_output, expected)

    def test_raises_when_seen_class_not_representable(self) -> None:
        base_logits = torch.zeros((2, 4), dtype=torch.float32)
        corrected_logits = torch.zeros((2, 3), dtype=torch.float32)
        plugin = _make_plugin(
            corrected_outputs=corrected_logits,
            seen_classes=[0, 3],
        )

        strategy = _DummyStrategy(
            model=_IdentityModel(),
            mb_output=base_logits,
            mb_x=torch.zeros((2, 1), dtype=torch.float32),
        )
        with pytest.raises(RuntimeError, match='does not cover all seen classes'):
            plugin.after_eval_forward(strategy)


########################
# Eval hook boundaries #
########################

class TestRepairControllerPluginEvalHooks:
    def test_eval_hooks_do_not_receive_strategy_or_experience(self) -> None:
        controller = _ScriptedRepairController(corrected_outputs=None)
        plugin = RepairControllerPlugin(
            controller=controller,
            fit_after_experience=False,
            repair_epochs=1,
            repair_batch_size=1,
        )
        strategy = _DummyStrategy(
            model=_IdentityModel(),
            mb_output=torch.zeros((1, 2), dtype=torch.float32),
            mb_x=torch.zeros((1, 1), dtype=torch.float32),
            experience=types.SimpleNamespace(current_experience=0),
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
