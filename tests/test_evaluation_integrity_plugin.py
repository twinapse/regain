"""
Tests for evaluation integrity plugin.
"""

import types
from typing import Any

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

# Ensure a stable import order for plugin module initialization.
import regain.experiments.orchestrator  # noqa: F401
from regain.avalanche_utils.plugins import EvaluationIntegrityPlugin
from regain.avalanche_utils.plugins import PreventionControllerPlugin
from regain.avalanche_utils.plugins import RepairControllerPlugin
from regain.models.controllers import PreventionController
from regain.models.controllers import RepairController


################
# Test helpers #
################


class _ToyModel(nn.Module):
    def __init__(self, *, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features=3, out_features=out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _DummyStrategy:
    def __init__(
        self,
        *,
        model: nn.Module,
        mb_output: Any,
        mb_y: Any,
        mb_x: torch.Tensor,
        experience: object | None = None,
        plugins: list[object] | None = None,
    ) -> None:
        self.model = model
        self.mb_output = mb_output
        self.mb_y = mb_y
        self.mb_x = mb_x
        self.experience = experience
        self.plugins = plugins if plugins is not None else []


class _IdentityRepairController(RepairController):
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
        return outputs


class _SpyPreventionController(PreventionController):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))


def _make_valid_strategy(*, model: nn.Module | None = None) -> _DummyStrategy:
    resolved_model = model if model is not None else _ToyModel(out_features=4)
    return _DummyStrategy(
        model=resolved_model,
        mb_output=torch.randn((2, 4), dtype=torch.float32),
        mb_y=torch.tensor([0, 1], dtype=torch.long),
        mb_x=torch.randn((2, 3), dtype=torch.float32),
    )


####################################
# Output contract integrity checks #
####################################


class TestEvaluationIntegrityPluginOutputContract:
    def test_raises_when_output_is_not_2d_tensor(self) -> None:
        strategy = _DummyStrategy(
            model=_ToyModel(out_features=2),
            mb_output=torch.tensor([1.0, 2.0], dtype=torch.float32),
            mb_y=torch.tensor([0], dtype=torch.long),
            mb_x=torch.randn((1, 3), dtype=torch.float32),
        )
        plugin = EvaluationIntegrityPlugin()

        with pytest.raises(RuntimeError, match='must be 2D logits'):
            plugin.after_eval_forward(strategy)

    def test_raises_when_output_batch_mismatches_targets(self) -> None:
        strategy = _DummyStrategy(
            model=_ToyModel(out_features=3),
            mb_output=torch.randn((2, 3), dtype=torch.float32),
            mb_y=torch.tensor([0], dtype=torch.long),
            mb_x=torch.randn((2, 3), dtype=torch.float32),
        )
        plugin = EvaluationIntegrityPlugin()

        with pytest.raises(RuntimeError, match='target batch size must match output batch size'):
            plugin.after_eval_forward(strategy)

    def test_raises_when_output_contains_non_finite_values(self) -> None:
        strategy = _DummyStrategy(
            model=_ToyModel(out_features=3),
            mb_output=torch.tensor([[0.1, float('nan'), 0.3]], dtype=torch.float32),
            mb_y=torch.tensor([1], dtype=torch.long),
            mb_x=torch.randn((1, 3), dtype=torch.float32),
        )
        plugin = EvaluationIntegrityPlugin()

        with pytest.raises(RuntimeError, match='contains non-finite values'):
            plugin.after_eval_forward(strategy)

    def test_raises_when_target_out_of_range(self) -> None:
        strategy = _DummyStrategy(
            model=_ToyModel(out_features=3),
            mb_output=torch.randn((2, 3), dtype=torch.float32),
            mb_y=torch.tensor([0, 5], dtype=torch.long),
            mb_x=torch.randn((2, 3), dtype=torch.float32),
        )
        plugin = EvaluationIntegrityPlugin()

        with pytest.raises(RuntimeError, match='out of range'):
            plugin.after_eval_forward(strategy)


#########################################
# Module immutability integrity checks #
#########################################


class TestEvaluationIntegrityPluginImmutability:
    def test_raises_when_backbone_state_mutates_during_eval_iteration(self) -> None:
        strategy = _make_valid_strategy()
        plugin = EvaluationIntegrityPlugin()
        plugin.before_eval(strategy)
        plugin.before_eval_exp(strategy)
        plugin.after_eval_forward(strategy)

        with torch.no_grad():
            strategy.model.linear.weight.add_(1.0)

        with pytest.raises(RuntimeError, match='signature changed during evaluation'):
            plugin.after_eval_iteration(strategy)

    def test_raises_when_controller_state_mutates_during_eval_iteration(self) -> None:
        controller = _SpyPreventionController()
        controller_plugin = PreventionControllerPlugin(controller=controller)
        strategy = _make_valid_strategy()
        strategy.plugins = [controller_plugin]

        plugin = EvaluationIntegrityPlugin(controller_plugin=controller_plugin)
        plugin.before_eval(strategy)
        plugin.before_eval_exp(strategy)
        plugin.after_eval_forward(strategy)

        with torch.no_grad():
            controller.scale.add_(1.0)

        with pytest.raises(RuntimeError, match='signature changed during evaluation'):
            plugin.after_eval_iteration(strategy)

    def test_passes_when_exact_snapshot_contains_unchanged_nan_values(self) -> None:
        strategy = _make_valid_strategy()
        with torch.no_grad():
            strategy.model.linear.weight.fill_(float('nan'))

        plugin = EvaluationIntegrityPlugin()
        plugin.before_eval(strategy)
        plugin.before_eval_exp(strategy)
        plugin.after_eval_forward(strategy)
        plugin.after_eval_iteration(strategy)
        plugin.after_eval_exp(strategy)
        plugin.after_eval(strategy)

    def test_passes_when_state_is_immutable(self) -> None:
        controller = _SpyPreventionController()
        controller_plugin = PreventionControllerPlugin(controller=controller)
        strategy = _make_valid_strategy()
        strategy.plugins = [controller_plugin]

        plugin = EvaluationIntegrityPlugin(controller_plugin=controller_plugin)
        plugin.before_eval(strategy)
        plugin.before_eval_exp(strategy)
        plugin.after_eval_forward(strategy)
        plugin.after_eval_iteration(strategy)
        plugin.after_eval_exp(strategy)
        plugin.after_eval(strategy)


###############################
# Anti-cheat ownership checks #
###############################


class TestEvaluationIntegrityPluginOwnership:
    def test_repair_plugin_does_not_duplicate_global_label_validation(self) -> None:
        repair_controller = _IdentityRepairController()
        repair_plugin = RepairControllerPlugin(
            controller=repair_controller,
            fit_after_experience=False,
            repair_epochs=1,
            repair_batch_size=1,
            budget_fraction=1.0,
            seed=1,
        )
        train_strategy = _DummyStrategy(
            model=_ToyModel(out_features=4),
            mb_output=torch.zeros((1, 4), dtype=torch.float32),
            mb_y=torch.tensor([0], dtype=torch.long),
            mb_x=torch.zeros((1, 3), dtype=torch.float32),
            experience=types.SimpleNamespace(
                classes_in_this_experience=[0, 1],
                dataset=None,
            ),
        )
        repair_plugin.before_training_exp(train_strategy)

        eval_strategy = _DummyStrategy(
            model=train_strategy.model,
            mb_output=torch.zeros((2, 4), dtype=torch.float32),
            mb_y=torch.tensor([0, 7], dtype=torch.long),
            mb_x=torch.zeros((2, 3), dtype=torch.float32),
        )

        repair_plugin.after_eval_forward(eval_strategy)

        integrity_plugin = EvaluationIntegrityPlugin(controller_plugin=repair_plugin)
        with pytest.raises(RuntimeError, match='out of range'):
            integrity_plugin.after_eval_forward(eval_strategy)
