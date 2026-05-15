"""
Tests for repair controller plugin.
"""

import random
import types
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from regain.avalanche_utils.plugins import RepairControllerPlugin
from regain.constants import _DEBUG_N_SAMPLES
from regain.constants import _DEBUG_NUM_CLASSES
# Ensure a stable import order for plugin module initialization.
import regain.experiments.orchestrator  # noqa: F401
import regain.debug.avalanche_utils as debug_utils
from regain.debug.avalanche_utils import DebugRepairControllerPlugin
from regain.models.controllers import RepairController

################
# Test helpers #
################

class _IdentityModel(nn.Module):
    """Minimal model that returns its inputs unchanged."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _DummyStrategy:
    """Small strategy stub exposing the attributes the plugin reads."""

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
    """Repair controller test double with configurable corrected outputs."""

    def __init__(
        self,
        *,
        corrected_outputs: torch.Tensor | None = None,
        fit_side_effect: Any | None = None,
    ) -> None:
        super().__init__()
        self.corrected_outputs = corrected_outputs
        self.fit_side_effect = fit_side_effect
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
        if self.fit_side_effect is not None:
            self.fit_side_effect()
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


class _ToyRepairDataset(Dataset):
    """Dataset helper that also supports subset selection like repair datasets."""

    def __init__(self, *, targets: list[int], original_indices: list[int]) -> None:
        self.targets = list(targets)
        self.original_indices = list(original_indices)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return torch.tensor([float(idx)]), int(self.targets[idx])

    def subset(self, indices: list[int]) -> '_ToyRepairDataset':
        local_indices = [int(index) for index in indices]
        return _ToyRepairDataset(
            targets=[int(self.targets[index]) for index in local_indices],
            original_indices=[int(self.original_indices[index]) for index in local_indices],
        )


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
        budget_fraction=1.0,
        seed=1,
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


def _assert_numpy_rng_state_equal(
    state_after: tuple[Any, ...],
    state_before: tuple[Any, ...],
) -> None:
    """Assert NumPy RNG state tuples are identical."""
    assert state_after[0] == state_before[0]
    assert np.array_equal(state_after[1], state_before[1])
    assert state_after[2:] == state_before[2:]


def _make_repair_strategy(
    *,
    model: nn.Module,
    repair_dataset: _ToyRepairDataset,
    seen_classes: list[int],
    exp_idx: int = 0,
) -> _DummyStrategy:
    """Build a strategy stub whose experience exposes a repair stream."""
    experience = types.SimpleNamespace(
        benchmark=types.SimpleNamespace(
            repair_stream=[types.SimpleNamespace(dataset=repair_dataset)],
        ),
        classes_in_this_experience=seen_classes,
        current_experience=exp_idx,
    )
    return _DummyStrategy(
        model=model,
        mb_output=torch.zeros((1, max(seen_classes) + 1), dtype=torch.float32),
        mb_x=torch.zeros((1, 1), dtype=torch.float32),
        experience=experience,
    )


#################################
# Budget selection + guard rules #
#################################


class TestRepairControllerPluginBudgetSelection:
    """Tests for repair budget selection behavior."""

    def test_requires_explicit_seed(self) -> None:
        with pytest.raises(TypeError, match="missing 1 required keyword-only argument: 'seed'"):
            RepairControllerPlugin(
                controller=_ScriptedRepairController(),
                fit_after_experience=False,
                repair_epochs=1,
                repair_batch_size=1,
            )

    def test_raises_when_budget_fraction_is_out_of_range(self) -> None:
        with pytest.raises(ValueError, match='range'):
            RepairControllerPlugin(
                controller=_ScriptedRepairController(),
                fit_after_experience=False,
                repair_epochs=1,
                repair_batch_size=1,
                budget_fraction=0.0,
                seed=1,
            )
        with pytest.raises(ValueError, match='range'):
            RepairControllerPlugin(
                controller=_ScriptedRepairController(),
                fit_after_experience=False,
                repair_epochs=1,
                repair_batch_size=1,
                budget_fraction=1.2,
                seed=7,
            )

    def test_selects_deterministic_nested_stratified_subsets(self) -> None:
        dataset = _ToyRepairDataset(
            targets=[0, 0, 0, 0, 1, 1, 1, 1],
            original_indices=[10, 11, 12, 13, 20, 21, 22, 23],
        )

        plugin_b1 = RepairControllerPlugin(
            controller=_ScriptedRepairController(),
            fit_after_experience=False,
            repair_epochs=1,
            repair_batch_size=1,
            budget_fraction=0.25,
            seed=42,
        )
        plugin_b2 = RepairControllerPlugin(
            controller=_ScriptedRepairController(),
            fit_after_experience=False,
            repair_epochs=1,
            repair_batch_size=1,
            budget_fraction=0.5,
            seed=42,
        )
        plugin_b2_repeat = RepairControllerPlugin(
            controller=_ScriptedRepairController(),
            fit_after_experience=False,
            repair_epochs=1,
            repair_batch_size=1,
            budget_fraction=0.5,
            seed=42,
        )

        selected_b1 = plugin_b1._select_budget_fraction(repair_dataset=dataset, exp_idx=0)
        selected_b2 = plugin_b2._select_budget_fraction(repair_dataset=dataset, exp_idx=0)
        selected_b2_repeat = plugin_b2_repeat._select_budget_fraction(
            repair_dataset=dataset,
            exp_idx=0,
        )

        assert selected_b1 is not None
        assert selected_b2 is not None
        assert selected_b2_repeat is not None

        assert selected_b1.targets.count(0) == 1
        assert selected_b1.targets.count(1) == 1
        assert selected_b2.targets.count(0) == 2
        assert selected_b2.targets.count(1) == 2
        assert set(selected_b1.original_indices).issubset(set(selected_b2.original_indices))
        assert selected_b2.original_indices == selected_b2_repeat.original_indices

    def test_preserves_distribution_on_imbalanced_repair_set(self) -> None:
        dataset = _ToyRepairDataset(
            targets=[0, 0, 0, 0, 0, 0, 1, 1],
            original_indices=[10, 11, 12, 13, 14, 15, 20, 21],
        )
        plugin = RepairControllerPlugin(
            controller=_ScriptedRepairController(),
            fit_after_experience=False,
            repair_epochs=1,
            repair_batch_size=1,
            budget_fraction=0.5,
            seed=11,
        )

        selected = plugin._select_budget_fraction(
            repair_dataset=dataset,
            exp_idx=0,
        )

        assert selected is not None
        assert len(selected) == 4
        assert selected.targets.count(0) == 3
        assert selected.targets.count(1) == 1


class TestRepairControllerPluginRngIsolation:
    """Tests that repair fitting does not perturb outer RNG streams."""

    @staticmethod
    def _make_rng_consuming_plugin(
        *,
        fit_side_effect: Any,
        fit_after_experience: bool,
    ) -> RepairControllerPlugin:
        """Build a repair plugin whose fit consumes RNG internally."""
        return RepairControllerPlugin(
            controller=_ScriptedRepairController(fit_side_effect=fit_side_effect),
            fit_after_experience=fit_after_experience,
            repair_epochs=1,
            repair_batch_size=2,
            budget_fraction=1.0,
            seed=17,
        )

    def test_fit_controller_on_repair_dataset_preserves_rng_state(self) -> None:
        dataset = _ToyRepairDataset(
            targets=[0, 1, 0, 1],
            original_indices=[0, 1, 2, 3],
        )
        plugin = self._make_rng_consuming_plugin(
            fit_side_effect=lambda: (
                random.random(),
                np.random.rand(),
                torch.rand(8),
            ),
            fit_after_experience=False,
        )

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        random.random()
        np.random.rand()
        torch.rand(1)

        python_state_before = random.getstate()
        numpy_state_before = np.random.get_state()
        torch_state_before = torch.random.get_rng_state()

        plugin._fit_controller_on_repair_dataset(
            model=_IdentityModel(),
            repair_dataset=dataset,
            new_classes=[0, 1],
            exp_idx=0,
        )

        python_state_after = random.getstate()
        numpy_state_after = np.random.get_state()
        torch_state_after = torch.random.get_rng_state()

        assert python_state_after == python_state_before
        _assert_numpy_rng_state_equal(numpy_state_after, numpy_state_before)
        assert torch.equal(torch_state_after, torch_state_before)

    def test_after_training_exp_leaves_same_torch_rng_state_across_controllers(self) -> None:
        dataset = _ToyRepairDataset(
            targets=[0, 1, 0, 1],
            original_indices=[10, 11, 12, 13],
        )

        def _run_after_training_exp(*, fit_side_effect: Any) -> torch.Tensor:
            plugin = self._make_rng_consuming_plugin(
                fit_side_effect=fit_side_effect,
                fit_after_experience=True,
            )
            strategy = _make_repair_strategy(
                model=_IdentityModel(),
                repair_dataset=dataset,
                seen_classes=[0, 1],
            )
            plugin.before_training_exp(strategy)

            random.seed(7)
            np.random.seed(7)
            torch.manual_seed(7)
            random.random()
            np.random.rand()
            torch.rand(1)

            plugin.after_training_exp(strategy)
            return torch.random.get_rng_state()

        state_a = _run_after_training_exp(
            fit_side_effect=lambda: (
                random.random(),
                np.random.rand(2),
                torch.rand(4),
            ),
        )
        state_b = _run_after_training_exp(
            fit_side_effect=lambda: (
                random.random(),
                np.random.rand(32),
                torch.rand(32),
            ),
        )

        assert torch.equal(state_a, state_b)

    def test_fit_seed_is_deterministic_and_separates_final_fit(self) -> None:
        seed_exp0_a = RepairControllerPlugin._fit_seed(seed=17, exp_idx=0)
        seed_exp0_b = RepairControllerPlugin._fit_seed(seed=17, exp_idx=0)
        seed_final = RepairControllerPlugin._fit_seed(seed=17, exp_idx=None)

        assert seed_exp0_a == seed_exp0_b
        assert seed_final != seed_exp0_a


class TestDebugRepairControllerPlugin:
    """Tests for debug repair controller plugin diagnostics behavior."""

    def test_debug_fit_uses_plugin_seed_for_diagnostics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        call_kwargs: list[dict[str, object]] = []

        def _fake_compute_repair_diagnostics(**kwargs: object) -> dict[str, object]:
            call_kwargs.append(dict(kwargs))
            return {
                _DEBUG_NUM_CLASSES: 2,
                _DEBUG_N_SAMPLES: 4,
            }

        monkeypatch.setattr(debug_utils, 'compute_repair_diagnostics', _fake_compute_repair_diagnostics)
        monkeypatch.setattr(debug_utils, 'compute_repair_health_score', lambda **kwargs: kwargs)
        monkeypatch.setattr(
            DebugRepairControllerPlugin,
            '_record_health_score',
            lambda self, *, health_payload, exp_idx, step: None,
        )

        plugin = DebugRepairControllerPlugin(
            controller=_ScriptedRepairController(),
            fit_after_experience=False,
            repair_epochs=1,
            repair_batch_size=2,
            budget_fraction=1.0,
            seed=11,
            debug_epochs=5,
            debug_experiences=3,
        )

        plugin._run_debug_fit(
            model=_IdentityModel(),
            repair_dataset=_ToyRepairDataset(
                targets=[0, 1, 0, 1],
                original_indices=[0, 1, 2, 3],
            ),
            new_classes=[0, 1],
            exp_idx=0,
        )

        assert len(call_kwargs) == 5
        assert all(kwargs['debug_seed'] == 11 for kwargs in call_kwargs)
        assert all('seed' not in kwargs for kwargs in call_kwargs)


###########################
# Output contract coverage #
###########################

class TestRepairControllerPluginOutputContract:
    """Tests for controller output-shape and seen-class update rules."""

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
    """Tests that repair controllers cannot alter unseen-class behavior."""

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
    """Tests for repair controller evaluation hook boundaries."""

    def test_eval_hooks_do_not_receive_strategy_or_experience(self) -> None:
        controller = _ScriptedRepairController(corrected_outputs=None)
        plugin = RepairControllerPlugin(
            controller=controller,
            fit_after_experience=False,
            repair_epochs=1,
            repair_batch_size=1,
            budget_fraction=1.0,
            seed=1,
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
