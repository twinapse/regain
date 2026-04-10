"""
Tests for numerical stability guard plugin.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from regain.analysis.metrics import MetricContext
from regain.analysis.metrics import MetricPhase
from regain.avalanche_utils.plugins import NumericalStabilityGuardPlugin
# Ensure a stable import order for plugin module initialization.
import regain.experiments.orchestrator  # noqa: F401


class _DummyStrategy:
    def __init__(
        self,
        *,
        model: nn.Module | None = None,
        loss: object | None = None,
        mb_output: object | None = None,
        experience_idx: int = 0,
        eval_tag: str = '',
    ) -> None:
        self.model = model if model is not None else nn.Linear(4, 3)
        self.loss = loss
        self.mb_output = mb_output
        self.experience = SimpleNamespace(current_experience=int(experience_idx))
        setattr(self, '_regain_eval_tag', str(eval_tag))


class TestNumericalStabilityGuardPlugin:
    def test_before_backward_raises_with_context_on_non_finite_loss(self) -> None:
        context = MetricContext()
        context.set_phase(MetricPhase.TRAIN)
        context.set_log_step(11)
        plugin = NumericalStabilityGuardPlugin(context=context)
        strategy = _DummyStrategy(
            loss=torch.tensor(float('nan')),
            mb_output=torch.randn((2, 3), dtype=torch.float32),
            experience_idx=3,
        )

        with pytest.raises(RuntimeError) as exc_info:
            plugin.before_backward(strategy)
        error_msg = str(exc_info.value)
        assert 'tensor=loss' in error_msg
        assert 'phase=run.train' in error_msg
        assert 'exp_idx=3' in error_msg

    def test_exposes_only_training_time_hooks(self) -> None:
        plugin = NumericalStabilityGuardPlugin(context=MetricContext())

        assert 'after_eval_forward' not in type(plugin).__dict__

    def test_after_training_epoch_raises_on_non_finite_parameters(self) -> None:
        context = MetricContext()
        context.set_phase(MetricPhase.TRAIN)
        context.set_log_step(7)
        plugin = NumericalStabilityGuardPlugin(context=context)
        model = nn.Linear(2, 2)
        with torch.no_grad():
            model.weight.fill_(float('nan'))
        strategy = _DummyStrategy(
            model=model,
            loss=torch.tensor(0.0),
            mb_output=torch.randn((1, 2), dtype=torch.float32),
            experience_idx=1,
        )

        with pytest.raises(RuntimeError) as exc_info:
            plugin.after_training_epoch(strategy)
        error_msg = str(exc_info.value)
        assert 'tensor=model.parameters' in error_msg
        assert 'phase=run.train' in error_msg
