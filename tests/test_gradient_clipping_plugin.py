"""
Tests for gradient clipping plugin behavior.
"""

from types import SimpleNamespace

import pytest
import torch

from regain.avalanche_utils.plugins import GradientClippingPlugin


class TestGradientClippingPlugin:
    def test_clips_gradients_before_update(self) -> None:
        model = torch.nn.Linear(3, 1, bias=False)
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 10.0)

        strategy = SimpleNamespace(model=model)
        plugin = GradientClippingPlugin(max_norm=1.0)

        plugin.before_update(strategy)

        grad_norm = torch.norm(model.weight.grad, p=2).item()
        assert grad_norm <= 1.0 + 1e-6

    def test_noops_when_no_gradients_are_available(self) -> None:
        model = torch.nn.Linear(3, 1)
        strategy = SimpleNamespace(model=model)
        plugin = GradientClippingPlugin(max_norm=1.0)

        plugin.before_update(strategy)

        assert all(parameter.grad is None for parameter in model.parameters())
