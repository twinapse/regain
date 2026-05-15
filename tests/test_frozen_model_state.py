"""
Tests for the frozen model-state evaluation guard.
"""

import pytest
import torch
from torch import nn

from regain.evaluation import frozen_model_state


class _ToyModel(nn.Module):
    """
    Toy model for testing.
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features=3, out_features=2)


class TestFrozenModelState:
    """
    Tests for FrozenModelState.
    """

    def test_allows_unchanged_state(self) -> None:
        model = _ToyModel()

        with frozen_model_state(model=model):
            _ = model.linear.weight

    def test_raises_when_model_state_mutates(self) -> None:
        model = _ToyModel()

        with pytest.raises(RuntimeError, match='state tensor signature changed'):
            with frozen_model_state(model=model):
                with torch.no_grad():
                    model.linear.weight.add_(1.0)
