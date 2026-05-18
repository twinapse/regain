"""
Tests for per-batch evaluation invariants.
"""

import pytest
import torch

from regain.evaluation import check_eval_batch


class TestCheckEvalBatch:
    """
    Tests for evaluation batch invariant checking.
    """

    def test_raises_when_logits_are_not_2d(self) -> None:
        with pytest.raises(RuntimeError, match='must be 2D'):
            check_eval_batch(
                logits=torch.tensor([1.0, 2.0], dtype=torch.float32),
                targets=torch.tensor([0], dtype=torch.long),
                num_classes=2,
            )

    def test_raises_when_batch_sizes_do_not_match(self) -> None:
        with pytest.raises(RuntimeError, match='must match logits batch size'):
            check_eval_batch(
                logits=torch.randn((2, 3), dtype=torch.float32),
                targets=torch.tensor([0], dtype=torch.long),
                num_classes=3,
            )

    def test_raises_when_logits_are_non_finite(self) -> None:
        with pytest.raises(RuntimeError, match='non-finite'):
            check_eval_batch(
                logits=torch.tensor([[1.0, float('nan')]], dtype=torch.float32),
                targets=torch.tensor([0], dtype=torch.long),
                num_classes=2,
            )

    def test_raises_when_targets_are_out_of_range(self) -> None:
        with pytest.raises(RuntimeError, match='out of range'):
            check_eval_batch(
                logits=torch.randn((1, 2), dtype=torch.float32),
                targets=torch.tensor([4], dtype=torch.long),
                num_classes=2,
            )
