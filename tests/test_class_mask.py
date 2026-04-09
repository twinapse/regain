"""
Tests for seen-class masking.
"""

import torch

from regain.evaluation import ClassMask


class TestClassMask:
    def test_masks_unseen_columns(self) -> None:
        mask = ClassMask.from_seen_classes([0, 2], mask_value=-5.0)
        logits = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=torch.float32,
        )

        masked = mask.apply(logits=logits)

        assert torch.equal(masked[:, 0], logits[:, 0])
        assert torch.equal(masked[:, 2], logits[:, 2])
        assert torch.all(masked[:, 1] == torch.tensor(-5.0))
