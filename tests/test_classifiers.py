"""
Tests for model classifiers.
"""

import torch

from regain.models.classifiers import ViTBaseClassifier
from regain.models.classifiers import ViTSmallClassifier


class TestVisionTransformerClassifiers:
    def test_vit_small_classifier_produces_logits(self) -> None:
        model = ViTSmallClassifier(n_classes=10)
        x = torch.randn(2, 3, 32, 32)

        logits = model(x)

        assert torch.is_tensor(logits)
        assert logits.shape == (2, 10)

    def test_vit_base_classifier_accepts_constructor_kwargs(self) -> None:
        model = ViTBaseClassifier(
            n_classes=7,
            image_size=32,
            patch_size=4,
            dropout=0.1,
        )
        x = torch.randn(3, 3, 32, 32)

        logits = model(x)

        assert torch.is_tensor(logits)
        assert logits.shape == (3, 7)
