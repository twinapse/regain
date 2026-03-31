"""
Tests for model classifiers.
"""

import pytest
import torch

from regain.models import backbones as backbone_models
from regain.models.classifiers import ViTBaseClassifier
from regain.models.classifiers import ViTSmallClassifier


class _DummyPretrainedViT(torch.nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.patch_embed = torch.nn.Module()
        self.patch_embed.proj = torch.nn.Conv2d(3, out_dim, kernel_size=16, stride=16)
        self.patch_embed.img_size = (224, 224)
        self.blocks = torch.nn.ModuleList([
            torch.nn.Identity(),
            torch.nn.Identity(),
        ])
        self._out_dim = int(out_dim)
        self.seen_input_shape: tuple[int, ...] | None = None

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        self.seen_input_shape = tuple(int(dim) for dim in x.shape)
        return torch.ones(
            (int(x.shape[0]), self._out_dim),
            dtype=x.dtype,
            device=x.device,
        )


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

    def test_vit_small_classifier_supports_pretrained_backbone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}
        dummy_model = _DummyPretrainedViT(out_dim=384)

        def _fake_build_pretrained_timm_vit(*, model_name: str, image_size: int) -> torch.nn.Module:
            captured['model_name'] = model_name
            captured['image_size'] = image_size
            return dummy_model

        monkeypatch.setattr(
            backbone_models,
            '_build_pretrained_timm_vit',
            _fake_build_pretrained_timm_vit,
        )
        model = ViTSmallClassifier(n_classes=10, pretrained_backbone=True)
        x = torch.randn(2, 3, 32, 32)

        logits = model(x)

        assert torch.is_tensor(logits)
        assert logits.shape == (2, 10)
        assert dummy_model.seen_input_shape == (2, 3, 224, 224)
        assert captured == {
            'model_name': 'vit_small_patch16_224.augreg_in21k_ft_in1k',
            'image_size': 224,
        }

    def test_vit_base_classifier_rejects_invalid_pretrained_patch_size(self) -> None:
        with pytest.raises(ValueError, match='patch_size=16'):
            ViTBaseClassifier(
                n_classes=7,
                pretrained_backbone=True,
                image_size=224,
                patch_size=8,
            )
