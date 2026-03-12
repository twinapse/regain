"""
Classifiers for continual learning experiments.
"""

import torch
import torch.nn as nn

from regain.models.backbones import ResNet18Backbone
from regain.models.backbones import ViTSmallBackbone
from regain.models.backbones import ViTBaseBackbone
from regain.models.heads import LinearClassifier

__all__ = [
    'ResNet18Classifier',
    'ViTBaseClassifier',
    'ViTSmallClassifier',
]


class ResNet18Classifier(nn.Module):
    """
    ResNet-18 backbone paired with a linear classifier head.

    This assembly is intended for class-incremental continual learning (CIL) experiments.

    Args:
        n_classes: Number of output classes.
        pretrained_backbone: Whether to initialize the backbone with pretrained weights.
    """

    def __init__(self, n_classes: int, pretrained_backbone: bool = False) -> None:
        super().__init__()
        self.backbone = ResNet18Backbone(pretrained=pretrained_backbone)
        self.classifier = LinearClassifier(in_dim=self.backbone.out_dim, n_classes=n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the backbone and classifier to produce logits.

        Args:
            x: Batch of images shaped `(batch, channels, height, width)`.

        Returns:
            Logits for each class with shape `(batch, n_classes)`.
        """
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits


class ViTSmallClassifier(nn.Module):
    """
    ViT-S backbone paired with a linear classifier head.

    Args:
        n_classes: Number of output classes.
        image_size: Input image side length.
        patch_size: Patch side length.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        n_classes: int,
        image_size: int = 32,
        patch_size: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = ViTSmallBackbone(
            image_size=image_size,
            patch_size=patch_size,
            dropout=dropout,
        )
        self.classifier = LinearClassifier(in_dim=self.backbone.out_dim, n_classes=n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the backbone and classifier to produce logits.

        Args:
            x: Batch of images shaped `(batch, channels, height, width)`.

        Returns:
            torch.Tensor: Logits for each class with shape `(batch, n_classes)`.
        """
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits


class ViTBaseClassifier(nn.Module):
    """
    ViT-B backbone paired with a linear classifier head.

    Args:
        n_classes: Number of output classes.
        image_size: Input image side length.
        patch_size: Patch side length.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        n_classes: int,
        image_size: int = 32,
        patch_size: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = ViTBaseBackbone(
            image_size=image_size,
            patch_size=patch_size,
            dropout=dropout,
        )
        self.classifier = LinearClassifier(in_dim=self.backbone.out_dim, n_classes=n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the backbone and classifier to produce logits.

        Args:
            x: Batch of images shaped `(batch, channels, height, width)`.

        Returns:
            torch.Tensor: Logits for each class with shape `(batch, n_classes)`.
        """
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits
