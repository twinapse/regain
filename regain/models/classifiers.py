"""
Classifiers for continual learning experiments.
"""

import torch
import torch.nn as nn

from regain.models.backbones import ResNet18Backbone
from regain.models.heads import LinearClassifier

__all__ = [
    'ResNet18Classifier',
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
