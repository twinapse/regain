"""
Backbone architectures used in retrieval experiments.
"""

import torch
import torch.nn as nn
import torchvision.models as tvm

__all__ = [
    'ResNet18Backbone',
]


class ResNet18Backbone(nn.Module):
    """
    ResNet-18 backbone that outputs flattened feature vectors.

    The final fully connected classification layer is removed; the attribute
    ``out_dim`` exposes the feature dimension used by downstream heads.

    Args:
        pretrained: Whether to load ImageNet-pretrained weights.
    """

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        # Use the legacy `pretrained` flag for broad torchvision compatibility.
        resnet = tvm.resnet18(pretrained=pretrained)
        # Strip the final classification layer.
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.out_dim = resnet.fc.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the backbone and flatten pooled features.

        Args:
            x: Batch of images shaped `(batch, channels, height, width)`.

        Returns:
            Flattened feature vectors of shape `(batch, out_dim)`.
        """
        feats = self.features(x)
        # ResNet outputs (B, C, 1, 1) after global pooling.
        return feats.view(feats.size(0), -1)
