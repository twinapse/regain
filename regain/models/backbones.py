"""
Backbone architectures used in retrieval experiments.
"""

import torch
import torch.nn as nn
import torchvision.models as tvm
from torchvision.models.vision_transformer import VisionTransformer

__all__ = [
    'ResNet18Backbone',
    'ViTBaseBackbone',
    'ViTSmallBackbone',
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
        # Use the `pretrained` argument for compatibility across torchvision variants.
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


class _TorchVisionViTBackbone(nn.Module):
    """
    Vision transformer backbone based on torchvision's `VisionTransformer`.

    Args:
        image_size (int): Input image side length (square inputs).
        patch_size (int): Patch side length.
        hidden_dim (int): Token embedding dimension.
        mlp_dim (int): Transformer feed-forward hidden dimension.
        num_layers (int): Number of transformer blocks.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
        attention_dropout (float): Attention dropout probability.
    """

    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        hidden_dim: int,
        mlp_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if int(image_size) <= 0:
            raise ValueError('image_size must be > 0.')
        if int(patch_size) <= 0:
            raise ValueError('patch_size must be > 0.')
        if int(image_size) % int(patch_size) != 0:
            raise ValueError('image_size must be divisible by patch_size.')
        if int(hidden_dim) <= 0:
            raise ValueError('hidden_dim must be > 0.')
        if int(mlp_dim) <= 0:
            raise ValueError('mlp_dim must be > 0.')
        if int(num_layers) <= 0:
            raise ValueError('num_layers must be > 0.')
        if int(num_heads) <= 0:
            raise ValueError('num_heads must be > 0.')
        if int(hidden_dim) % int(num_heads) != 0:
            raise ValueError('hidden_dim must be divisible by num_heads.')

        self.model = VisionTransformer(
            image_size=int(image_size),
            patch_size=int(patch_size),
            num_layers=int(num_layers),
            num_heads=int(num_heads),
            hidden_dim=int(hidden_dim),
            mlp_dim=int(mlp_dim),
            dropout=float(dropout),
            attention_dropout=float(attention_dropout),
            num_classes=1000,
        )
        self.out_dim = int(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the backbone and return CLS-token features.

        Args:
            x: Batch of images shaped `(batch, channels, height, width)`.

        Returns:
            torch.Tensor: Feature vectors shaped `(batch, out_dim)`.
        """
        n = int(x.shape[0])
        x = self.model.conv_proj(x)
        x = x.reshape(n, int(self.model.hidden_dim), -1).permute(0, 2, 1)
        class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([class_token, x], dim=1)
        x = self.model.encoder(x)
        return x[:, 0]


class ViTSmallBackbone(_TorchVisionViTBackbone):
    """
    ViT-S backbone

    Args:
        image_size: Input image side length.
        patch_size: Patch side length.
        dropout: Dropout probability.
        attention_dropout: Attention dropout probability.
    """

    def __init__(
        self,
        *,
        image_size: int = 32,
        patch_size: int = 4,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            hidden_dim=384,
            mlp_dim=1536,
            num_layers=12,
            num_heads=6,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )


class ViTBaseBackbone(_TorchVisionViTBackbone):
    """
    ViT-B backbone.

    Args:
        image_size: Input image side length.
        patch_size: Patch side length.
        dropout: Dropout probability.
        attention_dropout: Attention dropout probability.
    """

    def __init__(
        self,
        *,
        image_size: int = 32,
        patch_size: int = 4,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            hidden_dim=768,
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )
