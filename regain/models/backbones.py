"""
Backbone architectures used in retrieval experiments.
"""

import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from torchvision.models.vision_transformer import VisionTransformer

__all__ = [
    'ResNet18Backbone',
    'ViTBaseBackbone',
    'ViTSmallBackbone',
]

_DEFAULT_VIT_SCRATCH_IMAGE_SIZE = 32
_DEFAULT_VIT_SCRATCH_PATCH_SIZE = 4
_DEFAULT_VIT_PRETRAINED_IMAGE_SIZE = 224
_DEFAULT_VIT_PRETRAINED_PATCH_SIZE = 16
_TIMM_VIT_BASE_PRETRAINED_MODEL = 'vit_base_patch16_224.augreg_in21k_ft_in1k'
_TIMM_VIT_SMALL_PRETRAINED_MODEL = 'vit_small_patch16_224.augreg_in21k_ft_in1k'


def _resolve_vit_input_shape(
    *,
    image_size: int | None,
    patch_size: int | None,
    pretrained: bool,
) -> tuple[int, int]:
    """
    Resolve ViT input geometry defaults for scratch and pretrained variants.

    Args:
        image_size: Optional input image side length.
        patch_size: Optional patch side length.
        pretrained: Whether pretrained weights will be used.

    Returns:
        tuple[int, int]: Resolved `(image_size, patch_size)`.
    """
    default_image_size = (
        _DEFAULT_VIT_PRETRAINED_IMAGE_SIZE
        if pretrained
        else _DEFAULT_VIT_SCRATCH_IMAGE_SIZE
    )
    default_patch_size = (
        _DEFAULT_VIT_PRETRAINED_PATCH_SIZE
        if pretrained
        else _DEFAULT_VIT_SCRATCH_PATCH_SIZE
    )
    return (
        int(default_image_size if image_size is None else image_size),
        int(default_patch_size if patch_size is None else patch_size),
    )


def _build_pretrained_timm_vit(
    *,
    model_name: str,
    image_size: int,
) -> nn.Module:
    """
    Build a pretrained ViT backbone through `timm`.

    Args:
        model_name: `timm` model identifier.
        image_size: Input image side length.

    Returns:
        nn.Module: Pretrained timm ViT model.

    Raises:
        ModuleNotFoundError: If `timm` is unavailable.
        TypeError: If `timm.create_model` does not return a module.
    """
    try:
        timm = importlib.import_module('timm')
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'Pretrained ViT backbones require the optional `timm` dependency.'
        ) from exc

    model = timm.create_model(
        model_name,
        pretrained=True,
        img_size=int(image_size),
        num_classes=0,
    )
    if not isinstance(model, nn.Module):
        raise TypeError(
            f'timm.create_model returned {type(model).__name__}, expected torch.nn.Module.'
        )
    return model


def _resolve_pretrained_vit_image_size(*, model: nn.Module) -> tuple[int, int] | None:
    """
    Resolve the spatial input size expected by a pretrained timm ViT.

    Args:
        model: Pretrained timm ViT model.

    Returns:
        tuple[int, int] | None: Expected `(height, width)`, or None if unavailable.
    """
    patch_embed = getattr(model, 'patch_embed', None)
    image_size = getattr(patch_embed, 'img_size', None)
    if isinstance(image_size, (tuple, list)) and len(image_size) == 2:
        return int(image_size[0]), int(image_size[1])
    if isinstance(image_size, int):
        side = int(image_size)
        return side, side
    return None


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


class _VisionTransformerBackbone(nn.Module):
    """
    Vision transformer backbone with scratch and pretrained variants.

    Args:
        image_size (int | None): Input image side length (square inputs).
        patch_size (int | None): Patch side length.
        hidden_dim (int): Token embedding dimension.
        mlp_dim (int): Transformer feed-forward hidden dimension.
        num_layers (int): Number of transformer blocks.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
        attention_dropout (float): Attention dropout probability.
        pretrained (bool): Whether to load pretrained `timm` weights.
        pretrained_model_name (str | None): `timm` model identifier for pretrained mode.
    """

    def __init__(
        self,
        *,
        image_size: int | None,
        patch_size: int | None,
        hidden_dim: int,
        mlp_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        pretrained: bool = False,
        pretrained_model_name: str | None = None,
    ) -> None:
        super().__init__()
        image_size, patch_size = _resolve_vit_input_shape(
            image_size=image_size,
            patch_size=patch_size,
            pretrained=pretrained,
        )
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

        self._uses_pretrained_model = bool(pretrained)
        if self._uses_pretrained_model:
            if pretrained_model_name is None or str(pretrained_model_name).strip() == '':
                raise ValueError(
                    'pretrained_model_name is required when `pretrained=True`.'
                )
            if int(patch_size) != _DEFAULT_VIT_PRETRAINED_PATCH_SIZE:
                raise ValueError(
                    'Pretrained ViT backbones currently require patch_size=16.'
                )
            self.model = _build_pretrained_timm_vit(
                model_name=str(pretrained_model_name),
                image_size=int(image_size),
            )
        else:
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
        if self._uses_pretrained_model:
            expected_image_size = _resolve_pretrained_vit_image_size(model=self.model)
            if expected_image_size is not None:
                expected_height, expected_width = expected_image_size
                if (
                    int(x.shape[-2]) != expected_height
                    or int(x.shape[-1]) != expected_width
                ):
                    x = F.interpolate(
                        x,
                        size=(expected_height, expected_width),
                        mode='bilinear',
                        align_corners=False,
                    )
            features = self.model.forward_features(x)
            if not torch.is_tensor(features):
                raise TypeError('Pretrained ViT backbone must return tensor features.')
            if features.ndim == 2:
                return features
            if features.ndim == 3:
                return features[:, 0]
            raise ValueError(
                'Pretrained ViT backbone returned unsupported feature shape '
                f'{tuple(features.shape)}.'
            )

        n = int(x.shape[0])
        x = self.model.conv_proj(x)
        x = x.reshape(n, int(self.model.hidden_dim), -1).permute(0, 2, 1)
        class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([class_token, x], dim=1)
        x = self.model.encoder(x)
        return x[:, 0]


class ViTSmallBackbone(_VisionTransformerBackbone):
    """
    ViT-S backbone.

    Args:
        image_size: Input image side length.
        patch_size: Patch side length.
        dropout: Dropout probability.
        attention_dropout: Attention dropout probability.
        pretrained: Whether to load pretrained weights through `timm`.
    """

    def __init__(
        self,
        *,
        image_size: int | None = None,
        patch_size: int | None = None,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        pretrained: bool = False,
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
            pretrained=pretrained,
            pretrained_model_name=_TIMM_VIT_SMALL_PRETRAINED_MODEL,
        )


class ViTBaseBackbone(_VisionTransformerBackbone):
    """
    ViT-B backbone.

    Args:
        image_size: Input image side length.
        patch_size: Patch side length.
        dropout: Dropout probability.
        attention_dropout: Attention dropout probability.
        pretrained: Whether to load pretrained weights through `timm`.
    """

    def __init__(
        self,
        *,
        image_size: int | None = None,
        patch_size: int | None = None,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        pretrained: bool = False,
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
            pretrained=pretrained,
            pretrained_model_name=_TIMM_VIT_BASE_PRETRAINED_MODEL,
        )
