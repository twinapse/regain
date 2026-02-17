"""
Shared controller utilities.
"""

from torch import nn

from regain.constants import PARAM_BACKBONE

__all__ = [
    'resolve_backbone_or_raise',
]


def resolve_backbone_or_raise(
    *,
    model: nn.Module,
    error_message: str | None = None,
) -> nn.Module:
    """
    Resolve a backbone/encoder module from `model`.

    Args:
        model (nn.Module): Model expected to expose `.backbone` or `.encoder`.
        error_message (str | None): Optional error message override.

    Returns:
        nn.Module: Backbone/encoder module.

    Raises:
        ValueError: If neither `.backbone` nor `.encoder` is present.
    """
    # Prefer an explicit backbone, then fall back to an encoder attribute.
    backbone = getattr(model, PARAM_BACKBONE, None)
    # Return the backbone when present.
    if isinstance(backbone, nn.Module):
        return backbone
    encoder = getattr(model, 'encoder', None)
    # Return the encoder when present.
    if isinstance(encoder, nn.Module):
        return encoder
    if error_message is None:
        error_message = 'Controller requires the model to expose a `.backbone` (or `.encoder`) module.'
    raise ValueError(error_message)
