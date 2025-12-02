"""
Classifier heads used in retrieval and CL experiments.
"""

import torch
import torch.nn as nn

__all__ = [
    'LinearClassifier',
]


class LinearClassifier(nn.Module):
    """
    Linear classifier over backbone features.

    Args:
        in_dim: Input feature dimensionality from the backbone.
        n_classes: Number of output classes.
    """

    def __init__(self, in_dim: int, n_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute logits from backbone features.

        Args:
            x: Batch of feature vectors shaped `(batch, in_dim)`.

        Returns:
            Logits of shape `(batch, n_classes)`.
        """
        return self.fc(x)
