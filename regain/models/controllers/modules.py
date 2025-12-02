"""
Module definitions for controller models.
"""
import torch
from torch import nn


class BiasLayer(nn.Module):
    """
    Affine bias correction applied to a subset of class logits:
        logits[:, clss] = alpha * logits[:, clss] + beta
    """

    def __init__(self, clss: list[int]):
        super().__init__()
        clss_sorted = sorted({int(c) for c in clss})
        self.register_buffer("clss", torch.tensor(clss_sorted, dtype=torch.long))
        self.alpha = nn.Parameter(torch.ones(()))  # scalar
        self.beta = nn.Parameter(torch.zeros(()))  # scalar

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or self.clss.numel() == 0:
            return logits
        out = logits.clone()
        out[:, self.clss] = out[:, self.clss] * self.alpha + self.beta
        return out
