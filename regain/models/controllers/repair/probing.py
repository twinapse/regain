"""
Probe-based baselines.

This module implements a linear probe trained on backbone features using repair data, serving as a higher-capacity
readout baseline within the controller family.
"""

from typing import Any

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from regain.models.controllers.base import RepairController
from regain.models.controllers.repair.common import build_repair_dataloader
from regain.models.controllers.repair.common import build_sgd_optimizer_and_scheduler
from regain.models.controllers.repair.common import extract_probe_inputs
from regain.models.controllers.repair.common import fit_repair_controller
from regain.models.controllers.repair.common import resolve_backbone_or_raise
from regain.utils import module_device
from regain.utils import preserve_model_mode_after_eval

__all__ = [
    'LinearProbeController',
]


class LinearProbeController(RepairController):
    """
    Linear probe trained on frozen backbone features and used at evaluation time.

    Expected model interface: `model.backbone(x)` (or `model.encoder(x)`) returns feature vectors `(B, feat_dim)`.

    Args:
        lr: Learning rate for fitting.
        device: Device for fitting.
        seed: Random seed for dataloader shuffling.
        momentum: SGD momentum for probe fitting.
        weight_decay: Weight decay for probe fitting.
    """

    def __init__(
        self,
        lr: float,
        device: str | None = None,
        seed: int = 1,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()

        # Hyperparameters
        self.lr = float(lr)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = int(seed)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)

        # State
        self._fitted: bool = False
        self._probe: nn.Linear | None = None
        self._seen_classes: set[int] = set()

    def fit_on_repair_data(
        self,
        *,
        model: nn.Module,
        repair_dataset: Dataset | None,
        new_classes: list[int],
        num_epochs: int,
        batch_size: int,
    ) -> None:
        """
        Fit the probe on frozen backbone features.

        Args:
            model (nn.Module): Model exposing `.backbone` or `.encoder`.
            repair_dataset (Dataset | None): Repair dataset for fitting.
            new_classes (list[int]): Newly introduced classes.
            num_epochs (int): Number of epochs used for fitting.
            batch_size (int): Batch size used for fitting.

        Returns:
            None.

        Raises:
            ValueError: If the model does not expose a `.backbone` or `.encoder` module.
        """
        try:
            backbone = resolve_backbone_or_raise(model=model)
        except ValueError as exc:
            raise ValueError('Model must expose a `.backbone` or `.encoder` module.') from exc

        self._seen_classes.update(int(c) for c in new_classes)

        dataloader = build_repair_dataloader(
            repair_dataset=repair_dataset,
            batch_size=batch_size,
            seed=self.seed,
        )
        if dataloader is None:
            return

        model_device = module_device(model, self.device)

        feat_dim = self._infer_feat_dim(backbone=backbone, dataloader=dataloader, device=model_device)
        if feat_dim is None:
            return

        if not self._seen_classes:
            return
        num_classes = max(self._seen_classes) + 1
        probe = nn.Linear(feat_dim, num_classes).to(model_device)

        optimizer, scheduler = build_sgd_optimizer_and_scheduler(
            params=list(probe.parameters()),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            lr_milestones=None,
            lr_gamma=1.0,
        )
        criterion = CrossEntropyLoss()

        def _forward(x: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                feats = backbone(x)
            if not torch.is_tensor(feats) or feats.ndim != 2:
                return feats
            return probe(feats)

        fit_repair_controller(
            controller=probe,
            model=model,
            dataloader=dataloader,
            device=model_device,
            num_epochs=int(num_epochs),
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            forward_fn=_forward,
            reg_term_fn=None,
            logits_error='Expected logits shaped (B, C).',
        )

        self._probe = probe
        self._fitted = True

    def correct_outputs(self, *, outputs: Any, model: nn.Module | None = None, inputs: Any | None = None) -> Any:
        """
        Replace logits with probe logits during evaluation when enabled.

        Args:
            outputs (Any): Raw model outputs (ignored if probe is available).
            model (nn.Module | None): Model exposing `.backbone` or `.encoder`.
            inputs (Any | None): Input batch used to compute features.

        Returns:
            Any: Logits from the probe when available; otherwise `outputs`.
        """
        if not self.is_enabled() or not self._fitted or self._probe is None:
            return outputs
        if model is None or inputs is None:
            return outputs
        if not torch.is_tensor(inputs):
            return outputs

        try:
            backbone = resolve_backbone_or_raise(model=model)
        except ValueError:
            return outputs

        model_device = module_device(model, self.device)
        with preserve_model_mode_after_eval(model):
            with torch.inference_mode():
                feats = backbone(inputs.to(model_device))
                logits = self._probe(feats)
        return logits

    @staticmethod
    def _infer_feat_dim(
        *,
        backbone: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> int | None:
        """
        Infer the backbone feature dimensionality from the first batch.

        Args:
            backbone (nn.Module): Backbone module.
            dataloader (DataLoader): Dataloader yielding `(x, y, ...)` batches.
            device (torch.device): Device for running the forward pass.

        Returns:
            int | None: Feature dimension, or None if it cannot be inferred.
        """
        probe_inputs = extract_probe_inputs(dataloader=dataloader, device=device)
        if probe_inputs is None:
            return None

        with preserve_model_mode_after_eval(backbone):
            with torch.inference_mode():
                feats = backbone(probe_inputs)

        if not torch.is_tensor(feats) or feats.ndim != 2:
            return None

        return int(feats.shape[1])
