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
from regain.models.controllers.repair.common import as_batch_tensors
from regain.models.controllers.repair.common import build_repair_dataloader
from regain.models.controllers.repair.common import build_sgd_optimizer_and_scheduler
from regain.models.controllers.repair.common import extract_probe_inputs
from regain.models.controllers.repair.common import fit_repair_controller
from regain.models.controllers.utils import resolve_backbone_or_raise
from regain.utils import module_device
from regain.utils import preserve_model_mode_after_eval

__all__ = [
    'LinearProbeController',
    'PrototypeBlendController',
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
        Replace logits with probe logits during evaluation.

        Args:
            outputs (Any): Raw model outputs (ignored if probe is available).
            model (nn.Module | None): Model exposing `.backbone` or `.encoder`.
            inputs (Any | None): Input batch used to compute features.

        Returns:
            Any: Logits from the probe when available; otherwise `outputs`.
        """
        if not self._fitted or self._probe is None:
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


class PrototypeBlendController(RepairController):
    """
    Frozen-feature prototype baseline blended with classifier logits.

    Class prototypes are estimated from repair examples in the frozen backbone feature space. At evaluation time, the
    controller computes prototype-similarity scores for classes with prototypes and blends those scores into the
    corresponding classifier logits.

    Args:
        blend (float): Interpolation weight for prototype scores, where `0` keeps classifier logits and `1` uses
            prototype scores for prototype-backed classes.
        score_scale (float): Scalar applied to cosine prototype similarities before blending.
        normalize (bool): Whether to use cosine-style normalized features and prototypes.
        device (str | None): Device for feature extraction.
        seed (int): Random seed for dataloader ordering.
    """

    def __init__(
        self,
        *,
        blend: float = 0.5,
        score_scale: float = 10.0,
        normalize: bool = True,
        device: str | None = None,
        seed: int = 1,
    ) -> None:
        super().__init__()
        self.blend = float(blend)
        self.score_scale = float(score_scale)
        self.normalize = bool(normalize)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = int(seed)
        if not (0.0 <= self.blend <= 1.0):
            raise ValueError('`blend` must be in the range [0, 1].')

        self._fitted = False
        self._prototype_by_class: dict[int, torch.Tensor] = {}
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
        del num_epochs
        try:
            backbone = resolve_backbone_or_raise(model=model)
        except ValueError as exc:
            raise ValueError('Model must expose a `.backbone` or `.encoder` module.') from exc

        self._seen_classes.update(int(class_id) for class_id in new_classes)
        dataloader = build_repair_dataloader(
            repair_dataset=repair_dataset,
            batch_size=batch_size,
            seed=self.seed,
            shuffle=False,
        )
        if dataloader is None:
            return

        model_device = module_device(model, self.device)
        feat_sums: dict[int, torch.Tensor] = {}
        feat_counts: dict[int, int] = {}
        with preserve_model_mode_after_eval(backbone):
            with torch.inference_mode():
                for batch in dataloader:
                    tensors = as_batch_tensors(batch, device=model_device)
                    if tensors is None:
                        continue
                    x, y = tensors
                    feats = backbone(x)
                    if not torch.is_tensor(feats) or feats.ndim != 2:
                        return
                    feats = feats.detach().to(dtype=torch.float32)
                    if self.normalize:
                        feats = torch.nn.functional.normalize(feats, p=2, dim=1)
                    for class_id in sorted({int(value) for value in y.detach().cpu().tolist()}):
                        mask = y.eq(class_id)
                        if not bool(mask.any()):
                            continue
                        class_sum = feats[mask].sum(dim=0).detach().cpu()
                        feat_sums[class_id] = feat_sums.get(class_id, torch.zeros_like(class_sum)) + class_sum
                        feat_counts[class_id] = int(feat_counts.get(class_id, 0) + int(mask.sum().item()))

        prototypes: dict[int, torch.Tensor] = {}
        for class_id, feat_sum in feat_sums.items():
            count = int(feat_counts.get(class_id, 0))
            if count <= 0:
                continue
            prototype = feat_sum / float(count)
            if self.normalize:
                prototype = torch.nn.functional.normalize(prototype.unsqueeze(0), p=2, dim=1).squeeze(0)
            prototypes[int(class_id)] = prototype.detach().cpu()

        if prototypes:
            self._prototype_by_class = prototypes
            self._seen_classes.update(prototypes)
            self._fitted = True

    def correct_outputs(self, *, outputs: Any, model: nn.Module | None = None, inputs: Any | None = None) -> Any:
        if not self._fitted or not self._prototype_by_class:
            return outputs
        if model is None or inputs is None:
            return outputs
        if not torch.is_tensor(outputs) or outputs.ndim != 2:
            return outputs
        if not torch.is_tensor(inputs):
            return outputs

        try:
            backbone = resolve_backbone_or_raise(model=model)
        except ValueError:
            return outputs

        valid_classes = [
            int(class_id) for class_id in sorted(self._prototype_by_class) if 0 <= int(class_id) < int(outputs.shape[1])
        ]
        if not valid_classes:
            return outputs

        model_device = module_device(model, self.device)
        with preserve_model_mode_after_eval(model):
            with torch.inference_mode():
                feats = backbone(inputs.to(model_device))
        if not torch.is_tensor(feats) or feats.ndim != 2:
            return outputs

        feats = feats.to(device=outputs.device, dtype=outputs.dtype)
        if self.normalize:
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)

        prototype_matrix = torch.stack(
            [
                self._prototype_by_class[class_id].to(device=outputs.device, dtype=outputs.dtype)
                for class_id in valid_classes
            ],
            dim=0,
        )
        if self.normalize:
            prototype_matrix = torch.nn.functional.normalize(prototype_matrix, p=2, dim=1)

        prototype_scores = self.score_scale * feats.matmul(prototype_matrix.T)
        corrected = outputs.clone()
        class_tensor = torch.tensor(valid_classes, device=outputs.device, dtype=torch.long)
        corrected[:, class_tensor] = ((1.0 - self.blend) * outputs[:, class_tensor] + self.blend * prototype_scores)
        return corrected
