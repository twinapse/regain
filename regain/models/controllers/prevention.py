"""
Prevention controllers that modify training to reduce catastrophic forgetting.
"""
import copy
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset

from regain.models.controllers.base import BackboneControllerInterface
from regain.models.controllers.base import PreventionController
from regain.models.controllers.base import TrainingObjectiveControllerInterface
from regain.models.controllers.utils import resolve_backbone_or_raise
from regain.models.normalization import ContinualNormalization4
from regain.models.normalization import ContinualNormalization8
from regain.models.normalization import ContinualNormalization16
from regain.models.normalization import ContinualNormalization32
from regain.models.normalization import ContinualNormalization64
from regain.models.normalization import replace_batchnorm2d
from regain.models.normalization import TaskBalancedBatchNorm
from regain.utils import get_targets
from regain.utils import module_device

__all__ = [
    'ContinualNormalizationController',
    'TaskBalancedBatchNormController',
    'BaCEController',
]


class ContinualNormalizationController(PreventionController, BackboneControllerInterface):
    """
    Apply Continual Normalization (CN) by replacing BatchNorm2d layers.
    """

    def __init__(
        self,
        *,
        train_batch_size: int | None = None,
        replay_batch_size: int | None = None,
        replay_memory_size: int | None = None,
        groups: int = 32,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ) -> None:
        """
        Initialize the CN controller.

        Args:
            train_batch_size (int): Batch size for current experience training data.
            replay_batch_size (int): Batch size for previous experiences replay data.
            replay_memory_size (int): Total size of the replay memory.
            groups (int): Number of groups for the GroupNorm stage. Common values are 4, 8, 16, 32, and 64.
            eps (float): Numerical stability constant used by the replacement CN layers.
            momentum (float): Running-stat momentum used by the replacement CN layers.
        """
        super().__init__(
            train_batch_size=train_batch_size,
            replay_batch_size=replay_batch_size,
            replay_memory_size=replay_memory_size,
        )

        self.groups = int(groups)
        self.eps = float(eps)
        self.momentum = float(momentum)

    def correct_backbone(self, model: nn.Module) -> None:
        """
        Replace all `BatchNorm2d` modules under `model` with CN layers.

        Args:
            model: Model to modify in-place.

        Returns:
            None.
        """
        cn_cls = self._resolve_cn_variant(self.groups)

        def _factory(target: nn.BatchNorm2d) -> nn.Module:
            return cn_cls(target, eps=self.eps, momentum=self.momentum)

        replace_batchnorm2d(model, _factory)

    @staticmethod
    def _resolve_cn_variant(groups: int) -> type[nn.Module]:
        """
        Resolve a CN variant class by group count.

        Args:
            groups: Number of groups.

        Returns:
            CN layer class.
        """
        variants: dict[int, type[nn.Module]] = {
            4: ContinualNormalization4,
            8: ContinualNormalization8,
            16: ContinualNormalization16,
            32: ContinualNormalization32,
            64: ContinualNormalization64,
        }
        if groups in variants:
            return variants[groups]

        class _DynamicCN(ContinualNormalization32):

            def set_num_groups(self) -> None:
                # This override keeps original notation `setG` semantics via the clearer `num_groups` name.
                self.num_groups = groups

        return _DynamicCN


class TaskBalancedBatchNormController(PreventionController, BackboneControllerInterface):
    """
    Apply Task-Balanced Batch Normalization (TBBN) by replacing BatchNorm2d layers.

    This matches the official behavior:
      - All `BatchNorm2d` modules are replaced with `TaskBalancedBN`
      - At the start of each experience, each layer receives `set_number_of_task(t)` (0-indexed)
    """

    def __init__(
        self,
        *,
        train_batch_size: int | None = None,
        replay_batch_size: int | None = None,
        replay_memory_size: int | None = None,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ) -> None:
        """
        Initialize the TBBN controller.

        Args:
            train_batch_size (int): Batch size for current experience training data.
            replay_batch_size (int): Batch size for previous experiences replay data.
            replay_memory_size (int): Total size of the replay memory.
            eps (float): Numerical stability constant for TBBN layers.
            momentum (float): Running-stat momentum for TBBN layers.
        """
        super().__init__(train_batch_size=train_batch_size,
                         replay_batch_size=replay_batch_size,
                         replay_memory_size=replay_memory_size)

        # `current_batch_size` counts current-task samples per mixed minibatch (`B_c` in TBBN original notation).
        self.current_batch_size = train_batch_size
        # `replay_batch_size` counts replay-buffer samples in that same minibatch (`B_p` in TBBN original notation).
        self.replay_batch_size = replay_batch_size
        self.batch_ratio: int | None = None
        self.eps = float(eps)
        self.momentum = float(momentum)

        self._tbbn_layers: list[TaskBalancedBatchNorm] = []
        self._current_task: int = -1

        if self.current_batch_size is None or self.replay_batch_size is None:
            raise ValueError('TaskBalancedBatchNormController requires train_batch_size and replay_batch_size.')

        self._finalize_partition(
            current_batch_size=self.current_batch_size,
            replay_batch_size=self.replay_batch_size,
        )

    def on_train_experience_begin(self, dataset: Dataset | None) -> None:
        """
        Set the task number at the start of each experience.

        Args:
            dataset: Unused.

        Returns:
            None.
        """
        del dataset
        self._current_task += 1
        for layer in self._tbbn_layers:
            layer.set_number_of_task(self._current_task)

    @classmethod
    def requires_replay(cls) -> bool:
        return True

    def correct_backbone(self, model: nn.Module) -> None:
        """
        Replace all BatchNorm2d modules with TaskBalancedBN, preserving parameters and buffers.

        Args:
            model: Model to modify in-place.

        Returns:
            None.
        """
        if self.current_batch_size is None or self.replay_batch_size is None:
            raise ValueError(
                'TaskBalancedBatchNormController requires current_batch_size and replay_batch_size to be configured.')

        current_batch_size = int(self.current_batch_size)
        replay_batch_size = int(self.replay_batch_size)

        def _factory(target: nn.BatchNorm2d) -> nn.Module:
            layer = TaskBalancedBatchNorm(
                num_features=int(target.num_features),
                eps=float(self.eps),
                momentum=float(self.momentum),
                affine=bool(target.affine),
                track_running_stats=bool(target.track_running_stats),
                batch_ratio=int(self.batch_ratio) if self.batch_ratio is not None else None,
                current_batch_size=int(current_batch_size),
                replay_batch_size=int(replay_batch_size),
            )

            device = target.running_mean.device
            layer = layer.to(device=device, dtype=target.running_mean.dtype)

            with torch.no_grad():
                layer.running_mean.copy_(target.running_mean)
                layer.running_var.copy_(target.running_var)
                if layer.num_batches_tracked is not None and target.num_batches_tracked is not None:
                    layer.num_batches_tracked.copy_(target.num_batches_tracked)
                if target.affine and target.weight is not None and target.bias is not None:
                    layer.weight.copy_(target.weight)
                    layer.bias.copy_(target.bias)

            self._tbbn_layers.append(layer)
            return layer

        replace_batchnorm2d(model, _factory)

    def _finalize_partition(self, *, current_batch_size: int, replay_batch_size: int) -> None:
        """
        Validate and store the batch partition.

        Args:
            current_batch_size: Current-task batch size.
            replay_batch_size: Replay batch size.
        """
        # `resolved_*` are validated integer forms of the configured batch split.
        resolved_current_batch_size = int(current_batch_size)
        resolved_replay_batch_size = int(replay_batch_size)
        if resolved_current_batch_size <= 0 or resolved_replay_batch_size <= 0:
            raise ValueError('current_batch_size and replay_batch_size must be positive integers.')

        implied_ratio = resolved_current_batch_size // resolved_replay_batch_size
        if implied_ratio * resolved_replay_batch_size != resolved_current_batch_size:
            raise ValueError('current_batch_size must be an integer multiple of replay_batch_size for TBBN.')

        if self.batch_ratio is None:
            self.batch_ratio = implied_ratio
        elif self.batch_ratio <= 0 or self.batch_ratio != implied_ratio:
            raise ValueError(f'Inconsistent batch partition: batch_ratio={self.batch_ratio}, '
                             f'but current_batch_size={resolved_current_batch_size} and '
                             f'replay_batch_size={resolved_replay_batch_size} imply {implied_ratio}.')

        self.current_batch_size = resolved_current_batch_size
        self.replay_batch_size = resolved_replay_batch_size


class BaCEController(PreventionController, TrainingObjectiveControllerInterface):
    """
    BaCE: Balancing the Mutual Causal Effects in Class-Incremental Learning.

    This controller modifies the *training objective* to implement:
      - Effect_new: joint-score loss using KNNs in the teacher's feature space.
      - Effect_old: KL distillation on *old-class scores* between student and teacher on current-task data.
      - With replay/buffer: adds CE on buffer samples + MSE distillation on old-class logits (DER++-style).

    Notes:
        - Assumes class-incremental settings with disjoint classes per experience.
        - Assumes the model exposes a backbone via `.backbone` (preferred) or `.encoder`.
        - Updates the teacher model via EMA each training epoch as described in the paper.
    """

    def __init__(
        self,
        *,
        train_batch_size: int | None = None,
        replay_batch_size: int | None = None,
        replay_memory_size: int | None = None,
        num_neighbors: int = 5,
        w0: float = 0.95,
        beta: float = 0.9,
        alpha_no_buffer: float = 5.0,
        alpha_with_buffer: float = 1.0,
        replay_ce_weight: float = 1.0,
        replay_mse_weight: float = 1.0,
        bank_batch_size: int = 256,
        bank_max_samples: int | None = None,
        dist_eps: float = 1e-8,
        self_exclude_eps: float = 1e-12,
    ) -> None:
        """
        Initialize the BaCE controller.

        Args:
            train_batch_size (int): Batch size for current experience training data.
            replay_batch_size (int): Batch size for previous experiences replay data.
            replay_memory_size (int): Total size of the replay memory.
            num_neighbors (int): Number of KNN neighbors to use for joint-score computation.
            w0 (float): Weight for self-scores in joint-score computation.
            beta (float): EMA momentum for teacher model updates.
            alpha_no_buffer (float): Weight for Effect_old KL when no buffer samples are present.
            alpha_with_buffer (float): Weight for Effect_old KL when buffer samples are present.
            replay_ce_weight (float): Weight for CE loss on buffer samples.
            replay_mse_weight (float): Weight for MSE distillation on old-class logits for buffer samples.
            bank_batch_size (int): Batch size for building the KNN feature bank.
            bank_max_samples (int | None): Maximum number of samples to use for the KNN bank (None = all).
            dist_eps (float): Small constant for numerical stability in distance computations.
            self_exclude_eps (float): Distance threshold to exclude self-matches in KNN.
        """
        super().__init__(train_batch_size=train_batch_size,
                         replay_batch_size=replay_batch_size,
                         replay_memory_size=replay_memory_size)

        self.num_neighbors = int(num_neighbors)
        self.w0 = float(w0)
        self.beta = float(beta)
        self.alpha_no_buffer = float(alpha_no_buffer)
        self.alpha_with_buffer = float(alpha_with_buffer)
        self.replay_ce_weight = float(replay_ce_weight)
        self.replay_mse_weight = float(replay_mse_weight)
        self.bank_batch_size = int(bank_batch_size)
        self.bank_max_samples = int(bank_max_samples) if bank_max_samples is not None else None
        self.dist_eps = float(dist_eps)
        self.self_exclude_eps = float(self_exclude_eps)

        self._dataset: Dataset | None = None
        self._current_classes: list[int] = []
        self._old_classes: list[int] = []
        self._seen_classes: set[int] = set()

        self._teacher: nn.Module | None = None
        self._needs_teacher_init: bool = False

        self._bank_inputs: torch.Tensor | None = None
        self._bank_feats: torch.Tensor | None = None
        self._bank_sq_norm: torch.Tensor | None = None

    def on_train_experience_begin(self, dataset: Dataset | None) -> None:
        """
        Track experience classes, reset teacher/bank state for the new experience.

        Args:
            dataset: Dataset for the current experience.

        Returns:
            None.
        """
        self._dataset = dataset
        if dataset is None:
            self._current_classes = []
            self._old_classes = sorted(self._seen_classes)
        else:
            targets = get_targets(dataset)
            current = sorted({int(x) for x in np.unique(targets).tolist()})
            self._current_classes = current
            self._old_classes = sorted(self._seen_classes.difference(current))
            self._seen_classes.update(current)

        self._teacher = None
        self._needs_teacher_init = True
        self._bank_inputs = None
        self._bank_feats = None
        self._bank_sq_norm = None

    def on_train_epoch_begin(self, model: nn.Module) -> None:
        """
        Ensure teacher is initialized for this experience and (re)build the KNN feature bank.

        Args:
            model: The model being trained.

        Returns:
            None.
        """
        # No old classes => no need for teacher/bank yet
        if not self._old_classes:
            return

        if self._needs_teacher_init:
            self._teacher = copy.deepcopy(model).eval()
            for p in self._teacher.parameters():
                p.requires_grad = False
            self._needs_teacher_init = False

        if self._teacher is None:
            return
        if self._dataset is None:
            return
        self._build_knn_bank(model=model)

    def on_train_epoch_end(self, model: nn.Module) -> None:
        """
        EMA update the teacher model at the end of each epoch.

        Args:
            model: The model being trained.

        Returns:
            None.
        """
        if self._teacher is None:
            return
        self._ema_update_teacher(student=model)

    def correct_training_objective(
        self,
        *,
        loss: Any,
        outputs: Any,
        model: nn.Module,
        inputs: Any,
        targets: Any,
    ) -> torch.Tensor:
        """
        Replace the strategy's base loss with the BaCE objective.

        Args:
            loss: Base loss (ignored when BaCE is active; used as fallback).
            outputs: Student logits for the minibatch.
            model: Student model being trained.
            inputs: Minibatch inputs.
            targets: Minibatch labels.

        Returns:
            Scalar loss tensor to backpropagate.
        """

        if not torch.is_tensor(outputs) or not torch.is_tensor(inputs) or not torch.is_tensor(targets):
            if torch.is_tensor(loss):
                return loss
            raise ValueError('BaCEController requires tensor inputs/targets/outputs (or a tensor base loss).')

        # First experience: no old classes => don't use BaCE (avoid random KNN coupling).
        if not self._old_classes:
            if torch.is_tensor(loss):
                return loss
            return F.cross_entropy(outputs, targets)

        if self._teacher is None:
            if torch.is_tensor(loss):
                return loss
            return F.cross_entropy(outputs, targets)

        device = outputs.device

        current_mask = self._mask_from_classes(targets=targets, classes=self._current_classes)
        buffer_mask = (~current_mask) if self._old_classes else torch.zeros_like(current_mask)

        total_loss = torch.zeros((), device=device)

        # Effect_new: joint-score KNN loss on current-task samples
        if torch.any(current_mask):
            effect_new = self._effect_new_loss(
                model=model,
                student_logits=outputs,
                inputs=inputs,
                targets=targets,
                current_mask=current_mask,
            )
            total_loss = total_loss + effect_new
        else:
            # If no current samples are present, we cannot compute Effect_new; fall back to base CE on the batch.
            total_loss = total_loss + F.cross_entropy(outputs, targets)

        # Effect_old: KL distillation on old-class scores for current-task samples
        if self._old_classes and torch.any(current_mask):
            alpha = self.alpha_with_buffer if torch.any(buffer_mask) else self.alpha_no_buffer
            effect_old = self._effect_old_kl(
                student_logits=outputs,
                inputs=inputs,
                current_mask=current_mask,
                old_classes=self._old_classes,
            )
            total_loss = total_loss + alpha * effect_old

        # Replay enhancement (DER++-style): CE on buffer + MSE distillation on old logits
        if self._old_classes and torch.any(buffer_mask):
            buf_logits = outputs[buffer_mask]
            buf_targets = targets[buffer_mask]
            total_loss = total_loss + self.replay_ce_weight * F.cross_entropy(buf_logits, buf_targets)

            with torch.inference_mode():
                teacher_logits = self._teacher(inputs[buffer_mask].to(device))
            old_idx = torch.tensor(self._old_classes, device=device, dtype=torch.long)
            student_old = buf_logits.index_select(dim=1, index=old_idx)
            teacher_old = teacher_logits.index_select(dim=1, index=old_idx)
            total_loss = total_loss + self.replay_mse_weight * F.mse_loss(student_old, teacher_old)

        return total_loss

    ####################
    # Internal helpers #
    ####################

    @staticmethod
    def _mask_from_classes(targets: torch.Tensor, classes: list[int]) -> torch.Tensor:
        """
        Build a boolean mask selecting targets belonging to a set of class IDs.

        Args:
            targets: Label tensor of shape (B,).
            classes: Class IDs to select.

        Returns:
            Boolean mask of shape (B,).
        """
        if not classes:
            return torch.zeros_like(targets, dtype=torch.bool)
        cls = torch.tensor(classes, device=targets.device, dtype=targets.dtype)
        return (targets.unsqueeze(1) == cls.unsqueeze(0)).any(dim=1)

    def _build_knn_bank(self, *, model: nn.Module) -> None:
        """
        Build the teacher-feature bank over the current experience dataset.

        Args:
            model: Student model (used for device resolution).

        Returns:
            None.
        """
        if self._dataset is None or self._teacher is None:
            return

        teacher_backbone = resolve_backbone_or_raise(
            model=self._teacher,
            error_message='BaCEController requires the model to expose a `.backbone` (or `.encoder`) module.',
        )
        model_device = module_device(model, fallback='cpu')

        n = len(self._dataset)
        indices = list(range(n))

        if self.bank_max_samples is not None and n > self.bank_max_samples:
            indices = np.random.choice(np.asarray(indices), size=self.bank_max_samples, replace=False).tolist()
            indices.sort()

        subset = Subset(self._dataset, indices)
        loader = DataLoader(subset, batch_size=self.bank_batch_size, shuffle=False)

        inputs_list: list[torch.Tensor] = []
        feats_list: list[torch.Tensor] = []
        with torch.inference_mode():
            teacher_backbone.eval()
            for batch in loader:
                x = batch[0]
                if not torch.is_tensor(x):
                    raise ValueError('BaCEController expects the dataset to yield tensor inputs.')
                x = x.to(model_device)
                feats = teacher_backbone(x)  # pylint: disable=not-callable
                if not torch.is_tensor(feats) or feats.ndim != 2:
                    raise ValueError('BaCEController backbone must return 2D features (B, D).')
                feats_list.append(feats)
                inputs_list.append(x)

        if not feats_list:
            self._bank_inputs = None
            self._bank_feats = None
            self._bank_sq_norm = None
            return

        bank_inputs = torch.cat(inputs_list, dim=0).cpu()
        self._bank_inputs = bank_inputs

        bank_feats = torch.cat(feats_list, dim=0)
        self._bank_feats = bank_feats
        self._bank_sq_norm = (bank_feats * bank_feats).sum(dim=1)

    def _ema_update_teacher(self, *, student: nn.Module) -> None:
        """
        EMA-update teacher parameters and buffers toward the student model.

        Args:
            student: Student model being trained.

        Returns:
            None.
        """
        if self._teacher is None:
            return

        beta = self.beta

        teacher_state = dict(self._teacher.named_parameters())
        student_state = dict(student.named_parameters())

        with torch.no_grad():
            for name, t_param in teacher_state.items():
                s_param = student_state.get(name)
                if s_param is None:
                    continue
                t_param.mul_(beta).add_(s_param.detach(), alpha=1.0 - beta)

            # Buffers (e.g., BatchNorm running stats): copy directly (more stable than EMA for many buffers).
            teacher_buffers = dict(self._teacher.named_buffers())
            student_buffers = dict(student.named_buffers())
            for name, t_buf in teacher_buffers.items():
                s_buf = student_buffers.get(name)
                if s_buf is None:
                    continue
                if torch.is_tensor(t_buf) and torch.is_tensor(s_buf) and t_buf.shape == s_buf.shape:
                    t_buf.copy_(s_buf)

    def _effect_old_kl(
        self,
        *,
        student_logits: torch.Tensor,
        inputs: torch.Tensor,
        current_mask: torch.Tensor,
        old_classes: list[int],
    ) -> torch.Tensor:
        """
        Effect_old: KL distillation between student and teacher *old-class scores* on current-task samples.

        Args:
            student_logits: Student logits for the minibatch.
            inputs: Minibatch inputs.
            current_mask: Boolean mask selecting current-task samples.
            old_classes: List of old class IDs.

        Returns:
            Scalar KL loss.
        """
        device = student_logits.device
        old_idx = torch.tensor(old_classes, device=device, dtype=torch.long)

        cur_inputs = inputs[current_mask].to(device)
        cur_student = student_logits[current_mask].index_select(dim=1, index=old_idx)

        with torch.inference_mode():
            teacher_logits = self._teacher(cur_inputs)
            cur_teacher = teacher_logits.index_select(dim=1, index=old_idx)

        # Scores over old classes (normalized on the old-class subset).
        student_scores = F.softmax(cur_student, dim=1)
        teacher_scores = F.softmax(cur_teacher, dim=1)

        student_log = torch.log(student_scores + self.dist_eps)
        return F.kl_div(student_log, teacher_scores, reduction='batchmean')

    def _effect_new_loss(
        self,
        *,
        model: nn.Module,
        student_logits: torch.Tensor,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        current_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Effect_new: joint-score CE using KNNs in teacher feature space.

        Args:
            model: Student model.
            student_logits: Student logits for the minibatch.
            inputs: Minibatch inputs.
            targets: Minibatch labels.
            current_mask: Boolean mask selecting current-task samples.

        Returns:
            Scalar cross-entropy (negative log-likelihood) on joint scores.
        """
        if self.num_neighbors <= 0 or self.w0 >= 1.0:
            # Degenerates to standard CE on current samples when K=0 or W0=1.
            return F.cross_entropy(student_logits[current_mask], targets[current_mask])

        if self._bank_inputs is None or self._bank_feats is None or self._bank_sq_norm is None:
            # If KNN bank isn't available, fall back to standard CE on current samples.
            return F.cross_entropy(student_logits[current_mask], targets[current_mask])

        device = student_logits.device
        w0 = float(self.w0)

        cur_inputs = inputs[current_mask].to(device)
        cur_targets = targets[current_mask].to(device)
        cur_logits = student_logits[current_mask]

        # Self scores (student)
        self_scores = F.softmax(cur_logits, dim=1)  # (B, C)
        joint_scores = self_scores * w0

        # Teacher features for queries
        teacher_backbone = resolve_backbone_or_raise(
            model=self._teacher,
            error_message='BaCEController requires the model to expose a `.backbone` (or `.encoder`) module.',
        )
        with torch.inference_mode():
            query_feats = teacher_backbone(cur_inputs)  # pylint: disable=not-callable
        if not torch.is_tensor(query_feats) or query_feats.ndim != 2:
            return F.cross_entropy(cur_logits, cur_targets)

        # KNN distances (squared L2) in teacher feature space
        bank_feats = self._bank_feats.to(device)
        bank_sq = self._bank_sq_norm.to(device)
        query_sq = (query_feats * query_feats).sum(dim=1)  # (B,)
        dist2 = query_sq.unsqueeze(1) + bank_sq.unsqueeze(0) - 2.0 * (query_feats @ bank_feats.t())  # (B, N)
        dist2 = torch.clamp(dist2, min=0.0)

        k = min(int(self.num_neighbors) + 1, dist2.shape[1])
        top_dist, top_idx = torch.topk(dist2, k=k, dim=1, largest=False)

        # Exclude self-matches / zero-distance duplicates.
        neighbor_idx: list[list[int]] = []
        neighbor_w: list[list[float]] = []

        top_dist_np = top_dist.detach().cpu().numpy()
        top_idx_np = top_idx.detach().cpu().numpy()

        for i in range(top_idx_np.shape[0]):
            pairs: list[tuple[int, float]] = []
            for j in range(top_idx_np.shape[1]):
                d = float(top_dist_np[i, j])
                if d <= self.self_exclude_eps:
                    continue
                bank_pos = int(top_idx_np[i, j])
                pairs.append((bank_pos, d))
            pairs = pairs[:self.num_neighbors]

            if not pairs:
                neighbor_idx.append([])
                neighbor_w.append([])
                continue

            inv = np.asarray([1.0 / (d + self.dist_eps) for _, d in pairs], dtype=np.float64)
            inv_sum = float(inv.sum())
            if inv_sum <= 0.0:
                weights = np.full_like(inv, fill_value=(1.0 - w0) / float(len(inv)))
            else:
                weights = (1.0 - w0) * (inv / inv_sum)

            neighbor_idx.append([p for p, _ in pairs])
            neighbor_w.append(weights.tolist())

        # Collect neighbor inputs and weights for a single forward pass (keeps gradients for neighbor predictions).
        flat_inputs: list[torch.Tensor] = []
        flat_q_index: list[int] = []
        flat_weight: list[float] = []

        for qi, banks in enumerate(neighbor_idx):
            if not banks:
                continue
            wts = neighbor_w[qi]
            for bi, w in zip(banks, wts):
                x = self._bank_inputs[int(bi)]
                if not torch.is_tensor(x):
                    raise ValueError('BaCEController expects the dataset to yield tensor inputs.')
                flat_inputs.append(x)
                flat_q_index.append(qi)
                flat_weight.append(float(w))

        if flat_inputs:
            neigh_x = torch.stack(flat_inputs, dim=0).to(device)
            neigh_logits = model(neigh_x)
            neigh_scores = F.softmax(neigh_logits, dim=1)

            q_idx = torch.tensor(flat_q_index, device=device, dtype=torch.long)
            w_vec = torch.tensor(flat_weight, device=device, dtype=neigh_scores.dtype).unsqueeze(1)

            contrib = neigh_scores * w_vec
            joint_scores = joint_scores.index_add(0, q_idx, contrib)

        # NLL on joint scores (cross-entropy with one-hot target).
        p = joint_scores[torch.arange(joint_scores.shape[0], device=device), cur_targets]
        return (-torch.log(p + self.dist_eps)).mean()
