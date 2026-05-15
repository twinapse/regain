"""
Logit-space calibration controllers.

Controllers in this module operate on classifier logits to correct systematic score distortions such as
global/partial bias and class-incremental calibration drift.

They represent the lowest-capacity points in the controller family.
"""
import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset

from regain.models.controllers.base import RepairController
from regain.models.controllers.modules import BiasLayer
from regain.models.controllers.repair.common import build_repair_dataloader
from regain.models.controllers.repair.common import build_sgd_optimizer_and_scheduler
from regain.models.controllers.repair.common import fit_repair_controller
from regain.models.controllers.repair.common import model_logits
from regain.models.controllers.repair.common import prepare_repair_fit_context
from regain.models.controllers.repair.common import resolve_classifier_weight
from regain.utils import module_device
from regain.utils import preserve_model_mode_after_eval

__all__ = [
    'BiCController',
    'LogitBiasController',
    'IL2MController',
    'TCILLiteController',
    'TemperatureScalingController',
    'WeightAligningController',
]


class LogitBiasController(RepairController):
    """
    Additive per-class logit bias fitted using repair data.

    The controller learns a vector `b` such that, at evaluation time:

        logits' = logits + b

    Args:
        lr: Learning rate for fitting.
        device: Device for controller parameters and fitting data.
    """

    def __init__(
        self,
        lr: float,
        device: str | None = None,
    ) -> None:
        super().__init__()

        # Hyperparameters
        self.lr = float(lr)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # State
        self.bias = nn.Parameter(torch.zeros(0))  # Lazily expanded as new classes appear
        self.to(self.device)

    def initialize_parameters(self, *, model: nn.Module, sample_inputs: Any | None = None) -> None:
        """
        Initialize the bias vector size from the model's output width.

        Args:
            model (nn.Module): Model used to infer logits width.
            sample_inputs (Any | None): Representative input batch for probing.

        Returns:
            None.
        """
        if sample_inputs is None:
            return
        if not torch.is_tensor(sample_inputs):
            sample_inputs = torch.as_tensor(sample_inputs)

        model_device = module_device(model, self.device)
        self.to(model_device)
        x = sample_inputs.to(model_device)

        with preserve_model_mode_after_eval(model):
            with torch.inference_mode():
                logits = model(x)

        if torch.is_tensor(logits) and logits.ndim == 2:
            self._ensure_num_classes(int(logits.shape[1]))

    def _ensure_num_classes(self, num_classes: int) -> None:
        """
        Ensure the bias vector has at least `num_classes` elements.

        Args:
            num_classes (int): Target number of classes.

        Returns:
            None.
        """
        if num_classes <= self.bias.numel():
            return

        device = self.bias.device
        dtype = self.bias.dtype
        old_n = int(self.bias.numel())

        new_bias = torch.zeros(num_classes, device=device, dtype=dtype)
        if old_n > 0:
            with torch.no_grad():
                new_bias[:old_n].copy_(self.bias.detach())

        self.bias = nn.Parameter(new_bias)

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
        Fit the bias vector using observed repair data.

        Args:
            model (nn.Module): Model used to compute logits for fitting.
            repair_dataset (Dataset | None): Repair dataset for fitting.
            new_classes (list[int]): Newly introduced classes.
            num_epochs (int): Number of epochs used for fitting.
            batch_size (int): Batch size used for fitting.

        Returns:
            None.
        """
        def _ensure_initialized(*, model: nn.Module, device: torch.device, sample_inputs: torch.Tensor) -> None:
            if new_classes:
                self._ensure_num_classes(max(int(c) for c in new_classes) + 1)
            with torch.inference_mode():
                logits = model(sample_inputs.to(device))
            if torch.is_tensor(logits) and logits.ndim == 2:
                self._ensure_num_classes(int(logits.shape[1]))

        fit_context = prepare_repair_fit_context(
            controller=self,
            model=model,
            repair_dataset=repair_dataset,
            batch_size=batch_size,
            device=self.device,
            ensure_initialized_fn=_ensure_initialized,
        )
        if fit_context is None:
            return

        dataloader, model_device, trainable_params = fit_context

        optimizer, scheduler = build_sgd_optimizer_and_scheduler(
            params=trainable_params,
            lr=self.lr,
            momentum=0.0,
            weight_decay=0.0,
            lr_milestones=None,
            lr_gamma=1.0,
        )
        criterion = CrossEntropyLoss()

        def _forward(x: torch.Tensor) -> torch.Tensor:
            with torch.inference_mode():
                logits = model(x)

            if not torch.is_tensor(logits) or logits.ndim != 2:
                return logits

            self._ensure_num_classes(int(logits.shape[1]))

            bias = self.bias
            if bias.device != x.device:
                bias = bias.to(device=x.device)
            if int(bias.numel()) > int(logits.shape[1]):
                bias = bias[: int(logits.shape[1])]

            return logits + bias

        fit_repair_controller(
            controller=self,
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

    def correct_outputs(self, *, outputs: Any, model: nn.Module | None = None, inputs: Any | None = None) -> Any:
        """
        Add the learned bias to logits during evaluation.

        Args:
            outputs (Any): Logits shaped `(batch, num_classes)`.
            model (nn.Module | None): Unused.
            inputs (Any | None): Unused.

        Returns:
            Any: Adjusted logits.
        """
        if not torch.is_tensor(outputs):
            return outputs
        if outputs.ndim != 2:
            return outputs

        self._ensure_num_classes(int(outputs.shape[1]))
        bias = self.bias
        if int(bias.numel()) > int(outputs.shape[1]):
            bias = bias[: int(outputs.shape[1])]

        return outputs + bias.to(outputs.device)


class WeightAligningController(RepairController):
    """
    Weight Aligning baseline for class-incremental classifier-head correction.

    The controller estimates the average classifier-weight norm ratio between old and newly introduced classes and
    applies the resulting scalar to the logits of the new classes. This is a near-zero-cost post-hoc repair baseline:
    it reads frozen classifier weights and does not optimize on repair minibatches.

    Args:
        min_scale (float): Lower clamp for class logit scales.
        max_scale (float): Upper clamp for class logit scales.
    """

    def __init__(
        self,
        *,
        min_scale: float = 0.05,
        max_scale: float = 20.0,
    ) -> None:
        super().__init__()
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        if self.min_scale <= 0.0:
            raise ValueError('`min_scale` must be positive.')
        if self.max_scale < self.min_scale:
            raise ValueError('`max_scale` must be greater than or equal to `min_scale`.')

        self._seen_classes: set[int] = set()
        self._class_scales: dict[int, float] = {}

    @classmethod
    def requires_per_experience_fitting(cls) -> bool:
        return True

    def fit_on_repair_data(
        self,
        *,
        model: nn.Module,
        repair_dataset: Dataset | None,
        new_classes: list[int],
        num_epochs: int,
        batch_size: int,
    ) -> None:
        del repair_dataset, num_epochs, batch_size

        cur_classes = sorted({int(class_id) for class_id in new_classes})
        old_classes = sorted(self._seen_classes.difference(cur_classes))
        self._seen_classes.update(cur_classes)
        if not old_classes or not cur_classes:
            return

        weight = resolve_classifier_weight(model)
        if weight is None or weight.ndim != 2:
            return

        max_class = int(weight.shape[0]) - 1
        old_idx = [class_id for class_id in old_classes if 0 <= class_id <= max_class]
        cur_idx = [class_id for class_id in cur_classes if 0 <= class_id <= max_class]
        if not old_idx or not cur_idx:
            return

        norms = torch.linalg.vector_norm(weight.detach().float(), ord=2, dim=1)
        old_norm = float(norms[old_idx].mean().item())
        cur_norm = float(norms[cur_idx].mean().item())
        if cur_norm <= 0.0:
            return

        scale = max(self.min_scale, min(self.max_scale, old_norm / cur_norm))
        for class_id in cur_idx:
            self._class_scales[int(class_id)] = float(scale)

    def correct_outputs(self, *, outputs: Any, model: nn.Module | None = None, inputs: Any | None = None) -> Any:
        del model, inputs
        if not torch.is_tensor(outputs) or outputs.ndim != 2:
            return outputs
        if not self._class_scales:
            return outputs

        corrected = outputs.clone()
        for class_id, scale in self._class_scales.items():
            if 0 <= int(class_id) < int(corrected.shape[1]):
                corrected[:, int(class_id)] = corrected[:, int(class_id)] * float(scale)
        return corrected


class TemperatureScalingController(RepairController):
    """
    Scalar temperature-scaling baseline optimized with frozen logits and repair-set NLL.

    Args:
        min_temperature (float): Lower clamp for the fitted temperature.
        max_temperature (float): Upper clamp for the fitted temperature.
        device (str | None): Device for collecting frozen repair logits.
    """

    def __init__(
        self,
        *,
        min_temperature: float = 0.05,
        max_temperature: float = 10.0,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.min_temperature = float(min_temperature)
        self.max_temperature = float(max_temperature)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        if self.min_temperature <= 0.0:
            raise ValueError('`min_temperature` must be positive.')
        if self.max_temperature < self.min_temperature:
            raise ValueError('`max_temperature` must be greater than or equal to `min_temperature`.')

        self.temperature = torch.tensor(1.0, dtype=torch.float32)

    def fit_on_repair_data(
        self,
        *,
        model: nn.Module,
        repair_dataset: Dataset | None,
        new_classes: list[int],
        num_epochs: int,
        batch_size: int,
    ) -> None:
        del new_classes
        logits, targets = self._collect_logits_and_targets(
            model=model,
            repair_dataset=repair_dataset,
            batch_size=batch_size,
        )
        if logits is None or targets is None:
            return

        max_iter = num_epochs * math.ceil(len(repair_dataset) / batch_size)

        self.temperature = self._fit_temperature(logits=logits, targets=targets, max_iter=max_iter).detach().cpu()

    def correct_outputs(self, *, outputs: Any, model: nn.Module | None = None, inputs: Any | None = None) -> Any:
        del model, inputs
        if not torch.is_tensor(outputs) or outputs.ndim != 2:
            return outputs
        temperature = self.temperature.to(device=outputs.device, dtype=outputs.dtype).clamp(
            min=self.min_temperature,
            max=self.max_temperature,
        )
        return outputs / temperature

    def _collect_logits_and_targets(
        self,
        *,
        model: nn.Module,
        repair_dataset: Dataset | None,
        batch_size: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        dataloader = build_repair_dataloader(
            repair_dataset=repair_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        if dataloader is None:
            return None, None

        model_device = module_device(model, self.device)
        logits_chunks: list[torch.Tensor] = []
        target_chunks: list[torch.Tensor] = []
        for batch in dataloader:
            x, y, *_ = batch
            if not torch.is_tensor(x):
                x = torch.as_tensor(x)
            if not torch.is_tensor(y):
                y = torch.as_tensor(y)
            logits = model_logits(model=model, inputs=x.to(model_device))
            if not torch.is_tensor(logits) or logits.ndim != 2:
                return None, None
            logits_chunks.append(logits.detach().to(device='cpu', dtype=torch.float32))
            target_chunks.append(y.reshape(-1).detach().to(device='cpu', dtype=torch.long))

        if not logits_chunks:
            return None, None
        return torch.cat(logits_chunks, dim=0), torch.cat(target_chunks, dim=0)

    def _fit_temperature(self, *, logits: torch.Tensor, targets: torch.Tensor, max_iter: int) -> torch.Tensor:
        log_min = float(np.log(self.min_temperature))
        log_max = float(np.log(self.max_temperature))
        log_temperature = torch.zeros((), dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [log_temperature],
            lr=0.1,
            max_iter=max_iter,
            line_search_fn='strong_wolfe',
        )
        criterion = CrossEntropyLoss()

        def _closure() -> torch.Tensor:
            optimizer.zero_grad()
            temperature = torch.exp(log_temperature.clamp(min=log_min, max=log_max))
            loss = criterion(logits / temperature, targets)
            loss.backward()
            return loss

        optimizer.step(_closure)
        return torch.exp(log_temperature.detach().clamp(min=log_min, max=log_max))


class TCILLiteController(TemperatureScalingController):
    """
    T-CIL-Lite temperature baseline.

    This is a lightweight approximation of T-CIL rather than a faithful reproduction: it fits one temperature per
    observed experience class group and applies the group temperature to that group's logit columns.

    Args:
        min_temperature (float): Lower clamp for the fitted temperature.
        max_temperature (float): Upper clamp for the fitted temperature.
        device (str | None): Device for collecting frozen repair logits.
    """

    def __init__(
        self,
        *,
        min_temperature: float = 0.05,
        max_temperature: float = 10.0,
        device: str | None = None,
    ) -> None:
        super().__init__(
            min_temperature=min_temperature,
            max_temperature=max_temperature,
            device=device,
        )
        self._class_groups: list[list[int]] = []
        self._group_temperatures: list[float] = []

    @classmethod
    def requires_per_experience_fitting(cls) -> bool:
        return True

    def fit_on_repair_data(
        self,
        *,
        model: nn.Module,
        repair_dataset: Dataset | None,
        new_classes: list[int],
        num_epochs: int,
        batch_size: int,
    ) -> None:
        cur_group = sorted({int(class_id) for class_id in new_classes})
        if cur_group and cur_group not in self._class_groups:
            self._class_groups.append(cur_group)

        logits, targets = self._collect_logits_and_targets(
            model=model,
            repair_dataset=repair_dataset,
            batch_size=batch_size,
        )
        if logits is None or targets is None:
            return
        if not self._class_groups:
            return

        max_iter = num_epochs * math.ceil(len(repair_dataset) / batch_size)

        self._group_temperatures = self._fit_group_temperatures(
            logits=logits,
            targets=targets,
            max_iter=max_iter,
        )

    def correct_outputs(self, *, outputs: Any, model: nn.Module | None = None, inputs: Any | None = None) -> Any:
        del model, inputs
        if not torch.is_tensor(outputs) or outputs.ndim != 2:
            return outputs
        if not self._class_groups or not self._group_temperatures:
            return outputs

        corrected = outputs.clone()
        for class_group, temperature in zip(self._class_groups, self._group_temperatures, strict=False):
            valid_classes = [class_id for class_id in class_group if 0 <= int(class_id) < int(outputs.shape[1])]
            if not valid_classes:
                continue
            temp_tensor = torch.tensor(
                float(temperature),
                device=outputs.device,
                dtype=outputs.dtype,
            ).clamp(min=self.min_temperature, max=self.max_temperature)
            corrected[:, valid_classes] = corrected[:, valid_classes] / temp_tensor
        return corrected

    def _fit_group_temperatures(self, *, logits: torch.Tensor, targets: torch.Tensor, max_iter: int) -> list[float]:
        log_min = float(np.log(self.min_temperature))
        log_max = float(np.log(self.max_temperature))
        log_temperatures = torch.zeros(len(self._class_groups), dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [log_temperatures],
            lr=0.1,
            max_iter=max_iter,
            line_search_fn='strong_wolfe',
        )
        criterion = CrossEntropyLoss()

        class_to_group: dict[int, int] = {}
        for group_idx, class_group in enumerate(self._class_groups):
            for class_id in class_group:
                class_to_group[int(class_id)] = int(group_idx)

        column_group_idxs = torch.full((int(logits.shape[1]),), -1, dtype=torch.long)
        for class_id, group_idx in class_to_group.items():
            if 0 <= class_id < int(column_group_idxs.numel()):
                column_group_idxs[class_id] = int(group_idx)

        def _closure() -> torch.Tensor:
            optimizer.zero_grad()
            temperatures = torch.exp(log_temperatures.clamp(min=log_min, max=log_max))
            scaled_logits = logits.clone()
            for group_idx in range(len(self._class_groups)):
                mask = column_group_idxs.eq(group_idx)
                if bool(mask.any()):
                    scaled_logits[:, mask] = scaled_logits[:, mask] / temperatures[group_idx]
            loss = criterion(scaled_logits, targets)
            loss.backward()
            return loss

        optimizer.step(_closure)
        fitted = torch.exp(log_temperatures.detach().clamp(min=log_min, max=log_max))
        return [float(value) for value in fitted.tolist()]


class BiCController(RepairController):
    """
    Bias Correction (BiC), adapted from Avalanche's `BiCPlugin` stage-2 procedure.

    After each experience (except the first), it trains a scalar (alpha, beta) BiasLayer applied to the
    *current experience classes* only. The bias layer is trained on the repair dataset that aggregates
    all seen classes.

    Notes:
      - The base model is kept in eval() during bias fitting (BN stats frozen).
      - A single bias layer is maintained and overwritten after each experience (except the first).

    Args:
        lr: Learning rate for bias layer fitting.
        lr_milestones: Learning rate milestones for bias layer fitting.
        lr_gamma: Learning rate decay factor for bias layer fitting.
        l2_beta: L2 regularization factor for beta parameter.
        device: Device for fitting.
    """

    def __init__(
        self,
        *,
        lr: float = 0.1,
        lr_milestones: tuple[int, ...] = (50, 100, 150),
        lr_gamma: float = 0.1,
        l2_beta: float = 0.1,  # Avalanche uses 0.1 * beta^2 / 2
        device: str | None = None,
    ) -> None:
        super().__init__()

        # Hyperparameters
        self.lr = float(lr)
        self.lr_milestones = tuple(int(m) for m in lr_milestones)
        self.lr_gamma = float(lr_gamma)
        self.l2_beta = float(l2_beta)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # State
        self.bias_layer: BiasLayer | None = None  # Single bias layer (overwritten after each experience)
        self._exp_idx: int = 0

    def initialize_parameters(self, *, model: nn.Module, sample_inputs: Any | None = None) -> None:
        """
        Instantiate a neutral bias layer for later fitting.

        Args:
            model (nn.Module): Model used for device placement.
            sample_inputs (Any | None): Unused.

        Returns:
            None.
        """
        del sample_inputs
        if self.bias_layer is not None:
            return

        model_device = module_device(model, self.device)
        self.bias_layer = BiasLayer([]).to(model_device)

    @classmethod
    def requires_per_experience_fitting(cls) -> bool:
        return True

    def fit_on_repair_data(
        self,
        *,
        model: nn.Module,
        repair_dataset: Dataset | None,
        new_classes: list[int],
        num_epochs: int,
        batch_size: int,
    ) -> None:
        cur_classes = [int(c) for c in new_classes]

        # Avalanche behavior: skip bias correction after the first experience
        if self._exp_idx == 0:
            self._exp_idx += 1
            return

        if not cur_classes:
            self._exp_idx += 1
            return

        if repair_dataset is None or len(repair_dataset) <= 0:
            self._exp_idx += 1
            return
        val_loader = build_repair_dataloader(
            repair_dataset=repair_dataset,
            batch_size=batch_size,
        )
        if val_loader is None:
            self._exp_idx += 1
            return

        model_device = module_device(model, self.device)
        bias_layer = BiasLayer(sorted(cur_classes)).to(model_device)
        optimizer, scheduler = build_sgd_optimizer_and_scheduler(
            params=list(bias_layer.parameters()),
            lr=self.lr,
            momentum=0.9,
            weight_decay=0.0,
            lr_milestones=self.lr_milestones,
            lr_gamma=self.lr_gamma,
        )
        criterion = CrossEntropyLoss()

        def _forward(x: torch.Tensor) -> torch.Tensor:
            with torch.inference_mode():
                logits = model(x)
            return bias_layer(logits)

        def _reg_term(_aux: Any) -> torch.Tensor | None:
            if self.l2_beta <= 0.0:
                return None
            return self.l2_beta * (bias_layer.beta.sum() ** 2) / 2.0

        fit_repair_controller(
            controller=bias_layer,
            model=model,
            dataloader=val_loader,
            device=model_device,
            num_epochs=int(num_epochs),
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            forward_fn=_forward,
            reg_term_fn=_reg_term,
            logits_error='Expected logits shaped (B, C).',
        )

        for p in bias_layer.parameters():
            p.requires_grad = False

        self.bias_layer = bias_layer

        self._exp_idx += 1

    def correct_outputs(self, *, outputs: Any, model: nn.Module | None = None, inputs: Any | None = None) -> Any:
        if self.bias_layer is None:
            return outputs
        if not torch.is_tensor(outputs) or outputs.ndim != 2:
            return outputs

        out = outputs
        if next(self.bias_layer.parameters()).device != out.device:
            self.bias_layer = self.bias_layer.to(out.device)

        return self.bias_layer(out)


class IL2MController(RepairController):
    """
    IL2M, adapted from Avalanche's `IL2MPlugin`.

    After each experience it estimates:
      - current_classes_means: mean score for each *old* class in the current state
      - init_classes_means: mean score for each class in the state it was first introduced
      - models_confidence: mean top-1 score over *new-class samples* in the current state
      - classes2exp: mapping class -> experience index it was first introduced

    At evaluation time, if the predicted class is a 'new' class for the evaluated experience,
    it rectifies logits for 'old' classes using IL2M Eq.(1).

    Important: statistics are computed on raw model outputs (logits), matching Avalanche.

    Args:
        device: Device for fitting.
    """

    def __init__(
        self,
        *,
        device: str | None = None,
    ) -> None:
        super().__init__()

        # Hyperparameters
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # State
        self.n_classes = 0
        self.current_classes_means: list[float] = []
        self.init_classes_means: list[float] = []
        self.models_confidence: list[float] = []
        self.classes2exp: list[int] = []
        self._exp_idx: int = 0
        self._seen_classes: set[int] = set()
        self._eval_prev: list[int] = []
        self._eval_new: list[int] = []

    def _ensure_num_classes(self, num_classes: int) -> None:
        """
        Ensure internal structures can accommodate at least `num_classes` classes.

        Args:
            num_classes (int): Target number of classes.

        Returns:
            None.
        """
        if num_classes <= self.n_classes:
            return

        add = num_classes - self.n_classes
        self.current_classes_means.extend([0.0 for _ in range(add)])
        self.init_classes_means.extend([0.0 for _ in range(add)])
        self.classes2exp.extend([-1 for _ in range(add)])
        self.n_classes = num_classes

    @classmethod
    def requires_per_experience_fitting(cls) -> bool:
        return True

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

        while len(self.models_confidence) <= self._exp_idx:
            self.models_confidence.append(0.0)

        prev_classes_set = set(int(c) for c in self._seen_classes)
        cur_classes = [int(c) for c in new_classes]

        self._eval_prev = list(prev_classes_set)
        self._eval_new = cur_classes

        self._seen_classes.update(cur_classes)
        seen_classes = sorted(self._seen_classes)

        if repair_dataset is None or len(repair_dataset) <= 0 or not seen_classes:
            self._exp_idx += 1
            return

        self._ensure_num_classes(max(seen_classes) + 1)

        stat_loader = build_repair_dataloader(
            repair_dataset=repair_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        if stat_loader is None:
            self._exp_idx += 1
            return

        model_device = module_device(model, self.device)

        with preserve_model_mode_after_eval(model):
            self.current_classes_means = [0.0 for _ in range(self.n_classes)]
            counts = [0 for _ in range(self.n_classes)]

            for cls in cur_classes:
                if 0 <= cls < self.n_classes and self.classes2exp[cls] == -1:
                    self.init_classes_means[cls] = 0.0

            model_conf_sum = 0.0
            model_conf_count = 0

            for batch in stat_loader:
                x, y, *_ = batch
                if not torch.is_tensor(y):
                    y = torch.as_tensor(y)

                x = x.to(model_device)
                y = y.to(model_device)

                with torch.inference_mode():
                    logits = model(x)

                y_cpu = y.detach().to('cpu').tolist()
                scores_cpu = logits.detach().to('cpu').numpy()
                scores_width = int(scores_cpu.shape[1])

                for i, target in enumerate(y_cpu):
                    t = int(target)
                    if t < 0 or t >= scores_width:
                        continue
                    if t >= self.n_classes:
                        continue

                    counts[t] += 1

                    if t in prev_classes_set:
                        self.current_classes_means[t] += float(scores_cpu[i, t])
                    else:
                        self.init_classes_means[t] += float(scores_cpu[i, t])
                        model_conf_sum += float(np.max(scores_cpu[i, :]))
                        model_conf_count += 1

            for cls in prev_classes_set:
                c = counts[int(cls)]
                if c > 0:
                    self.current_classes_means[int(cls)] /= float(c)

            for cls in cur_classes:
                c = counts[int(cls)]
                if c > 0 and self.classes2exp[int(cls)] == -1:
                    self.init_classes_means[int(cls)] /= float(c)
                    self.classes2exp[int(cls)] = int(self._exp_idx)

            self.models_confidence[self._exp_idx] = (
                float(model_conf_sum / float(model_conf_count)) if model_conf_count > 0 else 0.0
            )

        self._exp_idx += 1

    def correct_outputs(self, *, outputs: Any, model: nn.Module | None = None, inputs: Any | None = None) -> Any:
        del model, inputs

        if not torch.is_tensor(outputs) or outputs.ndim != 2:
            return outputs

        self._ensure_num_classes(int(outputs.shape[1]))

        old_classes = [int(c) for c in self._eval_prev]
        new_classes = {int(c) for c in self._eval_new}

        if not old_classes or not new_classes or not self.models_confidence:
            return outputs

        cur_conf = float(self.models_confidence[-1])
        if cur_conf <= 0.0:
            return outputs

        predicted = torch.argmax(outputs, dim=1)
        mask = torch.tensor(
            [int(p) in new_classes for p in predicted.detach().to('cpu').tolist()],
            device=outputs.device,
        )
        if not bool(mask.any()):
            return outputs

        out = outputs.clone()

        for cls in old_classes:
            if cls < 0 or cls >= self.n_classes or cls >= int(outputs.shape[1]):
                continue

            o_exp = int(self.classes2exp[cls])
            if o_exp < 0 or o_exp >= len(self.models_confidence):
                continue

            old_conf = float(self.models_confidence[o_exp])
            if old_conf <= 0.0:
                continue

            if self.current_classes_means[cls] == 0:
                continue

            scale = (
                (self.init_classes_means[cls] / self.current_classes_means[cls])
                * (cur_conf / old_conf)
            )
            out[mask, cls] = out[mask, cls] * float(scale)

        return out
