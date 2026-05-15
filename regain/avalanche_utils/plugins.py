"""
Avalanche plugins.
"""
import hashlib
import math
from pathlib import Path
import random
import time
from typing import Mapping, Sequence

from avalanche.benchmarks import CLExperience
from avalanche.core import SupervisedPlugin
from avalanche.evaluation.metrics import loss_metrics
from avalanche.evaluation.metrics import timing_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.templates import BaseTemplate
import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from regain.analysis import MetricContext
from regain.analysis.artifacts import AnalysisArtifacts
from regain.analysis.metrics import MetricPhase
from regain.avalanche_utils.evaluation import RegainEvaluator
from regain.avalanche_utils.logging import MLflowTrainingLogger
from regain.constants import EXPERIENCE_KEY_PREFIX
from regain.constants import NAMESPACE_EVAL
from regain.constants import NAMESPACE_TRAIN
from regain.constants import NS_SEP
from regain.constants import RUN_REPAIR_SECONDS
from regain.constants import RUN_REPAIR_STEPS
from regain.constants import STREAM_REPAIR
from regain.models.controllers import BackboneControllerInterface
from regain.models.controllers import PreventionController
from regain.models.controllers import RepairController
from regain.models.controllers import TrainingObjectiveControllerInterface
from regain.models.controllers.repair.common import apply_repair_correction
from regain.models.controllers.repair.common import extract_probe_inputs
from regain.utils import extract_targets
from regain.utils import module_device
from regain.utils import preserve_rng_state
from regain.utils import RegainDataset

__all__ = [
    'BackboneCheckpointLoaderPlugin',
    'BackboneCheckpointWriterPlugin',
    'ControllerPlugin',
    'GradientClippingPlugin',
    'LRSchedulerPlugin',
    'make_training_evaluation_plugin',
    'MetricContextPlugin',
    'NumericalStabilityGuardPlugin',
    'PreventionControllerPlugin',
    'RepairControllerPlugin',
    'RegainEvaluationPlugin',
    'SeenClassesObserver',
]

_REF_BACKBONE_LOGITS_ATTR = '_regain_ref_backbone_logits'


def _sorted_unique_class_ids_for_experience(experience: CLExperience | None) -> list[int]:
    """
    Extract sorted unique class ids from one experience.

    Args:
        experience (CLExperience | None): Experience exposing `classes_in_this_experience`.

    Returns:
        list[int]: Sorted unique class ids.
    """
    class_ids = getattr(experience, 'classes_in_this_experience', []) or []
    return sorted({int(class_id) for class_id in class_ids})


class MetricContextPlugin(SupervisedPlugin):
    """
    Update MetricContext during Avalanche strategy lifecycles.
    """

    def __init__(self, context: MetricContext) -> None:
        super().__init__()
        self.context = context

    @staticmethod
    def _exp_idx(strategy: BaseTemplate) -> int:
        return int(strategy.experience.current_experience)

    def before_training(self, strategy, **kwargs) -> None:
        del strategy, kwargs
        self.context.set_phase(MetricPhase.TRAIN)
        self.context.set_log_namespace(NAMESPACE_TRAIN)
        self.context.set_log_enabled(True)
        self.context.set_experience(0)
        self.context.reset_training_counters()
        self.context.set_log_step(0)

    def before_training_exp(self, strategy, **kwargs) -> None:
        del kwargs
        self.context.set_phase(MetricPhase.TRAIN)
        self.context.set_log_namespace(NAMESPACE_TRAIN)
        self.context.set_log_enabled(True)
        self.context.set_experience(self._exp_idx(strategy))
        self.context.reset_experience_counters()

    def before_training_epoch(self, strategy, **kwargs) -> None:
        del kwargs
        self.context.set_phase(MetricPhase.TRAIN)
        self.context.set_log_namespace(NAMESPACE_TRAIN)
        self.context.set_log_enabled(True)
        self.context.set_experience(self._exp_idx(strategy))
        self.context.advance_training_epoch()

    def before_eval(self, strategy, **kwargs) -> None:
        del strategy, kwargs
        self.context.set_phase(MetricPhase.EVAL)
        if self.context.log_namespace in {NAMESPACE_TRAIN, NAMESPACE_EVAL}:
            self.context.set_log_namespace(NAMESPACE_EVAL)
            self.context.set_log_step(int(self.context.train_step))
        self.context.set_epoch(0)

    def before_eval_exp(self, strategy, **kwargs) -> None:
        del kwargs
        self.context.set_phase(MetricPhase.EVAL)
        if self.context.log_namespace in {NAMESPACE_TRAIN, NAMESPACE_EVAL}:
            self.context.set_log_namespace(NAMESPACE_EVAL)
        self.context.set_experience(self._exp_idx(strategy))
        self.context.set_epoch(0)


class BackboneCheckpointWriterPlugin(SupervisedPlugin):
    """
    Save model checkpoints after each training experience.
    """

    def __init__(self, checkpoint_dir: Path) -> None:
        """
        Initialize checkpoint writer plugin.

        Args:
            checkpoint_dir (Path): Directory where checkpoints will be stored.
        """
        super().__init__()
        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_paths: dict[int, Path] = {}

    @staticmethod
    def _snapshot_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
        """
        Copy model state to CPU tensors for serialization.

        Args:
            model (nn.Module): Model to snapshot.

        Returns:
            dict[str, torch.Tensor]: CPU copy of the model state dict.
        """
        return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    def after_training_exp(self, strategy: BaseTemplate, **kwargs) -> None:
        del kwargs
        # Persist a checkpoint for the completed training experience.
        model = strategy.model
        if not isinstance(model, nn.Module):
            raise TypeError('Strategy.model must be an nn.Module.')

        experience = strategy.experience
        exp_idx = int(experience.current_experience)
        checkpoint_path = self._checkpoint_dir / f'exp_{exp_idx:03d}.pt'
        # Serialize a CPU state snapshot to keep checkpoints device-agnostic.
        torch.save(
            {
                'experience': exp_idx,
                'model_state_dict': self._snapshot_model_state(model),
            },
            checkpoint_path,
        )
        # Track paths by experience so callers can retrieve a validated ordered list.
        self._checkpoint_paths[exp_idx] = checkpoint_path

    def checkpoint_paths(self, *, expected_count: int | None = None) -> list[Path]:
        """
        Return checkpoint paths ordered by experience index.

        Args:
            expected_count (int | None): Expected number of checkpoints.

        Returns:
            list[Path]: Ordered checkpoint paths.
        """
        ordered_indices = sorted(self._checkpoint_paths)
        if expected_count is not None:
            expected_indices = list(range(int(expected_count)))
            if ordered_indices != expected_indices:
                raise RuntimeError('Backbone checkpoints are incomplete or out of order. '
                                   f'expected={expected_indices}, observed={ordered_indices}')
        return [self._checkpoint_paths[idx] for idx in ordered_indices]


class BackboneCheckpointLoaderPlugin(SupervisedPlugin):
    """
    Load a precomputed model checkpoint at the start of each training experience.
    """

    def __init__(self, checkpoint_paths: Sequence[Path]) -> None:
        """
        Initialize checkpoint loader plugin.

        Args:
            checkpoint_paths (Sequence[Path]): Checkpoint path for each experience index.
        """
        super().__init__()
        self._checkpoint_paths = [Path(path) for path in checkpoint_paths]

    @staticmethod
    def _extract_state_dict(payload: object) -> dict[str, torch.Tensor]:
        """
        Extract a model state dict from checkpoint payload.

        Args:
            payload (object): Object returned by `torch.load`.

        Returns:
            dict[str, torch.Tensor]: Model state dictionary.
        """
        if isinstance(payload, dict) and 'model_state_dict' in payload:
            state_dict = payload['model_state_dict']
            if isinstance(state_dict, dict):
                return state_dict
        if isinstance(payload, dict):
            return payload
        raise TypeError('Invalid checkpoint payload. Expected a state_dict mapping.')

    def before_training_exp(self, strategy: BaseTemplate, **kwargs) -> None:
        del kwargs
        # Restore the checkpoint tied to the upcoming training experience.
        model = strategy.model
        if not isinstance(model, nn.Module):
            raise TypeError('Strategy.model must be an nn.Module.')

        experience = strategy.experience
        exp_idx = int(experience.current_experience)
        # Ensure the requested experience index is backed by a checkpoint.
        if exp_idx < 0 or exp_idx >= len(self._checkpoint_paths):
            raise ValueError('Missing backbone checkpoint for experience '
                             f'{exp_idx}. Available indices: 0..{max(0, len(self._checkpoint_paths) - 1)}')

        checkpoint_path = self._checkpoint_paths[exp_idx]
        if not checkpoint_path.exists():
            raise FileNotFoundError(f'Backbone checkpoint not found: {checkpoint_path}')

        # Load a CPU payload and enforce strict state compatibility on restore.
        payload = torch.load(str(checkpoint_path), map_location='cpu')
        state_dict = self._extract_state_dict(payload)
        model.load_state_dict(state_dict=state_dict, strict=True)


class LRSchedulerPlugin(SupervisedPlugin):
    """
    Reset and apply a learning rate schedule at the start of each training experience.

    At the beginning of every experience the optimizer's learning rate is restored to its
    initial value and a fresh scheduler is created from the provided class + ``kwargs``.
    The scheduler is stepped after each training epoch so that within-experience decay
    works correctly regardless of how many experiences precede the current one.
    """

    def __init__(
        self,
        *,
        scheduler_cls: type[torch.optim.lr_scheduler.LRScheduler],
        scheduler_kwargs: dict[str, object],
        initial_lr: float = 0.1,
    ) -> None:
        super().__init__()
        self._scheduler_cls = scheduler_cls
        self._scheduler_kwargs = dict(scheduler_kwargs)
        self._initial_lr = float(initial_lr)
        self._scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

    def before_training_exp(self, strategy: BaseTemplate, **kwargs) -> None:
        del kwargs
        for param_group in strategy.optimizer.param_groups:
            param_group['lr'] = self._initial_lr
        self._scheduler = self._scheduler_cls(
            optimizer=strategy.optimizer,
            **self._scheduler_kwargs,
        )

    def after_training_epoch(self, strategy: BaseTemplate, **kwargs) -> None:
        del strategy, kwargs
        if self._scheduler is not None:
            self._scheduler.step()


class GradientClippingPlugin(SupervisedPlugin):
    """
    Apply gradient clipping before each optimizer update.
    """

    def __init__(
        self,
        *,
        max_norm: float,
        norm_type: float = 2.0,
    ) -> None:
        super().__init__()
        self.max_norm = float(max_norm)
        self.norm_type = float(norm_type)

    def before_update(self, strategy: BaseTemplate, **kwargs) -> None:
        del kwargs
        trainable_params = [
            parameter for parameter in strategy.model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not trainable_params:
            return
        torch.nn.utils.clip_grad_norm_(
            trainable_params,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
        )


class PreventionControllerPlugin(SupervisedPlugin):
    """
    Wire a prevention controller into Avalanche strategy lifecycles.
    """

    def __init__(self, controller: PreventionController) -> None:
        """
        Initialize the plugin with a prevention controller.

        Args:
            controller (PreventionController): Controller to wire into Avalanche.

        Raises:
            TypeError: If the controller is not a PreventionController.

        Returns:
            None.
        """
        super().__init__()
        if not isinstance(controller, PreventionController):
            raise TypeError('PreventionControllerPlugin requires a PreventionController.')
        self.controller: PreventionController = controller

    def before_training(self, strategy: BaseTemplate, **kwargs) -> None:
        del kwargs
        # Initialize controller state from the strategy model once per training run.
        model = strategy.model
        if not isinstance(model, nn.Module):
            raise TypeError('Strategy.model must be an nn.Module.')

        self.controller.on_train_begin(model)

        # Some prevention controllers also patch backbone behavior up front.
        if isinstance(self.controller, BackboneControllerInterface):
            self.controller.correct_backbone(model)

    def before_training_exp(self, strategy: BaseTemplate, **kwargs) -> None:
        del kwargs
        # Resolve the experience dataset and forward the start-of-experience hook.
        experience = strategy.experience
        dataset: RegainDataset | None = None
        if experience is not None:
            if hasattr(experience, 'dataset'):
                dataset = experience.dataset
            elif hasattr(experience, '_dataset'):
                dataset = experience._dataset  # pylint: disable=protected-access

        self.controller.on_train_experience_begin(dataset)

    def before_training_epoch(self, strategy: BaseTemplate, **kwargs) -> None:
        del kwargs
        # Forward the epoch-start hook with a validated model instance.
        model = strategy.model
        if not isinstance(model, nn.Module):
            raise TypeError('Strategy.model must be an nn.Module.')

        self.controller.on_train_epoch_begin(model)

    def before_backward(self, strategy: BaseTemplate, **kwargs) -> None:
        del kwargs
        # Only objective-aware controllers are allowed to rewrite loss values.
        if not isinstance(self.controller, TrainingObjectiveControllerInterface):
            return

        model = strategy.model
        if not isinstance(model, nn.Module):
            raise TypeError('Strategy.model must be an nn.Module.')

        # Gather minibatch tensors required by controller loss correction.
        loss = strategy.loss
        outputs = strategy.mb_output
        inputs = strategy.mb_x
        targets = strategy.mb_y

        updated_loss = self.controller.correct_training_objective(
            loss=loss,
            outputs=outputs,
            model=model,
            inputs=inputs,
            targets=targets,
        )

        if torch.is_tensor(updated_loss):
            strategy.loss = updated_loss

    def after_training_epoch(self, strategy: BaseTemplate, **kwargs) -> None:
        del kwargs
        # Forward the epoch-end hook to keep controller state synchronized.
        model = strategy.model
        if not isinstance(model, nn.Module):
            raise TypeError('Strategy.model must be an nn.Module.')

        self.controller.on_train_epoch_end(model)

    def after_training_exp(self, strategy: BaseTemplate, **kwargs) -> None:
        del kwargs
        # Notify controller that an experience has finished training.
        model = strategy.model
        if not isinstance(model, nn.Module):
            raise TypeError('Strategy.model must be an nn.Module.')

        self.controller.on_train_experience_end(model)

    def after_training(self, strategy: BaseTemplate, **kwargs) -> None:
        del kwargs
        # Notify controller that the full training lifecycle has completed.
        model = strategy.model
        if not isinstance(model, nn.Module):
            raise TypeError('Strategy.model must be an nn.Module.')

        self.controller.on_train_end(model)

    def before_eval(self, strategy: BaseTemplate, **kwargs) -> None:
        # Do not leak strategy metadata during evaluation hooks.
        del strategy, kwargs
        self.controller.on_eval_begin()

    def after_eval(self, strategy: BaseTemplate, **kwargs) -> None:
        # Do not leak strategy metadata during evaluation hooks.
        del strategy, kwargs
        self.controller.on_eval_end()

    def before_eval_exp(self, strategy: BaseTemplate, **kwargs) -> None:
        # Do not leak per-experience strategy metadata during evaluation hooks.
        del strategy, kwargs
        self.controller.on_eval_experience_begin()

    def after_eval_exp(self, strategy: BaseTemplate, **kwargs) -> None:
        # Do not leak per-experience strategy metadata during evaluation hooks.
        del strategy, kwargs
        self.controller.on_eval_experience_end()


class RepairControllerPlugin(SupervisedPlugin):
    """
    Wire a repair controller into Avalanche strategy lifecycles.
    """

    def __init__(
        self,
        controller: RepairController,
        *,
        fit_after_experience: bool,
        repair_epochs: int,
        repair_batch_size: int,
        budget_fraction: float = 1.0,
        seed: int,
    ) -> None:
        """
        Initialize the plugin with a repair controller.

        Args:
            controller (RepairController): Controller to wire into Avalanche.
            fit_after_experience (bool): Whether to fit on repair data after each experience.
            repair_epochs (int): Number of epochs to use for repair fitting.
            repair_batch_size (int): Batch size to use for repair fitting.
            budget_fraction (float): Fraction of each fixed repair set used for repair fitting.
            seed (int): Global experiment seed used for deterministic subset selection.

        Raises:
            TypeError: If the controller is not a RepairController.
            ValueError: If budget/set values are invalid.

        Returns:
            None.
        """
        super().__init__()

        if not isinstance(controller, RepairController):
            raise TypeError('RepairControllerPlugin requires a RepairController.')

        if (not fit_after_experience) and controller.requires_per_experience_fitting():
            raise ValueError(f'{type(controller).__name__} requires per-experience fitting')

        self.controller: RepairController = controller
        self.repair_epochs = int(repair_epochs)
        self.repair_batch_size = int(repair_batch_size)
        self.fit_after_experience = bool(fit_after_experience)
        self.budget_fraction = float(budget_fraction)
        self.seed = int(seed)
        if self.repair_batch_size <= 0:
            raise ValueError('`repair_batch_size` must be positive.')
        if self.repair_epochs < 0:
            raise ValueError('`repair_epochs` must be non-negative.')
        if not 0.0 < self.budget_fraction <= 1.0:
            raise ValueError('`budget_fraction` must be in the range (0, 1].')
        self._repair_datasets: list[Dataset] = []
        self._seen_classes: set[int] = set()
        self._train_seen_classes: set[int] = set()
        self._repair_seconds_total: float = 0.0
        self._repair_steps_total: int = 0
        self.eval_correction_enabled: bool = True

    def enable_eval_correction(self) -> None:
        """
        Enable controller output correction during evaluation.

        Returns:
            None.
        """
        self.eval_correction_enabled = True

    def disable_eval_correction(self) -> None:
        """
        Disable controller output correction during evaluation.

        Returns:
            None.
        """
        self.eval_correction_enabled = False

    def initialize_parameters(self, *, model: nn.Module, dataset: Dataset | None) -> None:
        """
        Initialize the controller parameters with a probe batch from a dataset.

        Args:
            model (nn.Module): Model used to probe controller parameters.
            dataset (Dataset | None): Dataset used to extract a probe batch.

        Returns:
            None.
        """
        if dataset is not None:
            probe_loader = DataLoader(dataset, batch_size=1, shuffle=False)
            probe_inputs = extract_probe_inputs(dataloader=probe_loader, device=module_device(model, 'cpu'))
        else:
            probe_inputs = None
        self.controller.initialize_parameters(model=model, sample_inputs=probe_inputs)

    @staticmethod
    def _sample_score(
        *,
        seed: int,
        exp_idx: int,
        class_id: int,
        sample_id: int,
    ) -> int:
        """
        Compute a deterministic score for a candidate repair sample.

        Args:
            seed (int): Global experiment seed.
            exp_idx (int): Experience index.
            class_id (int): Class identifier.
            sample_id (int): Stable sample identifier (preferably original index).

        Returns:
            int: Deterministic sortable score (lower is selected first).
        """
        payload = f'{int(seed)}:{int(exp_idx)}:{int(class_id)}:{int(sample_id)}'.encode('ascii')
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, byteorder='big', signed=False)

    @staticmethod
    def _fit_seed(*, seed: int, exp_idx: int | None) -> int:
        """
        Derive a deterministic repair-fit seed from the experiment seed and fit boundary.

        Args:
            seed (int): Global experiment seed.
            exp_idx (int | None): Experience index, or None for final-only fitting.

        Returns:
            int: Positive int32-compatible seed for Python, NumPy, and Torch RNGs.
        """
        tag = 'final' if exp_idx is None else f'{int(exp_idx)}'
        payload = f'{int(seed)}:{tag}'.encode('ascii')
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, byteorder='big', signed=False) & 0x7FFFFFFF

    @staticmethod
    def _extract_original_indices(dataset: Dataset) -> list[int]:
        """
        Extract stable sample identifiers from the repair dataset.

        Args:
            dataset (Dataset): Dataset with `original_indices` attribute.

        Returns:
            list[int]: Original indices aligned with dataset rows.

        Raises:
            AttributeError: If the dataset lacks `original_indices`.
            ValueError: If the length does not match the dataset.
        """
        original_indices = dataset.original_indices
        values = [int(value) for value in list(original_indices)]
        if len(values) != len(dataset):
            raise ValueError(f'original_indices length ({len(values)}) does not match '
                             f'dataset length ({len(dataset)}).')
        return values

    def _select_budget_fraction(
        self,
        *,
        repair_dataset: Dataset,
        exp_idx: int,
    ) -> Dataset | None:
        """
        Select a deterministic stratified subset from a fixed repair set.

        Args:
            repair_dataset (Dataset): Fixed repair set for one experience.
            exp_idx (int): Experience index.

        Returns:
            Dataset | None: Deterministically selected subset.

        Raises:
            TypeError: If the repair dataset does not support subsetting.
        """
        if not hasattr(repair_dataset, 'subset'):
            raise TypeError('Repair dataset must expose a `subset` method.')

        targets = extract_targets(repair_dataset)
        if not targets:
            return None
        targets_arr = np.asarray(targets, dtype=np.int64)
        original_indices = self._extract_original_indices(repair_dataset)
        repair_set_size = int(len(targets_arr))
        budget_size = int(np.floor(float(self.budget_fraction) * float(repair_set_size)))
        if budget_size <= 0:
            return None

        # Compute deterministic stratified per-class selected counts.
        class_ids = sorted(int(cls) for cls in np.unique(targets_arr))
        class_counts: dict[int, int] = {class_id: int(np.sum(targets_arr == class_id)) for class_id in class_ids}
        class_target_counts: dict[int, float] = {}
        selected_count_by_class: dict[int, int] = {}
        selected_count_total = 0
        for class_id in class_ids:
            class_count = int(class_counts[class_id])
            class_target_count = float(self.budget_fraction) * float(class_count)
            class_target_counts[class_id] = class_target_count
            class_selected_count = int(np.floor(class_target_count))
            selected_count_by_class[class_id] = class_selected_count
            selected_count_total += class_selected_count

        remaining_slots = int(budget_size - selected_count_total)
        while remaining_slots > 0:
            candidate_class_ids = [
                class_id for class_id in class_ids if selected_count_by_class[class_id] < class_counts[class_id]
            ]
            if not candidate_class_ids:
                raise ValueError('Repair budget guard failed: could not place remaining stratified budget slots. '
                                 f'exp_idx={exp_idx}, remaining_slots={remaining_slots}.')
            selected_class_id = min(
                candidate_class_ids,
                key=lambda class_id: (
                    -(class_target_counts[class_id] - float(selected_count_by_class[class_id])),
                    class_id,
                ),
            )
            selected_count_by_class[selected_class_id] += 1
            remaining_slots -= 1

        selected_indices: list[int] = []
        for class_id in class_ids:
            class_budget_count = int(selected_count_by_class[class_id])
            if class_budget_count <= 0:
                continue
            class_local_indices = np.where(targets_arr == class_id)[0].tolist()

            class_local_indices = sorted(
                class_local_indices,
                key=lambda local_idx, cid=class_id: self._sample_score(
                    seed=self.seed,
                    exp_idx=exp_idx,
                    class_id=cid,
                    sample_id=int(original_indices[local_idx]),
                ),
            )
            selected_indices.extend(class_local_indices[:class_budget_count])

        selected_indices = sorted(int(idx) for idx in selected_indices)
        return repair_dataset.subset(selected_indices)

    @staticmethod
    def _repair_metric_step(exp_idx: int | None) -> int:
        """
        Build a stable step index for repair-resource metric logging.

        Args:
            exp_idx (int | None): Experience index being fitted, or None for final-only fitting.

        Returns:
            int: Metric step.
        """
        if exp_idx is None:
            return 0
        return int(exp_idx) + 1

    def _log_repair_resource_metrics(
        self,
        *,
        exp_idx: int | None,
        elapsed_seconds: float,
        repair_steps: int,
    ) -> None:
        """
        Log per-fit and cumulative repair resource metrics.

        Args:
            exp_idx (int | None): Experience index, or None for final-only fitting.
            elapsed_seconds (float): Wall-clock duration for this fit call in seconds.
            repair_steps (int): Optimization step count for this fit call,
                typically `epochs * ceil(n_repair / batch_size)`.
        """
        if mlflow.active_run() is None:
            return

        step = self._repair_metric_step(exp_idx)
        mlflow.log_metric(
            key=RUN_REPAIR_SECONDS,
            value=float(self._repair_seconds_total),
            step=step,
        )
        mlflow.log_metric(
            key=RUN_REPAIR_STEPS,
            value=float(self._repair_steps_total),
            step=step,
        )

        suffix = (f'{NS_SEP}{EXPERIENCE_KEY_PREFIX}{int(exp_idx):03d}' if exp_idx is not None else f'{NS_SEP}final')
        mlflow.log_metric(
            key=f'{RUN_REPAIR_SECONDS}{suffix}',
            value=float(elapsed_seconds),
            step=step,
        )
        mlflow.log_metric(
            key=f'{RUN_REPAIR_STEPS}{suffix}',
            value=float(repair_steps),
            step=step,
        )

    def _fit_controller_on_repair_dataset(
        self,
        *,
        model: nn.Module,
        repair_dataset: Dataset,
        new_classes: list[int],
        exp_idx: int | None,
    ) -> None:
        """
        Fit the controller and record repair-time resource metrics.

        Args:
            model (nn.Module): Backbone model.
            repair_dataset (Dataset): Dataset to fit on.
            new_classes (list[int]): Newly observed classes.
            exp_idx (int | None): Experience index, or None for final-only fitting.

        Notes:
            `repair_steps` is estimated as
            `repair_epochs * ceil(len(repair_dataset) / repair_batch_size)`.
        """
        n_samples = int(len(repair_dataset))
        repair_steps = int(self.repair_epochs * math.ceil(float(n_samples) / float(self.repair_batch_size)))
        started_at = time.perf_counter()
        fit_seed = self._fit_seed(seed=self.seed, exp_idx=exp_idx)
        with preserve_rng_state():
            random.seed(fit_seed)
            np.random.seed(fit_seed)
            torch.manual_seed(fit_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(fit_seed)
            self.controller.fit_on_repair_data(
                model=model,
                repair_dataset=repair_dataset,
                new_classes=new_classes,
                num_epochs=self.repair_epochs,
                batch_size=self.repair_batch_size,
            )
        elapsed_seconds = float(time.perf_counter() - started_at)
        self._repair_seconds_total += elapsed_seconds
        self._repair_steps_total += repair_steps
        self._log_repair_resource_metrics(
            exp_idx=exp_idx,
            elapsed_seconds=elapsed_seconds,
            repair_steps=repair_steps,
        )

    def _ingest_repair_dataset(self, *, experience: CLExperience) -> Dataset | None:
        """
        Resolve and store the budgeted repair subset for one experience.

        Args:
            experience (CLExperience): Avalanche experience.

        Returns:
            Dataset | None: Fixed repair set dataset for this experience.
        """
        repair_set_dataset = self._resolve_repair_dataset(experience)
        if repair_set_dataset is None:
            return None
        exp_idx = int(experience.current_experience)
        repair_used_dataset = self._select_budget_fraction(
            repair_dataset=repair_set_dataset,
            exp_idx=exp_idx,
        )
        if repair_used_dataset is not None:
            self._repair_datasets.append(repair_used_dataset)
        return repair_set_dataset

    def before_training_exp(self, strategy: BaseTemplate, **kwargs) -> None:
        # Track classes seen in the training stream for downstream anti-cheat checks.
        del kwargs
        experience = strategy.experience
        if experience is None:
            return
        self._train_seen_classes.update(self._resolve_training_classes(experience))

    def after_training_exp(self, strategy: BaseTemplate, **kwargs) -> None:
        # Resolve training outputs for this experience.
        del kwargs
        experience = strategy.experience
        if experience is None:
            return
        model = strategy.model
        if not isinstance(model, nn.Module):
            raise TypeError('Strategy.model must be an nn.Module.')
        exp_idx = int(experience.current_experience)

        # Aggregate budget-filtered repair data exposed by the benchmark.
        repair_set_ds = self._ingest_repair_dataset(experience=experience)

        # Track classes introduced at this experience boundary.
        new_classes = self._resolve_new_classes(experience, repair_set_ds)
        self._seen_classes.update(new_classes)

        # Notify controller lifecycle hook.
        self.controller.on_train_experience_end(model)

        # Run per-experience repair fitting when configured.
        if self.fit_after_experience:
            combined_dataset = self._combined_repair_dataset()
            if combined_dataset is None:
                return
            self._fit_controller_on_repair_dataset(
                model=model,
                repair_dataset=combined_dataset,
                new_classes=new_classes,
                exp_idx=exp_idx,
            )

    def after_training(self, strategy: BaseTemplate, **kwargs) -> None:
        # Resolve final model instance and close the training lifecycle.
        del kwargs
        model = strategy.model
        if not isinstance(model, nn.Module):
            raise TypeError('Strategy.model must be an nn.Module.')

        # Notify controller that full training is complete.
        self.controller.on_train_end(model)

        # If fitting was deferred, run one final fit on accumulated repair data.
        if not self.fit_after_experience:
            combined_dataset = self._combined_repair_dataset()
            if combined_dataset is None:
                return
            self._fit_controller_on_repair_dataset(
                model=model,
                repair_dataset=combined_dataset,
                new_classes=sorted(self._seen_classes),
                exp_idx=None,
            )

    def before_eval(self, strategy: BaseTemplate, **kwargs) -> None:
        del strategy, kwargs
        # Anti-cheat: don't expose strategy metadata to controllers during evaluation.
        self.controller.on_eval_begin()

    def after_eval(self, strategy: BaseTemplate, **kwargs) -> None:
        del strategy, kwargs
        # Anti-cheat: don't expose strategy metadata to controllers during evaluation.
        self.controller.on_eval_end()

    def before_eval_exp(self, strategy: BaseTemplate, **kwargs) -> None:
        del strategy, kwargs
        # Anti-cheat: don't expose per-experience metadata to controllers.
        self.controller.on_eval_experience_begin()

    def after_eval_exp(self, strategy: BaseTemplate, **kwargs) -> None:
        del strategy, kwargs
        # Anti-cheat: don't expose per-experience metadata to controllers.
        self.controller.on_eval_experience_end()

    def apply_repair_correction(
        self,
        *,
        model: nn.Module,
        inputs: torch.Tensor,
        backbone_outputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply controller correction while enforcing anti-cheat invariants.

        Args:
            model (nn.Module): Backbone model.
            inputs (torch.Tensor): Batch inputs.
            backbone_outputs (torch.Tensor): Backbone outputs.

        Returns:
            torch.Tensor: Corrected outputs.
        """
        return apply_repair_correction(
            controller=self.controller,
            model=model,
            inputs=inputs,
            backbone_outputs=backbone_outputs,
            train_seen_classes=self._train_seen_classes,
        )

    def after_eval_forward(self, strategy: BaseTemplate, **kwargs) -> None:
        # Apply controller output correction while enforcing anti-cheat invariants.
        del kwargs
        if not self.eval_correction_enabled:
            return
        model = strategy.model
        if not isinstance(model, nn.Module):
            raise TypeError('Strategy.model must be an nn.Module.')
        if (self._should_stash_ref_backbone_logits(strategy=strategy) and torch.is_tensor(strategy.mb_output)):
            setattr(
                strategy,
                _REF_BACKBONE_LOGITS_ATTR,
                strategy.mb_output.detach().clone(),
            )
        strategy.mb_output = self.apply_repair_correction(
            model=model,
            inputs=strategy.mb_x,
            backbone_outputs=strategy.mb_output,
        )

    @staticmethod
    def _should_stash_ref_backbone_logits(strategy: BaseTemplate) -> bool:
        """
        Check whether the current evaluation iteration needs backbone logits.

        Args:
            strategy (BaseTemplate): Strategy currently being evaluated.

        Returns:
            bool: True when the current minibatch should expose pre-correction
                backbone logits to downstream prediction capture.
        """
        capture_context = getattr(strategy, '_regain_prediction_capture_context', None)
        if not isinstance(capture_context, Mapping):
            return False
        if not bool(capture_context.get('ref_use_backbone_logits', False)):
            return False

        ref_test_exp_idx = capture_context.get('ref_test_exp_idx')
        experience = getattr(strategy, 'experience', None)
        current_exp_idx = getattr(experience, 'current_experience', None)
        if ref_test_exp_idx is None or current_exp_idx is None:
            return False
        return int(ref_test_exp_idx) == int(current_exp_idx)

    @staticmethod
    def _resolve_repair_dataset(experience: CLExperience) -> Dataset | None:
        """
        Resolve the repair dataset for the current experience from the benchmark streams.

        Args:
            experience (CLExperience): Avalanche experience instance.

        Returns:
            Dataset | None: Repair dataset for the experience, if available.
        """
        exp_id = int(experience.current_experience)
        benchmark = getattr(experience, 'benchmark', None)
        if benchmark is None:
            return None

        repair_exp = None
        if hasattr(benchmark, 'repair_stream'):
            repair_exp = benchmark.repair_stream[exp_id]
        elif hasattr(benchmark, 'streams') and STREAM_REPAIR in benchmark.streams:
            repair_exp = benchmark.streams[STREAM_REPAIR][exp_id]

        if repair_exp is None:
            return None

        if hasattr(repair_exp, 'dataset'):
            return repair_exp.dataset
        if hasattr(repair_exp, '_dataset'):
            return repair_exp._dataset  # pylint: disable=protected-access
        return None

    @staticmethod
    def _resolve_new_classes(
        experience: CLExperience,
        repair_dataset: Dataset | None,
    ) -> list[int]:
        """
        Resolve newly introduced class IDs for the current experience.

        Args:
            experience (CLExperience): Avalanche experience instance.
            repair_dataset (Dataset | None): Repair dataset for the experience.

        Returns:
            list[int]: Sorted class IDs introduced in the current experience.
        """
        del repair_dataset
        return _sorted_unique_class_ids_for_experience(experience)

    @staticmethod
    def _resolve_training_classes(experience: CLExperience) -> list[int]:
        """
        Resolve class IDs present in a training experience.

        Args:
            experience (CLExperience): Avalanche experience instance.

        Returns:
            list[int]: Sorted class IDs in the training experience.
        """
        return _sorted_unique_class_ids_for_experience(experience)

    def _combined_repair_dataset(self) -> Dataset | None:
        """
        Build a combined repair dataset from all seen repair experiences.

        Returns:
            Dataset | None: Combined dataset, or None if no repair data is available.
        """
        datasets = [ds for ds in self._repair_datasets if ds is not None]
        if not datasets:
            return None
        if len(datasets) == 1:
            return datasets[0]
        return ConcatDataset(datasets)


ControllerPlugin = PreventionControllerPlugin | RepairControllerPlugin


class NumericalStabilityGuardPlugin(SupervisedPlugin):
    """
    Fail fast when non-finite tensors are observed during training or evaluation.
    """

    def __init__(self, *, context: MetricContext) -> None:
        """
        Initialize the numerical stability guard plugin.

        Args:
            context (MetricContext): Shared metric context used to enrich failures.
        """
        super().__init__()
        self._context = context
        self._train_batch_idx = 0

    def before_training(self, strategy: BaseTemplate, **kwargs) -> None:
        del strategy, kwargs
        self._train_batch_idx = 0

    def before_backward(self, strategy: BaseTemplate, **kwargs) -> None:
        del kwargs
        self._train_batch_idx += 1
        context = self._error_context(
            strategy=strategy,
            batch_idx=self._train_batch_idx,
        )
        self._assert_finite_value(
            value=strategy.loss,
            tensor_name='loss',
            context=context,
        )
        self._assert_finite_value(
            value=strategy.mb_output,
            tensor_name='mb_output',
            context=context,
        )

    def after_training_epoch(self, strategy: BaseTemplate, **kwargs) -> None:
        del kwargs
        model = strategy.model
        if not isinstance(model, nn.Module):
            return
        total_non_finite = 0
        first_param_name: str | None = None
        first_non_finite_value: object | None = None
        for param_name, param in model.named_parameters():
            mask = ~torch.isfinite(param.detach())
            non_finite_count = int(torch.sum(mask).item())
            if non_finite_count <= 0:
                continue
            total_non_finite += non_finite_count
            if first_param_name is None:
                first_param_name = str(param_name)
                first_non_finite_value = (param.detach()[mask].reshape(-1)[0].to(device='cpu').item())
        if total_non_finite <= 0:
            return
        context = self._error_context(
            strategy=strategy,
            batch_idx=self._train_batch_idx,
        )
        self._raise_non_finite(
            tensor_name='model.parameters',
            non_finite_count=total_non_finite,
            context=context,
            tensor_shape='n/a',
            tensor_dtype='n/a',
            first_non_finite_value=first_non_finite_value,
            extra={
                'first_parameter': first_param_name,
            },
        )

    def _error_context(
        self,
        *,
        strategy: BaseTemplate,
        batch_idx: int,
    ) -> dict[str, object]:
        phase = self._context.phase
        if hasattr(phase, 'value'):
            phase_label = str(phase.value)
        else:
            phase_label = str(phase)
        experience = getattr(strategy, 'experience', None)
        exp_idx: int | None = None
        if (experience is not None and hasattr(experience, 'current_experience') and
                getattr(experience, 'current_experience') is not None):
            exp_idx = int(experience.current_experience)
        return {
            'phase': phase_label,
            'exp_idx': exp_idx,
            'step': int(self._context.log_step),
            'batch': int(batch_idx),
            'eval_tag': str(getattr(strategy, '_regain_eval_tag', '') or ''),
        }

    def _assert_finite_value(
        self,
        *,
        value: object,
        tensor_name: str,
        context: Mapping[str, object],
    ) -> None:
        if torch.is_tensor(value):
            mask = ~torch.isfinite(value.detach())
            non_finite_count = int(torch.sum(mask).item())
            if non_finite_count <= 0:
                return
            first_non_finite_value = (value.detach()[mask].reshape(-1)[0].to(device='cpu').item())
            self._raise_non_finite(
                tensor_name=tensor_name,
                non_finite_count=non_finite_count,
                context=context,
                tensor_shape=tuple(int(dim) for dim in value.shape),
                tensor_dtype=str(value.dtype),
                first_non_finite_value=first_non_finite_value,
            )
            return
        if isinstance(value, (float, int)) and not math.isfinite(float(value)):
            self._raise_non_finite(
                tensor_name=tensor_name,
                non_finite_count=1,
                context=context,
                tensor_shape=(),
                tensor_dtype=type(value).__name__,
                first_non_finite_value=float(value),
            )

    def _raise_non_finite(
        self,
        *,
        tensor_name: str,
        non_finite_count: int,
        context: Mapping[str, object],
        tensor_shape: object,
        tensor_dtype: str,
        first_non_finite_value: object,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        payload: dict[str, object] = {
            'tensor_name': str(tensor_name),
            'non_finite_count': int(non_finite_count),
            'tensor_shape': tensor_shape,
            'tensor_dtype': str(tensor_dtype),
            'first_non_finite_value': first_non_finite_value,
            'phase': context.get('phase'),
            'exp_idx': context.get('exp_idx'),
            'step': context.get('step'),
            'batch': context.get('batch'),
            'eval_tag': context.get('eval_tag'),
        }
        if extra is not None:
            for key, value in extra.items():
                payload[str(key)] = value
        raise RuntimeError('Non-finite tensor detected. '
                           f'tensor={payload["tensor_name"]}, '
                           f'non_finite_count={payload["non_finite_count"]}, '
                           f'phase={payload["phase"]}, '
                           f'exp_idx={payload["exp_idx"]}, '
                           f'step={payload["step"]}, '
                           f'batch={payload["batch"]}, '
                           f'eval_tag={payload["eval_tag"]}')


class SeenClassesObserver(SupervisedPlugin):
    """
    Track class ids observed in the training stream.
    """

    def __init__(self) -> None:
        """
        Initialize an empty seen-class cache.
        """
        super().__init__()
        self.seen_classes: set[int] = set()

    def before_training_exp(self, strategy: BaseTemplate, **kwargs) -> None:
        """
        Accumulate class ids from the current training experience.

        Args:
            strategy (BaseTemplate): Avalanche strategy.
        """
        del kwargs
        experience = strategy.experience
        dataset = None
        if experience is not None:
            dataset = getattr(experience, 'dataset', None)
            if dataset is None:
                dataset = getattr(experience, '_dataset', None)
        if dataset is None:
            return
        self.seen_classes.update(int(target) for target in extract_targets(dataset))


class RegainEvaluationPlugin(SupervisedPlugin):
    """
    Thin Avalanche hook adapter around `RegainEvaluator`.
    """

    def __init__(
        self,
        *,
        evaluator: RegainEvaluator,
        seen_classes_observer: SeenClassesObserver,
    ) -> None:
        """
        Initialize the thin posthoc-evaluation plugin.

        Args:
            evaluator (RegainEvaluator): Custom evaluator helper.
            seen_classes_observer (SeenClassesObserver): Seen-class observer.
        """
        super().__init__()
        self.evaluator = evaluator
        self.seen_classes_observer = seen_classes_observer

    @property
    def artifacts(self) -> AnalysisArtifacts | None:
        """
        Return the latest analysis artifact payload.

        Returns:
            AnalysisArtifacts | None: Latest artifact payload.
        """
        return self.evaluator.artifacts

    @property
    def last_posthoc_scalar_results(self) -> dict[str, float] | None:
        """
        Return the latest posthoc scalar metrics.

        Returns:
            dict[str, float] | None: Latest scalar metrics.
        """
        return self.evaluator.last_posthoc_scalar_results

    def before_training(self, strategy: BaseTemplate, **kwargs) -> None:
        """
        Bootstrap any pre-training evaluator state.

        Args:
            strategy (BaseTemplate): Avalanche strategy.
        """
        del strategy, kwargs
        self.evaluator.run_before_training()

    def before_eval(self, strategy: BaseTemplate, **kwargs) -> None:
        """
        Delegate retained strategy-managed eval setup to the helper.

        Args:
            strategy (BaseTemplate): Avalanche strategy.
        """
        del kwargs
        self.evaluator.before_strategy_eval(strategy=strategy)

    def before_eval_exp(self, strategy: BaseTemplate, **kwargs) -> None:
        """
        Delegate retained per-experience eval setup to the helper.

        Args:
            strategy (BaseTemplate): Avalanche strategy.
        """
        del kwargs
        self.evaluator.before_strategy_eval_exp(strategy=strategy)

    def after_eval_iteration(self, strategy: BaseTemplate, **kwargs) -> None:
        """
        Delegate retained eval-batch observation to the helper.

        Args:
            strategy (BaseTemplate): Avalanche strategy.
        """
        del kwargs
        self.evaluator.observe_strategy_eval_batch(strategy=strategy)

    def after_eval_exp(self, strategy: BaseTemplate, **kwargs) -> None:
        """
        Finalize one retained strategy-managed eval experience.

        Args:
            strategy (BaseTemplate): Avalanche strategy.
        """
        del strategy, kwargs
        self.evaluator.after_strategy_eval_exp()

    def after_eval(self, strategy: BaseTemplate, **kwargs) -> None:
        """
        Finalize one retained strategy-managed eval pass.

        Args:
            strategy (BaseTemplate): Avalanche strategy.
        """
        del strategy, kwargs
        self.evaluator.after_strategy_eval()

    def after_training_exp(self, strategy: BaseTemplate, **kwargs) -> None:
        """
        Delegate post-experience evaluation to the helper.

        Args:
            strategy (BaseTemplate): Avalanche strategy.
        """
        del kwargs
        self.evaluator.run_after_training_exp(
            strategy=strategy,
            seen_classes=self.seen_classes_observer.seen_classes,
        )

    def after_training(self, strategy: BaseTemplate, **kwargs) -> None:
        """
        Delegate end-of-training evaluation to the helper.

        Args:
            strategy (BaseTemplate): Avalanche strategy.
        """
        del strategy, kwargs
        self.evaluator.run_after_training(seen_classes=self.seen_classes_observer.seen_classes,)


def make_training_evaluation_plugin(
    *,
    context: MetricContext,
    keep_timestep_results: bool = True,
    log_to_console: bool = True,
    log_to_mlflow: bool = True,
) -> EvaluationPlugin:
    """
    Create the slim Avalanche evaluator retained on the training strategy.

    Args:
        context (MetricContext): Metric context for logging.
        keep_timestep_results (bool): Whether to keep per-time-step results.
        log_to_console (bool): Whether to include console logging.
        log_to_mlflow (bool): Whether to include MLflow logging for retained
            Avalanche training metrics.

    Returns:
        EvaluationPlugin: Configured strategy-side evaluation plugin.
    """
    loggers = []
    if log_to_console:
        loggers.append(InteractiveLogger())
    if log_to_mlflow:
        loggers.append(MLflowTrainingLogger(context=context))

    metrics = [
        loss_metrics(epoch=True),
        timing_metrics(epoch=True),
    ]
    return EvaluationPlugin(*metrics, loggers=loggers, collect_all=keep_timestep_results)
