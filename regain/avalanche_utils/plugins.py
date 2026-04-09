"""
Avalanche plugins.
"""
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import time
from typing import Mapping, Sequence

from avalanche.benchmarks.scenarios import NCScenario
from avalanche.core import SupervisedPlugin
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.evaluation.metrics import forgetting_metrics
from avalanche.evaluation.metrics import forward_transfer_metrics
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

from regain.analysis import build_analysis_artifacts
from regain.analysis import MetricContext
from regain.analysis import ordered_accuracies
from regain.analysis.artifacts import ARTIFACT_ACC_EXP_BASE
from regain.analysis.artifacts import ARTIFACT_ACC_FINAL_BASE
from regain.analysis.artifacts import ARTIFACT_RHO
from regain.analysis.artifacts import ARTIFACT_RHO_AVG
from regain.analysis.metrics import MetricPhase
from regain.avalanche_utils.logging import MLflowLogger
from regain.constants import COLUMN_STATUS
from regain.constants import DIAG_VECTOR_KEYS
from regain.constants import EXPERIENCE_KEY_PREFIX
from regain.constants import MLFLOW_ARTIFACT_ANALYSIS_FILE
from regain.constants import NAMESPACE_EVAL
from regain.constants import NAMESPACE_RUN
from regain.constants import NAMESPACE_TRAIN
from regain.constants import NS_SEP
from regain.constants import RUN_ACC_FINAL_TEST
from regain.constants import RUN_ACC_FINAL_TEST_AVG_BASE
from regain.constants import RUN_ACC_FINAL_TEST_AVG_CTRL
from regain.constants import RUN_ACC_FINAL_TRAIN
from regain.constants import RUN_ACC_FINAL_TRAIN_AVG_BASE
from regain.constants import RUN_ACC_FINAL_TRAIN_AVG_CTRL
from regain.constants import RUN_ACC_REF_TEST
from regain.constants import RUN_ACC_REF_TRAIN
from regain.constants import RUN_CALIB_AECE
from regain.constants import RUN_CALIB_BRIER
from regain.constants import RUN_CALIB_ECE
from regain.constants import RUN_CALIB_MAX_ECE
from regain.constants import RUN_CALIB_MCE
from regain.constants import RUN_CALIB_NLL
from regain.constants import RUN_DIAG_AVG_CONF
from regain.constants import RUN_DIAG_AVG_ENTROPY
from regain.constants import RUN_DIAG_LOGIT_AVG_DRIFT
from regain.constants import RUN_DIAG_OUT_OF_TASK_RATE
from regain.constants import RUN_EPS
from regain.constants import RUN_LATENCY_MS_PER_SAMPLE_BASE
from regain.constants import RUN_LATENCY_MS_PER_SAMPLE_CTRL
from regain.constants import RUN_LATENCY_MS_RATIO
from regain.constants import RUN_LATENCY_SAMPLES_PER_SEC_BASE
from regain.constants import RUN_LATENCY_SAMPLES_PER_SEC_CTRL
from regain.constants import RUN_REPAIR_SECONDS
from regain.constants import RUN_REPAIR_STEPS
from regain.constants import RUN_RHO
from regain.constants import RUN_RHO_AVG
from regain.constants import STREAM_REPAIR
from regain.experiments.utils import extract_scalar_metrics
from regain.models.controllers import BackboneControllerInterface
from regain.models.controllers import PreventionController
from regain.models.controllers import RepairController
from regain.models.controllers import TrainingObjectiveControllerInterface
from regain.models.controllers.repair.common import extract_probe_inputs
from regain.utils import extract_targets
from regain.utils import module_device
from regain.utils import RegainDataset

__all__ = [
    'BackboneCheckpointLoaderPlugin',
    'BackboneCheckpointWriterPlugin',
    'CalibrationDiagnosticsPlugin',
    'ControllerPlugin',
    'EvaluationIntegrityPlugin',
    'GradientClippingPlugin',
    'LRSchedulerPlugin',
    'PreventionControllerPlugin',
    'RepairControllerPlugin',
    'RegainEvaluationPlugin',
    'MetricContextPlugin',
    'NumericalStabilityGuardPlugin',
    'PredictionLoggingPlugin',
    'SeenClassesMaskPlugin',
    'make_evaluation_plugin',
]

_STATUS_INCOMPLETE_ACC_EXP_BASE = 'incomplete_acc_exp_base'
_METRIC_TOKEN_AVALANCHE_TOP1_ACC_EXP = 'Top1_Acc_Exp'
_METRIC_TOKEN_AVALANCHE_TOP1_ACC_STREAM = 'Top1_Acc_Stream'
_METRIC_TOKEN_AVALANCHE_TEST_STREAM = 'test_stream'
_METRIC_TOKEN_AVALANCHE_TRAIN_STREAM = 'train_stream'
_REF_BACKBONE_LOGITS_ATTR = '_regain_ref_backbone_logits'


def _sorted_unique_class_ids_for_experience(experience: object | None) -> list[int]:
    """
    Extract sorted unique class ids from one experience.

    Args:
        experience (object | None): Experience exposing `classes_in_this_experience`.

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
        self.context.set_phase(MetricPhase.TRAIN)
        self.context.set_log_namespace(NAMESPACE_TRAIN)
        self.context.set_log_enabled(True)
        self.context.set_experience(0)
        self.context.reset_training_counters()
        self.context.set_log_step(0)

    def before_training_exp(self, strategy, **kwargs) -> None:
        self.context.set_phase(MetricPhase.TRAIN)
        self.context.set_log_namespace(NAMESPACE_TRAIN)
        self.context.set_log_enabled(True)
        self.context.set_experience(self._exp_idx(strategy))
        self.context.reset_experience_counters()

    def before_training_epoch(self, strategy, **kwargs) -> None:
        self.context.set_phase(MetricPhase.TRAIN)
        self.context.set_log_namespace(NAMESPACE_TRAIN)
        self.context.set_log_enabled(True)
        self.context.set_experience(self._exp_idx(strategy))
        self.context.advance_training_epoch()

    def before_eval(self, strategy, **kwargs) -> None:
        self.context.set_phase(MetricPhase.EVAL)
        if self.context.log_namespace in {NAMESPACE_TRAIN, NAMESPACE_EVAL}:
            self.context.set_log_namespace(NAMESPACE_EVAL)
            self.context.set_log_step(int(self.context.train_step))
        self.context.set_epoch(0)

    def before_eval_exp(self, strategy, **kwargs) -> None:
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
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }

    def after_training_exp(self, strategy: BaseTemplate, **kwargs) -> None:
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
                raise RuntimeError(
                    'Backbone checkpoints are incomplete or out of order. '
                    f'expected={expected_indices}, observed={ordered_indices}'
                )
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
        # Restore the checkpoint tied to the upcoming training experience.
        model = strategy.model
        if not isinstance(model, nn.Module):
            raise TypeError('Strategy.model must be an nn.Module.')

        experience = strategy.experience
        exp_idx = int(experience.current_experience)
        # Ensure the requested experience index is backed by a checkpoint.
        if exp_idx < 0 or exp_idx >= len(self._checkpoint_paths):
            raise ValueError(
                'Missing backbone checkpoint for experience '
                f'{exp_idx}. Available indices: 0..{max(0, len(self._checkpoint_paths) - 1)}'
            )

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
        for param_group in strategy.optimizer.param_groups:
            param_group['lr'] = self._initial_lr
        self._scheduler = self._scheduler_cls(
            optimizer=strategy.optimizer,
            **self._scheduler_kwargs,
        )

    def after_training_epoch(self, strategy: BaseTemplate, **kwargs) -> None:
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
        trainable_params = [
            parameter
            for parameter in strategy.model.parameters()
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
        # Initialize controller state from the strategy model once per training run.
        model = strategy.model
        if not isinstance(model, nn.Module):
            raise TypeError('Strategy.model must be an nn.Module.')

        self.controller.on_train_begin(model)

        # Some prevention controllers also patch backbone behavior up front.
        if isinstance(self.controller, BackboneControllerInterface):
            self.controller.correct_backbone(model)

    def before_training_exp(self, strategy: BaseTemplate, **kwargs) -> None:
        # Resolve the experience dataset and forward the start-of-experience hook.
        experience = strategy.experience
        dataset: RegainDataset | None = None
        if experience is not None:
            if hasattr(experience, 'dataset'):
                dataset = experience.dataset
            elif hasattr(experience, '_dataset'):
                dataset = experience._dataset

        self.controller.on_train_experience_begin(dataset)

    def before_training_epoch(self, strategy: BaseTemplate, **kwargs) -> None:
        # Forward the epoch-start hook with a validated model instance.
        model = strategy.model
        if not isinstance(model, nn.Module):
            raise TypeError('Strategy.model must be an nn.Module.')

        self.controller.on_train_epoch_begin(model)

    def before_backward(self, strategy: BaseTemplate, **kwargs) -> None:
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
        # Forward the epoch-end hook to keep controller state synchronized.
        model = strategy.model
        if not isinstance(model, nn.Module):
            raise TypeError('Strategy.model must be an nn.Module.')

        self.controller.on_train_epoch_end(model)

    def after_training_exp(self, strategy: BaseTemplate, **kwargs) -> None:
        # Notify controller that an experience has finished training.
        model = strategy.model
        if not isinstance(model, nn.Module):
            raise TypeError('Strategy.model must be an nn.Module.')

        self.controller.on_train_experience_end(model)

    def after_training(self, strategy: BaseTemplate, **kwargs) -> None:
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
        seed: int = 1,
    ) -> None:
        """
        Initialize the plugin with a repair controller.

        Args:
            controller (RepairController): Controller to wire into Avalanche.
            fit_after_experience (bool): Whether to fit on repair data after each experience.
            repair_epochs (int): Number of epochs to use for repair fitting.
            repair_batch_size (int): Batch size to use for repair fitting.
            budget_fraction (float): Fraction of each fixed repair set used for repair fitting.
            seed (int): Global seed used for deterministic subset selection.

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
        if not (0.0 < self.budget_fraction <= 1.0):
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
            seed (int): Global seed.
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
            raise ValueError(
                f'original_indices length ({len(values)}) does not match '
                f'dataset length ({len(dataset)}).'
            )
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
        class_counts: dict[int, int] = {
            class_id: int(np.sum(targets_arr == class_id))
            for class_id in class_ids
        }
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
                class_id
                for class_id in class_ids
                if selected_count_by_class[class_id] < class_counts[class_id]
            ]
            if not candidate_class_ids:
                raise ValueError(
                    'Repair budget guard failed: could not place remaining stratified budget slots. '
                    f'exp_idx={exp_idx}, remaining_slots={remaining_slots}.'
                )
            selected_class_id = min(
                candidate_class_ids,
                key=lambda class_id: (
                    -(
                        class_target_counts[class_id]
                        - float(selected_count_by_class[class_id])
                    ),
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
                key=lambda local_idx: self._sample_score(
                    seed=self.seed,
                    exp_idx=exp_idx,
                    class_id=class_id,
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

        suffix = (
            f'{NS_SEP}{EXPERIENCE_KEY_PREFIX}{int(exp_idx):03d}'
            if exp_idx is not None
            else f'{NS_SEP}final'
        )
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

    def _ingest_repair_dataset(self, *, experience: object) -> Dataset | None:
        """
        Resolve and store the budgeted repair subset for one experience.

        Args:
            experience (object): Avalanche experience.

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
        inputs: object,
        backbone_outputs: object,
    ) -> object:
        """
        Apply controller correction while enforcing anti-cheat invariants.

        Args:
            model (nn.Module): Backbone model.
            inputs (object): Batch inputs.
            backbone_outputs (object): Backbone outputs.

        Returns:
            object: Corrected outputs.
        """
        corrected_outputs = self.controller.correct_outputs(
            outputs=backbone_outputs,
            model=model,
            inputs=inputs,
        )
        if not (torch.is_tensor(backbone_outputs) and torch.is_tensor(corrected_outputs)):
            return corrected_outputs
        if backbone_outputs.ndim != 2 or corrected_outputs.ndim != 2:
            return corrected_outputs

        # Normalize device/dtype and validate batch compatibility.
        if int(corrected_outputs.shape[0]) != int(backbone_outputs.shape[0]):
            raise RuntimeError(
                'Repair controller output batch dimension mismatch. '
                f'backbone_batch={int(backbone_outputs.shape[0])}, '
                f'controller_batch={int(corrected_outputs.shape[0])}'
            )
        if corrected_outputs.device != backbone_outputs.device:
            corrected_outputs = corrected_outputs.to(device=backbone_outputs.device)
        if corrected_outputs.dtype != backbone_outputs.dtype:
            corrected_outputs = corrected_outputs.to(dtype=backbone_outputs.dtype)

        # Enforce anti-cheat constraints for output width and seen-class coverage.
        backbone_width = int(backbone_outputs.shape[1])
        corrected_width = int(corrected_outputs.shape[1])
        if corrected_width > backbone_width:
            raise RuntimeError(
                'Repair controller output width exceeds backbone output width. '
                f'backbone_width={backbone_width}, controller_width={corrected_width}'
            )

        if self._train_seen_classes:
            max_seen_class = int(max(self._train_seen_classes))
            if max_seen_class >= backbone_width:
                raise RuntimeError(
                    'Seen class ID exceeds backbone output width. '
                    f'max_seen_class={max_seen_class}, backbone_width={backbone_width}'
                )
            if max_seen_class >= corrected_width:
                raise RuntimeError(
                    'Repair controller output does not cover all seen classes. '
                    f'max_seen_class={max_seen_class}, controller_width={corrected_width}'
                )

        # Merge only seen-class columns so unseen classes remain backbone-owned.
        merged_outputs = backbone_outputs.clone()
        seen_class_ids = sorted(
            cls for cls in self._train_seen_classes
            if 0 <= int(cls) < backbone_width and int(cls) < corrected_width
        )
        if seen_class_ids:
            merged_outputs[:, seen_class_ids] = corrected_outputs[:, seen_class_ids]
        return merged_outputs

    def after_eval_forward(self, strategy: BaseTemplate, **kwargs) -> None:
        # Apply controller output correction while enforcing anti-cheat invariants.
        del kwargs
        if not self.eval_correction_enabled:
            return
        model = strategy.model
        if not isinstance(model, nn.Module):
            raise TypeError('Strategy.model must be an nn.Module.')
        if (
            self._should_stash_ref_backbone_logits(strategy=strategy)
            and torch.is_tensor(strategy.mb_output)
        ):
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
    def _resolve_repair_dataset(experience: object) -> Dataset | None:
        """
        Resolve the repair dataset for the current experience from the benchmark streams.

        Args:
            experience (object): Avalanche experience instance.

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
            return repair_exp._dataset
        return None

    @staticmethod
    def _resolve_new_classes(experience: object, repair_dataset: Dataset | None) -> list[int]:
        """
        Resolve newly introduced class IDs for the current experience.

        Args:
            experience (object): Avalanche experience instance.
            repair_dataset (Dataset | None): Repair dataset for the experience.

        Returns:
            list[int]: Sorted class IDs introduced in the current experience.
        """
        del repair_dataset
        return _sorted_unique_class_ids_for_experience(experience)

    @staticmethod
    def _resolve_training_classes(experience: object) -> list[int]:
        """
        Resolve class IDs present in a training experience.

        Args:
            experience (object): Avalanche experience instance.

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
        self._eval_batch_idx = 0

    def before_training(self, strategy: BaseTemplate, **kwargs) -> None:
        del strategy, kwargs
        self._train_batch_idx = 0

    def before_eval(self, strategy: BaseTemplate, **kwargs) -> None:
        del strategy, kwargs
        self._eval_batch_idx = 0

    def before_eval_exp(self, strategy: BaseTemplate, **kwargs) -> None:
        del strategy, kwargs
        self._eval_batch_idx = 0

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
                first_non_finite_value = (
                    param.detach()[mask].reshape(-1)[0].to(device='cpu').item()
                )
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

    def after_eval_forward(self, strategy: BaseTemplate, **kwargs) -> None:
        del kwargs
        self._eval_batch_idx += 1
        context = self._error_context(
            strategy=strategy,
            batch_idx=self._eval_batch_idx,
        )
        self._assert_finite_value(
            value=strategy.mb_output,
            tensor_name='mb_output',
            context=context,
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
        if (
            experience is not None
            and hasattr(experience, 'current_experience')
            and getattr(experience, 'current_experience') is not None
        ):
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
            first_non_finite_value = (
                value.detach()[mask].reshape(-1)[0].to(device='cpu').item()
            )
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
        raise RuntimeError(
            'Non-finite tensor detected. '
            f'tensor={payload["tensor_name"]}, '
            f'non_finite_count={payload["non_finite_count"]}, '
            f'phase={payload["phase"]}, '
            f'exp_idx={payload["exp_idx"]}, '
            f'step={payload["step"]}, '
            f'batch={payload["batch"]}, '
            f'eval_tag={payload["eval_tag"]}'
        )


@dataclass(frozen=True)
class _TensorStateSignature:
    """
    Signature used for low-overhead mutation detection during evaluation.

    Args:
        tensor_id (int): Python object identifier for the tensor.
        data_ptr (int): Data pointer for the underlying storage.
        shape (tuple[int, ...]): Tensor shape.
        dtype (torch.dtype): Tensor dtype.
        device (torch.device): Tensor device.
        version (int): PyTorch internal tensor version counter.
    """

    tensor_id: int
    data_ptr: int
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    version: int


class EvaluationIntegrityPlugin(SupervisedPlugin):
    """
    Enforce global evaluation integrity and anti-mutation constraints.
    """

    def __init__(self, *, controller_plugin: ControllerPlugin | None = None) -> None:
        """
        Initialize the evaluation integrity plugin.

        Args:
            controller_plugin (ControllerPlugin | None): Optional controller plugin used in the run.

        Returns:
            None.
        """
        super().__init__()
        self._controller_plugin = controller_plugin
        self._tracked_modules: dict[str, nn.Module] = {}
        self._fast_signatures: dict[str, dict[str, _TensorStateSignature]] = {}
        self._exact_snapshots: dict[str, dict[str, torch.Tensor]] = {}

    def before_eval(self, strategy: BaseTemplate, **kwargs) -> None:
        # Start evaluation with a clean integrity snapshot state.
        del strategy, kwargs
        self._clear_snapshots()

    def before_eval_exp(self, strategy: BaseTemplate, **kwargs) -> None:
        # Resolve protected modules and rebuild per-experience baselines.
        del kwargs
        self._tracked_modules = self._resolve_tracked_modules(strategy=strategy)
        self._fast_signatures = {}
        self._exact_snapshots = {}
        # Keep both fast signatures and exact values for layered checks.
        for module_name, module in self._tracked_modules.items():
            self._fast_signatures[module_name] = self._build_fast_signature(module=module)
            self._exact_snapshots[module_name] = self._build_exact_snapshot(module=module)

    def after_eval_forward(self, strategy: BaseTemplate, **kwargs) -> None:
        # Validate the global output/label contract on final evaluation tensors.
        del kwargs
        outputs = strategy.mb_output
        targets = strategy.mb_y

        # Validate output tensor shape and rank.
        if not torch.is_tensor(outputs):
            raise RuntimeError(
                'Evaluation integrity violation: `strategy.mb_output` must be a tensor.'
            )
        if outputs.ndim != 2:
            raise RuntimeError(
                'Evaluation integrity violation: `strategy.mb_output` must be 2D logits. '
                f'observed_shape={tuple(outputs.shape)}'
            )
        non_finite_count = int(torch.sum(~torch.isfinite(outputs)).item())
        if non_finite_count > 0:
            raise RuntimeError(
                'Evaluation integrity violation: `strategy.mb_output` contains non-finite values. '
                f'non_finite_count={non_finite_count}'
            )

        # Validate target tensor type and batch alignment.
        if not torch.is_tensor(targets):
            raise RuntimeError(
                'Evaluation integrity violation: `strategy.mb_y` must be a tensor of class indices.'
            )
        target_vector = targets.reshape(-1) if targets.ndim > 0 else targets.view(1)
        batch_size = int(outputs.shape[0])
        target_batch = int(target_vector.shape[0])
        if target_batch != batch_size:
            raise RuntimeError(
                'Evaluation integrity violation: target batch size must match output batch size. '
                f'output_batch={batch_size}, target_batch={target_batch}, '
                f'target_shape={tuple(targets.shape)}'
            )

        # Validate integer target values and class-range bounds.
        if torch.is_floating_point(target_vector) or torch.is_complex(target_vector):
            raise RuntimeError(
                'Evaluation integrity violation: `strategy.mb_y` must use integer class indices. '
                f'observed_dtype={targets.dtype}'
            )
        if target_vector.numel() == 0:
            return
        num_classes = int(outputs.shape[1])
        invalid_mask = (target_vector < 0) | (target_vector >= num_classes)
        invalid_count = int(torch.sum(invalid_mask).item())
        if invalid_count > 0:
            min_target = int(torch.min(target_vector).item())
            max_target = int(torch.max(target_vector).item())
            raise RuntimeError(
                'Evaluation integrity violation: target class indices are out of range. '
                f'invalid_count={invalid_count}, target_min={min_target}, '
                f'target_max={max_target}, num_classes={num_classes}'
            )

    def after_eval_iteration(self, strategy: BaseTemplate, **kwargs) -> None:
        # Run cheap mutation checks after each evaluation iteration.
        del strategy, kwargs
        self._assert_all_fast_signatures_unchanged()

    def after_eval(self, strategy: BaseTemplate, **kwargs) -> None:
        # Clear all evaluation snapshots at the end of the eval loop.
        del strategy, kwargs
        self._clear_snapshots()

    def after_eval_exp(self, strategy: BaseTemplate, **kwargs) -> None:
        # Run fast and exact checks, then clear baselines even on failure.
        del strategy, kwargs
        try:
            self._assert_all_fast_signatures_unchanged()
            self._assert_all_exact_snapshots_unchanged()
        finally:
            self._clear_snapshots()

    def _resolve_tracked_modules(self, *, strategy: BaseTemplate) -> dict[str, nn.Module]:
        """
        Resolve modules to protect from mutation during evaluation.

        Args:
            strategy (BaseTemplate): Avalanche strategy.

        Returns:
            dict[str, nn.Module]: Named tracked modules.
        """
        model = strategy.model
        if not isinstance(model, nn.Module):
            raise TypeError('Strategy.model must be an nn.Module.')
        modules: dict[str, nn.Module] = {'model': model}

        controller_module = self._resolve_controller_module(strategy=strategy)
        if controller_module is not None:
            modules['controller'] = controller_module
        return modules

    def _resolve_controller_module(self, *, strategy: BaseTemplate) -> nn.Module | None:
        """
        Resolve an optional controller module from configured or attached plugins.

        Args:
            strategy (BaseTemplate): Avalanche strategy.

        Returns:
            nn.Module | None: Controller module if present.
        """
        plugin = self._controller_plugin
        if plugin is None:
            plugin = self._find_controller_plugin(strategy=strategy)
        if plugin is None:
            return None
        controller = getattr(plugin, 'controller', None)
        if not isinstance(controller, nn.Module):
            return None
        return controller

    @staticmethod
    def _find_controller_plugin(*, strategy: BaseTemplate) -> ControllerPlugin | None:
        """
        Find the first attached controller plugin from strategy plugins.

        Args:
            strategy (BaseTemplate): Avalanche strategy.

        Returns:
            ControllerPlugin | None: Attached controller plugin if present.
        """
        plugins = getattr(strategy, 'plugins', None)
        if plugins is None:
            return None
        for plugin in plugins:
            if isinstance(plugin, (PreventionControllerPlugin, RepairControllerPlugin)):
                return plugin
        return None

    @staticmethod
    def _named_state_tensors(*, module: nn.Module) -> dict[str, torch.Tensor]:
        """
        Enumerate named parameters and buffers for a module.

        Args:
            module (nn.Module): Module to inspect.

        Returns:
            dict[str, torch.Tensor]: Named tensors keyed by prefixed state names.
        """
        named_tensors: dict[str, torch.Tensor] = {}
        for name, parameter in module.named_parameters():
            named_tensors[f'parameter:{name}'] = parameter
        for name, buffer in module.named_buffers():
            named_tensors[f'buffer:{name}'] = buffer
        return named_tensors

    @staticmethod
    def _build_fast_signature(*, module: nn.Module) -> dict[str, _TensorStateSignature]:
        """
        Build a low-overhead signature snapshot for mutation checks.

        Args:
            module (nn.Module): Module to inspect.

        Returns:
            dict[str, _TensorStateSignature]: Fast signatures keyed by state tensor name.
        """
        signature: dict[str, _TensorStateSignature] = {}
        for name, tensor in EvaluationIntegrityPlugin._named_state_tensors(module=module).items():
            signature[name] = _TensorStateSignature(
                tensor_id=int(id(tensor)),
                data_ptr=int(tensor.data_ptr()),
                shape=tuple(int(dim) for dim in tensor.shape),
                dtype=tensor.dtype,
                device=tensor.device,
                version=int(getattr(tensor, '_version', -1)),
            )
        return signature

    @staticmethod
    def _build_exact_snapshot(*, module: nn.Module) -> dict[str, torch.Tensor]:
        """
        Build an exact CPU tensor snapshot for final deep comparison.

        Args:
            module (nn.Module): Module to inspect.

        Returns:
            dict[str, torch.Tensor]: Cloned CPU tensor snapshots keyed by state tensor name.
        """
        snapshot: dict[str, torch.Tensor] = {}
        for name, tensor in EvaluationIntegrityPlugin._named_state_tensors(module=module).items():
            snapshot[name] = tensor.detach().cpu().clone()
        return snapshot

    def _assert_all_fast_signatures_unchanged(self) -> None:
        """
        Assert that tracked module fast signatures have not changed.

        Returns:
            None.
        """
        if not self._tracked_modules:
            return
        for module_name, module in self._tracked_modules.items():
            self._assert_fast_signature_unchanged(module_name=module_name, module=module)

    def _assert_fast_signature_unchanged(self, *, module_name: str, module: nn.Module) -> None:
        """
        Assert fast signature invariants for one tracked module.

        Args:
            module_name (str): Tracked module name.
            module (nn.Module): Tracked module instance.

        Returns:
            None.
        """
        baseline = self._fast_signatures.get(module_name)
        if baseline is None:
            raise RuntimeError(
                'Evaluation integrity violation: missing fast signature baseline. '
                f'module={module_name}'
            )
        current = self._build_fast_signature(module=module)

        baseline_keys = set(baseline.keys())
        current_keys = set(current.keys())
        if baseline_keys != current_keys:
            missing_keys = sorted(baseline_keys - current_keys)
            new_keys = sorted(current_keys - baseline_keys)
            raise RuntimeError(
                'Evaluation integrity violation: state tensor membership changed during evaluation. '
                f'module={module_name}, missing_keys={missing_keys}, new_keys={new_keys}'
            )

        for tensor_name, current_signature in current.items():
            baseline_signature = baseline[tensor_name]
            if current_signature == baseline_signature:
                continue

            changed_fields: list[str] = []
            if current_signature.tensor_id != baseline_signature.tensor_id:
                changed_fields.append('tensor_id')
            if current_signature.data_ptr != baseline_signature.data_ptr:
                changed_fields.append('data_ptr')
            if current_signature.shape != baseline_signature.shape:
                changed_fields.append('shape')
            if current_signature.dtype != baseline_signature.dtype:
                changed_fields.append('dtype')
            if current_signature.device != baseline_signature.device:
                changed_fields.append('device')
            if current_signature.version != baseline_signature.version:
                changed_fields.append('version')

            raise RuntimeError(
                'Evaluation integrity violation: state tensor signature changed during evaluation. '
                f'module={module_name}, tensor={tensor_name}, changed_fields={changed_fields}'
            )

    def _assert_all_exact_snapshots_unchanged(self) -> None:
        """
        Assert exact tensor-value equality for tracked modules.

        Returns:
            None.
        """
        if not self._tracked_modules:
            return
        for module_name, module in self._tracked_modules.items():
            self._assert_exact_snapshot_unchanged(module_name=module_name, module=module)

    def _assert_exact_snapshot_unchanged(self, *, module_name: str, module: nn.Module) -> None:
        """
        Assert exact snapshot invariants for one tracked module.

        Args:
            module_name (str): Tracked module name.
            module (nn.Module): Tracked module instance.

        Returns:
            None.
        """
        baseline = self._exact_snapshots.get(module_name)
        if baseline is None:
            raise RuntimeError(
                'Evaluation integrity violation: missing exact snapshot baseline. '
                f'module={module_name}'
            )
        current = self._build_exact_snapshot(module=module)

        baseline_keys = set(baseline.keys())
        current_keys = set(current.keys())
        if baseline_keys != current_keys:
            missing_keys = sorted(baseline_keys - current_keys)
            new_keys = sorted(current_keys - baseline_keys)
            raise RuntimeError(
                'Evaluation integrity violation: exact snapshot membership changed during evaluation. '
                f'module={module_name}, missing_keys={missing_keys}, new_keys={new_keys}'
            )

        for tensor_name, current_value in current.items():
            baseline_value = baseline[tensor_name]
            # Fast path for exact bitwise equality.
            if torch.equal(current_value, baseline_value):
                continue

            # Treat aligned NaN values as unchanged for floating/complex tensors.
            if self._tensors_equal_for_snapshot(
                current_value=current_value,
                baseline_value=baseline_value,
            ):
                continue

            max_abs_delta = self._compute_snapshot_max_abs_delta(
                current_value=current_value,
                baseline_value=baseline_value,
            )
            raise RuntimeError(
                'Evaluation integrity violation: state tensor values changed during evaluation. '
                f'module={module_name}, tensor={tensor_name}, max_abs_delta={max_abs_delta}'
            )

    @staticmethod
    def _tensors_equal_for_snapshot(
        *,
        current_value: torch.Tensor,
        baseline_value: torch.Tensor,
    ) -> bool:
        """
        Check exact snapshot equality while treating aligned NaN values as equal.

        Args:
            current_value (torch.Tensor): Current tensor value.
            baseline_value (torch.Tensor): Baseline tensor value.

        Returns:
            bool: True when tensors are unchanged under snapshot semantics.
        """
        if current_value.shape != baseline_value.shape:
            return False
        if current_value.dtype != baseline_value.dtype:
            return False

        # Floating and complex tensors use exact tolerance with NaN-equivalence.
        if torch.is_floating_point(current_value) or torch.is_complex(current_value):
            return bool(
                torch.allclose(
                    current_value,
                    baseline_value,
                    rtol=0.0,
                    atol=0.0,
                    equal_nan=True,
                )
            )

        return bool(torch.equal(current_value, baseline_value))

    @staticmethod
    def _compute_snapshot_max_abs_delta(
        *,
        current_value: torch.Tensor,
        baseline_value: torch.Tensor,
    ) -> float | None:
        """
        Compute a stable max absolute delta used in immutability error messages.

        Args:
            current_value (torch.Tensor): Current tensor value.
            baseline_value (torch.Tensor): Baseline tensor value.

        Returns:
            float | None: Max absolute difference, or None for non-numeric tensors.
        """
        if not (torch.is_floating_point(current_value) or torch.is_complex(current_value)):
            return None

        # Compute on CPU with a high-precision dtype for stable diagnostics.
        target_dtype = torch.complex128 if torch.is_complex(current_value) else torch.float64
        current_cpu = current_value.detach().to(device='cpu', dtype=target_dtype)
        baseline_cpu = baseline_value.detach().to(device='cpu', dtype=target_dtype)
        delta = torch.abs(current_cpu - baseline_cpu)
        delta = torch.nan_to_num(delta, nan=float('inf'))
        if delta.numel() == 0:
            return 0.0
        return float(torch.max(delta).item())

    def _clear_snapshots(self) -> None:
        """
        Clear all cached module snapshots after evaluation.

        Returns:
            None.
        """
        self._tracked_modules = {}
        self._fast_signatures = {}
        self._exact_snapshots = {}


class SeenClassesMaskPlugin(SupervisedPlugin):
    """
    Maintain a set of seen classes and optionally mask unseen logits during evaluation.
    """

    def __init__(self, mask_value: float = -1e9) -> None:
        super().__init__()
        self.seen_classes: set[int] = set()
        self.mask_value: float = mask_value
        self.mask_enabled: bool = False

    def enable_masking(self) -> None:
        """
        Enable masking of unseen classes during evaluation.
        """
        self.mask_enabled = True

    def disable_masking(self) -> None:
        """
        Disable masking of unseen classes during evaluation.
        """
        self.mask_enabled = False

    def before_training_exp(self, strategy: BaseTemplate, **kwargs) -> None:
        # Update the seen-class set from the upcoming training dataset.
        experience = strategy.experience
        dataset = None
        if experience is not None:
            if hasattr(experience, 'dataset'):
                dataset = experience.dataset
            elif hasattr(experience, '_dataset'):
                dataset = experience._dataset

        if dataset is None:
            return

        targets = extract_targets(dataset)
        self.seen_classes.update(targets)

    def after_eval_forward(self, strategy: BaseTemplate, **kwargs) -> None:
        # Optionally mask logits for classes that have not been seen in training.
        if not self.mask_enabled:
            return
        outputs = strategy.mb_output
        if not torch.is_tensor(outputs) or outputs.ndim != 2:
            return
        # Build mask indices from the current output width and seen-class cache.
        num_classes = outputs.shape[1]
        unseen_classes = [cls for cls in range(num_classes) if cls not in self.seen_classes]
        if not unseen_classes:
            return
        outputs[:, unseen_classes] = self.mask_value
        strategy.mb_output = outputs


class CalibrationDiagnosticsPlugin(SupervisedPlugin):
    """
    Collect per-task calibration and diagnostic metrics from evaluation minibatches.

    Calibration metrics:
        `calibration.nll`, `calibration.brier`, `calibration.ece`,
        `calibration.aece`, `calibration.mce`, and pass-level
        `calibration.max_ece`.

    Diagnostic metrics:
        `run.diagnostics.out_of_task_rate`, `run.diagnostics.avg_conf`,
        `run.diagnostics.avg_entropy`, and
        `run.diagnostics.logit_avg_drift` (L2 norm between exp and final-base
        task-level mean logit vectors).
    """

    def __init__(self, *, num_bins: int = 15) -> None:
        """
        Initialize the plugin.

        Args:
            num_bins (int): Number of bins used for ECE/AECE/MCE.
        """
        super().__init__()
        self.num_bins = int(num_bins)
        if self.num_bins <= 0:
            raise ValueError('`num_bins` must be positive.')

        self._eval_tag: str = ''
        self._checkpoint_exp_idx: int | None = None
        self._capture_auxiliary_metrics: bool = True
        self._current_eval_metrics: dict[int, dict[str, object]] = {}
        self._current_exp_stats: dict[str, object] | None = None
        self._current_exp_idx: int | None = None

        self._latest_eval_metrics: dict[int, dict[str, object]] = {}
        self._ref_logit_means: dict[int, np.ndarray] = {}
        self._base_logit_means: dict[int, np.ndarray] = {}
        self._base_diagnostics: dict[int, dict[str, float]] = {}

    @staticmethod
    def _resolve_log_step(*, strategy: BaseTemplate) -> int:
        context = strategy._regain_metric_context
        return int(context.log_step)

    @staticmethod
    def _log_metric(*, key: str, value: float, step: int) -> None:
        if mlflow.active_run() is None:
            return
        mlflow.log_metric(key=key, value=float(value), step=int(step))

    @staticmethod
    def _exp_metric_key(*, base_key: str, exp_idx: int) -> str:
        return f'{base_key}{NS_SEP}{EXPERIENCE_KEY_PREFIX}{int(exp_idx):03d}'

    @staticmethod
    def _ece_and_mce(
        *,
        confidences: np.ndarray,
        correctness: np.ndarray,
        num_bins: int,
    ) -> tuple[float, float]:
        """
        Compute fixed-width expected and maximum calibration errors.

        ECE is computed over equal-width confidence bins in `[0, 1]` as
        `sum_b (|B_b|/n) * |acc(B_b) - conf(B_b)|`. MCE is
        `max_b |acc(B_b) - conf(B_b)|`.

        Args:
            confidences (np.ndarray): Max predicted probabilities.
            correctness (np.ndarray): Binary correctness indicators.
            num_bins (int): Number of equal-width bins.

        Returns:
            tuple[float, float]: `(ece, mce)`.
        """
        if confidences.size == 0:
            return 0.0, 0.0

        ece = 0.0
        mce = 0.0
        n = float(confidences.size)
        for bin_idx in range(int(num_bins)):
            lower = float(bin_idx) / float(num_bins)
            upper = float(bin_idx + 1) / float(num_bins)
            if bin_idx == 0:
                mask = (confidences >= lower) & (confidences <= upper)
            else:
                mask = (confidences > lower) & (confidences <= upper)
            if not np.any(mask):
                continue
            bin_acc = float(np.mean(correctness[mask]))
            bin_conf = float(np.mean(confidences[mask]))
            gap = abs(bin_acc - bin_conf)
            ece += (float(np.sum(mask)) / n) * gap
            mce = max(mce, gap)
        return float(ece), float(mce)

    @staticmethod
    def _adaptive_ece(
        *,
        confidences: np.ndarray,
        correctness: np.ndarray,
        num_bins: int,
    ) -> float:
        """
        Compute adaptive ECE using approximately equal-count bins.

        Args:
            confidences (np.ndarray): Max predicted probabilities.
            correctness (np.ndarray): Binary correctness indicators.
            num_bins (int): Number of equal-count bins.

        Returns:
            float: Adaptive expected calibration error.
        """
        if confidences.size == 0:
            return 0.0
        order = np.argsort(confidences)
        bins = np.array_split(order, int(num_bins))
        n = float(confidences.size)
        aece = 0.0
        for idxs in bins:
            if idxs.size == 0:
                continue
            bin_acc = float(np.mean(correctness[idxs]))
            bin_conf = float(np.mean(confidences[idxs]))
            aece += (float(idxs.size) / n) * abs(bin_acc - bin_conf)
        return float(aece)

    @staticmethod
    def _empty_exp_stats(*, class_ids: set[int]) -> dict[str, object]:
        return {
            'class_ids': set(class_ids),
            'n': 0,
            'nll_sum': 0.0,
            'brier_sum': 0.0,
            'conf_sum': 0.0,
            'entropy_sum': 0.0,
            'in_task_sum': 0.0,
            'conf_chunks': [],
            'corr_chunks': [],
            'logit_sum': None,
        }

    def before_eval(self, strategy: BaseTemplate, **kwargs) -> None:
        del kwargs
        self._eval_tag = str(getattr(strategy, '_regain_eval_tag', '') or '')
        capture_context = getattr(strategy, '_regain_prediction_capture_context', None)
        checkpoint_exp_idx_raw = None
        if isinstance(capture_context, Mapping):
            self._capture_auxiliary_metrics = bool(
                capture_context.get('capture_auxiliary_metrics', True)
            )
            checkpoint_exp_idx_raw = capture_context.get('checkpoint_exp_idx')
        else:
            self._capture_auxiliary_metrics = True
        self._checkpoint_exp_idx = (
            int(checkpoint_exp_idx_raw)
            if checkpoint_exp_idx_raw is not None
            else None
        )
        self._current_eval_metrics = {}
        self._current_exp_stats = None
        self._current_exp_idx = None

    def before_eval_exp(self, strategy: BaseTemplate, **kwargs) -> None:
        del kwargs
        if not self._capture_auxiliary_metrics:
            return
        experience = strategy.experience
        exp_idx = int(experience.current_experience)
        class_ids = {int(class_id) for class_id in experience.classes_in_this_experience}
        self._current_exp_idx = exp_idx
        self._current_exp_stats = self._empty_exp_stats(class_ids=class_ids)

    def after_eval_iteration(self, strategy: BaseTemplate, **kwargs) -> None:
        del kwargs
        if not self._capture_auxiliary_metrics:
            return
        if self._current_exp_stats is None:
            return

        logits = strategy.mb_output
        targets = strategy.mb_y
        if not torch.is_tensor(logits) or logits.ndim != 2:
            return
        if not torch.is_tensor(targets):
            return

        targets_vec = targets.reshape(-1).to(device=logits.device, dtype=torch.long)
        if int(targets_vec.shape[0]) != int(logits.shape[0]):
            return
        if targets_vec.numel() == 0:
            return

        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            conf, preds = torch.max(probs, dim=1)
            corr = preds.eq(targets_vec).to(dtype=torch.float32)
            p_true = probs.gather(1, targets_vec.unsqueeze(1)).squeeze(1).clamp(min=1e-12)
            nll_sum = float(torch.sum(-torch.log(p_true)).item())
            one_hot = torch.nn.functional.one_hot(
                targets_vec,
                num_classes=int(probs.shape[1]),
            ).to(dtype=probs.dtype)
            brier_sum = float(torch.sum(torch.sum((probs - one_hot) ** 2, dim=1)).item())
            entropy = -torch.sum(probs * torch.log(probs.clamp(min=1e-12)), dim=1)

            class_ids = self._current_exp_stats['class_ids']
            in_task_sum = 0.0
            if class_ids:
                in_task_mask = torch.zeros_like(preds, dtype=torch.bool)
                for class_id in class_ids:
                    in_task_mask |= preds.eq(int(class_id))
                in_task_sum = float(torch.sum(in_task_mask).item())

            logit_sum_tensor = torch.sum(logits.detach(), dim=0).to(device='cpu', dtype=torch.float64)
            existing_logit_sum = self._current_exp_stats['logit_sum']
            if existing_logit_sum is None:
                self._current_exp_stats['logit_sum'] = logit_sum_tensor
            else:
                self._current_exp_stats['logit_sum'] = existing_logit_sum + logit_sum_tensor

            self._current_exp_stats['n'] += int(targets_vec.shape[0])
            self._current_exp_stats['nll_sum'] += nll_sum
            self._current_exp_stats['brier_sum'] += brier_sum
            self._current_exp_stats['conf_sum'] += float(torch.sum(conf).item())
            self._current_exp_stats['entropy_sum'] += float(torch.sum(entropy).item())
            self._current_exp_stats['in_task_sum'] += in_task_sum
            self._current_exp_stats['conf_chunks'].append(conf.detach().cpu())
            self._current_exp_stats['corr_chunks'].append(corr.detach().cpu())

    def after_eval_exp(self, strategy: BaseTemplate, **kwargs) -> None:
        del kwargs
        if not self._capture_auxiliary_metrics:
            return
        if self._current_exp_stats is None or self._current_exp_idx is None:
            return

        n = int(self._current_exp_stats['n'])
        if n <= 0:
            return

        conf_chunks = self._current_exp_stats['conf_chunks']
        corr_chunks = self._current_exp_stats['corr_chunks']
        confidences = torch.cat(conf_chunks).numpy() if conf_chunks else np.asarray([], dtype=np.float64)
        correctness = torch.cat(corr_chunks).numpy() if corr_chunks else np.asarray([], dtype=np.float64)

        ece, mce = self._ece_and_mce(
            confidences=confidences,
            correctness=correctness,
            num_bins=self.num_bins,
        )
        aece = self._adaptive_ece(
            confidences=confidences,
            correctness=correctness,
            num_bins=self.num_bins,
        )

        nll = float(self._current_exp_stats['nll_sum']) / float(n)
        brier = float(self._current_exp_stats['brier_sum']) / float(n)
        mean_conf = float(self._current_exp_stats['conf_sum']) / float(n)
        mean_entropy = float(self._current_exp_stats['entropy_sum']) / float(n)
        class_ids = self._current_exp_stats['class_ids']
        out_of_task_rate = None
        if class_ids:
            out_of_task_rate = 1.0 - (float(self._current_exp_stats['in_task_sum']) / float(n))
        logit_sum_tensor = self._current_exp_stats['logit_sum']
        logit_mean = (
            logit_sum_tensor / float(n)
            if isinstance(logit_sum_tensor, torch.Tensor)
            else None
        )

        exp_idx = int(self._current_exp_idx)
        metrics_payload: dict[str, object] = {
            RUN_CALIB_ECE: float(ece),
            RUN_CALIB_AECE: float(aece),
            RUN_CALIB_MCE: float(mce),
            RUN_CALIB_NLL: float(nll),
            RUN_CALIB_BRIER: float(brier),
            RUN_DIAG_AVG_CONF: float(mean_conf),
            RUN_DIAG_AVG_ENTROPY: float(mean_entropy),
            'logit_mean': (
                logit_mean.detach().cpu().numpy()
                if isinstance(logit_mean, torch.Tensor)
                else None
            ),
        }
        if out_of_task_rate is not None:
            metrics_payload[RUN_DIAG_OUT_OF_TASK_RATE] = float(out_of_task_rate)

        self._current_eval_metrics[exp_idx] = metrics_payload

        step = self._resolve_log_step(strategy=strategy)
        for key in (
            RUN_CALIB_ECE,
            RUN_CALIB_AECE,
            RUN_CALIB_MCE,
            RUN_CALIB_NLL,
            RUN_CALIB_BRIER,
        ):
            self._log_metric(
                key=self._exp_metric_key(base_key=key, exp_idx=exp_idx),
                value=float(metrics_payload[key]),
                step=step,
            )

        if self._eval_tag == 'base':
            for pred_key in (
                RUN_DIAG_OUT_OF_TASK_RATE,
                RUN_DIAG_AVG_CONF,
                RUN_DIAG_AVG_ENTROPY,
            ):
                pred_value = metrics_payload.get(pred_key)
                if pred_value is None:
                    continue
                self._log_metric(
                    key=self._exp_metric_key(base_key=pred_key, exp_idx=exp_idx),
                    value=float(pred_value),
                    step=step,
                )

        if self._eval_tag == 'ref' and metrics_payload['logit_mean'] is not None:
            self._ref_logit_means[exp_idx] = np.asarray(metrics_payload['logit_mean'], dtype=np.float64)
        if (
            self._eval_tag == 'base'
            and self._checkpoint_exp_idx is not None
            and int(exp_idx) == int(self._checkpoint_exp_idx)
            and metrics_payload['logit_mean'] is not None
        ):
            self._ref_logit_means[exp_idx] = np.asarray(metrics_payload['logit_mean'], dtype=np.float64)
        if self._eval_tag == 'base':
            if metrics_payload['logit_mean'] is not None:
                self._base_logit_means[exp_idx] = np.asarray(metrics_payload['logit_mean'], dtype=np.float64)
            self._base_diagnostics[exp_idx] = {
                RUN_DIAG_AVG_CONF: float(metrics_payload[RUN_DIAG_AVG_CONF]),
                RUN_DIAG_AVG_ENTROPY: float(metrics_payload[RUN_DIAG_AVG_ENTROPY]),
                RUN_CALIB_ECE: float(metrics_payload[RUN_CALIB_ECE]),
                RUN_CALIB_AECE: float(metrics_payload[RUN_CALIB_AECE]),
                RUN_CALIB_NLL: float(metrics_payload[RUN_CALIB_NLL]),
            }
            if RUN_DIAG_OUT_OF_TASK_RATE in metrics_payload:
                self._base_diagnostics[exp_idx][RUN_DIAG_OUT_OF_TASK_RATE] = float(
                    metrics_payload[RUN_DIAG_OUT_OF_TASK_RATE]
                )

    def after_eval(self, strategy: BaseTemplate, **kwargs) -> None:
        del kwargs
        if not self._capture_auxiliary_metrics:
            self._checkpoint_exp_idx = None
            return
        self._latest_eval_metrics = {
            int(exp_idx): dict(values)
            for exp_idx, values in self._current_eval_metrics.items()
        }
        ece_values = [
            float(values[RUN_CALIB_ECE])
            for values in self._current_eval_metrics.values()
            if RUN_CALIB_ECE in values
        ]
        if ece_values:
            step = self._resolve_log_step(strategy=strategy)
            self._log_metric(
                key=RUN_CALIB_MAX_ECE,
                value=float(max(ece_values)),
                step=step,
            )

        if self._eval_tag == 'base':
            step = self._resolve_log_step(strategy=strategy)
            common_idxs = sorted(set(self._ref_logit_means).intersection(self._base_logit_means))
            for exp_idx in common_idxs:
                ref_mean = self._ref_logit_means[exp_idx]
                base_mean = self._base_logit_means[exp_idx]
                drift = float(np.linalg.norm(ref_mean - base_mean, ord=2))
                self._base_diagnostics.setdefault(exp_idx, {})[RUN_DIAG_LOGIT_AVG_DRIFT] = drift
                self._log_metric(
                    key=self._exp_metric_key(base_key=RUN_DIAG_LOGIT_AVG_DRIFT, exp_idx=exp_idx),
                    value=drift,
                    step=step,
                )
        self._checkpoint_exp_idx = None

    def base_diagnostic_vectors(self, *, expected_len: int) -> dict[str, list[float | None]]:
        """
        Build diagnostic vectors from controller-off final evaluation.

        Args:
            expected_len (int): Expected number of experiences.

        Returns:
            dict[str, list[float | None]]: Diagnostic vectors keyed by metric base
                names. `run.diagnostics.logit_avg_drift` is the L2 drift
                `||mu_exp - mu_base||_2` between task-level mean logit vectors.
        """
        keys = (
            RUN_DIAG_OUT_OF_TASK_RATE,
            RUN_DIAG_AVG_CONF,
            RUN_DIAG_AVG_ENTROPY,
            RUN_CALIB_ECE,
            RUN_CALIB_AECE,
            RUN_CALIB_NLL,
            RUN_DIAG_LOGIT_AVG_DRIFT,
        )
        vectors: dict[str, list[float | None]] = {
            key: [None for _ in range(int(expected_len))]
            for key in keys
        }

        for exp_idx, payload in self._base_diagnostics.items():
            if exp_idx < 0 or exp_idx >= int(expected_len):
                continue
            for key in keys:
                value = payload.get(key)
                vectors[key][exp_idx] = float(value) if value is not None else None

        for exp_idx in range(int(expected_len)):
            if vectors[RUN_DIAG_LOGIT_AVG_DRIFT][exp_idx] is not None:
                continue
            ref_mean = self._ref_logit_means.get(exp_idx)
            base_mean = self._base_logit_means.get(exp_idx)
            if ref_mean is None or base_mean is None:
                continue
            vectors[RUN_DIAG_LOGIT_AVG_DRIFT][exp_idx] = float(
                np.linalg.norm(ref_mean - base_mean, ord=2)
            )

        return vectors

    def latest_max_ece(self) -> float | None:
        """
        Return worst-task ECE from the latest completed evaluation pass.

        Returns:
            float | None: Maximum value over per-task
                `run.calibration.ece.exp###` in the latest completed eval call.
        """
        ece_values = [
            float(values[RUN_CALIB_ECE])
            for values in self._latest_eval_metrics.values()
            if RUN_CALIB_ECE in values
        ]
        if not ece_values:
            return None
        return float(max(ece_values))


class PredictionLoggingPlugin(SupervisedPlugin):
    """
    Persist per-experience evaluation logits and targets as compressed artifacts.

    The plugin listens to evaluation lifecycle hooks and writes one `.npz` file per
    evaluated test experience when REGAIN-managed evaluation attaches capture metadata
    to the strategy.
    """

    def __init__(
        self,
        *,
        artifact_root: Path,
        num_classes: int,
    ) -> None:
        """
        Initialize the prediction artifact plugin.

        Args:
            artifact_root (Path): Directory where prediction artifacts are staged.
            num_classes (int): Model output width expected for stored logits.
        """
        super().__init__()
        self.artifact_root = Path(artifact_root)
        self.num_classes = int(num_classes)
        if self.num_classes <= 0:
            raise ValueError('`num_classes` must be positive.')

        self._written_files: set[str] = set()
        self._capture_context: dict[str, object] | None = None
        self._current_exp_idx: int | None = None
        self._current_class_ids: list[int] = []
        self._current_logits_chunks: list[np.ndarray] = []
        self._current_targets_chunks: list[np.ndarray] = []
        self._derived_ref_test_accuracy_cache: dict[tuple[str, int], float] = {}
        self._current_ref_cache_key: tuple[str, int] | None = None
        self._current_ref_enabled: bool = False
        self._current_ref_seen_class_ids: list[int] = []
        self._current_ref_use_backbone_logits: bool = False
        self._current_ref_mask_value: float = -1e9
        self._current_ref_correct: int = 0
        self._current_ref_total: int = 0

    def has_artifacts(self) -> bool:
        """
        Check whether any prediction artifact files were written.

        Returns:
            bool: True when at least one prediction file has been staged.
        """
        return bool(self._written_files)

    def pop_derived_ref_test_accuracy(
        self,
        *,
        eval_tag: str,
        checkpoint_exp_idx: int,
    ) -> float | None:
        """
        Read and clear one derived current-test reference accuracy value.

        Args:
            eval_tag (str): Evaluation tag for the source evaluation call.
            checkpoint_exp_idx (int): Checkpoint experience index for the source
                evaluation call.

        Returns:
            float | None: Derived masked current-test reference accuracy, or None
                when no value is cached for the requested evaluation call.
        """
        return self._derived_ref_test_accuracy_cache.pop(
            (str(eval_tag), int(checkpoint_exp_idx)),
            None,
        )

    def _reset_current_experience(self) -> None:
        """
        Reset buffers for the current evaluated experience.

        Returns:
            None.
        """
        self._current_exp_idx = None
        self._current_class_ids = []
        self._current_logits_chunks = []
        self._current_targets_chunks = []
        self._current_ref_cache_key = None
        self._current_ref_enabled = False
        self._current_ref_seen_class_ids = []
        self._current_ref_use_backbone_logits = False
        self._current_ref_mask_value = -1e9
        self._current_ref_correct = 0
        self._current_ref_total = 0

    @staticmethod
    def _ref_cache_key_from_context(
        capture_context: Mapping[str, object] | None,
    ) -> tuple[str, int] | None:
        """
        Resolve the derived-reference cache key for one evaluation call.

        Args:
            capture_context (Mapping[str, object] | None): Active prediction
                capture context.

        Returns:
            tuple[str, int] | None: `(eval_tag, checkpoint_exp_idx)` when the
                evaluation call is configured to derive a current-test reference
                value, else None.
        """
        if not isinstance(capture_context, Mapping):
            return None
        if capture_context.get('ref_test_exp_idx') is None:
            return None

        eval_tag = str(capture_context.get('eval_tag') or '').strip()
        checkpoint_exp_idx = capture_context.get('checkpoint_exp_idx')
        if eval_tag == '' or checkpoint_exp_idx is None:
            return None
        return eval_tag, int(checkpoint_exp_idx)

    @staticmethod
    def _count_masked_ref_correct(
        *,
        logits: torch.Tensor,
        targets: torch.Tensor,
        seen_class_ids: Sequence[int],
        mask_value: float,
    ) -> tuple[int, int]:
        """
        Count correct predictions after applying seen-class masking semantics.

        Args:
            logits (torch.Tensor): Unmasked logits for one minibatch.
            targets (torch.Tensor): Integer class targets aligned to `logits`.
            seen_class_ids (Sequence[int]): Class ids that remain unmasked.
            mask_value (float): Logit value written into masked columns.

        Returns:
            tuple[int, int]: `(num_correct, num_samples)` for the masked top-1
                predictions.
        """
        if logits.ndim != 2:
            return 0, 0

        targets_vec = targets.reshape(-1).to(device=logits.device, dtype=torch.long)
        if int(targets_vec.shape[0]) != int(logits.shape[0]):
            return 0, 0
        if targets_vec.numel() <= 0:
            return 0, 0

        num_classes = int(logits.shape[1])
        seen_class_id_set = {
            int(class_id)
            for class_id in seen_class_ids
            if 0 <= int(class_id) < num_classes
        }
        unseen_class_ids = [
            class_id
            for class_id in range(num_classes)
            if class_id not in seen_class_id_set
        ]
        if unseen_class_ids:
            masked_logits = logits.detach().clone()
            masked_logits[:, unseen_class_ids] = float(mask_value)
        else:
            masked_logits = logits

        predictions = torch.argmax(masked_logits, dim=1)
        num_correct = int(torch.sum(predictions.eq(targets_vec)).item())
        return num_correct, int(targets_vec.shape[0])

    @staticmethod
    def _coerce_capture_context(strategy: BaseTemplate) -> dict[str, object] | None:
        """
        Resolve prediction-capture metadata from the strategy.

        Args:
            strategy (BaseTemplate): Strategy currently being evaluated.

        Returns:
            dict[str, object] | None: Capture metadata or None when disabled.
        """
        context = getattr(strategy, '_regain_prediction_capture_context', None)
        if not isinstance(context, Mapping):
            return None

        if not bool(context.get('capture_predictions', True)):
            return None

        eval_tag = str(context.get('eval_tag') or '').strip()
        checkpoint_exp_idx = context.get('checkpoint_exp_idx')
        if eval_tag == '' or checkpoint_exp_idx is None:
            return None

        checkpoint_exp_idx_int = int(checkpoint_exp_idx)
        if checkpoint_exp_idx_int < 0:
            return None

        capture_context: dict[str, object] = {
            'eval_tag': eval_tag,
            'checkpoint_exp_idx': checkpoint_exp_idx_int,
            'mask_enabled': bool(context.get('mask_enabled', False)),
        }
        ref_test_exp_idx = context.get('ref_test_exp_idx')
        if ref_test_exp_idx is not None:
            ref_seen_class_ids = context.get('ref_seen_class_ids', [])
            if (
                not isinstance(ref_seen_class_ids, Sequence)
                or isinstance(ref_seen_class_ids, (str, bytes))
            ):
                raise ValueError('`ref_seen_class_ids` must be a numeric sequence.')
            capture_context['ref_test_exp_idx'] = int(ref_test_exp_idx)
            capture_context['ref_seen_class_ids'] = [
                int(class_id)
                for class_id in ref_seen_class_ids
            ]
            capture_context['ref_use_backbone_logits'] = bool(
                context.get('ref_use_backbone_logits', False)
            )
            capture_context['ref_mask_value'] = float(
                context.get('ref_mask_value', -1e9)
            )
        return capture_context

    @staticmethod
    def _artifact_relative_path(
        *,
        eval_tag: str,
        checkpoint_exp_idx: int,
        test_exp_idx: int,
    ) -> Path:
        """
        Build the relative artifact path for one evaluation experience.

        Args:
            eval_tag (str): Evaluation tag such as `base` or `ctrl`.
            checkpoint_exp_idx (int): Checkpoint experience index.
            test_exp_idx (int): Evaluated test experience index.

        Returns:
            Path: Relative artifact path under the predictions root directory.
        """
        return Path(str(eval_tag)) / (
            f'test_{EXPERIENCE_KEY_PREFIX}{int(test_exp_idx):03d}'
            f'_after_{EXPERIENCE_KEY_PREFIX}{int(checkpoint_exp_idx):03d}.npz'
        )

    def before_eval(self, strategy: BaseTemplate, **kwargs) -> None:
        """
        Initialize capture state for an evaluation pass.

        Args:
            strategy (BaseTemplate): Strategy currently being evaluated.

        Returns:
            None.
        """
        del kwargs
        self._capture_context = self._coerce_capture_context(strategy)
        cache_key = self._ref_cache_key_from_context(self._capture_context)
        if cache_key is not None:
            self._derived_ref_test_accuracy_cache.pop(cache_key, None)
        self._reset_current_experience()

    def before_eval_exp(self, strategy: BaseTemplate, **kwargs) -> None:
        """
        Initialize buffers for one evaluated experience.

        Args:
            strategy (BaseTemplate): Strategy currently being evaluated.

        Returns:
            None.
        """
        del kwargs
        self._reset_current_experience()
        if self._capture_context is None:
            return

        experience = getattr(strategy, 'experience', None)
        exp_idx = getattr(experience, 'current_experience', None)
        if exp_idx is None:
            return

        self._current_exp_idx = int(exp_idx)
        self._current_class_ids = _sorted_unique_class_ids_for_experience(
            getattr(strategy, 'experience', None),
        )
        ref_test_exp_idx = self._capture_context.get('ref_test_exp_idx')
        if ref_test_exp_idx is None:
            return
        if int(ref_test_exp_idx) != self._current_exp_idx:
            return

        self._current_ref_cache_key = self._ref_cache_key_from_context(
            self._capture_context,
        )
        self._current_ref_enabled = True
        self._current_ref_seen_class_ids = list(
            self._capture_context.get('ref_seen_class_ids', []),
        )
        self._current_ref_use_backbone_logits = bool(
            self._capture_context.get('ref_use_backbone_logits', False),
        )
        self._current_ref_mask_value = float(
            self._capture_context.get('ref_mask_value', -1e9),
        )

    def after_eval_iteration(self, strategy: BaseTemplate, **kwargs) -> None:
        """
        Append logits and targets for the current evaluation minibatch.

        Args:
            strategy (BaseTemplate): Strategy currently being evaluated.

        Returns:
            None.
        """
        del kwargs
        try:
            if self._capture_context is None or self._current_exp_idx is None:
                return

            logits = strategy.mb_output
            targets = strategy.mb_y
            if not torch.is_tensor(logits) or logits.ndim != 2:
                return
            if not torch.is_tensor(targets):
                return

            targets_vec = targets.reshape(-1).to(device=logits.device, dtype=torch.long)
            if int(targets_vec.shape[0]) != int(logits.shape[0]):
                return
            if int(logits.shape[1]) != self.num_classes:
                raise ValueError(
                    'Prediction artifact width mismatch. '
                    f'expected={self.num_classes}, observed={int(logits.shape[1])}'
                )
            if targets_vec.numel() == 0:
                return

            if self._current_ref_enabled:
                ref_logits: torch.Tensor | None = logits
                if self._current_ref_use_backbone_logits:
                    ref_logits_raw = getattr(
                        strategy,
                        _REF_BACKBONE_LOGITS_ATTR,
                        None,
                    )
                    if torch.is_tensor(ref_logits_raw):
                        ref_logits = ref_logits_raw
                    else:
                        ref_logits = None
                if torch.is_tensor(ref_logits):
                    ref_targets = targets.reshape(-1).to(
                        device=ref_logits.device,
                        dtype=torch.long,
                    )
                    num_correct, num_samples = self._count_masked_ref_correct(
                        logits=ref_logits,
                        targets=ref_targets,
                        seen_class_ids=self._current_ref_seen_class_ids,
                        mask_value=self._current_ref_mask_value,
                    )
                    self._current_ref_correct += int(num_correct)
                    self._current_ref_total += int(num_samples)

            self._current_logits_chunks.append(
                logits.detach().to(device='cpu', dtype=torch.float32).numpy()
            )
            self._current_targets_chunks.append(
                targets_vec.detach().to(device='cpu', dtype=torch.int32).numpy()
            )
        finally:
            if hasattr(strategy, _REF_BACKBONE_LOGITS_ATTR):
                delattr(strategy, _REF_BACKBONE_LOGITS_ATTR)

    def after_eval_exp(self, strategy: BaseTemplate, **kwargs) -> None:
        """
        Flush one evaluated experience to a compressed `.npz` artifact.

        Args:
            strategy (BaseTemplate): Strategy currently being evaluated.

        Returns:
            None.
        """
        del strategy
        del kwargs
        if (
            self._current_ref_enabled
            and self._current_ref_cache_key is not None
            and self._current_ref_total > 0
        ):
            self._derived_ref_test_accuracy_cache[self._current_ref_cache_key] = (
                float(self._current_ref_correct) / float(self._current_ref_total)
            )
        if self._capture_context is None or self._current_exp_idx is None:
            return
        if not self._current_logits_chunks or not self._current_targets_chunks:
            self._reset_current_experience()
            return

        logits = np.concatenate(self._current_logits_chunks, axis=0).astype(np.float32, copy=False)
        targets = np.concatenate(self._current_targets_chunks, axis=0).astype(np.int32, copy=False)
        relative_path = self._artifact_relative_path(
            eval_tag=str(self._capture_context['eval_tag']),
            checkpoint_exp_idx=int(self._capture_context['checkpoint_exp_idx']),
            test_exp_idx=int(self._current_exp_idx),
        )
        output_path = self.artifact_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            output_path,
            logits=logits,
            targets=targets,
            class_ids=np.asarray(self._current_class_ids, dtype=np.int32),
        )
        self._written_files.add(relative_path.as_posix())
        self._reset_current_experience()

    def after_eval(self, strategy: BaseTemplate, **kwargs) -> None:
        """
        Clear capture state after an evaluation pass completes.

        Args:
            strategy (BaseTemplate): Strategy currently being evaluated.

        Returns:
            None.
        """
        if hasattr(strategy, _REF_BACKBONE_LOGITS_ATTR):
            delattr(strategy, _REF_BACKBONE_LOGITS_ATTR)
        del kwargs
        self._capture_context = None
        self._reset_current_experience()


class RegainEvaluationPlugin(SupervisedPlugin):
    """
    Run checkpoint evaluations and log analysis artifacts.
    """

    def __init__(
        self,
        *,
        benchmark: NCScenario,
        controller_plugin: ControllerPlugin | None,
        calibration_plugin: CalibrationDiagnosticsPlugin | None,
        repair_after_experience: bool,
        seen_mask_plugin: SeenClassesMaskPlugin,
        prediction_logging_plugin: PredictionLoggingPlugin,
        num_epochs_per_experience: int,
        context: MetricContext,
        backbone_analysis_baseline: Mapping[str, Sequence[float | None]] | None = None,
        eps: float = 1e-4,
    ) -> None:
        """
        Initialize the REGAIN evaluation plugin.

        Args:
            benchmark (NCScenario): Benchmark scenario used for analysis artifacts.
            controller_plugin (ControllerPlugin | None): Controller plugin attached to the strategy.
            calibration_plugin (CalibrationDiagnosticsPlugin | None): Plugin that tracks calibration and
                diagnostic metrics during evaluation.
            repair_after_experience (bool): Whether repair fitting occurs after each experience.
            seen_mask_plugin (SeenClassesMaskPlugin): Plugin used to mask unseen classes.
            prediction_logging_plugin (PredictionLoggingPlugin): Plugin that writes prediction artifacts.
            num_epochs_per_experience (int): Number of epochs per experience.
            context (MetricContext): Metric context for logging.
            backbone_analysis_baseline (Mapping[str, Sequence[float | None]] | None):
                Optional controller-off baseline vectors from the reserved backbone
                run. When present, expected keys are `acc.exp.base` and
                `acc.final.base`.
            eps (float): Threshold for retrieval-correctable fraction calculations.

        Returns:
            None.
        """
        super().__init__()
        self.benchmark = benchmark
        self.controller_plugin = controller_plugin
        self.calibration_plugin = calibration_plugin
        self.repair_after_experience = bool(repair_after_experience)
        self.seen_mask_plugin = seen_mask_plugin
        self.prediction_logging_plugin = prediction_logging_plugin
        self.num_epochs_per_experience = int(num_epochs_per_experience)
        self.context = context
        self.eps = eps
        self.a_exp_base: list[float] = []
        self._num_experiences = len(self.benchmark.test_stream)
        self.artifacts: dict[str, object] | None = None
        self.last_posthoc_scalar_results: dict[str, float] | None = None
        self.last_base_eval_results: dict[str, object] | None = None
        self.last_ctrl_eval_results: dict[str, object] | None = None
        self.last_posthoc_exp_idx: int | None = None
        self._backbone_a_exp_base: list[float] | None = None
        self._backbone_a_base: list[float] | None = None
        self._backbone_diag_vectors: dict[str, list[float | None]] | None = None
        if backbone_analysis_baseline is not None:
            self._backbone_a_exp_base = self._coerce_backbone_vector(
                baseline=backbone_analysis_baseline,
                key=ARTIFACT_ACC_EXP_BASE,
                expected_len=self._num_experiences,
            )
            self._backbone_a_base = self._coerce_backbone_vector(
                baseline=backbone_analysis_baseline,
                key=ARTIFACT_ACC_FINAL_BASE,
                expected_len=self._num_experiences,
            )
            if isinstance(self.controller_plugin, RepairControllerPlugin):
                diag_vectors: dict[str, list[float | None]] = {}
                for diag_key in DIAG_VECTOR_KEYS:
                    diag_vectors[diag_key] = self._coerce_required_nullable_backbone_vector(
                        baseline=backbone_analysis_baseline,
                        key=diag_key,
                        expected_len=self._num_experiences,
                    )
                self._backbone_diag_vectors = diag_vectors

        if isinstance(self.controller_plugin, RepairControllerPlugin):
            if (
                self._backbone_a_exp_base is None
                or self._backbone_a_base is None
                or self._backbone_diag_vectors is None
            ):
                raise ValueError(
                    'Repair-controller runs require `backbone_analysis_baseline` '
                    'with baseline accuracy and diagnostic vectors.'
                )

    @staticmethod
    def _coerce_backbone_vector(
        *,
        baseline: Mapping[str, Sequence[float | None]],
        key: str,
        expected_len: int,
    ) -> list[float]:
        """
        Validate and coerce a backbone baseline vector.

        Args:
            baseline (Mapping[str, Sequence[float | None]]): Baseline payload.
            key (str): Baseline key to read.
            expected_len (int): Expected vector length.

        Returns:
            list[float]: Coerced baseline vector.
        """
        values = baseline.get(key)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(f'Backbone baseline `{key}` must be a numeric sequence.')
        vector = [float(value) for value in values]
        if len(vector) != int(expected_len):
            raise ValueError(
                f'Backbone baseline `{key}` length mismatch. '
                f'expected={int(expected_len)}, observed={len(vector)}'
            )
        return vector

    @staticmethod
    def _coerce_required_nullable_backbone_vector(
        *,
        baseline: Mapping[str, Sequence[float | None]],
        key: str,
        expected_len: int,
    ) -> list[float | None]:
        """
        Validate and coerce a required baseline vector that may include missing values.

        Args:
            baseline (Mapping[str, Sequence[float | None]]): Baseline payload.
            key (str): Baseline key to read.
            expected_len (int): Expected vector length.

        Returns:
            list[float | None]: Coerced vector.
        """
        values = baseline.get(key)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(f'Backbone baseline `{key}` must be a sequence.')
        vector: list[float | None] = []
        for idx, value in enumerate(values):
            if value is None:
                vector.append(None)
                continue
            value_float = float(value)
            if not math.isfinite(value_float):
                raise ValueError(
                    f'Backbone baseline `{key}` contains non-finite value '
                    f'at index {idx}: {value!r}'
                )
            vector.append(value_float)
        if len(vector) != int(expected_len):
            raise ValueError(
                f'Backbone baseline `{key}` length mismatch. '
                f'expected={int(expected_len)}, observed={len(vector)}'
            )
        return vector

    @staticmethod
    def _log_analysis_metric(
        *,
        key: str,
        value: float,
        step: int,
        experience: int | None = None,
        variant: str | None = None,
    ) -> None:
        full_key = str(key)
        if experience is not None:
            full_key += f'{NS_SEP}{EXPERIENCE_KEY_PREFIX}{experience:03d}'
        if variant is not None:
            full_key += f'{NS_SEP}{variant}'
        mlflow.log_metric(key=full_key, value=float(value), step=int(step))

    def _toggle_mask(self, enable: bool) -> bool:
        previous_state = self.seen_mask_plugin.mask_enabled
        if enable:
            self.seen_mask_plugin.enable_masking()
        else:
            self.seen_mask_plugin.disable_masking()
        return previous_state

    def _toggle_eval_correction(self, *, enable: bool) -> bool | None:
        """
        Toggle repair-controller output correction for a scoped evaluation run.

        Args:
            enable (bool): Whether evaluation-time correction should be enabled.

        Returns:
            bool | None: Previous correction state for repair controllers, else `None`.
        """
        controller_plugin = getattr(self, 'controller_plugin', None)
        if not isinstance(controller_plugin, RepairControllerPlugin):
            return None
        previous_state = bool(controller_plugin.eval_correction_enabled)
        if enable:
            controller_plugin.enable_eval_correction()
        else:
            controller_plugin.disable_eval_correction()
        return previous_state

    @staticmethod
    def _prediction_capture_context(
        *,
        eval_tag: str,
        checkpoint_exp_idx: int,
        mask_enabled: bool,
        capture_predictions: bool = True,
        capture_auxiliary_metrics: bool = True,
        ref_test_exp_idx: int | None = None,
        ref_seen_class_ids: Sequence[int] | None = None,
        ref_use_backbone_logits: bool = False,
        ref_mask_value: float | None = None,
    ) -> dict[str, object]:
        """
        Build evaluation metadata for prediction artifact capture.

        Args:
            eval_tag (str): Evaluation tag such as `ref`, `base`, or `ctrl`.
            checkpoint_exp_idx (int): Checkpoint experience index being evaluated.
            mask_enabled (bool): Whether seen-class masking is enabled.
            ref_test_exp_idx (int | None): Optional current test experience
                whose masked reference accuracy should be derived from this eval call.
            ref_seen_class_ids (Sequence[int] | None): Seen class ids used to
                reproduce reference masking semantics.
            ref_use_backbone_logits (bool): Whether to read pre-correction
                backbone logits from the repair-controller handoff.
            ref_mask_value (float | None): Mask value used when reproducing the
                seen-class masking semantics.

        Returns:
            dict[str, object]: JSON-serializable capture metadata.
        """
        capture_context: dict[str, object] = {
            'eval_tag': str(eval_tag),
            'checkpoint_exp_idx': int(checkpoint_exp_idx),
            'mask_enabled': bool(mask_enabled),
            'capture_predictions': bool(capture_predictions),
            'capture_auxiliary_metrics': bool(capture_auxiliary_metrics),
        }
        if ref_test_exp_idx is not None:
            capture_context['ref_test_exp_idx'] = int(ref_test_exp_idx)
            capture_context['ref_seen_class_ids'] = [
                int(class_id)
                for class_id in (ref_seen_class_ids or [])
            ]
            capture_context['ref_use_backbone_logits'] = bool(
                ref_use_backbone_logits
            )
            if ref_mask_value is not None:
                capture_context['ref_mask_value'] = float(ref_mask_value)
        return capture_context

    def _run_eval_with_state(
        self,
        strategy: BaseTemplate,
        stream: Sequence[object],
        *,
        mask_enabled: bool,
        eval_tag: str,
        checkpoint_exp_idx: int,
        capture_predictions: bool = True,
        capture_auxiliary_metrics: bool = True,
        controller_enabled: bool = True,
        ref_test_exp_idx: int | None = None,
        ref_seen_class_ids: Sequence[int] | None = None,
        ref_use_backbone_logits: bool = False,
        ref_mask_value: float | None = None,
    ) -> dict[str, object]:
        prev_mask_state = self._toggle_mask(mask_enabled)
        prev_controller_state = self._toggle_eval_correction(enable=controller_enabled)
        prev_phase = self.context.phase
        prev_log_namespace = self.context.log_namespace
        prev_log_step = self.context.log_step
        prev_log_enabled = self.context.log_enabled
        prev_eval_tag = getattr(strategy, '_regain_eval_tag', None)
        prev_capture_context = getattr(strategy, '_regain_prediction_capture_context', None)
        try:
            setattr(strategy, '_regain_eval_tag', str(eval_tag))
            setattr(
                strategy,
                '_regain_prediction_capture_context',
                self._prediction_capture_context(
                    eval_tag=eval_tag,
                    checkpoint_exp_idx=checkpoint_exp_idx,
                    mask_enabled=mask_enabled,
                    capture_predictions=capture_predictions,
                    capture_auxiliary_metrics=capture_auxiliary_metrics,
                    ref_test_exp_idx=ref_test_exp_idx,
                    ref_seen_class_ids=ref_seen_class_ids,
                    ref_use_backbone_logits=ref_use_backbone_logits,
                    ref_mask_value=ref_mask_value,
                ),
            )
            self.context.set_phase(MetricPhase.EVAL)
            self.context.set_log_namespace(NAMESPACE_EVAL)
            self.context.set_log_enabled(False)
            return strategy.eval(stream)
        finally:
            if prev_eval_tag is None:
                if hasattr(strategy, '_regain_eval_tag'):
                    delattr(strategy, '_regain_eval_tag')
            else:
                setattr(strategy, '_regain_eval_tag', prev_eval_tag)
            if prev_capture_context is None:
                if hasattr(strategy, '_regain_prediction_capture_context'):
                    delattr(strategy, '_regain_prediction_capture_context')
            else:
                setattr(strategy, '_regain_prediction_capture_context', prev_capture_context)
            self.context.set_phase(prev_phase)
            self.context.set_log_namespace(prev_log_namespace)
            self.context.set_log_step(prev_log_step)
            self.context.set_log_enabled(prev_log_enabled)
            if prev_controller_state is not None:
                self._toggle_eval_correction(enable=bool(prev_controller_state))
            self._toggle_mask(prev_mask_state)

    def _run_eval_with_logging(
        self,
        strategy: BaseTemplate,
        stream: Sequence[object],
        *,
        mask_enabled: bool,
        log_namespace: str,
        log_step: int,
        eval_tag: str,
        checkpoint_exp_idx: int,
        capture_predictions: bool = True,
        capture_auxiliary_metrics: bool = True,
        controller_enabled: bool = True,
        ref_test_exp_idx: int | None = None,
        ref_seen_class_ids: Sequence[int] | None = None,
        ref_use_backbone_logits: bool = False,
        ref_mask_value: float | None = None,
    ) -> dict[str, object]:
        """
        Run evaluation with toggled mask state and metric logging enabled.

        Args:
            strategy (BaseTemplate): Avalanche strategy to evaluate.
            stream (Sequence[object]): Evaluation stream to pass to the strategy.
            mask_enabled (bool): Whether to enable seen-class masking.
            log_namespace (str): Namespace to apply to logged metrics.
            log_step (int): Step to assign to logged metrics.
            checkpoint_exp_idx (int): Checkpoint experience index being evaluated.
            ref_test_exp_idx (int | None): Optional current test experience
                whose masked reference accuracy should be derived from this eval call.
            ref_seen_class_ids (Sequence[int] | None): Seen class ids used to
                reproduce reference masking semantics.
            ref_use_backbone_logits (bool): Whether to read pre-correction
                backbone logits from the repair-controller handoff.
            ref_mask_value (float | None): Mask value used when reproducing the
                seen-class masking semantics.

        Returns:
            dict[str, object]: Avalanche evaluation results.
        """
        prev_mask_state = self._toggle_mask(mask_enabled)
        prev_controller_state = self._toggle_eval_correction(enable=controller_enabled)
        prev_phase = self.context.phase
        prev_log_namespace = self.context.log_namespace
        prev_log_step = self.context.log_step
        prev_log_enabled = self.context.log_enabled
        prev_eval_tag = getattr(strategy, '_regain_eval_tag', None)
        prev_capture_context = getattr(strategy, '_regain_prediction_capture_context', None)
        try:
            setattr(strategy, '_regain_eval_tag', str(eval_tag))
            setattr(
                strategy,
                '_regain_prediction_capture_context',
                self._prediction_capture_context(
                    eval_tag=eval_tag,
                    checkpoint_exp_idx=checkpoint_exp_idx,
                    mask_enabled=mask_enabled,
                    capture_predictions=capture_predictions,
                    capture_auxiliary_metrics=capture_auxiliary_metrics,
                    ref_test_exp_idx=ref_test_exp_idx,
                    ref_seen_class_ids=ref_seen_class_ids,
                    ref_use_backbone_logits=ref_use_backbone_logits,
                    ref_mask_value=ref_mask_value,
                ),
            )
            self.context.set_phase(MetricPhase.EVAL)
            self.context.set_log_namespace(log_namespace)
            self.context.set_log_step(int(log_step))
            self.context.set_log_enabled(True)
            return strategy.eval(stream)
        finally:
            if prev_eval_tag is None:
                if hasattr(strategy, '_regain_eval_tag'):
                    delattr(strategy, '_regain_eval_tag')
            else:
                setattr(strategy, '_regain_eval_tag', prev_eval_tag)
            if prev_capture_context is None:
                if hasattr(strategy, '_regain_prediction_capture_context'):
                    delattr(strategy, '_regain_prediction_capture_context')
            else:
                setattr(strategy, '_regain_prediction_capture_context', prev_capture_context)
            self.context.set_phase(prev_phase)
            self.context.set_log_namespace(prev_log_namespace)
            self.context.set_log_step(prev_log_step)
            self.context.set_log_enabled(prev_log_enabled)
            if prev_controller_state is not None:
                self._toggle_eval_correction(enable=bool(prev_controller_state))
            self._toggle_mask(prev_mask_state)

    def _run_checkpoint_eval(
        self,
        *,
        strategy: BaseTemplate,
        checkpoint_exp_idx: int,
        log_step: int,
        derive_current_test_ref: bool = False,
    ) -> dict[str, float]:
        """
        Run one full-stream evaluation pass for a checkpoint and cache its scalar results.

        Args:
            strategy (BaseTemplate): Avalanche strategy to evaluate.
            checkpoint_exp_idx (int): Checkpoint index represented by the current model state.
            log_step (int): Metric step for the evaluation pass.
            derive_current_test_ref (bool): Whether to derive the masked
                current-test reference accuracy from this full-stream pass.

        Returns:
            dict[str, float]: Scalar metric results from this checkpoint evaluation pass.
        """
        checkpoint_exp_idx_int = int(checkpoint_exp_idx)
        stream = [self.benchmark.test_stream[i] for i in range(self._num_experiences)]
        has_controller = self.controller_plugin is not None
        eval_tag = 'ctrl' if has_controller else 'base'
        ref_test_exp_idx = (
            checkpoint_exp_idx_int
            if derive_current_test_ref
            else None
        )
        ref_seen_class_ids = (
            sorted(int(class_id) for class_id in self.seen_mask_plugin.seen_classes)
            if derive_current_test_ref
            else None
        )
        ref_use_backbone_logits = (
            derive_current_test_ref
            and isinstance(self.controller_plugin, RepairControllerPlugin)
        )
        ref_mask_value = (
            float(self.seen_mask_plugin.mask_value)
            if derive_current_test_ref
            else None
        )
        eval_results = self._run_eval_with_logging(
            strategy=strategy,
            stream=stream,
            mask_enabled=False,
            log_namespace=NAMESPACE_EVAL,
            log_step=int(log_step),
            eval_tag=eval_tag,
            checkpoint_exp_idx=checkpoint_exp_idx_int,
            ref_test_exp_idx=ref_test_exp_idx,
            ref_seen_class_ids=ref_seen_class_ids,
            ref_use_backbone_logits=ref_use_backbone_logits,
            ref_mask_value=ref_mask_value,
        )
        scalar_results = extract_scalar_metrics(eval_results)
        self.last_posthoc_scalar_results = scalar_results
        if has_controller:
            self.last_base_eval_results = None
            self.last_ctrl_eval_results = eval_results
        else:
            self.last_base_eval_results = eval_results
            self.last_ctrl_eval_results = None
        self.last_posthoc_exp_idx = checkpoint_exp_idx_int
        return scalar_results

    @staticmethod
    def _extract_current_experience_accuracy(
        *,
        scalar_results: Mapping[str, float],
        exp_idx: int,
        stream_name: str,
    ) -> float:
        """
        Extract top-1 accuracy for one experience from a single-stream evaluation pass.

        Args:
            scalar_results (Mapping[str, float]): Raw scalar evaluation results.
            exp_idx (int): Experience index to resolve.
            stream_name (str): Expected stream token such as `train_stream`.

        Returns:
            float: Current-experience top-1 accuracy.

        Raises:
            RuntimeError: If no matching accuracy metric is present.
        """
        exp_token = f'Exp{int(exp_idx):03d}'
        for key, value in scalar_results.items():
            key_str = str(key)
            if (
                _METRIC_TOKEN_AVALANCHE_TOP1_ACC_EXP in key_str
                and stream_name in key_str
                and exp_token in key_str
            ):
                return float(value)

        for key, value in scalar_results.items():
            key_str = str(key)
            if (
                _METRIC_TOKEN_AVALANCHE_TOP1_ACC_STREAM in key_str
                and stream_name in key_str
            ):
                return float(value)

        raise RuntimeError(
            'Missing current-experience accuracy metric. '
            f'exp_idx={int(exp_idx)}, stream={stream_name}'
        )

    def _run_current_experience_ref_eval(
        self,
        *,
        strategy: BaseTemplate,
        stream: Sequence[object],
        exp_idx: int,
        stream_name: str,
    ) -> float:
        """
        Evaluate one experience under seen-class masking and return its base accuracy.

        Args:
            strategy (BaseTemplate): Avalanche strategy to evaluate.
            stream (Sequence[object]): Single-experience stream to evaluate.
            exp_idx (int): Experience index represented by the stream.
            stream_name (str): Expected stream token such as `train_stream`.

        Returns:
            float: Reference accuracy for the requested experience.
        """
        exp_idx_int = int(exp_idx)
        eval_results = self._run_eval_with_state(
            strategy=strategy,
            stream=stream,
            mask_enabled=True,
            eval_tag='ref',
            checkpoint_exp_idx=exp_idx_int,
            capture_predictions=False,
            capture_auxiliary_metrics=False,
            controller_enabled=not isinstance(self.controller_plugin, RepairControllerPlugin),
        )
        scalar_results = extract_scalar_metrics(eval_results)
        return self._extract_current_experience_accuracy(
            scalar_results=scalar_results,
            exp_idx=exp_idx_int,
            stream_name=stream_name,
        )

    def _run_current_train_ref_eval(
        self,
        *,
        strategy: BaseTemplate,
        exp_idx: int,
    ) -> float:
        """
        Evaluate the current training experience and return reference accuracy.

        Args:
            strategy (BaseTemplate): Avalanche strategy to evaluate.
            exp_idx (int): Experience index to evaluate.

        Returns:
            float: Seen-class masked training accuracy for the current experience.
        """
        exp_idx_int = int(exp_idx)
        train_experience = self.benchmark.train_stream[exp_idx_int]
        return self._run_current_experience_ref_eval(
            strategy=strategy,
            stream=[train_experience],
            exp_idx=exp_idx_int,
            stream_name=_METRIC_TOKEN_AVALANCHE_TRAIN_STREAM,
        )

    def _run_current_test_ref_eval(
        self,
        *,
        strategy: BaseTemplate,
        exp_idx: int,
    ) -> float:
        """
        Evaluate the current test experience and return reference accuracy.

        Args:
            strategy (BaseTemplate): Avalanche strategy to evaluate.
            exp_idx (int): Experience index to evaluate.

        Returns:
            float: Seen-class masked test accuracy for the current experience.
        """
        exp_idx_int = int(exp_idx)
        test_experience = self.benchmark.test_stream[exp_idx_int]
        return self._run_current_experience_ref_eval(
            strategy=strategy,
            stream=[test_experience],
            exp_idx=exp_idx_int,
            stream_name=_METRIC_TOKEN_AVALANCHE_TEST_STREAM,
        )

    def after_training_exp(self, strategy: BaseTemplate, **kwargs) -> None:
        # Collect end-of-experience base accuracy right after each training experience.
        experience = strategy.experience
        if experience is None or not hasattr(experience, 'current_experience'):
            raise ValueError('Strategy experience is required to compute analysis metrics.')
        exp_idx_raw = experience.current_experience
        if exp_idx_raw is None:
            raise ValueError('Strategy experience index is missing.')
        exp_idx = int(exp_idx_raw)
        log_step = int((exp_idx + 1) * self.num_epochs_per_experience)
        self._run_checkpoint_eval(
            strategy=strategy,
            checkpoint_exp_idx=exp_idx,
            log_step=log_step,
            derive_current_test_ref=True,
        )
        ref_test_accuracy = self.prediction_logging_plugin.pop_derived_ref_test_accuracy(
            eval_tag='ctrl' if self.controller_plugin is not None else 'base',
            checkpoint_exp_idx=exp_idx,
        )
        if ref_test_accuracy is None:
            raise RuntimeError(
                'Missing derived current-test reference accuracy. '
                f'eval_tag={"ctrl" if self.controller_plugin is not None else "base"}, '
                f'checkpoint_exp_idx={exp_idx}'
            )
        ref_train_accuracy = self._run_current_train_ref_eval(
            strategy=strategy,
            exp_idx=exp_idx,
        )
        self.a_exp_base.append(float(ref_test_accuracy))
        self._log_analysis_metric(
            key=RUN_ACC_REF_TEST,
            value=float(ref_test_accuracy),
            step=log_step,
            experience=exp_idx,
            variant='base',
        )
        self._log_analysis_metric(
            key=RUN_ACC_REF_TRAIN,
            value=float(ref_train_accuracy),
            step=log_step,
            experience=exp_idx,
            variant='base',
        )

    @staticmethod
    def _extract_batch_inputs(batch: object) -> torch.Tensor | None:
        """
        Extract input tensors from an evaluation DataLoader batch.

        Args:
            batch (object): Batch returned by the DataLoader.

        Returns:
            torch.Tensor | None: Input tensor if available.
        """
        if torch.is_tensor(batch):
            return batch
        if isinstance(batch, (tuple, list)) and batch:
            first = batch[0]
            if torch.is_tensor(first):
                return first
        return None

    def _measure_latency_stats(
        self,
        *,
        strategy: BaseTemplate,
        controller_on: bool,
        warmup_iters: int = 5,
        timed_iters: int = 20,
    ) -> tuple[float, float] | None:
        """
        Measure latency and throughput on a fixed evaluation stream.

        The function returns:
            - `ms_per_sample`: mean wall-clock latency (milliseconds/sample)
            - `samples_per_sec`: throughput (samples/second)

        Timing uses warm-up iterations followed by timed iterations.

        Args:
            strategy (BaseTemplate): Avalanche strategy.
            controller_on (bool): Whether to include controller output correction.
            warmup_iters (int): Number of warmup iterations (not timed).
            timed_iters (int): Number of timed iterations.

        Returns:
            tuple[float, float] | None: `(ms_per_sample, samples_per_sec)` when measurable.
        """
        if len(self.benchmark.test_stream) <= 0:
            return None
        exp0 = self.benchmark.test_stream[0]
        dataset = exp0.dataset

        model = strategy.model
        if not isinstance(model, nn.Module):
            raise TypeError('Strategy.model must be an nn.Module.')
        batch_size = int(strategy.eval_mb_size)
        if batch_size <= 0:
            return None

        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        if len(loader) == 0:
            return None

        device = module_device(model, 'cpu')
        was_training = bool(model.training)
        model.eval()
        total_samples = 0
        elapsed_seconds = 0.0
        iterator = iter(loader)
        total_iters = int(warmup_iters) + int(timed_iters)
        try:
            with torch.no_grad():
                for i in range(total_iters):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        iterator = iter(loader)
                        batch = next(iterator)

                    inputs = self._extract_batch_inputs(batch)
                    if inputs is None:
                        return None
                    inputs = inputs.to(device=device)

                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    started_at = time.perf_counter()

                    outputs = model(inputs)
                    if controller_on and isinstance(self.controller_plugin, RepairControllerPlugin):
                        outputs = self.controller_plugin.apply_repair_correction(
                            model=model,
                            inputs=inputs,
                            backbone_outputs=outputs,
                        )
                    del outputs

                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    if i >= int(warmup_iters):
                        elapsed_seconds += float(time.perf_counter() - started_at)
                        total_samples += int(inputs.shape[0])
        finally:
            model.train(was_training)

        if total_samples <= 0 or elapsed_seconds <= 0.0:
            return None
        ms_per_sample = 1000.0 * float(elapsed_seconds) / float(total_samples)
        samples_per_sec = float(total_samples) / float(elapsed_seconds)
        return ms_per_sample, samples_per_sec

    def _log_latency_overhead(self, *, strategy: BaseTemplate, step: int) -> None:
        """
        Log controller-off/controller-on latency metrics.

        Logged keys:
            - `run.latency.ms_per_sample.base`
            - `run.latency.samples_per_sec.base`
            - `run.latency.ms_per_sample.ctrl` (controller runs)
            - `run.latency.samples_per_sec.ctrl` (controller runs)
            - `run.latency.ms_ratio = ctrl_ms_per_sample / base_ms_per_sample`
              (controller runs)

        Args:
            strategy (BaseTemplate): Avalanche strategy.
            step (int): MLflow metric step.
        """
        if mlflow.active_run() is None:
            return

        off_stats = self._measure_latency_stats(strategy=strategy, controller_on=False)
        if off_stats is None:
            return
        off_ms, off_sps = off_stats
        mlflow.log_metric(key=RUN_LATENCY_MS_PER_SAMPLE_BASE, value=float(off_ms), step=int(step))
        mlflow.log_metric(key=RUN_LATENCY_SAMPLES_PER_SEC_BASE, value=float(off_sps), step=int(step))

        if self.controller_plugin is None:
            return

        on_stats = self._measure_latency_stats(strategy=strategy, controller_on=True)
        if on_stats is None:
            return
        on_ms, on_sps = on_stats
        mlflow.log_metric(key=RUN_LATENCY_MS_PER_SAMPLE_CTRL, value=float(on_ms), step=int(step))
        mlflow.log_metric(key=RUN_LATENCY_SAMPLES_PER_SEC_CTRL, value=float(on_sps), step=int(step))
        if off_ms > 0.0:
            mlflow.log_metric(
                key=RUN_LATENCY_MS_RATIO,
                value=float(on_ms / off_ms),
                step=int(step),
            )

    def _diagnostic_vectors_for_artifacts(self) -> dict[str, list[float | None]]:
        """
        Resolve diagnostic vectors to persist in `analysis_artifacts.json`.

        Returns:
            dict[str, list[float | None]]: Diagnostic vectors.
        """
        if isinstance(self.controller_plugin, RepairControllerPlugin):
            if self._backbone_diag_vectors is None:
                raise RuntimeError(
                    'Repair-controller runs require backbone diagnostic vectors '
                    'for analysis artifacts.'
                )
            return {
                key: [value for value in vector]
                for key, vector in self._backbone_diag_vectors.items()
            }
        if self.calibration_plugin is None:
            raise RuntimeError(
                'CalibrationDiagnosticsPlugin is required to produce diagnostic vectors.'
            )
        return self.calibration_plugin.base_diagnostic_vectors(
            expected_len=self._num_experiences
        )

    @staticmethod
    def _max_optional_vector(
        values: Sequence[float | None] | None,
    ) -> float | None:
        """
        Compute the maximum finite value from an optional vector with missing entries.

        Args:
            values (Sequence[float | None] | None): Optional vector of scalar values.

        Returns:
            float | None: Maximum finite value when present.
        """
        if values is None:
            return None
        finite_values: list[float] = []
        for value in values:
            if value is None:
                continue
            value_float = float(value)
            if not math.isfinite(value_float):
                continue
            finite_values.append(value_float)
        if not finite_values:
            return None
        return float(max(finite_values))

    def _calibration_max_ece_for_artifacts(self) -> float:
        """
        Resolve `calib.max_ece` value for `analysis_artifacts.json`.

        Policy:
            - Repair-controller runs are baseline-only and use inherited backbone
              controller-off `calib.ece` vectors.
            - Non-repair runs use the latest completed evaluation-pass max ECE.

        Returns:
            float: Scalar value to persist.
        """
        if isinstance(self.controller_plugin, RepairControllerPlugin):
            if self._backbone_diag_vectors is None:
                raise RuntimeError(
                    'Repair-controller runs require backbone calibration vectors '
                    'to compute `calib.max_ece`.'
                )
            max_ece = self._max_optional_vector(
                self._backbone_diag_vectors.get(RUN_CALIB_ECE)
            )
            if max_ece is None:
                raise RuntimeError(
                    'Repair-controller runs require finite `calib.ece` values '
                    'to compute `calib.max_ece`.'
                )
            return float(max_ece)
        if self.calibration_plugin is None:
            raise RuntimeError(
                'CalibrationDiagnosticsPlugin is required to compute `calib.max_ece`.'
            )
        max_ece = self.calibration_plugin.latest_max_ece()
        if max_ece is None:
            raise RuntimeError('Missing `calib.max_ece` from the latest evaluation pass.')
        return float(max_ece)

    @staticmethod
    def _mean_accuracy(values: Sequence[float]) -> float:
        """
        Compute the arithmetic mean of an accuracy vector.

        Args:
            values (Sequence[float]): Accuracy values.

        Returns:
            float: Mean accuracy.
        """
        return float(sum(values) / max(1, len(values)))

    def _log_accuracy_vector(
        self,
        *,
        key: str,
        values: Sequence[float],
        variant: str,
        step: int,
    ) -> None:
        """
        Log one per-experience accuracy vector under a unified namespace.

        Args:
            key (str): Metric prefix without experience or variant suffixes.
            values (Sequence[float]): Per-experience accuracies.
            variant (str): Accuracy variant such as `base` or `ctrl`.
            step (int): MLflow metric step.

        Returns:
            None.
        """
        for exp_idx, value in enumerate(values):
            self._log_analysis_metric(
                key=key,
                value=float(value),
                step=step,
                experience=int(exp_idx),
                variant=variant,
            )

    def _evaluate_stream_accuracies(
        self,
        *,
        strategy: BaseTemplate,
        stream: Sequence[object],
        eval_tag: str,
        checkpoint_exp_idx: int,
        controller_enabled: bool,
    ) -> list[float]:
        """
        Evaluate a full stream and return ordered per-experience accuracies.

        Args:
            strategy (BaseTemplate): Avalanche strategy to evaluate.
            stream (Sequence[object]): Stream to evaluate.
            eval_tag (str): Evaluation tag such as `base` or `ctrl`.
            checkpoint_exp_idx (int): Checkpoint experience index represented by the model state.
            controller_enabled (bool): Whether repair correction should be enabled.

        Returns:
            list[float]: Ordered top-1 accuracies for the evaluated stream.
        """
        eval_results = self._run_eval_with_state(
            strategy=strategy,
            stream=stream,
            mask_enabled=False,
            eval_tag=eval_tag,
            checkpoint_exp_idx=int(checkpoint_exp_idx),
            capture_predictions=False,
            capture_auxiliary_metrics=False,
            controller_enabled=controller_enabled,
        )
        return ordered_accuracies(eval_results, self._num_experiences)

    def after_training(self, strategy: BaseTemplate, **kwargs) -> None:
        # Finalize posthoc evaluation state and validate baseline completeness.
        del kwargs
        expected = int(self._num_experiences)
        have = int(len(self.a_exp_base))
        final_step = int(self._num_experiences * self.num_epochs_per_experience)
        should_run_final = (
            self.last_posthoc_scalar_results is None
            or self.last_posthoc_exp_idx != self._num_experiences - 1
            or (
                isinstance(self.controller_plugin, RepairControllerPlugin)
                and not self.repair_after_experience
            )
        )
        if should_run_final:
            self._run_checkpoint_eval(
                strategy=strategy,
                checkpoint_exp_idx=self._num_experiences - 1,
                log_step=final_step,
            )
        if self.last_posthoc_scalar_results is None:
            raise RuntimeError('Final checkpoint scalar metrics are missing.')

        # Emit an incomplete artifact payload when reference points are missing.
        if have != expected:
            self.artifacts = {
                COLUMN_STATUS: _STATUS_INCOMPLETE_ACC_EXP_BASE,
                'expected_num_experiences': expected,
                'observed_num_exp_points': have,
                RUN_EPS: self.eps,
                ARTIFACT_ACC_EXP_BASE: [float(value) for value in self.a_exp_base],
            }
            self._log_analysis_metric(
                key=f'{NAMESPACE_RUN}{NS_SEP}status{NS_SEP}{_STATUS_INCOMPLETE_ACC_EXP_BASE}',
                value=1.0,
                step=final_step,
            )
            mlflow.log_dict(self.artifacts, MLFLOW_ARTIFACT_ANALYSIS_FILE)
            return

        final_test_base: list[float]
        if isinstance(self.controller_plugin, RepairControllerPlugin):
            final_test_base = self._evaluate_stream_accuracies(
                strategy=strategy,
                stream=self.benchmark.test_stream,
                eval_tag='base',
                checkpoint_exp_idx=self._num_experiences - 1,
                controller_enabled=False,
            )
        else:
            base_results = self.last_base_eval_results
            if base_results is None and self.controller_plugin is not None:
                base_results = self.last_ctrl_eval_results
            if base_results is None:
                base_results = self._run_eval_with_state(
                    strategy=strategy,
                    stream=self.benchmark.test_stream,
                    mask_enabled=False,
                    eval_tag='base',
                    checkpoint_exp_idx=self._num_experiences - 1,
                    capture_predictions=False,
                    capture_auxiliary_metrics=False,
                    controller_enabled=True,
                )
            final_test_base = ordered_accuracies(base_results, self._num_experiences)

        final_train_base = self._evaluate_stream_accuracies(
            strategy=strategy,
            stream=self.benchmark.train_stream,
            eval_tag='base',
            checkpoint_exp_idx=self._num_experiences - 1,
            controller_enabled=not isinstance(self.controller_plugin, RepairControllerPlugin),
        )

        log_ctrl_metrics = isinstance(self.controller_plugin, RepairControllerPlugin)
        if log_ctrl_metrics:
            ctrl_results = self.last_ctrl_eval_results
            if ctrl_results is None:
                ctrl_results = self._run_eval_with_state(
                    strategy=strategy,
                    stream=self.benchmark.test_stream,
                    mask_enabled=False,
                    eval_tag='ctrl',
                    checkpoint_exp_idx=self._num_experiences - 1,
                    capture_predictions=False,
                    capture_auxiliary_metrics=False,
                    controller_enabled=True,
                )
            final_test_ctrl = ordered_accuracies(ctrl_results, self._num_experiences)
            final_train_ctrl = self._evaluate_stream_accuracies(
                strategy=strategy,
                stream=self.benchmark.train_stream,
                eval_tag='ctrl',
                checkpoint_exp_idx=self._num_experiences - 1,
                controller_enabled=True,
            )
            a_final_ctrl = list(final_test_ctrl)
        else:
            final_test_ctrl = None
            final_train_ctrl = None
            a_final_ctrl = list(final_test_base)

        # Compute and persist full analysis artifact payload.
        diagnostic_vectors = self._diagnostic_vectors_for_artifacts()
        max_ece = self._calibration_max_ece_for_artifacts()
        extra_scalars: dict[str, float] = {RUN_CALIB_MAX_ECE: float(max_ece)}

        artifacts = build_analysis_artifacts(
            a_exp_base=self.a_exp_base,
            a_base=final_test_base,
            a_final_ctrl=a_final_ctrl,
            eps=self.eps,
            extra_vectors=diagnostic_vectors,
            extra_scalars=extra_scalars,
        )

        self.artifacts = artifacts
        mlflow.log_dict(self.artifacts, MLFLOW_ARTIFACT_ANALYSIS_FILE)
        self._log_accuracy_vector(
            key=RUN_ACC_FINAL_TEST,
            values=final_test_base,
            variant='base',
            step=final_step,
        )
        self._log_analysis_metric(
            key=RUN_ACC_FINAL_TEST_AVG_BASE,
            value=self._mean_accuracy(final_test_base),
            step=final_step,
        )
        self._log_accuracy_vector(
            key=RUN_ACC_FINAL_TRAIN,
            values=final_train_base,
            variant='base',
            step=final_step,
        )
        self._log_analysis_metric(
            key=RUN_ACC_FINAL_TRAIN_AVG_BASE,
            value=self._mean_accuracy(final_train_base),
            step=final_step,
        )

        if log_ctrl_metrics:
            if final_test_ctrl is None or final_train_ctrl is None:
                raise RuntimeError('Repair-controller final accuracy vectors are missing.')
            self._log_accuracy_vector(
                key=RUN_ACC_FINAL_TEST,
                values=final_test_ctrl,
                variant='ctrl',
                step=final_step,
            )
            self._log_analysis_metric(
                key=RUN_ACC_FINAL_TEST_AVG_CTRL,
                value=self._mean_accuracy(final_test_ctrl),
                step=final_step,
            )
            self._log_accuracy_vector(
                key=RUN_ACC_FINAL_TRAIN,
                values=final_train_ctrl,
                variant='ctrl',
                step=final_step,
            )
            self._log_analysis_metric(
                key=RUN_ACC_FINAL_TRAIN_AVG_CTRL,
                value=self._mean_accuracy(final_train_ctrl),
                step=final_step,
            )
            for i, value in enumerate(artifacts[ARTIFACT_RHO]):
                if value is None:
                    continue
                self._log_analysis_metric(
                    key=RUN_RHO,
                    value=float(value),
                    step=final_step,
                    experience=i,
                )

        # Log aggregate controller summary metrics.
        rho_avg = artifacts.get(ARTIFACT_RHO_AVG)
        if rho_avg is not None and log_ctrl_metrics:
            self._log_analysis_metric(
                key=RUN_RHO_AVG,
                value=float(rho_avg),
                step=final_step,
            )

        self._log_latency_overhead(strategy=strategy, step=final_step)


def make_evaluation_plugin(
    *,
    context: MetricContext,
    keep_timestep_results: bool = True,
    log_to_console: bool = True,
    log_to_mlflow: bool = True,
    include_forward_transfer: bool = False,
) -> EvaluationPlugin:
    """
    Create an EvaluationPlugin with standard metrics and loggers.

    Args:
        context (MetricContext): Metric context for logging.
        keep_timestep_results (bool): Whether to keep per-time-step results 
                                      (accessible via `EvaluationPlugin.get_all_metrics()`)
        log_to_console (bool): Whether to include console logging.
        log_to_mlflow (bool): Whether to include MLflow logging.
        include_forward_transfer (bool): Whether to include forward transfer metrics.

    Returns:
        EvaluationPlugin: Configured evaluation plugin.
    """
    # Set loggers
    loggers = []

    if log_to_console:
        loggers.append(InteractiveLogger())

    if log_to_mlflow:
        loggers.append(MLflowLogger(context=context))

    # Set metrics
    metrics = [
        accuracy_metrics(epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True),
    ]
    if include_forward_transfer:
        metrics.append(forward_transfer_metrics(experience=True, stream=True))

    # Create EvaluationPlugin
    return EvaluationPlugin(*metrics, loggers=loggers, collect_all=keep_timestep_results)
