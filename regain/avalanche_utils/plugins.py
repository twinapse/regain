"""
Avalanche plugins.
"""
from pathlib import Path
from typing import Mapping, Sequence

from avalanche.benchmarks.scenarios import NCScenario
from avalanche.core import SupervisedPlugin
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.evaluation.metrics import forgetting_metrics
from avalanche.evaluation.metrics import loss_metrics
from avalanche.evaluation.metrics import timing_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.templates import BaseTemplate
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from regain.analysis import build_analysis_artifacts
from regain.analysis import extract_top1_by_experience
from regain.analysis import MetricContext
from regain.analysis import ordered_accuracies
from regain.analysis.metrics import MetricPhase
from regain.avalanche_utils.logging import MLflowLogger
from regain.avalanche_utils.scenarios import get_num_classes_from_experience
from regain.constants import COLUMN_STATUS
from regain.constants import EXPERIENCE_KEY_PREFIX
from regain.constants import METRIC_A_CTRL
from regain.constants import METRIC_A_POST
from regain.constants import METRIC_A_REF
from regain.constants import METRIC_EPS
from regain.constants import METRIC_PREFIX_ANALYSIS
from regain.constants import METRIC_PREFIX_SUMMARY
from regain.constants import METRIC_RHO
from regain.constants import METRIC_RHO_MEAN
from regain.constants import NAMESPACE_EVAL
from regain.constants import NAMESPACE_TRAIN
from regain.constants import NS_SEP
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
    'ControllerPlugin',
    'LRSchedulerPlugin',
    'PreventionControllerPlugin',
    'RepairControllerPlugin',
    'RegainEvaluationPlugin',
    'MetricContextPlugin',
    'SeenClassesMaskPlugin',
    'make_evaluation_plugin',
]

_METRIC_FINAL_A_CTRL_MEAN = 'final_a_ctrl_mean'
_METRIC_FINAL_A_POST_MEAN = 'final_a_post_mean'
_METRIC_FINAL_RHO_MEAN = 'final_rho_mean'
_METRIC_INCOMPLETE_A_REF = 'incomplete_a_ref'
_NAMESPACE_ANALYSIS = 'analysis'
_RUN_NAME_FINAL = 'final'


class MetricContextPlugin(SupervisedPlugin):
    """
    Update MetricContext during Avalanche strategy lifecycles.
    """

    def __init__(self, context: MetricContext) -> None:
        super().__init__()
        self.context = context

    @staticmethod
    def _exp_idx(strategy: BaseTemplate, fallback: int = 0) -> int:
        exp = getattr(strategy, 'experience', None)
        idx = getattr(exp, 'current_experience', None)
        return int(idx) if isinstance(idx, int) else int(fallback)

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
        """
        Save a checkpoint for the completed experience.

        Args:
            strategy (BaseTemplate): Strategy containing model and experience index.
        """
        model = strategy.model
        if not isinstance(model, nn.Module):
            raise TypeError('Strategy.model must be an nn.Module.')

        experience = strategy.experience
        exp_idx = int(experience.current_experience)
        checkpoint_path = self._checkpoint_dir / f'exp_{exp_idx:03d}.pt'
        torch.save(
            {
                'experience': exp_idx,
                'model_state_dict': self._snapshot_model_state(model),
            },
            checkpoint_path,
        )
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
        """
        Load the checkpoint corresponding to the current experience index.

        Args:
            strategy (BaseTemplate): Strategy containing model and experience index.
        """
        model = strategy.model
        if not isinstance(model, nn.Module):
            raise TypeError('Strategy.model must be an nn.Module.')

        experience = strategy.experience
        exp_idx = int(experience.current_experience)
        if exp_idx < 0 or exp_idx >= len(self._checkpoint_paths):
            raise ValueError(
                'Missing backbone checkpoint for experience '
                f'{exp_idx}. Available indices: 0..{max(0, len(self._checkpoint_paths) - 1)}'
            )

        checkpoint_path = self._checkpoint_paths[exp_idx]
        if not checkpoint_path.exists():
            raise FileNotFoundError(f'Backbone checkpoint not found: {checkpoint_path}')

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


# TODO: Wire missing hooks
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
        """
        Run controller initialization once before training begins.

        Args:
            strategy: Avalanche strategy containing the model.

        Returns:
            None.
        """
        model = strategy.model
        if not isinstance(model, nn.Module):
            raise TypeError('Strategy.model must be an nn.Module.')

        self.controller.on_train_begin(model)

        if isinstance(self.controller, BackboneControllerInterface):
            self.controller.correct_backbone(model)

    def before_training_exp(self, strategy: BaseTemplate, **kwargs) -> None:
        """
        Forward the training experience start hook to the controller.

        Args:
            strategy: Avalanche strategy providing the current experience.

        Returns:
            None.
        """
        experience = strategy.experience
        dataset: RegainDataset | None = None
        if experience is not None:
            if hasattr(experience, 'dataset'):
                dataset = experience.dataset
            elif hasattr(experience, '_dataset'):
                dataset = experience._dataset

        self.controller.on_train_experience_begin(dataset)

    def before_training_epoch(self, strategy: BaseTemplate, **kwargs) -> None:
        """
        Forward the training epoch start hook to the controller.

        Args:
            strategy: Avalanche strategy containing the model.

        Returns:
            None.
        """
        model = strategy.model
        if not isinstance(model, nn.Module):
            raise TypeError('Strategy.model must be an nn.Module.')

        self.controller.on_train_epoch_begin(model)

    def before_backward(self, strategy: BaseTemplate, **kwargs) -> None:
        """
        Allow controllers to modify the training loss before backpropagation.

        Args:
            strategy: Avalanche strategy containing minibatch tensors and loss.

        Returns:
            None.
        """
        if not isinstance(self.controller, TrainingObjectiveControllerInterface):
            return

        model = strategy.model
        if not isinstance(model, nn.Module):
            raise TypeError('Strategy.model must be an nn.Module.')

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
        """
        Forward the training epoch end hook to the controller.

        Args:
            strategy: Avalanche strategy containing the model.

        Returns:
            None.
        """
        model = strategy.model
        if not isinstance(model, nn.Module):
            raise TypeError('Strategy.model must be an nn.Module.')

        self.controller.on_train_epoch_end(model)

    def after_training_exp(self, strategy: BaseTemplate, **kwargs) -> None:
        """
        Forward the post-training experience hook to the controller.

        Args:
            strategy: Avalanche strategy containing the trained model.

        Returns:
            None.
        """
        model = strategy.model
        if not isinstance(model, nn.Module):
            raise TypeError('Strategy.model must be an nn.Module.')

        self.controller.on_train_experience_end(model)

    def after_training(self, strategy: BaseTemplate, **kwargs) -> None:
        """
        Forward the post-training hook to the controller.

        Args:
            strategy: Avalanche strategy containing the trained model.

        Returns:
            None.
        """
        model = strategy.model
        if not isinstance(model, nn.Module):
            raise TypeError('Strategy.model must be an nn.Module.')

        self.controller.on_train_end(model)


# TODO: Be more restrictive on the arguments passed to controller hooks.
#       Controllers shouldn't have access to the whole strategy or experience object.
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
    ) -> None:
        """
        Initialize the plugin with a repair controller.

        Args:
            controller (RepairController): Controller to wire into Avalanche.
            fit_after_experience (bool): Whether to fit on repair data after each experience.
            repair_epochs (int): Number of epochs to use for repair fitting.
            repair_batch_size (int): Batch size to use for repair fitting.

        Raises:
            TypeError: If the controller is not a RepairController.

        Returns:
            None.
        """
        super().__init__()

        if not isinstance(controller, RepairController):
            raise TypeError('RepairControllerPlugin requires a RepairController.')

        if (not fit_after_experience) and controller.requires_per_experience_fitting():
            raise ValueError(f'{type(controller).__name__} requires per-experience fitting')

        self.controller: RepairController = controller
        self.repair_epochs = repair_epochs
        self.repair_batch_size = repair_batch_size
        self.fit_after_experience = fit_after_experience
        self._repair_datasets: list[Dataset] = []
        self._seen_classes: set[int] = set()

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

    def after_training_exp(self, strategy: BaseTemplate, **kwargs) -> None:
        """
        Forward the post-training experience hook to the controller and surface repair data.

        Args:
            strategy: Avalanche strategy containing the trained model.

        Returns:
            None.
        """
        # Get the experience and model
        experience = strategy.experience
        model = strategy.model
        if not isinstance(model, nn.Module):
            raise TypeError('Strategy.model must be an nn.Module.')

        # Get the repair dataset
        repair_ds = self._resolve_repair_dataset(experience)
        if repair_ds is not None:
            self._repair_datasets.append(repair_ds)

        # Keep track of the new classes
        new_classes = self._resolve_new_classes(experience, repair_ds)
        self._seen_classes.update(new_classes)

        # Notify the controller of experience end
        self.controller.on_train_experience_end(model)

        # Fit on repair data if per-experience fitting is enabled
        if self.fit_after_experience:
            combined_dataset = self._combined_repair_dataset()
            if combined_dataset is None:
                return
            self.controller.fit_on_repair_data(
                model=model,
                repair_dataset=combined_dataset,
                new_classes=new_classes,
                num_epochs=self.repair_epochs,
                batch_size=self.repair_batch_size,
            )

    def after_training(self, strategy: BaseTemplate, **kwargs) -> None:
        """
        Forward the post-training hook to the controller and fit on repair data.

        Args:
            strategy: Avalanche strategy containing the trained model.

        Returns:
            None.
        """
        # Get and validate the model
        model = strategy.model
        if not isinstance(model, nn.Module):
            raise TypeError('Strategy.model must be an nn.Module.')

        # Notify the controller of training end
        self.controller.on_train_end(model)

        # Fit on repair data (only if per-experience fitting is not done)
        if not self.fit_after_experience:
            combined_dataset = self._combined_repair_dataset()
            if combined_dataset is None:
                return
            self.controller.fit_on_repair_data(
                model=model,
                repair_dataset=combined_dataset,
                new_classes=sorted(self._seen_classes),
                num_epochs=self.repair_epochs,
                batch_size=self.repair_batch_size,
            )

    def before_eval(self, strategy: BaseTemplate, **kwargs):
        self.controller.on_eval_begin(strategy)

    def after_eval(self, strategy: BaseTemplate, **kwargs):
        self.controller.on_eval_end(strategy)

    def before_eval_exp(self, strategy: BaseTemplate, **kwargs):
        self.controller.on_eval_experience_begin(strategy.experience)

    def after_eval_exp(self, strategy: BaseTemplate, **kwargs):
        self.controller.on_eval_experience_end(strategy.experience)

    def after_eval_forward(self, strategy: BaseTemplate, **kwargs) -> None:
        """
        Apply output correction after evaluation forward pass.

        Args:
            strategy: Avalanche strategy with minibatch outputs.

        Returns:
            None.
        """
        model = strategy.model
        if not isinstance(model, nn.Module):
            raise TypeError('Strategy.model must be an nn.Module.')

        inputs = strategy.mb_x
        strategy.mb_output = self.controller.correct_outputs(
            outputs=strategy.mb_output,
            model=model,
            inputs=inputs,
        )

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
        classes = getattr(experience, 'classes_in_this_experience', None)
        if classes is not None:
            try:
                return sorted({int(c) for c in classes})
            except Exception:
                pass
        targets = extract_targets(repair_dataset)
        return sorted({int(t) for t in targets})

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
        """
        Avalanche hook: update the seen class set from the upcoming training dataset.

        Args:
            strategy: Avalanche strategy providing the dataset.
        """
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
        """
        Mask logits for unseen classes when enabled.

        Args:
            strategy: Avalanche strategy exposing minibatch outputs.
        """
        if not self.mask_enabled:
            return
        outputs = strategy.mb_output
        if not torch.is_tensor(outputs) or outputs.ndim != 2:
            return
        num_classes = outputs.shape[1]
        unseen_classes = [cls for cls in range(num_classes) if cls not in self.seen_classes]
        if not unseen_classes:
            return
        outputs[:, unseen_classes] = self.mask_value
        strategy.mb_output = outputs


class RegainEvaluationPlugin(SupervisedPlugin):
    """
    Run reference/post/controller evaluations and log analysis artifacts.
    """

    def __init__(
        self,
        *,
        benchmark: NCScenario,
        controller_plugin: ControllerPlugin | None,
        repair_after_experience: bool,
        seen_mask_plugin: SeenClassesMaskPlugin,
        num_epochs_per_experience: int,
        context: MetricContext,
        backbone_analysis_baseline: Mapping[str, Sequence[float]] | None = None,
        eps: float = 1e-4,
    ) -> None:
        """
        Initialize the REGAIN evaluation plugin.

        Args:
            benchmark (NCScenario): Benchmark scenario used for analysis artifacts.
            controller_plugin (ControllerPlugin | None): Controller plugin attached to the strategy.
            repair_after_experience (bool): Whether repair fitting occurs after each experience.
            seen_mask_plugin (SeenClassesMaskPlugin): Plugin used to mask unseen classes.
            num_epochs_per_experience (int): Number of epochs per experience.
            context (MetricContext): Metric context for logging.
            backbone_analysis_baseline (Mapping[str, Sequence[float]] | None): Optional controller-off baseline vectors
                from the reserved backbone run. When present, expected keys are `a_ref` and `a_post`.
            eps (float): Threshold for retrieval-correctable fraction calculations.

        Returns:
            None.
        """
        super().__init__()
        self.benchmark = benchmark
        self.controller_plugin = controller_plugin
        self.repair_after_experience = bool(repair_after_experience)
        self.seen_mask_plugin = seen_mask_plugin
        self.num_epochs_per_experience = int(num_epochs_per_experience)
        self.context = context
        self.eps = eps
        self.a_ref: list[float] = []
        self._num_experiences = len(self.benchmark.test_stream)
        self.artifacts: dict[str, object] | None = None
        self.last_posthoc_scalar_results: dict[str, float] | None = None
        self.last_base_eval_results: dict[str, object] | None = None
        self.last_ctrl_eval_results: dict[str, object] | None = None
        self.last_posthoc_exp_idx: int | None = None
        self._backbone_a_ref: list[float] | None = None
        self._backbone_a_post: list[float] | None = None
        if backbone_analysis_baseline is not None:
            self._backbone_a_ref = self._coerce_backbone_vector(
                baseline=backbone_analysis_baseline,
                key=METRIC_A_REF,
                expected_len=self._num_experiences,
            )
            self._backbone_a_post = self._coerce_backbone_vector(
                baseline=backbone_analysis_baseline,
                key=METRIC_A_POST,
                expected_len=self._num_experiences,
            )

        if isinstance(self.controller_plugin, RepairControllerPlugin):
            if self._backbone_a_ref is None or self._backbone_a_post is None:
                raise ValueError(
                    'Repair-controller runs require `backbone_analysis_baseline` '
                    'with `a_ref` and `a_post` vectors.'
                )

    @staticmethod
    def _coerce_backbone_vector(
        *,
        baseline: Mapping[str, Sequence[float]],
        key: str,
        expected_len: int,
    ) -> list[float]:
        """
        Validate and coerce a backbone baseline vector.

        Args:
            baseline (Mapping[str, Sequence[float]]): Baseline payload.
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
    def _log_analysis_metric(key: str, value: float, step: int, experience: int | None = None) -> None:
        full_key = f'{METRIC_PREFIX_ANALYSIS}{key}'
        if experience is not None:
            full_key += f'{NS_SEP}{EXPERIENCE_KEY_PREFIX}{experience:03d}'

        mlflow.log_metric(key=full_key, value=value, step=step)

    @staticmethod
    def _log_summary_metric(key: str, value: float, step: int) -> None:
        mlflow.log_metric(
            key=f'{METRIC_PREFIX_SUMMARY}{key}',
            value=value,
            step=step,
        )

    def _toggle_mask(self, enable: bool) -> bool:
        previous_state = self.seen_mask_plugin.mask_enabled
        if enable:
            self.seen_mask_plugin.enable_masking()
        else:
            self.seen_mask_plugin.disable_masking()
        return previous_state

    def _run_eval_with_state(
        self,
        strategy: BaseTemplate,
        stream: Sequence[object],
        *,
        mask_enabled: bool,
    ) -> dict[str, object]:
        prev_mask_state = self._toggle_mask(mask_enabled)
        prev_phase = self.context.phase
        prev_log_namespace = self.context.log_namespace
        prev_log_step = self.context.log_step
        prev_log_enabled = self.context.log_enabled
        try:
            self.context.set_phase(MetricPhase.EVAL)
            self.context.set_log_namespace(_NAMESPACE_ANALYSIS)
            self.context.set_log_enabled(False)
            return strategy.eval(stream)
        finally:
            self.context.set_phase(prev_phase)
            self.context.set_log_namespace(prev_log_namespace)
            self.context.set_log_step(prev_log_step)
            self.context.set_log_enabled(prev_log_enabled)
            self._toggle_mask(prev_mask_state)

    def _run_eval_with_logging(
        self,
        strategy: BaseTemplate,
        stream: Sequence[object],
        *,
        mask_enabled: bool,
        log_namespace: str,
        log_step: int,
        run_name: str | None,
    ) -> dict[str, object]:
        """
        Run evaluation with toggled mask state and metric logging enabled.

        Args:
            strategy (BaseTemplate): Avalanche strategy to evaluate.
            stream (Sequence[object]): Evaluation stream to pass to the strategy.
            mask_enabled (bool): Whether to enable seen-class masking.
            log_namespace (str): Namespace to apply to logged metrics.
            log_step (int): Step to assign to logged metrics.
            run_name (str | None): Optional nested MLflow run name.

        Returns:
            dict[str, object]: Avalanche evaluation results.
        """
        prev_mask_state = self._toggle_mask(mask_enabled)
        prev_phase = self.context.phase
        prev_log_namespace = self.context.log_namespace
        prev_log_step = self.context.log_step
        prev_log_enabled = self.context.log_enabled
        try:
            self.context.set_phase(MetricPhase.EVAL)
            self.context.set_log_namespace(log_namespace)
            self.context.set_log_step(int(log_step))
            self.context.set_log_enabled(True)
            if run_name is None:
                return strategy.eval(stream)
            with mlflow.start_run(nested=True, run_name=run_name):
                return strategy.eval(stream)
        finally:
            self.context.set_phase(prev_phase)
            self.context.set_log_namespace(prev_log_namespace)
            self.context.set_log_step(prev_log_step)
            self.context.set_log_enabled(prev_log_enabled)
            self._toggle_mask(prev_mask_state)

    @staticmethod
    def _format_nested_eval_run_name(*, suffix: str | None) -> str:
        """
        Build the nested run name used for post-training evaluation.

        Args:
            suffix (str | None): Optional suffix to include in the run name.

        Returns:
            str: `final` or `<suffix>`.
        """
        if suffix:
            return suffix
        return _RUN_NAME_FINAL

    def _build_posthoc_stream(self, exp_idx: int | None) -> list[object]:
        if exp_idx is None:
            return [self.benchmark.test_stream[i] for i in range(self._num_experiences)]
        return [self.benchmark.test_stream[i] for i in range(exp_idx + 1)]

    def _run_posthoc_eval(
        self,
        *,
        strategy: BaseTemplate,
        exp_idx: int | None,
        log_step: int,
        run_name_suffix: str | None,
    ) -> dict[str, float]:
        """
        Run posthoc evaluation for the requested checkpoint.

        Args:
            strategy (BaseTemplate): Avalanche strategy to evaluate.
            exp_idx (int | None): Experience index for the stream checkpoint.
            log_step (int): Logging step for the evaluation metrics.
            run_name_suffix (str | None): Optional suffix to attach to nested run names.

        Returns:
            dict[str, float]: Scalar metric results for the posthoc checkpoint.
        """
        stream = self._build_posthoc_stream(exp_idx)
        final_exp_idx = exp_idx if exp_idx is not None else self._num_experiences - 1

        has_controller = self.controller_plugin is not None
        run_name: str | None = None
        if run_name_suffix is not None:
            run_name = self._format_nested_eval_run_name(suffix=run_name_suffix)
        elif self.repair_after_experience and isinstance(self.controller_plugin, RepairControllerPlugin):
            # Only keep a nested final run when per-experience checkpoints are also logged.
            run_name = self._format_nested_eval_run_name(suffix=None)
        eval_results = self._run_eval_with_logging(
            strategy=strategy,
            stream=stream,
            mask_enabled=False,
            log_namespace='',
            log_step=log_step,
            run_name=run_name,
        )
        scalar_results = extract_scalar_metrics(eval_results)
        self.last_posthoc_scalar_results = scalar_results
        if has_controller:
            self.last_base_eval_results = None
            self.last_ctrl_eval_results = eval_results
        else:
            self.last_base_eval_results = eval_results
            self.last_ctrl_eval_results = None
        self.last_posthoc_exp_idx = final_exp_idx
        return scalar_results

    def _run_posthoc_eval_after_experience(
        self,
        *,
        strategy: BaseTemplate,
        exp_idx: int,
    ) -> dict[str, float] | None:
        """
        Run posthoc evaluation after a training experience when configured.

        Args:
            strategy (BaseTemplate): Avalanche strategy to evaluate.
            exp_idx (int): Experience index that just completed.

        Returns:
            dict[str, float] | None: Scalar results if executed, otherwise None.
        """
        if not self.repair_after_experience:
            return None
        if not isinstance(self.controller_plugin, RepairControllerPlugin):
            return None

        log_step = int((exp_idx + 1) * self.num_epochs_per_experience)
        return self._run_posthoc_eval(
            strategy=strategy,
            exp_idx=exp_idx,
            log_step=log_step,
            run_name_suffix=f'{EXPERIENCE_KEY_PREFIX}{exp_idx:03d}',
        )

    def after_training_exp(self, strategy: BaseTemplate, **kwargs) -> None:
        """
        Record reference accuracy for the just-trained experience.

        Args:
            strategy: Trained Avalanche strategy.
        """
        experience = strategy.experience
        if experience is None or not hasattr(experience, 'current_experience'):
            raise ValueError('Strategy experience is required to compute analysis metrics.')
        exp_idx_raw = experience.current_experience
        if exp_idx_raw is None:
            raise ValueError('Strategy experience index is missing.')
        exp_idx = int(exp_idx_raw)
        if isinstance(self.controller_plugin, RepairControllerPlugin):
            # Repair runs inherit controller-off reference baselines from the reserved backbone run.
            if self._backbone_a_ref is None:
                raise RuntimeError('Missing backbone `a_ref` baseline for repair-controller run.')
            self.a_ref.append(float(self._backbone_a_ref[exp_idx]))
        else:
            ref_results = self._run_eval_with_state(
                strategy,
                [self.benchmark.test_stream[exp_idx]],
                mask_enabled=True,
            )
            acc_map = extract_top1_by_experience(ref_results, self._num_experiences)
            if exp_idx not in acc_map:
                raise ValueError(f'Missing reference accuracy for experience {exp_idx}.')
            self.a_ref.append(acc_map[exp_idx])
        self._log_analysis_metric(
            key=METRIC_A_REF,
            value=float(self.a_ref[-1]),
            step=int((exp_idx + 1) * self.num_epochs_per_experience),
            experience=exp_idx,
        )
        self._run_posthoc_eval_after_experience(strategy=strategy, exp_idx=exp_idx)

    def after_training(self, strategy: BaseTemplate, **kwargs) -> None:
        """
        Log analysis artifacts after training completes.

        Args:
            strategy: Trained Avalanche strategy.
        """
        expected = int(self._num_experiences)
        have = int(len(self.a_ref))
        final_step = int(self._num_experiences * self.num_epochs_per_experience)
        should_run_final = (
            self.last_posthoc_scalar_results is None
            or self.last_posthoc_exp_idx != self._num_experiences - 1
        )
        if should_run_final:
            self._run_posthoc_eval(
                strategy=strategy,
                exp_idx=None,
                log_step=final_step,
                run_name_suffix=None,
            )

        if have != expected:
            self.artifacts = {
                COLUMN_STATUS: _METRIC_INCOMPLETE_A_REF,
                'expected_num_experiences': expected,
                'observed_num_reference_points': have,
                METRIC_EPS: self.eps,
                METRIC_A_REF: [float(value) for value in self.a_ref],
            }
            self._log_analysis_metric(key=_METRIC_INCOMPLETE_A_REF, value=1.0, step=final_step)
            mlflow.log_dict(self.artifacts, 'analysis_artifacts.json')
            return

        if isinstance(self.controller_plugin, RepairControllerPlugin):
            # Repair runs inherit controller-off post-sequence baselines from the reserved backbone run.
            if self._backbone_a_post is None:
                raise RuntimeError('Missing backbone `a_post` baseline for repair-controller run.')
            a_post = [float(value) for value in self._backbone_a_post]
        else:
            a_post_results = self.last_base_eval_results
            if a_post_results is None and self.controller_plugin is not None:
                if not isinstance(self.controller_plugin, RepairControllerPlugin):
                    a_post_results = self.last_ctrl_eval_results
            if a_post_results is None:
                a_post_results = self._run_eval_with_state(
                    strategy,
                    self.benchmark.test_stream,
                    mask_enabled=False,
                )
            a_post = ordered_accuracies(a_post_results, self._num_experiences)

        if self.controller_plugin is None:
            a_ctrl = list(a_post)
        else:
            a_ctrl_results = self.last_ctrl_eval_results
            if a_ctrl_results is None:
                a_ctrl_results = self._run_eval_with_state(
                    strategy,
                    self.benchmark.test_stream,
                    mask_enabled=False,
                )
            a_ctrl = ordered_accuracies(a_ctrl_results, self._num_experiences)

        artifacts = build_analysis_artifacts(
            a_ref=self.a_ref,
            a_post=a_post,
            a_ctrl=a_ctrl,
            eps=self.eps,
        )

        self.artifacts = artifacts
        log_ctrl_metrics = isinstance(self.controller_plugin, RepairControllerPlugin)
        final_step = int(self._num_experiences * self.num_epochs_per_experience)
        for i, value in enumerate(a_post):
            self._log_analysis_metric(
                key=METRIC_A_POST,
                value=float(value),
                step=final_step,
                experience=i,
            )
        if log_ctrl_metrics:
            for i, value in enumerate(a_ctrl):
                self._log_analysis_metric(
                    key=METRIC_A_CTRL,
                    value=float(value),
                    step=final_step,
                    experience=i,
                )
            for i, value in enumerate(artifacts[METRIC_RHO]):
                if value is None:
                    continue
                self._log_analysis_metric(
                    key=METRIC_RHO,
                    value=float(value),
                    step=final_step,
                    experience=i,
                )

        rho_mean = artifacts.get(METRIC_RHO_MEAN, None)
        if rho_mean is not None and log_ctrl_metrics:
            self._log_analysis_metric(key=METRIC_RHO_MEAN, value=float(rho_mean), step=final_step)
            self._log_summary_metric(key=_METRIC_FINAL_RHO_MEAN, value=float(rho_mean), step=final_step)

        def _mean(values: list[float]) -> float:
            return float(sum(values) / max(1, len(values)))

        self._log_summary_metric(
            key=_METRIC_FINAL_A_POST_MEAN,
            value=_mean([float(value) for value in a_post]),
            step=final_step,
        )
        if log_ctrl_metrics:
            self._log_summary_metric(
                key=_METRIC_FINAL_A_CTRL_MEAN,
                value=_mean([float(value) for value in a_ctrl]),
                step=final_step,
            )


def make_evaluation_plugin(
    *,
    context: MetricContext,
    keep_timestep_results: bool = True,
    log_to_console: bool = True,
    log_to_mlflow: bool = True,
) -> EvaluationPlugin:
    """
    Create an EvaluationPlugin with standard metrics and loggers.

    Args:
        context (MetricContext): Metric context for logging.
        keep_timestep_results (bool): Whether to keep per-time-step results 
                                      (accessible via `EvaluationPlugin.get_all_metrics()`)
        log_to_console (bool): Whether to include console logging.
        log_to_mlflow (bool): Whether to include MLflow logging.

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

    # Create EvaluationPlugin
    return EvaluationPlugin(*metrics, loggers=loggers, collect_all=keep_timestep_results)
