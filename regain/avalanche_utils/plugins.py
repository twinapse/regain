"""
Avalanche plugins.
"""
from typing import Sequence

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
from regain.analysis.metrics import METRIC_NAMESPACE_SEPARATOR
from regain.analysis.metrics import MetricPhase
from regain.avalanche_utils.logging import MLflowLogger
from regain.avalanche_utils.scenarios import get_num_classes_from_experience
from regain.experiments.utils import EvalMode
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
    'ControllerPlugin',
    'PreventionControllerPlugin',
    'RepairControllerPlugin',
    'RegainEvaluationPlugin',
    'MetricContextPlugin',
    'SeenClassesMaskPlugin',
    'make_evaluation_plugin',
]


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
        self.context.set_log_namespace('train')
        self.context.set_log_enabled(True)
        self.context.set_experience(0)
        self.context.reset_training_counters()
        self.context.set_log_step(0)

    def before_training_exp(self, strategy, **kwargs) -> None:
        self.context.set_phase(MetricPhase.TRAIN)
        self.context.set_log_namespace('train')
        self.context.set_log_enabled(True)
        self.context.set_experience(self._exp_idx(strategy))
        self.context.reset_experience_counters()

    def before_training_epoch(self, strategy, **kwargs) -> None:
        self.context.set_phase(MetricPhase.TRAIN)
        self.context.set_log_namespace('train')
        self.context.set_log_enabled(True)
        self.context.set_experience(self._exp_idx(strategy))
        self.context.advance_training_epoch()

    def before_eval(self, strategy, **kwargs) -> None:
        self.context.set_phase(MetricPhase.EVAL)
        if self.context.log_namespace in {'train', 'eval'}:
            self.context.set_log_namespace('eval')
            self.context.set_log_step(int(self.context.train_step))
        self.context.set_epoch(0)

    def before_eval_exp(self, strategy, **kwargs) -> None:
        self.context.set_phase(MetricPhase.EVAL)
        if self.context.log_namespace in {'train', 'eval'}:
            self.context.set_log_namespace('eval')
        self.context.set_experience(self._exp_idx(strategy))
        self.context.set_epoch(0)


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
    Wire a repair controller into Avalanche strategy lifecycles with eval toggling.
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

    def enable(self) -> None:
        """
        Enable controller application during evaluation.

        Returns:
            None.
        """
        self.controller.enable()

    def disable(self) -> None:
        """
        Disable controller application during evaluation.

        Returns:
            None.
        """
        self.controller.disable()

    def is_enabled(self) -> bool:
        """
        Check if the controller is enabled for evaluation.

        Returns:
            bool: True if enabled, False otherwise.
        """
        return self.controller.is_enabled()

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
        if not self.controller.is_enabled():
            return
        self.controller.on_eval_begin(strategy)

    def after_eval(self, strategy: BaseTemplate, **kwargs):
        if not self.controller.is_enabled():
            return
        self.controller.on_eval_end(strategy)

    def before_eval_exp(self, strategy: BaseTemplate, **kwargs):
        if not self.controller.is_enabled():
            return
        self.controller.on_eval_experience_begin(strategy.experience)

    def after_eval_exp(self, strategy: BaseTemplate, **kwargs):
        if not self.controller.is_enabled():
            return
        self.controller.on_eval_experience_end(strategy.experience)

    def after_eval_forward(self, strategy: BaseTemplate, **kwargs) -> None:
        """
        Apply output correction after evaluation forward pass when enabled.

        Args:
            strategy: Avalanche strategy with minibatch outputs.

        Returns:
            None.
        """
        if not self.controller.is_enabled():
            return

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
        exp_id = int(getattr(experience, 'current_experience', 0))
        benchmark = getattr(experience, 'benchmark', None)
        if benchmark is None:
            return None

        repair_exp = None
        if hasattr(benchmark, 'repair_stream'):
            repair_exp = benchmark.repair_stream[exp_id]
        elif hasattr(benchmark, 'streams') and 'repair' in benchmark.streams:
            repair_exp = benchmark.streams['repair'][exp_id]

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
        eval_mode: EvalMode,
        repair_after_experience: bool,
        seen_mask_plugin: SeenClassesMaskPlugin,
        num_epochs_per_experience: int,
        context: MetricContext,
        eps: float = 1e-4,
    ) -> None:
        """
        Initialize the REGAIN evaluation plugin.

        Args:
            benchmark (NCScenario): Benchmark scenario used for analysis artifacts.
            controller_plugin (ControllerPlugin | None): Controller plugin attached to the strategy.
            eval_mode (EvalMode): Evaluation mode to mirror for posthoc evaluation.
            repair_after_experience (bool): Whether repair fitting occurs after each experience.
            seen_mask_plugin (SeenClassesMaskPlugin): Plugin used to mask unseen classes.
            num_epochs_per_experience (int): Number of epochs per experience.
            context (MetricContext): Metric context for logging.
            eps (float): Threshold for retrieval-correctable fraction calculations.

        Returns:
            None.
        """
        super().__init__()
        self.benchmark = benchmark
        self.controller_plugin = controller_plugin
        self.eval_mode = eval_mode
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

    @staticmethod
    def _log_analysis_metric(key: str, value: float, step: int, experience: int | None = None) -> None:
        full_key = f'analysis{METRIC_NAMESPACE_SEPARATOR}{key}'
        if experience is not None:
            full_key += f'{METRIC_NAMESPACE_SEPARATOR}exp{experience:03d}'

        mlflow.log_metric(key=full_key, value=value, step=step)

    @staticmethod
    def _log_summary_metric(key: str, value: float, step: int) -> None:
        mlflow.log_metric(key=f'summary{METRIC_NAMESPACE_SEPARATOR}{key}', value=value, step=step)

    def _set_controller_eval_state(self, enable: bool | None) -> bool | None:
        if enable is None or not isinstance(self.controller_plugin, RepairControllerPlugin):
            return None
        previous_state = self.controller_plugin.is_enabled()
        if enable:
            self.controller_plugin.enable()
        else:
            self.controller_plugin.disable()
        return previous_state

    def _restore_controller_state(self, previous_state: bool | None) -> None:
        if previous_state is None or not isinstance(self.controller_plugin, RepairControllerPlugin):
            return
        if previous_state:
            self.controller_plugin.enable()
        else:
            self.controller_plugin.disable()

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
        controller_enabled: bool | None,
        mask_enabled: bool,
    ) -> dict[str, object]:
        prev_controller_state = self._set_controller_eval_state(controller_enabled)
        prev_mask_state = self._toggle_mask(mask_enabled)
        prev_phase = self.context.phase
        prev_log_namespace = self.context.log_namespace
        prev_log_step = self.context.log_step
        prev_log_enabled = self.context.log_enabled
        try:
            self.context.set_phase(MetricPhase.EVAL)
            self.context.set_log_namespace('analysis')
            self.context.set_log_enabled(False)
            return strategy.eval(stream)
        finally:
            self.context.set_phase(prev_phase)
            self.context.set_log_namespace(prev_log_namespace)
            self.context.set_log_step(prev_log_step)
            self.context.set_log_enabled(prev_log_enabled)
            self._toggle_mask(prev_mask_state)
            self._restore_controller_state(prev_controller_state)

    def _run_eval_with_logging(
        self,
        strategy: BaseTemplate,
        stream: Sequence[object],
        *,
        controller_enabled: bool | None,
        mask_enabled: bool,
        log_namespace: str,
        log_step: int,
        run_name: str | None,
    ) -> dict[str, object]:
        """
        Run evaluation with toggled controller/mask state and metric logging enabled.

        Args:
            strategy (BaseTemplate): Avalanche strategy to evaluate.
            stream (Sequence[object]): Evaluation stream to pass to the strategy.
            controller_enabled (bool | None): Whether to enable the controller during eval.
            mask_enabled (bool): Whether to enable seen-class masking.
            log_namespace (str): Namespace to apply to logged metrics.
            log_step (int): Step to assign to logged metrics.
            run_name (str | None): Optional nested MLflow run name.

        Returns:
            dict[str, object]: Avalanche evaluation results.
        """
        prev_controller_state = self._set_controller_eval_state(controller_enabled)
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
            self._restore_controller_state(prev_controller_state)

    @staticmethod
    def _format_run_name(prefix: str, suffix: str | None) -> str:
        if suffix:
            return f'{prefix}_{suffix}'
        return prefix

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

        if self.eval_mode == 'single':
            repair_enabled = None
            if isinstance(self.controller_plugin, RepairControllerPlugin):
                repair_enabled = True
            log_namespace = 'ctrl' if self.controller_plugin is not None else 'base'
            eval_results = self._run_eval_with_logging(
                strategy=strategy,
                stream=stream,
                controller_enabled=repair_enabled,
                mask_enabled=False,
                log_namespace=log_namespace,
                log_step=log_step,
                run_name=self._format_run_name(log_namespace, run_name_suffix),
            )
            scalar_results = extract_scalar_metrics(eval_results)
            prefixed_results = {
                f'{log_namespace}_{name}': value
                for name, value in scalar_results.items()
            }
            self.last_posthoc_scalar_results = prefixed_results
            if log_namespace == 'base':
                self.last_base_eval_results = eval_results
                self.last_ctrl_eval_results = None
            else:
                self.last_base_eval_results = None
                self.last_ctrl_eval_results = eval_results
            self.last_posthoc_exp_idx = final_exp_idx
            return prefixed_results

        if self.eval_mode != 'compare':
            raise ValueError('eval_mode must be "single" or "compare".')

        if not isinstance(self.controller_plugin, RepairControllerPlugin):
            raise ValueError('eval_mode="compare" requires a toggleable repair controller.')

        base_results = self._run_eval_with_logging(
            strategy=strategy,
            stream=stream,
            controller_enabled=False,
            mask_enabled=False,
            log_namespace='base',
            log_step=log_step,
            run_name=self._format_run_name('base', run_name_suffix),
        )
        ctrl_results = self._run_eval_with_logging(
            strategy=strategy,
            stream=stream,
            controller_enabled=True,
            mask_enabled=False,
            log_namespace='ctrl',
            log_step=log_step,
            run_name=self._format_run_name('ctrl', run_name_suffix),
        )
        base_scalar = extract_scalar_metrics(base_results)
        ctrl_scalar = extract_scalar_metrics(ctrl_results)
        merged = {
            **{f'base_{name}': value for name, value in base_scalar.items()},
            **{f'ctrl_{name}': value for name, value in ctrl_scalar.items()},
        }
        self.last_posthoc_scalar_results = merged
        self.last_base_eval_results = base_results
        self.last_ctrl_eval_results = ctrl_results
        self.last_posthoc_exp_idx = final_exp_idx
        return merged

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
            run_name_suffix=f'exp{exp_idx:03d}',
        )

    def after_training_exp(self, strategy: BaseTemplate, **kwargs) -> None:
        """
        Evaluate reference accuracy for the just-trained experience with masking.

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
        ref_results = self._run_eval_with_state(
            strategy,
            [self.benchmark.test_stream[exp_idx]],
            controller_enabled=False,
            mask_enabled=True,
        )
        acc_map = extract_top1_by_experience(ref_results, self._num_experiences)
        if exp_idx not in acc_map:
            raise ValueError(f'Missing reference accuracy for experience {exp_idx}.')
        self.a_ref.append(acc_map[exp_idx])
        if mlflow.active_run() is not None:
            self._log_analysis_metric(
                key='a_ref',
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
                'status': 'incomplete_a_ref',
                'expected_num_experiences': expected,
                'observed_num_reference_points': have,
                'eps': self.eps,
                'a_ref': [float(value) for value in self.a_ref],
            }
            if mlflow.active_run() is not None:
                self._log_analysis_metric(key='incomplete_a_ref', value=1.0, step=final_step)
                mlflow.log_dict(self.artifacts, 'analysis_artifacts.json')
            return

        a_post_results = self.last_base_eval_results
        if a_post_results is None and self.controller_plugin is not None:
            if not isinstance(self.controller_plugin, RepairControllerPlugin):
                a_post_results = self.last_ctrl_eval_results
        if a_post_results is None:
            a_post_results = self._run_eval_with_state(
                strategy,
                self.benchmark.test_stream,
                controller_enabled=False,
                mask_enabled=False,
            )
        a_post = ordered_accuracies(a_post_results, self._num_experiences)

        if self.controller_plugin is None:
            a_ctrl = list(a_post)
        else:
            a_ctrl_results = self.last_ctrl_eval_results
            if a_ctrl_results is None:
                repair_enabled = isinstance(self.controller_plugin, RepairControllerPlugin)
                a_ctrl_results = self._run_eval_with_state(
                    strategy,
                    self.benchmark.test_stream,
                    controller_enabled=repair_enabled if repair_enabled else None,
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
        if mlflow.active_run() is not None:
            final_step = int(self._num_experiences * self.num_epochs_per_experience)
            for i, value in enumerate(a_post):
                self._log_analysis_metric(
                    key='a_post',
                    value=float(value),
                    step=final_step,
                    experience=i,
                )
            if log_ctrl_metrics:
                for i, value in enumerate(a_ctrl):
                    self._log_analysis_metric(
                        key='a_ctrl',
                        value=float(value),
                        step=final_step,
                        experience=i,
                    )
                for i, value in enumerate(artifacts['rho']):
                    if value is None:
                        continue
                    self._log_analysis_metric(
                        key='rho',
                        value=float(value),
                        step=final_step,
                        experience=i,
                    )

            rho_mean = artifacts.get('rho_mean', None)
            if rho_mean is not None and log_ctrl_metrics:
                self._log_analysis_metric(key='rho_mean', value=float(rho_mean), step=final_step)
                self._log_summary_metric(key='final_rho_mean', value=float(rho_mean), step=final_step)

            def _mean(values: list[float]) -> float:
                return float(sum(values) / max(1, len(values)))

            self._log_summary_metric(
                key='final_a_post_mean',
                value=_mean([float(value) for value in a_post]),
                step=final_step,
            )
            if log_ctrl_metrics:
                self._log_summary_metric(
                    key='final_a_ctrl_mean',
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
