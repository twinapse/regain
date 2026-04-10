"""
Avalanche-dependent orchestration for REGAIN's custom evaluation loop.
"""

from collections.abc import Iterable, Mapping
import math
import time

from avalanche.benchmarks import CLExperience
from avalanche.benchmarks import CLScenario
from avalanche.benchmarks import CLStream
from avalanche.training.templates import BaseTemplate
import mlflow
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from regain.analysis import build_analysis_artifacts
from regain.analysis import MetricContext
from regain.analysis.artifacts import AnalysisArtifacts
from regain.analysis.artifacts import ARTIFACT_ACC_EXP_BASE
from regain.analysis.artifacts import ARTIFACT_ACC_FINAL_BASE
from regain.analysis.artifacts import ARTIFACT_RHO
from regain.analysis.artifacts import ARTIFACT_RHO_AVG
from regain.analysis.metrics import MetricPhase
from regain.constants import COLUMN_STATUS
from regain.constants import DIAG_VECTOR_KEYS
from regain.constants import EXPERIENCE_KEY_PREFIX
from regain.constants import MLFLOW_ARTIFACT_ANALYSIS_FILE
from regain.constants import NAMESPACE_EVAL
from regain.constants import NS_SEP
from regain.constants import RUN_ACC_FINAL
from regain.constants import RUN_ACC_FINAL_AVG_BASE
from regain.constants import RUN_ACC_FINAL_AVG_CTRL
from regain.constants import RUN_ACC_REF
from regain.constants import RUN_CALIB_ECE
from regain.constants import RUN_CALIB_MAX_ECE
from regain.constants import RUN_EPS
from regain.constants import RUN_EVAL_FORGETTING
from regain.constants import RUN_EVAL_FORGETTING_STREAM
from regain.constants import RUN_EVAL_TRANSFER
from regain.constants import RUN_EVAL_TRANSFER_STREAM
from regain.constants import RUN_LATENCY_MS_PER_SAMPLE_BASE
from regain.constants import RUN_LATENCY_MS_PER_SAMPLE_CTRL
from regain.constants import RUN_LATENCY_MS_RATIO
from regain.constants import RUN_LATENCY_SAMPLES_PER_SEC_BASE
from regain.constants import RUN_LATENCY_SAMPLES_PER_SEC_CTRL
from regain.constants import RUN_RHO
from regain.constants import RUN_RHO_AVG
from regain.constants import RUN_STATUS_INCOMPLETE_ACC_EXP_BASE
from regain.constants import RUN_TRAIN_LOSS
from regain.evaluation import CalibrationCollector
from regain.evaluation import check_eval_batch
from regain.evaluation import ClassMask
from regain.evaluation import derive_masked_ref_accuracy
from regain.evaluation import EvaluationPassResult
from regain.evaluation import ForgettingTracker
from regain.evaluation import ForwardTransferTracker
from regain.evaluation import frozen_model_state
from regain.evaluation import PredictionRecorder
from regain.mlflow_utils import normalize_metric_name
from regain.models.controllers import Controller
from regain.models.controllers import RepairController
from regain.models.controllers.repair.common import apply_repair_correction
from regain.utils import module_device

__all__ = ['RegainEvaluator']


_STATUS_INCOMPLETE_ACC_EXP_BASE = 'incomplete_acc_exp_base'


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


def _resolve_experience_dataset(experience: CLExperience) -> object:
    """
    Resolve the dataset attached to an Avalanche experience.

    Args:
        experience (CLExperience): Experience exposing `dataset` or `_dataset`.

    Returns:
        object: Experience dataset accepted by `torch.utils.data.DataLoader`.

    Raises:
        TypeError: If no dataset is available.
    """
    dataset = getattr(experience, 'dataset', None)
    if dataset is None:
        dataset = getattr(experience, '_dataset', None)
    if dataset is None:
        raise TypeError('Evaluation experience must expose a dataset.')
    if not hasattr(dataset, '__len__') or not hasattr(dataset, '__getitem__'):
        raise TypeError(
            'Evaluation experience must expose a dataset compatible with `torch.utils.data.DataLoader`.'
        )
    return dataset


def _coerce_stream(stream: CLStream | CLExperience) -> list[object]:
    """
    Normalize a stream-like input into an ordered experience list.

    Args:
        stream (CLStream | CLExperience): Single experience or stream of experiences.

    Returns:
        list[object]: Ordered experiences to evaluate.
    """
    if hasattr(stream, 'dataset') or hasattr(stream, 'current_experience'):
        return [stream]
    if isinstance(stream, Iterable) and not isinstance(stream, (str, bytes)):
        return list(stream)
    return [stream]


def _extract_batch_tensors(batch: object) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract inputs and targets from one dataloader batch.

    Args:
        batch (object): Batch returned by the dataloader.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: `(inputs, targets)`.

    Raises:
        TypeError: If the batch shape is unsupported.
    """
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        inputs = batch[0]
        targets = batch[1]
        if torch.is_tensor(inputs) and torch.is_tensor(targets):
            return inputs, targets
    raise TypeError('Evaluation batch must be a tuple/list containing tensor inputs and targets.')


class RegainEvaluator:
    """
    Custom posthoc evaluator that keeps Avalanche for training only.
    """

    def __init__(
        self,
        *,
        benchmark: CLScenario,
        model: nn.Module,
        controller: Controller | None,
        seen_classes: set[int],
        device: torch.device,
        criterion: nn.Module,
        num_classes: int,
        calibration: CalibrationCollector | None,
        prediction_recorder: PredictionRecorder | None,
        context: MetricContext,
        batch_size: int,
        num_epochs_per_experience: int,
        repair_after_experience: bool,
        include_forward_transfer: bool,
        backbone_analysis_baseline: Mapping[str, list[float | None]] | None = None,
        eps: float = 1e-4,
        mask_value: float = -1e9,
    ) -> None:
        """
        Initialize the custom evaluator.

        Args:
            benchmark (CLScenario): Benchmark exposing `train_stream`, `test_stream`, and `n_classes`.
            model (nn.Module): Backbone model.
            controller (Controller | None): Optional controller.
            seen_classes (set[int]): Mutable reference to the set of class ids seen during training.
            device (torch.device): Evaluation device.
            criterion (nn.Module): Loss criterion.
            num_classes (int): Model output width.
            calibration (CalibrationCollector | None): Optional calibration collector.
            prediction_recorder (PredictionRecorder | None): Optional prediction recorder.
            context (MetricContext): Shared metric context.
            batch_size (int): Evaluation minibatch size.
            num_epochs_per_experience (int): Training epochs per experience used for MLflow steps.
            repair_after_experience (bool): Whether repair fitting happens after each experience.
            include_forward_transfer (bool): Whether to emit forward-transfer metrics.
            backbone_analysis_baseline (Mapping[str, list[float | None]] | None): Optional backbone
                artifact baseline used by repair runs.
            eps (float): Retrieval-correctable fraction epsilon.
            mask_value (float): Logit value written into masked columns.
        """
        self.benchmark = benchmark
        self.model = model
        self.controller = controller
        self.seen_classes = seen_classes
        self.device = torch.device(device)
        self.criterion = criterion
        self.num_classes = int(num_classes)
        self.calibration = calibration
        self.prediction_recorder = prediction_recorder
        self.context = context
        self.batch_size = int(batch_size)
        self.num_epochs_per_experience = int(num_epochs_per_experience)
        self.repair_after_experience = bool(repair_after_experience)
        self.include_forward_transfer = bool(include_forward_transfer)
        self.eps = float(eps)
        self.mask_value = float(mask_value)

        self._forgetting = ForgettingTracker()
        self._forward_transfer = (
            ForwardTransferTracker()
            if self.include_forward_transfer
            else None
        )

        self.acc_exp_base: list[float] = []
        self.artifacts: AnalysisArtifacts | None = None
        self.last_posthoc_scalar_results: dict[str, float] | None = None
        self.last_base_eval_result: EvaluationPassResult | None = None
        self.last_ctrl_eval_result: EvaluationPassResult | None = None
        self.last_posthoc_exp_idx: int | None = None

        self._num_experiences = len(getattr(self.benchmark, 'test_stream'))
        self._backbone_acc_exp_base: list[float] | None = None
        self._backbone_acc_base: list[float] | None = None
        self._backbone_diag_vectors: dict[str, list[float | None]] | None = None

        if backbone_analysis_baseline is not None:
            self._backbone_acc_exp_base = self._coerce_backbone_vector(
                baseline=backbone_analysis_baseline,
                key=ARTIFACT_ACC_EXP_BASE,
                expected_len=self._num_experiences,
            )
            self._backbone_acc_base = self._coerce_backbone_vector(
                baseline=backbone_analysis_baseline,
                key=ARTIFACT_ACC_FINAL_BASE,
                expected_len=self._num_experiences,
            )
            if self._is_repair_controller():
                diag_vectors: dict[str, list[float | None]] = {}
                for diag_key in DIAG_VECTOR_KEYS:
                    diag_vectors[diag_key] = self._coerce_required_nullable_backbone_vector(
                        baseline=backbone_analysis_baseline,
                        key=diag_key,
                        expected_len=self._num_experiences,
                    )
                self._backbone_diag_vectors = diag_vectors

        if self._is_repair_controller():
            if (
                self._backbone_acc_exp_base is None
                or self._backbone_acc_base is None
                or self._backbone_diag_vectors is None
            ):
                raise ValueError(
                    'Repair-controller runs require `backbone_analysis_baseline` '
                    'with baseline accuracy and diagnostic vectors.'
                )

    @staticmethod
    def _coerce_backbone_vector(
        *,
        baseline: Mapping[str, list[float | None]],
        key: str,
        expected_len: int,
    ) -> list[float]:
        """
        Validate and coerce a required numeric backbone vector.

        Args:
            baseline (Mapping[str, list[float | None]]): Baseline payload.
            key (str): Payload key to read.
            expected_len (int): Expected vector length.

        Returns:
            list[float]: Coerced vector.
        """
        values = baseline.get(key)
        if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
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
        baseline: Mapping[str, list[float | None]],
        key: str,
        expected_len: int,
    ) -> list[float | None]:
        """
        Validate and coerce a required nullable backbone vector.

        Args:
            baseline (Mapping[str, list[float | None]]): Baseline payload.
            key (str): Payload key to read.
            expected_len (int): Expected vector length.

        Returns:
            list[float | None]: Coerced vector.
        """
        values = baseline.get(key)
        if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
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

    def run_before_training(self) -> None:
        """
        Bootstrap forward-transfer initial accuracies before any training.
        """
        if self._forward_transfer is None or self._forward_transfer.has_initial:
            return

        eval_tag = 'ctrl' if self.controller is not None else 'base'
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            initial_result = self.eval_pass(
                self.benchmark.test_stream,
                label='initial',
                eval_tag=eval_tag,
                checkpoint_exp_idx=-1,
                controller_enabled=self.controller is not None,
                capture_logits=False,
                capture_backbone_logits=False,
                capture_predictions=False,
                capture_auxiliary_metrics=False,
                log_step=0,
                compute_accuracy=True,
                compute_loss=False,
            )
        finally:
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
        stream_value = self._forward_transfer.bootstrap(per_exp_acc=initial_result.per_exp_acc)
        self._log_canonical_metric(
            key=RUN_EVAL_FORGETTING_STREAM,
            value=0.0,
            step=0,
        )
        self._log_canonical_metric(
            key=RUN_EVAL_TRANSFER_STREAM,
            value=float(stream_value),
            step=0,
        )

    def before_strategy_eval(self, *, strategy: BaseTemplate) -> None:
        """
        Begin calibration collection for retained strategy-managed eval passes.

        Avalanche still runs its own `eval_every` schedule for train-side loss
        histories. Those passes must keep feeding the calibration collector so
        the persisted diagnostic histories stay identical to the legacy output.

        Args:
            strategy (object): Avalanche strategy.
        """
        if self.calibration is None:
            return

        capture_context = getattr(strategy, '_regain_prediction_capture_context', None)
        capture_auxiliary_metrics = True
        checkpoint_exp_idx = None
        if isinstance(capture_context, Mapping):
            capture_auxiliary_metrics = bool(
                capture_context.get('capture_auxiliary_metrics', True)
            )
            checkpoint_exp_idx_raw = capture_context.get('checkpoint_exp_idx')
            checkpoint_exp_idx = (
                int(checkpoint_exp_idx_raw)
                if checkpoint_exp_idx_raw is not None
                else None
            )

        self.calibration.begin_pass(
            eval_tag=str(getattr(strategy, '_regain_eval_tag', '') or ''),
            checkpoint_exp_idx=checkpoint_exp_idx,
            capture_auxiliary_metrics=capture_auxiliary_metrics,
        )

    def before_strategy_eval_exp(self, *, strategy: BaseTemplate) -> None:
        """
        Begin calibration collection for one strategy-managed eval experience.

        Args:
            strategy (object): Avalanche strategy.
        """
        if self.calibration is None:
            return

        experience = getattr(strategy, 'experience', None)
        if experience is None or getattr(experience, 'current_experience', None) is None:
            return

        self.calibration.begin_experience(
            exp_idx=int(experience.current_experience),
            class_ids=_sorted_unique_class_ids_for_experience(experience),
        )

    def observe_strategy_eval_batch(self, *, strategy: BaseTemplate) -> None:
        """
        Observe one strategy-managed eval minibatch for calibration parity.

        Args:
            strategy (object): Avalanche strategy.
        """
        if self.calibration is None:
            return

        logits = getattr(strategy, 'mb_output', None)
        targets = getattr(strategy, 'mb_y', None)
        if not torch.is_tensor(logits) or logits.ndim != 2:
            return
        if not torch.is_tensor(targets):
            return

        targets_vec = targets.reshape(-1)
        if int(targets_vec.shape[0]) != int(logits.shape[0]):
            return

        self.calibration.observe_batch(
            logits=logits,
            targets=targets_vec,
        )

    def after_strategy_eval_exp(self) -> None:
        """
        Finalize one strategy-managed eval experience for calibration parity.
        """
        if self.calibration is None:
            return
        self.calibration.end_experience(log_step=int(self.context.log_step))

    def after_strategy_eval(self) -> None:
        """
        Finalize one strategy-managed eval pass for calibration parity.
        """
        if self.calibration is None:
            return
        self.calibration.end_pass(log_step=int(self.context.log_step))

    def eval_pass(
        self,
        stream: CLStream | CLExperience,
        *,
        label: str,
        eval_tag: str,
        checkpoint_exp_idx: int,
        mask: ClassMask | None = None,
        controller_enabled: bool = True,
        capture_logits: bool = False,
        capture_backbone_logits: bool = False,
        capture_predictions: bool = True,
        capture_auxiliary_metrics: bool = True,
        log_step: int | None = None,
        compute_accuracy: bool = True,
        compute_loss: bool = True,
    ) -> EvaluationPassResult:
        """
        Run one direct evaluation pass over a stream.

        Args:
            stream (object): Single experience or iterable of experiences.
            label (str): Human-readable pass label.
            eval_tag (str): Evaluation tag such as `base`, `ctrl`, or `ref`.
            checkpoint_exp_idx (int): Checkpoint experience index represented by the pass.
            mask (ClassMask | None): Optional seen-class mask.
            controller_enabled (bool): Whether repair correction is enabled.
            capture_logits (bool): Whether to retain post-controller logits in the result.
            capture_backbone_logits (bool): Whether to retain pre-correction logits in the result.
            capture_predictions (bool): Whether to write prediction artifacts.
            capture_auxiliary_metrics (bool): Whether to collect calibration diagnostics.
            log_step (int | None): Optional MLflow step used for calibration logging.
            compute_accuracy (bool): Whether to aggregate accuracy statistics.
            compute_loss (bool): Whether to aggregate loss statistics.

        Returns:
            EvaluationPassResult: Aggregated pass result.
        """
        experiences = _coerce_stream(stream)
        controller_module = self._controller_module()
        previous_context = (
            self.context.phase,
            int(self.context.experience_index),
            int(self.context.epoch_index),
            int(self.context.log_step),
            str(self.context.log_namespace),
            bool(self.context.log_enabled),
        )
        previous_training_mode = bool(self.model.training)

        self.model.eval()
        self.context.set_phase(MetricPhase.EVAL)
        self.context.set_log_namespace(NAMESPACE_EVAL)
        self.context.set_log_step(int(log_step) if log_step is not None else 0)
        self.context.set_log_enabled(log_step is not None)

        if self.calibration is not None:
            self.calibration.begin_pass(
                eval_tag=eval_tag,
                checkpoint_exp_idx=checkpoint_exp_idx,
                capture_auxiliary_metrics=capture_auxiliary_metrics,
            )
        if self.prediction_recorder is not None:
            self.prediction_recorder.begin_pass(
                eval_tag=eval_tag,
                checkpoint_exp_idx=checkpoint_exp_idx,
                capture_predictions=capture_predictions,
            )

        per_exp_acc: dict[int, float] = {}
        per_exp_loss: dict[int, float] = {}
        per_exp_logits: dict[int, np.ndarray] | None = {} if capture_logits else None
        per_exp_backbone_logits: dict[int, np.ndarray] | None = (
            {}
            if capture_backbone_logits
            else None
        )
        per_exp_targets: dict[int, np.ndarray] | None = (
            {}
            if (capture_logits or capture_backbone_logits)
            else None
        )
        per_exp_class_ids: dict[int, list[int]] = {}

        started_at = time.perf_counter()
        try:
            with frozen_model_state(model=self.model, controller_module=controller_module):
                self._controller_on_eval_begin()
                try:
                    with torch.no_grad():
                        for experience in experiences:
                            exp_idx = int(getattr(experience, 'current_experience'))
                            class_ids = _sorted_unique_class_ids_for_experience(experience)
                            per_exp_class_ids[exp_idx] = class_ids
                            self.context.set_experience(exp_idx)
                            self.context.set_epoch(0)
                            self._controller_on_eval_experience_begin()

                            if self.calibration is not None:
                                self.calibration.begin_experience(
                                    exp_idx=exp_idx,
                                    class_ids=class_ids,
                                )
                            if self.prediction_recorder is not None:
                                self.prediction_recorder.begin_experience(
                                    exp_idx=exp_idx,
                                    class_ids=class_ids,
                                )

                            dataset = _resolve_experience_dataset(experience)
                            loader = DataLoader(
                                dataset=dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=0,
                            )

                            total_loss = 0.0
                            total_correct = 0
                            total_examples = 0
                            logits_chunks: list[np.ndarray] = []
                            backbone_logits_chunks: list[np.ndarray] = []
                            targets_chunks: list[np.ndarray] = []

                            for batch in loader:
                                inputs, targets = _extract_batch_tensors(batch)
                                inputs = inputs.to(device=self.device)
                                targets = targets.to(device=self.device)

                                outputs = self.model(inputs)
                                if not torch.is_tensor(outputs):
                                    raise TypeError('Model forward pass must return a tensor of logits.')
                                if mask is not None:
                                    outputs = mask.apply(logits=outputs)

                                backbone_logits: torch.Tensor | None = None
                                if capture_backbone_logits:
                                    backbone_logits = outputs.detach().clone()

                                if controller_enabled and self._is_repair_controller():
                                    assert isinstance(self.controller, RepairController)
                                    outputs = apply_repair_correction(
                                        controller=self.controller,
                                        model=self.model,
                                        inputs=inputs,
                                        backbone_outputs=outputs,
                                        train_seen_classes=self.seen_classes,
                                    )
                                    if not torch.is_tensor(outputs):
                                        raise TypeError(
                                            'Repair controller correction must return a tensor of logits.'
                                        )

                                check_eval_batch(
                                    logits=outputs,
                                    targets=targets,
                                    num_classes=self.num_classes,
                                )

                                batch_targets = targets.reshape(-1).to(device=outputs.device, dtype=torch.long)
                                batch_size = int(batch_targets.shape[0])
                                total_examples += batch_size

                                if compute_loss:
                                    loss_tensor = self.criterion(outputs, targets.reshape(-1).long())
                                    if not torch.is_tensor(loss_tensor):
                                        raise TypeError('Criterion must return a scalar tensor.')
                                    total_loss += float(loss_tensor.item()) * float(batch_size)

                                if compute_accuracy:
                                    predictions = torch.argmax(outputs, dim=1)
                                    batch_correct = int(torch.sum(predictions.eq(batch_targets)).item())
                                    total_correct += batch_correct

                                if self.calibration is not None:
                                    self.calibration.observe_batch(
                                        logits=outputs,
                                        targets=batch_targets,
                                    )
                                if self.prediction_recorder is not None:
                                    self.prediction_recorder.observe_batch(
                                        logits=outputs,
                                        targets=batch_targets,
                                    )

                                if capture_logits:
                                    logits_chunks.append(
                                        outputs.detach().to(device='cpu', dtype=torch.float32).numpy()
                                    )
                                if capture_backbone_logits and backbone_logits is not None:
                                    backbone_logits_chunks.append(
                                        backbone_logits.detach().to(device='cpu', dtype=torch.float32).numpy()
                                    )
                                if per_exp_targets is not None:
                                    targets_chunks.append(
                                        batch_targets.detach().to(device='cpu', dtype=torch.int32).numpy()
                                    )

                            if self.calibration is not None:
                                self.calibration.end_experience(log_step=log_step)
                            if self.prediction_recorder is not None:
                                self.prediction_recorder.end_experience()
                            self._controller_on_eval_experience_end()

                            if total_examples <= 0:
                                raise RuntimeError(
                                    'Evaluation pass produced no samples. '
                                    f'label={label}, exp_idx={exp_idx}'
                                )

                            if compute_accuracy:
                                per_exp_acc[exp_idx] = float(total_correct) / float(total_examples)
                            if compute_loss:
                                per_exp_loss[exp_idx] = float(total_loss) / float(total_examples)
                            if capture_logits and per_exp_logits is not None:
                                per_exp_logits[exp_idx] = np.concatenate(logits_chunks, axis=0)
                            if capture_backbone_logits and per_exp_backbone_logits is not None:
                                per_exp_backbone_logits[exp_idx] = np.concatenate(
                                    backbone_logits_chunks,
                                    axis=0,
                                )
                            if per_exp_targets is not None:
                                per_exp_targets[exp_idx] = np.concatenate(targets_chunks, axis=0)
                finally:
                    self._controller_on_eval_end()
                    if self.calibration is not None:
                        self.calibration.end_pass(log_step=log_step)
                    if self.prediction_recorder is not None:
                        self.prediction_recorder.end_pass()
        finally:
            self.model.train(previous_training_mode)
            (
                prev_phase,
                prev_experience_index,
                prev_epoch_index,
                prev_log_step,
                prev_log_namespace,
                prev_log_enabled,
            ) = previous_context
            self.context.set_phase(prev_phase)
            self.context.set_experience(prev_experience_index)
            self.context.set_epoch(prev_epoch_index)
            self.context.set_log_step(prev_log_step)
            self.context.set_log_namespace(prev_log_namespace)
            self.context.set_log_enabled(prev_log_enabled)

        elapsed_ms = 1000.0 * float(time.perf_counter() - started_at)

        return EvaluationPassResult(
            label=str(label),
            per_exp_acc=per_exp_acc,
            per_exp_loss=per_exp_loss,
            per_exp_logits=per_exp_logits,
            per_exp_backbone_logits=per_exp_backbone_logits,
            per_exp_targets=per_exp_targets,
            per_exp_class_ids=per_exp_class_ids,
            timing_ms=float(elapsed_ms),
        )

    def _log_current_experience_loss(
        self,
        *,
        experience: CLExperience,
        label: str,
        checkpoint_exp_idx: int,
        controller_enabled: bool,
        stage: str,
        step: int,
    ) -> None:
        """
        Evaluate and log one current-experience loss probe under `run.train.loss.*`.

        Args:
            experience (CLExperience): Experience to probe.
            label (str): Human-readable pass label.
            checkpoint_exp_idx (int): Training checkpoint identity.
            controller_enabled (bool): Whether controller correction is enabled.
            stage (str): Loss probe stage, either `train` or `test`.
            step (int): MLflow step.
        """
        if stage not in {'train', 'test'}:
            raise ValueError(f'Unsupported loss probe stage: {stage}')

        exp_idx = int(getattr(experience, 'current_experience'))
        result = self.eval_pass(
            [experience],
            label=label,
            eval_tag='loss',
            checkpoint_exp_idx=checkpoint_exp_idx,
            controller_enabled=controller_enabled,
            capture_logits=False,
            capture_backbone_logits=False,
            capture_predictions=False,
            capture_auxiliary_metrics=False,
            log_step=None,
            compute_accuracy=False,
            compute_loss=True,
        )
        loss_value = self._required_loss(
            per_exp_loss=result.per_exp_loss,
            exp_idx=exp_idx,
        )
        self._log_canonical_metric(
            key=f'{RUN_TRAIN_LOSS}{NS_SEP}{EXPERIENCE_KEY_PREFIX}{exp_idx:03d}{NS_SEP}{stage}',
            value=float(loss_value),
            step=step,
        )

    def run_after_training_exp(
        self,
        *,
        strategy: BaseTemplate,
        seen_classes: Iterable[int],
    ) -> None:
        """
        Run the post-experience evaluation schedule.

        Args:
            strategy (BaseTemplate): Avalanche strategy exposing `experience`.
            seen_classes (Iterable[int]): Class ids observed in training so far.
        """
        experience = getattr(strategy, 'experience')
        if experience is None or getattr(experience, 'current_experience', None) is None:
            raise ValueError('Strategy experience is required to compute analysis metrics.')

        exp_idx = int(experience.current_experience)
        analysis_step = int((exp_idx + 1) * self.num_epochs_per_experience)
        eval_metric_step = self._avalanche_eval_metric_step()
        eval_tag = 'ctrl' if self.controller is not None else 'base'
        loss_probe_controller_enabled = not self._is_repair_controller()

        ckpt_result = self.eval_pass(
            self.benchmark.test_stream,
            label='ckpt',
            eval_tag=eval_tag,
            checkpoint_exp_idx=exp_idx,
            controller_enabled=self.controller is not None,
            capture_logits=True,
            capture_backbone_logits=self._is_repair_controller(),
            capture_predictions=True,
            capture_auxiliary_metrics=True,
            log_step=eval_metric_step,
            compute_accuracy=True,
            compute_loss=False,
        )
        ref_test_accuracy = derive_masked_ref_accuracy(
            ckpt_result,
            exp_idx=exp_idx,
            seen_class_ids=set(int(class_id) for class_id in seen_classes),
            mask_value=self.mask_value,
        )
        if ref_test_accuracy is None:
            raise RuntimeError(
                'Missing derived current-test reference accuracy. '
                f'eval_tag={eval_tag}, checkpoint_exp_idx={exp_idx}'
            )

        self._log_current_experience_loss(
            experience=self.benchmark.train_stream[exp_idx],
            label='train_loss',
            checkpoint_exp_idx=exp_idx,
            controller_enabled=loss_probe_controller_enabled,
            stage='train',
            step=eval_metric_step,
        )
        self._log_current_experience_loss(
            experience=self.benchmark.test_stream[exp_idx],
            label='test_loss',
            checkpoint_exp_idx=exp_idx,
            controller_enabled=loss_probe_controller_enabled,
            stage='test',
            step=eval_metric_step,
        )

        self.acc_exp_base.append(float(ref_test_accuracy))
        self._log_analysis_metric(
            key=RUN_ACC_REF,
            value=float(ref_test_accuracy),
            step=analysis_step,
            experience=exp_idx,
            variant='base',
        )

        forgetting_values = self._forgetting.update(
            trained_exp_idx=exp_idx,
            per_exp_acc=ckpt_result.per_exp_acc,
        )
        for forgotten_exp_idx, forgetting_value in forgetting_values.items():
            self._log_canonical_metric(
                key=f'{RUN_EVAL_FORGETTING}{NS_SEP}{EXPERIENCE_KEY_PREFIX}{int(forgotten_exp_idx):03d}',
                value=float(forgetting_value),
                step=eval_metric_step,
            )
        self._log_canonical_metric(
            key=RUN_EVAL_FORGETTING_STREAM,
            value=self._forgetting.stream_forgetting(values=forgetting_values),
            step=eval_metric_step,
        )

        if self._forward_transfer is not None:
            emitted_transfer, stream_transfer = self._forward_transfer.update(
                trained_exp_idx=exp_idx,
                per_exp_acc=ckpt_result.per_exp_acc,
            )
            for transfer_exp_idx, transfer_value in emitted_transfer.items():
                self._log_canonical_metric(
                    key=f'{RUN_EVAL_TRANSFER}{NS_SEP}{EXPERIENCE_KEY_PREFIX}{int(transfer_exp_idx):03d}',
                    value=float(transfer_value),
                    step=eval_metric_step,
                )
            self._log_canonical_metric(
                key=RUN_EVAL_TRANSFER_STREAM,
                value=float(stream_transfer),
                step=eval_metric_step,
            )

        self.last_posthoc_scalar_results = {
            f'{RUN_ACC_REF}{NS_SEP}{EXPERIENCE_KEY_PREFIX}{exp_idx:03d}{NS_SEP}base': float(ref_test_accuracy)
        }
        if self.controller is not None:
            self.last_base_eval_result = None
            self.last_ctrl_eval_result = ckpt_result
        else:
            self.last_base_eval_result = ckpt_result
            self.last_ctrl_eval_result = None
        self.last_posthoc_exp_idx = exp_idx

    def run_after_training(
        self,
        *,
        strategy: BaseTemplate,
        seen_classes: Iterable[int],
    ) -> None:
        """
        Run the end-of-training evaluation schedule and artifact logging.

        Args:
            strategy (BaseTemplate): Avalanche strategy.
            seen_classes (Iterable[int]): Class ids observed in training so far.
        """
        del seen_classes
        expected = int(self._num_experiences)
        observed = int(len(self.acc_exp_base))
        final_step = int(self._num_experiences * self.num_epochs_per_experience)
        eval_metric_step = self._avalanche_eval_metric_step()
        last_exp_idx = self._num_experiences - 1

        should_run_final = (
            self.last_posthoc_scalar_results is None
            or self.last_posthoc_exp_idx != last_exp_idx
            or (
                self._is_repair_controller()
                and not self.repair_after_experience
            )
        )
        if should_run_final:
            final_ckpt = self.eval_pass(
                self.benchmark.test_stream,
                label='final_ckpt',
                eval_tag='ctrl' if self.controller is not None else 'base',
                checkpoint_exp_idx=last_exp_idx,
                controller_enabled=self.controller is not None,
                capture_logits=False,
                capture_backbone_logits=False,
                capture_predictions=True,
                capture_auxiliary_metrics=True,
                log_step=eval_metric_step,
                compute_accuracy=True,
                compute_loss=False,
            )
            forgetting_values = self._forgetting.update(
                trained_exp_idx=last_exp_idx,
                per_exp_acc=final_ckpt.per_exp_acc,
            )
            for forgotten_exp_idx, forgetting_value in forgetting_values.items():
                self._log_canonical_metric(
                    key=f'{RUN_EVAL_FORGETTING}{NS_SEP}{EXPERIENCE_KEY_PREFIX}{int(forgotten_exp_idx):03d}',
                    value=float(forgetting_value),
                    step=eval_metric_step,
                )
            self._log_canonical_metric(
                key=RUN_EVAL_FORGETTING_STREAM,
                value=self._forgetting.stream_forgetting(values=forgetting_values),
                step=eval_metric_step,
            )
            if self._forward_transfer is not None:
                emitted_transfer, stream_transfer = self._forward_transfer.update(
                    trained_exp_idx=last_exp_idx,
                    per_exp_acc=final_ckpt.per_exp_acc,
                )
                for transfer_exp_idx, transfer_value in emitted_transfer.items():
                    self._log_canonical_metric(
                        key=f'{RUN_EVAL_TRANSFER}{NS_SEP}{EXPERIENCE_KEY_PREFIX}{int(transfer_exp_idx):03d}',
                        value=float(transfer_value),
                        step=eval_metric_step,
                    )
                self._log_canonical_metric(
                    key=RUN_EVAL_TRANSFER_STREAM,
                    value=float(stream_transfer),
                    step=eval_metric_step,
                )
            if self.controller is not None:
                self.last_ctrl_eval_result = final_ckpt
            else:
                self.last_base_eval_result = final_ckpt
            self.last_posthoc_exp_idx = last_exp_idx

        if self.last_posthoc_scalar_results is None:
            raise RuntimeError('Final checkpoint scalar metrics are missing.')

        if observed != expected:
            self.artifacts = {
                COLUMN_STATUS: _STATUS_INCOMPLETE_ACC_EXP_BASE,
                'expected_num_experiences': expected,
                'observed_num_exp_points': observed,
                RUN_EPS: self.eps,
                ARTIFACT_ACC_EXP_BASE: [float(value) for value in self.acc_exp_base],
            }
            self._log_analysis_metric(
                key=RUN_STATUS_INCOMPLETE_ACC_EXP_BASE,
                value=1.0,
                step=final_step,
            )
            if mlflow.active_run() is not None:
                mlflow.log_dict(self.artifacts, MLFLOW_ARTIFACT_ANALYSIS_FILE)
            return

        if self._is_repair_controller():
            final_test_base_result = self.eval_pass(
                self.benchmark.test_stream,
                label='final_test_base',
                eval_tag='base',
                checkpoint_exp_idx=last_exp_idx,
                controller_enabled=False,
                capture_logits=False,
                capture_backbone_logits=False,
                capture_predictions=False,
                capture_auxiliary_metrics=False,
                log_step=None,
                compute_accuracy=True,
                compute_loss=False,
            )
            self.last_base_eval_result = final_test_base_result
            final_test_base = self._ordered_vector(final_test_base_result.per_exp_acc)
        else:
            base_result = self.last_base_eval_result
            if base_result is None and self.controller is not None:
                base_result = self.last_ctrl_eval_result
            if base_result is None:
                base_result = self.eval_pass(
                    self.benchmark.test_stream,
                    label='final_test_base',
                    eval_tag='base',
                    checkpoint_exp_idx=last_exp_idx,
                    controller_enabled=True,
                    capture_logits=False,
                    capture_backbone_logits=False,
                    capture_predictions=False,
                    capture_auxiliary_metrics=False,
                    log_step=None,
                    compute_accuracy=True,
                    compute_loss=False,
                )
                self.last_base_eval_result = base_result
            final_test_base = self._ordered_vector(base_result.per_exp_acc)

        log_ctrl_metrics = self._is_repair_controller()
        if log_ctrl_metrics:
            ctrl_result = self.last_ctrl_eval_result
            if ctrl_result is None:
                ctrl_result = self.eval_pass(
                    self.benchmark.test_stream,
                    label='final_test_ctrl',
                    eval_tag='ctrl',
                    checkpoint_exp_idx=last_exp_idx,
                    controller_enabled=True,
                    capture_logits=False,
                    capture_backbone_logits=False,
                    capture_predictions=False,
                    capture_auxiliary_metrics=False,
                    log_step=None,
                    compute_accuracy=True,
                    compute_loss=False,
                )
                self.last_ctrl_eval_result = ctrl_result
            final_test_ctrl = self._ordered_vector(ctrl_result.per_exp_acc)
            acc_final_ctrl = list(final_test_ctrl)
        else:
            final_test_ctrl = None
            acc_final_ctrl = list(final_test_base)

        diagnostic_vectors = self._diagnostic_vectors_for_artifacts()
        max_ece = self._calibration_max_ece_for_artifacts()
        artifacts = build_analysis_artifacts(
            a_exp_base=self.acc_exp_base,
            a_base=final_test_base,
            a_final_ctrl=acc_final_ctrl,
            eps=self.eps,
            extra_vectors=diagnostic_vectors,
            extra_scalars={RUN_CALIB_MAX_ECE: float(max_ece)},
        )
        self.artifacts = artifacts
        if mlflow.active_run() is not None:
            mlflow.log_dict(self.artifacts, MLFLOW_ARTIFACT_ANALYSIS_FILE)

        self._log_accuracy_vector(
            key=RUN_ACC_FINAL,
            values=final_test_base,
            variant='base',
            step=final_step,
        )
        self._log_analysis_metric(
            key=RUN_ACC_FINAL_AVG_BASE,
            value=self._mean_accuracy(final_test_base),
            step=final_step,
        )

        if log_ctrl_metrics:
            if final_test_ctrl is None:
                raise RuntimeError('Repair-controller final accuracy vectors are missing.')
            self._log_accuracy_vector(
                key=RUN_ACC_FINAL,
                values=final_test_ctrl,
                variant='ctrl',
                step=final_step,
            )
            self._log_analysis_metric(
                key=RUN_ACC_FINAL_AVG_CTRL,
                value=self._mean_accuracy(final_test_ctrl),
                step=final_step,
            )
            rho_values = artifacts.get(ARTIFACT_RHO)
            if isinstance(rho_values, list):
                for exp_idx, value in enumerate(rho_values):
                    if value is None:
                        continue
                    self._log_analysis_metric(
                        key=RUN_RHO,
                        value=float(value),
                        step=final_step,
                        experience=exp_idx,
                    )
            rho_avg = artifacts.get(ARTIFACT_RHO_AVG)
            if rho_avg is not None:
                self._log_analysis_metric(
                    key=RUN_RHO_AVG,
                    value=float(rho_avg),
                    step=final_step,
                )

        final_scalar_results = {
            f'{RUN_ACC_FINAL}{NS_SEP}{EXPERIENCE_KEY_PREFIX}{exp_idx:03d}{NS_SEP}base': float(value)
            for exp_idx, value in enumerate(final_test_base)
        }
        final_scalar_results[RUN_ACC_FINAL_AVG_BASE] = self._mean_accuracy(final_test_base)
        if log_ctrl_metrics and final_test_ctrl is not None:
            for exp_idx, value in enumerate(final_test_ctrl):
                final_scalar_results[
                    f'{RUN_ACC_FINAL}{NS_SEP}{EXPERIENCE_KEY_PREFIX}{exp_idx:03d}{NS_SEP}ctrl'
                ] = float(value)
            final_scalar_results[RUN_ACC_FINAL_AVG_CTRL] = self._mean_accuracy(final_test_ctrl)
        self.last_posthoc_scalar_results = final_scalar_results

        self._log_latency_overhead(step=final_step)

    def _log_latency_overhead(self, *, step: int) -> None:
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
            step (int): MLflow step.
        """
        if mlflow.active_run() is None:
            return

        off_stats = self._measure_latency_stats(controller_on=False)
        if off_stats is None:
            return
        off_ms, off_sps = off_stats
        mlflow.log_metric(key=RUN_LATENCY_MS_PER_SAMPLE_BASE, value=float(off_ms), step=int(step))
        mlflow.log_metric(
            key=RUN_LATENCY_SAMPLES_PER_SEC_BASE,
            value=float(off_sps),
            step=int(step),
        )

        if not self._is_repair_controller():
            return

        on_stats = self._measure_latency_stats(controller_on=True)
        if on_stats is None:
            return
        on_ms, on_sps = on_stats
        mlflow.log_metric(key=RUN_LATENCY_MS_PER_SAMPLE_CTRL, value=float(on_ms), step=int(step))
        mlflow.log_metric(
            key=RUN_LATENCY_SAMPLES_PER_SEC_CTRL,
            value=float(on_sps),
            step=int(step),
        )
        if off_ms > 0.0:
            mlflow.log_metric(
                key=RUN_LATENCY_MS_RATIO,
                value=float(on_ms / off_ms),
                step=int(step),
            )

    def _measure_latency_stats(
        self,
        *,
        controller_on: bool,
        warmup_iters: int = 5,
        timed_iters: int = 20,
    ) -> tuple[float, float] | None:
        """
        Measure latency and throughput on the first test experience.

        Args:
            controller_on (bool): Whether to include controller correction.
            warmup_iters (int): Warmup iterations.
            timed_iters (int): Timed iterations.

        Returns:
            tuple[float, float] | None: `(ms_per_sample, samples_per_sec)` when measurable.
        """
        if len(self.benchmark.test_stream) <= 0:
            return None
        exp0 = self.benchmark.test_stream[0]
        dataset = _resolve_experience_dataset(exp0)
        batch_size = int(self.batch_size)
        if batch_size <= 0:
            return None

        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        if len(loader) == 0:
            return None

        device = module_device(self.model, 'cpu')
        was_training = bool(self.model.training)
        self.model.eval()
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

                    inputs, _ = _extract_batch_tensors(batch)
                    inputs = inputs.to(device=device)

                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    started_at = time.perf_counter()

                    outputs = self.model(inputs)
                    if not torch.is_tensor(outputs):
                        return None
                    if controller_on and self._is_repair_controller():
                        assert isinstance(self.controller, RepairController)
                        outputs = apply_repair_correction(
                            controller=self.controller,
                            model=self.model,
                            inputs=inputs,
                            backbone_outputs=outputs,
                            train_seen_classes=self.seen_classes,
                        )
                    del outputs

                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    if i >= int(warmup_iters):
                        elapsed_seconds += float(time.perf_counter() - started_at)
                        total_samples += int(inputs.shape[0])
        finally:
            self.model.train(was_training)

        if total_samples <= 0 or elapsed_seconds <= 0.0:
            return None
        ms_per_sample = 1000.0 * float(elapsed_seconds) / float(total_samples)
        samples_per_sec = float(total_samples) / float(elapsed_seconds)
        return ms_per_sample, samples_per_sec

    def _ordered_vector(self, per_exp_acc: Mapping[int, float]) -> list[float]:
        """
        Convert a sparse accuracy map into an ordered dense vector.

        Args:
            per_exp_acc (Mapping[int, float]): Accuracy by experience.

        Returns:
            list[float]: Ordered accuracies.
        """
        ordered: list[float] = []
        for exp_idx in range(self._num_experiences):
            if exp_idx not in per_exp_acc:
                raise ValueError(f'Missing Top1 accuracy for experience {exp_idx}.')
            ordered.append(float(per_exp_acc[exp_idx]))
        return ordered

    @staticmethod
    def _required_accuracy(*, per_exp_acc: Mapping[int, float], exp_idx: int) -> float:
        """
        Return one required experience accuracy from a sparse result map.

        Args:
            per_exp_acc (Mapping[int, float]): Accuracy by experience.
            exp_idx (int): Required experience index.

        Returns:
            float: Accuracy for the requested experience.
        """
        exp_idx_int = int(exp_idx)
        if exp_idx_int not in per_exp_acc:
            raise ValueError(f'Missing Top1 accuracy for experience {exp_idx_int}.')
        return float(per_exp_acc[exp_idx_int])

    @staticmethod
    def _required_loss(*, per_exp_loss: Mapping[int, float], exp_idx: int) -> float:
        """
        Return one required experience loss from a sparse result map.

        Args:
            per_exp_loss (Mapping[int, float]): Loss by experience.
            exp_idx (int): Required experience index.

        Returns:
            float: Loss for the requested experience.
        """
        exp_idx_int = int(exp_idx)
        if exp_idx_int not in per_exp_loss:
            raise ValueError(f'Missing loss for experience {exp_idx_int}.')
        return float(per_exp_loss[exp_idx_int])

    def _diagnostic_vectors_for_artifacts(self) -> dict[str, list[float | None]]:
        """
        Resolve diagnostic vectors for `analysis_artifacts.json`.

        Returns:
            dict[str, list[float | None]]: Diagnostic vectors.
        """
        if self._is_repair_controller():
            if self._backbone_diag_vectors is None:
                raise RuntimeError(
                    'Repair-controller runs require backbone diagnostic vectors '
                    'for analysis artifacts.'
                )
            return {
                key: [value for value in vector]
                for key, vector in self._backbone_diag_vectors.items()
            }
        if self.calibration is None:
            raise RuntimeError(
                'CalibrationCollector is required to produce diagnostic vectors.'
            )
        return self.calibration.base_diagnostic_vectors(expected_len=self._num_experiences)

    @staticmethod
    def _max_optional_vector(values: list[float | None] | None) -> float | None:
        """
        Return the maximum finite value from a nullable vector.

        Args:
            values (list[float | None] | None): Optional vector.

        Returns:
            float | None: Maximum finite value or `None`.
        """
        if values is None:
            return None
        finite_values = [
            float(value)
            for value in values
            if value is not None and math.isfinite(float(value))
        ]
        if not finite_values:
            return None
        return float(max(finite_values))

    def _calibration_max_ece_for_artifacts(self) -> float:
        """
        Resolve `run.calibration.max_ece` for the artifact payload.

        Returns:
            float: Max ECE.
        """
        if self._is_repair_controller():
            if self._backbone_diag_vectors is None:
                raise RuntimeError(
                    'Repair-controller runs require backbone calibration vectors '
                    'to compute `calib.max_ece`.'
                )
            max_ece = self._max_optional_vector(self._backbone_diag_vectors.get(RUN_CALIB_ECE))
            if max_ece is None:
                raise RuntimeError(
                    'Repair-controller runs require finite `calib.ece` values '
                    'to compute `calib.max_ece`.'
                )
            return float(max_ece)
        if self.calibration is None:
            raise RuntimeError('CalibrationCollector is required to compute `calib.max_ece`.')
        max_ece = self.calibration.latest_max_ece()
        if max_ece is None:
            raise RuntimeError('Missing `calib.max_ece` from the latest evaluation pass.')
        return float(max_ece)

    def _log_accuracy_vector(
        self,
        *,
        key: str,
        values: list[float],
        variant: str,
        step: int,
    ) -> None:
        """
        Log a per-experience accuracy vector.

        Args:
            key (str): Metric prefix without experience or variant suffix.
            values (list[float]): Accuracy values.
            variant (str): Variant such as `base` or `ctrl`.
            step (int): MLflow step.
        """
        for exp_idx, value in enumerate(values):
            self._log_analysis_metric(
                key=key,
                value=float(value),
                step=step,
                experience=exp_idx,
                variant=variant,
            )

    @staticmethod
    def _mean_accuracy(values: list[float]) -> float:
        """
        Compute the arithmetic mean of an accuracy vector.

        Args:
            values (list[float]): Accuracy values.

        Returns:
            float: Mean accuracy.
        """
        return float(sum(values) / max(1, len(values)))

    @staticmethod
    def _log_analysis_metric(
        *,
        key: str,
        value: float,
        step: int,
        experience: int | None = None,
        variant: str | None = None,
    ) -> None:
        """
        Log one stable MLflow metric.

        Args:
            key (str): Base metric key.
            value (float): Metric value.
            step (int): MLflow step.
            experience (int | None): Optional experience suffix.
            variant (str | None): Optional variant suffix.
        """
        if mlflow.active_run() is None:
            return
        full_key = str(key)
        if experience is not None:
            full_key += f'{NS_SEP}{EXPERIENCE_KEY_PREFIX}{int(experience):03d}'
        if variant is not None:
            full_key += f'{NS_SEP}{str(variant)}'
        mlflow.log_metric(
            key=normalize_metric_name(full_key),
            value=float(value),
            step=int(step),
        )

    @staticmethod
    def _log_canonical_metric(
        *,
        key: str,
        value: float,
        step: int,
    ) -> None:
        """
        Log one already-canonical metric key.

        Args:
            key (str): Metric key.
            value (float): Metric value.
            step (int): MLflow step.
        """
        if mlflow.active_run() is None:
            return
        mlflow.log_metric(
            key=normalize_metric_name(key),
            value=float(value),
            step=int(step),
        )

    def _avalanche_eval_metric_step(self) -> int:
        """
        Mirror Avalanche eval-step selection for retained eval-side histories.

        Analysis metrics keep the explicit experience-based step convention
        `(exp_idx + 1) * num_epochs_per_experience`. Only the retained
        forgetting and transfer histories use this helper.

        Avalanche's `MetricContextPlugin.before_eval()` overwrites the active eval
        step with `context.train_step`. Checkpoint-backed repair runs never
        advance `train_step`, so these histories stay at step `0` even when the
        surrounding analysis metrics use experience-based
        steps.

        Returns:
            int: Eval metric step derived from `context.train_step`.
        """
        return int(self.context.train_step)

    def _is_repair_controller(self) -> bool:
        """
        Check whether the attached controller is a repair controller.

        Returns:
            bool: True for repair controllers.
        """
        return isinstance(self.controller, RepairController)

    def _controller_module(self) -> nn.Module | None:
        """
        Resolve the controller module to protect during evaluation.

        Returns:
            nn.Module | None: Controller module when present.
        """
        if isinstance(self.controller, nn.Module):
            return self.controller
        return None

    def _controller_on_eval_begin(self) -> None:
        """
        Forward the eval-begin lifecycle to the controller when present.
        """
        if self.controller is None:
            return
        self.controller.on_eval_begin()

    def _controller_on_eval_end(self) -> None:
        """
        Forward the eval-end lifecycle to the controller when present.
        """
        if self.controller is None:
            return
        self.controller.on_eval_end()

    def _controller_on_eval_experience_begin(self) -> None:
        """
        Forward the eval-experience-begin lifecycle to the controller.
        """
        if self.controller is None:
            return
        self.controller.on_eval_experience_begin()

    def _controller_on_eval_experience_end(self) -> None:
        """
        Forward the eval-experience-end lifecycle to the controller.
        """
        if self.controller is None:
            return
        self.controller.on_eval_experience_end()
