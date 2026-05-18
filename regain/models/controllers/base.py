"""
Base classes and interfaces for continual learning controllers.

Controllers are lightweight modules that can either:
  - prevent forgetting (by modifying the model or the training dynamics) or
  - perform post-hoc repair (without modifying the model or affecting the training dynamics).
"""

from abc import ABC
from abc import abstractmethod
from typing import Any

import torch
from torch import nn
from torch.utils.data import Dataset

__all__ = [
    'Controller',
    'PreventionController',
    'RepairController',
    'BackboneControllerInterface',
    'TrainingObjectiveControllerInterface',
]


class PreventionController(nn.Module, ABC):
    """
    Base class for training-time prevention controllers.

    A prevention controller reduces forgetting by intervening during training. It may observe experience/epoch
    boundaries, adjust model state, and optionally modify the training objective. Unlike repair controllers, prevention
    controllers may participate in the full training loop.

    Supported lifecycle hooks:
      - Training boundaries:
        - `on_train_begin(model)`: Called once before training starts.
        - `on_train_end(model)`: Called once after all training experiences are complete.
        - `on_train_experience_begin(dataset)`: Called before training on one experience.
        - `on_train_experience_end(model)`: Called after finishing training on one experience.
        - `on_train_epoch_begin(model)`: Called at the start of a training epoch.
        - `on_train_epoch_end(model)`: Called at the end of a training epoch.
      - Evaluation boundaries:
        - `on_eval_begin(...)`: Called at the start of an evaluation phase.
        - `on_eval_end(...)`: Called at the end of an evaluation phase.
        - `on_eval_experience_begin(...)`: Called before evaluating a single experience.
        - `on_eval_experience_end(...)`: Called after evaluating a single experience.

    Optional capabilities:
      - Backbone correction: Controllers that modify model parameters or structure should implement a
        backbone-correction interface (e.g., `correct_backbone(model)`), invoked at safe boundaries.
      - Training objective correction: Controllers that alter the optimization objective should implement a
        training-objective interface (e.g., `correct_training_objective(...)`), invoked before backpropagation.

    Default hook implementations are no-ops; subclasses should override only the methods they need.
    """

    def __init__(
            self,
            *,
            train_batch_size: int | None = None,  # pylint: disable=unused-argument
            replay_batch_size: int | None = None,  # pylint: disable=unused-argument
            replay_memory_size: int | None = None,  # pylint: disable=unused-argument
            **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        """
        Initialize the prevention controller.

        Note: `train_batch_size`, `replay_batch_size`, and `replay_memory_size` are provided for convenience,
              but not all controllers will need them.

        Args:
            train_batch_size (int): Batch size for current experience training data.
            replay_batch_size (int): Batch size for previous experiences replay data.
            replay_memory_size (int): Total size of the replay memory.
        """
        super().__init__()

    def on_train_begin(
            self,
            model: nn.Module,  # pylint: disable=unused-argument
    ) -> None:
        """
        Hook triggered before training begins (once).

        Args:
            model: Model that will be trained.

        Returns:
            None.
        """
        return

    def on_train_end(
            self,
            model: nn.Module,  # pylint: disable=unused-argument
    ) -> None:
        """
        Hook triggered after training completes (once).

        Args:
            model: Trained model.

        Returns:
            None.
        """
        return

    def on_train_experience_begin(
            self,
            dataset: Dataset | None,  # pylint: disable=unused-argument
    ) -> None:
        """
        Hook triggered before a training experience begins.

        Args:
            dataset: Dataset for the current experience, if available.

        Returns:
            None.
        """
        return

    def on_train_experience_end(
            self,
            model: nn.Module,  # pylint: disable=unused-argument
    ) -> None:
        """
        Hook triggered after a training experience finishes.

        Args:
            model: Model trained on the experience.

        Returns:
            None.
        """
        return

    def on_train_epoch_begin(
            self,
            model: nn.Module,  # pylint: disable=unused-argument
    ) -> None:
        """
        Hook triggered before each training epoch.

        Args:
            model: The model being trained.

        Returns:
            None.
        """
        return

    def on_train_epoch_end(
            self,
            model: nn.Module,  # pylint: disable=unused-argument
    ) -> None:
        """
        Hook triggered after each training epoch.

        Args:
            model: The model being trained.

        Returns:
            None.
        """
        return

    def on_eval_begin(
            self,
            *args,  # pylint: disable=unused-argument
            **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        """
        Hook triggered before evaluation begins.

        Returns:
            None
        """
        return

    def on_eval_end(
            self,
            *args,  # pylint: disable=unused-argument
            **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        """
        Hook triggered after evaluation ends.

        Returns:
            None
        """
        return

    def on_eval_experience_begin(
            self,
            *args,  # pylint: disable=unused-argument
            **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        """
        Hook triggered before an experience evaluation begins.

        Returns:
            None
        """
        return

    def on_eval_experience_end(
            self,
            *args,  # pylint: disable=unused-argument
            **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        """
        Hook triggered after an experience evaluation ends.

        Returns:
            None
        """
        return

    @classmethod
    def requires_replay(cls) -> bool:
        """
        Indicates whether the controller requires experience replay during training.

        Returns:
            bool: True if replay is required, False otherwise.
        """
        return False


class RepairController(nn.Module, ABC):
    """
    Base class for post-hoc (post-training) repair controllers.

    A repair controller mitigates forgetting after the base model has been trained. It does not participate in
    training-time optimization (no train-begin hooks, no epoch hooks, no loss/backward modification). It may update its
    own state at experience boundaries and/or at the end of training, and it may apply output-time corrections during
    evaluation.

    Supported lifecycle hooks:
      - Training boundaries:
        - `on_train_end(model)`: Called after all training experiences are complete.
        - `on_train_experience_end(model)`: Called after finishing training on one experience.
      - Evaluation boundaries:
        - `on_eval_begin(...)`: Called at the start of an evaluation phase.
        - `on_eval_end(...)`: Called at the end of an evaluation phase.
        - `on_eval_experience_begin(...)`: Called before evaluating a single experience.
        - `on_eval_experience_end(...)`: Called after evaluating a single experience.

    Output correction: `correct_outputs(...)` is called to adjust model predictions during evaluation.

    Repair data (optional): Controllers that require dedicated repair data should implement a repair-data interface
    (e.g., `fit_on_repair_data(...)`).

    Default hook implementations are no-ops; subclasses should override only the methods they need.
    """

    def initialize_parameters(self, *, model: nn.Module, sample_inputs: Any | None = None) -> None:
        """
        Initialize controller parameters based on the model and sample inputs.

        Args:
            model (nn.Module): Model used to determine lazy parameter shapes.
            sample_inputs (Any | None): Representative inputs for probing, if needed.

        Returns:
            None.
        """
        del model, sample_inputs
        return

    @classmethod
    def requires_per_experience_fitting(cls) -> bool:
        """
        Whether this controller must be fitted after each experience (streaming),
        and cannot be correctly fitted just once at the end of training.

        Returns:
            bool: True if per-experience fitting is required, False otherwise.
        """
        return False

    def on_train_end(
            self,
            model: nn.Module,  # pylint: disable=unused-argument
    ) -> None:
        """
        Hook triggered after training completes (once).

        Args:
            model: Trained model.

        Returns:
            None.
        """
        return

    def on_train_experience_end(
            self,
            model: nn.Module,  # pylint: disable=unused-argument
    ) -> None:
        """
        Hook triggered after a training experience finishes.

        Args:
            model: Model trained on the experience.

        Returns:
            None.
        """
        return

    def on_eval_begin(
            self,
            *args,  # pylint: disable=unused-argument
            **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        """
        Hook triggered before evaluation begins.

        Returns:
            None
        """
        return

    def on_eval_end(
            self,
            *args,  # pylint: disable=unused-argument
            **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        """
        Hook triggered after evaluation ends.

        Returns:
            None
        """
        return

    def on_eval_experience_begin(
            self,
            *args,  # pylint: disable=unused-argument
            **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        """
        Hook triggered before an experience evaluation begins.

        Returns:
            None
        """
        return

    def on_eval_experience_end(
            self,
            *args,  # pylint: disable=unused-argument
            **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        """
        Hook triggered after an experience evaluation ends.

        Returns:
            None
        """
        return

    @abstractmethod
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
        Fit the controller using repair data.

        Args:
            model (nn.Module): Model used to compute features/logits for fitting.
            repair_dataset (Dataset | None): Repair dataset for fitting.
            new_classes (list[int]): List of newly introduced classes in the current experience.
            num_epochs (int): Number of epochs used for fitting.
            batch_size (int): Batch size used for fitting.

        Returns:
            None.
        """

    @abstractmethod
    def correct_outputs(
        self,
        *,
        outputs: Any,
        model: nn.Module | None = None,
        inputs: Any | None = None,
    ) -> Any:
        """
        Correct model outputs.

        Args:
            outputs: Raw model outputs (e.g., logits).
            model: Model that produced the outputs (optional).
            inputs: Inputs corresponding to `outputs` (optional).

        Returns:
            Corrected outputs.
        """


Controller = PreventionController | RepairController


class BackboneControllerInterface(ABC):
    """
    Interface for controllers that modify the model internals (e.g., backbone, normalization).
    """

    @abstractmethod
    def correct_backbone(self, model: nn.Module) -> None:
        """
        Apply in-place modifications to a model.

        Args:
            model: Model to modify.

        Returns:
            None.
        """


class TrainingObjectiveControllerInterface(ABC):
    """
    Interface for controllers that modify the training objective (loss).
    """

    @abstractmethod
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
        Compute a (possibly) modified loss to use for backpropagation.

        Args:
            loss: Base loss produced by the strategy (may be ignored by the controller).
            outputs: Model outputs for the current minibatch.
            model: The model being trained.
            inputs: Inputs for the current minibatch.
            targets: Targets/labels for the current minibatch.

        Returns:
            A scalar loss tensor.
        """
