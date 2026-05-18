"""
Prediction artifact recording for custom evaluation passes.
"""

from pathlib import Path

import numpy as np
import torch

from regain.constants import EXPERIENCE_KEY_PREFIX

__all__ = ['PredictionRecorder']


class PredictionRecorder:
    """
    Persist per-experience logits and targets as compressed `.npz` files.
    """

    def __init__(
        self,
        *,
        artifact_root: Path,
        num_classes: int,
    ) -> None:
        """
        Initialize the prediction recorder.

        Args:
            artifact_root (Path): Root directory for staged prediction artifacts.
            num_classes (int): Expected logits width.
        """
        self.artifact_root = Path(artifact_root)
        self.num_classes = int(num_classes)
        if self.num_classes <= 0:
            raise ValueError('`num_classes` must be positive.')

        self._written_files: set[str] = set()
        self._capture_predictions: bool = True
        self._eval_tag: str = ''
        self._checkpoint_exp_idx: int = -1
        self._current_exp_idx: int | None = None
        self._current_class_ids: list[int] = []
        self._current_logits_chunks: list[np.ndarray] = []
        self._current_targets_chunks: list[np.ndarray] = []

    def has_artifacts(self) -> bool:
        """
        Check whether any prediction artifacts were written.

        Returns:
            bool: True when at least one prediction file exists.
        """
        return bool(self._written_files)

    def begin_pass(
        self,
        *,
        eval_tag: str,
        checkpoint_exp_idx: int,
        capture_predictions: bool,
    ) -> None:
        """
        Start one evaluation pass.

        Args:
            eval_tag (str): Evaluation tag such as `base` or `ctrl`.
            checkpoint_exp_idx (int): Checkpoint experience index represented by the pass.
            capture_predictions (bool): Whether to record `.npz` artifacts for the pass.
        """
        self._eval_tag = str(eval_tag)
        self._checkpoint_exp_idx = int(checkpoint_exp_idx)
        self._capture_predictions = bool(capture_predictions)
        self._reset_current_experience()

    def end_pass(self) -> None:
        """
        Clear pass-scoped capture state.

        Returns:
            None.
        """
        self._capture_predictions = True
        self._eval_tag = ''
        self._checkpoint_exp_idx = -1
        self._reset_current_experience()

    def begin_experience(
        self,
        *,
        exp_idx: int,
        class_ids: list[int],
    ) -> None:
        """
        Start recording one evaluated experience.

        Args:
            exp_idx (int): Experience index being evaluated.
            class_ids (list[int]): Sorted class ids in the experience.
        """
        self._reset_current_experience()
        if not self._capture_predictions:
            return

        self._current_exp_idx = int(exp_idx)
        self._current_class_ids = [int(class_id) for class_id in class_ids]

    def observe_batch(
        self,
        *,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """
        Record one minibatch of logits and targets.

        Args:
            logits (torch.Tensor): Batch logits.
            targets (torch.Tensor): Integer class targets aligned to `logits`.
        """
        if not self._capture_predictions or self._current_exp_idx is None:
            return
        if int(logits.shape[1]) != self.num_classes:
            raise ValueError('Prediction artifact width mismatch. '
                             f'expected={self.num_classes}, observed={int(logits.shape[1])}')

        targets_vec = targets.reshape(-1).to(device=logits.device, dtype=torch.long)
        if int(targets_vec.shape[0]) != int(logits.shape[0]):
            raise ValueError('Prediction artifact batch mismatch. '
                             f'logits_batch={int(logits.shape[0])}, target_batch={int(targets_vec.shape[0])}')
        if targets_vec.numel() <= 0:
            return

        self._current_logits_chunks.append(logits.detach().to(device='cpu', dtype=torch.float32).numpy())
        self._current_targets_chunks.append(targets_vec.detach().to(device='cpu', dtype=torch.int32).numpy())

    def end_experience(self) -> None:
        """
        Flush the current experience to a compressed `.npz` artifact.

        Returns:
            None.
        """
        if not self._capture_predictions or self._current_exp_idx is None:
            self._reset_current_experience()
            return
        if not self._current_logits_chunks or not self._current_targets_chunks:
            self._reset_current_experience()
            return

        logits = np.concatenate(self._current_logits_chunks, axis=0).astype(np.float32, copy=False)
        targets = np.concatenate(self._current_targets_chunks, axis=0).astype(np.int32, copy=False)
        relative_path = self._artifact_relative_path(
            eval_tag=self._eval_tag,
            checkpoint_exp_idx=self._checkpoint_exp_idx,
            test_exp_idx=self._current_exp_idx,
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

    @staticmethod
    def _artifact_relative_path(
        *,
        eval_tag: str,
        checkpoint_exp_idx: int,
        test_exp_idx: int,
    ) -> Path:
        """
        Build the relative artifact path for one recorded experience.

        Args:
            eval_tag (str): Evaluation tag such as `base`.
            checkpoint_exp_idx (int): Checkpoint experience index.
            test_exp_idx (int): Evaluated test experience index.

        Returns:
            Path: Relative artifact path.
        """
        return Path(str(eval_tag)) / (f'test_{EXPERIENCE_KEY_PREFIX}{int(test_exp_idx):03d}'
                                      f'_after_{EXPERIENCE_KEY_PREFIX}{int(checkpoint_exp_idx):03d}.npz')

    def _reset_current_experience(self) -> None:
        """
        Clear current-experience buffers.

        Returns:
            None.
        """
        self._current_exp_idx = None
        self._current_class_ids = []
        self._current_logits_chunks = []
        self._current_targets_chunks = []
