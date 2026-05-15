"""
Tests for prediction artifact recording.
"""

from pathlib import Path

import numpy as np
import torch

from regain.evaluation import PredictionRecorder


class TestPredictionRecorder:
    """
    Tests for PredictionRecorder.
    """

    def test_writes_npz_per_experience(self, tmp_path: Path) -> None:
        recorder = PredictionRecorder(
            artifact_root=tmp_path / 'predictions',
            num_classes=3,
        )

        recorder.begin_pass(
            eval_tag='base',
            checkpoint_exp_idx=4,
            capture_predictions=True,
        )
        recorder.begin_experience(exp_idx=2, class_ids=[7, 9])
        recorder.observe_batch(
            logits=torch.tensor(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                ],
                dtype=torch.float32,
            ),
            targets=torch.tensor([2, 1], dtype=torch.long),
        )
        recorder.observe_batch(
            logits=torch.tensor([[7.0, 8.0, 9.0]], dtype=torch.float32),
            targets=torch.tensor([0], dtype=torch.long),
        )
        recorder.end_experience()
        recorder.end_pass()

        output_path = tmp_path / 'predictions' / 'base' / 'test_exp002_after_exp004.npz'
        assert recorder.has_artifacts()
        assert output_path.exists()

        with np.load(output_path) as payload:
            np.testing.assert_array_equal(
                payload['targets'],
                np.asarray([2, 1, 0], dtype=np.int32),
            )
            np.testing.assert_array_equal(
                payload['logits'],
                np.asarray(
                    [
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0],
                    ],
                    dtype=np.float32,
                ),
            )
            np.testing.assert_array_equal(
                payload['class_ids'],
                np.asarray([7, 9], dtype=np.int32),
            )

    def test_skips_writes_when_prediction_capture_is_disabled(self, tmp_path: Path) -> None:
        recorder = PredictionRecorder(
            artifact_root=tmp_path / 'predictions',
            num_classes=2,
        )

        recorder.begin_pass(
            eval_tag='base',
            checkpoint_exp_idx=0,
            capture_predictions=False,
        )
        recorder.begin_experience(exp_idx=0, class_ids=[0, 1])
        recorder.observe_batch(
            logits=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            targets=torch.tensor([1], dtype=torch.long),
        )
        recorder.end_experience()
        recorder.end_pass()

        assert not recorder.has_artifacts()
        assert not (tmp_path / 'predictions').exists()
