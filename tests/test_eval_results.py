"""
Tests for pure evaluation-result helpers.
"""

import numpy as np
import pytest

from regain.evaluation import derive_masked_ref_accuracy
from regain.evaluation import EvaluationPassResult


class TestEvalResults:
    """
    Tests for EvalResults.
    """

    def test_derives_masked_ref_accuracy_from_backbone_logits(self) -> None:
        result = EvaluationPassResult(
            label='ckpt',
            per_exp_acc={0: 0.0},
            per_exp_loss={0: 0.0},
            per_exp_logits={0: np.asarray([[0.0, 5.0, 0.0]], dtype=np.float32)},
            per_exp_backbone_logits={0: np.asarray([[5.0, 0.0, 0.0]], dtype=np.float32)},
            per_exp_targets={0: np.asarray([0], dtype=np.int32)},
            per_exp_class_ids={0: [0, 1]},
            timing_ms=1.0,
        )

        accuracy = derive_masked_ref_accuracy(
            result,
            exp_idx=0,
            seen_class_ids=[0, 1],
        )

        assert accuracy == pytest.approx(1.0)

    def test_returns_none_when_logits_are_missing(self) -> None:
        result = EvaluationPassResult(
            label='ckpt',
            per_exp_acc={},
            per_exp_loss={},
            per_exp_logits=None,
            per_exp_backbone_logits=None,
            per_exp_targets=None,
            per_exp_class_ids={},
            timing_ms=1.0,
        )

        assert derive_masked_ref_accuracy(
            result,
            exp_idx=0,
            seen_class_ids=[0],
        ) is None
