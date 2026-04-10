"""
Tests for calibration collection.
"""

import mlflow
import pytest
import torch

from regain.evaluation import CalibrationCollector


class TestCalibrationCollector:
    def test_latest_max_ece_updates_from_completed_pass(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        collector = CalibrationCollector(num_bins=2)
        monkeypatch.setattr(mlflow, 'active_run', lambda: None)

        collector.begin_pass(
            eval_tag='base',
            checkpoint_exp_idx=0,
            capture_auxiliary_metrics=True,
        )
        collector.begin_experience(exp_idx=0, class_ids=[0, 1])
        collector.observe_batch(
            logits=torch.tensor([[3.0, 0.0], [0.0, 3.0]], dtype=torch.float32),
            targets=torch.tensor([0, 1], dtype=torch.long),
        )
        collector.end_experience(log_step=10)
        collector.end_pass(log_step=10)

        assert collector.latest_max_ece() == pytest.approx(0.047426, rel=1e-3)

    def test_capture_disabled_does_not_override_latest_metrics(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        collector = CalibrationCollector(num_bins=2)
        monkeypatch.setattr(mlflow, 'active_run', lambda: None)

        collector.begin_pass(
            eval_tag='base',
            checkpoint_exp_idx=0,
            capture_auxiliary_metrics=True,
        )
        collector.begin_experience(exp_idx=0, class_ids=[0, 1])
        collector.observe_batch(
            logits=torch.tensor([[3.0, 0.0], [0.0, 3.0]], dtype=torch.float32),
            targets=torch.tensor([0, 1], dtype=torch.long),
        )
        collector.end_experience(log_step=10)
        collector.end_pass(log_step=10)
        latest_max_ece = collector.latest_max_ece()
        assert latest_max_ece is not None

        collector.begin_pass(
            eval_tag='base',
            checkpoint_exp_idx=0,
            capture_auxiliary_metrics=False,
        )
        collector.end_pass(log_step=10)

        assert collector.latest_max_ece() == pytest.approx(latest_max_ece)
