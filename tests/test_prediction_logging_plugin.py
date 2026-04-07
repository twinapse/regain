"""
Tests for prediction artifact capture during evaluation.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

# Ensure a stable import order for plugin module initialization.
import regain.experiments.orchestrator  # noqa: F401
from regain.analysis.metrics import MetricContext
from regain.avalanche_utils.plugins import PredictionLoggingPlugin
from regain.avalanche_utils.plugins import RegainEvaluationPlugin


class _FakeSeenMaskPlugin:
    def __init__(self, *, mask_enabled: bool) -> None:
        self.mask_enabled = bool(mask_enabled)

    def enable_masking(self) -> None:
        self.mask_enabled = True

    def disable_masking(self) -> None:
        self.mask_enabled = False


class _FakeStrategy:
    def __init__(self) -> None:
        self.captured_context: dict[str, object] | None = None
        self.captured_eval_tag: str | None = None

    def eval(self, stream: list[object]) -> dict[str, int]:
        self.captured_context = dict(self._regain_prediction_capture_context)
        self.captured_eval_tag = str(self._regain_eval_tag)
        return {'stream_len': len(stream)}


class TestPredictionLoggingPlugin:
    def test_writes_npz_per_experience(
        self,
        tmp_path: Path,
    ) -> None:
        plugin = PredictionLoggingPlugin(
            artifact_root=tmp_path / 'predictions',
            num_classes=3,
        )
        strategy = SimpleNamespace(
            _regain_prediction_capture_context={
                'eval_tag': 'base',
                'checkpoint_exp_idx': 4,
                'mask_enabled': False,
            },
            experience=SimpleNamespace(
                current_experience=2,
                classes_in_this_experience=[9, 7],
            ),
        )

        plugin.before_eval(strategy=strategy)
        plugin.before_eval_exp(strategy=strategy)

        strategy.mb_output = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=torch.float32,
        )
        strategy.mb_y = torch.tensor([2, 1], dtype=torch.long)
        plugin.after_eval_iteration(strategy=strategy)

        strategy.mb_output = torch.tensor(
            [[7.0, 8.0, 9.0]],
            dtype=torch.float32,
        )
        strategy.mb_y = torch.tensor([0], dtype=torch.long)
        plugin.after_eval_iteration(strategy=strategy)

        plugin.after_eval_exp(strategy=strategy)
        plugin.after_eval(strategy=strategy)

        output_path = (
            tmp_path
            / 'predictions'
            / 'base'
            / 'test_exp002_after_exp004.npz'
        )

        assert plugin.has_artifacts()
        assert output_path.exists()
        assert not (tmp_path / 'predictions' / 'manifest.json').exists()

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

    def test_ignores_eval_without_capture_context(
        self,
        tmp_path: Path,
    ) -> None:
        plugin = PredictionLoggingPlugin(
            artifact_root=tmp_path / 'predictions',
            num_classes=2,
        )
        strategy = SimpleNamespace(
            experience=SimpleNamespace(
                current_experience=0,
                classes_in_this_experience=[0, 1],
            ),
            mb_output=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            mb_y=torch.tensor([1], dtype=torch.long),
        )

        plugin.before_eval(strategy=strategy)
        plugin.before_eval_exp(strategy=strategy)
        plugin.after_eval_iteration(strategy=strategy)
        plugin.after_eval_exp(strategy=strategy)
        plugin.after_eval(strategy=strategy)

        assert not plugin.has_artifacts()
        assert not (tmp_path / 'predictions' / 'manifest.json').exists()


class TestRegainEvaluationPredictionCaptureContext:
    def test_run_eval_with_logging_sets_and_restores_capture_context(self) -> None:
        plugin = object.__new__(RegainEvaluationPlugin)
        plugin.context = MetricContext()
        plugin.seen_mask_plugin = _FakeSeenMaskPlugin(mask_enabled=True)

        strategy = _FakeStrategy()

        results = plugin._run_eval_with_logging(
            strategy=strategy,
            stream=[object(), object()],
            mask_enabled=False,
            log_namespace='run.final',
            log_step=12,
            eval_tag='ctrl',
            checkpoint_exp_idx=6,
        )

        assert results == {'stream_len': 2}
        assert strategy.captured_eval_tag == 'ctrl'
        assert strategy.captured_context == {
            'eval_tag': 'ctrl',
            'checkpoint_exp_idx': 6,
            'mask_enabled': False,
        }
        assert not hasattr(strategy, '_regain_eval_tag')
        assert not hasattr(strategy, '_regain_prediction_capture_context')
        assert plugin.context.log_namespace == 'run.train'
        assert plugin.context.log_step == 0
        assert plugin.context.log_enabled is False
        assert plugin.context.phase.value == 'run.train'
        assert plugin.seen_mask_plugin.mask_enabled is True

    def test_run_eval_with_state_sets_and_restores_capture_context(self) -> None:
        plugin = object.__new__(RegainEvaluationPlugin)
        plugin.context = MetricContext()
        plugin.seen_mask_plugin = _FakeSeenMaskPlugin(mask_enabled=False)

        strategy = _FakeStrategy()

        results = plugin._run_eval_with_state(
            strategy=strategy,
            stream=[object()],
            mask_enabled=True,
            eval_tag='reference',
            checkpoint_exp_idx=3,
        )

        assert results == {'stream_len': 1}
        assert strategy.captured_eval_tag == 'reference'
        assert strategy.captured_context == {
            'eval_tag': 'reference',
            'checkpoint_exp_idx': 3,
            'mask_enabled': True,
        }
        assert not hasattr(strategy, '_regain_eval_tag')
        assert not hasattr(strategy, '_regain_prediction_capture_context')
        assert plugin.context.log_namespace == 'run.train'
        assert plugin.context.log_step == 0
        assert plugin.context.log_enabled is False
        assert plugin.context.phase.value == 'run.train'
        assert plugin.seen_mask_plugin.mask_enabled is False
