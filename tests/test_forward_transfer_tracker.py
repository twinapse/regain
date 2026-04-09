"""
Tests for the standalone forward-transfer tracker.
"""

import pytest

from regain.evaluation import ForwardTransferTracker


class TestForwardTransferTracker:
    def test_bootstrap_and_incremental_updates(self) -> None:
        tracker = ForwardTransferTracker()

        bootstrap_stream = tracker.bootstrap(per_exp_acc={0: 0.2, 1: 0.3, 2: 0.4})
        first_emitted, first_stream = tracker.update(
            trained_exp_idx=0,
            per_exp_acc={0: 0.9, 1: 0.5, 2: 0.4},
        )
        second_emitted, second_stream = tracker.update(
            trained_exp_idx=1,
            per_exp_acc={0: 0.8, 1: 0.7, 2: 0.9},
        )

        assert bootstrap_stream == pytest.approx(0.0)
        assert first_emitted == {1: pytest.approx(0.2)}
        assert first_stream == pytest.approx(0.2)
        assert second_emitted == {2: pytest.approx(0.5)}
        assert second_stream == pytest.approx(0.35)
