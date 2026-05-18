"""
Tests for the standalone forgetting tracker.
"""

import pytest

from regain.evaluation import ForgettingTracker


class TestForgettingTracker:
    """
    Tests for ForgettingTracker.
    """

    def test_only_emits_for_past_experiences(self) -> None:
        tracker = ForgettingTracker()

        first = tracker.update(
            trained_exp_idx=0,
            per_exp_acc={
                0: 0.9,
                1: 0.8
            },
        )
        second = tracker.update(
            trained_exp_idx=1,
            per_exp_acc={
                0: 0.7,
                1: 0.8,
                2: 0.6
            },
        )
        third = tracker.update(
            trained_exp_idx=2,
            per_exp_acc={
                0: 0.6,
                1: 0.7,
                2: 0.6
            },
        )

        assert not first
        assert second == {0: pytest.approx(0.2)}
        assert third == {0: pytest.approx(0.3), 1: pytest.approx(0.1)}
        assert tracker.stream_forgetting(values=third) == pytest.approx(0.2)
