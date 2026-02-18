"""
Tests for experiment utilities.
"""

import pytest

from regain.experiments.utils import resolve_avalanche_eval_every


#########################
# Schedule value mapping #
#########################


class TestResolveAvalancheEvalEvery:
    def test_returns_zero_for_per_experience_schedule(self) -> None:
        assert resolve_avalanche_eval_every(avalanche_schedule='per_experience') == 0

    def test_returns_negative_one_for_final_only_schedule(self) -> None:
        assert resolve_avalanche_eval_every(avalanche_schedule='final_only') == -1

    def test_raises_on_unsupported_schedule(self) -> None:
        with pytest.raises(ValueError, match='Unsupported eval schedule'):
            resolve_avalanche_eval_every(avalanche_schedule='invalid')
