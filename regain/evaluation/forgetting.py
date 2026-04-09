"""
Standalone forgetting and forward-transfer trackers.
"""

__all__ = ['ForgettingTracker', 'ForwardTransferTracker']


class ForgettingTracker:
    """
    Track per-experience forgetting without Avalanche plugin hooks.
    """

    def __init__(self) -> None:
        """
        Initialize empty forgetting state.
        """
        self._first_seen: dict[int, float] = {}

    def update(
        self,
        *,
        trained_exp_idx: int,
        per_exp_acc: dict[int, float],
    ) -> dict[int, float]:
        """
        Update forgetting from one checkpoint accuracy vector.

        Args:
            trained_exp_idx (int): Experience index that has just finished
                training for this checkpoint.
            per_exp_acc (dict[int, float]): Accuracy by experience.

        Returns:
            dict[int, float]: Forgetting values for experiences with both an
                initial and a later observation.
        """
        trained_exp_idx_int = int(trained_exp_idx)
        if trained_exp_idx_int < 0:
            return {}

        forgetting: dict[int, float] = {}
        for exp_idx in sorted(per_exp_acc):
            exp_idx_int = int(exp_idx)
            if exp_idx_int > trained_exp_idx_int:
                continue

            accuracy = per_exp_acc[exp_idx]
            accuracy_float = float(accuracy)
            if exp_idx_int not in self._first_seen:
                self._first_seen[exp_idx_int] = accuracy_float
                continue
            if exp_idx_int >= trained_exp_idx_int:
                continue
            forgetting[exp_idx_int] = self._first_seen[exp_idx_int] - accuracy_float
        return forgetting

    @staticmethod
    def stream_forgetting(*, values: dict[int, float]) -> float:
        """
        Compute stream forgetting for one evaluation pass.

        Args:
            values (dict[int, float]): Forgetting values returned by `update`.

        Returns:
            float: Mean forgetting across available experiences, or `0.0`
                when no forgetting value is available yet.
        """
        if not values:
            return 0.0
        return float(sum(float(value) for value in values.values()) / float(len(values)))


class ForwardTransferTracker:
    """
    Track forward transfer relative to initial random-initialization accuracy.
    """

    def __init__(self) -> None:
        """
        Initialize empty forward-transfer state.
        """
        self._initial: dict[int, float] = {}
        self._previous: dict[int, float] = {}

    @property
    def has_initial(self) -> bool:
        """
        Check whether initial accuracies have been bootstrapped.

        Returns:
            bool: True when initial values are available.
        """
        return bool(self._initial)

    def bootstrap(self, *, per_exp_acc: dict[int, float]) -> float:
        """
        Record random-initialization accuracies.

        Args:
            per_exp_acc (dict[int, float]): Initial accuracy by experience.

        Returns:
            float: Stream forward transfer value for the bootstrap pass.
        """
        self._initial = {
            int(exp_idx): float(accuracy)
            for exp_idx, accuracy in per_exp_acc.items()
        }
        self._previous = {}
        return 0.0

    def update(
        self,
        *,
        trained_exp_idx: int,
        per_exp_acc: dict[int, float],
    ) -> tuple[dict[int, float], float]:
        """
        Update forward transfer after training one experience.

        Args:
            trained_exp_idx (int): Experience index that just finished training.
            per_exp_acc (dict[int, float]): Accuracy by experience from the
                checkpoint pass after that training experience.

        Returns:
            tuple[dict[int, float], float]: Newly emitted per-experience
                transfer values for this training step and the current stream
                transfer mean.
        """
        next_exp_idx = int(trained_exp_idx) + 1
        emitted: dict[int, float] = {}
        if next_exp_idx in per_exp_acc and next_exp_idx in self._initial:
            next_accuracy = float(per_exp_acc[next_exp_idx])
            self._previous[next_exp_idx] = next_accuracy
            emitted[next_exp_idx] = next_accuracy - self._initial[next_exp_idx]

        if not self._previous:
            return emitted, 0.0

        stream_value = float(
            sum(
                float(previous) - float(self._initial[exp_idx])
                for exp_idx, previous in self._previous.items()
            )
            / float(len(self._previous))
        )
        return emitted, stream_value
