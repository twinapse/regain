"""
Tests for the thin posthoc evaluation plugin adapter.
"""

from types import SimpleNamespace

from regain.avalanche_utils.plugins import RegainEvaluationPlugin
from regain.avalanche_utils.plugins import SeenClassesObserver


class _FakeEvaluator:
    def __init__(self) -> None:
        self.artifacts = {'ok': True}
        self.last_posthoc_scalar_results = {'metric': 1.0}
        self.calls: list[tuple[str, object, set[int] | None]] = []

    def run_before_training(self) -> None:
        self.calls.append(('before_training', None, None))

    def before_strategy_eval(self, *, strategy: object) -> None:
        self.calls.append(('before_eval', strategy, None))

    def before_strategy_eval_exp(self, *, strategy: object) -> None:
        self.calls.append(('before_eval_exp', strategy, None))

    def observe_strategy_eval_batch(self, *, strategy: object) -> None:
        self.calls.append(('after_eval_iteration', strategy, None))

    def after_strategy_eval_exp(self) -> None:
        self.calls.append(('after_eval_exp', None, None))

    def after_strategy_eval(self) -> None:
        self.calls.append(('after_eval', None, None))

    def run_after_training_exp(self, *, strategy: object, seen_classes) -> None:
        self.calls.append(('after_training_exp', strategy, set(int(v) for v in seen_classes)))

    def run_after_training(self, *, strategy: object, seen_classes) -> None:
        self.calls.append(('after_training', strategy, set(int(v) for v in seen_classes)))


class TestRegainEvaluationPlugin:
    def test_delegates_to_helper_and_exposes_properties(self) -> None:
        evaluator = _FakeEvaluator()
        observer = SeenClassesObserver()
        observer.seen_classes.update({1, 2})
        plugin = RegainEvaluationPlugin(
            evaluator=evaluator,
            seen_classes_observer=observer,
        )
        strategy = SimpleNamespace(experience=SimpleNamespace(current_experience=0))

        plugin.before_training(strategy=strategy)
        plugin.before_eval(strategy=strategy)
        plugin.before_eval_exp(strategy=strategy)
        plugin.after_eval_iteration(strategy=strategy)
        plugin.after_eval_exp(strategy=strategy)
        plugin.after_eval(strategy=strategy)
        plugin.after_training_exp(strategy=strategy)
        plugin.after_training(strategy=strategy)

        assert plugin.artifacts == {'ok': True}
        assert plugin.last_posthoc_scalar_results == {'metric': 1.0}
        assert evaluator.calls == [
            ('before_training', None, None),
            ('before_eval', strategy, None),
            ('before_eval_exp', strategy, None),
            ('after_eval_iteration', strategy, None),
            ('after_eval_exp', None, None),
            ('after_eval', None, None),
            ('after_training_exp', strategy, {1, 2}),
            ('after_training', strategy, {1, 2}),
        ]
