"""
Tests for registry helpers.
"""

import pytest

from regain.registry import get_lr_scheduler_path
from regain.registry import get_scenario_builder_path
from regain.registry import list_lr_schedulers
from regain.registry import list_scenarios


##########################
# Scenario registry tests #
##########################


class TestScenarioRegistry:
    def test_list_scenarios_contains_supported_names(self) -> None:
        scenarios = list_scenarios()
        assert scenarios == tuple(sorted(scenarios))
        assert 'split_cifar100' in scenarios
        assert 'split_cub200' in scenarios
        assert 'split_imagenet_r' in scenarios
        assert 'split_tiny_imagenet' in scenarios

    def test_get_scenario_builder_path_resolves_split_cub200(self) -> None:
        builder_path = get_scenario_builder_path('split_cub200')
        assert builder_path == 'regain.avalanche_utils.scenarios.SplitCUB200ScenarioBuilder'

    def test_get_scenario_builder_path_resolves_split_imagenet_r(self) -> None:
        builder_path = get_scenario_builder_path('split_imagenet_r')
        assert builder_path == 'regain.avalanche_utils.scenarios.SplitImageNetRScenarioBuilder'

    def test_get_scenario_builder_path_raises_for_unknown_scenario(self) -> None:
        with pytest.raises(ValueError, match='Unsupported scenario'):
            get_scenario_builder_path('not_a_valid_scenario')


class TestLRSchedulerRegistry:
    def test_list_lr_schedulers_contains_supported_names(self) -> None:
        schedulers = list_lr_schedulers()
        assert schedulers == tuple(sorted(schedulers))
        assert 'multi_step' in schedulers
        assert 'warmup_cosine' in schedulers

    def test_get_lr_scheduler_path_resolves_warmup_cosine(self) -> None:
        scheduler_path = get_lr_scheduler_path('warmup_cosine')
        assert scheduler_path == 'regain.experiments.lr_schedulers.WarmupCosineLR'
