"""
Tests for experiment builders.
"""

import inspect
from typing import Any

import pytest
import torch

from regain.avalanche_utils.scenarios import CIFAR100ScenarioBuilder
from regain.avalanche_utils.scenarios import CUB200ScenarioBuilder
from regain.avalanche_utils.scenarios import ImageNetRScenarioBuilder
from regain.avalanche_utils.scenarios import ScenarioBuilder
from regain.avalanche_utils.scenarios import TinyImageNetScenarioBuilder
from regain.debug.avalanche_utils import DebugRepairControllerPlugin
from regain.experiments.builders import build_backbone
from regain.experiments.builders import build_benchmark
from regain.experiments.builders import build_controller
from regain.experiments.builders import build_controller_plugin
from regain.experiments.builders import build_lr_scheduler_plugin
from regain.experiments.builders import build_optimizer
import regain.experiments.builders as builders_module
from regain.experiments.config import BackboneConfig
from regain.experiments.config import ControllerConfig
from regain.experiments.config import EvaluationConfig
from regain.experiments.config import ExperimentConfig
from regain.experiments.config import OptimizerConfig
from regain.experiments.config import RepairConfig
from regain.experiments.config import TransformsConfig
from regain.models.controllers import PreventionController
from regain.models.controllers import RepairController


class _NoOpRepairController(RepairController):
    """Repair controller test double for builder-plugin coverage."""

    def fit_on_repair_data(
        self,
        *,
        model: torch.nn.Module,
        repair_dataset: object | None,
        new_classes: list[int],
        num_epochs: int,
        batch_size: int,
    ) -> None:
        del model, repair_dataset, new_classes, num_epochs, batch_size

    def correct_outputs(
        self,
        *,
        outputs: Any,
        model: torch.nn.Module | None = None,
        inputs: Any | None = None,
    ) -> Any:
        del model, inputs
        return outputs


###############################
# Controller replay contracts #
###############################


class TestBuildControllerReplayRequirements:
    """Tests for controller construction contracts."""

    def test_raises_clear_error_when_replay_required_controller_has_no_replay_batch_size(self) -> None:
        controller_config = ControllerConfig(name='tbbn', kwargs={})

        with pytest.raises(ValueError, match='requires a replay-based strategy'):
            build_controller(
                controller_config=controller_config,
                train_batch_size=32,
                replay_batch_size=None,
                replay_memory_size=None,
            )

    def test_builds_replay_required_controller_when_replay_batch_size_is_available(self) -> None:
        controller_config = ControllerConfig(name='tbbn', kwargs={})

        controller = build_controller(
            controller_config=controller_config,
            train_batch_size=32,
            replay_batch_size=16,
            replay_memory_size=200,
        )

        assert isinstance(controller, PreventionController)

    def test_rejects_controller_seed_kwarg(self) -> None:
        controller_config = ControllerConfig(
            name='logit_bias',
            kwargs={
                'lr': 0.1,
                'seed': 7,
            },
        )

        with pytest.raises(ValueError, match='should not receive the following keyword arguments: seed'):
            build_controller(
                controller_config=controller_config,
                train_batch_size=32,
                replay_batch_size=None,
                replay_memory_size=None,
            )


class TestBuildBenchmark:
    """Tests for scenario-builder forwarding behavior."""

    def test_forwards_none_transform_overrides_and_backbone_image_size(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured_kwargs: dict[str, object] = {}

        def _fake_scenario_builder(**kwargs: object) -> dict[str, object]:
            captured_kwargs.update(kwargs)
            return {'ok': True}

        monkeypatch.setattr(
            builders_module,
            'get_scenario_builder',
            lambda *, scenario: _fake_scenario_builder,
        )
        experiment_config = ExperimentConfig(
            experiment_name='unit_test_experiment',
            scenario='cifar100',
            num_experiences=20,
            backbone=BackboneConfig(
                name='vit_small',
                kwargs={
                    'image_size': 384,
                },
            ),
            repair=RepairConfig(split_fraction=0.2),
            transforms=TransformsConfig(),
            evaluation=EvaluationConfig(),
            runs=[],
            dataset_path='/tmp/datasets',
            seed=7,
        )

        benchmark = build_benchmark(
            experiment_config=experiment_config,
            repair_split_fraction=0.2,
        )

        assert benchmark == {'ok': True}
        assert captured_kwargs['num_experiences'] == 20
        assert captured_kwargs['return_task_id'] is False
        assert captured_kwargs['repair_split_fraction'] == pytest.approx(0.2)
        assert captured_kwargs['dataset_path'] == '/tmp/datasets'
        assert captured_kwargs['seed'] == 7
        assert captured_kwargs['transform_random_resized_crop'] is None
        assert captured_kwargs['transform_horizontal_flip'] is None
        assert captured_kwargs['transform_image_size'] == 384

    def test_forwards_explicit_transform_overrides_without_bool_coercion(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured_kwargs: dict[str, object] = {}

        def _fake_scenario_builder(**kwargs: object) -> dict[str, object]:
            captured_kwargs.update(kwargs)
            return {'ok': True}

        monkeypatch.setattr(
            builders_module,
            'get_scenario_builder',
            lambda *, scenario: _fake_scenario_builder,
        )
        experiment_config = ExperimentConfig(
            experiment_name='unit_test_experiment',
            scenario='tiny_imagenet',
            num_experiences=10,
            backbone=BackboneConfig(name='resnet18', kwargs={}),
            repair=RepairConfig(split_fraction=0.1),
            transforms=TransformsConfig(
                random_resized_crop=True,
                horizontal_flip=False,
            ),
            evaluation=EvaluationConfig(),
            runs=[],
        )

        benchmark = build_benchmark(
            experiment_config=experiment_config,
            repair_split_fraction=0.1,
        )

        assert benchmark == {'ok': True}
        assert captured_kwargs['transform_random_resized_crop'] is True
        assert captured_kwargs['transform_horizontal_flip'] is False
        assert captured_kwargs['transform_image_size'] is None


class TestBuildControllerPlugin:
    """Tests for repair controller plugin builder behavior."""

    def test_builds_debug_plugin_without_debug_seed(self) -> None:
        plugin = build_controller_plugin(
            controller=_NoOpRepairController(),
            fit_after_experience=False,
            num_epochs=2,
            batch_size=4,
            seed=11,
            debug=True,
            debug_epochs=5,
            debug_experiences=3,
        )

        assert isinstance(plugin, DebugRepairControllerPlugin)
        assert plugin.seed == 11

    def test_requires_explicit_seed_for_repair_plugins(self) -> None:

        def _build_plugin_with_kwargs(kwargs: dict[str, Any]) -> Any:
            return build_controller_plugin(**kwargs)

        with pytest.raises(TypeError, match="missing 1 required keyword-only argument: 'seed'"):
            _build_plugin_with_kwargs({
                'controller': _NoOpRepairController(),
                'fit_after_experience': False,
                'num_epochs': 2,
                'batch_size': 4,
            })

    @pytest.mark.parametrize(
        ('debug_epochs', 'debug_experiences'),
        [
            (None, 3),
            (5, None),
        ],
    )
    def test_debug_plugin_still_requires_debug_step_metadata(
        self,
        debug_epochs: int | None,
        debug_experiences: int | None,
    ) -> None:
        with pytest.raises(ValueError, match='requires debug_epochs and debug_experiences'):
            build_controller_plugin(
                controller=_NoOpRepairController(),
                fit_after_experience=False,
                num_epochs=2,
                batch_size=4,
                seed=11,
                debug=True,
                debug_epochs=debug_epochs,
                debug_experiences=debug_experiences,
            )


class TestExplicitSeedContracts:
    """Static contract checks for explicit internal seed propagation."""

    @pytest.mark.parametrize(
        'callable_obj',
        [
            build_controller_plugin,
            ScenarioBuilder.__call__,
            ScenarioBuilder._build_scenario,
            CIFAR100ScenarioBuilder._build_scenario,
            TinyImageNetScenarioBuilder._build_scenario,
            CUB200ScenarioBuilder._build_scenario,
            ImageNetRScenarioBuilder._build_scenario,
            ScenarioBuilder._add_repair_stream,
        ],
    )
    def test_seed_boundary_signatures_do_not_default_seed(self, callable_obj: object) -> None:
        signature = inspect.signature(callable_obj)

        assert signature.parameters['seed'].default is inspect._empty


class TestBuildBackbone:
    """Tests for backbone instantiation."""

    def test_builds_vit_backbone_with_constructor_kwargs(self) -> None:
        backbone = build_backbone(
            name='vit_small',
            num_classes=11,
            backbone_kwargs={
                'image_size': 32,
                'patch_size': 4,
                'dropout': 0.1,
            },
        )
        x = torch.randn(2, 3, 32, 32)

        logits = backbone(x)

        assert isinstance(backbone, torch.nn.Module)
        assert logits.shape == (2, 11)


class TestBuildOptimizer:
    """Tests for optimizer construction and validation."""

    def test_builds_adamw_optimizer_with_beta_sequence(self) -> None:
        model = torch.nn.Linear(4, 2)

        optimizer, optimizer_kwargs = build_optimizer(
            model=model,
            optimizer_config=OptimizerConfig(
                name='adamw',
                kwargs={
                    'lr': 5e-4,
                    'betas': [0.9, 0.999],
                    'eps': 1e-8,
                    'weight_decay': 1e-4,
                },
            ),
        )

        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.defaults['lr'] == pytest.approx(5e-4)
        assert optimizer.defaults['betas'] == pytest.approx((0.9, 0.999))
        assert optimizer_kwargs['betas'] == [0.9, 0.999]

    def test_rejects_string_beta_literal_for_adamw(self) -> None:
        model = torch.nn.Linear(4, 2)

        with pytest.raises(ValueError, match='YAML sequence'):
            build_optimizer(
                model=model,
                optimizer_config=OptimizerConfig(
                    name='adamw',
                    kwargs={
                        'betas': '(0.9, 0.999)',
                    },
                ),
            )


class TestBuildLRSchedulerPlugin:
    """Tests for LR scheduler plugin validation."""

    def test_rejects_missing_total_epochs_for_warmup_cosine(self) -> None:
        with pytest.raises(ValueError, match='total_epochs'):
            build_lr_scheduler_plugin(
                name='warmup_cosine',
                scheduler_kwargs={
                    'warmup_epochs': 5,
                },
            )
