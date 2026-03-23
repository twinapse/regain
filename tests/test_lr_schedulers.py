"""
Tests for local LR schedulers and scheduler plugins.
"""

from types import SimpleNamespace

import pytest
import torch

from regain.experiments.builders import build_lr_scheduler_plugin


def _collect_warmup_cosine_lrs(*, total_epochs: int, warmup_epochs: int, min_lr: float) -> list[float]:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    plugin = build_lr_scheduler_plugin(
        name='warmup_cosine',
        scheduler_kwargs={
            'warmup_epochs': warmup_epochs,
            'min_lr': min_lr,
        },
        initial_lr=0.5,
        total_epochs=total_epochs,
    )
    strategy = SimpleNamespace(optimizer=optimizer)
    plugin.before_training_exp(strategy)
    lrs = [float(optimizer.param_groups[0]['lr'])]
    for _ in range(total_epochs - 1):
        plugin.after_training_epoch(strategy)
        lrs.append(float(optimizer.param_groups[0]['lr']))
    return lrs


class TestWarmupCosineLRSchedulerPlugin:
    def test_tracks_expected_schedule_values(self) -> None:
        lrs = _collect_warmup_cosine_lrs(
            total_epochs=6,
            warmup_epochs=2,
            min_lr=0.1,
        )

        assert lrs[0] == pytest.approx(0.25)
        assert lrs[1] == pytest.approx(0.5)
        assert lrs[2] == pytest.approx(0.5)
        assert lrs[3] == pytest.approx(0.4)
        assert lrs[4] == pytest.approx(0.2)
        assert lrs[5] == pytest.approx(0.1)

    def test_resets_state_between_experiences(self) -> None:
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        plugin = build_lr_scheduler_plugin(
            name='warmup_cosine',
            scheduler_kwargs={
                'warmup_epochs': 2,
                'min_lr': 0.1,
            },
            initial_lr=0.5,
            total_epochs=6,
        )
        strategy = SimpleNamespace(optimizer=optimizer)

        plugin.before_training_exp(strategy)
        plugin.after_training_epoch(strategy)
        plugin.after_training_epoch(strategy)
        assert optimizer.param_groups[0]['lr'] == pytest.approx(0.5)

        plugin.before_training_exp(strategy)
        assert optimizer.param_groups[0]['lr'] == pytest.approx(0.25)

    def test_single_epoch_cosine_phase_starts_at_base_lr(self) -> None:
        lrs = _collect_warmup_cosine_lrs(
            total_epochs=2,
            warmup_epochs=1,
            min_lr=0.1,
        )

        assert lrs == pytest.approx([0.5, 0.5])
