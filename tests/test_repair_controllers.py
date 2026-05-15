"""
Tests for repair controller helpers and controller-adjacent diagnostics.
"""

import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from regain.constants import _DEBUG_N_SAMPLES
from regain.constants import _DEBUG_TOP1
from regain.debug.metrics import compute_repair_diagnostics
from regain.models.controllers import RepairController
from regain.models.controllers.repair.common import build_repair_dataloader


class _ParityClassifier(nn.Module):
    """Toy classifier that maps scalar parity to two logit columns."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parity = torch.remainder(x, 2.0)
        return torch.cat([1.0 - parity, parity], dim=1)


class _NoOpRepairController(RepairController):
    """Repair controller test double that leaves logits unchanged."""

    def fit_on_repair_data(
        self,
        *,
        model: nn.Module,
        repair_dataset: Dataset | None,
        new_classes: list[int],
        num_epochs: int,
        batch_size: int,
    ) -> None:
        del model, repair_dataset, new_classes, num_epochs, batch_size

    def correct_outputs(
        self,
        *,
        outputs: Any,
        model: nn.Module | None = None,
        inputs: Any | None = None,
    ) -> Any:
        del model, inputs
        return outputs


class _ToyRepairDataset(Dataset):
    """Dataset helper that mirrors small repair-dataset behavior."""

    def __init__(self, *, targets: list[int], original_indices: list[int]) -> None:
        self.targets = list(targets)
        self.original_indices = list(original_indices)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return torch.tensor([float(idx)]), int(self.targets[idx])


class TestBuildRepairDataloader:
    """Tests for repair dataloader behavior shared by controllers."""

    @staticmethod
    def _collect_targets_for_seed(*, dataset: Dataset, seed: int) -> list[int]:
        torch_state = torch.random.get_rng_state()
        try:
            torch.manual_seed(int(seed))
            dataloader = build_repair_dataloader(
                repair_dataset=dataset,
                batch_size=3,
                shuffle=True,
            )
            assert dataloader is not None

            targets: list[int] = []
            for _, batch_targets in dataloader:
                targets.extend(int(value) for value in batch_targets.tolist())
            return targets
        finally:
            torch.random.set_rng_state(torch_state)

    def test_shuffle_uses_global_torch_seed(self) -> None:
        dataset = _ToyRepairDataset(
            targets=list(range(12)),
            original_indices=list(range(12)),
        )

        order_a = self._collect_targets_for_seed(dataset=dataset, seed=7)
        order_b = self._collect_targets_for_seed(dataset=dataset, seed=7)
        order_c = self._collect_targets_for_seed(dataset=dataset, seed=11)

        assert order_a == order_b
        assert order_a != order_c


class TestRepairControllerDebugMetrics:
    """Tests for debug diagnostic behavior around repair controllers."""

    def test_compute_repair_diagnostics_remains_deterministic(self) -> None:
        dataset = _ToyRepairDataset(
            targets=[0, 1, 0, 1],
            original_indices=[0, 1, 2, 3],
        )
        model = _ParityClassifier()
        controller = _NoOpRepairController()

        metrics_a = compute_repair_diagnostics(
            model=model,
            controller=controller,
            dataset=dataset,
            batch_size=2,
            debug_seed=7,
            apply_controller=True,
        )
        metrics_b = compute_repair_diagnostics(
            model=model,
            controller=controller,
            dataset=dataset,
            batch_size=2,
            debug_seed=7,
            apply_controller=True,
        )

        assert metrics_a == metrics_b
        assert metrics_a[_DEBUG_N_SAMPLES] == 4
        assert metrics_a[_DEBUG_TOP1] == 1.0

    def test_compute_repair_diagnostics_preserves_rng_state(self) -> None:
        dataset = _ToyRepairDataset(
            targets=[0, 1, 0, 1],
            original_indices=[0, 1, 2, 3],
        )
        model = _ParityClassifier()
        controller = _NoOpRepairController()

        # Prime each RNG to a non-trivial state so the assertion is sensitive.
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        random.random()
        np.random.rand()
        torch.rand(1)

        python_state_before = random.getstate()
        numpy_state_before = np.random.get_state()
        torch_state_before = torch.random.get_rng_state()

        compute_repair_diagnostics(
            model=model,
            controller=controller,
            dataset=dataset,
            batch_size=2,
            debug_seed=7,
            apply_controller=False,
        )

        python_state_after = random.getstate()
        numpy_state_after = np.random.get_state()
        torch_state_after = torch.random.get_rng_state()

        assert python_state_after == python_state_before
        assert numpy_state_after[0] == numpy_state_before[0]
        assert np.array_equal(numpy_state_after[1], numpy_state_before[1])
        assert numpy_state_after[2:] == numpy_state_before[2:]
        assert torch.equal(torch_state_after, torch_state_before)
