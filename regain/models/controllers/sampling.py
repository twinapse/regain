"""
Sampling utilities for building balanced datasets and optional repair buffers.

The repair buffer machinery (with eviction and sampling policies) is useful for real-world deployments where you
need a fixed-capacity memory and do not know how many classes will appear in the future. The default experiments
use a "repair stream" instead of a buffer.
"""
from collections import defaultdict
from collections import deque
# TODO: Check Avalanche docs for built-in balanced sampling utilities
#       (keep in mind that controllers should be Avalanche-agnostic)
from typing import Iterable, Protocol, Sequence

import numpy as np
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset
from torch.utils.data import Subset

from regain.utils import extract_targets
from regain.utils import get_targets

__all__ = [
    'IndexDataset',
    'IndexItem',
    'ClassPools',
    'add_dataset_to_class_pools',
    'get_balanced_subset',
    'build_balanced_dataset',
    'RepairBuffer',
    'RepairBufferBalancedFIFOPolicy',
    'RepairBufferFIFOPolicy',
    'RepairBufferKeepNewBalancedPolicy',
    'RepairBufferPolicy',
    'RepairBufferView',
]

IndexItem = tuple[Dataset, int]


class IndexDataset(Dataset):
    """ Dataset backed by a list of (dataset, index) pairs. """

    def __init__(self, items: list[IndexItem]):
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int):
        ds, j = self._items[idx]
        return ds[j]


ClassPools = dict[int, list[IndexItem]]


def add_dataset_to_class_pools(
    dataset: Dataset,
    *,
    class_pools: ClassPools,
    targets: list[int] | None = None,
) -> list[int]:
    if targets is None:
        targets = extract_targets(dataset)
    cur_classes = sorted({int(t) for t in targets})
    for i, t in enumerate(targets):
        class_pools.setdefault(int(t), []).append((dataset, i))
    return cur_classes


def get_balanced_subset(dataset: Dataset, max_per_class: int, rng: np.random.Generator) -> Subset:
    """
    Build a balanced subset from a dataset given a maximum number of examples per class.

    Args:
        dataset: Dataset to sample from.
        max_per_class: Maximum number of examples per class.
        rng: NumPy random number generator.

    Returns:
        Subset containing sampled indices.
    """
    targets = get_targets(dataset)
    classes: np.ndarray = np.unique(targets)
    indices: list[int] = []
    for class_id in classes:
        class_indices: np.ndarray = np.where(targets == class_id)[0]
        if class_indices.size > max_per_class:
            chosen = rng.choice(class_indices, size=max_per_class, replace=False)
        else:
            chosen = class_indices
        indices.extend(chosen.tolist())
    return Subset(dataset, indices)


def build_balanced_dataset(
    datasets: Sequence[Dataset],
    max_per_class: int,
    rng: np.random.Generator,
) -> Dataset | None:
    """
    Concatenate balanced subsets from a sequence of datasets.

    Args:
        datasets: Sequence of datasets to draw from.
        max_per_class: Maximum number of examples per class in each dataset.
        rng: Random generator to split per dataset.

    Returns:
        Concatenated balanced dataset or None if no data.
    """
    subsets: list[Dataset] = []
    for dataset in datasets:
        child_seed = int(rng.integers(0, 2**32 - 1))
        child_rng = np.random.default_rng(child_seed)
        subsets.append(get_balanced_subset(dataset, max_per_class, child_rng))
    if not subsets:
        return None
    if len(subsets) == 1:
        return subsets[0]
    return ConcatDataset(subsets)


class RepairBufferPolicy(Protocol):
    """
    Policy interface for repair buffer maintenance and eviction.

    Policies are owned and invoked by the buffer/plugin, not by controllers.
    """

    def on_ingest(self, *, buffer: 'RepairBuffer') -> None:
        """
        Hook triggered after new items have been ingested into the buffer.

        Args:
            buffer (RepairBuffer): Buffer that ingested new data.

        Returns:
            None.
        """

    def evict(self, *, buffer: 'RepairBuffer') -> None:
        """
        Evict items after capacity has been exceeded.

        Args:
            buffer (RepairBuffer): Buffer to evict from.

        Returns:
            None.
        """


class RepairBufferFIFOPolicy:
    """
    Oldest-first eviction policy based on insertion order.
    """

    def on_ingest(
            self,
            *,
            buffer: 'RepairBuffer',  # pylint: disable=unused-argument
    ) -> None:
        """
        No-op.

        Args:
            buffer (RepairBuffer): Buffer that ingested new data.

        Returns:
            None.
        """
        return

    def evict(self, *, buffer: 'RepairBuffer') -> None:
        """
        Oldest-first removal.

        Args:
            buffer (RepairBuffer): Buffer to evict from.

        Raises:
            RuntimeError: If eviction fails to remove the oldest item.

        Returns:
            None.
        """
        while len(buffer) > buffer.capacity:
            oldest = buffer.pop_oldest()
            if oldest is None:
                raise RuntimeError('Repair buffer eviction failed: missing oldest item')


class RepairBufferBalancedFIFOPolicy:
    """
    Balanced FIFO eviction policy.

    When the buffer exceeds capacity, evict the oldest item from the currently most
    populated class (ties broken by global insertion order among tied classes).

    This encourages approximate class balancing in the stored buffer while keeping
    FIFO-like behavior (preference for older samples).
    """

    def on_ingest(
            self,
            *,
            buffer: 'RepairBuffer',  # pylint: disable=unused-argument
    ) -> None:
        """
        No-op.

        Args:
            buffer (RepairBuffer): Buffer that ingested new data.

        Returns:
            None.
        """
        return

    # TODO: This eviction is potentially expensive:
    #         - Each eviction recomputes counts by scanning all items (`O(n)`).
    #         - `remove_item` does linear removals from deques (`O(n)` worst-case).
    #       If the buffer exceeds capacity by a lot in one ingest, eviction can become noticeably slow.
    def evict(self, *, buffer: 'RepairBuffer') -> None:
        """
        Evict items to enforce buffer capacity while encouraging class balance.

        Strategy:
          1) Compute per-class counts from the global insertion order snapshot.
          2) Find the max count and the set of classes achieving it.
          3) Remove the oldest globally-inserted item among those classes.

        Args:
            buffer (RepairBuffer): Buffer to evict from.

        Raises:
            RuntimeError: If eviction fails to remove items to satisfy capacity.

        Returns:
            None.
        """
        while len(buffer) > buffer.capacity:
            items = buffer.all_items()
            if not items:
                raise RuntimeError('Repair buffer eviction failed: empty buffer while over capacity')

            counts: dict[int, int] = {}
            for class_id, _ in items:
                cls = int(class_id)
                counts[cls] = counts.get(cls, 0) + 1

            max_count = max(counts.values())
            candidates = {cls for cls, c in counts.items() if c == max_count}

            chosen: tuple[int, IndexItem] | None = None
            for class_id, item in items:
                cls = int(class_id)
                if cls in candidates:
                    chosen = (cls, item)
                    break

            if chosen is None:
                # Defensive fallback: behave like FIFO to ensure progress.
                oldest = buffer.pop_oldest()
                if oldest is None:
                    raise RuntimeError('Repair buffer eviction failed: missing oldest item')
                continue

            cls, item = chosen

            try:
                buffer.remove(class_id=cls, item=item)
            except ValueError as exc:
                raise RuntimeError(f'Repair buffer eviction failed: item not found ({item})') from exc


class RepairBufferKeepNewBalancedPolicy:
    """
    Keep-new, balanced-old eviction policy.

    On each ingest, the items added in that ingest are treated as "new" and are
    protected from eviction *for that eviction step*.

    If capacity is exceeded:
      1) Evict from OLD items only (non-new), distributing removals as evenly as
         possible across the classes present in the old portion.
      2) If old items are insufficient to restore capacity (e.g., new batch alone
         exceeds capacity), evict from the remaining items as well, again in a
         balanced-per-class way.
    """

    def __init__(self) -> None:
        # Buffer size after the *previous* ingest cycle finished (post-eviction).
        self._prev_len: int = 0
        # IDs (identity, not equality) of items ingested in the most recent ingest.
        self._protected_item_ids: set[int] = set()

    @staticmethod
    def _item_id(item: IndexItem) -> int:
        # Use identity of the (dataset, idx) tuple stored in the buffer.
        # This avoids collisions if the same (dataset, idx) appears more than once.
        return id(item)

    @staticmethod
    def _compute_keep_targets(counts: dict[int, int], keep_total: int) -> dict[int, int]:
        """
        Given per-class capacities `counts` and a desired total to keep `keep_total`,
        return per-class keep targets that are as balanced as possible, without
        exceeding each class' available count.
        """
        counts = {int(k): int(v) for k, v in counts.items()}
        total = sum(counts.values())

        if keep_total <= 0:
            return {c: 0 for c in counts}
        if keep_total >= total:
            return counts.copy()

        targets: dict[int, int] = {c: 0 for c in counts}
        remaining = int(keep_total)

        # Saturate classes with small caps first (water-filling with upper bounds).
        remaining_classes = sorted(counts.keys(), key=lambda c: counts[c])  # ascending cap

        while remaining_classes:
            n = len(remaining_classes)
            base = remaining // n

            # Saturate any class whose cap is below the current base.
            saturated = [c for c in remaining_classes if counts[c] < base]
            if saturated:
                sat_set = set(saturated)
                for c in saturated:
                    targets[c] = counts[c]
                    remaining -= counts[c]
                remaining_classes = [c for c in remaining_classes if c not in sat_set]
                continue

            # Now all remaining classes can take at least `base`.
            for c in remaining_classes:
                targets[c] = base
            remaining -= base * n

            # Distribute the leftover one-by-one among classes with slack.
            slack = [c for c in remaining_classes if targets[c] < counts[c]]
            slack.sort()  # deterministic

            idx = 0
            while remaining > 0 and slack:
                c = slack[idx % len(slack)]
                targets[c] += 1
                remaining -= 1

                if targets[c] >= counts[c]:
                    slack.remove(c)
                    if slack:
                        idx = idx % len(slack)
                else:
                    idx += 1
            break

        # Defensive sanity checks
        if sum(targets.values()) != keep_total:
            # Fallback: keep as much as possible in class-id order.
            targets = {c: 0 for c in counts}
            remaining = keep_total
            for c in sorted(counts.keys()):
                k = min(counts[c], remaining)
                targets[c] = k
                remaining -= k
                if remaining <= 0:
                    break

        return targets

    @staticmethod
    def _counts_by_class(buffer: 'RepairBuffer', *, protect_ids: set[int] | None) -> dict[int, int]:
        """
        Count items per class, optionally excluding protected item IDs.
        """
        counts: dict[int, int] = {}
        protect_ids = protect_ids or set()
        for cls, item in buffer.all_items():
            if id(item) in protect_ids:
                continue
            c = int(cls)
            counts[c] = counts.get(c, 0) + 1
        return counts

    @staticmethod
    def _remove_oldest_from_class(
        buffer: 'RepairBuffer',
        *,
        class_id: int,
        num_remove: int,
        protect_ids: set[int] | None,
    ) -> None:
        """
        Remove `num_remove` oldest items from `class_id`, skipping protected ones if provided.
        """
        if num_remove <= 0:
            return
        protect_ids = protect_ids or set()

        items = buffer.class_items(class_id=int(class_id))  # oldest -> newest
        to_remove: list[IndexItem] = []
        for it in items:
            if id(it) in protect_ids:
                continue
            to_remove.append(it)
            if len(to_remove) >= num_remove:
                break

        if len(to_remove) < num_remove:
            raise RuntimeError(f'KeepNewBalancedPolicy: asked to remove {num_remove} from class {class_id}, '
                               f'but only found {len(to_remove)} evictable items.')

        for it in to_remove:
            buffer.remove(class_id=int(class_id), item=it)

    def on_ingest(self, *, buffer: 'RepairBuffer') -> None:
        cur_len = len(buffer)
        added = max(0, cur_len - int(self._prev_len))

        if added <= 0:
            self._protected_item_ids.clear()
            self._prev_len = cur_len
            return

        items = buffer.all_items()
        tail = items[-added:] if added < len(items) else items
        self._protected_item_ids = {self._item_id(item) for _, item in tail}

        # If we're not going to evict, protection should not leak into the next ingest.
        if cur_len <= buffer.capacity:
            self._protected_item_ids.clear()
            self._prev_len = cur_len

    def evict(self, *, buffer: 'RepairBuffer') -> None:
        excess = len(buffer) - buffer.capacity
        if excess <= 0:
            self._protected_item_ids.clear()
            self._prev_len = len(buffer)
            return

        # ---- Stage 1: evict ONLY from old items (non-protected), balanced across old classes.
        old_counts = self._counts_by_class(buffer, protect_ids=self._protected_item_ids)
        old_total = sum(old_counts.values())

        if old_total > 0:
            remove_old = min(int(excess), int(old_total))
            keep_old_total = int(old_total) - int(remove_old)

            keep_targets = self._compute_keep_targets(old_counts, keep_old_total)
            for cls in sorted(old_counts.keys()):
                remove_n = old_counts[cls] - keep_targets.get(cls, 0)
                if remove_n > 0:
                    self._remove_oldest_from_class(
                        buffer,
                        class_id=int(cls),
                        num_remove=int(remove_n),
                        protect_ids=self._protected_item_ids,
                    )

            excess = len(buffer) - buffer.capacity  # recompute (safer than decrement bookkeeping)

        # ---- Stage 2: if still over capacity, we must evict from remaining items too (balanced).
        if excess > 0:
            all_counts = self._counts_by_class(buffer, protect_ids=None)
            total = sum(all_counts.values())
            keep_total = max(0, int(total) - int(excess))  # should equal capacity

            keep_targets = self._compute_keep_targets(all_counts, keep_total)
            for cls in sorted(all_counts.keys()):
                remove_n = all_counts[cls] - keep_targets.get(cls, 0)
                if remove_n > 0:
                    self._remove_oldest_from_class(
                        buffer,
                        class_id=int(cls),
                        num_remove=int(remove_n),
                        protect_ids=None,  # allow removing anything now
                    )

        # Final safety: ensure capacity is satisfied (guarantee progress).
        while len(buffer) > buffer.capacity:
            oldest = buffer.pop_oldest()
            if oldest is None:
                raise RuntimeError('KeepNewBalancedPolicy eviction failed: empty buffer while over capacity')

        # Protection only applies to this ingest/eviction cycle.
        self._protected_item_ids.clear()
        self._prev_len = len(buffer)


class RepairBufferView:
    """
    Concrete read-only view of a RepairBuffer.

    Important:
      - This class intentionally does NOT expose any mutation methods.
      - Controllers should only receive RepairBufferView, never the underlying RepairBuffer.

    Args:
        buffer (RepairBuffer): Buffer to wrap.
    """

    def __init__(self, buffer: 'RepairBuffer') -> None:
        self.__len_fn = buffer.__len__
        self.__class_ids_fn = buffer.class_ids
        self.__sample_balanced_fn = buffer._sample_balanced
        self.__sample_per_class_fn = buffer._sample_per_class
        self.__all_items_fn = buffer.all_items

    def __len__(self) -> int:
        return self.__len_fn()

    def class_ids(self) -> list[int]:
        """
        Return a sorted list of class IDs currently in the buffer.

        Returns:
            list[int]: Sorted list of class IDs.
        """
        return self.__class_ids_fn()

    def sample_balanced(
        self,
        *,
        max_items: int,
        rng: np.random.Generator,
        classes: Iterable[int] | None = None,
    ) -> list[IndexItem]:
        """
        Sample up to `max_items` items, approximately balanced across `classes`.

        Args:
            max_items (int): Maximum number of items to sample.
            rng (np.random.Generator): Random number generator.
            classes (Iterable[int] | None): Subset of classes to sample from (defaults to all).

        Returns:
            list[IndexItem]: Sampled items.
        """
        return self.__sample_balanced_fn(max_items=max_items, rng=rng, classes=classes)

    def sample_per_class(
        self,
        *,
        max_per_class: int,
        rng: np.random.Generator,
        classes: Iterable[int] | None = None,
    ) -> list[IndexItem]:
        """
        Sample up to `max_per_class` items per class from the buffer.

        Args:
            max_per_class (int): Maximum number of items to sample per class.
            rng (np.random.Generator): Random number generator.
            classes (Iterable[int] | None): Subset of classes to sample from (defaults to

        Returns:
            list[IndexItem]: Sampled items.
        """
        return self.__sample_per_class_fn(max_per_class=max_per_class, rng=rng, classes=classes)

    def all_items(self) -> list[IndexItem]:
        """
        Return a snapshot of all items in insertion order.

        Returns:
            list[IndexItem]: All items in insertion order.
        """
        return [item for _, item in self.__all_items_fn()]

    @staticmethod
    def to_dataset(items: list[IndexItem]) -> Dataset:
        """
        Build a Dataset from a list of IndexItems.

        Args:
            items (list[IndexItem]): Items to include in the dataset.

        Returns:
            Dataset: Dataset containing the specified items.
        """
        return IndexDataset(items)


class RepairBuffer:
    """
    Mutable repair buffer that enforces a fixed capacity across the whole experiment.

    Storage:
      - keeps (dataset, index) pairs (IndexItem) so it does not copy tensors.
      - maintains per-class pools for balanced sampling.
      - maintains a global insertion-order queue for eviction and bookkeeping.

    Notes:
      - Designed for streaming / infinite scenarios.
      - Eviction happens after ingesting the latest dataset, so the buffer may exceed capacity transiently.
      - The default policy is balanced FIFO (class-balanced oldest-first), but custom policies can be provided.

    Args:
        capacity (int): Maximum number of examples to store.
        policy (RepairBufferPolicy | None): Eviction/maintenance policy (defaults to RepairBufferBalancedFIFOPolicy()).
    """

    def __init__(self, *, capacity: int, policy: RepairBufferPolicy | None = None) -> None:
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError('Capacity must be a positive integer')

        self.capacity = capacity
        self.policy: RepairBufferPolicy = policy if policy is not None else RepairBufferBalancedFIFOPolicy()

        # class_id -> deque of IndexItem in insertion order for that class
        self._class_pools: dict[int, deque[IndexItem]] = defaultdict(deque)
        # Global insertion order of (class_id, IndexItem)
        self._insertion_queue: deque[tuple[int, IndexItem]] = deque()

        # Cache a stable view instance (controllers can keep the view; it stays up-to-date).
        self._view = RepairBufferView(self)

    def __len__(self) -> int:
        return len(self._insertion_queue)

    def view(self) -> RepairBufferView:
        """
        Return a read-only view suitable for passing to controllers.

        Args:
            None.

        Returns:
            RepairBufferView: Read-only view of this buffer.
        """
        return self._view

    def ingest(self, dataset: Dataset | None) -> list[int]:
        """
        Ingest a repair dataset into the buffer, enforcing capacity via the policy.

        Args:
            dataset (Dataset | None): Repair dataset to ingest.

        Returns:
            list[int]: Sorted list of class IDs observed in the incoming dataset (empty if dataset is None).
        """
        if dataset is None:
            return []

        targets = extract_targets(dataset)
        if not targets:
            return []

        new_classes = sorted({int(t) for t in targets})

        # Insert all items, then enforce capacity once (cheaper than enforcing per-item).
        for i, t in enumerate(targets):
            cls = int(t)
            item: IndexItem = (dataset, int(i))
            self._class_pools[cls].append(item)
            self._insertion_queue.append((cls, item))

        self.policy.on_ingest(buffer=self)

        if len(self) > self.capacity:
            self.policy.evict(buffer=self)

        if len(self) > self.capacity:
            raise RuntimeError('Repair buffer capacity exceeded after eviction. '
                               f'Ensure {type(self.policy).__name__} evicts enough items')
        return new_classes

    def remove(self, *, class_id: int, item: IndexItem) -> None:
        """
        Remove an item from the buffer.

        Args:
            class_id (int): Class ID for the item.
            item (IndexItem): Item to remove.

        Raises:
            ValueError: If the item is not present in the buffer.
            RuntimeError: If the buffer is internally inconsistent.
        """
        # Ensure the item exists in global order
        try:
            self._insertion_queue.index((class_id, item))
        except ValueError as exc:
            raise ValueError('Item not found in repair buffer') from exc

        # Ensure the item exists in the class pool
        pool = self._class_pools.get(class_id)
        if pool is None:
            raise RuntimeError('Repair buffer state is inconsistent: missing class pool')
        try:
            pool.index(item)
        except ValueError as exc:
            raise RuntimeError('Repair buffer state is inconsistent: missing item in class pool') from exc

        # Remove the item from both structures
        self._insertion_queue.remove((class_id, item))
        pool.remove(item)
        if not pool:
            self._class_pools.pop(class_id, None)

    def pop_oldest(self) -> tuple[int, IndexItem] | None:
        """
        Remove and return the oldest inserted item.

        Args:
            None.

        Raises:
            RuntimeError: If the buffer is internally inconsistent.

        Returns:
            tuple[int, IndexItem] | None: Oldest item, or None if the buffer is empty.
        """
        if not self._insertion_queue:
            return None

        class_id, item = self._insertion_queue.popleft()
        pool = self._class_pools.get(int(class_id))
        if pool is None or not pool:
            raise RuntimeError('Repair buffer state is inconsistent: missing class pool.')

        pool_item = pool.popleft()
        if pool_item != item:
            raise RuntimeError('Repair buffer state is inconsistent: oldest item mismatch.')

        if not pool:
            self._class_pools.pop(int(class_id), None)

        return int(class_id), item

    def peek_oldest(self) -> tuple[int, IndexItem] | None:
        """
        Return the oldest inserted item without removing it.

        Args:
            None.

        Returns:
            tuple[int, IndexItem] | None: Oldest item, or None if the buffer is empty.
        """
        if not self._insertion_queue:
            return None
        return self._insertion_queue[0]

    def class_ids(self) -> list[int]:
        """
        Return a sorted list of class IDs currently in the buffer.

        Args:
            None.

        Returns:
            list[int]: Sorted list of class IDs.
        """
        return sorted(self._class_pools.keys())

    def class_items(self, *, class_id: int) -> list[IndexItem]:
        """
        Return a snapshot of items for a class in insertion order.

        Args:
            class_id (int): Class ID to inspect.

        Returns:
            list[IndexItem]: Items for the requested class (empty if missing).
        """
        pool = self._class_pools.get(int(class_id))
        if pool is None:
            return []
        return list(pool)

    def all_items(self) -> list[tuple[int, IndexItem]]:
        """
        Return a snapshot of all items in insertion order.

        Args:
            None.

        Returns:
            list[tuple[int, IndexItem]]: All items in insertion order.
        """
        return list(self._insertion_queue)

    @staticmethod
    def _split_evenly(num_groups: int, total: int) -> list[int]:
        """
        Split `total` items across `num_groups` as evenly as possible.

        The result has length `num_groups` and sums to `total`.
        If `total` is not divisible by `num_groups`, earlier groups (lower indices) receive one extra item.

        Args:
            num_groups (int): Number of groups to split into.
            total (int): Total number of items to split.

        Returns:
            list[int]: List of lengths per group summing to `total`.
        """

        if num_groups <= 0 or total <= 0:
            return []
        base = total // num_groups
        rem = total % num_groups
        group_lengths = [base] * num_groups
        # TODO: First groups get the remainder, which can bias sampling when total is not divisible.
        for i in range(rem):
            group_lengths[i] += 1
        return group_lengths

    def _sample_balanced(
        self,
        *,
        max_items: int,
        rng: np.random.Generator,
        classes: Iterable[int] | None = None,
    ) -> list[IndexItem]:
        """
        Sample up to `max_items` items, approximately balanced across `classes`.

        Args:
            max_items (int): Maximum number of items to sample.
            rng (np.random.Generator): Random number generator.
            classes (Iterable[int] | None): Subset of classes to sample from (defaults to all).

        Returns:
            list[IndexItem]: Sampled items.
        """
        if max_items <= 0 or len(self) <= 0:
            return []

        if classes is None:
            classes_list = self.class_ids()
        else:
            classes_list = [int(c) for c in classes]
            classes_list = list(dict.fromkeys(classes_list))
            classes_list = [c for c in classes_list if c in self._class_pools]

        if not classes_list:
            return []

        lengths = self._split_evenly(num_groups=len(classes_list), total=int(max_items))
        chosen: list[IndexItem] = []

        for cls, k in zip(classes_list, lengths):
            pool = self._class_pools.get(int(cls))
            if pool is None or not pool or k <= 0:
                continue

            pool_list = list(pool)
            k_eff = min(int(k), len(pool_list))
            idx = rng.choice(len(pool_list), size=k_eff, replace=False)
            chosen.extend([pool_list[int(j)] for j in idx.tolist()])

        return chosen

    def _sample_per_class(
        self,
        *,
        max_per_class: int,
        rng: np.random.Generator,
        classes: Iterable[int] | None = None,
    ) -> list[IndexItem]:
        """
        Sample up to `max_per_class` items per class from the buffer.

        Args:
            max_per_class (int): Maximum number of items to sample per class.
            rng (np.random.Generator): Random number generator.
            classes (Iterable[int] | None): Subset of classes to sample from (defaults to all).

        Returns:
            list[IndexItem]: Sampled items.
        """
        if max_per_class <= 0 or len(self) <= 0:
            return []

        if classes is None:
            classes_list = self.class_ids()
        else:
            classes_list = [int(c) for c in classes]
            classes_list = list(dict.fromkeys(classes_list))
            classes_list = [c for c in classes_list if c in self._class_pools]

        items: list[IndexItem] = []

        for cls in classes_list:
            pool = self._class_pools.get(int(cls))
            if pool is None or not pool:
                continue

            pool_list = list(pool)
            k_eff = min(int(max_per_class), len(pool_list))
            idx = rng.choice(len(pool_list), size=k_eff, replace=False)
            items.extend([pool_list[int(j)] for j in idx.tolist()])

        return items
