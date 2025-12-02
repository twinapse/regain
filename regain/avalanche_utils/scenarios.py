from abc import ABC
from abc import abstractmethod
import dataclasses
import inspect
from pathlib import Path

# SplitCIFAR100 (in our pinned Avalanche commit) returns an NCScenario built on the
# deprecated dataset_scenario stack (StreamDef comes from avalanche.benchmarks.scenarios.deprecated.*).
# We therefore use the matching deprecated LazyDatasetSequence / ClassificationStream types here.
from avalanche.benchmarks import NCExperience
from avalanche.benchmarks import NCScenario
from avalanche.benchmarks import SplitCIFAR100
from avalanche.benchmarks.scenarios.deprecated.classification_scenario import ClassificationStream
from avalanche.benchmarks.scenarios.deprecated.dataset_scenario import StreamDef
from avalanche.benchmarks.scenarios.deprecated.lazy_dataset_sequence import LazyDatasetSequence
import numpy as np

from regain.registry import get_scenario_builder_path
from regain.registry import import_symbol

__all__ = [
    'ScenarioBuilder',
    'get_scenario_builder',
    'get_num_classes_from_experience',
]


class ScenarioBuilder(ABC):
    """
    Base class for callable scenario builders.
    """

    @abstractmethod
    def __call__(
        self,
        *,
        num_experiences: int,
        return_task_id: bool,
        repair_budget_per_class: int = 0,
        dataset_path: str | Path | None = None,
        seed: int = 1,
    ) -> NCScenario:
        """
        Create an Avalanche `NCScenario` for class-incremental or task-incremental learning.
        Setting `return_task_id=False` corresponds to class-incremental learning with no task IDs.

        Args:
            num_experiences (int): Number of experiences to split the classes into.
            return_task_id (bool): Whether Avalanche should emit task IDs with each sample.
            repair_budget_per_class (int): Number of repair samples per class to allocate to repair data.
                                           If greater than 0, each experience will have a repair stream entry.
                                           This dataset will be disjoint from the training dataset.
            dataset_path (str | Path | None): Optional root directory for storing the dataset.
            seed (int): Random seed controlling the split and dataloader shuffling.

        Returns:
            NCScenario: A new Avalanche `NCScenario`.
        """
        raise NotImplementedError

    def validate_common_args(self, *, num_experiences: int, repair_budget_per_class: int) -> None:
        """
        Validate arguments shared across scenario builders.

        Args:
            num_experiences (int): Number of experiences to split the classes into.
            repair_budget_per_class (int): Number of repair samples per class to allocate to repair data.

        Returns:
            None

        Raises:
            ValueError: If `num_experiences` is not a positive integer.
            ValueError: If `repair_budget_per_class` is negative.
        """
        if not isinstance(num_experiences, int) or num_experiences <= 0:
            raise ValueError('`num_experiences` must be a positive integer.')
        if int(repair_budget_per_class) < 0:
            raise ValueError('`repair_budget_per_class` must be non-negative.')

    def maybe_add_repair_stream(
        self,
        *,
        benchmark: NCScenario,
        repair_budget_per_class: int,
        seed: int,
    ) -> NCScenario:
        """
        Optionally add a repair stream to a scenario.

        Args:
            benchmark (NCScenario): Benchmark scenario to augment.
            repair_budget_per_class (int): Number of repair samples per class to allocate to repair data.
            seed (int): Random seed controlling the split and dataloader shuffling.

        Returns:
            NCScenario: The original benchmark or an augmented benchmark with a repair stream.
        """
        if int(repair_budget_per_class) <= 0:
            return benchmark
        return self._add_repair_stream_inplace(
            benchmark,
            repair_budget_per_class=int(repair_budget_per_class),
            seed=seed,
        )

    @staticmethod
    def _streamdef_replace(sd: StreamDef, **updates: object) -> StreamDef:
        """
        Return a modified copy of a `StreamDef`, updating the given fields.

        Avalanche has used different implementations for stream definitions across
        versions/commits (e.g., `NamedTuple`-like objects with `_replace`, dataclasses,
        or plain mutable objects). This helper provides a version-robust way to update
        fields such as `exps_data` and `is_lazy` without relying on a single concrete
        implementation.

        Update strategy (in order):
          1. If `sd` implements `._replace(...)` (NamedTuple-style), return `sd._replace(**updates)`.
          2. If `sd` is a dataclass instance, return `dataclasses.replace(sd, **updates)`.
          3. If `sd` appears mutable, attempt to set attributes in-place and return `sd`.
          4. As a fallback, reconstruct a new instance by calling the class constructor
             with the intersection of constructor parameters and available attributes.

        Args:
            sd (StreamDef): The stream definition object to update.
            **updates: Field/value pairs to update on the stream definition.

        Returns:
            StreamDef: The updated stream definition. This may be the same object
            (if mutated in-place) or a new object (if copying/reconstruction is used).

        Raises:
            TypeError: If the stream definition can't be rebuilt from its constructor
            and does not support mutation.
        """
        # NamedTuple case
        if hasattr(sd, '_replace'):
            return sd._replace(**updates)

        # Dataclass case
        if dataclasses.is_dataclass(sd):
            return dataclasses.replace(sd, **updates)

        # Mutable object case: try in-place mutation
        ok = True
        for k, v in updates.items():
            if not hasattr(sd, k):
                ok = False
                continue
            try:
                setattr(sd, k, v)
            except Exception:
                ok = False

        if ok:
            return sd

        # Fallback: rebuild by calling the class with whatever args it accepts
        cls = type(sd)
        sig = inspect.signature(cls)
        kwargs = {}
        for name in sig.parameters:
            if name == 'self':
                continue
            if name in updates:
                kwargs[name] = updates[name]
            elif hasattr(sd, name):
                kwargs[name] = getattr(sd, name)
        return cls(**kwargs)

    @staticmethod
    def _make_eager_lds(datasets: list) -> LazyDatasetSequence:
        """
        Build an eager (materialized) `LazyDatasetSequence` from a list of datasets.

        `LazyDatasetSequence` requires an explicit `stream_length` argument.
        Additionally, for non-lazy/eager scenarios Avalanche may call `load_all_experiences()` so that
        datasets are immediately available without deferred loading.

        This helper standardizes that behavior:
          - Creates `LazyDatasetSequence(datasets, len(datasets))`
          - Calls `load_all_experiences()` when available

        Args:
            datasets (list): A list of per-experience datasets (typically AvalancheDataset
                instances produced by `dataset.subset(...)`).

        Returns:
            LazyDatasetSequence: A sequence containing the provided datasets, ready
            for use as `StreamDef.exps_data`.
        """
        seq = LazyDatasetSequence(datasets, len(datasets))
        if hasattr(seq, 'load_all_experiences'):
            seq.load_all_experiences()
        return seq

    def _add_repair_stream_inplace(
        self,
        benchmark: NCScenario,
        repair_budget_per_class: int,
        seed: int = 1,
    ) -> NCScenario:
        """
        Add a persistent "repair" stream to an existing `NCScenario` (in-place).

        Why this is needed:
        - In Avalanche, iterating over a stream constructs *fresh* Experience objects.
          Mutating an Experience instance (e.g., setting `exp.repair_dataset`) does not
          persist across iterations.
        - To persist extra per-experience datasets, they must be stored at the scenario
          level (e.g., as an additional stream) so that each newly-created Experience
          can retrieve them deterministically.

        What this does:
        - For each training experience in `benchmark.train_stream`, splits its dataset
          into two disjoint subsets:
            * a reduced training subset
            * a repair subset
          The split is class-balanced: for each class, it assigns
          `min(repair_budget_per_class, n_class - 1)` samples to repair, clamped to `[1, n-1]`.
        - Replaces the underlying datasets for the existing "train" stream with the
          reduced training subsets.
        - Creates a new "repair" stream with the repair subsets, registers it in
          `benchmark.stream_definitions`, exposes `benchmark.repair_stream`, and adds
          it to `benchmark.streams` / `benchmark._streams` so that the deprecated
          classification scenario machinery can compute timelines without KeyErrors.

        Note:
            This function assumes the benchmark is built on the deprecated
            `dataset_scenario` stack (as returned by `SplitCIFAR100`).
            It uses the matching deprecated `StreamDef`, `LazyDatasetSequence`, and `ClassificationStream`.

        Args:
            benchmark (NCScenario): The scenario to augment.
            repair_budget_per_class (int): Number of examples per class to allocate to the
                repair stream for each experience.
            seed (int): Random seed used to deterministically split per-class indices.

        Returns:
            NCScenario: The same benchmark instance, with:
              - updated train stream datasets
              - a new `repair` stream and `benchmark.repair_stream` attribute

        Raises:
            ValueError: If `repair_budget_per_class` is negative.
        """
        if int(repair_budget_per_class) <= 0:
            return benchmark

        rng = np.random.default_rng(seed)

        new_train_exps = []
        repair_exps = []

        for exp in benchmark.train_stream:
            # Get the dataset from the experience
            if hasattr(exp, 'dataset'):
                dataset = exp.dataset
            elif hasattr(exp, '_dataset'):
                dataset = exp._dataset
            else:
                dataset = None
            if dataset is None:
                continue

            # Extract targets
            targets = dataset.targets if hasattr(dataset, 'targets') else None
            if targets is None:
                # Fallback to iterating dataset to infer targets
                targets = [int(y) for _, y, *rest in dataset]
            targets_arr = np.asarray(targets)

            # Split indices into training and repair subsets
            train_indices: list[int] = []
            repair_indices: list[int] = []
            for class_id in np.unique(targets_arr):
                class_indices = np.where(targets_arr == class_id)[0]
                n = class_indices.size
                n_rep = max(1, min(int(repair_budget_per_class), n - 1))
                permuted = rng.permutation(class_indices)
                repair_indices.extend(permuted[:n_rep].tolist())
                train_indices.extend(permuted[n_rep:].tolist())

            # Create and save training and repair subsets
            train_subset = dataset.subset(train_indices)
            repair_subset = dataset.subset(repair_indices)
            new_train_exps.append(train_subset)
            repair_exps.append(repair_subset)

        train_def = benchmark.stream_definitions['train']

        # Replace train stream datasets
        benchmark.stream_definitions['train'] = self._streamdef_replace(
            train_def,
            exps_data=self._make_eager_lds(new_train_exps),
            is_lazy=False,
        )

        # Add repair stream definition (clone train stream metadata, swap datasets)
        benchmark.stream_definitions['repair'] = self._streamdef_replace(
            train_def,
            exps_data=self._make_eager_lds(repair_exps),
            is_lazy=False,
        )

        # Expose a convenient stream handle
        benchmark.repair_stream = ClassificationStream('repair', benchmark)
        if hasattr(benchmark, '_make_stream_fields'):
            benchmark._make_stream_fields()

        if hasattr(benchmark, 'streams') and isinstance(benchmark.streams, dict):
            benchmark.streams['repair'] = benchmark.repair_stream

        # (some versions also keep a private dict)
        if hasattr(benchmark, '_streams') and isinstance(benchmark._streams, dict):
            benchmark._streams['repair'] = benchmark.repair_stream

        return benchmark


class _SplitCIFAR100ScenarioBuilder(ScenarioBuilder):
    """
    Scenario builder for Split CIFAR-100.
    """

    def __call__(
        self,
        *,
        num_experiences: int = 10,
        return_task_id: bool = False,
        repair_budget_per_class: int = 0,
        dataset_path: str | Path | None = None,
        seed: int = 1,
    ) -> NCScenario:
        """
        Build the standard Split CIFAR-100 class-incremental scenario.

        Args:
            num_experiences (int): Number of experiences to split the 100 classes into.
            return_task_id (bool): Whether Avalanche should emit task IDs with each sample.
            repair_budget_per_class (int): Number of repair samples per class to allocate to repair data.
                                           If greater than 0, each experience will have a repair stream entry.
                                           This dataset will be disjoint from the training dataset.
            dataset_path (str | Path | None): Optional root directory for storing the CIFAR-100 dataset.
            seed (int): Random seed controlling the split and dataloader shuffling.

        Returns:
            NCScenario: Avalanche scenario configured for class-incremental learning.
        """
        self.validate_common_args(
            num_experiences=num_experiences,
            repair_budget_per_class=repair_budget_per_class,
        )

        benchmark: NCScenario = SplitCIFAR100(
            n_experiences=num_experiences,
            return_task_id=return_task_id,
            seed=seed,
            class_ids_from_zero_from_first_exp=True,
            dataset_root=dataset_path,
        )

        return self.maybe_add_repair_stream(
            benchmark=benchmark,
            repair_budget_per_class=repair_budget_per_class,
            seed=seed,
        )


def get_scenario_builder(*, scenario: str) -> ScenarioBuilder:
    """
    Return a scenario builder by name.

    Args:
        scenario (str): Scenario name.

    Returns:
        ScenarioBuilder: Instantiated scenario builder.

    Raises:
        ValueError: If the scenario name is empty or not in the registered builder map.
    """
    builder_path = get_scenario_builder_path(scenario)
    builder_cls = import_symbol(builder_path)
    return builder_cls()


def get_num_classes_from_experience(experience: NCExperience) -> int:
    """
    Gets the number of classes seen until the given experience.

    This relies on common Avalanche experience attributes:
      - `classes_seen_so_far`
      - `previous_classes`
      - `classes_in_this_experience`

    Args:
        experience (NCExperience): Experience to inspect.

    Returns:
        int: The number of classes seen until this experience.
    """
    if not isinstance(experience, NCExperience):
        raise TypeError('Experience must be an `NCExperience`.')

    if hasattr(experience, 'classes_seen_so_far'):
        try:
            classes = list(getattr(experience, 'classes_seen_so_far'))
        except Exception:
            classes = []
    else:
        prev = getattr(experience, 'previous_classes', [])
        cur = getattr(experience, 'classes_in_this_experience', [])
        try:
            classes = list(prev) + list(cur)
        except Exception:
            classes = []

    if not classes:
        raise RuntimeError('Cannot extract classes from experience.')

    unique = []
    seen = set()
    for c in classes:
        try:
            c_int = int(c)
        except Exception:
            continue
        if c_int in seen:
            continue
        seen.add(c_int)
        unique.append(c_int)

    if not unique:
        raise RuntimeError('Cannot extract classes from experience.')

    return len(unique)
