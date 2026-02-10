from abc import ABC
from abc import abstractmethod
import dataclasses
import inspect
from pathlib import Path

####################################################################################################
# SplitCIFAR100 returns an NCScenario built on the deprecated dataset_scenario stack               #
# (StreamDef comes from avalanche.benchmarks.scenarios.deprecated.*).                              #
# We therefore use the matching deprecated LazyDatasetSequence / ClassificationStream types here.  #
####################################################################################################
from avalanche.benchmarks import SplitCIFAR100
from avalanche.benchmarks.scenarios.deprecated.classification_scenario import ClassificationStream
from avalanche.benchmarks.scenarios.deprecated.dataset_scenario import StreamDef
from avalanche.benchmarks.scenarios.deprecated.lazy_dataset_sequence import LazyDatasetSequence
from avalanche.benchmarks.scenarios.deprecated.new_classes.nc_scenario import NCExperience
from avalanche.benchmarks.scenarios.deprecated.new_classes.nc_scenario import NCScenario
####################################################################################################
from avalanche.benchmarks.utils.data_attribute import DataAttribute
import numpy as np

from regain.constants import STREAM_REPAIR
from regain.constants import STREAM_TRAIN
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

    ######################
    # Public entry point #
    ######################

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

        This method orchestrates the scenario creation workflow:
        1. Validates common arguments.
        2. Calls `_build_scenario` (implemented by subclasses).
        3. Injects `original_indices` data attribute in all experience datasets
           with the index of each example w.r.t. the original dataset.
        4. Optionally adds repair stream (`repair_stream`) if `repair_budget_per_class > 0`.
        5. Validates benchmark split integrity:
           - class IDs contiguous in `[0, n_classes-1]`
           - experience datasets expose `original_indices`
           - experiences are disjoint within each stream
           - streams that share the same origin are disjoint
           - per-origin-group union covers exactly the full origin dataset length

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

        Raises:
            ValueError: If `num_experiences` is not a positive integer or if `repair_budget_per_class` is negative.
            RuntimeError: If scenario validations fail
        """
        # Validate arguments
        if not isinstance(num_experiences, int) or num_experiences <= 0:
            raise ValueError('`num_experiences` must be a positive integer.')
        if int(repair_budget_per_class) < 0:
            raise ValueError('`repair_budget_per_class` must be non-negative.')

        # Let subclass build the base scenario
        benchmark = self._build_scenario(
            num_experiences=num_experiences,
            return_task_id=return_task_id,
            dataset_path=dataset_path,
            seed=seed,
        )

        # Inject global example indices (`original_indices`)
        self._inject_original_indices(benchmark)

        # Add repair stream (if requested)
        if repair_budget_per_class > 0:
            self._add_repair_stream(benchmark, repair_budget_per_class=repair_budget_per_class, seed=seed)

        # Validate scenario integrity
        self._validate_scenario(benchmark)

        return benchmark

    ############################
    # Subclass hook / contract #
    ############################

    @abstractmethod
    def _build_scenario(
        self,
        *,
        num_experiences: int,
        return_task_id: bool,
        dataset_path: str | Path | None = None,
        seed: int = 1,
    ) -> NCScenario:
        """
        Build the base scenario.

        Subclasses must implement this method to create their specific scenario.
        The base class will automatically inject `original_indices` and `repair_stream` after this method returns.

        Args:
            num_experiences (int): Number of experiences to split the classes into.
            return_task_id (bool): Whether Avalanche should emit task IDs with each sample.
            dataset_path (str | Path | None): Optional root directory for storing the dataset.
            seed (int): Random seed controlling the split and dataloader shuffling.

        Returns:
            NCScenario: A new Avalanche `NCScenario` (without `original_indices` and `repair_stream` yet).
        """
        raise NotImplementedError

    #################################
    # Core steps used by `__call__` #
    #################################

    @staticmethod
    def _inject_original_indices(benchmark: NCScenario):
        """
        Inject an `original_indices` data attribute into all stream datasets (in-place).

        This ensures that each experience dataset tracks the global indices of its samples
        w.r.t. the original dataset. The indices are automatically preserved when subsetting.

        Args:
            benchmark (NCScenario): The scenario to augment.

        Raises:
            RuntimeError: If the function cannot extract global indices from the dataset's internal structure.
        """
        for stream_name in dir(benchmark):
            if not stream_name.endswith('_stream'):
                continue
            stream = getattr(benchmark, stream_name, None)
            if stream is None:
                continue

            # Access stream definition if available
            stream_def = None
            if hasattr(benchmark, 'stream_definitions') and isinstance(
                    benchmark.stream_definitions, dict
            ):
                stream_key = stream_name.replace('_stream', '')
                stream_def = benchmark.stream_definitions.get(stream_key)

            if stream_def is None or not hasattr(stream_def, 'exps_data'):
                continue

            exps_data = stream_def.exps_data
            if not hasattr(exps_data, '__iter__'):
                continue

            # Inject original_indices into each experience dataset
            updated_datasets = []
            for dataset in exps_data:
                if dataset is None:
                    updated_datasets.append(dataset)
                    continue

                # Only inject if not already present
                if hasattr(dataset, 'original_indices'):
                    updated_datasets.append(dataset)
                    continue

                # Extract global indices from the dataset's internal structure
                # Avalanche tracks indices in the targets FlatData when subsetting
                global_indices = None
                if (
                        hasattr(dataset, 'targets')
                        and hasattr(dataset.targets, 'data')
                        and hasattr(dataset.targets.data, '_indices')
                ):
                    # FlatData._indices contains the global indices from the original dataset
                    global_indices = list(dataset.targets.data._indices)

                if global_indices is None:
                    raise RuntimeError(
                        'Cannot extract global indices from dataset. '
                        'Expected `dataset.targets.data._indices` to be available.'
                    )

                # Create original_indices attribute with global indices
                original_indices = DataAttribute(
                    global_indices,
                    name='original_indices',
                    use_in_getitem=False,
                )
                dataset_with_indices = dataset.update_data_attribute(
                    'original_indices', original_indices
                )
                updated_datasets.append(dataset_with_indices)

            # Update the stream definition with datasets that have original_indices
            if updated_datasets:
                benchmark.stream_definitions[stream_key] = (
                    ScenarioBuilder._streamdef_replace(
                        stream_def,
                        exps_data=ScenarioBuilder._make_eager_lds(updated_datasets),
                    )
                )

    def _add_repair_stream(
        self,
        benchmark: NCScenario,
        repair_budget_per_class: int,
        seed: int = 1,
    ):
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

        train_def = benchmark.stream_definitions[STREAM_TRAIN]

        # Replace train stream datasets
        benchmark.stream_definitions[STREAM_TRAIN] = self._streamdef_replace(
            train_def,
            exps_data=self._make_eager_lds(new_train_exps),
            is_lazy=False,
        )

        # Add repair stream definition (clone train stream metadata, swap datasets)
        benchmark.stream_definitions[STREAM_REPAIR] = self._streamdef_replace(
            train_def,
            exps_data=self._make_eager_lds(repair_exps),
            is_lazy=False,
        )

        # Expose a convenient stream handle
        benchmark.repair_stream = ClassificationStream(STREAM_REPAIR, benchmark)
        if hasattr(benchmark, '_make_stream_fields'):
            benchmark._make_stream_fields()

        if hasattr(benchmark, 'streams') and isinstance(benchmark.streams, dict):
            benchmark.streams[STREAM_REPAIR] = benchmark.repair_stream

        # (some versions also keep a private dict)
        if hasattr(benchmark, '_streams') and isinstance(benchmark._streams, dict):
            benchmark._streams[STREAM_REPAIR] = benchmark.repair_stream

    def _validate_scenario(self, benchmark: NCScenario) -> None:
        """
        Validate scenario-level invariants.

        Validations:
          - Verify class IDs are contiguous in `[0, n_classes-1]` for the training stream.
          - Verify `original_indices` are present and extractable for every experience dataset.
          - Verify experiences within each stream are disjoint by `original_indices`.
          - Verify streams that share the same origin are disjoint and jointly cover the full origin dataset.

        Raises:
            RuntimeError / ValueError on validation failure.
        """
        if not isinstance(benchmark, NCScenario):
            raise TypeError('Benchmark must be an `NCScenario`.')

        self._verify_classes_contiguous_from_zero(benchmark)
        self._verify_benchmark_index_partitioning(benchmark)

    ######################
    # Validation helpers #
    ######################

    @staticmethod
    def _verify_classes_contiguous_from_zero(benchmark: NCScenario) -> None:
        """
        Verify that all class IDs in the training stream are contiguous starting from zero.

        This matches the project’s expectation when scenarios are built with
        `class_ids_from_zero_from_first_exp=True`.
        """
        all_class_ids: list[int] = []
        for exp in benchmark.train_stream:
            all_class_ids.extend(list(getattr(exp, 'classes_in_this_experience', [])))

        expected = set(range(int(benchmark.n_classes)))
        seen = set(int(x) for x in all_class_ids)

        if seen != expected:
            missing = sorted(expected - seen)
            extra = sorted(seen - expected)
            raise ValueError(
                'Class IDs in the benchmark are not contiguous starting from zero: '
                f'missing={missing[:20]}{"..." if len(missing) > 20 else ""} '
                f'extra={extra[:20]}{"..." if len(extra) > 20 else ""}'
            )

        if len(seen) != int(benchmark.n_classes):
            raise ValueError(
                f'Expected {benchmark.n_classes} unique class IDs, got {len(seen)}.'
            )

    @staticmethod
    def _get_stream_names(benchmark: NCScenario) -> list[str]:
        """
        Enumerate stream names present on the benchmark.

        Prefers `benchmark.stream_definitions` when available (deprecated dataset_scenario stack),
        otherwise falls back to attributes ending with `_stream`.
        """
        if hasattr(benchmark, 'stream_definitions') and isinstance(benchmark.stream_definitions, dict):
            names = [str(k) for k in benchmark.stream_definitions.keys()]
            return sorted(set(names))

        names: list[str] = []
        for attr in dir(benchmark):
            if attr.endswith('_stream'):
                names.append(attr[:-7])
        return sorted(set(names))

    @staticmethod
    def _verify_benchmark_index_partitioning(benchmark: NCScenario) -> None:
        """
        Verify that:
          1) Within each stream, experiences are disjoint by `original_indices`.
          2) Across streams that share the same origin dataset, stream unions are disjoint and
             their combined union covers exactly the entire origin dataset.

        This is particularly important when a repair stream is enabled: train + repair must form a
        partition of the underlying training dataset.
        """
        stream_names = ScenarioBuilder._get_stream_names(benchmark)

        stream_unions: dict[str, set[int]] = {}
        stream_origin_keys: dict[str, tuple[str, int | None, int | None]] = {}

        # 1) Per-stream: experiences disjoint
        for stream_name in stream_names:
            stream = getattr(benchmark, f'{stream_name}_stream', None)
            if stream is None:
                continue
            union, origin_key = ScenarioBuilder._verify_stream_experience_disjointness(
                benchmark=benchmark,
                stream_name=stream_name,
            )
            stream_unions[stream_name] = union
            stream_origin_keys[stream_name] = origin_key

        # 2) Group streams by origin and verify partitioning per origin group
        groups: dict[tuple[str, int | None, int | None], dict[str, set[int]]] = {}
        for stream_name, origin_key in stream_origin_keys.items():
            groups.setdefault(origin_key, {})[stream_name] = stream_unions.get(stream_name, set())

        for origin_key, stream_to_indices in groups.items():
            # If we can’t infer identity, but multiple streams ended up in the same group_key anyway,
            # the disjoint+coverage check is still meaningful. If origin_len is missing, we already raise.
            ScenarioBuilder._verify_origin_group_partitioning(
                group_key=origin_key,
                stream_to_indices=stream_to_indices,
            )

    @staticmethod
    def _verify_stream_experience_disjointness(
            *,
            benchmark: NCScenario,
            stream_name: str,
    ) -> tuple[set[int], tuple[str, int | None, int | None]]:
        """
        Verify that experiences within a given stream are disjoint by `original_indices`,
        and that each experience dataset contains no duplicate indices internally.

        Returns:
            (stream_union_indices, origin_key)
        """
        stream = getattr(benchmark, f'{stream_name}_stream', None)
        if stream is None:
            return set(), ('unknown', None, None)

        union: set[int] = set()
        origin_key: tuple[str, int | None, int | None] | None = None

        for experience in stream:
            exp_idx = int(getattr(experience, 'current_experience', 0))
            dataset = getattr(experience, 'dataset', None)
            if dataset is None:
                raise RuntimeError(
                    'Benchmark split verification failed: missing dataset on experience: '
                    f'stream={stream_name} exp={exp_idx}.'
                )

            this_origin_key = ScenarioBuilder._origin_key_for_dataset(dataset)
            if origin_key is None:
                origin_key = this_origin_key
            else:
                # Enforce same origin identity when available
                if (
                        origin_key[1] is not None
                        and this_origin_key[1] is not None
                        and origin_key[1] != this_origin_key[1]
                ):
                    raise RuntimeError(
                        'Benchmark split verification failed: stream experiences do not share the same origin dataset: '
                        f'stream={stream_name} exp={exp_idx} origin_key={this_origin_key} expected_origin_key={origin_key}.'
                    )
                # Enforce consistent origin length when available
                if (
                        origin_key[2] is not None
                        and this_origin_key[2] is not None
                        and origin_key[2] != this_origin_key[2]
                ):
                    raise RuntimeError(
                        'Benchmark split verification failed: stream experiences disagree on origin dataset length: '
                        f'stream={stream_name} exp={exp_idx} origin_len={this_origin_key[2]} expected_origin_len={origin_key[2]}.'
                    )

            indices = ScenarioBuilder._extract_original_indices(dataset)
            if not indices:
                continue

            exp_set = set(indices)

            # No duplicates inside a single experience dataset
            if len(exp_set) != len(indices):
                raise ValueError(
                    'Benchmark split verification failed: duplicate `original_indices` found within an experience dataset: '
                    f'stream={stream_name} exp={exp_idx} size={len(indices)} unique={len(exp_set)}.'
                )

            # No overlaps across experiences within the stream
            overlap = union.intersection(exp_set)
            if overlap:
                overlap_sorted = sorted(overlap)
                raise ValueError(
                    'Benchmark split verification failed: overlapping example indices between experiences in the same stream: '
                    f'stream={stream_name} exp={exp_idx} overlap_count={len(overlap_sorted)} '
                    f'overlap_examples={overlap_sorted[:20]}{"..." if len(overlap_sorted) > 20 else ""}.'
                )

            union.update(exp_set)

        return union, (origin_key if origin_key is not None else ('unknown', None, None))

    @staticmethod
    def _verify_origin_group_partitioning(
            *,
            group_key: tuple[str, int | None, int | None],
            stream_to_indices: dict[str, set[int]],
    ) -> None:
        """
        Verify that streams in the same origin group are disjoint and cover exactly the origin dataset.

        Enforces:
          - pairwise disjoint stream unions within the group
          - combined union equals {0, 1, ..., origin_len-1}
        """
        origin_type, origin_id, origin_len = group_key
        if origin_len is None:
            raise RuntimeError(
                'Benchmark split verification failed: cannot infer original dataset length for an origin group: '
                f'origin_key={group_key} streams={sorted(stream_to_indices)}. '
                'Ensure datasets expose `len(origin_dataset)` through Avalanche internal structures.'
            )

        combined: set[int] = set()
        for stream_name, idxs in stream_to_indices.items():
            overlap = combined.intersection(idxs)
            if overlap:
                overlap_sorted = sorted(overlap)
                raise ValueError(
                    'Benchmark split verification failed: overlapping example indices across streams that share the same origin dataset: '
                    f'origin_key={group_key} streams={sorted(stream_to_indices)} overlap_count={len(overlap_sorted)} '
                    f'overlap_examples={overlap_sorted[:20]}{"..." if len(overlap_sorted) > 20 else ""}.'
                )
            combined.update(idxs)

        expected = set(range(int(origin_len)))
        if combined != expected:
            missing = sorted(expected - combined)
            extra = sorted(combined - expected)
            raise ValueError(
                'Benchmark split verification failed: streams do not sum to the full original dataset length for their origin group: '
                f'origin_key={group_key} streams={sorted(stream_to_indices)} '
                f'expected_len={int(origin_len)} union_len={len(combined)} '
                f'missing={missing[:20]}{"..." if len(missing) > 20 else ""} '
                f'extra={extra[:20]}{"..." if len(extra) > 20 else ""}.'
            )

    ############################
    # Dataset/origin utilities #
    ############################

    @staticmethod
    def _extract_original_indices(experience_dataset: object) -> list[int]:
        """
        Extract the index of each example in the given experience dataset w.r.t. the origin dataset.

        Raises:
            AttributeError / RuntimeError if not extractable.
        """
        if not hasattr(experience_dataset, 'original_indices'):
            raise AttributeError(
                'Experience dataset is missing `original_indices` data attribute.'
            )
        try:
            indices = list(getattr(experience_dataset, 'original_indices'))
            return [int(idx) for idx in indices]
        except Exception as exc:
            raise RuntimeError(
                f'Failed to extract indices from `original_indices`: {exc}'
            ) from exc

    @staticmethod
    def _unwrap_to_base_dataset(ds: object, *, max_depth: int = 64) -> object:
        """
        Best-effort unwrapping of Avalanche / PyTorch dataset wrappers to reach the underlying base dataset.

        Handles common wrapper patterns:
          - AvalancheDataset: has `_datasets` (list of child datasets)
          - torch.utils.data.Subset: has `.dataset`
          - some Avalanche wrappers: `_dataset`
        """
        cur = ds
        seen: set[int] = set()

        for _ in range(max_depth):
            if cur is None:
                break
            cur_id = id(cur)
            if cur_id in seen:
                break
            seen.add(cur_id)

            nxt = None

            # AvalancheDataset-like: underlying datasets stored in `_datasets`
            for attr in ("_datasets", "datasets"):
                if hasattr(cur, attr):
                    try:
                        dlist = getattr(cur, attr)
                    except Exception:
                        dlist = None
                    if isinstance(dlist, (list, tuple)) and len(dlist) == 1:
                        nxt = dlist[0]
                        break

            # PyTorch Subset / other wrappers: `.dataset`
            if nxt is None and hasattr(cur, "dataset"):
                try:
                    nxt = getattr(cur, "dataset")
                except Exception:
                    nxt = None

            # Some Avalanche wrappers: `_dataset`
            if nxt is None and hasattr(cur, "_dataset"):
                try:
                    nxt = getattr(cur, "_dataset")
                except Exception:
                    nxt = None

            if nxt is None or nxt is cur:
                break

            cur = nxt

        return cur

    @staticmethod
    def _infer_origin_container_and_len(experience_dataset: object) -> tuple[object | None, int | None]:
        """
        Infer the true base/origin dataset and its length.

        Prefer unwrapping the dataset chain to the underlying base dataset.
        Fallback to `max(original_indices)+1` only if we can't get a length.
        """
        base = ScenarioBuilder._unwrap_to_base_dataset(experience_dataset)

        origin_len: int | None
        try:
            origin_len = len(base)  # base should be CIFAR100 train/test, etc.
        except Exception:
            origin_len = None

        # Fallback: infer from original_indices if present
        if origin_len is None and hasattr(experience_dataset, "original_indices"):
            try:
                idxs = ScenarioBuilder._extract_original_indices(experience_dataset)
                if idxs:
                    origin_len = int(max(idxs)) + 1
            except Exception:
                pass

        return base, origin_len

    @staticmethod
    def _origin_signature(origin_obj: object, origin_len: int | None) -> tuple:
        """
        Build a comparable, process-stable signature for the origin dataset.
        """
        if origin_obj is None:
            return ("unknown", origin_len)

        parts: list[object] = [
            type(origin_obj).__module__,
            type(origin_obj).__qualname__,
            origin_len,
        ]

        # Common dataset descriptors (torchvision-style)
        for attr in ("root", "train", "split"):
            if hasattr(origin_obj, attr):
                try:
                    parts.append((attr, str(getattr(origin_obj, attr))))
                except Exception:
                    pass

        return tuple(parts)

    @staticmethod
    def _origin_key_for_dataset(experience_dataset: object) -> tuple[str, object | None, int | None]:
        """
        Grouping key for datasets that share the same origin.
        """
        origin_obj, origin_len = ScenarioBuilder._infer_origin_container_and_len(experience_dataset)
        origin_type = type(origin_obj).__name__ if origin_obj is not None else "unknown"
        origin_sig = ScenarioBuilder._origin_signature(origin_obj, origin_len)
        return (origin_type, origin_sig, origin_len)

    ###################################
    # Avalanche compatibility helpers #
    ###################################

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


class SplitCIFAR100ScenarioBuilder(ScenarioBuilder):
    """
    Scenario builder for Split CIFAR-100.
    """

    def _build_scenario(
        self,
        *,
        num_experiences: int,
        return_task_id: bool,
        dataset_path: str | Path | None = None,
        seed: int = 1,
    ) -> NCScenario:
        """
        Build the standard Split CIFAR-100 class-incremental scenario.

        Args:
            num_experiences (int): Number of experiences to split the 100 classes into.
            return_task_id (bool): Whether Avalanche should emit task IDs with each sample.
            dataset_path (str | Path | None): Optional root directory for storing the CIFAR-100 dataset.
            seed (int): Random seed controlling the split and dataloader shuffling.

        Returns:
            NCScenario: Avalanche scenario configured for class-incremental learning.
        """
        return SplitCIFAR100(
            n_experiences=num_experiences,
            return_task_id=return_task_id,
            seed=seed,
            class_ids_from_zero_from_first_exp=True,
            dataset_root=dataset_path,
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
