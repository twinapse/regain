"""
Tests for ImageNet-R scenario helper logic.
"""

from regain.avalanche_utils.scenarios import _ImageNetRSubsetRawDataset
from regain.avalanche_utils.scenarios import ImageNetRScenarioBuilder
from regain.avalanche_utils.scenarios import ScenarioBuilder


class _DummyImageFolderLikeDataset:
    """
    Minimal ImageFolder-like dataset used for unit tests.
    """

    def __init__(self) -> None:
        self.root = 'dummy_root'
        self.classes = ['class_0', 'class_1']
        self.class_to_idx = {'class_0': 0, 'class_1': 1}
        self.targets = [0, 0, 1, 1]
        self.samples = [
            ('img_0.jpg', 0),
            ('img_1.jpg', 0),
            ('img_2.jpg', 1),
            ('img_3.jpg', 1),
        ]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[str, int]:
        target = int(self.targets[int(index)])
        return f'image_{int(index)}', target


############################
# Holdout split generation #
############################


class TestImageNetRHoldoutIndices:
    """
    Tests for ImageNet-R holdout index computation.
    """

    def test_make_per_class_holdout_indices_is_disjoint_and_covering(self) -> None:
        targets = [0] * 10 + [1] * 10 + [2] * 10

        train_indices, test_indices = ImageNetRScenarioBuilder._make_per_class_holdout_indices(
            targets=targets,
            test_fraction=0.2,
            seed=7,
        )

        assert set(train_indices).isdisjoint(set(test_indices))
        assert set(train_indices).union(set(test_indices)) == set(range(len(targets)))

        for class_id in [0, 1, 2]:
            class_indices = {idx for idx, target in enumerate(targets) if target == class_id}
            assert class_indices.intersection(train_indices)
            assert class_indices.intersection(test_indices)


##############################
# Origin-key regression test #
##############################


class TestImageNetROriginKeys:
    """
    Tests for ImageNet-R origin key derivation.
    """

    def test_split_wrappers_have_distinct_origin_keys(self) -> None:
        base_dataset = _DummyImageFolderLikeDataset()
        train_dataset = _ImageNetRSubsetRawDataset(
            base_dataset=base_dataset,
            indices=[0, 2],
            split='train',
        )
        test_dataset = _ImageNetRSubsetRawDataset(
            base_dataset=base_dataset,
            indices=[1, 3],
            split='test',
        )

        train_origin_key = ScenarioBuilder._origin_key_for_dataset(train_dataset)
        test_origin_key = ScenarioBuilder._origin_key_for_dataset(test_dataset)

        assert train_origin_key != test_origin_key
