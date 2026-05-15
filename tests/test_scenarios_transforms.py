"""
Tests for scenario transform composition helpers.
"""

from typing import cast

from torchvision.transforms import CenterCrop
from torchvision.transforms import Compose
from torchvision.transforms import RandomCrop
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import Resize

from regain.avalanche_utils.scenarios import _build_imagenet_train_eval_transforms
from regain.avalanche_utils.scenarios import _build_square_dataset_train_eval_transforms
from regain.avalanche_utils.scenarios import _resolve_imagenet_eval_resize_size
from regain.avalanche_utils.scenarios import _resolve_transform_toggle
from regain.avalanche_utils.scenarios import ImageNetRScenarioBuilder


def _transform_type_names(*, transform: Compose) -> list[str]:
    """
    Return transform class names in order.

    Args:
        transform (Compose): Transform pipeline.

    Returns:
        list[str]: Ordered transform class names.
    """
    return [type(op).__name__ for op in transform.transforms]


def _resolve_center_crop_size(*, transform: CenterCrop) -> int:
    """
    Resolve a center-crop side length.

    Args:
        transform (CenterCrop): Center-crop transform.

    Returns:
        int: Square crop side length.
    """
    size = transform.size
    if isinstance(size, tuple):
        return int(size[0])
    return int(size)


class TestSquareDatasetTransforms:
    """
    Tests for square dataset transforms.
    """

    def test_cifar_like_transforms_disable_horizontal_flip_when_toggle_is_false(self) -> None:
        train_transform, _ = _build_square_dataset_train_eval_transforms(
            image_size=32,
            default_image_size=32,
            mean=(0.5071, 0.4865, 0.4409),
            std=(0.2673, 0.2564, 0.2762),
            random_resized_crop=False,
            horizontal_flip=False,
            include_default_random_crop=True,
            default_random_crop_padding=4,
        )

        transform_types = _transform_type_names(transform=train_transform)
        assert transform_types[0] == 'RandomCrop'
        assert 'RandomHorizontalFlip' not in transform_types

    def test_cifar_like_transforms_enable_horizontal_flip_when_toggle_is_true(self) -> None:
        train_transform, _ = _build_square_dataset_train_eval_transforms(
            image_size=32,
            default_image_size=32,
            mean=(0.5071, 0.4865, 0.4409),
            std=(0.2673, 0.2564, 0.2762),
            random_resized_crop=False,
            horizontal_flip=True,
            include_default_random_crop=True,
            default_random_crop_padding=4,
        )

        assert any(isinstance(op, RandomHorizontalFlip) for op in train_transform.transforms)

    def test_random_resized_crop_replaces_default_random_crop(self) -> None:
        train_transform, _ = _build_square_dataset_train_eval_transforms(
            image_size=32,
            default_image_size=32,
            mean=(0.5071, 0.4865, 0.4409),
            std=(0.2673, 0.2564, 0.2762),
            random_resized_crop=True,
            horizontal_flip=False,
            include_default_random_crop=True,
            default_random_crop_padding=4,
        )

        assert isinstance(train_transform.transforms[0], RandomResizedCrop)
        assert not any(isinstance(op, RandomCrop) for op in train_transform.transforms)


class TestImageNetTransforms:
    """
    Tests for ImageNet transforms.
    """

    def test_resolve_imagenet_resize_size_scales_with_crop_size(self) -> None:
        assert _resolve_imagenet_eval_resize_size(image_size=224) == 256
        assert _resolve_imagenet_eval_resize_size(image_size=384) == 439

    def test_imagenet_transforms_resize_before_large_center_crop_without_padding(self) -> None:
        image_size = 384
        expected_resize_size = _resolve_imagenet_eval_resize_size(image_size=image_size)
        train_transform, eval_transform = _build_imagenet_train_eval_transforms(
            image_size=image_size,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            random_resized_crop=False,
            horizontal_flip=False,
        )

        train_resize = cast(Resize, train_transform.transforms[0])
        eval_resize = cast(Resize, eval_transform.transforms[0])
        train_crop = cast(CenterCrop, train_transform.transforms[1])
        eval_crop = cast(CenterCrop, eval_transform.transforms[1])

        assert isinstance(train_resize, Resize)
        assert isinstance(eval_resize, Resize)
        assert int(train_resize.size) == expected_resize_size
        assert int(eval_resize.size) == expected_resize_size
        assert _resolve_center_crop_size(transform=train_crop) == image_size
        assert _resolve_center_crop_size(transform=eval_crop) == image_size

    def test_imagenet_default_builder_applies_image_size_even_with_all_toggles_disabled(self) -> None:
        image_size = 384
        expected_resize_size = _resolve_imagenet_eval_resize_size(image_size=image_size)
        train_transform, eval_transform = ImageNetRScenarioBuilder._build_default_transforms(
            transform_random_resized_crop=False,
            transform_horizontal_flip=False,
            transform_image_size=image_size,
        )

        train_resize = cast(Resize, train_transform.transforms[0])
        eval_resize = cast(Resize, eval_transform.transforms[0])
        train_crop = cast(CenterCrop, train_transform.transforms[1])
        eval_crop = cast(CenterCrop, eval_transform.transforms[1])

        assert isinstance(train_resize, Resize)
        assert isinstance(eval_resize, Resize)
        assert int(train_resize.size) == expected_resize_size
        assert int(eval_resize.size) == expected_resize_size
        assert _resolve_center_crop_size(transform=train_crop) == image_size
        assert _resolve_center_crop_size(transform=eval_crop) == image_size

    def test_imagenet_default_builder_uses_horizontal_flip_when_toggle_is_omitted(self) -> None:
        train_transform, _ = ImageNetRScenarioBuilder._build_default_transforms(
            transform_random_resized_crop=None,
            transform_horizontal_flip=None,
            transform_image_size=224,
        )

        assert any(isinstance(op, RandomHorizontalFlip) for op in train_transform.transforms)


class TestTransformToggleResolution:
    """
    Tests for transform toggle resolution.
    """

    def test_resolve_transform_toggle_uses_default_true_when_toggle_is_none(self) -> None:
        assert _resolve_transform_toggle(toggle=None, default_toggle=True) is True

    def test_resolve_transform_toggle_uses_default_false_when_toggle_is_none(self) -> None:
        assert _resolve_transform_toggle(toggle=None, default_toggle=False) is False
