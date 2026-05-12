"""
Cross-validation fold construction for the router analysis.
"""

from dataclasses import dataclass
from typing import Any, Optional

from regain.analysis.router.constants import HELD_DATASET
from regain.analysis.router.constants import HELD_SEED
from regain.analysis.router.constants import HELD_SETTING

__all__ = [
    'RouterFold',
]


@dataclass(frozen=True)
class RouterFold:
    """
    Container describing one validation fold.

    Attributes:
        validation_level: The validation strategy name (e.g., `held_seed`).
        fold_id: Stable fold identifier.
        train_indices: Row indices used for fitting.
        test_indices: Row indices held out for evaluation.
        heldout_group: Human-readable description of the held-out partition.
    """

    validation_level: str
    fold_id: str
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    heldout_group: str


def _stringify_group(value: Any) -> str:
    """
    Build a stable string representation for fold group keys.

    Args:
        value: Group key value.

    Returns:
        str: Stringified value, empty string when None.
    """
    if value is None:
        return ''
    return str(value)


def _build_held_seed_folds(
    *,
    feature_rows: list[dict[str, Any]],
    manifest_warnings: list[dict[str, Any]],
) -> tuple[list[RouterFold], list[dict[str, Any]]]:
    """
    Construct held-seed folds.

    Args:
        feature_rows: Feature-table rows.
        manifest_warnings: Mutable manifest warning list.

    Returns:
        tuple[list[RouterFold], list[dict[str, Any]]]: Folds and skipped-group metadata.
    """
    family_groups: dict[tuple[Any, ...], list[int]] = {}
    for index, row in enumerate(feature_rows):
        key = (
            row.get('scenario'),
            row.get('backbone_name'),
            row.get('strategy_name'),
            row.get('replay_mem_size'),
        )
        family_groups.setdefault(key, []).append(index)

    folds: list[RouterFold] = []
    skipped: list[dict[str, Any]] = []
    for family_key in sorted(family_groups.keys(), key=lambda key: tuple(_stringify_group(value) for value in key)):
        indices = family_groups[family_key]
        seeds_to_indices: dict[Any, list[int]] = {}
        for index in indices:
            seeds_to_indices.setdefault(feature_rows[index].get('seed'), []).append(index)
        if len(seeds_to_indices) < 2:
            skipped.append({
                'validation_level': HELD_SEED,
                'family_key': list(family_key),
                'reason': 'insufficient_seeds',
            })
            manifest_warnings.append({
                'code': 'held_seed_insufficient_seeds',
                'message': 'Held-seed validation skipped because fewer than two seeds were present.',
                'context': {
                    'scenario': family_key[0],
                    'backbone_name': family_key[1],
                    'strategy_name': family_key[2],
                    'replay_mem_size': family_key[3],
                },
            })
            continue
        for seed in sorted(seeds_to_indices.keys(), key=_stringify_group):
            test_indices = tuple(sorted(seeds_to_indices[seed]))
            train_indices = tuple(sorted(index for index in indices if index not in seeds_to_indices[seed]))
            if not train_indices or not test_indices:
                continue
            fold_id = f'family={_stringify_group(family_key)}|seed={_stringify_group(seed)}'
            folds.append(
                RouterFold(
                    validation_level=HELD_SEED,
                    fold_id=fold_id,
                    train_indices=train_indices,
                    test_indices=test_indices,
                    heldout_group=f'seed={_stringify_group(seed)}',
                ))
    return folds, skipped


def _build_held_setting_folds(
    *,
    feature_rows: list[dict[str, Any]],
    manifest_warnings: list[dict[str, Any]],
) -> tuple[list[RouterFold], Optional[str]]:
    """
    Construct held-setting folds.

    Args:
        feature_rows: Feature-table rows.
        manifest_warnings: Mutable manifest warning list.

    Returns:
        tuple[list[RouterFold], Optional[str]]: Folds and an optional skip reason.
    """
    grouped: dict[tuple[Any, ...], list[int]] = {}
    for index, row in enumerate(feature_rows):
        key = (
            row.get('replay_mem_size'),
            row.get('repair_budget_fraction'),
            row.get('repair_budget_total'),
        )
        grouped.setdefault(key, []).append(index)
    if len(grouped) < 2:
        manifest_warnings.append({
            'code': 'held_setting_insufficient_groups',
            'message': 'Held-setting validation skipped because fewer than two unique settings were present.',
            'context': {},
        })
        return [], 'insufficient_groups'
    folds: list[RouterFold] = []
    for key in sorted(grouped.keys(), key=lambda value: tuple(_stringify_group(item) for item in value)):
        test_indices = tuple(sorted(grouped[key]))
        train_indices = tuple(
            sorted(index for other_key, indices in grouped.items() if other_key != key for index in indices))
        if not train_indices or not test_indices:
            continue
        fold_id = (f'mem={_stringify_group(key[0])}|frac={_stringify_group(key[1])}'
                   f'|total={_stringify_group(key[2])}')
        folds.append(
            RouterFold(
                validation_level=HELD_SETTING,
                fold_id=fold_id,
                train_indices=train_indices,
                test_indices=test_indices,
                heldout_group=fold_id,
            ))
    return folds, None


def _build_held_dataset_folds(
    *,
    feature_rows: list[dict[str, Any]],
    manifest_warnings: list[dict[str, Any]],
) -> tuple[list[RouterFold], Optional[str]]:
    """
    Construct held-dataset folds.

    Args:
        feature_rows: Feature-table rows.
        manifest_warnings: Mutable manifest warning list.

    Returns:
        tuple[list[RouterFold], Optional[str]]: Folds and an optional skip reason.
    """
    grouped: dict[Any, list[int]] = {}
    for index, row in enumerate(feature_rows):
        grouped.setdefault(row.get('scenario'), []).append(index)
    if len(grouped) < 2:
        manifest_warnings.append({
            'code': 'held_dataset_insufficient_groups',
            'message': 'Held-dataset validation skipped because fewer than two scenarios were present.',
            'context': {},
        })
        return [], 'insufficient_groups'
    folds: list[RouterFold] = []
    for scenario in sorted(grouped.keys(), key=_stringify_group):
        test_indices = tuple(sorted(grouped[scenario]))
        train_indices = tuple(
            sorted(index for other_scenario, indices in grouped.items() if other_scenario != scenario
                   for index in indices))
        if not train_indices or not test_indices:
            continue
        folds.append(
            RouterFold(
                validation_level=HELD_DATASET,
                fold_id=f'scenario={_stringify_group(scenario)}',
                train_indices=train_indices,
                test_indices=test_indices,
                heldout_group=f'scenario={_stringify_group(scenario)}',
            ))
    return folds, None


def all_folds(
    *,
    feature_rows: list[dict[str, Any]],
    manifest_warnings: list[dict[str, Any]],
) -> tuple[list[RouterFold], dict[str, Any]]:
    """
    Build the full fold collection across all validation levels.

    Args:
        feature_rows: Feature-table rows.
        manifest_warnings: Mutable manifest warnings.

    Returns:
        tuple[list[RouterFold], dict[str, Any]]: Folds and metadata to record in the manifest.
    """
    held_seed_folds, held_seed_skipped = _build_held_seed_folds(
        feature_rows=feature_rows,
        manifest_warnings=manifest_warnings,
    )
    held_setting_folds, held_setting_skip = _build_held_setting_folds(
        feature_rows=feature_rows,
        manifest_warnings=manifest_warnings,
    )
    held_dataset_folds, held_dataset_skip = _build_held_dataset_folds(
        feature_rows=feature_rows,
        manifest_warnings=manifest_warnings,
    )
    folds = [*held_seed_folds, *held_setting_folds, *held_dataset_folds]
    metadata = {
        HELD_SEED: {
            'num_folds': len(held_seed_folds),
            'skipped_groups': held_seed_skipped,
        },
        HELD_SETTING: {
            'num_folds': len(held_setting_folds),
            'skipped_reason': held_setting_skip,
        },
        HELD_DATASET: {
            'num_folds': len(held_dataset_folds),
            'skipped_reason': held_dataset_skip,
        },
    }
    return folds, metadata
