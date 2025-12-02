"""
Shared helpers for analysis modules.
"""

import csv
import math
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

__all__ = [
    'mean',
    'stdev',
    'to_float',
    'to_int',
    'write_csv',
]


def to_int(value: Any, *, coerce_float: bool = False) -> Optional[int]:
    """
    Convert a value to int when possible.

    Args:
        value (Any): Input value.
        coerce_float (bool): Whether to coerce via float when direct int conversion fails.

    Returns:
        Optional[int]: Integer value or None when conversion fails.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        if not coerce_float:
            return None
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return None


def to_float(
    value: Any,
    *,
    allow_bool: bool = False,
    require_finite: bool = True,
    use_item: bool = True,
) -> Optional[float]:
    """
    Convert a value to float when possible.

    Args:
        value (Any): Input value.
        allow_bool (bool): Whether to allow booleans to be coerced to floats.
        require_finite (bool): Whether to reject non-finite floats (inf/nan).
        use_item (bool): Whether to try `.item()` when present.

    Returns:
        Optional[float]: Float value or None when conversion fails.
    """
    if value is None:
        return None
    if not allow_bool and isinstance(value, bool):
        return None
    if use_item and hasattr(value, 'item'):
        try:
            value = value.item()
        except Exception:
            return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if require_finite and not math.isfinite(v):
        return None
    return v


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    """
    Mean of finite floats, ignoring None.

    Args:
        values (Iterable[Optional[float]]): Iterable of optional floats.

    Returns:
        Optional[float]: Mean or None if no valid values exist.
    """
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return None
    arr = np.asarray(xs, dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return None
    return float(np.mean(arr[finite]))


def stdev(values: Iterable[Optional[float]]) -> Optional[float]:
    """
    Sample standard deviation of finite floats, ignoring None.

    Args:
        values (Iterable[Optional[float]]): Iterable of optional floats.

    Returns:
        Optional[float]: Sample standard deviation or None if fewer than 2 valid values exist.
    """
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return None
    arr = np.asarray(xs, dtype=float)
    finite = np.isfinite(arr)
    arr = arr[finite]
    if arr.size < 2:
        return None
    return float(np.std(arr, ddof=1))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """
    Write a list of dict rows to CSV.

    Args:
        path (Path): Output path.
        rows (list[dict[str, Any]]): Row dicts.

    Returns:
        None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return

    fieldnames: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
