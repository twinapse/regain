"""
Plotting helpers for repairability-frontier analysis outputs.
"""

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Optional

from regain.analysis.utils import mean
from regain.analysis.utils import to_float
from regain.constants import COLUMN_B
from regain.constants import COLUMN_CONTROLLER_NAME
from regain.constants import COLUMN_REPAIR_BUDGET_FRACTION
from regain.constants import COLUMN_REPAIR_BUDGET_TOTAL

_COLUMN_ACTION_REPAIR_BUDGET_FRACTION = 'action_repair_budget_fraction'
_COLUMN_ACTION_REPAIR_BUDGET_TOTAL = 'action_repair_budget_total'
_COLUMN_BACKBONE_NAME = 'backbone_name'
_COLUMN_CONTROLLER_ID = 'controller_id'
_COLUMN_MEAN_ABSOLUTE_RECOVERY = 'mean_absolute_recovery'
_COLUMN_MEAN_HARMED_TASK_FRACTION = 'mean_harmed_task_fraction'
_COLUMN_SCENARIO = 'scenario'
_COLUMN_STRATEGY_NAME = 'strategy_name'
_COLUMN_UTILITY_CONSERVATIVE = 'utility_conservative'
_PLOT_FILENAME_HARM_VS_RECOVERY = 'harm_vs_recovery.png'
_PLOT_FILENAME_UTILITY_VS_COST = 'utility_vs_cost.png'


@dataclass(frozen=True)
class PlotAnalysisResult:
    """
    Result metadata from repairability plot generation.

    Attributes:
        saved_paths (list[Path]): Paths of plot files written in this invocation.
        saved_filenames (list[str]): File names of saved plots.
        skipped (list[dict[str, Any]]): Structured skipped-plot metadata.
    """

    saved_paths: list[Path]
    saved_filenames: list[str]
    skipped: list[dict[str, Any]]


def _action_cost_x_value(row: dict[str, Any]) -> float | None:
    """
    Read the action-cost value for utility-vs-cost plotting.

    Args:
        row: Frontier row.

    Returns:
        float | None: Action cost value.
    """
    if row.get(_COLUMN_ACTION_REPAIR_BUDGET_FRACTION) is not None:
        return to_float(row.get(_COLUMN_ACTION_REPAIR_BUDGET_FRACTION))
    if row.get(_COLUMN_ACTION_REPAIR_BUDGET_TOTAL) is not None:
        return to_float(row.get(_COLUMN_ACTION_REPAIR_BUDGET_TOTAL))
    return None


def _action_cost_axis_label(rows: list[dict[str, Any]]) -> str:
    """
    Choose a utility-vs-cost axis label based on available action-cost columns.

    Args:
        rows: Frontier rows.

    Returns:
        str: X-axis label.
    """
    if any(row.get(_COLUMN_ACTION_REPAIR_BUDGET_FRACTION) is not None for row in rows):
        return _COLUMN_ACTION_REPAIR_BUDGET_FRACTION
    if any(row.get(_COLUMN_ACTION_REPAIR_BUDGET_TOTAL) is not None for row in rows):
        return _COLUMN_ACTION_REPAIR_BUDGET_TOTAL
    return 'action_repair_budget'


def _sorted_unique(values: Iterable[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[Any] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open('r', newline='', encoding='utf-8') as f:
        return [dict(row) for row in csv.DictReader(f)]


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f'Frontier manifest not found: {path}')
    with path.open('r', encoding='utf-8') as f:
        payload = json.load(f)
    payload.setdefault('plots', {})
    payload['plots'].setdefault('saved', [])
    payload['plots'].setdefault('skipped', [])
    return payload


def _write_manifest(*, path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def write_plot_manifest_update(
    *,
    source_manifest_path: Path,
    destination_manifest_path: Path,
    saved_filenames: list[str],
    skipped: list[dict[str, Any]],
) -> Path:
    """
    Write a manifest copy with updated plot metadata.

    Args:
        source_manifest_path (Path): Existing manifest to read.
        destination_manifest_path (Path): Path where the updated manifest copy should be written.
        saved_filenames (list[str]): Saved plot file names.
        skipped (list[dict[str, Any]]): Skipped plot metadata.

    Returns:
        Path: Destination manifest path.
    """
    manifest = _load_manifest(source_manifest_path)
    manifest.setdefault('plots', {})
    manifest['plots']['saved'] = list(saved_filenames)
    manifest['plots']['skipped'] = list(skipped)
    _write_manifest(path=destination_manifest_path, payload=manifest)
    return destination_manifest_path


def _slugify(value: Any) -> str:
    text = ''.join(
        character.lower() if character.isalnum() else '_'
        for character in str(value or '')
    )
    while '__' in text:
        text = text.replace('__', '_')
    return text.strip('_') or 'unknown'


def _record_plot_skip(
    *,
    skipped: list[dict[str, Any]],
    filename: str,
    reason: str,
    context: dict[str, Any] | None = None,
) -> None:
    skipped.append({
        'filename': filename,
        'reason': reason,
        'context': context or {},
    })


def _grouped_mean_rows(
    *,
    rows: list[dict[str, Any]],
    group_keys: tuple[str, ...],
    value_key: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float | None]] = {}
    exemplar_rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(group_key) for group_key in group_keys)
        grouped.setdefault(key, []).append(to_float(row.get(value_key)))
        exemplar_rows.setdefault(key, row)

    aggregated_rows: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: tuple('' if value is None else str(value) for value in item)):
        aggregated_row = dict(exemplar_rows[key])
        aggregated_row[value_key] = mean(grouped[key])
        aggregated_rows.append(aggregated_row)
    return aggregated_rows


def _choose_utility_delta_pair(*, frontier_rows: list[dict[str, Any]]) -> tuple[str, str] | None:
    controller_ids = sorted(
        _sorted_unique(
            row.get(_COLUMN_CONTROLLER_ID)
            for row in frontier_rows
            if row.get(_COLUMN_CONTROLLER_ID)
        )
    )
    if len(controller_ids) < 2:
        return None

    row_lookup: dict[str, set[tuple[Any, ...]]] = {}
    for row in frontier_rows:
        controller_id = row.get(_COLUMN_CONTROLLER_ID)
        if controller_id is None:
            continue
        setting_key = (
            row.get(_COLUMN_SCENARIO),
            row.get(_COLUMN_BACKBONE_NAME),
            row.get(_COLUMN_STRATEGY_NAME),
            row.get('seed'),
            row.get(COLUMN_B),
            row.get(COLUMN_REPAIR_BUDGET_FRACTION),
            row.get(COLUMN_REPAIR_BUDGET_TOTAL),
        )
        row_lookup.setdefault(str(controller_id), set()).add(setting_key)

    best_pair: tuple[str, str] | None = None
    best_overlap = -1
    for index, left in enumerate(controller_ids):
        for right in controller_ids[index + 1:]:
            overlap = len(row_lookup.get(left, set()) & row_lookup.get(right, set()))
            if overlap > best_overlap:
                best_pair = (left, right)
                best_overlap = overlap
    return best_pair


def plot_analysis_outputs(
    *,
    frontier_rows: list[dict[str, Any]] | None = None,
    impact_rows: list[dict[str, Any]] | None = None,
    analysis_out: str | Path | None = None,
    mode: str = 'show',
    save_dir: str | Path | None = None,
) -> PlotAnalysisResult:
    """
    Plot repairability-frontier analysis artifacts.

    Args:
        frontier_rows: Optional rows from `frontier/repair_frontier.csv`.
        impact_rows: Optional rows from `frontier/repair_impact.csv`.
        analysis_out: Root analysis output directory used to load default inputs.
        mode: One of `none`, `show`, `save`, or `both`.
        save_dir: Optional output directory for PNG files.

    Returns:
        PlotAnalysisResult: Saved-path and skipped-plot metadata.
    """
    mode = str(mode).lower().strip()
    if mode not in {'none', 'show', 'save', 'both'}:
        raise ValueError(f'Invalid plotting mode: {mode}. Expected one of: none/show/save/both')

    analysis_out_p: Optional[Path] = Path(analysis_out) if analysis_out is not None else None
    frontier_dir = analysis_out_p / 'frontier' if analysis_out_p is not None else None
    if frontier_rows is None and frontier_dir is not None:
        frontier_rows = _read_csv_rows(frontier_dir / 'repair_frontier.csv')
    if impact_rows is None and frontier_dir is not None:
        impact_rows = _read_csv_rows(frontier_dir / 'repair_impact.csv')

    frontier_rows = frontier_rows or []
    impact_rows = impact_rows or []
    if not frontier_rows and not impact_rows:
        return PlotAnalysisResult(
            saved_paths=[],
            saved_filenames=[],
            skipped=[],
        )

    save_paths: list[Path] = []
    skipped_plots: list[dict[str, Any]] = []
    if mode in {'save', 'both'}:
        if save_dir is not None:
            save_dir_p = Path(save_dir)
        elif analysis_out_p is not None:
            save_dir_p = analysis_out_p / 'plots'
        else:
            save_dir_p = Path('plots')
        save_dir_p.mkdir(parents=True, exist_ok=True)
    else:
        save_dir_p = None

    import matplotlib.pyplot as plt

    def _save_or_show(*, fig: Any, filename: str) -> None:
        if save_dir_p is not None:
            out_path = save_dir_p / filename
            fig.savefig(out_path, bbox_inches='tight')
            save_paths.append(out_path)
        if mode in {'show', 'both'}:
            return
        plt.close(fig)

    impact_recovery_rows = impact_rows or _grouped_mean_rows(
        rows=frontier_rows,
        group_keys=(
            _COLUMN_SCENARIO,
            _COLUMN_BACKBONE_NAME,
            _COLUMN_STRATEGY_NAME,
            COLUMN_CONTROLLER_NAME,
            _COLUMN_CONTROLLER_ID,
            COLUMN_B,
            COLUMN_REPAIR_BUDGET_FRACTION,
            COLUMN_REPAIR_BUDGET_TOTAL,
        ),
        value_key=_COLUMN_MEAN_ABSOLUTE_RECOVERY,
    )
    grouped_recovery_rows: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = {}
    for row in impact_recovery_rows:
        grouped_recovery_rows.setdefault(
            (
                row.get(_COLUMN_SCENARIO),
                row.get(_COLUMN_BACKBONE_NAME),
                row.get(_COLUMN_STRATEGY_NAME),
            ),
            [],
        ).append(row)

    for (scenario, backbone_name, strategy_name), rows in sorted(grouped_recovery_rows.items()):
        filename = (
            f'recovery_vs_budget__{_slugify(scenario)}__{_slugify(backbone_name)}__{_slugify(strategy_name)}.png'
        )
        valid_rows = [
            row for row in rows
            if to_float(row.get(COLUMN_REPAIR_BUDGET_FRACTION)) is not None
            and to_float(row.get(_COLUMN_MEAN_ABSOLUTE_RECOVERY)) is not None
        ]
        if not valid_rows:
            _record_plot_skip(
                skipped=skipped_plots,
                filename=filename,
                reason='No recovery-vs-budget points were available.',
                context={
                    _COLUMN_SCENARIO: scenario,
                    _COLUMN_BACKBONE_NAME: backbone_name,
                    _COLUMN_STRATEGY_NAME: strategy_name,
                },
            )
            continue

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        controller_ids = _sorted_unique(row.get(_COLUMN_CONTROLLER_ID) for row in valid_rows)
        for controller_id in controller_ids:
            controller_rows = sorted(
                [row for row in valid_rows if row.get(_COLUMN_CONTROLLER_ID) == controller_id],
                key=lambda row: to_float(row.get(COLUMN_REPAIR_BUDGET_FRACTION)) or float('inf'),
            )
            xs = [to_float(row.get(COLUMN_REPAIR_BUDGET_FRACTION)) for row in controller_rows]
            ys = [to_float(row.get(_COLUMN_MEAN_ABSOLUTE_RECOVERY)) for row in controller_rows]
            pts = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
            if not pts:
                continue
            xs2, ys2 = zip(*pts)
            label = str(controller_rows[0].get(COLUMN_CONTROLLER_NAME) or controller_id)
            ax.plot(xs2, ys2, marker='o', linestyle='-', label=label)

        ax.set_xlabel('repair_budget_fraction')
        ax.set_ylabel(_COLUMN_MEAN_ABSOLUTE_RECOVERY)
        ax.set_title(f'Recovery vs budget: {scenario} / {backbone_name} / {strategy_name}')
        ax.legend()
        _save_or_show(fig=fig, filename=filename)

    valid_frontier_rows = [
        row for row in frontier_rows
        if to_float(row.get(_COLUMN_MEAN_HARMED_TASK_FRACTION)) is not None
        and to_float(row.get(_COLUMN_MEAN_ABSOLUTE_RECOVERY)) is not None
    ]
    if valid_frontier_rows:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for controller_id in _sorted_unique(row.get(_COLUMN_CONTROLLER_ID) for row in valid_frontier_rows):
            controller_rows = [row for row in valid_frontier_rows if row.get(_COLUMN_CONTROLLER_ID) == controller_id]
            xs = [to_float(row.get(_COLUMN_MEAN_HARMED_TASK_FRACTION)) for row in controller_rows]
            ys = [to_float(row.get(_COLUMN_MEAN_ABSOLUTE_RECOVERY)) for row in controller_rows]
            pts = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
            if not pts:
                continue
            xs2, ys2 = zip(*pts)
            label = str(controller_rows[0].get(COLUMN_CONTROLLER_NAME) or controller_id)
            ax.scatter(xs2, ys2, label=label)
        ax.set_xlabel(_COLUMN_MEAN_HARMED_TASK_FRACTION)
        ax.set_ylabel(_COLUMN_MEAN_ABSOLUTE_RECOVERY)
        ax.set_title('Harm vs recovery')
        ax.legend()
        _save_or_show(fig=fig, filename=_PLOT_FILENAME_HARM_VS_RECOVERY)
    else:
        _record_plot_skip(
            skipped=skipped_plots,
            filename=_PLOT_FILENAME_HARM_VS_RECOVERY,
            reason='No frontier rows had both harm and recovery metrics.',
        )

    utility_cost_rows = [
        row for row in frontier_rows
        if _action_cost_x_value(row) is not None
        and to_float(row.get(_COLUMN_UTILITY_CONSERVATIVE)) is not None
    ]
    if utility_cost_rows:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for controller_id in _sorted_unique(row.get(_COLUMN_CONTROLLER_ID) for row in utility_cost_rows):
            controller_rows = [row for row in utility_cost_rows if row.get(_COLUMN_CONTROLLER_ID) == controller_id]
            xs = [_action_cost_x_value(row) for row in controller_rows]
            ys = [to_float(row.get(_COLUMN_UTILITY_CONSERVATIVE)) for row in controller_rows]
            pts = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
            if not pts:
                continue
            xs2, ys2 = zip(*pts)
            label = str(controller_rows[0].get(COLUMN_CONTROLLER_NAME) or controller_id)
            ax.scatter(xs2, ys2, label=label)
        ax.set_xlabel(_action_cost_axis_label(utility_cost_rows))
        ax.set_ylabel(_COLUMN_UTILITY_CONSERVATIVE)
        ax.set_title('Utility vs cost')
        ax.legend()
        _save_or_show(fig=fig, filename=_PLOT_FILENAME_UTILITY_VS_COST)
    else:
        _record_plot_skip(
            skipped=skipped_plots,
            filename=_PLOT_FILENAME_UTILITY_VS_COST,
            reason='No frontier rows had both budget and utility metrics.',
        )

    utility_pair = _choose_utility_delta_pair(frontier_rows=frontier_rows)
    if utility_pair is None:
        _record_plot_skip(
            skipped=skipped_plots,
            filename='utility_delta__<controller_a>__<controller_b>.png',
            reason='At least two controllers with overlapping settings are required.',
        )
    else:
        left_id, right_id = utility_pair
        left_rows = {
            (
                row.get(_COLUMN_SCENARIO),
                row.get(_COLUMN_BACKBONE_NAME),
                row.get(_COLUMN_STRATEGY_NAME),
                row.get('seed'),
                row.get(COLUMN_B),
                row.get(COLUMN_REPAIR_BUDGET_FRACTION),
                row.get(COLUMN_REPAIR_BUDGET_TOTAL),
            ): row
            for row in frontier_rows
            if row.get(_COLUMN_CONTROLLER_ID) == left_id
        }
        right_rows = {
            (
                row.get(_COLUMN_SCENARIO),
                row.get(_COLUMN_BACKBONE_NAME),
                row.get(_COLUMN_STRATEGY_NAME),
                row.get('seed'),
                row.get(COLUMN_B),
                row.get(COLUMN_REPAIR_BUDGET_FRACTION),
                row.get(COLUMN_REPAIR_BUDGET_TOTAL),
            ): row
            for row in frontier_rows
            if row.get(_COLUMN_CONTROLLER_ID) == right_id
        }
        overlap_keys = sorted(set(left_rows) & set(right_rows))
        delta_rows: list[tuple[float, float]] = []
        for overlap_key in overlap_keys:
            left_value = to_float(left_rows[overlap_key].get(_COLUMN_UTILITY_CONSERVATIVE))
            right_value = to_float(right_rows[overlap_key].get(_COLUMN_UTILITY_CONSERVATIVE))
            budget_value = to_float(left_rows[overlap_key].get(COLUMN_REPAIR_BUDGET_FRACTION))
            if left_value is None or right_value is None or budget_value is None:
                continue
            delta_rows.append((budget_value, left_value - right_value))

        filename = f'utility_delta__{_slugify(left_id)}__{_slugify(right_id)}.png'
        if not delta_rows:
            _record_plot_skip(
                skipped=skipped_plots,
                filename=filename,
                reason='No overlapping utility_conservative values were available.',
            )
        else:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            xs, ys = zip(*delta_rows)
            ax.scatter(xs, ys)
            ax.axhline(0.0, color='black', linestyle='--', linewidth=1.0)
            ax.set_xlabel('repair_budget_fraction')
            ax.set_ylabel(f'{left_id} - {right_id} utility_conservative')
            ax.set_title('Utility delta')
            _save_or_show(fig=fig, filename=filename)

    harm_budget_rows = impact_rows or _grouped_mean_rows(
        rows=frontier_rows,
        group_keys=(
            _COLUMN_SCENARIO,
            _COLUMN_BACKBONE_NAME,
            COLUMN_CONTROLLER_NAME,
            _COLUMN_CONTROLLER_ID,
            COLUMN_B,
            COLUMN_REPAIR_BUDGET_FRACTION,
            COLUMN_REPAIR_BUDGET_TOTAL,
        ),
        value_key=_COLUMN_MEAN_HARMED_TASK_FRACTION,
    )
    grouped_harm_rows: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for row in harm_budget_rows:
        grouped_harm_rows.setdefault(
            (
                row.get(_COLUMN_SCENARIO),
                row.get(_COLUMN_BACKBONE_NAME),
            ),
            [],
        ).append(row)

    for (scenario, backbone_name), rows in sorted(
        grouped_harm_rows.items(),
        key=lambda item: tuple('' if value is None else str(value) for value in item[0]),
    ):
        filename = f'harm_vs_budget__{_slugify(scenario)}__{_slugify(backbone_name)}.png'
        aggregated_rows = _grouped_mean_rows(
            rows=rows,
            group_keys=(
                _COLUMN_SCENARIO,
                _COLUMN_BACKBONE_NAME,
                COLUMN_CONTROLLER_NAME,
                _COLUMN_CONTROLLER_ID,
                COLUMN_B,
                COLUMN_REPAIR_BUDGET_FRACTION,
                COLUMN_REPAIR_BUDGET_TOTAL,
            ),
            value_key=_COLUMN_MEAN_HARMED_TASK_FRACTION,
        )
        valid_rows = [
            row for row in aggregated_rows
            if to_float(row.get(COLUMN_REPAIR_BUDGET_FRACTION)) is not None
            and to_float(row.get(_COLUMN_MEAN_HARMED_TASK_FRACTION)) is not None
        ]
        if not valid_rows:
            _record_plot_skip(
                skipped=skipped_plots,
                filename=filename,
                reason='No harm-vs-budget points were available.',
                context={
                    _COLUMN_SCENARIO: scenario,
                    _COLUMN_BACKBONE_NAME: backbone_name,
                },
            )
            continue

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for controller_id in _sorted_unique(row.get(_COLUMN_CONTROLLER_ID) for row in valid_rows):
            controller_rows = sorted(
                [row for row in valid_rows if row.get(_COLUMN_CONTROLLER_ID) == controller_id],
                key=lambda row: to_float(row.get(COLUMN_REPAIR_BUDGET_FRACTION)) or float('inf'),
            )
            xs = [to_float(row.get(COLUMN_REPAIR_BUDGET_FRACTION)) for row in controller_rows]
            ys = [to_float(row.get(_COLUMN_MEAN_HARMED_TASK_FRACTION)) for row in controller_rows]
            pts = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
            if not pts:
                continue
            xs2, ys2 = zip(*pts)
            label = str(controller_rows[0].get(COLUMN_CONTROLLER_NAME) or controller_id)
            ax.plot(xs2, ys2, marker='o', linestyle='-', label=label)

        ax.set_xlabel('repair_budget_fraction')
        ax.set_ylabel(_COLUMN_MEAN_HARMED_TASK_FRACTION)
        ax.set_title(f'Harm vs budget: {scenario} / {backbone_name}')
        ax.legend()
        _save_or_show(fig=fig, filename=filename)

    if mode in {'show', 'both'}:
        import matplotlib.pyplot as plt_show
        plt_show.show()
    else:
        import matplotlib.pyplot as plt_close
        plt_close.close('all')

    return PlotAnalysisResult(
        saved_paths=save_paths,
        saved_filenames=[path.name for path in save_paths],
        skipped=skipped_plots,
    )
