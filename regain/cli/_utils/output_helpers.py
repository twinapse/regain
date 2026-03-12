"""
Shared helpers for staged CLI output publishing and failure reporting.
"""

from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Sequence

__all__ = [
    'CliFailure',
    'StagedOutput',
    'add_failure',
    'finalize_staged_outputs',
    'has_directory_content',
    'resolve_exit_code',
    'print_failure_summary',
]


@dataclass(frozen=True)
class CliFailure:
    """
    Structured CLI failure entry.

    Attributes:
        scope (str): Failure scope label.
        message (str): Human-readable failure message.
    """

    scope: str
    message: str


@dataclass(frozen=True)
class StagedOutput:
    """
    Staged output item to publish.

    Attributes:
        scope (str): Scope label used for friendly error messages.
        source (Path): Staged source path in a temporary directory.
        destination (Path): Final output destination path.
    """

    scope: str
    source: Path
    destination: Path


def add_failure(
    *,
    failures: list[CliFailure],
    scope: str,
    error: Exception | str,
) -> None:
    """
    Append one structured failure.

    Args:
        failures (list[CliFailure]): Mutable failure list.
        scope (str): Scope label.
        error (Exception | str): Source exception or message.

    Returns:
        None
    """
    raw_message = str(error).strip()
    message = raw_message if raw_message else repr(error)
    failures.append(CliFailure(scope=scope, message=message))


def print_failure_summary(
    *,
    command_name: str,
    failures: Sequence[CliFailure],
) -> None:
    """
    Print a friendly summary for all collected failures.

    Args:
        command_name (str): Command display name.
        failures (Sequence[CliFailure]): Collected failures.

    Returns:
        None
    """
    if not failures:
        return
    print(
        f'{command_name} completed with {len(failures)} failure(s):',
        file=sys.stderr,
    )
    for index, failure in enumerate(failures, start=1):
        print(f'  {index}. [{failure.scope}] {failure.message}', file=sys.stderr)


def has_directory_content(*, directory: Path) -> bool:
    """
    Check whether a directory exists and contains at least one entry.

    Args:
        directory (Path): Directory to inspect.

    Returns:
        bool: True when the directory exists and is non-empty.
    """
    if not directory.exists() or not directory.is_dir():
        return False
    return any(True for _ in directory.iterdir())


def finalize_staged_outputs(
    *,
    outputs: Sequence[StagedOutput],
    failures: list[CliFailure],
    allow_partial: bool,
    overwrite: bool,
) -> int:
    """
    Publish staged outputs according to partial/overwrite policy.

    Args:
        outputs (Sequence[StagedOutput]): Staged output items.
        failures (list[CliFailure]): Mutable collected failures.
        allow_partial (bool): Whether partial publishing is allowed.
        overwrite (bool): Whether existing destination paths may be replaced.

    Returns:
        int: Number of successfully published items.
    """
    if not outputs:
        return 0

    duplicate_destination_indexes = _find_duplicate_destination_indexes(outputs=outputs)
    publishable_outputs: list[StagedOutput] = []

    for output_index, output in enumerate(outputs):
        if output_index in duplicate_destination_indexes:
            add_failure(
                failures=failures,
                scope=output.scope,
                error=f'Duplicate target path in staged outputs: {output.destination}',
            )
            continue

        if not _has_publishable_content(path=output.source):
            if output.source.exists():
                add_failure(
                    failures=failures,
                    scope=output.scope,
                    error=f'Staged output is empty: {output.source}',
                )
            else:
                add_failure(
                    failures=failures,
                    scope=output.scope,
                    error=f'Staged output does not exist: {output.source}',
                )
            continue

        if output.destination.exists() and not overwrite:
            add_failure(
                failures=failures,
                scope=output.scope,
                error=f'Target path already exists: {output.destination}',
            )
            continue

        publishable_outputs.append(output)

    if failures and not allow_partial:
        return 0
    if not publishable_outputs:
        return 0

    if allow_partial:
        return _publish_outputs_best_effort(
            outputs=publishable_outputs,
            failures=failures,
            overwrite=overwrite,
        )
    return _publish_outputs_transactionally(
        outputs=publishable_outputs,
        failures=failures,
        overwrite=overwrite,
    )


def resolve_exit_code(
    *,
    failures: Sequence[CliFailure],
    allow_partial: bool,
    published_count: int | None = None,
) -> int:
    """
    Resolve CLI exit code from failure list and partial policy.

    Args:
        failures (Sequence[CliFailure]): Collected failures.
        allow_partial (bool): Whether partial outputs are allowed.
        published_count (int | None): Number of published outputs when known.

    Returns:
        int: Exit code (0 or 1).
    """
    if not failures:
        return 0
    if not allow_partial:
        return 1
    if published_count is None:
        return 0
    return 0 if int(published_count) > 0 else 1


def _publish_output(
    *,
    source: Path,
    destination: Path,
    overwrite: bool,
) -> None:
    """
    Move a staged output path to its final destination.

    Args:
        source (Path): Staged source path.
        destination (Path): Final destination path.
        overwrite (bool): Whether existing destination paths may be replaced.

    Returns:
        None

    Raises:
        FileExistsError: If destination already exists and overwrite is disabled.
        FileNotFoundError: If source path does not exist.
    """
    if not source.exists():
        raise FileNotFoundError(f'Staged output does not exist: {source}')

    if destination.exists():
        if not overwrite:
            raise FileExistsError(f'Target path already exists: {destination}')
        _remove_existing_path(path=destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))


def _publish_outputs_best_effort(
    *,
    outputs: Sequence[StagedOutput],
    failures: list[CliFailure],
    overwrite: bool,
) -> int:
    """
    Publish outputs independently, allowing partial success.

    Args:
        outputs (Sequence[StagedOutput]): Publishable output items.
        failures (list[CliFailure]): Mutable collected failures.
        overwrite (bool): Whether existing destination paths may be replaced.

    Returns:
        int: Number of successfully published outputs.
    """
    published_count = 0
    for output in outputs:
        try:
            _publish_output(
                source=output.source,
                destination=output.destination,
                overwrite=overwrite,
            )
            published_count += 1
        except Exception as exc:
            add_failure(
                failures=failures,
                scope=output.scope,
                error=exc,
            )
    return published_count


def _publish_outputs_transactionally(
    *,
    outputs: Sequence[StagedOutput],
    failures: list[CliFailure],
    overwrite: bool,
) -> int:
    """
    Publish outputs atomically, rolling back on failure.

    Args:
        outputs (Sequence[StagedOutput]): Publishable output items.
        failures (list[CliFailure]): Mutable collected failures.
        overwrite (bool): Whether existing destination paths may be replaced.

    Returns:
        int: Number of successfully published outputs.
    """
    published_outputs: list[tuple[StagedOutput, Path | None]] = []
    with tempfile.TemporaryDirectory() as temp_dir:
        backup_root = Path(temp_dir)
        for output in outputs:
            destination_backup_path: Path | None = None
            try:
                if output.destination.exists():
                    if not overwrite:
                        raise FileExistsError(f'Target path already exists: {output.destination}')
                    destination_backup_path = _backup_existing_destination(
                        destination=output.destination,
                        backup_root=backup_root,
                        backup_index=len(published_outputs),
                    )
                output.destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(output.source), str(output.destination))
                published_outputs.append((output, destination_backup_path))
            except Exception as exc:
                _rollback_transaction(
                    published_outputs=published_outputs,
                    failed_output=output,
                    failed_output_backup=destination_backup_path,
                )
                add_failure(
                    failures=failures,
                    scope=output.scope,
                    error=exc,
                )
                return 0
    return len(outputs)


def _backup_existing_destination(
    *,
    destination: Path,
    backup_root: Path,
    backup_index: int,
) -> Path:
    """
    Move an existing destination path into a rollback backup location.

    Args:
        destination (Path): Existing destination path.
        backup_root (Path): Root temporary directory for rollback backups.
        backup_index (int): Stable index for backup naming.

    Returns:
        Path: Backup path containing the previous destination content.
    """
    backup_path = backup_root / f'backup_{backup_index}'
    shutil.move(str(destination), str(backup_path))
    return backup_path


def _rollback_transaction(
    *,
    published_outputs: Sequence[tuple[StagedOutput, Path | None]],
    failed_output: StagedOutput,
    failed_output_backup: Path | None,
) -> None:
    """
    Roll back previously published outputs after a transactional failure.

    Args:
        published_outputs (Sequence[tuple[StagedOutput, Path | None]]): Outputs already published.
        failed_output (StagedOutput): Output that failed during publish.
        failed_output_backup (Path | None): Backup for failed output destination, when present.

    Returns:
        None
    """
    if failed_output_backup is not None and failed_output_backup.exists():
        try:
            failed_output.destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(failed_output_backup), str(failed_output.destination))
        except Exception:
            pass

    for output, backup_path in reversed(list(published_outputs)):
        try:
            if output.destination.exists():
                output.source.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(output.destination), str(output.source))
            if backup_path is not None and backup_path.exists():
                output.destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(backup_path), str(output.destination))
        except Exception:
            continue


def _find_duplicate_destination_indexes(
    *,
    outputs: Sequence[StagedOutput],
) -> set[int]:
    """
    Identify indexes whose destination path is duplicated in the staged output list.

    Args:
        outputs (Sequence[StagedOutput]): Staged output items.

    Returns:
        set[int]: Indexes participating in duplicate destinations.
    """
    destination_to_indexes: dict[str, list[int]] = {}
    for output_index, output in enumerate(outputs):
        destination_key = str(output.destination.resolve(strict=False))
        destination_to_indexes.setdefault(destination_key, []).append(output_index)

    duplicate_indexes: set[int] = set()
    for output_indexes in destination_to_indexes.values():
        if len(output_indexes) <= 1:
            continue
        duplicate_indexes.update(output_indexes)
    return duplicate_indexes


def _has_publishable_content(*, path: Path) -> bool:
    """
    Check whether a staged output path contains publishable content.

    Args:
        path (Path): Path to validate.

    Returns:
        bool: True when path exists and has non-empty content.
    """
    if not path.exists():
        return False
    if path.is_dir() and not path.is_symlink():
        for child in path.rglob('*'):
            if child.is_dir():
                continue
            try:
                if child.stat().st_size > 0:
                    return True
            except OSError:
                continue
        return False
    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def _remove_existing_path(*, path: Path) -> None:
    """
    Remove an existing file or directory path.

    Args:
        path (Path): Path to remove.

    Returns:
        None
    """
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
        return
    path.unlink()
