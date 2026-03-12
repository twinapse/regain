"""
Tests for CLI staged-output publishing helpers.
"""

from pathlib import Path

from regain.cli._utils._output_helpers import CliFailure
from regain.cli._utils._output_helpers import finalize_staged_outputs
from regain.cli._utils._output_helpers import StagedOutput


def test_finalize_staged_outputs_rejects_empty_source(tmp_path: Path) -> None:
    empty_source = tmp_path / 'empty'
    empty_source.mkdir(parents=True, exist_ok=True)
    destination = tmp_path / 'published'
    failures: list[CliFailure] = []

    published_count = finalize_staged_outputs(
        outputs=[
            StagedOutput(
                scope='empty-source',
                source=empty_source,
                destination=destination,
            )
        ],
        failures=failures,
        allow_partial=False,
        overwrite=False,
    )

    assert published_count == 0
    assert not destination.exists()
    assert len(failures) == 1
    assert 'Staged output is empty' in failures[0].message


def test_finalize_staged_outputs_rolls_back_when_late_publish_fails(tmp_path: Path) -> None:
    first_source = tmp_path / 'first_source.txt'
    first_source.write_text('first', encoding='utf-8')
    second_source = tmp_path / 'second_source.txt'
    second_source.write_text('second', encoding='utf-8')

    first_destination = tmp_path / 'first_destination.txt'
    blocked_parent = tmp_path / 'blocked_parent'
    blocked_parent.write_text('blocking file', encoding='utf-8')
    second_destination = blocked_parent / 'second_destination.txt'

    failures: list[CliFailure] = []

    published_count = finalize_staged_outputs(
        outputs=[
            StagedOutput(
                scope='first',
                source=first_source,
                destination=first_destination,
            ),
            StagedOutput(
                scope='second',
                source=second_source,
                destination=second_destination,
            ),
        ],
        failures=failures,
        allow_partial=False,
        overwrite=False,
    )

    assert published_count == 0
    assert not first_destination.exists()
    assert first_source.exists()
    assert second_source.exists()
    assert len(failures) == 1
    assert failures[0].scope == 'second'


def test_finalize_staged_outputs_allow_partial_skips_empty_source(tmp_path: Path) -> None:
    empty_source = tmp_path / 'empty_source'
    empty_source.mkdir(parents=True, exist_ok=True)

    non_empty_source = tmp_path / 'non_empty_source.txt'
    non_empty_source.write_text('payload', encoding='utf-8')

    empty_destination = tmp_path / 'empty_destination'
    non_empty_destination = tmp_path / 'non_empty_destination.txt'
    failures: list[CliFailure] = []

    published_count = finalize_staged_outputs(
        outputs=[
            StagedOutput(
                scope='empty',
                source=empty_source,
                destination=empty_destination,
            ),
            StagedOutput(
                scope='non-empty',
                source=non_empty_source,
                destination=non_empty_destination,
            ),
        ],
        failures=failures,
        allow_partial=True,
        overwrite=False,
    )

    assert published_count == 1
    assert not empty_destination.exists()
    assert non_empty_destination.exists()
    assert len(failures) == 1
    assert failures[0].scope == 'empty'
    assert 'Staged output is empty' in failures[0].message
