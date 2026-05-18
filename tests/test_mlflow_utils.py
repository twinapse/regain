"""
Tests for MLflow utility helpers.
"""

from pathlib import Path
from types import SimpleNamespace

import mlflow
from mlflow.utils.mlflow_tags import MLFLOW_GIT_COMMIT
import pytest

from regain.constants import COLUMN_GIT_COMMIT
from regain.constants import MLFLOW_ARTIFACT_ERROR_FILE
from regain.constants import RUN_ACC_FINAL_AVG_BASE
from regain.constants import RUN_ACC_REF
from regain.mlflow_utils import _log_fatal_error_artifact
from regain.mlflow_utils import build_mlflow_run_columns
from regain.mlflow_utils import extract_mlflow_run_git_commit
from regain.mlflow_utils import log_fatal_error_context
from regain.mlflow_utils import resolve_git_commit


def _make_run(
    *,
    metrics: dict[str, float],
    params: dict[str, str] | None = None,
    tags: dict[str, str] | None = None,
) -> SimpleNamespace:
    """
    Build a minimal MLflow run stub for export tests.

    Args:
        metrics (dict[str, float]): Run metric payload.
        params (dict[str, str] | None): Optional run params.
        tags (dict[str, str] | None): Optional run tags.

    Returns:
        SimpleNamespace: Run stub.
    """
    return SimpleNamespace(
        info=SimpleNamespace(
            run_id='run_1',
            status='FINISHED',
            start_time=0,
            end_time=0,
        ),
        data=SimpleNamespace(
            params=params or {'seed': '1'},
            metrics=metrics,
            tags=tags,
        ),
    )


class _FakeHistoryClient:
    """
    Fake MLflow history client for testing.
    """

    def __init__(
        self,
        *,
        histories: dict[str, list[SimpleNamespace]],
        raising_metrics: set[str] | None = None,
    ) -> None:
        self._histories = histories
        self._raising_metrics = set(raising_metrics or set())

    def get_metric_history(self, run_id: str, metric_key: str) -> list[SimpleNamespace]:
        assert run_id == 'run_1'
        if metric_key in self._raising_metrics:
            raise RuntimeError(f'failed: {metric_key}')
        return list(self._histories.get(metric_key, []))


class TestLogFatalErrorArtifact:
    """
    Tests for fatal error artifact logging.
    """

    def test_logs_error_text_artifact_when_run_is_active(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        logged_payloads: list[tuple[str, str]] = []

        monkeypatch.setattr(mlflow, 'active_run', object)
        monkeypatch.setattr(
            mlflow,
            'log_text',
            lambda text, artifact_file: logged_payloads.append((str(text), str(artifact_file))),
        )

        _log_fatal_error_artifact(
            run_name='run_a',
            exc=RuntimeError('fatal failure'),
            traceback_text='Traceback (most recent call last):\nRuntimeError: fatal failure',
        )

        assert len(logged_payloads) == 1
        payload, artifact_path = logged_payloads[0]
        assert artifact_path == MLFLOW_ARTIFACT_ERROR_FILE
        assert 'timestamp_utc: ' in payload
        assert 'run_name: run_a' in payload
        assert 'exception_type: RuntimeError' in payload
        assert 'exception_message: fatal failure' in payload
        assert 'traceback:\nTraceback (most recent call last):' in payload

    def test_no_ops_when_mlflow_run_is_not_active(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        called = False

        monkeypatch.setattr(mlflow, 'active_run', lambda: None)

        def _log_text(text: str, artifact_file: str) -> None:
            nonlocal called
            called = True

        monkeypatch.setattr(mlflow, 'log_text', _log_text)

        _log_fatal_error_artifact(
            run_name='run_b',
            exc=RuntimeError('fatal failure'),
            traceback_text='Traceback',
        )

        assert called is False

    def test_swallows_mlflow_log_text_errors(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(mlflow, 'active_run', object)

        def _raise(*args, **kwargs) -> None:
            del args, kwargs
            raise RuntimeError('mlflow unavailable')

        monkeypatch.setattr(mlflow, 'log_text', _raise)

        _log_fatal_error_artifact(
            run_name='run_c',
            exc=RuntimeError('fatal failure'),
            traceback_text='Traceback',
        )


class TestLogFatalErrorContext:
    """
    Tests for fatal error logging context.
    """

    def test_logs_and_reraises_when_exception_occurs(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: list[tuple[str, Exception, str]] = []

        def _capture(*, run_name: str, exc: Exception, traceback_text: str) -> None:
            captured.append((run_name, exc, traceback_text))

        monkeypatch.setattr('regain.mlflow_utils._log_fatal_error_artifact', _capture)

        with pytest.raises(RuntimeError, match='boom'):
            with log_fatal_error_context(run_name='run_x'):
                raise RuntimeError('boom')

        assert len(captured) == 1
        run_name, exc, traceback_text = captured[0]
        assert run_name == 'run_x'
        assert isinstance(exc, RuntimeError)
        assert str(exc) == 'boom'
        assert 'RuntimeError: boom' in traceback_text


class TestGitCommitHelpers:
    """
    Tests for Git commit helper utilities.
    """

    def test_extract_mlflow_run_git_commit_returns_tag_when_present(self) -> None:
        run = _make_run(
            metrics={},
            tags={MLFLOW_GIT_COMMIT: 'a' * 40},
        )

        assert extract_mlflow_run_git_commit(run) == 'a' * 40

    def test_extract_mlflow_run_git_commit_returns_empty_string_when_tag_missing(self) -> None:
        run = _make_run(metrics={}, tags={'other_tag': 'value'})

        assert extract_mlflow_run_git_commit(run) == ''

    def test_extract_mlflow_run_git_commit_returns_empty_string_when_tags_are_none(self) -> None:
        run = _make_run(metrics={}, tags=None)

        assert extract_mlflow_run_git_commit(run) == ''

    def test_resolve_git_commit_returns_repo_commit_and_none_outside_repo(
        self,
        tmp_path: Path,
    ) -> None:
        repo_commit = resolve_git_commit()

        assert repo_commit is not None
        assert len(repo_commit) == 40
        assert set(repo_commit) <= set('0123456789abcdef')
        assert resolve_git_commit(tmp_path) is None


class TestBuildMlflowRunColumns:
    """
    Tests for MLflow run column building.
    """

    def test_includes_git_commit_and_reserves_column_name(self) -> None:
        run = _make_run(
            metrics={
                COLUMN_GIT_COMMIT: 0.5,
                RUN_ACC_FINAL_AVG_BASE: 0.77,
            },
            params={
                'seed': '1',
                COLUMN_GIT_COMMIT: 'param_override',
            },
            tags={MLFLOW_GIT_COMMIT: 'b' * 40},
        )

        columns = build_mlflow_run_columns(run=run)

        assert columns[COLUMN_GIT_COMMIT] == 'b' * 40
        assert columns[RUN_ACC_FINAL_AVG_BASE] == pytest.approx(0.77)

    def test_materializes_history_bearing_eval_metrics_with_actual_after_exp_tokens(self) -> None:
        run = _make_run(
            metrics={
                f'{RUN_ACC_REF}.exp000.base': 0.91,
                f'{RUN_ACC_REF}.exp001.base': 0.82,
                f'{RUN_ACC_REF}.exp002.base': 0.73,
                'run.eval.forgetting.exp000': 0.22,
                'run.eval.transfer.stream': 0.44,
                RUN_ACC_FINAL_AVG_BASE: 0.77,
            })
        client = _FakeHistoryClient(
            histories={
                f'{RUN_ACC_REF}.exp000.base': [SimpleNamespace(step=10, value=0.91)],
                f'{RUN_ACC_REF}.exp001.base': [SimpleNamespace(step=20, value=0.82)],
                f'{RUN_ACC_REF}.exp002.base': [SimpleNamespace(step=30, value=0.73)],
                'run.eval.forgetting.exp000': [
                    SimpleNamespace(step=20, value=0.11),
                    SimpleNamespace(step=30, value=0.22),
                ],
                'run.eval.transfer.stream': [
                    SimpleNamespace(step=20, value=0.33),
                    SimpleNamespace(step=30, value=0.44),
                ],
            })

        columns = build_mlflow_run_columns(
            run=run,
            client=client,
        )

        assert columns['run.eval.forgetting.after_exp001.exp000'] == pytest.approx(0.11)
        assert columns['run.eval.forgetting.after_exp002.exp000'] == pytest.approx(0.22)
        assert columns['run.eval.transfer.after_exp001.stream'] == pytest.approx(0.33)
        assert columns['run.eval.transfer.after_exp002.stream'] == pytest.approx(0.44)
        assert 'run.eval.forgetting.exp000' not in columns
        assert 'run.eval.transfer.stream' not in columns
        assert columns[RUN_ACC_FINAL_AVG_BASE] == pytest.approx(0.77)

    def test_raises_when_ref_accuracy_history_is_missing(self) -> None:
        run = _make_run(metrics={
            f'{RUN_ACC_REF}.exp000.base': 0.91,
            'run.eval.forgetting.exp000': 0.22,
        })
        client = _FakeHistoryClient(histories={
            'run.eval.forgetting.exp000': [SimpleNamespace(step=10, value=0.22)],
        })

        with pytest.raises(ValueError, match='Missing required MLflow metric history'):
            build_mlflow_run_columns(
                run=run,
                client=client,
            )

    def test_raises_when_ref_accuracy_history_is_ambiguous(self) -> None:
        run = _make_run(metrics={
            f'{RUN_ACC_REF}.exp000.base': 0.91,
            'run.eval.forgetting.exp000': 0.22,
        })
        client = _FakeHistoryClient(
            histories={
                f'{RUN_ACC_REF}.exp000.base': [
                    SimpleNamespace(step=10, value=0.91),
                    SimpleNamespace(step=20, value=0.92),
                ],
                'run.eval.forgetting.exp000': [SimpleNamespace(step=10, value=0.22)],
            })

        with pytest.raises(ValueError, match='exactly one checkpoint step'):
            build_mlflow_run_columns(
                run=run,
                client=client,
            )

    def test_raises_when_history_bearing_metric_history_is_missing(self) -> None:
        run = _make_run(metrics={
            f'{RUN_ACC_REF}.exp000.base': 0.91,
            'run.eval.forgetting.exp000': 0.22,
        })
        client = _FakeHistoryClient(histories={
            f'{RUN_ACC_REF}.exp000.base': [SimpleNamespace(step=10, value=0.91)],
        })

        with pytest.raises(ValueError, match='Missing required MLflow metric history'):
            build_mlflow_run_columns(
                run=run,
                client=client,
            )

    def test_skips_unmapped_bootstrap_step_zero_when_other_steps_are_mapped(self) -> None:
        run = _make_run(metrics={
            f'{RUN_ACC_REF}.exp000.base': 0.91,
            f'{RUN_ACC_REF}.exp001.base': 0.82,
            'run.eval.forgetting.exp000': 0.22,
        })
        client = _FakeHistoryClient(
            histories={
                f'{RUN_ACC_REF}.exp000.base': [SimpleNamespace(step=10, value=0.91)],
                f'{RUN_ACC_REF}.exp001.base': [SimpleNamespace(step=20, value=0.82)],
                'run.eval.forgetting.exp000': [
                    SimpleNamespace(step=0, value=0.05),
                    SimpleNamespace(step=20, value=0.22),
                ],
            })

        columns = build_mlflow_run_columns(
            run=run,
            client=client,
        )

        assert columns['run.eval.forgetting.after_exp001.exp000'] == pytest.approx(0.22)
        assert 'run.eval.forgetting.after_exp000.exp000' not in columns
        assert 'run.eval.forgetting.exp000' not in columns

    def test_skips_unmapped_bootstrap_step_zero_when_it_is_the_only_history_point(self) -> None:
        run = _make_run(metrics={
            f'{RUN_ACC_REF}.exp000.base': 0.91,
            'run.eval.forgetting.exp000': 0.22,
        })
        client = _FakeHistoryClient(
            histories={
                f'{RUN_ACC_REF}.exp000.base': [SimpleNamespace(step=10, value=0.91)],
                'run.eval.forgetting.exp000': [SimpleNamespace(step=0, value=0.22)],
            })

        columns = build_mlflow_run_columns(
            run=run,
            client=client,
        )

        assert not any(key.startswith('run.eval.forgetting.after_exp') for key in columns)
        assert 'run.eval.forgetting.exp000' not in columns

    def test_skips_unmapped_bootstrap_step_zero_for_stream_forgetting(self) -> None:
        run = _make_run(metrics={
            f'{RUN_ACC_REF}.exp000.base': 0.91,
            f'{RUN_ACC_REF}.exp001.base': 0.82,
            'run.eval.forgetting.stream': 0.18,
        })
        client = _FakeHistoryClient(
            histories={
                f'{RUN_ACC_REF}.exp000.base': [SimpleNamespace(step=10, value=0.91)],
                f'{RUN_ACC_REF}.exp001.base': [SimpleNamespace(step=20, value=0.82)],
                'run.eval.forgetting.stream': [
                    SimpleNamespace(step=0, value=0.0),
                    SimpleNamespace(step=10, value=0.07),
                    SimpleNamespace(step=20, value=0.18),
                ],
            })

        columns = build_mlflow_run_columns(
            run=run,
            client=client,
        )

        assert columns['run.eval.forgetting.after_exp000.stream'] == pytest.approx(0.07)
        assert columns['run.eval.forgetting.after_exp001.stream'] == pytest.approx(0.18)
        assert 'run.eval.forgetting.stream' not in columns

    def test_raises_when_history_bearing_metric_step_is_unmapped(self) -> None:
        run = _make_run(metrics={
            f'{RUN_ACC_REF}.exp000.base': 0.91,
            f'{RUN_ACC_REF}.exp001.base': 0.82,
            'run.eval.forgetting.exp000': 0.22,
        })
        client = _FakeHistoryClient(
            histories={
                f'{RUN_ACC_REF}.exp000.base': [SimpleNamespace(step=10, value=0.91)],
                f'{RUN_ACC_REF}.exp001.base': [SimpleNamespace(step=20, value=0.82)],
                'run.eval.forgetting.exp000': [SimpleNamespace(step=30, value=0.22)],
            })

        with pytest.raises(ValueError, match='no matching checkpoint identity'):
            build_mlflow_run_columns(
                run=run,
                client=client,
            )

    def test_raises_when_history_lookup_fails(self) -> None:
        run = _make_run(metrics={
            f'{RUN_ACC_REF}.exp000.base': 0.91,
            'run.eval.forgetting.exp000': 0.22,
        })
        client = _FakeHistoryClient(
            histories={
                f'{RUN_ACC_REF}.exp000.base': [SimpleNamespace(step=10, value=0.91)],
            },
            raising_metrics={'run.eval.forgetting.exp000'},
        )

        with pytest.raises(ValueError, match='Failed to fetch required MLflow metric history'):
            build_mlflow_run_columns(
                run=run,
                client=client,
            )
