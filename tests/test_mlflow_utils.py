"""
Tests for MLflow utility helpers.
"""

import mlflow
import pytest

from regain.constants import MLFLOW_ARTIFACT_ERROR_FILE
from regain.mlflow_utils import _log_fatal_error_artifact
from regain.mlflow_utils import log_fatal_error_context


class TestLogFatalErrorArtifact:
    def test_logs_error_text_artifact_when_run_is_active(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        logged_payloads: list[tuple[str, str]] = []

        monkeypatch.setattr(mlflow, 'active_run', lambda: object())
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

    def test_noops_when_mlflow_run_is_not_active(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        called = False

        monkeypatch.setattr(mlflow, 'active_run', lambda: None)

        def _log_text(_text: str, _artifact_file: str) -> None:
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
        monkeypatch.setattr(mlflow, 'active_run', lambda: object())

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
