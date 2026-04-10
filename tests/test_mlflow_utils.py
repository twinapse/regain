"""
Tests for MLflow utility helpers.
"""

from types import SimpleNamespace

import mlflow
import pytest

from regain.constants import RUN_ACC_REF_TEST
from regain.mlflow_utils import build_mlflow_run_columns
from regain.constants import MLFLOW_ARTIFACT_ERROR_FILE
from regain.mlflow_utils import _log_fatal_error_artifact
from regain.mlflow_utils import log_fatal_error_context


def _make_run(*, metrics: dict[str, float]) -> SimpleNamespace:
    """
    Build a minimal MLflow run stub for export tests.

    Args:
        metrics (dict[str, float]): Run metric payload.

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
            params={'seed': '1'},
            metrics=metrics,
        ),
    )


class _FakeHistoryClient:
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


class TestBuildMlflowRunColumns:
    def test_materializes_history_bearing_eval_metrics_with_actual_after_exp_tokens(self) -> None:
        run = _make_run(
            metrics={
                f'{RUN_ACC_REF_TEST}.exp000.base': 0.91,
                f'{RUN_ACC_REF_TEST}.exp001.base': 0.82,
                f'{RUN_ACC_REF_TEST}.exp002.base': 0.73,
                'run.eval.loss.exp.exp000': 1.23,
                'run.eval.forgetting.exp000': 0.22,
                'run.eval.transfer.stream': 0.44,
                'run.eval.acc.final.test.avg.base': 0.77,
            }
        )
        client = _FakeHistoryClient(
            histories={
                f'{RUN_ACC_REF_TEST}.exp000.base': [SimpleNamespace(step=10, value=0.91)],
                f'{RUN_ACC_REF_TEST}.exp001.base': [SimpleNamespace(step=20, value=0.82)],
                f'{RUN_ACC_REF_TEST}.exp002.base': [SimpleNamespace(step=30, value=0.73)],
                'run.eval.loss.exp.exp000': [
                    SimpleNamespace(step=20, value=1.11),
                    SimpleNamespace(step=30, value=1.23),
                ],
                'run.eval.forgetting.exp000': [
                    SimpleNamespace(step=20, value=0.11),
                    SimpleNamespace(step=30, value=0.22),
                ],
                'run.eval.transfer.stream': [
                    SimpleNamespace(step=20, value=0.33),
                    SimpleNamespace(step=30, value=0.44),
                ],
            }
        )

        columns = build_mlflow_run_columns(
            run=run,
            client=client,
        )

        assert columns['run.eval.loss.after_exp001.exp.exp000'] == pytest.approx(1.11)
        assert columns['run.eval.loss.after_exp002.exp.exp000'] == pytest.approx(1.23)
        assert columns['run.eval.forgetting.after_exp001.exp000'] == pytest.approx(0.11)
        assert columns['run.eval.forgetting.after_exp002.exp000'] == pytest.approx(0.22)
        assert columns['run.eval.transfer.after_exp001.stream'] == pytest.approx(0.33)
        assert columns['run.eval.transfer.after_exp002.stream'] == pytest.approx(0.44)
        assert 'run.eval.loss.exp.exp000' not in columns
        assert 'run.eval.forgetting.exp000' not in columns
        assert 'run.eval.transfer.stream' not in columns
        assert columns['run.eval.acc.final.test.avg.base'] == pytest.approx(0.77)

    def test_raises_when_ref_accuracy_history_is_missing(self) -> None:
        run = _make_run(
            metrics={
                f'{RUN_ACC_REF_TEST}.exp000.base': 0.91,
                'run.eval.forgetting.exp000': 0.22,
            }
        )
        client = _FakeHistoryClient(
            histories={
                'run.eval.forgetting.exp000': [SimpleNamespace(step=10, value=0.22)],
            }
        )

        with pytest.raises(ValueError, match='Missing required MLflow metric history'):
            build_mlflow_run_columns(
                run=run,
                client=client,
            )

    def test_raises_when_ref_accuracy_history_is_ambiguous(self) -> None:
        run = _make_run(
            metrics={
                f'{RUN_ACC_REF_TEST}.exp000.base': 0.91,
                'run.eval.forgetting.exp000': 0.22,
            }
        )
        client = _FakeHistoryClient(
            histories={
                f'{RUN_ACC_REF_TEST}.exp000.base': [
                    SimpleNamespace(step=10, value=0.91),
                    SimpleNamespace(step=20, value=0.92),
                ],
                'run.eval.forgetting.exp000': [SimpleNamespace(step=10, value=0.22)],
            }
        )

        with pytest.raises(ValueError, match='exactly one checkpoint step'):
            build_mlflow_run_columns(
                run=run,
                client=client,
            )

    def test_raises_when_history_bearing_metric_history_is_missing(self) -> None:
        run = _make_run(
            metrics={
                f'{RUN_ACC_REF_TEST}.exp000.base': 0.91,
                'run.eval.forgetting.exp000': 0.22,
            }
        )
        client = _FakeHistoryClient(
            histories={
                f'{RUN_ACC_REF_TEST}.exp000.base': [SimpleNamespace(step=10, value=0.91)],
            }
        )

        with pytest.raises(ValueError, match='Missing required MLflow metric history'):
            build_mlflow_run_columns(
                run=run,
                client=client,
            )

    def test_skips_unmapped_bootstrap_step_zero_when_other_steps_are_mapped(self) -> None:
        run = _make_run(
            metrics={
                f'{RUN_ACC_REF_TEST}.exp000.base': 0.91,
                f'{RUN_ACC_REF_TEST}.exp001.base': 0.82,
                'run.eval.forgetting.exp000': 0.22,
            }
        )
        client = _FakeHistoryClient(
            histories={
                f'{RUN_ACC_REF_TEST}.exp000.base': [SimpleNamespace(step=10, value=0.91)],
                f'{RUN_ACC_REF_TEST}.exp001.base': [SimpleNamespace(step=20, value=0.82)],
                'run.eval.forgetting.exp000': [
                    SimpleNamespace(step=0, value=0.05),
                    SimpleNamespace(step=20, value=0.22),
                ],
            }
        )

        columns = build_mlflow_run_columns(
            run=run,
            client=client,
        )

        assert columns['run.eval.forgetting.after_exp001.exp000'] == pytest.approx(0.22)
        assert 'run.eval.forgetting.after_exp000.exp000' not in columns
        assert 'run.eval.forgetting.exp000' not in columns

    def test_skips_unmapped_bootstrap_step_zero_when_it_is_the_only_history_point(self) -> None:
        run = _make_run(
            metrics={
                f'{RUN_ACC_REF_TEST}.exp000.base': 0.91,
                'run.eval.forgetting.exp000': 0.22,
            }
        )
        client = _FakeHistoryClient(
            histories={
                f'{RUN_ACC_REF_TEST}.exp000.base': [SimpleNamespace(step=10, value=0.91)],
                'run.eval.forgetting.exp000': [SimpleNamespace(step=0, value=0.22)],
            }
        )

        columns = build_mlflow_run_columns(
            run=run,
            client=client,
        )

        assert not any(key.startswith('run.eval.forgetting.after_exp') for key in columns)
        assert 'run.eval.forgetting.exp000' not in columns

    def test_skips_unmapped_bootstrap_step_zero_for_stream_forgetting(self) -> None:
        run = _make_run(
            metrics={
                f'{RUN_ACC_REF_TEST}.exp000.base': 0.91,
                f'{RUN_ACC_REF_TEST}.exp001.base': 0.82,
                'run.eval.forgetting.stream': 0.18,
            }
        )
        client = _FakeHistoryClient(
            histories={
                f'{RUN_ACC_REF_TEST}.exp000.base': [SimpleNamespace(step=10, value=0.91)],
                f'{RUN_ACC_REF_TEST}.exp001.base': [SimpleNamespace(step=20, value=0.82)],
                'run.eval.forgetting.stream': [
                    SimpleNamespace(step=0, value=0.0),
                    SimpleNamespace(step=10, value=0.07),
                    SimpleNamespace(step=20, value=0.18),
                ],
            }
        )

        columns = build_mlflow_run_columns(
            run=run,
            client=client,
        )

        assert columns['run.eval.forgetting.after_exp000.stream'] == pytest.approx(0.07)
        assert columns['run.eval.forgetting.after_exp001.stream'] == pytest.approx(0.18)
        assert 'run.eval.forgetting.stream' not in columns

    def test_raises_when_history_bearing_metric_step_is_unmapped(self) -> None:
        run = _make_run(
            metrics={
                f'{RUN_ACC_REF_TEST}.exp000.base': 0.91,
                f'{RUN_ACC_REF_TEST}.exp001.base': 0.82,
                'run.eval.forgetting.exp000': 0.22,
            }
        )
        client = _FakeHistoryClient(
            histories={
                f'{RUN_ACC_REF_TEST}.exp000.base': [SimpleNamespace(step=10, value=0.91)],
                f'{RUN_ACC_REF_TEST}.exp001.base': [SimpleNamespace(step=20, value=0.82)],
                'run.eval.forgetting.exp000': [SimpleNamespace(step=30, value=0.22)],
            }
        )

        with pytest.raises(ValueError, match='no matching checkpoint identity'):
            build_mlflow_run_columns(
                run=run,
                client=client,
            )

    def test_raises_when_history_lookup_fails(self) -> None:
        run = _make_run(
            metrics={
                f'{RUN_ACC_REF_TEST}.exp000.base': 0.91,
                'run.eval.forgetting.exp000': 0.22,
            }
        )
        client = _FakeHistoryClient(
            histories={
                f'{RUN_ACC_REF_TEST}.exp000.base': [SimpleNamespace(step=10, value=0.91)],
            },
            raising_metrics={'run.eval.forgetting.exp000'},
        )

        with pytest.raises(ValueError, match='Failed to fetch required MLflow metric history'):
            build_mlflow_run_columns(
                run=run,
                client=client,
            )
