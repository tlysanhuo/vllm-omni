# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""L1 contract tests for AsyncOmniRunner (RFC #8013 Phase A).

No GPU and no real engine: ``AsyncOmni`` and the cleanup helpers are
monkeypatched, mirroring ``test_runtime_omni_runner.py``.
"""

from __future__ import annotations

import pytest

from tests.helpers.runtime import AsyncOmniParams, AsyncOmniRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeAsyncOmni:
    """Records the kwargs it was built with and its shutdown calls."""

    instances: list[_FakeAsyncOmni] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.shutdown_calls = 0
        self.magic = 7
        _FakeAsyncOmni.instances.append(self)

    def shutdown(self, timeout: float | None = None) -> None:
        self.shutdown_calls += 1


@pytest.fixture(autouse=True)
def _reset_instances():
    _FakeAsyncOmni.instances.clear()
    yield
    _FakeAsyncOmni.instances.clear()


@pytest.fixture
def _tracked_cleanup(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    events: list[str] = []
    monkeypatch.setattr(
        "tests.helpers.runtime.cleanup_test_environment",
        lambda: events.append("env"),
    )
    monkeypatch.setattr(
        "tests.helpers.runtime.reap_leftover_engine_children",
        lambda: events.append("reap"),
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.async_omni.AsyncOmni",
        _FakeAsyncOmni,
    )
    return events


def test_runner_init_rolls_back_on_async_omni_startup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``__exit__`` is skipped when construction raises; rollback must still run."""

    class _BoomAsyncOmni:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("Async orchestrator initialization failed")

    monkeypatch.setattr(
        "vllm_omni.entrypoints.async_omni.AsyncOmni",
        _BoomAsyncOmni,
    )
    cleaned: list[str] = []
    monkeypatch.setattr(
        "tests.helpers.runtime.cleanup_test_environment",
        lambda: cleaned.append("env"),
    )
    monkeypatch.setattr(
        "tests.helpers.runtime.reap_leftover_engine_children",
        lambda: cleaned.append("reap"),
    )

    with pytest.raises(RuntimeError, match="Async orchestrator initialization failed"):
        with AsyncOmniRunner("fake-model"):
            raise AssertionError("context body must not run after a failed constructor")

    assert cleaned.count("reap") == 1
    # Once at the start of ``__init__``, once from the constructor rollback.
    assert cleaned.count("env") == 2


def test_runner_teardown_order_on_clean_exit(_tracked_cleanup: list[str]) -> None:
    with AsyncOmniRunner("fake-model", deploy_config=None, max_num_seqs=1) as runner:
        engine = runner.engine
        assert isinstance(engine, _FakeAsyncOmni)
        assert engine.kwargs["model"] == "fake-model"
        assert engine.kwargs["max_num_seqs"] == 1

    assert engine.shutdown_calls == 1
    assert _tracked_cleanup == ["env", "reap", "env"]


def test_runner_teardown_runs_on_body_exception(_tracked_cleanup: list[str]) -> None:
    with pytest.raises(ValueError, match="boom"):
        with AsyncOmniRunner("fake-model"):
            raise ValueError("boom")

    assert _tracked_cleanup == ["env", "reap", "env"]
    assert _FakeAsyncOmni.instances[0].shutdown_calls == 1


def test_runner_delegates_attributes_to_engine(_tracked_cleanup: list[str]) -> None:
    with AsyncOmniRunner("fake-model") as runner:
        assert runner.magic == 7
        assert runner.engine is _FakeAsyncOmni.instances[0]
        with pytest.raises(AttributeError):
            _ = runner.no_such_attribute


def test_async_omni_params_defaults() -> None:
    params = AsyncOmniParams(model="tiny/Qwen-Image")
    assert params.model == "tiny/Qwen-Image"
    assert params.deploy_config is None
    assert params.extra_omni_kwargs is None


def test_reap_is_scoped_to_the_current_process_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    """A matching process outside this process tree must be left alone.

    pytest-xdist loadgroup workers own sibling engines on a shared host;
    the reap helper must only ever see this process's own children.
    """
    import os

    from tests.helpers import clean as clean_mod

    terminated: list[int] = []

    class FakeProc:
        def __init__(self, pid: int, cmdline: list[str]) -> None:
            self.pid = pid
            self._cmdline = cmdline

        def cmdline(self) -> list[str]:
            return self._cmdline

        def name(self) -> str:
            return "python"

        def terminate(self) -> None:
            terminated.append(self.pid)

    child = FakeProc(101, ["python", "-m", "vllm-omni::engine"])
    sibling = FakeProc(202, ["python", "-m", "vllm-omni::engine"])

    class FakeRoot:
        def children(self, recursive: bool = True) -> list[FakeProc]:
            return [child]

    class FakePsutil:
        NoSuchProcess = clean_mod.psutil.NoSuchProcess
        AccessDenied = clean_mod.psutil.AccessDenied

        @staticmethod
        def Process(pid: int) -> FakeRoot:
            assert pid == os.getpid()
            return FakeRoot()

        @staticmethod
        def wait_procs(procs: list, timeout: float | None = None) -> tuple[list, list]:
            return list(procs), []

    monkeypatch.setattr(clean_mod, "psutil", FakePsutil)
    clean_mod.reap_leftover_engine_children()
    assert terminated == [101]
    assert not hasattr(sibling, "_terminated")
