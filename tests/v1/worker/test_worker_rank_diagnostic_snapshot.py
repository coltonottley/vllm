# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Focused CPU/mocked tests for the read-only per-rank diagnostic snapshot.

``Worker.get_rank_diagnostic_snapshot`` is the named collective-RPC-safe
worker seam (invoke on every rank via
``executor.collective_rpc("get_rank_diagnostic_snapshot")``). These tests
prove, on CPU with mocked CUDA, that the snapshot:

* only READS already-instantiated state and never mutates CUDA/model state;
* is JSON-serializable;
* stays bounded/redacted regardless of descriptor count, address list
  length, or error message size;
* reports lazy workspace slots only when already instantiated (never
  instantiating them itself).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

import vllm.envs as envs
from vllm.config import CUDAGraphMode
from vllm.config.compilation import CompilationMode
from vllm.forward_context import BatchDescriptor
from vllm.v1.worker.gpu_worker import Worker

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_worker(
    *,
    device: torch.device | None = None,
    rank: int = 0,
    local_rank: int = 0,
    model_runner: object | None = None,
    compilation_config: object | None = None,
) -> Worker:
    """A real Worker instance without the heavy ``__init__`` side effects.

    ``Worker.__new__`` skips the constructor (thread pool, sentinel, etc.);
    the snapshot method only touches the attributes set here, so this is the
    minimal canonical surface it reads.
    """
    worker = Worker.__new__(Worker)
    worker.rank = rank
    worker.local_rank = local_rank
    worker.is_driver_worker = rank == 0
    worker.device = torch.device("cpu") if device is None else device
    worker.model_runner = model_runner
    if compilation_config is None:
        compilation_config = SimpleNamespace(
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            backend="inductor",
        )
    worker.compilation_config = compilation_config
    return worker


def _default_runner() -> SimpleNamespace:
    """A V2-style model runner with an instantiated cudagraph manager."""
    cg_manager = SimpleNamespace(
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        use_breakable_cg=False,
        breakable_cg_runner=None,
    )
    return SimpleNamespace(cudagraph_manager=cg_manager, model=object())


def _enable_breakable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    envs.disable_envs_cache()


@pytest.fixture(autouse=True)
def _reset_workspace_manager():
    """Isolate workspace-manager module state between tests."""
    from vllm.v1.worker.workspace import reset_workspace_manager

    reset_workspace_manager()
    yield
    reset_workspace_manager()


# ---------------------------------------------------------------------------
# Snapshot-only semantics
# ---------------------------------------------------------------------------


def test_snapshot_is_json_serializable_and_reads_rank_state():
    worker = _make_worker(
        device=torch.device("cpu"),
        rank=3,
        local_rank=1,
        model_runner=_default_runner(),
    )
    snapshot = worker.get_rank_diagnostic_snapshot()
    # Round-trips through JSON: proves the return value is bounded to
    # JSON-serializable types (int/float/str/bool/list/dict/None).
    loaded = json.loads(json.dumps(snapshot))
    assert loaded["rank"] == 3
    assert loaded["local_rank"] == 1
    assert loaded["driver_worker"] is False
    assert loaded["device"] == "cpu"
    assert loaded["schema_version"] == Worker._SNAPSHOT_SCHEMA_VERSION
    # PID is process-local and always present.
    assert isinstance(loaded["pid"], int)


def test_snapshot_with_driver_worker_and_cuda_device_string():
    worker = _make_worker(
        device=torch.device("cuda:0"),
        rank=0,
        local_rank=0,
        model_runner=None,
    )
    with (
        patch("torch.cuda.memory_allocated", return_value=1),
        patch("torch.cuda.memory_reserved", return_value=2),
        patch("torch.cuda.max_memory_allocated", return_value=3),
        patch("torch.cuda.max_memory_reserved", return_value=4),
        patch("torch.cuda.mem_get_info", return_value=(5, 100)),
        patch("torch.cuda.memory_stats", return_value={}),
    ):
        snapshot = worker.get_rank_diagnostic_snapshot()
    assert snapshot["driver_worker"] is True
    assert snapshot["device"] == "cuda:0"
    # model_runner None -> no breakable, compilation from config only.
    assert snapshot["compilation"]["compilation_mode"] == "VLLM_COMPILE"
    assert snapshot["compilation"]["cudagraph_mode"] == "FULL_AND_PIECEWISE"
    assert snapshot["compilation"]["breakable_active"] is False


def test_cuda_memory_block_reports_exact_values_and_filters_stats():
    worker = _make_worker(device=torch.device("cuda:0"))
    memory_stats = {
        "allocated_bytes.all.current": 1000,  # NOT selected: not inactive-split/retry
        "inactive_split_bytes.all.current": 200,
        "inactive_split_bytes.small_pool.peak": 50,
        "inactive_split.all.current": 3,
        "num_alloc_retries": 7,
        "num_ooms": 1,
        "num_oom_rejections": 0,
        "num_device_alloc": 11,
        "num_device_free": 9,
        "oversize_allocations.current": 2,
        "max_split_size": 1024,
        "segment.all.current": 5,  # NOT selected
        "num_sync_all_streams": 4,  # NOT selected (sync counter, not requested)
    }
    with (
        patch("torch.cuda.memory_allocated", return_value=123),
        patch("torch.cuda.memory_reserved", return_value=456),
        patch("torch.cuda.max_memory_allocated", return_value=789),
        patch("torch.cuda.max_memory_reserved", return_value=999),
        patch("torch.cuda.mem_get_info", return_value=(1000, 8192)),
        patch("torch.cuda.memory_stats", return_value=memory_stats),
    ):
        cuda = worker.get_rank_diagnostic_snapshot()["cuda"]

    assert cuda["memory_allocated_bytes"] == 123
    assert cuda["memory_reserved_bytes"] == 456
    assert cuda["max_memory_allocated_bytes"] == 789
    assert cuda["max_memory_reserved_bytes"] == 999
    assert cuda["mem_get_info"] == {"free_bytes": 1000, "total_bytes": 8192}

    selected = cuda["memory_stats"]
    assert selected == {
        "inactive_split_bytes.all.current": 200,
        "inactive_split_bytes.small_pool.peak": 50,
        "inactive_split.all.current": 3,
        "num_alloc_retries": 7,
        "num_ooms": 1,
        "num_oom_rejections": 0,
        "num_device_alloc": 11,
        "num_device_free": 9,
        "oversize_allocations.current": 2,
        "max_split_size": 1024,
    }
    # Irrelevant / non-inactive-split keys are excluded (bounded output).
    assert "allocated_bytes.all.current" not in selected
    assert "segment.all.current" not in selected
    assert "num_sync_all_streams" not in selected
    json.loads(json.dumps(cuda))


def test_cuda_memory_block_errors_are_bounded_and_redacted():
    worker = _make_worker(device=torch.device("cuda:0"))
    huge_error = "boom " * 1000  # 5000 chars
    with (
        patch(
            "torch.cuda.memory_allocated",
            side_effect=RuntimeError(huge_error),
        ),
        patch("torch.cuda.memory_stats", return_value={}),
    ):
        cuda = worker.get_rank_diagnostic_snapshot()["cuda"]
    assert "error" in cuda
    assert len(cuda["error"]) <= Worker._SNAPSHOT_MAX_STRING + 64
    assert "boom" in cuda["error"]
    json.loads(json.dumps(cuda))


# ---------------------------------------------------------------------------
# Absence of mutating CUDA / model calls
# ---------------------------------------------------------------------------


def test_no_mutating_cuda_or_model_calls_cuda_path():
    worker = _make_worker(device=torch.device("cuda:0"))

    empty_cache = Mock()
    reset_peaks = Mock()
    synchronize = Mock()
    memory_snapshot = Mock()
    with (
        patch("torch.cuda.empty_cache", empty_cache),
        patch("torch.cuda.reset_peak_memory_stats", reset_peaks),
        patch("torch.cuda.synchronize", synchronize),
        patch("torch.cuda.memory_snapshot", memory_snapshot),
        patch("torch.cuda.memory_allocated", return_value=1),
        patch("torch.cuda.memory_reserved", return_value=2),
        patch("torch.cuda.max_memory_allocated", return_value=3),
        patch("torch.cuda.max_memory_reserved", return_value=4),
        patch("torch.cuda.mem_get_info", return_value=(1, 2)),
        patch("torch.cuda.memory_stats", return_value={}),
        patch.object(Worker, "determine_available_memory") as determine,
    ):
        worker.get_rank_diagnostic_snapshot()

    empty_cache.assert_not_called()
    reset_peaks.assert_not_called()
    synchronize.assert_not_called()
    memory_snapshot.assert_not_called()
    determine.assert_not_called()


def test_no_mutating_cuda_or_model_calls_breakable_path(
    monkeypatch: pytest.MonkeyPatch,
):
    """The breakable path reads stored wrapper state and never clears or
    captures graphs."""
    from vllm.compilation.breakable_cudagraph import (
        BreakableCUDAGraphCapture,
        BreakableCUDAGraphWrapper,
    )
    from vllm.v1.worker.gpu_worker import Worker as GpuWorker

    _enable_breakable(monkeypatch)

    wrapper = BreakableCUDAGraphWrapper.__new__(BreakableCUDAGraphWrapper)
    capture = BreakableCUDAGraphCapture.__new__(BreakableCUDAGraphCapture)
    capture._num_graphs = 2
    capture._num_eager_breaks = 1
    wrapper.entries = {
        BatchDescriptor(num_tokens=8, num_reqs=1): SimpleNamespace(
            batch_descriptor=BatchDescriptor(num_tokens=8, num_reqs=1),
            capture=capture,
            output=None,
            input_addresses=[0x1000, 0x2000],
        )
    }
    wrapper.graph_pool = 42

    runner = SimpleNamespace(
        cudagraph_manager=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            use_breakable_cg=True,
            breakable_cg_runner=wrapper,
        ),
        model=object(),
    )
    worker = _make_worker(device=torch.device("cuda:0"), model_runner=runner)

    empty_cache = Mock()
    reset_peaks = Mock()
    synchronize = Mock()
    memory_snapshot = Mock()
    clear_all_graphs = Mock()
    with (
        patch("torch.cuda.empty_cache", empty_cache),
        patch("torch.cuda.reset_peak_memory_stats", reset_peaks),
        patch("torch.cuda.synchronize", synchronize),
        patch("torch.cuda.memory_snapshot", memory_snapshot),
        patch("torch.cuda.memory_allocated", return_value=1),
        patch("torch.cuda.memory_reserved", return_value=2),
        patch("torch.cuda.max_memory_allocated", return_value=3),
        patch("torch.cuda.max_memory_reserved", return_value=4),
        patch("torch.cuda.mem_get_info", return_value=(1, 2)),
        patch("torch.cuda.memory_stats", return_value={}),
        patch.object(BreakableCUDAGraphWrapper, "clear_all_graphs", clear_all_graphs),
        patch.object(BreakableCUDAGraphWrapper, "clear_graphs") as clear_graphs,
        patch.object(GpuWorker, "determine_available_memory") as determine,
    ):
        snapshot = worker.get_rank_diagnostic_snapshot()

    empty_cache.assert_not_called()
    reset_peaks.assert_not_called()
    synchronize.assert_not_called()
    memory_snapshot.assert_not_called()
    clear_all_graphs.assert_not_called()
    clear_graphs.assert_not_called()
    determine.assert_not_called()

    breakable = snapshot["compilation"]["breakable"]
    assert breakable["entry_count"] == 1
    assert breakable["entries"][0]["graph_segments"] == 2
    assert breakable["entries"][0]["eager_breaks"] == 1
    assert breakable["entries"][0]["input_addresses"] == [0x1000, 0x2000]
    assert breakable["graph_pool_identity"] == id(42)
    json.loads(json.dumps(snapshot))


# ---------------------------------------------------------------------------
# Breakable mode: canonical owner reads + bounded serialization/redaction
# ---------------------------------------------------------------------------


def _build_wrapper_with_entries(num_entries: int, addresses_per_entry: int) -> object:
    from vllm.compilation.breakable_cudagraph import (
        BreakableCUDAGraphCapture,
        BreakableCUDAGraphWrapper,
    )

    wrapper = BreakableCUDAGraphWrapper.__new__(BreakableCUDAGraphWrapper)
    entries = {}
    for i in range(num_entries):
        desc = BatchDescriptor(num_tokens=8 + i, num_reqs=1)
        capture = BreakableCUDAGraphCapture.__new__(BreakableCUDAGraphCapture)
        capture._num_graphs = i + 1
        capture._num_eager_breaks = i
        entries[desc] = SimpleNamespace(
            batch_descriptor=desc,
            capture=capture,
            output=None,
            input_addresses=list(range(addresses_per_entry)),
        )
    wrapper.entries = entries
    wrapper.graph_pool = 1234
    return wrapper


def test_breakable_active_reports_entries_segments_and_addresses(
    monkeypatch: pytest.MonkeyPatch,
):
    _enable_breakable(monkeypatch)
    wrapper = _build_wrapper_with_entries(3, 5)
    runner = SimpleNamespace(
        cudagraph_manager=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            use_breakable_cg=True,
            breakable_cg_runner=wrapper,
        ),
        model=object(),
    )
    worker = _make_worker(device=torch.device("cpu"), model_runner=runner)

    compilation = worker.get_rank_diagnostic_snapshot()["compilation"]
    assert compilation["breakable_enabled"] is True
    assert compilation["breakable_active"] is True
    breakable = compilation["breakable"]
    assert breakable["entry_count"] == 3
    assert breakable["entries_truncated"] is False
    assert breakable["total_graph_segments"] == 1 + 2 + 3
    assert breakable["total_eager_breaks"] == 0 + 1 + 2
    assert breakable["total_input_addresses"] == 15
    assert breakable["graph_pool_identity"] == id(1234)
    # Per-entry summaries are exact and bounded.
    assert [e["input_address_count"] for e in breakable["entries"]] == [5, 5, 5]
    assert all(isinstance(e["descriptor"], str) for e in breakable["entries"])
    json.loads(json.dumps(breakable))


def test_breakable_bounded_serialization_redacts_entries_and_addresses(
    monkeypatch: pytest.MonkeyPatch,
):
    _enable_breakable(monkeypatch)
    wrapper = _build_wrapper_with_entries(200, 1000)
    runner = SimpleNamespace(
        cudagraph_manager=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            use_breakable_cg=True,
            breakable_cg_runner=wrapper,
        ),
        model=object(),
    )
    worker = _make_worker(device=torch.device("cpu"), model_runner=runner)

    breakable = worker.get_rank_diagnostic_snapshot()["compilation"]["breakable"]
    # Exact counts are kept; the serialized detail is bounded.
    assert breakable["entry_count"] == 200
    assert breakable["entries_reported"] == Worker._SNAPSHOT_MAX_ENTRIES
    assert breakable["entries_truncated"] is True
    assert len(breakable["entries"]) == Worker._SNAPSHOT_MAX_ENTRIES
    # Address detail per entry is capped; exact totals preserved.
    assert all(
        len(e["input_addresses"]) == Worker._SNAPSHOT_MAX_ADDRESSES
        for e in breakable["entries"]
    )
    assert breakable["total_input_addresses"] == 200 * 1000
    # Every descriptor string is bounded.
    for entry in breakable["entries"]:
        assert len(entry["descriptor"]) <= Worker._SNAPSHOT_MAX_STRING + 64
    json.loads(json.dumps(breakable))


def test_breakable_descriptor_repr_is_truncated(
    monkeypatch: pytest.MonkeyPatch,
):
    _enable_breakable(monkeypatch)
    from vllm.compilation.breakable_cudagraph import (
        BreakableCUDAGraphCapture,
        BreakableCUDAGraphWrapper,
    )

    class _HugeRepr:
        def __hash__(self) -> int:
            return 7

        def __repr__(self) -> str:
            return "x" * 1000

    wrapper = BreakableCUDAGraphWrapper.__new__(BreakableCUDAGraphWrapper)
    capture = BreakableCUDAGraphCapture.__new__(BreakableCUDAGraphCapture)
    capture._num_graphs = 0
    capture._num_eager_breaks = 0
    huge = _HugeRepr()
    wrapper.entries = {
        huge: SimpleNamespace(
            batch_descriptor=huge, capture=capture, output=None, input_addresses=None
        )
    }
    wrapper.graph_pool = None
    runner = SimpleNamespace(
        cudagraph_manager=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            use_breakable_cg=True,
            breakable_cg_runner=wrapper,
        ),
        model=object(),
    )
    worker = _make_worker(device=torch.device("cpu"), model_runner=runner)

    breakable = worker.get_rank_diagnostic_snapshot()["compilation"]["breakable"]
    descriptor = breakable["entries"][0]["descriptor"]
    assert len(descriptor) <= Worker._SNAPSHOT_MAX_STRING + 64
    assert "truncated" in descriptor
    json.loads(json.dumps(breakable))


def test_breakable_inactive_when_not_enabled_or_no_wrapper(
    monkeypatch: pytest.MonkeyPatch,
):
    # Breakable disabled: even a wrapped model reports inactive.
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper

    wrapper = BreakableCUDAGraphWrapper.__new__(BreakableCUDAGraphWrapper)
    wrapper.entries = {}
    wrapper.graph_pool = None
    runner = SimpleNamespace(
        cudagraph_manager=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            use_breakable_cg=False,
            breakable_cg_runner=None,
        ),
        model=wrapper,
    )
    worker = _make_worker(device=torch.device("cpu"), model_runner=runner)
    compilation = worker.get_rank_diagnostic_snapshot()["compilation"]
    assert compilation["breakable_enabled"] is False
    assert compilation["breakable_active"] is False
    assert compilation["breakable"] is None

    # Enabled but no wrapper instantiated: reports inactive, no breakable block.
    _enable_breakable(monkeypatch)
    runner2 = SimpleNamespace(
        cudagraph_manager=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            use_breakable_cg=True,
            breakable_cg_runner=None,
        ),
        model=object(),
    )
    worker2 = _make_worker(device=torch.device("cpu"), model_runner=runner2)
    compilation2 = worker2.get_rank_diagnostic_snapshot()["compilation"]
    assert compilation2["breakable_enabled"] is True
    assert compilation2["breakable_active"] is False
    assert compilation2["breakable"] is None


def test_v1_style_model_wrapped_breakable_is_detected(
    monkeypatch: pytest.MonkeyPatch,
):
    """The V1 model runner wraps the model directly; the snapshot must find
    the wrapper through that canonical owner path."""
    from vllm.compilation.breakable_cudagraph import (
        BreakableCUDAGraphCapture,
        BreakableCUDAGraphWrapper,
    )

    _enable_breakable(monkeypatch)
    wrapper = BreakableCUDAGraphWrapper.__new__(BreakableCUDAGraphWrapper)
    capture = BreakableCUDAGraphCapture.__new__(BreakableCUDAGraphCapture)
    capture._num_graphs = 1
    capture._num_eager_breaks = 1
    wrapper.entries = {
        BatchDescriptor(num_tokens=4, num_reqs=1): SimpleNamespace(
            batch_descriptor=BatchDescriptor(num_tokens=4, num_reqs=1),
            capture=capture,
            output=None,
            input_addresses=[0x1111],
        )
    }
    wrapper.graph_pool = None
    # V1 runner: no cudagraph_manager; the model IS the wrapper.
    runner = SimpleNamespace(cudagraph_manager=None, model=wrapper)
    worker = _make_worker(device=torch.device("cpu"), model_runner=runner)

    compilation = worker.get_rank_diagnostic_snapshot()["compilation"]
    assert compilation["breakable_active"] is True
    assert compilation["breakable"]["entry_count"] == 1
    assert compilation["breakable"]["entries"][0]["input_addresses"] == [0x1111]


# ---------------------------------------------------------------------------
# Lazy workspace: only already-instantiated slots reported
# ---------------------------------------------------------------------------


def test_workspace_reports_only_instantiated_slots():
    from vllm.v1.worker.workspace import (
        current_workspace_manager,
        init_workspace_manager,
    )

    init_workspace_manager(torch.device("cpu"), num_ubatches=2)
    manager = current_workspace_manager()
    # Instantiate slot 0 only (real lazy allocation path); slot 1 stays None.
    manager.get_simultaneous(((16,), torch.float32))
    assert manager._current_workspaces[0] is not None
    assert manager._current_workspaces[1] is None
    expected_bytes = (
        manager._current_workspaces[0].numel()
        * manager._current_workspaces[0].element_size()
    )
    expected_ptr = manager._current_workspaces[0].data_ptr()

    worker = _make_worker(device=torch.device("cpu"))
    workspace = worker.get_rank_diagnostic_snapshot()["workspace"]

    assert workspace["initialized"] is True
    assert workspace["locked"] is False
    assert workspace["num_ubatch_slots"] == 2
    assert workspace["instantiated_count"] == 1
    assert workspace["instantiated"][0]["ubatch_id"] == 0
    assert workspace["instantiated"][0]["bytes"] == expected_bytes
    assert workspace["instantiated"][0]["data_ptr"] == expected_ptr
    json.loads(json.dumps(workspace))

    # The snapshot did NOT instantiate slot 1.
    assert manager._current_workspaces[1] is None


def test_workspace_none_when_manager_not_initialized():
    worker = _make_worker(device=torch.device("cpu"))
    assert worker.get_rank_diagnostic_snapshot()["workspace"] is None


def test_workspace_reports_locked_state():
    from vllm.v1.worker.workspace import (
        init_workspace_manager,
        lock_workspace,
    )

    init_workspace_manager(torch.device("cpu"), num_ubatches=1)
    lock_workspace()
    worker = _make_worker(device=torch.device("cpu"))
    workspace = worker.get_rank_diagnostic_snapshot()["workspace"]
    assert workspace["locked"] is True
    assert workspace["instantiated_count"] == 0


# ---------------------------------------------------------------------------
# Bounded-repr helper
# ---------------------------------------------------------------------------


def test_bounded_repr_truncates_long_values_and_keeps_short_values():
    short = Worker._snapshot_bounded_repr(12345)
    assert short == "12345"
    long_value = Worker._snapshot_bounded_repr("y" * 1000, max_len=100)
    assert len(long_value) <= 100 + 64
    assert "truncated" in long_value
