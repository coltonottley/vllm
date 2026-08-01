# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hermetic same-handle cudaHostRegister/drain/unregister tests.

No live CUDA calls.  CudaRTLibrary instances via object.__new__ + fake
funcs; caller sites via monkeypatched fake classes.
"""

import pytest

from vllm.distributed.device_communicators.cuda_wrapper import (
    CudaRTLibrary,
    cudaError_t,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.ec_shared_region import (
    ECSharedRegion,
)
from vllm.v1.kv_offload.cpu import gpu_worker
from vllm.v1.kv_offload.cpu import shared_offload_region as sor
from vllm.v1.kv_offload.cpu.gpu_worker import pin_mmap_region
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

EC_MOD = "vllm.distributed.ec_transfer.ec_connector.cpu.ec_shared_region"
GP_MOD = "vllm.v1.kv_offload.cpu.gpu_worker"


def _lib(funcs):
    lib = object.__new__(CudaRTLibrary)
    lib.funcs = funcs
    lib.CUDART_CHECK = CudaRTLibrary.CUDART_CHECK.__get__(lib, CudaRTLibrary)
    return lib


class _Region:
    rank = 0
    base_ptr = 1234
    total_size_bytes = 65536
    is_pinned = False
    _cudart_lib = None


def _kv_obj():
    obj = object.__new__(SharedOffloadRegion)
    for a in (
        "is_pinned",
        "_cudart_lib",
        "_base",
        "mmap_obj",
        "fd",
        "_creator",
        "_views",
        "mmap_path",
    ):
        setattr(obj, a, None)
    obj.rank = 0
    return obj


def _ec_stub():
    s = ECSharedRegion.__new__(ECSharedRegion)
    for a in (
        "_blocks_ptr",
        "_blocks_nbytes",
        "_is_pinned",
        "_cudart_lib",
        "_is_creator",
        "_fd",
        "_mmap_obj",
        "_mmap_path",
    ):
        setattr(s, a, None)
    s._blocks_ptr = 1234
    s._blocks_nbytes = 65536
    s._is_pinned = False
    s._cudart_lib = None
    s._is_creator = False
    s._fd = None
    s._mmap_obj = None
    s._mmap_path = "/dev/shm/_t"
    return s


class _StatefulFake:
    """Stateful fake runtime with thread-local pending-error semantics.

    A failed ``cudaHostRegister`` sets a trailing pending CUDA error that the
    driver would otherwise surface on the *next* unrelated runtime API call (as
    ``cudaErrorInvalidValue``).  ``drain_pending_error`` consumes and clears
    that state.  ``unrelated_op`` simulates an unrelated runtime API call and
    returns ``0`` only if no stale pending error remains.
    """

    def __init__(self):
        self.pending_error = 0
        self.calls = []

    def cudaHostRegister(self, ptr, size, flags):
        self.calls.append("reg")
        self.pending_error = 1  # failed register leaves a pending CUDA error
        return 1

    def drain_pending_error(self):
        self.calls.append("drain")
        err = self.pending_error
        self.pending_error = 0  # consume + clear the pending error
        return err

    def unrelated_op(self):
        # Unrelated runtime API call: surfaces the lurking pending error as a
        # failure (cudaErrorInvalidValue) unless it was drained.
        self.calls.append("unrelated")
        return self.pending_error


# -- CudaRTLibrary wrapper + drain -----------------------------------------


class TestWrappers:
    def test_register_no_raise(self):
        r = _lib({"cudaHostRegister": lambda p, s, f: 1}).cudaHostRegister(0, 0, 0)
        assert r == 1

    def test_unregister_no_raise(self):
        r = _lib({"cudaHostUnregister": lambda p: 0}).cudaHostUnregister(0)
        assert r == 0

    def test_drain_returns_int(self):
        r = _lib({"cudaGetLastError": lambda: 7}).drain_pending_error()
        assert r == 7

    def test_drain_zero_after_success(self):
        r = _lib({"cudaGetLastError": lambda: 0}).drain_pending_error()
        assert r == 0


# -- CUDART_CHECK ----------------------------------------------------------


class TestCUDARTCheck:
    def test_nonzero_drains_then_raises(self):
        seen = []
        lib = _lib(
            {
                "cudaGetErrorString": lambda e: b"err",
                "cudaGetLastError": lambda: (seen.append(1), 99)[1],
            }
        )
        with pytest.raises(RuntimeError, match="CUDART error: err"):
            lib.CUDART_CHECK(1)
        assert seen == [1]

    def test_preserves_original_error(self):
        lib = _lib(
            {
                "cudaGetErrorString": lambda e: f"code_{e}".encode(),
                "cudaGetLastError": lambda: 99,
            }
        )
        with pytest.raises(RuntimeError) as exc:
            lib.CUDART_CHECK(7)
        assert "code_7" in str(exc.value)

    def test_zero_noop(self):
        _lib({}).CUDART_CHECK(0)


# -- pin_mmap_region (CPU-KV) ----------------------------------------------


class TestGPUWorkerPinMmap:
    def test_failure_drains_unpinned(self, monkeypatch):
        events = []

        class F:
            def cudaHostRegister(self, p, s, f):
                events.append("reg")
                return 1

            def drain_pending_error(self):
                events.append("drain")
                return 1

        monkeypatch.setattr(gpu_worker.current_platform, "is_cuda_alike", lambda: True)
        monkeypatch.setattr(f"{GP_MOD}.CudaRTLibrary", lambda: F())
        r = _Region()
        pin_mmap_region(r)
        assert events == ["reg", "drain"]
        assert not r.is_pinned and r._cudart_lib is None

    def test_success_pins_stores(self, monkeypatch):
        fake = type(
            "Fake",
            (),
            {
                "cudaHostRegister": lambda s, p, sz, f: 0,
                "drain_pending_error": lambda s: 0,
            },
        )()
        monkeypatch.setattr(gpu_worker.current_platform, "is_cuda_alike", lambda: True)
        monkeypatch.setattr(f"{GP_MOD}.CudaRTLibrary", lambda: fake)
        r = _Region()
        pin_mmap_region(r)
        assert r.is_pinned and r._cudart_lib is fake

    def test_non_cuda_skips(self, monkeypatch):
        monkeypatch.setattr(gpu_worker.current_platform, "is_cuda_alike", lambda: False)
        r = _Region()
        pin_mmap_region(r)
        assert not r.is_pinned

    def test_stateful_failure_drain_clears_pending_error(self, monkeypatch):
        """Stateful: a failed register sets a pending CUDA error, the
        production failure path drains it, and an immediately subsequent unrelated
        fake CUDA operation succeeds.  Proves the stale error cannot poison the
        next runtime call — not merely that ``drain`` appears in an event list."""
        fake = _StatefulFake()
        monkeypatch.setattr(gpu_worker.current_platform, "is_cuda_alike", lambda: True)
        monkeypatch.setattr(f"{GP_MOD}.CudaRTLibrary", lambda: fake)
        r = _Region()
        pin_mmap_region(r)
        # The failed cudaHostRegister left a pending runtime error...
        assert fake.calls.count("reg") == 1
        # ...and the production failure path cleared it via drain.
        assert fake.pending_error == 0
        # Immediately subsequent unrelated fake CUDA operation must succeed.
        assert fake.unrelated_op() == 0
        assert not r.is_pinned and r._cudart_lib is None


# -- SharedOffloadRegion.cleanup (CPU-KV) ----------------------------------


class TestKVOffloadCleanup:
    def _setup(self, is_pinned=True, handle=None):
        obj = _kv_obj()
        obj.is_pinned = is_pinned
        obj._cudart_lib = handle
        obj._base = type("B", (), {"data_ptr": lambda s: 42})() if is_pinned else None
        return obj

    def test_cleanup_unregisters_through_stored_handle(self, monkeypatch):
        monkeypatch.setattr(sor.current_platform, "is_cuda_alike", lambda: True)
        events = []

        class F:
            def cudaHostUnregister(self, p):
                events.append("unreg")
                return 0

        obj = self._setup(handle=F())
        obj.cleanup()
        assert events == ["unreg"] and not obj.is_pinned and obj._cudart_lib is None

    def test_cleanup_idempotent(self, monkeypatch):
        monkeypatch.setattr(sor.current_platform, "is_cuda_alike", lambda: True)
        n = [0]

        class F:
            def cudaHostUnregister(self, p):
                n[0] += 1
                return 0

        obj = self._setup(handle=F())
        obj.cleanup()
        obj.cleanup()
        assert n[0] == 1

    def test_cleanup_missing_handle_no_crash(self, monkeypatch):
        monkeypatch.setattr(sor.current_platform, "is_cuda_alike", lambda: True)
        obj = self._setup(handle=None)
        obj.cleanup()  # no exception


# -- ECSharedRegion.pin_memory + cleanup -----------------------------------


class TestECRegion:
    def test_failure_drains_and_no_unregister(self, monkeypatch):
        events = []

        class F:
            def cudaHostRegister(self, p, s, f):
                events.append("reg")
                return 1

            def drain_pending_error(self):
                events.append("drain")
                return 1

            def cudaHostUnregister(self, p):
                events.append("unreg")
                return 0

        monkeypatch.setattr(f"{EC_MOD}.CudaRTLibrary", lambda: F())
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        s = _ec_stub()
        s.pin_memory()
        assert events == ["reg", "drain"] and not s._is_pinned and s._cudart_lib is None
        events.clear()
        s.cleanup()
        assert "unreg" not in events

    def test_success_stores_handle(self, monkeypatch):
        fake = type(
            "Fake",
            (),
            {
                "cudaHostRegister": lambda s, p, sz, f: 0,
                "drain_pending_error": lambda s: 0,
                "cudaHostUnregister": lambda s, p: 0,
            },
        )()
        monkeypatch.setattr(f"{EC_MOD}.CudaRTLibrary", lambda: fake)
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        s = _ec_stub()
        s.pin_memory()
        assert s._is_pinned and s._cudart_lib is fake

    def test_register_then_unregister_same_instance(self, monkeypatch):
        events = []

        class F:
            def cudaHostRegister(self, p, s, f):
                events.append("reg")
                return 0

            def drain_pending_error(self):
                events.append("drain")
                return 0

            def cudaHostUnregister(self, p):
                events.append("unreg")
                return 0

        monkeypatch.setattr(f"{EC_MOD}.CudaRTLibrary", lambda: F())
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        s = _ec_stub()
        s.pin_memory()
        events.clear()
        s.cleanup()
        assert events == ["unreg"] and not s._is_pinned and s._cudart_lib is None

    def test_cleanup_idempotent(self, monkeypatch):
        n = [0]

        class F:
            def cudaHostRegister(self, p, s, f):
                return 0

            def drain_pending_error(self):
                return 0

            def cudaHostUnregister(self, p):
                n[0] += 1
                return 0

        monkeypatch.setattr(f"{EC_MOD}.CudaRTLibrary", lambda: F())
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        s = _ec_stub()
        s.pin_memory()
        s.cleanup()
        s.cleanup()
        assert n[0] == 1

    def test_no_cuda_skips(self, monkeypatch):
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        s = _ec_stub()
        s.pin_memory()
        assert not s._is_pinned

    def test_stateful_failure_drain_clears_pending_error(self, monkeypatch):
        """Stateful: a failed register sets a pending CUDA error, the
        production failure path drains it, and an immediately subsequent unrelated
        fake CUDA operation succeeds.  Proves the stale error cannot poison the
        next runtime call — not merely that ``drain`` appears in an event list."""
        fake = _StatefulFake()
        monkeypatch.setattr(f"{EC_MOD}.CudaRTLibrary", lambda: fake)
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        s = _ec_stub()
        s.pin_memory()
        # The failed cudaHostRegister left a pending runtime error...
        assert fake.calls.count("reg") == 1
        # ...and the production failure path cleared it via drain.
        assert fake.pending_error == 0
        # Immediately subsequent unrelated fake CUDA operation must succeed.
        assert fake.unrelated_op() == 0
        assert not s._is_pinned and s._cudart_lib is None


# -- ROCm mapping + argtypes ----------------------------------------------

ROCM_PAIRS = [
    ("cudaHostRegister", "hipHostRegister"),
    ("cudaHostUnregister", "hipHostUnregister"),
    ("cudaGetLastError", "hipGetLastError"),
]


class TestStruct:
    @pytest.mark.parametrize("cuda_n,hip_n", ROCM_PAIRS)
    def test_roc_mapping(self, cuda_n, hip_n):
        assert CudaRTLibrary.cuda_to_hip_mapping[cuda_n] == hip_n

    def test_both_exported(self):
        names = {f.name for f in CudaRTLibrary.exported_functions}
        assert "cudaHostRegister" in names and "cudaHostUnregister" in names

    def test_register_argtypes(self):
        for f in CudaRTLibrary.exported_functions:
            if f.name == "cudaHostRegister":
                assert f.restype == cudaError_t and len(f.argtypes) == 3
                return
        pytest.fail("cudaHostRegister missing")

    def test_unregister_argtypes(self):
        for f in CudaRTLibrary.exported_functions:
            if f.name == "cudaHostUnregister":
                assert f.restype == cudaError_t and len(f.argtypes) == 1
                return
        pytest.fail("cudaHostUnregister missing")
