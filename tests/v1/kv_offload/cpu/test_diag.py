# SPDX-License-Identifier: Apache-2.0
"""
Focused unit tests for the KV offload diagnostic writer module.

Tests verify:
1. Gate-off: no env vars → matches() is False, writer methods are no-ops.
2. Gate-on: env vars set → matching requests write JSONL; non-matching skipped.
3. SHA-256 key digest correctness with group-index differentiation.
4. All 6 event types produce expected schema shapes.
5. Per-request sequence isolation.
6. Deterministic collector test: an empty list passed to the lookup helpers
   must receive the first HIT/MISS/RETRY/HIT_PENDING result (catches the
   ``diag_c and diag_c.append()`` falsy-list bug).

The diag module is pure stdlib and can be tested without torch/vLLM.
"""

import hashlib
import json
import os

import pytest

try:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading import diag
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading import scheduler
except ImportError:
    diag = None
    scheduler = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def diag_mod():
    """Return the normally imported production diagnostic module."""
    if diag is None:
        pytest.skip("vLLM package not importable")
    for key in (
        "VLLM_DIAG_KV_OFFLOAD_REQUEST_PREFIX",
        "VLLM_DIAG_KV_OFFLOAD_PATH",
    ):
        os.environ.pop(key, None)
    diag._PREFIX = None
    diag._PATH = None
    diag._GATE = False
    diag._seqs.clear()
    return diag


# ============================================================================
# Gate-off tests
# ============================================================================


class TestGateOff:
    """When env vars are absent, gate must be closed and writer methods inert."""

    def test_matches_false(self, diag_mod):
        assert not diag_mod.matches("anything")

    def test_writer_no_output(self, diag_mod):
        w = diag_mod.KvOffloadDiagWriter()
        w.open()
        assert w._path == ""
        # All methods must be no-ops (no crash, no file write)
        w.group_specs("x", [])
        w.store_prepare("x", 0, 0, 0, 0, [], [], [], 0)
        w.store_complete("x", [], True)
        w.lookup_group("x", 0, 0, "full", False, 0, 0, 0, [], [], 0, 0, False, False, 0)
        w.lookup_terminal("x", None, 0, 0, 0, False)
        w.request_finished("x", [], [], [], [], False)
        w.close()

    def test_key_digest_works_standalone(self, diag_mod):
        assert diag_mod.key_digest(b"hello")


# ============================================================================
# Gate-on tests
# ============================================================================


class TestGateOn:
    """With env vars set, gate opens for matching request IDs."""

    @pytest.fixture(autouse=True)
    def setup_env(self, diag_mod):
        os.environ["VLLM_DIAG_KV_OFFLOAD_REQUEST_PREFIX"] = "MTPDIAG-"
        os.environ["VLLM_DIAG_KV_OFFLOAD_PATH"] = "/tmp/test-diag-pytest.jsonl"
        diag_mod._PREFIX = None
        diag_mod._PATH = None
        diag_mod._GATE = False
        diag_mod._seqs.clear()
        yield
        for k in ("VLLM_DIAG_KV_OFFLOAD_REQUEST_PREFIX",
                  "VLLM_DIAG_KV_OFFLOAD_PATH"):
            os.environ.pop(k, None)
        trace_path = self._trace_path()
        if os.path.exists(trace_path):
            os.unlink(trace_path)

    @staticmethod
    def _trace_path():
        return f"/tmp/test-diag-pytest.jsonl.pid{os.getpid()}"

    @classmethod
    def _read_records(cls):
        with open(cls._trace_path()) as f:
            return [json.loads(l) for l in f if l.strip()]

    @classmethod
    def _fresh_writer(cls, diag_mod):
        trace_path = cls._trace_path()
        if os.path.exists(trace_path):
            os.unlink(trace_path)
        w = diag_mod.KvOffloadDiagWriter()
        w.open()
        diag_mod._seqs.clear()
        return w

    # -- gate control ---------------------------------------------------------

    def test_matches_prefix(self, diag_mod):
        assert diag_mod.matches("MTPDIAG-COLD")
        assert diag_mod.matches("MTPDIAG-REPLAY")
        assert not diag_mod.matches("other-request")

    def test_non_match_filtered(self, diag_mod):
        w = self._fresh_writer(diag_mod)
        w.group_specs("non-matching", [])
        w.close()
        recs = self._read_records()
        assert len(recs) == 1  # only _session_start

    # -- key digest -----------------------------------------------------------

    def test_key_digest_uniqueness(self, diag_mod):
        kd = diag_mod.key_digest
        k0 = b"\x00" * 32 + (0).to_bytes(4, "big")
        k1 = b"\x00" * 32 + (1).to_bytes(4, "big")
        assert kd(k0) != kd(k1)
        assert kd(k0) == hashlib.sha256(k0).hexdigest()

    # -- event shapes ---------------------------------------------------------

    def test_group_specs(self, diag_mod):
        w = self._fresh_writer(diag_mod)
        w.group_specs("MTPDIAG-T", [{"group_idx": 0, "attention_kind": "full"}])
        w.close()
        r = self._read_records()[1]
        assert r["event"] == "group_specs"
        assert r["n"] == 1
        assert r["rid"] == "MTPDIAG-T"

    def test_store_prepare(self, diag_mod):
        w = self._fresh_writer(diag_mod)
        w.store_prepare("MTPDIAG-T", 0, 74, 0, 74,
                        ["cd1", "cd2"], ["sd1"], [0, 1, 2], 42)
        w.close()
        r = self._read_records()[1]
        assert r["event"] == "store_prepare"
        assert r["gidx"] == 0
        assert r["jid"] == 42
        assert r["bi"] == [0, 1, 2]
        assert r["cd"] == ["cd1", "cd2"]
        assert r["sd"] == ["sd1"]

    def test_store_complete(self, diag_mod):
        w = self._fresh_writer(diag_mod)
        w.store_complete("MTPDIAG-T", ["dig1", "dig2"], True)
        w.close()
        r = self._read_records()[1]
        assert r["event"] == "store_complete"
        assert r["ok"] is True
        assert r["digs"] == ["dig1", "dig2"]

    def test_lookup_group(self, diag_mod):
        w = self._fresh_writer(diag_mod)
        w.lookup_group("MTPDIAG-T", 0, 0, "sliding_window", True,
                       0, 4096, 4096, ["d1", "d2"], ["HIT", "MISS"],
                       1, 0, False, False, 2048)
        w.close()
        r = self._read_records()[1]
        assert r["event"] == "lookup_group"
        assert r["gt"] == "sliding_window"
        assert r["ie"] is True
        assert r["lr"] == ["HIT", "MISS"]
        assert r["rhc"] == 1
        assert r["peh"] == 0

    def test_lookup_terminal(self, diag_mod):
        w = self._fresh_writer(diag_mod)
        w.lookup_terminal("MTPDIAG-T", 0, 1, 0, 74, False)
        w.close()
        r = self._read_records()[1]
        assert r["event"] == "lookup_terminal"
        assert r["tgi"] == 0
        assert r["cp"] == 1
        assert r["fet"] == 74

    def test_request_finished(self, diag_mod):
        w = self._fresh_writer(diag_mod)
        w.request_finished("MTPDIAG-T", [74], [100], [300], [42], True)
        w.close()
        r = self._read_records()[1]
        assert r["event"] == "request_finished"
        assert r["hif"] is True
        assert r["pji"] == [42]
        assert r["nsi"] == [74]

    # -- sequence isolation ---------------------------------------------------

    def test_sequence_monotonic(self, diag_mod):
        w = self._fresh_writer(diag_mod)
        w.group_specs("MTPDIAG-A", [])
        w.store_prepare("MTPDIAG-A", 0, 0, 0, 0, [], [], [], 0)
        w.close()
        recs = self._read_records()
        gs = [r for r in recs if r["event"] == "group_specs"][0]
        sp = [r for r in recs if r["event"] == "store_prepare"][0]
        assert gs["seq"] == 1
        assert sp["seq"] == 2

    def test_sequence_independent(self, diag_mod):
        w = self._fresh_writer(diag_mod)
        w.group_specs("MTPDIAG-A", [])
        w.group_specs("MTPDIAG-B", [])
        w.group_specs("MTPDIAG-A", [])
        w.close()
        recs = [r for r in self._read_records() if "rid" in r]
        a = [r for r in recs if r["rid"] == "MTPDIAG-A"]
        b = [r for r in recs if r["rid"] == "MTPDIAG-B"]
        assert a[0]["seq"] == 1
        assert a[1]["seq"] == 2
        assert b[0]["seq"] == 1

    # -- production collector integration (catches falsy-list bug) ------------

    @pytest.mark.parametrize(
        ("lookup_result_name", "expected_count"),
        (("HIT", 1), ("MISS", 0), ("RETRY", None), ("HIT_PENDING", None)),
    )
    def test_maximal_lookup_captures_first_result(
        self, lookup_result_name, expected_count
    ):
        """Invoke the real scheduler helper with a fresh empty collector."""
        if scheduler is None:
            pytest.skip("vLLM package not importable")
        from types import SimpleNamespace
        from vllm.v1.kv_offload.base import LookupResult

        result = getattr(LookupResult, lookup_result_name)

        class FakeManager:
            def lookup(self, key, req_context):
                return result

        fake_scheduler = SimpleNamespace(manager=FakeManager())
        collector = []
        count = scheduler.OffloadingConnectorScheduler._maximal_prefix_lookup(
            fake_scheduler, [b"first-key"], object(), collector
        )
        assert count == expected_count
        assert collector == [
            (hashlib.sha256(b"first-key").hexdigest(), lookup_result_name)
        ]

    def test_sliding_lookup_captures_first_result(self):
        """Invoke the real reverse-scan helper, not a copied append pattern."""
        if scheduler is None:
            pytest.skip("vLLM package not importable")
        from types import SimpleNamespace
        from vllm.v1.kv_offload.base import LookupResult

        class FakeManager:
            def lookup(self, key, req_context):
                return LookupResult.HIT

        fake_scheduler = SimpleNamespace(manager=FakeManager())
        collector = []
        count = scheduler.OffloadingConnectorScheduler._sliding_window_lookup(
            fake_scheduler, [b"first-key"], 1, object(), collector
        )
        assert count == 1
        assert collector == [
            (hashlib.sha256(b"first-key").hexdigest(), "HIT")
        ]

    def test_original_code_uses_correct_pattern(self):
        """Reject the empty-list truthiness bug in the imported scheduler."""
        if scheduler is None:
            pytest.skip("vLLM package not importable")
        with open(scheduler.__file__) as f:
            src = f.read()
        # Count uses of the correct pattern
        correct = src.count("if diag_c is not None:")
        # The wrong pattern must not appear
        wrong = 0
        for line in src.splitlines():
            stripped = line.strip()
            if "diag_c and " in stripped or stripped == "diag_c and":
                wrong += 1
        assert correct >= 1, "No `if diag_c is not None:` guards found!"
        assert wrong == 0, f"Found {wrong} line(s) with `diag_c and` pattern!"


# ============================================================================
# Package import test (requires torch/vLLM environment)
# ============================================================================


class TestPackageImport:
    """Verify normal package import and integration symbols."""

    def test_diag_module_importable(self):
        assert diag is not None
        assert hasattr(diag, "matches")
        assert hasattr(diag, "key_digest")
        assert hasattr(diag, "KvOffloadDiagWriter")

    def test_scheduler_importable(self):
        assert scheduler is not None
        assert hasattr(scheduler, "OffloadingConnectorScheduler")

    def test_diagnostic_helpers_exist(self):
        cls = scheduler.OffloadingConnectorScheduler
        assert hasattr(cls, "_diag_term")
        assert hasattr(cls, "_maybe_diag_lookup_group")
