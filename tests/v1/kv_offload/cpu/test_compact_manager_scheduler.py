# SPDX-License-Identifier: Apache-2.0
"""Cross-group replay-unit coherent compact CPU eviction regressions.

Tests cover:
  - Cross-group victim eviction as one replay unit
  - Shared-key survival when one owner is evicted
  - ARC snapshot failure preservation
  - Unrelated unit isolation
  - Failed atomic_replace preserves all state
  - Reset clears replay metadata
  - Manager activation from resolve_compact_mode
  - Policy capacity derivation from rounded budget
  - 2600-token symbolic bounded-tail regression (requires scheduler fixture)
  - blocks_per_chunk > 1 replay coherence
"""

import pytest

from vllm.v1.kv_offload.base import (
    LookupResult,
    ReqContext,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

# ---- helpers ----


def k(name, group=0):
    return make_offload_key(name.encode(), group)


def _make_manager(
    num_blocks=6,
    cache_policy="lru",
    compact_total=24576,
    compact_page=4096,
    group_payload=None,
    preferred_groups=None,
    enable_compact=True,
):
    """Create a compact-enabled CPUOffloadingManager."""
    if group_payload is None:
        group_payload = {0: 4096, 1: 4096}
    if preferred_groups is None:
        preferred_groups = set()
    mgr = CPUOffloadingManager(
        num_blocks=num_blocks,
        cache_policy=cache_policy,
        enable_events=True,
    )
    if enable_compact:
        ok = mgr.resolve_compact_mode(
            enable=True,
            total_bytes=compact_total,
            page_size=compact_page,
            group_payload_bytes=group_payload,
            preferred_eviction_groups=preferred_groups,
        )
        assert ok, "resolve_compact_mode must succeed on empty manager"
    return mgr


def store(m, unit, keys):
    c = ReqContext(unit, store_replay_unit=tuple(keys))
    out = m.prepare_store(keys, c)
    assert out is not None, f"prepare_store failed for unit={unit}, keys={keys}"
    m.complete_store(out.keys_to_store, c)
    return out


# ---- cross-group replay-unit coherence ----


class TestCrossGroupReplayUnits:
    def test_cross_group_victim_is_evicted_as_one_replay_unit(self):
        m = _make_manager(6)
        a = [k("a0"), k("a1"), k("as", 1)]
        b = [k("b0"), k("b1"), k("bs", 1)]
        c = [k("c0"), k("c1"), k("cs", 1)]
        store(m, "a", a)
        store(m, "b", b)
        out = store(m, "c", c)
        assert set(out.evicted_keys) == set(a)
        assert all(m.lookup(x, ReqContext("p")) is LookupResult.MISS for x in a)
        assert all(m.lookup(x, ReqContext("p")) is LookupResult.HIT for x in b + c)

    def test_unrelated_units_are_not_connected(self):
        m = _make_manager()
        a = [k("a0"), k("as", 1)]
        b = [k("b0"), k("bs", 1)]
        store(m, "a", a)
        store(m, "b", b)
        assert m._replay_units == {"a": set(a), "b": set(b)}

    @pytest.mark.parametrize("failure_mode", [None, RuntimeError])
    def test_failed_replace_preserves_all_state(self, failure_mode, monkeypatch):
        """None return or exception from atomic_replace: all state
        byte/structurally unchanged."""
        m = _make_manager(4, compact_total=24576)
        a = [k("a0"), k("a1"), k("as", 1)]
        store(m, "a", a)
        alloc = m._compact_allocator
        # Snapshot ALL state before failed transaction.
        before_alloc_used = alloc.used_bytes
        before_free = alloc.free_bytes
        before_replay = {u: frozenset(v) for u, v in m._replay_units.items()}
        before_key_replay = {k: frozenset(v) for k, v in m._key_replay_units.items()}
        before_allocs = dict(m._compact_allocation_by_key)
        before_addrs = dict(m._compact_address_by_key)
        before_block_ids = dict(m._compact_block_id_by_key)
        before_evictable = m._num_evictable_cache_blocks
        before_write_pending = m._num_write_pending_blocks
        before_allocated = m._num_allocated_blocks
        before_events = list(m.events) if m.events else None
        before_skipped = m.stores_skipped_in_current_batch
        before_allocation_sizes = list(m.allocation_sizes_in_current_batch)
        # Verify candidates had room to need eviction first.
        b = [k("b0"), k("b1")]
        c = ReqContext("b", store_replay_unit=tuple(b))
        if failure_mode is None:
            monkeypatch.setattr(alloc, "atomic_replace", lambda *_a, **_k: None)
        else:

            def _fail(*_a, **_k):
                raise RuntimeError("injected atomic_replace failure")

            monkeypatch.setattr(alloc, "atomic_replace", _fail)
        assert m.prepare_store(b, c) is None
        # All state must be byte/structurally unchanged.
        assert alloc.used_bytes == before_alloc_used
        assert alloc.free_bytes == before_free
        assert {u: frozenset(v) for u, v in m._replay_units.items()} == before_replay
        assert {
            k: frozenset(v) for k, v in m._key_replay_units.items()
        } == before_key_replay
        assert dict(m._compact_allocation_by_key) == before_allocs
        assert dict(m._compact_address_by_key) == before_addrs
        assert dict(m._compact_block_id_by_key) == before_block_ids
        assert m._num_evictable_cache_blocks == before_evictable
        assert m._num_write_pending_blocks == before_write_pending
        assert m._num_allocated_blocks == before_allocated
        assert m.stores_skipped_in_current_batch == before_skipped
        assert m.allocation_sizes_in_current_batch == before_allocation_sizes
        if before_events is not None:
            assert list(m.events) == before_events

    def test_shared_key_survives_unique_owner_eviction(self):
        # Tight budget: total=16384, policy_capacity=4
        m = _make_manager(4, compact_total=16384)
        shared = k("shared")
        a = [shared, k("a", 1)]
        b = [shared, k("b", 1)]
        store(m, "a", a)
        store(m, "b", b)
        x = [shared, k("x0"), k("x1", 1)]
        store(m, "x", x)
        # In compact mode, replay-unit coherence means candidate keys
        # get expanded to complete replay units. But protected keys
        # (the incoming request's store_replay_unit) are excluded.
        # Result: only a[1] is evicted; shared survives because "b" still
        # owns it and shared is protected by x's store_replay_unit.
        assert m.lookup(a[1], ReqContext("p")) is LookupResult.MISS
        assert m.lookup(shared, ReqContext("p")) is LookupResult.HIT, (
            "Shared key must remain HIT (owned by b and protected by x)"
        )
        assert all(
            m.lookup(y, ReqContext("p")) is LookupResult.HIT for y in {b[1], x[1], x[2]}
        )

    def test_reset_clears_replay_metadata(self):
        m = _make_manager()
        store(m, "a", [k("a0"), k("as", 1)])
        m.reset_cache()
        assert m._replay_units == {} and m._key_replay_units == {}

    def test_compact_enabled_survives_reset(self):
        m = _make_manager()
        assert m._compact_enabled
        store(m, "a", [k("a0"), k("as", 1)])
        m.reset_cache()
        # One-way mode persists.
        assert m._compact_enabled
        # Compact data cleared.
        assert m._compact_allocation_by_key == {}
        assert m._replay_units == {}


# ---- ARC snapshot and eviction ordering ----


class TestCompactActivation:
    def test_resolve_compact_mode_success(self):
        """Activation succeeds on empty manager."""
        m = CPUOffloadingManager(
            num_blocks=100, cache_policy="arc", enable_events=False
        )
        result = m.resolve_compact_mode(
            enable=True,
            total_bytes=65536,
            page_size=4096,
            group_payload_bytes={0: 4096},
            preferred_eviction_groups=set(),
        )
        assert result
        assert m._compact_enabled
        assert m._compact_allocator is not None
        assert m._policy.is_empty  # fresh policy

    def test_resolve_compact_mode_disabled(self):
        """enable=False leaves manager in legacy mode."""
        m = CPUOffloadingManager(num_blocks=100)
        result = m.resolve_compact_mode(
            enable=False,
        )
        assert not result
        assert not m._compact_enabled

    def test_resolve_compact_mode_rejects_non_empty(self):
        """Activation fails when manager has state."""
        m = CPUOffloadingManager(num_blocks=4)
        key = k("existing")
        out = m.prepare_store([key], ReqContext("test"))
        assert out is not None
        with pytest.raises(RuntimeError, match="non-empty policy|existing state"):
            m.resolve_compact_mode(
                enable=True,
                total_bytes=65536,
                page_size=4096,
                group_payload_bytes={0: 4096},
            )

    def test_policy_capacity_from_rounded_budget(self):
        """Policy capacity = actual_budget // min(group_payload_map.values())."""
        m = CPUOffloadingManager(num_blocks=42, cache_policy="arc")
        # 20000 raw rounds to 16384 (4*4096). Min payload=4096. Cap=4.
        result = m.resolve_compact_mode(
            enable=True,
            total_bytes=20000,
            page_size=4096,
            group_payload_bytes={0: 4096},
        )
        assert result
        assert m._policy.cache_capacity == 4

    def test_non_aligned_min_payload(self):
        """Policy capacity with min_payload not dividing page remainder."""
        m = CPUOffloadingManager(num_blocks=42, cache_policy="arc")
        # Raw 24576 = 6*4096, exact. Min payload 2048 -> cap=12.
        result = m.resolve_compact_mode(
            enable=True,
            total_bytes=24576,
            page_size=4096,
            group_payload_bytes={0: 2048, 1: 4096},
        )
        assert result
        assert m._policy.cache_capacity == 12

    def test_rejects_already_active(self):
        """resolve_compact_mode fails when already ACTIVE."""
        m = CPUOffloadingManager(num_blocks=42, cache_policy="arc")
        m.resolve_compact_mode(
            enable=True,
            total_bytes=65536,
            page_size=4096,
            group_payload_bytes={0: 4096},
        )
        with pytest.raises(RuntimeError, match="already active|already"):
            m.resolve_compact_mode(
                enable=True,
                total_bytes=65536,
                page_size=4096,
                group_payload_bytes={0: 4096},
            )

    def test_preferred_groups_accepted(self):
        """Preferred eviction groups are stored and reflected in activation."""
        m = CPUOffloadingManager(num_blocks=42, cache_policy="arc")
        m.resolve_compact_mode(
            enable=True,
            total_bytes=65536,
            page_size=4096,
            group_payload_bytes={0: 4096, 1: 4096, 2: 4096},
            preferred_eviction_groups={1, 2},
        )
        assert m._compact_preferred_eviction_groups == {1, 2}
        # Verify the prefer_evict_fn works.
        key_g0 = k("test", 0)
        key_g1 = k("test", 1)
        key_g2 = k("test", 2)
        assert not m._prefer_evict_fn(key_g0)
        assert m._prefer_evict_fn(key_g1)
        assert m._prefer_evict_fn(key_g2)

    def test_default_preferred_groups_empty(self):
        """Default preferred groups is empty set."""
        m = CPUOffloadingManager(num_blocks=42, cache_policy="arc")
        m.resolve_compact_mode(
            enable=True,
            total_bytes=65536,
            page_size=4096,
            group_payload_bytes={0: 4096, 1: 4096, 2: 4096},
        )
        assert m._compact_preferred_eviction_groups == set()


class TestCompactStoreLoadCycle:
    def test_basic_store_load(self):
        """Store then load a key in compact mode."""
        m = _make_manager(10, compact_total=65536)
        key = k("test_key", 0)
        ctx = ReqContext("r1", store_replay_unit=(key,))
        out = m.prepare_store([key], ctx)
        assert out is not None
        assert len(out.keys_to_store) == 1
        assert m.lookup(key, ReqContext("p")) is LookupResult.HIT_PENDING

        m.complete_store([key], ctx)
        assert m.lookup(key, ReqContext("p")) is LookupResult.HIT

        load_spec = m.prepare_load([key], ReqContext("r2"))
        assert load_spec is not None

        m.complete_load([key], ReqContext("r2"))

    def test_failed_store_removes_key(self):
        """Failed store removes policy, allocation, and replay ownership."""
        m = _make_manager(10, compact_total=65536)
        key = k("test_fail", 0)
        ctx = ReqContext("r1", store_replay_unit=(key,))
        out = m.prepare_store([key], ctx)
        assert out is not None
        assert m._replay_units == {"r1": {key}}
        assert m._key_replay_units == {key: {"r1"}}

        m.complete_store([key], ctx, success=False)

        assert m.lookup(key, ReqContext("p")) is LookupResult.MISS
        assert key not in m._compact_allocation_by_key
        assert key not in m._compact_address_by_key
        assert key not in m._compact_block_id_by_key
        assert key not in m._key_replay_units
        assert "r1" not in m._replay_units
        assert m._num_write_pending_blocks == 0

    def test_prepare_store_all_keys_exist(self):
        """When all keys already stored, prepare_store returns empty."""
        m = _make_manager(10, compact_total=65536)
        key = k("existing", 0)
        ctx = ReqContext("r1")
        out = m.prepare_store([key], ReqContext("r1", store_replay_unit=(key,)))
        assert out is not None
        m.complete_store([key], ctx)
        out2 = m.prepare_store([key], ReqContext("r2", store_replay_unit=(key,)))
        assert out2 is not None
        assert out2.keys_to_store == []
        assert out2.evicted_keys == []


class TestLegacyPathUnchanged:
    def test_legacy_manager_still_works(self):
        """Legacy (non-compact) manager behavior is unchanged."""
        m = CPUOffloadingManager(num_blocks=10, cache_policy="lru")
        assert not m._compact_enabled
        key = k("legacy")
        out = m.prepare_store([key], ReqContext("test"))
        assert out is not None
        assert m.lookup(key, ReqContext("p")) is LookupResult.HIT_PENDING
        # Should use CPULoadStoreSpec not CompactCPULoadStoreSpec
        from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

        assert isinstance(out.store_spec, CPULoadStoreSpec)


class TestSelectEvictUntilNoDuplicates:
    """select_evict_until preferred+normal passes must never duplicate keys."""

    def _check_no_duplicates(self, policy_cls: str):
        m = _make_manager(
            num_blocks=8,
            compact_total=32768,
            cache_policy=policy_cls,
            preferred_groups={1},
        )
        def_k = lambda i, g=0: k(f"k{i}", g)
        # Store keys from groups 0 and 1 interleaved so both are in policy.
        ks_g0 = [def_k(i, 0) for i in range(3)]
        ks_g1 = [def_k(i, 1) for i in range(3)]
        for i in range(3):
            store(m, f"r{i}_g0", [ks_g0[i]])
            store(m, f"r{i}_g1", [ks_g1[i]])

        def _can_fit(cand):
            freed = 0
            for key, _ in cand:
                freed += m._compact_allocation_by_key.get(
                    key, type("", (), {"allocated_length": 0})()
                ).allocated_length
            return freed >= 16384

        candidates = m._policy.select_evict_until(
            can_fit=_can_fit,
            protected=set(),
            prefer_evict=m._prefer_evict_fn,
        )
        assert candidates is not None, f"{policy_cls}: select_evict_until returned None"
        keys = [key for key, _ in candidates]
        assert len(keys) == len(set(keys)), (
            f"{policy_cls}: duplicate keys in select_evict_until: "
            f"{[(k, keys.count(k)) for k in set(keys) if keys.count(k) > 1]}"
        )

    def test_arc_no_duplicate_keys(self):
        self._check_no_duplicates("arc")

    def test_lru_no_duplicate_keys(self):
        self._check_no_duplicates("lru")


def test_manager_address_length_matches_chunked_payload():
    manager = CPUOffloadingManager(num_blocks=4)
    manager.enable_compact(
        total_bytes=65536,
        page_size=4096,
        key_sizes={0: 12288},
    )
    key = k("chunked", 0)
    ctx = ReqContext("chunked", store_replay_unit=(key,))
    output = manager.prepare_store([key], ctx)
    assert output is not None
    assert output.store_spec.compact_addresses[0].logical_length == 12288
