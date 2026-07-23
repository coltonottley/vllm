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
  - ARC virtual-T1 interleaved T1/T2 candidate ordering
  - ARC preferred-then-normal partition ordering
"""

import pytest

from vllm.v1.kv_offload.base import (
    LookupResult,
    ReqContext,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus

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

    def test_failed_replace_returns_none_preserves_all_state(self, monkeypatch):
        """None return from atomic_replace (ordinary no-fit): prepare_store
        returns None; all state byte/structurally unchanged."""
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
        monkeypatch.setattr(alloc, "atomic_replace", lambda *_a, **_k: None)
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

    def test_failed_replace_raises_preserves_all_state(self, monkeypatch):
        """RuntimeError from atomic_replace (injected fault before swap):
        exception propagates through prepare_store; all state byte/structurally
        unchanged."""
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

        def _fail(*_a, **_k):
            raise RuntimeError("injected atomic_replace failure")

        monkeypatch.setattr(alloc, "atomic_replace", _fail)
        with pytest.raises(RuntimeError, match="injected atomic_replace failure"):
            m.prepare_store(b, c)

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

    # ---- replay-closure-aware fit predicate regression ----

    def test_replay_closure_fit_shared_key_survives_other_unit(self):
        """3-page pool, A={shared,a}, B={shared,b}, incoming X={x0,x1}.

        Raw policy candidates shared+a appear to free 2 pages, but replay
        closure reveals shared survives (owned by B), so only 1 page would
        actually be freed.  Old ``_can_fit`` naively summed raw bytes and
        returned True; ``_replay_eviction_plan`` then expanded closure but
        could only free a single page, causing ``atomic_replace`` to fail.

        Fix: ``_can_fit`` evaluates replay closure; it continues selecting
        until b is also included, unlocking full 3-page closure eviction.
        """
        m = _make_manager(
            num_blocks=6,
            compact_total=12288,  # 3 pages x 4096
            compact_page=4096,
            group_payload={0: 4096, 1: 4096},
        )
        shared = k("shared", 0)
        a = k("a", 1)
        b = k("b", 1)

        # Fill the 3-page pool.
        store(m, "a", [shared, a])  # uses 2 pages, 1 free
        store(m, "b", [shared, b])  # shared HIT, b uses last page

        # Incoming X needs 2 pages, pool is full.
        # Old _can_fit would see candidates [shared, a] freeing 2 raw pages
        # and return True — but replay closure frees only a (1 page) since
        # shared survives due to Unit B ownership.
        x0 = k("x0", 0)
        x1 = k("x1", 1)
        ctx = ReqContext("x", store_replay_unit=(x0, x1))
        out = m.prepare_store([x0, x1], ctx)
        assert out is not None, (
            "prepare_store must succeed: closure {shared,a,b} frees 3 pages, "
            "enough for X={x0,x1} needing 2 pages"
        )
        # Closure eviction: all three shared/a/b evicted.
        assert set(out.evicted_keys) == {shared, a, b}, (
            f"Expected full closure eviction {{shared,a,b}}, got {out.evicted_keys}"
        )
        m.complete_store(out.keys_to_store, ctx)

        # Evicted keys are MISS.
        assert m.lookup(shared, ReqContext("p")) is LookupResult.MISS
        assert m.lookup(a, ReqContext("p")) is LookupResult.MISS
        assert m.lookup(b, ReqContext("p")) is LookupResult.MISS

        # Stored X keys are HIT.
        assert m.lookup(x0, ReqContext("p")) is LookupResult.HIT
        assert m.lookup(x1, ReqContext("p")) is LookupResult.HIT

    def test_replay_closure_fit_protected_shared_no_fit_zero_mutation(self):
        """When a protected shared key prevents closure eviction from freeing
        enough space, prepare_store returns None and all state is unchanged.

        3-page pool, A={shared,a}, B={shared,b}.  Incoming X needs more bytes
        than full closure can free, so prepare_store must return None with
        zero mutation.
        """
        m = _make_manager(
            num_blocks=6,
            compact_total=12288,
            compact_page=4096,
            group_payload={0: 4096, 1: 4096},
        )
        shared = k("shared", 0)
        a = k("a", 1)
        b = k("b", 1)

        store(m, "a", [shared, a])
        store(m, "b", [shared, b])

        # Snapshot all mutable state.
        before_replay = {u: frozenset(v) for u, v in m._replay_units.items()}
        before_key_replay = {k: frozenset(v) for k, v in m._key_replay_units.items()}
        before_allocs = dict(m._compact_allocation_by_key)
        before_evictable = m._num_evictable_cache_blocks
        before_free = m._compact_allocator.free_bytes
        before_used = m._compact_allocator.used_bytes

        # Incoming X needs 4 pages (16384 bytes) — more than full
        # closure of 3 pages can provide.
        x0 = k("x0", 0)
        x1 = k("x1", 1)
        x2 = k("x2", 0)
        x3 = k("x3", 1)
        ctx = ReqContext("x", store_replay_unit=(x0, x1, x2, x3))
        out = m.prepare_store([x0, x1, x2, x3], ctx)
        assert out is None, (
            "No-fit: pool 3 pages = 12288 bytes, need 4 pages = 16384 bytes"
        )

        # Zero mutation: all state must be byte/structurally unchanged.
        assert {u: frozenset(v) for u, v in m._replay_units.items()} == before_replay
        assert {
            k: frozenset(v) for k, v in m._key_replay_units.items()
        } == before_key_replay
        assert dict(m._compact_allocation_by_key) == before_allocs
        assert m._num_evictable_cache_blocks == before_evictable
        assert m._compact_allocator.free_bytes == before_free
        assert m._compact_allocator.used_bytes == before_used

        # Original keys survive unharmed.
        assert m.lookup(shared, ReqContext("p")) is LookupResult.HIT
        assert m.lookup(a, ReqContext("p")) is LookupResult.HIT
        assert m.lookup(b, ReqContext("p")) is LookupResult.HIT


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

    def test_failed_store_shared_key_no_phantom_owner(self):
        """Regression: failed store must not leave phantom replay-unit
        ownership for shared keys.

        Scenario:
        - old={shared} completed successfully.
          shared owned by {'old'}.
        - new={shared, a(group0), b(group1)} prepared.
          shared already resident, a and b are new.
          After prepare_store / _commit_replay_unit,
          _replay_units['new'] == {shared, a, b}.
        - complete_store([a, b], new_ctx, success=False).

        After failure:
        - 'new' must not appear in _replay_units at all.
        - shared must have owners exactly {'old'}.
        - a and b absent from policy, allocations, addresses, block_ids.
        - shared remains HIT (old still owns it).
        - _num_write_pending_blocks == 0.
        - allocator used_bytes returned to pre-prepare state.
        """
        m = _make_manager(10, compact_total=65536, group_payload={0: 4096, 1: 4096})

        shared = k("shared")
        old_ctx = ReqContext("old", store_replay_unit=(shared,))
        out = m.prepare_store([shared], old_ctx)
        assert out is not None
        m.complete_store([shared], old_ctx)
        assert m.lookup(shared, ReqContext("p")) is LookupResult.HIT
        assert m._key_replay_units[shared] == {"old"}, (
            f"After old complete, shared owners must be {{'old'}}, "
            f"got {m._key_replay_units[shared]}"
        )

        alloc = m._compact_allocator
        before_bytes = alloc.used_bytes

        a = k("a", 0)
        b = k("b", 1)
        new_ctx = ReqContext("new", store_replay_unit=(shared, a, b))
        out = m.prepare_store([shared, a, b], new_ctx)
        assert out is not None
        assert set(out.keys_to_store) == {a, b}, (
            f"shared already stored, only a,b should be keys_to_store, "
            f"got {out.keys_to_store}"
        )
        assert m._replay_units.get("new") == {shared, a, b}, (
            f"After prepare_store, new unit should own {{shared,a,b}}, "
            f"got {m._replay_units.get('new')}"
        )
        assert m._key_replay_units[shared] == {"old", "new"}, (
            f"Shared must have owners {{'old','new'}} after prepare, "
            f"got {m._key_replay_units[shared]}"
        )

        m.complete_store([a, b], new_ctx, success=False)

        assert "new" not in m._replay_units, (
            f"Failed unit 'new' must be absent from _replay_units, "
            f"leftover: {m._replay_units.get('new')}"
        )
        assert m._key_replay_units.get(shared) == {"old"}, (
            f"Shared must have owners exactly {{'old'}}, "
            f"got {m._key_replay_units.get(shared)}"
        )
        assert m.lookup(a, ReqContext("p")) is LookupResult.MISS
        assert m.lookup(b, ReqContext("p")) is LookupResult.MISS
        assert a not in m._compact_allocation_by_key
        assert b not in m._compact_allocation_by_key
        assert a not in m._compact_address_by_key
        assert b not in m._compact_address_by_key
        assert a not in m._compact_block_id_by_key
        assert b not in m._compact_block_id_by_key
        assert a not in m._key_replay_units
        assert b not in m._key_replay_units
        assert m.lookup(shared, ReqContext("p")) is LookupResult.HIT
        assert m._num_write_pending_blocks == 0
        assert alloc.used_bytes == before_bytes, (
            f"Allocator used_bytes should be {before_bytes} "
            f"(pre-prepare), got {alloc.used_bytes}"
        )

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


class TestARCCandidateOrdering:
    """ARC select_evict_until must interleave T1/T2 by virtual T1 size."""

    def make_key(self, name: str, group: int = 0):
        return make_offload_key(name.encode(), group)

    def make_block(self, block_id: int = 0):
        b = BlockStatus(block_id)
        b.ref_cnt = 0
        return b

    def _snapshot(self, policy):
        return {
            "t1": dict(policy.t1),
            "t2": dict(policy.t2),
            "b1": dict(policy.b1),
            "b2": dict(policy.b2),
            "target": policy.target_t1_size,
        }

    def test_virtual_t1_interleaves_t1_t2(self):
        """T1=[t1a,t1b,t1c], T2=[t2a,t2b], target=3, predicate len>=2
        -> must select [t1a,t2a] (one from T1, then virtual_t1 drops below
        target so next must come from T2)."""
        from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy

        policy = ARCCachePolicy(cache_capacity=10)
        t1a, t1b, t1c = [self.make_key(f"t1{x}") for x in ("a", "b", "c")]
        t2a, t2b = [self.make_key(f"t2{x}") for x in ("a", "b")]
        blk = self.make_block()
        for k in (t1a, t1b, t1c):
            policy.t1[k] = blk
        for k in (t2a, t2b):
            policy.t2[k] = blk
        policy.target_t1_size = 3.0

        before = self._snapshot(policy)

        candidates = policy.select_evict_until(
            can_fit=lambda c: len(c) >= 2,
            protected=set(),
        )

        assert candidates is not None, "select_evict_until must return candidates"
        result_keys = [key for key, _ in candidates]
        assert result_keys == [t1a, t2a], f"Expected [t1a, t2a] but got {result_keys}"

        # Policy state must be unchanged (non-mutating).
        after = self._snapshot(policy)
        assert before == after, (
            f"Policy state changed after select_evict_until: "
            f"before={before}, after={after}"
        )

    def test_all_from_t1_when_virtual_t1_above_target(self):
        """When virtual T1 stays above target after each selection, all
        candidates come from T1."""
        from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy

        policy = ARCCachePolicy(cache_capacity=10)
        t1a, t1b = [self.make_key(f"t1{x}") for x in ("a", "b")]
        blk = self.make_block()
        for k in (t1a, t1b):
            policy.t1[k] = blk
        policy.target_t1_size = 1.0  # virtual_t1=2 >= 1, stays >= 1

        candidates = policy.select_evict_until(
            can_fit=lambda c: len(c) >= 2,
            protected=set(),
        )
        assert candidates is not None
        result_keys = [key for key, _ in candidates]
        assert result_keys == [t1a, t1b], f"Expected [t1a, t1b] but got {result_keys}"

    def test_preferred_order_within_t1(self):
        """Preferred keys offered before normal within T1 partition."""
        from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy

        policy = ARCCachePolicy(cache_capacity=10)
        t1_a = self.make_key("t1_a")
        t1_b = self.make_key("t1_b")
        t2_a = self.make_key("t2_a")
        t2_b = self.make_key("t2_b")
        blk = self.make_block()
        policy.t1[t1_a] = blk
        policy.t1[t1_b] = blk
        policy.t2[t2_a] = blk
        policy.t2[t2_b] = blk
        policy.target_t1_size = 3.0  # virtual_t1=2 < 3 -> start from T2

        def prefer_t2_pairs(key):
            return key in {t2_a, t2_b}

        candidates = policy.select_evict_until(
            can_fit=lambda c: len(c) >= 2,
            protected=set(),
            prefer_evict=prefer_t2_pairs,
        )
        assert candidates is not None
        result_keys = [key for key, _ in candidates]
        # virtual_t1=2 < target=3 -> T2 preferred first: t2_a (preferred),
        # then on next iteration, virtual_t1 still < target -> T2 normal: t2_b
        assert result_keys == [t2_a, t2_b], (
            f"Expected [t2_a, t2_b] (preferred from T2) but got {result_keys}"
        )

    def test_preferred_then_virtual_t1_drops_below_target(self):
        """Select preferred from T1 with virtual_t1 >= target; after decrement,
        virtual_t1 < target so next candidate comes from T2."""
        from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy

        policy = ARCCachePolicy(cache_capacity=10)
        t1_a = self.make_key("t1_a")
        t1_b = self.make_key("t1_b")
        t2_a = self.make_key("t2_a")
        blk = self.make_block()
        policy.t1[t1_a] = blk
        policy.t1[t1_b] = blk
        policy.t2[t2_a] = blk
        policy.target_t1_size = 2.0  # virtual_t1=2 >= 2

        def prefer_t1_b(key):
            return key == t1_b

        candidates = policy.select_evict_until(
            can_fit=lambda c: len(c) >= 2,
            protected=set(),
            prefer_evict=prefer_t1_b,
        )
        assert candidates is not None
        result_keys = [key for key, _ in candidates]
        # virtual_t1=2 >= target=2 -> T1 preferred first: t1_b
        # then virtual_t1=1: 1 < 2, so T2: t2_a
        assert result_keys == [t1_b, t2_a], (
            f"Expected [t1_b, t2_a] but got {result_keys}"
        )


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
