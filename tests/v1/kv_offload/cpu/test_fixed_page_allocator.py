# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for FixedPageAllocator — compact fixed-page CPU-KV allocator."""

import pytest

from vllm.v1.kv_offload.cpu.fixed_page_allocator import (
    FixedPageAllocator,
)


class TestFixedPageAllocator:
    def test_construction(self):
        alloc = FixedPageAllocator(total_bytes=65536, page_size=4096)
        assert alloc.total_bytes == 65536
        assert alloc.page_size == 4096
        assert alloc.free_bytes == 65536
        assert alloc.used_bytes == 0
        assert alloc.largest_free_block == 65536
        assert alloc.fragmentation == 0.0
        assert alloc.num_active_handles == 0

    def test_invalid_construction(self):
        with pytest.raises(ValueError, match="positive"):
            FixedPageAllocator(total_bytes=-1, page_size=4096)
        with pytest.raises(ValueError, match="positive"):
            FixedPageAllocator(total_bytes=4096, page_size=0)
        with pytest.raises(ValueError, match="divisible by page_size"):
            FixedPageAllocator(total_bytes=4097, page_size=4096)

    def test_allocate_and_free(self):
        alloc = FixedPageAllocator(total_bytes=16384, page_size=4096)
        # Allocate 1 page
        a1 = alloc.allocate(4096)
        assert a1 is not None
        assert a1.logical_length == 4096
        assert a1.allocated_length == 4096
        assert len(a1.page_ids) == 1
        assert alloc.used_bytes == 4096
        assert alloc.free_bytes == 12288

        # Allocate another page
        a2 = alloc.allocate(4096)
        assert a2 is not None
        assert alloc.used_bytes == 8192

        # Free first
        alloc.free(a1)
        assert alloc.used_bytes == 4096
        assert alloc.num_active_handles == 1

        # Free second
        alloc.free(a2)
        assert alloc.used_bytes == 0
        assert alloc.num_active_handles == 0

    def test_allocate_exhaustion(self):
        alloc = FixedPageAllocator(total_bytes=4096, page_size=4096)
        a1 = alloc.allocate(4096)
        assert a1 is not None
        a2 = alloc.allocate(4096)
        assert a2 is None  # exhausted

    def test_pages_required(self):
        alloc = FixedPageAllocator(total_bytes=65536, page_size=4096)
        assert alloc.pages_required(1) == 1
        assert alloc.pages_required(4096) == 1
        assert alloc.pages_required(4097) == 2
        assert alloc.pages_required(8192) == 2

    def test_reset(self):
        alloc = FixedPageAllocator(total_bytes=8192, page_size=4096)
        a1 = alloc.allocate(4096)
        assert a1 is not None
        alloc.reset()
        assert alloc.free_bytes == 8192
        assert alloc.num_active_handles == 0
        # Should allocate again
        a2 = alloc.allocate(4096)
        assert a2 is not None

    def test_free_reuse(self):
        """Freed pages should be reused by subsequent allocations."""
        alloc = FixedPageAllocator(total_bytes=8192, page_size=4096)
        a1 = alloc.allocate(4096)
        a2 = alloc.allocate(4096)
        assert a1 is not None and a2 is not None
        alloc.free(a1)
        a3 = alloc.allocate(4096)
        assert a3 is not None
        # Page IDs should be reused
        assert 0 in a3.page_ids or 1 in a3.page_ids

    def test_simulate_batch_allocation_ok(self):
        alloc = FixedPageAllocator(total_bytes=16384, page_size=4096)
        a1 = alloc.allocate(4096)
        assert a1 is not None
        ok = alloc.simulate_batch_allocation(
            sizes=[4096, 4096],
            frees=[a1],
        )
        assert ok

    def test_simulate_batch_allocation_fail(self):
        alloc = FixedPageAllocator(total_bytes=8192, page_size=4096)
        ok = alloc.simulate_batch_allocation(sizes=[4096, 4096, 4096])
        assert not ok

    def test_page_spans(self):
        alloc = FixedPageAllocator(total_bytes=16384, page_size=4096)
        a1 = alloc.allocate(4096)
        assert a1 is not None
        spans = alloc.page_spans(a1)
        assert len(spans) == 1
        offset, logical, allocated = spans[0]
        assert offset == 0
        assert logical == 4096
        assert allocated == 4096

    def test_page_spans_multi_page(self):
        alloc = FixedPageAllocator(total_bytes=16384, page_size=4096)
        a1 = alloc.allocate(8192)
        assert a1 is not None
        spans = alloc.page_spans(a1)
        # Could be coalesced if pages are contiguous
        total_logical = sum(s[1] for s in spans)
        total_allocated = sum(s[2] for s in spans)
        assert total_logical == 8192
        assert total_allocated == 8192

    def test_cross_allocator_error(self):
        alloc1 = FixedPageAllocator(total_bytes=8192, page_size=4096)
        alloc2 = FixedPageAllocator(total_bytes=8192, page_size=4096)
        a1 = alloc1.allocate(4096)
        assert a1 is not None
        with pytest.raises(ValueError, match="different page allocator"):
            alloc2.free(a1)
        with pytest.raises(ValueError, match="different page allocator"):
            alloc2.page_spans(a1)
        with pytest.raises(ValueError, match="different page allocator"):
            alloc2.simulate_batch_allocation(sizes=[4096], frees=[a1])

    def test_double_free_error(self):
        alloc = FixedPageAllocator(total_bytes=8192, page_size=4096)
        a1 = alloc.allocate(4096)
        assert a1 is not None
        alloc.free(a1)
        with pytest.raises(ValueError, match=r"unknown or already[ -]?freed"):
            alloc.free(a1)

    def test_concurrent_handles(self):
        alloc = FixedPageAllocator(total_bytes=40960, page_size=4096)
        handles = []
        for _ in range(5):
            h = alloc.allocate(4096)
            assert h is not None
            handles.append(h)
        assert alloc.num_active_handles == 5
        for h in handles:
            alloc.free(h)
        assert alloc.num_active_handles == 0

    def test_largest_free_block(self):
        alloc = FixedPageAllocator(total_bytes=16384, page_size=4096)
        assert alloc.largest_free_block == 16384
        alloc.allocate(4096)
        assert alloc.largest_free_block == 12288
        alloc.allocate(4096)
        assert alloc.largest_free_block == 8192

    def test_stale_handle_rejected(self):
        """Allocation from a different allocator instance is rejected."""
        alloc1 = FixedPageAllocator(total_bytes=8192, page_size=4096)
        alloc2 = FixedPageAllocator(total_bytes=8192, page_size=4096)
        a1 = alloc1.allocate(4096)
        assert a1 is not None
        with pytest.raises(ValueError, match="different page allocator"):
            alloc2.free(a1)
        with pytest.raises(ValueError, match="different page allocator"):
            alloc2.page_spans(a1)
        with pytest.raises(ValueError, match="different page allocator"):
            alloc2.simulate_batch_allocation(sizes=[4096], frees=[a1])

    def test_field_mismatch_rejected(self):
        """Forged/mismatched allocation fields are rejected on free."""
        alloc = FixedPageAllocator(total_bytes=8192, page_size=4096)
        a1 = alloc.allocate(4096)
        assert a1 is not None
        # Create a forged allocation with modified fields
        forged = a1.__class__(
            allocator_id=a1.allocator_id,
            id=a1.id,
            page_ids=a1.page_ids,
            logical_length=a1.logical_length,
            allocated_length=9999,  # tampered
        )
        with pytest.raises(ValueError, match="field mismatch"):
            alloc.free(forged)

    def test_atomic_replace_success(self):
        alloc = FixedPageAllocator(total_bytes=16384, page_size=4096)
        a1 = alloc.allocate(4096)
        a2 = alloc.allocate(4096)
        assert a1 is not None and a2 is not None
        result = alloc.atomic_replace(frees=[a1, a2], new_sizes=[8192])
        assert result is not None
        assert len(result) == 1
        assert result[0].allocated_length == 8192
        # After replace: only one handle active
        assert alloc.num_active_handles == 1

    def test_atomic_replace_failure_no_mutation(self):
        """When atomic_replace fails, internal state is unchanged."""
        alloc = FixedPageAllocator(total_bytes=8192, page_size=4096)
        a1 = alloc.allocate(4096)
        assert a1 is not None
        state_before = (
            alloc.free_bytes,
            alloc.used_bytes,
            alloc.num_active_handles,
            list(alloc._free_pages),
            dict(alloc._allocated),
            alloc._next_handle_id,
        )

        # Try to replace with size that requires more pages than available
        result = alloc.atomic_replace(frees=[a1], new_sizes=[8192, 4096])
        assert result is None

        # State unchanged
        assert alloc.free_bytes == state_before[0]
        assert alloc.used_bytes == state_before[1]
        assert alloc.num_active_handles == state_before[2]
        assert list(alloc._free_pages) == state_before[3]

    def test_atomic_replace_zero_mutation_on_failure(self):
        """Inject failure before commit — computed result succeeds but
        internal state is NOT mutated."""
        alloc = FixedPageAllocator(total_bytes=16384, page_size=4096)
        a1 = alloc.allocate(4096)
        assert a1 is not None
        state_before = (
            alloc.free_bytes,
            alloc.used_bytes,
            alloc.num_active_handles,
            list(alloc._free_pages),
            dict(alloc._allocated),
            alloc._next_handle_id,
        )

        with pytest.raises(RuntimeError, match="injected"):
            alloc.atomic_replace(
                frees=[a1],
                new_sizes=[4096],
                _inject_failure_before_commit=True,
            )

        # State must be unchanged — exception was raised before swap
        assert alloc.free_bytes == state_before[0]
        assert alloc.used_bytes == state_before[1]
        assert alloc.num_active_handles == state_before[2]
        assert list(alloc._free_pages) == state_before[3]

    def test_deterministic_accounting(self):
        """Allocations and frees produce deterministic byte accounting."""
        alloc = FixedPageAllocator(total_bytes=16384, page_size=4096)
        totals = [0, 0, 0, 0]
        handles = []
        for i in range(4):
            h = alloc.allocate(1)  # 1 byte -> 1 page (4096)
            assert h is not None
            handles.append(h)
            totals[i] = (i + 1) * 4096
            assert alloc.used_bytes == totals[i]
            assert alloc.free_bytes == 16384 - totals[i]
        # Free in reverse order
        for h in reversed(handles):
            alloc.free(h)
        assert alloc.used_bytes == 0
        assert alloc.free_bytes == 16384

    def test_reset_invalidates_stale_handles(self):
        """After reset, previously valid handles are rejected."""
        alloc = FixedPageAllocator(total_bytes=8192, page_size=4096)
        a1 = alloc.allocate(4096)
        assert a1 is not None
        alloc.reset()
        # The stale handle is now unknown
        with pytest.raises(ValueError, match=r"unknown or already[ -]?freed"):
            alloc.free(a1)
        with pytest.raises(ValueError, match="unknown page allocation"):
            alloc.page_spans(a1)
