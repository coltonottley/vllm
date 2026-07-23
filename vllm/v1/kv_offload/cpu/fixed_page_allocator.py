# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-page allocator for spatially robust compact CPU KV storage."""

from __future__ import annotations

import dataclasses
import heapq
import itertools
from collections.abc import Sequence

_allocator_ids = itertools.count()


@dataclasses.dataclass(frozen=True)
class PageAllocation:
    allocator_id: int
    id: int
    page_ids: tuple[int, ...]
    logical_length: int
    allocated_length: int


class FixedPageAllocator:
    """Allocate logical payloads over interchangeable fixed-size pages."""

    def __init__(self, total_bytes: int, page_size: int) -> None:
        if total_bytes <= 0:
            raise ValueError("total_bytes must be positive")
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        if total_bytes % page_size:
            raise ValueError("total_bytes must be divisible by page_size")
        self._total_bytes = total_bytes
        self._page_size = page_size
        self._allocator_id = next(_allocator_ids)
        self._free_pages = list(range(total_bytes // page_size))
        heapq.heapify(self._free_pages)
        self._allocated: dict[int, PageAllocation] = {}
        self._next_handle_id = 0

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def page_size(self) -> int:
        return self._page_size

    @property
    def free_bytes(self) -> int:
        return len(self._free_pages) * self._page_size

    @property
    def used_bytes(self) -> int:
        return self._total_bytes - self.free_bytes

    @property
    def largest_free_block(self) -> int:
        # Every free page is independently usable; no contiguous-fit requirement.
        return self.free_bytes

    @property
    def fragmentation(self) -> float:
        return 0.0

    @property
    def num_active_handles(self) -> int:
        return len(self._allocated)

    def pages_required(self, size: int) -> int:
        if size <= 0:
            raise ValueError("allocation size must be positive")
        return (size + self._page_size - 1) // self._page_size

    def allocate(self, size: int) -> PageAllocation | None:
        num_pages = self.pages_required(size)
        if num_pages > len(self._free_pages):
            return None
        page_ids = tuple(heapq.heappop(self._free_pages) for _ in range(num_pages))
        allocation = PageAllocation(
            allocator_id=self._allocator_id,
            id=self._next_handle_id,
            page_ids=page_ids,
            logical_length=size,
            allocated_length=num_pages * self._page_size,
        )
        self._allocated[allocation.id] = allocation
        self._next_handle_id += 1
        return allocation

    def free(self, allocation: PageAllocation) -> None:
        if allocation.allocator_id != self._allocator_id:
            raise ValueError("allocation belongs to a different page allocator")
        existing = self._allocated.get(allocation.id)
        if existing is None:
            raise ValueError("unknown or already-freed page allocation")
        if existing != allocation:
            raise ValueError("page allocation field mismatch")
        self._allocated.pop(allocation.id)
        for page_id in allocation.page_ids:
            heapq.heappush(self._free_pages, page_id)

    def reset(self) -> None:
        self._free_pages = list(range(self._total_bytes // self._page_size))
        heapq.heapify(self._free_pages)
        self._allocated.clear()

    def simulate_batch_allocation(
        self,
        sizes: Sequence[int],
        frees: Sequence[PageAllocation] | None = None,
    ) -> bool:
        if any(size <= 0 for size in sizes):
            raise ValueError("allocation size must be positive")
        available_pages = len(self._free_pages)
        seen: set[int] = set()
        for allocation in frees or ():
            if allocation.allocator_id != self._allocator_id:
                raise ValueError("candidate belongs to a different page allocator")
            existing = self._allocated.get(allocation.id)
            if existing is None:
                raise ValueError("candidate free has unknown allocation")
            if existing != allocation:
                raise ValueError("candidate free field mismatch")
            if allocation.id in seen:
                raise ValueError("duplicate candidate page allocation")
            seen.add(allocation.id)
            available_pages += len(allocation.page_ids)
        required_pages = sum(self.pages_required(size) for size in sizes)
        return required_pages <= available_pages

    def atomic_replace(
        self,
        frees: Sequence[PageAllocation],
        new_sizes: list[int],
        *,
        _inject_failure_before_commit: bool = False,
    ) -> list[PageAllocation] | None:
        """Atomically free *frees* and allocate *new_sizes* in one transaction.

        Validates all inputs against current state, then computes the
        result on a **copy** of the free heap, allocated map, and
        next-handle-id.  If allocation succeeds (enough pages), internal
        state is swapped atomically **once** and the new allocations
        are returned.  If allocation would fail, state is unchanged and
        None is returned.

        When ``_inject_failure_before_commit`` is True (test hook), a
        RuntimeError is raised **after** computing the result but
        **before** swapping state — proving that the computed result
        would succeed while guaranteeing zero mutation.

        No live mutation occurs before the single swap.
        """
        if any(size <= 0 for size in new_sizes):
            raise ValueError("allocation size must be positive")
        seen: set[int] = set()
        for alloc in frees:
            if alloc.allocator_id != self._allocator_id:
                raise ValueError("free candidate belongs to a different page allocator")
            existing = self._allocated.get(alloc.id)
            if existing is None:
                raise ValueError(
                    f"free candidate id={alloc.id} is unknown or already freed"
                )
            if existing != alloc:
                raise ValueError(f"free candidate id={alloc.id} field mismatch")
            if alloc.id in seen:
                raise ValueError(f"duplicate free candidate id={alloc.id}")
            seen.add(alloc.id)

        # --- Copy mutable state ---
        new_free = list(self._free_pages)
        new_allocated = dict(self._allocated)
        new_next_id = self._next_handle_id

        # --- Apply frees to the copy ---
        for alloc in frees:
            del new_allocated[alloc.id]
            for page_id in alloc.page_ids:
                heapq.heappush(new_free, page_id)

        # --- Allocate new sizes from the copy ---
        new_allocations: list[PageAllocation] = []
        for size in new_sizes:
            num_pages = self.pages_required(size)
            if num_pages > len(new_free):
                return None  # Not enough space
            page_ids = tuple(heapq.heappop(new_free) for _ in range(num_pages))
            allocation = PageAllocation(
                allocator_id=self._allocator_id,
                id=new_next_id,
                page_ids=page_ids,
                logical_length=size,
                allocated_length=num_pages * self._page_size,
            )
            new_allocated[allocation.id] = allocation
            new_next_id += 1
            new_allocations.append(allocation)

        # --- Injected failure hook (before swap) ---
        if _inject_failure_before_commit:
            raise RuntimeError("injected transaction failure before commit")

        # --- Single atomic swap ---
        self._free_pages = new_free
        self._allocated = new_allocated
        self._next_handle_id = new_next_id
        return new_allocations

    def page_spans(
        self, allocation: PageAllocation
    ) -> tuple[tuple[int, int, int], ...]:
        """Return coalesced ``(offset, logical, allocated)`` physical spans."""
        if allocation.allocator_id != self._allocator_id:
            raise ValueError("allocation belongs to a different page allocator")
        existing = self._allocated.get(allocation.id)
        if existing is None or existing != allocation:
            raise ValueError("unknown page allocation")
        runs: list[tuple[int, int]] = []
        for _, group in itertools.groupby(
            enumerate(allocation.page_ids), lambda item: item[1] - item[0]
        ):
            pages = [item[1] for item in group]
            runs.append((pages[0], len(pages)))

        remaining = allocation.logical_length
        spans: list[tuple[int, int, int]] = []
        for first_page, num_pages in runs:
            allocated = num_pages * self._page_size
            logical = min(remaining, allocated)
            spans.append((first_page * self._page_size, logical, allocated))
            remaining -= logical
        assert remaining == 0
        return tuple(spans)
