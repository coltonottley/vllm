# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict
from collections.abc import Callable, Iterable

from typing_extensions import override

from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy


class ARCCachePolicy(CachePolicy):
    """
    ARC (Adaptive Replacement Cache) cache policy.

    Data Structures:
        T1: Recent cache containing blocks accessed once.
        T2: Frequent cache containing blocks accessed multiple times.
        B1/B2: Ghost lists tracking recently evicted blocks from T1/T2.
        target_t1_size: Adaptive target size for the T1 partition.

    Algorithm Flow:
        1. Cache lookup (lookup):
           Searches T1 and T2 for block hashes and counts consecutive hits
           until a miss or non-ready block is encountered.

        2. Cache touch (touch) - Adaptive Learning:
           For each key (in reverse order):
           - If in T1: Move to T2 (promotion from recent to frequent).
           - If in T2: Move to MRU position (end of queue).
           - If in B1 ghost list: Increase target_t1_size.
           - If in B2 ghost list: Decrease target_t1_size.

        3. Block eviction (evict) - Adaptive Replacement:
           Determines eviction source based on adaptive target:
           - If T1 size >= target_t1_size: Evict from T1, add to B1.
           - Otherwise: Evict from T2, add to B2.
           Finally, bound each ghost list size.

        4. Block insertion (insert):
           New blocks are always inserted into T1 and removed from B1/B2 if
           present. Blocks may later be promoted to T2 during touch operations.

    Adaptive Behavior:
        The algorithm self-tunes the recency vs. frequency trade-off:
        - B1 hit: Recent access patterns matter more → increase T1.
        - B2 hit: Frequent access patterns matter more → decrease T1.
    """

    def __init__(self, cache_capacity: int):
        super().__init__(cache_capacity)
        self.target_t1_size: float = 0.0
        self.t1: OrderedDict[OffloadKey, BlockStatus] = OrderedDict()
        self.t2: OrderedDict[OffloadKey, BlockStatus] = OrderedDict()
        # key -> None (only care about presence)
        self.b1: OrderedDict[OffloadKey, None] = OrderedDict()
        self.b2: OrderedDict[OffloadKey, None] = OrderedDict()

    @override
    def get(self, key: OffloadKey) -> BlockStatus | None:
        return self.t1.get(key) or self.t2.get(key)

    @override
    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        self.t1[key] = block
        self.b1.pop(key, None)
        self.b2.pop(key, None)

    @override
    def remove(self, key: OffloadKey) -> None:
        if self.t1.pop(key, None) is None:
            self.t2.pop(key, None)

    @override
    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        for key in reversed(list(keys)):
            if key in self.t1:
                block = self.t1.pop(key)
                if not block.is_ready:
                    # block was just prepared to be stored, not really touched
                    # twice — keep it in T1 and mark as most recently used
                    self.t1[key] = block
                else:
                    self.t2[key] = block

            elif key in self.t2:
                self.t2.move_to_end(key)

            elif key in self.b1:
                delta = max(1, len(self.b2) / len(self.b1))
                self.target_t1_size = min(
                    self.target_t1_size + delta, self.cache_capacity
                )
                # move to MRU position (end) to keep it fresh in the ghost list
                self.b1.move_to_end(key)

            elif key in self.b2:
                delta = max(1, len(self.b1) / len(self.b2))
                self.target_t1_size = max(self.target_t1_size - delta, 0)
                # move to MRU position (end) to keep it fresh in the ghost list
                self.b2.move_to_end(key)

    @override
    def clear(self) -> None:
        self.t1.clear()
        self.t2.clear()
        self.b1.clear()
        self.b2.clear()
        self.target_t1_size = 0.0

    @property
    @override
    def is_empty(self) -> bool:
        return len(self.t1) == 0 and len(self.t2) == 0

    @override
    def select_evict_until(
        self,
        can_fit: "Callable[[list[tuple[OffloadKey, BlockStatus]]], bool]",
        protected: set[OffloadKey],
        prefer_evict: "Callable[[OffloadKey], bool] | None" = None,
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        """Non-mutating oldest-first candidate selection.

        Walks T1 then T2 (or T2 when target_t1_size is exceeded) in
        insertion order. Within each partition, preferred candidates
        are offered before normal ones. Preferred keys NEVER appear
        again in the normal pass. Returns candidates if
        can_fit returns True; does NOT mutate policy state.
        """
        candidates: list[tuple[OffloadKey, BlockStatus]] = []
        already_selected: set[OffloadKey] = set()

        def _walk(
            source: OrderedDict[OffloadKey, BlockStatus],
            pred: "Callable[[OffloadKey], bool] | None",
        ) -> bool:
            for key, block in source.items():
                if block.ref_cnt != 0 or key in protected:
                    continue
                if pred is not None and not pred(key):
                    continue
                if key in already_selected:
                    continue
                candidates.append((key, block))
                already_selected.add(key)
                if can_fit(candidates):
                    return True
            return False

        # Walk T1 first (or T2 when target exceeded), with preferred pass.
        if len(self.t1) >= int(self.target_t1_size):
            if prefer_evict is not None and _walk(self.t1, prefer_evict):
                return candidates
            if _walk(self.t1, None):
                return candidates
        if prefer_evict is not None and _walk(self.t2, prefer_evict):
            return candidates
        if _walk(self.t2, None):
            return candidates
        return None

    @override
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        if n == 0:
            return []

        # Collect candidates atomically: simulate T1 size changes as we select,
        # but do not modify actual data structures until all n are found.
        candidates: list[
            tuple[OffloadKey, BlockStatus, bool]
        ] = []  # (key, block, from_t1)
        already_selected: set[OffloadKey] = set()
        virtual_t1_size = len(self.t1)

        for _ in range(n):
            candidate: tuple[OffloadKey, BlockStatus, bool] | None = None

            if virtual_t1_size >= int(self.target_t1_size):
                for key, block in self.t1.items():
                    if (
                        block.ref_cnt == 0
                        and key not in protected
                        and key not in already_selected
                    ):
                        candidate = (key, block, True)
                        virtual_t1_size -= 1
                        break

            if candidate is None:
                for key, block in self.t2.items():
                    if (
                        block.ref_cnt == 0
                        and key not in protected
                        and key not in already_selected
                    ):
                        candidate = (key, block, False)
                        break
                if candidate is None:
                    return None

            candidates.append(candidate)
            already_selected.add(candidate[0])

        # Apply all evictions now that we know n candidates exist.
        result: list[tuple[OffloadKey, BlockStatus]] = []
        for key, block, from_t1 in candidates:
            if from_t1:
                del self.t1[key]
                self.b1[key] = None
            else:
                del self.t2[key]
                self.b2[key] = None
            result.append((key, block))

        # Trim ghost lists to cache_capacity.
        for ghost in (self.b1, self.b2):
            for _ in range(len(ghost) - self.cache_capacity):
                ghost.popitem(last=False)

        return result
