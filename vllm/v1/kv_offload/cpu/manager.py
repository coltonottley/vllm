# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict
from collections.abc import Collection, Iterable, Mapping
from typing import Literal

from typing_extensions import override

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.utils.math_utils import round_down
from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    LookupResult,
    Medium,
    OffloadingEvent,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    get_offload_group_idx,
)
from vllm.v1.kv_offload.cpu.common import (
    CompactCPUAddress,
    CompactCPUAddressSpan,
    CompactCPULoadStoreSpec,
    CPULoadStoreSpec,
    CPUOffloadingMetrics,
)
from vllm.v1.kv_offload.cpu.fixed_page_allocator import (
    FixedPageAllocator,
    PageAllocation,
)
from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy
from vllm.v1.kv_offload.cpu.policies.factory import CachePolicyFactory


class CPUOffloadingManager(OffloadingManager):
    """
    An OffloadingManager with a pluggable CachePolicy, resolved by name via
    CachePolicyFactory (built in: "lru", "arc"; external policies can either
    register their own or be loaded out-of-tree via cache_policy_module_path).

    The manager owns all shared logic: ref-counting, event emission,
    block pool management, and the prepare_store/complete_store skeletons.
    Policy-specific block organization and eviction decisions are delegated
    to the CachePolicy implementation.

    Compact mode (activated via :meth:`resolve_compact_mode`) replaces the
    legacy fixed-block pool with a ``FixedPageAllocator`` and returns
    ``CompactCPULoadStoreSpec`` with exact byte-granularity addresses.
    Legacy block-based mode is the default and unchanged when compact mode
    is not activated.
    """

    def __init__(
        self,
        num_blocks: int,
        cache_policy: str = "lru",
        cache_policy_module_path: str | None = None,
        enable_events: bool = False,
        store_threshold: int = 1,
        max_tracker_size: int = 64_000,
    ):
        self.medium: Medium = Medium.CPU
        self._num_blocks: int = num_blocks
        self._num_allocated_blocks: int = 0
        self._free_list: list[int] = []
        self.events: list[OffloadingEvent] | None = [] if enable_events else None
        policy_cls = CachePolicyFactory.get_cache_policy_cls(
            cache_policy, cache_policy_module_path
        )
        self._policy: CachePolicy = policy_cls(cache_capacity=num_blocks)
        # Track the number of blocks in the cache that are evictable. i.e. ref_cnt 0.
        self._num_evictable_cache_blocks: int = 0
        # Track blocks with an in-flight store (ref_cnt -1, not yet completed).
        self._num_write_pending_blocks: int = 0

        self.store_threshold: int = store_threshold
        self.max_tracker_size: int = max_tracker_size
        self.stores_skipped_in_current_batch: int = 0
        self.allocation_sizes_in_current_batch: list[int] = []

        # Number of block references. It is ordered so can evict the LRU entry in O(1).
        self.counts: OrderedDict[OffloadKey, int] | None = (
            OrderedDict() if store_threshold >= 2 else None
        )

        # --- Compact mode (one-way disabled by default) ---
        self._compact_enabled: bool = False
        self._compact_allocator: FixedPageAllocator | None = None
        # group_idx -> byte size for one compact offload block in that group.
        self._compact_key_sizes: dict[int, int] = {}
        self._compact_page_size: int = 0
        self._compact_total_bytes: int = 0
        # key -> PageAllocation (backing storage)
        self._compact_allocation_by_key: dict[OffloadKey, PageAllocation] = {}
        # key -> CompactCPUAddress (byte-level spec)
        self._compact_address_by_key: dict[OffloadKey, CompactCPUAddress] = {}
        # key -> virtual block_id (for BlockStatus)
        self._compact_block_id_by_key: dict[OffloadKey, int] = {}
        # Preferred eviction group indices (compact only).
        self._compact_preferred_eviction_groups: set[int] = set()

        # CPU entries are useful only as complete cross-group replay units.
        # A key may be shared by multiple units; evicting one unit removes only
        # keys no other resident unit still owns.
        self._replay_units: dict[str, set[OffloadKey]] = {}
        self._key_replay_units: dict[OffloadKey, set[str]] = {}

    # --- public activation API ---

    @override
    def enable_compact(
        self,
        total_bytes: int,
        page_size: int = 65536,
        key_sizes: dict[int, int] | None = None,
        preferred_groups: tuple[int, ...] | None = None,
    ) -> None:
        activated = self.resolve_compact_mode(
            enable=True,
            total_bytes=total_bytes,
            page_size=page_size,
            group_payload_bytes=key_sizes,
            preferred_eviction_groups=preferred_groups or (),
        )
        if not activated:
            raise RuntimeError("compact manager activation failed")

    def resolve_compact_mode(
        self,
        *,
        enable: bool,
        total_bytes: int | None = None,
        page_size: int | None = None,
        group_payload_bytes: Mapping[int, int] | None = None,
        preferred_eviction_groups: Collection[int] = (),
    ) -> bool:
        """One-shot compact mode activation on an empty manager.

        Parameters
        ----------
        enable:
            If False, compact mode is not requested; no other parameters
            are used and the manager continues in legacy mode.
        total_bytes:
            Total CPU memory bytes available for compact KV offload.
            Ignored when enable is False. Rounded down to the nearest
            page_size multiple.
        page_size:
            Fixed page size for the underlying FixedPageAllocator.
        group_payload_bytes:
            Map from KV-cache group index to the byte size of one compact
            offload block in that group. Must have at least one entry.
        preferred_eviction_groups:
            Group indices whose entries are preferred for early eviction.

        Returns
        -------
        True if compact mode was successfully activated, False if inputs
        are invalid or state prevents activation.
        """
        if not enable:
            return False
        if self._compact_enabled:
            raise RuntimeError(
                "resolve_compact_mode called when compact mode is already active"
            )
        if self._num_allocated_blocks > 0 or self._num_write_pending_blocks > 0:
            raise RuntimeError(
                "resolve_compact_mode called on a manager with existing state; "
                "activation must happen before any store"
            )
        if not self._policy.is_empty:
            raise RuntimeError(
                "resolve_compact_mode called with non-empty policy; "
                "activation must happen before any stateful operation"
            )
        if not group_payload_bytes:
            return False
        if any(s <= 0 for s in group_payload_bytes.values()):
            return False
        if not total_bytes or total_bytes <= 0:
            return False
        if not page_size or page_size <= 0:
            return False

        # Round budget to page_size boundary.
        actual_budget = round_down(total_bytes, page_size)
        if actual_budget < page_size:
            return False

        self._compact_allocator = FixedPageAllocator(
            total_bytes=actual_budget, page_size=page_size
        )
        self._compact_key_sizes = dict(group_payload_bytes)
        self._compact_page_size = page_size
        self._compact_total_bytes = actual_budget

        # Derive policy capacity from compact negotiated budget.
        min_payload = min(group_payload_bytes.values())
        policy_capacity = actual_budget // min_payload
        if policy_capacity < 1:
            policy_capacity = 1
        # Rebuild policy with compact capacity.
        cls = type(self._policy)
        self._policy = cls(cache_capacity=policy_capacity)
        self._num_blocks = policy_capacity

        self._compact_preferred_eviction_groups = set(preferred_eviction_groups)
        self._compact_enabled = True
        return True

    def _prefer_evict_fn(self, key: OffloadKey) -> bool:
        """Return True if key's group is preferred for eviction."""
        if not self._compact_preferred_eviction_groups:
            return False
        return get_offload_group_idx(key) in self._compact_preferred_eviction_groups

    # --- private helpers ---

    def _get_key_size(self, key: OffloadKey) -> int:
        """Return the compact byte size for a key's group."""
        group_idx = get_offload_group_idx(key)
        try:
            return self._compact_key_sizes[group_idx]
        except KeyError:
            raise KeyError(
                f"no compact key size defined for group index {group_idx} (key={key!r})"
            ) from None

    def _build_compact_address(
        self,
        allocation: PageAllocation,
        key: OffloadKey,
    ) -> CompactCPUAddress:
        """Convert a FixedPageAllocator allocation to a CompactCPUAddress."""
        spans = self._compact_allocator.page_spans(allocation)
        group_idx = get_offload_group_idx(key)
        logical_length = self._get_key_size(key)

        compact_spans = tuple(
            CompactCPUAddressSpan(
                byte_offset=span[0],
                logical_length=min(span[1], logical_length),
                allocated_length=span[2],
            )
            for span in spans
        )
        return CompactCPUAddress(
            byte_offset=compact_spans[0].byte_offset,
            logical_length=logical_length,
            allocated_length=allocation.allocated_length,
            group_idx=group_idx,
            spans=compact_spans,
        )

    # --- block pool ---

    def _get_num_free_blocks(self) -> int:
        if self._compact_enabled:
            # In compact mode, "free blocks" is a virtual count.
            # Each block occupies at least one page, so the number of
            # additional blocks is the number of free pages.
            return self._compact_allocator.free_bytes // self._compact_page_size
        return len(self._free_list) + self._num_blocks - self._num_allocated_blocks

    def _allocate_blocks(self, keys: list[OffloadKey]) -> list[BlockStatus]:
        if self._compact_enabled:
            blocks: list[BlockStatus] = []
            for key in keys:
                size = self._get_key_size(key)
                allocation = self._compact_allocator.allocate(size)
                if allocation is None:
                    # Rollback already-allocated keys
                    for bk, ba in zip(keys[: len(blocks)], blocks):
                        self._compact_allocator.free(
                            self._compact_allocation_by_key.pop(bk, None)
                        )
                        self._compact_address_by_key.pop(bk, None)
                        self._compact_block_id_by_key.pop(bk, None)
                    raise RuntimeError(
                        f"compact allocation failed for key={key!r} "
                        f"size={size} "
                        f"free={self._compact_allocator.free_bytes}"
                    )
                block_id = self._num_allocated_blocks + len(blocks)
                block = BlockStatus(block_id)
                address = self._build_compact_address(allocation, key)
                self._compact_allocation_by_key[key] = allocation
                self._compact_address_by_key[key] = address
                self._compact_block_id_by_key[key] = block_id
                blocks.append(block)
            self._num_allocated_blocks += len(blocks)
            return blocks

        num_fresh = min(len(keys), self._num_blocks - self._num_allocated_blocks)
        num_reused = len(keys) - num_fresh
        assert len(self._free_list) >= num_reused

        # allocate fresh blocks
        blocks: list[BlockStatus] = []
        for _ in range(num_fresh):
            blocks.append(BlockStatus(self._num_allocated_blocks))
            self._num_allocated_blocks += 1

        # allocate reused blocks
        for _ in range(num_reused):
            blocks.append(BlockStatus(self._free_list.pop()))
        return blocks

    def _free_block(self, block: BlockStatus) -> None:
        self._free_list.append(block.block_id)

    def _free_compact_block(self, key: OffloadKey, block: BlockStatus) -> None:
        """Free a compact-allocated block's backing storage."""
        if not self._compact_enabled:
            self._free_block(block)
            return
        allocation = self._compact_allocation_by_key.pop(key, None)
        if allocation is not None:
            self._compact_allocator.free(allocation)
        self._compact_address_by_key.pop(key, None)
        self._compact_block_id_by_key.pop(key, None)

    def _get_load_store_spec(
        self,
        keys: Iterable[OffloadKey],
        blocks: Iterable[BlockStatus],
    ) -> LoadStoreSpec:
        if self._compact_enabled:
            addresses = [self._compact_address_by_key[key] for key in keys]
            return CompactCPULoadStoreSpec(addresses)
        return CPULoadStoreSpec([block.block_id for block in blocks])

    # --- cross-group replay-unit coherence metadata ---

    def _replay_eviction_plan(
        self,
        keys: Collection[OffloadKey],
        protected: set[OffloadKey] | None = None,
    ) -> tuple[list[OffloadKey], set[str]]:
        """Purely plan complete replay units implied by policy candidates.

        ``protected`` is the set of keys that must NOT be evicted.  When
        expanding candidate keys to complete replay units, protected keys
        are excluded from the eviction set and their owning units are
        removed from victim consideration.
        """
        if protected is None:
            protected = set()
        victim_units: set[str] = set()
        pre_metadata_keys: list[OffloadKey] = []
        for key in keys:
            owners = self._key_replay_units.get(key)
            if owners and key not in protected:
                victim_units.add(min(owners))
            elif self._policy.get(key) is not None and key not in protected:
                # Pre-metadata entries remain independently evictable.
                pre_metadata_keys.append(key)
        if pre_metadata_keys:
            return sorted(pre_metadata_keys), set()

        # Filter out victim units whose membership keys are all protected.
        active_victim_units = set()
        for unit in victim_units:
            unit_keys = self._replay_units.get(unit, set())
            unprotected_unit_keys = unit_keys - protected
            if unprotected_unit_keys:
                active_victim_units.add(unit)

        evicted: set[OffloadKey] = set()
        for unit in active_victim_units:
            for key in self._replay_units.get(unit, ()):
                owners = self._key_replay_units.get(key, set())
                # Key is evicted if ALL its owners are being evicted
                # AND the key itself is not protected.
                if key not in protected and not (owners - active_victim_units):
                    evicted.add(key)
        return sorted(evicted), active_victim_units

    def _commit_replay_unit(self, unit: str, keys: Collection[OffloadKey]) -> None:
        resident = {key for key in keys if self._policy.get(key) is not None}
        old = self._replay_units.get(unit, set())
        for key in old - resident:
            owners = self._key_replay_units.get(key)
            if owners is not None:
                owners.discard(unit)
                if not owners:
                    self._key_replay_units.pop(key, None)
        self._replay_units[unit] = resident
        for key in resident:
            self._key_replay_units.setdefault(key, set()).add(unit)

    def _commit_replay_eviction(
        self, keys: Collection[OffloadKey], units: Collection[str]
    ) -> None:
        for unit in units:
            for key in self._replay_units.pop(unit, set()):
                owners = self._key_replay_units.get(key)
                if owners is not None:
                    owners.discard(unit)
                    if not owners:
                        self._key_replay_units.pop(key, None)

    def _remove_key_from_replay_units(self, key: OffloadKey) -> None:
        """Remove a failed key from every owning replay unit.

        Cleans up the key's reverse mapping in ``_key_replay_units`` and
        removes any replay unit that becomes empty.  Preserves unrelated
        shared-key owners in the same unit.
        """
        owners = self._key_replay_units.get(key)
        if owners is None:
            return
        for unit in owners:
            unit_keys = self._replay_units.get(unit)
            if unit_keys is not None:
                unit_keys.discard(key)
                if not unit_keys:
                    self._replay_units.pop(unit, None)
        self._key_replay_units.pop(key, None)

    # --- OffloadingManager interface ---

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        if self.counts is not None:
            if key in self.counts:
                self.counts.move_to_end(key)
                self.counts[key] += 1
            else:
                if len(self.counts) >= self.max_tracker_size:
                    self.counts.popitem(last=False)
                self.counts[key] = 1
        block = self._policy.get(key)
        if block is None:
            return LookupResult.MISS
        if not block.is_ready:
            return LookupResult.HIT_PENDING
        return LookupResult.HIT

    @override
    def prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        blocks = []
        for key in keys:
            block = self._policy.get(key)
            assert block is not None, f"Block {key!r} not found in cache"
            assert block.is_ready, f"Block {key!r} is not ready for reading"
            if block.ref_cnt == 0:
                self._policy.mark_non_evictable(key)
                self._num_evictable_cache_blocks -= 1  # ref_cnt 0 -> 1
                assert self._num_evictable_cache_blocks >= 0
            block.ref_cnt += 1
            blocks.append(block)
        return self._get_load_store_spec(keys, blocks)

    @override
    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        self._policy.touch(keys, req_context)

    @override
    def complete_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        for key in keys:
            block = self._policy.get(key)
            assert block is not None, f"Block {key!r} not found"
            assert block.ref_cnt > 0, f"Block {key!r} ref_cnt is already 0"
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                self._num_evictable_cache_blocks += 1  # ref_cnt 1 -> 0
                self._policy.mark_evictable(key)

    @override
    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        # Compact mode uses byte-granularity FixedPageAllocator with
        # replay-unit closure, atomic replace/rollback for eviction + allocation.
        if self._compact_enabled:
            return self._compact_prepare_store(keys, req_context)

        if self.counts is not None:
            num_keys = len(keys)
            keys = [k for k in keys if self.counts.get(k, 0) >= self.store_threshold]
            self.stores_skipped_in_current_batch += num_keys - len(keys)
        # filter out blocks that are already stored
        keys_to_store = [k for k in keys if self._policy.get(k) is None]

        if not keys_to_store:
            return PrepareStoreOutput(
                keys_to_store=[],
                store_spec=self._get_load_store_spec([], []),
                evicted_keys=[],
            )

        self.allocation_sizes_in_current_batch.append(len(keys_to_store))
        num_blocks_to_evict = len(keys_to_store) - self._get_num_free_blocks()

        to_evict: list[OffloadKey] = []
        if num_blocks_to_evict > 0:
            if num_blocks_to_evict > self._num_evictable_cache_blocks:
                # Eviction will fail.
                return None
            # There is a still a chance for eviction failure as some of the
            # idle blocks might be in the protected list.

            # Blocks from the original input are excluded from eviction candidates:
            # a block that was already stored must remain in the cache after this call.
            protected = set(keys)
            evicted = self._policy.evict(num_blocks_to_evict, protected)
            if evicted is None:
                return None

            # cache-policy removes only idle blocks.
            self._num_evictable_cache_blocks -= len(evicted)
            assert self._num_evictable_cache_blocks >= 0

            for key, block in evicted:
                self._free_block(block)
                to_evict.append(key)

        if to_evict and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=to_evict,
                    medium=self.medium,
                    removed=True,
                )
            )

        blocks = self._allocate_blocks(keys_to_store)
        assert len(blocks) == len(keys_to_store), (
            "Block pool did not allocate the expected number of blocks"
        )

        for key, block in zip(keys_to_store, blocks):
            self._policy.insert(key, block)
        self._num_write_pending_blocks += len(keys_to_store)

        # build store specs for allocated blocks
        store_spec = self._get_load_store_spec(keys_to_store, blocks)

        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=store_spec,
            evicted_keys=to_evict,
        )

    def _compact_prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        """Compact-mode prepare_store with FixedPageAllocator, cross-group
        replay-unit victim closure, and atomic replace/rollback.

        Semantics match the legacy path, but:
        - Eviction selects complete cross-group replay units when possible.
        - Allocator transaction (free victims + allocate new) is atomic and
          happens BEFORE any counter, event, policy, replay, address, or
          block-id mutation.
        - Policy replay metadata, events, and counters are committed only
          after allocator success.
        - Any failure before commit leaves all state unchanged.
        """
        skipped_count = 0
        if self.counts is not None:
            num_keys = len(keys)
            keys = [k for k in keys if self.counts.get(k, 0) >= self.store_threshold]
            skipped_count = num_keys - len(keys)

        keys_to_store = [k for k in keys if self._policy.get(k) is None]
        if not keys_to_store:
            return PrepareStoreOutput(
                keys_to_store=[],
                store_spec=self._get_load_store_spec([], []),
                evicted_keys=[],
            )

        new_sizes = [self._get_key_size(k) for k in keys_to_store]
        bytes_needed = sum(new_sizes) - self._compact_allocator.free_bytes

        to_evict: list[OffloadKey] = []
        victim_allocs: list[PageAllocation] = []
        evicted_victim_units: set[str] = set()

        if bytes_needed > 0:
            if self._num_evictable_cache_blocks == 0:
                return None

            # Replay unit: incoming keys and the call-scoped store_replay_unit
            # are protected (not evictable during this call).
            protected: set[OffloadKey] = set(keys)
            protected.update(req_context.store_replay_unit)

            def _can_fit(
                candidates: list[tuple[OffloadKey, "BlockStatus"]],
            ) -> bool:
                """Predicate: do candidates free enough bytes?"""
                freed = 0
                for key, _ in candidates:
                    alloc = self._compact_allocation_by_key.get(key)
                    if alloc is not None:
                        freed += alloc.allocated_length
                    if freed >= bytes_needed:
                        return True
                return freed >= bytes_needed

            prefer_fn = None
            if self._compact_preferred_eviction_groups:
                pref_groups = self._compact_preferred_eviction_groups
                prefer_fn = lambda k: get_offload_group_idx(k) in pref_groups

            # Non-mutating candidate selection from policy.
            candidates = self._policy.select_evict_until(
                _can_fit, protected, prefer_evict=prefer_fn
            )
            if candidates is None:
                return None

            # Expand candidates to complete cross-group replay units
            # WITHOUT mutating policy or replay metadata.
            candidate_keys = [k for k, _ in candidates]
            to_evict, evicted_victim_units = self._replay_eviction_plan(
                candidate_keys, protected=protected
            )

            # Gather victim PageAllocations WITHOUT mutation.
            for k in to_evict:
                alloc = self._compact_allocation_by_key.get(k)
                block = self._policy.get(k)
                if alloc is None or block is None:
                    return None
                if block.ref_cnt != 0:
                    return None
                victim_allocs.append(alloc)

        # --- Atomic replace: call BEFORE any counter/event/policy/replay/
        #     address/block-id mutation ---
        try:
            new_allocations = self._compact_allocator.atomic_replace(
                frees=victim_allocs,
                new_sizes=new_sizes,
            )
        except Exception:
            # Exception before any commit: state is byte/structurally unchanged.
            return None

        if new_allocations is None:
            # None before any commit: state is byte/structurally unchanged.
            return None

        # --- SUCCESS: commit metadata/policy/replay/counters once ---

        self.stores_skipped_in_current_batch += skipped_count
        self.allocation_sizes_in_current_batch.append(len(keys_to_store))

        # Update counters.
        self._num_evictable_cache_blocks -= len(to_evict)
        assert self._num_evictable_cache_blocks >= 0

        # Remove victim tracking metadata.
        for k in to_evict:
            self._compact_allocation_by_key.pop(k, None)
            self._compact_address_by_key.pop(k, None)
            self._compact_block_id_by_key.pop(k, None)

        # Build new BlockStatus and track addresses.
        blocks: list[BlockStatus] = []
        for i, key in enumerate(keys_to_store):
            allocation = new_allocations[i]
            block_id = self._num_allocated_blocks + len(blocks)
            block = BlockStatus(block_id)
            address = self._build_compact_address(allocation, key)
            self._compact_allocation_by_key[key] = allocation
            self._compact_address_by_key[key] = address
            self._compact_block_id_by_key[key] = block_id
            blocks.append(block)

        self._num_allocated_blocks += len(blocks)

        # --- Policy mutation begins only after allocator commit ---

        # Remove evicted keys from policy.
        for key in to_evict:
            existing = self._policy.get(key)
            if existing is not None:
                self._policy.remove(key)
        self._commit_replay_eviction(to_evict, evicted_victim_units)

        # Insert new entries into policy.
        for key, block in zip(keys_to_store, blocks):
            self._policy.insert(key, block)
        self._num_write_pending_blocks += len(keys_to_store)

        # Record replay unit ownership for cross-group coherence.
        if req_context.store_replay_unit:
            self._commit_replay_unit(req_context.req_id, req_context.store_replay_unit)

        store_spec = self._get_load_store_spec(keys_to_store, blocks)

        # Emit one eviction event after commit.
        if to_evict and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=to_evict,
                    medium=self.medium,
                    removed=True,
                )
            )

        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=store_spec,
            evicted_keys=to_evict,
        )

    @override
    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        stored_keys: list[OffloadKey] = []

        if success:
            for key in keys:
                block = self._policy.get(key)
                if block is not None and not block.is_ready:
                    block.ref_cnt = 0
                    self._num_write_pending_blocks -= 1
                    self._num_evictable_cache_blocks += 1
                    self._policy.mark_evictable(key)
                    stored_keys.append(key)
        else:
            for key in keys:
                block = self._policy.get(key)
                if block is not None and not block.is_ready:
                    self._num_write_pending_blocks -= 1
                    self._policy.remove(key)
                    self._free_compact_block(key, block)
                    self._remove_key_from_replay_units(key)

        if stored_keys and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=stored_keys,
                    medium=self.medium,
                    removed=False,
                )
            )

    @override
    def reset_cache(self) -> None:
        # Clear ALL blocks unconditionally.  Preserves compact mode
        # selection (one-way), but clears all data.
        self._policy.clear()
        self._num_evictable_cache_blocks = 0
        self._num_write_pending_blocks = 0

        if self._compact_enabled:
            self._compact_allocator.reset()
            self._compact_allocation_by_key.clear()
            self._compact_address_by_key.clear()
            self._compact_block_id_by_key.clear()
            self._replay_units.clear()
            self._key_replay_units.clear()

        self._free_list.clear()
        self._num_allocated_blocks = 0

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()

    def get_stats(self) -> OffloadingConnectorStats | None:
        stats = OffloadingConnectorStats()

        if self._compact_enabled:
            alloc = self._compact_allocator
            usage = (
                alloc.used_bytes / alloc.total_bytes if alloc.total_bytes > 0 else 0.0
            )
        else:
            # Compute cache usage from legacy block pool.
            num_used = (
                self._num_allocated_blocks
                - len(self._free_list)
                - self._num_evictable_cache_blocks
            )
            usage = num_used / self._num_blocks if self._num_blocks > 0 else 0.0

        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC, usage)

        for allocation_size in self.allocation_sizes_in_current_batch:
            stats.observe_histogram(
                CPUOffloadingMetrics.CPU_ALLOCATION_SIZE, allocation_size
            )
        self.allocation_sizes_in_current_batch.clear()

        if self._compact_enabled:
            alloc = self._compact_allocator
            total_virtual_blocks = alloc.total_bytes // self._compact_page_size
        else:
            total_virtual_blocks = self._num_blocks

        write_usage = (
            self._num_write_pending_blocks / total_virtual_blocks
            if total_virtual_blocks > 0
            else 0.0
        )
        read_usage = max(usage - write_usage, 0.0)
        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_WRITE_USAGE_PERC, write_usage)
        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_READ_USAGE_PERC, read_usage)

        if self.store_threshold >= 2:
            stats.increase_counter(
                CPUOffloadingMetrics.STORES_SKIPPED,
                self.stores_skipped_in_current_batch,
            )
            self.stores_skipped_in_current_batch = 0

        return stats
