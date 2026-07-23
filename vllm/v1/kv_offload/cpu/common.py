# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

from vllm.v1.kv_offload.base import BlockIDsLoadStoreSpec, LoadStoreSpec


class CPUOffloadingMetrics:
    STORES_SKIPPED = "vllm:kv_offload_stores_skipped"
    CPU_CACHE_USAGE_PERC = "vllm:kv_offload_cpu_cache_usage_perc"
    CPU_ALLOCATION_SIZE = "vllm:kv_offload_cpu_allocation_size"
    CPU_CACHE_WRITE_USAGE_PERC = "vllm:kv_offload_cpu_cache_write_usage_perc"
    CPU_CACHE_READ_USAGE_PERC = "vllm:kv_offload_cpu_cache_read_usage_perc"


class CPULoadStoreSpec(BlockIDsLoadStoreSpec):
    """
    Spec for loading/storing a KV block to CPU memory.
    """


@dataclass(frozen=True)
class CompactCPUAddressSpan:
    """One physical span backing a logical compact CPU payload."""

    byte_offset: int
    logical_length: int
    allocated_length: int

    def __post_init__(self) -> None:
        if self.byte_offset < 0:
            raise ValueError("span byte_offset must be non-negative")
        if self.logical_length <= 0:
            raise ValueError("span logical_length must be positive")
        if self.allocated_length < self.logical_length:
            raise ValueError("span allocated_length must cover logical_length")


@dataclass(frozen=True)
class CompactCPUAddress:
    """Byte-granularity address of one compact offload block in a CPU extent.

    Each offload *key* (one OffloadKey) has exactly one ``CompactCPUAddress``.

    Fields
    ------
    byte_offset:
        Byte offset from the start of this group's CPU extent (pool base).
    logical_length:
        Real (un-padded) payload bytes for this offload block.
    allocated_length:
        Actually allocated bytes (logical_length + alignment padding).
    group_idx:
        Index of the KV cache group this block belongs to.
    spans:
        Optional physical sub-spans for multi-page allocations.  When
        empty, the address describes a single contiguous extent.
    """

    byte_offset: int
    logical_length: int
    allocated_length: int
    group_idx: int = 0
    spans: tuple[CompactCPUAddressSpan, ...] = ()

    def __post_init__(self) -> None:
        if self.group_idx < 0:
            raise ValueError(f"group_idx must be non-negative, got {self.group_idx}")
        if self.byte_offset < 0:
            raise ValueError(
                f"byte_offset must be non-negative, got {self.byte_offset}"
            )
        if self.logical_length <= 0:
            raise ValueError(
                f"logical_length must be positive, got {self.logical_length}"
            )
        if self.allocated_length < self.logical_length:
            raise ValueError(
                f"allocated_length ({self.allocated_length}) must be >= "
                f"logical_length ({self.logical_length})"
            )
        if self.spans:
            if self.byte_offset != self.spans[0].byte_offset:
                raise ValueError("byte_offset must match the first physical span")
            if sum(span.logical_length for span in self.spans) != self.logical_length:
                raise ValueError(
                    "physical spans must cover the logical payload exactly"
                )

    @property
    def physical_spans(self) -> tuple[CompactCPUAddressSpan, ...]:
        """Return physical spans, falling back to a single synthetic span when
        no explicit spans were provided."""
        if self.spans:
            return self.spans
        return (
            CompactCPUAddressSpan(
                byte_offset=self.byte_offset,
                logical_length=self.logical_length,
                allocated_length=self.allocated_length,
            ),
        )


class CompactCPULoadStoreSpec(LoadStoreSpec):
    """Per-payload-class compact CPU addressing.

    Carries one :class:`CompactCPUAddress` per offload key, co-indexed
    with the keys.  Does not use ``block_ids`` (no fake IDs).
    """

    def __init__(self, compact_addresses: list[CompactCPUAddress]):
        self.compact_addresses: list[CompactCPUAddress] = list(compact_addresses)

    @property
    def addresses(self) -> list[CompactCPUAddress]:
        """Alias for compatibility with worker code that accesses .addresses."""
        return self.compact_addresses

    def __repr__(self) -> str:
        return f"CompactCPULoadStoreSpec({len(self.compact_addresses)} addresses)"
