# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.v1.kv_offload.base import BlockIDsLoadStoreSpec, LoadStoreSpec

if TYPE_CHECKING:
    import torch

    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.kv_offload.base import CanonicalPageMapping


@dataclass(frozen=True)
class CompactLayerGeometry:
    """Per-layer compact geometry derived from real runtime state.

    Frozen transport/planner input only.  Carries the owner mapping,
    byte extents, and GPU offset for one layer in a compact-enabled
    group.  Not a shadow layout object graph.
    """

    layer_name: str
    mapping: "CanonicalPageMapping"
    local_page_size_bytes: int
    canonical_page_size_bytes: int
    canonical_offset: int
    gpu_offset_bytes: int

    def __post_init__(self) -> None:
        if self.canonical_offset < 0:
            raise ValueError(
                f"canonical_offset must be non-negative, got {self.canonical_offset}"
            )
        if self.local_page_size_bytes <= 0:
            raise ValueError(
                f"local_page_size_bytes must be positive, got "
                f"{self.local_page_size_bytes}"
            )
        if self.canonical_page_size_bytes <= 0:
            raise ValueError(
                f"canonical_page_size_bytes must be positive, got "
                f"{self.canonical_page_size_bytes}"
            )
        if self.gpu_offset_bytes < 0:
            raise ValueError(
                f"gpu_offset_bytes must be non-negative, got {self.gpu_offset_bytes}"
            )


@dataclass(frozen=True)
class CompactGroupGeometry:
    """Per-group compact geometry: ordered layers with row stride and extents.

    Frozen transport/planner input only.  A present group must be complete,
    ordered, and certified by construction (no ``certified`` boolean).
    ``parallel_invariant`` is True when every layer's mapping is
    parallel-invariant (the canonical bytes are identical under any parallel
    configuration with the same block span).
    """

    layers: tuple[CompactLayerGeometry, ...]
    gpu_row_stride: int
    local_extent: int
    canonical_extent: int
    parallel_invariant: bool

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("layers tuple must be non-empty")
        if self.gpu_row_stride <= 0:
            raise ValueError(
                f"gpu_row_stride must be positive, got {self.gpu_row_stride}"
            )
        if self.local_extent <= 0:
            raise ValueError(f"local_extent must be positive, got {self.local_extent}")
        if self.canonical_extent <= 0:
            raise ValueError(
                f"canonical_extent must be positive, got {self.canonical_extent}"
            )


def derive_compact_group_geometry(
    kv_cache_config: "KVCacheConfig",
    mappings: dict[str, "CanonicalPageMapping"],
    kv_caches: dict[str, "torch.Tensor"],
    layer_is_packed: dict[str, bool],
) -> tuple[CompactGroupGeometry | None, ...]:
    """Derive per-group compact geometry.  One entry per group; ``None``
    means incomplete or nonpacked (not yet supported).  Present groups
    certified by construction.  Contradictions raise ``ValueError``."""
    import torch

    from vllm.v1.kv_cache_interface import (
        AttentionSpec,
        KVCacheTensor,
        UniformTypeKVCacheSpecs,
    )

    # layer → KVCacheTensor; reject ambiguity via None sentinel.
    t_by_layer: dict[str, KVCacheTensor | None] = {}
    for tensor in kv_cache_config.kv_cache_tensors:
        for ln in tensor.shared_by:
            t_by_layer[ln] = None if ln in t_by_layer else tensor

    result: list[CompactGroupGeometry | None] = []
    for group in kv_cache_config.kv_cache_groups:
        gspec = group.kv_cache_spec
        per_specs = (
            gspec.kv_cache_specs if isinstance(gspec, UniformTypeKVCacheSpecs) else {}
        )

        # Nonpacked groups are not yet an honest multi-layer representation.
        # This foundation targets hardware-proven packed DSV4.
        if any(not layer_is_packed.get(ln, False) for ln in group.layer_names):
            result.append(None)
            continue

        geoms: list[CompactLayerGeometry] = []
        coff = 0
        bad = False
        for ln in group.layer_names:
            spec = per_specs.get(ln, gspec)
            if not isinstance(spec, AttentionSpec):
                bad = True
                break
            mapping = mappings.get(ln)
            if mapping is None:
                bad = True
                break
            rt = kv_caches.get(ln)
            if rt is None or not isinstance(rt, torch.Tensor):
                bad = True
                break
            kv_t = t_by_layer.get(ln)
            if kv_t is None:
                bad = True
                break

            gpu_off = kv_t.offset
            roff = rt.storage_offset() * rt.element_size()
            if roff != kv_t.offset:
                raise ValueError(
                    f"Packed storage_offset {roff} != KVCacheTensor.offset "
                    f"{kv_t.offset} for {ln!r}"
                )

            geoms.append(
                CompactLayerGeometry(
                    layer_name=ln,
                    mapping=mapping,
                    local_page_size_bytes=mapping.local_page_size_bytes,
                    canonical_page_size_bytes=mapping.canonical_page_size_bytes,
                    canonical_offset=coff,
                    gpu_offset_bytes=gpu_off,
                )
            )
            coff += mapping.canonical_page_size_bytes

        if bad or not geoms:
            result.append(None)
            continue

        kvt = t_by_layer.get(group.layer_names[0])
        assert isinstance(kvt, KVCacheTensor)
        stride = kvt.block_stride
        for ln in group.layer_names[1:]:
            kv2 = t_by_layer.get(ln)
            if isinstance(kv2, KVCacheTensor) and kv2.block_stride != stride:
                raise ValueError(
                    f"Inconsistent packed block_stride for {ln!r}: "
                    f"{kv2.block_stride} != {stride}"
                )

        # Validate byte spans within [0, block_stride), non-overlapping.
        sl = sorted(geoms, key=lambda x: (x.gpu_offset_bytes, x.layer_name))
        for i, g in enumerate(sl):
            e = g.gpu_offset_bytes + g.local_page_size_bytes
            if e > stride:
                raise ValueError(
                    f"Layer span [{g.gpu_offset_bytes},{e})"
                    f" exceeds block_stride={stride}"
                )
            if i:
                pe = sl[i - 1].gpu_offset_bytes + sl[i - 1].local_page_size_bytes
                if pe > g.gpu_offset_bytes:
                    raise ValueError(
                        f"Overlap: {sl[i - 1].layer_name}@{pe}>"
                        f"{g.layer_name}@{g.gpu_offset_bytes}"
                    )

        result.append(
            CompactGroupGeometry(
                layers=tuple(geoms),
                gpu_row_stride=stride,
                local_extent=sum(g.local_page_size_bytes for g in geoms),
                canonical_extent=sum(g.canonical_page_size_bytes for g in geoms),
                parallel_invariant=all(g.mapping.parallel_invariant for g in geoms),
            )
        )

    return tuple(result)


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
