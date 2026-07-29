# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure descriptor planning for compact CPU KV transfers.

Direction-neutral plan builder that consumes owner objects directly --
ordered ``CanonicalPageMapping`` sequences, explicit per-layer canonical
offsets, GPU offsets, and CPU physical spans -- without dependence on
``GroupCanonicalLayout`` or group layout builders.

The caller derives per-layer geometry; this planner only translates it
into flat uint64 descriptor arrays.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from vllm.utils.math_utils import cdiv
from vllm.v1.kv_offload.cpu.common import CompactCPUAddress, CompactCPUAddressSpan

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import CanonicalPageMapping


@dataclass(frozen=True)
class CompactTransferPlan:
    """Byte pointers and sizes for one compact transfer submission.

    All arrays are immutable NumPy arrays of ``np.uint64``.  The worker
    swaps the semantic role of GPU and CPU pointers for store vs. load.
    """

    gpu_ptrs: np.ndarray
    cpu_ptrs: np.ndarray
    sizes: np.ndarray
    num_cpu_addresses: int

    def __post_init__(self) -> None:
        if self.gpu_ptrs.dtype != np.uint64:
            raise TypeError("gpu_ptrs must be uint64")
        if self.cpu_ptrs.dtype != np.uint64:
            raise TypeError("cpu_ptrs must be uint64")
        if self.sizes.dtype != np.uint64:
            raise TypeError("sizes must be uint64")
        if self.gpu_ptrs.shape != self.cpu_ptrs.shape:
            raise ValueError("gpu_ptrs and cpu_ptrs must have the same shape")
        if self.sizes.shape != self.gpu_ptrs.shape:
            raise ValueError("sizes must match pointer arrays")
        # sizes.size == 0 is valid for zero-descriptor plans
        # (e.g. non-writer store).  Shape consistency is enforced above.
        if self.num_cpu_addresses < 0:
            raise ValueError("num_cpu_addresses must be non-negative")

    @property
    def num_bytes(self) -> int:
        return int(self.sizes.sum())

    @property
    def num_descriptors(self) -> int:
        return int(self.sizes.size)


def _translate_through_spans(
    logical_offset: int,
    size: int,
    spans: tuple[CompactCPUAddressSpan, ...],
) -> list[tuple[int, int]]:
    """Translate a logical fragment range through physical spans.

    Each span covers a contiguous sub-range of the logical address.  This
    maps ``[logical_offset, logical_offset+size)`` through the spans,
    splitting at span boundaries.  Returns ``(physical_byte_offset,
    split_size)`` tuples in logical order.  Each physical offset is
    relative to the CPU region base.

    Raises ``ValueError`` when the range extends beyond the spans.
    """
    remaining = size
    cursor = logical_offset
    results: list[tuple[int, int]] = []

    # Cumulative logical start of each span
    span_logical_start = 0
    for span in spans:
        span_logical_end = span_logical_start + span.logical_length
        if cursor >= span_logical_end:
            span_logical_start = span_logical_end
            continue
        if cursor < span_logical_start:
            raise ValueError(
                f"Logical offset {cursor} falls in gap between spans "
                f"(span {span} starts at logical {span_logical_start})"
            )
        intra_span_offset = cursor - span_logical_start
        available = span_logical_end - cursor
        split_size = min(remaining, available)
        physical_offset = span.byte_offset + intra_span_offset
        results.append((physical_offset, split_size))
        cursor += split_size
        remaining -= split_size
        if remaining == 0:
            break
        span_logical_start = span_logical_end

    if remaining > 0:
        raise ValueError(
            f"Logical range [{logical_offset}, {logical_offset + size}) "
            f"exceeds physical spans (only {cursor - logical_offset} of "
            f"{size} bytes covered)"
        )
    return results


def plan_compact_transfer(
    *,
    gpu_base_ptr: int,
    gpu_row_stride: int,
    cpu_base_ptr: int,
    cpu_region_size: int,
    gpu_block_ids: np.ndarray,
    group_sizes: Sequence[int],
    block_indices: Sequence[int],
    compact_addresses: Sequence[CompactCPUAddress],
    per_group_mappings: Sequence[tuple[CanonicalPageMapping, ...]],
    per_group_canonical_offsets: Sequence[tuple[int, ...]],
    per_group_gpu_offsets: Sequence[tuple[int, ...]],
    blocks_per_chunk: int,
    direction: str = "store",
) -> CompactTransferPlan:
    """Plan compact per-layer copy descriptors using owner objects.

    Each compact address owns one static payload whose per-layer slices are
    laid out consecutively in canonical byte order; each slice contains
    ``blocks_per_chunk`` native GPU sub-blocks.  Partial transfers select
    sub-blocks inside that static layout.

    Parameters
    ----------
    gpu_base_ptr:
        Base GPU pointer for the KV cache allocation.
    gpu_row_stride:
        Byte stride of one GPU block row.
    cpu_base_ptr:
        Base CPU pointer for the compact CPU allocation.
    cpu_region_size:
        Total allocated CPU region size in bytes.
    gpu_block_ids:
        GPU block IDs whose rows must be transferred.
    group_sizes:
        Number of GPU block IDs per group.
    block_indices:
        Per-group partial-chunk offset into the first compact address.
    compact_addresses:
        Compact CPU addresses by group then address index.
    per_group_mappings:
        One tuple of ``CanonicalPageMapping`` per group, in layer order.
        The direction parameter selects which ``runs`` sequence to use
        (``runs`` for both store and load; writer/non-writer filtering via
        ``is_writer(block_id)``).
    per_group_canonical_offsets:
        One tuple of canonical byte offsets per group, one per layer,
        matching the layer order in ``per_group_mappings``.  These are
        the layer's canonical offset within the group's composite
        canonical page.
    per_group_gpu_offsets:
        One tuple of GPU byte offsets per group, one per layer, matching
        the layer order in ``per_group_mappings``.  These are the layer's
        packed byte offset within one GPU block row.
    blocks_per_chunk:
        Native GPU sub-blocks packed into one compact CPU address
        (equivalent to ``block_size_factor`` in the donor).
    direction:
        ``"store"`` (GPU to CPU) or ``"load"`` (CPU to GPU).
        For store direction, layers with ``is_writer(block_id)=False``
        (non-writer mappings) produce zero descriptors.

    Returns
    -------
    CompactTransferPlan with flat per-descriptor arrays.
    """
    if blocks_per_chunk <= 0:
        raise ValueError("blocks_per_chunk must be positive")
    if gpu_row_stride <= 0 or cpu_region_size <= 0:
        raise ValueError("GPU row stride and CPU region size must be positive")
    if len(group_sizes) != len(per_group_mappings):
        raise ValueError("group_sizes must match per_group_mappings")
    if len(block_indices) != len(per_group_mappings):
        raise ValueError("block_indices must match per_group_mappings")
    if len(per_group_canonical_offsets) != len(per_group_mappings):
        raise ValueError("per_group_canonical_offsets must match per_group_mappings")
    if len(per_group_gpu_offsets) != len(per_group_mappings):
        raise ValueError("per_group_gpu_offsets must match per_group_mappings")
    if sum(group_sizes) != len(gpu_block_ids):
        raise ValueError("group_sizes must cover every GPU block ID")
    if direction not in ("store", "load"):
        raise ValueError(f"direction must be 'store' or 'load', got {direction!r}")

    gpu_ptr_values: list[int] = []
    cpu_ptr_values: list[int] = []
    size_values: list[int] = []

    # Group compact addresses by group index
    addresses_by_group: list[list[CompactCPUAddress]] = [[] for _ in per_group_mappings]
    for address in compact_addresses:
        if address.group_idx < 0 or address.group_idx >= len(addresses_by_group):
            raise ValueError(
                f"compact address has out-of-range group index {address.group_idx}"
            )
        addresses_by_group[address.group_idx].append(address)

    gpu_cursor = 0
    for expected_group_idx, (group_size, block_idx) in enumerate(
        zip(group_sizes, block_indices)
    ):
        if group_size < 0 or block_idx < 0:
            raise ValueError("group sizes and block indices must be non-negative")

        group_mappings = per_group_mappings[expected_group_idx]
        group_canonical_offsets = per_group_canonical_offsets[expected_group_idx]
        group_gpu_offsets = per_group_gpu_offsets[expected_group_idx]

        if len(group_mappings) != len(group_canonical_offsets):
            raise ValueError(
                f"Group {expected_group_idx}: per_group_mappings "
                f"({len(group_mappings)}) and per_group_canonical_offsets "
                f"({len(group_canonical_offsets)}) must have same length"
            )
        if len(group_mappings) != len(group_gpu_offsets):
            raise ValueError(
                f"Group {expected_group_idx}: per_group_mappings "
                f"({len(group_mappings)}) and per_group_gpu_offsets "
                f"({len(group_gpu_offsets)}) must have same length"
            )

        # Validate canonical offsets and compute canonical group extent
        # from explicit offsets (allow gaps, reject overlaps/negatives).
        _layer_extents: list[tuple[int, int]] = []
        for _li, (_m, _co) in enumerate(zip(group_mappings, group_canonical_offsets)):
            if _co < 0:
                raise ValueError(
                    f"Group {expected_group_idx} layer {_li}: "
                    f"canonical offset {_co} must be non-negative"
                )
            _layer_extents.append((_co, _co + _m.canonical_page_size_bytes))
        # Sort by offset and check pairwise non-overlap
        _sorted = sorted(_layer_extents)
        for _si in range(1, len(_sorted)):
            if _sorted[_si][0] < _sorted[_si - 1][1]:
                raise ValueError(
                    f"Group {expected_group_idx}: canonical extents "
                    f"[{_sorted[_si - 1][0]}, {_sorted[_si - 1][1]}) "
                    f"and [{_sorted[_si][0]}, {_sorted[_si][1]}) overlap"
                )
        canonical_group_page_size = (
            max(_end for _, _end in _layer_extents) if _layer_extents else 0
        )

        # Compute number of compact CPU blocks needed
        first_sub_block = block_idx % blocks_per_chunk
        num_cpu_blocks = cdiv(first_sub_block + group_size, blocks_per_chunk)
        group_addresses = addresses_by_group[expected_group_idx]
        if len(group_addresses) != num_cpu_blocks:
            raise ValueError(
                f"compact addresses for group {expected_group_idx} "
                f"do not cover the GPU group: expected {num_cpu_blocks}, "
                f"got {len(group_addresses)}"
            )

        for logical_idx in range(group_size):
            compact_idx, sub_idx = divmod(
                first_sub_block + logical_idx, blocks_per_chunk
            )
            address = group_addresses[compact_idx]
            gpu_block_id = int(gpu_block_ids[gpu_cursor + logical_idx])
            if gpu_block_id < 0:
                raise ValueError("GPU block IDs must be non-negative")

            # Validate address logical_length matches expected
            expected_addr_len = canonical_group_page_size * blocks_per_chunk
            if address.logical_length != expected_addr_len:
                raise ValueError(
                    f"compact address logical_length="
                    f"{address.logical_length} does not match expected "
                    f"canonical_group_page_size"
                    f"({canonical_group_page_size}) * "
                    f"blocks_per_chunk({blocks_per_chunk}) = "
                    f"{expected_addr_len} for group {expected_group_idx}"
                )

            spans = address.physical_spans

            # Build descriptors using per-layer mappings
            for layer_idx, (
                mapping,
                can_offset,
                gpu_offset,
            ) in enumerate(
                zip(
                    group_mappings,
                    group_canonical_offsets,
                    group_gpu_offsets,
                )
            ):
                # Select runs based on direction
                if direction == "store":
                    if not mapping.is_writer(gpu_block_id):
                        # Non-writer store mapping -> zero descriptors
                        continue
                    runs = mapping.runs
                else:
                    runs = mapping.runs

                if not runs:
                    continue

                for run in runs:
                    for i in range(run.num_fragments):
                        local_frag_offset = run.local_offset + i * run.local_stride

                        # GPU pointer includes layer's GPU offset
                        gpu_ptr = (
                            gpu_base_ptr
                            + gpu_block_id * gpu_row_stride
                            + gpu_offset
                            + local_frag_offset
                        )

                        # Validate fragment stays within one GPU row
                        frag_start = gpu_offset + local_frag_offset
                        frag_end = frag_start + run.fragment_size
                        if frag_end > gpu_row_stride:
                            raise ValueError(
                                f"Fragment [{frag_start}, {frag_end}) "
                                f"exceeds GPU row stride {gpu_row_stride} "
                                f"for layer {layer_idx} in group "
                                f"{expected_group_idx}"
                            )

                        # Canonical CPU logical offset within the compact
                        # CPU address.
                        can_frag_offset = (
                            run.canonical_offset + i * run.canonical_stride
                        )
                        raw_cpu_logical = (
                            can_offset * blocks_per_chunk
                            + sub_idx * mapping.canonical_page_size_bytes
                            + can_frag_offset
                        )

                        # Validate canonical fragment stays within
                        # canonical sub-page.
                        can_sub_page_end = can_frag_offset + run.fragment_size
                        if can_sub_page_end > mapping.canonical_page_size_bytes:
                            raise ValueError(
                                f"Canonical fragment "
                                f"[{can_frag_offset}, "
                                f"{can_sub_page_end}) exceeds layer "
                                f"{layer_idx} canonical sub-page "
                                f"size "
                                f"{mapping.canonical_page_size_bytes}"
                            )

                        # Translate through physical spans
                        span_segments = _translate_through_spans(
                            raw_cpu_logical,
                            run.fragment_size,
                            spans,
                        )
                        gpu_cursor_advance = 0
                        for (
                            phys_cpu_offset,
                            split_size,
                        ) in span_segments:
                            cpu_ptr = cpu_base_ptr + phys_cpu_offset
                            gpu_ptr_values.append(gpu_ptr + gpu_cursor_advance)
                            cpu_ptr_values.append(cpu_ptr)
                            size_values.append(split_size)
                            gpu_cursor_advance += split_size

        gpu_cursor += group_size

    if gpu_cursor != len(gpu_block_ids):
        raise ValueError("GPU block IDs were not fully consumed")

    # Validate every CPU descriptor falls within the backing region.
    cpu_region_end = cpu_base_ptr + cpu_region_size
    for cpu_ptr, size in zip(cpu_ptr_values, size_values):
        cpu_end = cpu_ptr + size
        if cpu_ptr < cpu_base_ptr or cpu_end > cpu_region_end:
            raise ValueError(
                f"compact CPU descriptor [{cpu_ptr}, {cpu_end}) "
                f"exceeds backing region "
                f"[{cpu_base_ptr}, {cpu_region_end})"
            )

    # Validate all pointer/size values fit in uint64 range before
    # conversion.  numpy.asarray(..., dtype=np.uint64) silently wraps
    # negative or overflowing Python ints, so we must check explicitly.
    _UINT64_MAX = (1 << 64) - 1
    for _i, _v in enumerate(gpu_ptr_values):
        if not (0 <= _v <= _UINT64_MAX):
            raise ValueError(
                f"GPU pointer value {_v} at index {_i} is out of uint64 range"
            )
    for _i, _v in enumerate(cpu_ptr_values):
        if not (0 <= _v <= _UINT64_MAX):
            raise ValueError(
                f"CPU pointer value {_v} at index {_i} is out of uint64 range"
            )
    for _i, _v in enumerate(size_values):
        if not (0 <= _v <= _UINT64_MAX):
            raise ValueError(f"Size value {_v} at index {_i} is out of uint64 range")

    gpu_arr = np.asarray(gpu_ptr_values, dtype=np.uint64)
    cpu_arr = np.asarray(cpu_ptr_values, dtype=np.uint64)
    sz_arr = np.asarray(size_values, dtype=np.uint64)
    gpu_arr.flags.writeable = False
    cpu_arr.flags.writeable = False
    sz_arr.flags.writeable = False
    return CompactTransferPlan(
        gpu_ptrs=gpu_arr,
        cpu_ptrs=cpu_arr,
        sizes=sz_arr,
        num_cpu_addresses=len(compact_addresses),
    )
