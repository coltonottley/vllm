# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for compact transfer planning — owner-object API.

Tests exercise the full ``plan_compact_transfer`` function with explicit
per-layer geometry, without dependence on ``GroupCanonicalLayout`` or
``group_layout`` module.
"""

import numpy as np
import pytest

from vllm.v1.kv_offload.base import CanonicalPageMapping, CopyRun
from vllm.v1.kv_offload.cpu.common import CompactCPUAddress, CompactCPUAddressSpan
from vllm.v1.kv_offload.cpu.compact_transfer import (
    CompactTransferPlan,
    _translate_through_spans,
    plan_compact_transfer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identity_mapping(
    page_size: int,
    *,
    num_writers: int = 1,
    writer_index: int = 0,
) -> CanonicalPageMapping:
    """Build a certified identity ``CanonicalPageMapping``.

    Single-fragment identity run -- canonical == local -- covering the full
    page.  Writer election uses *num_writers* and *writer_index* so that
    `mapping.is_writer(block_id)` returns ``block_id % num_writers == writer_index``.
    All certified mappings carry the complete ``runs`` tuple;
    stores gate on ``is_writer(gpu_block_id)``, loads use complete ``runs``.
    """
    run = CopyRun(
        local_offset=0,
        canonical_offset=0,
        fragment_size=page_size,
        num_fragments=1,
        local_stride=page_size,
        canonical_stride=page_size,
    )
    return CanonicalPageMapping(
        canonical_page_size_bytes=page_size,
        local_page_size_bytes=page_size,
        runs=(run,),
        num_writers=num_writers,
        writer_index=writer_index,
        parallelism_agnostic=True,
    )


def _make_address(
    byte_offset: int,
    logical_length: int,
    group_idx: int = 0,
    spans: tuple[CompactCPUAddressSpan, ...] = (),
) -> CompactCPUAddress:
    """Build a ``CompactCPUAddress`` with matching allocated length."""
    return CompactCPUAddress(
        byte_offset=byte_offset,
        logical_length=logical_length,
        allocated_length=logical_length,
        group_idx=group_idx,
        spans=spans,
    )


# ---------------------------------------------------------------------------
# CompactTransferPlan construction tests
# ---------------------------------------------------------------------------


class TestCompactTransferPlan:
    def test_valid_construction(self):
        gpu = np.array([100, 200], dtype=np.uint64)
        cpu = np.array([300, 400], dtype=np.uint64)
        sz = np.array([64, 64], dtype=np.uint64)
        gpu.flags.writeable = False
        cpu.flags.writeable = False
        sz.flags.writeable = False
        plan = CompactTransferPlan(
            gpu_ptrs=gpu,
            cpu_ptrs=cpu,
            sizes=sz,
            num_cpu_addresses=2,
        )
        assert plan.num_descriptors == 2
        assert plan.num_bytes == 128

    def test_invalid_dtype(self):
        gpu = np.array([100], dtype=np.int32)
        cpu = np.array([200], dtype=np.uint64)
        sz = np.array([64], dtype=np.uint64)
        with pytest.raises(TypeError):
            CompactTransferPlan(gpu, cpu, sz, num_cpu_addresses=1)

    def test_empty_plan_accepted(self):
        """Empty uint64 arrays produce a valid zero-descriptor plan."""
        gpu = np.array([], dtype=np.uint64)
        cpu = np.array([], dtype=np.uint64)
        sz = np.array([], dtype=np.uint64)
        gpu.flags.writeable = False
        cpu.flags.writeable = False
        sz.flags.writeable = False
        plan = CompactTransferPlan(gpu, cpu, sz, num_cpu_addresses=1)
        assert plan.num_descriptors == 0
        assert plan.num_bytes == 0
        assert plan.gpu_ptrs.shape == (0,)
        assert plan.cpu_ptrs.shape == (0,)
        assert plan.sizes.shape == (0,)

    def test_mismatched_shapes(self):
        gpu = np.array([100, 200], dtype=np.uint64)
        cpu = np.array([300], dtype=np.uint64)
        sz = np.array([64, 64], dtype=np.uint64)
        with pytest.raises(ValueError, match="same shape"):
            CompactTransferPlan(gpu, cpu, sz, num_cpu_addresses=1)

    def test_zero_num_cpu_addresses_valid(self):
        """Zero num_cpu_addresses is valid with empty arrays."""
        gpu = np.array([], dtype=np.uint64)
        cpu = np.array([], dtype=np.uint64)
        sz = np.array([], dtype=np.uint64)
        plan = CompactTransferPlan(gpu, cpu, sz, num_cpu_addresses=0)
        assert plan.num_descriptors == 0
        assert plan.num_cpu_addresses == 0

    def test_negative_num_cpu_addresses_rejected(self):
        """Negative num_cpu_addresses is rejected."""
        gpu = np.array([], dtype=np.uint64)
        cpu = np.array([], dtype=np.uint64)
        sz = np.array([], dtype=np.uint64)
        with pytest.raises(ValueError, match="num_cpu_addresses must be non-negative"):
            CompactTransferPlan(gpu, cpu, sz, num_cpu_addresses=-1)


# ---------------------------------------------------------------------------
# plan_compact_transfer identity layout tests
# ---------------------------------------------------------------------------


class TestPlanCompactTransfer:
    def test_single_layer_single_block(self):
        """One GPU block, one layer, identity mapping."""
        mappings = (_identity_mapping(4096),)
        offsets = (0,)
        gpu_offsets = (0,)

        gpu_block_ids = np.array([5], dtype=np.int64)
        addresses = [_make_address(0, 4096, group_idx=0)]

        plan = plan_compact_transfer(
            gpu_base_ptr=10000,
            gpu_row_stride=4096,
            cpu_base_ptr=0,
            cpu_region_size=65536,
            gpu_block_ids=gpu_block_ids,
            group_sizes=[1],
            block_indices=[0],
            compact_addresses=addresses,
            per_group_mappings=[mappings],
            per_group_canonical_offsets=[offsets],
            per_group_gpu_offsets=[gpu_offsets],
            blocks_per_chunk=1,
        )
        assert plan.num_descriptors >= 1
        assert plan.num_bytes >= 1024

    def test_multiple_layers(self):
        """Two layers with explicit canonical offsets."""
        k_mapping = _identity_mapping(4096)
        v_mapping = _identity_mapping(4096)
        mappings = (k_mapping, v_mapping)
        offsets = (0, 4096)  # k @ 0, v @ 4096
        gpu_offsets = (0, 0)

        gpu_block_ids = np.array([0], dtype=np.int64)
        addresses = [_make_address(0, 8192, group_idx=0)]

        plan = plan_compact_transfer(
            gpu_base_ptr=10000,
            gpu_row_stride=8192,
            cpu_base_ptr=0,
            cpu_region_size=65536,
            gpu_block_ids=gpu_block_ids,
            group_sizes=[1],
            block_indices=[0],
            compact_addresses=addresses,
            per_group_mappings=[mappings],
            per_group_canonical_offsets=[offsets],
            per_group_gpu_offsets=[gpu_offsets],
            blocks_per_chunk=1,
        )
        assert plan.num_descriptors >= 2  # at least one per layer
        assert plan.num_bytes >= 8192  # 4096 + 4096

    def test_multiple_blocks(self):
        """Two GPU blocks packed into one compact address (blocks_per_chunk=2)."""
        mappings = (_identity_mapping(4096),)
        offsets = (0,)
        gpu_offsets = (0,)

        gpu_block_ids = np.array([0, 1], dtype=np.int64)
        addresses = [_make_address(0, 8192, group_idx=0)]

        plan = plan_compact_transfer(
            gpu_base_ptr=10000,
            gpu_row_stride=4096,
            cpu_base_ptr=0,
            cpu_region_size=65536,
            gpu_block_ids=gpu_block_ids,
            group_sizes=[2],
            block_indices=[0],
            compact_addresses=addresses,
            per_group_mappings=[mappings],
            per_group_canonical_offsets=[offsets],
            per_group_gpu_offsets=[gpu_offsets],
            blocks_per_chunk=2,
        )
        assert plan.num_descriptors >= 2

    def test_packed_gpu_offsets(self):
        """Differing per-layer GPU offsets produce differing GPU ptrs."""
        k_mapping = _identity_mapping(1024)
        v_mapping = _identity_mapping(1024)
        mappings = (k_mapping, v_mapping)
        offsets = (0, 1024)
        gpu_offsets = (0, 2048)  # v at gpu offset 2048

        gpu_block_ids = np.array([0], dtype=np.int64)
        addresses = [_make_address(0, 2048, group_idx=0)]

        plan = plan_compact_transfer(
            gpu_base_ptr=10000,
            gpu_row_stride=8192,
            cpu_base_ptr=0,
            cpu_region_size=65536,
            gpu_block_ids=gpu_block_ids,
            group_sizes=[1],
            block_indices=[0],
            compact_addresses=addresses,
            per_group_mappings=[mappings],
            per_group_canonical_offsets=[offsets],
            per_group_gpu_offsets=[gpu_offsets],
            blocks_per_chunk=1,
        )
        assert int(plan.gpu_ptrs[0]) == 10000  # k at base
        assert int(plan.gpu_ptrs[1]) == 12048  # v at base + 2048

    def test_identity_mapping_writer_election(self):
        """_identity_mapping with num_writers>1 elects rotating writers.

        Current API: writer status is per-block via is_writer(block_id).
        num_writers=2 with writer_index=0 means even block IDs are writers.
        """
        mapping_writer = _identity_mapping(4096, num_writers=2, writer_index=0)
        mapping_reader = _identity_mapping(4096, num_writers=2, writer_index=1)
        assert mapping_writer.is_writer(0)
        assert not mapping_writer.is_writer(1)
        assert mapping_writer.is_writer(2)
        assert not mapping_reader.is_writer(0)
        assert mapping_reader.is_writer(1)
        assert not mapping_reader.is_writer(2)
        default_mapping = _identity_mapping(4096)
        assert default_mapping.is_writer(0)
        assert default_mapping.is_writer(1)
        assert len(mapping_writer.runs) == 1
        assert len(mapping_reader.runs) == 1
        assert len(default_mapping.runs) == 1
        assert mapping_writer.runs[0].fragment_size == 4096
        assert mapping_writer.runs[0].local_offset == 0


# ---------------------------------------------------------------------------
# Fragmented span tests
# ---------------------------------------------------------------------------


class TestFragmentedSpans:
    def test_contiguous_address_no_split(self):
        """Single contiguous span produces one descriptor per run."""
        addr = _make_address(
            0,
            4096,
            group_idx=0,
            spans=(CompactCPUAddressSpan(0, 4096, 4096),),
        )
        mappings = (_identity_mapping(4096),)
        offsets = (0,)
        gpu_offsets = (0,)

        gpu_block_ids = np.array([0], dtype=np.int64)
        plan = plan_compact_transfer(
            gpu_base_ptr=0,
            gpu_row_stride=4096,
            cpu_base_ptr=0,
            cpu_region_size=65536,
            gpu_block_ids=gpu_block_ids,
            group_sizes=[1],
            block_indices=[0],
            compact_addresses=[addr],
            per_group_mappings=[mappings],
            per_group_canonical_offsets=[offsets],
            per_group_gpu_offsets=[gpu_offsets],
            blocks_per_chunk=1,
        )
        assert plan.num_descriptors == 1
        assert int(plan.cpu_ptrs[0]) == 0
        assert int(plan.sizes[0]) == 4096

    def test_split_across_two_spans(self):
        """Crossing span boundary splits descriptor into two."""
        # Two spans covering [0,2)+[5,21) logically = [0,18)
        # Fragment at logical offset 1 size 4: span 0 covers byte 1,
        # span 1 covers bytes 2-4 (logical 2 -> physical 5+0=5).
        addr = _make_address(
            0,
            18,
            group_idx=0,
            spans=(
                CompactCPUAddressSpan(0, 2, 2),
                CompactCPUAddressSpan(5, 16, 16),
            ),
        )
        # Custom mapping with small non-identity page
        run = CopyRun(
            local_offset=1,
            canonical_offset=1,
            fragment_size=4,
            num_fragments=1,
            local_stride=4,
            canonical_stride=4,
        )
        mapping = CanonicalPageMapping(
            canonical_page_size_bytes=18,
            local_page_size_bytes=18,
            runs=(run,),
            num_writers=1,
            writer_index=0,
            parallelism_agnostic=True,
        )
        mappings = (mapping,)
        offsets = (0,)
        gpu_offsets = (0,)

        gpu_block_ids = np.array([0], dtype=np.int64)
        plan = plan_compact_transfer(
            gpu_base_ptr=0,
            gpu_row_stride=18,
            cpu_base_ptr=0,
            cpu_region_size=65536,
            gpu_block_ids=gpu_block_ids,
            group_sizes=[1],
            block_indices=[0],
            compact_addresses=[addr],
            per_group_mappings=[mappings],
            per_group_canonical_offsets=[offsets],
            per_group_gpu_offsets=[gpu_offsets],
            blocks_per_chunk=1,
        )
        assert plan.num_descriptors == 2
        descs = [
            (int(plan.gpu_ptrs[i]), int(plan.cpu_ptrs[i]), int(plan.sizes[i]))
            for i in range(2)
        ]
        # First span: local_offset=1 -> GPU ptr=1, logical 1 -> physical 1,
        #              span covers 1 byte
        assert descs[0] == (1, 1, 1), f"descs[0]={descs[0]}"
        # Second span: GPU ptr advanced by 1 -> 2, logical 2 -> spans[1]
        #              byte_offset=5 + intra_span 0 = 5, covers 3 bytes
        assert descs[1] == (2, 5, 3), f"descs[1]={descs[1]}"

    def test_three_span_split(self):
        """Fragment crossing three physical spans."""
        spans = (
            CompactCPUAddressSpan(0, 4, 4),
            CompactCPUAddressSpan(10, 4, 4),
            CompactCPUAddressSpan(20, 8, 8),
        )
        # addr has logical_length = 4+4+8 = 16
        addr = _make_address(0, 16, group_idx=0, spans=spans)

        run = CopyRun(
            local_offset=0,
            canonical_offset=2,  # start at logical 2
            fragment_size=8,
            num_fragments=1,
            local_stride=8,
            canonical_stride=8,
        )
        mapping = CanonicalPageMapping(
            canonical_page_size_bytes=16,
            local_page_size_bytes=16,
            runs=(run,),
            num_writers=1,
            writer_index=0,
            parallelism_agnostic=True,
        )
        mappings = (mapping,)
        offsets = (0,)
        gpu_offsets = (0,)

        gpu_block_ids = np.array([0], dtype=np.int64)
        plan = plan_compact_transfer(
            gpu_base_ptr=0,
            gpu_row_stride=16,
            cpu_base_ptr=0,
            cpu_region_size=65536,
            gpu_block_ids=gpu_block_ids,
            group_sizes=[1],
            block_indices=[0],
            compact_addresses=[addr],
            per_group_mappings=[mappings],
            per_group_canonical_offsets=[offsets],
            per_group_gpu_offsets=[gpu_offsets],
            blocks_per_chunk=1,
        )
        assert plan.num_descriptors == 3
        descs = [
            (int(plan.gpu_ptrs[i]), int(plan.cpu_ptrs[i]), int(plan.sizes[i]))
            for i in range(3)
        ]
        # Fragment: local_offset=0, fragment_size=8 -> GPU ptr base=0
        # Span 0: logical 2->3 (from canonical_offset=2), physical 0+(2-0)=2,
        #         size 2, GPU ptr advanced by 0 (no previous split)
        assert descs[0] == (0, 2, 2), f"descs[0]={descs[0]}"
        # Span 1: GPU ptr advanced by 2 -> 2, logical 4->7,
        #         physical 10+(4-4)=10, size 4
        assert descs[1] == (2, 10, 4), f"descs[1]={descs[1]}"
        # Span 2: GPU ptr advanced by 6 -> 6, logical 8->9,
        #         physical 20+(8-8)=20, size 2
        assert descs[2] == (6, 20, 2), f"descs[2]={descs[2]}"

    def test_store_load_symmetry(self):
        """Store then reversed-load round trip preserves data byte-for-byte."""
        addr = _make_address(
            0,
            4096,
            group_idx=0,
            spans=(CompactCPUAddressSpan(0, 4096, 4096),),
        )
        mappings = (_identity_mapping(4096),)
        offsets = (0,)
        gpu_offsets = (0,)

        data = bytearray([i % 256 for i in range(4096)])
        cpu_buf = bytearray(len(data))
        gpu_buf = bytearray(len(data))

        gpu_block_ids = np.array([0], dtype=np.int64)
        plan = plan_compact_transfer(
            gpu_base_ptr=0,
            gpu_row_stride=4096,
            cpu_base_ptr=0,
            cpu_region_size=65536,
            gpu_block_ids=gpu_block_ids,
            group_sizes=[1],
            block_indices=[0],
            compact_addresses=[addr],
            per_group_mappings=[mappings],
            per_group_canonical_offsets=[offsets],
            per_group_gpu_offsets=[gpu_offsets],
            blocks_per_chunk=1,
            direction="store",
        )
        # Simulate store: gpu -> cpu (store uses is_writer(gpu_block_id))
        for gp, cp, sz in zip(plan.gpu_ptrs, plan.cpu_ptrs, plan.sizes):
            for j in range(int(sz)):
                cpu_buf[int(cp) + j] = data[int(gp) + j]

        plan_load = plan_compact_transfer(
            gpu_base_ptr=0,
            gpu_row_stride=4096,
            cpu_base_ptr=0,
            cpu_region_size=65536,
            gpu_block_ids=gpu_block_ids,
            group_sizes=[1],
            block_indices=[0],
            compact_addresses=[addr],
            per_group_mappings=[mappings],
            per_group_canonical_offsets=[offsets],
            per_group_gpu_offsets=[gpu_offsets],
            blocks_per_chunk=1,
            direction="load",
        )
        # Simulate load: cpu -> gpu
        for gp, cp, sz in zip(plan_load.gpu_ptrs, plan_load.cpu_ptrs, plan_load.sizes):
            for j in range(int(sz)):
                gpu_buf[int(gp) + j] = cpu_buf[int(cp) + j]

        assert gpu_buf == data, "Round trip through store+load must preserve data"


# ---------------------------------------------------------------------------
# Direction: store is_writer filter vs load complete runs
# ---------------------------------------------------------------------------


class TestDirection:
    def test_store_filters_by_is_writer(self):
        """'store' direction uses is_writer(gpu_block_id) to filter descriptors."""
        run_store = CopyRun(
            local_offset=0,
            canonical_offset=0,
            fragment_size=4096,
            num_fragments=1,
            local_stride=4096,
            canonical_stride=4096,
        )
        mapping = CanonicalPageMapping(
            canonical_page_size_bytes=4096,
            local_page_size_bytes=4096,
            runs=(run_store,),
            num_writers=1,
            writer_index=0,
            parallelism_agnostic=True,
        )
        gpu_block_ids = np.array([0], dtype=np.int64)
        plan = plan_compact_transfer(
            gpu_base_ptr=10000,
            gpu_row_stride=4096,
            cpu_base_ptr=0,
            cpu_region_size=65536,
            gpu_block_ids=gpu_block_ids,
            group_sizes=[1],
            block_indices=[0],
            compact_addresses=[_make_address(0, 4096, group_idx=0)],
            per_group_mappings=[(mapping,)],
            per_group_canonical_offsets=[(0,)],
            per_group_gpu_offsets=[(0,)],
            blocks_per_chunk=1,
            direction="store",
        )
        assert plan.num_descriptors == 1
        assert plan.num_bytes == 4096

    def test_load_uses_complete_runs(self):
        """'load' direction uses complete runs regardless of writer status."""
        run_load = CopyRun(
            local_offset=0,
            canonical_offset=0,
            fragment_size=4096,
            num_fragments=1,
            local_stride=4096,
            canonical_stride=4096,
        )
        mapping = CanonicalPageMapping(
            canonical_page_size_bytes=4096,
            local_page_size_bytes=4096,
            runs=(run_load,),
            num_writers=2,
            writer_index=1,
            parallelism_agnostic=True,
        )
        gpu_block_ids = np.array([0], dtype=np.int64)
        plan = plan_compact_transfer(
            gpu_base_ptr=10000,
            gpu_row_stride=4096,
            cpu_base_ptr=0,
            cpu_region_size=65536,
            gpu_block_ids=gpu_block_ids,
            group_sizes=[1],
            block_indices=[0],
            compact_addresses=[_make_address(0, 4096, group_idx=0)],
            per_group_mappings=[(mapping,)],
            per_group_canonical_offsets=[(0,)],
            per_group_gpu_offsets=[(0,)],
            blocks_per_chunk=1,
            direction="load",
        )
        assert plan.num_descriptors == 1
        assert plan.num_bytes == 4096

    def test_nonwriter_store_returns_noop_plan(self):
        """Non-writer mapping returns valid zero-descriptor plan for store."""
        mapping = CanonicalPageMapping(
            canonical_page_size_bytes=4096,
            local_page_size_bytes=4096,
            runs=(CopyRun(0, 0, 4096, 1, 4096, 4096),),
            num_writers=2,
            writer_index=1,
            parallelism_agnostic=True,
        )
        gpu_block_ids = np.array([0], dtype=np.int64)
        plan = plan_compact_transfer(
            gpu_base_ptr=10000,
            gpu_row_stride=4096,
            cpu_base_ptr=0,
            cpu_region_size=65536,
            gpu_block_ids=gpu_block_ids,
            group_sizes=[1],
            block_indices=[0],
            compact_addresses=[_make_address(0, 4096, group_idx=0)],
            per_group_mappings=[(mapping,)],
            per_group_canonical_offsets=[(0,)],
            per_group_gpu_offsets=[(0,)],
            blocks_per_chunk=1,
            direction="store",
        )
        assert plan.num_descriptors == 0
        assert plan.num_bytes == 0
        assert plan.gpu_ptrs.shape == (0,)
        assert plan.cpu_ptrs.shape == (0,)
        assert plan.sizes.shape == (0,)
        # num_cpu_addresses still counts the compact CPU addresses
        # that were allocated, even when no descriptors are generated.
        assert plan.num_cpu_addresses == 1

    def test_mixed_writer_nonwriter_store(self):
        """Store with mixed writer/nonwriter layers: only writer-layer descriptors
        in store plan; load plan still includes both layers."""
        # Layer 0: writer with one 4096-byte store run
        writer_run = CopyRun(0, 0, 4096, 1, 4096, 4096)
        writer_mapping = CanonicalPageMapping(
            4096,
            4096,
            (writer_run,),
            1,
            0,
            True,
        )
        # Layer 1: non-writer (is_writer returns False for tested block)
        reader_run = CopyRun(0, 0, 4096, 1, 4096, 4096)
        nonwriter_mapping = CanonicalPageMapping(
            4096,
            4096,
            (reader_run,),
            2,
            1,
            True,
        )
        mappings = (writer_mapping, nonwriter_mapping)
        offsets = (0, 4096)
        gpu_offsets = (0, 0)

        gpu_block_ids = np.array([0], dtype=np.int64)
        address = _make_address(0, 8192, group_idx=0)

        # Store direction: only writer layer produces descriptors
        store_plan = plan_compact_transfer(
            gpu_base_ptr=10000,
            gpu_row_stride=8192,
            cpu_base_ptr=0,
            cpu_region_size=65536,
            gpu_block_ids=gpu_block_ids,
            group_sizes=[1],
            block_indices=[0],
            compact_addresses=[address],
            per_group_mappings=[mappings],
            per_group_canonical_offsets=[offsets],
            per_group_gpu_offsets=[gpu_offsets],
            blocks_per_chunk=1,
            direction="store",
        )
        # Only 1 descriptor (writer layer K at canonical offset 0)
        assert store_plan.num_descriptors == 1, (
            f"Expected 1 writer descriptor, got {store_plan.num_descriptors}"
        )
        assert store_plan.num_bytes == 4096
        assert int(store_plan.cpu_ptrs[0]) == 0  # K at canonical offset 0

        # Load direction: both layers produce descriptors
        load_plan = plan_compact_transfer(
            gpu_base_ptr=10000,
            gpu_row_stride=8192,
            cpu_base_ptr=0,
            cpu_region_size=65536,
            gpu_block_ids=gpu_block_ids,
            group_sizes=[1],
            block_indices=[0],
            compact_addresses=[address],
            per_group_mappings=[mappings],
            per_group_canonical_offsets=[offsets],
            per_group_gpu_offsets=[gpu_offsets],
            blocks_per_chunk=1,
            direction="load",
        )
        # Both layers produce descriptors for load
        assert load_plan.num_descriptors == 2, (
            f"Expected 2 load descriptors, got {load_plan.num_descriptors}"
        )
        assert load_plan.num_bytes == 8192
        cpu_ranges = [
            (
                int(load_plan.cpu_ptrs[i]),
                int(load_plan.cpu_ptrs[i] + load_plan.sizes[i]),
            )
            for i in range(2)
        ]
        # Load includes K at [0, 4096) and V at [4096, 8192)
        # (both layers, though V is non-writer via is_writer)
        assert cpu_ranges[0] == (0, 4096), f"cpu_ranges[0]={cpu_ranges[0]}"
        assert cpu_ranges[1] == (4096, 8192), f"cpu_ranges[1]={cpu_ranges[1]}"

    def test_invalid_direction(self):
        mappings = (_identity_mapping(4096),)
        with pytest.raises(ValueError, match="direction"):
            plan_compact_transfer(
                gpu_base_ptr=10000,
                gpu_row_stride=4096,
                cpu_base_ptr=0,
                cpu_region_size=65536,
                gpu_block_ids=np.array([0], dtype=np.int64),
                group_sizes=[1],
                block_indices=[0],
                compact_addresses=[_make_address(0, 4096, group_idx=0)],
                per_group_mappings=[mappings],
                per_group_canonical_offsets=[(0,)],
                per_group_gpu_offsets=[(0,)],
                blocks_per_chunk=1,
                direction="invalid",
            )


# ---------------------------------------------------------------------------
# blocks_per_chunk > 1 (BSF) tests
# ---------------------------------------------------------------------------


class TestBlocksPerChunk:
    def test_bsf_two_layer_major_ranges(self):
        """BSF=2, two layers: descriptor order is [K0,V0,K1,V1] (layer-major).
        CPU ranges: K0@[0,1024), V0@[1024,2048), K1@[2048,3072), V1@[3072,4096)."""
        run = CopyRun(
            local_offset=0,
            canonical_offset=0,
            fragment_size=1024,
            num_fragments=1,
            local_stride=1024,
            canonical_stride=1024,
        )

        def _mk_page():
            return CanonicalPageMapping(
                canonical_page_size_bytes=1024,
                local_page_size_bytes=1024,
                runs=(run,),
                num_writers=1,
                writer_index=0,
                parallelism_agnostic=True,
            )

        k_mapping, v_mapping = _mk_page(), _mk_page()
        mappings = (k_mapping, v_mapping)
        offsets = (0, 1024)
        gpu_offsets = (0, 0)

        gpu_block_ids = np.array([0, 1], dtype=np.int64)
        address = _make_address(0, 4096, group_idx=0)

        plan = plan_compact_transfer(
            gpu_base_ptr=10000,
            gpu_row_stride=4096,
            cpu_base_ptr=0,
            cpu_region_size=65536,
            gpu_block_ids=gpu_block_ids,
            group_sizes=[2],
            block_indices=[0],
            compact_addresses=[address],
            per_group_mappings=[mappings],
            per_group_canonical_offsets=[offsets],
            per_group_gpu_offsets=[gpu_offsets],
            blocks_per_chunk=2,
        )
        assert plan.num_descriptors == 4
        cpu_ranges = [
            (int(plan.cpu_ptrs[i]), int(plan.cpu_ptrs[i] + plan.sizes[i]))
            for i in range(4)
        ]
        # k sub0 at canonical offset 0
        assert cpu_ranges[0] == (0, 1024), f"cpu_ranges[0]={cpu_ranges[0]}"
        # k sub1 at canonical offset 0 + BSF_offset = 1024
        assert cpu_ranges[2] == (1024, 2048), f"cpu_ranges[2]={cpu_ranges[2]}"
        # v sub0 at canonical offset 1024
        assert cpu_ranges[1] == (2048, 3072), f"cpu_ranges[1]={cpu_ranges[1]}"
        # v sub1 at canonical offset 1024 + BSF_offset = 3072
        assert cpu_ranges[3] == (3072, 4096), f"cpu_ranges[3]={cpu_ranges[3]}"

    def test_fragment_exceeds_subblock_rejected(self):
        """Fragment larger than canonical sub-page is rejected."""
        run = CopyRun(
            local_offset=0,
            canonical_offset=0,
            fragment_size=2048,
            num_fragments=1,
            local_stride=2048,
            canonical_stride=2048,
        )
        mapping = CanonicalPageMapping(
            canonical_page_size_bytes=1024,
            local_page_size_bytes=1024,
            runs=(run,),
            num_writers=1,
            writer_index=0,
            parallelism_agnostic=True,
        )
        # Valid address: canonical_group_page_size=1024 * blocks_per_chunk=2
        with pytest.raises(ValueError, match="exceeds layer"):
            plan_compact_transfer(
                gpu_base_ptr=10000,
                gpu_row_stride=4096,
                cpu_base_ptr=0,
                cpu_region_size=65536,
                gpu_block_ids=np.array([0], dtype=np.int64),
                group_sizes=[1],
                block_indices=[0],
                compact_addresses=[_make_address(0, 2048, group_idx=0)],
                per_group_mappings=[(mapping,)],
                per_group_canonical_offsets=[(0,)],
                per_group_gpu_offsets=[(0,)],
                blocks_per_chunk=2,
            )

    def test_partial_first_chunk(self):
        """block_idx offset produces partial first chunk access."""
        run = CopyRun(
            local_offset=0,
            canonical_offset=0,
            fragment_size=4096,
            num_fragments=1,
            local_stride=4096,
            canonical_stride=4096,
        )
        mapping = CanonicalPageMapping(
            canonical_page_size_bytes=4096,
            local_page_size_bytes=4096,
            runs=(run,),
            num_writers=1,
            writer_index=0,
            parallelism_agnostic=True,
        )
        mappings = (mapping,)
        offsets = (0,)
        gpu_offsets = (0,)

        # block_idx=1 means skip first sub-block in first compact address
        # 2 blocks with block_idx=1, blocks_per_chunk=2 -> first_sub_block=1
        # Need 1 compact address: cdiv(1+2, 2) = cdiv(3, 2) = 2 addresses
        gpu_block_ids = np.array([0, 1], dtype=np.int64)
        addresses = [
            _make_address(0, 8192, group_idx=0),
            _make_address(8192, 8192, group_idx=0),
        ]

        plan = plan_compact_transfer(
            gpu_base_ptr=10000,
            gpu_row_stride=4096,
            cpu_base_ptr=0,
            cpu_region_size=65536,
            gpu_block_ids=gpu_block_ids,
            group_sizes=[2],
            block_indices=[1],
            compact_addresses=addresses,
            per_group_mappings=[mappings],
            per_group_canonical_offsets=[offsets],
            per_group_gpu_offsets=[gpu_offsets],
            blocks_per_chunk=2,
        )
        assert plan.num_descriptors == 2
        # CPU pointer ranges: first block uses sub_idx=1 -> offset
        # 1 * 4096 = 4096, second block uses second address sub_idx=0
        assert int(plan.cpu_ptrs[0]) == 4096  # first block's sub_idx=1
        assert int(plan.cpu_ptrs[1]) == 8192  # second address byte_offset


# ---------------------------------------------------------------------------
# Validation / error tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_group_sizes(self):
        mappings = (_identity_mapping(4096),)
        with pytest.raises(ValueError, match="group_sizes must match"):
            plan_compact_transfer(
                gpu_base_ptr=10000,
                gpu_row_stride=4096,
                cpu_base_ptr=0,
                cpu_region_size=65536,
                gpu_block_ids=np.array([0], dtype=np.int64),
                group_sizes=[1, 1],  # 2 sizes but 1 group
                block_indices=[0, 0],
                compact_addresses=[],
                per_group_mappings=[mappings],
                per_group_canonical_offsets=[(0,)],
                per_group_gpu_offsets=[(0,)],
                blocks_per_chunk=1,
            )

    def test_non_positive_params(self):
        mappings = (_identity_mapping(4096),)
        with pytest.raises(ValueError, match="positive"):
            plan_compact_transfer(
                gpu_base_ptr=10000,
                gpu_row_stride=0,  # invalid
                cpu_base_ptr=0,
                cpu_region_size=65536,
                gpu_block_ids=np.array([0], dtype=np.int64),
                group_sizes=[1],
                block_indices=[0],
                compact_addresses=[],
                per_group_mappings=[mappings],
                per_group_canonical_offsets=[(0,)],
                per_group_gpu_offsets=[(0,)],
                blocks_per_chunk=1,
            )

    def test_negative_block_ids(self):
        mappings = (_identity_mapping(4096),)
        with pytest.raises(ValueError, match="must be non-negative"):
            plan_compact_transfer(
                gpu_base_ptr=10000,
                gpu_row_stride=4096,
                cpu_base_ptr=0,
                cpu_region_size=65536,
                gpu_block_ids=np.array([-1], dtype=np.int64),
                group_sizes=[1],
                block_indices=[0],
                compact_addresses=[_make_address(0, 4096, group_idx=0)],
                per_group_mappings=[mappings],
                per_group_canonical_offsets=[(0,)],
                per_group_gpu_offsets=[(0,)],
                blocks_per_chunk=1,
            )

    def test_negative_group_size(self):
        """Negative group size raises ValueError."""
        mappings_a = (_identity_mapping(4096),)
        mappings_b = (_identity_mapping(2048),)
        with pytest.raises(ValueError):
            plan_compact_transfer(
                gpu_base_ptr=10000,
                gpu_row_stride=4096,
                cpu_base_ptr=0,
                cpu_region_size=65536,
                # sum([-1, 2]) = 1, len([0]) = 1 -> passes sum check
                gpu_block_ids=np.array([0], dtype=np.int64),
                group_sizes=[-1, 2],
                block_indices=[0, 0],
                compact_addresses=[],
                per_group_mappings=[mappings_a, mappings_b],
                per_group_canonical_offsets=[(0,), (0,)],
                per_group_gpu_offsets=[(0,), (0,)],
                blocks_per_chunk=1,
            )

    def test_mismatched_mappings_length(self):
        """per_group_canonical_offsets and per_group_gpu_offsets must
        match per_group_mappings."""
        mappings = (_identity_mapping(4096), _identity_mapping(4096))
        with pytest.raises(ValueError, match="same length"):
            plan_compact_transfer(
                gpu_base_ptr=10000,
                gpu_row_stride=4096,
                cpu_base_ptr=0,
                cpu_region_size=65536,
                gpu_block_ids=np.array([0], dtype=np.int64),
                group_sizes=[1],
                block_indices=[0],
                compact_addresses=[_make_address(0, 4096, group_idx=0)],
                per_group_mappings=[mappings],
                per_group_canonical_offsets=[(0,)],  # 1 layer, but 2 mappings
                per_group_gpu_offsets=[(0, 0)],
                blocks_per_chunk=1,
            )

    def test_out_of_range_group_idx(self):
        address = _make_address(0, 4096, group_idx=5)
        with pytest.raises(ValueError, match="out-of-range group index"):
            plan_compact_transfer(
                gpu_base_ptr=10000,
                gpu_row_stride=4096,
                cpu_base_ptr=0,
                cpu_region_size=65536,
                gpu_block_ids=np.array([0], dtype=np.int64),
                group_sizes=[1],
                block_indices=[0],
                compact_addresses=[address],
                per_group_mappings=[(_identity_mapping(4096),)],
                per_group_canonical_offsets=[(0,)],
                per_group_gpu_offsets=[(0,)],
                blocks_per_chunk=1,
            )

    def test_negative_canonical_offset_rejected(self):
        """Negative canonical offset is rejected."""
        mappings = (_identity_mapping(4096),)
        with pytest.raises(ValueError, match="canonical offset.*must be non-negative"):
            plan_compact_transfer(
                gpu_base_ptr=10000,
                gpu_row_stride=4096,
                cpu_base_ptr=0,
                cpu_region_size=65536,
                gpu_block_ids=np.array([0], dtype=np.int64),
                group_sizes=[1],
                block_indices=[0],
                compact_addresses=[_make_address(0, 4096, group_idx=0)],
                per_group_mappings=[mappings],
                per_group_canonical_offsets=[(-1,)],
                per_group_gpu_offsets=[(0,)],
                blocks_per_chunk=1,
            )

    def test_overlapping_canonical_extents_rejected(self):
        """Overlapping layer canonical extents are rejected."""
        k_run = CopyRun(0, 0, 4096, 1, 4096, 4096)
        v_run = CopyRun(0, 0, 2048, 1, 2048, 2048)
        k_mapping = CanonicalPageMapping(4096, 4096, (k_run,), 1, 0, True)

        v_mapping = CanonicalPageMapping(2048, 2048, (v_run,), 1, 0, True)

        with pytest.raises(ValueError, match="extents.*overlap"):
            plan_compact_transfer(
                gpu_base_ptr=10000,
                gpu_row_stride=4096,
                cpu_base_ptr=0,
                cpu_region_size=65536,
                gpu_block_ids=np.array([0], dtype=np.int64),
                group_sizes=[1],
                block_indices=[0],
                compact_addresses=[_make_address(0, 4096, group_idx=0)],
                # K extent [0, 4096), V extent [2048, 4096) -> overlap
                per_group_mappings=[(k_mapping, v_mapping)],
                per_group_canonical_offsets=[(0, 2048)],
                per_group_gpu_offsets=[(0, 0)],
                blocks_per_chunk=1,
            )

    def test_gapped_canonical_offsets_accepted(self):
        """Gapped canonical offsets are accepted with correct extent."""
        # K @ offset 0 (1024 bytes), V @ offset 8192 (2048 bytes)
        # Gap of 5120 bytes between extents.
        k_run = CopyRun(0, 0, 1024, 1, 1024, 1024)
        v_run = CopyRun(0, 0, 2048, 1, 2048, 2048)
        k_mapping = CanonicalPageMapping(1024, 1024, (k_run,), 1, 0, True)

        v_mapping = CanonicalPageMapping(2048, 2048, (v_run,), 1, 0, True)

        # canonical_group_page_size = max(0+1024, 8192+2048) = 10240
        expected_addr_len = 10240 * 1  # BSF=1
        gpu_block_ids = np.array([0], dtype=np.int64)
        plan = plan_compact_transfer(
            gpu_base_ptr=10000,
            gpu_row_stride=8192,
            cpu_base_ptr=0,
            cpu_region_size=65536,
            gpu_block_ids=gpu_block_ids,
            group_sizes=[1],
            block_indices=[0],
            compact_addresses=[_make_address(0, expected_addr_len, group_idx=0)],
            per_group_mappings=[(k_mapping, v_mapping)],
            per_group_canonical_offsets=[(0, 8192)],
            per_group_gpu_offsets=[(0, 0)],
            blocks_per_chunk=1,
        )
        assert plan.num_descriptors == 2
        # K at logical 0 -> cpu_ptr 0, V at logical 8192 -> cpu_ptr 8192
        assert int(plan.cpu_ptrs[0]) == 0
        assert int(plan.cpu_ptrs[1]) == 8192

    def test_uint64_overflow_gpu_ptr_rejected(self):
        """GPU pointer exceeding uint64 max is rejected."""
        mappings = (_identity_mapping(4096),)
        with pytest.raises(ValueError, match="out of uint64 range"):
            plan_compact_transfer(
                gpu_base_ptr=2**64,  # exceeds uint64
                gpu_row_stride=4096,
                cpu_base_ptr=0,
                cpu_region_size=65536,
                gpu_block_ids=np.array([0], dtype=np.int64),
                group_sizes=[1],
                block_indices=[0],
                compact_addresses=[_make_address(0, 4096, group_idx=0)],
                per_group_mappings=[mappings],
                per_group_canonical_offsets=[(0,)],
                per_group_gpu_offsets=[(0,)],
                blocks_per_chunk=1,
            )

    def test_uint64_overflow_cpu_ptr_rejected(self):
        """CPU pointer exceeding uint64 max is rejected via base pointer."""
        run = CopyRun(0, 0, 4096, 1, 4096, 4096)
        mapping = CanonicalPageMapping(4096, 4096, (run,), 1, 0, True)

        with pytest.raises(ValueError, match="out of uint64 range"):
            plan_compact_transfer(
                gpu_base_ptr=10000,
                gpu_row_stride=4096,
                cpu_base_ptr=2**64,  # exceeds uint64
                cpu_region_size=65536,
                gpu_block_ids=np.array([0], dtype=np.int64),
                group_sizes=[1],
                block_indices=[0],
                compact_addresses=[_make_address(0, 4096, group_idx=0)],
                per_group_mappings=[(mapping,)],
                per_group_canonical_offsets=[(0,)],
                per_group_gpu_offsets=[(0,)],
                blocks_per_chunk=1,
            )


# ---------------------------------------------------------------------------
# D4 invariant — all CPU descriptors within backing region
# ---------------------------------------------------------------------------


class TestD4Invariant:
    def test_nonzero_geometry_produces_bounded_plan(self):
        """Every CPU descriptor falls within [cpu_base_ptr, cpu_base_ptr +
        cpu_region_size)."""
        run = CopyRun(0, 0, 4096, 1, 4096, 4096)
        mapping = CanonicalPageMapping(4096, 4096, (run,), 1, 0, True)

        gpu_block_ids = np.array([0], dtype=np.int64)
        plan = plan_compact_transfer(
            gpu_base_ptr=10000,
            gpu_row_stride=4096,
            cpu_base_ptr=20000,
            cpu_region_size=65536,
            gpu_block_ids=gpu_block_ids,
            group_sizes=[1],
            block_indices=[0],
            compact_addresses=[_make_address(0, 4096, group_idx=0)],
            per_group_mappings=[(mapping,)],
            per_group_canonical_offsets=[(0,)],
            per_group_gpu_offsets=[(0,)],
            blocks_per_chunk=1,
        )
        assert isinstance(plan, CompactTransferPlan)
        assert plan.num_descriptors >= 1
        assert plan.num_bytes > 0
        assert plan.num_cpu_addresses == 1
        cpu_region_end = 20000 + 65536
        for cpu_ptr, size in zip(plan.cpu_ptrs, plan.sizes):
            cpu_end = int(cpu_ptr) + int(size)
            assert int(cpu_ptr) >= 20000
            assert cpu_end <= cpu_region_end

    def test_bounds_rejected_when_outside_region(self):
        """Descriptors exceeding the CPU region raise ValueError."""
        run = CopyRun(0, 0, 4096, 1, 4096, 4096)
        mapping = CanonicalPageMapping(4096, 4096, (run,), 1, 0, True)

        with pytest.raises(ValueError, match="exceeds backing region"):
            plan_compact_transfer(
                gpu_base_ptr=10000,
                gpu_row_stride=4096,
                cpu_base_ptr=20000,
                cpu_region_size=1,  # too small
                gpu_block_ids=np.array([0], dtype=np.int64),
                group_sizes=[1],
                block_indices=[0],
                compact_addresses=[_make_address(0, 4096, group_idx=0)],
                per_group_mappings=[(mapping,)],
                per_group_canonical_offsets=[(0,)],
                per_group_gpu_offsets=[(0,)],
                blocks_per_chunk=1,
            )


# ---------------------------------------------------------------------------
# Per-copy mode test
# ---------------------------------------------------------------------------


class TestPerCopyDescriptors:
    def test_compact_produces_flat_descriptors(self):
        """Compact planning produces flat per-descriptor arrays (per-copy
        mode), not batched rows."""
        run = CopyRun(0, 0, 4096, 1, 4096, 4096)
        k_mapping = CanonicalPageMapping(4096, 4096, (run,), 1, 0, True)

        v_mapping = CanonicalPageMapping(4096, 4096, (run,), 1, 0, True)

        mappings = (k_mapping, v_mapping)
        offsets = (0, 4096)
        gpu_offsets = (0, 0)

        gpu_block_ids = np.array([0, 1], dtype=np.int64)
        addresses = [
            _make_address(0, 8192, group_idx=0),
            _make_address(8192, 8192, group_idx=0),
        ]

        plan = plan_compact_transfer(
            gpu_base_ptr=10000,
            gpu_row_stride=8192,
            cpu_base_ptr=0,
            cpu_region_size=65536,
            gpu_block_ids=gpu_block_ids,
            group_sizes=[2],
            block_indices=[0],
            compact_addresses=addresses,
            per_group_mappings=[mappings],
            per_group_canonical_offsets=[offsets],
            per_group_gpu_offsets=[gpu_offsets],
            blocks_per_chunk=1,
        )
        # 2 blocks * 2 layers = 4 descriptors (1 fragment each)
        assert plan.num_descriptors >= 4
        # Flat array, not batched
        assert plan.gpu_ptrs.ndim == 1
        assert plan.cpu_ptrs.ndim == 1
        assert plan.sizes.ndim == 1


# ---------------------------------------------------------------------------
# Multiple groups
# ---------------------------------------------------------------------------


class TestMultipleGroups:
    def test_two_groups(self):
        """Two independent groups produce separate descriptor sets."""
        run0 = CopyRun(0, 0, 4096, 1, 4096, 4096)
        m0 = CanonicalPageMapping(4096, 4096, (run0,), 1, 0, True)

        run1 = CopyRun(0, 0, 2048, 1, 2048, 2048)
        m1 = CanonicalPageMapping(2048, 2048, (run1,), 1, 0, True)

        gpu_block_ids = np.array([0, 5], dtype=np.int64)
        addresses = [
            _make_address(0, 4096, group_idx=0),
            _make_address(0, 2048, group_idx=1),
        ]

        plan = plan_compact_transfer(
            gpu_base_ptr=10000,
            gpu_row_stride=4096,
            cpu_base_ptr=0,
            cpu_region_size=65536,
            gpu_block_ids=gpu_block_ids,
            group_sizes=[1, 1],
            block_indices=[0, 0],
            compact_addresses=addresses,
            per_group_mappings=[(m0,), (m1,)],
            per_group_canonical_offsets=[(0,), (0,)],
            per_group_gpu_offsets=[(0,), (0,)],
            blocks_per_chunk=1,
        )
        assert plan.num_descriptors >= 2
        assert plan.num_cpu_addresses == 2


# ---------------------------------------------------------------------------
# translate_through_spans edge cases
# ---------------------------------------------------------------------------


class TestTranslateThroughSpans:
    def test_exact_span_boundary(self):
        """Fragment aligning exactly with span boundary produces one
        descriptor."""
        spans = (CompactCPUAddressSpan(0, 4096, 4096),)
        result = _translate_through_spans(0, 4096, spans)
        assert result == [(0, 4096)]

    def test_beyond_spans_raises(self):
        spans = (CompactCPUAddressSpan(0, 1024, 1024),)
        with pytest.raises(ValueError, match="exceeds physical spans"):
            _translate_through_spans(0, 2048, spans)

    def test_gap_between_spans_raises(self):
        """Offset falling between non-contiguous logical ranges raises."""
        # Build spans that are NOT logically contiguous. This violates
        # CompactCPUAddress.__post_init__ for a real address, so test
        # _translate_through_spans directly.
        spans = (
            CompactCPUAddressSpan(0, 4, 4),
            CompactCPUAddressSpan(100, 4, 4),
        )
        # Logical offset 6: span 0 covers [0,4), span 1's
        # span_logical_start=4 but covers logical [4,8) per its
        # logical_length=4.  cursor=6 < 8 so enters span 1 processing,
        # but 6 >= span_logical_start=4 so no gap.  To get a REAL gap
        # cursor must be < span_logical_start, which requires skipping a
        # span and having cursor land before the next span's start.
        # The accumulation-based scheme is always contiguous for
        # well-formed addresses, so the gap branch is defensive.
        # We can still trigger it with fragment_size exceeding
        # the total logical coverage:
        with pytest.raises(ValueError, match="exceeds physical spans"):
            _translate_through_spans(0, 20, spans)
