# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import mmap
import random
import time
import uuid

import numpy as np
import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
)
from vllm.v1.kv_offload.cpu import gpu_worker
from vllm.v1.kv_offload.cpu.common import (
    CompactCPULoadStoreSpec,
    CPULoadStoreSpec,
)
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

PAGE_SIZE = mmap.PAGESIZE

NUM_GPU_BLOCKS = [64]
NUM_CPU_BLOCKS = [256]
GPU_PAGE_SIZES = [512, 1024]
BLOCKS_PER_CHUNK_VALUES = [1, 3]
NUM_TENSORS = [4]
SEEDS = [0]
DEVICE_TYPE = current_platform.device_type
DEVICES = [f"{DEVICE_TYPE}:0"]
NUM_MAPPINGS = [3]
NUM_MAPPINGS_PER_GROUP = [2]


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-specific test")
def test_rocm_cpu_to_gpu_uses_dma(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_worker, "HAS_TRITON", True)
    monkeypatch.setattr(gpu_worker.current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(gpu_worker.current_platform, "is_rocm", lambda: True)

    refs = [[CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=512)]]
    assert gpu_worker._select_swap_blocks_fn(refs, gpu_to_cpu=False) is (
        ops.swap_blocks_batch
    )


@pytest.mark.parametrize("gpu_to_cpu", [True, False])
@pytest.mark.parametrize("num_mappings", NUM_MAPPINGS)
@pytest.mark.parametrize("gpu_page_size_bytes", GPU_PAGE_SIZES)
@pytest.mark.parametrize("blocks_per_chunk", BLOCKS_PER_CHUNK_VALUES)
@pytest.mark.parametrize("num_gpu_blocks", NUM_GPU_BLOCKS)
@pytest.mark.parametrize("num_cpu_blocks", NUM_CPU_BLOCKS)
@pytest.mark.parametrize("num_tensors", NUM_TENSORS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    ("use_shared_memory", "replicated_layout"),
    [(False, False), (True, False), (True, True)],
)
@torch.inference_mode()
def test_transfer(
    default_vllm_config,
    gpu_to_cpu: bool,
    num_mappings: int,
    gpu_page_size_bytes: int,
    blocks_per_chunk: int,
    num_gpu_blocks: int,
    num_cpu_blocks: int,
    num_tensors: int,
    seed: int,
    device: str,
    use_shared_memory: bool,
    replicated_layout: bool,
) -> None:
    set_random_seed(seed)

    # build CanonicalKVCacheTensor list: one per tensor
    kv_cache_tensors: list[CanonicalKVCacheTensor] = []
    for i in range(num_tensors):
        gpu_tensor = torch.zeros(
            (num_gpu_blocks, gpu_page_size_bytes),
            dtype=torch.int8,
            device=device,
        )
        kv_cache_tensors.append(
            CanonicalKVCacheTensor(
                tensor=gpu_tensor,
                page_size_bytes=gpu_page_size_bytes,
            )
        )

    # one group containing all tensors, one data ref per tensor
    kv_cache_groups_data_refs: list[list[CanonicalKVCacheRef]] = [
        [
            CanonicalKVCacheRef(
                tensor_idx=i,
                page_size_bytes=gpu_page_size_bytes,
            )
            for i in range(num_tensors)
        ]
    ]

    kv_caches = CanonicalKVCaches(
        tensors=kv_cache_tensors,
        group_data_refs=kv_cache_groups_data_refs,
    )

    mmap_region: SharedOffloadRegion | None = None
    if use_shared_memory:
        cpu_page_size = round_up(
            gpu_page_size_bytes * num_tensors * blocks_per_chunk,
            SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT,
        )
        simulated_world_size = 2
        kv_bytes_per_block = (
            cpu_page_size if replicated_layout else cpu_page_size * simulated_world_size
        )
        mmap_region = SharedOffloadRegion(
            engine_id=str(uuid.uuid4()),
            num_blocks=num_cpu_blocks,
            rank=0,
            kv_bytes_per_block=kv_bytes_per_block,
            cpu_page_size=cpu_page_size,
        )

    worker = CPUOffloadingWorker(
        kv_caches=kv_caches,
        blocks_per_chunk=blocks_per_chunk,
        num_cpu_blocks=num_cpu_blocks,
        mmap_region=mmap_region,
    )

    # select block mappings
    gpu_blocks = random.sample(range(num_gpu_blocks), num_mappings * blocks_per_chunk)
    cpu_blocks = random.sample(range(num_cpu_blocks), num_mappings)

    # expand cpu blocks to gpu-page granularity for uniform comparison:
    # each cpu block maps to blocks_per_chunk consecutive sub-blocks
    cpu_blocks_expanded = [
        cpu_block * blocks_per_chunk + j
        for cpu_block in cpu_blocks
        for j in range(blocks_per_chunk)
    ]

    # maybe skip some GPU blocks to test reading/writing from the middle of a CPU block
    blocks_to_skip = blocks_per_chunk - 1
    if blocks_to_skip > 0:
        gpu_blocks = gpu_blocks[blocks_to_skip:]
        cpu_blocks_expanded = cpu_blocks_expanded[blocks_to_skip:]

    # set transfer direction
    if gpu_to_cpu:
        handler = worker._store_handler
        src_spec = GPULoadStoreSpec(
            gpu_blocks, group_sizes=(len(gpu_blocks),), block_indices=(blocks_to_skip,)
        )
        dst_spec = CPULoadStoreSpec(cpu_blocks)
        dst_to_src = dict(zip(cpu_blocks_expanded, gpu_blocks))
        num_dst_sub_blocks = num_gpu_blocks
    else:
        handler = worker._load_handler
        src_spec = CPULoadStoreSpec(cpu_blocks)
        dst_spec = GPULoadStoreSpec(
            gpu_blocks, group_sizes=(len(gpu_blocks),), block_indices=(blocks_to_skip,)
        )
        dst_to_src = dict(zip(gpu_blocks, cpu_blocks_expanded))
        num_dst_sub_blocks = num_gpu_blocks

    # randomize src and dst tensors before transfer
    for tensor in handler.src_tensors:
        tensor.random_()
    for tensor in handler.dst_tensors:
        tensor.random_()

    # clone src and dst tensors before transfer
    orig_src_tensors = [x.clone() for x in handler.src_tensors]
    orig_dst_tensors = [x.clone() for x in handler.dst_tensors]

    # call transfer function via public API
    start_time = time.time()
    if gpu_to_cpu:
        assert worker.submit_store(1, src_spec, dst_spec)
    else:
        assert worker.submit_load(1, src_spec, dst_spec)
    assert {x.job_id for x in handler._transfers} == {1}

    # wait for transfer to complete
    end_time = time.time() + 10
    while time.time() < end_time:
        finished = worker.get_finished()
        if finished:
            assert finished[0].job_id == 1
            assert finished[0].success
            assert finished[0].transfer_size == (
                len(gpu_blocks)
                * sum([x.page_size_bytes for x in handler.kv_cache_groups_data_refs[0]])
            )
            assert finished[0].transfer_time > 0
            assert finished[0].transfer_time < (time.time() - start_time)
            break
        time.sleep(0.1)

    # verify src tensors did not change
    for orig_tensor, tensor in zip(orig_src_tensors, handler.src_tensors):
        assert torch.equal(orig_tensor, tensor)

    # verify dst tensors at gpu-page granularity.
    for src_tensor, dst_tensor, orig_dst_tensor in zip(
        handler.src_tensors,
        handler.dst_tensors,
        orig_dst_tensors,
    ):
        # view both GPU and CPU tensors as (n, gpu_page_size_bytes) for comparison.
        src_view = src_tensor.reshape(-1, gpu_page_size_bytes)
        dst_view = dst_tensor.reshape(-1, gpu_page_size_bytes)
        orig_dst_view = orig_dst_tensor.reshape(-1, gpu_page_size_bytes)
        for dst_sub_block in range(num_dst_sub_blocks):
            src_sub_block = dst_to_src.get(dst_sub_block)
            if src_sub_block is not None:
                expected = src_view[src_sub_block]
            else:
                expected = orig_dst_view[dst_sub_block]
            torch.testing.assert_close(dst_view[dst_sub_block].cpu(), expected.cpu())

    # Drop loop-variable refs so mmap_obj has no exported buffers at cleanup.
    del orig_tensor, tensor, src_tensor, dst_tensor, orig_dst_tensor
    del src_view, dst_view, orig_dst_view, expected

    worker.shutdown()
    if mmap_region:
        mmap_region.cleanup()


@pytest.mark.parametrize("gpu_to_cpu", [True, False])
@pytest.mark.parametrize("num_mappings_per_group", NUM_MAPPINGS_PER_GROUP)
@pytest.mark.parametrize("gpu_page_size_bytes", GPU_PAGE_SIZES)
@pytest.mark.parametrize("blocks_per_chunk", BLOCKS_PER_CHUNK_VALUES)
@pytest.mark.parametrize("num_gpu_blocks", NUM_GPU_BLOCKS)
@pytest.mark.parametrize("num_cpu_blocks", NUM_CPU_BLOCKS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_transfer_multi_group(
    default_vllm_config,
    gpu_to_cpu: bool,
    num_mappings_per_group: int,
    gpu_page_size_bytes: int,
    blocks_per_chunk: int,
    num_gpu_blocks: int,
    num_cpu_blocks: int,
    seed: int,
    device: str,
) -> None:
    """Test transfers with three KV cache groups:
    - Group 0: aligned transfer with num_mappings_per_group blocks
    - Group 1: zero blocks (empty group)
    - Group 2: unaligned CPU->GPU transfer (logical_offset=blocks_per_chunk-1,
      causing the implementation to skip source sub-blocks) with
      num_mappings_per_group blocks
    """
    set_random_seed(seed)

    # 3 groups, each with 2 tensors
    num_groups = 3
    tensors_per_group = 2
    num_tensors = num_groups * tensors_per_group
    kv_cache_tensors: list[CanonicalKVCacheTensor] = []
    for _ in range(num_tensors):
        gpu_tensor = torch.zeros(
            (num_gpu_blocks, gpu_page_size_bytes),
            dtype=torch.int8,
            device=device,
        )
        kv_cache_tensors.append(
            CanonicalKVCacheTensor(
                tensor=gpu_tensor,
                page_size_bytes=gpu_page_size_bytes,
            )
        )

    kv_cache_groups_data_refs: list[list[CanonicalKVCacheRef]] = [
        [
            CanonicalKVCacheRef(
                tensor_idx=g * tensors_per_group + i,
                page_size_bytes=gpu_page_size_bytes,
            )
            for i in range(tensors_per_group)
        ]
        for g in range(num_groups)
    ]

    canonical_kv_caches = CanonicalKVCaches(
        tensors=kv_cache_tensors, group_data_refs=kv_cache_groups_data_refs
    )

    worker = CPUOffloadingWorker(
        kv_caches=canonical_kv_caches,
        blocks_per_chunk=blocks_per_chunk,
        num_cpu_blocks=num_cpu_blocks,
    )

    # group 0: aligned, group 1: empty, group 2: unaligned on CPU->GPU
    group_sizes_in_cpu_blocks = [num_mappings_per_group, 0, num_mappings_per_group]

    total_cpu_blocks = sum(group_sizes_in_cpu_blocks)
    total_gpu_blocks_needed = total_cpu_blocks * blocks_per_chunk
    gpu_blocks_all = random.sample(range(num_gpu_blocks), total_gpu_blocks_needed)
    cpu_blocks_all = random.sample(range(num_cpu_blocks), total_cpu_blocks)

    # split gpu/cpu blocks per group
    gpu_blocks_per_group: list[list[int]] = []
    cpu_blocks_per_group: list[list[int]] = []
    gpu_offset = 0
    cpu_offset = 0
    for size in group_sizes_in_cpu_blocks:
        gpu_count = size * blocks_per_chunk
        gpu_blocks_per_group.append(gpu_blocks_all[gpu_offset : gpu_offset + gpu_count])
        cpu_blocks_per_group.append(cpu_blocks_all[cpu_offset : cpu_offset + size])
        gpu_offset += gpu_count
        cpu_offset += size

    # expand cpu blocks to gpu-page granularity
    cpu_blocks_expanded_per_group = [
        [
            cpu_block * blocks_per_chunk + j
            for cpu_block in cpu_blocks
            for j in range(blocks_per_chunk)
        ]
        for cpu_blocks in cpu_blocks_per_group
    ]

    # skip sub-blocks from group 2 to test unaligned transfers.
    sub_blocks_to_skip = blocks_per_chunk - 1  # e.g. 2 when blocks_per_chunk=3
    if sub_blocks_to_skip > 0:
        gpu_blocks_per_group[2] = gpu_blocks_per_group[2][
            sub_blocks_to_skip:-sub_blocks_to_skip
        ]
        cpu_blocks_expanded_per_group[2] = cpu_blocks_expanded_per_group[2][
            sub_blocks_to_skip:-sub_blocks_to_skip
        ]

    # build flat gpu_blocks list and group_sizes in GPU blocks
    gpu_blocks: list[int] = []
    group_sizes: list[int] = []
    for gpu_blks in gpu_blocks_per_group:
        gpu_blocks.extend(gpu_blks)
        group_sizes.append(len(gpu_blks))

    # build flat cpu_blocks list
    cpu_blocks = []
    for cpu_blks in cpu_blocks_per_group:
        cpu_blocks.extend(cpu_blks)

    # block_indices: only relevant for unaligned transfers
    block_indices: list[int] = [0, 0, sub_blocks_to_skip]

    if gpu_to_cpu:
        handler = worker._store_handler
        src_spec = GPULoadStoreSpec(
            gpu_blocks, group_sizes=group_sizes, block_indices=block_indices
        )
        dst_spec = CPULoadStoreSpec(cpu_blocks)
        # per-group mapping: cpu sub-block -> gpu sub-block
        dst_to_src_per_group = [
            dict(zip(expanded, gpu_blks))
            for expanded, gpu_blks in zip(
                cpu_blocks_expanded_per_group, gpu_blocks_per_group
            )
        ]
        num_dst_sub_blocks = num_cpu_blocks * blocks_per_chunk
    else:
        handler = worker._load_handler
        src_spec = CPULoadStoreSpec(cpu_blocks)
        dst_spec = GPULoadStoreSpec(
            gpu_blocks, group_sizes=group_sizes, block_indices=block_indices
        )
        # per-group mapping: gpu sub-block -> cpu sub-block
        dst_to_src_per_group = [
            dict(zip(gpu_blks, expanded))
            for gpu_blks, expanded in zip(
                gpu_blocks_per_group, cpu_blocks_expanded_per_group
            )
        ]
        num_dst_sub_blocks = num_gpu_blocks

    # randomize src and dst tensors before transfer
    for tensor in handler.src_tensors:
        tensor.random_()
    for tensor in handler.dst_tensors:
        tensor.random_()

    orig_src_tensors = [x.clone() for x in handler.src_tensors]
    orig_dst_tensors = [x.clone() for x in handler.dst_tensors]

    if gpu_to_cpu:
        assert worker.submit_store(1, src_spec, dst_spec)
    else:
        assert worker.submit_load(1, src_spec, dst_spec)
    assert {x.job_id for x in handler._transfers} == {1}

    end_time = time.time() + 10
    while time.time() < end_time:
        finished = worker.get_finished()
        if finished:
            assert finished[0].job_id == 1
            assert finished[0].success
            expected_bytes = sum(
                group_size * sum([x.page_size_bytes for x in data_refs])
                for group_size, data_refs in zip(
                    group_sizes, handler.kv_cache_groups_data_refs
                )
            )
            assert finished[0].transfer_size == expected_bytes
            break
        time.sleep(0.1)

    # verify src tensors did not change
    for orig_tensor, tensor in zip(orig_src_tensors, handler.src_tensors):
        assert torch.equal(orig_tensor, tensor)

    # verify dst tensors at gpu-page granularity
    for group_idx, dst_to_src in enumerate(dst_to_src_per_group):
        group_tensor_offset = group_idx * tensors_per_group
        for tensor_idx in range(tensors_per_group):
            src_tensor = handler.src_tensors[group_tensor_offset + tensor_idx]
            dst_tensor = handler.dst_tensors[group_tensor_offset + tensor_idx]
            orig_dst_tensor = orig_dst_tensors[group_tensor_offset + tensor_idx]
            src_view = src_tensor.view(-1, gpu_page_size_bytes)
            dst_view = dst_tensor.view(-1, gpu_page_size_bytes)
            orig_dst_view = orig_dst_tensor.view(-1, gpu_page_size_bytes)
            for dst_sub_block in range(num_dst_sub_blocks):
                src_sub_block = dst_to_src.get(dst_sub_block)
                if src_sub_block is not None:
                    expected = src_view[src_sub_block]
                else:
                    expected = orig_dst_view[dst_sub_block]
                torch.testing.assert_close(
                    dst_view[dst_sub_block].cpu(), expected.cpu()
                )

    worker.shutdown()


# ---------------------------------------------------------------------------
# Worker reference retention (foundation for future compact descriptor)
# ---------------------------------------------------------------------------


def _make_kv_caches(
    num_tensors: int = 2,
    num_gpu_blocks: int = 64,
    gpu_page_size_bytes: int = 512,
) -> CanonicalKVCaches:
    _device = "cuda:0" if torch.cuda.is_available() else "cpu"
    kv_cache_tensors: list[CanonicalKVCacheTensor] = []
    for _ in range(num_tensors):
        gpu_tensor = torch.zeros(
            (num_gpu_blocks, gpu_page_size_bytes),
            dtype=torch.int8,
            device=_device,
        )
        kv_cache_tensors.append(
            CanonicalKVCacheTensor(
                tensor=gpu_tensor,
                page_size_bytes=gpu_page_size_bytes,
            )
        )
    kv_cache_groups_data_refs: list[list[CanonicalKVCacheRef]] = [
        [
            CanonicalKVCacheRef(
                tensor_idx=i,
                page_size_bytes=gpu_page_size_bytes,
            )
            for i in range(num_tensors)
        ]
    ]
    return CanonicalKVCaches(
        tensors=kv_cache_tensors,
        group_data_refs=kv_cache_groups_data_refs,
    )


@torch.inference_mode()
def test_worker_retains_mmap_region_reference():
    """CPUOffloadingWorker must retain the mmap_region as _mmap_region."""
    kv_caches = _make_kv_caches()
    mmap_region = SharedOffloadRegion(
        engine_id=str(uuid.uuid4()),
        num_blocks=64,
        rank=0,
        kv_bytes_per_block=2 * PAGE_SIZE,
        cpu_page_size=2 * PAGE_SIZE,
    )
    try:
        worker = CPUOffloadingWorker(
            kv_caches=kv_caches,
            blocks_per_chunk=1,
            num_cpu_blocks=64,
            mmap_region=mmap_region,
        )
        assert worker._mmap_region is mmap_region
        worker.shutdown()
        # Worker owns the shared region and cleans it exactly once after both
        # direction handlers release their borrowed references.
        assert worker._mmap_region is None, (
            "worker._mmap_region must be None after shutdown"
        )
        assert mmap_region.base_ptr == 0, (
            "mmap base_ptr must be 0 after worker-owned shutdown"
        )
        assert mmap_region.base_tensor is None, (
            "mmap base_tensor must be None after shutdown"
        )
    finally:
        mmap_region.cleanup()


@torch.inference_mode()
def test_worker_without_mmap_region():
    """Without mmap_region, _mmap_region must be None."""
    kv_caches = _make_kv_caches()
    worker = CPUOffloadingWorker(
        kv_caches=kv_caches,
        blocks_per_chunk=1,
        num_cpu_blocks=64,
    )
    assert worker._mmap_region is None
    worker.shutdown()


@torch.inference_mode()
def test_worker_mmap_region_reference_passed_to_both_handlers():
    """Both directions borrow the same mmap; the worker owns cleanup."""
    kv_caches = _make_kv_caches()
    mmap_region = SharedOffloadRegion(
        engine_id=str(uuid.uuid4()),
        num_blocks=64,
        rank=0,
        kv_bytes_per_block=2 * PAGE_SIZE,
        cpu_page_size=2 * PAGE_SIZE,
    )
    try:
        worker = CPUOffloadingWorker(
            kv_caches=kv_caches,
            blocks_per_chunk=1,
            num_cpu_blocks=64,
            mmap_region=mmap_region,
        )
        assert worker._store_handler._mmap_region is mmap_region
        assert worker._load_handler._mmap_region is mmap_region
        worker.shutdown()
        assert worker._store_handler._mmap_region is None
        assert worker._load_handler._mmap_region is None
        assert mmap_region.base_ptr == 0
    finally:
        mmap_region.cleanup()


@torch.inference_mode()
def test_worker_fallback_pinned_tensors():
    """Without mmap_region, CPU tensors are allocated as pinned torch.zeros
    (no mmap registration, no extra references besides pinning)."""
    kv_caches = _make_kv_caches()
    worker = CPUOffloadingWorker(
        kv_caches=kv_caches,
        blocks_per_chunk=1,
        num_cpu_blocks=64,
    )
    # Verify no mmap region reference is held
    assert worker._mmap_region is None
    # cpu_tensors are pinned via torch.zeros(pin_memory=True) when no mmap
    for cpu_t in worker._store_handler.dst_tensors:
        assert cpu_t.is_pinned()
    worker.shutdown()


# ---------------------------------------------------------------------------
# Compact descriptor transfer tests
# ---------------------------------------------------------------------------


def _compact_identity_mapping(
    page_size: int,
    *,
    num_writers: int = 1,
    writer_index: int = 0,
):
    """Build an identity ``CanonicalPageMapping`` for compact test helpers.
    Single-fragment identity run covering the full page.
    """
    from vllm.v1.kv_offload.base import CanonicalPageMapping, CopyRun

    run = CopyRun(0, 0, page_size, 1, page_size, page_size)
    return CanonicalPageMapping(
        canonical_page_size_bytes=page_size,
        local_page_size_bytes=page_size,
        runs=(run,),
        num_writers=num_writers,
        writer_index=writer_index,
        parallelism_agnostic=True,
    )


def _compact_address(
    byte_offset: int,
    logical_length: int,
    group_idx: int = 0,
):
    from vllm.v1.kv_offload.cpu.common import CompactCPUAddress

    return CompactCPUAddress(
        byte_offset=byte_offset,
        logical_length=logical_length,
        allocated_length=logical_length,
        group_idx=group_idx,
    )


def _make_compact_geometry(
    layer_names: list[str],
    layer_mappings: list,
    gpu_offset_bytes: list[int],
    gpu_row_stride: int,
):
    from vllm.v1.kv_offload.cpu.common import (
        CompactGroupGeometry,
        CompactLayerGeometry,
    )

    canonical_offset = 0
    layers: list[CompactLayerGeometry] = []
    for ln, mapping, goff in zip(layer_names, layer_mappings, gpu_offset_bytes):
        layers.append(
            CompactLayerGeometry(
                layer_name=ln,
                mapping=mapping,
                local_page_size_bytes=mapping.local_page_size_bytes,
                canonical_page_size_bytes=mapping.canonical_page_size_bytes,
                canonical_offset=canonical_offset,
                gpu_offset_bytes=goff,
            )
        )
        canonical_offset += mapping.canonical_page_size_bytes
    local_extent = sum(m.local_page_size_bytes for m in layer_mappings)
    canonical_extent = sum(m.canonical_page_size_bytes for m in layer_mappings)
    return (
        CompactGroupGeometry(
            layers=tuple(layers),
            gpu_row_stride=gpu_row_stride,
            local_extent=local_extent,
            canonical_extent=canonical_extent,
            parallel_invariant=all(m.parallelism_agnostic for m in layer_mappings),
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_compact_nonwriter_zero_descriptors(mocker):
    """Non-writer compact store (empty store_runs) produces zero descriptors
    through the compact path.  swap_blocks_batch is NOT called but the
    completion gate still registers job_id."""
    gpu_row_stride = 1024
    local_page_size = 1024
    blocks_per_chunk = 1

    m = _compact_identity_mapping(local_page_size, num_writers=2, writer_index=1)
    geometry = _make_compact_geometry(["l0"], [m], [0], gpu_row_stride)

    num_gpu_blocks = 8
    num_cpu_blocks = 16
    gpu_tensor = torch.zeros(
        num_gpu_blocks, gpu_row_stride, dtype=torch.int8, device="cuda"
    )
    cpu_tensor = torch.zeros(
        num_cpu_blocks,
        gpu_row_stride * blocks_per_chunk,
        dtype=torch.int8,
        device="cpu",
        pin_memory=True,
    )

    from vllm.v1.kv_offload.cpu.gpu_worker import SingleDirectionOffloadingHandler

    kv_cache_groups_data_refs = [
        [
            type("DR", (), dict(tensor_idx=0, page_size_bytes=gpu_row_stride))(),
        ]
    ]
    for dr in kv_cache_groups_data_refs[0]:
        dr.mapping = None

    handler = SingleDirectionOffloadingHandler(
        gpu_tensors=[gpu_tensor],
        cpu_tensors=[cpu_tensor],
        blocks_per_chunk=blocks_per_chunk,
        kv_cache_groups_data_refs=kv_cache_groups_data_refs,
        gpu_to_cpu=True,
        compact_geometry=geometry,
    )

    swap_mock = mocker.patch("vllm.v1.kv_offload.cpu.gpu_worker.ops.swap_blocks_batch")

    handler._mmap_region = _mock_region_for_cpu_tensor(cpu_tensor)

    gpu_block_ids = np.array([0], dtype=np.int64)
    src_spec = GPULoadStoreSpec(
        gpu_block_ids.tolist(),
        group_sizes=(1,),
        block_indices=(0,),
    )
    dst_spec = CompactCPULoadStoreSpec(
        [_compact_address(0, local_page_size * blocks_per_chunk, group_idx=0)]
    )

    result = handler.transfer_async(17, src_spec, dst_spec)
    assert result is True

    swap_mock.assert_not_called()
    assert 17 in handler._transfer_events

    handler.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_compact_routing_normal_batch_unchanged(mocker):
    """Normal (non-compact) transfer_async still routes through the batch API,
    even when compact geometry is configured.  The legacy path uses
    default ``use_batch_api=True`` forwarding."""
    gpu_row_stride = 1024
    blocks_per_chunk = 1

    m = _compact_identity_mapping(1024)
    geometry = _make_compact_geometry(["l0"], [m], [0], gpu_row_stride)

    num_gpu_blocks = 8
    num_cpu_blocks = 16
    gpu_tensor = torch.zeros(
        num_gpu_blocks, gpu_row_stride, dtype=torch.int8, device="cuda"
    )
    cpu_tensor = torch.zeros(
        num_cpu_blocks,
        gpu_row_stride * blocks_per_chunk,
        dtype=torch.int8,
        device="cpu",
        pin_memory=True,
    )

    from vllm.v1.kv_offload.cpu.gpu_worker import SingleDirectionOffloadingHandler

    kv_cache_groups_data_refs = [
        [
            type("DR", (), dict(tensor_idx=i, page_size_bytes=gpu_row_stride))()
            for i in range(2)
        ]
    ]
    for dr in kv_cache_groups_data_refs[0]:
        dr.mapping = None

    handler = SingleDirectionOffloadingHandler(
        gpu_tensors=[gpu_tensor, gpu_tensor.clone()],
        cpu_tensors=[cpu_tensor, cpu_tensor.clone()],
        blocks_per_chunk=blocks_per_chunk,
        kv_cache_groups_data_refs=kv_cache_groups_data_refs,
        gpu_to_cpu=True,
        compact_geometry=geometry,
    )

    swap_mock = mocker.patch("vllm.v1.kv_offload.cpu.gpu_worker.ops.swap_blocks_batch")

    gpu_block_ids = np.array([0], dtype=np.int64)
    src_spec = GPULoadStoreSpec(
        gpu_block_ids.tolist(),
        group_sizes=(1,),
        block_indices=(0,),
    )
    dst_spec = CPULoadStoreSpec([0])

    result = handler.transfer_async(33, src_spec, dst_spec)
    assert result is True

    assert swap_mock.called, (
        "swap_blocks_batch must be called for normal batch transfer"
    )
    _, kwargs = swap_mock.call_args
    assert kwargs.get("use_batch_api") is True or "use_batch_api" not in kwargs, (
        "legacy batch path must use default use_batch_api=True"
    )

    handler.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_compact_with_physical_spans(mocker):
    """Compact transfer with fragmented physical spans."""
    from vllm.v1.kv_offload.cpu.common import CompactCPUAddressSpan

    gpu_row_stride = 1024
    local_page_size = 1024
    blocks_per_chunk = 1

    m = _compact_identity_mapping(local_page_size)
    geometry = _make_compact_geometry(["l0"], [m], [0], gpu_row_stride)

    num_gpu_blocks = 8
    num_cpu_blocks = 32
    gpu_tensor = torch.zeros(
        num_gpu_blocks, gpu_row_stride, dtype=torch.int8, device="cuda"
    )
    cpu_tensor = torch.zeros(
        num_cpu_blocks,
        gpu_row_stride * blocks_per_chunk,
        dtype=torch.int8,
        device="cpu",
        pin_memory=True,
    )

    from vllm.v1.kv_offload.cpu.gpu_worker import SingleDirectionOffloadingHandler

    kv_cache_groups_data_refs = [
        [
            type("DR", (), dict(tensor_idx=0, page_size_bytes=gpu_row_stride))(),
        ]
    ]
    for dr in kv_cache_groups_data_refs[0]:
        dr.mapping = None

    handler = SingleDirectionOffloadingHandler(
        gpu_tensors=[gpu_tensor],
        cpu_tensors=[cpu_tensor],
        blocks_per_chunk=blocks_per_chunk,
        kv_cache_groups_data_refs=kv_cache_groups_data_refs,
        gpu_to_cpu=True,
        compact_geometry=geometry,
    )

    swap_mock = mocker.patch("vllm.v1.kv_offload.cpu.gpu_worker.ops.swap_blocks_batch")

    handler._mmap_region = _mock_region_for_cpu_tensor(cpu_tensor)

    addr = _compact_address(0, local_page_size * blocks_per_chunk, group_idx=0)
    object.__setattr__(
        addr,
        "spans",
        (
            CompactCPUAddressSpan(0, 512, 512),
            CompactCPUAddressSpan(4096, 512, 512),
        ),
    )

    gpu_block_ids = np.array([0], dtype=np.int64)
    src_spec = GPULoadStoreSpec(
        gpu_block_ids.tolist(),
        group_sizes=(1,),
        block_indices=(0,),
    )
    dst_spec = CompactCPULoadStoreSpec([addr])

    result = handler.transfer_async(77, src_spec, dst_spec)
    assert result is True

    assert swap_mock.called
    _, kwargs = swap_mock.call_args
    assert kwargs.get("use_batch_api") is False
    assert 77 in handler._transfer_events

    handler.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_compact_positional_group_mismatch():
    """Positional group geometry is preserved: compact transfer raises
    RuntimeError when geometry is None for a referenced group.

    Tests both:
    1. group_idx out of range
    2. group_idx present but geometry is None (incomplete group)
    """
    gpu_row_stride = 1024
    local_page_size = 1024
    blocks_per_chunk = 1

    m = _compact_identity_mapping(local_page_size)

    # Positional geometry: group 1 is None (incomplete).
    # Tuple preserves position: (full, None).
    positional_geom = (
        _make_compact_geometry(["l0"], [m], [0], gpu_row_stride)[0],
        None,
    )

    num_gpu_blocks = 8
    num_cpu_blocks = 16
    gpu_tensor = torch.zeros(
        num_gpu_blocks, gpu_row_stride, dtype=torch.int8, device="cuda"
    )
    cpu_tensor = torch.zeros(
        num_cpu_blocks,
        gpu_row_stride * blocks_per_chunk,
        dtype=torch.int8,
        device="cpu",
        pin_memory=True,
    )

    from vllm.v1.kv_offload.cpu.gpu_worker import SingleDirectionOffloadingHandler

    kv_cache_groups_data_refs = [
        [
            type("DR", (), dict(tensor_idx=0, page_size_bytes=gpu_row_stride))(),
        ]
    ]
    for dr in kv_cache_groups_data_refs[0]:
        dr.mapping = None

    handler = SingleDirectionOffloadingHandler(
        gpu_tensors=[gpu_tensor],
        cpu_tensors=[cpu_tensor],
        blocks_per_chunk=blocks_per_chunk,
        kv_cache_groups_data_refs=kv_cache_groups_data_refs,
        gpu_to_cpu=True,
        compact_geometry=positional_geom,
    )

    handler._mmap_region = _mock_region_for_cpu_tensor(cpu_tensor)

    addr = _compact_address(0, local_page_size * blocks_per_chunk, group_idx=1)
    gpu_block_ids = np.array([0], dtype=np.int64)
    src_spec = GPULoadStoreSpec(
        gpu_block_ids.tolist(),
        group_sizes=(1,),
        block_indices=(0,),
    )
    dst_spec = CompactCPULoadStoreSpec([addr])

    with pytest.raises(RuntimeError, match="compact geometry is None"):
        handler.transfer_async(55, src_spec, dst_spec)

    addr2 = _compact_address(0, local_page_size * blocks_per_chunk, group_idx=2)
    dst_spec2 = CompactCPULoadStoreSpec([addr2])

    with pytest.raises(RuntimeError, match="out of range"):
        handler.transfer_async(56, src_spec, dst_spec2)

    handler.shutdown()


def _mock_region_for_cpu_tensor(cpu_tensor: torch.Tensor) -> object:
    """Build a minimal mock SharedOffloadRegion for the given CPU tensor.

    Compact transfers now require ``_mmap_region`` to be set on the
    handler; this helper creates an object with the expected public
    interface (``base_ptr``, ``total_size_bytes``) pointing at the
    tensor's backing storage.

    Also provides a no-op ``cleanup()`` so handler shutdown does not
    raise.
    """

    def _noop():
        pass

    return type(
        "_MockRegion",
        (),
        {
            "base_ptr": int(cpu_tensor.data_ptr()),
            "total_size_bytes": int(cpu_tensor.numel()),
            "cleanup": _noop,
        },
    )()


# ---------------------------------------------------------------------------
# CPU-testable: worker->both-handlers propagation without CUDA
# ---------------------------------------------------------------------------


def test_worker_compact_geometry_propagates_to_both_handlers_cpu():
    """CPU-testable: ``CPUOffloadingWorker.configure_compact_geometry``
    must one-shot propagate through each handler's public ``configure``
    method.  This test runs without CUDA, proving propagation works
    independent of GPU availability.

    Pre-fix regression: at HEAD ``cdc9c2a9`` the worker's configure only
    stores ``self._compact_geometry`` without calling the handlers, so
    the handler-level assertions would fail.
    """

    from vllm.v1.kv_offload.cpu.common import (
        CompactGroupGeometry,
        CompactLayerGeometry,
    )

    m = _compact_identity_mapping(1024)
    layer = CompactLayerGeometry(
        layer_name="l0",
        mapping=m,
        local_page_size_bytes=1024,
        canonical_page_size_bytes=1024,
        canonical_offset=0,
        gpu_offset_bytes=0,
    )
    geometry = (
        CompactGroupGeometry(
            layers=(layer,),
            gpu_row_stride=1024,
            local_extent=1024,
            canonical_extent=1024,
            parallel_invariant=True,
        ),
    )

    # Create the worker object without calling __init__ (which needs CUDA),
    # then wire up real-configured mock handlers to test propagation.
    worker = object.__new__(CPUOffloadingWorker)

    class _Handler:
        """Minimal handler stub that mirrors configure_compact_geometry."""

        def __init__(self):
            self._compact_geometry = None

        def configure_compact_geometry(self, groups):
            if self._compact_geometry is not None:
                raise RuntimeError(
                    "compact geometry is already configured and may not be "
                    "replaced; one-shot configuration expected."
                )
            self._compact_geometry = groups

    worker._store_handler = _Handler()
    worker._load_handler = _Handler()
    worker._compact_geometry = None
    worker._mmap_region = None

    # Before configure: all three are None.
    assert worker._compact_geometry is None
    assert worker._store_handler._compact_geometry is None
    assert worker._load_handler._compact_geometry is None

    # Configure once.
    worker.configure_compact_geometry(geometry)

    # Worker stores it.
    assert worker._compact_geometry is geometry
    # Both handlers received it via their public configure method.
    assert worker._store_handler._compact_geometry is geometry
    assert worker._load_handler._compact_geometry is geometry

    # Second call must reject.
    import pytest

    with pytest.raises(RuntimeError, match="already configured"):
        worker.configure_compact_geometry(geometry)


# ---------------------------------------------------------------------------
# Full worker submit_store/submit_load compact route test (native op mocked)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_worker_compact_store_route(mocker):
    """Full worker-level compact ``submit_store`` route through
    ``_transfer_async_compact`` with native ``ops.swap_blocks_batch``
    mocked only.  Verifies:

    - Worker ``configure_compact_geometry`` propagates geometry.
    - ``submit_store`` with ``CompactCPULoadStoreSpec`` routes through
      compact path and calls ``ops.swap_blocks_batch`` (native) with
      ``use_batch_api=False``.
    - ``get_finished`` returns a completed result.
    """
    import uuid

    from vllm.v1.kv_offload.cpu.common import CompactCPULoadStoreSpec

    gpu_row_stride = 2048
    local_page_size = 1024
    blocks_per_chunk = 2

    m0 = _compact_identity_mapping(local_page_size)
    m1 = _compact_identity_mapping(local_page_size)
    geometry = _make_compact_geometry(["l0", "l1"], [m0, m1], [0, 1024], gpu_row_stride)

    num_gpu_blocks = 16
    num_cpu_blocks = 32
    kv_cache_tensors: list[CanonicalKVCacheTensor] = [
        CanonicalKVCacheTensor(
            tensor=torch.zeros(
                num_gpu_blocks, gpu_row_stride, dtype=torch.int8, device="cuda"
            ),
            page_size_bytes=gpu_row_stride,
        )
    ]
    kv_cache_groups_data_refs: list[list[CanonicalKVCacheRef]] = [
        [CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=gpu_row_stride)]
    ]
    kv_caches = CanonicalKVCaches(
        tensors=kv_cache_tensors,
        group_data_refs=kv_cache_groups_data_refs,
    )

    # Create shared mmap region for compact CPU backing.
    cpu_page_size = gpu_row_stride * blocks_per_chunk
    mmap_region = SharedOffloadRegion(
        engine_id=str(uuid.uuid4()),
        num_blocks=num_cpu_blocks,
        rank=0,
        kv_bytes_per_block=cpu_page_size,
        cpu_page_size=cpu_page_size,
    )

    try:
        worker = CPUOffloadingWorker(
            kv_caches=kv_caches,
            blocks_per_chunk=blocks_per_chunk,
            num_cpu_blocks=num_cpu_blocks,
            mmap_region=mmap_region,
        )

        # Configure compact geometry (propagates to both handlers).
        worker.configure_compact_geometry(geometry)

        # Mock only the native op.
        swap_mock = mocker.patch(
            "vllm.v1.kv_offload.cpu.gpu_worker.ops.swap_blocks_batch"
        )

        # Build a compact store spec.
        gpu_block_ids = [0, 1]
        src_spec = GPULoadStoreSpec(
            gpu_block_ids,
            group_sizes=(2,),
            block_indices=(0,),
        )
        canonical_page_size = local_page_size * 2  # two layers
        dst_spec = CompactCPULoadStoreSpec(
            [_compact_address(0, canonical_page_size * blocks_per_chunk, group_idx=0)]
        )

        # Submit store.
        result = worker.submit_store(42, src_spec, dst_spec)
        assert result is True

        # Native op must have been called with use_batch_api=False.
        assert swap_mock.called, (
            "ops.swap_blocks_batch must be called for compact store"
        )
        _, kwargs = swap_mock.call_args
        assert kwargs.get("use_batch_api") is False, (
            "compact path must use use_batch_api=False"
        )

        # Completion gate.
        import time

        end_time = time.time() + 10
        finished = None
        while time.time() < end_time:
            finished = worker.get_finished()
            if finished:
                break
            time.sleep(0.1)
        assert finished, "compact store must complete"
        assert finished[0].job_id == 42
        assert finished[0].success
        assert finished[0].transfer_size > 0

    finally:
        worker.shutdown()
        mmap_region.cleanup()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_worker_compact_load_route(mocker):
    """Full worker-level compact ``submit_load`` route through
    ``_transfer_async_compact`` with native ``ops.swap_blocks_batch``
    mocked only.  Verifies:

    - ``submit_load`` with ``CompactCPULoadStoreSpec`` routes through
      compact path and calls ``ops.swap_blocks_batch`` with
      ``use_batch_api=False``.
    - Completion gate works.
    """
    import uuid

    from vllm.v1.kv_offload.cpu.common import CompactCPULoadStoreSpec

    gpu_row_stride = 1024
    local_page_size = 1024
    blocks_per_chunk = 1

    m = _compact_identity_mapping(local_page_size)
    geometry = _make_compact_geometry(["l0"], [m], [0], gpu_row_stride)

    num_gpu_blocks = 16
    num_cpu_blocks = 32
    kv_cache_tensors: list[CanonicalKVCacheTensor] = [
        CanonicalKVCacheTensor(
            tensor=torch.zeros(
                num_gpu_blocks, gpu_row_stride, dtype=torch.int8, device="cuda"
            ),
            page_size_bytes=gpu_row_stride,
        )
    ]
    kv_cache_groups_data_refs: list[list[CanonicalKVCacheRef]] = [
        [CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=gpu_row_stride)]
    ]
    kv_caches = CanonicalKVCaches(
        tensors=kv_cache_tensors,
        group_data_refs=kv_cache_groups_data_refs,
    )

    cpu_page_size = gpu_row_stride * blocks_per_chunk
    mmap_region = SharedOffloadRegion(
        engine_id=str(uuid.uuid4()),
        num_blocks=num_cpu_blocks,
        rank=0,
        kv_bytes_per_block=cpu_page_size,
        cpu_page_size=cpu_page_size,
    )

    try:
        worker = CPUOffloadingWorker(
            kv_caches=kv_caches,
            blocks_per_chunk=blocks_per_chunk,
            num_cpu_blocks=num_cpu_blocks,
            mmap_region=mmap_region,
        )
        worker.configure_compact_geometry(geometry)

        swap_mock = mocker.patch(
            "vllm.v1.kv_offload.cpu.gpu_worker.ops.swap_blocks_batch"
        )

        gpu_block_ids = [0]
        src_spec = CompactCPULoadStoreSpec(
            [_compact_address(0, local_page_size * blocks_per_chunk, group_idx=0)]
        )
        dst_spec = GPULoadStoreSpec(
            gpu_block_ids,
            group_sizes=(1,),
            block_indices=(0,),
        )

        result = worker.submit_load(99, src_spec, dst_spec)
        assert result is True

        assert swap_mock.called
        _, kwargs = swap_mock.call_args
        assert kwargs.get("use_batch_api") is False, (
            "compact load must use use_batch_api=False"
        )

        import time

        end_time = time.time() + 10
        finished = None
        while time.time() < end_time:
            finished = worker.get_finished()
            if finished:
                break
            time.sleep(0.1)
        assert finished, "compact load must complete"
        assert finished[0].job_id == 99
        assert finished[0].success

    finally:
        worker.shutdown()
        mmap_region.cleanup()


# ---------------------------------------------------------------------------
# Permanent regressions: exact positional src/dst pointer verification
# and real CUDA byte round-trip
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_compact_descriptor_exact_src_dst_pointers(mocker):
    """Mocked compact descriptor transfer verifies exact positional
    src/dst pointer arrays for both store and load.

    The divergence ledger identifies the bug: the owner-aligned rewrite
    passed ``fn(gpu_ptrs, cpu_ptrs, ...)`` regardless of direction, so
    compact load always copied GPU→CPU instead of CPU→GPU.  This test
    asserts that ``ops.swap_blocks_batch`` receives the correct pointer
    arrays in the correct positional order for both directions.

    For store (GPU→CPU):
      - positional arg[0] (src_ptrs) must contain GPU tensor addresses
      - positional arg[1] (dst_ptrs) must contain CPU mmap addresses

    For load (CPU→GPU):
      - positional arg[0] (src_ptrs) must contain CPU mmap addresses
      - positional arg[1] (dst_ptrs) must contain GPU tensor addresses

    Uses identity geometry with a single block so the mapping is
    trivially verifiable: gpu_ptrs[0] == gpu_base_ptr + gpu_offset_bytes,
    cpu_ptrs[0] == cpu_base_ptr + compact_address_byte_offset.
    """

    from vllm.v1.kv_offload.cpu.common import CompactCPULoadStoreSpec

    gpu_row_stride = 1024
    local_page_size = 1024
    blocks_per_chunk = 1

    m = _compact_identity_mapping(local_page_size)
    geometry = _make_compact_geometry(["l0"], [m], [0], gpu_row_stride)

    num_gpu_blocks = 8
    num_cpu_blocks = 16
    gpu_tensor = torch.zeros(
        num_gpu_blocks, gpu_row_stride, dtype=torch.int8, device="cuda"
    )
    cpu_tensor = torch.zeros(
        num_cpu_blocks,
        gpu_row_stride * blocks_per_chunk,
        dtype=torch.int8,
        device="cpu",
        pin_memory=True,
    )

    from vllm.v1.kv_offload.cpu.gpu_worker import SingleDirectionOffloadingHandler

    kv_cache_groups_data_refs = [
        [
            type("DR", (), dict(tensor_idx=0, page_size_bytes=gpu_row_stride))(),
        ]
    ]
    for dr in kv_cache_groups_data_refs[0]:
        dr.mapping = None

    gpu_base_ptr = int(gpu_tensor.data_ptr())
    cpu_base_ptr = int(cpu_tensor.data_ptr())

    # Shared test body for both directions.
    def _run_direction(gpu_to_cpu: bool) -> tuple:
        """Run one compact transfer and return (captured_src, captured_dst)."""
        handler = SingleDirectionOffloadingHandler(
            gpu_tensors=[gpu_tensor],
            cpu_tensors=[cpu_tensor],
            blocks_per_chunk=blocks_per_chunk,
            kv_cache_groups_data_refs=kv_cache_groups_data_refs,
            gpu_to_cpu=gpu_to_cpu,
            compact_geometry=geometry,
        )

        swap_mock = mocker.patch(
            "vllm.v1.kv_offload.cpu.gpu_worker.ops.swap_blocks_batch"
        )
        handler._mmap_region = _mock_region_for_cpu_tensor(cpu_tensor)

        gpu_block_ids = np.array([0], dtype=np.int64)
        if gpu_to_cpu:
            src_spec = GPULoadStoreSpec(
                gpu_block_ids.tolist(),
                group_sizes=(1,),
                block_indices=(0,),
            )
            dst_spec = CompactCPULoadStoreSpec(
                [_compact_address(0, local_page_size * blocks_per_chunk, group_idx=0)]
            )
        else:
            src_spec = CompactCPULoadStoreSpec(
                [_compact_address(0, local_page_size * blocks_per_chunk, group_idx=0)]
            )
            dst_spec = GPULoadStoreSpec(
                gpu_block_ids.tolist(),
                group_sizes=(1,),
                block_indices=(0,),
            )

        handler.transfer_async(1, src_spec, dst_spec)
        handler.shutdown()

        assert swap_mock.called, (
            f"swap_blocks_batch must be called (gpu_to_cpu={gpu_to_cpu})"
        )
        captured_src = int(swap_mock.call_args[0][0][0].item())
        captured_dst = int(swap_mock.call_args[0][1][0].item())
        return captured_src, captured_dst

    store_src, store_dst = _run_direction(gpu_to_cpu=True)
    load_src, load_dst = _run_direction(gpu_to_cpu=False)

    # For store: src=GPU region, dst=CPU region
    assert gpu_base_ptr <= store_src < gpu_base_ptr + gpu_tensor.numel(), (
        f"Store src_ptrs[0]={store_src:#x} must be in GPU tensor range "
        f"[{gpu_base_ptr:#x}, {gpu_base_ptr + gpu_tensor.numel():#x})"
    )
    assert cpu_base_ptr <= store_dst < cpu_base_ptr + cpu_tensor.numel(), (
        f"Store dst_ptrs[0]={store_dst:#x} must be in CPU tensor range "
        f"[{cpu_base_ptr:#x}, {cpu_base_ptr + cpu_tensor.numel():#x})"
    )

    # For load: src=CPU region, dst=GPU region
    assert cpu_base_ptr <= load_src < cpu_base_ptr + cpu_tensor.numel(), (
        f"Load src_ptrs[0]={load_src:#x} must be in CPU tensor range "
        f"[{cpu_base_ptr:#x}, {cpu_base_ptr + cpu_tensor.numel():#x})"
    )
    assert gpu_base_ptr <= load_dst < gpu_base_ptr + gpu_tensor.numel(), (
        f"Load dst_ptrs[0]={load_dst:#x} must be in GPU tensor range "
        f"[{gpu_base_ptr:#x}, {gpu_base_ptr + gpu_tensor.numel():#x})"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_compact_byte_round_trip():
    """Real CUDA compact byte round-trip without mocks.

    Seeds GPU with nonzero pattern, compact stores to CPU, overwrites
    GPU with a different pattern, compact loads back, and asserts exact
    byte-level GPU equality.  This catches the direction bug that the
    divergence ledger's mocked tests missed: if compact load copies
    GPU→CPU instead of CPU→GPU, GPU bytes remain overwritten and the
    final assertion fails.

    Uses a single-block identity mapping with ``blocks_per_chunk=1``
    for the simplest possible round-trip.
    """
    import uuid

    from vllm.v1.kv_offload.cpu.common import CompactCPULoadStoreSpec

    gpu_row_stride = 4096
    local_page_size = 4096
    blocks_per_chunk = 1

    m = _compact_identity_mapping(local_page_size)
    geometry = _make_compact_geometry(["l0"], [m], [0], gpu_row_stride)

    num_gpu_blocks = 8
    num_cpu_blocks = 32
    kv_cache_tensors: list[CanonicalKVCacheTensor] = [
        CanonicalKVCacheTensor(
            tensor=torch.zeros(
                num_gpu_blocks, gpu_row_stride, dtype=torch.int8, device="cuda"
            ),
            page_size_bytes=gpu_row_stride,
        )
    ]
    kv_cache_groups_data_refs: list[list[CanonicalKVCacheRef]] = [
        [CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=gpu_row_stride)]
    ]
    kv_caches = CanonicalKVCaches(
        tensors=kv_cache_tensors,
        group_data_refs=kv_cache_groups_data_refs,
    )

    cpu_page_size = gpu_row_stride * blocks_per_chunk
    worker = None
    mmap_region = None
    try:
        mmap_region = SharedOffloadRegion(
            engine_id=str(uuid.uuid4()),
            num_blocks=num_cpu_blocks,
            rank=0,
            kv_bytes_per_block=cpu_page_size,
            cpu_page_size=cpu_page_size,
        )

        worker = CPUOffloadingWorker(
            kv_caches=kv_caches,
            blocks_per_chunk=blocks_per_chunk,
            num_cpu_blocks=num_cpu_blocks,
            mmap_region=mmap_region,
        )
        worker.configure_compact_geometry(geometry)

        gpu_tensor = kv_cache_tensors[0].tensor  # (num_gpu_blocks, row_stride), int8

        # Phase 1: Seed GPU with nonzero pattern A (0x42).
        pattern_a = torch.full_like(gpu_tensor, 0x42, dtype=torch.int8, device="cuda")
        gpu_tensor.copy_(pattern_a)
        assert torch.equal(gpu_tensor, pattern_a), "GPU must be seeded with pattern A"

        # Phase 2: Compact store GPU → CPU.
        gpu_block_ids = [0]
        store_src = GPULoadStoreSpec(
            gpu_block_ids,
            group_sizes=(1,),
            block_indices=(0,),
        )
        store_dst = CompactCPULoadStoreSpec(
            [_compact_address(0, local_page_size * blocks_per_chunk, group_idx=0)]
        )
        assert worker.submit_store(101, store_src, store_dst), "store must submit"

        # Wait for store completion.
        import time

        end_time = time.time() + 10
        while time.time() < end_time:
            finished = worker.get_finished()
            if finished:
                assert finished[0].success
                break
            time.sleep(0.1)
        else:
            pytest.fail("compact store did not complete within 10 s")

        # Phase 3: Overwrite GPU with pattern B (0x7F).
        pattern_b = torch.full_like(gpu_tensor, 0x7F, dtype=torch.int8, device="cuda")
        gpu_tensor.copy_(pattern_b)
        assert torch.equal(gpu_tensor, pattern_b), (
            "GPU must be overwritten with pattern B"
        )

        # Phase 4: Compact load CPU → GPU (restore pattern A).
        load_src = CompactCPULoadStoreSpec(
            [_compact_address(0, local_page_size * blocks_per_chunk, group_idx=0)]
        )
        load_dst = GPULoadStoreSpec(
            gpu_block_ids,
            group_sizes=(1,),
            block_indices=(0,),
        )
        assert worker.submit_load(102, load_src, load_dst), "load must submit"

        end_time = time.time() + 10
        while time.time() < end_time:
            finished = worker.get_finished()
            if finished:
                assert finished[0].success
                break
            time.sleep(0.1)
        else:
            pytest.fail("compact load did not complete within 10 s")

        # Phase 5: Verify GPU is restored to pattern A.
        assert torch.equal(gpu_tensor[:1], pattern_a[:1]), (
            "GPU block 0 must be restored to pattern A after compact load"
        )

        # GPU blocks that were NOT in the round-trip should remain pattern B.
        assert torch.equal(gpu_tensor[1:], pattern_b[1:]), (
            "GPU blocks 1+ must remain pattern B (not touched by load)"
        )

        # Also verify CPU stored the correct data by checking that the
        # compact CPU region contains pattern A at the expected offset.
        cpu_loaded = torch.frombuffer(
            memoryview(
                mmap_region.mmap_obj  # type: ignore[arg-type]
            ),
            dtype=torch.int8,
            offset=0,  # compact_address.byte_offset == 0
            count=local_page_size * blocks_per_chunk,
        ).clone()
        expected_cpu = pattern_a[:1].cpu().flatten()
        torch.testing.assert_close(cpu_loaded, expected_cpu)

    finally:
        if worker is not None:
            worker.shutdown()
        if mmap_region is not None:
            mmap_region.cleanup()


# ---------------------------------------------------------------------------
# Hardware acceptance requirement: TP2 writer/nonwriter compact coverage
# ---------------------------------------------------------------------------

# Real TP2 (two-rank compact store/load with writer/nonwriter roles and
# all-rank load) cannot be tested honestly in the existing unit-test
# suite because there is no distributed runtime or NCCL test fixture
# available.  A fake single-process "TP2" test that duplicates local
# state would not exercise the cross-rank consensus, shared-region
# partitioning, rank-conditional store gating, or per-rank CPU→GPU load
# that real TP2 requires.
#
# Until TP2 hardware acceptance is added to a separate distributed test:
#
#   1. Deploy the fixed candidate to TP2 with two physical ranks.
#   2. Run a warm 50K-token CPU-hop replay.
#   3. Verify:
#      - writer rank stores nonzero bytes through compact path.
#      - nonwriter rank produces zero-byte store completion (store
#        gated by _is_store_writer) through the zero-descriptor path.
#      - both ranks complete compact load with nonzero bytes and
#        recover exact rank-local KV content.
#      - exact replay output, same PID, zero errors.
#      - external tokens are fully resolved (no zero local hit rate).
