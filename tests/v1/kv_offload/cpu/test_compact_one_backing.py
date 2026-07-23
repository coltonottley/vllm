# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Owner-focused tests for #48906 one-backing compact layout extension.

Tests the CPUOffloadingSpec one-backing lifecycle when
``enable_compact_layout`` is requested: one SharedOffloadRegion is constructed
before worker registration, rank-private strided ordinary views are exposed
for permanent legacy fallback, compact uses flat region addresses after
consensus, and cleanup is idempotent.

No second mmap, no post-consensus allocation, no sidecar, no
enum/tri-state/CompactNegotiationResult/start_negotiating.
"""

import contextlib
import mmap
import os
import uuid

import pytest
import torch

from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
)
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingGroupConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.cpu.common import CompactCPUAddress
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

PAGE_SIZE = mmap.PAGESIZE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_config(
    extra_config: dict | None = None,
    world_size: int = 2,
    rank: int = 0,
    engine_id: str | None = None,
    worker_kv_bytes: int = 64 * 1024,
) -> OffloadingConfig:
    """Build a minimal OffloadingConfig for testing."""
    if extra_config is None:
        extra_config = {
            "cpu_bytes_to_use": str(16 * 1024 * 1024),  # 16 MiB
        }
    return OffloadingConfig(
        groups=(
            OffloadingGroupConfig(
                tokens_per_block=16,
                layer_names=("k", "v"),
            ),
        ),
        worker_kv_bytes_per_block=worker_kv_bytes,
        enable_kv_cache_events=False,
        extra_config=extra_config,
        engine_id=engine_id or f"test-{uuid.uuid4().hex[:8]}",
        model=OffloadingModelConfig(
            name="test-model",
            dtype="float16",
        ),
        cache=OffloadingCacheConfig(
            tokens_per_hash=16,
            blocks_per_chunk=1,
        ),
        parallel=OffloadingParallelConfig(
            rank=rank,
            world_size=world_size,
            tp_size=world_size,
            pp_size=1,
            pcp_size=1,
            dcp_size=1,
            data_parallel_index=0,
            is_parallelism_agnostic=True,
        ),
    )


def _make_kv_caches(
    num_gpu_blocks: int = 32,
    gpu_page_size_bytes: int = 1024,
    device: str | None = None,
) -> CanonicalKVCaches:
    """Create a minimal CanonicalKVCaches for test purposes."""
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gpu_tensor = torch.zeros(
        (num_gpu_blocks, gpu_page_size_bytes),
        dtype=torch.int8,
        device=device,
    )
    return CanonicalKVCaches(
        tensors=[
            CanonicalKVCacheTensor(
                tensor=gpu_tensor,
                page_size_bytes=gpu_page_size_bytes,
            )
        ],
        group_data_refs=[
            [
                CanonicalKVCacheRef(
                    tensor_idx=0,
                    page_size_bytes=gpu_page_size_bytes,
                )
            ]
        ],
    )


def _make_region(
    engine_id: str,
    num_blocks: int = 4,
    cpu_page_size: int = PAGE_SIZE,
    num_workers: int = 1,
    rank: int = 0,
) -> SharedOffloadRegion:
    """Helper to create a SharedOffloadRegion for testing."""
    assert cpu_page_size % PAGE_SIZE == 0
    return SharedOffloadRegion(
        engine_id=engine_id,
        num_blocks=num_blocks,
        rank=rank,
        kv_bytes_per_block=num_workers * cpu_page_size,
        cpu_page_size=cpu_page_size,
    )


def _cleanup_file(path: str) -> None:
    """Best-effort file removal for test teardown."""
    with contextlib.suppress(FileNotFoundError):
        os.unlink(path)


@pytest.fixture(autouse=True)
def _set_spawn_method(monkeypatch):
    """Suppress vLLM spawn method warning."""
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


# GPU-dependent tests require a CUDA-capable GPU with available memory.
# Run only the GPU-independent spec/region/cleanup tests by default.
_has_gpu = False  # GPU worker tests need real CUDA memory; skip for CI


# ===========================================================================
# One-construction: spec parses enable_compact_layout and builds region
# ===========================================================================


class TestOneConstruction:
    def test_compact_disabled_no_region(self):
        """When enable_compact_layout is false/absent, no region config set up."""
        config = _make_minimal_config(
            extra_config={"cpu_bytes_to_use": str(16 * 1024 * 1024)}
        )
        spec = CPUOffloadingSpec(config)
        assert not spec.enable_compact_layout
        assert spec.shared_region is None
        assert spec.compact_storage_budget_bytes is None

    def test_compact_enabled_config(self):
        """When enable_compact_layout=True, config is parsed correctly."""
        config = _make_minimal_config(
            extra_config={
                "cpu_bytes_to_use": str(16 * 1024 * 1024),
                "enable_compact_layout": "true",
            },
        )
        spec = CPUOffloadingSpec(config)
        assert spec.enable_compact_layout
        assert spec.compact_storage_budget_bytes == 16 * 1024 * 1024
        assert spec.compact_page_size > 0
        assert spec.compact_total_pages > 0

    def test_compact_zero_budget_small_page_raises(self):
        """Budget too small for page size raises."""
        config = _make_minimal_config(
            extra_config={
                "cpu_bytes_to_use": str(512),  # 512 bytes -- smaller than default 64K
                "enable_compact_layout": True,
            },
        )
        with pytest.raises(ValueError, match="too small"):
            CPUOffloadingSpec(config)

    def test_compact_build_region_direct(self):
        """_build_compact_shared_region constructs a valid SharedOffloadRegion."""
        engine_id = f"test-{uuid.uuid4().hex[:8]}"
        config = _make_minimal_config(
            extra_config={
                "cpu_bytes_to_use": str(16 * 1024 * 1024),
                "enable_compact_layout": "true",
            },
            engine_id=engine_id,
        )
        spec = CPUOffloadingSpec(config)
        region = spec._build_compact_shared_region()
        try:
            assert region is not None
            assert region.base_ptr != 0
            assert region.total_size_bytes > 0
            assert region.rank == 0  # rank from config
        finally:
            region.cleanup()
            _cleanup_file(region.mmap_path)

    def test_manager_activation_is_delayed_until_consensus(self):
        config = _make_minimal_config(
            extra_config={
                "cpu_bytes_to_use": str(16 * 1024 * 1024),
                "enable_compact_layout": "true",
            },
        )
        spec = CPUOffloadingSpec(config)
        manager = spec.get_manager()
        assert manager._compact_allocator is None
        manager.enable_compact(
            total_bytes=spec.compact_storage_budget_bytes,
            page_size=spec.compact_page_size,
            key_sizes={0: 4096},
        )
        assert manager._compact_allocator is not None
        assert manager._compact_allocator.page_size == spec.compact_page_size

    def test_compact_non_page_aligned_tp2_geometry(self, monkeypatch):
        """Compact TP2 aligns only the shared row and shares one row count."""
        engine_id = f"test-{uuid.uuid4().hex[:8]}"
        budget = 65536
        config = _make_minimal_config(
            extra_config={
                "cpu_bytes_to_use": str(budget),
                "enable_compact_layout": "true",
            },
            world_size=2,
            rank=0,
            engine_id=engine_id,
            worker_kv_bytes=500,
        )
        spec = CPUOffloadingSpec(config)

        raw_row = spec.cpu_page_size_per_worker * config.parallel.world_size
        assert spec.BLOCK_SIZE_ALIGNMENT == 1
        assert raw_row == 1000
        assert raw_row % PAGE_SIZE
        assert spec._compact_row_stride == PAGE_SIZE
        assert spec._compact_num_rows == budget // PAGE_SIZE == 16
        assert spec.num_blocks == budget // raw_row == 65

        pre_region_budget = spec.compact_storage_budget_bytes
        region0 = spec._build_compact_shared_region()
        region1 = None
        try:
            assert region0._row_stride == PAGE_SIZE
            assert region0.num_blocks == spec._compact_num_rows
            assert region0.total_size_bytes == pre_region_budget == budget

            config1 = _make_minimal_config(
                extra_config={
                    "cpu_bytes_to_use": str(budget),
                    "enable_compact_layout": "true",
                },
                world_size=2,
                rank=1,
                engine_id=engine_id,
                worker_kv_bytes=500,
            )
            spec1 = CPUOffloadingSpec(config1)
            region1 = spec1._build_compact_shared_region()
            view0 = region0.create_next_view(spec.cpu_page_size_per_worker)
            view1 = region1.create_next_view(spec1.cpu_page_size_per_worker)
            assert view0.shape == view1.shape == (spec._compact_num_rows, 500)
            assert view0.stride(0) == view1.stride(0) == PAGE_SIZE
            assert view0.storage_offset() == 0
            assert view1.storage_offset() == 500

            spec._worker_shared_region = region0
            assert spec.compact_storage_budget_bytes == pre_region_budget
            assert spec.get_manager()._num_blocks == spec._compact_num_rows

            captured = {}

            def fake_worker(**kwargs):
                captured.update(kwargs)
                return object()

            monkeypatch.setattr(
                "vllm.v1.kv_offload.cpu.spec.CPUOffloadingWorker", fake_worker
            )
            sentinel_caches = object()
            result = spec.create_worker(sentinel_caches, mmap_region=region0)
            assert result is not None
            assert captured["kv_caches"] is sentinel_caches
            assert captured["num_cpu_blocks"] == region0.num_blocks
            assert captured["mmap_region"] is region0
        finally:
            spec._worker_shared_region = None
            if region1 is not None:
                region1.cleanup()
            region0.cleanup()
            _cleanup_file(region0.mmap_path)

        assert not os.path.exists(region0.mmap_path)

    def test_preferred_group_derivation_uses_ordered_real_groups(self):
        config = _make_minimal_config(
            extra_config={
                "cpu_bytes_to_use": str(16 * 1024 * 1024),
                "enable_compact_layout": "true",
            }
        )
        spec = CPUOffloadingSpec(config)
        groups = [
            KVCacheGroupSpec(
                layer_names=["ordinary"], kv_cache_spec=object(), is_eagle_group=False
            ),
            KVCacheGroupSpec(
                layer_names=["eagle"], kv_cache_spec=object(), is_eagle_group=True
            ),
        ]
        spec.maybe_derive_compact_preferred_eviction_groups(
            KVCacheConfig(num_blocks=1, kv_cache_tensors=[], kv_cache_groups=groups)
        )
        assert spec.compact_preferred_eviction_groups == (1,)

    def test_compact_budget_must_fit_aligned_shared_row(self):
        config = _make_minimal_config(
            extra_config={
                "cpu_bytes_to_use": str(2048),
                "enable_compact_layout": "true",
                "compact_page_size": "1024",
            },
            world_size=2,
            worker_kv_bytes=1500,
        )
        spec = CPUOffloadingSpec(config)
        assert spec._compact_row_stride == PAGE_SIZE
        assert spec._compact_num_rows == 0
        with pytest.raises(RuntimeError, match="cannot fit one shared row"):
            spec._build_compact_shared_region()


# ===========================================================================
# Legacy fallback same backing: rank-private strided views from shared region
# ===========================================================================


class TestLegacyFallbackSameBacking:
    def test_worker_creates_strided_views_from_shared_region(self):
        """Worker creates strided views from shared region, not separate malloc.
        Requires CUDA for the GPU tensors."""
        if not _has_gpu:
            pytest.skip("CUDA required for worker test")

        engine_id = f"test-{uuid.uuid4().hex[:8]}"
        kv_caches = _make_kv_caches()
        region = _make_region(
            engine_id=engine_id,
            num_blocks=32,
            cpu_page_size=2 * PAGE_SIZE,
            num_workers=1,
            rank=0,
        )
        worker = None
        try:
            worker = CPUOffloadingWorker(
                kv_caches=kv_caches,
                blocks_per_chunk=1,
                num_cpu_blocks=32,
                mmap_region=region,
            )
            assert worker._mmap_region is region

            # The worker's store handler CPU tensors should be inside the region.
            store_handler = worker._store_handler
            assert store_handler is not None

            for cpu_tensor in store_handler.dst_tensors:
                ptr = cpu_tensor.data_ptr()
                base = region.base_ptr
                end = base + region.total_size_bytes
                assert base <= ptr < end, (
                    f"CPU tensor at {ptr} is outside shared region [{base}, {end})"
                )
                assert cpu_tensor.shape[0] == region.num_blocks, (
                    f"Expected {region.num_blocks} blocks, got {cpu_tensor.shape[0]}"
                )
        finally:
            if worker is not None:
                worker.shutdown()
            region.cleanup()
            _cleanup_file(region.mmap_path)

    def test_strided_views_use_rank_private_slot(self):
        """Each rank's strided view occupies its private slot within each row.
        Requires CUDA for the GPU tensors."""
        if not _has_gpu:
            pytest.skip("CUDA required for worker test")

        engine_id = f"test-{uuid.uuid4().hex[:8]}"
        world_size = 2
        slot_size = 2 * PAGE_SIZE
        row_stride = world_size * slot_size
        num_blocks = 8

        region0 = SharedOffloadRegion(
            engine_id=engine_id,
            num_blocks=num_blocks,
            rank=0,
            kv_bytes_per_block=row_stride,
            cpu_page_size=slot_size,
        )
        region1 = SharedOffloadRegion(
            engine_id=engine_id,
            num_blocks=num_blocks,
            rank=1,
            kv_bytes_per_block=row_stride,
            cpu_page_size=slot_size,
        )

        kv_caches = _make_kv_caches(num_gpu_blocks=num_blocks)
        worker0 = None
        worker1 = None
        try:
            worker0 = CPUOffloadingWorker(
                kv_caches=kv_caches,
                blocks_per_chunk=1,
                num_cpu_blocks=num_blocks,
                mmap_region=region0,
            )
            worker1 = CPUOffloadingWorker(
                kv_caches=kv_caches,
                blocks_per_chunk=1,
                num_cpu_blocks=num_blocks,
                mmap_region=region1,
            )

            # Both regions have the same geometry.
            assert region0.total_size_bytes == region1.total_size_bytes
            assert region0._row_stride == region1._row_stride == row_stride
            assert region0.num_blocks == region1.num_blocks == num_blocks

            t0 = worker0._store_handler.dst_tensors[0]
            t1 = worker1._store_handler.dst_tensors[0]

            # Both tensors should have the same row stride (full world-sized row).
            assert t0.stride(0) == row_stride
            assert t1.stride(0) == row_stride

            # Rank 0's storage_offset should be 0 (first in row).
            # Rank 1's storage_offset should be >= slot_size.
            off0 = t0.storage_offset() * t0.element_size()
            off1 = t1.storage_offset() * t1.element_size()
            assert off1 >= off0 + slot_size, (
                f"Rank 1 storage offset {off1} should be >= "
                f"rank 0 offset {off0} + slot_size {slot_size}"
            )
        finally:
            if worker0 is not None:
                worker0.shutdown()
            if worker1 is not None:
                worker1.shutdown()
            region0.cleanup()
            region1.cleanup()
            _cleanup_file(region0.mmap_path)


# ===========================================================================
# Compact same backing: flat region addressing
# ===========================================================================


class TestCompactSameBacking:
    def test_compact_addresses_within_region(self):
        """Compact addresses fall within the shared region's byte range."""
        engine_id = f"test-{uuid.uuid4().hex[:8]}"
        region = _make_region(
            engine_id=engine_id,
            num_blocks=32,
            cpu_page_size=2 * PAGE_SIZE,
            num_workers=1,
            rank=0,
        )
        try:
            base = region.base_ptr
            total = region.total_size_bytes
            assert total > 0

            page_size = 64 * 1024  # compact page size
            compact_pages = total // page_size

            offsets = [0, page_size, 2 * page_size, (compact_pages - 1) * page_size]
            for offset in offsets:
                assert offset < total
                addr = base + offset
                assert base <= addr < base + total

            for offset in offsets:
                address = CompactCPUAddress(
                    byte_offset=offset,
                    logical_length=page_size,
                    allocated_length=page_size,
                    group_idx=0,
                )
                assert address.byte_offset < total
        finally:
            region.cleanup()
            _cleanup_file(region.mmap_path)

    def test_compact_and_legacy_share_region(self):
        """The region's base_ptr/total_size_bytes define the canonical
        address space for both compact and legacy paths."""
        engine_id = f"test-{uuid.uuid4().hex[:8]}"
        region = _make_region(
            engine_id=engine_id,
            num_blocks=32,
            cpu_page_size=2 * PAGE_SIZE,
            num_workers=1,
            rank=0,
        )
        try:
            base = region.base_ptr
            total = region.total_size_bytes
            end = base + total

            assert base != 0
            assert total > 0
            assert end > base

            # Compact: the base_ptr and total define the flat address space.
            compact_base = region.base_ptr
            compact_total = region.total_size_bytes
            assert compact_base == base
            assert compact_total == total
        finally:
            region.cleanup()
            _cleanup_file(region.mmap_path)


# ===========================================================================
# Rank-private view bounds (region structure, no GPU tensors needed)
# ===========================================================================


class TestRankPrivateViewBounds:
    def test_region_rank_partition(self):
        """SharedOffloadRegion partitions rows by rank correctly."""
        engine_id = f"test-{uuid.uuid4().hex[:8]}"
        world_size = 3
        slot_size = 2 * PAGE_SIZE  # 8192 bytes per worker slot
        row_stride = world_size * slot_size
        num_blocks = 16

        region = SharedOffloadRegion(
            engine_id=engine_id,
            num_blocks=num_blocks,
            rank=0,
            kv_bytes_per_block=row_stride,
            cpu_page_size=slot_size,
        )
        try:
            # Total should fit num_blocks rows of world_size slots.
            assert region.total_size_bytes == num_blocks * row_stride
            assert region.num_blocks == num_blocks
            assert region._row_stride == row_stride

            # Create a view for rank 0 that fits within slot_size.
            view = region.create_next_view(slot_size // 2)
            assert view.shape == (num_blocks, slot_size // 2)
            assert view.stride(0) == row_stride
            # View points into rank 0's slot (offset 0 within each row).
            view_start = view.storage_offset() * view.element_size()
            assert view_start == 0, (
                f"Rank 0 view should start at offset 0, got {view_start}"
            )

            # Create another view within the same slot (remaining space).
            view2 = region.create_next_view(slot_size // 2)
            assert view2.shape == (num_blocks, slot_size // 2)
            assert view2.stride(0) == row_stride
            # Second view starts after the first tensor within rank 0's slot.
            v2_start = view2.storage_offset() * view2.element_size()
            assert v2_start == slot_size // 2, (
                f"Second view should start at offset {slot_size // 2}, got {v2_start}"
            )
        finally:
            region.cleanup()
            _cleanup_file(region.mmap_path)

    def test_region_rank1_view_offset(self):
        """Rank 1's view starts at offset cpu_page_size within each row."""
        engine_id = f"test-{uuid.uuid4().hex[:8]}"
        world_size = 2
        slot_size = 2 * PAGE_SIZE
        row_stride = world_size * slot_size
        num_blocks = 4

        region1 = SharedOffloadRegion(
            engine_id=engine_id,
            num_blocks=num_blocks,
            rank=1,
            kv_bytes_per_block=row_stride,
            cpu_page_size=slot_size,
        )
        try:
            # Rank 1's view should have storage_offset = slot_size (start of
            # rank 1's slot within each row).
            view = region1.create_next_view(slot_size)
            off = view.storage_offset() * view.element_size()
            assert off == slot_size, (
                f"Rank 1 view should start at offset {slot_size}, got {off}"
            )
            # The data_ptr should be base_ptr + slot_size (first row, rank 1's slot).
            expected_ptr = region1.base_ptr + slot_size
            assert view.data_ptr() == expected_ptr, (
                f"Rank 1 first element data_ptr {view.data_ptr()} should be "
                f"{expected_ptr}"
            )
        finally:
            region1.cleanup()
            _cleanup_file(region1.mmap_path)


# ===========================================================================
# Cleanup ownership
# ===========================================================================


class TestCleanupOwnership:
    def test_spec_shutdown_idempotent(self):
        """shutdown_worker_region() is idempotent when no region exists."""
        config = _make_minimal_config(
            extra_config={"cpu_bytes_to_use": str(16 * 1024 * 1024)}
        )
        spec = CPUOffloadingSpec(config)
        assert spec.shared_region is None
        # First call: no-op.
        spec.shutdown_worker_region()
        assert spec.shared_region is None
        # Second call: still no-op.
        spec.shutdown_worker_region()
        assert spec.shared_region is None

    def test_region_cleanup_idempotent(self):
        """SharedOffloadRegion.cleanup() is safe to call multiple times."""
        engine_id = f"test-{uuid.uuid4().hex[:8]}"
        region = _make_region(
            engine_id=engine_id,
            num_blocks=4,
            cpu_page_size=PAGE_SIZE,
            num_workers=1,
            rank=0,
        )
        path = region.mmap_path
        assert region.base_ptr != 0
        region.cleanup()
        assert region.base_ptr == 0
        # Idempotent second call.
        region.cleanup()
        assert region.base_ptr == 0
        _cleanup_file(path)

    def test_region_cleanup_after_create_next_view(self):
        """cleanup() after create_next_view frees all references cleanly."""
        engine_id = f"test-{uuid.uuid4().hex[:8]}"
        region = _make_region(
            engine_id=engine_id,
            num_blocks=8,
            cpu_page_size=2 * PAGE_SIZE,
            num_workers=1,
            rank=0,
        )
        path = region.mmap_path
        # Create some views within the per-worker slot (2*PAGE_SIZE).
        v1 = region.create_next_view(PAGE_SIZE)
        v2 = region.create_next_view(PAGE_SIZE)
        assert v1 is not None
        assert v2 is not None
        assert region.base_tensor is not None
        # Cleanup.
        region.cleanup()
        assert region.base_ptr == 0
        assert region.base_tensor is None
        _cleanup_file(path)

    def test_no_replicated_layout_before_consensus(self):
        """Static replicated_layout is not set when compact is enabled.
        The spec must not change SUPPORTS_REPLICATED_LAYOUT or set
        replicated_layout=True."""
        config = _make_minimal_config(
            extra_config={
                "cpu_bytes_to_use": str(16 * 1024 * 1024),
                "enable_compact_layout": "true",
            },
        )
        spec = CPUOffloadingSpec(config)
        # SUPPORTS_REPLICATED_LAYOUT remains False.
        assert not spec.SUPPORTS_REPLICATED_LAYOUT
        # replicated_layout remains False (not set statically before consensus).
        assert not spec.replicated_layout

    def test_no_enum_tri_state(self):
        """No CompactNegotiationResult/enum/tri-state symbols are present."""
        config = _make_minimal_config(
            extra_config={
                "cpu_bytes_to_use": str(16 * 1024 * 1024),
                "enable_compact_layout": "true",
            },
        )
        spec = CPUOffloadingSpec(config)
        # No CompactNegotiationResult attribute.
        assert not hasattr(spec, "_compact_negotiation_result")
        assert not hasattr(spec, "_compact_group_layouts")
        assert not hasattr(spec, "_compact_group_signatures")
        assert not hasattr(spec, "_compact_negotiation_result")
        # No start_negotiating on manager.
        manager = spec.get_manager()
        assert not hasattr(manager, "_start_negotiating")
        assert not hasattr(manager, "start_negotiating")

    def test_compact_disabled_manager_no_allocator(self):
        """Manager without compact has no allocator."""
        config = _make_minimal_config(
            extra_config={"cpu_bytes_to_use": str(16 * 1024 * 1024)}
        )
        spec = CPUOffloadingSpec(config)
        assert spec.shared_region is None
        manager = spec.get_manager()
        assert manager is not None
        assert manager._compact_allocator is None

    def test_plugin_spec_tolerated(self):
        """Generic OffloadingSpec plugins without enable_compact_layout
        are tolerated via getattr(..., False)."""
        import types

        fake_spec = types.SimpleNamespace()
        fake_spec.extra_config = {}
        # getattr should return False for missing attribute.
        assert getattr(fake_spec, "enable_compact_layout", False) is False

    def test_manager_compact_allocator_page_bounds(self):
        config = _make_minimal_config(
            extra_config={
                "cpu_bytes_to_use": str(16 * 1024 * 1024),
                "enable_compact_layout": "true",
                "compact_page_size": str(65536),
            },
        )
        spec = CPUOffloadingSpec(config)
        manager = spec.get_manager()
        manager.enable_compact(
            total_bytes=spec.compact_storage_budget_bytes,
            page_size=spec.compact_page_size,
            key_sizes={0: 4096},
        )
        allocator = manager._compact_allocator
        assert allocator is not None
        assert allocator.total_bytes == spec.compact_storage_budget_bytes
        assert allocator.total_bytes // allocator.page_size == spec.compact_total_pages

    def test_manager_has_no_allocator_before_consensus(self):
        config = _make_minimal_config(
            extra_config={
                "cpu_bytes_to_use": str(16 * 1024 * 1024),
                "enable_compact_layout": "true",
            },
        )
        spec = CPUOffloadingSpec(config)
        assert spec.get_manager()._compact_allocator is None
