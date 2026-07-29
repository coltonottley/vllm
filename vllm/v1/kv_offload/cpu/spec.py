# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
from typing_extensions import override

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingHistogramMetadata,
    OffloadingManager,
    OffloadingMetricMetadata,
    OffloadingSpec,
    OffloadingWorker,
)
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.cpu.common import CPUOffloadingMetrics
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

logger = init_logger(__name__)


def _parse_enable_compact_layout(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError(f"enable_compact_layout must be a boolean, got {value!r}")


class CPUOffloadingSpec(OffloadingSpec):
    BLOCK_SIZE_ALIGNMENT = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
    SUPPORTS_REPLICATED_LAYOUT = False

    @classmethod
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        definitions: dict[str, OffloadingMetricMetadata] = {
            CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of CPU KV-cache space currently pinned by active "
                    "transfers (0.0 = idle, 1.0 = saturated). Sustained high "
                    "values indicate transfers (stores or promotions) may be "
                    "dropped due to insufficient capacity."
                ),
            ),
            CPUOffloadingMetrics.CPU_ALLOCATED_BYTES: OffloadingGaugeMetadata(
                documentation=(
                    "Exact bytes currently resident in compact CPU KV storage. "
                    "Zero in legacy (non-compact) mode."
                ),
            ),
            CPUOffloadingMetrics.CPU_FREE_BYTES: OffloadingGaugeMetadata(
                documentation="Exact free bytes in compact CPU KV storage.",
            ),
            CPUOffloadingMetrics.CPU_LARGEST_FREE_EXTENT_BYTES: (
                OffloadingGaugeMetadata(
                    documentation=(
                        "Largest contiguous free extent in compact CPU KV storage."
                    ),
                )
            ),
            CPUOffloadingMetrics.CPU_FRAGMENTATION_RATIO: OffloadingGaugeMetadata(
                documentation=(
                    "External fragmentation ratio of compact CPU KV free space."
                ),
            ),
            CPUOffloadingMetrics.CPU_CACHE_WRITE_USAGE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of CPU KV-cache space currently pinned by "
                    "in-flight stores that have not yet "
                    "completed (0.0 = idle, 1.0 = saturated)."
                ),
            ),
            CPUOffloadingMetrics.CPU_CACHE_READ_USAGE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of CPU KV-cache space currently pinned by "
                    "in-flight loads that have not yet "
                    "completed (0.0 = idle, 1.0 = saturated)."
                ),
            ),
            CPUOffloadingMetrics.CPU_ALLOCATION_SIZE: OffloadingHistogramMetadata(
                documentation=(
                    "Histogram of the number of CPU blocks requested by each "
                    "KV offload prepare_store call."
                ),
                buckets=(1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144),
            ),
        }
        store_threshold = int(extra_config.get("store_threshold", 0))
        if store_threshold >= 2:
            definitions[CPUOffloadingMetrics.STORES_SKIPPED] = (
                OffloadingCounterMetadata(
                    documentation=(
                        "Number of KV offload stores skipped because the reuse "
                        "threshold was not reached."
                    ),
                )
            )
        return definitions

    def __init__(self, config: OffloadingConfig):
        super().__init__(config)

        cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
        if not cpu_bytes_to_use:
            raise Exception(
                "cpu_bytes_to_use must be specified in kv_connector_extra_config"
            )

        world_size = config.parallel.world_size
        self.num_blocks = 0
        self.kv_bytes_per_chunk = 0
        self.cpu_page_size_per_worker = 0
        self.replicated_layout = (
            config.replicated_layout and self.SUPPORTS_REPLICATED_LAYOUT
        )
        if config.worker_kv_bytes_per_block > 0 and world_size > 0:
            num_copies = 1 if self.replicated_layout else world_size
            kv_bytes_per_block = config.worker_kv_bytes_per_block * num_copies
            kv_bytes_per_chunk = kv_bytes_per_block * self.blocks_per_chunk

            # calculate cpu_page_size_per_worker
            self.cpu_page_size_per_worker = kv_bytes_per_chunk // num_copies

            # calculate num_blocks
            aligned_kv_bytes_per_chunk = round_up(
                kv_bytes_per_chunk, self.BLOCK_SIZE_ALIGNMENT
            )
            self.num_blocks = int(cpu_bytes_to_use) // aligned_kv_bytes_per_chunk

            # Expose aligned_kv_bytes_per_chunk as
            # kv_bytes_per_chunk. Note that this might contain
            # some padding. i.e. each offloaded block is of the form,
            # |--- W0-B0---|---- W1-B0---| ... |---- Wn-B0---| *** maybe-pad *** |
            # or |--- B0 (single copy) ---| *** maybe-pad *** |
            self.kv_bytes_per_chunk = aligned_kv_bytes_per_chunk

        # scheduler-side
        self._manager: OffloadingManager | None = None

        # worker-side
        self._worker: CPUOffloadingWorker | None = None

        self.eviction_policy: str = self.extra_config.get("eviction_policy", "lru")
        self.cache_policy_module_path: str | None = self.extra_config.get(
            "cache_policy_module_path"
        )

        raw_compact = self.extra_config.get("enable_compact_layout")
        self._compact_layout_requested = (
            _parse_enable_compact_layout(raw_compact)
            if raw_compact is not None
            else False
        )
        self._compact_page_size = int(self.extra_config.get("compact_page_size", 65536))
        if self._compact_page_size <= 0:
            raise ValueError("compact_page_size must be positive")
        self._compact_cpu_bytes = int(cpu_bytes_to_use)
        if (
            self._compact_layout_requested
            and self._compact_cpu_bytes < self._compact_page_size
        ):
            raise ValueError("compact CPU budget is too small for one compact page")
        self._compact_preferred_eviction_groups: set[int] = set(
            self.extra_config.get("compact_preferred_eviction_groups", [])
        )
        self.offload_latest_prompt_tail_only = bool(
            self.extra_config.get("offload_latest_prompt_tail_only", False)
        )
        self._worker_shared_region: SharedOffloadRegion | None = None

    # --- Preferred eviction group derivation ---

    def maybe_derive_compact_preferred_eviction_groups(
        self, kv_cache_config: KVCacheConfig
    ) -> None:
        """Derive preferred eviction groups from KVCacheConfig metadata.

        A group is preferred if it has a sliding window (SWA) or is an
        EAGLE group.  If ``compact_preferred_eviction_groups`` was already
        set explicitly in config, it is preserved unchanged.
        """
        if self._compact_preferred_eviction_groups:
            return

        derived: set[int] = set()
        for group_idx, group in enumerate(kv_cache_config.kv_cache_groups):
            if _is_preferred_eviction_group(group):
                derived.add(group_idx)
        self._compact_preferred_eviction_groups = derived

    # --- Manager construction ---

    @override
    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            store_threshold = int(self.extra_config.get("store_threshold", 0))
            max_tracker_size = int(self.extra_config.get("max_tracker_size", 64_000))

            num_blocks = (
                self._compact_num_rows
                if self._compact_layout_requested
                else self.num_blocks
            )
            self._manager = CPUOffloadingManager(
                num_blocks=num_blocks,
                cache_policy=self.eviction_policy,  # type: ignore[arg-type]
                cache_policy_module_path=self.cache_policy_module_path,
                enable_events=self.kv_events_config.enable_kv_cache_events,
                store_threshold=store_threshold,
                max_tracker_size=max_tracker_size,
            )
        return self._manager

    @property
    def compact_layout_requested(self) -> bool:
        return self._compact_layout_requested

    @property
    def enable_compact_layout(self) -> bool:
        return self._compact_layout_requested

    @property
    def shared_region(self) -> SharedOffloadRegion | None:
        return self._worker_shared_region

    @property
    def compact_preferred_eviction_groups(self) -> tuple[int, ...]:
        return tuple(sorted(self._compact_preferred_eviction_groups))

    @property
    def compact_storage_budget_bytes(self) -> int | None:
        if not self._compact_layout_requested:
            return None
        if self._worker_shared_region is not None:
            return self._worker_shared_region.total_size_bytes
        return self._compact_num_rows * self._compact_row_stride

    @property
    def compact_page_size(self) -> int:
        return self._compact_page_size

    @property
    def compact_total_pages(self) -> int:
        budget = self.compact_storage_budget_bytes
        return 0 if budget is None else budget // self._compact_page_size

    @property
    def _compact_row_stride(self) -> int:
        """Aligned row stride for the compact shared region.

        The raw row (``cpu_page_size_per_worker * world_size``) may not
        be page-aligned.  Align to ``SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT``
        (``mmap.PAGESIZE``) so that ``SharedOffloadRegion.__init__`` asserts
        ``kv_bytes_per_block % page_size == 0``.
        """
        world_size = self.config.parallel.world_size
        raw_row = self.cpu_page_size_per_worker * world_size
        return round_up(raw_row, SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT)

    @property
    def _compact_num_rows(self) -> int:
        """Number of rows in the compact shared region.

        ``floor(compact_cpu_bytes / aligned_row_stride)``.  Returns 0
        when the budget cannot fit one full row.
        """
        stride = self._compact_row_stride
        if stride <= 0 or self._compact_cpu_bytes < stride:
            return 0
        return self._compact_cpu_bytes // stride

    def _build_compact_shared_region(self) -> SharedOffloadRegion:
        rank = self.config.parallel.rank
        row_stride = self._compact_row_stride
        num_rows = self._compact_num_rows
        if num_rows == 0:
            raise RuntimeError("compact CPU budget cannot fit one shared row")
        return SharedOffloadRegion(
            engine_id=self.config.engine_id,
            num_blocks=num_rows,
            rank=rank,
            kv_bytes_per_block=row_stride,
            cpu_page_size=self.cpu_page_size_per_worker,
        )

    def create_worker(
        self,
        kv_caches: CanonicalKVCaches,
        mmap_region: SharedOffloadRegion | None = None,
    ) -> CPUOffloadingWorker:
        num_cpu_blocks = (
            self._compact_num_rows
            if self._compact_layout_requested
            else self.num_blocks
        )
        return CPUOffloadingWorker(
            kv_caches=kv_caches,
            blocks_per_chunk=self.blocks_per_chunk,
            num_cpu_blocks=num_cpu_blocks,
            mmap_region=mmap_region,
        )

    @override
    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        if not self._worker:
            if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
                raise Exception(
                    "CPU Offloading is currently only supported on CUDA-alike "
                    "and XPU GPUs"
                )
            mmap_region = None
            if self._compact_layout_requested:
                mmap_region = self._build_compact_shared_region()
                self._worker_shared_region = mmap_region
            self._worker = self.create_worker(kv_caches, mmap_region=mmap_region)

        assert self._worker is not None
        return self._worker


def _is_preferred_eviction_group(group: KVCacheGroupSpec) -> bool:
    """Return True if the group should be preferred for early eviction.

    A group is preferred if it is an EAGLE group or if all its layers use
    sliding window (SWA).
    """
    if group.is_eagle_group:
        return True
    gspec = group.kv_cache_spec
    if isinstance(gspec, SlidingWindowSpec):
        return True
    if isinstance(gspec, UniformTypeKVCacheSpecs):
        # All sub-specs must be SlidingWindowSpec.
        return all(
            isinstance(s, SlidingWindowSpec) for s in gspec.kv_cache_specs.values()
        )
    return False
