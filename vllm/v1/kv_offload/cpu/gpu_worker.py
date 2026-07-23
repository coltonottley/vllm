# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, triton
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.kv_offload.base import (
    BlockIDsLoadStoreSpec,
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.cpu.common import (
    CompactCPULoadStoreSpec,
    CompactGroupGeometry,
)
from vllm.v1.kv_offload.cpu.compact_transfer import (
    plan_compact_transfer,
)
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.cpu.swap_blocks_triton import (
    THRESHOLD_BYTES,
    swap_blocks_batch,
)

logger = init_logger(__name__)


def _select_swap_blocks_fn(
    kv_cache_groups_data_refs: list[list[CanonicalKVCacheRef]],
    gpu_to_cpu: bool,
):
    """Resolve the swap_blocks function for a handler at init time."""
    # GPU->CPU is bandwidth-bound; the dedicated copy engine beats Triton.
    if gpu_to_cpu:
        return ops.swap_blocks_batch
    # Fall back to the C++ DMA path on platforms where Triton isn't usable
    # (e.g. ROCm builds without Triton) or where GPU kernels cannot directly
    # dereference CPU pointers (XPU lacks CUDA's unified virtual address space,
    # so the Triton kernel's tl.load(cpu_ptr) is invalid on XPU).
    if not HAS_TRITON or current_platform.is_xpu():
        return ops.swap_blocks_batch
    page_sizes = [r.page_size_bytes for g in kv_cache_groups_data_refs for r in g]
    # Triton wins only on small, 8-byte-aligned payloads.
    if (
        not page_sizes
        or max(page_sizes) >= THRESHOLD_BYTES
        or any(s % 8 for s in page_sizes)
    ):
        return ops.swap_blocks_batch
    chunk = min(triton.next_power_of_2(max(page_sizes)), 8192)
    return functools.partial(swap_blocks_batch, bytes_per_chunk=chunk)


@dataclass
class Transfer:
    job_id: int
    stream: torch.cuda.Stream
    start_event: torch.Event
    end_event: torch.Event
    num_bytes: int
    batch_src: torch.Tensor
    batch_dst: torch.Tensor
    batch_sizes: torch.Tensor


def compute_sub_block_ptrs(
    block_ids: np.ndarray,
    blocks_per_chunk: int,
    output: np.ndarray,
    tensor: torch.Tensor,
    skip_count: int = 0,
):
    """
    Compute byte pointers for sub-blocks of the given block IDs.

    Each block in block_ids contains blocks_per_chunk sub-blocks.
    The pointer for sub-block j of block b is:
        base_ptr + b * row_stride + j * block_page_size

    where block_page_size = tensor.shape[1] // blocks_per_chunk (gpu page size).

    This handles tensors where row_stride != blocks_per_chunk * block_page_size
    (e.g. non-contiguous CPU tensors).

    Args:
        block_ids: array of block IDs at the tensor's native granularity.
        blocks_per_chunk: number of sub-blocks per block.
        output: pre-allocated pointer array to write pointers into.
        tensor: the source or destination tensor.
        skip_count: sub-blocks to skip in the first block.
    """
    assert skip_count < blocks_per_chunk

    num_sub_blocks = len(output)
    base_ptr = tensor.data_ptr()
    row_stride = tensor.stride(0)

    if blocks_per_chunk == 1:
        # Fast path: 1:1 mapping, no sub-block expansion needed.
        output[:] = base_ptr + block_ids.astype(np.uint64)[:num_sub_blocks] * row_stride
        return

    # Vectorized expansion for blocks_per_chunk > 1.
    assert tensor.shape[1] % blocks_per_chunk == 0
    block_page_size = tensor.shape[1] // blocks_per_chunk
    sub_offsets = np.arange(blocks_per_chunk, dtype=np.uint64) * block_page_size
    # (num_blocks, 1) + (1, blocks_per_chunk) -> (num_blocks, blocks_per_chunk)
    all_ptrs = (
        base_ptr + block_ids.astype(np.uint64)[:, np.newaxis] * row_stride
    ) + sub_offsets[np.newaxis, :]
    # Flatten and apply skip_count / truncation
    flat = all_ptrs.ravel()
    output[:] = flat[skip_count : skip_count + num_sub_blocks]


def pin_mmap_region(region: SharedOffloadRegion) -> None:
    """Register the entire mmap as CUDA pinned memory via cudaHostRegister."""
    if not current_platform.is_cuda_alike():
        logger.info(
            "Skipping mmap host registration on %s; cudaHostRegister is only "
            "available on CUDA/ROCm.",
            current_platform.device_name,
        )
        return

    rank = region.rank

    base_ptr = region.base_ptr
    result = torch.cuda.cudart().cudaHostRegister(base_ptr, region.total_size_bytes, 0)
    if result.value != 0:
        logger.warning(
            "cudaHostRegister failed for rank=%d (code=%d) — "
            "transfers will still work but may be slower (unpinned DMA)",
            rank,
            result,
        )
    else:
        logger.debug(
            "cudaHostRegister rank=%d %.2f GB",
            rank,
            region.total_size_bytes / 1e9,
        )
        region.is_pinned = True


def _new_descriptor_buffers(
    num_copy_ops: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pin = PIN_MEMORY
    # CUDA cache_kernels.cu requires int64; XPU DMA engine requires uint64.
    ptr_dtype = torch.uint64 if current_platform.is_xpu() else torch.int64
    return (
        torch.empty(num_copy_ops, dtype=ptr_dtype, pin_memory=pin),
        torch.empty(num_copy_ops, dtype=ptr_dtype, pin_memory=pin),
        torch.empty(num_copy_ops, dtype=ptr_dtype, pin_memory=pin),
    )


class SingleDirectionOffloadingHandler:
    """
    Handles transfers for a single direction, either CPU->GPU or GPU->CPU.
    Transfers are guaranteed to be executed in order of their submission.
    Each transfer uses a unique CUDA stream, and its stream will start
    executing only after the streams of previous transfers have finished.

    Supports both block-swap transfers (via ``transfer_async`` with
    ``BlockIDsLoadStoreSpec``) and compact descriptor transfers (via
    ``transfer_async`` with ``CompactCPULoadStoreSpec``).  Compact transfers
    use ``ops.swap_blocks_batch`` directly (no Triton), bypassing the
    batch API resolution that selects between Triton and DMA for ordinary
    block swaps.  The compact path always passes an explicit ``swap_fn``
    to ``_submit_descriptors`` so the native op is used regardless of
    the handler's resolved ``_swap_blocks_batch``.
    """

    def __init__(
        self,
        gpu_tensors: list[torch.Tensor],
        cpu_tensors: list[torch.Tensor],
        blocks_per_chunk: int,
        kv_cache_groups_data_refs: list[list[CanonicalKVCacheRef]],
        gpu_to_cpu: bool,
        mmap_region: SharedOffloadRegion | None = None,
        compact_geometry: tuple[CompactGroupGeometry | None, ...] | None = None,
    ):
        """
        Initialize a SingleDirectionOffloadingHandler.

        Args:
            gpu_tensors: list of GPU KV cache tensors.
                Each of shape (num_gpu_blocks, gpu_page_size_bytes) with dtype int8.
            cpu_tensors: list of CPU KV cache tensors.
                Each of shape (num_cpu_blocks, cpu_page_size_bytes) with dtype int8.
                Order should match gpu_tensors.
            kv_cache_groups_data_refs: list of CanonicalKVCacheRef per group.
            gpu_to_cpu: if True, transfer from GPU to CPU; otherwise CPU to GPU.
            mmap_region: optional shared mmap region for CPU storage.
            compact_geometry: per-group compact geometry for compact descriptor
                transfers.  ``None`` (default) means compact transfers cannot
                be used on this handler.
        """
        assert len(gpu_tensors) == len(cpu_tensors)
        assert len(gpu_tensors) > 0
        assert blocks_per_chunk > 0

        # assert input tensors are as expected
        for gpu_tensor, cpu_tensor in zip(gpu_tensors, cpu_tensors):
            assert gpu_tensor.dtype == torch.int8
            assert gpu_tensor.ndim == 2
            assert gpu_tensor.is_cuda or gpu_tensor.is_xpu
            assert cpu_tensor.dtype == torch.int8
            assert cpu_tensor.ndim == 2
            assert cpu_tensor.device.type == "cpu"
            _, gpu_page_size = gpu_tensor.shape
            _, cpu_page_size = cpu_tensor.shape
            assert cpu_page_size == gpu_page_size * blocks_per_chunk

        self.src_tensors: list[torch.Tensor] = (
            gpu_tensors if gpu_to_cpu else cpu_tensors
        )
        self.dst_tensors: list[torch.Tensor] = (
            cpu_tensors if gpu_to_cpu else gpu_tensors
        )
        self.gpu_to_cpu: bool = gpu_to_cpu
        self.kv_cache_groups_data_refs = kv_cache_groups_data_refs
        self._swap_blocks_batch = _select_swap_blocks_fn(
            kv_cache_groups_data_refs, gpu_to_cpu
        )
        self._blocks_per_chunk: int = blocks_per_chunk

        # GPU blocks may be smaller
        # cpu_page_size = gpu_page_size * blocks_per_chunk.
        self.src_blocks_per_chunk = 1 if self.gpu_to_cpu else blocks_per_chunk
        self.dst_blocks_per_chunk = blocks_per_chunk if self.gpu_to_cpu else 1

        # mmap_region to clean up on shutdown (gpu_to_cpu handler owns it)
        self._mmap_region = mmap_region
        # job_id -> event
        self._transfer_events: dict[int, torch.Event] = {}
        # queue of transfers (job_id, stream, event)
        self._transfers: deque[Transfer] = deque()
        # list of CUDA streams available for re-use
        self._stream_pool: list[torch.cuda.Stream] = []
        # list of CUDA events available for re-use
        self._event_pool: list[torch.Event] = []
        # list of pinned descriptor buffer sets available for re-use
        self._buffer_pool: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        # Compact geometry for descriptor-based transfers.  Set once
        # via configure_compact_geometry() or passed directly to __init__.
        self._compact_geometry: tuple[CompactGroupGeometry | None, ...] | None = (
            compact_geometry
        )

    # ------------------------------------------------------------------
    # Compact geometry configuration (one-shot, like worker-level API)
    # ------------------------------------------------------------------

    def configure_compact_geometry(
        self, groups: tuple[CompactGroupGeometry | None, ...]
    ) -> None:
        """Accept compact geometry exactly once while unset.

        Stores the geometry as an immutable tuple.  Rejects a second
        call even if the argument is equal.
        """
        if self._compact_geometry is not None:
            raise RuntimeError(
                "compact geometry is already configured and may not be "
                "replaced; one-shot configuration expected."
            )
        self._compact_geometry = groups

    # ------------------------------------------------------------------
    # Shared submit lifecycle for descriptor-based transfers
    # ------------------------------------------------------------------

    def _submit_descriptors(
        self,
        job_id: int,
        src_ptrs: torch.Tensor,
        dst_ptrs: torch.Tensor,
        sizes: torch.Tensor,
        num_bytes: int,
        is_src_access_order_any: bool = False,
        use_batch_api: bool = True,
        swap_fn: Callable | None = None,
    ) -> bool:
        """Submit pre-built descriptor tensors through the native swap
        with full stream/event/pending gate lifecycle.

        Owns the pooled stream/event/buffer lifecycle shared with the
        ordinary block-swap path.  Returns ``True`` on success.

        When ``num_bytes == 0`` (zero-descriptor plan, e.g. non-writer
        store), the native op is skipped but the completion gate
        (``_transfer_events`` / ``_transfers`` queue) is still
        registered so ``get_finished`` works correctly.

        Args:
            swap_fn: optional override for the swap function.  When
                ``None`` (default), uses ``self._swap_blocks_batch``
                which was resolved at init time.  The compact path
                passes ``ops.swap_blocks_batch`` directly to ensure
                the canonical native op is always used.
        """
        num_descriptors = src_ptrs.numel()

        stream = (
            self._stream_pool.pop() if self._stream_pool else current_platform.Stream()
        )
        start_event = (
            self._event_pool.pop()
            if self._event_pool
            else torch.Event(enable_timing=True)
        )
        end_event = (
            self._event_pool.pop()
            if self._event_pool
            else torch.Event(enable_timing=True)
        )

        if self.gpu_to_cpu:
            # wait for model computation to finish before offloading
            stream.wait_stream(current_platform.current_stream())
        if self._transfers:
            last_transfer: Transfer = self._transfers[-1]
            last_event = last_transfer.end_event
            # assure job will start only after the previous one completes
            stream.wait_event(last_event)
        with current_platform.stream(stream):
            start_event.record(stream)
            if num_descriptors > 0:
                fn = swap_fn if swap_fn is not None else self._swap_blocks_batch
                fn(
                    src_ptrs,
                    dst_ptrs,
                    sizes,
                    is_src_access_order_any=is_src_access_order_any,
                    use_batch_api=use_batch_api,
                )
            end_event.record(stream)

        self._transfer_events[job_id] = end_event
        self._transfers.append(
            Transfer(
                job_id=job_id,
                stream=stream,
                start_event=start_event,
                end_event=end_event,
                num_bytes=num_bytes,
                batch_src=src_ptrs,
                batch_dst=dst_ptrs,
                batch_sizes=sizes,
            )
        )
        return True

    # ------------------------------------------------------------------
    # Compact transfer routing helpers
    # ------------------------------------------------------------------

    def _derive_compact_gpu_base_ptr(self) -> int:
        """Return the base GPU pointer for compact descriptor planning.

        The compact GPU tensor is the first GPU source (store) or
        destination (load) tensor.  For packed layouts, all layers share
        one packed tensor, so ``self.src_tensors[0]`` (store) or
        ``self.dst_tensors[0]`` (load) is the authoritative packed view.
        """
        tensor = self.src_tensors if self.gpu_to_cpu else self.dst_tensors
        return int(tensor[0].data_ptr())

    def _derive_compact_cpu_base_ptr_and_region(
        self,
    ) -> tuple[int, int]:
        """Return (cpu_base_ptr, cpu_region_size) for compact planning.

        Uses the shared mmap region's public ``base_ptr`` / ``total_size_bytes``.
        Raises ``RuntimeError`` when no ``SharedOffloadRegion`` is available,
        because compact transfers require the shared mmap backing.
        """
        if self._mmap_region is None:
            raise RuntimeError(
                "compact transfers require a SharedOffloadRegion; "
                "no mmap region is configured on this handler"
            )
        base = self._mmap_region.base_ptr
        size = self._mmap_region.total_size_bytes
        return base, size

    def _transfer_async_compact(
        self,
        job_id: int,
        src_spec: LoadStoreSpec,
        dst_spec: LoadStoreSpec,
    ) -> bool:
        """Submit a compact descriptor transfer through the shared
        ``_submit_descriptors`` lifecycle.

        Preserves positional ``_compact_geometry`` (does not filter
        ``None`` entries).  Validates that the geometry is available for
        the requested group *before* any transfer work; fails loud if not.
        """
        if self.gpu_to_cpu:
            gpu_spec = src_spec
            compact_spec = dst_spec
            direction = "store"
        else:
            compact_spec = src_spec
            gpu_spec = dst_spec
            direction = "load"

        assert isinstance(gpu_spec, GPULoadStoreSpec), (
            f"compact-mode GPU spec must be GPULoadStoreSpec, got {type(gpu_spec)}"
        )
        assert isinstance(compact_spec, CompactCPULoadStoreSpec), (
            f"compact-mode compact spec must be CompactCPULoadStoreSpec, "
            f"got {type(compact_spec)}"
        )
        assert self._compact_geometry is not None, (
            "compact geometry must be configured before compact transfers"
        )

        geom = self._compact_geometry

        # Preserve positional geometry: validate geometry is available for
        # every group referenced by any compact address *before* planning.
        for addr in compact_spec.compact_addresses:
            g = addr.group_idx
            if g < 0 or g >= len(geom):
                raise RuntimeError(
                    f"compact address group_idx={g} is out of range for "
                    f"{len(geom)} geometry entries"
                )
            if geom[g] is None:
                raise RuntimeError(
                    f"compact geometry is None for group_idx={g}; "
                    "compact transfers require non-null geometry for "
                    "every referenced group"
                )

        # Extract geometry data, preserving positional index.
        per_group_mappings: list[tuple] = []
        per_group_canonical_offsets: list[tuple[int, ...]] = []
        per_group_gpu_offsets: list[tuple[int, ...]] = []
        gpu_row_stride: int | None = None
        for gidx, g in enumerate(geom):
            if g is None:
                per_group_mappings.append(())
                per_group_canonical_offsets.append(())
                per_group_gpu_offsets.append(())
            else:
                per_group_mappings.append(tuple(ly.mapping for ly in g.layers))
                per_group_canonical_offsets.append(
                    tuple(ly.canonical_offset for ly in g.layers)
                )
                per_group_gpu_offsets.append(
                    tuple(ly.gpu_offset_bytes for ly in g.layers)
                )
                if gpu_row_stride is None:
                    gpu_row_stride = g.gpu_row_stride
                elif g.gpu_row_stride != gpu_row_stride:
                    raise RuntimeError(
                        f"group {gidx} gpu_row_stride={g.gpu_row_stride} "
                        f"does not match previously observed "
                        f"gpu_row_stride={gpu_row_stride}; "
                        "all non-None compact groups must share the same stride"
                    )

        assert gpu_row_stride is not None, (
            "at least one non-None group expected when compact transfer is active"
        )

        gpu_base_ptr = self._derive_compact_gpu_base_ptr()
        cpu_base_ptr, cpu_region_size = self._derive_compact_cpu_base_ptr_and_region()

        gpu_block_ids = gpu_spec.block_ids
        group_sizes = gpu_spec.group_sizes
        block_indices = gpu_spec.block_indices
        compact_addresses = compact_spec.compact_addresses

        plan = plan_compact_transfer(
            gpu_base_ptr=gpu_base_ptr,
            gpu_row_stride=gpu_row_stride,
            cpu_base_ptr=cpu_base_ptr,
            cpu_region_size=cpu_region_size,
            gpu_block_ids=gpu_block_ids,
            group_sizes=group_sizes,
            block_indices=block_indices,
            compact_addresses=compact_addresses,
            per_group_mappings=per_group_mappings,
            per_group_canonical_offsets=per_group_canonical_offsets,
            per_group_gpu_offsets=per_group_gpu_offsets,
            blocks_per_chunk=self._blocks_per_chunk,
            direction=direction,
        )

        num_bytes = plan.num_bytes
        num_descriptors = plan.num_descriptors

        # Convert numpy arrays to writable torch tensors for the native op.
        ptr_dtype = torch.uint64 if current_platform.is_xpu() else torch.int64
        gpu_ptr_t = torch.from_numpy(plan.gpu_ptrs.copy()).to(ptr_dtype)
        cpu_ptr_t = torch.from_numpy(plan.cpu_ptrs.copy()).to(ptr_dtype)
        sz_t = torch.from_numpy(plan.sizes.copy()).to(torch.int64)

        is_src_access_order_any = not self.gpu_to_cpu

        # For zero-descriptor plans (non-writer store): skip the native
        # op but still register the completion gate.
        if num_descriptors == 0:
            return self._submit_descriptors(
                job_id=job_id,
                src_ptrs=gpu_ptr_t,
                dst_ptrs=cpu_ptr_t,
                sizes=sz_t,
                num_bytes=num_bytes,
                is_src_access_order_any=is_src_access_order_any,
                use_batch_api=False,
                swap_fn=ops.swap_blocks_batch,
            )

        # Reuse a pooled buffer set for the descriptor arrays, growing
        # if this transfer needs more room.
        batch_src, batch_dst, batch_sizes = (
            self._buffer_pool.pop()
            if self._buffer_pool
            else _new_descriptor_buffers(num_descriptors)
        )
        if batch_src.numel() < num_descriptors:
            batch_src, batch_dst, batch_sizes = _new_descriptor_buffers(num_descriptors)

        # Explicit direction-based assignment: the native op consumes
        # positional (src_ptrs, dst_ptrs), not semantic (gpu, cpu).
        # Store copies GPU→CPU; load copies CPU→GPU.
        if self.gpu_to_cpu:
            batch_src[:num_descriptors].copy_(gpu_ptr_t)
            batch_dst[:num_descriptors].copy_(cpu_ptr_t)
        else:
            batch_src[:num_descriptors].copy_(cpu_ptr_t)
            batch_dst[:num_descriptors].copy_(gpu_ptr_t)
        batch_sizes[:num_descriptors].copy_(sz_t)

        return self._submit_descriptors(
            job_id=job_id,
            src_ptrs=batch_src[:num_descriptors],
            dst_ptrs=batch_dst[:num_descriptors],
            sizes=batch_sizes[:num_descriptors],
            num_bytes=num_bytes,
            is_src_access_order_any=is_src_access_order_any,
            use_batch_api=False,
            swap_fn=ops.swap_blocks_batch,
        )

    # ------------------------------------------------------------------
    # Main transfer_async — routes compact vs. legacy
    # ------------------------------------------------------------------

    def transfer_async(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        # Detect compact CPU spec (dst for store, src for load).
        compact_spec = dst_spec if self.gpu_to_cpu else src_spec
        if isinstance(compact_spec, CompactCPULoadStoreSpec):
            return self._transfer_async_compact(job_id, src_spec, dst_spec)

        assert isinstance(src_spec, BlockIDsLoadStoreSpec), (
            f"expected BlockIDsLoadStoreSpec, got {type(src_spec)}"
        )
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec), (
            f"expected BlockIDsLoadStoreSpec, got {type(dst_spec)}"
        )

        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        assert src_blocks.ndim == 1
        assert dst_blocks.ndim == 1

        num_src_blocks = len(src_blocks)
        num_dst_blocks = len(dst_blocks)

        # There are 2 types of transfers:
        # 1. GPU -> CPU
        # 2. CPU -> GPU
        #
        # transfers are also to CPU blocks, EXCEPT MAYBE for the first and last block.
        # i.e. the first and last CPU blocks in src_blocks can match against
        # a smaller (byte-wise) set of GPU blocks in dst_blocks.
        # In such cases, we may need to skip some gpu-sized sub-blocks,
        # and start reading/writing from the middle of the first CPU block.
        # If we have multiple KV cache groups (when using HMA with hybrid models),
        # we may have a partial first/last CPU block per each group.
        # The group_sizes parameter encodes the size of each group of blocks
        # in the GPU dst_blocks.
        # If group_sizes is None, we assume all blocks belong to a single group.
        # The logical_offset parameter maps each group of blocks to its logical
        # offset inside the request, counting in GPU blocks.
        # This allows us to find the correct starting position
        # in the matching first CPU block.

        # extract group_sizes from the GPU spec
        gpu_spec = src_spec if self.gpu_to_cpu else dst_spec
        assert isinstance(gpu_spec, GPULoadStoreSpec)
        group_sizes = gpu_spec.group_sizes
        assert len(group_sizes) == len(self.kv_cache_groups_data_refs)

        # extract block indices from the GPU spec
        block_indices = gpu_spec.block_indices
        assert len(block_indices) == len(self.kv_cache_groups_data_refs)

        num_copy_ops = 0
        for group_size, group_data_refs in zip(
            group_sizes, self.kv_cache_groups_data_refs
        ):
            num_copy_ops += group_size * len(group_data_refs)

        # reuse a pooled buffer set, growing it if this transfer needs more room
        batch_src, batch_dst, batch_sizes = (
            self._buffer_pool.pop()
            if self._buffer_pool
            else _new_descriptor_buffers(num_copy_ops)
        )
        if batch_src.numel() < num_copy_ops:
            batch_src, batch_dst, batch_sizes = _new_descriptor_buffers(num_copy_ops)

        src = batch_src[:num_copy_ops]
        dst = batch_dst[:num_copy_ops]
        sizes = batch_sizes[:num_copy_ops]
        all_src = src.numpy()
        all_dst = dst.numpy()
        all_sizes = sizes.numpy()

        src_offset = 0
        dst_offset = 0
        op_idx = 0
        # count total number of bytes copied
        num_transfer_bytes = 0
        for group_size, block_idx, group_data_refs in zip(
            group_sizes, block_indices, self.kv_cache_groups_data_refs
        ):
            if group_size == 0:
                continue

            src_logical_blocks_to_skip = block_idx % self.src_blocks_per_chunk
            dst_logical_blocks_to_skip = block_idx % self.dst_blocks_per_chunk
            src_logical_blocks_count = group_size + src_logical_blocks_to_skip
            dst_logical_blocks_count = group_size + dst_logical_blocks_to_skip

            dst_blocks_count = cdiv(dst_logical_blocks_count, self.dst_blocks_per_chunk)
            dst_end_offset = dst_offset + dst_blocks_count
            assert dst_end_offset <= num_dst_blocks

            src_blocks_count = cdiv(src_logical_blocks_count, self.src_blocks_per_chunk)
            src_end_offset = src_offset + src_blocks_count
            assert src_end_offset <= num_src_blocks

            group_src = src_blocks[src_offset:src_end_offset]
            group_dst = dst_blocks[dst_offset:dst_end_offset]

            for data_ref in group_data_refs:
                t_idx = data_ref.tensor_idx
                end_idx = op_idx + group_size

                compute_sub_block_ptrs(
                    group_src,
                    self.src_blocks_per_chunk,
                    all_src[op_idx:end_idx],
                    self.src_tensors[t_idx],
                    skip_count=src_logical_blocks_to_skip,
                )
                compute_sub_block_ptrs(
                    group_dst,
                    self.dst_blocks_per_chunk,
                    all_dst[op_idx:end_idx],
                    self.dst_tensors[t_idx],
                    skip_count=dst_logical_blocks_to_skip,
                )

                all_sizes[op_idx:end_idx] = data_ref.page_size_bytes
                num_transfer_bytes += group_size * data_ref.page_size_bytes
                op_idx = end_idx

            src_offset = src_end_offset
            dst_offset = dst_end_offset

        assert src_offset == num_src_blocks
        assert dst_offset == num_dst_blocks
        assert op_idx == num_copy_ops

        is_src_access_order_any = not self.gpu_to_cpu
        return self._submit_descriptors(
            job_id=job_id,
            src_ptrs=src,
            dst_ptrs=dst,
            sizes=sizes,
            num_bytes=num_transfer_bytes,
            is_src_access_order_any=is_src_access_order_any,
            use_batch_api=True,
        )

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        while self._transfers and self._transfers[0].end_event.query():
            transfer = self._transfers.popleft()
            transfer_time = (
                transfer.start_event.elapsed_time(transfer.end_event) * 1e-3
            )  # elapsed_time is in milliseconds
            result = TransferResult(
                job_id=transfer.job_id,
                success=True,
                transfer_size=transfer.num_bytes,
                transfer_time=transfer_time,
            )

            results.append(result)
            self._stream_pool.append(transfer.stream)
            self._event_pool.append(transfer.end_event)
            self._event_pool.append(transfer.start_event)
            self._buffer_pool.append(
                (transfer.batch_src, transfer.batch_dst, transfer.batch_sizes)
            )
            del self._transfer_events[transfer.job_id]
        return results

    def wait(self, job_ids: set[int]):
        for job_id in job_ids:
            event = self._transfer_events.get(job_id)
            if event is not None:
                event.synchronize()

    def shutdown(self) -> None:
        while self._transfers:
            transfer = self._transfers.popleft()
            transfer.end_event.synchronize()
        self._transfer_events.clear()
        self._stream_pool.clear()
        self._event_pool.clear()
        self._buffer_pool.clear()
        self.src_tensors.clear()
        self.dst_tensors.clear()
        self._mmap_region = None


class CPUOffloadingWorker(OffloadingWorker):
    """OffloadingWorker for CPU offloading.

    Composes two SingleDirectionOffloadingHandler instances (one for each
    direction) and exposes them through the explicit submit_store /
    submit_load API.
    """

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        blocks_per_chunk: int,
        num_cpu_blocks: int,
        mmap_region: SharedOffloadRegion | None = None,
    ):
        pin_memory = PIN_MEMORY
        logger.info("Allocating %d CPU tensors...", len(kv_caches.tensors))
        if mmap_region is not None and pin_memory:
            pin_mmap_region(mmap_region)

        gpu_tensors: list[torch.Tensor] = []
        cpu_tensors: list[torch.Tensor] = []
        for kv_cache_tensor in kv_caches.tensors:
            gpu_page_size_bytes = kv_cache_tensor.page_size_bytes
            gpu_tensor = kv_cache_tensor.tensor.view(torch.int8).view(
                (-1, gpu_page_size_bytes)
            )
            cpu_page_size_bytes = gpu_page_size_bytes * blocks_per_chunk

            if mmap_region is not None:
                cpu_tensor = mmap_region.create_next_view(cpu_page_size_bytes)
            else:
                t0 = time.monotonic()
                cpu_tensor = torch.zeros(
                    (num_cpu_blocks, cpu_page_size_bytes),
                    dtype=torch.int8,
                    device="cpu",
                    pin_memory=pin_memory,
                )
                logger.debug(
                    "torch.zeros pinned tensor %d×%d (%.2f GB): %.3f s",
                    num_cpu_blocks,
                    cpu_page_size_bytes,
                    num_cpu_blocks * cpu_page_size_bytes / 1e9,
                    time.monotonic() - t0,
                )

            gpu_tensors.append(gpu_tensor)
            cpu_tensors.append(cpu_tensor)

        self._store_handler = SingleDirectionOffloadingHandler(
            gpu_tensors=gpu_tensors,
            cpu_tensors=cpu_tensors,
            blocks_per_chunk=blocks_per_chunk,
            kv_cache_groups_data_refs=kv_caches.group_data_refs,
            gpu_to_cpu=True,
            mmap_region=mmap_region,
        )

        self._load_handler = SingleDirectionOffloadingHandler(
            gpu_tensors=gpu_tensors,
            cpu_tensors=cpu_tensors,
            blocks_per_chunk=blocks_per_chunk,
            kv_cache_groups_data_refs=kv_caches.group_data_refs,
            gpu_to_cpu=False,
            mmap_region=mmap_region,
        )

        # Retain the base region reference for future compact descriptor planning.
        # The compact planner needs the full base, not handler-private traversal.
        self._mmap_region: SharedOffloadRegion | None = mmap_region

        # Compact geometry: set once via configure_compact_geometry().
        self._compact_geometry: tuple[CompactGroupGeometry | None, ...] | None = None

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        """Async GPU -> CPU."""
        return self._store_handler.transfer_async(job_id, src_spec, dst_spec)

    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        """Async CPU -> GPU."""
        return self._load_handler.transfer_async(job_id, src_spec, dst_spec)

    def get_finished(self) -> list[TransferResult]:
        return self._store_handler.get_finished() + self._load_handler.get_finished()

    def wait(self, job_ids: set[int]) -> None:
        self._store_handler.wait(job_ids)
        self._load_handler.wait(job_ids)

    def shutdown(self) -> None:
        self._store_handler.shutdown()
        self._load_handler.shutdown()
        if self._mmap_region is not None:
            self._mmap_region.cleanup()
            self._mmap_region = None
        self._compact_geometry = None

    def configure_compact_geometry(
        self, groups: tuple[CompactGroupGeometry | None, ...]
    ) -> None:
        """Accept compact geometry exactly once while unset.

        One-shot propagates through each handler's public ``configure``
        method.  Rejects a second call even if the argument is equal.
        """
        if self._compact_geometry is not None:
            raise RuntimeError(
                "compact geometry is already configured and may not be "
                "replaced; one-shot configuration expected."
            )
        # Propagate to both handlers before storing on self so that any
        # handler-level rejection happens before the worker-level store.
        self._store_handler.configure_compact_geometry(groups)
        self._load_handler.configure_compact_geometry(groups)
        self._compact_geometry = groups
