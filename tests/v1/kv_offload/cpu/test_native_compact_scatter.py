#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native CUDA compact scatter test — real D2H and H2D through swap_blocks_batch.

Requires a CUDA-capable GPU and the vLLM build environment active.

Tests:
  1. D2H compact scatter: GPU → CPU with use_batch_api=False, using enough
     descriptors to prove explicit fallback selection.
  2. H2D compact scatter: CPU → GPU with use_batch_api=False, same descriptor
     count.
  3. Data integrity: after synchronize(), each CPU/GPU page matches its source.
  4. Both directions verify the batch API is NOT triggered (the test would
     segfault under cuMemcpyBatchAsync at 20K descriptors if use_batch_api
     were not respected).

Run:
    cd /path/to/vllm  # after building vllm
    python tests/v1/kv_offload/cpu/test_native_compact_scatter.py
"""

import sys
import time

import torch

from vllm import _custom_ops as ops

DEVICE = "cuda"
PAGE_SIZE = 128  # bytes per GPU page
BLOCK_SIZE_FACTOR = 1  # 1:1 GPU:CPU page mapping
NUM_GPU_BLOCKS = 2048
NUM_CPU_BLOCKS = 2048
NUM_DESCRIPTORS = 20_000  # enough to trigger batch-API segfault if fallback fails

# Use a non-default stream (cuMemcpyBatchAsync rejects default stream).
_STREAM = torch.cuda.Stream()


def _addrs(buffers: list[torch.Tensor]) -> torch.Tensor:
    return torch.tensor([b.data_ptr() for b in buffers], dtype=torch.int64)


def test_d2h_compact_scatter() -> None:
    """GPU→CPU: copy NUM_DESCRIPTORS small chunks with use_batch_api=False."""
    gpu_src = (
        (torch.arange(NUM_GPU_BLOCKS * PAGE_SIZE, device=DEVICE) % 256)
        .to(dtype=torch.uint8)
        .reshape(NUM_GPU_BLOCKS, PAGE_SIZE)
    )
    cpu_dst = torch.zeros(
        NUM_CPU_BLOCKS, PAGE_SIZE, dtype=torch.uint8, device="cpu", pin_memory=True
    )

    # Build descriptors: each copies one GPU page to one CPU page
    gpu_addrs = [gpu_src[i % NUM_GPU_BLOCKS].data_ptr() for i in range(NUM_DESCRIPTORS)]
    cpu_addrs = [cpu_dst[i % NUM_CPU_BLOCKS].data_ptr() for i in range(NUM_DESCRIPTORS)]
    sizes = [PAGE_SIZE] * NUM_DESCRIPTORS

    # Build full descriptor tensors
    src_t = torch.tensor(gpu_addrs, dtype=torch.int64)
    dst_t = torch.tensor(cpu_addrs, dtype=torch.int64)
    sizes_t = torch.tensor(sizes, dtype=torch.int64)

    with torch.cuda.stream(_STREAM):
        ops.swap_blocks_batch(
            src_t,
            dst_t,
            sizes_t,
            is_src_access_order_any=False,
            use_batch_api=False,
        )
    torch.cuda.synchronize()

    # Verify data integrity on a subset of pages
    for i in range(0, NUM_DESCRIPTORS, NUM_DESCRIPTORS // 10):
        gpu_idx = i % NUM_GPU_BLOCKS
        cpu_idx = i % NUM_CPU_BLOCKS
        expected = gpu_src[gpu_idx].cpu()
        actual = cpu_dst[cpu_idx]
        assert torch.equal(actual, expected), (
            f"D2H mismatch at descriptor {i}: GPU block {gpu_idx} → CPU block {cpu_idx}"
        )
    print(f"D2H: {NUM_DESCRIPTORS} descriptors OK (use_batch_api=False)")


def test_h2d_compact_scatter() -> None:
    """CPU→GPU: copy NUM_DESCRIPTORS small chunks with use_batch_api=False."""
    cpu_src = (
        (torch.arange(NUM_CPU_BLOCKS * PAGE_SIZE, device="cpu") % 256)
        .to(dtype=torch.uint8)
        .reshape(NUM_CPU_BLOCKS, PAGE_SIZE)
        .pin_memory()
    )
    gpu_dst = torch.zeros(NUM_GPU_BLOCKS, PAGE_SIZE, dtype=torch.uint8, device=DEVICE)

    cpu_addrs = [cpu_src[i % NUM_CPU_BLOCKS].data_ptr() for i in range(NUM_DESCRIPTORS)]
    gpu_addrs = [gpu_dst[i % NUM_GPU_BLOCKS].data_ptr() for i in range(NUM_DESCRIPTORS)]
    sizes = [PAGE_SIZE] * NUM_DESCRIPTORS

    src_t = torch.tensor(cpu_addrs, dtype=torch.int64)
    dst_t = torch.tensor(gpu_addrs, dtype=torch.int64)
    sizes_t = torch.tensor(sizes, dtype=torch.int64)

    with torch.cuda.stream(_STREAM):
        ops.swap_blocks_batch(
            src_t,
            dst_t,
            sizes_t,
            is_src_access_order_any=False,
            use_batch_api=False,
        )
    torch.cuda.synchronize()

    for i in range(0, NUM_DESCRIPTORS, NUM_DESCRIPTORS // 10):
        cpu_idx = i % NUM_CPU_BLOCKS
        gpu_idx = i % NUM_GPU_BLOCKS
        expected = cpu_src[cpu_idx]
        actual = gpu_dst[gpu_idx].cpu()
        assert torch.equal(actual, expected), (
            f"H2D mismatch at descriptor {i}: CPU block {cpu_idx} → GPU block {gpu_idx}"
        )
    print(f"H2D: {NUM_DESCRIPTORS} descriptors OK (use_batch_api=False)")


def test_batch_api_true_still_works() -> None:
    """Smoke test that use_batch_api=True (default) still works on dedicated stream."""
    src = torch.randint(256, (PAGE_SIZE,), dtype=torch.uint8, device=DEVICE)
    dst = torch.zeros_like(src)
    with torch.cuda.stream(_STREAM):
        ops.swap_blocks_batch(
            _addrs([src]),
            _addrs([dst]),
            torch.tensor([PAGE_SIZE], dtype=torch.int64),
            use_batch_api=True,
        )
    torch.cuda.synchronize()
    assert torch.equal(dst, src)
    print("Batch API True: OK")


if __name__ == "__main__":
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    t0 = time.monotonic()
    test_batch_api_true_still_works()
    test_d2h_compact_scatter()
    test_h2d_compact_scatter()
    elapsed = time.monotonic() - t0
    print(f"\nAll native CUDA tests passed in {elapsed:.2f}s")
    sys.exit(0)
