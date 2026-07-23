# SPDX-License-Identifier: Apache-2.0
"""Symbolic bounded final-tail scheduler regressions."""

from vllm.v1.kv_offload.base import make_offload_key

# ---------------------------------------------------------------------------
# 2600-token symbolic bounded-tail counterexample
# ---------------------------------------------------------------------------


def test_2600_token_bounded_tail_symbolic():
    """Symbolic regression: bounded tail gives correct per-group store counts.

    This test validates the `latest_prompt_group_boundary` and
    `latest_prompt_store_range` helper functions directly, independent
    of the full scheduler fixture.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        latest_prompt_store_range,
    )

    # DSV4 layout:
    # Group 0: full-attention, block_size=256 -> tokens_per_chunk=256
    # Groups 1-4: sliding_window (SWA)
    # Prompt: 2600 tokens total

    # Group 0 (full-attention): tokens_per_chunk=256, sliding_window=None
    fa_chunk = 256
    fa_chunks_total = 2600 // fa_chunk  # 10 full chunks + 40 remainder
    assert fa_chunks_total == 10

    # Group 1 (SWA): tokens_per_chunk=64, sliding_window=128 -> 2 chunks
    swa1_chunk = 64
    swa1_chunks_total = 2600 // swa1_chunk  # 40 chunks

    # With bounded tail: sliding_window_chunks=2, prompt_final=True
    start, count = latest_prompt_store_range(
        next_stored_block_idx=0,
        storable_block_count=swa1_chunks_total,
        sliding_window_blocks=2,
        is_eagle_group=False,
        enabled=True,
        prompt_final=True,
    )
    # Expected: tail_start = max(0, 40 - 2) = 38, store_range = [38, 40)
    assert start == 38, f"Expected 38, got {start}"
    assert count == 40, f"Expected 40, got {count}"
    tail_chunks = count - start
    assert tail_chunks == 2, f"Expected 2 tail chunks, got {tail_chunks}"

    # Group 0 (full-attention): no bounded tail (sliding_window_blocks is None)
    fa_start, fa_count = latest_prompt_store_range(
        next_stored_block_idx=0,
        storable_block_count=fa_chunks_total,
        sliding_window_blocks=None,
        is_eagle_group=False,
        enabled=True,
        prompt_final=True,
    )
    # Full attention stores all chunks (enabled but sliding_window_blocks=None)
    assert fa_start == 0
    assert fa_count == fa_chunks_total

    # Group 2 (EAGLE): sliding_window=128, 2 chunks + 1 lookahead
    eagle_start, eagle_count = latest_prompt_store_range(
        next_stored_block_idx=0,
        storable_block_count=swa1_chunks_total,
        sliding_window_blocks=2,
        is_eagle_group=True,
        enabled=True,
        prompt_final=True,
    )
    # Expected: reachable_tail = 2 + 1 = 3, tail_start = 40 - 3 = 37
    assert eagle_start == 37, f"Expected 37, got {eagle_start}"
    assert eagle_count == 40, f"Expected 40, got {eagle_count}"
    assert eagle_count - eagle_start == 3, "EAGLE needs 3 chunks"


def test_latest_prompt_group_boundary_alignment():
    """Group boundary alignment for EAGLE groups."""
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        latest_prompt_group_boundary,
    )

    # Full-attention alignment tokens = 256 (tokens_per_chunk of FA group)
    result = latest_prompt_group_boundary(
        raw_prompt_boundary=300,
        full_attention_alignment_tokens=256,
        tokens_per_chunk=64,
        is_eagle_group=False,
    )
    # Aligned = round_down(300, 256) = 256
    assert result == 256, f"Expected 256, got {result}"

    # EAGLE group: one-chunk lookahead
    result_eagle = latest_prompt_group_boundary(
        raw_prompt_boundary=300,
        full_attention_alignment_tokens=256,
        tokens_per_chunk=64,
        is_eagle_group=True,
    )
    # aligned = min(300, 256 + 64) = min(300, 320) = 300
    assert result_eagle == 300, f"Expected 300, got {result_eagle}"


# ---------------------------------------------------------------------------
# blocks_per_chunk > 1 replay coherence
# ---------------------------------------------------------------------------


def test_blocks_per_chunk_replay_coherence():
    """With blocks_per_chunk=3, replay unit still covers all group keys."""
    from vllm.v1.kv_offload.base import (
        LookupResult,
        ReqContext,
    )
    from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

    def bpc_key(group, chunk_idx, block_idx):
        return make_offload_key(f"g{group}c{chunk_idx}b{block_idx}".encode(), group)

    # Budget exactly enough for one full replay unit (9 keys * 4096 = 36864).
    m = CPUOffloadingManager(
        num_blocks=9,
        cache_policy="arc",
        enable_events=True,
    )
    m.resolve_compact_mode(
        enable=True,
        total_bytes=9 * 4096,
        page_size=4096,
        group_payload_bytes={0: 4096, 1: 4096},
        preferred_eviction_groups={1},
    )

    # Group 0 (FA), 3 chunks, blocks_per_chunk=3
    # Group 1 (SWA), 3 chunks
    a = []
    for c in range(3):
        a.append(bpc_key(0, c, 0))
        a.append(bpc_key(0, c, 1))
        a.append(bpc_key(1, c, 0))

    b = []
    for c in range(3):
        b.append(bpc_key(0, c + 3, 0))
        b.append(bpc_key(1, c + 3, 0))

    ctx_a = ReqContext("a", store_replay_unit=tuple(a))
    out_a = m.prepare_store(a, ctx_a)
    assert out_a is not None
    m.complete_store(out_a.keys_to_store, ctx_a)

    # Budget is full, store b must succeed by evicting the full a.
    ctx_b = ReqContext("b", store_replay_unit=tuple(b))
    out_b = m.prepare_store(b, ctx_b)
    assert out_b is not None, "prepare_store b must succeed (evicts a)"

    # Verify a was evicted as a complete unit.
    a_evicted = all(m.lookup(k, ReqContext("p")) is LookupResult.MISS for k in a)
    assert a_evicted, "Failed: not all keys from unit a were evicted"

    m.complete_store(out_b.keys_to_store, ctx_b)

    # b keys should be HIT
    for k in b:
        assert m.lookup(k, ReqContext("p")) is LookupResult.HIT, (
            f"Key {k!r} from unit b should be HIT"
        )
