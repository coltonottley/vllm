# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.canonical_mapping import (
    _layer_mapping,
    _opaque_fallback_mapping,
    _RankContext,
    _verify_tiling,
    derive_canonical_mappings,
    derive_canonical_mappings_with_receipt,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVQuantMode,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)
from vllm.v1.kv_offload.base import CanonicalPageMapping, CopyRun

NUM_BLOCKS = 3


def _ctx(rank, tp=1, dcp=1, pcp=1, interleave=1, total=None, heads=2):
    return _RankContext(
        tp_size=tp,
        dcp_size=dcp,
        pcp_size=pcp,
        interleave=interleave,
        total_kv_heads=heads * tp if total is None else total,
        rank=rank,
    )


def _full_spec(num_kv_heads: int = 2, **kwargs) -> FullAttentionSpec:
    # block_size=4, head_dim=64, int8
    return FullAttentionSpec(
        block_size=4,
        num_kv_heads=num_kv_heads,
        head_size=64,
        dtype=torch.int8,
        **kwargs,
    )


def _mla_spec(**kwargs) -> MLAAttentionSpec:
    # page = 4 * 64 = 256B, one 64B latent row per token
    return MLAAttentionSpec(
        block_size=4, num_kv_heads=1, head_size=64, dtype=torch.int8, **kwargs
    )


def _split_nhd_cache(spec) -> torch.Tensor:
    return torch.zeros(
        NUM_BLOCKS,
        2,
        spec.block_size,
        spec.num_kv_heads,
        spec.head_size,
        dtype=torch.int8,
    )


def _split_hnd_cache(spec) -> torch.Tensor:
    return torch.zeros(
        NUM_BLOCKS,
        2,
        spec.num_kv_heads,
        spec.block_size,
        spec.head_size,
        dtype=torch.int8,
    ).permute(0, 1, 3, 2, 4)


def _packed_nhd_cache(spec) -> torch.Tensor:
    """Logical (num_blocks, heads, block_size, 2 * head_size) over an NHD
    physical layout — the FlashAttention/FlashInfer/Triton/Flex form."""
    return torch.zeros(
        NUM_BLOCKS,
        spec.block_size,
        spec.num_kv_heads,
        2 * spec.head_size,
        dtype=torch.int8,
    ).permute(0, 2, 1, 3)


def _packed_hnd_cache(spec) -> torch.Tensor:
    return torch.zeros(
        NUM_BLOCKS,
        spec.num_kv_heads,
        spec.block_size,
        2 * spec.head_size,
        dtype=torch.int8,
    )


CACHE_BUILDERS = {
    "split_nhd": _split_nhd_cache,
    "split_hnd": _split_hnd_cache,
    "packed_nhd": _packed_nhd_cache,
    "packed_hnd": _packed_hnd_cache,
}


def _try_mapping(spec, kv_cache, ctx) -> CanonicalPageMapping | None:
    return _layer_mapping(spec, kv_cache, NUM_BLOCKS, ctx)


def _mapping(spec, kv_cache, ctx) -> CanonicalPageMapping:
    mapping = _try_mapping(spec, kv_cache, ctx)
    assert mapping is not None
    return mapping


def _triples(runs: tuple[CopyRun, ...]) -> list[tuple[int, int, int]]:
    """Expand runs to explicit (local_offset, canonical_offset, size) copies."""
    out = []
    for run in runs:
        for i in range(run.num_fragments):
            out.append(
                (
                    run.local_offset + i * run.local_stride,
                    run.canonical_offset + i * run.canonical_stride,
                    run.fragment_size,
                )
            )
    return out


# ---------------------------------------------------------------------------
# TP-only placement (byte-compatible with the uniform interleave layout)
# ---------------------------------------------------------------------------


def test_split_nhd_placement_rank2_of_4():
    spec = _full_spec()
    mapping = _mapping(spec, _split_nhd_cache(spec), _ctx(rank=2, tp=4))
    assert mapping.canonical_page_size_bytes == 4 * 1024
    assert mapping.parallelism_agnostic
    k_dst = [256, 768, 1280, 1792]
    assert _triples(mapping.runs) == [
        (local, canonical, 128)
        for local, canonical in zip(
            [0, 128, 256, 384, 512, 640, 768, 896],
            k_dst + [2048 + o for o in k_dst],
        )
    ]
    # Heads are sharded, not replicated: this rank writes every block
    assert mapping.num_writers == 1


def test_packed_nhd_placement_rank2_of_4():
    spec = _full_spec()
    mapping = _mapping(spec, _packed_nhd_cache(spec), _ctx(rank=2, tp=4))
    assert _triples(mapping.runs) == [
        (0, 512, 256),
        (256, 1536, 256),
        (512, 2560, 256),
        (768, 3584, 256),
    ]


def test_packed_hnd_placement_rank1_of_4():
    spec = _full_spec()
    mapping = _mapping(spec, _packed_hnd_cache(spec), _ctx(rank=1, tp=4))
    # Two heads, each a contiguous canonical head region of 4 tokens x 128B
    assert _triples(mapping.runs) == [(0, 1024, 1024)]


@pytest.mark.parametrize("form", sorted(CACHE_BUILDERS))
def test_single_rank_coalesces_to_one_run(form):
    spec = _full_spec()
    mapping = _mapping(spec, CACHE_BUILDERS[form](spec), _ctx(rank=0))
    assert mapping.canonical_page_size_bytes == 1024
    assert _triples(mapping.runs) == [(0, 0, 1024)]


# ---------------------------------------------------------------------------
# Replication and writer election
# ---------------------------------------------------------------------------


def test_gqa_replicated_heads_rotate_writer():
    # total 2 KV heads on tp=4: replication factor 2, head shard = rank // 2
    spec = _full_spec(num_kv_heads=1)
    cache = _split_nhd_cache(spec)
    ctx = lambda rank: _ctx(rank, tp=4, total=2)  # noqa: E731
    rank2 = _mapping(spec, cache, ctx(2))
    rank3 = _mapping(spec, cache, ctx(3))
    assert rank2.canonical_page_size_bytes == 2 * 512
    # K region: head shard 1 at 64B offsets within 128B token rows
    assert _triples(rank2.runs)[:4] == [
        (0, 64, 64),
        (64, 192, 64),
        (128, 320, 64),
        (192, 448, 64),
    ]
    # Same head shard, so identical bytes: the two take alternate blocks
    assert rank3.runs == rank2.runs
    assert (rank2.is_writer(0), rank3.is_writer(0)) == (True, False)
    assert (rank2.is_writer(1), rank3.is_writer(1)) == (False, True)
    _verify_tiling("gqa", [_mapping(spec, cache, ctx(r)) for r in range(4)])


def test_mla_replicas_rotate_writer():
    spec = _mla_spec()
    rank0 = _mapping(spec, None, _ctx(rank=0, tp=2))
    rank1 = _mapping(spec, None, _ctx(rank=1, tp=2))
    # Latent pages are stored once per block, not once per rank
    assert rank0.canonical_page_size_bytes == 256
    assert _triples(rank0.runs) == [(0, 0, 256)]
    assert rank1.runs == rank0.runs
    assert (rank0.is_writer(0), rank1.is_writer(0)) == (True, False)
    assert (rank0.is_writer(1), rank1.is_writer(1)) == (False, True)


# ---------------------------------------------------------------------------
# DCP / PCP token sharding
# ---------------------------------------------------------------------------


def test_dcp_interleaves_tokens_within_replicas():
    # tp=4, dcp=2, total 2 KV heads: head shard = rank // 2, cp rank = rank % 2
    spec = _full_spec(num_kv_heads=1)
    cache = _split_nhd_cache(spec)
    ctx = lambda rank: _ctx(rank, tp=4, dcp=2, total=2)  # noqa: E731
    per_rank = [_mapping(spec, cache, ctx(rank)) for rank in range(4)]
    assert all(m is not None for m in per_rank)
    # 8 canonical tokens x 2 heads x 64B per region
    assert per_rank[0].canonical_page_size_bytes == 2048
    assert not per_rank[0].parallelism_agnostic
    # rank 2 = head shard 1, cp rank 0: K tokens 0,2,4,6 at head offset 64
    assert _triples(per_rank[2].runs)[:4] == [
        (0, 64, 64),
        (64, 320, 64),
        (128, 576, 64),
        (192, 832, 64),
    ]
    # every rank contributes (dcp == replication: no residual replicas)
    assert all(m.num_writers == 1 for m in per_rank)
    _verify_tiling("dcp", per_rank)


def test_mla_dcp_shards_latent_tokens():
    spec = _mla_spec()
    ctx = lambda rank: _ctx(rank, tp=2, dcp=2)  # noqa: E731
    rank0 = _mapping(spec, None, ctx(0))
    rank1 = _mapping(spec, None, ctx(1))
    assert rank0.canonical_page_size_bytes == 512
    assert _triples(rank0.runs) == [(o, 2 * o, 64) for o in (0, 64, 128, 192)]
    assert _triples(rank1.runs) == [(o, 2 * o + 64, 64) for o in (0, 64, 128, 192)]
    _verify_tiling("mla-dcp", [rank0, rank1])


def test_pcp_tokens_and_tp_heads_compose():
    # tp=2 x pcp=2: rank = pcp_rank * 2 + tp_rank; 4 workers tile the page
    spec = _full_spec(num_kv_heads=1)
    cache = _packed_nhd_cache(spec)
    per_rank = [
        _mapping(spec, cache, _ctx(rank, tp=2, pcp=2, total=2)) for rank in range(4)
    ]
    assert all(m is not None and m.runs for m in per_rank)
    _verify_tiling("pcp", per_rank)


def test_interleave_chunks_stay_contiguous():
    # interleave=2: chunks of 2 tokens alternate between the 2 cp ranks and
    # coalesce into one contiguous fragment per chunk
    spec = _mla_spec()
    mapping = _mapping(spec, None, _ctx(rank=0, tp=2, dcp=2, interleave=2))
    assert _triples(mapping.runs) == [(0, 0, 128), (128, 256, 128)]
    _verify_tiling(
        "interleave",
        [
            _mapping(spec, None, _ctx(rank, tp=2, dcp=2, interleave=2))
            for rank in range(2)
        ],
    )


@pytest.mark.parametrize("form", sorted(CACHE_BUILDERS))
def test_all_ranks_tile_canonical_page(form):
    spec = _full_spec()
    per_rank = [
        _mapping(spec, CACHE_BUILDERS[form](spec), _ctx(rank, tp=4))
        for rank in range(4)
    ]
    _verify_tiling("layer", per_rank)


# ---------------------------------------------------------------------------
# Byte-level round trips
# ---------------------------------------------------------------------------


def _store_all(mappings, pages, size: int, block_id: int = 0) -> bytes:
    buf = bytearray(size)
    for mapping, page in zip(mappings, pages):
        if not mapping.is_writer(block_id):
            continue
        for local, canonical, n in _triples(mapping.runs):
            buf[canonical : canonical + n] = page[local : local + n]
    return bytes(buf)


def _load_one(mapping, canonical_bytes: bytes) -> bytes:
    page = bytearray(mapping.local_page_size_bytes)
    for local, canonical, n in _triples(mapping.runs):
        page[local : local + n] = canonical_bytes[canonical : canonical + n]
    return bytes(page)


@pytest.mark.parametrize("form", sorted(CACHE_BUILDERS))
def test_cross_tp_store_load(form):
    """Bytes stored under one TP size are the bytes another TP size loads."""
    total_heads, canonical_size = 8, 4096
    reference = bytes((7 + 31 * i) % 256 for i in range(canonical_size))

    def mappings_at(tp: int):
        spec = _full_spec(num_kv_heads=total_heads // tp)
        cache = CACHE_BUILDERS[form](spec)
        return [
            _mapping(spec, cache, _ctx(rank, tp=tp, total=total_heads))
            for rank in range(tp)
        ]

    for tp in (4, 2, 1):
        mappings = mappings_at(tp)
        assert all(m is not None for m in mappings)
        pages = [_load_one(m, reference) for m in mappings]
        assert _store_all(mappings, pages, canonical_size) == reference


def test_cp_round_trip():
    # tp=4 / dcp=2 / 2 KV heads: 4 workers jointly hold one canonical page
    spec = _full_spec(num_kv_heads=1)
    cache = _split_nhd_cache(spec)
    mappings = [
        _mapping(spec, cache, _ctx(rank, tp=4, dcp=2, total=2)) for rank in range(4)
    ]
    reference = bytes((3 + 17 * i) % 256 for i in range(2048))
    pages = [_load_one(m, reference) for m in mappings]
    assert _store_all(mappings, pages, 2048) == reference


def test_replica_rotation_round_trip():
    """Whichever replica a block elects reproduces the same canonical page."""
    spec = _mla_spec()
    mappings = [_mapping(spec, None, _ctx(rank, tp=2)) for rank in range(2)]
    reference = bytes((5 + 11 * i) % 256 for i in range(256))
    pages = [_load_one(m, reference) for m in mappings]
    for block_id in (0, 1):
        assert _store_all(mappings, pages, 256, block_id) == reference


# ---------------------------------------------------------------------------
# Fail-closed gates
# ---------------------------------------------------------------------------


def test_fail_closed_cases():
    spec = _full_spec()
    nhd = _split_nhd_cache(spec)
    # Spec heads inconsistent with total heads / tp
    assert _try_mapping(spec, nhd, _ctx(0, tp=4, total=2)) is None
    # tp not divisible by total KV heads
    one_head = _full_spec(num_kv_heads=1)
    assert (
        _try_mapping(one_head, _split_nhd_cache(one_head), _ctx(0, tp=3, total=2))
        is None
    )
    # DCP wider than the KV replication factor (tokens would shard across
    # ranks holding different heads)
    assert _try_mapping(spec, nhd, _ctx(0, tp=4, dcp=2)) is None
    # Interleave must divide the block size
    assert _try_mapping(spec, nhd, _ctx(0, tp=2, dcp=2, interleave=3, total=2)) is None
    # Per-token-head scales are packed with the data
    quant_spec = _full_spec(kv_quant_mode=KVQuantMode.FP8_PER_TOKEN_HEAD)
    assert _try_mapping(quant_spec, _split_nhd_cache(quant_spec), _ctx(0, tp=4)) is None
    # Compressed MLA slots are not 1:1 with tokens
    assert _try_mapping(_mla_spec(compress_ratio=2), None, _ctx(0, tp=2, dcp=2)) is None
    # Unrecognized physical layouts
    swapped = torch.zeros(
        NUM_BLOCKS,
        spec.block_size,
        2,
        spec.num_kv_heads,
        spec.head_size,
        dtype=torch.int8,
    ).permute(0, 2, 1, 3, 4)
    assert _try_mapping(spec, swapped, _ctx(0, tp=4)) is None
    # Non-attention specs
    assert _try_mapping(KVCacheSpec(block_size=4), None, _ctx(0, tp=4, total=8)) is None


def test_opaque_fallback_places_page_whole():
    mapping = _opaque_fallback_mapping(1024, 4, 2)
    assert mapping.canonical_page_size_bytes == 4096
    assert not mapping.parallelism_agnostic
    assert _triples(mapping.runs) == [(0, 2048, 1024)]
    assert mapping.num_writers == 1
    _verify_tiling("opaque", [_opaque_fallback_mapping(1024, 4, r) for r in range(4)])


# ---------------------------------------------------------------------------
# derive_canonical_mappings end to end
# ---------------------------------------------------------------------------


def _vllm_config(tp=1, dcp=1, pcp=1, pp=1, interleave=1, total_kv_heads=2):
    config = MagicMock()
    config.parallel_config.tensor_parallel_size = tp
    config.parallel_config.decode_context_parallel_size = dcp
    config.parallel_config.prefill_context_parallel_size = pcp
    config.parallel_config.cp_kv_cache_interleave_size = interleave
    config.parallel_config.world_size = pp * tp * pcp
    config.parallel_config.rank = 0
    config.model_config.get_total_num_kv_heads.return_value = total_kv_heads
    return config


def _kv_cache_config(groups):
    config = MagicMock()
    config.kv_cache_groups = groups
    config.num_blocks = NUM_BLOCKS
    return config


def test_derive_mixed_model_with_dcp():
    attn_spec = _full_spec(num_kv_heads=1)
    mla_spec = _mla_spec()
    quant_spec = _full_spec(
        num_kv_heads=1, kv_quant_mode=KVQuantMode.FP8_PER_TOKEN_HEAD
    )
    kv_cache_config = _kv_cache_config(
        [
            KVCacheGroupSpec(layer_names=["attn"], kv_cache_spec=attn_spec),
            KVCacheGroupSpec(layer_names=["mla"], kv_cache_spec=mla_spec),
            KVCacheGroupSpec(layer_names=["quant"], kv_cache_spec=quant_spec),
        ]
    )
    kv_caches = {
        "attn": _split_nhd_cache(attn_spec),
        "quant": _split_nhd_cache(quant_spec),
    }
    mappings = derive_canonical_mappings(
        _vllm_config(tp=4, dcp=2, total_kv_heads=2), kv_cache_config, kv_caches
    )
    assert set(mappings) == {"attn", "mla", "quant"}
    assert not mappings["attn"].parallelism_agnostic
    assert mappings["attn"].runs
    # Uncertifiable layers degrade to an opaque page, never disappear
    assert not mappings["quant"].parallelism_agnostic
    assert (
        mappings["quant"].canonical_page_size_bytes
        == 4 * quant_spec.unpadded_page_size_bytes
    )


def test_derive_refuses_foreign_worker_groups():
    attn_spec = _full_spec()
    kv_cache_config = _kv_cache_config(
        [KVCacheGroupSpec(layer_names=["attn"], kv_cache_spec=attn_spec)]
    )
    kv_caches = {"attn": _split_nhd_cache(attn_spec)}
    assert (
        derive_canonical_mappings(
            _vllm_config(tp=2, pp=2, total_kv_heads=4), kv_cache_config, kv_caches
        )
        == {}
    )


# ---------------------------------------------------------------------------
# Compressed MLA (compress_ratio > 1): TP-replicated, rank 0 writes
# ---------------------------------------------------------------------------


def test_compressed_mla_rank0_writer_tp_only():
    """Compressed MLA with tp=2: rank 0 writes whole-page identity, rank 1 loads."""
    spec = _mla_spec(compress_ratio=4)
    # block_size=4, storage_block_size=1, page=1*1*64*1=64B
    writer = _mapping(spec, None, _ctx(rank=0, tp=2))
    reader = _mapping(spec, None, _ctx(rank=1, tp=2))
    assert writer.canonical_page_size_bytes == 64
    assert writer.local_page_size_bytes == 64
    assert _triples(writer.runs) == [(0, 0, 64)]
    assert _triples(reader.runs) == [(0, 0, 64)]
    assert writer.is_writer(0)  # rank 0 writes block 0
    assert not reader.is_writer(0)  # rank 1 does not write block 0
    assert writer.parallelism_agnostic
    assert reader.parallelism_agnostic


def test_compressed_mla_dsv4_ratio_128():
    """Compressed MLA with compress_ratio=128, DSV4-like geometry."""
    spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.int8,
        compress_ratio=128,
    )
    # storage_block_size=2, page=2*1*512*1=1024B
    writer = _mapping(spec, None, _ctx(rank=0, tp=4))
    reader = _mapping(spec, None, _ctx(rank=3, tp=4))
    assert writer.canonical_page_size_bytes == 1024
    assert _triples(writer.runs) == [(0, 0, 1024)]
    assert reader.runs == writer.runs
    assert writer.is_writer(0)  # rank 0 writes block 0
    assert not reader.is_writer(0)  # rank 3 does not write block 0
    assert writer.parallelism_agnostic


def test_compressed_mla_byte_roundtrip():
    """Store via rank 0, load via rank 1: bytes survive cleanly."""
    spec = _mla_spec(compress_ratio=4)
    rank0 = _mapping(spec, None, _ctx(rank=0, tp=2))
    rank1 = _mapping(spec, None, _ctx(rank=1, tp=2))
    ref = bytes((11 + 7 * i) % 256 for i in range(64))
    # rank 0 stores into canonical buffer
    buf = bytearray(64)
    for local, canonical, n in _triples(rank0.runs):
        if rank0.is_writer(0):
            buf[canonical : canonical + n] = ref[local : local + n]
    # rank 1 loads from canonical
    page = bytearray(rank1.local_page_size_bytes)
    for local, canonical, n in _triples(rank1.runs):
        page[local : local + n] = buf[canonical : canonical + n]
    assert bytes(page) == ref


def test_compressed_mla_fail_closed():
    """Compressed MLA fails closed with CP, per-token-head quant, negative page."""
    spec = _mla_spec(compress_ratio=4)
    # CP (dcp=2) not allowed for compressed MLA
    assert _try_mapping(spec, None, _ctx(0, tp=2, dcp=2)) is None
    # Per-token-head quant
    quant = _mla_spec(compress_ratio=4, kv_quant_mode=KVQuantMode.FP8_PER_TOKEN_HEAD)
    assert _try_mapping(quant, None, _ctx(0, tp=2)) is None


# ---------------------------------------------------------------------------
# SlidingWindowMLASpec: 3-D contiguous-inner slab, rank 0 writes
# ---------------------------------------------------------------------------


def _swa_mla_spec(**kwargs) -> SlidingWindowMLASpec:
    """Default SlidingWindowMLASpec: block_size=4, num_kv_heads=1, uint8."""
    base = dict(
        block_size=4,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.int8,
        sliding_window=128,
    )
    base.update(kwargs)
    return SlidingWindowMLASpec(**base)


def test_sliding_window_mla_identity():
    """Basic 3-D contiguous slab, rank 0 writes, rank 1 loads."""
    spec = _swa_mla_spec()
    # SlidingWindowMLASpec page = storage_block_size * num_kv_heads * head_size
    # with compress_ratio=1: page = 4 * 1 * 64 = 256B
    cache = torch.zeros(NUM_BLOCKS, 4, 64, dtype=torch.int8)
    writer = _mapping(spec, cache, _ctx(rank=0, tp=2, total=1))
    reader = _mapping(spec, cache, _ctx(rank=1, tp=2, total=1))
    assert writer.canonical_page_size_bytes == 256
    assert _triples(writer.runs) == [(0, 0, 256)]
    assert _triples(reader.runs) == [(0, 0, 256)]
    assert writer.is_writer(0)
    assert not reader.is_writer(0)
    assert writer.parallelism_agnostic


def test_sliding_window_mla_dsv4_fp8():
    """DSV4 FP8 SWA: (N,64,584) slab with padded outer stride & storage offset."""
    slab_dim = 64
    inner_dim = 584
    page = slab_dim * inner_dim  # 37376 for uint8
    spec = SlidingWindowMLASpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        sliding_window=4096,
        compress_ratio=4,
        cache_dtype_str="fp8_ds_mla",
        model_version="deepseek_v4",
    )
    assert spec.real_page_size_bytes == page
    # Build a 1-D buffer and use as_strided to create padded outer stride
    # with nonzero storage offset
    padded_stride = page + 64  # outer stride larger than page
    storage_size = (NUM_BLOCKS - 1) * padded_stride + page + 128
    data = torch.zeros(storage_size, dtype=torch.uint8)
    cache = torch.as_strided(
        data,
        size=(NUM_BLOCKS, slab_dim, inner_dim),
        stride=(padded_stride, inner_dim, 1),
        storage_offset=32,
    )
    assert cache.storage_offset() == 32
    assert cache.stride()[0] == padded_stride
    mapping = _mapping(spec, cache, _ctx(rank=0, tp=2, total=1))
    assert mapping.canonical_page_size_bytes == page
    assert _triples(mapping.runs) == [(0, 0, page)]
    assert mapping.parallelism_agnostic


def test_sliding_window_mla_byte_roundtrip():
    """SWA MLA store/load roundtrip via identity mapping."""
    spec = _swa_mla_spec()
    cache = torch.zeros(NUM_BLOCKS, 4, 64, dtype=torch.int8)
    rank0 = _mapping(spec, cache, _ctx(rank=0, tp=2, total=1))
    rank1 = _mapping(spec, cache, _ctx(rank=1, tp=2, total=1))
    ref = bytes((13 + 5 * i) % 256 for i in range(256))
    buf = bytearray(256)
    for local, canonical, n in _triples(rank0.runs):
        if rank0.is_writer(0):
            buf[canonical : canonical + n] = ref[local : local + n]
    page = bytearray(rank1.local_page_size_bytes)
    for local, canonical, n in _triples(rank1.runs):
        page[local : local + n] = buf[canonical : canonical + n]
    assert bytes(page) == ref


def test_sliding_window_mla_fail_closed():
    """SlidingWindowMLASpec fails closed for CP, wrong heads, bad geometry."""
    spec = _swa_mla_spec()
    good = torch.zeros(NUM_BLOCKS, 4, 64, dtype=torch.int8)
    # CP not allowed
    assert _try_mapping(spec, good, _ctx(0, tp=2, dcp=2, total=1)) is None
    # Wrong num_kv_heads (requires 1)
    multi_head = _swa_mla_spec(num_kv_heads=2)
    assert _try_mapping(multi_head, good, _ctx(0, tp=4, total=4)) is None
    # Wrong total_kv_heads (requires 1)
    assert _try_mapping(spec, good, _ctx(0, tp=2, total=2)) is None
    # 2-D tensor (not 3-D)
    flat = torch.zeros(NUM_BLOCKS, 256, dtype=torch.int8)
    assert _try_mapping(spec, flat, _ctx(0, tp=2, total=1)) is None
    # 4-D tensor
    four_d = torch.zeros(NUM_BLOCKS, 4, 64, 1, dtype=torch.int8)
    assert _try_mapping(spec, four_d, _ctx(0, tp=2, total=1)) is None
    # Wrong inner stride (non-contiguous)
    non_contig = torch.zeros(NUM_BLOCKS, 4, 64, dtype=torch.int8).transpose(1, 2)
    # After transpose: shape (3, 64, 4), stride (256, 1, 64) — inner is packed
    # but totalling to page=256 fails because D1*D2*elem != 256
    assert _try_mapping(spec, non_contig, _ctx(0, tp=2, total=1)) is None
    # Wrong outer dim (num_blocks mismatch)
    wrong_blocks = torch.zeros(NUM_BLOCKS + 1, 4, 64, dtype=torch.int8)
    assert _try_mapping(spec, wrong_blocks, _ctx(0, tp=2, total=1)) is None
    # Per-token-head quant (fp8 per-token-head scales) fails closed
    quant_spec = _swa_mla_spec(kv_quant_mode=KVQuantMode.FP8_PER_TOKEN_HEAD)
    assert _try_mapping(quant_spec, good, _ctx(0, tp=2, total=1)) is None


# ---------------------------------------------------------------------------
# Generic attention unaffected by new branches
# ---------------------------------------------------------------------------


def test_generic_attention_works_alongside_new_branches():
    """Pre-existing full-attention specs pass through the generic path."""
    spec = _full_spec()
    cache = _split_nhd_cache(spec)
    mapping = _mapping(spec, cache, _ctx(rank=0, tp=2))
    # K + V regions, each 4 tokens x 2 heads x 128B = 1024B; 2 regions
    assert mapping.canonical_page_size_bytes == 2 * 1024
    # 8 fragments: 4 tokens x 2 regions, canonical stride=2x local stride
    assert len(_triples(mapping.runs)) == 8
    assert len(mapping.runs) > 0


# ---------------------------------------------------------------------------
# derive_canonical_mappings_with_receipt tests
# ---------------------------------------------------------------------------


def _receipt_config(tp=1, dcp=1, pcp=1, pp=1, interleave=1, total_kv_heads=2):
    config = MagicMock()
    config.parallel_config.tensor_parallel_size = tp
    config.parallel_config.decode_context_parallel_size = dcp
    config.parallel_config.prefill_context_parallel_size = pcp
    config.parallel_config.cp_kv_cache_interleave_size = interleave
    config.parallel_config.world_size = pp * tp * pcp
    config.parallel_config.rank = 0
    config.model_config.get_total_num_kv_heads.return_value = total_kv_heads
    return config


def test_receipt_basic_mla():
    """Receipt is produced for valid MLA config and contains correct fields."""
    spec = _mla_spec()
    kv_cache_config = _kv_cache_config(
        [KVCacheGroupSpec(layer_names=["mla"], kv_cache_spec=spec)]
    )
    mappings, receipt = derive_canonical_mappings_with_receipt(
        _receipt_config(tp=2), kv_cache_config, {}
    )
    assert "mla" in mappings
    assert receipt is not None
    assert receipt.certified
    assert not receipt.fallback
    assert receipt.layer_names == ("mla",)
    assert len(receipt.per_rank) == 2  # one per rank (tp=2)
    for rr in receipt.per_rank:
        assert 0 <= rr.rank < 2
        assert rr.layer_name == "mla"
        assert rr.num_writers == 2
    # Receipt is identical for the same config (deterministic)
    mappings2, receipt2 = derive_canonical_mappings_with_receipt(
        _receipt_config(tp=2), kv_cache_config, {}
    )
    assert receipt == receipt2
    assert hash(receipt) == hash(receipt2)


def test_receipt_rank0_rank1_identical():
    """Rank 0 and rank 1 produce identical receipts even though local
    mappings differ (rotating writer_index).  Regression: per_layer was
    previously populated from my_rank, making receipts unequal."""
    spec = _mla_spec()
    kv_cache_config = _kv_cache_config(
        [KVCacheGroupSpec(layer_names=["mla"], kv_cache_spec=spec)]
    )
    # Rank 0 config
    cfg0 = _receipt_config(tp=2)
    cfg0.parallel_config.rank = 0
    mappings0, receipt0 = derive_canonical_mappings_with_receipt(
        cfg0, kv_cache_config, {}
    )
    # Rank 1 config (same except rank)
    cfg1 = _receipt_config(tp=2)
    cfg1.parallel_config.rank = 1
    mappings1, receipt1 = derive_canonical_mappings_with_receipt(
        cfg1, kv_cache_config, {}
    )
    # Local mappings differ by writer_index
    assert mappings0["mla"].writer_index == 0
    assert mappings1["mla"].writer_index == 1
    assert mappings0["mla"] != mappings1["mla"]
    # But receipts MUST be identical (role-neutral, both contain all ranks)
    assert receipt0 == receipt1
    assert hash(receipt0) == hash(receipt1)
    # Each receipt has both ranks' data
    assert len(receipt0.per_rank) == 2
    rank_indices = sorted(rr.rank for rr in receipt0.per_rank)
    assert rank_indices == [0, 1]


def test_receipt_malformed_writer_indices_rejected():
    """_verify_tiling rejects duplicate or missing writer indices."""
    # Build a set of mappings where one rank claims to be a writer
    # for all blocks but the other doesn't — causes tiling failure.
    spec = _mla_spec(compress_ratio=4)  # compressed MLA
    cache = None
    ctx = lambda rank: _RankContext(
        tp_size=2,
        dcp_size=1,
        pcp_size=1,
        interleave=1,
        total_kv_heads=1,
        rank=rank,
    )
    per_rank = [_layer_mapping(spec, cache, 3, ctx(r)) for r in range(2)]
    # Both mappings are valid identity mappings with num_writers=2
    assert all(m is not None for m in per_rank)
    # This should pass _verify_tiling since both have correct rotating writers
    _verify_tiling("test", per_rank)

    # Manually create a broken mapping with duplicate writer claim
    run = CopyRun(0, 0, 64, 1, 64, 64)
    good = CanonicalPageMapping(64, 64, (run,), 2, 0, True)
    duplicate = CanonicalPageMapping(
        64, 64, (run,), 2, 0, True
    )  # same writer_index as good!
    # Two ranks both claim writer_index=0 with num_writers=2
    # Block 0: both write (duplicate cover) — assertion fires here
    with pytest.raises(AssertionError, match="do not tile"):
        _verify_tiling("dup", [good, duplicate])


def test_receipt_empty_when_no_layers():
    """No certifiable layers returns None receipt."""
    spec = MagicMock()  # Non-attention spec
    kv_cache_config = _kv_cache_config(
        [KVCacheGroupSpec(layer_names=["unknown"], kv_cache_spec=spec)]
    )
    mappings, receipt = derive_canonical_mappings_with_receipt(
        _receipt_config(tp=2), kv_cache_config, {}
    )
    assert receipt is None


def test_tp2_adjacent_block_rotation_tiles_once():
    """TP2 adjacent block IDs elect rotating writers that each tile canonical
    page exactly once."""
    spec = _mla_spec()  # compress_ratio=1, tp=2
    cache = None
    ctx0 = _ctx(rank=0, tp=2)
    ctx1 = _ctx(rank=1, tp=2)
    m0 = _mapping(spec, cache, ctx0)
    m1 = _mapping(spec, cache, ctx1)
    # With tp=2, both ranks hold identical bytes, num_writers=2
    assert m0.num_writers == 2
    assert m1.num_writers == 2
    assert m0.writer_index == 0
    assert m1.writer_index == 1
    # Adjacent block IDs rotate writer
    assert m0.is_writer(0)  # rank 0 writes block 0
    assert not m1.is_writer(0)  # rank 1 does NOT write block 0
    assert not m0.is_writer(1)  # rank 0 does NOT write block 1
    assert m1.is_writer(1)  # rank 1 writes block 1
    assert m0.is_writer(2)  # rank 0 writes block 2
    assert not m1.is_writer(2)
    # Both retain complete runs (identical)
    assert m0.runs == m1.runs
    # _verify_tiling passes: block 0 tiled by rank0, block 1 by rank1
    _verify_tiling("tp2-rotation", [m0, m1])
