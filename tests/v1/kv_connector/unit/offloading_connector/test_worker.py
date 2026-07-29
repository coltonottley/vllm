# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import defaultdict
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch

from vllm.config import DeviceConfig, ParallelConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    TransferJob,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.utils import set_kv_cache_layout
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalPageMapping,
    CopyRun,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingManager,
    OffloadingSpec,
    OffloadingWorker,
)
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.cpu.common import (
    CompactGroupGeometry,
    CompactLayerGeometry,
    derive_compact_group_geometry,
)
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker

NUM_BLOCKS = 10
BLOCK_SIZE = 16
NUM_KV_HEADS = 4
HEAD_SIZE = 64
DTYPE = torch.float16
DEVICE_TYPE = current_platform.device_type

# Attention backends to test
ATTN_BACKENDS: list[str] = []
if current_platform.is_cuda():
    ATTN_BACKENDS = [
        "FLASH_ATTN",
        "FLEX_ATTENTION",
        "FLASHINFER",
        "TRITON_ATTN",
    ]
elif current_platform.is_rocm():
    ATTN_BACKENDS = ["TRITON_ATTN"]
elif current_platform.is_xpu():
    ATTN_BACKENDS = ["TRITON_ATTN", "FLASH_ATTN"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _allocate_and_reshape_kv_caches(
    kv_cache_config: KVCacheConfig,
    attn_groups: list[list],
    device: torch.device,
):
    """
    Use the real GPUModelRunner allocation and reshape methods to produce
    kv_caches, just like the model runner does during initialization.
    """
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    # Some backends (e.g. FlashAttention) query the KV cache layout during
    # reshape, which ultimately calls get_current_vllm_config(). Setting
    # the layout override avoids needing a full VllmConfig context.
    set_kv_cache_layout("NHD")
    try:
        runner = object.__new__(GPUModelRunner)
        runner.device = device
        runner.runner_only_attn_layers = set()
        runner.attn_groups = attn_groups
        runner.kv_cache_config = kv_cache_config
        runner.cache_config = MagicMock(cache_dtype="auto")
        runner.shared_kv_cache_layers = {}
        runner.model_config = MagicMock()
        runner.model_config.hf_config.model_type = ""
        runner.compilation_config = MagicMock(
            static_forward_context=defaultdict(MagicMock)
        )
        runner.kv_caches = []

        kernel_block_sizes = [BLOCK_SIZE] * len(kv_cache_config.kv_cache_groups)
        return runner.initialize_kv_cache_tensors(kv_cache_config, kernel_block_sizes)
    finally:
        set_kv_cache_layout(None)


def _single_rank_vllm_config(total_kv_heads: int):
    """A one-rank (TP=1) parallel config, as canonical mappings are derived
    from it."""
    vllm_config = MagicMock()
    parallel_config = vllm_config.parallel_config
    parallel_config.tensor_parallel_size = 1
    parallel_config.decode_context_parallel_size = 1
    parallel_config.prefill_context_parallel_size = 1
    parallel_config.cp_kv_cache_interleave_size = 1
    parallel_config.world_size = 1
    parallel_config.rank = 0
    vllm_config.model_config.get_total_num_kv_heads.return_value = total_kv_heads
    return vllm_config


def _make_worker(
    kv_cache_config: KVCacheConfig,
    replicated_layout: bool = False,
    rank: int = 0,
    vllm_config: VllmConfig | None = None,
):
    """
    Create an OffloadingConnectorWorker with mocked dependencies.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
        OffloadingConnectorWorker,
    )

    if vllm_config is None:
        vllm_config = VllmConfig(
            device_config=DeviceConfig("cpu"),
            parallel_config=ParallelConfig(pipeline_parallel_size=2),
        )

    spec = MagicMock(spec=OffloadingSpec)
    spec.replicated_layout = replicated_layout
    spec.config = MagicMock()
    spec.config.parallel.rank = rank
    spec.get_worker.return_value = MagicMock()

    worker = OffloadingConnectorWorker(
        spec=spec,
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
    )
    worker.worker = MagicMock()

    return worker, spec


def _store_metadata(job_id: int) -> OffloadingConnectorMetadata:
    return OffloadingConnectorMetadata(
        load_jobs={},
        store_jobs={
            job_id: TransferJob(
                req_id="req",
                src_spec=GPULoadStoreSpec([0], group_sizes=(1,), block_indices=(0,)),
                dst_spec=LoadStoreSpec(),
            )
        },
    )


def _load_metadata(job_id: int) -> OffloadingConnectorMetadata:
    return OffloadingConnectorMetadata(
        load_jobs={
            job_id: TransferJob(
                req_id="req",
                src_spec=LoadStoreSpec(),
                dst_spec=GPULoadStoreSpec([0], group_sizes=(1,), block_indices=(0,)),
            )
        },
        store_jobs={},
    )


def _empty_metadata() -> OffloadingConnectorMetadata:
    return OffloadingConnectorMetadata(load_jobs={}, store_jobs={})


def _offloading_config(rank: int = 0) -> OffloadingConfig:
    return OffloadingConfig(
        groups=(),
        worker_kv_bytes_per_block=0,
        enable_kv_cache_events=False,
        extra_config={},
        engine_id="test-engine",
        model=OffloadingModelConfig(name="test-model", dtype="float16"),
        cache=OffloadingCacheConfig(tokens_per_hash=16, blocks_per_chunk=1),
        parallel=OffloadingParallelConfig(
            rank=rank,
            world_size=2,
            tp_size=2,
            pp_size=1,
            pcp_size=1,
            dcp_size=1,
            data_parallel_index=0,
            is_parallelism_agnostic=False,
        ),
    )


class BareExternalOffloadingSpec(OffloadingSpec):
    def get_manager(self) -> OffloadingManager:
        raise NotImplementedError

    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_prepare_store_kv_non_writer_marks_completed_without_submit():
    worker, _ = _make_worker(
        KVCacheConfig(num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[]),
        replicated_layout=True,
        rank=1,
    )

    worker.prepare_store_kv(_store_metadata(7))
    worker.start_kv_transfers(_empty_metadata())

    assert worker._unsubmitted_store_jobs == []
    assert worker.worker is not None
    worker.worker.submit_store.assert_not_called()
    meta = worker.build_connector_worker_meta()
    assert meta is not None
    assert meta.completed_jobs == {7: 1}


def test_prepare_store_kv_writer_submits_store():
    worker, _ = _make_worker(
        KVCacheConfig(num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[]),
        replicated_layout=True,
        rank=0,
    )

    worker.prepare_store_kv(_store_metadata(8))
    assert worker.build_connector_worker_meta() is None
    worker.start_kv_transfers(_empty_metadata())

    assert worker.worker is not None
    worker.worker.submit_store.assert_called_once()


def test_prepare_store_kv_non_replicated_rank_gt_zero_queues_store():
    worker, _ = _make_worker(
        KVCacheConfig(num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[]),
        replicated_layout=False,
        rank=1,
    )

    worker.prepare_store_kv(_store_metadata(9))
    assert worker.build_connector_worker_meta() is None
    assert len(worker._unsubmitted_store_jobs) == 1

    worker.start_kv_transfers(_empty_metadata())

    assert worker.worker is not None
    worker.worker.submit_store.assert_called_once()
    assert worker._unsubmitted_store_jobs == []


def test_handle_preemptions_non_writer_acks_flushed_store():
    worker, _ = _make_worker(
        KVCacheConfig(num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[]),
        replicated_layout=True,
        rank=1,
    )
    metadata = _store_metadata(10)
    metadata.jobs_to_flush = {10}

    worker.handle_preemptions(metadata)

    assert worker.worker is not None
    worker.worker.submit_store.assert_not_called()
    worker.worker.wait.assert_called_once_with({10})
    assert metadata.store_jobs == {}
    meta = worker.build_connector_worker_meta()
    assert meta is not None
    assert meta.completed_jobs == {10: 1}


def test_start_kv_transfers_non_writer_still_submits_load():
    worker, _ = _make_worker(
        KVCacheConfig(num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[]),
        replicated_layout=True,
        rank=1,
    )

    worker.start_kv_transfers(_load_metadata(10))

    assert worker.worker is not None
    worker.worker.submit_load.assert_called_once()
    assert worker.build_connector_worker_meta() is None


def test_offloading_connector_worker_accepts_plugin_spec_default_layout():
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
        OffloadingConnectorWorker,
    )

    spec = BareExternalOffloadingSpec(_offloading_config(rank=1))

    OffloadingConnectorWorker(
        spec=spec,
        vllm_config=VllmConfig(
            device_config=DeviceConfig("cpu"),
            parallel_config=ParallelConfig(pipeline_parallel_size=2),
        ),
        kv_cache_config=KVCacheConfig(
            num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[]
        ),
    )

    assert spec.replicated_layout is False


@pytest.mark.parametrize("backend", ATTN_BACKENDS)
def test_register_kv_caches(backend):
    """Test register_kv_caches with multiple groups covering all layer types.

    Creates one FullAttention group, one MLA group, one Mamba group, and
    one Mamba-padded group. Each group has GROUP_SIZE layers.

    KVCacheTensors are shared across all groups mirroring the real allocation
    in kv_cache_utils.py: tensor i is shared by layer i from every group.
    The padded-mamba group has a different page size so its layers get their
    own dedicated tensors.

    Uses the real GPUModelRunner.initialize_kv_cache_tensors to produce
    the raw per-layer kv_caches registered by the connector.

    Verifies that the canonicalized CanonicalKVCaches has the correct
    block tensors, tensor_idx references, and page sizes across all groups.
    """
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerBackend,
    )
    from vllm.v1.worker.utils import AttentionGroup

    MLA_HEAD_SIZE = NUM_KV_HEADS * HEAD_SIZE * 2

    # padded mamba (missing HEAD_SIZE)
    CONV_STATE_SHAPE = (BLOCK_SIZE * NUM_KV_HEADS, HEAD_SIZE)
    UNALIGNED_SSM_STATE_SHAPE = (BLOCK_SIZE * NUM_KV_HEADS - 1, HEAD_SIZE)

    PAGE_SIZE_BYTES = 2 * BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE * get_dtype_size(DTYPE)
    unaligned_mamba_page_size = PAGE_SIZE_BYTES - HEAD_SIZE * get_dtype_size(DTYPE)

    # unpadded mamba (fills page exactly)
    ALIGNED_SSM_STATE_SHAPE = (BLOCK_SIZE * NUM_KV_HEADS, HEAD_SIZE)

    backend_cls = AttentionBackendEnum[backend].get_class()

    attn_spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
    )
    mla_spec = MLAAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=MLA_HEAD_SIZE,
        dtype=DTYPE,
    )
    unaligned_mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=(CONV_STATE_SHAPE, UNALIGNED_SSM_STATE_SHAPE),
        dtypes=(DTYPE, DTYPE),
        page_size_padded=PAGE_SIZE_BYTES,
    )
    aligned_mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=(CONV_STATE_SHAPE, ALIGNED_SSM_STATE_SHAPE),
        dtypes=(DTYPE, DTYPE),
        page_size_padded=PAGE_SIZE_BYTES,
    )

    assert attn_spec.page_size_bytes == PAGE_SIZE_BYTES
    assert mla_spec.page_size_bytes == PAGE_SIZE_BYTES
    assert unaligned_mamba_spec.page_size_bytes == PAGE_SIZE_BYTES
    assert aligned_mamba_spec.page_size_bytes == PAGE_SIZE_BYTES

    GROUP_SIZE = 3

    # -- Build per-group layer info ----------------------------------------
    layer_idx = 0

    attn_layer_names = []
    for _ in range(GROUP_SIZE):
        attn_layer_names.append(f"model.layers.{layer_idx}.self_attn")
        layer_idx += 1

    mla_layer_names = []
    for _ in range(GROUP_SIZE):
        mla_layer_names.append(f"model.layers.{layer_idx}.self_attn")
        layer_idx += 1

    unaligned_mamba_layer_names = []
    for _ in range(GROUP_SIZE):
        unaligned_mamba_layer_names.append(f"model.layers.{layer_idx}.mamba_unpadded")
        layer_idx += 1

    aligned_mamba_layer_names = []
    for _ in range(GROUP_SIZE - 1):
        aligned_mamba_layer_names.append(f"model.layers.{layer_idx}.mamba_padded")
        layer_idx += 1

    layer_groups = [
        attn_layer_names,
        mla_layer_names,
        unaligned_mamba_layer_names,
        aligned_mamba_layer_names,
    ]

    kv_cache_tensors: list[KVCacheTensor] = []
    for i in range(GROUP_SIZE):
        shared_by: list[str] = []
        for group_layer_names in layer_groups:
            if len(group_layer_names) > i:
                shared_by.append(group_layer_names[i])
        kv_cache_tensors.append(
            KVCacheTensor(
                size=PAGE_SIZE_BYTES * NUM_BLOCKS,
                shared_by=shared_by,
            )
        )

    kv_cache_groups = [
        KVCacheGroupSpec(layer_names=attn_layer_names, kv_cache_spec=attn_spec),
        KVCacheGroupSpec(layer_names=mla_layer_names, kv_cache_spec=mla_spec),
        KVCacheGroupSpec(
            layer_names=unaligned_mamba_layer_names, kv_cache_spec=unaligned_mamba_spec
        ),
        KVCacheGroupSpec(
            layer_names=aligned_mamba_layer_names, kv_cache_spec=aligned_mamba_spec
        ),
    ]

    attn_groups = [
        [
            AttentionGroup(
                backend=backend_cls,
                layer_names=attn_layer_names,
                kv_cache_spec=attn_spec,
                kv_cache_group_id=0,
            ),
            AttentionGroup(
                backend=DeepseekV32IndexerBackend,
                layer_names=mla_layer_names,
                kv_cache_spec=mla_spec,
                kv_cache_group_id=1,
            ),
            AttentionGroup(
                backend=DeepseekV32IndexerBackend,  # unused for mamba
                layer_names=unaligned_mamba_layer_names,
                kv_cache_spec=unaligned_mamba_spec,
                kv_cache_group_id=2,
            ),
            AttentionGroup(
                backend=DeepseekV32IndexerBackend,  # unused for mamba
                layer_names=aligned_mamba_layer_names,
                kv_cache_spec=aligned_mamba_spec,
                kv_cache_group_id=3,
            ),
        ]
    ]

    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
    )

    kv_caches = _allocate_and_reshape_kv_caches(
        kv_cache_config,
        attn_groups,
        device=torch.device(f"{DEVICE_TYPE}:0"),
    )

    worker, spec = _make_worker(kv_cache_config)
    worker.register_kv_caches(kv_caches)

    canonical = spec.get_worker.call_args[0][0]
    assert isinstance(canonical, CanonicalKVCaches)

    # -- Expected block tensors ----------------------------------------------
    # All tensors have the same padded page size (PAGE_SIZE_BYTES).
    # Tensor 0: shared by attn[0], mla[0], mamba_unaligned[0], mamba_aligned[0]
    # Tensor 1: shared by attn[1], mla[1], mamba_unaligned[1], mamba_aligned[1]
    # Tensor 2: shared by attn[2], mla[2], mamba_unaligned[2]
    #           (mamba_aligned has only GROUP_SIZE-1 = 2 layers)
    expected_tensors = [
        (NUM_BLOCKS, PAGE_SIZE_BYTES),
        (NUM_BLOCKS, PAGE_SIZE_BYTES),
        (NUM_BLOCKS, PAGE_SIZE_BYTES),
    ]

    # -- Expected group data refs (order matches kv_cache_groups) -------------
    ref = CanonicalKVCacheRef
    expected_group_refs = [
        # attn group: layers attn[0..2] → tensors 0,1,2 with full page size
        [
            ref(tensor_idx=0, page_size_bytes=PAGE_SIZE_BYTES),
            ref(tensor_idx=1, page_size_bytes=PAGE_SIZE_BYTES),
            ref(tensor_idx=2, page_size_bytes=PAGE_SIZE_BYTES),
        ],
        # mla group: layers mla[0..2] → tensors 0,1,2 with full page size
        [
            ref(tensor_idx=0, page_size_bytes=PAGE_SIZE_BYTES),
            ref(tensor_idx=1, page_size_bytes=PAGE_SIZE_BYTES),
            ref(tensor_idx=2, page_size_bytes=PAGE_SIZE_BYTES),
        ],
        # unaligned mamba group: layers [0..2] → tensors 0,1,2 with unaligned page
        [
            ref(tensor_idx=0, page_size_bytes=unaligned_mamba_page_size),
            ref(tensor_idx=1, page_size_bytes=unaligned_mamba_page_size),
            ref(tensor_idx=2, page_size_bytes=unaligned_mamba_page_size),
        ],
        # aligned mamba group: layers [0..1] → tensors 0,1 with full page size
        [
            ref(tensor_idx=0, page_size_bytes=PAGE_SIZE_BYTES),
            ref(tensor_idx=1, page_size_bytes=PAGE_SIZE_BYTES),
        ],
    ]

    # Verify block tensors
    assert len(canonical.tensors) == len(expected_tensors)
    for block_tensor, (exp_num_blocks, exp_page_size) in zip(
        canonical.tensors, expected_tensors
    ):
        tensor = block_tensor.tensor
        assert tensor.dtype == torch.int8
        assert tensor.shape == (exp_num_blocks, exp_page_size)
        assert block_tensor.page_size_bytes == exp_page_size

    # Verify group data refs
    assert len(canonical.group_data_refs) == len(expected_group_refs)
    for actual_refs, exp_refs in zip(canonical.group_data_refs, expected_group_refs):
        assert len(actual_refs) == len(exp_refs)
        for actual, expected in zip(actual_refs, exp_refs):
            assert actual.tensor_idx == expected.tensor_idx
            assert actual.page_size_bytes == expected.page_size_bytes
            # Every layer gets a canonical mapping, certified or opaque
            assert actual.mapping is not None


@pytest.mark.parametrize("backend", ATTN_BACKENDS)
def test_register_kv_caches_uniform_type(backend):
    """Test register_kv_caches with UniformTypeKVCacheSpecs.

    Two attention layers use the same backend but different num_kv_heads,
    giving them different per-layer page sizes. Each has its own
    KVCacheTensor and are wrapped in a UniformTypeKVCacheSpecs group.
    Verifies that each layer gets the correct tensor_idx and
    page_size_bytes in its block data ref.
    """
    from vllm.v1.worker.utils import AttentionGroup

    backend_cls = AttentionBackendEnum[backend].get_class()

    layer_a = "model.layers.0.self_attn"
    layer_b = "model.layers.1.self_attn"
    spec_a = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
    )
    spec_b = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS * 2,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
    )
    assert spec_a.page_size_bytes != spec_b.page_size_bytes

    uniform_spec = UniformTypeKVCacheSpecs(
        block_size=BLOCK_SIZE,
        kv_cache_specs={layer_a: spec_a, layer_b: spec_b},
    )

    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[
            KVCacheTensor(
                size=spec_a.page_size_bytes * NUM_BLOCKS,
                shared_by=[layer_a],
            ),
            KVCacheTensor(
                size=spec_b.page_size_bytes * NUM_BLOCKS,
                shared_by=[layer_b],
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=[layer_a, layer_b],
                kv_cache_spec=uniform_spec,
            )
        ],
    )

    attn_groups = [
        [
            AttentionGroup(
                backend=backend_cls,
                layer_names=[layer_a],
                kv_cache_spec=spec_a,
                kv_cache_group_id=0,
            ),
            AttentionGroup(
                backend=backend_cls,
                layer_names=[layer_b],
                kv_cache_spec=spec_b,
                kv_cache_group_id=0,
            ),
        ]
    ]

    kv_caches = _allocate_and_reshape_kv_caches(
        kv_cache_config,
        attn_groups,
        device=torch.device(f"{DEVICE_TYPE}:0"),
    )

    worker, spec = _make_worker(kv_cache_config)
    worker.register_kv_caches(kv_caches)

    canonical = spec.get_worker.call_args[0][0]
    assert isinstance(canonical, CanonicalKVCaches)

    for block_tensor in canonical.tensors:
        assert block_tensor.tensor.dtype == torch.int8

    # Single group with refs from both layers
    assert len(canonical.group_data_refs) == 1
    group_refs = canonical.group_data_refs[0]
    assert len(group_refs) == 2

    assert len(canonical.tensors) == 2
    assert canonical.tensors[0].page_size_bytes == spec_a.page_size_bytes
    assert canonical.tensors[1].page_size_bytes == spec_b.page_size_bytes
    assert canonical.tensors[0].tensor.shape == (NUM_BLOCKS, spec_a.page_size_bytes)
    assert canonical.tensors[1].tensor.shape == (NUM_BLOCKS, spec_b.page_size_bytes)

    for ref, expected_tensor_idx, expected_spec in (
        (group_refs[0], 0, spec_a),
        (group_refs[1], 1, spec_b),
    ):
        assert ref.tensor_idx == expected_tensor_idx
        assert ref.page_size_bytes == expected_spec.page_size_bytes
        assert ref.mapping is not None

    # Only layer_a matches the model's total KV head count, so layer_b gets an
    # opaque mapping rather than a certified, parallelism-agnostic one
    assert group_refs[0].mapping.parallelism_agnostic
    assert not group_refs[1].mapping.parallelism_agnostic


# Compact geometry tests


class _GeometryRecordingHandler:
    def __init__(self):
        self.geometry = None

    def configure_compact_geometry(self, groups):
        if self.geometry is not None:
            raise RuntimeError("one-shot")
        self.geometry = groups


class CompactRecordingWorker(CPUOffloadingWorker):
    def __init__(self):
        self._compact_geometry = None
        self._store_handler = _GeometryRecordingHandler()
        self._load_handler = _GeometryRecordingHandler()
        self.configure_calls: list[tuple[CompactGroupGeometry | None, ...]] = []

    def configure_compact_geometry(self, groups):
        self.configure_calls.append(groups)
        super().configure_compact_geometry(groups)


def _mock_vllm_config():
    c = VllmConfig(
        device_config=DeviceConfig("cpu"),
        parallel_config=ParallelConfig(),
    )
    c.model_config = MagicMock()
    c.model_config.get_total_num_kv_heads.return_value = 4
    return c


def test_compact_geometry_types_and_one_shot():
    run = CopyRun(0, 0, 64, 1, 64, 64)
    mapping = CanonicalPageMapping(64, 64, (run,), 1, 0, True)
    with pytest.raises(ValueError, match="non-negative"):
        CompactLayerGeometry("l", mapping, 64, 64, -1, 0)
    layer = CompactLayerGeometry("l", mapping, 64, 64, 0, 0)
    with pytest.raises(ValueError, match="non-empty"):
        CompactGroupGeometry((), 64, 64, 64, True)
    group = CompactGroupGeometry((layer,), 64, 64, 64, True)
    worker = CompactRecordingWorker()
    worker.configure_compact_geometry((group,))
    assert worker._compact_geometry == (group,)
    with pytest.raises(RuntimeError, match="one-shot"):
        worker.configure_compact_geometry((group,))


def _pk_spec():
    return MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.uint8,
        page_size_padded=8192,
        compress_ratio=2,
        indexes_kv_by_block_stride=True,
    )


def _pk_cfg(spec, layers_by_offset):
    specs = {ln: spec for names in layers_by_offset.values() for ln in names}
    group = KVCacheGroupSpec(
        layer_names=list(specs),
        kv_cache_spec=UniformTypeKVCacheSpecs(block_size=64, kv_cache_specs=specs),
    )
    tensors = [
        KVCacheTensor(size=8192 * 4, shared_by=names, offset=off, block_stride=8192)
        for off, names in layers_by_offset.items()
    ]
    return KVCacheConfig(
        num_blocks=4, kv_cache_tensors=tensors, kv_cache_groups=[group]
    )


def _pk_views(config):
    size = max(t.block_stride * config.num_blocks for t in config.kv_cache_tensors)
    storage = torch.zeros(size, dtype=torch.uint8)
    return {
        ln: torch.as_strided(
            storage, (config.num_blocks, 64), (t.block_stride, 1), t.offset
        )
        for t in config.kv_cache_tensors
        for ln in t.shared_by
    }


def test_derive_compact_group_geometry_errors():
    run = CopyRun(0, 0, 64, 1, 64, 64)
    m = CanonicalPageMapping(64, 64, (run,), 1, 0, True)
    attn = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=64, dtype=torch.float16
    )
    pk = _pk_spec()
    ma = MambaSpec(block_size=16, shapes=((16, 64),), dtypes=(torch.float16,))
    c0 = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[
            KVCacheTensor(size=1024, shared_by=["attn.0"]),
            KVCacheTensor(size=4352, shared_by=["mamba.0"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["attn.0", "mamba.0"],
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=16, kv_cache_specs={"attn.0": attn, "mamba.0": ma}
                ),
            )
        ],
    )
    assert derive_compact_group_geometry(
        c0,
        {"attn.0": m},
        {
            "attn.0": torch.zeros(4, 256, dtype=torch.uint8),
            "mamba.0": torch.zeros(4, 4352, dtype=torch.uint8),
        },
        {"attn.0": False, "mamba.0": False},
    ) == (None,)
    c1 = _pk_cfg(pk, {0: ["pk.0", "pk.1"]})
    assert derive_compact_group_geometry(
        c1,
        {"pk.0": m},
        {"pk.0": torch.zeros(4, 8192, dtype=torch.uint8)},
        {"pk.0": True, "pk.1": True},
    ) == (None,)
    bad = torch.empty(4 * 8192 + 64, dtype=torch.uint8).as_strided(
        (4, 8192), (8192, 1), 64
    )
    with pytest.raises(ValueError, match="Packed storage_offset"):
        derive_compact_group_geometry(
            c1,
            {"pk.0": m},
            {**_pk_views(c1), "pk.0": bad},
            {"pk.0": True, "pk.1": True},
        )
    spk = MLAAttentionSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.uint8,
        page_size_padded=128,
        compress_ratio=2,
        indexes_kv_by_block_stride=True,
    )
    cd = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[
            KVCacheTensor(size=128 * 4, shared_by=["pk.0"], offset=0, block_stride=64)
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["pk.0"],
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=8, kv_cache_specs={"pk.0": spk}
                ),
            )
        ],
    )
    with pytest.raises(ValueError, match="exceeds block_stride"):
        derive_compact_group_geometry(
            cd,
            {"pk.0": CanonicalPageMapping(128, 128, (run,), 1, 0, True)},
            {"pk.0": torch.zeros(4, 128, dtype=torch.uint8)},
            {"pk.0": True},
        )
    ce = _pk_cfg(pk, {0: ["pk.0"], 32: ["pk.1"]})
    with pytest.raises(ValueError, match="Overlap"):
        derive_compact_group_geometry(
            ce,
            {"pk.0": m, "pk.1": m},
            _pk_views(ce),
            {"pk.0": True, "pk.1": True},
        )


def test_derive_compact_group_geometry_success():
    pk = _pk_spec()
    run = CopyRun(0, 0, 64, 1, 64, 64)
    m = CanonicalPageMapping(64, 64, (run,), 1, 0, True)
    c = _pk_cfg(pk, {0: ["pk.0"], 64: ["pk.1"]})
    g = derive_compact_group_geometry(
        c,
        {"pk.0": m, "pk.1": m},
        _pk_views(c),
        {"pk.0": True, "pk.1": True},
    )[0]
    assert g is not None and g.gpu_row_stride == 8192
    assert g.layers[0].gpu_offset_bytes == 0
    assert g.layers[1].gpu_offset_bytes == 64
    assert g.layers[0].canonical_offset == 0
    assert g.layers[1].canonical_offset == 64
    assert g.parallel_invariant


def test_register_kv_caches_compact_geometry():
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.core.kv_cache_utils import _get_kv_cache_config_packed
    from vllm.v1.worker.utils import AttentionGroup

    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
    )
    names0 = ["model.layers.0.self_attn", "model.layers.1.self_attn"]
    names1 = ["model.layers.2.self_attn", "model.layers.3.self_attn"]
    g0 = KVCacheGroupSpec(
        layer_names=names0,
        kv_cache_spec=UniformTypeKVCacheSpecs(
            block_size=BLOCK_SIZE, kv_cache_specs={ln: spec for ln in names0}
        ),
    )
    g1 = KVCacheGroupSpec(
        layer_names=names1,
        kv_cache_spec=UniformTypeKVCacheSpecs(
            block_size=BLOCK_SIZE, kv_cache_specs={ln: spec for ln in names1}
        ),
    )
    groups = [g0, g1]
    nb, pts = _get_kv_cache_config_packed(_mock_vllm_config(), groups, 8 * 1024 * 1024)
    bc = AttentionBackendEnum.CPU_ATTN.get_class()
    kcc = KVCacheConfig(num_blocks=nb, kv_cache_tensors=pts, kv_cache_groups=groups)
    ag = [
        [
            AttentionGroup(
                backend=bc, layer_names=[ln], kv_cache_spec=spec, kv_cache_group_id=gi
            )
            for ln in gg.layer_names
        ]
        for gi, gg in enumerate(groups)
    ]
    kv = _allocate_and_reshape_kv_caches(kcc, ag, device=torch.device("cpu"))
    rec = CompactRecordingWorker()
    sm = MagicMock(spec=OffloadingSpec)
    sm.replicated_layout = False
    sm.config = MagicMock()
    sm.config.parallel.rank = 0
    sm.config.parallel.world_size = 2
    sm.config.worker_kv_bytes_per_block = 1024
    sm.blocks_per_chunk = 1
    sm.extra_config = {"cpu_bytes_to_use": 10**9}
    sm.get_worker.return_value = rec
    # Enable compact layout so worker derives rank evidence and geometry.
    type(sm).compact_layout_requested = PropertyMock(return_value=True)
    type(sm).compact_page_size = PropertyMock(return_value=65536)
    type(sm).compact_storage_budget_bytes = PropertyMock(return_value=10**9)
    sm.get_worker.return_value = rec
    w = OffloadingConnectorWorker(
        spec=sm, vllm_config=_mock_vllm_config(), kv_cache_config=kcc
    )
    w.register_kv_caches(kv)
    assert isinstance(sm.get_worker.call_args[0][0], CanonicalKVCaches)
    assert len(rec.configure_calls) == 1
    geom = rec.configure_calls[0]
    assert len(geom) == len(groups)
    for gi, g in enumerate(geom):
        assert g is not None
        assert g.gpu_row_stride == pts[0].block_stride
        assert g.parallel_invariant
        for li, ln in enumerate(groups[gi].layer_names):
            ly = g.layers[li]
            kvt = next(t for t in pts if ln in t.shared_by)
            assert ly.layer_name == ln
            assert ly.gpu_offset_bytes == kvt.offset
            assert ly.local_page_size_bytes == spec.page_size_bytes
            assert ly.canonical_page_size_bytes == spec.page_size_bytes
            assert ly.canonical_offset == li * spec.page_size_bytes
            assert ly.mapping.parallelism_agnostic


def test_register_kv_caches_plugin_no_geometry():
    spec = MagicMock(spec=OffloadingSpec)
    spec.replicated_layout = False
    spec.config = MagicMock()
    spec.config.parallel.rank = 0
    # Ensure compact_layout_requested is False so compact geometry is
    # not derived for non-packed groups.
    type(spec).compact_layout_requested = PropertyMock(return_value=False)
    spec.get_worker.return_value = MagicMock(spec=OffloadingWorker)
    attn = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
    )
    cfg = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[
            KVCacheTensor(size=attn.page_size_bytes * NUM_BLOCKS, shared_by=["attn.0"])
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["attn.0"], kv_cache_spec=attn)],
    )
    OffloadingConnectorWorker(
        spec=spec, vllm_config=_mock_vllm_config(), kv_cache_config=cfg
    ).register_kv_caches(
        {
            "attn.0": torch.zeros(
                NUM_BLOCKS, attn.page_size_bytes, dtype=torch.uint8, device="cpu"
            )
        }
    )
    assert spec.get_worker.called
    assert isinstance(spec.get_worker.call_args[0][0], CanonicalKVCaches)
    assert not hasattr(spec.get_worker.return_value, "_compact_geometry")


def test_register_kv_caches_receipt_propagation():
    """Exactly one call to derive_canonical_mappings_with_receipt; its
    returned receipt reaches _compact_rank_evidence.receipt via the real
    registration path.  Does not copy production logic."""
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.canonical_mapping import (  # noqa: E501, I001
        CanonicalMappingReceipt,
        derive_canonical_mappings_with_receipt as _real_deriv,
    )
    from vllm.v1.attention.backends.registry import AttentionBackendEnum  # noqa: I001
    from vllm.v1.core.kv_cache_utils import _get_kv_cache_config_packed  # noqa: I001
    from vllm.v1.worker.utils import AttentionGroup  # noqa: I001

    # Reuse the existing packed compact fixture from
    # test_register_kv_caches_compact_geometry.
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
    )
    names = ["model.layers.0.self_attn"]
    g = KVCacheGroupSpec(
        layer_names=names,
        kv_cache_spec=UniformTypeKVCacheSpecs(
            block_size=BLOCK_SIZE, kv_cache_specs={ln: spec for ln in names}
        ),
    )
    groups = [g]
    nb, pts = _get_kv_cache_config_packed(_mock_vllm_config(), groups, 8 * 1024 * 1024)
    bc = AttentionBackendEnum.CPU_ATTN.get_class()
    kcc = KVCacheConfig(num_blocks=nb, kv_cache_tensors=pts, kv_cache_groups=groups)
    ag = [
        [
            AttentionGroup(
                backend=bc, layer_names=[ln], kv_cache_spec=spec, kv_cache_group_id=gi
            )
            for ln in gg.layer_names
        ]
        for gi, gg in enumerate(groups)
    ]
    kv = _allocate_and_reshape_kv_caches(kcc, ag, device=torch.device("cpu"))
    rec = CompactRecordingWorker()
    sm = MagicMock(spec=OffloadingSpec)
    sm.replicated_layout = False
    sm.config = MagicMock()
    sm.config.parallel.rank = 0
    sm.config.parallel.world_size = 1
    sm.config.worker_kv_bytes_per_block = 1024
    sm.blocks_per_chunk = 1
    sm.extra_config = {"cpu_bytes_to_use": 10**9}
    sm.get_worker.return_value = rec
    type(sm).compact_layout_requested = PropertyMock(return_value=True)
    type(sm).compact_page_size = PropertyMock(return_value=65536)
    type(sm).compact_storage_budget_bytes = PropertyMock(return_value=10**9)

    # Patch the exact symbol used by worker.py. The wrapper calls the real
    # owner and records the exact receipt object it returned.
    deriv_path = (
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker"
        ".derive_canonical_mappings_with_receipt"
    )
    returned_receipts = []

    def recording_derivation(*args, **kwargs):
        mappings, receipt = _real_deriv(*args, **kwargs)
        returned_receipts.append(receipt)
        return mappings, receipt

    with patch(deriv_path, side_effect=recording_derivation) as mock_deriv:
        w = OffloadingConnectorWorker(
            spec=sm, vllm_config=_mock_vllm_config(), kv_cache_config=kcc
        )
        w.register_kv_caches(kv)

        # Exactly one derivation call — no duplicate.
        assert mock_deriv.call_count == 1, (
            f"expected 1 call, got {mock_deriv.call_count}"
        )

    # Real geometry was constructed.
    assert len(rec.configure_calls) == 1
    geom = rec.configure_calls[0]
    assert geom is not None

    # The exact receipt returned by the owner reached compact rank evidence.
    assert w._compact_rank_evidence is not None
    rct = w._compact_rank_evidence.receipt
    assert returned_receipts == [rct]
    assert rct is returned_receipts[0]
    assert rct is not None
    assert isinstance(rct, CanonicalMappingReceipt)
    assert rct.layer_names == ("model.layers.0.self_attn",)
    assert rct.certified
    # The receipt contains the complete layer-major, rank-minor data.
    assert len(rct.per_rank) == 1  # one rank, one layer
    assert rct.per_rank[0].rank == 0
    assert rct.per_rank[0].layer_name == "model.layers.0.self_attn"
