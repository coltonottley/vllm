# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU-less causal tests for stage-local TP dedupe (``stage_replicated_layout``)
and the scheduler-side minimum-stage chunk bound (rule A).

The geometry under test mirrors the Prime full-GLM FP8 engine at TP4 x PP2:
PP0 stage page 1,738,752 B/block, PP1 stage page 1,763,840 B/block (the MTP
draft layer's rows included), block size 64, one prefix-cacheable MLA group
per stage, and ``pipeline_stages`` stamped into every config by the engine
core. Aligned stage rows: PP0 1,740,800, PP1 1,765,376, sum 3,506,176.
"""

from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.v1.kv_connector.unit.offloading_connector.utils import (
    make_kv_cache_group,
)
from vllm.config import KVTransferConfig, ParallelConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    TransferJob,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    build_offloading_config,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupRole,
    KVCacheStage,
    KVCacheTensor,
    MLAAttentionSpec,
)
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    ReqContext,
    StoreTrigger,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

# Prime full-GLM FP8, TP4 x PP2, block 64, one MLA group per stage.
PP0_PAGE = 1_738_752
PP1_PAGE = 1_763_840
GIB = 1024**3
# round_up(page, 4096): the per-stage region row strides.
PP0_ROW = 1_740_800
PP1_ROW = 1_765_376
STAGE_ROWS_SUM = PP0_ROW + PP1_ROW

DEFAULT_STAGES = (
    KVCacheStage(bytes_per_block=PP0_PAGE),
    KVCacheStage(bytes_per_block=PP1_PAGE),
)


def _make_vllm_config(
    *,
    extra_config: dict[str, Any] | None = None,
    tensor_parallel_size: int = 4,
    pipeline_parallel_size: int = 2,
    prefill_context_parallel_size: int = 1,
    decode_context_parallel_size: int = 1,
    data_parallel_size: int = 1,
    rank: int = 0,
    use_mla: bool = True,
    distributed_executor_backend: Any = "mp",
    nnodes: int = 1,
) -> VllmConfig:
    config = MagicMock()
    config.num_hash_lookahead_tokens = 0
    config.cache_config.block_size = 64
    config.cache_config.enable_prefix_caching = True
    config.cache_config.prefix_match_unit = None
    config.cache_config.cache_dtype = torch.float16
    config.cache_config.prefix_cache_retention_interval = None
    config.cache_config.kv_cache_layout = None
    config.model_config.model = "test-glm"
    config.model_config.max_model_len = 1_000_000
    config.model_config.use_mla = use_mla
    world_size = (
        tensor_parallel_size * pipeline_parallel_size * prefill_context_parallel_size
    )
    with patch.object(current_platform, "device_count", return_value=world_size):
        config.parallel_config = ParallelConfig(
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            prefill_context_parallel_size=prefill_context_parallel_size,
            decode_context_parallel_size=decode_context_parallel_size,
            data_parallel_size=data_parallel_size,
        )
    # Assign the executor backend post-construction: constructing with a
    # non-default backend re-validates (and would require, e.g., ray to be
    # installed), which is not what these gate rows exercise.
    config.parallel_config.distributed_executor_backend = distributed_executor_backend
    config.parallel_config.rank = rank
    config.parallel_config.nnodes = nnodes
    config.kv_events_config = None
    config.kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config=dict(extra_config or {}),
    )
    return cast(VllmConfig, config)


def _make_stage_kv_cache_config(
    stage_page: int,
    *,
    kv_tp_replicas: int = 4,
    num_blocks: int = 1,
    pipeline_stages: tuple[KVCacheStage, ...] | None = None,
    hisparse_host_num_blocks: int | None = None,
    hisparse_indexer_role: bool = False,
) -> KVCacheConfig:
    """One prefix-cacheable MLA group; the tensor run carries the stage page.

    The tensors branch of build_offloading_config derives
    worker_kv_bytes_per_block = kv_cache_tensors[0].size // num_blocks, which
    is how the live engine derives each stage's page.
    """
    layer_name = "model.layers.0.self_attn"
    spec = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        max_tp_shards=1,
        head_size=512,
        dtype=torch.float32,
    )
    group_kwargs: dict[str, Any] = {}
    if hisparse_indexer_role:
        group_kwargs["role"] = KVCacheGroupRole.HISPARSE_INDEXER
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=stage_page * num_blocks,
                layers=[layer_name],
                layer_stride=spec.page_size_bytes * num_blocks,
                block_stride=spec.page_size_bytes,
            )
        ],
        kv_cache_groups=[make_kv_cache_group([layer_name], spec, **group_kwargs)],
        hisparse_host_num_blocks=hisparse_host_num_blocks,
        kv_tp_replicas=kv_tp_replicas,
        pipeline_stages=(
            DEFAULT_STAGES if pipeline_stages is None else pipeline_stages
        ),
    )


def _stage_extra_config(cpu_bytes_to_use: int) -> dict[str, Any]:
    return {
        "stage_replicated_layout": True,
        "cpu_bytes_to_use": cpu_bytes_to_use,
        "eviction_policy": "arc",
    }


# ---------------------------------------------------------------------------
# G1: the eligibility gate matrix (fail-closed, naming the failed gate).
# ---------------------------------------------------------------------------


def test_g1_stage_flag_eligible_at_tp4_pp2():
    config = _make_vllm_config(extra_config=_stage_extra_config(GIB))
    kv_cache_config = _make_stage_kv_cache_config(PP0_PAGE)

    offloading_config = build_offloading_config(config, kv_cache_config)

    assert offloading_config.stage_replicated_layout
    assert offloading_config.stage_kv_bytes_per_block == (PP0_PAGE, PP1_PAGE)
    assert offloading_config.parallel.pp_rank == 0
    assert offloading_config.parallel.tp_rank_in_group == 0
    # Rule A is scheduler-only; a role-blind (worker-shaped) build keeps 0.
    assert offloading_config.scheduler_max_stage_slot_bytes == 0


def test_g1_stage_flag_scheduler_role_also_populates_rule_a_bound():
    config = _make_vllm_config(extra_config=_stage_extra_config(GIB))
    kv_cache_config = _make_stage_kv_cache_config(PP0_PAGE)

    offloading_config = build_offloading_config(
        config, kv_cache_config, role=KVConnectorRole.SCHEDULER
    )

    assert offloading_config.stage_replicated_layout
    # Harmless under the flag (the stage sizing branch wins), but populated.
    assert offloading_config.scheduler_max_stage_slot_bytes == PP1_PAGE


def _default_stage_kv() -> KVCacheConfig:
    return _make_stage_kv_cache_config(PP0_PAGE)


def _single_stage_kv() -> KVCacheConfig:
    """pipeline_stages matching pp == 1: one stage only."""
    return _make_stage_kv_cache_config(
        PP0_PAGE, pipeline_stages=(KVCacheStage(bytes_per_block=PP0_PAGE),)
    )


@pytest.mark.parametrize(
    ("config_factory", "kv_factory", "gate_name"),
    [
        pytest.param(
            lambda: _make_vllm_config(
                extra_config=_stage_extra_config(GIB), use_mla=False
            ),
            _default_stage_kv,
            "model must use MLA",
            id="non-mla",
        ),
        pytest.param(
            lambda: _make_vllm_config(
                extra_config=_stage_extra_config(GIB),
                tensor_parallel_size=1,
                pipeline_parallel_size=2,
            ),
            _default_stage_kv,
            "tensor_parallel_size must be > 1",
            id="tp1",
        ),
        pytest.param(
            lambda: _make_vllm_config(
                extra_config=_stage_extra_config(GIB),
                tensor_parallel_size=4,
                pipeline_parallel_size=1,
            ),
            _single_stage_kv,
            "pipeline_parallel_size must be > 1",
            id="pp1",
        ),
        pytest.param(
            lambda: _make_vllm_config(extra_config=_stage_extra_config(GIB)),
            lambda: _make_stage_kv_cache_config(PP0_PAGE, kv_tp_replicas=2),
            "per-stage KV must be TP-replicated",
            id="kv-tp-replicas-below-tp",
        ),
        pytest.param(
            lambda: _make_vllm_config(extra_config=_stage_extra_config(GIB)),
            lambda: _make_stage_kv_cache_config(PP0_PAGE, num_blocks=0),
            "worker_kv_bytes_per_block must be > 0",
            id="zero-worker-bytes",
        ),
        pytest.param(
            lambda: _make_vllm_config(
                extra_config=_stage_extra_config(GIB),
                data_parallel_size=2,
            ),
            _default_stage_kv,
            "data_parallel_size must be 1",
            id="dp2",
        ),
        pytest.param(
            lambda: _make_vllm_config(
                extra_config=_stage_extra_config(GIB),
                prefill_context_parallel_size=2,
            ),
            _default_stage_kv,
            "prefill_context_parallel_size must be 1",
            id="pcp2",
        ),
        pytest.param(
            lambda: _make_vllm_config(
                extra_config=_stage_extra_config(GIB),
                decode_context_parallel_size=2,
            ),
            _default_stage_kv,
            "decode_context_parallel_size must be 1",
            id="dcp2",
        ),
        pytest.param(
            lambda: _make_vllm_config(
                extra_config=_stage_extra_config(GIB),
                decode_context_parallel_size=2,
            ),
            _default_stage_kv,
            "decode_context_parallel_size must be 1",
            id="dcp2",
        ),
        pytest.param(
            lambda: _make_vllm_config(
                extra_config=_stage_extra_config(GIB),
                prefill_context_parallel_size=2,
            ),
            _default_stage_kv,
            "world_size must equal pipeline_parallel_size * tensor_parallel_size",
            id="pcp2-also-breaks-world-identity",
        ),
        pytest.param(
            lambda: _make_vllm_config(
                extra_config=_stage_extra_config(GIB),
                distributed_executor_backend="ray",
            ),
            _default_stage_kv,
            "distributed_executor_backend must be 'mp'",
            id="ray",
        ),
        pytest.param(
            lambda: _make_vllm_config(extra_config=_stage_extra_config(GIB), nnodes=2),
            _default_stage_kv,
            "nnodes_within_dp must be 1",
            id="multi-node",
        ),
        pytest.param(
            lambda: _make_vllm_config(extra_config=_stage_extra_config(GIB)),
            lambda: _make_stage_kv_cache_config(
                PP0_PAGE,
                hisparse_host_num_blocks=4,
                hisparse_indexer_role=True,
            ),
            "hisparse_host_num_blocks must be None",
            id="hisparse",
        ),
        pytest.param(
            lambda: _make_vllm_config(extra_config=_stage_extra_config(GIB)),
            lambda: _make_stage_kv_cache_config(PP0_PAGE, pipeline_stages=()),
            "pipeline_stages must cover every pipeline stage",
            id="stages-missing",
        ),
        pytest.param(
            lambda: _make_vllm_config(extra_config=_stage_extra_config(GIB)),
            lambda: _make_stage_kv_cache_config(
                PP0_PAGE,
                pipeline_stages=(
                    KVCacheStage(bytes_per_block=PP0_PAGE),
                    KVCacheStage(bytes_per_block=0),
                ),
            ),
            "every pipeline stage must have bytes_per_block > 0",
            id="stage-bytes-zero",
        ),
    ],
)
def test_g1_stage_flag_violations_fail_closed(
    config_factory, kv_factory, gate_name: str
):
    config = config_factory()
    kv_cache_config = kv_factory()

    with pytest.raises(ValueError, match="ineligible") as exc_info:
        build_offloading_config(config, kv_cache_config)
    assert gate_name in str(exc_info.value)


def test_g1_stage_flag_rejects_canonical_layout():
    extra = _stage_extra_config(GIB)
    extra["canonical_layout"] = True
    config = _make_vllm_config(extra_config=extra)
    kv_cache_config = _make_stage_kv_cache_config(PP0_PAGE)

    with pytest.raises(ValueError, match="canonical_layout"):
        build_offloading_config(config, kv_cache_config)


def test_g1_stage_flag_rejects_inconsistent_stage_geometry():
    """A PP1-shaped tensor page at pp_rank 0 must refuse, not diverge."""
    config = _make_vllm_config(extra_config=_stage_extra_config(GIB), rank=0)
    kv_cache_config = _make_stage_kv_cache_config(PP1_PAGE)

    with pytest.raises(ValueError, match="inconsistent stage geometry"):
        build_offloading_config(config, kv_cache_config)


def test_g1_stage_flag_rejects_tiering_spec():
    config = _make_vllm_config(extra_config=_stage_extra_config(GIB))
    kv_cache_config = _make_stage_kv_cache_config(PP0_PAGE)
    offloading_config = build_offloading_config(config, kv_cache_config)

    with pytest.raises(ValueError, match="TieringOffloadingSpec"):
        TieringOffloadingSpec(offloading_config)


def test_g1_flag_absent_keeps_legacy_behavior(monkeypatch):
    """No flag: stage fields stay at their defaults and sizing is legacy."""
    # Pin the general (non-shared-region) sizing branch: this host may or may
    # not report a CUDA platform, and the legacy expectation below is the
    # general-branch number.
    monkeypatch.setattr(CPUOffloadingSpec, "_uses_shared_region", lambda self: False)
    config = _make_vllm_config(
        extra_config={"cpu_bytes_to_use": 8 * GIB, "eviction_policy": "arc"}
    )
    kv_cache_config = _make_stage_kv_cache_config(PP0_PAGE)

    offloading_config = build_offloading_config(config, kv_cache_config)

    assert not offloading_config.stage_replicated_layout
    assert offloading_config.stage_kv_bytes_per_block == ()
    assert offloading_config.scheduler_max_stage_slot_bytes == 0
    assert not offloading_config.replicated_layout
    # Legacy general-branch sizing: 8 GiB over round_up(PP0_PAGE * 8, 4096).
    spec = CPUOffloadingSpec(offloading_config)
    assert spec.num_chunks == 617
    assert spec.kv_bytes_per_chunk == PP0_PAGE * 8
    assert spec.cpu_page_size_per_worker == PP0_PAGE


# ---------------------------------------------------------------------------
# G2: equal stage-flag sizing across roles (the scheduler/PP0/PP1 divergence
# class designed out). 32 GiB tier.
# ---------------------------------------------------------------------------


def _build_stage_spec(
    role: KVConnectorRole | None,
    rank: int,
    stage_page: int,
    cpu_bytes_to_use: int,
) -> CPUOffloadingSpec:
    config = _make_vllm_config(
        extra_config=_stage_extra_config(cpu_bytes_to_use), rank=rank
    )
    kv_cache_config = _make_stage_kv_cache_config(stage_page)
    offloading_config = build_offloading_config(config, kv_cache_config, role=role)
    return CPUOffloadingSpec(offloading_config)


def test_g2_equal_stage_flag_sizing_across_roles(monkeypatch):
    # The spec-level stage flag requires a shared-region platform; pin it so
    # the stage sizing branch runs regardless of the host's platform.
    monkeypatch.setattr(CPUOffloadingSpec, "_uses_shared_region", lambda self: True)
    cpu_bytes = 32 * GIB
    scheduler_spec = _build_stage_spec(
        KVConnectorRole.SCHEDULER, 0, PP0_PAGE, cpu_bytes
    )
    pp0_spec = _build_stage_spec(KVConnectorRole.WORKER, 0, PP0_PAGE, cpu_bytes)
    pp1_spec = _build_stage_spec(KVConnectorRole.WORKER, 4, PP1_PAGE, cpu_bytes)

    specs = (scheduler_spec, pp0_spec, pp1_spec)
    # Corrected arithmetic: aligned rows 1,740,800 + 1,765,376 = 3,506,176;
    # 32 GiB // 3,506,176 = 9,799 chunks on every role.
    assert [spec.num_chunks for spec in specs] == [9_799, 9_799, 9_799]
    assert [spec.kv_bytes_per_chunk for spec in specs] == [
        STAGE_ROWS_SUM,
        STAGE_ROWS_SUM,
        STAGE_ROWS_SUM,
    ]
    # Each worker keeps its own stage page.
    assert pp0_spec.cpu_page_size_per_worker == PP0_PAGE
    assert pp1_spec.cpu_page_size_per_worker == PP1_PAGE
    # Honest aggregate reporting on all roles.
    for spec in specs:
        assert spec.cpu_info.cpu_num_chunks == 9_799
        assert spec.cpu_info.cpu_kv_cache_size_bytes == 9_799 * STAGE_ROWS_SUM
    # Region row math: the worker's own stage page, page-aligned.
    from vllm.utils.math_utils import round_up

    assert round_up(pp0_spec.cpu_page_size_per_worker, 4096) == PP0_ROW
    assert round_up(pp1_spec.cpu_page_size_per_worker, 4096) == PP1_ROW
    # Tier tokens at 64 tokens per chunk.
    assert 9_799 * 64 == 627_136


def test_g2_stage_flag_pp0_worker_page_mismatch_fails_closed():
    """A PP1 page presented at rank 0 (pp_rank 0) refuses to build."""
    config = _make_vllm_config(extra_config=_stage_extra_config(32 * GIB), rank=0)
    kv_cache_config = _make_stage_kv_cache_config(PP1_PAGE)

    with pytest.raises(ValueError, match="inconsistent stage geometry"):
        build_offloading_config(config, kv_cache_config, role=KVConnectorRole.WORKER)


# ---------------------------------------------------------------------------
# G3: stage identities, writer election, and the acker path.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rank", "expected"),
    [
        (0, (0, 0)),
        (1, (0, 1)),
        (2, (0, 2)),
        (3, (0, 3)),
        (4, (1, 0)),
        (5, (1, 1)),
        (6, (1, 2)),
        (7, (1, 3)),
    ],
)
def test_g3_stage_identities_per_rank(rank: int, expected: tuple[int, int]):
    config = _make_vllm_config(extra_config=_stage_extra_config(GIB), rank=rank)
    stage_page = PP0_PAGE if rank < 4 else PP1_PAGE
    kv_cache_config = _make_stage_kv_cache_config(stage_page)

    offloading_config = build_offloading_config(config, kv_cache_config)

    assert (
        offloading_config.parallel.pp_rank,
        offloading_config.parallel.tp_rank_in_group,
    ) == expected


def _make_stage_worker_spec(rank: int) -> CPUOffloadingSpec:
    from tests.v1.kv_offload.conftest import make_offloading_config

    return CPUOffloadingSpec(
        make_offloading_config(
            cpu_bytes_to_use=1 << 16,
            rank=rank,
            world_size=8,
            tp_size=4,
            pp_size=2,
            pp_rank=rank // 4,
            tp_rank_in_group=rank % 4,
            stage_replicated_layout=True,
            stage_kv_bytes_per_block=(16, 16),
            extra_config={"eviction_policy": "arc"},
        )
    )


@pytest.mark.parametrize(
    ("rank", "is_writer"),
    [
        (0, True),
        (1, False),
        (2, False),
        (3, False),
        (4, True),
        (5, False),
        (6, False),
        (7, False),
    ],
)
def test_g3_store_writer_election(rank: int, is_writer: bool, monkeypatch):
    # The spec-level stage flag requires a shared-region platform; the
    # election itself only reads the flag, so emulate the GPU platform.
    monkeypatch.setattr(CPUOffloadingSpec, "_uses_shared_region", lambda self: True)
    spec = _make_stage_worker_spec(rank)
    assert spec.stage_replicated_layout

    worker = OffloadingConnectorWorker(spec, MagicMock(), MagicMock())
    assert worker._is_store_writer is is_writer


def _store_job_metadata(job_id: int = 7) -> OffloadingConnectorMetadata:
    return OffloadingConnectorMetadata(
        load_jobs={},
        store_jobs={
            job_id: TransferJob(
                req_id="req-0",
                src_spec=GPULoadStoreSpec(
                    block_ids=[1], group_sizes=[1], block_indices=[0]
                ),
                dst_spec=GPULoadStoreSpec(
                    block_ids=[0], group_sizes=[1], block_indices=[0]
                ),
            )
        },
    )


def test_g3_acker_marks_completed_without_submitting(monkeypatch):
    monkeypatch.setattr(CPUOffloadingSpec, "_uses_shared_region", lambda self: True)
    spec = _make_stage_worker_spec(rank=1)
    worker = OffloadingConnectorWorker(spec, MagicMock(), MagicMock())
    assert not worker._is_store_writer

    worker.prepare_store_kv(_store_job_metadata())

    # The acker never queues a store: it completes the job for settlement
    # (pending_count still waits for world_size acks per job).
    assert worker._unsubmitted_store_jobs == []
    meta = worker.build_connector_worker_meta()
    assert meta is not None
    assert meta.completed_jobs == {7: 1}
    assert worker.worker is None


def test_g3_elected_writer_queues_the_store(monkeypatch):
    monkeypatch.setattr(CPUOffloadingSpec, "_uses_shared_region", lambda self: True)
    spec = _make_stage_worker_spec(rank=0)
    worker = OffloadingConnectorWorker(spec, MagicMock(), MagicMock())
    assert worker._is_store_writer

    worker.prepare_store_kv(_store_job_metadata())

    assert len(worker._unsubmitted_store_jobs) == 1
    assert worker._unsubmitted_store_jobs[0][0] == 7
    meta = worker.build_connector_worker_meta()
    assert meta is None


# ---------------------------------------------------------------------------
# G4: rule A — the scheduler-side minimum-stage chunk bound, and its no-op
# at pp == 1.
# ---------------------------------------------------------------------------


def _build_rule_a_spec(
    role: KVConnectorRole | None,
    cpu_bytes_to_use: int,
    stage_page: int = PP0_PAGE,
) -> CPUOffloadingSpec:
    config = _make_vllm_config(
        extra_config={"cpu_bytes_to_use": cpu_bytes_to_use, "eviction_policy": "arc"}
    )
    kv_cache_config = _make_stage_kv_cache_config(stage_page)
    offloading_config = build_offloading_config(config, kv_cache_config, role=role)
    return CPUOffloadingSpec(offloading_config)


def test_g4_scheduler_sizes_at_min_stage_chunk_count(monkeypatch):
    """8 GiB, flag off: the scheduler pool is the min over stages (608),
    while workers keep their own stage geometry."""
    # Pin the general sizing branch (the live-branch numbers are covered
    # below; both branches bound the scheduler at the min over stages).
    monkeypatch.setattr(CPUOffloadingSpec, "_uses_shared_region", lambda self: False)
    cpu_bytes = 8 * GIB
    scheduler_spec = _build_rule_a_spec(KVConnectorRole.SCHEDULER, cpu_bytes)
    pp0_spec = _build_rule_a_spec(KVConnectorRole.WORKER, cpu_bytes, PP0_PAGE)
    pp1_spec = _build_rule_a_spec(KVConnectorRole.WORKER, cpu_bytes, PP1_PAGE)

    # General branch (CPU test platform): 8 GiB // round_up(page * 8, 4096).
    assert scheduler_spec.num_chunks == 608
    assert pp0_spec.num_chunks == 617
    assert pp1_spec.num_chunks == 608
    # The bound: every assignable id fits every stage's region.
    assert scheduler_spec.num_chunks == min(pp0_spec.num_chunks, pp1_spec.num_chunks)


def test_g4_scheduler_bound_matches_live_branch_numbers(monkeypatch):
    """Under the shared-region platform branch (the live engine's), the
    canary-logged legacy worker numbers are 616/608 and rule A holds 608."""
    monkeypatch.setattr(CPUOffloadingSpec, "_uses_shared_region", lambda self: True)
    cpu_bytes = 8 * GIB
    scheduler_spec = _build_rule_a_spec(KVConnectorRole.SCHEDULER, cpu_bytes)
    pp0_spec = _build_rule_a_spec(KVConnectorRole.WORKER, cpu_bytes, PP0_PAGE)
    pp1_spec = _build_rule_a_spec(KVConnectorRole.WORKER, cpu_bytes, PP1_PAGE)

    assert scheduler_spec.num_chunks == 608
    assert pp0_spec.num_chunks == 616
    assert pp1_spec.num_chunks == 608


def test_g4_rule_a_is_no_op_at_pp1():
    config = _make_vllm_config(
        extra_config={"cpu_bytes_to_use": 8 * GIB},
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
    )
    kv_cache_config = _make_stage_kv_cache_config(
        PP0_PAGE, pipeline_stages=(KVCacheStage(bytes_per_block=PP0_PAGE),)
    )

    scheduler_config = build_offloading_config(
        config, kv_cache_config, role=KVConnectorRole.SCHEDULER
    )
    worker_config = build_offloading_config(config, kv_cache_config)

    assert scheduler_config.scheduler_max_stage_slot_bytes == 0
    assert CPUOffloadingSpec(scheduler_config).num_chunks == (
        CPUOffloadingSpec(worker_config).num_chunks
    )


def test_g4_legacy_scheduler_without_rule_a_is_unbounded(monkeypatch):
    """Role-blind builds keep the legacy (unbounded) scheduler pool: the
    documented PP2 hazard rule A removes."""
    monkeypatch.setattr(CPUOffloadingSpec, "_uses_shared_region", lambda self: False)
    spec = _build_rule_a_spec(None, 8 * GIB)
    assert spec.num_chunks == 617


# ---------------------------------------------------------------------------
# C7-shaped: flag-off traffic simulation — a store storm past the min-stage
# bound must never assign an id beyond any stage's region.
# ---------------------------------------------------------------------------


def _fresh_keys(count: int, salt: int) -> list:
    return [
        make_offload_key(f"c7-{salt}-{index}".encode(), 0) for index in range(count)
    ]


def _req_context(req_id: str) -> ReqContext:
    return ReqContext(req_id=req_id, kv_transfer_params=None)


def test_c7_flag_off_store_storm_never_exceeds_min_worker_capacity():
    """Flag off, rule A on: 620 distinct chunks offered at 8 GiB (past the
    609-distinct-chunk crash threshold) assign only ids < 608, the minimum
    per-stage region row count."""
    cpu_bytes = 8 * GIB
    scheduler_spec = _build_rule_a_spec(KVConnectorRole.SCHEDULER, cpu_bytes)
    pp1_spec = _build_rule_a_spec(KVConnectorRole.WORKER, cpu_bytes, PP1_PAGE)
    min_worker_rows = pp1_spec.num_chunks
    assert min_worker_rows == 608

    manager = scheduler_spec.get_manager()
    ctx = _req_context("c7-storm")

    first = manager.prepare_store(
        _fresh_keys(620, salt=0), ctx, trigger=StoreTrigger.ON_COMPUTE
    )
    assert first is not None
    assert len(first.keys_to_store) == 608
    assert len(first.keys_to_retry) == 12
    assert isinstance(first.store_spec, CPULoadStoreSpec)
    assigned = list(first.store_spec.chunk_ids)
    assert max(assigned) < min_worker_rows
    assert min(assigned) == 0
    assert len(set(assigned)) == 608

    manager.complete_store(first.keys_to_store, ctx, trigger=StoreTrigger.ON_COMPUTE)

    # A second storm round re-uses evicted slots; ids stay in range.
    second = manager.prepare_store(
        _fresh_keys(20, salt=1), ctx, trigger=StoreTrigger.ON_COMPUTE
    )
    assert second is not None
    assert len(second.keys_to_store) == 20
    assert isinstance(second.store_spec, CPULoadStoreSpec)
    assert max(second.store_spec.chunk_ids) < min_worker_rows


def test_c7_legacy_pool_would_assign_out_of_range_ids(monkeypatch):
    """The contrast arm: without rule A the scheduler pool is PP0-shaped
    (617), so assigned ids reach 616 — beyond every PP1 region (608 rows).
    This is the machinery-proven OOB class rule A removes."""
    monkeypatch.setattr(CPUOffloadingSpec, "_uses_shared_region", lambda self: False)
    spec = _build_rule_a_spec(None, 8 * GIB)
    assert spec.num_chunks == 617

    manager = spec.get_manager()
    ctx = _req_context("c7-legacy")

    output = manager.prepare_store(
        _fresh_keys(620, salt=0), ctx, trigger=StoreTrigger.ON_COMPUTE
    )
    assert output is not None
    assert len(output.keys_to_store) == 617
    assert isinstance(output.store_spec, CPULoadStoreSpec)
    assert max(output.store_spec.chunk_ids) == 616
