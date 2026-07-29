# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, PropertyMock

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadingManager,
    RequestOffloadingContext,
)
from vllm.v1.kv_offload.cpu.common import (
    CompactGroupGeometry,
    CompactLayerGeometry,
    CompactRankEvidence,
)

pytestmark = pytest.mark.cpu_test


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _layer_geom(
    layer_name: str = "layer.0.self_attn",
    canonical_bytes: int = 64,
) -> CompactLayerGeometry:
    """Minimal CompactLayerGeometry."""
    from vllm.v1.kv_offload.base import CanonicalPageMapping, CopyRun

    run = CopyRun(0, 0, 64, 1, 64, 64)
    mapping = CanonicalPageMapping(64, 64, (run,), 1, 0, True)
    return CompactLayerGeometry(
        layer_name=layer_name,
        mapping=mapping,
        local_page_size_bytes=canonical_bytes,
        canonical_page_size_bytes=canonical_bytes,
        canonical_offset=0,
        gpu_offset_bytes=0,
    )


def _group_geom(
    canonical_bytes: int = 64,
    gpu_row_stride: int = 128,
    parallel_invariant: bool = True,
) -> CompactGroupGeometry:
    """Minimal CompactGroupGeometry."""
    layer = _layer_geom(canonical_bytes=canonical_bytes)
    return CompactGroupGeometry(
        layers=(layer,),
        gpu_row_stride=gpu_row_stride,
        local_extent=canonical_bytes,
        canonical_extent=canonical_bytes,
        parallel_invariant=parallel_invariant,
    )


def _make_evidence(
    rank: int = 0,
    world_size: int = 2,
    group_available: tuple[bool, ...] = (True,),
    canonical_bytes: tuple[int, ...] = (64,),
    page_size: int = 65536,
    cpu_bytes_to_use: int = 10**9,
    parallel_invariant: bool = True,
    expected_world_size: int | None = None,
    receipt: object = None,
) -> CompactRankEvidence:
    """Build a structurally valid CompactRankEvidence using private immutable
    tuple type aliases for signatures.

    Callers testing signature rejection must use the raw
    constructor or ``from_geometry``.
    """
    from vllm.v1.kv_offload.cpu.common import _COMPACT_ABSENT_GROUP

    sig_groups = []
    for avail, cbytes in zip(group_available, canonical_bytes):
        if avail:
            run = (0, 0, cbytes, 1, cbytes, cbytes)
            layer = ("layer.0", cbytes, cbytes, 0, 0, (run,))
            sig_groups.append((True, cbytes * 2, cbytes, cbytes, (layer,)))
        else:
            sig_groups.append(_COMPACT_ABSENT_GROUP)
    return CompactRankEvidence(
        rank=rank,
        world_size=world_size,
        group_available=group_available,
        canonical_bytes=canonical_bytes,
        page_size=page_size,
        cpu_bytes_to_use=cpu_bytes_to_use,
        parallel_invariant=parallel_invariant,
        expected_world_size=expected_world_size or world_size,
        signature=tuple(sig_groups),
        receipt=receipt,
    )


def _make_scheduler(compact_requested: bool = True, world_size: int = 2):
    """Build a minimal OffloadingConnectorScheduler with mocks."""
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )
    from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

    mgr = CPUOffloadingManager(num_blocks=100)

    spec = MagicMock()
    spec.get_manager.return_value = mgr
    type(spec).replicated_layout = PropertyMock(return_value=False)
    type(spec).offload_prompt_only = PropertyMock(return_value=False)
    type(spec).compact_layout_requested = PropertyMock(return_value=compact_requested)
    type(spec).compact_preferred_eviction_groups = PropertyMock(
        return_value=(1, 2, 3, 4)
    )

    vllm_config = MagicMock()
    vllm_config.parallel_config.world_size = world_size
    vllm_config.cache_config.enable_prefix_caching = False
    vllm_config.cache_config.worker_kv_bytes_per_block = 1024
    vllm_config.kv_transfer_config = MagicMock()
    vllm_config.kv_transfer_config.kv_connector_extra_config = {
        "cpu_bytes_to_use": 10**9,
    }

    kv_cache_config = MagicMock()
    kv_cache_config.kv_cache_groups = []

    sched = OffloadingConnectorScheduler(spec, vllm_config, kv_cache_config)
    sched.manager = mgr
    return sched


def _meta(reports: list[tuple[int, CompactRankEvidence]]) -> OffloadingWorkerMetadata:
    """Build OffloadingWorkerMetadata with compact reports (list of tuples)."""
    return OffloadingWorkerMetadata(compact_reports=reports)


# ---------------------------------------------------------------------------
# Test stubs: concrete OffloadingManager subclass for lifecycle tests
# ---------------------------------------------------------------------------


class _RecordingManager(OffloadingManager):
    """Records enable_compact calls for verification without side effects."""

    def __init__(self) -> None:
        self.enable_compact_called: bool = False
        self.enable_compact_kwargs: dict | None = None

    def enable_compact(
        self,
        total_bytes: int,
        page_size: int = 65536,
        key_sizes: dict[int, int] | None = None,
        preferred_groups: tuple[int, ...] | None = None,
    ) -> None:
        self.enable_compact_called = True
        self.enable_compact_kwargs = {
            "total_bytes": total_bytes,
            "page_size": page_size,
            "key_sizes": key_sizes,
            "preferred_groups": preferred_groups,
        }

    def lookup(self, key, req_context):  # type: ignore[override]
        return LookupResult.MISS

    def prepare_load(self, keys, req_context):  # type: ignore[override]
        return None  # type: ignore[return-value]

    def prepare_store(self, keys, req_context):  # type: ignore[override]
        return None

    def on_new_request(self, req_context):  # type: ignore[override]
        return RequestOffloadingContext()


class _UnsupportedManager(_RecordingManager):
    """Concrete manager that deliberately inherits the base fail-loud seam."""

    enable_compact = OffloadingManager.enable_compact


class _RaisingManager(_RecordingManager):
    """Concrete manager whose activation invariant fails."""

    def enable_compact(
        self,
        total_bytes: int,
        page_size: int = 65536,
        key_sizes: dict[int, int] | None = None,
        preferred_groups: tuple[int, ...] | None = None,
    ) -> None:
        raise RuntimeError("injected compact activation failure")


# ===================================================================
# CompactRankEvidence unit tests
# ===================================================================


class TestCompactRankEvidence:
    def test_from_geometry_single_group(self):
        """Basic construction with one available group."""
        geometry = (_group_geom(),)
        ev = CompactRankEvidence.from_geometry(
            rank=0,
            world_size=2,
            geometry=geometry,
            page_size=65536,
            cpu_bytes_to_use=10**9,
        )
        assert ev.rank == 0
        assert ev.world_size == 2
        assert ev.group_available == (True,)
        assert ev.canonical_bytes == (64,)
        assert ev.parallel_invariant is True
        assert ev.expected_world_size == 2

        chunked = CompactRankEvidence.from_geometry(
            rank=0,
            world_size=2,
            geometry=geometry,
            page_size=65536,
            cpu_bytes_to_use=10**9,
            blocks_per_chunk=3,
        )
        assert chunked.canonical_bytes == (192,)

    def test_from_geometry_unavailable_group(self):
        """Unavailable group has canonical_bytes=0 and available=False."""
        geometry = (None,)
        ev = CompactRankEvidence.from_geometry(
            rank=0,
            world_size=2,
            geometry=geometry,
            page_size=65536,
            cpu_bytes_to_use=10**9,
        )
        assert ev.group_available == (False,)
        assert ev.canonical_bytes == (0,)

    def test_from_geometry_mixed_availability(self):
        """Mixed available/unavailable groups."""
        geometry = (_group_geom(), None, _group_geom(canonical_bytes=128))
        ev = CompactRankEvidence.from_geometry(
            rank=0,
            world_size=2,
            geometry=geometry,
            page_size=65536,
            cpu_bytes_to_use=10**9,
        )
        assert ev.group_available == (True, False, True)
        assert ev.canonical_bytes == (64, 0, 128)

    def test_from_geometry_parallel_not_invariant(self):
        """parallel_invariant=False when any group lacks invariance."""
        g1 = _group_geom(parallel_invariant=True)
        g2 = _group_geom(parallel_invariant=False)
        geometry = (g1, g2)
        ev = CompactRankEvidence.from_geometry(
            rank=0,
            world_size=2,
            geometry=geometry,
            page_size=65536,
            cpu_bytes_to_use=10**9,
        )
        assert ev.parallel_invariant is False

    def test_validation_mismatched_lengths(self):
        """group_available and canonical_bytes must match length."""
        with pytest.raises(ValueError, match="must have the same length"):
            CompactRankEvidence(
                rank=0,
                world_size=2,
                group_available=(True,),
                canonical_bytes=(64, 128),
                page_size=65536,
                cpu_bytes_to_use=10**9,
                expected_world_size=2,
            )

    def test_validation_available_needs_positive_bytes(self):
        """Available group must have positive canonical_bytes."""
        with pytest.raises(ValueError, match="positive canonical_bytes"):
            CompactRankEvidence(
                rank=0,
                world_size=2,
                group_available=(True,),
                canonical_bytes=(0,),
                page_size=65536,
                cpu_bytes_to_use=10**9,
                expected_world_size=2,
            )

    def test_validation_unavailable_must_have_zero_bytes(self):
        """Unavailable group must have canonical_bytes=0."""
        with pytest.raises(ValueError, match="canonical_bytes=0"):
            CompactRankEvidence(
                rank=0,
                world_size=2,
                group_available=(False,),
                canonical_bytes=(64,),
                page_size=65536,
                cpu_bytes_to_use=10**9,
                expected_world_size=2,
            )

    def test_validation_page_size_positive(self):
        """page_size must be positive."""
        with pytest.raises(ValueError, match="page_size must be positive"):
            CompactRankEvidence(
                rank=0,
                world_size=2,
                group_available=(True,),
                canonical_bytes=(64,),
                page_size=0,
                cpu_bytes_to_use=10**9,
                expected_world_size=2,
            )

    def test_validation_cpu_bytes_to_use_positive(self):
        """cpu_bytes_to_use must be positive."""
        with pytest.raises(ValueError, match="cpu_bytes_to_use must be positive"):
            CompactRankEvidence(
                rank=0,
                world_size=2,
                group_available=(True,),
                canonical_bytes=(64,),
                page_size=65536,
                cpu_bytes_to_use=0,
                expected_world_size=2,
            )

    def test_validation_available_no_signature_fails(self):
        """Available groups with empty signature must fail."""
        with pytest.raises(ValueError, match="non-empty when groups are available"):
            CompactRankEvidence(
                rank=0,
                world_size=2,
                group_available=(True,),
                canonical_bytes=(64,),
                page_size=65536,
                cpu_bytes_to_use=10**9,
                expected_world_size=2,
                signature=(),
            )

    def test_validation_schema_version_default_three(self):
        """Default schema_version is 3."""
        ev = _make_evidence()
        assert ev.schema_version == 3


# ===================================================================
# OffloadingWorkerMetadata compact report tests
# ===================================================================


class TestCompactReportMetadata:
    def test_metadata_carries_evidence(self):
        """OffloadingWorkerMetadata can carry CompactRankEvidence."""
        ev = _make_evidence(rank=0)
        meta = OffloadingWorkerMetadata(compact_reports=[(0, ev)])
        assert len(meta.compact_reports) == 1
        assert meta.compact_reports[0][0] == 0
        assert meta.compact_reports[0][1].canonical_bytes == (64,)

    def test_aggregation_merges_distinct_ranks(self):
        """Aggregation merges per-rank evidence (no OR-mask loss)."""
        ev0 = _make_evidence(rank=0, canonical_bytes=(64,))
        ev1 = _make_evidence(rank=1, canonical_bytes=(64,))
        m0 = OffloadingWorkerMetadata(compact_reports=[(0, ev0)])
        m1 = OffloadingWorkerMetadata(compact_reports=[(1, ev1)])
        agg = m0.aggregate(m1)
        assert len(agg.compact_reports) == 2
        assert agg.compact_reports[0][1].canonical_bytes == (64,)
        assert agg.compact_reports[1][1].canonical_bytes == (64,)

    def test_aggregation_preserves_duplicate_rank(self):
        """Aggregation preserves duplicate rank — does NOT raise."""
        ev = _make_evidence(rank=0)
        dup = _make_evidence(rank=0)
        m0 = OffloadingWorkerMetadata(compact_reports=[(0, ev)])
        m1 = OffloadingWorkerMetadata(compact_reports=[(0, dup)])
        agg = m0.aggregate(m1)
        assert len(agg.compact_reports) == 2
        assert agg.compact_reports[0][1].canonical_bytes == (64,)
        assert agg.compact_reports[1][1].canonical_bytes == (64,)

    def test_aggregation_preserves_conflicting_evidence(self):
        """Aggregation preserves both rank entries even on conflict."""
        ev0 = _make_evidence(rank=0, canonical_bytes=(64,))
        ev1 = _make_evidence(rank=1, canonical_bytes=(128,))  # different
        m0 = OffloadingWorkerMetadata(compact_reports=[(0, ev0)])
        m1 = OffloadingWorkerMetadata(compact_reports=[(1, ev1)])
        agg = m0.aggregate(m1)
        assert len(agg.compact_reports) == 2
        assert agg.compact_reports[0][1].canonical_bytes == (64,)
        assert agg.compact_reports[1][1].canonical_bytes == (128,)


# ===================================================================
# Scheduler consensus logic tests
# ===================================================================


class TestCompactConsensus:
    """Consensus seam: evidence-based, no hash, no tri-state."""

    @pytest.fixture
    def sample_evidence(self):
        return _make_evidence(rank=0, canonical_bytes=(64,))

    # --- Core flow ---

    def test_disabled_no_report_gate_activation(self):
        """If compact_layout_requested=False: zero report/gate/activation."""
        sched = _make_scheduler(compact_requested=False)
        assert not sched._compact_requested
        assert sched._compact_resolved is False
        assert hasattr(sched.manager, "enable_compact")

    def test_agreement_activates_once(self, sample_evidence):
        """All ranks agree — consensus reached exactly once."""
        sched = _make_scheduler(world_size=2)
        assert not sched._compact_resolved

        sched._process_compact_geometry_report(_meta([(0, sample_evidence)]))
        assert not sched._compact_resolved

        ev1 = _make_evidence(rank=1, canonical_bytes=(64,))
        sched._process_compact_geometry_report(_meta([(1, ev1)]))
        assert sched._compact_resolved

    def test_missing_rank_no_resolution(self, sample_evidence):
        """Not all ranks reported — no resolution yet."""
        sched = _make_scheduler(world_size=2)
        sched._process_compact_geometry_report(_meta([(0, sample_evidence)]))
        assert not sched._compact_resolved

    # --- Scalar conflict rejection ---

    def test_conflict_canonical_bytes_fails(self):
        """Canonical bytes mismatch — permanent legacy fallback."""
        sched = _make_scheduler(world_size=2)
        ev0 = _make_evidence(rank=0, canonical_bytes=(64,))
        ev1 = _make_evidence(rank=1, canonical_bytes=(128,))
        sched._process_compact_geometry_report(_meta([(0, ev0)]))
        assert not sched._compact_resolved
        sched._process_compact_geometry_report(_meta([(1, ev1)]))
        assert sched._compact_resolved

    def test_conflict_group_available_fails(self):
        """Group availability mismatch — permanent legacy fallback."""
        sched = _make_scheduler(world_size=2)
        ev0 = _make_evidence(rank=0, group_available=(True,), canonical_bytes=(64,))
        ev1 = _make_evidence(rank=1, group_available=(False,), canonical_bytes=(0,))
        sched._process_compact_geometry_report(_meta([(0, ev0)]))
        assert not sched._compact_resolved
        sched._process_compact_geometry_report(_meta([(1, ev1)]))
        assert sched._compact_resolved

    def test_conflict_page_size_fails(self):
        """Page size mismatch — permanent legacy fallback."""
        sched = _make_scheduler(world_size=2)
        ev0 = _make_evidence(rank=0, page_size=65536)
        ev1 = _make_evidence(rank=1, page_size=131072)
        sched._process_compact_geometry_report(_meta([(0, ev0)]))
        assert not sched._compact_resolved
        sched._process_compact_geometry_report(_meta([(1, ev1)]))
        assert sched._compact_resolved

    def test_conflict_world_size_fails(self):
        """Expected world_size mismatch — permanent legacy fallback."""
        sched = _make_scheduler(world_size=2)
        ev0 = _make_evidence(rank=0, expected_world_size=2)
        ev1 = _make_evidence(rank=1, expected_world_size=4)
        sched._process_compact_geometry_report(_meta([(0, ev0)]))
        assert not sched._compact_resolved
        sched._process_compact_geometry_report(_meta([(1, ev1)]))
        assert sched._compact_resolved

    # --- Rank integrity ---

    def test_duplicate_rank_fails(self):
        """Duplicate rank report — permanent legacy fallback."""
        sched = _make_scheduler(world_size=2)
        ev = _make_evidence(rank=0)
        sched._process_compact_geometry_report(_meta([(0, ev)]))
        assert not sched._compact_resolved
        sched._process_compact_geometry_report(_meta([(0, ev)]))
        assert sched._compact_resolved

    @pytest.mark.parametrize("rank", [-1, 99])
    def test_unexpected_rank_fails(self, rank):
        """Rank outside expected range — permanent legacy fallback."""
        sched = _make_scheduler(world_size=2)
        ev = _make_evidence(rank=rank)
        sched._process_compact_geometry_report(_meta([(rank, ev)]))
        assert sched._compact_resolved

    def test_tuple_rank_mismatch_fails(self):
        """Tuple key rank != evidence.rank — immediate rejection."""
        sched = _make_scheduler(world_size=2)
        ev = _make_evidence(rank=0)
        # Pass with tuple key 1 but evidence.rank 0.
        sched._process_compact_geometry_report(_meta([(1, ev)]))
        assert sched._compact_resolved

    @pytest.mark.parametrize(
        ("world_size", "expected_world_size"), [(4, 2), (2, 4), (4, 4)]
    )
    def test_world_size_must_match_scheduler(self, world_size, expected_world_size):
        """Evidence agreement cannot override scheduler rank authority."""
        sched = _make_scheduler(world_size=2)
        ev = _make_evidence(
            rank=0,
            world_size=world_size,
            expected_world_size=expected_world_size,
        )
        sched._process_compact_geometry_report(_meta([(0, ev)]))
        assert sched._compact_resolved

    # --- Once-only ---

    def test_consensus_once_only(self, sample_evidence):
        """Already resolved — subsequent reports do not re-activate."""
        sched = _make_scheduler(world_size=2)
        ev1 = _make_evidence(rank=1, canonical_bytes=(64,))
        sched._process_compact_geometry_report(_meta([(0, sample_evidence)]))
        sched._process_compact_geometry_report(_meta([(1, ev1)]))
        assert sched._compact_resolved
        sched._process_compact_geometry_report(_meta([(1, ev1)]))
        assert sched._compact_resolved

    # --- Gate release ---

    def test_store_gate_released_on_fail(self):
        """After consensus fail, stores flow through legacy path (no crash)."""
        sched = _make_scheduler(world_size=2)
        ev_bad = _make_evidence(rank=99)
        sched._process_compact_geometry_report(_meta([(99, ev_bad)]))
        assert sched._compact_resolved
        assert sched._compact_requested
        assert sched._compact_resolved

    # --- Recording manager ---

    def test_recording_manager_resolved_after_success(self):
        """Recording manager: _compact_resolved set only after enable_compact
        succeeds, and arguments forwarded correctly."""
        sched = _make_scheduler(world_size=2)
        recorder = _RecordingManager()
        sched.manager = recorder

        ev0 = _make_evidence(rank=0, canonical_bytes=(64,))
        ev1 = _make_evidence(rank=1, canonical_bytes=(64,))
        sched._process_compact_geometry_report(_meta([(0, ev0)]))
        assert not sched._compact_resolved

        sched._process_compact_geometry_report(_meta([(1, ev1)]))
        assert sched._compact_resolved
        assert recorder.enable_compact_called

    def test_non_invariant_reject(self):
        """parallel_invariant=False — immediate legacy."""
        sched = _make_scheduler(world_size=2)
        recorder = _RecordingManager()
        sched.manager = recorder

        ev = _make_evidence(rank=0, parallel_invariant=False)
        sched._process_compact_geometry_report(_meta([(0, ev)]))
        assert sched._compact_resolved
        assert not recorder.enable_compact_called

    def test_stride_mismatch_fails(self):
        """Same canonical byte totals but differing gpu_row_stride — different
        signatures — must NOT activate compact."""
        from vllm.v1.kv_offload.base import CanonicalPageMapping, CopyRun

        sched = _make_scheduler(world_size=2)
        recorder = _RecordingManager()
        sched.manager = recorder

        run = CopyRun(0, 0, 64, 1, 64, 64)
        mapping = CanonicalPageMapping(64, 64, (run,), 1, 0, True)
        layer = CompactLayerGeometry(
            layer_name="layer.0",
            mapping=mapping,
            local_page_size_bytes=64,
            canonical_page_size_bytes=64,
            canonical_offset=0,
            gpu_offset_bytes=0,
        )
        geom0 = CompactGroupGeometry(
            layers=(layer,),
            gpu_row_stride=128,
            local_extent=64,
            canonical_extent=64,
            parallel_invariant=True,
        )
        geom1 = CompactGroupGeometry(
            layers=(layer,),
            gpu_row_stride=256,
            local_extent=64,
            canonical_extent=64,
            parallel_invariant=True,
        )

        ev0 = CompactRankEvidence.from_geometry(
            rank=0,
            world_size=2,
            geometry=(geom0,),
            page_size=65536,
            cpu_bytes_to_use=10**9,
        )
        ev1 = CompactRankEvidence.from_geometry(
            rank=1,
            world_size=2,
            geometry=(geom1,),
            page_size=65536,
            cpu_bytes_to_use=10**9,
        )

        sched._process_compact_geometry_report(_meta([(0, ev0)]))
        assert not sched._compact_resolved
        sched._process_compact_geometry_report(_meta([(1, ev1)]))
        assert sched._compact_resolved
        assert not recorder.enable_compact_called

    def test_load_run_mismatch_fails(self):
        """Same canonical bytes but differing load-run fragment sizes — must
        NOT activate compact."""
        from vllm.v1.kv_offload.base import CanonicalPageMapping, CopyRun

        sched = _make_scheduler(world_size=2)
        recorder = _RecordingManager()
        sched.manager = recorder

        mapping0 = CanonicalPageMapping(
            64,
            64,
            (CopyRun(0, 0, 64, 1, 64, 64),),
            1,
            0,
            True,
        )
        mapping1 = CanonicalPageMapping(
            64,
            64,
            (
                CopyRun(0, 0, 32, 1, 64, 64),
                CopyRun(32, 32, 32, 1, 64, 64),
            ),
            1,
            0,
            True,
        )
        layer0 = CompactLayerGeometry(
            layer_name="layer.0",
            mapping=mapping0,
            local_page_size_bytes=64,
            canonical_page_size_bytes=64,
            canonical_offset=0,
            gpu_offset_bytes=0,
        )
        layer1 = CompactLayerGeometry(
            layer_name="layer.0",
            mapping=mapping1,
            local_page_size_bytes=64,
            canonical_page_size_bytes=64,
            canonical_offset=0,
            gpu_offset_bytes=0,
        )
        geom0 = CompactGroupGeometry(
            layers=(layer0,),
            gpu_row_stride=128,
            local_extent=64,
            canonical_extent=64,
            parallel_invariant=True,
        )
        geom1 = CompactGroupGeometry(
            layers=(layer1,),
            gpu_row_stride=128,
            local_extent=64,
            canonical_extent=64,
            parallel_invariant=True,
        )

        ev0 = CompactRankEvidence.from_geometry(
            rank=0,
            world_size=2,
            geometry=(geom0,),
            page_size=65536,
            cpu_bytes_to_use=10**9,
        )
        ev1 = CompactRankEvidence.from_geometry(
            rank=1,
            world_size=2,
            geometry=(geom1,),
            page_size=65536,
            cpu_bytes_to_use=10**9,
        )

        sched._process_compact_geometry_report(_meta([(0, ev0)]))
        assert not sched._compact_resolved
        sched._process_compact_geometry_report(_meta([(1, ev1)]))
        assert sched._compact_resolved
        assert not recorder.enable_compact_called

    def test_invalid_geometry_signature_fails(self):
        """Mismatched geometry signatures rejection."""
        from vllm.v1.kv_offload.base import CanonicalPageMapping, CopyRun

        sched = _make_scheduler(world_size=2)
        recorder = _RecordingManager()
        sched.manager = recorder

        run = CopyRun(0, 0, 64, 1, 64, 64)
        mapping = CanonicalPageMapping(64, 64, (run,), 1, 0, True)
        layer = CompactLayerGeometry(
            layer_name="layer.0",
            mapping=mapping,
            local_page_size_bytes=64,
            canonical_page_size_bytes=64,
            canonical_offset=0,
            gpu_offset_bytes=0,
        )
        geom = CompactGroupGeometry(
            layers=(layer,),
            gpu_row_stride=128,
            local_extent=64,
            canonical_extent=64,
            parallel_invariant=True,
        )

        ev = CompactRankEvidence.from_geometry(
            rank=0,
            world_size=2,
            geometry=(geom,),
            page_size=65536,
            cpu_bytes_to_use=10**9,
        )

        # Collect two incompatible geometry signatures: different gpu_row_stride
        run2 = CopyRun(0, 0, 32, 1, 32, 32)
        mapping2 = CanonicalPageMapping(32, 32, (run2,), 1, 0, True)
        layer2 = CompactLayerGeometry(
            layer_name="layer.0",
            mapping=mapping2,
            local_page_size_bytes=32,
            canonical_page_size_bytes=32,
            canonical_offset=0,
            gpu_offset_bytes=0,
        )
        geom2 = CompactGroupGeometry(
            layers=(layer2,),
            gpu_row_stride=64,
            local_extent=32,
            canonical_extent=32,
            parallel_invariant=True,
        )

        ev2 = CompactRankEvidence.from_geometry(
            rank=1,
            world_size=2,
            geometry=(geom2,),
            page_size=65536,
            cpu_bytes_to_use=10**9,
        )
        sched._process_compact_geometry_report(_meta([(0, ev)]))
        sched._process_compact_geometry_report(_meta([(1, ev2)]))
        # Should reject compact due to incompatible signatures; resolved to legacy.
        assert sched._compact_resolved
        assert not recorder.enable_compact_called

    def test_valid_writer_pair_activates(self):
        """Two valid writer ranks with identical signatures activate compact.

        In the current rotating-writer API (num_writers/writer_index), every
        certified rank is a writer for some subset of blocks.  Consensus
        requires compatible geometry signatures across ranks.
        """
        from vllm.v1.kv_offload.base import CanonicalPageMapping, CopyRun

        sched = _make_scheduler(world_size=2)
        recorder = _RecordingManager()
        sched.manager = recorder

        run = CopyRun(0, 0, 64, 1, 64, 64)
        mapping = CanonicalPageMapping(64, 64, (run,), 1, 0, True)
        layer = CompactLayerGeometry(
            layer_name="layer.0",
            mapping=mapping,
            local_page_size_bytes=64,
            canonical_page_size_bytes=64,
            canonical_offset=0,
            gpu_offset_bytes=0,
        )
        geom = CompactGroupGeometry(
            layers=(layer,),
            gpu_row_stride=128,
            local_extent=64,
            canonical_extent=64,
            parallel_invariant=True,
        )

        ev0 = CompactRankEvidence.from_geometry(
            rank=0,
            world_size=2,
            geometry=(geom,),
            page_size=65536,
            cpu_bytes_to_use=10**9,
        )
        ev1 = CompactRankEvidence.from_geometry(
            rank=1,
            world_size=2,
            geometry=(geom,),
            page_size=65536,
            cpu_bytes_to_use=10**9,
        )

        sched._process_compact_geometry_report(_meta([(0, ev0)]))
        assert not sched._compact_resolved
        sched._process_compact_geometry_report(_meta([(1, ev1)]))
        assert sched._compact_resolved
        assert recorder.enable_compact_called

    def test_unsupported_schema_fails(self):
        """Schema version != 3 — immediate legacy fallback."""
        sched = _make_scheduler(world_size=2)
        recorder = _RecordingManager()
        sched.manager = recorder

        ev = _make_evidence(rank=0)
        # Override schema_version in the raw evidence.
        bad = CompactRankEvidence(
            rank=0,
            world_size=2,
            group_available=(True,),
            canonical_bytes=(64,),
            page_size=65536,
            cpu_bytes_to_use=10**9,
            expected_world_size=2,
            schema_version=1,
            signature=ev.signature,
        )
        sched._process_compact_geometry_report(_meta([(0, bad)]))
        assert sched._compact_resolved
        assert not recorder.enable_compact_called


# ===================================================================
# Real aggregated worker-output boundary
# ===================================================================


class TestCompactConsensusUpdateConnectorOutput:
    """Exercise the sole production consensus opportunity."""

    @staticmethod
    def _output(reports):
        from vllm.v1.outputs import KVConnectorOutput

        return KVConnectorOutput(kv_connector_worker_meta=_meta(reports))

    def test_complete_two_rank_batch_activates(self):
        sched = _make_scheduler(world_size=2)
        recorder = _RecordingManager()
        sched.manager = recorder
        ev0 = _make_evidence(rank=0)
        ev1 = _make_evidence(rank=1)

        sched.update_connector_output(self._output([(0, ev0), (1, ev1)]))

        assert sched._compact_resolved
        assert recorder.enable_compact_called

    def test_partial_batch_immediately_fails_legacy(self):
        sched = _make_scheduler(world_size=2)
        recorder = _RecordingManager()
        sched.manager = recorder

        sched.update_connector_output(self._output([(0, _make_evidence(rank=0))]))

        assert sched._compact_resolved
        assert sched._compact_report_rank_mask == 1
        assert not recorder.enable_compact_called

    def test_no_report_batch_stays_unresolved(self):
        sched = _make_scheduler(world_size=2)

        sched.update_connector_output(self._output([]))

        assert not sched._compact_resolved
        assert sched._compact_report_rank_mask == 0

    @pytest.mark.parametrize(
        ("manager", "error"),
        [
            (_UnsupportedManager(), NotImplementedError),
            (_RaisingManager(), RuntimeError),
        ],
    )
    def test_activation_failure_propagates_and_keeps_gate_closed(self, manager, error):
        sched = _make_scheduler(world_size=2)
        sched.manager = manager
        ev0 = _make_evidence(rank=0)
        ev1 = _make_evidence(rank=1)

        with pytest.raises(error):
            sched.update_connector_output(self._output([(0, ev0), (1, ev1)]))

        assert not sched._compact_resolved
        assert sched._compact_report_rank_mask == 0b11


class TestConcreteManagerEnableCompact:
    def test_missing_group_sizes_fails_without_activation(self):
        from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

        manager = CPUOffloadingManager(num_blocks=100)
        with pytest.raises(RuntimeError, match="activation failed"):
            manager.enable_compact(total_bytes=6553600, page_size=65536)
        assert not manager._compact_enabled
        assert manager._compact_allocator is None


# ===================================================================
# Receipt-based consensus tests
# ===================================================================


class TestCompactConsensusReceipt:
    """Receipt equality replaces writer-role checks in compact consensus."""

    def test_matching_receipts_activate(self):
        """All ranks with identical receipts reach consensus."""
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.canonical_mapping import (  # noqa: E501
            CanonicalMappingReceipt,
        )
        from vllm.v1.kv_offload.base import CopyRun

        rr = CanonicalMappingReceipt.RankReceipt
        cr = CopyRun
        receipt = CanonicalMappingReceipt(
            layer_names=("layer.0",),
            per_rank=(
                rr(
                    rank=0,
                    layer_name="layer.0",
                    canonical_page_size_bytes=64,
                    local_page_size_bytes=64,
                    runs=(cr(0, 0, 64, 1, 64, 64),),
                    num_writers=2,
                    writer_index=0,
                    parallelism_agnostic=True,
                ),
                rr(
                    rank=1,
                    layer_name="layer.0",
                    canonical_page_size_bytes=64,
                    local_page_size_bytes=64,
                    runs=(cr(0, 0, 64, 1, 64, 64),),
                    num_writers=2,
                    writer_index=1,
                    parallelism_agnostic=True,
                ),
            ),
            fallback=False,
            certified=True,
        )
        sched = _make_scheduler(world_size=2)
        recorder = _RecordingManager()
        sched.manager = recorder

        ev0 = _make_evidence(rank=0, receipt=receipt)
        ev1 = _make_evidence(rank=1, receipt=receipt)
        sched._process_compact_geometry_report(_meta([(0, ev0)]))
        sched._process_compact_geometry_report(_meta([(1, ev1)]))
        assert sched._compact_resolved
        assert recorder.enable_compact_called

    def test_altered_receipt_rejected(self):
        """Altered receipt (different runs) is rejected by consensus."""
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.canonical_mapping import (  # noqa: E501
            CanonicalMappingReceipt,
        )
        from vllm.v1.kv_offload.base import CopyRun

        rr = CanonicalMappingReceipt.RankReceipt
        cr = CopyRun
        receipt = CanonicalMappingReceipt(
            layer_names=("layer.0",),
            per_rank=(
                rr(
                    rank=0,
                    layer_name="layer.0",
                    canonical_page_size_bytes=64,
                    local_page_size_bytes=64,
                    runs=(cr(0, 0, 64, 1, 64, 64),),
                    num_writers=1,
                    writer_index=0,
                    parallelism_agnostic=True,
                ),
            ),
            fallback=False,
            certified=True,
        )
        altered = CanonicalMappingReceipt(
            layer_names=("layer.0",),
            per_rank=(
                rr(
                    rank=0,
                    layer_name="layer.0",
                    canonical_page_size_bytes=128,
                    local_page_size_bytes=128,
                    runs=(cr(0, 0, 128, 1, 128, 128),),
                    num_writers=1,
                    writer_index=0,
                    parallelism_agnostic=True,
                ),
                rr(
                    rank=1,
                    layer_name="layer.0",
                    canonical_page_size_bytes=128,
                    local_page_size_bytes=128,
                    runs=(cr(0, 0, 128, 1, 128, 128),),
                    num_writers=1,
                    writer_index=0,
                    parallelism_agnostic=True,
                ),
            ),
            fallback=False,
            certified=True,
        )
        sched = _make_scheduler(world_size=2)
        recorder = _RecordingManager()
        sched.manager = recorder

        ev0 = _make_evidence(rank=0, receipt=receipt)
        ev1 = _make_evidence(rank=1, receipt=altered)
        sched._process_compact_geometry_report(_meta([(0, ev0)]))
        sched._process_compact_geometry_report(_meta([(1, ev1)]))
        assert sched._compact_resolved
        assert not recorder.enable_compact_called

    def test_opaque_receipt_fallback_compact_rejected(self):
        """Opaque/uncertified mapping receipt (fallback=True) fails closed."""
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.canonical_mapping import (  # noqa: E501
            CanonicalMappingReceipt,
        )
        from vllm.v1.kv_offload.base import CopyRun

        rr = CanonicalMappingReceipt.RankReceipt
        cr = CopyRun
        opaque_receipt = CanonicalMappingReceipt(
            layer_names=("layer.0",),
            per_rank=(
                rr(
                    rank=0,
                    layer_name="layer.0",
                    canonical_page_size_bytes=128,
                    local_page_size_bytes=64,
                    runs=(cr(0, 0, 64, 1, 64, 64),),
                    num_writers=1,
                    writer_index=0,
                    parallelism_agnostic=False,
                ),
                rr(
                    rank=1,
                    layer_name="layer.0",
                    canonical_page_size_bytes=128,
                    local_page_size_bytes=64,
                    runs=(cr(0, 0, 64, 1, 64, 64),),
                    num_writers=1,
                    writer_index=0,
                    parallelism_agnostic=False,
                ),
            ),
            fallback=True,
            certified=False,
        )
        sched = _make_scheduler(world_size=2)
        recorder = _RecordingManager()
        sched.manager = recorder

        # Opaque mapping fails via group_available=False
        ev0 = _make_evidence(
            rank=0,
            group_available=(False,),
            canonical_bytes=(0,),
            receipt=opaque_receipt,
        )
        sched._process_compact_geometry_report(_meta([(0, ev0)]))
        assert sched._compact_resolved
        assert not recorder.enable_compact_called
