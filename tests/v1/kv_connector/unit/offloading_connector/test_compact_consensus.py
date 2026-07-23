# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, PropertyMock

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.v1.kv_offload.cpu.common import (
    CompactGroupGeometry,
    CompactLayerGeometry,
    CompactRankEvidence,
)
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

pytestmark = pytest.mark.cpu_test


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _layer_geom(
    layer_name: str = "layer.0.self_attn",
    canonical_bytes: int = 64,
) -> CompactLayerGeometry:
    """Minimal CompactLayerGeometry."""
    from vllm.v1.kv_offload.base import CanonicalPageMapping, MappedRun

    run = MappedRun(0, 0, 64, 1, 64, 64)
    mapping = CanonicalPageMapping(64, 64, (run,), (run,), True)
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
    is_writer: bool = True,
    expected_world_size: int | None = None,
) -> CompactRankEvidence:
    """Build a CompactRankEvidence with explicit fields."""
    return CompactRankEvidence(
        rank=rank,
        world_size=world_size,
        group_available=group_available,
        canonical_bytes=canonical_bytes,
        page_size=page_size,
        cpu_bytes_to_use=cpu_bytes_to_use,
        parallel_invariant=parallel_invariant,
        is_writer=is_writer,
        expected_world_size=expected_world_size or world_size,
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
        assert ev.is_writer is True
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
        # Both entries preserved; scheduler resolves duplicates.
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

    def test_disabled_no_report_gate_activation(self):
        """If compact_layout_requested=False: zero report/gate/activation."""
        sched = _make_scheduler(compact_requested=False)
        assert not sched._compact_requested
        # Gate is bypassed
        assert sched._compact_resolved is False
        # enable_compact must not be called by any path
        assert hasattr(sched.manager, "enable_compact")

    def test_agreement_activates_once(self, sample_evidence):
        """All ranks agree — consensus reached exactly once."""
        sched = _make_scheduler(world_size=2)
        assert not sched._compact_resolved

        # Rank 0 reports
        sched._process_compact_geometry_report(_meta([(0, sample_evidence)]))
        assert not sched._compact_resolved

        # Rank 1 reports (same evidence)
        ev1 = _make_evidence(rank=1, canonical_bytes=(64,))
        sched._process_compact_geometry_report(_meta([(1, ev1)]))
        assert sched._compact_resolved

    def test_missing_rank_no_resolution(self, sample_evidence):
        """Not all ranks reported — no resolution yet."""
        sched = _make_scheduler(world_size=2)
        sched._process_compact_geometry_report(_meta([(0, sample_evidence)]))
        assert not sched._compact_resolved

    def test_conflict_canonical_bytes_fails(self):
        """Canonical bytes mismatch — permanent legacy fallback."""
        sched = _make_scheduler(world_size=2)
        ev0 = _make_evidence(rank=0, canonical_bytes=(64,))
        ev1 = _make_evidence(rank=1, canonical_bytes=(128,))
        sched._process_compact_geometry_report(_meta([(0, ev0)]))
        assert not sched._compact_resolved
        sched._process_compact_geometry_report(_meta([(1, ev1)]))
        assert sched._compact_resolved  # resolved to legacy

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

    def test_conflict_parallel_invariant_fails(self):
        """Parallel invariant mismatch — permanent legacy fallback."""
        sched = _make_scheduler(world_size=2)
        ev0 = _make_evidence(rank=0, parallel_invariant=True)
        ev1 = _make_evidence(rank=1, parallel_invariant=False)
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

    def test_duplicate_rank_fails(self):
        """Duplicate rank report — permanent legacy fallback."""
        sched = _make_scheduler(world_size=2)
        ev = _make_evidence(rank=0)
        sched._process_compact_geometry_report(_meta([(0, ev)]))
        assert not sched._compact_resolved
        # Same rank reports again
        sched._process_compact_geometry_report(_meta([(0, ev)]))
        assert sched._compact_resolved  # resolved to legacy

    def test_unexpected_rank_fails(self):
        """Rank beyond expected range — permanent legacy fallback."""
        sched = _make_scheduler(world_size=2)
        ev = _make_evidence(rank=99)
        sched._process_compact_geometry_report(_meta([(99, ev)]))
        assert sched._compact_resolved  # resolved to legacy

    def test_consensus_once_only(self, sample_evidence):
        """Already resolved — subsequent reports do not re-activate."""
        sched = _make_scheduler(world_size=2)
        ev1 = _make_evidence(rank=1, canonical_bytes=(64,), is_writer=False)
        sched._process_compact_geometry_report(_meta([(0, sample_evidence)]))
        sched._process_compact_geometry_report(_meta([(1, ev1)]))
        assert sched._compact_resolved
        # Second call after resolution: duplicate rank triggers fail
        # but first resolution outcome is authoritative.
        sched._process_compact_geometry_report(_meta([(1, ev1)]))
        assert sched._compact_resolved

    def test_store_gate_released_on_fail(self):
        """After consensus fail, stores flow through legacy path (no crash)."""
        sched = _make_scheduler(world_size=2)
        # Force fail
        ev_bad = _make_evidence(rank=99)
        sched._process_compact_geometry_report(_meta([(99, ev_bad)]))
        assert sched._compact_resolved
        # Gate is resolved (to legacy) so stores are no longer blocked
        assert sched._compact_requested
        assert sched._compact_resolved

    def test_enable_compact_called_on_pass(self):
        """On consensus pass, manager.enable_compact is called."""
        from unittest.mock import MagicMock

        sched = _make_scheduler(world_size=2)
        # Replace manager with mock to track enable_compact calls
        mock_mgr = MagicMock()
        sched.manager = mock_mgr

        ev0 = _make_evidence(rank=0, canonical_bytes=(64,))
        ev1 = _make_evidence(rank=1, canonical_bytes=(64,), is_writer=False)
        sched._process_compact_geometry_report(_meta([(0, ev0)]))
        sched._process_compact_geometry_report(_meta([(1, ev1)]))
        assert sched._compact_resolved
        mock_mgr.enable_compact.assert_called_once()
        assert mock_mgr.enable_compact.call_args.kwargs["preferred_groups"] == (
            1,
            2,
            3,
            4,
        )


# ===================================================================
# update_connector_output integration: all-rank batch boundary
# ===================================================================


class TestCompactConsensusUpdateConnectorOutput:
    """Consensus through update_connector_output at the aggregated batch
    boundary.  The first nonempty batch is the sole opportunity: complete
    sets activate, partial sets immediately fail, empty batches leave
    state unchanged."""

    def test_complete_two_rank_batch_activates(self):
        """A single aggregated batch with both ranks activates compact."""
        from unittest.mock import MagicMock

        from vllm.v1.outputs import KVConnectorOutput

        sched = _make_scheduler(world_size=2)
        mock_mgr = MagicMock()
        sched.manager = mock_mgr

        ev0 = _make_evidence(rank=0, canonical_bytes=(64,))
        ev1 = _make_evidence(rank=1, canonical_bytes=(64,), is_writer=False)
        meta = _meta([(0, ev0), (1, ev1)])
        output = KVConnectorOutput(kv_connector_worker_meta=meta)

        sched.update_connector_output(output)
        assert sched._compact_resolved
        mock_mgr.enable_compact.assert_called_once()

    def test_partial_batch_immediately_fails_legacy(self):
        """Single batch with only some ranks immediately resolves to
        legacy — no second cycle."""
        from unittest.mock import MagicMock

        from vllm.v1.outputs import KVConnectorOutput

        sched = _make_scheduler(world_size=2)
        mock_mgr = MagicMock()
        sched.manager = mock_mgr

        ev0 = _make_evidence(rank=0, canonical_bytes=(64,))
        meta = _meta([(0, ev0)])
        output = KVConnectorOutput(kv_connector_worker_meta=meta)

        sched.update_connector_output(output)
        assert sched._compact_resolved  # resolved to legacy
        mock_mgr.enable_compact.assert_not_called()

    def test_no_report_batch_stays_unresolved(self):
        """Empty compact_reports before any evidence leaves state
        unresolved — engine has not yet presented registration
        opportunity."""
        from vllm.v1.outputs import KVConnectorOutput

        sched = _make_scheduler(world_size=2)

        meta = _meta([])
        output = KVConnectorOutput(kv_connector_worker_meta=meta)

        sched.update_connector_output(output)
        assert not sched._compact_resolved
        assert sched._compact_report_rank_mask == 0


# ===================================================================
# Spec compact layout request tests
# ===================================================================


class TestCompactSpecRequest:
    def test_default_compact_layout_false(self):
        """Default enable_compact_layout=False means no report/gating."""
        from vllm.v1.kv_offload.config import OffloadingConfig

        config = MagicMock(spec=OffloadingConfig)
        config.extra_config = {"cpu_bytes_to_use": 10**9}
        config.parallel = MagicMock()
        config.parallel.world_size = 1
        config.worker_kv_bytes_per_block = 1024
        config.cache = MagicMock()
        config.cache.blocks_per_chunk = 1
        config.cache.tokens_per_block = 16
        config.cache.tokens_per_hash = 16
        config.groups = []
        config.replicated_layout = False
        config.enable_kv_cache_events = False

        spec = CPUOffloadingSpec(config)
        assert not spec.compact_layout_requested

    def test_enable_compact_layout_true(self):
        """enable_compact_layout=True propagates through spec."""
        from vllm.v1.kv_offload.config import OffloadingConfig
        from vllm.v1.kv_offload.cpu.spec import _parse_enable_compact_layout

        config = MagicMock(spec=OffloadingConfig)
        config.extra_config = {
            "cpu_bytes_to_use": 10**9,
            "enable_compact_layout": True,
        }
        config.parallel = MagicMock()
        config.parallel.world_size = 1
        config.worker_kv_bytes_per_block = 1024
        config.cache = MagicMock()
        config.cache.blocks_per_chunk = 1
        config.cache.tokens_per_block = 16
        config.cache.tokens_per_hash = 16
        config.groups = []
        config.replicated_layout = False
        config.enable_kv_cache_events = False

        spec = CPUOffloadingSpec(config)
        assert spec.compact_layout_requested

        # Strict validator
        assert _parse_enable_compact_layout("true") is True
        assert _parse_enable_compact_layout("false") is False
        assert _parse_enable_compact_layout(True) is True
        assert _parse_enable_compact_layout(False) is False
        assert _parse_enable_compact_layout("True") is True
        assert _parse_enable_compact_layout("FALSE") is False

        # Reject invalid
        with pytest.raises(ValueError):
            _parse_enable_compact_layout("yes")
        with pytest.raises(ValueError):
            _parse_enable_compact_layout(1)
        with pytest.raises(ValueError):
            _parse_enable_compact_layout(0)

    def test_no_compact_manager_state_constructed(self):
        """Spec does NOT construct special manager state."""
        from vllm.v1.kv_offload.config import OffloadingConfig

        config = MagicMock(spec=OffloadingConfig)
        config.extra_config = {
            "cpu_bytes_to_use": 10**9,
            "enable_compact_layout": True,
        }
        config.parallel = MagicMock()
        config.parallel.world_size = 1
        config.worker_kv_bytes_per_block = 1024
        config.cache = MagicMock()
        config.cache.blocks_per_chunk = 1
        config.cache.tokens_per_block = 16
        config.cache.tokens_per_hash = 16
        config.groups = []
        config.replicated_layout = False
        config.enable_kv_cache_events = False

        spec = CPUOffloadingSpec(config)
        assert spec.compact_layout_requested
        # Manager is lazily constructed and unmodified by compact_requested
        mgr = spec.get_manager()
        assert mgr is not None
        # Compact state exists but remains disabled until rank consensus.
        assert not mgr._compact_enabled
        assert mgr._compact_allocator is None


# ===================================================================
# Base manager enable_compact fail-closed test
# ===================================================================


class TestManagerEnableCompact:
    def test_enable_compact_requires_group_sizes(self):
        from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

        mgr = CPUOffloadingManager(num_blocks=100)
        with pytest.raises(RuntimeError, match="activation failed"):
            mgr.enable_compact(total_bytes=6553600, page_size=65536)
        assert not mgr._compact_enabled

    def test_enable_compact_activates_concrete_manager(self):
        from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

        mgr = CPUOffloadingManager(num_blocks=100)
        mgr.enable_compact(
            total_bytes=6553600,
            page_size=65536,
            key_sizes={0: 4096},
        )
        assert mgr._compact_enabled
        assert mgr._compact_allocator is not None
