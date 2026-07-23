# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CompactCPUAddressSpan, CompactCPUAddress, CompactCPULoadStoreSpec."""

import pytest

from vllm.v1.kv_offload.cpu.common import (
    CompactCPUAddress,
    CompactCPUAddressSpan,
    CompactCPULoadStoreSpec,
)


class TestCompactCPUAddressSpan:
    def test_valid_construction(self):
        span = CompactCPUAddressSpan(
            byte_offset=0, logical_length=4096, allocated_length=4096
        )
        assert span.byte_offset == 0
        assert span.logical_length == 4096
        assert span.allocated_length == 4096

    def test_negative_offset(self):
        with pytest.raises(ValueError, match="byte_offset must be non-negative"):
            CompactCPUAddressSpan(
                byte_offset=-1, logical_length=4096, allocated_length=4096
            )

    def test_zero_logical_length(self):
        with pytest.raises(ValueError, match="logical_length must be positive"):
            CompactCPUAddressSpan(byte_offset=0, logical_length=0, allocated_length=0)

    def test_allocated_less_than_logical(self):
        with pytest.raises(ValueError, match="allocated_length must cover"):
            CompactCPUAddressSpan(
                byte_offset=0, logical_length=4096, allocated_length=2048
            )

    def test_immutability(self):
        span = CompactCPUAddressSpan(
            byte_offset=0, logical_length=4096, allocated_length=4096
        )
        assert span.__dataclass_fields__  # is a dataclass
        with pytest.raises(AttributeError, match="cannot assign to field"):
            span.byte_offset = 100

    def test_allocated_can_exceed_logical(self):
        span = CompactCPUAddressSpan(
            byte_offset=0, logical_length=100, allocated_length=4096
        )
        assert span.allocated_length == 4096


class TestCompactCPUAddress:
    def test_minimal(self):
        addr = CompactCPUAddress(
            byte_offset=0, logical_length=4096, allocated_length=4096
        )
        assert addr.byte_offset == 0
        assert addr.logical_length == 4096
        assert addr.allocated_length == 4096
        assert addr.group_idx == 0
        assert addr.spans == ()

    def test_with_group_idx(self):
        addr = CompactCPUAddress(
            byte_offset=8192, logical_length=4096, allocated_length=4096, group_idx=3
        )
        assert addr.group_idx == 3
        assert addr.byte_offset == 8192

    def test_with_spans(self):
        span = CompactCPUAddressSpan(
            byte_offset=0, logical_length=4096, allocated_length=4096
        )
        addr = CompactCPUAddress(
            byte_offset=0, logical_length=4096, allocated_length=4096, spans=(span,)
        )
        assert addr.spans == (span,)
        assert addr.physical_spans == (span,)

    def test_physical_spans_fallback(self):
        addr = CompactCPUAddress(
            byte_offset=0, logical_length=4096, allocated_length=4096
        )
        spans = addr.physical_spans
        assert len(spans) == 1
        assert spans[0].byte_offset == 0
        assert spans[0].logical_length == 4096
        assert spans[0].allocated_length == 4096

    def test_negative_group_idx(self):
        with pytest.raises(ValueError, match="group_idx must be non-negative"):
            CompactCPUAddress(
                byte_offset=0, logical_length=4096, allocated_length=4096, group_idx=-1
            )

    def test_negative_byte_offset(self):
        with pytest.raises(ValueError, match="byte_offset must be non-negative"):
            CompactCPUAddress(
                byte_offset=-1, logical_length=4096, allocated_length=4096
            )

    def test_zero_logical_length(self):
        with pytest.raises(ValueError, match="logical_length must be positive"):
            CompactCPUAddress(byte_offset=0, logical_length=0, allocated_length=0)

    def test_allocated_less_than_logical(self):
        with pytest.raises(ValueError, match="allocated_length.*must be >=.*logical"):
            CompactCPUAddress(byte_offset=0, logical_length=4096, allocated_length=100)

    def test_byte_offset_mismatch_with_first_span(self):
        span = CompactCPUAddressSpan(
            byte_offset=1024, logical_length=2048, allocated_length=4096
        )
        with pytest.raises(
            ValueError, match="byte_offset must match the first physical span"
        ):
            CompactCPUAddress(
                byte_offset=0, logical_length=2048, allocated_length=4096, spans=(span,)
            )

    def test_span_logical_sum_mismatch(self):
        span1 = CompactCPUAddressSpan(
            byte_offset=0, logical_length=2048, allocated_length=4096
        )
        span2 = CompactCPUAddressSpan(
            byte_offset=4096, logical_length=2048, allocated_length=4096
        )
        with pytest.raises(
            ValueError, match="physical spans must cover the logical payload"
        ):
            CompactCPUAddress(
                byte_offset=0,
                logical_length=2048,
                allocated_length=8192,
                spans=(span1, span2),
            )

    def test_fragmented_spans(self):
        span1 = CompactCPUAddressSpan(
            byte_offset=0, logical_length=2048, allocated_length=4096
        )
        span2 = CompactCPUAddressSpan(
            byte_offset=8192, logical_length=2048, allocated_length=4096
        )
        addr = CompactCPUAddress(
            byte_offset=0,
            logical_length=4096,
            allocated_length=8192,
            spans=(span1, span2),
        )
        assert len(addr.spans) == 2
        assert addr.physical_spans == (span1, span2)

    def test_immutability(self):
        addr = CompactCPUAddress(
            byte_offset=0, logical_length=4096, allocated_length=4096
        )
        with pytest.raises(AttributeError, match="cannot assign to field"):
            addr.byte_offset = 100

    def test_page_rounding(self):
        """Allocated_length may be page-rounded larger than logical_length."""
        addr = CompactCPUAddress(
            byte_offset=0, logical_length=100, allocated_length=4096
        )
        assert addr.allocated_length == 4096
        assert addr.logical_length == 100


class TestCompactCPULoadStoreSpec:
    def test_empty(self):
        spec = CompactCPULoadStoreSpec([])
        assert spec.addresses == []
        assert spec.compact_addresses == []

    def test_single_address(self):
        addr = CompactCPUAddress(
            byte_offset=0, logical_length=4096, allocated_length=4096
        )
        spec = CompactCPULoadStoreSpec([addr])
        assert len(spec.addresses) == 1
        assert spec.addresses[0] == addr
        assert spec.compact_addresses[0] == addr

    def test_addresses_alias(self):
        addr1 = CompactCPUAddress(
            byte_offset=0, logical_length=4096, allocated_length=4096
        )
        addr2 = CompactCPUAddress(
            byte_offset=4096, logical_length=2048, allocated_length=4096, group_idx=1
        )
        spec = CompactCPULoadStoreSpec([addr1, addr2])
        assert spec.addresses == spec.compact_addresses

    def test_repr(self):
        spec = CompactCPULoadStoreSpec([])
        r = repr(spec)
        assert "CompactCPULoadStoreSpec" in r
        assert "0 addresses" in r
        spec2 = CompactCPULoadStoreSpec(
            [
                CompactCPUAddress(
                    byte_offset=0, logical_length=4096, allocated_length=4096
                ),
            ]
        )
        r2 = repr(spec2)
        assert "1 addresses" in r2  # plural

    def test_list_copy_on_init(self):
        """Internal list is a copy, not a reference."""
        original = [
            CompactCPUAddress(
                byte_offset=0, logical_length=4096, allocated_length=4096
            ),
        ]
        spec = CompactCPULoadStoreSpec(original)
        original.append(
            CompactCPUAddress(
                byte_offset=4096, logical_length=4096, allocated_length=4096
            )
        )
        assert len(spec.addresses) == 1
