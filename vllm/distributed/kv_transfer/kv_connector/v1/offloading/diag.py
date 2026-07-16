# SPDX-License-Identifier: Apache-2.0
"""KV offload diagnostic JSONL writer.  Pure stdlib -- no vLLM imports."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence

# Module-level gate: resolved once on first access.
_PREFIX: str | None = None
_PATH: str | None = None
_GATE: bool = False


def _resolve() -> None:
    global _PREFIX, _PATH, _GATE  # noqa: PLW0603
    if _PREFIX is not None:
        return
    p = os.environ.get("VLLM_DIAG_KV_OFFLOAD_REQUEST_PREFIX", "")
    q = os.environ.get("VLLM_DIAG_KV_OFFLOAD_PATH", "")
    if p and q:
        _PREFIX, _PATH, _GATE = p, q, True
    else:
        _PREFIX, _PATH, _GATE = "", "", False


def matches(req_id: str) -> bool:
    """Cheap boolean: True when gate is open and *req_id* starts with prefix."""
    _resolve()
    return _GATE and req_id.startswith(_PREFIX)


def key_digest(key: bytes) -> str:
    """SHA-256 hex digest of full ``OffloadKey`` bytes."""
    return hashlib.sha256(key).hexdigest()


# Per-request sequence counter.
_seqs: dict[str, int] = {}


class KvOffloadDiagWriter:
    """Request-ID-gated JSONL diagnostic writer.  All public methods are
    no-ops when the environment gate is closed or the request does not match."""

    def __init__(self) -> None:
        self._path: str = ""

    def open(self) -> None:
        _resolve()
        self._path = _PATH or ""
        if self._path:
            self._write({"schema": 1, "event": "_session_start", "ts": os.times()[4]})

    def close(self) -> None:
        self._path = ""

    # -- low-level ---------------------------------------------------------

    def _write(self, rec: dict) -> None:
        if not self._path:
            return
        try:
            with open(self._path, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        except OSError:
            pass

    def _emit(self, rid: str, event: str, **kw: object) -> None:
        if not self._path or not matches(rid):
            return
        s = _seqs.get(rid, 0) + 1
        _seqs[rid] = s
        self._write({"schema": 1, "rid": rid, "event": event, "seq": s, **kw})

    # -- event helpers ----------------------------------------------------

    def group_specs(self, req_id: str, groups: list) -> None:
        self._emit(req_id, "group_specs", n=len(groups), groups=groups)

    def store_prepare(self, req_id: str, group_idx: int,
                      storable: int, next_before: int, next_after: int,
                      cand_digests: Sequence[str],
                      selected_digests: Sequence[str],
                      block_indices: Sequence[int],
                      job_id: int) -> None:
        self._emit(req_id, "store_prepare", gidx=group_idx, s=storable,
                   nb=next_before, na=next_after,
                   cd=list(cand_digests), sd=list(selected_digests),
                   bi=[int(i) for i in block_indices], jid=job_id)

    def store_complete(self, req_id: str,
                       completed_digests: Sequence[str],
                       success: bool) -> None:
        self._emit(req_id, "store_complete",
                   digs=list(completed_digests), ok=success)

    def lookup_group(self, req_id: str,
                     convergence_pass: int, group_idx: int,
                     group_type: str, is_eagle: bool,
                     start_block_idx: int, query_max: int,
                     token_boundary: int,
                     replay_digests: Sequence[str],
                     lookup_results: Sequence[str],
                     raw_hit_count: int, post_eagle_hit_count: int,
                     eagle_verified: bool, defer_lookup: bool,
                     tightened_boundary: int) -> None:
        self._emit(req_id, "lookup_group",
                   cp=convergence_pass, gidx=group_idx, gt=group_type,
                   ie=is_eagle, sbi=start_block_idx, qm=query_max,
                   tb=token_boundary, rd=list(replay_digests),
                   lr=list(lookup_results), rhc=raw_hit_count,
                   peh=post_eagle_hit_count, ev=eagle_verified,
                   dl=defer_lookup, tgb=tightened_boundary)

    def lookup_terminal(self, req_id: str,
                        terminal_group_idx: int | None,
                        total_passes: int,
                        final_local_tokens: int,
                        final_external_tokens: int,
                        defer_lookup: bool) -> None:
        self._emit(req_id, "lookup_terminal",
                   tgi=terminal_group_idx, cp=total_passes,
                   flt=final_local_tokens, fet=final_external_tokens,
                   dl=defer_lookup)

    def request_finished(self, req_id: str,
                         next_stored_block_indices: Sequence[int],
                         total_keys: Sequence[int],
                         total_block_ids: Sequence[int],
                         pending_job_ids: Sequence[int],
                         has_in_flight_jobs: bool) -> None:
        self._emit(req_id, "request_finished",
                   nsi=[int(i) for i in next_stored_block_indices],
                   tk=[int(i) for i in total_keys],
                   tbi=[int(i) for i in total_block_ids],
                   pji=[int(j) for j in pending_job_ids],
                   hif=has_in_flight_jobs)
