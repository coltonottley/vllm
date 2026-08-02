# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic CPU-pure tests for the DeepSeek-V4 no-breakable policy.

DeepSeek-V4 must NOT auto-enable ``VLLM_USE_BREAKABLE_CUDAGRAPH`` when the
env var is absent; explicit ``VLLM_USE_BREAKABLE_CUDAGRAPH=1`` must still
work. Breakable cudagraph disables the torch.compile pipeline and is 1.5-3.8x
slower for MTP decode on SM12x (measured on RTX PRO 6000 / GB10); the
default is FULL_AND_PIECEWISE + torch.compile.
"""

import os
from types import SimpleNamespace

import pytest

import vllm.envs as envs
from vllm.config.vllm import (
    _should_auto_enable_deepseek_v4_breakable_cudagraph,
)


def _model_config(*architectures: str):
    return SimpleNamespace(architectures=list(architectures))


def test_deepseek_v4_does_not_auto_enable_breakable_cudagraph():
    # Architecture-independent: never auto-enable breakable for DeepSeek-V4,
    # on any device path.
    assert not _should_auto_enable_deepseek_v4_breakable_cudagraph(
        _model_config("DeepseekV4ForCausalLM")
    )
    assert not _should_auto_enable_deepseek_v4_breakable_cudagraph(
        _model_config("DeepSeekV4MTPModel")
    )


def test_non_deepseek_v4_does_not_auto_enable_breakable_cudagraph():
    assert not _should_auto_enable_deepseek_v4_breakable_cudagraph(
        _model_config("Qwen3ForCausalLM")
    )


def test_breakable_env_absent_means_not_auto_enabled(monkeypatch: pytest.MonkeyPatch):
    # When the env var is absent, the breakable path is OFF by default.
    monkeypatch.delenv("VLLM_USE_BREAKABLE_CUDAGRAPH", raising=False)
    assert envs.VLLM_USE_BREAKABLE_CUDAGRAPH is False


def test_breakable_env_1_still_works(monkeypatch: pytest.MonkeyPatch):
    # Explicit opt-in with VLLM_USE_BREAKABLE_CUDAGRAPH=1 must still work.
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    assert envs.VLLM_USE_BREAKABLE_CUDAGRAPH is True
    assert os.environ.get("VLLM_USE_BREAKABLE_CUDAGRAPH") == "1"
