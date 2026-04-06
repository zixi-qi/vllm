# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for speculative_config.attention_backend and its effect on
draft model attention backend selection."""

import pytest

from vllm.config import (
    AttentionConfig,
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.config.load import LoadConfig
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum

model_dir = "meta-llama/Llama-3.1-8B-Instruct"
eagle_dir = "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"

DEVICE_TYPE = current_platform.device_type


def _make_vllm_config(
    target_attention_backend: AttentionBackendEnum | None = None,
    draft_attention_backend: str | AttentionBackendEnum | None = None,
) -> VllmConfig:
    """Create a VllmConfig with the given target and draft attention backends."""
    model_config = ModelConfig(
        model=model_dir,
        runner="generate",
        max_model_len=100,
    )

    speculative_config = SpeculativeConfig(
        target_model_config=model_config,
        target_parallel_config=ParallelConfig(),
        model=eagle_dir,
        method="eagle",
        num_speculative_tokens=3,
        attention_backend=draft_attention_backend,
    )

    return VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(block_size=16),
        speculative_config=speculative_config,
        device_config=DeviceConfig(device=DEVICE_TYPE),
        parallel_config=ParallelConfig(),
        load_config=LoadConfig(),
        scheduler_config=SchedulerConfig(
            max_model_len=model_config.max_model_len,
            is_encoder_decoder=model_config.is_encoder_decoder,
        ),
        attention_config=AttentionConfig(backend=target_attention_backend),
    )


# ---------------------------------------------------------------------------
# SpeculativeConfig.attention_backend field validation
# ---------------------------------------------------------------------------


class TestAttentionBackendValidator:
    """Test the field_validator on SpeculativeConfig.attention_backend."""

    def test_none_returns_none(self):
        assert SpeculativeConfig._validate_attention_backend(None) is None

    def test_auto_returns_sentinel(self):
        assert SpeculativeConfig._validate_attention_backend("auto") == "auto"

    def test_auto_case_insensitive(self):
        assert SpeculativeConfig._validate_attention_backend("AUTO") == "auto"
        assert SpeculativeConfig._validate_attention_backend("Auto") == "auto"

    def test_specific_backend_string(self):
        result = SpeculativeConfig._validate_attention_backend("FLASH_ATTN")
        assert result == AttentionBackendEnum.FLASH_ATTN

    def test_specific_backend_case_insensitive(self):
        result = SpeculativeConfig._validate_attention_backend("flash_attn")
        assert result == AttentionBackendEnum.FLASH_ATTN

    def test_enum_passthrough(self):
        result = SpeculativeConfig._validate_attention_backend(
            AttentionBackendEnum.FLASH_ATTN
        )
        assert result == AttentionBackendEnum.FLASH_ATTN

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError):
            SpeculativeConfig._validate_attention_backend("NOT_A_BACKEND")


# ---------------------------------------------------------------------------
# SpeculativeConfig.draft_attention_backend property
# ---------------------------------------------------------------------------


class TestDraftAttentionBackendProperty:
    """Test that the draft_attention_backend property resolves correctly."""

    def _make_spec_config(
        self, attention_backend: str | None = None
    ) -> SpeculativeConfig:
        model_config = ModelConfig(
            model=model_dir, runner="generate", max_model_len=100
        )
        return SpeculativeConfig(
            target_model_config=model_config,
            target_parallel_config=ParallelConfig(),
            model=eagle_dir,
            method="eagle",
            num_speculative_tokens=3,
            attention_backend=attention_backend,
        )

    def test_none_resolves_to_none(self):
        cfg = self._make_spec_config(attention_backend=None)
        assert cfg.draft_attention_backend is None

    def test_auto_resolves_to_none(self):
        cfg = self._make_spec_config(attention_backend="auto")
        assert cfg.draft_attention_backend is None

    def test_explicit_backend_passes_through(self):
        cfg = self._make_spec_config(attention_backend="FLASH_ATTN")
        assert cfg.draft_attention_backend == AttentionBackendEnum.FLASH_ATTN


# ---------------------------------------------------------------------------
# _create_draft_vllm_config attention backend override
# ---------------------------------------------------------------------------


class TestCreateDraftVllmConfig:
    """Test that _create_draft_vllm_config respects attention_backend."""

    def _get_draft_attention_backend(
        self,
        target_backend: AttentionBackendEnum | None,
        draft_backend_setting: str | AttentionBackendEnum | None,
    ) -> AttentionBackendEnum | None:
        """Helper: create proposer and return the draft config's backend."""
        from vllm.v1.spec_decode.eagle import EagleProposer

        vllm_config = _make_vllm_config(
            target_attention_backend=target_backend,
            draft_attention_backend=draft_backend_setting,
        )
        proposer = EagleProposer(vllm_config=vllm_config, device=DEVICE_TYPE)
        draft_config = proposer._create_draft_vllm_config()
        return draft_config.attention_config.backend

    def test_default_inherits_target_backend(self):
        """Default (None) should inherit the target model's backend."""
        backend = self._get_draft_attention_backend(
            target_backend=AttentionBackendEnum.FLASH_ATTN,
            draft_backend_setting=None,
        )
        assert backend == AttentionBackendEnum.FLASH_ATTN

    def test_default_inherits_none_target(self):
        """Default (None) with no target backend should stay None."""
        backend = self._get_draft_attention_backend(
            target_backend=None,
            draft_backend_setting=None,
        )
        assert backend is None

    def test_auto_overrides_to_none(self):
        """'auto' should reset backend to None (auto-select)."""
        backend = self._get_draft_attention_backend(
            target_backend=AttentionBackendEnum.FLASHINFER_MLA,
            draft_backend_setting="auto",
        )
        assert backend is None

    def test_specific_backend_override(self):
        """Explicit backend should override the target's backend."""
        backend = self._get_draft_attention_backend(
            target_backend=AttentionBackendEnum.FLASHINFER_MLA,
            draft_backend_setting="FLASH_ATTN",
        )
        assert backend == AttentionBackendEnum.FLASH_ATTN

    def test_target_config_unchanged(self):
        """The original vllm_config should not be mutated."""
        from vllm.v1.spec_decode.eagle import EagleProposer

        vllm_config = _make_vllm_config(
            target_attention_backend=AttentionBackendEnum.FLASHINFER_MLA,
            draft_attention_backend="auto",
        )
        proposer = EagleProposer(vllm_config=vllm_config, device=DEVICE_TYPE)
        proposer._create_draft_vllm_config()
        # Target config should still have the original backend.
        assert (
            vllm_config.attention_config.backend == AttentionBackendEnum.FLASHINFER_MLA
        )
