# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for skipping decode-side tokenization in PD disaggregated serving.

When the router forwards pre-tokenized prompt_token_ids from the prefill
response, the decode side should skip chat template rendering and
tokenization (render_chat_async) and use the token IDs directly.
"""

from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from vllm.config import MultiModalConfig
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.models.serving import (
    BaseModelPath,
    OpenAIServingModels,
)
from vllm.entrypoints.serve.render.serving import OpenAIServingRender
from vllm.renderers.hf import HfRenderer
from vllm.tokenizers.registry import tokenizer_args_from_config
from vllm.v1.engine.async_llm import AsyncLLM

MODEL_NAME = "openai-community/gpt2"
BASE_MODEL_PATHS = [BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME)]
# GPT-2's default chat template
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['role'] }}: {{ message['content'] }}\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}assistant: {% endif %}"
)


@dataclass
class MockHFConfig:
    model_type: str = "any"


@dataclass
class MockModelConfig:
    task = "generate"
    runner_type = "generate"
    model = MODEL_NAME
    tokenizer = MODEL_NAME
    trust_remote_code = False
    tokenizer_mode = "auto"
    max_model_len = 100
    tokenizer_revision = None
    multimodal_config = MultiModalConfig()
    hf_config = MockHFConfig()
    hf_text_config = MockHFConfig()
    logits_processors: list[str] | None = None
    diff_sampling_param: dict | None = None
    allowed_local_media_path: str = ""
    allowed_media_domains: list[str] | None = None
    encoder_config = None
    generation_config: str = "auto"
    override_generation_config: dict[str, Any] = field(default_factory=dict)
    media_io_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    skip_tokenizer_init: bool = False
    is_encoder_decoder: bool = False
    is_multimodal_model: bool = False

    def get_diff_sampling_param(self):
        return self.diff_sampling_param or {}


@dataclass
class MockParallelConfig:
    _api_process_rank: int = 0


@dataclass
class MockVllmConfig:
    model_config: MockModelConfig
    parallel_config: MockParallelConfig


def _build_renderer(model_config: MockModelConfig):
    _, tokenizer_name, _, kwargs = tokenizer_args_from_config(model_config)
    return HfRenderer.from_config(
        MockVllmConfig(model_config, parallel_config=MockParallelConfig()),
        tokenizer_kwargs={**kwargs, "tokenizer_name": tokenizer_name},
    )


def _build_serving_render(
    engine,
    model_registry,
) -> OpenAIServingRender:
    return OpenAIServingRender(
        model_config=engine.model_config,
        renderer=engine.renderer,
        io_processor=engine.io_processor,
        model_registry=model_registry,
        request_logger=None,
        chat_template=CHAT_TEMPLATE,
        chat_template_content_format="auto",
    )


def _build_serving_chat(engine) -> OpenAIServingChat:
    models = OpenAIServingModels(
        engine_client=engine,
        base_model_paths=BASE_MODEL_PATHS,
    )
    openai_serving_render = _build_serving_render(engine, models.registry)
    return OpenAIServingChat(
        engine,
        models,
        response_role="assistant",
        openai_serving_render=openai_serving_render,
        chat_template=CHAT_TEMPLATE,
        chat_template_content_format="auto",
        request_logger=None,
    )


def _build_mock_engine():
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.io_processor = MagicMock()
    mock_engine.renderer = _build_renderer(mock_engine.model_config)
    return mock_engine


class TestPromptTokenIdsRequestField:
    """Tests for the prompt_token_ids field on ChatCompletionRequest."""

    def test_request_without_prompt_token_ids(self):
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "test-model",
            }
        )
        assert request.prompt_token_ids is None

    def test_request_with_prompt_token_ids(self):
        token_ids = [100, 200, 300, 400, 500]
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "test-model",
                "prompt_token_ids": token_ids,
            }
        )
        assert request.prompt_token_ids == token_ids

    def test_request_with_prompt_token_ids_and_kv_transfer_params(self):
        """Typical decode-side request from router."""
        token_ids = list(range(16000))
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "test-model",
                "prompt_token_ids": token_ids,
                "kv_transfer_params": {
                    "do_remote_prefill": True,
                    "do_remote_decode": False,
                    "remote_block_ids": [[1, 2, 3]],
                    "remote_engine_id": "engine-1",
                },
            }
        )
        assert request.prompt_token_ids == token_ids
        assert request.kv_transfer_params["do_remote_prefill"] is True
        assert len(request.prompt_token_ids) == 16000

    def test_request_with_empty_prompt_token_ids(self):
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "test-model",
                "prompt_token_ids": [],
            }
        )
        assert request.prompt_token_ids == []


class TestSkipRenderCondition:
    """Tests for the skip_render condition logic."""

    def _should_skip_render(self, request: ChatCompletionRequest) -> bool:
        """Reproduce the skip_render condition from serving.py."""
        return bool(
            request.kv_transfer_params
            and request.kv_transfer_params.get("do_remote_prefill")
            and request.prompt_token_ids
        )

    def test_skip_render_decode_with_token_ids(self):
        """Decode side with prompt_token_ids -> should skip."""
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "test-model",
                "prompt_token_ids": [1, 2, 3],
                "kv_transfer_params": {"do_remote_prefill": True},
            }
        )
        assert self._should_skip_render(request) is True

    def test_no_skip_render_without_kv_params(self):
        """Normal request without PD -> should not skip."""
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "test-model",
            }
        )
        assert self._should_skip_render(request) is False

    def test_no_skip_render_prefill_side(self):
        """Prefill side (do_remote_decode) -> should not skip."""
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "test-model",
                "kv_transfer_params": {"do_remote_decode": True},
            }
        )
        assert self._should_skip_render(request) is False

    def test_no_skip_render_without_token_ids(self):
        """Decode side but no prompt_token_ids (old router) -> should not skip."""
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "test-model",
                "kv_transfer_params": {"do_remote_prefill": True},
            }
        )
        assert self._should_skip_render(request) is False

    def test_no_skip_render_empty_token_ids(self):
        """Decode side with empty prompt_token_ids -> should not skip."""
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "test-model",
                "prompt_token_ids": [],
                "kv_transfer_params": {"do_remote_prefill": True},
            }
        )
        assert self._should_skip_render(request) is False

    def test_skip_render_with_echo(self):
        """Decode side with echo=True -> still skip (conversation preserved)."""
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "test-model",
                "prompt_token_ids": [1, 2, 3],
                "kv_transfer_params": {"do_remote_prefill": True},
                "echo": True,
            }
        )
        assert self._should_skip_render(request) is True


class TestPrefillResponseIncludesTokenIds:
    """Tests for the prefill response prompt_token_ids inclusion logic."""

    def _should_include_prompt_token_ids(
        self,
        return_token_ids: bool,
        kv_transfer_params: dict | None,
    ) -> bool:
        """Reproduce the condition from serving.py response construction."""
        return bool(
            return_token_ids
            or (kv_transfer_params and kv_transfer_params.get("do_remote_prefill"))
        )

    def test_include_when_return_token_ids(self):
        assert self._should_include_prompt_token_ids(True, None) is True

    def test_include_when_do_remote_prefill(self):
        """Prefill response for PD -> should include for router to forward."""
        assert (
            self._should_include_prompt_token_ids(False, {"do_remote_prefill": True})
            is True
        )

    def test_exclude_when_no_flags(self):
        assert self._should_include_prompt_token_ids(False, None) is False

    def test_exclude_when_do_remote_decode(self):
        assert (
            self._should_include_prompt_token_ids(False, {"do_remote_decode": True})
            is False
        )


class TestSkipTokenizationE2E:
    """End-to-end tests: normal path vs skip-tokenization path produce
    equivalent engine prompts (same token IDs, conversation, cache_salt)."""

    @pytest.mark.asyncio
    async def test_same_token_ids_single_turn(self):
        """Single-turn: skip path with prefill token IDs matches normal."""
        engine = _build_mock_engine()
        serving_chat = _build_serving_chat(engine)

        messages = [{"role": "user", "content": "what is 1+1?"}]

        # --- Normal path (no skip) ---
        normal_req = ChatCompletionRequest(
            model=MODEL_NAME,
            messages=messages,
        )
        normal_result = await serving_chat.render_chat_request(normal_req)
        assert isinstance(normal_result, tuple)
        normal_conv, normal_prompts = normal_result
        normal_token_ids = normal_prompts[0]["prompt_token_ids"]

        # --- Skip path (with prompt_token_ids) ---
        skip_result = await serving_chat.render_chat_request(
            normal_req,
            prompt_token_ids=normal_token_ids,
        )
        assert isinstance(skip_result, tuple)
        skip_conv, skip_prompts = skip_result

        # Token IDs must be identical.
        assert skip_prompts[0]["prompt_token_ids"] == normal_token_ids

        # Conversation messages should have the same roles and content.
        assert len(skip_conv) == len(normal_conv)
        for skip_msg, normal_msg in zip(skip_conv, normal_conv):
            assert skip_msg["role"] == normal_msg["role"]
            assert skip_msg["content"] == normal_msg["content"]

    @pytest.mark.asyncio
    async def test_same_token_ids_multi_turn(self):
        """Multi-turn conversation: token IDs match between paths."""
        engine = _build_mock_engine()
        serving_chat = _build_serving_chat(engine)

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help you?"},
            {"role": "user", "content": "What is the weather today?"},
        ]

        # Normal path
        normal_req = ChatCompletionRequest(
            model=MODEL_NAME,
            messages=messages,
        )
        normal_result = await serving_chat.render_chat_request(normal_req)
        assert isinstance(normal_result, tuple)
        normal_conv, normal_prompts = normal_result
        normal_token_ids = normal_prompts[0]["prompt_token_ids"]

        # Skip path
        skip_result = await serving_chat.render_chat_request(
            normal_req,
            prompt_token_ids=normal_token_ids,
        )
        assert isinstance(skip_result, tuple)
        skip_conv, skip_prompts = skip_result

        assert skip_prompts[0]["prompt_token_ids"] == normal_token_ids
        assert len(skip_conv) == len(normal_conv)

    @pytest.mark.asyncio
    async def test_cache_salt_preserved_in_skip_path(self):
        """cache_salt should be propagated even when tokenization is skipped."""
        engine = _build_mock_engine()
        serving_chat = _build_serving_chat(engine)

        messages = [{"role": "user", "content": "test"}]

        # Get token IDs from normal path
        normal_req = ChatCompletionRequest(
            model=MODEL_NAME,
            messages=messages,
        )
        normal_result = await serving_chat.render_chat_request(normal_req)
        assert isinstance(normal_result, tuple)
        normal_token_ids = normal_result[1][0]["prompt_token_ids"]

        # Skip path with cache_salt
        req_with_salt = ChatCompletionRequest(
            model=MODEL_NAME,
            messages=messages,
            cache_salt="my-salt",
        )
        skip_result = await serving_chat.render_chat_request(
            req_with_salt,
            prompt_token_ids=normal_token_ids,
        )
        assert isinstance(skip_result, tuple)
        _, skip_prompts = skip_result
        assert skip_prompts[0]["cache_salt"] == "my-salt"

    @pytest.mark.asyncio
    async def test_normal_path_no_cache_salt(self):
        """Verify cache_salt is absent when not set (both paths)."""
        engine = _build_mock_engine()
        serving_chat = _build_serving_chat(engine)

        messages = [{"role": "user", "content": "test"}]
        req = ChatCompletionRequest(model=MODEL_NAME, messages=messages)

        normal_result = await serving_chat.render_chat_request(req)
        assert isinstance(normal_result, tuple)
        normal_token_ids = normal_result[1][0]["prompt_token_ids"]
        assert "cache_salt" not in normal_result[1][0]

        skip_result = await serving_chat.render_chat_request(
            req,
            prompt_token_ids=normal_token_ids,
        )
        assert isinstance(skip_result, tuple)
        assert "cache_salt" not in skip_result[1][0]

    @pytest.mark.asyncio
    async def test_create_chat_completion_uses_skip_path(self):
        """Full create_chat_completion with PD skip condition passes
        prompt_token_ids through render_chat_request."""
        engine = _build_mock_engine()
        serving_chat = _build_serving_chat(engine)

        # Capture what render_chat_request receives
        orig_render = serving_chat.render_chat_request
        captured_kwargs: list[dict] = []

        async def tracking_render(request, prompt_token_ids=None):
            captured_kwargs.append({"prompt_token_ids": prompt_token_ids})
            return await orig_render(request, prompt_token_ids=prompt_token_ids)

        serving_chat.render_chat_request = tracking_render

        # Normal request -- prompt_token_ids should be None
        normal_req = ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Hello"}],
        )
        with suppress(Exception):
            await serving_chat.create_chat_completion(normal_req)

        assert captured_kwargs[-1]["prompt_token_ids"] is None

        # PD decode request -- prompt_token_ids should be passed through
        pd_req = ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Hello"}],
            prompt_token_ids=[1, 2, 3, 4, 5],
            kv_transfer_params={"do_remote_prefill": True},
        )
        with suppress(Exception):
            await serving_chat.create_chat_completion(pd_req)

        assert captured_kwargs[-1]["prompt_token_ids"] == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_model_validation_runs_in_skip_path(self):
        """_check_model and engine health checks still run when skipping."""
        engine = _build_mock_engine()
        serving_chat = _build_serving_chat(engine)

        # Request with invalid model name
        req = ChatCompletionRequest(
            model="nonexistent-model",
            messages=[{"role": "user", "content": "Hello"}],
        )

        # Normal path -- should return ErrorResponse
        from vllm.entrypoints.openai.engine.protocol import ErrorResponse

        normal_result = await serving_chat.render_chat_request(req)
        assert isinstance(normal_result, ErrorResponse)

        # Skip path -- should ALSO return ErrorResponse (not bypass validation)
        skip_result = await serving_chat.render_chat_request(
            req,
            prompt_token_ids=[1, 2, 3],
        )
        assert isinstance(skip_result, ErrorResponse)

    @pytest.mark.asyncio
    async def test_engine_health_check_in_skip_path(self):
        """Engine dead error is raised even when skip path is used."""
        engine = _build_mock_engine()
        engine.errored = True
        engine.dead_error = RuntimeError("Engine is dead")

        serving_chat = _build_serving_chat(engine)
        req = ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Hello"}],
        )

        with pytest.raises(RuntimeError, match="Engine is dead"):
            await serving_chat.render_chat_request(
                req,
                prompt_token_ids=[1, 2, 3],
            )
