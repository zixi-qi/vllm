# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptLinearMethod,
    resolve,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
    nvfp4_gathered_bias,
)
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator


def _markov_head(weight: torch.Tensor) -> DSparkMarkovHead:
    head = DSparkMarkovHead.__new__(DSparkMarkovHead)
    nn.Module.__init__(head)
    head.markov_w2 = nn.Linear(
        weight.shape[1], weight.shape[0], bias=False, dtype=weight.dtype
    )
    head.markov_w2.weight.data.copy_(weight)
    head.markov_w2._retain_weight_for_gather = False
    head.markov_w2.is_w4a16_nvfp4 = False
    return head


def test_gathered_markov_bias_overwrites_dense_logits():
    weight = torch.arange(21, dtype=torch.float32).view(7, 3) / 10
    markov_embed = torch.tensor([[0.5, -1.0, 0.25], [1.0, 0.5, -0.5]])
    logits = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        ]
    )
    values, index = logits.topk(3, dim=-1)
    values = torch.stack((values, torch.zeros_like(values)), dim=1)[:, 0]
    expected = values + torch.bmm(weight[index], markov_embed.unsqueeze(-1)).squeeze(-1)
    logits.fill_(float("-inf"))

    result = _markov_head(weight).apply_bias_gathered(
        markov_embed, logits, values, index
    )

    assert result is logits
    torch.testing.assert_close(result.gather(1, index), expected)
    selected = torch.zeros_like(result, dtype=torch.bool).scatter_(1, index, True)
    assert torch.isneginf(result.masked_select(~selected)).all()


def test_gathered_markov_bias_matches_dense_at_full_vocab():
    weight = torch.arange(15, dtype=torch.float32).view(5, 3) / 10
    markov_embed = torch.tensor([[0.5, -1.0, 0.25]])
    logits = torch.tensor([[0.1, 0.4, -0.2, 0.3, 0.0]])
    original = logits.clone()
    values, index = logits.topk(logits.shape[-1], dim=-1)
    scale = 0.5
    logits.fill_(float("-inf"))

    result = _markov_head(weight).apply_bias_gathered(
        markov_embed, logits, values, index, scale
    )

    expected = original + markov_embed @ weight.T * scale
    torch.testing.assert_close(result, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("use_fp64", [False, True])
def test_dspark_revisited_position_uses_step_seed(use_fp64):
    """A cached draft at a revisited position must use its current step."""
    device = torch.device("cuda")
    batch = 4096
    spec = DSparkSpeculator.__new__(DSparkSpeculator)
    spec.acceptance_estimator = None
    spec._d2t_scatter_index = None
    spec.draft_watermarker = None
    spec.use_fp64_gumbel = use_fp64
    spec.temperature = torch.ones(batch, device=device)
    spec.seeds = torch.arange(batch, dtype=torch.int64, device=device)
    spec._step_cols = torch.arange(2, dtype=torch.int32, device=device)
    spec.draft_logits = torch.empty(batch, 2, 2, device=device)
    logits = torch.tensor([0.7, 0.3], device=device).log().expand(batch, -1)
    mapping = torch.arange(batch, device=device)
    positions = torch.full((batch,), 22, dtype=torch.int64, device=device)

    first = spec._sample_logits(logits, mapping, positions, 1)
    repeated = spec._sample_logits(logits, mapping, positions, 1)
    next_round = spec._sample_logits(logits, mapping, positions, 0)
    torch.testing.assert_close(first, repeated)
    expected = gumbel_sample(
        logits,
        mapping,
        spec.temperature,
        spec.seeds,
        positions,
        apply_temperature=True,
        is_drafting=True,
        logits_cache=spec.draft_logits,
        logits_cache_col=spec._step_cols[0],
        use_fp64=use_fp64,
    )
    torch.testing.assert_close(next_round, expected)
    mismatch_rate = (first != next_round).float().mean().item()
    assert 0.35 < mismatch_rate < 0.5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("probabilistic", [True, False])
@pytest.mark.parametrize("reduced_vocab", [False, True])
@pytest.mark.parametrize("num_reqs", [1, 3, 37])
def test_markov_sampling_matches_adding_bias_first(
    probabilistic: bool, reduced_vocab: bool, num_reqs: int
):
    """Every drafting mode gives the draft ids and cached draft logits of adding
    the Markov bias first. Probabilistic full-vocab drafting adds it inside the
    Gumbel kernel and writes the ids in place."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_steps, max_num_reqs, hidden, rank = 4, 64, 64, 32
    # Not a multiple of the Gumbel kernel's block size.
    draft_vocab = 3000
    target_vocab = draft_vocab + 500 if reduced_vocab else draft_vocab
    g = torch.Generator(device=device).manual_seed(num_reqs)

    def randn(*shape: int, scale: float = 1.0) -> torch.Tensor:
        x = torch.randn(*shape, generator=g, device=device)
        return (x * scale).to(dtype)

    # Unit-scale logits and bias, so most draws are not the argmax.
    lm_head = randn(draft_vocab, hidden, scale=hidden**-0.5)
    markov_w1 = randn(target_vocab, rank)
    markov_w2 = randn(draft_vocab, rank, scale=rank**-0.5)
    d2t = None
    if reduced_vocab:
        # Draft id i maps to target id i + d2t[i].
        target_ids = torch.randperm(target_vocab, generator=g, device=device)
        d2t = target_ids[:draft_vocab].sort().values
        d2t -= torch.arange(draft_vocab, device=device)
    head_hidden = randn(num_reqs * num_steps, hidden)
    slots = torch.randperm(max_num_reqs, generator=g, device=device)[:num_reqs]
    last_pos = torch.randint(10, 5000, (num_reqs, 1), generator=g, device=device)
    # A mix of greedy and random requests.
    temps = torch.tensor([0.0, 0.6, 1.0, 1.3], device=device)
    temperature = temps[torch.randint(4, (max_num_reqs,), generator=g, device=device)]
    seeds = torch.randint(2**62, (max_num_reqs,), generator=g, device=device)
    input_ids = torch.randint(
        target_vocab,
        (max_num_reqs * num_steps,),
        generator=g,
        device=device,
        dtype=torch.int32,
    )

    def make_speculator() -> DSparkSpeculator:
        spec = DSparkSpeculator.__new__(DSparkSpeculator)
        spec.model = SimpleNamespace(
            compute_draft_logits=lambda h: h @ lm_head.T,
            markov_embed=lambda ids: markov_w1[ids],
            markov_bias=lambda e: e @ markov_w2.T,
            map_draft_to_target=lambda ids: ids if d2t is None else ids + d2t[ids],
        )
        spec.num_speculative_steps = num_steps
        spec._draft_topk = None
        spec.use_confidence_head = False
        spec.acceptance_estimator = None
        spec.draft_watermarker = None
        spec.use_fp64_gumbel = False
        spec.temperature = temperature
        spec.seeds = seeds
        spec._step_cols = torch.arange(num_steps, dtype=torch.int32, device=device)
        spec._anchor_idx = torch.arange(max_num_reqs, device=device) * num_steps
        spec.sample_indices = torch.arange(num_reqs * num_steps, device=device)
        spec.sample_idx_mapping = slots.repeat_interleave(num_steps).int()
        spec.sample_pos = (last_pos + torch.arange(num_steps, device=device)).view(-1)
        spec.input_buffers = SimpleNamespace(input_ids=input_ids)
        spec.draft_tokens = torch.zeros(
            max_num_reqs, num_steps, dtype=torch.int64, device=device
        )
        spec.draft_logits = None
        spec._d2t_scatter_index = None
        if probabilistic:
            spec.draft_logits = torch.zeros(
                max_num_reqs, num_steps, target_vocab, dtype=dtype, device=device
            )
        if probabilistic and d2t is not None:
            spec._d2t_scatter_index = torch.arange(draft_vocab, device=device) + d2t
            spec._draft_scatter_buf = torch.full(
                (max_num_reqs, target_vocab), float("-inf"), dtype=dtype, device=device
            )
        return spec

    fused, reference = make_speculator(), make_speculator()
    fused._sample_sequential(num_reqs, head_hidden)
    # The reference adds the bias to the logits before sampling, in every mode.
    base_logits = reference.model.compute_draft_logits(head_hidden)
    base_logits = base_logits.view(num_reqs, num_steps, -1)
    idx_map = reference.sample_idx_mapping.view(num_reqs, num_steps)
    draw_pos = reference.sample_pos.view(num_reqs, num_steps) - 1
    prev = input_ids[reference._anchor_idx[:num_reqs]]
    for i in range(num_steps):
        bias = reference.model.markov_bias(reference.model.markov_embed(prev))
        prev = reference._sample_logits(
            base_logits[:, i] + bias, idx_map[:, i], draw_pos[:, i], i
        )
        reference.draft_tokens[:num_reqs, i] = prev

    assert torch.equal(fused.draft_tokens, reference.draft_tokens)
    if probabilistic:
        draft_logits = fused.draft_logits
        assert draft_logits is not None and draft_logits.any()
        assert torch.equal(draft_logits, reference.draft_logits)


@pytest.mark.parametrize("cuda_alike_platform", [False, True])
def test_gathered_markov_bias_dequantizes_selected_w4a16_rows(
    monkeypatch, cuda_alike_platform
):
    packed_weight = torch.tensor(
        [[0x00] * 8, [0x22] * 8, [0xAA] * 8, [0x31] * 8], dtype=torch.uint8
    )
    weight_scale = torch.ones((4, 1), dtype=torch.float8_e4m3fn)

    layer = nn.Module()
    layer.weight = nn.Parameter(packed_weight, requires_grad=False)
    layer.weight_scale = nn.Parameter(weight_scale, requires_grad=False)
    layer.weight_scale_2 = nn.Parameter(torch.ones(1), requires_grad=False)
    layer._retain_weight_for_gather = True

    spec, ctx, fmt = resolve("W4A16_NVFP4", type("Config", (), {"group_size": 16}), "")
    method = ModelOptLinearMethod(spec, ctx, fmt)

    class RepackingKernel:
        @staticmethod
        def process_weights_after_loading(layer):
            layer.weight = nn.Parameter(
                torch.zeros((1, 1), dtype=torch.int32), requires_grad=False
            )
            layer.weight_scale = nn.Parameter(
                torch.zeros((1, 1), dtype=torch.float32), requires_grad=False
            )
            layer.weight_global_scale = nn.Parameter(
                torch.zeros(1, dtype=torch.float32), requires_grad=False
            )

    method.kernel = RepackingKernel()
    method.process_weights_after_loading(layer)
    assert layer.is_w4a16_nvfp4

    head = DSparkMarkovHead.__new__(DSparkMarkovHead)
    nn.Module.__init__(head)
    head.markov_w2 = layer
    markov_embed = torch.tensor([[1.0] * 16, [0.5, -0.5] * 8], dtype=torch.float32)
    index = torch.tensor([[1, 3], [2, 0]])
    values = torch.tensor([[0.25, -0.5], [1.0, 2.0]])
    original_values = values.clone()
    logits = torch.full((2, 4), float("-inf"))

    # CPU tensors must use the fallback even on CUDA/ROCm hosts.
    monkeypatch.setattr(current_platform, "is_cuda_alike", lambda: cuda_alike_platform)
    result = head.apply_bias_gathered(markov_embed, logits, values, index)

    dense_weight = torch.tensor([[0.0] * 16, [1.0] * 16, [-1.0] * 16, [0.5, 1.5] * 8])
    expected = original_values + torch.bmm(
        dense_weight[index], markov_embed.unsqueeze(-1)
    ).squeeze(-1)
    torch.testing.assert_close(result.gather(1, index), expected)
    torch.testing.assert_close(values, original_values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_nvfp4_gathered_bias_rejects_non_16_group_size():
    device = torch.device("cuda")
    rank = 32
    vocab_size = 2

    with pytest.raises(ValueError, match="group size of 16"):
        nvfp4_gathered_bias(
            torch.zeros((1, rank), dtype=torch.bfloat16, device=device),
            torch.zeros((vocab_size, rank // 2), dtype=torch.uint8, device=device),
            torch.ones(
                (vocab_size, rank // 32), dtype=torch.float8_e4m3fn, device=device
            ),
            torch.ones(1, dtype=torch.float32, device=device),
            torch.zeros((1, 1), dtype=torch.bfloat16, device=device),
            torch.zeros((1, 1), dtype=torch.int64, device=device),
            torch.full(
                (1, vocab_size), float("-inf"), dtype=torch.bfloat16, device=device
            ),
            1.0,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_nvfp4_gathered_markov_bias_kernel_matches_reference(batch_size: int):
    torch.manual_seed(0)
    device = torch.device("cuda")
    topk = 256
    rank = 512
    vocab_size = 512
    alpha = 0.75

    packed_weight = torch.randint(
        0, 256, (vocab_size, rank // 2), dtype=torch.uint8, device=device
    )
    weight_scale = torch.full(
        (vocab_size, rank // 16),
        0.5,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    global_scale = torch.tensor([0.25], dtype=torch.float32, device=device)

    markov_storage = torch.randn(
        batch_size, 2, rank, dtype=torch.bfloat16, device=device
    )
    values_storage = torch.randn(
        batch_size, 2, topk, dtype=torch.bfloat16, device=device
    )
    index_storage = torch.empty(batch_size, 2, topk, dtype=torch.int64, device=device)
    for batch_idx in range(batch_size):
        index_storage[batch_idx, 0] = (
            torch.arange(topk, device=device) + batch_idx
        ) % vocab_size
    logits_storage = torch.full(
        (batch_size, 2, vocab_size),
        float("-inf"),
        dtype=torch.bfloat16,
        device=device,
    )

    markov_embed = markov_storage[:, 0]
    values = values_storage[:, 0]
    index = index_storage[:, 0]
    logits = logits_storage[:, 0]

    layer = nn.Module()
    layer.register_buffer("_nvfp4_weight_for_gather", packed_weight)
    layer.register_buffer("_nvfp4_weight_scale_for_gather", weight_scale)
    layer.register_buffer("_nvfp4_weight_global_scale_for_gather", global_scale)
    layer._nvfp4_group_size_for_gather = 16
    layer._retain_weight_for_gather = True
    layer.is_w4a16_nvfp4 = True
    head = DSparkMarkovHead.__new__(DSparkMarkovHead)
    nn.Module.__init__(head)
    head.markov_w2 = layer

    flat_index = index.reshape(-1)
    selected_weight = dequantize_to_dtype(
        packed_weight.index_select(0, flat_index),
        weight_scale.index_select(0, flat_index),
        global_scale,
        dtype=torch.bfloat16,
        block_size=16,
        swizzle=False,
    ).view(batch_size, topk, rank)
    expected = (
        values
        + torch.bmm(selected_weight, markov_embed.unsqueeze(-1)).squeeze(-1) * alpha
    )

    result = head.apply_bias_gathered(markov_embed, logits, values, index, alpha)

    torch.testing.assert_close(
        result.gather(1, index), expected, rtol=1e-2, atol=6.25e-2
    )
    selected = torch.zeros_like(result, dtype=torch.bool).scatter_(1, index, True)
    assert torch.isneginf(result.masked_select(~selected)).all()

    if batch_size == 1:
        logits.fill_(float("-inf"))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = head.apply_bias_gathered(
                markov_embed, logits, values, index, alpha
            )
        logits.fill_(float("-inf"))
        graph.replay()
        assert captured is logits
        torch.testing.assert_close(
            logits.gather(1, index), expected, rtol=1e-2, atol=6.25e-2
        )
