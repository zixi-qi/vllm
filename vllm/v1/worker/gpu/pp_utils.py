# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline Parallelism utils for V2 Model Runner."""

import torch

import vllm.envs as envs
from vllm.distributed.parallel_state import get_pp_group


def pp_broadcast(
    sampled_token_ids: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
) -> None:
    if envs.VLLM_PP_DISABLE_SAMPLED_TOKEN_BROADCAST:
        return
    pp = get_pp_group()
    assert pp.is_last_rank

    assert sampled_token_ids.dtype == torch.int64
    torch.distributed.broadcast(
        sampled_token_ids.contiguous(), src=pp.last_rank, group=pp.device_group
    )

    combined = torch.stack((num_sampled, num_rejected), dim=0)
    torch.distributed.broadcast(combined, src=pp.last_rank, group=pp.device_group)


def pp_receive(
    num_reqs: int, max_sample_len: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pp = get_pp_group()
    assert not pp.is_last_rank

    sampled_tokens = torch.empty(
        num_reqs, max_sample_len, dtype=torch.int64, device=pp.device
    )
    if envs.VLLM_PP_DISABLE_SAMPLED_TOKEN_BROADCAST:
        # Skip both broadcasts. The postprocess kernel reads num_sampled and
        # num_rejected to advance num_computed_tokens, so synthesize zeros
        # locally — equivalent to the chunked-prefill triton zeroing path on
        # the last rank. sampled_tokens stays uninitialized and is gated by
        # `if num_sampled > 0` in the kernel. Safe only in prefill-only
        # PD-disagg engines (no decode steps consume the sampled token here).
        zeros = torch.zeros(num_reqs, dtype=torch.int32, device=pp.device)
        return sampled_tokens, zeros, zeros
    torch.distributed.broadcast(sampled_tokens, src=pp.last_rank, group=pp.device_group)

    combined = torch.empty(2, num_reqs, dtype=torch.int32, device=pp.device)
    torch.distributed.broadcast(combined, src=pp.last_rank, group=pp.device_group)
    num_sampled, num_rejected = combined.unbind(dim=0)
    return sampled_tokens, num_sampled, num_rejected
