# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline-parallel layer map helpers for Mooncake bootstrap metadata."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

from vllm.model_executor.models.utils import extract_layer_index


class _ShardLike(Protocol):
    # Note: field types match `ShardInfo` exactly so structural matching
    # is unambiguous under mypy — Protocols compare invariantly for
    # mutable types and a Sequence[str] annotation here would not accept
    # ShardInfo's list[str].
    pp_rank: int
    start_layer: int
    end_layer: int
    registered_layer_names: list[str]
    registered_layer_group_ids: list[int]


@dataclass(frozen=True)
class MooncakePPLayerMap:
    pp_size: int
    boundaries: tuple[tuple[int, int], ...]

    @classmethod
    def from_shards(
        cls,
        shards: Iterable[_ShardLike],
        pp_size: int,
        total_num_hidden_layers: int,
        consumer_layer_name_to_group_id: Mapping[str, int],
    ) -> "MooncakePPLayerMap":
        if total_num_hidden_layers <= 0:
            raise ValueError(
                "total_num_hidden_layers must be positive; got "
                f"{total_num_hidden_layers}"
            )
        if pp_size <= 0:
            raise ValueError(f"pp_size must be positive; got {pp_size}")

        shard_list = list(shards)
        if not shard_list:
            raise ValueError("cannot build MooncakePPLayerMap without shards")

        boundaries_by_rank: dict[int, tuple[int, int]] = {}
        registered_by_rank: dict[int, tuple[tuple[str, ...], tuple[int, ...]]] = {}

        for shard in shard_list:
            if not 0 <= shard.pp_rank < pp_size:
                raise ValueError(f"pp_rank {shard.pp_rank} is outside [0, {pp_size})")
            valid_range = (
                0 <= shard.start_layer < shard.end_layer <= total_num_hidden_layers
            )
            if not valid_range:
                raise ValueError(
                    "metadata shard has invalid layer range "
                    f"[{shard.start_layer}, {shard.end_layer}) for total "
                    f"{total_num_hidden_layers}"
                )
            if len(shard.registered_layer_names) != len(
                shard.registered_layer_group_ids
            ):
                raise ValueError(
                    "registered_layer_names must align with "
                    f"registered_layer_group_ids for pp_rank {shard.pp_rank}"
                )

            for layer_name in shard.registered_layer_names:
                try:
                    layer_idx = extract_layer_index(layer_name)
                except Exception as exc:
                    raise ValueError(
                        f"could not parse layer index from {layer_name!r} "
                        f"for pp_rank {shard.pp_rank}"
                    ) from exc
                if not shard.start_layer <= layer_idx < shard.end_layer:
                    raise ValueError(
                        f"shard claims pp_rank={shard.pp_rank} "
                        f"[{shard.start_layer},{shard.end_layer}) but "
                        f"advertised {layer_name}"
                    )

            group_id_by_name: dict[str, int] = {}
            for layer_name, group_id in zip(
                shard.registered_layer_names, shard.registered_layer_group_ids
            ):
                prev_group_id = group_id_by_name.setdefault(layer_name, group_id)
                if prev_group_id != group_id:
                    raise ValueError(
                        "layer name maps to multiple group IDs within "
                        f"pp_rank {shard.pp_rank}: {layer_name!r} -> "
                        f"{prev_group_id} and {group_id}"
                    )

            boundary = (shard.start_layer, shard.end_layer)
            registered = (
                tuple(shard.registered_layer_names),
                tuple(shard.registered_layer_group_ids),
            )
            if shard.pp_rank in boundaries_by_rank:
                if (
                    boundaries_by_rank[shard.pp_rank] != boundary
                    or registered_by_rank[shard.pp_rank] != registered
                ):
                    expected_start, expected_end = boundaries_by_rank[shard.pp_rank]
                    expected_names, expected_group_ids = registered_by_rank[
                        shard.pp_rank
                    ]
                    raise ValueError(
                        "conflicting metadata shards for "
                        f"pp_rank {shard.pp_rank}: expected "
                        f"start_layer={expected_start}, end_layer={expected_end}, "
                        f"registered_layer_names={list(expected_names)}, "
                        f"registered_layer_group_ids={list(expected_group_ids)}; "
                        f"got start_layer={shard.start_layer}, "
                        f"end_layer={shard.end_layer}, "
                        "registered_layer_names="
                        f"{shard.registered_layer_names}, "
                        "registered_layer_group_ids="
                        f"{shard.registered_layer_group_ids}"
                    )
                continue

            boundaries_by_rank[shard.pp_rank] = boundary
            registered_by_rank[shard.pp_rank] = registered

        missing = [rank for rank in range(pp_size) if rank not in boundaries_by_rank]
        if missing:
            raise ValueError(f"missing metadata shards for pp_rank(s): {missing}")

        boundaries = tuple(boundaries_by_rank[rank] for rank in range(pp_size))
        expected_start = 0
        for pp_rank, (start, end) in enumerate(boundaries):
            if start != expected_start:
                raise ValueError(
                    "PP layer boundaries must be contiguous and cover the "
                    f"model; pp_rank {pp_rank} starts at {start}, expected "
                    f"{expected_start}"
                )
            expected_start = end
        if expected_start != total_num_hidden_layers:
            raise ValueError(
                "PP layer boundaries must cover all hidden layers; last end "
                f"{expected_start}, expected {total_num_hidden_layers}"
            )

        for shard in shard_list:
            for layer_name, producer_group_id in zip(
                shard.registered_layer_names, shard.registered_layer_group_ids
            ):
                consumer_group_id = consumer_layer_name_to_group_id.get(layer_name)
                if consumer_group_id is None:
                    continue
                if consumer_group_id != producer_group_id:
                    raise ValueError(
                        "Mooncake producer/consumer KV-cache group-id mismatch "
                        f"for {layer_name!r}: producer={producer_group_id}, "
                        f"consumer={consumer_group_id}"
                    )

        return cls(pp_size=pp_size, boundaries=boundaries)
