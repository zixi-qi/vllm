# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline-parallel layer map helpers for NIXL metadata."""

from dataclasses import dataclass

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlAgentMetadata,
)


@dataclass(frozen=True)
class PPLayerMap:
    pp_size: int
    boundaries: tuple[tuple[int, int], ...]
    registered_layer_indices: tuple[tuple[int, ...], ...]
    total_num_hidden_layers: int

    @classmethod
    def from_metadata_shards(
        cls, metas: list[NixlAgentMetadata], total_num_hidden_layers: int
    ) -> "PPLayerMap":
        if total_num_hidden_layers <= 0:
            raise ValueError(
                "total_num_hidden_layers must be positive; got "
                f"{total_num_hidden_layers}"
            )
        if not metas:
            raise ValueError("cannot build PPLayerMap without metadata shards")

        pp_size = metas[0].pp_size
        if pp_size <= 0:
            raise ValueError(f"pp_size must be positive; got {pp_size}")

        boundaries_by_rank: dict[int, tuple[int, int]] = {}
        registered_by_rank: dict[int, tuple[int, ...]] = {}
        for meta in metas:
            if meta.pp_size != pp_size:
                raise ValueError(
                    "all metadata shards must advertise the same pp_size; "
                    f"got {meta.pp_size} and {pp_size}"
                )
            if not 0 <= meta.pp_rank < pp_size:
                raise ValueError(
                    f"pp_rank {meta.pp_rank} is outside [0, {pp_size})"
                )
            if not 0 <= meta.start_layer < meta.end_layer <= total_num_hidden_layers:
                raise ValueError(
                    "metadata shard has invalid layer range "
                    f"[{meta.start_layer}, {meta.end_layer}) for total "
                    f"{total_num_hidden_layers}"
                )
            if len(meta.registered_layer_indices) != len(meta.kv_caches_base_addr):
                raise ValueError(
                    "registered_layer_indices must align with kv_caches_base_addr "
                    f"for pp_rank {meta.pp_rank}"
                )
            if len(meta.registered_layer_indices) != len(meta.block_lens):
                raise ValueError(
                    "registered_layer_indices must align with block_lens "
                    f"for pp_rank {meta.pp_rank}"
                )
            if len(meta.registered_layer_indices) != len(
                meta.registered_layer_names
            ):
                raise ValueError(
                    "registered_layer_indices must align with "
                    f"registered_layer_names for pp_rank {meta.pp_rank}"
                )
            registered_layers = tuple(meta.registered_layer_indices)
            if any(
                layer < meta.start_layer or layer >= meta.end_layer
                for layer in registered_layers
            ):
                raise ValueError(
                    "registered_layer_indices must be within the advertised "
                    f"layer range for pp_rank {meta.pp_rank}"
                )

            boundary = (meta.start_layer, meta.end_layer)
            if meta.pp_rank in boundaries_by_rank:
                if (
                    boundaries_by_rank[meta.pp_rank] != boundary
                    or registered_by_rank[meta.pp_rank] != registered_layers
                ):
                    raise ValueError(
                        "conflicting metadata shards for pp_rank "
                        f"{meta.pp_rank}"
                    )
                continue

            boundaries_by_rank[meta.pp_rank] = boundary
            registered_by_rank[meta.pp_rank] = registered_layers

        missing = [rank for rank in range(pp_size) if rank not in boundaries_by_rank]
        if missing:
            raise ValueError(f"missing metadata shards for pp_rank(s): {missing}")

        boundaries = tuple(boundaries_by_rank[rank] for rank in range(pp_size))
        registered_layer_indices = tuple(
            registered_by_rank[rank] for rank in range(pp_size)
        )
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

        return cls(
            pp_size=pp_size,
            boundaries=boundaries,
            registered_layer_indices=registered_layer_indices,
            total_num_hidden_layers=total_num_hidden_layers,
        )

    def producer_pp_rank_for_global_layer(self, g: int) -> int:
        if not 0 <= g < self.total_num_hidden_layers:
            raise ValueError(
                f"global layer {g} is outside [0, {self.total_num_hidden_layers})"
            )
        for pp_rank, (start, end) in enumerate(self.boundaries):
            if start <= g < end:
                return pp_rank
        raise ValueError(f"global layer {g} is not covered by any PP shard")

    def producer_local_idx(self, g: int, pp_rank: int) -> int:
        start, end = self._producer_boundary(pp_rank)
        if not start <= g < end:
            raise ValueError(
                f"global layer {g} is outside pp_rank {pp_rank} range "
                f"[{start}, {end})"
            )
        return g - start

    def producer_layer_range(self, pp_rank: int) -> range:
        start, end = self._producer_boundary(pp_rank)
        return range(start, end)

    def producer_registered_layer_indices(self, pp_rank: int) -> tuple[int, ...]:
        self._check_pp_rank(pp_rank)
        return self.registered_layer_indices[pp_rank]

    def producer_registered_positions(self, pp_rank: int, g: int) -> tuple[int, ...]:
        start, end = self._producer_boundary(pp_rank)
        if not start <= g < end:
            raise ValueError(
                f"global layer {g} is outside pp_rank {pp_rank} range "
                f"[{start}, {end})"
            )
        return tuple(
            idx
            for idx, layer in enumerate(self.registered_layer_indices[pp_rank])
            if layer == g
        )

    def _producer_boundary(self, pp_rank: int) -> tuple[int, int]:
        self._check_pp_rank(pp_rank)
        return self.boundaries[pp_rank]

    def _check_pp_rank(self, pp_rank: int) -> None:
        if not 0 <= pp_rank < self.pp_size:
            raise ValueError(f"pp_rank {pp_rank} is outside [0, {self.pp_size})")
