# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.pp_layer_map import (
    MooncakePPLayerMap,
)


@dataclass
class Shard:
    pp_rank: int
    tp_rank: int
    start_layer: int
    end_layer: int
    registered_layer_names: list[str]
    registered_layer_group_ids: list[int]


def _shard(
    pp_rank: int,
    start_layer: int,
    end_layer: int,
    names: list[str] | None = None,
    group_ids: list[int] | None = None,
    tp_rank: int = 0,
) -> Shard:
    if names is None:
        names = [f"layers.{idx}.attn" for idx in range(start_layer, end_layer)]
    if group_ids is None:
        group_ids = [0] * len(names)
    return Shard(
        pp_rank=pp_rank,
        tp_rank=tp_rank,
        start_layer=start_layer,
        end_layer=end_layer,
        registered_layer_names=names,
        registered_layer_group_ids=group_ids,
    )


def test_pp_size_round_trip_and_four_arg_signature():
    layer_map = MooncakePPLayerMap.from_shards(
        [_shard(0, 0, 2), _shard(1, 2, 4)],
        pp_size=2,
        total_num_hidden_layers=4,
        consumer_layer_name_to_group_id={
            "layers.0.attn": 0,
            "layers.1.attn": 0,
            "layers.2.attn": 0,
            "layers.3.attn": 0,
        },
    )

    assert layer_map.pp_size == 2
    assert layer_map.boundaries == ((0, 2), (2, 4))


@pytest.mark.parametrize(
    "shards,match",
    [
        ([], "without shards"),
        ([_shard(2, 0, 1)], "outside"),
        ([_shard(0, -1, 1)], "invalid layer range"),
        ([_shard(0, 0, 2, ["layers.0.attn"], [0, 1])], "must align"),
        ([_shard(0, 0, 1, ["layers.2.attn"], [0])], "advertised"),
    ],
)
def test_per_shard_well_formedness_rejections(shards, match):
    with pytest.raises(ValueError, match=match):
        MooncakePPLayerMap.from_shards(
            shards,
            pp_size=2,
            total_num_hidden_layers=2,
            consumer_layer_name_to_group_id={},
        )


def test_rejects_non_positive_dimensions():
    with pytest.raises(ValueError, match="total_num_hidden_layers"):
        MooncakePPLayerMap.from_shards(
            [_shard(0, 0, 1)],
            pp_size=1,
            total_num_hidden_layers=0,
            consumer_layer_name_to_group_id={},
        )

    with pytest.raises(ValueError, match="pp_size"):
        MooncakePPLayerMap.from_shards(
            [_shard(0, 0, 1)],
            pp_size=0,
            total_num_hidden_layers=1,
            consumer_layer_name_to_group_id={},
        )


def test_split_kv_duplicate_layer_names_are_accepted():
    layer_map = MooncakePPLayerMap.from_shards(
        [_shard(0, 0, 1, ["layers.0.attn", "layers.0.attn"], [0, 0])],
        pp_size=1,
        total_num_hidden_layers=1,
        consumer_layer_name_to_group_id={"layers.0.attn": 0},
    )

    assert layer_map.boundaries == ((0, 1),)


def test_layer_name_to_group_id_must_be_a_function_within_shard():
    with pytest.raises(ValueError, match="multiple group IDs"):
        MooncakePPLayerMap.from_shards(
            [_shard(0, 0, 1, ["layers.0.attn", "layers.0.attn"], [0, 1])],
            pp_size=1,
            total_num_hidden_layers=1,
            consumer_layer_name_to_group_id={},
        )


def test_duplicate_tp_collapse_accepts_identical_siblings():
    layer_map = MooncakePPLayerMap.from_shards(
        [
            _shard(0, 0, 2, tp_rank=0),
            _shard(0, 0, 2, tp_rank=1),
            _shard(1, 2, 4, tp_rank=0),
            _shard(1, 2, 4, tp_rank=1),
        ],
        pp_size=2,
        total_num_hidden_layers=4,
        consumer_layer_name_to_group_id={},
    )

    assert layer_map.boundaries == ((0, 2), (2, 4))


def test_duplicate_tp_collapse_rejects_mismatched_siblings():
    with pytest.raises(ValueError) as exc_info:
        MooncakePPLayerMap.from_shards(
            [
                _shard(0, 0, 2, tp_rank=0),
                _shard(
                    0,
                    0,
                    2,
                    ["layers.0.attn", "layers.1.swa"],
                    [0, 1],
                    tp_rank=1,
                ),
            ],
            pp_size=1,
            total_num_hidden_layers=2,
            consumer_layer_name_to_group_id={},
        )
    msg = str(exc_info.value)
    assert "conflicting metadata shards" in msg
    assert "registered_layer_names" in msg
    assert "registered_layer_group_ids" in msg
    assert "layers.1.swa" in msg


def test_coverage_rejection():
    with pytest.raises(ValueError, match="missing metadata shards"):
        MooncakePPLayerMap.from_shards(
            [_shard(0, 0, 2)],
            pp_size=2,
            total_num_hidden_layers=4,
            consumer_layer_name_to_group_id={},
        )


def test_contiguity_rejections():
    with pytest.raises(ValueError, match="starts at 3, expected 2"):
        MooncakePPLayerMap.from_shards(
            [_shard(0, 0, 2), _shard(1, 3, 4, ["layers.3.attn"], [0])],
            pp_size=2,
            total_num_hidden_layers=4,
            consumer_layer_name_to_group_id={},
        )

    with pytest.raises(ValueError, match="last end 3, expected 4"):
        MooncakePPLayerMap.from_shards(
            [_shard(0, 0, 2), _shard(1, 2, 3)],
            pp_size=2,
            total_num_hidden_layers=4,
            consumer_layer_name_to_group_id={},
        )


def test_group_id_agreement_with_consumer_mapping():
    with pytest.raises(ValueError, match="producer=1, consumer=0"):
        MooncakePPLayerMap.from_shards(
            [_shard(0, 0, 1, ["layers.0.attn"], [1])],
            pp_size=1,
            total_num_hidden_layers=1,
            consumer_layer_name_to_group_id={"layers.0.attn": 0},
        )
