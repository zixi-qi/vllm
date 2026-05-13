# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict

import msgspec

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NIXL_CONNECTOR_VERSION,
    NixlAgentMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
    _ShardDescLayout,
)

from tests.v1.kv_connector.nixl_integration.test_consumer_shard_refactor import (
    REMOTE_ENGINE_ID,
    _make_worker,
)
from tests.v1.kv_connector.nixl_integration.test_pp_layer_map import _meta


def _attn(layer_idx: int) -> str:
    return f"model.layers.{layer_idx}.attn"


def _swa(layer_idx: int) -> str:
    return f"model.layers.{layer_idx}.attn.swa_cache"


def _compressor(layer_idx: int) -> str:
    return f"model.layers.{layer_idx}.attn.compressor.state_cache"


def _configure_hma_worker(layer_names: list[str], group_ids: list[int]):
    worker = _make_worker(total_layers=128)
    worker.local_registered_layer_names = layer_names
    worker.local_registered_layer_indices = [
        int(layer_name.split(".")[2]) for layer_name in layer_names
    ]
    worker.block_len_per_layer = [1024] * len(layer_names)
    worker.kv_caches_base_addr[worker.engine_id][worker._local_kv_cache_key] = [
        100_000 + i * 10_000 for i in range(len(layer_names))
    ]
    worker._layer_name_to_kv_group_index = dict(zip(layer_names, group_ids))
    worker._is_hma_required = True
    return worker


def test_asymmetric_dsv4_pool_case_resolves_by_layer_name():
    local_layer_names = [
        _attn(15),
        _compressor(13),
        _swa(14),
        _swa(15),
        _attn(16),
        _compressor(16),
        _swa(16),
        _swa(17),
        _attn(18),
        _compressor(18),
    ]
    worker = _configure_hma_worker(local_layer_names, [0] * len(local_layer_names))

    producer_region_names = [
        _compressor(15),
        _swa(15),
        _attn(16),
        _compressor(16),
        _swa(16),
    ]
    # The old pool-subset matcher failed this shape because these names span
    # two decode-side HMA pools. Per-layer regions need only exact names.
    worker.local_registered_layer_names.insert(1, _compressor(15))
    worker.block_len_per_layer.insert(1, 1024)
    worker.kv_caches_base_addr[worker.engine_id][worker._local_kv_cache_key].insert(
        1, 110_000
    )
    worker._layer_name_to_kv_group_index[_compressor(15)] = 0

    assert worker._local_region_indices_for_layer_names(producer_region_names) == [
        worker.local_registered_layer_names.index(name)
        for name in producer_region_names
    ]


def test_descriptor_ids_are_per_layer_and_kv_group_specific():
    layer_names = [_attn(15), _swa(15), _attn(16), _compressor(16)]
    worker = _configure_hma_worker(layer_names, [0, 1, 0, 2])
    worker._xfer_desc_layouts[(REMOTE_ENGINE_ID, 1, "local")] = _ShardDescLayout(
        num_regions=4,
        num_blocks=10,
        region_group_ids=(0, 1, 0, 2),
    )

    desc_ids = worker._get_block_descs_ids_for_shard(
        REMOTE_ENGINE_ID,
        1,
        "local",
        ([1, 2], [7], [4, 5]),
    )

    assert desc_ids.tolist() == [1, 2, 17, 21, 22, 34, 35]


def test_repeated_layer_name_uses_matching_occurrence_for_split_regions():
    layer_name = "model.layers.3.self_attn"
    worker = _configure_hma_worker([layer_name, layer_name], [0, 0])

    assert worker._local_region_indices_for_layer_names([layer_name, layer_name]) == [
        0,
        1,
    ]


def test_nixl_agent_metadata_v5_registered_layer_names_round_trip():
    meta = _meta(
        0,
        0,
        4,
        pp_size=1,
        registered_layer_indices=[0, 2],
        registered_layer_names=[
            "model.layers.0.self_attn",
            "model.layers.2.indexer",
        ],
    )

    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(meta), type=NixlAgentMetadata
    )

    assert NIXL_CONNECTOR_VERSION == 5
    assert decoded.registered_layer_names == [
        "model.layers.0.self_attn",
        "model.layers.2.indexer",
    ]


def test_shard_local_handler_uses_registered_layer_names():
    layer_names = [_attn(15), _swa(15), _attn(16)]
    worker = _configure_hma_worker(layer_names, [0, 1, 0])
    worker.src_xfer_handles_by_remote = {}
    worker.src_blocks_data_by_remote = {}
    worker._xfer_desc_layouts = {}
    worker.nixl_wrapper._next_handle = 0
    worker._remote_agent_metadata = defaultdict(dict)

    meta = _meta(
        1,
        15,
        17,
        pp_size=4,
        registered_layer_indices=[15, 16],
        registered_layer_names=[_swa(15), _attn(16)],
    )
    worker._remote_agent_metadata[REMOTE_ENGINE_ID][(1, 0)] = meta

    _, blocks_data, layout = worker._register_local_xfer_handler_for_shard(
        REMOTE_ENGINE_ID, 1, worker.block_size
    )

    assert layout.region_group_ids == (1, 1, 0, 0)
    assert [blocks_data[i][0] for i in (0, 8)] == [110_000, 120_000]
