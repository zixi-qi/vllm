# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import contextlib
import queue
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import msgspec
import pytest
import torch
import zmq.asyncio

from vllm.config import set_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    KVConnectorRole,
    MooncakeConnector,
    MooncakeConnectorMetadata,
    MooncakeConnectorWorker,
    MooncakeXferMetadata,
    MooncakeXferResponse,
    MooncakeXferResponseStatus,
    PullReqMeta,
    SendBlockMeta,
    TransferRegion,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
    MooncakeBootstrapServer,
    ShardInfo,
)
from vllm.utils.network_utils import get_open_port
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import RequestStatus

from .utils import create_request, create_scheduler, create_vllm_config


def _make_test_kv_cache_config() -> KVCacheConfig:
    return KVCacheConfig(num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[])


def _make_layered_kv_cache_config(
    groups: list[list[str]],
    block_size: int = 16,
) -> KVCacheConfig:
    kv_cache_groups = []
    for group_idx, layer_names in enumerate(groups):
        spec = (
            FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=4,
                head_size=16,
                dtype=torch.float16,
            )
            if group_idx == 0
            else SlidingWindowSpec(
                block_size=block_size,
                num_kv_heads=4,
                head_size=16,
                dtype=torch.float16,
                sliding_window=128,
            )
        )
        kv_cache_groups.append(KVCacheGroupSpec(layer_names, spec))
    return KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=kv_cache_groups,
    )


class FakeMooncakeWrapper:
    """Mock Mooncake TransferEngine for unit testing environments."""

    def __init__(self, *args, **kwargs):
        pass

    def initialize(self, local_hostname, metadata_server, protocol, device_name) -> int:
        return 0

    def get_rpc_port(self) -> int:
        return 12345

    def batch_transfer_sync_write(
        self, target_hostname, buffers, peer_buffer_addresses, lengths
    ) -> int:
        return 0

    def batch_register_memory(self, buffer_addresses, capacities) -> int:
        return 0


def test_basic_interface():
    """Unit test for basic MooncakeConnector interface functionality."""

    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_consumer"
    )
    scheduler = create_scheduler(vllm_config)

    # 2 Full Blocks and 1 Half Block.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_EXTERNAL_FULL_BLOCKS = 2
    NUM_TOKENS = int(BLOCK_SIZE * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))

    request = create_request(
        request_id=1,
        block_size=BLOCK_SIZE,
        num_tokens=NUM_TOKENS,
        do_remote_prefill=True,
    )
    request_id = request.request_id
    request.kv_transfer_params.update(
        {
            "transfer_id": request_id,
            "remote_bootstrap_addr": 54321,
        }
    )

    scheduler.add_request(request)

    # Remote Prefill, triggers NixlConnectorMetadata.
    scheduler_output = scheduler.schedule()
    kv_connector_metadata = scheduler_output.kv_connector_metadata
    assert kv_connector_metadata is not None
    assert isinstance(kv_connector_metadata, MooncakeConnectorMetadata)

    assert len(kv_connector_metadata.reqs_to_recv) == 1
    assert request_id in kv_connector_metadata.reqs_to_recv["my-engine-id"]
    req_meta = kv_connector_metadata.reqs_to_recv["my-engine-id"][request_id]

    # local_block_ids is list[list[int]] (per-group); flatten for comparison.
    all_block_ids = [bid for group in req_meta.local_block_ids for bid in group]
    for block_id, block in zip(
        all_block_ids,
        scheduler.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks[
            request_id
        ],
    ):
        assert block_id == block.block_id


def test_prompt_less_than_block_size():
    """Test that we can handle case where prompt is < block."""

    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_consumer"
    )
    scheduler = create_scheduler(vllm_config)

    # Half of a block.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_TOKENS = int(BLOCK_SIZE * 0.5)

    # Request will have 1 partial remote block.
    request = create_request(
        request_id=1,
        block_size=BLOCK_SIZE,
        num_tokens=NUM_TOKENS,
        do_remote_prefill=True,
        num_remote_blocks=1,
    )
    request.kv_transfer_params.update(
        {
            "transfer_id": request.request_id,
            "remote_bootstrap_addr": 54321,
        }
    )

    scheduler.add_request(request)
    scheduler_output = scheduler.schedule()

    # This request will read async.
    kv_connector_metadata = scheduler_output.kv_connector_metadata
    assert kv_connector_metadata is not None
    assert isinstance(kv_connector_metadata, MooncakeConnectorMetadata)
    assert len(kv_connector_metadata.reqs_to_recv["my-engine-id"]) == 1
    assert len(scheduler_output.scheduled_new_reqs) == 0


@pytest.fixture
def bootstrap_server():
    """Fixture to launch and cleanup a Mooncake Bootstrap HTTP Server."""

    port = get_open_port()
    server = MooncakeBootstrapServer("127.0.0.1", port)
    server.start()
    yield server
    server.shutdown()


@pytest.mark.asyncio
async def test_bootstrap_server(bootstrap_server: MooncakeBootstrapServer):
    """
    Tests the bootstrap server's api for worker registration and querying.

    Validates DP/TP/PP rank indexing and replacement on duplicate registration.
    """

    import httpx

    base_url = f"http://127.0.0.1:{bootstrap_server.port}"

    # Query when empty
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{base_url}/query")
        assert response.status_code == 200
        assert response.json() == {}

    # Register a worker
    payload1 = {
        "engine_id": "eng-1",
        "dp_rank": 0,
        "tp_rank": 0,
        "pp_rank": 0,
        "addr": "tcp://1.1.1.1:1111",
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{base_url}/register", json=payload1)
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    # Query after registration
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{base_url}/query")
        assert response.status_code == 200
        data = response.json()
        assert "0" in data
        assert data["0"]["engine_id"] == "eng-1"
        assert data["0"]["worker_addr"]["0"]["0"] == "tcp://1.1.1.1:1111"
        assert data["0"]["pp_size"] == 1
        assert data["0"]["tp_size"] == 1
        assert data["0"]["shard_info"]["0"]["0"]["addr"] == "tcp://1.1.1.1:1111"

    # Re-registering the same worker replaces both legacy and enriched maps.
    replacement = dict(payload1)
    replacement["addr"] = "tcp://1.1.1.1:2222"
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{base_url}/register", json=replacement)
        assert response.status_code == 200
        response = await client.get(f"{base_url}/query")
        data = response.json()
        assert data["0"]["worker_addr"]["0"]["0"] == "tcp://1.1.1.1:2222"
        assert data["0"]["shard_info"]["0"]["0"]["addr"] == "tcp://1.1.1.1:2222"

    # Test failure: engine_id mismatch for same dp_rank
    payload3_fail = {
        "engine_id": "eng-2",
        "dp_rank": 0,
        "tp_rank": 1,
        "pp_rank": 0,
        "addr": "tcp://3.3.3.3:3333",
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{base_url}/register", json=payload3_fail)
        assert response.status_code == 400
        assert "Engine ID mismatch" in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload_update,expected",
    [
        ({"pp_rank": 1, "pp_size": 1}, "pp_rank=1"),
        ({"tp_rank": 1, "tp_size": 1}, "tp_rank=1"),
    ],
)
async def test_bootstrap_rank_ingest_validation(
    bootstrap_server: MooncakeBootstrapServer, payload_update, expected
):
    import httpx

    payload = {
        "engine_id": "eng-rank-check",
        "dp_rank": 0,
        "tp_rank": 0,
        "pp_rank": 0,
        "addr": "tcp://1.1.1.1:1111",
    }
    payload.update(payload_update)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"http://127.0.0.1:{bootstrap_server.port}/register",
            json=payload,
        )

    assert response.status_code == 400
    assert expected in response.text


@pytest.mark.asyncio
async def test_bootstrap_topology_mismatch_rejects_409(
    bootstrap_server: MooncakeBootstrapServer,
):
    import httpx

    base_url = f"http://127.0.0.1:{bootstrap_server.port}"
    payload = {
        "engine_id": "eng-topology",
        "dp_rank": 0,
        "tp_rank": 0,
        "pp_rank": 0,
        "addr": "tcp://1.1.1.1:1111",
        "tp_size": 2,
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(f"{base_url}/register", json=payload)
        assert response.status_code == 200
        payload["tp_rank"] = 1
        payload["tp_size"] = 3
        payload["addr"] = "tcp://1.1.1.1:2222"
        response = await client.post(f"{base_url}/register", json=payload)

    assert response.status_code == 409
    assert "topology mismatch" in response.text


@pytest.mark.asyncio
async def test_connect_to_prefiller_bootstrap_modern_and_legacy_paths():
    worker = _minimal_worker_for_bootstrap()
    modern_payload = {
        "0": {
            "engine_id": "modern-engine",
            "worker_addr": {"0": {"0": "tcp://modern:1234"}},
            "pp_size": 1,
            "tp_size": 1,
            "shard_info": {
                "0": {
                    "0": {
                        "addr": "tcp://modern:1234",
                        "pp_rank": 0,
                        "tp_rank": 0,
                        "start_layer": 0,
                        "end_layer": 2,
                        "registered_layer_names": [
                            "layers.0.attn",
                            "layers.1.attn",
                        ],
                        "registered_layer_group_ids": [0, 0],
                    }
                }
            },
        }
    }

    legacy_payload = {
        "0": {
            "engine_id": "legacy-engine",
            "worker_addr": {
                "0": {"0": "tcp://legacy0:1234"},
                "1": {"0": "tcp://legacy1:1234"},
            },
        }
    }

    mock_client = AsyncMock()
    mock_client.get.side_effect = [
        _BootstrapResponse(modern_payload),
        _BootstrapResponse(legacy_payload),
    ]
    with patch("httpx.AsyncClient") as mock_async_client:
        mock_async_client.return_value.__aenter__.return_value = mock_client
        await worker._connect_to_prefiller_bootstrap("http://bootstrap")
        await worker._connect_to_prefiller_bootstrap("http://bootstrap")

    modern_shard = worker._remote_agents["modern-engine"][0][0]
    assert modern_shard.addr == "tcp://modern:1234"
    assert worker._tp_size["modern-engine"] == 1
    assert worker._remote_engine_pp_size["modern-engine"] == 1
    assert worker._pp_layer_map["modern-engine"].boundaries == ((0, 2),)

    assert worker._remote_agents["legacy-engine"][1][0] == "tcp://legacy1:1234"
    assert worker._tp_size["legacy-engine"] == 2
    assert worker._remote_engine_pp_size["legacy-engine"] == 1
    assert worker._pp_layer_map["legacy-engine"] is None


@pytest.mark.asyncio
async def test_connect_to_prefiller_bootstrap_rejects_legacy_pp_shape():
    worker = _minimal_worker_for_bootstrap()
    legacy_pp_payload = {
        "0": {
            "engine_id": "legacy-pp-engine",
            "worker_addr": {"0": {"0": "tcp://p0:1234", "1": "tcp://p1:1234"}},
        }
    }
    mock_client = AsyncMock()
    mock_client.get.return_value = _BootstrapResponse(legacy_pp_payload)
    with patch("httpx.AsyncClient") as mock_async_client:
        mock_async_client.return_value.__aenter__.return_value = mock_client
        with pytest.raises(RuntimeError, match="multiple pp_rank"):
            await worker._connect_to_prefiller_bootstrap("http://bootstrap")


@pytest.mark.asyncio
@pytest.mark.parametrize("pp_size", [2, 4])
async def test_connect_to_prefiller_bootstrap_walks_all_pp_shards(pp_size):
    worker = _minimal_worker_for_bootstrap()
    _configure_bootstrap_worker_for_pp(worker, pp_size)
    payload = _make_pp_bootstrap_payload(
        "pp-engine",
        pp_size=pp_size,
        tp_size=2,
    )

    mock_client = AsyncMock()
    mock_client.get.return_value = _BootstrapResponse(payload)
    with patch("httpx.AsyncClient") as mock_async_client:
        mock_async_client.return_value.__aenter__.return_value = mock_client
        await worker._connect_to_prefiller_bootstrap("http://bootstrap")

    assert worker._tp_size["pp-engine"] == 2
    assert worker._remote_engine_pp_size["pp-engine"] == pp_size
    assert worker._pp_layer_map["pp-engine"].boundaries == tuple(
        (idx, idx + 1) for idx in range(pp_size)
    )
    for pp_rank in range(pp_size):
        for tp_rank in range(2):
            shard = worker._remote_agents["pp-engine"][tp_rank][pp_rank]
            assert isinstance(shard, ShardInfo)
            assert shard.addr == f"tcp://p{pp_rank}-t{tp_rank}:1234"


@pytest.mark.asyncio
async def test_readiness_barrier_rejects_partial_tp_race(monkeypatch):
    worker = _minimal_worker_for_bootstrap()
    _configure_bootstrap_worker_for_pp(worker, pp_size=2)
    worker._bootstrap_ready_timeout_s = 0
    # Regression for the round-1 draft bug: len(entry.worker_addr) would see
    # one TP rank with both PP shards and incorrectly declare the topology ready.
    payload = _make_pp_bootstrap_payload(
        "partial-tp-engine",
        pp_size=2,
        tp_size=2,
        shard_keys={(0, 0), (0, 1)},
        worker_keys={(0, 0), (0, 1)},
    )
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
        "mooncake_connector.MOONCAKE_BOOTSTRAP_READY_POLL_S",
        0,
    )

    mock_client = AsyncMock()
    mock_client.get.return_value = _BootstrapResponse(payload)
    with patch("httpx.AsyncClient") as mock_async_client:
        mock_async_client.return_value.__aenter__.return_value = mock_client
        with pytest.raises(RuntimeError) as exc_info:
            await worker._connect_to_prefiller_bootstrap("http://bootstrap")

    msg = str(exc_info.value)
    assert "missing=[(1, 0), (1, 1)]" in msg
    assert "never_registered=[(1, 0), (1, 1)]" in msg


@pytest.mark.asyncio
async def test_readiness_barrier_error_message_splits_missing_keys(monkeypatch):
    worker = _minimal_worker_for_bootstrap()
    _configure_bootstrap_worker_for_pp(worker, pp_size=2)
    worker._bootstrap_ready_timeout_s = 0
    payload = _make_pp_bootstrap_payload(
        "split-missing-engine",
        pp_size=2,
        tp_size=2,
        shard_keys={(0, 0)},
        worker_keys={(0, 0), (0, 1)},
    )
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
        "mooncake_connector.MOONCAKE_BOOTSTRAP_READY_POLL_S",
        0,
    )

    mock_client = AsyncMock()
    mock_client.get.return_value = _BootstrapResponse(payload)
    with patch("httpx.AsyncClient") as mock_async_client:
        mock_async_client.return_value.__aenter__.return_value = mock_client
        with pytest.raises(RuntimeError) as exc_info:
            await worker._connect_to_prefiller_bootstrap("http://bootstrap")

    msg = str(exc_info.value)
    assert "legacy_only=[(0, 1)]" in msg
    assert "never_registered=[(1, 0), (1, 1)]" in msg


def test_scheduler_request_finished():
    """
    Tests the scheduler-side logic when a request finishes.

    Differentiates between 'Finished' (requires transfer)
    and 'Aborted' (immediate free).
    """

    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_producer"
    )
    scheduler = create_scheduler(vllm_config)
    scheduler_connector = scheduler.get_kv_connector().connector_scheduler

    request = create_request(request_id=1, do_remote_decode=True)
    request.kv_transfer_params["transfer_id"] = request.request_id

    # Case: Capped length (Successful prefill, need to send to decoder)
    request.status = RequestStatus.FINISHED_LENGTH_CAPPED
    delay_free, _ = scheduler_connector.request_finished(request, block_ids=([10, 11],))
    assert delay_free is True
    assert "id-1" in scheduler_connector._reqs_need_send
    assert scheduler_connector._reqs_need_send["id-1"][1] == [[10, 11]]

    # Case: Aborted (No need to transfer, free blocks immediately)
    scheduler_connector._reqs_need_send.clear()
    request.status = RequestStatus.FINISHED_ABORTED
    delay_free, _ = scheduler_connector.request_finished(request, block_ids=([12],))
    assert delay_free is False
    assert len(scheduler_connector._reqs_need_send) == 0
    assert "id-1" in scheduler_connector._reqs_not_processed


@contextlib.contextmanager
def patch_worker_dependencies():
    """Helper to mock all distributed and network dependencies for Worker tests."""

    with (
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.TransferEngine",
            FakeMooncakeWrapper,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.get_ip",
            return_value="127.0.0.1",
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.get_tensor_model_parallel_rank",
            return_value=0,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.get_pp_group"
        ) as mock_pp,
        patch("vllm.distributed.parallel_state.is_local_first_rank", return_value=True),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.should_launch_bootstrap_server",
            return_value=False,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.make_zmq_socket"
        ) as mock_make_zmq,
        patch("httpx.AsyncClient") as mock_async_client,
    ):
        # Mock PP group
        mock_pp_group = MagicMock()
        mock_pp_group.rank_in_group = 0
        mock_pp.return_value = mock_pp_group

        # Mock ZMQ socket
        mock_socket_object = AsyncMock()
        mock_socket_object.setsockopt = MagicMock()
        mock_socket_ctx = MagicMock()
        mock_socket_ctx.__enter__.return_value = mock_socket_object
        mock_make_zmq.return_value = mock_socket_ctx

        # Mock httpx client
        mock_http_client_instance = AsyncMock()
        mock_async_client.return_value = mock_http_client_instance

        yield {
            "mock_make_zmq": mock_make_zmq,
            "mock_socket_object": mock_socket_object,
            "mock_async_client": mock_async_client,
            "mock_http_client": mock_http_client_instance,
        }


def _minimal_worker_for_accounting():
    worker = object.__new__(MooncakeConnectorWorker)
    worker.async_zmq_ctx = MagicMock()
    worker.is_kv_consumer = True
    worker.is_kv_producer = True
    worker.finished_recving_reqs = set()
    worker._failed_block_ids = queue.Queue()
    worker.xfer_stats = MagicMock()
    return worker


def _minimal_worker_for_bootstrap():
    worker = object.__new__(MooncakeConnectorWorker)
    worker.async_zmq_ctx = MagicMock()
    worker.is_kv_consumer = True
    worker.is_kv_producer = True
    worker._remote_agents = {}
    worker._remote_engine_pp_size = {}
    worker._tp_size = {}
    worker._pp_layer_map = {}
    worker._pending_bootstrap_queries = {}
    worker._bootstrap_ready_timeout_s = 60
    worker._last_requery_time = {}
    worker._remote_bootstrap_addrs = {}
    worker.total_num_hidden_layers = 2
    worker._layer_name_to_group_id = {
        "layers.0.attn": 0,
        "layers.1.attn": 0,
    }
    return worker


class _BootstrapResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _make_pp_bootstrap_payload(
    engine_id: str,
    pp_size: int,
    tp_size: int,
    *,
    shard_keys: set[tuple[int, int]] | None = None,
    worker_keys: set[tuple[int, int]] | None = None,
) -> dict[str, dict]:
    if shard_keys is None:
        shard_keys = {
            (tp_rank, pp_rank)
            for tp_rank in range(tp_size)
            for pp_rank in range(pp_size)
        }
    if worker_keys is None:
        worker_keys = set(shard_keys)

    worker_addr: dict[str, dict[str, str]] = {}
    shard_info: dict[str, dict[str, dict]] = {}
    for tp_rank, pp_rank in sorted(worker_keys):
        worker_addr.setdefault(str(tp_rank), {})[str(pp_rank)] = (
            f"tcp://p{pp_rank}-t{tp_rank}:1234"
        )
    for tp_rank, pp_rank in sorted(shard_keys):
        layer_name = f"layers.{pp_rank}.attn"
        shard_info.setdefault(str(tp_rank), {})[str(pp_rank)] = {
            "addr": f"tcp://p{pp_rank}-t{tp_rank}:1234",
            "pp_rank": pp_rank,
            "tp_rank": tp_rank,
            "start_layer": pp_rank,
            "end_layer": pp_rank + 1,
            "registered_layer_names": [layer_name],
            "registered_layer_group_ids": [0],
        }

    return {
        "0": {
            "engine_id": engine_id,
            "worker_addr": worker_addr,
            "pp_size": pp_size,
            "tp_size": tp_size,
            "shard_info": shard_info,
        }
    }


def _configure_bootstrap_worker_for_pp(
    worker: MooncakeConnectorWorker, pp_size: int
) -> None:
    worker.total_num_hidden_layers = pp_size
    worker._layer_name_to_group_id = {f"layers.{idx}.attn": 0 for idx in range(pp_size)}


@pytest.mark.asyncio
@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.TransferEngine",
    FakeMooncakeWrapper,
)
async def test_kv_producer(monkeypatch):
    """
    Simulates a Producer Worker (Prefiller) receiving a transfer request
    from a Consumer (Decoder).

    Verifies memory offset calculation: ptr = base_addr + block_id * block_len.
    """

    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_producer"
    )

    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        prefill_connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            _make_test_kv_cache_config(),
        )
        prefill_worker = prefill_connector.connector_worker
        prefill_worker.kv_caches_base_addr = [0x1000]
        block_len = 4096
        prefill_worker.block_len_per_layer = [block_len]

        # Override loop to use current test loop
        origin_sender_loop = prefill_worker.sender_loop
        prefill_worker.sender_loop = asyncio.get_event_loop()

        # A request is finished on Producer and ready to be sent.
        transfer_id = "xfer-req-1"
        send_meta = SendBlockMeta(
            p_req_id="p-req-1",
            transfer_id=transfer_id,
            local_block_ids=[[10, 11]],
            ready=asyncio.Event(),
        )
        prefill_worker.reqs_need_send[transfer_id] = send_meta
        send_meta.ready.set()

        # Remote consumer request metadata
        xfer_meta = MooncakeXferMetadata(
            remote_hostname="consumer-host",
            remote_port=54321,
            remote_tp_size=1,
            remote_tp_rank=0,
            req_blocks={"d-req-1": (transfer_id, [[20, 21]])},
            kv_caches_base_addr=[0x2000],
            block_lens=[block_len],
        )

        mock_socket = AsyncMock(spec=zmq.asyncio.Socket)
        mock_socket.send_multipart = AsyncMock()
        identity = b"consumer-id"

        with patch.object(
            prefill_worker, "_send_blocks", return_value=0
        ) as mock_send_blocks:
            # With blocks-first layout, each block is virtually split
            # into K and V halves, producing non-coalesced transfers.
            kv_half = block_len // 2

            def expected_split_transfers(src_base, dst_base, src_blocks, dst_blocks):
                """Build expected (src_ptrs, dst_ptrs, lengths) for
                virtual-split K/V transfers."""
                src_ptrs, dst_ptrs, lengths = [], [], []
                for kv_offset in (0, kv_half):
                    for sb, db in zip(src_blocks, dst_blocks):
                        src_ptrs.append(src_base + sb * block_len + kv_offset)
                        dst_ptrs.append(dst_base + db * block_len + kv_offset)
                        lengths.append(kv_half)
                return src_ptrs, dst_ptrs, lengths

            # Normal case: 2 blocks to 2 blocks
            await prefill_worker.send_kv_to_decode(identity, mock_socket, xfer_meta)
            src, dst, lens = expected_split_transfers(
                0x1000, 0x2000, [10, 11], [20, 21]
            )
            mock_send_blocks.assert_called_once_with(
                "consumer-host:54321",
                src,
                dst,
                lens,
            )
            mock_socket.send_multipart.assert_called_once()

            # Verify the response sent back to the consumer
            sent_call = mock_socket.send_multipart.call_args[0][0]
            sent_identity, sent_payload = sent_call
            assert sent_identity == identity
            response = prefill_worker._xfer_resp_decoder.decode(sent_payload)
            assert response.status == MooncakeXferResponseStatus.FINISH
            assert response.ok_reqs == ["d-req-1"]

            # Verify internal state cleanup
            assert transfer_id not in prefill_worker.reqs_need_send
            assert "p-req-1" in prefill_worker.finished_sending_reqs

            # More cases:
            # Consumer only needs 1 block (less than P)
            mock_send_blocks.reset_mock()
            mock_socket.send_multipart.reset_mock()
            prefill_worker.reqs_need_send[transfer_id] = send_meta
            send_meta.sent = 0
            send_meta.ready.set()
            xfer_meta.req_blocks["d-req-1"] = (transfer_id, [[20]])
            # Worker processes the consumer's request
            await prefill_worker.send_kv_to_decode(identity, mock_socket, xfer_meta)
            # Verify transfer parameters are correct: 11 to 20
            src, dst, lens = expected_split_transfers(0x1000, 0x2000, [11], [20])
            mock_send_blocks.assert_called_once_with(
                "consumer-host:54321",
                src,
                dst,
                lens,
            )
            mock_socket.send_multipart.assert_called_once()

            # Consumer needs 3 blocks (more than P, error case)
            mock_send_blocks.reset_mock()
            mock_socket.send_multipart.reset_mock()
            prefill_worker.reqs_need_send[transfer_id] = send_meta
            send_meta.sent = 0
            send_meta.ready.set()
            xfer_meta.req_blocks["d-req-1"] = (transfer_id, [[20, 21, 22]])
            # Worker processes the consumer's request
            await prefill_worker.send_kv_to_decode(identity, mock_socket, xfer_meta)
            # This should not be called because error.
            mock_send_blocks.assert_not_called()
            mock_socket.send_multipart.assert_called_once()
            _, sent_payload = mock_socket.send_multipart.call_args[0][0]
            response = prefill_worker._xfer_resp_decoder.decode(sent_payload)
            assert response.err_msg == "P num blocks less than D"
            assert response.err_reqs == ["d-req-1"]

            # Timeout
            mock_send_blocks.reset_mock()
            mock_socket.send_multipart.reset_mock()
            prefill_worker.reqs_need_send[transfer_id] = send_meta
            send_meta.sent = 0
            send_meta.ready.clear()
            xfer_meta.req_blocks["d-req-1"] = (transfer_id, [[20, 21]])
            # Worker processes the consumer's request
            await prefill_worker.send_kv_to_decode(identity, mock_socket, xfer_meta)
            # This should not be called because timeout.
            mock_send_blocks.assert_not_called()
            mock_socket.send_multipart.assert_called_once()
            _, sent_payload = mock_socket.send_multipart.call_args[0][0]
            response = prefill_worker._xfer_resp_decoder.decode(sent_payload)
            assert response.err_msg == "Timeout waiting for P side ready."
            assert response.err_reqs == ["d-req-1"]

        # Transfer error
        with patch.object(
            prefill_worker, "_send_blocks", return_value=123
        ) as mock_send_blocks:
            mock_socket.send_multipart.reset_mock()
            prefill_worker.reqs_need_send[transfer_id] = send_meta
            send_meta.sent = 0
            send_meta.ready.set()
            xfer_meta.req_blocks["d-req-1"] = (transfer_id, [[20, 21]])
            # Worker processes the consumer's request
            await prefill_worker.send_kv_to_decode(identity, mock_socket, xfer_meta)
            mock_send_blocks.assert_called_once()
            mock_socket.send_multipart.assert_called_once()
            _, sent_payload = mock_socket.send_multipart.call_args[0][0]
            response = prefill_worker._xfer_resp_decoder.decode(sent_payload)
            assert response.err_msg == "Mooncake transfer engine returned 123"
            assert response.err_reqs == ["d-req-1"]

        # Clean up
        prefill_worker.sender_loop = origin_sender_loop
        prefill_worker.shutdown()


@pytest.mark.asyncio
@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
    "mooncake_connector.TransferEngine",
    FakeMooncakeWrapper,
)
async def test_build_transfer_params_routes_blocks_by_region_group_id(monkeypatch):
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_producer"
    )

    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            _make_test_kv_cache_config(),
        )
        worker = connector.connector_worker

        block_len = 100
        transfer_id = "xfer-hma-route"
        send_meta = SendBlockMeta(
            p_req_id="p-hma-route",
            transfer_id=transfer_id,
            local_block_ids=[[10, 11], [50, 51]],
            ready=asyncio.Event(),
        )
        xfer_meta = MooncakeXferMetadata(
            remote_hostname="consumer-host",
            remote_port=54321,
            remote_tp_size=1,
            remote_tp_rank=0,
            req_blocks={"d-hma-route": (transfer_id, [[20, 21], [60, 61]])},
            kv_caches_base_addr=[0x2000, 0x4000],
            block_lens=[block_len, block_len],
            registered_layer_names=["layers.0.attn", "layers.1.swa"],
            registered_layer_group_ids=[0, 1],
        )
        local_regions = [
            TransferRegion(0x1000, block_len, block_len, group_id=0),
            TransferRegion(0x3000, block_len, block_len, group_id=1),
        ]
        remote_regions = [
            TransferRegion(0x2000, block_len, block_len, group_id=0),
            TransferRegion(0x4000, block_len, block_len, group_id=1),
        ]

        (
            src_ptrs,
            dst_ptrs,
            lengths,
            err_reqs,
            err_msg,
        ) = await worker._build_transfer_params(
            [("d-hma-route", send_meta)],
            xfer_meta,
            local_regions,
            remote_regions,
        )

        assert err_reqs == []
        assert err_msg is None
        assert src_ptrs == [
            0x1000 + 10 * block_len,
            0x3000 + 50 * block_len,
        ]
        assert dst_ptrs == [
            0x2000 + 20 * block_len,
            0x4000 + 60 * block_len,
        ]
        assert lengths == [2 * block_len, 2 * block_len]
        worker.shutdown()


def test_split_kv_duplicate_names_matcher():
    """split-K/V layer-name pattern: producer registers each layer's K and V
    halves as two entries sharing the same layer name. The per-name matcher
    must pair occurrences in order across producer and consumer.

    The connector-level path (`send_kv_to_decode`) interleaves this with
    HND's is_kv_layout_blocks_first expansion and the asyncio sender loop,
    which is exercised by other tests (`test_kv_producer` etc.). This test
    isolates the matcher behavior on its own.
    """

    worker = object.__new__(MooncakeConnectorWorker)
    worker.async_zmq_ctx = MagicMock()
    worker.is_kv_consumer = True
    worker.is_kv_producer = True
    worker.transfer_topo = MagicMock()
    worker.transfer_topo.is_kv_layout_blocks_first = False
    worker.registered_layer_names = ["layers.0.attn", "layers.0.attn"]
    worker.registered_layer_group_ids = [0, 0]

    block_len = 100
    local_regions_all = [
        TransferRegion(0x1000, block_len, block_len, group_id=0),
        TransferRegion(0x3000, block_len, block_len, group_id=0),
    ]
    remote_regions_all = [
        TransferRegion(0x2000, block_len, block_len, group_id=0),
        TransferRegion(0x4000, block_len, block_len, group_id=0),
    ]
    meta = MooncakeXferMetadata(
        remote_hostname="consumer-host",
        remote_port=54321,
        remote_tp_size=1,
        remote_tp_rank=0,
        req_blocks={},
        kv_caches_base_addr=[0x2000, 0x4000],
        block_lens=[block_len, block_len],
        registered_layer_names=["layers.0.attn", "layers.0.attn"],
        registered_layer_group_ids=[0, 0],
    )

    err, local_regions, remote_regions = worker._match_regions_by_name(
        meta, local_regions_all, remote_regions_all
    )

    assert err is None
    assert [r.base_addr for r in local_regions] == [0x1000, 0x3000]
    assert [r.base_addr for r in remote_regions] == [0x2000, 0x4000]


@pytest.mark.asyncio
@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
    "mooncake_connector.TransferEngine",
    FakeMooncakeWrapper,
)
async def test_send_kv_to_decode_occurrence_count_mismatch(monkeypatch):
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_producer"
    )

    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            _make_test_kv_cache_config(),
        )
        worker = connector.connector_worker
        worker.kv_caches_base_addr = [0x1000]
        worker.block_len_per_layer = [100]
        worker.registered_layer_names = ["x"]
        worker.registered_layer_group_ids = [0]

        xfer_meta = MooncakeXferMetadata(
            remote_hostname="consumer-host",
            remote_port=54321,
            remote_tp_size=1,
            remote_tp_rank=0,
            req_blocks={},
            kv_caches_base_addr=[0x2000, 0x3000],
            block_lens=[100, 100],
            registered_layer_names=["x", "x"],
            registered_layer_group_ids=[0, 0],
        )
        mock_socket = AsyncMock(spec=zmq.asyncio.Socket)
        mock_socket.send_multipart = AsyncMock()

        await worker.send_kv_to_decode(b"consumer", mock_socket, xfer_meta)

        _, sent_payload = mock_socket.send_multipart.call_args[0][0]
        response = worker._xfer_resp_decoder.decode(sent_payload)
        assert response.status == MooncakeXferResponseStatus.ERROR
        assert "producer has 1, consumer has 2" in response.err_msg
        worker.shutdown()


@pytest.mark.asyncio
async def test_kv_consumuer(monkeypatch):
    """
    Simulates a Consumer Worker (Decoder) initiating a pull from a Producer.

    Verifies that MooncakeXferMetadata is correctly serialized and sent via ZMQ.
    """

    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_consumer"
    )

    with set_current_vllm_config(vllm_config), patch_worker_dependencies() as mocks:
        decode_connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            _make_test_kv_cache_config(),
        )
        decode_worker = decode_connector.connector_worker
        decode_worker.kv_caches_base_addr = [0x1000]
        decode_worker.rpc_port = 54321

        # A request to pull data arrives.
        pull_metas = {
            "d-req-1": PullReqMeta(
                d_req_id="d-req-1",
                transfer_id="xfer-req-1",
                local_block_ids=[[100, 101]],
                remote_engine_id="p-engine",
                remote_bootstrap_addr="http://bootstrap:33333",
                pull_tasks_count=1,
            )
        }
        decode_worker._remote_agents = {"p-engine": {0: {0: "tcp://producer:1234"}}}
        decode_worker._tp_size["p-engine"] = 1
        decode_worker._remote_engine_pp_size["p-engine"] = 1

        # Mock the response from the producer.
        mock_response = MooncakeXferResponse(
            status=MooncakeXferResponseStatus.FINISH, ok_reqs=["d-req-1"]
        )
        encoded_response = decode_worker._encoder.encode(mock_response)
        mocks["mock_socket_object"].recv.return_value = encoded_response

        # Trigger the receive logic.
        decode_worker.receive_kv("p-engine", pull_metas)
        await asyncio.sleep(1)  # Allow async task to run

        # Verify the metadata sent to the producer.
        mocks["mock_make_zmq"].assert_called_with(
            decode_worker.async_zmq_ctx,
            "tcp://producer:1234",
            zmq.DEALER,
            bind=False,
            linger=0,
        )
        sent_payload = mocks["mock_socket_object"].send.call_args[0][0]
        sent_meta = decode_worker._xfer_meta_decoder.decode(sent_payload)

        assert sent_meta.remote_hostname == "127.0.0.1"
        assert sent_meta.remote_port == 54321
        assert sent_meta.req_blocks["d-req-1"] == ("xfer-req-1", [[100, 101]])

        # Verify internal state is updated correctly.
        assert "d-req-1" in decode_worker.finished_recving_reqs

        # Clean up
        decode_worker.shutdown()


def _minimal_worker_for_receive():
    worker = object.__new__(MooncakeConnectorWorker)
    worker.async_zmq_ctx = MagicMock()
    worker.is_kv_consumer = True
    worker.is_kv_producer = True
    worker.hostname = "127.0.0.1"
    worker.rpc_port = 54321
    worker.tp_size = 1
    worker.tp_rank = 0
    worker.kv_caches_base_addr = [0x1000, 0x2000]
    worker.block_len_per_layer = [128, 256]
    worker.registered_layer_names = ["layers.0.attn", "layers.1.swa"]
    worker.registered_layer_group_ids = [0, 1]
    worker.finished_recving_reqs = set()
    worker._failed_block_ids = queue.Queue()
    worker.xfer_stats = MagicMock()
    worker._encoder = msgspec.msgpack.Encoder()
    worker._xfer_meta_decoder = msgspec.msgpack.Decoder(MooncakeXferMetadata)
    worker._xfer_resp_decoder = msgspec.msgpack.Decoder(MooncakeXferResponse)
    worker.transfer_topo = MagicMock()
    worker.transfer_topo.handshake_target_ranks.return_value = [0, 1]
    worker._tp_size = {"p-engine": 2}
    worker._remote_engine_pp_size = {"p-engine": 2}
    worker._remote_agents = {
        "p-engine": {
            0: {
                0: "tcp://p0-t0:1234",
                1: "tcp://p1-t0:1234",
            },
            1: {
                0: "tcp://p0-t1:1234",
                1: "tcp://p1-t1:1234",
            },
        }
    }
    worker._remote_bootstrap_addrs = {}
    worker._last_requery_time = {}
    worker._bootstrap_ready_timeout_s = 60
    return worker


class _FakeMooncakeSocketContext:
    def __init__(self, socket):
        self.socket = socket

    def __enter__(self):
        return self.socket

    def __exit__(self, *args):
        return None


@pytest.mark.asyncio
async def test_receive_kv_fans_out_to_all_pp_tp_shards():
    worker = _minimal_worker_for_receive()
    pull_metas = {
        "d-req": PullReqMeta(
            d_req_id="d-req",
            transfer_id="xfer-d-req",
            local_block_ids=[[100, 101], [200, 201]],
            remote_engine_id="p-engine",
            remote_bootstrap_addr="http://bootstrap:33333",
        )
    }
    sent_by_addr: dict[str, MooncakeXferMetadata] = {}
    response = worker._encoder.encode(
        MooncakeXferResponse(
            status=MooncakeXferResponseStatus.FINISH,
            ok_reqs=["d-req"],
        )
    )

    def fake_make_zmq_socket(_ctx, addr, *_args, **_kwargs):
        socket = AsyncMock()
        socket.setsockopt = MagicMock()

        async def send(payload):
            sent_by_addr[addr] = worker._xfer_meta_decoder.decode(payload)

        socket.send = AsyncMock(side_effect=send)
        socket.recv = AsyncMock(return_value=response)
        return _FakeMooncakeSocketContext(socket)

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
        "mooncake_connector.make_zmq_socket",
        side_effect=fake_make_zmq_socket,
    ):
        worker.receive_kv("p-engine", pull_metas)
        for _ in range(20):
            if len(sent_by_addr) == 4:
                break
            await asyncio.sleep(0)

    assert set(sent_by_addr) == {
        "tcp://p0-t0:1234",
        "tcp://p0-t1:1234",
        "tcp://p1-t0:1234",
        "tcp://p1-t1:1234",
    }
    assert pull_metas["d-req"].pull_tasks_count == 0
    assert worker.finished_recving_reqs == {"d-req"}
    for sent_meta in sent_by_addr.values():
        assert sent_meta.req_blocks["d-req"] == (
            "xfer-d-req",
            [[100, 101], [200, 201]],
        )
        assert sent_meta.registered_layer_names == [
            "layers.0.attn",
            "layers.1.swa",
        ]
        assert sent_meta.registered_layer_group_ids == [0, 1]


@pytest.mark.asyncio
async def test_receive_failure_refreshes_topology_for_next_request():
    worker = _minimal_worker_for_receive()
    worker.transfer_topo.handshake_target_ranks.return_value = [0]
    worker._tp_size = {"p-engine": 1}
    worker._remote_engine_pp_size = {"p-engine": 1}
    worker.total_num_hidden_layers = 1
    worker._layer_name_to_group_id = {"layers.0.attn": 0}
    worker._remote_bootstrap_addrs = {"p-engine": "http://bootstrap:33333"}
    worker._remote_agents = {
        "p-engine": {
            0: {
                0: ShardInfo(
                    addr="tcp://old:1234",
                    pp_rank=0,
                    tp_rank=0,
                    start_layer=0,
                    end_layer=1,
                    registered_layer_names=["layers.0.attn"],
                    registered_layer_group_ids=[0],
                )
            }
        }
    }
    refreshed_payload = _make_pp_bootstrap_payload(
        "p-engine",
        pp_size=1,
        tp_size=1,
    )
    refreshed_payload["0"]["worker_addr"]["0"]["0"] = "tcp://new:1234"
    refreshed_payload["0"]["shard_info"]["0"]["0"]["addr"] = "tcp://new:1234"
    worker._query_bootstrap = AsyncMock(return_value=refreshed_payload)

    response = worker._encoder.encode(
        MooncakeXferResponse(
            status=MooncakeXferResponseStatus.FINISH,
            ok_reqs=["d-req-2"],
        )
    )
    used_addrs: list[str] = []

    def fake_make_zmq_socket(_ctx, addr, *_args, **_kwargs):
        used_addrs.append(addr)
        socket = AsyncMock()
        socket.setsockopt = MagicMock()
        socket.send = AsyncMock()
        if addr == "tcp://old:1234":
            socket.recv = AsyncMock(side_effect=RuntimeError("stale producer"))
        else:
            socket.recv = AsyncMock(return_value=response)
        return _FakeMooncakeSocketContext(socket)

    first_pull = {
        "d-req-1": PullReqMeta(
            d_req_id="d-req-1",
            transfer_id="xfer-d-req-1",
            local_block_ids=[[100]],
            remote_engine_id="p-engine",
            remote_bootstrap_addr="http://bootstrap:33333",
        )
    }
    second_pull = {
        "d-req-2": PullReqMeta(
            d_req_id="d-req-2",
            transfer_id="xfer-d-req-2",
            local_block_ids=[[101]],
            remote_engine_id="p-engine",
            remote_bootstrap_addr="http://bootstrap:33333",
        )
    }

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
        "mooncake_connector.make_zmq_socket",
        side_effect=fake_make_zmq_socket,
    ):
        worker.receive_kv("p-engine", first_pull)
        for _ in range(20):
            shard = worker._remote_agents["p-engine"][0][0]
            if isinstance(shard, ShardInfo) and shard.addr == "tcp://new:1234":
                break
            await asyncio.sleep(0)

        worker.receive_kv("p-engine", second_pull)
        for _ in range(20):
            if "d-req-2" in worker.finished_recving_reqs:
                break
            await asyncio.sleep(0)

    assert used_addrs == ["tcp://old:1234", "tcp://new:1234"]
    assert "d-req-1" in worker.finished_recving_reqs
    assert "d-req-2" in worker.finished_recving_reqs


def _pull_meta(
    local_block_ids: list[list[int]],
    pull_tasks_count: int,
    req_id: str = "d-req",
) -> PullReqMeta:
    return PullReqMeta(
        d_req_id=req_id,
        transfer_id=f"xfer-{req_id}",
        local_block_ids=local_block_ids,
        remote_engine_id="p-engine",
        remote_bootstrap_addr="http://bootstrap:33333",
        pull_tasks_count=pull_tasks_count,
    )


def test_terminal_accounting_error_path_non_hma():
    worker = _minimal_worker_for_accounting()
    pull_metas = {"d-req": _pull_meta([[1, 2, 3]], pull_tasks_count=2)}

    acked: set[str] = set()
    worker.process_pulling_result(
        MooncakeXferResponse(
            status=MooncakeXferResponseStatus.ERROR,
            err_reqs=["d-req"],
            err_msg="boom",
        ),
        pull_metas,
        acked,
    )
    worker._ack_remaining(pull_metas, acked, failed=True)
    assert pull_metas["d-req"].pull_tasks_count == 1
    assert worker.get_block_ids_with_load_errors() == {1, 2, 3}

    acked = set()
    worker.process_pulling_result(
        MooncakeXferResponse(
            status=MooncakeXferResponseStatus.FINISH,
            ok_reqs=["d-req"],
        ),
        pull_metas,
        acked,
    )
    worker._ack_remaining(pull_metas, acked, failed=True)
    assert pull_metas["d-req"].pull_tasks_count == 0
    assert worker.finished_recving_reqs == {"d-req"}


def test_terminal_accounting_error_path_hma_fail_fast():
    worker = _minimal_worker_for_accounting()
    pull_metas = {"d-req": _pull_meta([[1, 2], [10, 11]], pull_tasks_count=1)}
    acked: set[str] = set()

    worker.process_pulling_result(
        MooncakeXferResponse(
            status=MooncakeXferResponseStatus.ERROR,
            err_reqs=["d-req"],
            err_msg="boom",
        ),
        pull_metas,
        acked,
    )
    worker._ack_remaining(pull_metas, acked, failed=True)

    assert worker.finished_recving_reqs == {"d-req"}
    assert worker.get_block_ids_with_load_errors() == {1, 10}


def test_terminal_accounting_finish_with_err_reqs():
    worker = _minimal_worker_for_accounting()
    pull_metas = {"d-req": _pull_meta([[7, 8]], pull_tasks_count=1)}
    acked: set[str] = set()

    worker.process_pulling_result(
        MooncakeXferResponse(
            status=MooncakeXferResponseStatus.FINISH,
            err_reqs=["d-req"],
            err_msg="timeout",
        ),
        pull_metas,
        acked,
    )
    worker._ack_remaining(pull_metas, acked, failed=True)

    assert worker.finished_recving_reqs == {"d-req"}
    assert worker.get_block_ids_with_load_errors() == {7, 8}


def test_terminal_accounting_exception_path_marks_unacked_failed():
    worker = _minimal_worker_for_accounting()
    pull_metas = {
        "d-req": _pull_meta([[1], [10]], pull_tasks_count=1),
        "d-req-2": _pull_meta([[2], [20]], pull_tasks_count=1, req_id="d-req-2"),
    }

    worker._ack_remaining(pull_metas, acked=set(), failed=True)

    assert worker.finished_recving_reqs == {"d-req", "d-req-2"}
    assert worker.get_block_ids_with_load_errors() == {1, 10, 2, 20}


def test_terminal_accounting_no_double_decrement_continue_then_error():
    worker = _minimal_worker_for_accounting()
    pull_metas = {
        req_id: _pull_meta([[idx]], pull_tasks_count=1, req_id=req_id)
        for idx, req_id in enumerate(["r1", "r2", "r3"], start=1)
    }
    acked: set[str] = set()

    worker.process_pulling_result(
        MooncakeXferResponse(
            status=MooncakeXferResponseStatus.CONTINUE,
            ok_reqs=["r1", "r2"],
        ),
        pull_metas,
        acked,
    )
    worker.process_pulling_result(
        MooncakeXferResponse(
            status=MooncakeXferResponseStatus.ERROR,
            err_reqs=["r2", "r3"],
            err_msg="boom",
        ),
        pull_metas,
        acked,
    )
    worker._ack_remaining(pull_metas, acked, failed=True)

    assert [pull_metas[req_id].pull_tasks_count for req_id in ["r1", "r2", "r3"]] == [
        0,
        0,
        0,
    ]
    assert worker.finished_recving_reqs == {"r1", "r2", "r3"}
    assert worker.get_block_ids_with_load_errors() == {3}


def test_failed_block_queue_thread_safety_smoke():
    worker = _minimal_worker_for_accounting()
    seen: set[int] = set()
    stop = threading.Event()

    def drain_loop():
        while not stop.is_set():
            seen.update(worker.get_block_ids_with_load_errors())

    drain_thread = threading.Thread(target=drain_loop)
    drain_thread.start()
    for block_id in range(200):
        pull_metas = {
            str(block_id): _pull_meta(
                [[block_id]], pull_tasks_count=1, req_id=str(block_id)
            )
        }
        worker._shard_ack(pull_metas, str(block_id), failed=True)
    stop.set()
    drain_thread.join()
    seen.update(worker.get_block_ids_with_load_errors())

    assert seen == set(range(200))


@pytest.mark.asyncio
async def test_worker_get_finished_timeout(monkeypatch):
    """Tests the cleanup mechanism for requests."""

    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_producer"
    )
    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        prefill_connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            _make_test_kv_cache_config(),
        )
        prefill_worker = prefill_connector.connector_worker

        # Add an expired request (expire_time is in the past).
        prefill_worker.reqs_need_send["tx-expired"] = SendBlockMeta(
            p_req_id="p-req-expired",
            transfer_id="tx-expired",
            local_block_ids=[[1, 2]],
            ready=MagicMock(),
            expire_time=time.perf_counter() - 100,
        )

        # Add a non-expired request.
        prefill_worker.reqs_need_send["tx-active"] = SendBlockMeta(
            p_req_id="p-req-active",
            transfer_id="tx-active",
            local_block_ids=[[3, 4]],
            ready=MagicMock(),
            expire_time=time.perf_counter() + 100,
        )

        finished_reqs = await prefill_worker.fetch_finished_sending_reqs()

        assert "p-req-expired" in finished_reqs
        assert "p-req-active" not in finished_reqs
        assert "tx-expired" not in prefill_worker.reqs_need_send
        assert "tx-active" in prefill_worker.reqs_need_send


def test_register_kv_caches():
    """Per-layer-name wire lists keep memory registration deduplicated."""

    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_consumer"
    )
    kv_cache_config = _make_layered_kv_cache_config(
        [["layers.0.attn", "layers.1.attn"], ["layers.2.swa"]]
    )

    with (
        set_current_vllm_config(vllm_config),
        patch_worker_dependencies(),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.threading.Event"
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.threading.Thread"
        ) as mock_thread,
    ):
        connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            kv_cache_config,
        )
        worker = connector.connector_worker
        mock_thread.return_value.is_alive.return_value = False

        kv_cache_shape = FlashAttentionBackend.get_kv_cache_shape(
            num_blocks=2, block_size=16, num_kv_heads=4, head_size=64
        )
        full_pool = torch.zeros(*kv_cache_shape, dtype=torch.float16)
        swa_pool = torch.zeros(*kv_cache_shape, dtype=torch.float16)
        kv_caches = {
            "layers.0.attn": full_pool,
            "layers.1.attn": full_pool,
            "layers.2.swa": swa_pool,
        }

        with patch.object(
            worker.engine, "batch_register_memory", return_value=0
        ) as mock_batch_register:
            connector.register_kv_caches(kv_caches)

            mock_batch_register.assert_called_once()
            registered_ptrs, registered_lens = mock_batch_register.call_args[0]
            assert registered_ptrs == [full_pool.data_ptr(), swa_pool.data_ptr()]
            block_len = full_pool.stride(0) * full_pool.element_size()
            assert registered_lens == [2 * block_len, 2 * block_len]

            assert worker.kv_caches_base_addr == [
                full_pool.data_ptr(),
                full_pool.data_ptr(),
                swa_pool.data_ptr(),
            ]
            assert worker.block_len_per_layer == [block_len, block_len, block_len]
            assert worker.registered_layer_names == [
                "layers.0.attn",
                "layers.1.attn",
                "layers.2.swa",
            ]
            assert worker.registered_layer_group_ids == [0, 0, 1]


class _CrossLayerTransferTopology:
    cross_layers_blocks = True

    def __init__(self, *args, **kwargs):
        pass


@pytest.mark.parametrize("kv_role", ["kv_producer", "kv_consumer"])
@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
    "mooncake_connector.TransferEngine",
    FakeMooncakeWrapper,
)
def test_cross_layer_blocks_with_pp_raises_at_construction(kv_role):
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role=kv_role,
    )
    vllm_config.parallel_config.pipeline_parallel_size = 2

    with (
        set_current_vllm_config(vllm_config),
        patch_worker_dependencies(),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_connector.TransferTopology",
            _CrossLayerTransferTopology,
        ),
        pytest.raises(RuntimeError, match="cross-layer-blocks"),
    ):
        MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            _make_test_kv_cache_config(),
        )


def test_register_kv_caches_supports_mixed_mla_and_eagle_shapes():
    """Mixed MLA+Eagle caches should register by byte length, not shape."""

    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_consumer"
    )
    kv_cache_config = _make_layered_kv_cache_config(
        [["layers.0.attn", "layers.1.attn"]]
    )

    with (
        set_current_vllm_config(vllm_config),
        patch_worker_dependencies(),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.threading.Event"
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.threading.Thread"
        ) as mock_thread,
    ):
        connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            kv_cache_config,
        )
        worker = connector.connector_worker
        mock_thread.return_value.is_alive.return_value = False

        worker.use_mla = True
        worker.transfer_topo.is_mla = True

        # MLA cache tensor: shape[-2] is the block size.
        mla_cache = torch.zeros((2, 16, 96), dtype=torch.float16)
        # Eagle3/GQA-like cache tensor: shape[-2] is num_kv_heads, not block size.
        eagle_cache = torch.zeros((2, 16, 8, 64), dtype=torch.float16)
        kv_caches = {"layers.0.attn": mla_cache, "layers.1.attn": eagle_cache}

        with patch.object(
            worker.engine, "batch_register_memory", return_value=0
        ) as mock_batch_register:
            connector.register_kv_caches(kv_caches)

        mock_batch_register.assert_called_once()
        registered_ptrs, registered_lens = mock_batch_register.call_args[0]
        assert registered_ptrs == [mla_cache.data_ptr(), eagle_cache.data_ptr()]
        assert registered_lens == [mla_cache.nbytes, eagle_cache.nbytes]
        assert worker.block_len_per_layer == [
            mla_cache.nbytes // mla_cache.shape[0],
            eagle_cache.nbytes // eagle_cache.shape[0],
        ]


@pytest.mark.asyncio
@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
    "mooncake_connector.TransferEngine",
    FakeMooncakeWrapper,
)
@pytest.mark.parametrize("d_tp_size", [1, 4], ids=["p_tp2_d_tp1", "p_tp2_d_tp4"])
async def test_kv_producer_heterogeneous_tp(monkeypatch, d_tp_size):
    """
    Tests heterogeneous TP support in the producer transfer path.

    Verifies correct pointer and offset calculation when producer TP=2
    sends to consumer with TP=1 (P>D) or TP=4 (P<D).

    Parametrized cases:
    - P TP=2 > D TP=1: one D rank receives; dst_offset based on P rank
    - P TP=2 < D TP=4: two D ranks receive; src_offset based on D rank
    """

    P_TP_SIZE = 2
    P_TP_RANK = 0
    LOCAL_BLOCK_LEN = 4096

    local_block_len = LOCAL_BLOCK_LEN
    remote_block_len = LOCAL_BLOCK_LEN * P_TP_SIZE // d_tp_size

    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_producer"
    )

    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        prefill_connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            _make_test_kv_cache_config(),
        )
        prefill_worker = prefill_connector.connector_worker

        # Override TP rank/size to simulate P TP=2
        prefill_worker.tp_rank = P_TP_RANK
        prefill_worker.tp_size = P_TP_SIZE
        prefill_worker._tp_size[prefill_worker.engine_id] = P_TP_SIZE
        prefill_worker.transfer_topo.tp_rank = P_TP_RANK
        prefill_worker.transfer_topo.tp_size = P_TP_SIZE

        prefill_worker.kv_caches_base_addr = [0x1000]
        prefill_worker.block_len_per_layer = [local_block_len]

        origin_sender_loop = prefill_worker.sender_loop
        prefill_worker.sender_loop = asyncio.get_event_loop()

        transfer_id = "xfer-hetero-1"
        local_block_ids = [[10, 11]]
        send_meta = SendBlockMeta(
            p_req_id="p-req-h1",
            transfer_id=transfer_id,
            local_block_ids=local_block_ids,
            ready=asyncio.Event(),
        )
        prefill_worker.reqs_need_send[transfer_id] = send_meta
        send_meta.ready.set()

        # Compute target D ranks using the production code path
        target_d_ranks = prefill_worker.transfer_topo.handshake_target_ranks(d_tp_size)

        mock_socket = AsyncMock(spec=zmq.asyncio.Socket)
        mock_socket.send_multipart = AsyncMock()
        identity = b"consumer-hetero"

        # Assign different remote block IDs per D rank (nested per-group)
        d_rank_remote_blocks = {
            rank: [[20 + i * 10, 21 + i * 10]] for i, rank in enumerate(target_d_ranks)
        }

        with patch.object(
            prefill_worker, "_send_blocks", return_value=0
        ) as mock_send_blocks:
            for d_rank in target_d_ranks:
                remote_block_ids = d_rank_remote_blocks[d_rank]
                xfer_meta = MooncakeXferMetadata(
                    remote_hostname="consumer-host",
                    remote_port=54321,
                    remote_tp_size=d_tp_size,
                    remote_tp_rank=d_rank,
                    req_blocks={
                        f"d-req-h1-r{d_rank}": (
                            transfer_id,
                            remote_block_ids,
                        )
                    },
                    kv_caches_base_addr=[0x2000],
                    block_lens=[remote_block_len],
                )

                mock_send_blocks.reset_mock()
                mock_socket.send_multipart.reset_mock()

                await prefill_worker.send_kv_to_decode(identity, mock_socket, xfer_meta)

                # Verify _send_blocks was called
                mock_send_blocks.assert_called_once()
                call_args = mock_send_blocks.call_args[0]
                src_ptrs = call_args[1]
                dst_ptrs = call_args[2]
                lengths = call_args[3]

                # Flatten nested per-group block IDs for assertions
                flat_local = [b for g in local_block_ids for b in g]
                flat_remote = [b for g in remote_block_ids for b in g]
                num_blocks = len(flat_local)

                # With blocks-first layout, virtual split halves block
                # lengths and doubles transfer regions (K + V).
                local_kv_block_len = local_block_len // 2
                remote_kv_block_len = remote_block_len // 2

                assert len(src_ptrs) == 2 * num_blocks
                assert len(dst_ptrs) == 2 * num_blocks
                assert len(lengths) == 2 * num_blocks

                # Compute expected offsets using kv_block_len
                if d_tp_size <= P_TP_SIZE:
                    tp_ratio = P_TP_SIZE // d_tp_size
                    expected_src_off = 0
                    expected_dst_off = (P_TP_RANK % tp_ratio) * local_kv_block_len
                    expected_xfer_len = local_kv_block_len
                else:
                    ratio_abs = d_tp_size // P_TP_SIZE
                    expected_src_off = (d_rank % ratio_abs) * remote_kv_block_len
                    expected_dst_off = 0
                    expected_xfer_len = remote_kv_block_len

                # First num_blocks entries are K region,
                # next num_blocks are V region.
                for region_idx in range(2):
                    local_region_base = 0x1000 + region_idx * local_kv_block_len
                    remote_region_base = 0x2000 + region_idx * remote_kv_block_len
                    for blk_idx, (lblk, rblk) in enumerate(
                        zip(flat_local, flat_remote)
                    ):
                        idx = region_idx * num_blocks + blk_idx
                        assert src_ptrs[idx] == (
                            local_region_base
                            + lblk * local_block_len
                            + expected_src_off
                        )
                        assert dst_ptrs[idx] == (
                            remote_region_base
                            + rblk * remote_block_len
                            + expected_dst_off
                        )
                        assert lengths[idx] == expected_xfer_len

                # Verify successful response sent back to consumer
                mock_socket.send_multipart.assert_called_once()
                _, sent_payload = mock_socket.send_multipart.call_args[0][0]
                response = prefill_worker._xfer_resp_decoder.decode(sent_payload)
                assert response.status == MooncakeXferResponseStatus.FINISH
                assert response.ok_reqs == [f"d-req-h1-r{d_rank}"]

        # After serving all D ranks, the request should be complete
        assert transfer_id not in prefill_worker.reqs_need_send
        assert "p-req-h1" in prefill_worker.finished_sending_reqs

        prefill_worker.sender_loop = origin_sender_loop
        prefill_worker.shutdown()
