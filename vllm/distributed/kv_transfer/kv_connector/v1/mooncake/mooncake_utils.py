# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
import time
from dataclasses import dataclass, field

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from vllm.config import ParallelConfig
from vllm.distributed.kv_transfer.kv_connector.utils import EngineId
from vllm.logger import init_logger

WorkerAddr = str

logger = init_logger(__name__)


def get_mooncake_dp_engine_index(parallel_config: ParallelConfig) -> int:
    """Return the per-engine DP index used for Mooncake side channels."""
    if parallel_config.local_engines_only:
        assert parallel_config.data_parallel_rank_local is not None
        return parallel_config.data_parallel_rank_local

    return parallel_config.data_parallel_index


class RegisterWorkerPayload(BaseModel):
    engine_id: EngineId
    dp_rank: int
    tp_rank: int
    pp_rank: int
    addr: WorkerAddr
    pp_size: int = 1
    tp_size: int = 1
    start_layer: int = 0
    end_layer: int = 0
    registered_layer_names: list[str] = []
    registered_layer_group_ids: list[int] = []


@dataclass
class ShardInfo:
    addr: WorkerAddr
    pp_rank: int
    tp_rank: int
    start_layer: int
    end_layer: int
    registered_layer_names: list[str]
    registered_layer_group_ids: list[int]


@dataclass
class EngineEntry:
    engine_id: EngineId
    # {tp_rank: {pp_rank: worker_addr}}
    worker_addr: dict[int, dict[int, WorkerAddr]]
    pp_size: int = 1
    tp_size: int = 1
    shard_info: dict[int, dict[int, ShardInfo]] = field(default_factory=dict)


class MooncakeBootstrapServer:
    """
    A centralized server running on the global rank 0 prefiller worker.
    Prefiller workers register their connection info (IP, port, ranks) here.
    """

    def __init__(self, host: str, port: int):
        self.workers: dict[int, EngineEntry] = {}

        self.host = host
        self.port = port
        self.app = FastAPI()
        self._register_routes()
        self.server_thread: threading.Thread | None = None
        self.server: uvicorn.Server | None = None

    def __del__(self):
        self.shutdown()

    def _register_routes(self):
        # All methods are async. No need to use lock to protect data.
        self.app.post("/register")(self.register_worker)
        self.app.get("/query", response_model=dict[int, EngineEntry])(self.query)

    def start(self):
        if self.server_thread:
            return

        config = uvicorn.Config(app=self.app, host=self.host, port=self.port)
        self.server = uvicorn.Server(config=config)
        self.server_thread = threading.Thread(
            target=self.server.run, name="mooncake_bootstrap_server", daemon=True
        )
        self.server_thread.start()
        while not self.server.started:
            time.sleep(0.1)  # Wait for the server to start
        logger.info("Mooncake Bootstrap Server started at %s:%d", self.host, self.port)

    def shutdown(self):
        if self.server_thread is None or self.server is None or not self.server.started:
            return

        self.server.should_exit = True
        self.server_thread.join()
        logger.info("Mooncake Bootstrap Server stopped.")

    async def register_worker(self, payload: RegisterWorkerPayload):
        """Handles registration of a prefiller worker."""
        fields_set = payload.model_fields_set
        pp_size_explicit = "pp_size" in fields_set
        tp_size_explicit = "tp_size" in fields_set
        effective_pp_size = payload.pp_size
        effective_tp_size = payload.tp_size
        if not pp_size_explicit:
            effective_pp_size = max(effective_pp_size, payload.pp_rank + 1)
        if not tp_size_explicit:
            effective_tp_size = max(effective_tp_size, payload.tp_rank + 1)

        if effective_pp_size <= 0:
            raise HTTPException(
                status_code=400,
                detail=f"pp_size must be positive, got {payload.pp_size}",
            )
        if effective_tp_size <= 0:
            raise HTTPException(
                status_code=400,
                detail=f"tp_size must be positive, got {payload.tp_size}",
            )
        if not 0 <= payload.pp_rank < effective_pp_size:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"pp_rank={payload.pp_rank} is outside [0, {effective_pp_size})"
                ),
            )
        if not 0 <= payload.tp_rank < effective_tp_size:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"tp_rank={payload.tp_rank} is outside [0, {effective_tp_size})"
                ),
            )
        if len(payload.registered_layer_names) != len(
            payload.registered_layer_group_ids
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "registered_layer_names and registered_layer_group_ids "
                    "must have the same length"
                ),
            )
        if effective_pp_size > 1:
            if not 0 <= payload.start_layer < payload.end_layer:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "PP workers must advertise a non-empty layer range; "
                        f"got [{payload.start_layer}, {payload.end_layer})"
                    ),
                )
            if not payload.registered_layer_names:
                raise HTTPException(
                    status_code=400,
                    detail="PP workers must advertise at least one layer",
                )

        if payload.dp_rank not in self.workers:
            self.workers[payload.dp_rank] = EngineEntry(
                engine_id=payload.engine_id,
                worker_addr={},
                pp_size=effective_pp_size,
                tp_size=effective_tp_size,
            )

        dp_entry = self.workers[payload.dp_rank]
        if dp_entry.engine_id != payload.engine_id:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Engine ID mismatch for dp_rank={payload.dp_rank}: "
                    f"expected {dp_entry.engine_id}, got {payload.engine_id}"
                ),
            )
        if pp_size_explicit and dp_entry.pp_size != effective_pp_size:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Worker topology mismatch for "
                    f"engine_id={payload.engine_id}, dp_rank={payload.dp_rank}: "
                    f"expected pp_size={dp_entry.pp_size}; got "
                    f"pp_size={effective_pp_size}"
                ),
            )
        if tp_size_explicit and dp_entry.tp_size != effective_tp_size:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Worker topology mismatch for "
                    f"engine_id={payload.engine_id}, dp_rank={payload.dp_rank}: "
                    f"expected tp_size={dp_entry.tp_size}; got "
                    f"tp_size={effective_tp_size}"
                ),
            )
        if not pp_size_explicit and effective_pp_size > dp_entry.pp_size:
            dp_entry.pp_size = effective_pp_size
        if not tp_size_explicit and effective_tp_size > dp_entry.tp_size:
            dp_entry.tp_size = effective_tp_size

        tp_entry = dp_entry.worker_addr.setdefault(payload.tp_rank, {})
        shard_tp_entry = dp_entry.shard_info.setdefault(payload.tp_rank, {})
        prev_addr = tp_entry.get(payload.pp_rank)
        if prev_addr is not None:
            logger.warning(
                "Replacing stale registration for "
                "(engine=%s, dp=%d, tp=%d, pp=%d): %s -> %s",
                payload.engine_id,
                payload.dp_rank,
                payload.tp_rank,
                payload.pp_rank,
                prev_addr,
                payload.addr,
            )

        tp_entry[payload.pp_rank] = payload.addr
        shard_tp_entry[payload.pp_rank] = ShardInfo(
            addr=payload.addr,
            pp_rank=payload.pp_rank,
            tp_rank=payload.tp_rank,
            start_layer=payload.start_layer,
            end_layer=payload.end_layer,
            registered_layer_names=payload.registered_layer_names,
            registered_layer_group_ids=payload.registered_layer_group_ids,
        )
        logger.debug(
            "Registered worker: engine_id=%s, dp_rank=%d, tp_rank=%d, pp_rank=%d at %s",
            payload.engine_id,
            payload.dp_rank,
            payload.tp_rank,
            payload.pp_rank,
            payload.addr,
        )

        return {"status": "ok"}

    async def query(self) -> dict[int, EngineEntry]:
        return self.workers
