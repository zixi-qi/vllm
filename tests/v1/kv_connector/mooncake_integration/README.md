# Mooncake PP Integration Tests

These tests are skipped by default because they require a multi-node Mooncake
P/D deployment. They are intended for GB200 or similar cluster runs where the
operator can launch separate PP prefill and decode instances.

Suggested smoke shape:

- prefill: MooncakeConnector, `pipeline_parallel_size=2`
- decode: MooncakeConnector, `pipeline_parallel_size=1`
- model: a small causal LM that fits on the selected nodes
- prompts: short deterministic prompts, temperature 0

Suggested stress shape:

- prefill: `pp=4`, `tp=4`
- decode: compatible TP for the target deployment
- batch: 128 requests
- transfer mode: `batch_transfer_sync_write`

The stress test exists to validate descriptor-list scaling against per-NIC and
per-CQ outstanding work-request limits. Remove the skip marker only in an
environment that can provision the required Mooncake transport and GPUs.
