# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Skipped-by-default Mooncake PP integration canaries.

These tests document the multi-node shapes operators should run manually.
They intentionally avoid importing cluster-only launch helpers during
collection so CPU-only CI can still collect the module.
"""

import pytest

pytestmark = pytest.mark.skip(reason="requires multi-node Mooncake setup")


def test_pp2_prefill_pp1_decode_smoke() -> None:
    """PP-2 prefill with PP-1 decode should preserve output correctness."""
    raise AssertionError("remove the skip marker and launch the documented setup")


def test_pp4_tp4_batch128_descriptor_scaling() -> None:
    """Stress descriptor-list scaling under batch_transfer_sync_write.

    The binding constraint is the transport's per-NIC/per-CQ outstanding
    work-request limit, not Python collection overhead.
    """
    raise AssertionError("remove the skip marker and launch the documented setup")
