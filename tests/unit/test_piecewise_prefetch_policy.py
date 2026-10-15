# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for :class:`~flextensor.piecewise_prefetch_policy.PiecewisePrefetchPolicy`."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from flextensor.core.transfer import CudaTransferRuntime
from flextensor.loaders import (
    PreallocatedBatchTransferTensorLoader,
    PreallocatedBatchTransferTensorLoaderReordered,
)
from flextensor.piecewise_prefetch_policy import (
    OutstandingPrefetch,
    PiecewisePrefetchPolicy,
    PiecewisePrefetchPolicyError,
)


class TestDisabledPiecewisePrefetchPolicy:
    def test_disabled_is_noop(self, caplog: pytest.LogCaptureFixture) -> None:
        policy = PiecewisePrefetchPolicy(enabled=False)
        policy.on_schedule("L1", "L4")
        with caplog.at_level(logging.WARNING):
            assert policy.on_piece_join() == []
        assert "piecewise prefetch policy" not in caplog.text


class TestPiecewisePrefetchPolicy:
    def test_same_layer_outstanding_at_join_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Parent enter/exit split across nested pieces (e.g. 1 vs 1.1/1.2)."""
        policy = PiecewisePrefetchPolicy(enabled=True, strict=False)
        policy.on_schedule("layers.0", "layers.0")
        with caplog.at_level(logging.WARNING):
            broken = policy.on_piece_join()
        assert broken == [OutstandingPrefetch("layers.0", "layers.0")]
        assert "layers.0" in caplog.text
        assert "H2D prefetch" in caplog.text

    def test_remapped_outstanding_warns_once(self, caplog: pytest.LogCaptureFixture) -> None:
        policy = PiecewisePrefetchPolicy(enabled=True, strict=False)
        policy.on_schedule("L1", "L4")
        with caplog.at_level(logging.WARNING):
            broken = policy.on_piece_join()
        assert broken == [OutstandingPrefetch("L1", "L4")]
        assert "L1" in caplog.text and "L4" in caplog.text

        with caplog.at_level(logging.WARNING):
            policy.on_schedule("L1", "L4")
            again = policy.on_piece_join()
        assert again == [OutstandingPrefetch("L1", "L4")]
        assert caplog.text.count("H2D prefetch") == 1

    def test_wait_clears_outstanding(self, caplog: pytest.LogCaptureFixture) -> None:
        policy = PiecewisePrefetchPolicy(enabled=True, strict=False)
        policy.on_schedule("L1", "L4")
        policy.on_wait("L4")
        with caplog.at_level(logging.WARNING):
            assert policy.on_piece_join() == []
        assert "piecewise prefetch policy" not in caplog.text

    def test_strict_raises(self) -> None:
        policy = PiecewisePrefetchPolicy(enabled=True, strict=True)
        policy.on_schedule("L1", "L4")
        with pytest.raises(PiecewisePrefetchPolicyError, match=r"L1.*L4"):
            policy.on_piece_join()

    def test_reset_clears_state(self) -> None:
        policy = PiecewisePrefetchPolicy(enabled=True, strict=True)
        policy.on_schedule("L1", "L4")
        policy.reset()
        assert policy.on_piece_join() == []

    def test_last_layer_ignores_next_iter_prefetch_after_wait(self, caplog: pytest.LogCaptureFixture) -> None:
        """Reordered last-layer schedule for an already-waited label is next-iter."""
        policy = PiecewisePrefetchPolicy(enabled=True, strict=True)
        policy.reset()
        # Earlier in the forward: waited L0's prefetch, then last layer
        # re-schedules L0 for the next iteration.
        policy.on_schedule("L2", "L0")
        policy.on_wait("L0")
        policy.on_schedule("L_last", "L0")

        with caplog.at_level(logging.WARNING):
            assert policy.on_piece_join(at_last_layer=True) == []
        assert "piecewise prefetch policy" not in caplog.text
        assert policy._outstanding == {}  # noqa: SLF001

    def test_last_layer_still_flags_never_waited_prefetch(self) -> None:
        """Missed wait this forward (no prior on_wait) remains a real hit."""
        policy = PiecewisePrefetchPolicy(enabled=True, strict=True)
        policy.reset()
        policy.on_schedule("L1", "L4")  # L4 never waited this pass
        with pytest.raises(PiecewisePrefetchPolicyError, match="end-of-forward"):
            policy.on_piece_join(at_last_layer=True)

    def test_mid_piece_join_still_flags_next_iter_shaped_outstanding(self) -> None:
        """``at_last_layer=False`` must not apply the next-iter exemption."""
        policy = PiecewisePrefetchPolicy(enabled=True, strict=True)
        policy.reset()
        policy.on_schedule("L2", "L0")
        policy.on_wait("L0")
        policy.on_schedule("L_last", "L0")
        with pytest.raises(PiecewisePrefetchPolicyError, match="PIECEWISE"):
            policy.on_piece_join(at_last_layer=False)


def _bare_loader(cls: type) -> PreallocatedBatchTransferTensorLoader:
    """Minimal loader instance for join/exit map-clear tests (no CUDA init)."""
    loader = object.__new__(cls)
    loader.piecewise_prefetch_policy = PiecewisePrefetchPolicy(enabled=True, strict=True)
    loader.scheduled_transfers = {"keep": object()}
    loader._transfer_runtime = MagicMock(spec=CudaTransferRuntime)
    loader._transfer_runtime.is_active = False
    loader._compute_to_block_ids = {}
    loader.transfer_stream = MagicMock()
    loader.last_layer_label = "L_last"
    loader.offload_timing_collector = MagicMock()
    return loader  # type: ignore[return-value]


class TestLoaderStrictJoinClearsMaps:
    def test_join_after_forward_clears_maps_when_strict_raises(self) -> None:
        loader = _bare_loader(PreallocatedBatchTransferTensorLoader)
        loader.piecewise_prefetch_policy.on_schedule("L0", "L1")

        with (
            patch("torch.cuda.current_stream", return_value=MagicMock()),
            pytest.raises(PiecewisePrefetchPolicyError),
        ):
            loader.join_after_forward()

        assert loader.scheduled_transfers == {}

    def test_last_layer_exit_clears_maps_when_strict_raises(self) -> None:
        loader = _bare_loader(PreallocatedBatchTransferTensorLoader)
        loader.piecewise_prefetch_policy.on_schedule("L0", "L1")

        with (
            patch("torch.cuda.current_stream", return_value=MagicMock()),
            pytest.raises(PiecewisePrefetchPolicyError),
        ):
            loader.exit("L_last")

        assert loader.scheduled_transfers == {}

    def test_reordered_last_layer_exit_allows_next_iter_prefetch(self) -> None:
        """Strict reordered last-layer must not raise on next-iter outstanding."""
        loader = _bare_loader(PreallocatedBatchTransferTensorLoaderReordered)
        pp = loader.piecewise_prefetch_policy
        pp.reset()
        pp.on_schedule("L1", "L0")
        pp.on_wait("L0")
        pp.on_schedule("L_last", "L0")

        with patch("torch.cuda.current_stream", return_value=MagicMock()):
            loader.exit("L_last")  # must not raise

        assert loader.scheduled_transfers == {}
