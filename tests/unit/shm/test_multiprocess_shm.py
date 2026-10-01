# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import multiprocessing
import os
import sys
import uuid
from pathlib import Path

import pytest
import torch

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent / ".." / ".." / ".."))

# Import the classes directly from the file
from flextensor.allocation_block import AllocationManager
from flextensor.host_pinning import HostPinner, is_available


def shm_owner_process(shm_name_prefix, ready_event, finished_event, pinned_memory):
    """Main process that owns shared memory and creates the initial data."""
    manager = None
    try:
        # Create manager as shm owner (load_from_shm=False means we create the SHM)
        manager = AllocationManager(
            shm_block_name_prefix=shm_name_prefix,
            load_from_shm=False,
            pinned_memory=pinned_memory,
            host_pinner=HostPinner(),
            memory_alignment=128,
        )

        # Create and populate first block (as shm owner)
        b1 = manager.block(device="cpu")
        t1 = torch.ones(5, dtype=torch.float32) * 2
        t2 = torch.ones(3, dtype=torch.float64) * 3
        b1.add(t1)
        b1.add(t2)
        b1.allocate()

        # Create and populate second block (as shm owner)
        b2 = manager.block(device="cpu")
        t3 = torch.ones(4, dtype=torch.float32) * 4
        t4 = torch.ones(6, dtype=torch.int32) * 5
        b2.add(t3)
        b2.add(t4)
        b2.allocate()

        # Signal that shm is ready
        ready_event.set()

        # Wait for secondary processes to finish
        finished_event.wait(timeout=30)
    finally:
        # Always signal ready event so secondary process doesn't hang
        ready_event.set()
        # Clean up resources
        if manager is not None:
            manager.release()


def secondary_process_1(shm_name_prefix, ready_event, finished_event, pinned_memory):
    """First secondary process that reads from shared memory."""
    manager = None
    try:
        # Wait for main process to set up shm
        ready_event.wait(timeout=10)

        # Create manager as non-owner (load_from_shm=True means we connect to existing SHM)
        manager = AllocationManager(
            shm_block_name_prefix=shm_name_prefix,
            load_from_shm=True,
            pinned_memory=pinned_memory,
            host_pinner=HostPinner(),
            memory_alignment=128,
        )

        # Connect to existing first block (don't own the shm)
        b1 = manager.block(device="cpu")
        t1 = torch.ones(5, dtype=torch.float32) * 20
        t2 = torch.ones(3, dtype=torch.float64) * 30
        b1.add(t1)
        b1.add(t2)
        views1 = b1.allocate()

        # Connect to existing second block (don't own the shm)
        b2 = manager.block(device="cpu")
        t3 = torch.ones(4, dtype=torch.float32) * 40
        t4 = torch.ones(6, dtype=torch.int32) * 50
        b2.add(t3)
        b2.add(t4)
        views2 = b2.allocate()

        # Verify data from shm matches expected values
        expected_t1 = torch.ones(5, dtype=torch.float32) * 2
        expected_t2 = torch.ones(3, dtype=torch.float64) * 3
        expected_t3 = torch.ones(4, dtype=torch.float32) * 4
        expected_t4 = torch.ones(6, dtype=torch.int32) * 5

        assert torch.allclose(views1[0], expected_t1), "Block 1 tensor 1 data mismatch"
        assert torch.allclose(views1[1], expected_t2), "Block 1 tensor 2 data mismatch"
        assert torch.allclose(views2[0], expected_t3), "Block 2 tensor 1 data mismatch"
        assert torch.allclose(views2[1], expected_t4), "Block 2 tensor 2 data mismatch"

        print("Secondary process 1: Successfully verified block 1 data from shm")
    finally:
        # Always signal finished event so owner process doesn't hang
        finished_event.set()
        # Clean up resources
        if manager is not None:
            manager.release()


class TestMultiprocessSharedMemory:
    """Test multiprocess shared memory functionality."""

    def test_shm_without_pinned_memory(self):
        self._run_workflow(pinned_memory=False)

    @pytest.mark.skipif(
        not is_available(),
        reason="requires CUDA host registration for pinned shared memory",
    )
    def test_shm_with_pinned_memory(self):
        self._run_workflow(pinned_memory=True)

    def _run_workflow(self, pinned_memory=False):
        """Test view projection workflow across multiple processes with shared memory."""
        # Use unique prefix to avoid collisions with leftover SHM from previous runs
        shm_name_prefix = f"test_mp_{os.getpid()}_{uuid.uuid4().hex[:8]}"

        # CUDA cannot be reinitialized safely in a forked child. Use one
        # explicit context for both synchronization primitives and workers so
        # the pinned-memory case exercises a fresh CUDA runtime per process.
        context = multiprocessing.get_context("spawn")

        # Create synchronization events
        ready_event = context.Event()
        finished_event = context.Event()

        try:
            # Create processes
            owner_proc = context.Process(
                target=shm_owner_process,
                args=(shm_name_prefix, ready_event, finished_event, pinned_memory),
            )

            secondary_proc1 = context.Process(
                target=secondary_process_1,
                args=(shm_name_prefix, ready_event, finished_event, pinned_memory),
            )

            # Start all processes
            owner_proc.start()
            secondary_proc1.start()

            # Wait for all processes to complete
            owner_proc.join(timeout=60)
            secondary_proc1.join(timeout=60)

            # Verify all processes completed successfully
            assert owner_proc.exitcode == 0, "Owner process failed"
            assert secondary_proc1.exitcode == 0, "Secondary process 1 failed"

            print("All processes completed successfully!")

        finally:
            # Cleanup any remaining processes
            for proc in [owner_proc, secondary_proc1]:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
