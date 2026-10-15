# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise package imports and non-SHM allocation without POSIX dependencies."""

import subprocess  # noqa: S404 - isolated platform simulation
import sys
import textwrap


def test_windows_import_and_allocation_without_posix_dependencies() -> None:
    script = textwrap.dedent(
        """
        import os
        import sys

        # Load third-party dependencies on the real platform before simulating
        # Windows for a fresh FlexTensor import (including beartype annotations).
        import beartype.claw
        import numpy
        import psutil
        import proxytypes3
        import pydantic
        import pytest
        import scipy.optimize
        import shared_memory_dict
        import torch
        import yaml

        os.environ.pop("DISABLE_BEARTYPE", None)
        sys.platform = "win32"
        sys.modules["posix_ipc"] = None
        sys.modules["fcntl"] = None

        import flextensor as ft
        from flextensor.allocation_block import AllocationBlock
        from flextensor.host_pinning import NoOpHostPinner
        from flextensor.shm import FlexibleSharedMemory, ProcessFileLock, SemaphoreLock, SharedMultiString

        config = ft.OffloadConfig(pinned_memory=False)
        assert config.shm_enabled is False
        block = AllocationBlock(device="cpu", host_pinner=NoOpHostPinner())
        source = torch.arange(8, dtype=torch.float32)
        block.add(source)
        views = block.allocate()
        assert len(views) == 1
        assert torch.equal(views[0], source)
        block.release()

        message = "SHM.*not supported on Windows"
        with pytest.raises(ValueError, match=message):
            ft.OffloadConfig(shm_enabled=True, shm_namespace="test")
        with pytest.raises(NotImplementedError, match=message):
            ft.SharedOffloadSession("test")
        with pytest.raises(NotImplementedError, match=message):
            FlexibleSharedMemory("test", shm_size=128)
        for lock_type in (ProcessFileLock, SemaphoreLock):
            with pytest.raises(NotImplementedError, match=message):
                lock_type("test", locked=False)
        with pytest.raises(NotImplementedError, match=message):
            SharedMultiString("test", create=True)
        """
    )
    result = subprocess.run(  # noqa: S603 - fixed test script, no external input
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
