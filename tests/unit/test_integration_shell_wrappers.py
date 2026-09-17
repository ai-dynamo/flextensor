# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the environment passed by integration entrypoints to test processes."""

import os
import shutil
import subprocess  # noqa: S404 - exercise the integration shell entrypoints
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_SCRIPTS = sorted((REPO_ROOT / "tests" / "integration").rglob("test.sh"))


@pytest.mark.parametrize("script", TEST_SCRIPTS, ids=lambda path: path.parent.name)
@pytest.mark.parametrize("visible_devices", [None, "", "0", "4,5,6,7", "GPU-uuid-1,GPU-uuid-2"])
def test_integration_scripts_preserve_cuda_visible_devices(
    script: Path, visible_devices: str | None, tmp_path: Path
) -> None:
    # Run unmodified entrypoints in a temporary tree to isolate generated results.
    for source in (
        script,
        REPO_ROOT / "tests/integration/_install_requirements.sh",
        REPO_ROOT / "examples/vllm/install.sh",
    ):
        destination = tmp_path / source.relative_to(REPO_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # Replace dependencies, but keep the shell flow through timeout and Python launch.
    python_stub = (
        '#!/bin/sh\ncase "$1" in -c|-) exit 0;; esac\n'
        'printf "%s:%s\\n" "${CUDA_VISIBLE_DEVICES+x}" "${CUDA_VISIBLE_DEVICES-}" >> "$CUDA_MASK_LOG"\n'
    )
    for name, content in {
        "python": python_stub,
        "python3": python_stub,
        "pip": "#!/bin/sh\nexit 0\n",
        "timeout": '#!/bin/sh\nshift\nexec "$@"\n',
    }.items():
        executable = bin_dir / name
        executable.write_text(content)
        executable.chmod(0o755)

    mask_log = tmp_path / "cuda_masks.txt"
    env = {
        **os.environ,
        "CI": "true",
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "CUDA_MASK_LOG": str(mask_log),
    }
    env.pop("CUDA_VISIBLE_DEVICES", None)
    if visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = visible_devices

    result = subprocess.run(  # noqa: S603 - repository script with stubbed test commands
        ["/bin/bash", str(tmp_path / script.relative_to(REPO_ROOT))],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    masks = mask_log.read_text().splitlines()
    assert masks
    assert set(masks) == {":" if visible_devices is None else f"x:{visible_devices}"}
