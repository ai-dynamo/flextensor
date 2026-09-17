# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strategy profile reload must materialize both offloaded and permanent weights."""

from pathlib import Path

import pytest
import torch
from torch import nn

import flextensor as ft
from flextensor.config import ProfileMode
from flextensor.offload_manager import OffloadPhase

pytestmark = [
    pytest.mark.gpu_vram_min_24g,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


@pytest.mark.parametrize("profile_mode", ["getter", "view"])
@pytest.mark.parametrize("pinned_memory", [False, True])
def test_strategy_profile_reload_from_cpu_matches_fresh_inference(
    tmp_path: Path, profile_mode: ProfileMode, pinned_memory: bool
) -> None:
    """A generated profile must execute after restoring an ordinary CPU model."""
    device = torch.device("cuda", torch.cuda.current_device())
    config = ft.OffloadConfig(
        gpu_device=device.index or 0,
        include_patterns=["0", "1", "2"],
        transfer_mode="strategy",
        profile_mode=profile_mode,
        pinned_memory=pinned_memory,
        profiling_iters=3,
        load_strategy=ft.NthLayerStrategy(1, threshold_mb=0.0),
    )
    model = nn.Sequential(*(nn.Linear(8, 8) for _ in range(3))).eval()
    weights = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    inputs = torch.linspace(-1.0, 1.0, 16, device=device).view(2, 8)
    source_name = f"profile-source-{profile_mode}-{pinned_memory}"
    restored_name = f"profile-restored-{profile_mode}-{pinned_memory}"
    try:
        original = ft.offload(model, config=config, name=source_name)
        source_manager = ft.get_offload_manager(source_name)
        with torch.no_grad():
            for _ in range(source_manager.iters_before_inference):
                original(inputs)
            expected = original(inputs).clone()
        ft.save_profile(str(tmp_path), name=source_name)
        tensor_manager = source_manager.get_tensor_manager()
        assert tensor_manager is not None
        state = tensor_manager.load_state(str(tmp_path))
        assert set(state.gpu_tensors_names) == {"0.weight", "0.bias"}
        assert {stat.name for stats in state.load_strategy.values() for stat in stats} == {
            "1.weight",
            "1.bias",
            "2.weight",
            "2.bias",
        }

        restored_model = nn.Sequential(*(nn.Linear(8, 8) for _ in range(3))).eval()
        restored_model.load_state_dict(weights)
        assert all(parameter.device.type == "cpu" for parameter in restored_model.parameters())
        restored = ft.offload_from_profile(restored_model, str(tmp_path), config=config, name=restored_name)
        assert ft.get_offload_manager(restored_name).phase is OffloadPhase.INFERENCE
        with torch.no_grad():
            for _ in range(3):
                torch.testing.assert_close(restored(inputs), expected)
    finally:
        ft.release(name=restored_name)
        ft.release(name=source_name)
