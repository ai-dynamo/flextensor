# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User hook behavior through real discovery, profiling, and inference swaps."""

import copy

import pytest
import torch
from torch import nn

import flextensor as ft

pytestmark = [
    pytest.mark.gpu_vram_min_24g,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]


@pytest.mark.parametrize(
    ("profile_mode", "transfer_mode"),
    [
        ("view", "allocation_block_transfer"),
        ("view", "strategy"),
        ("getter", "strategy"),
        ("torch_function", "strategy"),
    ],
)
def test_phase_transitions_preserve_hook_output_and_removal(profile_mode, transfer_mode):
    torch.manual_seed(0)
    model = nn.Sequential(*(nn.Linear(8, 8) for _ in range(3))).eval()
    inputs = torch.ones(2, 8, device="cuda")
    with torch.no_grad():
        expected = copy.deepcopy(model).cuda()(inputs)
    calls = []

    def hook(_module, _args, output):
        calls.append(1)
        return output + 1

    name = f"hook-transfer-{profile_mode}-{transfer_mode}"
    try:
        proxy = ft.offload(
            model,
            ft.OffloadConfig(
                include_patterns=["0", "1", "2"],
                profile_mode=profile_mode,
                transfer_mode=transfer_mode,
                profiling_iters=3,
                pinned_memory=False,
            ),
            name=name,
        )
        handle = proxy[2].register_forward_hook(hook)
        manager = ft.get_offload_manager(name)
        with torch.no_grad():
            for _ in range(manager.iters_before_inference + 2):
                before = len(calls)
                output = proxy(inputs)
                assert len(calls) - before == 1
                torch.testing.assert_close(output, expected + 1)

            handle.remove()
            before = len(calls)
            torch.testing.assert_close(proxy(inputs), expected)
            assert len(calls) == before
    finally:
        ft.release(name=name)
