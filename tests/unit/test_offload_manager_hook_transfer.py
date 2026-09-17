# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hook transfer with the shared registries used by real phase model copies."""

import copy

import pytest
import torch
from torch import nn

from flextensor.offload_manager import OffloadManager
from flextensor.tensor_processors import create_model_with_shared_tensors


@pytest.mark.parametrize("copy_model", [copy.copy, create_model_with_shared_tensors], ids=["shallow", "shared-tensors"])
@pytest.mark.parametrize("hook_kind", ["forward", "forward_pre", "full_backward", "full_backward_pre"])
@pytest.mark.parametrize("registrations", [1, 2])
def test_shared_hook_transfer_preserves_calls_and_removable_handles(copy_model, hook_kind, registrations):
    original = nn.Linear(2, 2)
    calls = []

    def hook(*_args):
        calls.append(1)

    register = getattr(original, f"register_{hook_kind}_hook")
    # Registering the same callable twice is intentional and must remain valid.
    handles = [register(hook) for _ in range(registrations)]
    manager = OffloadManager("shared-hook-transfer")
    profile = copy_model(original)
    manager._transfer_hooks(original, profile)
    inference = copy_model(profile)
    manager._transfer_hooks(profile, inference)

    for remaining in range(registrations, -1, -1):
        calls.clear()
        inference(torch.ones(1, 2, requires_grad=True)).sum().backward()
        assert calls == [1] * remaining
        if handles:
            handles.pop().remove()


def test_transfer_checks_each_hook_registry_independently():
    original = nn.Linear(2, 2)
    calls = []

    def pre_hook(_module, _args):
        calls.append("pre")

    def post_hook(_module, _args, _output):
        calls.append("post")

    original.register_forward_pre_hook(pre_hook)
    handle = original.register_forward_hook(post_hook)
    copied = copy.copy(original)
    # One registry needs transfer even though the other is already shared.
    copied._forward_pre_hooks = type(original._forward_pre_hooks)()
    manager = OffloadManager("partly-shared-hook-transfer")
    manager._transfer_hooks(original, copied)

    copied(torch.ones(1, 2))
    assert calls == ["pre", "post"]
    handle.remove()
    calls.clear()
    copied(torch.ones(1, 2))
    assert calls == ["pre"]
