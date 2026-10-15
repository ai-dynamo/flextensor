# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from flextensor.lazy_model_init import ModulePatcher


@pytest.mark.parametrize("registration", ["buffer", "parameter"])
def test_patched_registration_requires_saved_callback(monkeypatch, registration: str) -> None:
    monkeypatch.setattr(ModulePatcher, f"_original_register_{registration}", None)
    module = torch.nn.Module()

    with pytest.raises(RuntimeError, match=f"original register_{registration} callback is missing"):
        if registration == "buffer":
            ModulePatcher._patched_register_buffer(module, "value", None)
        else:
            ModulePatcher._patched_register_parameter(module, "value", None)
