# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from typing import Any

import torch


class TraceTensor(torch.Tensor):
    def to(  # type: ignore[override]
        self,
        device: Any = None,
        dtype: Any = None,
        non_blocking: bool = False,
        copy: bool = False,
        memory_format: Any = torch.preserve_format,
    ) -> Any:
        return super().to(
            device=device,
            dtype=dtype,
            non_blocking=non_blocking,
            copy=copy,
            memory_format=memory_format,
        )

    @staticmethod
    def __new__(cls: Any, data: Any, name: str | None = None) -> Any:
        return super().__new__(cls, data)

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., Any],
        types: tuple[Any, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        res = super().__torch_function__(func, types, args, kwargs)
        if func.__name__[-1] == "_":
            return res
        if isinstance(res, TraceTensor):
            return torch.Tensor(res)
        return res


def wrap_trace_tensor(tensor: Any) -> Any:
    if isinstance(tensor, TraceTensor):
        return tensor
    if isinstance(tensor, torch.Tensor):
        return TraceTensor(tensor)
    return tensor
