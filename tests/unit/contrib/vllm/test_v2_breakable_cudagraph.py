# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import logging
import sys
import types

import pytest

MODULE_NAME = "flextensor.contrib.vllm.v2.breakable_cudagraph"


@pytest.fixture
def fake_vllm(monkeypatch):
    events: list[str] = []

    class BreakableCUDAGraphCapture:
        def __init__(self) -> None:
            self._capturing = True

        def _end_segment(self) -> None:
            if self._capturing:
                events.append("capture-end")
                self._capturing = False

        def add_eager(self, fn):
            self._end_segment()
            result = fn()
            self._capturing = True
            events.append("capture-begin")
            return result

    class Offloader:
        def join_after_forward(self) -> None:
            events.append("offloader-join")

    offloader = Offloader()
    breakable = types.ModuleType("vllm.compilation.breakable_cudagraph")
    breakable.BreakableCUDAGraphCapture = BreakableCUDAGraphCapture
    offloader_base = types.ModuleType("vllm.model_executor.offloader.base")
    offloader_base.get_offloader = lambda: offloader
    vllm_logger = types.ModuleType("vllm.logger")
    vllm_logger.init_logger = logging.getLogger
    modules = {
        "vllm": types.ModuleType("vllm"),
        "vllm.logger": vllm_logger,
        "vllm.compilation": types.ModuleType("vllm.compilation"),
        "vllm.compilation.breakable_cudagraph": breakable,
        "vllm.model_executor": types.ModuleType("vllm.model_executor"),
        "vllm.model_executor.offloader": types.ModuleType("vllm.model_executor.offloader"),
        "vllm.model_executor.offloader.base": offloader_base,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, MODULE_NAME, raising=False)
    module = importlib.import_module(MODULE_NAME)
    yield module, BreakableCUDAGraphCapture, events
    sys.modules.pop(MODULE_NAME, None)


def test_offloader_joins_before_eager_break_ends_segment(fake_vllm) -> None:
    module, capture_cls, events = fake_vllm

    assert module.install_breakable_segment_join()
    result = capture_cls().add_eager(lambda: events.append("eager") or "out")

    assert result == "out"
    assert events == ["offloader-join", "capture-end", "eager", "capture-begin"]


def test_no_join_when_capture_is_not_recording(fake_vllm) -> None:
    module, capture_cls, events = fake_vllm
    module.install_breakable_segment_join()
    capture = capture_cls()
    capture._capturing = False

    capture.add_eager(lambda: events.append("eager"))

    assert events == ["eager", "capture-begin"]


def test_install_is_idempotent(fake_vllm) -> None:
    module, capture_cls, events = fake_vllm

    assert module.install_breakable_segment_join()
    installed = capture_cls.add_eager
    assert module.install_breakable_segment_join()

    assert capture_cls.add_eager is installed
    capture_cls().add_eager(lambda: None)
    assert events.count("offloader-join") == 1


def test_install_without_breakable_cudagraph_support(fake_vllm, monkeypatch) -> None:
    module, _capture_cls, _events = fake_vllm
    monkeypatch.setitem(sys.modules, "vllm.compilation.breakable_cudagraph", None)

    assert module.install_breakable_segment_join() is False
