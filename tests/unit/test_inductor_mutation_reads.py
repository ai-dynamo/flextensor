# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression for mutation markers changing Inductor's normalization tiling."""

import inspect
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from threading import Event
from types import MethodType

import pytest
import torch
from torch.nn import functional

from flextensor.compiled_offload import build_compiled_offload_forward
from flextensor.compiler_utils import patch_inductor_mutation_reads


@pytest.fixture(autouse=True)
def restore_inductor_state(monkeypatch):
    monkeypatch.setattr("flextensor.compiler_utils._INDUCTOR_PATCH_WARNED", False)
    tiling = pytest.importorskip("torch._inductor.tiling_utils")
    config = pytest.importorskip("torch.compiler.config")
    cache = pytest.importorskip("torch._inductor.codecache")
    monkeypatch.setattr(cache.FxGraphHashDetails, "__init__", inspect.unwrap(cache.FxGraphHashDetails.__init__))
    monkeypatch.setattr(tiling, "extract_normalized_read_writes", inspect.unwrap(tiling.extract_normalized_read_writes))
    monkeypatch.setattr(config, "cache_key_tag", "caller-cache-tag")


@pytest.fixture
def unfixed_inductor():
    import torch._inductor.tiling_utils as tiling

    source = inspect.getsource(tiling.extract_normalized_read_writes)
    if "        for inp in inputs:\n            for expr in body.get_all_read_expr(inp):" not in source:
        pytest.skip("installed Inductor does not have the known unfixed lookup")


def test_mutation_fix_separates_cache_once(unfixed_inductor):
    import torch._inductor.tiling_utils as tiling
    import torch.compiler.config as config
    from torch._inductor.codecache import FxGraphHashDetails

    original = tiling.extract_normalized_read_writes
    assert patch_inductor_mutation_reads()
    corrected = tiling.extract_normalized_read_writes
    assert corrected is not original
    hash_init = FxGraphHashDetails.__init__
    details = FxGraphHashDetails(torch.fx.symbolic_trace(torch.nn.ReLU()), [torch.ones(4)], {}, [])
    assert details.cache_key_tag == "caller-cache-tag|flextensor-mutation-reads-v1"
    assert config.cache_key_tag == "caller-cache-tag"
    assert patch_inductor_mutation_reads()
    assert tiling.extract_normalized_read_writes is corrected
    assert FxGraphHashDetails.__init__ is hash_init


def test_mutation_fix_waits_for_active_torch_compile(unfixed_inductor):
    import torch._inductor.tiling_utils as tiling
    from torch._inductor.codecache import compiled_fx_graph_hash

    compiling = Event()
    resume = Event()
    installing = Event()
    original = tiling.extract_normalized_read_writes

    def backend(graph, inputs):
        before = compiled_fx_graph_hash(graph, inputs, {}, [])[0]
        compiling.set()
        assert resume.wait(10), "test did not release the active compilation"
        assert compiled_fx_graph_hash(graph, inputs, {}, [])[0] == before
        assert tiling.extract_normalized_read_writes is original
        return graph.forward

    def install():
        installing.set()
        return patch_inductor_mutation_reads()

    compiled = torch.compile(torch.nn.ReLU(), backend=backend, fullgraph=True)
    with ThreadPoolExecutor(max_workers=2) as workers:
        compilation = workers.submit(compiled, torch.ones(4))
        try:
            assert compiling.wait(10), "torch.compile did not reach the backend"
            installation = workers.submit(install)
            assert installing.wait(10), "patch installer did not start"
            with pytest.raises(FutureTimeoutError):
                installation.result(timeout=0.5)
        finally:
            resume.set()
        torch.testing.assert_close(compilation.result(timeout=10), torch.ones(4))
        assert installation.result(timeout=10)
    assert tiling.extract_normalized_read_writes is not original


@pytest.mark.parametrize("source_kind", ["unknown", "comment", "fixed"])
def test_mutation_fix_leaves_other_implementations_alone(monkeypatch, caplog, source_kind):
    import torch._inductor.tiling_utils as tiling
    import torch.compiler.config as config

    original = tiling.extract_normalized_read_writes
    source = inspect.getsource(original)
    if source_kind == "fixed":
        source = source.replace(
            "        for inp in inputs:\n",
            "        for inp in inputs:\n            inp = V.graph.scheduler.mutation_real_name.get(inp, inp)\n",
        )
    elif source_kind == "comment":
        source = source.replace("for inp in inputs:", "for inp in inputs:  # upstream comment")
    else:
        source = "def extract_normalized_read_writes(node): return None\n"
    monkeypatch.setattr(inspect, "getsource", lambda _fn: source)
    assert not patch_inductor_mutation_reads()
    assert tiling.extract_normalized_read_writes is original
    assert config.cache_key_tag == "caller-cache-tag"
    assert not patch_inductor_mutation_reads()
    warnings = [r for r in caplog.records if "compatibility fix not applied" in r.message]
    assert len(warnings) == (0 if source_kind == "fixed" else 1)


@pytest.fixture
def mock_cuda_device_properties():
    """Compilation needs the real device properties, not the unit-suite mock."""
    yield


class _NormResidual(torch.nn.Module):
    def forward(self, x, temb, table, weight, bias, norm_weight, norm_bias):
        shift, scale, gate, *_ = (table + temb.float()).chunk(6, dim=1)
        norm = (functional.layer_norm(x.float(), (1536,), eps=1e-6) * (1 + scale) + shift).to(x.dtype)
        hidden = (x.float() + functional.linear(norm, weight, bias) * gate).to(x.dtype)
        return functional.layer_norm(hidden.float(), (1536,), norm_weight, norm_bias, eps=1e-6).to(x.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Inductor")
def test_ft_boundary_preserves_compiled_normalization(monkeypatch):
    """Removing the mutation-read correction changes BF16 output on Torch 2.14."""
    import torch.compiler.config as config

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        with torch.random.fork_rng(devices=[0]), torch.inference_mode():
            torch.manual_seed(42)
            args = (
                torch.randn(1, 1536, 320, device="cuda", dtype=torch.bfloat16).transpose(1, 2),
                torch.randn(1, 6, 1536, device="cuda", dtype=torch.bfloat16),
                torch.randn(1, 6, 1536, device="cuda"),
                torch.randn(1536, 1536, device="cuda", dtype=torch.bfloat16) * 0.02,
                torch.randn(1536, device="cuda", dtype=torch.bfloat16),
                torch.randn(1536, device="cuda"),
                torch.randn(1536, device="cuda"),
            )
            # Compile the unmodified reference before setting up any FT forwards.
            reference = torch.compile(_NormResidual(), fullgraph=True)(*args)
            candidate = _NormResidual()
            with config.patch(cache_key_tag="model-setup"):
                candidate.forward = MethodType(
                    build_compiled_offload_forward(_NormResidual.forward, "probe"), candidate
                )
            # Compile after setup's cache tag has been restored.
            actual = torch.compile(candidate, fullgraph=True)(*args)
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    finally:
        torch.use_deterministic_algorithms(deterministic)


def test_lazy_cache_key_stays_isolated_after_setup_context(unfixed_inductor):
    import torch.compiler.config as config
    from torch._inductor.codecache import compiled_fx_graph_hash

    graph = torch.fx.symbolic_trace(torch.nn.ReLU())
    inputs = [torch.ones(4)]

    def cache_key():
        return compiled_fx_graph_hash(graph, inputs, {}, [])[0]

    before = cache_key()
    with config.patch(cache_key_tag="model-setup"):
        build_compiled_offload_forward(torch.nn.ReLU.forward, "block")
    assert config.cache_key_tag == "caller-cache-tag"
    after = cache_key()  # Lazy compilation: no second call to the patcher.
    assert after != before
    assert cache_key() == after
    with config.patch(cache_key_tag="another-model"):
        assert cache_key() != after
    assert cache_key() == after
