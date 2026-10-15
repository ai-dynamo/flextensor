# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar

import pytest
import torch

if TYPE_CHECKING:
    from collections.abc import Iterator

ROOT = Path(__file__).parents[4]
SERVE_PATH = ROOT / "examples" / "ltx23" / "outpaint" / "serve_infer.py"


@pytest.fixture
def serve_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    # Only optional third-party dependencies are stubbed; execute the complete local modules.
    optional_imports = {
        "av": (),
        "huggingface_hub": ("hf_hub_download", "snapshot_download"),
        "ltx_core.components.noisers": ("GaussianNoiser",),
        "ltx_core.loader": ("LTXV_LORA_COMFY_RENAMING_MAP", "LoraPathStrengthAndSDOps"),
        "ltx_core.model.transformer": ("X0Model",),
        "ltx_core.model.video_vae": ("TilingConfig", "get_video_chunks_number"),
        "ltx_core.types": ("Audio", "VideoPixelShape"),
        "ltx_pipelines.ic_lora": ("ICLoraPipeline",),
        "ltx_pipelines.iclora_utils": ("append_ic_lora_reference_video_conditionings",),
        "ltx_pipelines.utils.blocks": ("DiffusionStage", "PromptEncoder"),
        "ltx_pipelines.utils.constants": ("DISTILLED_SIGMAS", "STAGE_2_DISTILLED_SIGMAS"),
        "ltx_pipelines.utils.denoisers": ("SimpleDenoiser",),
        "ltx_pipelines.utils.helpers": ("assert_resolution", "combined_image_conditionings"),
        "ltx_pipelines.utils.media_io": ("encode_video", "get_videostream_metadata"),
        "ltx_pipelines.utils.types": ("ModalitySpec",),
    }
    for name, symbols in optional_imports.items():
        stub = ModuleType(name)
        stub.__dict__.update(dict.fromkeys(symbols))
        monkeypatch.setitem(sys.modules, name, stub)

    # Also restore serve_infer's own sys.path insertion at teardown.
    monkeypatch.syspath_prepend(str(SERVE_PATH.parent))
    for name, path in (
        ("context_parallel", SERVE_PATH.with_name("context_parallel.py")),
        ("serve_http", SERVE_PATH.with_name("serve_http.py")),
        ("_common", SERVE_PATH.parent.parent / "_common.py"),
        ("serve_infer", SERVE_PATH),
    ):
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    return module


class FakeContextParallel:  # noqa: B903 - mirrors the mutable runtime object used by the example.
    def __init__(self, *, enabled: bool = False, size: int = 1, rank: int = 0) -> None:
        self.enabled = enabled
        self.size = size
        self.rank = rank
        self.local_rank = rank


class FakeX0Model:
    def __init__(self, velocity_model: object) -> None:
        self.velocity_model = velocity_model
        self.eval_calls = 0

    def eval(self) -> FakeX0Model:
        self.eval_calls += 1
        return self


class FakeDiffusionStage:
    build_calls: ClassVar[list[tuple[object, dict[str, Any]]]] = []

    @staticmethod
    def _build_transformer(stage: object, **kwargs: Any) -> object:
        FakeDiffusionStage.build_calls.append((stage, kwargs))
        return SimpleNamespace(label="full-transformer")


class FakePromptEncoder:
    pass


class FakeFlexTensor:
    def __init__(self) -> None:
        self.profile_calls: list[tuple[object, str, object, str]] = []

    def offload_from_profile(self, model: object, directory: str, *, config: object, name: str) -> object:
        self.profile_calls.append((model, directory, config, name))
        return SimpleNamespace(label="legacy-profile-proxy")

    def offload(self, model: object, *, config: object, name: str) -> object:
        return SimpleNamespace(label="profile-proxy")


def configure_cache_dependencies(
    serve_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> tuple[FakeFlexTensor, list[tuple[object, FakeContextParallel]]]:
    fake_flextensor = FakeFlexTensor()
    context_parallel_installs: list[tuple[object, FakeContextParallel]] = []

    def install_context_parallel(model: object, runtime: FakeContextParallel) -> None:
        context_parallel_installs.append((model, runtime))

    for name, dependency in {
        "DiffusionStage": FakeDiffusionStage,
        "flextensor": fake_flextensor,
        "install_context_parallel": install_context_parallel,
        "PromptEncoder": FakePromptEncoder,
        "X0Model": FakeX0Model,
    }.items():
        monkeypatch.setattr(serve_module, name, dependency)
    monkeypatch.setattr(FakeDiffusionStage, "build_calls", [])
    monkeypatch.setattr(FakePromptEncoder, "_text_encoder_ctx", None, raising=False)
    return fake_flextensor, context_parallel_installs


def test_cache_uses_shared_session_lazily_for_stage_and_keeps_disabled_path_unchanged(
    serve_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_flextensor, context_parallel_installs = configure_cache_dependencies(serve_module, monkeypatch)
    config = SimpleNamespace(include_patterns=["blocks.*"])
    context_parallel = FakeContextParallel()
    stage = SimpleNamespace(
        _device="cuda:0",
        _dtype="bfloat16",
        _transformer_builder=SimpleNamespace(),
    )

    disabled_cache = serve_module.RequestTransformerCache(
        config,
        Path("/profiles"),
        from_profile=True,
        context_parallel=context_parallel,
    )
    disabled_result = disabled_cache.build_transformer(stage, 1, video_tools="tools")
    assert disabled_result.label == "legacy-profile-proxy"
    assert FakeDiffusionStage.build_calls == [(stage, {"video_tools": "tools", "device": torch.device("cpu")})]
    assert fake_flextensor.profile_calls[0][1:] == (
        "/profiles/stage1",
        config,
        "ltx_stage1",
    )
    assert context_parallel_installs == [(fake_flextensor.profile_calls[0][0], context_parallel)]

    class CapturingSession:
        def __init__(self) -> None:
            self.call: dict[str, Any] | None = None

        def offload_from_profile(
            self,
            profile_directory: str,
            *,
            config: object,
            name: str,
            **kwargs: Any,
        ) -> object:
            self.call = {
                "profile_directory": profile_directory,
                "config": config,
                "name": name,
                **kwargs,
            }
            return SimpleNamespace(label="shm-proxy")

    session = CapturingSession()
    context_parallel_installs.clear()
    FakeDiffusionStage.build_calls.clear()
    enabled_cache = serve_module.RequestTransformerCache(
        config,
        Path("/profiles"),
        from_profile=True,
        context_parallel=context_parallel,
        shared_session=session,
    )
    result = enabled_cache.build_transformer(stage, 2, marker="value")

    assert result.label == "shm-proxy"
    assert session.call is not None
    assert session.call["name"] == "ltx_stage2"
    assert session.call["profile_directory"] == "/profiles/stage2"
    assert session.call["config"] is config
    assert session.call["materialize_device"] == "cuda:0"
    assert FakeDiffusionStage.build_calls == []
    assert context_parallel_installs == []

    full_transformer = session.call["creator_model_factory"]()
    assert FakeDiffusionStage.build_calls == [(stage, {"marker": "value", "device": torch.device("cpu")})]
    assert context_parallel_installs == [(full_transformer, context_parallel)]


def test_cache_shm_meta_builders_use_official_ltx_builder_flow(
    tmp_path: Path, serve_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, context_parallel_installs = configure_cache_dependencies(serve_module, monkeypatch)
    events: list[tuple[Any, ...]] = []
    context_parallel = FakeContextParallel(enabled=True, size=2, rank=1)
    (tmp_path / "parallelism.json").write_text(json.dumps({"context_parallel_size": 2}))

    class RawModel:
        def to(self, *, dtype: object) -> RawModel:
            events.append(("to", dtype))
            return self

        def eval(self) -> RawModel:
            events.append(("raw_eval",))
            return self

    raw = RawModel()

    class Builder:
        @property
        def module_ops(self) -> tuple[str, ...]:
            events.append(("module_ops",))
            return ("op",)

        def model_config(self) -> dict[str, int]:
            events.append(("config",))
            return {"layers": 48}

        def meta_model(self, config: object, module_ops: object) -> RawModel:
            events.append(("meta", config, module_ops))
            return raw

        def build(self, *, device: object, dtype: object) -> RawModel:
            events.append(("full_text", device, dtype))
            return raw

    class CapturingSession:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def offload_from_profile(
            self,
            profile_directory: str,
            *,
            config: object,
            name: str,
            **kwargs: Any,
        ) -> object:
            self.calls.append({
                "profile_directory": profile_directory,
                "config": config,
                "name": name,
                **kwargs,
            })
            return SimpleNamespace(label=f"{name}-proxy")

    session = CapturingSession()
    config = SimpleNamespace(include_patterns=["blocks.*"])
    text_config = SimpleNamespace(include_patterns=["layers.*"])
    cache = serve_module.RequestTransformerCache(
        config,
        tmp_path,
        from_profile=True,
        context_parallel=context_parallel,
        text_config=text_config,
        shared_session=session,
    )
    cache._install_text()
    prompt_encoder = SimpleNamespace(
        _device="cuda:7",
        _dtype="bfloat16",
        _text_encoder_builder=Builder(),
    )
    with FakePromptEncoder._text_encoder_ctx(prompt_encoder) as managed_text:
        assert managed_text.label == "ltx_text-proxy"

    text_call = session.calls[0]
    assert events == []
    text_call["creator_model_factory"]()
    assert events == [("full_text", torch.device("cpu"), "bfloat16"), ("raw_eval",)]
    events.clear()
    assert text_call["follower_model_factory"]() is raw
    assert events == [
        ("config",),
        ("module_ops",),
        ("meta", {"layers": 48}, ("op",)),
        ("to", "bfloat16"),
        ("raw_eval",),
    ]

    stage = SimpleNamespace(
        _device="cuda:7",
        _dtype="bfloat16",
        _transformer_builder=Builder(),
    )
    cache.build_transformer(stage, 1)
    stage_call = session.calls[1]
    assert context_parallel_installs == []

    events.clear()
    wrapped = stage_call["follower_model_factory"]()
    assert isinstance(wrapped, FakeX0Model)
    assert wrapped.velocity_model is raw
    assert wrapped.eval_calls == 1
    assert events == [
        ("config",),
        ("module_ops",),
        ("meta", {"layers": 48}, ("op",)),
        ("to", "bfloat16"),
    ]
    assert context_parallel_installs == [(wrapped, context_parallel)]


def test_meta_builder_casts_checkpoint_placeholders_but_preserves_constructor_buffers(serve_module: ModuleType) -> None:
    raw_model = torch.nn.Module()
    raw_model.weight = torch.nn.Parameter(torch.empty(2, device="meta", dtype=torch.float32))
    raw_model.register_buffer("checkpoint_buffer", torch.empty(2, device="meta", dtype=torch.float32))
    raw_model.register_buffer("embed_scale", torch.tensor(8.0, device="cpu", dtype=torch.float32))

    class Builder:
        module_ops: tuple[()] = ()

        @staticmethod
        def model_config() -> dict[str, int]:
            return {"layers": 1}

        @staticmethod
        def meta_model(_config: object, _module_ops: object) -> torch.nn.Module:
            return raw_model

    result = serve_module.build_meta_model(Builder(), dtype=torch.bfloat16)

    assert result.weight.device.type == "meta"
    assert result.weight.dtype is torch.bfloat16
    assert result.checkpoint_buffer.device.type == "meta"
    assert result.checkpoint_buffer.dtype is torch.bfloat16
    assert result.embed_scale.device.type == "cpu"
    assert result.embed_scale.dtype is torch.float32


@pytest.mark.parametrize("timeout", [0.0, 17.0])
def test_parse_args_keeps_shm_options_serve_only(
    serve_module: ModuleType, monkeypatch: pytest.MonkeyPatch, timeout: float
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "serve_infer.py",
            "serve",
            "--conditioning-video",
            "input.mp4",
            "--shm-namespace",
            "run",
            "--shm-wait-timeout",
            str(timeout),
        ],
    )
    args = serve_module.parse_args()
    assert (args.shm_namespace, args.shm_wait_timeout) == ("run", timeout)

    monkeypatch.setattr(sys, "argv", ["serve_infer.py", "profile", "--conditioning-video", "input.mp4"])
    profile_args = serve_module.parse_args()
    assert not hasattr(profile_args, "shm_namespace")


@pytest.mark.parametrize("namespace, timeout", [("bad/name", "17"), ("run", "-1")])
def test_parse_args_rejects_invalid_shm_options(
    serve_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    namespace: str,
    timeout: str,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "serve_infer.py",
            "serve",
            "--conditioning-video",
            "input.mp4",
            "--shm-namespace",
            namespace,
            "--shm-wait-timeout",
            timeout,
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        serve_module.parse_args()
    assert exc_info.value.code == 2
    assert "error: Shared offload" in capsys.readouterr().err


@pytest.mark.parametrize("namespace", [None, "run"])
def test_serve_scopes_shared_session_around_follower_lifecycle(
    serve_module: ModuleType, monkeypatch: pytest.MonkeyPatch, namespace: str | None
) -> None:
    events: list[str] = []
    sessions: list[Any] = []

    class Session:
        def __init__(self, actual_namespace: str, *, wait_timeout: float) -> None:
            assert actual_namespace == namespace
            assert wait_timeout == 17.0
            self.active = False
            sessions.append(self)
            events.append("session-init")

        def __enter__(self) -> Session:
            self.active = True
            events.append("session-enter")
            return self

        def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
            events.append("session-exit")
            self.active = False

    class Service:
        def __init__(
            self,
            _args: object,
            _context_parallel: object,
            *,
            from_profile: bool,
            shared_session: Session | None,
        ) -> None:
            assert from_profile is True
            assert shared_session is None or shared_session.active
            self.shared_session = shared_session
            events.append("service")

    class FollowerContext(FakeContextParallel):
        is_leader = False

        def barrier(self) -> None:
            events.append("barrier")

    def leader_requested_shutdown(*_args: object) -> bool:
        events.append("shutdown-check")
        return False

    def run_follower(service: Service, _context_parallel: object) -> None:
        assert service.shared_session is None or service.shared_session.active
        events.append("follower")

    for name, dependency in {
        "_leader_requested_shutdown": leader_requested_shutdown,
        "_run_context_parallel_follower": run_follower,
        "flextensor": SimpleNamespace(SharedOffloadSession=Session),
        "OutpaintFlexTensorService": Service,
    }.items():
        monkeypatch.setattr(serve_module, name, dependency)

    serve_module.serve(
        SimpleNamespace(shm_namespace=namespace, shm_wait_timeout=17.0),
        FollowerContext(),
        SimpleNamespace(),
        SimpleNamespace(),
    )

    if namespace is None:
        assert events == ["service", "barrier", "shutdown-check", "follower"]
        assert sessions == []
    else:
        assert events == [
            "session-init",
            "session-enter",
            "service",
            "barrier",
            "shutdown-check",
            "follower",
            "session-exit",
        ]
        assert len(sessions) == 1
        assert not sessions[0].active


def test_serve_scopes_shared_session_around_leader_lifecycle(
    serve_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    session = SimpleNamespace(active=False)

    @contextmanager
    def shared_session(namespace: str, *, wait_timeout: float) -> Iterator[SimpleNamespace]:
        assert namespace == "run"
        assert wait_timeout == 17.0
        session.active = True
        events.append("session-enter")
        try:
            yield session
        finally:
            session.active = False
            events.append("session-exit")

    class Service:
        def __init__(
            self,
            _args: object,
            _context_parallel: object,
            *,
            from_profile: bool,
            shared_session: object,
        ) -> None:
            assert from_profile is True
            assert shared_session is session
            assert session.active
            events.append("service")

        def generate(self, payload: dict[str, str]) -> dict[str, str]:
            assert payload == {"output_path": "warmup.mp4"}
            assert session.active
            events.append("warmup")
            return {}

    def serve_forever() -> None:
        assert session.active
        events.append("serve-forever")

    def finalize_follower_shutdown(
        _context_parallel: object, *, http_loop_stopped: bool, request_in_flight: bool, logger: object
    ) -> None:
        assert session.active
        assert http_loop_stopped
        assert not request_in_flight
        events.append("finalize-shutdown")

    server = SimpleNamespace(serve_forever=serve_forever, request_in_flight=False)
    context_parallel = SimpleNamespace(is_leader=True, enabled=True, size=2, barrier=lambda: None)
    shutdown = SimpleNamespace(requested=False, attach_server=lambda _server: None)
    for name, dependency in {
        "flextensor": SimpleNamespace(SharedOffloadSession=shared_session),
        "OutpaintFlexTensorService": Service,
        "ReplicaRequestServer": lambda *_args, **_kwargs: server,
        "_leader_requested_shutdown": lambda *_args: False,
        "finalize_follower_shutdown": finalize_follower_shutdown,
    }.items():
        monkeypatch.setattr(serve_module, name, dependency)

    serve_module.serve(
        SimpleNamespace(
            shm_namespace="run",
            shm_wait_timeout=17.0,
            host="127.0.0.1",
            port=8020,
            warmup_output_path="warmup.mp4",
        ),
        context_parallel,
        shutdown,
        SimpleNamespace(received_signal=None),
    )

    assert events == ["session-enter", "service", "warmup", "serve-forever", "finalize-shutdown", "session-exit"]
    assert not session.active
