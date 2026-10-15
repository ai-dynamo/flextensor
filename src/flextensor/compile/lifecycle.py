# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Concrete compiled-offload orchestrator (piecewise custom-ops + replan tail)."""

import logging
import statistics
import threading
import types
from collections.abc import Callable
from typing import Any, cast

from torch import nn

from flextensor.compile.module_swap import (
    resolve_compile_targets,
    resolve_compile_targets_on_model,
)
from flextensor.compile.warmup_tail import CompiledOffloadTailState, WarmupTail
from flextensor.compiled_offload import (
    build_profile_compile_forward,
    bump_dynamo_limits_for_compiled_offload,
)
from flextensor.config import BLOCK_TRANSFER_MODES, OffloadConfig
from flextensor.tensor_discovery import is_offload_patched_module

LOGGER = logging.getLogger(__name__)

# Passive compiled-offload tail: after INFERENCE installs the loader and applies
# ``compile_fn`` per block, FlexTensor rides the caller's own forwards. The first
# ``COMPILED_WARMUP_FORWARDS`` let each block's lazy ``torch.compile`` settle
# (unmeasured); the next window is timed; then strategy is re-planned.
COMPILED_WARMUP_FORWARDS = 3

# Eager profiling seed used only while an explicit compiled replan is active
# (external ``compiled_offload`` path). Real timings then come from the compiled
# measure + replan tail. Must be >= 1 (layers without duration are dropped).
COMPILED_EAGER_PROFILE_FORWARDS = 3

# Unmeasured view-profile forwards while each block's lazy ``compile_fn`` settles.
PROFILE_COMPILE_WARMUP_FORWARDS = 3

# Compiled offload drives ``pre_compute/post_compute`` custom ops through a
# :class:`~flextensor.loaders.PreallocatedLoader`. Strategy transfer mode is
# incompatible (it builds :class:`~flextensor.loaders.TensorStrategyLoader`).
_NEXT_MANAGER_ID = 0
_MANAGER_ID_LOCK = threading.Lock()


def allocate_compiled_offload_manager_id() -> int:
    """Allocate a process-local id for ``custom_ops`` loader registries."""
    global _NEXT_MANAGER_ID
    with _MANAGER_ID_LOCK:
        manager_id = _NEXT_MANAGER_ID
        _NEXT_MANAGER_ID += 1
        return manager_id


class _ProfileInnerCarrier(nn.Module):
    """Thin carrier so ``compile_fn`` can target block compute during view profile.

    ``_owner`` must not be registered as a submodule: the compiled carrier is
    attached on the owner for the profile window, and a registered back-reference
    would form ``owner -> carrier -> owner`` and make ``state_dict`` / ``.to`` /
    ``.train`` recurse indefinitely.
    """

    def __init__(self, owner: nn.Module, unbound_forward: Callable[..., Any]) -> None:
        super().__init__()
        object.__setattr__(self, "_owner", owner)
        self._unbound_forward = unbound_forward

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._unbound_forward(self._owner, *args, **kwargs)


class CompiledOffload:
    """Owns piecewise compile + warm/measure/replan for one :class:`OffloadManager`.

    ``host`` is duck-typed and expected to expose:

    - ``_model``, ``_tensor_manager``, ``_patched_modules``, ``config``
    - ``get_layer_label_by_idx()``
    - ``_install_compiled_forwards()`` (OM still owns forward install for now)
    """

    def __init__(self, host: Any, manager_id: int | None = None) -> None:
        self._host = host
        self.manager_id = manager_id if manager_id is not None else allocate_compiled_offload_manager_id()
        self.compile_fn: Callable[[nn.Module], nn.Module] | None = None
        self.active = False
        self.replan_active = False
        self.profile_active = False
        self.profile_compile_warm_remaining = 0
        # Owner modules that received ``_ft_profile_compiled_inner`` (cleared on teardown).
        self._profile_compiled_blocks: list[nn.Module] = []
        self._tail = WarmupTail(
            warmup_forwards=COMPILED_WARMUP_FORWARDS,
            measure_forwards=1,  # updated when arming from config
        )
        self.substitutions: list[tuple[Callable[[nn.Module], None], nn.Module]] = []

    # -- state mirrors (tests / OM properties poke these) -------------------

    @property
    def tail_state(self) -> CompiledOffloadTailState:
        return self._tail.state

    @tail_state.setter
    def tail_state(self, value: CompiledOffloadTailState) -> None:
        self._tail.state = value

    @property
    def tail_armed(self) -> bool:
        """Whether a replan tail was armed (a replan was actually requested)."""
        return self._tail.armed

    @property
    def replan_applied(self) -> bool | None:
        """``True``/``False`` once the tail finished; ``None`` while unfinished."""
        return self._tail.replan_applied

    @property
    def tail_failure(self) -> BaseException | None:
        return self._tail.failure

    @tail_failure.setter
    def tail_failure(self, value: BaseException | None) -> None:
        self._tail.failure = value

    @property
    def warm_seen(self) -> int:
        return self._tail.warm_seen

    @warm_seen.setter
    def warm_seen(self, value: int) -> None:
        self._tail.warm_seen = value

    @property
    def measure_seen(self) -> int:
        return self._tail.measure_seen

    @measure_seen.setter
    def measure_seen(self, value: int) -> None:
        self._tail.measure_seen = value

    # -- activation ---------------------------------------------------------

    def resolve_activation(
        self,
        effective_config: OffloadConfig,
        compile_fn: Callable[[nn.Module], nn.Module] | None,
    ) -> bool:
        """Resolve compiled-offload flags from ``compile_fn`` / config.

        Path selection:

        * ``compile_fn`` + ``profile_mode='view'`` → compiled view-profile
          (strategy from compiled timings; ``replan_active=False``).
        * ``compile_fn`` + non-view (e.g. ``getter``) → eager profile, compile at
          INFERENCE; ``replan_active=True`` so source weights are preserved for
          :meth:`~flextensor.OffloadManager.request_strategy_replan`.
        * ``external_compile`` (no ``compile_fn``) → ``replan_active`` so
          :meth:`~flextensor.OffloadManager.request_strategy_replan` works
          after an external ``torch.compile``.

        Validates the prospective activation before tearing down any prior
        compiled state, so an invalid re-offload leaves a working session intact.

        Raises:
            ValueError: If compiled offload would activate with a non-block
                ``transfer_mode`` (custom ops require a ``PreallocatedLoader``).
        """
        # Validate the replacement first — do not tear down a live compiled
        # session only to reject the new config.
        if compile_fn is not None or effective_config.external_compile:
            self._require_compiled_transfer_mode(effective_config)

        # Re-offload may call us without an intervening ``release()``.
        self.teardown()
        if compile_fn is not None:
            self.compile_fn = compile_fn
            self.active = True
            self.profile_active = effective_config.profile_mode == "view"
            # View-profile already builds the strategy under compile_fn — no replan.
            # Non-view (e.g. getter) profiles eagerly; replan is expected after
            # compile so source weights must be preserved before the first loader.
            self.replan_active = not self.profile_active
            return True
        if effective_config.external_compile:
            self.active = True
            self.replan_active = True
            self.profile_active = False
            return True
        self.active = False
        self.replan_active = False
        self.profile_active = False
        return False

    @staticmethod
    def _require_compiled_transfer_mode(effective_config: OffloadConfig) -> None:
        """Reject transfer modes that cannot back ``install_active_loader``."""
        if effective_config.transfer_mode in BLOCK_TRANSFER_MODES:
            return
        raise ValueError(
            "compiled offload requires a block transfer_mode "
            f"(allocation_block_transfer or raw_block_transfer); got "
            f"transfer_mode={effective_config.transfer_mode!r}. "
            "transfer_mode='strategy' builds TensorStrategyLoader, which is "
            "incompatible with PreallocatedLoader / pre_compute/post_compute custom ops."
        )

    def arm_non_destructive_first_loader(self) -> None:
        """Ask TensorManager to keep source weights for a post-compile replan.

        Only when :attr:`replan_active` is set (external ``external_compile``, or
        ``compile_fn`` with a non-view profile). Default ``compile_fn`` + view
        builds the strategy under compile and must not retain another model-sized
        host copy beside the pinned CPU blocks.
        """
        tm = self._host._tensor_manager  # noqa: SLF001
        if not (self.active and self.replan_active and tm is not None):
            return
        tm.arm_non_destructive_first_loader()

    # -- schedules ----------------------------------------------------------

    def eager_profiling_iters(self) -> int:
        if self.active and self.replan_active and not self.profile_active:
            return COMPILED_EAGER_PROFILE_FORWARDS
        return cast("int", self._host.config.profiling_iters)

    def measure_forwards(self) -> int:
        return cast("int", self._host.config.profiling_iters)

    def extra_iters_before_inference(self) -> int:
        # Compile-warmup forwards for view compiled-profile (not counted in profiling_iters).
        if self.profile_active:
            return PROFILE_COMPILE_WARMUP_FORWARDS
        return 0

    # -- phase hooks --------------------------------------------------------

    def on_enter_profile(self) -> None:
        if not self.profile_active:
            return
        model = self._host._model  # noqa: SLF001
        if model is None or self.compile_fn is None:
            return
        self.profile_compile_warm_remaining = PROFILE_COMPILE_WARMUP_FORWARDS
        self.install_profile_compiled_forwards(model)
        self.apply_profile_compile_fn(model)
        LOGGER.info(
            "FlexTensor compiled-profile: view profile armed (%d compile-warmup forwards, then "
            "%d measured profile forwards).",
            PROFILE_COMPILE_WARMUP_FORWARDS,
            self._host.config.profiling_iters,
        )

    def on_enter_inference(self) -> None:
        """Wire loader / compile_fn after OM installs compiled forwards."""
        if self.compile_fn is not None:
            # Integrated compile_fn uses setup_inference_no_replan: view-profile
            # already measured under compile_fn; direct/external paths that need
            # a compiled-timing rebuild call request_strategy_replan() explicitly.
            self.setup_inference_no_replan()
        elif self.active:
            self.setup_external_compiled_offload()

    def on_forward(self) -> None:
        """Advance the passive warm → measure → replan tail (INFERENCE only)."""
        self.advance_tail()

    def teardown(self) -> None:
        for setter, original in reversed(self.substitutions):
            try:
                setter(original)
            except Exception:  # cleanup must never raise
                LOGGER.debug(
                    "FlexTensor compiled-offload: block un-wrap ignored an error",
                    exc_info=True,
                )
        self.substitutions.clear()
        for module in self._profile_compiled_blocks:
            module.__dict__.pop("_ft_profile_compiled_inner", None)
        self._profile_compiled_blocks.clear()
        self.profile_compile_warm_remaining = 0
        self.compile_fn = None
        self._tail.reset()

        # Always drop loader ownership for this manager. Do not gate on
        # ``active``: a prior resolve that flipped ``active`` off must not
        # strand an installed custom-op loader until a later clear.
        try:
            from flextensor.custom_ops import clear_active_loader

            clear_active_loader(self.manager_id)
        except Exception:  # cleanup must never raise
            LOGGER.debug(
                "FlexTensor compiled-offload: clear_active_loader ignored an error",
                exc_info=True,
            )

        # Drop TensorManager replan arming / source snapshot so a later
        # eager or view re-offload cannot keep another model-sized host copy.
        tm = self._host._tensor_manager  # noqa: SLF001
        if tm is not None:
            tm.clear_replan_state()

        self.active = False
        self.replan_active = False
        self.profile_active = False
        self.profile_compile_warm_remaining = 0

    # -- inference wiring ---------------------------------------------------

    def setup_external_compiled_offload(self) -> None:
        if self.compile_fn is not None or not self.active:
            return
        self.require_compiled_loader()

    def setup_compiled_tail(self) -> None:
        if self.compile_fn is None or not self.active:
            return
        self.require_compiled_loader()
        self.apply_compile_fn()
        if self.replan_active:
            self.arm_replan_tail(compiled_warm_forwards=0)
        else:
            self._tail.mark_done()

    def setup_inference_no_replan(self) -> None:
        self.require_compiled_loader()
        self.apply_compile_fn()
        self._tail.mark_done()

    def require_compiled_loader(self) -> None:
        from flextensor.custom_ops import clear_active_loader, install_active_loader
        from flextensor.loaders import PreallocatedLoader

        tm = self._host._tensor_manager  # noqa: SLF001
        loader = getattr(tm, "tensor_layer_loader", None) if tm is not None else None
        if loader is None:
            clear_active_loader(self.manager_id)
            raise RuntimeError(
                "FlexTensor compiled-offload: no inference loader after the INFERENCE transition. "
                "pre_compute/post_compute custom ops would stay no-ops (pre-install) or raise "
                "(after install_active_loader armed the manager) and offloaded blocks may "
                "read empty weights. Verify the config uses a block transfer_mode "
                "(allocation_block_transfer or raw_block_transfer) and that "
                "discovery/profiling completed successfully."
            )
        if not isinstance(loader, PreallocatedLoader):
            clear_active_loader(self.manager_id)
            raise RuntimeError(
                "FlexTensor compiled-offload: inference loader "
                f"{type(loader).__name__} is not a PreallocatedLoader. "
                "pre_compute/post_compute custom ops require a block transfer_mode "
                "(allocation_block_transfer or raw_block_transfer); "
                f"got transfer_mode={self._host.config.transfer_mode!r}."
            )
        labels = self._host.get_layer_label_by_idx()
        install_active_loader(loader, self.manager_id)
        LOGGER.info(
            "FlexTensor compiled-offload: registered rolling loader %s (manager_id=%d) for %d offload unit(s).",
            type(loader).__name__,
            self.manager_id,
            len(labels),
        )

    def request_strategy_replan(self) -> int:
        """Arm the passive warm→measure→replan tail; returns forwards to drive.

        For ``external_compile`` after ``torch.compile``, or for ``compile_fn``
        when profile was not under compile (e.g. ``getter``). Not needed for
        default ``compile_fn`` + view-profile.

        CUDA-graph replay (no forward hooks) must use
        :meth:`~flextensor.OffloadManager.request_strategy_replan` with
        ``manual_update_state=True`` instead — that path measures via transfer
        timing, not this custom-op measure tail.

        Returns ``0`` unless :attr:`replan_active` was set at activation (source
        weights retained before the first loader build). Calling this on the
        default view-profile path is a no-op — that path never snapshot
        originals, so a rebuild would be refused.
        """
        if not (self.active and self.replan_active):
            if self.active and not self.replan_active:
                LOGGER.warning(
                    "FlexTensor compiled-offload: request_strategy_replan() ignored — "
                    "replan was not armed (default view-profile keeps source weights "
                    "off; use external_compile=True or profile_mode='getter' when a "
                    "post-compile rebuild is required)."
                )
            elif not self.active:
                LOGGER.warning(
                    "FlexTensor compiled-offload: request_strategy_replan() ignored — compiled offload is not active."
                )
            return 0
        return self.arm_replan_tail(compiled_warm_forwards=0)

    def arm_replan_tail(
        self,
        *,
        compiled_warm_forwards: int = 0,
        enable_profiling: bool = True,
        finish_replan: Callable[[], bool | None] | None = None,
    ) -> int:
        self._tail.measure_forwards = self.measure_forwards()
        remaining = self._tail.arm(
            credited_warm=compiled_warm_forwards,
            enable_profiling=enable_profiling,
        )
        if self._tail.state == CompiledOffloadTailState.MEASURING:
            if self._tail.enable_profiling:
                from flextensor.custom_ops import enable_compiled_profiling

                enable_compiled_profiling(self.manager_id)
            if self._tail.measure_forwards == 0:
                # No measure budget (profiling_iters=0): finish without waiting for a forward.
                self._complete_replan_tail(finish_replan)
                return 0
        remaining_warm = COMPILED_WARMUP_FORWARDS - self._tail.warm_seen
        LOGGER.info(
            "FlexTensor compiled-offload: armed passive re-plan tail "
            "(%d remaining warmup + %d measure forwards ride the caller's loop).",
            remaining_warm,
            self._tail.measure_forwards,
        )
        return remaining

    def apply_compile_fn(self) -> None:
        model = self._host._model  # noqa: SLF001
        if self.compile_fn is None or model is None:
            return
        targets = resolve_compile_targets(model, self._host._patched_modules)  # noqa: SLF001
        if not targets:
            LOGGER.warning(
                "FlexTensor compiled-offload: compile_fn supplied but no offloaded units "
                "found to compile; nothing compiled. (Did any modules match the offload "
                "include patterns?)"
            )
            return
        bump_dynamo_limits_for_compiled_offload(len(targets))
        compiled_count = 0
        for setter, module in targets:
            try:
                new_module = self.compile_fn(module)
            except Exception:
                LOGGER.exception(
                    "FlexTensor compiled-offload: compile_fn raised on an offloaded unit; leaving it eager."
                )
                continue
            setter(new_module)
            self.substitutions.append((setter, module))
            compiled_count += 1
        LOGGER.info(
            "FlexTensor compiled-offload: applied compile_fn to %d/%d offloaded unit(s) (one compiled graph each).",
            compiled_count,
            len(targets),
        )

    # -- profile compile ----------------------------------------------------

    def should_record_profile_compile_duration(self) -> bool:
        """Whether unit timings should be recorded (after compile-warmup model forwards)."""
        return self.profile_compile_warm_remaining <= 0

    def advance_profile_compile_warmup(self) -> None:
        """Consume one model-forward warmup slot (call once per root forward)."""
        if self.profile_compile_warm_remaining > 0:
            self.profile_compile_warm_remaining -= 1

    def install_profile_compiled_forwards(self, model: nn.Module) -> None:
        if self._host._tensor_manager is None:  # noqa: SLF001
            return
        host = self._host
        swapped = 0
        for _name, module in model.named_modules():
            if not is_offload_patched_module(module):
                continue
            original_forward = module.__dict__.get("_ft_original_forward_func")
            offload_name = module.__dict__.get("_ft_offload_name")
            if original_forward is None or offload_name is None:
                continue
            profile_forward = build_profile_compile_forward(
                original_forward,
                offload_name,
                get_tensor_manager=lambda: host._tensor_manager,  # noqa: SLF001
                should_record_duration=self.should_record_profile_compile_duration,
            )
            module.forward = types.MethodType(profile_forward, module)
            swapped += 1
        LOGGER.info(
            "FlexTensor compiled-profile: installed traceable view-profile forwards on %d module(s).",
            swapped,
        )

    def apply_profile_compile_fn(self, model: nn.Module) -> None:
        if self.compile_fn is None:
            return
        targets = resolve_compile_targets_on_model(model)
        if not targets:
            LOGGER.warning(
                "FlexTensor compiled-profile: no offloaded units found on the profile model; profile will stay eager."
            )
            return
        bump_dynamo_limits_for_compiled_offload(len(targets))
        compiled_count = 0
        for _setter, module in targets:
            original_forward = module.__dict__.get("_ft_original_forward_func")
            if original_forward is None:
                continue
            try:
                carrier = _ProfileInnerCarrier(module, original_forward)
                compiled_inner = self.compile_fn(carrier)
            except Exception:
                LOGGER.exception(
                    "FlexTensor compiled-profile: compile_fn raised on %r; leaving block eager.",
                    module.__dict__.get("_ft_offload_name"),
                )
                continue
            # Bypass nn.Module.__setattr__ so this is not registered under _modules.
            module.__dict__["_ft_profile_compiled_inner"] = compiled_inner
            self._profile_compiled_blocks.append(module)
            compiled_count += 1
        LOGGER.info(
            "FlexTensor compiled-profile: applied compile_fn to inner compute of %d/%d block(s).",
            compiled_count,
            len(targets),
        )

    # -- replan tail --------------------------------------------------------

    def advance_tail(self, finish_replan: Callable[[], bool | None] | None = None) -> None:
        finish = finish_replan or self.finish_replan
        state = self._tail.state
        match state:
            case CompiledOffloadTailState.FAILED:
                raise RuntimeError(
                    "FlexTensor compiled-offload: re-plan previously failed; inference is unsafe. Restart the process."
                ) from self._tail.failure
            case CompiledOffloadTailState.IDLE | CompiledOffloadTailState.DONE:
                return
            case CompiledOffloadTailState.WARMING:
                self._tail.warm_seen += 1
                if self._tail.warm_seen >= COMPILED_WARMUP_FORWARDS:
                    if self._tail.enable_profiling:
                        from flextensor.custom_ops import enable_compiled_profiling

                        enable_compiled_profiling(self.manager_id)
                    self._tail.state = CompiledOffloadTailState.MEASURING
                    if self.measure_forwards() == 0:
                        self._complete_replan_tail(finish)
                return
            case CompiledOffloadTailState.MEASURING:
                if self.measure_forwards() == 0:
                    self._complete_replan_tail(finish)
                    return
                self._tail.measure_seen += 1
                if self._tail.measure_seen >= self.measure_forwards():
                    self._complete_replan_tail(finish)
            case _:
                raise RuntimeError(
                    f"FlexTensor compiled-offload: unexpected tail state {state!r}. "
                    "This is an internal error — please report it."
                )

    def _complete_replan_tail(self, finish_replan: Callable[[], bool | None] | None = None) -> None:
        finish = finish_replan or self.finish_replan
        try:
            applied = finish()
        except Exception as exc:
            from flextensor.custom_ops import clear_active_loader

            clear_active_loader(self.manager_id)
            self._tail.mark_failed(exc)
            raise
        # ``False`` = soft keep-current (empty budgets / rebuild refused).
        # ``None`` / ``True`` = treated as applied (void callbacks return None).
        # Still DONE so inference continues; ``replan_applied`` distinguishes.
        if applied is False:
            LOGGER.warning(
                "FlexTensor compiled-offload: re-plan finished without applying a "
                "new strategy; keeping the current loader "
                "(tail DONE, replan_applied=False)."
            )
            self._tail.mark_done(applied=False)
        else:
            self._tail.mark_done(applied=True)

    def finish_replan(self, durations_by_label: dict[str, float] | None = None) -> bool:
        """Rebuild strategy from per-layer compute durations (ms) and reinstall loader.

        When ``durations_by_label`` is omitted, drain compiled custom-op samples
        from the measure tail. Offload-timing / CUDA-graph budgets are applied
        via :meth:`~flextensor.OffloadManager.request_strategy_replan`, which
        drains the collector and passes ``label → ms`` here.
        """
        if durations_by_label is None:
            from flextensor.custom_ops import finish_compiled_profiling

            durations_by_label = self._durations_by_label_from_samples(finish_compiled_profiling(self.manager_id))
            if not durations_by_label:
                LOGGER.warning(
                    "FlexTensor compiled-offload: no compiled per-layer timings captured; "
                    "keeping the eager-profile strategy."
                )
                return False
        elif not durations_by_label:
            return False

        tm = self._host._tensor_manager  # noqa: SLF001
        model = self._host._model  # noqa: SLF001
        if tm is None or model is None:
            return False
        if not tm.replan_from_compiled_durations(durations_by_label, model):
            return False
        self.reinstall_compiled_loader()
        return True

    def _durations_by_label_from_samples(
        self,
        durations: dict[str, list[float]],
    ) -> dict[str, float]:
        return {unit_name: statistics.median(samples) for unit_name, samples in durations.items() if samples}

    def reinstall_compiled_loader(self) -> None:
        from flextensor.custom_ops import clear_active_loader, install_active_loader
        from flextensor.loaders import PreallocatedLoader

        tm = self._host._tensor_manager  # noqa: SLF001
        loader = getattr(tm, "tensor_layer_loader", None) if tm is not None else None
        labels = self._host.get_layer_label_by_idx()
        if loader is None:
            clear_active_loader(self.manager_id)
            raise RuntimeError(
                "FlexTensor compiled-offload: no loader after re-plan; pre_compute/post_compute ops would "
                "stay pointed at the released loader. Inference is unsafe — restart the process."
            )
        if not isinstance(loader, PreallocatedLoader):
            clear_active_loader(self.manager_id)
            raise RuntimeError(
                "FlexTensor compiled-offload: rebuilt loader "
                f"{type(loader).__name__} is not a PreallocatedLoader after re-plan. "
                "Inference is unsafe — restart the process."
            )
        clear_active_loader(self.manager_id)
        install_active_loader(loader, self.manager_id)
        LOGGER.info(
            "FlexTensor compiled-offload: strategy re-planned from compiled timings; "
            "rebuilt loader %s re-installed (manager_id=%d) for %d offload unit(s).",
            type(loader).__name__,
            self.manager_id,
            len(labels),
        )
