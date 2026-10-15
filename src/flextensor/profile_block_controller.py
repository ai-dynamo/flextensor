# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""View-mode profile block controller.

Backs ``profile_mode='view'``. Owns a single CPU and a single GPU buffer
that hold the profile-time tensors; the profile model is patched with
views into the GPU buffer, so for block-transfer loaders the forward path
during profile matches the access pattern used at inference. See
:class:`ProfileBlockController` for the layout details
and :meth:`flextensor.tensor_manager.TensorManager.prepare_profile_direct_mode_model`
for how the views are wired into the model.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping  # noqa: TC003 (beartype needs runtime symbols for parameter annotations)
from dataclasses import dataclass

import torch

from flextensor.collectors import IterativeLayerStatistics  # noqa: TC001 (beartype needs runtime symbol)
from flextensor.host_pinning import HostPinner, PinnedMemoryMode, make_host_pinner
from flextensor.utils import _DEFAULT_PACKED_TENSOR_ALIGNMENT_BYTES as _DEFAULT_SLOT_ALIGNMENT_BYTES
from flextensor.utils import is_dense_layout

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SlotMeta:
    """Layout metadata for a single tensor's slot inside a block."""

    start: int
    end: int
    shape: torch.Size
    dtype: torch.dtype
    stride: tuple[int, ...]

    def __post_init__(self) -> None:
        # ``_slot_view`` truncates a misaligned ``start`` via integer-division
        # by ``dtype.itemsize``, so the typed view ends up offset from the
        # copied bytes. Layout code keeps ``start`` aligned; this is a guard.
        itemsize = torch.empty(0, dtype=self.dtype).element_size()
        if self.start % itemsize != 0:
            raise ValueError(f"_SlotMeta.start={self.start} is not aligned to dtype={self.dtype} itemsize={itemsize}.")
        # ``end - start`` must cover an integer number of elements; otherwise
        # the slot's byte span doesn't match the typed view's footprint.
        span = self.end - self.start
        if span < 0 or span % itemsize != 0:
            raise ValueError(
                f"_SlotMeta span={span} (end={self.end} - start={self.start}) "
                f"is not a non-negative multiple of dtype={self.dtype} "
                f"itemsize={itemsize}."
            )


def _tensor_byte_view(tensor: torch.Tensor) -> torch.Tensor:
    """Return a flat ``uint8`` view of *tensor* preserving on-disk byte order.

    Mirrors the per-element-size handling in
    :meth:`flextensor.loaders.RawBlockController.combine_tensors`.
    """
    if tensor.numel() == 0:
        return torch.empty(0, dtype=torch.uint8, device=tensor.device)

    if tensor.is_contiguous():
        return tensor.contiguous().view(torch.uint8).flatten()

    if is_dense_layout(tensor):
        nbytes = tensor.numel() * tensor.element_size()
        offset_bytes = tensor.storage_offset() * tensor.element_size()
        byte_tensor = torch.empty(0, dtype=torch.uint8, device=tensor.device)
        byte_tensor.set_(tensor.untyped_storage(), offset_bytes, (nbytes,))  # type: ignore[call-overload]
        return byte_tensor

    # Non-dense with gaps: fall back to a contiguous copy so the bytes are packed.
    return tensor.contiguous().view(torch.uint8).flatten()


def _slot_view(block: torch.Tensor, meta: _SlotMeta) -> torch.Tensor:
    """Build a typed view into ``block`` at ``meta``'s slot.

    Mirrors :meth:`flextensor.loaders.RawBlockController.reconstruct_original_shapes`
    for a single slot, but does not assume the block is on CPU.
    """
    typed = torch.empty(0, dtype=meta.dtype, device=block.device)
    storage_offset_elems = meta.start // typed.element_size()
    typed.set_(block.untyped_storage(), storage_offset_elems, meta.shape, meta.stride)
    return typed


class ProfileBlockController:
    """View-mode profile loader with a rotating block and a shared prefix.

    Layout (offsets are byte offsets into the underlying ``uint8`` block):

    .. code-block:: text

        [ shared prefix          | per-label rotating region                        ]
          0 .. shared_size          shared_size .. shared_size + max_label_size

    Tensors that appear in ``layer_stats`` for two or more labels are placed
    in the shared prefix at canonical offsets and copied **once** at
    construction time. Tensors private to a single label are placed in the
    rotating region at per-label offsets; their bytes are repacked on every
    :meth:`enter`.
    """

    def __init__(
        self,
        layer_stats: list[IterativeLayerStatistics],
        tensors_map: Mapping[int, torch.Tensor],
        device_gpu: torch.device,
        pinned_memory: bool = False,
        pinned_memory_mode: PinnedMemoryMode = "torch",
        gpu_budget_bytes: int | None = None,
    ) -> None:
        self.device_gpu = device_gpu
        self.tensors_map = tensors_map
        # Own pinner/registry so shutdown() can release profile pins without touching the manager's inference registry.
        self._pinner: HostPinner = make_host_pinner(pinned_memory, pinned_memory_mode)

        # Snapshot ``.data`` for every managed tensor. Once the profile model
        # is patched its ``param.data`` points at our GPU view, so ``enter()``
        # (read source) and ``teardown()`` (restore) need a stable handle.
        self._original_data: dict[int, torch.Tensor] = {}
        # Tensor-valued attrs (e.g. ``weight.scale``) the patcher rebinds to
        # views; teardown restores these alongside ``.data``.
        self._original_attrs: dict[int, dict[str, torch.Tensor]] = {}
        self._label_to_tensor_ids: dict[str, list[int]] = {}
        label_count = self._collect_label_tensors(layer_stats, tensors_map)

        self._shared_ids: tuple[int, ...] = tuple(tid for tid, count in label_count.items() if count >= 2)
        shared_id_set = set(self._shared_ids)

        self._shared_layout, self._shared_size = self._layout_shared_prefix(self._shared_ids)
        self._label_layout, self._rotating_size = self._layout_rotating_region(shared_id_set, self._shared_size)
        self._block_size = self._shared_size + self._rotating_size

        self._preflight_gpu_budget(gpu_budget_bytes)
        self._cpu_block, self._gpu_block = self._allocate_blocks(device_gpu)
        self._tensor_id_to_view_map = self._build_view_map(self._gpu_block)
        # CPU mirror lets ``enter()`` copy in the source dtype (no uint8 reinterpret).
        self._cpu_id_to_view_map = self._build_view_map(self._cpu_block)
        self._load_shared_prefix()

        LOGGER.debug(
            "ProfileBlockController: shared=%d B, rotating=%d B, total=%d B, labels=%d, "
            "shared_tensors=%d, per_slot_alignment=%d B",
            self._shared_size,
            self._rotating_size,
            self._block_size,
            len(self._label_layout),
            len(self._shared_ids),
            _DEFAULT_SLOT_ALIGNMENT_BYTES,
        )

    def _preflight_gpu_budget(self, gpu_budget_bytes: int | None) -> None:
        """Fail fast when the view block provably exceeds the GPU budget.

        Only runs when a budget is set (``max_gpu_mem_fraction`` is None means
        latency mode, no constraint).
        """
        if gpu_budget_bytes is None or self._block_size <= gpu_budget_bytes:
            return
        gib = 1 << 30
        raise ValueError(
            f"profile_mode='view' needs {self._block_size / gib:.2f} GiB of GPU memory for its "
            f"profile block (shared={self._shared_size / gib:.2f} GiB + "
            f"rotating={self._rotating_size / gib:.2f} GiB), but only "
            f"{gpu_budget_bytes / gib:.2f} GiB are available under max_gpu_mem_fraction. "
            f"Set profile_mode='getter' for a lower-footprint profile phase, or raise "
            f"max_gpu_mem_fraction."
        )

    def _collect_label_tensors(
        self,
        layer_stats: list[IterativeLayerStatistics],
        tensors_map: Mapping[int, torch.Tensor],
    ) -> dict[int, int]:
        """Walk ``layer_stats``, populate ``_label_to_tensor_ids`` and
        ``_original_data``, return per-tid label-reference counts."""
        label_count: dict[int, int] = {}
        for stat in layer_stats:
            seen: set[int] = set()
            ordered: list[int] = []
            for tid in stat.tensor_ids:
                if tid in seen or tid not in tensors_map:
                    continue
                seen.add(tid)
                ordered.append(tid)
            self._label_to_tensor_ids[stat.label] = ordered
            for tid in ordered:
                label_count[tid] = label_count.get(tid, 0) + 1
                if tid not in self._original_data:
                    src = tensors_map[tid]
                    self._original_data[tid] = src.data
                    attrs = self._snapshot_tensor_attrs(src)
                    if attrs:
                        self._original_attrs[tid] = attrs
        return label_count

    @staticmethod
    def _snapshot_tensor_attrs(tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        """Capture public, tensor-valued instance attributes (e.g. ``weight.scale``)."""
        snapshot: dict[str, torch.Tensor] = {}
        for name in set(dir(tensor)) - set(dir(type(tensor))):
            if name.startswith("_"):
                continue
            value = getattr(tensor, name, None)
            if isinstance(value, torch.Tensor):
                snapshot[name] = value
        return snapshot

    def _layout_shared_prefix(self, shared_ids: tuple[int, ...]) -> tuple[dict[int, _SlotMeta], int]:
        layout: dict[int, _SlotMeta] = {}
        cursor = 0
        for tid in shared_ids:
            meta = self._build_slot_meta(self._original_data[tid], start=cursor)
            layout[tid] = meta
            cursor = meta.end
        return layout, cursor

    def _layout_rotating_region(
        self, shared_id_set: set[int], shared_size: int
    ) -> tuple[dict[str, list[tuple[int, _SlotMeta]]], int]:
        layout: dict[str, list[tuple[int, _SlotMeta]]] = {}
        max_label_bytes = 0
        for label, tids in self._label_to_tensor_ids.items():
            cursor = shared_size
            slots: list[tuple[int, _SlotMeta]] = []
            for tid in tids:
                if tid in shared_id_set:
                    continue
                meta = self._build_slot_meta(self._original_data[tid], start=cursor)
                slots.append((tid, meta))
                cursor = meta.end
            layout[label] = slots
            max_label_bytes = max(max_label_bytes, cursor - shared_size)
        return layout, max_label_bytes

    def _allocate_blocks(self, device_gpu: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self._block_size == 0:
            cpu = torch.empty(0, dtype=torch.uint8, device="cpu")
            gpu = torch.empty(0, dtype=torch.uint8, device=device_gpu)
        else:
            cpu = torch.zeros(self._block_size, dtype=torch.uint8, device="cpu")
            gpu = torch.zeros(self._block_size, dtype=torch.uint8, device=device_gpu)
        cpu = self._pinner.pin(cpu)
        return cpu, gpu

    def _build_view_map(self, block: torch.Tensor) -> dict[int, torch.Tensor]:
        view_map: dict[int, torch.Tensor] = {}
        for tid, meta in self._shared_layout.items():
            view_map[tid] = _slot_view(block, meta)
        for slots in self._label_layout.values():
            for tid, meta in slots:
                view_map[tid] = _slot_view(block, meta)
        return view_map

    def _load_shared_prefix(self) -> None:
        if self._shared_size <= 0:
            return
        for tid in self._shared_layout:
            self._cpu_id_to_view_map[tid].copy_(self._original_data[tid].detach().cpu())
        self._gpu_block[: self._shared_size].copy_(self._cpu_block[: self._shared_size])
        # Ensure prefix is resident before any subsequent GPU work.
        if self._gpu_block.is_cuda:
            torch.cuda.synchronize(self._gpu_block.device)

    @staticmethod
    def _build_slot_meta(
        tensor: torch.Tensor,
        *,
        start: int,
        alignment: int | None = None,
    ) -> _SlotMeta:
        """Compute slot metadata for *tensor*; ``start`` is rounded up to
        ``max(tensor.element_size(), alignment)``.

        Parameters
        ----------
        tensor:
            Source tensor whose layout (``shape``, ``dtype``, ``stride``)
            is recorded into the slot. The tensor itself is not retained.
        start:
            Caller-supplied byte offset into the block. May be sub-aligned;
            this function rounds it up.
        alignment:
            Minimum byte alignment for the returned ``start``. ``None``
            falls back to :data:`_DEFAULT_SLOT_ALIGNMENT_BYTES` (matches
            :attr:`AllocationBlock.memory_alignment`). Callers that
            explicitly want pre-fix behaviour (round to dtype itemsize
            only -- e.g. unit tests pinning the original semantics) can
            pass ``alignment=1``.
        """
        itemsize = tensor.element_size()
        required = max(itemsize, alignment if alignment is not None else _DEFAULT_SLOT_ALIGNMENT_BYTES)
        aligned_start = -(-start // required) * required
        nbytes = tensor.numel() * itemsize
        return _SlotMeta(
            start=aligned_start,
            end=aligned_start + nbytes,
            shape=tensor.shape,
            dtype=tensor.dtype,
            stride=tuple(
                tensor.stride() if tensor.is_contiguous() or is_dense_layout(tensor) else tensor.contiguous().stride()
            ),
        )

    # ------------------------------------------------------------------
    # Loader interface (subset used by TrapProfileView).
    # ------------------------------------------------------------------

    def enter(self, label: str) -> None:
        """Pack ``label``'s non-shared tensors and copy the active region to GPU.

        Unknown labels and labels whose tensors are all in the shared prefix
        are a no-op. Mirrors :meth:`flextensor.loaders.TensorLayerLoader.enter`.
        """
        if label not in self._label_layout:
            LOGGER.debug("ProfileBlockController: unknown label %r (not in layer_stats).", label)
            return
        slots = self._label_layout[label]
        if not slots:
            return

        for tid, _ in slots:
            self._cpu_id_to_view_map[tid].copy_(self._original_data[tid].detach().cpu())

        # Copy only the active suffix (rotating region used by this label) — the
        # shared prefix is already on GPU and must not be overwritten.
        suffix_end = max((meta.end for _, meta in slots), default=self._shared_size)
        if suffix_end > self._shared_size:
            self._gpu_block[self._shared_size : suffix_end].copy_(self._cpu_block[self._shared_size : suffix_end])
        # Block on the GPU block's device until the H2D completes so the trap's
        # kernel timing window starts on a quiescent stream.
        if self._gpu_block.is_cuda:
            torch.cuda.synchronize(self._gpu_block.device)

    def exit(self, label: str) -> None:
        """No-op; rotating-region state is managed by ``enter``."""
        return None

    def release_memory(self) -> None:
        """No-op per-step hook; block release happens in :meth:`shutdown`."""

    def get(self, tensor_id: int) -> torch.Tensor | None:
        """Return the canonical view for ``tensor_id`` if managed by this controller."""
        return self._tensor_id_to_view_map.get(tensor_id)

    def shutdown(self) -> None:
        """Drop CPU/GPU blocks, snapshots, and the view maps (idempotent)."""
        self._tensor_id_to_view_map = {}
        self._cpu_id_to_view_map = {}
        self._original_data = {}
        self._original_attrs = {}
        self._cpu_block = torch.empty(0, dtype=torch.uint8, device="cpu")
        if self._gpu_block.numel() > 0:
            self._gpu_block = torch.empty(0, dtype=torch.uint8, device=self._gpu_block.device)
        else:
            self._gpu_block = torch.empty(0, dtype=torch.uint8, device=self.device_gpu)
        self._pinner.release_all()

    # ------------------------------------------------------------------
    # View-model integration.
    # ------------------------------------------------------------------

    def get_tensor_id_to_view_mapping(self) -> dict[int, torch.Tensor]:
        """Mapping from tensor id to canonical GPU view inside the block."""
        return dict(self._tensor_id_to_view_map)

    def get_gpu_memory_bytes(self) -> int:
        """Total GPU bytes held by this controller's block."""
        return self._gpu_block.untyped_storage().nbytes() if self._gpu_block.numel() > 0 else 0

    # ------------------------------------------------------------------
    # Teardown helpers.
    # ------------------------------------------------------------------

    def teardown(
        self,
        model: torch.nn.Module | dict[str, object] | None,
        tensors_map: Mapping[int, torch.Tensor],
    ) -> None:
        """Unpatch the model, then drop the blocks (always, via ``shutdown()``).

        Restores attrs before ``.data`` so no view lingers on the shared param;
        ``model`` is unpatched by :meth:`_restore_container_entries`. A failed
        restore is blanked best-effort and surfaced via a chained ``RuntimeError``.
        """
        failures: list[tuple[int, Exception]] = []
        try:
            self._restore_container_entries(model, tensors_map)
            for tid, original in self._original_data.items():
                # Fall back to the construction-time map if the passed one diverged.
                live = tensors_map.get(tid)
                if live is None:
                    live = self.tensors_map.get(tid)
                if live is None:
                    # No handle in either map: .data can't be restored (dangling-view leak).
                    failures.append((tid, KeyError(tid)))
                    LOGGER.warning(
                        "ProfileBlockController.teardown: tid=%s missing from both the passed "
                        "and construction-time tensors_map; its .data was not restored.",
                        tid,
                    )
                    continue
                # ``.data`` last so a mid-way failure still ends with no view
                # aliasing the param's ``.data``.
                for name, value in (*self._original_attrs.get(tid, {}).items(), ("data", original)):
                    self._restore_or_blank(live, name, value, tid, failures)
        finally:
            self.shutdown()

        if failures:
            failed_ids = list(dict.fromkeys(tid for tid, _ in failures))
            raise RuntimeError(
                f"ProfileBlockController.teardown: failed to restore "
                f"{len(failed_ids)} tensor(s) (ids={failed_ids}); see logs for "
                f"per-tensor exceptions."
            ) from failures[0][1]

    def _restore_container_entries(
        self,
        model: torch.nn.Module | dict[str, object] | None,
        tensors_map: Mapping[int, torch.Tensor],
    ) -> None:
        """Rewrite ``dict`` entries that alias block views back to their originals.

        Without this a returned profile dict would keep pinning the block past
        ``shutdown()`` (nn.Module params restore in place via ``.data`` instead).
        """
        if not isinstance(model, dict):
            return
        view_to_original: dict[int, torch.Tensor] = {}
        for tid, view in self._tensor_id_to_view_map.items():
            original = tensors_map.get(tid)
            if original is None:
                original = self.tensors_map.get(tid)
            if original is not None:
                view_to_original[id(view)] = original
        for key, value in list(model.items()):
            original = view_to_original.get(id(value))
            if original is not None:
                model[key] = original

    @staticmethod
    def _restore_or_blank(
        live: object,
        name: str,
        value: torch.Tensor,
        tid: int,
        failures: list[tuple[int, Exception]],
    ) -> None:
        """Rebind ``live.<name>`` to ``value``; on failure, blank it so no
        profile-block view survives. Best-effort: a hostile setter may reject
        the blank too."""
        label = ".data" if name == "data" else f"attr {name!r}"
        try:
            setattr(live, name, value)
            return
        except Exception as exc:
            LOGGER.exception("ProfileBlockController.teardown: failed to restore %s for tid=%s", label, tid)
            failures.append((tid, exc))
        try:
            setattr(live, name, torch.empty(0, dtype=value.dtype, device="cpu"))
        except Exception:
            LOGGER.exception("ProfileBlockController.teardown: failed to blank %s for tid=%s", label, tid)

    # ------------------------------------------------------------------
    # Diagnostics.
    # ------------------------------------------------------------------

    @property
    def shared_size(self) -> int:
        """Bytes occupied by the shared prefix (loaded once)."""
        return self._shared_size

    @property
    def rotating_size(self) -> int:
        """Bytes of the per-label rotating region (max over labels)."""
        return self._rotating_size

    @property
    def block_size(self) -> int:
        """Total bytes of the underlying CPU/GPU blocks (``shared + rotating``)."""
        return self._block_size

    def shared_tensor_ids(self) -> Iterable[int]:
        """Tensor ids placed in the shared prefix (referenced by ≥ 2 labels)."""
        return iter(self._shared_ids)
