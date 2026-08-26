# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integrated E2E-optimized LTX-2.3 outpaint pipeline (torch eager), combining the
per-stage speedups measured separately, into ONE run:

  - DiT   : CP (Ulysses) x TP sharding  (cp_tp_e2e.py install(); WORLD>1 only)
  - VAE encode (stage-1 reference) : temporal tiling + tile-sharding across ranks
             (round-robin tiles + all-reduce of the weighted accumulation buffers)
  - VAE decode : tile-sharding across ranks (round-robin + all-reduce group buffers)
  - Gemma : compute-once-on-rank0 + broadcast (video_ctx, audio_ctx, mask)

Same script runs the single-device baseline (WORLD==1: DiT vanilla; VAE/Gemma
patches degrade to the exact non-sharded path) and the sharded integrated run
(WORLD>1). This makes baseline-vs-integrated directly comparable.

Run baseline : TP=1 NF=49 ITERS=1 torchrun --nproc_per_node=1 integrated_e2e.py
Run CP2xTP2  : TP=2 NF=49 ITERS=2 torchrun --nproc_per_node=4 integrated_e2e.py
PSNR check   : python repro/scripts/psnr.py A.mp4 B.mp4
"""

import functools
import glob
import os
import sys
import time

sys.path.insert(0, os.environ.get("EX_DIR", "/workspace/flex-tensor/examples/ltx23/outpaint"))
from dataclasses import replace
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

# ---- run-mode flags ----
# TRT_DIT=1 : run BOTH DiT stages on per-rank single native TRT engines (needs trt11venv +
#             patched Myelin/libnvinfer on LD_LIBRARY_PATH). 0 = torch-eager DiT (default).
# ENC_SHARD / DECODE_SHARD / GEMMA_BCAST : pipeline optimizations (default on); turn OFF for the
#             B0 "sharded-DiT, pipeline-unoptimized" baseline.
TRT_DIT = os.environ.get("TRT_DIT", "0") == "1"
ENC_SHARD = os.environ.get("ENC_SHARD", "1") == "1"
GEMMA_BCAST = os.environ.get("GEMMA_BCAST", "1") == "1"
NF = int(os.environ.get("NF", "49"))
TP = int(os.environ.get("TP", "2"))
COND = os.environ.get("COND", "/workspace/inputs/real_input_2s.mp4")
OUTDIR = os.environ.get("OUTDIR", "/workspace/out")
SHARD_DIR = os.environ.get("SHARD_DIR", "/workspace/out/shard721")
os.makedirs(OUTDIR, exist_ok=True)

if TRT_DIT:
    # cuDNN in trt11venv nightly is version-mismatched -> force non-cuDNN SDPA for the eager
    # stages (Gemma / VAE). (verbatim from cp4tp2_single_render.py)
    try:
        torch.backends.cudnn.enabled = False
    except Exception:
        pass
    try:
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass
    try:
        from torch.nn.attention import SDPBackend as _SDPB
        from torch.nn.attention import sdpa_kernel as _sdpa_kernel

        _NONCUDNN = [_SDPB.FLASH_ATTENTION, _SDPB.EFFICIENT_ATTENTION, _SDPB.MATH]
        _orig_sdpa = F.scaled_dot_product_attention

        def _sdpa_noncudnn(*a, **k):
            with _sdpa_kernel(_NONCUDNN):
                return _orig_sdpa(*a, **k)

        F.scaled_dot_product_attention = _sdpa_noncudnn
        torch.nn.functional.scaled_dot_product_attention = _sdpa_noncudnn
    except Exception:
        pass
    import torch_tensorrt
    import torch_tensorrt.runtime as ttrt_rt
    from torch_tensorrt.distributed import setup_nccl_for_torch_tensorrt
    from torch_tensorrt.distributed._nccl_utils import initialize_nccl_comm

    setup_nccl_for_torch_tensorrt()
    # native single-engine collectives: fp32 Myelin boundary + all_gather->all_reduce, each routed
    # over its CP/TP subgroup communicator (patched Myelin/libnvinfer).
    os.environ["TRT_COLL_BOUNDARY"] = "castb"
    os.environ["AG_VIA_ALLREDUCE"] = "1"

dist.init_process_group("nccl", timeout=timedelta(seconds=int(os.environ.get("PG_TIMEOUT", "180"))))
LR = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(LR)
if TRT_DIT:
    initialize_nccl_comm()
RANK = dist.get_rank()
WORLD = dist.get_world_size()
CP = WORLD // TP
tp_rank = RANK % TP
cp_rank = RANK // TP


def log(*a):
    if RANK == 0:
        print(*a, flush=True)


def memlog(tag):
    torch.cuda.synchronize()
    a = torch.cuda.memory_allocated() / 1e9
    r = torch.cuda.memory_reserved() / 1e9
    print(f"[mem] rank={RANK} {tag} alloc={a:.1f}GB reserved={r:.1f}GB", flush=True)


# ---- CP/TP subgroups (create all groups on all ranks, collectively) ----
tp_group = cp_group = None
if WORLD > 1:
    for c in range(CP):
        g = dist.new_group([c * TP + t for t in range(TP)])
        if c == cp_rank:
            tp_group = g
    for t in range(TP):
        g = dist.new_group([c * TP + t for c in range(CP)])
        if t == tp_rank:
            cp_group = g
if TRT_DIT:
    # tell the patched TRT collective converters this rank's CP/TP subgroup global ranks (2-D mesh)
    os.environ["TRT_CP_RANKS"] = ",".join(str(c * TP + tp_rank) for c in range(CP))
    os.environ["TRT_TP_RANKS"] = ",".join(str(cp_rank * TP + t) for t in range(TP))
log(f"WORLD={WORLD} TP={TP} CP={CP} rank={RANK} tp_rank={tp_rank} cp_rank={cp_rank} TRT_DIT={TRT_DIT}")


# ---- stage-timer: local elapsed -> MAX over WORLD (parallel wall) ----
def timed(tag, store):
    class _T:
        def __enter__(self):
            torch.cuda.synchronize()
            if WORLD > 1:
                dist.barrier()
            self.t0 = time.time()
            return self

        def __exit__(self, *a):
            torch.cuda.synchronize()
            dt = torch.tensor([time.time() - self.t0], device="cuda")
            if WORLD > 1:
                dist.all_reduce(dt, op=dist.ReduceOp.MAX)
            store[tag] = store.get(tag, 0.0) + dt.item()

    return _T()


STAGE = {}


# =========================================================================== #
# DiT  CP x TP  (verbatim from repro/scripts/cp_tp_e2e.py install())
# =========================================================================== #
def col_shard(lin):
    o = lin.weight.shape[0]
    s = o // TP
    sl = slice(tp_rank * s, (tp_rank + 1) * s)
    lin.weight = nn.Parameter(lin.weight.data[sl].clone())
    if lin.bias is not None:
        lin.bias = nn.Parameter(lin.bias.data[sl].clone())


def row_shard(lin, tp_group):
    i = lin.weight.shape[1]
    s = i // TP
    sl = slice(tp_rank * s, (tp_rank + 1) * s)
    lin.weight = nn.Parameter(lin.weight.data[:, sl].clone())
    if lin.bias is not None:
        lin.bias = nn.Parameter(lin.bias.data.clone()) if tp_rank == 0 else None
    lin.register_forward_hook(lambda m, inp, out: dist.all_reduce(out, group=tp_group))


class DistRMSNorm(nn.Module):
    def __init__(s, w_shard, full_dim, eps, tp_group):
        super().__init__()
        s.weight = nn.Parameter(w_shard.clone())
        s.full = full_dim
        s.eps = eps
        s.g = tp_group

    def forward(s, x):
        ss = (x.float() ** 2).sum(-1, keepdim=True)
        dist.all_reduce(ss, group=s.g)
        return (x.float() * torch.rsqrt(ss / s.full + s.eps)).to(x.dtype) * s.weight


def tp_shard_attn(attn, tp_group):
    inner = attn.heads * attn.dim_head
    for name in ["to_q", "to_k", "to_v"]:
        col_shard(getattr(attn, name))
    row_shard(attn.to_out[0], tp_group)
    if getattr(attn, "to_gate_logits", None) is not None:
        col_shard(attn.to_gate_logits)
    eps = attn.q_norm.eps if hasattr(attn.q_norm, "eps") else 1e-6
    s = inner // TP
    sl = slice(tp_rank * s, (tp_rank + 1) * s)
    attn.q_norm = DistRMSNorm(attn.q_norm.weight.data[sl], inner, eps, tp_group)
    attn.k_norm = DistRMSNorm(attn.k_norm.weight.data[sl], inner, eps, tp_group)
    attn.heads = attn.heads // TP


def tp_shard_ff(ff, tp_group):
    col_shard(ff.net[0].proj)
    row_shard(ff.net[2], tp_group)


class UlyssesAttn:
    def __init__(s, grp, w):
        s.grp = grp
        s.w = w

    def _a2a(s, x):
        o = torch.empty_like(x)
        dist.all_to_all_single(o, x, group=s.grp)
        return o

    def __call__(s, q, k, v, heads, mask=None):
        B, Tloc, inner = q.shape
        D = inner // heads
        sp = s.w
        hl = heads // sp

        def tf(t):
            a = t.view(B, Tloc, sp, hl, D).permute(2, 0, 1, 3, 4).contiguous()
            a = s._a2a(a)
            return a.permute(1, 0, 2, 3, 4).reshape(B, sp * Tloc, hl, D).transpose(1, 2)

        o = F.scaled_dot_product_attention(tf(q), tf(k), tf(v))
        o = o.transpose(1, 2).reshape(B, sp, Tloc, hl, D).permute(1, 0, 2, 3, 4).contiguous()
        return s._a2a(o).permute(1, 2, 0, 3, 4).reshape(B, Tloc, heads * D)


class GatherKVAttn:
    def __init__(s, grp, w):
        s.grp = grp
        s.w = w

    def _g(s, x):
        xs = [torch.empty_like(x) for _ in range(s.w)]
        dist.all_gather(xs, x.contiguous(), group=s.grp)
        return torch.cat(xs, 1)

    def __call__(s, q, k, v, heads, mask=None):
        k = s._g(k)
        v = s._g(v)
        B, Tq, inner = q.shape
        D = inner // heads
        qh = q.view(B, Tq, heads, D).transpose(1, 2)
        kh = k.view(B, -1, heads, D).transpose(1, 2)
        vh = v.view(B, -1, heads, D).transpose(1, 2)
        return F.scaled_dot_product_attention(qh, kh, vh).transpose(1, 2).reshape(B, Tq, heads * D)


Hh = 32
hs = Hh // TP
hsl = slice(tp_rank * hs, (tp_rank + 1) * hs)


def sh_seq(t, d):
    return t.narrow(d, cp_rank * (t.shape[d] // CP), t.shape[d] // CP).contiguous() if t is not None else None


def pe_v(pe):
    return (
        None
        if pe is None
        else tuple(p[:, hsl].narrow(2, cp_rank * (p.shape[2] // CP), p.shape[2] // CP).contiguous() for p in pe)
    )


def pe_a(pe):
    return None if pe is None else tuple(p[:, hsl].contiguous() for p in pe)


SHARD_ADALN = os.environ.get("SHARD_ADALN", "0") == "1"
SHARD_OUTPUT = os.environ.get("SHARD_OUTPUT", "0") == "1"

# COMPILE_DIT: off | inductor | trt
#   inductor -> torch.compile(backend="inductor") per DiT block
#   trt      -> torch.compile(backend="torch_tensorrt") per DiT block. NOTE this is the MULTI-engine
#               Torch-TensorRT path: the CP/TP collectives inside each block are graph breaks unless
#               the installed torch-tensorrt has native TRT collectives, so a block lowers to several
#               engines rather than one. A true SINGLE native engine for a CP x TP mesh additionally
#               needs the patched TensorRT+Myelin stack (per-subgroup NCCL communicators); a CP-only
#               mesh is a single world group and can single-engine on a stock native-collectives
#               build. Compiling per block, not per stage, keeps the collectives at block boundaries.
COMPILE_DIT = os.environ.get("COMPILE_DIT", "off").lower()


def _compile_dit(model):
    if COMPILE_DIT in ("off", "", "none"):
        return
    backend = {"inductor": "inductor", "trt": "torch_tensorrt", "torch_tensorrt": "torch_tensorrt"}.get(COMPILE_DIT)
    if backend is None:
        raise SystemExit(f"COMPILE_DIT must be off|inductor|trt; got {COMPILE_DIT!r}")
    vm = getattr(model, "velocity_model", model)
    opts = {}
    if backend == "torch_tensorrt":
        import torch_tensorrt  # noqa: F401  (registers the backend)

        # TRT sizes its build workspace from the block's activation shapes. Uncapped, a DiT block at
        # 361 frames asks for ~49 GiB on top of the resident weights and OOMs even on an 80 GB card,
        # so the cap is not tuning -- without it the build does not complete at all.
        _ws_mb = int(os.environ.get("TRT_WORKSPACE_MB", "8192"))
        opts = dict(
            options={
                "min_block_size": int(os.environ.get("TRT_MIN_BLOCK", "1")),
                "use_python_runtime": True,
                "use_distributed_mode_trace": True,
                "immutable_weights": True,
                "truncate_double": True,
                "workspace_size": _ws_mb * 1024 * 1024,
            }
        )
    vm.transformer_blocks = torch.nn.ModuleList(
        torch.compile(b, backend=backend, dynamic=False, **opts) for b in vm.transformer_blocks
    )
    log(f"[compile_dit] backend={backend} blocks={len(vm.transformer_blocks)} (compile happens on first call)")


def _time_dit(model):
    """Accumulate per-stage DiT wall time so the report can separate DiT from the rest of the pipeline."""
    vm = getattr(model, "velocity_model", model)
    if getattr(vm, "_dit_timed", False):
        return
    _fwd = vm.forward
    # The pipeline rebuilds both DiT stages on every generate, so this runs once per stage per
    # iteration. Key by position within the generate (1=base, 2=upscaler) rather than by build
    # count, otherwise the keys drift (dit_s5/dit_s6 on the third iteration) and are not comparable
    # across configurations. STAGE is cleared per iteration, so each key holds one generate.
    _idx = len(STAGE_VMS_T) % 2 + 1
    STAGE_VMS_T.append(vm)

    def timed(*a, **k):
        torch.cuda.synchronize()
        _t = time.time()
        r = _fwd(*a, **k)
        torch.cuda.synchronize()
        STAGE[f"dit_s{_idx}"] = STAGE.get(f"dit_s{_idx}", 0.0) + (time.time() - _t)
        return r

    vm.forward = timed
    vm._dit_timed = True


STAGE_VMS_T = []

# FLAT_MOVE: replace nn.Module.to() ping-ponging for the CPU-resident DiT.
#
# `m.to("cuda")` walks every parameter and buffer and issues one copy each. A 22B DiT shard is
# thousands of small tensors in *pageable* host memory, which measures ~1.2 GB/s -- 44 GB of traffic
# per generate, ~38 s, larger than all DiT compute combined. Two things are wrong with it:
#   1. thousands of small pageable copies instead of a few large pinned ones;
#   2. it copies the weights *back* to the host afterwards, which is pointless -- inference never
#      modifies them, so the host copy is already authoritative.
# This packs each stage's tensors into one pinned host buffer per dtype and moves them with a single
# copy per dtype, host->device only. Parking back is a pointer swap, no traffic at all.
FLAT_MOVE = os.environ.get("FLAT_MOVE", "0") == "1"


class FlatPark:
    """Park a module's tensors in pinned host memory; move to device as one copy per dtype."""

    _dev_pool = {}  # dtype -> device buffer, shared across stages (only one is ever resident)

    def __init__(self, module):
        self.groups = {}  # dtype -> {"cpu": pinned flat tensor, "items": [(owner, name, off, shape)]}
        by_dtype = {}
        for owner in module.modules():
            for name, t in list(owner.named_parameters(recurse=False)) + list(owner.named_buffers(recurse=False)):
                if t is None:
                    continue
                by_dtype.setdefault(t.dtype, []).append((owner, name, t))
        for dt, items in by_dtype.items():
            total = sum(t.numel() for _, _, t in items)
            cpu_flat = torch.empty(total, dtype=dt, pin_memory=True)
            recs, off = [], 0
            for owner, name, t in items:
                n = t.numel()
                cpu_flat[off : off + n].copy_(t.detach().reshape(-1))
                recs.append((owner, name, off, tuple(t.shape)))
                off += n
            self.groups[dt] = {"cpu": cpu_flat, "items": recs}
        self._point_at("cpu")
        _gc.collect()
        torch.cuda.empty_cache()

    def _point_at(self, where):
        for dt, g in self.groups.items():
            base = g["cpu"] if where == "cpu" else FlatPark._dev_pool[dt]
            for owner, name, off, shape in g["items"]:
                n = 1
                for s in shape:
                    n *= s
                view = base[off : off + n].view(shape)
                tgt = owner._parameters.get(name)
                if tgt is not None:
                    tgt.data = view
                else:
                    owner._buffers[name] = view

    def to_device(self):
        for dt, g in self.groups.items():
            need = g["cpu"].numel()
            buf = FlatPark._dev_pool.get(dt)
            if buf is None or buf.numel() < need:
                FlatPark._dev_pool[dt] = torch.empty(need, dtype=dt, device="cuda")
                buf = FlatPark._dev_pool[dt]
            buf[:need].copy_(g["cpu"], non_blocking=True)
        torch.cuda.synchronize()
        self._point_at("cuda")

    def to_host(self):
        # No copy: inference does not modify weights, so the pinned host buffer is still correct.
        self._point_at("cpu")
        # Drop the reference so the device buffer is not pinned across the other stage's turn --
        # otherwise both stages' flat buffers coexist and peak grows by a full stage (~19 GiB), which
        # does not fit a 40 GB card. Deliberately NOT empty_cache(): the caching allocator keeps the
        # block and hands the identical-sized buffer to the next stage, so this costs no realloc.
        FlatPark._dev_pool.clear()


# The AdaLN and output-gather patches are the production implementations from
# ``context_parallel.py``, not copies. This driver used to carry its own, and they had already
# drifted: the production video-AdaLN guard is on the *per-sample* token count, while the copy here
# guarded on the flattened batch*token total. Those coincide only at batch 1 -- which is all this
# benchmark runs -- so parity passed while the benchmark was exercising the older logic and could
# not have caught a regression in the code actually being merged. Importing them means the measured
# numbers and the parity gates apply to the shipped path.
from context_parallel import ContextParallelRuntime
from context_parallel import _install_process_output_patch as _cp_install_process_output_patch
from context_parallel import _install_sharded_adaln as _cp_install_sharded_adaln


@functools.lru_cache(maxsize=1)
def _cp_runtime():
    """Describe this rank's context-parallel group the way ``context_parallel.py`` expects.

    One shared instance: both patches close over it, and the runtime carries the poisoned flag they
    consult before entering a collective, so handing them separate objects would let that state
    diverge.
    """
    return ContextParallelRuntime(size=CP, rank=cp_rank, local_rank=LR, group=cp_group)


def install(vm):
    from ltx_core.model.transformer.model import LTXModel

    if SHARD_ADALN and CP > 1:
        _cp_install_sharded_adaln(vm, _cp_runtime())
    for blk in vm.transformer_blocks:
        for an in ["attn1", "attn2", "audio_to_video_attn", "video_to_audio_attn"]:
            at = getattr(blk, an, None)
            if at is not None:
                tp_shard_attn(at, tp_group)
        tp_shard_ff(blk.ff, tp_group)
        blk.attn1.attention_function = UlyssesAttn(cp_group, CP)
        if getattr(blk, "video_to_audio_attn", None) is not None:
            blk.video_to_audio_attn.attention_function = GatherKVAttn(cp_group, CP)
    if getattr(LTXModel, "_cptp", False):
        return
    orig = LTXModel._process_transformer_blocks

    def patched(self, video, audio, perturbations):
        saved = None
        if video is not None:
            saved = video.embedded_timestep

            def sh(t, d):
                return sh_seq(t, d)

            video = replace(
                video,
                x=sh(video.x, 1),
                # Shard only if still at full length: with SHARD_ADALN the wide timestep projection
                # is already CP-sharded in the preprocessor, and sharding twice would be wrong.
                timesteps=(sh(video.timesteps, 1) if video.timesteps.shape[1] == video.x.shape[1] else video.timesteps),
                embedded_timestep=sh(video.embedded_timestep, 1),
                positional_embeddings=pe_v(video.positional_embeddings),
                cross_positional_embeddings=pe_v(video.cross_positional_embeddings),
                # Same "already sharded?" guard as timesteps above (SHARD_ADALN pre-shards these).
                cross_scale_shift_timestep=(
                    sh(video.cross_scale_shift_timestep, 1)
                    if video.cross_scale_shift_timestep is not None
                    and video.cross_scale_shift_timestep.shape[1] == video.x.shape[1]
                    else video.cross_scale_shift_timestep
                ),
                cross_gate_timestep=(
                    sh(video.cross_gate_timestep, 1)
                    if video.cross_gate_timestep is not None and video.cross_gate_timestep.shape[1] == video.x.shape[1]
                    else video.cross_gate_timestep
                ),
            )
        if audio is not None:
            audio = replace(audio, cross_positional_embeddings=pe_a(audio.cross_positional_embeddings))
        vo, ao = orig(self, video, audio, perturbations)
        if vo is not None:
            if SHARD_OUTPUT:
                # Leave x AND embedded_timestep CP-sharded: _process_output is per-token, so it can
                # run on the shard and we gather after proj_out instead (context_parallel.py's
                # _install_process_output_patch, installed below).
                pass
            else:
                xs = [torch.empty_like(vo.x) for _ in range(CP)]
                dist.all_gather(xs, vo.x.contiguous(), group=cp_group)
                vo = replace(vo, x=torch.cat(xs, 1), embedded_timestep=saved)
        return vo, ao

    LTXModel._process_transformer_blocks = patched
    LTXModel._cptp = True
    if SHARD_OUTPUT and CP > 1:
        _cp_install_process_output_patch(_cp_runtime())


# =========================================================================== #
# VAE encode + decode  tile-sharding  (patch the VAE model classes)
# =========================================================================== #
from einops import rearrange
from ltx_core.model.video_vae import video_vae as _vv
from ltx_core.model.video_vae.video_vae import prepare_tiles_for_encoding
from ltx_core.types import VIDEO_SCALE_FACTORS, VideoLatentShape

_orig_tiled_encode = _vv.VideoEncoder.tiled_encode


def _sharded_tiled_encode(self, video, tiling_config=None):
    """Round-robin tiles across ranks + all-reduce(SUM) of weighted accumulation
    buffers. WORLD==1 -> exactly the original tiled_encode (all tiles, no reduce)."""
    dev = next(self.parameters()).device
    dt = next(self.parameters()).dtype
    b, _, frames, hh, ww = video.shape
    if os.environ.get("INSTRUMENT") == "1" and RANK == 0:
        print(f"[encdiag] tiled_encode called video={tuple(video.shape)}", flush=True)
    if (frames - 1) % VIDEO_SCALE_FACTORS.time != 0:
        fc = (frames - 1) % VIDEO_SCALE_FACTORS.time
        video = video[:, :, :-fc, ...]
        frames = video.shape[2]
    latent_shape = VideoLatentShape(
        batch=b,
        channels=self.latent_channels,
        frames=(frames - 1) // VIDEO_SCALE_FACTORS.time + 1,
        height=hh // VIDEO_SCALE_FACTORS.height,
        width=ww // VIDEO_SCALE_FACTORS.width,
    )
    tiles = prepare_tiles_for_encoding(video, tiling_config)
    lat = torch.zeros(latent_shape.to_torch_shape(), device=dev, dtype=dt)
    wts = torch.zeros_like(lat)
    _shard_enc = WORLD > 1 and ENC_SHARD
    with timed("vae_encode", STAGE):
        for i, tile in enumerate(tiles):
            if _shard_enc and (i % WORLD) != RANK:
                continue
            vt = video[tile.in_coords]
            if vt.device != dev or vt.dtype != dt:
                vt = vt.to(device=dev, dtype=dt)
            lt = self.forward(vt)
            mask = tile.blend_mask.to(device=dev, dtype=dt)
            lat[tile.out_coords] += lt * mask
            wts[tile.out_coords] += mask
            del lt, mask, vt
        if _shard_enc:
            dist.all_reduce(lat, op=dist.ReduceOp.SUM)
            dist.all_reduce(wts, op=dist.ReduceOp.SUM)
    return lat / wts.clamp(min=1e-8)


_vv.VideoEncoder.tiled_encode = _sharded_tiled_encode


# DECODE_SHARD=1 (default): round-robin tiles across ranks + all-reduce group buffers.
# DECODE_SHARD=0: every rank decodes every tile locally (stock tiled decode, NO cross-rank
#   collectives) -> fully symmetric, cannot desync. Correct (identical full output on all ranks),
#   but not sped up. Used to get a robust 30s E2E while the sharded-decode 30s desync is diagnosed.
DECODE_SHARD = os.environ.get("DECODE_SHARD", "1") == "1"
# DECODE_BALANCE=1 (default): cost-balanced (LPT) tile->rank assignment instead of round-robin.
# The decode wall = max over ranks, so balancing the per-rank sum of tile costs (H*W*T of each
# decoded tile) minimizes the wall. Deterministic (all ranks compute the same owners) -> the
# bit-exact all-reduce accumulation is unchanged. DECODE_BALANCE=0 -> legacy round-robin.
DECODE_BALANCE = os.environ.get("DECODE_BALANCE", "1") == "1"


def _lpt_owners(costs, world):
    """Longest-processing-time greedy bin-packing: assign each tile (desc cost) to the least-loaded
    rank. Returns owners[i] in [0,world). Minimizes the max per-rank load (the parallel wall)."""
    load = [0.0] * world
    owners = [0] * len(costs)
    for idx in sorted(range(len(costs)), key=lambda i: -costs[i]):
        r = min(range(world), key=lambda r: load[r])
        owners[idx] = r
        load[r] += costs[idx]
    return owners


def _tile_cost(tile):
    """Decode cost proxy = temporal*height*width of the tile's INPUT latent region."""
    c = tile.in_coords
    n = 1
    for d in (2, 3, 4):
        s = c[d]
        if isinstance(s, slice):
            n *= (s.stop - s.start) if (s.stop is not None and s.start is not None) else 1
    return float(n)


def _sharded_decode_video(self, latent, tiling_config=None, generator=None):
    """Sharded tiled decode (round-robin tiles + all-reduce group buffers), yielding
    RGB chunks [f,h,w,c] identical to VideoDecoder.decode_video. WORLD==1 -> original."""
    SHARD_DEC = WORLD > 1 and DECODE_SHARD

    def to_rgb(frames):
        v = rearrange(frames[0], "c f h w -> f h w c")
        return v.add_(1.0).mul_(0.5).clamp_(0.0, 1.0)

    if tiling_config is None:
        yield to_rgb(self(latent, generator=generator))
        return
    # Decode-tile override (env): default tiles (spatial 768px / temporal 80 frames) make ONE
    # tile-forward need ~75GB -> OOMs at 30s (edge of 80GB, tips over on rank0 due to Gemma-fwd
    # fragmentation). Shrink the decode tile (esp. temporal frames) to fit. Blending is seamless
    # for any tile size, so output is unchanged. Applies to decode ONLY (encode keeps default).
    from ltx_core.model.video_vae.tiling import SpatialTilingConfig as _STC
    from ltx_core.model.video_vae.tiling import TemporalTilingConfig as _TTC

    _px = int(os.environ.get("DEC_TILE_PX", "0"))
    _fr = int(os.environ.get("DEC_TILE_FR", "0"))
    if _px or _fr:
        _sc = tiling_config.spatial_config
        _tc = tiling_config.temporal_config
        if _px and _sc is not None:
            _sc = _STC(tile_size_in_pixels=_px, tile_overlap_in_pixels=min(_sc.tile_overlap_in_pixels, _px - 32))
        if _fr and _tc is not None:
            _tc = _TTC(tile_size_in_frames=_fr, tile_overlap_in_frames=min(_tc.tile_overlap_in_frames, _fr - 8))
        tiling_config = type(tiling_config)(spatial_config=_sc, temporal_config=_tc)
    full_video_shape = VideoLatentShape.from_torch_shape(latent.shape).upscale(self.video_downscale_factors)
    tiles = self._prepare_tiles(latent, tiling_config)
    tgroups = self._group_tiles_by_temporal_slice(tiles)
    print(
        f"[decode-diag] rank={RANK} latent={tuple(latent.shape)} ntiles={len(tiles)} "
        f"ntgroups={len(tgroups)} grp_sizes={[len(g) for g in tgroups]} shard={SHARD_DEC}",
        flush=True,
    )
    prev_chunk = prev_w = prev_slice = None
    for _gi, tg in enumerate(tgroups):
        if RANK == 0 and _gi % 4 == 0:
            print(
                f"[dec-mem] rank=0 grp={_gi}/{len(tgroups)} alloc={torch.cuda.memory_allocated() / 1e9:.1f}GB",
                flush=True,
            )
        curr = tg[0].out_coords[2]
        buf_shape = full_video_shape._replace(frames=curr.stop - curr.start)
        buffer = torch.zeros(buf_shape.to_torch_shape(), device=latent.device, dtype=latent.dtype)
        weights = torch.zeros_like(buffer)
        tslice = curr
        owners = (
            _lpt_owners([_tile_cost(t) for t in tg], WORLD)
            if (SHARD_DEC and DECODE_BALANCE)
            else [i % WORLD for i in range(len(tg))]
        )
        for i, tile in enumerate(tg):
            if SHARD_DEC and owners[i] != RANK:
                continue
            decoded = self.forward(latent[tile.in_coords], None, generator)
            mask = tile.blend_mask.to(device=buffer.device, dtype=buffer.dtype)
            toff = tile.out_coords[2].start - tslice.start
            exp_len = tile.out_coords[2].stop - tile.out_coords[2].start
            act_len = min(exp_len, decoded.shape[2], buffer.shape[2] - toff)
            cc = (slice(None), slice(None), slice(toff, toff + act_len), tile.out_coords[3], tile.out_coords[4])
            ms = mask[:, :, :act_len, :, :] if mask.shape[2] > 1 else mask
            buffer[cc] += decoded[:, :, :act_len, :, :] * ms
            weights[cc] += ms
        if SHARD_DEC:
            dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
            dist.all_reduce(weights, op=dist.ReduceOp.SUM)
        if prev_chunk is not None:
            if prev_slice.stop > curr.start:
                ov = prev_slice.stop - curr.start
                tos = slice(curr.start - prev_slice.start, None)
                prev_chunk[:, :, tos, :, :] += buffer[:, :, slice(0, ov), :, :]
                prev_w[:, :, tos, :, :] += weights[:, :, slice(0, ov), :, :]
                buffer[:, :, slice(0, ov), :, :] = prev_chunk[:, :, tos, :, :]
                weights[:, :, slice(0, ov), :, :] = prev_w[:, :, tos, :, :]
            prev_w = prev_w.clamp(min=1e-8)
            yl = curr.start - prev_slice.start
            yield to_rgb((prev_chunk / prev_w)[:, :, :yl, :, :])
        prev_chunk, prev_w, prev_slice = buffer, weights, curr
    if prev_chunk is not None:
        prev_w = prev_w.clamp(min=1e-8)
        yield to_rgb(prev_chunk / prev_w)


_vv.VideoDecoder.decode_video = _sharded_decode_video


# =========================================================================== #
# Gemma compute-once-on-rank0 + broadcast
# =========================================================================== #
from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessorOutput
from ltx_pipelines.utils.blocks import PromptEncoder

_DT = {
    0: torch.bfloat16,
    1: torch.float16,
    2: torch.float32,
    3: torch.int64,
    4: torch.bool,
    5: torch.int32,
    6: torch.float64,
}
_DTI = {v: k for k, v in _DT.items()}
_orig_pe_call = PromptEncoder.__call__


def _bcast_tensor(t):
    """Broadcast one tensor from rank0 (shape+dtype negotiated). All ranks call identically."""
    meta = torch.zeros(8, dtype=torch.long, device="cuda")
    if RANK == 0:
        meta[0] = t.dim()
        meta[1] = _DTI[t.dtype]
        for i, s in enumerate(t.shape):
            meta[2 + i] = s
    dist.broadcast(meta, 0)
    nd = int(meta[0])
    dtype = _DT[int(meta[1])]
    shape = [int(meta[2 + i]) for i in range(nd)]
    if RANK != 0:
        t = torch.empty(shape, dtype=dtype, device="cuda")
    dist.broadcast(t, 0)
    return t


def _pe_broadcast(self, prompts, **kw):
    if WORLD == 1 or not GEMMA_BCAST:
        # B0 / single-device: every rank computes Gemma itself (no load-once-broadcast).
        with timed("gemma", STAGE):
            return _orig_pe_call(self, prompts, **kw)
    with timed("gemma", STAGE):
        if RANK == 0:
            out = _orig_pe_call(self, prompts, **kw)[0]
            v, a, m = out.video_encoding, out.audio_encoding, out.attention_mask
            has_a = torch.tensor([1 if a is not None else 0], dtype=torch.long, device="cuda")
        else:
            v = m = None
            a = None
            has_a = torch.zeros(1, dtype=torch.long, device="cuda")
        if WORLD == 1:
            return [EmbeddingsProcessorOutput(v, a, m)]
        dist.broadcast(has_a, 0)
        v = _bcast_tensor(v)
        if int(has_a[0]) == 1:
            a = _bcast_tensor(a)
        else:
            a = None
        m = _bcast_tensor(m)
    return [EmbeddingsProcessorOutput(v, a, m)]


PromptEncoder.__call__ = _pe_broadcast


# =========================================================================== #
# Build pipeline, shard DiT, generate
# =========================================================================== #
H = os.environ.get("HF_HUB", "/workspace/hf/hub")
D = glob.glob(f"{H}/models--Lightricks--LTX-2.3/snapshots/*/ltx-2.3-22b-distilled-1.1.safetensors")[0]
U = glob.glob(f"{H}/models--Lightricks--LTX-2.3/snapshots/*/ltx-2.3-spatial-upscaler-x2-1.1.safetensors")[0]
L = glob.glob(
    f"{H}/models--oumoumad--LTX-2.3-22b-IC-LoRA-Outpaint/snapshots/*/ltx-2.3-22b-ic-lora-outpaint.safetensors"
)[0]
G = glob.glob(f"{H}/models--google--gemma-3-12b-it-qat-q4_0-unquantized/snapshots/*")[0]
import serve_infer as si
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.ic_lora import ICLoraPipeline
from ltx_pipelines.utils.blocks import DiffusionStage
from ltx_pipelines.utils.media_io import encode_video

si.install_stage2_video_conditioning_patch()
si.install_stage2_lora_patch()

# stage-1 reference encode hardcodes tiling_config=None -> untiled full-clip VAE encode OOMs at
# long clips. Default it to a temporal tiling config so it routes through (now sharded) tiled_encode.
#
# CACHE_COND=1 (TASK #13a): the image_conditioner is called TWICE per generate (stage-1 + stage-2) and
# EACH call rebuilds the VAE encoder (~35 s CPU-bound safetensors parse, NOT disk — warm iters still pay
# it) and stage-1 re-decodes+re-encodes the 720-frame reference mp4. Warm decomposition: image_conditioner
# x2 = 80.5 s of a 220 s wall, ~91 % pure build/decode overhead. Fix (output-preserving / bit-identical):
#   (1) build the VAE encoder ONCE, keep it resident, reuse across both calls + all iters;
#   (2) cache the reference conditioning (decode+preprocess+encode) keyed by (paths,H,W,NF) so it runs
#       ONCE per unique reference (correct for the fixed-reference outpaint use case).
CACHE_COND = os.environ.get("CACHE_COND", "1") == "1"  # default ON (bit-identical; part of best 165.5s path)
import ltx_pipelines.ic_lora as _iclora_mod

_ref_cond_cache = {}


def _install_tiled_cond():
    from ltx_core.model.video_vae import TilingConfig as _TC

    _orig_app = _iclora_mod.append_ic_lora_reference_video_conditionings
    if getattr(_orig_app, "_tiled_default", False):
        return

    def _app(*a, **k):
        if k.get("tiling_config") is None:
            k["tiling_config"] = _TC.default()
        if not CACHE_COND:
            return _orig_app(*a, **k)
        conds = a[0]
        vc = a[1] if len(a) > 1 else k.get("video_conditioning")
        key = (tuple(tuple(x) for x in vc), k.get("height"), k.get("width"), k.get("num_frames"))
        cached = _ref_cond_cache.get(key)
        if cached is not None:
            conds.extend(cached)
            if RANK == 0:
                print(f"[cache] ref-cond HIT {k.get('height')}x{k.get('width')} nf={k.get('num_frames')}", flush=True)
            return
        n0 = len(conds)
        r = _orig_app(*a, **k)
        _ref_cond_cache[key] = list(conds[n0:])
        if RANK == 0:
            print(f"[cache] ref-cond MISS->store {k.get('height')}x{k.get('width')}", flush=True)
        return r

    _app._tiled_default = True
    _iclora_mod.append_ic_lora_reference_video_conditionings = _app


_install_tiled_cond()

if CACHE_COND:
    # resident VAE encoder: build once per ImageConditioner instance, reuse (skip the per-call rebuild+free)
    from ltx_pipelines.utils.blocks import ImageConditioner as _ICc

    _enc_cache = {}

    def _icc_resident(self, fn):
        torch.cuda.synchronize()
        _tb = time.time()
        e = _enc_cache.get(id(self))
        if e is None:
            e = self._build_encoder().to(self._device)
            _enc_cache[id(self)] = e
            if RANK == 0:
                print("[cache] built resident VAE encoder (reused across calls+iters)", flush=True)
        torch.cuda.synchronize()
        _tf = time.time()
        r = fn(e)
        torch.cuda.synchronize()
        if RANK == 0 and os.environ.get("INSTRUMENT") == "1":
            print(f"[icdiag] build={_tf - _tb:.2f} fn={time.time() - _tf:.2f}", flush=True)
        return r

    _ICc.__call__ = _icc_resident

# FAST_PREP=1 (TASK #13, ref-encode lever): the stage-2 reference conditioning spends ~50s NOT in the
# VAE encode (that is already tile-sharded, ~5.8s) but in `video_preprocess`, which builds the full-res
# (1,3,720,704,1280) reference tensor via `torch.cat([result, frame], dim=2)` INSIDE the per-frame loop
# -> O(n^2) reallocation over 720 frames (~7.8 GB tensor copied ~720x). Fix (bit-identical): collect the
# per-frame tensors in a list and cat ONCE (O(n)). Benefits stage-1 AND stage-2, single-request + warm.
FAST_PREP = os.environ.get("FAST_PREP", "1") == "1"  # default ON (bit-identical O(n) cat; part of best 165.5s path)
# BCAST_PREP=1 (TASK #13 lever B): the reference mp4 decode+resize (~18s/stage) is done REDUNDANTLY on
# all 8 ranks. Decode+resize ONCE on rank0 and broadcast the preprocessed tensor (like Gemma load-once).
# NOTE: this removes redundant work; whether it cuts the WALL depends on whether the 8-way concurrent
# decode was contended (measured via [vpp] rank0 solo time). Bit-identical (rank0's tensor == each rank's).
BCAST_PREP = (
    os.environ.get("BCAST_PREP", "0") == "1"
)  # default OFF: no wall win (8-way decode was parallel-redundant -> rank0-solo same ~18s; real fix = shard the decode)
# SHARD_PREP=1 (FIX 2, default ON): the reference mp4 decode+resize (~18s/stage) is done REDUNDANTLY on
# all 8 ranks. Broadcast did NOT help (rank0 still spends ~18s; decode is sequential). SHARD the WORK
# instead: microbench shows the 18s is ~14.5s CODEC decode (sequential) + ~3.6s convert/resize -> a
# contiguous starting_frame split is useless (codec decode is not skipped). So each rank SEEKS to its
# contiguous temporal segment (av.seek to nearest keyframe, decode forward, select by pts->frame index),
# decodes+resizes ONLY its 1/WORLD of frames (~2.4s measured for a 120-frame segment), then all_gather
# reconstructs the full (1,C,F,H,W) tensor in frame order. Bit-exact (same codec frames as sequential;
# verified via NF=49 SHARD vs non-SHARD parity). Applies to stage-1 (352x640) and stage-2 (704x1280).
SHARD_PREP = os.environ.get("SHARD_PREP", "1") == "1"
if FAST_PREP:
    import av as _av
    import ltx_pipelines.iclora_utils as _icu2
    import ltx_pipelines.utils.media_io as _mio
    from ltx_pipelines.utils.media_io import normalize_images as _ni
    from ltx_pipelines.utils.media_io import resize_and_center_crop as _rcc

    _orig_decode_vbf = _mio.decode_video_by_frame
    _prep_shard = {}  # set by the (sharded) decode wrapper, read by _fast_video_preprocess (same rank, sequential)

    def _shard_decode_by_frame(path, device, starting_frame=0, frame_cap=None):
        """Seek-based per-rank temporal-segment decoder. When SHARD_PREP+WORLD>1+known frame_cap, rank r
        decodes only its contiguous chunk [r*per, r*per+per) by seeking to the frame's pts and selecting
        frames by pts->index. Otherwise identical to the original sequential decoder."""
        if not (SHARD_PREP and WORLD > 1 and frame_cap is not None):
            _prep_shard.clear()
            yield from _orig_decode_vbf(path, device, starting_frame=starting_frame, frame_cap=frame_cap)
            return
        total = frame_cap
        per = (total + WORLD - 1) // WORLD
        start = starting_frame + RANK * per
        end = min(starting_frame + total, start + per)  # exclusive
        cnt = max(0, end - start)
        _prep_shard.clear()
        _prep_shard.update(total=total, per=per, cnt=cnt, start=start)
        if cnt == 0:
            return
        container = _av.open(path)
        try:
            vs = next(s for s in container.streams if s.type == "video")
            rate = float(vs.average_rate)
            tb = vs.time_base
            target_ts = int((start / rate) / float(tb))  # pts of the first frame this rank owns
            container.seek(target_ts, stream=vs, backward=True, any_frame=False)
            for frame in container.decode(vs):
                idx = int(round(float(frame.pts * tb) * rate))
                if idx < start:
                    continue
                if idx >= end:
                    break
                yield torch.tensor(frame.to_rgb().to_ndarray(), dtype=torch.uint8, device=device).unsqueeze(0)
        finally:
            container.close()

    def _fast_video_preprocess(frames, height, width, dtype, device):
        _t = time.time()
        _do_bcast = BCAST_PREP and not SHARD_PREP and WORLD > 1
        if _do_bcast and RANK != 0:
            res = _bcast_tensor(None)  # non-rank0: skip decode+resize, receive rank0's tensor
            if RANK == 0 and os.environ.get("INSTRUMENT") == "1":
                print(f"[vpp] video_preprocess -> {tuple(res.shape)} secs={time.time() - _t:.2f} bcast", flush=True)
            return res
        out = []
        for f in frames:
            fr = _rcc(f.to(torch.float32), height, width)
            out.append(_ni(fr, device, dtype))
        sh = _prep_shard.get("total") if (SHARD_PREP and WORLD > 1) else None
        if sh is not None:
            # gather the per-rank contiguous chunks (padded to `per`), trim to each rank's ACTUAL
            # decoded frame count, and cat in frame order. The video may be shorter than frame_cap
            # (e.g. 30s clip = 720 frames but num_frames=721) -> the last populated rank decodes fewer
            # than `per`; trusting frame_cap would fabricate a padding frame and change the token count.
            per = _prep_shard["per"]
            C = out[0].shape[1] if out else 3
            n_local = len(out)
            local = torch.cat(out, dim=2) if out else torch.zeros((1, C, 0, height, width), device=device, dtype=dtype)
            cnt_t = torch.tensor([n_local], device=device, dtype=torch.long)
            cl = [torch.zeros_like(cnt_t) for _ in range(WORLD)]
            dist.all_gather(cl, cnt_t)
            counts = [int(c.item()) for c in cl]
            if local.shape[2] < per:
                pad = torch.zeros((1, C, per - local.shape[2], height, width), device=device, dtype=dtype)
                local = torch.cat([local, pad], dim=2)
            g = [torch.empty_like(local) for _ in range(WORLD)]
            dist.all_gather(g, local.contiguous())
            chunks = [g[r][:, :, : counts[r], :, :] for r in range(WORLD) if counts[r] > 0]
            res = torch.cat(chunks, dim=2).contiguous()
        else:
            if not out:
                raise ValueError("video_preprocess received an empty frame generator.")
            res = torch.cat(out, dim=2).contiguous()
        if RANK == 0 and os.environ.get("INSTRUMENT") == "1":
            print(
                f"[vpp] video_preprocess -> {tuple(res.shape)} secs={time.time() - _t:.2f} "
                f"shard={sh is not None} local_frames={len(out)}",
                flush=True,
            )
        return res

    _mio.video_preprocess = _fast_video_preprocess
    _icu2.video_preprocess = _fast_video_preprocess  # append_ic_lora resolves it from iclora_utils
    _mio.decode_video_by_frame = _shard_decode_by_frame
    _icu2.decode_video_by_frame = _shard_decode_by_frame

import gc as _gc

BUILT = []
STAGE_VMS = []  # DiT velocity_models in build order: [stage-1 base, stage-2 upscaler]
ENGINES = []  # per-stage compiled TRT engines (parallel to STAGE_VMS); TRT_DIT only

RESIDENT_DIT = (
    os.environ.get("RESIDENT_DIT", "0") == "1"
)  # default OFF: REGRESSED (GPU-resident OOMs at 30s; CPU-resident H2D move 37.5s > 27s reload)


def free_dit():
    if RESIDENT_DIT:
        # keep the built+sharded DiT stages resident across iters (TASK #13b): skip freeing the blocks
        _gc.collect()
        torch.cuda.empty_cache()
        return
    for m in BUILT:
        vm = getattr(m, "velocity_model", m)
        if hasattr(vm, "transformer_blocks"):
            vm.transformer_blocks = torch.nn.ModuleList()
    _gc.collect()
    torch.cuda.empty_cache()


if TRT_DIT:
    # ===== BOTH DiT stages -> per-rank single native TRT engines (compile inline from the NF=721
    #       CP4xTP2 sharded .ep; collectives fuse into Myelin + route over CP/TP subgroup comms) =====
    from ltx_core.model.transformer.model import LTXModel

    def targs(a):
        return [
            a.x,
            a.context,
            a.timesteps,
            a.embedded_timestep,
            a.positional_embeddings[0],
            a.positional_embeddings[1],
            a.cross_positional_embeddings[0],
            a.cross_positional_embeddings[1],
            a.cross_scale_shift_timestep,
            a.cross_gate_timestep,
            a.prompt_timestep,
        ]

    WS_BUDGET = int(os.environ.get("WS_BUDGET_GB", "6")) * (1024**3)
    TRT_STAGES = [int(c) for c in os.environ.get("TRT_STAGES", "12") if c in "12"]  # [1,2] both, or [1]
    # TRT_WS (FIX 1): 1 = weight-streamed engines (default, needed when both stages TRT'd / don't fit);
    # 0 = RESIDENT engines (no enable_weight_streaming, no runtime streaming). For TRT_STAGES=1 the
    # stage-1 engine (~11GB/rank CP2xTP4) fits resident alongside the ~46GB pipeline peak (~57GB < 80GB)
    # -> the real ~2x DiT speedup (WS'd stage-1 was 42.5s > 26s eager; resident targets ~13s). See INTEGRATED_e2e.md.
    TRT_WS = os.environ.get("TRT_WS", "1") == "1"
    ENGINE_BY_STAGE = {}  # stage_num -> compiled engine
    _ws_ctxs = []
    log(
        f"[trt] TRT_STAGES={TRT_STAGES} TRT_WS={TRT_WS} DIST_TRACE={os.environ.get('DIST_TRACE', '1')} OFFLOAD_CPU={os.environ.get('OFFLOAD_CPU', '1')}"
    )
    for _s in TRT_STAGES:
        _base = f"dit_stage{_s}_rank{RANK}"
        log(f"[trt] compiling {_base} (min_block_size=1, WS={'on' if TRT_WS else 'OFF/resident'})...")
        _ep = torch.export.load(f"{SHARD_DIR}/{_base}.pt2")
        _raw = torch.load(f"{SHARD_DIR}/{_base}_inputs.pt")
        _ein = [(t.cuda() if torch.is_tensor(t) else t) for t in _raw]
        dist.barrier()
        _t0 = time.time()
        _ckw = dict(
            min_block_size=1,
            use_distributed_mode_trace=bool(int(os.environ.get("DIST_TRACE", "1"))),
            use_python_runtime=False,
            use_explicit_typing=True,
        )
        if TRT_WS:
            _ckw["enable_weight_streaming"] = True  # FIX 1: omit -> resident engine
        if os.environ.get("OFFLOAD_CPU", "1") == "1":
            _ckw["offload_module_to_cpu"] = True
        _cmb = os.environ.get("CPU_MEM_BUDGET")
        if _cmb:
            _ckw["cpu_memory_budget"] = int(_cmb)
        _opt = os.environ.get("OPT")
        if _opt not in (None, ""):
            _ckw["optimization_level"] = int(_opt)
        _wsm = os.environ.get("WORKSPACE_MB")
        if _wsm:
            _ckw["workspace_size"] = int(_wsm) * 1024 * 1024
        _eng = torch_tensorrt.dynamo.compile(_ep, inputs=_ein, **_ckw)
        _nsub = sum(1 for n, _ in _eng.named_modules() if "_run_on_acc" in n)
        log(f"[trt] {_base} COMPILE_OK secs={time.time() - _t0:.0f} submodules={_nsub}")
        ENGINE_BY_STAGE[_s] = _eng
        del _ep, _ein, _raw
        _gc.collect()
        torch.cuda.empty_cache()
        if TRT_WS:
            # enter weight-streaming on this engine immediately (keep it resident-low while the NEXT
            # stage compiles, through pipeline build, and across all benchmark iters)
            try:
                _c = ttrt_rt.weight_streaming(_eng)
                _ws = _c.__enter__()
                try:
                    _ws.device_budget = WS_BUDGET
                    log(f"[trt] {_base} weight_streaming budget={WS_BUDGET} total={_ws.total_device_budget}")
                except Exception as e:
                    log("[trt] ws budget note:", repr(e))
                _ws_ctxs.append(_c)
            except Exception as e:
                log("[trt] weight_streaming unavailable:", repr(e))
        else:
            log(f"[trt] {_base} RESIDENT (no weight-streaming)")
            memlog(f"after_compile_{_base}")
        _gc.collect()
        torch.cuda.empty_cache()

    _rescache = {}

    def resident_call(self, *a, **k):
        key = id(self)
        if key not in _rescache:
            m = self._build_transformer(device=torch.device("cuda"), **k)
            vm = getattr(m, "velocity_model", m)
            install(vm)  # TP/CP shard + Ulysses (mirrors proven render)
            _stage_num = len(STAGE_VMS) + 1  # build order: 1=base, 2=upscaler
            STAGE_VMS.append(vm)
            if _stage_num in ENGINE_BY_STAGE:
                vm.transformer_blocks = torch.nn.ModuleList()  # TRT'd stage: blocks live in the engine
            # else: keep blocks -> this stage runs eager (installed CP/TP shard)
            _rescache[key] = m
            BUILT.append(m)
            _gc.collect()
            torch.cuda.empty_cache()
        return self.run(_rescache[key], *a, **k)

    DiffusionStage.__call__ = resident_call

    _orig_fwd = LTXModel.forward

    def wired_fwd(self, video, audio, perturbations):
        _stage_num = (STAGE_VMS.index(self) + 1) if self in STAGE_VMS else 0
        eng = ENGINE_BY_STAGE.get(_stage_num)
        if video is not None and eng is not None:
            va = self.video_args_preprocessor.prepare(video, audio)
            aa = self.audio_args_preprocessor.prepare(audio, video)
            va = replace(
                va,
                x=sh_seq(va.x, 1),
                timesteps=sh_seq(va.timesteps, 1) if va.timesteps.shape[1] > 1 else va.timesteps,
                embedded_timestep=sh_seq(va.embedded_timestep, 1),
                positional_embeddings=pe_v(va.positional_embeddings),
                cross_positional_embeddings=pe_v(va.cross_positional_embeddings),
                cross_scale_shift_timestep=(
                    sh_seq(va.cross_scale_shift_timestep, 1)
                    if va.cross_scale_shift_timestep is not None and va.cross_scale_shift_timestep.shape[1] > 1
                    else va.cross_scale_shift_timestep
                ),
                cross_gate_timestep=(
                    sh_seq(va.cross_gate_timestep, 1)
                    if va.cross_gate_timestep is not None and va.cross_gate_timestep.shape[1] > 1
                    else va.cross_gate_timestep
                ),
            )
            aa = replace(aa, cross_positional_embeddings=pe_a(aa.cross_positional_embeddings))
            args = [(t.contiguous() if torch.is_tensor(t) else t) for t in (targs(va) + targs(aa))]
            torch.cuda.synchronize()
            _t = time.time()
            vxo, axo = eng(*args)
            xs = [torch.empty_like(vxo) for _ in range(CP)]
            dist.all_gather(xs, vxo.contiguous(), group=cp_group)
            out = torch.cat(xs, 1)
            torch.cuda.synchronize()
            STAGE[f"dit_s{_stage_num}"] = STAGE.get(f"dit_s{_stage_num}", 0.0) + (time.time() - _t)
            return out, axo
        # eager fallback for a known stage (e.g. stage-2 not TRT'd): time it for the breakdown
        if self in STAGE_VMS:
            _sn = STAGE_VMS.index(self) + 1
            torch.cuda.synchronize()
            _te = time.time()
            _r = _orig_fwd(self, video, audio, perturbations)
            torch.cuda.synchronize()
            STAGE[f"dit_s{_sn}"] = STAGE.get(f"dit_s{_sn}", 0.0) + (time.time() - _te)
            return _r
        return _orig_fwd(self, video, audio, perturbations)

    LTXModel.forward = wired_fwd
else:
    _ws_ctxs = []
    _ob = DiffusionStage._build_transformer
    # SHARD_ON_CPU (40GB boxes): the stock build loads the full unsharded 22B (~44GB bf16) straight
    # onto the GPU and only then slices it for TP, so the *build spike* is 44GB even at TP=2. That is
    # fine on 80GB and fatal on A100-40GB (OOMs mid-`load_state_dict`). Build on CPU, install the
    # CP/TP shard there (pure slicing + hooks, device-agnostic), then move only the ~22GB per-rank
    # shard to the GPU. Numerically identical -- same slices, same hooks, same order.
    SHARD_ON_CPU = os.environ.get("SHARD_ON_CPU", "0") == "1"

    def _bt(self, *a, **k):
        if SHARD_ON_CPU and WORLD > 1:
            k = dict(k)
            k["device"] = torch.device("cpu")
            m = _ob(self, *a, **k)
            install(getattr(m, "velocity_model", m))
            m = m.to("cuda")
        else:
            m = _ob(self, *a, **k)
            if WORLD > 1:
                install(getattr(m, "velocity_model", m))
        _compile_dit(m)
        _time_dit(m)
        BUILT.append(m)
        torch.cuda.empty_cache()
        return m

    DiffusionStage._build_transformer = _bt
    if RESIDENT_DIT:
        # TASK #13b: the block lifecycle rebuilds+reloads the 22B DiT (both stages, ~27s) every generate.
        # GPU-resident BOTH stages OOMs at 30s (2x sharded 22B + 79,640-token stage-2 activations > 80GB).
        # -> CPU-resident: build+shard ONCE per stage, keep the weights in CPU RAM, and fast-move ONLY the
        # ACTIVE stage to GPU for its denoise (move back after). One stage's weights on GPU at a time (fits
        # like the build/free lifecycle) but replaces the ~14s disk rebuild with a ~0.5-1s H2D copy.
        # RESIDENT_DIT_MODE=gpu forces both-on-GPU (works at NF=49, OOMs at 30s).
        from contextlib import contextmanager as _ctxmgr

        _dit_resident = {}
        _dit_park = {}
        RESIDENT_GPU = os.environ.get("RESIDENT_DIT_MODE", "cpu") == "gpu"

        @_ctxmgr
        def _resident_transformer_ctx(self, **kwargs):
            key = id(self)
            if key not in _dit_resident:
                m = self._build_transformer(**kwargs)  # _bt: build + install shard, on GPU
                if not RESIDENT_GPU:
                    if FLAT_MOVE:
                        _dit_park[key] = FlatPark(m)  # pinned host buffers, one per dtype
                    else:
                        m.to("cpu")
                        torch.cuda.empty_cache()  # park weights in CPU RAM
                _dit_resident[key] = m
                if RANK == 0:
                    print(
                        f"[resident] built+sharded DiT stage mode={'gpu' if RESIDENT_GPU else 'cpu'} (n={len(_dit_resident)})",
                        flush=True,
                    )
            m = _dit_resident[key]
            if not RESIDENT_GPU:
                torch.cuda.synchronize()
                _tm = time.time()
                if FLAT_MOVE:
                    _dit_park[key].to_device()  # one pinned copy per dtype
                else:
                    m.to("cuda")  # fast-move active stage onto GPU
                torch.cuda.synchronize()
                if RANK == 0:
                    STAGE["dit_hbm_move"] = STAGE.get("dit_hbm_move", 0.0) + (time.time() - _tm)
            try:
                yield m
            finally:
                if not RESIDENT_GPU:
                    torch.cuda.synchronize()
                    _tm2 = time.time()
                    if FLAT_MOVE:
                        _dit_park[key].to_host()  # pointer swap; weights are read-only
                    else:
                        m.to("cpu")
                        torch.cuda.empty_cache()  # park back in CPU RAM
                    if RANK == 0:
                        STAGE["dit_hbm_move"] = STAGE.get("dit_hbm_move", 0.0) + (time.time() - _tm2)

        DiffusionStage._transformer_ctx = _resident_transformer_ctx

# ---- OPTIONAL full-wall instrumentation (INSTRUMENT=1): attribute EVERY second of the warm E2E ----
# Wraps the pipeline's real cost centers with cuda-synced timers (rank0-local). Top-level attribution:
#   wall = gemma + vae_encode + stage1_call + upsampler + stage2_call + vae_decode + residual
# where stageN_call = stageN_build_load + ditN_denoise_fwd(xSteps) + step/sampler overhead.
PROF = {}


def _prof_add(k, dt):
    e = PROF.setdefault(k, [0.0, 0])
    e[0] += dt
    e[1] += 1


_fwd_order = []
_ds_order = []
if os.environ.get("INSTRUMENT") == "1":
    from ltx_core.model.transformer.model import LTXModel as _LM
    from ltx_pipelines.utils.blocks import VideoUpsampler as _VU

    _shape_logged = set()
    _orig_lm_fwd_i = _LM.forward

    def _lm_fwd_i(self, video, audio, perturbations):
        torch.cuda.synchronize()
        _t = time.time()
        r = _orig_lm_fwd_i(self, video, audio, perturbations)
        torch.cuda.synchronize()
        if id(self) not in _fwd_order:
            _fwd_order.append(id(self))
        _sn = _fwd_order.index(id(self)) + 1
        _prof_add(f"dit{_sn}_denoise_fwd", time.time() - _t)
        if _sn not in _shape_logged and video is not None:
            _shape_logged.add(_sn)
            try:
                log(f"[instr] stage{_sn} fwd video.x={tuple(video.x.shape)}")
            except Exception:
                pass
        return r

    _LM.forward = _lm_fwd_i
    _orig_ds_call_i = DiffusionStage.__call__

    def _ds_call_i(self, *a, **k):
        if id(self) not in _ds_order:
            _ds_order.append(id(self))
        _sn = _ds_order.index(id(self)) + 1
        torch.cuda.synchronize()
        _t = time.time()
        r = _orig_ds_call_i(self, *a, **k)
        torch.cuda.synchronize()
        _prof_add(f"stage{_sn}_call", time.time() - _t)
        return r

    DiffusionStage.__call__ = _ds_call_i
    _orig_bt_i = DiffusionStage._build_transformer

    def _bt_i(self, *a, **k):
        _sn = (_ds_order.index(id(self)) + 1) if id(self) in _ds_order else 0
        torch.cuda.synchronize()
        _t = time.time()
        m = _orig_bt_i(self, *a, **k)
        torch.cuda.synchronize()
        _prof_add(f"stage{_sn}_build_load", time.time() - _t)
        return m

    DiffusionStage._build_transformer = _bt_i
    _orig_install_i = install

    def _install_i(vm):
        torch.cuda.synchronize()
        _t = time.time()
        r = _orig_install_i(vm)
        torch.cuda.synchronize()
        _prof_add("dit_install_shard", time.time() - _t)
        return r

    install = _install_i
    _orig_vu_i = _VU.__call__

    def _vu_i(self, *a, **k):
        torch.cuda.synchronize()
        _t = time.time()
        r = _orig_vu_i(self, *a, **k)
        torch.cuda.synchronize()
        _prof_add("upsampler", time.time() - _t)
        return r

    _VU.__call__ = _vu_i
    # residual attribution: each of these blocks rebuilds+reloads its model EVERY call
    #   image_conditioner = VAE-encoder build/load + reference-mp4 decode + tiled encode (called x2)
    #   video_decoder = VAE-decoder build/load (decode compute is the lazy generator -> vae_decode)
    #   audio_decode  = audio decoder+vocoder build/load + decode compute
    from ltx_pipelines.utils.blocks import AudioDecoder as _AD
    from ltx_pipelines.utils.blocks import ImageConditioner as _IC
    from ltx_pipelines.utils.blocks import VideoDecoder as _VD

    def _wrap_call(cls, name):
        _o = cls.__call__

        def _w(self, *a, **k):
            torch.cuda.synchronize()
            _t = time.time()
            r = _o(self, *a, **k)
            torch.cuda.synchronize()
            _prof_add(name, time.time() - _t)
            return r

        cls.__call__ = _w

    _wrap_call(_IC, "image_conditioner")
    _wrap_call(_VD, "video_decoder_build")
    _wrap_call(_AD, "audio_decode")
    log("[instr] INSTRUMENT=1 active")

# OFFLOAD_MODE (TASK #14): none|cpu|disk. cpu = FlexTensor weight-streaming (pinned CPU RAM -> small GPU
# buffer, layer-by-layer) so the 22B DiT fits on 1 GPU for the single-device 30s baseline (which OOMs at
# OffloadMode.NONE). Only meaningful at WORLD==1.
from ltx_pipelines.utils.types import OffloadMode as _OffloadMode

_offmode = _OffloadMode(os.environ.get("OFFLOAD_MODE", "none"))
if _offmode != _OffloadMode.NONE:
    log(f"[offload] ICLoraPipeline offload_mode={_offmode}")
pipe = ICLoraPipeline(
    distilled_checkpoint_path=D,
    spatial_upsampler_path=U,
    gemma_root=G,
    loras=[LoraPathStrengthAndSDOps(L, 1.0, LTXV_LORA_COMFY_RENAMING_MAP)],
    offload_mode=_offmode,
)
call = getattr(type(pipe).__call__, "__wrapped__", type(pipe).__call__)

dist.barrier()
walls = []
_INSTR = os.environ.get("INSTRUMENT") == "1"
for it in range(int(os.environ.get("ITERS", "2"))):
    STAGE.clear()
    if _INSTR:
        PROF.clear()
        _fwd_order.clear()
    dist.barrier()
    t0 = time.time()
    with torch.no_grad():
        video, audio = call(
            pipe,
            prompt="extend the scene naturally, consistent with the original footage",
            seed=171198,
            height=704,
            width=1280,
            num_frames=NF,
            frame_rate=24,
            images=[],
            video_conditioning=[(COND, 1.0)],
            tiling_config=TilingConfig.default(),
            enhance_prompt=False,
        )
    # free DiT before consuming the (sharded) decode iterator so it fits; decode collectives
    # are driven by ALL ranks (each rank iterates the generator in lockstep).
    memlog("after_generate")
    free_dit()
    memlog("after_free_dit")
    # CRITICAL: iterate the decode generator UNDER no_grad. call() is the pipeline's
    # __wrapped__ (its @torch.inference_mode() is stripped), and the decode is a lazy
    # generator consumed HERE, outside the generate no_grad block. Without this, every
    # decode forward builds an autograd graph whose activations are retained across tile
    # groups (buffer/prev_chunk carry grad) -> memory grows ~linearly per group -> OOM at
    # 30s (fine at NF=49 = 1 group). Also move each chunk to CPU as produced.
    with torch.no_grad(), timed("vae_decode", STAGE):
        video_chunks = [c.detach().to("cpu") for c in video]
    memlog("after_decode")
    torch.cuda.synchronize()
    dist.barrier()
    walls.append(time.time() - t0)
    log(f"iter {it} wall_s={walls[-1]:.2f} stages={ {k: round(v, 3) for k, v in STAGE.items()} }")
    if _INSTR and RANK == 0:

        def _P(k):
            return PROF.get(k, [0.0, 0])

        w = walls[-1]
        g = STAGE.get("gemma", 0.0)
        ve = STAGE.get("vae_encode", 0.0)
        vd = STAGE.get("vae_decode", 0.0)
        s1 = _P("stage1_call")[0]
        s2 = _P("stage2_call")[0]
        up = _P("upsampler")[0]
        ic = _P("image_conditioner")[0]
        vdb = _P("video_decoder_build")[0]
        ad = _P("audio_decode")[0]
        # mutually-exclusive top-level buckets (ic INCLUDES the vae_encode compute; vd = lazy decode consume)
        acc = g + ic + s1 + up + s2 + vdb + vd + ad
        d1 = _P("dit1_denoise_fwd")
        d2 = _P("dit2_denoise_fwd")
        b1 = _P("stage1_build_load")[0]
        b2 = _P("stage2_build_load")[0]
        print(
            f"DECOMP iter{it} wall={w:.2f} sum_top={acc:.2f} residual={w - acc:.2f}\n"
            f"  gemma={g:.2f}\n"
            f"  image_conditioner(x{_P('image_conditioner')[1]})={ic:.2f} [of which vae_encode_compute={ve:.2f}, rest=VAEenc_build+ref_decode]\n"
            f"  stage1_call={s1:.2f} [build_load={b1:.2f} denoise_fwd={d1[0]:.2f}x{d1[1]} step_other={s1 - b1 - d1[0]:.2f}]\n"
            f"  upsampler={up:.2f}\n"
            f"  stage2_call={s2:.2f} [build_load={b2:.2f} denoise_fwd={d2[0]:.2f}x{d2[1]} step_other={s2 - b2 - d2[0]:.2f}]\n"
            f"  video_decoder_build={vdb:.2f}  vae_decode_compute={vd:.2f}  audio_decode={ad:.2f}\n"
            f"  install_shard_total={_P('dit_install_shard')[0]:.2f}\n"
            f"  PROF={ {k: [round(v[0], 2), v[1]] for k, v in PROF.items()} }",
            flush=True,
        )

for _c in _ws_ctxs:
    try:
        _c.__exit__(None, None, None)
    except Exception:
        pass
pk = torch.tensor([torch.cuda.max_memory_allocated() / 1e9], device="cuda")
dist.all_reduce(pk, op=dist.ReduceOp.MAX)
# Every rank needs this tensor: it is broadcast from rank 0 after encoding so
# the whole job can exit non-zero when no mp4 was written.
encode_ok = torch.tensor([1], device="cuda", dtype=torch.int32)
if RANK == 0:
    import statistics as _st

    _dit = "trt" if TRT_DIT else "eager"
    _btag = os.environ.get("BENCH_TAG", "")
    tag = f"cp{CP}tp{TP}_{_dit}" + (f"_{_btag}" if _btag else "")
    _warm = walls[1:] if len(walls) > 1 else walls
    _med = _st.median(_warm)
    # Encode before announcing RESULT. The harness treats a RESULT line as "this
    # configuration succeeded" and then looks for the mp4 to parity-check, so
    # emitting it first meant a failed encode produced a successful-looking run
    # with no output: the parity step would find nothing to compare and skip it.
    outp = f"{OUTDIR}/integ_{tag}_nf{NF}.mp4"
    try:
        encode_video(
            video=iter(video_chunks),
            fps=24,
            audio=None,
            output_path=outp,
            video_chunks_number=get_video_chunks_number(NF, TilingConfig.default()),
        )
    except Exception as e:
        print(f"ENCODE_FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
        encode_ok[0] = 0
    else:
        print(f"SAVED {outp}", flush=True)
        print(
            f"RESULT {tag} NF={NF} median_wall_s={_med:.2f} warm={[round(w, 2) for w in _warm]} all={[round(w, 2) for w in walls]} "
            f"peak_gpu_gb={pk.item():.1f} stages={ {k: round(v, 3) for k, v in STAGE.items()} }",
            flush=True,
        )
# Tell every rank whether rank 0 produced an output. Exiting from rank 0 alone
# would leave the others blocked in the barrier below.
dist.broadcast(encode_ok, src=0)
dist.barrier()
dist.destroy_process_group()
if int(encode_ok.item()) == 0:
    sys.exit(1)
