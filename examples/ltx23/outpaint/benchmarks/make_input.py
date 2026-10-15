# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthesise a conforming conditioning clip so the benchmark runs without source footage.

The published measurements were taken on customer footage that cannot be committed. The benchmark's
cost, though, is set by the canvas, frame count and frame rate -- not by what the pixels show: the
DiT sees a fixed 1280x704 latent grid either way. So a deterministic synthetic clip reproduces the
*timing* comparison exactly, and lets someone verify the speedup claims with nothing but this repo.

What it does not reproduce is the *look* of the published mp4s. Parity is unaffected, because the
gate compares a candidate against a reference generated from the same input in the same run -- never
against our published output.

Geometry matches ``letterbox.py``: the content is centred on the canvas and everything outside it is
exact RGB(0,0,0), which is the signal the Outpaint IC-LoRA fills.

    python make_input.py --output inputs/real_input_8s.mp4
"""

from __future__ import annotations

import argparse

import av
import numpy as np

# Enough structure that a wrong-looking output is obvious to the eye, and enough motion that
# frame-to-frame caching cannot accidentally flatter a configuration.
BAR_HZ = 0.75


def build_frame(index: int, total: int, inner_w: int, inner_h: int) -> np.ndarray:
    """Return one deterministic inner-content frame as uint8 RGB."""
    phase = index / max(total - 1, 1)
    # Broadcast both axes to the full frame up front: a term that depends on only one axis would
    # otherwise stay (H,1) or (1,W) and fail to stack with the others.
    ys, xs = np.meshgrid(
        np.linspace(0.0, 1.0, inner_h, dtype=np.float32),
        np.linspace(0.0, 1.0, inner_w, dtype=np.float32),
        indexing="ij",
    )

    # A drifting diagonal gradient plus a travelling bar: smooth low-frequency content in one
    # channel, a hard moving edge in another, so both smooth and sharp regions are exercised.
    red = np.clip((xs + ys) * 0.5 + 0.25 * np.sin(2 * np.pi * (phase + xs)), 0.0, 1.0)
    green = np.clip(0.5 + 0.5 * np.sin(2 * np.pi * (BAR_HZ * phase + ys * 3.0)), 0.0, 1.0)
    bar = np.abs(xs - ((phase * 1.6) % 1.0)) < 0.04
    blue = np.where(bar, 1.0, 0.15 + 0.35 * ys)

    frame = np.stack([red, green, blue], axis=-1)
    return (frame * 255.0).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, default=1280, help="canvas width (multiple of 64)")
    parser.add_argument("--height", type=int, default=704, help="canvas height (multiple of 64)")
    parser.add_argument("--inner-width", type=int, default=768)
    parser.add_argument("--inner-height", type=int, default=448)
    parser.add_argument("--seconds", type=float, default=8.5)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    if args.inner_width > args.width or args.inner_height > args.height:
        raise SystemExit("Inner content must fit inside the canvas; outpaint only extends it.")
    for name, value in (("width", args.width), ("height", args.height)):
        if value % 64:
            raise SystemExit(f"Canvas {name} {value} must be a multiple of 64.")

    total = round(args.seconds * args.fps)
    x0 = (args.width - args.inner_width) // 2
    y0 = (args.height - args.inner_height) // 2

    container = av.open(args.output, mode="w")
    stream = container.add_stream("libx264", rate=args.fps)
    stream.width, stream.height = args.width, args.height
    # Lossless, and no chroma subsampling -- matching letterbox.py. At crf 16 / yuv420p the
    # high-contrast inner content bleeds into the bars (measured: peak 22, not 0), and the Outpaint
    # IC-LoRA keys on regions being *exactly* RGB(0,0,0). Near-black is not the same signal.
    stream.pix_fmt = "yuv444p"
    stream.options = {"crf": "0"}

    for index in range(total):
        canvas = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        canvas[y0 : y0 + args.inner_height, x0 : x0 + args.inner_width] = build_frame(
            index, total, args.inner_width, args.inner_height
        )
        packet_frame = av.VideoFrame.from_ndarray(canvas, format="rgb24")
        for packet in stream.encode(packet_frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    print(
        f"synthetic conditioning clip: {args.output} "
        f"canvas={args.width}x{args.height} inner={args.inner_width}x{args.inner_height} "
        f"frames={total} fps={args.fps}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
