# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the looped conditioning clip.

The compute nodes have no ffmpeg, and PyAV is what the LTX environment already ships. Source
timestamps are not monotonic across loop iterations, so they are dropped and the encoder reassigns
them -- otherwise the mux fails with EINVAL and still leaves a partial, decodable file behind.
"""

import sys

import av


def main() -> None:
    src, dst, want = sys.argv[1], sys.argv[2], int(sys.argv[3])
    with av.open(src) as probe:
        stream = probe.streams.video[0]
        width = stream.codec_context.width
        height = stream.codec_context.height
        rate = stream.average_rate

    out = av.open(dst, "w")
    ostream = out.add_stream("libx264", rate=rate)
    # Lossless, no chroma subsampling. At crf 16 / yuv420p the inner content bleeds into the
    # letterbox bars, and the Outpaint IC-LoRA fills regions that are *exactly* RGB(0,0,0).
    ostream.width, ostream.height, ostream.pix_fmt = width, height, "yuv444p"
    ostream.options = {"crf": "0"}

    written = 0
    while written < want:
        before = written
        with av.open(src) as clip:
            for frame in clip.decode(video=0):
                if written >= want:
                    break
                frame.pts = None
                for packet in ostream.encode(frame):
                    out.mux(packet)
                written += 1
        # A malformed or empty source decodes to nothing, and the loop would otherwise spin
        # forever. Fail loudly instead -- a benchmark that hangs is worse than one that stops.
        if written == before:
            out.close()
            raise SystemExit(f"{src} yielded no frames; cannot build a {want}-frame clip")
    for packet in ostream.encode():
        out.mux(packet)
    out.close()

    with av.open(dst) as check:
        stream = check.streams.video[0]
        print(f"BUILT {dst} frames {stream.frames} {width}x{height} fps {float(rate)}")


if __name__ == "__main__":
    main()
