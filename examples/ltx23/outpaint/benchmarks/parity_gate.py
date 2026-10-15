# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare two generated videos and **fail** if they diverge beyond the acceptance threshold.

Reports PSNR and cosine, and exits non-zero when the bar is not met, so a parity regression fails
the job instead of merely appearing in its log.

Cosine is the meaningful metric for this bf16 DiT. Perturbing the eager reference's input at bf16
rounding scale already moves cosine to ~0.9957, so a max-absolute-error bound would reject correct
results. ``--exact`` additionally demands cosine 1.0 and PSNR 99 dB, for changes claimed to be
bit-exact.

    python parity_gate.py REF.mp4 CANDIDATE.mp4 [--min-cosine 0.995] [--exact] [--label NAME]
"""

from __future__ import annotations

import argparse
import sys

import imageio.v3 as iio
import numpy as np

EXACT_PSNR_DB = 99.0
EXACT_COSINE = 1.0


def load(path: str) -> np.ndarray:
    return np.stack(list(iio.imiter(path))).astype(np.float32)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return EXACT_PSNR_DB if mse < 1e-9 else float(10 * np.log10((255.0**2) / mse))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--min-cosine", type=float, default=0.995)
    parser.add_argument("--min-psnr", type=float, default=0.0)
    parser.add_argument(
        "--exact",
        action="store_true",
        help="require bit-exactness (cosine 1.000000 / PSNR 99 dB), for changes claimed exact",
    )
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    tag = f"[{args.label}] " if args.label else ""
    reference, candidate = load(args.reference), load(args.candidate)
    frames = min(len(reference), len(candidate))
    reference, candidate = reference[:frames], candidate[:frames]

    # A shape mismatch is a failure, not a note: the two runs did not produce the same workload.
    if reference.shape != candidate.shape:
        print(f"{tag}PARITY_FAIL shape mismatch ref={reference.shape} cand={candidate.shape}")
        return 1

    overall_psnr = psnr(reference, candidate)
    flat_reference, flat_candidate = reference.ravel(), candidate.ravel()
    cosine = float(
        np.dot(flat_reference, flat_candidate)
        / (np.linalg.norm(flat_reference) * np.linalg.norm(flat_candidate) + 1e-12)
    )
    per_frame = [psnr(reference[i], candidate[i]) for i in range(frames)]

    print(
        f"{tag}frames={frames} shape={reference.shape[1:]} overall_PSNR={overall_psnr:.2f}dB "
        f"cosine={cosine:.6f} per_frame_PSNR_min={min(per_frame):.2f} "
        f"max={max(per_frame):.2f} mean={float(np.mean(per_frame)):.2f}"
    )

    min_cosine = EXACT_COSINE if args.exact else args.min_cosine
    min_psnr = EXACT_PSNR_DB if args.exact else args.min_psnr
    failures = []
    if cosine < min_cosine:
        failures.append(f"cosine {cosine:.6f} < {min_cosine:.6f}")
    if overall_psnr < min_psnr:
        failures.append(f"PSNR {overall_psnr:.2f} < {min_psnr:.2f}")
    if failures:
        print(f"{tag}PARITY_FAIL {'; '.join(failures)}")
        return 1
    print(f"{tag}PARITY_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
