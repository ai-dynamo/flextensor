# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reclaim one stopped Outpaint run without loading models or using a GPU."""

from __future__ import annotations

import argparse
import sys

from flextensor import SharedOffloadSession


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shm-namespace", required=True)
    args = parser.parse_args()
    try:
        SharedOffloadSession.cleanup_namespace(
            args.shm_namespace,
            manager_names=("ltx_stage1", "ltx_stage2", "ltx_text"),
            timeout=60,
        )
    except Exception as error:
        print(f"SHM cleanup failed for namespace {args.shm_namespace}: {error}", file=sys.stderr)
        return 1
    print(f"SHM cleanup verified for namespace: {args.shm_namespace}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
