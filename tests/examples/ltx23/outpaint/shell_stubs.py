# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small command stubs for executing the Outpaint shell scripts without models or services."""

from __future__ import annotations

import json
import os
import resource
import signal
import subprocess  # noqa: S404 - inspects process groups created by the test.
import sys
import time
from pathlib import Path
from string import Template
from typing import Any, NoReturn


def group_is_alive(pgid: int) -> bool:
    result = subprocess.run(  # noqa: S603 - inspects a process group created by the test.
        ["ps", "-o", "stat=", "-g", str(pgid)],  # noqa: S607 - standard system utility.
        capture_output=True,
        text=True,
        check=False,
    )
    return any(state and not state.startswith("Z") for state in result.stdout.split())


def _start_ranks(state: Path, args: list[str]) -> None:
    # Like torchrun, execute each rank in its own session, including any --no-python wrapper.
    command_index = next(index for index, arg in enumerate(args) if not arg.startswith("--"))
    command = args[command_index:]
    if "--no-python" not in args:
        command = [str(Path(sys.argv[1]).with_name("python")), *command]
    rank_count = int(next(arg.split("=", 1)[1] for arg in args if arg.startswith("--nproc-per-node=")))
    ranks = [
        subprocess.Popen(  # noqa: S603 - runs the launcher's rank command with the fixture's Python stub.
            [Template(arg).safe_substitute(local_rank=str(rank)) for arg in command],
            env=os.environ | {"SHELL_TEST_RANK": str(rank)},
            start_new_session=True,
        )
        for rank in range(rank_count)
    ]

    def stop_agent(*_: object) -> NoReturn:
        for rank in ranks:
            if rank.poll() is None:
                os.killpg(rank.pid, signal.SIGTERM)
        if os.environ.get("SHELL_TEST_AGENT_EXIT") != "1":
            for rank in ranks:
                rank.wait(timeout=30)
        sys.exit(0)

    signal.signal(signal.SIGTERM, stop_agent)
    for rank in ranks:
        while not (state / f"rank-{rank.pid}").exists():
            if rank.poll() is not None:
                sys.exit(1)
            time.sleep(0.01)


def _serve_rank(state: Path) -> NoReturn:
    if os.environ.get("SHELL_TEST_STUBBORN_CHILD"):
        signal.signal(signal.SIGTERM, lambda *_: (state / f"term-{os.getpid()}").touch())
    (state / f"group-{os.getpgrp()}").touch()
    (state / f"rank-{os.getpid()}").write_text(str(os.getpgrp()))
    while True:
        signal.pause()


def _serve_worker(state: Path, call: dict[str, Any]) -> NoReturn:
    if "SHELL_TEST_RANK" in os.environ:
        _serve_rank(state)
    args = call["args"]
    port = args[args.index("--port") + 1]
    if call["command"] == "torchrun":
        _start_ranks(state, args)
    elif os.environ.get("SHELL_TEST_STUBBORN_CHILD"):
        child = os.fork()
        if child == 0:
            signal.signal(signal.SIGTERM, lambda *_: (state / f"term-{os.getpid()}").touch())
            (state / f"child-{os.getpid()}").touch()
            while True:
                signal.pause()
        while not (state / f"child-{child}").exists():
            time.sleep(0.01)

    worker_file = state / f"worker-{port}.json"
    worker_file.with_suffix(".tmp").write_text(json.dumps(call))
    worker_file.with_suffix(".tmp").replace(worker_file)
    replicas = (
        int(os.environ["NUM_WORKERS"]) // int(os.environ.get("CONTEXT_PARALLEL_SIZE", "1"))
        if "--shm-namespace" in args
        else 1
    )
    # Readiness requires all replicas to start, catching a launcher that waits after each spawn.
    while len(list(state.glob("worker-*.json"))) < replicas:
        time.sleep(0.01)
    host = args[args.index("--host") + 1]
    print(f"Serving on http://{host}:{port}", flush=True)
    while True:
        signal.pause()


def _exec_wrapper(state: Path, command: str, args: list[str]) -> NoReturn:
    if command == "setsid":
        os.setsid()
        (state / f"group-{os.getpid()}").touch()
        os.execv(args[0], args)  # noqa: S606 - executes the fixture's worker stub.

    assert args[0] == "--interleave=all"
    if os.environ.get("SHELL_TEST_NUMA_DENIED"):
        print("set_mempolicy: Operation not permitted", file=sys.stderr)
        sys.exit(1)
    os.environ["SHELL_TEST_NUMA_POLICY"] = "interleave:all"
    os.execvp(args[1], args[1:])  # noqa: S606 - executes the fixture's worker stub or true probe.


def _cleanup_shm(state: Path, args: list[str]) -> NoReturn:
    assert not any(group_is_alive(int(path.name[6:])) for path in state.glob("group-*"))
    namespace = args[args.index("--shm-namespace") + 1]
    if os.environ.get("SHELL_TEST_CLEANUP_FAIL"):
        print(f"SHM cleanup failed for namespace {namespace}", file=sys.stderr)
        sys.exit(1)
    print(f"SHM cleanup verified for namespace: {namespace}")
    sys.exit(0)


def _serve_nginx(state: Path, call: dict[str, Any]) -> NoReturn:
    while "Frontend ready" not in (state / "launcher.log").read_text():
        time.sleep(0.01)
    if os.environ.get("SHELL_TEST_NGINX_EXIT") == "0":
        sys.exit(0)
    assert call["parent_pid"] > 1
    os.kill(call["parent_pid"], signal.SIGTERM)
    while True:
        signal.pause()


def main() -> None:
    command = Path(sys.argv[1]).name
    args = sys.argv[2:]
    state = Path(os.environ["SHELL_TEST_STATE"])
    call = {
        "command": command,
        "args": args,
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "pgid": os.getpgrp(),
        "cuda": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "nofile": resource.getrlimit(resource.RLIMIT_NOFILE)[0],
        "rank_dir": os.environ.get("OUTPAINT_RANK_DIR"),
        "numa_policy": os.environ.get("SHELL_TEST_NUMA_POLICY"),
    }
    with (state / "calls.jsonl").open("a") as log:
        log.write(json.dumps(call) + "\n")

    if command in {"setsid", "numactl"}:
        _exec_wrapper(state, command, args)

    if command == "python" and args[0] in {"-", "-c"}:
        # Execute actual inline metadata validation and namespace generation.
        os.execv(sys.executable, [sys.executable, *args])  # noqa: S606 - runs local inline Python.

    if command in {"python", "torchrun"} and "serve" in args:
        _serve_worker(state, call)

    if command == "python" and Path(args[0]).name == "cleanup_shm.py":
        _cleanup_shm(state, args)

    if command == "curl" and (failure := os.environ.get("SHELL_TEST_FAILURE")):
        health_port, failed_port, status = {
            "during_wait": (8020, 8021, 1),
            "after_last_health": (8021, 8020, 0),
        }[failure]
        if args[-1].endswith(f":{health_port}/healthz"):
            worker = json.loads((state / f"worker-{failed_port}.json").read_text())
            os.kill(worker["pid"], signal.SIGTERM)
            while group_is_alive(worker["pgid"]):
                time.sleep(0.01)
            sys.exit(status)

    if command == "nginx" and "-t" not in args:
        _serve_nginx(state, call)


if __name__ == "__main__":
    main()
