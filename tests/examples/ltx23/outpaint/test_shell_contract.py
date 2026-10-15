# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import json
import os
import re
import runpy
import shlex
import shutil
import signal
import subprocess  # noqa: S404 - runs local launchers with stub external commands.
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from flextensor import SharedOffloadSession
from tests.examples.ltx23.outpaint.shell_stubs import group_is_alive

REPO_ROOT = Path(__file__).resolve().parents[4]
EXAMPLE_DIR = REPO_ROOT / "examples" / "ltx23" / "outpaint"
BASH = shutil.which("bash") or "/bin/bash"


@dataclass
class Launcher:
    root: Path
    env: dict[str, str]

    def calls(self, command: str) -> list[dict[str, Any]]:
        path = self.root / "state" / "calls.jsonl"
        calls = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        return [call for call in calls if call["command"] == command]

    def run(self, script: str = "nginx_serve.sh", **overrides: str) -> subprocess.CompletedProcess[str]:
        # Start below the requested nofile limit so workers must inherit the launcher's increase.
        command = [BASH, "-c", 'ulimit -Sn 128; exec "$@"', "launcher-test", BASH, str(EXAMPLE_DIR / script)]
        output = self.root / "state" / "launcher.log"
        with (
            output.open("w") as stdout,
            subprocess.Popen(  # noqa: S603 - fixed local script, isolated environment and stub commands.
                command,
                env=self.env | overrides,
                cwd=self.root,
                stdout=stdout,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            ) as process,
        ):
            try:
                _, stderr = process.communicate(timeout=20)
                result = subprocess.CompletedProcess(command, process.returncode, output.read_text(), stderr)
                if script == "nginx_serve.sh":
                    groups = self.root.joinpath("state").glob("group-*")
                    assert not any(group_is_alive(int(path.name[6:])) for path in groups), result
                return result
            finally:
                # Also clean up if a regression prevents the launcher's own cleanup from completing.
                for path in self.root.joinpath("state").glob("group-*"):
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(int(path.name[6:]), signal.SIGKILL)
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)


@pytest.fixture
def launcher(tmp_path: Path) -> Launcher:
    root = tmp_path / "launcher workspace"
    for directory in ("bin", "state", "profile/stage1", "profile/stage2", "profile/text", "gemma", "ltx/.git"):
        (root / directory).mkdir(parents=True)
    for name in ("stage1", "stage2", "text"):
        (root / "profile" / name / "profile.json").write_text("{}")
    for name in ("conditioning video.mp4", "distilled", "upsampler", "lora"):
        (root / name).touch()

    stub = root / "bin" / "stub"
    stub.write_text(
        "#!/bin/sh\nexec "
        + shlex.join([sys.executable, str(Path(__file__).with_name("shell_stubs.py"))])
        + ' "$0" "$@"\n'
    )
    stub.chmod(0o755)
    for command in ("python", "torchrun", "setsid", "numactl", "curl", "nginx", "apt-get", "git", "uv", "rm"):
        (root / "bin" / command).symlink_to(stub)

    return Launcher(
        root,
        {
            "PATH": f"{root / 'bin'}:{os.environ['PATH']}",
            "USER": "test.user",
            "SHELL_TEST_STATE": str(root / "state"),
            "WORKSPACE_DIR": str(root),
            "EX": str(EXAMPLE_DIR),
            "VENV_DIR": str(root),
            "OUTPUT_DIR": str(root / "output"),
            "PROFILE_DIR": str(root / "profile"),
            "LETTERBOXED_VIDEO": str(root / "conditioning video.mp4"),
            "DISTILLED": str(root / "distilled"),
            "UPSAMPLER": str(root / "upsampler"),
            "GEMMA": str(root / "gemma"),
            "LORA": str(root / "lora"),
            "LTX_DIR": str(root / "ltx"),
            "NGINX_CONF": str(root / "nginx.conf"),
            "NUM_WORKERS": "2",
            "OFFLOAD_TEXT": "0",
            "SHM_NOFILE_SOFT_LIMIT": "256",
            "SERVER_READY_ATTEMPTS": "5",
            "WORKER_TERM_GRACE": "1",
            "TMPDIR": str(root),
        },
    )


@pytest.mark.parametrize("script_name", ["nginx_serve.sh", "setup.sh", "single_serve.sh"])
def test_shell_scripts_parse(script_name: str) -> None:
    result = subprocess.run(  # noqa: S603 - every argument is fixed by the test.
        [BASH, "-n", str(EXAMPLE_DIR / script_name)], capture_output=True, check=False, text=True
    )
    assert result.returncode == 0, result.stderr


def test_nginx_generated_namespaces_differ_when_user_port_and_pid_repeat(launcher: Launcher) -> None:
    # Load definitions without starting workers. Repeating in one shell keeps $$
    # unchanged, reproducing the collision between successive PID-1 containers.
    definitions = (EXAMPLE_DIR / "nginx_serve.sh").read_text().split("\nvalidate_num_frames\n", 1)[0]
    result = subprocess.run(  # noqa: S603 - runs local launcher definitions with fixture commands.
        [
            BASH,
            "-c",
            definitions
            + r"""
unset USER
for launch in 1 2; do
  SHM_NAMESPACE=""
  configure_shm
  printf 'namespace=%s\n' "$SHM_NAMESPACE"
done
""",
        ],
        env=launcher.env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    namespaces = [
        line.removeprefix("namespace=") for line in result.stdout.splitlines() if line.startswith("namespace=")
    ]
    assert len(namespaces) == len(set(namespaces)) == 2
    assert namespaces[0].rsplit("_", 1)[0] == namespaces[1].rsplit("_", 1)[0]
    assert all(re.fullmatch(r"ltx23_outpaint_user_8080_\d+_[0-9a-f]{32}", name) for name in namespaces)
    assert all(len(name) <= 96 for name in namespaces)


@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("namespace", ["", "explicit_run"])
@pytest.mark.parametrize("numa_interleave", [None, "1", "0"])
def test_nginx_launcher_starts_all_replicas_with_shared_options(
    launcher: Launcher, cp_size: int, namespace: str, numa_interleave: str | None
) -> None:
    if cp_size > 1:
        (launcher.root / "profile" / "parallelism.json").write_text(json.dumps({"context_parallel_size": cp_size}))
    options = {"SHM_NAMESPACE": namespace, "SHM_WAIT_TIMEOUT": "17.5"} if namespace else {}
    if numa_interleave is not None:
        options["NUMA_INTERLEAVE"] = numa_interleave
    if numa_interleave == "0":
        # Disabling NUMA must work even when setting a policy is denied.
        options["SHELL_TEST_NUMA_DENIED"] = "1"
        options["SHELL_TEST_NUMA_POLICY"] = "inherited"
    result = launcher.run(CONTEXT_PARALLEL_SIZE=str(cp_size), NUM_WORKERS=str(2 * cp_size), **options)

    # The nginx stub requests shutdown once the frontend is published.
    assert result.returncode == 130, result
    assert "Frontend ready" in result.stdout
    assert "Worker processes stopped." in result.stdout
    assert "SHM cleanup verified" in result.stdout
    numa_calls = launcher.calls("numactl")
    if numa_interleave == "0":
        assert numa_calls == []
    else:
        assert len(numa_calls) == 3  # One probe and two replica launches.
        assert numa_calls[0]["args"] == ["--interleave=all", "true"]
    # Check policy inheritance in both the agents and their detached ranks.
    for call in launcher.calls("python") + launcher.calls("torchrun"):
        if "serve" in call["args"]:
            assert call["numa_policy"] == ("inherited" if numa_interleave == "0" else "interleave:all")
    workers = launcher.calls("python" if cp_size == 1 else "torchrun")
    workers = [worker for worker in workers if "serve" in worker["args"]]
    assert len(workers) == 2
    namespaces = set()
    for index, worker in enumerate(sorted(workers, key=lambda worker: worker["cuda"])):
        args = worker["args"]
        namespaces.add(args[args.index("--shm-namespace") + 1])
        assert args[args.index("--shm-wait-timeout") + 1] == ("17.5" if namespace else "3600")
        assert args[args.index("--context-parallel-size") + 1] == str(cp_size)
        assert worker["cuda"] == ",".join(str(gpu) for gpu in range(index * cp_size, (index + 1) * cp_size))
        assert worker["nofile"] == 256
        assert worker["pgid"] == worker["pid"]
        assert "--shm-enabled" not in args
        if cp_size > 1:
            assert f"--nproc-per-node={cp_size}" in args
            assert "--max-restarts=0" in args
    namespace_pattern = re.escape(namespace) or (
        rf"ltx23_outpaint_test_user_8080_{workers[0]['parent_pid']}_[0-9a-f]{{32}}"
    )
    assert len(namespaces) == 1
    run_namespace = namespaces.pop()
    assert re.fullmatch(namespace_pattern, run_namespace)
    cleanup_calls = [call for call in launcher.calls("python") if "cleanup_shm.py" in call["args"][0]]
    assert len(cleanup_calls) == 1
    assert cleanup_calls[0]["args"] == [str(EXAMPLE_DIR / "cleanup_shm.py"), "--shm-namespace", run_namespace]
    assert [call["args"][-1] for call in launcher.calls("curl")] == [
        "http://127.0.0.1:8020/healthz",
        "http://127.0.0.1:8021/healthz",
    ]
    config = (launcher.root / "nginx.conf").read_text()
    assert "server 127.0.0.1:8020 " in config
    assert "server 127.0.0.1:8021 " in config


@pytest.mark.parametrize(
    "options, error",
    [
        ({"NUMA_INTERLEAVE": "2"}, "NUMA_INTERLEAVE must be 0 or 1"),
        ({"SHM_NAMESPACE": "bad/name"}, "SHM_NAMESPACE must contain"),
        ({"SHM_WAIT_TIMEOUT": "0"}, "SHM_WAIT_TIMEOUT must be a positive number"),
        ({"CONTEXT_PARALLEL_SIZE": "2", "TORCHRUN_MAX_RESTARTS": "1"}, "must be 0 when CONTEXT_PARALLEL_SIZE>1"),
        ({"TORCHRUN_MAX_RESTARTS": "08"}, "non-negative integer without leading zeros"),
    ],
)
def test_nginx_launcher_rejects_invalid_options_before_starting_workers(
    launcher: Launcher, options: dict[str, str], error: str
) -> None:
    result = launcher.run(**options)
    assert result.returncode == 1, result
    assert error in result.stderr
    assert launcher.calls("setsid") == []
    assert launcher.calls("nginx") == []


def test_nginx_launcher_requires_numactl_before_starting_workers(launcher: Launcher) -> None:
    empty_bin = launcher.root / "empty-bin"
    empty_bin.mkdir()
    result = launcher.run(PATH=str(empty_bin))
    assert result.returncode == 1, result
    assert "requires numactl" in result.stderr
    assert "NUMA_INTERLEAVE=0" in result.stderr
    assert launcher.calls("setsid") == []
    assert launcher.calls("nginx") == []


def test_nginx_launcher_rejects_denied_numa_policy_before_starting_workers(launcher: Launcher) -> None:
    result = launcher.run(SHELL_TEST_NUMA_DENIED="1")
    assert result.returncode == 1, result
    assert "Operation not permitted" in result.stderr
    assert "--cap-add SYS_NICE" in result.stderr
    assert "NUMA_INTERLEAVE=0" in result.stderr
    assert launcher.calls("setsid") == []
    assert launcher.calls("nginx") == []


@pytest.mark.parametrize(
    "missing", ["stage1/profile.json", "stage2/profile.json", "text/profile.json", "parallelism.json"]
)
def test_nginx_launcher_requires_profile_files_before_starting_workers(launcher: Launcher, missing: str) -> None:
    metadata = launcher.root / "profile" / "parallelism.json"
    metadata.write_text('{"context_parallel_size": 2}')
    (launcher.root / "profile" / missing).unlink()
    result = launcher.run(CONTEXT_PARALLEL_SIZE="2", OFFLOAD_TEXT="1")
    assert result.returncode == 1, result
    assert "Missing " in result.stderr
    assert missing in result.stderr
    assert launcher.calls("setsid") == []


@pytest.mark.parametrize(
    "cp_size, metadata, error",
    [
        (4, '{"context_parallel_size": 2}', "was created for CP2, but this launcher requested CP4"),
        (1, '{"context_parallel_size": 2}', "was created for CP2, but this launcher requested CP1"),
        (4, "not JSON", "Invalid context-parallel profile metadata"),
        (4, "{}", "Invalid context-parallel profile metadata"),
        (2, '{"context_parallel_size": 2.9}', "context_parallel_size must be an integer"),
        (1, None, "Missing context-parallel profile metadata"),
    ],
)
def test_nginx_launcher_rejects_invalid_parallelism_metadata(
    launcher: Launcher, cp_size: int, metadata: str | None, error: str
) -> None:
    path = launcher.root / "profile" / "parallelism.json"
    if metadata is None:
        path.mkdir()
    else:
        path.write_text(metadata)
    result = launcher.run(CONTEXT_PARALLEL_SIZE=str(cp_size))
    assert result.returncode == 1, result
    assert error in result.stderr
    assert launcher.calls("setsid") == []


@pytest.mark.parametrize("failure, failed_port", [("during_wait", 8021), ("after_last_health", 8020)])
def test_nginx_launcher_aborts_if_a_replica_exits_before_publishing_frontend(
    launcher: Launcher, failure: str, failed_port: int
) -> None:
    result = launcher.run(SHELL_TEST_FAILURE=failure)
    assert result.returncode == 1, result
    if failure == "during_wait":
        assert "Worker startup aborted while waiting for port 8020" in result.stderr
    assert f"port {failed_port} exited during startup" in result.stderr
    assert launcher.calls("nginx") == []
    assert not (launcher.root / "nginx.conf").exists()
    assert "SHM cleanup verified" in result.stdout


def test_nginx_cleanup_kills_worker_children_that_ignore_term(launcher: Launcher) -> None:
    result = launcher.run(SHELL_TEST_STUBBORN_CHILD="1")
    assert result.returncode == 130, result
    children = list((launcher.root / "state").glob("child-*"))
    assert len(children) == 2
    assert all(child.with_name(child.name.replace("child-", "term-")).exists() for child in children)
    assert "SHM cleanup verified" in result.stdout
    assert "WARNING: SIGKILL required" in result.stderr
    assert "SHM namespace:" in result.stderr
    assert not any("/dev/shm" in arg for call in launcher.calls("rm") for arg in call["args"])  # noqa: S108


@pytest.mark.parametrize("agent_exit", ["0", "1"])
def test_nginx_cleanup_kills_detached_ranks_that_ignore_term(launcher: Launcher, agent_exit: str) -> None:
    (launcher.root / "profile" / "parallelism.json").write_text('{"context_parallel_size": 2}')
    result = launcher.run(CONTEXT_PARALLEL_SIZE="2", SHELL_TEST_STUBBORN_CHILD="1", SHELL_TEST_AGENT_EXIT=agent_exit)
    assert result.returncode == 130, result
    ranks = list((launcher.root / "state").glob("rank-*"))
    assert len(ranks) == 2
    agent_pgid = launcher.calls("torchrun")[0]["pgid"]
    for rank in ranks:
        pid = int(rank.name.removeprefix("rank-"))
        assert int(rank.read_text()) == pid != agent_pgid
        assert rank.with_name(f"term-{pid}").exists()
    assert "SHM cleanup verified" in result.stdout
    assert "WARNING: SIGKILL required" in result.stderr


def test_nginx_cleanup_kills_detached_ranks_after_agent_startup_failure(launcher: Launcher) -> None:
    (launcher.root / "profile" / "parallelism.json").write_text('{"context_parallel_size": 2}')
    result = launcher.run(
        CONTEXT_PARALLEL_SIZE="2",
        NUM_WORKERS="4",
        SHELL_TEST_STUBBORN_CHILD="1",
        SHELL_TEST_AGENT_EXIT="1",
        SHELL_TEST_FAILURE="after_last_health",
    )
    assert result.returncode == 1, result
    assert "exited during startup" in result.stderr
    assert len(list((launcher.root / "state").glob("rank-*"))) == 4
    assert launcher.calls("nginx") == []
    assert "SHM cleanup verified" in result.stdout


@pytest.mark.parametrize(
    "options, expected_status",
    [({}, 130), ({"SHELL_TEST_FAILURE": "during_wait"}, 1), ({"SHELL_TEST_NGINX_EXIT": "0"}, 1)],
)
def test_nginx_reports_cleanup_failure(launcher: Launcher, options: dict[str, str], expected_status: int) -> None:
    result = launcher.run(SHELL_TEST_CLEANUP_FAIL="1", SHM_NAMESPACE="cleanup_failure", **options)
    assert result.returncode == expected_status, result
    assert "Worker processes stopped." in result.stdout
    assert "SHM cleanup verified" not in result.stdout
    assert "SHM cleanup incomplete for namespace: cleanup_failure" in result.stderr


def test_nginx_clean_exit_reports_reclamation(launcher: Launcher) -> None:
    result = launcher.run(SHELL_TEST_NGINX_EXIT="0")
    assert result.returncode == 0, result
    assert "SHM cleanup verified" in result.stdout


def test_nginx_skips_shm_cleanup_if_processes_still_appear_alive(launcher: Launcher) -> None:
    # Simulate an uninterruptible process in the launcher's ps observations.
    # The fixture checks actual process exit with its own unmodified PATH.
    ps = launcher.root / "bin" / "ps"
    ps.write_text("#!/bin/sh\nprintf 'S\\n'\n")
    ps.chmod(0o755)
    result = launcher.run(SHELL_TEST_NGINX_EXIT="0", SHM_NAMESPACE="still_alive")
    assert result.returncode == 1, result
    assert "Processes remain alive; skipping SHM cleanup for namespace: still_alive" in result.stderr
    assert not any("cleanup_shm.py" in call["args"][0] for call in launcher.calls("python"))
    assert "Worker processes stopped." not in result.stdout


@pytest.mark.parametrize("fail", [False, True])
def test_cleanup_command_uses_all_managers_without_importing_serving(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], fail: bool
) -> None:
    calls = []

    def cleanup(namespace, **kwargs):
        calls.append((namespace, kwargs))
        if fail:
            raise TimeoutError("participants are live")

    monkeypatch.setattr(SharedOffloadSession, "cleanup_namespace", cleanup)
    monkeypatch.setattr(sys, "argv", ["cleanup_shm.py", "--shm-namespace", "stopped_run"])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(EXAMPLE_DIR / "cleanup_shm.py"), run_name="__main__")
    assert result.value.code == int(fail)
    assert calls == [("stopped_run", {"manager_names": ("ltx_stage1", "ltx_stage2", "ltx_text"), "timeout": 60})]
    output = capsys.readouterr()
    if fail:
        assert "SHM cleanup failed for namespace stopped_run: participants are live" in output.err
        assert "verified" not in output.out
    else:
        assert "SHM cleanup verified for namespace: stopped_run" in output.out


def test_nginx_rank_wrapper_refuses_to_start_after_cleanup(launcher: Launcher) -> None:
    (launcher.root / "profile" / "parallelism.json").write_text('{"context_parallel_size": 2}')
    launcher.run(CONTEXT_PARALLEL_SIZE="2")
    agent = launcher.calls("torchrun")[0]
    wrapper_index = agent["args"].index("--no-python") + 1
    wrapper = agent["args"][wrapper_index : wrapper_index + 4]
    started = launcher.root / "late-rank-started"
    result = subprocess.run(  # noqa: S603 - reruns the launcher's wrapper with an isolated marker command.
        [
            *wrapper,
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; Path(sys.argv[1]).touch()",
            str(started),
        ],
        env=launcher.env | {"OUTPAINT_RANK_DIR": agent["rank_dir"]},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert not started.exists()


def test_setup_installs_local_flextensor_checkout(launcher: Launcher) -> None:
    del launcher.env["EX"]
    result = launcher.run("setup.sh")
    assert result.returncode == 0, result
    assert [call["args"] for call in launcher.calls("apt-get")] == [
        ["update"],
        ["install", "-y", "nginx", "numactl"],
    ]
    assert [call["args"] for call in launcher.calls("uv")] == [
        ["sync"],
        ["pip", "install", "--python", str(launcher.root / "bin" / "python"), "-e", str(REPO_ROOT)],
    ]
    assert launcher.calls("git")[-1]["args"] == [
        "-C",
        str(launcher.root / "ltx"),
        "checkout",
        "--detach",
        "4f8905737aac86a554637cac86c178877a39c744",
    ]
    assert "profile" in launcher.calls("python")[-1]["args"]


@pytest.mark.parametrize("allocator", [None, "", "expandable_segments:False", "max_split_size_mb:128"])
def test_single_worker_launcher_remains_non_shm(launcher: Launcher, allocator: str | None) -> None:
    overrides = {} if allocator is None else {"PYTORCH_CUDA_ALLOC_CONF": allocator}
    result = launcher.run("single_serve.sh", **overrides)
    assert result.returncode == 0, result
    workers = launcher.calls("python")
    assert len(workers) == 1
    assert "serve" in workers[0]["args"]
    assert workers[0]["cuda_alloc_conf"] == (allocator or "expandable_segments:True")
    assert not any(arg.startswith("--shm-") for arg in workers[0]["args"])
