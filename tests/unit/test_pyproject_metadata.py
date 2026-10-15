# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pathlib
import re
import tomllib
from collections.abc import Iterable


def _dependency_names(dependencies: Iterable[str | dict[str, str]]) -> set[str]:
    names = set()
    for dependency in dependencies:
        if isinstance(dependency, str):
            names.add(re.split(r"[<>=~!;\[]", dependency, maxsplit=1)[0])
        elif "include-group" not in dependency:
            names.add(dependency["name"])
    return names


def test_development_dependencies_are_dependency_groups() -> None:
    pyproject_path = pathlib.Path(__file__).parents[2] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text())

    optional_dependencies = pyproject["project"].get("optional-dependencies", {})
    dependency_groups = pyproject["dependency-groups"]

    assert "required-version" in pyproject["tool"]["uv"]

    assert "test" not in optional_dependencies
    assert "docs" not in optional_dependencies
    assert "dev" not in optional_dependencies
    assert "all" not in optional_dependencies

    assert {"pytest", "pytest-cov", "mypy"}.issubset(_dependency_names(dependency_groups["test"]))
    assert {"mkdocs", "mkdocs-material", "mike", "mkdocstrings"}.issubset(_dependency_names(dependency_groups["docs"]))
    assert {"include-group": "test"} in dependency_groups["dev"]
    assert {"build", "pre-commit"}.issubset(_dependency_names(dependency_groups["dev"]))
    assert {"include-group": "test"} in dependency_groups["all"]
    assert {"include-group": "dev"} in dependency_groups["all"]
    assert {"include-group": "docs"} in dependency_groups["all"]


def test_mutmut_targets_cpu_safe_core_logic() -> None:
    """Ensure mutation testing targets existing CPU-safe core paths."""
    repo_root = pathlib.Path(__file__).parents[2]
    pyproject_path = repo_root / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text())

    assert "mutmut>=3.6.0" in pyproject["dependency-groups"]["test"]
    mutmut_config = pyproject["tool"]["mutmut"]
    assert mutmut_config["source_paths"] == ["src"]
    assert mutmut_config["only_mutate"] == [
        "src/flextensor/strategy/*",
        "src/flextensor/gpu_budget.py",
        "src/flextensor/memory_block_planner.py",
    ]
    assert mutmut_config["pytest_add_cli_args_test_selection"] == [
        "tests/unit/test_strategy.py",
        "tests/unit/test_strategy_utils.py",
        "tests/unit/test_adaptive_strategy.py",
        "tests/unit/test_global_offload_strategy.py",
        "tests/unit/test_global_tensor_selection_strategy.py",
        "tests/unit/test_memory_block_planner.py",
        "tests/unit/test_resolve_gpu_budget.py",
        "tests/unit/test_unmapped_gpu_budget.py",
        "tests/unit/test_resolve_gpu_mem_bytes.py",
    ]

    for source_path in mutmut_config["source_paths"]:
        assert (repo_root / source_path).is_dir(), f"Missing mutmut source path: {source_path}"
    for target in mutmut_config["only_mutate"]:
        assert any(path.is_file() for path in repo_root.glob(target)), f"Missing mutmut target: {target}"
    for test_path in mutmut_config["pytest_add_cli_args_test_selection"]:
        assert (repo_root / test_path).is_file(), f"Missing selected test file: {test_path}"
