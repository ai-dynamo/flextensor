# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OffloadConfig and environment variable loading.

This test suite validates:
1. OffloadConfig class behavior (defaults, validation, properties)
2. Environment variable loading with type conversion
3. File-based configuration loading (INI, JSON, YAML)
4. Edge cases (case sensitivity, invalid values, etc.)
"""

import logging
import os
import warnings
from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import ValidationError

import flextensor.config as config_module
from flextensor.config import (
    BLOCK_TRANSFER_MODES,
    OFFLOAD_TRANSFER_MODES,
    OffloadConfig,
    _get_field_types,
    _parse_bool,
    _parse_none,
    load_config,
    load_config_from_env,
    load_config_from_file,
)
from flextensor.strategy import AdaptiveStrategy, GreedyStrategy, KnapsackStrategy, NthLayerStrategy
from flextensor.utils import config_field_was_set


def test_transfer_mode_sets_define_complete_and_block_contracts() -> None:
    assert (
        frozenset({
            "allocation_block_transfer",
            "raw_block_transfer",
        })
        == BLOCK_TRANSFER_MODES
    )
    assert (
        frozenset({
            "strategy",
            "allocation_block_transfer",
            "raw_block_transfer",
        })
        == OFFLOAD_TRANSFER_MODES
    )
    assert BLOCK_TRANSFER_MODES < OFFLOAD_TRANSFER_MODES


def test_resolve_load_strategy_returns_explicit_strategy_by_identity() -> None:
    strategy = NthLayerStrategy(nth_layer=2)

    assert config_module.resolve_load_strategy(OffloadConfig(load_strategy=strategy)) is strategy


def test_resolve_load_strategy_builds_fresh_configured_defaults() -> None:
    config = OffloadConfig(
        transfer_budget_scale=0.75,
        transfer_mode="raw_block_transfer",
        num_blocks=7,
        min_blocks=3,
    )

    first = config_module.resolve_load_strategy(config)
    second = config_module.resolve_load_strategy(config)

    assert isinstance(first, AdaptiveStrategy)
    assert first is not second
    assert first.scale == 0.75
    assert first.loader_type == "raw_block_transfer"
    assert first.n_blocks == 7
    assert first.min_blocks == 3


@pytest.fixture(autouse=True)
def isolate_config_environment():
    original_env = os.environ.copy()
    for name in tuple(os.environ):
        if name.startswith(("FT_", "CUSTOM_")) or name == "MY_CONFIG_FILE":
            os.environ.pop(name)
    yield
    os.environ.clear()
    os.environ.update(original_env)


class TestOffloadConfig:
    """Test OffloadConfig class behavior."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = OffloadConfig()
        assert config.gpu_device == 0
        assert config.pinned_memory is True
        assert config.pinned_memory_mode == "torch"
        assert config.shm_enabled is False
        assert config.discovery_iters == 1
        assert config.profiling_iters == 10
        assert config.transfer_budget_scale == 1.0
        assert config.transfer_mode == "allocation_block_transfer"
        assert config.num_blocks == 4
        assert config.profile_storage_dir is None
        assert config.profile_read_only is False
        assert config.load_strategy is None
        assert config.min_blocks == 4
        assert config.max_gpu_mem_fraction is None
        assert config.external_compile is False

    def test_max_gpu_mem_fraction_default(self):
        """Default max_gpu_mem_fraction is None for latency-first mode."""
        config = OffloadConfig()
        assert config.max_gpu_mem_fraction is None

    def test_max_gpu_mem_fraction_none_latency_mode(self):
        """Setting max_gpu_mem_fraction=None enables latency mode."""
        config = OffloadConfig(max_gpu_mem_fraction=None)
        assert config.max_gpu_mem_fraction is None

    def test_max_gpu_mem_fraction_custom(self):
        """Custom fraction is accepted."""
        config = OffloadConfig(max_gpu_mem_fraction=0.75)
        assert config.max_gpu_mem_fraction == 0.75

    def test_max_gpu_mem_fraction_boundary_values(self):
        """Boundary values: just above 0 and exactly 1.0 are valid."""
        config_low = OffloadConfig(max_gpu_mem_fraction=0.01)
        config_high = OffloadConfig(max_gpu_mem_fraction=1.0)
        assert config_low.max_gpu_mem_fraction == 0.01
        assert config_high.max_gpu_mem_fraction == 1.0

    def test_max_gpu_mem_fraction_zero_invalid(self):
        """max_gpu_mem_fraction=0.0 is rejected (gt=0.0 constraint)."""
        with pytest.raises(ValidationError):
            OffloadConfig(max_gpu_mem_fraction=0.0)

    def test_max_gpu_mem_fraction_above_one_invalid(self):
        """max_gpu_mem_fraction > 1.0 is rejected."""
        with pytest.raises(ValidationError):
            OffloadConfig(max_gpu_mem_fraction=1.1)

    def test_pinned_memory_mode_default(self):
        """Default pinned_memory_mode is 'torch'."""
        assert OffloadConfig().pinned_memory_mode == "torch"

    def test_pinned_memory_mode_host_register_accepted(self):
        """'host_register' is a valid pinned_memory_mode."""
        config = OffloadConfig(pinned_memory_mode="host_register")
        assert config.pinned_memory_mode == "host_register"

    def test_pinned_memory_mode_invalid_rejected(self):
        """Unknown pinned_memory_mode values are rejected with a helpful message."""
        with pytest.raises(ValidationError, match="pinned_memory_mode"):
            OffloadConfig(pinned_memory_mode="cuda_alloc")

    def test_custom_values(self):
        """Test setting custom values."""
        config = OffloadConfig(
            gpu_device=2,
            pinned_memory=False,
            discovery_iters=5,
            profiling_iters=20,
            transfer_budget_scale=2.0,
            transfer_mode="custom_mode",
            num_blocks=8,
        )
        assert config.gpu_device == 2
        assert config.pinned_memory is False
        assert config.discovery_iters == 5
        assert config.profiling_iters == 20
        assert config.transfer_budget_scale == 2.0
        assert config.transfer_mode == "custom_mode"
        assert config.num_blocks == 8

    def test_gpu_device_validation_negative(self):
        """Test that negative gpu_device raises validation error."""
        with pytest.raises(ValidationError):
            OffloadConfig(gpu_device=-1)

    def test_discovery_iters_validation_negative(self):
        """Test that negative discovery_iters raises validation error."""
        with pytest.raises(ValidationError):
            OffloadConfig(discovery_iters=-1)

    def test_profiling_iters_validation_negative(self):
        """Test that negative profiling_iters raises validation error."""
        with pytest.raises(ValidationError):
            OffloadConfig(profiling_iters=-1)

    def test_transfer_budget_scale_validation_zero(self):
        """Test that zero transfer_budget_scale raises validation error."""
        with pytest.raises(ValidationError):
            OffloadConfig(transfer_budget_scale=0.0)

    def test_transfer_budget_scale_validation_negative(self):
        """Test that negative transfer_budget_scale raises validation error."""
        with pytest.raises(ValidationError):
            OffloadConfig(transfer_budget_scale=-1.0)

    def test_num_blocks_validation_one(self):
        """Test that num_blocks=1 raises validation error (ge=2 constraint)."""
        with pytest.raises(ValidationError):
            OffloadConfig(num_blocks=1)

    def test_num_blocks_validation_zero(self):
        """Test that zero num_blocks raises validation error."""
        with pytest.raises(ValidationError):
            OffloadConfig(num_blocks=0)

    def test_num_blocks_validation_negative(self):
        """Test that negative num_blocks raises validation error."""
        with pytest.raises(ValidationError):
            OffloadConfig(num_blocks=-1)

    def test_min_blocks_custom(self):
        """Test setting custom min_blocks value."""
        config = OffloadConfig(min_blocks=3)
        assert config.min_blocks == 3

    def test_min_blocks_validation_below_minimum(self):
        """Test that min_blocks < 2 raises validation error."""
        with pytest.raises(ValidationError):
            OffloadConfig(min_blocks=1)

    def test_min_blocks_validation_at_minimum(self):
        """Test that min_blocks=2 is accepted."""
        config = OffloadConfig(min_blocks=2)
        assert config.min_blocks == 2

    def test_num_blocks_less_than_min_blocks_raises(self):
        """Test that num_blocks < min_blocks raises validation error."""
        with pytest.raises(ValidationError, match=r"num_blocks.*must be >= min_blocks"):
            OffloadConfig(num_blocks=2, min_blocks=4)

    def test_num_blocks_equal_min_blocks_accepted(self):
        """Test that num_blocks == min_blocks is accepted."""
        config = OffloadConfig(num_blocks=3, min_blocks=3)
        assert config.num_blocks == 3
        assert config.min_blocks == 3

    def test_profile_mode_default(self):
        """Default profile_mode is 'view' to match TensorManager."""
        assert OffloadConfig().profile_mode == "view"

    def test_profile_mode_view_with_block_loader(self):
        """profile_mode='view' is accepted with the default block-transfer loader."""
        config = OffloadConfig(profile_mode="view")
        assert config.profile_mode == "view"

    def test_profile_mode_view_accepts_strategy_transfer_mode(self):
        """profile_mode='view' is accepted with transfer_mode='strategy'.

        The view-mode profile controller is self-contained and torn down before
        the inference loader is built, so profile mode and inference transfer
        mode are independent knobs.
        """
        config = OffloadConfig(profile_mode="view", transfer_mode="strategy")
        assert config.profile_mode == "view"
        assert config.transfer_mode == "strategy"

    def test_profile_mode_torch_function_with_strategy_ok(self):
        """profile_mode='torch_function' is accepted with strategy transfer mode."""
        config = OffloadConfig(profile_mode="torch_function", transfer_mode="strategy")
        assert config.profile_mode == "torch_function"

    def test_profile_mode_torch_function_rejects_block_loader(self):
        """profile_mode='torch_function' rejects block transfer loaders."""
        with pytest.raises(ValidationError, match="profile_mode='torch_function'"):
            OffloadConfig(profile_mode="torch_function", transfer_mode="allocation_block_transfer")

    def test_external_compile_rejects_strategy_transfer_mode(self):
        """external_compile requires a PreallocatedLoader-backed transfer mode."""
        with pytest.raises(ValidationError, match="external_compile=True requires a block transfer_mode"):
            OffloadConfig(external_compile=True, transfer_mode="strategy")

    def test_offload_timing_rejects_strategy_transfer_mode(self):
        """offload_timing needs PreallocatedLoader enter/exit hooks."""
        for mode in ("eager", "cuda_graph"):
            with pytest.raises(ValidationError, match=r"offload_timing=.*requires a block transfer_mode"):
                OffloadConfig(offload_timing=mode, transfer_mode="strategy")

    def test_offload_timing_accepts_block_transfer_modes(self):
        for transfer_mode in ("allocation_block_transfer", "raw_block_transfer"):
            config = OffloadConfig(offload_timing="eager", transfer_mode=transfer_mode)
            assert config.offload_timing == "eager"
            assert config.transfer_mode == transfer_mode

    def test_external_compile_accepts_block_transfer_modes(self):
        """external_compile is valid with either block transfer_mode."""
        for transfer_mode in ("allocation_block_transfer", "raw_block_transfer"):
            config = OffloadConfig(external_compile=True, transfer_mode=transfer_mode)
            assert config.external_compile is True
            assert config.transfer_mode == transfer_mode

    def test_profile_mode_invalid_rejected(self):
        """Unknown profile_mode values are rejected by the Literal validator."""
        with pytest.raises(ValidationError, match="profile_mode"):
            OffloadConfig(profile_mode="bogus")

    def test_knapsack_strategy_assignment(self):
        """Test assigning KnapsackStrategy to load_strategy."""
        strategy = KnapsackStrategy(scale=1.5, cyclic=True, group_size=2)
        config = OffloadConfig(load_strategy=strategy)
        assert config.load_strategy is strategy
        assert isinstance(config.load_strategy, KnapsackStrategy)

    def test_greedy_strategy_assignment(self):
        """Test assigning GreedyStrategy to load_strategy."""
        strategy = GreedyStrategy(threshold_mb=0.2)
        config = OffloadConfig(load_strategy=strategy)
        assert config.load_strategy is strategy
        assert isinstance(config.load_strategy, GreedyStrategy)

    def test_nth_layer_strategy_assignment(self):
        """Test assigning NthLayerStrategy to load_strategy."""
        strategy = NthLayerStrategy(nth_layer=2, threshold_mb=0.2)
        config = OffloadConfig(load_strategy=strategy)
        assert config.load_strategy is strategy
        assert isinstance(config.load_strategy, NthLayerStrategy)

    def test_edge_values_discovery_iters_zero(self):
        """Test discovery_iters at boundary value 0."""
        config = OffloadConfig(discovery_iters=0)
        assert config.discovery_iters == 0

    def test_edge_values_profiling_iters_zero(self):
        """Test profiling_iters at boundary value 0."""
        # profiling_iters=0 round-trips under either skip_discovery value; no
        # validator relates the two. OffloadManager.iters_before_inference
        # floors the profile budget at 1 so the documented drive loop can still
        # reach INFERENCE — see
        # tests/unit/test_skip_discovery.py::TestItersBeforeInferenceFloor.
        config = OffloadConfig(profiling_iters=0)
        assert config.profiling_iters == 0

    def test_include_patterns_default(self):
        """Test include_patterns default value is ['*']."""
        config = OffloadConfig()
        assert config.include_patterns == ["*"]

    def test_include_patterns_custom(self):
        """Test setting custom include_patterns."""
        config = OffloadConfig(include_patterns=["layers.*", "head"])
        assert config.include_patterns == ["layers.*", "head"]

    def test_include_patterns_empty_list_accepted(self):
        """Empty include_patterns is accepted for manual offload_block() usage."""
        config = OffloadConfig(include_patterns=[])
        assert config.include_patterns == []

    def test_exclude_patterns_default(self):
        """Test exclude_patterns default value is []."""
        config = OffloadConfig()
        assert config.exclude_patterns == []

    def test_exclude_patterns_custom(self):
        """Test setting custom exclude_patterns."""
        config = OffloadConfig(exclude_patterns=["lm_head", "*.norm"])
        assert config.exclude_patterns == ["lm_head", "*.norm"]

    def test_exclude_patterns_empty_list(self):
        """Test that empty exclude_patterns list is allowed."""
        config = OffloadConfig(exclude_patterns=[])
        assert config.exclude_patterns == []

    def test_include_patterns_accepts_all_three_forms(self):
        # Valid: bare glob, name-prefixed, class-prefixed.
        config = OffloadConfig(include_patterns=["bare", "name:embed", "class:MoELayer"])
        assert config.include_patterns == ["bare", "name:embed", "class:MoELayer"]

    def test_include_patterns_rejects_bare_class_prefix(self):
        # Empty body is almost always a typo for ``class:<Glob>``.
        with pytest.raises(ValidationError, match="empty body"):
            OffloadConfig(include_patterns=["class:"])

    def test_include_patterns_rejects_bare_name_prefix(self):
        with pytest.raises(ValidationError, match="empty body"):
            OffloadConfig(include_patterns=["name:"])

    def test_exclude_patterns_rejects_bare_class_prefix(self):
        # Same validator applies symmetrically to exclude_patterns.
        with pytest.raises(ValidationError, match="empty body"):
            OffloadConfig(exclude_patterns=["class:"])

    @pytest.mark.parametrize("entry", ["class: ", "class:\t", "class:  \n", "name: ", "name:\t"])
    def test_pattern_validator_rejects_whitespace_only_body(self, entry):
        # Trailing-whitespace bodies are typos; bare-prefix check must not
        # rely on exact equality with ``"class:"`` / ``"name:"``.
        with pytest.raises(ValidationError, match="empty body"):
            OffloadConfig(include_patterns=[entry])

    def test_include_patterns_rejects_non_string_entries(self):
        # Pydantic's default ``list[str]`` coercion would silently turn 42
        # into "42"; the validator catches it before coercion.
        with pytest.raises(ValidationError, match="must be strings"):
            OffloadConfig(include_patterns=[42])

    def test_exclude_patterns_rejects_non_string_entries(self):
        with pytest.raises(ValidationError, match="must be strings"):
            OffloadConfig(exclude_patterns=[None])

    def test_pattern_validator_error_identifies_index(self):
        # Surfacing the index helps users locate the bad entry in long lists.
        with pytest.raises(ValidationError, match="index 1"):
            OffloadConfig(include_patterns=["layers.*", "class:", "head"])

    def test_include_patterns_valid_tuple_accepted(self):
        # Tuples are coerced to ``list[str]`` by pydantic; the validator
        # must let valid sequence-like inputs through unchanged.
        config = OffloadConfig(include_patterns=("layers.*", "head"))
        assert config.include_patterns == ["layers.*", "head"]

    def test_include_patterns_tuple_with_empty_body_rejected(self):
        # Regression: ``isinstance(v, list)`` previously let tuples bypass the
        # body-emptiness check, so ``OffloadConfig(include_patterns=("class:",))``
        # constructed successfully and only blew up at the first
        # ``partition_patterns(...)`` call inside ``offload()``.  The validator
        # now treats any non-str/bytes ``Sequence`` symmetrically with ``list``.
        with pytest.raises(ValidationError, match="empty body"):
            OffloadConfig(include_patterns=("class:",))

    def test_exclude_patterns_tuple_with_non_string_entry_rejected(self):
        # Symmetric coverage for the non-string branch on tuple inputs.
        with pytest.raises(ValidationError, match="must be strings"):
            OffloadConfig(exclude_patterns=("layers.*", 42))

    @pytest.mark.parametrize(
        "raw,normalized",
        [
            ("class: Linear", "class:Linear"),
            ("class:Linear ", "class:Linear"),
            ("class: Linear ", "class:Linear"),
            ("  class: Linear  ", "class:Linear"),
            (" class:Linear", "class:Linear"),
            ("name: embed", "name:embed"),
            ("name:embed ", "name:embed"),
            ("  layers.0  ", "layers.0"),
        ],
    )
    def test_pattern_validator_normalizes_whitespace(self, raw, normalized):
        # Regression: pre-fix, ``OffloadConfig(include_patterns=["class: Linear"])`` constructed
        # successfully and stored the body verbatim as ``" Linear"``, which never matched any
        # ``cls.__name__``.  The validator now strips leading/trailing whitespace from both the
        # entry and (for prefix patterns) the body before storage.
        config = OffloadConfig(include_patterns=[raw])
        assert config.include_patterns == [normalized]

    @pytest.mark.parametrize("entry", ["clas:Linear", "Class:Linear", "klass:Linear", "weird:thing"])
    def test_pattern_validator_rejects_typo_prefixes(self, entry):
        # Regression: pre-fix, ``"clas:Foo"`` / ``"Class:Foo"`` silently routed to
        # ``name_bodies`` (since they don't start with ``class:``) and matched
        # nothing.  Patterns that contain ``:`` but lack a recognised prefix are
        # rejected loudly so the user fixes the typo at construction.
        with pytest.raises(ValidationError, match="contains ':' but doesn't start with"):
            OffloadConfig(include_patterns=[entry])

    def test_pattern_validator_typo_message_suggests_class_prefix(self):
        # The "did you mean class:<glob>?" hint helps users recover from the
        # most common typo (``clas:`` / ``Class:``).
        with pytest.raises(ValidationError, match="did you mean class:<glob>"):
            OffloadConfig(include_patterns=["clas:Linear"])

    @pytest.mark.parametrize("entry", ["", " ", "\t", "\n", "   \t  "])
    def test_pattern_validator_rejects_empty_entries(self, entry):
        # Empty / whitespace-only entries previously stripped to ``""`` and
        # were stored verbatim, producing a generic "did not match" warning
        # at offload time. Reject at construction for parity with ``"class:"``.
        with pytest.raises(ValidationError, match="empty or whitespace-only"):
            OffloadConfig(include_patterns=[entry])

    def test_json_schema_has_field_descriptions(self):
        """All OffloadConfig fields must have descriptions in JSON Schema."""
        schema = OffloadConfig.model_json_schema()
        properties = schema["properties"]
        missing = [name for name, prop in properties.items() if "description" not in prop]
        assert missing == [], (
            f"Fields missing 'description' in JSON Schema: {missing}. "
            "Each field must have an attribute docstring and model_config must have "
            "use_attribute_docstrings=True."
        )


class TestParseBool:
    """Test _parse_bool helper function."""

    def test_parse_true_lowercase(self):
        """Test parsing 'true' (lowercase)."""
        assert _parse_bool("true") is True

    def test_parse_true_uppercase(self):
        """Test parsing 'TRUE' (uppercase)."""
        assert _parse_bool("TRUE") is True

    def test_parse_true_mixed_case(self):
        """Test parsing 'True' (mixed case)."""
        assert _parse_bool("True") is True

    def test_parse_one(self):
        """Test parsing '1'."""
        assert _parse_bool("1") is True

    def test_parse_yes_lowercase(self):
        """Test parsing 'yes' (lowercase)."""
        assert _parse_bool("yes") is True

    def test_parse_yes_uppercase(self):
        """Test parsing 'YES' (uppercase)."""
        assert _parse_bool("YES") is True

    def test_parse_yes_mixed_case(self):
        """Test parsing 'Yes' (mixed case)."""
        assert _parse_bool("Yes") is True

    def test_parse_y(self):
        """Test parsing 'y'."""
        assert _parse_bool("y") is True

    def test_parse_on(self):
        """Test parsing 'on'."""
        assert _parse_bool("on") is True

    def test_parse_false_lowercase(self):
        """Test parsing 'false' (lowercase)."""
        assert _parse_bool("false") is False

    def test_parse_false_uppercase(self):
        """Test parsing 'FALSE' (uppercase)."""
        assert _parse_bool("FALSE") is False

    def test_parse_false_mixed_case(self):
        """Test parsing 'False' (mixed case)."""
        assert _parse_bool("False") is False

    def test_parse_zero(self):
        """Test parsing '0'."""
        assert _parse_bool("0") is False

    def test_parse_no_lowercase(self):
        """Test parsing 'no' (lowercase)."""
        assert _parse_bool("no") is False

    def test_parse_no_uppercase(self):
        """Test parsing 'NO' (uppercase)."""
        assert _parse_bool("NO") is False

    def test_parse_no_mixed_case(self):
        """Test parsing 'No' (mixed case)."""
        assert _parse_bool("No") is False

    def test_parse_n(self):
        """Test parsing 'n'."""
        assert _parse_bool("n") is False

    def test_parse_off(self):
        """Test parsing 'off'."""
        assert _parse_bool("off") is False

    def test_parse_invalid_value(self):
        """Test parsing invalid value raises ValueError."""
        with pytest.raises(ValueError, match="Cannot parse 'invalid' as boolean"):
            _parse_bool("invalid")

    def test_parse_empty_string(self):
        """Test parsing empty string raises ValueError."""
        with pytest.raises(ValueError):
            _parse_bool("")


class TestParseNone:
    """Test _parse_none helper function."""

    def test_parse_none_lowercase(self):
        """Test parsing 'none' (lowercase)."""
        assert _parse_none("none") is None

    def test_parse_none_uppercase(self):
        """Test parsing 'NONE' (uppercase)."""
        assert _parse_none("NONE") is None

    def test_parse_none_mixed_case(self):
        """Test parsing 'None' (mixed case)."""
        assert _parse_none("None") is None

    def test_parse_null_lowercase(self):
        """Test parsing 'null' (lowercase)."""
        assert _parse_none("null") is None

    def test_parse_null_uppercase(self):
        """Test parsing 'NULL' (uppercase)."""
        assert _parse_none("NULL") is None

    def test_parse_null_mixed_case(self):
        """Test parsing 'Null' (mixed case)."""
        assert _parse_none("Null") is None

    def test_parse_invalid_value(self):
        """Test parsing invalid value raises ValueError."""
        with pytest.raises(ValueError, match="Cannot parse 'invalid' as None"):
            _parse_none("invalid")


class TestLoadConfigFromEnv:
    """Test load_config_from_env function."""

    def test_load_with_no_env_vars(self):
        """Test loading config with no environment variables uses defaults."""
        config = load_config_from_env()
        assert config.gpu_device == 0
        assert config.discovery_iters == 1
        assert config.profiling_iters == 10

    def test_enabled_default_false(self):
        """Test that enabled defaults to False when loading from env."""
        config = load_config_from_env()
        assert config.enabled is False

    def test_enabled_explicitly_set_true(self):
        """Test that enabled can be explicitly set to True via env var."""
        os.environ["FT_ENABLED"] = "1"
        config = load_config_from_env()
        assert config.enabled is True

    def test_enabled_explicitly_set_false(self):
        """Test that enabled can be explicitly set to False via env var."""
        os.environ["FT_ENABLED"] = "0"
        config = load_config_from_env()
        assert config.enabled is False

    def test_enabled_override_with_kwargs(self):
        """Test that enabled can be overridden with kwargs."""
        config = load_config_from_env(enabled=True)
        assert config.enabled is True

    def test_load_int_field(self):
        """Test loading integer field from environment."""
        os.environ["FT_GPU_DEVICE"] = "2"
        config = load_config_from_env()
        assert config.gpu_device == 2

    def test_load_bool_field_true(self):
        """Test loading boolean field (True) from environment."""
        os.environ["FT_ENABLE_DIAGNOSTICS"] = "true"
        config = load_config_from_env()
        assert config.enable_diagnostics is True

    def test_load_bool_field_false(self):
        """Test loading boolean field (False) from environment."""
        os.environ["FT_PINNED_MEMORY"] = "false"
        config = load_config_from_env()
        assert config.pinned_memory is False

    def test_load_bool_field_case_insensitive(self):
        """Test loading boolean field with different cases."""
        os.environ["FT_ENABLE_DIAGNOSTICS"] = "TRUE"
        config = load_config_from_env()
        assert config.enable_diagnostics is True

        os.environ["FT_ENABLE_DIAGNOSTICS"] = "False"
        config = load_config_from_env()
        assert config.enable_diagnostics is False

    def test_load_float_field(self):
        """Test loading float field from environment."""
        os.environ["FT_TRANSFER_BUDGET_SCALE"] = "2.5"
        config = load_config_from_env()
        assert config.transfer_budget_scale == 2.5

    def test_load_string_field(self):
        """Test loading string field from environment."""
        os.environ["FT_TRANSFER_MODE"] = "custom_transfer"
        config = load_config_from_env()
        assert config.transfer_mode == "custom_transfer"

    def test_load_pinned_memory_mode_from_env(self):
        """FT_PINNED_MEMORY_MODE is picked up from the environment."""
        os.environ["FT_PINNED_MEMORY_MODE"] = "host_register"
        config = load_config_from_env()
        assert config.pinned_memory_mode == "host_register"

    def test_load_profile_mode_from_env(self):
        """FT_PROFILE_MODE is picked up from the environment."""
        os.environ["FT_PROFILE_MODE"] = "view"
        config = load_config_from_env()
        assert config.profile_mode == "view"

    def test_load_profile_mode_from_env_rejects_invalid(self):
        """Invalid FT_PROFILE_MODE values must surface as a ValidationError.

        Same regression guard as ``test_load_pinned_memory_mode_from_env_rejects_invalid``:
        Literal-typed env vars must reach Pydantic's validator instead of being
        dropped by the env-var resolver.
        """
        os.environ["FT_PROFILE_MODE"] = "bogus"
        with pytest.raises(ValidationError, match="profile_mode"):
            load_config_from_env()

    def test_load_pinned_memory_mode_from_env_rejects_invalid(self):
        """Invalid FT_PINNED_MEMORY_MODE values must surface as a
        ValidationError, not be silently dropped.

        Regression guard for the ``Literal[...]`` field annotation: an earlier
        revision of the env-var resolver classified Literal fields as
        ``object`` and skipped them entirely, so a typo in the env var would
        be ignored and the default would be used. Now the env var must reach
        Pydantic and trip Literal validation.
        """
        os.environ["FT_PINNED_MEMORY_MODE"] = "cuda_alloc"
        with pytest.raises(ValidationError, match="pinned_memory_mode"):
            load_config_from_env()

    def test_load_multiple_fields(self):
        """Test loading multiple fields from environment."""
        os.environ["FT_GPU_DEVICE"] = "1"
        os.environ["FT_DISCOVERY_ITERS"] = "5"
        os.environ["FT_PROFILING_ITERS"] = "15"
        os.environ["FT_ENABLE_DIAGNOSTICS"] = "true"

        config = load_config_from_env()
        assert config.gpu_device == 1
        assert config.discovery_iters == 5
        assert config.profiling_iters == 15
        assert config.enable_diagnostics is True

    def test_load_with_custom_prefix(self):
        """Test loading config with custom prefix."""
        os.environ["CUSTOM_GPU_DEVICE"] = "3"
        os.environ["CUSTOM_DISCOVERY_ITERS"] = "7"

        config = load_config_from_env(prefix="CUSTOM_")
        assert config.gpu_device == 3
        assert config.discovery_iters == 7

    def test_load_partial_config(self):
        """Test loading config with only some environment variables set."""
        os.environ["FT_GPU_DEVICE"] = "2"
        # Other fields should use defaults
        config = load_config_from_env()
        assert config.gpu_device == 2
        assert config.discovery_iters == 1  # default
        assert config.profiling_iters == 10  # default

    def test_kwargs_override_env_vars(self):
        """Test that kwargs override environment variables."""
        os.environ["FT_GPU_DEVICE"] = "1"
        os.environ["FT_DISCOVERY_ITERS"] = "5"

        config = load_config_from_env(gpu_device=2, discovery_iters=10)
        assert config.gpu_device == 2  # kwargs override
        assert config.discovery_iters == 10  # kwargs override

    def test_invalid_int_raises_error(self):
        """Test that invalid integer value raises error."""
        os.environ["FT_GPU_DEVICE"] = "not_an_int"

        with pytest.raises(ValueError, match="Failed to convert FT_GPU_DEVICE"):
            load_config_from_env()

    def test_invalid_float_raises_error(self):
        """Test that invalid float value raises error."""
        os.environ["FT_TRANSFER_BUDGET_SCALE"] = "not_a_float"

        with pytest.raises(ValueError, match="Failed to convert FT_TRANSFER_BUDGET_SCALE"):
            load_config_from_env()

    def test_invalid_bool_raises_error(self):
        """Test that invalid boolean value raises error."""
        os.environ["FT_ENABLE_DIAGNOSTICS"] = "not_a_bool"

        with pytest.raises(ValueError, match="Failed to convert FT_ENABLE_DIAGNOSTICS"):
            load_config_from_env()

    def test_validation_error_propagates(self):
        """Test that pydantic validation errors are raised."""
        os.environ["FT_GPU_DEVICE"] = "-1"  # Invalid: must be >= 0

        with pytest.raises(ValidationError):
            load_config_from_env()

    def test_all_bool_fields(self):
        """Test loading all boolean fields from environment."""
        os.environ["FT_PINNED_MEMORY"] = "false"
        os.environ["FT_ENABLE_INSTRUMENTATION"] = "true"
        os.environ["FT_ENABLE_DIAGNOSTICS"] = "true"

        config = load_config_from_env()
        assert config.pinned_memory is False
        assert config.enable_instrumentation is True
        assert config.enable_diagnostics is True

    def test_min_blocks_from_env(self):
        """Test loading min_blocks from FT_MIN_BLOCKS env var."""
        os.environ["FT_MIN_BLOCKS"] = "3"
        config = load_config_from_env()
        assert config.min_blocks == 3

    def test_max_gpu_mem_fraction_from_env(self):
        """Test loading max_gpu_mem_fraction from FT_MAX_GPU_MEM_FRACTION env var."""
        os.environ["FT_MAX_GPU_MEM_FRACTION"] = "0.75"
        config = load_config_from_env()
        assert config.max_gpu_mem_fraction == 0.75

    def test_max_gpu_mem_fraction_none_from_env(self):
        """Test setting max_gpu_mem_fraction to None via FT_MAX_GPU_MEM_FRACTION=none."""
        os.environ["FT_MAX_GPU_MEM_FRACTION"] = "none"
        config = load_config_from_env()
        assert config.max_gpu_mem_fraction is None

    def test_all_int_fields(self):
        """Test loading all integer fields from environment."""
        os.environ["FT_GPU_DEVICE"] = "3"
        os.environ["FT_DISCOVERY_ITERS"] = "8"
        os.environ["FT_PROFILING_ITERS"] = "25"
        os.environ["FT_NUM_BLOCKS"] = "10"
        os.environ["FT_MIN_BLOCKS"] = "3"

        config = load_config_from_env()
        assert config.gpu_device == 3
        assert config.discovery_iters == 8
        assert config.profiling_iters == 25
        assert config.num_blocks == 10
        assert config.min_blocks == 3

    def test_all_float_fields(self):
        """Test loading all float fields from environment."""
        os.environ["FT_TRANSFER_BUDGET_SCALE"] = "2.5"

        config = load_config_from_env()
        assert config.transfer_budget_scale == 2.5

    def test_bool_variations_yes_no(self):
        """Test boolean parsing with 'yes' and 'no'."""
        os.environ["FT_ENABLE_DIAGNOSTICS"] = "yes"
        config = load_config_from_env()
        assert config.enable_diagnostics is True

        os.environ["FT_ENABLE_DIAGNOSTICS"] = "no"
        config = load_config_from_env()
        assert config.enable_diagnostics is False

    def test_bool_variations_one_zero(self):
        """Test boolean parsing with '1' and '0'."""
        os.environ["FT_ENABLE_DIAGNOSTICS"] = "1"
        config = load_config_from_env()
        assert config.enable_diagnostics is True

        os.environ["FT_ENABLE_DIAGNOSTICS"] = "0"
        config = load_config_from_env()
        assert config.enable_diagnostics is False

    def test_load_strategy_field_skipped(self):
        """Test that load_strategy field is skipped (complex type)."""
        # load_strategy is a complex type that can't be set via env vars
        os.environ["FT_LOAD_STRATEGY"] = "something"
        config = load_config_from_env()
        # Should ignore the env var and use default (None)
        assert config.load_strategy is None

    def test_unrecognized_ft_env_var_raises_with_suggestion(self):
        os.environ["FT_ENABLE_DIAGNOSTCIS"] = "1"

        with pytest.raises(ValueError, match="FT_ENABLE_DIAGNOSTCIS") as exc_info:
            load_config_from_env()

        assert str(exc_info.value) == (
            "Unrecognized environment variable with FT_ prefix: "
            "FT_ENABLE_DIAGNOSTCIS. Did you mean: FT_ENABLE_DIAGNOSTICS?"
        )

    def test_unrecognized_custom_env_var_raises(self):
        os.environ["CUSTOM_UNKNOWN"] = "1"

        with pytest.raises(
            ValueError,
            match="Unrecognized environment variable with CUSTOM_ prefix: CUSTOM_UNKNOWN",
        ) as exc_info:
            load_config_from_env(prefix="CUSTOM_")
        assert "Did you mean" not in str(exc_info.value)

    def test_removed_env_var_is_recognized_and_ignored(self, caplog):
        os.environ["FT_RELEASE_TENSORS"] = "1"

        with caplog.at_level(logging.WARNING, logger="flextensor.config"):
            load_config_from_env()

        assert "'release_tensors' was removed" in caplog.text

    def test_mixed_env_and_kwargs_with_validation(self):
        """Test mixed environment and kwargs with validation."""
        os.environ["FT_GPU_DEVICE"] = "1"

        config = load_config_from_env(discovery_iters=20, enable_diagnostics=True)
        assert config.gpu_device == 1  # from env
        assert config.discovery_iters == 20  # from kwargs
        assert config.enable_diagnostics is True  # from kwargs

    def test_include_patterns_from_env(self):
        """Test loading include_patterns from FT_INCLUDE_PATTERNS env var."""
        os.environ["FT_INCLUDE_PATTERNS"] = "layers.*,head"
        config = load_config_from_env()
        assert config.include_patterns == ["layers.*", "head"]

    def test_include_patterns_from_env_with_spaces(self):
        """Test loading include_patterns with spaces in env var."""
        os.environ["FT_INCLUDE_PATTERNS"] = "layers.*, head , model.norm"
        config = load_config_from_env()
        assert config.include_patterns == ["layers.*", "head", "model.norm"]

    def test_include_patterns_from_env_single(self):
        """Test loading single module pattern from env var."""
        os.environ["FT_INCLUDE_PATTERNS"] = "model.*"
        config = load_config_from_env()
        assert config.include_patterns == ["model.*"]

    def test_include_patterns_from_env_wildcard(self):
        """Test loading wildcard module pattern from env var."""
        os.environ["FT_INCLUDE_PATTERNS"] = "*"
        config = load_config_from_env()
        assert config.include_patterns == ["*"]

    def test_include_patterns_kwargs_override_env(self):
        """Test that kwargs override env var for include_patterns."""
        os.environ["FT_INCLUDE_PATTERNS"] = "layers.*"
        config = load_config_from_env(include_patterns=["custom.*"])
        assert config.include_patterns == ["custom.*"]

    def test_include_patterns_none_literal_kept_as_string(self):
        """Ensure 'none' in a list[str] env var stays a string, not Python None."""
        os.environ["FT_INCLUDE_PATTERNS"] = "layers.*,none,head"
        config = load_config_from_env()
        assert config.include_patterns == ["layers.*", "none", "head"]

    def test_exclude_patterns_none_literal_kept_as_string(self):
        """Ensure 'none' in exclude_patterns env var stays a string."""
        os.environ["FT_EXCLUDE_PATTERNS"] = "none,lm_head"
        config = load_config_from_env()
        assert config.exclude_patterns == ["none", "lm_head"]

    def test_str_none_field_parses_none_as_python_none(self):
        """Ensure 'none' in a top-level str|None field becomes Python None."""
        os.environ["FT_SHM_NAMESPACE"] = "none"
        config = load_config_from_env()
        assert config.shm_namespace is None

    def test_exclude_patterns_from_env(self):
        """Test loading exclude_patterns from FT_EXCLUDE_PATTERNS env var."""
        os.environ["FT_EXCLUDE_PATTERNS"] = "lm_head,*.norm"
        config = load_config_from_env()
        assert config.exclude_patterns == ["lm_head", "*.norm"]

    def test_exclude_patterns_from_env_single(self):
        """Test loading single exclude pattern from env var."""
        os.environ["FT_EXCLUDE_PATTERNS"] = "lm_head"
        config = load_config_from_env()
        assert config.exclude_patterns == ["lm_head"]

    def test_external_compile_from_env(self):
        os.environ["FT_EXTERNAL_COMPILE"] = "1"
        config = load_config_from_env()
        assert config.external_compile is True


class TestLoadConfigFromFile:
    """Test load_config_from_file function."""

    def test_load_ini_file(self, tmp_path):
        """Test loading config from INI file."""
        config_file = tmp_path / "test.conf"
        config_file.write_text("""[flextensor]
enabled = true
gpu_device = 2
discovery_iters = 5
enable_diagnostics = true
""")
        config = load_config_from_file(config_file)
        assert config.enabled is True
        assert config.gpu_device == 2
        assert config.discovery_iters == 5
        assert config.enable_diagnostics is True

    def test_load_ini_file_with_ini_extension(self, tmp_path):
        """Test loading config from .ini file."""
        config_file = tmp_path / "test.ini"
        config_file.write_text("""[flextensor]
enabled = true
gpu_device = 3
""")
        config = load_config_from_file(config_file)
        assert config.enabled is True
        assert config.gpu_device == 3

    def test_load_json_file(self, tmp_path):
        """Test loading config from JSON file."""
        config_file = tmp_path / "test.json"
        config_file.write_text("""{
    "enabled": true,
    "gpu_device": 4,
    "discovery_iters": 8,
    "enable_diagnostics": false
}""")
        config = load_config_from_file(config_file)
        assert config.enabled is True
        assert config.gpu_device == 4
        assert config.discovery_iters == 8
        assert config.enable_diagnostics is False

    def test_load_json_file_with_patterns(self, tmp_path):
        """Test loading include_patterns and exclude_patterns from JSON file."""
        config_file = tmp_path / "test_patterns.json"
        config_file.write_text("""{
    "enabled": true,
    "include_patterns": ["layers.*", "head"],
    "exclude_patterns": ["layers.*.norm"]
}""")
        config = load_config_from_file(config_file)
        assert config.include_patterns == ["layers.*", "head"]
        assert config.exclude_patterns == ["layers.*.norm"]

    def test_load_yaml_file(self, tmp_path):
        """Test loading config from YAML file."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("""enabled: true
gpu_device: 5
discovery_iters: 10
enable_diagnostics: true
""")
        config = load_config_from_file(config_file)
        assert config.enabled is True
        assert config.gpu_device == 5
        assert config.discovery_iters == 10
        assert config.enable_diagnostics is True

    def test_load_yaml_file_with_patterns(self, tmp_path):
        """Test loading include_patterns and exclude_patterns from YAML file."""
        config_file = tmp_path / "test_patterns.yaml"
        config_file.write_text("""enabled: true
include_patterns:
  - "layers.*"
  - "embed_tokens"
exclude_patterns:
  - "layers.*.norm"
  - "lm_head"
""")
        config = load_config_from_file(config_file)
        assert config.include_patterns == ["layers.*", "embed_tokens"]
        assert config.exclude_patterns == ["layers.*.norm", "lm_head"]

    def test_load_yml_file(self, tmp_path):
        """Test loading config from .yml file."""
        config_file = tmp_path / "test.yml"
        config_file.write_text("""enabled: false
gpu_device: 6
""")
        config = load_config_from_file(config_file)
        assert config.enabled is False
        assert config.gpu_device == 6

    def test_load_from_env_var(self, tmp_path):
        """Test loading config from FT_CONFIG_FILE env var."""
        config_file = tmp_path / "env_test.yaml"
        config_file.write_text("""enabled: true
gpu_device: 7
""")
        os.environ["FT_CONFIG_FILE"] = str(config_file)
        config = load_config_from_file()
        assert config.enabled is True
        assert config.gpu_device == 7

    def test_load_from_custom_env_var(self, tmp_path):
        """Test loading config from custom env var."""
        config_file = tmp_path / "custom_env.yaml"
        config_file.write_text("""enabled: true
gpu_device: 8
""")
        os.environ["MY_CONFIG_FILE"] = str(config_file)
        config = load_config_from_file(env_var="MY_CONFIG_FILE")
        assert config.enabled is True
        assert config.gpu_device == 8

    def test_kwargs_override_file_values(self, tmp_path):
        """Test that kwargs override file values."""
        config_file = tmp_path / "override.yaml"
        config_file.write_text("""enabled: true
gpu_device: 1
discovery_iters: 5
""")
        config = load_config_from_file(config_file, gpu_device=10, discovery_iters=20)
        assert config.enabled is True  # from file
        assert config.gpu_device == 10  # from kwargs
        assert config.discovery_iters == 20  # from kwargs

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config_from_file("/nonexistent/path/config.yaml")

    def test_no_config_specified(self):
        """Test that FileNotFoundError is raised when no config specified."""
        with pytest.raises(FileNotFoundError, match="No config file specified"):
            load_config_from_file()

    def test_unsupported_format(self, tmp_path):
        """Test that ValueError is raised for unsupported format."""
        config_file = tmp_path / "test.txt"
        config_file.write_text("some content")
        with pytest.raises(ValueError, match="Unsupported config file format"):
            load_config_from_file(config_file)

    def test_ini_missing_section(self, tmp_path):
        """Test that ValueError is raised for INI file without [flextensor] section."""
        config_file = tmp_path / "bad.conf"
        config_file.write_text("""[other_section]
gpu_device = 1
""")
        with pytest.raises(ValueError, match="must contain"):
            load_config_from_file(config_file)

    def test_partial_config_file(self, tmp_path):
        """Test loading file with only some fields uses defaults for rest."""
        config_file = tmp_path / "partial.yaml"
        config_file.write_text("""gpu_device: 3
""")
        config = load_config_from_file(config_file)
        assert config.gpu_device == 3
        assert config.discovery_iters == 1  # default
        assert config.enabled is True  # default (not False like env loading)

    def test_json_with_string_values(self, tmp_path):
        """Test loading JSON with string values that need conversion."""
        config_file = tmp_path / "string_vals.json"
        config_file.write_text("""{
    "enabled": "true",
    "gpu_device": "5",
    "transfer_budget_scale": "1.5"
}""")
        config = load_config_from_file(config_file)
        assert config.enabled is True
        assert config.gpu_device == 5
        assert config.transfer_budget_scale == 1.5

    def test_yaml_with_string_values(self, tmp_path):
        """Test loading YAML with string values that need conversion."""
        config_file = tmp_path / "string_vals.yaml"
        config_file.write_text("""enabled: "yes"
gpu_device: "6"
transfer_budget_scale: "1.5"
""")
        config = load_config_from_file(config_file)
        assert config.enabled is True
        assert config.gpu_device == 6
        assert config.transfer_budget_scale == 1.5

    def test_empty_yaml_file(self, tmp_path):
        """Test loading empty YAML file uses all defaults."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        config = load_config_from_file(config_file)
        assert config.gpu_device == 0  # default
        assert config.enabled is True  # default

    def test_path_object(self, tmp_path):
        """Test that Path object works as config_path."""
        config_file = tmp_path / "path_test.yaml"
        config_file.write_text("""gpu_device: 9
""")
        config = load_config_from_file(Path(config_file))
        assert config.gpu_device == 9

    def test_string_path(self, tmp_path):
        """Test that string path works as config_path."""
        config_file = tmp_path / "string_test.yaml"
        config_file.write_text("""gpu_device: 11
""")
        config = load_config_from_file(str(config_file))
        assert config.gpu_device == 11

    def test_all_fields_ini(self, tmp_path):
        """Test loading all supported fields from INI file."""
        config_file = tmp_path / "all_fields.conf"
        config_file.write_text("""[flextensor]
enabled = true
gpu_device = 1
pinned_memory = false
discovery_iters = 3
profiling_iters = 15
transfer_budget_scale = 2.0
transfer_mode = custom_mode
num_blocks = 8
""")
        config = load_config_from_file(config_file)
        assert config.enabled is True
        assert config.gpu_device == 1
        assert config.pinned_memory is False
        assert config.discovery_iters == 3
        assert config.profiling_iters == 15
        assert config.transfer_budget_scale == 2.0
        assert config.transfer_mode == "custom_mode"
        assert config.num_blocks == 8

    def test_validation_error_from_file(self, tmp_path):
        """Test that pydantic validation errors are raised for invalid values in file."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("""gpu_device: -1
""")
        with pytest.raises(ValidationError):
            load_config_from_file(config_file)

    def test_min_blocks_from_yaml_file(self, tmp_path):
        """Test loading min_blocks from YAML file."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("""min_blocks: 3
""")
        config = load_config_from_file(config_file)
        assert config.min_blocks == 3

    def test_max_gpu_mem_fraction_from_yaml_file(self, tmp_path):
        """Test loading max_gpu_mem_fraction from YAML file."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("max_gpu_mem_fraction: 0.8\n")
        config = load_config_from_file(config_file)
        assert config.max_gpu_mem_fraction == 0.8

    def test_max_gpu_mem_fraction_none_from_yaml_file(self, tmp_path):
        """Test that max_gpu_mem_fraction: null in YAML file results in None."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("max_gpu_mem_fraction: null\n")
        config = load_config_from_file(config_file)
        assert config.max_gpu_mem_fraction is None

    def test_max_gpu_mem_fraction_default_when_absent_from_yaml(self, tmp_path):
        """Test that max_gpu_mem_fraction defaults to None when not in YAML file."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("gpu_device: 0\n")
        config = load_config_from_file(config_file)
        assert config.max_gpu_mem_fraction is None

    def test_max_gpu_mem_fraction_from_json_file(self, tmp_path):
        """Test loading max_gpu_mem_fraction from JSON file."""
        config_file = tmp_path / "test.json"
        config_file.write_text('{"max_gpu_mem_fraction": 0.6}')
        config = load_config_from_file(config_file)
        assert config.max_gpu_mem_fraction == 0.6

    def test_include_patterns_from_yaml_file(self, tmp_path):
        """Test loading include_patterns from YAML file."""
        config_file = tmp_path / "patterns.yaml"
        config_file.write_text("""include_patterns:
  - "layers.*"
  - "head"
  - "norm"
""")
        config = load_config_from_file(config_file)
        assert config.include_patterns == ["layers.*", "head", "norm"]

    def test_include_patterns_from_json_file(self, tmp_path):
        """Test loading include_patterns from JSON file."""
        config_file = tmp_path / "patterns.json"
        config_file.write_text("""{
    "include_patterns": ["model.*", "lm_head"]
}""")
        config = load_config_from_file(config_file)
        assert config.include_patterns == ["model.*", "lm_head"]

    def test_include_patterns_default_when_not_in_file(self, tmp_path):
        """Test that include_patterns uses default when not in config file."""
        config_file = tmp_path / "no_patterns.yaml"
        config_file.write_text("""gpu_device: 1
""")
        config = load_config_from_file(config_file)
        assert config.include_patterns == ["*"]  # default

    def test_exclude_patterns_from_yaml_file(self, tmp_path):
        """Test loading exclude_patterns from YAML file."""
        config_file = tmp_path / "exclude.yaml"
        config_file.write_text("""exclude_patterns:
  - "lm_head"
  - "*.norm"
""")
        config = load_config_from_file(config_file)
        assert config.exclude_patterns == ["lm_head", "*.norm"]

    def test_exclude_patterns_from_json_file(self, tmp_path):
        """Test loading exclude_patterns from JSON file."""
        config_file = tmp_path / "exclude.json"
        config_file.write_text("""{
    "include_patterns": ["*"],
    "exclude_patterns": ["lm_head", "*.scale"]
}""")
        config = load_config_from_file(config_file)
        assert config.include_patterns == ["*"]
        assert config.exclude_patterns == ["lm_head", "*.scale"]

    def test_pinned_memory_mode_round_trips_through_yaml(self, tmp_path):
        """``pinned_memory_mode: host_register`` survives a YAML file load.

        Locks in the documented behaviour from
        ``docs/explanation/configuration.md`` that the mode field is a
        first-class config option deployable via file. The Pydantic
        ``Literal["torch", "host_register"]`` typing means a YAML string
        passes through validation and lands on the model unchanged; this
        test pins that contract for the new field.
        """
        config_file = tmp_path / "pin_mode.yaml"
        config_file.write_text("""enabled: true
pinned_memory: true
pinned_memory_mode: host_register
""")
        config = load_config_from_file(config_file)
        assert config.pinned_memory is True
        assert config.pinned_memory_mode == "host_register"

    def test_pinned_memory_mode_invalid_yaml_value_rejected(self, tmp_path):
        """An unknown ``pinned_memory_mode`` in a YAML file must surface as
        a ``ValidationError`` at load time, not silently fall through to
        the default. Mirrors ``test_pinned_memory_mode_invalid_rejected``
        for the construction path.
        """
        config_file = tmp_path / "bad_pin_mode.yaml"
        config_file.write_text("""pinned_memory_mode: cuda_alloc
""")
        with pytest.raises(ValidationError, match="pinned_memory_mode"):
            load_config_from_file(config_file)


class TestGetFieldTypes:
    """Test _get_field_types helper function."""

    def test_returns_dict(self):
        """Test that _get_field_types returns a dictionary."""
        field_types = _get_field_types()
        assert isinstance(field_types, dict)

    def test_contains_all_fields(self):
        """Test that all OffloadConfig fields are present."""
        field_types = _get_field_types()
        for field_name in OffloadConfig.model_fields:
            assert field_name in field_types

    def test_bool_fields(self):
        """Test that boolean fields are correctly identified."""
        field_types = _get_field_types()
        assert field_types["enabled"] is bool
        assert field_types["pinned_memory"] is bool
        assert field_types["enable_instrumentation"] is bool

    def test_int_fields(self):
        """Test that integer fields are correctly identified."""
        field_types = _get_field_types()
        assert field_types["gpu_device"] is int
        assert field_types["discovery_iters"] is int
        assert field_types["num_blocks"] is int
        assert field_types["min_blocks"] is int

    def test_float_fields(self):
        """Test that float fields are correctly identified."""
        field_types = _get_field_types()
        assert field_types["transfer_budget_scale"] is float
        assert field_types["max_gpu_mem_fraction"] is float

    def test_str_fields(self):
        """Test that string fields are correctly identified."""
        field_types = _get_field_types()
        assert field_types["transfer_mode"] is str

    def test_complex_fields(self):
        """Test that complex fields (like strategies) are marked as object."""
        field_types = _get_field_types()
        assert field_types["load_strategy"] is object

    def test_list_fields(self):
        """Test that list fields are correctly identified."""
        field_types = _get_field_types()
        assert field_types["include_patterns"] is list
        assert field_types["exclude_patterns"] is list


class TestLoadConfig:
    """Test unified load_config function with file + env + kwargs precedence."""

    def test_env_only_no_file(self):
        """Test loading from env only when no file is specified."""
        os.environ["FT_GPU_DEVICE"] = "5"
        config = load_config()
        assert config.gpu_device == 5
        assert config.enabled is False  # default when env-only

    def test_env_only_enabled_default_false(self):
        """Test that enabled defaults to False when loading from env only."""
        config = load_config()
        assert config.enabled is False

    def test_env_override_file(self, tmp_path):
        """Test that env vars override file values."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("""enabled: true
gpu_device: 1
discovery_iters: 5
""")
        os.environ["FT_GPU_DEVICE"] = "10"

        config = load_config(config_path=config_file)
        assert config.enabled is True  # from file
        assert config.gpu_device == 10  # from env (overrides file)
        assert config.discovery_iters == 5  # from file

    def test_env_overrides_pinned_memory_mode_in_file(self, tmp_path):
        """``FT_PINNED_MEMORY_MODE`` overrides the value set in a YAML file.

        Documented precedence is kwargs > env > file. This pins that
        ordering for the new ``pinned_memory_mode`` field — a deployment
        that ships ``pinned_memory_mode: torch`` in its baseline YAML can
        opt into ``host_register`` per-host via the env var.
        """
        config_file = tmp_path / "pin_mode.yaml"
        config_file.write_text("""enabled: true
pinned_memory: true
pinned_memory_mode: torch
""")
        os.environ["FT_PINNED_MEMORY_MODE"] = "host_register"

        config = load_config(config_path=config_file)
        assert config.pinned_memory is True  # from file
        assert config.pinned_memory_mode == "host_register"  # from env (overrides file)

    def test_kwargs_override_env_and_file(self, tmp_path):
        """Test that kwargs override both env and file values."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("""enabled: true
gpu_device: 1
discovery_iters: 5
""")
        os.environ["FT_GPU_DEVICE"] = "10"
        os.environ["FT_DISCOVERY_ITERS"] = "15"

        config = load_config(config_path=config_file, gpu_device=20, discovery_iters=25)
        assert config.enabled is True  # from file
        assert config.gpu_device == 20  # from kwargs (overrides env and file)
        assert config.discovery_iters == 25  # from kwargs (overrides env and file)

    def test_file_with_use_env_false(self, tmp_path):
        """Test file loading without env override when use_env=False."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("""enabled: true
gpu_device: 1
""")
        os.environ["FT_GPU_DEVICE"] = "10"

        config = load_config(config_path=config_file, use_env=False)
        assert config.enabled is True  # from file
        assert config.gpu_device == 1  # from file (env not used)

    def test_file_from_env_var(self, tmp_path):
        """Test loading file path from FT_CONFIG_FILE env var."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("""enabled: true
gpu_device: 7
""")
        os.environ["FT_CONFIG_FILE"] = str(config_file)

        config = load_config()
        assert config.enabled is True
        assert config.gpu_device == 7

    def test_file_from_custom_env_var_with_matching_prefix(self, tmp_path):
        """Test loading a file from a custom env var under the validated prefix."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("gpu_device: 7\n")
        os.environ["CUSTOM_CONFIG_PATH"] = str(config_file)

        config = load_config(
            env_prefix="CUSTOM_",
            config_file_env_var="CUSTOM_CONFIG_PATH",
        )

        assert config.gpu_device == 7

    def test_file_from_env_var_with_env_override(self, tmp_path):
        """Test file from env var with additional env overrides."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("""enabled: true
gpu_device: 1
discovery_iters: 5
""")
        os.environ["FT_CONFIG_FILE"] = str(config_file)
        os.environ["FT_GPU_DEVICE"] = "10"

        config = load_config()
        assert config.enabled is True  # from file
        assert config.gpu_device == 10  # from env (overrides file)
        assert config.discovery_iters == 5  # from file

    def test_custom_env_prefix(self, tmp_path):
        """Test using custom env prefix."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("""gpu_device: 1
""")
        os.environ["CUSTOM_GPU_DEVICE"] = "20"

        config = load_config(config_path=config_file, env_prefix="CUSTOM_")
        assert config.gpu_device == 20

    def test_file_not_found(self):
        """Test FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config(config_path="/nonexistent/path.yaml")

    def test_file_not_found_from_env_var(self, tmp_path):
        """Test FileNotFoundError when FT_CONFIG_FILE points to missing file."""
        os.environ["FT_CONFIG_FILE"] = "/nonexistent/path.yaml"
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config()

    def test_full_precedence_chain(self, tmp_path):
        """Test full precedence: file < env < kwargs."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("""enabled: false
gpu_device: 1
discovery_iters: 5
profiling_iters: 10
""")
        os.environ["FT_GPU_DEVICE"] = "2"
        os.environ["FT_DISCOVERY_ITERS"] = "10"
        os.environ["FT_PROFILING_ITERS"] = "20"

        config = load_config(
            config_path=config_file,
            discovery_iters=15,
            profiling_iters=25,
        )

        # enabled: from file (not overridden)
        assert config.enabled is False
        # gpu_device: from env (overrides file)
        assert config.gpu_device == 2
        # discovery_iters: from kwargs (overrides env and file)
        assert config.discovery_iters == 15
        # profiling_iters: from kwargs (overrides env and file)
        assert config.profiling_iters == 25

    def test_enabled_from_env_overrides_file(self, tmp_path):
        """Test that FT_ENABLED from env overrides file value."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("""enabled: false
""")
        os.environ["FT_ENABLED"] = "1"

        config = load_config(config_path=config_file)
        assert config.enabled is True  # from env (overrides file)


class TestShmConfigFields:
    """Test new SHM config fields on OffloadConfig."""

    def test_shm_config_fields_exist(self):
        """New SHM config fields have correct defaults."""
        config = OffloadConfig()
        assert config.shm_enabled is False
        assert config.shm_namespace is None
        assert config.shm_wait_timeout == 0.0

    def test_valid_shm_config_construction_no_warning(self):
        """A complete SHM configuration emits no deprecation warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            config = OffloadConfig(shm_enabled=True, shm_namespace="my_model")

        assert config.shm_wait_timeout == 0.0

    @pytest.mark.parametrize("namespace", [None, "", "   \t"])
    def test_shm_enabled_requires_non_empty_namespace(self, namespace):
        """SHM configuration fails fast without a stable visible namespace."""
        with pytest.raises(ValidationError, match=r"shm_namespace.*shm_enabled=True"):
            OffloadConfig(shm_enabled=True, shm_namespace=namespace)

    def test_model_copy_revalidates_shm_namespace(self):
        """Enabling SHM through model_copy cannot bypass the namespace invariant."""
        with pytest.raises(ValidationError, match=r"shm_namespace.*shm_enabled=True"):
            OffloadConfig().model_copy(update={"shm_enabled": True})

    def test_shm_env_vars_loaded(self, monkeypatch):
        """FT_SHM_* env vars populate config fields."""
        monkeypatch.setenv("FT_SHM_ENABLED", "1")
        monkeypatch.setenv("FT_SHM_NAMESPACE", "my_model")
        monkeypatch.setenv("FT_SHM_WAIT_TIMEOUT", "300")
        config = load_config_from_env()
        assert config.shm_enabled is True
        assert config.shm_namespace == "my_model"
        assert config.shm_wait_timeout == 300.0


class TestExpiredCompatibilityFields:
    @pytest.mark.parametrize(
        "field",
        [
            "knapsack_scale",
            "module_patterns",
            "max_gpu_mem_bytes",
            "use_shared_memory",
        ],
    )
    def test_expired_field_is_not_exposed(self, field):
        with pytest.raises(AttributeError):
            getattr(OffloadConfig(), field)


class TestRemovedFieldsRejection:
    """Verify that removed OffloadConfig fields are rejected with clear errors."""

    _REMOVED: ClassVar[list[str]] = [
        "release_tensors",
        "enable_direct_mode",
        "enable_tracing",
        "rearrange_transfers",
        "compute_transfer_gap",
        "enable_untraced_tensor_discovery",
        "enable_module_tracker",
    ]

    @pytest.mark.parametrize("field", _REMOVED)
    def test_constructor_rejects_removed_field(self, field):
        """Passing a removed field to OffloadConfig() raises ValueError."""
        with pytest.raises(ValidationError, match=f"'{field}' was removed in v0.2.0"):
            OffloadConfig(**{field: False})

    @pytest.mark.parametrize("field", _REMOVED)
    def test_env_var_warns_for_removed_field(self, field, monkeypatch, caplog):
        """Setting a removed FT_* env var logs a warning."""
        env_var = f"FT_{field.upper()}"
        monkeypatch.setenv(env_var, "false")
        with caplog.at_level(logging.WARNING, logger="flextensor.config"):
            load_config_from_env()
        assert f"'{env_var}' is ignored" in caplog.text

    @pytest.mark.parametrize("field", _REMOVED)
    def test_ini_file_rejects_removed_field(self, field, tmp_path):
        """Removed field in an INI config file raises ValueError."""
        config_file = tmp_path / "test.conf"
        config_file.write_text(f"[flextensor]\n{field} = false\n")
        with pytest.raises(ValidationError, match=f"'{field}' was removed in v0.2.0"):
            load_config_from_file(config_file)

    @pytest.mark.parametrize("field", _REMOVED)
    def test_json_file_rejects_removed_field(self, field, tmp_path):
        """Removed field in a JSON config file raises ValueError."""
        import json

        config_file = tmp_path / "test.json"
        config_file.write_text(json.dumps({field: False}))
        with pytest.raises(ValidationError, match=f"'{field}' was removed in v0.2.0"):
            load_config_from_file(config_file)

    def test_pinned_memory_not_rejected(self):
        """pinned_memory is still a valid field and must not be rejected."""
        config = OffloadConfig(pinned_memory=False)
        assert config.pinned_memory is False


@pytest.mark.parametrize("field", ["warmup_iters", "profile_iters"])
def test_v040_constructor_rejects_removed_iter_field(field):
    with pytest.raises(ValidationError, match=rf"'{field}' was removed in v0\.4\.0"):
        OffloadConfig(**{field: 1})


@pytest.mark.parametrize("field", ["warmup_iters", "profile_iters"])
def test_v040_model_copy_rejects_removed_iter_field(field):
    with pytest.raises(ValueError, match=rf"'{field}' was removed in v0\.4\.0"):
        OffloadConfig().model_copy(update={field: 1})


@pytest.mark.parametrize("name", ["all_warmup_iters", "pre_inference_iters"])
def test_v040_removed_iter_property_is_absent(name):
    assert not hasattr(OffloadConfig(), name)


class TestModelCopyPreservesFieldsSet:
    """``model_copy(update=...)`` must not inflate ``model_fields_set``.

    ``model_copy`` re-parses through ``model_validate`` to re-assert the
    ``@model_validator`` invariants that ``model_copy`` bypasses. Re-parsing a
    full ``model_dump()`` would mark *every* field as explicitly set, which
    silently breaks :func:`flextensor.utils.config_field_was_set` — the
    "did the user customize this?" signal the vLLM worker uses to decide
    whether to apply its default include/exclude patterns.
    """

    def test_unset_field_stays_unset_after_copy(self) -> None:
        config = OffloadConfig(discovery_iters=1)
        copied = config.model_copy(update={"profiling_iters": 3})

        assert not config_field_was_set(copied, "exclude_patterns"), (
            "re-parsing the full dump marked an untouched field as explicitly "
            "set; config_field_was_set can no longer distinguish user intent"
        )

    def test_copy_does_not_mark_every_field_set(self) -> None:
        config = OffloadConfig(discovery_iters=1)
        copied = config.model_copy(update={"profiling_iters": 3})

        assert len(copied.model_fields_set) < len(type(config).model_fields)

    def test_explicitly_set_and_updated_fields_are_reported_set(self) -> None:
        config = OffloadConfig(discovery_iters=1)
        copied = config.model_copy(update={"profiling_iters": 3})

        assert config_field_was_set(copied, "discovery_iters")
        assert config_field_was_set(copied, "profiling_iters")

    def test_copy_still_enforces_validators(self) -> None:
        """The re-parse must keep rejecting invariant violations."""
        config = OffloadConfig(num_blocks=4, min_blocks=2)
        with pytest.raises(ValidationError):
            config.model_copy(update={"num_blocks": 1, "min_blocks": 8})
