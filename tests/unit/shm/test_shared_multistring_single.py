# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-process unit tests for SharedMultiString class."""

import struct
from multiprocessing.shared_memory import SharedMemory

import pytest

import flextensor.shm.multiprocess_condition as condition_module
from flextensor.shm import HEADER_SIZE, SharedMultiString


class TestSharedMultiStringSingle:
    """Single-process tests for SharedMultiString class."""

    def setup_method(self):
        """Setup test fixtures before each test method."""
        self.test_name = "sms_s"  # Short name to avoid POSIX name limits
        self.test_size = 1024

    def teardown_method(self):
        """Cleanup after each test method."""
        # Cleanup any leftover shared memory
        try:
            from multiprocessing.shared_memory import SharedMemory

            try:
                shm = SharedMemory(name=self.test_name)
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
        except ImportError:
            pass

    def test_shared_multistring_creation(self):
        """Test SharedMultiString creation."""
        sms = SharedMultiString(name=self.test_name, size=self.test_size, create=True)

        assert sms.shm is not None
        assert sms.shm.size >= self.test_size  # System may round up to page size
        # Check position in shared memory header
        pos = struct.unpack("I", sms.shm.buf[0:4])[0]
        assert pos == HEADER_SIZE

        # Check initial state
        assert not sms.is_ready()
        assert len(sms.get_list()) == 0  # No strings appended yet
        assert sms.get_list() == []

        sms.close()
        sms.unlink()

        with pytest.raises(FileNotFoundError):
            SharedMemory(name=self.test_name, create=False)

    @pytest.mark.parametrize("unlink_error", [FileNotFoundError(), PermissionError("denied")])
    def test_unlink_name_balances_tracker_after_unlink_failure(self, monkeypatch, unlink_error):
        """A failed unlink must undo the temporary resource-tracker registration."""
        calls = []
        tracker_name = f"/{self.test_name}"

        class MissingSharedMemory(SharedMemory):
            def __init__(self):
                self._name = tracker_name

            def unlink(self):
                raise unlink_error

            def close(self):
                pass

        sms = SharedMultiString.__new__(SharedMultiString)
        sms.shm = MissingSharedMemory()
        sms._name_unlinked = False
        monkeypatch.setattr(
            condition_module,
            "resource_tracker_register",
            lambda name, kind: calls.append(("register", name, kind)),
        )
        monkeypatch.setattr(
            condition_module,
            "resource_tracker_unregister",
            lambda name, kind: calls.append(("unregister", name, kind)),
        )

        if isinstance(unlink_error, FileNotFoundError):
            sms.unlink_name()
        else:
            with pytest.raises(PermissionError, match="denied"):
                sms.unlink_name()

        assert calls == [
            ("register", f"/{self.test_name}", "shared_memory"),
            ("unregister", f"/{self.test_name}", "shared_memory"),
        ]
        assert sms._name_unlinked is isinstance(unlink_error, FileNotFoundError)

    def test_append_single_string(self):
        """Test appending a single string."""
        sms = SharedMultiString(name=self.test_name, size=self.test_size, create=True)

        test_string = "hello_world"
        sms.append(test_string)

        # Check the string was added
        strings = sms.get_list()
        assert len(strings) == 1  # new string
        assert strings[0] == test_string

        # Check position updated correctly in shared memory
        pos = struct.unpack("I", sms.shm.buf[0:4])[0]
        expected_pos = HEADER_SIZE + len(test_string)
        assert pos == expected_pos

        sms.close()
        sms.unlink()

    def test_append_multiple_strings(self):
        """Test appending multiple strings."""
        sms = SharedMultiString(name=self.test_name, size=self.test_size, create=True)

        test_strings = ["first", "second", "third", "fourth"]

        for string in test_strings:
            sms.append(string)

        # Check all strings were added
        result_strings = sms.get_list()
        assert len(result_strings) == len(test_strings)

        for i, test_string in enumerate(test_strings):
            assert result_strings[i] == test_string

        sms.close()
        sms.unlink()

    def test_ready_flag_operations(self):
        """Test ready flag set/get operations."""
        sms = SharedMultiString(name=self.test_name, size=self.test_size, create=True)

        # Initially not ready
        assert not sms.is_ready()

        # Set ready
        sms.set_ready()
        assert sms.is_ready()

        # Should remain ready
        assert sms.is_ready()

        sms.close()
        sms.unlink()

    def test_memory_size_limit(self):
        """Test behavior when approaching memory size limit."""
        small_size = 64  # Small size to test limits
        sms = SharedMultiString(name=self.test_name, size=small_size, create=True)

        # Calculate available space using actual shm size (OS may round up)
        # Current position is already at HEADER_SIZE
        pos = struct.unpack("I", sms.shm.buf[0:4])[0]
        available_space = sms.shm.size - pos

        # Try to add a string that would exceed the limit
        large_string = "x" * (available_space + 10)

        with pytest.raises(ValueError, match="Shared memory is full"):
            sms.append(large_string)

        sms.close()
        sms.unlink()

    def test_first_string_no_separator(self):
        """Test that first string doesn't get a separator."""
        sms = SharedMultiString(name=self.test_name, size=self.test_size, create=True)

        first_string = "first"
        sms.append(first_string)

        # Check raw buffer to ensure no separator before first string
        content_start = HEADER_SIZE
        content_end = struct.unpack("I", sms.shm.buf[0:4])[0]
        raw_content = bytes(sms.shm.buf[content_start:content_end]).decode("utf-8")

        # Should start with first_string
        assert raw_content == first_string

        sms.close()
        sms.unlink()

    def test_header_structure(self):
        """Test the shared memory header structure."""
        sms = SharedMultiString(name=self.test_name, size=self.test_size, create=True)

        # Check initial header
        length_bytes = sms.shm.buf[0:4]
        ready_byte = sms.shm.buf[4:5]

        initial_length = struct.unpack("I", length_bytes)[0]
        initial_ready = struct.unpack("?", ready_byte)[0]

        assert initial_length == HEADER_SIZE  # Position starts at HEADER_SIZE
        assert not initial_ready

        # Add some content
        test_string = "test"
        sms.append(test_string)

        # Check updated header
        length_bytes = sms.shm.buf[0:4]
        updated_length = struct.unpack("I", length_bytes)[0]

        expected_length = HEADER_SIZE + len(test_string)
        assert updated_length == expected_length

        # Set ready and check
        sms.set_ready()
        ready_byte = sms.shm.buf[4:5]
        updated_ready = struct.unpack("?", ready_byte)[0]
        assert updated_ready

        # Clean up references to buffer views before closing
        del length_bytes, ready_byte

        sms.close()
        sms.unlink()

    def test_get_list_parsing(self):
        """Test string list parsing with various content."""
        sms = SharedMultiString(name=self.test_name, size=self.test_size, create=True)

        # Test with empty strings, special characters, etc.
        # Note: Cannot include "|" in strings since it's used as the delimiter
        test_cases = [
            "",
            "with spaces",
            "with-pipe-chars",  # Changed from "with|pipe|chars" since "|" is the delimiter
            "123456",
            "special!@#$%^&*()",
        ]

        for test_case in test_cases:
            sms.append(test_case)

        result_list = sms.get_list()

        # Check all test cases are present
        assert len(result_list) == len(test_cases) - 1
        for i, result in enumerate(result_list):
            assert result == test_cases[i + 1]

        sms.close()
        sms.unlink()

    def test_unicode_string_support(self):
        """Test support for unicode strings."""
        sms = SharedMultiString(name=self.test_name, size=self.test_size, create=True)

        unicode_strings = [
            "hello",
            "héllo",  # accented characters
            "你好",  # Chinese
            "🚀🌟",  # emojis
        ]

        for string in unicode_strings:
            sms.append(string)

        result_list = sms.get_list()

        # Check all unicode strings are preserved correctly
        for i, expected in enumerate(unicode_strings):
            assert result_list[i] == expected

        sms.close()
        sms.unlink()

    def test_connect_to_existing_memory(self):
        """Test connecting to existing shared memory."""
        # Create initial shared memory
        sms1 = SharedMultiString(name=self.test_name, size=self.test_size, create=True)
        sms1.append("test_string")
        sms1.set_ready()

        # Connect to existing memory
        sms2 = SharedMultiString(name=self.test_name, create=False)

        # Should see the same content
        assert sms2.is_ready()
        strings = sms2.get_list()
        assert "test_string" in strings

        sms2.close()
        sms1.close()
        sms1.unlink()

    @pytest.mark.parametrize("size", [64, 256, 1024, 4096])
    def test_different_memory_sizes(self, size):
        """Test with different memory sizes."""
        sms = SharedMultiString(name=self.test_name, size=size, create=True)

        # Use actual allocated size (OS may round up to page size)
        actual_size = sms.shm.size
        assert actual_size >= size

        # Calculate how many small strings we can fit using actual size
        pos = struct.unpack("I", sms.shm.buf[0:4])[0]
        available_space = actual_size - pos
        string_size = 10  # "test_XXX" + separator
        max_strings = available_space // string_size

        # Add strings up to near the limit
        for i in range(min(max_strings - 1, 50)):  # Cap at 50 for test performance
            sms.append(f"test_{i:03d}")

        # Should still be able to set ready
        sms.set_ready()
        assert sms.is_ready()

        sms.close()
        sms.unlink()
