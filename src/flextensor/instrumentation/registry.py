# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Instrumentation registry for capturing component initialization arguments.

This module provides a thread-safe singleton registry that stores component
init args for debugging and analysis purposes.
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ComponentRecord:
    """Record of a component's initialization.

    Attributes:
        class_name: Name of the component class.
        module_path: Full module path of the component.
        init_timestamp: ISO format timestamp when component was initialized.
        args: Dictionary of initialization arguments.
    """

    class_name: str
    module_path: str
    init_timestamp: str
    args: dict[str, Any] = field(default_factory=dict)


class InstrumentationRegistry:
    """Thread-safe singleton registry for component initialization data.

    This registry stores initialization arguments from decorated components,
    enabling debugging and analysis of component configurations.

    Example:
        >>> registry = InstrumentationRegistry.get_instance()
        >>> registry.register("MyClass", "my_module.MyClass", {"arg1": "value1"})
        >>> records = registry.get_records()
    """

    _instance: "InstrumentationRegistry | None" = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool

    def __new__(cls) -> "InstrumentationRegistry":
        """Create or return the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    instance = super().__new__(cls)
                    object.__setattr__(instance, "_initialized", False)
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Initialize the registry (only once for singleton)."""
        if self._initialized:
            return
        self._records: list[ComponentRecord] = []
        self._records_lock = threading.Lock()
        self._enabled = False
        self._initialized = True

    @classmethod
    def get_instance(cls) -> "InstrumentationRegistry":
        """Get the singleton registry instance.

        Returns:
            The singleton InstrumentationRegistry instance.
        """
        return cls()

    @property
    def enabled(self) -> bool:
        """Check if instrumentation is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable instrumentation."""
        self._enabled = value

    def register(
        self,
        class_name: str,
        module_path: str,
        args: dict[str, Any],
    ) -> None:
        """Register a component's initialization arguments.

        Args:
            class_name: Name of the component class.
            module_path: Full module path of the component.
            args: Dictionary of initialization arguments.
        """
        if not self._enabled:
            return

        record = ComponentRecord(
            class_name=class_name,
            module_path=module_path,
            init_timestamp=datetime.now().isoformat(),
            args=args,
        )

        with self._records_lock:
            self._records.append(record)

    def get_records(self) -> list[ComponentRecord]:
        """Get all registered component records.

        Returns:
            List of ComponentRecord instances.
        """
        with self._records_lock:
            return list(self._records)

    def clear(self) -> None:
        """Clear all registered records."""
        with self._records_lock:
            self._records.clear()

    def __len__(self) -> int:
        """Return the number of registered records."""
        with self._records_lock:
            return len(self._records)
