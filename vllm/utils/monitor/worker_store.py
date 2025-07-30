"""
WorkerStore: Per-GPU singleton data store for inference monitoring.

This module provides a singleton store for each GPU worker process to collect
and store statistics, activations, and other intermediate data during inference.
"""

from typing import Any, Optional
import torch
import numpy as np
from collections import defaultdict
import pickle

from vllm.utils.monitor.singleton_base import BaseSingleTon


class WorkerStore(BaseSingleTon):
    """
    Per-GPU singleton data store for collecting inference statistics.

    This class acts as an in-memory key-value store for any statistics or
    intermediate tensors produced during inference. Each GPU worker process
    has exactly one instance.
    """

    def init(self):
        """Initialize the worker store (called once per singleton)."""
        self._store: dict[str, Any] = {}
        self._device_id: Optional[int] = None
        self._worker_id: Optional[int] = None

        self._is_capturing_latency: bool = False

        # Store for hook data - organized by module name and then by key
        self._hook_data: dict[str, dict[str, list[Any]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # Metadata about stored tensors
        self._tensor_metadata: dict[str, dict[str, Any]] = {}

    @property
    def is_capturing_latency(self) -> bool:
        """Check if the worker store is capturing latency."""
        return self._is_capturing_latency

    def set_device_info(self, device_id: int, worker_id: Optional[int] = None):
        """Set the device and worker information for this store."""
        self._device_id = device_id
        self._worker_id = worker_id

    @property
    def device_id(self) -> Optional[int]:
        """Get the GPU device ID associated with this store."""
        return self._device_id

    @property
    def worker_id(self) -> Optional[int]:
        """Get the worker ID associated with this store."""
        return self._worker_id

    def put(
        self, key: str, value: Any, metadata: Optional[dict[str, Any]] = None
    ):
        """
        Store a value with an optional metadata.

        Args:
            key: The key to store the value under
            value: The value to store (can be tensor, list, dict, etc.)
            metadata: Optional metadata about the stored value
        """
        self._store[key] = value
        if metadata:
            self._tensor_metadata[key] = metadata

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value by key."""
        return self._store.get(key, default)

    def update(self, key: str, value: Any, aggregation: str = "append"):
        """
        Update a value with aggregation strategy.

        Args:
            key: The key to update
            value: The new value
            aggregation: How to aggregate ("append", "sum", "mean", "replace")
        """
        if key not in self._store:
            if aggregation == "append":
                self._store[key] = [value]
            else:
                self._store[key] = value
        else:
            if aggregation == "append":
                if not isinstance(self._store[key], list):
                    self._store[key] = [self._store[key]]
                self._store[key].append(value)
            elif aggregation == "sum":
                self._store[key] = self._store[key] + value
            elif aggregation == "mean":
                # For mean, we need to track count
                count_key = f"{key}_count"
                if count_key not in self._store:
                    self._store[count_key] = 1
                    self._store[key] = value
                else:
                    count = self._store[count_key]
                    self._store[key] = (self._store[key] * count + value) / (
                        count + 1
                    )
                    self._store[count_key] = count + 1
            elif aggregation == "replace":
                self._store[key] = value

    def record_hook_data(self, module_name: str, data_key: str, data: Any):
        """
        Record data from a hook.

        Args:
            module_name: Name of the module where hook was registered
            data_key: Key for the type of data (e.g., "activations", "gradients")
            data: The actual data to store
        """
        # Convert tensors to CPU to avoid memory issues
        if isinstance(data, torch.Tensor):
            if data.dtype == torch.bfloat16:
                data = data.to(torch.float32)
            data = data.detach().cpu().numpy().tolist()
        self._hook_data[module_name][data_key].append(data)

    def get_hook_data(
        self, module_name: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Get hook data, optionally filtered by module name.

        Args:
            module_name: If provided, only return data for this module

        Returns:
            Dictionary of hook data
        """
        if module_name:
            return dict(self._hook_data.get(module_name, {}))
        else:
            return {k: dict(v) for k, v in self._hook_data.items()}

    def clear(self, keys: Optional[list[str]] = None):
        """
        Clear stored data.

        Args:
            keys: If provided, only clear these keys. Otherwise clear all.
        """
        if keys:
            for key in keys:
                self._store.pop(key, None)
                self._tensor_metadata.pop(key, None)
        else:
            self._store.clear()
            self._tensor_metadata.clear()
            self._hook_data.clear()

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of stored data without the actual values."""
        summary = {
            "device_id": self._device_id,
            "worker_id": self._worker_id,
            "num_keys": len(self._store),
            "keys": list(self._store.keys()),
            "hook_modules": list(self._hook_data.keys()),
            "total_size_bytes": sum(
                self._get_size(v) for v in self._store.values()
            ),
        }
        return summary

    def _get_size(self, obj: Any) -> int:
        """Estimate the size of an object in bytes."""
        if isinstance(obj, torch.Tensor):
            return obj.element_size() * obj.nelement()
        elif isinstance(obj, np.ndarray):
            return obj.nbytes
        elif isinstance(obj, (list, tuple)):
            return sum(self._get_size(item) for item in obj)
        elif isinstance(obj, dict):
            return sum(
                self._get_size(k) + self._get_size(v) for k, v in obj.items()
            )
        else:
            try:
                return len(pickle.dumps(obj))
            except:
                return 0

    def to_dict(self) -> dict[str, Any]:
        """Export all data as a dictionary."""
        return {
            "device_id": self._device_id,
            "worker_id": self._worker_id,
            "store": dict(self._store),
            "hook_data": {k: dict(v) for k, v in self._hook_data.items()},
            "metadata": dict(self._tensor_metadata),
        }
