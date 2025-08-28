"""
LLMMonitor: Orchestrator-level monitoring interface for multi-GPU inference.

This module provides the main interface for registering hooks and aggregating
data across all GPU workers in a distributed inference system.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import pandas as pd
from collections import defaultdict
import warnings
from torch.utils.hooks import RemovableHandle

from vllm import LLM
from vllm.utils.monitor.worker_store import WorkerStore
from transformers import PreTrainedTokenizerBase
from vllm.v1.core.sched.output import SchedulerOutput

class LLMMonitor:
    """
    Monitor class for aggregating data and managing hooks across GPU workers.

    This class provides:
    - Dynamic hook registration across all workers
    - Data aggregation from all WorkerStore instances
    - Thread-safe operations for concurrent access
    """

    def __init__(self, llm: Optional[LLM] = None):
        """
        Initialize the LLM Monitor.

        Args:
            llm: The LLM instance to monitor. If None, manual worker registration is required.
        """
        self.llm = llm

    def collective_rpc(self, method: str, *args, **kwargs) -> list[Any]:
        """Call a method on all workers."""
        # Cache method existence results
        if not hasattr(self, "_method_cache"):
            self._method_cache = {}

        if method not in self._method_cache:
            method_exists: list[bool] = self.llm.llm_engine.collective_rpc(
                "check_method_exists", args=(method,)
            )
            self._method_cache[method] = any(method_exists)

        if not self._method_cache[method]:
            raise ValueError(
                f"Method {method} does not exist in the LLM engine."
            )

        # WARNING(shlee): llm monitor의 function들은 2D list (rank -> result)를 반환한다.
        # 따라서 결과를 출력하기 전에 1D list로 변환해야 한다.
        return self.llm.llm_engine.collective_rpc(
            method, args=args, kwargs=kwargs
        )
    
    def get_scheduled_outputs(self) -> list[list[SchedulerOutput]]:
        """Get the scheduled outputs from all workers."""
        return self.collective_rpc("get_scheduled_outputs")
    
    def get_module_names(self) -> list[str]:
        """Get the names of all modules in the model."""
        return self.collective_rpc("get_module_names")

    def get_worker_store_rank(self) -> list[Tuple[Optional[int], Optional[int]]]:
        """Get the device and worker IDs for each worker."""
        return self.collective_rpc("get_worker_store_rank")
    
    def get_worker_store_dict(self) -> dict[int, dict[str, Any]]:
        """Get the dictionary of worker store data."""
        return self.collective_rpc("get_worker_store_dict")

    def register_latency_hooks(
        self, module_names: list[str], tokenizer: Optional[PreTrainedTokenizerBase]=None,
    ) -> dict[str, tuple[RemovableHandle, RemovableHandle]]:
        """Register latency hooks on the specified modules."""
        return self.collective_rpc("register_latency_hooks", module_names, tokenizer)
    
    def register_moe_hooks(
        self, module_names: list[str], moe_gate_name: str="gate"
    ):
        """Register latency and moe expert score hooks on the specified modules"""
        return self.collective_rpc("register_moe_hooks", module_names, moe_gate_name)
    
    def register_slo_hooks(
        self, module_names: list[str]
    ):
        """Register SLO hooks on the specified modules."""
        return self.collective_rpc("register_slo_hooks", module_names)

    def aggregate_async_latencies(
        self, module_names: list[str] | None = None, pop: bool = False
    ) -> dict[str, list[float]]:
        """Aggregate latency data from all workers.
        CAUTION! latency unit is milliseconds (ms).
        """
        return self.collective_rpc(
            "aggregate_async_latencies", module_names, pop
        )
    
    def aggregate_async_moe_results(
        self, module_names: Optional[list[str]] = None, pop: bool = False
    ):
        """Aggregate latency and expert score from all workers.
        CAUSTION! latency unit is milliseconds (ms).
        """
        return self.collective_rpc(
            "aggregate_async_moe_results", module_names, pop
        )

    def _convert_to_dataframe(self, aggregated: Dict) -> pd.DataFrame:
        """Convert aggregated data to pandas DataFrame."""
        rows = []

        # Convert regular data
        for key, values in aggregated["data"].items():
            for item in values:
                rows.append(
                    {
                        "type": "data",
                        "key": key,
                        "worker_id": item["worker_id"],
                        "value": str(item["value"])[
                            :100
                        ],  # Truncate for display
                        "module": None,
                    }
                )

        # Convert hook data
        for module_name, module_data in aggregated["hook_data"].items():
            for data_key, data_list in module_data.items():
                for item in data_list:
                    rows.append(
                        {
                            "type": "hook",
                            "key": data_key,
                            "worker_id": item["worker_id"],
                            "value": str(item["data"])[
                                :100
                            ],  # Truncate for display
                            "module": module_name,
                        }
                    )

        return pd.DataFrame(rows)
