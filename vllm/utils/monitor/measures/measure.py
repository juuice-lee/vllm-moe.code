"""
Measurement utilities for vLLM performance monitoring.
"""

from __future__ import annotations
from copy import deepcopy
import json
import pandas as pd
import numpy as np
from typing import Any, Optional
from vllm import LLM, SamplingParams
from vllm.entrypoints.chat_utils import ChatCompletionMessageParam

from vllm.utils.monitor.llm_monitor import LLMMonitor

class LatencyMeasure:
    def __init__(
        self,
        llm: LLM,
        module_names: Optional[list[str]] = None,
        save_json: bool = True,
        json_prefix: str = "latency_results",
    ):
        self.llm = llm
        self.monitor = LLMMonitor(llm)
        self.module_names = module_names
        self.save_json = save_json
        self.json_prefix = json_prefix

        # # Validaiton
        # if (
        #     self.llm.llm_engine.vllm_config.closer_config._closer_scheduler_variant
        #     != CloserSchedulerVariant.SARATHI
        # ):
        #     raise ValueError(
        #         "LatencyMeasure is only supported for V1 scheduler. "
        #         "Please set closer_scheduler_variant to 'v1' in the config."
        #     )

        # Register hooks
        # Get module names if not specified
        if module_names is None or len(module_names) == 0:
            all_modules = self.monitor.get_module_names()
            if all_modules and len(all_modules) > 0:
                module_names = all_modules[0]
            else:
                raise ValueError("No modules found in the model")

        print(f"Registering hooks on {len(module_names)} modules...")
        self.handles = self.monitor.register_latency_hooks(module_names)

    # def push_prompts(
    #     self, prompts: list[VLMRequest], sampling_params: SamplingParams
    # ) -> dict[str, Any]:
    #     self.llm.chat(format_vlm_messages(prompts), sampling_params)

    def get_intermediate_results(self) -> dict[str, Any]:
        """Get intermediate results without clearing the stored data."""
        # Collect results without popping
        result_dict = self.monitor.aggregate_async_latencies(pop=False)

        # Process results
        if result_dict and len(result_dict) > 0:
            return result_dict[0]
        else:
            return {}

    def aggregate_results(self, print_results: bool = True) -> dict[str, Any]:
        # Collect results
        print("Aggregating results...")
        result_dict = self.monitor.aggregate_async_latencies(pop=True)

        # Process results
        if result_dict and len(result_dict) > 0:
            rank_results = result_dict[0]

            if print_results:
                print_latency_results(rank_results)

            if self.save_json:
                save_results_to_json(
                    rank_results, f"{self.json_prefix}.json", format="grouped"
                )

            return rank_results
        else:
            print("No results collected")
            return {}

    @staticmethod
    def measure_latencies_once(
        llm: LLM,
        prompts: list[ChatCompletionMessageParam],
        sampling_params: Any,
        module_names: Optional[list[str]] = None,
        save_json: bool = True,
        json_prefix: str = "latency_results",
        print_results: bool = True,
    ) -> dict[str, Any]:
        """
        Measure latencies and input shapes for specified modules during generation.

        WARNING(shlee): This is an all-in-one function.
        If you want to measure latencies for multiple iterations,
        you should call another function.

        Args:
            llm: Initialized LLM instance
            prompts: List of prompts to generate from
            sampling_params: SamplingParams for generation
            module_names: List of module names to monitor (None = all modules)
            save_json: Whether to save results to JSON files
            json_prefix: Prefix for JSON output files
            print_results: Whether to print results to console

        Returns:
            Dictionary containing the measurement results
        """
        # Create monitor
        monitor = LLMMonitor(llm)

        # Get module names if not specified
        if module_names is None:
            all_modules = monitor.get_module_names()
            if all_modules and len(all_modules) > 0:
                module_names = all_modules[0]
            else:
                raise ValueError("No modules found in the model")

        # Register hooks
        print(f"Registering hooks on {len(module_names)} modules...")
        handles = monitor.register_latency_hooks(module_names)

        # Run generation
        print(f"Running generation with {len(prompts)} prompts...")
        outputs = llm.chat(prompts, sampling_params)

        # Collect results
        print("Aggregating results...")
        result_dict = monitor.aggregate_async_latencies()

        # Process results
        if result_dict and len(result_dict) > 0:
            rank_results = result_dict[0]
            
            if print_results:
                print_latency_results(rank_results)

            if save_json:
                # save_results_to_json(
                #     rank_results, f"{json_prefix}_flat.json", format="flat"
                # )
                save_results_to_json(
                    rank_results, f"{json_prefix}_grouped.json", format="grouped"
                )

            return rank_results
        else:
            print("No results collected")
            return {}

class MoEMeasure:
    def __init__(
        self,
        llm: LLM,
        module_names: Optional[list[str]] = None,
        save_json: bool = True,
        json_prefix: str = "moe_results",
        hook_expert_score: bool = False,
        moe_gate_name: str = "gate",
    ):
        self.llm = llm
        self.monitor = LLMMonitor(llm)
        self.module_names = module_names
        self.save_json = save_json
        self.json_prefix = json_prefix
        self.hook_expert_score = hook_expert_score

        # Register hooks
        # Get module names if not specified
        if module_names is None or len(module_names) == 0:
            all_modules = self.monitor.get_module_names()
            if all_modules and len(all_modules) > 0:
                module_names = all_modules[0]
            else:
                raise ValueError("No modules found in the model")

        print(f"Registering hooks on {len(module_names)} modules...")
        if hook_expert_score:
            self.handles = self.monitor.register_moe_hooks(module_names, moe_gate_name)
        else:
            self.handles = self.monitor.register_latency_hooks(module_names)
        
    def get_intermediate_results(self) -> dict[str, Any]:
        """Get intermediate results without clearing the stored data."""
        # Collect results without popping
        if self.hook_expert_score:
            result_dict = self.monitor.aggregate_async_moe_results(pop=False)
        else:
            result_dict = self.monitor.aggregate_async_latencies(pop=False)
        # Process results
        if result_dict and len(result_dict) > 0:
            return result_dict[0]
        else:
            return {}

    def aggregate_results(self, print_results: bool = True) -> dict[str, Any]:
        # Collect results
        print("Aggregating results...")
        if self.hook_expert_score:
            result_list = self.monitor.aggregate_async_moe_results()
        else:
            result_list = self.monitor.aggregate_async_latencies()
            
        # Process results
        if result_list and len(result_list) > 0:
            rank_results = result_list[0]

            if print_results:
                print_latency_results(rank_results)

            if self.save_json:
                save_all_gpus_results_to_json(
                    result_list, f"{self.json_prefix}.json", format="grouped"
                )

            return result_list
        else:
            print("No results collected")
            return {}

    @staticmethod
    def measure_latencies_once(
        llm: LLM,
        prompts: list[ChatCompletionMessageParam],
        sampling_params: Any,
        module_names: Optional[list[str]] = None,
        save_json: bool = True,
        json_prefix: str = "latency_results",
        print_results: bool = True,
        moe_gate_name: str = "gate", # the word specifing gate module name 
    ) -> dict[str, Any]:
        """
        Measure latencies and input shapes for specified modules during generation.

        WARNING(shlee): This is an all-in-one function.
        If you want to measure latencies for multiple iterations,
        you should call another function.

        Args:
            llm: Initialized LLM instance
            prompts: List of prompts to generate from
            sampling_params: SamplingParams for generation
            module_names: List of module names to monitor (None = all modules)
            save_json: Whether to save results to JSON files
            json_prefix: Prefix for JSON output files
            print_results: Whether to print results to console

        Returns:
            Dictionary containing the measurement results
        """
        # Create monitor
        monitor = LLMMonitor(llm)

        # Get module names if not specified
        if module_names is None:
            all_modules = monitor.get_module_names()
            if all_modules and len(all_modules) > 0:
                module_names = all_modules[0]
            else:
                raise ValueError("No modules found in the model")

        # Register hooks
        print(f"Registering hooks on {len(module_names)} modules...")
        handles = monitor.register_moe_hooks(module_names, moe_gate_name)

        # Run generation
        print(f"Running generation with {len(prompts)} prompts...")
        outputs = llm.chat(prompts, sampling_params)

        # Collect results
        print("Aggregating results...")
        result_list = monitor.aggregate_async_moe_results()

        # Process results
        if result_list and len(result_list) > 0:
            rank_results = result_list[0]
            
            if print_results:
                print_latency_results(rank_results)

            if save_json:
                # save_results_to_json(
                #     rank_results, f"{json_prefix}_flat.json", format="flat"
                # )
                save_all_gpus_results_to_json(
                    result_list, f"{json_prefix}_grouped.json", format="grouped"
                )

            return rank_results
        else:
            print("No results collected")
            return {}
        
        


def print_latency_results(rank_results: dict[str, Any]):
    """Print latency results in a formatted way."""
    print("\n=== Module Latencies and Input Shapes (Paired) ===")

    for module_name, data in rank_results.items():
        print(f"\nModule: {module_name}")

        if isinstance(data, dict) and "paired_results" in data:
            print(f"  Forward passes: {len(data['paired_results'])}")

            for i, pair in enumerate(data["paired_results"]):
                print(f"\n  Pass #{i + 1}:")
                print(f"    Latency: {pair['latency_ms']:.3f} ms")

                if pair["input_shapes"]:
                    print(f"    Input shapes:")
                    for param_name, shape in pair["input_shapes"].items():
                        print(f"      - {param_name}: {shape}")
                else:
                    print(f"    Input shapes: N/A")

                if pair.get("metadata"):
                    print(f"    Metadata:")
                    for key, value in pair["metadata"].items():
                        print(f"      - {key}: {value}")

    # Create summary table
    print("\n\n=== Summary Table ===")
    summary_data = []

    for module_name, data in rank_results.items():
        if isinstance(data, dict) and "paired_results" in data:
            for i, pair in enumerate(data["paired_results"]):
                row = {
                    "Module": module_name.split(".")[-1]
                    if "." in module_name
                    else module_name,
                    "Pass": i + 1,
                    "Latency (ms)": f"{pair['latency_ms']:.3f}",
                }

                # Add shape info
                if pair["input_shapes"]:
                    for param_name, shape in pair["input_shapes"].items():
                        row[f"{param_name}_shape"] = str(shape)

                # Add metadata info
                if pair.get("metadata"):
                    for key, value in pair["metadata"].items():
                        row[f"metadata_{key}"] = str(value)

                summary_data.append(row)

    # if summary_data:
    #     df = pd.DataFrame(summary_data)
    #     print(df.to_string(index=False))


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_results_to_json(
    rank_results: dict[str, Any],
    filename: str = "latency_results.json",
    format: str = "grouped",
):
    """
    Save the latency and input shape results to a JSON file.

    Args:
        rank_results: The results dictionary from monitor.aggregate_async_latencies()[0]
        filename: Output filename for the JSON file
        format: "flat" for a list of entries, or "grouped" for grouping by module
    """
    if format == "flat":
        json_results = []

        for module_name, data in rank_results.items():
            if isinstance(data, dict) and "paired_results" in data:
                for i, pair in enumerate(data["paired_results"]):
                    result_entry = {
                        "module_name": module_name,
                        "pass_number": i + 1,
                        "latency_ms": pair["latency_ms"],
                        "input_shapes": pair["input_shapes"]
                        if pair["input_shapes"]
                        else {},
                        "metadata": pair.get("metadata", {}),
                    }
                    json_results.append(result_entry)

    elif format == "grouped":
        json_results = {}

        for module_name, data in rank_results.items():
            if isinstance(data, dict) and "paired_results" in data:
                module_results = []
                for i, pair in enumerate(data["paired_results"]):
                    result_entry = {
                        "pass_number": i + 1,
                        "latency_ms": pair["latency_ms"],
                        "input_shapes": pair["input_shapes"]
                        if pair["input_shapes"]
                        else {},
                        "metadata": pair.get("metadata", {}),
                    }
                    module_results.append(result_entry)
                json_results[module_name] = module_results

    else:
        raise ValueError(f"Unknown format: {format}. Use 'flat' or 'grouped'")

    # Save to JSON file with custom encoder
    with open(filename, "w") as f:
        json.dump(json_results, f, indent=2, cls=NumpyEncoder)

    print(f"\nResults saved to {filename} (format: {format})")
    return json_results

def save_all_gpus_results_to_json(
    results: list[dict[str, Any]],
    filename: str = "latency_results.json",
    format: str = "grouped",
):
    """
    Save the latency and input shape results to a JSON file.

    Args:
        results: The results dictionary from monitor.aggregate_async_latencies()
        filename: Output filename for the JSON file
        format: "flat" for a list of entries, or "grouped" for grouping by module
    """
    if format == "flat":
        all_gpus_results=[]
        for rank_results in results:
            json_results = []

            for module_name, data in rank_results.items():
                if isinstance(data, dict) and "paired_results" in data:
                    for i, pair in enumerate(data["paired_results"]):
                        result_entry = {
                            "module_name": module_name,
                            "pass_number": i + 1,
                            "latency_ms": pair["latency_ms"],
                            "input_shapes": pair["input_shapes"]
                            if pair["input_shapes"]
                            else {},
                            "metadata": pair.get("metadata", {}),
                        }
                        if "expert_scores" in pair.keys():
                            # TODO deserialize해야 한다.
                            result_entry["expert_scores"] = pair["expert_scores"]
                        json_results.append(result_entry)
            all_gpus_results.append(json_results)

    elif format == "grouped":
        all_gpus_results = {}
        for gpu_idx, rank_results in enumerate(results):
            json_results = {}

            for module_name, data in rank_results.items():
                if isinstance(data, dict) and "paired_results" in data:
                    module_results = []
                    for i, pair in enumerate(data["paired_results"]):
                        result_entry = {
                            "pass_number": i + 1,
                            "latency_ms": pair["latency_ms"],
                            "input_shapes": pair["input_shapes"]
                            if pair["input_shapes"]
                            else {},
                            "metadata": pair.get("metadata", {}),
                        }
                        if "expert_scores" in pair.keys():
                            result_entry["expert_scores"] = pair["expert_scores"]
                        module_results.append(result_entry)
                    json_results[module_name] = module_results
            all_gpus_results[f"GPU_{gpu_idx}"] = json_results

    else:
        raise ValueError(f"Unknown format: {format}. Use 'flat' or 'grouped'")

    # Save to JSON file with custom encoder
    with open(filename, "w") as f:
        json.dump(all_gpus_results, f, indent=2, cls=NumpyEncoder)

    print(f"\nResults saved to {filename} (format: {format})")
    return all_gpus_results


def filter_modules_by_pattern(module_names: list[str], pattern: str) -> list[str]:
    """Filter module names by a pattern."""
    return [name for name in module_names if pattern in name]


def get_layer_modules(module_names: list[str], layer_idx: int) -> list[str]:
    """Get all modules for a specific layer index."""
    return [name for name in module_names if f".layers.{layer_idx}." in name]


# utils/merge.py
"""
Merge latency-measurement checkpoints **that store only `paired_results`.**

Each JSON chunk produced by `aggregate_async_latencies()` is expected to look
like this now:

{
  "module.name": {
      "paired_results": [
          {"latency_ms": 0.42,
           "input_shapes": {...} | null,
           "metadata": {...} | null},
          ...
      ]
  },
  ...
}

The helper below concatenates `paired_results` for every module while
leaving all other fields untouched (if they ever appear).
"""


def _concat(a: list[Any] | None, b: list[Any] | None) -> list[Any]:
    """Robust list concatenation that handles None gracefully."""
    if a and b:
        return a + b
    return list(a or b or [])


def merge_grouped_json(
    baseline: dict[str, dict[str, Any]],
    add: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Parameters
    ----------
    baseline : dict
        Already-collected measurements.
    add : dict
        Newly aggregated measurements to merge.

    Returns
    -------
    merged : dict
        Same schema, with `paired_results` lists concatenated per module.
    """
    merged: dict[str, dict[str, Any]] = deepcopy(baseline)

    for mod, add_entry in add.items():
        if mod not in merged:
            merged[mod] = {"paired_results": list(add_entry["paired_results"])}
            continue

        merged_pairs = _concat(
            merged[mod].get("paired_results"), add_entry.get("paired_results")
        )

        # Preserve **only** paired_results; ignore any legacy keys.
        merged[mod] = {"paired_results": merged_pairs}

    return merged
