"""
Utilities for loading only selected layers from models.

This module provides a simple, reusable way to load only specific layers
from encoder or decoder models for efficient testing and experimentation.

IMPORTANT: Always use get_selected_layer_loader() to obtain a layer loader.
This ensures consistent environment variable naming across all models.

Supported environment variables:
- VLLM_NUM_{layer_type}_HIDDEN_LAYERS_OVERRIDE: Load first N layers
- VLLM_SELECTED_{layer_type}_LAYERS: Load specific layers by index

Where {layer_type} is replaced with ENCODER, DECODER, etc.

Example:
    # Get a layer loader for encoder layers
    loader = get_selected_layer_loader("encoder", total_layers=24)

    # Get a layer loader for decoder layers
    loader = get_selected_layer_loader("decoder", total_layers=32)
"""

import os
from typing import Optional, List, Dict, Tuple, Literal
from vllm.logger import init_logger, Colors

logger = init_logger(__name__)

# VLLM_NUM_ENCODER_HIDDEN_LAYERS_OVERRIDE
# VLLM_NUM_DECODER_HIDDEN_LAYERS_OVERRIDE
# VLLM_SELECTED_ENCODER_LAYERS
# VLLM_SELECTED_DECODER_LAYERS


class SelectedLayerLoader:
    """Helper class for loading only selected layers from a model."""

    def __init__(
        self,
        layer_type: Literal["encoder", "decoder"],  # "encoder" or "decoder"
        total_layers: int,
        override_env_var: Optional[str] = None,
        selected_env_var: Optional[str] = None,
    ):
        """
        Initialize the selected layer loader.

        Args:
            layer_type: Type of layers ("encoder" or "decoder")
            total_layers: Total number of layers in the full model
            override_env_var: Environment variable for layer count override
            selected_env_var: Environment variable for selected layers
        """
        self.layer_type = layer_type
        self.total_layers = total_layers
        self.override_env_var = (
            override_env_var
            or f"VLLM_NUM_{layer_type.upper()}_HIDDEN_LAYERS_OVERRIDE"
        )
        self.selected_env_var = (
            selected_env_var or f"VLLM_SELECTED_{layer_type.upper()}_LAYERS"
        )

        # Parse configuration
        self.num_layers, self.layer_indices, self.is_selected_mode = (
            self._parse_config()
        )

        logger.info(f"Selected {self.layer_type} layers: {self.layer_indices}")

        # Create mappings
        self.layer_idx_mapping = self._create_forward_mapping()
        self.reverse_mapping = self._create_reverse_mapping()

    def _parse_config(self) -> Tuple[int, Optional[List[int]], bool]:
        """Parse environment variables and return configuration."""
        override_value = (
            os.environ.get(self.override_env_var)
            if self.override_env_var
            else None
        )
        selected_value = (
            os.environ.get(self.selected_env_var)
            if self.selected_env_var
            else None
        )

        # Validate not both set
        if override_value and selected_value:
            raise ValueError(
                f"Cannot use both {self.override_env_var} and {self.selected_env_var} "
                f"simultaneously. Please use only one."
            )

        if selected_value:
            # Parse selected layers
            try:
                layer_indices = [
                    int(idx.strip()) for idx in selected_value.split(",")
                ]
                # Validate indices
                for idx in layer_indices:
                    if idx < 0 or idx >= self.total_layers:
                        raise ValueError(
                            f"Invalid layer index {idx} in {self.selected_env_var}. "
                            f"Must be between 0 and {self.total_layers - 1}"
                        )
                # Sort to maintain order
                layer_indices = sorted(
                    set(layer_indices)
                )  # Remove duplicates and sort
                num_layers = len(layer_indices)

                logger.infowc(
                    Colors.BLUE,
                    f"Loading only selected {self.layer_type} layers: {layer_indices} "
                    f"(out of {self.total_layers} total layers)",
                )
                return num_layers, layer_indices, True

            except ValueError as e:
                raise ValueError(
                    f"Invalid {self.selected_env_var} format. "
                    f"Expected comma-separated integers, got: {selected_value}"
                ) from e

        elif override_value:
            # Use override count
            num_layers = int(override_value)
            logger.info(
                f"Using {num_layers} {self.layer_type} layers instead of {self.total_layers}"
            )
            return num_layers, None, False

        else:
            # Use all layers
            return self.total_layers, None, False

    def _create_forward_mapping(self) -> Dict[int, int]:
        """Create mapping from module index to original layer index."""
        if self.layer_indices is not None:
            # Selected mode: map sequential indices to selected layer indices
            return {
                i: layer_idx for i, layer_idx in enumerate(self.layer_indices)
            }
        else:
            # Normal mode: identity mapping
            return {i: i for i in range(self.num_layers)}

    def _create_reverse_mapping(self) -> Dict[int, int]:
        """Create mapping from original layer index to module index."""
        return {v: k for k, v in self.layer_idx_mapping.items()}

    def get_num_layers(self) -> int:
        """Get the number of layers to create."""
        return self.num_layers

    def get_layer_index(self, module_idx: int) -> int:
        """Get the original layer index for a module index."""
        return self.layer_idx_mapping.get(module_idx, module_idx)

    def should_load_weight(
        self, weight_name: str, layer_idx_in_name: int
    ) -> bool:
        """
        Check if a weight should be loaded based on layer index.

        Args:
            weight_name: Full name of the weight
            layer_idx_in_name: Layer index extracted from the weight name

        Returns:
            True if the weight should be loaded, False otherwise
        """
        if self.is_selected_mode:
            # In selected mode, only load if layer is in our selected set
            should_load = layer_idx_in_name in self.reverse_mapping
            if not should_load:
                logger.debug(
                    f"Skipping {self.layer_type} weight {weight_name} "
                    f"(layer {layer_idx_in_name} not in selected layers)"
                )
            return should_load
        else:
            # In override mode, load if within range
            should_load = layer_idx_in_name < self.num_layers
            if not should_load:
                logger.debug(
                    f"Skipping {self.layer_type} weight {weight_name} "
                    f"(layer {layer_idx_in_name} >= {self.num_layers})"
                )
            return should_load

    def remap_layer_prefix(self, prefix: str) -> str:
        """
        Remap a layer prefix to use the correct layer index.

        Args:
            prefix: Original prefix like "model.layers.0"

        Returns:
            Remapped prefix with correct layer index
        """
        if not self.is_selected_mode:
            return prefix

        # Extract parts and layer index
        parts = prefix.split(".")
        for i, part in enumerate(parts):
            if part.isdigit():
                module_idx = int(part)
                if module_idx in self.layer_idx_mapping:
                    parts[i] = str(self.layer_idx_mapping[module_idx])
                break

        return ".".join(parts)
    
    def is_using_selected_layers(self) -> bool:
        """Check if we're using selected layers (not all layers)."""
        return self.is_selected_mode or self.num_layers < self.total_layers


def get_selected_layer_loader(
    layer_type: str, total_layers: int
) -> SelectedLayerLoader:
    """
    Create a layer loader for selected layer loading.

    This is the RECOMMENDED way to get a layer loader. It automatically sets up
    the standard environment variable names for the given layer type.

    Args:
        layer_type: "encoder" or "decoder"
        total_layers: Total number of layers in the full model

    Returns:
        Layer loader instance
    """
    return SelectedLayerLoader(layer_type, total_layers)


def extract_layer_index(name: str, layer_key: str = "layers") -> Optional[int]:
    """
    Extract layer index from a parameter name.

    Args:
        name: Parameter name like "model.layers.5.self_attn.qkv"
        layer_key: Key to look for (default: "layers")

    Returns:
        Layer index if found, None otherwise
    """
    parts = name.split(".")
    try:
        key_idx = parts.index(layer_key)
        if key_idx + 1 < len(parts) and parts[key_idx + 1].isdigit():
            return int(parts[key_idx + 1])
    except (ValueError, IndexError):
        pass
    return None
