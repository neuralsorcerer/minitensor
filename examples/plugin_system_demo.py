# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Minitensor Plugin System Demonstration

This example shows how to create, register, and use custom plugins
with the minitensor library.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "minitensor"))

import numpy as np

import minitensor as mt

try:  # pragma: no cover - optional feature
    import minitensor.plugins as plugins
except Exception as exc:  # pragma: no cover - optional feature
    print(f"Plugin system unavailable: {exc}")
    sys.exit(0)


def demo_version_info():
    """Demonstrate version information and compatibility checking"""
    print("=== Version Information Demo ===")

    # Get current minitensor version
    current = plugins.VersionInfo.current()
    print(f"Current minitensor version: {current}")

    # Create version info objects
    v1 = plugins.VersionInfo(1, 0, 0)
    v2 = plugins.VersionInfo(1, 2, 0)
    v3 = plugins.VersionInfo(2, 0, 0)

    print(f"Version 1: {v1}")
    print(f"Version 2: {v2}")
    print(f"Version 3: {v3}")

    # Test compatibility
    print(f"v2 compatible with v1: {v2.is_compatible_with(v1)}")
    print(f"v1 compatible with v2: {v1.is_compatible_with(v2)}")
    print(f"v3 compatible with v1: {v3.is_compatible_with(v1)}")

    # Parse version from string
    parsed = plugins.VersionInfo.parse("1.5.2")
    print(f"Parsed version: {parsed}")
    print()


def create_math_operations():
    """Create custom mathematical operations"""

    # Custom operation builder is not exposed in the current Python API.
    # Return an empty list so the rest of the demo can run without errors.

    return []


def create_math_plugin():
    """Create a plugin with mathematical operations"""

    # Create plugin info
    plugin = (
        plugins.PluginBuilder()
        .name("math_extensions")
        .version(plugins.VersionInfo(1, 0, 0))
        .description("Additional mathematical operations for minitensor")
        .author("Plugin Demo Team")
        .min_minitensor_version(plugins.VersionInfo(0, 1, 0))
        .build()
    )

    # Define plugin functions
    def initialize(registry):
        print("Math extensions plugin initialized!")
        return True

    def cleanup(registry):
        print("Math extensions plugin cleaned up!")
        return True

    # Set plugin functions
    plugin.set_initialize_fn(initialize)
    plugin.set_cleanup_fn(cleanup)

    return plugin


class CustomNormalizationLayer(plugins.CustomLayer):
    """Custom layer that performs layer normalization with learnable parameters"""

    def __new__(cls, normalized_shape, eps=1e-5):
        return super(CustomNormalizationLayer, cls).__new__(cls, "custom_layer_norm")

    def __init__(self, normalized_shape, eps=1e-5):
        self.normalized_shape = normalized_shape
        self.eps = eps

        # Initialize learnable parameters
        self.add_parameter("weight", mt.ones(normalized_shape))
        self.add_parameter("bias", mt.zeros(normalized_shape))

    def forward(self, inputs):
        """Forward pass of layer normalization"""
        x = inputs[0]

        # Get parameters
        weight = self.get_parameter("weight")
        bias = self.get_parameter("bias")

        # Compute mean and variance
        dim = len(x.shape) - 1
        mean = x.mean(dim=dim, keepdim=True)
        var = x.var(dim=dim, keepdim=True)

        # Normalize
        x_norm = (x - mean) / (var + self.eps).sqrt()

        # Apply learnable parameters
        output = weight * x_norm + bias

        return output


def create_layers_plugin():
    """Create a plugin with custom layers"""

    plugin = (
        plugins.PluginBuilder()
        .name("custom_layers")
        .version(plugins.VersionInfo(1, 0, 0))
        .description("Custom neural network layers")
        .author("Layers Team")
        .min_minitensor_version(plugins.VersionInfo(0, 1, 0))
        .build()
    )

    def initialize(registry):
        print("Custom layers plugin initialized!")
        return True

    def cleanup(registry):
        print("Custom layers plugin cleaned up!")
        return True

    plugin.set_initialize_fn(initialize)
    plugin.set_cleanup_fn(cleanup)

    return plugin


def demo_plugin_registry():
    """Demonstrate plugin registry functionality"""
    print("=== Plugin Registry Demo ===")

    # Create registry
    registry = plugins.PluginRegistry()

    # Create and register plugins
    math_plugin = create_math_plugin()
    layers_plugin = create_layers_plugin()

    print("Registering plugins...")
    registry.register(math_plugin)
    registry.register(layers_plugin)

    # List registered plugins
    print("\nRegistered plugins:")
    for plugin_info in registry.list_plugins():
        print(f"  - {plugin_info.name} v{plugin_info.version}")
        print(f"    Author: {plugin_info.author}")
        print(f"    Description: {plugin_info.description}")
        print(f"    Min minitensor version: {plugin_info.min_minitensor_version}")

    # Check if plugins are registered
    print(f"\nMath plugin registered: {registry.is_registered('math_extensions')}")
    print(f"Layers plugin registered: {registry.is_registered('custom_layers')}")
    print(f"Unknown plugin registered: {registry.is_registered('unknown_plugin')}")

    # Get specific plugin
    try:
        math_plugin_retrieved = registry.get_plugin("math_extensions")
        print(f"\nRetrieved plugin: {math_plugin_retrieved.info.name}")
    except Exception as e:
        print(f"Error retrieving plugin: {e}")

    # Unregister plugin
    print("\nUnregistering math plugin...")
    registry.unregister("math_extensions")
    print(f"Math plugin still registered: {registry.is_registered('math_extensions')}")

    print()


def demo_custom_operations():
    """Demonstrate custom operations usage"""
    print("=== Custom Operations Demo ===")
    print("Custom operation builder is not available; skipping demo.")
    print()


def demo_custom_layers():
    """Demonstrate custom layers usage"""
    print("=== Custom Layers Demo ===")

    # Create custom layer
    layer_norm = CustomNormalizationLayer([4])  # Normalize last dimension of size 4

    # Create test input
    x = mt.randn([2, 4])  # Batch size 2, feature size 4
    print(f"Input shape: {x.shape}")
    print(f"Input tensor:\n{x}")

    # Apply custom layer
    try:
        output = layer_norm.forward([x])
        print(f"Output shape: {output.shape}")
        print(f"Output tensor:\n{output}")

        # Check that output has approximately zero mean and unit variance
        dim = len(output.shape) - 1
        mean = output.mean(dim=dim, keepdim=True)
        var = output.var(dim=dim, keepdim=True)
        print(f"Output mean (should be ~0): {mean}")
        print(f"Output variance (should be ~1): {var}")

    except Exception as e:
        print(f"Error in custom layer: {e}")

    # List layer parameters
    print(f"\nLayer parameters: {layer_norm.list_parameters()}")

    print()


def demo_global_plugin_management():
    """Demonstrate global plugin management"""
    print("=== Global Plugin Management Demo ===")

    # Note: Dynamic loading requires the 'dynamic-loading' feature
    # For this demo, we'll show the API without actually loading files

    print("Plugin loading API (requires dynamic-loading feature):")
    print("  plugins.load_plugin('./my_plugin.so')")
    print("  plugins.unload_plugin('my_plugin')")

    # List currently loaded plugins (should be empty in this demo)
    try:
        loaded_plugins = plugins.list_plugins()
        print(f"\nCurrently loaded plugins: {len(loaded_plugins)}")
        for plugin_info in loaded_plugins:
            print(f"  - {plugin_info}")
    except Exception as e:
        print(f"Error listing plugins: {e}")

    # Check if a plugin is loaded
    try:
        is_loaded = plugins.is_plugin_loaded("example_plugin")
        print(f"Example plugin loaded: {is_loaded}")
    except Exception as e:
        print(f"Error checking plugin: {e}")

    print()


def demo_error_handling():
    """Demonstrate error handling in plugin system"""
    print("=== Error Handling Demo ===")

    registry = plugins.PluginRegistry()

    # Try to register plugin with duplicate name
    plugin1 = create_math_plugin()
    plugin2 = create_math_plugin()  # Same name

    try:
        registry.register(plugin1)
        print("First plugin registered successfully")

        registry.register(plugin2)  # This should fail
        print("Second plugin registered (unexpected!)")
    except Exception as e:
        print(f"Expected error for duplicate plugin: {e}")

    # Try to get non-existent plugin
    try:
        unknown_plugin = registry.get_plugin("unknown_plugin")
        print(f"Got unknown plugin: {unknown_plugin}")
    except Exception as e:
        print(f"Expected error for unknown plugin: {e}")

    # Try to unregister non-existent plugin
    try:
        registry.unregister("unknown_plugin")
        print("Unregistered unknown plugin (unexpected!)")
    except Exception as e:
        print(f"Expected error for unregistering unknown plugin: {e}")

    # Version compatibility error
    try:
        incompatible_plugin = (
            plugins.PluginBuilder()
            .name("incompatible_plugin")
            .version(plugins.VersionInfo(1, 0, 0))
            .description("Plugin with incompatible version")
            .author("Test")
            .min_minitensor_version(plugins.VersionInfo(999, 0, 0))  # Future version
            .build()
        )

        # This would fail in real plugin registration due to version check
        print("Created plugin with incompatible version requirement")

    except Exception as e:
        print(f"Error with incompatible version: {e}")

    print()


def main():
    """Run all plugin system demonstrations"""
    print("Minitensor Plugin System Demonstration")
    print("=" * 50)

    try:
        demo_version_info()
        demo_plugin_registry()
        demo_custom_operations()
        demo_custom_layers()
        demo_global_plugin_management()
        demo_error_handling()

        print("Plugin system demonstration completed successfully!")

    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()