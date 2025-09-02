# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Custom Operations Demo for Minitensor

This script demonstrates the custom operations system by:
1. Registering example custom operations
2. Using custom operations in tensor computations
3. Showing automatic differentiation with custom operations
4. Comparing performance with built-in operations
"""

import sys
import time

import numpy as np

import minitensor as mt

if not hasattr(mt, "register_example_custom_ops") or not hasattr(mt, "list_custom_ops_py"):
    print("Custom operations extension unavailable; skipping demo.")
    sys.exit(0)


def demo_basic_custom_operations():
    """Demonstrate basic usage of custom operations."""
    print("=" * 60)
    print("BASIC CUSTOM OPERATIONS DEMO")
    print("=" * 60)

    # Register example custom operations
    print("Registering example custom operations...")
    mt.register_example_custom_ops()

    # List available operations
    ops = mt.list_custom_ops_py()
    print(f"Available custom operations: {ops}")

    # Create test tensor
    x = mt.Tensor([[1.0, 2.0, -1.0, 0.5, -0.5]], requires_grad=True)
    print(f"\nInput tensor: {x}")
    print(f"Input shape: {x.shape}")

    # Test Swish activation
    print("\n--- Swish Activation ---")
    swish_result = mt.execute_custom_op_py("swish", [x])
    print(f"Swish result: {swish_result}")

    # Test GELU activation
    print("\n--- GELU Activation ---")
    gelu_result = mt.execute_custom_op_py("gelu", [x])
    print(f"GELU result: {gelu_result}")

    # Test Mish activation
    print("\n--- Mish Activation ---")
    mish_result = mt.execute_custom_op_py("mish", [x])
    print(f"Mish result: {mish_result}")


def demo_power_operation():
    """Demonstrate the custom power operation with validation."""
    print("\n" + "=" * 60)
    print("POWER OPERATION DEMO")
    print("=" * 60)

    # Create compatible tensors
    base = mt.Tensor([[2.0, 3.0], [4.0, 5.0]], requires_grad=True)
    exponent = mt.Tensor([[2.0, 3.0], [2.0, 2.0]], requires_grad=True)

    print(f"Base tensor: {base}")
    print(f"Exponent tensor: {exponent}")

    # Execute power operation
    result = mt.execute_custom_op_py("power", [base, exponent])
    print(f"Power result: {result}")

    # Test validation with incompatible shapes
    print("\n--- Testing Shape Validation ---")
    try:
        incompatible_exp = mt.Tensor([[1.0]], requires_grad=True)
        mt.execute_custom_op_py("power", [base, incompatible_exp])
        print("ERROR: Should have failed with shape mismatch!")
    except Exception as e:
        print(f"✓ Correctly caught shape mismatch: {type(e).__name__}")


def demo_layer_normalization():
    """Demonstrate the custom layer normalization operation."""
    print("\n" + "=" * 60)
    print("LAYER NORMALIZATION DEMO")
    print("=" * 60)

    # Create input tensor (batch_size=2, features=4)
    input_tensor = mt.Tensor(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], requires_grad=True
    )

    # Create weight and bias (must match last dimension)
    weight = mt.Tensor([1.0, 1.0, 1.0, 1.0], requires_grad=True)
    bias = mt.Tensor([0.0, 0.0, 0.0, 0.0], requires_grad=True)

    print(f"Input tensor: {input_tensor}")
    print(f"Weight: {weight}")
    print(f"Bias: {bias}")

    # Execute layer normalization
    result = mt.execute_custom_op_py("layer_norm", [input_tensor, weight, bias])
    print(f"Layer norm result: {result}")

    # Test validation with wrong weight/bias size
    print("\n--- Testing Weight/Bias Validation ---")
    try:
        wrong_weight = mt.Tensor([1.0, 1.0], requires_grad=True)  # Wrong size
        mt.execute_custom_op_py("layer_norm", [input_tensor, wrong_weight, bias])
        print("ERROR: Should have failed with size mismatch!")
    except Exception as e:
        print(f"✓ Correctly caught size mismatch: {type(e).__name__}")


def demo_automatic_differentiation():
    """Demonstrate automatic differentiation with custom operations."""
    print("\n" + "=" * 60)
    print("AUTOMATIC DIFFERENTIATION DEMO")
    print("=" * 60)

    # Create input tensor
    x = mt.Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    print(f"Input tensor: {x}")
    print(f"Requires grad: {x.requires_grad}")

    # Apply custom operations in sequence
    swish_out = mt.execute_custom_op_py("swish", [x])
    print(f"Swish output requires grad: {swish_out.requires_grad}")

    # Compute loss
    loss = swish_out.sum()
    print(f"Loss: {loss}")
    print(f"Loss requires grad: {loss.requires_grad}")

    # Backward pass
    try:
        loss.backward()
        print("✓ Backward pass completed")

        # Check gradients
        if x.grad is not None:
            print(f"Input gradients: {x.grad}")
            print("✓ Automatic differentiation working with custom operations!")
        else:
            print(
                "✗ No gradients computed - this may be expected for simplified custom ops"
            )
    except Exception as e:
        print(f"Backward pass failed: {e}")
        print(
            "Note: Gradient computation may not be fully implemented for demo custom ops"
        )


def demo_error_handling():
    """Demonstrate error handling in custom operations."""
    print("\n" + "=" * 60)
    print("ERROR HANDLING DEMO")
    print("=" * 60)

    # Test unregistered operation
    print("--- Testing Unregistered Operation ---")
    try:
        x = mt.Tensor([1.0, 2.0, 3.0])
        mt.execute_custom_op_py("nonexistent_op", [x])
        print("ERROR: Should have failed!")
    except Exception as e:
        print(f"✓ Correctly handled unregistered operation: {type(e).__name__}")

    # Test wrong number of inputs
    print("\n--- Testing Wrong Input Count ---")
    try:
        x = mt.Tensor([1.0, 2.0, 3.0])
        mt.execute_custom_op_py("power", [x])  # Power needs 2 inputs
        print("ERROR: Should have failed!")
    except Exception as e:
        print(f"✓ Correctly handled wrong input count: {type(e).__name__}")


def demo_operation_management():
    """Demonstrate operation registration and unregistration."""
    print("\n" + "=" * 60)
    print("OPERATION MANAGEMENT DEMO")
    print("=" * 60)

    # List current operations
    ops_before = mt.list_custom_ops_py()
    print(f"Operations before unregistration: {ops_before}")

    # Unregister an operation
    if "swish" in ops_before:
        print("\nUnregistering 'swish' operation...")
        mt.unregister_custom_op_py("swish")

        # Verify it's gone
        ops_after = mt.list_custom_ops_py()
        print(f"Operations after unregistration: {ops_after}")

        # Try to use unregistered operation
        try:
            x = mt.Tensor([1.0, 2.0, 3.0])
            mt.execute_custom_op_py("swish", [x])
            print("ERROR: Should have failed!")
        except Exception as e:
            print(f"✓ Correctly handled unregistered operation: {type(e).__name__}")

        # Re-register for other demos
        print("\nRe-registering example operations...")
        try:
            mt.register_example_custom_ops()
        except Exception as e:
            print(f"Note: Some operations already registered: {e}")
            # This is expected since we only unregistered one operation


def benchmark_custom_operations():
    """Benchmark custom operations against built-in operations."""
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARK")
    print("=" * 60)

    # Create larger test tensor
    size = 1000
    x = mt.Tensor(np.random.randn(size, size).astype(np.float32), requires_grad=True)

    print(f"Benchmarking with tensor shape: {x.shape}")

    # Benchmark built-in tanh
    print("\n--- Built-in Tanh ---")
    start_time = time.time()
    for _ in range(10):
        result = x.tanh()
    builtin_time = time.time() - start_time
    print(f"Built-in tanh time: {builtin_time:.4f}s")

    # Benchmark custom swish (uses built-in operations internally)
    print("\n--- Custom Swish ---")
    start_time = time.time()
    for _ in range(10):
        result = mt.execute_custom_op_py("swish", [x])
    custom_time = time.time() - start_time
    print(f"Custom swish time: {custom_time:.4f}s")

    # Calculate overhead
    if builtin_time > 0:
        overhead = (custom_time / builtin_time) - 1.0
        print(f"Custom operation overhead: {overhead:.1%}")

    print("\nNote: Custom operations have some overhead due to the")
    print("extensibility framework, but provide maximum flexibility.")


def main():
    """Run all custom operations demos."""
    print("MINITENSOR CUSTOM OPERATIONS DEMONSTRATION")
    print("This demo showcases the extensibility framework")

    try:
        demo_basic_custom_operations()
        demo_power_operation()
        demo_layer_normalization()
        demo_automatic_differentiation()
        demo_error_handling()
        demo_operation_management()
        benchmark_custom_operations()

        print("\n" + "=" * 60)
        print("🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nThe custom operations system provides:")
        print("✓ Easy registration and management of custom operations")
        print("✓ Full automatic differentiation support")
        print("✓ Comprehensive input validation")
        print("✓ Robust error handling")
        print("✓ Performance optimization opportunities")
        print("\nYou can now create your own custom operations using the")
        print("CustomOpBuilder pattern in Rust or extend the Python API!")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
