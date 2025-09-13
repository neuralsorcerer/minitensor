# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Example usage of Minitensor neural network components.

This example demonstrates how to create a simple neural network using
the Minitensor library's Python bindings.
"""


def create_simple_network():
    """Create a simple feedforward neural network."""
    try:
        from minitensor import nn

        print("Creating a simple neural network...")

        # Create layers
        dense1 = nn.DenseLayer(784, 128)  # Input layer: 784 -> 128
        relu1 = nn.ReLU()

        dense2 = nn.DenseLayer(128, 64)  # Hidden layer: 128 -> 64
        relu2 = nn.ReLU()

        dense3 = nn.DenseLayer(64, 10)  # Output layer: 64 -> 10

        print("Created network layers:")
        print(f"  - {dense1}")
        print(f"  - {relu1}")
        print(f"  - {dense2}")
        print(f"  - {relu2}")
        print(f"  - {dense3}")

        return True
    except Exception as e:
        print(f"Failed to create network: {e}")
        print("Make sure to build the extension with: maturin develop --release")
        return False


def train_network():
    """Run a minimal training loop on random data."""
    try:
        import minitensor as mt
        from minitensor import nn, optim

        print("Running training demo...")

        x = mt.randn(64, 784)
        y = mt.randn(64, 10)

        model = nn.Sequential(
            [
                nn.DenseLayer(784, 128),
                nn.ReLU(),
                nn.DenseLayer(128, 10),
            ]
        )

        criterion = nn.MSELoss()
        optimizer = optim.SGD(0.01, 0.0, 0.0, False)
        params = model.parameters()

        for epoch in range(5):
            preds = model(x)
            loss = criterion(preds, y)
            optimizer.zero_grad(params)
            loss.backward()
            optimizer.step(params)
            loss_val = float(loss.numpy().ravel()[0])
            print(f"Epoch {epoch+1}: loss {loss_val:.4f}")

        return True
    except Exception as e:
        print(f"Training failed: {e}")
        return False


def create_loss_and_optimizer():
    """Create loss function and optimizer."""
    try:
        from minitensor import nn, optim

        print("\nCreating loss function and optimizer...")

        # Create loss function
        criterion = nn.CrossEntropyLoss()
        print(f"Created loss function: {criterion}")

        # Create optimizer (would normally pass model parameters)
        optimizer = optim.Adam(0.001, betas=(0.9, 0.999), epsilon=1e-8)
        print(f"Created optimizer: {optimizer}")

        return True
    except Exception as e:
        print(f"Failed to create loss/optimizer: {e}")
        return False


def demonstrate_tensor_operations():
    """Demonstrate basic tensor operations."""
    try:
        import minitensor as mt

        print("\nDemonstrating tensor operations...")

        # Create tensors
        x = mt.randn(32, 784)  # Batch of 32 samples, 784 features each
        print(f"Created input tensor: shape {x.shape}")

        # Create target tensor
        y = mt.zeros(32, dtype="int64")  # 32 class labels
        print(f"Created target tensor: shape {y.shape}")

        # Demonstrate tensor operations
        x_mean = x.mean()
        print(f"Computed mean: {x_mean}")

        x_sum = x.sum()
        print(f"Computed sum: {x_sum}")

        return True
    except Exception as e:
        print(f"✗ Failed tensor operations: {e}")
        return False


def demonstrate_activation_functions():
    """Demonstrate different activation functions."""
    try:
        import minitensor as mt
        from minitensor import nn

        print("\nDemonstrating activation functions...")

        # Create a sample tensor
        x = mt.randn(2, 3)
        print(f"Input tensor: {x}")

        # Test different activations
        activations = [
            ("ReLU", nn.ReLU()),
            ("Sigmoid", nn.Sigmoid()),
            ("Tanh", nn.Tanh()),
            ("LeakyReLU", nn.LeakyReLU(0.1)),
            ("ELU", nn.ELU(1.0)),
            ("GELU", nn.GELU()),
        ]

        for name, activation in activations:
            print(f"Created {name} activation: {activation}")

        return True
    except Exception as e:
        print(f"Failed activation demo: {e}")
        return False


def demonstrate_loss_functions():
    """Demonstrate different loss functions."""
    try:
        import minitensor as mt
        from minitensor import nn

        print("\nDemonstrating loss functions...")

        # Create sample predictions and targets
        predictions = mt.randn(4, 3)  # 4 samples, 3 classes
        targets = mt.randn(4, 3)  # 4 target values

        print(f"Predictions shape: {predictions.shape}")
        print(f"Targets shape: {targets.shape}")

        # Test different loss functions
        losses = [
            ("MSE", nn.MSELoss()),
            ("MAE", nn.MAELoss()),
            ("Huber", nn.HuberLoss(1.0)),
            ("CrossEntropy", nn.CrossEntropyLoss()),
            ("BCE", nn.BCELoss()),
            ("Focal", nn.FocalLoss(0.25, 2.0)),
        ]

        for name, loss_fn in losses:
            print(f"Created {name} loss: {loss_fn}")

        return True
    except Exception as e:
        print(f"Failed loss demo: {e}")
        return False


def main():
    """Run the neural network example."""
    print("Minitensor Neural Network Example")
    print("=" * 40)

    examples = [
        ("Training Loop", train_network),
        ("Simple Network Creation", create_simple_network),
        ("Loss and Optimizer", create_loss_and_optimizer),
        ("Tensor Operations", demonstrate_tensor_operations),
        ("Activation Functions", demonstrate_activation_functions),
        ("Loss Functions", demonstrate_loss_functions),
    ]

    success_count = 0

    for example_name, example_func in examples:
        print(f"\n--- {example_name} ---")
        try:
            if example_func():
                success_count += 1
                print(f"{example_name} completed successfully")
            else:
                print(f"{example_name} failed")
        except Exception as e:
            print(f"{example_name} failed with exception: {e}")

    print(f"\n{'='*40}")
    print(f"Completed {success_count}/{len(examples)} examples successfully")

    if success_count == len(examples):
        print("All examples completed successfully!")
        print("\nNext steps:")
        print("- Build the Rust extension with: maturin develop")
        print("- Run this example to test the actual implementation")
        print("- Implement the remaining tasks for full functionality")
    else:
        print(
            "Some examples failed - this is expected until the Rust extension is built"
        )


if __name__ == "__main__":
    main()
