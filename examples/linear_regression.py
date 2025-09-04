# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Linear regression training example for Minitensor.

This script demonstrates a full training loop using gradient descent
on a synthetic dataset. All tensor computations are handled by the
Rust backend, while the training loop is written in Python.
"""

from __future__ import annotations

import minitensor as mt


def main() -> None:  # pragma: no cover - example script
    # Generate synthetic data: y = 2x - 3 + noise
    n_samples = 200
    x = mt.randn(n_samples, 1)
    noise = 0.1 * mt.randn(n_samples, 1)
    y = 2.0 * x - 3.0 + noise

    # Model parameters
    w = mt.randn(1, 1, requires_grad=True)
    b = mt.zeros(1, requires_grad=True)

    lr = 0.1
    epochs = 100

    for epoch in range(epochs):
        # Forward pass
        preds = x.matmul(w) + b
        diff = preds - y
        loss = (diff * diff).mean()

        # Compute gradients manually
        grad_w = x.transpose().matmul(2 * diff) / n_samples
        grad_b = (2 * diff).mean()

        # Gradient descent update
        w = (w - lr * grad_w).detach().requires_grad_()
        b = (b - lr * grad_b).detach().requires_grad_()

        if (epoch + 1) % 20 == 0:
            loss_val = float(loss.numpy().ravel()[0])
            print(f"Epoch {epoch+1:03d} | Loss: {loss_val:.4f}")

    print("Trained parameters:")
    w_val = float(w.numpy().ravel()[0])
    b_val = float(b.numpy().ravel()[0])
    print("w:", w_val, "b:", b_val)


if __name__ == "__main__":  # pragma: no cover - example script
    main()
