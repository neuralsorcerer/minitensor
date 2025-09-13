# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Linear regression training example for Minitensor.

This script demonstrates a complete training loop using the high level
``nn`` modules and ``optim`` optimisers on a synthetic dataset.
All tensor computations are handled by the Rust backend, while the
training loop is written in Python.
"""

from __future__ import annotations

import minitensor as mt
from minitensor import nn, optim


def main() -> None:  # pragma: no cover - example script
    # Generate synthetic data: y = 2x - 3 + noise
    n_samples = 200
    x = mt.randn(n_samples, 1)
    noise = 0.1 * mt.randn(n_samples, 1)
    y = 2.0 * x - 3.0 + noise

    model = nn.DenseLayer(1, 1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    epochs = 100

    for epoch in range(epochs):
        preds = model(x)
        loss = criterion(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            loss_val = float(loss.numpy().ravel()[0])
            print(f"Epoch {epoch+1:03d} | Loss: {loss_val:.4f}")

    # Fetch updated parameters after training
    params = model.parameters()
    print("Trained parameters:")
    w, b = params[0], params[1]
    print("w:", float(w.numpy().ravel()[0]), "b:", float(b.numpy().ravel()[0]))


if __name__ == "__main__":  # pragma: no cover - example script
    main()
