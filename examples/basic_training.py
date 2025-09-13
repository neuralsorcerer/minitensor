# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Basic training example for Minitensor.

This script demonstrates a full training lifecycle using a ``DenseLayer``
and mean squared error loss on synthetic data. It shows how to set up a
model, define a loss and optimizer, run a training loop and report the
final loss.
"""

from __future__ import annotations

import minitensor as mt
from minitensor import nn, optim


def main():  # pragma: no cover - example script
    # Synthetic dataset: y = 3x + 0.5 with noise
    x = mt.randn(128, 1)
    y = 3 * x + 0.5 + 0.1 * mt.randn(128, 1)

    model = nn.DenseLayer(1, 1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(100):
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            loss_val = float(loss.numpy().ravel()[0])
            print(f"Epoch {epoch+1:03d} | Loss: {loss_val:.4f}")

    print("Final loss:", float(loss.numpy().ravel()[0]))


if __name__ == "__main__":  # pragma: no cover - example script
    main()
