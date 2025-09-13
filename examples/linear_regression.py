# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Linear regression training example for Minitensor.

This script demonstrates a complete training loop using the high level
``nn`` modules and ``optim`` optimisers on a deterministic linear dataset.
All tensor computations are handled by the Rust backend, while the
training loop is written in Python.
"""

from __future__ import annotations

import minitensor as mt
from minitensor import nn, optim


def train_model(verbose: bool = True):
    """Train a linear model on noise-free synthetic data.

    Parameters
    ----------
    verbose:
        If ``True``, prints the loss at the first epoch and every 20 thereafter.

    Returns
    -------
    tuple[float, float, float]
        Final loss, learned weight and bias.
    """

    # Deterministic dataset: y = 2x - 3
    x = mt.arange(-1.0, 1.0, 0.01).reshape(200, 1)
    y = 2.0 * x - 3.0

    model = nn.DenseLayer(1, 1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05)

    epochs = 200

    for epoch in range(epochs):
        preds = model(x)
        loss = criterion(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and (epoch == 0 or (epoch + 1) % 20 == 0):
            loss_val = float(loss.numpy().ravel()[0])
            print(f"Epoch {epoch+1:03d} | Loss: {loss_val:.6f}")

    final_loss = float(loss.numpy().ravel()[0])
    params = model.parameters()
    w = float(params[0].numpy().ravel()[0])
    b = float(params[1].numpy().ravel()[0]) if len(params) > 1 else 0.0
    return final_loss, w, b


def main() -> None:  # pragma: no cover - example script
    loss, w, b = train_model(verbose=True)
    print("Final loss:", loss)
    print("Trained parameters:")
    print("w:", w, "b:", b)


if __name__ == "__main__":  # pragma: no cover - example script
    main()
