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


def train_model(verbose: bool = True):
    """Train a single dense layer on a synthetic linear dataset.

    Parameters
    ----------
    verbose:
        If ``True``, prints the loss every 20 epochs.

    Returns
    -------
    tuple[float, float, float]
        Final loss, weight and bias values.
    """

    # Deterministic linear dataset: y = 3x + 0.5
    x = mt.arange(-1.0, 1.0, 0.02).reshape(100, 1)
    y = 3 * x + 0.5

    model = nn.DenseLayer(1, 1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05)

    for epoch in range(200):
        pred = model(x)
        loss = criterion(pred, y)
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


def main():  # pragma: no cover - example script
    loss, w, b = train_model(verbose=True)
    print("Final loss:", loss)
    print("Trained parameters:")
    print("w:", w, "b:", b)


if __name__ == "__main__":  # pragma: no cover - example script
    main()
