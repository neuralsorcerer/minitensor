# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Binary classification with logistic regression using Minitensor.

This example trains a logistic regression model using the high level
``nn`` and ``optim`` APIs. It demonstrates a full training cycle including
model definition, loss computation, backpropagation and parameter updates.
"""

from __future__ import annotations

import numpy as np

import minitensor as mt
from minitensor import nn, optim


def main() -> None:  # pragma: no cover - example script
    rng = np.random.default_rng(0)
    num_samples = 200

    # Synthetic dataset linearly separable by true_w and true_b
    x_np = rng.normal(size=(num_samples, 2)).astype(np.float32)
    true_w = np.array([[1.5], [-2.0]], dtype=np.float32)
    true_b = -0.5
    logits_np = x_np @ true_w + true_b
    y_np = (logits_np > 0).astype(np.float32)

    x = mt.Tensor(x_np)
    y = mt.Tensor(y_np.reshape(-1, 1))

    model = nn.Sequential([nn.DenseLayer(2, 1), nn.Sigmoid()])
    criterion = nn.BCELoss()
    optimizer = optim.SGD(0.1, 0.0, 0.0, False)
    params = model.parameters()

    epochs = 100

    for epoch in range(epochs):
        preds = model(x)
        loss = criterion(preds, y)
        optimizer.zero_grad(params)
        loss.backward()
        optimizer.step(params)

        if (epoch + 1) % 20 == 0:
            preds_np = preds.detach().numpy() > 0.5
            acc = (preds_np.flatten() == y.numpy().flatten()).mean()
            loss_val = float(loss.numpy().ravel()[0])
            print(f"Epoch {epoch+1:03d} | Loss: {loss_val:.4f} | Acc: {acc:.2f}")

    final_preds = model(x).detach().numpy() > 0.5
    acc = (final_preds.flatten() == y.numpy().flatten()).mean()
    print(f"Final accuracy: {acc:.2f}")

    # Retrieve trained parameters
    params = model.parameters()
    w, b = params[0], params[1]
    print("Learned weights:", w.numpy().ravel(), "bias:", b.numpy().ravel())


if __name__ == "__main__":  # pragma: no cover - example script
    main()
