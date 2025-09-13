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

import minitensor as mt
from minitensor import nn, optim


def main() -> None:  # pragma: no cover - example script
    num_samples = 200

    # Synthetic dataset linearly separable by true_w and true_b
    x = mt.randn(num_samples, 2)
    true_w = mt.Tensor([[1.5], [-2.0]])
    true_b = mt.Tensor([-0.5])
    logits = x @ true_w + true_b
    y = (logits > 0).astype("float32")

    model = nn.Sequential([nn.DenseLayer(2, 1), nn.Sigmoid()])
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    epochs = 100

    for epoch in range(epochs):
        preds = model(x)
        loss = criterion(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            preds_bin = preds.detach() > 0.5
            acc = (
                preds_bin.eq(y.astype("bool"))
                .astype("float32")
                .mean()
                .numpy()
                .ravel()[0]
            )
            loss_val = float(loss.numpy().ravel()[0])
            print(f"Epoch {epoch+1:03d} | Loss: {loss_val:.4f} | Acc: {acc:.2f}")

    final_preds = model(x).detach() > 0.5
    acc = float(
        final_preds.eq(y.astype("bool")).astype("float32").mean().numpy().ravel()[0]
    )
    print(f"Final accuracy: {acc:.2f}")

    # Retrieve trained parameters
    params = model.parameters()
    w, b = params[0], params[1]
    print("Learned weights:", w.numpy().ravel(), "bias:", b.numpy().ravel())


if __name__ == "__main__":  # pragma: no cover - example script
    main()
