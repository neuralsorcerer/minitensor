# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Binary classification with logistic regression using Minitensor.

A simple demonstration of training a logistic regression model on a
synthetic dataset. All heavy computations are executed in Rust via the
Minitensor Python bindings.
"""

from __future__ import annotations

import numpy as np
import minitensor as mt


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

    w = mt.randn(2, 1, requires_grad=True)
    b = mt.zeros(1, requires_grad=True)

    lr = 0.1
    epochs = 100

    for epoch in range(epochs):
        logits = x.matmul(w) + b
        probs = logits.sigmoid()
        eps = 1e-7
        term1 = y * (probs + eps).log()
        term2 = (1 - y) * (1 - probs + eps).log()
        loss = (term1 + term2).mean() * -1

        # Gradient of BCE with sigmoid
        grad_logits = probs - y
        grad_w = x.transpose().matmul(grad_logits) / num_samples
        grad_b = grad_logits.mean()

        w = (w - lr * grad_w).detach().requires_grad_()
        b = (b - lr * grad_b).detach().requires_grad_()

        if (epoch + 1) % 20 == 0:
            preds = probs.detach().numpy() > 0.5
            acc = (preds.flatten() == y.numpy().flatten()).mean()
            loss_val = float(loss.numpy().ravel()[0])
            print(f"Epoch {epoch+1:03d} | Loss: {loss_val:.4f} | Acc: {acc:.2f}")

    preds = (x.matmul(w) + b).sigmoid().detach().numpy() > 0.5
    acc = (preds.flatten() == y.numpy().flatten()).mean()
    print(f"Final accuracy: {acc:.2f}")
    print("Learned weights:", w.numpy().ravel(), "bias:", b.numpy().ravel())


if __name__ == "__main__":  # pragma: no cover - example script
    main()
