"""Multiclass classification demo using Minitensor.

This example trains a small neural network to recover a
synthetic three-class linear decision boundary. It exercises
Tensor ops, autograd, neural network modules, loss functions,
and optimizers.
"""

from __future__ import annotations

import minitensor as mt
from minitensor import nn, optim


def make_dataset(num_samples: int = 300):
    # Deterministic features distributed along a grid
    x = mt.Tensor.linspace(-1.0, 1.0, num_samples * 2).reshape(num_samples, 2)
    # Known weights and bias producing three classes
    true_w = mt.Tensor([[2.0, -2.0, 0.5], [0.5, 1.0, -1.0]])
    true_b = mt.Tensor([0.1, -0.3, 0.2])
    logits = x.matmul(true_w) + true_b
    y = logits.argmax(dim=1).astype("int64")
    return x, y


def main() -> None:  # pragma: no cover - example script
    x, y = make_dataset()

    model = nn.Sequential(
        [
            nn.DenseLayer(2, 16),
            nn.ReLU(),
            nn.DenseLayer(16, 3),
        ]
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 200
    for epoch in range(epochs):
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            preds = logits.argmax(dim=1)
            acc = preds.eq(y).astype("float32").mean().numpy().ravel()[0]
            loss_val = float(loss.numpy().ravel()[0])
            print(f"Epoch {epoch + 1:03d} | Loss: {loss_val:.4f} | Acc: {acc:.2f}")

    final_preds = model(x).argmax(dim=1)
    final_acc = final_preds.eq(y).astype("float32").mean().numpy().ravel()[0]
    print(f"Final accuracy: {final_acc:.2f}")


if __name__ == "__main__":  # pragma: no cover - example script
    main()
