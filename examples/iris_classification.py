"""Iris classification with Minitensor.

This example trains a small multilayer perceptron on the classic Iris
flower dataset. It demonstrates data loading, tensor operations,
autograd, neural network modules, loss functions and optimizers in a
single script.
"""

from __future__ import annotations

import csv
from pathlib import Path
from random import Random

import minitensor as mt
from minitensor import nn, optim

DATA_PATH = Path(__file__).resolve().parent / "data" / "iris.csv"


def load_iris():
    """Load the Iris dataset from a local CSV file."""
    with open(DATA_PATH, newline="") as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row]
    features = [[float(v) for v in row[:4]] for row in rows]
    label_map = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
    labels = [label_map[row[4]] for row in rows]
    return features, labels


def train(epochs: int = 200) -> float:
    x, y = load_iris()
    rng = Random(0)
    indices = list(range(len(x)))
    rng.shuffle(indices)
    x = [x[i] for i in indices]
    y = [y[i] for i in indices]
    split = int(0.8 * len(x))
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    x_train = mt.Tensor(x_train, dtype="float32")
    y_train = mt.Tensor(y_train, dtype="int64")
    x_test = mt.Tensor(x_test, dtype="float32")
    y_test = mt.Tensor(y_test, dtype="int64")

    model = nn.Sequential(
        [
            nn.DenseLayer(4, 16),
            nn.ReLU(),
            nn.DenseLayer(16, 3),
        ]
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        logits = model(x_train)
        loss = criterion(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            preds = logits.argmax(dim=1)
            acc = preds.eq(y_train).astype("float32").mean().numpy().ravel()[0]
            loss_val = float(loss.numpy().ravel()[0])
            print(f"Epoch {epoch + 1:03d} | Loss: {loss_val:.4f} | Acc: {acc:.2f}")

    test_logits = model(x_test)
    test_preds = test_logits.argmax(dim=1)
    test_acc = test_preds.eq(y_test).astype("float32").mean().numpy().ravel()[0]
    print(f"Test accuracy: {test_acc:.2f}")
    return test_acc


def main() -> None:  # pragma: no cover - example script
    train()


if __name__ == "__main__":  # pragma: no cover - example script
    main()
