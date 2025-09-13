"""Digits classification with a simple CNN."""

from __future__ import annotations

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import minitensor as mt
from minitensor import nn, optim


def load_data():
    digits = load_digits()
    x = digits.images.astype("float32") / 16.0
    y = digits.target.astype("int64")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0
    )
    x_train = mt.from_numpy(x_train).reshape(x_train.shape[0], 1, 8, 8)
    x_test = mt.from_numpy(x_test).reshape(x_test.shape[0], 1, 8, 8)
    y_train = mt.from_numpy(y_train)
    y_test = mt.from_numpy(y_test)
    return x_train, x_test, y_train, y_test


def train(epochs: int = 20) -> float:
    x_train, x_test, y_train, y_test = load_data()
    feature_extractor = nn.Sequential(
        [
            nn.Conv2d(
                1,
                8,
                (3, 3),
                stride=(1, 1),
                padding=(0, 0),
                bias=True,
                device=None,
                dtype=None,
            ),
            nn.ReLU(),
        ]
    )
    classifier = nn.DenseLayer(8 * 6 * 6, 10)
    params = feature_extractor.parameters() + classifier.parameters()
    optimizer = optim.Adam(params, lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        feats = feature_extractor(x_train)
        feats = feats.reshape(feats.shape[0], 8 * 6 * 6)
        logits = classifier(feats)
        loss = criterion(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0:
            preds = logits.argmax(dim=1)
            acc = preds.eq(y_train).astype("float32").mean().numpy().ravel()[0]
            loss_val = float(loss.numpy().ravel()[0])
            print(f"Epoch {epoch + 1:03d} | Loss: {loss_val:.4f} | Acc: {acc:.2f}")

    test_feats = feature_extractor(x_test).reshape(x_test.shape[0], 8 * 6 * 6)
    test_logits = classifier(test_feats)
    test_preds = test_logits.argmax(dim=1)
    test_acc = test_preds.eq(y_test).astype("float32").mean().numpy().ravel()[0]
    print(f"Test accuracy: {test_acc:.2f}")
    return test_acc


def main() -> None:  # pragma: no cover - example script
    train()


if __name__ == "__main__":  # pragma: no cover - example script
    main()
