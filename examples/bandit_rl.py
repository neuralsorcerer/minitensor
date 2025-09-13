# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Epsilon-greedy Q-learning on a multi-armed bandit."""

from __future__ import annotations

from random import Random

import minitensor as mt


def train(steps: int = 5000) -> mt.Tensor:
    probs = [0.2, 0.5, 0.8]
    q = mt.zeros(len(probs), dtype="float32")
    counts = mt.zeros(len(probs), dtype="float32")
    eps = 0.1
    rng = Random(0)

    for _ in range(steps):
        if rng.random() < eps:
            action = rng.randrange(len(probs))
        else:
            action = int(q.argmax().numpy().ravel()[0])
        reward = 1.0 if rng.random() < probs[action] else 0.0
        counts[action] = counts[action] + 1.0
        q[action] = q[action] + (reward - q[action]) / counts[action]
    print("Estimated values:", q.numpy().tolist())
    return q


def main() -> None:  # pragma: no cover - example script
    train()


if __name__ == "__main__":  # pragma: no cover - example script
    main()
