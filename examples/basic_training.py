# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Basic forward-pass example for Minitensor.

This script demonstrates creating a ``DenseLayer`` and computing a mean
square error loss on synthetic data. Full training with optimizers is not
yet implemented; the example focuses on showing how to perform a forward
pass and compute a loss without raising runtime errors.
"""

from __future__ import annotations

import minitensor as mt
from minitensor import nn


def main():  # pragma: no cover - example script
    # Synthetic dataset: y = 3x + 0.5 with noise
    x = mt.randn(128, 1)
    y = 3 * x + 0.5 + 0.1 * mt.randn(128, 1)

    model = nn.DenseLayer(1, 1)
    pred = model.forward(x._tensor)
    criterion = nn.MSELoss()
    loss = criterion.forward(pred, y._tensor)
    print("Initial loss:", loss)


if __name__ == "__main__":  # pragma: no cover - example script
    main()
