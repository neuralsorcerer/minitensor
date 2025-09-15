# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import examples.basic_training as bt


def test_basic_training_converges():
    loss, w, b = bt.train_model()
    assert loss < 1e-4
    assert abs(w - 3.0) < 1e-2
    assert abs(b - 0.5) < 1e-2
