# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import examples.linear_regression as lr


def test_linear_regression_converges():
    loss, w, b = lr.train_model()
    assert loss < 1e-4
    assert abs(w - 2.0) < 1e-2
    assert abs(b + 3.0) < 1e-2
