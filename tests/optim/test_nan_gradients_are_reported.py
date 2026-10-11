# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A NaN gradient shows up in the parameter it reached, under every optimizer.

A diverged gradient has to be visible, and the parameter is where a training
loop looks. Most optimizers carry the NaN there through their arithmetic. The
two that read only a gradient's sign did not: `Lion` took the sign of NaN to
be zero, and since the NaN then sat in its momentum, every later direction was
NaN and every later step zero -- the parameter froze where it stood, finite and
wrong, for good. `Rprop` skipped the step as though the gradient were zero.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

optim = mt.optim

OPTIMIZERS = [
    optim.SGD,
    optim.Adam,
    optim.AdamW,
    optim.Adamax,
    optim.NAdam,
    optim.RAdam,
    optim.RMSprop,
    optim.Adagrad,
    optim.Adadelta,
    optim.Rprop,
    optim.Lion,
]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("make", OPTIMIZERS, ids=lambda cls: cls.__name__)
def test_a_nan_gradient_makes_its_parameter_nan_and_no_other(make, dtype):
    weight = mt.Tensor([1.0, 2.0, -3.0], dtype=dtype, requires_grad=True)
    optimizer = make([weight], lr=0.1)
    poison = mt.Tensor([float("nan"), 1.0, -1.0], dtype=dtype)
    (weight * poison).sum().backward()
    optimizer.step()

    values = weight.numpy()
    assert np.isnan(values[0]), f"{make.__name__} hid the NaN: {values}"
    assert np.isfinite(values[1:]).all(), f"{make.__name__} spread it: {values}"

    # Later, ordinary gradients leave the NaN where it is and keep moving the
    # rest, so the divergence stays on record rather than being trained over.
    before = values[1:].copy()
    for _ in range(2):
        (weight * mt.Tensor([1.0, 1.0, 1.0], dtype=dtype)).sum().backward()
        optimizer.step()
    values = weight.numpy()
    assert np.isnan(values[0])
    assert np.isfinite(values[1:]).all() and (values[1:] != before).all()
