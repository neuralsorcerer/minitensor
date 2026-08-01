# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from pytest import raises

import minitensor as mt
from minitensor import nn, optim


def test_adam_accepts_betas_tuple():
    model = nn.DenseLayer(1, 1)
    params = model.parameters()
    optimizer = optim.Adam(params, 0.01, betas=(0.8, 0.888))

    x = mt.randn(10, 1)
    y = 2 * x + 1
    criterion = nn.MSELoss()
    preds = model(x)
    loss = criterion(preds, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert isinstance(optimizer, optim.Adam)


def test_adam_rejects_non_tuple_betas():
    model = nn.DenseLayer(1, 1)
    params = model.parameters()
    with raises(TypeError):
        optim.Adam(params, 0.01, betas=0.9)


def test_adam_accepts_beta1_beta2():
    model = nn.DenseLayer(1, 1)
    params = model.parameters()
    optimizer = optim.Adam(params, 0.01, beta1=0.8, beta2=0.888)

    x = mt.randn(5, 1)
    y = 3 * x - 1
    criterion = nn.MSELoss()
    loss = criterion(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert isinstance(optimizer, optim.Adam)


def test_adam_rejects_mixed_beta_args():
    model = nn.DenseLayer(1, 1)
    params = model.parameters()
    with raises(TypeError):
        optim.Adam(params, 0.01, betas=(0.9, 0.999), beta1=0.9)


def _run_adam(amsgrad, gradients):
    """Drive Adam with a fixed gradient sequence and return the final parameter."""
    import numpy as np

    parameter = mt.Tensor(np.array([1.0]), dtype="float64").requires_grad_(True)
    optimizer = optim.Adam([parameter], 0.1, amsgrad=amsgrad)
    for gradient in gradients:
        optimizer.zero_grad(True)
        (parameter * float(gradient)).sum().backward()
        optimizer.step()
    return float(parameter.numpy()[0])


def test_adam_amsgrad_is_reachable_and_changes_the_update():
    # The engine has carried `with_amsgrad` and a tested `v_hat` update from the
    # start; nothing bound it, so the variant was unreachable from Python.
    import numpy as np

    assert optim.Adam([mt.zeros((1,), requires_grad=True)], 1e-3).amsgrad is False
    assert (
        optim.Adam([mt.zeros((1,), requires_grad=True)], 1e-3, amsgrad=True).amsgrad
        is True
    )

    # AMSGrad keeps the running maximum of the second moment, so after one large
    # gradient it takes strictly smaller steps than plain Adam does.
    gradients = [10.0, 0.1, 0.1, 0.1, 0.1, 0.1]
    plain = _run_adam(False, gradients)
    ams = _run_adam(True, gradients)
    assert not np.isclose(plain, ams)
    assert ams > plain  # smaller total descent from the same start


def test_adam_repr_reports_amsgrad_with_python_booleans():
    optimizer = optim.Adam([mt.zeros((1,), requires_grad=True)], 1e-3, amsgrad=True)
    text = repr(optimizer)
    assert "amsgrad=True" in text
    # Rust spells its booleans lower case; a repr has to be valid Python.
    assert "true" not in text and "false" not in text
