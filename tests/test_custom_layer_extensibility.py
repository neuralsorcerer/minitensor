# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`CustomLayer` is the Python extension path, so its contract needs pinning.

The reference documented `CustomLayer` as far as `add_parameter` and
`list_parameters` and stopped, which left the half that makes it useful --
`set_forward` and `forward` -- undescribed. The signature is not the obvious
one: forward receives a *list* of tensors rather than a single tensor, so the
natural first attempt raises `TypeError: 'Tensor' object is not an instance of
'list'`.

What works is worth locking down too: a Python forward records onto the
autograd graph like any built-in op, so custom parameters train under a normal
optimizer, and a custom layer composes with built-in layers through an ordinary
Python function.

The composition limits are asserted rather than described, so that relaxing one
later is a deliberate act with a failing test to notice it.
"""

import numpy as np
import pytest

import minitensor as mt
from minitensor import nn
from minitensor.plugins import CustomLayer


def _scale_layer(gain_value, size=3):
    layer = CustomLayer("scale")
    gain = mt.Tensor(
        np.full(size, gain_value), dtype="float64", requires_grad=True
    )
    layer.add_parameter("gain", gain)
    layer.set_forward(lambda inputs: [inputs[0] * layer.get_parameter("gain")])
    return layer, gain


def test_forward_takes_a_list_not_a_tensor():
    layer, _ = _scale_layer(2.0)
    x = mt.Tensor(np.array([1.0, 2.0, 3.0]), dtype="float64")

    with pytest.raises(TypeError):
        layer.forward(x)

    np.testing.assert_allclose(layer.forward([x])[0].numpy(), [2.0, 4.0, 6.0])


def test_forward_before_set_forward_raises():
    layer = CustomLayer("empty")
    with pytest.raises(NotImplementedError):
        layer.forward([mt.Tensor(np.ones(2), dtype="float64")])


def test_a_forward_may_return_a_bare_tensor():
    layer = CustomLayer("bare")
    layer.set_forward(lambda inputs: inputs[0] * 2.0)
    result = layer.forward([mt.Tensor(np.array([1.0, 2.0]), dtype="float64")])
    # Returned as given rather than wrapped, so callers must not assume a list.
    np.testing.assert_allclose(np.asarray(result.numpy()), [2.0, 4.0])


def test_multiple_inputs_and_outputs():
    layer = CustomLayer("pair")
    layer.set_forward(lambda inputs: [inputs[0] + inputs[1], inputs[0] * inputs[1]])
    a = mt.Tensor(np.array([1.0, 2.0]), dtype="float64")
    b = mt.Tensor(np.array([3.0, 4.0]), dtype="float64")

    total, product = layer.forward([a, b])
    np.testing.assert_allclose(total.numpy(), [4.0, 6.0])
    np.testing.assert_allclose(product.numpy(), [3.0, 8.0])


def test_gradients_flow_through_a_python_forward():
    layer, gain = _scale_layer(2.0)
    x = mt.Tensor(np.array([1.0, 2.0, 3.0]), dtype="float64")

    layer.forward([x])[0].sum().backward()

    # d/dgain of sum(x * gain) is x.
    np.testing.assert_allclose(gain.grad.numpy(), [1.0, 2.0, 3.0])
    mt.clear_autograd_graph()


def test_a_custom_layer_trains_alongside_a_built_in_one():
    mt.manual_seed(0)
    dense = nn.DenseLayer(3, 3, dtype="float64")
    layer, gain = _scale_layer(2.0)
    x = mt.Tensor(np.ones((4, 3)), dtype="float64")

    optimizer = mt.optim.SGD(list(dense.parameters()) + [gain], lr=0.01)
    first = None
    for step in range(40):
        optimizer.zero_grad()
        loss = (layer.forward([dense(x)])[0] ** 2).sum()
        if step == 0:
            first = loss.item()
        loss.backward()
        optimizer.step()

    assert gain.grad is not None, "gradient never reached the custom parameter"
    assert loss.item() < first / 1000
    mt.clear_autograd_graph()


def test_the_composition_limits_are_what_they_are():
    # Each of these is a deliberate boundary rather than an accident; a change
    # here should break this test and be noticed.
    layer, _ = _scale_layer(1.0)
    assert not isinstance(layer, nn.Module)

    with pytest.raises(TypeError, match="Module"):
        nn.Sequential([layer])

    inner = nn.Sequential([nn.DenseLayer(4, 4), nn.ReLU()])
    with pytest.raises(TypeError, match="Nested Sequential"):
        nn.Sequential([inner])
    with pytest.raises(TypeError, match="Nested Sequential"):
        nn.Sequential([nn.DenseLayer(4, 4)]).add_module("block", inner)
