# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`nn.Softmax(dim=...)` and negative axes.

`dim=-1` is how a softmax is almost always written -- it is the last axis that
holds the classes, whatever the batch dimensions in front of it happen to be.
The layer took an unsigned dimension, so that call raised
`OverflowError: can't convert negative int to unsigned` from the binding's
integer conversion, before any tensor code ran. The functional `mt.softmax(x,
-1)` accepted it all along, so the two disagreed.

A layer is constructed before it sees a tensor, so the axis cannot be resolved
at construction: it is kept as given and resolved against the input's rank in
`forward`.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt
from minitensor import nn

SHAPE = (3, 4, 5)


def _softmax_reference(values: np.ndarray, axis: int) -> np.ndarray:
    shifted = values - values.max(axis, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis, keepdims=True)


@pytest.mark.parametrize("dim", [-3, -2, -1, 0, 1, 2])
def test_softmax_layer_accepts_every_valid_axis(dim):
    rng = np.random.default_rng(2)
    values = rng.standard_normal(SHAPE).astype(np.float32)
    tensor = mt.Tensor(values)

    got = nn.Softmax(dim)(tensor).numpy()

    expected = _softmax_reference(values.astype(np.float64), dim).astype(np.float32)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("dim", [-3, -2, -1, 0, 1, 2])
def test_softmax_layer_agrees_with_the_functional_form(dim):
    # The two took different integer types, which is how they came to disagree.
    rng = np.random.default_rng(3)
    tensor = mt.Tensor(rng.standard_normal(SHAPE).astype(np.float32))

    assert np.array_equal(
        nn.Softmax(dim)(tensor).numpy(), mt.softmax(tensor, dim).numpy()
    )


def test_softmax_layer_defaults_to_the_last_axis():
    rng = np.random.default_rng(4)
    values = rng.standard_normal(SHAPE).astype(np.float32)
    tensor = mt.Tensor(values)

    np.testing.assert_allclose(
        nn.Softmax()(tensor).numpy(), nn.Softmax(-1)(tensor).numpy(), rtol=0, atol=0
    )


@pytest.mark.parametrize("dim", [-3, -1, 0, 2])
def test_softmax_layer_reports_the_axis_it_was_given(dim):
    # Not the resolved one: the layer does not know the rank yet.
    assert nn.Softmax(dim).dim == dim


@pytest.mark.parametrize("dim", [3, 7, -4, -9])
def test_softmax_layer_rejects_an_out_of_range_axis(dim):
    tensor = mt.Tensor(np.zeros(SHAPE, dtype=np.float32))
    with pytest.raises(IndexError) as caught:
        nn.Softmax(dim)(tensor)
    # The message must name the axis the caller passed. Reporting the resolved
    # one told a reader who wrote -4 that -1 was out of bounds.
    assert str(dim) in str(caught.value)


def test_softmax_layer_differentiates_along_a_negative_axis():
    rng = np.random.default_rng(5)
    values = rng.standard_normal(SHAPE)
    tensor = mt.Tensor(values, dtype="float64").requires_grad_(True)

    out = nn.Softmax(-1)(tensor)
    upstream = rng.standard_normal(SHAPE)
    mt.sum(out * mt.Tensor(upstream, dtype="float64")).backward()

    probabilities = _softmax_reference(values, -1)
    expected = probabilities * (
        upstream - (upstream * probabilities).sum(-1, keepdims=True)
    )
    np.testing.assert_allclose(mt.get_gradient(tensor).numpy(), expected, atol=1e-14)


def test_a_softmax_layer_inside_a_sequential_uses_the_negative_axis():
    model = nn.Sequential([nn.DenseLayer(6, 4), nn.Softmax(-1)])
    out = model(mt.Tensor(np.zeros((2, 6), dtype=np.float32)))

    np.testing.assert_allclose(out.numpy().sum(-1), np.ones(2), rtol=1e-6)
