# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""There was no way to pad a tensor.

Constant padding could be spelled with `cat` and a tensor of zeros, awkwardly
and only for the value zero unless you built a filled tensor first. Reflect and
replicate could not be spelled at all: they read the input back at reflected or
clamped coordinates, which is index arithmetic no composition of the existing
operations performs.

All three are one mechanism here -- every output position maps to an input
position, or to nothing. Constant is the mode where "or to nothing" happens and
the fill value applies; the other two always land somewhere real. That is also
what makes their gradients differ: reflect and replicate send many output
positions to one input, so gradient *accumulates* there. A replicated edge
copied five times has five gradients arriving at it, and dropping four would be
a silent under-count of exactly the elements padding touched.

`padding` is flat and innermost-axis-first -- `[left, right]` pads the last
axis -- which is PyTorch's convention and the reverse of NumPy's. The reference
below converts explicitly rather than relying on either, because getting that
backwards produces a plausible-looking tensor of the wrong shape.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

MODES = ["constant", "reflect", "replicate"]
CASES = [
    ((5,), [2, 3]),
    ((3, 4), [1, 2]),
    ((3, 4), [1, 2, 2, 1]),
    ((2, 3, 4), [1, 1, 2, 2, 1, 0]),
    ((2, 3, 4, 5), [1, 1]),
    ((4, 4), [0, 3]),
    ((4, 4), [3, 0]),
]


def _numpy_pad(array, padding, mode, value=0.0):
    """NumPy takes pairs outermost-first; this takes them innermost-first."""
    pairs = [(0, 0)] * array.ndim
    for axis, (left, right) in enumerate(zip(padding[0::2], padding[1::2])):
        pairs[array.ndim - 1 - axis] = (left, right)
    if mode == "constant":
        return np.pad(array, pairs, mode="constant", constant_values=value)
    return np.pad(array, pairs, mode="reflect" if mode == "reflect" else "edge")


def _numeric_grad(f, arr, eps=1e-6):
    grad = np.zeros_like(arr)
    flat, gflat = arr.reshape(-1), grad.reshape(-1)
    for i in range(flat.size):
        old = flat[i]
        flat[i] = old + eps
        high = f(arr)
        flat[i] = old - eps
        low = f(arr)
        flat[i] = old
        gflat[i] = (high - low) / (2 * eps)
    return grad


@pytest.mark.parametrize("shape,padding", CASES)
@pytest.mark.parametrize("mode", MODES)
def test_it_matches_numpy(shape, padding, mode):
    values = np.random.default_rng(0).standard_normal(shape)
    got = mt.Tensor(values, dtype="float64").pad(padding, mode).numpy()
    np.testing.assert_array_equal(got, _numpy_pad(values, padding, mode))


def test_the_pairs_are_innermost_axis_first():
    """The convention itself, on a shape where getting it backwards still
    produces a valid tensor -- just the wrong one."""
    values = np.zeros((2, 3))
    got = mt.Tensor(values, dtype="float64").pad([1, 0])
    assert tuple(got.shape) == (2, 4), "one column added on the left of the last axis"
    got = mt.Tensor(values, dtype="float64").pad([0, 0, 1, 0])
    assert tuple(got.shape) == (3, 3), "one row added on top"


def test_axes_the_padding_does_not_reach_are_left_alone():
    values = np.random.default_rng(3).standard_normal((2, 3, 4))
    got = mt.Tensor(values, dtype="float64").pad([1, 1]).numpy()
    assert got.shape == (2, 3, 6)
    np.testing.assert_array_equal(got[:, :, 1:-1], values)


@pytest.mark.parametrize("value", [0.0, -7.5, 1e9])
def test_constant_uses_the_value(value):
    values = np.ones((2, 2))
    got = mt.Tensor(values, dtype="float64").pad([1, 1], "constant", value).numpy()
    np.testing.assert_array_equal(got, _numpy_pad(values, [1, 1], "constant", value))


def test_reflect_does_not_repeat_the_edge():
    """The difference between the two non-constant modes, on the smallest case
    that shows it."""
    values = np.array([1.0, 2.0, 3.0])
    t = mt.Tensor(values, dtype="float64")
    np.testing.assert_array_equal(
        t.pad([2, 2], "reflect").numpy(), [3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0]
    )
    np.testing.assert_array_equal(
        t.pad([2, 2], "replicate").numpy(), [1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0]
    )


def test_zero_padding_returns_the_input_unchanged():
    values = np.random.default_rng(5).standard_normal((3, 4))
    for mode in MODES:
        np.testing.assert_array_equal(
            mt.Tensor(values, dtype="float64").pad([0, 0, 0, 0], mode).numpy(), values
        )


@pytest.mark.parametrize(
    "dtype,values",
    [
        ("float32", np.arange(6, dtype=np.float32).reshape(2, 3)),
        ("int32", np.arange(6, dtype=np.int32).reshape(2, 3)),
        ("int64", np.arange(6, dtype=np.int64).reshape(2, 3)),
        ("bool", (np.arange(6) % 2 == 0).reshape(2, 3)),
    ],
)
@pytest.mark.parametrize("mode", MODES)
def test_every_dtype_pads(dtype, values, mode):
    got = mt.Tensor(values, dtype=dtype).pad([1, 1], mode).numpy()
    np.testing.assert_array_equal(got, _numpy_pad(values, [1, 1], mode))
    assert mt.Tensor(values, dtype=dtype).pad([1, 1], mode).dtype == dtype


def test_reflect_needs_room_to_reflect_into():
    """Padding as wide as the axis would fold the reflection back over itself,
    and there is no agreed answer for what that should give."""
    values = mt.Tensor(np.arange(3.0), dtype="float64")
    with pytest.raises(Exception):
        values.pad([3, 0], "reflect")
    with pytest.raises(Exception):
        values.pad([0, 3], "reflect")
    # One less is fine.
    assert values.pad([2, 2], "reflect").numpy().shape == (7,)


def test_reflect_refuses_a_single_element_axis():
    """A run of one has no period to mirror around."""
    with pytest.raises(Exception):
        mt.Tensor(np.zeros((1, 4)), dtype="float64").pad([0, 0, 1, 1], "reflect")


def test_an_unknown_mode_is_refused():
    with pytest.raises(Exception):
        mt.Tensor(np.zeros(3), dtype="float64").pad([1, 1], "circular")


def test_negative_padding_is_refused():
    """Trimming is a different operation and this is not it."""
    with pytest.raises(Exception):
        mt.Tensor(np.zeros(5), dtype="float64").pad([-1, 0])


def test_too_many_pairs_is_refused():
    with pytest.raises(Exception):
        mt.Tensor(np.zeros(5), dtype="float64").pad([1, 1, 1, 1])


def test_an_odd_number_of_padding_values_is_refused():
    with pytest.raises(Exception):
        mt.Tensor(np.zeros((2, 3)), dtype="float64").pad([1, 1, 1])


# --- gradients -------------------------------------------------------------


@pytest.mark.parametrize(
    "shape,padding", [((5,), [2, 3]), ((3, 4), [1, 2]), ((3, 4), [1, 2, 2, 1])]
)
@pytest.mark.parametrize("mode", MODES)
def test_the_gradient_matches_numerical_differentiation(shape, padding, mode):
    rng = np.random.default_rng(7)
    values = rng.standard_normal(shape)
    out_shape = mt.Tensor(values, dtype="float64").pad(padding, mode).numpy().shape
    weights = rng.standard_normal(out_shape)

    def loss(v):
        return float(
            (mt.Tensor(v, dtype="float64").pad(padding, mode).numpy() * weights).sum()
        )

    t = mt.Tensor(values.copy(), dtype="float64", requires_grad=True)
    (t.pad(padding, mode) * mt.Tensor(weights, dtype="float64")).sum().backward()
    np.testing.assert_allclose(
        t.grad.numpy(), _numeric_grad(loss, values.copy()), rtol=1e-6, atol=1e-8
    )


def test_a_replicated_edge_collects_every_copy_of_its_gradient():
    """Spelled out rather than left to the numerical check: the left edge is
    copied twice into the padding plus itself, so three gradients arrive."""
    t = mt.Tensor(np.array([1.0, 2.0, 3.0]), dtype="float64", requires_grad=True)
    t.pad([2, 1], "replicate").sum().backward()
    np.testing.assert_array_equal(t.grad.numpy(), [3.0, 1.0, 2.0])


def test_constant_padding_sends_back_exactly_one_gradient_each():
    """The injective case: nothing accumulates, and the padded positions
    contribute nothing at all."""
    t = mt.Tensor(np.array([1.0, 2.0, 3.0]), dtype="float64", requires_grad=True)
    t.pad([2, 1], "constant").sum().backward()
    np.testing.assert_array_equal(t.grad.numpy(), [1.0, 1.0, 1.0])


def test_a_reflected_element_collects_every_copy_of_its_gradient():
    t = mt.Tensor(np.array([1.0, 2.0, 3.0]), dtype="float64", requires_grad=True)
    t.pad([2, 1], "reflect").sum().backward()
    # [3 2 | 1 2 3 | 2]: element 1 appears once, 2 three times, 3 twice.
    np.testing.assert_array_equal(t.grad.numpy(), [1.0, 3.0, 2.0])


def test_the_module_level_function_agrees_with_the_method():
    values = np.random.default_rng(11).standard_normal((3, 4))
    t = mt.Tensor(values, dtype="float64")
    for mode in MODES:
        np.testing.assert_array_equal(
            mt.pad(t, [1, 2, 1, 0], mode).numpy(), t.pad([1, 2, 1, 0], mode).numpy()
        )


def test_padding_then_slicing_recovers_the_input():
    """A round-trip that does not depend on NumPy: whatever was added, the
    middle is still the original."""
    values = np.random.default_rng(13).standard_normal((4, 5))
    for mode in MODES:
        padded = mt.Tensor(values, dtype="float64").pad([2, 3, 1, 1], mode).numpy()
        np.testing.assert_array_equal(padded[1:-1, 2:-3], values)
