# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`amax`/`amin`: the extremum without its index.

`max(dim=...)` returns `(values, indices)`, and finding the index is most of
the work -- it turns a fold that compiles to a vectorized compare-and-select
into one that has to carry a position alongside and branch to update it. On a
2048x1024 float32 matrix, reducing along the last axis measured 0.109ms for the
values alone against 0.833ms for the pair.

So a caller who writes `t.max(dim=1)[0]` -- which is the ordinary way to spell
"row maxima" -- was paying about 7.6x for an index tensor they then dropped.
The values-only reduction already existed in the engine (`logsumexp` uses it to
take the column max for stability); it simply had no name in the Python API.

`amax`/`amin` are what NumPy and PyTorch both call this, so the name is not a
new idea to learn.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

RNG = np.random.default_rng(17)
SHAPES = [(6,), (4, 5), (3, 4, 5), (2, 3, 2, 4)]


def _dims(shape):
    return [None] + list(range(len(shape))) + [-1, -len(shape)]


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
@pytest.mark.parametrize("shape", SHAPES)
def test_amax_and_amin_match_numpy(shape, dtype):
    if dtype.startswith("int"):
        values = RNG.integers(-50, 50, shape).astype(dtype)
    else:
        values = (RNG.standard_normal(shape) * 5).astype(dtype)
    t = mt.Tensor(values, dtype=dtype)

    for dim in _dims(shape):
        for keepdim in (False, True):
            if dim is None:
                got_max, got_min = t.amax(), t.amin()
                want_max, want_min = values.max(), values.min()
                np.testing.assert_array_equal(got_max.numpy(), want_max)
                np.testing.assert_array_equal(got_min.numpy(), want_min)
                continue
            np.testing.assert_array_equal(
                t.amax(dim, keepdim).numpy(),
                values.max(axis=dim, keepdims=keepdim),
                err_msg=f"amax dim={dim} keepdim={keepdim}",
            )
            np.testing.assert_array_equal(
                t.amin(dim, keepdim).numpy(),
                values.min(axis=dim, keepdims=keepdim),
                err_msg=f"amin dim={dim} keepdim={keepdim}",
            )


@pytest.mark.parametrize("shape", SHAPES)
def test_amax_agrees_with_the_values_half_of_max(shape):
    """The whole point is that this is the same answer more cheaply, so the two
    must not drift apart."""
    values = (RNG.standard_normal(shape) * 5).astype(np.float64)
    t = mt.Tensor(values, dtype="float64")
    for dim in range(len(shape)):
        for keepdim in (False, True):
            paired, _ = t.max(dim=dim, keepdim=keepdim)
            np.testing.assert_array_equal(t.amax(dim, keepdim).numpy(), paired.numpy())
            paired, _ = t.min(dim=dim, keepdim=keepdim)
            np.testing.assert_array_equal(t.amin(dim, keepdim).numpy(), paired.numpy())


def test_nan_propagates_through_amax_and_amin():
    """`amax`/`amin` are the NaN-*propagating* pair, matching `max`/`min`; the
    `nanamax`/`nanamin` spellings are the ones that skip."""
    values = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])
    t = mt.Tensor(values, dtype="float64")

    got = t.amax(1).numpy()
    assert np.isnan(got[0]), got
    assert got[1] == 6.0
    got = t.amin(1).numpy()
    assert np.isnan(got[0]), got
    assert got[1] == 4.0


def test_nanamax_and_nanamin_skip_nan():
    values = np.array([[1.0, np.nan, 3.0], [np.nan, np.nan, 2.0]])
    t = mt.Tensor(values, dtype="float64")
    np.testing.assert_array_equal(t.nanamax(1).numpy(), np.nanmax(values, axis=1))
    np.testing.assert_array_equal(t.nanamin(1).numpy(), np.nanmin(values, axis=1))


def test_an_all_nan_slice_reduces_to_nan():
    """There is no non-NaN element to report, so NaN is the answer rather than
    an infinity sentinel that a real input could have collided with."""
    values = np.array([[np.nan, np.nan], [1.0, 2.0]])
    t = mt.Tensor(values, dtype="float64")
    got = t.nanamax(1).numpy()
    assert np.isnan(got[0])
    assert got[1] == 2.0


@pytest.mark.parametrize("name", ["amax", "amin", "nanamax", "nanamin"])
def test_the_module_function_and_the_method_agree(name):
    values = (RNG.standard_normal((4, 6))).astype(np.float32)
    t = mt.Tensor(values, dtype="float32")
    np.testing.assert_array_equal(
        getattr(mt, name)(t, 1).numpy(), getattr(t, name)(1).numpy()
    )


def test_gradients_flow_to_the_selected_positions():
    """`amax` drops the indices from its *output*, not from its backward: the
    gradient still has to reach the elements that won."""
    values = np.array([[1.0, 5.0, 3.0], [9.0, 2.0, 4.0]])
    t = mt.Tensor(values, dtype="float64", requires_grad=True)
    t.amax(1).sum().backward()

    grad = mt.get_gradient(t).numpy()
    expected = np.zeros_like(values)
    expected[0, 1] = 1.0
    expected[1, 0] = 1.0
    np.testing.assert_array_equal(grad, expected)


@pytest.mark.parametrize("dim", [3, -4])
def test_a_dimension_that_does_not_exist_is_declined(dim):
    t = mt.Tensor(np.zeros((2, 3)), dtype="float64")
    with pytest.raises(Exception):
        t.amax(dim)
    with pytest.raises(Exception):
        t.amin(dim)
