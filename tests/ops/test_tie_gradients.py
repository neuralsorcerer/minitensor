# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Where a tied maximum sends its gradient.

`max` is not differentiable at a tie, so an implementation picks a convention,
and any convex combination of the tied elements is a valid subgradient. Two
conventions are in use here and the difference is invisible until two elements
are exactly equal:

- The forms that **report an index** -- `max(dim)`, `min(dim)`, their NaN-aware
  variants, and `median(dim)` -- send the whole gradient to the element that
  index names. Value, index and gradient then all say the same thing about
  which element was selected.
- The forms that **report no index** -- `amax`, `amin`, `nanmedian` -- divide
  the gradient **evenly** among everything equal to the extremum. That is the
  mean subgradient, and what PyTorch's `amax` does. With no index to be
  consistent with, there is no reason to prefer one tied element.
- `cummax`, `cummin` and `scatter_reduce`'s `"amax"`/`"amin"` give the whole
  gradient to the **first** element that won.

Nothing pinned any of them, and the API reference asserted the opposite: it
said `max` gave a tie to the first contributor "where PyTorch spreads a tie
evenly", which was backwards, and named `mode`, which has no gradient at all.
A convention nothing checks is a convention that drifts, and this one drifts
silently -- every test built on distinct values passes either way, which is
the last test here.

Until recently every one of these reductions took the split, so `max(dim)`
returned index 1 and then fed the gradient to positions 1 *and* 2. Either half
is a defensible subgradient alone; disagreeing with each other is not, and it
is the index that has to win, because the index is the part a caller can act
on.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

TIED = [1.0, 3.0, 3.0, 2.0]  # the maximum appears twice


def _grad_of(build, data=TIED):
    mt.clear_autograd_graph()
    tensor = mt.Tensor(
        np.asarray(data, dtype=np.float64), dtype="float64", requires_grad=True
    )
    result = build(tensor)
    values, indices = result if isinstance(result, tuple) else (result, None)
    values.sum().backward()
    grad = np.asarray(tensor.grad).ravel().copy()
    index = None if indices is None else np.asarray(indices).ravel().copy()
    mt.clear_autograd_graph()
    return grad, index


@pytest.mark.parametrize(
    "name,build",
    [
        ("amax", lambda t: mt.amax(t, 0)),
        ("amin", lambda t: mt.amin(t, 0)),
        ("nanmedian", lambda t: mt.nanmedian(t, 0)),
    ],
)
def test_the_index_free_reductions_share_a_ties_gradient_evenly(name, build):
    data = TIED if name == "amax" else [-v for v in TIED]
    expected = np.array([0.0, 0.5, 0.5, 0.0])
    if name == "nanmedian":
        data, expected = [3.0, 3.0, 3.0, 1.0], np.array([1 / 3, 1 / 3, 1 / 3, 0.0])
    grad, _ = _grad_of(build, data)
    np.testing.assert_allclose(
        grad, expected, err_msg=f"{name} no longer divides a tie evenly"
    )


@pytest.mark.parametrize(
    "name,build,data",
    [
        ("max", lambda t: mt.max(t, 0), TIED),
        ("min", lambda t: mt.min(t, 0), [-v for v in TIED]),
        ("nanmax", lambda t: mt.nanmax(t, 0), TIED),
        ("nanmin", lambda t: mt.nanmin(t, 0), [-v for v in TIED]),
        ("median", lambda t: mt.median(t, 0, False), [3.0, 3.0, 3.0, 1.0]),
        ("median-even", lambda t: mt.median(t, 0, False), [1.0, 2.0, 2.0, 5.0]),
    ],
)
def test_the_gradient_goes_to_the_element_the_index_names(name, build, data):
    grad, index = _grad_of(build, data)
    position = int(index[0])
    expected = np.zeros(len(data))
    expected[position] = 1.0
    np.testing.assert_array_equal(
        grad,
        expected,
        err_msg=f"{name} reported index {position} but its gradient went elsewhere",
    )


def test_an_all_nan_slice_no_longer_divides_by_a_zero_tie_count():
    """The equality-mask path cannot serve these: `NaN == NaN` is false, so the
    mask is empty, the tie count is zero, and the gradient is 0/0."""
    grad, index = _grad_of(
        lambda t: mt.nanmax(t, 0), [float("nan"), float("nan"), float("nan")]
    )
    assert not np.isnan(grad).any(), f"all-NaN slice gave {grad}"
    expected = np.zeros(3)
    expected[int(index[0])] = 1.0
    np.testing.assert_array_equal(grad, expected)


@pytest.mark.parametrize(
    "name,data",
    [("cummax", [3.0, 3.0, 1.0]), ("cummin", [1.0, 1.0, 3.0])],
)
def test_the_scans_give_a_ties_gradient_to_the_first_element(name, data):
    grad, index = _grad_of(lambda t: getattr(mt, name)(t, 0), data)
    np.testing.assert_array_equal(grad, np.array([3.0, 0.0, 0.0]))
    np.testing.assert_array_equal(index, np.array([0, 0, 0]))


def test_scatter_reduce_amax_gives_a_tie_to_the_first_contributor():
    mt.clear_autograd_graph()
    source = mt.Tensor(np.array([3.0, 3.0, 1.0]), dtype="float64", requires_grad=True)
    index = mt.Tensor(np.array([0, 0, 1], dtype=np.int64), dtype="int64")
    base = mt.Tensor(np.array([-9.0, -9.0]), dtype="float64")

    mt.scatter_reduce(base, 0, index, source, "amax").sum().backward()

    np.testing.assert_array_equal(
        np.asarray(source.grad).ravel(),
        np.array([1.0, 0.0, 1.0]),
        err_msg="the first of the two tied contributors should take it all",
    )
    mt.clear_autograd_graph()


def test_mode_has_no_gradient():
    """Named in the API reference's tie rule, which it cannot follow."""
    mt.clear_autograd_graph()
    tensor = mt.Tensor(np.asarray(TIED), dtype="float64", requires_grad=True)
    values, _ = mt.mode(tensor, 0)
    with pytest.raises(Exception):
        values.sum().backward()
    mt.clear_autograd_graph()


def test_distinct_values_cannot_tell_the_conventions_apart():
    """Why none of the existing tests caught the documentation being backwards."""
    distinct = [1.0, 4.0, 3.0, 2.0]
    grad, index = _grad_of(lambda t: mt.max(t, 0), distinct)
    expected = np.zeros(4)
    expected[int(index[0])] = 1.0
    np.testing.assert_array_equal(grad, expected)
