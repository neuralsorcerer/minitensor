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

- `max`, `min`, `amax`, `amin` divide the gradient **evenly** among the tied
  elements. This is the mean-subgradient convention, and what PyTorch's `amax`
  does.
- `cummax`, `cummin` and `scatter_reduce`'s `"amax"`/`"amin"` give the whole
  gradient to the **first** element that won.

Nothing pinned either of them, and the API reference asserted the opposite of
the first: it said `max` gave a tie to the first contributor "where PyTorch
spreads a tie evenly", which is backwards for `max` and names `mode`, which has
no gradient at all. A convention nothing checks is a convention that drifts,
and this one drifts silently -- every test built on distinct values passes
either way.

`max(dim)` and `min(dim)` are worth a second look. They return an index as well
as a value, and that index names only the *first* tied element while the
gradient reaches all of them. Both halves are defensible on their own; together
they say two different things about which element was selected. PyTorch's
`max(dim)` resolves it the other way, sending the whole gradient to the index it
returned. This file pins what the code does rather than changing it, because
changing a gradient convention changes what people's training runs produce.
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
        ("max", lambda t: mt.max(t, 0)),
        ("min", lambda t: mt.min(t, 0)),
        ("amax", lambda t: mt.amax(t, 0)),
        ("amin", lambda t: mt.amin(t, 0)),
    ],
)
def test_the_reductions_share_a_ties_gradient_evenly(name, build):
    data = TIED if name in ("max", "amax") else [-v for v in TIED]
    grad, _ = _grad_of(build, data)
    np.testing.assert_array_equal(
        grad,
        np.array([0.0, 0.5, 0.5, 0.0]),
        err_msg=f"{name} no longer divides a tie evenly",
    )


@pytest.mark.parametrize("name", ["max", "min"])
def test_the_returned_index_names_only_the_first_tied_element(name):
    """Deliberately pinned, and deliberately not the same thing as the gradient.

    The index says "position 1"; the gradient credits positions 1 and 2. That
    is the inconsistency this file exists to make visible rather than hide.
    """
    data = TIED if name == "max" else [-v for v in TIED]
    grad, index = _grad_of(lambda t: getattr(mt, name)(t, 0), data)

    position = int(index[0])
    assert position == 1, "the index should be the first of the tied elements"
    only_the_index = np.zeros(4)
    only_the_index[position] = 1.0
    assert not np.array_equal(grad, only_the_index), (
        "the gradient now follows the returned index -- if that was intended, "
        "this test and the note in the API reference both need updating"
    )


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
