# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`flip` reverses every requested axis in one copy, and carries one gradient.

Reversing an axis is an index remapping, the same as rolling one: output
position `i` along a reversed dimension reads input position `size - 1 - i`. It
was done a dimension at a time through `index_select`, which allocates and
copies a whole intermediate tensor for each, and left a gradient edge per
dimension as well. On 4096x1024 float32, against NumPy and against a plain
contiguous copy of the same tensor as the floor:

                    before     after    NumPy    copy
    flip dim 1     4.22 ms   1.36 ms   2.27 ms   1.20 ms
    flip dim 0     2.13 ms   1.27 ms   1.54 ms
    flip both     11.70 ms   1.26 ms   2.20 ms

Flipping both axes now costs what flipping one does, which is the point: the
work is one pass whatever the dimension count.

Two things the rewrite has to keep. The first is the answer for every *subset*
of axes, not just the ones a benchmark uses -- the leading dimensions are
decoded from a row index and mirrored individually, so a shape whose strides
happen to coincide can hide a wrong one. These sweep every subset.

The second is the gradient. `flip` is its own inverse, so the backward is the
same flip applied to the incoming gradient; it used to fall out of
`index_select`'s chain and is now a single node, which is a different thing that
has to produce the same numbers.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import minitensor as mt

DTYPES = ["float32", "float64", "int32", "int64", "bool"]

# Rank 1 through 4, unit axes, and sizes either side of PAR_THRESHOLD (131072).
SHAPES = [
    (1,),
    (7,),
    (2, 3),
    (1, 6),
    (6, 1),
    (3, 4, 5),
    (2, 1, 3),
    (2, 3, 4, 5),
    (64, 64),  # 4096 elements: below the threshold
    (5, 600),  # 3000
    (400, 400),  # 160000: above it
]

EMPTY_SHAPES = [(0,), (0, 3), (3, 0), (2, 0, 4)]


def _values(shape, dtype, seed=0):
    rng = np.random.default_rng(seed)
    if dtype == "bool":
        return rng.integers(0, 2, size=shape).astype(dtype)
    return (rng.standard_normal(shape) * 10).astype(dtype)


def _subsets(rank):
    for size in range(rank + 1):
        yield from itertools.combinations(range(rank), size)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES, ids=[str(s) for s in SHAPES])
def test_every_subset_of_axes_matches_numpy(shape, dtype):
    """Including the empty subset, which reverses nothing."""
    values = _values(shape, dtype)
    tensor = mt.Tensor(values, dtype=dtype)

    for axes in _subsets(len(shape)):
        got = tensor.flip(list(axes)).numpy()
        expected = np.flip(values, axis=axes) if axes else values
        assert got.shape == expected.shape
        np.testing.assert_array_equal(got, expected, err_msg=f"{shape} axes={axes}")


@pytest.mark.parametrize("shape", SHAPES, ids=[str(s) for s in SHAPES])
def test_negative_axes_name_the_same_dimensions(shape):
    values = _values(shape, "float32")
    tensor = mt.Tensor(values)
    rank = len(shape)

    for axes in _subsets(rank):
        if not axes:
            continue
        negative = [axis - rank for axis in axes]
        np.testing.assert_array_equal(
            tensor.flip(negative).numpy(), tensor.flip(list(axes)).numpy()
        )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", EMPTY_SHAPES, ids=[str(s) for s in EMPTY_SHAPES])
def test_an_empty_tensor_keeps_its_shape(shape, dtype):
    """A zero-length axis leaves no rows to copy, and the row length can itself
    be zero."""
    values = _values(shape, dtype)
    tensor = mt.Tensor(values, dtype=dtype)

    for axis in range(len(shape)):
        flipped = tensor.flip([axis])
        assert flipped.shape == list(shape)
        np.testing.assert_array_equal(flipped.numpy(), np.flip(values, axis=axis))


def test_flipping_the_same_axes_twice_is_the_identity():
    values = _values((2, 3, 4), "float64")
    tensor = mt.Tensor(values, dtype="float64")

    np.testing.assert_array_equal(tensor.flip([0, 2]).flip([0, 2]).numpy(), values)


def test_a_unit_axis_reverses_to_itself():
    """`size - 1 - i` is `i` when the axis has one element, which is a fine
    place for an off-by-one to hide."""
    values = _values((1, 5, 1), "float32")
    tensor = mt.Tensor(values)

    for axes in [(0,), (2,), (0, 2), (0, 1, 2)]:
        np.testing.assert_array_equal(
            tensor.flip(list(axes)).numpy(), np.flip(values, axis=axes)
        )


# --- the gradient, which is now one node rather than one per axis ------------


@pytest.mark.parametrize("shape", [(5,), (2, 3), (3, 4, 5), (2, 3, 4)])
def test_the_gradient_is_the_same_flip(shape):
    """`flip` is an involution, so the backward reverses the same axes."""
    weights = _values(shape, "float64", seed=1)

    for axes in _subsets(len(shape)):
        if not axes:
            continue
        mt.clear_autograd_graph()
        x = mt.Tensor(_values(shape, "float64"), dtype="float64", requires_grad=True)

        (x.flip(list(axes)) * mt.Tensor(weights, dtype="float64")).sum().backward()

        gradient = mt.get_gradient(x)
        assert gradient is not None, f"axes={axes} produced no gradient"
        np.testing.assert_allclose(
            gradient.numpy(), np.flip(weights, axis=axes), rtol=1e-12
        )
    mt.clear_autograd_graph()


def test_the_gradient_of_a_double_flip_is_untouched():
    mt.clear_autograd_graph()
    x = mt.Tensor(
        np.arange(24, dtype=np.float64).reshape(2, 3, 4),
        dtype="float64",
        requires_grad=True,
    )

    x.flip([0, 2]).flip([0, 2]).sum().backward()

    np.testing.assert_array_equal(mt.get_gradient(x).numpy(), np.ones((2, 3, 4)))
    mt.clear_autograd_graph()


def test_flip_under_no_grad_is_not_tracked():
    x = mt.Tensor(np.ones((2, 2)), dtype="float64", requires_grad=True)
    with mt.no_grad():
        assert x.flip([0]).requires_grad is False


# --- the arguments it still has to reject ------------------------------------


def test_a_repeated_axis_is_rejected():
    """Reversing an axis twice is the identity, but asking for it is a mistake
    worth naming rather than silently honouring once."""
    with pytest.raises(Exception, match="unique"):
        mt.randn(2, 3).flip([0, 0])


@pytest.mark.parametrize("axis", [-3, 2, 9])
def test_an_out_of_range_axis_is_rejected(axis):
    with pytest.raises(IndexError):
        mt.randn(2, 3).flip([axis])


def test_flipping_nothing_returns_the_same_values():
    values = _values((3, 4), "float32")
    np.testing.assert_array_equal(mt.Tensor(values).flip([]).numpy(), values)
