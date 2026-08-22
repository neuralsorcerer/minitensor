# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`cat` split its work by the wrong thing, so joining on axis 0 was serial.

The copy was chunked by the output's concatenated block -- one task per
position along the axes *outside* `dim`. Join on an inner axis and that is
thousands of tasks; join on axis 0 and it is exactly one. Two 16MB float32
matrices joined on dimension 0 ran on a single core at 17.9ms against NumPy's
5.5ms, while the same tensors joined on dimension 1 took 6.9ms because that
shape happened to split.

Sizing the task by the output makes the split independent of which axis is
being joined: 6.7ms on dimension 0 now, and eight 2MB pieces joined on
dimension 0 went 1.33ms to 0.57ms, which is quicker than NumPy.

The cost is that a task no longer lines up with a source. It starts wherever
the split put it -- part-way through one input, spanning the boundary into the
next, possibly crossing several -- and has to work out where it is from its
offset alone. That arithmetic is what these tests are about, so the sizes are
chosen to be awkward: prime-ish widths, unequal inputs, and enough elements to
be split more than one way.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

# The task floor is 16384 elements, so anything above that gets split and
# anything below stays in one piece. Both sides need covering.
BIG = 300_000


def _parts(shapes, dtype="float32", seed=3):
    rng = np.random.default_rng(seed)
    arrays = []
    for shape in shapes:
        if dtype.startswith("float"):
            arrays.append(rng.standard_normal(shape).astype(dtype))
        elif dtype == "bool":
            arrays.append(rng.random(shape) < 0.5)
        else:
            arrays.append(rng.integers(-1000, 1000, size=shape).astype(dtype))
    return arrays


def _check(arrays, dim, dtype="float32"):
    tensors = [mt.Tensor(a, dtype=dtype) for a in arrays]
    got = mt.cat(tensors, dim).numpy()
    want = np.concatenate(arrays, axis=dim)
    np.testing.assert_array_equal(got, want)
    return got


@pytest.mark.parametrize(
    "shapes,dim",
    [
        # Axis 0 on a large 2-D output: the case that was serial.
        ([(2000, 151), (2000, 151)], 0),
        ([(1000, 151), (3000, 151), (7, 151)], 0),
        # Inner axes, which used to split and must still be right.
        ([(151, 2000), (151, 2000)], 1),
        ([(37, 53, 101), (37, 53, 101)], 2),
        ([(37, 53, 101), (37, 11, 101)], 1),
        ([(37, 53, 101), (13, 53, 101)], 0),
        # 1-D, where a block is the whole thing.
        ([(BIG,), (BIG,)], 0),
        ([(BIG,), (1,), (BIG,)], 0),
        # 4-D, joining the second axis.
        ([(3, 17, 29, 41), (3, 5, 29, 41)], 1),
    ],
)
def test_it_matches_numpy(shapes, dim):
    _check(_parts(shapes), dim)


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64", "bool"])
def test_every_dtype_copies_the_same(dtype):
    _check(_parts([(2000, 151), (2000, 151)], dtype=dtype), 0, dtype=dtype)


@pytest.mark.parametrize("count", [1, 2, 3, 8, 33])
def test_many_sources_land_in_order(count):
    """A task can cross several inputs at once when they are narrow, which is
    where an off-by-one in locating the source shows up as a shuffled result."""
    shapes = [(37, 151)] * count
    _check(_parts(shapes), 0)


def test_inputs_narrower_than_a_task_are_still_placed_correctly():
    """Each input is far smaller than the 16384-element floor, so a single task
    walks through dozens of them and must restart its arithmetic at each."""
    arrays = _parts([(11, 7)] * 60, seed=5)
    _check(arrays, 0)
    _check(arrays, 1)


def test_a_source_wider_than_a_task_is_split_across_tasks():
    """The other direction: one input long enough that several tasks each take
    a slice out of the middle of it."""
    arrays = _parts([(BIG, 1), (3, 1)], seed=7)
    _check(arrays, 0)


@pytest.mark.parametrize("dim", [0, 1])
def test_empty_inputs_are_skipped(dim):
    """A zero-width input contributes nothing, and the walk has to step over it
    rather than stalling on a zero-length run."""
    rng = np.random.default_rng(11)
    a = rng.standard_normal((100, 40)).astype(np.float32)
    b = rng.standard_normal((100, 40)).astype(np.float32)
    empty_shape = (0, 40) if dim == 0 else (100, 0)
    empty = np.zeros(empty_shape, dtype=np.float32)
    _check([a, empty, b], dim)
    _check([empty, a, b], dim)
    _check([a, b, empty], dim)


def test_uneven_widths_across_a_large_output():
    """Widths that share no factor with the task length, so every task boundary
    lands somewhere different inside a source."""
    arrays = _parts([(4001, 13), (7, 13), (1, 13), (99991, 13)], seed=13)
    _check(arrays, 0)


def test_stack_goes_through_the_same_copy():
    arrays = _parts([(500, 301), (500, 301), (500, 301)], seed=17)
    tensors = [mt.Tensor(a, dtype="float32") for a in arrays]
    for dim in (0, 1, 2):
        np.testing.assert_array_equal(
            mt.stack(tensors, dim).numpy(), np.stack(arrays, axis=dim)
        )


def test_the_result_does_not_depend_on_how_it_was_split():
    """Repeated calls may be scheduled differently; a copy is a copy."""
    arrays = _parts([(2000, 151), (2000, 151)], seed=19)
    tensors = [mt.Tensor(a, dtype="float32") for a in arrays]
    first = mt.cat(tensors, 0).numpy()
    for _ in range(10):
        np.testing.assert_array_equal(mt.cat(tensors, 0).numpy(), first)


def test_the_gradient_still_splits_back_to_the_inputs():
    a = mt.Tensor(
        np.ones((3, 4), dtype=np.float32), dtype="float32", requires_grad=True
    )
    b = mt.Tensor(
        np.ones((5, 4), dtype=np.float32) * 2, dtype="float32", requires_grad=True
    )
    out = mt.cat([a, b], 0)
    (
        out * mt.Tensor(np.arange(32, dtype=np.float32).reshape(8, 4), dtype="float32")
    ).sum().backward()
    np.testing.assert_array_equal(
        a.grad.numpy(), np.arange(12, dtype=np.float32).reshape(3, 4)
    )
    np.testing.assert_array_equal(
        b.grad.numpy(), np.arange(12, 32, dtype=np.float32).reshape(5, 4)
    )
