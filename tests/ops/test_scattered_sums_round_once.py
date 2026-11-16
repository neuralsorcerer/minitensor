# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A destination several values are scattered to holds their exact sum, rounded once.

Added one at a time in the element type, a scattered sum took a rounding for
each value past the first, in the order they arrived. Each destination of
`scatter_add`, of a `scatter_reduce` sum or mean, of `index_add`, an
accumulating `put`, `fold` and `embedding_bag` is now the float nearest the
exact sum of everything addressed to it -- its own value among them where it
counts -- and so is each gradient that is a sum of the entries that read one
position: `gather`'s, `index_select`'s and an embedding's, a repeated
element's, and a value broadcast into an assignment. Where each value is scaled
first -- `index_add`'s `alpha`, `embedding_bag`'s `per_sample_weights` -- the
sum is of the exact products, where rounding each product before the sum
rounded again had put a third of the destinations an ulp or more out.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt


def _float32(value: Fraction) -> float:
    """The float32 nearest `value`, ties to even."""

    if value == 0:
        return 0.0
    sign = -1.0 if value < 0 else 1.0
    magnitude = abs(value)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if Fraction(2) ** exponent > magnitude:
        exponent -= 1
    unit = Fraction(2) ** max(exponent - 23, -149)
    rounded = round(magnitude / unit) * unit
    return sign * (math.inf if rounded >= Fraction(2) ** 128 else float(rounded))


def _nearest(values, dtype: str, over: int = 1) -> float:
    """The float of `dtype` nearest the exact sum of `values` over `over`."""

    total = sum((Fraction(float(v)) for v in values), Fraction(0)) / over
    if dtype == "float32":
        return _float32(total)
    return total.numerator / total.denominator


def _draw(rng, shape, dtype: str):
    values = rng.standard_normal(shape) * 10.0 ** rng.uniform(-6, 6, shape)
    return values.astype(dtype)


def _arrivals(index: np.ndarray, source: np.ndarray, dim: int) -> dict:
    """The values of `source` addressed to each destination, in order."""

    lists: dict = {}
    for position in np.ndindex(*index.shape):
        destination = list(position)
        destination[dim] = int(index[position])
        lists.setdefault(tuple(destination), []).append(source[position])
    return lists


def _scattered(base, index, source, dim, dtype, own=True, mean=False) -> list:
    """What a scattered sum or mean leaves, from the definition."""

    lists = _arrivals(index, source, dim)
    want = np.array(base, dtype=np.float64)
    for destination, values in lists.items():
        values = list(values) + ([base[destination]] if own else [])
        want[destination] = _nearest(values, dtype, len(values) if mean else 1)
    return want.tolist()


DTYPES = ["float32", "float64"]

# (destination shape, axis, sources along it): few destinations each taking
# many values, a middle axis, and enough destinations for several blocks.
LAYOUTS = [((7, 5), 0, 40), ((6, 9), 1, 30), ((2, 9000, 3), 1, 12000)]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape, dim, rows", LAYOUTS)
def test_scatter_add(dtype, shape, dim, rows):
    rng = np.random.default_rng(1031 + rows)
    base = _draw(rng, shape, dtype)
    source_shape = shape[:dim] + (rows,) + shape[dim + 1 :]
    source = _draw(rng, source_shape, dtype)
    index = rng.integers(0, shape[dim], source_shape)
    got = mt.scatter_add(
        mt.Tensor(base, dtype=dtype),
        dim,
        mt.Tensor(index, dtype="int64"),
        mt.Tensor(source, dtype=dtype),
    )
    assert str(got.dtype) == dtype
    assert got.tolist() == _scattered(base, index, source, dim, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("own", [True, False])
@pytest.mark.parametrize("reduce", ["sum", "mean"])
def test_scatter_reduce(dtype, own, reduce):
    rng = np.random.default_rng(1033)
    base, source = _draw(rng, (9, 4), dtype), _draw(rng, (50, 4), dtype)
    # Destination 0 takes nothing, and keeps what it holds.
    index = rng.integers(1, 9, (50, 4))
    got = mt.scatter_reduce(
        mt.Tensor(base, dtype=dtype),
        0,
        mt.Tensor(index, dtype="int64"),
        mt.Tensor(source, dtype=dtype),
        reduce,
        include_self=own,
    )
    want = _scattered(base, index, source, 0, dtype, own, reduce == "mean")
    assert got.tolist() == want
    assert got.tolist()[0] == base[0].tolist()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("width", [3, 40])
def test_index_add_and_an_accumulating_put(dtype, width):
    # Whole slices added into slices, a few values wide and many.
    rng = np.random.default_rng(1039 + width)
    base, source = _draw(rng, (6, width), dtype), _draw(rng, (25, width), dtype)
    rows = rng.integers(0, 6, 25)
    got = mt.index_add(
        mt.Tensor(base, dtype=dtype),
        0,
        mt.Tensor(rows, dtype="int64"),
        mt.Tensor(source, dtype=dtype),
    )
    index = np.broadcast_to(rows[:, None], (25, width))
    assert got.tolist() == _scattered(base, index, source, 0, dtype)

    across = _draw(rng, (width, 30), dtype)
    columns = rng.integers(0, 6, 30)
    got = mt.index_add(
        mt.Tensor(base.T.copy(), dtype=dtype),
        1,
        mt.Tensor(columns, dtype="int64"),
        mt.Tensor(across, dtype=dtype),
    )
    index = np.broadcast_to(columns[None, :], (width, 30))
    assert got.tolist() == _scattered(base.T, index, across, 1, dtype)

    flat = rng.integers(0, base.size, 200)
    values = _draw(rng, 200, dtype)
    got = mt.put(
        mt.Tensor(base, dtype=dtype),
        mt.Tensor(flat, dtype="int64"),
        mt.Tensor(values, dtype=dtype),
        accumulate=True,
    )
    want = _scattered(base.ravel(), flat, values, 0, dtype)
    assert got.reshape(-1).tolist() == want


def _exact(value: Fraction, dtype: str) -> float:
    return (
        _float32(value) if dtype == "float32" else value.numerator / value.denominator
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("alpha", [0.1, 1.0 / 3.0, -2.5e-3, 7.0])
def test_index_add_sums_the_exact_scaled_values(dtype, alpha):
    # `alpha` at the tensor's width, as a literal beside it is; each
    # destination the float nearest what it held plus `alpha` times each value
    # that arrives, exactly.
    rng = np.random.default_rng(1061)
    base, source = _draw(rng, (5, 4), dtype), _draw(rng, (30, 4), dtype)
    rows = rng.integers(0, 5, 30)
    got = mt.index_add(
        mt.Tensor(base, dtype=dtype),
        0,
        mt.Tensor(rows, dtype="int64"),
        mt.Tensor(source, dtype=dtype),
        alpha=alpha,
    ).tolist()
    scale = Fraction(float(np.array(alpha, dtype=dtype)))
    want = [
        [
            _exact(
                Fraction(float(base[r, c]))
                + sum(
                    scale * Fraction(float(source[i, c]))
                    for i in np.flatnonzero(rows == r)
                ),
                dtype,
            )
            for c in range(4)
        ]
        for r in range(5)
    ]
    assert got == want


def test_a_scaled_scatter_keeps_its_zeros_and_infinities():
    def index_add(base, rows, source):
        return mt.index_add(
            mt.Tensor(np.array(base)),
            0,
            mt.Tensor(np.array(rows), dtype="int64"),
            mt.Tensor(np.array(source)),
            alpha=0.1,
        ).tolist()

    got = index_add([-0.0, 0.0], [0, 0, 1], [-0.0, -0.0, -0.0])
    assert [math.copysign(1.0, v) for v in got] == [-1.0, 1.0] and got == [0.0, 0.0]
    got = index_add([0.0, 1.0, 2.0], [0, 1, 1, 2], [math.inf, 3.0, -math.inf, math.nan])
    assert got[0] == math.inf and got[1] == -math.inf and math.isnan(got[2])


@pytest.mark.parametrize("dtype", DTYPES)
def test_a_weighted_embedding_bag_sums_the_exact_products(dtype):
    rng = np.random.default_rng(1063)
    table, weights = _draw(rng, (6, 3), dtype), _draw(rng, 40, dtype)
    tokens = rng.integers(0, 6, 40)
    offsets = np.array([0, 9, 9, 25])
    got = mt.nn.embedding_bag(
        mt.Tensor(tokens, dtype="int64"),
        mt.Tensor(table, dtype=dtype),
        mt.Tensor(offsets, dtype="int64"),
        mode="sum",
        per_sample_weights=mt.Tensor(weights, dtype=dtype),
    ).tolist()
    ends = list(offsets[1:]) + [40]
    want = [
        [
            _exact(
                sum(
                    (
                        Fraction(float(weights[i]))
                        * Fraction(float(table[tokens[i], j]))
                        for i in range(start, end)
                    ),
                    Fraction(0),
                ),
                dtype,
            )
            for j in range(3)
        ]
        for start, end in zip(offsets, ends)
    ]
    assert got == want


def test_the_gradients_of_a_scaled_scatter():
    base = mt.Tensor(np.zeros((3, 2)), requires_grad=True)
    source = mt.Tensor(np.arange(8.0).reshape(4, 2) + 0.5, requires_grad=True)
    out = mt.index_add(
        base, 0, mt.Tensor(np.array([0, 2, 0, 1]), dtype="int64"), source, alpha=0.1
    )
    seed = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    (out * mt.Tensor(seed)).sum().backward()
    assert base.grad.tolist() == seed.tolist()
    assert source.grad.tolist() == (np.float64(0.1) * seed[[0, 2, 0, 1]]).tolist()

    table = mt.Tensor(
        np.array([[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]]), requires_grad=True
    )
    weights = mt.Tensor(np.array([0.5, -1.25, 3.0]), requires_grad=True)
    bag = mt.nn.embedding_bag(
        mt.Tensor(np.array([1, 2, 1]), dtype="int64"),
        table,
        mt.Tensor(np.array([0]), dtype="int64"),
        mode="sum",
        per_sample_weights=weights,
    )
    (bag * mt.Tensor(np.array([[2.0, -1.0]]))).sum().backward()
    assert weights.grad.tolist() == [7.0, -3.0, 7.0]
    assert table.grad.tolist() == [[0.0, 0.0], [7.0, -3.5], [-2.5, 1.25]]


@pytest.mark.parametrize("dtype", DTYPES)
def test_one_value_at_a_destination_is_one_addition(dtype):
    rng = np.random.default_rng(1049)
    base, source = _draw(rng, (40, 3), dtype), _draw(rng, (40, 3), dtype)
    index = np.stack([rng.permutation(40) for _ in range(3)], axis=1)
    got = mt.scatter_add(
        mt.Tensor(base, dtype=dtype),
        0,
        mt.Tensor(index, dtype="int64"),
        mt.Tensor(source, dtype=dtype),
    ).numpy()
    want = base.copy()
    np.add.at(want, (index, np.arange(3)[None, :]), source)
    assert np.array_equal(got, want)


def test_zeros_and_special_values():
    def scatter(base, index, source):
        return mt.scatter_add(
            mt.Tensor(np.array(base, dtype=np.float32)),
            0,
            mt.Tensor(np.array(index), dtype="int64"),
            mt.Tensor(np.array(source, dtype=np.float32)),
        ).tolist()

    # A zero is negative only when every value added, the destination's own
    # among them, is a negative zero; values that cancel leave a positive one.
    got = scatter([-0.0, 0.0, -0.0, -0.0], [0, 0, 1, 1, 2, 2], [-0.0] * 5 + [0.0])
    assert [math.copysign(1.0, v) for v in got] == [-1.0, 1.0, 1.0, -1.0]
    got = scatter([0.0, -0.0], [0, 0, 1, 1], [1.5, -1.5, 2.0**-140, -(2.0**-140)])
    assert got == [0.0, 0.0] and [math.copysign(1.0, v) for v in got] == [1.0, 1.0]
    # Infinities and NaNs sum as they would one by one.
    got = scatter(
        [1.0, 2.0, 3.0],
        [0, 0, 1, 1, 2, 2],
        [math.inf, 1.0, math.inf, -math.inf, math.nan, 1.0],
    )
    assert got[0] == math.inf and math.isnan(got[1]) and math.isnan(got[2])
    # A sum past the largest float32 on the way that comes back is finite.
    big = float(np.finfo(np.float32).max)
    assert scatter([0.0], [0, 0, 0], [big, big, -big]) == [big]


def test_the_gradient_of_a_sum_that_leaves_the_destination_out():
    # Where something arrives the destination's own value is replaced, so it
    # takes no gradient there; where nothing does it passes through.
    base = mt.Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    source = mt.Tensor(np.array([10.0, 20.0]), requires_grad=True)
    index = mt.Tensor(np.array([0, 0]), dtype="int64")
    out = mt.scatter_reduce(base, 0, index, source, "sum", include_self=False)
    assert out.tolist() == [30.0, 2.0, 3.0]
    (out * mt.Tensor(np.array([5.0, 7.0, 11.0]))).sum().backward()
    assert base.grad.tolist() == [0.0, 7.0, 11.0]
    assert source.grad.tolist() == [5.0, 5.0]


@pytest.mark.parametrize("dtype", DTYPES)
def test_fold_and_embedding_bag(dtype):
    rng = np.random.default_rng(1051)
    # Blocks of 3x3 at stride 1 over 6x7: an inner position is read by nine.
    blocks = _draw(rng, (2, 2 * 9, 4 * 5), dtype)
    got = mt.nn.fold(mt.Tensor(blocks, dtype=dtype), (6, 7), 3).numpy()
    lists: dict = {}
    for n, packed, block in np.ndindex(*blocks.shape):
        channel, tap = divmod(packed, 9)
        row, column = block // 5 + tap // 3, block % 5 + tap % 3
        lists.setdefault((n, channel, row, column), []).append(blocks[n, packed, block])
    for place, values in lists.items():
        assert got[place] == _nearest(values, dtype), place

    table = _draw(rng, (12, 5), dtype)
    tokens = rng.integers(0, 12, (7, 9))
    for mode in ("sum", "mean"):
        got = mt.nn.embedding_bag(
            mt.Tensor(tokens, dtype="int64"), mt.Tensor(table, dtype=dtype), mode=mode
        ).tolist()
        over = 9 if mode == "mean" else 1
        want = [
            [_nearest(table[tokens[bag], j], dtype, over) for j in range(5)]
            for bag in range(7)
        ]
        assert got == want, mode


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_gradient_of_gather(dtype):
    rng = np.random.default_rng(1061)
    x0 = _draw(rng, (5, 6), dtype)
    index = rng.integers(0, 6, (5, 40))
    upstream = _draw(rng, (5, 40), dtype)
    x = mt.Tensor(x0, dtype=dtype, requires_grad=True)
    picked = mt.gather(x, 1, mt.Tensor(index, dtype="int64"))
    (picked * mt.Tensor(upstream, dtype=dtype)).sum().backward()
    lists = _arrivals(index, upstream, 1)
    want = [
        [_nearest(lists.get((i, j), [0.0]), dtype) for j in range(6)] for i in range(5)
    ]
    assert x.grad.tolist() == want


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("width", [4, 48])
def test_the_gradient_of_index_select_and_an_embedding(dtype, width):
    rng = np.random.default_rng(1063 + width)
    table0 = _draw(rng, (9, width), dtype)
    # Row 8 is never read, row 7 once, the rest several times.
    tokens = np.concatenate([rng.integers(0, 7, 60), [7]])
    upstream = _draw(rng, (61, width), dtype)
    want = [
        [_nearest(upstream[tokens == row, j], dtype) for j in range(width)]
        for row in range(9)
    ]

    table = mt.Tensor(table0, dtype=dtype, requires_grad=True)
    rows = mt.index_select(table, 0, mt.Tensor(tokens, dtype="int64"))
    (rows * mt.Tensor(upstream, dtype=dtype)).sum().backward()
    assert table.grad.tolist() == want

    table = mt.Tensor(table0, dtype=dtype, requires_grad=True)
    rows = mt.nn.embedding(mt.Tensor(tokens, dtype="int64"), table)
    (rows * mt.Tensor(upstream, dtype=dtype)).sum().backward()
    assert table.grad.tolist() == want

    # Along a later axis, with rows either side of it.
    x0 = _draw(rng, (3, 9, width), dtype)
    upstream = _draw(rng, (3, 61, width), dtype)
    x = mt.Tensor(x0, dtype=dtype, requires_grad=True)
    picked = mt.index_select(x, 1, mt.Tensor(tokens, dtype="int64"))
    (picked * mt.Tensor(upstream, dtype=dtype)).sum().backward()
    got = x.grad.numpy()
    for n in range(3):
        for row in range(9):
            for j in range(0, width, 3):
                values = upstream[n, tokens == row, j]
                assert got[n, row, j] == _nearest(values, dtype), (n, row, j)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("width", [2, 20])
def test_the_gradient_of_repeated_elements(dtype, width):
    rng = np.random.default_rng(1069 + width)
    x0 = _draw(rng, (4, width), dtype)
    repeats = [3, 0, 1, 7]
    upstream = _draw(rng, (11, width), dtype)
    x = mt.Tensor(x0, dtype=dtype, requires_grad=True)
    out = mt.repeat_interleave(x, mt.Tensor(np.array(repeats), dtype="int64"), 0)
    (out * mt.Tensor(upstream, dtype=dtype)).sum().backward()
    starts = np.concatenate([[0], np.cumsum(repeats)])
    want = [
        [
            _nearest(upstream[starts[row] : starts[row + 1], j], dtype)
            for j in range(width)
        ]
        for row in range(4)
    ]
    assert x.grad.tolist() == want


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_gradient_of_a_value_broadcast_into_an_assignment(dtype):
    rng = np.random.default_rng(1087)
    upstream = _draw(rng, (30, 4), dtype)
    value = mt.Tensor(_draw(rng, 4, dtype), dtype=dtype, requires_grad=True)
    target = mt.Tensor(np.zeros((30, 4), dtype=dtype), dtype=dtype)
    target[:] = value
    (target * mt.Tensor(upstream, dtype=dtype)).sum().backward()
    assert value.grad.tolist() == [_nearest(upstream[:, j], dtype) for j in range(4)]
