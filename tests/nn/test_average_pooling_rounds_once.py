# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""An average pool gives each mean, and each gradient, as the float nearest
its exact value.

A window's mean is the exact sum of what it reads over what it divides by;
added in the element type and then divided, it took a rounding for each value
past the first and one more. The gradient of a cell is the sum, over the
windows that read it, of each window's gradient over its divisor; divided
window by window and then added, it took a rounding for each. Both are now
rounded once -- for fixed windows with a stride and padding, the padding
counted or not, for adaptive windows of uneven sizes, for the 1-D forms, and
for windows of volumes, which were pooled a depth tap at a time and the taps
added and scaled.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt

DTYPES = ["float32", "float64"]


def _nearest(value: Fraction, dtype: str) -> float:
    """The float of `dtype` nearest `value`, ties to even."""

    if dtype == "float64":
        return value.numerator / value.denominator
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


def _fixed(size, kernel, stride, padding):
    """The positions each window of a fixed size reads along one axis."""

    out = (size + 2 * padding - kernel) // stride + 1
    return [
        [
            i * stride + k - padding
            for k in range(kernel)
            if 0 <= i * stride + k - padding < size
        ]
        for i in range(out)
    ]


def _adaptive(size, out):
    """The positions each adaptive window reads along one axis."""

    return [list(range(i * size // out, -(-(i + 1) * size // out))) for i in range(out)]


def _exact(x, g, rows, cols, fixed_divisor, dtype):
    """The pooled means of `x` and the gradient of `sum(pooled * g)` for `x`,
    from the definition: every sum exact, and each entry rounded once."""

    x = np.asarray(x)
    g = np.asarray(g)
    means = np.zeros(x.shape[:2] + (len(rows), len(cols)), dtype=dtype)
    gradient = np.zeros(x.shape, dtype=dtype)
    for n in range(x.shape[0]):
        for c in range(x.shape[1]):
            shares = [
                [Fraction(0) for _ in range(x.shape[3])] for _ in range(x.shape[2])
            ]
            for i, r in enumerate(rows):
                for j, s in enumerate(cols):
                    divisor = fixed_divisor or len(r) * len(s)
                    if divisor == 0:
                        continue
                    total = sum(
                        (Fraction(float(x[n, c, a, b])) for a in r for b in s),
                        Fraction(0),
                    )
                    means[n, c, i, j] = _nearest(total / divisor, dtype)
                    share = Fraction(float(g[n, c, i, j])) / divisor
                    for a in r:
                        for b in s:
                            shares[a][b] += share
            for a in range(x.shape[2]):
                for b in range(x.shape[3]):
                    gradient[n, c, a, b] = _nearest(shares[a][b], dtype)
    return means, gradient


def _exact_volume(x, g, depths, rows, cols, fixed_divisor, dtype):
    """As `_exact`, for windows of volumes: `depths`, `rows` and `cols` the
    positions each slice, row and column of windows reads."""

    x = np.asarray(x)
    g = np.asarray(g)
    means = np.zeros(x.shape[:2] + (len(depths), len(rows), len(cols)), dtype=dtype)
    gradient = np.zeros(x.shape, dtype=dtype)
    for n in range(x.shape[0]):
        for c in range(x.shape[1]):
            shares = {}
            for i, d in enumerate(depths):
                for j, r in enumerate(rows):
                    for k, s in enumerate(cols):
                        cells = [(a, b, e) for a in d for b in r for e in s]
                        divisor = fixed_divisor or len(cells)
                        total = sum(
                            (Fraction(float(x[(n, c) + cell])) for cell in cells),
                            Fraction(0),
                        )
                        means[n, c, i, j, k] = _nearest(total / divisor, dtype)
                        share = Fraction(float(g[n, c, i, j, k])) / divisor
                        for cell in cells:
                            shares[cell] = shares.get(cell, Fraction(0)) + share
            for cell in np.ndindex(*x.shape[2:]):
                gradient[(n, c) + cell] = _nearest(shares.get(cell, Fraction(0)), dtype)
    return means, gradient


def _same(got, want, what):
    """`got` is `want` bit for bit, the sign of a zero included."""

    got = np.ascontiguousarray(got)
    want = np.ascontiguousarray(want)
    assert got.dtype == want.dtype and got.shape == want.shape, what
    differ = got.view(np.uint8).reshape(got.size, -1) != want.view(np.uint8).reshape(
        want.size, -1
    )
    count = int(differ.any(axis=1).sum())
    assert count == 0, f"{what}: {count} of {got.size} entries differ"


def _values(rng, shape, dtype):
    """Values of mixed magnitudes and signs, with zeros among them."""

    values = rng.standard_normal(shape) * 2.0 ** rng.integers(-6, 7, shape)
    values[rng.random(shape) < 0.08] = 0.0
    return values.astype(dtype)


def _pool_and_gradient(pool, x, g):
    """`pool(x)` and the gradient of `sum(pool(x) * g)` for `x`."""

    tensor = mt.tensor(x, requires_grad=True)
    pooled = pool(tensor)
    (pooled * mt.tensor(g)).sum().backward()
    return pooled.numpy(), tensor.grad.numpy()


FIXED = [
    # input, kernel, stride, padding, count_include_pad
    ((7, 13), (3, 3), (1, 1), (1, 1), True),
    ((7, 13), (3, 3), (1, 1), (1, 1), False),
    ((8, 15), (2, 3), (2, 2), (1, 1), False),
    ((8, 15), (2, 3), (2, 2), (1, 1), True),
    ((7, 7), (3, 3), (2, 2), (1, 1), True),
    ((6, 6), (2, 2), (2, 2), (0, 0), True),
    ((9, 20), (2, 2), (2, 2), (0, 0), True),
    ((5, 12), (5, 4), (1, 1), (2, 2), False),
    ((4, 12), (2, 2), (3, 1), (0, 1), True),
    ((2, 2), (4, 4), (1, 1), (2, 2), False),
    ((2, 9), (4, 4), (1, 1), (2, 2), False),
    ((3, 30), (3, 1), (1, 1), (0, 0), True),
    ((6, 9), (6, 9), (1, 1), (0, 0), True),
    ((5, 16), (1, 7), (1, 3), (0, 3), False),
    ((6, 11), (3, 2), (1, 1), (0, 0), True),
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("size,kernel,stride,padding,count_include_pad", FIXED)
def test_fixed_windows_round_once(
    dtype, size, kernel, stride, padding, count_include_pad
):
    rng = np.random.default_rng(sum(size) + 7 * kernel[0] + stride[1])
    x = _values(rng, (2, 2) + size, dtype)
    rows = _fixed(size[0], kernel[0], stride[0], padding[0])
    cols = _fixed(size[1], kernel[1], stride[1], padding[1])
    g = _values(rng, (2, 2, len(rows), len(cols)), dtype)
    means, gradient = _pool_and_gradient(
        lambda t: mt.nn.avg_pool2d(
            t, kernel, stride, padding, count_include_pad=count_include_pad
        ),
        x,
        g,
    )
    fixed_divisor = kernel[0] * kernel[1] if count_include_pad else None
    want_means, want_gradient = _exact(x, g, rows, cols, fixed_divisor, dtype)
    _same(means, want_means, "means")
    _same(gradient, want_gradient, "gradient")


ADAPTIVE = [
    ((7, 10), (3, 4)),
    ((5, 5), (7, 9)),
    ((8, 12), (4, 3)),
    ((6, 6), (1, 1)),
    ((9, 16), (3, 16)),
    ((10, 7), (4, 7)),
    ((3, 17), (2, 11)),
    ((12, 12), (6, 6)),
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("size,out", ADAPTIVE)
def test_adaptive_windows_round_once(dtype, size, out):
    rng = np.random.default_rng(sum(size) + 3 * out[0] + out[1])
    x = _values(rng, (2, 2) + size, dtype)
    g = _values(rng, (2, 2) + out, dtype)
    means, gradient = _pool_and_gradient(
        lambda t: mt.nn.adaptive_avg_pool2d(t, out), x, g
    )
    rows, cols = _adaptive(size[0], out[0]), _adaptive(size[1], out[1])
    want_means, want_gradient = _exact(x, g, rows, cols, None, dtype)
    _same(means, want_means, "means")
    _same(gradient, want_gradient, "gradient")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "length,kernel,stride,padding,count_include_pad",
    [
        (23, 3, 1, 1, True),
        (23, 3, 1, 1, False),
        (24, 4, 2, 2, False),
        (20, 5, 5, 0, True),
        (9, 9, 1, 0, True),
    ],
)
def test_one_dimensional_windows_round_once(
    dtype, length, kernel, stride, padding, count_include_pad
):
    rng = np.random.default_rng(length + kernel)
    x = _values(rng, (2, 3, length), dtype)
    cols = _fixed(length, kernel, stride, padding)
    g = _values(rng, (2, 3, len(cols)), dtype)
    means, gradient = _pool_and_gradient(
        lambda t: mt.nn.avg_pool1d(
            t, kernel, stride, padding, count_include_pad=count_include_pad
        ),
        x,
        g,
    )
    want_means, want_gradient = _exact(
        x[:, :, None, :],
        g[:, :, None, :],
        [[0]],
        cols,
        kernel if count_include_pad else None,
        dtype,
    )
    _same(means, want_means[:, :, 0, :], "means")
    _same(gradient, want_gradient[:, :, 0, :], "gradient")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("length,out", [(17, 5), (6, 9), (12, 4), (8, 1)])
def test_one_dimensional_adaptive_windows_round_once(dtype, length, out):
    rng = np.random.default_rng(length + out)
    x = _values(rng, (2, 3, length), dtype)
    g = _values(rng, (2, 3, out), dtype)
    means, gradient = _pool_and_gradient(
        lambda t: mt.nn.adaptive_avg_pool1d(t, out), x, g
    )
    want_means, want_gradient = _exact(
        x[:, :, None, :], g[:, :, None, :], [[0]], _adaptive(length, out), None, dtype
    )
    _same(means, want_means[:, :, 0, :], "means")
    _same(gradient, want_gradient[:, :, 0, :], "gradient")


VOLUMES = [
    # input, kernel, stride, padding, count_include_pad
    ((4, 5, 6), (2, 3, 3), (1, 1, 2), (1, 1, 1), True),
    ((4, 5, 6), (2, 3, 3), (1, 1, 2), (1, 1, 1), False),
    ((5, 4, 7), (3, 2, 2), (2, 1, 1), (1, 0, 1), False),
    ((4, 4, 4), (2, 2, 2), (2, 2, 2), (0, 0, 0), True),
    ((3, 5, 5), (3, 3, 3), (1, 2, 2), (1, 1, 1), False),
    ((6, 3, 4), (5, 1, 1), (1, 1, 1), (2, 0, 0), True),
    ((2, 3, 24), (2, 2, 3), (1, 1, 1), (1, 1, 1), False),
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("size,kernel,stride,padding,count_include_pad", VOLUMES)
def test_windows_of_volumes_round_once(
    dtype, size, kernel, stride, padding, count_include_pad
):
    rng = np.random.default_rng(sum(size) + 5 * kernel[0] + stride[2])
    x = _values(rng, (1, 2) + size, dtype)
    depths, rows, cols = (
        _fixed(size[axis], kernel[axis], stride[axis], padding[axis])
        for axis in range(3)
    )
    g = _values(rng, (1, 2, len(depths), len(rows), len(cols)), dtype)
    means, gradient = _pool_and_gradient(
        lambda t: mt.nn.avg_pool3d(t, kernel, stride, padding, count_include_pad),
        x,
        g,
    )
    fixed_divisor = kernel[0] * kernel[1] * kernel[2] if count_include_pad else None
    want_means, want_gradient = _exact_volume(
        x, g, depths, rows, cols, fixed_divisor, dtype
    )
    _same(means, want_means, "means")
    _same(gradient, want_gradient, "gradient")


@pytest.mark.parametrize("dtype", DTYPES)
def test_a_mean_of_a_volume_halfway_between_two_floats_goes_to_the_even_one(dtype):
    # Two slices a last place apart: each mean of the pair is a halfway point.
    step = _last_place(dtype)
    x = np.stack([np.ones((3, 4)), np.full((3, 4), 1 + step)]).astype(dtype)
    pooled = mt.nn.avg_pool3d(mt.tensor(x.reshape(1, 1, 2, 3, 4)), (2, 1, 1))
    assert (pooled.numpy() == np.array(1.0, dtype=dtype)).all()
    x = np.stack([np.full((3, 4), 1 + step), np.full((3, 4), 1 + 2 * step)])
    pooled = mt.nn.avg_pool3d(
        mt.tensor(x.astype(dtype).reshape(1, 1, 2, 3, 4)), (2, 1, 1)
    )
    assert (pooled.numpy() == np.array(1 + 2 * step, dtype=dtype)).all()


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_layers_round_as_the_functions_do(dtype):
    rng = np.random.default_rng(5)
    x = _values(rng, (1, 2, 9, 11), dtype)
    rows, cols = _fixed(9, 3, 2, 1), _fixed(11, 3, 2, 1)
    g = np.ones((1, 2, len(rows), len(cols)), dtype=dtype)
    layer = mt.nn.AvgPool2d(3, stride=2, padding=1)
    means, gradient = _pool_and_gradient(layer, x, g)
    want_means, want_gradient = _exact(x, g, rows, cols, 9, dtype)
    _same(means, want_means, "means")
    _same(gradient, want_gradient, "gradient")

    layer = mt.nn.AdaptiveAvgPool2d((4, 5))
    g = np.ones((1, 2, 4, 5), dtype=dtype)
    means, gradient = _pool_and_gradient(layer, x, g)
    want_means, want_gradient = _exact(
        x, g, _adaptive(9, 4), _adaptive(11, 5), None, dtype
    )
    _same(means, want_means, "means")
    _same(gradient, want_gradient, "gradient")


@pytest.mark.parametrize("dtype", DTYPES)
def test_a_wide_plane_of_many_windows_rounds_once(dtype):
    # Enough windows across to be summed side by side many times over.
    rng = np.random.default_rng(11)
    x = _values(rng, (1, 1, 12, 150), dtype)
    for kernel, stride, padding, count_include_pad in [
        ((3, 3), (1, 1), (1, 1), False),
        ((2, 2), (2, 2), (0, 0), True),
        ((3, 5), (2, 1), (1, 2), True),
    ]:
        rows = _fixed(12, kernel[0], stride[0], padding[0])
        cols = _fixed(150, kernel[1], stride[1], padding[1])
        g = _values(rng, (1, 1, len(rows), len(cols)), dtype)
        means, gradient = _pool_and_gradient(
            lambda t: mt.nn.avg_pool2d(
                t, kernel, stride, padding, count_include_pad=count_include_pad
            ),
            x,
            g,
        )
        fixed_divisor = kernel[0] * kernel[1] if count_include_pad else None
        want_means, want_gradient = _exact(x, g, rows, cols, fixed_divisor, dtype)
        _same(means, want_means, f"means of {kernel}")
        _same(gradient, want_gradient, f"gradient of {kernel}")


def _last_place(dtype):
    """The gap between one and the float after it."""

    return float(np.finfo(dtype).eps)


@pytest.mark.parametrize("dtype", DTYPES)
def test_means_halfway_between_two_floats_go_to_the_even_one(dtype):
    step = _last_place(dtype)
    # Windows of two: a mean halfway between neighbours a last place apart.
    row = np.array([1 + step * i for i in range(12)], dtype=dtype)
    x = np.tile(row, (1, 1, 2, 1))
    means = mt.nn.avg_pool2d(mt.tensor(x), (1, 2), (1, 1)).numpy()
    want, _ = _exact(
        x, np.zeros((1, 1, 2, 11)), _fixed(2, 1, 1, 0), _fixed(12, 2, 1, 0), 2, dtype
    )
    _same(means, want, "means of two")
    assert means[0, 0, 0, 0] == 1.0 and means[0, 0, 0, 1] == 1 + 2 * step

    # Windows of nine and of three: sums of nine and of three times a halfway
    # point. Nine and four and a half last places make a mean of one and half
    # a last place, which goes down to one; nine and thirteen and a half make
    # one and one and a half, which goes up to the even one past it.
    for extra, even in [(5, 1.0), (14, 1 + 2 * step)]:
        nine = np.array(
            [[1 + extra * step, 1 - step / 2, 1], [1, 1, 1], [1, 1, 1]], dtype=dtype
        ).reshape(1, 1, 3, 3)
        total = sum((Fraction(float(v)) for v in nine.ravel()), Fraction(0))
        assert total == 9 + Fraction(2 * extra - 1, 2) * Fraction(step)
        for pooled in [
            mt.nn.avg_pool2d(mt.tensor(nine), 3),
            mt.nn.adaptive_avg_pool2d(mt.tensor(nine), 1),
            mt.nn.avg_pool2d(mt.tensor(np.tile(nine, (1, 1, 1, 12))), 3),
        ]:
            assert (pooled.numpy() == np.array(even, dtype=dtype)).all()
    three = np.array([1, 2, 4.5 * step, 1, 2, 1.5 * step, 0, 0, 0], dtype=dtype)
    means = mt.nn.avg_pool1d(mt.tensor(three.reshape(1, 1, 9)), 3).numpy().ravel()
    # 1 + 1.5 and 1 + 0.5 last places: up to the even 1 + 2, down to the even 1.
    assert means.tolist() == [1 + 2 * step, 1.0, 0.0]
    wide = np.tile(three, (1, 1, 1, 4))
    means = mt.nn.avg_pool2d(mt.tensor(wide), (1, 3), (1, 3)).numpy().ravel()
    assert means.tolist() == [1 + 2 * step, 1.0, 0.0] * 4


@pytest.mark.parametrize("dtype", DTYPES)
def test_gradients_halfway_between_two_floats_go_to_the_even_one(dtype):
    step = _last_place(dtype)
    # Two windows of two on each cell, their halves a last place apart.
    g = np.array([1 + step * i for i in range(11)], dtype=dtype).reshape(1, 1, 1, 11)
    x = np.zeros((1, 1, 1, 12), dtype=dtype)
    _, gradient = _pool_and_gradient(
        lambda t: mt.nn.avg_pool2d(t, (1, 2), (1, 1)), x, g
    )
    _, want = _exact(x, g, [[0]], _fixed(12, 2, 1, 0), 2, dtype)
    _same(gradient, want, "gradient of halves")
    assert gradient[0, 0, 0, 1] == 1.0 and gradient[0, 0, 0, 2] == 1 + 2 * step


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kernel", [3, 15])
def test_shares_of_differing_divisors_meet_at_a_halfway_point(dtype, kernel):
    # The corner cell of a padded pool that leaves its padding out is read by
    # windows of several sizes. A third and a sixth from two windows of one
    # size, and a half and half a last place from a window of another, come to
    # one and half a last place exactly: the even neighbour, one. With three
    # halves of a last place they come to the even neighbour above.
    step = _last_place(dtype)
    pad = kernel // 2
    size = kernel + 2
    full = pad + 1
    rows = _fixed(size, kernel, 1, pad)
    x = np.zeros((1, 1, size, size), dtype=dtype)
    for halves, even in [(1, 1.0), (3, 1 + 2 * step)]:
        g = np.zeros((1, 1, size, size), dtype=dtype)
        # Window (0, 0) reads `full * full` cells, (0, 1) and (1, 0) one row
        # or column more.
        corner, edge = full * full, full * (full + 1)
        g[0, 0, 0, 0] = corner * (0.5 + halves * step / 2)
        g[0, 0, 0, 1] = edge / 3
        g[0, 0, 1, 0] = edge / 6
        assert float(g[0, 0, 0, 0]) == corner * (0.5 + halves * step / 2)
        _, gradient = _pool_and_gradient(
            lambda t: mt.nn.avg_pool2d(t, kernel, 1, pad, count_include_pad=False),
            x,
            g,
        )
        _, want = _exact(x, g, rows, rows, None, dtype)
        _same(gradient, want, f"gradient with {halves} halves")
        assert gradient[0, 0, 0, 0] == even


@pytest.mark.parametrize("dtype", DTYPES)
def test_zero_means_and_gradients_are_positive_zeros(dtype):
    x = np.full((1, 2, 4, 9), -0.0, dtype=dtype)
    x[0, 0, 1, 3], x[0, 0, 1, 4] = 1.5, -1.5
    g = np.full((1, 2, 4, 9), -0.0, dtype=dtype)
    g[0, 1, 2, 2], g[0, 1, 2, 3] = 3.0, -3.0
    pools = [
        lambda t: mt.nn.avg_pool2d(t, 3, 1, 1),
        lambda t: mt.nn.avg_pool2d(t, 3, 1, 1, count_include_pad=False),
        lambda t: mt.nn.avg_pool2d(t, (1, 2), (1, 1), (0, 1)),
    ]
    for pool in pools:
        tensor = mt.tensor(x, requires_grad=True)
        pooled = pool(tensor)
        shaped = np.resize(g, pooled.numpy().shape).astype(dtype)
        (pooled * mt.tensor(shaped)).sum().backward()
        for values in (pooled.numpy(), tensor.grad.numpy()):
            assert not np.signbit(values[values == 0]).any()
            assert (values == 0).any() and (values != 0).any()
    for out in [(2, 4), (4, 9), (1, 1), (5, 11)]:
        tensor = mt.tensor(x, requires_grad=True)
        pooled = mt.nn.adaptive_avg_pool2d(tensor, out)
        shaped = np.resize(g, pooled.numpy().shape).astype(dtype)
        (pooled * mt.tensor(shaped)).sum().backward()
        for values in (pooled.numpy(), tensor.grad.numpy()):
            assert not np.signbit(values[values == 0]).any()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("count_include_pad", [True, False])
def test_special_values_reach_only_what_reads_them(dtype, count_include_pad):
    x = np.ones((1, 1, 5, 12), dtype=dtype)
    x[0, 0, 1, 2] = np.inf
    x[0, 0, 3, 8] = -np.inf
    x[0, 0, 3, 9] = np.inf
    x[0, 0, 0, 11] = np.nan
    means = mt.nn.avg_pool2d(
        mt.tensor(x), 3, 1, 1, count_include_pad=count_include_pad
    ).numpy()
    rows, cols = _fixed(5, 3, 1, 1), _fixed(12, 3, 1, 1)
    for i, r in enumerate(rows):
        for j, s in enumerate(cols):
            window = [float(x[0, 0, a, b]) for a in r for b in s]
            got = float(means[0, 0, i, j])
            if any(math.isnan(v) for v in window) or (
                math.inf in window and -math.inf in window
            ):
                assert math.isnan(got), (i, j)
            elif math.inf in window or -math.inf in window:
                assert got == (math.inf if math.inf in window else -math.inf), (i, j)
            else:
                assert math.isfinite(got), (i, j)

    g = np.ones((1, 1, 5, 12), dtype=dtype)
    g[0, 0, 0, 0] = np.inf
    g[0, 0, 2, 6] = np.nan
    g[0, 0, 4, 10] = -np.inf
    g[0, 0, 4, 11] = np.inf
    tensor = mt.tensor(np.ones((1, 1, 5, 12), dtype=dtype), requires_grad=True)
    pooled = mt.nn.avg_pool2d(tensor, 3, 1, 1, count_include_pad=count_include_pad)
    pooled.backward(mt.tensor(g))
    gradient = tensor.grad.numpy()
    for a in range(5):
        for b in range(12):
            shares = [
                float(g[0, 0, i, j])
                for i, r in enumerate(rows)
                for j, s in enumerate(cols)
                if a in r and b in s
            ]
            got = float(gradient[0, 0, a, b])
            if any(math.isnan(v) for v in shares) or (
                math.inf in shares and -math.inf in shares
            ):
                assert math.isnan(got), (a, b)
            elif math.inf in shares or -math.inf in shares:
                assert got == (math.inf if math.inf in shares else -math.inf), (a, b)
            else:
                assert math.isfinite(got), (a, b)


def test_sums_past_the_float_range_still_divide_exactly():
    # A window whose sum overflows before it is divided, and a gradient
    # whose shares do.
    huge = float(np.finfo(np.float64).max)
    x = np.full((1, 1, 1, 12), huge, dtype=np.float64)
    means = mt.nn.avg_pool2d(mt.tensor(x), (1, 3), (1, 1), (0, 1)).numpy().ravel()
    assert means[5] == huge and means[0] == huge / 3 * 2
    tensor = mt.tensor(x, requires_grad=True)
    pooled = mt.nn.avg_pool2d(tensor, (1, 3), (1, 1), (0, 1))
    pooled.backward(mt.tensor(x))
    gradient = tensor.grad.numpy().ravel()
    assert gradient[5] == huge and gradient[0] == huge / 3 * 2

    huge = float(np.finfo(np.float32).max)
    x = np.full((1, 1, 1, 12), huge, dtype=np.float32)
    means = mt.nn.avg_pool2d(mt.tensor(x), (1, 3), (1, 1), (0, 1)).numpy().ravel()
    assert means[5] == np.float32(huge)
    assert means[0] == np.float32(_nearest(Fraction(huge) * 2 / 3, "float32"))
    assert float(mt.nn.adaptive_avg_pool2d(mt.tensor(x), 1).numpy().ravel()[0]) == huge


def test_shares_under_the_float_range_still_round_once():
    # Shares that fall below the least float, alone and together.
    for dtype in DTYPES:
        least = float(np.nextafter(np.zeros((), dtype=dtype), np.ones((), dtype=dtype)))
        g = np.full((1, 1, 1, 12), least, dtype=dtype)
        for count_include_pad in (True, False):
            tensor = mt.tensor(np.zeros((1, 1, 1, 12), dtype=dtype), requires_grad=True)
            pooled = mt.nn.avg_pool2d(
                tensor, (1, 3), (1, 1), (0, 1), count_include_pad=count_include_pad
            )
            pooled.backward(mt.tensor(g))
            _, want = _exact(
                np.zeros((1, 1, 1, 12)),
                g,
                [[0]],
                _fixed(12, 3, 1, 1),
                3 if count_include_pad else None,
                dtype,
            )
            _same(tensor.grad.numpy(), want, f"{dtype} shares of the least float")
            tensor = mt.tensor(np.zeros((1, 1, 1, 12), dtype=dtype), requires_grad=True)
            pooled = mt.nn.avg_pool2d(
                tensor, (1, 3), (1, 3), (0, 0), count_include_pad=count_include_pad
            )
            pooled.backward(mt.tensor(-g[:, :, :, :4]))
            gradient = tensor.grad.numpy()
            # A third of the least float, below zero: a negative zero.
            assert (gradient == 0).all() and np.signbit(gradient).all()
