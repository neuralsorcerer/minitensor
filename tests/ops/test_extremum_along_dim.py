# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The value-only extremum along a dimension folds in lanes, and NaN survives it.

`float_extremum_all!` already split the whole-tensor `max`/`min` across several
independent accumulators, because a single one makes the compare-and-select a
serial dependency chain that cannot vectorize. The along-a-dimension fold never
got the same treatment: it walked its column one element at a time, testing each
for NaN and breaking out of the loop on one -- a data-dependent branch. So on a
4096x1024 f32 tensor:

    max along the last axis    2.86 ms      sum along the same axis   0.21 ms

for the same single pass. The wide-axis fold had it too, from the same closure:
`max` along dimension 0 cost 3.60 ms against `sum`'s 0.41 ms. On this machine,
after giving both the lane treatment:

                                    before     after     sum, for scale
    max along the last axis        2.863 ms   0.377 ms   0.244 ms
    max along dimension 0          3.596 ms   0.727 ms   0.414 ms
    max, middle axis of a 3-D      1.539 ms   0.742 ms

`norm(inf, dim)` is exactly this reduction: 3.52 ms to 0.67 ms along the last
axis, a 5.3x gain.

The lane split is what these tests are really about. NaN is now tracked as a
separate flag so the value loop can stay a bare comparison -- `v > best` is
false for a NaN, so a NaN never displaces a real value and the flag decides the
result at the end -- and a fold that gets that wrong loses NaN propagation
entirely. So the lengths below straddle the lane counts (8 for f32, 4 for f64)
and the special values are placed in a full block *and* in the ragged tail,
which a separate remainder loop handles.

Three layouts reach three different kernels, and the shapes below cover all of
them: a contiguous column (`inner == 1`), a wide axis split into row bands and
merged, and a wide axis with several slabs to hand out. The merge is the one
place a NaN can be dropped after being found correctly, so a NaN has to land in
every band position, not just the first.

Reached from Python through `norm(+/-inf, dim)`, which is `max|x|` and `min|x|`
along the dim, and through `logsumexp`, which takes the column max for stability.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

DTYPES = ["float32", "float64"]

# Straddles both lane counts and their remainders: 8 for f32, 4 for f64.
LENGTHS = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 1000]

# Chosen so that between them the dims reach all three kernels: a contiguous
# column (`inner == 1`), a narrow strided one, and a wide axis (`inner >= 256`)
# both as a single slab and as several.
SHAPES = [
    (7,),
    (8,),
    (9,),
    (4, 8),
    (4, 9),
    (3, 4, 5),
    (2, 3, 4, 5),
    (64, 64),
    (3, 300),  # dim 0: wide axis, one slab
    (300, 3),  # dim 0: narrow, strided
    (2, 3, 400),  # dim 1: wide axis, two slabs
    (5, 600),  # dim 0: wide axis, one slab, more rows than threads
]


def _base(shape, dtype, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) * 10).astype(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("position", [0, 1, -2, -1])
def test_a_nan_anywhere_in_the_column_propagates(length, position, dtype):
    """Including one that lands in the ragged tail past the last full lane
    block, which a separate loop handles."""
    if not -length <= position < length:
        pytest.skip("position does not exist at this length")
    values = _base((length,), dtype)
    values[position] = np.nan

    tensor = mt.Tensor(values, dtype=dtype)

    assert np.isnan(tensor.norm(float("inf"), 0).numpy())
    assert np.isnan(tensor.norm(float("-inf"), 0).numpy())


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("length", LENGTHS)
def test_no_nan_means_no_nan(length, dtype):
    """The other half: the flag must not fire on ordinary data."""
    values = _base((length,), dtype)
    tensor = mt.Tensor(values, dtype=dtype)

    np.testing.assert_allclose(
        tensor.norm(float("inf"), 0).numpy(), np.abs(values).max(), rtol=1e-6
    )
    np.testing.assert_allclose(
        tensor.norm(float("-inf"), 0).numpy(), np.abs(values).min(), rtol=1e-6
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES, ids=[str(s) for s in SHAPES])
def test_extremum_along_every_dim_matches_numpy(shape, dtype):
    values = _base(shape, dtype)
    tensor = mt.Tensor(values, dtype=dtype)
    magnitudes = np.abs(values.astype(np.float64))

    for dim in range(len(shape)):
        np.testing.assert_allclose(
            tensor.norm(float("inf"), dim).numpy(),
            magnitudes.max(axis=dim),
            rtol=1e-6,
            err_msg=f"{shape} dim={dim}",
        )
        np.testing.assert_allclose(
            tensor.norm(float("-inf"), dim).numpy(),
            magnitudes.min(axis=dim),
            rtol=1e-6,
            err_msg=f"{shape} dim={dim}",
        )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES, ids=[str(s) for s in SHAPES])
@pytest.mark.parametrize("where", ["first", "middle", "last"])
def test_only_the_column_holding_the_nan_goes_nan(shape, where, dtype):
    """A lane fold that mixed its accumulators up would smear one column's NaN
    across its neighbours."""
    values = _base(shape, dtype)
    flat = values.reshape(-1)
    index = {"first": 0, "middle": flat.size // 2, "last": flat.size - 1}[where]
    flat[index] = np.nan

    tensor = mt.Tensor(values, dtype=dtype)
    magnitudes = np.abs(values.astype(np.float64))

    for dim in range(len(shape)):
        got = tensor.norm(float("inf"), dim).numpy()
        expected = magnitudes.max(axis=dim)
        np.testing.assert_array_equal(np.isnan(got), np.isnan(expected))
        finite = ~np.isnan(expected)
        np.testing.assert_allclose(got[finite], expected[finite], rtol=1e-6)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("row", [0, 1, 7, 31, 63, 99])
def test_a_nan_survives_the_band_merge_from_any_row(row, dtype):
    """A wide axis with one slab is folded in row bands and the partials are
    merged afterwards. The merge is the one place a NaN can be found correctly
    and then dropped -- a bare comparison against a NaN partial is false -- and
    only a NaN outside the first band exercises it."""
    rows, cols = 100, 400  # cols >= 256 puts this on the wide-axis path
    values = _base((rows, cols), dtype)
    values[row, 5] = np.nan

    got = mt.Tensor(values, dtype=dtype).norm(float("inf"), 0).numpy()

    expected = np.abs(values.astype(np.float64)).max(axis=0)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(expected))
    assert np.isnan(got[5]), "the NaN column did not come back NaN"
    finite = ~np.isnan(expected)
    np.testing.assert_allclose(got[finite], expected[finite], rtol=1e-6)


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_band_merge_keeps_the_true_extremum(dtype):
    """The winner has to be found across bands, not within one. Placing it in
    each row in turn puts it in every band."""
    rows, cols = 100, 400
    for row in (0, 1, 50, 98, 99):
        values = np.zeros((rows, cols), dtype=dtype)
        values[row, :] = 7.5

        got = mt.Tensor(values, dtype=dtype).norm(float("inf"), 0).numpy()

        np.testing.assert_allclose(got, np.full(cols, 7.5), rtol=1e-6)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("length", [3, 8, 9, 17])
def test_infinities_are_ordered_not_swallowed(length, dtype):
    values = _base((length,), dtype)
    values[0] = np.inf
    tensor = mt.Tensor(values, dtype=dtype)
    assert np.isinf(tensor.norm(float("inf"), 0).numpy())

    values = _base((length,), dtype)
    values[0] = -np.inf
    tensor = mt.Tensor(values, dtype=dtype)
    # `|-inf|` is `inf`, so it wins the maximum magnitude.
    assert np.isinf(tensor.norm(float("inf"), 0).numpy())


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("length", [1, 8, 9, 17])
def test_an_all_equal_column_returns_that_value(length, dtype):
    """A degenerate fold where a lane that never updated would still read as
    the seed, `-inf` or `+inf`."""
    values = np.full(length, 2.5, dtype=dtype)
    tensor = mt.Tensor(values, dtype=dtype)

    np.testing.assert_allclose(tensor.norm(float("inf"), 0).numpy(), 2.5, rtol=1e-6)
    np.testing.assert_allclose(tensor.norm(float("-inf"), 0).numpy(), 2.5, rtol=1e-6)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("length", [1, 8, 9, 17])
def test_a_column_of_zeros_stays_zero(length, dtype):
    """The seed for the maximum is `-inf`; a lane left untouched would show."""
    values = np.zeros(length, dtype=dtype)
    tensor = mt.Tensor(values, dtype=dtype)

    assert tensor.norm(float("inf"), 0).numpy() == 0.0
    assert tensor.norm(float("-inf"), 0).numpy() == 0.0


@pytest.mark.parametrize("dtype", DTYPES)
def test_signed_zeros_compare_equal(dtype):
    values = np.array([-0.0, 0.0, -0.0], dtype=dtype)
    tensor = mt.Tensor(values, dtype=dtype)

    assert tensor.norm(float("inf"), 0).numpy() == 0.0
    assert tensor.norm(float("-inf"), 0).numpy() == 0.0


# --- logsumexp, the other caller of the column maximum -----------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(9,), (4, 9), (3, 4, 5), (3, 300)])
def test_logsumexp_matches_a_stable_reference(shape, dtype):
    values = _base(shape, dtype)
    tensor = mt.Tensor(values, dtype=dtype)
    reference = values.astype(np.float64)

    for dim in range(len(shape)):
        peak = reference.max(axis=dim, keepdims=True)
        expected = np.squeeze(
            peak + np.log(np.exp(reference - peak).sum(axis=dim, keepdims=True)),
            axis=dim,
        )
        np.testing.assert_allclose(
            tensor.logsumexp(dim).numpy(), expected, rtol=1e-4, atol=1e-5
        )


@pytest.mark.parametrize("dtype", DTYPES)
def test_logsumexp_carries_the_nonfinite_cases(dtype):
    """`logsumexp` takes the column maximum for stability, so what that fold
    does with a non-finite value decides the answer."""

    def check(values, expected):
        got = float(mt.Tensor(np.array(values, dtype=dtype), dtype=dtype).logsumexp(0))
        if np.isnan(expected):
            assert np.isnan(got), f"{values} gave {got}"
        else:
            assert got == expected or np.isclose(got, expected), f"{values} gave {got}"

    check([np.inf, 1.0, 2.0], np.inf)
    check([np.inf], np.inf)
    check([-np.inf], -np.inf)
    check([-np.inf, -np.inf], -np.inf)
    check([np.nan, 1.0], np.nan)
    check([1.0, np.nan], np.nan)
    # A -inf term contributes exp(-inf) = 0 and must not poison the rest.
    finite = np.array([1.0, 2.0], dtype=np.float64)
    peak = finite.max()
    check([1.0, 2.0, -np.inf], peak + np.log(np.exp(finite - peak).sum()))
