# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Windows, sliding products, and the constructors that were missing.

A window function is five lines of arithmetic over sample positions, and the
*ends* are where a naive transcription goes wrong: a window meant for a
spectrum repeats seamlessly (`periodic=True`, what an FFT wants and what
`torch` defaults to) while one meant for filter design is symmetric about its
middle (`periodic=False`, what NumPy's `hanning` gives). A one-sample window is
1, not whatever the cosine happens to be at position zero.

`correlate` and `convolve` are the same sliding product read two ways -- the
second reverses one signal first, which is exactly the distinction the
machine-learning use of the word "convolution" loses. Every overlap is computed
once and the narrower modes are a window onto it; where that window starts is
not symmetric between the two, which is what these tests pin.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

LENGTHS = [0, 1, 2, 5, 8, 33, 64]


@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize(
    "ours,theirs",
    [
        (mt.hann_window, np.hanning),
        (mt.hamming_window, np.hamming),
        (mt.blackman_window, np.blackman),
        (mt.bartlett_window, np.bartlett),
    ],
    ids=["hann", "hamming", "blackman", "bartlett"],
)
def test_the_symmetric_windows_match_numpy(length, ours, theirs):
    np.testing.assert_allclose(
        ours(length, periodic=False).numpy(), theirs(length), atol=1e-12
    )


@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("beta", [0.0, 5.0, 8.6, 14.0])
def test_the_kaiser_window_matches_numpy(length, beta):
    np.testing.assert_allclose(
        mt.kaiser_window(length, periodic=False, beta=beta).numpy(),
        np.kaiser(length, beta),
        atol=1e-12,
    )


def test_a_periodic_window_is_the_symmetric_one_of_the_next_length():
    """That is what periodic means: the sample that would repeat is left off."""
    for length in [2, 5, 16]:
        np.testing.assert_allclose(
            mt.hann_window(length, periodic=True).numpy(),
            np.hanning(length + 1)[:-1],
            atol=1e-12,
        )


def test_a_window_of_one_sample_is_the_whole_window():
    for window in [
        mt.hann_window,
        mt.hamming_window,
        mt.blackman_window,
        mt.bartlett_window,
        mt.kaiser_window,
    ]:
        assert window(1).numpy().tolist() == [1.0]
        assert window(0).numpy().size == 0
    with pytest.raises(ValueError, match="non-negative"):
        mt.hann_window(-1)


@pytest.mark.parametrize("signal", [1, 2, 3, 4, 5, 8])
@pytest.mark.parametrize("kernel", [1, 2, 3, 4, 5, 7])
@pytest.mark.parametrize("mode", ["full", "same", "valid"])
def test_the_sliding_products_match_numpy(signal, kernel, mode):
    first = np.arange(1.0, signal + 1)
    second = np.arange(1.0, kernel + 1) * 0.5
    np.testing.assert_allclose(
        mt.convolve(mt.from_numpy(first), mt.from_numpy(second), mode).numpy(),
        np.convolve(first, second, mode),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        mt.correlate(mt.from_numpy(first), mt.from_numpy(second), mode).numpy(),
        np.correlate(first, second, mode),
        atol=1e-10,
    )


def test_convolve_is_correlate_with_one_signal_reversed():
    rng = np.random.default_rng(113)
    first = rng.standard_normal(9)
    second = rng.standard_normal(4)
    np.testing.assert_allclose(
        mt.convolve(mt.from_numpy(first), mt.from_numpy(second), "full").numpy(),
        mt.correlate(mt.from_numpy(first), mt.from_numpy(second[::-1].copy()), "full").numpy(),
        atol=1e-10,
    )


def test_the_sliding_products_refuse_nothing_to_slide():
    with pytest.raises(ValueError, match="non-empty"):
        mt.convolve(mt.from_numpy(np.array([], dtype=np.float64)), mt.from_numpy(np.ones(3)))
    with pytest.raises(ValueError, match="'full', 'same' or 'valid'"):
        mt.convolve(mt.from_numpy(np.ones(3)), mt.from_numpy(np.ones(2)), "most")


@pytest.mark.parametrize(
    "args",
    [(1, 1000, 4), (-1, -1000, 4), (0.5, 8, 5), (3, 9, 1), (1, 256, 9), (1e-5, 1e5, 11)],
    ids=str,
)
def test_geomspace_matches_numpy(args):
    np.testing.assert_allclose(mt.geomspace(*args).numpy(), np.geomspace(*args))


def test_geomspace_ends_exactly_where_it_was_told_to():
    """The exponential of a logarithm is not the number that went in."""
    values = mt.geomspace(3.0, 7.0, 9).numpy()
    assert values[0] == 3.0
    assert values[-1] == 7.0
    # One step is the start alone, not the end.
    assert mt.geomspace(3.0, 9.0, 1).numpy().tolist() == [3.0]

    for start, stop in [(0.0, 5.0), (-1.0, 5.0), (1.0, 0.0)]:
        with pytest.raises(ValueError):
            mt.geomspace(start, stop, 3)


def test_tri_indices_and_fromfunction_match_numpy():
    np.testing.assert_allclose(mt.tri(4).numpy(), np.tri(4))
    np.testing.assert_allclose(mt.tri(4, 5, -1).numpy(), np.tri(4, 5, -1))
    np.testing.assert_allclose(mt.tri(3, 6, 2).numpy(), np.tri(3, 6, 2))

    np.testing.assert_array_equal(mt.indices((2, 3)).numpy(), np.indices((2, 3)))
    np.testing.assert_array_equal(mt.indices((2, 3, 4)).numpy(), np.indices((2, 3, 4)))
    sparse = mt.indices((2, 3), sparse=True)
    assert [tuple(grid.shape) for grid in sparse] == [
        grid.shape for grid in np.indices((2, 3), sparse=True)
    ]

    np.testing.assert_allclose(
        mt.fromfunction(lambda i, j: i + j, (3, 4)).numpy(),
        np.fromfunction(lambda i, j: i + j, (3, 4)),
    )
    np.testing.assert_allclose(
        mt.fromfunction(lambda i, j: i * 2.0 - j, (3, 4)).numpy(),
        np.fromfunction(lambda i, j: i * 2.0 - j, (3, 4)),
    )


def test_ix_selects_an_open_mesh():
    values = np.arange(20).reshape(4, 5)
    rows = np.array([0, 2])
    columns = np.array([1, 3])
    grids = mt.ix_(mt.from_numpy(rows), mt.from_numpy(columns))
    expected = np.ix_(rows, columns)
    assert [tuple(grid.shape) for grid in grids] == [
        grid.shape for grid in expected
    ]
    # A boolean sequence becomes the positions it selects.
    mask = np.array([True, False, True, False])
    np.testing.assert_array_equal(
        mt.ix_(mt.from_numpy(mask), mt.from_numpy(columns))[0].numpy().reshape(-1),
        np.ix_(mask, columns)[0].reshape(-1),
    )
    with pytest.raises(TypeError, match="integer or boolean"):
        mt.ix_(mt.from_numpy(np.array([0.5, 1.5])))
    del values


def test_the_array_api_products_contract_the_axis_they_name():
    rng = np.random.default_rng(127)
    first, second = rng.standard_normal((3, 4)), rng.standard_normal((3, 4))
    np.testing.assert_allclose(
        mt.vecdot(mt.from_numpy(first), mt.from_numpy(second)).numpy(),
        np.vecdot(first, second),
    )
    matrices = rng.standard_normal((2, 3, 4))
    vectors = rng.standard_normal((2, 4))
    np.testing.assert_allclose(
        mt.matvec(mt.from_numpy(matrices), mt.from_numpy(vectors)).numpy(),
        np.matvec(matrices, vectors),
    )
    rows = rng.standard_normal((2, 3))
    np.testing.assert_allclose(
        mt.vecmat(mt.from_numpy(rows), mt.from_numpy(matrices)).numpy(),
        np.vecmat(rows, matrices),
    )
    with pytest.raises(ValueError, match="matrix and a vector"):
        mt.matvec(mt.from_numpy(vectors[0]), mt.from_numpy(vectors[0]))


def test_frexp_and_ldexp_are_inverses():
    values = np.array(
        [0.0, -0.0, 1.0, 2.0, 0.5, -3.75, 1e300, -1e-300, np.inf, -np.inf, np.nan, 1024.0]
    )
    mantissa, exponent = mt.frexp(mt.from_numpy(values))
    expected_mantissa, expected_exponent = np.frexp(values)
    np.testing.assert_array_equal(mantissa.numpy(), expected_mantissa)
    np.testing.assert_array_equal(exponent.numpy(), expected_exponent)
    # Exactly, because the only arithmetic is by powers of two.
    np.testing.assert_array_equal(mt.ldexp(mantissa, exponent).numpy(), values)


def test_cbrt_takes_the_real_root_of_a_negative():
    values = np.array([-27.0, -1.0, -0.125, 0.0, 0.125, 1.0, 8.0, 27.0])
    np.testing.assert_allclose(
        mt.cbrt(mt.from_numpy(values)).numpy(), np.cbrt(values), atol=1e-12
    )
    # The obvious spelling is not this: a fractional power of a negative is NaN.
    with np.errstate(invalid="ignore"):
        assert np.isnan(np.float64(-27.0) ** (1.0 / 3.0))


def test_divmod_agrees_with_its_two_halves():
    left = np.array([7.0, -7.0, 7.0, -7.0])
    right = np.array([3.0, 3.0, -3.0, -3.0])
    quotient, remainder = mt.divmod(mt.from_numpy(left), mt.from_numpy(right))
    expected_quotient, expected_remainder = np.divmod(left, right)
    np.testing.assert_allclose(quotient.numpy(), expected_quotient)
    np.testing.assert_allclose(remainder.numpy(), expected_remainder)
    # The identity the two are defined by.
    np.testing.assert_allclose(
        (quotient * mt.from_numpy(right) + remainder).numpy(), left
    )


def test_positive_is_a_copy():
    values = np.array([1.0, -2.0, 0.0])
    np.testing.assert_array_equal(mt.positive(mt.from_numpy(values)).numpy(), values)
