# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A reduction along the first axis has one output row, and one row was one core.

These kernels cut their output into one piece per outer position, which is the
right cut until there is a single outer position -- and a reduction along the
first axis has exactly one however large the tensor is. `median(dim=0)` of a
512x4096 tensor took 53ms while `quantile(0.5, dim=0)` took 4.5 computing the
same thing by the same selection: the difference was entirely that one of them
ran on one core.

The columns of that one position are contiguous in the output even though they
are a stride apart in the input, so they cut apart cleanly. That is what
`reduction_band` decides, and it is deliberately narrow: with several outer
positions there is already work to share out, and a band that does not divide
the row exactly would then span the end of one position and the start of the
next -- which is the bug these shapes are chosen to catch. Awkward widths, one
column, one row, and sizes that no band count divides evenly.

Each column still accumulates its own steps in its own order, so the cut cannot
move an answer; `test_reduction_determinism.py` runs these at several thread
counts to keep it that way.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

SHAPES = [
    (4096, 512),
    (512, 4096),
    (64, 64, 512),
    (513, 997),
    (37, 4096),
    (3, 400),
    (2, 3, 400),
    (7, 13),
    (1, 9),
    (9, 1),
    (2, 3, 5, 7),
]


def _data(shape, seed):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) * 3).astype(np.float32)


@pytest.mark.parametrize("shape", SHAPES)
def test_variance_and_deviation_match_numpy_along_every_axis(shape):
    values = _data(shape, 41)
    wide = values.astype(np.float64)
    t = mt.from_numpy(values)
    for dim in range(len(shape)):
        if shape[dim] == 1:
            # One sample has no unbiased variance; NumPy warns rather than
            # answering, so the agreement is checked separately below.
            assert np.isnan(mt.var(t, dim).numpy()).all()
            assert np.isnan(mt.std(t, dim).numpy()).all()
            continue
        # The library's default is the unbiased estimator, NumPy's is not.
        np.testing.assert_allclose(
            mt.var(t, dim).numpy(), np.var(wide, dim, ddof=1), rtol=3e-4, atol=3e-5
        )
        np.testing.assert_allclose(
            mt.std(t, dim).numpy(), np.std(wide, dim, ddof=1), rtol=3e-4, atol=3e-5
        )


@pytest.mark.parametrize("shape", SHAPES)
def test_the_product_matches_numpy_along_every_axis(shape):
    # Scaled to stay inside float32 over a 4096-long axis.
    values = (_data(shape, 43) * 0.01 + 1.0).astype(np.float32)
    t = mt.from_numpy(values)
    for dim in range(len(shape)):
        np.testing.assert_allclose(
            mt.prod(t, dim).numpy(),
            np.prod(values.astype(np.float64), dim),
            rtol=2e-4,
        )


@pytest.mark.parametrize("shape", SHAPES)
def test_the_median_and_its_index_agree_with_a_sort(shape):
    values = _data(shape, 47)
    t = mt.from_numpy(values)
    for dim in range(len(shape)):
        got, where = mt.median(t, dim)
        # An even count takes the lower of the two middles, which is the
        # `(n - 1) // 2`th element of the sorted slice.
        expected = np.sort(values, dim).take((shape[dim] - 1) // 2, axis=dim)
        np.testing.assert_array_equal(got.numpy(), expected)

        # The index has to point at the value that was returned.
        picked = np.take_along_axis(
            values, np.expand_dims(where.numpy(), dim), dim
        ).squeeze(dim)
        np.testing.assert_array_equal(picked, got.numpy())


def test_a_median_slice_holding_a_nan_is_nan():
    values = _data((5, 600), 53)
    values[2, ::7] = np.nan
    got, _ = mt.median(mt.from_numpy(values), 0)
    assert np.isnan(got.numpy()[::7]).all()
    np.testing.assert_array_equal(
        got.numpy()[1::7], np.sort(values, 0)[2][1::7]
    )


def test_the_banded_and_unbanded_cuts_agree():
    """The same values reduced along the first axis of a tall tensor and along
    the middle axis of the same values with a leading 1 take different cuts --
    one banded, one not -- and must not disagree."""
    values = _data((512, 997), 59)
    flat = mt.from_numpy(values)
    nested = mt.from_numpy(values.reshape(1, 512, 997))

    for op in (mt.var, mt.std, mt.prod):
        np.testing.assert_array_equal(
            op(flat, 0).numpy(), op(nested, 1).numpy().reshape(-1)
        )
    np.testing.assert_array_equal(
        mt.median(flat, 0)[0].numpy(), mt.median(nested, 1)[0].numpy().reshape(-1)
    )
