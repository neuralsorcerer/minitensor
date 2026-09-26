# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The indexed extremum along the last axis, now two vectorized passes a row.

`argmax`, `argmin`, their NaN-skipping forms and `max`/`min(dim)` along a
contiguous row find the extremum by a lane fold and then search the row for
its first occurrence; a few very long rows are cut into bands whose winners
are compared in order. Both have to answer exactly what the element-by-element
walk did: the lowest index on a tie, the first NaN when NaN propagates, the
first real extremum when it is skipped, and the element's own sign of zero.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

# Many short rows, rows long enough to band, and one row spanning many bands.
SHAPES = [(64, 33), (128, 4096), (3, 40_000), (1, 100_000), (2, 70_001)]


def _reference(a, is_max, skip_nan):
    """The walk: first index whose value no later value beats."""
    rows = a.reshape(-1, a.shape[-1])
    values = np.empty(rows.shape[0], a.dtype)
    indices = np.empty(rows.shape[0], np.int64)
    for r, row in enumerate(rows):
        nan = np.isnan(row)
        if not skip_nan and nan.any():
            at = int(np.argmax(nan))
        elif skip_nan and nan.all():
            values[r], indices[r] = np.nan, 0
            continue
        else:
            real = np.where(nan, -np.inf if is_max else np.inf, row)
            best = real.max() if is_max else real.min()
            at = int(np.argmax((row == best) & ~nan))
        values[r], indices[r] = row[at], at
    return values.reshape(a.shape[:-1]), indices.reshape(a.shape[:-1])


def _inputs(shape, dtype, rng):
    plain = rng.standard_normal(shape).astype(dtype)
    zeros = np.where(plain > 0, 0.0, plain).astype(dtype)
    zeros.reshape(-1)[::5] = -0.0
    holed = plain.copy()
    holed.reshape(-1)[::9] = np.nan
    edge = np.full(shape, -np.inf, dtype)
    edge.reshape(-1)[::3] = np.nan
    tied = np.round(plain)
    late = np.full(shape, np.nan, dtype)
    late[..., -1] = 1.0
    return {
        "plain": plain,
        "zeros": zeros,
        "nan": holed,
        "edge": edge,
        "tied": tied,
        "late": late,
    }


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("shape", SHAPES)
def test_the_row_path_answers_as_the_walk_did(shape, dtype):
    rng = np.random.default_rng(sum(shape))
    for case, a in _inputs(shape, dtype, rng).items():
        t = mt.from_numpy(np.ascontiguousarray(a))
        d = a.ndim - 1
        for is_max in (True, False):
            want_v, want_i = _reference(a, is_max, skip_nan=False)
            v, i = t.max(d) if is_max else t.min(d)
            v, i = v.numpy(), i.numpy()
            assert np.array_equal(i, want_i), (case, is_max)
            assert np.array_equal(v, want_v, equal_nan=True), (case, is_max)
            assert np.array_equal(np.signbit(v), np.signbit(want_v)), (case, is_max)
            arg = t.argmax(d) if is_max else t.argmin(d)
            assert np.array_equal(arg.numpy(), want_i), (case, is_max)

            _, want_nan_i = _reference(a, is_max, skip_nan=True)
            if np.isnan(a).all(axis=-1).any():
                continue
            nan_arg = t.nanargmax(d) if is_max else t.nanargmin(d)
            assert np.array_equal(nan_arg.numpy(), want_nan_i), (case, is_max)


# Along an axis that is not the last: narrow and wide slabs, one slab and many,
# tall and short, so each way `slab_arg_extremum` cuts its work is exercised.
SLAB_CASES = [
    ((1000, 2), 0),
    ((70_000, 2), 0),
    ((20_000, 33), 0),
    ((4096, 1024), 0),
    ((3, 4096), 0),
    ((16, 3000, 3), 1),
    ((2, 500, 300), 1),
    ((8, 512, 512), 1),
]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("shape,dim", SLAB_CASES)
def test_the_slab_path_answers_as_the_walk_did(shape, dim, dtype):
    """`max(dim)` and the arg reductions down a non-last axis take each block
    of rows' extremum in one vectorized pass and then search only the block
    that holds a column's first match; they must still name the first match
    of the whole column, and its value."""
    rng = np.random.default_rng(sum(shape) + dim)
    for case, a in _inputs(shape, dtype, rng).items():
        t = mt.from_numpy(np.ascontiguousarray(a))
        moved = np.moveaxis(a, dim, -1)
        for is_max in (True, False):
            want_v, want_i = _reference(moved, is_max, skip_nan=False)
            v, i = t.max(dim) if is_max else t.min(dim)
            v, i = v.numpy(), i.numpy()
            assert np.array_equal(i, want_i), (case, is_max)
            assert np.array_equal(v, want_v, equal_nan=True), (case, is_max)
            assert np.array_equal(np.signbit(v), np.signbit(want_v)), (case, is_max)
            arg = t.argmax(dim) if is_max else t.argmin(dim)
            assert np.array_equal(arg.numpy(), want_i), (case, is_max)
            np.testing.assert_array_equal(
                (t.amax(dim) if is_max else t.amin(dim)).numpy(),
                moved.max(axis=-1) if is_max else moved.min(axis=-1),
            )

            _, want_nan_i = _reference(moved, is_max, skip_nan=True)
            if np.isnan(moved).all(axis=-1).any():
                continue
            nan_arg = t.nanargmax(dim) if is_max else t.nanargmin(dim)
            assert np.array_equal(nan_arg.numpy(), want_nan_i), (case, is_max)
