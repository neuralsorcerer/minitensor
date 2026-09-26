# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`nanmean`, `nanvar` and `nanstd` answer the same whether or not they are
being trained through.

Without gradients to record, one axis or all of them go through a fused kernel
-- three reads of the input -- instead of the composition of a dozen full-size
ops that defines them and carries their gradients. `nanvar` of a
`(16, 100000, 2)` tensor over its middle axis went from 14.6ms to about 0.5.
Two implementations of one operation must not disagree, so these compare them
on the inputs where they could: NaN anywhere, infinities, all-NaN slices,
slices of one, empty axes, and either correction.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import minitensor as mt

SHAPES = [(7,), (1,), (0,), (3, 0), (3, 5), (5, 3), (2, 3, 4), (1, 9000), (9000, 1)]
SHAPES += [(3, 20000), (20000, 2), (6, 2, 5000), (2, 700, 3)]


def _holed(shape, dtype, case, rng):
    values = rng.standard_normal(shape).astype(dtype)
    flat = values.reshape(-1)
    if flat.size == 0:
        return values
    if case in ("nan", "inf"):
        flat[rng.integers(0, flat.size, max(1, flat.size // 7))] = np.nan
    if case == "inf":
        flat[rng.integers(0, flat.size, 2)] = np.inf
    if case == "all-nan":
        flat[:] = np.nan
    if case == "nan-row" and values.ndim > 1:
        values[0] = np.nan
    return values


def _calls(dim, keepdim):
    kw = {"keepdim": keepdim}
    args = () if dim is None else (dim,)
    return {
        "nanmean": lambda t: t.nanmean(*args, **kw),
        "nanvar": lambda t: t.nanvar(*args, **kw),
        "nanvar-biased": lambda t: t.nanvar(*args, unbiased=False, **kw),
        "nanstd": lambda t: t.nanstd(*args, **kw),
    }


def _same(got, want, rtol):
    assert got.shape == want.shape
    for kind in (np.isnan, np.isposinf, np.isneginf):
        np.testing.assert_array_equal(kind(got), kind(want))
    finite = np.isfinite(want)
    np.testing.assert_allclose(got[finite], want[finite], rtol=rtol, atol=rtol * 1e-3)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("case", ["plain", "nan", "inf", "all-nan", "nan-row"])
@pytest.mark.parametrize("shape", SHAPES)
def test_the_fused_nan_statistics_match_the_composition(shape, case, dtype):
    rng = np.random.default_rng(sum(shape) + len(case))
    values = _holed(shape, dtype, case, rng)
    rtol = 2e-5 if dtype == "float32" else 1e-12
    plain = mt.from_numpy(np.ascontiguousarray(values))
    tracked = mt.Tensor(values, dtype=dtype, requires_grad=True)
    dims = [None, *range(len(shape)), -1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for dim in dims:
            for keepdim in (False, True):
                for name, call in _calls(dim, keepdim).items():
                    fused = call(plain).numpy()
                    composed = call(tracked).detach().numpy()
                    try:
                        _same(fused, composed, rtol)
                    except AssertionError as error:
                        raise AssertionError(
                            f"{name}(dim={dim}, keepdim={keepdim})"
                        ) from error


@pytest.mark.parametrize(
    "shape,dim",
    [((16, 100_000, 2), 1), ((4096, 1024), 0), ((1024, 4096), 1), ((8, 512, 512), 1)],
)
def test_nanvar_stays_within_a_few_ulps(shape, dim):
    rng = np.random.default_rng(len(shape) + dim)
    values = (rng.random(shape) + 0.5).astype(np.float32)
    values.reshape(-1)[::13] = np.nan
    exact = np.nanvar(values.astype(np.float64), axis=dim, ddof=1).astype(np.float32)
    got = mt.from_numpy(values).nanvar(dim).numpy()
    ulps = np.abs(got.view(np.int32).astype(np.int64) - exact.view(np.int32)).max()
    assert ulps <= 8
