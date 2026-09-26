# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The element-wise functions NumPy computes faster cross to it, unchanged.

In float64 the engine has only scalar `libm` for `tanh`, `log1p`, `cbrt` and the
rest of the list `_core.dispatch.delegated_ufuncs()` reports, where NumPy runs
vectorized loops that are faster and at least as accurate; so above a size threshold they are handed over, zero-copy,
with a large input split across threads. What has to hold is that the route
makes no difference anyone could observe: the same values as NumPy, the same
NaN, infinities and signed zeros as the engine's own loop, no floating-point
warnings the engine never gave, and the same gradients.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

import minitensor as mt

dispatch = mt._core.dispatch

requires_delegation = pytest.mark.skipif(
    not dispatch.PROVIDER_EXPECTED,
    reason="built with --features blas: nothing is delegated",
)

_NUMPY = {"arcsinh": np.arcsinh, "arctanh": np.arctanh}
_OURS = {"arcsinh": "asinh", "arctanh": "atanh"}
DELEGATED = dispatch.delegated_ufuncs()


def _values(dtype, n=6000, seed=0):
    rng = np.random.default_rng(seed)
    specials = [0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, np.nan, 1e308, -1e308]
    specials += [5e-324, 700.0, -700.0, 1e-300, 0.5, -0.5, 2.0]
    with np.errstate(all="ignore"):
        return np.concatenate(
            [rng.standard_normal(n // 2) * 3, rng.random(n // 2), specials]
        ).astype(dtype)


def _through(threshold, call):
    previous = dispatch.set_ufunc_threshold(threshold)
    try:
        return call()
    finally:
        dispatch.set_ufunc_threshold(previous)


def _ulps(a, b):
    kind = np.int32 if a.dtype == np.float32 else np.int64
    finite = np.isfinite(a) & np.isfinite(b)
    if not finite.any():
        return 0
    diff = a[finite].view(kind).astype(np.int64) - b[finite].view(kind).astype(np.int64)
    return int(np.abs(diff).max())


def test_the_delegated_list_is_what_the_measurements_chose():
    # Float64 only. NumPy was also faster at float32 `cbrt` and `atanh`, but
    # those are correctly rounded here and not there, and speed that costs the
    # answer is not what delegation is for.
    if not dispatch.PROVIDER_EXPECTED:
        assert DELEGATED == []
        return
    assert {dtype for _, dtype in DELEGATED} == {"float64"}
    assert len(DELEGATED) == 14


@requires_delegation
@pytest.mark.parametrize("name,dtype", DELEGATED)
@pytest.mark.filterwarnings("error")
def test_delegated_values_are_numpys_and_agree_with_the_engine(name, dtype):
    values = _values(dtype)
    op = getattr(mt, _OURS.get(name, name))
    t = mt.from_numpy(values)
    delegated = _through(0, lambda: op(t).numpy())
    native = _through(2**62, lambda: op(t).numpy())
    with np.errstate(all="ignore"):
        reference = (_NUMPY.get(name) or getattr(np, name))(values)

    np.testing.assert_array_equal(delegated, reference)
    np.testing.assert_array_equal(np.signbit(delegated), np.signbit(reference))
    for kind in (np.isnan, np.isposinf, np.isneginf):
        np.testing.assert_array_equal(kind(native), kind(reference))
    assert _ulps(native, reference) <= 4


@requires_delegation
@pytest.mark.parametrize("name,dtype", [("tanh", "float64"), ("log", "float64")])
def test_a_large_input_split_across_threads_gives_the_same_answer(name, dtype):
    values = np.abs(_values(dtype, n=400_000, seed=3)) + 0.25
    values[~np.isfinite(values)] = 1.0
    op = getattr(mt, name)
    got = _through(0, lambda: op(mt.from_numpy(values)).numpy())
    np.testing.assert_array_equal(got, getattr(np, name)(values))


@requires_delegation
def test_concurrent_callers_each_get_their_own_answer():
    inputs = [np.linspace(-3, 3, 300_000) * (k + 1) / 4 for k in range(4)]
    results = [None] * 4

    def work(k):
        results[k] = mt.tanh(mt.from_numpy(inputs[k])).numpy()

    threads = [threading.Thread(target=work, args=(k,)) for k in range(4)]
    _through(0, lambda: [t.start() for t in threads] and [t.join() for t in threads])
    for k in range(4):
        np.testing.assert_array_equal(results[k], np.tanh(inputs[k]))


@requires_delegation
@pytest.mark.parametrize("name", ["tanh", "log1p", "cbrt", "expm1"])
def test_gradients_through_a_delegated_function(name):
    rng = np.random.default_rng(7)
    x = mt.Tensor(rng.random(600) + 0.1, dtype="float64", requires_grad=True)
    op = getattr(mt, name)
    assert _through(0, lambda: mt.gradcheck(lambda v: op(v).sum(), (x,)))


def test_the_threshold_moves_and_comes_back():
    before = dispatch.ufunc_threshold()
    assert before == dispatch.DEFAULT_MIN_UFUNC_LEN
    assert dispatch.set_ufunc_threshold(3) == before
    assert dispatch.ufunc_threshold() == 3
    dispatch.set_ufunc_threshold(before)
    assert dispatch.ufunc_threshold() == before
