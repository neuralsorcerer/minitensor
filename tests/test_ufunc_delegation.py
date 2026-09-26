# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The element-wise functions NumPy computes faster cross to it, unchanged.

In float64 the engine has only scalar `libm` for `tanh`, `log1p`, `cbrt`, `pow`
and the rest of the list `_core.dispatch.delegated_ufuncs()` reports, where
NumPy runs vectorized loops that are faster and as accurate; so above a size
threshold they are handed over, zero-copy, with a large input split across
threads. What has to hold is that the route
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

_OURS = {
    "arcsinh": "asinh",
    "arctanh": "atanh",
    "arcsin": "asin",
    "arccos": "acos",
    "arctan": "atan",
    "arccosh": "acosh",
    "arctan2": "atan2",
    "power": "pow",
}
_BINARY = {"arctan2", "power"}
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
    # Float64 only, and never float32: NumPy was faster at several float32
    # functions too, but those are correctly rounded here and not there, and
    # speed that costs the answer is not what delegation is for.
    if not dispatch.PROVIDER_EXPECTED:
        assert DELEGATED == []
        return
    assert len(DELEGATED) == 20
    assert all(hasattr(np, name) for name in DELEGATED)


def _operands(name):
    """What `name` is called on: one array, or two for the binary ones, whose
    second operand runs through its own specials against the first's."""
    values = _values("float64")
    if name not in _BINARY:
        return (values,)
    return values, np.roll(_values("float64", seed=1), 5)


@requires_delegation
@pytest.mark.parametrize("name", DELEGATED)
@pytest.mark.filterwarnings("error")
def test_delegated_values_are_numpys_and_agree_with_the_engine(name):
    operands = _operands(name)
    op = getattr(mt, _OURS.get(name, name))
    tensors = [mt.from_numpy(values) for values in operands]
    delegated = _through(0, lambda: op(*tensors).numpy())
    native = _through(2**62, lambda: op(*tensors).numpy())
    with np.errstate(all="ignore"):
        reference = getattr(np, name)(*operands)

    np.testing.assert_array_equal(delegated, reference)
    np.testing.assert_array_equal(np.signbit(delegated), np.signbit(reference))
    for kind in (np.isnan, np.isposinf, np.isneginf):
        np.testing.assert_array_equal(kind(native), kind(reference))
    assert _ulps(native, reference) <= 4


@requires_delegation
@pytest.mark.parametrize("name", ["tanh", "log"])
def test_a_large_input_split_across_threads_gives_the_same_answer(name):
    values = np.abs(_values("float64", n=400_000, seed=3)) + 0.25
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
@pytest.mark.parametrize("n", [3, 400_000])
def test_a_single_element_operand_is_broadcast_in_every_chunk(n):
    """`x ** 2.5`, `2.5 ** x` and `atan2(y, 1)` hand one side over as a single
    element; split across threads, each chunk has to see that element rather
    than a slice of it."""
    x = np.linspace(0.1, 5.0, n)
    t = mt.from_numpy(x)
    half = mt.from_numpy(np.array([2.5]))
    np.testing.assert_array_equal(_through(0, lambda: (t**2.5).numpy()), x**2.5)
    np.testing.assert_array_equal(_through(0, lambda: mt.pow(half, t).numpy()), 2.5**x)
    np.testing.assert_array_equal(
        _through(0, lambda: mt.atan2(t, half).numpy()), np.arctan2(x, 2.5)
    )


@requires_delegation
def test_a_general_broadcast_is_not_offered_but_still_right():
    rows = np.linspace(0.5, 2.0, 700).reshape(700, 1)
    cols = np.linspace(-1.0, 3.0, 900).reshape(1, 900)
    got = _through(0, lambda: mt.pow(mt.from_numpy(rows), mt.from_numpy(cols)).numpy())
    np.testing.assert_allclose(got, rows**cols, rtol=4e-16, atol=0)


@requires_delegation
@pytest.mark.parametrize("name", ["tanh", "log1p", "cbrt", "expm1", "asin", "atan"])
def test_gradients_through_a_delegated_function(name):
    rng = np.random.default_rng(7)
    x = mt.Tensor(rng.random(600) * 0.8 + 0.1, dtype="float64", requires_grad=True)
    op = getattr(mt, name)
    assert _through(0, lambda: mt.gradcheck(lambda v: op(v).sum(), (x,)))


@requires_delegation
@pytest.mark.parametrize("name", ["pow", "atan2"])
def test_gradients_through_a_delegated_binary_function(name):
    rng = np.random.default_rng(8)
    x = mt.Tensor(rng.random(600) + 0.5, dtype="float64", requires_grad=True)
    y = mt.Tensor(rng.random(600) + 0.5, dtype="float64", requires_grad=True)
    op = getattr(mt, name)
    assert _through(0, lambda: mt.gradcheck(lambda a, b: op(a, b).sum(), (x, y)))


def test_the_threshold_moves_and_comes_back():
    before = dispatch.ufunc_threshold()
    assert before == dispatch.DEFAULT_MIN_UFUNC_LEN
    assert dispatch.set_ufunc_threshold(3) == before
    assert dispatch.ufunc_threshold() == 3
    dispatch.set_ufunc_threshold(before)
    assert dispatch.ufunc_threshold() == before
