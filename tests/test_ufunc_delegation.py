# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The element-wise functions NumPy computes faster cross to it, unchanged.

In float64 the engine has only scalar loops for `tanh`, `log1p`, `log10`,
`acosh` and the rest of the list `_core.dispatch.delegated_ufuncs()` reports,
where NumPy runs vectorized loops that are faster and, against exact values,
at least as accurate; so above a size threshold they are handed over,
zero-copy, with a large input split across threads. What has to hold is that
the route makes no difference anyone could observe: the same values as NumPy,
the same NaN, infinities and signed zeros as the engine's own loop, no
floating-point warnings the engine never gave, and the same gradients. Where
NumPy's loop is faster but less accurate, the function is not handed over.
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

_OURS = {"arcsinh": "asinh", "arctanh": "atanh", "arccosh": "acosh"}
_BINARY = {"remainder", "fmod"}
# Exact operations: whoever computes them owes the same bits.
_EXACT = {"remainder", "fmod"}
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
    assert len(DELEGATED) == 12
    assert all(hasattr(np, name) for name in DELEGATED)


# NumPy's loops for these were faster and less accurate than the engine's own:
# on an AVX-512 machine they missed the nearest float at up to a tenth of
# inputs, and by up to 0.78 ulps, where the engine's miss at under 0.2%.
_KEPT = {
    "exp": "exp",
    "exp2": "exp2",
    "log": "log",
    "cbrt": "cbrt",
    "tan": "tan",
    "asin": "arcsin",
    "acos": "arccos",
    "atan": "arctan",
}


@pytest.mark.parametrize("ours", sorted(_KEPT))
def test_a_function_numpy_computes_less_accurately_stays_with_the_engine(ours):
    assert _KEPT[ours] not in DELEGATED
    with np.errstate(all="ignore"):
        values = np.abs(_values("float64", seed=4)) % 1.0
    t = mt.from_numpy(values)
    op = getattr(mt, ours)
    delegated = _through(0, lambda: op(t).numpy())
    native = _through(2**62, lambda: op(t).numpy())
    np.testing.assert_array_equal(delegated, native)


def test_atan2_and_pow_stay_with_the_engine():
    assert "arctan2" not in DELEGATED and "power" not in DELEGATED
    rng = np.random.default_rng(9)
    a = mt.from_numpy(rng.random(2000) + 0.5)
    b = mt.from_numpy(rng.random(2000) * 4.0 - 2.0)
    for op in (mt.atan2, mt.pow):
        delegated = _through(0, lambda: op(a, b).numpy())
        native = _through(2**62, lambda: op(a, b).numpy())
        np.testing.assert_array_equal(delegated, native)


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
    if name in _EXACT:
        np.testing.assert_array_equal(native, reference)
        np.testing.assert_array_equal(np.signbit(native), np.signbit(reference))


@requires_delegation
@pytest.mark.parametrize("name", ["tanh", "log1p"])
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
    """`x % 2.5`, `2.5 % x` and `fmod(x, 2.5)` hand one side over as a single
    element; split across threads, each chunk has to see that element rather
    than a slice of it."""
    x = np.linspace(0.1, 5.0, n)
    t = mt.from_numpy(x)
    half = mt.from_numpy(np.array([2.5]))
    np.testing.assert_array_equal(
        _through(0, lambda: mt.remainder(t, half).numpy()), np.remainder(x, 2.5)
    )
    np.testing.assert_array_equal(
        _through(0, lambda: mt.remainder(half, t).numpy()), np.remainder(2.5, x)
    )
    np.testing.assert_array_equal(
        _through(0, lambda: mt.fmod(t, half).numpy()), np.fmod(x, 2.5)
    )


@requires_delegation
def test_a_general_broadcast_is_not_offered_but_still_right():
    rows = np.linspace(0.5, 2.0, 700).reshape(700, 1)
    cols = np.linspace(-1.0, 3.0, 900).reshape(1, 900)
    got = _through(0, lambda: mt.pow(mt.from_numpy(rows), mt.from_numpy(cols)).numpy())
    np.testing.assert_allclose(got, rows**cols, rtol=4e-16, atol=0)


@requires_delegation
@pytest.mark.parametrize("name", ["tanh", "log1p", "expm1", "log10", "asinh", "atanh"])
def test_gradients_through_a_delegated_function(name):
    rng = np.random.default_rng(7)
    x = mt.Tensor(rng.random(600) * 0.8 + 0.1, dtype="float64", requires_grad=True)
    op = getattr(mt, name)
    assert _through(0, lambda: mt.gradcheck(lambda v: op(v).sum(), (x,)))


@requires_delegation
@pytest.mark.parametrize("name", ["remainder", "fmod"])
def test_gradients_through_a_delegated_binary_function(name):
    rng = np.random.default_rng(8)
    # Below one divisor, so no quotient's floor moves under the differences.
    x = mt.Tensor(rng.random(600) * 2.0 + 0.2, dtype="float64", requires_grad=True)
    y = mt.Tensor(rng.random(600) * 0.2 + 2.3, dtype="float64", requires_grad=True)
    op = getattr(mt, name)
    assert _through(0, lambda: mt.gradcheck(lambda a, b: op(a, b).sum(), (x, y)))


def test_the_threshold_moves_and_comes_back():
    before = dispatch.ufunc_threshold()
    assert before == dispatch.DEFAULT_MIN_UFUNC_LEN
    assert dispatch.set_ufunc_threshold(3) == before
    assert dispatch.ufunc_threshold() == 3
    dispatch.set_ufunc_threshold(before)
    assert dispatch.ufunc_threshold() == before
