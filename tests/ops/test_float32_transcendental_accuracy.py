# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""float32 `tanh`, `sinh`, `expm1` and `log1p` are computed through float64.

glibc's `tanhf`, `sinhf`, `expm1f` and `log1pf` carry roughly twice the error of
rounding a correctly-computed `f64` result once, so these four kernels promote.
Half an ulp from an `f64` round is an argument about rounding rather than about
any particular libm, so the bound below should hold wherever this runs.

The rest of the f32 math surface deliberately does *not* promote: `expf`,
`logf`, `sinf`, `cosf` and `cbrtf` are 1.7x-2.9x faster than promoting at equal
accuracy, so the tolerances here are set only for the four that changed.
"""

import numpy as np
import pytest

import minitensor as mt

# One f32 ulp near 1.0 is about 1.19e-07; half an ulp is ~6e-08. Allowing a
# little room above that keeps this from being a rounding-mode tripwire.
_HALF_ULP_BUDGET = 7.5e-8

_SAMPLE = (np.random.default_rng(1).standard_normal(200_000) * 3).astype(np.float32)


def _worst_relative_error(actual, reference):
    actual = np.asarray(actual, dtype=np.float64)
    significant = np.abs(reference) > 1e-30
    return float(
        np.max(
            np.abs(actual[significant] - reference[significant])
            / np.abs(reference[significant])
        )
    )


PROMOTED = [
    ("tanh", lambda t: t.tanh(), np.tanh, _SAMPLE),
    ("sinh", lambda t: t.sinh(), np.sinh, _SAMPLE),
    ("expm1", lambda t: t.expm1(), np.expm1, _SAMPLE),
    ("log1p", lambda t: t.log1p(), np.log1p, np.abs(_SAMPLE)),
]


@pytest.mark.parametrize("name,op,reference,sample", PROMOTED, ids=[c[0] for c in PROMOTED])
def test_stays_within_half_an_ulp_of_the_float64_result(name, op, reference, sample):
    tensor = mt.from_numpy(sample)
    exact = reference(sample.astype(np.float64))
    error = _worst_relative_error(op(tensor).numpy(), exact)
    assert error < _HALF_ULP_BUDGET, f"{name}: worst relative error {error:.3e}"


@pytest.mark.parametrize("name,op,reference,sample", PROMOTED, ids=[c[0] for c in PROMOTED])
def test_at_least_as_accurate_as_numpy(name, op, reference, sample):
    # NumPy uses vectorised f32 kernels here, which are faster and less precise.
    # This is the comparison that motivated the change, so it is the one pinned.
    tensor = mt.from_numpy(sample)
    exact = reference(sample.astype(np.float64))
    ours = _worst_relative_error(op(tensor).numpy(), exact)
    theirs = _worst_relative_error(reference(sample), exact)
    assert ours <= theirs, f"{name}: {ours:.3e} vs numpy {theirs:.3e}"


@pytest.mark.parametrize(
    "value",
    [0.0, -0.0, 1e-30, -1e-30, 1.0, -1.0, 88.0, -88.0, np.inf, -np.inf, np.nan],
)
def test_promotion_preserves_the_edge_cases(value):
    """Promotion must not move an overflow boundary or lose a zero's sign.

    `f64` has more headroom, so anything that saturates in `f32` has to still
    saturate after rounding back down. The expected value is the correctly
    rounded one rather than NumPy's: at `expm1(88)` NumPy is a full 1.3e-07
    off while this returns the nearest representable `f32`, so asserting
    equality with NumPy would fail on the improvement itself.
    """
    sample = np.array([value], dtype=np.float32)
    tensor = mt.from_numpy(sample)
    for name, op, reference in (
        ("tanh", lambda t: t.tanh(), np.tanh),
        ("sinh", lambda t: t.sinh(), np.sinh),
        ("expm1", lambda t: t.expm1(), np.expm1),
    ):
        got = op(tensor).numpy()[0]
        want = np.float32(reference(np.float64(value)))
        if np.isnan(want):
            assert np.isnan(got), f"{name}({value})"
            continue
        assert got == want, f"{name}({value}): {got} != correctly rounded {want}"
        if want == 0.0:
            assert np.signbit(got) == np.signbit(want), f"{name}({value}) sign"


def test_the_unpromoted_functions_are_still_accurate():
    # Guards the other half of the decision: these were left on the f32 libm
    # path for speed, so their accuracy needs to stay within one ulp.
    sample = np.abs(_SAMPLE) + 0.5
    tensor = mt.from_numpy(sample)
    for name, op, reference in (
        ("exp", lambda t: t.exp(), np.exp),
        ("log", lambda t: t.log(), np.log),
        ("sin", lambda t: t.sin(), np.sin),
        ("cos", lambda t: t.cos(), np.cos),
    ):
        exact = reference(sample.astype(np.float64))
        error = _worst_relative_error(op(tensor).numpy(), exact)
        assert error < 1.5e-7, f"{name}: {error:.3e}"


# `tanh` and `sigmoid` backward were the only two gradient kernels built by
# chaining public tensor ops -- `Tensor::ones`, then `sub`, then `mul`, then
# `mul` -- which allocated and traversed a full-size tensor per link. Fusing
# each into a single pass cut them 4.7x and 4.2x (9.49ms -> 2.03ms and 9.13ms
# -> 2.17ms on a 2048x1024 f32 tensor), bringing them in line with `relu`'s
# 2.1ms. The values must not move, including where the derivative underflows.
_SATURATING = np.concatenate(
    [
        np.random.default_rng(0).standard_normal(2000) * 3,
        [0.0, -0.0, 1e-30, 20.0, -20.0, 50.0, -50.0, np.inf, -np.inf, np.nan],
    ]
)


@pytest.mark.parametrize("dtype,tolerance", [("float32", 1e-6), ("float64", 1e-12)])
@pytest.mark.parametrize("name", ["tanh", "sigmoid"])
def test_fused_backward_matches_the_analytic_derivative(name, dtype, tolerance):
    sample = _SATURATING.astype(dtype)
    tensor = mt.Tensor(sample, dtype=dtype, requires_grad=True)
    getattr(tensor, name)().sum().backward()

    exact = sample.astype(np.float64)
    if name == "tanh":
        expected = 1.0 - np.tanh(exact) ** 2
    else:
        sigmoid = 1.0 / (1.0 + np.exp(-exact))
        expected = sigmoid * (1.0 - sigmoid)

    np.testing.assert_allclose(
        tensor.grad.numpy(), expected.astype(dtype), atol=tolerance, equal_nan=True
    )
    mt.clear_autograd_graph()


@pytest.mark.parametrize("name", ["tanh", "sigmoid"])
def test_fused_backward_handles_saturation_without_producing_nan(name):
    # Far out in the tail the derivative underflows to zero; it must not become
    # NaN through an intermediate that overflowed first.
    sample = np.array([-100.0, -50.0, 50.0, 100.0], dtype=np.float32)
    tensor = mt.Tensor(sample, dtype="float32", requires_grad=True)
    getattr(tensor, name)().sum().backward()

    gradient = tensor.grad.numpy()
    assert np.all(np.isfinite(gradient)), gradient
    np.testing.assert_allclose(gradient, np.zeros(4), atol=1e-20)
    mt.clear_autograd_graph()


# float32 `tanh` no longer promotes element by element -- `ops::simd::
# transcendental` computes the same f64-then-round value with a vectorized,
# runtime-dispatched kernel (AVX-512, AVX2+FMA, or portable), which measured
# 10.4x faster single-threaded and took the op from 11.8x slower than NumPy at
# a million elements to roughly parity.
#
# Its contract is stronger than a tolerance and is what these pin: *identical
# bits* to the scalar routine it replaced. The failure modes a tolerance would
# miss are structural -- a tail block past the last full vector handled
# differently from the body, or the sequential and parallel sides of the
# threshold disagreeing -- so the lengths below straddle the vector widths (4,
# 8, 16), the rayon block size (1024) and the parallel threshold (16384).
_TANH_LENGTHS = [1, 3, 7, 8, 15, 17, 1023, 1024, 1025, 16383, 16384, 16385, 40000]


def _tanh_reference(sample):
    """What the previous scalar kernel returned: f64 `tanh`, rounded once."""
    return np.tanh(sample.astype(np.float64)).astype(np.float32)


# `expm1`, `sinh` and `cosh` are vectorized on the same `expm1` core and carry
# the same bit-identity contract. `expm1` and `sinh` replaced promoted scalars,
# so this preserves what they already returned; `cosh` replaced glibc's `coshf`,
# which misrounds 22,628,918 of the 2^32 float32 inputs, so there it is an
# accuracy gain too.
_EXP_FAMILY = [
    ("expm1", lambda t: t.expm1(), np.expm1),
    ("sinh", lambda t: t.sinh(), np.sinh),
    ("cosh", lambda t: t.cosh(), np.cosh),
]


@pytest.mark.parametrize("name,op,reference", _EXP_FAMILY, ids=[c[0] for c in _EXP_FAMILY])
@pytest.mark.parametrize("length", [1, 7, 8, 17, 1023, 1024, 16383, 16384, 40000])
def test_exp_family_is_bit_identical_to_the_float64_reference(name, op, reference, length):
    rng = np.random.default_rng(777 + length)
    # Across the whole useful range: near zero where `expm1` must not lose the
    # leading term, the mid range that steps through every `n`, and out past
    # where the float32 result overflows to infinity.
    sample = np.concatenate(
        [
            rng.standard_normal(length) * 5.0,
            rng.standard_normal(length) * 1e-5,
            rng.uniform(-95.0, 95.0, length),
        ]
    ).astype(np.float32)[:length]

    got = op(mt.from_numpy(sample)).numpy()
    with np.errstate(over="ignore"):  # |x| up to 95 overflows float32 for cosh/sinh
        want = reference(sample.astype(np.float64)).astype(np.float32)
    mismatched = got.view(np.uint32) != want.view(np.uint32)
    assert not mismatched.any(), (
        f"{name} length {length}: {int(mismatched.sum())} of {length} differ, "
        f"first at x={sample[mismatched][0]!r}"
    )


@pytest.mark.parametrize("name,op,reference", _EXP_FAMILY, ids=[c[0] for c in _EXP_FAMILY])
def test_exp_family_handles_the_saturating_ends(name, op, reference):
    """Overflow to infinity, and the underflow that `sinh` used to get wrong.

    Evaluating `sinh` as `u(u+2)/(2(u+1))` at negative `x` divides by `exp(x)`,
    which underflows to zero -- `sinh(-100)` came back as -inf until the kernel
    was made to work on `|x|` and restore the sign afterwards.
    """
    sample = np.array(
        [0.0, -0.0, 1e-30, -1e-30, 88.0, -88.0, 89.0, -89.0, 100.0, -100.0, 1e30, -1e30],
        dtype=np.float32,
    )
    got = op(mt.from_numpy(sample)).numpy()
    with np.errstate(over="ignore"):  # the reference overflows here on purpose
        want = reference(sample.astype(np.float64)).astype(np.float32)
    np.testing.assert_array_equal(got, want, err_msg=f"{name} at the saturating ends")
    assert np.all(np.isfinite(got) | np.isinf(want)), f"{name} produced a spurious infinity"


@pytest.mark.parametrize("length", _TANH_LENGTHS)
def test_vectorized_tanh_is_bit_identical_to_the_promoted_reference(length):
    rng = np.random.default_rng(20240607 + length)
    # Spread across every regime the kernel branches on: the near-zero range
    # where the polynomial carries the result, the mid range that exercises
    # each argument-reduction step, and past the clamp where it saturates.
    sample = np.concatenate(
        [
            rng.standard_normal(length) * 4.0,
            rng.standard_normal(length) * 1e-4,
            rng.uniform(8.5, 12.0, length) * rng.choice([-1.0, 1.0], length),
        ]
    ).astype(np.float32)[:length]

    got = mt.from_numpy(sample).tanh().numpy()
    want = _tanh_reference(sample)
    mismatched = got.view(np.uint32) != want.view(np.uint32)
    assert not mismatched.any(), (
        f"length {length}: {int(mismatched.sum())} of {length} differ, "
        f"first at x={sample[mismatched][0]!r}: "
        f"{got[mismatched][0]!r} != {want[mismatched][0]!r}"
    )


def test_vectorized_tanh_agrees_across_the_parallel_threshold():
    # The same values, once below the threshold and once above it, must come
    # back the same: block splitting must not be observable in the output.
    rng = np.random.default_rng(99)
    head = (rng.standard_normal(16_000) * 3).astype(np.float32)
    padded = np.concatenate([head, (rng.standard_normal(20_000) * 3).astype(np.float32)])

    sequential = mt.from_numpy(head).tanh().numpy()
    parallel = mt.from_numpy(padded).tanh().numpy()[: head.size]
    np.testing.assert_array_equal(sequential.view(np.uint32), parallel.view(np.uint32))


def test_vectorized_tanh_stays_odd():
    # Oddness is not automatic: the argument reduction rounds `n` separately
    # for `x` and `-x`, so the two sides could in principle disagree.
    sample = np.concatenate(
        [np.linspace(1e-6, 9.5, 20_001), [9.010913, 9.011, 10.0, 1e30]]
    ).astype(np.float32)
    positive = mt.from_numpy(sample).tanh().numpy()
    negative = mt.from_numpy(-sample).tanh().numpy()
    np.testing.assert_array_equal(positive.view(np.uint32), (-negative).view(np.uint32))


# float32 `erf` and both GELU variants are vectorized too (`ops::simd::
# transcendental`), replacing scalar `libm::erff` and `tanhf`. Unlike `tanh`,
# these are not bit-exact against float64 -- they are within one ulp of it,
# which is still far better than what they replace: `erff` misrounds 2.97% of
# all float32 inputs, this misrounds 68 of 2^32.
#
# The references below are written cancellation-free on purpose. `1 + erf(u)`
# is `erfc(-u)` and `0.5*(1 + tanh(v))` is the logistic `1/(1 + exp(-2v))`;
# spelling them the obvious way makes the *reference* the inaccurate side in
# the negative tail, which is exactly the bug these kernels had to fix.
_SQRT1_2 = 1.0 / np.sqrt(2.0)


def _ulps_apart(got, want):
    g = np.asarray(got, dtype=np.float32).view(np.int32).astype(np.int64)
    w = np.asarray(want, dtype=np.float32).view(np.int32).astype(np.int64)
    g = np.where(g < 0, np.int64(-(2**31)) - g, g)
    w = np.where(w < 0, np.int64(-(2**31)) - w, w)
    return np.abs(g - w)


def _erf_reference(sample):
    from math import erf

    return np.array([erf(float(v)) for v in sample.astype(np.float64)], dtype=np.float32)


def _gelu_erf_reference(sample):
    from math import erfc

    x = sample.astype(np.float64)
    return np.array(
        [0.5 * v * erfc(-v * _SQRT1_2) for v in x], dtype=np.float32
    )


def _gelu_tanh_reference(sample):
    x = sample.astype(np.float64)
    inner = 0.7978845608028654 * (x + 0.044715 * x**3)
    with np.errstate(over="ignore"):
        return (x / (1.0 + np.exp(-2.0 * inner))).astype(np.float32)


_VECTORIZED = [
    ("erf", lambda t: t.erf(), _erf_reference),
    ("gelu", lambda t: t.gelu(), _gelu_erf_reference),
    ("gelu-tanh", lambda t: t.gelu("tanh"), _gelu_tanh_reference),
]


@pytest.mark.parametrize("name,op,reference", _VECTORIZED, ids=[c[0] for c in _VECTORIZED])
@pytest.mark.parametrize("length", [1, 7, 8, 17, 1023, 1024, 16383, 16384, 40000])
def test_vectorized_erf_and_gelu_stay_within_one_ulp(name, op, reference, length):
    rng = np.random.default_rng(4242 + length)
    # Spread over both erf branches (|x| <= 2 and above), the clamp, and the
    # negative tail where the cancellation bug lived.
    sample = np.concatenate(
        [
            rng.standard_normal(length) * 2.0,
            rng.uniform(1.9, 2.1, length),
            rng.uniform(-16.0, -4.0, length),
        ]
    ).astype(np.float32)[:length]

    got = op(mt.from_numpy(sample)).numpy()
    bad = _ulps_apart(got, reference(sample)) > 1
    assert not bad.any(), (
        f"{name} length {length}: {int(bad.sum())} of {length} off by >1 ulp, "
        f"first at x={sample[bad][0]!r}"
    )


@pytest.mark.parametrize("name,op,reference", _VECTORIZED, ids=[c[0] for c in _VECTORIZED])
def test_vectorized_erf_and_gelu_handle_the_edge_cases(name, op, reference):
    sample = np.array(
        [0.0, -0.0, 1e-30, -1e-30, 2.0, -2.0, 4.0, -4.0, 11.0, -11.0, 1e30, -1e30],
        dtype=np.float32,
    )
    got = op(mt.from_numpy(sample)).numpy()
    np.testing.assert_array_equal(_ulps_apart(got, reference(sample)) <= 1, True)
    # Signed zero survives: erf and both GELUs are zero at zero.
    assert not np.signbit(got[0]) and np.signbit(got[1]), got[:2]


def test_gelu_negative_tail_does_not_bottom_out():
    """The `erf` clamp must not leak a residual that `x` then amplifies.

    With the clamp at |x| = 4, `erf` returned 0.99999998 rather than 1, and
    `gelu(-20)` came back as -1.5e-7 instead of -0. Both GELU variants decay
    monotonically to zero here; nothing may plateau.
    """
    xs = np.array([-6.0, -8.0, -10.0, -12.0, -14.0, -20.0, -50.0], dtype=np.float32)
    for name, op, reference in _VECTORIZED[1:]:
        got = op(mt.from_numpy(xs)).numpy()
        want = reference(xs)
        assert np.all(np.abs(got) <= np.abs(want) * 1.5 + 1e-45), f"{name}: {got}"
        assert np.all(np.diff(np.abs(got.astype(np.float64))) <= 0), (
            f"{name} stopped decaying: {got}"
        )
        assert got[-1] == 0.0, f"{name}: gelu(-50) should underflow to zero, got {got[-1]}"


# The float32 GELU *gradient* is vectorized too, and was the most expensive
# gradient in the activation set (7.9ms per million elements, more than the
# forward pass). float64 still goes through the scalar path, so both dtypes are
# checked here -- the two must agree.
def _gelu_erf_grad_reference(sample):
    from math import erfc, exp, pi, sqrt

    return np.array(
        [
            0.5 * erfc(-v / sqrt(2.0)) + v * exp(-0.5 * v * v) / sqrt(2.0 * pi)
            for v in sample.astype(np.float64)
        ]
    )


def _gelu_tanh_grad_reference(sample):
    x = sample.astype(np.float64)
    v = 0.7978845608028654 * (x + 0.044715 * x**3)
    with np.errstate(over="ignore"):
        s = 1.0 / (1.0 + np.exp(-2.0 * v))  # 0.5*(1 + tanh(v))
    sech2 = 4.0 * s * (1.0 - s)
    return s + 0.5 * x * sech2 * 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * x**2)


_GELU_GRADS = [
    ("gelu", None, _gelu_erf_grad_reference),
    ("gelu-tanh", "tanh", _gelu_tanh_grad_reference),
]


@pytest.mark.parametrize("name,approximate,reference", _GELU_GRADS, ids=[c[0] for c in _GELU_GRADS])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_gelu_gradient_matches_the_analytic_derivative(name, approximate, reference, dtype):
    sample = np.concatenate(
        [
            np.random.default_rng(5).standard_normal(5000) * 3,
            np.linspace(-12.0, -4.0, 500),
            [0.0, -0.0, 1e-30, 2.0, -2.0, 11.0, -11.0],
        ]
    ).astype(dtype)
    tensor = mt.Tensor(sample, dtype=dtype, requires_grad=True)
    tensor.gelu(approximate).sum().backward()
    got = tensor.grad.numpy()
    mt.clear_autograd_graph()

    expected = reference(sample)
    tolerance = 2e-7 if dtype == "float32" else 1e-12
    np.testing.assert_allclose(got, expected.astype(dtype), rtol=tolerance, atol=1e-45)


@pytest.mark.parametrize("name,approximate,reference", _GELU_GRADS, ids=[c[0] for c in _GELU_GRADS])
def test_gelu_gradient_tail_does_not_bottom_out(name, approximate, reference):
    """Same failure mode as the forward pass, reached through the derivative.

    `Phi(x)` and `0.5*(1 + tanh(v))` both collapse to zero as `x -> -inf`, and
    reconstructing either by subtraction leaves a floor that never decays.
    """
    xs = np.array([-4.0, -6.0, -8.0, -10.0, -12.0, -14.0, -20.0], dtype="float32")
    tensor = mt.Tensor(xs, dtype="float32", requires_grad=True)
    tensor.gelu(approximate).sum().backward()
    got = np.abs(tensor.grad.numpy().astype(np.float64))
    mt.clear_autograd_graph()

    assert np.all(np.diff(got) <= 0), f"{name} gradient stopped decaying: {got}"
    assert got[-1] == 0.0, f"{name}: gradient at -20 should underflow, got {got[-1]}"
    # And it is the real tail, not an early truncation to zero.
    assert got[1] > 0.0 and got[3] > 0.0, f"{name} truncated too early: {got}"


# `erfc` shares the erf kernel: `erfc(x)` is `1 + erf(-x)`, and above |x| = 2
# the value comes from the erfc branch directly rather than from a subtraction.
# That is the whole point of having a separate `erfc` -- once `erf(x)` rounds to
# 1, `1 - erf(x)` is exactly 0 and the tail is gone.
@pytest.mark.parametrize("length", [1, 7, 17, 1023, 1024, 16383, 16384, 40000])
def test_vectorized_erfc_stays_within_one_ulp(length):
    rng = np.random.default_rng(31337 + length)
    sample = np.concatenate(
        [
            rng.standard_normal(length) * 2.0,
            rng.uniform(1.9, 2.1, length),
            rng.uniform(2.0, 11.0, length),
        ]
    ).astype(np.float32)[:length]

    got = mt.from_numpy(sample).erfc().numpy()
    want = np.array(
        [__import__("math").erfc(float(v)) for v in sample.astype(np.float64)],
        dtype=np.float32,
    )
    bad = _ulps_apart(got, want) > 1
    assert not bad.any(), (
        f"length {length}: {int(bad.sum())} off by >1 ulp, first at x={sample[bad][0]!r}"
    )


def test_erfc_keeps_the_tail_erf_cannot():
    """Past x = 4, erf(x) is 1.0 in float32 and `1 - erf(x)` is exactly zero.

    erfc has to keep decaying there; that is what it is for. The values below
    span 30 orders of magnitude and must all be positive and monotone.
    """
    from math import erfc as erfc_ref

    xs = np.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float32)
    got = mt.from_numpy(xs).erfc().numpy().astype(np.float64)
    want = np.array([erfc_ref(float(v)) for v in xs])

    assert np.all(got > 0.0), f"erfc collapsed to zero: {got}"
    assert np.all(np.diff(got) < 0), f"erfc stopped decaying: {got}"
    # Compared in ulps, not rtol: erfc(10) is 2.09e-45, which lands between
    # float32 subnormals spaced 1.4e-45 apart, so no relative tolerance is
    # meaningful there -- but the correctly rounded value still is.
    assert np.all(_ulps_apart(got.astype(np.float32), want.astype(np.float32)) <= 1), (
        f"got {got}, want {want}"
    )
    # And the identity that motivates the separate routine really does fail:
    assert np.all(1.0 - mt.from_numpy(xs).erf().numpy()[3:] == 0.0)


# `log`, `log1p` and `softplus` run on a second reduction (`u = 2^k * m`), not
# the exp core. `log` comes out bit-identical to the float64 value where
# `f32::ln` misrounds 416,909 of the 2^32 inputs; `log1p` misses exactly one.
#
# The interesting case is small `x`: `1 + x` rounds away most of it, so a naive
# `log(1 + x)` keeps about six digits at `x = 1e-10`. The kernel passes the
# exact residual of that sum through to the log, which restores them.
_LOG_FAMILY = [
    ("log", lambda t: t.log(), np.log, lambda n, rng: np.abs(rng.standard_normal(n)) + 1e-3),
    ("log1p", lambda t: t.log1p(), np.log1p, lambda n, rng: rng.uniform(-0.999, 5.0, n)),
]


@pytest.mark.parametrize("name,op,reference,gen", _LOG_FAMILY, ids=[c[0] for c in _LOG_FAMILY])
@pytest.mark.parametrize("length", [1, 7, 17, 1023, 1024, 16383, 16384, 40000])
def test_log_family_stays_within_one_ulp(name, op, reference, gen, length):
    rng = np.random.default_rng(8080 + length)
    sample = gen(length, rng).astype(np.float32)
    got = op(mt.from_numpy(sample)).numpy()
    want = reference(sample.astype(np.float64)).astype(np.float32)
    bad = _ulps_apart(got, want) > 1
    assert not bad.any(), (
        f"{name} length {length}: {int(bad.sum())} off by >1 ulp, "
        f"first at x={sample[bad][0]!r}"
    )


def test_log1p_keeps_the_digits_that_one_plus_x_would_round_away():
    """The reason `log1p` exists: `log(1 + x)` cannot be formed naively.

    At these magnitudes `1 + x` in float32 *is* 1, and even in float64 it keeps
    only a few digits of `x`. `log1p(x)` must still come back as `x` to within
    an ulp.
    """
    xs = np.array([1e-3, 1e-5, 1e-7, 1e-9, 1e-20, -1e-9, -1e-20], dtype=np.float32)
    got = mt.from_numpy(xs).log1p().numpy()
    want = np.log1p(xs.astype(np.float64)).astype(np.float32)
    np.testing.assert_array_equal(_ulps_apart(got, want) <= 1, True, err_msg=f"{got} vs {want}")
    # Naive float32 reconstruction really does fail here, which is the point:
    # from 1e-9 down, `1 + x` in float32 is exactly 1 and the value is gone.
    assert np.all((1.0 + xs.astype(np.float32))[3:] == 1.0)


@pytest.mark.parametrize("value,expected", [(-1.0, -np.inf), (-1.5, np.nan), (0.0, 0.0)])
def test_log1p_domain_edges(value, expected):
    got = mt.from_numpy(np.array([value], dtype=np.float32)).log1p().numpy()[0]
    if np.isnan(expected):
        assert np.isnan(got)
    else:
        assert got == expected


def test_log1p_preserves_signed_zero():
    got = mt.from_numpy(np.array([0.0, -0.0], dtype=np.float32)).log1p().numpy()
    assert not np.signbit(got[0]) and np.signbit(got[1]), got


def test_softplus_matches_the_stable_reference():
    # softplus(x) = log1p(exp(x)); the reference is written as the max form so
    # the large-x side does not overflow before the comparison.
    sample = np.concatenate(
        [np.random.default_rng(11).standard_normal(20_000) * 8, np.linspace(-60, 60, 2000)]
    ).astype(np.float32)
    got = mt.from_numpy(sample).softplus().numpy()
    x = sample.astype(np.float64)
    want = (np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))).astype(np.float32)
    np.testing.assert_allclose(got, want, rtol=2e-6, atol=1e-45)


def test_softplus_negative_tail_is_exponential_not_zero():
    """For very negative x, softplus(x) decays like exp(x) and must not floor.

    Going through `log(2 + expm1(v))` instead of `log1p(exp(v))` rounds the tail
    away and leaves it about 0.2% wrong. Compared against the true
    `log1p(exp(x))`, not against `exp(x)` -- those differ by `exp(2x)/2`, which
    is 2.3e-5 relative at x = -10 and would make the test measure the asymptote
    rather than the kernel.
    """
    xs = np.array([-10.0, -20.0, -30.0, -40.0, -60.0], dtype=np.float32)
    got = mt.from_numpy(xs).softplus().numpy().astype(np.float64)
    want = np.log1p(np.exp(xs.astype(np.float64)))
    np.testing.assert_allclose(got, want, rtol=1e-6)
    assert np.all(np.diff(got) < 0), f"softplus tail stopped decaying: {got}"
    assert got[-1] > 0.0, "softplus underflowed to zero at x = -60"
