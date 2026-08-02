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
