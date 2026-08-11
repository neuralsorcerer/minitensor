# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Two gradients that were wrong only where the ordinary formula breaks down.

Both were found by finite-difference gradcheck at the points the usual
derivation cannot reach -- a zero inside a running product, and a variance with
one sample -- rather than at a random interior point, where 64 ops all agreed.

`cumprod`'s gradient divides the running sum by the input, so a zero needs its
own handling. The code split at the first zero and got that case right, then
took a *separate* branch for two or more zeros that set the whole run to zero:

    cumprod([2, 0, 3, 0]).sum().backward()   ->  [0, 0, 0, 0]
    true gradient                            ->  [1, 8, 0, 0]

The split at the first zero never needed a second case. Positions before it
only see outputs before it, since every later one carries that zero as a
factor; the position of the zero itself is assembled from a prefix and a
running suffix product, and a second zero drives that suffix to zero on its own,
so the terms beyond it drop out. Positions after the first zero are zero either
way. So the fix was to delete the extra branch, and these tests sweep zero
patterns exhaustively to keep it deleted.

`var(unbiased=True)` over an axis of one element is NaN, because Bessel's
correction divides by `n - 1`. That much was right. But it was produced by
swapping in a freshly built NaN tensor, which carries no `grad_fn` and no graph
node -- so the result claimed `requires_grad = True` and then left the input
with *no gradient at all*:

    x.var(1).sum().backward()   ->  get_gradient(x) is None

A missing gradient reads as "this parameter was not used" and an optimizer skips
it without a word. NaN says plainly that
something is undefined. Routing the degenerate case through the same multiply as
every other correction keeps the chain intact and lets the NaN through.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pytest

import minitensor as mt

DTYPES = ["float32", "float64"]


def _grad(fn, values, dtype="float64"):
    x = mt.Tensor(np.asarray(values, dtype=dtype), dtype=dtype, requires_grad=True)
    fn(x).sum().backward()
    return mt.get_gradient(x)


def _numeric(fn, values, eps=1e-6):
    """Central differences on the float64 forward pass."""
    values = np.asarray(values, dtype=np.float64)
    out = np.zeros_like(values)
    for i in range(values.size):
        high = values.copy().reshape(-1)
        low = values.copy().reshape(-1)
        high[i] += eps
        low[i] -= eps
        a = fn(mt.Tensor(high.reshape(values.shape), dtype="float64")).numpy().sum()
        b = fn(mt.Tensor(low.reshape(values.shape), dtype="float64")).numpy().sum()
        out.reshape(-1)[i] = (float(a) - float(b)) / (2 * eps)
    return out


# --- cumprod ----------------------------------------------------------------


def test_the_reported_case():
    """Two zeros in one run, stated directly."""
    got = _grad(lambda t: t.cumprod(1), [[2.0, 0.0, 3.0, 0.0]]).numpy()
    np.testing.assert_allclose(got.ravel(), [1.0, 8.0, 0.0, 0.0], rtol=1e-9)


@pytest.mark.parametrize(
    "values",
    [
        [2.0, 3.0, 4.0],  # no zero: the plain division path
        [2.0, 0.0, 3.0, 4.0],  # one zero
        [2.0, 0.0, 3.0, 0.0],  # two, separated
        [1.0, 2.0, 0.0, 0.0, 5.0],  # two, adjacent
        [0.0, 0.0, 3.0],  # two, leading
        [0.0, 2.0, 0.0, 3.0],
        [2.0, 3.0, 0.0],  # trailing
        [2.0, 0.0, 0.0, 0.0],  # three
        [0.0, 0.0, 0.0],  # all
        [0.0],  # the whole run is one zero
    ],
    ids=lambda v: "".join("z" if x == 0 else "n" for x in v),
)
def test_cumprod_gradient_matches_central_differences(values):
    rows = [values]
    analytic = _grad(lambda t: t.cumprod(1), rows).numpy().astype(np.float64)
    np.testing.assert_allclose(
        analytic, _numeric(lambda t: t.cumprod(1), rows), rtol=1e-4, atol=1e-6
    )


def test_every_zero_pattern_in_a_short_run():
    """Exhaustive over which positions are zero, which is the only thing the
    deleted branch keyed on."""
    for pattern in itertools.product([0.0, 2.0], repeat=5):
        rows = [list(pattern)]
        analytic = _grad(lambda t: t.cumprod(1), rows).numpy().astype(np.float64)
        np.testing.assert_allclose(
            analytic,
            _numeric(lambda t: t.cumprod(1), rows),
            rtol=1e-4,
            atol=1e-6,
            err_msg=f"pattern {pattern}",
        )


@pytest.mark.parametrize(
    "values",
    [
        [2.0, 3.0, 4.0],
        [2.0, 0.0, 3.0, 4.0],
        [2.0, 0.0, 3.0, 0.0],
        [0.0, 0.0, 3.0],
        [1.0, 2.0, 0.0, 0.0, 5.0],
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ],
    ids=lambda v: "".join("z" if x == 0 else "n" for x in v),
)
def test_a_one_dimensional_run_has_its_own_kernel(values):
    """Rank 1, rank 2 and rank 3 are three separate copies of the split, and a
    2-D test leaves the 1-D one entirely unexercised."""
    analytic = _grad(lambda t: t.cumprod(0), values).numpy().astype(np.float64)
    np.testing.assert_allclose(
        analytic, _numeric(lambda t: t.cumprod(0), values), rtol=1e-4, atol=1e-6
    )


def test_every_zero_pattern_in_a_one_dimensional_run():
    for pattern in itertools.product([0.0, 2.0], repeat=5):
        values = list(pattern)
        analytic = _grad(lambda t: t.cumprod(0), values).numpy().astype(np.float64)
        np.testing.assert_allclose(
            analytic,
            _numeric(lambda t: t.cumprod(0), values),
            rtol=1e-4,
            atol=1e-6,
            err_msg=f"pattern {pattern}",
        )


@pytest.mark.parametrize("dim", [0, 1])
def test_cumprod_gradient_along_either_dim(dim):
    values = [[2.0, 0.0], [0.0, 2.0], [3.0, 0.0], [0.0, 4.0]]
    analytic = _grad(lambda t: t.cumprod(dim), values).numpy().astype(np.float64)
    np.testing.assert_allclose(
        analytic, _numeric(lambda t: t.cumprod(dim), values), rtol=1e-4, atol=1e-6
    )


def test_cumprod_gradient_on_a_three_dimensional_tensor():
    """The N-D path is a fourth copy of the same split and had the same branch."""
    values = [[[2.0, 0.0, 3.0, 0.0], [1.0, 2.0, 0.0, 0.0]]]
    analytic = _grad(lambda t: t.cumprod(2), values).numpy().astype(np.float64)
    np.testing.assert_allclose(
        analytic, _numeric(lambda t: t.cumprod(2), values), rtol=1e-4, atol=1e-6
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_both_float_dtypes_agree(dtype):
    values = [[2.0, 0.0, 3.0, 0.0, 5.0]]
    got = _grad(lambda t: t.cumprod(1), values, dtype).numpy().astype(np.float64)
    np.testing.assert_allclose(
        got, _numeric(lambda t: t.cumprod(1), values), rtol=1e-4, atol=1e-6
    )


def test_a_negative_dim_names_the_same_axis():
    values = [[2.0, 0.0, 3.0, 0.0]]
    np.testing.assert_array_equal(
        _grad(lambda t: t.cumprod(-1), values).numpy(),
        _grad(lambda t: t.cumprod(1), values).numpy(),
    )


def test_the_forward_pass_was_never_wrong():
    """Only the gradient was affected; pinning the forward keeps a fix to one
    from silently changing the other."""
    values = np.array([[2.0, 0.0, 3.0, 0.0], [1.0, 2.0, 3.0, 4.0]])
    got = mt.Tensor(values, dtype="float64").cumprod(1).numpy()
    np.testing.assert_allclose(got, np.cumprod(values, axis=1))


# --- var / std with a single sample -----------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_an_undefined_variance_still_reaches_its_input(dtype):
    """The gradient must exist and be NaN, not be missing entirely."""
    gradient = _grad(lambda t: t.var(1), [[3.0], [5.0]], dtype)

    assert gradient is not None, "backward() left the input with no gradient"
    assert np.isnan(gradient.numpy()).all()


@pytest.mark.parametrize("dtype", DTYPES)
def test_std_carries_it_too(dtype):
    gradient = _grad(lambda t: t.std(1), [[3.0], [5.0]], dtype)

    assert gradient is not None
    assert np.isnan(gradient.numpy()).all()


def test_a_one_element_tensor_reduced_entirely():
    gradient = _grad(lambda t: t.var(), [[3.0]])

    assert gradient is not None
    assert np.isnan(gradient.numpy()).all()


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_biased_form_is_defined_and_unaffected(dtype):
    """`unbiased=False` divides by `n`, so one sample is a perfectly good zero
    and its gradient is a real number."""
    gradient = _grad(lambda t: t.var(1, unbiased=False), [[3.0], [5.0]], dtype)

    assert gradient is not None
    np.testing.assert_array_equal(gradient.numpy(), np.zeros((2, 1), dtype=dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_undefined_forward_value_is_unchanged(dtype):
    """It was NaN before and has to stay NaN -- matching `numpy.var(ddof=1)`."""
    values = np.array([[3.0], [5.0]], dtype=dtype)

    got = mt.Tensor(values, dtype=dtype).var(1).numpy()

    assert np.isnan(got).all()
    with warnings.catch_warnings():
        # NumPy warns about the degrees of freedom rather than returning quietly.
        warnings.simplefilter("ignore", RuntimeWarning)
        reference = np.var(values, 1, ddof=1)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(reference))


@pytest.mark.parametrize("width", [2, 3, 5])
def test_ordinary_variances_are_untouched(width):
    """The degenerate case now shares the multiply with every other correction,
    so the ordinary path has to come out exactly as before."""
    rng = np.random.default_rng(0)
    values = rng.standard_normal((4, width))

    got = mt.Tensor(values, dtype="float64").var(1).numpy()
    np.testing.assert_allclose(got, np.var(values, axis=1, ddof=1), rtol=1e-10)

    analytic = _grad(lambda t: t.var(1), values).numpy()
    np.testing.assert_allclose(
        analytic, _numeric(lambda t: t.var(1), values), rtol=1e-4, atol=1e-6
    )


def test_a_variance_that_requires_grad_still_matches_the_fast_path():
    """`var` has a fused kernel for tensors that do not require gradients and a
    composed one for those that do; they must agree."""
    rng = np.random.default_rng(1)
    values = rng.standard_normal((4, 6))

    plain = mt.Tensor(values, dtype="float64").var(1).numpy()
    tracked = mt.Tensor(values, dtype="float64", requires_grad=True).var(1)

    np.testing.assert_allclose(plain, tracked.numpy(), rtol=1e-12)
