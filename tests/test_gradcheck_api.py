# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`mt.gradcheck`: the check a hand-written backward had no way to get.

The engine's own gradients are checked against finite differences all over this
suite. What had no route to the same check was the other kind -- a backward
written through `autograd.Function` or `register_custom_op` -- because the only
implementations of it were `_numeric_grad` helpers private to nine test modules.
A user could not call any of them.

The case that matters is the last two tests here: a `Function` whose forward is
right and whose backward is wrong by a constant factor. It passes every test of
the forward's values, trains a slightly wrong model, and nothing says so.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt
from minitensor.autograd import Function


@pytest.fixture(autouse=True)
def _clean_graph():
    yield
    mt.clear_autograd_graph()


def _f64(values, requires_grad=True):
    return mt.Tensor(
        np.ascontiguousarray(np.asarray(values, dtype=np.float64)),
        dtype="float64",
        requires_grad=requires_grad,
    )


class Cube(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * x * x

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * x * x * 3.0


class CubeMissingTheThree(Function):
    """`Cube` with the factor of 3 dropped -- the forward is still exact."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * x * x

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * x * x


@pytest.mark.parametrize(
    "name,fn",
    [
        ("sum", lambda x: x.sum()),
        ("square", lambda x: (x * x).sum()),
        ("tanh", lambda x: x.tanh().sum()),
        ("sigmoid", lambda x: x.sigmoid().sum()),
        ("exp", lambda x: x.exp().sum()),
        ("softplus", lambda x: (x.exp() + 1.0).log().sum()),
    ],
)
def test_builtin_gradients_pass(name, fn):
    assert mt.gradcheck(fn, (_f64([-1.1, -0.25, 0.4, 1.7]),))


def test_several_inputs_are_each_checked():
    a, b = _f64([0.5, -1.5]), _f64([2.0, 0.25])
    assert mt.gradcheck(lambda p, q: (p * q).sum(), (a, b))


def test_an_input_without_requires_grad_is_passed_through():
    a, b = _f64([0.5, -1.5]), _f64([2.0, 0.25], requires_grad=False)
    assert mt.gradcheck(lambda p, q: (p * q).sum(), (a, b))
    assert b.grad is None


def test_a_matmul_gradient_passes():
    """The engine's own GEMM: 6x5x4 is far below the delegation threshold."""
    rng = np.random.default_rng(3)
    a, b = _f64(rng.standard_normal((6, 5))), _f64(rng.standard_normal((5, 4)))
    assert mt.gradcheck(lambda p, q: p.matmul(q).sum(), (a, b))


def test_a_delegated_matmul_gradient_passes():
    """The same product through NumPy's BLAS instead.

    A shape this small never reaches the provider on its own -- the threshold
    is there precisely because a product this size is cheaper to compute than
    to hand over -- so the thresholds come down for the duration. Without that
    the delegated path has no gradient anyone has checked.
    """
    dispatch = mt._core.dispatch
    if not dispatch.provider_installed():
        pytest.skip("no NumPy provider in this build")

    previous = dispatch.set_gemm_thresholds(0, 0)
    try:
        rng = np.random.default_rng(4)
        a, b = _f64(rng.standard_normal((6, 5))), _f64(rng.standard_normal((5, 4)))
        assert mt.gradcheck(lambda p, q: p.matmul(q).sum(), (a, b))
    finally:
        dispatch.set_gemm_thresholds(*previous)


def test_the_same_tensor_passed_twice():
    """One variable in two argument slots.

    Its analytic gradient accumulates from both occurrences, so both have to be
    perturbed together; perturbing one would measure half of it and fail a
    backward that was correct. For `sum(a * b)` with `a is b` the gradient is
    `2x`, not `x`.
    """
    x = _f64([1.0, 2.0, 3.0])
    assert mt.gradcheck(lambda a, b: (a * b).sum(), (x, x))
    np.testing.assert_allclose(np.asarray(x.grad), [2.0, 4.0, 6.0], rtol=1e-9)


def test_a_non_tensor_return_is_refused_by_name():
    with pytest.raises(ValueError, match="scalar tensor; got float"):
        mt.gradcheck(lambda t: 3.0, (_f64([1.0, 2.0]),))


def test_a_correct_custom_function_passes():
    assert mt.gradcheck(lambda t: Cube.apply(t).sum(), (_f64([0.7, -1.3, 2.1]),))


def test_a_wrong_custom_function_is_caught():
    with pytest.raises(AssertionError, match="gradient mismatch"):
        mt.gradcheck(
            lambda t: CubeMissingTheThree.apply(t).sum(), (_f64([0.7, -1.3, 2.1]),)
        )


def test_raise_exception_false_returns_false_instead():
    got = mt.gradcheck(
        lambda t: CubeMissingTheThree.apply(t).sum(),
        (_f64([0.7, -1.3, 2.1]),),
        raise_exception=False,
    )
    assert got is False


def test_float32_is_refused_rather_than_silently_weakened():
    x = mt.Tensor([1.0, 2.0], dtype="float32", requires_grad=True)
    with pytest.raises(ValueError, match="float64"):
        mt.gradcheck(lambda t: t.sum(), (x,))


def test_a_non_scalar_output_is_refused():
    with pytest.raises(ValueError, match="scalar"):
        mt.gradcheck(lambda t: t * 2.0, (_f64([1.0, 2.0]),))


def test_no_differentiable_input_is_refused():
    with pytest.raises(ValueError, match="requires_grad"):
        mt.gradcheck(lambda t: t.sum(), (_f64([1.0, 2.0], requires_grad=False),))


def test_the_message_locates_the_disagreement():
    with pytest.raises(AssertionError) as excinfo:
        mt.gradcheck(lambda t: CubeMissingTheThree.apply(t).sum(), (_f64([2.0]),))
    message = str(excinfo.value)
    assert "flat position 0" in message
    assert "1 of 1 elements disagree" in message


def test_a_stale_gradient_does_not_leak_into_the_check():
    """Gradients accumulate here, so a check run twice on one tensor would see
    double the second time if it did not clear first."""
    x = _f64([0.5, 1.5, -0.5])
    assert mt.gradcheck(lambda t: t.tanh().sum(), (x,))
    assert mt.gradcheck(lambda t: t.tanh().sum(), (x,))


def test_gradcheck_leaves_no_graph_behind():
    x = _f64([0.5, 1.5, -0.5])
    mt.clear_autograd_graph()
    mt.gradcheck(lambda t: t.tanh().sum(), (x,))
    before, _ = mt.autograd_graph_size()
    mt.gradcheck(lambda t: t.tanh().sum(), (x,))
    after, _ = mt.autograd_graph_size()
    assert after == before


def test_a_non_finite_comparison_says_so_rather_than_counting_zero():
    """An out-of-domain input makes the numerical gradient NaN.

    Every comparison against a NaN is false, so the ordinary report would count
    zero elements as disagreeing while still failing -- a message that reads
    like a contradiction. `acos` outside (-1, 1) returns NaN rather than
    raising, which is how a caller lands here without doing anything obviously
    wrong.
    """
    x = _f64([0.37, 1.43])  # 1.43 is outside acos's domain
    with pytest.raises(AssertionError) as excinfo:
        mt.gradcheck(lambda t: t.acos().sum(), (x,))
    message = str(excinfo.value)
    assert "could not be compared" in message
    assert "NaN or infinite" in message
    assert "outside the function's domain" in message
    assert "elements disagree" not in message
