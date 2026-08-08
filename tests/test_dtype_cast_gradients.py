# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A dtype cast is differentiable, and mixed-dtype arithmetic depends on it.

`astype` built its result with `Tensor::new`, which mints a fresh id and no
`grad_fn` and never joins the graph, while still copying `requires_grad` across.
So the output claimed to be tracked and had nothing behind it:

    w = Tensor(..., dtype="float64", requires_grad=True)
    w.astype("float32").matmul(x).sum().backward()
    get_gradient(w)   ->  None

Writing `astype` by hand is the rarer way to hit that. Binary operands are
coerced to a common dtype through the same call, so an `f32` tensor added to an
`f64` one was promoted and lost its gradient outright -- while the `f64` side of
the same expression kept its own, because it was the one not being cast. The
whole point of the promotion is that a caller does not have to think about it,
and the failure is silent: an optimizer sees a parameter with no gradient and
skips it exactly as it would one that took no part in the loss.

A cast is the identity on values, so its gradient is the identity, carried back
at whatever precision the input was held in. Only float-to-float conversions get
one; an integer or bool result cannot carry a gradient at all and is now marked
as not requiring one, rather than claiming to and delivering nothing.

That in turn made the construction order matter. `Tensor([1.0, 2.0],
requires_grad=True)` reaches its dtype by converting at float64 and casting, and
with the cast differentiable the tensor handed back stopped being a leaf -- it
became an interior node whose input was a throwaway temporary, so the first
backward pass released its stored gradient with the rest of the subgraph, and
the second reported none. Construction now marks the flag last, after every
conversion, which is why the accumulation tests below sit in this file: they are
what a cast becoming differentiable can break.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

FLOAT_DTYPES = ["float32", "float64"]
NON_FLOAT_DTYPES = ["int32", "int64", "bool"]


def _tensor(values, dtype, requires_grad=False):
    return mt.Tensor(
        np.asarray(values, dtype=dtype), dtype=dtype, requires_grad=requires_grad
    )


# --- the cast itself --------------------------------------------------------


@pytest.mark.parametrize("source", FLOAT_DTYPES)
@pytest.mark.parametrize("target", FLOAT_DTYPES)
def test_a_float_cast_passes_its_gradient_straight_through(source, target):
    """The identity on values, so the identity on gradients."""
    x = _tensor([1.0, 2.0, 3.0, 4.0], source, requires_grad=True)

    x.astype(target).sum().backward()

    gradient = mt.get_gradient(x)
    assert gradient is not None, "the cast severed the graph"
    assert gradient.dtype == source, "the gradient came back at the wrong precision"
    np.testing.assert_allclose(gradient.numpy(), np.ones(4))


@pytest.mark.parametrize("source", FLOAT_DTYPES)
@pytest.mark.parametrize("target", FLOAT_DTYPES)
def test_a_cast_scales_the_gradient_of_what_follows(source, target):
    x = _tensor([1.0, 2.0], source, requires_grad=True)
    weights = _tensor([3.0, 4.0], target)

    (x.astype(target) * weights).sum().backward()

    np.testing.assert_allclose(mt.get_gradient(x).numpy(), [3.0, 4.0], rtol=1e-6)


def test_a_round_trip_through_lower_precision_still_reaches_the_input():
    x = _tensor([1.0, 2.0, 3.0], "float64", requires_grad=True)

    x.astype("float32").astype("float64").sum().backward()

    np.testing.assert_allclose(mt.get_gradient(x).numpy(), np.ones(3))


@pytest.mark.parametrize("target", NON_FLOAT_DTYPES)
@pytest.mark.parametrize("source", FLOAT_DTYPES)
def test_a_cast_that_cannot_carry_a_gradient_says_so(source, target):
    """An integer or bool result has no gradient to give, so it must not claim
    to be tracked -- that is the state that reads as "tracked" and delivers
    nothing."""
    x = _tensor([1.0, 2.0, 3.0], source, requires_grad=True)

    assert x.astype(target).requires_grad is False


def test_a_cast_under_no_grad_is_not_tracked():
    x = _tensor([1.0, 2.0], "float64", requires_grad=True)

    with mt.no_grad():
        assert x.astype("float32").requires_grad is False


@pytest.mark.parametrize("source", FLOAT_DTYPES + NON_FLOAT_DTYPES)
@pytest.mark.parametrize("target", FLOAT_DTYPES + NON_FLOAT_DTYPES)
def test_the_values_a_cast_produces_are_unchanged(source, target):
    """Only the graph wiring changed; pin the arithmetic so a later change to
    one cannot quietly move the other."""
    values = (
        np.array([True, False, True])
        if source == "bool"
        else np.array([0, 1, 2]).astype(source)
    )

    got = mt.Tensor(values, dtype=source).astype(target).numpy()

    np.testing.assert_array_equal(got, values.astype(target))


# --- the path most callers reach it by --------------------------------------


@pytest.mark.parametrize("op", ["add", "mul", "sub", "div"])
def test_the_promoted_operand_keeps_its_gradient(op):
    """`f32` against `f64` promotes the `f32` side, which is exactly the side
    that used to lose its gradient."""
    x = _tensor([1.0, 2.0], "float32", requires_grad=True)
    other = _tensor([4.0, 5.0], "float64")

    result = {
        "add": lambda: x + other,
        "mul": lambda: x * other,
        "sub": lambda: x - other,
        "div": lambda: x / other,
    }[op]()
    result.sum().backward()

    expected = {
        "add": [1.0, 1.0],
        "mul": [4.0, 5.0],
        "sub": [1.0, 1.0],
        "div": [0.25, 0.2],
    }[op]
    gradient = mt.get_gradient(x)
    assert gradient is not None, "the promoted operand lost its gradient"
    np.testing.assert_allclose(gradient.numpy(), expected, rtol=1e-6)


def test_the_operand_that_is_not_promoted_was_always_fine():
    """The other half of the asymmetry: only one side was being cast, so only
    one side was affected."""
    x = _tensor([1.0, 2.0], "float64", requires_grad=True)
    other = _tensor([3.0, 4.0], "float32")

    (x * other).sum().backward()

    np.testing.assert_allclose(mt.get_gradient(x).numpy(), [3.0, 4.0], rtol=1e-6)


def test_both_operands_get_their_own_gradient():
    x = _tensor([1.0, 2.0], "float32", requires_grad=True)
    y = _tensor([3.0, 4.0], "float64", requires_grad=True)

    (x * y).sum().backward()

    np.testing.assert_allclose(mt.get_gradient(x).numpy(), [3.0, 4.0], rtol=1e-6)
    np.testing.assert_allclose(mt.get_gradient(y).numpy(), [1.0, 2.0], rtol=1e-6)


def test_an_integer_operand_does_not_need_one():
    x = _tensor([1.0, 2.0], "float32", requires_grad=True)
    counts = mt.Tensor(np.array([3, 4], dtype=np.int64), dtype="int64")

    (x * counts).sum().backward()

    np.testing.assert_allclose(mt.get_gradient(x).numpy(), [3.0, 4.0], rtol=1e-6)


def test_a_parameter_kept_in_higher_precision_still_trains():
    """The use case: weights held at float64, arithmetic done at float32."""
    weights = _tensor([[1.0, 2.0], [3.0, 4.0]], "float64", requires_grad=True)
    batch = _tensor([[1.0], [1.0]], "float32")

    weights.astype("float32").matmul(batch).sum().backward()

    gradient = mt.get_gradient(weights)
    assert gradient is not None
    np.testing.assert_allclose(gradient.numpy(), np.ones((2, 2)))


# --- what a differentiable cast must not break ------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda: mt.Tensor([1.0, 2.0, 3.0], requires_grad=True),
        lambda: mt.Tensor(
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            dtype="float32",
            requires_grad=True,
        ),
        lambda: mt.Tensor(
            np.array([1, 2, 3], dtype=np.int32), dtype="float64", requires_grad=True
        ),
        lambda: mt.Tensor((1.0, 2.0, 3.0), requires_grad=True),
    ],
    ids=["list", "downcast_array", "int_array", "tuple"],
)
def test_a_constructed_tensor_is_a_leaf_however_it_was_converted(build):
    """Construction reaches the requested dtype by casting, so a differentiable
    cast turned what the caller built into an interior node."""
    tensor = build()

    assert tensor.requires_grad is True
    assert tensor.is_leaf is True


@pytest.mark.parametrize(
    "build",
    [
        lambda: mt.Tensor([1.0, 2.0, 3.0], requires_grad=True),
        lambda: mt.Tensor(
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            dtype="float32",
            requires_grad=True,
        ),
    ],
    ids=["list", "downcast_array"],
)
def test_gradients_still_accumulate_across_backward_calls(build):
    """A non-leaf has its stored gradient released with the rest of the
    subgraph, so this reported the first pass and then nothing."""
    x = build()

    (x * 2.0).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), [2.0, 2.0, 2.0])

    (x * 3.0).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), [5.0, 5.0, 5.0])

    x.zero_grad()
    (x * 4.0).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), [4.0, 4.0, 4.0])

    mt.clear_autograd_graph()


def test_as_tensor_keeps_the_flag_it_was_given():
    array = np.arange(6, dtype=np.int32).reshape((2, 3))

    tensor = mt.Tensor.as_tensor(array, dtype="float64", requires_grad=True)

    assert tensor.dtype == "float64"
    assert tensor.requires_grad is True
    np.testing.assert_allclose(tensor.numpy(), array.astype(np.float64))


def test_construction_inside_no_grad_is_unaffected():
    with mt.no_grad():
        assert mt.Tensor([1.0, 2.0], requires_grad=True).requires_grad is False
