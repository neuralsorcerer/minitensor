# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`autograd.Function`: an operation defined at one call site, with a `ctx`.

`register_custom_op` was already there and already enough for an operation the
whole program reaches for by name. What it cannot express is an operation whose
backward needs something its own forward computed, because the registry holds
one operation and every call goes through it -- two live calls would share
whatever the second one stashed.

`Function` gives each `apply` its own node and its own `ctx`, and puts neither
in the registry. So these tests are as much about what does *not* happen: no
name is invented, nothing is registered, and two calls that overlap cannot see
each other's saved state.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt
from minitensor.autograd import Function, FunctionCtx


def _t(values, requires_grad=False):
    return mt.Tensor(
        np.ascontiguousarray(np.asarray(values, dtype=np.float64)),
        dtype="float64",
        requires_grad=requires_grad,
    )


class Triple(Function):
    """The smallest thing with a hand-written backward."""

    @staticmethod
    def forward(ctx, x):
        return x * 3.0

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * 3.0


class Clamp(Function):
    """Non-tensor arguments, and a backward that needs them."""

    @staticmethod
    def forward(ctx, x, low, high):
        ctx.save_for_backward(x)
        ctx.low = low
        ctx.high = high
        return x.clip(low, high)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        inside = (x > ctx.low) & (x < ctx.high)
        return grad_output * inside.astype("float64"), None, None


class Mul(Function):
    """Two tensor arguments, each with its own gradient."""

    @staticmethod
    def forward(ctx, p, q):
        ctx.save_for_backward(p, q)
        return p * q

    @staticmethod
    def backward(ctx, grad_output):
        p, q = ctx.saved_tensors
        return grad_output * q, grad_output * p


# --- the operation runs and differentiates -----------------------------------


def test_apply_computes_the_forward():
    out = Triple.apply(_t([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(out.numpy(), [3.0, 6.0, 9.0])


def test_the_hand_written_backward_is_the_gradient():
    x = _t([1.0, 2.0, 3.0], requires_grad=True)
    Triple.apply(x).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), [3.0, 3.0, 3.0])


def test_the_output_requires_grad_when_an_input_does():
    assert Triple.apply(_t([1.0], requires_grad=True)).requires_grad
    assert not Triple.apply(_t([1.0])).requires_grad


def test_a_forward_that_no_input_needs_a_gradient_for_stays_off_the_graph():
    out = Triple.apply(_t([1.0, 2.0]))
    with pytest.raises(RuntimeError):
        out.sum().backward()


def test_the_gradient_reaches_every_tensor_argument():
    p = _t([1.0, 2.0, 3.0], requires_grad=True)
    q = _t([4.0, 5.0, 6.0], requires_grad=True)
    Mul.apply(p, q).sum().backward()
    np.testing.assert_allclose(p.grad.numpy(), [4.0, 5.0, 6.0])
    np.testing.assert_allclose(q.grad.numpy(), [1.0, 2.0, 3.0])


def test_a_function_composes_with_the_ops_around_it():
    x = _t([1.0, 2.0], requires_grad=True)
    (Triple.apply(x * 2.0) + 1.0).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), [6.0, 6.0])


# --- non-tensor arguments ----------------------------------------------------


def test_non_tensor_arguments_reach_the_forward_unchanged():
    x = _t([-2.0, 0.0, 2.0])
    out = Clamp.apply(x, -1.0, 1.0)
    np.testing.assert_allclose(out.numpy(), np.clip(x.numpy(), -1.0, 1.0))


def test_a_backward_may_answer_for_the_non_tensor_positions_too():
    x = _t([-2.0, 0.0, 2.0], requires_grad=True)
    Clamp.apply(x, -1.0, 1.0).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), [0.0, 1.0, 0.0])


def test_a_backward_may_answer_for_the_tensors_alone():
    class TensorsOnly(Clamp):
        @staticmethod
        def backward(ctx, grad_output):
            (x,) = ctx.saved_tensors
            inside = (x > ctx.low) & (x < ctx.high)
            return grad_output * inside.astype("float64")

    x = _t([-2.0, 0.0, 2.0], requires_grad=True)
    TensorsOnly.apply(x, -1.0, 1.0).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), [0.0, 1.0, 0.0])


def test_a_backward_of_the_wrong_arity_says_both_lengths_it_would_take():
    class Wrong(Function):
        @staticmethod
        def forward(ctx, x, scale):
            return x * scale

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output, grad_output, grad_output, grad_output

    x = _t([1.0], requires_grad=True)
    with pytest.raises(
        ValueError, match=r"returned 4 gradient\(s\); expected 1 .* or 2"
    ):
        Wrong.apply(x, 2.0).sum().backward()


def test_apply_needs_a_tensor_somewhere_in_its_arguments():
    with pytest.raises(TypeError, match="at least one tensor"):
        Triple.apply(3.0)


# --- one ctx per call --------------------------------------------------------


def test_two_live_calls_do_not_share_their_saved_state():
    """The reason `Function` exists at all.

    Both nodes are on the graph before either backward runs, so a `ctx` on the
    operation rather than the call would have the second call's bounds
    answering for the first.
    """

    x = _t([-2.0, 0.0, 2.0], requires_grad=True)
    y = _t([-2.0, 0.0, 2.0], requires_grad=True)

    narrow = Clamp.apply(x, -1.0, 1.0)
    wide = Clamp.apply(y, -5.0, 5.0)
    (narrow.sum() + wide.sum()).backward()

    np.testing.assert_allclose(x.grad.numpy(), [0.0, 1.0, 0.0])
    np.testing.assert_allclose(y.grad.numpy(), [1.0, 1.0, 1.0])


def test_the_ctx_survives_until_the_backward_runs():
    x = _t([1.0, 2.0], requires_grad=True)
    out = Mul.apply(x, _t([3.0, 4.0]))
    for _ in range(3):
        Mul.apply(_t([9.0]), _t([9.0]))  # other calls, other contexts
    out.sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), [3.0, 4.0])


# --- nothing is registered ---------------------------------------------------


def test_apply_registers_nothing():
    before = set(mt.list_custom_ops_py())
    Triple.apply(_t([1.0], requires_grad=True)).sum().backward()
    assert set(mt.list_custom_ops_py()) == before
    assert not mt.is_custom_op_registered_py("Triple")


def test_the_same_function_applies_as_many_times_as_it_likes():
    """A registry would refuse the second one as a duplicate name."""

    total = None
    for _ in range(5):
        out = Triple.apply(_t([1.0], requires_grad=True))
        total = out if total is None else total + out
    np.testing.assert_allclose(total.numpy(), [15.0])


# --- FunctionCtx ------------------------------------------------------------


def test_saved_tensors_come_back_in_the_order_they_were_saved():
    ctx = FunctionCtx((True, True))
    first, second = _t([1.0]), _t([2.0])
    ctx.save_for_backward(first, second)
    got = ctx.saved_tensors
    assert len(got) == 2
    np.testing.assert_allclose(got[0].numpy(), [1.0])
    np.testing.assert_allclose(got[1].numpy(), [2.0])


def test_asking_for_saved_tensors_that_were_never_saved_says_so():
    ctx = FunctionCtx((True,))
    with pytest.raises(RuntimeError, match="save_for_backward"):
        _ = ctx.saved_tensors


def test_save_for_backward_with_no_arguments_is_not_the_same_as_not_saving():
    ctx = FunctionCtx((True,))
    ctx.save_for_backward()
    assert ctx.saved_tensors == ()


def test_needs_input_grad_has_one_flag_per_argument():
    seen = {}

    class Recorder(Function):
        @staticmethod
        def forward(ctx, a, b, scale):
            seen["flags"] = ctx.needs_input_grad
            return a * b * scale

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output, grad_output, None

    Recorder.apply(_t([1.0], requires_grad=True), _t([2.0]), 3.0)
    assert seen["flags"] == (True, False, False)


def test_the_ctx_carries_whatever_the_forward_puts_on_it():
    ctx = FunctionCtx(())
    ctx.anything = {"a": 1}
    assert ctx.anything == {"a": 1}


def test_the_repr_says_how_much_is_saved():
    ctx = FunctionCtx((True, False))
    assert repr(ctx) == "FunctionCtx(saved_tensors=0, needs_input_grad=(True, False))"
    ctx.save_for_backward(_t([1.0]))
    assert repr(ctx) == "FunctionCtx(saved_tensors=1, needs_input_grad=(True, False))"


# --- the base class on its own ----------------------------------------------


def test_a_subclass_that_defines_no_forward_says_so_before_it_runs_anything():
    """Its own traceback, not one wrapped in a report that the forward raised."""

    class Empty(Function):
        pass

    with pytest.raises(TypeError, match="defines no forward"):
        Empty.apply(_t([1.0]))


def test_a_subclass_that_defines_no_backward_differentiates_by_composition():
    class Composed(Function):
        @staticmethod
        def forward(ctx, x, scale):
            return (x * scale).exp()

    x = _t([0.5, 1.0], requires_grad=True)
    out = Composed.apply(x, 2.0)
    np.testing.assert_allclose(out.numpy(), np.exp(np.array([0.5, 1.0]) * 2.0))

    out.sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), 2.0 * np.exp(np.array([0.5, 1.0]) * 2.0))


def test_a_backward_replaces_the_composed_gradient_rather_than_adding_to_it():
    """The forward runs with recording off once a backward exists.

    Recorded as well, the graph would hold two paths to the same output and
    `backward()` would return their sum -- here 3 + 3 rather than 3.
    """

    class Lies(Function):
        @staticmethod
        def forward(ctx, x):
            return x * 3.0

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output * 3.0

    x = _t([1.0, 2.0], requires_grad=True)
    Lies.apply(x).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), [3.0, 3.0])


def test_a_backward_may_be_any_function_of_the_gradient_not_only_the_true_one():
    """A straight-through estimator, which is why the mode exists."""

    class RoundStraightThrough(Function):
        @staticmethod
        def forward(ctx, x):
            return x.round()

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output

    x = _t([0.2, 1.7, -0.4], requires_grad=True)
    out = RoundStraightThrough.apply(x)
    np.testing.assert_allclose(out.numpy(), [0.0, 2.0, -0.0])
    out.sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), [1.0, 1.0, 1.0])


def test_the_module_exports_what_it_documents():
    assert mt.autograd.__all__ == ["Function", "FunctionCtx"]
    assert mt.autograd.Function is Function
    assert mt.autograd.FunctionCtx is FunctionCtx


# --- against finite differences ---------------------------------------------


def _numeric_gradient(f, tensor, step=1e-6):
    """`d f / d tensor`, one central difference per element."""

    base = tensor.numpy().copy()
    out = np.empty_like(base)
    flat = base.reshape(-1)
    for index in range(flat.size):
        original = flat[index]

        flat[index] = original + step
        high = float(f(_t(base.reshape(tensor.shape))).numpy())

        flat[index] = original - step
        low = float(f(_t(base.reshape(tensor.shape))).numpy())

        flat[index] = original
        out.reshape(-1)[index] = (high - low) / (2.0 * step)
    return out


@pytest.mark.parametrize("argument", [0, 1])
def test_the_backward_agrees_with_finite_differences(argument):
    values = [
        np.array([0.5, -1.25, 2.0]),
        np.array([1.5, 0.25, -0.75]),
    ]
    tensors = [_t(v, requires_grad=i == argument) for i, v in enumerate(values)]
    Mul.apply(*tensors).sum().backward()

    def scalar(candidate):
        pair = list(tensors)
        pair[argument] = candidate
        return Mul.apply(*pair).sum()

    expected = _numeric_gradient(scalar, tensors[argument])
    np.testing.assert_allclose(
        tensors[argument].grad.numpy(), expected, rtol=1e-6, atol=1e-8
    )
