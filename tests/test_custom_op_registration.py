# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Registering an operation from Python, which is what "extensible" has to mean.

The registry could already *run* an operation, but only ones written in Rust and
compiled in -- so a caller who wanted an operation the library did not have
still needed a Rust toolchain and a rebuild. `register_custom_op` takes Python
callables, and the operation it makes participates in autograd on the same terms
as a built-in one.

Two modes, and which one a caller gets is decided by whether they wrote a
backward. Without one, the forward is recorded and the operation differentiates
by composition. With one, the forward runs with recording off and the gradient
is whatever the backward says -- which is the only way to write a
straight-through estimator, or any op whose useful derivative is not its true
one.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import minitensor as mt

_NAMES = itertools.count()


def _t(values, requires_grad=False):
    return mt.Tensor(
        np.ascontiguousarray(np.asarray(values, dtype=np.float64)),
        dtype="float64",
        requires_grad=requires_grad,
    )


@pytest.fixture
def name():
    """A registry name no other test has used.

    The registry is process-wide and rejects a duplicate, which is the right
    behaviour and would otherwise make these tests depend on their order.
    """

    chosen = f"_test_op_{next(_NAMES)}"
    yield chosen
    if mt.is_custom_op_registered_py(chosen):
        mt.unregister_custom_op_py(chosen)


def _run(name, *inputs):
    return mt.execute_custom_op_py(name, list(inputs))


# --- without a backward: differentiable by composition -----------------------


def test_an_op_without_a_backward_differentiates_through_its_forward(name):
    mt.register_custom_op(name, lambda x: x * 3.0 + 1.0)

    x = _t([1.0, 2.0], requires_grad=True)
    out = _run(name, x)
    np.testing.assert_allclose(out.numpy(), [4.0, 7.0])

    out.sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), [3.0, 3.0])
    mt.clear_autograd_graph()


def test_a_composed_op_is_differentiable_all_the_way_down(name):
    # The forward is ordinary tensor code, so every op inside it records and
    # the chain rule runs through the lot.
    mt.register_custom_op(name, lambda x: (x * x).exp())

    x = _t([0.5, -1.5], requires_grad=True)
    _run(name, x).sum().backward()
    values = np.array([0.5, -1.5])
    np.testing.assert_allclose(
        x.grad.numpy(), 2 * values * np.exp(values**2), rtol=1e-12
    )
    mt.clear_autograd_graph()


# --- with a backward: the gradient is whatever it says -----------------------


def test_a_backward_replaces_the_gradient_the_forward_would_have_given(name):
    # A straight-through estimator: the forward is a step function, whose true
    # derivative is zero everywhere and useless, and the backward hands back the
    # identity instead. There is no way to write this without a custom gradient.
    def step(x):
        return mt.Tensor((x.numpy() > 0).astype(np.float64), dtype="float64")

    mt.register_custom_op(name, step, lambda grad, inputs, output: grad)

    x = _t([-1.0, 0.5, 2.0], requires_grad=True)
    out = _run(name, x)
    np.testing.assert_array_equal(out.numpy(), [0.0, 1.0, 1.0])

    out.sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), [1.0, 1.0, 1.0])
    mt.clear_autograd_graph()


def test_the_forward_is_not_recorded_when_a_backward_was_written(name):
    # Otherwise the graph would hold two paths to the same gradient and they
    # would add: the forward's own 3.0 plus the backward's 10.0.
    mt.register_custom_op(
        name,
        lambda x: x * 3.0,
        lambda grad, inputs, output: grad * 10.0,
    )

    x = _t([1.0], requires_grad=True)
    _run(name, x).sum().backward()
    assert x.grad.numpy()[0] == 10.0, "the forward's own gradient leaked in"
    mt.clear_autograd_graph()


def test_the_backward_receives_the_inputs_and_the_output_it_was_given(name):
    seen = {}

    def remember(grad, inputs, output):
        seen["grad"] = grad.numpy().copy()
        seen["inputs"] = [i.numpy().copy() for i in inputs]
        seen["output"] = output.numpy().copy()
        return grad

    mt.register_custom_op(name, lambda x: x * 2.0, remember)
    x = _t([3.0, -4.0], requires_grad=True)
    _run(name, x).sum().backward()
    mt.clear_autograd_graph()

    np.testing.assert_array_equal(seen["grad"], [1.0, 1.0])
    np.testing.assert_array_equal(seen["inputs"][0], [3.0, -4.0])
    np.testing.assert_array_equal(seen["output"], [6.0, -8.0])


# --- more than one input -----------------------------------------------------


def test_an_op_of_two_inputs_takes_a_gradient_for_each(name):
    mt.register_custom_op(
        name,
        lambda a, b: a * b,
        lambda grad, inputs, output: (grad * inputs[1], grad * inputs[0]),
        num_inputs=2,
    )

    a = _t([1.0, 2.0], requires_grad=True)
    b = _t([3.0, 4.0], requires_grad=True)
    _run(name, a, b).sum().backward()
    np.testing.assert_array_equal(a.grad.numpy(), [3.0, 4.0])
    np.testing.assert_array_equal(b.grad.numpy(), [1.0, 2.0])
    mt.clear_autograd_graph()


def test_none_means_an_input_takes_no_gradient(name):
    # A real answer, not an omission: an index operand has no gradient to take.
    mt.register_custom_op(
        name,
        lambda a, b: a * b,
        lambda grad, inputs, output: (grad * inputs[1], None),
        num_inputs=2,
    )

    a = _t([1.0, 2.0], requires_grad=True)
    b = _t([3.0, 4.0], requires_grad=True)
    _run(name, a, b).sum().backward()
    np.testing.assert_array_equal(a.grad.numpy(), [3.0, 4.0])
    assert b.grad is None
    mt.clear_autograd_graph()


def test_a_single_input_may_return_a_bare_tensor_or_a_sequence(name):
    second = f"{name}_seq"
    mt.register_custom_op(name, lambda x: x * 2.0, lambda g, i, o: g * 5.0)
    mt.register_custom_op(second, lambda x: x * 2.0, lambda g, i, o: (g * 5.0,))
    try:
        for op in (name, second):
            x = _t([1.0], requires_grad=True)
            _run(op, x).sum().backward()
            assert x.grad.numpy()[0] == 5.0, op
            mt.clear_autograd_graph()
    finally:
        mt.unregister_custom_op_py(second)


def test_the_wrong_number_of_gradients_is_reported(name):
    mt.register_custom_op(
        name,
        lambda a, b: a + b,
        lambda grad, inputs, output: (grad,),
        num_inputs=2,
    )
    a = _t([1.0], requires_grad=True)
    b = _t([2.0], requires_grad=True)
    with pytest.raises(Exception, match="1 gradients for 2 inputs"):
        _run(name, a, b).sum().backward()
    mt.clear_autograd_graph()


# --- the registry ------------------------------------------------------------


def test_a_registered_op_is_listed_and_can_be_removed(name):
    assert not mt.is_custom_op_registered_py(name)
    mt.register_custom_op(name, lambda x: x)
    assert mt.is_custom_op_registered_py(name)
    assert name in mt.list_custom_ops_py()

    mt.unregister_custom_op_py(name)
    assert not mt.is_custom_op_registered_py(name)
    assert name not in mt.list_custom_ops_py()


def test_registering_the_same_name_twice_is_refused(name):
    mt.register_custom_op(name, lambda x: x)
    with pytest.raises(Exception):
        mt.register_custom_op(name, lambda x: x * 2.0)


def test_the_declared_input_count_is_enforced(name):
    mt.register_custom_op(name, lambda a, b: a + b, num_inputs=2)
    x = _t([1.0])
    with pytest.raises(Exception, match="expects 2 inputs, got 1"):
        _run(name, x)


# --- what the errors say -----------------------------------------------------


def test_a_forward_that_raises_reports_its_own_message(name):
    def explode(x):
        raise ValueError("the fixture said no")

    mt.register_custom_op(name, explode)
    with pytest.raises(Exception, match="the fixture said no"):
        _run(name, _t([1.0]))


def test_a_forward_that_returns_the_wrong_thing_says_what_it_returned(name):
    mt.register_custom_op(name, lambda x: 5)
    with pytest.raises(Exception, match="returned int, not a Tensor"):
        _run(name, _t([1.0]))


def test_a_backward_that_raises_reports_its_own_message(name):
    def explode(grad, inputs, output):
        raise RuntimeError("no gradient today")

    mt.register_custom_op(name, lambda x: x * 2.0, explode)
    with pytest.raises(Exception, match="no gradient today"):
        _run(name, _t([1.0], requires_grad=True)).sum().backward()
    mt.clear_autograd_graph()


def test_a_gradient_of_the_wrong_shape_is_refused(name):
    # It would otherwise be accumulated into a buffer it does not fit, which is
    # the kind of mistake that shows up much later as a wrong answer.
    mt.register_custom_op(
        name, lambda x: x * 2.0, lambda grad, inputs, output: _t([1.0, 2.0, 3.0])
    )
    with pytest.raises(Exception, match="gradient of shape"):
        _run(name, _t([1.0, 2.0], requires_grad=True)).sum().backward()
    mt.clear_autograd_graph()


def test_a_gradient_of_the_wrong_dtype_is_refused(name):
    mt.register_custom_op(
        name,
        lambda x: x * 2.0,
        lambda grad, inputs, output: mt.Tensor(
            np.array([1.0, 2.0], dtype=np.float32), dtype="float32"
        ),
    )
    with pytest.raises(Exception, match="gradient for a"):
        _run(name, _t([1.0, 2.0], requires_grad=True)).sum().backward()
    mt.clear_autograd_graph()


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"forward": 42}, "must be callable"),
        ({"forward": lambda x: x, "backward": "nope"}, "must be callable"),
        ({"forward": lambda x: x, "num_inputs": 0}, "at least one input"),
    ],
)
def test_a_registration_that_could_not_work_is_refused_up_front(name, kwargs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        mt.register_custom_op(name, **kwargs)


# --- it composes with the rest of the library --------------------------------


def test_a_custom_op_can_sit_in_the_middle_of_an_ordinary_graph(name):
    mt.register_custom_op(name, lambda x: x * x, lambda g, i, o: g * 2.0 * i[0])

    x = _t([2.0, 3.0], requires_grad=True)
    # sum(exp(custom(x)) * 0.5), whose gradient is exp(x^2) * x.
    loss = (_run(name, x).exp() * 0.5).sum()
    loss.backward()
    values = np.array([2.0, 3.0])
    np.testing.assert_allclose(x.grad.numpy(), np.exp(values**2) * values, rtol=1e-10)
    mt.clear_autograd_graph()


def test_a_custom_op_trains_a_parameter(name):
    # The end of the road for "extensible": a user's own operation moving a
    # weight through an optimizer, with no Rust involved anywhere.
    mt.register_custom_op(name, lambda w: w * w, lambda g, i, o: g * 2.0 * i[0])

    w = _t([3.0], requires_grad=True)
    optimizer = mt.optim.SGD([w], lr=0.1)
    for _ in range(20):
        optimizer.zero_grad()
        _run(name, w).sum().backward()
        optimizer.step()
    mt.clear_autograd_graph()

    # Minimising w^2 from 3.0 walks it towards zero.
    assert abs(w.numpy()[0]) < 0.1, w.numpy()
