# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Adadelta, Adamax, RAdam and Rprop, past the reference-algorithm check.

`test_optim_reference_algorithms.py` compares all four step for step against a
NumPy transcription of their published algorithm boxes. What is left, and what
this file covers, is the behaviour each one is chosen *for* -- Adadelta's
freedom from a tuned learning rate, Adamax's geometric forgetting of a spike,
RAdam's built-in warmup, Rprop's indifference to gradient magnitude -- plus the
Python surface: constructor validation, the hyperparameter getters, the repr,
and a state round trip.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt
from minitensor import optim

FAMILY = [
    ("Adadelta", optim.Adadelta, {}),
    ("Adamax", optim.Adamax, {}),
    ("RAdam", optim.RAdam, {}),
    ("Rprop", optim.Rprop, {}),
]


def _param(values, requires_grad=True):
    return mt.Tensor(
        np.ascontiguousarray(np.asarray(values, dtype=np.float64)),
        dtype="float64",
        requires_grad=requires_grad,
    )


def _drive(optimizer, param, grads):
    """Step `optimizer` once per row of `grads`, returning the parameter after
    each step. The surrogate is linear, so its gradient is exactly the row."""

    trajectory = []
    for grad in grads:
        optimizer.zero_grad()
        (
            param * mt.Tensor(np.asarray(grad, dtype=np.float64), dtype="float64")
        ).sum().backward()
        optimizer.step()
        trajectory.append(param.numpy().copy())
    mt.clear_autograd_graph()
    return trajectory


# --- Adadelta ---------------------------------------------------------------


def test_adadelta_barely_notices_the_gradient_s_scale():
    # The claim the method exists for. A gradient-scaled optimizer would move a
    # million times further for the larger gradient; this one moves about four
    # times further, because its step is measured in the parameter's units.
    steps = []
    for magnitude in (1e-3, 1e3):
        param = _param([0.0])
        _drive(optim.Adadelta([param], lr=1.0), param, [[magnitude]])
        steps.append(abs(param.numpy()[0]))

    ratio = steps[1] / steps[0]
    assert 1.0 < ratio < 10.0, f"a million-fold gradient gave a {ratio}-fold step"


def test_adadelta_defaults_to_a_learning_rate_of_one():
    opt = optim.Adadelta([_param([1.0])])
    assert opt.lr == 1.0
    assert opt.rho == 0.9
    assert opt.epsilon == 1e-6
    assert opt.weight_decay == 0.0


def test_adadelta_reports_itself():
    opt = optim.Adadelta([_param([1.0])], lr=0.5, rho=0.8)
    assert repr(opt) == "Adadelta(lr=0.5, rho=0.8, eps=1e-6, weight_decay=0.0)"


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"lr": 0.0}, "Learning rate"),
        ({"rho": 1.0}, "rho"),
        ({"rho": -0.1}, "rho"),
        ({"eps": 0.0}, "Epsilon"),
        ({"weight_decay": -1.0}, "Weight decay"),
    ],
)
def test_adadelta_rejects_a_configuration_that_has_no_meaning(kwargs, message):
    with pytest.raises(ValueError, match=message):
        optim.Adadelta([_param([1.0])], **kwargs)


# --- Adamax -----------------------------------------------------------------


def test_adamax_lets_a_spike_decay_out_of_the_denominator_geometrically():
    # The infinity norm is a decaying maximum, so a single enormous gradient
    # leaves the denominator at exactly beta2 per step. Adam squares it into a
    # mean, which takes far longer to forget.
    beta2 = 0.9
    param = _param([0.0])
    # beta1 = 0 makes the numerator the gradient itself, so each step is
    # visibly `g / u`.
    opt = optim.Adamax([param], lr=1.0, beta1=0.0, beta2=beta2)
    trajectory = _drive(opt, param, [[100.0], [1.0], [1.0], [1.0]])

    previous = trajectory[0][0]
    for k, value in enumerate(trajectory[1:], start=1):
        moved = previous - value[0]
        # `eps` is 1e-8 against a denominator of 90 and below, so it moves the
        # answer by about a part in 1e10 and no further.
        assert moved == pytest.approx(1.0 / (100.0 * beta2**k), rel=1e-9)
        previous = value[0]


def test_adamax_defaults_match_the_paper():
    opt = optim.Adamax([_param([1.0])])
    assert opt.lr == 0.002
    assert (opt.beta1, opt.beta2) == (0.9, 0.999)
    assert opt.epsilon == 1e-8


def test_adamax_accepts_betas_as_a_tuple_or_as_two_arguments():
    tupled = optim.Adamax([_param([1.0])], betas=(0.8, 0.9))
    separate = optim.Adamax([_param([1.0])], beta1=0.8, beta2=0.9)
    assert (tupled.beta1, tupled.beta2) == (separate.beta1, separate.beta2)
    with pytest.raises(TypeError, match="not both"):
        optim.Adamax([_param([1.0])], betas=(0.8, 0.9), beta1=0.5)


def test_adamax_reports_itself():
    opt = optim.Adamax([_param([1.0])], lr=0.01)
    assert (
        repr(opt) == "Adamax(lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)"
    )


# --- RAdam ------------------------------------------------------------------


def test_radam_takes_a_plain_step_until_the_variance_is_measurable():
    # The warmup that falls out of the method rather than being scheduled:
    # while the second moment has fewer than five effective samples the step is
    # `lr * m_hat`, which with beta1 = 0 is exactly `lr * g`.
    lr = 0.1
    param = _param([0.0])
    opt = optim.RAdam([param], lr=lr, beta1=0.0, beta2=0.999)
    trajectory = _drive(opt, param, [[2.0]] * 3)

    for step, value in enumerate(trajectory, start=1):
        assert value[0] == pytest.approx(-lr * 2.0 * step, rel=1e-12)


def test_radam_becomes_adaptive_once_it_has_the_samples():
    # Past the threshold the step is scaled by the rectifier and divided by the
    # second moment, so it stops tracking the gradient's own scale.
    lr = 0.1
    param = _param([0.0])
    opt = optim.RAdam([param], lr=lr, beta1=0.0, beta2=0.999)
    trajectory = _drive(opt, param, [[2.0]] * 12)

    early = abs(trajectory[0][0])
    late = abs(trajectory[-1][0] - trajectory[-2][0])
    assert early == pytest.approx(lr * 2.0, rel=1e-12), "the plain step"
    assert late != pytest.approx(lr * 2.0, rel=1e-6), "the rectified one"


def test_radam_defaults_match_the_paper():
    opt = optim.RAdam([_param([1.0])])
    assert opt.lr == 0.001
    assert (opt.beta1, opt.beta2) == (0.9, 0.999)
    assert opt.epsilon == 1e-8


def test_radam_refuses_a_beta2_of_one():
    # The effective sample count is `2 / (1 - beta2) - 1`; at 1 there is no
    # such number to rectify against.
    with pytest.raises(ValueError, match="beta2"):
        optim.RAdam([_param([1.0])], betas=(0.9, 1.0))


def test_radam_reports_itself():
    opt = optim.RAdam([_param([1.0])], lr=0.002)
    assert (
        repr(opt) == "RAdam(lr=0.002, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)"
    )


# --- Rprop ------------------------------------------------------------------


def test_rprop_reads_only_the_sign_of_the_gradient():
    # Twelve orders of magnitude apart, agreeing in sign, identical paths.
    signs = [[1.0], [1.0], [-1.0], [-1.0], [1.0]]
    paths = []
    for scale in (1e-9, 1e3):
        param = _param([1.0])
        paths.append(
            _drive(optim.Rprop([param], lr=0.1), param, [[s[0] * scale] for s in signs])
        )
    np.testing.assert_array_equal(np.array(paths[0]), np.array(paths[1]))


def test_rprop_grows_an_agreeing_step_and_declines_to_take_a_reversed_one():
    lr, eta_minus, eta_plus = 0.1, 0.5, 1.2
    param = _param([0.0])
    opt = optim.Rprop([param], lr=lr, etas=(eta_minus, eta_plus))
    got = [v[0] for v in _drive(opt, param, [[1.0], [1.0], [-1.0], [-1.0]])]

    # Nothing to agree with yet, so the first step is `lr` itself.
    assert got[0] == pytest.approx(-lr, abs=1e-15)
    assert got[1] == pytest.approx(got[0] - lr * eta_plus, abs=1e-15)
    # A reversal shrinks the step and takes none of it.
    assert got[2] == pytest.approx(got[1], abs=1e-15)
    # It also forgets the gradient, so the next step has nothing to agree with
    # and the shrunk size stands.
    assert got[3] == pytest.approx(got[2] + lr * eta_plus * eta_minus, abs=1e-15)


def test_rprop_clamps_the_step_size_to_its_bounds():
    param = _param([0.0])
    opt = optim.Rprop([param], lr=0.1, etas=(0.5, 2.0), step_sizes=(1e-6, 0.15))
    got = [v[0] for v in _drive(opt, param, [[1.0]] * 4)]

    # 0.1, then 0.2 clamped to 0.15, and 0.15 thereafter.
    steps = [abs(b - a) for a, b in zip([0.0] + got, got)]
    assert steps[0] == pytest.approx(0.1)
    for step in steps[1:]:
        assert step == pytest.approx(0.15)


def test_rprop_defaults_match_the_paper():
    opt = optim.Rprop([_param([1.0])])
    assert opt.lr == 0.01
    assert (opt.eta_minus, opt.eta_plus) == (0.5, 1.2)
    assert (opt.step_min, opt.step_max) == (1e-6, 50.0)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"lr": -1.0}, "Learning rate"),
        ({"etas": (1.5, 1.2)}, r"etas\[0\]"),
        ({"etas": (0.5, 0.9)}, r"etas\[1\]"),
        ({"step_sizes": (0.0, 1.0)}, "step_sizes"),
        ({"step_sizes": (2.0, 1.0)}, "step_sizes"),
    ],
)
def test_rprop_rejects_a_configuration_that_has_no_meaning(kwargs, message):
    with pytest.raises(ValueError, match=message):
        optim.Rprop([_param([1.0])], **kwargs)


def test_rprop_reports_itself():
    opt = optim.Rprop([_param([1.0])], lr=0.05)
    assert repr(opt) == "Rprop(lr=0.05, etas=(0.5, 1.2), step_sizes=(1e-6, 50.0))"


# --- shared -----------------------------------------------------------------


@pytest.mark.parametrize("name,build,kwargs", FAMILY, ids=[f[0] for f in FAMILY])
def test_each_optimizer_is_an_optimizer_and_moves_the_parameter(name, build, kwargs):
    param = _param([1.0, -2.0, 0.5])
    opt = build([param], lr=0.1, **kwargs)
    assert isinstance(opt, optim.Optimizer)
    assert opt.step_count == 0

    before = param.numpy().copy()
    _drive(opt, param, [[0.5, -0.5, 0.25]] * 3)
    assert opt.step_count == 3
    assert not np.array_equal(param.numpy(), before)
    assert np.all(np.isfinite(param.numpy()))


@pytest.mark.parametrize("name,build,kwargs", FAMILY, ids=[f[0] for f in FAMILY])
def test_a_reloaded_optimizer_resumes_where_it_left_off(name, build, kwargs):
    grads = [[0.3, -0.7, 0.2], [0.1, 0.4, -0.6]]
    next_grad = [[0.5, 0.5, -0.5]]

    param = _param([1.0, -2.0, 0.5])
    opt = build([param], lr=0.1, **kwargs)
    _drive(opt, param, grads)
    state = opt.state_dict()
    assert state.step_count == 2

    continued = mt.Tensor(param.numpy().copy(), dtype="float64", requires_grad=True)
    _drive(opt, param, next_grad)

    fresh = build([continued], lr=0.1, **kwargs)
    fresh.load_state_dict(state)
    _drive(fresh, continued, next_grad)

    np.testing.assert_allclose(continued.numpy(), param.numpy(), rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("name,build,kwargs", FAMILY, ids=[f[0] for f in FAMILY])
def test_the_learning_rate_is_readable_and_writable(name, build, kwargs):
    opt = build([_param([1.0])], lr=0.1, **kwargs)
    assert opt.lr == 0.1
    opt.lr = 0.05
    assert opt.lr == 0.05
    assert "lr=0.05" in repr(opt)


@pytest.mark.parametrize("name,build,kwargs", FAMILY, ids=[f[0] for f in FAMILY])
def test_an_integer_parameter_is_refused_by_name(name, build, kwargs):
    param = mt.Tensor(np.array([1, 2], dtype=np.int64), dtype="int64").requires_grad_(
        True
    )
    opt = build([param], lr=0.1, **kwargs)
    with pytest.raises(Exception, match="floating point"):
        (
            param * mt.Tensor(np.array([1, 1], dtype=np.int64), dtype="int64")
        ).sum().backward()
        opt.step()
    mt.clear_autograd_graph()


@pytest.mark.parametrize("name,build,kwargs", FAMILY, ids=[f[0] for f in FAMILY])
def test_a_large_parameter_is_stepped_the_same_way_as_a_small_one(name, build, kwargs):
    # The update runs sequentially below the element threshold and across
    # rayon's workers above it; the two have to agree.
    long = 70_000
    wide = _param(np.ones(long))
    narrow = _param([1.0])
    wide_opt = build([wide], lr=0.1, **kwargs)
    narrow_opt = build([narrow], lr=0.1, **kwargs)

    for g in (0.5, -0.25, 0.75):
        _drive(wide_opt, wide, [np.full(long, g)])
        _drive(narrow_opt, narrow, [[g]])

    np.testing.assert_allclose(
        wide.numpy(), np.full(long, narrow.numpy()[0]), rtol=1e-12
    )
