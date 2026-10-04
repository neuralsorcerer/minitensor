# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Every optimizer's update rule, step by step, against the formula it implements.

A wrong update rule trains anyway -- just worse -- so nothing else here would
notice one. Each optimizer runs six steps on a float64 problem whose gradient
is not constant, and the parameters are compared with the same steps written
out in NumPy.
"""

import numpy as np
import pytest

import minitensor as mt
import minitensor.optim as optim

rng = np.random.default_rng(0)
A = rng.standard_normal((4,))
W0 = rng.standard_normal((4,))


def grad(w):
    return 2 * (w - A) * np.array([1.0, 2.0, 3.0, 0.5]) + np.sin(
        w
    )  # not constant, so momentum and averages matter


def loss_t(w):
    a = mt.tensor(A, dtype="float64")
    s = mt.tensor([1.0, 2.0, 3.0, 0.5], dtype="float64")
    return (((w - a) ** 2) * s).sum() - w.cos().sum()


STEPS = 6


def run_mt(cls, **kw):
    w = mt.tensor(W0.copy(), dtype="float64", requires_grad=True)
    opt = cls([w], **kw)
    for _ in range(STEPS):
        opt.zero_grad()
        l = loss_t(w)
        l.backward()
        opt.step()
    return w.numpy()


def sgd(lr, momentum=0, dampening=0, weight_decay=0, nesterov=False):
    w = W0.copy()
    b = None
    for t in range(STEPS):
        g = grad(w) + weight_decay * w
        if momentum:
            b = g.copy() if b is None else momentum * b + (1 - dampening) * g
            g = g + momentum * b if nesterov else b
        w = w - lr * g
    return w


def adam(
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0,
    amsgrad=False,
    decoupled=False,
):
    w = W0.copy()
    m = np.zeros(4)
    v = np.zeros(4)
    vmax = np.zeros(4)
    for t in range(1, STEPS + 1):
        g = grad(w)
        if decoupled:
            w = w * (1 - lr * weight_decay)
        else:
            g = g + weight_decay * w
        m = betas[0] * m + (1 - betas[0]) * g
        v = betas[1] * v + (1 - betas[1]) * g * g
        mh = m / (1 - betas[0] ** t)
        if amsgrad:
            vmax = np.maximum(vmax, v)
            vh = vmax / (1 - betas[1] ** t)
        else:
            vh = v / (1 - betas[1] ** t)
        w = w - lr * mh / (np.sqrt(vh) + eps)
    return w


def adamax(lr=0.002, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
    w = W0.copy()
    m = np.zeros(4)
    u = np.zeros(4)
    for t in range(1, STEPS + 1):
        g = grad(w) + weight_decay * w
        m = betas[0] * m + (1 - betas[0]) * g
        u = np.maximum(betas[1] * u, np.abs(g) + eps)
        w = w - lr / (1 - betas[0] ** t) * m / u
    return w


def nadam(
    lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0, momentum_decay=0.004
):
    w = W0.copy()
    m = np.zeros(4)
    v = np.zeros(4)
    mu_prod = 1.0
    for t in range(1, STEPS + 1):
        g = grad(w) + weight_decay * w
        mu = beta1 * (1 - 0.5 * 0.96 ** (t * momentum_decay))
        mu_next = beta1 * (1 - 0.5 * 0.96 ** ((t + 1) * momentum_decay))
        mu_prod *= mu
        mu_prod_next = mu_prod * mu_next
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        denom = np.sqrt(v / (1 - beta2**t)) + eps
        w = (
            w
            - lr * (1 - mu) / (1 - mu_prod) * g / denom
            - lr * mu_next / (1 - mu_prod_next) * m / denom
        )
    return w


def radam(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
    w = W0.copy()
    m = np.zeros(4)
    v = np.zeros(4)
    rinf = 2 / (1 - betas[1]) - 1
    for t in range(1, STEPS + 1):
        g = grad(w) + weight_decay * w
        m = betas[0] * m + (1 - betas[0]) * g
        v = betas[1] * v + (1 - betas[1]) * g * g
        mh = m / (1 - betas[0] ** t)
        rt = rinf - 2 * t * betas[1] ** t / (1 - betas[1] ** t)
        if rt > 5:
            l = np.sqrt(1 - betas[1] ** t) / (np.sqrt(v) + eps)
            r = np.sqrt((rt - 4) * (rt - 2) * rinf / ((rinf - 4) * (rinf - 2) * rt))
            w = w - lr * mh * r * l
        else:
            w = w - lr * mh
    return w


def rmsprop(lr, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
    w = W0.copy()
    sq = np.zeros(4)
    ga = np.zeros(4)
    b = np.zeros(4)
    for t in range(STEPS):
        g = grad(w) + weight_decay * w
        sq = alpha * sq + (1 - alpha) * g * g
        avg = sq - (ga := alpha * ga + (1 - alpha) * g) ** 2 if centered else sq
        avg = np.sqrt(avg) + eps
        if momentum:
            b = momentum * b + g / avg
            w = w - lr * b
        else:
            w = w - lr * g / avg
    return w


def adagrad(
    lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10
):
    w = W0.copy()
    s = np.full(4, float(initial_accumulator_value))
    for t in range(1, STEPS + 1):
        g = grad(w) + weight_decay * w
        clr = lr / (1 + (t - 1) * lr_decay)
        s = s + g * g
        w = w - clr * g / (np.sqrt(s) + eps)
    return w


def adadelta(lr=1.0, rho=0.9, eps=1e-6, weight_decay=0):
    w = W0.copy()
    sq = np.zeros(4)
    acc = np.zeros(4)
    for t in range(STEPS):
        g = grad(w) + weight_decay * w
        sq = rho * sq + (1 - rho) * g * g
        d = np.sqrt(acc + eps) / np.sqrt(sq + eps) * g
        acc = rho * acc + (1 - rho) * d * d
        w = w - lr * d
    return w


def lion(lr=1e-4, betas=(0.9, 0.99), weight_decay=0):
    w = W0.copy()
    m = np.zeros(4)
    for t in range(STEPS):
        g = grad(w)
        w = w * (1 - lr * weight_decay)
        u = np.sign(betas[0] * m + (1 - betas[0]) * g)
        w = w - lr * u
        m = betas[1] * m + (1 - betas[1]) * g
    return w


def rprop(lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-6, 50.0)):
    w = W0.copy()
    prev = np.zeros(4)
    step = np.full(4, lr)
    for t in range(STEPS):
        g = grad(w)
        s = g * prev
        step = np.where(
            s > 0,
            np.minimum(step * etas[1], step_sizes[1]),
            np.where(s < 0, np.maximum(step * etas[0], step_sizes[0]), step),
        )
        g = np.where(s < 0, 0, g)
        w = w - np.sign(g) * step
        prev = g
    return w


CASES = [
    ("SGD", optim.SGD, sgd, dict(lr=0.05)),
    ("SGD momentum", optim.SGD, sgd, dict(lr=0.05, momentum=0.9)),
    (
        "SGD dampened",
        optim.SGD,
        sgd,
        dict(lr=0.05, momentum=0.9, dampening=0.3, weight_decay=0.1),
    ),
    ("SGD nesterov", optim.SGD, sgd, dict(lr=0.05, momentum=0.9, nesterov=True)),
    ("Adam", optim.Adam, adam, dict(lr=0.1)),
    ("Adam decay", optim.Adam, adam, dict(lr=0.1, weight_decay=0.1)),
    ("Adam amsgrad", optim.Adam, adam, dict(lr=0.1, amsgrad=True)),
    (
        "AdamW",
        optim.AdamW,
        lambda **k: adam(decoupled=True, **k),
        dict(lr=0.1, weight_decay=0.1),
    ),
    ("Adamax", optim.Adamax, adamax, dict(lr=0.1, weight_decay=0.05)),
    ("NAdam", optim.NAdam, nadam, dict(lr=0.1)),
    ("NAdam decay", optim.NAdam, nadam, dict(lr=0.1, weight_decay=0.1)),
    # `eps` goes in before the bias correction here, so it is negligible for
    # the comparison rather than placed differently.
    ("RAdam", optim.RAdam, radam, dict(lr=0.1, eps=1e-300)),
    ("RMSprop", optim.RMSprop, rmsprop, dict(lr=0.05)),
    (
        "RMSprop centered",
        optim.RMSprop,
        rmsprop,
        dict(lr=0.05, momentum=0.9, centered=True, weight_decay=0.1),
    ),
    (
        "Adagrad",
        optim.Adagrad,
        adagrad,
        dict(lr=0.1, lr_decay=0.1, weight_decay=0.1, initial_accumulator_value=0.5),
    ),
    ("Adadelta", optim.Adadelta, adadelta, dict(lr=1.0, weight_decay=0.1)),
    ("Lion", optim.Lion, lion, dict(lr=0.05, weight_decay=0.1)),
    ("Rprop", optim.Rprop, rprop, dict(lr=0.05)),
]


@pytest.mark.parametrize("name,cls,reference,kwargs", CASES, ids=[c[0] for c in CASES])
def test_update_rule_matches_its_formula(name, cls, reference, kwargs):
    got = run_mt(cls, **kwargs)
    expected = reference(**kwargs)
    assert np.allclose(got, expected, rtol=1e-9, atol=1e-12), name
