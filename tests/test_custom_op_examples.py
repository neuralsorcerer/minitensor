# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The bundled example custom operations compute what they claim.

These are the worked examples someone writes their own operation from, and they
register into the process-wide registry under real names -- so a caller reaching
`execute_custom_op("gelu", ...)` gets GELU. Both properties failed before:
`gelu` computed `x * (1 + tanh(x))`, `mish` computed `x * tanh(x)`, `power`
computed `x * y`, `layer_norm` was the identity, and every backward returned
ones or passed the gradient straight through.
"""

import math

import numpy as np
import pytest

import minitensor as mt

EPS = 1e-5  # the eps the example layer_norm uses


@pytest.fixture(scope="module", autouse=True)
def _registered():
    mt.register_example_custom_ops()
    yield


def _t(array):
    return mt.Tensor(np.ascontiguousarray(array, dtype=np.float64), dtype="float64")


def _reference_layer_norm(x, weight, bias):
    mean = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mean) / np.sqrt(var + EPS) * weight + bias


def _sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


FORWARD_CASES = [
    ("swish", lambda x: x * _sigmoid(x)),
    ("gelu", lambda x: x * 0.5 * (1 + np.vectorize(math.erf)(x / np.sqrt(2)))),
    ("mish", lambda x: x * np.tanh(np.log1p(np.exp(x)))),
]


@pytest.mark.parametrize(
    "name,reference", FORWARD_CASES, ids=[c[0] for c in FORWARD_CASES]
)
def test_activation_examples_compute_their_named_function(name, reference):
    x = np.array([1.0, -2.0, 3.0, 0.5, -0.25, 0.0], dtype=np.float64)
    got = np.asarray(mt.execute_custom_op_py(name, [_t(x)]).numpy())
    np.testing.assert_allclose(got, reference(x), rtol=1e-12, atol=1e-13)


def test_activation_examples_agree_with_the_built_ins():
    # `gelu` and `swish` shadow real operations; the custom ones must not
    # disagree with the library's own.
    x = np.array([1.0, -2.0, 3.0, 0.5], dtype=np.float64)
    tensor = _t(x)
    np.testing.assert_allclose(
        np.asarray(mt.execute_custom_op_py("gelu", [tensor]).numpy()),
        np.asarray(tensor.gelu().numpy()),
        rtol=1e-12,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        np.asarray(mt.execute_custom_op_py("swish", [tensor]).numpy()),
        np.asarray(tensor.silu().numpy()),  # SiLU is Swish
        rtol=1e-12,
        atol=1e-13,
    )


def test_power_example_raises_rather_than_multiplies():
    base = np.array([2.0, 3.0, 0.5], dtype=np.float64)
    exponent = np.array([3.0, 2.0, 4.0], dtype=np.float64)
    got = np.asarray(mt.execute_custom_op_py("power", [_t(base), _t(exponent)]).numpy())
    np.testing.assert_allclose(got, base**exponent, rtol=1e-12)


def test_layer_norm_example_normalizes_rather_than_passing_through():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 4))
    weight = rng.standard_normal(4)
    bias = rng.standard_normal(4)
    got = np.asarray(
        mt.execute_custom_op_py("layer_norm", [_t(x), _t(weight), _t(bias)]).numpy()
    )
    np.testing.assert_allclose(
        got, _reference_layer_norm(x, weight, bias), rtol=1e-10, atol=1e-12
    )
    assert not np.allclose(got, x), "the example must not be the identity"


def _analytic(name, sources, which):
    mt.clear_autograd_graph()
    tensors = [
        mt.Tensor(
            np.ascontiguousarray(s, dtype=np.float64),
            dtype="float64",
            requires_grad=(i == which),
        )
        for i, s in enumerate(sources)
    ]
    mt.execute_custom_op_py(name, tensors).sum().backward()
    grad = tensors[which].grad
    result = None if grad is None else grad.numpy().copy()
    mt.clear_autograd_graph()
    return result


def _numeric(name, sources, which, eps=1e-6):
    base = [np.asarray(s, dtype=np.float64) for s in sources]
    flat = base[which].reshape(-1).copy()
    out = np.zeros_like(flat)
    for i in range(flat.size):
        values = {}
        for sign in (1, -1):
            shifted = flat.copy()
            shifted[i] += sign * eps
            args = list(base)
            args[which] = shifted.reshape(base[which].shape)
            values[sign] = (
                mt.execute_custom_op_py(name, [_t(a) for a in args]).sum().item()
            )
        out[i] = (values[1] - values[-1]) / (2 * eps)
    return out.reshape(base[which].shape)


_RNG = np.random.default_rng(19)
_X = _RNG.standard_normal(6)
# A positive base keeps d/dy x^y = x^y ln(x) finite.
_BASE = np.abs(_RNG.standard_normal(5)) + 0.4
_EXPONENT = _RNG.standard_normal(5)
_LN_X = _RNG.standard_normal((2, 4))
_LN_W = _RNG.standard_normal(4)
_LN_B = _RNG.standard_normal(4)

GRAD_CASES = [
    ("swish", [_X], 0),
    ("gelu", [_X], 0),
    ("mish", [_X], 0),
    ("power", [_BASE, _EXPONENT], 0),
    ("power", [_BASE, _EXPONENT], 1),
    ("layer_norm", [_LN_X, _LN_W, _LN_B], 0),
    ("layer_norm", [_LN_X, _LN_W, _LN_B], 1),
    ("layer_norm", [_LN_X, _LN_W, _LN_B], 2),
]


@pytest.mark.parametrize(
    "name,sources,which", GRAD_CASES, ids=[f"{n}[{w}]" for n, _, w in GRAD_CASES]
)
def test_example_backwards_match_finite_differences(name, sources, which):
    analytic = _analytic(name, sources, which)
    assert analytic is not None, f"{name} produced no gradient for input {which}"
    np.testing.assert_allclose(
        analytic, _numeric(name, sources, which), rtol=1e-5, atol=1e-7
    )


def test_layer_norm_backward_handles_a_rank_one_input():
    # No leading axes to reduce over, so the weight and bias gradients are the
    # per-element terms themselves.
    rng = np.random.default_rng(23)
    sources = [rng.standard_normal(4), rng.standard_normal(4), rng.standard_normal(4)]
    for which in range(3):
        analytic = _analytic("layer_norm", sources, which)
        assert analytic is not None
        np.testing.assert_allclose(
            analytic, _numeric("layer_norm", sources, which), rtol=1e-5, atol=1e-7
        )
