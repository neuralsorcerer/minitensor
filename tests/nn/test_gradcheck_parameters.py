# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The gradients a layer's *parameters* get, checked against finite differences.

Every finite-difference sweep in this suite perturbs one tensor: the sweeps in
`test_gradcheck_differential` are built as `fn(x)`, and where an operation takes
more than one operand the rest are module-level constants. So
`scaled_dot_product_attention` is checked by perturbing the query while the key
and value sit fixed, and nothing has ever perturbed a weight or a bias.

That is the half that training consumes. An optimizer never touches the input
gradient; it moves parameters, using exactly the gradients no check reached.
A wrong weight gradient is also the hardest kind to notice from outside -- the
forward is right, the loss goes down, and the model just ends up somewhere
slightly wrong.

So each case here fixes the input and perturbs the parameters.

One trap worth naming, because an earlier draft of this file fell into it and
reported seven confident failures: every operand except the one being perturbed
has to be built **once**, outside the callable. Building it inside means
`gradcheck`'s 2N forwards each see different data, and the difference between
two unrelated forwards divided by `2 * eps` is a number in the millions that
looks like a catastrophic gradient bug and is nothing but noise.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

F = mt.functional

_RNG = np.random.default_rng(3)


def _t(values, requires_grad=True):
    return mt.Tensor(
        np.ascontiguousarray(np.asarray(values, np.float64)),
        dtype="float64",
        requires_grad=requires_grad,
    )


def _fixed(shape):
    """An operand held constant across every perturbed forward."""
    return _t(_RNG.standard_normal(shape), requires_grad=False)


# Fixed inputs, built once at import. See the note above on why this matters.
_QUERY = _fixed((1, 2, 3, 4))
_ROWS = _fixed((2, 3))
_FEATURES = _fixed((2, 4))
_SIGNAL_1D = _fixed((1, 2, 6))
_IMAGE = _fixed((1, 2, 5, 5))
_CHANNELS = _fixed((1, 4, 3))
_INDICES = mt.Tensor([0, 2, 1], dtype="int64")


CASES = [
    (
        "attention_key_and_value",
        lambda key, value: mt.scaled_dot_product_attention(_QUERY, key, value).sum(),
        lambda: (
            _t(_RNG.standard_normal((1, 2, 3, 4))),
            _t(_RNG.standard_normal((1, 2, 3, 4))),
        ),
    ),
    (
        "dense_weight_and_bias",
        lambda weight, bias: F.dense_layer(_ROWS, weight, bias).sum(),
        lambda: (_t(_RNG.standard_normal((4, 3))), _t(_RNG.standard_normal(4))),
    ),
    (
        "layer_norm_weight_and_bias",
        lambda weight, bias: mt.layer_norm(_FEATURES, [4], weight, bias).sum(),
        lambda: (_t(_RNG.standard_normal(4)), _t(_RNG.standard_normal(4))),
    ),
    (
        "rms_norm_weight",
        lambda weight: mt.rms_norm(_FEATURES, [4], weight).sum(),
        lambda: (_t(_RNG.standard_normal(4)),),
    ),
    (
        "group_norm_weight_and_bias",
        lambda weight, bias: F.group_norm(_CHANNELS, 2, weight, bias).sum(),
        lambda: (_t(_RNG.standard_normal(4)), _t(_RNG.standard_normal(4))),
    ),
    (
        "batch_norm_weight_and_bias",
        lambda weight, bias: F.batch_norm(_ROWS, None, None, weight, bias, True).sum(),
        lambda: (_t(_RNG.standard_normal(3)), _t(_RNG.standard_normal(3))),
    ),
    (
        "embedding_weight",
        lambda weight: F.embedding(_INDICES, weight).sum(),
        lambda: (_t(_RNG.standard_normal((4, 3))),),
    ),
    (
        "conv1d_weight_and_bias",
        lambda weight, bias: F.conv1d(_SIGNAL_1D, weight, bias).sum(),
        lambda: (_t(_RNG.standard_normal((3, 2, 3))), _t(_RNG.standard_normal(3))),
    ),
    (
        "conv2d_weight_and_bias",
        lambda weight, bias: F.conv2d(_IMAGE, weight, bias).sum(),
        lambda: (
            _t(_RNG.standard_normal((3, 2, 3, 3))),
            _t(_RNG.standard_normal(3)),
        ),
    ),
]


@pytest.fixture(autouse=True)
def _clean_graph():
    yield
    mt.clear_autograd_graph()


def test_the_table_is_populated():
    """A refactor that empties the table must not leave a passing file."""
    assert len(CASES) >= 9, f"only {len(CASES)} parameter cases"


@pytest.mark.parametrize(
    "func,build", [(f, b) for _, f, b in CASES], ids=[name for name, _, _ in CASES]
)
def test_parameter_gradients_match_finite_differences(func, build):
    assert mt.gradcheck(func, build(), atol=1e-6, rtol=1e-5)


def test_the_input_gradient_is_still_checked_too():
    """The half that was already covered, kept alongside so the pair is visible."""
    query, key, value = (_t(_RNG.standard_normal((1, 2, 3, 4))) for _ in range(3))
    assert mt.gradcheck(
        lambda q, k, v: mt.scaled_dot_product_attention(q, k, v).sum(),
        (query, key, value),
        atol=1e-6,
        rtol=1e-5,
    )
