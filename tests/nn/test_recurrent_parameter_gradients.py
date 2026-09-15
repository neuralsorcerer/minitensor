# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""LSTM and GRU parameter gradients, against central differences.

`test_recurrent` already checks that every parameter comes back with a gradient
that is neither `None` nor all zero, and that the gradient *of the input*
matches central differences. What neither reaches is the value of the parameter
gradients themselves -- and those are what a recurrent layer is hard to get
right about.

A parameter here is used once per timestep, so its gradient is a sum over the
whole sequence. Dropping one step's contribution, or counting the last one
twice, leaves a gradient that is non-`None`, non-zero, correctly shaped and
wrong. Stacking layers and running bidirectionally add more of the same: each
direction and each layer contributes its own term to the sum.

`mt.gradcheck` cannot be used here because it perturbs tensors passed as
arguments, and there is no functional form of these layers -- the parameters
live inside the module. So the perturbation is done in place through `copy_`,
which writes through to the module's own storage.

`test_the_check_would_notice_a_wrong_gradient` keeps this file honest: it scales
the analytic gradient by 1.001 and requires the comparison to fail. Without it a
harness that quietly compared something to itself would pass every case here and
mean nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

EPS = 1e-6
ATOL = 1e-6
RTOL = 1e-5


def _build(kind, **kwargs):
    with mt.default_dtype("float64"):
        return getattr(mt.nn, kind)(2, 3, **kwargs)


def _worst_excess(layer, steps=3, batch=1, corrupt=1.0, seed=11):
    """Largest amount by which any parameter gradient exceeds tolerance.

    Negative means every element is inside it. `corrupt` scales the analytic
    gradient, so 1.0 asks the real question and anything else asks whether this
    function can still answer it.
    """
    rng = np.random.default_rng(seed)
    inputs = mt.Tensor(rng.standard_normal((steps, batch, 2)), dtype="float64")
    # A random probe rather than a plain sum: summing the output makes many
    # different wrong gradients coincide with the right one.
    probe = mt.Tensor(
        rng.standard_normal((steps, batch, layer_width(layer))), dtype="float64"
    )

    def loss():
        out = layer(inputs)
        hidden = out[0] if isinstance(out, tuple) else out
        return (hidden * probe).sum()

    mt.clear_autograd_graph()
    for parameter in layer.parameters():
        parameter.zero_grad(True)
    loss().backward()
    analytic = [np.asarray(p.grad).copy() * corrupt for p in layer.parameters()]
    mt.clear_autograd_graph()

    worst = -np.inf
    for index, parameter in enumerate(layer.parameters()):
        base = np.asarray(parameter).copy()
        numeric = np.zeros_like(base)
        for position in np.ndindex(base.shape):
            values = {}
            for sign in (+1, -1):
                perturbed = base.copy()
                perturbed[position] += sign * EPS
                parameter.copy_(mt.Tensor(perturbed, dtype="float64"))
                with mt.no_grad():
                    values[sign] = float(np.asarray(loss()))
                mt.clear_autograd_graph()
            numeric[position] = (values[1] - values[-1]) / (2 * EPS)
        parameter.copy_(mt.Tensor(base, dtype="float64"))
        allowed = ATOL + RTOL * np.abs(numeric)
        worst = max(worst, float((np.abs(analytic[index] - numeric) - allowed).max()))
    mt.clear_autograd_graph()
    return worst


def layer_width(layer):
    """Output width, doubled when the layer runs in both directions."""
    return 3 * (2 if getattr(layer, "bidirectional", False) else 1)


CONFIGURATIONS = [
    ("LSTM", {}),
    ("GRU", {}),
    ("LSTM", {"num_layers": 2}),
    ("GRU", {"num_layers": 2}),
    ("LSTM", {"bidirectional": True}),
    ("GRU", {"bidirectional": True}),
    ("LSTM", {"bias": False}),
    ("GRU", {"bias": False}),
]


@pytest.fixture(autouse=True)
def _clean_graph():
    yield
    mt.clear_autograd_graph()


@pytest.mark.parametrize(
    "kind,options",
    CONFIGURATIONS,
    ids=[
        f"{kind}{''.join('_' + k + str(v) for k, v in opts.items())}"
        for kind, opts in CONFIGURATIONS
    ],
)
def test_parameter_gradients_match_central_differences(kind, options):
    excess = _worst_excess(_build(kind, **options))
    assert (
        excess <= 0.0
    ), f"{kind} {options}: a parameter gradient is outside tolerance by {excess:.3e}"


@pytest.mark.parametrize("kind", ["LSTM", "GRU"])
def test_the_check_would_notice_a_wrong_gradient(kind):
    """A gradient wrong by one part in a thousand must fail the comparison.

    This is what stops the rest of the file from being a test that compares
    something to itself.
    """
    excess = _worst_excess(_build(kind), corrupt=1.001)
    assert excess > 0.0, "a 0.1% error passed; the comparison proves nothing"
