# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A layer and its free function are two spellings of one computation.

`nn.Conv2d` and `functional.conv2d`, `nn.LayerNorm` and
`functional.layer_norm` -- for most of `nn` there are two ways to ask for the
same thing, and nothing in the suite has ever checked that they answer the
same. Each side is tested against its own expectations, which is exactly the
arrangement in which a default drifts on one side and no test notices: an
`eps`, a `reduction`, a padding convention. The layer keeps passing its tests,
the function keeps passing its tests, and a model that mixes the two spellings
silently computes two different things.

The comparison here is only meaningful because the functional call is handed
the *module's own* parameters. Rebuilt ones would differ by initialisation and
prove nothing.

`GELU` is the one deliberate exception, and it is pinned here rather than
skipped: the layer computes the tanh approximation because it is measurably
quicker, while the free function computes the error function. Both spellings
agree once asked for the same form, which is the part that would otherwise be
free to drift.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

F = mt.functional

# Far tighter than any approximation difference -- the GELU gap this file
# documents is 5e-4, eight orders above this -- and loose enough that a kernel
# that reassociates a sum is not called a regression.
RTOL = 1e-12
ATOL = 1e-15


def _rng():
    return np.random.default_rng(20240607)


def _t(shape, rng):
    return mt.Tensor(rng.standard_normal(shape), dtype="float64")


def _params(module):
    """A loss carries no parameters and does not offer the accessor at all."""
    getter = getattr(module, "parameters", None)
    return list(getter()) if getter is not None else []


def _randomize(module, rng):
    """Give every parameter a value that actually changes the answer.

    Freshly built layers are not usable for this comparison. Every bias in the
    library initialises to exactly zero and `LayerNorm`'s weight to exactly
    one, so a function that ignored its bias, or applied the weight twice,
    produced output identical to the layer's and the comparison passed while
    proving nothing. Verified: dropping the bias from the `conv2d` call passed
    against an unrandomised module and fails against a randomised one.
    """
    for parameter in _params(module):
        shape = np.asarray(parameter).shape
        parameter.copy_(mt.Tensor(rng.standard_normal(shape), dtype="float64"))
    return module


def _bias(params):
    return params[1] if len(params) > 1 else None


# Each case builds a module, then calls the free function with that module's
# own parameters. `inputs` is a factory so no two cases share a tensor.
CASES = [
    (
        "relu",
        lambda: mt.nn.ReLU(),
        lambda rng: (_t((3, 5), rng),),
        lambda x, p: F.relu(x),
    ),
    (
        "sigmoid",
        lambda: mt.nn.Sigmoid(),
        lambda rng: (_t((3, 5), rng),),
        lambda x, p: F.sigmoid(x),
    ),
    (
        "tanh",
        lambda: mt.nn.Tanh(),
        lambda rng: (_t((3, 5), rng),),
        lambda x, p: F.tanh(x),
    ),
    ("elu", lambda: mt.nn.ELU(), lambda rng: (_t((3, 5), rng),), lambda x, p: F.elu(x)),
    (
        "softmax",
        lambda: mt.nn.Softmax(dim=1),
        lambda rng: (_t((3, 5), rng),),
        lambda x, p: F.softmax(x, 1),
    ),
    (
        "max_pool1d",
        lambda: mt.nn.MaxPool1d(kernel_size=2),
        lambda rng: (_t((2, 3, 8), rng),),
        lambda x, p: F.max_pool1d(x, 2),
    ),
    (
        "avg_pool1d",
        lambda: mt.nn.AvgPool1d(kernel_size=2),
        lambda rng: (_t((2, 3, 8), rng),),
        lambda x, p: F.avg_pool1d(x, 2),
    ),
    (
        "max_pool2d",
        lambda: mt.nn.MaxPool2d(kernel_size=2),
        lambda rng: (_t((2, 3, 8, 8), rng),),
        lambda x, p: F.max_pool2d(x, 2),
    ),
    (
        "avg_pool2d",
        lambda: mt.nn.AvgPool2d(kernel_size=2),
        lambda rng: (_t((2, 3, 8, 8), rng),),
        lambda x, p: F.avg_pool2d(x, 2),
    ),
    (
        "adaptive_avg_pool1d",
        lambda: mt.nn.AdaptiveAvgPool1d(output_size=4),
        lambda rng: (_t((2, 3, 8), rng),),
        lambda x, p: F.adaptive_avg_pool1d(x, 4),
    ),
    (
        "adaptive_max_pool1d",
        lambda: mt.nn.AdaptiveMaxPool1d(output_size=4),
        lambda rng: (_t((2, 3, 8), rng),),
        lambda x, p: F.adaptive_max_pool1d(x, 4),
    ),
    (
        "adaptive_avg_pool2d",
        lambda: mt.nn.AdaptiveAvgPool2d(output_size=4),
        lambda rng: (_t((2, 3, 8, 8), rng),),
        lambda x, p: F.adaptive_avg_pool2d(x, 4),
    ),
    (
        "adaptive_max_pool2d",
        lambda: mt.nn.AdaptiveMaxPool2d(output_size=4),
        lambda rng: (_t((2, 3, 8, 8), rng),),
        lambda x, p: F.adaptive_max_pool2d(x, 4),
    ),
    (
        "conv1d",
        lambda: mt.nn.Conv1d(3, 4, 3),
        lambda rng: (_t((2, 3, 8), rng),),
        lambda x, p: F.conv1d(x, p[0], _bias(p)),
    ),
    (
        "conv2d",
        lambda: mt.nn.Conv2d(3, 4, 3),
        lambda rng: (_t((2, 3, 8, 8), rng),),
        lambda x, p: F.conv2d(x, p[0], _bias(p)),
    ),
    (
        "conv_transpose1d",
        lambda: mt.nn.ConvTranspose1d(3, 4, 3),
        lambda rng: (_t((2, 3, 8), rng),),
        lambda x, p: F.conv_transpose1d(x, p[0], _bias(p)),
    ),
    (
        "conv_transpose2d",
        lambda: mt.nn.ConvTranspose2d(3, 4, 3),
        lambda rng: (_t((2, 3, 8, 8), rng),),
        lambda x, p: F.conv_transpose2d(x, p[0], _bias(p)),
    ),
    (
        "layer_norm",
        lambda: mt.nn.LayerNorm(5),
        lambda rng: (_t((3, 5), rng),),
        lambda x, p: F.layer_norm(x, [5], p[0], _bias(p)),
    ),
    (
        "dense_layer",
        lambda: mt.nn.DenseLayer(5, 4),
        lambda rng: (_t((3, 5), rng),),
        lambda x, p: F.dense_layer(x, p[0], _bias(p)),
    ),
    (
        "huber_loss",
        lambda: mt.nn.HuberLoss(),
        lambda rng: (_t((4, 3), rng), _t((4, 3), rng)),
        lambda xs, p: F.huber_loss(*xs),
    ),
    (
        "smooth_l1_loss",
        lambda: mt.nn.SmoothL1Loss(),
        lambda rng: (_t((4, 3), rng), _t((4, 3), rng)),
        lambda xs, p: F.smooth_l1_loss(*xs),
    ),
    (
        "log_cosh_loss",
        lambda: mt.nn.LogCoshLoss(),
        lambda rng: (_t((4, 3), rng), _t((4, 3), rng)),
        lambda xs, p: F.log_cosh_loss(*xs),
    ),
    (
        "focal_loss",
        lambda: mt.nn.FocalLoss(),
        lambda rng: (
            _t((4, 3), rng),
            mt.Tensor(rng.integers(0, 3, size=(4,)), dtype="int64"),
        ),
        lambda xs, p: F.focal_loss(*xs),
    ),
]


@pytest.fixture(autouse=True)
def _clean_graph():
    yield
    mt.clear_autograd_graph()


@pytest.mark.parametrize(
    "name,build,inputs,functional", CASES, ids=[c[0] for c in CASES]
)
def test_the_layer_and_the_free_function_compute_the_same_thing(
    name, build, inputs, functional
):
    rng = _rng()
    with mt.default_dtype("float64"):
        module = _randomize(build(), rng)
        args = inputs(rng)

        from_module = module(*args)
        # One argument stays one argument; the losses take two.
        from_function = functional(args[0] if len(args) == 1 else args, _params(module))

    a = np.asarray(from_module)
    b = np.asarray(from_function)
    assert (
        a.shape == b.shape
    ), f"{name}: {a.shape} from the layer, {b.shape} from the function"
    np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL, err_msg=f"{name}")


def test_embedding_matches_its_free_function():
    """Separately, because its input is indices rather than a float tensor."""
    rng = _rng()
    with mt.default_dtype("float64"):
        module = _randomize(mt.nn.Embedding(10, 4), rng)
        indices = mt.Tensor(rng.integers(0, 10, size=(3, 2)), dtype="int64")
        np.testing.assert_allclose(
            np.asarray(module(indices)),
            np.asarray(F.embedding(indices, _params(module)[0])),
            rtol=RTOL,
            atol=ATOL,
        )


def test_the_sweep_covers_the_pairs_that_exist():
    """A rename that drops a pair out of the table must not pass silently."""
    paired = set()
    functional_names = {n for n in dir(F) if not n.startswith("_")}
    for name in dir(mt.nn):
        if name.startswith("_") or not isinstance(getattr(mt.nn, name), type):
            continue
        snake = "".join("_" + c.lower() if c.isupper() else c for c in name).lstrip("_")
        for candidate in (name.lower(), snake):
            if candidate in functional_names:
                paired.add(candidate)
                break

    checked = {c[0] for c in CASES} | {"embedding"}
    # `gelu` is covered by test_gelu_layer.py, which pins the two spellings as
    # deliberately different functions. `dropout` and `dropout2d` are random.
    deliberately_absent = {"gelu", "dropout", "dropout2d"}
    missing = paired - checked - deliberately_absent
    assert not missing, f"layer/function pairs nothing compares: {sorted(missing)}"


def test_gelu_is_the_documented_exception_and_agrees_once_asked_the_same():
    """The defaults differ on purpose; matched forms must not.

    `test_gelu_layer.py` owns the claim that the two defaults are different
    functions. What belongs here is the other half: that the difference is
    exactly the approximation and nothing else, so neither spelling can drift
    while still looking correct on its own.
    """
    rng = _rng()
    with mt.default_dtype("float64"):
        x = _t((3, 5), rng)
        exact_layer = np.asarray(mt.nn.GELU(approximate="none")(x))
        tanh_layer = np.asarray(mt.nn.GELU()(x))

    np.testing.assert_allclose(exact_layer, np.asarray(F.gelu(x)), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        tanh_layer, np.asarray(F.gelu(x, "tanh")), rtol=RTOL, atol=ATOL
    )
    assert np.abs(exact_layer - tanh_layer).max() > 1e-5, (
        "the two defaults stopped differing; either the layer changed form or "
        "test_gelu_layer.py's premise is stale"
    )
