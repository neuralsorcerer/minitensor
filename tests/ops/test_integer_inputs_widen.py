# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""An op whose value is a real number takes an integer by widening it.

`ops::util::widen_integer_input` states the rule and does the work: `sqrt(4)`
is 2 and `sin(1)` is 0.841..., neither has an integer answer, so the argument
widens -- `int32` to `float32`, `int64` to `float64`, the widths `mean`
documents -- rather than being refused.

Sixty-two ops followed it and twenty-four did not, and the split ran through
the middle of families rather than between them: `sigmoid` widened and
`softmax` refused, `selu` widened and `elu` refused, `mean` widened and `std`
refused, `cov` widened and the variance it is built on refused, `rand_like`
widened and `he_normal_like` refused. Nothing separated the two groups except
which of them had been written first.

The line that does exist is between elementwise maths and *layer-shaped* ops.
`conv1d`, `max_pool2d`, `layer_norm`, `batch_norm`, the losses and attention
all refuse an integer, uniformly, and they stay that way: they take model
activations, which are float by construction, and no member of those families
widens. `matmul` is not one of them -- an integer matrix product is exact and
meaningful, and it stays integer.

This sweep reads the surface rather than a list, so an op added later that
refuses what its siblings accept fails here.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import minitensor as mt

# The widths the rule names, and the ops' own reference values computed in
# float64 by NumPy where there is one to compare against.
WIDENS_TO = {"int32": "float32", "int64": "float64"}


def _ints(dtype):
    return mt.Tensor([1, 2, 3, 4], dtype=dtype)


def _floats():
    return mt.Tensor([1.0, 2.0, 3.0, 4.0], dtype="float64")


# Ops that legitimately keep an integer answer: every value they report was
# already in the input, or is a count, an index or a rounding of one.
_INTEGER_VALUED = {
    "abs",
    "absolute",
    "all",
    "amax",
    "amin",
    "any",
    "argmax",
    "argmin",
    "argsort",
    "argwhere",
    "as_tensor",
    "block",
    "ceil",
    "clamp",
    "clip",
    "combinations",
    "conj",
    "count_nonzero",
    "cumprod",
    "cumsum",
    "cumulative_sum",
    "diag",
    "diag_embed",
    "diagflat",
    "diff",
    "ediff1d",
    "empty_like",
    "flatnonzero",
    "flatten",
    "flipud",
    "floor",
    "frac",
    "hardtanh",
    "imag",
    "isfinite",
    "isinf",
    "isnan",
    "isneginf",
    "isposinf",
    "isreal",
    "lexsort",
    "logical_not",
    "max",
    "median",
    "min",
    "msort",
    "nan_to_num",
    "nanamax",
    "nanamin",
    "nanargmax",
    "nanargmin",
    "nancumprod",
    "nancumsum",
    "nanmax",
    "nanmedian",
    "nanmin",
    "nanprod",
    "nansum",
    "neg",
    "negative",
    "nonzero",
    "ones_like",
    "positive",
    "prod",
    "ptp",
    "ravel",
    "real",
    "relu",
    "relu6",
    "round",
    "sgn",
    "sign",
    "signbit",
    "square",
    "squeeze",
    "sum",
    "t",
    "trim_zeros",
    "trunc",
    "unique",
    "unique_consecutive",
    "unique_values",
    "vander",
    "zeros_like",
    # These report values from the input alongside their positions, so both
    # halves are integer when the input is.
    "cummax",
    "cummin",
    "mode",
    "sort",
    "unbind",
    "unstack",
    "unique_all",
    "unique_counts",
    "unique_inverse",
    # A mantissa and an exponent: float and integer by definition, in one
    # result, which is why it is named here rather than swept.
    "frexp",
}

# Takes probabilities, not values. An integer tensor of them is almost always
# a mistake, PyTorch refuses it too, and the message says so.
_REFUSES = {"bernoulli"}


def _only_tensors(result):
    pieces = result if isinstance(result, tuple) else (result,)
    return bool(pieces) and all(isinstance(piece, mt.Tensor) for piece in pieces)


def _single_argument_ops():
    found = {}
    for name in sorted(dir(mt)):
        if name.startswith("_"):
            continue
        function = getattr(mt, name)
        if not callable(function) or inspect.isclass(function):
            continue
        try:
            parameters = list(inspect.signature(function).parameters.values())
        except (ValueError, TypeError):
            continue
        required = [
            parameter
            for parameter in parameters
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        ]
        if len(required) != 1:
            continue
        found[name] = function
    return found


def test_a_real_valued_op_takes_an_integer_by_widening_it():
    mt.manual_seed(20260920)
    refused, widened = [], 0

    for name, function in _single_argument_ops().items():
        if name in _INTEGER_VALUED or name in _REFUSES:
            continue
        try:
            from_floats = function(_floats())
        except Exception:  # noqa: BLE001 - not a float op at all
            continue
        if not _only_tensors(from_floats):
            continue  # not a tensor op: `set_grad_enabled` and friends
        for dtype, target in WIDENS_TO.items():
            try:
                result = function(_ints(dtype))
            except Exception as error:  # noqa: BLE001 - reported below
                refused.append(f"{name}({dtype}): {type(error).__name__}: {error}")
                continue
            for piece in result if isinstance(result, tuple) else (result,):
                if "float" not in str(piece.dtype):
                    refused.append(f"{name}({dtype}) came back as {piece.dtype}")
            widened += 1

    assert not refused, "\n".join(refused)
    assert widened > 100, f"only {widened} calls reached -- the sweep broke"


@pytest.mark.parametrize(
    "name,call,reference",
    [
        ("std", lambda x: mt.std(x), lambda a: np.std(a, ddof=1)),
        ("var", lambda x: mt.var(x), lambda a: np.var(a, ddof=1)),
        ("nanvar", lambda x: mt.nanvar(x), lambda a: np.var(a, ddof=1)),
        ("nanstd", lambda x: mt.nanstd(x), lambda a: np.std(a, ddof=1)),
        ("norm", lambda x: mt.norm(x), np.linalg.norm),
        ("logsumexp", lambda x: mt.logsumexp(x, 0), lambda a: np.log(np.exp(a).sum())),
        ("quantile", lambda x: mt.quantile(x, 0.4), lambda a: np.quantile(a, 0.4)),
        (
            "nanquantile",
            lambda x: mt.nanquantile(x, 0.4),
            lambda a: np.nanquantile(a, 0.4),
        ),
        (
            "percentile",
            lambda x: mt.percentile(x, 40.0),
            lambda a: np.percentile(a, 40.0),
        ),
        ("softmax", lambda x: mt.softmax(x, 0), lambda a: np.exp(a) / np.exp(a).sum()),
        (
            "log_softmax",
            lambda x: mt.log_softmax(x, 0),
            lambda a: a - np.log(np.exp(a).sum()),
        ),
        ("sigmoid", lambda x: mt.sigmoid(x), lambda a: 1 / (1 + np.exp(-a))),
        ("softplus", lambda x: mt.softplus(x), lambda a: np.log1p(np.exp(a))),
        (
            "leaky_relu",
            lambda x: mt.leaky_relu(x),
            lambda a: np.where(a > 0, a, 0.01 * a),
        ),
        ("elu", lambda x: mt.elu(x), lambda a: np.where(a > 0, a, np.expm1(a))),
        (
            "hardshrink",
            lambda x: mt.hardshrink(x),
            lambda a: np.where(np.abs(a) > 0.5, a, 0.0),
        ),
        (
            "logcumsumexp",
            lambda x: mt.logcumsumexp(x, 0),
            lambda a: np.log(np.cumsum(np.exp(a))),
        ),
        ("mean", lambda x: mt.mean(x), np.mean),
    ],
)
@pytest.mark.parametrize("dtype", sorted(WIDENS_TO))
def test_the_widened_answer_is_the_float_answer(name, call, reference, dtype):
    """Widening is only right if the values come out where the float
    computation puts them, at the width the rule names."""
    got = call(_ints(dtype))
    assert str(got.dtype) == WIDENS_TO[dtype], name

    want = reference(np.array([1.0, 2.0, 3.0, 4.0]))
    np.testing.assert_allclose(
        np.asarray(got.numpy(), dtype=np.float64), want, rtol=1e-6, atol=1e-7
    )


def test_a_layer_shaped_op_still_refuses_an_integer():
    """The other side of the line, checked so it stays a line: these take model
    activations, and every member of each family refuses together."""
    image = mt.Tensor(np.arange(16).reshape(1, 1, 4, 4).tolist(), dtype="int64")
    rows = mt.Tensor([[1, 2, 3]], dtype="int64")

    for name, call in (
        ("max_pool2d", lambda: mt.functional.max_pool2d(image, 2)),
        ("avg_pool2d", lambda: mt.functional.avg_pool2d(image, 2)),
        ("layer_norm", lambda: mt.functional.layer_norm(rows, [3])),
        ("rms_norm", lambda: mt.rms_norm(rows, [3])),
        ("mse_loss", lambda: mt.functional.mse_loss(rows, rows)),
        (
            "attention",
            lambda: mt.functional.scaled_dot_product_attention(rows, rows, rows),
        ),
    ):
        with pytest.raises(Exception, match="float"):
            call()

    # `matmul` is not one of them: an integer product is exact and stays one.
    assert str(mt.matmul(rows, rows.transpose(0, 1)).dtype) == "int64"


def test_an_initializer_takes_its_shape_from_an_integer_reference():
    """`rand_like` and `randn_like` fall back to the default float dtype for a
    non-float reference; the six fan initializers answered "only supports
    float32 or float64" for the same argument."""
    reference = _ints("int64")
    for name in (
        "he_normal_like",
        "he_uniform_like",
        "lecun_normal_like",
        "lecun_uniform_like",
        "xavier_normal_like",
        "xavier_uniform_like",
    ):
        built = getattr(mt, name)(reference.reshape([2, 2]))
        assert str(built.dtype) == str(mt.rand_like(reference).dtype), name
        assert list(built.shape) == [2, 2]

    # An explicit integer dtype is still a request an initializer cannot fill.
    with pytest.raises(ValueError, match="float32 or float64"):
        mt.he_normal([2, 2], dtype="int32")


def test_bool_is_left_alone():
    """The helper says so: what a mask should mean to a reduction is a question
    this library has declined to answer, and answering it for `sqrt` alone
    would be worse than not answering it."""
    flags = mt.Tensor([True, False, True], dtype="bool")
    for call in (
        lambda: mt.sqrt(flags),
        lambda: mt.softmax(flags, 0),
        lambda: mt.var(flags),
        lambda: mt.norm(flags),
    ):
        with pytest.raises(Exception):
            call()
