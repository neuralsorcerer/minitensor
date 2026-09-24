# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Every rejection is a normal exception, never a Rust panic.

A panic that reaches Python through pyo3 arrives as
``pyo3_runtime.PanicException``, which derives from ``BaseException`` rather
than ``Exception``. That makes it a poor way to refuse input three times over:
it slips past ``except Exception``, so a caller's error handling does not see
it; it carries a Rust backtrace where a message belongs; and it means the
library aborted an operation rather than declining one.

The suite already pins the *type* of specific rejections -- ``IndexError`` for
an out-of-range dimension, and so on -- at a couple of dozen named call sites.
This asserts the weaker property across a much wider surface: whatever an
operation does with input it dislikes, it does it as an ``Exception``.

The inputs are chosen to be awkward rather than random, since the shapes that
provoke a panic are the degenerate ones: empty tensors in both orientations,
0-d, the integer limits, NaN and infinity, dimensions that do not exist, and
sizes that do not divide.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt
from minitensor import nn, optim


def _assert_no_panic(label, call):
    """Run `call`; a raised `BaseException` that is not an `Exception` is a panic."""
    try:
        call()
    except Exception:  # noqa: BLE001 - declining the input is the expected outcome
        pass
    except BaseException as exc:  # pragma: no cover - the failure this exists for
        pytest.fail(
            f"{label} raised {type(exc).__name__}, which is not an Exception. "
            f"A pyo3 PanicException means the operation panicked rather than "
            f"returning an error: {str(exc).splitlines()[0][:200]}"
        )


def _t(array):
    return mt.from_numpy(np.ascontiguousarray(array))


AWKWARD = {
    "empty-1d": np.zeros(0, dtype=np.float32),
    "empty-rows": np.zeros((0, 3), dtype=np.float32),
    "empty-cols": np.zeros((3, 0), dtype=np.float32),
    "zero-dim": np.array(3.0, dtype=np.float32),
    "specials": np.array([np.nan, np.inf, -np.inf, 0.0], dtype=np.float32),
    "int32-limits": np.array([-(2**31), 0, 2**31 - 1], dtype=np.int32),
    "int64-limits": np.array([-(2**63), 0, 2**63 - 1], dtype=np.int64),
    "bool": np.array([True, False]),
    "rank-3": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
}

UNARY = [
    "abs",
    "exp",
    "log",
    "log1p",
    "sqrt",
    "rsqrt",
    "reciprocal",
    "sin",
    "acos",
    "atanh",
    "floor",
    "ceil",
    "trunc",
    "frac",
    "round",
    "sign",
    "relu",
    "relu6",
    "hardsigmoid",
    "hardswish",
    "logsigmoid",
    "mish",
    "tanhshrink",
    "sigmoid",
    "softplus",
    "erf",
    "erfinv",
    "exp2",
    "sinc",
    "lgamma",
    "digamma",
    "logit",
    "isnan",
    "isfinite",
    "bitwise_not",
    "trace",
    "flatten",
    "squeeze",
]

ALONG_DIM = [
    "sum",
    "mean",
    "prod",
    "max",
    "min",
    "argmax",
    "argmin",
    "cumsum",
    "cumprod",
    "softmax",
    "log_softmax",
    "logsumexp",
    "all",
    "any",
    "sort",
    "median",
    "norm",
    "std",
    "var",
    "nanmean",
    "nansum",
    "unsqueeze",
    "flip",
]


@pytest.mark.parametrize("name", sorted(AWKWARD))
def test_unary_ops_decline_awkward_input_without_panicking(name):
    tensor = _t(AWKWARD[name])
    for op in UNARY:
        fn = getattr(mt, op, None)
        if fn is not None:
            _assert_no_panic(f"{op}({name})", lambda fn=fn: fn(tensor))


@pytest.mark.parametrize("name", sorted(AWKWARD))
def test_reductions_decline_dimensions_that_do_not_exist(name):
    tensor = _t(AWKWARD[name])
    for op in ALONG_DIM:
        fn = getattr(mt, op, None)
        if fn is None:
            continue
        for dim in (None, 0, 1, -1, 7, -7):
            _assert_no_panic(
                f"{op}({name}, dim={dim})", lambda fn=fn, dim=dim: fn(tensor, dim=dim)
            )


@pytest.mark.parametrize("name", sorted(AWKWARD))
def test_shape_ops_decline_impossible_targets(name):
    tensor = _t(AWKWARD[name])
    for shape in ([], [0], [1], [-1], [2, 3], [-1, -1], [0, 0], [10**6]):
        _assert_no_panic(
            f"reshape({name},{shape})", lambda s=shape: mt.reshape(tensor, s)
        )
        _assert_no_panic(
            f"broadcast_to({name},{shape})", lambda s=shape: mt.broadcast_to(tensor, s)
        )
    for dim in (0, 1, -1, 7):
        _assert_no_panic(
            f"cat({name},{dim})", lambda d=dim: mt.cat([tensor, tensor], dim=d)
        )
        _assert_no_panic(
            f"stack({name},{dim})", lambda d=dim: mt.stack([tensor, tensor], dim=d)
        )
    for k in (0, 1, 5, 100):
        _assert_no_panic(f"topk({name},{k})", lambda k=k: mt.topk(tensor, k))
    for q in (-0.5, 0.0, 0.5, 1.0, 1.5):
        _assert_no_panic(f"quantile({name},{q})", lambda q=q: mt.quantile(tensor, q))
    _assert_no_panic(f"transpose({name})", lambda: mt.transpose(tensor, 0, 1))
    _assert_no_panic(f"matmul({name})", lambda: mt.matmul(tensor, tensor))
    _assert_no_panic(f"diagonal({name})", lambda: mt.diagonal(tensor))


def test_binary_ops_decline_every_mismatched_pair():
    tensors = {name: _t(array) for name, array in AWKWARD.items()}
    ops = [
        "add",
        "sub",
        "mul",
        "div",
        "maximum",
        "minimum",
        "logaddexp",
        "remainder",
        "fmod",
        "gcd",
        "lcm",
        "heaviside",
        "nextafter",
        "floor_divide",
        "eq",
        "lt",
    ]
    for lhs_name, lhs in tensors.items():
        for rhs_name, rhs in tensors.items():
            for op in ops:
                fn = getattr(mt, op, None)
                if fn is not None:
                    _assert_no_panic(
                        f"{op}({lhs_name},{rhs_name})",
                        lambda fn=fn, a=lhs, b=rhs: fn(a, b),
                    )
            _assert_no_panic(
                f"where({lhs_name},{rhs_name})",
                lambda a=lhs, b=rhs: mt.where(mt.isnan(a), a, b),
            )


def test_indexing_declines_exotic_keys():
    tensors = [
        _t(AWKWARD["rank-3"]),
        _t(np.arange(5, dtype=np.float32)),
        _t(AWKWARD["empty-rows"]),
    ]
    keys = [
        0,
        -1,
        5,
        -99,
        slice(None),
        slice(0, 0),
        slice(3, 1),
        slice(None, None, 2),
        slice(None, None, -1),
        slice(None, None, 0),
        (0, 0),
        (0, 0, 0, 0),
        (slice(None), 0),
        (Ellipsis, 0),
        ...,
        (0, ...),
        (slice(1, 100),),
    ]
    for i, tensor in enumerate(tensors):
        for key in keys:
            _assert_no_panic(f"getitem(t{i},{key!r})", lambda t=tensor, k=key: t[k])
        for indices in ([0, 1], [-1], [99], [], [0, 0, 0]):
            index = _t(np.array(indices, dtype=np.int64))
            _assert_no_panic(
                f"index_select(t{i})",
                lambda t=tensor, x=index: mt.index_select(t, 0, x),
            )
            _assert_no_panic(
                f"gather(t{i})", lambda t=tensor, x=index: mt.gather(t, 0, x)
            )


def test_assignment_declines_every_value_it_cannot_write():
    """`__setitem__` was the one indexing entry point this sweep never called,
    and the only one that could still panic: the write reads the value through
    the destination's own accessor, so a value of any other dtype had no slice
    there and the `unwrap` on it ended the process instead of the call."""

    keys = [0, -1, slice(None), slice(1, 3), (0, 0), Ellipsis, 99, slice(3, 1)]
    values = [
        _t(np.array([7.0], dtype=np.float32)),
        _t(np.array([7.0, 8.0], dtype=np.float64)),
        _t(np.array([7, 8], dtype=np.int64)),
        _t(np.array([True, False])),
        _t(AWKWARD["empty-1d"]),
        _t(AWKWARD["rank-3"]),
        _t(AWKWARD["zero-dim"]),
        7.0,
        [7.0, 8.0],
    ]
    for i, source in enumerate((np.arange(5, dtype=np.float32), AWKWARD["rank-3"])):
        for key in keys:
            for j, value in enumerate(values):

                def assign(k=key, v=value, s=source):
                    _t(s.copy())[k] = v

                _assert_no_panic(f"setitem(t{i},{key!r},v{j})", assign)


def test_autograd_misuse_declines_without_panicking():
    def backward_twice():
        x = _t(np.ones(3, dtype=np.float32)).requires_grad_(True)
        loss = mt.sum(x * x)
        loss.backward()
        loss.backward()

    def in_place_after_the_graph_saw_it():
        x = _t(np.ones(4, dtype=np.float32)).requires_grad_(True)
        y = x * 2
        x[0] = 5.0
        mt.sum(y).backward()

    _assert_no_panic("backward twice", backward_twice)
    _assert_no_panic("in-place after graph", in_place_after_the_graph_saw_it)
    _assert_no_panic(
        "backward on a non-scalar",
        lambda: (_t(np.ones(3, dtype=np.float32)).requires_grad_(True) * 2).backward(),
    )
    _assert_no_panic(
        "backward without grad",
        lambda: mt.sum(_t(np.ones(3, dtype=np.float32))).backward(),
    )


LAYERS = [
    ("DenseLayer", lambda: nn.DenseLayer(4, 3)),
    ("Conv1d", lambda: nn.Conv1d(2, 3, 3)),
    ("Conv2d", lambda: nn.Conv2d(2, 3, 3)),
    ("LayerNorm", lambda: nn.LayerNorm([4])),
    ("RMSNorm", lambda: nn.RMSNorm([4])),
    ("BatchNorm1d", lambda: nn.BatchNorm1d(4)),
    ("Embedding", lambda: nn.Embedding(10, 4)),
    ("MultiheadAttention", lambda: nn.MultiheadAttention(4, 2)),
    ("MaxPool2d", lambda: nn.MaxPool2d(2)),
]

WRONG_SHAPES = [
    np.zeros((0, 4), np.float32),
    np.zeros((1,), np.float32),
    np.zeros((2, 99), np.float32),
    np.zeros((2, 4), np.float32),
    np.zeros((1, 2, 4, 4), np.float32),
    np.array(3.0, np.float32),
]


@pytest.mark.parametrize("layer_name,build", LAYERS, ids=[name for name, _ in LAYERS])
def test_layers_decline_wrongly_shaped_input(layer_name, build):
    for i, array in enumerate(WRONG_SHAPES):
        _assert_no_panic(
            f"{layer_name}(shape {array.shape})",
            lambda a=array: build()(_t(a)),
        )


@pytest.mark.parametrize(
    "optimizer_name,build",
    [
        ("SGD", optim.SGD),
        ("Adam", optim.Adam),
        ("RMSprop", optim.RMSprop),
        ("Adagrad", optim.Adagrad),
        ("Lion", optim.Lion),
        ("NAdam", optim.NAdam),
        ("Adadelta", optim.Adadelta),
        ("Adamax", optim.Adamax),
        ("RAdam", optim.RAdam),
        ("Rprop", optim.Rprop),
    ],
)
def test_optimizers_decline_degenerate_configuration(optimizer_name, build):
    def step_without_gradients():
        parameter = _t(np.ones(3, dtype=np.float32)).requires_grad_(True)
        build([parameter], lr=1e-3).step()

    def step_on_integer_parameters():
        parameter = _t(np.ones(3, dtype=np.int64)).requires_grad_(True)
        build([parameter], lr=1e-3).step()

    _assert_no_panic(f"{optimizer_name} without gradients", step_without_gradients)
    _assert_no_panic(f"{optimizer_name} on integers", step_on_integer_parameters)
    _assert_no_panic(
        f"{optimizer_name} with no parameters", lambda: build([], lr=1e-3).step()
    )
    for lr in (0.0, -1.0, float("nan"), float("inf")):
        _assert_no_panic(
            f"{optimizer_name} lr={lr}",
            lambda lr=lr: build(
                [_t(np.ones(3, np.float32)).requires_grad_(True)], lr=lr
            ),
        )


def test_state_dict_declines_a_mismatched_checkpoint():
    _assert_no_panic(
        "state_dict mismatch",
        lambda: nn.DenseLayer(4, 3).load_state_dict(nn.DenseLayer(8, 3).state_dict()),
    )
    _assert_no_panic(
        "state_dict empty", lambda: nn.DenseLayer(4, 3).load_state_dict({})
    )


def test_lstsq_declines_a_right_hand_side_with_no_rows():
    """A 0-d right-hand side used to index at `usize::MAX - 1`.

    `lstsq` reads its row count as `dims[ndim - 2]`. The vector case is
    promoted to a matrix before that, but anything still shorter subtracted
    past zero, and the caller got a Rust panic quoting an index of
    18446744073709551614 rather than a sentence about what the argument
    should have been.
    """
    matrix = mt.Tensor(np.eye(3), dtype="float64")
    scalar = mt.Tensor(np.zeros(()), dtype="float64")

    with pytest.raises(ValueError, match="right-hand side"):
        matrix.lstsq(scalar)

    # The two shapes it does accept are unaffected.
    vector = mt.Tensor(np.array([1.0, 2.0, 3.0]), dtype="float64")
    np.testing.assert_allclose(np.asarray(matrix.lstsq(vector)), [1.0, 2.0, 3.0])
    assert list(matrix.lstsq(mt.Tensor(np.eye(3), dtype="float64")).shape) == [3, 3]


# --- sizes and bounds that arithmetic cannot represent ------------------------
#
# Found by calling every method on the type with deliberately hostile arguments
# rather than by listing cases: 7,436 one-argument calls, of which these were
# the ones that came back as something other than an `Exception`.


@pytest.mark.parametrize("offset", [10**18, -(10**18), 2**63])
def test_diag_embed_declines_an_offset_it_cannot_square(offset):
    """The output is `side x side` where `side` is the last dim plus `|offset|`.

    Both the sum and the square overflow, and `Shape::numel` panics on overflow
    rather than returning, so a large offset arrived as a Rust panic about
    `usize` instead of a sentence.
    """
    # `2**63` does not fit the signed integer the argument is converted to, so
    # pyo3 declines it before the guard is reached. That is still a refusal
    # rather than a panic, which is what this file is about.
    with pytest.raises((ValueError, OverflowError)):
        mt.Tensor(np.arange(6.0).reshape(2, 3), dtype="float64").diag_embed(offset)


@pytest.mark.parametrize("bound", [float("inf"), float("-inf"), float("nan")])
def test_arange_declines_bounds_that_are_not_finite(bound):
    """A float-to-integer cast saturates; it does not raise.

    `inf` became `usize::MAX` elements and panicked on capacity overflow. `nan`
    became zero, which is worse than a panic: an empty tensor returned as if it
    were the answer.
    """
    with pytest.raises(ValueError, match="finite"):
        mt.arange(bound)


def test_repeat_and_repeat_interleave_decline_growth_they_cannot_hold():
    """These reach an impossible size by multiplying rather than by taking one."""
    tensor = mt.Tensor(np.arange(6.0).reshape(2, 3), dtype="float64")
    with pytest.raises((ValueError, MemoryError)):
        tensor.repeat(10**9, 10**9)
    with pytest.raises((ValueError, MemoryError)):
        tensor.repeat_interleave(10**18)

    # The ordinary forms are untouched.
    assert tensor.repeat(2, 2).numel() == 24
    assert tensor.repeat_interleave(2).numel() == 12


@pytest.mark.parametrize("name", ["histc", "histogram", "histogram_bin_edges"])
def test_a_bin_count_no_memory_holds_is_declined(name):
    # Building `bins + 1` edges overflowed a capacity and panicked.
    values = _t(np.array([0.5], dtype=np.float32))
    _assert_no_panic(name, lambda: getattr(mt, name)(values, 2**62))
    with pytest.raises(Exception):
        getattr(mt, name)(values, 2**62)


@pytest.mark.parametrize("name", ["tensor_split", "array_split", "hsplit", "vsplit"])
def test_a_section_count_no_list_holds_fails_at_once(name):
    """The split points were appended one at a time, so an absurd count ground
    on until memory ran out; built by repetition, it fails the way NumPy's
    does, immediately."""

    import time

    values = _t(np.arange(6, dtype=np.float32).reshape(2, 3))
    start = time.perf_counter()
    with pytest.raises(MemoryError):
        getattr(mt, name)(values, 2**62)
    assert time.perf_counter() - start < 1.0


@pytest.mark.parametrize(
    "label, build",
    [
        ("DenseLayer(0, 0)", lambda: nn.DenseLayer(0, 0)),
        ("DenseLayer(0, 5)", lambda: nn.DenseLayer(0, 5)),
        ("Conv2d(0, 0, 0)", lambda: nn.Conv2d(0, 0, 0)),
        ("ConvTranspose1d(0, 0, 0)", lambda: nn.ConvTranspose1d(0, 0, 0)),
        ("ConvTranspose2d(0, 0, 0)", lambda: nn.ConvTranspose2d(0, 0, 0)),
        ("ConvTranspose2d(0, 2**62, 3)", lambda: nn.ConvTranspose2d(0, 2**62, 3)),
        ("GRU(1, 1, 2**62)", lambda: nn.GRU(1, 1, 2**62)),
        ("LSTM(1, 1, 2**62)", lambda: nn.LSTM(1, 1, 2**62)),
        ("LSTM(3, 2**62)", lambda: nn.LSTM(3, 2**62)),
    ],
)
def test_layers_with_empty_or_absurd_sizes_decline_or_build(label, build):
    """A layer with no inputs drew its bias from `U(-1/sqrt(0), 1/sqrt(0))`,
    which the sampler refused by panicking; a recurrent stack's sizes wrapped
    or overflowed a capacity; an empty weight's strides overflowed."""

    _assert_no_panic(label, build)


def test_a_layer_with_no_inputs_starts_from_a_zero_bias():
    # PyTorch's rule for a fan-in of zero: the bound is zero, not infinite.
    np.testing.assert_array_equal(nn.DenseLayer(0, 5).bias.numpy(), np.zeros(5))


@pytest.mark.parametrize(
    "label, run",
    [
        (
            "AdaptiveAvgPool1d(2**62)",
            lambda: nn.AdaptiveAvgPool1d(2**62)(mt.zeros([1, 2, 3])),
        ),
        (
            "AdaptiveMaxPool1d(2**62)",
            lambda: nn.AdaptiveMaxPool1d(2**62)(mt.zeros([1, 2, 3])),
        ),
        (
            "AdaptiveAvgPool2d(2**62)",
            lambda: nn.AdaptiveAvgPool2d(2**62)(mt.zeros([1, 1, 4, 4])),
        ),
        (
            "AdaptiveMaxPool2d(2**62)",
            lambda: nn.AdaptiveMaxPool2d(2**62)(mt.zeros([1, 1, 4, 4])),
        ),
        (
            "Upsample(2**62) on no planes",
            lambda: nn.Upsample(2**62)(mt.zeros([1, 0, 4])),
        ),
        ("BatchNorm1d(0)", lambda: nn.BatchNorm1d(0)(mt.zeros([1, 0, 4]))),
    ],
)
def test_forwards_with_absurd_or_empty_extents_decline(label, run):
    _assert_no_panic(label, run)
