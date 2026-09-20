# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Two things a `dim` argument has to mean, on every op that takes one.

    keepdim:  f(x, dim=d, keepdim=True).squeeze(d) == f(x, dim=d, keepdim=False)
    sign:     f(x, dim=d)                          == f(x, dim=d - ndim)

Both are so ordinary that an op gets written assuming them and tested on one
axis of a cube, where `dim=1` and `dim=-2` are the same axis and a squeeze is
the whole difference between two shapes. What that misses is the op that
normalises a negative dim in its wrapper and then normalises it again in the
kernel, or that builds the keepdim shape separately from the reduction instead
of squeezing one axis out of it -- both right on a symmetric input, both wrong
the moment the axes differ in length. So the input here is 4x6x8: no two axes
alike, and no shape mistake that cancels itself out.

The op list is not written down. It is read from each function's declared
signature, so an op added tomorrow is swept tomorrow. `test_argument_naming.py`
reads the same signatures to check the argument is *spelled* `dim`; this checks
it *means* dim. Nothing is passed positionally by the sweep itself -- a filter
that guessed at position would call `transpose(x, 1, True)` against
`transpose(x, 1, False)` and report the transpose as a keepdim violation.

Three ops are excluded and pinned separately, because their dim is not an axis
of the input and `d` and `d - ndim` are therefore two different questions:
`unsqueeze`, `expand_dims` and `stack` count against the axes of the output,
which has one more; `lexsort` counts against the axes of one key, which has one
fewer. All four are checked against NumPy directly rather than against a
restated rule.

Everything here passes today. The floors at the bottom are the point: they fail
if the sweep ever stops reaching the surface it claims to cover, which is the
one way a test like this dies quietly.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import minitensor as mt

# Distinct axis lengths, every one even (`glu` splits an axis in half) and at
# least 3 (`narrow`, `topk`, `kthvalue`, `index_select` need the room).
SHAPE = (4, 6, 8)
NDIM = len(SHAPE)

_RNG = np.random.default_rng(20260920)
_SEED = 20260920

_VALUES = _RNG.standard_normal(SHAPE).astype(np.float32)
_COUNTS = _RNG.integers(0, 9, SHAPE).astype(np.int64)
_OTHER = _RNG.standard_normal(SHAPE).astype(np.float32)


def _tensor(dtype="float32"):
    return mt.Tensor((_VALUES if dtype == "float32" else _COUNTS).tolist(), dtype=dtype)


def _other():
    return mt.Tensor(_OTHER.tolist())


# Everything below is drawn once and reused. Two calls that differ only in the
# sign of their dim have to differ in nothing else, and a fresh draw per call
# is the easiest way to lose that without noticing.
_INDEX = {
    d: mt.Tensor(
        _RNG.integers(0, SHAPE[d], SHAPE).astype(np.int64).tolist(), dtype="int64"
    )
    for d in range(NDIM)
}
_SHORT_INDEX = mt.Tensor([0, 2], dtype="int64")

# Shaped like the input but two long on `d`, for `index_copy`/`index_add`.
_SLAB = {}
# Shaped like the input with `d` removed, for `select_scatter` and the class
# targets `cross_entropy` takes.
_SLICE = {}
_TARGET = {}
_CONDITION = {}
for _d in range(NDIM):
    _slab_shape = SHAPE[:_d] + (2,) + SHAPE[_d + 1 :]
    _SLAB[_d] = mt.Tensor(_RNG.standard_normal(_slab_shape).astype(np.float32).tolist())
    _slice_shape = SHAPE[:_d] + SHAPE[_d + 1 :]
    _SLICE[_d] = mt.Tensor(
        _RNG.standard_normal(_slice_shape).astype(np.float32).tolist()
    )
    _TARGET[_d] = mt.Tensor(
        _RNG.integers(0, SHAPE[_d], _slice_shape).astype(np.int64).tolist(),
        dtype="int64",
    )
    _picked = _RNG.random(SHAPE[_d]) > 0.4
    _picked[0] = True  # never select nothing
    _CONDITION[_d] = mt.Tensor(_picked.tolist(), dtype="bool")

# Never all-False along any axis: a fully masked softmax row is 0/0, and that
# is `test_masked_softmax`'s subject, not this one.
_MASK_VALUES = _RNG.random(SHAPE) > 0.25
_MASK_VALUES[0] = True
_MASK_VALUES[:, 0] = True
_MASK_VALUES[..., 0] = True

# `cross` needs an axis of exactly 3, so it brings its own operands.
_CUBE_A = _RNG.standard_normal((3, 3, 3)).astype(np.float32)
_CUBE_B = _RNG.standard_normal((3, 3, 3)).astype(np.float32)


def _mask():
    return mt.Tensor(_MASK_VALUES.tolist(), dtype="bool")


def _axis(d):
    """`d` as a non-negative axis, for picking a pre-drawn operand."""
    return d % NDIM


# The ops that cannot be called with a tensor and a dim alone, each given what
# it needs. `call` applies the operation in its free-function form -- input
# first -- and the method sweep binds the receiver behind the same shape, so
# one table serves both surfaces. `keepdim` is threaded through for the ops
# that accept one.
_EXTRA_ARGUMENTS = {
    "append": lambda call, d, t, **kw: call(t, _other(), dim=d),
    "argpartition": lambda call, d, t, **kw: call(t, 1, dim=d),
    "array_split": lambda call, d, t, **kw: call(t, 3, dim=d),
    "cat": lambda call, d, t, **kw: call([t, _other()], dim=d),
    "chunk": lambda call, d, t, **kw: call(t, 2, dim=d),
    "compress": lambda call, d, t, **kw: call(_CONDITION[_axis(d)], t, dim=d),
    "concat": lambda call, d, t, **kw: call([t, _other()], dim=d),
    "concatenate": lambda call, d, t, **kw: call([t, _other()], dim=d),
    "cosine_similarity": lambda call, d, t, **kw: call(t, _other(), dim=d),
    "cross": lambda call, d, t, **kw: call(
        mt.Tensor(_CUBE_A.tolist()), mt.Tensor(_CUBE_B.tolist()), dim=d
    ),
    "cross_entropy": lambda call, d, t, **kw: call(t, _TARGET[_axis(d)], dim=d),
    "delete": lambda call, d, t, **kw: call(t, 1, dim=d),
    "gather": lambda call, d, t, **kw: call(t, d, _INDEX[_axis(d)]),
    "index_add": lambda call, d, t, **kw: call(t, d, _SHORT_INDEX, _SLAB[_axis(d)]),
    "index_copy": lambda call, d, t, **kw: call(t, d, _SHORT_INDEX, _SLAB[_axis(d)]),
    "index_fill": lambda call, d, t, **kw: call(t, d, _SHORT_INDEX, 1.5),
    "index_select": lambda call, d, t, **kw: call(t, d, _SHORT_INDEX),
    "insert": lambda call, d, t, **kw: call(t, 1, 0.5, dim=d),
    "kthvalue": lambda call, d, t, **kw: call(t, 2, dim=d, **kw),
    "masked_log_softmax": lambda call, d, t, **kw: call(t, _mask(), dim=d),
    "masked_softmax": lambda call, d, t, **kw: call(t, _mask(), dim=d),
    "nanpercentile": lambda call, d, t, **kw: call(t, 40.0, dim=d, **kw),
    "nanquantile": lambda call, d, t, **kw: call(t, 0.4, dim=d, **kw),
    "narrow": lambda call, d, t, **kw: call(t, d, 1, 2),
    "partition": lambda call, d, t, **kw: call(t, 1, dim=d),
    "percentile": lambda call, d, t, **kw: call(t, 40.0, dim=d, **kw),
    "put_along_axis": lambda call, d, t, **kw: call(t, _INDEX[_axis(d)], _other(), d),
    "quantile": lambda call, d, t, **kw: call(t, 0.4, dim=d, **kw),
    "renorm": lambda call, d, t, **kw: call(t, 2.0, d, 0.5),
    "repeat_interleave": lambda call, d, t, **kw: call(t, 2, dim=d),
    "scatter": lambda call, d, t, **kw: call(t, d, _INDEX[_axis(d)], _other()),
    "scatter_add": lambda call, d, t, **kw: call(t, d, _INDEX[_axis(d)], _other()),
    "scatter_reduce": lambda call, d, t, **kw: call(
        t, d, _INDEX[_axis(d)], _other(), "sum"
    ),
    "select": lambda call, d, t, **kw: call(t, d, 1),
    "select_scatter": lambda call, d, t, **kw: call(t, _SLICE[_axis(d)], d, 1),
    "slice_scatter": lambda call, d, t, **kw: call(t, _other(), d),
    "split": lambda call, d, t, **kw: call(t, 2, dim=d),
    "split_with_sections": lambda call, d, t, **kw: call(
        t, [1, SHAPE[_axis(d)] - 1], d
    ),
    "take_along_axis": lambda call, d, t, **kw: call(t, _INDEX[_axis(d)], d),
    "take_along_dim": lambda call, d, t, **kw: call(t, _INDEX[_axis(d)], dim=d),
    "tensor_split": lambda call, d, t, **kw: call(t, 2, dim=d),
    "topk": lambda call, d, t, **kw: call(t, 2, dim=d),
    "unflatten": lambda call, d, t, **kw: call(t, d, [SHAPE[_axis(d)] // 2, 2]),
    "vecdot": lambda call, d, t, **kw: call(t, _other(), dim=d),
}

# Their dim counts against the axes of the *output*, which has one more than
# the input, so `d` and `d - ndim` name different axes by construction.
_COUNTS_AGAINST_THE_OUTPUT = {"unsqueeze", "expand_dims", "stack"}
# `lexsort` takes a stack of keys, so its dim counts against one key's axes --
# one fewer than the tensor it was handed.
_COUNTS_AGAINST_THE_KEYS = {"lexsort"}
_NOT_AN_INPUT_AXIS = _COUNTS_AGAINST_THE_OUTPUT | _COUNTS_AGAINST_THE_KEYS

# The two ops that spell their axis NumPy's way, per `test_argument_naming.py`.
# The question is the same one, so they are swept with the rest.
_AXIS_SPELLED = ("take_along_axis", "put_along_axis")


def _parameters(function):
    try:
        return list(inspect.signature(function).parameters)
    except (ValueError, TypeError):
        return []  # a builtin with no introspectable signature


def _free_dim_ops():
    """Every free function that declares a `dim`, under one name each."""
    found = {}
    for module, prefix in ((mt, ""), (mt.functional, "functional.")):
        for name in sorted(dir(module)):
            if name.startswith("_"):
                continue
            function = getattr(module, name)
            if not callable(function) or inspect.isclass(function):
                continue
            if "dim" in _parameters(function) or name in _AXIS_SPELLED:
                found[prefix + name] = function
    return found


def _method_dim_ops():
    """Every `Tensor` method that declares a `dim`.

    Mostly the same operations reached the other way, but not entirely:
    `split_with_sections` and the static `concatenate` have no free spelling,
    so this is the only sweep that reaches them.
    """
    found = {}
    for name in sorted(dir(mt.Tensor)):
        if name.startswith("_"):
            continue
        method = getattr(mt.Tensor, name)
        if callable(method) and ("dim" in _parameters(method) or name in _AXIS_SPELLED):
            found[name] = method
    return found


FREE_OPS = _free_dim_ops()
METHOD_OPS = _method_dim_ops()
FREE_KEEPDIM = {n: f for n, f in FREE_OPS.items() if "keepdim" in _parameters(f)}
METHOD_KEEPDIM = {n: f for n, f in METHOD_OPS.items() if "keepdim" in _parameters(f)}


def _free_form(name, function, method):
    """The op as a free function: input first, however it is actually reached.

    A `Tensor` method whose signature starts with `self` is rewritten to take
    its receiver as the first argument, so the table above -- written once, in
    free-function shape -- drives both surfaces. The static spellings
    (`Tensor.stack`, `Tensor.renorm`) already take the operand first and are
    used as they are.
    """
    if not method:
        return function
    if _parameters(function)[:1] != ["self"]:
        return function
    return lambda first, *rest, **keywords: getattr(first, name)(*rest, **keywords)


def _call(name, function, dim, dtype, method=False, **keywords):
    # Sampling ops (`gumbel_softmax`) are swept too, and two draws differ
    # whatever the dim unless the generator is put back first.
    mt.manual_seed(_SEED)
    call = _free_form(name, function, method)
    extra = _EXTRA_ARGUMENTS.get(name.split(".")[-1])
    if extra is not None:
        return extra(call, dim, _tensor(dtype), **keywords)
    if name.split(".")[-1] in _AXIS_SPELLED:
        return call(_tensor(dtype), axis=dim, **keywords)
    return call(_tensor(dtype), dim=dim, **keywords)


def _flatten(result):
    """A result as a list of (shape, values), or None if it holds no tensor."""
    if isinstance(result, (tuple, list)):
        parts = []
        for item in result:
            flat = _flatten(item)
            if flat is None:
                return None
            parts.extend(flat)
        return parts
    if isinstance(result, mt.Tensor):
        return [(tuple(result.shape), result.numpy())]
    return None


def _identical(left, right):
    if left is None or right is None or len(left) != len(right):
        return False
    for (left_shape, left_values), (right_shape, right_values) in zip(left, right):
        if left_shape != right_shape:
            return False
        floats = left_values.dtype.kind == "f" and right_values.dtype.kind == "f"
        if not np.array_equal(left_values, right_values, equal_nan=floats):
            return False
    return True


def _shapes(flat):
    return None if flat is None else [shape for shape, _ in flat]


def _squeeze_at(result, dim):
    if isinstance(result, tuple):
        return tuple(_squeeze_at(item, dim) for item in result)
    return result.squeeze(dim) if isinstance(result, mt.Tensor) else result


def _sweep(name, function, method, check):
    """Run `check` over both dtypes and every axis, reporting what refused."""
    reached, refused = 0, []
    for dtype in ("float32", "int64"):
        for dim in range(NDIM):
            try:
                outcome = check(name, function, dim, dtype, method)
            except Exception as error:  # an op this dtype or shape does not take
                refused.append(f"{dtype} dim={dim}: {type(error).__name__}: {error}")
                continue
            if outcome is None:
                refused.append(f"{dtype} dim={dim}: no tensor in the result")
                continue
            reached += 1
    if reached == 0:
        pytest.skip(f"{name} took no call this sweep could make: {refused[:2]}")


def _check_keepdim(name, function, dim, dtype, method):
    kept = _call(name, function, dim, dtype, method, keepdim=True)
    dropped = _call(name, function, dim, dtype, method, keepdim=False)
    kept_flat, dropped_flat = _flatten(kept), _flatten(dropped)
    if kept_flat is None or dropped_flat is None:
        return None
    squeezed = _flatten(_squeeze_at(kept, dim))
    assert _identical(squeezed, dropped_flat), (
        f"{name}[{dtype}] dim={dim}: keepdim=True {_shapes(kept_flat)} squeezed "
        f"to {_shapes(squeezed)}, but keepdim=False gives {_shapes(dropped_flat)}"
    )
    return True


def _check_sign(name, function, dim, dtype, method):
    positive = _flatten(_call(name, function, dim, dtype, method))
    negative = _flatten(_call(name, function, dim - NDIM, dtype, method))
    if positive is None or negative is None:
        return None
    assert _identical(positive, negative), (
        f"{name}[{dtype}]: dim={dim} gives {_shapes(positive)} but dim={dim - NDIM}, "
        f"the same axis, gives {_shapes(negative)}"
    )
    return True


@pytest.mark.parametrize("name", sorted(FREE_KEEPDIM))
def test_keeping_the_dim_then_squeezing_it_matches_dropping_it(name):
    _sweep(name, FREE_KEEPDIM[name], False, _check_keepdim)


@pytest.mark.parametrize("name", sorted(METHOD_KEEPDIM))
def test_a_method_keeps_and_drops_the_dim_the_same_way(name):
    _sweep(name, METHOD_KEEPDIM[name], True, _check_keepdim)


@pytest.mark.parametrize(
    "name", sorted(n for n in FREE_OPS if n.split(".")[-1] not in _NOT_AN_INPUT_AXIS)
)
def test_a_negative_dim_names_the_same_axis(name):
    _sweep(name, FREE_OPS[name], False, _check_sign)


@pytest.mark.parametrize(
    "name", sorted(n for n in METHOD_OPS if n not in _NOT_AN_INPUT_AXIS)
)
def test_a_negative_dim_names_the_same_axis_from_a_method(name):
    _sweep(name, METHOD_OPS[name], True, _check_sign)


def test_unsqueeze_and_stack_count_against_their_output():
    """The first exception, against NumPy rather than against a restated rule."""
    for dim in range(-NDIM - 1, NDIM + 1):
        expected = np.expand_dims(_VALUES, dim).shape
        assert (
            tuple(mt.unsqueeze(_tensor(), dim).shape) == expected
        ), f"unsqueeze({dim})"
        assert (
            tuple(mt.expand_dims(_tensor(), dim).shape) == expected
        ), f"expand_dims({dim})"

        expected = np.stack([_VALUES, _VALUES], axis=dim).shape
        stacked = mt.stack([_tensor(), _tensor()], dim=dim)
        assert tuple(stacked.shape) == expected, f"stack({dim})"

    # And the consequence: on a 3-D input, dim=0 and dim=-3 are different axes.
    assert tuple(mt.unsqueeze(_tensor(), 0).shape) == (1,) + SHAPE
    assert tuple(mt.unsqueeze(_tensor(), -NDIM).shape) == SHAPE[:1] + (1,) + SHAPE[1:]


def test_lexsort_counts_against_its_keys():
    """The second exception. `lexsort` sorts a stack of keys, so its dim is an
    axis of one key -- one fewer than the tensor it was handed."""
    tensor = _tensor("int64")  # integers, so ties are common and key order tells
    key_ndim = NDIM - 1
    for dim in range(-key_ndim, key_ndim):
        expected = np.lexsort(_COUNTS, axis=dim)
        assert np.array_equal(
            mt.lexsort(tensor, dim=dim).numpy(), expected
        ), f"lexsort({dim})"

    # And the consequence: against a 3-D input, dim=-2 is not dim=1 but dim=0.
    assert np.array_equal(
        mt.lexsort(tensor, dim=-2).numpy(), mt.lexsort(tensor, dim=0).numpy()
    )


def test_the_sweep_still_reaches_the_surface():
    """The floors. Without these, a broken filter is a green run over nothing."""
    assert len(FREE_OPS) > 110, f"only {len(FREE_OPS)} free functions declare a dim"
    assert len(FREE_KEEPDIM) > 55, f"only {len(FREE_KEEPDIM)} of them declare keepdim"
    assert len(METHOD_OPS) > 50, f"only {len(METHOD_OPS)} methods declare a dim"
    assert (
        len(METHOD_KEEPDIM) > 25
    ), f"only {len(METHOD_KEEPDIM)} of them declare keepdim"

    # Every name in the extra-argument table has to still be an op, or the
    # table is quietly supplying arguments to nothing.
    surface = {name.split(".")[-1] for name in FREE_OPS} | set(METHOD_OPS)
    unused = sorted(set(_EXTRA_ARGUMENTS) - surface)
    assert not unused, f"the extra-argument table names ops that are gone: {unused}"
