# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""One name for one thing, across the whole surface.

The axis argument is `dim` and keeping it is `keepdim` -- PyTorch's names,
used even on the functions NumPy contributed, where NumPy would say `axis` and
`keepdims`. The exception is a function whose own name says which word it
wants: `swapaxes`, `take_along_axis` and `put_along_axis` take `axis`, and
`swapdims` takes `dim`, exactly as PyTorch spells each of them.

This is checked rather than remembered because it is the kind of rule that
drifts one function at a time. `trim_zeros` was added with NumPy's `axis` and
nothing noticed; `swapaxes` had a method taking `dim0` and a free function
taking `axis0`, so the same operation wanted a different keyword depending on
which spelling you reached for.
"""

import inspect

import pytest

import minitensor as mt

# An argument naming an axis, under either library's spelling.
_AXIS_WORDS = {
    "axis",
    "axes",
    "axis0",
    "axis1",
    "axis2",
    "dim",
    "dims",
    "dim0",
    "dim1",
    "dim2",
}
_NUMPY_WORDS = {name for name in _AXIS_WORDS if name.startswith("ax")}

# The functions whose own name picks the word, and what it picks.
_NAMED_FOR_THEIR_ARGUMENT = {
    "swapaxes": "axis",
    "take_along_axis": "axis",
    "put_along_axis": "axis",
    "apply_along_axis": "axis",
    "permute_dims": "axes",  # the array API's own spelling of `permute`
}


def _exported_callables():
    for name in sorted(mt.__all__):
        function = getattr(mt, name, None)
        if (
            not callable(function)
            or inspect.isclass(function)
            or inspect.ismodule(function)
        ):
            continue
        try:
            parameters = list(inspect.signature(function).parameters)
        except (ValueError, TypeError):
            continue  # a builtin with no introspectable signature
        yield name, function, parameters


def test_the_axis_argument_is_called_dim_unless_the_name_says_otherwise():
    offenders = []
    checked = 0
    for name, _, parameters in _exported_callables():
        axis_like = [p for p in parameters if p in _AXIS_WORDS]
        if not axis_like:
            continue
        checked += 1
        numpy_spelled = [p for p in axis_like if p in _NUMPY_WORDS]
        if not numpy_spelled:
            continue
        if _NAMED_FOR_THEIR_ARGUMENT.get(name) in {"axis", "axes"}:
            continue
        offenders.append(f"{name}{tuple(numpy_spelled)}")

    assert not offenders, (
        "these spell the axis argument NumPy's way without a name that asks "
        "for it: " + ", ".join(offenders)
    )
    assert checked > 80, f"only {checked} functions reached -- the sweep broke"


def test_keeping_the_axis_is_called_keepdim():
    offenders = [
        name
        for name, _, parameters in _exported_callables()
        if "keepdims" in parameters
    ]
    assert not offenders, "these say `keepdims` rather than `keepdim`: " + ", ".join(
        offenders
    )


@pytest.mark.parametrize(
    "name", sorted(n for n, _, _ in _exported_callables() if hasattr(mt.Tensor, n))
)
def test_a_method_and_its_free_function_take_the_same_keywords(name):
    """`t.swapaxes(dim0=0)` worked while `mt.swapaxes(t, dim0=0)` raised.

    The free function takes the tensor as its first argument and the method
    takes it as `self`, so past that first parameter the two have to agree --
    otherwise which keyword a caller may write depends on which spelling they
    reached for, and one of the two is always wrong.
    """

    try:
        free = list(inspect.signature(getattr(mt, name)).parameters)
        method = list(inspect.signature(getattr(mt.Tensor, name)).parameters)
    except (ValueError, TypeError):
        pytest.skip(f"{name} has no introspectable signature")

    # Drop the tensor each form takes first, however it is spelled.
    free_rest = [p for p in free[1:] if p not in ("self",)]
    method_rest = [p for p in method if p not in ("self",)]
    shared = set(free_rest) & _AXIS_WORDS
    other = set(method_rest) & _AXIS_WORDS
    if not shared and not other:
        pytest.skip(f"{name} takes no axis argument")

    assert shared == other, (
        f"mt.{name} names its axis argument {sorted(shared)} while "
        f"Tensor.{name} names it {sorted(other)}"
    )


# The free function takes the tensor first under one of these names; the method
# takes it as `self`. Past that, the two describe the same call.
_LEADING_TENSOR = {"input", "a", "x", "tensor"}

# Where the two forms legitimately differ past the leading tensor, with why.
_DIFFERENT_BY_DESIGN = {
    # `mt.where(condition, input, other)` picks between two tensors, while
    # `t.where(condition, other)` makes `t` the "if true" branch -- so the
    # method's argument list is the free function's with `input` removed from
    # the middle rather than the front. PyTorch is arranged the same way.
    "where",
    # `mt.polygamma(order, input)` takes the order first, as NumPy and SciPy do.
    "polygamma",
    # These take the tensor under a name of their own (`a`, `b`, `factor`),
    # which the method supplies as `self` from a different position.
    "cholesky_solve",
    "lstsq",
    "lu_solve",
    "solve",
    "solve_triangular",
    "tensorinv",
    "tensorsolve",
    "vander",
    "cross",
}


def _free_and_method_arguments(name):
    """The two argument lists, with the leading tensor dropped from each."""

    free = list(inspect.signature(getattr(mt, name)).parameters)
    method = [
        p for p in inspect.signature(getattr(mt.Tensor, name)).parameters if p != "self"
    ]
    if free and free[0] in _LEADING_TENSOR:
        free = free[1:]
    # A Python-level forwarder is written as `f(input, ...)` and bound as a
    # method, so its own first parameter is the tensor on both sides.
    if method and method[0] in _LEADING_TENSOR and method[:1] != free[:1]:
        method = method[1:]
    return free, method


@pytest.mark.parametrize(
    "name",
    sorted(
        n
        for n, _, _ in _exported_callables()
        if hasattr(mt.Tensor, n) and n not in _DIFFERENT_BY_DESIGN
    ),
)
def test_a_method_and_its_free_function_describe_the_same_call(name):
    """`t.chunk(sections=2)` worked while `mt.chunk(t, sections=2)` did not.

    Three of these were live when this was written: `chunk` said `chunks` on
    one side and `sections` on the other, `pow` said `other` and `exponent`,
    and `Tensor.rms_norm` required the two arguments its free function -- and
    the reference -- give defaults to.
    """

    try:
        free, method = _free_and_method_arguments(name)
    except (ValueError, TypeError):
        pytest.skip(f"{name} has no introspectable signature")

    assert free == method, f"mt.{name} takes {free} while Tensor.{name} takes {method}"


@pytest.mark.parametrize(
    "name",
    sorted(
        n
        for n, _, _ in _exported_callables()
        if hasattr(mt.Tensor, n) and n not in _DIFFERENT_BY_DESIGN
    ),
)
def test_a_method_makes_the_same_arguments_optional(name):
    """An argument with a default on one side must have one on the other.

    `Tensor.rms_norm` required `weight` and `eps` where the free function
    defaulted them, so the method could not be called the way the reference
    documented it.
    """

    try:
        free_params = inspect.signature(getattr(mt, name)).parameters
        method_params = inspect.signature(getattr(mt.Tensor, name)).parameters
        free, method = _free_and_method_arguments(name)
    except (ValueError, TypeError):
        pytest.skip(f"{name} has no introspectable signature")
    if free != method:
        pytest.skip(f"{name} has differing argument lists")

    empty = inspect.Parameter.empty
    mismatched = [
        p
        for p in free
        if (free_params[p].default is empty) != (method_params[p].default is empty)
    ]
    assert (
        not mismatched
    ), f"{name}: {mismatched} are optional on one spelling and required on the other"

    # And the same default, not merely some default. A PyO3 method taking
    # `Option<f64>` and calling `unwrap_or(1.0)` advertises `alpha=None`, which
    # tells a reader of `help()` nothing about what alpha will be -- while the
    # free function beside it says `alpha=1.0`.
    differing = [
        f"{p}: {free_params[p].default!r} vs {method_params[p].default!r}"
        for p in free
        if free_params[p].default is not empty
        and method_params[p].default is not empty
        and free_params[p].default != method_params[p].default
    ]
    assert (
        not differing
    ), f"{name} advertises different defaults on its two spellings: " + ", ".join(
        differing
    )
