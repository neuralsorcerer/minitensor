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
