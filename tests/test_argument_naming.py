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
import re

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


def _every_public_callable():
    """Every callable a user can reach, across all six surfaces."""

    for holder, label in (
        (mt, "minitensor"),
        (mt.Tensor, "Tensor"),
        (mt.functional, "functional"),
        (mt.numpy_compat, "numpy_compat"),
        (mt.nn, "nn"),
        (mt.optim, "optim"),
    ):
        for name in sorted(n for n in dir(holder) if not n.startswith("_")):
            function = getattr(holder, name, None)
            if (
                not callable(function)
                or inspect.isclass(function)
                or inspect.ismodule(function)
            ):
                continue
            yield f"{label}.{name}", function


def test_every_signature_says_what_its_defaults_are():
    """`help()` has to answer "what happens if I leave this out?".

    PyO3 renders a default it cannot spell -- a negative number, a `Some(0)` --
    as `...`, so `help(mt.diagonal)` said `dim1=Ellipsis, dim2=Ellipsis` for
    the two axes a caller most needs the defaults of, and sixteen signatures
    were like it. A `text_signature` fixes each, and one written by hand can be
    wrong in a way the generated one cannot: `keepdim=false` is Rust, and
    `inspect` rejects the whole signature over it.
    """

    unparseable, vague = [], []
    for label, function in _every_public_callable():
        try:
            signature = inspect.signature(function)
        except Exception as exc:  # a hand-written signature Python cannot read
            unparseable.append(f"{label}: {type(exc).__name__}")
            continue
        vague.extend(
            f"{label}({parameter})"
            for parameter, value in signature.parameters.items()
            if value.default is Ellipsis
        )

    assert not unparseable, "signatures Python cannot parse: " + ", ".join(unparseable)
    assert not vague, "defaults rendered as `...`: " + ", ".join(vague)


def test_the_numerical_floor_is_called_eps_everywhere():
    """One name for one thing, on the layers and the optimizers alike.

    Twenty-two places called it `eps` and five called it `epsilon`, and the
    split ran through the optimizers themselves: `Adamax(eps=...)` beside
    `Adam(epsilon=...)`. It was an accident that accumulated -- the
    reference-algorithm sweep had a line in it naming both spellings so its
    own guard would keep working.
    """

    offenders = []
    for label, function in _every_public_callable():
        signature = getattr(function, "__text_signature__", None)
        if signature is None:
            try:
                signature = str(inspect.signature(function))
            except Exception:
                continue
        if re.search(r"\bepsilon\b", signature):
            offenders.append(label)

    assert not offenders, "these say `epsilon` rather than `eps`: " + ", ".join(
        offenders
    )


def test_no_signature_hides_a_default_behind_none():
    """`reduction=None` said nothing; the answer is always `"mean"`.

    A PyO3 argument taken as `Option<T>` and resolved with `unwrap_or` in the
    body advertises `None`, which tells a reader of `help()` neither the value
    nor that there is one. Twenty-five losses did that for `reduction` while
    two of their neighbours said `reduction="mean"` outright, and the
    activations, the optimizers and `HuberLoss`'s `delta` were the same.

    `None` is still a fine default where it *is* the value -- an absent
    `weight`, an absent `pos_weight`, a `dim` meaning "all of them". What this
    catches is the arguments below, which have a real default and hid it.
    """

    named = (
        "reduction",
        "alpha",
        "gamma",
        "delta",
        "beta",
        "eps",
        "lambd",
        "negative_slope",
        "approximate",
        "momentum",
        "threshold",
    )
    # `logit`'s `eps` is the exception that proves the rule: `None` there means
    # "do not clamp", which is a value of the argument and what PyTorch's
    # `logit` means by it too, so it has no hidden default to reveal.
    allowed = {"minitensor.logit", "Tensor.logit", "functional.logit"}
    offenders = []
    for label, function in _every_public_callable():
        if label in allowed:
            continue
        signature = getattr(function, "__text_signature__", None)
        if signature is None:
            try:
                signature = str(inspect.signature(function))
            except Exception:
                continue
        offenders.extend(
            f"{label}({argument}=None)"
            for argument in named
            if f"{argument}=None" in signature
        )

    assert not offenders, "defaults hidden behind `None`: " + ", ".join(offenders)
