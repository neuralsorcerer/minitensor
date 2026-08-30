# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The free-function forms of the operators, and the spellings built on them.

`a + b` has always worked; `mt.add(a, b)` had not, and neither had `a.add(b)`,
which is what most code that moves between array libraries actually writes.
These are those names -- written once here, in terms of the operators, so
`mt.add` cannot drift from `+`.

The rest of the file is the same idea applied further out: `lerp` is one
multiply-add, `fmax` is a `maximum` with the NaN cases picked out, `signbit` is
a `copysign` compared against zero, `square` is a product. Each is an
arrangement, so none of them adds a kernel and each inherits the accuracy,
the dtype rules and the gradient of what it is arranged from.
"""

from __future__ import annotations

import math as _math

from . import _core as _C
from ._shape import _atleast_tensor

Tensor = _C.Tensor
_F = _C.functional


def _scaled(other: object, alpha: float) -> object:
    """`other * alpha`, leaving a Python number a Python number.

    That last part is load-bearing: a bare `0.1` has no dtype and is read at
    the width of the tensor it meets, so converting it to a tensor here would
    pick a width before the operand it is going to meet is known -- and for a
    float64 tensor that costs the eighth digit of the number the caller wrote.
    """

    if alpha == 1:
        return other
    if isinstance(other, (int, float)) and not isinstance(other, bool):
        return other * alpha
    return _atleast_tensor(other) * alpha


# --- the operators, as functions ------------------------------------------


def add(input: object, other: object, alpha: float = 1) -> Tensor:
    """`input + alpha * other`, element-wise with broadcasting."""

    return _atleast_tensor(input) + _scaled(other, alpha)


def sub(input: object, other: object, alpha: float = 1) -> Tensor:
    """`input - alpha * other`, element-wise with broadcasting."""

    return _atleast_tensor(input) - _scaled(other, alpha)


def mul(input: object, other: object) -> Tensor:
    """`input * other`, element-wise with broadcasting."""

    return _atleast_tensor(input) * other


def div(input: object, other: object, rounding_mode: str | None = None) -> Tensor:
    """`input / other`, element-wise with broadcasting.

    `rounding_mode` picks a quotient rounded towards negative infinity
    (`"floor"`) or towards zero (`"trunc"`) instead of the exact one. The two
    disagree only for a mixed-sign quotient, which is where every remainder
    convention parts company too.
    """

    tensor = _atleast_tensor(input)
    if rounding_mode is None:
        return tensor / other
    if rounding_mode == "floor":
        return _F.floor_divide(tensor, other)
    if rounding_mode == "trunc":
        return _F.trunc(tensor / other)
    raise ValueError(
        f"div rounding_mode must be None, 'floor' or 'trunc', got {rounding_mode!r}"
    )


def neg(input: object) -> Tensor:
    """`-input`, element-wise."""

    return -_atleast_tensor(input)


def square(input: object) -> Tensor:
    """`input * input`, element-wise.

    A product rather than `pow(input, 2)`: the general power goes through
    `exp(2 * log(x))` for a non-integral exponent and is both slower and less
    exact than a multiplication, which is exact for every input.
    """

    tensor = _atleast_tensor(input)
    return tensor * tensor


# --- angles ---------------------------------------------------------------


def deg2rad(input: object) -> Tensor:
    """Degrees to radians."""

    return _atleast_tensor(input) * (_math.pi / 180.0)


def rad2deg(input: object) -> Tensor:
    """Radians to degrees."""

    return _atleast_tensor(input) * (180.0 / _math.pi)


# --- interpolation and fused forms ----------------------------------------


def lerp(input: object, end: object, weight: object) -> Tensor:
    """`input + weight * (end - input)`: the point `weight` of the way from
    `input` to `end`.

    Written as a step from `input` rather than as `(1 - w) a + w b`, so
    `weight = 0` returns `input` and `weight = 1` returns `end` exactly, with
    no rounding in between them to make either endpoint approximate.
    """

    start = _atleast_tensor(input)
    return start + _atleast_tensor(weight) * (_atleast_tensor(end) - start)


def addcmul(
    input: object, tensor1: object, tensor2: object, value: float = 1
) -> Tensor:
    """`input + value * tensor1 * tensor2`, element-wise."""

    return _atleast_tensor(input) + _scaled(
        _atleast_tensor(tensor1) * _atleast_tensor(tensor2), value
    )


def addcdiv(
    input: object, tensor1: object, tensor2: object, value: float = 1
) -> Tensor:
    """`input + value * tensor1 / tensor2`, element-wise."""

    return _atleast_tensor(input) + _scaled(
        _atleast_tensor(tensor1) / _atleast_tensor(tensor2), value
    )


# --- powers and exponentials ----------------------------------------------


def float_power(input: object, exponent: object) -> Tensor:
    """`input ** exponent` computed in float64, whatever the inputs are.

    An integer power overflows silently once the answer leaves the dtype's
    range; promoting first is the point of the name.
    """

    return _F.pow(_atleast_tensor(input).astype("float64"), exponent)


def logaddexp2(input: object, other: object) -> Tensor:
    """`log2(2**input + 2**other)`, without forming either power.

    The base-2 form of `logaddexp`, and computed by rescaling it rather than by
    a second stable implementation of the same shift-and-add.
    """

    scale = _math.log(2.0)
    return (
        _F.logaddexp(_atleast_tensor(input) * scale, _atleast_tensor(other) * scale)
        / scale
    )


def ldexp(input: object, other: object) -> Tensor:
    """`input * 2**other`, element-wise.

    Computed as the product, so an `other` large enough to overflow `2**other`
    gives infinity even where the product itself would have been finite. The
    exponent is exact, which is the part that matters: no rounding enters
    through it.
    """

    return _atleast_tensor(input) * _F.exp2(_atleast_tensor(other).astype("float64"))


# --- the NaN-skipping extrema ---------------------------------------------


def _nan_aware(input: object, other: object, pick: object) -> Tensor:
    """`pick` with each operand's NaN replaced by the other one.

    `maximum` and `minimum` propagate NaN, which is right for a comparison and
    wrong for a running extremum over data with holes in it. Two `where`s pick
    the other operand wherever one is NaN, and leave NaN only where both are.
    """

    left = _atleast_tensor(input)
    right = _atleast_tensor(other)
    return _F.where(
        _F.isnan(left), right, _F.where(_F.isnan(right), left, pick(left, right))
    )


def fmax(input: object, other: object) -> Tensor:
    """Element-wise maximum, ignoring a NaN in either operand.

    `maximum` propagates NaN; this one propagates it only where both operands
    are NaN and there is genuinely nothing to compare.
    """

    return _nan_aware(input, other, _F.maximum)


def fmin(input: object, other: object) -> Tensor:
    """Element-wise minimum, ignoring a NaN in either operand."""

    return _nan_aware(input, other, _F.minimum)


# --- predicates -----------------------------------------------------------


def isposinf(input: object) -> Tensor:
    """Whether each element is `+inf`."""

    tensor = _atleast_tensor(input)
    return _F.logical_and(_F.isinf(tensor), tensor > 0)


def isneginf(input: object) -> Tensor:
    """Whether each element is `-inf`."""

    tensor = _atleast_tensor(input)
    return _F.logical_and(_F.isinf(tensor), tensor < 0)


def isreal(input: object) -> Tensor:
    """Whether each element has no imaginary part -- true everywhere here.

    Every dtype in this library is real, so the answer is always true. The
    name exists because code written against NumPy asks, and a missing
    attribute is a worse answer than the correct one.

    Built from two comparisons rather than a tensor of ones so it is true for
    a NaN as well -- NaN has no imaginary part either, and `x == x` alone
    would say otherwise.
    """

    tensor = _atleast_tensor(input)
    return _F.eq(tensor, tensor) | _F.ne(tensor, tensor)


def signbit(input: object) -> Tensor:
    """Whether each element's sign *bit* is set.

    Not `input < 0`: negative zero is not less than zero but carries the bit,
    and telling the two zeros apart is the only reason to ask. `copysign`
    reads the bit, so borrowing it is what makes this exact.
    """

    tensor = _atleast_tensor(input)
    return _F.copysign(_C.Tensor.ones_like(tensor), tensor) < 0


def sgn(input: object) -> Tensor:
    """-1, 0 or 1 according to each element's sign.

    The same function as `sign` for real numbers; the name is the one used
    where a complex version would differ.
    """

    return _F.sign(_atleast_tensor(input))


#: Attached to the top level *and* to `Tensor`, so `mt.add(a, b)` and
#: `a.add(b)` are one definition rather than two. Each takes its tensor first,
#: which is what lets the same function serve as both.
_ELEMENTWISE = (
    "add",
    "addcdiv",
    "addcmul",
    "deg2rad",
    "div",
    "float_power",
    "fmax",
    "fmin",
    "isneginf",
    "isposinf",
    "isreal",
    "ldexp",
    "lerp",
    "logaddexp2",
    "mul",
    "neg",
    "rad2deg",
    "sgn",
    "signbit",
    "square",
    "sub",
)

#: Second spellings of operations that already exist. NumPy and PyTorch each
#: settled on a different name for several of these, and code moving between
#: them writes whichever it learned; one object under two names costs nothing
#: and a missing attribute costs the caller a rewrite.
_ALIASES = {
    "absolute": "abs",
    "concat": "cat",
    "divide": "div",
    "greater": "gt",
    "greater_equal": "ge",
    "less": "lt",
    "less_equal": "le",
    "multiply": "mul",
    "negative": "neg",
    "not_equal": "ne",
    "subtract": "sub",
    "true_divide": "div",
}
