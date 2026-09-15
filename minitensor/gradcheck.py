# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Checking a backward against the forward it claims to differentiate.

    x = mt.Tensor([0.3, -1.2, 2.0], dtype="float64", requires_grad=True)
    assert mt.gradcheck(lambda t: t.tanh().sum(), (x,))

The engine's own operations are checked this way in its test suite. What has
had no way to be checked is the other kind: an operation whose backward someone
wrote, through `minitensor.autograd.Function` or `register_custom_op`. Nothing
verifies such a backward, because nothing can -- the forward is right, so every
test of the forward's values passes, and the gradient is wrong only where
somebody looks. This is how to look.

**Run it in float64.** A central difference subtracts two nearly equal numbers,
so its accuracy is bounded by the input's precision rather than by `eps`: in
float32 the surviving agreement is around 1e-2, loose enough to miss a dropped
factor on one branch of a piecewise function, while in float64 it is around
1e-9. `gradcheck` refuses a float32 input rather than report a pass it cannot
stand behind.

That 1e-9 is what the *method* resolves, not what is asserted. The default
`atol` and `rtol` are the conventional 1e-5 and 1e-3, which catch the errors
that matter -- a dropped factor, a missing chain-rule term, a sign -- and leave
room for the conditioning of whatever is being differentiated. Tighten them
when checking something smooth and well-scaled; there are several orders of
headroom before the difference itself becomes the limit.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from . import _core as _C

Tensor = _C.Tensor

#: Large enough that the central difference's truncation error (order `eps^2`)
#: stays under float64's subtraction error (order `1e-16 / eps`), small enough
#: that it stays negligible. The product of the two is smallest near here.
_DEFAULT_EPS = 1e-6


def gradcheck(
    func: Callable[..., Any],
    inputs: Sequence[Any],
    *,
    eps: float = _DEFAULT_EPS,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    raise_exception: bool = True,
) -> bool:
    """Compare `func`'s analytic gradients against numerical ones.

    Parameters
    ----------
    func:
        Called as `func(*inputs)`. Must return a scalar tensor; reduce with
        `.sum()` if it does not, which leaves the check equivalent.
    inputs:
        Every tensor among them with `requires_grad` set is checked. The rest
        are passed through untouched, which is how a non-differentiable
        argument is expressed.

        They must be **independent** of one another. The same tensor passed
        twice is fine and handled -- both occurrences are perturbed together --
        but one input computed *from* another is not: perturbing the first
        leaves the second at its old value, so the numerical derivative is
        missing a path the analytic gradient has, and the mismatch is reported
        against a backward that was correct. Build each input from its own
        data.
    eps:
        Half-width of the central difference.
    atol, rtol:
        Tolerances, as `numpy.allclose` uses them.
    raise_exception:
        Raise with a description of the worst mismatch (the default) rather
        than returning `False`.

    Returns
    -------
    `True` when every checked input agrees.
    """

    inputs = tuple(inputs)

    # Group the positions by tensor *identity*, not by position. One tensor
    # passed twice -- `gradcheck(lambda a, b: (a * b).sum(), (x, x))` -- is one
    # variable appearing twice, and its analytic gradient accumulates from both
    # occurrences. Perturbing only the first would measure half of that and
    # report a mismatch against a backward that was right, which is the worst
    # thing a checking tool can do.
    groups: dict[int, list[int]] = {}
    for index, value in enumerate(inputs):
        if isinstance(value, Tensor) and value.requires_grad:
            groups.setdefault(id(value), []).append(index)
    if not groups:
        raise ValueError(
            "gradcheck needs at least one input tensor with requires_grad=True"
        )
    for positions in groups.values():
        first = positions[0]
        if str(inputs[first].dtype) != "float64":
            raise ValueError(
                f"input {first} is {inputs[first].dtype}; gradcheck needs float64, "
                "because a central difference in float32 agrees to about 1e-2 and "
                "that is not evidence of anything"
            )

    analytic = _analytic(func, inputs, groups)
    for key, positions in groups.items():
        numeric = _numeric(func, inputs, positions, eps)
        got = analytic[key]
        if got is None:
            # No gradient reached this input at all, which is a claim of zero.
            got = np.zeros_like(numeric)
        if got.shape != numeric.shape:
            # `np.allclose` would broadcast these and could agree by accident.
            # The engine rejects a mis-shaped gradient before it reaches here,
            # so this guards the day that stops being true rather than a case
            # seen in practice.
            if not raise_exception:
                return False
            raise AssertionError(
                f"gradient for input {positions[0]} has shape {got.shape}, but the "
                f"input is {numeric.shape}"
            )
        if np.allclose(got, numeric, atol=atol, rtol=rtol):
            continue
        if not raise_exception:
            return False
        raise AssertionError(_describe(positions[0], got, numeric, atol, rtol))
    return True


def _analytic(func, inputs, groups) -> dict[int, np.ndarray | None]:
    # `set_to_none=True`, not a zero fill: gradients accumulate across backward
    # passes here, so a stale one would be added to this one, and a cleared
    # `.grad` is what keeps "no gradient reached this input" distinguishable
    # from "the gradient is zero".
    for positions in groups.values():
        inputs[positions[0]].zero_grad(True)

    output = func(*inputs)
    if not isinstance(output, Tensor):
        raise ValueError(
            f"func must return a scalar tensor; got {type(output).__name__}."
        )
    if output.numel() != 1:
        raise ValueError(
            f"func must return a scalar tensor; got shape {tuple(output.shape)}. "
            "Reduce it with .sum(): the check is equivalent and the gradient is "
            "the one you wanted."
        )
    output.backward()

    gradients: dict[int, np.ndarray | None] = {}
    for key, positions in groups.items():
        grad = inputs[positions[0]].grad
        gradients[key] = None if grad is None else np.asarray(grad, np.float64).copy()
    return gradients


def _numeric(func, inputs, positions, eps) -> np.ndarray:
    """One element at a time, with recording off.

    `positions` is every argument slot holding the tensor under test -- more
    than one when the caller passed it twice. All of them take the perturbed
    copy, so what this measures is the total derivative, which is what the
    accumulated analytic gradient is.

    The perturbed forwards must not touch the tape: a gradcheck over a large
    input runs thousands of them, and every one would be a node that the
    backward already taken has no use for.

    Elements are addressed by index rather than through `reshape(-1)`, which is
    a view only for a contiguous array and a silent copy otherwise -- and a copy
    would throw every perturbation away.
    """

    base = np.asarray(inputs[positions[0]], np.float64).copy()
    out = np.empty_like(base)

    probe = list(inputs)

    def evaluate() -> float:
        perturbed = Tensor(base, dtype="float64")
        for slot in positions:
            probe[slot] = perturbed
        return float(np.asarray(func(*probe), np.float64).reshape(()))

    with _C.no_grad():
        for index in np.ndindex(base.shape):
            original = base[index]

            base[index] = original + eps
            high = evaluate()

            base[index] = original - eps
            low = evaluate()

            base[index] = original
            out[index] = (high - low) / (2.0 * eps)

    return out


def _describe(index, analytic, numeric, atol, rtol) -> str:
    unusable = ~(np.isfinite(analytic) & np.isfinite(numeric))
    if unusable.any():
        # `np.allclose` refuses these, but every comparison against a NaN is
        # False, so the ordinary report would count zero elements as disagreeing
        # while still failing. Say what actually happened instead.
        #
        # The common cause is not a wrong backward but an input outside the
        # function's domain: `acos(1.4)` returns NaN rather than raising, so the
        # perturbed forwards are NaN and the difference of two of them is too.
        first = tuple(int(axis) for axis in np.argwhere(unusable)[0])
        return (
            f"gradient for input {index} could not be compared: "
            f"{int(unusable.sum())} of {unusable.size} elements are NaN or "
            f"infinite (first at index {first}).\n"
            f"  analytic {float(analytic[first])!r}\n"
            f"  numeric  {float(numeric[first])!r}\n"
            "  A non-finite numerical gradient usually means the input is "
            "outside the function's domain, where the perturbed forward is "
            "itself non-finite -- check the input before the backward."
        )

    difference = np.abs(analytic - numeric)
    allowed = atol + rtol * np.abs(numeric)
    worst = int(np.argmax(difference - allowed))
    return (
        f"gradient mismatch for input {index} at flat position {worst}:\n"
        f"  analytic {analytic.reshape(-1)[worst]!r}\n"
        f"  numeric  {numeric.reshape(-1)[worst]!r}\n"
        f"  absolute difference {difference.reshape(-1)[worst]:.3e}, "
        f"allowed {allowed.reshape(-1)[worst]:.3e}\n"
        f"  {int((difference > allowed).sum())} of {difference.size} elements disagree"
    )


__all__ = ["gradcheck"]
