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
factor on one branch of a piecewise function, and in float64 it is around 1e-9,
which misses nothing. `gradcheck` refuses a float32 input rather than report a
pass it cannot stand behind.
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
    checked = [
        index
        for index, value in enumerate(inputs)
        if isinstance(value, Tensor) and value.requires_grad
    ]
    if not checked:
        raise ValueError(
            "gradcheck needs at least one input tensor with requires_grad=True"
        )
    for index in checked:
        if str(inputs[index].dtype) != "float64":
            raise ValueError(
                f"input {index} is {inputs[index].dtype}; gradcheck needs float64, "
                "because a central difference in float32 agrees to about 1e-2 and "
                "that is not evidence of anything"
            )

    analytic = _analytic(func, inputs, checked)
    for index in checked:
        numeric = _numeric(func, inputs, index, eps)
        got = analytic[index]
        if got is None:
            # No gradient reached this input at all, which is a claim of zero.
            got = np.zeros_like(numeric)
        if np.allclose(got, numeric, atol=atol, rtol=rtol):
            continue
        if not raise_exception:
            return False
        raise AssertionError(_describe(index, got, numeric, atol, rtol))
    return True


def _analytic(func, inputs, checked) -> dict[int, np.ndarray | None]:
    # `set_to_none=True`, not a zero fill: gradients accumulate across backward
    # passes here, so a stale one would be added to this one, and a cleared
    # `.grad` is what keeps "no gradient reached this input" distinguishable
    # from "the gradient is zero".
    for index in checked:
        inputs[index].zero_grad(True)

    output = func(*inputs)
    if output.numel() != 1:
        raise ValueError(
            f"func must return a scalar tensor; got shape {tuple(output.shape)}. "
            "Reduce it with .sum(): the check is equivalent and the gradient is "
            "the one you wanted."
        )
    output.backward()

    gradients: dict[int, np.ndarray | None] = {}
    for index in checked:
        grad = inputs[index].grad
        gradients[index] = None if grad is None else np.asarray(grad, np.float64).copy()
    return gradients


def _numeric(func, inputs, index, eps) -> np.ndarray:
    """One element at a time, with recording off.

    The perturbed forwards must not touch the tape: a gradcheck over a large
    input runs thousands of them, and every one would be a node that the
    backward already taken has no use for.
    """

    base = np.asarray(inputs[index], np.float64).copy()
    flat = base.reshape(-1)
    out = np.empty_like(flat)

    probe = list(inputs)
    with _C.no_grad():
        for i in range(flat.size):
            original = flat[i]

            flat[i] = original + eps
            probe[index] = Tensor(base, dtype="float64")
            high = float(np.asarray(func(*probe), np.float64).reshape(()))

            flat[i] = original - eps
            probe[index] = Tensor(base, dtype="float64")
            low = float(np.asarray(func(*probe), np.float64).reshape(()))

            flat[i] = original
            out[i] = (high - low) / (2.0 * eps)

    return out.reshape(base.shape)


def _describe(index, analytic, numeric, atol, rtol) -> str:
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
