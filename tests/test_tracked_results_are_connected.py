# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A result that says it is tracked has to be tracked.

`requires_grad` is a claim about a tensor, and `grad_fn` is what makes the
claim true. A result carrying the first without the second is not an error
anywhere: `backward()` reaches it, finds no edge to follow, treats it as a
leaf and stops. Nothing raises, nothing warns, and the gradient that should
have gone to the input simply does not arrive.

That is not hypothetical. `quantile(x, [0.1, 0.9])` did exactly this: the
batched kernel reads every probability out of one sorted pass and attached no
backward, so a pinball loss over several quantiles trained nothing and said
nothing about it. The fix is elsewhere; this is the sweep that would have
found it, run over the whole surface rather than over the one op somebody
happened to check.

Two properties, each swept rather than listed:

  * With a tracked leaf as its first argument, an op whose float result claims
    `requires_grad` has to send a gradient back to that leaf.
  * Under `no_grad`, nothing an op *computes* may claim to be tracked.

Both have principled exceptions, and naming them is most of the value here.
`zeros_like` and its family take a tensor and return a *new leaf* shaped like
it -- inheriting `requires_grad` is this library's documented choice, and the
values do not depend on the input, so no gradient is owed. `conj`, `real` and
`block` hand back the argument itself when there is nothing to do, and
`as_tensor` hands back the same tensor under a new wrapper; an identity that
returns what it was given cannot strip a flag it never set.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import minitensor as mt

RNG = np.random.default_rng(20260920)
# Positive and away from zero, so `log`, `sqrt`, `acosh`-style domains and the
# reciprocals are all defined and none of the gradients is an exact zero that
# could hide a missing edge.
BASE = np.abs(RNG.standard_normal((4, 4))) + 0.5


def _leaf():
    return mt.Tensor(BASE.tolist(), dtype="float64", requires_grad=True)


def _operand(seed):
    values = np.abs(np.random.default_rng(seed).standard_normal((4, 4))) + 0.5
    return mt.Tensor(values.tolist(), dtype="float64")


# A new leaf shaped like its argument, whose values do not come from it.
_CONSTRUCTORS = (
    "empty_like",
    "ones_like",
    "zeros_like",
    "rand_like",
    "randn_like",
    "uniform_like",
    "he_normal_like",
    "he_uniform_like",
    "lecun_normal_like",
    "lecun_uniform_like",
    "truncated_normal_like",
    "xavier_normal_like",
    "xavier_uniform_like",
)

# Hands back what it was given when there is nothing to do.
_IDENTITIES = ("as_tensor", "block", "conj", "real")


def _callable_ops():
    """Ops reachable with one tracked tensor and at most one other operand."""
    found = {}
    for module, prefix in ((mt, ""), (mt.functional, "functional.")):
        for name in sorted(dir(module)):
            if name.startswith("_"):
                continue
            function = getattr(module, name)
            if not callable(function) or inspect.isclass(function):
                continue
            try:
                parameters = inspect.signature(function).parameters
            except (ValueError, TypeError):
                continue
            required = [
                parameter
                for parameter in parameters.values()
                if parameter.default is inspect.Parameter.empty
                and parameter.kind
                in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
            ]
            if not required or len(required) > 2:
                continue
            found[prefix + name] = (function, len(required))
    return found


def _tensors_in(result):
    if isinstance(result, mt.Tensor):
        return [result]
    if isinstance(result, (tuple, list)):
        return [item for item in result if isinstance(item, mt.Tensor)]
    return []


def test_a_result_that_claims_to_be_tracked_sends_a_gradient_back():
    mt.manual_seed(20260920)
    unreached, reached, uncallable = [], 0, 0

    for name, (function, arity) in _callable_ops().items():
        if name.split(".")[-1] in _CONSTRUCTORS:
            continue
        leaf = _leaf()
        try:
            result = function(leaf, *[_operand(arity) for _ in range(arity - 1)])
        except Exception:  # noqa: BLE001 - not every op takes this shape
            uncallable += 1
            continue

        tracked = [
            tensor
            for tensor in _tensors_in(result)
            if tensor.requires_grad and "float" in str(tensor.dtype)
        ]
        if not tracked:
            continue

        total = None
        for tensor in tracked:
            piece = tensor.sum()
            total = piece if total is None else total + piece
        try:
            total.backward()
        except Exception as error:  # noqa: BLE001 - reported below
            unreached.append(f"{name}: backward raised {type(error).__name__}: {error}")
            continue

        if leaf.grad is None:
            unreached.append(
                f"{name}: result claims requires_grad, input got no gradient"
            )
        else:
            reached += 1

    assert not unreached, "\n".join(unreached)
    assert reached > 300, f"only {reached} ops reached -- the sweep broke"


def test_a_constructor_shaped_like_its_argument_is_a_leaf():
    """The exception, stated rather than skipped: these inherit the flag and
    owe no gradient, because none of their values came from the input."""
    for name in _CONSTRUCTORS:
        function = getattr(mt, name, None)
        if function is None:
            continue
        leaf = _leaf()
        built = function(leaf)

        assert built.requires_grad, f"{name} dropped the flag it inherits"
        built.sum().backward()
        assert leaf.grad is None, f"{name} sent a gradient to an input it ignored"
        assert built.grad is not None, f"{name} is a leaf and should collect its own"


def test_no_grad_leaves_nothing_claiming_to_be_tracked():
    mt.manual_seed(20260920)
    claimed, checked = [], 0

    for name, (function, arity) in _callable_ops().items():
        leaf = _leaf()
        with mt.no_grad():
            try:
                result = function(leaf, *[_operand(arity) for _ in range(arity - 1)])
            except Exception:  # noqa: BLE001 - not every op takes this shape
                continue
            tensors = _tensors_in(result)
            if not tensors:
                continue
            checked += 1
            for tensor in tensors:
                if not tensor.requires_grad:
                    continue
                if (
                    name.split(".")[-1] in _IDENTITIES
                    or name.split(".")[-1] in _CONSTRUCTORS
                ):
                    continue
                claimed.append(name)
                break

    assert (
        not claimed
    ), "these computed a result under no_grad and called it tracked: " + ", ".join(
        sorted(set(claimed))
    )
    assert checked > 450, f"only {checked} ops reached -- the sweep broke"


@pytest.mark.parametrize("name", _IDENTITIES)
def test_an_identity_hands_back_what_it_was_given(name):
    """Why these are allowed to keep the flag under `no_grad`: they return the
    argument, or the same tensor under another wrapper, rather than computing
    anything. `as_tensor` is the second kind -- a different Python object over
    the same storage and the same place on the tape, so a gradient through it
    still reaches the original."""
    leaf = _leaf()
    with mt.no_grad():
        passed = getattr(mt, name)(leaf)

    if passed is not leaf:
        assert name == "as_tensor", f"{name} built something new"
        assert (
            np.asarray(passed).__array_interface__["data"][0]
            == np.asarray(leaf).__array_interface__["data"][0]
        )
        (passed * _operand(1)).sum().backward()
        assert leaf.grad is not None, "a gradient through as_tensor missed the original"
