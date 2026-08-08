# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A rejected `t[::-1]` says which axis, and what to write instead.

Reversing by subscript is not supported, which is also PyTorch's position --
`flip` is the operation for it. But the refusal used to read

    IndexError: slice step must be positive

which states the rule and leaves the caller to find the remedy. It does not say
which axis of a multi-axis subscript was at fault, and it does not mention
`flip` at all, so someone arriving from NumPy with `x[::-1]` has to go looking.

The message now names the axis and spells out the equivalent call, including
the positive stride that a step other than `-1` still needs: `x[::-2]` is
`x.flip(0)[::2]`, not `x.flip(0)`.

The test that matters here is `test_the_suggested_call_is_the_right_one`. A
suggestion that does not actually reproduce what the caller asked for is worse
than none, so it is executed against NumPy's own negative-step slice rather
than eyeballed.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

import minitensor as mt

VALUES = np.arange(24, dtype=np.float64).reshape(2, 3, 4)


def _tensor():
    return mt.Tensor(VALUES, dtype="float64")


def _message(subscript):
    with pytest.raises(IndexError) as excinfo:
        subscript(_tensor())
    return str(excinfo.value)


@pytest.mark.parametrize(
    "subscript,axis",
    [
        (lambda t: t[::-1], 0),
        (lambda t: t[:, ::-1], 1),
        (lambda t: t[:, :, ::-1], 2),
        (lambda t: t[..., ::-1], 2),
        (lambda t: t[0, ::-1], 1),
    ],
    ids=["axis0", "axis1", "axis2", "ellipsis", "after_int"],
)
def test_the_message_names_the_axis_at_fault(subscript, axis):
    """A subscript with several entries has to say which one was rejected."""
    message = _message(subscript)
    assert f"axis {axis}" in message, message
    assert f"flip({axis})" in message, message


@pytest.mark.parametrize("step", [-1, -2, -3, -5])
def test_the_message_reports_the_step_that_was_written(step):
    message = _message(lambda t: t[::step])
    assert f"step of {step}" in message, message


@pytest.mark.parametrize(
    "axis,step",
    [(0, -1), (0, -2), (1, -1), (1, -3), (2, -1), (2, -2)],
)
def test_the_suggested_call_is_the_right_one(axis, step):
    """Execute the suggestion and compare it against the slice it stands in
    for. A suggestion that does not reproduce the request is worse than none."""
    subscript = [slice(None)] * 3
    subscript[axis] = slice(None, None, step)
    message = _message(lambda t: t[tuple(subscript)])

    suggestion = re.search(r"`x\.(flip\(\d+\)(?:\[[:, ]*::\d+\])?)`", message)
    assert suggestion is not None, f"no suggestion found in: {message}"

    got = eval(f"tensor.{suggestion.group(1)}", {"tensor": _tensor()})
    np.testing.assert_array_equal(got.numpy(), VALUES[tuple(subscript)])


def test_a_step_of_minus_one_needs_no_stride_afterwards():
    """`flip` alone is the whole answer there; suggesting `[::1]` would be
    noise."""
    assert "flip(0)`" in _message(lambda t: t[::-1])


def test_a_larger_step_keeps_its_stride():
    assert "flip(0)[::2]" in _message(lambda t: t[::-2])
    # and on a later axis the stride has to land on that axis, not the first
    assert "flip(1)[:, ::2]" in _message(lambda t: t[:, ::-2])
    assert "flip(2)[:, :, ::3]" in _message(lambda t: t[:, :, ::-3])


def test_the_old_wording_is_gone():
    """It stated the rule and nothing else."""
    message = _message(lambda t: t[::-1])
    assert "slice step must be positive" not in message
    assert "flip" in message


def test_a_zero_step_is_still_its_own_error():
    """Zero is not a direction, so `flip` is not the answer for it."""
    with pytest.raises(Exception) as excinfo:
        _tensor()[::0]
    assert "zero" in str(excinfo.value)
    assert "flip" not in str(excinfo.value)


def test_positive_steps_are_untouched():
    tensor = _tensor()
    for subscript in [
        (slice(None, None, 1),),
        (slice(None, None, 2),),
        (slice(1, None, 3),),
        (slice(None), slice(None, None, 2)),
    ]:
        np.testing.assert_array_equal(
            tensor[subscript].numpy(), VALUES[subscript], err_msg=str(subscript)
        )
