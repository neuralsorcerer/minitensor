# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""An exception raised in a custom operation's own code reaches the caller as
itself.

The engine's error type holds no Python objects, so an exception raised in a
forward or backward the engine called came back as a `ValueError` carrying
only its text: a `KeyError` in a backward lost its type and the traceback down
to the line that raised it. It is now raised as itself, with a note saying
which operation and which stage it came from.
"""

import traceback

import pytest

import minitensor as mt
from minitensor.autograd import Function


class _RaisesInBackward(Function):
    @staticmethod
    def forward(ctx, x):
        return x * 1

    @staticmethod
    def backward(ctx, grad):
        raise KeyError("missing entry")


class _RaisesInForward(Function):
    @staticmethod
    def forward(ctx, x):
        return {}["absent"]

    @staticmethod
    def backward(ctx, grad):
        return grad


def test_a_backward_exception_keeps_its_type_traceback_and_origin():
    x = mt.tensor([1.0, 3.0], requires_grad=True)
    with pytest.raises(KeyError, match="missing entry") as info:
        _RaisesInBackward.apply(x).sum().backward()
    assert (
        "raised in the backward of custom op '_RaisesInBackward'"
        in info.value.__notes__
    )
    frames = "".join(traceback.format_exception(info.value))
    assert 'raise KeyError("missing entry")' in frames


def test_a_forward_exception_keeps_its_type_and_origin():
    with pytest.raises(KeyError, match="absent") as info:
        _RaisesInForward.apply(mt.tensor([1.0], requires_grad=True))
    assert (
        "raised in the forward of custom op '_RaisesInForward'" in info.value.__notes__
    )


def test_a_registered_operation_reports_its_own_exception():
    def forward(x):
        raise ZeroDivisionError("by design")

    mt.register_custom_op("raises_by_design", forward)
    try:
        with pytest.raises(ZeroDivisionError, match="by design") as info:
            mt.execute_custom_op_py("raises_by_design", [mt.tensor([1.0])])
        assert any("raises_by_design" in note for note in info.value.__notes__)
    finally:
        mt.unregister_custom_op_py("raises_by_design")


def test_the_next_unrelated_error_is_its_own():
    with pytest.raises(KeyError):
        _RaisesInBackward.apply(mt.tensor([1.0], requires_grad=True)).sum().backward()
    with pytest.raises(ValueError, match="Shape mismatch"):
        mt.zeros(2) + mt.zeros(3)


def test_the_engines_own_checks_still_read_as_before():
    class WrongShape(Function):
        @staticmethod
        def forward(ctx, x):
            return x * 1

        @staticmethod
        def backward(ctx, grad):
            return mt.ones(5)

    with pytest.raises(ValueError, match="gradient of shape"):
        WrongShape.apply(mt.tensor([1.0, 2.0], requires_grad=True)).sum().backward()
