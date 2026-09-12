# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Defining an operation the library does not have, in Python.

`register_custom_op` files a forward and a backward under a name, which is what
you want for an operation the whole program reaches for. `Function` is for the
other case: an operation belonging to one call site, whose backward needs to
remember something from its forward.

    class Clamp(Function):
        @staticmethod
        def forward(ctx, x, low, high):
            ctx.save_for_backward(x)
            ctx.low, ctx.high = low, high
            return x.clip(low, high)

        @staticmethod
        def backward(ctx, grad_output):
            (x,) = ctx.saved_tensors
            inside = (x > ctx.low) & (x < ctx.high)
            return grad_output * inside.astype(str(grad_output.dtype)), None, None

    y = Clamp.apply(x, -1.0, 1.0)

Each `apply` builds its own node, so each has its own `ctx` -- which is what
lets the backward above read the `x` and the bounds its own forward saw, rather
than whichever call ran last. The node carries no name and is in no registry:
naming an operation used once would be a global string invented to be unique,
and a lookup on every call to get back what the caller is already holding.

A subclass that writes no `backward` gets the other of `register_custom_op`'s
two modes: the forward is recorded and the operation differentiates by
composition. Writing one switches the forward to running with recording off and
makes the gradient whatever the backward returns.
"""

from __future__ import annotations

from typing import Any, ClassVar

from . import _core as _C

__all__ = ["Function", "FunctionCtx"]

Tensor = _C.Tensor


def _is_tensor(value: object) -> bool:
    """Whether `value` is a tensor, however it is wrapped."""

    return isinstance(value, Tensor) or isinstance(
        getattr(value, "_tensor", None), Tensor
    )


class FunctionCtx:
    """What a `Function`'s forward leaves for its backward.

    Anything may be set on it as an attribute; `save_for_backward` is separate
    only because the tensors it holds are the ones the backward is most likely
    to want and the ones worth naming in an error when it forgets to save them.
    """

    # Deliberately no `__slots__`: a forward stashes whatever its backward
    # will need, and the bounds of a clamp or the axis of a reduction are as
    # much a part of that as the tensors are. One of these exists per graph
    # node, beside the tensors it keeps alive, so the dict costs nothing worth
    # the restriction.

    def __init__(self, needs_input_grad: tuple[bool, ...]) -> None:
        self._saved: tuple[Any, ...] | None = None
        #: One flag per positional argument of `apply`, `True` where that
        #: argument is a tensor that wants a gradient. A backward may use it to
        #: skip work whose result would be discarded.
        self.needs_input_grad = needs_input_grad

    def save_for_backward(self, *tensors: Any) -> None:
        """Keep `tensors` for the backward pass.

        They are stored, not copied: a tensor shares its storage, so this costs
        nothing beyond keeping the buffer alive until the graph is released.
        """

        self._saved = tensors

    @property
    def saved_tensors(self) -> tuple[Any, ...]:
        if self._saved is None:
            raise RuntimeError(
                "saved_tensors is empty: the forward never called "
                "ctx.save_for_backward(...)"
            )
        return self._saved

    def __repr__(self) -> str:
        saved = 0 if self._saved is None else len(self._saved)
        return f"FunctionCtx(saved_tensors={saved}, needs_input_grad={self.needs_input_grad})"


class Function:
    """Base class for an operation defined at one call site.

    Subclasses define `forward(ctx, *args)` as a static method, optionally a
    static `backward(ctx, grad_output)`, and are used through `apply`.

    With a `backward`, the forward runs with gradient recording *off*: what it
    does internally is an implementation detail, and recording it as well would
    put a second path to the same gradient in the graph for the two to add. The
    gradient is whatever `backward` returns, which is the only way to write a
    straight-through estimator, or any operation whose useful derivative is not
    its true one.

    Without one, the forward is recorded and the operation differentiates by
    composition -- the same two modes `register_custom_op` offers, and picked
    the same way. A subclass in that mode has no use for `ctx`; it is still
    worth writing as a `Function` for the argument handling and for the one
    place a `backward` can later be added.

    `apply` takes tensors and non-tensors alike and hands all of them to
    `forward`. Only the tensors become graph inputs, so `backward` returns one
    gradient per argument with `None` in the non-tensor positions, the way it
    reads at the call site. A tensor position may be `None` too, meaning no
    gradient flows there.

    One output. An operation returning several is expressible as several
    operations, and the graph node this compiles to carries one tensor.
    """

    #: Filled in by `apply`; a subclass has no reason to touch it.
    _op_name: ClassVar[str | None] = None

    @staticmethod
    def forward(ctx: FunctionCtx, *args: Any) -> Any:
        raise NotImplementedError(
            "a Function subclass must define a static forward(ctx, *args)"
        )

    #: Left undefined on purpose. `apply` compares against this to decide which
    #: of the two modes the subclass asked for, so overriding it is the whole
    #: signal -- there is no flag to set and nothing to call up to.
    backward: ClassVar[Any] = None

    @classmethod
    def apply(cls, *args: Any) -> Any:
        """Run the operation and put its node on the graph."""

        if cls.forward is Function.forward:
            raise TypeError(
                f"{cls.__name__} defines no forward; a Function subclass must "
                "define a static forward(ctx, *args)"
            )

        positions = [index for index, value in enumerate(args) if _is_tensor(value)]
        if not positions:
            raise TypeError(
                f"{cls.__name__}.apply needs at least one tensor argument; "
                f"got {len(args)} argument(s), none of them tensors"
            )

        ctx = FunctionCtx(
            tuple(bool(getattr(value, "requires_grad", False)) for value in args)
        )

        def _forward(*tensors: Any) -> Any:
            # The engine hands back the tensor arguments in order; the
            # non-tensor ones it never saw, so they come from the call.
            merged = list(args)
            for slot, tensor in zip(positions, tensors):
                merged[slot] = tensor
            return cls.forward(ctx, *merged)

        def _backward(grad_output: Any, _inputs: Any, _output: Any) -> Any:
            produced = cls.backward(ctx, grad_output)
            if _is_tensor(produced) or produced is None:
                produced = (produced,)
            gradients = tuple(produced)
            # A backward may answer for every argument, as it is written at the
            # call site, or for the tensors alone. Both are unambiguous.
            if len(gradients) == len(args):
                return tuple(gradients[slot] for slot in positions)
            if len(gradients) == len(positions):
                return gradients
            raise ValueError(
                f"{cls.__name__}.backward returned {len(gradients)} gradient(s); "
                f"expected {len(positions)} (one per tensor argument) or "
                f"{len(args)} (one per argument)"
            )

        handle = _C.build_custom_op(
            cls._op_name or cls.__name__,
            _forward,
            _backward if cls.backward is not None else None,
            len(positions),
        )
        return handle.apply(tuple(args[slot] for slot in positions))
