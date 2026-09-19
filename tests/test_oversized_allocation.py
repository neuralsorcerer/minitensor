# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A size this machine cannot allocate is refused, not fatal.

`test_no_panics_on_bad_input.py` holds the line that a rejection arrives as an
`Exception` rather than a Rust panic. This holds a line below that one. Rust's
`Vec` allocation calls `handle_alloc_error` when the allocator says no, and
that **aborts the process** -- not an exception, not even the
`PanicException` the other file guards against, just the interpreter gone and
whatever was in it with it. Every constructor that takes a size from the caller
reached it:

    >>> mt.zeros([10**18])
    memory allocation of 4000000000000000000 bytes failed

One mistyped exponent in a notebook took the kernel down, and no `except`
clause could have stopped it.

The count alone was already checked -- `reject_overflowing_shape` refuses a
shape whose dimensions multiply past `usize`. What was missing is that a count
well inside `usize` can still ask for more bytes than exist. 10**18 is a
perfectly representable number of elements and 3.5 EiB of memory.

These tests cannot assert the *absence* of an abort from inside the process
that would be aborted, so each one runs in a subprocess: a killed child is
distinguishable from a child that raised, which is exactly the difference being
tested.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# Every constructor that takes a size from the caller. 10**18 elements is far
# beyond any machine, and `eye` squares its argument to get there.
OVERSIZED = [
    "mt.zeros([10**18])",
    "mt.ones([10**18])",
    "mt.empty([10**18])",
    "mt.full([10**18], 1.0)",
    "mt.rand([10**18])",
    "mt.randn([10**18])",
    "mt.arange(10**18)",
    "mt.linspace(0.0, 1.0, 10**18)",
    "mt.logspace(0.0, 1.0, 10**18)",
    "mt.eye(10**9)",
    "mt.xavier_uniform([10**18])",
    "mt.Tensor.zeros(10**18)",
    "mt.Tensor.arange(10**18)",
    # Methods that take a size from the caller rather than a constructor.
    "mt.Tensor([1.0, 2.0]).new_empty(10**18)",
    "mt.Tensor([1.0, 2.0]).new_zeros(10**18)",
    "mt.Tensor([1.0, 2.0]).new_ones(10**18)",
    "mt.Tensor([1.0, 2.0]).new_full(10**18, 1.0)",
    # `repeat` reaches the same place by multiplying rather than by taking a
    # size: a two-element tensor repeated 10**9 by 10**9.
    "mt.Tensor([1.0, 2.0]).repeat(10**9, 10**9)",
    # `expand` materialises rather than returning a strided view, so the shape
    # asked for is the shape allocated.
    "mt.zeros([1, 1]).expand(10**6, 10**6)",
]

ORDINARY = [
    "mt.zeros([2, 3])",
    "mt.ones([4])",
    "mt.arange(10)",
    "mt.eye(4)",
    "mt.linspace(0.0, 1.0, 5)",
    "mt.full([2, 2], 7.0)",
    # Large enough to be worth allocating, small enough that it must succeed.
    "mt.zeros([64, 1024, 1024])",
    "mt.Tensor([1.0, 2.0]).new_zeros([3, 4])",
    "mt.Tensor([1.0, 2.0]).repeat(2, 3)",
    "mt.zeros([1, 3]).expand(4, 3)",
    "mt.zeros([1, 3]).expand(2, -1)",
]

_RUNNER = """
import warnings
warnings.simplefilter("ignore")
import minitensor as mt
try:
    {expr}
    print("RETURNED")
except (MemoryError, ValueError):
    # ValueError when the byte count itself overflows the address space, which
    # `repeat` reaches before the allocator is ever asked.
    print("MEMORYERROR")
except Exception as exc:
    print("OTHER:" + type(exc).__name__)
"""


def _run(expr):
    """Run one expression in its own interpreter; report how it ended."""
    finished = subprocess.run(
        [sys.executable, "-c", _RUNNER.format(expr=expr)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    return finished.returncode, finished.stdout.strip()


@pytest.mark.parametrize("expr", OVERSIZED)
def test_an_impossible_size_raises_instead_of_killing_the_process(expr):
    returncode, out = _run(expr)

    assert returncode == 0, (
        f"{expr} ended the interpreter with code {returncode} rather than "
        f"raising; an aborted allocation takes the whole process with it"
    )
    assert out == "MEMORYERROR", f"{expr} ended with {out!r}, expected MemoryError"


@pytest.mark.parametrize("expr", ORDINARY)
def test_a_size_that_fits_is_still_built(expr):
    """The guard must not have bought safety by refusing ordinary work."""
    returncode, out = _run(expr)
    assert (
        returncode == 0 and out == "RETURNED"
    ), f"{expr} gave {out!r} (code {returncode})"


# --- layers, whose parameters are sized by their constructor arguments --------

OVERSIZED_LAYERS = [
    "mt.nn.DenseLayer(10**12, 10**12)",
    "mt.nn.Embedding(10**12, 10**12)",
    "mt.nn.LayerNorm(10**18)",
    "mt.nn.Conv2d(10**6, 10**6, 3)",
    "mt.nn.LSTM(10**9, 10**9)",
    "mt.nn.GRU(10**9, 10**9)",
]

ORDINARY_LAYERS = [
    "mt.nn.DenseLayer(4, 3)",
    "mt.nn.Embedding(10, 4)",
    "mt.nn.LayerNorm(8)",
    "mt.nn.Conv2d(3, 8, 3)",
    "mt.nn.LSTM(4, 8)",
]


@pytest.mark.parametrize("expr", OVERSIZED_LAYERS)
def test_a_layer_too_large_to_build_says_so(expr):
    """Both failure modes lived here: a shape whose element count overflows
    `usize` (panic inside `Shape::numel`) and one that fits but cannot be
    allocated (abort inside the allocator). Every layer's parameters are built
    through one function, so one check covers all of them."""
    returncode, out = _run(expr)
    assert (
        returncode == 0
    ), f"{expr} ended the interpreter with code {returncode} rather than raising"
    assert out == "MEMORYERROR", f"{expr} ended with {out!r}"


@pytest.mark.parametrize("expr", ORDINARY_LAYERS)
def test_an_ordinary_layer_is_still_built(expr):
    returncode, out = _run(expr)
    assert (
        returncode == 0 and out == "RETURNED"
    ), f"{expr} gave {out!r} (code {returncode})"


# --- results too large, where the size is implied by the operation ----------
#
# These are the cases the reference used to say simply abort. The allocation
# happens inside the engine, so each is checked where its output shape is
# computed rather than where an argument is parsed.

OVERSIZED_RESULTS = [
    "mt.zeros([10**5, 10]).matmul(mt.zeros([10, 10**5]))",
    "mt.zeros([10**5, 10]).mm(mt.zeros([10, 10**5]))",
    "mt.cat([mt.zeros([10**9]) for _ in range(20)])",
    "mt.stack([mt.zeros([10**9]) for _ in range(20)])",
    "mt.vstack([mt.zeros([10**9]) for _ in range(20)])",
    "mt.hstack([mt.zeros([10**9]) for _ in range(20)])",
    "mt.functional.pad(mt.zeros([4]), [10**10, 10**10])",
    "mt.functional.interpolate(mt.zeros([1, 1, 2, 2]), [10**6, 10**6])",
    "mt.functional.conv_transpose2d(mt.zeros([1,1,2,2]), mt.zeros([1,1,2,2]), None, 10**5)",
    # A broadcast is the one shape operation whose result can dwarf both
    # inputs: two megabytes in, a trillion elements out.
    "mt.zeros([10**6, 1]) * mt.zeros([1, 10**6])",
    "mt.zeros([10**6, 1]) + mt.zeros([1, 10**6])",
    "mt.kron(mt.zeros([10**6]), mt.zeros([10**6]))",
    # The same broadcast at another dtype, and through a method rather than an
    # operator. The sizes here are beyond every machine on purpose: that the
    # refusal is computed from bytes rather than from the element count is
    # pinned exactly in `storage.rs`, because the size at which a particular
    # host starts saying no is a property of the host. A 20 GB reservation
    # Linux declines, a Windows runner with a large page file grants.
    "mt.zeros([10**6,1],dtype='float64') * mt.zeros([1,10**6],dtype='float64')",
    "mt.zeros([10**6,1],dtype='float64') + mt.zeros([1,10**6],dtype='float64')",
    "mt.zeros([10**6,1],dtype='float64').maximum(mt.zeros([1,10**6],dtype='float64'))",
]

ORDINARY_RESULTS = [
    "mt.zeros([4, 3]).matmul(mt.zeros([3, 2]))",
    "mt.cat([mt.zeros([4]), mt.zeros([3])])",
    "mt.zeros([3, 1]) * mt.zeros([1, 4])",
    "mt.kron(mt.zeros([2]), mt.zeros([3]))",
    "mt.functional.interpolate(mt.zeros([1, 1, 2, 2]), [4, 4])",
    "mt.zeros([2, 3], dtype='float64') + mt.zeros([3], dtype='float64')",
]


@pytest.mark.parametrize("expr", OVERSIZED_RESULTS)
def test_a_result_too_large_to_hold_is_refused(expr):
    returncode, out = _run(expr)
    assert (
        returncode == 0
    ), f"{expr} ended the interpreter with code {returncode} rather than raising"
    assert out == "MEMORYERROR", f"{expr} ended with {out!r}"


@pytest.mark.parametrize("expr", ORDINARY_RESULTS)
def test_an_ordinary_result_is_still_computed(expr):
    returncode, out = _run(expr)
    assert (
        returncode == 0 and out == "RETURNED"
    ), f"{expr} gave {out!r} (code {returncode})"


def test_one_hot_declines_a_label_that_would_set_an_impossible_width():
    """`one_hot` infers its class count as the largest label plus one, so a
    single large label decides the width of the whole result."""
    returncode, out = _run("mt.functional.one_hot(10**18)")
    assert returncode == 0, "one_hot ended the interpreter rather than raising"
    assert out == "MEMORYERROR", f"one_hot ended with {out!r}"
