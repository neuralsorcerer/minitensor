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
]

_RUNNER = """
import warnings
warnings.simplefilter("ignore")
import minitensor as mt
try:
    {expr}
    print("RETURNED")
except MemoryError:
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
