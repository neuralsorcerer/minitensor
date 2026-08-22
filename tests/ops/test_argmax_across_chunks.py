# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`argmax` over a whole tensor now folds chunks, so chunks can disagree.

It used to be `par_iter().enumerate().reduce_with(..)`: one rayon work item per
element, folding `(index, value)` tuples through a closure rayon cannot inline.
Over two million float32 that took 0.71ms where `max` -- the same scan without
the index -- took 0.18ms. Folding a chunk at a time takes it to 0.55ms with the
comparison inlined into a plain loop.

The scan is per chunk now, so everything the answer depends on has to survive
being split across them: a tie whose halves land in different chunks still goes
to the lower index, and a NaN in a late chunk still beats a large value in an
early one. The chunk is 1024 elements, so the cases below are built around that
boundary rather than around round numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

CHUNK = 1024  # ops::map::PAR_CHUNK


def _flat(n, seed=3):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n).astype(np.float32)


@pytest.mark.parametrize("n", [1, 2, 1023, 1024, 1025, 2048, 100_000, 3_000_000])
@pytest.mark.parametrize("op", ["argmax", "argmin"])
def test_it_agrees_with_numpy(n, op):
    values = _flat(n)
    got = getattr(mt.Tensor(values, dtype="float32"), op)().item()
    assert got == int(getattr(values, op)())


@pytest.mark.parametrize(
    "first,second", [(0, CHUNK), (CHUNK - 1, CHUNK), (5, 3 * CHUNK + 7)]
)
def test_a_tie_split_across_chunks_goes_to_the_lower_index(first, second):
    """Both chunks report the same winning value; the combine has to prefer the
    earlier one however rayon happened to order the reduction."""
    values = np.full(4 * CHUNK, -1.0, dtype=np.float32)
    values[first] = 5.0
    values[second] = 5.0
    assert mt.Tensor(values, dtype="float32").argmax().item() == first

    values = np.full(4 * CHUNK, 1.0, dtype=np.float32)
    values[first] = -5.0
    values[second] = -5.0
    assert mt.Tensor(values, dtype="float32").argmin().item() == first


def test_a_tie_repeated_in_every_chunk_still_goes_to_the_first():
    values = np.zeros(16 * CHUNK, dtype=np.float32)
    assert mt.Tensor(values, dtype="float32").argmax().item() == 0
    assert mt.Tensor(values, dtype="float32").argmin().item() == 0


@pytest.mark.parametrize("nan_at", [0, CHUNK, CHUNK + 1, 7 * CHUNK + 3])
def test_a_nan_anywhere_beats_every_real_value(nan_at):
    """A NaN wins outright, so a chunk holding one must beat a chunk holding
    the largest real value -- including when the NaN is in the later chunk."""
    values = _flat(8 * CHUNK, seed=5)
    values[100] = 1e30  # a huge real value in an early chunk
    values[nan_at] = np.nan
    assert mt.Tensor(values, dtype="float32").argmax().item() == nan_at
    assert mt.Tensor(values, dtype="float32").argmin().item() == nan_at


def test_the_first_of_several_nans_wins():
    """Two of them share the first chunk, which is what distinguishes reporting
    a chunk's first NaN from reporting its last: both answers survive the
    combine across chunks, and only the per-chunk scan can tell them apart."""
    values = _flat(8 * CHUNK, seed=7)
    for i in (3 * CHUNK, 5, 900, 6 * CHUNK + 900, 5000):
        values[i] = np.nan
    assert mt.Tensor(values, dtype="float32").argmax().item() == 5
    assert mt.Tensor(values, dtype="float32").argmin().item() == 5


def test_all_nan_reports_the_first_index():
    values = np.full(4 * CHUNK, np.nan, dtype=np.float32)
    assert mt.Tensor(values, dtype="float32").argmax().item() == 0
    assert mt.Tensor(values, dtype="float32").argmin().item() == 0


@pytest.mark.parametrize("dtype", ["float64", "int32", "int64"])
@pytest.mark.parametrize("op", ["argmax", "argmin"])
def test_the_other_dtypes_take_the_same_fold(dtype, op):
    rng = np.random.default_rng(11)
    if dtype == "float64":
        values = rng.standard_normal(100_000)
    else:
        values = rng.integers(-(10**6), 10**6, size=100_000).astype(dtype)
    got = getattr(mt.Tensor(values, dtype=dtype), op)().item()
    assert got == int(getattr(values, op)())


@pytest.mark.parametrize("op", ["argmax", "argmin"])
def test_integer_ties_across_chunks(op):
    """The integer arms compile the NaN test away entirely, so the tie rule is
    all that is left of the fold -- worth its own case."""
    values = np.zeros(8 * CHUNK, dtype=np.int64)
    target = 1 if op == "argmax" else -1
    values[3 * CHUNK] = target
    values[6 * CHUNK] = target
    assert getattr(mt.Tensor(values, dtype="int64"), op)().item() == 3 * CHUNK


@pytest.mark.parametrize("op", ["argmax", "argmin"])
def test_bool_is_unchanged(op):
    """Bool never went through the fold -- it short-circuits to the first
    `true` or the first `false` -- so this is a guard, not a change."""
    values = np.zeros(4 * CHUNK, dtype=bool)
    values[2000] = True
    got = getattr(mt.Tensor(values, dtype="bool"), op)().item()
    assert got == (2000 if op == "argmax" else 0)


def test_it_is_reproducible():
    """The combine is order-independent, so repeated calls -- which rayon may
    split differently -- must agree."""
    values = _flat(3_000_000, seed=13)
    t = mt.Tensor(values, dtype="float32")
    first = t.argmax().item()
    assert all(t.argmax().item() == first for _ in range(20))
