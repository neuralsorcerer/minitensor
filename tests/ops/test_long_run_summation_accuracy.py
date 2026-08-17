# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Summing the same numbers two ways has to give the same accuracy.

`t.sum()` and `t.sum(dim=0)` on a 1-D tensor are the same arithmetic over the
same values, and they were not equally accurate. The whole-tensor reduction
folds 8192-element partials pairwise, so its error grows like `log n`; the
along-a-dimension form walked the run with a single vectorized accumulator,
whose eight lanes leave the error growing like `n`. On 4M positive float32
values the axis form was 613 times less accurate -- 4.4e-6 against 7.1e-9 --
and nothing in the suite compared them.

It is not only the 1-D case. A reduction over a 2-D tensor's rows takes the
same path per row, so one very wide row has the same problem; that is exactly
the shape `norm(dim=1)` and any per-sample statistic over a long feature axis
produce.

The fix routes both through the same chunked pairwise accumulation, so the
tests below are equality-of-accuracy tests rather than absolute-tolerance ones:
what matters is that the two spellings do not diverge, not that either hits a
particular number.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

# All-positive data is the worst case: nothing cancels, so every rounding error
# accumulates in the same direction instead of partly undoing the last one.
LENGTHS = [8192, 8193, 100_000, 1_000_000]


def _positive(n, dtype="float32"):
    rng = np.random.default_rng(1234)
    return (rng.random(n) + 0.5).astype(dtype)


def _relative_error(got, exact):
    return abs(float(got) - exact) / abs(exact)


@pytest.mark.parametrize("n", LENGTHS)
def test_sum_and_sum_along_the_only_axis_are_equally_accurate(n):
    values = _positive(n)
    exact = float(values.astype(np.float64).sum())
    t = mt.Tensor(values, dtype="float32")

    whole = _relative_error(t.sum().item(), exact)
    axis = _relative_error(t.sum(0).item(), exact)

    # Equal, not merely close: both now accumulate the same way, so the only
    # licence here is for a different chunk *count* to land differently.
    assert axis <= max(whole * 4.0, 1e-9), f"axis {axis:.3e} vs whole {whole:.3e}"


@pytest.mark.parametrize("n", LENGTHS)
def test_a_single_wide_row_is_as_accurate_as_the_flat_reduction(n):
    """`sum(dim=1)` over one row of `n` is the same numbers as `sum()`."""
    values = _positive(n).reshape(1, n)
    exact = float(values.astype(np.float64).sum())
    t = mt.Tensor(values, dtype="float32")

    row = _relative_error(t.sum(1).numpy()[0], exact)
    whole = _relative_error(t.sum().item(), exact)
    assert row <= max(whole * 4.0, 1e-9), f"row {row:.3e} vs whole {whole:.3e}"


@pytest.mark.parametrize("n", [100_000, 1_000_000])
def test_the_long_run_sum_beats_a_naive_accumulator(n):
    """The property the chunking exists for: error must not grow like `n`.

    A running float32 total over the same data is the thing being improved on,
    so it is computed here rather than assumed."""
    values = _positive(n)
    exact = float(values.astype(np.float64).sum())

    naive = np.float32(0.0)
    for v in values[:20000]:  # a prefix is enough to show the growth
        naive = np.float32(naive + v)
    naive_error = _relative_error(naive, float(values[:20000].astype(np.float64).sum()))

    ours = _relative_error(mt.Tensor(values, dtype="float32").sum(0).item(), exact)
    assert ours < naive_error, f"ours {ours:.3e} vs naive-prefix {naive_error:.3e}"


@pytest.mark.parametrize("n", [100_000, 2_000_000])
def test_norm_is_at_least_as_accurate_as_numpy(n):
    """`norm` inherits the run accumulation, and it used to be far worse than
    NumPy here: 2.3e-5 against 8.1e-7 at 2M elements, because it summed the
    scaled squares with the same lane accumulator."""
    rng = np.random.default_rng(7)
    values = ((rng.random(n) - 0.5) * 10).astype(np.float32)
    exact = float(np.linalg.norm(values.astype(np.float64)))

    ours = _relative_error(mt.Tensor(values, dtype="float32").norm().item(), exact)
    theirs = _relative_error(np.linalg.norm(values), exact)
    assert ours <= theirs, f"ours {ours:.3e} vs numpy {theirs:.3e}"


@pytest.mark.parametrize("n", [100_000, 1_000_000])
def test_nansum_matches_sum_when_nothing_is_nan(n):
    """`nansum`'s run accumulator was a bare `iter().sum()` -- one rounding per
    element, so worse than the plain sum it should agree with."""
    values = _positive(n)
    t = mt.Tensor(values, dtype="float32")
    assert t.nansum().item() == t.sum().item()
    assert t.nansum(0).item() == t.sum(0).item()


def test_nansum_still_skips_nan_after_the_change():
    values = np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float32)
    t = mt.Tensor(values, dtype="float32")
    assert t.nansum().item() == pytest.approx(9.0)
    assert t.nansum(0).item() == pytest.approx(9.0)


@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_integer_runs_are_untouched(dtype):
    """Integer accumulation is exact and wraps deliberately; the chunking must
    not change what it produces."""
    values = np.arange(1, 100_001, dtype=dtype)
    t = mt.Tensor(values, dtype=dtype)
    expected = int(values.astype(np.int64).sum())
    assert t.sum().item() == expected
    assert t.sum(0).item() == expected
