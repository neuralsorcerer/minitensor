# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The kernel every reduction bottoms out in has to fold like NumPy's.

`simd_sum_f32` walked a slice with eight accumulator lanes. The lanes are what
let it vectorize, and they divide the error by eight -- but eight lanes over
8192 elements is still a run of 1024 additions per lane, and the error of a run
that deep grows with its length where a pairwise fold's grows like `log n`.
Averaged over 40 draws it was 2.96 times NumPy's error at 8192 elements and
1.83 times at 1024.

Those are single-ulp numbers, and no one test case shows them: a float32 sum
lands within an ulp or two of the truth either way, so which of two orderings
looks better on any one input is mostly which draw rounded which direction.
The average over many draws is what separates them, and it is what the tests
here measure.

It still mattered, for two reasons. Everything reduces through this kernel --
`sum`, `mean`, `nansum`, `norm`, the per-row sums, the blocked run sums built
on top of it -- and the error was the kind that grows with the size of the
tensor rather than staying put.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

# Around and across the 128-element leaf, and far past it.
LENGTHS = [1, 8, 127, 128, 129, 255, 256, 1000, 4096, 8192, 100_000, 1_000_000]


def _positive(n, seed=3, dtype="float32"):
    """All-positive: nothing cancels, so every rounding error accumulates in
    the same direction rather than partly undoing the last one."""
    rng = np.random.default_rng(seed)
    return (rng.random(n) + 0.5).astype(dtype)


def _relative(got, exact):
    return abs(float(got) - exact) / abs(exact)


@pytest.mark.parametrize("n,trials", [(1024, 40), (8192, 40), (65_536, 20)])
def test_sum_is_as_accurate_as_numpy(n, trials):
    """Averaged over many inputs, not measured on one.

    A single float32 sum lands within an ulp or two of the truth either way, so
    one sample cannot tell a good accumulation from a bad one -- it says more
    about which draw happened to round which direction. What separates them is
    the average over many draws: before this, the mean error at 8192 was 2.96
    times NumPy's; a fold with the same depth as NumPy's is within noise of 1.
    """
    ours = []
    theirs = []
    for seed in range(trials):
        values = _positive(n, seed)
        exact = float(values.astype(np.float64).sum())
        ours.append(_relative(mt.Tensor(values, dtype="float32").sum().item(), exact))
        theirs.append(_relative(values.sum(), exact))

    ratio = float(np.mean(ours) / np.mean(theirs))
    assert ratio < 1.5, f"mean error {np.mean(ours):.3e} vs numpy {np.mean(theirs):.3e}"


@pytest.mark.parametrize("n", LENGTHS)
def test_no_single_sum_is_far_off(n):
    """The per-input bound the average above cannot see: whatever the draw, a
    float32 sum of positive values may not be more than a couple of ulps out."""
    values = _positive(n, seed=11)
    exact = float(values.astype(np.float64).sum())
    assert _relative(mt.Tensor(values, dtype="float32").sum().item(), exact) < 2e-7


@pytest.mark.parametrize("n", [1000, 100_000, 4_000_000])
def test_the_error_does_not_track_the_length(n):
    """The property the fold exists for. A thousand values and four million
    should land within a small factor of each other, not a factor of `n`."""
    values = _positive(n, seed=5)
    exact = float(values.astype(np.float64).sum())
    assert _relative(mt.Tensor(values, dtype="float32").sum().item(), exact) < 5e-7


@pytest.mark.parametrize("n", [8192, 1_000_000])
def test_float64_sums_match_numpy_too(n):
    """The f64 kernel is the same shape with four lanes instead of eight."""
    rng = np.random.default_rng(7)
    values = rng.random(n) + 0.5
    total = 0.0
    comp = 0.0
    for v in values:  # Kahan: the reference f64 has no wider type to check with
        term = v - comp
        run = total + term
        comp = (run - total) - term
        total = run
    ours = _relative(mt.Tensor(values, dtype="float64").sum().item(), total)
    theirs = _relative(values.sum(), total)
    assert ours <= max(theirs * 3.0, 4e-16)


@pytest.mark.parametrize("n", [4096, 100_000])
@pytest.mark.parametrize("op", ["mean", "norm"])
def test_the_reductions_built_on_it_inherit_the_accuracy(n, op):
    values = _positive(n, seed=13)
    wide = values.astype(np.float64)
    if op == "mean":
        exact = float(wide.mean())
        ours = _relative(mt.Tensor(values, dtype="float32").mean().item(), exact)
        theirs = _relative(values.mean(), exact)
    else:
        exact = float(np.linalg.norm(wide))
        ours = _relative(mt.Tensor(values, dtype="float32").norm().item(), exact)
        theirs = _relative(np.linalg.norm(values), exact)
    assert ours <= max(theirs * 3.0, 2e-7), f"ours {ours:.3e} numpy {theirs:.3e}"


@pytest.mark.parametrize("n", [127, 128, 129, 8191, 8192, 8193])
def test_the_leaf_boundary_is_not_special(n):
    """A slice shorter than one leaf takes the lane loop directly and a longer
    one starts folding. Neither may be the worse of the two."""
    values = _positive(n, seed=17)
    exact = float(values.astype(np.float64).sum())
    assert _relative(mt.Tensor(values, dtype="float32").sum().item(), exact) < 2e-7


@pytest.mark.parametrize("shape,dim", [((4, 100_000), 1), ((100_000, 4), 0)])
def test_summing_along_a_dimension_inherits_it(shape, dim):
    values = _positive(shape[0] * shape[1], seed=19).reshape(shape)
    exact = values.astype(np.float64).sum(axis=dim)

    def rel(v):
        return float(
            np.abs(np.asarray(v, np.float64) - exact).max() / np.abs(exact).max()
        )

    ours = rel(mt.Tensor(values, dtype="float32").sum(dim).numpy())
    theirs = rel(values.sum(axis=dim))
    assert ours <= max(theirs * 3.0, 2e-7), f"ours {ours:.3e} numpy {theirs:.3e}"


def test_the_fold_is_still_deterministic():
    """The tree depends on the length and nothing else, so repeated calls --
    which rayon may schedule differently -- have to agree exactly."""
    values = _positive(4_000_000, seed=23)
    t = mt.Tensor(values, dtype="float32")
    first = t.sum().item()
    assert all(t.sum().item() == first for _ in range(20))


def test_small_sums_are_still_exactly_right():
    """Values that add up without rounding must come out exact, whatever the
    tree does with them."""
    values = np.array([1.0, 2.0, 4.0, 8.0, 16.0], dtype=np.float32)
    assert mt.Tensor(values, dtype="float32").sum().item() == 31.0
    ramp = np.arange(1, 1001, dtype=np.float32)
    assert mt.Tensor(ramp, dtype="float32").sum().item() == 500500.0


def test_non_finite_values_still_propagate():
    for bad, check in ((np.inf, np.isinf), (np.nan, np.isnan)):
        values = _positive(5000, seed=29).copy()
        values[4000] = bad
        assert check(mt.Tensor(values, dtype="float32").sum().item())


def test_an_empty_sum_is_zero():
    assert mt.Tensor(np.zeros(0, dtype=np.float32), dtype="float32").sum().item() == 0.0
