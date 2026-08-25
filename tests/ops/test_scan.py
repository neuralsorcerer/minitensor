# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Running totals that are not sums.

`cummax` and `cummin` carry the running extremum *and* where it came from;
`logcumsumexp` accumulates probabilities held as logarithms. None composes: a
scan is a recurrence, and nothing else in the library runs one.

Two conventions need pinning because neither falls out of the arithmetic -- what
a tie does, and what a NaN does -- and one numerical claim needs demonstrating
rather than asserting: that `logcumsumexp` is not `cumsum` of `exp`, because the
latter is unusable at the lengths this is for.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt


def _t(a, requires_grad=False):
    return mt.Tensor.from_numpy(
        np.ascontiguousarray(np.asarray(a, dtype=np.float64)),
        requires_grad=requires_grad,
    )


def _reference_logcumsumexp(a, axis):
    """Shifted by the maximum so the reference itself does not underflow."""
    top = a.max(axis=axis, keepdims=True)
    return np.log(np.cumsum(np.exp(a - top), axis=axis)) + top


# --------------------------------------------------------------------------
# cummax and cummin
# --------------------------------------------------------------------------


@pytest.mark.parametrize("trial", range(10))
def test_cummax_and_cummin_match_numpy(trial):
    rng = np.random.default_rng(trial)
    shape = tuple(int(v) for v in rng.integers(1, 5, int(rng.integers(1, 4))))
    a = rng.normal(size=shape)
    for dim in range(len(shape)):
        assert np.array_equal(
            mt.cummax(_t(a), dim)[0].numpy(), np.maximum.accumulate(a, axis=dim)
        )
        assert np.array_equal(
            mt.cummin(_t(a), dim)[0].numpy(), np.minimum.accumulate(a, axis=dim)
        )


@pytest.mark.parametrize("trial", range(6))
def test_the_reported_index_holds_the_reported_value(trial):
    """The property tying the two outputs together, which the values alone
    cannot check -- an index off by one still names a plausible number."""
    rng = np.random.default_rng(100 + trial)
    a = rng.normal(size=(6, 4))
    for dim in (0, 1):
        values, indices = mt.cummax(_t(a), dim)
        picked = np.take_along_axis(a, indices.numpy(), axis=dim)
        assert np.array_equal(picked, values.numpy())


def test_a_tie_keeps_the_earliest_position():
    """A choice rather than a consequence, and the same one `mode` makes: a
    running extremum that is merely equalled does not move."""
    _, indices = mt.cummax(_t([1.0, 3.0, 3.0, 2.0, 3.0]), 0)
    assert np.array_equal(indices.numpy(), [0, 1, 1, 1, 1])


def test_a_nan_takes_over_and_holds():
    """As it does for `max`. And the *first* NaN keeps the index -- without a
    guard on the running value a later one would move the index while changing
    nothing about the value, which is invisible until you look at the index."""
    values, indices = mt.cummax(_t([1.0, 3.0, np.nan, 5.0, np.nan]), 0)
    assert np.array_equal(values.numpy()[:2], [1.0, 3.0])
    assert np.isnan(values.numpy()[2:]).all()
    assert np.array_equal(indices.numpy(), [0, 1, 2, 2, 2])


def test_cummin_is_not_cummax():
    """They differ only in the comparison, which is exactly the kind of thing
    that survives a copy-paste, so the two are pinned against each other."""
    a = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    assert np.array_equal(mt.cummax(_t(a), 0)[0].numpy(), [3.0, 3.0, 4.0, 4.0, 5.0])
    assert np.array_equal(mt.cummin(_t(a), 0)[0].numpy(), [3.0, 1.0, 1.0, 1.0, 1.0])


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_cummax_of_every_scannable_dtype(dtype):
    a = np.array([[3, 1, 4], [1, 5, 9]], dtype=dtype)
    values, indices = mt.cummax(mt.Tensor.from_numpy(np.ascontiguousarray(a)), 1)
    assert values.numpy().dtype == dtype
    assert indices.dtype == "int64"
    assert np.array_equal(values.numpy(), np.maximum.accumulate(a, axis=1))


def test_cummax_rejects_booleans():
    with pytest.raises(Exception, match="boolean"):
        mt.cummax(mt.Tensor.from_numpy(np.array([True, False])), 0)


def test_the_default_axis_is_the_last():
    a = np.array([[1.0, 3.0, 2.0], [5.0, 4.0, 6.0]])
    assert np.array_equal(mt.cummax(_t(a))[0].numpy(), np.maximum.accumulate(a, axis=1))


@pytest.mark.parametrize("trial", range(6))
def test_the_cummax_gradient_goes_to_the_winning_positions(trial):
    """Each output took its value from one input position, so its gradient goes
    there. A position that won several prefixes collects all of them, which is
    why this accumulates rather than assigns."""
    rng = np.random.default_rng(200 + trial)
    a = rng.normal(size=(5, 4))
    upstream = rng.normal(size=(5, 4))
    tensor = _t(a, requires_grad=True)
    values, indices = mt.cummax(tensor, 0)
    (values * _t(upstream)).sum().backward()

    want = np.zeros_like(a)
    for d in range(5):
        for j in range(4):
            want[indices.numpy()[d, j], j] += upstream[d, j]
    assert np.allclose(tensor.grad.numpy(), want)


def test_a_repeated_winner_accumulates_every_prefix_it_won():
    """The clearest case: a single maximum at the front wins every prefix, so
    its gradient is the whole column's worth."""
    tensor = _t([5.0, 1.0, 2.0, 3.0], requires_grad=True)
    mt.cummax(tensor, 0)[0].sum().backward()
    assert np.array_equal(tensor.grad.numpy(), [4.0, 0.0, 0.0, 0.0])


def test_the_indices_carry_no_gradient():
    tensor = _t([1.0, 2.0], requires_grad=True)
    values, indices = mt.cummax(tensor, 0)
    assert values.requires_grad
    assert not indices.requires_grad


# --------------------------------------------------------------------------
# logcumsumexp
# --------------------------------------------------------------------------


@pytest.mark.parametrize("trial", range(10))
def test_logcumsumexp_matches_a_shifted_reference(trial):
    rng = np.random.default_rng(300 + trial)
    a = rng.normal(0, 30, (int(rng.integers(1, 25)), 3))
    for dim in (0, 1):
        got = mt.logcumsumexp(_t(a), dim).numpy()
        assert np.allclose(got, _reference_logcumsumexp(a, dim), atol=1e-12)


def test_the_first_element_is_itself():
    a = np.array([[-3.0, 7.0], [1.0, 2.0]])
    assert np.allclose(mt.logcumsumexp(_t(a), 0).numpy()[0], a[0])


def test_it_is_not_cumsum_of_exp():
    """The whole reason it exists. `exp(-800)` is zero in float64, so the naive
    reading reports `-inf` for every position; staying in the log domain keeps
    every step representable."""
    a = np.full((4000, 1), -800.0)
    got = mt.logcumsumexp(_t(a), 0).numpy()
    assert np.isfinite(got).all()
    # log(sum of n copies of exp(-800)) == -800 + log(n)
    assert got[-1, 0] == pytest.approx(-800.0 + np.log(4000), abs=1e-9)

    with np.errstate(divide="ignore"):
        naive = np.log(np.cumsum(np.exp(a), axis=0))
    assert not np.isfinite(naive).any()


def test_a_long_axis_of_ordinary_log_probabilities_stays_finite():
    rng = np.random.default_rng(4)
    a = np.log(rng.dirichlet(np.ones(2000)))
    got = mt.logcumsumexp(_t(a), 0).numpy()
    assert np.isfinite(got).all()
    # The total of a probability distribution is one, so its log is zero.
    assert got[-1] == pytest.approx(0.0, abs=1e-12)


def test_negative_infinity_contributes_nothing():
    """`-inf` is a probability of zero, and adding it must leave the running
    total alone rather than producing a NaN."""
    a = np.array([-np.inf, 0.0, -np.inf, 1.0])
    got = mt.logcumsumexp(_t(a), 0).numpy()
    assert got[0] == -np.inf
    assert got[1] == pytest.approx(0.0)
    assert got[2] == pytest.approx(0.0)
    assert got[3] == pytest.approx(np.logaddexp(0.0, 1.0))


def test_all_negative_infinity_stays_negative_infinity():
    got = mt.logcumsumexp(_t(np.full(5, -np.inf)), 0).numpy()
    assert np.all(got == -np.inf)
    assert not np.isnan(got).any()


@pytest.mark.parametrize("trial", range(8))
def test_the_logcumsumexp_gradient_matches_central_differences(trial):
    """The upstream gradient deliberately mixes signs. The backward splits it
    into a positive and a negative scan and subtracts them, and an all-positive
    seed would never reach the second one."""
    rng = np.random.default_rng(400 + trial)
    a = rng.normal(0, 2.0, (5, 3))
    upstream = rng.normal(size=(5, 3))
    assert (upstream > 0).any() and (upstream < 0).any()

    tensor = _t(a, requires_grad=True)
    (mt.logcumsumexp(tensor, 0) * _t(upstream)).sum().backward()
    got = tensor.grad.numpy()

    step = 1e-6
    want = np.zeros_like(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            up, down = a.copy(), a.copy()
            up[i, j] += step
            down[i, j] -= step
            want[i, j] = (
                (_reference_logcumsumexp(up, 0) * upstream).sum()
                - (_reference_logcumsumexp(down, 0) * upstream).sum()
            ) / (2 * step)
    assert np.allclose(got, want, atol=2e-6)


def test_the_gradient_of_the_sum_is_one_per_column():
    """A consequence of the definition: summing every prefix, each input
    contributes exactly its share of each prefix it appears in, and the shares
    of any one prefix sum to one. So the total gradient is the axis length."""
    rng = np.random.default_rng(5)
    tensor = _t(rng.normal(size=(7, 2)), requires_grad=True)
    mt.logcumsumexp(tensor, 0).sum().backward()
    assert np.allclose(tensor.grad.numpy().sum(axis=0), 7.0)


def test_logcumsumexp_rejects_integers():
    with pytest.raises(Exception, match="floating point"):
        mt.logcumsumexp(mt.Tensor.from_numpy(np.array([1, 2], dtype=np.int64)), 0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_both_floating_dtypes(dtype):
    rng = np.random.default_rng(6)
    a = rng.normal(size=(5, 3)).astype(dtype)
    got = mt.logcumsumexp(mt.Tensor.from_numpy(np.ascontiguousarray(a)), 0)
    assert got.numpy().dtype == dtype
    assert np.allclose(
        got.numpy(), _reference_logcumsumexp(a.astype(np.float64), 0), rtol=1e-6
    )


# --------------------------------------------------------------------------
# Shapes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [0, 1, 2, -1, -2, -3])
def test_every_axis_of_a_three_dimensional_tensor(dim):
    rng = np.random.default_rng(7)
    a = rng.normal(size=(2, 3, 4))
    assert np.array_equal(
        mt.cummax(_t(a), dim)[0].numpy(), np.maximum.accumulate(a, axis=dim)
    )
    assert np.allclose(
        mt.logcumsumexp(_t(a), dim).numpy(), _reference_logcumsumexp(a, dim)
    )


def test_a_length_one_axis_is_the_input():
    a = np.array([[1.0], [2.0]])
    assert np.array_equal(mt.cummax(_t(a), 1)[0].numpy(), a)
    assert np.allclose(mt.logcumsumexp(_t(a), 1).numpy(), a)


def test_an_empty_axis_gives_an_empty_answer():
    a = np.zeros((2, 0))
    assert mt.cummax(_t(a), 1)[0].numpy().shape == (2, 0)
    assert mt.logcumsumexp(_t(a), 1).numpy().shape == (2, 0)


def test_the_axis_must_exist():
    with pytest.raises(Exception):
        mt.cummax(_t([1.0, 2.0]), 3)
