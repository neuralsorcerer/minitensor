# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Combining what lands on the same destination.

`scatter` overwrites and `scatter_add` accumulates; those were a boolean between
them. `scatter_reduce` is the enum that boolean was standing in for, with three
more ways of combining and a flag for whether the destination's own value takes
part.

The gradients are where the content is. `amax`/`amin` route to the contributor
that won, `mean` divides by the same count the forward divided by, and `prod`
has to produce the product of everything *except* each factor -- where the
obvious shortcut of dividing the total is wrong exactly when a factor is zero,
which is when the question is interesting.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

REDUCTIONS = ["sum", "prod", "amax", "amin", "mean"]
IDENTITY = {"sum": 0.0, "prod": 1.0, "amax": -np.inf, "amin": np.inf, "mean": 0.0}


def _t(a, requires_grad=False):
    return mt.Tensor.from_numpy(
        np.ascontiguousarray(np.asarray(a, dtype=np.float64)),
        requires_grad=requires_grad,
    )


def _i(a):
    return mt.Tensor.from_numpy(np.ascontiguousarray(np.asarray(a, dtype=np.int64)))


def _reference(base, index, src, reduce, include_self):
    """The definition, one destination at a time."""
    out = base.copy().astype(np.float64)
    touched = set(int(d) for d in index)
    if not include_self:
        for d in touched:
            out[d] = IDENTITY[reduce]
    counts = {d: (1 if include_self else 0) for d in touched}
    for k, d in enumerate(index):
        d = int(d)
        value = src[k]
        if reduce in ("sum", "mean"):
            out[d] += value
        elif reduce == "prod":
            out[d] *= value
        elif reduce == "amax":
            out[d] = max(out[d], value)
        else:
            out[d] = min(out[d], value)
        counts[d] += 1
    if reduce == "mean":
        for d in touched:
            out[d] /= counts[d]
    return out


def _call(base, index, src, reduce, include_self=True):
    return mt.scatter_reduce(
        _t(base), 0, _i(index), _t(src), reduce, include_self
    ).numpy()


# --------------------------------------------------------------------------
# Forward
# --------------------------------------------------------------------------


@pytest.mark.parametrize("reduce", REDUCTIONS)
@pytest.mark.parametrize("include_self", [True, False])
@pytest.mark.parametrize("trial", range(6))
def test_against_a_reference(reduce, include_self, trial):
    rng = np.random.default_rng(hash((reduce, include_self, trial)) % 2**32)
    base = rng.normal(size=6)
    src = rng.normal(size=9)
    index = rng.integers(0, 6, 9)
    got = _call(base, index, src, reduce, include_self)
    assert np.allclose(
        got, _reference(base, index, src, reduce, include_self), atol=1e-12
    )


@pytest.mark.parametrize("reduce", REDUCTIONS)
def test_a_destination_nothing_writes_to_keeps_its_value(reduce):
    """Whatever the reduction, and either way round on `include_self`: an
    untouched entry is not seeded, not averaged, and not reduced."""
    base = np.array([7.0, 8.0, 9.0])
    for include_self in (True, False):
        got = _call(base, np.array([1]), np.array([2.0]), reduce, include_self)
        assert got[0] == 7.0 and got[2] == 9.0


def test_include_self_starts_from_the_identity():
    """The flag's whole content: with it off, a written destination forgets what
    it held and starts from whatever the reduction's identity is."""
    base = np.array([100.0])
    index, src = np.array([0, 0]), np.array([2.0, 3.0])
    assert _call(base, index, src, "sum", True)[0] == pytest.approx(105.0)
    assert _call(base, index, src, "sum", False)[0] == pytest.approx(5.0)
    assert _call(base, index, src, "prod", True)[0] == pytest.approx(600.0)
    assert _call(base, index, src, "prod", False)[0] == pytest.approx(6.0)
    assert _call(base, index, src, "amax", True)[0] == pytest.approx(100.0)
    assert _call(base, index, src, "amax", False)[0] == pytest.approx(3.0)
    assert _call(base, index, src, "amin", False)[0] == pytest.approx(2.0)


def test_mean_divides_by_the_number_that_arrived():
    base = np.array([10.0])
    index, src = np.array([0, 0, 0]), np.array([1.0, 2.0, 3.0])
    # With the destination counting itself there are four values, not three.
    assert _call(base, index, src, "mean", True)[0] == pytest.approx(16.0 / 4)
    assert _call(base, index, src, "mean", False)[0] == pytest.approx(6.0 / 3)


def test_scatter_and_scatter_add_are_two_of_these():
    """They kept their names, and this says the enum did not change them."""
    rng = np.random.default_rng(1)
    base, src = rng.normal(size=5), rng.normal(size=7)
    index = rng.integers(0, 5, 7)
    assert np.allclose(
        mt.scatter_add(_t(base), 0, _i(index), _t(src)).numpy(),
        _call(base, index, src, "sum", True),
    )


def test_mean_over_integers_is_refused():
    """It would truncate every average, which is a surprise rather than a
    result. PyTorch allows it; saying so plainly is better than a wrong number."""
    with pytest.raises(Exception, match="mean"):
        mt.scatter_reduce(_i([1, 2, 3]), 0, _i([0, 1]), _i([4, 5]), "mean", True)


def test_an_unknown_reduction_is_refused():
    with pytest.raises(Exception, match="reduction"):
        _call(np.zeros(3), np.array([0]), np.array([1.0]), "median")


def test_booleans_take_only_replacement():
    """`bool` has no addition, so the accumulating kernels are not even
    generated for it -- which makes this an error rather than a wrong answer."""
    flags = mt.Tensor.from_numpy(np.array([True, False, True]))
    one = mt.Tensor.from_numpy(np.array([True]))
    with pytest.raises(Exception, match="boolean"):
        mt.scatter_reduce(flags, 0, _i([0]), one, "sum", True)


# --------------------------------------------------------------------------
# Gradients
# --------------------------------------------------------------------------


def _grads(base, index, src, reduce, include_self, weights):
    tb, ts = _t(base, True), _t(src, True)
    out = mt.scatter_reduce(tb, 0, _i(index), ts, reduce, include_self)
    (out * _t(weights)).sum().backward()
    return tb.grad.numpy(), ts.grad.numpy()


@pytest.mark.parametrize("reduce", ["prod", "amax", "amin", "mean"])
@pytest.mark.parametrize("include_self", [True, False])
@pytest.mark.parametrize("trial", range(4))
def test_the_gradients_match_central_differences(reduce, include_self, trial):
    rng = np.random.default_rng(hash((reduce, include_self, trial, "g")) % 2**32)
    # Shifted away from zero so the difference quotient means something for
    # `prod`, and so `amax`/`amin` ties stay improbable.
    base = rng.normal(size=5) + 1.5
    src = rng.normal(size=8) + 1.5
    index = rng.integers(0, 5, 8)
    weights = rng.normal(size=5)

    def forward(b, s):
        return float(
            (
                mt.scatter_reduce(
                    _t(b), 0, _i(index), _t(s), reduce, include_self
                ).numpy()
                * weights
            ).sum()
        )

    got_base, got_src = _grads(base, index, src, reduce, include_self, weights)

    step = 1e-6
    want_base = np.zeros_like(base)
    want_src = np.zeros_like(src)
    for k in range(base.size):
        up, down = base.copy(), base.copy()
        up[k] += step
        down[k] -= step
        want_base[k] = (forward(up, src) - forward(down, src)) / (2 * step)
    for k in range(src.size):
        up, down = src.copy(), src.copy()
        up[k] += step
        down[k] -= step
        want_src[k] = (forward(base, up) - forward(base, down)) / (2 * step)

    assert np.allclose(got_base, want_base, atol=1e-5)
    assert np.allclose(got_src, want_src, atol=1e-5)


@pytest.mark.parametrize(
    "src,expected",
    [
        # Nothing zero: each factor's gradient is the product of the others.
        ([3.0, 4.0, 5.0], [40.0, 30.0, 24.0]),
        # One zero: only the zero itself has a surviving excluded product. This
        # is the case `total / factor` divides by zero on.
        ([3.0, 0.0, 5.0], [0.0, 30.0, 0.0]),
        # Two zeros: excluding either still leaves one, so every gradient is
        # zero -- and `total / factor` would be 0/0 twice over.
        ([0.0, 0.0, 5.0], [0.0, 0.0, 0.0]),
    ],
)
def test_the_prod_gradient_survives_zeros(src, expected):
    """The product rule excluding one factor, counted rather than divided."""
    base = np.array([2.0, 1.0])
    index = np.array([0, 0, 0])
    _, got = _grads(base, index, np.array(src), "prod", True, np.array([1.0, 1.0]))
    assert np.array_equal(got, expected)


def test_the_extremum_gradient_goes_to_the_winner():
    base = np.array([0.0])
    index = np.array([0, 0, 0])
    src = np.array([2.0, 9.0, 4.0])
    _, got = _grads(base, index, src, "amax", False, np.array([1.0]))
    assert np.array_equal(got, [0.0, 1.0, 0.0])


def test_an_extremum_tie_goes_to_the_first_contributor():
    """A tie has no natural winner, so the rule is fixed here -- and it is the
    same one `max`, `mode` and `cummax` follow in this library. PyTorch spreads
    a tie evenly instead; one convention across the library beats matching
    another project one operation at a time."""
    base = np.array([0.0])
    index = np.array([0, 0, 0])
    src = np.array([5.0, 5.0, 1.0])
    _, got = _grads(base, index, src, "amax", False, np.array([1.0]))
    assert np.array_equal(got, [1.0, 0.0, 0.0])


def test_the_destination_wins_a_tie_when_it_counts_itself():
    """It was there before anything arrived, so it is the earliest claimant."""
    base = np.array([5.0])
    index = np.array([0])
    src = np.array([5.0])
    got_base, got_src = _grads(base, index, src, "amax", True, np.array([1.0]))
    assert got_base[0] == 1.0
    assert got_src[0] == 0.0


@pytest.mark.parametrize("reduce", ["prod", "amax", "amin", "mean"])
def test_an_untouched_destination_passes_its_gradient_through(reduce):
    """Its output is its input, unchanged, so the gradient is the identity --
    not zero, and not divided by anything."""
    base = np.array([1.0, 2.0, 3.0])
    got_base, _ = _grads(
        base, np.array([1]), np.array([5.0]), reduce, False, np.array([1.0, 1.0, 1.0])
    )
    assert got_base[0] == 1.0 and got_base[2] == 1.0


@pytest.mark.parametrize("reduce", ["prod", "amax", "amin", "mean"])
def test_the_destination_gets_nothing_when_it_does_not_count_itself(reduce):
    """With `include_self=False` a written destination took no part in its own
    result, so a gradient there would be telling a caller to change a number
    that did nothing."""
    base = np.array([4.0, 7.0])
    got_base, _ = _grads(
        base, np.array([0]), np.array([5.0]), reduce, False, np.array([1.0, 1.0])
    )
    assert got_base[0] == 0.0
    assert got_base[1] == 1.0


def test_mean_divides_the_gradient_by_the_same_count():
    base = np.array([0.0])
    index = np.array([0, 0, 0, 0])
    src = np.array([1.0, 2.0, 3.0, 4.0])
    _, got = _grads(base, index, src, "mean", False, np.array([1.0]))
    assert np.allclose(got, [0.25, 0.25, 0.25, 0.25])


def test_only_the_gradients_asked_for_are_built():
    base = np.array([1.0, 2.0])
    src = np.array([3.0, 4.0])
    tensor = _t(base, requires_grad=True)
    out = mt.scatter_reduce(tensor, 0, _i([0, 1]), _t(src), "prod", True)
    out.sum().backward()
    assert tensor.grad is not None


# --------------------------------------------------------------------------
# Shapes
# --------------------------------------------------------------------------


def test_it_works_along_any_axis():
    rng = np.random.default_rng(2)
    base = rng.normal(size=(3, 4))
    src = rng.normal(size=(3, 4))
    index = rng.integers(0, 3, (3, 4))
    got = mt.scatter_reduce(_t(base), 0, _i(index), _t(src), "amax", True).numpy()
    want = base.copy()
    for i in range(3):
        for j in range(4):
            want[index[i, j], j] = max(want[index[i, j], j], src[i, j])
    assert np.allclose(got, want)


def test_a_negative_axis_counts_from_the_end():
    rng = np.random.default_rng(3)
    base = rng.normal(size=(2, 5))
    src = rng.normal(size=(2, 5))
    index = rng.integers(0, 5, (2, 5))
    assert np.allclose(
        mt.scatter_reduce(_t(base), -1, _i(index), _t(src), "sum", True).numpy(),
        mt.scatter_reduce(_t(base), 1, _i(index), _t(src), "sum", True).numpy(),
    )


def test_an_out_of_range_index_is_refused():
    with pytest.raises(Exception):
        _call(np.zeros(3), np.array([5]), np.array([1.0]), "sum")
