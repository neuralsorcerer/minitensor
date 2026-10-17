# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Softmax over a long axis has to still sum to one.

Every softmax kernel here shifted by the slice max and then added the
exponentials into a single running total. Over a vocabulary-sized axis that is
the accumulation pattern with the worst error growth there is: after the shift
exactly one term is 1.0 and the rest are small, so each addition rounds off a
piece of the small one, in the same direction, a million times over. On a
1,000,000-class axis the probabilities came back summing to 1.0018 rather than
1, and the values themselves were 1.6e-3 off -- against NumPy's 2.5e-8 for the
same arithmetic in the same precision, a factor of 65,000.

`log_softmax` was wrong by the same mechanism, in the log domain, and both
layouts were affected: the contiguous one where the softmax axis is last, and
the column-wise one where it is not. In the column-wise layout NumPy is no
better than we were -- it is the same running total -- so those cases are
checked against a float64 reference rather than against NumPy.

The fix routes each of the six accumulators through the blocked, pairwise sums
in `ops::util`. The kernels still make exactly one `exp` call per element: the
column-wise pass computes it, stores it, and adds it in one go, because
computing it twice cost 64% and even storing it and reading it back cost 26%.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

# Long enough that a running total visibly degrades, and long enough to cross
# several block boundaries in both layouts.
CONTIGUOUS = [((2, 250_000), 1), ((2, 1_000_000), 1), ((8, 128, 2048), 2)]
COLUMNWISE = [((250_000, 2), 0), ((1_000_000, 2), 0), ((2048, 8, 128), 0)]


def _data(shape, seed=3):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) * 3).astype(np.float32)


def _softmax_reference(x, axis):
    peak = x.max(axis=axis, keepdims=True)
    e = np.exp(x - peak)
    return e / e.sum(axis=axis, keepdims=True)


def _log_softmax_reference(x, axis):
    peak = x.max(axis=axis, keepdims=True)
    return x - peak - np.log(np.exp(x - peak).sum(axis=axis, keepdims=True))


def _relative(got, exact):
    return float(
        np.abs(np.asarray(got, np.float64) - exact).max() / np.abs(exact).max()
    )


@pytest.mark.parametrize("shape,dim", CONTIGUOUS + COLUMNWISE)
def test_softmax_still_sums_to_one_over_a_long_axis(shape, dim):
    """The property the whole change exists for.

    float32 carries about seven digits; 1e-5 is a loose bound on the total of a
    million of them, and a running sum misses it by two orders of magnitude."""
    values = _data(shape)
    got = mt.Tensor(values, dtype="float32").softmax(dim).numpy()
    totals = got.astype(np.float64).sum(axis=dim)
    assert float(np.abs(totals - 1.0).max()) < 1e-5


@pytest.mark.parametrize("shape,dim", CONTIGUOUS + COLUMNWISE)
@pytest.mark.parametrize("op", ["softmax", "log_softmax"])
def test_the_long_axis_result_is_accurate_in_absolute_terms(shape, dim, op):
    values = _data(shape, seed=5)
    wide = values.astype(np.float64)
    t = mt.Tensor(values, dtype="float32")
    if op == "softmax":
        exact = _softmax_reference(wide, dim)
        got = t.softmax(dim).numpy()
    else:
        exact = _log_softmax_reference(wide, dim)
        got = t.log_softmax(dim).numpy()
    assert _relative(got, exact) < 1e-5


@pytest.mark.parametrize("shape,dim", CONTIGUOUS)
@pytest.mark.parametrize("op", ["softmax", "log_softmax"])
def test_the_contiguous_layout_is_no_worse_than_numpy(shape, dim, op):
    """NumPy's two-pass form is the thing being matched here: it reduces with
    pairwise summation, which is why it stayed accurate where we did not.

    A small multiple rather than equality: the two block differently. The
    multiple is what set the block size -- at 8192 the million-class case came
    out 27 times worse than NumPy, and at 1024 it matches to the digit. What is
    left is short axes, where NumPy's recursion still bottoms out finer than one
    1024-term block: a 2048-long axis lands at 5.3e-7 against 1.2e-7, both a
    handful of float32 ulps."""
    values = _data(shape, seed=7)
    wide = values.astype(np.float64)
    reference = _softmax_reference if op == "softmax" else _log_softmax_reference
    exact = reference(wide, dim)

    t = mt.Tensor(values, dtype="float32")
    ours = _relative(getattr(t, op)(dim).numpy(), exact)
    theirs = _relative(reference(values, dim), exact)
    assert ours <= max(theirs * 10.0, 1e-9), f"ours {ours:.3e} numpy {theirs:.3e}"


@pytest.mark.parametrize("shape,dim", COLUMNWISE[:2])
@pytest.mark.parametrize("op", ["softmax", "log_softmax"])
def test_the_columnwise_layout_beats_numpy(shape, dim, op):
    """Reducing a leading axis is where NumPy keeps a running total of its own:
    at a million rows it lands at 1.9e-3 while the blocked version here is at
    5.1e-6."""
    values = _data(shape, seed=11)
    wide = values.astype(np.float64)
    reference = _softmax_reference if op == "softmax" else _log_softmax_reference
    exact = reference(wide, dim)

    t = mt.Tensor(values, dtype="float32")
    ours = _relative(getattr(t, op)(dim).numpy(), exact)
    theirs = _relative(reference(values, dim), exact)
    assert ours < theirs, f"ours {ours:.3e} numpy {theirs:.3e}"


@pytest.mark.parametrize("n", [8191, 8192, 8193, 16384, 16385])
def test_the_block_boundary_is_not_special(n):
    """A short axis takes one block and a slightly longer one takes two, and
    the answer may not jump between them.

    The two layouts split at different lengths -- the contiguous one every 1024
    terms, the column-wise one every 8192 steps -- so these lengths straddle a
    boundary in one or both."""
    values = _data((2, n), seed=13)
    exact = _softmax_reference(values.astype(np.float64), 1)
    got = mt.Tensor(values, dtype="float32").softmax(1).numpy()
    assert _relative(got, exact) < 1e-6

    values_t = np.ascontiguousarray(values.T)
    exact_t = _softmax_reference(values_t.astype(np.float64), 0)
    got_t = mt.Tensor(values_t, dtype="float32").softmax(0).numpy()
    # The column-wise bound is looser by design: its accumulators are a vector
    # per column, so it cannot afford the eight lanes the contiguous path
    # splits into, and one block of 8192 steps is a run of 8192 additions. That
    # caps the error where an unblocked run did not -- at a million rows this
    # is still 5.1e-6 while NumPy, which keeps a running total here, is at
    # 1.9e-3.
    assert _relative(got_t, exact_t) < 1e-5


@pytest.mark.parametrize("shape,dim", [((2, 250_000), 1), ((250_000, 2), 0)])
def test_log_softmax_agrees_with_the_log_of_softmax(shape, dim):
    """Two spellings of one computation, taking different kernels: `softmax`
    accumulates the exponentials it writes out, `log_softmax` accumulates them
    without storing any. They have to land in the same place."""
    values = _data(shape, seed=17)
    t = mt.Tensor(values, dtype="float32")
    direct = t.log_softmax(dim).numpy().astype(np.float64)
    composed = np.log(t.softmax(dim).numpy().astype(np.float64))
    assert float(np.abs(direct - composed).max()) < 1e-5


@pytest.mark.parametrize("shape,dim", [((2, 250_000), 1), ((250_000, 2), 0)])
@pytest.mark.parametrize("op", ["masked_softmax", "masked_log_softmax"])
def test_the_masked_kernels_are_accurate_over_a_long_axis(shape, dim, op):
    """The masked kernels keep their own copy of the accumulation, so they get
    the same treatment and the same test. A true entry is excluded."""
    values = _data(shape, seed=19)
    rng = np.random.default_rng(23)
    mask = rng.random(shape) < 0.25

    kept = np.where(mask, -np.inf, values.astype(np.float64))
    if op == "masked_softmax":
        exact = _softmax_reference(kept, dim)
        exact = np.where(mask, 0.0, exact)
    else:
        exact = _log_softmax_reference(kept, dim)

    got = getattr(mt, op)(
        mt.Tensor(values, dtype="float32"), mt.Tensor(mask, dtype="bool"), dim
    ).numpy()
    finite = np.isfinite(exact)
    error = float(
        np.abs(got.astype(np.float64)[finite] - exact[finite]).max()
        / np.abs(exact[finite]).max()
    )
    assert error < 1e-5
    assert np.array_equal(np.isfinite(got), finite)


def test_masked_softmax_still_zeroes_a_fully_masked_slice():
    """A slice with nothing left is zeros, not NaN -- the blocking must not
    have moved the short-circuit that produces them."""
    values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mask = np.array([[True, True], [False, False]])
    got = mt.masked_softmax(
        mt.Tensor(values, dtype="float32"), mt.Tensor(mask, dtype="bool"), 1
    ).numpy()
    np.testing.assert_array_equal(got[0], np.zeros(2, dtype=np.float32))
    assert float(got[1].sum()) == pytest.approx(1.0)


def test_softmax_still_handles_an_all_negative_infinity_slice():
    """`-inf` everywhere has no max to shift by; the answer is zeros."""
    values = np.array([[-np.inf, -np.inf], [0.0, 0.0]], dtype=np.float32)
    got = mt.Tensor(values, dtype="float32").softmax(1).numpy()
    np.testing.assert_array_equal(got[0], np.zeros(2, dtype=np.float32))
    np.testing.assert_allclose(got[1], [0.5, 0.5])

    column = np.ascontiguousarray(values.T)
    got_col = mt.Tensor(column, dtype="float32").softmax(0).numpy()
    np.testing.assert_array_equal(got_col[:, 0], np.zeros(2, dtype=np.float32))
    np.testing.assert_allclose(got_col[:, 1], [0.5, 0.5])


_INF = np.inf
# One `+inf`, two of them, one beside a NaN, an ordinary row, and `+inf` among
# `-inf`s -- with what each comes to.
_INFINITE_ROWS = np.array(
    [
        [_INF, 1.0, 2.0],
        [_INF, _INF, 0.0],
        [_INF, np.nan, 1.0],
        [1.0, 2.0, 3.0],
        [-_INF, _INF, -_INF],
    ]
)
_INFINITE_SOFTMAX = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [np.nan, np.nan, np.nan],
        np.exp([1.0, 2.0, 3.0]) / np.exp([1.0, 2.0, 3.0]).sum(),
        [0.0, 1.0, 0.0],
    ]
)


def _layouts(rows):
    """The same slices along the last axis, down the first, and down a middle
    axis of a stack -- the row, column and block walks."""
    yield "last", rows, 1, lambda out: out
    yield "first", np.ascontiguousarray(rows.T), 0, lambda out: out.T
    stack = np.stack([rows.T, rows.T])
    yield "middle", stack, 1, lambda out: out[1].T


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("log", [False, True], ids=["softmax", "log_softmax"])
def test_a_slice_with_an_infinite_maximum_takes_its_limit(dtype, log):
    """`softmax([inf, 1, 2])` was NaN everywhere: no shift makes `inf - inf`
    finite. As those entries grow without bound they take all the mass, so
    the answer is `[1, 0, 0]`; equal infinities share it, and a NaN in the
    slice still makes all of it NaN."""
    with np.errstate(divide="ignore"):
        expected = np.log(_INFINITE_SOFTMAX) if log else _INFINITE_SOFTMAX
    for name, values, dim, back in _layouts(_INFINITE_ROWS):
        tensor = mt.Tensor(values, dtype=dtype)
        got = back((tensor.log_softmax(dim) if log else tensor.softmax(dim)).numpy())
        np.testing.assert_allclose(got, expected, rtol=1e-6, err_msg=name)
        assert (np.isinf(got) == np.isinf(expected)).all(), name
    if log:
        # The single infinity is log(1), a plain zero.
        got = mt.Tensor([[_INF, 1.0]], dtype=dtype).log_softmax(1).numpy()
        assert got[0, 0] == 0.0 and not np.signbit(got[0, 0])


@pytest.mark.parametrize("log", [False, True], ids=["softmax", "log_softmax"])
def test_the_banded_walk_takes_the_limit_too(log):
    """One block down the first axis is cut into row bands; an infinity in
    one band still settles its whole column."""
    values = np.random.default_rng(3).standard_normal((200_000, 3))
    values[123_456, 1] = _INF
    tensor = mt.Tensor(values, dtype="float64")
    got = (tensor.log_softmax(0) if log else tensor.softmax(0)).numpy()
    column = got[:, 1]
    if log:
        assert column[123_456] == 0.0 and np.isneginf(np.delete(column, 123_456)).all()
    else:
        assert column[123_456] == 1.0 and (np.delete(column, 123_456) == 0).all()
    # The other columns are ordinary softmaxes.
    sums = np.exp(got[:, [0, 2]]).sum(0) if log else got[:, [0, 2]].sum(0)
    np.testing.assert_allclose(sums, [1.0, 1.0], rtol=1e-10)


def test_masked_softmax_takes_the_limit_over_what_is_unmasked():
    values = mt.Tensor([[_INF, 1.0, 2.0], [1.0, _INF, 2.0]], dtype="float64")
    mask = mt.Tensor([[False, False, False], [False, True, False]], dtype="bool")
    got = mt.masked_softmax(values, mask, 1).numpy()
    np.testing.assert_array_equal(got[0], [1.0, 0.0, 0.0])
    # A masked infinity is out of the slice, which is an ordinary one.
    np.testing.assert_allclose(
        got[1], [np.exp(-1) / (1 + np.exp(-1)), 0.0, 1 / (1 + np.exp(-1))]
    )


def test_the_gradient_through_an_infinite_row_is_zero():
    """With all the mass on one entry the output no longer moves with the
    inputs, and the gradient says so rather than being NaN."""
    logits = mt.Tensor([[_INF, 1.0, 2.0]], dtype="float64", requires_grad=True)
    (logits.softmax(1) * mt.Tensor([[1.0, 2.0, 3.0]], dtype="float64")).sum().backward()
    np.testing.assert_array_equal(logits.grad.numpy(), [[0.0, 0.0, 0.0]])
    mt.clear_autograd_graph()


@pytest.mark.parametrize("shape,dim", [((3, 5), 1), ((5, 3), 0), ((2, 3, 4), 1)])
@pytest.mark.parametrize("op", ["softmax", "log_softmax"])
def test_small_shapes_are_unchanged(shape, dim, op):
    """Short axes take the single-block walk, which must be the plain loop."""
    values = _data(shape, seed=29)
    wide = values.astype(np.float64)
    reference = _softmax_reference if op == "softmax" else _log_softmax_reference
    got = getattr(mt.Tensor(values, dtype="float32"), op)(dim).numpy()
    # `atol` carries the `log_softmax` entries that sit near zero, where a
    # relative bound measures float32's cancellation rather than the sum.
    np.testing.assert_allclose(got, reference(wide, dim), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("shape,dim", [((2, 100_000), 1), ((100_000, 2), 0)])
def test_the_gradient_path_gets_the_same_forward_values(shape, dim):
    """Softmax with `requires_grad` takes the same forward kernel; the blocking
    is not conditioned on the flag the way the `var`/`logsumexp` fusions are."""
    values = _data(shape, seed=31)
    plain = mt.Tensor(values, dtype="float32").softmax(dim).numpy()
    tracked = (
        mt.Tensor(values, dtype="float32", requires_grad=True).softmax(dim).numpy()
    )
    np.testing.assert_array_equal(plain, tracked)
