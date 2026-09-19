# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sort and the selections built on it, against NumPy, in the cases that separate
implementations.

Eight entry points here -- `sort`, `argsort`, `msort`, `topk`, `kthvalue`,
`median`, `mode`, `quantile` -- are all answers to the same question, and the
interesting thing about them is that they agree. One model explains every one:

    the ascending order is stable, NaN sorts as though it were `+inf`, ties
    keep their input positions, and the descending order is that order with
    each run of equal values reversed -- *not* the ascending order reversed,
    which would put tied elements back to front.

`sort_by_key32` in `reduction/sort.rs` says so ("ties still fall back to the
*ascending* position and the answer stays the mirror of the ascending one
rather than its reverse"), and that is a promise nothing was checking. It is
also the promise most easily broken by accident: a comparator that negates
rather than complements, a parallel merge that is not stable, a `topk` that
reverses a slice. None of those would fail a test that sorts distinct values.

So the oracle here is that model, written out in NumPy, and every entry point
is required to agree with it. Each case carries ties, and most carry NaN, both
infinities and both zeros.

Two conventions are deliberately *not* NumPy's, and are pinned as the library
documents them rather than against `np.median`:

- `median` of an even count returns the **lower** of the two middle values, not
  their average, which is what lets it also report the index it came from.
  `np.median` averages. `quantile(0.5)` is the interpolated definition and *is*
  checked against NumPy.
- `-0.0` and `0.0` compare **equal**, so they are ordered by input position
  rather than by sign bit. A total order over the bit patterns would separate
  them; this does not, and neither does PyTorch.

Values are compared as bytes rather than with `==`, because `==` is blind to a
dropped sign on zero and cries wolf on a faithful NaN.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

# --- the oracle ------------------------------------------------------------


def _keys(run: np.ndarray) -> np.ndarray:
    """The values as an ordering key: NaN becomes `+inf`, integers pass through."""
    if np.issubdtype(run.dtype, np.floating):
        return np.where(np.isnan(run), np.inf, run).astype(np.float64)
    return run.astype(np.int64)


def _ascending(run: np.ndarray) -> np.ndarray:
    # `lexsort` takes the primary key last, so this is "by value, then by
    # position" -- which is what stability means when it is spelled out.
    return np.lexsort((np.arange(run.shape[0]), _keys(run)))


def _descending(run: np.ndarray) -> np.ndarray:
    order = _ascending(run)
    keys = _keys(run)[order]
    runs, start = [], 0
    for i in range(1, order.shape[0] + 1):
        if i == order.shape[0] or keys[i] != keys[start]:
            runs.append(order[start:i])
            start = i
    return np.concatenate(runs[::-1]) if runs else order


def _order_along(values: np.ndarray, dim: int, descending: bool) -> np.ndarray:
    """The expected permutation, applied independently to every run along `dim`."""
    moved = np.moveaxis(values, dim, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    out = np.empty(flat.shape, dtype=np.int64)
    for row in range(flat.shape[0]):
        out[row] = _descending(flat[row]) if descending else _ascending(flat[row])
    return np.moveaxis(out.reshape(moved.shape), -1, dim)


def _gather(values: np.ndarray, order: np.ndarray, dim: int) -> np.ndarray:
    return np.take_along_axis(values, order, axis=dim)


def _same_bits(actual, expected, what: str) -> None:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape, f"{what}: {actual.shape} vs {expected.shape}"
    assert actual.dtype == expected.dtype, f"{what}: {actual.dtype} vs {expected.dtype}"
    assert (
        actual.tobytes() == expected.tobytes()
    ), f"{what}: got {actual.ravel()[:12]} expected {expected.ravel()[:12]}"


# --- the cases -------------------------------------------------------------

NAN, INF = float("nan"), float("inf")

FLOAT_RUNS = {
    "ties": [3.0, 1.0, 3.0, 1.0, 2.0, 1.0],
    "all-equal": [7.0, 7.0, 7.0, 7.0],
    "with-nan": [3.0, NAN, 1.0, NAN, 2.0, 1.0],
    "all-nan": [NAN, NAN, NAN],
    "infinities": [INF, -INF, 0.0, INF, -INF, 1.0],
    "signed-zeros": [0.0, -0.0, 1.0, -0.0, 0.0, -1.0],
    "already-sorted": [1.0, 2.0, 3.0, 4.0],
    "reversed": [4.0, 3.0, 2.0, 1.0],
    "single": [42.0],
    "extremes": [np.finfo(np.float32).max, np.finfo(np.float32).min, 0.0, NAN],
}

INT_RUNS = {
    "ties": [3, 1, 3, 1, 2, 1],
    "extremes": [np.iinfo(np.int64).max, np.iinfo(np.int64).min, 0, -1, 1],
    "single": [7],
}


def _float_tensor(run, dtype="float32"):
    values = np.asarray(run, dtype=np.float32 if dtype == "float32" else np.float64)
    return values, mt.Tensor(values, dtype=dtype)


def _int_tensor(run):
    values = np.asarray(run, dtype=np.int64)
    return values, mt.Tensor(values, dtype="int64")


ALL_RUNS = [(f"f32:{n}", r, "float32") for n, r in FLOAT_RUNS.items()]
ALL_RUNS += [(f"f64:{n}", r, "float64") for n, r in FLOAT_RUNS.items()]
ALL_RUNS += [(f"i64:{n}", r, "int64") for n, r in INT_RUNS.items()]


def _build(run, dtype):
    if dtype == "int64":
        return _int_tensor(run)
    return _float_tensor(run, dtype)


# --- sort, argsort, msort --------------------------------------------------


@pytest.mark.parametrize("name,run,dtype", ALL_RUNS, ids=[c[0] for c in ALL_RUNS])
@pytest.mark.parametrize("descending", [False, True], ids=["asc", "desc"])
def test_sort_matches_the_stable_model(name, run, dtype, descending):
    values, tensor = _build(run, dtype)
    order = _order_along(values, 0, descending)

    got_values, got_indices = mt.sort(tensor, 0, descending)
    _same_bits(got_values, _gather(values, order, 0), f"{name} values")
    _same_bits(got_indices, order, f"{name} indices")


@pytest.mark.parametrize("name,run,dtype", ALL_RUNS, ids=[c[0] for c in ALL_RUNS])
@pytest.mark.parametrize("descending", [False, True], ids=["asc", "desc"])
def test_argsort_returns_exactly_sorts_indices(name, run, dtype, descending):
    _, tensor = _build(run, dtype)
    _, from_sort = mt.sort(tensor, 0, descending)
    _same_bits(mt.argsort(tensor, 0, descending), np.asarray(from_sort), name)


@pytest.mark.parametrize("stable", [False, True], ids=["stable=False", "stable=True"])
def test_the_stable_flag_changes_nothing(stable):
    """`stable=False` is accepted and ignored: this sort is always stable, and a
    caller who asks for the cheaper unstable one must not get different ties."""
    values, tensor = _float_tensor(FLOAT_RUNS["with-nan"] * 3)
    for descending in (False, True):
        got, indices = mt.sort(tensor, 0, descending, stable)
        order = _order_along(values, 0, descending)
        _same_bits(got, _gather(values, order, 0), f"stable={stable}")
        _same_bits(indices, order, f"stable={stable} indices")


def test_msort_is_sort_along_the_first_dimension():
    values = np.array([[3.0, 1.0], [1.0, 3.0], [2.0, 2.0]], dtype=np.float32)
    tensor = mt.Tensor(values, dtype="float32")
    order = _order_along(values, 0, False)
    _same_bits(mt.msort(tensor), _gather(values, order, 0), "msort")


# --- every dimension, including the negative spellings ---------------------

SHAPES = [(4,), (3, 5), (2, 3, 4)]


@pytest.mark.parametrize("shape", SHAPES, ids=[str(s) for s in SHAPES])
@pytest.mark.parametrize("descending", [False, True], ids=["asc", "desc"])
def test_every_dimension_sorts_its_own_runs(shape, descending):
    rng = np.random.default_rng(20260919)
    # Few distinct values, so ties are everywhere rather than incidental.
    values = rng.integers(0, 3, size=shape).astype(np.float32)
    values.ravel()[:: max(values.size // 3, 1)] = NAN
    tensor = mt.Tensor(values, dtype="float32")

    for dim in range(len(shape)):
        for spelling in (dim, dim - len(shape)):
            order = _order_along(values, dim, descending)
            got_values, got_indices = mt.sort(tensor, spelling, descending)
            _same_bits(got_values, _gather(values, order, dim), f"dim={spelling}")
            _same_bits(got_indices, order, f"dim={spelling} indices")


# --- topk, kthvalue --------------------------------------------------------


@pytest.mark.parametrize("name,run,dtype", ALL_RUNS, ids=[c[0] for c in ALL_RUNS])
@pytest.mark.parametrize("largest", [True, False], ids=["largest", "smallest"])
def test_topk_is_the_head_of_the_matching_order(name, run, dtype, largest):
    values, tensor = _build(run, dtype)
    order = _order_along(values, 0, largest)

    for k in range(1, values.shape[0] + 1):
        got_values, got_indices = mt.topk(tensor, k, 0, largest, True)
        _same_bits(got_values, _gather(values, order, 0)[:k], f"{name} k={k}")
        _same_bits(got_indices, order[:k], f"{name} k={k} indices")


@pytest.mark.parametrize("name,run,dtype", ALL_RUNS, ids=[c[0] for c in ALL_RUNS])
def test_kthvalue_walks_the_ascending_order(name, run, dtype):
    values, tensor = _build(run, dtype)
    order = _order_along(values, 0, False)

    for k in range(1, values.shape[0] + 1):
        value, index = mt.kthvalue(tensor, k, 0, False)
        _same_bits(value, values[order[k - 1]], f"{name} k={k}")
        assert int(np.asarray(index)) == int(order[k - 1]), f"{name} k={k} index"


# --- median and mode -------------------------------------------------------


def _median_position(values: np.ndarray) -> int:
    """Where `median` should point: the first NaN, or the lower middle.

    A NaN anywhere makes the whole median NaN -- that is why `nanmedian`
    exists -- and the index must then name a NaN rather than whichever
    element the ordering would otherwise have picked.
    """
    if np.issubdtype(values.dtype, np.floating):
        nans = np.flatnonzero(np.isnan(values))
        if nans.size:
            return int(nans[0])
    return int(_ascending(values)[(values.shape[0] - 1) // 2])


@pytest.mark.parametrize("name,run,dtype", ALL_RUNS, ids=[c[0] for c in ALL_RUNS])
def test_median_takes_the_lower_middle_and_can_name_it(name, run, dtype):
    values, tensor = _build(run, dtype)
    position = _median_position(values)

    value, index = mt.median(tensor, 0, False)
    _same_bits(value, values[position], f"{name} median")
    assert int(np.asarray(index)) == position, f"{name} median index"


@pytest.mark.parametrize("name,run,dtype", ALL_RUNS, ids=[c[0] for c in ALL_RUNS])
def test_medians_index_names_the_element_it_returned(name, run, dtype):
    """`gather(input, indices) == values`, which is the whole point of the
    lower-middle convention and the part a NaN used to break."""
    values, tensor = _build(run, dtype)
    value, index = mt.median(tensor, 0, False)
    recovered = values[int(np.asarray(index))]
    _same_bits(np.asarray(value), recovered, f"{name} gather-back")


def test_a_nan_row_reports_the_nans_own_position():
    """The defect this pins: the NaN branch wrote the value and left the index
    at whatever the output buffer held, which was zero. A row whose NaN is not
    at position zero showed a NaN next to the index of an unrelated element."""
    values = np.array([[1.0, 2.0, NAN], [5.0, 4.0, 6.0]], dtype=np.float32)
    tensor = mt.Tensor(values, dtype="float32")
    got_values, got_indices = mt.median(tensor, 1, False)

    indices = np.asarray(got_indices)
    assert indices.tolist() == [2, 0], "row 0 must name its NaN, row 1 its median"
    recovered = np.take_along_axis(values, indices.reshape(-1, 1), axis=1).ravel()
    _same_bits(np.asarray(got_values), recovered, "nan row gather-back")


def test_median_of_an_even_count_is_the_lower_not_the_average():
    """The one place this deliberately parts company with `np.median`."""
    values = np.array([4.0, 1.0, 3.0, 2.0], dtype=np.float32)
    tensor = mt.Tensor(values, dtype="float32")
    value, index = mt.median(tensor, 0, False)
    assert float(np.asarray(value)) == 2.0
    assert int(np.asarray(index)) == 3
    assert float(np.median(values)) == 2.5  # what NumPy would have said
    # ...and the interpolated definition is available, and agrees with NumPy.
    interpolated = mt.quantile(tensor, 0.5, interpolation="linear")
    assert float(np.asarray(interpolated)) == pytest.approx(2.5)


@pytest.mark.parametrize(
    "run,expected_value,expected_index",
    [
        ([1.0, 3.0, 1.0, 3.0, 2.0], 1.0, 0),  # tie on count -> smaller value
        ([5.0, 5.0, 2.0], 5.0, 0),
        ([2.0, 1.0, 1.0], 1.0, 1),  # index is the first position of the winner
        ([9.0], 9.0, 0),
    ],
)
def test_mode_breaks_ties_towards_the_smaller_value(
    run, expected_value, expected_index
):
    tensor = mt.Tensor(np.asarray(run, dtype=np.float32), dtype="float32")
    value, index = mt.mode(tensor, 0, False)
    assert float(np.asarray(value)) == expected_value
    assert int(np.asarray(index)) == expected_index


# --- quantile, against NumPy's own ----------------------------------------

METHODS = ["linear", "lower", "higher", "midpoint", "nearest"]
QUANTILES = [0.0, 0.1, 0.25, 1.0 / 3.0, 0.5, 0.75, 0.9, 1.0]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("q", QUANTILES, ids=[f"q={q:.4g}" for q in QUANTILES])
def test_quantile_matches_numpy(method, q):
    rng = np.random.default_rng(7)
    values = rng.standard_normal(11).astype(np.float64)
    tensor = mt.Tensor(values, dtype="float64")
    got = float(np.asarray(mt.quantile(tensor, q, interpolation=method)))
    assert got == pytest.approx(float(np.quantile(values, q, method=method)), abs=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_quantile_along_a_dimension_matches_numpy(method):
    rng = np.random.default_rng(11)
    values = rng.standard_normal((3, 7)).astype(np.float64)
    tensor = mt.Tensor(values, dtype="float64")
    for dim in (0, 1, -1):
        got = np.asarray(mt.quantile(tensor, 0.4, dim, False, method))
        expected = np.quantile(values, 0.4, axis=dim, method=method)
        assert got == pytest.approx(expected, abs=1e-12), f"dim={dim} {method}"


@pytest.mark.parametrize("method", METHODS)
def test_nanquantile_ignores_nan_the_way_numpy_does(method):
    rng = np.random.default_rng(13)
    values = rng.standard_normal((3, 7)).astype(np.float64)
    values[0, 2] = values[1, 5] = values[2, 0] = NAN
    tensor = mt.Tensor(values, dtype="float64")
    got = np.asarray(mt.nanquantile(tensor, 0.6, 1, False, method))
    expected = np.nanquantile(values, 0.6, axis=1, method=method)
    assert got == pytest.approx(expected, abs=1e-12), method


def test_nanmedian_skips_nan_and_still_takes_the_lower_middle():
    values = np.array([4.0, NAN, 1.0, 3.0, NAN, 2.0], dtype=np.float32)
    tensor = mt.Tensor(values, dtype="float32")
    value = np.asarray(mt.nanmedian(tensor, 0, False))
    # The four finite values are 1, 2, 3, 4; the lower middle is 2.
    assert float(value) == 2.0


# --- the harness must be able to fail --------------------------------------


def test_the_oracle_would_notice_a_reversed_descending_order():
    """The mistake this file exists to catch, made on purpose.

    Reversing the ascending order puts tied elements back to front, which every
    test above would otherwise accept -- the *values* are identical either way.
    Only the indices tell them apart.
    """
    values = np.asarray(FLOAT_RUNS["ties"], dtype=np.float32)
    correct = _order_along(values, 0, True)
    naive = _order_along(values, 0, False)[::-1]

    assert np.array_equal(
        _gather(values, correct, 0), _gather(values, naive, 0)
    ), "the two orders must agree on values, or this proves nothing"
    assert not np.array_equal(correct, naive), "but they must disagree on indices"

    _, actual = mt.sort(mt.Tensor(values, dtype="float32"), 0, True)
    _same_bits(actual, correct, "descending indices")
