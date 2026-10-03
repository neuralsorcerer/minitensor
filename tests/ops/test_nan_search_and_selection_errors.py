# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A NaN is searched for where `sort` puts it, and a bad `k` is a ValueError.

`sort` places NaN after every number. A NaN in the searched sequence already
behaved that way, but a NaN *value* compared less than nothing and landed at
index 0 -- before everything it sorts after -- so `searchsorted`, `bucketize`
and `digitize` answered for a different order than the one the sequence was
sorted in.

`topk`, `sort` and `argsort` raised RuntimeError for an invalid argument, as a
special case, where every other operation raises ValueError for the same kind
of mistake; and a negative `k` was an OverflowError from the method but a
RuntimeError from the function.
"""

import math

import numpy as np
import pytest

import minitensor as mt

NAN = math.nan


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("right", [False, True])
def test_a_nan_value_lands_where_sort_put_the_nans(dtype, right):
    raw = np.array([3.0, NAN, 1.0, NAN, -np.inf, 2.0], dtype=dtype)
    ordered = mt.from_numpy(raw).sort()[0]
    assert np.isnan(ordered.numpy()[-2:]).all()
    queries = mt.from_numpy(np.array([NAN, 1.0, np.inf, -np.inf], dtype=dtype))
    got = mt.searchsorted(ordered, queries, right=right).numpy().tolist()
    expected = np.searchsorted(
        np.sort(raw), queries.numpy(), side="right" if right else "left"
    )
    assert got == expected.tolist()
    # The NaN's answer is the run of NaNs: its start, or past its end.
    assert got[0] == (6 if right else 4)


def test_a_nan_value_in_a_sequence_without_nans_goes_to_the_end():
    ordered = mt.Tensor([0.0, 1.0, 2.0])
    for right in (False, True):
        assert mt.searchsorted(ordered, mt.Tensor([NAN]), right=right).tolist() == [3]


def test_a_nan_value_in_a_batch_of_sequences():
    ordered = mt.Tensor([[0.0, 1.0, NAN], [0.0, 1.0, 2.0]])
    values = mt.Tensor([[NAN], [NAN]])
    assert mt.searchsorted(ordered, values).tolist() == [[2], [3]]
    assert mt.searchsorted(ordered, values, right=True).tolist() == [[3], [3]]


def test_bucketize_and_digitize_follow_the_same_order():
    edges = mt.Tensor([0.0, 1.0, 2.0])
    values = mt.Tensor([NAN, 0.5])
    assert mt.bucketize(values, edges).tolist() == [3, 1]
    # Past every increasing edge, and before every decreasing one.
    assert mt.digitize(values, edges).tolist() == [3, 1]
    assert mt.digitize(values, mt.Tensor([2.0, 1.0, 0.0])).tolist() == [0, 2]


def test_a_nan_is_still_left_out_of_a_histogram_and_never_a_member():
    sample = mt.Tensor([[NAN, 0.5], [0.25, 0.75]])
    counts, _ = mt.histogramdd(sample, bins=2, range=[0.0, 1.0, 0.0, 1.0])
    assert counts.numpy().sum() == 1.0
    assert mt.isin(mt.Tensor([NAN, 1.0]), mt.Tensor([1.0, NAN])).tolist() == [
        False,
        True,
    ]


def test_an_integer_sequence_is_unaffected():
    ordered = mt.from_numpy(np.array([1, 3, 3, 7], dtype=np.int64))
    values = mt.from_numpy(np.array([0, 3, 8], dtype=np.int64))
    assert mt.searchsorted(ordered, values).tolist() == [0, 1, 4]
    assert mt.searchsorted(ordered, values, right=True).tolist() == [0, 3, 4]


@pytest.mark.parametrize(
    "call",
    [
        lambda t: t.topk(4),
        lambda t: mt.topk(t, 4),
    ],
    ids=["method", "function"],
)
def test_a_k_past_the_axis_is_a_value_error(call):
    with pytest.raises(ValueError, match="selected index k out of range"):
        call(mt.Tensor([1.0, 2.0, 3.0]))


@pytest.mark.parametrize(
    "call",
    [lambda t: t.topk(-1), lambda t: mt.topk(t, -1)],
    ids=["method", "function"],
)
def test_a_negative_k_is_a_value_error_naming_k(call):
    with pytest.raises(ValueError, match="k must be non-negative, got -1"):
        call(mt.Tensor([1.0, 2.0, 3.0]))
