# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Every reduction that reports an index must name the element it returned.

Twelve entry points return `(values, indices)`. For all of them one property
holds and is worth stating once rather than twelve times:

    take_along_dim(input, indices, dim) == values

bit for bit. It is the whole reason these reductions return an index at all --
`median` gives the *lower* of two middle values rather than averaging them
precisely so that the answer is an element the caller can point at -- and it is
cheap to check, which matters because the way it breaks is quiet.

`median` broke it. A NaN anywhere makes the slice's median NaN, and the branch
that handled that wrote the value and returned without writing the index, so
the index was whatever the output buffer happened to hold: zero. `median` of
`[1, nan, 3, 2]` reported `nan` at index 0, and index 0 is `1.0`. Nothing
failed, because no test compared the two outputs to each other.

That is a shape of mistake, not a one-off: any of these kernels writes two
buffers in step, and a branch that leaves one of them alone produces a
plausible-looking pair. NaN is the input that finds it, because NaN is what
sends these kernels down their special-cased branches.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

NAN = float("nan")

CASES = {
    # A NaN away from position zero, so an unwritten index cannot pass by luck.
    "nan-off-zero": [[1.0, NAN, 3.0, 2.0], [5.0, 4.0, 6.0, 7.0]],
    "nan-everywhere": [[NAN, NAN, NAN, NAN], [1.0, NAN, 2.0, NAN]],
    "ties": [[2.0, 1.0, 2.0, 1.0], [3.0, 3.0, 3.0, 3.0]],
    "no-nan": [[4.0, 1.0, 3.0, 2.0], [5.0, 8.0, 6.0, 7.0]],
    "infinities": [[float("inf"), 1.0, -float("inf"), NAN], [0.0, -0.0, 1.0, -1.0]],
}


def _tensor(values, dtype):
    return mt.Tensor(values, dtype=dtype)


def _gather(values: np.ndarray, indices: np.ndarray, dim: int) -> np.ndarray:
    """`take_along_dim`, written out so the test does not lean on the library."""
    return np.take_along_axis(values, indices, axis=dim)


def _check_pair(source: np.ndarray, values, indices, dim: int, what: str) -> None:
    got_values = np.asarray(values)
    got_indices = np.asarray(indices)

    # Reductions drop the axis; scans keep it. Either way the indices index
    # `source` along `dim`, so restore the axis before gathering.
    if got_indices.ndim == source.ndim - 1:
        gathered = _gather(source, np.expand_dims(got_indices, dim), dim)
        gathered = np.squeeze(gathered, axis=dim)
    else:
        gathered = _gather(source, got_indices, dim)

    assert (
        gathered.shape == got_values.shape
    ), f"{what}: {gathered.shape} vs {got_values.shape}"
    assert gathered.tobytes() == got_values.tobytes(), (
        f"{what}: values {got_values.ravel()[:8]} but indices {got_indices.ravel()[:8]} "
        f"name {gathered.ravel()[:8]}"
    )


# `(name, call)` where `call(tensor, dim)` returns `(values, indices)`.
PAIRED = [
    ("median", lambda t, d: mt.median(t, d, False)),
    ("mode", lambda t, d: mt.mode(t, d, False)),
    ("kthvalue(k=1)", lambda t, d: mt.kthvalue(t, 1, d, False)),
    ("kthvalue(k=2)", lambda t, d: mt.kthvalue(t, 2, d, False)),
    ("topk(k=1, largest)", lambda t, d: mt.topk(t, 1, d, True, True)),
    ("topk(k=2, largest)", lambda t, d: mt.topk(t, 2, d, True, True)),
    ("topk(k=2, smallest)", lambda t, d: mt.topk(t, 2, d, False, True)),
    ("sort", lambda t, d: mt.sort(t, d, False)),
    ("sort(descending)", lambda t, d: mt.sort(t, d, True)),
    ("max", lambda t, d: mt.max(t, d)),
    ("min", lambda t, d: mt.min(t, d)),
    ("nanmax", lambda t, d: mt.nanmax(t, d)),
    ("nanmin", lambda t, d: mt.nanmin(t, d)),
    ("cummax", lambda t, d: mt.cummax(t, d)),
    ("cummin", lambda t, d: mt.cummin(t, d)),
]


@pytest.mark.parametrize("case", sorted(CASES), ids=sorted(CASES))
@pytest.mark.parametrize("name,call", PAIRED, ids=[n for n, _ in PAIRED])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_the_index_names_the_value(case, name, call, dtype):
    source = np.asarray(
        CASES[case], dtype=np.float32 if dtype == "float32" else np.float64
    )
    for dim in (1, 0):
        result = call(_tensor(source, dtype), dim)
        assert isinstance(result, tuple), f"{name} stopped returning a pair"
        _check_pair(source, result[0], result[1], dim, f"{name} {case} dim={dim}")


@pytest.mark.parametrize("case", sorted(CASES), ids=sorted(CASES))
def test_argmax_and_argmin_name_the_extreme_they_found(case):
    """`argmax`/`argmin` return only an index, so the property is that the
    element they name equals what `max`/`min` returned."""
    source = np.asarray(CASES[case], dtype=np.float32)
    tensor = _tensor(source, "float32")
    for dim in (1, 0):
        for arg, reduce_to in ((mt.argmax, mt.max), (mt.argmin, mt.min)):
            indices = np.asarray(arg(tensor, dim))
            named = np.squeeze(
                _gather(source, np.expand_dims(indices, dim), dim), axis=dim
            )
            expected = np.asarray(reduce_to(tensor, dim)[0])
            assert named.tobytes() == expected.tobytes(), (
                f"{arg.__name__} {case} dim={dim}: names {named} but the reduction "
                f"returned {expected}"
            )


def test_the_check_would_have_caught_medians_unwritten_index():
    """The defect this file generalises, spelled out.

    Before the fix, `median` of the first row reported `nan` with index 0, and
    index 0 is `1.0`. Both outputs looked reasonable on their own; only holding
    them against each other showed the disagreement.
    """
    source = np.array([[1.0, NAN, 3.0, 2.0], [5.0, 4.0, 6.0, 7.0]], dtype=np.float32)
    values, indices = mt.median(_tensor(source, "float32"), 1, False)

    assert np.isnan(np.asarray(values)[0]), "a NaN in the slice makes the median NaN"
    assert int(np.asarray(indices)[0]) == 1, "and the index must name that NaN"
    _check_pair(source, values, indices, 1, "median")
