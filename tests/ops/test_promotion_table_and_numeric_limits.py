# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Two things that are easy to change by accident and hard to notice.

**The promotion table.** Individual pairs are checked in `test_math_ops.py`,
but not the table as a whole, and it is the kind of thing a refactor moves one
cell of. Widening the integer accumulation in `sum`/`prod` came close to doing
exactly that -- the reductions changed their output dtype deliberately, and
nothing would have complained if binary arithmetic had drifted with them.

The table also encodes a deliberate divergence from NumPy that is worth having
stated in a test rather than only in prose: an integer operand takes the *float
operand's width*, so `int64 + float32` is `float32`. NumPy promotes that pair
to `float64` on the grounds that `int64` does not fit in `float32`. PyTorch
does what this library does, and for an array library aimed at models it is the
better trade -- the NumPy rule silently doubles the memory and halves the speed
of any expression that mixes an index tensor into an activation.

**The numerical limits.** `softmax`, `sigmoid`, `log1p` and `expm1` all exist in
a specific form because the obvious form loses. Written naively they overflow to
`inf`, saturate to `0/0`, or lose every significant digit near zero. Those
properties live in kernel internals, so a rewrite can drop one and still pass
every test that only checks ordinary inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

DTYPES = ["bool", "int32", "int64", "float32", "float64"]

# The complete binary promotion table. Read it as `EXPECTED[a][b]`.
#
# Symmetric by construction, and the ordering is a total one: bool < int32 <
# int64 < float32 < float64, with the *wider category* winning outright rather
# than the wider *width*. That last part is the NumPy divergence: NumPy would
# put `float64` in the four (int32|int64, float32) cells.
EXPECTED = {
    "bool": {
        "bool": "bool",
        "int32": "int32",
        "int64": "int64",
        "float32": "float32",
        "float64": "float64",
    },
    "int32": {
        "bool": "int32",
        "int32": "int32",
        "int64": "int64",
        "float32": "float32",
        "float64": "float64",
    },
    "int64": {
        "bool": "int64",
        "int32": "int64",
        "int64": "int64",
        "float32": "float32",
        "float64": "float64",
    },
    "float32": {
        "bool": "float32",
        "int32": "float32",
        "int64": "float32",
        "float32": "float32",
        "float64": "float64",
    },
    "float64": {
        "bool": "float64",
        "int32": "float64",
        "int64": "float64",
        "float32": "float64",
        "float64": "float64",
    },
}

NUMPY_DIVERGES = {
    ("int32", "float32"),
    ("float32", "int32"),
    ("int64", "float32"),
    ("float32", "int64"),
}


def _operand(dtype):
    if dtype == "bool":
        values = np.array([True, False, True, True])
    elif dtype.startswith("int"):
        values = np.array([3, 2, 5, 7], dtype=dtype)
    else:
        values = np.array([3.0, 2.0, 5.0, 7.0], dtype=dtype)
    return values, mt.Tensor(values, dtype=dtype)


@pytest.mark.parametrize("left", DTYPES)
@pytest.mark.parametrize("right", DTYPES)
@pytest.mark.parametrize("op", ["+", "*", "-"])
def test_the_promotion_table_holds_for_every_pair(left, right, op):
    if op == "-" and left == "bool" and right == "bool":
        pytest.skip("bool - bool has no boolean result to land in and is rejected")

    _, a = _operand(left)
    _, b = _operand(right)
    got = {"+": lambda: a + b, "*": lambda: a * b, "-": lambda: a - b}[op]()
    assert str(got.dtype) == EXPECTED[left][right], f"{left} {op} {right}"


@pytest.mark.parametrize("left", DTYPES)
@pytest.mark.parametrize("right", DTYPES)
def test_the_values_match_numpy_once_the_dtype_is_accounted_for(left, right):
    """The divergence is in the *width* the result lands in, not in the
    arithmetic: cast NumPy's answer into our dtype and the values agree."""
    a_np, a = _operand(left)
    b_np, b = _operand(right)
    got = a + b
    want = (a_np + b_np).astype(EXPECTED[left][right])
    np.testing.assert_allclose(got.numpy(), want, rtol=1e-6)


@pytest.mark.parametrize("pair", sorted(NUMPY_DIVERGES))
def test_the_numpy_divergence_is_real_and_deliberate(pair):
    """Pins the disagreement itself. If NumPy ever adopts the PyTorch rule this
    test fails, which is the moment to revisit the prose in the API reference
    rather than to quietly follow along."""
    left, right = pair
    a_np, a = _operand(left)
    b_np, b = _operand(right)

    assert str((a + b).dtype) == "float32"
    assert (a_np + b_np).dtype.name == "float64"


def test_promotion_is_symmetric():
    """`a + b` and `b + a` cannot disagree about the result dtype."""
    for left in DTYPES:
        for right in DTYPES:
            _, a = _operand(left)
            _, b = _operand(right)
            assert str((a + b).dtype) == str((b + a).dtype), f"{left} vs {right}"


# --- numerical limits -------------------------------------------------------


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_softmax_does_not_overflow_on_large_inputs(dtype):
    """`exp(1000)` is `inf`, so a softmax that does not subtract the row
    maximum first returns `inf/inf = NaN` for every element of the row."""
    values = np.array(
        [[1000.0, 1001.0, 999.0], [-1000.0, -1001.0, -999.0]], dtype=dtype
    )
    got = mt.softmax(mt.Tensor(values, dtype=dtype), dim=1).numpy()

    assert np.isfinite(got).all(), got
    shifted = np.exp(values.astype(np.float64) - values.max(axis=1, keepdims=True))
    want = shifted / shifted.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(got.sum(axis=1), 1.0, rtol=1e-6)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_logsumexp_does_not_overflow_on_large_inputs(dtype):
    values = np.array(
        [[1000.0, 1001.0, 999.0], [-1000.0, -1001.0, -999.0]], dtype=dtype
    )
    got = mt.Tensor(values, dtype=dtype).logsumexp(1).numpy()

    assert np.isfinite(got).all(), got
    wide = values.astype(np.float64)
    want = np.log(
        np.exp(wide - wide.max(axis=1, keepdims=True)).sum(axis=1)
    ) + wide.max(axis=1)
    np.testing.assert_allclose(got, want, rtol=1e-5)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_sigmoid_saturates_rather_than_overflowing(dtype):
    """`1 / (1 + exp(-x))` overflows for very negative `x`; the stable form
    switches expression at zero. The answers at the extremes are exactly 0 and
    exactly 1, and never NaN."""
    values = np.array([-800.0, -100.0, -1.0, 0.0, 1.0, 100.0, 800.0], dtype=dtype)
    got = mt.Tensor(values, dtype=dtype).sigmoid().numpy()

    assert not np.isnan(got).any(), got
    assert ((got >= 0.0) & (got <= 1.0)).all(), got
    assert got[0] == pytest.approx(0.0, abs=1e-12)
    assert got[-1] == pytest.approx(1.0, abs=1e-12)
    assert got[3] == pytest.approx(0.5, abs=1e-7)
    # Monotone across the whole range, saturation included.
    assert np.all(np.diff(got) >= 0.0), got


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_log1p_and_expm1_keep_their_precision_near_zero(dtype):
    """The entire reason these two exist rather than `log(1 + x)` and
    `exp(x) - 1`: near zero the naive forms lose the answer to cancellation.
    At `1e-18` in float64, `log(1 + x)` is exactly 0 and `log1p(x)` is `x`."""
    tiny = np.array([1e-7, 1e-10, 1e-12, -1e-7, -1e-12], dtype=dtype)
    t = mt.Tensor(tiny, dtype=dtype)

    rtol = 1e-6 if dtype == "float32" else 1e-12
    np.testing.assert_allclose(t.log1p().numpy(), np.log1p(tiny), rtol=rtol)
    np.testing.assert_allclose(t.expm1().numpy(), np.expm1(tiny), rtol=rtol)

    # And they beat the naive spelling, which is the point.
    naive_log = np.log(1.0 + tiny.astype(np.float64))
    exact_log = np.log1p(tiny.astype(np.float64))
    worst_naive = np.max(np.abs(naive_log - exact_log) / np.abs(exact_log))
    worst_ours = np.max(
        np.abs(t.log1p().numpy().astype(np.float64) - exact_log) / np.abs(exact_log)
    )
    assert worst_ours < worst_naive
