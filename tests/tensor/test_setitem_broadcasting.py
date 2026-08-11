# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Assignment matches the value to the selection by shape, not by element count.

`t[subscript] = value` used to check `value.numel() == selection.numel()` and
then write the value's elements in flat order. That is wrong in both directions,
and the permissive direction is the one that costs data.

Any value with the right *number* of elements was accepted whatever its shape,
so assigning a transposed block stored the wrong arrangement and said nothing:

    t = Tensor(arange(24).reshape(2, 3, 4))
    m = Tensor(arange(12).reshape(4, 3))
    t[0] = m          # selection is (3, 4); numel matches, so this "worked"

That has to raise, because `(4, 3)` does not broadcast to `(3, 4)`. Here it
silently wrote `m`'s elements row-major into a differently shaped block -- the
kind of mistake that surfaces much later as a model that trains badly.

The same check refused real broadcasts, since their counts differ. `t[0] = row`
could not fill a `(3, 4)` block from a `(4,)` row, which broadcasting allows
and which `t[mask] = value` already allowed here.

Both follow from matching shapes right-aligned instead: each of the value's
dimensions equals the selection's or is 1, extra leading dimensions of the value
must be 1, and a broadcast dimension reads the same element for every coordinate
along it. The sweep below is the real test -- every subscript against every
value shape, accepted or rejected the same way NumPy does, and where both accept,
storing the same bytes.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

BASE = np.arange(24, dtype=np.float64).reshape(2, 3, 4)

SUBSCRIPTS = {
    "[0]": 0,
    "[-1]": -1,
    "[:, 0]": (slice(None), 0),
    "[..., 0]": (Ellipsis, 0),
    "[:1]": slice(None, 1),
    "[::2]": slice(None, None, 2),
    "[0, 1]": (0, 1),
    "[:, 1:3]": (slice(None), slice(1, 3)),
    "[1, :, 2]": (1, slice(None), 2),
    "[:]": slice(None),
    "[1:1]": slice(1, 1),
}

# Exact matches, genuine broadcasts, same-count-different-shape, and mismatches.
VALUE_SHAPES = [
    (),
    (1,),
    (2,),
    (3,),
    (4,),
    (8,),
    (12,),
    (24,),
    (1, 1),
    (1, 4),
    (3, 1),
    (2, 4),
    (3, 4),
    (4, 3),
    (1, 1, 1),
    (1, 1, 4),
    (1, 2, 4),
    (1, 3, 4),
    (2, 1, 4),
    (2, 3, 4),
]


def _value(shape):
    if not shape:
        return np.float64(100.0)
    return np.arange(int(np.prod(shape)), dtype=np.float64).reshape(shape) + 100.0


@pytest.mark.parametrize("label", list(SUBSCRIPTS), ids=list(SUBSCRIPTS))
@pytest.mark.parametrize("shape", VALUE_SHAPES, ids=[str(s) for s in VALUE_SHAPES])
def test_assignment_agrees_with_numpy(label, shape):
    """Accepted or rejected the same way, and when accepted, the same result."""
    subscript = SUBSCRIPTS[label]
    value = _value(shape)

    reference = BASE.copy()
    numpy_accepted = True
    try:
        reference[subscript] = value
    except Exception:
        numpy_accepted = False

    tensor = mt.Tensor(BASE.copy(), dtype="float64")
    got_accepted = True
    try:
        tensor[subscript] = mt.Tensor(value, dtype="float64")
    except Exception:
        got_accepted = False

    assert got_accepted == numpy_accepted, (
        f"t{label} = {shape}: NumPy "
        f"{'accepts' if numpy_accepted else 'rejects'} and this "
        f"{'accepts' if got_accepted else 'rejects'}"
    )
    if numpy_accepted:
        np.testing.assert_array_equal(tensor.numpy(), reference)


# --- the direction that silently wrote wrong data ---------------------------


def test_a_transposed_block_is_refused():
    """Same element count, incompatible shape. This used to be accepted and
    written row-major, which is a wrong answer rather than an error."""
    tensor = mt.Tensor(BASE.copy(), dtype="float64")
    transposed = mt.Tensor(
        np.arange(12, dtype=np.float64).reshape(4, 3), dtype="float64"
    )

    with pytest.raises(Exception) as excinfo:
        tensor[0] = transposed

    message = str(excinfo.value)
    assert "[4, 3]" in message and "[3, 4]" in message, message


@pytest.mark.parametrize("shape", [(12,), (2, 6), (6, 2), (4, 3)])
def test_no_reshaping_is_inferred_from_a_matching_count(shape):
    tensor = mt.Tensor(BASE.copy(), dtype="float64")
    with pytest.raises(Exception):
        tensor[0] = mt.Tensor(_value(shape), dtype="float64")


def test_the_message_names_both_shapes():
    """The fix is a shape rule, so the complaint has to report shapes."""
    tensor = mt.Tensor(BASE.copy(), dtype="float64")
    with pytest.raises(Exception) as excinfo:
        tensor[:, 0] = mt.Tensor(np.zeros((3, 3)), dtype="float64")
    message = str(excinfo.value)
    assert "[3, 3]" in message
    assert "[2, 4]" in message
    assert "broadcast" in message


# --- the direction that refused legitimate assignments ----------------------


@pytest.mark.parametrize(
    "subscript,shape",
    [
        (0, (4,)),
        (0, (1, 4)),
        (0, (3, 1)),
        (0, (1, 1)),
        ((slice(None), 0), (4,)),
        ((slice(None), 0), (1,)),
        (slice(None, 1), (3, 4)),
        (slice(None), (4,)),
    ],
)
def test_a_broadcast_value_fills_the_selection(subscript, shape):
    reference = BASE.copy()
    reference[subscript] = _value(shape)

    tensor = mt.Tensor(BASE.copy(), dtype="float64")
    tensor[subscript] = mt.Tensor(_value(shape), dtype="float64")

    np.testing.assert_array_equal(tensor.numpy(), reference)


def test_extra_leading_ones_on_the_value_are_stripped():
    """They are ignored rather than counted against the rank."""
    reference = BASE.copy()
    reference[0] = _value((1, 3, 4))

    tensor = mt.Tensor(BASE.copy(), dtype="float64")
    tensor[0] = mt.Tensor(_value((1, 3, 4)), dtype="float64")

    np.testing.assert_array_equal(tensor.numpy(), reference)


# --- what must not have changed ---------------------------------------------


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64", "bool"])
def test_every_dtype_still_assigns(dtype):
    """The write loop is repeated per dtype, so a change to it has to be made
    in all of them."""
    values = (
        (np.arange(6) % 2).astype(dtype)
        if dtype == "bool"
        else np.arange(6).astype(dtype)
    )
    filler = True if dtype == "bool" else 1

    reference = values.reshape(2, 3).copy()
    reference[0] = filler
    tensor = mt.Tensor(values.reshape(2, 3), dtype=dtype)
    tensor[0] = filler
    np.testing.assert_array_equal(tensor.numpy(), reference)

    row = np.array([filler] * 3).astype(dtype)
    reference = values.reshape(2, 3).copy()
    reference[0] = row
    tensor = mt.Tensor(values.reshape(2, 3), dtype=dtype)
    tensor[0] = mt.Tensor(row, dtype=dtype)
    np.testing.assert_array_equal(tensor.numpy(), reference)


def test_mask_assignment_is_unaffected():
    """It already broadcast correctly and goes through a different path."""
    reference = BASE.copy()
    mask = np.array([True, False])
    reference[mask] = 7.0

    tensor = mt.Tensor(BASE.copy(), dtype="float64")
    tensor[mt.Tensor(mask, dtype="bool")] = 7.0

    np.testing.assert_array_equal(tensor.numpy(), reference)


def test_an_empty_selection_writes_nothing():
    reference = BASE.copy()
    reference[1:1] = 9.0

    tensor = mt.Tensor(BASE.copy(), dtype="float64")
    tensor[1:1] = 9.0

    np.testing.assert_array_equal(tensor.numpy(), reference)
