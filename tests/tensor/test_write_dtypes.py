# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""What dtype a write may be handed, and what dtype it leaves behind.

Every function here writes values into a tensor, and each had its own answer
to the same two questions. They now have one answer, in two parts.

A value that carries a dtype of its own keeps it, and a disagreement with the
destination is refused -- writing a float64 source into a float32 tensor is
two typed operands disagreeing, not something to silently resolve. A Python
number or list carries no dtype, so it takes the destination's, which is what
`x[i] = 7.0` has always done and what `index_fill` and `masked_fill` already
did with the value they are handed.

Neither part held before. A literal was built at the default dtype and then
checked against the destination, so `put(x, i, 7.0)` worked on a float32
tensor and raised on a float64 one -- the same expression refused for the
dtype of the tensor it was writing into. And `masked_scatter` reached `where`
rather than a `scatter` kernel; `where` promotes, so a float64 source turned a
float32 destination into a float64 result. A write that retypes what it writes
into is the one outcome none of these functions should have.

That is why the tests below run at float64. At float32, the library's default,
a literal happens to arrive with the right dtype and the whole question is
invisible -- which is exactly how it stayed unnoticed.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

DTYPES = ["float32", "float64", "int32", "int64"]


def _dest(dtype, values=(0, 1, 2, 3)):
    return mt.Tensor(np.asarray(values).astype(dtype), dtype=dtype)


def _index(values):
    return mt.Tensor.from_numpy(np.asarray(values, dtype=np.int64))


def _mask(values):
    return mt.Tensor.from_numpy(np.asarray(values, dtype=bool))


def _writes(destination):
    """One entry per way to write a value into `destination`, each a callable
    taking the value and returning the result."""

    index, mask = _index([1, 2]), _mask([False, True, True, False])
    return {
        "put": lambda v: mt.put(destination, index, v),
        "index_copy": lambda v: mt.index_copy(destination, 0, index, v),
        "index_add": lambda v: mt.index_add(destination, 0, index, v),
        "masked_scatter": lambda v: mt.masked_scatter(destination, mask, v),
        "slice_scatter": lambda v: mt.slice_scatter(destination, v, 0, 1, 3),
    }


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", list(_writes(_dest("float32"))))
def test_a_python_list_takes_the_dtype_of_what_it_is_written_into(name, dtype):
    destination = _dest(dtype)
    result = _writes(destination)[name]([7, 8])
    assert str(result.dtype) == dtype
    assert str(destination.dtype) == dtype


@pytest.mark.parametrize("dtype", DTYPES)
def test_a_python_scalar_takes_it_too(dtype):
    """A scalar broadcasts to the region, so it reaches the writes that take
    one value as readily as the ones that take a row."""

    destination = _dest(dtype)
    for write in (
        lambda: mt.put(destination, _index([1, 2]), 7),
        lambda: mt.slice_scatter(destination, 7, 0, 1, 3),
        lambda: mt.select_scatter(destination, 7, 0, 1),
    ):
        assert str(write().dtype) == dtype


@pytest.mark.parametrize("name", list(_writes(_dest("float32"))))
def test_a_source_of_another_dtype_is_refused_rather_than_resolved(name):
    """Not cast, not promoted: the write is declined. Whichever direction the
    disagreement runs, the result of a write must have the dtype of the tensor
    written into, and neither casting silently nor promoting silently gives
    the caller that guarantee."""

    with pytest.raises(TypeError):
        _writes(_dest("float32"))[name](_dest("float64", (7, 8)))
    with pytest.raises(TypeError):
        _writes(_dest("float64"))[name](_dest("float32", (7, 8)))


def test_the_refusal_names_the_destination_as_the_dtype_expected():
    """The message used to name the source as expected and the destination as
    what it got, so its suggestion told the caller to convert the tensor they
    were writing into -- the opposite of the fix."""

    with pytest.raises(TypeError) as raised:
        mt.put(_dest("float64"), _index([1]), _dest("float32", (7,)))
    message = str(raised.value)
    assert "expected Float64" in message and "got Float32" in message


@pytest.mark.parametrize("dtype", DTYPES)
def test_an_array_carries_its_own_dtype_the_way_a_tensor_does(dtype):
    """The rule is about operands with no dtype, not about operands that are
    not tensors. An array has one, so it is held to it."""

    assert (
        str(mt.put(_dest(dtype), _index([1, 2]), np.array([7, 8]).astype(dtype)).dtype)
        == dtype
    )
    with pytest.raises(TypeError):
        mt.put(_dest("float64"), _index([1, 2]), np.array([7.0, 8.0], dtype=np.float32))


@pytest.mark.parametrize("dtype", DTYPES)
def test_assignment_agrees_with_the_functional_writes(dtype):
    """`x[1:3] = 7` is the form all of these are the expression version of, so
    it answers both questions the same way."""

    destination = _dest(dtype)
    destination[1:3] = 7
    assert str(destination.dtype) == dtype
    np.testing.assert_array_equal(destination.numpy(), np.array([0, 7, 7, 3]))

    other = "float32" if dtype != "float32" else "float64"
    with pytest.raises(TypeError):
        destination[1:3] = _dest(other, (7, 7, 7, 7))


def test_a_mismatched_assignment_is_an_exception_and_not_a_panic():
    """The write reads the value through the destination's own accessor, so a
    value of another dtype had no slice there to read and the `unwrap` on it
    ended the process. Refusing before the accessor is what makes it a
    catchable error."""

    destination = _dest("float32")
    try:
        destination[1:3] = _dest("float64", (7, 7, 7, 7))
    except Exception:  # noqa: BLE001 - declining the value is the point
        pass
    except BaseException as exc:  # pragma: no cover - the failure this exists for
        pytest.fail(
            f"assignment raised {type(exc).__name__}, which is not an Exception"
        )
    np.testing.assert_array_equal(destination.numpy(), np.array([0, 1, 2, 3]))
