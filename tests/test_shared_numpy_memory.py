# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Importing a NumPy array's memory, and exporting a tensor's through the buffer
protocol.

`test_array_interface.py` covers the export side through
`__array_interface__`. These are the other two halves: `from_numpy_shared`,
which for a long time copied and said so in a comment, and the buffer protocol,
which reaches everything that does not speak NumPy.

Sharing is not observable from a timing, so it is checked the only way that
settles it -- write through one side and read the other, and watch what happens
to the array when the last tensor holding it goes away.
"""

from __future__ import annotations

import gc
import sys

import numpy as np
import pytest

import minitensor as mt

# --- importing an array's memory ---------------------------------------------


def test_from_numpy_shared_shares():
    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    tensor = mt.Tensor.from_numpy_shared(source)

    source[0, 0] = -5.0
    assert float(np.asarray(tensor)[0, 0]) == -5.0


def test_from_numpy_copies():
    """The contrast, so the two are not confused for each other."""
    source = np.arange(4, dtype=np.float32)
    tensor = mt.Tensor.from_numpy(source)

    source[0] = -5.0
    assert float(np.asarray(tensor)[0]) == 0.0


def test_the_tensor_keeps_the_array_alive():
    def make():
        return mt.Tensor.from_numpy_shared(np.arange(256, dtype=np.float64))

    tensor = make()
    gc.collect()
    # If the array had been collected this reads freed memory; the value is
    # what says it did not.
    assert float(mt.sum(tensor).item()) == float(255 * 256 // 2)


def test_the_array_is_referenced_rather_than_copied():
    source = np.arange(4, dtype=np.float32)
    before = sys.getrefcount(source)
    tensor = mt.Tensor.from_numpy_shared(source)
    assert sys.getrefcount(source) > before
    del tensor
    gc.collect()
    assert sys.getrefcount(source) == before


def test_a_shared_tensor_computes_correctly():
    source = np.arange(9, dtype=np.float64).reshape(3, 3)
    shared = mt.Tensor.from_numpy_shared(source)
    np.testing.assert_allclose(np.asarray(mt.matmul(shared, shared)), source @ source)


def test_contiguous_copy_detaches_from_the_array():
    """`.contiguous()` on shared storage gives a tensor that owns its bytes."""
    source = np.arange(4, dtype=np.float64)
    owned = mt.Tensor.from_numpy_shared(source).contiguous().clone()
    source[0] = 99.0
    assert float(np.asarray(owned)[0]) == 0.0


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64", "bool"])
def test_every_shareable_dtype_round_trips(dtype):
    source = np.array([[1, 0, 1], [0, 1, 0]]).astype(dtype)
    tensor = mt.Tensor.from_numpy_shared(source)
    np.testing.assert_array_equal(np.asarray(tensor), source)


def test_an_empty_array_shares():
    source = np.zeros((0, 3), dtype=np.float32)
    tensor = mt.Tensor.from_numpy_shared(source)
    assert tuple(tensor.shape) == (0, 3)


# --- what cannot be shared is refused, not quietly copied --------------------


def test_non_contiguous_is_refused():
    source = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]
    with pytest.raises(ValueError, match="C-contiguous"):
        mt.Tensor.from_numpy_shared(source)


def test_an_unstorable_dtype_is_refused():
    with pytest.raises(ValueError, match="dtype"):
        mt.Tensor.from_numpy_shared(np.arange(4, dtype=np.complex128))


def test_a_byte_swapped_array_is_refused():
    source = np.arange(4, dtype=np.dtype(">f8" if sys.byteorder == "little" else "<f8"))
    with pytest.raises(ValueError, match="byte order"):
        mt.Tensor.from_numpy_shared(source)


def test_something_that_is_not_an_array_is_refused():
    with pytest.raises(ValueError, match="NumPy array"):
        mt.Tensor.from_numpy_shared([1.0, 2.0, 3.0])


# --- exporting through the buffer protocol -----------------------------------


def test_memoryview_sees_the_tensor():
    tensor = mt.Tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype="float64")
    with memoryview(tensor) as view:
        assert view.shape == (2, 3)
        assert view.format == "d"
        assert view.readonly
        assert view.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]


@pytest.mark.parametrize(
    "dtype,format_code",
    [("float32", "f"), ("float64", "d"), ("int32", "i"), ("int64", "q"), ("bool", "?")],
)
def test_each_dtype_reports_its_struct_code(dtype, format_code):
    with memoryview(mt.Tensor([1, 0, 1], dtype=dtype)) as view:
        assert view.format == format_code


def test_the_exported_buffer_cannot_be_written():
    with memoryview(mt.Tensor([1.0, 2.0, 3.0], dtype="float32")) as view:
        assert view.readonly
        with pytest.raises(TypeError, match="read-only"):
            view[0] = 2.0


def test_frombuffer_reads_the_same_bytes():
    tensor = mt.Tensor(list(range(8)), dtype="float32")
    with memoryview(tensor) as view:
        rebuilt = np.frombuffer(view, dtype=np.float32)
        np.testing.assert_array_equal(rebuilt, np.asarray(tensor))


def test_the_buffer_keeps_the_tensor_alive():
    def make():
        return memoryview(mt.Tensor(list(range(64)), dtype="float64"))

    view = make()
    gc.collect()
    assert view[0] == 0.0 and view[63] == 63.0
    view.release()


def test_exporting_and_releasing_repeatedly_is_stable():
    # `__getbuffer__` leaks a shape/stride allocation into the view and
    # `__releasebuffer__` reclaims it; a mismatch shows up here as a leak or a
    # double free rather than anywhere useful.
    tensor = mt.Tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    for _ in range(1000):
        with memoryview(tensor) as view:
            assert view.shape == (2, 2)


def test_a_zero_dimensional_tensor_exports():
    with memoryview(mt.Tensor(3.5, dtype="float64")) as view:
        assert view.shape == ()
        assert view.tolist() == 3.5
