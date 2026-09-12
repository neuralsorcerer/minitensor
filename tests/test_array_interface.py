# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`numpy.asarray(tensor)` hands back the tensor's own memory.

Every tensor is contiguous and row-major, which is exactly what an array
header describes, so there is nothing to rearrange and nothing to copy: a
16MB tensor crosses in 9us where `.numpy()` takes 15ms. The array is read-only
because several tensors can share one buffer -- `detach`, `reshape`, a no-op
`astype` -- and each is supposed to have its own values, so a NumPy array
writing into that buffer would change all of them at once.

The copying spellings are still there and still writeable: `numpy.array(t)`
and `t.numpy()`. The point of this is that a caller who only wants to *read*
the values no longer pays for a copy of them to say so.
"""

from __future__ import annotations

import gc
import sys

import numpy as np
import pytest

import minitensor as mt

_DTYPES = ["float32", "float64", "int32", "int64", "bool"]


def _address(tensor):
    return tensor.__array_interface__["data"][0]


# --- it is the same memory ---------------------------------------------------


def test_asarray_points_at_the_tensors_own_buffer():
    tensor = mt.Tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    array = np.asarray(tensor)
    assert array.__array_interface__["data"][0] == _address(tensor)
    np.testing.assert_array_equal(array, [[1.0, 2.0], [3.0, 4.0]])


def test_two_views_of_one_tensor_share_their_memory():
    tensor = mt.Tensor([1.0, 2.0, 3.0], dtype="float32")
    assert np.shares_memory(np.asarray(tensor), np.asarray(tensor))


def test_a_reshape_is_the_same_buffer_seen_differently():
    tensor = mt.Tensor([1.0, 2.0, 3.0, 4.0], dtype="float32")
    assert _address(tensor.reshape([2, 2])) == _address(tensor)


def test_crossing_over_does_not_scale_with_the_tensor():
    """A copy would; this is a header.

    The threshold is chosen so far above what a header costs, and so far below
    what copying 64MB costs, that it cannot be met by a fast copy.
    """

    import time

    big = mt.zeros([4000, 4000], dtype="float32")  # 64MB
    start = time.perf_counter()
    array = np.asarray(big)
    elapsed = time.perf_counter() - start

    assert array.shape == (4000, 4000)
    assert elapsed < 5e-3, f"{elapsed * 1e6:.0f}us -- something copied"


# --- it is read-only, and why ------------------------------------------------


def test_the_view_refuses_to_be_written_to():
    array = np.asarray(mt.Tensor([1.0, 2.0], dtype="float32"))
    assert not array.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        array[0] = 5.0


def test_the_copying_spellings_are_still_writeable():
    tensor = mt.Tensor([1.0, 2.0], dtype="float32")

    for made in (np.array(tensor), tensor.numpy(), tensor.numpy_copy()):
        assert made.flags.writeable
        made[0] = 99.0
        # And the tensor is untouched, which is what a copy means.
        np.testing.assert_array_equal(tensor.numpy(), [1.0, 2.0])


def test_a_dtype_conversion_still_produces_a_copy_to_own():
    tensor = mt.Tensor([1, 2, 3], dtype="float32")
    converted = np.asarray(tensor, dtype=np.float64)
    assert converted.dtype == np.float64
    assert not np.shares_memory(converted, np.asarray(tensor))
    np.testing.assert_array_equal(converted, [1.0, 2.0, 3.0])


# --- the sharing goes both ways ----------------------------------------------


def test_writing_the_tensor_in_place_writes_through_a_view_it_alone_holds():
    tensor = mt.Tensor([1.0, 2.0, 3.0], dtype="float32")
    array = np.asarray(tensor)
    tensor.fill_(9.0)
    np.testing.assert_array_equal(array, [9.0, 9.0, 9.0])


def test_writing_a_shared_buffer_copies_and_leaves_the_view_on_the_old_one():
    """Copy-on-write, seen from outside.

    `detach` shares the buffer, so filling the tensor cannot write into it --
    that would change the detached tensor too. The tensor moves to a fresh
    buffer and the array stays on the old one, still valid and still holding
    the values it had.
    """

    tensor = mt.Tensor([1.0, 2.0, 3.0], dtype="float32")
    detached = tensor.detach()
    array = np.asarray(tensor)

    tensor.fill_(9.0)

    np.testing.assert_array_equal(array, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(detached.numpy(), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(tensor.numpy(), [9.0, 9.0, 9.0])


# --- nothing is freed underneath it ------------------------------------------


def test_the_array_keeps_the_tensor_alive():
    def make():
        return np.asarray(mt.Tensor(list(range(512)), dtype="float32"))

    array = make()
    gc.collect()
    assert isinstance(array.base, mt.Tensor)
    np.testing.assert_array_equal(array[:3], [0.0, 1.0, 2.0])
    assert array[-1] == 511.0


def test_the_tensor_is_referenced_rather_than_copied_into_the_array():
    tensor = mt.Tensor([1.0, 2.0], dtype="float32")
    before = sys.getrefcount(tensor)
    array = np.asarray(tensor)
    assert sys.getrefcount(tensor) > before
    del array
    gc.collect()
    assert sys.getrefcount(tensor) == before


# --- every dtype -------------------------------------------------------------


@pytest.mark.parametrize("dtype", _DTYPES)
def test_each_dtype_crosses_as_itself(dtype):
    tensor = mt.Tensor([[1, 0], [1, 1]], dtype=dtype)
    array = np.asarray(tensor)
    assert array.dtype == np.dtype(dtype)
    assert array.shape == (2, 2)
    np.testing.assert_array_equal(array, np.array([[1, 0], [1, 1]], dtype=dtype))


@pytest.mark.parametrize("dtype", _DTYPES)
def test_the_type_string_is_the_one_numpy_would_write(dtype):
    tensor = mt.Tensor([1, 0], dtype=dtype)
    assert tensor.__array_interface__["typestr"] == np.dtype(dtype).str


# --- the shapes that have no elements to point at ----------------------------


def test_a_scalar_tensor_has_an_empty_shape():
    tensor = mt.Tensor(3.5)
    array = np.asarray(tensor)
    assert array.shape == ()
    assert float(array) == 3.5


@pytest.mark.parametrize("shape", [[0], [0, 3], [3, 0], [2, 0, 4]])
def test_an_empty_tensor_crosses_with_its_shape_intact(shape):
    array = np.asarray(mt.zeros(shape, dtype="float32"))
    assert array.shape == tuple(shape)
    assert array.size == 0


# --- the interface itself ----------------------------------------------------


def test_the_interface_says_what_the_protocol_requires():
    tensor = mt.Tensor([[1.0, 2.0, 3.0]], dtype="float64")
    interface = tensor.__array_interface__

    assert interface["version"] == 3
    assert interface["shape"] == (1, 3)
    assert interface["typestr"] == np.dtype("float64").str
    assert interface["data"] == (_address(tensor), True)
    # Absent strides is the protocol's way of saying C-contiguous, which every
    # tensor is; spelling them out would only be a second place to be wrong.
    assert "strides" not in interface


def test_a_tensor_round_trips_through_the_interface():
    values = np.ascontiguousarray(
        np.random.default_rng(0).standard_normal((4, 5)), dtype="float32"
    )
    tensor = mt.from_numpy(values)
    back = mt.from_numpy(np.ascontiguousarray(np.asarray(tensor)))
    np.testing.assert_array_equal(back.numpy(), values)


def test_a_tensor_that_wants_a_gradient_still_exports_its_values():
    """The array is values; the graph does not come with them.

    Which is the same thing `.numpy()` has always done, and the reason both
    are worth saying out loud: nothing flows back to the tensor through a
    NumPy array.
    """

    tensor = mt.Tensor([1.0, 2.0], dtype="float32", requires_grad=True)
    array = np.asarray(tensor)
    np.testing.assert_array_equal(array, [1.0, 2.0])
    assert not hasattr(array, "requires_grad")
