# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

import minitensor as mt


def test_one_hot_infers_classes_for_integer_tensor():
    labels = mt.Tensor([[0, 2], [1, 2]], dtype="int64")

    encoded = mt.one_hot(labels)

    expected = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    assert encoded.shape_vec() == [2, 2, 3]
    assert encoded.dtype == "float32"
    np.testing.assert_array_equal(encoded.numpy(), expected)


def test_one_hot_accepts_sequence_and_output_dtype():
    encoded = mt.functional.one_hot([2, 0, 1], num_classes=4, dtype="int64")

    expected = np.array(
        [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]],
        dtype=np.int64,
    )
    assert encoded.shape_vec() == [3, 4]
    assert encoded.dtype == "int64"
    np.testing.assert_array_equal(encoded.numpy(), expected)


def test_one_hot_supports_empty_input_with_explicit_classes():
    labels = mt.Tensor([], dtype="int64")

    encoded = mt.one_hot(labels, num_classes=3, dtype="bool")

    assert encoded.shape_vec() == [0, 3]
    assert encoded.dtype == "bool"
    np.testing.assert_array_equal(encoded.numpy(), np.empty((0, 3), dtype=bool))


def test_one_hot_preserves_nested_python_sequence_shape():
    encoded = mt.one_hot(((0, 1), (2, 1)), dtype="int32")

    expected = np.array(
        [
            [[1, 0, 0], [0, 1, 0]],
            [[0, 0, 1], [0, 1, 0]],
        ],
        dtype=np.int32,
    )
    assert encoded.shape_vec() == [2, 2, 3]
    assert encoded.dtype == "int32"
    np.testing.assert_array_equal(encoded.numpy(), expected)


def test_one_hot_accepts_numpy_integer_arrays_and_bool_sequences():
    encoded = mt.one_hot(np.array([1, 0], dtype=np.int32), num_classes=2)
    np.testing.assert_array_equal(
        encoded.numpy(), np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    )

    bool_encoded = mt.one_hot([True, False], num_classes=2, dtype="bool")
    np.testing.assert_array_equal(
        bool_encoded.numpy(), np.array([[False, True], [True, False]])
    )


def test_one_hot_rejects_invalid_labels_and_class_counts():
    with pytest.raises(ValueError, match="non-negative"):
        mt.one_hot(mt.Tensor([-1], dtype="int64"))

    with pytest.raises(ValueError, match="valid range"):
        mt.one_hot(mt.Tensor([3], dtype="int64"), num_classes=3)

    with pytest.raises(ValueError, match="must be provided"):
        mt.one_hot(mt.Tensor([], dtype="int64"))

    with pytest.raises(TypeError, match="integer or bool dtype"):
        mt.one_hot(mt.Tensor([0.0], dtype="float32"))

    with pytest.raises(TypeError, match="integer or bool dtype"):
        mt.one_hot(1.5)
