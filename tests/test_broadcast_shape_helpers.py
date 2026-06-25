# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

import minitensor as mt


class IndexLike:
    def __init__(self, value: int) -> None:
        self.value = value

    def __index__(self) -> int:
        return self.value


def test_broadcast_shapes_matches_numpy_style_rules() -> None:
    assert mt.broadcast_shapes() == ()
    assert mt.broadcast_shapes((), (2, 3)) == (2, 3)
    assert mt.broadcast_shapes((5, 1, 4), (1, 3, 1), (3, 4)) == (5, 3, 4)
    assert mt.broadcast_shapes(3, (2, 1)) == (2, 3)
    assert mt.broadcast_shapes(np.int64(3), (2, 1)) == (2, 3)
    assert mt.broadcast_shapes((IndexLike(1), IndexLike(4)), (3, 1)) == (3, 4)


def test_broadcast_shapes_accepts_tensor_shape_objects() -> None:
    tensor = mt.zeros(2, 1, 4)
    assert mt.broadcast_shapes(tensor.shape, (3, 4)) == (2, 3, 4)


def test_broadcast_shapes_rejects_incompatible_shapes() -> None:
    with pytest.raises(ValueError, match="cannot be broadcast"):
        mt.broadcast_shapes((2, 3), (4, 3))


def test_broadcast_shapes_validates_dimensions() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        mt.broadcast_shapes((2, -1))
    with pytest.raises(ValueError, match="non-negative"):
        mt.broadcast_shapes(IndexLike(-1))
    with pytest.raises(TypeError, match="integers"):
        mt.broadcast_shapes((2, 1.5))
    with pytest.raises(TypeError, match="not bool"):
        mt.broadcast_shapes(True)


def test_can_broadcast_returns_boolean_without_raising() -> None:
    assert mt.can_broadcast((1, 3), (2, 3)) is True
    assert mt.can_broadcast((2, 3), (4, 3)) is False
    assert mt.can_broadcast((2, "bad")) is False


def test_broadcast_helpers_are_public_api_entries() -> None:
    top_level = mt.list_public_api()["top_level"]
    assert "broadcast_shapes" in top_level
    assert "can_broadcast" in top_level
