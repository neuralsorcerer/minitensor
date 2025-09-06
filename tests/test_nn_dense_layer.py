# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from minitensor.tensor import Tensor
from minitensor.nn import DenseLayer

def test_dense_layer_forward_and_stats():
    layer = DenseLayer(3, 2)
    x = Tensor.rand([4, 3])
    y = layer.forward(x._tensor)
    assert y.shape == [4, 2]
    stats = layer.parameter_stats()
    assert stats["total_parameters"] == 3 * 2 + 2
    mem = layer.memory_usage()
    assert mem["total_bytes"] == (3 * 2 + 2) * 4
    assert mem["bytes_by_dtype"]["Float32"] == (3 * 2 + 2) * 4
    summary = layer.summary()
    assert "Total Parameters" in summary

def test_dense_layer_forward_shape_error():
    layer = DenseLayer(3, 2)
    bad_input = Tensor.rand([4, 4])
    with pytest.raises(ValueError):
        layer.forward(bad_input._tensor)
