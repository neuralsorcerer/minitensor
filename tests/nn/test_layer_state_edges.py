# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""What a layer holds agrees with what it computes, at the edges.

- `Embedding(padding_idx=p)` gives token `p` a zero embedding, but drew its
  weight row like every other: the layer returned zero while the table it
  saved and handed to the functional `embedding` held random values.
- Batch statistics over one value per channel have no spread. The output was
  exactly zero whatever the input, and the running variance was pulled toward
  zero, inflating every later output in evaluation mode. It is refused now.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

nn = mt.nn


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_the_padding_row_is_zero_in_the_table_the_layer_holds(dtype):
    mt.manual_seed(0)
    layer = nn.Embedding(6, 4, padding_idx=2, dtype=dtype)
    table = layer.state_dict()["weight"].numpy()

    np.testing.assert_array_equal(table[2], np.zeros(4))
    assert np.count_nonzero(np.delete(table, 2, axis=0)) > 0

    ids = mt.Tensor(np.array([2, 0, 2]), dtype="int64")
    np.testing.assert_array_equal(
        layer(ids).numpy(),
        nn.embedding(ids, layer.parameters()[0], padding_idx=2).numpy(),
    )


def test_the_padding_row_still_takes_no_gradient():
    layer = nn.Embedding(4, 3, padding_idx=0)
    layer(mt.Tensor(np.array([0, 1, 0]), dtype="int64")).sum().backward()
    grad = layer.parameters()[0].grad.numpy()
    np.testing.assert_array_equal(grad[0], np.zeros(3))
    np.testing.assert_array_equal(grad[1], np.ones(3))


@pytest.mark.parametrize(
    "build,shape",
    [
        (lambda: nn.BatchNorm1d(3), (1, 3)),
        (lambda: nn.BatchNorm2d(2), (1, 2, 1, 1)),
    ],
    ids=["BatchNorm1d", "BatchNorm2d"],
)
def test_batch_statistics_over_one_value_per_channel_are_refused(build, shape):
    layer = build()
    before = {k: v.numpy().copy() for k, v in layer.state_dict().items()}

    with pytest.raises(ValueError, match="more than one value per channel"):
        layer(mt.ones(*shape))
    for name, values in before.items():
        np.testing.assert_array_equal(layer.state_dict()[name].numpy(), values)

    # Evaluation mode normalizes with the running statistics, so one sample
    # is fine there, and so is one sample with spatial positions to pool.
    layer.eval()
    assert tuple(layer(mt.ones(*shape)).shape) == shape
    layer.train()
    if len(shape) == 4:
        assert tuple(layer(mt.ones(1, 2, 2, 2)).shape) == (1, 2, 2, 2)


def test_functional_batch_norm_refuses_it_too():
    with pytest.raises(ValueError, match="more than one value per channel"):
        nn.batch_norm(mt.ones(1, 2), None, None, training=True)
