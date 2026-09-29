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


BAD_EPS = [-1.0, float("nan"), float("inf")]


@pytest.mark.parametrize("eps", BAD_EPS)
@pytest.mark.parametrize(
    "build",
    [
        lambda eps: nn.LayerNorm(4, eps=eps),
        lambda eps: nn.RMSNorm(4, eps=eps),
        lambda eps: nn.BatchNorm1d(4, eps=eps),
        lambda eps: nn.BatchNorm2d(4, eps=eps),
        lambda eps: nn.batch_norm(mt.randn(4, 3), None, None, eps=eps),
        lambda eps: nn.group_norm(mt.ones(2, 4, 3), 2, eps=eps),
        lambda eps: nn.instance_norm(mt.randn(2, 3, 4), eps=eps),
    ],
    ids=[
        "LayerNorm",
        "RMSNorm",
        "BatchNorm1d",
        "BatchNorm2d",
        "batch_norm",
        "group_norm",
        "instance_norm",
    ],
)
def test_a_normalization_refuses_an_eps_that_is_not_a_stabilizer(build, eps):
    """`eps` is added to a variance before its root, so a negative one took
    the root of a negative number: LayerNorm(eps=-1) answered NaN for every
    input with a small spread, without complaint."""
    with pytest.raises(ValueError, match="eps to be finite and non-negative"):
        build(eps)


@pytest.mark.parametrize("momentum", [-0.1, 2.0])
@pytest.mark.parametrize(
    "build",
    [
        lambda m: nn.BatchNorm1d(4, momentum=m),
        lambda m: nn.batch_norm(mt.randn(4, 3), mt.zeros(3), mt.ones(3), momentum=m),
        lambda m: nn.instance_norm(
            mt.randn(2, 3, 4),
            running_mean=mt.zeros(3),
            running_var=mt.ones(3),
            momentum=m,
        ),
    ],
    ids=["BatchNorm1d", "batch_norm", "instance_norm"],
)
def test_running_statistics_refuse_a_momentum_outside_the_unit_interval(
    build, momentum
):
    """Outside [0, 1] the running update is no longer an average and the
    estimate diverges."""
    with pytest.raises(ValueError, match=r"momentum to lie in \[0, 1\]"):
        build(momentum)


def test_the_boundary_values_are_allowed():
    assert tuple(nn.LayerNorm(4, eps=0.0)(mt.randn(2, 4)).shape) == (2, 4)
    nn.BatchNorm1d(4, momentum=0.0)
    nn.BatchNorm1d(4, momentum=1.0)
