# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`padding="valid"` and `padding="same"` for the convolutions and their layers.

Asking for either used to fail with `TypeError: Can't extract 'str' to 'Vec'`
-- the binding's internal type, not the argument the caller wrote. `"valid"` is
no padding. `"same"` keeps a stride-1 output the size of the input: each axis
needs `dilation * (kernel - 1)` zeros in all, and an odd total -- any even
kernel at an odd dilation -- puts its extra zero after the axis. Each case is
checked against padding the input explicitly by that split and convolving with
no padding at all.
"""

import copy

import numpy as np
import pytest

import minitensor as mt

F = mt.functional
nn = mt.nn
RNG = np.random.default_rng(29)


def _split(kernel, dilation):
    total = dilation * (kernel - 1)
    return total // 2, total - total // 2


@pytest.mark.parametrize("kernel", [1, 2, 3, 4])
@pytest.mark.parametrize("dilation", [1, 2])
def test_conv1d_same_matches_its_explicit_padding(kernel, dilation):
    x = RNG.standard_normal((2, 3, 9))
    w = mt.from_numpy(RNG.standard_normal((4, 3, kernel)))
    b = mt.from_numpy(RNG.standard_normal(4))
    padded = np.pad(x, ((0, 0), (0, 0), _split(kernel, dilation)))
    expected = F.conv1d(mt.from_numpy(padded), w, b, 1, 0, dilation)
    got = F.conv1d(mt.from_numpy(x), w, b, 1, "same", dilation)
    assert tuple(got.shape) == (2, 4, 9)
    np.testing.assert_allclose(got.numpy(), expected.numpy(), rtol=1e-12)


@pytest.mark.parametrize("kernel", [(3, 3), (2, 4), (4, 1), (1, 2)])
@pytest.mark.parametrize("dilation", [(1, 1), (2, 1), (1, 3)])
def test_conv2d_same_matches_its_explicit_padding(kernel, dilation):
    x = RNG.standard_normal((2, 3, 7, 8))
    w = mt.from_numpy(RNG.standard_normal((4, 3) + kernel))
    pads = (
        (0, 0),
        (0, 0),
        _split(kernel[0], dilation[0]),
        _split(kernel[1], dilation[1]),
    )
    expected = F.conv2d(mt.from_numpy(np.pad(x, pads)), w, None, 1, 0, dilation)
    got = F.conv2d(mt.from_numpy(x), w, None, 1, "same", dilation)
    assert tuple(got.shape) == (2, 4, 7, 8)
    np.testing.assert_allclose(got.numpy(), expected.numpy(), rtol=1e-12)


def test_conv3d_same_matches_its_explicit_padding():
    x = RNG.standard_normal((1, 2, 5, 6, 7))
    w = mt.from_numpy(RNG.standard_normal((3, 2, 2, 3, 4)))
    pads = ((0, 0), (0, 0), _split(2, 1), _split(3, 1), _split(4, 1))
    expected = F.conv3d(mt.from_numpy(np.pad(x, pads)), w)
    got = F.conv3d(mt.from_numpy(x), w, padding="same")
    assert tuple(got.shape) == (1, 3, 5, 6, 7)
    np.testing.assert_allclose(got.numpy(), expected.numpy(), rtol=1e-12)


@pytest.mark.parametrize(
    "call,expected_shape",
    [
        (
            lambda x: F.conv1d(
                x[:, :, 0, :], mt.zeros((4, 3, 3), dtype="float64"), None, 1, "valid"
            ),
            (2, 4, 6),
        ),
        (
            lambda x: F.conv2d(
                x, mt.zeros((4, 3, 3, 3), dtype="float64"), None, 1, "valid"
            ),
            (2, 4, 5, 6),
        ),
    ],
    ids=["conv1d", "conv2d"],
)
def test_valid_is_no_padding(call, expected_shape):
    assert tuple(call(mt.zeros((2, 3, 7, 8), dtype="float64")).shape) == expected_shape


@pytest.mark.parametrize("kernel", [3, 4])
def test_the_layers_take_same_and_keep_it(kernel):
    conv2d = nn.Conv2d(3, 4, (kernel, kernel), padding="same").astype("float64")
    x = mt.from_numpy(RNG.standard_normal((2, 3, 7, 8)))
    params = dict(conv2d.named_parameters())
    split = _split(kernel, 1)
    expected = F.conv2d(
        mt.from_numpy(np.pad(x.numpy(), ((0, 0), (0, 0), split, split))),
        params["weight"],
        params["bias"],
    )
    np.testing.assert_allclose(conv2d(x).numpy(), expected.numpy(), rtol=1e-12)
    assert "padding='same'" in repr(conv2d)

    conv1d = nn.Conv1d(3, 4, kernel, padding="same", dilation=2).astype("float64")
    signal = mt.from_numpy(RNG.standard_normal((2, 3, 9)))
    assert tuple(conv1d(signal).shape) == (2, 4, 9)
    assert conv1d.padding == "same"
    assert "padding='same'" in repr(conv1d)

    duplicate = copy.deepcopy(conv1d)
    assert duplicate.padding == "same"
    np.testing.assert_array_equal(duplicate(signal).numpy(), conv1d(signal).numpy())
    restored = nn.Conv1d(3, 4, kernel, padding="same", dilation=2).astype("float64")
    restored.load_state_dict(conv1d.state_dict())
    np.testing.assert_array_equal(restored(signal).numpy(), conv1d(signal).numpy())


def test_a_numeric_padding_still_reads_and_shows_as_before():
    assert nn.Conv1d(3, 4, 3, padding=2).padding == 2
    assert "padding=2" in repr(nn.Conv1d(3, 4, 3, padding=2))
    assert "padding" not in repr(nn.Conv2d(3, 4, 3, padding="valid"))


def test_same_padding_carries_the_gradient():
    x = mt.from_numpy(RNG.standard_normal((1, 2, 5, 6))).requires_grad_(True)
    w = mt.from_numpy(RNG.standard_normal((3, 2, 2, 4))).requires_grad_(True)
    assert mt.gradcheck(
        lambda a, b: (F.conv2d(a, b, None, 1, "same") ** 2).sum(), [x, w]
    )
    layer = nn.Conv2d(2, 3, 2, padding="same").astype("float64")
    assert mt.gradcheck(lambda a: (layer(a) ** 2).sum(), [x])


@pytest.mark.parametrize(
    "call",
    [
        lambda: F.conv2d(
            mt.zeros((1, 3, 7, 8)), mt.zeros((4, 3, 3, 3)), None, 2, "same"
        ),
        lambda: F.conv1d(mt.zeros((1, 3, 9)), mt.zeros((4, 3, 3)), None, 2, "same"),
        lambda: F.conv3d(
            mt.zeros((1, 2, 5, 5, 5)),
            mt.zeros((3, 2, 3, 3, 3)),
            stride=2,
            padding="same",
        ),
        lambda: nn.Conv2d(3, 4, 3, padding="same", stride=2),
        lambda: nn.Conv1d(3, 4, 3, padding="same", stride=2),
    ],
)
def test_same_with_a_larger_stride_is_refused(call):
    with pytest.raises(ValueError, match="needs a stride of 1"):
        call()


@pytest.mark.parametrize(
    "call",
    [
        lambda: F.conv2d(
            mt.zeros((1, 3, 7, 8)), mt.zeros((4, 3, 3, 3)), None, 1, "full"
        ),
        lambda: F.conv1d(mt.zeros((1, 3, 9)), mt.zeros((4, 3, 3)), None, 1, "half"),
        lambda: F.conv3d(
            mt.zeros((1, 2, 5, 5, 5)), mt.zeros((3, 2, 3, 3, 3)), padding="full"
        ),
        lambda: nn.Conv2d(3, 4, 3, padding="full"),
        lambda: nn.Conv1d(3, 4, 3, padding="full"),
    ],
)
def test_an_unknown_padding_name_is_refused_by_name(call):
    with pytest.raises(ValueError, match="padding must be 'valid', 'same'"):
        call()
