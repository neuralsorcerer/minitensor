# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Weight initialization.

Six of these already existed as tensor constructors -- `mt.xavier_uniform`,
`mt.he_normal` and so on. What `nn.init` adds is the namespace a PyTorch user
looks in, `calculate_fan_in_and_fan_out`, the Kaiming/Glorot spellings, and a
`requires_grad` default suited to building parameters.

That last one is a difference between two spellings of the same scheme, so it
is pinned here rather than left to be discovered:
`test_the_two_spellings_differ_only_in_their_requires_grad_default`.

Each scheme is checked against its closed form rather than a recorded trace: a
uniform one by the bound it must not exceed and the standard deviation that
bound implies, a normal one by its standard deviation. That is what makes a
transposed `fan_in` visible -- the usual failure, because a weight here is
stored `[out_features, in_features]` and the fan the formulas want is the
trailing dimension, so getting it backwards still produces plausible-looking
numbers.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import minitensor as mt
from minitensor.nn import init

# A deliberately non-square shape: fan_in and fan_out differ, so a scheme that
# reads the wrong one is off by sqrt(2) rather than being indistinguishable.
SHAPE = [512, 256]
FAN_IN, FAN_OUT = 256, 512

# scheme -> (kind, scale), where scale is the bound for a uniform and the
# standard deviation for a normal.
SCHEMES = {
    "xavier_uniform": ("uniform", math.sqrt(6 / (FAN_IN + FAN_OUT))),
    "xavier_normal": ("normal", math.sqrt(2 / (FAN_IN + FAN_OUT))),
    "he_uniform": ("uniform", math.sqrt(6 / FAN_IN)),
    "he_normal": ("normal", math.sqrt(2 / FAN_IN)),
    "lecun_uniform": ("uniform", math.sqrt(3 / FAN_IN)),
    "lecun_normal": ("normal", math.sqrt(1 / FAN_IN)),
}


@pytest.fixture(autouse=True)
def _seed():
    mt.manual_seed(0)
    np.random.seed(0)


def test_fan_follows_the_weight_layout():
    """A weight is `[out_features, in_features]`, so fan_in is the trailing
    dimension. Swapping these is the mistake the schemes below would inherit."""
    assert init.calculate_fan_in_and_fan_out([512, 256]) == (256, 512)


def test_fan_for_a_convolution_weight_counts_the_receptive_field():
    """`[out_channels, in_channels, kh, kw]`."""
    assert init.calculate_fan_in_and_fan_out([32, 3, 5, 5]) == (3 * 25, 32 * 25)
    assert init.calculate_fan_in_and_fan_out([16, 8, 3]) == (8 * 3, 16 * 3)


def test_fan_for_low_rank_shapes():
    assert init.calculate_fan_in_and_fan_out([]) == (1, 1)
    assert init.calculate_fan_in_and_fan_out([7]) == (7, 7)


@pytest.mark.parametrize("name", sorted(SCHEMES))
def test_scale_matches_the_closed_form(name):
    kind, scale = SCHEMES[name]
    values = getattr(init, name)(SHAPE).numpy()

    # A uniform over +/- b has standard deviation b / sqrt(3).
    expected_std = scale / math.sqrt(3) if kind == "uniform" else scale
    assert values.std() == pytest.approx(expected_std, rel=0.03)
    assert values.mean() == pytest.approx(0.0, abs=0.02 * expected_std + 1e-4)


@pytest.mark.parametrize(
    "name", sorted(n for n, (kind, _) in SCHEMES.items() if kind == "uniform")
)
def test_uniform_schemes_stay_inside_their_bound(name):
    bound = SCHEMES[name][1]
    values = getattr(init, name)(SHAPE).numpy()

    assert np.abs(values).max() <= bound
    # ... and reach it: a scheme drawing from a much narrower range would pass
    # the bound check alone.
    assert np.abs(values).max() > 0.99 * bound


@pytest.mark.parametrize("name", sorted(SCHEMES))
def test_reading_the_wrong_fan_would_be_visible(name):
    """Guards the tests above from being satisfiable by a transposed fan.

    Xavier reads `fan_in + fan_out`, which is symmetric, so only the fan-in
    schemes can detect a swap -- this asserts which is which rather than
    assuming.
    """
    kind, scale = SCHEMES[name]
    swapped = {
        "xavier_uniform": math.sqrt(6 / (FAN_OUT + FAN_IN)),
        "xavier_normal": math.sqrt(2 / (FAN_OUT + FAN_IN)),
        "he_uniform": math.sqrt(6 / FAN_OUT),
        "he_normal": math.sqrt(2 / FAN_OUT),
        "lecun_uniform": math.sqrt(3 / FAN_OUT),
        "lecun_normal": math.sqrt(1 / FAN_OUT),
    }[name]

    if name.startswith("xavier"):
        assert swapped == scale, "Xavier is symmetric in the two fans"
    else:
        assert abs(swapped / scale - 1) > 0.25, (
            f"{name} would look the same with the fans swapped"
        )


@pytest.mark.parametrize("name", sorted(SCHEMES))
def test_schemes_produce_trainable_parameters_by_default(name):
    tensor = getattr(init, name)(SHAPE)
    assert tensor.requires_grad
    assert tuple(tensor.shape) == tuple(SHAPE)
    assert tensor.dtype == "float32"

    plain = getattr(init, name)(SHAPE, requires_grad=False)
    assert not plain.requires_grad


@pytest.mark.parametrize("name", sorted(SCHEMES))
def test_schemes_honour_float64(name):
    tensor = getattr(init, name)(SHAPE, dtype="float64")
    assert tensor.dtype == "float64"
    assert tensor.numpy().dtype == np.float64


@pytest.mark.parametrize("name", sorted(SCHEMES) + ["uniform", "normal"])
def test_random_schemes_refuse_integer_dtypes(name):
    """`int32` has no meaningful draw, and the engine's rejection happens deep
    enough that its message does not mention initialization."""
    with pytest.raises(ValueError) as excinfo:
        getattr(init, name)([4, 4], dtype="int32")
    assert "float dtype" in str(excinfo.value)


@pytest.mark.parametrize(
    "alias,target",
    [
        ("kaiming_uniform", "he_uniform"),
        ("kaiming_normal", "he_normal"),
        ("glorot_uniform", "xavier_uniform"),
        ("glorot_normal", "xavier_normal"),
    ],
)
def test_alternate_spellings_are_the_same_function(alias, target):
    """He and Kaiming are the same person; Xavier and Glorot likewise. A user
    who reaches for the other spelling should not have to discover which one
    this library picked."""
    assert getattr(init, alias) is getattr(init, target)


def test_constant_zeros_and_ones():
    np.testing.assert_array_equal(init.zeros([2, 3]).numpy(), np.zeros((2, 3)))
    np.testing.assert_array_equal(init.ones([2, 3]).numpy(), np.ones((2, 3)))
    np.testing.assert_array_equal(
        init.constant([2, 3], 4.5).numpy(), np.full((2, 3), 4.5, dtype=np.float32)
    )


def test_constant_accepts_integer_dtypes():
    """Unlike the sampling schemes -- a constant is well defined for any dtype,
    and a zeroed int buffer is a reasonable thing to ask for."""
    values = init.constant([3], 7, dtype="int64", requires_grad=False).numpy()
    np.testing.assert_array_equal(values, [7, 7, 7])


def test_uniform_and_normal_take_their_parameters():
    values = init.uniform([4096], a=-3.0, b=5.0).numpy()
    assert values.min() >= -3.0 and values.max() <= 5.0
    assert values.mean() == pytest.approx(1.0, abs=0.15)

    values = init.normal([16384], mean=2.0, std=0.5).numpy()
    assert values.mean() == pytest.approx(2.0, abs=0.02)
    assert values.std() == pytest.approx(0.5, rel=0.05)


@pytest.mark.parametrize(
    "a,b", [(1.0, 1.0), (2.0, 1.0), (float("nan"), 1.0), (0.0, float("inf"))]
)
def test_uniform_rejects_a_range_it_cannot_draw_from(a, b):
    with pytest.raises(ValueError) as excinfo:
        init.uniform([4], a=a, b=b)
    assert "a < b" in str(excinfo.value)


def test_truncated_normal_respects_its_bounds():
    values = init.truncated_normal([8192], mean=0.0, std=1.0).numpy()
    assert values.min() >= -2.0 and values.max() <= 2.0
    # Truncation at +/-2 sigma leaves a standard deviation near 0.88.
    assert values.std() == pytest.approx(0.88, rel=0.08)


def test_truncated_normal_takes_explicit_bounds():
    values = init.truncated_normal([4096], mean=1.0, std=2.0, lower=0.0, upper=1.5)
    values = values.numpy()
    assert values.min() >= 0.0 and values.max() <= 1.5


def test_negative_dimensions_are_refused():
    with pytest.raises(ValueError) as excinfo:
        init.zeros([2, -3])
    assert "negative" in str(excinfo.value)


def test_a_custom_layer_can_be_initialized_with_these():
    """The use case: `plugins.CustomLayer` builds parameters as plain tensors,
    so before this a custom Dense-like layer had to hand-roll its scheme."""
    from minitensor.plugins import CustomLayer

    layer = CustomLayer("linear")
    weight = init.he_normal([4, 3])
    layer.add_parameter("weight", weight)
    layer.set_forward(
        lambda inputs: [
            mt.matmul(inputs[0], layer.get_parameter("weight").transpose(0, 1))
        ]
    )

    out = layer.forward([mt.randn(2, 3)])[0]
    assert tuple(out.shape) == (2, 4)

    mt.sum(out).backward()
    assert weight.grad is not None


# --- relationship to the top-level constructors ---------------------------
#
# `mt.xavier_uniform(shape)` and `nn.init.xavier_uniform(shape)` are the same
# scheme reached two ways, and they do NOT agree on `requires_grad`. The
# top-level one sits beside `mt.zeros` and `mt.randn` as a way to make a
# tensor and defaults to False like they do; this one exists to make a
# *parameter*, and a parameter created without `requires_grad` does not train,
# silently. Both take the argument explicitly.

SHARED_WITH_TOP_LEVEL = [
    "xavier_uniform",
    "xavier_normal",
    "he_uniform",
    "he_normal",
    "lecun_uniform",
    "lecun_normal",
    "uniform",
    "truncated_normal",
    "zeros",
    "ones",
]


@pytest.mark.parametrize("name", SHARED_WITH_TOP_LEVEL)
def test_the_two_spellings_differ_only_in_their_requires_grad_default(name):
    assert hasattr(mt, name), f"mt.{name} is the older spelling and must keep working"

    from_init = getattr(init, name)([4, 3])
    from_top = getattr(mt, name)([4, 3])

    assert from_init.requires_grad is True, "nn.init builds parameters"
    assert from_top.requires_grad is False, "the tensor constructors build tensors"

    # Everything else about them agrees.
    assert tuple(from_init.shape) == tuple(from_top.shape)
    assert from_init.dtype == from_top.dtype

    # ... and each can be asked for the other behaviour.
    assert getattr(init, name)([4, 3], requires_grad=False).requires_grad is False
    assert getattr(mt, name)([4, 3], requires_grad=True).requires_grad is True


@pytest.mark.parametrize("name", ["xavier_uniform", "he_normal", "lecun_uniform"])
def test_both_spellings_draw_from_the_same_distribution(name):
    """Not just the same name -- the same scheme.

    If these ever diverge, one of them is wrong and nothing else would say so.
    """
    mt.manual_seed(0)
    left = getattr(init, name)(SHAPE).numpy()
    mt.manual_seed(0)
    right = getattr(mt, name)(SHAPE).numpy()
    np.testing.assert_array_equal(left, right)


def test_the_like_variants_stay_on_the_top_level_constructors():
    """`nn.init` takes shapes only. The `_like` forms, which read shape, dtype
    and device off a reference tensor, have no equivalent here and are not
    duplicated."""
    reference = mt.randn(6, 4)
    assert tuple(mt.he_normal_like(reference).shape) == (6, 4)
    assert not hasattr(init, "he_normal_like")
