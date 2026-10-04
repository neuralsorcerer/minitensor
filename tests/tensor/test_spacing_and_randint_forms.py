# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The spacing constructors agree with one another, and `randint(high, shape)`.

`geomspace` documented zero steps as allowed and then failed inside
`logspace`, which with `linspace` refused them -- though an empty range is an
ordinary answer here, `arange(1, 0)` being one. `geomspace` also came back
float64 whenever no dtype was named, where `linspace` and `logspace` give the
default dtype. And `randint(5, (3,))` read the shape as the upper bound and
refused it as "'tuple' object cannot be interpreted as an integer".
"""

import pytest

import minitensor as mt

SPACINGS = [
    pytest.param(lambda steps, **kw: mt.linspace(0, 1, steps, **kw), id="linspace"),
    pytest.param(lambda steps, **kw: mt.logspace(0, 1, steps, **kw), id="logspace"),
    pytest.param(lambda steps, **kw: mt.geomspace(1, 8, steps, **kw), id="geomspace"),
]


@pytest.mark.parametrize("space", SPACINGS)
def test_zero_steps_is_an_empty_range(space):
    empty = space(0)
    assert empty.shape == (0,) and empty.tolist() == []
    assert space(0, dtype="float64").dtype == "float64"


@pytest.mark.parametrize("space", SPACINGS)
def test_the_default_dtype_is_the_default(space):
    assert space(4).dtype == "float32"
    previous = mt.get_default_dtype()
    mt.set_default_dtype("float64")
    try:
        assert space(4).dtype == "float64"
    finally:
        mt.set_default_dtype(previous)


def test_geomspace_keeps_its_exact_ends_in_the_default_dtype():
    assert mt.geomspace(1, 8, 4).tolist() == [1.0, 2.0, 4.0, 8.0]
    assert mt.geomspace(3, 9, 1).tolist() == [3.0]


@pytest.mark.parametrize("shape", [(3,), [2, 3]])
def test_randint_takes_one_bound_and_a_shape(shape):
    sample = mt.randint(5, shape)
    assert tuple(sample.shape) == tuple(shape)
    assert sample.dtype == "int64"
    assert 0 <= sample.min().item() and sample.max().item() < 5


def test_randint_still_takes_both_bounds_and_loose_dims():
    sample = mt.randint(2, 5, 40, 3)
    assert tuple(sample.shape) == (40, 3)
    assert 2 <= sample.min().item() and sample.max().item() < 5


@pytest.mark.parametrize("bad", [2.5, "5"])
def test_a_bound_that_is_not_an_integer_is_named(bad):
    with pytest.raises(TypeError, match="bounds as integers"):
        mt.randint(5, bad)
