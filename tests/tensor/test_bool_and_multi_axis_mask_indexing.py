# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A bool is not a position, and a mask may span several axes of a subscript.

A Python bool converts to an integer, so ``t[True]`` was ``t[1]``,
``t[:, False]`` was ``t[:, 0]`` and ``t[[1, True]]`` picked row 1 twice -- a
condition that happened to be a Python bool picked a row without a word, and
assignment wrote there just as quietly. Each is now refused by name.

A mask of two or more dimensions selected only as the whole subscript:
``t[m]`` worked, while ``t[m, ...]``, ``t[(m,)]`` and ``t[:, m]`` raised a bare
``Invalid index type``. A mask inside a subscript now counts its positions
across every axis it spans, for reading and for writing.
"""

import numpy as np
import pytest

import minitensor as mt

RNG = np.random.default_rng(0)
X = np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5)
M01 = RNG.random((2, 3)) > 0.5
M12 = RNG.random((3, 4)) > 0.4
M23 = RNG.random((4, 5)) > 0.5
M123 = RNG.random((3, 4, 5)) > 0.5

KEYS = [
    (M01,),
    (M01, Ellipsis),
    (M01, slice(None, None, 2)),
    (None, M01),
    (slice(None), M12),
    (slice(None), M12, 2),
    (slice(None), None, M12),
    (slice(0, 1), M12, slice(1, 4)),
    (1, M12),
    (Ellipsis, M23),
    (0, M123),
    (slice(None), M123),
    (slice(None), np.zeros((3, 4), dtype=bool)),
]


@pytest.mark.parametrize("key", KEYS, ids=range(len(KEYS)))
def test_a_mask_over_several_axes_reads_as_it_selects(key):
    got = mt.from_numpy(X.copy())[key]
    expected = X[key]
    assert tuple(got.shape) == expected.shape
    np.testing.assert_array_equal(got.numpy(), expected)


def test_the_mask_may_be_a_tensor():
    got = mt.from_numpy(X.copy())[:, mt.from_numpy(M12)]
    np.testing.assert_array_equal(got.numpy(), X[:, M12])


@pytest.mark.parametrize("key", KEYS, ids=range(len(KEYS)))
def test_a_mask_over_several_axes_writes_where_it_reads(key):
    for value in (-1.0, None):
        expected = X.copy()
        target = mt.from_numpy(X.copy())
        if value is None:
            shape = X[key].shape
            value = -np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
            target[key] = mt.from_numpy(value)
        else:
            target[key] = value
        expected[key] = value
        np.testing.assert_array_equal(target.numpy(), expected)


def test_a_value_broadcasts_across_the_axes_after_the_mask():
    row = np.arange(5, dtype=np.float32)
    expected = X.copy()
    expected[:, M12] = row
    target = mt.from_numpy(X.copy())
    target[:, M12] = mt.from_numpy(row)
    np.testing.assert_array_equal(target.numpy(), expected)


@pytest.mark.parametrize(
    "key,where",
    [
        ((slice(None), np.ones((3, 5), dtype=bool)), "dimensions 1 to 2"),
        ((slice(None), slice(None), M12), "dimensions 2 to 3"),
    ],
)
def test_a_mask_of_the_wrong_shape_names_the_axes_it_landed_on(key, where):
    with pytest.raises(IndexError, match=where):
        mt.from_numpy(X.copy())[key]


@pytest.mark.parametrize(
    "key",
    [True, False, (0, True), (slice(None), False), (True, Ellipsis)],
    ids=repr,
)
def test_a_bool_is_not_a_position(key):
    t = mt.arange(12).reshape(3, 4)
    with pytest.raises(TypeError, match="a bool cannot index"):
        t[key]
    with pytest.raises(TypeError, match="a bool cannot index"):
        t[key] = -1.0
    np.testing.assert_array_equal(t.numpy(), np.arange(12).reshape(3, 4))


@pytest.mark.parametrize(
    "key", [[1, True], [[0, 1], [True, 2]], (slice(None), [1, True])], ids=repr
)
def test_a_bool_among_positions_is_refused(key):
    t = mt.arange(12).reshape(3, 4)
    with pytest.raises(TypeError, match="mixes ints and bools"):
        t[key]


def test_a_bool_list_is_still_a_mask():
    t = mt.arange(12).reshape(3, 4)
    np.testing.assert_array_equal(
        t[[True, False, True]].numpy(), np.arange(12).reshape(3, 4)[[0, 2]]
    )


@pytest.mark.parametrize(
    "key,name", [(1.5, "float"), ("a", "str"), (np.True_, "numpy.bool")]
)
def test_an_unusable_index_names_its_type(key, name):
    with pytest.raises(TypeError, match=f"^{name} cannot index a tensor"):
        mt.arange(4)[key]
