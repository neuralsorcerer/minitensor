# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""An advanced index anywhere in the subscript, not only at the front.

``t[idx]`` selects rows and ``t[mask]`` selects blocks, but the commoner
spelling in practice is ``t[:, idx]`` -- a column selection, an embedding
lookup along the last axis, a head slice. That used to be a bare
``TypeError: Invalid index type`` even though ``index_select`` does exactly
that job.

With exactly one advanced index the answer does not depend on the order the
basic and advanced parts are applied in, and the selected axis stays where it
was written -- NumPy's rule about advanced indices moving to the front needs
two of them, separated. These tests pin that against NumPy for every way the
one index can be spelled and every basic entry it can be mixed with. Two
advanced indices mean something else (they pair up elementwise) and are
refused by name. Assignment takes the same forms: reading ``t[:, idx]``
and not being able to write it would be half a feature.
"""

import numpy as np
import pytest

import minitensor as mt


@pytest.fixture()
def x():
    return np.arange(24, dtype=np.float32).reshape(2, 3, 4)


def same(actual, expected):
    assert tuple(actual.shape) == expected.shape
    np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize(
    "index",
    [
        [0, 2],
        np.array([0, 2], dtype=np.int64),
        np.array([0, 2], dtype=np.int32),
    ],
    ids=["list", "int64", "int32"],
)
def test_an_index_array_selects_a_middle_axis(x, index):
    same(mt.from_numpy(x)[:, index], x[:, [0, 2]])


def test_a_minitensor_tensor_is_an_index_too(x):
    idx = mt.from_numpy(np.array([2, 0, 2], dtype=np.int64))
    same(mt.from_numpy(x)[:, idx], x[:, [2, 0, 2]])


def test_the_index_mixes_with_every_basic_entry(x):
    t = mt.from_numpy(x)
    same(t[1:2, [0, 2]], x[1:2, [0, 2]])
    same(t[..., [1, 3]], x[..., [1, 3]])
    same(t[1, [0, 2]], x[1, [0, 2]])
    same(t[[1, 0], 2], x[[1, 0], 2])
    same(t[:, 1, [0, 3]], x[:, 1, [0, 3]])
    same(t[None, :, [0, 2]], x[None, :, [0, 2]])
    same(t[:, None, [0, 2]], x[:, None, [0, 2]])
    same(t[::2, [0, 2]], x[::2, [0, 2]])
    same(t[..., 1, [0, 2]], x[..., 1, [0, 2]])
    same(t[[0, 1], ..., 1], x[[0, 1], ..., 1])
    same(t[:, 1:3, [0, 2]], x[:, 1:3, [0, 2]])


def test_negative_positions_wrap_against_the_axis_they_land_on(x):
    """The axis is the one the array sits above, not dimension zero.

    Wrapping ``-1`` against the wrong extent is silent: it reads a real
    element, just not the one that was asked for.
    """
    same(mt.from_numpy(x)[:, [-1, -3]], x[:, [-1, -3]])
    same(mt.from_numpy(x)[:, :, [-1, -4]], x[:, :, [-1, -4]])

    with pytest.raises(IndexError):
        mt.from_numpy(x)[:, [3]]  # in range for axis 0, not for axis 1
    with pytest.raises(IndexError):
        mt.from_numpy(x)[:, [-4]]


def test_an_index_array_carries_its_own_shape(x):
    """A 2-D index array puts both of its axes where the one axis was."""
    same(mt.from_numpy(x)[:, [[0, 2], [1, 1]]], x[:, [[0, 2], [1, 1]]])
    same(
        mt.from_numpy(x)[:, np.array([[0, 2], [1, 1]])],
        x[:, np.array([[0, 2], [1, 1]])],
    )
    # And as the whole key, where the leading-row path does not reach.
    same(mt.from_numpy(x)[[[0, 1]]], x[[[0, 1]]])


def test_an_empty_index_selects_an_empty_axis(x):
    same(mt.from_numpy(x)[:, []], x[:, []])
    same(mt.from_numpy(x)[:, [[]]], x[:, [[]]])


@pytest.mark.parametrize(
    "mask",
    [[True, False, True], np.array([True, False, True])],
    ids=["list", "ndarray"],
)
def test_a_boolean_mask_selects_along_the_axis_it_is_written_on(x, mask):
    same(mt.from_numpy(x)[:, mask], x[:, np.array([True, False, True])])


def test_the_mask_forms_agree_with_numpy(x):
    t = mt.from_numpy(x)
    m3 = np.array([True, False, True])
    m4 = np.array([True, False, True, False])
    same(t[:, mt.from_numpy(m3)], x[:, m3])
    same(t[..., m4], x[..., m4])
    same(t[1, m3], x[1, m3])
    same(t[:, m3, 2], x[:, m3, 2])
    same(t[:, None, m3], x[:, None, m3])
    same(t[:, np.zeros(3, dtype=bool)], x[:, np.zeros(3, dtype=bool)])


def test_a_mask_of_the_wrong_length_says_so(x):
    with pytest.raises(IndexError, match="boolean index has 4"):
        mt.from_numpy(x)[:, np.array([True, False, True, False])]


def test_two_advanced_indices_are_refused_by_name(x):
    """Two of them pair up elementwise -- a different operation, so say which.

    Answering with the outer product instead would be wrong quietly, which is
    worse than not answering.
    """
    with pytest.raises(IndexError, match="gather"):
        mt.from_numpy(x)[[0, 1], [1, 2]]
    with pytest.raises(IndexError, match="gather"):
        mt.from_numpy(x)[:, np.array([True, False, True]), [0, 1]]


def test_too_many_indices_is_an_index_error(x):
    with pytest.raises(IndexError):
        mt.from_numpy(x)[:, :, :, [0]]


def test_the_gradient_reaches_exactly_the_selected_columns(x):
    t = mt.from_numpy(x.copy())
    t.requires_grad_(True)
    (t[:, [0, 2]] * 2.0).sum().backward()

    expected = np.zeros_like(x)
    expected[:, [0, 2]] = 2.0
    np.testing.assert_allclose(mt.get_gradient(t).numpy(), expected, rtol=1e-6)
    mt.clear_autograd_graph()


def test_a_repeated_position_accumulates_its_gradient(x):
    """``[1, 1]`` reads one column twice, so its gradient arrives twice."""
    t = mt.from_numpy(x.copy())
    t.requires_grad_(True)
    t[:, [1, 1]].sum().backward()

    expected = np.zeros_like(x)
    expected[:, 1] = 2.0
    np.testing.assert_allclose(mt.get_gradient(t).numpy(), expected, rtol=1e-6)
    mt.clear_autograd_graph()


def test_the_selection_matches_index_select(x):
    """Same answer as the named op, which is the operation being reached for."""
    t = mt.from_numpy(x)
    idx = mt.from_numpy(np.array([2, 0], dtype=np.int64))
    np.testing.assert_array_equal(t[:, idx].numpy(), mt.index_select(t, 1, idx).numpy())


def test_the_same_forms_can_be_written_to(x):
    """Reading `t[:, idx]` and not being able to write it is a half-feature."""

    def both(key, value, np_value=None):
        actual = mt.from_numpy(x.copy())
        actual[key] = value
        expected = x.copy()
        expected[key] = np_value if np_value is not None else value
        np.testing.assert_array_equal(actual.numpy(), expected)

    both((slice(None), [0, 2]), 0.0)
    both([0, 1], -1.0)
    both((Ellipsis, [1, 3]), 7.0)
    both((1, [0, 2]), 5.0)
    both((slice(None), [-1, -3]), 2.0)
    both((slice(None), slice(1, 3), [0, 2]), 4.0)
    both((slice(None, None, 2), [0, 2]), 6.0)
    both((slice(None), np.array([True, False, True])), 3.0)
    both((slice(None), [[0, 2], [1, 1]]), 8.0)
    both((slice(None), []), 9.0)


def test_a_written_value_spreads_over_the_selection(x):
    """The value lines up with the whole selection, then each position takes
    its share -- so a value that stops short of the indexed axis is repeated
    and one that spans it is split."""

    def both(key, value):
        actual = mt.from_numpy(x.copy())
        actual[key] = mt.from_numpy(value)
        expected = x.copy()
        expected[key] = value
        np.testing.assert_array_equal(actual.numpy(), expected)

    both((slice(None), [0, 2]), np.arange(4, dtype=np.float32))
    both((slice(None), [0, 2]), np.arange(16, dtype=np.float32).reshape(2, 2, 4))
    both((1, [0, 2]), np.arange(8, dtype=np.float32).reshape(2, 4))
    both(
        (slice(None), np.array([True, False, True])),
        np.arange(16, dtype=np.float32).reshape(2, 2, 4),
    )
    both(
        (slice(None), [[0, 2], [1, 1]]),
        np.arange(32, dtype=np.float32).reshape(2, 2, 2, 4),
    )
    both(
        (slice(None), None, [0, 2]), np.arange(16, dtype=np.float32).reshape(2, 1, 2, 4)
    )


def test_a_position_written_twice_keeps_the_last_write(x):
    actual = mt.from_numpy(x.copy())
    value = np.arange(16, dtype=np.float32).reshape(2, 2, 4)
    actual[:, [1, 1]] = mt.from_numpy(value)

    expected = x.copy()
    expected[:, [1, 1]] = value
    np.testing.assert_array_equal(actual.numpy(), expected)


def test_a_value_read_from_the_target_is_read_before_it_is_written(x):
    """`t[:, [0, 1]] = t[:, [1, 0]]` is a swap, not a double copy.

    The positions are written one at a time, so a value aliasing the target
    would be read back after the first write had already changed it.
    """
    actual = mt.from_numpy(x.copy())
    actual[:, [0, 1]] = actual[:, [1, 0]]

    expected = x.copy()
    expected[:, [0, 1]] = expected[:, [1, 0]]
    np.testing.assert_array_equal(actual.numpy(), expected)


def test_a_value_that_does_not_fit_names_both_shapes(x):
    with pytest.raises(ValueError, match=r"shape \[2, 3, 4\] into the selection"):
        mt.from_numpy(x.copy())[:, [0, 2]] = mt.from_numpy(x.copy())


def test_writing_through_a_parameter_reaches_the_layer():
    """The write is a basic assignment per position, so it shares the storage
    the rest of `__setitem__` does -- a copy here would silently do nothing."""
    from minitensor import nn

    layer = nn.DenseLayer(3, 2)
    parameter = layer.parameters()[0]
    parameter[:, [0]] = mt.Tensor(0.0)

    assert np.abs(layer.parameters()[0].numpy()[:, 0]).max() == 0.0


def test_a_newaxis_changes_how_the_value_lines_up_not_where_it_lands(x):
    """`None` adds an axis to the selection but names no axis to write into."""
    actual = mt.from_numpy(x.copy())
    actual[None] = 5.0
    expected = x.copy()
    expected[None] = 5.0
    np.testing.assert_array_equal(actual.numpy(), expected)

    actual = mt.from_numpy(x.copy())
    actual[0, None] = mt.from_numpy(np.arange(12, dtype=np.float32).reshape(1, 3, 4))
    expected = x.copy()
    expected[0, None] = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    np.testing.assert_array_equal(actual.numpy(), expected)


def test_writing_is_refused_where_reading_is(x):
    with pytest.raises(IndexError, match="gather"):
        mt.from_numpy(x.copy())[[0, 1], [1, 2]] = 0.0
    with pytest.raises(IndexError):
        mt.from_numpy(x.copy())[:, [3]] = 0.0
    with pytest.raises(IndexError, match="boolean index has 4"):
        mt.from_numpy(x.copy())[:, np.array([True, False, True, False])] = 0.0


def test_a_write_a_pending_backward_still_needs_is_refused(x):
    """The same guard the basic path has: the write goes through the storage a
    saved tensor is holding."""
    t = mt.from_numpy(x.copy())
    t.requires_grad_(True)
    kept = t * 2.0

    with pytest.raises(ValueError, match="pending backward"):
        t[:, [0]] = 0.0

    kept.sum().backward()
    mt.clear_autograd_graph()
