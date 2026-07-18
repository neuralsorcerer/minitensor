# Copyright (c) 2026 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""NumPy-style fancy indexing: boolean masks and integer-list row selection.

Boolean masks follow NumPy's rule: the mask's shape must equal the tensor's
leading ``mask.ndim`` dimensions, and selection stacks the remaining blocks —
so a full-shape mask yields a 1-D tensor of elements and a 1-D mask over a
matrix yields rows. Integer lists/arrays/tensors select rows along dim 0 with
negative wrapping.
"""

import numpy as np
import pytest

import minitensor as mt


@pytest.fixture()
def x():
    rng = np.random.RandomState(53)
    return rng.randn(4, 3).astype(np.float32)


def test_full_shape_bool_mask_selects_elements(x):
    m = x > 0
    np.testing.assert_allclose(mt.from_numpy(x)[mt.from_numpy(m)].numpy(), x[m])
    np.testing.assert_allclose(mt.from_numpy(x)[m].numpy(), x[m])


def test_row_bool_mask_selects_rows(x):
    rm = np.array([True, False, True, False])
    np.testing.assert_allclose(mt.from_numpy(x)[rm].numpy(), x[rm])
    np.testing.assert_allclose(
        mt.from_numpy(x)[[True, False, True, False]].numpy(), x[rm]
    )
    np.testing.assert_allclose(mt.from_numpy(x)[mt.from_numpy(rm)].numpy(), x[rm])


def test_mask_edge_shapes(x):
    all_false = np.zeros(4, dtype=bool)
    assert mt.from_numpy(x)[all_false].numpy().shape == (0, 3)
    # 0-d masks add a leading axis exactly like NumPy.
    np.testing.assert_allclose(
        mt.from_numpy(x)[np.array(True)].numpy(), x[np.array(True)]
    )
    assert mt.from_numpy(x)[np.array(False)].numpy().shape == (0, 4, 3)

    y = np.random.RandomState(7).randn(2, 3, 4).astype(np.float32)
    m2 = y[:, :, 0] > 0
    np.testing.assert_allclose(mt.from_numpy(y)[m2].numpy(), y[m2])


def test_int_row_selection(x):
    np.testing.assert_allclose(
        mt.from_numpy(x)[[2, 0, 2, -1]].numpy(), x[[2, 0, 2, -1]]
    )
    np.testing.assert_allclose(
        mt.from_numpy(x)[np.array([1, -2], dtype=np.int64)].numpy(), x[[1, -2]]
    )
    np.testing.assert_allclose(
        mt.from_numpy(x)[np.array([0, 3], dtype=np.int32)].numpy(), x[[0, 3]]
    )
    np.testing.assert_allclose(
        mt.from_numpy(x)[mt.from_numpy(np.array([3, 1], dtype=np.int64))].numpy(),
        x[[3, 1]],
    )
    assert mt.from_numpy(x)[[]].numpy().shape == (0, 3)


def test_fancy_index_errors(x):
    with pytest.raises(Exception):
        mt.from_numpy(x)[np.array([True, False])]  # mask length mismatch
    with pytest.raises(IndexError):
        mt.from_numpy(x)[[7]]
    with pytest.raises(IndexError):
        mt.from_numpy(x)[[-5]]


def test_basic_indexing_unaffected(x):
    np.testing.assert_allclose(mt.from_numpy(x)[1].numpy(), x[1])
    np.testing.assert_allclose(mt.from_numpy(x)[1:3].numpy(), x[1:3])
    np.testing.assert_allclose(mt.from_numpy(x)[np.int64(2)].numpy(), x[2])


def test_masked_index_gradients(x):
    # Gradient scatters selected blocks back; unselected positions get zero.
    m = x > 0
    mask_t = mt.from_numpy(m)
    t = mt.from_numpy(x.copy())
    t.requires_grad_(True)
    (t[mask_t] * 2.0).sum().backward()
    np.testing.assert_allclose(
        mt.get_gradient(t).numpy(), np.where(m, 2.0, 0.0), rtol=1e-6
    )
    mt.clear_autograd_graph()

    rm = mt.from_numpy(np.array([True, False, True, False]))
    t = mt.from_numpy(x.copy())
    t.requires_grad_(True)
    (t[rm] * 3.0).sum().backward()
    expected = np.zeros_like(x)
    expected[[0, 2]] = 3.0
    np.testing.assert_allclose(mt.get_gradient(t).numpy(), expected, rtol=1e-6)
    mt.clear_autograd_graph()


def test_bool_mask_setitem_scalar(x):
    m = x > 0
    t = mt.from_numpy(x.copy())
    t[mt.from_numpy(m)] = 0.0
    ref = x.copy()
    ref[m] = 0.0
    np.testing.assert_allclose(t.numpy(), ref)

    rm = np.array([True, False, True, False])
    t = mt.from_numpy(x.copy())
    t[rm] = -1.5
    ref = x.copy()
    ref[rm] = -1.5
    np.testing.assert_allclose(t.numpy(), ref)

    it = mt.from_numpy(np.arange(6, dtype=np.int64).reshape(2, 3))
    it[mt.from_numpy(np.array([True, False]))] = 7
    ref_i = np.arange(6).reshape(2, 3)
    ref_i[np.array([True, False])] = 7
    np.testing.assert_array_equal(it.numpy(), ref_i)


def test_bool_mask_setitem_rejections(x):
    with pytest.raises(TypeError):
        t = mt.from_numpy(x.copy())
        t[mt.from_numpy(x > 0)] = mt.from_numpy(x)  # tensor values unsupported
    with pytest.raises(IndexError):
        t = mt.from_numpy(x.copy())
        t[mt.from_numpy(np.array([True]))] = 0.0  # mask shape mismatch
