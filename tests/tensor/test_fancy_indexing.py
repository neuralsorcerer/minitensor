# Copyright (c) Soumyadip Sarkar.
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


def test_bool_mask_setitem_tensor_values(x):
    # NumPy semantics: values broadcast to the selection shape
    # [n_true] + trailing dims.
    m = x > 0
    n_true = int(m.sum())
    vals = np.arange(n_true, dtype=np.float32)
    t = mt.from_numpy(x.copy())
    t[mt.from_numpy(m)] = mt.from_numpy(vals)
    ref = x.copy()
    ref[m] = vals
    np.testing.assert_allclose(t.numpy(), ref)

    # A row-shaped value is copied into every selected row.
    rm = np.array([True, False, True, False])
    row = np.array([9.0, 8.0, 7.0], dtype=np.float32)
    t = mt.from_numpy(x.copy())
    t[rm] = mt.from_numpy(row)
    ref = x.copy()
    ref[rm] = row
    np.testing.assert_allclose(t.numpy(), ref)

    # Exact [n_rows, trailing] values assign block-by-block.
    rows = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    t = mt.from_numpy(x.copy())
    t[rm] = mt.from_numpy(rows)
    ref = x.copy()
    ref[rm] = rows
    np.testing.assert_allclose(t.numpy(), ref)

    # Plain lists convert (and cast to the tensor's dtype).
    t = mt.from_numpy(x.copy())
    t[rm] = [1, 2, 3]
    ref = x.copy()
    ref[rm] = [1, 2, 3]
    np.testing.assert_allclose(t.numpy(), ref)


def test_bool_mask_setitem_rejections(x):
    with pytest.raises(ValueError):
        t = mt.from_numpy(x.copy())
        # (4, 3) cannot broadcast to the (n_true,) selection shape.
        t[mt.from_numpy(x > 0)] = mt.from_numpy(x)
    with pytest.raises(ValueError):
        t = mt.from_numpy(x.copy())
        # Wrong number of per-element values.
        t[mt.from_numpy(x > 0)] = mt.from_numpy(
            np.array([1.0], dtype=np.float32).repeat(2)
        )
    with pytest.raises(IndexError):
        t = mt.from_numpy(x.copy())
        t[mt.from_numpy(np.array([True]))] = 0.0  # mask shape mismatch


def test_setitem_self_referential():
    # `t[mask] = t` and `t[i] = t[j]` used to die with "Already mutably
    # borrowed": the &mut receiver held the PyCell borrow while extracting
    # the value. The engine op also snapshots the source before mutating, so
    # aliased storage stays sound.
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    t = mt.from_numpy(v.copy())
    t[mt.from_numpy(np.ones(3, dtype=bool))] = t
    np.testing.assert_allclose(t.numpy(), v)

    t = mt.from_numpy(v.copy())
    t[0] = t[1]
    np.testing.assert_allclose(t.numpy(), np.array([2.0, 2.0, 3.0], dtype=np.float32))


def test_bool_mask_setitem_casts_value_dtype():
    it = mt.from_numpy(np.arange(4, dtype=np.int64))
    it[mt.from_numpy(np.array([True, False, True, False]))] = mt.from_numpy(
        np.array([9.7, 8.2], dtype=np.float32)
    )
    np.testing.assert_array_equal(it.numpy(), np.array([9, 1, 8, 3]))


def test_assignment_through_parameters_reaches_the_layer():
    """`parameters()` hands out live handles, so writing to one must stick.

    These share storage with the layer. Copying on write instead made the
    assignment silently update a private copy: it appeared to succeed and the
    layer kept its old weights, which is the worst way for this to fail.
    """
    from minitensor import nn

    layer = nn.DenseLayer(3, 2)
    param = layer.parameters()[0]
    param[...] = mt.Tensor(np.zeros_like(param.numpy()))

    assert np.abs(layer.parameters()[0].numpy()).max() == 0.0


def test_assignment_through_an_explicit_copy_stays_independent():
    """The flip side: only live handles alias, explicit copies never do.

    `clone` deep-copies, `detach` produces a non-gradient tensor and a reshape
    carries a grad_fn -- all three take the copy-on-write path, so assigning to
    them must leave the original alone.
    """
    original = mt.Tensor([1.0, 2.0], requires_grad=True)

    copy = original.clone()
    copy[0] = mt.Tensor(99.0)
    assert original.tolist() == [1.0, 2.0]
    assert copy.tolist() == [99.0, 2.0]

    detached = original.detach()
    detached[0] = mt.Tensor(77.0)
    assert original.tolist() == [1.0, 2.0]
    assert detached.tolist() == [77.0, 2.0]

    reshaped = original.reshape(2, 1)
    reshaped[0] = mt.Tensor([55.0])
    assert original.tolist() == [1.0, 2.0]
    assert reshaped.tolist() == [[55.0], [2.0]]


def test_optimizer_updates_still_reach_the_layer():
    """Assignment now uses the same write rule optimizers already used.

    Optimizers mutate parameters through shared storage; this guards the path
    they depend on while the assignment path shares it.
    """
    from minitensor import nn

    layer = nn.DenseLayer(3, 2)
    before = layer.parameters()[0].numpy().copy()
    optimizer = mt.optim.SGD(layer.parameters(), lr=0.1)

    layer(mt.Tensor(np.ones((4, 3), dtype=np.float32))).sum().backward()
    optimizer.step()

    assert not np.allclose(before, layer.parameters()[0].numpy())
    mt.clear_autograd_graph()


def test_every_assignment_form_agrees_about_reaching_the_layer():
    """`__setitem__` has three branches; they must not disagree.

    The boolean-mask branch always went through the shared-write rule, while the
    slice branch had its own copy-on-write. Same operator, same tensor, opposite
    outcome depending on how you spelled the index. All three now agree.
    """
    from minitensor import nn

    def wrote_through(apply):
        layer = nn.DenseLayer(3, 2)
        param = layer.parameters()[0]
        apply(param)
        return np.abs(layer.parameters()[0].numpy()).max() == 0.0

    def by_ellipsis(p):
        p[...] = mt.Tensor(np.zeros_like(p.numpy()))

    def by_mask(p):
        mask = mt.Tensor(np.ones(p.numpy().shape, dtype=bool), dtype="bool")
        p[mask] = mt.Tensor(0.0)

    def by_slice(p):
        p[0:2] = mt.Tensor(np.zeros_like(p.numpy()[0:2]))

    assert wrote_through(by_ellipsis)
    assert wrote_through(by_mask)
    assert wrote_through(by_slice)
