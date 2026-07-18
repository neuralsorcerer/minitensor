# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

import minitensor as mt
from minitensor import (
    expand,
    flatten,
)
from minitensor import functional as F
from minitensor import (
    gather,
    permute,
    ravel,
    reshape,
    squeeze,
    swapaxes,
    swapdims,
    transpose,
    unsqueeze,
    view,
)
from minitensor.functional import moveaxis as F_moveaxis
from minitensor.functional import movedim as F_movedim
from minitensor.tensor import Tensor


def test_functional_reshape():
    t = Tensor.arange(0, 24).reshape([2, 3, 4])
    r = F.reshape(t, (4, 6))
    assert r.shape == (4, 6)
    assert np.array_equal(r.numpy(), t.reshape(4, 6).numpy())


def test_top_level_reshape():
    t = Tensor.arange(0, 24)
    r = reshape(t, (2, 3, 4))
    assert r.shape == (2, 3, 4)
    assert np.array_equal(r.numpy(), t.reshape(2, 3, 4).numpy())


def test_flatten_and_ravel():
    t = Tensor.ones([2, 3, 4])
    f = t.flatten()
    r = t.ravel()
    assert f.shape == (24,)
    assert r.shape == (24,)
    assert np.array_equal(f.numpy(), r.numpy())


def test_flatten_range_and_error():
    t = Tensor.arange(0, 24).reshape([2, 3, 4])
    f = t.flatten(1, -1)
    assert f.shape == (2, 12)
    with pytest.raises(ValueError):
        t.flatten(2, 0)


def test_flatten_negative_start_dim():
    t = Tensor.arange(0, 24).reshape([2, 3, 4])
    f = t.flatten(-3, -2)
    assert f.shape == (6, 4)


def test_functional_flatten():
    t = Tensor.arange(0, 24).reshape([2, 3, 4])
    f = F.flatten(t, 1, -1)
    assert f.shape == (2, 12)
    assert np.array_equal(f.numpy(), t.flatten(1, -1).numpy())


def test_top_level_flatten():
    t = Tensor.arange(0, 24).reshape([2, 3, 4])
    f = flatten(t, 1, -1)
    assert f.shape == (2, 12)
    assert np.array_equal(f.numpy(), t.flatten(1, -1).numpy())


def test_functional_ravel():
    t = Tensor.arange(0, 24).reshape([2, 3, 4])
    r = F.ravel(t)
    assert r.shape == (24,)
    assert np.array_equal(r.numpy(), t.ravel().numpy())


def test_top_level_ravel():
    t = Tensor.arange(0, 24).reshape([2, 3, 4])
    r = ravel(t)
    assert r.shape == (24,)
    assert np.array_equal(r.numpy(), t.ravel().numpy())


def test_functional_view():
    t = Tensor.arange(0, 24)
    v = F.view(t, 2, 12)
    assert v.shape == (2, 12)
    assert np.array_equal(v.numpy(), t.view(2, 12).numpy())


def test_top_level_view():
    t = Tensor.arange(0, 24)
    v = view(t, 2, 12)
    assert v.shape == (2, 12)
    assert np.array_equal(v.numpy(), t.view(2, 12).numpy())


def test_view_invalid_shape():
    t = Tensor.arange(0, 10)
    with pytest.raises(ValueError):
        F.view(t, 3, 4)


def test_unsqueeze_negative_dim():
    t = Tensor([[1, 2, 3]])
    u = t.unsqueeze(-1)
    assert u.shape == (1, 3, 1)
    u2 = t.unsqueeze(-2)
    assert u2.shape == (1, 1, 3)
    with pytest.raises(IndexError):
        t.unsqueeze(3)
    with pytest.raises(IndexError):
        t.unsqueeze(-4)


def test_squeeze_behavior():
    t = Tensor([[[1.0], [2.0], [3.0]]])  # shape (1,3,1)
    s = t.squeeze()
    assert s.shape == (3,)
    s_neg = t.squeeze(-1)
    assert s_neg.shape == (1, 3)
    s_neg2 = t.squeeze(-3)
    assert s_neg2.shape == (3, 1)
    scalar = Tensor([[[1.0]]]).squeeze()
    assert scalar.shape == ()
    with pytest.raises(IndexError):
        t.squeeze(3)
    with pytest.raises(IndexError):
        t.squeeze(-4)


def test_functional_squeeze_unsqueeze():
    t = Tensor.arange(0, 6).reshape([1, 2, 3, 1])

    s = F.squeeze(t, 0)
    assert np.array_equal(s.numpy(), t.squeeze(0).numpy())

    u = F.unsqueeze(s, 0)
    assert np.array_equal(u.numpy(), s.unsqueeze(0).numpy())


def test_top_level_squeeze_unsqueeze():
    t = Tensor.arange(0, 6).reshape([2, 3])

    u = unsqueeze(t, 0)
    assert np.array_equal(u.numpy(), t.unsqueeze(0).numpy())

    s = squeeze(u, 0)
    assert np.array_equal(s.numpy(), t.numpy())


def test_expand_basic():
    t = Tensor([1, 2, 3]).unsqueeze(0)
    e = t.expand(4, 3)
    np.testing.assert_array_equal(
        e.numpy(), np.array([[1, 2, 3]] * 4, dtype=np.float32)
    )


def test_expand_negative_one():
    t = Tensor([[1], [2]])
    e = t.expand(-1, 3)
    np.testing.assert_array_equal(
        e.numpy(), np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float32)
    )


def test_expand_invalid():
    t = Tensor([1, 2, 3])
    # An existing dimension of size != 1 cannot change size.
    with pytest.raises(Exception):
        t.expand(4)
    with pytest.raises(Exception):
        t.expand(2, 4)


def test_functional_expand():
    t = Tensor([[1], [2]])
    e = F.expand(t, -1, 3)
    np.testing.assert_array_equal(
        e.numpy(), np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float32)
    )


def test_top_level_expand():
    t = Tensor([[1], [2]])
    e = expand(t, -1, 3)
    np.testing.assert_array_equal(
        e.numpy(), np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float32)
    )


def test_permute_reorders_dimensions():
    x = mt.arange(24).reshape(2, 3, 4)
    y = x.permute(2, 0, 1)
    expected = np.arange(24, dtype=np.float32).reshape(2, 3, 4).transpose(2, 0, 1)
    assert np.allclose(y.numpy(), expected)


def test_permute_supports_negative_dims():
    x = mt.arange(24).reshape(2, 3, 4)
    y = x.permute(2, -3, -2)
    expected = np.arange(24, dtype=np.float32).reshape(2, 3, 4).transpose(2, 0, 1)
    assert np.allclose(y.numpy(), expected)


def test_permute_accepts_sequence():
    x = mt.arange(24).reshape(2, 3, 4)
    y = x.permute([2, 0, 1])
    expected = np.arange(24, dtype=np.float32).reshape(2, 3, 4).transpose(2, 0, 1)
    assert np.allclose(y.numpy(), expected)


def test_permute_invalid_dims_raises():
    x = mt.arange(6).reshape(1, 2, 3)
    with pytest.raises(ValueError):
        x.permute(0, 0, 1)


def test_permute_dim_length_mismatch():
    x = mt.arange(6).reshape(1, 2, 3)
    with pytest.raises(ValueError):
        x.permute(0, 1)


def test_permute_out_of_range():
    x = mt.arange(6).reshape(1, 2, 3)
    with pytest.raises(IndexError):
        x.permute(0, 1, 3)


def test_functional_transpose():
    t = Tensor.arange(0, 24).reshape([2, 3, 4])
    tr = F.transpose(t, 0, 1)
    assert tr.shape == (3, 2, 4)
    assert np.array_equal(tr.numpy(), t.transpose(0, 1).numpy())


def test_top_level_transpose():
    t = Tensor.arange(0, 24).reshape([2, 3, 4])
    tr = transpose(t, 1, 2)
    assert tr.shape == (2, 4, 3)
    assert np.array_equal(tr.numpy(), t.transpose(1, 2).numpy())


def test_functional_permute():
    t = Tensor.arange(0, 24).reshape([2, 3, 4])
    p = F.permute(t, (1, 2, 0))
    assert p.shape == (3, 4, 2)
    assert np.array_equal(p.numpy(), t.permute(1, 2, 0).numpy())


def test_top_level_permute():
    t = Tensor.arange(0, 24).reshape([2, 3, 4])
    p = permute(t, (2, 0, 1))
    assert p.shape == (4, 2, 3)
    assert np.array_equal(p.numpy(), t.permute(2, 0, 1).numpy())


def test_tensor_swapaxes():
    t = Tensor.arange(0, 24).reshape([2, 3, 4])
    s = t.swapaxes(0, 2)
    assert s.shape == (4, 3, 2)
    assert np.array_equal(s.numpy(), t.transpose(0, 2).numpy())


def test_functional_swapaxes():
    t = Tensor.arange(0, 24).reshape([2, 3, 4])
    s = F.swapaxes(t, 1, 2)
    assert s.shape == (2, 4, 3)
    assert np.array_equal(s.numpy(), t.swapaxes(1, 2).numpy())


def test_top_level_swapaxes():
    t = Tensor.arange(0, 24).reshape([2, 3, 4])
    s = swapaxes(t, 0, 1)
    assert s.shape == (3, 2, 4)
    assert np.array_equal(s.numpy(), t.swapaxes(0, 1).numpy())


def test_swapdims_alias():
    t = Tensor.arange(0, 24).reshape([2, 3, 4])
    s0 = t.swapdims(0, 1)
    assert np.array_equal(s0.numpy(), t.swapaxes(0, 1).numpy())
    s1 = F.swapdims(t, 0, 2)
    assert np.array_equal(s1.numpy(), t.swapaxes(0, 2).numpy())
    s2 = swapdims(t, 1, 2)
    assert np.array_equal(s2.numpy(), t.swapaxes(1, 2).numpy())


def test_tensor_movedim_single_and_negative():
    x = mt.arange(6).reshape(1, 2, 3)
    y = x.movedim(0, -1)
    expected = x.permute(1, 2, 0)
    assert y.tolist() == expected.tolist()
    assert y.shape == (2, 3, 1)


def test_tensor_movedim_sequence():
    x = mt.arange(24).reshape(2, 3, 4)
    y = x.movedim((0, 2), (1, 0))
    expected = x.permute(2, 0, 1)
    assert y.tolist() == expected.tolist()
    assert y.shape == (4, 2, 3)


def test_tensor_moveaxis_alias():
    x = mt.arange(6).reshape(1, 2, 3)
    y = x.moveaxis(0, 2)
    expected = x.movedim(0, 2)
    assert y.tolist() == expected.tolist()


def test_functional_and_top_level_movedim_moveaxis():
    x = mt.arange(6).reshape(1, 2, 3)
    expected = x.permute(1, 2, 0)
    assert F_movedim(x, 0, 2).tolist() == expected.tolist()
    assert F_moveaxis(x, 0, 2).tolist() == expected.tolist()
    assert mt.movedim(x, 0, 2).tolist() == expected.tolist()
    assert mt.moveaxis(x, 0, 2).tolist() == expected.tolist()


def test_movedim_invalid_length():
    x = mt.arange(6)
    with pytest.raises(ValueError):
        x.movedim((0, 1), (0,))


def test_narrow_1d():
    t = mt.arange(0, 10)
    r = t.narrow(0, 2, 5)
    assert np.array_equal(r.numpy(), np.arange(2, 7))


def test_narrow_negative_dim():
    t = mt.arange(0, 12).reshape((3, 4))
    r = t.narrow(-1, 1, 2)
    expected = np.array([[1, 2], [5, 6], [9, 10]], dtype=np.float32)
    assert np.array_equal(r.numpy(), expected)


def test_narrow_out_of_bounds():
    t = mt.arange(0, 5)
    with pytest.raises((ValueError, IndexError)):
        t.narrow(0, 3, 3)


def test_functional_and_top_level_narrow():
    t = mt.arange(0, 6)
    r_method = t.narrow(0, 2, 2)
    r_func = F.narrow(t, 0, 2, 2)
    r_top = mt.narrow(t, 0, 2, 2)
    expected = np.arange(2, 4)
    assert np.array_equal(r_method.numpy(), expected)
    assert np.array_equal(r_func.numpy(), expected)
    assert np.array_equal(r_top.numpy(), expected)


def test_all_any():
    t = mt.Tensor([[1.0, 0.0], [2.0, 3.0]])
    assert t.any().tolist() is True
    assert t.all().tolist() is False
    b = mt.Tensor([[True, False], [True, True]], dtype="bool")
    res = b.all(dim=1)
    assert res.tolist() == [False, True]


def test_indexing_and_assignment():
    t = mt.Tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    assert t[0, 1].tolist() == 1.0
    t[0, 1] = 10.0
    assert t[0, 1].tolist() == 10.0
    col = t[:, 1]
    assert col.tolist() == [10.0, 4.0]


def test_negative_indexing_and_bounds():
    t = mt.Tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    assert t[1, -1].tolist() == 5.0
    with pytest.raises(IndexError):
        _ = t[2, 0]
    with pytest.raises(IndexError):
        _ = t[0, -4]


def test_slice_with_start():
    t = mt.arange(10)
    sliced = t[5:]
    assert sliced.tolist() == [5.0, 6.0, 7.0, 8.0, 9.0]


def test_slice_with_positive_step():
    t = mt.arange(6)
    sliced = t[::2]
    np.testing.assert_allclose(
        sliced.numpy(), np.array([0.0, 2.0, 4.0], dtype=np.float32)
    )


def test_slice_out_of_range_empty():
    t = mt.arange(5)
    assert t[10:].tolist() == []


def test_reverse_slice_error():
    t = mt.arange(10)
    with pytest.raises(IndexError):
        _ = t[::-1]


def test_slice_assignment_with_step():
    t = mt.arange(6)
    t[1::2] = mt.Tensor([10.0, 20.0, 30.0])
    np.testing.assert_allclose(
        t.numpy(), np.array([0.0, 10.0, 2.0, 20.0, 4.0, 30.0], dtype=np.float32)
    )


def test_multi_dim_slice():
    t = mt.Tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    sub = t[1:, :2]
    np.testing.assert_allclose(
        sub.numpy(), np.array([[3.0, 4.0], [6.0, 7.0]], dtype=np.float32)
    )


def test_boolean_mask_indexing():
    # Formerly unsupported (this test asserted a TypeError); boolean masks now
    # follow NumPy semantics. Full coverage lives in test_fancy_indexing.py.
    t = mt.Tensor([0.0, 1.0, 2.0, 3.0])
    mask = t.gt(mt.Tensor([1.0]))
    np.testing.assert_allclose(t[mask].numpy(), np.array([2.0, 3.0], dtype=np.float32))


def test_integer_array_indexing():
    # Formerly unsupported (this test asserted a TypeError); integer lists now
    # select rows along dim 0 like NumPy.
    t = mt.Tensor([10.0, 20.0, 30.0, 40.0])
    np.testing.assert_allclose(
        t[[2, 0]].numpy(), np.array([30.0, 10.0], dtype=np.float32)
    )


def test_tensor_index_select_dim0():
    t = mt.arange(0, 6).reshape((3, 2))
    out = t.index_select(0, [0, 2])
    expected = np.arange(0, 6, dtype=np.float32).reshape(3, 2)[[0, 2], :]
    np.testing.assert_allclose(out.numpy(), expected)


def test_tensor_index_select_neg_dim():
    t = mt.arange(0, 6).reshape((2, 3))
    out = t.index_select(-1, [2, 0])
    expected = np.arange(0, 6, dtype=np.float32).reshape(2, 3)[:, [2, 0]]
    np.testing.assert_allclose(out.numpy(), expected)


def test_functional_and_top_level_index_select():
    t = mt.arange(0, 6).reshape((3, 2))
    expected = np.arange(0, 6, dtype=np.float32).reshape(3, 2)[:, [1]]
    out_func = F.index_select(t, 1, [1])
    out_top = mt.index_select(t, 1, [1])
    np.testing.assert_allclose(out_func.numpy(), expected)
    np.testing.assert_allclose(out_top.numpy(), expected)


def test_index_select_out_of_range():
    t = mt.arange(0, 6).reshape((3, 2))
    with pytest.raises(IndexError):
        t.index_select(0, [3])


def test_gather_basic():
    t = Tensor([[1, 2, 3], [4, 5, 6]])
    idx = Tensor([[0, 1, 1], [2, 0, 0]], dtype="int64")
    g = t.gather(1, idx)
    expected = np.array([[1, 2, 2], [6, 4, 4]], dtype=np.float32)
    assert np.array_equal(g.numpy(), expected)


def test_gather_error():
    t = Tensor([1, 2, 3])
    idx = Tensor([3], dtype="int64")
    with pytest.raises(Exception):
        t.gather(0, idx)


def test_functional_top_level():
    t = Tensor([10, 20, 30])
    idx = Tensor([2, 0, 1], dtype="int64")
    g1 = F.gather(t, 0, idx)
    g2 = gather(t, 0, idx)
    expected = np.array([30.0, 10.0, 20.0], dtype=np.float32)
    assert np.array_equal(g1.numpy(), expected)
    assert np.array_equal(g2.numpy(), expected)


def test_tensor_diagonal_default():
    tensor = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    diag = tensor.diagonal()
    np.testing.assert_allclose(diag.numpy(), np.array([1.0, 4.0], dtype=np.float32))


def test_tensor_diagonal_offsets():
    tensor = mt.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    upper = tensor.diagonal(offset=1)
    lower = tensor.diagonal(offset=-1)
    np.testing.assert_allclose(upper.numpy(), np.array([2.0, 6.0], dtype=np.float32))
    np.testing.assert_allclose(lower.numpy(), np.array([4.0], dtype=np.float32))


def test_tensor_diagonal_high_dimension_shape():
    tensor = mt.arange(24, dtype="float32").reshape(2, 3, 4)
    diag = tensor.diagonal(dim1=1, dim2=2)
    assert diag.shape == (2, 3)


def test_tensor_diagonal_empty_for_large_offset():
    tensor = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    diag = tensor.diagonal(offset=5)
    assert diag.shape == (0,)
    assert diag.numel() == 0


def test_tensor_trace_matches_numpy():
    tensor = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    traced = tensor.trace()
    np.testing.assert_allclose(traced.numpy(), np.array(5.0, dtype=np.float32))


def test_functional_diagonal_and_trace():
    tensor = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    diag_fn = mt.diagonal(tensor)
    trace_fn = mt.trace(tensor)
    np.testing.assert_allclose(diag_fn.numpy(), np.array([1.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(trace_fn.numpy(), np.array(5.0, dtype=np.float32))


def test_diagonal_backward_gradients():
    tensor = mt.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    diag = tensor.diagonal()
    loss = diag.sum()
    loss.backward()
    grad = tensor.grad
    assert grad is not None
    np.testing.assert_allclose(
        grad.numpy(), np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    )
    mt.clear_autograd_graph()


def test_getitem_newaxis_matches_numpy():
    x = np.arange(20, dtype=np.float64).reshape(4, 5)
    t = mt.Tensor(x)
    cases = [
        (t[None], x[None]),
        (t[:, None], x[:, None]),
        (t[None, :], x[None, :]),
        (t[:, :, None], x[:, :, None]),
        (t[None, None], x[None, None]),
        (t[1, None], x[1, None]),
        (t[None, 1], x[None, 1]),
        (t[None, 1, None], x[None, 1, None]),
        (t[1:3, None], x[1:3, None]),
        (t[:, None, 2], x[:, None, 2]),
    ]
    for got, exp in cases:
        got = got.numpy()
        assert got.shape == exp.shape, (got.shape, exp.shape)
        np.testing.assert_allclose(got, exp)


def test_getitem_newaxis_3d_and_scalar():
    y = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    ty = mt.Tensor(y)
    np.testing.assert_allclose(ty[:, :, None, :].numpy(), y[:, :, None, :])
    np.testing.assert_allclose(ty[1, None, :, 2].numpy(), y[1, None, :, 2])
    s = mt.Tensor(3.5)
    assert s[None].numpy().shape == (1,)
    np.testing.assert_allclose(s[None].numpy(), np.array(3.5)[None])


def test_index_select_accepts_tensor_indices():
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    tx = mt.Tensor(x)
    idx = mt.Tensor([2, 0], dtype="int64")

    result = tx.index_select(1, idx)
    np.testing.assert_allclose(result.numpy(), x[:, [2, 0]])

    # int32 index tensors are accepted as well
    idx32 = mt.Tensor([1, 2], dtype="int32")
    result32 = tx.index_select(0, idx32)
    np.testing.assert_allclose(result32.numpy(), x[[1, 2]])

    # top-level function accepts tensors too
    result_fn = mt.index_select(tx, 1, idx)
    np.testing.assert_allclose(result_fn.numpy(), x[:, [2, 0]])

    # lists keep working
    result_list = tx.index_select(1, [3, 1])
    np.testing.assert_allclose(result_list.numpy(), x[:, [3, 1]])


def test_index_select_rejects_bad_index_tensors():
    tx = mt.Tensor(np.arange(6, dtype=np.float32).reshape(2, 3))
    with pytest.raises(TypeError):
        tx.index_select(0, mt.Tensor([0.5, 1.0]))
    with pytest.raises(ValueError):
        tx.index_select(0, mt.Tensor([[0], [1]], dtype="int64"))
    with pytest.raises(ValueError):
        tx.index_select(0, [-1])


def test_expand_adds_leading_dimensions():
    x = np.arange(4, dtype=np.float32)
    tx = mt.Tensor(x)
    expanded = tx.expand(3, 2, 4)
    np.testing.assert_allclose(expanded.numpy(), np.broadcast_to(x, (3, 2, 4)))

    with pytest.raises(ValueError):
        tx.expand(-1, 4)


def test_ellipsis_indexing_matches_numpy():
    src = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    t = mt.Tensor(src)

    np.testing.assert_allclose(t[..., 0].numpy(), src[..., 0])
    np.testing.assert_allclose(t[..., 1:3].numpy(), src[..., 1:3])
    np.testing.assert_allclose(t[0, ...].numpy(), src[0, ...])
    np.testing.assert_allclose(t[0, ..., 2].numpy(), src[0, ..., 2])
    np.testing.assert_allclose(t[...].numpy(), src)
    np.testing.assert_allclose(t[None, ..., 0].numpy(), src[None][..., 0])

    with pytest.raises(IndexError):
        _ = t[..., ..., 0]


def test_ellipsis_setitem_matches_numpy():
    src = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

    t = mt.Tensor(src.copy())
    t[..., 0] = 0.0
    ref = src.copy()
    ref[..., 0] = 0.0
    np.testing.assert_allclose(t.numpy(), ref)

    t = mt.Tensor(src.copy())
    t[0, ...] = 7.0
    ref = src.copy()
    ref[0, ...] = 7.0
    np.testing.assert_allclose(t.numpy(), ref)


def test_scalar_tensor_setitem_raises_cleanly():
    # Previously reached shape[0] on a 0-d tensor and panicked; must be a
    # regular IndexError now.
    t = mt.Tensor(np.array(1.0, dtype=np.float32))
    with pytest.raises(IndexError):
        t[0] = 2.0
