# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import warnings

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F
from minitensor.tensor import Tensor


def test_argmax_dim():
    x = Tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], dtype="float32")
    result = x.argmax(dim=1)
    assert np.array_equal(result.numpy(), np.array([1, 2], dtype=np.int64))


def test_argmax_negative_dim():
    x = Tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], dtype="float32")
    result = x.argmax(dim=-1)
    assert np.array_equal(result.numpy(), np.array([1, 2], dtype=np.int64))


def test_argmin_dim_keepdim():
    x = Tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], dtype="float32")
    result = x.argmin(dim=1, keepdim=True)
    assert result.shape == (2, 1)
    assert np.array_equal(result.numpy(), np.array([[0], [1]], dtype=np.int64))


def test_argmax_no_dim_first_index():
    x = Tensor([1.0, 5.0, 5.0, -1.0], dtype="float32")
    result = x.argmax()
    assert result.shape == ()
    assert result.numpy().item() == 1


def test_argmin_all_equal_returns_zero():
    x = Tensor([2.0, 2.0, 2.0], dtype="float32")
    result = x.argmin()
    assert result.numpy().item() == 0


def test_argmin_negative_dim():
    x = Tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], dtype="float32")
    result = x.argmin(dim=-2)
    assert np.array_equal(result.numpy(), np.array([0, 1, 0], dtype=np.int64))


def test_functional_extrema_return_values_and_indices_like_tensor_methods():
    x_np = np.array([[3.0, 1.0, 2.0], [4.0, 6.0, 5.0]], dtype=np.float32)
    x = Tensor(x_np.tolist())

    max_values, max_indices = F.max(x, dim=1)
    min_values, min_indices = mt.min(x, dim=0)

    np.testing.assert_allclose(max_values.numpy(), np.max(x_np, axis=1))
    assert max_indices.numpy().tolist() == np.argmax(x_np, axis=1).tolist()
    np.testing.assert_allclose(min_values.numpy(), np.min(x_np, axis=0))
    assert min_indices.numpy().tolist() == np.argmin(x_np, axis=0).tolist()
    np.testing.assert_allclose(F.max(x).numpy(), x_np.max())
    np.testing.assert_allclose(mt.min(x).numpy(), x_np.min())


def test_functional_arg_reductions_support_keepdim_and_top_level_exports():
    x_np = np.array([[3.0, 1.0, 2.0], [4.0, 6.0, 5.0]], dtype=np.float32)
    x = Tensor(x_np.tolist())

    assert F.argmax(x, dim=1, keepdim=True).shape == (2, 1)
    assert F.argmax(x, dim=1, keepdim=True).numpy().tolist() == [[0], [1]]
    assert mt.argmin(x, dim=-1).numpy().tolist() == np.argmin(x_np, axis=-1).tolist()


def test_max_min_with_indices():
    x = mt.Tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], dtype="float32")
    max_vals, max_idx = x.max(dim=1)
    assert np.array_equal(max_vals.numpy(), np.array([5.0, 6.0], dtype=np.float32))
    assert np.array_equal(max_idx.numpy(), np.array([1, 2], dtype=np.int64))

    min_vals, min_idx = x.min(dim=1)
    assert np.array_equal(min_vals.numpy(), np.array([1.0, 2.0], dtype=np.float32))
    assert np.array_equal(min_idx.numpy(), np.array([0, 1], dtype=np.int64))


def test_max_min_with_nan_inf():
    t = mt.Tensor([np.nan, 1.0, np.inf, -np.inf], dtype="float32")
    max_val = t.max()
    min_val = t.min()
    assert np.isinf(max_val.numpy()) and max_val.numpy() > 0
    assert np.isinf(min_val.numpy()) and min_val.numpy() < 0


def test_max_min_all_equal():
    t = mt.Tensor([3.0, 3.0, 3.0], dtype="float32")
    assert t.max().numpy() == 3.0
    assert t.min().numpy() == 3.0


def test_max_min_empty_tensor_values():
    t = mt.Tensor(np.array([], dtype=np.float32))
    assert np.isneginf(t.max().numpy())
    assert np.isinf(t.min().numpy())


def test_max_min_all_nan_returns_extremes():
    t = mt.Tensor([np.nan, np.nan], dtype="float32")
    assert np.isneginf(t.max().numpy())
    assert np.isinf(t.min().numpy())


def test_max_min_empty_tensor_with_dim():
    t = mt.Tensor(np.empty((0, 3), dtype=np.float32))
    max_vals, max_idx = t.max(dim=0)
    assert np.isneginf(max_vals.numpy()).all()
    assert np.array_equal(max_idx.numpy(), np.zeros(3, dtype=np.int64))

    min_vals, min_idx = t.min(dim=0)
    assert np.isposinf(min_vals.numpy()).all()
    assert np.array_equal(min_idx.numpy(), np.zeros(3, dtype=np.int64))


def test_max_min_empty_tensor_with_dim_keepdim():
    t = mt.Tensor(np.empty((0, 3), dtype=np.float32))
    max_vals, max_idx = t.max(dim=0, keepdim=True)
    assert max_vals.shape == (1, 3)
    assert np.isneginf(max_vals.numpy()).all()
    assert np.array_equal(max_idx.numpy(), np.zeros((1, 3), dtype=np.int64))

    min_vals, min_idx = t.min(dim=0, keepdim=True)
    assert min_vals.shape == (1, 3)
    assert np.isposinf(min_vals.numpy()).all()
    assert np.array_equal(min_idx.numpy(), np.zeros((1, 3), dtype=np.int64))


def test_max_min_all_nan_with_dim_returns_extremes():
    t = mt.Tensor(np.array([[np.nan, np.nan], [np.nan, np.nan]], dtype=np.float32))
    max_vals, max_idx = t.max(dim=1)
    assert np.isneginf(max_vals.numpy()).all()
    assert np.array_equal(max_idx.numpy(), np.zeros(2, dtype=np.int64))

    min_vals, min_idx = t.min(dim=1)
    assert np.isposinf(min_vals.numpy()).all()
    assert np.array_equal(min_idx.numpy(), np.zeros(2, dtype=np.int64))


def test_max_min_all_nan_with_dim_keepdim():
    t = mt.Tensor(np.array([[np.nan, np.nan], [np.nan, np.nan]], dtype=np.float32))
    max_vals, max_idx = t.max(dim=1, keepdim=True)
    assert max_vals.shape == (2, 1)
    assert np.isneginf(max_vals.numpy()).all()
    assert np.array_equal(max_idx.numpy(), np.zeros((2, 1), dtype=np.int64))

    min_vals, min_idx = t.min(dim=1, keepdim=True)
    assert min_vals.shape == (2, 1)
    assert np.isposinf(min_vals.numpy()).all()
    assert np.array_equal(min_idx.numpy(), np.zeros((2, 1), dtype=np.int64))


def test_max_min_empty_int_tensor_with_dim():
    t = mt.Tensor(np.empty((0, 2), dtype=np.int32), dtype="int32")
    max_vals, max_idx = t.max(dim=0)
    assert max_vals.numpy().tolist() == [np.iinfo(np.int32).min] * 2
    assert np.array_equal(max_idx.numpy(), np.zeros(2, dtype=np.int64))

    min_vals, min_idx = t.min(dim=0)
    assert min_vals.numpy().tolist() == [np.iinfo(np.int32).max] * 2
    assert np.array_equal(min_idx.numpy(), np.zeros(2, dtype=np.int64))


def test_median_global_even_length():
    x = mt.Tensor([3.0, 1.0, 4.0, 2.0], dtype="float32")
    median = x.median()
    assert median.shape == ()
    assert median.numpy() == pytest.approx(2.0)


def test_median_with_dim_returns_indices():
    x = mt.Tensor([[1.0, 3.0, 2.0], [4.0, 6.0, 5.0]], dtype="float32")
    values, indices = x.median(dim=1)
    np.testing.assert_allclose(values.numpy(), np.array([2.0, 5.0], dtype=np.float32))
    np.testing.assert_array_equal(indices.numpy(), np.array([2, 2], dtype=np.int64))


def test_median_keepdim_matches_pytorch_shape():
    x = mt.Tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    values, indices = x.median(dim=1, keepdim=True)
    assert values.shape == (2, 1)
    assert indices.shape == (2, 1)
    np.testing.assert_allclose(
        values.numpy(), np.array([[1.0], [3.0]], dtype=np.float32)
    )
    np.testing.assert_array_equal(indices.numpy(), np.zeros((2, 1), dtype=np.int64))


def test_median_empty_tensor_raises():
    x = mt.Tensor(np.empty((0, 3), dtype=np.float32))
    with pytest.raises(RuntimeError):
        x.median()


def test_median_nan_propagates_global():
    x = mt.Tensor([1.0, float("nan"), 2.0], dtype="float32")
    median = x.median()
    assert median.shape == ()
    assert np.isnan(median.numpy())


def test_median_nan_propagates_with_dim():
    x = mt.Tensor([[1.0, float("nan"), 3.0], [2.0, 4.0, 6.0]], dtype="float32")
    values, indices = x.median(dim=1)
    assert np.isnan(values.numpy()[0])
    assert values.numpy()[1] == pytest.approx(4.0)
    np.testing.assert_array_equal(indices.numpy(), np.array([0, 1], dtype=np.int64))


def test_var_std_support_tuple_dims_and_keepdim():
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    tensor = mt.Tensor(data)

    var = tensor.var(dim=(1, 2), unbiased=False, keepdim=True)
    std = tensor.std(dim=(1, 2), unbiased=False, keepdim=False)

    np.testing.assert_allclose(
        var.numpy(), data.var(axis=(1, 2), keepdims=True), rtol=1e-6
    )
    np.testing.assert_allclose(std.numpy(), data.std(axis=(1, 2)), rtol=1e-6)


def test_unbiased_var_single_sample_returns_nan_without_warning():
    tensor = mt.Tensor([[1.0], [2.0]], dtype="float32")
    result = tensor.var(dim=1, unbiased=True)

    assert result.shape == (2,)
    assert np.isnan(result.numpy()).all()


def test_var_rejects_duplicate_and_invalid_dims_like_other_reductions():
    tensor = mt.Tensor(np.arange(6, dtype=np.float32).reshape(2, 3))

    np.testing.assert_allclose(
        tensor.var(dim=(1, -1), unbiased=False).numpy(),
        tensor.var(dim=1, unbiased=False).numpy(),
        rtol=1e-6,
    )

    with pytest.raises(IndexError):
        tensor.var(dim=2)


def test_functional_std_var_preserve_unbiased_and_keepdim_semantics():
    x_np = np.array([[1.0, 2.0, 3.0], [4.0, 6.0, 8.0]], dtype=np.float32)
    x = mt.Tensor(x_np.tolist())

    np.testing.assert_allclose(
        mt.functional.var(x, dim=1, unbiased=False, keepdim=True).numpy(),
        np.var(x_np, axis=1, keepdims=True),
    )
    np.testing.assert_allclose(
        mt.std(x, dim=0, unbiased=True).numpy(), np.std(x_np, axis=0, ddof=1)
    )


def test_quantile_matches_numpy_scalar():
    data = mt.tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
    quant = data.quantile(0.25)
    expected = np.quantile(data.numpy(), 0.25, method="linear")

    assert quant.shape == ()
    assert quant.requires_grad == data.requires_grad
    np.testing.assert_allclose(quant.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_quantile_dim_keepdim_higher():
    data = mt.tensor([[1.0, 5.0, 2.0], [4.0, 3.0, 6.0]], requires_grad=True)
    quant = data.quantile(0.5, dim=1, keepdim=True, interpolation="higher")
    expected = np.quantile(data.numpy(), 0.5, axis=1, keepdims=True, method="higher")

    assert quant.shape == (2, 1)
    assert quant.requires_grad
    np.testing.assert_allclose(quant.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_functional_quantile_nearest_matches_numpy():
    data = mt.tensor([[0.0, 10.0, 20.0], [30.0, 40.0, 50.0]])
    values = mt.functional.quantile(data, 0.6, dim=1, interpolation="nearest")
    expected = np.quantile(data.numpy(), 0.6, axis=1, method="nearest")

    np.testing.assert_allclose(values.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_functional_quantile_sequence_matches_numpy():
    data = mt.tensor([[0.0, 10.0, 20.0], [30.0, 40.0, 50.0]])
    probs = [0.2, 0.8]
    values = mt.functional.quantile(data, probs, dim=1, keepdim=False)
    expected = np.quantile(data.numpy(), probs, axis=1, method="linear")

    assert values.shape == (2, 2)
    np.testing.assert_allclose(values.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_quantile_invalid_inputs_raise():
    data = mt.tensor([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        data.quantile(1.1)

    with pytest.raises(ValueError):
        data.quantile(0.5, interpolation="invalid")


def test_quantile_sequence_matches_numpy():
    data = mt.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    probs = [0.25, 0.5, 0.75]
    quant = data.quantile(probs)
    expected = np.quantile(data.numpy(), probs, method="linear")

    assert quant.shape == (3,)
    np.testing.assert_allclose(quant.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_quantile_sequence_keepdim_no_dim():
    data = mt.tensor([[1.0, 5.0, 2.0], [4.0, 3.0, 6.0]])
    probs = [0.25, 0.75]
    quant = data.quantile(probs, keepdim=True)
    expected = np.quantile(data.numpy(), probs, keepdims=True, method="linear")

    assert quant.shape == (2, 1, 1)
    np.testing.assert_allclose(quant.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_quantile_sequence_with_dim_keepdim():
    data = mt.tensor([[1.0, 5.0, 2.0], [4.0, 3.0, 6.0]])
    probs = [0.25, 0.75]
    quant = data.quantile(probs, dim=1, keepdim=True)
    expected = np.quantile(data.numpy(), probs, axis=1, keepdims=True, method="linear")

    assert quant.shape == (2, 2, 1)
    np.testing.assert_allclose(quant.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_quantile_with_nan_propagates():
    data = mt.tensor([1.0, np.nan, 2.0])
    quant = data.quantile(0.5)
    expected = np.quantile(data.numpy(), 0.5, method="linear")

    assert quant.shape == ()
    np.testing.assert_allclose(
        quant.numpy(), expected, rtol=1e-6, atol=1e-6, equal_nan=True
    )


def test_quantile_dim_with_nan_propagates_per_slice():
    data = mt.tensor([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])
    probs = [0.25, 0.75]
    quant = data.quantile(probs, dim=1, keepdim=False)
    expected = np.quantile(data.numpy(), probs, axis=1, method="linear")

    assert quant.shape == (2, 2)
    np.testing.assert_allclose(
        quant.numpy(), expected, rtol=1e-6, atol=1e-6, equal_nan=True
    )


def test_nanquantile_matches_numpy():
    data = mt.tensor([np.nan, 1.0, 3.0, np.nan, 5.0])
    quant = data.nanquantile(0.5)
    expected = np.nanquantile(data.numpy(), 0.5, method="linear")

    assert quant.shape == ()
    np.testing.assert_allclose(quant.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_nanmedian_matches_numpy_and_top_level_export():
    data = mt.tensor([np.nan, 1.0, 3.0, np.nan, 5.0])
    expected = np.nanmedian(data.numpy())

    method_result = data.nanmedian()
    functional_result = F.nanmedian(data)
    top_level_result = mt.nanmedian(data)

    assert method_result.shape == ()
    np.testing.assert_allclose(method_result.numpy(), expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        functional_result.numpy(), expected, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(top_level_result.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_nanmedian_dim_keepdim_matches_numpy():
    data = mt.tensor([[1.0, np.nan, 5.0, 7.0], [2.0, 4.0, np.nan, 10.0]])
    values = data.nanmedian(dim=1, keepdim=True)
    expected = np.nanmedian(data.numpy(), axis=1, keepdims=True)

    assert values.shape == (2, 1)
    np.testing.assert_allclose(values.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_nanmedian_all_nan_and_empty_slices_return_nan():
    all_nan = mt.tensor([np.nan, np.nan])
    assert np.isnan(all_nan.nanmedian().numpy())

    data = mt.tensor([[np.nan, np.nan], [1.0, np.nan]])
    values = data.nanmedian(dim=1)

    np.testing.assert_allclose(
        values.numpy(), np.array([np.nan, 1.0], dtype=np.float32), equal_nan=True
    )

    empty = mt.tensor(np.empty((0, 2), dtype=np.float32))
    empty_values = empty.nanmedian(dim=0)
    assert empty_values.shape == (2,)
    assert np.isnan(empty_values.numpy()).all()


def test_nanmedian_rejects_non_float_tensors():
    data = mt.tensor([1, 2, 3], dtype="int32")
    with pytest.raises(ValueError, match="nanmedian"):
        data.nanmedian()


def test_nanquantile_sequence_dim_keepdim():
    data = mt.tensor([[1.0, np.nan, 5.0], [2.0, 4.0, np.nan]])
    probs = [0.25, 0.75]
    quant = data.nanquantile(probs, dim=1, keepdim=True)
    expected = np.nanquantile(
        data.numpy(), probs, axis=1, keepdims=True, method="linear"
    )

    assert quant.shape == (2, 2, 1)
    np.testing.assert_allclose(quant.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_nanquantile_sequence_keepdim_no_dim():
    data = mt.tensor([[1.0, np.nan, 5.0], [2.0, 4.0, np.nan]])
    probs = [0.25, 0.75]
    quant = data.nanquantile(probs, keepdim=True)
    expected = np.nanquantile(data.numpy(), probs, keepdims=True, method="linear")

    assert quant.shape == (2, 1, 1)
    np.testing.assert_allclose(quant.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_functional_nanquantile_matches_numpy():
    data = mt.tensor([[0.0, np.nan, 20.0], [30.0, 40.0, np.nan]])
    probs = [0.1, 0.9]
    values = mt.functional.nanquantile(data, probs, dim=1)
    expected = np.nanquantile(data.numpy(), probs, axis=1, method="linear")

    np.testing.assert_allclose(values.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_nanquantile_all_nan_raises():
    data = mt.tensor([np.nan, np.nan])
    with pytest.raises(ValueError):
        data.nanquantile(0.5)


def test_quantile_sequence_interpolation_modes_match_numpy():
    data = mt.tensor([[3.0, 7.0, 1.0, 9.0], [4.0, 2.0, 8.0, 6.0]])
    probs = [0.25, 0.5, 0.75]

    for interpolation in ["lower", "higher", "nearest", "linear", "midpoint"]:
        quant = data.quantile(probs, dim=1, interpolation=interpolation)
        expected = np.quantile(data.numpy(), probs, axis=1, method=interpolation)

        np.testing.assert_allclose(quant.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_nanquantile_sequence_interpolation_modes_match_numpy():
    data = mt.tensor([[3.0, np.nan, 1.0, 9.0], [4.0, 2.0, np.nan, 6.0]])
    probs = [0.25, 0.5, 0.75]

    for interpolation in ["lower", "higher", "nearest", "linear", "midpoint"]:
        quant = data.nanquantile(probs, dim=1, interpolation=interpolation)
        expected = np.nanquantile(data.numpy(), probs, axis=1, method=interpolation)

        np.testing.assert_allclose(quant.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_quantile_nearest_tie_breaks_to_even_index():
    data = mt.tensor([1.0, 3.0, 9.0])

    quant = data.quantile(0.25, interpolation="nearest")
    expected = np.quantile(data.numpy(), 0.25, method="nearest")

    np.testing.assert_allclose(quant.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_nanquantile_nearest_tie_breaks_to_even_index():
    data = mt.tensor([1.0, np.nan, 3.0, 9.0])

    quant = data.nanquantile(0.75, interpolation="nearest")
    expected = np.nanquantile(data.numpy(), 0.75, method="nearest")

    np.testing.assert_allclose(quant.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_quantile_nearest_tie_break_even_with_two_values():
    data = mt.tensor([10.0, 20.0])

    quant = data.quantile(0.5, interpolation="nearest")
    expected = np.quantile(data.numpy(), 0.5, method="nearest")

    np.testing.assert_allclose(quant.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_nanquantile_nearest_tie_break_even_with_filtered_values():
    data = mt.tensor([10.0, np.nan, 20.0, 30.0, 40.0])

    quant = data.nanquantile(0.5, interpolation="nearest")
    expected = np.nanquantile(data.numpy(), 0.5, method="nearest")

    np.testing.assert_allclose(quant.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_quantile_nearest_endpoint_probabilities_match_numpy():
    data = mt.tensor([4.0, 1.0, 9.0, 2.0, 7.0])

    for q in (0.0, 1.0):
        quant = data.quantile(q, interpolation="nearest")
        expected = np.quantile(data.numpy(), q, method="nearest")

        np.testing.assert_allclose(quant.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_nanquantile_nearest_endpoint_probabilities_match_numpy():
    data = mt.tensor([4.0, np.nan, 1.0, 9.0, np.nan, 2.0, 7.0])

    for q in (0.0, 1.0):
        quant = data.nanquantile(q, interpolation="nearest")
        expected = np.nanquantile(data.numpy(), q, method="nearest")

        np.testing.assert_allclose(quant.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_nearest_quantile_singleton_is_stable_across_probabilities():
    data = mt.tensor([42.0])

    for q in (0.0, 0.25, 0.5, 0.75, 1.0):
        quant = data.quantile(q, interpolation="nearest")
        nanquant = data.nanquantile(q, interpolation="nearest")

        np.testing.assert_allclose(quant.numpy(), np.array(42.0), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            nanquant.numpy(), np.array(42.0), rtol=1e-6, atol=1e-6
        )


def test_nansum_nanmean_basic():
    data = np.array([[1.0, np.nan, 3.0], [np.nan, 5.0, 6.0]], dtype=np.float32)
    t = mt.Tensor(data)

    assert np.isclose(t.nansum().numpy(), np.nansum(data))
    assert np.isclose(t.nanmean().numpy(), np.nanmean(data))
    assert np.isclose(mt.nansum(t).numpy(), np.nansum(data))
    assert np.isclose(mt.nanmean(t).numpy(), np.nanmean(data))
    assert np.isclose(mt.numpy_compat.nansum(t).numpy(), np.nansum(data))
    assert np.isclose(mt.numpy_compat.nanmean(t).numpy(), np.nanmean(data))

    assert np.allclose(t.nansum(dim=0).numpy(), np.nansum(data, axis=0))
    assert np.allclose(
        t.nansum(dim=1, keepdim=True).numpy(), np.nansum(data, axis=1, keepdims=True)
    )
    assert np.allclose(
        mt.numpy_compat.nansum(t, axis=0).numpy(),
        np.nansum(data, axis=0),
    )

    assert np.allclose(t.nanmean(dim=0).numpy(), np.nanmean(data, axis=0))
    assert np.allclose(
        t.nanmean(dim=1, keepdim=True).numpy(), np.nanmean(data, axis=1, keepdims=True)
    )
    assert np.allclose(
        mt.numpy_compat.nanmean(t, axis=0).numpy(),
        np.nanmean(data, axis=0),
    )


def test_nansum_all_nan_returns_zero():
    t = mt.Tensor([np.nan, np.nan], dtype="float32")
    assert t.nansum().numpy() == 0.0


def test_nanmean_all_nan_returns_nan():
    t = mt.Tensor([np.nan, np.nan], dtype="float32")
    assert np.isnan(t.nanmean().numpy())

    data = np.array([[np.nan, np.nan], [1.0, np.nan]], dtype=np.float32)
    t = mt.Tensor(data)
    result = t.nanmean(dim=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = np.nanmean(data, axis=1)
    assert np.isnan(result.numpy()[0])
    assert np.isclose(result.numpy()[1], expected[1])


def test_nanmax_nanmin_values_and_indices():
    data = np.array([[np.nan, 2.0, 1.0], [4.0, np.nan, -1.0]], dtype=np.float32)
    t = mt.Tensor(data)

    max_vals, max_idx = t.nanmax(dim=1)
    expected_vals = np.nanmax(data, axis=1)
    expected_idx = np.nanargmax(data, axis=1)
    assert np.array_equal(max_vals.numpy(), expected_vals)
    assert np.array_equal(max_idx.numpy(), expected_idx.astype(np.int64))

    min_vals, min_idx = t.nanmin(dim=1)
    expected_vals = np.nanmin(data, axis=1)
    expected_idx = np.nanargmin(data, axis=1)
    assert np.array_equal(min_vals.numpy(), expected_vals)
    assert np.array_equal(min_idx.numpy(), expected_idx.astype(np.int64))

    fn_max_vals, fn_max_idx = mt.nanmax(t, dim=1)
    fn_min_vals, fn_min_idx = mt.nanmin(t, dim=1)
    expected_max_vals = np.nanmax(data, axis=1)
    expected_max_idx = np.nanargmax(data, axis=1)
    assert np.array_equal(fn_max_vals.numpy(), expected_max_vals)
    assert np.array_equal(fn_max_idx.numpy(), expected_max_idx.astype(np.int64))
    assert np.array_equal(fn_min_vals.numpy(), expected_vals)
    assert np.array_equal(fn_min_idx.numpy(), expected_idx.astype(np.int64))

    np_max = mt.numpy_compat.nanmax(t, axis=1).numpy()
    np_min = mt.numpy_compat.nanmin(t, axis=1).numpy()
    assert np.array_equal(np_max, expected_max_vals)
    assert np.array_equal(np_min, expected_vals)


def test_nanmax_nanmin_all_nan_slice():
    data = np.array([[np.nan, np.nan], [1.0, np.nan]], dtype=np.float32)
    t = mt.Tensor(data)

    max_vals, max_idx = t.nanmax(dim=1)
    min_vals, min_idx = t.nanmin(dim=1)

    assert np.isnan(max_vals.numpy()[0])
    assert np.isnan(min_vals.numpy()[0])
    assert max_idx.numpy()[0] == 0
    assert min_idx.numpy()[0] == 0
    assert np.isclose(max_vals.numpy()[1], 1.0)
    assert np.isclose(min_vals.numpy()[1], 1.0)


def test_nanmax_nanmin_all_nan_tensor():
    t = mt.Tensor([np.nan, np.nan], dtype="float32")
    assert np.isnan(t.nanmax().numpy())
    assert np.isnan(t.nanmin().numpy())


def test_cumsum_cumprod():
    t = Tensor.arange(1, 7, dtype="float32").reshape([2, 3])
    c0 = t.cumsum(0)
    np.testing.assert_array_equal(
        c0.numpy(), np.array([[1, 2, 3], [5, 7, 9]], dtype=np.float32)
    )
    c1 = t.cumsum(1)
    np.testing.assert_array_equal(
        c1.numpy(), np.array([[1, 3, 6], [4, 9, 15]], dtype=np.float32)
    )
    p0 = t.cumprod(0)
    np.testing.assert_array_equal(
        p0.numpy(), np.array([[1, 2, 3], [4, 10, 18]], dtype=np.float32)
    )
    p1 = t.cumprod(1)
    np.testing.assert_array_equal(
        p1.numpy(), np.array([[1, 2, 6], [4, 20, 120]], dtype=np.float32)
    )


def test_cumulative_invalid_axis():
    t = Tensor.arange(1, 7, dtype="float32").reshape([2, 3])
    with pytest.raises(IndexError):
        t.cumsum(2)
    with pytest.raises(IndexError):
        t.cumprod(2)


def test_functional_cumulative_reductions_support_negative_dims():
    x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    x = Tensor(x_np.tolist())

    np.testing.assert_allclose(F.cumsum(x, dim=-1).numpy(), np.cumsum(x_np, axis=-1))
    np.testing.assert_allclose(F.cumprod(x, dim=0).numpy(), np.cumprod(x_np, axis=0))


def test_sort_default_last_dim():
    x = mt.tensor([[3.0, 1.0, 2.0], [0.5, -1.0, 0.0]], dtype="float32")
    values, indices = x.sort()

    assert values.numpy().tolist() == [[1.0, 2.0, 3.0], [-1.0, 0.0, 0.5]]
    assert indices.numpy().tolist() == [[1, 2, 0], [1, 2, 0]]


def test_sort_along_dim_zero_int_tensor():
    x = mt.tensor([[3, 1], [2, 0], [5, -2]], dtype="int32")
    values, indices = x.sort(dim=0)

    assert values.numpy().tolist() == [[2, -2], [3, 0], [5, 1]]
    assert indices.numpy().tolist() == [[1, 2], [0, 1], [2, 0]]


def test_sort_descending_with_nan():
    x = mt.tensor([float("nan"), 3.0, 1.0], dtype="float32")
    values, indices = x.sort(descending=True)

    # NaNs should be placed first to align with PyTorch semantics
    assert math.isnan(values.numpy()[0])
    assert indices.numpy().tolist()[1:] == [1, 2]


def test_sort_stable_keeps_duplicate_order():
    x = mt.tensor([1.0, 2.0, 1.0, 1.0], dtype="float64")
    values, indices = x.sort(stable=True)

    assert values.numpy().tolist() == [1.0, 1.0, 1.0, 2.0]
    assert indices.numpy().tolist() == [0, 2, 3, 1]


def test_argsort_matches_sort_indices():
    x = mt.tensor([[3.0, 1.0], [2.5, -4.0]], dtype="float32")
    values, indices = x.sort(dim=1, descending=True)
    argsorted = x.argsort(dim=1, descending=True)

    assert values.numpy().tolist() == [[3.0, 1.0], [2.5, -4.0]]
    assert indices.numpy().tolist() == argsorted.numpy().tolist()


def test_sort_scalar_returns_zero_index():
    x = mt.tensor(5.0)
    values, indices = x.sort()

    assert pytest.approx(values.item()) == 5.0
    assert indices.item() == 0


def test_sort_scalar_invalid_dim_raises():
    x = mt.tensor(1.0)
    with pytest.raises(IndexError):
        x.sort(dim=1)


def test_top_level_sort_and_argsort_dispatch():
    x = mt.tensor([True, False, True], dtype="bool")
    values, indices = mt.sort(x, descending=True)
    args = mt.argsort(x, descending=True)

    assert values.numpy().tolist() == [True, True, False]
    assert indices.numpy().tolist() == args.numpy().tolist()


np = pytest.importorskip("numpy")


def test_topk_default_last_dim():
    x = mt.tensor([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], requires_grad=True)
    values, indices = x.topk(2)

    np.testing.assert_allclose(
        values.numpy(),
        np.array([[3.0, 2.0], [5.0, 4.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        indices.numpy(),
        np.array([[1, 2], [2, 0]], dtype=np.int64),
    )
    assert values.requires_grad is True
    assert indices.requires_grad is False
    assert indices.dtype == "int64"


def test_topk_smallest_unsorted():
    x = mt.tensor([1.0, -2.0, 3.5, 0.0], dtype="float32")
    values, indices = x.topk(2, largest=False, sorted=False)

    pairs = sorted(zip(indices.numpy().tolist(), values.numpy().tolist()))
    assert pairs == [(1, -2.0), (3, 0.0)]


def test_topk_with_dim_argument():
    x = mt.tensor([[1, 4, 2], [3, -1, 0]], dtype="float32")
    values, indices = x.topk(1, dim=1, largest=False)

    np.testing.assert_array_equal(
        values.numpy(), np.array([[1.0], [-1.0]], dtype=np.float32)
    )
    np.testing.assert_array_equal(indices.numpy(), np.array([[0], [1]], dtype=np.int64))


def test_topk_zero_k():
    x = mt.tensor([1.0, 2.0, 3.0])
    values, indices = x.topk(0)

    assert values.shape == (0,)
    assert indices.shape == (0,)
    assert values.numpy().size == 0
    assert indices.numpy().size == 0


def test_topk_out_of_range():
    x = mt.tensor([[1.0, 2.0, 3.0]])
    with pytest.raises(RuntimeError, match="selected index k out of range"):
        x.topk(4, dim=1)
