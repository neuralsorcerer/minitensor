# Copyright (c) 2026 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import numpy as np

import minitensor as mt


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
