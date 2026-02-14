# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

import minitensor as mt


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
