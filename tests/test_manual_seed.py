# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import minitensor as mt


def test_manual_seed_makes_rand_deterministic():
    mt.manual_seed(123)
    first = mt.rand(2, 3).numpy()
    mt.manual_seed(123)
    second = mt.rand(2, 3).numpy()
    np.testing.assert_array_equal(first, second)


def test_manual_seed_makes_randn_deterministic():
    mt.manual_seed(321)
    first = mt.randn(2, 3).numpy()
    mt.manual_seed(321)
    second = mt.randn(2, 3).numpy()
    np.testing.assert_array_equal(first, second)


def test_manual_seed_makes_rand_like_deterministic():
    base = mt.ones((4, 2), dtype="float32")
    mt.manual_seed(111)
    first = mt.rand_like(base).numpy()
    mt.manual_seed(111)
    second = mt.rand_like(base).numpy()
    np.testing.assert_array_equal(first, second)


def test_manual_seed_makes_randn_like_deterministic():
    base = mt.ones((3, 5), dtype="float64")
    mt.manual_seed(222)
    first = mt.randn_like(base).numpy()
    mt.manual_seed(222)
    second = mt.randn_like(base).numpy()
    np.testing.assert_array_equal(first, second)


def test_manual_seed_affects_module_initialization():
    mt.manual_seed(999)
    layer_a = mt.nn.DenseLayer(4, 3)
    mt.manual_seed(999)
    layer_b = mt.nn.DenseLayer(4, 3)

    params_a = [param.numpy() for param in layer_a.parameters()]
    params_b = [param.numpy() for param in layer_b.parameters()]

    for left, right in zip(params_a, params_b):
        np.testing.assert_array_equal(left, right)


def test_manual_seed_controls_dropout_mask():
    x = mt.ones(10, dtype="float32")

    mt.manual_seed(42)
    dropout_a = mt.nn.Dropout(0.5)
    out_a = dropout_a(x).numpy()

    mt.manual_seed(42)
    dropout_b = mt.nn.Dropout(0.5)
    out_b = dropout_b(x).numpy()

    np.testing.assert_array_equal(out_a, out_b)


def test_manual_seed_makes_randint_deterministic():
    mt.manual_seed(2025)
    first = mt.randint(0, 10, 6).numpy()
    mt.manual_seed(2025)
    second = mt.randint(0, 10, 6).numpy()
    np.testing.assert_array_equal(first, second)


def test_manual_seed_makes_randperm_deterministic():
    mt.manual_seed(77)
    first = mt.randperm(9).numpy()
    mt.manual_seed(77)
    second = mt.randperm(9).numpy()
    np.testing.assert_array_equal(first, second)


def test_manual_seed_makes_randint_like_deterministic():
    base = mt.ones((2, 4), dtype="float32")
    mt.manual_seed(333)
    first = mt.randint_like(base, 0, 5).numpy()
    mt.manual_seed(333)
    second = mt.randint_like(base, 0, 5).numpy()
    np.testing.assert_array_equal(first, second)
