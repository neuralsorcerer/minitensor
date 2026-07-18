# Copyright (c) 2026 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Floor division (`//`), remainder (`%`), and bitwise NOT (`~`).

Python semantics throughout: `//` rounds toward negative infinity, `%` takes
the divisor's sign, and `a == (a // b) * b + a % b` holds for every dtype.
"""

import numpy as np
import pytest

import minitensor as mt

FLOAT_A = np.array([[7.5, -7.5, 3.0], [-3.0, 0.0, 5.25]], dtype=np.float32)
FLOAT_B = np.array([2.0, -2.0, 1.5], dtype=np.float32)
INT_A = np.array([7, -7, 8, -8, 0, 2**40 + 3], dtype=np.int64)
INT_B = np.array([2, 2, -3, -3, 5, 7], dtype=np.int64)


def test_float_floordiv_and_mod_match_numpy():
    a, b = mt.from_numpy(FLOAT_A.copy()), mt.from_numpy(FLOAT_B.copy())
    np.testing.assert_allclose((a // b).numpy(), FLOAT_A // FLOAT_B, rtol=1e-6)
    np.testing.assert_allclose((a % b).numpy(), FLOAT_A % FLOAT_B, rtol=1e-6)
    np.testing.assert_allclose((a // 2.0).numpy(), FLOAT_A // 2.0, rtol=1e-6)
    np.testing.assert_allclose((10.0 // b).numpy(), 10.0 // FLOAT_B, rtol=1e-6)
    np.testing.assert_allclose((10.0 % b).numpy(), 10.0 % FLOAT_B, rtol=1e-6)


def test_floordiv_mod_identity():
    q = (mt.from_numpy(FLOAT_A.copy()) // mt.from_numpy(FLOAT_B.copy())).numpy()
    r = (mt.from_numpy(FLOAT_A.copy()) % mt.from_numpy(FLOAT_B.copy())).numpy()
    np.testing.assert_allclose(q * FLOAT_B + r, FLOAT_A, rtol=1e-5)

    q = (mt.from_numpy(INT_A) // mt.from_numpy(INT_B)).numpy()
    r = (mt.from_numpy(INT_A) % mt.from_numpy(INT_B)).numpy()
    np.testing.assert_array_equal(q * INT_B + r, INT_A)


def test_int_floordiv_and_mod_exact():
    # Integer results stay integral (no float round-trip), so values beyond
    # float32's 2^24 mantissa stay exact.
    q = mt.from_numpy(INT_A) // mt.from_numpy(INT_B)
    r = mt.from_numpy(INT_A) % mt.from_numpy(INT_B)
    assert "int64" in str(q.dtype) and "int64" in str(r.dtype)
    np.testing.assert_array_equal(q.numpy(), INT_A // INT_B)
    np.testing.assert_array_equal(r.numpy(), INT_A % INT_B)

    a32 = np.array([7, -7, 8, -8], dtype=np.int32)
    b32 = np.array([2, 2, -3, -3], dtype=np.int32)
    np.testing.assert_array_equal(
        (mt.from_numpy(a32) // mt.from_numpy(b32)).numpy(), a32 // b32
    )
    np.testing.assert_array_equal(
        (mt.from_numpy(a32) % mt.from_numpy(b32)).numpy(), a32 % b32
    )


def test_int_min_over_minus_one_wraps():
    lo = np.iinfo(np.int32).min
    out = mt.from_numpy(np.array([lo], dtype=np.int32)) // mt.from_numpy(
        np.array([-1], dtype=np.int32)
    )
    assert out.numpy()[0] == lo  # wraps instead of panicking


def test_mixed_int_float_promotes():
    out = mt.from_numpy(INT_A) // 2.5
    np.testing.assert_allclose(out.numpy(), INT_A // 2.5, rtol=1e-6)


def test_integer_zero_divisor_raises():
    z = mt.from_numpy(np.array([1, 0, 1, 1, 1, 1], dtype=np.int64))
    with pytest.raises(Exception):
        mt.from_numpy(INT_A) // z
    with pytest.raises(Exception):
        mt.from_numpy(INT_A) % z


def test_float_zero_divisor_is_nan():
    out = (mt.from_numpy(np.array([1.5, -1.5], dtype=np.float32)) % 0.0).numpy()
    assert np.isnan(out).all()


def test_bool_operands_rejected():
    with pytest.raises(Exception):
        mt.from_numpy(np.array([True])) // 1
    with pytest.raises(Exception):
        mt.from_numpy(np.array([True])) % 1


def test_broadcasting_floordiv_mod():
    a = np.arange(12, dtype=np.float32).reshape(3, 4) - 5.0
    b = np.array([[2.0], [3.0], [-2.0]], dtype=np.float32)
    np.testing.assert_allclose(
        (mt.from_numpy(a) // mt.from_numpy(b)).numpy(), a // b, rtol=1e-6
    )
    np.testing.assert_allclose(
        (mt.from_numpy(a) % mt.from_numpy(b)).numpy(), a % b, rtol=1e-6
    )


def test_invert_operator():
    bb = np.array([True, False, True])
    np.testing.assert_array_equal((~mt.from_numpy(bb)).numpy(), ~bb)
    ii = np.array([0, 1, -5, 7], dtype=np.int64)
    np.testing.assert_array_equal((~mt.from_numpy(ii)).numpy(), ~ii)
    i3 = np.array([0, 255, -1], dtype=np.int32)
    np.testing.assert_array_equal((~mt.from_numpy(i3)).numpy(), ~i3)
    with pytest.raises(Exception):
        ~mt.from_numpy(FLOAT_A)


def test_named_methods_match_operators():
    a, b = mt.from_numpy(FLOAT_A.copy()), mt.from_numpy(FLOAT_B.copy())
    np.testing.assert_allclose(a.floor_divide(b).numpy(), FLOAT_A // FLOAT_B, rtol=1e-6)
    np.testing.assert_allclose(a.remainder(b).numpy(), FLOAT_A % FLOAT_B, rtol=1e-6)
    bb = np.array([True, False])
    np.testing.assert_array_equal(mt.from_numpy(bb).bitwise_not().numpy(), ~bb)


def test_remainder_gradients_match_finite_difference():
    # rem(x, y) = x - floor(x/y) * y => d/dx = 1, d/dy = -floor(x/y).
    # Inputs are kept away from the floor jumps so the derivative is defined.
    divisor = mt.from_numpy(np.array([[2.3, -1.7, 3.1]], dtype=np.float32))
    src = np.array([[5.05, -4.45, 7.3], [1.15, -0.85, 2.6]], dtype=np.float32)

    def analytic(fn, arr):
        x = mt.from_numpy(arr.copy())
        x.requires_grad_(True)
        fn(x).sum().backward()
        g = mt.get_gradient(x).numpy().astype(np.float64)
        mt.clear_autograd_graph()
        return g

    def numeric(fn, arr, eps=1e-3):
        grad = np.zeros(arr.shape, dtype=np.float64)
        flat = arr.reshape(-1).astype(np.float64)
        for i in range(flat.size):
            p = flat.copy()
            p[i] += eps
            m = flat.copy()
            m[i] -= eps
            fp = float(
                fn(mt.from_numpy(p.reshape(arr.shape).astype(np.float32))).sum().numpy()
            )
            fm = float(
                fn(mt.from_numpy(m.reshape(arr.shape).astype(np.float32))).sum().numpy()
            )
            grad.reshape(-1)[i] = (fp - fm) / (2 * eps)
        return grad

    np.testing.assert_allclose(
        analytic(lambda x: x % divisor, src),
        numeric(lambda x: x % divisor, src),
        rtol=3e-2,
        atol=3e-2,
    )

    dividend = mt.from_numpy(src.copy())
    src_div = np.array([[2.3, -1.7, 3.1]], dtype=np.float32)
    np.testing.assert_allclose(
        analytic(lambda y: dividend % y, src_div),
        numeric(lambda y: dividend % y, src_div),
        rtol=3e-2,
        atol=3e-2,
    )


def test_floor_divide_is_not_differentiable():
    x = mt.from_numpy(FLOAT_A.copy())
    x.requires_grad_(True)
    assert (x // 2.0).requires_grad is False
