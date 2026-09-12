# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A batch of products crosses to the BLAS once, and gives the same answer.

`test_gemm_delegation.py` covers the single product. This covers the stacked
one, which reaches the provider by a different route and for a different
reason: a single product is offered because a tuned BLAS beats the engine's
kernel on it, and a *batch* is offered because handing matrices over one at a
time would spend the crossing per matrix -- at a batch of 256 more than the
whole product costs to compute.

What has to hold is that the route makes no difference to the answer. The
provider is asked before the batch axis is split across the thread pool, so the
same call can end up on either of two quite different code paths depending only
on a threshold, and the tests below run both and compare.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

dispatch = mt._core.dispatch

# Out of reach of any real product, so nothing is delegated.
_NEVER = 2**62


@pytest.fixture
def restore_thresholds():
    """Put the delegation boundary back however the test leaves it."""
    previous = dispatch.gemm_thresholds()
    yield
    dispatch.set_gemm_thresholds(*previous)


def _both_paths(build):
    """`build()` run once with delegation off and once with it forced on."""
    previous = dispatch.gemm_thresholds()
    try:
        dispatch.set_gemm_thresholds(_NEVER, _NEVER)
        native = np.asarray(build()).copy()
        dispatch.set_gemm_thresholds(0, 0)
        delegated = np.asarray(build()).copy()
    finally:
        dispatch.set_gemm_thresholds(*previous)
    return native, delegated


def _t(values):
    """A tensor of the array's own dtype.

    `mt.Tensor(array)` takes the global default dtype rather than the array's,
    so a float64 reference would silently be compared against a float32 answer
    and the test would be measuring rounding.
    """
    values = np.ascontiguousarray(values)
    return mt.Tensor(values, dtype=str(values.dtype))


class TestTheTwoPathsAgree:
    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    @pytest.mark.parametrize(
        "batch,m,k,n",
        [
            (1, 8, 8, 8),
            (2, 5, 7, 3),
            # Straddles the thread count: below it the engine walks the batch
            # serially, at or above it the batch axis fills the pool. Both are
            # branches the provider has to be offered ahead of.
            (3, 16, 16, 16),
            (8, 16, 16, 16),
            (64, 4, 4, 4),
            (5, 1, 32, 1),
        ],
    )
    def test_batched_matmul(self, dtype, batch, m, k, n):
        rng = np.random.default_rng(0)
        a = rng.standard_normal((batch, m, k)).astype(dtype)
        b = rng.standard_normal((batch, k, n)).astype(dtype)
        native, delegated = _both_paths(lambda: mt.bmm(_t(a), _t(b)))

        tol = 1e-4 if dtype == "float32" else 1e-11
        np.testing.assert_allclose(delegated, native, rtol=tol, atol=tol)
        np.testing.assert_allclose(delegated, a @ b, rtol=tol, atol=tol)

    def test_broadcast_batch_dimensions(self):
        rng = np.random.default_rng(1)
        a = rng.standard_normal((6, 3, 5))
        b = rng.standard_normal((5, 2))
        native, delegated = _both_paths(lambda: mt.matmul(_t(a), _t(b)))
        np.testing.assert_allclose(delegated, native, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(delegated, a @ b, rtol=1e-11, atol=1e-11)

    def test_a_batch_of_one_is_the_plain_product(self):
        rng = np.random.default_rng(2)
        a = rng.standard_normal((1, 12, 9))
        b = rng.standard_normal((1, 9, 7))
        native, delegated = _both_paths(lambda: mt.bmm(_t(a), _t(b)))
        np.testing.assert_allclose(delegated, native, rtol=1e-11, atol=1e-11)

    def test_every_matrix_in_the_batch_is_its_own_product(self):
        # A batch where each element differs, so a kernel that broadcast one
        # matrix over the batch -- or offset into the wrong one -- is caught.
        rng = np.random.default_rng(3)
        a = rng.standard_normal((7, 4, 4))
        b = rng.standard_normal((7, 4, 4))
        _, delegated = _both_paths(lambda: mt.bmm(_t(a), _t(b)))
        for index in range(7):
            np.testing.assert_allclose(
                delegated[index], a[index] @ b[index], rtol=1e-11, atol=1e-11
            )


class TestGradientsAgree:
    @pytest.mark.parametrize("min_flops,min_k", [(0, 0), (_NEVER, _NEVER)])
    def test_batched_backward(self, min_flops, min_k, restore_thresholds):
        dispatch.set_gemm_thresholds(min_flops, min_k)
        rng = np.random.default_rng(4)
        a_host = rng.standard_normal((5, 3, 4))
        b_host = rng.standard_normal((5, 4, 2))
        a = mt.Tensor(np.ascontiguousarray(a_host), dtype="float64", requires_grad=True)
        b = mt.Tensor(np.ascontiguousarray(b_host), dtype="float64", requires_grad=True)

        mt.bmm(a, b).sum().backward()

        ones = np.ones((5, 3, 2))
        np.testing.assert_allclose(
            np.asarray(a.grad), ones @ b_host.transpose(0, 2, 1), rtol=1e-11
        )
        np.testing.assert_allclose(
            np.asarray(b.grad), a_host.transpose(0, 2, 1) @ ones, rtol=1e-11
        )
        mt.clear_autograd_graph()


class TestIntegersStayExact:
    def test_integer_batches_never_delegate(self, restore_thresholds):
        # The provider only has float entry points, so an integer product must
        # come back exact however the thresholds are set.
        dispatch.set_gemm_thresholds(0, 0)
        a = np.arange(2 * 3 * 3, dtype=np.int64).reshape(2, 3, 3)
        product = np.asarray(mt.bmm(_t(a), _t(a)))
        np.testing.assert_array_equal(product, a @ a)


class TestTheKnobs:
    def test_thresholds_round_trip(self, restore_thresholds):
        before = dispatch.gemm_thresholds()
        previous = dispatch.set_gemm_thresholds(1, 2)
        assert previous == before
        assert dispatch.gemm_thresholds() == (1, 2)

    def test_the_defaults_are_what_the_module_documents(self, restore_thresholds):
        dispatch.set_gemm_thresholds(dispatch.DEFAULT_MIN_FLOPS, dispatch.DEFAULT_MIN_K)
        assert dispatch.gemm_thresholds() == (
            dispatch.DEFAULT_MIN_FLOPS,
            dispatch.DEFAULT_MIN_K,
        )

    def test_a_provider_is_installed(self):
        # NumPy is a hard dependency, so the provider should always be up. If
        # this fails the delegated path is silently not being exercised
        # anywhere, including in the tests above.
        assert dispatch.gemm_provider_installed() is True
