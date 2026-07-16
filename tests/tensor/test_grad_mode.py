# Copyright (c) 2026 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for gradient-recording mode (no_grad / enable_grad) and related
per-tensor gradient semantics."""

import pytest

import minitensor as mt


class TestGradMode:
    def test_grad_enabled_by_default(self):
        assert mt.is_grad_enabled()

    def test_no_grad_disables_and_restores(self):
        assert mt.is_grad_enabled()
        with mt.no_grad():
            assert not mt.is_grad_enabled()
        assert mt.is_grad_enabled()

    def test_no_grad_restores_on_exception(self):
        with pytest.raises(RuntimeError):
            with mt.no_grad():
                assert not mt.is_grad_enabled()
                raise RuntimeError("boom")
        assert mt.is_grad_enabled()

    def test_nested_enable_grad(self):
        with mt.no_grad():
            assert not mt.is_grad_enabled()
            with mt.enable_grad():
                assert mt.is_grad_enabled()
            assert not mt.is_grad_enabled()
        assert mt.is_grad_enabled()

    def test_set_grad_enabled_returns_previous(self):
        prev = mt.set_grad_enabled(False)
        try:
            assert prev is True
            assert not mt.is_grad_enabled()
        finally:
            mt.set_grad_enabled(True)
        assert mt.is_grad_enabled()

    def test_op_results_inside_no_grad_are_detached_leaves(self):
        x = mt.randn(3, 3)
        x.requires_grad_(True)
        with mt.no_grad():
            y = x * 2.0 + 1.0
            assert not y.requires_grad
        # Backward on a detached result must fail like PyTorch.
        with pytest.raises(RuntimeError):
            y.sum().backward()

    def test_new_tensors_inside_no_grad_do_not_require_grad(self):
        with mt.no_grad():
            t = mt.randn(2, 2)
            t2 = t + 1.0
            assert not t2.requires_grad

    def test_explicit_opt_in_inside_no_grad(self):
        with mt.no_grad():
            t = mt.randn(2, 2)
            t.requires_grad_(True)
            assert t.requires_grad

    def test_grad_flows_normally_after_no_grad_block(self):
        x = mt.randn(2, 2)
        x.requires_grad_(True)
        with mt.no_grad():
            frozen = x * 3.0
            assert not frozen.requires_grad
        y = (x * x).sum()
        y.backward()
        grad = mt.get_gradient(x)
        assert grad is not None
        assert grad.shape == x.shape


class TestRequiresGradChaining:
    def test_requires_grad_returns_self(self):
        x = mt.randn(2, 2).requires_grad_(True)
        assert x is not None
        assert x.requires_grad

        y = x.requires_grad_(False)
        assert y is not None
        assert not y.requires_grad


class TestPerTensorZeroGrad:
    def test_zero_grad_only_clears_own_gradient(self):
        a = mt.randn(2, 2)
        a.requires_grad_(True)
        b = mt.randn(2, 2)
        b.requires_grad_(True)

        loss = (a * b).sum()
        loss.backward()

        assert mt.get_gradient(a) is not None
        assert mt.get_gradient(b) is not None

        a.zero_grad(set_to_none=True)

        assert mt.get_gradient(a) is None
        # b's gradient must survive a.zero_grad().
        assert mt.get_gradient(b) is not None

        mt.clear_autograd_graph()
