# Copyright (c) Soumyadip Sarkar.
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
        # Backward on a detached result must fail.
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


class TestGradModeObjects:
    """A grad mode object can be entered more than once and used as a
    decorator."""

    def test_one_object_entered_twice_restores_the_mode(self):
        """Each object kept one saved mode, so the inner entry overwrote the
        outer one's and the outer exit restored nothing: recording stayed off
        for the rest of the thread."""
        guard = mt.no_grad()
        with guard:
            with guard:
                assert not mt.is_grad_enabled()
            assert not mt.is_grad_enabled()
        assert mt.is_grad_enabled()

    def test_no_grad_decorates_a_function(self):
        x = mt.Tensor([1.0, 2.0], requires_grad=True)

        @mt.no_grad()
        def predict(t):
            """Scaled."""
            return t * 2

        assert not predict(x).requires_grad
        assert mt.is_grad_enabled()
        assert predict.__name__ == "predict"
        assert predict.__doc__ == "Scaled."

    def test_a_decorated_method_binds_self(self):
        x = mt.Tensor([1.0], requires_grad=True)

        class Model:
            factor = 3.0

            @mt.no_grad()
            def run(self, t):
                return t * self.factor

        assert not Model().run(x).requires_grad

    def test_the_mode_is_restored_after_an_exception(self):
        @mt.no_grad()
        def fail():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            fail()
        assert mt.is_grad_enabled()

    def test_a_decorated_function_can_recurse(self):
        x = mt.Tensor([1.0], requires_grad=True)

        @mt.no_grad()
        def depth(n):
            return x * 1.0 if n == 0 else depth(n - 1)

        assert not depth(4).requires_grad
        assert mt.is_grad_enabled()

    def test_enable_grad_decorates_inside_no_grad(self):
        x = mt.Tensor([1.0], requires_grad=True)

        @mt.enable_grad()
        def tracked(t):
            return t * 2

        with mt.no_grad():
            assert tracked(x).requires_grad
            assert not mt.is_grad_enabled()

    def test_decorating_something_not_callable_is_refused(self):
        with pytest.raises(TypeError, match="decorates a function"):
            mt.no_grad()(5)
