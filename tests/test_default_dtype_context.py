# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import minitensor as mt


def test_default_dtype_context_restores():
    original = mt.get_default_dtype()
    with mt.default_dtype("float64"):
        assert mt.get_default_dtype() == "float64"
        t = mt.Tensor([1.0, 2.0])
        assert t.dtype == "float64"
    assert mt.get_default_dtype() == original


def test_default_dtype_context_restores_on_exception():
    original = mt.get_default_dtype()
    with pytest.raises(RuntimeError):
        with mt.default_dtype("float64"):
            raise RuntimeError("boom")
    assert mt.get_default_dtype() == original


def test_default_dtype_context_invalid_dtype():
    original = mt.get_default_dtype()
    with pytest.raises(ValueError):
        with mt.default_dtype("not-a-real-dtype"):
            pass
    assert mt.get_default_dtype() == original
