# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""What the Tensor type offers Python, and what it should not.

`#[pymethods]` decides what Python sees; the Rust visibility modifier does not.
A `pub(crate) fn` inside such a block is `pub(crate)` to the rest of the crate
*and* a public method on the class, which is how six helpers named
`eq_from_py`, `ne_from_py`, `lt_from_py`, `le_from_py`, `gt_from_py` and
`ge_from_py` came to stand beside `eq`, `ne`, `lt`, `le`, `gt` and `ge` doing
exactly the same thing. They were found by asking which public callables no
test names -- 6 of the 8 the whole surface turned up.

The suffix guard is the general form. The rest of this file checks the
comparisons still behave, since moving them is the kind of change that is meant
to be invisible and has to be shown to be.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

# Suffixes that name an implementation detail. A method carrying one is a
# helper that escaped into the public surface rather than an offer to callers.
INTERNAL_SUFFIXES = ("_from_py", "_from_python", "_inner", "_impl", "_unchecked")


def _public_methods(obj):
    return {n for n in dir(obj) if not n.startswith("_")}


def test_no_internal_helper_escaped_onto_the_tensor_type():
    escaped = sorted(
        name for name in _public_methods(mt.Tensor) if name.endswith(INTERNAL_SUFFIXES)
    )
    assert not escaped, (
        f"these look like internal helpers exported by being inside a "
        f"#[pymethods] block: {escaped}. Move them to a plain `impl` block; "
        f"`pub(crate)` alone does not hide them from Python."
    )


def test_the_six_comparison_helpers_are_gone_by_name():
    """Named explicitly, so the suffix rule cannot be loosened without notice."""
    for stem in ("eq", "ne", "lt", "le", "gt", "ge"):
        assert not hasattr(mt.Tensor, f"{stem}_from_py"), f"{stem}_from_py is back"


@pytest.mark.parametrize(
    "name,operator",
    [
        ("eq", lambda a, b: a == b),
        ("ne", lambda a, b: a != b),
        ("lt", lambda a, b: a < b),
        ("le", lambda a, b: a <= b),
        ("gt", lambda a, b: a > b),
        ("ge", lambda a, b: a >= b),
    ],
)
def test_the_named_comparison_and_the_operator_agree(name, operator):
    tensor = mt.Tensor([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(
        np.asarray(getattr(tensor, name)(2.0)), np.asarray(operator(tensor, 2.0))
    )


@pytest.mark.parametrize(
    "name,expected",
    [
        ("eq", [False, True, False]),
        ("ne", [True, False, True]),
        ("lt", [True, False, False]),
        ("le", [True, True, False]),
        ("gt", [False, False, True]),
        ("ge", [False, True, True]),
    ],
)
def test_each_comparison_against_a_scalar(name, expected):
    tensor = mt.Tensor([1.0, 2.0, 3.0])
    result = getattr(tensor, name)(2.0)
    assert result.dtype == "bool"
    np.testing.assert_array_equal(np.asarray(result), np.array(expected))


@pytest.mark.parametrize(
    "dtype,itemsize",
    [("float64", 8), ("float32", 4), ("int64", 8), ("int32", 4), ("bool", 1)],
)
def test_memory_usage_bytes_counts_the_buffer(dtype, itemsize):
    """One of the two public callables left that no test named."""
    tensor = mt.Tensor(np.zeros((3, 5)), dtype=dtype)
    assert tensor.memory_usage_bytes() == 15 * itemsize


def test_memory_usage_bytes_of_an_empty_tensor_is_zero():
    assert mt.Tensor(np.zeros((0, 4)), dtype="float32").memory_usage_bytes() == 0


def test_the_scheduler_base_is_a_base_and_not_a_scheduler():
    """`optim.LRScheduler` is the other unnamed callable: abstract on purpose."""
    with pytest.raises(TypeError):
        mt.optim.LRScheduler()
