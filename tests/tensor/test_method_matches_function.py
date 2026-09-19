# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`t.relu()` and `minitensor.relu(t)` have to be the same operation.

Most of the tensor surface is reachable two ways, and the two spellings are
separate wrappers over one kernel. A wrapper is exactly where a default goes
astray -- one side passing a dimension, a flag or an epsilon the other leaves
alone -- and neither side's own tests would notice, because each is correct
about what it does.

The sweep walks `dir(minitensor)` against `dir(Tensor)` rather than listing
names, so an operation added later is covered the day it appears.

Two things it deliberately does not treat as disagreement:

An operation whose input is out of its domain returns NaN rather than raising,
and NaN never equals NaN. `acos`, `acosh`, `asin`, `atanh`, `erfinv` and
`logit` all land here on a general input, so the comparison matches the NaN
masks and then compares the finite elements -- an earlier draft that subtracted
directly reported all six as failures when both spellings agreed exactly.

And a handful of names mean different things on the two sides: `minitensor.zeros`
builds a tensor from a shape while `Tensor.zeros` is a method on an existing
one. Those are not two spellings of one operation, and they are named rather
than skipped on exception, so a real breakage cannot hide among them.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

# Same name on both sides, different operation. `minitensor.<name>` builds a
# tensor from a shape or a spec; the method does something else entirely.
NOT_PAIRS = {
    "cpu",
    "empty",
    "he_normal",
    "he_uniform",
    "lecun_normal",
    "lecun_uniform",
    "ones",
    "rand",
    "randn",
    "truncated_normal",
    "uniform",
    "xavier_normal",
    "xavier_uniform",
    "zeros",
}

# Below this the sweep has stopped sweeping -- a rename or a packaging change
# emptied it -- and every assertion would pass by checking nothing.
MINIMUM_PAIRS = 120


def _subject():
    rng = np.random.default_rng(1)
    return mt.Tensor(np.abs(rng.standard_normal((3, 4))) + 0.3, dtype="float64")


def _no_argument_pairs():
    """Names callable with no arguments as a method and with the tensor alone."""
    tensor = _subject()
    names = {n for n in dir(mt) if not n.startswith("_")} & {
        n for n in dir(type(tensor)) if not n.startswith("_")
    }
    found = []
    for name in sorted(names - NOT_PAIRS):
        function, method = getattr(mt, name), getattr(tensor, name, None)
        if not callable(function) or not callable(method):
            continue
        try:
            method()
        except Exception:
            continue  # needs arguments; the no-argument sweep cannot reach it
        found.append(name)
    return found


PAIRS = _no_argument_pairs()


@pytest.fixture(autouse=True)
def _clean_graph():
    yield
    mt.clear_autograd_graph()


def test_the_sweep_found_the_surface():
    assert len(PAIRS) >= MINIMUM_PAIRS, (
        f"only {len(PAIRS)} method/function pairs discovered, expected at least "
        f"{MINIMUM_PAIRS}; the sweep is no longer reaching the tensor surface"
    )


def _compare(label, from_method, from_function):
    """One output, or one position of a factorisation that returns several."""
    a, b = np.asarray(from_method), np.asarray(from_function)

    assert (
        a.shape == b.shape
    ), f"{label}: {a.shape} as a method, {b.shape} as a function"
    assert (
        a.dtype == b.dtype
    ), f"{label}: {a.dtype} as a method, {b.dtype} as a function"

    if a.dtype == bool:
        assert np.array_equal(a, b), label
        return

    np.testing.assert_array_equal(
        np.isnan(a),
        np.isnan(b),
        err_msg=f"{label}: the two spellings put NaN in different places",
    )
    finite = ~np.isnan(a)
    np.testing.assert_array_equal(a[finite], b[finite], err_msg=label)


@pytest.mark.parametrize("name", PAIRS)
def test_the_method_and_the_free_function_agree(name):
    tensor = _subject()
    from_method = getattr(tensor, name)()
    from_function = getattr(mt, name)(tensor)

    # `qr`, `svd` and the other factorisations hand back several tensors, and
    # how many is itself part of the contract -- a `full_matrices` or `some`
    # default that differed between the two spellings would show up here as a
    # different arity or a different shape.
    method_tuple = isinstance(from_method, tuple)
    assert method_tuple == isinstance(
        from_function, tuple
    ), f"{name}: one spelling returns a tuple and the other does not"

    if not method_tuple:
        _compare(name, from_method, from_function)
        return

    assert len(from_method) == len(from_function), (
        f"{name}: {len(from_method)} outputs as a method, "
        f"{len(from_function)} as a function"
    )
    for index, (left, right) in enumerate(zip(from_method, from_function)):
        _compare(f"{name}[{index}]", left, right)


def test_the_names_excluded_are_still_not_pairs():
    """If one of these becomes a real pair, it should join the sweep.

    Excluding a name permanently on the strength of one reading is how a sweep
    quietly shrinks. Each is checked to still behave as the two different
    operations the exclusion claims.
    """
    tensor = _subject()
    still_excluded = []
    for name in sorted(NOT_PAIRS):
        function = getattr(mt, name, None)
        if function is None or not callable(function):
            continue
        try:
            function(tensor)
        except Exception:
            still_excluded.append(name)

    assert set(still_excluded) == {
        n for n in NOT_PAIRS if callable(getattr(mt, n, None))
    }, (
        "a name on the exclusion list now accepts a tensor; if it has become a "
        "real pair, move it into the sweep"
    )
