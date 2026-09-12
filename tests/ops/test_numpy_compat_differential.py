# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Every `numpy_compat` export is checked against NumPy itself.

The module's whole promise is that these names behave like their NumPy
counterparts, which makes NumPy the obvious oracle -- yet 17 of the 29 exports
had no test at all: `sum`, `mean`, `max`, `min`, `prod`, `var`, `tensor_std`,
`dot`, `matmul`, `cross`, `hstack`, `vstack`, `hsplit`, `vsplit`, `asarray`,
`allclose` and `array_equal`. All of them turned out correct; the gap was in
the checking, not the code.

The completeness test at the bottom keeps that from recurring: a new export
without a case fails rather than passing unnoticed.
"""

import numpy as np
import pytest

import minitensor as mt

nc = mt.numpy_compat

_RNG = np.random.default_rng(0)
A = _RNG.standard_normal((3, 4))
B = _RNG.standard_normal((3, 4))
V = _RNG.standard_normal(3)
W = _RNG.standard_normal(3)
M43 = _RNG.standard_normal((4, 3))


def T(array):
    return mt.from_numpy(array)


def _value(result):
    return np.asarray(result.numpy() if hasattr(result, "numpy") else result)


# (name, callable producing the minitensor result, callable producing NumPy's)
CASES = [
    ("sum", lambda: nc.sum(T(A)), lambda: np.sum(A)),
    ("sum-axis", lambda: nc.sum(T(A), 0), lambda: np.sum(A, axis=0)),
    ("mean", lambda: nc.mean(T(A)), lambda: np.mean(A)),
    ("mean-axis", lambda: nc.mean(T(A), 1), lambda: np.mean(A, axis=1)),
    ("prod", lambda: nc.prod(T(A)), lambda: np.prod(A)),
    ("max", lambda: nc.max(T(A)), lambda: np.max(A)),
    ("min", lambda: nc.min(T(A)), lambda: np.min(A)),
    ("var", lambda: nc.var(T(A)), lambda: np.var(A)),
    ("tensor_std", lambda: nc.tensor_std(T(A)), lambda: np.std(A)),
    ("nansum", lambda: nc.nansum(T(A)), lambda: np.nansum(A)),
    ("nanmean", lambda: nc.nanmean(T(A)), lambda: np.nanmean(A)),
    ("nanmax", lambda: nc.nanmax(T(A)), lambda: np.nanmax(A)),
    ("nanmin", lambda: nc.nanmin(T(A)), lambda: np.nanmin(A)),
    ("dot", lambda: nc.dot(T(V), T(W)), lambda: np.dot(V, W)),
    ("matmul", lambda: nc.matmul(T(A), T(M43)), lambda: np.matmul(A, M43)),
    ("cross", lambda: nc.cross(T(V), T(W)), lambda: np.cross(V, W)),
    ("where", lambda: nc.where(T(A > 0), T(A), T(B)), lambda: np.where(A > 0, A, B)),
    (
        "concatenate",
        lambda: nc.concatenate([T(A), T(B)], 0),
        lambda: np.concatenate([A, B], 0),
    ),
    # `axis=None` is a value with a meaning of its own to NumPy -- flatten
    # every operand and return a line -- not "the default axis". Answering
    # `(6, 4)` here instead of `(24,)` was a wrong result rather than an error.
    (
        "concatenate-flattened",
        lambda: nc.concatenate([T(A), T(B)], None),
        lambda: np.concatenate([A, B], None),
    ),
    (
        "concatenate-flattened-ragged",
        lambda: nc.concatenate([T(A), T(V)], axis=None),
        lambda: np.concatenate([A, V], axis=None),
    ),
    (
        "concatenate-default-axis",
        lambda: nc.concatenate([T(A), T(B)]),
        lambda: np.concatenate([A, B]),
    ),
    ("stack", lambda: nc.stack([T(A), T(B)], 0), lambda: np.stack([A, B], 0)),
    ("hstack", lambda: nc.hstack([T(A), T(B)]), lambda: np.hstack([A, B])),
    ("vstack", lambda: nc.vstack([T(A), T(B)]), lambda: np.vstack([A, B])),
    ("zeros_like", lambda: nc.zeros_like(T(A)), lambda: np.zeros_like(A)),
    ("ones_like", lambda: nc.ones_like(T(A)), lambda: np.ones_like(A)),
    ("full_like", lambda: nc.full_like(T(A), 7.0), lambda: np.full_like(A, 7.0)),
    ("asarray", lambda: nc.asarray(A), lambda: A),
]


@pytest.mark.parametrize("name,got,want", CASES, ids=[c[0] for c in CASES])
def test_matches_numpy(name, got, want):
    actual, expected = _value(got()), np.asarray(want())
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-9)


SPLITS = [
    ("split", lambda: nc.split(T(A), 2, 1), lambda: np.split(A, 2, 1)),
    ("hsplit", lambda: nc.hsplit(T(A), 2), lambda: np.hsplit(A, 2)),
    (
        "vsplit",
        lambda: nc.vsplit(T(np.vstack([A, B])), 2),
        lambda: np.vsplit(np.vstack([A, B]), 2),
    ),
]


@pytest.mark.parametrize("name,got,want", SPLITS, ids=[c[0] for c in SPLITS])
def test_splits_match_numpy(name, got, want):
    parts, expected = got(), want()
    assert len(parts) == len(expected)
    for part, reference in zip(parts, expected):
        np.testing.assert_allclose(_value(part), reference, rtol=1e-6)


# What makes these four themselves is the rank promotion, and every case above
# hands them a matrix, where there is none to do. `numpy_compat` used to carry
# its own versions -- `concatenate` and `chunk` on a fixed axis -- which passed
# every matrix case and were wrong for all four of these: `vstack` joined two
# vectors into one long one where NumPy makes two rows, and the other three
# raised `IndexError` or accepted what NumPy rejects.
_RANK_PROMOTION = [
    ("vstack-1d", lambda: nc.vstack([T(V), T(W)]), lambda: np.vstack([V, W])),
    ("hstack-1d", lambda: nc.hstack([T(V), T(W)]), lambda: np.hstack([V, W])),
    ("hsplit-1d", lambda: nc.hsplit(T(V), 3), lambda: np.hsplit(V, 3)),
]


@pytest.mark.parametrize(
    "name,got,want", _RANK_PROMOTION, ids=[c[0] for c in _RANK_PROMOTION]
)
def test_the_stacks_promote_a_vector_the_way_numpy_does(name, got, want):
    ours, expected = got(), want()
    if isinstance(expected, list):
        assert len(ours) == len(expected)
        for part, reference in zip(ours, expected):
            np.testing.assert_allclose(_value(part), reference, rtol=1e-6)
    else:
        actual = _value(ours)
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_vsplit_still_refuses_a_vector():
    # There is no second axis to split, and NumPy says so rather than falling
    # back to the first -- which the compiled version used to do.
    with pytest.raises(ValueError):
        nc.vsplit(T(V), 2)


def test_the_stacks_are_the_ones_the_top_level_exports():
    # One implementation, not two. The second one was the broken one.
    for name in ("vstack", "hstack", "hsplit", "vsplit"):
        assert getattr(nc, name) is getattr(mt, name), f"{name} has two implementations"


@pytest.mark.parametrize(
    "name,got,want",
    [
        ("allclose-same", lambda: nc.allclose(T(A), T(A)), True),
        ("allclose-diff", lambda: nc.allclose(T(A), T(B)), False),
        ("array_equal-same", lambda: nc.array_equal(T(A), T(A)), True),
        ("array_equal-diff", lambda: nc.array_equal(T(A), T(B)), False),
    ],
    ids=["allclose-same", "allclose-diff", "array_equal-same", "array_equal-diff"],
)
def test_predicates_match_numpy(name, got, want):
    result = got()
    assert bool(_value(result)) is want


def test_empty_like_matches_numpy_shape_and_dtype():
    # Contents are undefined, so only shape and dtype are meaningful.
    result = nc.empty_like(T(A))
    assert tuple(result.shape) == A.shape
    assert result.numpy().dtype == A.dtype


def test_every_export_is_covered():
    """A new `numpy_compat` name must come with a comparison against NumPy.

    Seventeen of these had no test when this file was written. The module is
    only worth having if it behaves like NumPy, and that claim is checkable, so
    leaving any of it unchecked is the one thing this file exists to prevent.
    """
    exported = {name for name in dir(nc) if not name.startswith("_")}
    covered = {case[0].split("-")[0] for case in CASES + SPLITS}
    covered |= {"allclose", "array_equal", "empty_like"}

    missing = sorted(exported - covered)
    assert (
        not missing
    ), "numpy_compat exports with no comparison against NumPy: " + ", ".join(missing)
