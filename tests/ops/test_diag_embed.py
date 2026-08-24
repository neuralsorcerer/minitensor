# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Building a diagonal, which the library could only read.

`diagonal` extracts one and `trace` sums it, but nothing put a vector *onto* a
diagonal -- so reconstructing a matrix from a factorisation, the thing every
`svd` and `eigh` docstring writes as `U @ diag(s) @ Vh`, could not be written
that way at all. It had to be spelled as a broadcast multiply, which is what the
factorisations do internally and is not what a caller reaching for `diag` has in
mind.

The two operations are each other's inverse, and each other's derivative: a
permutation of elements into a larger zeroed tensor always has its own transpose
for a gradient. Both directions of that are checked below, because it is the
kind of claim that is easy to make and easy to get backwards.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt


def _t(a):
    # Not `ascontiguousarray`: it promotes a 0-d array to shape `(1,)`, which
    # would quietly turn the rank-rejection test below into a 1x1 matrix.
    return mt.Tensor.from_numpy(np.asarray(a) if a.ndim == 0 else np.ascontiguousarray(a))


# --------------------------------------------------------------------------
# diag, the NumPy spelling
# --------------------------------------------------------------------------


@pytest.mark.parametrize("offset", [-3, -1, 0, 1, 3])
@pytest.mark.parametrize("n", [1, 2, 5])
def test_diag_of_a_vector_matches_numpy(n, offset):
    v = np.arange(1.0, n + 1)
    assert np.array_equal(mt.diag(_t(v), offset).numpy(), np.diag(v, offset))


@pytest.mark.parametrize("offset", [-2, -1, 0, 1, 2])
def test_diag_of_a_matrix_matches_numpy(offset):
    a = np.arange(20.0).reshape(4, 5)
    assert np.array_equal(mt.diag(_t(a), offset).numpy(), np.diag(a, offset))


def test_diag_round_trips_through_both_directions():
    v = np.array([2.0, 5.0, 9.0])
    assert np.array_equal(mt.diag(mt.diag(_t(v))).numpy(), v)


def test_diag_defaults_to_the_main_diagonal():
    assert np.array_equal(mt.diag(_t(np.array([1.0, 2.0]))).numpy(), np.eye(2) * [1, 2])


@pytest.mark.parametrize("bad", [np.zeros(()), np.zeros((2, 3, 4))])
def test_diag_rejects_ranks_it_cannot_read(bad):
    """Zero dimensions has no diagonal and three is ambiguous -- NumPy raises
    for both, and guessing would be worse than saying so."""
    with pytest.raises(Exception, match="1- or 2-dimensional"):
        mt.diag(_t(bad))


# --------------------------------------------------------------------------
# diag_embed, the batched form
# --------------------------------------------------------------------------


@pytest.mark.parametrize("offset", [-2, -1, 0, 1, 2])
def test_diag_embed_puts_the_values_where_diagonal_finds_them(offset):
    """The defining relationship: what goes in comes back out."""
    v = np.array([1.0, 2.0, 3.0, 4.0])
    built = mt.diag_embed(_t(v), offset).numpy()
    assert np.array_equal(np.diag(built, offset), v)
    assert built.shape == (4 + abs(offset), 4 + abs(offset))


def test_diag_embed_is_zero_off_the_diagonal():
    built = mt.diag_embed(_t(np.array([1.0, 2.0, 3.0]))).numpy()
    assert np.count_nonzero(built) == 3
    assert np.array_equal(built, np.diag([1.0, 2.0, 3.0]))


@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 3, 4), (2, 1, 3, 5)])
def test_diag_embed_shapes(shape):
    built = mt.diag_embed(_t(np.zeros(shape))).numpy()
    assert built.shape == shape + (shape[-1],)


@pytest.mark.parametrize("shape", [(2, 3), (4, 2, 5)])
def test_diag_embed_batched_matches_per_matrix(shape):
    a = np.random.default_rng(0).standard_normal(shape)
    built = mt.diag_embed(_t(a)).numpy()
    flat = a.reshape(-1, shape[-1])
    for index, row in enumerate(flat):
        assert np.array_equal(built.reshape(-1, shape[-1], shape[-1])[index], np.diag(row))


def test_diag_embed_chooses_its_axes():
    """`dim1` and `dim2` are positions in the output, which has one axis more."""
    a = np.random.default_rng(1).standard_normal((2, 3))
    default = mt.diag_embed(_t(a), 0, -2, -1).numpy()
    swapped = mt.diag_embed(_t(a), 0, 0, 1).numpy()

    assert default.shape == (2, 3, 3)
    assert swapped.shape == (3, 3, 2)
    for batch in range(2):
        assert np.array_equal(swapped[:, :, batch], default[batch])


def test_diag_embed_rejects_a_repeated_axis():
    with pytest.raises(Exception, match="distinct"):
        mt.diag_embed(_t(np.zeros(3)), 0, 1, 1)


@pytest.mark.parametrize(
    "values,dtype",
    [
        ([1, 2, 3], "int32"),
        ([1, 2, 3], "int64"),
        ([True, False, True], "bool"),
        ([1.5, 2.5], "float32"),
    ],
)
def test_diag_embed_keeps_its_dtype(values, dtype):
    """Including `bool`, which the gradient path has no meaning for and which
    the shared walk reaches only because it assigns rather than accumulates."""
    t = mt.Tensor(values, dtype=dtype)
    built = mt.diag_embed(t).numpy()
    assert str(built.dtype) == dtype
    assert np.array_equal(np.diag(built), np.array(values))


def test_diag_embed_of_an_empty_vector():
    assert mt.diag_embed(_t(np.zeros(0))).numpy().shape == (0, 0)


# --------------------------------------------------------------------------
# What it was for
# --------------------------------------------------------------------------


def test_reconstructing_a_matrix_from_its_singular_values():
    """`U @ diag(s) @ Vh`, written the way every docstring writes it."""
    a = np.random.default_rng(2).standard_normal((5, 3))
    u, s, vt = mt.svd(_t(a), False)
    rebuilt = (u @ mt.diag_embed(s) @ vt).numpy()
    assert np.allclose(rebuilt, a, atol=1e-12)


def test_reconstructing_a_symmetric_matrix_from_its_eigenvalues():
    values = np.random.default_rng(3).standard_normal((4, 4))
    a = (values + values.T) / 2
    w, v = mt.eigh(_t(a))
    rebuilt = (v @ mt.diag_embed(w) @ v.transpose(-1, -2)).numpy()
    assert np.allclose(rebuilt, a, atol=1e-12)


def test_reconstructing_a_batch():
    a = np.random.default_rng(4).standard_normal((3, 4, 4))
    u, s, vt = mt.svd(_t(a), False)
    assert np.allclose((u @ mt.diag_embed(s) @ vt).numpy(), a, atol=1e-12)


# --------------------------------------------------------------------------
# Gradients
# --------------------------------------------------------------------------


@pytest.mark.parametrize("offset", [-1, 0, 2])
def test_diag_embed_gradient_is_the_diagonal_of_what_arrives(offset):
    """Whatever landed on the diagonal is what reaches the input; the zeros
    around it came from nowhere and lead nowhere."""
    v = np.array([1.0, 2.0, 3.0])
    weights = np.random.default_rng(5).standard_normal((3 + abs(offset),) * 2)

    t = mt.Tensor.from_numpy(v, requires_grad=True)
    (mt.diag_embed(t, offset) * _t(weights)).sum().backward()
    assert np.allclose(t.grad.numpy(), np.diag(weights, offset))


def test_diag_embed_and_diagonal_are_each_others_derivative():
    """One direction is checked above; this is the other."""
    a = np.random.default_rng(6).standard_normal((4, 4))
    weights = np.array([1.0, 2.0, 3.0, 4.0])

    t = mt.Tensor.from_numpy(a, requires_grad=True)
    (mt.diagonal(t) * _t(weights)).sum().backward()
    assert np.allclose(t.grad.numpy(), np.diag(weights))


def test_gradient_flows_through_a_reconstruction():
    """The round trip is the identity, so its gradient has to be as well."""
    v = np.array([2.0, 3.0, 5.0])
    t = mt.Tensor.from_numpy(v, requires_grad=True)
    mt.diagonal(mt.diag_embed(t)).sum().backward()
    assert np.allclose(t.grad.numpy(), np.ones(3))


def test_no_grad_when_not_required():
    assert not mt.diag_embed(_t(np.zeros(3))).requires_grad
