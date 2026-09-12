# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A dense product may be computed by NumPy's BLAS; it must not matter that it was.

The engine hands a single large `float32` or `float64` product to the BLAS that
`numpy` already brought, on the engine's own buffers, and runs its own kernel
for everything else. Which side of that line a product lands on is a
performance decision that depends on its shape and dtype, and it is allowed to
change. What is not allowed to change is the answer.

So these tests do not check where a product was computed. They check the
things a reader would otherwise have to take on trust: that both sides agree
with a float64 reference to the precision of the dtype asked for, across
shapes that straddle every threshold; that a weight matrix read transposed in
place gives what transposing it first gives; that gradients still arrive; and
that the delegation survives the two things that would break a hand-off to an
interpreter -- being called from several threads at once, and being called from
inside the engine's own thread pool.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

import minitensor as mt
from minitensor import nn

# Tolerances for a product accumulated over `k` terms, relative to the largest
# value in the answer. Float32 carries ~7 decimal digits and a BLAS reorders
# the accumulation, so the error grows with the contraction; float64 has ~16
# and the same slack is far beyond anything either kernel produces.
_TOLERANCE = {"float32": 1e-5, "float64": 1e-12}

# Shapes chosen to sit either side of every rule the dispatch uses: too small
# to be worth a call, too thin a contraction to be worth one, and the sizes
# where it clearly is.
_SHAPES = [
    (1, 1, 1),
    (2, 3, 4),
    (8, 8, 8),
    (32, 32, 32),  # right at the smallest contraction that qualifies
    (31, 31, 31),
    (33, 33, 33),
    (64, 64, 64),
    (128, 128, 128),
    (17, 129, 23),  # nothing a power of two
    (256, 1, 256),  # an outer product: one multiply per output element
    (256, 8, 256),  # a contraction too short to hand over
    (256, 64, 256),
    (1, 1024, 1024),  # a vector times a matrix
    (1024, 1024, 1),  # a matrix times a vector
    (200, 300, 100),
]


def _operands(m, k, n, dtype, seed=0):
    rng = np.random.default_rng(seed)
    lhs = np.ascontiguousarray(rng.standard_normal((m, k)), dtype=dtype)
    rhs = np.ascontiguousarray(rng.standard_normal((k, n)), dtype=dtype)
    return lhs, rhs


def _relative_error(got, reference):
    scale = max(float(np.abs(reference).max()), 1e-30)
    return float(np.abs(got - reference).max()) / scale


# --- the value ---------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("m,k,n", _SHAPES)
def test_the_product_matches_a_float64_reference(m, k, n, dtype):
    lhs, rhs = _operands(m, k, n, dtype)
    got = mt.from_numpy(lhs).matmul(mt.from_numpy(rhs)).numpy()
    reference = lhs.astype(np.float64) @ rhs.astype(np.float64)

    assert got.dtype == np.dtype(dtype)
    assert _relative_error(got, reference) < _TOLERANCE[dtype]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("m,k,n", _SHAPES)
def test_the_dense_layer_matches_the_same_reference(m, k, n, dtype):
    """`input @ weight^T`, with the weight read transposed where it lies."""

    lhs, rhs = _operands(m, k, n, dtype, seed=1)
    weight = np.ascontiguousarray(rhs.T)  # `[out, in]`, as a checkpoint stores it
    got = nn.dense_layer(mt.from_numpy(lhs), mt.from_numpy(weight)).numpy()
    reference = lhs.astype(np.float64) @ rhs.astype(np.float64)

    assert got.dtype == np.dtype(dtype)
    assert _relative_error(got, reference) < _TOLERANCE[dtype]


@pytest.mark.parametrize("m,k,n", [(64, 64, 64), (16, 1024, 1024), (256, 8, 256)])
def test_reading_the_weight_transposed_agrees_with_transposing_it_first(m, k, n):
    """The two spellings of one product, to float32's precision.

    Not bit for bit: a BLAS dispatches on all three extents and on how each
    operand is stored, so the transposed form can land on a kernel that
    accumulates `k` in a different order. Both are held to the reference
    instead, which is what the library actually promises.
    """

    lhs, rhs = _operands(m, k, n, "float32", seed=2)
    weight = np.ascontiguousarray(rhs.T)
    tx, tw = mt.from_numpy(lhs), mt.from_numpy(weight)

    fused = nn.dense_layer(tx, tw).numpy()
    composed = tx.matmul(tw.transpose(0, 1)).numpy()
    reference = lhs.astype(np.float64) @ rhs.astype(np.float64)

    assert _relative_error(fused, reference) < 1e-5
    assert _relative_error(composed, reference) < 1e-5


def test_a_degenerate_extent_gives_an_empty_answer_not_a_crash():
    for m, k, n in [(0, 4, 5), (3, 0, 5), (3, 4, 0)]:
        lhs, rhs = _operands(m, k, n, "float32")
        got = mt.from_numpy(lhs).matmul(mt.from_numpy(rhs)).numpy()
        assert got.shape == (m, n)
        if got.size:
            # A zero-length contraction sums nothing, which is zero.
            np.testing.assert_array_equal(got, np.zeros((m, n), dtype=np.float32))


@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_an_integer_product_is_exact(dtype):
    """Integers never reach a BLAS, in this library or in NumPy."""

    rng = np.random.default_rng(3)
    lhs = np.ascontiguousarray(rng.integers(-9, 9, (64, 64)), dtype=dtype)
    rhs = np.ascontiguousarray(rng.integers(-9, 9, (64, 64)), dtype=dtype)
    got = mt.from_numpy(lhs).matmul(mt.from_numpy(rhs)).numpy()
    np.testing.assert_array_equal(got, lhs @ rhs)


# --- the shapes that route around the two-dimensional path -------------------


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_a_batched_product_matches_the_reference(dtype):
    rng = np.random.default_rng(4)
    lhs = np.ascontiguousarray(rng.standard_normal((6, 40, 50)), dtype=dtype)
    rhs = np.ascontiguousarray(rng.standard_normal((6, 50, 30)), dtype=dtype)
    got = mt.from_numpy(lhs).matmul(mt.from_numpy(rhs)).numpy()
    reference = lhs.astype(np.float64) @ rhs.astype(np.float64)
    assert _relative_error(got, reference) < _TOLERANCE[dtype]


def test_a_shared_right_operand_folds_into_one_product_and_still_agrees():
    """`[b, m, k] @ [k, n]`, which the engine flattens into a single GEMM."""

    rng = np.random.default_rng(5)
    lhs = np.ascontiguousarray(rng.standard_normal((7, 40, 64)), dtype="float32")
    rhs = np.ascontiguousarray(rng.standard_normal((64, 48)), dtype="float32")
    got = mt.from_numpy(lhs).matmul(mt.from_numpy(rhs)).numpy()
    reference = lhs.astype(np.float64) @ rhs.astype(np.float64)
    assert _relative_error(got, reference) < 1e-5


def test_a_broadcast_batch_still_agrees():
    rng = np.random.default_rng(6)
    lhs = np.ascontiguousarray(rng.standard_normal((1, 33, 40)), dtype="float32")
    rhs = np.ascontiguousarray(rng.standard_normal((5, 40, 21)), dtype="float32")
    got = mt.from_numpy(lhs).matmul(mt.from_numpy(rhs)).numpy()
    reference = lhs.astype(np.float64) @ rhs.astype(np.float64)
    assert _relative_error(got, reference) < 1e-5


@pytest.mark.parametrize("shape", [(1024,), (1, 1024)])
def test_a_vector_operand_keeps_its_promoted_axis_removed(shape):
    rng = np.random.default_rng(7)
    vector = np.ascontiguousarray(rng.standard_normal(shape), dtype="float32")
    matrix = np.ascontiguousarray(rng.standard_normal((1024, 512)), dtype="float32")
    got = mt.from_numpy(vector).matmul(mt.from_numpy(matrix)).numpy()
    reference = vector.astype(np.float64) @ matrix.astype(np.float64)
    assert got.shape == reference.shape
    assert _relative_error(got, reference) < 1e-5


def test_a_non_contiguous_operand_is_materialised_before_the_product():
    rng = np.random.default_rng(8)
    base = np.ascontiguousarray(rng.standard_normal((128, 128)), dtype="float32")
    tensor = mt.from_numpy(base)

    got = tensor.transpose(0, 1).matmul(tensor).numpy()
    reference = base.astype(np.float64).T @ base.astype(np.float64)
    assert _relative_error(got, reference) < 1e-5


# --- gradients ---------------------------------------------------------------


@pytest.mark.parametrize("m,k,n", [(64, 64, 64), (16, 1024, 128), (8, 8, 8)])
def test_both_operands_get_the_gradient_the_product_implies(m, k, n):
    lhs, rhs = _operands(m, k, n, "float64", seed=9)
    tl = mt.from_numpy(lhs).requires_grad_(True)
    tr = mt.from_numpy(rhs).requires_grad_(True)

    cotangent = np.ascontiguousarray(
        np.random.default_rng(10).standard_normal((m, n)), dtype="float64"
    )
    (tl.matmul(tr) * mt.from_numpy(cotangent)).sum().backward()

    np.testing.assert_allclose(tl.grad.numpy(), cotangent @ rhs.T, rtol=1e-10)
    np.testing.assert_allclose(tr.grad.numpy(), lhs.T @ cotangent, rtol=1e-10)
    mt.clear_autograd_graph()


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    "m,k,n", [(64, 512, 512), (16, 1024, 1024), (7, 5, 4), (33, 8, 65)]
)
def test_the_dense_layer_backward_matches_its_closed_forms(m, k, n, dtype):
    """`grad_input = g @ W` and `grad_weight = g^T @ x`, both of them products.

    The second hands `g` over transposed where it lies for the same reason the
    forward hands the weight over that way: the copy is the size of the
    activations, and a GEMM reads either layout.
    """

    rng = np.random.default_rng(14)
    inputs = np.ascontiguousarray(rng.standard_normal((m, k)), dtype=dtype)
    weight = np.ascontiguousarray(rng.standard_normal((n, k)), dtype=dtype)
    cotangent = np.ascontiguousarray(rng.standard_normal((m, n)), dtype=dtype)

    tx = mt.from_numpy(inputs).requires_grad_(True)
    tw = mt.from_numpy(weight).requires_grad_(True)
    (nn.dense_layer(tx, tw) * mt.from_numpy(cotangent)).sum().backward()

    wide = (
        inputs.astype(np.float64),
        weight.astype(np.float64),
        cotangent.astype(np.float64),
    )
    assert _relative_error(tx.grad.numpy(), wide[2] @ wide[1]) < _TOLERANCE[dtype]
    assert _relative_error(tw.grad.numpy(), wide[2].T @ wide[0]) < _TOLERANCE[dtype]
    mt.clear_autograd_graph()


def test_a_dense_layer_still_trains():
    """The end of the road: a delegated GEMM inside a training step."""

    rng = np.random.default_rng(11)
    inputs = mt.from_numpy(
        np.ascontiguousarray(rng.standard_normal((64, 128)), dtype="float32")
    )
    targets = mt.from_numpy(
        np.ascontiguousarray(rng.standard_normal((64, 32)), dtype="float32")
    )
    layer = nn.DenseLayer(128, 32)
    optimizer = mt.optim.SGD(list(layer.parameters()), lr=0.05)

    losses = []
    for _ in range(15):
        optimizer.zero_grad()
        loss = mt.functional.mse_loss(layer(inputs), targets)
        losses.append(float(loss.numpy()))
        loss.backward()
        optimizer.step()
    mt.clear_autograd_graph()

    assert losses[-1] < losses[0] * 0.9, losses


# --- it survives the ways a hand-off to an interpreter goes wrong ------------


def test_products_from_several_threads_at_once_all_come_out_right():
    """Every caller holds the interpreter lock; none may end up waiting on itself."""

    lhs, rhs = _operands(256, 256, 256, "float32", seed=12)
    reference = lhs.astype(np.float64) @ rhs.astype(np.float64)
    tl, tr = mt.from_numpy(lhs), mt.from_numpy(rhs)

    errors: list[float] = []
    failures: list[BaseException] = []
    lock = threading.Lock()

    def run():
        try:
            for _ in range(8):
                error = _relative_error(tl.matmul(tr).numpy(), reference)
                with lock:
                    errors.append(error)
        except BaseException as exc:  # pragma: no cover - failure path is the point
            with lock:
                failures.append(exc)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    assert not any(thread.is_alive() for thread in threads), "a product never returned"
    assert not failures, failures
    assert len(errors) == 32
    assert max(errors) < 1e-5


def test_a_product_reached_from_inside_the_engines_own_pool_still_answers():
    """Attention runs several products under one parallel walk over the batch.

    A worker that stopped to take the interpreter lock would be waiting on the
    thread that holds it and is waiting on the worker, so the provider is not
    offered work from inside the pool at all. This is the shape that would hang
    if that guard were dropped.
    """

    rng = np.random.default_rng(13)
    shape = (4, 6, 32, 64)
    query, key, value = (
        mt.from_numpy(np.ascontiguousarray(rng.standard_normal(shape), dtype="float32"))
        for _ in range(3)
    )

    out = mt.functional.scaled_dot_product_attention(query, key, value)
    assert tuple(out.shape) == shape
    assert np.isfinite(out.numpy()).all()
