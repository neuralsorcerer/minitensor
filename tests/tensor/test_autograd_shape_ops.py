# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Autograd coverage for indexing/shape/selection ops.

These operations previously produced tensors that reported ``requires_grad`` but
silently returned ``None`` gradients (no backward was wired). Each test compares
the analytic gradient against a float64 central finite difference so the
gradient math is checked end to end.
"""

import numpy as np
import pytest

import minitensor as mt


def _numeric_grad(fn, x_np, weights, eps=1e-6):
    grad = np.zeros_like(x_np)
    it = np.nditer(x_np, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        plus = x_np.copy()
        plus[idx] += eps
        minus = x_np.copy()
        minus[idx] -= eps
        grad[idx] = (fn(plus) - fn(minus)) / (2 * eps)
    return grad


def _analytic_grad(build, x_np, weights):
    x = mt.Tensor(x_np.tolist(), dtype="float64", requires_grad=True)
    out = build(x)
    if isinstance(out, tuple):
        out = out[0]
    (out * mt.Tensor(weights.tolist(), dtype="float64")).sum().backward()
    return x.grad.numpy()


def _out_shape(build, x_np):
    out = build(mt.Tensor(x_np.tolist(), dtype="float64"))
    if isinstance(out, tuple):
        out = out[0]
    return out.numpy().shape


def _check(build, shape=(3, 4), seed=0):
    rng = np.random.default_rng(seed)
    x_np = rng.standard_normal(shape)
    weights = rng.standard_normal(_out_shape(build, x_np))

    def scalar(x_perturbed):
        out = build(mt.Tensor(x_perturbed.tolist(), dtype="float64"))
        if isinstance(out, tuple):
            out = out[0]
        return float((out.numpy() * weights).sum())

    analytic = _analytic_grad(build, x_np, weights)
    numeric = _numeric_grad(scalar, x_np, weights)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-6)


def test_abs_grad():
    _check(lambda t: t.abs())


def test_clamp_grad():
    _check(lambda t: t.clamp(-0.3, 0.5))
    _check(lambda t: t.clamp_min(-0.2))
    _check(lambda t: t.clamp_max(0.4))


def test_basic_indexing_grad():
    _check(lambda t: t[:, 1:3])
    _check(lambda t: t[::2, 1:])
    _check(lambda t: t[1])
    _check(lambda t: t.narrow(1, 0, 2))


def test_index_select_and_gather_grad():
    _check(lambda t: t.index_select(1, [0, 2, 2, 1]))
    idx = mt.Tensor([[0, 1, 2, 3], [3, 2, 1, 0], [1, 1, 2, 2]], dtype="int64")
    _check(lambda t: t.gather(1, idx))


def test_flip_roll_repeat_grad():
    _check(lambda t: t.flip([0, 1]))
    _check(lambda t: t.roll(1, 1))
    _check(lambda t: t.roll(3))  # flattened, no-axis path
    _check(lambda t: t.roll([1, 2], [0, 1]))
    _check(lambda t: t.repeat(2, 2))
    _check(lambda t: t.repeat(3), shape=(4,))


def test_cat_stack_grad():
    _check(lambda t: mt.cat([t, t], 1))  # repeated input accumulates
    _check(lambda t: mt.cat([t, t * 2, t], 0))
    _check(lambda t: mt.stack([t, t], 0))


def test_min_max_grad():
    _check(lambda t: t.max())
    _check(lambda t: t.min())
    _check(lambda t: t.max(dim=1))
    _check(lambda t: t.min(dim=0))
    _check(lambda t: t.max(dim=1, keepdim=True))


def test_sort_topk_grad():
    _check(lambda t: t.sort(1))
    _check(lambda t: t.sort(0, descending=True))
    _check(lambda t: t.topk(2, 1))
    _check(lambda t: t.topk(3, 0))


def test_topk_forward_non_trailing_axis_matches_numpy():
    # Regression: topk on a non-last dim used to write output elements in the
    # wrong storage order for inner > 1.
    x_np = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    values = mt.Tensor(x_np.tolist(), dtype="float64").topk(2, dim=1)[0].numpy()
    expected = -np.sort(-x_np, axis=1)[:, :2, :]
    np.testing.assert_allclose(values, expected)


def test_median_and_nan_reduction_grad():
    # Odd length keeps the median/quantile bracketing unambiguous.
    _check(lambda t: t.median(), shape=(3, 5))
    _check(lambda t: t.median(dim=1), shape=(3, 5))
    _check(lambda t: t.nanmedian(), shape=(3, 5))
    _check(lambda t: t.nanmax(), shape=(3, 5))
    _check(lambda t: t.nanmin(dim=1), shape=(3, 5))


@pytest.mark.parametrize("q", [0.0, 0.3, 0.5, 1.0])
@pytest.mark.parametrize(
    "interpolation", ["linear", "lower", "higher", "midpoint", "nearest"]
)
def test_quantile_grad(q, interpolation):
    _check(lambda t: t.quantile(q, dim=1, interpolation=interpolation), shape=(3, 5))


def test_quantile_global_grad():
    _check(lambda t: t.quantile(0.4), shape=(3, 5))


def test_nanquantile_grad():
    _check(lambda t: t.nanquantile(0.5), shape=(3, 5))
    _check(lambda t: t.nanquantile(0.4, dim=1), shape=(3, 5))


def test_nanquantile_ignores_nan_in_gradient():
    x = mt.Tensor(
        [[1.0, float("nan"), 3.0, 5.0, 4.0]], dtype="float64", requires_grad=True
    )
    x.nanquantile(0.0).backward()  # min over the non-NaN values is 1.0
    np.testing.assert_allclose(x.grad.numpy(), [[1.0, 0.0, 0.0, 0.0, 0.0]])


def test_repeat_interleave_grad():
    _check(lambda t: t.repeat_interleave(2), shape=(3, 5))  # flattened path
    _check(lambda t: t.repeat_interleave(2, dim=1), shape=(3, 5))
    _check(
        lambda t: t.repeat_interleave(mt.Tensor([1, 2, 1, 2, 1], dtype="int64"), dim=1),
        shape=(3, 5),
    )


def test_nanmax_skips_nan_in_gradient():
    x = mt.Tensor([[1.0, float("nan"), 3.0, 2.0]], dtype="float64", requires_grad=True)
    x.nanmax().backward()
    np.testing.assert_allclose(x.grad.numpy(), [[0.0, 0.0, 1.0, 0.0]])


def test_max_global_ties_distribute_equally():
    x_np = np.array([[5.0, 5.0, 1.0], [2.0, 5.0, 3.0]])
    x = mt.Tensor(x_np.tolist(), dtype="float64", requires_grad=True)
    x.max().backward()
    mask = (x_np == 5.0).astype(float)
    np.testing.assert_allclose(x.grad.numpy(), mask / mask.sum())


@pytest.mark.parametrize(
    "build",
    [
        lambda t: t.unsqueeze(0),
        lambda t: t.unsqueeze(2),
        lambda t: t.unsqueeze(-1),
        lambda t: t.reshape(2, 1, 3, 4).squeeze(),
        lambda t: t.unsqueeze(1).squeeze(1),
        lambda t: t.flatten(),
        lambda t: t.flatten(1, 2),
        lambda t: t.ravel(),
        lambda t: mt.stack([t, t], 0),
    ],
    ids=[
        "unsqueeze0",
        "unsqueeze2",
        "unsqueeze-1",
        "squeeze_all",
        "squeeze_dim",
        "flatten",
        "flatten_range",
        "ravel",
        "stack",
    ],
)
def test_view_family_gradients_keep_input_shape(build):
    # These ops used to hand back a gradient with the *view's* shape (an extra or
    # missing size-1 axis) because they aliased the input's tensor id.
    x_np = np.random.default_rng(3).standard_normal((2, 3, 4))
    x = mt.Tensor(x_np.tolist(), dtype="float64", requires_grad=True)
    out = build(x)
    # How many times each input element appears in the output (stack uses x twice).
    multiplicity = out.numpy().size // x_np.size
    out.sum().backward()
    assert x.grad.numpy().shape == x_np.shape
    np.testing.assert_allclose(x.grad.numpy(), np.full(x_np.shape, float(multiplicity)))


@pytest.mark.parametrize("op", ["cat", "stack"])
def test_cat_stack_same_tensor_twice_accumulates(op):
    x_np = np.array([[1.0, 2.0], [3.0, 4.0]])
    x = mt.Tensor(x_np.tolist(), dtype="float64", requires_grad=True)
    if op == "cat":
        mt.cat([x, x], 0).sum().backward()
    else:
        mt.stack([x, x], 0).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), 2.0 * np.ones_like(x_np))


_SCATTER_IDX = mt.Tensor([[0, 0, 2], [1, 1, 1]], dtype="int64")


def _scatter_ref(base, dim, index, src, accumulate):
    out = base.copy()
    for pos in np.ndindex(*index.shape):
        target = list(pos)
        target[dim] = index[pos]
        if accumulate:
            out[tuple(target)] += src[pos]
        else:
            out[tuple(target)] = src[pos]
    return out


@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("accumulate", [False, True])
def test_scatter_forward_matches_reference(dim, accumulate):
    rng = np.random.default_rng(0)
    base = rng.standard_normal((3, 4, 5)).astype(np.float32)
    index = rng.integers(0, base.shape[dim], size=(3, 4, 5)).astype(np.int64)
    src = rng.standard_normal((3, 4, 5)).astype(np.float32)

    tensor = mt.Tensor(base)
    method = tensor.scatter_add if accumulate else tensor.scatter
    got = method(dim, mt.Tensor(index, dtype="int64"), mt.Tensor(src)).numpy()

    np.testing.assert_allclose(
        got, _scatter_ref(base, dim, index, src, accumulate), atol=1e-5
    )


def test_scatter_add_is_the_adjoint_of_gather():
    """<gather(x), v> == <x, scatter_add(0, v)> for all x and v.

    This is the property that makes scatter_add the right primitive: it is
    exactly the operation gather's own backward pass performs.
    """
    rng = np.random.default_rng(4)
    x = rng.standard_normal((4, 6))
    index = rng.integers(0, 6, size=(4, 3)).astype(np.int64)
    v = rng.standard_normal((4, 3))
    idx = mt.Tensor(index, dtype="int64")

    gathered = mt.Tensor(x, dtype="float64").gather(1, idx).numpy()
    scattered = (
        mt.Tensor(np.zeros((4, 6)), dtype="float64")
        .scatter_add(1, idx, mt.Tensor(v, dtype="float64"))
        .numpy()
    )
    assert float((gathered * v).sum()) == pytest.approx(float((x * scattered).sum()))


@pytest.mark.parametrize("method", ["scatter", "scatter_add"])
def test_scatter_gradient_wrt_input(method):
    src = mt.Tensor(
        np.random.default_rng(2).standard_normal((2, 3)).tolist(), dtype="float64"
    )
    _check(lambda t: getattr(t, method)(1, _SCATTER_IDX, src), shape=(2, 3))


@pytest.mark.parametrize("method", ["scatter", "scatter_add"])
def test_scatter_gradient_wrt_source(method):
    base = mt.Tensor(
        np.random.default_rng(1).standard_normal((2, 3)).tolist(), dtype="float64"
    )
    _check(lambda t: getattr(base, method)(1, _SCATTER_IDX, t), shape=(2, 3))


def test_scatter_add_gives_every_duplicate_writer_the_gradient():
    # Column 1 of row 1 is written by all three source positions. Addition is
    # linear, so each one sees the destination's gradient in full.
    base = mt.Tensor(np.zeros((2, 3)), dtype="float64", requires_grad=True)
    src = mt.Tensor(np.ones((2, 3)), dtype="float64", requires_grad=True)
    weights = mt.Tensor([[1.0, 2.0, 4.0], [8.0, 16.0, 32.0]], dtype="float64")
    (base.scatter_add(1, _SCATTER_IDX, src) * weights).sum().backward()

    # index row 1 is [1, 1, 1]; every writer collects the weight at column 1.
    np.testing.assert_allclose(src.grad.numpy()[1], [16.0, 16.0, 16.0])
    # index row 0 is [0, 0, 2]: two writers on column 0, one on column 2.
    np.testing.assert_allclose(src.grad.numpy()[0], [1.0, 1.0, 4.0])
    # Accumulating leaves the original value in place, so input passes through.
    np.testing.assert_allclose(base.grad.numpy(), weights.numpy())
    mt.clear_autograd_graph()


def test_scatter_gives_the_gradient_only_to_the_surviving_writer():
    """Overwriting severs dependencies that accumulating preserves.

    A source position whose value was overwritten never reached the output, and
    an input slot that was written no longer affects it either. Both must read
    back as exactly zero, not as a small number.
    """
    base = mt.Tensor(np.zeros((2, 3)), dtype="float64", requires_grad=True)
    src = mt.Tensor(np.ones((2, 3)), dtype="float64", requires_grad=True)
    weights = mt.Tensor([[1.0, 2.0, 4.0], [8.0, 16.0, 32.0]], dtype="float64")
    (base.scatter(1, _SCATTER_IDX, src) * weights).sum().backward()

    # Row 0 writes columns [0, 0, 2]: of the two writers aimed at column 0 only
    # the second survives, so the first earns nothing.
    np.testing.assert_allclose(src.grad.numpy()[0], [0.0, 1.0, 4.0])
    # Row 1 writes [1, 1, 1]: only the last of the three survives.
    np.testing.assert_allclose(src.grad.numpy()[1], [0.0, 0.0, 16.0])
    # Written slots no longer depend on the input; untouched ones still do.
    np.testing.assert_allclose(base.grad.numpy(), [[0.0, 2.0, 0.0], [8.0, 0.0, 32.0]])
    mt.clear_autograd_graph()


def test_scatter_add_is_deterministic_under_heavy_collision():
    """Float addition is not associative, so accumulation order must be fixed.

    Every update here lands on one of eight slots, which is the worst case for
    a parallel kernel: if the order varied with thread scheduling, repeated runs
    would differ in the low bits and results would not reproduce.
    """
    rng = np.random.default_rng(5)
    n = 200_000
    base = mt.Tensor(np.zeros((1, 8), dtype=np.float32))
    index = mt.Tensor(rng.integers(0, 8, size=(1, n)).astype(np.int64), dtype="int64")
    src = mt.Tensor(rng.standard_normal((1, n)).astype(np.float32))

    results = {base.scatter_add(1, index, src).numpy().tobytes() for _ in range(10)}
    assert len(results) == 1


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
def test_scatter_supports_numeric_dtypes(dtype):
    base = mt.Tensor(np.zeros((1, 3)), dtype=dtype)
    src = mt.Tensor(np.ones((1, 3)), dtype=dtype)
    index = mt.Tensor([[0, 0, 1]], dtype="int64")
    assert base.scatter(1, index, src).tolist() == [[1, 1, 0]]
    assert base.scatter_add(1, index, src).tolist() == [[2, 1, 0]]


def test_scatter_bool_is_supported_but_scatter_add_is_not():
    base = mt.Tensor(np.zeros((1, 3), dtype=bool), dtype="bool")
    src = mt.Tensor(np.ones((1, 3), dtype=bool), dtype="bool")
    index = mt.Tensor([[0, 0, 1]], dtype="int64")
    assert base.scatter(1, index, src).tolist() == [[True, True, False]]
    with pytest.raises(Exception, match="boolean"):
        base.scatter_add(1, index, src)


def test_scatter_rejects_malformed_arguments():
    base = mt.Tensor(np.zeros((2, 3), dtype=np.float32))
    src = mt.Tensor(np.ones((2, 3), dtype=np.float32))
    good = mt.Tensor([[0, 1, 2], [0, 1, 2]], dtype="int64")

    with pytest.raises(Exception):  # index out of range for the scattered axis
        base.scatter(1, mt.Tensor([[0, 1, 9], [0, 1, 2]], dtype="int64"), src)
    with pytest.raises(Exception):  # negative index
        base.scatter(1, mt.Tensor([[0, 1, -1], [0, 1, 2]], dtype="int64"), src)
    with pytest.raises(Exception):  # index dtype must be int64
        base.scatter(1, mt.Tensor([[0.0, 1.0, 2.0]], dtype="float32"), src)
    with pytest.raises(Exception):  # index and src shapes must agree
        base.scatter(1, good, mt.Tensor(np.ones((2, 2), dtype=np.float32)))
    with pytest.raises(Exception):  # src dtype must match input
        base.scatter(1, good, mt.Tensor(np.ones((2, 3)), dtype="float64"))
    with pytest.raises(Exception):  # dim out of range
        base.scatter(5, good, src)


@pytest.mark.parametrize("method", ["scatter", "scatter_add"])
def test_scatter_propagates_gradient_through_a_computed_source(method):
    """The source operand must be walkable, not merely receive a gradient.

    When ``src`` is a leaf its gradient lands in the output map either way, so a
    backward that fails to declare ``src`` as an input still looks correct.
    Building ``src`` from an earlier op is what exposes the difference: the
    engine has to traverse into it to reach ``w``.
    """
    w = mt.Tensor(np.ones((2, 3)), dtype="float64", requires_grad=True)
    src = w * mt.Tensor(2.0, dtype="float64")
    base = mt.Tensor(np.zeros((2, 3)), dtype="float64", requires_grad=True)
    index = mt.Tensor([[0, 1, 2], [0, 1, 2]], dtype="int64")

    getattr(base, method)(1, index, src).sum().backward()

    assert w.grad is not None, "gradient did not reach past src"
    np.testing.assert_allclose(w.grad.numpy(), np.full((2, 3), 2.0))
    mt.clear_autograd_graph()
