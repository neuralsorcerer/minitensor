# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Products, distances and statistics defined in terms of other operations.

Every function here is a short arrangement of existing kernels -- a reshape and
a product, a subtraction and a norm -- so each is written once, in Python,
rather than added to the extension. That keeps the shipped binary the size of
the operations that actually need a kernel, and it means these inherit the
accuracy and the gradients of the ones underneath rather than restating them.
"""

from __future__ import annotations

import operator as _operator

from . import _core as _C
from ._shape import _atleast_tensor, _normalize_axis

Tensor = _C.Tensor
_F = _C.functional


def _require_float(tensor: Tensor, name: str) -> Tensor:
    if "float" not in str(tensor.dtype):
        raise ValueError(f"{name} requires a floating point tensor, got {tensor.dtype}")
    return tensor


def outer(input: object, other: object) -> Tensor:
    """The outer product of two flattened tensors: `out[i, j] = a[i] * b[j]`."""

    a = _atleast_tensor(input).reshape(-1, 1)
    b = _atleast_tensor(other).reshape(1, -1)
    return a * b


def vdot(input: object, other: object) -> Tensor:
    """The inner product of two flattened tensors, of any matching shape.

    `dot` insists on 1-D operands; this flattens first, which is the only
    difference between them for real tensors.
    """

    a = _atleast_tensor(input).reshape(-1)
    b = _atleast_tensor(other).reshape(-1)
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"vdot needs the same number of elements in each operand, "
            f"got {a.shape[0]} and {b.shape[0]}"
        )
    return _F.dot(a, b)


def kron(input: object, other: object) -> Tensor:
    """The Kronecker product: each element of `input` scaling a copy of `other`.

    Built as one broadcast product rather than a loop over blocks. Interleaving
    a length-1 axis into each operand puts each element of `input` against a
    whole copy of `other`, and the final reshape merges the interleaved pairs
    back into single axes.
    """

    a = _atleast_tensor(input)
    b = _atleast_tensor(other)

    rank = max(a.ndim(), b.ndim())
    a_dims = (1,) * (rank - a.ndim()) + tuple(a.shape)
    b_dims = (1,) * (rank - b.ndim()) + tuple(b.shape)

    interleaved_a: list[int] = []
    interleaved_b: list[int] = []
    for a_dim, b_dim in zip(a_dims, b_dims):
        interleaved_a += [a_dim, 1]
        interleaved_b += [1, b_dim]

    product = a.reshape(interleaved_a) * b.reshape(interleaved_b)
    return product.reshape([a_dim * b_dim for a_dim, b_dim in zip(a_dims, b_dims)])


def dist(input: object, other: object, p: float = 2.0) -> Tensor:
    """The `p`-norm of the difference: how far apart two tensors are."""

    a = _atleast_tensor(input)
    b = _atleast_tensor(other)
    return _F.norm(a - b, p)


def cdist(input: object, other: object, p: float = 2.0) -> Tensor:
    """Every pairwise `p`-distance between the rows of two batches.

    `input` is `(..., n, d)` and `other` is `(..., m, d)`; the result is
    `(..., n, m)`. The difference is formed in full, so this costs `n * m * d`
    elements of memory -- fine for the batch sizes a distance matrix is usually
    wanted at, and the reason to reach for `matmul` instead when it is not.
    """

    a = _require_float(_atleast_tensor(input), "cdist")
    b = _require_float(_atleast_tensor(other), "cdist")
    if a.ndim() < 2 or b.ndim() < 2:
        raise ValueError("cdist requires tensors with at least two dimensions")
    if a.shape[-1] != b.shape[-1]:
        raise ValueError(
            f"cdist needs matching feature counts, got {a.shape[-1]} and {b.shape[-1]}"
        )

    return _F.norm(a.unsqueeze(-2) - b.unsqueeze(-3), p, [-1])


def diff(input: object, n: int = 1, dim: int = -1) -> Tensor:
    """The `n`-th discrete difference along `dim`.

    Each pass shortens the axis by one, so `n` passes over a length-`k` axis
    leave `max(k - n, 0)` elements.
    """

    tensor = _atleast_tensor(input)
    if tensor.ndim() == 0:
        raise ValueError("diff requires a tensor with at least one dimension")

    order = _operator.index(n)
    if order < 0:
        raise ValueError(f"diff requires a non-negative order, got {order}")

    axis = _normalize_axis(dim, tensor.ndim(), "diff")
    for _ in range(order):
        length = tensor.shape[axis]
        if length == 0:
            break
        tensor = _F.narrow(tensor, axis, 1, length - 1) - _F.narrow(
            tensor, axis, 0, length - 1
        )
    return tensor


def trapezoid(
    y: object, x: object | None = None, dx: float = 1.0, dim: int = -1
) -> Tensor:
    """The trapezoidal integral of `y` along `dim`.

    With `x` given, the sample spacing comes from it and may be uneven; with
    only `dx`, the samples are taken as evenly spaced.
    """

    values = _require_float(_atleast_tensor(y), "trapezoid")
    if values.ndim() == 0:
        raise ValueError("trapezoid requires a tensor with at least one dimension")

    axis = _normalize_axis(dim, values.ndim(), "trapezoid")
    length = values.shape[axis]
    if length < 2:
        # No interval to integrate over; the answer is zero, shaped like the
        # reduction would be.
        return _F.sum(_F.narrow(values, axis, 0, 0), [axis])

    left = _F.narrow(values, axis, 0, length - 1)
    right = _F.narrow(values, axis, 1, length - 1)
    heights = (left + right) * 0.5

    if x is None:
        return _F.sum(heights, [axis]) * float(dx)

    positions = _require_float(_atleast_tensor(x), "trapezoid")
    if positions.ndim() == 1 and values.ndim() > 1:
        # A shared 1-D coordinate: give it the axis it measures and let the
        # widths broadcast over the rest.
        shape = [1] * values.ndim()
        shape[axis] = positions.shape[0]
        positions = positions.reshape(shape)
    widths = diff(positions, 1, axis)
    return _F.sum(heights * widths, [axis])


# `numpy` spells it `trapz` as well, and enough code says that for the alias to
# be worth the line.
trapz = trapezoid


def cov(
    input: object,
    correction: int = 1,
    fweights: object | None = None,
    aweights: object | None = None,
) -> Tensor:
    """The covariance matrix of the *rows* of `input`.

    Each row is a variable and each column an observation, which is NumPy's and
    PyTorch's convention and the opposite of a design matrix's. A 1-D input is
    one variable, so the result is its scalar variance.

    `fweights` counts repeats of each observation and `aweights` weights their
    reliability; the difference shows in the denominator, where `aweights`
    reduces the effective sample size rather than the count.
    """

    matrix = _require_float(_atleast_tensor(input), "cov")
    if matrix.ndim() > 2:
        raise ValueError("cov requires a 1-D or 2-D tensor")
    # A single variable has a scalar variance, not a one-by-one matrix.
    single_variable = matrix.ndim() <= 1
    if single_variable:
        matrix = matrix.reshape(1, -1)

    observations = matrix.shape[1]
    if observations == 0:
        raise ValueError("cov requires at least one observation")

    weights = None
    if fweights is not None:
        weights = _check_weights(fweights, observations, "fweights")
    if aweights is not None:
        scaled = _check_weights(aweights, observations, "aweights")
        weights = scaled if weights is None else weights * scaled

    if weights is None:
        total = float(observations)
        centred = matrix - _F.mean(matrix, [1], True)
        scale = total - correction
        weighted = centred
    else:
        total = _F.sum(weights).item()
        if total <= 0.0:
            raise ValueError("cov requires the weights to sum to a positive value")
        centred = matrix - _F.sum(matrix * weights, [1], True) / total
        if aweights is None:
            scale = total - correction
        else:
            # `aweights` shrinks the effective sample size rather than the
            # count, which is what makes an unbiased estimate under unequal
            # reliability.
            effective = _F.sum(weights * scaled).item()
            scale = total - correction * effective / total
        weighted = centred * weights

    if scale <= 0.0:
        raise ValueError(
            f"cov: the correction {correction} leaves a non-positive divisor "
            f"for {observations} observations"
        )

    result = _F.matmul(weighted, _F.transpose(centred, 0, 1)) / scale
    return result.reshape([]) if single_variable else result


def _check_weights(weights: object, observations: int, name: str) -> Tensor:
    tensor = _atleast_tensor(weights)
    if tensor.ndim() != 1 or tensor.shape[0] != observations:
        raise ValueError(
            f"cov requires {name} to be 1-D with one entry per observation "
            f"({observations})"
        )
    return tensor.astype("float64") if "float" not in str(tensor.dtype) else tensor


def corrcoef(input: object) -> Tensor:
    """The Pearson correlation matrix of the rows of `input`.

    The covariance divided by the outer product of the standard deviations, and
    then clamped: the division is exact in theory and can land a hair outside
    `[-1, 1]` in floating point, which a caller comparing against 1 would see.
    """

    covariance = cov(input, correction=1)
    if covariance.ndim() == 0:
        # One variable is perfectly correlated with itself, unless it never
        # varies at all -- and `0 / 0` is what says so.
        return covariance / covariance

    deviations = _F.sqrt(_F.diagonal(covariance))
    normalized = covariance / outer(deviations, deviations)
    return _F.clamp(normalized, -1.0, 1.0)
