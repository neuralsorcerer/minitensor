# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Matrix products, inverses and rescalings, in terms of what already exists.

`matmul`, `solve` and `svd` are the kernels. Everything here is one of those
pointed at a rearranged operand: `mm` and `mv` are `matmul` with the ranks
checked, `tensordot` is `matmul` with the contracted axes moved to the end and
flattened, `inverse` is `solve` against an identity, and `pinverse` is `svd`
with the small singular values dropped. None of them earns a kernel, so none
gets one, and each inherits the accuracy and the gradient of the one
underneath.
"""

from __future__ import annotations

import math as _math
import operator as _operator

from . import _core as _C
from ._elementwise import signbit as _signbit
from ._shape import _atleast_tensor, _normalize_axis

Tensor = _C.Tensor
_F = _C.functional


def _require_rank(tensor: Tensor, rank: int, name: str, what: str) -> Tensor:
    if tensor.ndim() != rank:
        raise ValueError(
            f"{name} requires a {rank}-dimensional {what}, got {tensor.ndim()} dimensions"
        )
    return tensor


# --- the named matmuls ------------------------------------------------------


def t(input: object) -> Tensor:
    """The transpose of a matrix, and anything of lower rank unchanged.

    `transpose` names the two dimensions to swap; this is the shorthand for
    the only interesting case, and it declines a rank above two rather than
    guessing which two axes were meant.
    """

    tensor = _atleast_tensor(input)
    if tensor.ndim() < 2:
        return tensor
    if tensor.ndim() > 2:
        raise ValueError(
            f"t expects at most two dimensions -- name them with `transpose` "
            f"for a tensor of rank {tensor.ndim()}"
        )
    return _F.transpose(tensor, 0, 1)


def numel(input: object) -> int:
    """How many elements the tensor holds, as a Python int."""

    return _atleast_tensor(input).numel()


def mm(input: object, mat2: object) -> Tensor:
    """The product of two matrices.

    `matmul` also broadcasts batches and promotes vectors; this rejects
    anything that is not two matrices, which is the point of asking for it by
    this name.
    """

    left = _require_rank(_atleast_tensor(input), 2, "mm", "first operand")
    right = _require_rank(_atleast_tensor(mat2), 2, "mm", "second operand")
    return _F.matmul(left, right)


def mv(input: object, vec: object) -> Tensor:
    """A matrix times a vector."""

    matrix = _require_rank(_atleast_tensor(input), 2, "mv", "matrix")
    vector = _require_rank(_atleast_tensor(vec), 1, "mv", "vector")
    return _F.matmul(matrix, vector)


def inner(input: object, other: object) -> Tensor:
    """The sum-product over the last axis of each operand.

    For two vectors this is the dot product; for higher ranks every pair of
    trailing rows is contracted, leaving the leading axes of both side by side.
    """

    return tensordot(input, other, ([-1], [-1]))


def _contraction_axes(dims: object, left_rank: int, right_rank: int) -> tuple[list[int], list[int]]:
    """The two axis lists `tensordot` contracts over.

    An integer means "the last `n` of the left against the first `n` of the
    right", which is the convention NumPy took and the reason `dims=2` is its
    default rather than something more obviously symmetric.
    """

    if isinstance(dims, int):
        count = _operator.index(dims)
        if count < 0:
            raise ValueError(f"tensordot requires a non-negative dims, got {dims}")
        if count > left_rank or count > right_rank:
            raise ValueError(
                f"tensordot cannot contract {count} axes of tensors with "
                f"{left_rank} and {right_rank} dimensions"
            )
        return (
            [left_rank - count + i for i in range(count)],
            list(range(count)),
        )

    try:
        left_axes, right_axes = dims
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "tensordot dims must be an integer or a pair of axis sequences"
        ) from exc

    left_list = [_normalize_axis(a, left_rank, "tensordot") for a in _as_sequence(left_axes)]
    right_list = [_normalize_axis(a, right_rank, "tensordot") for a in _as_sequence(right_axes)]
    if len(left_list) != len(right_list):
        raise ValueError(
            f"tensordot needs the same number of axes on each side, got "
            f"{len(left_list)} and {len(right_list)}"
        )
    return left_list, right_list


def _as_sequence(value: object) -> list:
    if isinstance(value, int):
        return [value]
    return list(value)  # type: ignore[arg-type]


def tensordot(input: object, other: object, dims: object = 2) -> Tensor:
    """Contract `input` and `other` over the axes `dims` names.

    Done by moving the contracted axes to the end of the left operand and the
    front of the right, flattening each side into a matrix and calling
    `matmul` once. That is the whole operation: a general contraction is a
    matrix product with the axes rearranged, and doing it this way inherits the
    blocked matmul rather than writing a loop over indices.
    """

    left = _atleast_tensor(input)
    right = _atleast_tensor(other)
    left_axes, right_axes = _contraction_axes(dims, left.ndim(), right.ndim())

    left_kept = [i for i in range(left.ndim()) if i not in left_axes]
    right_kept = [i for i in range(right.ndim()) if i not in right_axes]

    left_shape = list(left.shape)
    right_shape = list(right.shape)
    for a, b in zip(left_axes, right_axes):
        if left_shape[a] != right_shape[b]:
            raise ValueError(
                f"tensordot cannot contract axes of length {left_shape[a]} and "
                f"{right_shape[b]}"
            )

    kept_rows = _math.prod([left_shape[i] for i in left_kept]) if left_kept else 1
    kept_cols = _math.prod([right_shape[i] for i in right_kept]) if right_kept else 1
    shared = _math.prod([left_shape[i] for i in left_axes]) if left_axes else 1

    rows = _F.permute(left, left_kept + left_axes).reshape([kept_rows, shared])
    cols = _F.permute(right, right_axes + right_kept).reshape([shared, kept_cols])
    product = _F.matmul(rows, cols)

    result_shape = [left_shape[i] for i in left_kept] + [right_shape[i] for i in right_kept]
    return product.reshape(result_shape)


def addmm(
    input: object, mat1: object, mat2: object, beta: float = 1, alpha: float = 1
) -> Tensor:
    """`beta * input + alpha * (mat1 @ mat2)`, the fused form a linear layer
    is written in."""

    return _atleast_tensor(input) * beta + mm(mat1, mat2) * alpha


def baddbmm(
    input: object, batch1: object, batch2: object, beta: float = 1, alpha: float = 1
) -> Tensor:
    """`beta * input + alpha * (batch1 @ batch2)` over a batch of matrices."""

    left = _require_rank(_atleast_tensor(batch1), 3, "baddbmm", "first batch")
    right = _require_rank(_atleast_tensor(batch2), 3, "baddbmm", "second batch")
    return _atleast_tensor(input) * beta + _F.bmm(left, right) * alpha


# --- inverses ---------------------------------------------------------------


def inverse(input: object) -> Tensor:
    """The inverse of each square matrix in the stack.

    Solved against an identity rather than formed by cofactors: `solve` is the
    factorization that a general inverse is computed from anyway, and it is
    what carries the gradient. A caller who wants `inverse(A) @ b` should ask
    `solve(A, b)` instead -- it is the same answer without forming the inverse,
    and it is both faster and better conditioned.
    """

    matrix = _atleast_tensor(input)
    if matrix.ndim() < 2:
        raise ValueError(
            f"inverse requires at least two dimensions, got {matrix.ndim()}"
        )
    size = matrix.shape[-1]
    if matrix.shape[-2] != size:
        raise ValueError(
            f"inverse requires square matrices, got {matrix.shape[-2]} by {size}"
        )
    identity = _C.Tensor.eye(size, dtype=str(matrix.dtype))
    return _F.solve(matrix, identity)


def pinverse(input: object, rcond: float = 1e-15) -> Tensor:
    """The Moore-Penrose pseudo-inverse.

    `V diag(1/s) U^T` over the singular values above `rcond * s_max`. The
    threshold is what makes it a pseudo-inverse rather than a division by
    nearly zero: a singular value at the noise floor carries no information
    about the matrix, and inverting it would amplify that noise without bound.
    """

    matrix = _atleast_tensor(input)
    if matrix.ndim() != 2:
        raise ValueError(
            f"pinverse requires a two-dimensional tensor, got {matrix.ndim()}"
        )

    u, s, vh = _F.svd(matrix, False)
    largest = _F.amax(s).item()
    cutoff = rcond * largest
    # Zero rather than a huge reciprocal for the singular values that are
    # under the threshold; multiplying by zero drops their direction from the
    # answer, which is what the pseudo-inverse is defined to do.
    inverted = _F.where(s > cutoff, 1.0 / _F.clamp_min(s, cutoff), s * 0.0)
    return _F.matmul(_F.transpose(vh, 0, 1) * inverted, _F.transpose(u, 0, 1))


def logdet(input: object) -> Tensor:
    """The natural log of the determinant, or `-inf` where it is not positive.

    Taken from `slogdet` rather than from `log(det(A))`: the determinant of a
    large matrix leaves float64's range long before its logarithm becomes
    uninteresting, and `slogdet` is the form that survives that.
    """

    sign, log_absolute = _F.slogdet(_atleast_tensor(input))
    negative_infinity = float("-inf")
    return _F.where(
        sign > 0,
        log_absolute,
        _C.Tensor.full(list(log_absolute.shape), negative_infinity, dtype=str(log_absolute.dtype)),
    )


# --- rescaling --------------------------------------------------------------


def renorm(input: object, p: float, dim: int, maxnorm: float) -> Tensor:
    """Scale down the sub-tensors along `dim` whose `p`-norm exceeds `maxnorm`.

    The ones already under it are left exactly as they are, not rescaled by a
    factor of one: an untouched row has to come back bit-for-bit, which is what
    makes this usable as an embedding constraint applied every step.
    """

    tensor = _atleast_tensor(input)
    if maxnorm < 0:
        raise ValueError(f"renorm requires a non-negative maxnorm, got {maxnorm}")
    axis = _normalize_axis(dim, tensor.ndim(), "renorm")

    # The norm of each slice, over every axis but `dim`, kept broadcastable.
    others = [i for i in range(tensor.ndim()) if i != axis]
    norms = _F.norm(tensor, p, others, True) if others else _F.abs(tensor)
    # `clamp_max` on the *ratio* rather than a branch: a slice under the limit
    # gets a factor of exactly 1, and multiplying by 1 is exact.
    scale = _F.clamp_max(maxnorm / _F.clamp_min(norms, 1e-30), 1.0)
    return tensor * scale


def vander(x: object, N: int | None = None, increasing: bool = False) -> Tensor:
    """The Vandermonde matrix of `x`: each row a geometric series in one entry.

    Column `j` is `x ** j` with `increasing=True`, and `x ** (N - 1 - j)`
    without -- the descending order NumPy defaults to, because that is the one
    that makes `vander(x) @ c` evaluate a polynomial with `c` in the order
    people write coefficients.
    """

    values = _atleast_tensor(x)
    if values.ndim() != 1:
        raise ValueError(f"vander requires a 1-D tensor, got {values.ndim()}")
    columns = values.shape[0] if N is None else _operator.index(N)
    if columns < 0:
        raise ValueError(f"vander requires a non-negative N, got {N}")

    powers = _C.Tensor.arange(0, columns, 1, dtype=str(values.dtype))
    if not increasing:
        powers = _F.flip(powers, [0])
    return _F.pow(values.reshape(-1, 1), powers.reshape(1, -1))


# --- the real-valued answers to complex questions ---------------------------


def real(input: object) -> Tensor:
    """The real part, which is the whole of it.

    Every dtype here is real, so this returns its input. The name exists
    because code written against NumPy asks for it defensively, and an
    AttributeError is a worse answer than the correct one.
    """

    return _atleast_tensor(input)


def conj(input: object) -> Tensor:
    """The complex conjugate, which for a real tensor is the tensor."""

    return _atleast_tensor(input)


def imag(input: object) -> Tensor:
    """The imaginary part, which is zero everywhere.

    A constant, and detached, because that is what it is: the imaginary part of
    a real tensor is zero for every input and carries no gradient. Written as
    `input * 0` instead it would inherit the graph -- and answer NaN for an
    infinite input, since `inf * 0` is not zero.
    """

    return _C.Tensor.zeros_like(_atleast_tensor(input), requires_grad=False)


def angle(input: object) -> Tensor:
    """The argument of each element: `0` where it is positive, `pi` where it is
    negative.

    A real number sits on the real axis, so its argument is one of two values.
    The test is `signbit` rather than `x < 0` because negative zero is on the
    negative side of it -- `angle(-0.0)` is `pi`, as NumPy has it -- and a
    comparison cannot see that.
    """

    tensor = _atleast_tensor(input)
    if "float" not in str(tensor.dtype):
        tensor = tensor.astype("float64")
    # Every branch is a detached constant, so the result carries no gradient.
    # That is the honest answer for a function of the sign *bit*: it is
    # piecewise constant, so its derivative is zero wherever it exists and
    # undefined at the only place it is not.
    shape = list(tensor.shape)
    name = str(tensor.dtype)
    zero = _C.Tensor.zeros_like(tensor, requires_grad=False)
    half_turn = _C.Tensor.full(shape, _math.pi, dtype=name)
    signed = _F.where(_signbit(tensor), half_turn, zero)
    # A NaN has no argument, and answering zero for it would be a claim.
    return _F.where(
        _F.isnan(tensor), _C.Tensor.full(shape, float("nan"), dtype=name), signed
    )


#: Attached to the top level, to `functional` and -- where the first argument
#: is the tensor -- to `Tensor`. See `_elementwise._ELEMENTWISE`.
_MATRIX = (
    "addmm",
    "angle",
    "baddbmm",
    "conj",
    "imag",
    "inner",
    "inverse",
    "logdet",
    "mm",
    "mv",
    "numel",
    "pinverse",
    "real",
    "renorm",
    "t",
    "tensordot",
    "vander",
)
