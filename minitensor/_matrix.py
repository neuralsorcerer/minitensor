# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Matrix products, inverses and rescalings, in terms of what already exists.

`matmul`, `solve` and `svd` are the kernels. Everything here is one of those
pointed at a rearranged operand: `mm` and `mv` are `matmul` with the ranks
checked, `tensordot` is `matmul` with the contracted axes moved to the end and
flattened, and `inverse` and `pinverse` are the `torch` spellings of `inv` and
`pinv`. None of them earns a kernel, so none gets one, and each inherits the
accuracy and the gradient of the one underneath -- which is the point, and the
reason the two named after another library are forwarded rather than written
again here.

`matrix_exp` is the one that looks like it should need a kernel and does not.
Scaling and squaring with a Pade approximant is a real algorithm, but every
step of it is a `matmul`, a `solve` or a scalar multiply -- so it is written
here, and its gradient is the derivative of the approximant that was actually
evaluated rather than a second formula that has to be kept in agreement with
the first.
"""

from __future__ import annotations

import math as _math
import operator as _operator

from . import _core as _C
from ._elementwise import signbit as _signbit
from ._shape import _atleast_tensor, _element_count, _normalize_axis

Tensor = _C.Tensor
_F = _C.functional


def _matrix_stack(input: object, name: str) -> Tensor:
    """`input` as a stack of matrices: at least two dimensions."""

    matrix = _atleast_tensor(input)
    if matrix.ndim() < 2:
        raise ValueError(
            f"{name} requires at least two dimensions, got {matrix.ndim()}"
        )
    return matrix


def _square_matrix(input: object, name: str) -> tuple[Tensor, int]:
    """`input` as a stack of square matrices, with the side length."""

    matrix = _matrix_stack(input, name)
    size = int(matrix.shape[-1])
    if int(matrix.shape[-2]) != size:
        raise ValueError(
            f"{name} requires square matrices, got {matrix.shape[-2]} by {size}"
        )
    return matrix, size


def _identity_like(matrix: Tensor, size: int) -> Tensor:
    """The `size` identity, in the dtype and on the device of `matrix`."""

    return _C.Tensor.eye(size, dtype=str(matrix.dtype), device=_C.Device(matrix.device))


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


def _contraction_axes(
    dims: object, left_rank: int, right_rank: int
) -> tuple[list[int], list[int]]:
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

    left_list = [
        _normalize_axis(a, left_rank, "tensordot") for a in _as_sequence(left_axes)
    ]
    right_list = [
        _normalize_axis(a, right_rank, "tensordot") for a in _as_sequence(right_axes)
    ]
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

    result_shape = [left_shape[i] for i in left_kept] + [
        right_shape[i] for i in right_kept
    ]
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

    The spelling `torch.inverse` uses, for `inv`. It used to be a second
    implementation instead -- `solve` against an identity built here -- and the
    identity was built flat, so it could not be paired with a stack of matrices
    and every batched call raised a shape mismatch against a docstring that
    promised a stack. `inv` solves against an identity too, and does it for
    each matrix in the stack.

    A caller who wants `inverse(A) @ b` should ask `solve(A, b)` instead -- it
    is the same answer without forming the inverse, and it is both faster and
    better conditioned.
    """

    matrix, _ = _square_matrix(input, "inverse")
    return _F.inv(matrix)


def pinverse(input: object, rcond: float = 1e-15) -> Tensor:
    """The Moore-Penrose pseudo-inverse.

    The spelling `torch.pinverse` uses, for `pinv`, and it keeps that name's
    threshold of `1e-15` rather than `pinv`'s own `max(m, n) * eps`. It was a
    second implementation of the same `V diag(1/s) U^T` until it was not: one
    that took a single matrix where `pinv` takes a stack, and that agreed with
    it to the last bit on everything both would accept.
    """

    return _F.pinv(_matrix_stack(input, "pinverse"), rcond)


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
        _C.Tensor.full(
            list(log_absolute.shape), negative_infinity, dtype=str(log_absolute.dtype)
        ),
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


# --- matrix norms and the conditioning they measure -------------------------

#: The orders `matrix_norm` accepts. `"fro"` is elementwise; `1` and `inf` are
#: induced norms, which are absolute column and row sums; the rest are Schatten
#: norms, which only the singular values give.
_MATRIX_NORM_ORDERS = ("fro", "nuc", 1, -1, 2, -2, _math.inf, -_math.inf)


def matrix_norm(input: object, ord: object = "fro", keepdim: bool = False) -> Tensor:
    """A norm of each matrix in the stack, taken over its last two axes.

    `"fro"` is the elementwise 2-norm and `"nuc"` the sum of the singular
    values. `1` and `inf` are the induced norms -- the largest absolute column
    sum and the largest absolute row sum -- and `2` is the largest singular
    value. Each negative order is the same quantity minimised rather than
    maximised, which is what `cond` is built from and is not itself a norm.

    The axes are the last two, as they are for `inverse`, `diagonal` and `svd`.
    `permute` first to use others; `torch.linalg.matrix_norm` takes them as an
    argument instead, and this does not, so that every matrix operation in this
    module reads the same way.

    A condition number in an order other than 2 is
    `matrix_norm(a, ord) * matrix_norm(inverse(a), ord)`. `cond` itself is the
    2-norm one, which has a kernel because the ratio of the extreme singular
    values needs no inverse.
    """

    matrix = _matrix_stack(input, "matrix_norm")
    if isinstance(ord, str):
        if ord not in ("fro", "nuc"):
            raise ValueError(
                f"matrix_norm takes one of {_MATRIX_NORM_ORDERS} as its order, "
                f"got {ord!r}"
            )
        order: object = ord
    else:
        try:
            order = float(ord)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise ValueError(
                f"matrix_norm takes one of {_MATRIX_NORM_ORDERS} as its order, "
                f"got {ord!r}"
            ) from None
        if order not in (1.0, -1.0, 2.0, -2.0, _math.inf, -_math.inf):
            raise ValueError(
                f"matrix_norm takes one of {_MATRIX_NORM_ORDERS} as its order, "
                f"got {ord!r}"
            )

    if order == "fro":
        result = _F.norm(matrix, 2.0, [-2, -1])
    elif order in ("nuc", 2.0, -2.0):
        values = _F.svdvals(matrix)
        if order == "nuc":
            result = _F.sum(values, -1)
        else:
            # `svdvals` is descending, so the ends are the extremes.
            result = _F.amax(values, -1) if order == 2.0 else _F.amin(values, -1)
    else:
        # An induced 1-norm sums down the columns and an induced inf-norm
        # across the rows; the sign of the order says whether to take the
        # largest of those sums or the smallest.
        summed = _F.sum(_F.abs(matrix), -2 if abs(order) == 1.0 else -1)
        result = _F.amax(summed, -1) if order > 0 else _F.amin(summed, -1)

    if not keepdim:
        return result
    return result.reshape(list(result.shape) + [1, 1])


# --- the matrix exponential -------------------------------------------------

#: The largest 1-norm each Pade degree handles at the precision named, from
#: Higham, "The scaling and squaring method for the matrix exponential
#: revisited" (2005), tables 2.3 and 2.4. Below a threshold the approximant is
#: accurate to the unit roundoff of that precision on its own; above the last
#: one the matrix is halved until it is not.
_PADE_THRESHOLDS = {
    "float64": (
        (3, 1.495585217958292e-2),
        (5, 2.539398330063230e-1),
        (7, 9.504178996162932e-1),
        (9, 2.097847961257068e0),
        (13, 5.371920351148152e0),
    ),
    # Single precision stops at degree 7 -- not to save work, but because the
    # degree-13 coefficients run to 6e16, and adding a term that size to one of
    # order 1 in float32 discards the smaller one entirely.
    "float32": (
        (3, 4.258730016922831e-1),
        (5, 1.880152677804762e0),
        (7, 3.925724783138660e0),
    ),
}

#: The Pade numerator coefficients b_0 .. b_m of each degree.
_PADE_COEFFICIENTS = {
    3: (120.0, 60.0, 12.0, 1.0),
    5: (30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0),
    7: (17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0, 1.0),
    9: (
        17643225600.0,
        8821612800.0,
        2075673600.0,
        302702400.0,
        30270240.0,
        2162160.0,
        110880.0,
        3960.0,
        90.0,
        1.0,
    ),
    13: (
        64764752532480000.0,
        32382376266240000.0,
        7771770303897600.0,
        1187353796428800.0,
        129060195264000.0,
        10559470521600.0,
        670442572800.0,
        33522128640.0,
        1323241920.0,
        40840800.0,
        960960.0,
        16380.0,
        182.0,
        1.0,
    ),
}


def _pade_halves(
    matrix: Tensor, degree: int, identity: Tensor
) -> tuple[Tensor, Tensor]:
    """The odd and even halves `U` and `V` of the degree-`m` Pade approximant.

    `exp(A) ~ (V - U)^-1 (V + U)`, where `U` collects the odd powers of `A` and
    `V` the even ones. Both are built from `A^2` rather than from `A`, so a
    degree-`m` approximant costs about `m / 2` products instead of `m`.
    """

    coefficients = _PADE_COEFFICIENTS[degree]
    squared = _F.matmul(matrix, matrix)

    if degree == 13:
        # Higham's grouping for the top degree: three powers and two products
        # rather than six powers, which is where most of its cost would be.
        fourth = _F.matmul(squared, squared)
        sixth = _F.matmul(fourth, squared)
        odd = _F.matmul(
            sixth,
            sixth * coefficients[13]
            + fourth * coefficients[11]
            + squared * coefficients[9],
        )
        odd = odd + (
            sixth * coefficients[7]
            + fourth * coefficients[5]
            + squared * coefficients[3]
            + identity * coefficients[1]
        )
        even = _F.matmul(
            sixth,
            sixth * coefficients[12]
            + fourth * coefficients[10]
            + squared * coefficients[8],
        )
        even = even + (
            sixth * coefficients[6]
            + fourth * coefficients[4]
            + squared * coefficients[2]
            + identity * coefficients[0]
        )
        return _F.matmul(matrix, odd), even

    odd = identity * coefficients[1]
    even = identity * coefficients[0]
    power = identity
    for exponent in range(2, degree + 1, 2):
        power = _F.matmul(power, squared)
        odd = odd + power * coefficients[exponent + 1]
        even = even + power * coefficients[exponent]
    return _F.matmul(matrix, odd), even


def matrix_exp(input: object) -> Tensor:
    """The matrix exponential, `sum_k A^k / k!`, of each square matrix.

    Not `exp` applied elementwise -- that is `exp`. This is the solution
    operator of `dx/dt = A x`, so `matrix_exp(A t) @ x0` is where a linear
    system started at `x0` has got to by time `t`.

    Computed by scaling and squaring: halve `A` until its 1-norm is small
    enough for a Pade approximant to be accurate to the unit roundoff, evaluate
    the approximant, and square the result back. The degree and the number of
    halvings come from Higham's 2005 analysis, and the thresholds depend on the
    precision, so a float32 matrix takes a different route than a float64 one
    rather than the same route at a worse answer.

    Every step is a `matmul`, a `solve` or a scalar multiply, so the operation
    is differentiable by composition and the gradient is the exact derivative
    of the approximant that was evaluated. The number of halvings is chosen
    from the norm and then held fixed: it is a discrete choice, constant under
    small changes to `A`, and its derivative is zero wherever it is defined.

    A batch shares one scaling, chosen from the largest norm in it. Scaling a
    matrix more than it needs costs a squaring, not accuracy, which is the
    trade that keeps a batch one sequence of operations rather than many.
    """

    matrix, size = _square_matrix(input, "matrix_exp")
    precision = str(matrix.dtype)
    if precision not in _PADE_THRESHOLDS:
        raise ValueError(f"matrix_exp needs a floating-point matrix, got {precision}")

    # The 1-norm is the largest absolute column sum, and the largest over a
    # batch decides for all of it.
    norm = float(_F.abs(matrix).sum(-2).max().item())
    thresholds = _PADE_THRESHOLDS[precision]
    squarings = 0
    for degree, threshold in thresholds:
        if norm <= threshold:
            break
    else:
        degree, threshold = thresholds[-1]
        # `log2(norm / threshold)` halvings bring the norm under it; the
        # ceiling is what makes "under" hold rather than "close to".
        squarings = max(0, int(_math.ceil(_math.log2(norm / threshold))))
        matrix = matrix * (0.5**squarings)

    identity = _identity_like(matrix, size)
    odd, even = _pade_halves(matrix, degree, identity)
    result = _F.solve(even - odd, even + odd)
    for _ in range(squarings):
        result = _F.matmul(result, result)
    return result


# --- linear systems over more than two axes ---------------------------------


def tensorsolve(a: object, b: object, axes: object = None) -> Tensor:
    """Solve `a x = b` where the contraction runs over several axes at once.

    `a` has the shape of `b` followed by the shape of the answer, and the
    system is the square one you get by flattening each half. `axes` names axes
    of `a` to move to the end first, for when they are not already there.

    This is `solve` with a reshape on each side; the reshape is the whole
    operation, and the gradient is `solve`'s.
    """

    tensor = _atleast_tensor(a)
    rhs = _atleast_tensor(b)
    if axes is not None:
        moved = [
            _normalize_axis(axis, tensor.ndim(), "tensorsolve")
            for axis in _as_sequence(axes)
        ]
        if len(set(moved)) != len(moved):
            raise ValueError(f"tensorsolve was given a repeated axis in {axes}")
        order = [axis for axis in range(tensor.ndim()) if axis not in moved] + moved
        tensor = _F.permute(tensor, order)

    if tensor.ndim() < rhs.ndim():
        raise ValueError(
            f"tensorsolve needs a coefficient tensor of at least the rank of the "
            f"right-hand side, got {tensor.ndim()} against {rhs.ndim()}"
        )
    answer_shape = list(tensor.shape)[rhs.ndim() :]
    unknowns = _element_count(answer_shape)
    if _element_count(tensor.shape) != unknowns * unknowns:
        raise ValueError(
            f"tensorsolve needs a square system: {list(tensor.shape)} does not "
            f"flatten to {unknowns} by {unknowns}"
        )
    if _element_count(rhs.shape) != unknowns:
        raise ValueError(
            f"tensorsolve needs {unknowns} right-hand side values for "
            f"{unknowns} unknowns, got {_element_count(rhs.shape)}"
        )

    solved = _F.solve(tensor.reshape(unknowns, unknowns), rhs.reshape(unknowns, 1))
    return solved.reshape(answer_shape)


def tensorinv(a: object, ind: int = 2) -> Tensor:
    """The inverse of `a` seen as a matrix split at axis `ind`.

    The axes before `ind` are the rows and those after are the columns, so the
    result has them the other way round -- which is what makes
    `tensordot(tensorinv(a), a, ind)` the identity of that shape.
    """

    tensor = _atleast_tensor(a)
    split = _operator.index(ind)
    if not 0 < split < tensor.ndim():
        raise ValueError(
            f"tensorinv splits at an axis strictly inside the tensor, and {ind} "
            f"is not one for a rank of {tensor.ndim()}"
        )
    shape = list(tensor.shape)
    rows, columns = _element_count(shape[:split]), _element_count(shape[split:])
    if rows != columns:
        raise ValueError(
            f"tensorinv needs the two halves of {shape} to have the same number "
            f"of elements, got {rows} and {columns}"
        )
    return inverse(tensor.reshape(rows, columns)).reshape(shape[split:] + shape[:split])


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
    "matrix_exp",
    "matrix_norm",
    "mm",
    "mv",
    "numel",
    "pinverse",
    "real",
    "renorm",
    "t",
    "tensordot",
    "tensorinv",
    "tensorsolve",
    "vander",
)
