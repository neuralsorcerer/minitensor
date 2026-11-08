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

import builtins
import math as _math
import operator as _operator

from . import _core as _C
from ._elementwise import frexp as _frexp
from ._indexing import ravel_multi_index as _ravel_multi_index
from ._indexing import triu_indices as _triu_indices
from ._shape import (
    _atleast_tensor,
    _beside,
    _flat,
    _normalize_axis,
    _normalize_axis_tuple,
    _operand_pair,
    _promote_pair,
    _promoted_dtype,
)
from ._shape import broadcast_shapes as _broadcast_shapes
from ._shape import broadcast_tensors as _broadcast_tensors
from .autograd import Function as _Function

Tensor = _C.Tensor
_F = _C.functional


def _require_float(tensor: Tensor, name: str) -> Tensor:
    """`tensor` as a float, widening an integer rather than refusing it.

    A distance between integer points, a normalized integer vector and an
    integral of integers are all real numbers, so these take an integer
    argument by widening it -- to `float32` for `int32` and `float64` for
    `int64`, the width `mean` widens to and the one the element-wise maths
    uses. A `bool` has no width to take and is still refused, as it is there.
    """

    dtype = str(tensor.dtype)
    if "float" in dtype:
        return tensor
    widened = {"int32": "float32", "int64": "float64"}.get(dtype)
    if widened is not None:
        return tensor.astype(widened)
    raise ValueError(f"{name} requires a floating point tensor, got {tensor.dtype}")


def outer(input: object, other: object) -> Tensor:
    """The outer product of two flattened tensors: `out[i, j] = a[i] * b[j]`."""

    a, b = _operand_pair(input, other)
    return a.reshape(-1, 1) * b.reshape(1, -1)


def vdot(input: object, other: object) -> Tensor:
    """The inner product of two flattened tensors, of any matching shape.

    `dot` insists on 1-D operands; this flattens first, which is the only
    difference between them for real tensors.
    """

    a, b = _operand_pair(input, other)
    if a.numel() != b.numel():
        raise ValueError(
            f"vdot needs the same number of elements in each operand, "
            f"got {a.numel()} and {b.numel()}"
        )
    return _F.dot(_flat(a), _flat(b))


def kron(input: object, other: object) -> Tensor:
    """The Kronecker product: each element of `input` scaling a copy of `other`.

    Built as one broadcast product rather than a loop over blocks. Interleaving
    a length-1 axis into each operand puts each element of `input` against a
    whole copy of `other`, and the final reshape merges the interleaved pairs
    back into single axes.
    """

    a, b = _operand_pair(input, other)

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


def _rounded_once(p: object) -> bool:
    """Whether a `p`-distance is taken from the exact differences and rounded
    once: the 1- and the 2-distance."""

    return isinstance(p, (int, float)) and not isinstance(p, bool) and p in (1, 2)


class _Distances(_Function):
    """Distances rounded once -- between corresponding rows, each difference
    shifted by `shift`, or with `cross` between every row of one batch and
    every row of the other -- with the gradient of the norm of the
    difference, zero where two rows coincide."""

    @staticmethod
    def forward(ctx, a, b, p, shift, cross):
        if cross:
            result = _F._cross_distances(a, b, p)
        else:
            result = _F._row_distances(a, b, p, shift)
        ctx.save_for_backward(a, b)
        ctx.p, ctx.shift, ctx.cross, ctx.result = p, shift, cross, result
        return result

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        if ctx.cross:
            difference = a.unsqueeze(-2) - b.unsqueeze(-3)
        else:
            difference = a - b + ctx.shift if ctx.shift else a - b
        if ctx.p == 1:
            direction = _F.sign(difference)
        else:
            length = ctx.result.unsqueeze(-1)
            direction = _F.where(length == 0, difference * 0, difference / length)
        weighted = grad_output.unsqueeze(-1) * direction
        if ctx.cross:
            return weighted.sum(-2), -weighted.sum(-3), None, None, None
        return weighted, -weighted, None, None, None


def dist(input: object, other: object, p: float = 2.0) -> Tensor:
    """The `p`-norm of the difference: how far apart two tensors are.

    Integer inputs widen, as they do for every other distance here. This was
    the one that did not: it reached `norm` with an integer difference and
    came back with "norm requires floating point tensors", while `cdist` --
    the same distance, taken pairwise -- answered.
    """

    a, b = _operand_pair(input, other)
    a, b = _require_float(a, "dist"), _require_float(b, "dist")
    if _rounded_once(p):
        a, b = _broadcast_tensors(*_promote_pair(a, b))
        whole = [1, int(a.numel())]
        distance = _Distances.apply(
            a.reshape(whole), b.reshape(whole), float(p), 0.0, False
        )
        return distance.reshape([])
    return _F.norm(a - b, p)


def cdist(input: object, other: object, p: float = 2.0) -> Tensor:
    """Every pairwise `p`-distance between the rows of two batches.

    `input` is `(..., n, d)` and `other` is `(..., m, d)`; the result is
    `(..., n, m)`. For `p` 1 and 2 each distance is the float nearest the exact
    one, taken a pair of rows at a time. Any other `p` forms the difference in
    full, at `n * m * d` elements of memory -- fine for the batch sizes a
    distance matrix is usually wanted at, and the reason to reach for `matmul`
    instead when it is not.
    """

    a, b = _operand_pair(input, other)
    a, b = _require_float(a, "cdist"), _require_float(b, "cdist")
    if a.ndim() < 2 or b.ndim() < 2:
        raise ValueError("cdist requires tensors with at least two dimensions")
    if a.shape[-1] != b.shape[-1]:
        raise ValueError(
            f"cdist needs matching feature counts, got {a.shape[-1]} and {b.shape[-1]}"
        )

    if _rounded_once(p):
        a, b = _promote_pair(a, b)
        batch = list(_broadcast_shapes(tuple(a.shape)[:-2], tuple(b.shape)[:-2]))
        rows, others, width = int(a.shape[-2]), int(b.shape[-2]), int(a.shape[-1])
        count = _math.prod(batch)
        a = a.expand([*batch, rows, width]).reshape([count, rows, width])
        b = b.expand([*batch, others, width]).reshape([count, others, width])
        distances = _Distances.apply(a, b, float(p), 0.0, True)
        return distances.reshape([*batch, rows, others])
    return _F.norm(a.unsqueeze(-2) - b.unsqueeze(-3), p, [-1])


class _Normalized(_Function):
    """Each slice along `axis` over the greater of its 1- or 2-norm and
    `eps`, each element rounded once, with the gradient of that quotient."""

    @staticmethod
    def forward(ctx, tensor, p, axis, eps):
        result = _F._normalized(tensor, p, axis, eps)
        ctx.save_for_backward(tensor)
        ctx.p, ctx.axis, ctx.eps, ctx.result = p, axis, eps, result
        return result

    @staticmethod
    def backward(ctx, grad_output):
        (tensor,) = ctx.saved_tensors
        norm = _F.norm(tensor, ctx.p, [ctx.axis], True)
        length = _F.clamp(norm, ctx.eps, None)
        # Above the floor the length moves with the slice: along its own
        # direction for the 2-norm, and along the signs for the 1-norm. A
        # NaN `eps` is no floor, as it is in the forward.
        direction = ctx.result if ctx.p == 2 else _F.sign(tensor)
        along = _F.sum(ctx.result * grad_output, [ctx.axis], True)
        moving = (grad_output - direction * along) / length
        floored = _F.where(norm < ctx.eps, grad_output / length, moving)
        return floored, None, None, None


def normalize(
    input: object, p: float = 2.0, dim: int = 1, eps: float = 1e-12
) -> Tensor:
    """`input` scaled so each slice along `dim` has unit `p`-norm.

    `eps` is a floor under the norm, not a term added to it: a zero vector
    comes back as zero rather than as a division by zero, and every other
    vector is divided by its own norm exactly. Adding `eps` instead would
    shrink every vector slightly, which is a bias no caller asked for.

    For `p` 1 and 2 each element is the float nearest the exact quotient of
    the element by the exact norm, or by `eps` as the dtype holds it where
    the norm is smaller; any other `p` divides by the norm as computed.
    """

    tensor = _require_float(_atleast_tensor(input), "normalize")
    axis = _normalize_axis(dim, tensor.ndim(), "normalize")
    if _rounded_once(p):
        return _Normalized.apply(tensor, float(p), axis, float(eps))
    lengths = _F.clamp(_F.norm(tensor, p, [axis], True), float(eps), None)
    return tensor / lengths


def pairwise_distance(
    input: object,
    other: object,
    p: float = 2.0,
    eps: float = 1e-6,
    keepdim: bool = False,
) -> Tensor:
    """The `p`-distance between corresponding rows, one number per row.

    `cdist` gives every pair; this gives the diagonal of that, which is what a
    loss over matched pairs wants and costs `n` distances rather than `n * m`.
    The operands are broadcast, so a single row can be compared against a
    batch of them.

    `eps` is added to the difference before the norm, which biases every
    distance upward: two identical rows are `eps * d ** (1 / p)` apart rather
    than zero.

    It is not needed for the gradient. A `p`-norm has no derivative at the
    origin, and a shift is the usual way to keep a loss that pulls two rows
    together from producing NaN at the moment it succeeds -- but this library's
    `norm` already answers zero for that gradient rather than NaN. So
    `eps=0.0` is safe and gives the true distance.
    """

    a, b = _operand_pair(input, other)
    a, b = _require_float(a, "pairwise_distance"), _require_float(
        b, "pairwise_distance"
    )
    if _rounded_once(p):
        a, b = _broadcast_tensors(*_promote_pair(a, b))
        if a.ndim() > 0:
            distances = _Distances.apply(a, b, float(p), float(eps), False)
            return distances.unsqueeze(-1) if keepdim else distances
    return _F.norm(a - b + float(eps), p, [-1], keepdim)


def pdist(input: object, p: float = 2.0) -> Tensor:
    """The `p`-distance between every pair of rows, without the repeats.

    An `(n, d)` input gives `n * (n - 1) / 2` distances, ordered by row and
    then by column -- the strict upper triangle of what `cdist(x, x)` would
    give, which is the same numbers with the diagonal and one of each mirrored
    pair dropped.

    It is built from the pairs rather than from that matrix, so it forms
    `n * (n - 1) / 2` differences instead of `n * n` of them: the half that is
    wanted, and not the half that would be discarded.
    """

    tensor = _require_float(_atleast_tensor(input), "pdist")
    if tensor.ndim() != 2:
        raise ValueError(
            f"pdist takes a single matrix of rows, got {tensor.ndim()} dimensions"
        )
    rows = int(tensor.shape[0])
    pairs = _triu_indices(rows, rows, 1)
    left = _F.index_select(tensor, 0, _F.squeeze(_F.narrow(pairs, 0, 0, 1), 0))
    right = _F.index_select(tensor, 0, _F.squeeze(_F.narrow(pairs, 0, 1, 1), 0))
    if _rounded_once(p):
        return _Distances.apply(left, right, float(p), 0.0, False)
    return _F.norm(left - right, p, [-1])


def _histogram_specification(value: object, dims: int, name: str) -> list:
    """One entry per dimension, from one value or a sequence of them."""

    if value is None or isinstance(value, (int, float)) or _is_tensor(value):
        return [value] * dims
    entries = list(value)  # type: ignore[arg-type]
    if len(entries) != dims:
        raise ValueError(
            f"histogramdd expects one {name} per dimension ({dims}), got {len(entries)}"
        )
    return entries


def _is_tensor(value: object) -> bool:
    return isinstance(value, Tensor) or hasattr(value, "__array__")


def _histogram_bounds(bounds: object, dims: int) -> list:
    """`range` as one `(low, high)` pair per dimension, or `None` for each.

    Both spellings are taken: a flat sequence of `2 * dims` numbers, and a
    sequence of pairs, which is what the one-dimensional `histogram`
    generalises to. They cannot be
    confused, because one has sequences in it and the other does not.
    """

    if bounds is None:
        return [None] * dims
    entries = list(bounds)  # type: ignore[arg-type]
    if entries and not isinstance(entries[0], (list, tuple)):
        if len(entries) != 2 * dims:
            raise ValueError(
                f"histogramdd expects a flat range of {2 * dims} values or "
                f"{dims} pairs, got {len(entries)} values"
            )
        entries = [(entries[2 * d], entries[2 * d + 1]) for d in range(dims)]
    if len(entries) != dims:
        raise ValueError(
            f"histogramdd expects one range per dimension ({dims}), got {len(entries)}"
        )
    return [(float(low), float(high)) for low, high in entries]


def _histogram_edges(column: Tensor, specification: object, bounds: object) -> Tensor:
    """The bin edges for one dimension, from a count or given outright."""

    if _is_tensor(specification):
        edges = _atleast_tensor(specification)
        if edges.ndim() != 1 or int(edges.shape[0]) < 2:
            raise ValueError(
                "histogramdd needs at least two edges per dimension, got "
                f"{list(edges.shape)}"
            )
        return edges

    count = _operator.index(specification)
    if count < 1:
        raise ValueError(f"histogramdd requires at least one bin, got {count}")
    if bounds is None:
        low, high = float(column.min().item()), float(column.max().item())
    else:
        low, high = bounds
    if low == high:
        # A column that never varies would otherwise have no width to divide,
        # so it is widened by half on each side.
        low, high = low - 0.5, high + 0.5
    if not low < high:
        raise ValueError(f"histogramdd needs an increasing range, got ({low}, {high})")
    return _C.Tensor.linspace(low, high, count + 1, dtype=str(column.dtype))


def histogramdd(
    input: object,
    bins: object = 10,
    range: object = None,
    weight: object | None = None,
    density: bool = False,
) -> tuple[Tensor, list[Tensor]]:
    """The joint histogram of `(points, dimensions)` samples, and its edges.

    `histogram` counts along a line; this counts in a box. `bins` is a count for
    every dimension, one count each, or the edges themselves; `range` bounds
    each dimension when the edges are to be computed.

    Which cell a point falls in is the same question in every dimension, asked
    once per axis and then combined -- so the axes are bucketed separately and
    `ravel_multi_index` folds the coordinates into the one flat position that
    `bincount` counts. No cell is ever visited, and nothing is nested: the cost
    is the samples, not the grid, which matters because the grid is the thing
    that grows exponentially.

    A point outside the edges of any dimension is dropped, and the last bin of
    each holds its own right edge, both as they are in `histogram`. `density`
    divides each cell by the total and by its own volume, so the result
    integrates to one over cells that need not be equal.
    """

    sample = _require_float(_atleast_tensor(input), "histogramdd")
    if sample.ndim() == 1:
        sample = sample.reshape(int(sample.shape[0]), 1)
    if sample.ndim() != 2:
        raise ValueError(
            f"histogramdd takes a (points, dimensions) sample, got "
            f"{sample.ndim()} dimensions"
        )
    points, dims = (int(size) for size in sample.shape)

    specifications = _histogram_specification(bins, dims, "bin count")
    bounds = _histogram_bounds(range, dims)

    edges: list[Tensor] = []
    located: list[Tensor] = []
    inside: Tensor | None = None
    for axis in builtins.range(dims):
        column = _F.squeeze(_F.narrow(sample, 1, axis, 1), 1)
        boundary = _histogram_edges(column, specifications[axis], bounds[axis])
        edges.append(boundary)
        last = int(boundary.shape[0]) - 2

        # `searchsorted` counts the edges at or below the value, so one less is
        # the bin -- except at the top edge, which the closing of the last bin
        # puts inside it rather than past it.
        cell = _F.searchsorted(boundary, column, True) - 1
        at_top = column == _F.narrow(boundary, 0, last + 1, 1)
        cell = _F.where(at_top, _C.Tensor.full([1], last, dtype=str(cell.dtype)), cell)

        within = (cell >= 0) & (cell <= last)
        inside = within if inside is None else inside & within
        located.append(_F.clamp(cell, 0, last))

    sizes = tuple(int(boundary.shape[0]) - 1 for boundary in edges)
    # A dropped point is counted with a weight of zero rather than removed,
    # which keeps this one pass over the samples with no compaction in it.
    kept = inside.astype(str(sample.dtype))
    if weight is not None:
        kept = kept * _atleast_tensor(weight).reshape(points)
    counts = _F.bincount(
        _ravel_multi_index(located, sizes), kept, _math.prod(sizes)
    ).reshape(list(sizes))

    if not density:
        return counts, edges

    total = counts.sum()
    volume = None
    for axis, boundary in enumerate(edges):
        shape = [1] * dims
        shape[axis] = sizes[axis]
        widths = diff(boundary).reshape(shape)
        volume = widths if volume is None else volume * widths
    return counts / (total * volume), edges


def _diff_end(end: object, tensor: Tensor, axis: int, name: str) -> Tensor:
    """`prepend` or `append` as a block that joins `tensor` along `axis`.

    A scalar fills one slice; anything else must have the tensor's rank and
    match it on every other axis, since there is no single way to line up a
    lower-rank block against the axes it leaves out.
    """

    block = _beside(end, tensor)
    shape = list(tensor.shape)
    if block.ndim() == 0:
        shape[axis] = 1
        return block.expand(shape).contiguous()
    other = [size for i, size in enumerate(block.shape) if i != axis]
    if block.ndim() != tensor.ndim() or other != [
        size for i, size in enumerate(shape) if i != axis
    ]:
        raise ValueError(
            f"diff: {name} has shape {tuple(block.shape)}, which does not join a "
            f"tensor of shape {tuple(shape)} along dim {axis}; give a scalar or "
            "a tensor matching every other dimension"
        )
    return block


def diff(
    input: object,
    n: int = 1,
    dim: int = -1,
    prepend: object | None = None,
    append: object | None = None,
) -> Tensor:
    """The `n`-th discrete difference along `dim`.

    Each pass shortens the axis by one, so `n` passes over a length-`k` axis
    leave `max(k - n, 0)` elements. `prepend` and `append` are joined on along
    `dim` first -- a scalar as one slice -- which is how a difference is made
    to keep the length of what it was taken from.
    """

    tensor = _atleast_tensor(input)
    if tensor.ndim() == 0:
        raise ValueError("diff requires a tensor with at least one dimension")

    order = _operator.index(n)
    if order < 0:
        raise ValueError(f"diff requires a non-negative order, got {order}")

    axis = _normalize_axis(dim, tensor.ndim(), "diff")
    result = None
    if prepend is not None or append is not None:
        # The ends promote with the tensor rather than being cast onto its
        # dtype, so a fractional end against an integer tensor keeps its value.
        ends = [
            (_diff_end(end, tensor, axis, name), first)
            for end, name, first in (
                (prepend, "prepend", True),
                (append, "append", False),
            )
            if end is not None
        ]
        promoted = tensor
        for block, _ in ends:
            promoted, _ = _promote_pair(promoted, block)
        working = str(promoted.dtype)
        # A float32 result beside integers is worked in float64, which holds
        # every int32 and float32 exactly, and rounded to float32 once at the
        # end. Joined at float32, two large integers were each rounded before
        # their difference was taken: `diff([16777217, 16777219], append=0.5)`
        # began from 4 rather than 2.
        pieces = [tensor] + [block for block, _ in ends]
        if working == "float32" and any(_is_integral(piece) for piece in pieces):
            result, working = working, "float64"
        pieces = [block.astype(working) for block, first in ends if first]
        pieces.append(tensor.astype(working))
        pieces += [block.astype(working) for block, first in ends if not first]
        tensor = _F.cat(pieces, axis)
    tensor = _nth_difference(tensor, order, axis)
    return tensor if result is None else tensor.astype(result)


def _is_integral(tensor: Tensor) -> bool:
    return str(tensor.dtype) in ("int32", "int64")


def _nth_difference(tensor: Tensor, order: int, axis: int) -> Tensor:
    """`order` first differences of `tensor` along `axis`."""

    if order and not tensor.requires_grad and str(tensor.dtype) != "bool":
        # Nothing to record, so each pass is the engine's first difference:
        # one parallel subtraction of the shifted rows from the unshifted
        # ones, with neither copied out first, on every core. Over a million
        # float64 it takes 288us; below, each pass copies both
        # halves out before subtracting them. The engine path stays for a
        # gradient to be recorded, and for `bool`, which it refuses.
        for _ in range(order):
            if tensor.shape[axis] == 0:
                break
            tensor = _F._first_difference(tensor, axis)
        return tensor
    for _ in range(order):
        length = tensor.shape[axis]
        if length == 0:
            break
        tensor = _F.narrow(tensor, axis, 1, length - 1) - _F.narrow(
            tensor, axis, 0, length - 1
        )
    return tensor


class _Trapezoid(_Function):
    """The trapezoidal integral along an axis, rounded once, with the
    gradient of the exact integral."""

    @staticmethod
    def forward(ctx, values, positions, step, axis):
        ctx.save_for_backward(values)
        ctx.positions, ctx.step, ctx.axis = positions, step, axis
        return _F._trapezoid(values, positions, step, axis)

    @staticmethod
    def backward(ctx, grad_output):
        (values,) = ctx.saved_tensors
        axis, length = ctx.axis, values.shape[ctx.axis]
        dtype = str(values.dtype)
        grad = grad_output.unsqueeze(axis)
        along = [1] * values.ndim()
        along[axis] = length
        if ctx.positions is None:
            # Half a step at each end and a whole one inside.
            weights = [ctx.step] * length
            weights[0] = weights[-1] = ctx.step / 2
            weights = Tensor(weights, dtype=dtype).reshape(along)
            return grad * weights, None, None, None
        positions = ctx.positions.detach()
        shared = positions.shape != values.shape
        if shared:
            positions = positions.reshape(along)
        widths = diff(positions, 1, axis)
        edge = _F.narrow(widths, axis, 0, 1) * 0.0
        # Each sample's weight: half the intervals either side of it.
        weights = (_F.cat([edge, widths], axis) + _F.cat([widths, edge], axis)) * 0.5
        grad_values = grad * weights
        grad_positions = None
        if ctx.needs_input_grad[1]:
            samples = values.detach()
            heights = (
                _F.narrow(samples, axis, 0, length - 1)
                + _F.narrow(samples, axis, 1, length - 1)
            ) * 0.5
            edge = _F.narrow(heights, axis, 0, 1) * 0.0
            # A position moves its interval on the left one way and the one
            # on its right the other.
            moving = _F.cat([edge, heights], axis) - _F.cat([heights, edge], axis)
            grad_positions = grad * moving
            if shared:
                others = [index for index in range(values.ndim()) if index != axis]
                if others:
                    grad_positions = _F.sum(grad_positions, others)
                grad_positions = grad_positions.reshape([length])
        return grad_values, grad_positions, None, None


def trapezoid(
    y: object, x: object | None = None, dx: float = 1.0, dim: int = -1
) -> Tensor:
    """The trapezoidal integral of `y` along `dim`.

    With `x` given, the sample spacing comes from it and may be uneven: a 1-D
    `x` is shared by every slice along `dim`, and any other broadcasts against
    `y` with a position for each sample. With only `dx`, the samples are
    evenly spaced, `dx` read at the samples' width as a number beside them is.
    Each integral is the float nearest the exact one, not the heights, the
    widths, their products and their sum each rounded on the way.
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

    if x is None:
        return _Trapezoid.apply(values, None, float(dx), axis)

    positions = _require_float(_atleast_tensor(x), "trapezoid")
    dtype = _promoted_dtype(values, positions)
    values, positions = values.astype(dtype), positions.astype(dtype)
    # A 1-D coordinate is shared by every slice, along the axis it measures;
    # any other broadcasts against `y`, keeping a position for each sample.
    # Either way `dim` names an axis of `y`, counted from the end once
    # broadcasting has put axes in front.
    from_end = values.ndim() - axis
    if positions.ndim() == 1:
        given = positions.shape[0]
    elif positions.ndim() >= from_end:
        given = positions.shape[positions.ndim() - from_end]
    else:
        given = None
    if given != length:
        raise ValueError(
            f"trapezoid needs a position for each of the {length} samples along "
            f"dim {dim}, got positions shaped {list(positions.shape)}"
        )
    if positions.ndim() > 1:
        values, positions = _broadcast_tensors(values, positions)
        values, positions = values.contiguous(), positions.contiguous()
        axis = values.ndim() - from_end
    return _Trapezoid.apply(values, positions, float(dx), axis)


# `trapz` is the other common name, and enough code says it for the alias to
# be worth the line.
trapz = trapezoid


class _Covariance(_Function):
    """The covariance or correlation matrix of the rows of a 2-D tensor, each
    entry rounded once, with the gradient of the exact matrix."""

    @staticmethod
    def forward(ctx, matrix, correction, frequencies, reliabilities, correlation):
        result = _F._covariance(
            matrix, correction, frequencies, reliabilities, correlation
        )
        ctx.save_for_backward(matrix)
        ctx.correction, ctx.correlation, ctx.result = correction, correlation, result
        ctx.frequencies, ctx.reliabilities = frequencies, reliabilities
        return result

    @staticmethod
    def backward(ctx, grad_output):
        (matrix,) = ctx.saved_tensors
        # Each row scaled by a power of two, exactly: a covariance moves with
        # the scales of its two rows and a correlation not at all, and scaled
        # the arithmetic below stays inside the range however large or small
        # the rows are.
        rows = matrix.detach().astype("float64")
        scales = _row_scales(rows)
        rows = rows / scales
        weights = _cov_weights(ctx.frequencies, ctx.reliabilities, rows.shape[1])
        total = _F.sum(weights)
        centred = rows - _F.sum(rows * weights, [1], True) / total
        if ctx.reliabilities is None:
            scale = total - ctx.correction
        else:
            effective = _F.sum(weights * ctx.reliabilities) / total
            scale = total - ctx.correction * effective
        grad = grad_output.astype("float64")
        if ctx.correlation:
            # Of `r = c / sqrt(c_ii c_jj)`: the entry's own share, and what
            # its row's and its column's variances take from the diagonal.
            covariance = (
                _F.matmul(centred * weights, _F.transpose(centred, 0, 1)) / scale
            )
            variance = _F.diagonal(covariance)
            deviation = _F.sqrt(variance)
            weighted = grad * ctx.result.astype("float64")
            through = (_F.sum(weighted, [1]) + _F.sum(weighted, [0])) / (2.0 * variance)
            grad = grad / outer(deviation, deviation) - _F.diag_embed(through)
        else:
            grad = grad * (scales * _F.transpose(scales, 0, 1))
        moved = _F.matmul(grad + _F.transpose(grad, 0, 1), centred * weights) / scale
        # Less what moving the weighted means takes back: nothing in exact
        # arithmetic, where the deviations' weighted sums are zero.
        moved = moved - _F.sum(moved, [1], True) * (weights / total)
        return (moved / scales).astype(str(matrix.dtype)), None, None, None, None


def _cov_weights(
    frequencies: Tensor | None, reliabilities: Tensor | None, observations: int
) -> Tensor:
    """Each observation's weight in float64: its count times its
    reliability, either alone, or one."""

    if frequencies is None and reliabilities is None:
        return Tensor.ones([observations], dtype="float64")
    if frequencies is None:
        return reliabilities
    if reliabilities is None:
        return frequencies
    return frequencies * reliabilities


def cov(
    input: object,
    correction: int = 1,
    fweights: object | None = None,
    aweights: object | None = None,
) -> Tensor:
    """The covariance matrix of the *rows* of `input`.

    Each row is a variable and each column an observation, the opposite of a
    design matrix's layout. A 1-D input is
    one variable, so the result is its scalar variance.

    `fweights` counts repeats of each observation and `aweights` weights their
    reliability; the difference shows in the denominator, where `aweights`
    reduces the effective sample size rather than the count.

    Each entry is the float nearest the exact covariance of the values as
    given: `(W Σ w x y - Σ w x Σ w y) / D` for weights `w` totalling `W` and
    the divisor `D` the correction leaves, rounded once -- to float32 for a
    float32 or int32 input, and to float64 otherwise. An integer input is
    taken exactly, int64s past `2^53` apart included, and an entry in range
    is the right one however far out of range its products would be. A row
    with a NaN or an infinity gives NaN entries.
    """

    matrix, single_variable, frequencies, reliabilities = _cov_operands(
        input, fweights, aweights, "cov"
    )
    result = _Covariance.apply(
        matrix, float(correction), frequencies, reliabilities, False
    )
    return result.reshape([]) if single_variable else result


def _cov_operands(
    input: object, fweights: object | None, aweights: object | None, name: str
) -> tuple[Tensor, bool, Tensor | None, Tensor | None]:
    """`cov`'s operands, checked: the rows as a 2-D tensor of their own
    dtype, whether there was one variable, and the counts and reliabilities
    in float64."""

    matrix = _atleast_tensor(input)
    if matrix.ndim() > 2:
        raise ValueError(f"{name} requires a 1-D or 2-D tensor")
    # A single variable has a scalar variance, not a one-by-one matrix.
    single_variable = matrix.ndim() <= 1
    if single_variable:
        matrix = matrix.reshape(1, -1)
    if str(matrix.dtype) not in ("int32", "int64"):
        matrix = _require_float(matrix, name)

    observations = matrix.shape[1]
    if observations == 0:
        raise ValueError(f"{name} requires at least one observation")

    frequencies = reliabilities = None
    if fweights is not None:
        frequencies = _check_weights(fweights, observations, "fweights")
    if aweights is not None:
        reliabilities = _check_weights(aweights, observations, "aweights")
    return matrix, single_variable, frequencies, reliabilities


def _row_scales(matrix: Tensor) -> Tensor:
    """For each row of a float matrix, the power of two at or below its largest
    finite magnitude, one for a row with none, as a detached column.

    Dividing by a power of two is exact, and leaves every row's largest finite
    magnitude in `[1, 2)`, where the products of its deviations cannot leave
    the range.
    """

    magnitude = _F.abs(matrix.detach())
    finite = _F.where(_F.isfinite(magnitude), magnitude, Tensor.zeros_like(magnitude))
    largest = _F.amax(finite, [1], True)
    _, exponent = _frexp(largest)
    ones = Tensor.ones_like(largest)
    return _F.where(largest > 0, ones.ldexp(exponent - 1), ones)


def _check_weights(weights: object, observations: int, name: str) -> Tensor:
    tensor = _atleast_tensor(weights)
    if tensor.ndim() != 1 or tensor.shape[0] != observations:
        raise ValueError(
            f"cov requires {name} to be 1-D with one entry per observation "
            f"({observations})"
        )
    # In float64 with the rows, which holds every float32 and every integer
    # weight a count could be.
    return tensor.astype("float64")


def corrcoef(input: object) -> Tensor:
    """The Pearson correlation matrix of the rows of `input`.

    Each entry is the float nearest the exact correlation
    `n_xy / sqrt(n_xx n_yy)`, the `n` the covariances' numerators: in
    `[-1, 1]` with no clamp, one exactly on the diagonal, and NaN for a row
    that never varies, or that holds a NaN or an infinity. A correlation does
    not depend on the scale of a row, and rows whose covariances leave the
    float range correlate like any other. A float32 or int32 input gives
    float32, and any other float64.
    """

    matrix, single_variable, _, _ = _cov_operands(input, None, None, "corrcoef")
    result = _Covariance.apply(matrix, 1.0, None, None, True)
    return result.reshape([]) if single_variable else result


def ptp(input: object, dim: object = None, keepdim: bool = False) -> Tensor:
    """The peak-to-peak span: the largest value less the smallest.

    Two reductions rather than one pass, which is what `amax` and `amin`
    already are; the point of the name is that a span is what was wanted.

    `dim` may name several axes, since both halves can.
    """

    tensor = _atleast_tensor(input)
    if dim is None:
        span = _F.amax(tensor) - _F.amin(tensor)
        # Every axis is reduced, so `keepdim` keeps every one, at length one.
        return span.reshape([1] * tensor.ndim()) if keepdim else span
    axes = list(_normalize_axis_tuple(dim, tensor.ndim(), "ptp"))
    return _F.amax(tensor, axes, keepdim) - _F.amin(tensor, axes, keepdim)


def average(
    input: object,
    dim: object = None,
    weights: object | None = None,
    keepdim: bool = False,
    returned: bool = False,
):
    """The mean, or the weighted mean when `weights` is given.

    Without weights this is `mean`. With them it is
    `sum(a * w) / sum(w)` over the reduced axis, which is not the same as
    weighting after the fact: the divisor is the weight total, so weights that
    do not sum to one still give an average rather than a scaled one.

    `weights` may have the tensor's shape, or be one-dimensional and as long as
    the reduced axis, which is what makes
    `average(x, dim=0, weights=[1, 2, 3])` mean what it looks like. The
    one-dimensional form needs a single axis to line up with, so weighting an
    average over several axes takes weights shaped like the input.

    With `returned=True` the weight total comes back alongside, which is what a
    caller combining averages needs and cannot recover afterwards.
    """

    tensor = _atleast_tensor(input)
    if weights is None:
        axes = (
            None
            if dim is None
            else list(_normalize_axis_tuple(dim, tensor.ndim(), "average"))
        )
        if axes is None:
            result = _F.mean(tensor)
            # Every axis is reduced, so `keepdim` keeps every one, at length one.
            if keepdim:
                result = result.reshape([1] * tensor.ndim())
        else:
            result = _F.mean(tensor, axes, keepdim)
        if not returned:
            return result
        count = tensor.numel()
        if axes is not None:
            count = 1
            for axis in axes:
                count *= tensor.shape[axis]
        return result, Tensor.full(
            list(result.shape), float(count), dtype=str(result.dtype)
        )

    weight = _atleast_tensor(weights)
    if dim is None:
        if list(weight.shape) != list(tensor.shape):
            raise ValueError(
                "average over every axis needs weights shaped like the input, "
                f"got {tuple(weight.shape)} for {tuple(tensor.shape)}"
            )
        total = _F.sum(weight)
        result = _weighted_sum(tensor, weight, None, False, total)
        if keepdim:
            kept = [1] * tensor.ndim()
            result, total = result.reshape(kept), total.reshape(kept)
        return (result, total) if returned else result

    axes = list(_normalize_axis_tuple(dim, tensor.ndim(), "average"))
    if weight.ndim() == 1 and tensor.ndim() != 1:
        if len(axes) != 1:
            raise ValueError(
                "average over several axes needs weights shaped like the "
                f"input, got a one-dimensional {tuple(weight.shape)} for axes "
                f"{tuple(axes)}"
            )
        axis = axes[0]
        if weight.shape[0] != tensor.shape[axis]:
            raise ValueError(
                f"average needs one weight per position along dimension {axis}, "
                f"got {weight.shape[0]} for an axis of {tensor.shape[axis]}"
            )
        # Line the weights up with the reduced axis and leave every other axis
        # to broadcast.
        spread = [1] * tensor.ndim()
        spread[axis] = weight.shape[0]
        weight = weight.reshape(spread)
    elif list(weight.shape) != list(tensor.shape):
        raise ValueError(
            "average needs weights shaped like the input or one-dimensional "
            f"along the reduced axis, got {tuple(weight.shape)} for "
            f"{tuple(tensor.shape)}"
        )

    divisor = _F.sum(weight.expand(list(tensor.shape)), axes, keepdim)
    result = _weighted_sum(tensor, weight, axes, keepdim, divisor)
    return (result, divisor) if returned else result


def _weighted_sum(
    tensor: Tensor, weight: Tensor, axes: list | None, keepdim: bool, divisor: Tensor
) -> Tensor:
    """`sum(tensor * weight) / divisor` over `axes`, every axis for `None`.

    The exact weighted sum over the exact weight total, rounded once to the
    float dtype the pair promotes to, and to float64 for two integer
    operands. Formed in floats, each product, both sums and the quotient
    were rounded on their own: a float32 pair landed several ulps from the
    exact weighted mean, and carried in float64 and narrowed it was still
    rounded twice. Two integer operands were multiplied in their own dtype,
    where a product past it wrapped.
    """

    integral = ("bool", "int32", "int64")
    if str(tensor.dtype) in integral and str(weight.dtype) in integral:
        # Every product exact in 128 bits and both totals in whole limbs.
        values = tensor.astype("int64")
        weights = weight.astype("int64").expand(list(tensor.shape))
        return _F.weighted_mean(values, weights, axes, keepdim, "float64")
    dtype = _promoted_dtype(tensor, weight)
    if dtype in ("float32", "float64") and _held_in_float64(tensor, weight):
        # Read as they are when both are float32, and in float64 otherwise,
        # which holds every value they can have.
        both_single = str(tensor.dtype) == str(weight.dtype) == "float32"
        read = "float32" if both_single else "float64"
        values = tensor.astype(read)
        weights = weight.astype(read).expand(list(tensor.shape))
        if values.requires_grad or weights.requires_grad:
            return _WeightedMean.apply(values, weights, axes, keepdim, dtype)
        return _F.weighted_mean(values, weights, axes, keepdim, dtype)
    product = tensor * weight
    total = _F.sum(product) if axes is None else _F.sum(product, axes, keepdim)
    return total / divisor


def _held_in_float64(*tensors: Tensor) -> bool:
    """Whether float64 holds every value of each tensor: all but an int64
    past 2^53."""

    for tensor in tensors:
        if str(tensor.dtype) == "int64" and tensor.numel() > 0:
            if _F.amax(tensor).item() > 2**53 or _F.amin(tensor).item() < -(2**53):
                return False
    return True


class _WeightedMean(_Function):
    """The weighted mean rounded once, with the gradient of
    `sum(values * weights) / sum(weights)`."""

    @staticmethod
    def forward(ctx, values, weights, axes, keepdim, dtype):
        result = _F.weighted_mean(values, weights, axes, keepdim, dtype)
        ctx.save_for_backward(values, weights)
        ctx.axes, ctx.result = axes, result
        return result

    @staticmethod
    def backward(ctx, grad_output):
        values, weights = ctx.saved_tensors
        axes = list(range(values.ndim())) if ctx.axes is None else list(ctx.axes)
        kept = [1 if axis in axes else size for axis, size in enumerate(values.shape)]
        grad = grad_output.reshape(kept)
        total = _F.sum(weights, axes, True)
        mean = ctx.result.reshape(kept)
        return grad * weights / total, grad * (values - mean) / total


def percentile(
    input: object,
    q: object,
    dim: object = None,
    keepdim: bool = False,
    interpolation: str = "linear",
) -> Tensor:
    """The `q`-th percentile, with `q` in `[0, 100]`.

    `quantile` in the units people quote: the same computation with `q` divided
    by a hundred, so the two cannot drift apart.
    """

    return _F.quantile(
        _atleast_tensor(input),
        _percentile_fraction(q, "percentile"),
        dim,
        keepdim,
        interpolation,
    )


def nanpercentile(
    input: object,
    q: object,
    dim: object = None,
    keepdim: bool = False,
    interpolation: str = "linear",
) -> Tensor:
    """`percentile` ignoring NaN, as `nanquantile` is to `quantile`."""

    return _F.nanquantile(
        _atleast_tensor(input),
        _percentile_fraction(q, "nanpercentile"),
        dim,
        keepdim,
        interpolation,
    )


def _percentile_fraction(q: object, name: str) -> object:
    """`q` in percent, checked and turned into a fraction."""

    if isinstance(q, (builtins.int, builtins.float)) and not isinstance(
        q, builtins.bool
    ):
        if not 0.0 <= builtins.float(q) <= 100.0:
            raise ValueError(f"{name} requires q in [0, 100], got {q}")
        return builtins.float(q) / 100.0

    fractions = _atleast_tensor(q)
    if fractions.numel() and (
        builtins.float(_F.amin(fractions).item()) < 0.0
        or builtins.float(_F.amax(fractions).item()) > 100.0
    ):
        raise ValueError(f"{name} requires every q in [0, 100]")
    return fractions / 100.0


def interp(
    x: object,
    xp: object,
    fp: object,
    left: float | None = None,
    right: float | None = None,
    period: float | None = None,
) -> Tensor:
    """Piecewise linear interpolation of the samples `(xp, fp)` at `x`.

    `xp` must increase. Each point is placed between two samples by binary
    search rather than by scanning, so this is `n log m` in the sample count
    and works on a whole tensor of query points at once.

    Outside the sample range the value is held at the end point, or at `left` /
    `right` when those are given. `period` instead wraps both the queries and
    the samples onto one period, so the ends join up.
    """

    query = _require_float(_atleast_tensor(x).astype("float64"), "interp")
    points = _require_float(_atleast_tensor(xp).astype("float64"), "interp").reshape(-1)
    values = _require_float(_atleast_tensor(fp).astype("float64"), "interp").reshape(-1)

    if points.shape[0] != values.shape[0]:
        raise ValueError(
            f"interp needs as many sample values as sample points, got "
            f"{values.shape[0]} and {points.shape[0]}"
        )
    if points.shape[0] == 0:
        raise ValueError("interp needs at least one sample point")

    if period is not None:
        span = builtins.float(period)
        if span == 0.0:
            raise ValueError("interp requires a non-zero period")
        span = builtins.abs(span)
        # Wrap both onto `[0, period)`, then repeat the first sample past the
        # end and the last one before the start so the wrap has neighbours to
        # interpolate between.
        query = query - _F.floor(query / span) * span
        points = points - _F.floor(points / span) * span
        order = _F.argsort(points)
        points = _F.index_select(points, 0, order)
        values = _F.index_select(values, 0, order)
        points = _F.cat(
            [
                _F.narrow(points, 0, points.shape[0] - 1, 1) - span,
                points,
                _F.narrow(points, 0, 0, 1) + span,
            ]
        )
        values = _F.cat(
            [
                _F.narrow(values, 0, values.shape[0] - 1, 1),
                values,
                _F.narrow(values, 0, 0, 1),
            ]
        )

    if points.shape[0] == 1:
        # One sample is the same value everywhere -- there is no segment to
        # interpolate along, and both ends of the range are that sample.
        held = _F.index_select(values, 0, _atleast_tensor([0]).astype("int64"))
        return held * Tensor.full(list(query.shape), 1.0, dtype="float64")

    count = points.shape[0]
    # `right=True` puts a query that lands exactly on a sample at that sample's
    # own segment, so an exact hit returns its value exactly rather than to
    # within a rounding of the slope.
    upper = _F.clamp(
        _F.searchsorted(points, query.reshape(-1), True).astype("int64"), 1, count - 1
    )
    lower = upper - 1
    x_lo = _F.index_select(points, 0, lower)
    x_hi = _F.index_select(points, 0, upper)
    y_lo = _F.index_select(values, 0, lower)
    y_hi = _F.index_select(values, 0, upper)
    slope = (y_hi - y_lo) / (x_hi - x_lo)
    flat = query.reshape(-1)
    result = y_lo + slope * (flat - x_lo)

    if period is None:
        first = builtins.float(
            _F.index_select(points, 0, _atleast_tensor([0]).astype("int64")).item()
        )
        last = builtins.float(
            _F.index_select(
                points, 0, _atleast_tensor([count - 1]).astype("int64")
            ).item()
        )
        low_value = (
            builtins.float(left)
            if left is not None
            else builtins.float(
                _F.index_select(values, 0, _atleast_tensor([0]).astype("int64")).item()
            )
        )
        high_value = (
            builtins.float(right)
            if right is not None
            else builtins.float(
                _F.index_select(
                    values, 0, _atleast_tensor([count - 1]).astype("int64")
                ).item()
            )
        )
        result = _F.where(
            flat < first,
            Tensor.full(list(flat.shape), low_value, dtype="float64"),
            result,
        )
        result = _F.where(
            flat > last,
            Tensor.full(list(flat.shape), high_value, dtype="float64"),
            result,
        )

    return result.reshape(list(query.shape))


def digitize(input: object, bins: object, right: bool = False) -> Tensor:
    """Which bin each value falls in, as an index into `bins`.

    The bins are the edges, and the answer is how many of them a value is past:
    `0` for a value below every edge and `len(bins)` for one above them all.
    `right` moves which side of an edge counts as inside it.

    The edges may increase or decrease. A decreasing sequence is the same
    question asked from the other end, so it is answered by searching the
    reversed edges and counting back from the total -- which is why this is not
    simply `searchsorted`, whose contract is an increasing sequence.
    """

    values = _atleast_tensor(input)
    edges = _atleast_tensor(bins).reshape(-1)
    if edges.shape[0] < 2:
        # One edge or none is ordered either way; take it as increasing.
        return _F.searchsorted(edges, values, not right).astype("int64")

    ascending = builtins.bool(
        _F.all(
            _F.narrow(edges, 0, 1, edges.shape[0] - 1)
            >= _F.narrow(edges, 0, 0, edges.shape[0] - 1)
        ).item()
    )
    if ascending:
        return _F.searchsorted(edges, values, not right).astype("int64")

    # Counting back from the total flips which side of an edge is inside it,
    # so the search takes the opposite `right` to the one asked for.
    flipped = _F.flip(edges, [0])
    return Tensor.full(
        list(values.shape), edges.shape[0], dtype="int64"
    ) - _F.searchsorted(flipped, values, not right).astype("int64")


def histogram2d(
    x: object,
    y: object,
    bins: object = 10,
    range: object = None,
    weights: object | None = None,
    density: bool = False,
):
    """The joint histogram of two sequences, and the edges it used.

    `histogramdd` over the pair, which is where the counting lives; this is the
    two-dimensional spelling of it, with the edges handed back separately
    rather than as a list.
    """

    first = _atleast_tensor(x).reshape(-1, 1)
    second = _atleast_tensor(y).reshape(-1, 1)
    if first.shape[0] != second.shape[0]:
        raise ValueError(
            f"histogram2d needs the same number of values in each sequence, "
            f"got {first.shape[0]} and {second.shape[0]}"
        )
    counts, edges = histogramdd(
        _F.cat([first, second], 1), bins, range, weights, density
    )
    return counts, edges[0], edges[1]


def nancumsum(input: object, dim: int | None = None) -> Tensor:
    """The running sum along `dim`, treating NaN as zero.

    A NaN contributes nothing and, unlike in `cumsum`, does not poison every
    total after it. With no `dim` the tensor is flattened first.
    """

    tensor = _atleast_tensor(input)
    if "float" in str(tensor.dtype):
        # Only the NaN becomes zero. `nan_to_num` with its defaults also
        # replaces the infinities, with the dtype's finite extremes, so
        # `nancumsum([inf, -inf])` summed 1.8e308 and its negation and
        # answered zero where the running total is `inf` and then NaN.
        zeros = Tensor.full(list(tensor.shape), 0.0, dtype=str(tensor.dtype))
        filled = _F.where(_F.isnan(tensor), zeros, tensor)
    else:
        filled = tensor
    axis = None if dim is None else _normalize_axis(dim, tensor.ndim(), "nancumsum")
    return _F.cumsum(filled, axis)


def nancumprod(input: object, dim: int | None = None) -> Tensor:
    """The running product along `dim`, treating NaN as one."""

    tensor = _atleast_tensor(input)
    if "float" in str(tensor.dtype):
        ones = Tensor.full(list(tensor.shape), 1.0, dtype=str(tensor.dtype))
        filled = _F.where(_F.isnan(tensor), ones, tensor)
    else:
        filled = tensor
    axis = None if dim is None else _normalize_axis(dim, tensor.ndim(), "nancumprod")
    return _F.cumprod(filled, axis)


def ediff1d(
    input: object, to_end: object | None = None, to_begin: object | None = None
) -> Tensor:
    """The differences between consecutive elements of the flattened tensor.

    `diff` along the one axis a flattened tensor has, with optional values
    joined on at either end -- which is what makes this the one to reach for
    when the result has to line up with something of the original length.
    """

    flat = _atleast_tensor(input).reshape(-1)
    if to_begin is None and to_end is None:
        return diff(flat) if flat.shape[0] > 1 else _F.narrow(flat, 0, 0, 0)
    # The joined-on values promote with the differences rather than being cast
    # onto their dtype: `to_begin=2.5` against an integer input used to arrive
    # as 2. Promoting keeps the value, which is what `append` does with the
    # same argument.
    begin, end = (
        None if value is None else _beside(value, flat).reshape(-1)
        for value in (to_begin, to_end)
    )
    promoted = flat
    for block in (begin, end):
        if block is not None:
            promoted, _ = _promote_pair(promoted, block)
    result = str(promoted.dtype)
    length = flat.shape[0]
    pieces = []
    if begin is not None:
        pieces.append(begin.astype(result))
    if length > 1:
        # An integer input's differences are taken exactly -- in int64, where
        # no int32 difference wraps -- and given the result's dtype once.
        # Promoted first, two large integers were each rounded before their
        # difference was taken.
        if _is_integral(flat) and result.startswith("float"):
            pieces.append(diff(flat.astype("int64")).astype(result))
        else:
            pieces.append(diff(flat.astype(result)))
    elif not pieces and end is None:
        return _F.narrow(flat.astype(result), 0, 0, 0)
    if end is not None:
        pieces.append(end.astype(result))
    if not pieces:
        return _F.narrow(flat.astype(result), 0, 0, 0)
    return pieces[0] if len(pieces) == 1 else _F.cat(pieces)
