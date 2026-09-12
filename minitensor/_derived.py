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

import numpy as _np

from . import _core as _C
from ._indexing import ravel_multi_index as _ravel_multi_index
from ._indexing import triu_indices as _triu_indices
from ._shape import _atleast_tensor, _normalize_axis

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


def normalize(
    input: object, p: float = 2.0, dim: int = 1, eps: float = 1e-12
) -> Tensor:
    """`input` scaled so each slice along `dim` has unit `p`-norm.

    `eps` is a floor under the norm, not a term added to it: a zero vector
    comes back as zero rather than as a division by zero, and every other
    vector is divided by its own norm exactly. Adding `eps` instead would
    shrink every vector slightly, which is a bias no caller asked for.
    """

    tensor = _require_float(_atleast_tensor(input), "normalize")
    axis = _normalize_axis(dim, tensor.ndim(), "normalize")
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
    than zero. It is here because `torch.nn.functional.pairwise_distance` has
    it and defaults it to the same value, so ported code gets the same numbers.

    The reason it exists there does not apply here. A `p`-norm has no
    derivative at the origin, and PyTorch needs the shift to keep a loss that
    pulls two rows together from producing NaN at the moment it succeeds --
    but this library's `norm` already answers zero for that gradient rather
    than NaN. So `eps=0.0` is safe, gives the true distance, and is the better
    choice for anything not being compared against torch.
    """

    a = _require_float(_atleast_tensor(input), "pairwise_distance")
    b = _require_float(_atleast_tensor(other), "pairwise_distance")
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

    Both spellings are taken: a flat sequence of `2 * dims` numbers, which is
    what `torch.histogramdd` uses, and a sequence of pairs, which is what this
    library's own one-dimensional `histogram` generalises to. They cannot be
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
        # A column that never varies would otherwise have no width to divide;
        # widening by half on each side is what `numpy` does for it too.
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


def ptp(input: object, dim: int | None = None, keepdim: bool = False) -> Tensor:
    """The peak-to-peak span: the largest value less the smallest.

    Two reductions rather than one pass, which is what `amax` and `amin`
    already are; the point of the name is that a span is what was wanted.
    """

    tensor = _atleast_tensor(input)
    if dim is None:
        return _F.amax(tensor) - _F.amin(tensor)
    axis = _normalize_axis(dim, tensor.ndim(), "ptp")
    return _F.amax(tensor, axis, keepdim) - _F.amin(tensor, axis, keepdim)


def average(
    input: object,
    dim: int | None = None,
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
    the reduced axis -- NumPy's rule, and the one that makes
    `average(x, dim=0, weights=[1, 2, 3])` mean what it looks like.

    With `returned=True` the weight total comes back alongside, which is what a
    caller combining averages needs and cannot recover afterwards.
    """

    tensor = _atleast_tensor(input)
    if weights is None:
        result = (
            _F.mean(tensor)
            if dim is None
            else _F.mean(
                tensor, _normalize_axis(dim, tensor.ndim(), "average"), keepdim
            )
        )
        if not returned:
            return result
        count = (
            tensor.numel()
            if dim is None
            else tensor.shape[_normalize_axis(dim, tensor.ndim(), "average")]
        )
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
        return (
            (_F.sum(tensor * weight) / total, total)
            if returned
            else _F.sum(tensor * weight) / total
        )

    axis = _normalize_axis(dim, tensor.ndim(), "average")
    if weight.ndim() == 1 and tensor.ndim() != 1:
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

    total = _F.sum(tensor * weight, axis, keepdim)
    divisor = _F.sum(weight.expand(list(tensor.shape)), axis, keepdim)
    return (total / divisor, divisor) if returned else total / divisor


def percentile(
    input: object,
    q: object,
    dim: int | None = None,
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
    dim: int | None = None,
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


def histogram_bin_edges(
    input: object,
    bins: object = 10,
    range: object = None,
    weights: object | None = None,
) -> Tensor:
    """The edges `histogram` would use, without counting anything.

    For choosing one set of edges and reusing it across several tensors, which
    is the only way two histograms are comparable. `weights` is accepted and
    ignored, as it is in NumPy: no edge rule here depends on them.
    """

    del weights
    return _F.histogram(_atleast_tensor(input), bins, range, None, False)[1]


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
    total after it. With no `dim` the tensor is flattened first, as NumPy does.
    """

    tensor = _atleast_tensor(input)
    filled = _F.nan_to_num(tensor) if "float" in str(tensor.dtype) else tensor
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
    length = flat.shape[0]
    pieces = []
    if to_begin is not None:
        pieces.append(_atleast_tensor(to_begin).reshape(-1).astype(str(flat.dtype)))
    if length > 1:
        pieces.append(
            _F.narrow(flat, 0, 1, length - 1) - _F.narrow(flat, 0, 0, length - 1)
        )
    elif not pieces and to_end is None:
        return _F.narrow(flat, 0, 0, 0)
    if to_end is not None:
        pieces.append(_atleast_tensor(to_end).reshape(-1).astype(str(flat.dtype)))
    if not pieces:
        return _F.narrow(flat, 0, 0, 0)
    return pieces[0] if len(pieces) == 1 else _F.cat(pieces)
