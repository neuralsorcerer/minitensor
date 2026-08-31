# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Neural-network pieces defined in terms of the kernels that already exist.

`nll_loss` is a gather and a weighted mean, `prelu` is two rectifiers,
`gumbel_softmax` is a softmax of perturbed logits, and the pixel shuffles are
a reshape and a permute. None of them earns a kernel, so none of them gets
one: each is written here, once, in terms of operations whose accuracy and
gradients are already established.

`unfold` and `fold` are the same idea aimed at the largest target in the file.
A convolution is a matrix product once the sliding blocks are laid out as
columns, so an `unfold` plus a `matmul` is a convolution -- and the caller who
writes it that way can vary it however they like without a Rust toolchain. The
layout is a gather at positions computed from the kernel geometry, and because
the backward of a gather is a scatter-add, `fold` comes out as the exact
adjoint of `unfold` without the two sharing a line of code.

`cross_entropy` is the one that does have a kernel, because it has an
analytical backward worth having -- and `nll_loss` is the half of it that takes
log-probabilities it did not compute itself, which is what makes the two
different functions rather than one written twice.
"""

from __future__ import annotations

import math as _math
import operator as _operator

import numpy as _np

from . import _core as _C
from ._sampling import bernoulli as _bernoulli
from ._shape import (
    _atleast_tensor,
    _constant_like,
    _element_count,
    _index_tensor,
    _normalize_axis,
    broadcast_to,
)

Tensor = _C.Tensor
_F = _C.functional


def nll_loss(
    input: object,
    target: object,
    weight: object | None = None,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> Tensor:
    """The negative log-likelihood of `target` under log-probabilities `input`.

    `input` is `(n, c)` or `(n, c, d1, ...)` of *log*-probabilities -- what
    `log_softmax` produces -- and `target` holds one class index per position.
    Pairing it with `log_softmax` gives what `cross_entropy` gives; the two are
    separate because a model that already carries its own log-probabilities
    should not have them recomputed.

    `weight` scales each class's contribution, and with `reduction="mean"` the
    denominator becomes the total weight rather than the count -- which is what
    makes a weighted mean an average and not a scaled sum. `ignore_index`
    removes positions from both the numerator and that denominator.
    """

    scores = _atleast_tensor(input)
    indices = _atleast_tensor(target)
    if scores.ndim() < 2:
        raise ValueError(
            f"nll_loss expects at least two dimensions on the input, got {scores.ndim()}"
        )
    if indices.ndim() + 1 != scores.ndim():
        raise ValueError(
            f"nll_loss expects a target of rank {scores.ndim() - 1} for an input "
            f"of rank {scores.ndim()}, got {indices.ndim()}"
        )

    classes = scores.shape[1]
    kept = (indices != ignore_index).astype(scores.dtype)
    # An ignored position still has to name a valid class for the gather; which
    # one does not matter, because its weight below is zero.
    safe = _F.where(indices != ignore_index, indices, indices * 0)

    picked = _F.gather(scores, 1, safe.unsqueeze(1).astype("int64")).squeeze(1)

    if weight is None:
        scale = kept
    else:
        per_class = _atleast_tensor(weight)
        if per_class.ndim() != 1 or per_class.shape[0] != classes:
            raise ValueError(
                f"nll_loss expects one weight per class ({classes}), got "
                f"{list(per_class.shape)}"
            )
        chosen = _F.index_select(per_class, 0, safe.reshape(-1).astype("int64"))
        scale = chosen.reshape(list(kept.shape)).astype(scores.dtype) * kept

    weighted = -picked * scale
    if reduction == "none":
        return weighted
    if reduction == "sum":
        return _F.sum(weighted)
    if reduction == "mean":
        # The mean of a weighted loss divides by the total weight. Dividing by
        # the count instead would shrink the loss whenever the weights are
        # small, which is a rescaling of the learning rate dressed as a loss.
        return _F.sum(weighted) / _F.sum(scale)
    raise ValueError(
        f"nll_loss reduction must be 'none', 'mean' or 'sum', got {reduction!r}"
    )


def prelu(input: object, weight: object) -> Tensor:
    """`max(x, 0) + weight * min(x, 0)`: a leaky rectifier whose slope is
    learned rather than fixed.

    `weight` is either a single value shared by every channel or one value per
    channel, in which case it is lined up with dimension 1 -- the channel axis
    for the `(n, c, ...)` layout the convolutions use. The gradient reaches it,
    which is the entire point of the op.
    """

    tensor = _atleast_tensor(input)
    slope = _atleast_tensor(weight)

    if slope.ndim() > 1:
        raise ValueError(
            f"prelu expects a scalar or 1-D weight, got {slope.ndim()} dimensions"
        )
    if slope.ndim() == 1 and slope.shape[0] != 1:
        channel_axis = 1 if tensor.ndim() > 1 else 0
        if tensor.shape[channel_axis] != slope.shape[0]:
            raise ValueError(
                f"prelu expects one weight per channel ({tensor.shape[channel_axis]}), "
                f"got {slope.shape[0]}"
            )
        shape = [1] * tensor.ndim()
        shape[channel_axis] = slope.shape[0]
        slope = slope.reshape(shape)

    # `relu(x) + w (x - relu(x))` rather than a branch. `x - relu(x)` is the
    # negative part, exactly zero above the origin and exactly `x` below it, so
    # the value is the same to the last bit as the branch would give -- and the
    # gradient reaches `weight` without a kernel of its own.
    #
    # At exactly zero the derivative comes out as `weight`, because `relu`
    # takes the flat side there. That is the side `leaky_relu` takes for its
    # own fixed slope, so the two agree on the one input where a rectifier has
    # a choice.
    positive = _F.relu(tensor)
    return positive + slope * (tensor - positive)


def gumbel_softmax(
    logits: object,
    tau: float = 1.0,
    hard: bool = False,
    dim: int = -1,
    eps: float = 1e-20,
) -> Tensor:
    """A differentiable sample from the categorical distribution `logits` names.

    Adds Gumbel noise to the logits and takes a softmax at temperature `tau`:
    as `tau` falls the result approaches a one-hot draw, and at every `tau` it
    stays differentiable in the logits, which sampling itself is not.

    With `hard=True` the result *is* one-hot, and the gradient is still the
    soft one -- the straight-through estimator, written as
    `(hard - soft).detach() + soft` so the forward value is the hard vector and
    the backward value the soft one's.
    """

    scores = _atleast_tensor(logits)
    if tau <= 0.0:
        raise ValueError(f"gumbel_softmax requires a positive tau, got {tau}")
    axis = _normalize_axis(dim, scores.ndim(), "gumbel_softmax")

    # Gumbel(0, 1) by inverse transform: -log(-log(u)) for u uniform on (0, 1).
    # `eps` keeps both logarithms off zero, where the sample would be infinite
    # and the softmax would answer NaN rather than a distribution.
    uniform = _C.Tensor.rand_like(scores)
    noise = -_F.log(-_F.log(uniform + eps) + eps)

    soft = _F.softmax((scores + noise) / tau, axis)
    if not hard:
        return soft

    index = _F.argmax(soft, axis, True)
    # One at the winning position and zero elsewhere, built from `soft` so it
    # inherits its dtype and device without either being named here.
    ones = _C.Tensor.ones_like(_F.narrow(soft, axis, 0, 1))
    drawn = _F.scatter(_C.Tensor.zeros_like(soft), axis, index, ones)
    # Straight through: the value that leaves is the one-hot draw, and the
    # gradient that comes back is the soft one's, because the difference
    # between them is detached and so contributes nothing to it.
    return (drawn - soft).detach() + soft


def pixel_shuffle(input: object, upscale_factor: int) -> Tensor:
    """Trade `upscale_factor**2` channels for that much height and width.

    `(n, c * r * r, h, w)` becomes `(n, c, h * r, w * r)`. This is the last
    layer of a super-resolution network: upsampling by rearranging channels
    costs nothing and invents nothing, where a transposed convolution does
    both.
    """

    tensor = _atleast_tensor(input)
    factor = int(upscale_factor)
    if factor < 1:
        raise ValueError(
            f"pixel_shuffle requires a positive factor, got {upscale_factor}"
        )
    if tensor.ndim() < 3:
        raise ValueError(
            f"pixel_shuffle expects at least three dimensions, got {tensor.ndim()}"
        )

    *batch, channels, height, width = list(tensor.shape)
    block = factor * factor
    if channels % block:
        raise ValueError(
            f"pixel_shuffle needs the channel count ({channels}) to divide by "
            f"the squared factor ({block})"
        )
    out_channels = channels // block

    # Split the channel axis into (c, r, r), then interleave each r with the
    # spatial axis it belongs to. The permutation is the whole operation.
    unpacked = tensor.reshape([*batch, out_channels, factor, factor, height, width])
    lead = len(batch)
    order = list(range(lead)) + [
        lead,
        lead + 3,
        lead + 1,
        lead + 4,
        lead + 2,
    ]
    return _F.permute(unpacked, order).reshape(
        [*batch, out_channels, height * factor, width * factor]
    )


def pixel_unshuffle(input: object, downscale_factor: int) -> Tensor:
    """The inverse of `pixel_shuffle`: `(n, c, h * r, w * r)` back to
    `(n, c * r * r, h, w)`."""

    tensor = _atleast_tensor(input)
    factor = int(downscale_factor)
    if factor < 1:
        raise ValueError(
            f"pixel_unshuffle requires a positive factor, got {downscale_factor}"
        )
    if tensor.ndim() < 3:
        raise ValueError(
            f"pixel_unshuffle expects at least three dimensions, got {tensor.ndim()}"
        )

    *batch, channels, height, width = list(tensor.shape)
    if height % factor or width % factor:
        raise ValueError(
            f"pixel_unshuffle needs both spatial sizes ({height}, {width}) to "
            f"divide by the factor ({factor})"
        )
    out_height = height // factor
    out_width = width // factor

    unpacked = tensor.reshape([*batch, channels, out_height, factor, out_width, factor])
    lead = len(batch)
    order = list(range(lead)) + [
        lead,
        lead + 2,
        lead + 4,
        lead + 1,
        lead + 3,
    ]
    return _F.permute(unpacked, order).reshape(
        [*batch, channels * factor * factor, out_height, out_width]
    )


# --- putting a pooled tensor back where it came from ------------------------


def _unpool(
    op: str,
    input: object,
    indices: object,
    spatial: tuple[int, ...],
    rank: int,
) -> Tensor:
    """`input` scattered back to the positions `indices` names.

    `max_pool` reports where each maximum came from, as a flat offset into the
    plane of one channel of one sample. Every plane names its own positions, so
    the planes are laid end to end and each one's offsets are shifted by where
    it starts -- which is arithmetic on sizes, so NumPy computes the shifts.

    The rest is `scatter` into zeros, whose gradient is a gather at the same
    positions: exactly what `max_pool`'s own backward does, and for the same
    reason.
    """

    values = _atleast_tensor(input)
    positions = _atleast_tensor(indices)
    if values.ndim() != rank:
        raise ValueError(
            f"{op} expects a {rank}-dimensional input of (batch, channels, "
            f"{'positions' if rank == 3 else 'height, width'}), got "
            f"{values.ndim()} dimensions"
        )
    if list(positions.shape) != list(values.shape):
        raise ValueError(
            f"{op} needs one index per value, got {list(positions.shape)} "
            f"against {list(values.shape)}"
        )
    if "int" not in str(positions.dtype):
        raise TypeError(f"{op} requires integer indices, got {positions.dtype}")

    sizes = [int(size) for size in values.shape]
    planes = sizes[0] * sizes[1]
    plane = _element_count(spatial)
    pooled = _element_count(sizes[2:])

    low, high = int(positions.min().item()), int(positions.max().item())
    if planes and pooled and (low < 0 or high >= plane):
        raise IndexError(
            f"{op} was given positions in [{low}, {high}] for a plane of "
            f"{spatial}, which holds {plane}"
        )

    shifts = _index_tensor((_np.arange(planes) * plane).reshape(planes, 1), positions)
    flat = (positions.reshape(planes, pooled).astype("int64") + shifts).reshape(
        planes * pooled
    )
    target = Tensor.zeros(
        [planes * plane], dtype=values.dtype, device=_C.Device(values.device)
    )
    scattered = _F.scatter(target, 0, flat, values.reshape(planes * pooled))
    return scattered.reshape([sizes[0], sizes[1], *spatial])


def _unpool_size(
    op: str,
    pooled: list[int],
    output_size: object,
    kernel: tuple[int, ...],
    step: tuple[int, ...],
    margin: tuple[int, ...],
) -> tuple[int, ...]:
    """The plane to scatter into: the one given, or the smallest that pools to
    this shape.

    Pooling loses the remainder, so several input sizes give the same output
    and the inverse is not determined -- `output_size` is how a caller says
    which one they had. Without it the smallest is assumed, which is what
    `torch` does too.
    """

    if output_size is not None:
        spatial = tuple(_operator.index(size) for size in output_size)
        if len(spatial) != len(kernel):
            raise ValueError(
                f"{op} expects an output size with one entry per spatial axis "
                f"({len(kernel)}), got {len(spatial)}"
            )
        return spatial
    return tuple(
        (count - 1) * s + k - 2 * pad
        for count, k, s, pad in zip(pooled, kernel, step, margin)
    )


def max_unpool1d(
    input: object,
    indices: object,
    kernel_size: object,
    stride: object | None = None,
    padding: object = 0,
    output_size: object = None,
) -> Tensor:
    """Scatter a pooled signal back to where `max_pool1d` found it.

    The partial inverse of `max_pool1d(..., return_indices=True)`: the maxima
    go back to the positions they came from and everything else is zero, which
    is what an unpooling decoder wants -- it recovers the shape and the
    locations, and the values it discarded stay discarded.
    """

    kernel = _sliding_argument(kernel_size, "kernel_size", 1, 1, "max_unpool1d")
    step = (
        kernel
        if stride is None
        else _sliding_argument(stride, "stride", 1, 1, "max_unpool1d")
    )
    margin = _sliding_argument(padding, "padding", 1, 0, "max_unpool1d")
    values = _atleast_tensor(input)
    spatial = _unpool_size(
        "max_unpool1d",
        [int(size) for size in values.shape][2:],
        output_size,
        kernel,
        step,
        margin,
    )
    return _unpool("max_unpool1d", values, indices, spatial, 3)


def max_unpool2d(
    input: object,
    indices: object,
    kernel_size: object,
    stride: object | None = None,
    padding: object = 0,
    output_size: object = None,
) -> Tensor:
    """Scatter a pooled image back to where `max_pool2d` found it.

    See `max_unpool1d`. `output_size` matters more here than there: pooling a
    seven-wide input by two gives the same three columns as pooling a six-wide
    one, so without it the smaller is assumed.
    """

    kernel = _sliding_argument(kernel_size, "kernel_size", 2, 1, "max_unpool2d")
    step = (
        kernel
        if stride is None
        else _sliding_argument(stride, "stride", 2, 1, "max_unpool2d")
    )
    margin = _sliding_argument(padding, "padding", 2, 0, "max_unpool2d")
    values = _atleast_tensor(input)
    spatial = _unpool_size(
        "max_unpool2d",
        [int(size) for size in values.shape][2:],
        output_size,
        kernel,
        step,
        margin,
    )
    return _unpool("max_unpool2d", values, indices, spatial, 4)


# --- the stochastic regularizers -------------------------------------------

#: The value SELU saturates to, `-scale * alpha`. Dropping an element to this
#: rather than to zero is what makes `alpha_dropout` the one that suits a
#: self-normalizing network: zero is not a neutral value for an activation
#: whose negative limit is here.
_SELU_SATURATION = -1.7580993408473766


def _dropout_probability(op: str, p: object) -> float:
    probability = float(p)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{op} probability must be between 0 and 1, got {probability}")
    return probability


def _channelwise(op: str, input: object, p: float, training: bool, rank: int) -> Tensor:
    """`dropout2d` at a rank it does not take, by way of one it does.

    Dropping a whole channel is the same operation whatever the positions are
    shaped like, so the positions are flattened into one axis and a second of
    length one is added -- which is a four-dimensional tensor, and the kernel
    handles it.
    """

    tensor = _atleast_tensor(input)
    if tensor.ndim() != rank:
        raise ValueError(
            f"{op} expects a {rank}-dimensional input of (batch, channels, "
            f"{'positions' if rank == 3 else 'depth, height, width'}), got "
            f"{tensor.ndim()} dimensions"
        )
    sizes = [int(size) for size in tensor.shape]
    flattened = tensor.reshape([sizes[0], sizes[1], _element_count(sizes[2:]), 1])
    return _F.dropout2d(flattened, p, training).reshape(sizes)


def dropout1d(input: object, p: float = 0.5, training: bool = True) -> Tensor:
    """Zero whole channels of a `(batch, channels, positions)` input.

    `dropout2d` for a signal rather than an image. Dropping a whole channel
    rather than scattered elements is what makes it worth having: adjacent
    positions in a feature map are correlated, so zeroing one of them leaves
    its value recoverable from its neighbours and regularises very little.
    """

    return _channelwise("dropout1d", input, p, training, 3)


def dropout3d(input: object, p: float = 0.5, training: bool = True) -> Tensor:
    """Zero whole channels of a `(batch, channels, depth, height, width)` input."""

    return _channelwise("dropout3d", input, p, training, 5)


def _alpha_dropout(
    op: str, input: object, p: float, training: bool, per_channel: bool
) -> Tensor:
    tensor = _atleast_tensor(input)
    probability = _dropout_probability(op, p)
    if not training or probability == 0.0:
        return tensor
    if per_channel and tensor.ndim() < 2:
        raise ValueError(
            f"{op} expects a batch and a channel axis, got {tensor.ndim()} dimensions"
        )

    saturation = _SELU_SATURATION
    if probability == 1.0:
        # Every element takes the saturation value, so the answer has no
        # variance for the rescaling to restore; the limit of the formula below
        # is the mean it would shift to, which is zero. Computing it instead
        # gives `inf - inf`.
        return tensor * 0.0

    # The affine correction that keeps a standard normal standard: with `m` the
    # keep mask, `a * (x * m + s * (1 - m)) + b` has mean 0 and variance 1 when
    # `x` does. `a` is what rescales the variance and `b` recentres the mean.
    keep = 1.0 - probability
    scale = (keep * (1.0 + probability * saturation * saturation)) ** -0.5
    shift = -scale * saturation * probability

    shape = list(tensor.shape)
    if per_channel:
        shape = shape[:2] + [1] * (len(shape) - 2)
    mask = _bernoulli(Tensor.full(shape, keep, dtype=tensor.dtype))
    kept = tensor * mask + (1.0 - mask) * saturation
    return kept * scale + shift


def alpha_dropout(input: object, p: float = 0.5, training: bool = True) -> Tensor:
    """Dropout that leaves a self-normalizing network self-normalizing.

    Ordinary dropout zeroes an element and rescales the rest, which keeps the
    mean and changes the variance. That is fine after a rectifier, whose
    negative side is zero anyway, and wrong after `selu`, whose whole point is
    that activations hold a mean of zero and a variance of one from layer to
    layer.

    So this drops to `selu`'s own negative saturation value rather than to
    zero, and then applies the affine correction that restores both moments at
    once. A standard normal input comes back standard normal, at any `p`, which
    is the property and is what the test checks.
    """

    return _alpha_dropout("alpha_dropout", input, p, training, False)


def feature_alpha_dropout(
    input: object, p: float = 0.5, training: bool = True
) -> Tensor:
    """`alpha_dropout` over whole channels, as `dropout2d` is over `dropout`.

    The same correction and the same saturation value; only the mask is
    coarser, one draw per channel rather than one per element.
    """

    return _alpha_dropout("feature_alpha_dropout", input, p, training, True)


def rrelu(
    input: object,
    lower: float = 1.0 / 8,
    upper: float = 1.0 / 3,
    training: bool = True,
) -> Tensor:
    """A leaky rectifier whose negative slope is drawn rather than fixed.

    In training each element gets its own slope, uniform on `[lower, upper]`;
    in evaluation every element gets the midpoint, so the network sees the
    average of what it was trained against rather than a fresh draw.

    Written as `relu(x) + slope * (x - relu(x))`, which is bit-exact on the
    positive side -- `x - relu(x)` is exactly zero there -- and takes the
    negative side's derivative at the origin, agreeing with `leaky_relu` and
    `prelu` on the one point where the two sides disagree.
    """

    tensor = _atleast_tensor(input)
    low, high = float(lower), float(upper)
    if low > high:
        raise ValueError(
            f"rrelu takes a range with the lower bound first, got [{low}, {high}]"
        )
    if low < 0.0:
        raise ValueError(f"rrelu requires non-negative slopes, got {low}")

    # A fresh slope per element while training; the midpoint otherwise, so the
    # network sees the average of what it was trained against. A range with
    # nothing in it is the midpoint either way.
    if training and low != high:
        slope: object = Tensor.rand_like(tensor) * (high - low) + low
    else:
        slope = (low + high) / 2.0
    positive = _F.relu(tensor)
    return positive + slope * (tensor - positive)


# --- completing the pairs ---------------------------------------------------


def embedding(input: object, weight: object, padding_idx: int | None = None) -> Tensor:
    """The rows of `weight` that `input` names, one per index.

    `index_select` over a flattened index and a reshape back, which is what a
    lookup table is. `nn.Embedding` is this with the table owned by a module;
    this is the form for a table the caller already holds -- a frozen one, or
    one shared between two models.

    `padding_idx` names a row that takes no gradient. The forward is unaffected
    -- the row is whatever it holds -- and only the gradient is masked, which
    is `torch.nn.functional.embedding`'s contract. The mask is arithmetic
    rather than a special case in a backward: the row is read through a
    detached copy of itself, which is the same number and no longer a path the
    gradient can take.
    """

    indices = _atleast_tensor(input)
    table = _atleast_tensor(weight)
    if table.ndim() != 2:
        raise ValueError(
            f"embedding expects a two-dimensional table, got {table.ndim()} dimensions"
        )
    rows, features = (int(size) for size in table.shape)

    if padding_idx is not None:
        position = _operator.index(padding_idx)
        if position < 0:
            position += rows
        if not 0 <= position < rows:
            raise IndexError(
                f"embedding padding_idx {padding_idx} is out of range for a table "
                f"of {rows} rows"
            )
        keep = _np.ones((rows, 1))
        keep[position, 0] = 0.0
        mask = _constant_like(keep, table)
        # Numerically `table`, since the two branches partition the rows; the
        # detached branch is simply not a path a gradient can travel.
        table = table * mask + table.detach() * (1.0 - mask)

    flat = indices.reshape(_element_count(indices))
    looked_up = _F.index_select(table, 0, flat)
    return looked_up.reshape([int(size) for size in indices.shape] + [features])


def embedding_bag(
    input: object,
    weight: object,
    offsets: object | None = None,
    mode: str = "mean",
    per_sample_weights: object | None = None,
    include_last_offset: bool = False,
    padding_idx: int | None = None,
) -> Tensor:
    """One vector per bag of indices, reduced without ever forming the bags.

    `embedding` followed by a reduction, for the case where the reduction is
    the only reason the lookup happened -- a bag-of-words encoder, or the
    sparse features of a recommender. Doing it in two steps is correct and
    costs a `(total, dim)` intermediate that is thrown away; this is the same
    two steps with the intermediate never named, which is the whole reason the
    fused operation exists elsewhere.

    Bags arrive either way. A two-dimensional `input` is one bag per row, all
    the same length. A one-dimensional `input` needs `offsets` saying where
    each bag starts, which is what allows them to differ -- and with
    `include_last_offset` the final entry is the end rather than a start, which
    is how a caller passes the cumulative counts they already have.

    `mode` is `"sum"`, `"mean"` or `"max"`. An empty bag reduces to zero in all
    three, having nothing else to reduce to. `per_sample_weights` scales each
    row before it is summed and is meaningful only for `"sum"`, which is what
    `torch` says too -- a weighted mean would need the weights in the divisor
    as well, and that is a different operation.
    """

    if mode not in ("sum", "mean", "max"):
        raise ValueError(
            f'embedding_bag takes a mode of "sum", "mean" or "max", got {mode!r}'
        )
    indices = _atleast_tensor(input)
    table = _atleast_tensor(weight)

    if indices.ndim() == 2:
        if offsets is not None:
            raise ValueError(
                "embedding_bag takes offsets with a one-dimensional input; a "
                "two-dimensional one already says where each bag is"
            )
        bags, length = (int(size) for size in indices.shape)
        starts = _index_tensor(_np.arange(bags) * length, indices)
        flat = indices.reshape(bags * length)
    elif indices.ndim() == 1:
        if offsets is None:
            raise ValueError(
                "embedding_bag needs offsets to divide a one-dimensional input "
                "into bags"
            )
        boundaries = _atleast_tensor(offsets)
        if boundaries.ndim() != 1:
            raise ValueError(
                f"embedding_bag expects one-dimensional offsets, got "
                f"{boundaries.ndim()} dimensions"
            )
        given = int(boundaries.shape[0])
        bags = given - 1 if include_last_offset else given
        if bags < 0:
            raise ValueError(
                "embedding_bag with include_last_offset needs at least one offset"
            )
        # Only the starts take part; a trailing end would otherwise open a bag
        # of its own.
        starts = _F.narrow(boundaries, 0, 0, bags)
        flat = indices
    else:
        raise ValueError(
            f"embedding_bag takes a one- or two-dimensional input, got "
            f"{indices.ndim()} dimensions"
        )

    rows = embedding(flat, table, padding_idx)
    total, features = (int(size) for size in rows.shape)

    if per_sample_weights is not None:
        if mode != "sum":
            raise ValueError(
                f'embedding_bag takes per_sample_weights only with mode="sum", '
                f"not {mode!r}"
            )
        scaling = _atleast_tensor(per_sample_weights).reshape(total, 1)
        rows = rows * scaling

    # Which bag each row belongs to: the number of starts at or below its
    # position, less one. Repeated starts mean an empty bag, and this steps
    # over it rather than into it.
    positions = _index_tensor(_np.arange(total), rows)
    bag_of = _F.searchsorted(starts.astype("int64"), positions, True) - 1
    if total and bags:
        low = int(bag_of.min().item())
        if low < 0:
            raise IndexError(
                "embedding_bag needs its first offset at zero, so that every "
                "index belongs to a bag"
            )
    scattered = broadcast_to(bag_of.reshape(total, 1), (total, features))
    target = Tensor.zeros(
        [bags, features], dtype=rows.dtype, device=_C.Device(rows.device)
    )

    if mode == "max":
        # From the reduction's identity rather than from the zeros already
        # there, so a bag of negatives reduces to its own maximum -- and a bag
        # with nothing in it is never written to and keeps the zero.
        return _F.scatter_reduce(target, 0, scattered, rows, "amax", False)

    summed = _F.scatter_add(target, 0, scattered, rows)
    if mode == "sum":
        return summed
    counts = _F.bincount(bag_of, None, bags).reshape(bags, 1).astype(str(rows.dtype))
    # An empty bag has nothing to average, and zero over one is the zero its
    # sum already is.
    return summed / _F.clamp(counts, 1.0, None)


def channel_shuffle(input: object, groups: int) -> Tensor:
    """Interleave the channels so each group draws from all the others.

    `(n, g * c, ...)` is read as `(n, g, c, ...)`, the two are swapped, and it
    is flattened back. That is the whole operation, and its point: a grouped
    convolution never mixes its groups, so stacking two of them leaves two
    networks side by side. One shuffle between them is what makes it one
    network -- and it costs a permutation, no parameters and no arithmetic.
    """

    tensor = _atleast_tensor(input)
    if tensor.ndim() < 2:
        raise ValueError(
            f"channel_shuffle expects a batch and a channel axis, got "
            f"{tensor.ndim()} dimensions"
        )
    count = _operator.index(groups)
    if count < 1:
        raise ValueError(f"channel_shuffle requires at least one group, got {groups}")
    sizes = [int(size) for size in tensor.shape]
    channels = sizes[1]
    if channels % count:
        raise ValueError(
            f"channel_shuffle needs the channel count ({channels}) to divide by "
            f"the number of groups ({count})"
        )

    split = tensor.reshape([sizes[0], count, channels // count, *sizes[2:]])
    order = [0, 2, 1] + list(range(3, len(sizes) + 1))
    return _F.permute(split, order).reshape(sizes)


def _lp_pool(
    op: str,
    pooling: object,
    input: object,
    norm_type: float,
    kernel_size: object,
    stride: object,
    rank: int,
) -> Tensor:
    tensor = _atleast_tensor(input)
    power = float(norm_type)
    if power <= 0:
        raise ValueError(f"{op} requires a positive norm type, got {norm_type}")
    window = _sliding_argument(kernel_size, "kernel_size", rank, 1, op)
    step = (
        window if stride is None else _sliding_argument(stride, "stride", rank, 1, op)
    )

    # `abs` before the power, which is what makes this the L_p norm of the
    # window at every `norm_type` rather than only at even ones. `torch` raises
    # `x` itself to the power, so an odd norm over negative values gives it the
    # root of a negative number; the two agree wherever that one is defined.
    raised = _F.abs(tensor) ** power
    area = float(_math.prod(window))
    averaged = pooling(
        raised, window if rank > 1 else window[0], step if rank > 1 else step[0]
    )
    return (averaged * area) ** (1.0 / power)


def lp_pool1d(
    input: object,
    norm_type: float,
    kernel_size: object,
    stride: object | None = None,
) -> Tensor:
    """The `p`-norm of each window along the last axis, not its largest or mean.

    A norm type of 1 is the sum and a large one approaches the maximum, so this
    is the family `avg_pool1d` and `max_pool1d` are the ends of -- with a
    gradient everywhere, which is what makes it trainable where the maximum's
    hard selection is not.
    """

    return _lp_pool(
        "lp_pool1d", _F.avg_pool1d, input, norm_type, kernel_size, stride, 1
    )


def lp_pool2d(
    input: object,
    norm_type: float,
    kernel_size: object,
    stride: object | None = None,
) -> Tensor:
    """The `p`-norm of each 2-D window. See `lp_pool1d`."""

    return _lp_pool(
        "lp_pool2d", _F.avg_pool2d, input, norm_type, kernel_size, stride, 2
    )


def affine_grid(theta: object, size: object, align_corners: bool = False) -> Tensor:
    """The sampling grid an affine transform describes, for `grid_sample`.

    `theta` is `(n, 2, 3)` over an `(n, c, h, w)` output or `(n, 3, 4)` over an
    `(n, c, d, h, w)` one: the matrix that takes an output coordinate to the
    input coordinate it should read. Feeding the result to `grid_sample` is a
    spatial transformer, and the gradient reaches `theta`, which is what lets
    the transform be learned rather than specified.

    The base grid -- the normalised coordinate of every output position --
    depends only on the output size, so NumPy builds it, per "Where an
    operation belongs" in `docs/development.md`. `align_corners` decides
    whether -1 and 1 name the outer sample centres or the edges of the volume,
    and it must match what `grid_sample` is then given.
    """

    matrix = _atleast_tensor(theta)
    if matrix.ndim() != 3:
        raise ValueError(
            f"affine_grid expects a batch of matrices, got {matrix.ndim()} dimensions"
        )
    sizes = [_operator.index(value) for value in size]
    if len(sizes) not in (4, 5):
        raise ValueError(
            f"affine_grid takes a four- or five-element output size, got {len(sizes)}"
        )
    spatial = sizes[2:]
    rank = len(spatial)
    if [int(value) for value in matrix.shape] != [sizes[0], rank, rank + 1]:
        raise ValueError(
            f"affine_grid expects theta of shape {[sizes[0], rank, rank + 1]} for an "
            f"output of {sizes}, got {[int(value) for value in matrix.shape]}"
        )
    if any(value < 1 for value in sizes):
        raise ValueError(f"affine_grid requires positive sizes, got {sizes}")

    # One axis at a time: with the corners aligned the ends are exactly -1 and
    # 1, and without them each sample sits at the centre of its own cell.
    def _coordinates(length: int) -> "_np.ndarray":
        if align_corners:
            return _np.linspace(-1.0, 1.0, length) if length > 1 else _np.zeros(1)
        return (2.0 * _np.arange(length) + 1.0) / length - 1.0

    # `grid_sample` reads coordinates in `x, y (, z)` order, which is the
    # reverse of the axes they index, so the spatial sizes are reversed here.
    axes = _np.meshgrid(*[_coordinates(length) for length in spatial], indexing="ij")
    base = _np.stack([*reversed(axes), _np.ones(spatial)], axis=-1)

    homogeneous = _constant_like(base.reshape(1, -1, rank + 1), matrix)
    mapped = _F.matmul(homogeneous, _F.transpose(matrix, -1, -2))
    return mapped.reshape([sizes[0], *spatial, rank])


# --- normalizing over something other than the batch ------------------------


def _feature_shape(channels: int, rank: int) -> list[int]:
    """`(1, channels, 1, 1, ...)`: a per-channel vector lined up with a batch."""

    return [1, channels] + [1] * (rank - 2)


def _affine(
    normalized: Tensor, weight: object | None, bias: object | None, op: str
) -> Tensor:
    """Scale and shift by one value per channel, if either was given."""

    if weight is None and bias is None:
        return normalized
    channels = int(normalized.shape[1])
    shape = _feature_shape(channels, normalized.ndim())
    for name, value in (("weight", weight), ("bias", bias)):
        if value is None:
            continue
        parameter = _atleast_tensor(value)
        if int(parameter.reshape(-1).shape[0]) != channels:
            raise ValueError(
                f"{op} expects one {name} per channel ({channels}), got "
                f"{int(parameter.reshape(-1).shape[0])}"
            )
        normalized = (
            normalized * parameter.reshape(shape)
            if name == "weight"
            else normalized + parameter.reshape(shape)
        )
    return normalized


def _normalized_groups(
    op: str, tensor: Tensor, groups: int, eps: float
) -> tuple[Tensor, Tensor, Tensor]:
    """`tensor` with each group centred and scaled, and the statistics used.

    The groups are `(batch, groups, everything else)`, so one reshape puts every
    element a group owns on a single axis and the mean and variance are one
    reduction each. Which elements a group owns -- some channels and all of
    their positions -- is the whole difference between this, `layer_norm` and
    `batch_norm`; the arithmetic afterwards is the same in all three.
    """

    sizes = [int(size) for size in tensor.shape]
    batch, channels = sizes[0], sizes[1]
    if channels % groups:
        raise ValueError(
            f"{op} needs the channel count ({channels}) to divide by the number "
            f"of groups ({groups})"
        )
    per_group = (channels // groups) * _math.prod(sizes[2:])
    grouped = tensor.reshape(batch, groups, per_group)

    mean = _F.mean(grouped, -1, True)
    variance = _F.var(grouped, -1, False, True)
    centred = (grouped - mean) / _F.sqrt(variance + float(eps))
    return centred.reshape(sizes), mean, variance


def group_norm(
    input: object,
    num_groups: int,
    weight: object | None = None,
    bias: object | None = None,
    eps: float = 1e-5,
) -> Tensor:
    """Normalize over each group of channels and all of their positions.

    Between `layer_norm`, which takes every channel together, and
    `instance_norm`, which takes each on its own: `num_groups` says how finely
    to divide them, and those two are the ends of that range. Unlike
    `batch_norm` the statistics never cross the batch, so the result for one
    sample does not depend on which others it was computed with -- which is
    what makes it usable at a batch size of one.

    `weight` and `bias` are one value per channel, not per group.
    """

    tensor = _atleast_tensor(input)
    if tensor.ndim() < 2:
        raise ValueError(
            f"group_norm expects a batch and a channel axis, got {tensor.ndim()} "
            "dimensions"
        )
    groups = _operator.index(num_groups)
    if groups < 1:
        raise ValueError(f"group_norm requires at least one group, got {num_groups}")
    normalized, _mean, _variance = _normalized_groups("group_norm", tensor, groups, eps)
    return _affine(normalized, weight, bias, "group_norm")


def instance_norm(
    input: object,
    running_mean: object | None = None,
    running_var: object | None = None,
    weight: object | None = None,
    bias: object | None = None,
    use_input_stats: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    """Normalize each channel of each sample over its own positions.

    `group_norm` with one group per channel, which is what it is -- so it is
    written that way rather than twice. What it adds is the running statistics:
    with buffers passed in and `use_input_stats`, they are updated from this
    batch, and with `use_input_stats` false they are used instead of it, which
    is how an evaluation pass reproduces training-time behaviour.

    The buffers are updated with the *unbiased* variance while the
    normalization uses the biased one, which is what `batch_norm` does here and
    in torch: the divisor that makes a good estimate of the population variance
    is not the one that makes this batch have unit variance.
    """

    tensor = _atleast_tensor(input)
    if tensor.ndim() < 3:
        raise ValueError(
            "instance_norm expects a batch, a channel and at least one position "
            f"axis, got {tensor.ndim()} dimensions"
        )
    sizes = [int(size) for size in tensor.shape]
    channels = sizes[1]
    positions = _math.prod(sizes[2:])

    if not use_input_stats:
        if running_mean is None or running_var is None:
            raise ValueError(
                "instance_norm without input statistics needs both running_mean "
                "and running_var"
            )
        shape = _feature_shape(channels, tensor.ndim())
        mean = _atleast_tensor(running_mean).reshape(shape)
        variance = _atleast_tensor(running_var).reshape(shape)
        normalized = (tensor - mean) / _F.sqrt(variance + float(eps))
        return _affine(normalized, weight, bias, "instance_norm")

    normalized, _mean, _variance = _normalized_groups(
        "instance_norm", tensor, channels, eps
    )
    if running_mean is not None or running_var is not None:
        _update_running(
            tensor.reshape(sizes[0], channels, positions),
            running_mean,
            running_var,
            float(momentum),
        )
    return _affine(normalized, weight, bias, "instance_norm")


def _update_running(
    flat: Tensor,
    running_mean: object | None,
    running_var: object | None,
    momentum: float,
) -> None:
    """Move the running buffers towards this batch's per-channel statistics.

    Written in place, because a buffer is what the caller keeps -- and outside
    the graph, because a running average of past batches is not something the
    loss should differentiate through.
    """

    with _C.no_grad():
        if running_mean is not None:
            buffer = _atleast_tensor(running_mean)
            batch_mean = _F.mean(_F.mean(flat, -1), 0).detach()
            buffer.copy_(buffer * (1.0 - momentum) + batch_mean * momentum)
        if running_var is not None:
            buffer = _atleast_tensor(running_var)
            batch_var = _F.mean(_F.var(flat, -1, True), 0).detach()
            buffer.copy_(buffer * (1.0 - momentum) + batch_var * momentum)


def local_response_norm(
    input: object,
    size: int,
    alpha: float = 1e-4,
    beta: float = 0.75,
    k: float = 1.0,
) -> Tensor:
    """Divide each element by the energy in a window of neighbouring channels.

    `x / (k + alpha * mean(x**2 over `size` channels around it)) ** beta`. The
    normalization AlexNet used: a channel that responds strongly suppresses the
    same position in the channels beside it, so only a few channels stay large.
    Superseded by `batch_norm` and the ones beside it, and still wanted to run
    a network from when it was not.

    The window is an average over the channel axis, which is `avg_pool3d` with
    a window of one in the two positional axes -- so the channel axis is moved
    where a pooling expects to find a spatial one, and no new kernel is needed.
    """

    tensor = _atleast_tensor(input)
    if tensor.ndim() < 3:
        raise ValueError(
            "local_response_norm expects a batch, a channel and at least one "
            f"position axis, got {tensor.ndim()} dimensions"
        )
    window = _operator.index(size)
    if window < 1:
        raise ValueError(
            f"local_response_norm requires a window of at least one, got {size}"
        )

    sizes = [int(each) for each in tensor.shape]
    positions = _math.prod(sizes[2:])
    # `(batch, 1, channels, positions, 1)`: the channel axis in the depth slot
    # of a 3-D pooling, and one on either side of it so the window is 1-D.
    stacked = (tensor * tensor).reshape(sizes[0], 1, sizes[1], positions, 1)
    # An even window has no centre, so it takes one more from below than above
    # -- the same side `torch` takes it from.
    padded = _F.pad(stacked, [0, 0, 0, 0, window // 2, (window - 1) // 2])
    energy = avg_pool3d(padded, (window, 1, 1), (1, 1, 1))
    divisor = (energy.reshape(sizes) * float(alpha) + float(k)) ** float(beta)
    return tensor / divisor


# --- three spatial axes, out of the two the kernels have --------------------


def _volume_input(op: str, input: object) -> Tensor:
    tensor = _atleast_tensor(input)
    if tensor.ndim() != 5:
        raise ValueError(
            f"{op} expects a five-dimensional input of (batch, channels, depth, "
            f"height, width), got {tensor.ndim()} dimensions"
        )
    return tensor


def _depth_planes(
    op: str,
    tensor: Tensor,
    kernel: tuple[int, ...],
    step: tuple[int, ...],
    margin: tuple[int, ...],
    spaced: tuple[int, ...],
    fill: float,
) -> tuple[list[Tensor], int, int]:
    """The 2-D problem each depth tap of a 3-D window reduces to.

    A 3-D window is a stack of 2-D windows, one per depth tap, so a 3-D
    operation is the 2-D one run over each of those and then combined along the
    depth. That leaves the work with the `conv2d` and `max_pool2d` kernels,
    where laying the volume out as columns would take it away from them and
    cost a copy of every window besides.

    Each plane comes back as `(batch * out_depth, channels, height, width)`,
    which is the shape the 2-D kernels take, and `fill` is what the depth
    padding holds -- zero for a sum, negative infinity for a maximum.
    """

    batch, channels, depth = (int(size) for size in list(tensor.shape)[:3])
    reach = spaced[0] * (kernel[0] - 1) + 1
    padded = depth + 2 * margin[0]
    if reach > padded:
        raise ValueError(
            f"{op} has a kernel that does not fit along the depth: a dilated "
            f"extent of {reach} over a padded depth of {padded}"
        )
    out_depth = (padded - reach) // step[0] + 1
    if margin[0]:
        # `pad` takes its axes innermost-first, so the depth is the third pair.
        tensor = _F.pad(tensor, [0, 0, 0, 0, margin[0], margin[0]], value=fill)

    # Depth in front of the channels once, rather than once per tap: the 2-D
    # kernels want the channel axis second, and the batch is what absorbs the
    # depth.
    moved = _F.permute(tensor, [0, 2, 1, 3, 4])
    height, width = (int(size) for size in list(moved.shape)[3:])
    starts = _np.arange(out_depth) * step[0]

    planes = []
    for tap in range(kernel[0]):
        positions = _index_tensor(starts + tap * spaced[0], moved)
        taken = _F.index_select(moved, 1, positions)
        planes.append(taken.reshape(batch * out_depth, channels, height, width))
    return planes, batch, out_depth


def _as_volume(planes: Tensor, batch: int, out_depth: int) -> Tensor:
    """A stack of 2-D results back into `(batch, channels, depth, h, w)`."""

    channels, height, width = (int(size) for size in list(planes.shape)[1:])
    return _F.permute(
        planes.reshape(batch, out_depth, channels, height, width), [0, 2, 1, 3, 4]
    )


def _pooling_geometry(
    op: str, kernel_size: object, stride: object, padding: object
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """A pooling window, its stride -- defaulting to the window -- and padding."""

    kernel = _sliding_argument(kernel_size, "kernel_size", 3, 1, op)
    step = kernel if stride is None else _sliding_argument(stride, "stride", 3, 1, op)
    margin = _sliding_argument(padding, "padding", 3, 0, op)
    if any(pad * 2 > size for pad, size in zip(margin, kernel)):
        # Otherwise a window can be all padding, and there is no sensible
        # maximum or mean of nothing. `torch` refuses the same case.
        raise ValueError(
            f"{op} takes padding of at most half the window, got {margin} "
            f"for a window of {kernel}"
        )
    return kernel, step, margin


def conv3d(
    input: object,
    weight: object,
    bias: object | None = None,
    stride: object = 1,
    padding: object = 0,
    dilation: object = 1,
    groups: int = 1,
) -> Tensor:
    """3-D cross-correlation of a volume with a `(out, in / groups, kD, kH, kW)`
    kernel.

    Written as `kD` two-dimensional convolutions rather than as one big matrix
    product: each depth tap of the kernel is a 2-D kernel, applied to the depth
    slices that tap reads, and the results are summed. So the arithmetic is
    done by the `conv2d` kernel that already exists, at its memory cost rather
    than at im2col's, and `dilation` and `groups` mean what they mean there
    because they are passed straight to it.

    The bias is added once at the end, not once per tap, which is both right
    and cheaper.
    """

    tensor = _volume_input("conv3d", input)
    taps = _atleast_tensor(weight)
    if taps.ndim() != 5:
        raise ValueError(
            f"conv3d expects a five-dimensional weight of (out_channels, "
            f"in_channels / groups, kD, kH, kW), got {taps.ndim()} dimensions"
        )
    kernel = tuple(int(size) for size in list(taps.shape)[2:])
    step = _sliding_argument(stride, "stride", 3, 1, "conv3d")
    margin = _sliding_argument(padding, "padding", 3, 0, "conv3d")
    spaced = _sliding_argument(dilation, "dilation", 3, 1, "conv3d")

    planes, batch, out_depth = _depth_planes(
        "conv3d", tensor, kernel, step, margin, spaced, 0.0
    )
    total = None
    for tap, plane in enumerate(planes):
        flat = _F.squeeze(_F.narrow(taps, 2, tap, 1), 2)
        partial = _F.conv2d(
            plane,
            flat,
            None,
            (step[1], step[2]),
            (margin[1], margin[2]),
            (spaced[1], spaced[2]),
            groups,
        )
        total = partial if total is None else total + partial

    volume = _as_volume(total, batch, out_depth)
    if bias is None:
        return volume
    channels = int(volume.shape[1])
    return volume + _atleast_tensor(bias).reshape(1, channels, 1, 1, 1)


def max_pool3d(
    input: object,
    kernel_size: object,
    stride: object | None = None,
    padding: object = 0,
) -> Tensor:
    """The largest value in each 3-D window. Stride defaults to the window.

    The maximum over a stack is the maximum of the maxima, so this is
    `max_pool2d` on each depth tap and then an elementwise maximum across them.
    The depth padding is negative infinity for the same reason `max_pool2d`
    pads with it: a padded position must never be the one that wins.
    """

    tensor = _volume_input("max_pool3d", input)
    kernel, step, margin = _pooling_geometry("max_pool3d", kernel_size, stride, padding)
    planes, batch, out_depth = _depth_planes(
        "max_pool3d", tensor, kernel, step, margin, (1, 1, 1), float("-inf")
    )

    best = None
    for plane in planes:
        pooled = _F.max_pool2d(
            plane, (kernel[1], kernel[2]), (step[1], step[2]), (margin[1], margin[2])
        )
        best = pooled if best is None else _F.maximum(best, pooled)
    return _as_volume(best, batch, out_depth)


def _volume_window_sum(
    tensor: Tensor,
    kernel: tuple[int, ...],
    step: tuple[int, ...],
    margin: tuple[int, ...],
) -> Tensor:
    """The mean of each 3-D window counting the padding as zero.

    `avg_pool2d` with `count_include_pad` divides each tap by the whole 2-D
    window, so summing the taps and dividing by the depth gives a sum over the
    3-D window divided by the whole of it -- which is the mean when the padding
    counts.
    """

    planes, batch, out_depth = _depth_planes(
        "avg_pool3d", tensor, kernel, step, margin, (1, 1, 1), 0.0
    )
    total = None
    for plane in planes:
        pooled = _F.avg_pool2d(
            plane,
            (kernel[1], kernel[2]),
            (step[1], step[2]),
            (margin[1], margin[2]),
            True,
        )
        total = pooled if total is None else total + pooled
    return _as_volume(total, batch, out_depth) * (1.0 / kernel[0])


def avg_pool3d(
    input: object,
    kernel_size: object,
    stride: object | None = None,
    padding: object = 0,
    count_include_pad: bool = True,
) -> Tensor:
    """The mean of each 3-D window. Stride defaults to the window.

    With `count_include_pad` the divisor is the whole window; without it, only
    the positions that are really there. The second is the first divided by the
    same computation run over a volume of ones -- which gives the fraction of
    each window that is real -- so the two cannot disagree about which
    positions those are.
    """

    tensor = _volume_input("avg_pool3d", input)
    kernel, step, margin = _pooling_geometry("avg_pool3d", kernel_size, stride, padding)
    summed = _volume_window_sum(tensor, kernel, step, margin)
    if count_include_pad:
        return summed
    covered = _volume_window_sum(
        Tensor.ones_like(tensor, requires_grad=False), kernel, step, margin
    )
    return summed / covered


#: What `minitensor/__init__.py` attaches to the `nn` namespace. Listed here,
#: beside the definitions, so a new function is exported by being written
#: rather than by being remembered somewhere else.
def _sliding_argument(
    value: object, name: str, rank: int, minimum: int, op: str
) -> tuple[int, ...]:
    """`value` as one integer per spatial axis, checked against `minimum`."""

    try:
        entries = (_operator.index(value),) * rank
    except TypeError:
        try:
            entries = tuple(_operator.index(entry) for entry in value)  # type: ignore[union-attr]
        except TypeError:
            raise TypeError(
                f"{op} expects {name} to be an integer or a sequence of them, "
                f"got {type(value).__name__}"
            ) from None
        if len(entries) != rank:
            raise ValueError(
                f"{op} expects one {name} per spatial axis ({rank}), "
                f"got {len(entries)}"
            )
    if any(entry < minimum for entry in entries):
        raise ValueError(f"{op} requires {name} of at least {minimum}, got {entries}")
    return entries


def _sliding_rank(*candidates: object) -> int:
    """How many spatial axes the arguments describe.

    `fold` has no input shape to read the rank off -- its input arrives already
    flattened -- so it comes from whichever argument was given as a sequence,
    and from the two-dimensional case when every one of them is a bare integer.
    Two is the only case `torch.nn.functional.fold` supports at all, so it is
    also the one an unannotated call means.
    """

    for candidate in candidates:
        if isinstance(candidate, (tuple, list)):
            return len(candidate)
    return 2


def _sliding_geometry(
    op: str,
    spatial: tuple[int, ...],
    kernel_size: object,
    dilation: object,
    padding: object,
    stride: object,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], "_np.ndarray"]:
    """The kernel, the padding, the padded plane and the map of every tap.

    The map has shape `(taps, blocks)`: row `t`, column `b` is the flat position
    in the padded plane that tap `t` of block `b` reads. Every value in it comes
    from the geometry integers alone -- no tensor is involved and no gradient
    can be -- so NumPy computes it, per "Where an operation belongs" in
    `docs/development.md`. `ravel_multi_index` is what makes the rank a
    parameter rather than a special case, which is why any number of spatial
    axes works here and not just the two a 2-D convolution needs.
    """

    rank = len(spatial)
    kernel = _sliding_argument(kernel_size, "kernel_size", rank, 1, op)
    spaced = _sliding_argument(dilation, "dilation", rank, 1, op)
    margin = _sliding_argument(padding, "padding", rank, 0, op)
    step = _sliding_argument(stride, "stride", rank, 1, op)

    padded = tuple(size + 2 * pad for size, pad in zip(spatial, margin))
    # A dilated kernel reaches `d * (k - 1) + 1` positions, not `k` of them.
    reach = tuple(space * (k - 1) + 1 for k, space in zip(kernel, spaced))
    if any(extent > size for extent, size in zip(reach, padded)):
        raise ValueError(
            f"{op} has a kernel that does not fit: a dilated extent of {reach} "
            f"over a padded input of {padded}"
        )
    blocks = tuple(
        (size - extent) // s + 1 for size, extent, s in zip(padded, reach, step)
    )

    taps = _np.meshgrid(
        *[_np.arange(k) * space for k, space in zip(kernel, spaced)], indexing="ij"
    )
    starts = _np.meshgrid(
        *[_np.arange(count) * s for count, s in zip(blocks, step)], indexing="ij"
    )
    coordinates = [
        tap.reshape(-1, 1) + start.reshape(1, -1) for tap, start in zip(taps, starts)
    ]
    return kernel, margin, padded, _np.ravel_multi_index(coordinates, padded)


def unfold(
    input: object,
    kernel_size: object,
    dilation: object = 1,
    padding: object = 0,
    stride: object = 1,
) -> Tensor:
    """Every sliding block of `input`, laid out one per column.

    `(n, c, *spatial)` becomes `(n, c * taps, blocks)`, where `taps` is the
    product of `kernel_size` and each column holds one window flattened over
    the channels and the kernel -- im2col. What that buys is a convolution
    written as a single matrix product:

        cols = unfold(x, 3, padding=1)
        out = (weight.reshape(out_channels, -1) @ cols).reshape(n, -1, h, w)

    which is the same numbers `conv2d` produces, and is a form the caller can
    vary -- a different weight sharing, a learned aggregation over the window,
    a sparsity pattern -- without needing a kernel written for it.

    The gradient comes from the gather underneath, so a position read by
    several overlapping blocks accumulates all of their gradients.

    `kernel_size`, `dilation`, `padding` and `stride` are each one integer for
    every spatial axis, or a single integer meaning the same for all of them.
    Any number of spatial axes is allowed rather than only the two
    `torch.nn.functional.unfold` accepts, so a 3-D convolution is this same
    matrix product with a rank-three kernel.
    """

    tensor = _atleast_tensor(input)
    if tensor.ndim() < 3:
        raise ValueError(
            "unfold expects a batch, a channel and at least one spatial axis, "
            f"got {tensor.ndim()} dimensions"
        )
    batch, channels, *spatial = [int(size) for size in tensor.shape]
    _kernel, margin, padded, index = _sliding_geometry(
        "unfold", tuple(spatial), kernel_size, dilation, padding, stride
    )
    if any(margin):
        # `pad` takes its axes innermost-first, two entries each.
        flat_padding: list[int] = []
        for pad in reversed(margin):
            flat_padding += [pad, pad]
        tensor = _F.pad(tensor, flat_padding)

    taps, columns = index.shape
    # The flattened spatial size is spelled out rather than inferred with -1:
    # an empty batch or an empty channel axis leaves nothing for the inference
    # to divide by, and the failure would read as a reshape error rather than
    # as the empty answer it should be.
    plane = _element_count(padded)
    gathered = _F.index_select(
        tensor.reshape(batch, channels, plane),
        2,
        _index_tensor(index.reshape(-1), tensor),
    )
    # `(n, c, taps * blocks)` and `(n, c * taps, blocks)` are the same buffer:
    # the tap axis is already between the channel and the block.
    return gathered.reshape(batch, channels * taps, columns)


def fold(
    input: object,
    output_size: object,
    kernel_size: object,
    dilation: object = 1,
    padding: object = 0,
    stride: object = 1,
) -> Tensor:
    """Sum the sliding blocks of `input` back into one `output_size` plane.

    The inverse of `unfold` in the only sense a lossy map has one: it is its
    adjoint, so `(unfold(x) * y).sum()` and `(x * fold(y)).sum()` are the same
    number -- to within the order the two sums are taken in, which is a ULP or
    so, not to the bit. Positions covered by more than one block are *summed*,
    which is what makes this the backward of a convolution written as a matrix
    product rather than a way of putting an image back together. To average
    instead, fold a matching tensor of ones and divide by it.

    Against the gradient the agreement is exact, and not maintained by hand:
    the two functions share no code, and `unfold`'s backward is bit-identical
    to `fold` because the backward of a gather is a scatter-add over the very
    positions the gather read.

    `output_size` is the spatial shape to fold into, before padding is removed.
    The remaining arguments mean what they mean in `unfold`, and must be the
    ones that produced the input.
    """

    tensor = _atleast_tensor(input)
    if tensor.ndim() != 3:
        raise ValueError(
            "fold expects a three-dimensional input of (batch, channels * taps, "
            f"blocks), got {tensor.ndim()} dimensions"
        )
    batch, packed, columns = [int(size) for size in tensor.shape]
    rank = _sliding_rank(output_size, kernel_size, stride, padding, dilation)
    spatial = _sliding_argument(output_size, "output_size", rank, 0, "fold")
    _kernel, margin, padded, index = _sliding_geometry(
        "fold", spatial, kernel_size, dilation, padding, stride
    )

    taps, expected = index.shape
    if packed % taps:
        raise ValueError(
            f"fold needs the packed channel count ({packed}) to divide by the "
            f"number of kernel taps ({taps})"
        )
    if columns != expected:
        raise ValueError(
            f"fold expects {expected} block(s) for an output of {spatial} under "
            f"this geometry, got {columns}"
        )
    channels = packed // taps

    # The map is the same for every image and every channel, so it is expanded
    # rather than tiled: materialising one copy per (batch, channel) would cost
    # more memory than the data being folded.
    positions = broadcast_to(
        _index_tensor(index.reshape(1, 1, -1), tensor),
        (batch, channels, taps * columns),
    )
    target = Tensor.zeros(
        [batch, channels, _element_count(padded)],
        dtype=tensor.dtype,
        device=_C.Device(tensor.device),
    )
    summed = _F.scatter_add(
        target, 2, positions, tensor.reshape(batch, channels, taps * columns)
    )
    folded = summed.reshape([batch, channels, *padded])
    for axis, (pad, size) in enumerate(zip(margin, spatial)):
        if pad:
            folded = _F.narrow(folded, 2 + axis, pad, size)
    return folded


_NN_EXTRAS = (
    "affine_grid",
    "alpha_dropout",
    "avg_pool3d",
    "channel_shuffle",
    "conv3d",
    "dropout1d",
    "dropout3d",
    "embedding",
    "embedding_bag",
    "feature_alpha_dropout",
    "fold",
    "group_norm",
    "gumbel_softmax",
    "instance_norm",
    "local_response_norm",
    "lp_pool1d",
    "lp_pool2d",
    "nll_loss",
    "pixel_shuffle",
    "max_pool3d",
    "max_unpool1d",
    "max_unpool2d",
    "pixel_unshuffle",
    "prelu",
    "rrelu",
    "unfold",
)
