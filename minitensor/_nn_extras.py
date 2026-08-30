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

import operator as _operator

import numpy as _np

from . import _core as _C
from ._shape import _atleast_tensor, _normalize_axis, broadcast_to

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


def _positions(index: "_np.ndarray", like: Tensor) -> Tensor:
    """The index map as an int64 tensor beside the data it will address."""

    return Tensor.from_numpy(index.astype(_np.int64)).to(_C.Device(like.device))


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
    plane = int(_np.prod(padded, dtype=_np.int64))
    gathered = _F.index_select(
        tensor.reshape(batch, channels, plane), 2, _positions(index.reshape(-1), tensor)
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
        _positions(index.reshape(1, 1, -1), tensor), (batch, channels, taps * columns)
    )
    target = Tensor.zeros(
        [batch, channels, int(_np.prod(padded, dtype=_np.int64))],
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
    "fold",
    "gumbel_softmax",
    "nll_loss",
    "pixel_shuffle",
    "pixel_unshuffle",
    "prelu",
    "unfold",
)
