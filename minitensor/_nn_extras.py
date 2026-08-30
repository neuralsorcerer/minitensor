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
        positions = _positions(starts + tap * spaced[0], moved)
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
    "avg_pool3d",
    "conv3d",
    "fold",
    "gumbel_softmax",
    "nll_loss",
    "pixel_shuffle",
    "max_pool3d",
    "pixel_unshuffle",
    "prelu",
    "unfold",
)
