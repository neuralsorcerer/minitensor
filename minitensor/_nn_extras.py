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

`cross_entropy` is the one that does have a kernel, because it has an
analytical backward worth having -- and `nll_loss` is the half of it that takes
log-probabilities it did not compute itself, which is what makes the two
different functions rather than one written twice.
"""

from __future__ import annotations

from . import _core as _C
from ._shape import _atleast_tensor, _normalize_axis

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
        raise ValueError(f"pixel_shuffle requires a positive factor, got {upscale_factor}")
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

    unpacked = tensor.reshape(
        [*batch, channels, out_height, factor, out_width, factor]
    )
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
_NN_EXTRAS = (
    "gumbel_softmax",
    "nll_loss",
    "pixel_shuffle",
    "pixel_unshuffle",
    "prelu",
)
