# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Draws from the three distributions a tensor library is asked for.

`rand` and `randn` are the kernels -- a uniform and a standard normal. Each
function here is one of those transformed: a Bernoulli draw is a uniform
compared against the probability, a general normal is the standard one scaled
and shifted, and a categorical draw is a uniform located in the cumulative
distribution. Writing them as transformations rather than as three more
generators means one stream of randomness, one seed, and no second
implementation to disagree with the first.
"""

from __future__ import annotations

import operator as _operator

from . import _core as _C
from ._shape import _atleast_tensor

Tensor = _C.Tensor
_F = _C.functional


def bernoulli(input: object) -> Tensor:
    """A 0/1 draw per element, with `input` giving each element's probability.

    A uniform draw below the probability, which is the definition. The result
    is float, of the input's dtype, rather than boolean: it is a *sample*, and
    the usual next thing to do with it is arithmetic.
    """

    probabilities = _atleast_tensor(input)
    if "float" not in str(probabilities.dtype):
        raise ValueError(
            f"bernoulli requires a floating point tensor of probabilities, "
            f"got {probabilities.dtype}"
        )
    draw = _C.Tensor.rand_like(probabilities)
    return (draw < probabilities).astype(str(probabilities.dtype))


def normal(mean: object = 0.0, std: object = 1.0, size: object = None) -> Tensor:
    """A normal draw, shifted by `mean` and scaled by `std`.

    With `size` omitted the shape comes from `mean` and `std`, whichever is a
    tensor -- so `normal(mu, sigma)` draws one value per element of them, and
    `normal(0.0, 1.0, (3, 4))` draws a block of the shape asked for.
    """

    mean_is_tensor = isinstance(mean, Tensor)
    std_is_tensor = isinstance(std, Tensor)

    if size is not None:
        shape = [
            _operator.index(dim) for dim in (size if _is_sequence(size) else (size,))
        ]
    elif mean_is_tensor or std_is_tensor:
        template = mean if mean_is_tensor else std
        shape = list(template.shape)
    else:
        raise ValueError(
            "normal needs a size when neither mean nor std is a tensor to take "
            "one from"
        )

    dtype = "float32"
    for operand in (mean, std):
        if isinstance(operand, Tensor) and "float64" in str(operand.dtype):
            dtype = "float64"

    if std_is_tensor:
        negative = _F.amin(std).item() < 0.0
    else:
        negative = float(std) < 0.0
    if negative:
        raise ValueError("normal requires a non-negative standard deviation")

    draw = _C.Tensor.randn(*shape, dtype=dtype)
    return draw * std + mean


def _is_sequence(value: object) -> bool:
    return isinstance(value, (list, tuple)) or hasattr(value, "__len__")


def multinomial(input: object, num_samples: int, replacement: bool = False) -> Tensor:
    """Draw `num_samples` indices with probability proportional to `input`.

    `input` is a vector of non-negative weights, or a batch of them as rows.
    They need not sum to one; they are normalized here, which is what makes a
    row of counts as usable as a row of probabilities.

    With `replacement` the draw is a uniform located in the cumulative
    distribution -- one `searchsorted` per sample. Without it, the draw is the
    order statistic of `log(w) + Gumbel noise`, which is exactly a weighted
    sample without replacement and takes one sort instead of a loop that
    removes a row's chosen entry and renormalizes `num_samples` times.
    """

    weights = _atleast_tensor(input)
    if weights.ndim() not in (1, 2):
        raise ValueError(
            f"multinomial requires a 1-D or 2-D tensor of weights, got {weights.ndim()}"
        )
    if "float" not in str(weights.dtype):
        weights = weights.astype("float64")

    flat = weights.ndim() == 1
    rows = weights.reshape(1, -1) if flat else weights
    categories = rows.shape[1]
    count = _operator.index(num_samples)
    if count < 0:
        raise ValueError(
            f"multinomial requires a non-negative num_samples, got {count}"
        )
    if not replacement and count > categories:
        raise ValueError(
            f"multinomial cannot draw {count} of {categories} categories without "
            "replacement"
        )
    if _F.amin(rows).item() < 0.0:
        raise ValueError("multinomial requires non-negative weights")

    totals = _F.sum(rows, [1], True)
    if _F.amin(totals).item() <= 0.0:
        raise ValueError("multinomial requires each row of weights to sum above zero")
    probabilities = rows / totals

    if replacement:
        # Where a uniform draw lands in the cumulative distribution is the
        # index it selects, which is the inverse-transform definition.
        cumulative = _F.cumsum(probabilities, 1)
        draws = _C.Tensor.rand(rows.shape[0], count, dtype=str(rows.dtype))
        picked = _stack_rows(
            [
                _F.searchsorted(_row(cumulative, r), _row(draws, r))
                for r in range(rows.shape[0])
            ]
        )
        # A draw of exactly 1.0, or a cumulative sum that rounds a hair under
        # it, would land one past the end.
        picked = _F.clamp(picked, 0, categories - 1)
    else:
        # Gumbel top-k: the largest `k` of `log(w) + Gumbel(0, 1)` are exactly
        # a weighted sample without replacement, which is why no removal loop
        # is needed.
        eps = 1e-20
        uniform = _C.Tensor.rand_like(probabilities)
        keys = _F.log(probabilities + eps) - _F.log(-_F.log(uniform + eps) + eps)
        picked = _F.topk(keys, count, 1, True, True)[1]

    return picked.reshape(-1) if flat else picked


def _row(tensor: Tensor, index: int) -> Tensor:
    return _F.squeeze(_F.narrow(tensor, 0, index, 1), 0)


def _stack_rows(rows: list[Tensor]) -> Tensor:
    return _F.stack(rows, 0)
