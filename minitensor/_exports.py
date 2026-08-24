# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export lists and binding helpers for the MiniTensor package namespace."""

from __future__ import annotations

import sys as _sys
from collections.abc import MutableMapping

_FUNCTIONAL_FORWARDERS = (
    "cat",
    "stack",
    "split",
    "chunk",
    "index_select",
    "gather",
    "narrow",
    "topk",
    "sort",
    "argsort",
    "median",
    "nanmedian",
    "quantile",
    "nanquantile",
    "nansum",
    "nanmean",
    "nanmax",
    "nanmin",
    "nanamax",
    "nanamin",
    "isnan",
    "isinf",
    "isfinite",
    "nan_to_num",
    "logsumexp",
    "norm",
    "scatter",
    "scatter_add",
    "softmax",
    "log_softmax",
    "masked_softmax",
    "masked_log_softmax",
    "sum",
    "prod",
    "mean",
    "all",
    "any",
    "pad",
    "nonzero",
    "count_nonzero",
    "masked_select",
    "max",
    "min",
    "amax",
    "amin",
    "argmax",
    "argmin",
    "cumsum",
    "cumprod",
    "std",
    "var",
    "relu",
    "hardshrink",
    "sigmoid",
    "softplus",
    "gelu",
    "elu",
    "leaky_relu",
    "selu",
    "silu",
    "softsign",
    "tanh",
    "layer_norm",
    "rms_norm",
    "scaled_dot_product_attention",
    "rope",
    "glu",
    "rsqrt",
    "reciprocal",
    "sign",
    # The basics. These existed only as tensor methods while their own
    # variants (log1p, log2, log10, expm1, rsqrt) were free functions.
    "abs",
    "sqrt",
    "exp",
    "log",
    "pow",
    # Likewise reachable only as methods or dunders: `a @ b` and
    # `a.matmul(b)` worked, `mt.matmul(a, b)` did not.
    "matmul",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "floor_divide",
    "remainder",
    "bitwise_not",
    "reshape",
    "view",
    "triu",
    "tril",
    "diagonal",
    "trace",
    "solve",
    "det",
    "slogdet",
    "inv",
    "cholesky",
    "qr",
    "eigh",
    "eigvalsh",
    "svd",
    "svdvals",
    "einsum",
    "searchsorted",
    "bucketize",
    "histogram",
    "histc",
    "pinv",
    "matrix_rank",
    "cond",
    "lstsq",
    "matrix_power",
    "diag_embed",
    "diag",
    "flatten",
    "ravel",
    "transpose",
    "permute",
    "movedim",
    "moveaxis",
    "swapaxes",
    "swapdims",
    "squeeze",
    "unsqueeze",
    "expand",
    "repeat",
    "repeat_interleave",
    "flip",
    "roll",
    "clip",
    "clamp",
    "clamp_min",
    "clamp_max",
    "round",
    "floor",
    "ceil",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "asinh",
    "acosh",
    "atanh",
    "log1p",
    "log2",
    "log10",
    "erf",
    "erfc",
    "expm1",
    "logaddexp",
    "maximum",
    "minimum",
    "isclose",
    "array_equal",
    "allclose",
    "where",
    "one_hot",
    "bincount",
    "masked_fill",
)

# Public members of `functional` that deliberately stay namespaced. These are
# the layer-shaped ops -- they take weights, running statistics, or a training
# flag, so `mt.functional.conv2d(x, w, b)` reads better at a call site than a
# bare `mt.conv2d`, and the top-level namespace stays about tensor math.
#
# Listing them is what makes `_bind_functional_forwarders` able to check both
# directions. Forwarding alone only catches a name that disappeared from
# `functional`; a name *added* to `functional` and forgotten here would simply
# never show up as `mt.<name>`, with nothing to notice.
_FUNCTIONAL_ONLY = (
    "adaptive_avg_pool1d",
    "adaptive_avg_pool2d",
    "adaptive_max_pool1d",
    "adaptive_max_pool2d",
    "avg_pool1d",
    "avg_pool2d",
    "batch_norm",
    "binary_cross_entropy",
    "binary_cross_entropy_with_logits",
    "conv1d",
    "conv2d",
    "conv_transpose1d",
    "conv_transpose2d",
    "cross_entropy",
    "dense_layer",
    "dropout",
    "dropout2d",
    "log_cosh_loss",
    "max_pool1d",
    "max_pool2d",
    "mse_loss",
    "smooth_l1_loss",
    "interpolate",
    "huber_loss",
    "l1_loss",
    "kl_div",
    "focal_loss",
    # Gradient utilities. `nn` is the conventional place to look for these,
    # so they stay under `mt.nn` rather than becoming `mt.clip_grad_norm_`.
    "clip_grad_norm_",
    "clip_grad_value_",
    "grad_norm",
    "count_parameters_with_gradients",
    # Exported at the top level by the core module itself, so forwarding them
    # here would be a second binding of the same object.
    "bmm",
    "dot",
)


def _public_namespace() -> MutableMapping[str, object]:
    return _sys.modules["minitensor"].__dict__


def _find_duplicate_names(names: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for name in names:
        if name in seen:
            duplicates.add(name)
        else:
            seen.add(name)

    return sorted(duplicates)


def _ensure_unique_names(names: tuple[str, ...], label: str) -> None:
    duplicates = _find_duplicate_names(names)
    if duplicates:
        raise RuntimeError(f"Duplicate {label}: " + ", ".join(duplicates))


def _bind_functional_forwarders(
    names: tuple[str, ...], namespace: MutableMapping[str, object] | None = None
) -> None:
    namespace = _public_namespace() if namespace is None else namespace
    functional = namespace["functional"]

    _ensure_unique_names(names, "functional forwarders")

    missing = [name for name in names if not hasattr(functional, name)]
    if missing:
        raise RuntimeError("Missing functional forwarders: " + ", ".join(missing))

    # Every public name in `functional` must be accounted for: forwarded to the
    # top level, or listed as deliberately namespaced. Without this, adding an
    # op to `functional` and forgetting this file leaves it reachable only as
    # `mt.functional.<name>`, and nothing says so.
    unaccounted = sorted(
        name
        for name in dir(functional)
        if not name.startswith("_")
        and name not in set(names)
        and name not in set(_FUNCTIONAL_ONLY)
    )
    if unaccounted:
        raise RuntimeError(
            "functional exports not listed in _exports.py: "
            + ", ".join(unaccounted)
            + " -- add each to _FUNCTIONAL_FORWARDERS to expose it as mt.<name>, "
            "or to _FUNCTIONAL_ONLY to keep it namespaced"
        )

    for name in names:
        namespace[name] = getattr(functional, name)
