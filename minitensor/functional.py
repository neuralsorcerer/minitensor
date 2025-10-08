# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Functional interface exposing Rust-backed tensor operations."""

from __future__ import annotations

try:
    from . import _core as _backend
except ImportError as exc:  # pragma: no cover - surfaced during import
    raise ImportError(
        "The minitensor core extension is not built. "
        "Run `maturin develop` or install the package."
    ) from exc


_FUNCTIONAL_EXPORTS = (
    "flatten",
    "ravel",
    "reshape",
    "view",
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
    "narrow",
    "cat",
    "stack",
    "split",
    "chunk",
    "index_select",
    "gather",
    "where",
    "masked_fill",
    "softmax",
    "log_softmax",
    "logsumexp",
    "relu",
    "hardshrink",
    "sigmoid",
    "softplus",
    "gelu",
    "elu",
    "selu",
    "silu",
    "softsign",
    "tanh",
    "log1p",
    "expm1",
    "sin",
    "cos",
    "tan",
    "rsqrt",
    "logaddexp",
    "triu",
    "tril",
    "topk",
    "sort",
    "argsort",
    "median",
    "layer_norm",
)

_NN_EXPORTS = (
    "dense_layer",
    "conv2d",
    "batch_norm",
    "dropout",
    "dropout2d",
    "mse_loss",
    "cross_entropy",
    "binary_cross_entropy",
)

_FUNCTIONAL_MODULE = _backend.functional
_NN_MODULE = _backend.nn

for _name in _FUNCTIONAL_EXPORTS:
    globals()[_name] = getattr(_FUNCTIONAL_MODULE, _name)

for _name in _NN_EXPORTS:
    globals()[_name] = getattr(_NN_MODULE, _name)

__all__ = sorted(set(_FUNCTIONAL_EXPORTS) | set(_NN_EXPORTS))
