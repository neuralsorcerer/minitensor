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
    "quantile",
    "nanquantile",
    "nansum",
    "nanmean",
    "nanmax",
    "nanmin",
    "nan_to_num",
    "logsumexp",
    "softmax",
    "log_softmax",
    "masked_softmax",
    "masked_log_softmax",
    "sum",
    "prod",
    "mean",
    "all",
    "any",
    "max",
    "min",
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
    "selu",
    "silu",
    "softsign",
    "tanh",
    "layer_norm",
    "rsqrt",
    "reciprocal",
    "sign",
    "reshape",
    "view",
    "triu",
    "tril",
    "diagonal",
    "trace",
    "solve",
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
    "expm1",
    "logaddexp",
    "maximum",
    "minimum",
    "array_equal",
    "allclose",
    "where",
    "one_hot",
    "masked_fill",
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

    for name in names:
        namespace[name] = getattr(functional, name)
