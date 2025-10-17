# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Expose functional and NN helpers directly from the Rust backend."""

from ._core import functional as _functional
from ._core import nn as _nn

__all__ = [name for name in dir(_functional) if not name.startswith("_")]
for name in __all__:
    globals()[name] = getattr(_functional, name)

_NN_FUNCTION_EXPORTS = [
    name
    for name in dir(_nn)
    if not name.startswith("_") and name[0].islower() and callable(getattr(_nn, name))
]

for name in _NN_FUNCTION_EXPORTS:
    globals()[name] = getattr(_nn, name)
    if name not in __all__:
        __all__.append(name)
