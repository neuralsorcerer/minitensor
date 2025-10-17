# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Expose optimizers from the Rust backend."""

from .._core import optim as _optim

__all__ = [name for name in dir(_optim) if not name.startswith("_")]
for name in __all__:
    globals()[name] = getattr(_optim, name)
