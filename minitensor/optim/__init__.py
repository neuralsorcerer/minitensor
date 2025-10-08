# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Optimizers re-exported from the Rust backend."""

from __future__ import annotations

from typing import List as _List

from .. import _core as _backend

Optimizer = _backend.optim.Optimizer
SGD = _backend.optim.SGD
Adam = _backend.optim.Adam
RMSprop = _backend.optim.RMSprop

__all__: _List[str] = ["Optimizer", "SGD", "Adam", "RMSprop"]

try:
    del annotations  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - best effort cleanup
    pass
