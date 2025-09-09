# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from ._core import numpy_compat as _numpy_compat
from .tensor import Tensor


def cross(a, b, axis=-1):
    """Compute 3D cross product."""
    rust_a = getattr(a, "_tensor", a)
    rust_b = getattr(b, "_tensor", b)
    rust_result = _numpy_compat.cross(rust_a, rust_b, axis=axis)
    result = Tensor.__new__(Tensor)
    result._tensor = rust_result
    return result


def empty_like(a, dtype=None):
    rust_a = getattr(a, "_tensor", a)
    rust_result = _numpy_compat.empty_like(rust_a, dtype)
    result = Tensor.__new__(Tensor)
    result._tensor = rust_result
    return result


# Re-export remaining public symbols from the Rust implementation
__all__ = [
    name
    for name in dir(_numpy_compat)
    if not name.startswith("_") and name not in {"cross", "empty_like"}
] + ["cross", "empty_like"]
for name in __all__:
    if name not in {"cross", "empty_like"}:
        globals()[name] = getattr(_numpy_compat, name)
