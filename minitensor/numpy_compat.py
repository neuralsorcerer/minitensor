# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from ._core import numpy_compat as _numpy_compat
from .tensor import Tensor


def cross(a, b):
    """Compute 3D cross product with fallback to NumPy."""
    rust_a = getattr(a, "_tensor", a)
    rust_b = getattr(b, "_tensor", b)
    try:
        rust_result = _numpy_compat.cross(rust_a, rust_b)
        result = Tensor.__new__(Tensor)
        result._tensor = rust_result
        return result
    except NotImplementedError:
        a_tensor = Tensor.__new__(Tensor)
        a_tensor._tensor = rust_a
        b_tensor = Tensor.__new__(Tensor)
        b_tensor._tensor = rust_b
        np_result = np.cross(a_tensor.numpy(), b_tensor.numpy())
        return Tensor.from_numpy(np_result)


# Re-export remaining public symbols from the Rust implementation
__all__ = [
    name for name in dir(_numpy_compat) if not name.startswith("_") and name != "cross"
] + ["cross"]
for name in __all__:
    if name != "cross":
        globals()[name] = getattr(_numpy_compat, name)
