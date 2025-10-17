# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from ._core import numpy_compat as _numpy_compat

cross = _numpy_compat.cross
empty_like = _numpy_compat.empty_like


# Re-export remaining public symbols from the Rust implementation
__all__ = [name for name in dir(_numpy_compat) if not name.startswith("_")]
for name in __all__:
    globals()[name] = getattr(_numpy_compat, name)
