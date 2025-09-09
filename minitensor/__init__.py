# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

# Import the compiled Rust extension
try:
    from . import _core as _minitensor_core
except ImportError as e:
    raise ImportError(
        "The minitensor core extension is not built. "
        "Run `maturin develop` or install the package."
    ) from e

import sys

from . import functional, nn, optim

# Re-export core classes and functions
from .tensor import Tensor, set_default_dtype, get_default_dtype

try:
    from . import numpy_compat
except ImportError:
    numpy_compat = None

# Custom operations and plugin system (if available)
try:
    from ._core import (
        execute_custom_op_py,
        is_custom_op_registered_py,
        list_custom_ops_py,
    )
    from ._core import plugins as _plugins
    from ._core import register_example_custom_ops, unregister_custom_op_py

    plugins = _plugins
    sys.modules[__name__ + ".plugins"] = plugins
except Exception:
    execute_custom_op_py = None
    is_custom_op_registered_py = None
    list_custom_ops_py = None
    register_example_custom_ops = None
    unregister_custom_op_py = None
    plugins = None

# Serialization (if available)
try:
    serialization = _minitensor_core.serialization
except Exception:
    serialization = None

# Version information
try:
    from ._version import __version__, __version_tuple__
except ImportError:
    # Fallback version if _version.py is not available
    __version__ = "0.1.0"
    __version_tuple__ = (0, 1, 0)

# Also try to get version from Rust extension if available
try:
    _rust_version = _minitensor_core.__version__
    # Use Rust version if it's different (for development builds)
    if _rust_version != __version__:
        __version__ = _rust_version
except (AttributeError, NameError):
    pass


# Core tensor creation functions
def zeros(*args, **kwargs):
    """Create a tensor filled with zeros."""
    return Tensor.zeros(*args, **kwargs)


def ones(*args, **kwargs):
    """Create a tensor filled with ones."""
    return Tensor.ones(*args, **kwargs)


def rand(*args, **kwargs):
    """Create a tensor with random values from uniform distribution."""
    return Tensor.rand(*args, **kwargs)


def randn(*args, **kwargs):
    """Create a tensor with random values from normal distribution."""
    return Tensor.randn(*args, **kwargs)


def eye(*args, **kwargs):
    """Create an identity matrix."""
    return Tensor.eye(*args, **kwargs)


def full(*args, **kwargs):
    """Create a tensor filled with a specific value."""
    return Tensor.full(*args, **kwargs)


def arange(*args, **kwargs):
    """Create a tensor with values from a range."""
    return Tensor.arange(*args, **kwargs)


def from_numpy(array, requires_grad=False):
    """Create a tensor from a NumPy array."""
    try:
        return Tensor.from_numpy(array, requires_grad)
    except AttributeError:
        # Fallback if Rust extension is not available
        raise NotImplementedError(
            "from_numpy requires the Rust extension to be built. Please build the project with 'maturin develop' or 'pip install -e .'"
        )


def from_numpy_shared(array, requires_grad=False):
    """Create a tensor from a NumPy array with zero-copy when possible."""
    try:
        return Tensor.from_numpy_shared(array, requires_grad)
    except AttributeError:
        # Fallback to regular from_numpy
        return from_numpy(array, requires_grad)


# NumPy compatibility functions (commonly used ones at top level)
def asarray(data, dtype=None, requires_grad=False):
    """Convert input to tensor (NumPy compatibility)."""
    return numpy_compat.asarray(data, dtype, requires_grad)


def concatenate(tensors, axis=0):
    """Concatenate tensors along an axis (NumPy compatibility)."""
    return numpy_compat.concatenate(tensors, axis)


def stack(tensors, axis=0):
    """Stack tensors along a new axis (NumPy compatibility)."""
    return numpy_compat.stack(tensors, axis)


def cross(a, b, axis=-1):
    """Compute the 3D cross product (NumPy compatibility)."""
    return numpy_compat.cross(a, b, axis=axis)


# Device management
def device(device_str):
    """Create a device object."""
    return _minitensor_core.Device(device_str)


def cpu():
    """Get CPU device."""
    return device("cpu")


def cuda(device_id=0):
    """Get CUDA device."""
    return device(f"cuda:{device_id}")



__all__ = [
    "Tensor",
    "nn",
    "optim",
    "functional",
    "zeros",
    "ones",
    "rand",
    "randn",
    "eye",
    "full",
    "arange",
    "from_numpy",
    "from_numpy_shared",
    "asarray",
    "concatenate",
    "stack",
    "cross",
    "device",
    "cpu",
    "cuda",
    "set_default_dtype",
    "get_default_dtype",
]

if numpy_compat is not None:
    __all__.append("numpy_compat")
# Add plugins to __all__ if available
if plugins is not None:
    __all__.append("plugins")
if register_example_custom_ops is not None:
    __all__.extend(
        [
            "register_example_custom_ops",
            "unregister_custom_op_py",
            "execute_custom_op_py",
            "list_custom_ops_py",
            "is_custom_op_registered_py",
        ]
    )
if serialization is not None:
    __all__.append("serialization")
