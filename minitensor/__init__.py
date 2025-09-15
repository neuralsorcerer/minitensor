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
from typing import Optional, Sequence, Union

from . import functional, nn, optim

# Re-export core classes and functions
from .tensor import Tensor, get_default_dtype, set_default_dtype

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
    from ._core import (
        register_example_custom_ops,
        unregister_custom_op_py,
    )

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


def cat(tensors, dim: int = 0):
    """Concatenate tensors along an existing dimension."""
    return functional.cat(tensors, dim)


def stack(tensors, dim: int = 0):
    """Stack tensors along a new dimension."""
    return functional.stack(tensors, dim)


def split(input, split_size_or_sections, dim: int = 0):
    """Split ``input`` into chunks along ``dim``."""
    return functional.split(input, split_size_or_sections, dim)


def chunk(input, chunks: int, dim: int = 0):
    """Split ``input`` into ``chunks`` along ``dim``."""
    return functional.chunk(input, chunks, dim)


def index_select(input, dim: int, indices):
    """Select elements along ``dim`` using integer ``indices``."""
    return functional.index_select(input, dim, indices)


def gather(input, dim: int, index):
    """Gather elements from ``input`` along ``dim`` using ``index`` tensor."""
    return functional.gather(input, dim, index)


def narrow(input, dim: int, start: int, length: int):
    """Narrow ``input`` along ``dim`` starting at ``start`` for ``length`` elements."""
    return functional.narrow(input, dim, start, length)


def cross(a, b, axis=-1):
    """Compute the 3D cross product (NumPy compatibility)."""
    return numpy_compat.cross(a, b, axis=axis)


# Functional operation wrappers
def reshape(input, shape):
    """Return ``input`` with a new shape."""
    return functional.reshape(input, shape)


def view(input, *shape):
    """Return a view of ``input`` with a new shape."""
    return functional.view(input, *shape)


def flatten(input, start_dim: int = 0, end_dim: int = -1):
    """Return a view of ``input`` with dimensions flattened."""
    return functional.flatten(input, start_dim, end_dim)


def ravel(input):
    """Flatten all dimensions of ``input``."""
    return functional.ravel(input)


def transpose(input, dim0: int, dim1: int):
    """Swap two dimensions of ``input``."""
    return functional.transpose(input, dim0, dim1)


def permute(input, dims: Sequence[int]):
    """Reorder dimensions of ``input`` according to ``dims``."""
    return functional.permute(input, dims)


def movedim(input, source, destination):
    """Move tensor dimensions to new positions."""
    return functional.movedim(input, source, destination)


moveaxis = movedim


def swapaxes(input, axis0: int, axis1: int):
    """Swap two axes of ``input``."""
    return functional.swapaxes(input, axis0, axis1)


swapdims = swapaxes


def squeeze(input, dim: Optional[int] = None):
    """Remove dimensions of size 1 from ``input``."""
    return functional.squeeze(input, dim)


def unsqueeze(input, dim: int):
    """Insert a dimension of size 1 at position ``dim`` in ``input``."""
    return functional.unsqueeze(input, dim)


def expand(input, *shape: int):
    """Return ``input`` expanded to a larger ``shape``."""
    return functional.expand(input, *shape)


def repeat(input, *repeats: int):
    """Repeat ``input`` along each dimension."""
    return functional.repeat(input, *repeats)


def flip(input, dims: Union[int, Sequence[int]]):
    """Flip ``input`` along specified dimensions."""
    return functional.flip(input, dims)


def roll(input, shifts, dims: Optional[Union[int, Sequence[int]]] = None):
    """Roll ``input`` along specified dimensions."""
    return functional.roll(input, shifts, dims)


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
    "cat",
    "stack",
    "chunk",
    "index_select",
    "gather",
    "narrow",
    "reshape",
    "view",
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
    "flip",
    "roll",
    "split",
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
