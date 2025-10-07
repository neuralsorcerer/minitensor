# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tensor class with NumPy compatibility and automatic differentiation support.
"""

from __future__ import annotations

import importlib

try:
    from ._backend import (
        autograd_get_gradient,
        autograd_is_graph_consumed,
        autograd_mark_graph_consumed,
        core as _minitensor_core,
    )
except ImportError as e:
    raise ImportError(
        "The minitensor core extension is not built. "
        "Run `pip install -e .` or `maturin develop` to compile the Rust backend."
    ) from e

if not hasattr(_minitensor_core.Tensor, "detach_"):
    raise RuntimeError(
        "The loaded Rust backend is missing Tensor.detach_. "
        "Rebuild minitensor (for example with `pip install -e .`) so that the "
        "Python package and compiled extension stay in sync."
    )

try:  # pragma: no cover - debug helpers may not be compiled in minimal builds
    _TensorDebugger = _minitensor_core.debug.TensorDebugger
except AttributeError:  # pragma: no cover - debug module unavailable
    _TensorDebugger = None

if _TensorDebugger is not None:  # pragma: no branch - evaluate once during import
    try:
        _TENSOR_DEBUGGER = _TensorDebugger()
    except Exception:  # pragma: no cover - instantiation failures should not break import
        _TENSOR_DEBUGGER = None
else:
    _TENSOR_DEBUGGER = None

from numbers import Integral, Real
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Union
from weakref import WeakSet

np: Any | None = None
_HAS_NUMPY = False
_TENSOR_TO_NP_DTYPE: Dict[str, Any] = {}
_NP_TO_TENSOR_DTYPE: Dict[Any, str] = {}
_NUMPY_GENERIC: Tuple[type, ...] = ()
_NUMPY_ARRAY: Tuple[type, ...] = ()

_GLOBAL_UFUNC_BINARY_MAP: Dict[Any, Any] = {}
_GLOBAL_UFUNC_UNARY_MAP: Dict[Any, Any] = {}


def _refresh_numpy_dispatch_tables() -> None:
    """Rebuild NumPy ufunc dispatch tables based on availability."""

    _GLOBAL_UFUNC_BINARY_MAP.clear()
    _GLOBAL_UFUNC_UNARY_MAP.clear()

    if not _HAS_NUMPY or np is None:
        return

    _GLOBAL_UFUNC_BINARY_MAP.update(
        {
            np.add: lambda a, b: a + b,
            np.subtract: lambda a, b: a - b,
            np.multiply: lambda a, b: a * b,
            np.true_divide: lambda a, b: a / b,
            np.power: lambda a, b: a.pow(b),
            np.maximum: lambda a, b: a.maximum(b),
            np.minimum: lambda a, b: a.minimum(b),
        }
    )

    _GLOBAL_UFUNC_UNARY_MAP.update(
        {
            np.negative: lambda a: -a,
            np.abs: lambda a: a.abs(),
            np.exp: lambda a: a.exp(),
            np.log: lambda a: a.log(),
            np.sin: lambda a: a.sin(),
            np.cos: lambda a: a.cos(),
            np.tan: lambda a: a.tan(),
            np.sqrt: lambda a: a.sqrt(),
        }
    )


def _initialize_numpy_bindings(np_module: Any) -> None:
    """Populate cached NumPy metadata after importing the module."""

    global np, _HAS_NUMPY, _TENSOR_TO_NP_DTYPE, _NP_TO_TENSOR_DTYPE
    global _NUMPY_GENERIC, _NUMPY_ARRAY

    np = np_module
    _HAS_NUMPY = True
    _TENSOR_TO_NP_DTYPE = {
        "float32": np_module.dtype(np_module.float32),
        "float64": np_module.dtype(np_module.float64),
        "int32": np_module.dtype(np_module.int32),
        "int64": np_module.dtype(np_module.int64),
        "bool": np_module.dtype(np_module.bool_),
    }
    _NP_TO_TENSOR_DTYPE = {v: k for k, v in _TENSOR_TO_NP_DTYPE.items()}
    _NUMPY_GENERIC = (np_module.generic,)
    _NUMPY_ARRAY = (np_module.ndarray,)
    _refresh_numpy_dispatch_tables()


def _attempt_enable_numpy() -> bool:
    """Import NumPy lazily and cache metadata if it becomes available."""

    if _HAS_NUMPY:
        return True

    try:
        np_module = importlib.import_module("numpy")
    except ModuleNotFoundError:
        return False

    _initialize_numpy_bindings(np_module)
    return True


def _ensure_numpy_available(message: str) -> None:
    """Ensure NumPy is importable, raising ``ModuleNotFoundError`` otherwise."""

    if _attempt_enable_numpy():
        return
    raise ModuleNotFoundError(message)


_attempt_enable_numpy()

# Supported dtype names irrespective of NumPy availability
_SUPPORTED_DTYPES = {"float32", "float64", "int32", "int64", "bool"}

_VALID_DTYPES = set(_SUPPORTED_DTYPES)


def _query_tensor_debug_flag(tensor_core: Any, attribute: str, default: bool) -> bool:
    """Best-effort helper to retrieve autograd metadata via debug bindings."""

    debugger = _TENSOR_DEBUGGER
    if debugger is None:
        return default

    try:
        info = debugger.get_info(tensor_core)
    except Exception:
        return default

    value = getattr(info, attribute, None)
    if value is None:
        return default
    return bool(value)



def _normalize_device(device: Optional[str]) -> Optional[str]:
    """Normalize device strings returned from the Rust backend."""

    if device is None:
        return None

    if isinstance(device, str) and device.startswith("device"):
        try:
            inside = device.split("{", 1)[1].split("}", 1)[0]
            fields = {}
            for part in inside.split(","):
                if ":" in part:
                    key, value = part.split(":", 1)
                    fields[key.strip()] = value.strip()
            device_type = fields.get("device_type")
            device_id = fields.get("device_id")
            if device_type:
                if not device_id or device_id in {"none", "default"}:
                    return device_type
                return f"{device_type}:{device_id}"
        except Exception:
            return device
    return device


# Global default dtype management


def set_default_dtype(dtype: str) -> None:
    """Set the global default data type for new tensors."""
    
    try:
        _minitensor_core.set_default_dtype(dtype)
    except Exception as exc:  # pragma: no cover - backend reports invalid dtype
        raise ValueError(f"Unsupported dtype '{dtype}'") from exc


def get_default_dtype() -> str:
    """Get the current global default data type."""
    
    return _minitensor_core.get_default_dtype()


class Tensor:
    """
    A multi-dimensional array with automatic differentiation support and NumPy compatibility.
    This class wraps a Rust backend for efficient tensor computations and provides a unified interface
    for both NumPy and Python.
    """

    # Ensure NumPy treats Tensor as having higher priority in operations
    # so that dispatch prefers Tensor's implementations over NumPy's defaults.
    __array_priority__ = 1000
    __hash__ = object.__hash__

    # Mapping of NumPy ufuncs to Tensor operations. These lambdas ensure that
    # all computations are executed by the Rust backend by leveraging the
    # Tensor's arithmetic and math methods. The dictionaries are shared at the
    # module level so they stay in sync when NumPy becomes available lazily.
    _UFUNC_BINARY_MAP: ClassVar[Dict[Any, Any]] = _GLOBAL_UFUNC_BINARY_MAP
    _UFUNC_UNARY_MAP: ClassVar[Dict[Any, Any]] = _GLOBAL_UFUNC_UNARY_MAP

    _CONSUMED_TENSORS: ClassVar["WeakSet[Tensor]"] = WeakSet()

    @classmethod
    def _wrap_core_tensor(cls, core_tensor: Any) -> "Tensor":
        """Instantiate a ``Tensor`` (or subclass) backed by ``core_tensor``.

        This helper centralises the boilerplate that previously repeated
        ``Tensor.__new__(Tensor)`` followed by manual ``_tensor`` assignment.
        Creating a dedicated wrapper guarantees that every Python ``Tensor``
        instance goes through ``Tensor.__new__`` so debug metadata such as the
        ``_graph_consumed`` flag is initialised consistently.
        """

        instance = cls.__new__(cls)
        instance._tensor = core_tensor
        return instance

    @classmethod
    def _wrap_gradient_tensor(cls, core_tensor: Any) -> "Tensor":
        """Wrap ``core_tensor`` and detach it so gradients stay out of Python autograd."""

        gradient = cls._wrap_core_tensor(core_tensor)
        try:
            if gradient.requires_grad:
                gradient = gradient.detach()
        except AttributeError:  # pragma: no cover - legacy backends may omit setter
            pass
        return gradient

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance._graph_consumed = False
        return instance

    @staticmethod
    def _ensure_on_device(tensor: "Tensor", device: Optional[str]) -> "Tensor":
        """Move ``tensor`` to ``device`` when necessary."""

        if device is None:
            return tensor

        current = _normalize_device(tensor.device)
        if current == device:
            return tensor

        return tensor.to(device)

    @staticmethod
    def _from_array_like(value: Any, device: Optional[str]) -> Optional["Tensor"]:
        """Convert array-like Python inputs to a ``Tensor`` on ``device`` if possible."""

        if not _attempt_enable_numpy():
            return None

        if np is None:  # Defensive: should not happen when attempt succeeds
            return None

        np_source = None
        if isinstance(value, _NUMPY_ARRAY):
            np_source = value
        elif isinstance(value, (list, tuple)):
            try:
                np_source = np.array(value)
            except Exception:  # pragma: no cover - rely on scalar coercion fallback
                np_source = None

        if np_source is None:
            return None

        mapped_dtype = _NP_TO_TENSOR_DTYPE.get(np_source.dtype, None)
        try:
            return Tensor(np_source, dtype=mapped_dtype, device=device)
        except Exception:  # pragma: no cover - fall back to scalar promotion
            return None

    def __init__(
        self,
        data: Any,
        requires_grad: bool = False,
        dtype: Optional[str] = None,
        device=None,
    ):
        """
        Initialize a tensor.

        Args:
            data: Input data (list, numpy array, scalar, or another tensor)
            requires_grad: Whether to track gradients for automatic differentiation
            dtype: Data type ('float32', 'float64', 'int32', 'int64', 'bool')
            device: Device to place tensor on (CPU, CUDA, etc.)

        Examples:
            >>> t1 = Tensor([1, 2, 3])
            >>> t2 = Tensor([[1, 2], [3, 4]], dtype='float64')
            >>> t3 = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        """
        if isinstance(data, Tensor):
            # Copy constructor
            self._tensor = data._tensor.clone()
        else:
            # Create new tensor from data
            if isinstance(device, _minitensor_core.Device):
                device_obj = device
            else:
                normalized_device = _normalize_device(device)
                device_obj = (
                    _minitensor_core.Device(normalized_device)
                    if normalized_device is not None
                    else None
                )
            self._tensor = _minitensor_core.Tensor(
                data, dtype, device_obj, requires_grad
            )

    # Core properties
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get tensor shape as tuple."""
        return tuple(self._tensor.shape)

    @property
    def dtype(self) -> str:
        """Get tensor data type."""
        return self._tensor.dtype

    @property
    def device(self) -> str:
        """Get tensor device."""
        return self._tensor.device

    @property
    def requires_grad(self) -> bool:
        """Check if tensor requires gradients."""
        return self._tensor.requires_grad

    @property
    def grad(self) -> Optional["Tensor"]:
        """Get gradient tensor from the global autograd graph."""
        if not self.requires_grad:
            return None

        rust_grad = autograd_get_gradient(self._tensor)
        if rust_grad is None:
            return None

        return self._wrap_gradient_tensor(rust_grad)

    @property
    def is_leaf(self) -> bool:
        """Return ``True`` when the tensor is a leaf in the autograd graph."""

        attr = getattr(self._tensor, "is_leaf", None)
        if attr is not None:
            return bool(attr)

        return _query_tensor_debug_flag(self._tensor, "is_leaf", True)

    # NumPy compatibility properties
    @property
    def size(self) -> int:
        """Total number of elements."""
        return self._tensor.size

    @property
    def itemsize(self) -> int:
        """Size of each element in bytes."""
        return self._tensor.itemsize

    @property
    def nbytes(self) -> int:
        """Total bytes consumed by the tensor."""
        return self._tensor.nbytes

    @property
    def strides(self) -> Tuple[int, ...]:
        """Strides of the tensor."""
        return tuple(self._tensor.strides)

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self._tensor.ndim()

    @property
    def T(self) -> "Tensor":
        """Transpose."""
        return self.transpose()

    # Basic tensor info methods
    def numel(self) -> int:
        """Get total number of elements."""
        return self._tensor.numel()

    def dim(self) -> int:
        """Get number of dimensions."""
        return self.ndim

    def is_contiguous(self) -> bool:
        """Check if tensor is contiguous in memory."""
        return self._tensor.is_contiguous()

    def element_size(self) -> int:
        """Get size of each element in bytes."""
        return self.itemsize

    # Data conversion methods
    def numpy(self) -> "np.ndarray":
        """Convert to numpy array with zero-copy when possible."""
        _ensure_numpy_available(
            "NumPy is required to materialize Tensor data as a NumPy array."
        )
        try:
            return self._tensor.numpy()
        except NotImplementedError:
            return self._tensor.numpy_copy()

    def numpy_copy(self) -> "np.ndarray":
        """Convert to numpy array with explicit copy."""
        _ensure_numpy_available(
            "NumPy is required to materialize Tensor data as a NumPy array."
        )
        return self._tensor.numpy_copy()

    def __array__(self, dtype: Optional["np.dtype"] = None) -> "np.ndarray":
        """Support NumPy's array protocol for seamless interoperability."""
        _ensure_numpy_available(
            "NumPy is required to expose Tensor data through the array protocol."
        )
        array = self.numpy()
        if dtype is not None:
            return array.astype(dtype, copy=False)
        return array

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Dispatch NumPy ufuncs to Tensor operations executed in Rust."""
        _ensure_numpy_available(
            "NumPy is required to dispatch ufuncs for Tensor operands."
        )
        if method != "__call__" or kwargs.get("out") is not None:
            return NotImplemented

        tensor_inputs: List[Tensor] = []
        target_device = _normalize_device(self.device)

        for arg in inputs:
            if isinstance(arg, Tensor):
                tensor_inputs.append(arg)
                continue

            if isinstance(arg, _NUMPY_ARRAY):
                if arg.dtype in _NP_TO_TENSOR_DTYPE:
                    converted = Tensor.from_numpy(arg)
                    tensor_inputs.append(
                        Tensor._ensure_on_device(converted, target_device)
                    )
                    continue
                tensor_inputs.append(Tensor(arg.tolist(), device=target_device))
                continue

            maybe_tensor = Tensor._from_array_like(arg, target_device)
            if maybe_tensor is not None:
                tensor_inputs.append(maybe_tensor)
                continue

            try:
                tensor_inputs.append(Tensor(arg, device=target_device))
            except Exception:
                return NotImplemented

        if ufunc in self._UFUNC_BINARY_MAP and len(tensor_inputs) == 2:
            return self._UFUNC_BINARY_MAP[ufunc](tensor_inputs[0], tensor_inputs[1])

        if ufunc in self._UFUNC_UNARY_MAP and len(tensor_inputs) == 1:
            return self._UFUNC_UNARY_MAP[ufunc](tensor_inputs[0])

        return NotImplemented

    def tolist(self) -> Any:
        """Convert to Python list."""
        return self._tensor.tolist()

    def item(self) -> Union[float, int, bool]:
        """Return the Python scalar value for a single-element tensor."""
        try:
            return self._tensor.item()
        except ValueError as exc:
            raise RuntimeError(str(exc)) from None

    # Tensor manipulation methods
    def reshape(self, *shape: Union[int, Sequence[int]]) -> "Tensor":
        """Reshape tensor to new shape."""
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = list(shape[0])
        else:
            shape = list(shape)

        return self._wrap_core_tensor(self._tensor.reshape(shape))

    def view(self, *shape: Union[int, Sequence[int]]) -> "Tensor":
        """Alias for reshape."""
        return self.reshape(*shape)

    def transpose(self, dim0: int = 0, dim1: int = 1) -> "Tensor":
        """Transpose tensor dimensions."""
        return self._wrap_core_tensor(self._tensor.transpose(dim0, dim1))

    def permute(self, *dims: int) -> "Tensor":
        """Permute tensor dimensions."""
        if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
            dims = list(dims[0])
        else:
            dims = list(dims)
        return self._wrap_core_tensor(self._tensor.permute(dims))

    def movedim(
        self, source: Union[int, Sequence[int]], destination: Union[int, Sequence[int]]
    ) -> "Tensor":
        """Move tensor dimensions to new positions."""

        return self._wrap_core_tensor(self._tensor.movedim(source, destination))

    moveaxis = movedim

    def swapaxes(self, axis0: int, axis1: int) -> "Tensor":
        """Swap two dimensions of the tensor."""

        return self.transpose(axis0, axis1)

    swapdims = swapaxes

    def squeeze(self, dim: Optional[int] = None) -> "Tensor":
        """Remove dimensions of size 1."""
        return self._wrap_core_tensor(self._tensor.squeeze(dim))

    def unsqueeze(self, dim: int) -> "Tensor":
        """Add a dimension of size 1."""
        return self._wrap_core_tensor(self._tensor.unsqueeze(dim))

    def expand(self, *shape: int) -> "Tensor":
        """Expand tensor dimensions without allocating new memory."""
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            dims = list(shape[0])
        else:
            dims = list(shape)
        return self._wrap_core_tensor(self._tensor.expand(dims))

    def repeat(self, *repeats: int) -> "Tensor":
        """Repeat the tensor along each dimension."""

        if len(repeats) == 1 and isinstance(repeats[0], (list, tuple)):
            repeat_spec = repeats[0]
        else:
            repeat_spec = repeats

        return self._wrap_core_tensor(self._tensor.repeat(repeat_spec))

    def repeat_interleave(
        self,
        repeats: Union[int, Sequence[int], "Tensor"],
        dim: Optional[int] = None,
        output_size: Optional[int] = None,
    ) -> "Tensor":
        """Repeat elements of the tensor along a given dimension."""

        return self._wrap_core_tensor(
            self._tensor.repeat_interleave(repeats, dim, output_size)
        )

    def flip(self, dims: Union[int, Sequence[int]]) -> "Tensor":
        """Flip the tensor along given dimensions."""

        if isinstance(dims, int):
            dims_list = [dims]
        else:
            dims_list = list(dims)

        return self._wrap_core_tensor(self._tensor.flip(dims_list))

    def roll(
        self,
        shifts: Union[int, Sequence[int]],
        dims: Optional[Union[int, Sequence[int]]] = None,
    ) -> "Tensor":
        """Roll the tensor along given dimensions with wrap-around."""

        if isinstance(shifts, int):
            shift_list = [int(shifts)]
        else:
            shift_list = [int(s) for s in shifts]

        if dims is None:
            dims_list = None
        else:
            if isinstance(dims, int):
                dims_list = [int(dims)]
            else:
                dims_list = [int(d) for d in dims]

        return self._wrap_core_tensor(self._tensor.roll(shift_list, dims_list))

    def narrow(self, dim: int, start: int, length: int) -> "Tensor":
        """Narrow the tensor along ``dim`` starting at ``start`` for ``length`` elements."""

        return self._wrap_core_tensor(self._tensor.narrow(dim, start, length))

    def index_select(self, dim: int, indices: Sequence[int]) -> "Tensor":
        """Select elements along ``dim`` using integer ``indices``."""

        idx_list = [int(i) for i in indices]
        return self._wrap_core_tensor(self._tensor.index_select(dim, idx_list))

    def gather(self, dim: int, index: "Tensor") -> "Tensor":
        return self._wrap_core_tensor(self._tensor.gather(dim, index._tensor))

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> "Tensor":
        """Flatten tensor dimensions."""
        return self._wrap_core_tensor(self._tensor.flatten(start_dim, end_dim))

    def ravel(self) -> "Tensor":
        """Return flattened tensor (NumPy compatibility)."""
        return self.flatten()

    def split(
        self,
        split_size_or_sections: Union[int, Sequence[int]],
        dim: int = 0,
    ) -> List["Tensor"]:
        """Split the tensor into chunks along ``dim``.

        Args:
            split_size_or_sections: Size of each chunk or list/tuple of sizes
                for each chunk.
            dim: Dimension along which to split. May be negative to index
                from the end.

        Returns:
            List[Tensor]: Tensors resulting from the split.
        """

        if not isinstance(split_size_or_sections, int):
            split_size_or_sections = list(split_size_or_sections)
        parts = self._tensor.split(split_size_or_sections, dim)
        result: List[Tensor] = []
        for p in parts:
            result.append(Tensor._wrap_core_tensor(p))
        return result

    def chunk(self, chunks: int, dim: int = 0) -> List["Tensor"]:
        """Split the tensor into equal sized chunks along ``dim``.

        Args:
            chunks: Number of chunks to return. The tensor size along ``dim``
                must be divisible by ``chunks``.
            dim: Dimension along which to split the tensor.

        Returns:
            List[Tensor]: List of ``chunks`` tensors split from this tensor.
        """

        parts = self._tensor.chunk(int(chunks), dim)
        result = []
        for p in parts:
            result.append(Tensor._wrap_core_tensor(p))
        return result

    # Tensor operations
    def clone(self) -> "Tensor":
        """Create a copy of the tensor."""
        return self._wrap_core_tensor(self._tensor.clone())

    def copy(self) -> "Tensor":
        """Create a copy of the tensor (NumPy compatibility)."""
        return self.clone()

    def detach(self) -> "Tensor":
        """Detach tensor from computation graph."""
        return self._wrap_core_tensor(self._tensor.detach())

    def detach_(self) -> "Tensor":
        """Detach tensor from the computation graph in-place."""

        try:
            self._tensor.detach_()
        except AttributeError as exc:  # pragma: no cover - backend contract violation
            raise RuntimeError(
                "The loaded Rust backend does not provide Tensor.detach_. "
                "Rebuild minitensor so that the Python package and compiled "
                "extension stay in sync."
            ) from exc
        self._graph_consumed = False
        Tensor._CONSUMED_TENSORS.discard(self)
        return self

    def contiguous(self) -> "Tensor":
        """Create a contiguous copy of the tensor."""
        return self._wrap_core_tensor(self._tensor.contiguous())

    def to(
        self,
        device_or_dtype: Optional[Union[str, _minitensor_core.Device]] = None,
        *,
        dtype: Optional[str] = None,
        device: Optional[Union[str, _minitensor_core.Device]] = None,
    ) -> "Tensor":
        """Move the tensor to another device and/or dtype using the Rust backend."""

        target_dtype = dtype

        def _resolve_device(
            spec: Optional[Union[str, _minitensor_core.Device]],
        ) -> Optional[_minitensor_core.Device]:
            if spec is None:
                return None
            if isinstance(spec, _minitensor_core.Device):
                return spec
            if isinstance(spec, str):
                normalized = _normalize_device(spec)
                return _minitensor_core.Device(normalized)
            raise TypeError(
                "to() expects device specifications as strings or Device objects"
            )

        target_device = _resolve_device(device)

        if isinstance(device_or_dtype, _minitensor_core.Device):
            if target_device is not None:
                raise TypeError("to() received multiple device specifications")
            target_device = device_or_dtype
        elif isinstance(device_or_dtype, str):
            normalized = _normalize_device(device_or_dtype)
            if normalized in _VALID_DTYPES:
                if target_dtype is not None and target_dtype != normalized:
                    raise TypeError("dtype specified both positionally and via keyword")
                target_dtype = normalized
            else:
                if target_device is not None:
                    raise TypeError("to() received multiple device specifications")
                target_device = _resolve_device(normalized)
        elif device_or_dtype is not None:
            raise TypeError("to() expects dtype or device specifications")

        if target_dtype is not None and target_dtype not in _VALID_DTYPES:
            raise ValueError(f"Unsupported dtype '{target_dtype}'")

        tensor_obj = self._tensor
        mutated = False

        if target_dtype is not None and target_dtype != self.dtype:
            tensor_obj = tensor_obj.astype(target_dtype)
            mutated = True

        if target_device is not None:
            desired_device = _normalize_device(str(target_device))
            current_device = _normalize_device(self.device)
            if desired_device != current_device:
                tensor_obj = tensor_obj.to(target_device)
                mutated = True

        if not mutated:
            return self

        return self._wrap_core_tensor(tensor_obj)

    def cpu(self) -> "Tensor":
        """Move tensor to CPU."""
        return self._wrap_core_tensor(self._tensor.cpu())

    def cuda(self, device: Optional[int] = None) -> "Tensor":
        """Move tensor to a CUDA device using Rust execution."""
        spec = "cuda" if device is None else f"cuda:{device}"
        return self.to(spec)

    def astype(self, dtype: str) -> "Tensor":
        """Convert tensor to a different data type."""
        return self._wrap_core_tensor(self._tensor.astype(dtype))

    # Gradient operations
    def _prepare_backward_gradient(
        self, gradient: Optional["Tensor" | Any]
    ) -> Optional["Tensor"]:
        """Normalise ``gradient`` to share this tensor's dtype and device."""

        if gradient is None:
            if self.numel() != 1:
                raise RuntimeError(
                    "grad can be implicitly created only for scalar outputs"
                )
            return None

        if not isinstance(gradient, Tensor):
            try:
                gradient_tensor = Tensor(
                    gradient,
                    dtype=self.dtype,
                    device=_normalize_device(self.device),
                )
            except Exception as exc:  # pragma: no cover - rely on Tensor errors
                raise TypeError(
                    "backward() expected gradient to be a Tensor or convertible to Tensor"
                ) from exc
        else:
            gradient_tensor = gradient

        expected_device = _normalize_device(self.device)
        gradient_tensor = Tensor._ensure_on_device(gradient_tensor, expected_device)

        if gradient_tensor.shape != self.shape:
            raise RuntimeError(
                "backward() expected gradient tensor with shape "
                f"{self.shape}, but got {gradient_tensor.shape}"
            )

        if gradient_tensor.dtype != self.dtype:
            try:
                gradient_tensor = gradient_tensor.astype(self.dtype)
            except Exception as exc:  # pragma: no cover - backend reports casts
                raise TypeError(
                    "backward() expected gradient tensor with dtype "
                    f"{self.dtype}, but got {gradient_tensor.dtype}"
                ) from exc

        if gradient_tensor.requires_grad:
            gradient_tensor = gradient_tensor.detach()

        return gradient_tensor

    def backward(
        self,
        gradient: Optional["Tensor"] = None,
        retain_graph: bool = False,
        create_graph: bool = False,
    ):
        """Compute gradients via backpropagation."""
        if create_graph:
            raise NotImplementedError(
                "create_graph=True is not supported; all computations execute in the Rust backend"
            )

        if not self.requires_grad and self.is_leaf:
            raise RuntimeError(
                "element 0 of tensors does not require grad and does not have a grad_fn"
            )

        if autograd_is_graph_consumed():
            raise RuntimeError(
                "Computation graph has been freed. Re-run the forward pass or call backward(retain_graph=True)."
            )

        if getattr(self, "_graph_consumed", False):
            raise RuntimeError(
                "Computation graph has been freed. Re-run the forward pass or call backward(retain_graph=True)."
            )

        gradient_tensor = self._prepare_backward_gradient(gradient)

        backend_gradient = None if gradient_tensor is None else gradient_tensor._tensor

        self._tensor.backward(backend_gradient)

        if not retain_graph:
            autograd_mark_graph_consumed()
            self._graph_consumed = True
            Tensor._CONSUMED_TENSORS.add(self)

    @classmethod
    def _reset_graph_consumed_flags(cls) -> None:
        """Clear per-instance graph consumption markers after backend reset."""

        if not cls._CONSUMED_TENSORS:
            return

        stale = list(cls._CONSUMED_TENSORS)
        cls._CONSUMED_TENSORS.clear()
        for tensor in stale:
            tensor._graph_consumed = False

    def requires_grad_(self, requires_grad: bool = True) -> "Tensor":
        """Set requires_grad flag in-place."""
        self._tensor.requires_grad_(requires_grad)
        return self

    def zero_grad(self, set_to_none: bool = False):
        """Zero the gradient."""
        self._tensor.zero_grad(set_to_none)

    # Arithmetic operations with broadcasting support
    def __neg__(self) -> "Tensor":
        """Unary negation returning a Tensor."""
        return self._wrap_core_tensor(self._tensor.__neg__())

    def __add__(self, other: Union["Tensor", float, int]) -> "Tensor":
        return self._wrap_core_tensor(self._tensor.__add__(other))

    def __radd__(self, other: Union[float, int]) -> "Tensor":
        return self._wrap_core_tensor(self._tensor.__radd__(other))

    def __sub__(self, other: Union["Tensor", float, int]) -> "Tensor":
        return self._wrap_core_tensor(self._tensor.__sub__(other))

    def __rsub__(self, other: Union[float, int]) -> "Tensor":
        return self._wrap_core_tensor(self._tensor.__rsub__(other))

    def __mul__(self, other: Union["Tensor", float, int]) -> "Tensor":
        return self._wrap_core_tensor(self._tensor.__mul__(other))

    def __rmul__(self, other: Union[float, int]) -> "Tensor":
        return self._wrap_core_tensor(self._tensor.__rmul__(other))

    def __truediv__(self, other: Union["Tensor", float, int]) -> "Tensor":
        return self._wrap_core_tensor(self._tensor.__truediv__(other))

    def __rtruediv__(self, other: Union[float, int]) -> "Tensor":
        return self._wrap_core_tensor(self._tensor.__rtruediv__(other))

    def __pow__(self, exponent: Union["Tensor", float, int]) -> "Tensor":
        """Element-wise power operation."""
        exp_tensor = exponent._tensor if isinstance(exponent, Tensor) else exponent
        return self._wrap_core_tensor(self._tensor.pow(exp_tensor))

    def logaddexp(self, other: Union["Tensor", float, int]) -> "Tensor":
        """Stable logaddexp combining with ``other``."""
        return self._wrap_core_tensor(self._tensor.logaddexp(other))



    def pow(self, exponent: Union["Tensor", float, int]) -> "Tensor":
        """Alias for the ``**`` operator."""
        return self.__pow__(exponent)

    def __rpow__(self, base: Union["Tensor", float, int]) -> "Tensor":
        """Support right-hand exponentiation so scalars delegate to the Rust backend."""

        if isinstance(base, Tensor):
            return base.__pow__(self)

        kwargs = {"dtype": self.dtype}
        try:
            kwargs["device"] = _minitensor_core.Device(self.device)
        except Exception:
            # Fallback to the default device if construction fails (e.g., CPU-only builds).
            pass

        try:
            base_tensor = Tensor(base, **kwargs)
        except Exception as exc:  # pragma: no cover
            raise TypeError(
                f"unsupported base type for power: {type(base).__name__}"
            ) from exc

        return base_tensor.__pow__(self)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        """Matrix multiplication operator (@)."""
        return self.matmul(other)

    # Matrix operations
    def matmul(self, other: "Tensor") -> "Tensor":
        """Matrix multiplication."""
        if not isinstance(other, Tensor):
            raise TypeError("matmul requires another Tensor")
        
        return self._wrap_core_tensor(self._tensor.matmul(other._tensor))

    def mm(self, other: "Tensor") -> "Tensor":
        """Matrix multiplication."""
        return self.matmul(other)

    def triu(self, diagonal: int = 0) -> "Tensor":
        """Return the upper triangular part of the matrix."""

        diag = int(diagonal)

        return self._wrap_core_tensor(self._tensor.triu(diag))

    def tril(self, diagonal: int = 0) -> "Tensor":
        """Return the lower triangular part of the matrix."""

        diag = int(diagonal)

        return self._wrap_core_tensor(self._tensor.tril(diag))

    def dot(self, other: "Tensor") -> "Tensor":
        """Dot product of two 1D tensors computed via Rust kernels."""

        if not isinstance(other, Tensor):
            raise TypeError("dot requires another Tensor instance")

        return self._wrap_core_tensor(self._tensor.dot(other._tensor))

    def where(self, condition: Any, other: Any) -> "Tensor":
        """Select elements from ``self`` or ``other`` based on ``condition``.

        The Rust backend validates the mask dtype, performs broadcasting, and
        coerces scalar or tensor operands to the appropriate dtype and device.
        """

        return self._wrap_core_tensor(self._tensor.where(condition, other))

    def masked_fill(
        self,
        mask: Any,
        value: Union["Tensor", Real, bool],
    ) -> "Tensor":
        """Fill elements of the tensor where ``mask`` is ``True``.

        Device placement, dtype promotion, and scalar handling are delegated to
        the compiled backend to avoid Python-side computations.
        """

        return Tensor._wrap_core_tensor(self._tensor.masked_fill(mask, value))

    def cross(self, other: "Tensor", axis: int = -1) -> "Tensor":
        """Compute the 3D cross product with another tensor.

        Args:
            other: The tensor to compute the cross product with. If a plain
                Python value is provided it will be converted to a ``Tensor``
                with the same dtype as ``self``.
            axis: The axis along which to compute the cross product. Defaults
                to the last dimension.

        Returns:
            Tensor: The resulting cross product tensor.
        """
        if not isinstance(other, Tensor):
            other = Tensor(other, dtype=self.dtype)

        return Tensor._wrap_core_tensor(
            _minitensor_core.numpy_compat.cross(
                self._tensor, other._tensor, axis=axis
            )
        )

    # Reduction operations
    def prod(
        self, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False
    ) -> "Tensor":
        """Product along specified dimensions."""
        return self._wrap_core_tensor(self._tensor.prod(dim, keepdim))

    def sum(
        self, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False
    ) -> "Tensor":
        """Sum along specified dimensions."""
        return self._wrap_core_tensor(self._tensor.sum(dim, keepdim))

    def logsumexp(
        self, dim: Optional[Union[int, Sequence[int]]] = None, keepdim: bool = False
    ) -> "Tensor":
        """Compute the log of summed exponentials along ``dim``."""
        try:
            return self._wrap_core_tensor(self._tensor.logsumexp(dim, keepdim))
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

    def mean(
        self, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False
    ) -> "Tensor":
        """Mean along specified dimensions."""
        return self._wrap_core_tensor(self._tensor.mean(dim, keepdim))

    def max(
        self, dim: Optional[int] = None, keepdim: bool = False
    ) -> Union["Tensor", Tuple["Tensor", "Tensor"]]:
        """Maximum values along dimension."""
        if dim is None:
            return self._wrap_core_tensor(self._tensor.max(dim, keepdim))

        values = self._wrap_core_tensor(self._tensor.max(dim, keepdim))
        indices = self._wrap_core_tensor(self._tensor.argmax(dim, keepdim))

        return values, indices

    def min(
        self, dim: Optional[int] = None, keepdim: bool = False
    ) -> Union["Tensor", Tuple["Tensor", "Tensor"]]:
        """Minimum values along dimension."""
        if dim is None:
            return self._wrap_core_tensor(self._tensor.min(dim, keepdim))

        values = self._wrap_core_tensor(self._tensor.min(dim, keepdim))
        indices = self._wrap_core_tensor(self._tensor.argmin(dim, keepdim))

        return values, indices

    def median(
        self, dim: Optional[int] = None, keepdim: bool = False
    ) -> Union["Tensor", Tuple["Tensor", "Tensor"]]:
        """Median values along dimension, with indices when ``dim`` is provided."""
        values_backend, indices_backend = self._tensor.median(dim, keepdim)

        values = Tensor._wrap_core_tensor(values_backend)

        if dim is None:
            return values

        if indices_backend is None:
            raise RuntimeError("median returned no indices for the requested dimension")

        indices = Tensor._wrap_core_tensor(indices_backend)

        return values, indices

    def argmax(self, dim: Optional[int] = None, keepdim: bool = False) -> "Tensor":
        """Indices of maximum values."""
        return self._wrap_core_tensor(self._tensor.argmax(dim, keepdim))

    def argmin(self, dim: Optional[int] = None, keepdim: bool = False) -> "Tensor":
        """Indices of minimum values."""
        if dim is not None:
            dim = dim + self.ndim if dim < 0 else dim
            if dim < 0 or dim >= self.ndim:
                raise IndexError("Dimension out of range")
        return self._wrap_core_tensor(self._tensor.argmin(dim, keepdim))

    def topk(
        self,
        k: int,
        dim: Optional[int] = None,
        largest: bool = True,
        sorted: bool = True,
    ) -> Tuple["Tensor", "Tensor"]:
        """Return the top-``k`` elements and their indices along ``dim``."""
        if not isinstance(k, int):
            raise TypeError("k must be an integer")
        if k < 0:
            raise RuntimeError("k must be non-negative")

        values_backend, indices_backend = self._tensor.topk(k, dim, largest, sorted)

        return (
            Tensor._wrap_core_tensor(values_backend),
            Tensor._wrap_core_tensor(indices_backend),
        )

    def sort(
        self,
        dim: Optional[int] = -1,
        descending: bool = False,
        stable: bool = False,
    ) -> Tuple["Tensor", "Tensor"]:
        """Sort ``self`` along ``dim`` returning values and indices."""
        backend_dim = dim if dim is not None else None
        values_backend, indices_backend = self._tensor.sort(
            backend_dim, descending, stable
        )

        return (
            Tensor._wrap_core_tensor(values_backend),
            Tensor._wrap_core_tensor(indices_backend),
        )

    def argsort(
        self,
        dim: Optional[int] = -1,
        descending: bool = False,
        stable: bool = False,
    ) -> "Tensor":
        """Return the indices that would sort ``self`` along ``dim``."""
        backend_dim = dim if dim is not None else None
        return self._wrap_core_tensor(self._tensor.argsort(backend_dim, descending, stable))

    def std(
        self, dim: Optional[int] = None, keepdim: bool = False, unbiased: bool = True
    ) -> "Tensor":
        """Standard deviation along dimension."""
        return self._wrap_core_tensor(self._tensor.std(dim, keepdim, unbiased))

    def var(
        self, dim: Optional[int] = None, keepdim: bool = False, unbiased: bool = True
    ) -> "Tensor":
        """Variance along dimension."""
        return self._wrap_core_tensor(self._tensor.var(dim, keepdim, unbiased))

    # Mathematical functions
    def abs(self) -> "Tensor":
        """Absolute value."""
        return self._wrap_core_tensor(self._tensor.abs())

    def sqrt(self) -> "Tensor":
        """Square root."""
        return self._wrap_core_tensor(self._tensor.sqrt())

    def rsqrt(self) -> "Tensor":
        """Reciprocal square root computed in Rust."""
        return self._wrap_core_tensor(self._tensor.rsqrt())

    def exp(self) -> "Tensor":
        """Exponential function."""
        return self._wrap_core_tensor(self._tensor.exp())

    def log(self) -> "Tensor":
        """Natural logarithm."""
        return self._wrap_core_tensor(self._tensor.log())

    def log1p(self) -> "Tensor":
        """Compute ``log(1 + x)`` element-wise."""
        return self._wrap_core_tensor(self._tensor.log1p())

    def expm1(self) -> "Tensor":
        """Compute ``exp(x) - 1`` element-wise."""
        return self._wrap_core_tensor(self._tensor.expm1())

    def sin(self) -> "Tensor":
        """Element-wise sine computed in Rust."""
        return self._wrap_core_tensor(self._tensor.sin())

    def cos(self) -> "Tensor":
        """Element-wise cosine computed in Rust."""
        return self._wrap_core_tensor(self._tensor.cos())

    def tan(self) -> "Tensor":
        """Element-wise tangent computed in Rust."""
        return self._wrap_core_tensor(self._tensor.tan())

    # Activation functions
    def relu(self) -> "Tensor":
        """ReLU activation function."""
        return self._wrap_core_tensor(self._tensor.relu())

    def hardshrink(self, lambd: float = 0.5) -> "Tensor":
        """Hardshrink activation that zeros values within ``[-lambd, lambd]``."""

        return self._wrap_core_tensor(self._tensor.hardshrink(lambd))

    def sigmoid(self) -> "Tensor":
        """Sigmoid activation function."""
        return self._wrap_core_tensor(self._tensor.sigmoid())

    def softplus(self, beta: float = 1.0, threshold: float = 20.0) -> "Tensor":
        """Softplus activation computed in Rust."""
        return self._wrap_core_tensor(self._tensor.softplus(beta, threshold))

    def gelu(self, approximate: str = "none") -> "Tensor":
        """Gaussian Error Linear Unit activation."""

        return self._wrap_core_tensor(self._tensor.gelu(approximate))

    def elu(self, alpha: float = 1.0) -> "Tensor":
        """Exponential Linear Unit activation."""

        return self._wrap_core_tensor(self._tensor.elu(alpha))

    def selu(self) -> "Tensor":
        """Scaled Exponential Linear Unit activation."""

        return self._wrap_core_tensor(self._tensor.selu())

    def silu(self) -> "Tensor":
        """Sigmoid Linear Unit (Swish) activation."""

        return self._wrap_core_tensor(self._tensor.silu())

    def softsign(self) -> "Tensor":
        """Softsign activation computed in Rust."""

        return self._wrap_core_tensor(self._tensor.softsign())

    def tanh(self) -> "Tensor":
        """Hyperbolic tangent activation function."""
        return self._wrap_core_tensor(self._tensor.tanh())

    def softmax(self, dim: int = -1) -> "Tensor":
        """Softmax activation function."""
        return self._wrap_core_tensor(self._tensor.softmax(dim))

    def log_softmax(self, dim: int = -1) -> "Tensor":
        """Log-softmax activation function."""
        return self._wrap_core_tensor(self._tensor.log_softmax(dim))

    def layer_norm(
        self,
        normalized_shape: Union[int, Sequence[int]],
        weight: Optional["Tensor"] = None,
        bias: Optional["Tensor"] = None,
        eps: float = 1e-5,
    ) -> "Tensor":
        """Layer normalization computed entirely within the Rust backend."""

        if isinstance(normalized_shape, Integral):
            shape_tuple = (int(normalized_shape),)
        else:
            shape_tuple = tuple(int(dim) for dim in normalized_shape)
        if not shape_tuple:
            raise ValueError("normalized_shape must contain at least one dimension")

        weight_tensor = None if weight is None else weight._tensor
        bias_tensor = None if bias is None else bias._tensor
        return Tensor._wrap_core_tensor(
            self._tensor.layer_norm(
                list(shape_tuple), weight_tensor, bias_tensor, eps
            )
        )

    # Comparison operations
    def eq(self, other: Any) -> "Tensor":
        """Element-wise equality comparison."""
        return self._wrap_core_tensor(self._tensor.__eq__(other))

    def ne(self, other: Any) -> "Tensor":
        """Element-wise not-equal comparison."""
        return self._wrap_core_tensor(self._tensor.__ne__(other))

    def lt(self, other: Any) -> "Tensor":
        """Element-wise less-than comparison."""
        return self._wrap_core_tensor(self._tensor.__lt__(other))

    def le(self, other: Any) -> "Tensor":
        """Element-wise less-than-or-equal comparison."""
        return self._wrap_core_tensor(self._tensor.__le__(other))

    def gt(self, other: Any) -> "Tensor":
        """Element-wise greater-than comparison."""
        return self._wrap_core_tensor(self._tensor.__gt__(other))

    def ge(self, other: Any) -> "Tensor":
        """Element-wise greater-than-or-equal comparison."""
        return self._wrap_core_tensor(self._tensor.__ge__(other))

    def maximum(self, other: Union["Tensor", float, int]) -> "Tensor":
        return self._wrap_core_tensor(self._tensor.maximum(other))

    def minimum(self, other: Union["Tensor", float, int]) -> "Tensor":
        return self._wrap_core_tensor(self._tensor.minimum(other))

    # Python special methods for comparisons
    def __eq__(self, other: object) -> "Tensor":
        return self._wrap_core_tensor(self._tensor.__eq__(other))

    def __ne__(self, other: object) -> "Tensor":
        return self._wrap_core_tensor(self._tensor.__ne__(other))

    def __lt__(self, other: object) -> "Tensor":
        return self._wrap_core_tensor(self._tensor.__lt__(other))

    def __le__(self, other: object) -> "Tensor":
        return self._wrap_core_tensor(self._tensor.__le__(other))

    def __gt__(self, other: object) -> "Tensor":
        return self._wrap_core_tensor(self._tensor.__gt__(other))

    def __ge__(self, other: object) -> "Tensor":
        return self._wrap_core_tensor(self._tensor.__ge__(other))

    # Utility methods
    def all(self, dim: Optional[int] = None, keepdim: bool = False) -> "Tensor":
        """Test if all elements evaluate to True."""
        return self._wrap_core_tensor(self._tensor.all(dim, keepdim))

    def any(self, dim: Optional[int] = None, keepdim: bool = False) -> "Tensor":
        """Test if any element evaluates to True."""
        return self._wrap_core_tensor(self._tensor.any(dim, keepdim))

    def cumsum(self, dim: int) -> "Tensor":
        """Cumulative sum along a dimension."""
        return self._wrap_core_tensor(self._tensor.cumsum(dim))

    def cumprod(self, dim: int) -> "Tensor":
        """Cumulative product along a dimension."""
        return self._wrap_core_tensor(self._tensor.cumprod(dim))

    def clamp(
        self, min_val: Optional[float] = None, max_val: Optional[float] = None
    ) -> "Tensor":
        """Clamp tensor values to range."""
        return self._wrap_core_tensor(self._tensor.clamp(min_val, max_val))

    def clip(
        self, min_val: Optional[float] = None, max_val: Optional[float] = None
    ) -> "Tensor":
        """Clip tensor values to range (NumPy compatibility)."""
        return self.clamp(min_val, max_val)

    # Array testing
    def isnan(self) -> "Tensor":
        """Test for NaN values."""
        return self._wrap_core_tensor(self._tensor.isnan())

    def isinf(self) -> "Tensor":
        """Test for infinite values."""
        return self._wrap_core_tensor(self._tensor.isinf())

    def isfinite(self) -> "Tensor":
        """Test for finite values."""
        return self._wrap_core_tensor(self._tensor.isfinite())

    # Comparison with other tensors
    def allclose(self, other: "Tensor", rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Check if tensors are approximately equal."""
        return self._tensor.allclose(other._tensor, rtol, atol)

    def array_equal(self, other: "Tensor") -> bool:
        """Check if tensors are exactly equal."""
        return self._tensor.array_equal(other._tensor)

    # String representations
    def __repr__(self) -> str:
        return self._tensor.__repr__()

    def __str__(self) -> str:
        return self._tensor.__str__()

    def __len__(self) -> int:
        return self._tensor.__len__()

    def __bool__(self) -> bool:
        return self._tensor.__bool__()

    # Indexing and slicing
    def __getitem__(self, key):
        """Tensor indexing and slicing."""
        return self._wrap_core_tensor(self._tensor.__getitem__(key))

    def __setitem__(self, key, value):
        """Tensor item assignment."""
        if isinstance(value, Tensor):
            self._tensor.__setitem__(key, value._tensor)
        else:
            self._tensor.__setitem__(key, value)

    # Static tensor creation methods
    @staticmethod
    def zeros(
        *shape: Union[int, Sequence[int]],
        dtype: Optional[str] = None,
        device=None,
        requires_grad: bool = False,
    ) -> "Tensor":
        """Create a tensor filled with zeros."""
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        return Tensor._wrap_core_tensor(
            _minitensor_core.Tensor.zeros(list(shape), dtype, device, requires_grad)
        )

    @staticmethod
    def ones(
        *shape: Union[int, Sequence[int]],
        dtype: Optional[str] = None,
        device=None,
        requires_grad: bool = False,
    ) -> "Tensor":
        """Create a tensor filled with ones."""
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        return Tensor._wrap_core_tensor(
            _minitensor_core.Tensor.ones(list(shape), dtype, device, requires_grad)
        )

    @staticmethod
    def full(
        shape: Sequence[int],
        fill_value: float,
        dtype: Optional[str] = None,
        device=None,
        requires_grad: bool = False,
    ) -> "Tensor":
        """Create a tensor filled with a specific value."""
        return Tensor._wrap_core_tensor(
            _minitensor_core.Tensor.full(
                list(shape), fill_value, dtype, device, requires_grad
            )
        )

    @staticmethod
    def rand(
        *shape: Union[int, Sequence[int]],
        dtype: Optional[str] = None,
        device=None,
        requires_grad: bool = False,
    ) -> "Tensor":
        """Create a tensor with random values from uniform distribution [0, 1)."""
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        return Tensor._wrap_core_tensor(
            _minitensor_core.Tensor.rand(list(shape), dtype, device, requires_grad)
        )

    @staticmethod
    def randn(
        *shape: Union[int, Sequence[int]],
        dtype: Optional[str] = None,
        device=None,
        requires_grad: bool = False,
    ) -> "Tensor":
        """Create a tensor with random values from standard normal distribution."""
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        return Tensor._wrap_core_tensor(
            _minitensor_core.Tensor.randn(list(shape), dtype, device, requires_grad)
        )

    @staticmethod
    def eye(
        n: int,
        m: Optional[int] = None,
        dtype: Optional[str] = None,
        device=None,
        requires_grad: bool = False,
    ) -> "Tensor":
        """Create an identity matrix."""
        return Tensor._wrap_core_tensor(_minitensor_core.Tensor.eye(n, m, dtype, device, requires_grad))

    @staticmethod
    def arange(
        start: float,
        end: Optional[float] = None,
        step: float = 1.0,
        dtype: Optional[str] = None,
        device=None,
        requires_grad: bool = False,
    ) -> "Tensor":
        """Create a tensor with evenly spaced values."""
        if end is None:
            end = start
            start = 0.0
        return Tensor._wrap_core_tensor(
            _minitensor_core.Tensor.arange(
                start, end, step, dtype, device, requires_grad
            )
        )

    @staticmethod
    def linspace(
        start: float,
        end: float,
        steps: int,
        dtype: Optional[str] = None,
        device=None,
        requires_grad: bool = False,
    ) -> "Tensor":
        """Create a tensor with linearly spaced values."""
        if steps <= 0:
            raise ValueError("Number of steps must be positive")

        return Tensor._wrap_core_tensor(
            _minitensor_core.Tensor.linspace(
                start, end, steps, dtype, device, requires_grad
            )
        )

    @staticmethod
    def logspace(
        start: float,
        end: float,
        steps: int,
        base: float = 10.0,
        dtype: Optional[str] = None,
        device=None,
        requires_grad: bool = False,
    ) -> "Tensor":
        """Create a tensor with logarithmically spaced values."""
        if steps <= 0:
            raise ValueError("Number of steps must be positive")

        return Tensor._wrap_core_tensor(
            _minitensor_core.Tensor.logspace(
                start, end, steps, base, dtype, device, requires_grad
            )
        )

    @staticmethod
    def from_numpy(array: "np.ndarray", requires_grad: bool = False) -> "Tensor":
        """Create a tensor from a NumPy array."""
        _ensure_numpy_available(
            "NumPy is required to construct tensors from NumPy arrays."
        )
        return Tensor._wrap_core_tensor(_minitensor_core.Tensor.from_numpy(array, requires_grad))

    @staticmethod
    def from_numpy_shared(array: "np.ndarray", requires_grad: bool = False) -> "Tensor":
        """Create a tensor from a NumPy array with zero-copy when possible."""
        _ensure_numpy_available(
            "NumPy is required to construct tensors from NumPy arrays."
        )
        return Tensor._wrap_core_tensor(_minitensor_core.Tensor.from_numpy_shared(array, requires_grad))


# Convenience functions for tensor creation (NumPy-style)
def tensor(
    data: Any, dtype: Optional[str] = None, device=None, requires_grad: bool = False
) -> Tensor:
    """Create a tensor from data."""
    return Tensor(data, requires_grad=requires_grad, dtype=dtype, device=device)


def zeros(
    *shape: Union[int, Sequence[int]],
    dtype: Optional[str] = None,
    device=None,
    requires_grad: bool = False,
) -> Tensor:
    """Create a tensor filled with zeros."""
    return Tensor.zeros(*shape, dtype=dtype, device=device, requires_grad=requires_grad)


def ones(
    *shape: Union[int, Sequence[int]],
    dtype: Optional[str] = None,
    device=None,
    requires_grad: bool = False,
) -> Tensor:
    """Create a tensor filled with ones."""
    return Tensor.ones(*shape, dtype=dtype, device=device, requires_grad=requires_grad)


def full(
    shape: Sequence[int],
    fill_value: float,
    dtype: Optional[str] = None,
    device=None,
    requires_grad: bool = False,
) -> Tensor:
    """Create a tensor filled with a specific value."""
    return Tensor.full(
        shape, fill_value, dtype=dtype, device=device, requires_grad=requires_grad
    )


def rand(
    *shape: Union[int, Sequence[int]],
    dtype: Optional[str] = None,
    device=None,
    requires_grad: bool = False,
) -> Tensor:
    """Create a tensor with random values from uniform distribution."""
    return Tensor.rand(*shape, dtype=dtype, device=device, requires_grad=requires_grad)


def randn(
    *shape: Union[int, Sequence[int]],
    dtype: Optional[str] = None,
    device=None,
    requires_grad: bool = False,
) -> Tensor:
    """Create a tensor with random values from normal distribution."""
    return Tensor.randn(*shape, dtype=dtype, device=device, requires_grad=requires_grad)


def eye(
    n: int,
    m: Optional[int] = None,
    dtype: Optional[str] = None,
    device=None,
    requires_grad: bool = False,
) -> Tensor:
    """Create an identity matrix."""
    return Tensor.eye(n, m, dtype=dtype, device=device, requires_grad=requires_grad)


def arange(
    start: float,
    end: Optional[float] = None,
    step: float = 1.0,
    dtype: Optional[str] = None,
    device=None,
    requires_grad: bool = False,
) -> Tensor:
    """Create a tensor with evenly spaced values."""
    return Tensor.arange(
        start, end, step, dtype=dtype, device=device, requires_grad=requires_grad
    )


def linspace(
    start: float,
    end: float,
    steps: int,
    dtype: Optional[str] = None,
    device=None,
    requires_grad: bool = False,
) -> Tensor:
    """Create a tensor with linearly spaced values."""
    return Tensor.linspace(
        start, end, steps, dtype=dtype, device=device, requires_grad=requires_grad
    )


def from_numpy(array: "np.ndarray", requires_grad: bool = False) -> Tensor:
    """Create a tensor from a NumPy array."""
    _ensure_numpy_available(
        "NumPy is required to construct tensors from NumPy arrays."
    )
    return Tensor.from_numpy(array, requires_grad=requires_grad)


# Export all public symbols
__all__ = [
    "Tensor",
    "tensor",
    "zeros",
    "ones",
    "full",
    "rand",
    "randn",
    "eye",
    "arange",
    "linspace",
    "from_numpy",
    "set_default_dtype",
    "get_default_dtype",
]
