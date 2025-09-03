# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Enhanced Tensor class with comprehensive NumPy compatibility and automatic differentiation support.
"""

try:
    from . import _core as _minitensor_core
except ImportError:
    # Fallback for development - try direct import
    try:
        import minitensor._core as _minitensor_core
    except ImportError:
        # Final fallback for development
        import minitensor as _minitensor_core

from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np


class Tensor:
    """
    A multi-dimensional array with automatic differentiation support and NumPy compatibility.

    This tensor class provides a PyTorch-like interface with comprehensive NumPy compatibility,
    making it easy to migrate from NumPy-based code while gaining automatic differentiation.
    """

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
            self._bool_data = getattr(data, "_bool_data", None)
        else:
            # Create new tensor from data
            self._tensor = _minitensor_core.Tensor(data, dtype, device, requires_grad)
            self._bool_data = np.array(data, dtype=bool) if dtype == "bool" else None

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
        """Get gradient tensor."""
        rust_grad = self._tensor.grad
        if rust_grad is not None:
            result = Tensor.__new__(Tensor)
            result._tensor = rust_grad
            return result
        return None

    # NumPy compatibility properties
    @property
    def size(self) -> int:
        """Total number of elements (NumPy compatibility)."""
        return self._tensor.size

    @property
    def itemsize(self) -> int:
        """Size of each element in bytes (NumPy compatibility)."""
        return self._tensor.itemsize

    @property
    def nbytes(self) -> int:
        """Total bytes consumed by the tensor (NumPy compatibility)."""
        return self._tensor.nbytes

    @property
    def strides(self) -> Tuple[int, ...]:
        """Strides of the tensor (NumPy compatibility)."""
        return tuple(self._tensor.strides)

    @property
    def ndim(self) -> int:
        """Number of dimensions (NumPy compatibility)."""
        return self._tensor.ndim()

    @property
    def T(self) -> "Tensor":
        """Transpose (NumPy compatibility)."""
        return self.transpose()

    # Basic tensor info methods
    def numel(self) -> int:
        """Get total number of elements."""
        return self._tensor.numel()

    def dim(self) -> int:
        """Get number of dimensions (PyTorch compatibility)."""
        return self.ndim

    def is_contiguous(self) -> bool:
        """Check if tensor is contiguous in memory."""
        return self._tensor.is_contiguous()

    def element_size(self) -> int:
        """Get size of each element in bytes."""
        return self.itemsize

    # Data conversion methods
    def numpy(self) -> np.ndarray:
        """Convert to numpy array with zero-copy when possible."""
        bool_data = getattr(self, "_bool_data", None)
        if bool_data is not None:
            return bool_data.copy()
        try:
            return self._tensor.numpy()
        except NotImplementedError:
            try:
                return self._tensor.numpy_copy()
            except NotImplementedError:
                dtype_map = {
                    "int64": np.int64,
                    "int32": np.int32,
                }
                np_dtype = dtype_map.get(self.dtype, np.float32)
                float_array = self._tensor.to("float32").numpy_copy()
                return float_array.astype(np_dtype)

    def numpy_copy(self) -> np.ndarray:
        """Convert to numpy array with explicit copy."""
        return self._tensor.numpy_copy()

    def tolist(self) -> List:
        """Convert to Python list."""
        return self._tensor.tolist()

    def item(self) -> Union[float, int, bool]:
        """Get scalar value from single-element tensor."""
        if self.numel() != 1:
            raise ValueError("item() can only be called on tensors with one element")
        return self.tolist()

    # Tensor manipulation methods
    def reshape(self, *shape: Union[int, Sequence[int]]) -> "Tensor":
        """Reshape tensor to new shape."""
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        result = Tensor.__new__(Tensor)
        result._tensor = self._tensor.reshape(list(shape))
        return result

    def view(self, *shape: Union[int, Sequence[int]]) -> "Tensor":
        """Alias for reshape (PyTorch compatibility)."""
        return self.reshape(*shape)

    def transpose(self, dim0: int = 0, dim1: int = 1) -> "Tensor":
        """Transpose tensor dimensions."""
        result = Tensor.__new__(Tensor)
        result._tensor = self._tensor.transpose(dim0, dim1)
        return result

    def permute(self, *dims: int) -> "Tensor":
        """Permute tensor dimensions (PyTorch compatibility)."""
        # Accept dimensions as positional arguments or a single iterable
        if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
            dims = tuple(dims[0])

        # Support arbitrary dimension permutations using successive transposes
        if len(dims) != self.ndim:
            raise ValueError("dims must match number of dimensions")

        # Normalise negative dimensions and validate permutation
        ndim = self.ndim
        dims = [d + ndim if d < 0 else d for d in dims]
        if sorted(dims) != list(range(ndim)):
            raise ValueError("dims must be a permutation of dimensions")

        # Apply a sequence of transposes to achieve the desired order
        result = self
        current = list(range(ndim))
        for i, d in enumerate(dims):
            j = current.index(d)
            if i != j:
                result = result.transpose(i, j)
                current[i], current[j] = current[j], current[i]
        return result

    def squeeze(self, dim: Optional[int] = None) -> "Tensor":
        """Remove dimensions of size 1."""
        result = Tensor.__new__(Tensor)
        result._tensor = self._tensor.squeeze(dim)
        return result

    def unsqueeze(self, dim: int) -> "Tensor":
        """Add a dimension of size 1."""
        result = Tensor.__new__(Tensor)
        result._tensor = self._tensor.unsqueeze(dim)
        return result

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> "Tensor":
        """Flatten tensor dimensions."""
        result = Tensor.__new__(Tensor)
        result._tensor = self._tensor.flatten(start_dim, end_dim)
        return result

    def ravel(self) -> "Tensor":
        """Return flattened tensor (NumPy compatibility)."""
        return self.flatten()

    # Tensor operations
    def clone(self) -> "Tensor":
        """Create a copy of the tensor."""
        result = Tensor.__new__(Tensor)
        result._tensor = self._tensor.clone()
        return result

    def copy(self) -> "Tensor":
        """Create a copy of the tensor (NumPy compatibility)."""
        return self.clone()

    def detach(self) -> "Tensor":
        """Detach tensor from computation graph."""
        result = Tensor.__new__(Tensor)
        result._tensor = self._tensor.detach()
        return result

    def contiguous(self) -> "Tensor":
        """Create a contiguous copy of the tensor."""
        result = Tensor.__new__(Tensor)
        result._tensor = self._tensor.contiguous()
        return result

    def to(self, device_or_dtype: Union[str, "Tensor"]) -> "Tensor":
        """Move tensor to device or convert dtype."""
        if isinstance(device_or_dtype, str):
            if device_or_dtype in ["cpu", "cuda", "metal"]:
                # Device conversion
                result = Tensor.__new__(Tensor)
                result._tensor = self._tensor.to(device_or_dtype)
                return result
            else:
                # Dtype conversion - would need engine support
                raise NotImplementedError("Dtype conversion not yet implemented")
        else:
            raise TypeError("to() expects device string or dtype")

    def cpu(self) -> "Tensor":
        """Move tensor to CPU."""
        result = Tensor.__new__(Tensor)
        result._tensor = self._tensor.cpu()
        return result

    def cuda(self, device: Optional[int] = None) -> "Tensor":
        """Move tensor to CUDA device."""
        # Simplified - would need proper CUDA device handling
        return self.to("cuda")

    # Gradient operations
    def backward(
        self,
        gradient: Optional["Tensor"] = None,
        retain_graph: bool = False,
        create_graph: bool = False,
    ):
        """Compute gradients via backpropagation."""
        if gradient is not None:
            self._tensor.backward(gradient._tensor)
        else:
            self._tensor.backward()

    def requires_grad_(self, requires_grad: bool = True) -> "Tensor":
        """Set requires_grad flag in-place."""
        self._tensor.requires_grad_(requires_grad)
        return self

    def zero_grad(self):
        """Zero the gradient."""
        self._tensor.zero_grad()

    # Arithmetic operations with broadcasting support
    def __add__(self, other: Union["Tensor", float, int]) -> "Tensor":
        if isinstance(other, Tensor):
            result = Tensor.__new__(Tensor)
            result._tensor = self._tensor.__add__(other._tensor)
            return result
        else:
            # Scalar addition
            scalar_tensor = Tensor(other)
            return self.__add__(scalar_tensor)

    def __radd__(self, other: Union[float, int]) -> "Tensor":
        return self.__add__(other)

    def __sub__(self, other: Union["Tensor", float, int]) -> "Tensor":
        if isinstance(other, Tensor):
            result = Tensor.__new__(Tensor)
            result._tensor = self._tensor.__sub__(other._tensor)
            return result
        else:
            scalar_tensor = Tensor(other)
            return self.__sub__(scalar_tensor)

    def __rsub__(self, other: Union[float, int]) -> "Tensor":
        scalar_tensor = Tensor(other)
        return scalar_tensor.__sub__(self)

    def __mul__(self, other: Union["Tensor", float, int]) -> "Tensor":
        if isinstance(other, Tensor):
            result = Tensor.__new__(Tensor)
            result._tensor = self._tensor.__mul__(other._tensor)
            return result
        else:
            scalar_tensor = Tensor(other)
            return self.__mul__(scalar_tensor)

    def __rmul__(self, other: Union[float, int]) -> "Tensor":
        return self.__mul__(other)

    def __truediv__(self, other: Union["Tensor", float, int]) -> "Tensor":
        if isinstance(other, Tensor):
            result = Tensor.__new__(Tensor)
            result._tensor = self._tensor.__truediv__(other._tensor)
            return result
        else:
            scalar_tensor = Tensor(other)
            return self.__truediv__(scalar_tensor)

    def __rtruediv__(self, other: Union[float, int]) -> "Tensor":
        scalar_tensor = Tensor(other)
        return scalar_tensor.__truediv__(self)

    def __pow__(self, exponent: Union["Tensor", float, int]) -> "Tensor":
        """Element-wise power operation."""
        if isinstance(exponent, Tensor):
            exponent_tensor = exponent
        else:
            exponent_tensor = Tensor(exponent)
        # Use exp(exponent * log(self)) for broad dtype support
        return (exponent_tensor * self.log()).exp()

    def pow(self, exponent: Union["Tensor", float, int]) -> "Tensor":
        """Alias for the ``**`` operator."""
        return self.__pow__(exponent)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        """Matrix multiplication operator (@)."""
        return self.matmul(other)

    # Matrix operations
    def matmul(self, other: "Tensor") -> "Tensor":
        """Matrix multiplication."""
        if not isinstance(other, Tensor):
            raise TypeError("matmul requires another Tensor")
        result = Tensor.__new__(Tensor)
        result._tensor = self._tensor.matmul(other._tensor)
        return result

    def mm(self, other: "Tensor") -> "Tensor":
        """Matrix multiplication (PyTorch compatibility)."""
        return self.matmul(other)

    def dot(self, other: "Tensor") -> "Tensor":
        """Dot product (NumPy compatibility)."""
        if self.ndim == 1 and other.ndim == 1:
            return (self * other).sum()
        else:
            return self.matmul(other)

    # Reduction operations
    def sum(
        self, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False
    ) -> "Tensor":
        """Sum along specified dimensions."""
        if isinstance(dim, int):
            dim = [dim]
        elif isinstance(dim, tuple):
            dim = list(dim)
        result = Tensor.__new__(Tensor)
        result._tensor = self._tensor.sum(dim, keepdim)
        return result

    def mean(
        self, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False
    ) -> "Tensor":
        """Mean along specified dimensions."""
        if isinstance(dim, int):
            dim = [dim]
        elif isinstance(dim, tuple):
            dim = list(dim)
        result = Tensor.__new__(Tensor)
        result._tensor = self._tensor.mean(dim, keepdim)
        return result

    def max(
        self, dim: Optional[int] = None, keepdim: bool = False
    ) -> Union["Tensor", Tuple["Tensor", "Tensor"]]:
        """Maximum values along dimension."""
        result = Tensor.__new__(Tensor)
        result._tensor = self._tensor.max(dim, keepdim)
        if dim is None:
            return result
        else:
            indices = Tensor.__new__(Tensor)
            indices._tensor = self._tensor.argmax(dim, keepdim)
            return result, indices

    def min(
        self, dim: Optional[int] = None, keepdim: bool = False
    ) -> Union["Tensor", Tuple["Tensor", "Tensor"]]:
        """Minimum values along dimension."""
        result = Tensor.__new__(Tensor)
        result._tensor = self._tensor.min(dim, keepdim)
        if dim is None:
            return result
        else:
            indices = Tensor.__new__(Tensor)
            indices._tensor = self._tensor.argmin(dim, keepdim)
            return result, indices

    def argmax(self, dim: Optional[int] = None, keepdim: bool = False) -> "Tensor":
        """Indices of maximum values."""
        result = Tensor.__new__(Tensor)
        result._tensor = self._tensor.argmax(dim, keepdim)
        return result

    def argmin(self, dim: Optional[int] = None, keepdim: bool = False) -> "Tensor":
        """Indices of minimum values."""
        result = Tensor.__new__(Tensor)
        result._tensor = self._tensor.argmin(dim, keepdim)
        return result

    def std(
        self, dim: Optional[int] = None, keepdim: bool = False, unbiased: bool = True
    ) -> "Tensor":
        """Standard deviation along dimension."""
        result = Tensor.__new__(Tensor)
        result._tensor = self._tensor.std(dim, keepdim)
        return result

    def var(
        self, dim: Optional[int] = None, keepdim: bool = False, unbiased: bool = True
    ) -> "Tensor":
        """Variance along dimension."""
        result = Tensor.__new__(Tensor)
        result._tensor = self._tensor.var(dim, keepdim)
        return result

    # Mathematical functions
    def abs(self) -> "Tensor":
        """Absolute value."""
        result = Tensor.__new__(Tensor)
        result._tensor = self._tensor.abs()
        return result

    def sqrt(self) -> "Tensor":
        """Square root."""
        result = Tensor.__new__(Tensor)
        result._tensor = self._tensor.sqrt()
        return result

    def exp(self) -> "Tensor":
        """Exponential function."""
        result = Tensor.__new__(Tensor)
        result._tensor = self._tensor.exp()
        return result

    def log(self) -> "Tensor":
        """Natural logarithm."""
        result = Tensor.__new__(Tensor)
        result._tensor = self._tensor.log()
        return result

    def sin(self) -> "Tensor":
        """Element-wise sine computed in Rust."""
        result = Tensor.__new__(Tensor)
        result._tensor = self._tensor.sin()
        return result

    def cos(self) -> "Tensor":
        """Element-wise cosine computed in Rust."""
        result = Tensor.__new__(Tensor)
        result._tensor = self._tensor.cos()
        return result

    def tan(self) -> "Tensor":
        """Element-wise tangent computed in Rust."""
        result = Tensor.__new__(Tensor)
        result._tensor = self._tensor.tan()
        return result

    # Activation functions
    def relu(self) -> "Tensor":
        """ReLU activation function."""
        result = Tensor.__new__(Tensor)
        result._tensor = self._tensor.relu()
        return result

    def sigmoid(self) -> "Tensor":
        """Sigmoid activation function."""
        result = Tensor.__new__(Tensor)
        result._tensor = self._tensor.sigmoid()
        return result

    def tanh(self) -> "Tensor":
        """Hyperbolic tangent activation function."""
        result = Tensor.__new__(Tensor)
        result._tensor = self._tensor.tanh()
        return result

    def softmax(self, dim: int = -1) -> "Tensor":
        """Softmax activation function."""
        # Numerically stable softmax implementation
        x_max, _ = self.max(dim=dim, keepdim=True)
        x_shifted = self - x_max
        exp_x = x_shifted.exp()
        return exp_x / exp_x.sum(dim=dim, keepdim=True)

    def log_softmax(self, dim: int = -1) -> "Tensor":
        """Log-softmax activation function."""
        return self.softmax(dim).log()

    # Comparison operations
    def eq(self, other: "Tensor") -> "Tensor":
        """Element-wise equality comparison."""
        result = Tensor.__new__(Tensor)
        result._tensor = self._tensor.eq(other._tensor)
        return result

    def ne(self, other: "Tensor") -> "Tensor":
        """Element-wise not-equal comparison."""
        result = Tensor.__new__(Tensor)
        result._tensor = self._tensor.ne(other._tensor)
        return result

    def lt(self, other: "Tensor") -> "Tensor":
        """Element-wise less-than comparison."""
        result = Tensor.__new__(Tensor)
        result._tensor = self._tensor.lt(other._tensor)
        return result

    def le(self, other: "Tensor") -> "Tensor":
        """Element-wise less-than-or-equal comparison."""
        result = Tensor.__new__(Tensor)
        result._tensor = self._tensor.le(other._tensor)
        return result

    def gt(self, other: "Tensor") -> "Tensor":
        """Element-wise greater-than comparison."""
        result = Tensor.__new__(Tensor)
        result._tensor = self._tensor.gt(other._tensor)
        return result

    def ge(self, other: "Tensor") -> "Tensor":
        """Element-wise greater-than-or-equal comparison."""
        result = Tensor.__new__(Tensor)
        result._tensor = self._tensor.ge(other._tensor)
        return result

    # Python special methods for comparisons
    def __eq__(self, other: object) -> "Tensor":
        if not isinstance(other, Tensor):
            return NotImplemented
        return self.eq(other)

    def __ne__(self, other: object) -> "Tensor":
        if not isinstance(other, Tensor):
            return NotImplemented
        return self.ne(other)

    def __lt__(self, other: "Tensor") -> "Tensor":
        return self.lt(other)

    def __le__(self, other: "Tensor") -> "Tensor":
        return self.le(other)

    def __gt__(self, other: "Tensor") -> "Tensor":
        return self.gt(other)

    def __ge__(self, other: "Tensor") -> "Tensor":
        return self.ge(other)

    # Utility methods
    def all(self, dim: Optional[int] = None, keepdim: bool = False) -> "Tensor":
        """Test if all elements evaluate to True."""
        result = Tensor.__new__(Tensor)
        result._tensor = self._tensor.all(dim, keepdim)
        return result

    def any(self, dim: Optional[int] = None, keepdim: bool = False) -> "Tensor":
        """Test if any element evaluates to True."""
        result = Tensor.__new__(Tensor)
        result._tensor = self._tensor.any(dim, keepdim)
        return result

    def clamp(
        self, min_val: Optional[float] = None, max_val: Optional[float] = None
    ) -> "Tensor":
        """Clamp tensor values to range."""
        result = Tensor.__new__(Tensor)
        result._tensor = self._tensor.clamp(min_val, max_val)
        return result

    def clip(
        self, min_val: Optional[float] = None, max_val: Optional[float] = None
    ) -> "Tensor":
        """Clip tensor values to range (NumPy compatibility)."""
        return self.clamp(min_val, max_val)

    # Array testing
    def isnan(self) -> "Tensor":
        """Test for NaN values."""
        result = Tensor.__new__(Tensor)
        result._tensor = self._tensor.isnan()
        return result

    def isinf(self) -> "Tensor":
        """Test for infinite values."""
        result = Tensor.__new__(Tensor)
        result._tensor = self._tensor.isinf()
        return result

    def isfinite(self) -> "Tensor":
        """Test for finite values."""
        result = Tensor.__new__(Tensor)
        result._tensor = self._tensor.isfinite()
        return result

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

    # Indexing and slicing (simplified)
    def __getitem__(self, key):
        """Tensor indexing and slicing."""
        result_np = self.numpy()[key]
        return Tensor(np.array(result_np, dtype=np.float32))

    def __setitem__(self, key, value):
        """Tensor item assignment."""
        array = self.numpy_copy()
        if isinstance(value, Tensor):
            array[key] = value.numpy()
        else:
            array[key] = value
        self._tensor = _minitensor_core.Tensor.from_numpy(array, self.requires_grad)

    # Static tensor creation methods
    @staticmethod
    def zeros(
        *shape: Union[int, Sequence[int]],
        dtype: str = "float32",
        device=None,
        requires_grad: bool = False,
    ) -> "Tensor":
        """Create a tensor filled with zeros."""
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        result = Tensor.__new__(Tensor)
        result._tensor = _minitensor_core.Tensor.zeros(
            list(shape), dtype, device, requires_grad
        )
        return result

    @staticmethod
    def ones(
        *shape: Union[int, Sequence[int]],
        dtype: str = "float32",
        device=None,
        requires_grad: bool = False,
    ) -> "Tensor":
        """Create a tensor filled with ones."""
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        result = Tensor.__new__(Tensor)
        result._tensor = _minitensor_core.Tensor.ones(
            list(shape), dtype, device, requires_grad
        )
        return result

    @staticmethod
    def full(
        shape: Sequence[int],
        fill_value: float,
        dtype: str = "float32",
        device=None,
        requires_grad: bool = False,
    ) -> "Tensor":
        """Create a tensor filled with a specific value."""
        result = Tensor.__new__(Tensor)
        result._tensor = _minitensor_core.Tensor.full(
            list(shape), fill_value, dtype, device, requires_grad
        )
        return result

    @staticmethod
    def rand(
        *shape: Union[int, Sequence[int]],
        dtype: str = "float32",
        device=None,
        requires_grad: bool = False,
    ) -> "Tensor":
        """Create a tensor with random values from uniform distribution [0, 1)."""
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        result = Tensor.__new__(Tensor)
        result._tensor = _minitensor_core.Tensor.rand(
            list(shape), dtype, device, requires_grad
        )
        return result

    @staticmethod
    def randn(
        *shape: Union[int, Sequence[int]],
        dtype: str = "float32",
        device=None,
        requires_grad: bool = False,
    ) -> "Tensor":
        """Create a tensor with random values from standard normal distribution."""
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        result = Tensor.__new__(Tensor)
        result._tensor = _minitensor_core.Tensor.randn(
            list(shape), dtype, device, requires_grad
        )
        return result

    @staticmethod
    def eye(
        n: int,
        m: Optional[int] = None,
        dtype: str = "float32",
        device=None,
        requires_grad: bool = False,
    ) -> "Tensor":
        """Create an identity matrix."""
        result = Tensor.__new__(Tensor)
        result._tensor = _minitensor_core.Tensor.eye(n, m, dtype, device, requires_grad)
        return result

    @staticmethod
    def arange(
        start: float,
        end: Optional[float] = None,
        step: float = 1.0,
        dtype: str = "float32",
        device=None,
        requires_grad: bool = False,
    ) -> "Tensor":
        """Create a tensor with evenly spaced values."""
        if end is None:
            end = start
            start = 0.0
        result = Tensor.__new__(Tensor)
        result._tensor = _minitensor_core.Tensor.arange(
            start, end, step, dtype, device, requires_grad
        )
        return result

    @staticmethod
    def linspace(
        start: float,
        end: float,
        steps: int,
        dtype: str = "float32",
        device=None,
        requires_grad: bool = False,
    ) -> "Tensor":
        """Create a tensor with linearly spaced values."""
        if steps <= 1:
            raise ValueError("Number of steps must be greater than 1")
        step = (end - start) / (steps - 1)
        return Tensor.arange(start, end + step / 2, step, dtype, device, requires_grad)

    @staticmethod
    def logspace(
        start: float,
        end: float,
        steps: int,
        base: float = 10.0,
        dtype: str = "float32",
        device=None,
        requires_grad: bool = False,
    ) -> "Tensor":
        """Create a tensor with logarithmically spaced values."""
        linear = Tensor.linspace(start, end, steps, dtype, device, requires_grad)
        return Tensor(base) ** linear

    @staticmethod
    def from_numpy(array: np.ndarray, requires_grad: bool = False) -> "Tensor":
        """Create a tensor from a NumPy array."""
        result = Tensor.__new__(Tensor)
        result._tensor = _minitensor_core.Tensor.from_numpy(array, requires_grad)
        return result

    @staticmethod
    def from_numpy_shared(array: np.ndarray, requires_grad: bool = False) -> "Tensor":
        """Create a tensor from a NumPy array with zero-copy when possible."""
        result = Tensor.__new__(Tensor)
        result._tensor = _minitensor_core.Tensor.from_numpy_shared(array, requires_grad)
        return result


# Convenience functions for tensor creation (NumPy-style)
def tensor(
    data: Any, dtype: Optional[str] = None, device=None, requires_grad: bool = False
) -> Tensor:
    """Create a tensor from data."""
    return Tensor(data, requires_grad=requires_grad, dtype=dtype, device=device)


def zeros(
    *shape: Union[int, Sequence[int]],
    dtype: str = "float32",
    device=None,
    requires_grad: bool = False,
) -> Tensor:
    """Create a tensor filled with zeros."""
    return Tensor.zeros(*shape, dtype=dtype, device=device, requires_grad=requires_grad)


def ones(
    *shape: Union[int, Sequence[int]],
    dtype: str = "float32",
    device=None,
    requires_grad: bool = False,
) -> Tensor:
    """Create a tensor filled with ones."""
    return Tensor.ones(*shape, dtype=dtype, device=device, requires_grad=requires_grad)


def full(
    shape: Sequence[int],
    fill_value: float,
    dtype: str = "float32",
    device=None,
    requires_grad: bool = False,
) -> Tensor:
    """Create a tensor filled with a specific value."""
    return Tensor.full(
        shape, fill_value, dtype=dtype, device=device, requires_grad=requires_grad
    )


def rand(
    *shape: Union[int, Sequence[int]],
    dtype: str = "float32",
    device=None,
    requires_grad: bool = False,
) -> Tensor:
    """Create a tensor with random values from uniform distribution."""
    return Tensor.rand(*shape, dtype=dtype, device=device, requires_grad=requires_grad)


def randn(
    *shape: Union[int, Sequence[int]],
    dtype: str = "float32",
    device=None,
    requires_grad: bool = False,
) -> Tensor:
    """Create a tensor with random values from normal distribution."""
    return Tensor.randn(*shape, dtype=dtype, device=device, requires_grad=requires_grad)


def eye(
    n: int,
    m: Optional[int] = None,
    dtype: str = "float32",
    device=None,
    requires_grad: bool = False,
) -> Tensor:
    """Create an identity matrix."""
    return Tensor.eye(n, m, dtype=dtype, device=device, requires_grad=requires_grad)


def arange(
    start: float,
    end: Optional[float] = None,
    step: float = 1.0,
    dtype: str = "float32",
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
    dtype: str = "float32",
    device=None,
    requires_grad: bool = False,
) -> Tensor:
    """Create a tensor with linearly spaced values."""
    return Tensor.linspace(
        start, end, steps, dtype=dtype, device=device, requires_grad=requires_grad
    )


def from_numpy(array: np.ndarray, requires_grad: bool = False) -> Tensor:
    """Create a tensor from a NumPy array."""
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
]
