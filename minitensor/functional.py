# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Functional interface for neural network operations.

This module provides functional versions of neural network operations
that can be used without creating layer objects.
"""

try:
    from . import _core as _minitensor_core
except ImportError as e:
    raise ImportError(
        "The minitensor core extension is not built. "
        "Run `maturin develop` or install the package."
    ) from e

from typing import Optional, Sequence, Union

from .tensor import Tensor


def relu(input: Tensor) -> Tensor:
    """Rectified Linear Unit activation.

    Args:
        input: Input tensor to activate.

    Returns:
        Tensor: A new tensor with the ReLU function applied element-wise.
    """
    return input.relu()


def sigmoid(input: Tensor) -> Tensor:
    """Sigmoid activation function.

    Args:
        input: Input tensor to activate.

    Returns:
        Tensor: A new tensor with values in the range (0, 1).
    """
    return input.sigmoid()


def tanh(input: Tensor) -> Tensor:
    """Hyperbolic tangent activation function.

    Args:
        input: Input tensor to activate.

    Returns:
        Tensor: A new tensor with values in the range (-1, 1).
    """
    return input.tanh()


def sin(input: Tensor) -> Tensor:
    """Sine function."""
    return input.sin()


def cos(input: Tensor) -> Tensor:
    """Cosine function."""
    return input.cos()


def tan(input: Tensor) -> Tensor:
    """Tangent function computed as sin(x)/cos(x)."""
    return input.tan()


def reshape(input: Tensor, shape) -> Tensor:
    """Reshape ``input`` to ``shape``.

    Args:
        input: Tensor to reshape.
        shape: New shape as a sequence of integers or a single integer.

    Returns:
        Tensor: A view of ``input`` with the specified shape.
    """

    return input.reshape(shape)


def view(input: Tensor, *shape: Union[int, Sequence[int]]) -> Tensor:
    """Return a view of ``input`` with a new shape.

    Args:
        input: Tensor to reshape.
        *shape: Desired view shape. ``-1`` indicates that the size of the
            corresponding dimension should be inferred.

    Returns:
        Tensor: A view of ``input`` with the specified shape.
    """

    return input.view(*shape)


def flatten(input: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
    """Flatten input tensor across specified dimensions.

    Args:
        input: Tensor to be flattened.
        start_dim: First dimension to flatten.
        end_dim: Last dimension to flatten.

    Returns:
        Tensor: A flattened view of the input tensor.
    """
    return input.flatten(start_dim, end_dim)


def ravel(input: Tensor) -> Tensor:
    """Flatten all dimensions of the input tensor.

    Args:
        input: Tensor to be flattened.

    Returns:
        Tensor: A view of ``input`` flattened to one dimension.
    """

    return input.ravel()


def transpose(input: Tensor, dim0: int, dim1: int) -> Tensor:
    """Swap two dimensions of the input tensor.

    Args:
        input: Tensor to transpose.
        dim0: First dimension to swap.
        dim1: Second dimension to swap.

    Returns:
        Tensor: A view of ``input`` with ``dim0`` and ``dim1`` swapped.
    """

    return input.transpose(dim0, dim1)


def permute(input: Tensor, dims: Sequence[int]) -> Tensor:
    """Permute the dimensions of the input tensor.

    Args:
        input: Tensor whose dimensions to permute.
        dims: Sequence specifying the new order of dimensions.

    Returns:
        Tensor: A view of ``input`` with dimensions reordered as ``dims``.
    """

    return input.permute(dims)


def movedim(
    input: Tensor,
    source: Union[int, Sequence[int]],
    destination: Union[int, Sequence[int]],
) -> Tensor:
    """Move tensor dimensions to new positions.

    Args:
        input: Tensor whose dimensions to move.
        source: Source dimensions to move. Can be a single integer or a
            sequence of integers.
        destination: Destination dimensions to move to. Can be a single
            integer or a sequence of integers.

    Returns:
        Tensor: A view of ``input`` with dimensions moved to new positions.
    """

    return input.movedim(source, destination)


# Alias matching PyTorch's ``moveaxis`` name
moveaxis = movedim


def swapaxes(input: Tensor, axis0: int, axis1: int) -> Tensor:
    """Swap two axes of ``input``.

    Args:
        input: Tensor whose axes to swap.
        axis0: First axis to swap.
        axis1: Second axis to swap.

    Returns:
        Tensor: A view of ``input`` with axes ``axis0`` and ``axis1`` swapped.
    """

    return input.swapaxes(axis0, axis1)


swapdims = swapaxes


def squeeze(input: Tensor, dim: Optional[int] = None) -> Tensor:
    """Remove dimensions of size 1 from ``input``.

    Args:
        input: Tensor to squeeze.
        dim: Specific dimension to squeeze. If ``None``, all dimensions of
            size 1 are removed.

    Returns:
        Tensor: A view of ``input`` with singleton dimensions removed.
    """

    return input.squeeze(dim)


def unsqueeze(input: Tensor, dim: int) -> Tensor:
    """Insert a dimension of size 1 into ``input`` at ``dim``.

    Args:
        input: Tensor to unsqueeze.
        dim: Position at which to insert the singleton dimension.

    Returns:
        Tensor: A view of ``input`` with an extra dimension of size 1.
    """

    return input.unsqueeze(dim)


def expand(input: Tensor, *shape: int) -> Tensor:
    """Expand ``input`` to a larger size without allocating new memory.

    Args:
        input: Tensor to expand.
        *shape: Desired expanded shape or a single sequence specifying the
            size. ``-1`` indicates that the size of the corresponding
            dimension should not be changed.

    Returns:
        Tensor: A view of ``input`` expanded to ``shape``.
    """

    return input.expand(*shape)


def repeat(input: Tensor, *repeats: int) -> Tensor:
    """Repeat ``input`` along each dimension.

    Args:
        input: Tensor to repeat.
        *repeats: The number of repeats for each dimension. May also be a
            single sequence specifying the repeats.

    Returns:
        Tensor: The repeated tensor.
    """

    return input.repeat(*repeats)


def flip(input: Tensor, dims: Union[int, Sequence[int]]) -> Tensor:
    """Flip ``input`` along specified dimensions."""

    return input.flip(dims)


def roll(
    input: Tensor,
    shifts: Union[int, Sequence[int]],
    dims: Optional[Union[int, Sequence[int]]] = None,
) -> Tensor:
    """Roll tensor elements along specified dimensions."""

    return input.roll(shifts, dims)


def narrow(input: Tensor, dim: int, start: int, length: int) -> Tensor:
    """Narrow ``input`` along ``dim`` starting at ``start`` for ``length`` elements."""

    return input.narrow(dim, start, length)


def cat(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    """Concatenate ``tensors`` along an existing dimension.

    Args:
        tensors: Sequence of tensors to concatenate.
        dim: Dimension along which to concatenate. May be negative to
            index from the end.

    Returns:
        Tensor: A single tensor formed by joining ``tensors`` along ``dim``.
    """

    core_tensors = [t._tensor for t in tensors]
    result = Tensor.__new__(Tensor)
    result._tensor = _minitensor_core.Tensor.concatenate(core_tensors, dim)
    return result


def stack(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    """Stack ``tensors`` along a new dimension ``dim``.

    Args:
        tensors: Sequence of tensors to stack.
        dim: The index at which to insert the new dimension. May be
            negative to index from the end.

    Returns:
        Tensor: A tensor containing all inputs stacked along the new
        dimension.
    """

    core_tensors = [t._tensor for t in tensors]
    result = Tensor.__new__(Tensor)
    result._tensor = _minitensor_core.Tensor.stack(core_tensors, dim)
    return result


def split(
    input: Tensor, split_size_or_sections: Union[int, Sequence[int]], dim: int = 0
) -> list[Tensor]:
    """Split ``input`` into chunks along ``dim``.

    Args:
        input: Tensor to split.
        split_size_or_sections: Size of each chunk or list of sizes for each
            chunk.
        dim: Dimension along which to split the tensor.

    Returns:
        list[Tensor]: List of tensors resulting from the split.
    """

    return input.split(split_size_or_sections, dim)


def chunk(input: Tensor, chunks: int, dim: int = 0) -> list[Tensor]:
    """Split ``input`` into ``chunks`` along ``dim``.

    Args:
        input: Tensor to split.
        chunks: Number of chunks to return. The size of ``input`` along ``dim``
            must be divisible by ``chunks``.
        dim: Dimension along which to split the tensor.

    Returns:
        list[Tensor]: List of tensors resulting from the split.
    """

    return input.chunk(chunks, dim)


def index_select(input: Tensor, dim: int, indices: Sequence[int]) -> Tensor:
    """Select elements along ``dim`` using integer ``indices``.

    Args:
        input: Tensor to index.
        dim: Dimension along which to select.
        indices: Sequence of integer indices specifying which elements to
            gather.

    Returns:
        Tensor: Tensor containing the selected elements.
    """

    return input.index_select(dim, indices)


def gather(input: Tensor, dim: int, index: Tensor) -> Tensor:
    """Gather elements from ``input`` along ``dim`` using ``index`` tensor.

    Args:
        input: Source tensor.
        dim: Dimension along which to gather.
        index: Tensor containing indices to gather.

    Returns:
        Tensor: Result tensor with same shape as ``index``.
    """

    return input.gather(dim, index)


def softmax(input: Tensor, dim: Optional[int] = None) -> Tensor:
    """Softmax activation function.

    Args:
        input: Input tensor.
        dim: Dimension along which to apply softmax. Defaults to the last
            dimension.

    Returns:
        Tensor: Probability distribution computed along ``dim``.
    """
    axis = dim
    if axis is None:
        axis = len(input.shape) - 1
    elif axis < 0:
        axis = len(input.shape) + axis
    return input.softmax(axis)


def dense_layer(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    """Dense layer transformation ``y = xW^T + b``.

    Args:
        input: Input tensor of shape ``(N, in_features)``.
        weight: Weight tensor of shape ``(out_features, in_features)``.
        bias: Optional bias tensor of shape ``(out_features)``.

    Returns:
        Tensor: Output tensor of shape ``(N, out_features)``.
    """
    result = input.matmul(weight.transpose())
    if bias is not None:
        result = result + bias
    return result


def conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, tuple] = 1,
    padding: Union[int, tuple] = 0,
) -> Tensor:
    """2D convolution operation.

    Args:
        input: Input tensor of shape ``(N, C_in, H, W)``.
        weight: Convolution filters of shape ``(C_out, C_in, kH, kW)``.
        bias: Optional bias tensor of shape ``(C_out)``.
        stride: Stride of the convolution (int or tuple of two ints).
        padding: Implicit zero padding on both sides (int or tuple of two ints).

    Returns:
        Tensor: Result of the convolution.
    """
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    bias_tensor = None if bias is None else bias._tensor
    result = _minitensor_core.nn.conv2d(
        input._tensor, weight._tensor, bias_tensor, stride, padding
    )
    tensor = Tensor.__new__(Tensor)
    tensor._tensor = result
    return tensor


def batch_norm(
    input: Tensor,
    running_mean: Optional[Tensor] = None,
    running_var: Optional[Tensor] = None,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    training: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    """Batch normalization.

    Args:
        input: Input tensor.
        running_mean: Running mean for evaluation mode.
        running_var: Running variance for evaluation mode.
        weight: Learnable scale parameter.
        bias: Learnable shift parameter.
        training: Whether batch statistics or running estimates are used.
        momentum: Momentum for updating running statistics.
        eps: Small value added to variance for numerical stability.

    Returns:
        Tensor: Normalized tensor.
    """
    rm_tensor = None if running_mean is None else running_mean._tensor
    rv_tensor = None if running_var is None else running_var._tensor
    weight_tensor = None if weight is None else weight._tensor
    bias_tensor = None if bias is None else bias._tensor
    result = _minitensor_core.nn.batch_norm(
        input._tensor,
        rm_tensor,
        rv_tensor,
        weight_tensor,
        bias_tensor,
        training,
        momentum,
        eps,
    )
    tensor = Tensor.__new__(Tensor)
    tensor._tensor = result
    return tensor


def dropout(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """Dropout regularization.

    This function forwards to the Rust-backed ``Dropout`` module. The probability
    ``p`` specifies how likely each element is zeroed. When ``training`` is
    ``False`` the input tensor is returned unchanged, matching standard deep
    learning library semantics.

    Args:
        input: Input tensor.
        p: Probability of an element to be zeroed.
        training: Apply dropout if ``True``; return input unchanged otherwise.

    Returns:
        Tensor: Tensor with randomly zeroed elements when training, or the
        original tensor during evaluation mode.
    """
    layer = _minitensor_core.nn.Dropout(p)
    if training:
        layer.train()
    else:
        layer.eval()
    result = layer.forward(input._tensor)
    tensor = Tensor.__new__(Tensor)
    tensor._tensor = result
    return tensor


def mse_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """Mean squared error loss.

    Args:
        input: Predicted values.
        target: Ground truth values.
        reduction: Specifies the reduction to apply to the output: ``'none'``,
            ``'mean'`` or ``'sum'``.

    Returns:
        Tensor: Scalar loss tensor.
    """
    # This would use the MSELoss implementation from the Rust backend
    loss_fn = _minitensor_core.nn.MSELoss(reduction)
    result = loss_fn.forward(input._tensor, target._tensor)
    tensor = Tensor.__new__(Tensor)
    tensor._tensor = result
    return tensor


def cross_entropy(
    input: Tensor, target: Tensor, reduction: str = "mean", dim: int = 1
) -> Tensor:
    """Cross entropy loss.

    The input is treated as raw logits and the target may contain class indices
    or one-hot encoded probabilities. All dimension handling and reduction
    computations are delegated to the Rust backend.

    Args:
        input: Predicted logit values.
        target: Target class indices or one-hot vectors matching ``input``.
        reduction: Specifies the reduction to apply to the output. One of
            ``"mean"``, ``"sum"`` or ``"none"``.
        dim: Dimension representing the classes. Negative values are supported
            to index from the end.

    Returns:
        Tensor: Loss tensor. A scalar for ``"mean"``/``"sum"`` reductions or a
        tensor with ``input.shape`` excluding ``dim`` when ``reduction="none"``.
    """

    result = _minitensor_core.nn.cross_entropy(
        input._tensor, target._tensor, reduction, dim
    )
    tensor = Tensor.__new__(Tensor)
    tensor._tensor = result
    return tensor


def binary_cross_entropy(
    input: Tensor, target: Tensor, reduction: str = "mean"
) -> Tensor:
    """Binary cross entropy loss.

    Args:
        input: Predicted probabilities.
        target: Target probabilities.
        reduction: Specifies the reduction to apply to the output.

    Returns:
        Tensor: Scalar loss tensor.
    """
    # This would use the BCELoss implementation from the Rust backend
    loss_fn = _minitensor_core.nn.BCELoss(reduction)
    result = loss_fn.forward(input._tensor, target._tensor)
    tensor = Tensor.__new__(Tensor)
    tensor._tensor = result
    return tensor


__all__ = [
    "relu",
    "sigmoid",
    "tanh",
    "sin",
    "cos",
    "tan",
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
    "narrow",
    "cat",
    "stack",
    "split",
    "chunk",
    "index_select",
    "gather",
    "softmax",
    "dense_layer",
    "conv2d",
    "batch_norm",
    "dropout",
    "mse_loss",
    "cross_entropy",
    "binary_cross_entropy",
]
