# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
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

from typing import Optional, Union
import math

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
    "softmax",
    "dense_layer",
    "conv2d",
    "batch_norm",
    "dropout",
    "mse_loss",
    "cross_entropy",
    "binary_cross_entropy",
]
