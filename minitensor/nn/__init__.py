# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Neural network building blocks backed by the Rust core."""

from __future__ import annotations

from typing import List as _List
from typing import Type as _Type

from .. import _core as _minitensor_core  # type: ignore
from ..tensor import Tensor as _Tensor

_nn = _minitensor_core.nn

__all__: _List[str] = [
    "DenseLayer",
    "BatchNorm1d",
    "BatchNorm2d",
    "Conv2d",
    "Dropout",
    "Dropout2d",
    "ELU",
    "FocalLoss",
    "GELU",
    "HuberLoss",
    "LeakyReLU",
    "MAELoss",
    "MSELoss",
    "ReLU",
    "Sequential",
    "Sigmoid",
    "Softmax",
    "Tanh",
    "BCELoss",
    "CrossEntropyLoss",
    "batch_norm",
    "conv2d",
    "cross_entropy",
]


def _unwrap(obj):
    return obj._tensor if isinstance(obj, _Tensor) else obj


def _wrap(obj):
    if isinstance(obj, _minitensor_core.Tensor):
        tensor = _Tensor.__new__(_Tensor)
        tensor._tensor = obj
        return tensor
    return obj


def _wrap_module_class(cls: _Type) -> _Type:
    class WrappedModule:
        def __init__(self, *args, **kwargs):
            self._module = cls(*args, **kwargs)

        def __getattr__(self, name):  # pragma: no cover - thin wrapper
            return getattr(self._module, name)

        def forward(self, *args, **kwargs):
            saw_tensor = False
            new_args = []
            for a in args:
                if isinstance(a, _Tensor):
                    saw_tensor = True
                    new_args.append(a._tensor)
                else:
                    new_args.append(a)
            new_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, _Tensor):
                    saw_tensor = True
                    new_kwargs[k] = v._tensor
                else:
                    new_kwargs[k] = v
            result = self._module.forward(*new_args, **new_kwargs)
            if saw_tensor:
                return _wrap(result)
            return result

        __call__ = forward

        def parameters(self):  # pragma: no cover - thin wrapper
            params: _List[_Tensor] = []
            if hasattr(self._module, "parameters"):
                for obj in self._module.parameters():
                    params.append(_wrap(obj))
            for attr in dir(self):
                child = getattr(self, attr)
                if isinstance(child, list):
                    for item in child:
                        if hasattr(item, "_module"):
                            params.extend(item.parameters())
                elif hasattr(child, "_module") and child is not self:
                    params.extend(child.parameters())
            return params

        def summary(self, name: str | None = None):  # pragma: no cover - thin wrapper
            if not hasattr(self._module, "summary"):
                raise AttributeError("Underlying module lacks summary")
            if name is None:
                name = self.__class__.__name__
            return self._module.summary(name)

    WrappedModule.__name__ = cls.__name__
    return WrappedModule


class DenseLayer(_wrap_module_class(_nn.DenseLayer)):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

    def parameters(self):  # pragma: no cover - thin wrapper
        params: _List[_Tensor] = []
        w = _Tensor.__new__(_Tensor)
        w._tensor = self._module.weight
        params.append(w)
        if getattr(self._module, "bias", None) is not None:
            b = _Tensor.__new__(_Tensor)
            b._tensor = self._module.bias
            params.append(b)
        return params


class BatchNorm1d(_wrap_module_class(_nn.BatchNorm1d)):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(num_features, eps, momentum, affine, device, dtype)


class BatchNorm2d(_wrap_module_class(_nn.BatchNorm2d)):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(num_features, eps, momentum, affine, device, dtype)


class Conv2d(_wrap_module_class(_nn.Conv2d)):
    pass


class Dropout(_wrap_module_class(_nn.Dropout)):
    pass


class Dropout2d(_wrap_module_class(_nn.Dropout2d)):
    pass


class ELU(_wrap_module_class(_nn.ELU)):
    pass


class FocalLoss(_wrap_module_class(_nn.FocalLoss)):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__(alpha, gamma, reduction)


class GELU(_wrap_module_class(_nn.GELU)):
    pass


class HuberLoss(_wrap_module_class(_nn.HuberLoss)):
    def __init__(self, delta: float = 1.0, reduction: str = "mean") -> None:
        super().__init__(delta, reduction)


class LeakyReLU(_wrap_module_class(_nn.LeakyReLU)):
    pass


class MAELoss(_wrap_module_class(_nn.MAELoss)):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(reduction)


class MSELoss(_wrap_module_class(_nn.MSELoss)):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(reduction)


class ReLU(_wrap_module_class(_nn.ReLU)):
    pass


class Sequential(_wrap_module_class(_nn.Sequential)):
    def __init__(self, layers):
        self.layers = layers
        core_layers = [getattr(l, "_module", l) for l in layers]
        super().__init__(core_layers)


class Sigmoid(_wrap_module_class(_nn.Sigmoid)):
    pass


class Softmax(_wrap_module_class(_nn.Softmax)):
    pass


class Tanh(_wrap_module_class(_nn.Tanh)):
    pass


class BCELoss(_wrap_module_class(_nn.BCELoss)):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(reduction)


class CrossEntropyLoss(_wrap_module_class(_nn.CrossEntropyLoss)):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(reduction)


def batch_norm(
    input,
    running_mean=None,
    running_var=None,
    weight=None,
    bias=None,
    training: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
):
    out = _nn.batch_norm(
        _unwrap(input),
        _unwrap(running_mean),
        _unwrap(running_var),
        _unwrap(weight),
        _unwrap(bias),
        training,
        momentum,
        eps,
    )
    return _wrap(out)


def conv2d(input, weight, bias=None, stride=None, padding=None):
    out = _nn.conv2d(
        _unwrap(input),
        _unwrap(weight),
        _unwrap(bias),
        stride,
        padding,
    )
    return _wrap(out)


def cross_entropy(input, target, reduction: str = "mean", dim: int = 1):
    out = _nn.cross_entropy(
        _unwrap(input),
        _unwrap(target),
        reduction,
        dim,
    )
    return _wrap(out)


try:
    del annotations  # type: ignore  # noqa: F401
except Exception:
    pass
