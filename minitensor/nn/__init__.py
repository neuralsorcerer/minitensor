# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Neural network modules and layers for Minitensor.

This module provides building blocks for creating neural networks including
layers, activation functions, and utilities. The heavy lifting is implemented
in Rust and exposed through PyO3 bindings.

The original implementation assumed that every piece of functionality was
always present in the compiled extension.  The stripped down test environment
ships a minimal core where many of these components are missing which caused
importing :mod:`minitensor` to fail with ``AttributeError``.  To make the Python
API robust we now populate the namespace dynamically - only symbols that exist
in the core are exported.  This mirrors the behaviour of optional components in
other numerical libraries and allows parts of the library (such as the tensor
API) to be used independently of the neural network layers.
"""

from typing import Dict, List, Type

from ..tensor import Tensor

# Try to import the Rust core.  During development the extension might not be
# built yet, so we swallow errors and simply expose an empty module.
try:  # pragma: no cover - behaviour is tested indirectly
    from .. import _core as _minitensor_core  # type: ignore
except Exception:  # pylint: disable=broad-except
    try:
        import minitensor._core as _minitensor_core  # type: ignore
    except Exception:  # pragma: no cover
        _minitensor_core = None  # type: ignore

__all__: List[str] = []

if _minitensor_core is not None and hasattr(_minitensor_core, "nn"):
    _nn = _minitensor_core.nn

    def _wrap_module_class(cls: Type) -> Type:
        class WrappedModule:
            """Thin Python wrapper around a Rust ``nn`` module.

            It unwraps :class:`~minitensor.tensor.Tensor` arguments to their
            underlying Rust representations and wraps tensor outputs back into
            the Python ``Tensor`` class.
            """

            def __init__(self, *args, **kwargs):
                self._module = cls(*args, **kwargs)

            def __getattr__(self, name):
                return getattr(self._module, name)

            def _wrap(self, obj):
                if isinstance(obj, _minitensor_core.Tensor):
                    tensor = Tensor.__new__(Tensor)
                    tensor._tensor = obj
                    return tensor
                return obj

            def forward(self, *args, **kwargs):
                saw_python_tensor = False
                new_args = []
                for a in args:
                    if isinstance(a, Tensor):
                        saw_python_tensor = True
                        new_args.append(a._tensor)
                    else:
                        new_args.append(a)
                new_kwargs = {}
                for k, v in kwargs.items():
                    if isinstance(v, Tensor):
                        saw_python_tensor = True
                        new_kwargs[k] = v._tensor
                    else:
                        new_kwargs[k] = v
                result = self._module.forward(*new_args, **new_kwargs)
                if saw_python_tensor:
                    return self._wrap(result)
                return result

            __call__ = forward

            def parameters(self):
                """Return trainable parameters as live Python ``Tensor`` objects."""

                params: list[Tensor] = []
                if hasattr(self._module, "parameters"):
                    for obj in self._module.parameters():
                        tensor = Tensor.__new__(Tensor)
                        tensor._tensor = obj
                        params.append(tensor)

                # Recurse into known child modules stored on the Python wrapper
                # itself (e.g. ``Sequential.layers``).
                for attr in dir(self):
                    child = getattr(self, attr)
                    if isinstance(child, list):
                        for item in child:
                            if hasattr(item, "_module"):
                                params.extend(item.parameters())
                    elif hasattr(child, "_module") and child is not self:
                        params.extend(child.parameters())

                return params

            def summary(
                self, name: str | None = None
            ):  # pragma: no cover - thin wrapper
                if not hasattr(self._module, "summary"):
                    raise AttributeError("Underlying module lacks summary")
                if name is None:
                    name = self.__class__.__name__
                return self._module.summary(name)

        WrappedModule.__name__ = cls.__name__
        return WrappedModule

    _special_wrappers: Dict[str, Type] = {}

    def _install(name: str, cls: Type) -> None:
        wrapped = _wrap_module_class(cls)

        if name == "DenseLayer":

            class DenseLayer(wrapped):  # pragma: no cover - thin wrapper
                def __init__(
                    self,
                    in_features: int,
                    out_features: int,
                    bias: bool = True,
                    device=None,
                    dtype=None,
                ) -> None:
                    super().__init__(in_features, out_features, bias, device, dtype)

            _special_wrappers[name] = DenseLayer
        elif name == "BatchNorm1d":

            class BatchNorm1d(wrapped):  # pragma: no cover - thin wrapper
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

            _special_wrappers[name] = BatchNorm1d
        elif name == "BatchNorm2d":

            class BatchNorm2d(wrapped):  # pragma: no cover - thin wrapper
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

            _special_wrappers[name] = BatchNorm2d
        elif name == "MSELoss":

            class MSELoss(wrapped):  # pragma: no cover - thin wrapper
                def __init__(self, reduction: str = "mean") -> None:
                    super().__init__(reduction)

            _special_wrappers[name] = MSELoss
        elif name == "MAELoss":

            class MAELoss(wrapped):  # pragma: no cover - thin wrapper
                def __init__(self, reduction: str = "mean") -> None:
                    super().__init__(reduction)

            _special_wrappers[name] = MAELoss
        elif name == "HuberLoss":

            class HuberLoss(wrapped):  # pragma: no cover - thin wrapper
                def __init__(self, delta: float = 1.0, reduction: str = "mean") -> None:
                    super().__init__(delta, reduction)

            _special_wrappers[name] = HuberLoss
        elif name == "CrossEntropyLoss":

            class CrossEntropyLoss(wrapped):  # pragma: no cover - thin wrapper
                def __init__(self, reduction: str = "mean") -> None:
                    super().__init__(reduction)

            _special_wrappers[name] = CrossEntropyLoss
        elif name == "BCELoss":

            class BCELoss(wrapped):  # pragma: no cover - thin wrapper
                def __init__(self, reduction: str = "mean") -> None:
                    super().__init__(reduction)

            _special_wrappers[name] = BCELoss
        elif name == "FocalLoss":

            class FocalLoss(wrapped):  # pragma: no cover - thin wrapper
                def __init__(
                    self,
                    alpha: float = 0.25,
                    gamma: float = 2.0,
                    reduction: str = "mean",
                ) -> None:
                    super().__init__(alpha, gamma, reduction)

            _special_wrappers[name] = FocalLoss
        elif name == "Sequential":

            class Sequential(wrapped):  # pragma: no cover - thin wrapper
                def __init__(self, layers):
                    self.layers = layers
                    core_layers = [getattr(l, "_module", l) for l in layers]
                    super().__init__(core_layers)

            _special_wrappers[name] = Sequential
        else:
            _special_wrappers[name] = wrapped

    for _name in dir(_nn):
        if _name.startswith("_"):
            continue
        attr = getattr(_nn, _name)
        if isinstance(attr, type):
            _install(_name, attr)

    for _name, _cls in _special_wrappers.items():
        globals()[_name] = _cls
        __all__.append(_name)
# When the core is missing we simply expose an empty namespace so that importing
# :mod:`minitensor.nn` never raises an exception.
