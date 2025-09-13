# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Optimisation algorithms for Minitensor.

The optimisers are implemented in Rust and exported through PyO3 bindings.  In
minimal builds some of them might be absent which previously resulted in import
errors when :mod:`minitensor` was imported.  This module mirrors the approach
used in :mod:`minitensor.nn` and only exposes the symbols that are actually
available in the core.
"""

from typing import Dict, List

from ..tensor import Tensor

try:  # pragma: no cover - tested via public API
    from .. import _core as _minitensor_core  # type: ignore
except Exception:  # pylint: disable=broad-except
    try:
        import minitensor._core as _minitensor_core  # type: ignore
    except Exception:  # pragma: no cover
        _minitensor_core = None  # type: ignore

__all__: List[str] = []

if _minitensor_core is not None and hasattr(_minitensor_core, "optim"):
    _optim = _minitensor_core.optim

    def _wrap_optimizer(cls):
        class WrappedOpt:
            def __init__(self, *args, **kwargs):
                self._opt = cls(*args, **kwargs)

            def _unwrap(self, params):
                return [p._tensor if isinstance(p, Tensor) else p for p in params]

            def zero_grad(self, params):
                self._opt.zero_grad(self._unwrap(params))

            def step(self, params):
                self._opt.step(self._unwrap(params))

        WrappedOpt.__name__ = cls.__name__
        return WrappedOpt

    for _name in dir(_optim):
        if _name.startswith("_"):
            continue
        obj = getattr(_optim, _name)
        if isinstance(obj, type):
            globals()[_name] = _wrap_optimizer(obj)
        else:
            globals()[_name] = obj
        __all__.append(_name)
# If the core is missing nothing is exported which keeps imports harmless.

if "SGD" not in globals():

    class SGD:
        """Basic stochastic gradient descent optimizer.

        Parameters are updated in-place using tensor operations that dispatch to
        the Rust backend, ensuring all heavy computation remains in native code.
        Momentum is supported but optional and defaults to the simple form used in
        the examples.
        """

        def __init__(
            self,
            lr: float,
            momentum: float = 0.0,
            weight_decay: float = 0.0,
            nesterov: bool = False,
        ) -> None:
            self.lr = lr
            self.momentum = momentum
            self.weight_decay = weight_decay
            self.nesterov = nesterov
            self._velocity: Dict[int, Tensor] = {}

        def zero_grad(self, params) -> None:
            for p in params:
                p.zero_grad()

        def step(self, params) -> None:
            for p in params:
                grad = p.grad
                if grad is None:
                    continue
                grad = grad.detach()
                if self.weight_decay != 0.0:
                    grad = (grad + self.weight_decay * p).detach()
                if self.momentum != 0.0:
                    v = self._velocity.setdefault(
                        id(p), Tensor.zeros(p.shape, dtype=p.dtype, device=p.device)
                    )
                    v = (self.momentum * v + grad).detach()
                    self._velocity[id(p)] = v
                    grad = v + self.momentum * v if self.nesterov else v
                update = (p - self.lr * grad).detach()
                p[...] = update

    __all__.append("SGD")
