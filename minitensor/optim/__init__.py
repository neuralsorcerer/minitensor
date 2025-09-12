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

from typing import List

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
