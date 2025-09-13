# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Optimisation algorithms backed by the Rust core."""

from __future__ import annotations

from typing import List as _List

from .. import _core as _minitensor_core  # type: ignore
from ..tensor import Tensor as _Tensor

_optim = _minitensor_core.optim

__all__: _List[str] = ["SGD", "Adam", "RMSprop"]


def _unwrap(params):
    return [p._tensor if isinstance(p, _Tensor) else p for p in params]


class _OptimizerWrapper:
    """Thin wrapper mirroring PyTorch's optimiser interface."""

    def __init__(self, opt, params) -> None:
        self._opt = opt
        self._params = _unwrap(list(params))

    def zero_grad(self, set_to_none: bool = False):
        """Reset gradients of the tracked parameters."""
        self._opt.zero_grad(self._params, set_to_none)

    def step(self) -> None:
        """Update the tracked parameters."""
        self._opt.step(self._params)


class SGD(_OptimizerWrapper):  # pragma: no cover - thin wrapper
    def __init__(
        self,
        params,
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        opt = _optim.SGD(
            lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        super().__init__(opt, params)


class Adam(_OptimizerWrapper):  # pragma: no cover - thin wrapper
    def __init__(
        self,
        params,
        lr: float,
        betas: tuple[float, float] | None = None,
        beta1: float | None = None,
        beta2: float | None = None,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if betas is not None and (beta1 is not None or beta2 is not None):
            raise TypeError("specify either betas tuple or beta1/beta2, not both")
        if betas is None:
            if beta1 is None and beta2 is None:
                betas = (0.9, 0.999)
            elif beta1 is not None and beta2 is not None:
                betas = (beta1, beta2)
            else:
                raise TypeError("both beta1 and beta2 must be provided")
        if not isinstance(betas, tuple) or len(betas) != 2:
            raise TypeError("betas must be a tuple of two floats")
        opt = _optim.Adam(
            lr,
            betas=betas,
            epsilon=epsilon,
            weight_decay=weight_decay,
        )
        super().__init__(opt, params)


class RMSprop(_OptimizerWrapper):  # pragma: no cover - thin wrapper
    def __init__(
        self,
        params,
        lr: float,
        alpha: float = 0.99,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = False,
    ) -> None:
        opt = _optim.RMSprop(
            lr,
            alpha=alpha,
            epsilon=epsilon,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered,
        )
        super().__init__(opt, params)


try:
    del annotations  # type: ignore  # noqa: F401
except Exception:
    pass
