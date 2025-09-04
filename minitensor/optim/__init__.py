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

    def _export(name: str) -> None:
        obj = getattr(_optim, name, None)
        if obj is not None:
            globals()[name] = obj
            __all__.append(name)

    for _name in ["Optimizer", "SGD", "Adam", "RMSprop"]:
        _export(_name)
# If the core is missing nothing is exported which keeps imports harmless.
