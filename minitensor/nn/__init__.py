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
API robust we now populate the namespace dynamically – only symbols that exist
in the core are exported.  This mirrors the behaviour of optional components in
other numerical libraries and allows parts of the library (such as the tensor
API) to be used independently of the neural network layers.
"""

from typing import List

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

    def _export(name: str) -> None:
        """Export ``name`` from the Rust ``nn`` module if available."""
        obj = getattr(_nn, name, None)
        if obj is not None:
            globals()[name] = obj
            __all__.append(name)

    for _name in [
        # Base module / layers
        "Module",
        "DenseLayer",
        "Conv2d",
        "BatchNorm1d",
        "BatchNorm2d",
        "Dropout",
        "Sequential",
        # Activations
        "ReLU",
        "Sigmoid",
        "Tanh",
        "Softmax",
        "LeakyReLU",
        "ELU",
        "GELU",
        # Loss functions
        "MSELoss",
        "MAELoss",
        "HuberLoss",
        "CrossEntropyLoss",
        "BCELoss",
        "FocalLoss",
    ]:
        _export(_name)
# When the core is missing we simply expose an empty namespace so that importing
# :mod:`minitensor.nn` never raises an exception.
