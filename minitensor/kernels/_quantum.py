# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""State-vector evolution and measurement, differentiable end to end.

A state over `q` qubits is `2**q` complex amplitudes. Applying a single-qubit
gate touches every amplitude exactly once, in pairs — the two whose indices
differ only in the target bit. That is a butterfly, the same access pattern as
one radix-2 stage of an FFT, and there is no elementwise vocabulary for it: the
composed form reshapes to `(2**(q-t-1), 2, 2**t)`, slices, multiplies and
concatenates, which moves numbers between two halves of one buffer through
several full copies of it.

Written directly it is one pass with no intermediate, and the pairs are
disjoint so it parallelises exactly.

**Complex numbers.** The engine stores real tensors, so a state is shaped
`(..., 2**q, 2)` with the last axis holding `(real, imaginary)`. Interleaved
rather than split into two planes, because both halves of an amplitude are
always used together and interleaving puts them in one cache line. Leading axes
are a batch, so an ensemble of states evolves in one call.

**Why the derivatives are cheap.** Gate application is linear in the state, so
its derivative is the adjoint gate applied to the incoming gradient — the same
butterfly with `U†`, and nothing from the forward pass needs saving. A long
circuit therefore costs no tape memory beyond its states. Measurement is
quadratic, so its derivative is `2 conj(amplitude)` weighted by the incoming
gradient: also one pass.
"""

from __future__ import annotations

from typing import Any, Sequence

from .. import _core as _C

__all__ = ["apply_gate", "probabilities", "prefix_trace", "expect_z"]


def _kernels() -> Any:
    """The native `quantum` kernels, or a clear account of their absence.

    Resolved per call rather than bound at import. Binding it eagerly would
    make importing `minitensor` fail outright on a build whose extension lacks
    the submodule, and the package is deliberately built so that a partial
    build degrades to a missing attribute rather than a failed import -- see
    how `plugins` and `serialization` are handled in `__init__`.
    """
    domains = getattr(_C, "domains", None)
    module = getattr(domains, "quantum", None) if domains is not None else None
    if module is None:
        raise RuntimeError(
            "this build of minitensor._core has no domains.quantum submodule"
        )
    return module


def apply_gate(state: Any, gate: str | Sequence[float], qubit: int) -> Any:
    """Apply a single-qubit gate, counting qubits from the low index bit.

    `state` is `(..., 2**q, 2)` and the result has the same shape. `gate` is
    either a name — `"h"`, `"x"`, `"z"` — or eight reals giving an arbitrary
    `[[a, b], [c, d]]` as `[a.real, a.imag, b.real, b.imag, c.real, c.imag,
    d.real, d.imag]`.

    The gradient flows to `state`. The gate itself is a constant here: a
    trainable gate is a different operation, since a parameterised rotation is
    better differentiated through its angle than through its four entries.
    """
    return _kernels().apply_gate_1q(state, gate, qubit)


def probabilities(state: Any) -> Any:
    """Born-rule probabilities, `|amplitude|**2` per basis state.

    `state` is `(..., 2**q, 2)`; the result is `(..., 2**q)`.
    """
    return _kernels().probabilities(state)


def prefix_trace(state: Any, keep: int) -> Any:
    """Marginal probabilities of the first `keep` qubits.

    Tracing out the remaining qubits of a pure state means summing
    `|amplitude|**2` over every configuration of the traced ones. With the
    prefix in the high bits of the index, the amplitudes contributing to one
    prefix outcome form a contiguous run — which is why the prefix is the
    subset worth having a kernel for, and why this is a strided sum with
    perfect locality rather than a gather.

    `state` is `(..., 2**q, 2)`; the result is `(..., 2**keep)`.
    """
    return _kernels().prefix_trace(state, keep)


def expect_z(state: Any, qubit: int) -> Any:
    """`<Z_q>`, the Pauli-Z expectation on one qubit.

    `+1` when the qubit is certainly `|0>`, `-1` when certainly `|1>`, and the
    probability difference in between. `state` is `(..., 2**q, 2)`; the result
    has the batch shape.
    """
    return _kernels().expect_z(state, qubit)
