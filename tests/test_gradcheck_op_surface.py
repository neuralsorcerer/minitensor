# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Every no-argument unary op, gradchecked in float64 at a tight tolerance.

`test_gradcheck_float64_surface` checks about forty operations this way and
explains why float64 is what makes the check worth anything. The rest of the
surface is covered by `test_gradcheck_differential`, at `rtol=3e-2` -- loose
enough that a gradient wrong by a couple of percent passes it. This closes that
gap for the ops reachable without arguments: it finds them by walking
`dir(Tensor)`, so an operation added later is covered the day it appears rather
than the day somebody remembers to list it.

The whole sweep runs in about a tenth of a second, which is why it can afford to
be exhaustive rather than a sample.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

#: Three input regimes, so a domain-restricted op gets values it accepts.
#: Values avoid 0, 1 and the integers, where several of these ops have kinks or
#: poles that a central difference straddles rather than measures.
REGIMES = {
    "positive": [0.37, 0.81, 1.43, 2.17],
    "unit": [-0.71, -0.24, 0.31, 0.66],
    "greater_than_one": [1.28, 1.93, 2.61, 3.44],
}

#: Methods that are not a differentiable map from this tensor to another:
#: accessors, converters, and the in-place mutators.
NOT_A_UNARY_OP = {
    "astype",
    "backward",
    "contiguous",
    "copy_",
    "cpu",
    "cuda",
    "detach",
    "detach_",
    "dim",
    "element_size",
    "fill_",
    "get_device",
    "has_grad",
    "is_complex",
    "is_contiguous",
    "is_floating_point",
    "item",
    "ndimension",
    "nelement",
    "numel",
    "numpy",
    "requires_grad_",
    "retain_grad",
    "size",
    "to",
    "tolist",
    "zero_grad",
}

#: A floor on how many ops the walk must find. Without it a change that breaks
#: discovery -- a rename, a signature change -- turns this file into a test that
#: passes by checking nothing.
MINIMUM_COVERED = 90


def _tensor(values):
    return mt.Tensor(
        np.ascontiguousarray(np.asarray(values, np.float64)),
        dtype="float64",
        requires_grad=True,
    )


def _unary_ops():
    for name in sorted(dir(mt.Tensor)):
        if name.startswith("_") or name in NOT_A_UNARY_OP:
            continue
        if callable(getattr(mt.Tensor, name, None)):
            yield name


def _usable_regime(name):
    """The first regime this op is differentiable and finite on, or None.

    Two things disqualify a regime, and neither is a fault in the operation:

    An out-of-domain input does not raise -- `acos(1.43)` returns NaN rather
    than complaining -- so a regime only counts when the forward is finite
    throughout. Gradchecking a NaN forward reports a mismatch that is really the
    test's own bad input.

    An output that does not require grad means the engine has declared the
    operation non-differentiable and cut it from the graph: `floor`, `ceil`,
    `round`, `trunc`, `sign`, `sgn` and `imag` are zero-gradient almost
    everywhere, and this library refuses a backward through them rather than
    handing back the zeros. Asking for a gradient there is asking for something
    the operation says it does not have. Detected rather than listed, so an op
    that changes its mind in either direction is followed automatically.
    """
    for regime, values in REGIMES.items():
        tensor = _tensor(values)
        try:
            result = getattr(tensor, name)()
            if not isinstance(result, mt.Tensor):
                return None
            if str(result.dtype) != "float64":
                return None
            if not result.requires_grad:
                return None
            if np.all(np.isfinite(np.asarray(result, np.float64))):
                return regime
        except Exception:
            continue
        finally:
            mt.clear_autograd_graph()
    return None


OPS = [(name, _usable_regime(name)) for name in _unary_ops()]
COVERED = [(name, regime) for name, regime in OPS if regime is not None]


def test_the_walk_finds_the_operations():
    """Guard against discovery silently finding nothing."""
    assert len(COVERED) >= MINIMUM_COVERED, (
        f"only {len(COVERED)} unary ops discovered, expected at least "
        f"{MINIMUM_COVERED}; has the tensor surface been renamed?"
    )


@pytest.mark.parametrize("name,regime", COVERED, ids=[n for n, _ in COVERED])
def test_unary_gradient_matches_finite_differences(name, regime):
    tensor = _tensor(REGIMES[regime])
    try:
        assert mt.gradcheck(
            lambda t: getattr(t, name)().sum(), (tensor,), atol=1e-6, rtol=1e-5
        )
    finally:
        mt.clear_autograd_graph()
