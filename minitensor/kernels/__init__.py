# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Domain kernels: operations a general array library has no reason to carry.

Each of these is an operation a specialised caller cannot do without and would
otherwise write as a chain of eight or twenty tensor calls, where every link
allocates a full intermediate that nothing wants. Written as one kernel it
allocates its answer and nothing else, and its derivative is a closed form
rather than a chain of recorded nodes.

They come in two kinds, and the difference is a judgement about where the
saving is:

* `soft_assignment` is an `autograd.Function` written in Python. Its forward is
  two ordinary calls and the whole saving is in the backward, which is written
  over the `[n, k]` responsibilities rather than the `[n, k, d]` differences.
  It is here as much to be read as to be used — a worked example of the
  extension path, in the same handful of lines a user would write for their
  own.
* The finance and quantum kernels are native. Black-Scholes is twenty
  elementwise operations whose intermediates are never wanted and whose five
  partials all share `d1`; a quantum gate is a strided butterfly that no
  elementwise vocabulary expresses. Neither fusion survives being composed in
  Python, whatever the backward looks like.
"""

from __future__ import annotations

from ._clustering import soft_assignment
from ._finance import black_scholes, implied_volatility
from ._quantum import apply_gate, expect_z, prefix_trace, probabilities

__all__ = [
    "soft_assignment",
    "black_scholes",
    "implied_volatility",
    "apply_gate",
    "probabilities",
    "prefix_trace",
    "expect_z",
]
