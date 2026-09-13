# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Black-Scholes pricing, where the derivative is the thing being asked for.

Options pricing is the one place in this package where automatic
differentiation and the domain want exactly the same object. The Greeks *are*
the partial derivatives of the price: delta is `dV/dS`, vega is `dV/dsigma`,
rho is `dV/dr`. A desk computes them constantly, and the usual way to get them
out of a tensor library is to write the price as a chain of twenty operations
and let the tape differentiate it.

That works and it is wasteful twice over. Forward, the twenty intermediates are
allocated and walked and none of them is wanted. Backward, each of the five
partials becomes its own chain of gradient kernels, and every one of them
recomputes `d1` — the quantity all five share.

So this is a kernel: one pass for the price, and a backward that evaluates
`d1`, `d2` and the discount factor once and reads all five Greeks off them.

    d1 = [ln(S/K) + (r + sigma^2/2) T] / (sigma sqrt(T))
    d2 = d1 - sigma sqrt(T)
    call = S Phi(d1) - K e^{-rT} Phi(d2)
    put  = K e^{-rT} Phi(-d2) - S Phi(-d1)

Unlike `soft_assignment`, the fusion here has to be native: twenty elementwise
operations composed in Python would be twenty allocations whatever the backward
looked like.
"""

from __future__ import annotations

from typing import Any

from .. import _core as _C

__all__ = ["black_scholes", "implied_volatility"]


def _kernels() -> Any:
    """The native `finance` kernels, or a clear account of their absence.

    Resolved per call rather than bound at import. Binding it eagerly would
    make importing `minitensor` fail outright on a build whose extension lacks
    the submodule, and the package is deliberately built so that a partial
    build degrades to a missing attribute rather than a failed import -- see
    how `plugins` and `serialization` are handled in `__init__`.
    """
    domains = getattr(_C, "domains", None)
    module = getattr(domains, "finance", None) if domains is not None else None
    if module is None:
        raise RuntimeError(
            "this build of minitensor._core has no domains.finance submodule"
        )
    return module


def black_scholes(
    spot: Any,
    strike: Any,
    rate: Any,
    vol: Any,
    time: Any,
    kind: str = "call",
) -> Any:
    """Price European options elementwise, differentiably.

    All five arguments are tensors of the same shape and dtype; broadcasting is
    the caller's to arrange with `expand`, because guessing which of five
    operands was meant to be the scalar is how a book quietly gets priced
    against the wrong strike.

    `kind` is `"call"` or `"put"`. The gradient with respect to each argument
    is the corresponding Greek, so `spot.grad` after a backward pass is delta,
    `vol.grad` is vega, and `rate.grad` is rho.

    At `sigma sqrt(T) == 0` — an expired option, or a zero-volatility
    assumption — the formula is `0 * inf` and the limit is the intrinsic value,
    which is what comes back. Its derivative is a step, and at exactly
    `S == K e^{-rT}` there is none; a subgradient of zero is returned there,
    the same convention `relu` uses at the origin and for the same reason.
    """
    return _kernels().black_scholes(spot, strike, rate, vol, time, kind)


def implied_volatility(
    price: Any,
    spot: Any,
    strike: Any,
    rate: Any,
    time: Any,
    kind: str = "call",
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> Any:
    """The volatility that reproduces an observed price.

    Inverting Black-Scholes has no closed form, so this iterates: Newton's
    method on vega, which is already computed alongside the price, bracketed by
    a bisection so that a step leaving the bracket is replaced by the midpoint
    rather than diverging. Vega vanishes deep in and out of the money, which is
    exactly where unbracketed Newton fails.

    NaN where no volatility reproduces the price — a quote below intrinsic
    value, or above the spot — because there is no answer and a clamped bound
    would look like one.

    The opposite case is the one to watch, and it is silent. `tolerance` bounds
    the *price*, so the volatility is pinned only to about `tolerance / vega`,
    and where vega underflows it is not pinned at all: a deep in-the-money call
    one month out is worth its intrinsic value to the last bit of a double. At
    `S=300, K=100, r=0.05, T=0.1` the volatilities 0.05, 0.1 and 0.2 all price
    to exactly 200.49875208073178, with vega at 0.2 equal to 1.66e-65, so the
    first iterate already meets the tolerance and 0.2 — the value the search
    starts from — is what comes back. It is a correct inverse and a useless one,
    and it does not look useless.

    No vega threshold is imposed to catch that, because every threshold also
    refuses quotes that are legitimately informative. To check it yourself,
    evaluate `black_scholes` at the recovered volatility with `requires_grad` on
    `vol` and compare `tolerance / vega` against the precision you need.

    The result does not carry a gradient, and does not need to: by the implicit
    function theorem `dsigma/dprice = 1 / vega`, so a caller who wants that
    gradient gets it exactly by evaluating `black_scholes` at the recovered
    volatility, which is a recorded operation. Differentiating the Newton
    iteration instead would differentiate the solver rather than the solution.
    """
    return _kernels().implied_volatility(
        price, spot, strike, rate, time, kind, tolerance, max_iterations
    )
