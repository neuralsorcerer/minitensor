# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Measure every kernel against NumPy, so delegation is a measurement and not a guess.

`engine::ops::linalg::gemm_provider` sends dense products to the BLAS that
NumPy brought and leaves everything else on the engine's own kernels. That
split is only right if it is true here, on this machine, and it is not
portable: it depends on the host BLAS, the core count and the SIMD width. This
script is how the claim is re-taken.

Run it as ``python benchmarks/dispatch_bench.py`` for a readable table, or with
``--json out.json`` to keep the numbers. ``--filter`` narrows to one op family,
and ``--native`` pushes the delegation thresholds out of reach so the same
sweep measures the engine's own kernels -- running it both ways is how the two
paths are compared on one build.

Each case times the *kernel only*. Operands are built once, outside the timing
loop, in each library's own native container, so no conversion cost is charged
to either side. Timing is min-of-repeats rather than mean: the minimum is the
run least disturbed by the scheduler, and for a deterministic kernel the
disturbance is all that the spread measures.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

import minitensor as mt

# A case is only worth reporting if it ran long enough to be measured above the
# clock's own noise. `time.perf_counter` resolves to well under a microsecond on
# every platform we build for, so 50us of work leaves two orders of magnitude of
# headroom while keeping the whole sweep to a few seconds.
_MIN_BATCH_SECONDS = 5e-5


@dataclass(frozen=True)
class Case:
    """One (op, shape, dtype) measurement point."""

    family: str
    op: str
    shape: str
    dtype: str
    mt_fn: Callable[[], object]
    np_fn: Callable[[], object]
    flops: float | None = None


@dataclass
class Result:
    case: Case
    mt_seconds: float
    np_seconds: float

    @property
    def speedup(self) -> float:
        """How many times faster the Rust kernel is. Below 1.0 means NumPy wins."""
        return self.np_seconds / self.mt_seconds

    @property
    def verdict(self) -> str:
        # A margin, not a threshold at 1.0: a 5% difference is inside the
        # run-to-run spread of either library and is not a reason to move an op
        # across the dispatch boundary.
        if self.speedup >= 1.05:
            return "rust"
        if self.speedup <= 0.95:
            return "numpy"
        return "tie"


def _time(fn: Callable[[], object], *, repeats: int = 7) -> float:
    """Seconds per call, min over `repeats` batches, batch size chosen adaptively."""
    # Warm up: first call pays for lazy thread-pool spin-up in both libraries.
    fn()
    calls = 1
    while True:
        start = time.perf_counter()
        for _ in range(calls):
            fn()
        elapsed = time.perf_counter() - start
        if elapsed >= _MIN_BATCH_SECONDS or calls >= 1 << 20:
            break
        calls *= 8
    best = elapsed
    for _ in range(repeats - 1):
        start = time.perf_counter()
        for _ in range(calls):
            fn()
        best = min(best, time.perf_counter() - start)
    return best / calls


def _pair(shape: Sequence[int], dtype: str, seed: int = 0) -> tuple[np.ndarray, object]:
    """The same values as a NumPy array and as a minitensor tensor."""
    rng = np.random.default_rng(seed)
    if dtype in ("float32", "float64"):
        host = rng.standard_normal(shape).astype(dtype)
    else:
        host = rng.integers(-8, 8, size=shape).astype(dtype)
    return host, mt.Tensor.from_numpy(host)


def _gemm_cases(dtype: str) -> Iterable[Case]:
    shapes = [
        (16, 16, 16),
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        # The shapes an actual model spends its time in: a wide activation
        # against a square weight, and a single inference row.
        (1, 4096, 4096),
        (32, 4096, 4096),
        (2048, 512, 512),
    ]
    for m, k, n in shapes:
        a_np, a_mt = _pair((m, k), dtype, seed=1)
        b_np, b_mt = _pair((k, n), dtype, seed=2)
        yield Case(
            family="gemm",
            op="matmul",
            shape=f"{m}x{k}@{k}x{n}",
            dtype=dtype,
            mt_fn=lambda a=a_mt, b=b_mt: mt.matmul(a, b),
            np_fn=lambda a=a_np, b=b_np: np.matmul(a, b),
            flops=2.0 * m * k * n,
        )
    for batch, m, k, n in [(8, 128, 128, 128), (64, 64, 64, 64), (256, 32, 32, 32)]:
        a_np, a_mt = _pair((batch, m, k), dtype, seed=3)
        b_np, b_mt = _pair((batch, k, n), dtype, seed=4)
        yield Case(
            family="gemm",
            op="bmm",
            shape=f"{batch}x{m}x{k}@{batch}x{k}x{n}",
            dtype=dtype,
            mt_fn=lambda a=a_mt, b=b_mt: mt.bmm(a, b),
            np_fn=lambda a=a_np, b=b_np: np.matmul(a, b),
            flops=2.0 * batch * m * k * n,
        )


_ELEMENTWISE: list[tuple[str, Callable, Callable]] = [
    ("add", lambda a, b: a + b, lambda a, b: a + b),
    ("mul", lambda a, b: a * b, lambda a, b: a * b),
    ("div", lambda a, b: a / b, lambda a, b: a / b),
    ("maximum", lambda a, b: mt.maximum(a, b), lambda a, b: np.maximum(a, b)),
]

_UNARY: list[tuple[str, Callable, Callable]] = [
    ("exp", lambda a: mt.exp(a), lambda a: np.exp(a)),
    ("log", lambda a: mt.log(a), lambda a: np.log(a)),
    ("tanh", lambda a: mt.tanh(a), lambda a: np.tanh(a)),
    ("sqrt", lambda a: mt.sqrt(a), lambda a: np.sqrt(a)),
    ("sigmoid", lambda a: mt.sigmoid(a), lambda a: 1.0 / (1.0 + np.exp(-a))),
    ("relu", lambda a: mt.relu(a), lambda a: np.maximum(a, 0)),
    # Fused in Rust, three passes and two temporaries in NumPy. This is the
    # shape of op the dispatch policy keeps native on purpose.
    (
        "gelu",
        lambda a: mt.gelu(a),
        lambda a: a
        * 0.5
        * (1.0 + np.tanh(0.7978845608028654 * (a + 0.044715 * a * a * a))),
    ),
    ("softmax", lambda a: mt.softmax(a, -1), lambda a: _np_softmax(a)),
]

_REDUCTION: list[tuple[str, Callable, Callable]] = [
    ("sum", lambda a: mt.sum(a), lambda a: np.sum(a)),
    ("sum_axis0", lambda a: mt.sum(a, 0), lambda a: np.sum(a, 0)),
    ("mean", lambda a: mt.mean(a), lambda a: np.mean(a)),
    ("max", lambda a: mt.max(a), lambda a: np.max(a)),
    ("argmax", lambda a: mt.argmax(a), lambda a: np.argmax(a)),
    ("norm", lambda a: mt.norm(a), lambda a: np.linalg.norm(a.ravel())),
]


def _np_softmax(a: np.ndarray) -> np.ndarray:
    shifted = a - a.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


def _sized_cases(dtype: str) -> Iterable[Case]:
    # Sizes chosen to cross the interesting boundaries: below the parallel
    # threshold, at L2, and past last-level cache where the kernel is
    # memory-bound and only the number of passes over the data matters.
    shapes = [(1_000,), (100_000,), (1_000, 1_000), (4_000, 4_000)]
    for shape in shapes:
        label = "x".join(str(d) for d in shape)
        a_np, a_mt = _pair(shape, dtype, seed=5)
        b_np, b_mt = _pair(shape, dtype, seed=6)
        # Keep unary inputs positive so `log`/`sqrt` measure the arithmetic
        # rather than a NaN path.
        pos_np = np.abs(a_np) + 0.5
        pos_mt = mt.Tensor.from_numpy(pos_np)

        for name, f_mt, f_np in _ELEMENTWISE:
            yield Case(
                "elementwise",
                name,
                label,
                dtype,
                lambda f=f_mt, a=a_mt, b=b_mt: f(a, b),
                lambda f=f_np, a=a_np, b=b_np: f(a, b),
            )
        for name, f_mt, f_np in _UNARY:
            if name == "softmax" and len(shape) == 1:
                continue
            yield Case(
                "unary",
                name,
                label,
                dtype,
                lambda f=f_mt, a=pos_mt: f(a),
                lambda f=f_np, a=pos_np: f(a),
            )
        for name, f_mt, f_np in _REDUCTION:
            if name == "sum_axis0" and len(shape) == 1:
                continue
            yield Case(
                "reduction",
                name,
                label,
                dtype,
                lambda f=f_mt, a=a_mt: f(a),
                lambda f=f_np, a=a_np: f(a),
            )


def _shape_cases(dtype: str) -> Iterable[Case]:
    a_np, a_mt = _pair((1_000, 1_000), dtype, seed=7)
    yield Case(
        "shape",
        "transpose+contig",
        "1000x1000",
        dtype,
        lambda a=a_mt: mt.transpose(a, 0, 1),
        lambda a=a_np: np.ascontiguousarray(a.T),
    )
    yield Case(
        "shape",
        "reshape",
        "1000x1000",
        dtype,
        lambda a=a_mt: mt.reshape(a, (1_000_000,)),
        lambda a=a_np: a.reshape(1_000_000),
    )
    yield Case(
        "shape",
        "concat",
        "1000x1000",
        dtype,
        lambda a=a_mt: mt.cat([a, a], 0),
        lambda a=a_np: np.concatenate([a, a], 0),
    )


def collect_cases(dtypes: Sequence[str]) -> list[Case]:
    cases: list[Case] = []
    for dtype in dtypes:
        cases.extend(_gemm_cases(dtype))
        cases.extend(_sized_cases(dtype))
        cases.extend(_shape_cases(dtype))
    return cases


def run(cases: Iterable[Case]) -> list[Result]:
    results: list[Result] = []
    for case in cases:
        try:
            mt_s = _time(case.mt_fn)
            np_s = _time(case.np_fn)
        except Exception as exc:  # a missing op should not abort the sweep
            print(
                f"  skipped {case.op} {case.shape} {case.dtype}: {exc}", file=sys.stderr
            )
            continue
        results.append(Result(case, mt_s, np_s))
    return results


def _fmt_time(seconds: float) -> str:
    if seconds < 1e-6:
        return f"{seconds * 1e9:7.1f}ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:7.1f}us"
    return f"{seconds * 1e3:7.2f}ms"


def report(results: Sequence[Result]) -> None:
    header = f"{'family':<12} {'op':<18} {'shape':<22} {'dtype':<8} {'rust':>10} {'numpy':>10} {'x':>7}  verdict"
    print(header)
    print("-" * len(header))
    for r in results:
        c = r.case
        print(
            f"{c.family:<12} {c.op:<18} {c.shape:<22} {c.dtype:<8} "
            f"{_fmt_time(r.mt_seconds):>10} {_fmt_time(r.np_seconds):>10} "
            f"{r.speedup:>7.2f}  {r.verdict}"
        )

    print()
    print("Per-family summary (geometric mean of rust/numpy speedup):")
    families: dict[str, list[float]] = {}
    for r in results:
        families.setdefault(r.case.family, []).append(r.speedup)
    for family, speedups in sorted(families.items()):
        geo = statistics.geometric_mean(speedups)
        wins = sum(1 for s in speedups if s >= 1.05)
        losses = sum(1 for s in speedups if s <= 0.95)
        print(
            f"  {family:<12} geomean {geo:5.2f}x   rust wins {wins:3d}   numpy wins {losses:3d}   ties {len(speedups) - wins - losses:3d}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dtype",
        action="append",
        choices=["float32", "float64"],
        help="repeatable; defaults to float32 and float64",
    )
    parser.add_argument(
        "--filter", help="only run families/ops whose name contains this"
    )
    parser.add_argument("--json", help="write raw measurements here")
    parser.add_argument(
        "--native",
        action="store_true",
        help="push the delegation thresholds out of reach, so every product "
        "runs on the engine's own kernel; this is how the two paths are "
        "compared on one build",
    )
    args = parser.parse_args(argv)

    if args.native:
        mt._core.dispatch.set_gemm_thresholds(2**62, 2**62)

    dtypes = args.dtype or ["float32", "float64"]
    cases = collect_cases(dtypes)
    if args.filter:
        needle = args.filter.lower()
        cases = [
            c for c in cases if needle in c.family.lower() or needle in c.op.lower()
        ]

    provider = mt._core.dispatch.gemm_provider_installed()
    min_flops, min_k = mt._core.dispatch.gemm_thresholds()
    print(
        f"numpy {np.__version__}  minitensor {mt.__version__}  {len(cases)} cases  "
        f"provider={'numpy' if provider else 'none'}  "
        f"min_flops={min_flops} min_k={min_k}",
        file=sys.stderr,
    )
    results = run(cases)
    report(results)

    if args.json:
        payload = [
            {
                "family": r.case.family,
                "op": r.case.op,
                "shape": r.case.shape,
                "dtype": r.case.dtype,
                "rust_seconds": r.mt_seconds,
                "numpy_seconds": r.np_seconds,
                "speedup": r.speedup,
                "verdict": r.verdict,
                "gflops_rust": (
                    (r.case.flops / r.mt_seconds / 1e9) if r.case.flops else None
                ),
                "gflops_numpy": (
                    (r.case.flops / r.np_seconds / 1e9) if r.case.flops else None
                ),
            }
            for r in results
        ]
        with open(args.json, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nwrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
