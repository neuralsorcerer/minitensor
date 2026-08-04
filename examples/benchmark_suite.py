# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmarks for the paths that dominate a training step.

Run it twice -- once per build -- to compare two versions of the library:

    python examples/benchmark_suite.py --json before.json
    python examples/benchmark_suite.py --json after.json
    python examples/benchmark_suite.py --compare before.json after.json

On the measurement, which is easy to get wrong here:

* Every case is timed in several **rounds**, and the rounds are **interleaved**
  across cases rather than run one case to completion. Timing case A fully and
  then case B lets any drift over the run -- clock scaling, allocator state,
  page cache -- land entirely on one of them. Measured sequentially on this
  suite, disabling a layer's bias appeared to save 5 ms of an 11 ms step;
  interleaved, the same comparison is flat.

* The reported figure is the **median** of per-round minima. A plain minimum
  rewards a single lucky run, and a mean is dragged around by scheduler
  outliers. Even so, comparing a build against *itself* on a 4-core machine
  spread individual cases by 13% at `--rounds 2` -- one of them by 68% -- and
  narrowed to 5.6% at `--rounds 6`, with nothing flagged. So the honest first
  step with a new machine is an A/A comparison: whatever it reports is the
  floor, and `--rounds` is the dial that lowers it.

* The thread pool is woken before anything is timed. Rayon's first parallel
  call pays for spinning up its workers, which lands on whichever case runs
  first -- worth ~0.5 ms, enough to invent a difference that is not there.

* `backward` cases clear the autograd graph each iteration. Without that,
  gradients accumulate into the parameters and every iteration after the first
  also pays for a full-size add -- which is real work, but not the work being
  measured.

* Conv and dense backward cases give their input `requires_grad`, as every
  layer but the first in a network has. The gradient-with-respect-to-input path
  is skipped otherwise, and it is the larger half of a conv backward.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

# Running a script by path puts *its* directory on `sys.path`, not the working
# directory, so `python examples/benchmark_suite.py` from a source checkout
# cannot see the package next to it unless the repository root is added.
if __package__ in (None, "") and "minitensor" not in sys.modules:
    _root = Path(__file__).resolve().parent.parent
    if (_root / "minitensor" / "__init__.py").exists():
        sys.path.insert(0, str(_root))

import minitensor as mt
from minitensor import nn, optim

# Ratio outside which a difference is worth reporting. On a 4-core machine,
# comparing a build against itself spread cases by 13% at `--rounds 2` and by
# 5.6% at `--rounds 6`, so this sits above the latter and well under the
# former: at low round counts it will still report some of its own noise.
# Calibrate with a same-build comparison before trusting a result.
NOISE_BAND = 1.15


def _clear() -> None:
    mt.clear_autograd_graph()


def _wake_thread_pool() -> None:
    warm = mt.randn(512, 512)
    for _ in range(30):
        mt.matmul(warm, warm)


class Case:
    """One named measurement."""

    def __init__(self, name: str, group: str, run: Callable[[], None], iters: int = 10):
        self.name = name
        self.group = group
        self.run = run
        self.iters = iters

    def round_min(self) -> float:
        best = float("inf")
        for _ in range(self.iters):
            start = time.perf_counter()
            self.run()
            best = min(best, (time.perf_counter() - start) * 1000.0)
        return best


def build_cases() -> list[Case]:
    cases: list[Case] = []

    def add(name, group, run, iters=10):
        cases.append(Case(name, group, run, iters))

    # --- raw GEMM: the shapes where splitting the work matters most ---------
    for m, k, n in [
        (16, 1024, 1024),
        (64, 1024, 1024),
        (1024, 1024, 1024),
        (4096, 1024, 16),
    ]:
        a, b = mt.randn(m, k), mt.randn(k, n)
        add(f"matmul {m}x{k}x{n}", "gemm", lambda a=a, b=b: mt.matmul(a, b), iters=15)

    # --- dense layers: forward, and a full training step --------------------
    for batch in (16, 64, 256):
        layer = nn.DenseLayer(1024, 1024)
        x = mt.randn(batch, 1024).requires_grad_(True)

        def fwd(layer=layer, x=x):
            with mt.no_grad():
                layer(x)

        def fwd_bwd(layer=layer, x=x):
            _clear()
            mt.sum(layer(x)).backward()

        add(f"dense 1024x1024 fwd b={batch}", "dense", fwd, iters=15)
        add(f"dense 1024x1024 fwd+bwd b={batch}", "dense", fwd_bwd, iters=10)

    mlp = nn.Sequential(
        [
            nn.DenseLayer(1024, 1024),
            nn.ReLU(),
            nn.DenseLayer(1024, 1024),
            nn.ReLU(),
            nn.DenseLayer(1024, 512),
        ]
    )
    opt = optim.Adam(mlp.parameters(), lr=1e-3)
    for batch in (16, 256):
        x, y = mt.randn(batch, 1024), mt.randn(batch, 512)

        def step(x=x, y=y):
            opt.zero_grad()
            nn.mse_loss(mlp(x), y).backward()
            opt.step()

        add(f"MLP 3-layer Adam step b={batch}", "mlp", step, iters=8)

    # --- conv: the input must require a gradient for grad_input to run ------
    for label, (n, cin, size, cout) in {
        "3->32 64x64 N8": (8, 3, 64, 32),
        "64->128 32x32 N8": (8, 64, 32, 128),
        "256->512 16x16 N2": (2, 256, 16, 512),
    }.items():
        conv = nn.Conv2d(cin, cout, 3, padding=1)
        x = mt.randn(n, cin, size, size).requires_grad_(True)

        def cfwd(conv=conv, x=x):
            with mt.no_grad():
                conv(x)

        def cfb(conv=conv, x=x):
            _clear()
            mt.sum(conv(x)).backward()

        add(f"conv2d {label} fwd", "conv", cfwd, iters=8)
        add(f"conv2d {label} fwd+bwd", "conv", cfb, iters=6)

    # --- recurrent and attention -------------------------------------------
    seq = mt.randn(16, 64, 128)
    for label, layer in [("LSTM", nn.LSTM(128, 256)), ("GRU", nn.GRU(128, 256))]:

        def rfwd(layer=layer):
            with mt.no_grad():
                layer(seq)

        def rfb(layer=layer):
            _clear()
            mt.sum(layer(seq)).backward()

        add(f"{label} 128->256 T64 fwd", "recurrent", rfwd, iters=6)
        add(f"{label} 128->256 T64 fwd+bwd", "recurrent", rfb, iters=4)

    attn_in = mt.randn(8, 128, 256)
    mha = nn.MultiheadAttention(256, 8)

    def afwd():
        with mt.no_grad():
            mha(attn_in)

    def afb():
        _clear()
        mt.sum(mha(attn_in)).backward()

    add("MHA E256 H8 T128 fwd", "attention", afwd, iters=8)
    add("MHA E256 H8 T128 fwd+bwd", "attention", afb, iters=5)

    # --- elementwise, normalisation, loss, indexing -------------------------
    big = mt.randn(1 << 21)
    for name, fn in [
        ("tanh", mt.tanh),
        ("gelu", mt.gelu),
        ("sigmoid", mt.sigmoid),
        ("exp", mt.exp),
        ("softplus", mt.softplus),
    ]:
        add(f"{name} 2M f32", "elementwise", lambda fn=fn: fn(big), iters=10)

    rows = mt.randn(4096, 1024)
    add("softmax (4096,1024)", "elementwise", lambda: mt.softmax(rows, -1), iters=10)
    ln, rms = nn.LayerNorm([1024]), nn.RMSNorm([1024])
    add("LayerNorm (4096,1024)", "norm", lambda: ln(rows), iters=10)
    add("RMSNorm (4096,1024)", "norm", lambda: rms(rows), iters=10)

    logits = mt.randn(4096, 1000).requires_grad_(True)
    targets = mt.Tensor(
        np.random.randint(0, 1000, 4096).astype(np.int64), dtype="int64"
    )

    def ce():
        _clear()
        nn.cross_entropy(logits, targets).backward()

    add("cross_entropy 4096x1000 fwd+bwd", "loss", ce, iters=8)

    src = mt.randn(4096, 256)
    add("slice [100:2100]", "indexing", lambda: src[100:2100], iters=20)
    add("slice [:, :128]", "indexing", lambda: src[:, 0:128], iters=20)
    add("slice [::2, ::2]", "indexing", lambda: src[::2, ::2], iters=20)

    return cases


def measure(cases: list[Case], rounds: int) -> dict[str, float]:
    for case in cases:  # warm every case once, outside the timing
        case.run()
    samples: dict[str, list[float]] = {case.name: [] for case in cases}
    for _ in range(rounds):
        for case in cases:  # interleaved: drift is shared, not attributed
            samples[case.name].append(case.round_min())
    return {name: statistics.median(values) for name, values in samples.items()}


def report(cases: list[Case], results: dict[str, float]) -> None:
    group = None
    for case in cases:
        if case.group != group:
            group = case.group
            print(f"\n{group}")
        print(f"  {case.name:38s} {results[case.name]:9.3f} ms")


def compare(before_path: str, after_path: str) -> None:
    with open(before_path) as handle:
        before = json.load(handle)
    with open(after_path) as handle:
        after = json.load(handle)

    shared = [name for name in after if name in before]
    print(f"{'case':40s} {'before':>10s} {'after':>10s} {'change':>9s}")
    flagged = 0
    for name in shared:
        old, new = before[name], after[name]
        ratio = old / new if new else float("inf")
        significant = not (1 / NOISE_BAND) < ratio < NOISE_BAND
        flagged += significant
        note = "" if not significant else ("  faster" if ratio > 1 else "  SLOWER")
        print(f"{name:40s} {old:9.3f}ms {new:9.3f}ms {ratio:8.2f}x{note}")
    print(
        f"\n{flagged} of {len(shared)} cases moved by more than "
        f"{(NOISE_BAND - 1) * 100:.0f}%. Comparing a build against *itself* is how "
        "to check\nthat threshold against this machine: whatever that run flags "
        "is the noise floor,\nnot a change. Raise --rounds until it flags nothing."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rounds", type=int, default=5, help="interleaved timing rounds"
    )
    parser.add_argument("--json", help="write results here for a later --compare")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    mt.manual_seed(0)
    np.random.seed(0)
    _wake_thread_pool()

    cases = build_cases()
    results = measure(cases, args.rounds)
    report(cases, results)

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":  # pragma: no cover - example script
    main()
