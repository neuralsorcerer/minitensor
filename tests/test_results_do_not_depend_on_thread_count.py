# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The same input must give the same bits on any number of threads.

`par_out_chunks` states the rule the parallel kernels are built to: "Every
output chunk is computed independently of the others, so the partition cannot
affect the result... and unsafe to use where the partition decides how values
are *grouped* into an accumulation (see `reduce_along_dim0`, which fixes its
row bands to constants for exactly that reason)."

That is a real promise and an easy one to lose. Floating-point addition is not
associative, so any reduction that splits its input by `current_num_threads()`
and sums the pieces gives a different answer on a four-core machine than on a
one-core machine -- not a wrong answer, just a different one, which is worse,
because it means a training run is not reproducible and a regression test that
passes on a laptop fails in CI for no reason anyone can see. The banding
constants that prevent it look like tuning parameters, and the next person to
"optimise" them by deriving them from the thread count would break this
without failing anything else.

`RAYON_NUM_THREADS` is read once when rayon builds its pool, so each thread
count needs its own interpreter; these run as subprocesses for that reason.
Measured 33.9 ms against 9.3 ms for `relu` on 4096x4096 between one thread and
four, so the variable does reach the pool and these are not four runs of the
same thing.

`test_the_inputs_are_order_sensitive` keeps the rest honest: it shows that the
data really does add up differently in a different order, so agreement across
thread counts means the partitioning was controlled rather than the arithmetic
being too benign to notice.

This passes today, and is meant to. Every split that currently follows the
thread count is one that cannot change an answer -- `reduce_along_dim0` bands
*columns* by thread count, and each output still accumulates its rows in index
order, which its comment says in as many words. The value here is that the
property is now checked rather than only argued, so a later change to those
bands has something to fail.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

THREAD_COUNTS = [1, 2, 3, 4]

# Big enough to cross every parallel threshold in `ops::map` (the largest is
# 131072 elements), and spread across magnitudes so the order of summation
# changes the sum.
_RUNNER = """
import hashlib, warnings
warnings.simplefilter("ignore")
import numpy as np, minitensor as mt

rng = np.random.default_rng(12345)
shape = (512, 1024)
scale = (10.0 ** rng.integers(-6, 6, shape)).astype(np.float32)
values = (rng.standard_normal(shape).astype(np.float32) * scale).astype(np.float32)

wide = mt.Tensor(values, dtype="float32")
flat = mt.Tensor(values.reshape(-1).copy(), dtype="float32")
square = mt.Tensor(values[:256, :256].copy(), dtype="float32")
other = mt.Tensor(np.ascontiguousarray(values[:256, :256].T), dtype="float32")

def digest(tensor):
    return hashlib.sha256(np.asarray(tensor).tobytes()).hexdigest()

cases = {
    "sum_all": lambda: mt.sum(wide),
    "sum_dim0": lambda: mt.sum(wide, 0),
    "sum_dim1": lambda: mt.sum(wide, 1),
    "sum_flat": lambda: mt.sum(flat),
    "mean_all": lambda: mt.mean(wide),
    "mean_dim0": lambda: mt.mean(wide, 0),
    "var_dim0": lambda: wide.var(dim=0),
    "std_dim1": lambda: wide.std(dim=1),
    "prod_dim1": lambda: mt.prod(wide, 1),
    "cumsum_dim1": lambda: mt.cumsum(wide, 1),
    "nansum_all": lambda: mt.nansum(wide),
    "norm": lambda: mt.norm(wide),
    "logsumexp_dim1": lambda: mt.logsumexp(wide, 1),
    "softmax_dim1": lambda: mt.softmax(wide, 1),
    "max_dim0": lambda: mt.max(wide, 0)[0],
    "matmul": lambda: mt.matmul(square, other),
    "exp": lambda: mt.exp(wide),
    "tanh": lambda: mt.tanh(wide),
    "sigmoid": lambda: mt.sigmoid(wide),
}

lines = []
for name in sorted(cases):
    lines.append(name + " " + digest(cases[name]()))

# A backward pass too: gradient accumulation is where an order dependence is
# most likely to hide, and least likely to be noticed.
mt.manual_seed(0)
x = mt.Tensor(values[:128].copy(), dtype="float32", requires_grad=True)
w = mt.Tensor(np.ascontiguousarray(values.T[:, :64]), dtype="float32", requires_grad=True)
loss = mt.matmul(x, w).tanh().sum()
loss.backward()
lines.append("grad_x " + digest(x.grad))
lines.append("grad_w " + digest(w.grad))

print("\\n".join(lines))
"""


def _run_with(threads: int) -> dict[str, str]:
    # Inherit the environment and override one variable: a hand-built `env`
    # drops what Python needs to find its own DLLs on Windows.
    environment = dict(os.environ)
    environment["RAYON_NUM_THREADS"] = str(threads)
    finished = subprocess.run(
        [sys.executable, "-c", _RUNNER],
        capture_output=True,
        text=True,
        timeout=600,
        env=environment,
    )
    assert (
        finished.returncode == 0
    ), f"RAYON_NUM_THREADS={threads} failed:\n{finished.stderr[-2000:]}"
    out = {}
    for line in finished.stdout.strip().splitlines():
        name, _, value = line.partition(" ")
        out[name] = value
    assert out, f"RAYON_NUM_THREADS={threads} produced nothing"
    return out


@pytest.fixture(scope="module")
def digests() -> dict[int, dict[str, str]]:
    return {threads: _run_with(threads) for threads in THREAD_COUNTS}


def test_every_case_agrees_across_thread_counts(digests):
    reference = digests[THREAD_COUNTS[0]]
    disagreed = []
    for threads in THREAD_COUNTS[1:]:
        assert set(digests[threads]) == set(
            reference
        ), "the runs computed different cases"
        for name in sorted(reference):
            if digests[threads][name] != reference[name]:
                disagreed.append(
                    f"{name}: 1 thread {reference[name][:12]} vs "
                    f"{threads} threads {digests[threads][name][:12]}"
                )
    assert not disagreed, (
        "these gave different bits on a different number of threads, so a run "
        "is not reproducible:\n  " + "\n  ".join(disagreed)
    )


def test_the_inputs_are_order_sensitive():
    """Without this, agreement above could just mean the arithmetic is benign.

    The same float32 values added front-to-back and back-to-front must give
    different sums, or the test above proves nothing about partitioning.
    """
    rng = np.random.default_rng(12345)
    shape = (512, 1024)
    scale = (10.0 ** rng.integers(-6, 6, shape)).astype(np.float32)
    values = (rng.standard_normal(shape).astype(np.float32) * scale).astype(np.float32)
    flat = values.reshape(-1)

    forward = np.float32(0.0)
    for v in flat[:4096]:
        forward = np.float32(forward + v)
    backward = np.float32(0.0)
    for v in flat[:4096][::-1]:
        backward = np.float32(backward + v)

    assert forward != backward, (
        "this data sums the same in both directions, so it cannot detect a "
        "partition-dependent reduction"
    )
