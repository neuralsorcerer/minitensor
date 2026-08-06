# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parallel reductions must not depend on how rayon schedules the work.

`par_iter().sum()` and `par_chunks(n).map(..).sum()` both look deterministic and
are not: chunking fixes the accumulation order inside a chunk, but `sum()` on a
parallel iterator combines the chunk partials in split-and-steal order. Floating
point addition is not associative, so the total moved between runs -- `nansum`
over 10^7 values produced ten distinct results in twelve calls, and `sum` was
intermittently unstable. A seeded training run that reports a different loss
each time is not reproducible.

The affected paths were `sum`, `nansum`, `nanmean`, the cross-entropy row sums,
the gradient norm behind `clip_grad_norm_`, and the dot-product path of
`matmul`. All now collect their partials in index order and combine them with a
fixed binary tree, which is both stable and as accurate as the tree rayon was
using (a sequential fold would be stable too, but grows error linearly in the
chunk count instead of logarithmically).

Repeated calls in one process are a weak check -- the original bug only showed
up sometimes. `test_reductions_are_invariant_across_thread_counts` is the real
one: it re-runs under different `RAYON_NUM_THREADS` values, where a
scheduling-dependent result cannot hide.
"""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import minitensor as mt

_N = 1 << 20


def _sample(with_nans=False):
    rng = np.random.default_rng(2)
    array = rng.standard_normal(_N).astype(np.float32)
    if with_nans:
        array[::997] = np.nan
    return array


@pytest.mark.parametrize(
    "name, with_nans",
    [("sum", False), ("nansum", True), ("nanmean", True), ("mean", False)],
)
def test_repeated_calls_return_identical_bits(name, with_nans):
    tensor = mt.from_numpy(_sample(with_nans))
    results = {getattr(tensor, name)().numpy().tobytes() for _ in range(16)}
    assert len(results) == 1, f"{name} produced {len(results)} distinct results"


def test_gradient_norm_is_stable():
    # Feeds `clip_grad_norm_`, so drift here rescales every gradient by a
    # slightly different factor from run to run.
    rng = np.random.default_rng(5)
    values = rng.standard_normal(_N).astype(np.float32)
    norms = set()
    for _ in range(16):
        param = mt.Tensor(values.copy(), dtype="float32", requires_grad=True)
        (param * param).sum().backward()
        norms.add(mt.nn.grad_norm([param]))
        mt.clear_autograd_graph()
    assert len(norms) == 1, f"grad_norm produced {len(norms)} distinct results"


_CHILD = textwrap.dedent("""
    import numpy as np, minitensor as mt
    rng = np.random.default_rng(2)
    a = rng.standard_normal(1 << 20).astype(np.float32)
    a[::997] = np.nan
    t = mt.from_numpy(a)
    b = mt.from_numpy(rng.standard_normal(1 << 18).astype(np.float32))
    c = mt.from_numpy(rng.standard_normal(1 << 18).astype(np.float32))
    print(
        float(mt.from_numpy(np.nan_to_num(a)).sum().numpy()).hex(),
        float(t.nansum().numpy()).hex(),
        float(t.nanmean().numpy()).hex(),
        float((b * c).sum().numpy()).hex(),
    )
    """)


@pytest.mark.parametrize("threads", ["1", "2", "4", "8"])
def test_reductions_are_invariant_across_thread_counts(threads):
    # The decisive check: a result that depends on rayon's split-and-steal
    # decisions cannot survive being run at a different thread count.
    def run(n):
        env = dict(os.environ, RAYON_NUM_THREADS=n)
        env["PYTHONPATH"] = os.pathsep.join(
            [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
            + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )
        out = subprocess.run(
            [sys.executable, "-c", _CHILD],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        return out.stdout.strip()

    assert run(threads) == run("1")
