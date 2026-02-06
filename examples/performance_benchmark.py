# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Performance benchmark for Minitensor.

This script compares the execution time of a matrix multiplication operation
against PyTorch and TensorFlow when those libraries are available. The script is
safe to run even if the optional dependencies are missing; in that case the
corresponding benchmarks are skipped.
"""

from __future__ import annotations

import timeit


def _benchmark(lib_name: str, setup: str, stmt: str, number: int = 10):
    """Utility helper to run a benchmark with ``timeit``.

    Args:
        lib_name: Name of the library being benchmarked (for display only).
        setup: Setup string executed once before timing.
        stmt: Statement string to be timed.
        number: Number of executions.

    Returns:
        float | None: Execution time in seconds or ``None`` if the benchmark
        cannot be executed.
    """

    try:
        return timeit.timeit(stmt, setup=setup, number=number)
    except Exception as exc:  # pragma: no cover - best effort only
        print(f"Skipping {lib_name} benchmark: {exc}")
        return None


def main():  # pragma: no cover - example script
    size = 256

    setup_mt = (
        "import minitensor as mt, numpy as np\n"
        f"a=mt.randn({size},{size})\n"
        f"b=mt.randn({size},{size})"
    )
    stmt_mt = "a.matmul(b).sum().item()"
    mt_time = _benchmark("Minitensor", setup_mt, stmt_mt)

    setup_torch = (
        "import torch\n"
        f"a=torch.randn({size},{size})\n"
        f"b=torch.randn({size},{size})"
    )
    stmt_torch = "(a @ b).sum().item()"
    torch_time = _benchmark("PyTorch", setup_torch, stmt_torch)

    setup_tf = (
        "import tensorflow as tf\n"
        f"a=tf.random.normal(({size},{size}))\n"
        f"b=tf.random.normal(({size},{size}))"
    )
    stmt_tf = "tf.reduce_sum(tf.matmul(a,b))"
    tf_time = _benchmark("TensorFlow", setup_tf, stmt_tf)

    print(f"Matrix multiplication benchmark ({size}x{size})")
    if mt_time is not None:
        print(f"Minitensor: {mt_time:.4f}s")
    if torch_time is not None:
        print(f"PyTorch:   {torch_time:.4f}s")
    if tf_time is not None:
        print(f"TensorFlow:{tf_time:.4f}s")


if __name__ == "__main__":  # pragma: no cover - example script
    main()
