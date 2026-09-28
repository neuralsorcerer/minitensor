# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A forked child computes what its parent computes, and does not hang.

A fork keeps only the thread that called it. The parallel kernels ran on
rayon's global pool, whose workers were therefore gone from any child forked
after the parent had used them, while the pool still counted them: the child's
first large operation handed its work to threads that did not exist and waited
for them forever. `multiprocessing` forks by default on Linux before Python
3.14, so every worker of a process pool started after one large operation in
the parent hung on its first large operation.

Each case below reaches the pool by a different path -- an element map, a
fold, a sort, the convolution's GEMM bands, a row-partitioned kernel, a copy, a
backward pass, the random stream -- and runs in the parent first, so that the
pool exists before the fork. The child then runs every case again and must
reproduce the parent's result exactly, within a timeout that turns a hang into
a failure. A plain `matmul` is left out: at any size worth splitting it goes to
NumPy, whose BLAS decides for itself whether it survives a fork.
"""

import os
import signal
import time
import warnings

import numpy as np
import pytest

import minitensor as mt
import minitensor.functional as F

pytestmark = pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")

CHILD_TIMEOUT = 60.0


def _inputs():
    rng = np.random.default_rng(0)
    return {
        "vector": mt.from_numpy(rng.standard_normal(1 << 20).astype(np.float32)),
        "matrix": mt.from_numpy(rng.standard_normal((512, 1024)).astype(np.float32)),
        "images": mt.from_numpy(
            rng.standard_normal((8, 16, 32, 32)).astype(np.float32)
        ),
        "kernels": mt.from_numpy(
            rng.standard_normal((32, 16, 3, 3)).astype(np.float32)
        ),
        "integers": mt.from_numpy(rng.integers(0, 1000, 1 << 18).astype(np.int64)),
    }


def _cases(inputs):
    vector, matrix = inputs["vector"], inputs["matrix"]

    def backward():
        x = matrix.detach().requires_grad_(True)
        (F.softmax(x * 2.0, dim=-1) * matrix).sum().backward()
        grad = x.grad.numpy()
        mt.clear_autograd_graph()
        return grad

    def dropout():
        mt.manual_seed(7)
        return F.dropout(matrix, 0.5, training=True).numpy()

    return {
        "element map": lambda: (vector * 2.0 + 1.0).exp().numpy(),
        "sum": lambda: vector.sum().numpy(),
        "argmax": lambda: vector.argmax().numpy(),
        "nanmean": lambda: vector.nanmean().numpy(),
        "sort": lambda: mt.sort(vector)[0].numpy(),
        "unique": lambda: mt.unique(inputs["integers"]).numpy(),
        "conv2d": lambda: F.conv2d(inputs["images"], inputs["kernels"]).numpy(),
        "softmax": lambda: F.softmax(matrix, dim=-1).numpy(),
        "layer_norm": lambda: F.layer_norm(matrix, [1024]).numpy(),
        "transpose copy": lambda: matrix.transpose(0, 1).contiguous().numpy(),
        "flip": lambda: mt.flip(matrix, [1]).numpy(),
        "cumsum": lambda: mt.cumsum(matrix, 1).numpy(),
        "backward": backward,
        "dropout": dropout,
    }


def _wait(pid):
    deadline = time.monotonic() + CHILD_TIMEOUT
    while time.monotonic() < deadline:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            return os.waitstatus_to_exitcode(status)
        time.sleep(0.05)
    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)
    return None


def test_a_forked_child_runs_every_parallel_path_and_agrees_with_its_parent():
    cases = _cases(_inputs())
    expected = {name: case() for name, case in cases.items()}

    read, write = os.pipe()
    # The pool's threads make this process multi-threaded, which Python 3.12+
    # warns about on every fork; the warning is the hazard this test covers.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        pid = os.fork()
    if pid == 0:  # pragma: no cover - runs in the child
        os.close(read)
        failed = []
        try:
            for name, case in cases.items():
                got = case()
                if got.shape != expected[name].shape or not np.array_equal(
                    got, expected[name], equal_nan=True
                ):
                    failed.append(name)
        except BaseException as error:  # noqa: BLE001 - reported to the parent
            failed.append(f"raised {error!r}")
        os.write(write, ",".join(failed).encode())
        os._exit(0)

    os.close(write)
    code = _wait(pid)
    report = os.read(read, 1 << 16).decode()
    os.close(read)
    assert code is not None, f"the forked child hung (timed out after {CHILD_TIMEOUT}s)"
    assert code == 0, f"the forked child exited with {code}"
    assert report == "", f"the forked child disagreed with its parent on: {report}"


def test_a_child_of_a_child_still_has_a_pool():
    matrix = _inputs()["matrix"]
    expected = F.softmax(matrix, dim=-1).numpy()

    def descend(depth):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            pid = os.fork()
        if pid == 0:  # pragma: no cover - runs in the child
            same = np.array_equal(F.softmax(matrix, dim=-1).numpy(), expected)
            code = 1 if not same else (descend(depth - 1) if depth > 1 else 0)
            os._exit(code if code is not None else 2)
        return _wait(pid)

    assert descend(2) == 0
