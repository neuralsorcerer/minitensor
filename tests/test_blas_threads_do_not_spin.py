# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The BLAS a dense product is handed to does not keep the cores busy after it.

After each product OpenBLAS keeps its threads busy-waiting for the next one,
by default for about 130ms, without yielding. The engine's own pool runs the
operations between products on the same cores, and its workers waited behind
those threads for a scheduler tick: a training step's 99th percentile was
13-21ms against a median of 1.2ms. Importing minitensor first now shortens the
wait to about 2ms, unless the variable that sets it is already set.
"""

import os
import subprocess
import sys

import pytest

_TIMEOUT = """
import os
import minitensor
print(os.environ.get("OPENBLAS_THREAD_TIMEOUT"))
"""


def _run(code, **env):
    environ = {k: v for k, v in os.environ.items() if k != "OPENBLAS_THREAD_TIMEOUT"}
    environ.update(env)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        env=environ,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout.strip()


def test_importing_minitensor_shortens_the_blas_spin():
    assert _run(_TIMEOUT) == "22"


def test_a_value_already_set_is_kept():
    assert _run(_TIMEOUT, OPENBLAS_THREAD_TIMEOUT="28") == "28"


# CPU time the process's other threads spend in the half second after one
# delegated product, with the engine's pool idle the whole time.
_SPIN = """
import os
import time
import minitensor as mt

def others():
    total = 0
    for tid in os.listdir("/proc/self/task"):
        if int(tid) == os.getpid():
            continue
        with open(f"/proc/self/task/{tid}/stat") as stat:
            fields = stat.read().rsplit(")", 1)[1].split()
        total += int(fields[11]) + int(fields[12])
    return total / os.sysconf("SC_CLK_TCK")

a = mt.randn(256, 256)
b = mt.randn(256, 256)
a @ b
time.sleep(1.0)
before = others()
a @ b
time.sleep(0.5)
print(others() - before)
"""


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="reads /proc/self/task"
)
def test_the_blas_threads_rest_after_a_product():
    # Spinning for the default 130ms on three threads is about 0.4s of CPU.
    assert float(_run(_SPIN)) < 0.1
