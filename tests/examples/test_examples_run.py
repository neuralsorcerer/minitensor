# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Every example must run the way the docs say to run it.

Eleven of the thirteen could not. `python examples/foo.py` puts *the script's*
directory on `sys.path`, not the working directory, so a source checkout cannot
see the package sitting next to it, and every one of them died on
`import minitensor`.

Two were worse than dead. `example_neural_network.py` and
`performance_benchmark.py` caught the ImportError, printed that they had
failed, and exited 0 -- one of them advising the reader to build the extension
with `maturin develop --release`, which they had already done, since that is
not what was wrong. `docs/performance.md` tells the reader to run the second of
those by name.

The other tests in this directory import an example's functions, which works
because pytest runs from the repository root. That is exactly why none of them
noticed: the failure is in the invocation, not the code. These run the scripts
as scripts.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"

SCRIPTS = sorted(
    path.name
    for path in EXAMPLES.glob("*.py")
    if path.name != "__init__.py" and "import minitensor" in path.read_text()
)

# Long-running by design; they are checked for *startup*, below.
SLOW = {"benchmark_suite.py", "performance_benchmark.py", "digits_cnn.py"}


def _run(name, timeout):
    """Run an example the way a reader would: by path, from the repo root."""
    return subprocess.run(
        [sys.executable, str(EXAMPLES / name)],
        cwd=EXAMPLES.parent,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_there_are_examples_to_check():
    """Guards the parametrisation: an empty SCRIPTS list would make every test
    below vacuous."""
    assert len(SCRIPTS) >= 10


@pytest.mark.parametrize("name", [n for n in SCRIPTS if n not in SLOW])
def test_example_runs_to_completion(name):
    result = _run(name, timeout=600)
    assert (
        result.returncode == 0
    ), f"{name} exited {result.returncode}\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"


@pytest.mark.parametrize("name", SCRIPTS)
def test_example_actually_imports_the_library(name):
    """Exit 0 is not enough.

    Two examples wrapped their imports in `try/except` and reported failure on
    stdout while exiting cleanly, so they passed a returncode check while doing
    nothing at all.
    """
    try:
        result = _run(name, timeout=600)
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired as expired:
        # The slow ones are allowed to be killed by the timeout, but whatever
        # they printed before that must still show the import succeeded.
        output = (expired.stdout or b"").decode() + (expired.stderr or b"").decode()

    assert (
        "No module named 'minitensor'" not in output
    ), f"{name} could not import the library it demonstrates:\n{output[-2000:]}"


@pytest.mark.parametrize("name", SCRIPTS)
def test_example_does_not_point_at_the_wrong_fix(name):
    """`maturin develop --release` was the advice given for a `sys.path`
    problem, which sends the reader to rebuild something that is already
    built."""
    source = (EXAMPLES / name).read_text()
    if "maturin develop" not in source:
        return
    # If it still suggests rebuilding, that must not be reachable when the
    # package is importable -- which it is, here.
    result = _run(name, timeout=600)
    assert (
        "maturin develop" not in result.stdout
    ), f"{name} advises rebuilding the extension even though it imported fine"
