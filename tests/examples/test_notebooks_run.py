# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The notebooks must run where a reader will open them.

All three died on cell 1 with `ModuleNotFoundError: No module named
'minitensor'`. A Jupyter kernel starts in the notebook's own directory, so
`examples/notebooks/` was the working directory and a source checkout could not
see the package three levels up. They worked only if the kernel happened to be
started from the repository root -- which is not what `jupyter notebook
examples/notebooks/00_tensor_and_functional_api.ipynb` does, and `docs/index.md`
points readers at them.

The cells are executed in order in a subprocess, from the notebook's own
directory, which is what a kernel does. Matplotlib is only used for plots; when
it is absent -- it lives in the `examples` extra -- a stub stands in so the
MiniTensor half is still exercised rather than the whole file being skipped.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

NOTEBOOKS = Path(__file__).resolve().parents[2] / "examples" / "notebooks"
NAMES = sorted(path.name for path in NOTEBOOKS.glob("*.ipynb"))

RUNNER = """
import json, sys, os, pathlib
# A Jupyter kernel puts the working directory on sys.path; a script run by path
# does not, so emulate the kernel rather than the script.
sys.path.insert(0, os.getcwd())
cells = [
    c for c in json.loads(pathlib.Path(sys.argv[1]).read_text())["cells"]
    if c["cell_type"] == "code"
]
env = {"__name__": "__main__"}
for number, cell in enumerate(cells, 1):
    source = "".join(cell["source"])
    if source.strip():
        exec(compile(source, f"cell{number}", "exec"), env)
"""

# Enough of pyplot to let a notebook's plotting calls pass through untouched.
STUB = """
def _noop(*args, **kwargs):
    return _Anything()


class _Anything:
    def __getattr__(self, _):
        return _noop

    def __iter__(self):
        return iter((_Anything(), _Anything()))


def __getattr__(name):
    return _noop
"""


@pytest.fixture(scope="module")
def stub_root(tmp_path_factory):
    """A matplotlib stand-in, used only when the real one is not installed."""
    try:
        import matplotlib  # noqa: F401

        return None
    except ImportError:
        root = tmp_path_factory.mktemp("stub")
        package = root / "matplotlib"
        package.mkdir()
        (package / "__init__.py").write_text("def use(*a, **k):\n    pass\n")
        (package / "pyplot.py").write_text(textwrap.dedent(STUB))
        return root


def _run(name, cwd, stub_root, tmp_path):
    runner = tmp_path / "runner.py"
    runner.write_text(textwrap.dedent(RUNNER))
    env = None
    if stub_root is not None:
        import os

        env = dict(os.environ, PYTHONPATH=str(stub_root))
    return subprocess.run(
        [sys.executable, str(runner), str(NOTEBOOKS / name)],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )


def test_there_are_notebooks_to_check():
    assert len(NAMES) >= 3


@pytest.mark.parametrize("name", NAMES)
def test_notebook_runs_from_its_own_directory(name, stub_root, tmp_path):
    """Jupyter's default working directory, and the one that was broken."""
    result = _run(name, NOTEBOOKS, stub_root, tmp_path)
    assert result.returncode == 0, f"{name}\n{result.stderr[-2500:]}"


@pytest.mark.parametrize("name", NAMES)
def test_notebook_runs_from_the_repository_root(name, stub_root, tmp_path):
    """The only place they used to work, which must keep working."""
    result = _run(name, NOTEBOOKS.parents[1], stub_root, tmp_path)
    assert result.returncode == 0, f"{name}\n{result.stderr[-2500:]}"


@pytest.mark.parametrize("name", NAMES)
def test_notebook_imports_the_library_it_demonstrates(name, stub_root, tmp_path):
    result = _run(name, NOTEBOOKS, stub_root, tmp_path)
    assert "No module named 'minitensor'" not in result.stderr, (
        f"{name} cannot find the library from a kernel started in its own "
        f"directory:\n{result.stderr[-2500:]}"
    )


@pytest.mark.parametrize("name", NAMES)
def test_notebook_outputs_are_not_stale_errors(name):
    """A committed traceback tells a reader the library is broken before they
    have run anything."""
    document = json.loads((NOTEBOOKS / name).read_text())
    for number, cell in enumerate(document["cells"], 1):
        for output in cell.get("outputs", []):
            assert output.get("output_type") != "error", (
                f"{name} cell {number} has a stored {output.get('ename')}"
            )
