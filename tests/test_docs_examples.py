# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Execute the documentation examples so the docs cannot drift from the API.

Every ```python block in `docs/*.md` and `README.md` is run. When a block is
immediately followed by a ```text block, that text is treated as the block's
expected stdout and compared exactly. This catches three kinds of rot that a
prose review misses: renamed or removed APIs, changed call signatures, and
changed printed values.
"""

from __future__ import annotations

import io
import re
from contextlib import redirect_stdout
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FENCE = re.compile(r"```(\w+)\n(.*?)```", re.S)


def _documents():
    docs = sorted((ROOT / "docs").glob("*.md"))
    readme = ROOT / "README.md"
    if readme.exists():
        docs.append(readme)
    return docs


def _blocks():
    """Yield (doc_name, index, source, expected_stdout_or_None)."""
    for path in _documents():
        # Pin the encoding: the docs contain characters (─, ✅, ᵀ) that a
        # non-UTF-8 default locale, as on Windows CI, cannot decode.
        fences = FENCE.findall(path.read_text(encoding="utf-8"))
        for index, (lang, body) in enumerate(fences):
            if lang != "python":
                continue
            # Illustrative fragments, not runnable programs.
            if body.strip().startswith("...") or "<name>" in body:
                continue
            expected = None
            if index + 1 < len(fences) and fences[index + 1][0] == "text":
                expected = fences[index + 1][1].strip()
            yield path.name, index, body, expected


CASES = list(_blocks())


def test_documentation_contains_examples():
    """Guard against the extraction silently finding nothing."""
    assert len(CASES) > 20, f"only found {len(CASES)} runnable doc examples"


@pytest.mark.parametrize(
    "doc,index,source,expected",
    CASES,
    ids=[f"{name}:block{index}" for name, index, _, _ in CASES],
)
def test_doc_example_runs(doc, index, source, expected):
    buffer = io.StringIO()
    namespace: dict = {"__name__": "__doc_example__"}

    try:
        with redirect_stdout(buffer):
            exec(compile(source, f"{doc}:block{index}", "exec"), namespace)
    except Exception as exc:  # pragma: no cover - failure path is the point
        pytest.fail(
            f"{doc} block {index} raised {type(exc).__name__}: {exc}\n\n{source}"
        )

    if expected is not None:
        actual = buffer.getvalue().strip()
        assert actual == expected, (
            f"{doc} block {index} printed unexpected output.\n"
            f"expected:\n{expected}\n\nactual:\n{actual}"
        )
