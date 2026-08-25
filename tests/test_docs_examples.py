# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hold the documentation to the API it documents.

Every ```python block in `docs/*.md` and `README.md` is run. When a block is
immediately followed by a ```text block, that text is treated as the block's
expected stdout and compared exactly. This catches three kinds of rot that a
prose review misses: renamed or removed APIs, changed call signatures, and
changed printed values.

Running the examples says nothing about the *lists* of names in the same files,
though, and one of those is the map a reader uses to find out an op exists at
all. `test_documented_top_level_names_are_exactly_the_forwarded_ones` checks
that one against the exports it claims to enumerate.
"""

from __future__ import annotations

import io
import re
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from minitensor._exports import _FUNCTIONAL_FORWARDERS

ROOT = Path(__file__).resolve().parent.parent
FENCE = re.compile(r"```(\w+)\n(.*?)```", re.S)
# The prose introducing the bare fenced list of every name reachable as both
# `minitensor.<name>` and `minitensor.functional.<name>`.
NAME_LIST_HEADING = "Each of the following names is accessible from:"


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


def _documented_top_level_names() -> set[str]:
    reference = (ROOT / "docs" / "api_reference.md").read_text(encoding="utf-8")
    heading = reference.index(NAME_LIST_HEADING)
    listing = re.search(r"```\n(.*?)```", reference[heading:], re.S)
    assert (
        listing is not None
    ), "the name list under the heading is no longer a fenced block"
    # The list is grouped under `# ...` headings so a reader can find a name by
    # what it does; those lines are not names.
    names = " ".join(
        line for line in listing.group(1).splitlines() if not line.startswith("#")
    )
    return {name.strip() for name in names.split(",") if name.strip()}


def test_documented_top_level_names_are_exactly_the_forwarded_ones():
    """The listed names and `_FUNCTIONAL_FORWARDERS` must be the same set.

    This list had drifted 37 names behind by the time it was checked: every op
    added since -- `einsum`, `svd`, `qr`, `cholesky`, `unique`, `searchsorted`,
    the whole `lu` family -- was reachable but undocumented, and a reader
    consulting the reference would have concluded the library could not do any
    of it.
    """
    documented = _documented_top_level_names()
    forwarded = set(_FUNCTIONAL_FORWARDERS)

    missing = sorted(forwarded - documented)
    assert (
        not missing
    ), "forwarded but absent from the api_reference.md name list: " + ", ".join(missing)

    extra = sorted(documented - forwarded)
    assert (
        not extra
    ), "listed in api_reference.md but not forwarded to the top level: " + ", ".join(
        extra
    )
