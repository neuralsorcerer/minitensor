# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sphinx configuration for the MiniTensor documentation."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

project = "MiniTensor"
author = "Soumyadip Sarkar"
copyright = f"{datetime.now(UTC):%Y}, {author}"

# Keep docs builds independent from importing the compiled extension. The package
# version is mirrored from pyproject.toml by packaging metadata when installed;
# fall back to the repository version for source-only documentation builds.
release = "0.1.7"
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]
templates_path = ["_templates"]
html_static_path = ["_static"]

html_theme = "furo"
html_title = f"{project} documentation"
html_short_title = project
html_logo = "_static/img/minitensor-small.png"
html_favicon = "_static/img/minitensor-small.png"
html_css_files = ["custom.css"]
html_show_sourcelink = True
html_show_sphinx = False
html_last_updated_fmt = "%Y-%m-%d"
html_baseurl = os.environ.get("SPHINX_HTML_BASEURL", "")
html_theme_options = {
    "light_logo": "img/minitensor-small.png",
    "dark_logo": "img/minitensor-dark-small.png",
    "source_repository": "https://github.com/neuralsorcerer/minitensor/",
    "source_branch": os.environ.get("GITHUB_REF_NAME", "main"),
    "source_directory": "docs/",
}
html_context = {
    "display_github": True,
    "github_user": "neuralsorcerer",
    "github_repo": "minitensor",
    "github_version": os.environ.get("GITHUB_REF_NAME", "main"),
    "conf_py_path": "/docs/",
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "linkify",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3
myst_links_external_new_tab = True
myst_substitutions = {
    "project": project,
    "version": release,
}

# Make duplicate section titles deterministic and useful in generated anchors.
autosectionlabel_prefix_document = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
extlinks = {
    "issue": ("https://github.com/neuralsorcerer/minitensor/issues/%s", "#%s"),
    "pull": ("https://github.com/neuralsorcerer/minitensor/pull/%s", "PR #%s"),
}

todo_include_todos = False
nitpicky = True
suppress_warnings = [
    # Markdown guide links intentionally point to repository-local examples that
    # are outside the Sphinx source tree.
    "myst.xref_missing",
]

linkcheck_ignore = [
    r"http://127\.0\.0\.1:\d+/?",
    r"http://localhost:\d+/?",
]
linkcheck_timeout = 20
linkcheck_retries = 2
