# Development Guide

This guide is for contributors changing MiniTensor's Python package, PyO3
bindings, Rust engine, examples, tests, or documentation.

## Repository layout

| Path | Purpose |
| --- | --- |
| `minitensor/` | Python package shim that re-exports the compiled Rust extension and adds convenience helpers. |
| `bindings/` | PyO3 extension module exposed to Python as `minitensor._core`. |
| `engine/` | Core Rust tensor engine, operations, autograd, backends, memory management, custom ops, and plugins. |
| `tests/` | Python test suite for user-facing behavior. |
| `engine/tests/` | Rust integration and engine-level tests. |
| `examples/` | Runnable Python and Rust examples. |
| `docs/` | Markdown documentation and static assets. |

## Environment setup

MiniTensor source builds require Python 3.10 or newer, Rust/Cargo, and maturin.
For detailed platform notes, see [the installation guide](installation.md).

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[dev]'
pre-commit install
```

The `dev` extra installs test, lint, formatting, benchmark, typing, and notebook
formatting tools. Use `python -m pip` so tools are installed into the same
interpreter used to run `python -m pytest`.

## Build workflow

Rebuild the extension after changing Rust code in `engine/` or PyO3 bindings in
`bindings/`:

```bash
python -m pip install -e .
```

For an optimized local extension, use:

```bash
maturin develop --release
```

For debugging native code, use:

```bash
maturin develop --debug
```

Pure Python, test-only, example-only, and documentation-only changes usually do
not require rebuilding the extension unless they depend on newly added Rust API.

## Validation commands

Run the checks that match the files you changed, and prefer the full suite before
opening a pull request:

```bash
cargo test --workspace --all-targets
python -m pytest
pre-commit run --all-files
```

Useful targeted checks:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
black --check .
isort --check-only .
python -m pytest tests/test_tensor_creation.py
```

## Documentation workflow

- Keep examples copy-pasteable and prefer commands that work from the repository
  root.
- Use `python -m ...` invocations for Python tooling so the intended interpreter
  is unambiguous.
- Update [the API reference](api_reference.md) whenever public symbols are added,
  renamed, removed, or given new behavior.
- Link from [the documentation index](index.md) when adding a new guide.
- Keep README examples concise; move lengthy explanations into `docs/`.

## Pull request checklist

Before submitting changes:

- Rebuild the extension if Rust or binding code changed.
- Run relevant Rust and Python tests.
- Run formatting and lint checks.
- Update docs and examples for user-facing behavior.
- Confirm generated artifacts, virtual environments, and build outputs are not
  committed.

## Publishing documentation

MiniTensor publishes the Markdown documentation in `docs/` with Sphinx, MyST, and
GitHub Pages. The production build is configured in `docs/conf.py` and deployed
by `.github/workflows/docs.yml` whenever documentation changes are pushed to the
`main` branch. Pull requests build the same Sphinx site with warnings treated as
errors, but deployment is restricted to `main`.

To preview the documentation locally from the repository root:

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m http.server 8000 --directory docs/_build/html
```

Then open <http://127.0.0.1:8000/>.

Documentation publishing rules:

- Keep `docs/index.md` as the root document and update its Sphinx toctree when
  adding or removing pages.
- Run the local Sphinx build before opening documentation pull requests.
- Do not commit generated files under `docs/_build/`.
- Keep repository-local links relative so the Markdown remains readable on
  GitHub and in the generated site.
