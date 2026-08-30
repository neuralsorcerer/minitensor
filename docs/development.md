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

## Where an operation belongs

A new operation goes in one of three places, and which one is decided by what
its arguments carry rather than by how much code it takes.

**A kernel in `engine/`** when the operation has an algorithm of its own: a
loop, a decomposition, a recurrence, or an analytical backward that is better
than the one composition would give. `matmul`, `svd`, `sort`, `conv2d` and
`cross_entropy` are here because each is a real algorithm, not an arrangement
of other ones.

**A Python arrangement in `minitensor/_*.py`** when the operation is existing
kernels pointed at rearranged operands. `tensordot` is `matmul` with the
contracted axes moved and flattened; `nll_loss` is a gather and a weighted
mean; `signbit` is a `copysign` compared against zero. Writing these in Python
keeps the shipped extension the size of the operations that genuinely need a
kernel, and each one inherits the accuracy, the dtype rules and the gradient of
what it is arranged from instead of restating them.

**NumPy** when the computation is over Python integers -- shapes, axes, index
sets, split points -- with no tensor operand at all. `broadcast_shapes` is
`np.broadcast_shapes`; `tril_indices` and `triu_indices` are `np.tril_indices`
and `np.triu_indices`. NumPy is a hard runtime dependency, so this costs
nothing to ship, it is faster than the kernel form at any size where the time
matters, and -- the part that outlasts a benchmark -- it is one fewer
implementation that can disagree with the library everyone compares against.

The line between the second and third cases is the one worth stating, because
it is easy to get backwards: **NumPy computes indices and shapes; the engine
computes values.** A tensor operand carries a device, a dtype and possibly a
gradient. Handing it to NumPy would copy it off the device, drop the gradient
silently -- the operation still returns a right-looking answer, so nothing that
checks values would catch it -- and, for the sampling functions, draw from
NumPy's random stream instead of the one `manual_seed` controls. A Python
integer carries none of those, so nothing is lost by letting NumPy do the
arithmetic. `isin` and `multinomial` stay in terms of kernels for exactly this
reason even though NumPy has both: their arguments are tensors.

## Environment setup

MiniTensor source builds require Python 3.10 or newer, Rust/Cargo, and maturin.
For detailed platform notes, see [the installation guide](installation.md).

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[dev]'
pre-commit install
```

The `dev` extra installs `pytest` (with the benchmark and coverage plugins),
`black`, `mypy`, `pre-commit`, and `numpy`. Use `python -m pip` so tools are
installed into the same interpreter used to run `python -m pytest`.

Two tools are deliberately not in the `dev` extra:

- `maturin` is a build backend, so install it separately when you need
  `maturin develop` (see [the installation guide](installation.md)).
- `isort` is run through its pinned `pre-commit` hook rather than from the
  environment. Use `pre-commit run isort --all-files`, or
  `python -m pip install isort` if you want to invoke it directly.

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
pre-commit run isort --all-files
python -m pytest tests/tensor/test_tensor_core.py
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
