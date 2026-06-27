# Installation Guide

This guide covers the supported ways to install MiniTensor for end users and
contributors. MiniTensor is a Python package backed by Rust extension modules,
so source installs require both Python and the Rust toolchain.

## Requirements

| Requirement | Needed for | Notes |
| --- | --- | --- |
| Python 3.10 or newer | All installs | Use `python -m pip` so packages are installed into the interpreter you will run. |
| Rust and Cargo | Source installs and development | Install with [`rustup`](https://rustup.rs/) if `cargo --version` is unavailable. |
| maturin | Source builds | Linux source builds should install `maturin[patchelf]`; macOS and Windows can use `maturin`. |
| A virtual environment | Recommended | Keeps MiniTensor and development tools isolated from system Python. |

## Install from PyPI

For normal use, install the published wheel from PyPI:

```bash
python -m pip install --upgrade pip
python -m pip install minitensor
```

Verify the install:

```bash
python - <<'PY'
import minitensor as mt
print(mt.__version__)
print(mt.Tensor([1, 2, 3]).shape)
PY
```

## Install from source

Clone the repository first:

```bash
git clone https://github.com/neuralsorcerer/minitensor.git
cd minitensor
```

### Manual source install

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate      # Windows PowerShell
python -m pip install --upgrade pip setuptools wheel
```

Install the build backend and compile the extension:

```bash
python -m pip install 'maturin[patchelf]'  # Linux
# python -m pip install maturin            # macOS/Windows
maturin develop --release
```

For an editable pip install, use:

```bash
python -m pip install -e .
```

Editable pip installs use the release profile configured in `pyproject.toml`.
Use `maturin develop --debug` when you intentionally want an unoptimized debug
build for local debugging.

## Contributor setup

Contributor installs should include the `dev` extra. The extra includes pytest,
coverage and benchmark plugins, mypy, pre-commit, isort, and `black[jupyter]` so
notebook formatting uses the same dependencies as the pre-commit Black hook.

```bash
python -m pip install -e '.[dev]'
pre-commit install
```

Rebuild after changing Rust code under `engine/` or PyO3 bindings under
`bindings/`:

```bash
python -m pip install -e .
```

Pure Python and documentation changes do not require rebuilding the Rust
extension.

## Validation commands

Run the same checks before opening a pull request:

```bash
cargo test --workspace --all-targets
python -m pytest
pre-commit run --all-files
```

For explicit formatting and lint checks outside pre-commit:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
black --check .
isort --check-only .
```

## Troubleshooting

- If Python imports an older MiniTensor build, reinstall with the same
  interpreter you use to run tests: `python -m pip install -e .`.
- If maturin reports a malformed or empty shared library after an interrupted
  build, remove stale build artifacts and rebuild:

  ```bash
  rm -f target/maturin/libminitensor.so target/release/libminitensor.so target/release/deps/libminitensor.so
  python -m pip install -e .
  ```

- If `black --check .` mentions missing Jupyter dependencies, reinstall the dev
  extra: `python -m pip install -e '.[dev]'`.
