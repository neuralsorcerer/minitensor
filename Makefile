# Makefile for minitensor development

.PHONY: help check-deps dev-install install build build-release test test-rust \
        test-python lint format clean release check-deps-security benchmark dev ci

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install the package in development mode"
	@echo "  build        - Build the package"
	@echo "  build-release - Build the package in release mode"
	@echo "  test         - Run all tests"
	@echo "  test-rust    - Run Rust tests only"
	@echo "  test-python  - Run Python tests only"
	@echo "  lint         - Run linting tools"
	@echo "  format       - Format code"
	@echo "  clean        - Clean build artifacts"
	@echo "  dev-install  - Install development dependencies"
	@echo "  release      - Build release packages"
	@echo "  check-deps   - Verify cargo/python/maturin are installed"
	@echo "  check-deps-security - Audit dependencies for advisories"
	@echo "  benchmark    - Run the bundled performance benchmark"

# Check if required tools are installed
check-deps:
	@command -v cargo >/dev/null 2>&1 || { echo "Rust/Cargo is required but not installed. Aborting." >&2; exit 1; }
	@command -v python >/dev/null 2>&1 || { echo "Python is required but not installed. Aborting." >&2; exit 1; }
	@python -c "import maturin" 2>/dev/null || { echo "Maturin is required. Install with: pip install maturin[patchelf]" >&2; exit 1; }

# Install development dependencies
dev-install:
	@command -v python >/dev/null 2>&1 || { echo "Python is required but not installed. Aborting." >&2; exit 1; }
	python -m pip install -e ".[dev]"

# Install in development mode
install: check-deps
	maturin develop --release

# Build the package
build: check-deps
	maturin build

# Build in release mode
build-release: check-deps
	maturin build --release --out dist

# Run all tests
test: test-rust test-python

# Run Rust tests
test-rust:
	RUSTFLAGS="$(RUSTFLAGS) -D warnings" cargo test --workspace --all-targets

# Run Python tests
test-python: dev-install
	python -m pytest tests/ -v

# Run linting
lint:
	cargo fmt --all -- --check
	cargo clippy --workspace --all-targets -- -D warnings
	python -m black --check .
	python -m isort --check-only .

# Format code
format:
	cargo fmt --all
	python -m black .
	python -m isort .

# Clean build artifacts
clean:
	cargo clean
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + || true
	find . -type f -name "*.pyc" -delete || true

# Build release packages
release: clean build-release
	@echo "Release packages built in dist/"
	@ls -la dist/

# Security audit
check-deps-security:
	cargo audit
	pip-audit || echo "pip-audit not available, skipping Python security check"

# Run the bundled benchmark (examples/performance_benchmark.py). There is no
# `benchmarks/` directory; this target used to point at one and so always
# reported "No benchmarks found", including via the `make benchmark` line in
# docs/performance.md.
benchmark: install
	python examples/performance_benchmark.py

# Quick development cycle
dev: format lint test

# CI simulation
ci: check-deps lint test
