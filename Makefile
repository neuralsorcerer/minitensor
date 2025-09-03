# Makefile for minitensor development

.PHONY: help install build test lint clean dev-install release check-deps

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
	@echo "  check-deps   - Check for security vulnerabilities"
	@echo "  benchmark    - Run performance benchmarks"

# Check if required tools are installed
check-deps:
	@command -v cargo >/dev/null 2>&1 || { echo "Rust/Cargo is required but not installed. Aborting." >&2; exit 1; }
	@command -v python >/dev/null 2>&1 || { echo "Python is required but not installed. Aborting." >&2; exit 1; }
	@python -c "import maturin" 2>/dev/null || { echo "Maturin is required. Install with: pip install maturin[patchelf]" >&2; exit 1; }

# Install development dependencies
dev-install: check-deps
	pip install -e ".[dev]"
	cargo install cargo-audit || true

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
	cargo test --manifest-path engine/Cargo.toml
	cargo test --manifest-path bindings/Cargo.toml

# Run Python tests
test-python: install
	python -m pytest tests/ -v

# Run linting
lint:
	cargo fmt --all -- --check
	cargo clippy --all-targets --all-features -- -D warnings
	python -m black --check minitensor/ examples/ *.py
	python -m isort --check-only minitensor/ examples/ *.py

# Format code
format:
	cargo fmt --all
	python -m black minitensor/ examples/ *.py
	python -m isort minitensor/ examples/ *.py

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

# Run benchmarks
benchmark: install
	python -m pytest benchmarks/ --benchmark-only || echo "No benchmarks found"

# Quick development cycle
dev: format lint test

# CI simulation
ci: check-deps lint test

# Version management
version-current:
	python scripts/version.py --current

version-bump-patch:
	python scripts/version.py --bump patch

version-bump-minor:
	python scripts/version.py --bump minor

version-bump-major:
	python scripts/version.py --bump major
