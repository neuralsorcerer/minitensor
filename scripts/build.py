# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Build script for minitensor package.

This script provides utilities for building, testing, and packaging minitensor
across different platforms and Python versions.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def run_command(
    cmd: List[str], cwd: Optional[Path] = None, check: bool = True
) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    if cwd:
        print(f"Working directory: {cwd}")

    result = subprocess.run(cmd, cwd=cwd, capture_output=False, check=False)

    if check and result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    return result


def build_rust_engine(release: bool = True) -> None:
    """Build the Rust engine library."""
    print("Building Rust engine...")

    cmd = ["cargo", "build"]
    if release:
        cmd.append("--release")

    run_command(cmd)


def build_python_bindings(release: bool = True, develop: bool = False) -> None:
    """Build Python bindings using maturin."""
    print("Building Python bindings...")

    cmd = ["maturin"]
    if develop:
        cmd.append("develop")
    else:
        cmd.extend(["build", "--out", "dist"])

    if release:
        cmd.append("--release")

    run_command(cmd)


def run_rust_tests() -> None:
    """Run Rust tests."""
    print("Running Rust tests...")

    # Test engine
    run_command(["cargo", "test", "--manifest-path", "engine/Cargo.toml"])

    # Test bindings
    run_command(["cargo", "test", "--manifest-path", "bindings/Cargo.toml"])


def run_python_tests() -> None:
    """Run Python tests."""
    print("Running Python tests...")

    # Install test dependencies
    run_command([sys.executable, "-m", "pip", "install", "pytest", "numpy"])

    # Run tests
    test_files = [
        "tests/",
        "test_*.py",
    ]

    for test_pattern in test_files:
        if Path(test_pattern).exists():
            run_command([sys.executable, "-m", "pytest", test_pattern, "-v"])


def lint_code() -> None:
    """Run linting tools."""
    print("Running linting...")

    # Rust linting
    run_command(["cargo", "fmt", "--all", "--", "--check"])
    run_command(
        ["cargo", "clippy", "--all-targets", "--all-features", "--", "-D", "warnings"]
    )

    # Python linting (if tools are available)
    try:
        run_command(
            [
                sys.executable,
                "-m",
                "black",
                "--check",
                "minitensor/",
                "examples/",
                "*.py",
            ]
        )
        run_command(
            [
                sys.executable,
                "-m",
                "isort",
                "--check-only",
                "minitensor/",
                "examples/",
                "*.py",
            ]
        )
    except subprocess.CalledProcessError:
        print("Python linting tools not available, skipping...")


def clean_build() -> None:
    """Clean build artifacts."""
    print("Cleaning build artifacts...")

    # Clean Rust artifacts
    run_command(["cargo", "clean"])

    # Clean Python artifacts
    import shutil

    for path in ["dist", "build", "*.egg-info"]:
        if Path(path).exists():
            if Path(path).is_dir():
                shutil.rmtree(path)
            else:
                Path(path).unlink()


def install_dev_dependencies() -> None:
    """Install development dependencies."""
    print("Installing development dependencies...")

    # Install Rust tools
    run_command(["cargo", "install", "cargo-audit"], check=False)

    # Install Python tools
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "maturin[patchelf]",
            "pytest",
            "black",
            "isort",
            "mypy",
            "numpy",
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Build script for minitensor")
    parser.add_argument("--debug", action="store_true", help="Build in debug mode")
    parser.add_argument(
        "--develop", action="store_true", help="Install in development mode"
    )
    parser.add_argument("--test", action="store_true", help="Run tests after building")
    parser.add_argument("--lint", action="store_true", help="Run linting")
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts")
    parser.add_argument(
        "--install-deps", action="store_true", help="Install development dependencies"
    )

    args = parser.parse_args()

    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    try:
        if args.clean:
            clean_build()
            return

        if args.install_deps:
            install_dev_dependencies()
            return

        if args.lint:
            lint_code()
            return

        # Build
        release = not args.debug

        build_rust_engine(release=release)
        build_python_bindings(release=release, develop=args.develop)

        if args.test:
            run_rust_tests()
            run_python_tests()

        print("Build completed successfully!")

    except KeyboardInterrupt:
        print("\nBuild interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Build failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
