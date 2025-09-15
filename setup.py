# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.


import sys
from pathlib import Path

# Ensure we have the minimum Python version
if sys.version_info < (3, 10):
    print("Error: minitensor requires Python 3.10 or later")
    sys.exit(1)

try:
    from setuptools import find_packages, setup
except ImportError:
    print("Error: setuptools is required to build minitensor")
    print("Please install it with: pip install setuptools")
    sys.exit(1)

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

# Minimal dependencies - keep this lean for production
INSTALL_REQUIRES = [
    "numpy",
]

# Development dependencies
EXTRAS_REQUIRE = {
    "dev": [
        "pytest",
        "pytest-benchmark",
        "black",
        "isort",
        "mypy",
        "maturin",
    ],
    "docs": [
        "sphinx",
        "sphinx-rtd-theme",
        "myst-parser",
    ],
    "examples": [
        "matplotlib",
        "jupyter",
        "notebook",
        "scikit-learn",
    ],
    "test": [
        "pytest",
        "pytest-benchmark",
        "numpy",
    ],
}

# Add 'all' extra that includes everything
EXTRAS_REQUIRE["all"] = sorted(
    {dep for deps in EXTRAS_REQUIRE.values() for dep in deps}
)

setup(
    name="minitensor",
    use_scm_version={
        "write_to": "minitensor/_version.py",
        "fallback_version": "0.1.0",
    },
    description="A lightweight, high-performance tensor operations library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Soumyadip Sarkar",
    url="https://github.com/neuralsorcerer/minitensor",
    project_urls={
        "Homepage": "https://github.com/neuralsorcerer/minitensor#readme",
        "Repository": "https://github.com/neuralsorcerer/minitensor",
        "Bug Tracker": "https://github.com/neuralsorcerer/minitensor/issues",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Rust",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    packages=find_packages(),
    package_dir={"minitensor": "minitensor"},
    package_data={
        "minitensor": ["py.typed", "*.pyi"],
    },
    include_package_data=True,
    zip_safe=False,
    setup_requires=["setuptools_scm"],
    keywords=["deep-learning", "neural-networks", "tensor", "rust", "python"],
)
