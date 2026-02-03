<div align="center">

![MiniTensor Logo](docs/_static/img/minitensor-small.png)

</div>
<h3 align="center">
A lightweight, high-performance tensor operations library with automatic differentiation, inspired by <a href="https://github.com/pytorch/pytorch">PyTorch</a> and powered by Rust engine.
</h3>

---

<div align="center">

[![Current Release](https://img.shields.io/github/release/neuralsorcerer/minitensor.svg)](https://github.com/neuralsorcerer/minitensor/releases)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-fcbc2c.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![rustc 1.89+](https://img.shields.io/badge/rustc-1.89+-blue.svg?logo=rust&logoColor=white)](https://rust-lang.github.io/rfcs/2495-min-rust-version.html)
[![Test Linux](https://github.com/neuralsorcerer/minitensor/actions/workflows/test_ubuntu.yml/badge.svg)](https://github.com/neuralsorcerer/minitensor/actions/workflows/test_ubuntu.yml?query=branch%3Amain)
[![Test Windows](https://github.com/neuralsorcerer/minitensor/actions/workflows/test_windows.yml/badge.svg)](https://github.com/neuralsorcerer/minitensor/actions/workflows/test_windows.yml?query=branch%3Amain)
[![Test MacOS](https://github.com/neuralsorcerer/minitensor/actions/workflows/test_macos.yml/badge.svg)](https://github.com/neuralsorcerer/minitensor/actions/workflows/test_macos.yml?query=branch%3Amain)
[![Lints](https://github.com/neuralsorcerer/minitensor/actions/workflows/lints.yml/badge.svg)](https://github.com/neuralsorcerer/minitensor/actions/workflows/lints.yml?query=branch%3Amain)
[![arXiv](https://img.shields.io/badge/arXiv-2602.00125-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2602.00125)
[![DOI:48550/arXiv.2602.00125](https://img.shields.io/badge/DOI-10.48550/arXiv.2602.00125-blue.svg)](https://doi.org/10.48550/arXiv.2602.00125)
[![License](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white)](./LICENSE)

</div>

## Highlights

- **High Performance**: Rust engine for maximum speed and memory efficiency
- **Python-Friendly**: Familiar PyTorch-like API for easy adoption
- **Neural Networks**: Complete neural network layers and optimizers
- **NumPy Integration**: Seamless interoperability with NumPy arrays
- **Automatic Differentiation**: Built-in gradient computation for training
- **Extensible**: Modular design for easy customization and extension

## Quick Start

### Installation

**From PyPi:**

```bash
pip install minitensor
```

**From Source:**

```bash
# Clone the repository
git clone https://github.com/neuralsorcerer/minitensor.git
cd minitensor

# Quick install with make (Linux/macOS)
make install

# Or manually with maturin
pip install maturin[patchelf]
maturin develop --release

# Optional: editable install with pip (debug build by default)
pip install -e .
```

> _Note:_ `pip install -e .` builds a debug version by default; pass `--config-settings=--release` for a release build.

**Using the install script (Linux/macOS/Windows):**

```bash
bash install.sh
```

Common options:

```bash
bash install.sh --no-venv          # Use current Python env (no virtualenv)
bash install.sh --venv .myvenv     # Create/use a specific venv directory
bash install.sh --debug            # Debug build (default is --release)
bash install.sh --python /usr/bin/python3.12   # Use a specific Python
```

The script ensures Python 3.10+, sets up a virtual environment by default, installs Rust (via rustup if needed), installs maturin (with patchelf on Linux), builds MiniTensor, and verifies the installation.

### Basic Usage

```python
import minitensor as mt
from minitensor import nn, optim

# Create tensors
x = mt.randn(32, 784)  # Batch of 32 samples
y = mt.zeros(32, 10)   # Target labels

# Build a neural network
model = nn.Sequential([
    nn.DenseLayer(784, 128),
    nn.ReLU(),
    nn.DenseLayer(128, 10)
])

# Set up training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(0.001, betas=(0.9, 0.999), epsilon=1e-8)

print(f"Model: {model}")
print(f"Input shape: {x.shape}")
```

## Documentation

### Core Components

#### Tensors

```python
import minitensor as mt
import numpy as np

# Create tensors
x = mt.zeros(3, 4)          # Zeros
y = mt.ones(3, 4)           # Ones
z = mt.randn(2, 2)          # Random normal
np_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
w = mt.from_numpy(np_array) # From NumPy

# Operations
result = x + y                      # Element-wise addition
product = x.matmul(y.T)             # Matrix multiplication
mean_val = x.mean()                 # Reduction operations
max_val = x.max()                   # -inf for empty or all-NaN tensors
min_vals, min_idx = x.min(dim=1)    # Returns values & indices; empty dims yield (inf, 0)
```

#### Neural Networks

```python
from minitensor import nn

# Layers
dense = nn.DenseLayer(10, 5)        # Dense layer (fully connected)
conv = nn.Conv2d(3, 16, 3)          # 2D convolution
bn = nn.BatchNorm1d(128)            # Batch normalization
dropout = nn.Dropout(0.5)           # Dropout regularization

# Activations
relu = nn.ReLU()                    # ReLU activation
sigmoid = nn.Sigmoid()              # Sigmoid activation
tanh = nn.Tanh()                    # Tanh activation
gelu = nn.GELU()                    # GELU activation

# Loss functions
mse = nn.MSELoss()                  # Mean squared error
ce = nn.CrossEntropyLoss()          # Cross entropy
bce = nn.BCELoss()                  # Binary cross entropy
```

#### Optimizers

```python
from minitensor import optim

# Optimizers
sgd = optim.SGD(0.01, 0.9, 0.0, False)                              # SGD with momentum
adam = optim.Adam(0.001, betas=(0.9, 0.999), epsilon=1e-8)          # Adam optimizer
adamw = optim.AdamW(0.001, betas=(0.9, 0.999), weight_decay=0.01)   # Decoupled AdamW
rmsprop = optim.RMSprop(0.01, 0.99, 1e-8, 0.0, 0.0)                 # RMSprop optimizer
```

## Architecture

Minitensor is built with a modular architecture:

```text
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Python API    │    │   PyO3 Bindings  │    │   Rust Engine   │
│                 │<-->│                  │<-->│                 │
│ • Tensor        │    │ • Type Safety    │    │ • Performance   │
│ • nn.Module     │    │ • Memory Mgmt    │    │ • Autograd      │
│ • Optimizers    │    │ • Error Handling │    │ • SIMD/GPU      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Components

- **Engine**: High-performance Rust backend with SIMD optimizations
- **Bindings**: PyO3-based Python bindings for seamless interop
- **Python API**: Familiar PyTorch-like interface for ease of use

## Examples

### Simple Neural Network

```python
import minitensor as mt
from minitensor import nn, optim

# Create a simple classifier
model = nn.Sequential([
    nn.DenseLayer(784, 128),
    nn.ReLU(),
    nn.DenseLayer(128, 10),
])

# Initialize model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(0.001, betas=(0.9, 0.999), epsilon=1e-8)
```

### Training Loop

```python
import minitensor as mt
from minitensor import nn, optim

# Synthetic data: y = 3x + 0.5
x = mt.randn(64, 1)
y = 3 * x + 0.5

# Model, loss, optimizer
model = nn.DenseLayer(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    pred = model(x)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 20 == 0:
        loss_val = float(loss.numpy().ravel()[0])
        print(f"Epoch {epoch+1:03d} | Loss: {loss_val:.4f}")
```

## Development & Testing

The Python package is a thin wrapper around the compiled Rust engine. Whenever
you make changes under `engine/` or `bindings/`, rebuild the extension so the
Python layer talks to the latest native implementation before running tests:

```bash
# 1. Build the extension in editable mode (release profile)
pip install -e . --config-settings=--release

# 2. Run the Rust unit and integration tests
cargo test

# 3. Execute the Python test
pytest
```

Running `pip install -e .` refreshes the `minitensor._core` module. Skipping this
step leaves Python bound to an older shared library, which can surface as missing
attributes or stale logic when validating changes locally.

### Code Style

- **Rust**: Follow `rustfmt` and `clippy` recommendations
- **Python**: Use `black` and `isort` for formatting

## Performance

Minitensor is designed for performance:

- **Memory Efficient**: Zero-copy operations where possible
- **SIMD Optimized**: Vectorized operations for maximum throughput
- **GPU Ready**: CUDA and Metal backend support (coming soon)
- **Parallel**: Multi-threaded operations for large tensors

## Citation

If you use minitensor in your work and wish to refer to it, please use the following BibTeX entry.

```bibtex
@misc{sarkar2026minitensorlightweighthighperformancetensor,
      title={MiniTensor: A Lightweight, High-Performance Tensor Operations Library}, 
      author={Soumyadip Sarkar},
      year={2026},
      eprint={2602.00125},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.00125}, 
}
```

## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by [PyTorch's design and API](https://pytorch.org)
- Built with [Rust's](https://www.rust-lang.org) performance and safety
- Powered by [PyO3](https://github.com/PyO3/pyo3) for Python integration
