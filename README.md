<div align="center">

![MiniTensor Logo](docs/_static/img/minitensor-small.png)

</div>
<h3 align="center">
A lightweight, high-performance tensor operations library inspired by <a href="https://github.com/pytorch/pytorch">PyTorch</a> with Rust backend.
</h3>

---

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![rustc 1.85+](https://img.shields.io/badge/rustc-1.85+-blue.svg?style=flat-square&logo=rust&logoColor=white)](https://rust-lang.github.io/rfcs/2495-min-rust-version.html)
[![Test Linux](https://github.com/neuralsorcerer/minitensor/actions/workflows/test_ubuntu.yml/badge.svg)](https://github.com/neuralsorcerer/minitensor/actions/workflows/test_ubuntu.yml?query=branch%3Amain)
[![Test Windows](https://github.com/neuralsorcerer/minitensor/actions/workflows/test_windows.yml/badge.svg)](https://github.com/neuralsorcerer/minitensor/actions/workflows/test_windows.yml?query=branch%3Amain)
[![Test MacOS](https://github.com/neuralsorcerer/minitensor/actions/workflows/test_macos.yml/badge.svg)](https://github.com/neuralsorcerer/minitensor/actions/workflows/test_macos.yml?query=branch%3Amain)
[![license](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square)](./LICENSE)

</div>

## Features

- **High Performance**: Rust backend for maximum speed and memory efficiency
- **Python-Friendly**: Familiar PyTorch-like API for easy adoption
- **Neural Networks**: Complete neural network layers and optimizers
- **NumPy Integration**: Seamless interoperability with NumPy arrays
- **Automatic Differentiation**: Built-in gradient computation for training
- **Extensible**: Modular design for easy customization and extension

## Quick Start

### Installation

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
> *Note:* `pip install -e .` builds a debug version by default; pass `--config-settings=--release` for a release build.

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
optimizer = optim.Adam(0.001)

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
sgd = optim.SGD(0.01, momentum=0.9)         # SGD with momentum
adam = optim.Adam(0.001, betas=(0.9, 0.999)) # Adam optimizer
rmsprop = optim.RMSprop(0.01, alpha=0.99)   # RMSprop optimizer
```

## Architecture

Minitensor is built with a modular architecture:

```
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
optimizer = optim.Adam(0.001, betas=(0.9, 0.999))
```

### Training Loop (Conceptual)

```python
# Training data
train_loader = get_data_loader()  # Your data loading logic

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item():.6f}')
```

### Code Style

- **Rust**: Follow `rustfmt` and `clippy` recommendations
- **Python**: Use `black` for formatting and `mypy` for type checking

## Performance

Minitensor is designed for performance:

- **Memory Efficient**: Zero-copy operations where possible
- **SIMD Optimized**: Vectorized operations for maximum throughput
- **GPU Ready**: CUDA and Metal backend support (coming soon)
- **Parallel**: Multi-threaded operations for large tensors

## Citation

If you use minitensor in your work and wish to refer to it, please use the following BibTeX entry.

```bibtex
@software{minitensor2025,
  author = {Soumyadip Sarkar},
  title = {MiniTensor: A lightweight, high-performance tensor operations library},
  url = {http://github.com/neuralsorcerer/minitensor},
  year = {2025},
}
```

## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by [PyTorch's design and API](https://pytorch.org)
- Built with [Rust's](https://www.rust-lang.org) performance and safety
- Powered by [PyO3](https://github.com/PyO3/pyo3) for Python integration
