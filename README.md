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
[![Documentation](https://github.com/neuralsorcerer/minitensor/actions/workflows/docs.yml/badge.svg)](https://github.com/neuralsorcerer/minitensor/actions/workflows/docs.yml?query=branch%3Amain)
[![License](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white)](./LICENSE)
[![codecov](https://codecov.io/github/neuralsorcerer/minitensor/graph/badge.svg?token=BNV0Z7SI6A)](https://codecov.io/github/neuralsorcerer/minitensor)
[![arXiv](https://img.shields.io/badge/arXiv-2602.00125-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2602.00125)
[![DOI:48550/arXiv.2602.00125](https://img.shields.io/badge/DOI-10.48550/arXiv.2602.00125-blue.svg)](https://doi.org/10.48550/arXiv.2602.00125)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/minitensor?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/minitensor)

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

MiniTensor can be installed either from PyPI or from source. Source installs
compile the Rust extension, so they require Python 3.10+, Rust/Cargo, and
maturin. The full, platform-aware instructions live in the
[installation guide](docs/installation.md).

**From PyPI:**

```bash
pip install minitensor
```

**From source with the installer (recommended for contributors):**

```bash
# Clone the repository
git clone https://github.com/neuralsorcerer/minitensor.git
cd minitensor

bash install.sh
```

The installer creates `.venv` by default, installs maturin (with `patchelf` on
Linux), installs Rust with rustup if needed, builds a release extension, and
verifies the import.

**Manual source install:**

```bash
python -m pip install 'maturin[patchelf]'  # Linux; use `maturin` on macOS/Windows
maturin develop --release
```

For contributor tooling and editable builds, use the `dev` extra:

```bash
python -m pip install -e '.[dev]'
pre-commit install
```

> _Note:_ Editable pip installs use the release profile configured in
> `pyproject.toml`; use `maturin develop --debug` when you intentionally need
> a debug build.

### Basic Usage

```python
import minitensor as mt
from minitensor import nn, optim

# Create tensors
mt.manual_seed(7)
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
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), epsilon=1e-8)

print(f"Model type: {type(model).__name__}")
print(f"Input shape: {x.shape}")
```

```text
Model type: Sequential
Input shape: Shape([32, 784])
```

## Documentation

MiniTensor ships documentation in [`docs/`](docs), starting with the
[documentation index](docs/index.md). Key guides include the
[installation guide](docs/installation.md), [API reference](docs/api_reference.md),
[development guide](docs/development.md), [custom operations guide](docs/custom_operations.md),
[plugin system guide](docs/plugin_system.md), and [performance guide](docs/performance.md).
For a runtime overview of what's available, use the introspection helpers below.

```python
import minitensor as mt

submodules = mt.available_submodules()
nn_api = mt.list_public_api()["nn"]
loss_hits = mt.search_api("loss")
ce_desc = mt.describe_api("nn.CrossEntropyLoss")

print(f"has submodules: {len(submodules) > 0}")
print(f"has nn API entries: {len(nn_api) > 0}")
print(f"loss search non-empty: {len(loss_hits) > 0}")
print(f"CrossEntropyLoss described: {'CrossEntropyLoss' in ce_desc}")
```

```text
has submodules: True
has nn API entries: True
loss search non-empty: True
CrossEntropyLoss described: True
```

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
result = x + y                             # Element-wise addition
product = x.matmul(y.transpose(0, 1))      # Matrix multiplication
mean_val = x.mean()                        # Reduction operations
nan_median = mt.nanmedian(mt.Tensor([1.0, np.nan, 5.0]))
std_by_row = w.astype("float32").std(dim=1, unbiased=False)
var_all = w.astype("float32").var(dim=(0, 1), unbiased=False)
max_val = x.max()                          # -inf for empty or all-NaN tensors
min_vals, min_idx = x.min(dim=1)           # Returns values & indices; empty dims yield (inf, 0)
close = mt.allclose([0.0, float("inf")], [-0.0, float("inf")])
exact = mt.array_equal([1, 2], mt.tensor([1.0, 2.0], dtype="float32"))
broadcasted_shape = mt.broadcast_shapes(x.shape, (1, 4))
broadcasted_x, broadcasted_row = mt.broadcast_tensors(
    mt.ones(3, 1), mt.Tensor([[1.0, 2.0, 3.0, 4.0]])
)
row = mt.atleast_2d(mt.Tensor([1.0, 2.0, 3.0]))

print(result.shape)                        # Shape([3, 4])
print(product.shape)                       # Shape([3, 3])
print(float(mean_val.numpy().ravel()[0]))  # 0.0
print(float(nan_median.numpy().ravel()[0]))  # 3.0
print(std_by_row.shape)                    # Shape([2])
print(float(var_all.numpy().ravel()[0]))   # 2.9166667
print(float(max_val.numpy().ravel()[0]))   # 0.0
print(min_idx.numpy())                     # [0 0 0]
print(close, exact)                        # True True
print(broadcasted_shape)                    # (3, 4)
print(broadcasted_x.shape, broadcasted_row.shape)  # Shape([3, 4]) Shape([3, 4])
print(row.shape)                           # Shape([1, 3])
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

print(type(dense).__name__, type(conv).__name__, type(relu).__name__, type(ce).__name__)
```

```text
DenseLayer Conv2d ReLU CrossEntropyLoss
```

#### Optimizers

```python
from minitensor import nn, optim

# Optimizers
model = nn.DenseLayer(10, 5)
params = model.parameters()

sgd = optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.0, nesterov=False)
adam = optim.Adam(params, lr=0.001, betas=(0.9, 0.999), epsilon=1e-8, weight_decay=0.0)
adamw = optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), epsilon=1e-8, weight_decay=0.01)
rmsprop = optim.RMSprop(params, lr=0.01, alpha=0.99, epsilon=1e-8, weight_decay=0.0, momentum=0.0)

print(type(sgd).__name__, type(adam).__name__, type(adamw).__name__, type(rmsprop).__name__)
```

```text
SGD Adam AdamW RMSprop
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
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), epsilon=1e-8)

print(type(model).__name__, type(optimizer).__name__)
```

```text
Sequential Adam
```

### Training Loop

```python
import minitensor as mt
from minitensor import nn, optim

# Synthetic data: y = 3x + 0.5 + noise
mt.manual_seed(7)
x = mt.randn(256, 1)
noise = 0.1 * mt.randn(256, 1)
y = 3 * x + 0.5 + noise

# Model, loss, optimizer
model = nn.DenseLayer(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)

for epoch in range(100):
    pred = model(x)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 20 == 0:
        loss_val = float(loss.numpy().ravel()[0])
        w = float(model.weight.numpy().ravel()[0])
        b = float(model.bias.numpy().ravel()[0])
        print(f"Epoch {epoch+1:03d} | Loss: {loss_val:.4f} | w: {w:.3f} | b: {b:.3f}")
```

```text
Epoch 020 | Loss: 0.2520 | w: 2.545 | b: 0.407
Epoch 040 | Loss: 0.0150 | w: 2.934 | b: 0.485
Epoch 060 | Loss: 0.0103 | w: 2.988 | b: 0.498
Epoch 080 | Loss: 0.0102 | w: 2.995 | b: 0.500
Epoch 100 | Loss: 0.0102 | w: 2.996 | b: 0.501
```

## Development & Testing

The Python package is a thin wrapper around the compiled Rust engine, so native
and Python changes should be validated in a deterministic order. See the
[development guide](docs/development.md) and
[installation guide](docs/installation.md#contributor-setup) for full setup and
troubleshooting details.

```bash
# 1) One-time contributor setup (installs dev tooling + editable release extension)
python -m pip install -e '.[dev]'
pre-commit install

# 2) Rebuild the extension after changes under engine/ or bindings/
python -m pip install -e .

# 3) Run Rust unit/integration tests
cargo test --workspace --all-targets

# 4) Run Python tests with warnings treated as errors by project config
python -m pytest

# 5) Run formatting/lint hooks
pre-commit run --all-files
```

Notes:
- Use `python -m pip` so installs target the same interpreter used for `python -m pytest`.
- The `dev` extra installs `black[jupyter]`, matching the pre-commit Black hook and
  avoiding missing-Jupyter warnings when checking notebooks.
- Step 2 is only required when Rust or PyO3 bindings changed; pure-Python/docs edits
  can skip it.
- Keep Step 1 as one-time setup unless dev dependencies change.

### Code Style

- **Rust**: Follow `rustfmt` and `clippy` recommendations
- **Python**: Use `black[jupyter]` and `isort` for formatting Python files and notebooks

## Performance

Minitensor is designed for performance:

- **Memory Efficient**: Zero-copy operations where possible
- **SIMD Optimized**: Vectorized operations for maximum throughput
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
