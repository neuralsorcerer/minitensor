# Minitensor

A lightweight, high-performance deep learning library inspired by [PyTorch](https://pytorch.org) with Rust backend and Python bindings.

> [!CAUTION]
> This library is in active development stage. Things may break often. Use carefully.

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
```

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

# Create tensors
x = mt.zeros(3, 4)          # Zeros
y = mt.ones(2, 3)           # Ones
z = mt.randn(2, 2)          # Random normal
w = mt.from_numpy(np_array) # From NumPy

# Operations
result = x + y              # Element-wise addition
product = x.matmul(y)       # Matrix multiplication
mean_val = x.mean()         # Reduction operations
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
│                 │◄──►│                  │◄──►│                 │
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
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.DenseLayer(784, 128)
        self.fc2 = nn.DenseLayer(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(0.001)
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
@software{minitensor,
  author = {Soumyadip Sarkar},
  title = {MiniTensor: A lightweight, high-performance deep learning library.},
  url = {http://github.com/neuralsorcerer/minitensor},
  year = {2025},
}
```

## 📄 License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by PyTorch's design and API
- Built with Rust's performance and safety
- Powered by PyO3 for Python integration
