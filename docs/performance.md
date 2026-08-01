# Performance Benchmarks

This guide explains how to measure MiniTensor performance, compare it with other
frameworks, and avoid common benchmarking mistakes.

## Benchmark commands

Run the bundled Python benchmark from the repository root:

```bash
python examples/performance_benchmark.py
```

You can also use the Makefile target:

```bash
make benchmark
```

The benchmark script attempts to import optional comparison frameworks such as
PyTorch and TensorFlow. Missing optional frameworks are skipped rather than
failing the MiniTensor benchmark.

## Choosing a GEMM backend

Matrix multiplication runs through `matrixmultiply` by default -- pure Rust, no
system library. The `blas` feature routes it to an installed OpenBLAS instead.
Which is faster depends on the machine and the size, so measure rather than
assume:

```bash
cargo run --release --example gemm_benchmark
cargo run --release --features blas --example gemm_benchmark
```

On the machine these docs were last measured on (x86-64, OpenBLAS 0.3 from
`libopenblas-dev`), square float32 matmul came out:

| size | matrixmultiply | OpenBLAS | speedup |
| --- | --- | --- | --- |
| 128 | 49 GFLOP/s | 55 GFLOP/s | 1.1x |
| 256 | 101 GFLOP/s | 156 GFLOP/s | 1.5x |
| 512 | 150 GFLOP/s | 226 GFLOP/s | 1.5x |
| 1024 | 189 GFLOP/s | 287 GFLOP/s | 1.5x |

The gap widens with size and is small enough at 128 that the extra build
dependency may not be worth it for workloads dominated by small matrices. These
are single-run figures from one machine; treat them as a reason to run the
benchmark, not as a result to quote.

## Recommended benchmark setup

For stable measurements:

1. Build an optimized extension before timing native operations:

   ```bash
   maturin develop --release
   ```

2. Close unrelated CPU- and GPU-heavy processes.
3. Run each benchmark more than once and compare medians, not a single run.
4. Keep input sizes, dtypes, devices, and warmup behavior identical across
   frameworks.
5. Record hardware, operating system, Python version, Rust version, MiniTensor
   version, and backend feature flags with each result.

## Interpreting results

Performance numbers are only meaningful when the workload matches your use case.
Small tensors can be dominated by Python call overhead and allocation costs,
while large tensors are more likely to show the Rust engine, SIMD, memory layout,
and backend behavior.

When comparing with another library, verify that both implementations use the
same:

- dtype and shape;
- device/backend;
- operation semantics;
- thread count or backend scheduling policy;
- warmup and synchronization points;
- data-transfer policy between host and accelerator memory.

## Wrap inference in `no_grad()`

Any operation on a tensor that requires gradients records a node in the
autograd graph, and that graph is released when a backward pass consumes it or
when the optimizer steps. A loop that only runs forward passes does neither, so
the graph keeps growing:

```python
import minitensor as mt

model = mt.nn.Sequential(
    [mt.nn.DenseLayer(4, 8), mt.nn.ReLU(), mt.nn.DenseLayer(8, 2)]
)
validation_data = [mt.Tensor([[0.1, 0.2, 0.3, 0.4]]) for _ in range(3)]

# Grows without bound -- the model's parameters require gradients, so every
# forward records a graph that is never consumed.
for batch in validation_data:
    predictions = model(batch)

# Bounded -- nothing is recorded.
with mt.no_grad():
    for batch in validation_data:
        predictions = model(batch)
```

Measured over 1000 forward passes of a small MLP, the first form grew resident
memory by about 215 MB and kept climbing linearly; the second was flat, as was a
normal training loop that calls `backward()` and `optimizer.step()`.

`model.eval()` does **not** imply this. It switches dropout and batch norm to
inference behaviour and nothing else, so use both together:

```python
import minitensor as mt

model = mt.nn.Sequential([mt.nn.DenseLayer(4, 2), mt.nn.Dropout(0.5)])
batch = mt.Tensor([[0.1, 0.2, 0.3, 0.4]])

model.eval()
with mt.no_grad():
    predictions = model(batch)
```

Use `no_grad()` for validation loops, metric computation, and any inference
path. If you need gradients again inside such a block, `enable_grad()` re-enables
recording locally.

## Optimization checklist

- Wrap inference and validation loops in `no_grad()`.
- Prefer vectorized tensor operations over Python loops.
- Keep tensors contiguous before expensive operations when possible.
- Reuse tensors and avoid unnecessary conversions to and from NumPy.
- Use GPU backends for workloads large enough to amortize transfer and launch
  overhead.
- Batch many small operations into fewer larger operations when practical.
- Run release builds for performance measurements; debug builds are for
  correctness debugging, not speed.

## Profiling pointers

Start with the highest-level benchmark that reproduces the slowdown. Then narrow
it down to a specific operation, input shape, dtype, and backend. For Rust-side
work, combine targeted Rust tests or examples with standard profilers available
on your platform. For Python-side work, compare the cost of tensor creation,
operation execution, NumPy conversion, and training-loop overhead separately.

## Related files

- [`examples/performance_benchmark.py`](../examples/performance_benchmark.py) —
  bundled matrix-multiplication benchmark.
- [`Makefile`](../Makefile) — project convenience targets.
- [Development guide](development.md) — validation and contributor workflow.
