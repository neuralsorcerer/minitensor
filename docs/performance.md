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

## Optimization checklist

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
