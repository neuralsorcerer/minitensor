# Performance Benchmarks

This guide explains how to measure the runtime characteristics of the
Minitensor library and provides tips for obtaining the best performance.

## Benchmark script

Run the benchmark script to compare matrix multiplication speed against
popular frameworks:

```bash
python examples/performance_benchmark.py
```

You can also run `make benchmark` to execute any available benchmark suite.

The script will attempt to import PyTorch and TensorFlow. If either library is
missing, its benchmark will be skipped. Each benchmark multiplies two randomly
initialized matrices and reports the execution time.

## Optimization tips

- Use the GPU backends (CUDA, Metal, etc.) when available for large workloads.
- Keep tensors contiguous in memory before launching heavy operations.
- Prefer in-place operations to reduce memory allocations.
