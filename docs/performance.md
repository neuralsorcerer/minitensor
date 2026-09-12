# Performance Benchmarks

This guide explains how to measure MiniTensor performance, compare it with other
frameworks, and avoid common benchmarking mistakes.

## Benchmark commands

Run the benchmark suite, which covers the operations a training step is
actually made of:

```bash
python examples/benchmark_suite.py
python examples/benchmark_suite.py --json before.json   # then again, after a change
python examples/benchmark_suite.py --compare before.json after.json
```

Or the framework comparison, which times one matmul against PyTorch and
TensorFlow when they are installed:

```bash
python examples/performance_benchmark.py
```

You can also use the Makefile target:

```bash
make benchmark
```

`performance_benchmark.py` skips any comparison framework it cannot import
rather than failing. Note that it skips the MiniTensor half the same way, so
a run that prints only "Skipping" lines is reporting an import problem, not a
result. `benchmark_suite.py` adds the repository root to `sys.path` itself, so
it works when run by path from a source checkout.

## Conversion to and from NumPy

NumPy is already a hard dependency of the extension, and the conversion paths
lean on it rather than reimplementing what it does in C:

- Building a tensor from a Python list or tuple goes through `numpy.asarray`,
  both to infer the dtype and to read the values. Walking the object graph in
  Rust instead cost about 880ns per element against NumPy's ~16ns. Anything
  NumPy cannot type -- ragged nesting, object arrays, strings -- falls back to
  the element-wise walk so its error messages are unchanged.
- `.numpy()` copies the buffer directly when the tensor is contiguous, which it
  almost always is, instead of walking a multi-dimensional index.

Measured on 20k-element sequences and a 100k-element float32 tensor:

| path | before | after |
| --- | --- | --- |
| `as_tensor(flat list)` | 17.6 ms | 0.86 ms |
| `as_tensor(nested list)` | 17.4 ms | 0.92 ms |
| `Tensor(list)` | 17.6 ms | 0.42 ms |
| `.numpy()` | 232 us | 9.5 us |

`.numpy()` is now within noise of a plain `ndarray.copy()` (9.4us), which is
the floor for a copy of that size.

Note the dtype rules for sequences remain this library's, not NumPy's: a list
of Python floats infers `float32`, the configured default, where NumPy infers
`float64`.

## Where the time goes

Measured against NumPy on a 4-core x86-64 container, float32, release build.
Ratios below 1 mean MiniTensor is faster. They are ratios against a moving
target: NumPy's elementwise and transcendental kernels have gained SIMD
implementations over recent releases, so the same MiniTensor build measures
differently against a different NumPy. Re-measure before drawing a conclusion
from any single figure, and prefer the 1M column — at 4K and 64K the per-call
costs below are large enough that run-to-run variance on a shared machine can
swamp the difference being measured.

| op | 4K | 64K | 1M |
| --- | --- | --- | --- |
| `add` / `mul` | 0.85x | 0.62x | 0.81x |
| `relu` | 0.35x | 0.42x | 0.41x |
| `abs`, `sqrt` | 0.86x | 0.99x | 0.51x |
| `sum` | 0.30x | 0.62x | 0.19x |
| `exp` | 1.41x | 0.80x | 0.51x |
| `tanh` | 2.56x | 1.45x | 0.94x |
| `sigmoid` | 0.80x | 0.46x | 0.20x |
| `log` | 2.07x | 0.98x | 0.67x |
| `sin` | 2.09x | 0.91x | 0.61x |

The float32 transcendentals used to be the outlier -- `tanh` was 12-19x slower
than NumPy, because this library called the scalar libm routine per element
while NumPy ships hand-written SIMD kernels. They now have their own block
kernels (`engine/src/ops/simd/transcendental.rs`), compiled once per instruction
set and selected at run time, so at a megabyte and up they are at or ahead of
NumPy.

What remains is a per-call overhead of a few microseconds, which is why the 4K
column is worse than the 1M column for every transcendental and better for
none. At 4096 elements the work is a few microseconds either way, so the
crossing into the Python binding and the output allocation dominate; by 1M they
are noise. Read the 4K column as the cost of calling an op, not the cost of the
op.

Accuracy did not pay for the speed. `tanh`, `exp`, `expm1`, `sinh`, `cosh`,
`log`, `sin`, `cos` and `tan` are bit-identical to the correctly-rounded
float64 value on **all 2^32 float32 inputs**, checked exhaustively; `erf`,
`erfc` and `log1p` carry a stated budget of at most one ulp on a bounded number
of inputs. The tests are `#[ignore]`d because a sweep takes minutes:

```bash
cargo test --release -p engine -- --ignored --nocapture transcendental
```

`sort` deserves a note because it is easy to mis-measure: `mt.sort` always
returns values *and* indices, so the comparable NumPy operation is `argsort`
plus a gather, not `np.sort`. Against that it is roughly at parity — 1.1x to
1.4x at 1M depending on the machine and the NumPy build. Against `np.sort`
alone it looks several times slower again (9x here, 25x on an older NumPy), but
that is comparing different work, and the gap moves with whatever SIMD sort the
installed NumPy ships rather than with anything in this library.

## Where a dense product is computed

A single large `float32` or `float64` product does not run on this library's
own kernel. It runs on the BLAS that `numpy` already brought -- OpenBLAS on the
wheels -- on the buffers the engine allocated, with nothing copied on the way
in or out. Every install has `numpy` in it, every install of `numpy` has a
tuned BLAS in it, and that BLAS is per-architecture assembly selected at load
time from a table of implementations. A portable Rust kernel does not close
that gap.

Measured against NumPy on a 4-core x86-64 container, threads pinned to 4,
release build. Ratios above 1 mean MiniTensor is slower:

| product | dtype | own kernel | delegated |
| --- | --- | --- | --- |
| 256x256 @ 256x256 | float32 | 3.42x | 1.36x |
| 512x512 @ 512x512 | float32 | 2.20x | 1.34x |
| 1024x1024 @ 1024x1024 | float32 | 1.86x | 1.15x |
| 2048x2048 @ 2048x2048 | float32 | 1.75x | 1.12x |
| 512x2048 @ 2048x512 | float32 | 2.05x | 1.06x |
| 1024x1024 @ 1024x1024 | float64 | 1.80x | 1.13x |
| 2048x2048 @ 2048x2048 | float64 | 1.85x | 1.09x |

The largest wins are not in that table. A matrix-vector product is the shape a
blocked kernel is worst at and the shape a BLAS has a routine written for:
`1x1024 @ 1024x1024` goes from 874us to 109us, and `1024x1024 @ 1024x1` from
2086us to 36us -- 8x and 58x.

`nn.Linear` goes the same way. Its weight is stored `[out, in]` for a product
that wants `[in, out]`, and it is handed over that way rather than transposed
into a copy first, so the saving the fused forward exists for is kept:

| `rows x in x out` | dtype | own kernel | delegated |
| --- | --- | --- | --- |
| 8 x 256 x 256 | float32 | 136us | 52us |
| 64 x 512 x 512 | float32 | 618us | 350us |
| 16 x 1024 x 1024 | float32 | 888us | 493us |
| 256 x 768 x 768 | float32 | 2259us | 1808us |
| 32 x 4096 x 4096 | float32 | 13241us | 10676us |

### What stays on the engine's own kernel

Handing the product over costs three array headers and a call, a little over a
microsecond, and for a whole class of shapes a BLAS has nothing to offer
anyway. Those go to the engine's kernel, which is measurably the right answer
for them:

- **Anything under ~1.3e5 multiply-accumulates.** A 48x48x48 product finishes
  inside the call overhead. Below that the engine's kernel wins outright.
- **A contraction shorter than 32.** Such a product is bound by writing its
  answer rather than by computing it, and there NumPy's `matmul` measured
  1.4-2.9x *slower*: `256x16 @ 16x256` is 44us on the engine's kernel and 83us
  delegated, and the outer product `512x1 @ 1x512` is 148us against 721us.
- **Batched products.** Several matrices at once already fill the thread pool
  with one whole product per worker, which is what delegating was for. These
  measure between 0.95x and 1.5x of NumPy without it.
- **Integers.** Neither library sends an integer product to a BLAS.
- **Anything running inside the engine's own thread pool.** A worker that
  stopped to take the interpreter lock would be waiting on the thread that
  holds it and is waiting on the worker.
- **A build with `--features blas`.** That build linked a BLAS into the engine
  deliberately, and calls it directly.

Where a product is computed is a performance decision that depends on its shape
and dtype, and it may change. The answer does not: both paths agree with a
float64 reference to the precision of the dtype asked for. They do not agree
bit for bit, and neither does a BLAS with itself across shapes -- it dispatches
on all three extents and on how each operand is stored, so reordering the
accumulation of `k` is normal.

## Choosing a GEMM backend

Under `--features blas`, matrix multiplication routes to an installed OpenBLAS
inside the engine instead of to `numpy`'s. Without it the engine's own kernel
is `matrixmultiply` -- pure Rust, no system library -- and it is what runs for
everything the section above lists. Which is faster depends on the machine and
the size, so measure rather than assume. On the machine these docs were last
measured on (x86-64, OpenBLAS 0.3 from `libopenblas-dev`), square float32
matmul came out:

| size | matrixmultiply, older | matrixmultiply, now | OpenBLAS, older |
| --- | --- | --- | --- |
| 128 | 49 GFLOP/s | 69 GFLOP/s | 55 GFLOP/s |
| 256 | 101 GFLOP/s | 211 GFLOP/s | 156 GFLOP/s |
| 512 | 150 GFLOP/s | 265 GFLOP/s | 226 GFLOP/s |
| 1024 | 189 GFLOP/s | 365 GFLOP/s | 287 GFLOP/s |

The middle column is current; the outer two are from when this section was
first measured. `matrixmultiply` is no longer built with its own `threading`
feature -- a single product is now divided across rayon by `gemm_f32` itself,
along whichever output axis is longer -- and that moved the pure-Rust path past
the OpenBLAS figures previously recorded here.

Those OpenBLAS numbers were not re-measured, so the honest reading is that the
comparison needs redoing on your machine rather than that one backend now wins.
Run both and see:

```bash
cargo run --release --example gemm_benchmark
cargo run --release --features blas --example gemm_benchmark
```

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

Over 1000 forward passes of the model above, the first form leaves 5005 nodes
in the graph -- five per pass, climbing linearly and never released -- while the
second leaves none, as does a normal training loop that calls `backward()` and
`optimizer.step()`.

What that costs in memory depends on the model, since each node holds the
tensors its backward would need: the 4-to-8-to-2 network above grows about
2.7 MB over those 1000 passes, but a network with megabyte activations grows by
megabytes per pass. The node count is the part that is stable enough to check,
and `autograd_graph_size()` returns it:

```python
import minitensor as mt

nodes, gradients = mt.autograd_graph_size()
```

A count that rises across iterations of a loop is this bug; one that returns to
the same value each iteration is not.

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

- [`examples/benchmark_suite.py`](../examples/benchmark_suite.py) — benchmarks
  for the paths a training step spends its time in, with `--json` and
  `--compare` for diffing two builds. Its module docstring covers the
  measurement pitfalls specific to this workload.
- [`examples/performance_benchmark.py`](../examples/performance_benchmark.py) —
  bundled matrix-multiplication benchmark against PyTorch and TensorFlow.
- [`Makefile`](../Makefile) — project convenience targets.
- [Development guide](development.md) — validation and contributor workflow.
