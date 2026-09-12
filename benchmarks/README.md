# Where the work goes, and why

`engine::ops::linalg::gemm_provider` sends dense matrix products to the BLAS
that NumPy brought and leaves every other kernel on the engine's own. That split
is a measurement, not a preference, and `dispatch_bench.py` is how it is
re-taken:

```
python benchmarks/dispatch_bench.py                      # everything
python benchmarks/dispatch_bench.py --filter gemm        # one family
python benchmarks/dispatch_bench.py --native             # engine kernels only
python benchmarks/dispatch_bench.py --json results.json  # keep the numbers
```

Running it with and without `--native` is how the two paths are compared on one
build: `--native` pushes the delegation thresholds out of reach so the same
sweep measures the engine's kernels.

Every case times the kernel only. Operands are built once, outside the loop, in
each library's own container, so no conversion is charged to either side. Timing
is min-of-repeats: for a deterministic kernel the spread is scheduler noise, and
the minimum has least of it.

All numbers below are from one 4-core x86-64 container, NumPy 2.4.6 against
OpenBLAS 0.3.31, float32 unless stated. **They are not portable.** The shape of
the result should hold anywhere; the crossovers will not.

## What is delegated

A single product, and a batch of them. Against `numpy.matmul`, engine kernels
(`--native`) and delegated:

| shape | native | delegated | NumPy |
|---|---|---|---|
| 256×256 @ 256×256 | 374.9 µs | 122.0 µs | 100.1 µs |
| 512×512 @ 512×512 | 3.95 ms | 733.5 µs | 684.7 µs |
| 1024×1024 @ 1024×1024 | 9.32 ms | 5.03 ms | 4.80 ms |
| 2048×512 @ 512×512 | 4.49 ms | 2.52 ms | 2.26 ms |
| 64×288 @ 288×900 | 669.4 µs | 165.7 µs | 127.3 µs |

`matrixmultiply` is a good portable GEMM. It is not per-microarchitecture
assembly with operand packing, and against one of those it loses by 1.8–5.4×.
Linking a BLAS into the wheel would close that and cost tens of megabytes;
borrowing the one already loaded costs nothing.

### The batched case, and why it needed its own request

A stack of matrices used to reach the provider one matrix at a time or not at
all — the batch axis is split across the thread pool, and a provider is withheld
from worker threads, so the largest batches were never offered. Handing them
over singly is not the fix either: at a batch of 256 the per-matrix crossing
costs more than the whole product. So the request carries a `batch` and the
stack crosses once.

| shape | before | after | NumPy |
|---|---|---|---|
| 8×128×128 | 399.1 µs | **190.8 µs** | 155.3 µs |
| 64×64×64 | 909.4 µs | **394.4 µs** | 346.2 µs |
| 3×128×128 | 234.5 µs | **76.4 µs** | 64.6 µs |
| 2×512×512 | 2.91 ms | **1.41 ms** | 1.29 ms |
| 16×256×256 | 2.53 ms | **2.08 ms** | 1.69 ms |

1.2–3.1× faster, and the ratio against NumPy went from 0.27–0.66 to 0.81–0.98.
The residual is the cost of being a tensor library rather than an array: one
PyO3 call, one output allocation, three array headers.

### What was tried and did not pay: convolution

Convolution is im2col plus one large dense product, so the product looks like it
belongs on the same path. Measured across eight shapes, delegating it gives
0.74, 0.76, 0.78, 0.98, 0.99, 1.01, 1.01 and 1.71 — a geometric mean of **0.94×**,
a loss.

The reason is that a conv GEMM is not an isolated product. The forward lowers
and multiplies one cache-sized block at a time, and the lowering already fills
the thread pool; handing each block's GEMM to OpenBLAS puts a second thread pool
on the same four cores and pays a crossing per block. The one shape that wins
(1×512×7×7 → 512 channels, a single block with a deep contraction) is one point
out of eight, and a rule fitted to it would be a rule about this machine. So
convolution stays native, and the code carries no special case.

## What is not delegated

Ratios above 1.0 mean the engine's kernel is faster. Fused activations, where
the win is that no intermediate is ever written:

| op | 1M elements | 16M elements |
|---|---|---|
| `gelu` | 3.0× | **5.5×** |
| `sigmoid` | 2.9× | 3.3× |
| `softmax` | 1.9× | 2.7× |
| `relu` | 3.9× | 1.8× |
| `sqrt` | 1.7× | 1.2× |

NumPy computes `gelu` as roughly ten passes over the array, each allocating and
each reading its input back from memory. The kernel is one pass: 22.9 ms against
124.7 ms at 16M elements.

Reductions, where the win is parallelism plus a single pass:

| op | 16M elements |
|---|---|
| `sum` | 5.9× |
| `mean` | 4.5× |
| `sum(axis=0)` | 2.5× |
| `max` | 2.4× |
| `argmax` | 1.2× |

Where NumPy wins and the op stays native anyway: `tanh`, `log` and `exp` are
within a few percent either way, because NumPy dispatches them to a vectorised
transcendental library; and `norm` is 2–3× faster in NumPy. Moving any of them
across would cost a Python call on every activation in a network, which is more
than the margin is worth — and `norm` is worth a look at the kernel rather than
at the boundary.

## Reading a run

`verdict` is `rust`, `numpy` or `tie`, with a 5% dead band: a difference smaller
than that is inside the run-to-run spread and is not a reason to move an
operation. The per-family geometric mean at the bottom is the summary worth
watching after a change.
