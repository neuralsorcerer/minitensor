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

## A note on the harness itself

The batch size this script times over was too small once, and it is worth
recording because of what it did rather than what it was.

Batches grow until one reaches `_MIN_BATCH_SECONDS`. That was 50 µs — enough to
beat the clock, which is the obvious thing to size it against. But any kernel
slower than 50 µs per call then satisfied it on the first try, with a batch of
**one**, and so did every repeat. Between those single calls sat Python loop
overhead, which is long enough for rayon's workers to park, so each measured
call woke the pool again and the wake landed inside the measurement. A `gelu`
doing 75 µs of real work read as 261 µs.

It biased one band and left the rest alone, which is why it survived: under
~50 µs a batch holds many calls and the pool stays hot; over a few milliseconds
the pool stays hot *within* one call; in between, every call paid. It also
biased in one direction — NumPy is single-threaded for these kernels and never
paid the wake — so it made this library look slower than it is, at the sizes
where the comparison is most interesting.

Across the sweep it showed as an unexplained dip at 100k elements in almost
every operation at once: `gelu` reading 3.09×, 0.65×, 3.44× as the size grew,
which is not a shape any kernel has. That is the tell worth remembering — a
whole column moving together is a fact about the harness, not about the code.

`_MIN_BATCH_SECONDS` is 20 ms now, which is also what a real workload looks
like: calls back to back on a pool that is already awake. Every number below is
from after the fix.

## What is delegated

Dense products, single and batched. `--native` against delegated, and NumPy for
scale:

| shape | native | delegated | NumPy | |
|---|---|---|---|---|
| 128×128 @ 128×128 | 76.6 µs | 33.7 µs | 26.5 µs | 2.3× |
| 256×256 @ 256×256 | 409.5 µs | 145.2 µs | 121.2 µs | 2.8× |
| 512×512 @ 512×512 | 1.44 ms | 802.9 µs | 713.1 µs | 1.8× |
| 1024×1024 @ 1024×1024 | 9.33 ms | 5.23 ms | 4.95 ms | 1.8× |
| 1×4096 @ 4096×4096 | 4.81 ms | 1.03 ms | 906.4 µs | 4.7× |
| 2048×512 @ 512×512 | 4.38 ms | 2.54 ms | 2.27 ms | 1.7× |

`matrixmultiply` is a good portable GEMM. It is not per-microarchitecture
assembly with operand packing, and against one of those it loses by these
margins. Linking a BLAS into the wheel would close that and cost tens of
megabytes; borrowing the one already loaded costs nothing. The residual gap to
raw NumPy is the cost of being a tensor library rather than an array: one PyO3
call, one output allocation, three array headers.

### The batched case, and why it needed its own request

A stack of matrices used to reach the provider one matrix at a time or not at
all — the batch axis is split across the thread pool, and a provider is withheld
from worker threads, so the largest batches were never offered. Handing them
over singly is not the fix either: at a batch of 256 the per-matrix crossing
costs more than the whole product. So the request carries a `batch` and the
stack crosses once.

| shape | native | delegated | NumPy | |
|---|---|---|---|---|
| 8×128×128 | 433.6 µs | 256.7 µs | 211.1 µs | 1.7× |
| 256×32×32 | 335.3 µs | 253.2 µs | 210.3 µs | 1.3× |
| 64×64×64 | 430.4 µs | 373.4 µs | 332.9 µs | 1.2× |

### What was tried and did not pay: convolution

Convolution is im2col plus one large dense product, so the product looks like it
belongs on the same path. Measured across eight shapes, delegating it gives
0.74, 0.76, 0.78, 0.98, 0.99, 1.01, 1.01 and 1.71 — a geometric mean of **0.94×**,
a loss.

A conv GEMM is not an isolated product. The forward lowers and multiplies one
cache-sized block at a time, and the lowering already fills the thread pool;
handing each block's GEMM to OpenBLAS puts a second thread pool on the same four
cores and pays a crossing per block. The one shape that wins (1×512×7×7 → 512
channels, a single block with a deep contraction) is one point out of eight, and
a rule fitted to it would be a rule about this machine. So convolution stays
native, and the code carries no special case.

## What is not delegated

Ratios above 1.0 mean the engine's kernel is faster.

Fused activations, where the win is that no intermediate is ever written:

| op | 1M elements | 16M elements |
|---|---|---|
| `gelu` | 3.7× | **5.3×** |
| `sigmoid` | 3.1× | 3.8× |
| `softmax` | 1.6× | 2.7× |
| `relu` | 4.2× | 1.8× |
| `sqrt` | 1.6× | 1.4× |
| `exp` | 1.5× | 1.2× |

NumPy computes `gelu` as roughly ten passes over the array, each allocating and
each reading its input back from memory. The kernel is one pass: 21.7 ms against
115.2 ms at 16M elements.

Reductions, where the win is parallelism plus a single pass:

| op | 16M elements |
|---|---|
| `mean` | 8.0× |
| `sum` | 5.6× |
| `sum(axis=0)` | 3.4× |
| `norm` | 3.3× |
| `max` | 3.3× |
| `argmax` | 1.2× |

`norm` was on the wrong side of this table until the sweep found it: 0.33× at
16M, while `sum` over the same data was 5.6×. It was not an accuracy tax — the
kernel is far more accurate than NumPy's there (7.6e-8 relative against 2.9e-5),
but so is `sum`, at a tenth of the cost. The 2-norm was computing `mul` into a
full-size temporary and then summing it, which at 16M means allocating 64MB,
writing it, and reading it straight back. The squares are wanted one at a time
by an accumulator and never again, so it is a dot product against the input
twice. Fused, the whole-tensor norm went 15377 µs → 1430 µs, and the row norms
of a 4000×4000 went 13613 µs → 1485 µs. Same accumulation, same accuracy, no
buffer.

Elementwise arithmetic, where the win is only parallelism and a fused output:

| op | 1M elements | 16M elements |
|---|---|---|
| `add` | 2.1× | 1.5× |
| `mul` | 2.1× | 2.0× |
| `div` | 1.9× | 1.7× |
| `maximum` | 1.8× | 1.6× |

## What still loses

At 16M elements there is **nothing** in float32 that NumPy computes faster, and
one thing in float64: `tanh`. It is not only a 16M problem — it loses at every
size, and worse at the small ones:

```
tanh float64        1,000     14.6us  vs    2.2us    0.15x
                  100,000    524.6us  vs  158.3us    0.30x
                1,000,000      4.48ms vs    1.56ms   0.35x
               16,000,000     85.64ms vs   43.83ms   0.51x
```

`ops::simd::transcendental` holds hand-vectorized kernels for `tanh`, `erf`,
`exp`, `log` and the rest — float32 only, computing internally in float64 and
rounding once. A float64 *output* cannot borrow that trick, and the reason is
worth stating precisely, because the obvious one is wrong.

It is **not** the polynomial. Extending the module's `expm1` Taylor series from
r¹² to r¹⁴ moves the worst case from 6 ulp to 4 and stops there. Feed the same
`tanh(x) = u/(u+2), u = expm1(2x)` form an `expm1` from libm — better than any
polynomial worth writing — and it still reaches **3 ulp**, with 109,151 of
2,000,001 sampled points over 1 ulp. The error is in the algebraic form: `u`
cancels against `+2` for negative `x`, and `2ⁿ·p + (2ⁿ − 1)` cancels again
inside the recombination. No polynomial fixes either.

That form is nevertheless exactly right for a float32 result, and the module
already proves it rather than assuming it: its `tanh` is bit-identical to
`(x as f64).tanh() as f32` on **all 2³² float32 inputs**, on every dispatch
path, checked exhaustively by an ignored test rather than sampled. One float32
ulp is 2²⁹ float64 ulps, so three of the latter disappear entirely in the
rounding — that headroom is exactly what the float32 kernel is spending. A
float64 output has none to spend.

The bar for float64 is higher than libm, not equal to it. Measured against a
200-bit `mpmath` reference over 23,997 points, NumPy's array `np.tanh` is
faithful to **≤ 1 ulp**, while the scalar `tanh` glibc gives the engine today
reaches **2 ulp**. So for float64 `tanh` NumPy is currently both faster *and*
more accurate than what we do, and matching it means a different algorithm — a
segmented table with a polynomial per segment, which is what NumPy has — rather
than a better polynomial in this one.

One thing does help, for anyone who takes it on: folding to `|x|` before the
reduction removes the first of the two cancellations, and drops the points over
1 ulp from 109,151 to 3,641. The maximum stays at 3.

By the logic that sends GEMM to NumPy, this op is a candidate for the same
treatment — it is the one kernel where NumPy is better on both axes. It is not
delegated because a provider path costs a trait, a registration, a threshold
and a rayon-worker guard, and this is one operation at one dtype; it would also
leave float64 `tanh` returning different bits to the Python package than to a
Rust embedder. If a second op ever joins it, that arithmetic changes.

Everything built on these kernels inherits the shape but not the loss — float64
`sigmoid` and `gelu` still win (2.9× and 2.3×), because their cost is dominated
by the passes they avoid rather than by the transcendental itself. float64
`exp` and `log` do lose at 1000 elements (0.29× and 0.38×), for the reason the
next paragraph gives, but close as the array grows and are ahead or level by
16M: `exp` 1.07–1.11× and `log` 0.99–1.02× over repeat runs. `tanh` is the only
one that stays behind at every size, which is why it is the only one described
here.

Below 16M the remaining losses are small ones at small sizes, where a few
microseconds of call overhead is the whole measurement, and in the band the
parallel thresholds in `ops::map` govern. Those thresholds carry their own
measurements in that file, taken on a different machine; re-tuning them to this
container would improve these numbers and regress that one, so they are left
alone. If you are tuning for a specific host, that file is where to look and
this script is how to check.

## Reading a run

`verdict` is `rust`, `numpy` or `tie`, with a 5% dead band: a difference smaller
than that is inside the run-to-run spread and is not a reason to move an
operation. The per-family geometric mean at the bottom is the summary worth
watching after a change. As of the run these tables come from:

```
elementwise  1.32x     gemm  0.87x     reduction  1.77x
shape        1.19x     unary 1.41x
```

`gemm` sits below 1.0 by design — those products are NumPy's, and the number is
what the crossing costs.
