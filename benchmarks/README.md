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

All numbers below are from one 4-core x86-64 container, NumPy 2.5.3 against
OpenBLAS 0.3.34, float32 unless stated. **They are not portable.** The shape of
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
| `argmax` | 1.1× |

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

The `argmax` row is the same shape of finding, one column over: it read 0.50×
against NumPy until the index was folded the way the value already was. `max`
over a million float32 takes 0.065 ms and `argmax` took 0.34 — five times the
cost of the same scan, for a number NumPy carries for about a fifth more. The
fold threaded `(Option<usize>, Option<(usize, T)>)` through one accumulator,
which is a serial dependency chain over two `Option` discriminants and
vectorises to nothing. Lane-blocking it the way `max` already was, and *flagging*
NaN per lane rather than carrying its position — carrying it costs a `usize` min
and select per element, which for f32 is two 64-bit-lane vectors against the one
the values occupy — took 4000×4000 float32 from 4.55 ms to 1.90, and float64
from 4.93 to 2.82. The one input that loses by locating the NaN instead of
carrying it is an array whose first element is NaN, where NumPy returns
immediately and this still scans.

`nanargmax` and `nanargmin` were the same story with a worse multiplier. They
were `argmax(where(isnan(x), -inf, x))` behind an all-NaN check built from a
second full-size count: seven passes and two full-size temporaries around a
reduction that is one pass, 1.750 ms over a million float32 where the `argmax`
underneath took 0.134. A NaN satisfies no comparison, so the fold already skips
it — the NaN-skipping index reduction *is* the instantiation the integers use —
and they now read 6.7–7.8× instead of 0.63–0.90. That also changed an answer:
the substitution could not tell a NaN pushed to `-inf` from an `-inf` that was
always there, so `nanargmax([nan, -inf])` named index 0, a NaN, from the
reduction whose whole job is to skip them. NumPy still does. This deliberately
does not.

Elementwise arithmetic, where the win is only parallelism and a fused output:

| op | 1M elements | 16M elements |
|---|---|---|
| `add` | 2.1× | 1.5× |
| `mul` | 2.1× | 2.0× |
| `div` | 1.9× | 1.7× |
| `maximum` | 1.8× | 1.6× |

## What still loses

Of the kernels in the tables above, at 16M elements there is **nothing** in
float32 that NumPy computes faster, and one thing in float64: `tanh`. The
breadth sweep further down finds ten more float64 transcendentals behind for
the same reason, which is the one set out here -- this section is where the
reason lives, that one is where the list is.

`tanh` is not only a 16M problem: it loses at every size, and worse at the
small ones:

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
16M: `exp` 1.07–1.11× and `log` 0.99–1.02× over repeat runs. Of the ops in
these tables `tanh` is the only one that stays behind at every size, which is
why it is the one described here.

Below 16M the remaining losses are small ones at small sizes, where a few
microseconds of call overhead is the whole measurement, and in the band the
parallel thresholds in `ops::map` govern. Those thresholds carry their own
measurements in that file, taken on a different machine; re-tuning them to this
container would improve these numbers and regress that one, so they are left
alone. If you are tuning for a specific host, that file is where to look and
this script is how to check.

### Two of them were not slow. They were wrong.

The float32 side of that module grew six kernels while this file was being
re-taken, and three of them were free: `log2`, `log10` and `exp2` are `log` and
`exp` times a constant, and were calling scalar `log2f`, `log10f` and `exp2f`
while the kernels they are a constant away from sat in the same crate. The
scale has to be applied *inside*, in float64, before the single narrowing --
a `log` followed by a `mul` would round twice, which is the property these
kernels exist for. They went 0.48, 0.23 and 0.48 to 1.08, 1.15 and 1.52.

`asinh` and `acosh` were not free, and they turned up something worth writing
down. `asinh` was the slowest routine in the crate: 8.1ms over a million
float32 where the `log` kernel costs 0.324, twenty-five logarithms for a
function that is one. Written as `log1p(|x| + x²/(1 + sqrt(1 + x²)))` and
`log(x + sqrt((x-1)(x+1)))` both are bit-identical to a float64 reference on
all 2³² float32 inputs, on every dispatch path.

The routines they replaced are not merely imprecise:

```
                       differ of 2^32   worst
    f32::asinh             98,197,734   1,020,169,704 ulp
    f32::acosh             25,756,405   1,020,169,704 ulp
```

A billion ulp is not a rounding. `f32::asinh`, `f32::acosh`, `f64::asinh` and
`f64::acosh` all form `2x` in their working width, which is infinity for every
`x > MAX/2`, so all four answer **infinity** where the true value is about 88.7
in float32 and 710.5 in float64. `f64::acosh` has a second defect: measured
against an 80-bit reference it is out by 25,216,050 ulp next to its own domain
edge, because `acosh(1+t)` is about `sqrt(2t)` and adding that to one discards
half the result before the logarithm sees it. The float64 arms are rewritten
around `t = x - 1` and read 2.6 ulp.

Two things about the float64 rewrite are costs rather than wins, and are worth
stating as such. `acosh` is 3.202ms against the 2.288 of the routine it
replaces, because `ln_1p` costs twice what `ln` does and the edge needs it --
the threshold sits at 1.125, where the two forms were measured to agree within
one ulp. And float64 `asinh` is a hair *looser* than what it replaced, 1.47 ulp
against 1.40: that one used `hypot`, which is more accurate and is most of why
it cost 11ms.

The sixth is `atan2`, and it is the one that had to be built rather than
borrowed. It is `atan(y / x)` plus the quadrant that quotient loses, formed in
float64 because no pair of float32 operands can overflow one -- the widest
ratio they can ask for is about 1e83 against float64's 1e308. The inputs left
over are the ones where the quotient is not a slope at all: `x` a zero, and
both operands infinite. A binary op has no exhaustive sweep -- 2^64 pairs is
out of reach -- so one operand is swept over all 2^32 against each of fourteen
fixed values, in both directions, on every backend. Across those 84 sweeps
exactly **two** inputs differ from the correctly rounded answer, by one ulp,
identically on all three backends. Over eight million random pairs spanning the
whole exponent range, `np.arctan2` differs from that same reference on 507,567
of them (6.3%) by up to three ulp, and this kernel on none.

It also paid for a change the trig kernels wanted. Both compute a vectorized
answer and then walk the block again fixing the elements it cannot express,
and that walk was unconditional -- three memory streams re-run over kernels
whose whole cost is memory. Flagging in the first pass whether any element will
want the second, branch-free, makes it skippable: `atan2` 1.005ms to 0.574,
`sin` 0.455 to 0.330, `cos` 0.478 to 0.351, `tan` 0.500 to 0.434. The output is
identical by construction, and the exhaustive sweeps were re-run to say so
rather than to assume it.

The exhaustive sweep earned its runtime twice over here. It caught a bug in the
first draft of `acosh` that the author had confidently documented as impossible
-- below -2²⁶ the `- 1` is lost to rounding, `sqrt(x² - 1)` comes back as
exactly `|x|`, and `x + |x|` is a zero the logarithm reads as legitimate, so
855,638,014 of the 2³² inputs got a number where the answer is NaN. Sampling
would not have found it. A comment in `acosh_one` now records the mistake
rather than the claim.

## Every name NumPy also has

The tables above come from hand-written cases: a few dozen operations at four
sizes, chosen because they are where the dispatch decision lives. They say
nothing about the other hundred and forty names this library shares with NumPy,
and a library is not fast because its six most-discussed kernels are.

The `surface` family answers for the rest, and it is not a list. It pairs every
public `minitensor` name against the NumPy name spelled the same way, gives both
the same values, and times them at a million elements — past last-level cache,
where the kernel is what is being measured rather than the call around it.
Nothing has to be added to it when an operation is added to the library, which
is the point: a list is a thing someone has to remember to extend, and the
operations that go quietly wrong are exactly the ones nobody was thinking about.

Four questions decide whether a name belongs, and each is asked of the code
rather than answered in advance:

- **Does the call work on both sides?** Most do not, on the sweep's operands —
  `eye`, `full`, `tile` and forty others want a shape or a repeat count where
  they are being handed an array. They raise, and a name that raises is skipped.
  Integer operands are tried when the float ones are refused, which is how
  `bitwise_and`, `gcd` and the shifts get measured at all.
- **Did both do the same thing?** Matching output shapes is the cheapest honest
  test. `mt.sort` returns values *and* indices where `np.sort` returns values
  alone, so timing one against the other would report a ratio that is really a
  statement about the two APIs; `nonzero`, `divmod`, `frexp` and the `unique_*`
  family go the same way.
- **Did ours do anything at all?** `conj` of a real tensor *is* that tensor, and
  `np.conj` copies it. Timing the two measures a copy against nothing and reports
  five thousand, which is true and is not about a kernel. A call that hands an
  operand straight back is dropped.
- **Is the cost linear in the input?** `convolve` is a fine operation and a
  meaningless row: it is O(n·m), so the sweep would spend minutes on it and
  report a number that says nothing. Rather than name the quadratic ones, the
  sweep measures two small probes four sizes apart. Everything genuinely
  quadratic reads 13× or more — `convolve` 13.7, `kron` 13.8, `outer` 17.3,
  `diagflat` 21.3 — and the limit sits at 10, above the 7.4× that a perfectly
  linear `cos` shows when the thread pool happens to turn on between the two
  probes.

308 cases survive that, and on this container the script reports **1.89x**
overall: 224 ahead of NumPy, 73 behind, 11 level. Per dtype, leaving out the
two rows per dtype that read 0.00 because they are answers rather than
measurements (`empty_like` and `flipud`, both explained below):

```
float64  2.10x    int64  1.79x
float32  2.19x    int32  1.59x
```

### What it found the first time it ran

Fourteen operations were behind by more than the kernels they are built from,
and in every case the reason was work being done *around* the kernel rather
than in it.

The columns are the ratio against NumPy, not milliseconds, and that is not
cosmetic. Between the first of these sweeps and the last, the 106 float64
operations none of this work touched got a median of 16% faster in our column
and 6% in NumPy's -- it is a shared container -- while their *ratios* moved by
8%, because both sides of a ratio are measured in the same run. The ratio
drifts less, not none: a gain under about 1.2x here is not distinguishable
from the machine, which is worth remembering for the bottom two rows and for
nothing above them.

| | before | after | |
|---|---|---|---|
| `delete` | 0.02× | 0.80× | 40× |
| `signbit` | 0.28× | 2.52× | 9.0× |
| `compress` | 0.44× | 3.62× | 8.2× |
| `cov` | 0.23× | 1.16× | 5.0× |
| `take` | 0.21× | 1.01× | 4.8× |
| `partition` | 0.13× | 0.61× | 4.7× |
| `argpartition` | 0.13× | 0.59× | 4.5× |
| `isin` | 0.24× | 0.95× | 4.0× |
| `flatnonzero` | 0.46× | 1.66× | 3.6× |
| `argwhere` | 0.60× | 2.00× | 3.3× |
| `vecdot` | 0.34× | 0.94× | 2.8× |
| `setdiff1d` | 0.56× | 1.01× | 1.8× |
| `setxor1d` | 0.22× | 0.37× | 1.7× |
| `intersect1d` | 0.46× | 0.73× | 1.6× |

Seven of them are ahead of NumPy now rather than behind it, and `isin` is
within a twentieth of it. In absolute terms the largest was `delete`, which
went from 148 ms to under 3 on a million elements.

Six causes, none of which the deep families could have shown, because each
only appears at a size they do not reach with an argument they do not have:

- **A million positions taken one at a time.** `delete` and `partition` built
  their index lists with Python loops, and `index_select` narrowed an `i64`
  index tensor into a `Vec<usize>` -- two allocations and three passes over the
  index to move nothing. A selection of a million elements cost 9.6 ms of which
  the gather was 0.7.
- **A loop that was never parallel.** `searchsorted` ran its binary searches on
  one core. They are independent by construction; that is the shape of the
  operation. It is what `isin` and the four set operations spend their time in.
- **A compaction that could not be.** `masked_select` and `nonzero` collected
  the true positions into a vector and then copied one element per entry, both
  serially. Counting the bands first makes both halves parallel and the vector
  unnecessary -- and then `delete` and `compress` could stop converting their
  masks into positions at all.
- **A full-size temporary.** `vecdot` multiplied and then summed, writing eight
  megabytes to produce eight bytes. It folds each run in place now, the way
  `dot` and `norm` already did.
- **A question asked the long way round.** `signbit` built a float carrying the
  sign and compared it against zero, to recover a bit that was already in the
  input.
- **Work to prepare for work.** `take` rewrote its whole index to wrap negative
  positions that were not there; `cov` transposed a `(1, n)` matrix into an
  `(n, 1)` one, which is eight megabytes copied to produce the bytes it already
  had.

`unique` is the one row that did not move, and it is the one whose time is
genuinely inside the kernel: a comparison sort, which is the group below.

The sweep has also caught a regression since, which is the other half of what
it is for. `index_select` was validating its positions with `find_first` so the
error message would name the first bad one deterministically; that cost eight
times as much to find nothing, and `take` showed up at 0.15x in a run where it
had been 0.77x with NumPy's column unmoved.

### What it still says is behind

Five groups. The first is down to one name and the fourth is empty, so what
is left is really three:

- **Transcendentals with no vectorised kernel at all.** `cbrt` 0.34–0.66×.
  Behind in *both* dtypes, which is what separates it from the group below:
  the fix is a kernel, not an algorithm.

  `pow` was the other name here at 0.53–0.79 and is no longer behind in
  float32. Built out of the `exp` and `log` kernels rather than left on
  `powf`, it reads 1.21; its float64 0.73 is the ordinary float64 gap and it
  has moved to that bullet. `cbrt` halved the same way — four passes became
  one kernel, 0.19 to 0.66 — without reaching parity, which is the honest
  reason it is still here: the remaining distance is a dedicated kernel worth
  about thirty float64 operations to beat 0.67ns an element, and that lands
  near parity rather than past it.

  This bullet held six names two re-takings ago and called itself the cheapest
  gap in the file. It was. `log2`, `log10` and `exp2` were `log` and `exp`
  times a constant and were reaching scalar `log2f`/`log10f`/`exp2f` anyway;
  scaling inside the kernel, before its single rounding, moved `exp2` to 1.42
  and the two logarithms to parity — 1.05 and 0.92 in the run these tables
  come from, and 1.07/0.99/0.89 and 0.96/0.98/1.01 over three more. An earlier
  re-taking recorded 1.08 and 1.15 and read them as a lead; they are one run's
  end of a spread that straddles 1.0. What is real is that NumPy's `log2` and
  `log10` are about 28% quicker than its `log` where ours are about 10% slower
  than ours, so the same kernel that puts `log` at 1.44–1.51 puts these two
  level.

  `asinh` was the slowest routine in the crate at 8.1ms over a million
  float32 and reads 0.99; `acosh` came along as the same construction and
  reads 11.32. `atan2` was the last and the most stubborn -- monomorphizing
  its call did nothing, because the call was to `atan2f` -- and it took a
  kernel of its own to go 5.555ms to 0.574, 0.15× to 1.41.
- **Transcendentals in float64.** `log1p` 0.30×, `sinh` 0.31, `tan` 0.39,
  `asinh` 0.39, `expm1` 0.40, `log10` 0.44, `cosh` 0.48, `atan2` 0.48, `atan`
  0.52, `tanh` 0.57, `exp2` 0.66, `pow` 0.73, `log2` 0.84 — all but one of
  them at or above 1.0 in float32, most well above, and the exception is
  `log10` level at 0.92.
  `ops::simd::transcendental` is float32-only by design: it computes in
  float64 and rounds once, which is exactly the headroom a float64 output does
  not have. The section above says why that is a different algorithm rather
  than a better polynomial. It is the largest gap left, and the one with no
  cheap version.
- **Sorting and selection.** `setxor1d` 0.23–0.32×, `unique` 0.41–0.58,
  `union1d` 0.46–0.49, `argpartition` 0.49–0.60, `intersect1d` 0.53–0.67,
  `partition` 0.63–0.65. Every one of them is a comparison sort or a selection
  underneath, and ours is a portable one where NumPy's is vectorised per
  microarchitecture. This is the group with the most operations in it and the
  one a single change would move furthest.

  Two plausible changes are not that change, and both are worth writing down
  so the next person does not spend the afternoon on either.

  `unique` sorts with a comparator closure, while `ops::order` holds
  order-preserving integer keys whose header records comparisons falling from
  62 ms to 18. Sorting a million float32 as keys rather than through the
  comparator is **1.16×** -- 11.52 ms against 9.89 including the pass that
  builds the keys -- where the distance to NumPy is 2.3×. In a debug build the
  same measurement reads 1.87×, which is what makes it tempting; in release the
  comparator inlines.

  And `sort` does not pay for the indices it returns. It looks like it should:
  `mt.sort` and `mt.argsort` cost the same to the microsecond, so the values
  are getting an argsort whether or not the caller wanted one. But it already
  sorts a packed key-and-position `u64` rather than a `(value, index)` pair,
  and that packing is what makes carrying the position nearly free -- measured
  standalone over a million float32, the pair costs 14.78 ms, the packed word
  10.58, and the values alone 9.21. A values-only path is worth about 1.3x,
  not the 3x the API shape suggests.

  What is left in both cases is the algorithm, and matching it means a
  vectorised quicksort.
- **Compositions where NumPy has a kernel.** This group is empty, and it is
  the only one that has emptied. It held most of this list once: `fmax`/`fmin`
  were five passes and are 2.28–3.12 now, `nanmax`/`nanmin` and
  `nanargmax`/`nanargmin` are 1.9–2.7 and 6.3–7.7, and `vecdot`, `cov` and
  `signbit` left the same way.

  `histogram_bin_edges` was the last of them at 0.37–0.76× and reads
  3.21–5.47. It is a min, a max and a `linspace` over the answer, and
  essentially all of it was the scan: a per-element rayon chain that widened
  every value to float64 on the way past and kept one running pair, which
  makes the compare-and-replace a serial dependency nothing can vectorize.
  0.846ms over a million float32, where `min` and `max` through the value
  reductions cost 0.134 between them. It is now the same chunked, lane-blocked
  fold those use.

  They are the pattern, and the pattern is that what costs is the
  intermediate, not the arithmetic — and that a composition is worth checking
  against the kernels it is made of before it is worth optimising at all.
- **Two that are answers rather than problems.** `empty_like` reads 0.00×
  because it zeroes: the kernels take `&mut [T]`, and reading uninitialised
  floats is undefined behaviour, so an "uninitialised" buffer here is a zeroed
  one. `flipud` reads 0.00× because NumPy returns a view with a negative stride
  and a tensor here is always contiguous, so a flip is a copy.

`inner` 0.87–0.98× sits just under the dead band rather than in a group;
`squeeze` was next to it at 0.93 and is now 1.04–1.05, which is inside it.
Fifteen of the 166 names measured here are below 0.95 in both dtypes, down
from sixteen and from twenty-two before that, and every name that left did so
by reaching a kernel rather than by getting a better one. Fourteen is the
fairer figure: `log10` is on the list because this run put its float32 at
0.92, and re-timing it gives 0.96, 0.98 and 1.01. A threshold counted against
a single run will do that, which is the argument for watching the family mean
rather than the membership of a list.

### One row that is measuring the wrong question

`percentile` and `nanpercentile` read 0.48× here, and that number is about this
harness rather than about them. The breadth family gives a two-argument name
both of its operands, so it asks for **a million quantiles of a million
elements** -- which is a real operation, and not one anybody reaches for. At
the counts people do ask for, over the same million float32:

```
    q=1      3.73ms  vs   9.09    2.44x
    q=2      3.71   vs    3.75    1.01x
    q=4      5.83   vs   22.20    3.81x
    q=11     8.95   vs   32.02    3.58x
    q=101   14.11   vs   51.40    3.64x
```

Those were 31ms flat across every count until recently, because two quantiles
took a full sort to answer two questions where NumPy selects. A quantile wants
*ranks*, not order, so the ranks the interpolation will read are quickselected
-- middle rank first, which leaves the lower ones in the prefix and the higher
in the suffix, so k of them cost `O(n log k)` -- and only past 64 distinct
ranks does a sort win. That sort now runs on the thread pool as well.

Two rows are worth reading with care rather than believing. `array_equal` and
`allclose`, which read in the thousands and the tens, stop at the first element
that differs, and the
sweep gives them two different random arrays, so they stop at the first one.
That is a real advantage on data that differs early and no advantage at all on
data that does not, which is the kind of thing a single number cannot say.

## Reading a run

`verdict` is `rust`, `numpy` or `tie`, with a 5% dead band: a difference smaller
than that is inside the run-to-run spread and is not a reason to move an
operation. The per-family geometric mean at the bottom is the summary worth
watching after a change. As of the run these tables come from:

```
elementwise  1.42x     gemm  0.90x     reduction  1.98x
shape        1.67x     unary 1.32x     surface    2.13x
```

Four of those moved by less than the 8% a ratio drifts between runs on a
shared container, which is the point of watching the family rather than the
row: `reduction` reading 1.98 where it read 2.13 is not a regression anybody
introduced, and neither is `surface` reading 2.13 where it read 2.10.

`gemm` sits below 1.0 by design — those products are NumPy's, and the number is
what the crossing costs.
