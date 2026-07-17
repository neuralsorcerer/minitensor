# Architecture Review and Refactoring Report

This document records a full architectural assessment of minitensor (engine,
bindings, and Python layer), the problems identified, the changes implemented,
and a prioritized plan for the remaining work. It is intended as the living
reference for architectural decisions; update it as the items below are
addressed.

## 1. Current architecture (assessment)

minitensor is a three-layer system:

```text
minitensor/            Pure-Python facade: re-exports, introspection helpers,
                       broadcasting utilities (~1.1k LOC)
bindings/  (PyO3)      Python <-> Rust boundary: PyTensor, functional API,
                       nn/optim wrappers, NumPy interop (~13k LOC)
engine/    (Rust)      Storage, ops, autograd, nn, optim, serialization,
                       plugins, backends (~57k LOC)
```

Key mechanics discovered during the review:

- **Storage**: `TensorData` owns either a `Vec<u8>` (CPU) or a raw
  device pointer, and is shared between tensors via `Arc<TensorData>`.
  `Tensor` carries shape/strides/dtype/device plus autograd metadata
  (`grad_fn`, `grad`, `tensor_id`).
- **Autograd**: a *thread-local global* `ComputationGraph` maps `TensorId ->
  GraphNode`. Every differentiable op allocates an output tensor, attaches a
  `*Backward` gradient function (which stores cloned operand tensors), and
  registers the node with the global graph. `backward()` walks the graph;
  gradients live in a map inside the graph and are read back via
  `autograd::get_gradient` (optimizers, `.grad` in Python).
- **Views**: almost every op materializes its output (`transpose` copies,
  `reshape` re-strides). Only `expand` produces true strided (stride-0)
  views. The FFI boundary (`PyTensor::from_tensor`) force-materializes any
  non-contiguous tensor before it reaches Python.
- **Execution**: CPU kernels are hand-written per dtype with rayon
  parallelism and a SIMD fast path for f32/f64. GPU backends (CUDA, Metal,
  OpenCL) exist behind cargo features but are not compiled by default, and
  allocation silently falls back to CPU.
- **Python layer**: thin re-export surface plus API introspection helpers;
  `nn`/`optim`/`functional` are implemented in Rust and exposed as modules.

## 2. Problems, risks, and technical debt found

Ordered roughly by severity. Items marked **[FIXED]** were addressed in this
refactor; the rest are documented with a migration path in section 7.

1. **[FIXED] `Tensor::view`/`reshape` corrupted non-contiguous tensors.**
   `view` re-strided unconditionally, so `expand(...).reshape(...)` at the
   engine level produced a tensor whose shape claimed N elements over storage
   holding fewer (verified: shape `[12]` over 3 stored elements). The Python
   surface was protected only by a blanket copy in `PyTensor::from_tensor`.
   `view` now rejects non-contiguous tensors; `reshape` (both the `Tensor`
   method and the `shape_ops` op) materializes a contiguous copy first.
2. **[FIXED] Backward pass scaled with the whole recorded graph, not the
   traced subgraph.** The graph topologically sorted *every node ever
   recorded on the thread* on each backward call and iterated all of them.
   Backward now plans only the subgraph reachable from the loss tensor.
3. **[FIXED] The full gradient map was cloned on every backward call** and
   immediately discarded by every production caller. `backward` now returns
   `Result<()>`; gradients stay in the graph store and are read individually.
   Tests that want the full map use the new `backward_collect`.
4. **[FIXED] Gradient-kernel registration was suppressed by accident.**
   During backward, ops run inside gradient functions (e.g. `matmul` with a
   non-detached saved operand) attempt to register new autograd nodes. This
   was prevented only because the graph's `RefCell` happened to be borrowed —
   `add_to_graph` used `try_borrow_mut` and *silently dropped* registrations.
   There is now an explicit thread-local grad-recording mode (`NoGradGuard`,
   `is_grad_enabled`): the backward executor disables recording, and
   `add_to_graph` is a loud `borrow_mut` otherwise. This also provides the
   primitive for a future user-facing `no_grad()`.
5. **[FIXED] Saved tensors were held until the next optimizer step.** The
   graph (including every activation captured by `*Backward` structs) was
   only freed by `optimizer.step()`. After a non-retaining `backward()`, the
   bindings now release the reachable interior nodes immediately
   (`release_saved_subgraph`), which both frees memory earlier and makes the
   "graph has been freed" error truthful.
6. **[FIXED] `TensorData` carried a vestigial manual reference count.**
   `inc_ref`/`dec_ref` were never called outside their own tests, while
   `Drop` only deallocated raw device buffers when the counter happened to
   equal 1 — a leak trap. Sharing is `Arc`'s job; the counter is gone and
   `Drop` unconditionally returns raw buffers to the allocator.
7. **[FIXED] Unsafe per-element parallel loops in gradient accumulation.**
   `add_inplace` erased slices to raw `usize` addresses and indexed them from
   a parallel loop. Replaced with safe chunked `par_chunks_mut`/`zip` loops
   (`binary_assign_slices`), which keep bounds information visible to the
   compiler and remove 10 `unsafe` blocks from the hottest accumulation path.
8. **[FIXED] `DivBackward` allocated a ones tensor and ran 6 kernels** to
   compute two gradients. It now runs 4 kernels with no scratch ones tensor.
9. **[FIXED] `Tensor::data_mut` mutated shared `Arc` storage through a
   raw-pointer cast** for all leaf tensors that require gradients — undefined
   behavior under Rust's aliasing rules whenever the storage was shared. The
   cast is gone: storage moved to `UnsafeCell` and `data_mut` returns a
   `DataMut` token whose `Shared` variant routes in-place parameter updates
   through interior mutability with a documented aliasing contract
   (section 7, third change set).
10. **Thread-affine autograd.** The graph is thread-local: tensors created on
    one thread silently lose their history on another. Single-threaded
    Python use is safe (GIL), but the engine API allows cross-thread misuse
    with silent wrong results. (Documented; see section 7.)
11. **[FIXED] `t.zero_grad()` on one tensor cleared every gradient on the
    thread** (`autograd::zero_gradients()` wiped the global map). It now
    removes only that tensor's gradient.
12. **Per-dtype kernel duplication.** Nearly every op repeats the same body
    five times (f32/f64/i32/i64/bool) — thousands of duplicated lines (e.g.
    `elementwise.rs` ~880 lines for 4 ops). A dtype-dispatch macro or generic
    kernels would cut the ops layer dramatically. (Documented.)
13. **[PARTIALLY FIXED] Speculative/dead subsystems.** `operations/fusion.rs`
    (623 lines, never called) is removed — a breaking change for the
    `engine` crate only, documented in section 7. `hardware/`
    (profiler/optimizer, ~2k LOC, referenced only by an example and a
    compatibility test) and the pooled GPU memory manager remain, flagged
    for the maintainer as a semver decision; `debug.rs` is exposed through
    the Python API and stays.
14. **[FIXED] `#![allow(clippy::all)]` on the engine crate and no clippy in
    CI** (lints.yml only ran rustfmt via pre-commit). The blanket allow is
    gone, the workspace is warning-free, and CI now runs
    `cargo clippy --workspace --all-targets -- -D warnings`. This also
    surfaced and fixed real issues: UB-producing initializer kernels in
    `nn/init.rs` and two deny-level raw-pointer lints in the memory stack.
15. **`include!`-based module layout** (`autograd/mod/*.rs`,
    `tensor/mod/*.rs`, bindings `pytensor/*.rs`) merges many files into one
    module, defeating visibility boundaries, slowing incremental compiles,
    and confusing tools. (Documented.)
16. **[PARTIALLY FIXED] Uninitialized `Vec<u8>` buffers** (`set_len` before
    write) were relied on throughout. The genuinely dangerous cases are
    fixed: `nn/init.rs` no longer creates `&mut` slices over uninitialized
    values, `bool` storage is always zero-initialized, and `clone_data`
    copies safely. Op *output* buffers remain uninitialized behind one
    documented helper because full zero-initialization measured 30–35%
    slower on large element-wise ops; the tracked fix is `MaybeUninit`-typed
    kernel writes (section 7).
17. **[PARTIALLY FIXED] API-inconsistencies at the Python layer.**
    `requires_grad_()` now returns `self` (chaining works, matching
    PyTorch). Still open: `expand` materializes at the FFI boundary, so it
    is semantically `repeat` with extra steps, and `get_gradient` can return
    gradients for tensors that do not require grad.
18. **[FIXED] Unchecked shape products.** `Shape::numel` computed
    `dims.iter().product()`, which wraps in release builds: an absurd shape
    could report a small element count, under-allocate storage, and turn
    later stride-based indexing into out-of-bounds access. `numel` and
    `Strides::from_shape` now use checked multiplication in every profile.
19. **[FIXED] No inference mode.** Repeated forward passes with
    `requires_grad` parameters grew the thread-local graph (and its saved
    operands) without bound; there was no way to turn recording off.
    `no_grad()` / `enable_grad()` now exist end to end (see section 7).

## 3. Target architecture

The refactor converges on the following boundaries (largely realized for the
autograd path in this change set):

- **Storage layer** (`tensor::data`): dumb, `Arc`-shared buffers; no
  reference counting of its own; eventual interior mutability
  (`UnsafeCell`-backed cells behind a safe API) so parameter updates do not
  need aliasing casts.
- **Tensor layer** (`tensor::mod`): shape/stride bookkeeping with *enforced*
  invariants — a `Tensor`'s shape must always describe its storage; `view`
  is contiguous-only, `reshape` materializes, `expand` is the only stride
  producer until strided kernels exist.
- **Autograd layer** (`autograd`): tape building (`add_to_graph`, gated by an
  explicit grad-mode), pass planning (`plan_backward`, reachable subgraph
  only), pass execution (`execute_backward_plan`, borrow-free, grad-mode
  off), and storage (`set_gradients`/`get_gradient`), with node release
  decoupled from optimizer stepping (`release_saved_subgraph`).
- **Ops layer** (`operations`): kernels stay dtype-specialized but should be
  generated (macro/generics) rather than hand-copied; broadcasting via
  shape-level utilities; no direct graph manipulation beyond `add_to_graph`.
- **Boundary layer** (`bindings`): conversion, validation, and Python
  semantics only; no correctness-critical patching of engine behavior (the
  `from_tensor` materialization stays as defense-in-depth but is no longer
  load-bearing).

Trade-offs taken: the thread-local graph is retained (a process-global graph
with locking would penalize the single-threaded common case and PyO3 usage);
per-op saved-tensor `detach()` was preferred over reworking every gradient
function's captures; eager materialization remains the norm because the
kernel layer indexes storage in logical order — introducing lazy views
engine-wide is a larger project listed below.

## 4. Recommended folder/module structure

Realized now: no file moves (churn would obscure the behavioral fixes).
Recommended next structure, in order of value:

```text
engine/src/
  autograd/
    graph.rs        # tape + planning + execution (done, in place)
    ops/            # one file per Backward family, real submodules (replace include!)
  tensor/
    storage.rs      # TensorData (rename of data.rs)
    tensor.rs       # Tensor core (replace include!-merged mod/)
  ops/
    kernels/        # dtype-generic kernel bodies (macro-generated)
    ...
  # delete or feature-gate: hardware/, operations/fusion.rs, debug.rs
```

## 5. Component responsibilities and boundaries (after this change)

| Component | Responsibility | May touch |
|---|---|---|
| `TensorData` | own one buffer, expose typed slices | allocator |
| `Tensor` | shape/stride invariants, COW mutation policy | `TensorData`, autograd registration |
| `ComputationGraph` | record nodes, plan/execute/store backward | gradient functions |
| `NoGradGuard` / grad mode | decide whether ops record | thread-local flag |
| ops (`operations::*`) | math + broadcasting + attaching `*Backward` | tensors, `add_to_graph` |
| optimizers | read grads (`get_gradient`), update params in place | tensors |
| bindings | conversion + Python semantics | public engine API only |

## 6. Data / control / dependency flow

Forward: Python call → binding extracts `Tensor` → op validates devices,
coerces dtypes, broadcasts shapes → kernel writes a fresh output buffer → if
grad mode is on and an input requires grad, a `*Backward` (with saved
operands) is attached and the node registered in the thread-local graph.

Backward: `loss.backward()` → plan = reachable reverse-topological subgraph
(single borrow) → executor runs gradient functions with grad recording
disabled, accumulating into a local map (no graph borrow held) → map stored
in the graph → bindings release interior nodes (non-retaining case) →
optimizer reads `get_gradient(param)` and updates parameters in place →
`optimizer.step()` clears the graph.

Dependencies flow one way: bindings → engine ops → tensor/storage; autograd
sits beside ops (ops register, autograd executes ops through trait objects).

## 7. Prioritized remaining migration plan

Completed in the second change set:

- **User-facing `no_grad()` / `enable_grad()`** — gradient recording is now
  gated centrally (`Tensor::new`, `Tensor::set_grad_fn`, `add_to_graph`), so
  results inside `no_grad()` are detached leaves, nothing is saved for
  backward, and inference no longer grows the graph. Exposed in Python as
  `mt.no_grad()`, `mt.enable_grad()`, `mt.is_grad_enabled()`,
  `mt.set_grad_enabled()`. One documented divergence from PyTorch: factory
  functions called inside `no_grad` also produce non-grad tensors; use
  `requires_grad_(True)` (which expresses explicit intent and is never
  gated) to opt back in.
- **Per-tensor `zero_grad`** — `Tensor::zero_grad` now removes only its own
  entry from the global gradient map (`autograd::clear_gradient`) instead of
  wiping every gradient on the thread.
- **`requires_grad_` returns `self`** in Python, enabling
  `mt.randn(...).requires_grad_(True)` chaining.
- **Clippy enabled** — `#![allow(clippy::all)]` removed; the workspace is
  clean under `cargo clippy --workspace --all-targets -- -D warnings`, which
  now runs in CI (lints workflow). Three style lints are allowed crate-wide
  with documented justification (`needless_range_loop`,
  `too_many_arguments`, `items_after_test_module` — the last is an artifact
  of the `include!` layout).
- **Overflow-safe shape arithmetic** — `Shape::numel` and
  `Strides::from_shape` use checked multiplication in all build profiles; a
  wrapped element count could previously under-allocate storage while
  indexing code still trusted the dimensions (out-of-bounds writes via the
  raw-pointer kernels).
- **Undefined behavior removed from initializer kernels** — `nn/init.rs`
  built `&mut` slices over uninitialized memory (instant UB for `bool`);
  all twelve sites now write through `Vec::extend`. `clone_data` uses
  `Vec::clone` instead of `set_len` + copy.

Completed in the third change set:

- **Interior mutability for parameter storage.** `TensorData`'s buffer now
  lives in an `UnsafeCell`, and `Tensor::data_mut` returns a `DataMut`
  access token instead of fabricating `&mut TensorData` from a shared
  `Arc` (which was undefined behavior whenever any other handle existed —
  i.e. always). `DataMut::Unique` is ordinary exclusive access (after
  copy-on-write); `DataMut::Shared` is the one documented exception —
  in-place parameter updates that must stay visible through every handle —
  and routes through UnsafeCell-backed accessors with an explicit aliasing
  contract. Call sites did not change shape (`t.data_mut().as_…_slice_mut()`
  still works; the token consumes itself and returns a slice borrowing from
  the tensor).
- **dtype-dispatch macro, first application.** The ten hand-copied typed
  slice accessors (plus the five new shared-mutation variants) are
  generated by one `typed_slice_accessors!` macro — the template for
  collapsing the per-dtype duplication in the ops layer.
- **Frozen inputs skip gradient work.** `AddBackward`/`SubBackward`/
  `MulBackward`/`DivBackward` now carry per-input `requires_grad` flags
  (as the min/max/where/matmul functions already did) and skip the entire
  gradient chain for inputs that do not require gradients. This both fixes
  `get_gradient` returning gradients for non-grad tensors (root cause, for
  arithmetic ops) and measurably speeds up graphs with frozen operands:
  the wide fan-out benchmark (`x * 0.5` per node — a frozen scalar that
  previously got a full-size multiply plus a reduce-to-scalar every
  backward) improved 15–30% in interleaved A/B runs.

Completed in the fourth change set:

- **Arithmetic kernel dedup.** The eighteen hand-copied `*_direct` binary
  kernels (add/sub/mul/div × dtypes) are generated by two macros
  (`binary_kernel!` / `binary_kernel_simd!`), and `neg`'s five dtype arms
  by one — ~440 lines of copy-paste removed with byte-identical behavior
  (interleaved A/B benchmarks show exact parity). This is the pattern to
  roll out across the rest of `operations/`.
- **Dead `fusion` subsystem removed** (623 lines). It was compiled and
  re-exported (`engine::operations::fusion::*`) but never called by any
  execution path, the bindings, or the Python API. **Breaking change for
  the `engine` crate only** (the Python package surface is unchanged):
  anything that depended on `engine::operations::fusion` should pin
  `engine 0.2.1` or vendor the module from git history.

Completed in the fifth change set:

- **Loss functions no longer compute gradients for frozen targets.** All
  seven loss backward functions (MSE, MAE, Huber, CrossEntropy, BCE,
  KLDiv, Focal) carry per-input `requires_grad` flags. Previously
  MSE/MAE/Huber allocated and negated a full-size target gradient on
  *every training step*, and KLDiv ran an entire `log`/`sub`/`add`/`mul`
  chain for the target distribution — all discarded work whenever targets
  are constants (the overwhelmingly common case). Prediction gradients are
  likewise skipped when predictions are frozen. Regression test:
  `test_loss_targets_receive_no_gradient`.

Completed in the sixth change set:

- **Gating audit finished.** The last multi-input gradient functions
  without per-input `requires_grad` flags — `LogAddExpBackward` (which ran
  an `exp`/`sub`/`mul`/reduce chain per side unconditionally) and
  `ConcatBackward` (which extracted a gradient slice per input) — are now
  gated. `PowBackward` and `LayerNormBackward` already had flags;
  single-input functions need none (they are only reached when their
  input requires grad). Regression test:
  `test_concat_frozen_inputs_receive_no_gradient`.
- **Comparison kernels deduplicated.** The five hand-copied `cmp_*`
  helpers collapsed into one `cmp_kernel!` macro, continuing the pattern
  from the storage accessors and arithmetic kernels.

Completed in the seventh change set:

- **Autograd converted from `include!` to real modules.** The seven
  merged files under `autograd/mod/` are now proper submodules with their
  own imports and explicit `pub(crate)` boundaries for shared helpers,
  re-exported through `autograd/mod.rs` so every `crate::autograd::X`
  path is unchanged. The conversion also surfaced layout artifacts the
  merged namespace had been hiding: the shared broadcast-reduction helper
  lived in the *tests* include file (now in `core`), and
  `PowBackward`'s trait impl lives in a different file than its struct.
  This is the template for converting the remaining `include!` clusters.

Still open, in priority order:

1. **dtype dispatch macro for the remaining ops files** — activation and
   reduction kernels still carry per-dtype copies (storage accessors,
   arithmetic, and comparison are done).
2. **Convert the remaining `include!` clusters** — engine-side conversion
   is complete: `autograd`, `tensor`, and all seven `operations` clusters
   (27 files) are real modules. Kernel-only submodules re-export at
   `pub(crate)` so internal helpers stop leaking into the public API. The
   `tensor` conversion uses a children-of-core layout (`ops`/`indexing`/
   `autograd`/`utils` are child modules of the module declaring `Tensor`),
   preserving the struct's field privacy — no field had to become
   `pub(crate)`. Left: the feature-gated `backends/opencl` pair (not
   compiled by default, so a conversion could not be validated here) and
   the bindings clusters (`tensor.rs`: 19 files, `nn.rs`: 2); then drop
   the crate-wide `items_after_test_module` allow.
3. **Feature-gate or remove the remaining speculative subsystems**
   (`hardware`, pooled allocator; `debug` is exposed to Python and stays)
   — semver-major for the engine crate, maintainer's call.
4. **`MaybeUninit`-typed kernel output writes** — the op output buffers are
   still allocated uninitialized behind a single documented helper
   (`TensorData::owned_output_buffer`). Zero-initializing them instead was
   implemented and measured: 30–35% slower on large element-wise ops
   (`calloc` must memset reused allocations, doubling output write
   traffic), so it was reverted. The proper fix is writing through
   `MaybeUninit` in the kernels, which is sound without the extra pass.

## 8-11. Implementation, tests, validation, benchmarks

See the commit(s) accompanying this document for the full implementation.
Validation performed:

- Rust: full workspace suite (`cargo test --workspace --all-targets`),
  including new regression tests for non-contiguous `view`/`reshape`,
  reachable-subgraph planning, and post-backward node release.
- Python: full pytest suite (778 passed, 5 skipped) against a rebuilt
  release wheel.
- Benchmarks: interleaved A/B on release wheels (baseline vs refactor
  alternating on the same machine, 3 rounds, median of per-round best;
  shared-cloud CPU, so treat small deltas as noise):

| Benchmark | Baseline | Refactor | Delta |
|---|---|---|---|
| training step (3-layer MLP, batch 64) | 0.87 ms | 0.83 ms | ~ noise |
| deep chain backward (200 ops) | 36.0 ms | 36.6 ms | ~ noise |
| wide fan-out backward (50 reuses of one tensor) | 9.7 ms | 8.0 ms | **−15 %** |
| elementwise add+sum (2000×2000) | 2.44 ms | 2.35 ms | ~ noise |
| 30 forward passes, no backward | 6.26 ms | 5.87 ms | ~ noise |

The fan-out case improves consistently in every round (9.3–10.0 ms vs
7.9–8.5 ms): gradient accumulation dominates there, which benefits from the
subgraph-scoped planning and the removal of the per-backward gradient-map
clone. The memory-side changes (saved tensors released right after
backward, no full-map clone) do not show up in wall-time microbenchmarks
but reduce peak footprint between `backward()` and `optimizer.step()`.
