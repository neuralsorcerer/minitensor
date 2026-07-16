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
9. **[CONFINED] `Tensor::data_mut` mutated shared `Arc` storage through a
   raw-pointer cast** for all leaf tensors that require gradients — undefined
   behavior under Rust's aliasing rules whenever the storage was shared, and
   previously taken even when storage was uniquely owned. The unique case now
   uses the sound `Arc::get_mut` path; the shared case (in-place parameter
   updates that must stay visible through all handles, as in PyTorch) is
   documented at the call site. The full fix is interior mutability at the
   storage layer (see section 7).
10. **Thread-affine autograd.** The graph is thread-local: tensors created on
    one thread silently lose their history on another. Single-threaded
    Python use is safe (GIL), but the engine API allows cross-thread misuse
    with silent wrong results. (Documented; see section 7.)
11. **`t.zero_grad()` on one tensor clears every gradient on the thread**
    (`autograd::zero_gradients()` wipes the global map). Semantics differ
    from PyTorch and can surprise multi-model setups. (Documented.)
12. **Per-dtype kernel duplication.** Nearly every op repeats the same body
    five times (f32/f64/i32/i64/bool) — thousands of duplicated lines (e.g.
    `elementwise.rs` ~880 lines for 4 ops). A dtype-dispatch macro or generic
    kernels would cut the ops layer dramatically. (Documented.)
13. **Speculative/dead subsystems.** `hardware/` (profiler/optimizer,
    ~2k LOC), `operations/fusion.rs`, `debug.rs`, and the pooled GPU memory
    manager are not wired into any execution path (hardware/fusion are only
    referenced by examples/tests). They inflate the maintenance surface and
    the public API. (Documented; removal is a semver decision.)
14. **`#![allow(clippy::all)]` on the engine crate and no clippy in CI**
    (lints.yml only runs rustfmt via pre-commit). Lint debt accumulates
    invisibly. (Documented.)
15. **`include!`-based module layout** (`autograd/mod/*.rs`,
    `tensor/mod/*.rs`, bindings `pytensor/*.rs`) merges many files into one
    module, defeating visibility boundaries, slowing incremental compiles,
    and confusing tools. (Documented.)
16. **Uninitialized `Vec<u8>` buffers** (`set_len` before write) are relied
    on for output allocation. All writers fill before read, but the pattern
    is UB-adjacent and should move to `MaybeUninit`/`Vec::spare_capacity_mut`.
    (Documented.)
17. **API-inconsistencies at the Python layer.** `requires_grad_()` returns
    `None` instead of `self` (breaks chaining, diverges from PyTorch);
    `expand` materializes at the FFI boundary, so it is semantically `repeat`
    with extra steps; `get_gradient` can return gradients for tensors that
    do not require grad. (Documented.)

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

1. **Interior mutability for parameter storage** — remove the last aliasing
   cast in `data_mut` (medium effort, high value; unblocks `Send`-safe use).
2. **dtype dispatch macro** — collapse per-dtype kernel copies (large LOC
   win, low risk, mechanical).
3. **User-facing `no_grad()`** — the guard exists; expose a context manager
   in bindings and gate `requires_grad` in `Tensor::new` (small effort,
   fixes unbounded graph growth during inference).
4. **Replace `include!` layout with real modules** — mechanical, improves
   tooling and visibility control.
5. **Delete or feature-gate speculative subsystems** (`hardware`, `fusion`,
   `debug`, pooled allocator) — semver-major cleanup.
6. **Per-tensor `zero_grad` semantics** — store grads on tensors (or key
   deletion by id) instead of wiping the global map.
7. **Enable clippy in CI** and remove `#![allow(clippy::all)]`, burning down
   warnings incrementally.
8. **`MaybeUninit` output buffers**; **`requires_grad_` returning self**;
   dtype-aware `get_gradient` gating.

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
