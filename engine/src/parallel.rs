// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! The thread pools every parallel kernel runs on.
//!
//! Two of them. The flat loops -- a chunked map, a chunked reduction, a loop
//! over independent tasks -- go through [`for_each_task`] and its wrappers,
//! on a pool built for that one shape and for being called back to back; see
//! the task pool's section below. Everything else, a sort or a recursive
//! split, runs on rayon through [`install`].
//!
//! Kernels enter it through [`install`] rather than rayon's global pool, which
//! cannot survive a `fork`. A forked process keeps only the thread that forked:
//! the global pool's workers are gone from the child while the pool still
//! counts them, and the first job handed to them waits forever. `multiprocessing`
//! forks by default on Linux before Python 3.14, so that was every worker
//! process started after the parent had run one large operation. This pool is
//! forgotten in a forked child and built again the first time the child needs
//! it.
//!
//! Code already running on a rayon worker stays where it is, whether that is
//! this pool or one a caller installed to choose its own thread count. The pool
//! takes its size from `RAYON_NUM_THREADS` when that is set, as rayon's global
//! pool would.

use rayon::{ThreadPool, ThreadPoolBuilder};
use std::sync::atomic::{AtomicPtr, Ordering};

/// The pool, once built. A pool is never freed once it is published here, so a
/// reference to it stays valid for the life of the process.
static POOL: AtomicPtr<ThreadPool> = AtomicPtr::new(std::ptr::null_mut());

fn pool() -> &'static ThreadPool {
    let current = POOL.load(Ordering::Acquire);
    if !current.is_null() {
        // SAFETY: published pools are never freed.
        return unsafe { &*current };
    }
    build()
}

#[cold]
fn build() -> &'static ThreadPool {
    register_fork_handler();
    let pool = ThreadPoolBuilder::new()
        .thread_name(|index| format!("minitensor-{index}"))
        .build()
        .expect("the thread pool starts");
    let built = Box::into_raw(Box::new(pool));
    match POOL.compare_exchange(
        std::ptr::null_mut(),
        built,
        Ordering::AcqRel,
        Ordering::Acquire,
    ) {
        // SAFETY: just published, and never freed.
        Ok(_) => unsafe { &*built },
        Err(first) => {
            // Another thread published one first. Nothing else has seen this
            // one, so it can go, and its workers with it.
            // SAFETY: `built` came from `Box::into_raw` above and was never shared.
            drop(unsafe { Box::from_raw(built) });
            // SAFETY: published pools are never freed.
            unsafe { &*first }
        }
    }
}

/// Run `work` on the pool, where rayon's parallel iterators inside it split
/// across the pool's threads. On a rayon worker already, `work` runs in place.
pub fn install<R: Send>(work: impl FnOnce() -> R + Send) -> R {
    if rayon::current_thread_index().is_some() {
        return work();
    }
    // Rayon's workers want every core, and the task pool's may still be
    // spinning on them from the last loop.
    quiesce();
    let mut work = Some(work);
    let mut result = None;
    install_erased(&mut || result = work.take().map(|work| work()));
    result.expect("the pool ran the work")
}

/// The one instantiation of rayon's cross-thread entry, for every caller of
/// [`install`]. Panics inside `work` come back out of here as they would from
/// `ThreadPool::install`.
#[inline(never)]
fn install_erased(work: &mut (dyn FnMut() + Send)) {
    pool().install(work)
}

/// How many threads a parallel split made here would run on: those of the pool
/// the calling thread is on, or of this one.
pub fn current_num_threads() -> usize {
    if rayon::current_thread_index().is_some() {
        rayon::current_num_threads()
    } else {
        configured_threads()
    }
}

/// The size of every pool here: `RAYON_NUM_THREADS` when it is set to a
/// positive count, as rayon's own default reads it, and one per core otherwise.
fn configured_threads() -> usize {
    static THREADS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *THREADS.get_or_init(|| {
        std::env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|&threads| threads > 0)
            .unwrap_or_else(|| {
                std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
            })
    })
}

// ---------------------------------------------------------------------------
// The task pool: flat parallel loops
// ---------------------------------------------------------------------------
//
// Most of the engine's parallelism is one shape: `count` independent tasks,
// each a chunk of an output, whose results do not depend on which thread ran
// them. Rayon runs that shape correctly, but its workers fall asleep within
// microseconds of running out of work, and each call from outside the pool
// then wakes them one after another and blocks the caller on a condition
// variable until they are done. On a four-core machine a float64 dot product
// of 262,144 elements, called back to back from Python, took 29.6us at best
// and 124.8us at the median that way, with a tenth of calls past 174us.
//
// This pool keeps its workers spinning for a short window after each loop, so
// a loop that follows another finds them awake, and the submitting thread
// takes tasks itself rather than waiting idle. The same dot product reads
// 22.2us at best, 30.3 at the median and 46.3 at the ninetieth percentile.
// Past the window the workers park, so an idle process costs nothing; and a
// crossing into rayon tells them to park at once rather than compete with
// its workers for the same cores.

/// How long a worker keeps looking for the next loop before it parks.
///
/// Long enough to span the interpreter's time between two operations, which
/// is a few microseconds to a few tens of them, and short enough that a burst
/// of work ending costs a tenth of a millisecond of three cores.
const SPIN_WINDOW: std::time::Duration = std::time::Duration::from_micros(100);

thread_local! {
    /// Whether this thread is running a task of the task pool. A loop started
    /// from inside one runs where it is: the enclosing loop already has every
    /// thread, and a task waiting on a loop of its own would wait on itself.
    static IN_TASK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// One submitted loop.
struct Batch {
    /// The loop body, its lifetime erased. It is dereferenced only by a
    /// thread holding a claimed task, and the submitter does not return while
    /// any task is unfinished, so it outlives every use.
    task: *const (dyn Fn(usize) + Sync),
    count: usize,
    /// A claim takes this fraction of the tasks not yet claimed, and never
    /// less than one. Claiming one at a time costs a contended atomic per
    /// task, a visible share of an elementwise map's short tasks; claiming a
    /// fixed block leaves the last block running alone, which cost a 13ms
    /// loop over a million float32 a tenth of its time. Shrinking claims are
    /// large while there is plenty left and single tasks at the end.
    split: usize,
    next: std::sync::atomic::AtomicUsize,
    remaining: std::sync::atomic::AtomicUsize,
    panic: std::sync::Mutex<Option<Box<dyn std::any::Any + Send>>>,
}

// SAFETY: the task pointer is to a `Sync` closure and is only dereferenced
// under the lifetime argument on the field; everything else is atomics or a
// mutex.
unsafe impl Send for Batch {}
// SAFETY: as above.
unsafe impl Sync for Batch {}

impl Batch {
    /// Claim and run tasks until none are left to claim.
    fn work(&self) {
        use std::sync::atomic::Ordering;
        let mut start = self.next.load(Ordering::Relaxed);
        loop {
            if start >= self.count {
                return;
            }
            let end = start + ((self.count - start) / self.split).max(1);
            if let Err(taken) =
                self.next
                    .compare_exchange_weak(start, end, Ordering::Relaxed, Ordering::Relaxed)
            {
                start = taken;
                continue;
            }
            // SAFETY: tasks `start..end` are claimed and not yet counted off
            // `remaining`, so the submitter is still waiting and the closure
            // behind the pointer is alive.
            let task = unsafe { &*self.task };
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                for index in start..end {
                    task(index);
                }
            }));
            if let Err(payload) = outcome {
                let mut first = self.panic.lock().unwrap_or_else(|e| e.into_inner());
                first.get_or_insert(payload);
            }
            // Release, so the submitter's acquire of zero sees every write the
            // tasks made. After this the claimed tasks no longer hold the
            // closure alive, and nothing here touches it again.
            self.remaining.fetch_sub(end - start, Ordering::Release);
            start = self.next.load(Ordering::Relaxed);
        }
    }
}

struct TaskPool {
    /// Threads a loop runs on, the submitter included.
    threads: usize,
    /// Bumped once per submitted loop; workers watch it.
    epoch: std::sync::atomic::AtomicU64,
    /// The loop in flight, if any.
    current: std::sync::Mutex<Option<std::sync::Arc<Batch>>>,
    /// Held by the one thread submitting. A second thread finding it taken
    /// runs its loop on rayon instead, so two callers never wait on each other.
    busy: std::sync::atomic::AtomicBool,
    /// Set by a crossing into rayon: spinning workers park at once.
    rest: std::sync::atomic::AtomicBool,
    sleepers: std::sync::atomic::AtomicUsize,
    sleep_lock: std::sync::Mutex<()>,
    wake: std::sync::Condvar,
}

/// The task pool, once built. Like [`POOL`], never freed once published.
static TASK_POOL: AtomicPtr<TaskPool> = AtomicPtr::new(std::ptr::null_mut());

fn task_pool() -> &'static TaskPool {
    let current = TASK_POOL.load(Ordering::Acquire);
    if !current.is_null() {
        // SAFETY: published pools are never freed.
        return unsafe { &*current };
    }
    build_task_pool()
}

#[cold]
fn build_task_pool() -> &'static TaskPool {
    register_fork_handler();
    let built = Box::into_raw(Box::new(TaskPool {
        threads: configured_threads(),
        epoch: std::sync::atomic::AtomicU64::new(0),
        current: std::sync::Mutex::new(None),
        busy: std::sync::atomic::AtomicBool::new(false),
        rest: std::sync::atomic::AtomicBool::new(false),
        sleepers: std::sync::atomic::AtomicUsize::new(0),
        sleep_lock: std::sync::Mutex::new(()),
        wake: std::sync::Condvar::new(),
    }));
    match TASK_POOL.compare_exchange(
        std::ptr::null_mut(),
        built,
        Ordering::AcqRel,
        Ordering::Acquire,
    ) {
        Ok(_) => {
            // SAFETY: just published, and never freed.
            let pool: &'static TaskPool = unsafe { &*built };
            for index in 1..pool.threads {
                let spawned = std::thread::Builder::new()
                    .name(format!("minitensor-task-{index}"))
                    .spawn(move || pool.worker());
                if spawned.is_err() {
                    // Fewer workers only means fewer hands: the submitter
                    // takes whatever no worker claims.
                    break;
                }
            }
            pool
        }
        Err(first) => {
            // SAFETY: `built` came from `Box::into_raw` above and was never shared.
            drop(unsafe { Box::from_raw(built) });
            // SAFETY: published pools are never freed.
            unsafe { &*first }
        }
    }
}

impl TaskPool {
    fn worker(&'static self) {
        IN_TASK.with(|flag| flag.set(true));
        let mut seen = self.epoch.load(Ordering::Acquire);
        loop {
            seen = self.wait_for_loop(seen);
            let batch = self
                .current
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            if let Some(batch) = batch {
                batch.work();
            }
        }
    }

    /// Wait until a loop newer than `seen` is submitted, and return its epoch:
    /// spinning through [`SPIN_WINDOW`], then parked.
    fn wait_for_loop(&self, seen: u64) -> u64 {
        let started = std::time::Instant::now();
        let mut rounds = 0u32;
        while !self.rest.load(Ordering::Relaxed) {
            let now = self.epoch.load(Ordering::Acquire);
            if now != seen {
                return now;
            }
            rounds += 1;
            if rounds < 64 {
                std::hint::spin_loop();
            } else {
                if started.elapsed() > SPIN_WINDOW {
                    break;
                }
                // Hands the core to anything else runnable on it, so a
                // spinning worker never holds off a thread with work.
                std::thread::yield_now();
            }
        }
        let mut guard = self.sleep_lock.lock().unwrap_or_else(|e| e.into_inner());
        // Counted before the epoch is read, and the submitter bumps the epoch
        // before reading the count: whichever comes second sees the other,
        // so a submission can never slip between the check and the wait.
        self.sleepers.fetch_add(1, Ordering::SeqCst);
        loop {
            let now = self.epoch.load(Ordering::SeqCst);
            if now != seen {
                self.sleepers.fetch_sub(1, Ordering::SeqCst);
                return now;
            }
            guard = self.wake.wait(guard).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Run `count` tasks across the pool, the calling thread among them.
    fn run(&self, count: usize, task: &(dyn Fn(usize) + Sync)) {
        // SAFETY: only the lifetime changes. `run` does not return until
        // `remaining` is zero, after which no thread dereferences the pointer.
        let erased: *const (dyn Fn(usize) + Sync) = unsafe {
            std::mem::transmute::<&(dyn Fn(usize) + Sync), &'static (dyn Fn(usize) + Sync)>(task)
        };
        let batch = std::sync::Arc::new(Batch {
            task: erased,
            count,
            split: self.threads * 4,
            next: std::sync::atomic::AtomicUsize::new(0),
            remaining: std::sync::atomic::AtomicUsize::new(count),
            panic: std::sync::Mutex::new(None),
        });
        *self.current.lock().unwrap_or_else(|e| e.into_inner()) = Some(batch.clone());
        self.rest.store(false, Ordering::Relaxed);
        self.epoch.fetch_add(1, Ordering::SeqCst);
        if self.sleepers.load(Ordering::SeqCst) > 0 {
            let _guard = self.sleep_lock.lock().unwrap_or_else(|e| e.into_inner());
            self.wake.notify_all();
        }

        IN_TASK.with(|flag| flag.set(true));
        batch.work();
        IN_TASK.with(|flag| flag.set(false));

        // What is left is running on workers. Spin briefly, then give the core
        // up between looks: a long straggler should not cost a core of its own.
        let mut rounds = 0u32;
        while batch.remaining.load(Ordering::Acquire) != 0 {
            rounds += 1;
            if rounds < 256 {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
        *self.current.lock().unwrap_or_else(|e| e.into_inner()) = None;
        let panic = batch.panic.lock().unwrap_or_else(|e| e.into_inner()).take();
        self.busy.store(false, Ordering::Release);
        if let Some(payload) = panic {
            std::panic::resume_unwind(payload);
        }
    }
}

/// Tell the task pool's spinning workers to park now rather than at the end
/// of their window. For a crossing into another pool, which wants the cores:
/// rayon's, or the threads of whatever a provider hands work to.
pub fn quiesce() {
    let current = TASK_POOL.load(Ordering::Acquire);
    if !current.is_null() {
        // SAFETY: published pools are never freed.
        unsafe { &*current }.rest.store(true, Ordering::Relaxed);
    }
}

/// Run `task(i)` for every `i` in `0..count`, in parallel, and return when all
/// of them have.
///
/// The tasks must be independent: which thread runs which, and in what order,
/// is up to the pool, so a task's effect may depend on its index and nothing
/// else. A panic in one is raised here once every task has finished.
///
/// From inside a task the loop runs in place, in index order. On a rayon
/// worker it runs on that worker's pool, so a caller that installed its own
/// pool to choose a thread count keeps it.
pub(crate) fn for_each_task(count: usize, task: &(dyn Fn(usize) + Sync)) {
    if count == 0 {
        return;
    }
    if count == 1 || IN_TASK.with(std::cell::Cell::get) {
        for index in 0..count {
            task(index);
        }
        return;
    }
    if rayon::current_thread_index().is_some() {
        use rayon::prelude::*;
        (0..count).into_par_iter().for_each(task);
        return;
    }
    let pool = task_pool();
    if pool.threads <= 1 {
        for index in 0..count {
            task(index);
        }
        return;
    }
    if pool.busy.swap(true, Ordering::Acquire) {
        // Another thread's loop is in flight. Rayon runs this one rather than
        // having it wait for that one to finish.
        use rayon::prelude::*;
        install(|| (0..count).into_par_iter().for_each(task));
        return;
    }
    pool.run(count, task);
}

/// A pointer to the start of a buffer whose disjoint parts tasks write.
pub(crate) struct Parts<T>(*mut T);

// SAFETY: each task touches a part no other task touches, and `T: Send`
// makes moving those writes between threads sound.
unsafe impl<T: Send> Sync for Parts<T> {}

impl<T> Parts<T> {
    /// Parts of `buffer`, which stays mutably borrowed for as long as the
    /// result is used: the borrow checker does not see through the pointer.
    pub(crate) fn of(buffer: &mut [T]) -> Self {
        Self(buffer.as_mut_ptr())
    }

    /// The buffer's start. A method rather than a field read, so a closure
    /// captures the whole `Parts` -- which is `Sync` -- and not the bare
    /// pointer inside it, which is not.
    pub(crate) fn start(&self) -> *mut T {
        self.0
    }
}

/// Cut `out` into `chunk`-element pieces, the last perhaps shorter, and run
/// `work(start, piece)` on every piece in parallel, `start` being the index of
/// the piece's first element.
pub(crate) fn for_each_chunk_mut<T: Send>(
    out: &mut [T],
    chunk: usize,
    work: &(dyn Fn(usize, &mut [T]) + Sync),
) {
    let len = out.len();
    if len == 0 {
        return;
    }
    let chunk = chunk.max(1);
    let base = Parts(out.as_mut_ptr());
    for_each_task(len.div_ceil(chunk), &|index| {
        let start = index * chunk;
        let end = (start + chunk).min(len);
        // SAFETY: pieces `start..end` for distinct indices are disjoint and
        // inside `out`, which is borrowed mutably for the whole loop.
        let piece = unsafe { std::slice::from_raw_parts_mut(base.start().add(start), end - start) };
        work(start, piece);
    });
}

/// [`for_each_chunk_mut`] for a body that also returns something, collected in
/// piece order however the pool ran them.
pub(crate) fn map_chunks_mut<T: Send, R: Send>(
    out: &mut [T],
    chunk: usize,
    work: &(dyn Fn(usize, &mut [T]) -> R + Sync),
) -> Vec<R> {
    let len = out.len();
    if len == 0 {
        return Vec::new();
    }
    let chunk = chunk.max(1);
    let pieces = len.div_ceil(chunk);
    let base = Parts(out.as_mut_ptr());
    map_tasks(pieces, &|index| {
        let start = index * chunk;
        let end = (start + chunk).min(len);
        // SAFETY: as in `for_each_chunk_mut`.
        let piece = unsafe { std::slice::from_raw_parts_mut(base.start().add(start), end - start) };
        work(start, piece)
    })
}

/// `work(i)` for every `i` in `0..count`, in parallel, collected in index
/// order.
pub(crate) fn map_tasks<R: Send>(count: usize, work: &(dyn Fn(usize) -> R + Sync)) -> Vec<R> {
    let mut results: Vec<std::mem::MaybeUninit<R>> = Vec::with_capacity(count);
    // SAFETY: `MaybeUninit` needs no initialization, and every slot is
    // written below before any is read.
    unsafe { results.set_len(count) };
    let slots = Parts(results.as_mut_ptr());
    for_each_task(count, &|index| {
        // SAFETY: each index writes its own slot, once.
        unsafe { (*slots.start().add(index)).write(work(index)) };
    });
    // SAFETY: every slot was written: `for_each_task` returns only once every
    // task has run, and a panic in one never gets here. A panic does leak the
    // results the other tasks wrote, which is the cost of not tracking them.
    let mut results = std::mem::ManuallyDrop::new(results);
    unsafe { Vec::from_raw_parts(results.as_mut_ptr() as *mut R, count, results.capacity()) }
}

/// Rayon's parallel sort hands work to other threads only once a partition is
/// longer than this, the `MAX_SEQUENTIAL` of its quicksort.
const SORT_SPLITS_ABOVE: usize = 2000;

/// Sort `v` unstably by `compare`: on the pool when it is long enough for a
/// parallel sort to split, and in place, without crossing to it, otherwise.
pub fn sort_unstable_by<T: Send>(
    v: &mut [T],
    compare: impl Fn(&T, &T) -> std::cmp::Ordering + Send + Sync,
) {
    use rayon::slice::ParallelSliceMut;
    if v.len() <= SORT_SPLITS_ABOVE {
        v.sort_unstable_by(compare);
    } else {
        install(|| v.par_sort_unstable_by(compare));
    }
}

#[cfg(unix)]
fn register_fork_handler() {
    static REGISTER: std::sync::Once = std::sync::Once::new();
    extern "C" fn forget_pool_in_child() {
        // The old pools' workers did not survive the fork, so each is left
        // behind rather than dropped, which would signal them through locks
        // they may have held when the process forked.
        POOL.store(std::ptr::null_mut(), Ordering::Release);
        TASK_POOL.store(std::ptr::null_mut(), Ordering::Release);
    }
    REGISTER.call_once(|| {
        // SAFETY: registers a handler that only stores to an atomic, which is
        // async-signal-safe as a forked child requires.
        let status = unsafe { libc::pthread_atfork(None, None, Some(forget_pool_in_child)) };
        debug_assert_eq!(status, 0, "pthread_atfork failed");
    });
}

#[cfg(not(unix))]
fn register_fork_handler() {}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;

    #[test]
    fn work_runs_on_the_pool_and_stays_on_an_enclosing_one() {
        assert!(rayon::current_thread_index().is_none());
        let sum: u64 = install(|| {
            assert!(rayon::current_thread_index().is_some());
            (0..10_000u64).into_par_iter().sum()
        });
        assert_eq!(sum, 49_995_000);

        let two = ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        two.install(|| {
            assert_eq!(current_num_threads(), 2);
            install(|| assert_eq!(rayon::current_num_threads(), 2));
        });
        assert_eq!(current_num_threads(), pool().current_num_threads());
    }

    #[test]
    fn every_task_runs_exactly_once() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        for count in [0, 1, 2, 3, 7, 64, 1000, 10_007] {
            let hits: Vec<AtomicUsize> = (0..count).map(|_| AtomicUsize::new(0)).collect();
            for_each_task(count, &|index| {
                hits[index].fetch_add(1, Ordering::Relaxed);
            });
            assert!(
                hits.iter().all(|hit| hit.load(Ordering::Relaxed) == 1),
                "{count}"
            );
        }
    }

    #[test]
    fn results_come_back_in_index_order() {
        let squares = map_tasks(5000, &|index| index * index);
        assert!(squares.iter().enumerate().all(|(i, &v)| v == i * i));

        let mut out = vec![0usize; 10_001];
        let firsts = map_chunks_mut(&mut out, 64, &|start, piece| {
            for (offset, slot) in piece.iter_mut().enumerate() {
                *slot = start + offset;
            }
            start
        });
        assert!(out.iter().enumerate().all(|(i, &v)| v == i));
        assert_eq!(firsts, (0..10_001).step_by(64).collect::<Vec<_>>());
    }

    #[test]
    fn a_panicking_task_reaches_the_caller_and_the_pool_survives() {
        let caught = std::panic::catch_unwind(|| {
            for_each_task(100, &|index| {
                if index == 37 {
                    panic!("task 37");
                }
            })
        });
        assert!(caught.is_err());
        assert_eq!(map_tasks(100, &|index| index).iter().sum::<usize>(), 4950);
    }

    #[test]
    fn a_loop_inside_a_task_runs_in_place() {
        let totals = map_tasks(16, &|outer| {
            map_tasks(16, &|inner| outer * 16 + inner)
                .iter()
                .sum::<usize>()
        });
        let expected: Vec<usize> = (0..16)
            .map(|outer| (0..16).map(|inner| outer * 16 + inner).sum())
            .collect();
        assert_eq!(totals, expected);
    }

    #[test]
    fn concurrent_submitters_each_get_their_own_answer() {
        let threads: Vec<_> = (0..4u64)
            .map(|seed| {
                std::thread::spawn(move || {
                    for round in 0..200u64 {
                        let got: u64 = map_tasks(257, &|index| index as u64 * seed + round)
                            .iter()
                            .sum();
                        let want: u64 = (0..257u64).map(|index| index * seed + round).sum();
                        assert_eq!(got, want);
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
    }

    #[test]
    fn a_panic_inside_the_pool_reaches_the_caller() {
        let caught = std::panic::catch_unwind(|| install(|| panic!("inside the pool")));
        assert!(caught.is_err());
        assert_eq!(install(|| 7), 7);
    }
}
