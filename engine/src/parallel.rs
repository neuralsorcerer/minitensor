// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! The thread pool every parallel kernel runs on.
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
        pool().current_num_threads()
    }
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
        // The old pool's workers did not survive the fork, so it is left
        // behind rather than dropped, which would signal them through locks
        // they may have held when the process forked.
        POOL.store(std::ptr::null_mut(), Ordering::Release);
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
    fn a_panic_inside_the_pool_reaches_the_caller() {
        let caught = std::panic::catch_unwind(|| install(|| panic!("inside the pool")));
        assert!(caught.is_err());
        assert_eq!(install(|| 7), 7);
    }
}
