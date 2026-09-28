// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! A global allocator that keeps recently freed large blocks for reuse.
//!
//! A fresh block costs a page fault per page the first time it is written,
//! and the system allocator hands large blocks straight back to the operating
//! system once several are freed together -- which is what a chain of
//! operations over large tensors does on every call. `angle` over a million
//! float64 values made 10456 page faults a call and took 12.9ms, most of it in
//! the faults. Kept and reused, its temporaries fault once.
//!
//! Only blocks of at least [`MIN_BYTES`], below which the system allocator's
//! own reuse already works, and never more than [`CAPACITY_BYTES`] held at
//! once, the oldest going first. A block is reused only for a request of
//! exactly its size and alignment, which is what a loop over same-shaped
//! tensors asks for, and which lets every block go back to the system
//! allocator under the layout it was allocated with. A failed allocation gives
//! the cache back before trying again.
//!
//! The cache is a fixed array behind a spin lock, so the allocator never
//! allocates, and its critical sections are a scan of that array. `fork` takes
//! the lock before it copies the process, so a child never starts with the
//! lock held by a thread it does not have, or with the array half updated.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, Ordering};

/// Blocks smaller than this are not cached.
pub const MIN_BYTES: usize = 1 << 20;
/// The most the cache holds at once.
pub const CAPACITY_BYTES: usize = 256 << 20;
/// How many blocks the cache can hold.
const SLOTS: usize = 64;

/// The system allocator, with recently freed large blocks kept for reuse.
pub struct BlockCachingAllocator;

#[derive(Clone, Copy)]
struct Slot {
    ptr: *mut u8,
    size: usize,
    align: usize,
    /// When the block was cached, for evicting the oldest first.
    age: u64,
}

const EMPTY: Slot = Slot {
    ptr: std::ptr::null_mut(),
    size: 0,
    align: 0,
    age: 0,
};

struct Blocks {
    slots: [Slot; SLOTS],
    bytes: usize,
    clock: u64,
}

struct Cache {
    busy: AtomicBool,
    blocks: UnsafeCell<Blocks>,
}

// SAFETY: `blocks` is only reached through `with`, which holds `busy`.
unsafe impl Sync for Cache {}

static CACHE: Cache = Cache {
    busy: AtomicBool::new(false),
    blocks: UnsafeCell::new(Blocks {
        slots: [EMPTY; SLOTS],
        bytes: 0,
        clock: 0,
    }),
};

impl Cache {
    fn with<R>(&self, f: impl FnOnce(&mut Blocks) -> R) -> R {
        register_fork_handlers();
        self.lock();
        // SAFETY: holding `busy` makes this the only reference.
        let result = f(unsafe { &mut *self.blocks.get() });
        self.unlock();
        result
    }

    fn lock(&self) {
        while self
            .busy
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            std::hint::spin_loop();
        }
    }

    fn unlock(&self) {
        self.busy.store(false, Ordering::Release);
    }
}

/// Hold the lock across `fork`, released on both sides once it returns.
#[cfg(unix)]
fn register_fork_handlers() {
    static REGISTER: std::sync::Once = std::sync::Once::new();
    extern "C" fn lock() {
        CACHE.lock();
    }
    extern "C" fn unlock() {
        CACHE.unlock();
    }
    REGISTER.call_once(|| {
        // SAFETY: the handlers only take and release a spin lock, which is
        // async-signal-safe as a forked child requires, and never allocate.
        let status = unsafe { libc::pthread_atfork(Some(lock), Some(unlock), Some(unlock)) };
        debug_assert_eq!(status, 0, "pthread_atfork failed");
    });
}

#[cfg(not(unix))]
fn register_fork_handlers() {}

/// A cached block of exactly `layout`, if there is one.
fn take(layout: Layout) -> Option<*mut u8> {
    CACHE.with(|blocks| {
        let slot = blocks.slots.iter_mut().find(|slot| {
            !slot.ptr.is_null() && slot.size == layout.size() && slot.align == layout.align()
        })?;
        let ptr = slot.ptr;
        blocks.bytes -= slot.size;
        *slot = EMPTY;
        Some(ptr)
    })
}

/// Cache `ptr`, or report that it should be freed. Whatever had to make room
/// is freed here, outside the lock.
///
/// # Safety
/// `ptr` must be a block the system allocator gave out for `layout`, and
/// nothing may use it afterwards.
unsafe fn give(ptr: *mut u8, layout: Layout) -> bool {
    let size = layout.size();
    if size > CAPACITY_BYTES {
        return false;
    }
    let mut evicted = [EMPTY; SLOTS];
    let mut count = 0;
    CACHE.with(|blocks| {
        loop {
            let full = blocks.slots.iter().all(|slot| !slot.ptr.is_null());
            if !full && blocks.bytes + size <= CAPACITY_BYTES {
                break;
            }
            let oldest = blocks
                .slots
                .iter_mut()
                .filter(|slot| !slot.ptr.is_null())
                .min_by_key(|slot| slot.age)
                .expect("a full or over-budget cache holds a block");
            blocks.bytes -= oldest.size;
            evicted[count] = *oldest;
            count += 1;
            *oldest = EMPTY;
        }
        blocks.clock += 1;
        let age = blocks.clock;
        let slot = blocks
            .slots
            .iter_mut()
            .find(|slot| slot.ptr.is_null())
            .expect("room was made above");
        *slot = Slot {
            ptr,
            size,
            align: layout.align(),
            age,
        };
        blocks.bytes += size;
    });
    for slot in &evicted[..count] {
        // SAFETY: every cached block came from the system allocator with the
        // layout it was cached under.
        unsafe {
            System.dealloc(
                slot.ptr,
                Layout::from_size_align_unchecked(slot.size, slot.align),
            )
        };
    }
    true
}

/// Give every cached block back to the system allocator.
pub fn release_cached_blocks() {
    let mut released = [EMPTY; SLOTS];
    CACHE.with(|blocks| {
        released = blocks.slots;
        blocks.slots = [EMPTY; SLOTS];
        blocks.bytes = 0;
    });
    for slot in released.iter().filter(|slot| !slot.ptr.is_null()) {
        // SAFETY: as in `give`.
        unsafe {
            System.dealloc(
                slot.ptr,
                Layout::from_size_align_unchecked(slot.size, slot.align),
            )
        };
    }
}

/// Bytes the cache holds at the moment.
pub fn cached_bytes() -> usize {
    CACHE.with(|blocks| blocks.bytes)
}

// SAFETY: every block handed out is either fresh from the system allocator or
// a cached one it gave out for exactly the same layout, and every block goes
// back to it under the layout it was allocated with.
unsafe impl GlobalAlloc for BlockCachingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() >= MIN_BYTES
            && let Some(ptr) = take(layout)
        {
            return ptr;
        }
        // SAFETY: forwarded; `layout` has a non-zero size by `alloc`'s contract.
        let ptr = unsafe { System.alloc(layout) };
        if ptr.is_null() {
            release_cached_blocks();
            // SAFETY: as above.
            return unsafe { System.alloc(layout) };
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if layout.size() >= MIN_BYTES
            && let Some(ptr) = take(layout)
        {
            // SAFETY: the block holds `layout.size()` bytes.
            unsafe { ptr.write_bytes(0, layout.size()) };
            return ptr;
        }
        // SAFETY: forwarded.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if ptr.is_null() {
            release_cached_blocks();
            // SAFETY: as above.
            return unsafe { System.alloc_zeroed(layout) };
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller hands over `ptr`, allocated for `layout`.
        if layout.size() >= MIN_BYTES && unsafe { give(ptr, layout) } {
            return;
        }
        // SAFETY: as above.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // Every live block is one the system allocator gave out for `layout`,
        // cached or not, so it can resize it in place.
        // SAFETY: forwarded.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The cache is process-wide, so these run as one test rather than racing
    // each other over it.
    #[test]
    fn a_freed_large_block_is_reused_for_the_same_layout_only() {
        let allocator = BlockCachingAllocator;
        release_cached_blocks();
        let big = Layout::from_size_align(MIN_BYTES * 2, 64).unwrap();
        unsafe {
            let first = allocator.alloc(big);
            assert!(!first.is_null());
            first.write_bytes(7, big.size());
            allocator.dealloc(first, big);
            assert_eq!(cached_bytes(), big.size());

            // A different alignment, or a different size, is not served from it.
            let other = Layout::from_size_align(MIN_BYTES * 2, 32).unwrap();
            let unrelated = allocator.alloc(other);
            assert_ne!(unrelated, first);
            allocator.dealloc(unrelated, other);

            // The same layout gets the block back, zeroed when asked.
            let again = allocator.alloc_zeroed(big);
            assert_eq!(again, first);
            assert!(
                std::slice::from_raw_parts(again, big.size())
                    .iter()
                    .all(|&b| b == 0)
            );
            allocator.dealloc(again, big);

            // Small blocks are never held.
            let small = Layout::from_size_align(MIN_BYTES - 1, 8).unwrap();
            let before = cached_bytes();
            let block = allocator.alloc(small);
            allocator.dealloc(block, small);
            assert_eq!(cached_bytes(), before);

            // The budget holds: the oldest blocks go first.
            let slab = Layout::from_size_align(CAPACITY_BYTES / 4 + 1, 8).unwrap();
            let blocks: Vec<*mut u8> = (0..5).map(|_| allocator.alloc(slab)).collect();
            for &block in &blocks {
                allocator.dealloc(block, slab);
                assert!(cached_bytes() <= CAPACITY_BYTES);
            }

            release_cached_blocks();
            assert_eq!(cached_bytes(), 0);
        }

        #[cfg(unix)]
        a_child_forked_while_the_lock_is_busy_can_still_allocate(big);
    }

    /// A thread keeps taking the lock while the process forks, over and over.
    /// A child that inherited it held would spin on its first large block.
    #[cfg(unix)]
    fn a_child_forked_while_the_lock_is_busy_can_still_allocate(big: Layout) {
        use std::sync::Arc;
        use std::time::{Duration, Instant};

        let stop = Arc::new(AtomicBool::new(false));
        let churn = {
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    unsafe {
                        let block = BlockCachingAllocator.alloc(big);
                        BlockCachingAllocator.dealloc(block, big);
                    }
                }
            })
        };
        for _ in 0..100 {
            // SAFETY: the child only allocates, frees and exits.
            let pid = unsafe { libc::fork() };
            assert!(pid >= 0, "fork failed");
            if pid == 0 {
                unsafe {
                    let block = BlockCachingAllocator.alloc(big);
                    BlockCachingAllocator.dealloc(block, big);
                    libc::_exit(0);
                }
            }
            let deadline = Instant::now() + Duration::from_secs(10);
            let mut status = 0;
            loop {
                // SAFETY: waits on the child forked above.
                let done = unsafe { libc::waitpid(pid, &mut status, libc::WNOHANG) };
                if done == pid {
                    break;
                }
                if Instant::now() > deadline {
                    unsafe { libc::kill(pid, libc::SIGKILL) };
                    stop.store(true, Ordering::Relaxed);
                    panic!("a forked child could not take the cache's lock");
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            assert!(libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0);
        }
        stop.store(true, Ordering::Relaxed);
        churn.join().unwrap();
        release_cached_blocks();
    }
}
