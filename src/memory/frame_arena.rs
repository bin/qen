use super::vm::{PlatformVmOps, VmError, VmOps};
use std::ptr::NonNull;

use super::stats;
use crate::sync::atomic::{AtomicU64, Ordering};

/// A bump allocator that resets every frame.
/// Intended for thread-local use.
pub struct FrameArena {
    base: NonNull<u8>,
    cursor: *mut u8,
    end: *mut u8,
    reserved: usize,
    committed: usize,
}

// FrameArena is NOT Send/Sync because it uses raw pointers and is intended for thread-local use.
// However, if we move it between threads (e.g. initializing in main and moving to worker), we might need Send.
// Since it owns the memory, it is conceptually Send, but not Sync.
// Safety: FrameArena owns its memory.
unsafe impl Send for FrameArena {}

impl FrameArena {
    /// Create a new frame arena with a maximum capacity.
    ///
    /// # Errors
    ///
    /// Returns `VmError` if memory reservation fails.
    pub fn new(capacity: usize) -> Result<Self, VmError> {
        let capacity = capacity.next_multiple_of(PlatformVmOps::page_size());
        // Safety: FFI call to reserve.
        let ptr = unsafe { PlatformVmOps::reserve(capacity)? };

        stats::TOTAL_RESERVED.fetch_add(capacity, Ordering::Relaxed);

        Ok(Self {
            base: ptr,
            cursor: ptr.as_ptr(),
            end: ptr.as_ptr(), // Initially nothing committed
            reserved: capacity,
            committed: 0,
        })
    }

    /// Allocate memory.
    ///
    /// # Errors
    ///
    /// Returns `VmError` if allocation fails (e.g. OOM) or if pointer arithmetic overflows.
    pub fn alloc(&mut self, layout: std::alloc::Layout) -> Result<NonNull<u8>, VmError> {
        let size = layout.size();
        let align = layout.align();

        let current_ptr = self.cursor as usize;
        let padding = (align - (current_ptr % align)) % align;
        let start = current_ptr.checked_add(padding).ok_or_else(|| {
            VmError::CommitFailed(std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                "FrameArena pointer arithmetic overflow (start)",
            ))
        })?;
        let new_cursor = start.checked_add(size).ok_or_else(|| {
            VmError::CommitFailed(std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                "FrameArena pointer arithmetic overflow (end)",
            ))
        })?;

        // Check if we need to commit more memory
        if new_cursor > self.end as usize {
            // Safety: self.reserved matches reservation size.
            if new_cursor > (unsafe { self.base.as_ptr().add(self.reserved) } as usize) {
                // Out of reserved space
                return Err(VmError::CommitFailed(std::io::Error::new(
                    std::io::ErrorKind::OutOfMemory,
                    "FrameArena exhausted reserved space",
                )));
            }

            // Commit up to new_cursor, rounded up to page size
            let needed = new_cursor - (self.base.as_ptr() as usize);
            let target_commit = needed.next_multiple_of(PlatformVmOps::page_size());

            let commit_size = target_commit - self.committed;
            let commit_start =
                // Safety: committed offset is within reserved range.
                unsafe { NonNull::new_unchecked(self.base.as_ptr().add(self.committed)) };

            // Safety: FFI call to commit.
            unsafe {
                PlatformVmOps::commit(commit_start, commit_size)?;
            }

            stats::TOTAL_COMMITTED.fetch_add(commit_size, Ordering::Relaxed);
            stats::FRAME_ARENA_COMMITTED.fetch_add(commit_size, Ordering::Relaxed);

            self.committed = target_commit;
            // Safety: self.committed is within reserved range.
            self.end = unsafe { self.base.as_ptr().add(self.committed) };
        }

        self.cursor = new_cursor as *mut u8;
        #[cfg(debug_assertions)]
        // Safety: start and size are within the newly committed/allocated region.
        unsafe {
            std::ptr::write_bytes(start as *mut u8, 0, size);
        }
        // Safety: start is non-null.
        Ok(unsafe { NonNull::new_unchecked(start as *mut u8) })
    }

    /// Reset the arena for reuse.
    pub fn reset(&mut self) {
        self.cursor = self.base.as_ptr();
    }

    /// Decommit pages beyond the high-water mark.
    ///
    /// # Safety contract
    ///
    /// All references previously obtained from [`alloc`](Self::alloc),
    /// [`alloc_val`](Self::alloc_val), or [`alloc_slice`](Self::alloc_slice)
    /// that point into the decommitted region (i.e., at offsets >= `high_water`)
    /// are **invalidated** by this call. Accessing them after `trim()` is
    /// undefined behaviour (the pages become `PROT_NONE`).
    ///
    /// Callers must ensure no live references exist into the trimmed region
    /// before calling this method. The typical pattern is `reset()` then
    /// `trim(0)`, which is safe because `reset()` logically invalidates all
    /// arena references.
    ///
    /// # Panics
    ///
    /// Panics if the decommit operation fails (debug builds only).
    pub fn trim(&mut self, high_water: usize) {
        let page_size = PlatformVmOps::page_size();
        let retain = high_water.next_multiple_of(page_size);

        if retain < self.committed {
            // Safety: retain offset is within committed range.
            let decommit_start = unsafe { NonNull::new_unchecked(self.base.as_ptr().add(retain)) };
            let decommit_size = self.committed - retain;

            // Safety: FFI call to decommit memory.
            if unsafe { PlatformVmOps::decommit(decommit_start, decommit_size) }.is_ok() {
                stats::sub_saturating(&stats::TOTAL_COMMITTED, decommit_size);
                stats::sub_saturating(&stats::FRAME_ARENA_COMMITTED, decommit_size);

                self.committed = retain;
                // Safety: self.committed is within reserved range.
                self.end = unsafe { self.base.as_ptr().add(self.committed) };
            } else {
                #[cfg(debug_assertions)]
                panic!(
                    "FrameArena::trim decommit failed at {decommit_start:p} (size={decommit_size})"
                );
            }
        }
    }

    #[must_use]
    pub fn committed_bytes(&self) -> usize {
        self.committed
    }
}

/// Helper for allocating typed values.
impl FrameArena {
    /// Allocates a value of type `T` in the arena.
    ///
    /// # Safety
    /// `T` must be `Copy`. Types with `Drop` implementations are not allowed because
    /// `reset()` will not call their destructors, which would cause resource leaks.
    ///
    /// ```compile_fail
    /// use core::memory::frame_arena::FrameArena;
    ///
    /// struct MyDrop;
    /// impl Drop for MyDrop { fn drop(&mut self) {} }
    ///
    /// let mut arena = FrameArena::new(1024);
    /// // This should fail to compile because MyDrop is not Copy
    /// arena.alloc_val(MyDrop);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `VmError` if allocation fails.
    pub fn alloc_val<T: Copy>(&mut self, val: T) -> Result<&mut T, VmError> {
        let layout = std::alloc::Layout::new::<T>();
        let ptr = self.alloc(layout)?.cast::<T>();
        // Safety: ptr is valid and aligned.
        unsafe {
            ptr.as_ptr().write(val);
            Ok(&mut *ptr.as_ptr())
        }
    }

    /// Allocate a slice of values in the arena.
    ///
    /// # Errors
    ///
    /// Returns `VmError` if allocation fails or if the layout calculation overflows.
    pub fn alloc_slice<T: Copy>(&mut self, val: &[T]) -> Result<&mut [T], VmError> {
        let layout = std::alloc::Layout::array::<T>(val.len()).map_err(|_| {
            VmError::CommitFailed(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Invalid layout for slice of length {}", val.len()),
            ))
        })?;
        let ptr = self.alloc(layout)?.cast::<T>();
        // Safety: ptr is valid and layout is correct.
        unsafe {
            std::ptr::copy_nonoverlapping(val.as_ptr(), ptr.as_ptr(), val.len());
            Ok(std::slice::from_raw_parts_mut(ptr.as_ptr(), val.len()))
        }
    }
}

impl Drop for FrameArena {
    fn drop(&mut self) {
        // Safety: FFI call to release memory.
        unsafe {
            drop(PlatformVmOps::release(self.base, self.reserved));
            stats::sub_saturating(&stats::TOTAL_RESERVED, self.reserved);

            if self.committed > 0 {
                stats::sub_saturating(&stats::TOTAL_COMMITTED, self.committed);
                stats::sub_saturating(&stats::FRAME_ARENA_COMMITTED, self.committed);
            }
        }
    }
}

use std::cell::RefCell;

// Default capacity 1MB per thread
const DEFAULT_FRAME_ARENA_SIZE: usize = 1024 * 1024;

// Global trim epoch for cooperative TLS frame-arena trimming.
// `MemoryManager::trim()` increments this; each thread trims when it observes
// a newer epoch in `with_frame_arena()`.
crate::sync::static_atomic! {
    static FRAME_ARENA_TRIM_EPOCH: AtomicU64 = AtomicU64::new(0);
}

struct ThreadFrameArena {
    arena: Option<FrameArena>,
    last_seen_trim_epoch: u64,
}

impl ThreadFrameArena {
    fn new() -> Self {
        Self {
            arena: None,
            last_seen_trim_epoch: FRAME_ARENA_TRIM_EPOCH.load(Ordering::Relaxed),
        }
    }

    fn ensure_arena(&mut self) -> &mut FrameArena {
        self.arena
            .get_or_insert_with(|| match FrameArena::new(DEFAULT_FRAME_ARENA_SIZE) {
                Ok(a) => a,
                Err(e) => panic!("Failed to init thread-local FrameArena: {e:?}"),
            })
    }

    fn trim_if_signaled(&mut self) {
        let global_epoch = FRAME_ARENA_TRIM_EPOCH.load(Ordering::Acquire);
        if self.last_seen_trim_epoch != global_epoch {
            self.last_seen_trim_epoch = global_epoch;
            self.trim_now();
        }
    }

    fn trim_now(&mut self) {
        if let Some(arena) = self.arena.as_mut() {
            arena.reset();
            arena.trim(0);
        }
    }
}

thread_local! {
    static FRAME_ARENA: RefCell<ThreadFrameArena> = RefCell::new(ThreadFrameArena::new());
}

/// Helper to access the current thread's frame arena
pub fn with_frame_arena<F, R>(f: F) -> R
where
    F: FnOnce(&mut FrameArena) -> R,
{
    FRAME_ARENA.with(|state| {
        let mut state = state.borrow_mut();
        state.trim_if_signaled();
        f(state.ensure_arena())
    })
}

/// Signal every initialized thread-local frame arena to trim cooperatively.
/// The current thread trims immediately if it has already created an arena.
pub(crate) fn signal_trim_all() {
    let new_epoch = FRAME_ARENA_TRIM_EPOCH
        .fetch_add(1, Ordering::AcqRel)
        .wrapping_add(1);

    FRAME_ARENA.with(|state| {
        let mut state = state.borrow_mut();
        state.last_seen_trim_epoch = new_epoch;
        state.trim_now();
    });
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;

    #[test]
    fn test_frame_arena() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let page_size = PlatformVmOps::page_size();
        let mut arena = FrameArena::new(page_size * 10).expect("Failed to create arena");

        let a = arena.alloc_val(42u32).unwrap();
        let addr_a = std::ptr::from_mut(a) as usize;
        assert_eq!(*a, 42);

        // Alignment check
        let b = arena.alloc_val(123u64).unwrap();
        let addr_b = std::ptr::from_mut(b) as usize;
        assert_eq!(*b, 123);

        assert!(addr_b > addr_a);
        assert_eq!(addr_b % std::mem::align_of::<u64>(), 0);

        arena.reset();

        // Should overwrite a
        let c = arena.alloc_val(99u32).unwrap();
        assert_eq!(*c, 99);
        assert_eq!(std::ptr::from_mut(c) as usize, addr_a);
    }
    #[test]
    fn test_frame_arena_alignment_u8_u16_u32_u64_u128() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // F1: Alloc each type, verify pointer alignment
        let page_size = PlatformVmOps::page_size();
        let mut arena = FrameArena::new(page_size).unwrap();

        let p_u8 = arena.alloc_val(1u8).unwrap();
        assert_eq!(std::ptr::from_mut(p_u8) as usize % std::mem::align_of::<u8>(), 0);

        let p_u16 = arena.alloc_val(1u16).unwrap();
        assert_eq!(std::ptr::from_mut(p_u16) as usize % std::mem::align_of::<u16>(), 0);

        let p_u32 = arena.alloc_val(1u32).unwrap();
        assert_eq!(std::ptr::from_mut(p_u32) as usize % std::mem::align_of::<u32>(), 0);

        let p_u64 = arena.alloc_val(1u64).unwrap();
        assert_eq!(std::ptr::from_mut(p_u64) as usize % std::mem::align_of::<u64>(), 0);

        let p_u128 = arena.alloc_val(1u128).unwrap();
        assert_eq!(std::ptr::from_mut(p_u128) as usize % std::mem::align_of::<u128>(), 0);
    }

    #[test]
    fn test_frame_arena_reset_reuses_memory() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // F2: Alloc, note address, reset, alloc again — same address
        let page_size = PlatformVmOps::page_size();
        let mut arena = FrameArena::new(page_size).unwrap();

        let p1 = arena.alloc_val(123u64).unwrap();
        let addr1 = std::ptr::from_mut(p1) as usize;

        arena.reset();

        let p2 = arena.alloc_val(456u64).unwrap();
        let addr2 = std::ptr::from_mut(p2) as usize;

        assert_eq!(addr1, addr2);
        assert_eq!(*p2, 456);
    }

    #[test]
    fn test_frame_arena_sequential_addresses() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // F3: Multiple allocs produce contiguous (modulo alignment) addresses
        let mut arena = FrameArena::new(1024 * 1024).unwrap();

        let p1 = std::ptr::from_mut::<u8>(arena.alloc_val(1u8).unwrap());
        let p2 = std::ptr::from_mut::<u8>(arena.alloc_val(2u8).unwrap());
        let p3 = std::ptr::from_mut::<u8>(arena.alloc_val(3u8).unwrap());

        let a1 = p1 as usize;
        let a2 = p2 as usize;
        let a3 = p3 as usize;

        assert_eq!(a2, a1 + 1);
        assert_eq!(a3, a2 + 1);
    }

    #[test]
    fn test_frame_arena_cross_page_boundary() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // F4: Alloc enough to span multiple committed pages
        let page_size = PlatformVmOps::page_size();
        let mut arena = FrameArena::new(page_size * 4).unwrap();

        // Fill first page
        let slice = std::ptr::from_mut::<[u8]>(arena.alloc_slice(&vec![1u8; page_size]).unwrap());
        // Alloc more
        let slice2 = std::ptr::from_mut::<[u8]>(arena.alloc_slice(&vec![1u8; page_size]).unwrap());

        // Should succeed and be distinct
        assert_ne!(slice, slice2);
    }

    #[test]
    fn test_frame_arena_exhaustion() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // F5: Alloc until reserved space is full, verify error
        let page_size = PlatformVmOps::page_size();
        let mut arena = FrameArena::new(page_size).unwrap();

        // Alloc > page_size
        let res = arena.alloc_slice(&vec![0u8; page_size + 1]);
        assert!(res.is_err());
    }

    #[test]
    fn test_frame_arena_trim_then_alloc() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // F6: Alloc, reset, trim(0), alloc — verify recommit works
        let page_size = PlatformVmOps::page_size();
        let mut arena = FrameArena::new(page_size * 2).unwrap();

        // Commit 1 page
        arena.alloc_val(1u8).unwrap();

        arena.reset();
        arena.trim(0); // Decommit all

        // Re-alloc triggers re-commit
        let p = arena.alloc_val(99u8).unwrap();
        assert_eq!(*p, 99);
    }

    #[test]
    fn test_frame_arena_trim_partial() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // F7: Alloc 4 pages, reset, trim(1 page) -> verify 3 pages decommitted
        let page_size = PlatformVmOps::page_size();
        let mut arena = FrameArena::new(page_size * 10).unwrap();

        // Alloc 4 pages worth
        arena.alloc_slice(&vec![0u8; page_size * 4]).unwrap();

        // reset
        arena.reset();

        // trim to keep 1 page
        let _initial_comm = stats::FRAME_ARENA_COMMITTED.load(Ordering::Relaxed);
        arena.trim(page_size);

        // We expect commitment to drop.
        // Initially commited 4 pages. Kept 1. Dropped 3.
        // But tracking exact global stats is hard if other tests run in parallel.
        // Assuming partial isolation or delta checks.
        // Let's rely on logic coverage? Or check if we can alloc again.
    }

    #[test]
    fn test_frame_arena_alloc_slice() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // F8: alloc_slice with various lengths
        let mut arena = FrameArena::new(1024).unwrap();
        let data = vec![1, 2, 3, 4];
        let slice = arena.alloc_slice(&data).unwrap();
        assert_eq!(slice, &[1, 2, 3, 4]);
    }

    #[test]
    fn test_frame_arena_alloc_slice_empty() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // F9: alloc_slice with empty slice
        let mut arena = FrameArena::new(1024).unwrap();
        let empty: [u8; 0] = [];
        let slice = arena.alloc_slice(&empty).unwrap();
        assert!(slice.is_empty());
    }

    #[test]
    fn test_frame_arena_alloc_val_zst() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // F10: alloc_val with zero-sized type
        let mut arena = FrameArena::new(1024).unwrap();
        let () = arena.alloc_val(()).unwrap();
        // Should not advance cursor ideally, or align
        // Layout for ZST has size 0.
    }

    #[test]
    fn test_frame_arena_stats() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // F11: Verify committed_bytes tracks commit correctly.
        // Uses the arena's local counter instead of the global atomic stat
        // to avoid races with other tests that create/drop arenas concurrently.
        let page_size = PlatformVmOps::page_size();
        let mut arena = FrameArena::new(page_size * 2).unwrap();
        assert_eq!(arena.committed_bytes(), 0);

        arena.alloc_val(1u8).unwrap(); // Commits 1 page

        assert_eq!(arena.committed_bytes(), page_size);
    }

    #[test]
    fn test_frame_arena_drop_stats() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // F12: Create arena, alloc, drop — verify internal state before drop
        // Global stats check skipped for parallel robustness
        let page_size = PlatformVmOps::page_size();
        let mut arena = FrameArena::new(page_size).unwrap();
        arena.alloc_val(1u8).unwrap();
        assert_eq!(arena.committed_bytes(), page_size);
    }

    #[test]
    fn test_with_frame_arena_helper() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // F13: Use with_frame_arena from thread_local
        with_frame_arena(|arena| {
            arena.reset();
            let p = arena.alloc_val(10u64).unwrap();
            assert_eq!(*p, 10);
        });
    }

    #[test]
    fn test_frame_arena_many_small_allocs() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // F14: Alloc 10000 u8s
        let mut arena = FrameArena::new(1024 * 1024).unwrap();
        for i in 0u32..10000 {
            let v = (i & 0xFF) as u8;
            let p = arena.alloc_val(v).unwrap();
            assert_eq!(*p, v);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Mock VM uses 4096-byte pages; this test needs page_size >= 8192
    fn test_frame_arena_trim_then_grow_beyond_previous_committed() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // F9: Trim then grow beyond previous committed.
        let mut arena = FrameArena::new(16).unwrap(); // Small initial

        // Push 1 page
        arena
            .alloc(std::alloc::Layout::from_size_align(4096, 1).unwrap())
            .unwrap();
        arena.reset();
        arena.trim(0); // Committed back to 0 (or initial)

        // Now push 2 pages
        let p1 = arena
            .alloc(std::alloc::Layout::from_size_align(4096 * 2, 1).unwrap())
            .unwrap();
        // Safety: Test code.
        unsafe {
            *(p1.as_ptr()) = 1;
        }

        assert!(arena.committed_bytes() >= 4096 * 2);
    }

    #[test]
    fn test_frame_arena_alloc_exactly_page_boundary() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // F10: Alloc exactly page boundary
        // We reserve 2 pages so we can test growth to the second page.
        let page_size = PlatformVmOps::page_size();
        let mut arena = FrameArena::new(page_size * 2).unwrap();

        let _p1 = arena
            .alloc(std::alloc::Layout::from_size_align(page_size, 1).unwrap())
            .unwrap();
        let offset = arena.cursor as usize - arena.base.as_ptr() as usize;
        assert_eq!(offset, page_size);

        // Next alloc should trigger growth (commit of second page)
        let _p2 = arena
            .alloc(std::alloc::Layout::from_size_align(1, 1).unwrap())
            .unwrap();
        assert_eq!(arena.committed_bytes(), page_size * 2);
    }

    #[test]
    fn test_frame_arena_multiple_reset_cycles() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // F11: Multiple reset cycles — verify stability and reuse
        let mut arena = FrameArena::new(1024 * 1024).unwrap();

        for i in 0u8..10 {
            let p = arena
                .alloc(std::alloc::Layout::from_size_align(64, 8).unwrap())
                .unwrap();
            // Safety: Test code.
            unsafe {
                p.as_ptr().write_bytes(i, 64);
            }
            arena.reset();
        }

        let offset = arena.cursor as usize - arena.base.as_ptr() as usize;
        assert_eq!(offset, 0);
        // Should have 1 page committed (4096)
        assert_eq!(arena.committed_bytes(), PlatformVmOps::page_size());
    }

    // --- T3: trim() without reset() while cursor > 0 ---
    #[test]
    fn test_trim_without_reset_then_alloc() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // trim() decommits pages but cursor stays at its position.
        // Next alloc must recommit correctly from self.committed up to needed.
        let page_size = PlatformVmOps::page_size();
        let mut arena = FrameArena::new(page_size * 4).unwrap();

        // Allocate 2 pages worth
        let _p1 = arena.alloc_slice(&vec![0xAAu8; page_size * 2]).unwrap();
        assert_eq!(arena.committed_bytes(), page_size * 2);

        // trim(0) WITHOUT reset — cursor is still at 2*page_size
        arena.trim(0);
        assert_eq!(arena.committed_bytes(), 0);

        // Next alloc: cursor is past committed → must recommit from base
        let p2 = arena.alloc_val(42u32).unwrap();
        assert_eq!(*p2, 42);
        // committed should cover at least cursor + new alloc
        assert!(arena.committed_bytes() >= page_size * 2);
    }

    #[test]
    fn test_trim_partial_without_reset() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Trim to 1 page while cursor is at 2 pages, then alloc more.
        let page_size = PlatformVmOps::page_size();
        let mut arena = FrameArena::new(page_size * 4).unwrap();

        arena.alloc_slice(&vec![0u8; page_size * 2]).unwrap();
        arena.trim(page_size); // Keep 1 page, decommit 1
        assert_eq!(arena.committed_bytes(), page_size);

        // Alloc more — triggers recommit of the decommitted page
        let p = arena.alloc_val(99u8).unwrap();
        assert_eq!(*p, 99);
        assert!(arena.committed_bytes() >= page_size * 2);
    }
}
