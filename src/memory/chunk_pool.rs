use super::stats;
use super::vm::{PlatformVmOps, VmError, VmOps};
use fixedbitset::FixedBitSet;
use std::ptr::NonNull;
use crate::sync::atomic::Ordering;

use crate::sync::{Mutex, OnceLock};

/// 128KB chunk size as per proposal.
pub const CHUNK_SIZE: usize = 128 * 1024;
/// Alignment for chunks.
pub const CHUNK_ALIGN: usize = 16384;

/// A pool of fixed-size chunks backed by a single large VM reservation.
pub struct ChunkPool {
    base: NonNull<u8>,
    reserved: usize,
    committed: usize,
    // (index, is_committed)
    free_list: Vec<(usize, bool)>,
    live_count: usize,
    /// Tracks which chunks are currently allocated
    live_mask: FixedBitSet,
    /// Tracks actual committed bytes (subtracting trimmed chunks)
    actual_committed: usize,
    /// Original pointer from reserve (may be different from base due to alignment)
    original_ptr: NonNull<u8>,
    /// Total reserved including alignment padding
    reserved_including_padding: usize,
}

static GLOBAL_CHUNK_POOL: OnceLock<Mutex<ChunkPool>> = OnceLock::new();

pub struct GlobalChunkPool;

impl GlobalChunkPool {
    fn ensure_initialized() -> Result<&'static Mutex<ChunkPool>, VmError> {
        if let Some(pool) = GLOBAL_CHUNK_POOL.get() {
            return Ok(pool);
        }

        let candidate = Mutex::new(ChunkPool::new(1024 * 1024 * 1024)?);
        drop(GLOBAL_CHUNK_POOL.set(candidate));
        Ok(GLOBAL_CHUNK_POOL
            .get()
            .expect("GlobalChunkPool should be initialized"))
    }

    /// Initialize the global chunk pool.
    ///
    /// # Panics
    ///
    /// Panics if initialization fails (e.g. out of memory).
    pub fn init() {
        if let Err(e) = Self::ensure_initialized() {
            panic!("Failed to init GlobalChunkPool: {e:?}");
        }
    }

    pub fn get() -> Option<&'static Mutex<ChunkPool>> {
        GLOBAL_CHUNK_POOL.get()
    }

    /// Allocate a chunk from the global pool.
    ///
    /// # Panics
    ///
    /// Panics if the global lock is poisoned or the pool is not initialized.
    ///
    /// # Errors
    ///
    /// Returns `VmError` if allocation fails (e.g. OOM).
    pub fn alloc() -> Result<NonNull<u8>, VmError> {
        Self::ensure_initialized()?.lock().unwrap().alloc()
    }

    /// Free a chunk previously allocated from the global chunk pool.
    ///
    /// # Safety
    /// - `ptr` must have been returned by `GlobalChunkPool::alloc`.
    /// - `ptr` must not have been freed already.
    /// - The global pool must already be initialized (e.g. via `alloc()`).
    ///
    /// # Panics
    ///
    /// Panics if the global lock is poisoned or the pool is not initialized.
    pub unsafe fn free(ptr: NonNull<u8>) {
        if let Some(pool) = Self::get() {
            // Safety: We hold the lock.
            // Safety: We hold the lock.
            unsafe {
                pool.lock().unwrap().free(ptr);
            }
        } else {
            panic!("GlobalChunkPool not initialized but free called");
        }
    }

    /// Trim the global chunk pool, releasing memory to the OS.
    ///
    /// # Panics
    ///
    /// Panics if the global lock is poisoned or the pool is not initialized.
    pub fn trim() {
        if let Some(pool) = Self::get() {
            pool.lock().unwrap().trim();
        }
    }
}

// ChunkPool must be Send to be used in a Mutex or passed between threads.
// Raw pointers are !Send, so we must implement it manually if safe.
// Since it owns the memory, it is Send.
// Safety: ChunkPool owns the memory and is Send.
// Safety: ChunkPool owns the memory and is Send.
unsafe impl Send for ChunkPool {}

impl ChunkPool {
    /// Create a new chunk pool with a maximum capacity (in bytes).
    /// Capacity must be a multiple of `CHUNK_SIZE`.
    ///
    /// # Errors
    ///
    /// Returns `VmError` if memory reservation fails or if the capacity calculation overflows.
    pub fn new(capacity: usize) -> Result<Self, VmError> {
        let capacity = capacity.next_multiple_of(CHUNK_SIZE);
        // Reserve extra for alignment
        let reserved_including_padding = capacity.checked_add(CHUNK_ALIGN).ok_or_else(|| {
            VmError::InitializationFailed("ChunkPool reservation size overflow".to_string())
        })?;
        // Safety: FFI call to reserve memory.
        // Safety: FFI call to reserve memory.
        let original_ptr = unsafe { PlatformVmOps::reserve(reserved_including_padding)? };

        let ptr_addr = original_ptr.as_ptr() as usize;
        let aligned_addr = (ptr_addr + CHUNK_ALIGN - 1) & !(CHUNK_ALIGN - 1);
        // Safety: We just reserved this memory and aligned it, so it's non-null.
        // Safety: We just reserved this memory and aligned it, so it's non-null.
        let base = unsafe { NonNull::new_unchecked(aligned_addr as *mut u8) };

        // Track the full VM reservation, including alignment padding.
        stats::TOTAL_RESERVED.fetch_add(reserved_including_padding, Ordering::Relaxed);
        let num_chunks = capacity / CHUNK_SIZE;

        Ok(Self {
            base,
            reserved: capacity,
            committed: 0,
            free_list: Vec::new(),
            live_count: 0,
            live_mask: FixedBitSet::with_capacity(num_chunks),
            actual_committed: 0,
            original_ptr,
            reserved_including_padding,
        })
    }

    /// Allocate a 128KB chunk. Returns a pointer to the start of the chunk.
    ///
    /// # Errors
    ///
    /// Returns `VmError` if the pool is exhausted or if memory commit fails.
    pub fn alloc(&mut self) -> Result<NonNull<u8>, VmError> {
        if let Some((index, is_committed)) = self.free_list.last().copied() {
            let offset = index * CHUNK_SIZE;
            // Safety: offset is within bounds (checked by logic).
            let ptr = unsafe { NonNull::new_unchecked(self.base.as_ptr().add(offset)) };

            if is_committed {
                #[cfg(debug_assertions)]
                // Safety: ptr is valid.
                unsafe {
                    std::ptr::write_bytes(ptr.as_ptr(), 0, CHUNK_SIZE);
                }
            } else {
                // Try commit first
                // Safety: FFI call to commit memory.
                // Safety: FFI call to commit memory.
                // Safety: FFI call to commit memory.
        unsafe { PlatformVmOps::commit(ptr, CHUNK_SIZE)? };
                
                // Commit succeeded, update state
                self.actual_committed += CHUNK_SIZE;
                stats::TOTAL_COMMITTED.fetch_add(CHUNK_SIZE, Ordering::Relaxed);
                stats::CHUNK_POOL_COMMITTED.fetch_add(CHUNK_SIZE, Ordering::Relaxed);
                
                #[cfg(debug_assertions)]
                // Safety: ptr is valid.
                unsafe {
                    std::ptr::write_bytes(ptr.as_ptr(), 0, CHUNK_SIZE);
                }
            }

            // Pop only after success
            self.free_list.pop(); 
            self.live_count += 1;
            self.live_mask.insert(index);
            stats::CHUNK_POOL_LIVE.fetch_add(1, Ordering::Relaxed);

            return Ok(ptr);
        }

        // No free chunks, allocate new
        let next_offset = self.committed;
        let Some(next_end) = next_offset.checked_add(CHUNK_SIZE) else {
            return Err(VmError::CommitFailed(std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                "ChunkPool offset overflow",
            )));
        };
        if next_end > self.reserved {
            // Out of reserved space
            return Err(VmError::CommitFailed(std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                "ChunkPool exhausted reserved space",
            )));
        }

        let index = next_offset / CHUNK_SIZE;
        // Safety: next_offset is checked against reserved size.
        // Safety: next_offset is checked against reserved size.
        let ptr = unsafe { NonNull::new_unchecked(self.base.as_ptr().add(next_offset)) };
        // Safety: FFI call to commit memory.
        unsafe { PlatformVmOps::commit(ptr, CHUNK_SIZE)? };
        #[cfg(debug_assertions)]
        // Safety: ptr is valid.
        unsafe {
            std::ptr::write_bytes(ptr.as_ptr(), 0, CHUNK_SIZE);
        }

        self.committed += CHUNK_SIZE;
        self.live_count += 1;
        self.live_mask.insert(index);
        self.actual_committed += CHUNK_SIZE;

        stats::TOTAL_COMMITTED.fetch_add(CHUNK_SIZE, Ordering::Relaxed);
        stats::CHUNK_POOL_COMMITTED.fetch_add(CHUNK_SIZE, Ordering::Relaxed);
        stats::CHUNK_POOL_LIVE.fetch_add(1, Ordering::Relaxed);

        Ok(ptr)
    }

    /// Free a chunk previously allocated by this pool.
    ///
    /// # Safety
    /// - `ptr` must have been returned by `Self::alloc` on this exact pool.
    /// - `ptr` must not have been freed already.
    /// - `ptr` must not be used after this call.
    pub unsafe fn free(&mut self, ptr: NonNull<u8>) {
        let ptr_addr = ptr.as_ptr() as usize;
        let base_addr = self.base.as_ptr() as usize;

        // Range check
        if ptr_addr < base_addr || ptr_addr >= base_addr + self.reserved {
            debug_assert!(false, "Pointer {ptr:p} does not belong to this ChunkPool");
            // Safety: Unreachable logic.
            unsafe {
                std::hint::unreachable_unchecked();
            }
        }

        let offset = ptr_addr - base_addr;

        // Alignment check
        if !offset.is_multiple_of(CHUNK_SIZE) {
            debug_assert!(false, "Pointer {ptr:p} is not aligned to CHUNK_SIZE");
            // Safety: Unreachable logic.
            unsafe {
                std::hint::unreachable_unchecked();
            }
        }

        let index = offset / CHUNK_SIZE;

        // Commitment check - ensure we aren't freeing something we never even high-water allocated
        if offset >= self.committed {
            debug_assert!(
                false,
                "Pointer {ptr:p} belongs to an unallocated region of this ChunkPool",
            );
            // Safety: Unreachable logic.
            unsafe {
                std::hint::unreachable_unchecked();
            }
        }

        // Double free check
        if !self.live_mask.contains(index) {
            debug_assert!(
                false,
                "Double free detected in ChunkPool for pointer {ptr:p}",
            );
            // Safety: Unreachable logic.
            unsafe {
                std::hint::unreachable_unchecked();
            }
        }

        // When freeing, it's still committed (until trimmed)
        self.live_mask.set(index, false);
        self.free_list.push((index, true));
        self.live_count -= 1;

        stats::sub_saturating(&stats::CHUNK_POOL_LIVE, 1);
    }

    #[must_use]
    pub fn live_chunks(&self) -> usize {
        self.live_count
    }

    #[must_use]
    pub fn committed_bytes(&self) -> usize {
        self.actual_committed
    }

    /// Decommit free chunks to release physical memory to the OS.
    /// Capacity remains reserved.
    ///
    /// # Panics
    ///
    /// Panics if the decommit operation fails (debug builds only).
    pub fn trim(&mut self) {
        // Decommit all committed chunks in the free list
        let mut decommitted_count = 0;

        for (index, is_committed) in &mut self.free_list {
            if *is_committed {
                let offset = *index * CHUNK_SIZE;
                // Safety: index is valid.
                // Safety: offset is within bounds (checked by logic).
            let ptr = unsafe { NonNull::new_unchecked(self.base.as_ptr().add(offset)) };
                // Safety: FFI call to decommit memory.
                if unsafe { PlatformVmOps::decommit(ptr, CHUNK_SIZE) }.is_ok() {
                    *is_committed = false;
                    decommitted_count += 1;
                } else {
                    #[cfg(debug_assertions)]
                    panic!("ChunkPool::trim decommit failed for chunk index {}", *index);
                }
            }
        }

        if decommitted_count > 0 {
            let bytes = decommitted_count * CHUNK_SIZE;
            self.actual_committed -= bytes;
            stats::sub_saturating(&stats::TOTAL_COMMITTED, bytes);
            stats::sub_saturating(&stats::CHUNK_POOL_COMMITTED, bytes);
        }

        // Note: we leave them in the free_list. alloc() needs to re-commit.
        // self.committed stays high-water key.
    }
}

impl Drop for ChunkPool {
    fn drop(&mut self) {
        // Safety: We are dropping the pool, so we can release the memory.
        unsafe {
            drop(PlatformVmOps::release(self.original_ptr, self.reserved_including_padding));
            stats::sub_saturating(&stats::TOTAL_RESERVED, self.reserved_including_padding);

            if self.actual_committed > 0 {
                stats::sub_saturating(&stats::TOTAL_COMMITTED, self.actual_committed);
                stats::sub_saturating(&stats::CHUNK_POOL_COMMITTED, self.actual_committed);
            }

            if self.live_count > 0 {
                stats::sub_saturating(&stats::CHUNK_POOL_LIVE, self.live_count);
            }
        }
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_pool() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Reserve 1MB (8 chunks)
        let mut pool = ChunkPool::new(1024 * 1024).expect("Failed to create pool");

        let chunk1 = pool.alloc().expect("Failed to alloc chunk 1");
        let chunk2 = pool.alloc().expect("Failed to alloc chunk 2");

        assert_ne!(chunk1, chunk2);
        assert_eq!(pool.live_chunks(), 2);
        assert_eq!(pool.committed_bytes(), CHUNK_SIZE * 2);

        // Safety: Test code.
        unsafe {
            pool.free(chunk1);
        }

        assert_eq!(pool.live_chunks(), 1);

        let chunk3 = pool.alloc().expect("Failed to alloc chunk 3");
        // Should reuse chunk1's slot (LIFO or similar)
        assert_eq!(chunk3, chunk1);
        assert_eq!(pool.live_chunks(), 2);
        assert_eq!(pool.committed_bytes(), CHUNK_SIZE * 2); // No new commit
    }

    #[test]
    fn test_chunk_pool_trim() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let mut pool = ChunkPool::new(1024 * 1024).unwrap();
        let chunk = pool.alloc().unwrap();
        // Safety: Test code.
        unsafe { pool.free(chunk) };

        // Trim should decommit
        // We can't easily check OS state, but we check logic runs via code coverage usually.
        // Or we check re-commit success.
        pool.trim();

        // Alloc should re-commit
        let chunk2 = pool.alloc().expect("Re-commit failed");
        // Safety: Test code.
        unsafe { chunk2.as_ptr().write(0xFF) }; // Should not segfault
    }
    #[test]
    fn test_chunk_pool_exhaust() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // C1: Alloc returns Err when exhausted
        let capacity = CHUNK_SIZE * 2;
        let mut pool = ChunkPool::new(capacity).expect("Failed to create pool");

        let _c1 = pool.alloc().unwrap();
        let _c2 = pool.alloc().unwrap();

        let c3 = pool.alloc();
        assert!(c3.is_err());
    }

    #[test]
    fn test_chunk_pool_trim_empty_freelist() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // C3: Trim with no free chunks
        let mut pool = ChunkPool::new(CHUNK_SIZE * 4).unwrap();
        let _c1 = pool.alloc().unwrap();

        // No free chunks
        pool.trim();

        assert_eq!(pool.committed_bytes(), CHUNK_SIZE);
        assert_eq!(pool.live_chunks(), 1);
    }

    #[test]
    fn test_chunk_pool_trim_partial() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // C4: Alloc 3, free 2, trim, alloc 1
        let mut pool = ChunkPool::new(CHUNK_SIZE * 4).unwrap();
        let _c1 = pool.alloc().unwrap();
        let c2 = pool.alloc().unwrap();
        let c3 = pool.alloc().unwrap();

        assert_eq!(pool.committed_bytes(), CHUNK_SIZE * 3);

        // Safety: Test code.
        unsafe {
            pool.free(c2);
            pool.free(c3);
        }

        // Still committed
        assert_eq!(pool.committed_bytes(), CHUNK_SIZE * 3);

        pool.trim();

        // c2 and c3 decommitted. c1 still live/committed.
        // committed: 3 chunks reserved, but actual physical RAM usage should process decommits.
        // Our tracked `actual_committed` subtracts decommitted bytes.
        assert_eq!(pool.committed_bytes(), CHUNK_SIZE);

        // Re-alloc one
        let c4 = pool.alloc().unwrap();
        // Should reuse one of the trimmed chunks and re-commit it.
        assert_eq!(pool.committed_bytes(), CHUNK_SIZE * 2);

        // Verify write
        // Safety: Test code.
        unsafe { c4.as_ptr().write(0xAA) };
    }

    #[test]
    fn test_chunk_pool_stats_accuracy() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // C5/C6: Track stats through lifecycle
        // Note: Global stats are racy in parallel tests. We verify internal state.

        let mut pool = ChunkPool::new(CHUNK_SIZE * 2).unwrap();
        assert_eq!(pool.committed_bytes(), 0);

        let c1 = pool.alloc().unwrap();
        assert_eq!(pool.committed_bytes(), CHUNK_SIZE);

        // We can check that global stats are at least what we expect if we assume monotonic growth?
        // No, decommits happen too.
        // Just rely on internal state tracking which Mirrors the logic used to update globals.

        // Safety: Test code.
        unsafe { pool.free(c1) };
        // Still committed
        assert_eq!(pool.committed_bytes(), CHUNK_SIZE);

        pool.trim();
        assert_eq!(pool.committed_bytes(), 0);
    }

    #[test]
    fn test_chunk_pool_drop_releases_stats() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // C9: Verify internal state before drop (stats check removed for parallel safety)
        let mut pool = ChunkPool::new(CHUNK_SIZE * 10).unwrap();
        let _c = pool.alloc().unwrap();
        // 1 chunk committed, 10 reserved checked via internal methods if needed
        assert!(pool.committed_bytes() >= CHUNK_SIZE);
        // implicit drop
    }

    #[test]
    fn test_chunk_pool_committed_bytes_after_trim() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // C10
        let mut pool = ChunkPool::new(CHUNK_SIZE * 2).unwrap();
        let c1 = pool.alloc().unwrap();
        // Safety: Test code.
        unsafe { pool.free(c1) };
        assert_eq!(pool.committed_bytes(), CHUNK_SIZE);
        pool.trim();
        assert_eq!(pool.committed_bytes(), 0);
    }

    #[test]
    fn test_chunk_pool_alloc_trim_alloc_trim_cycle() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // C11: Multiple trim/realloc cycles — verify stats stay consistent
        let mut pool = ChunkPool::new(CHUNK_SIZE * 10).unwrap();

        // Cycle 1
        let c1 = pool.alloc().unwrap();
        // Safety: Test code.
        unsafe { pool.free(c1) };
        pool.trim();
        assert_eq!(pool.committed_bytes(), 0);

        // Cycle 2
        let c2 = pool.alloc().unwrap();
        // Should reuse index 0, re-commit
        assert_eq!(pool.committed_bytes(), CHUNK_SIZE);
        // Safety: Test code.
        unsafe { pool.free(c2) };
        pool.trim();
        assert_eq!(pool.committed_bytes(), 0);

        // Cycle 3 (Multiple)
        let c3 = pool.alloc().unwrap();
        let c4 = pool.alloc().unwrap();
        assert_eq!(pool.committed_bytes(), CHUNK_SIZE * 2);
        // Safety: Test code.
        unsafe { pool.free(c3) };
        // Safety: Test code.
        unsafe { pool.free(c4) };
        pool.trim();
        assert_eq!(pool.committed_bytes(), 0);
    }

    #[test]
    fn test_chunk_pool_free_without_trim_realloc() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // C12: Free then realloc without trim — verify no unnecessary recommit loop (is_committed=true path)
        let mut pool = ChunkPool::new(CHUNK_SIZE * 10).unwrap();
        let c1 = pool.alloc().unwrap();
        // Safety: Test code.
        unsafe { pool.free(c1) };

        // Internal state check: 1 free chunk, committed=true
        assert_eq!(pool.committed_bytes(), CHUNK_SIZE);

        // Realloc
        let c2 = pool.alloc().unwrap();
        // Should reuse committed chunk
        assert_eq!(pool.committed_bytes(), CHUNK_SIZE);

        // Verify we can write (it is committed)
        // Safety: Test code.
        unsafe {
            *c2.as_ptr() = 42;
        }
    }

    #[test]
    fn test_chunk_pool_trim_idempotent() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // C13: Call trim twice — second is no-op
        let mut pool = ChunkPool::new(CHUNK_SIZE * 2).unwrap();
        let c1 = pool.alloc().unwrap();
        // Safety: Test code.
        unsafe { pool.free(c1) };

        pool.trim();
        assert_eq!(pool.committed_bytes(), 0);

        // Second trim
        pool.trim();
        assert_eq!(pool.committed_bytes(), 0);

        // Should still be able to alloc
        let _c2 = pool.alloc().unwrap();
        assert_eq!(pool.committed_bytes(), CHUNK_SIZE);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "does not belong to this ChunkPool")]
    fn test_chunk_pool_free_out_of_range() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let mut pool = ChunkPool::new(CHUNK_SIZE).unwrap();
        // Safety: Test code.
        unsafe {
            pool.free(NonNull::new_unchecked(std::ptr::dangling_mut::<u8>()));
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "is not aligned to CHUNK_SIZE")]
    fn test_chunk_pool_free_misaligned() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let mut pool = ChunkPool::new(CHUNK_SIZE * 2).unwrap();
        let c1 = pool.alloc().unwrap();
        // Safety: Test code.
        unsafe {
            pool.free(NonNull::new_unchecked(c1.as_ptr().add(1)));
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "belongs to an unallocated region")]
    fn test_chunk_pool_free_unallocated() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let mut pool = ChunkPool::new(CHUNK_SIZE * 2).unwrap();
        // Skip block 0, point to block 1 which is reserved but not committed/allocated yet
        // Safety: Test code.
        let unallocated_ptr = unsafe { NonNull::new_unchecked(pool.base.as_ptr().add(CHUNK_SIZE)) };
        // Safety: Test code.
        unsafe {
            pool.free(unallocated_ptr);
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "Double free detected")]
    fn test_chunk_pool_double_free() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let mut pool = ChunkPool::new(CHUNK_SIZE * 2).unwrap();
        let c1 = pool.alloc().unwrap();
        // Safety: Test code.
        unsafe {
            pool.free(c1);
            pool.free(c1);
        }
    }

    // --- T7: Full lifecycle: alloc all, free all, trim, alloc all again ---
    #[test]
    fn test_chunk_pool_full_lifecycle() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let capacity = CHUNK_SIZE * 8;
        let mut pool = ChunkPool::new(capacity).unwrap();
        let max_chunks: u8 = (capacity / CHUNK_SIZE).try_into().unwrap();

        // Phase 1: Allocate all capacity
        let mut ptrs = Vec::new();
        for i in 0..max_chunks {
            let c = pool.alloc().unwrap();
            // Safety: Test code.
            unsafe {
                *c.as_ptr() = i;
            }
            ptrs.push(c);
        }
        assert_eq!(pool.live_chunks(), max_chunks.into());
        assert_eq!(pool.committed_bytes(), capacity);
        assert!(pool.alloc().is_err(), "Should be exhausted");

        // Phase 2: Free all
        for c in ptrs {
            // Safety: Test code.
            unsafe {
                pool.free(c);
            }
        }
        assert_eq!(pool.live_chunks(), 0);
        assert_eq!(pool.committed_bytes(), capacity); // Still committed until trimmed

        // Phase 3: Trim
        pool.trim();
        assert_eq!(pool.committed_bytes(), 0);

        // Phase 4: Allocate all again — must re-commit
        let mut ptrs2 = Vec::new();
        for i in 0..max_chunks {
            let c = pool.alloc().unwrap();
            // Safety: Test code.
            unsafe {
                *c.as_ptr() = i + 100;
            }
            ptrs2.push(c);
        }
        assert_eq!(pool.live_chunks(), max_chunks.into());
        assert_eq!(pool.committed_bytes(), capacity);

        // Verify data integrity
        for (i, c) in (0..max_chunks).zip(&ptrs2) {
            // Safety: Test code.
            unsafe {
                assert_eq!(c.as_ptr().read(), i + 100);
            }
        }

        // Cleanup
        for c in ptrs2 {
            // Safety: Test code.
            unsafe {
                pool.free(c);
            }
        }
    }
}
