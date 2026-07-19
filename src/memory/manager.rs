use super::binned::GlobalBinnedAllocator;
use super::chunk_pool::GlobalChunkPool;
use super::command_arena::GlobalSharedPagePool;
use super::frame_arena;
use super::stats;
use crate::sync::atomic::Ordering;

pub struct MemoryStats {
    pub total_reserved: usize,
    pub total_committed: usize,
    pub chunk_pool_committed: usize,
    pub chunk_pool_live: usize,
    pub frame_arena_committed: usize,
    pub binned_allocator_committed: usize,
    pub large_alloc_cache_committed: usize,
    pub command_arena_committed: usize,
}

/// Stateless facade over the global memory subsystems.
///
/// The subsystems are process-wide singletons (`GlobalBinnedAllocator`,
/// `GlobalChunkPool`, `GlobalSharedPagePool`, per-thread frame arenas), so
/// there is nothing to construct or own here — both operations are
/// associated functions.
pub struct MemoryManager;

impl MemoryManager {
    /// Release all unused memory to the OS. Frame arenas trim
    /// cooperatively: other threads flush the next time they touch their
    /// arena.
    pub fn trim() {
        GlobalBinnedAllocator::trim();
        GlobalSharedPagePool::trim();
        GlobalChunkPool::trim();
        frame_arena::signal_trim_all();
    }

    /// Snapshot the diagnostic counters. Values are `Relaxed` reads:
    /// individually eventually-consistent, and cross-counter sums may be
    /// transiently inconsistent (see `stats`).
    #[must_use]
    pub fn stats() -> MemoryStats {
        MemoryStats {
            total_reserved: stats::TOTAL_RESERVED.load(Ordering::Relaxed),
            total_committed: stats::TOTAL_COMMITTED.load(Ordering::Relaxed),
            chunk_pool_committed: stats::CHUNK_POOL_COMMITTED.load(Ordering::Relaxed),
            chunk_pool_live: stats::CHUNK_POOL_LIVE.load(Ordering::Relaxed),
            frame_arena_committed: stats::FRAME_ARENA_COMMITTED.load(Ordering::Relaxed),
            binned_allocator_committed: stats::BINNED_ALLOCATOR_COMMITTED.load(Ordering::Relaxed),
            large_alloc_cache_committed: stats::LARGE_ALLOC_CACHE_COMMITTED.load(Ordering::Relaxed),
            command_arena_committed: stats::COMMAND_ARENA_COMMITTED.load(Ordering::Relaxed),
        }
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::super::chunk_pool::ChunkPool;
    use super::super::frame_arena::with_frame_arena;
    use super::*;

    #[test]
    fn test_memory_manager_integration() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // M1: Snapshot is readable
        let _stats = MemoryManager::stats();
    }

    #[test]
    fn test_memory_stats_aggregation() {
        let _guard = crate::memory::TEST_MUTEX.write().unwrap();
        // M2: Alloc from subsystem, verify manager stats
        let initial = MemoryManager::stats().chunk_pool_committed;

        // Scope to force drop
        {
            let mut pool = ChunkPool::new(128 * 1024).unwrap();
            let _c = pool.alloc().unwrap();

            let current = MemoryManager::stats();
            assert!(current.chunk_pool_committed >= initial + 128 * 1024);
        }

        // After drop (and implicit release/trim by Drop impls if any)
        // ChunkPool Drop releases reservation and stats.
        let final_stats = MemoryManager::stats();
        assert_eq!(final_stats.chunk_pool_committed, initial);
    }

    #[test]
    fn test_memory_manager_trim() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // M3: Verify trim runs against all global subsystems without panicking
        MemoryManager::trim();
    }

    #[test]
    fn test_memory_manager_trim_includes_current_thread_frame_arena() {
        let _guard = crate::memory::TEST_MUTEX.write().unwrap();

        // Assert on THIS thread's arena, not the global gauge: other tests'
        // threads release their TLS arenas in thread-death destructors that
        // run after those tests drop TEST_MUTEX, so the global
        // FRAME_ARENA_COMMITTED value can move even under the write lock.
        with_frame_arena(|arena| {
            arena.reset();
            let _ = arena.alloc_val(1u64).unwrap();
            assert!(arena.committed_bytes() > 0);
        });

        // Trim signals all threads; the current thread trims immediately.
        MemoryManager::trim();

        let committed = with_frame_arena(|arena| arena.committed_bytes());
        assert_eq!(
            committed, 0,
            "MemoryManager::trim did not trim the current thread's frame arena"
        );
    }
}
