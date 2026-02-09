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

/// Central manager for memory subsystems.
pub struct MemoryManager {
    // In a real engine, these might be Arc<Mutex<...>> or similar shared references.
    // For this implementation, we just define the structure.
    // To actually manage them, we'd need to register them or own them.
    // Proposal: "Trim all subsystems... Call after level unload..."

    // We'll use a registry pattern or just static access in a real engine.
    // Here we'll just provide the methods that WOULD operate on them.
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryManager {
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }

    /// Release all unused memory to the OS.
    pub fn trim() {
        GlobalBinnedAllocator::trim();
        GlobalSharedPagePool::trim();
        GlobalChunkPool::trim();
        frame_arena::signal_trim_all();
    }

    pub fn stats(&self) -> MemoryStats {
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
        // M1: Create manager
        let manager = MemoryManager::new();
        let _stats = manager.stats();
    }

    #[test]
    fn test_memory_stats_aggregation() {
        let _guard = crate::memory::TEST_MUTEX.write().unwrap();
        // M2: Alloc from subsystem, verify manager stats
        let manager = MemoryManager::new();
        let initial = manager.stats().chunk_pool_committed;
        let initial_total = manager.stats().total_committed;

        // Scope to force drop
        {
            let mut pool = ChunkPool::new(128 * 1024).unwrap();
            let _c = pool.alloc().unwrap();

            let current = manager.stats();
            assert!(current.chunk_pool_committed >= initial + 128 * 1024);
            assert!(current.total_committed >= initial_total);
        }

        // After drop (and implicit release/trim by Drop impls if any)
        // ChunkPool Drop releases reservation and stats.
        let final_stats = manager.stats();
        assert_eq!(final_stats.chunk_pool_committed, initial);
    }

    #[test]
    fn test_memory_manager_trim() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // M3: Verify trim doesn't panic (logic placeholder)
        MemoryManager::trim();
    }

    #[test]
    fn test_memory_manager_trim_includes_current_thread_frame_arena() {
        let _guard = crate::memory::TEST_MUTEX.write().unwrap();

        with_frame_arena(|arena| {
            arena.reset();
            let _ = arena.alloc_val(1u64).unwrap();
        });

        let before = stats::FRAME_ARENA_COMMITTED.load(Ordering::Relaxed);
        assert!(before > 0);

        MemoryManager::trim();

        let after = stats::FRAME_ARENA_COMMITTED.load(Ordering::Relaxed);
        assert_eq!(after, 0);
    }
}
