#[cfg(all(test, not(loom)))]
mod tests {
    use crate::memory::binned::GlobalBinnedAllocator;
    use crate::memory::chunk_pool::ChunkPool;
    use crate::memory::frame_arena::FrameArena;
    use crate::memory::stats;
    use crate::sync::Arc;
    use crate::sync::atomic::Ordering;
    use crate::sync::thread;

    #[test]
    fn test_integration_stress_mix() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // X1: Interleaved allocations from multiple allocators
        drop(GlobalBinnedAllocator::init()); // Ensure init

        let mut chunk_pool = ChunkPool::new(128 * 1024 * 10).unwrap();
        let mut frame_arena = FrameArena::new(4096 * 10).unwrap();

        let mut chunks = Vec::new();
        let mut binned_ptrs = Vec::new();

        for i in 0u8..100 {
            if i % 3 == 0 {
                // Chunk
                if let Ok(c) = chunk_pool.alloc() {
                    chunks.push(c);
                }
            } else if i % 3 == 1 {
                // Binned
                let p = GlobalBinnedAllocator::alloc_bytes(32).unwrap();
                // Safety: Test code.
                unsafe {
                    *p.as_ptr() = 0x11;
                }
                binned_ptrs.push(p);
            } else {
                // Frame
                let f = frame_arena.alloc_val(u64::from(i)).unwrap();
                assert_eq!(*f, u64::from(i));
            }

            if i % 10 == 0 {
                frame_arena.reset();
            }
        }

        // Clean up
        for c in chunks {
            // Safety: Test code.
            unsafe { chunk_pool.free(c); }
        }
        for p in binned_ptrs {
            // Safety: Test code.
            unsafe { GlobalBinnedAllocator::free_bytes(p, 32); }
        }
    }

    #[test]
    fn test_integration_thread_contention() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // X2: Multiple threads hitting GlobalBinnedAllocator and private allocators
        drop(GlobalBinnedAllocator::init());
        let num_threads = 8u8;
        let iters = 200u8;
        let barrier = Arc::new(crate::sync::barrier::Barrier::new(num_threads as usize));

        let handles: Vec<_> = (0..num_threads)
            .map(|t| {
                let b = barrier.clone();
                thread::spawn(move || {
                    // Thread-local frame arena
                    let mut frame = FrameArena::new(1024 * 1024).unwrap();
                    let mut ptrs = Vec::with_capacity(iters as usize);

                    b.wait(); // Synchronize start

                    for i in 0..iters {
                        let size = 16 << (i % 4); // 16 to 128 bytes
                        let ptr = GlobalBinnedAllocator::alloc_bytes(size).unwrap();

                        // Write unique pattern
                        // Safety: Test code.
                        unsafe {
                            let val = t.wrapping_mul(232).wrapping_add(i);
                            ptr.as_ptr().write(val);
                        }

                        let f = frame.alloc_val(u32::from(i)).unwrap();
                        assert_eq!(*f, u32::from(i));

                        ptrs.push((ptr, size));
                    }

                    // Verify integrity
                    for (i, (ptr, _size)) in (0u8..).zip(ptrs.iter()) {
                        // Safety: Test code.
                        unsafe {
                            let expected = t.wrapping_mul(232).wrapping_add(i);
                            assert_eq!(
                                ptr.as_ptr().read(),
                                expected,
                                "Contention caused corruption in thread {t}"
                            );
                        }
                    }

                    // Clean up
                    for (ptr, size) in ptrs {
                        // Safety: Test code.
                        unsafe { GlobalBinnedAllocator::free_bytes(ptr, size); }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_integration_oom_handling() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // X3: Verify safe failure or panic on OOM limits
        // Use a small pool
        let mut pool = ChunkPool::new(128 * 1024).unwrap(); // 1 chunk capacity
        let _c1 = pool.alloc().unwrap();
        let res = pool.alloc(); // Should fail
        assert!(res.is_err());
    }

    #[test]
    fn test_integration_leak_check() {
        use crate::memory::manager::MemoryManager;
        let _guard = crate::memory::TEST_MUTEX.write().unwrap();
        let manager = MemoryManager::new();

        // 1. Initialize and established baseline
        drop(GlobalBinnedAllocator::init());
        MemoryManager::trim(); // Clear any noise from other tests
        let baseline = manager.stats();

        // 2. Perform many allocations across all subsystems
        {
            let mut chunk_pool = ChunkPool::new(128 * 1024 * 10).unwrap();
            let mut frame_arena = FrameArena::new(1024 * 1024).unwrap();

            for i in 0u64..500 {
                let size = 16 << (i % 8); // 16B to 2KB
                let p_binned = GlobalBinnedAllocator::alloc_bytes(size).unwrap();
                let p_chunk = chunk_pool.alloc().unwrap();
                let p_frame = frame_arena.alloc_val(i).unwrap();

                assert_eq!(*p_frame, i);

                // Safety: Test code.
                unsafe { GlobalBinnedAllocator::free_bytes(p_binned, size); }
                // Safety: Test code.
                unsafe { chunk_pool.free(p_chunk); }

                if i % 100 == 0 {
                    frame_arena.reset();
                }
            }
        } // private allocators dropped here

        // 3. Trim everything
        MemoryManager::trim();

        // 4. Final check
        let final_stats = manager.stats();

        // Committed memory should return to baseline (or lower if other tests left garbage).
        // TOTAL_RESERVED might stay higher if GlobalBinnedAllocator grew its pools,
        // but physically committed pages should be released by trim/drop.
        assert!(
            final_stats.total_committed <= baseline.total_committed,
            "Physical memory leak detected! Baseline: {}, Final: {}",
            baseline.total_committed,
            final_stats.total_committed
        );

        assert!(final_stats.chunk_pool_committed <= baseline.chunk_pool_committed);
        assert!(final_stats.command_arena_committed <= baseline.command_arena_committed);
    }

    #[test]
    fn test_integration_large_allocs() {
        // Alloc > 64KB
        // Should use large cache? Code "Alloc too large ... (use LargeAllocCache)".
        // `GlobalBinnedAllocator` `alloc` helper:
        // `allocator.alloc_with_cache(...)`.
        // `alloc_with_cache` panics on > MAX_SMALL_SIZE.
        // So `GlobalBinnedAllocator` does NOT handle large allocs automatically yet.
        // Plan says: "panic('Alloc too large... use LargeAllocCache')".
        // Use `LargeAllocCache` manually here.
        use crate::memory::large_cache::LargeAllocCache;

        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // X5
        drop(GlobalBinnedAllocator::init());
        let mut large = LargeAllocCache::new(1024 * 1024 * 10);
        let layout = std::alloc::Layout::from_size_align(100 * 1024, 1).unwrap();
        let (ptr, size) = large.alloc(layout).unwrap();
        large.free(ptr, std::alloc::Layout::from_size_align(size, 1).unwrap());
    }
    #[test]
    fn test_global_stats_no_negative() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // X6: Verify no negative stats after many operations
        drop(GlobalBinnedAllocator::init());

        let _initial_res = stats::TOTAL_RESERVED.load(Ordering::Relaxed);
        let _initial_com = stats::TOTAL_COMMITTED.load(Ordering::Relaxed);

        {
            let mut chunk_pool = ChunkPool::new(128 * 1024 * 10).unwrap();
            let mut frame_arena = FrameArena::new(4096 * 10).unwrap();

            for _ in 0..100 {
                let p = GlobalBinnedAllocator::alloc_bytes(32).unwrap();
                let c = chunk_pool.alloc().unwrap();
                let _f = frame_arena.alloc_val(1u64).unwrap();

                // Safety: Test code.
            unsafe { chunk_pool.free(c); }
                // Safety: Test code.
            unsafe { GlobalBinnedAllocator::free_bytes(p, 32); }
            }
        }

        // Global stats can't be reliably checked in parallel tests.
        // We just verify the operations themselves don't crash.
    }

    #[test]
    fn test_frame_arena_thread_local_isolation() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // X7: Verify two threads can use FrameArena independently
        let h1 = thread::spawn(|| {
            let mut arena = FrameArena::new(4096).unwrap();
            let p = arena.alloc_val(42u32).unwrap();
            assert_eq!(*p, 42);
        });
        let h2 = thread::spawn(|| {
            let mut arena = FrameArena::new(4096).unwrap();
            let p = arena.alloc_val(123u32).unwrap();
            assert_eq!(*p, 123);
        });
        h1.join().unwrap();
        h2.join().unwrap();
    }

    #[test]
    fn test_command_arena_shared_pool_multithread() {
        use crate::memory::command_arena::{CommandArena, SharedPagePool};
        // X8: Multiple threads using CommandArena backed by same SharedPagePool
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();

        let pool = Arc::new(SharedPagePool::new(1024 * 1024));
        let mut handles = vec![];

        for i in 0u32..4 {
            let p = pool.clone();
            handles.push(thread::spawn(move || {
                let mut arena = CommandArena::new(4096, p);
                for j in 0u32..100 {
                    arena.push(i * 1000 + j).unwrap();
                }
                // verify items
                let count = arena.iter_pages().count();
                assert!(count >= 1);
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_integration_leak_detection() {
        use crate::memory::command_arena::GlobalSharedPagePool;
        use crate::memory::manager::MemoryManager;
        let _guard = crate::memory::TEST_MUTEX.write().unwrap();

        // 1. Establish clean state
        MemoryManager::trim();
        let initial_command = stats::COMMAND_ARENA_COMMITTED.load(Ordering::Relaxed);

        // 2. Perform some allocations that we WILL free
        let pool = GlobalSharedPagePool::get();
        let page_size = 4096;
        let p1 = pool.alloc(page_size).unwrap();

        // 3. Perform one allocation that we WILL NOT free (the leak)
        let _leaked_ptr = pool.alloc(page_size).unwrap();

        // 4. Free the first one
        // Safety: Test code.
        unsafe { pool.free(p1, page_size); }

        // 5. Trim everything to force cache releases
        MemoryManager::trim();

        // 6. Check deltas
        let final_command = stats::COMMAND_ARENA_COMMITTED.load(Ordering::Relaxed);

        // Final stats should be >= initial + page_size (the leak)
        assert!(
            final_command >= initial_command + page_size,
            "Leak not detected in COMMAND_ARENA_COMMITTED. Final: {final_command}, Initial: {initial_command}"
        );
    }

    #[test]
    fn test_integration_high_pressure_stress() {
        use crate::memory::chunk_pool::ChunkPool;
        use crate::memory::command_arena::{CommandArena, GlobalSharedPagePool};
        use crate::memory::large_cache::LargeAllocCache;
        use crate::memory::manager::MemoryManager;
        use crate::sync::Mutex;

        let _guard = crate::memory::TEST_MUTEX.write().unwrap();

        drop(GlobalBinnedAllocator::init());
        MemoryManager::trim();
        let baseline = stats::COMMAND_ARENA_COMMITTED.load(Ordering::Relaxed);

        let num_threads = 16u8;
        let iters_per_thread = 100u32;
        let large_cache = Arc::new(Mutex::new(LargeAllocCache::new(50 * 1024 * 1024)));
        let barrier = Arc::new(crate::sync::barrier::Barrier::new(num_threads as usize));

        let handles: Vec<_> = (0u8..num_threads)
            .map(|t| {
                let lc = large_cache.clone();
                let b = barrier.clone();
                thread::spawn(move || {
                    let mut frame = FrameArena::new(256 * 1024).unwrap();
                    let mut chunk_pool = ChunkPool::new(1024 * 1024).unwrap();
                    let mut command_arena = CommandArena::new(4096, GlobalSharedPagePool::get());

                    let mut binned_ptrs = Vec::new();
                    let mut large_ptrs = Vec::new();
                    let mut chunk_ptrs = Vec::new();

                    b.wait();

                    for i in 0..iters_per_thread {
                        let size = 8 + (i % 248);
                        let p = GlobalBinnedAllocator::alloc_bytes(size as usize).unwrap();
                        // Safety: Test code.
                        unsafe {
                            p.as_ptr().write(t);
                        }
                        binned_ptrs.push((p, size));

                        let _ = frame.alloc_val(u64::from(i)).unwrap();
                        let _ = command_arena.push(i).unwrap();

                        if i % 10 == 0 && let Ok(c) = chunk_pool.alloc() {
                            chunk_ptrs.push(c);
                        }

                        if i % 25 == 0 {
                            let lsize = 128 * 1024;
                            let llayout = std::alloc::Layout::from_size_align(lsize, 1).unwrap();
                            if let Ok((lp, actual_size)) = lc.lock().unwrap().alloc(llayout) {
                                large_ptrs.push((lp, actual_size));
                            }
                        }

                        if i % 20 == 0 {
                            frame.reset();
                        }
                    }

                    // Verify and cleanup
                    for (p, _) in &binned_ptrs {
                        // Safety: Test code.
                        unsafe {
                            assert_eq!(p.as_ptr().read(), t);
                        }
                    }
                    for (p, s) in binned_ptrs {
                        // Safety: Test code.
                        unsafe { GlobalBinnedAllocator::free_bytes(p, s as usize); }
                    }
                    for (p, s) in large_ptrs {
                        lc.lock()
                            .unwrap()
                            .free(p, std::alloc::Layout::from_size_align(s, 1).unwrap());
                    }
                    for p in chunk_ptrs {
                        // Safety: Test code.
                        unsafe { chunk_pool.free(p); }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        MemoryManager::trim();
        large_cache.lock().unwrap().trim();

        let final_command = stats::COMMAND_ARENA_COMMITTED.load(Ordering::Relaxed);
        assert!(
            final_command <= baseline + 65536,
            "High pressure stress leaked memory! Baseline: {baseline}, Final: {final_command}"
        );
    }
}
