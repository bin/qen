/// Loom-based concurrency tests.
///
/// Run w/ `RUSTFLAGS="--cfg loom" cargo test --lib --release`
///
/// Exercise lock-free and Mutex-protected data structures
/// under every possible thread interleaving that loom can explore.
/// (At least I think)
///
/// # Design notes
///
/// Loom exhaustively enumerates thread interleavings, so:
///   - Thread counts kept to 2–3 (state space is exponential).
///   - Loop iterations minimised to 1–3 per thread.
///   - Tests that use BinnedAllocator create fresh instance per iteration
///     (BinnedAllocator::new() goes through VmOps mock under cfg(loom)).
///   - GlobalBinnedAllocator NOT tested directly bc its OnceLock
///     static does not reset between loom iterations.  All concurrency
///     it exercises (Pool Mutex, GlobalRecycler CAS, ThreadCache flush) is
///     reachable through instance-based BinnedAllocator tests.
///   - Recycler's spin-loop on odd generations causes state-space
///     explosion; recycler tests use `preemption_bound(2)`.
#[cfg(loom)]
mod tests {
    use crate::sync::atomic::Ordering;
    use crate::sync::Arc;

    // =====================================================================
    // Helpers
    // =====================================================================

    /// Allocate a 64-byte aligned buffer and initialise the loom-tracked
    /// `AtomicUsize` at the recycler link offset (bytes 8..16).
    /// By putting a loom-tracked `AtomicUsize` in memory at exact offset
    /// `GlobalRecycler` expects allows loom to track "invisible" link ptrs.
    fn alloc_fake_node() -> (std::ptr::NonNull<u8>, std::alloc::Layout) {
        let layout = std::alloc::Layout::from_size_align(64, 64).unwrap();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        let nn = std::ptr::NonNull::new(ptr).expect("alloc failed");

        unsafe { nn.as_ptr().cast::<usize>().write(0); }

        // Place a loom-tracked AtomicUsize at offset 8 (recycler link field).
        unsafe {
            let link_ptr = nn.as_ptr().add(std::mem::size_of::<usize>())
                .cast::<crate::sync::atomic::AtomicUsize>();
            std::ptr::write(link_ptr, crate::sync::atomic::AtomicUsize::new(0));
        }

        (nn, layout)
    }

    fn bounded(preemption: usize) -> loom::model::Builder {
        let mut b = loom::model::Builder::new();
        b.preemption_bound = Some(preemption);
        b
    }

    // =====================================================================
    // 1. stats::Counter
    // =====================================================================

    #[test]
    fn loom_counter_concurrent_add_sub() {
        use crate::memory::stats::Counter;

        loom::model(|| {
            let counter = Arc::new(Counter::new());
            let c1 = counter.clone();
            let c2 = counter.clone();

            let t1 = loom::thread::spawn(move || {
                c1.add(10);
                c1.add(5);
            });

            let t2 = loom::thread::spawn(move || {
                c2.sub(3);
                c2.add(8);
            });

            t1.join().unwrap();
            t2.join().unwrap();

            // 10 + 5 - 3 + 8 = 20
            assert_eq!(counter.get(), 20);
        });
    }

    // =====================================================================
    // 2. GlobalRecycler — 128-bit DWCAS Treiber stack
    // =====================================================================

    #[test]
    fn loom_recycler_push_pop_single_thread() {
        use crate::memory::binned::GlobalRecycler;

        bounded(2).check(|| {
            let recycler = GlobalRecycler::new(16);
            let (node, layout) = alloc_fake_node();

            let overflow = recycler.push(0, node, 1);
            assert!(overflow.is_none(), "should not overflow with limit 16");

            let popped = recycler.pop(0);
            assert!(popped.is_some());
            assert_eq!(popped.unwrap().as_ptr(), node.as_ptr());
            assert!(recycler.pop(0).is_none());

            unsafe { std::alloc::dealloc(node.as_ptr(), layout); }
        });
    }

    #[test]
    fn loom_recycler_concurrent_push() {
        use crate::memory::binned::GlobalRecycler;

        bounded(2).check(|| {
            let recycler = Arc::new(GlobalRecycler::new(16));

            let (node_a, layout_a) = alloc_fake_node();
            let (node_b, layout_b) = alloc_fake_node();

            let r1 = recycler.clone();
            let r2 = recycler.clone();
            let na = node_a.as_ptr() as usize;
            let nb = node_b.as_ptr() as usize;

            let t1 = loom::thread::spawn(move || {
                let node = std::ptr::NonNull::new(na as *mut u8).unwrap();
                r1.push(0, node, 1);
            });

            let t2 = loom::thread::spawn(move || {
                let node = std::ptr::NonNull::new(nb as *mut u8).unwrap();
                r2.push(0, node, 1);
            });

            t1.join().unwrap();
            t2.join().unwrap();

            let mut popped = Vec::new();
            while let Some(p) = recycler.pop(0) {
                popped.push(p.as_ptr() as usize);
            }
            assert_eq!(popped.len(), 2);
            assert!(popped.contains(&na));
            assert!(popped.contains(&nb));

            unsafe {
                std::alloc::dealloc(node_a.as_ptr(), layout_a);
                std::alloc::dealloc(node_b.as_ptr(), layout_b);
            }
        });
    }

    #[test]
    fn loom_recycler_push_while_pop() {
        use crate::memory::binned::GlobalRecycler;

        bounded(2).check(|| {
            let recycler = Arc::new(GlobalRecycler::new(16));

            let (node_a, layout_a) = alloc_fake_node();
            let (node_b, layout_b) = alloc_fake_node();

            recycler.push(0, node_a, 1);

            let r_push = recycler.clone();
            let r_pop = recycler.clone();
            let nb = node_b.as_ptr() as usize;

            let t_push = loom::thread::spawn(move || {
                let node = std::ptr::NonNull::new(nb as *mut u8).unwrap();
                r_push.push(0, node, 1);
            });

            let t_pop = loom::thread::spawn(move || {
                r_pop.pop(0)
            });

            t_push.join().unwrap();
            let popped = t_pop.join().unwrap();

            let mut remaining = Vec::new();
            while let Some(p) = recycler.pop(0) {
                remaining.push(p.as_ptr() as usize);
            }

            let total = remaining.len() + usize::from(popped.is_some());
            assert_eq!(total, 2);

            unsafe {
                std::alloc::dealloc(node_a.as_ptr(), layout_a);
                std::alloc::dealloc(node_b.as_ptr(), layout_b);
            }
        });
    }

    // =====================================================================
    // 3. SharedPagePool (Mutex-protected) — migrated from command_arena.rs
    // =====================================================================

    /// Two threads alloc+free from same SharedPagePool.
    /// Exercises Mutex acquire/release ordering.
    #[test]
    fn loom_shared_page_pool_concurrent() {
        use crate::memory::command_arena::SharedPagePool;

        loom::model(|| {
            let pool = Arc::new(SharedPagePool::new(1024 * 1024));
            let p1 = pool.clone();
            let p2 = pool.clone();

            let t1 = loom::thread::spawn(move || {
                if let Ok(ptr) = p1.alloc(4096) {
                    unsafe { p1.free(ptr, 4096); }
                }
            });

            let t2 = loom::thread::spawn(move || {
                if let Ok(ptr) = p2.alloc(4096) {
                    unsafe { p2.free(ptr, 4096); }
                }
            });

            t1.join().unwrap();
            t2.join().unwrap();
        });
    }

    /// Multiple alloc+free rounds from two threads — interleaved access.
    /// Migrated from command_arena::test_shared_page_pool_concurrent_alloc_free
    #[test]
    fn loom_shared_page_pool_interleaved() {
        use crate::memory::command_arena::SharedPagePool;

        bounded(2).check(|| {
            let pool = Arc::new(SharedPagePool::new(1024 * 1024));
            let p1 = pool.clone();
            let p2 = pool.clone();

            let t1 = loom::thread::spawn(move || {
                for _ in 0..2 {
                    let ptr = p1.alloc(4096).unwrap();
                    unsafe { p1.free(ptr, 4096); }
                }
            });

            let t2 = loom::thread::spawn(move || {
                for _ in 0..2 {
                    let ptr = p2.alloc(4096).unwrap();
                    unsafe { p2.free(ptr, 4096); }
                }
            });

            t1.join().unwrap();
            t2.join().unwrap();
        });
    }

    // =====================================================================
    // 4. CommandArena — threads sharing a pool
    //    Migrated from integration::test_command_arena_shared_pool_multithread
    // =====================================================================

    #[test]
    fn loom_command_arena_shared_pool() {
        use crate::memory::command_arena::{CommandArena, SharedPagePool};

        bounded(2).check(|| {
            let pool = Arc::new(SharedPagePool::new(1024 * 1024));
            let p1 = pool.clone();
            let p2 = pool.clone();

            let t1 = loom::thread::spawn(move || {
                let mut arena = CommandArena::new(4096, p1);
                arena.push(1u32).unwrap();
                arena.push(2u32).unwrap();
                let count = arena.iter_pages().count();
                assert!(count >= 1);
            });

            let t2 = loom::thread::spawn(move || {
                let mut arena = CommandArena::new(4096, p2);
                arena.push(3u32).unwrap();
                arena.push(4u32).unwrap();
                let count = arena.iter_pages().count();
                assert!(count >= 1);
            });

            t1.join().unwrap();
            t2.join().unwrap();
        });
    }

    // =====================================================================
    // 5. LargeAllocCache (Mutex-protected)
    //    Migrated from large_cache::test_large_cache_concurrent
    // =====================================================================

    #[test]
    fn loom_large_cache_concurrent() {
        use crate::memory::large_cache::LargeAllocCache;
        use crate::sync::Mutex;

        loom::model(|| {
            let limit = 10 * 1024 * 1024;
            let cache = Arc::new(Mutex::new(LargeAllocCache::new(limit)));
            let c1 = cache.clone();
            let c2 = cache.clone();

            let t1 = loom::thread::spawn(move || {
                let layout = std::alloc::Layout::from_size_align(4096, 1).unwrap();
                let (ptr, actual) = c1.lock().unwrap().alloc(layout).unwrap();
                c1.lock().unwrap().free(ptr, std::alloc::Layout::from_size_align(actual, 1).unwrap());
            });

            let t2 = loom::thread::spawn(move || {
                let layout = std::alloc::Layout::from_size_align(8192, 1).unwrap();
                let (ptr, actual) = c2.lock().unwrap().alloc(layout).unwrap();
                c2.lock().unwrap().free(ptr, std::alloc::Layout::from_size_align(actual, 1).unwrap());
            });

            t1.join().unwrap();
            t2.join().unwrap();

            cache.lock().unwrap().trim();
            assert_eq!((*cache.lock().unwrap()).total_cached_bytes(), 0);
        });
    }

    /// Interleaved alloc/free on LargeAllocCache — exercises cache reuse path.
    #[test]
    fn loom_large_cache_interleaved_alloc_free() {
        use crate::memory::large_cache::LargeAllocCache;
        use crate::sync::Mutex;

        bounded(2).check(|| {
            let cache = Arc::new(Mutex::new(LargeAllocCache::new(10 * 1024 * 1024)));
            let c1 = cache.clone();
            let c2 = cache.clone();

            let t1 = loom::thread::spawn(move || {
                let layout = std::alloc::Layout::from_size_align(4096, 1).unwrap();
                // alloc, free, alloc again (may hit cache reuse)
                let (ptr, actual) = c1.lock().unwrap().alloc(layout).unwrap();
                c1.lock().unwrap().free(ptr, std::alloc::Layout::from_size_align(actual, 1).unwrap());
                let (ptr2, actual2) = c1.lock().unwrap().alloc(layout).unwrap();
                c1.lock().unwrap().free(ptr2, std::alloc::Layout::from_size_align(actual2, 1).unwrap());
            });

            let t2 = loom::thread::spawn(move || {
                let layout = std::alloc::Layout::from_size_align(4096, 1).unwrap();
                let (ptr, actual) = c2.lock().unwrap().alloc(layout).unwrap();
                c2.lock().unwrap().free(ptr, std::alloc::Layout::from_size_align(actual, 1).unwrap());
            });

            t1.join().unwrap();
            t2.join().unwrap();
        });
    }

    // =====================================================================
    // 6. BinnedAllocator — concurrent alloc/free (instance-based)
    //    Migrated from binned::test_binned_allocator_thread_safety
    // =====================================================================

    /// Two threads each alloc+verify+free small number of items from
    /// same BinnedAllocator instance.  Exercises Pool Mutex contention,
    /// bit-tree CAS, and freelist.
    #[test]
    fn loom_binned_allocator_thread_safety() {
        use crate::memory::binned::BinnedAllocator;

        bounded(2).check(|| {
            let allocator = Arc::new(BinnedAllocator::new().unwrap());
            let a1 = allocator.clone();
            let a2 = allocator.clone();

            let t1 = loom::thread::spawn(move || {
                let p1 = a1.alloc_bytes(64).unwrap();
                unsafe { p1.as_ptr().write(0xAA); }
                let p2 = a1.alloc_bytes(256).unwrap();
                unsafe { p2.as_ptr().write(0xBB); }

                assert_eq!(unsafe { p1.as_ptr().read() }, 0xAA);
                assert_eq!(unsafe { p2.as_ptr().read() }, 0xBB);

                unsafe {
                    a1.free_bytes(p1, 64);
                    a1.free_bytes(p2, 256);
                }
            });

            let t2 = loom::thread::spawn(move || {
                let p1 = a2.alloc_bytes(64).unwrap();
                unsafe { p1.as_ptr().write(0xCC); }
                let p2 = a2.alloc_bytes(1024).unwrap();
                unsafe { p2.as_ptr().write(0xDD); }

                assert_eq!(unsafe { p1.as_ptr().read() }, 0xCC);
                assert_eq!(unsafe { p2.as_ptr().read() }, 0xDD);

                unsafe {
                    a2.free_bytes(p1, 64);
                    a2.free_bytes(p2, 1024);
                }
            });

            t1.join().unwrap();
            t2.join().unwrap();
        });
    }

    // =====================================================================
    // 7. BinnedAllocator — cross-thread alloc/free
    //    Migrated from binned::test_thread_cache_cross_thread
    // =====================================================================

    /// Thread A allocates, thread B frees; exercises cross-thread
    /// recycler path where freeing thread doesn't own the chunk.
    #[test]
    fn loom_binned_cross_thread_free() {
        use crate::memory::binned::BinnedAllocator;

        bounded(2).check(|| {
            let alloc = Arc::new(BinnedAllocator::new().unwrap());
            let a1 = alloc.clone();

            let t = loom::thread::spawn(move || {
                a1.alloc_bytes(64).unwrap().as_ptr() as usize
            });

            let ptr_addr = t.join().unwrap();
            let ptr = std::ptr::NonNull::new(ptr_addr as *mut u8).unwrap();
            unsafe { alloc.free_bytes(ptr, 64); }
        });
    }

    // =====================================================================
    // 8. BinnedAllocator — cross-thread alloc/free with ThreadCache
    //    Migrated from binned::test_producer_consumer_with_cache
    //
    //    The original test used std::sync::mpsc which loom cannot intercept.
    //    Restructured: producer allocates & stores pointer, consumer frees.
    //    The interesting interleaving is at the Pool Mutex + recycler level.
    // =====================================================================

    /// Producer thread allocates w/cache, consumer thread frees w/cache.
    /// Both threads use ThreadCache bound to same allocator.
    #[test]
    fn loom_binned_producer_consumer() {
        use crate::memory::binned::{BinnedAllocator, ThreadCache};
        use crate::sync::atomic::AtomicUsize;

        bounded(2).check(|| {
            let alloc = Arc::new(BinnedAllocator::new().unwrap());
            let ptr_slot = Arc::new(AtomicUsize::new(0));

            let alloc_p = alloc.clone();
            let slot = ptr_slot.clone();
            let producer = loom::thread::spawn(move || {
                let mut cache = ThreadCache::new();
                unsafe {
                    cache.bind(std::mem::transmute::<
                        &BinnedAllocator,
                        &'static BinnedAllocator,
                    >(&*alloc_p));
                }
                let layout = std::alloc::Layout::from_size_align(32, 1).unwrap();
                let ptr = alloc_p.alloc_with_cache(&mut cache, layout).unwrap();
                unsafe { *ptr.as_ptr() = 0x42; }
                slot.store(ptr.as_ptr() as usize, Ordering::Release);
            });

            producer.join().unwrap();

            let alloc_c = alloc.clone();
            let addr = ptr_slot.load(Ordering::Acquire);
            let consumer = loom::thread::spawn(move || {
                let mut cache = ThreadCache::new();
                unsafe {
                    cache.bind(std::mem::transmute::<
                        &BinnedAllocator,
                        &'static BinnedAllocator,
                    >(&*alloc_c));
                }
                let ptr = std::ptr::NonNull::new(addr as *mut u8).unwrap();
                let layout = std::alloc::Layout::from_size_align(32, 1).unwrap();
                alloc_c.free_with_cache(&mut cache, ptr, layout);
            });

            consumer.join().unwrap();
        });
    }

    // =====================================================================
    // 9. BinnedAllocator — mixed small + large sizes
    //    Migrated from binned::test_mixed_small_large_concurrent
    // =====================================================================

    /// Two threads doing mixed small + large (>65536) allocs concurrently.
    /// Large allocations go through LargeAllocCache path.
    #[test]
    fn loom_binned_mixed_small_large() {
        use crate::memory::binned::BinnedAllocator;

        bounded(2).check(|| {
            let alloc = Arc::new(BinnedAllocator::new().unwrap());
            let a1 = alloc.clone();
            let a2 = alloc.clone();

            let t1 = loom::thread::spawn(move || {
                // Small alloc
                let p_small = a1.alloc_bytes(64).unwrap();
                unsafe { p_small.as_ptr().write(0x11); }
                // Large alloc (> max bin size)
                let p_large = a1.alloc_bytes(100_000).unwrap();
                unsafe { p_large.as_ptr().write(0x22); }

                assert_eq!(unsafe { p_small.as_ptr().read() }, 0x11);
                assert_eq!(unsafe { p_large.as_ptr().read() }, 0x22);

                unsafe {
                    a1.free_bytes(p_small, 64);
                    a1.free_bytes(p_large, 100_000);
                }
            });

            let t2 = loom::thread::spawn(move || {
                let p_small = a2.alloc_bytes(256).unwrap();
                unsafe { p_small.as_ptr().write(0x33); }
                let p_large = a2.alloc_bytes(200_000).unwrap();
                unsafe { p_large.as_ptr().write(0x44); }

                assert_eq!(unsafe { p_small.as_ptr().read() }, 0x33);
                assert_eq!(unsafe { p_large.as_ptr().read() }, 0x44);

                unsafe {
                    a2.free_bytes(p_small, 256);
                    a2.free_bytes(p_large, 200_000);
                }
            });

            t1.join().unwrap();
            t2.join().unwrap();
        });
    }

    // =====================================================================
    // 10. BinnedAllocator — cache flush triggering recycler contention
    //     Migrated from binned::test_cache_recycler_contention
    // =====================================================================

    /// Two threads alloc/free >cache_limit items to trigger cache flush,
    /// exercising GlobalRecycler push/pop under real contention from
    /// allocator (not synthetic fake nodes).
    #[test]
    fn loom_binned_cache_recycler_contention() {
        use crate::memory::binned::{BinnedAllocator, ThreadCache};

        bounded(2).check(|| {
            let alloc = Arc::new(BinnedAllocator::new().unwrap());
            let a1 = alloc.clone();
            let a2 = alloc.clone();

            // Each thread: alloc a batch, then free them all, triggering
            // cache flush → recycler push.
            let t1 = loom::thread::spawn(move || {
                let mut cache = ThreadCache::new();
                unsafe {
                    cache.bind(std::mem::transmute::<
                        &BinnedAllocator,
                        &'static BinnedAllocator,
                    >(&*a1));
                }
                let layout = std::alloc::Layout::from_size_align(32, 1).unwrap();
                let p1 = a1.alloc_with_cache(&mut cache, layout).unwrap();
                let p2 = a1.alloc_with_cache(&mut cache, layout).unwrap();
                a1.free_with_cache(&mut cache, p1, layout);
                a1.free_with_cache(&mut cache, p2, layout);
            });

            let t2 = loom::thread::spawn(move || {
                let mut cache = ThreadCache::new();
                unsafe {
                    cache.bind(std::mem::transmute::<
                        &BinnedAllocator,
                        &'static BinnedAllocator,
                    >(&*a2));
                }
                let layout = std::alloc::Layout::from_size_align(32, 1).unwrap();
                let p1 = a2.alloc_with_cache(&mut cache, layout).unwrap();
                let p2 = a2.alloc_with_cache(&mut cache, layout).unwrap();
                a2.free_with_cache(&mut cache, p1, layout);
                a2.free_with_cache(&mut cache, p2, layout);
            });

            t1.join().unwrap();
            t2.join().unwrap();
        });
    }

    // =====================================================================
    // 11. FrameArena — thread-local isolation
    //     Migrated from integration::test_frame_arena_thread_local_isolation
    // =====================================================================

    /// Each thread creates its own FrameArena and allocates independently.
    /// No shared state verifies the VmOps mock works from multiple threads.
    #[test]
    fn loom_frame_arena_thread_isolation() {
        use crate::memory::frame_arena::FrameArena;

        loom::model(|| {
            let t1 = loom::thread::spawn(|| {
                let mut arena = FrameArena::new(4096).unwrap();
                let p = arena.alloc_val(42u32).unwrap();
                assert_eq!(*p, 42);
            });

            let t2 = loom::thread::spawn(|| {
                let mut arena = FrameArena::new(4096).unwrap();
                let p = arena.alloc_val(123u32).unwrap();
                assert_eq!(*p, 123);
            });

            t1.join().unwrap();
            t2.join().unwrap();
        });
    }

    // =====================================================================
    // 12. Trim epoch — concurrent signal + observe
    // =====================================================================

    #[test]
    fn loom_trim_epoch_visibility() {
        use crate::sync::atomic::AtomicU64;

        loom::model(|| {
            let epoch = Arc::new(AtomicU64::new(0));
            let e1 = epoch.clone();
            let e2 = epoch.clone();

            let writer = loom::thread::spawn(move || {
                e1.fetch_add(1, Ordering::AcqRel);
            });

            let reader = loom::thread::spawn(move || {
                e2.load(Ordering::Acquire)
            });

            writer.join().unwrap();
            let val = reader.join().unwrap();
            assert!(val <= 1);
        });
    }

    // =====================================================================
    // 13. Trim epoch — two writers, one reader
    // =====================================================================

    /// Three-way: two writers incrementing, one reader observing.
    /// Verifies no torn reads.
    #[test]
    fn loom_trim_epoch_two_writers() {
        use crate::sync::atomic::AtomicU64;

        loom::model(|| {
            let epoch = Arc::new(AtomicU64::new(0));
            let e1 = epoch.clone();
            let e2 = epoch.clone();
            let e3 = epoch.clone();

            let w1 = loom::thread::spawn(move || {
                e1.fetch_add(1, Ordering::AcqRel);
            });

            let w2 = loom::thread::spawn(move || {
                e2.fetch_add(1, Ordering::AcqRel);
            });

            let reader = loom::thread::spawn(move || {
                e3.load(Ordering::Acquire)
            });

            w1.join().unwrap();
            w2.join().unwrap();
            let val = reader.join().unwrap();
            assert!(val <= 2);
        });
    }

    // =====================================================================
    // 14. BinnedAllocator — alloc on two threads, free on main
    //     Exercises concurrent Pool lock contention + cross-thread free
    // =====================================================================

    #[test]
    fn loom_binned_concurrent_alloc_sequential_free() {
        use crate::memory::binned::BinnedAllocator;
        use crate::sync::atomic::AtomicUsize;

        bounded(2).check(|| {
            let alloc = Arc::new(BinnedAllocator::new().unwrap());
            let result_a = Arc::new(AtomicUsize::new(0));
            let result_b = Arc::new(AtomicUsize::new(0));

            let a1 = alloc.clone();
            let r1 = result_a.clone();
            let t1 = loom::thread::spawn(move || {
                let p = a1.alloc_bytes(128).unwrap();
                unsafe { p.as_ptr().write(0xEE); }
                r1.store(p.as_ptr() as usize, Ordering::Release);
            });

            let a2 = alloc.clone();
            let r2 = result_b.clone();
            let t2 = loom::thread::spawn(move || {
                let p = a2.alloc_bytes(128).unwrap();
                unsafe { p.as_ptr().write(0xFF); }
                r2.store(p.as_ptr() as usize, Ordering::Release);
            });

            t1.join().unwrap();
            t2.join().unwrap();

            let pa = result_a.load(Ordering::Acquire);
            let pb = result_b.load(Ordering::Acquire);
            assert_ne!(pa, pb, "two allocs must return distinct pointers");

            let ptr_a = std::ptr::NonNull::new(pa as *mut u8).unwrap();
            let ptr_b = std::ptr::NonNull::new(pb as *mut u8).unwrap();
            unsafe {
                assert_eq!(ptr_a.as_ptr().read(), 0xEE);
                assert_eq!(ptr_b.as_ptr().read(), 0xFF);
                alloc.free_bytes(ptr_a, 128);
                alloc.free_bytes(ptr_b, 128);
            }
        });
    }

    // =====================================================================
    // 15. SharedPagePool — alloc+free with interleaved ownership
    //     Thread A allocs, thread B frees A's pointer (and vice versa)
    // =====================================================================

    #[test]
    fn loom_shared_page_pool_cross_thread_free() {
        use crate::memory::command_arena::SharedPagePool;
        use crate::sync::atomic::AtomicUsize;

        loom::model(|| {
            let pool = Arc::new(SharedPagePool::new(1024 * 1024));
            let ptr_slot = Arc::new(AtomicUsize::new(0));

            let p1 = pool.clone();
            let slot1 = ptr_slot.clone();
            let t1 = loom::thread::spawn(move || {
                let ptr = p1.alloc(4096).unwrap();
                slot1.store(ptr.as_ptr() as usize, Ordering::Release);
            });

            t1.join().unwrap();

            let addr = ptr_slot.load(Ordering::Acquire);
            let ptr = std::ptr::NonNull::new(addr as *mut u8).unwrap();

            // Free from a different thread than the one that allocated
            let p2 = pool.clone();
            let t2 = loom::thread::spawn(move || {
                unsafe { p2.free(ptr, 4096); }
            });

            t2.join().unwrap();
        });
    }
}
