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
///   - CAS retry loops (recycler push/pop, NodePool free list) make the
///     unbounded interleaving space explode, so those tests use
///     `preemption_bound(2)`: every schedule with at most 2 forced
///     preemptions is explored exhaustively. This is standard bounded
///     model checking — small preemption bounds catch the overwhelming
///     majority of concurrency bugs (cf. CHESS, Musuvathi & Qadeer) at a
///     tractable cost. (An earlier revision justified the bound by an
///     "odd-generation spin" reservation protocol that no longer exists.)
#[cfg(loom)]
mod tests {
    use crate::sync::Arc;
    use crate::sync::atomic::Ordering;

    // =====================================================================
    // Helpers
    // =====================================================================

    /// Allocate a 64-byte aligned buffer and initialise the loom-tracked
    /// `AtomicPtr` at the recycler link offset (bytes 8..16).
    /// By putting a loom-tracked `AtomicPtr` in memory at exact offset
    /// `GlobalRecycler` expects allows loom to track "invisible" link ptrs.
    fn alloc_fake_node() -> (std::ptr::NonNull<u8>, std::alloc::Layout) {
        let layout = std::alloc::Layout::from_size_align(64, 64).unwrap();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        let nn = std::ptr::NonNull::new(ptr).expect("alloc failed");

        unsafe {
            nn.as_ptr().cast::<*mut u8>().write(std::ptr::null_mut());
        }

        // Place a loom-tracked AtomicPtr at offset 8 (recycler link field).
        unsafe {
            let link_ptr = nn
                .as_ptr()
                .add(std::mem::size_of::<usize>())
                .cast::<crate::sync::atomic::AtomicPtr<u8>>();
            std::ptr::write(
                link_ptr,
                crate::sync::atomic::AtomicPtr::new(std::ptr::null_mut()),
            );
        }

        (nn, layout)
    }

    fn bounded(preemption: usize) -> loom::model::Builder {
        let mut b = loom::model::Builder::new();
        b.preemption_bound = Some(preemption);
        // Whole-allocator models (e.g. concurrent trims) execute one branch
        // per pool lock/atomic across all NUM_SIZE_CLASSES pools, which
        // outgrew loom's default 1000-branch budget when the class table
        // widened. The preemption bound keeps exploration tractable; this
        // only admits longer straight-line executions.
        b.max_branches = 20_000;
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

            let overflow = recycler.push(0, node);
            assert!(overflow.is_none(), "should not overflow with limit 16");

            let popped = recycler.pop(0, &mut None);
            assert!(popped.is_some());
            assert_eq!(popped.unwrap().as_ptr(), node.as_ptr());
            assert!(recycler.pop(0, &mut None).is_none());

            unsafe {
                std::alloc::dealloc(node.as_ptr(), layout);
            }
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
                r1.push(0, node);
            });

            let t2 = loom::thread::spawn(move || {
                let node = std::ptr::NonNull::new(nb as *mut u8).unwrap();
                r2.push(0, node);
            });

            t1.join().unwrap();
            t2.join().unwrap();

            let mut popped = Vec::new();
            while let Some(p) = recycler.pop(0, &mut None) {
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

            recycler.push(0, node_a);

            let r_push = recycler.clone();
            let r_pop = recycler.clone();
            let nb = node_b.as_ptr() as usize;

            let t_push = loom::thread::spawn(move || {
                let node = std::ptr::NonNull::new(nb as *mut u8).unwrap();
                r_push.push(0, node);
            });

            let t_pop = loom::thread::spawn(move || r_pop.pop(0, &mut None));

            t_push.join().unwrap();
            let popped = t_pop.join().unwrap();

            let mut remaining = Vec::new();
            while let Some(p) = recycler.pop(0, &mut None) {
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
                    unsafe {
                        p1.free(ptr, 4096);
                    }
                }
            });

            let t2 = loom::thread::spawn(move || {
                if let Ok(ptr) = p2.alloc(4096) {
                    unsafe {
                        p2.free(ptr, 4096);
                    }
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
                    unsafe {
                        p1.free(ptr, 4096);
                    }
                }
            });

            let t2 = loom::thread::spawn(move || {
                for _ in 0..2 {
                    let ptr = p2.alloc(4096).unwrap();
                    unsafe {
                        p2.free(ptr, 4096);
                    }
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
                let count = arena.iter().count();
                assert!(count >= 1);
            });

            let t2 = loom::thread::spawn(move || {
                let mut arena = CommandArena::new(4096, p2);
                arena.push(3u32).unwrap();
                arena.push(4u32).unwrap();
                let count = arena.iter().count();
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

        bounded(2).check(|| {
            let limit = 10 * 1024 * 1024;
            let cache = Arc::new(LargeAllocCache::new(limit));
            let c1 = cache.clone();
            let c2 = cache.clone();

            let t1 = loom::thread::spawn(move || {
                let layout = std::alloc::Layout::from_size_align(4096, 1).unwrap();
                let (ptr, actual) = c1.alloc(layout).unwrap();
                c1.free(ptr, std::alloc::Layout::from_size_align(actual, 1).unwrap());
            });

            let t2 = loom::thread::spawn(move || {
                let layout = std::alloc::Layout::from_size_align(8192, 1).unwrap();
                let (ptr, actual) = c2.alloc(layout).unwrap();
                c2.free(ptr, std::alloc::Layout::from_size_align(actual, 1).unwrap());
            });

            t1.join().unwrap();
            t2.join().unwrap();

            cache.trim();
            assert_eq!(cache.total_cached_bytes(), 0);
        });
    }

    /// Interleaved alloc/free on LargeAllocCache — exercises cache reuse path.
    #[test]
    fn loom_large_cache_interleaved_alloc_free() {
        use crate::memory::large_cache::LargeAllocCache;

        bounded(2).check(|| {
            let cache = Arc::new(LargeAllocCache::new(10 * 1024 * 1024));
            let c1 = cache.clone();
            let c2 = cache.clone();

            let t1 = loom::thread::spawn(move || {
                let layout = std::alloc::Layout::from_size_align(4096, 1).unwrap();
                // alloc, free, alloc again (may hit cache reuse)
                let (ptr, actual) = c1.alloc(layout).unwrap();
                c1.free(ptr, std::alloc::Layout::from_size_align(actual, 1).unwrap());
                let (ptr2, actual2) = c1.alloc(layout).unwrap();
                c1.free(
                    ptr2,
                    std::alloc::Layout::from_size_align(actual2, 1).unwrap(),
                );
            });

            let t2 = loom::thread::spawn(move || {
                let layout = std::alloc::Layout::from_size_align(4096, 1).unwrap();
                let (ptr, actual) = c2.alloc(layout).unwrap();
                c2.free(ptr, std::alloc::Layout::from_size_align(actual, 1).unwrap());
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
                unsafe {
                    p1.as_ptr().write(0xAA);
                }
                let p2 = a1.alloc_bytes(256).unwrap();
                unsafe {
                    p2.as_ptr().write(0xBB);
                }

                assert_eq!(unsafe { p1.as_ptr().read() }, 0xAA);
                assert_eq!(unsafe { p2.as_ptr().read() }, 0xBB);

                unsafe {
                    a1.free_bytes(p1, 64);
                    a1.free_bytes(p2, 256);
                }
            });

            let t2 = loom::thread::spawn(move || {
                let p1 = a2.alloc_bytes(64).unwrap();
                unsafe {
                    p1.as_ptr().write(0xCC);
                }
                let p2 = a2.alloc_bytes(1024).unwrap();
                unsafe {
                    p2.as_ptr().write(0xDD);
                }

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
    // 6b. BinnedAllocator — trim racing alloc/free
    //     Exercises the three-phase trim protocol (begin_trim detaches
    //     blocks under the lock, decommit runs unlocked, finish_trim
    //     reintegrates): an allocation must never land in a detached block,
    //     and the allocator must stay consistent in every interleaving.
    // =====================================================================

    #[test]
    fn loom_binned_trim_decommit_races_alloc() {
        use crate::memory::binned::{BinnedAllocator, BinnedAllocatorConfig};

        bounded(2).check(|| {
            let config = BinnedAllocatorConfig {
                decommit_cooldown: 0, // make the empty block immediately eligible
                ..Default::default()
            };
            let allocator = Arc::new(BinnedAllocator::with_config(config).unwrap());

            // Leave one fully-empty block behind so trim has real work.
            let p = allocator.alloc_bytes(64).unwrap();
            // Safety: p was allocated with this size just above.
            unsafe { allocator.free_bytes(p, 64) };

            let a1 = allocator.clone();
            let a2 = allocator.clone();

            let t_alloc = loom::thread::spawn(move || {
                let p = a1.alloc_bytes(64).unwrap();
                unsafe {
                    p.as_ptr().write(0xEE);
                }
                assert_eq!(unsafe { p.as_ptr().read() }, 0xEE);
                // Safety: p was allocated with this size just above.
                unsafe { a1.free_bytes(p, 64) };
            });
            let t_trim = loom::thread::spawn(move || {
                a2.trim();
            });

            t_alloc.join().unwrap();
            t_trim.join().unwrap();

            // Allocator remains fully usable afterwards.
            let p = allocator.alloc_bytes(64).unwrap();
            // Safety: p was allocated with this size just above.
            unsafe { allocator.free_bytes(p, 64) };
        });
    }

    #[test]
    fn loom_binned_concurrent_trims() {
        use crate::memory::binned::{BinnedAllocator, BinnedAllocatorConfig};

        bounded(2).check(|| {
            let config = BinnedAllocatorConfig {
                decommit_cooldown: 0,
                ..Default::default()
            };
            let allocator = Arc::new(BinnedAllocator::with_config(config).unwrap());

            let p = allocator.alloc_bytes(64).unwrap();
            // Safety: p was allocated with this size just above.
            unsafe { allocator.free_bytes(p, 64) };

            let a1 = allocator.clone();
            let a2 = allocator.clone();
            let pool_idx = BinnedAllocator::size_class_min_align(64);

            // Two concurrent trims: the `decommitting` guard bit must keep
            // them from double-selecting (and double-uncommitting) a block.
            // Target the initialized class so model exploration is spent on
            // the trim protocol, rather than locking 95 empty class tables.
            let t1 = loom::thread::spawn(move || a1.trim_size_class_for_loom(pool_idx));
            let t2 = loom::thread::spawn(move || a2.trim_size_class_for_loom(pool_idx));
            t1.join().unwrap();
            t2.join().unwrap();

            let p = allocator.alloc_bytes(64).unwrap();
            // Safety: p was allocated with this size just above.
            unsafe { allocator.free_bytes(p, 64) };
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

            let t = loom::thread::spawn(move || a1.alloc_bytes(64).unwrap().as_ptr() as usize);

            let ptr_addr = t.join().unwrap();
            let ptr = std::ptr::NonNull::new(ptr_addr as *mut u8).unwrap();
            unsafe {
                alloc.free_bytes(ptr, 64);
            }
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
                unsafe {
                    *ptr.as_ptr() = 0x42;
                }
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

    /// Two threads doing mixed small + large (> MAX_SMALL_SIZE) allocs
    /// concurrently. Large allocations go through LargeAllocCache path.
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
                unsafe {
                    p_small.as_ptr().write(0x11);
                }
                // Large alloc (> max bin size)
                let p_large = a1.alloc_bytes(300_000).unwrap();
                unsafe {
                    p_large.as_ptr().write(0x22);
                }

                assert_eq!(unsafe { p_small.as_ptr().read() }, 0x11);
                assert_eq!(unsafe { p_large.as_ptr().read() }, 0x22);

                unsafe {
                    a1.free_bytes(p_small, 64);
                    a1.free_bytes(p_large, 300_000);
                }
            });

            let t2 = loom::thread::spawn(move || {
                let p_small = a2.alloc_bytes(256).unwrap();
                unsafe {
                    p_small.as_ptr().write(0x33);
                }
                let p_large = a2.alloc_bytes(400_000).unwrap();
                unsafe {
                    p_large.as_ptr().write(0x44);
                }

                assert_eq!(unsafe { p_small.as_ptr().read() }, 0x33);
                assert_eq!(unsafe { p_large.as_ptr().read() }, 0x44);

                unsafe {
                    a2.free_bytes(p_small, 256);
                    a2.free_bytes(p_large, 400_000);
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

            let reader = loom::thread::spawn(move || e2.load(Ordering::Acquire));

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

            let reader = loom::thread::spawn(move || e3.load(Ordering::Acquire));

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
                unsafe {
                    p.as_ptr().write(0xEE);
                }
                r1.store(p.as_ptr() as usize, Ordering::Release);
            });

            let a2 = alloc.clone();
            let r2 = result_b.clone();
            let t2 = loom::thread::spawn(move || {
                let p = a2.alloc_bytes(128).unwrap();
                unsafe {
                    p.as_ptr().write(0xFF);
                }
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
            let t2 = loom::thread::spawn(move || unsafe {
                p2.free(ptr, 4096);
            });

            t2.join().unwrap();
        });
    }

    // =====================================================================
    // 16. GlobalRecycler — drain_all racing with concurrent push
    //     Exercises trim-path drain while another thread is pushing bundles.
    // =====================================================================

    #[test]
    fn loom_recycler_drain_while_push() {
        use crate::memory::binned::GlobalRecycler;

        bounded(2).check(|| {
            let recycler = Arc::new(GlobalRecycler::new(16));
            let (node_a, layout_a) = alloc_fake_node();
            let (node_b, layout_b) = alloc_fake_node();

            // Pre-seed one bundle
            recycler.push(0, node_a);

            let r_push = recycler.clone();
            let r_drain = recycler.clone();
            let nb = node_b.as_ptr() as usize;

            let t_push = loom::thread::spawn(move || {
                let node = std::ptr::NonNull::new(nb as *mut u8).unwrap();
                r_push.push(0, node);
            });

            let t_drain = loom::thread::spawn(move || r_drain.drain_all(0));

            t_push.join().unwrap();
            let drained = t_drain.join().unwrap();

            // Collect whatever drain got
            let mut drained_addrs = Vec::new();
            if let Some(head) = drained {
                drained_addrs.push(head.as_ptr() as usize);
                let mut cur = head.as_ptr();
                loop {
                    let next = unsafe {
                        (*GlobalRecycler::recycler_link_atomic_ptr(cur)).load(Ordering::Relaxed)
                    };
                    if next.is_null() {
                        break;
                    }
                    drained_addrs.push(next as usize);
                    cur = next;
                }
            }

            // Collect whatever remains in the recycler
            let mut remaining = Vec::new();
            while let Some(p) = recycler.pop(0, &mut None) {
                remaining.push(p.as_ptr() as usize);
            }

            // Together must account for both nodes
            let total = drained_addrs.len() + remaining.len();
            assert_eq!(total, 2, "drain + remaining must account for both nodes");

            let na = node_a.as_ptr() as usize;
            let nb_val = node_b.as_ptr() as usize;
            let mut all: Vec<usize> = drained_addrs.into_iter().chain(remaining).collect();
            all.sort();
            all.dedup();
            assert!(all.contains(&na));
            assert!(all.contains(&nb_val));

            unsafe {
                std::alloc::dealloc(node_a.as_ptr(), layout_a);
                std::alloc::dealloc(node_b.as_ptr(), layout_b);
            }
        });
    }

    // =====================================================================
    // 17. GlobalRecycler — pop with remainder push-back racing with push
    //     When pop detaches a chain of >1 bundles, it pushes the remainder
    //     back via push_chain_back. Test this racing with another push.
    // =====================================================================

    #[test]
    fn loom_recycler_pop_remainder_races_push() {
        use crate::memory::binned::GlobalRecycler;

        bounded(2).check(|| {
            let recycler = Arc::new(GlobalRecycler::new(16));

            // Create 3 nodes: push A then B so the stack is B→A.
            // Pop will detach B→A, return B, push_chain_back(A).
            // Meanwhile another thread pushes C.
            let (node_a, layout_a) = alloc_fake_node();
            let (node_b, layout_b) = alloc_fake_node();
            let (node_c, layout_c) = alloc_fake_node();

            recycler.push(0, node_a);
            recycler.push(0, node_b);

            let r_pop = recycler.clone();
            let r_push = recycler.clone();
            let nc = node_c.as_ptr() as usize;

            let t_pop = loom::thread::spawn(move || r_pop.pop(0, &mut None));

            let t_push = loom::thread::spawn(move || {
                let node = std::ptr::NonNull::new(nc as *mut u8).unwrap();
                r_push.push(0, node);
            });

            let popped = t_pop.join().unwrap();
            t_push.join().unwrap();

            // Collect all from recycler
            let mut remaining = Vec::new();
            while let Some(p) = recycler.pop(0, &mut None) {
                remaining.push(p.as_ptr() as usize);
            }

            let na = node_a.as_ptr() as usize;
            let nb = node_b.as_ptr() as usize;
            let nc_val = node_c.as_ptr() as usize;

            let mut all: Vec<usize> = remaining;
            if let Some(p) = popped {
                all.push(p.as_ptr() as usize);
            }
            all.sort();
            all.dedup();
            assert_eq!(all.len(), 3, "all 3 nodes must be accounted for");
            assert!(all.contains(&na));
            assert!(all.contains(&nb));
            assert!(all.contains(&nc_val));

            unsafe {
                std::alloc::dealloc(node_a.as_ptr(), layout_a);
                std::alloc::dealloc(node_b.as_ptr(), layout_b);
                std::alloc::dealloc(node_c.as_ptr(), layout_c);
            }
        });
    }

    // =====================================================================
    // 18. GlobalRecycler — push overflow (recycler full)
    //     Verify that when max_bundles is reached, push returns the bundle.
    // =====================================================================

    #[test]
    fn loom_recycler_push_overflow_concurrent() {
        use crate::memory::binned::GlobalRecycler;

        bounded(2).check(|| {
            // max_bundles=1 → max_bundles_per_shard=1, only 1 bundle fits
            let recycler = Arc::new(GlobalRecycler::new(1));

            let (node_a, layout_a) = alloc_fake_node();
            let (node_b, layout_b) = alloc_fake_node();

            let r1 = recycler.clone();
            let r2 = recycler.clone();
            let na = node_a.as_ptr() as usize;
            let nb = node_b.as_ptr() as usize;

            let t1 = loom::thread::spawn(move || {
                let node = std::ptr::NonNull::new(na as *mut u8).unwrap();
                r1.push(0, node)
            });

            let t2 = loom::thread::spawn(move || {
                let node = std::ptr::NonNull::new(nb as *mut u8).unwrap();
                r2.push(0, node)
            });

            let overflow_a = t1.join().unwrap();
            let overflow_b = t2.join().unwrap();

            // Exactly one should overflow (only 1 slot available)
            let overflows = [overflow_a.is_some(), overflow_b.is_some()];
            let overflow_count = overflows.iter().filter(|&&x| x).count();
            assert_eq!(overflow_count, 1, "exactly one push must overflow");

            // Pop the one that succeeded
            let popped = recycler.pop(0, &mut None);
            assert!(popped.is_some());
            assert!(recycler.pop(0, &mut None).is_none());

            unsafe {
                std::alloc::dealloc(node_a.as_ptr(), layout_a);
                std::alloc::dealloc(node_b.as_ptr(), layout_b);
            }
        });
    }

    // =====================================================================
    // 18b. GlobalRecycler — occupancy cap holds across a push/pop race.
    //     Regression test: the old design kept the bundle count in a
    //     separate AtomicU32 that cas_detach reset to 0 *after* the detach
    //     CAS had already opened the slot to new pushes. A push landing in
    //     that window had its increment wiped, so the shard could exceed
    //     max_bundles. The count now rides in the 128-bit tagged word, so
    //     the cap must hold in every interleaving.
    // =====================================================================

    #[test]
    fn loom_recycler_cap_holds_across_push_pop_race() {
        use crate::memory::binned::GlobalRecycler;

        bounded(2).check(|| {
            // max_bundles=1 → exactly one bundle fits per shard.
            let recycler = Arc::new(GlobalRecycler::new(1));
            let (node_a, layout_a) = alloc_fake_node();
            let (node_b, layout_b) = alloc_fake_node();
            let (node_c, layout_c) = alloc_fake_node();

            recycler.push(0, node_a);

            let r_pop = recycler.clone();
            let r_push = recycler.clone();
            let nb = node_b.as_ptr() as usize;

            let t_pop = loom::thread::spawn(move || r_pop.pop(0, &mut None));
            let t_push = loom::thread::spawn(move || {
                let node = std::ptr::NonNull::new(nb as *mut u8).unwrap();
                r_push.push(0, node)
            });

            let popped = t_pop.join().unwrap();
            let push_overflow = t_push.join().unwrap();

            assert!(popped.is_some(), "shard held node_a before the race");

            if push_overflow.is_none() {
                // node_b entered the shard, so occupancy is exactly 1 and a
                // further push must overflow. (The racy counter could read 0
                // here and admit a second bundle past the cap.)
                let rejected = recycler.push(0, node_c);
                assert!(
                    rejected.is_some(),
                    "cap of 1 exceeded: shard admitted a second bundle"
                );
            }

            // Drain whatever remains before freeing the fake nodes.
            while recycler.pop(0, &mut None).is_some() {}

            unsafe {
                std::alloc::dealloc(node_a.as_ptr(), layout_a);
                std::alloc::dealloc(node_b.as_ptr(), layout_b);
                std::alloc::dealloc(node_c.as_ptr(), layout_c);
            }
        });
    }

    // =====================================================================
    // 18c. GlobalRecycler — cross-shard probe and multi-shard drain.
    //     RECYCLER_SHARD_COUNT is 2 under loom; one bundle sits on each
    //     shard. The popping thread's primary shard is 0, so recovering
    //     both bundles exercises pop's alternate-shard probe, and the
    //     racing drain_all exercises its multi-shard chain stitching.
    // =====================================================================

    #[test]
    fn loom_recycler_cross_shard_pop_and_drain() {
        use crate::memory::binned::{GlobalRecycler, loom_shard};

        bounded(2).check(|| {
            let recycler = Arc::new(GlobalRecycler::new(16));
            let (node_a, layout_a) = alloc_fake_node();
            let (node_b, layout_b) = alloc_fake_node();

            // One bundle on each shard.
            loom_shard::set_shard(0);
            recycler.push(0, node_a);
            loom_shard::set_shard(1);
            recycler.push(0, node_b);
            loom_shard::set_shard(0);

            let r_pop = recycler.clone();
            let r_drain = recycler.clone();

            // Spawned threads default to shard 0.
            let t_pop = loom::thread::spawn(move || r_pop.pop(0, &mut None));
            let t_drain = loom::thread::spawn(move || r_drain.drain_all(0));

            let popped = t_pop.join().unwrap();
            let drained = t_drain.join().unwrap();

            // Every node must be recovered exactly once.
            let mut all = Vec::new();
            if let Some(p) = popped {
                all.push(p.as_ptr() as usize);
            }
            if let Some(head) = drained {
                let mut cur = head.as_ptr();
                while !cur.is_null() {
                    all.push(cur as usize);
                    cur = unsafe {
                        (*GlobalRecycler::recycler_link_atomic_ptr(cur)).load(Ordering::Relaxed)
                    };
                }
            }
            while let Some(p) = recycler.pop(0, &mut None) {
                all.push(p.as_ptr() as usize);
            }
            all.sort_unstable();
            all.dedup();
            assert_eq!(
                all.len(),
                2,
                "both shards' bundles must be recovered exactly once"
            );

            unsafe {
                std::alloc::dealloc(node_a.as_ptr(), layout_a);
                std::alloc::dealloc(node_b.as_ptr(), layout_b);
            }
        });
    }

    // =====================================================================
    // 19. GlobalRecycler — drain_all racing with pop
    //     Trim-path drain vs alloc-path pop on the same shard.
    // =====================================================================

    #[test]
    fn loom_recycler_drain_races_pop() {
        use crate::memory::binned::GlobalRecycler;

        bounded(2).check(|| {
            let recycler = Arc::new(GlobalRecycler::new(16));
            let (node_a, layout_a) = alloc_fake_node();
            let (node_b, layout_b) = alloc_fake_node();

            recycler.push(0, node_a);
            recycler.push(0, node_b);

            let r_pop = recycler.clone();
            let r_drain = recycler.clone();

            let t_pop = loom::thread::spawn(move || r_pop.pop(0, &mut None));

            let t_drain = loom::thread::spawn(move || r_drain.drain_all(0));

            let popped = t_pop.join().unwrap();
            let drained = t_drain.join().unwrap();

            // Count everything recovered
            let mut all = Vec::new();
            if let Some(p) = popped {
                all.push(p.as_ptr() as usize);
            }
            if let Some(head) = drained {
                all.push(head.as_ptr() as usize);
                let mut cur = head.as_ptr();
                loop {
                    let next = unsafe {
                        (*GlobalRecycler::recycler_link_atomic_ptr(cur)).load(Ordering::Relaxed)
                    };
                    if next.is_null() {
                        break;
                    }
                    all.push(next as usize);
                    cur = next;
                }
            }
            // Also drain anything left in recycler
            while let Some(p) = recycler.pop(0, &mut None) {
                all.push(p.as_ptr() as usize);
            }

            all.sort();
            all.dedup();
            assert_eq!(all.len(), 2, "both nodes must be accounted for");

            unsafe {
                std::alloc::dealloc(node_a.as_ptr(), layout_a);
                std::alloc::dealloc(node_b.as_ptr(), layout_b);
            }
        });
    }

    // =====================================================================
    // 20. GlobalRecycler — 3-way: concurrent push + pop + drain
    //     Maximum contention: one thread pushes, one pops, one drains.
    // =====================================================================

    #[test]
    fn loom_recycler_push_pop_drain_three_way() {
        use crate::memory::binned::GlobalRecycler;

        bounded(2).check(|| {
            let recycler = Arc::new(GlobalRecycler::new(16));
            let (node_a, layout_a) = alloc_fake_node();
            let (node_b, layout_b) = alloc_fake_node();

            // Pre-seed node_a
            recycler.push(0, node_a);

            let r1 = recycler.clone();
            let r2 = recycler.clone();
            let r3 = recycler.clone();
            let nb = node_b.as_ptr() as usize;

            // Thread 1: push node_b
            let t1 = loom::thread::spawn(move || {
                let node = std::ptr::NonNull::new(nb as *mut u8).unwrap();
                r1.push(0, node)
            });

            // Thread 2: pop
            let t2 = loom::thread::spawn(move || r2.pop(0, &mut None));

            // Thread 3: drain_all
            let t3 = loom::thread::spawn(move || r3.drain_all(0));

            let push_overflow = t1.join().unwrap();
            let popped = t2.join().unwrap();
            let drained = t3.join().unwrap();

            // Collect everything
            let mut all = Vec::new();
            if let Some(ov) = push_overflow {
                all.push(ov.as_ptr() as usize);
            }
            if let Some(p) = popped {
                all.push(p.as_ptr() as usize);
            }
            if let Some(head) = drained {
                all.push(head.as_ptr() as usize);
                let mut cur = head.as_ptr();
                loop {
                    let next = unsafe {
                        (*GlobalRecycler::recycler_link_atomic_ptr(cur)).load(Ordering::Relaxed)
                    };
                    if next.is_null() {
                        break;
                    }
                    all.push(next as usize);
                    cur = next;
                }
            }
            while let Some(p) = recycler.pop(0, &mut None) {
                all.push(p.as_ptr() as usize);
            }

            let na = node_a.as_ptr() as usize;
            let nb_val = node_b.as_ptr() as usize;
            all.sort();
            all.dedup();
            assert_eq!(all.len(), 2, "both nodes must be accounted for");
            assert!(all.contains(&na));
            assert!(all.contains(&nb_val));

            unsafe {
                std::alloc::dealloc(node_a.as_ptr(), layout_a);
                std::alloc::dealloc(node_b.as_ptr(), layout_b);
            }
        });
    }

    // =====================================================================
    // 21. BinnedAllocator — alloc_with_cache + free_with_cache concurrent
    //     Exercises L0 micro-cache, recycler, and pool mutex all together.
    // =====================================================================

    #[test]
    fn loom_binned_alloc_free_with_cache_concurrent() {
        use crate::memory::binned::{BinnedAllocator, ThreadCache};

        bounded(2).check(|| {
            let alloc = Arc::new(BinnedAllocator::new().unwrap());
            let a1 = alloc.clone();
            let a2 = alloc.clone();

            let t1 = loom::thread::spawn(move || {
                let mut cache = ThreadCache::new();
                unsafe {
                    cache.bind(std::mem::transmute::<
                        &BinnedAllocator,
                        &'static BinnedAllocator,
                    >(&*a1));
                }
                let layout = std::alloc::Layout::from_size_align(16, 1).unwrap();
                // Alloc two, free two — exercises L0 push/pop
                let p1 = a1.alloc_with_cache(&mut cache, layout).unwrap();
                let p2 = a1.alloc_with_cache(&mut cache, layout).unwrap();
                a1.free_with_cache(&mut cache, p1, layout);
                a1.free_with_cache(&mut cache, p2, layout);
                // Re-alloc — should hit L0 cache
                let p3 = a1.alloc_with_cache(&mut cache, layout).unwrap();
                a1.free_with_cache(&mut cache, p3, layout);
            });

            let t2 = loom::thread::spawn(move || {
                let mut cache = ThreadCache::new();
                unsafe {
                    cache.bind(std::mem::transmute::<
                        &BinnedAllocator,
                        &'static BinnedAllocator,
                    >(&*a2));
                }
                let layout = std::alloc::Layout::from_size_align(16, 1).unwrap();
                let p1 = a2.alloc_with_cache(&mut cache, layout).unwrap();
                a2.free_with_cache(&mut cache, p1, layout);
            });

            t1.join().unwrap();
            t2.join().unwrap();
        });
    }

    // =====================================================================
    // 22. BinnedAllocator — trim racing with alloc_with_cache
    //     Trim drains the recycler while another thread is allocating.
    // =====================================================================

    #[test]
    fn loom_binned_trim_races_alloc() {
        use crate::memory::binned::{BinnedAllocator, ThreadCache};

        bounded(2).check(|| {
            let alloc = Arc::new(BinnedAllocator::new().unwrap());
            let a1 = alloc.clone();
            let a2 = alloc.clone();

            // Pre-allocate and free to populate recycler
            {
                let mut cache = ThreadCache::new();
                unsafe {
                    cache.bind(std::mem::transmute::<
                        &BinnedAllocator,
                        &'static BinnedAllocator,
                    >(&*alloc));
                }
                let layout = std::alloc::Layout::from_size_align(32, 1).unwrap();
                let p = alloc.alloc_with_cache(&mut cache, layout).unwrap();
                alloc.free_with_cache(&mut cache, p, layout);
            }

            // Thread 1: trim (drains recycler)
            let t1 = loom::thread::spawn(move || {
                a1.trim();
            });

            // Thread 2: alloc (may pop from recycler)
            let t2 = loom::thread::spawn(move || {
                let p = a2.alloc_bytes(32).unwrap();
                unsafe {
                    a2.free_bytes(p, 32);
                }
            });

            t1.join().unwrap();
            t2.join().unwrap();
        });
    }

    // =====================================================================
    // 23. Remote-mask channel — the dirty-flag handshake and ownership
    //     races (see remote_mask.rs module docs). Models operate on the
    //     protocol structures directly with a real 4 KiB buffer so link
    //     writes hit valid memory; loom-capped tables keep state small.
    // =====================================================================

    fn mask_fixture() -> (
        Arc<crate::memory::remote_mask::RemoteMaskTable>,
        usize,
        std::alloc::Layout,
    ) {
        use crate::memory::binned::compute_reciprocal;
        use crate::memory::remote_mask::{PoolRemote, RemoteMaskTable};

        let layout = std::alloc::Layout::from_size_align(4096, 4096).unwrap();
        // Safety: valid non-zero layout; freed by the caller at model end.
        let buf = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!buf.is_null());
        let base = buf as usize;
        let table = Arc::new(RemoteMaskTable::new(4096));
        let counter = crate::sync::atomic::AtomicU64::new(0);
        table.publish(
            PoolRemote::new(
                0,
                base,
                64,
                4096,
                64,
                4096,
                (compute_reciprocal(4096), compute_reciprocal(64)),
            ),
            &counter,
        );
        (table, base, layout)
    }

    fn link(nn: std::ptr::NonNull<u8>, prev: *mut u8) {
        unsafe { *nn.cast::<*mut u8>().as_ptr() = prev }
    }

    fn chain_len(head: std::ptr::NonNull<u8>) -> u32 {
        let mut n = 0;
        let mut cur = Some(head);
        while let Some(c) = cur {
            n += 1;
            cur = std::ptr::NonNull::new(unsafe { *c.cast::<*mut u8>().as_ptr() });
        }
        n
    }

    /// One publisher vs the owner reconciling: the bin must be delivered
    /// exactly once (during or after the race), never stranded or duped.
    #[test]
    fn loom_remote_mask_publish_vs_reconcile() {
        bounded(2).check(|| {
            let (table, base, layout) = mask_fixture();
            let (slot, generation) = table.claim_slot().unwrap();
            let pr_ptr = std::ptr::NonNull::new(base as *mut u8).unwrap();
            table
                .lookup(pr_ptr)
                .unwrap()
                .claim_block(0, slot, generation, link);

            let bin = std::ptr::NonNull::new((base + 64) as *mut u8).unwrap();
            // Loom-tracked payload proxy at bin offset 8 (offset 0 is the
            // chain link): a Relaxed store the publish's release edge must
            // make visible to the reconciling owner's Relaxed load — this
            // is the freed bin's contents crossing the handoff.
            let payload = bin.as_ptr() as usize + 8;
            unsafe {
                std::ptr::write(
                    payload as *mut crate::sync::atomic::AtomicU64,
                    crate::sync::atomic::AtomicU64::new(0),
                );
            }
            let t_free = {
                let table = table.clone();
                let addr = bin.as_ptr() as usize;
                loom::thread::spawn(move || {
                    let bin = std::ptr::NonNull::new(addr as *mut u8).unwrap();
                    unsafe { &*((addr + 8) as *const crate::sync::atomic::AtomicU64) }
                        .store(0xAB, Ordering::Relaxed);
                    table.publish_bin(bin, u16::MAX, 0)
                })
            };
            let check_payload = |h: std::ptr::NonNull<u8>| {
                let v = unsafe {
                    &*((h.as_ptr() as usize + 8) as *const crate::sync::atomic::AtomicU64)
                }
                .load(Ordering::Relaxed);
                assert_eq!(v, 0xAB, "bin contents must be visible to the receiver");
            };
            let t_owner = {
                let table = table.clone();
                loom::thread::spawn(move || {
                    let mut got = 0u32;
                    table.reconcile(slot, generation, link, |_, h, _, n| {
                        assert_eq!(chain_len(h), n);
                        check_payload(h);
                        got += n;
                    });
                    got
                })
            };

            let published = t_free.join().unwrap();
            let mut total = t_owner.join().unwrap();
            table.reconcile(slot, generation, link, |_, h, _, n| {
                check_payload(h);
                total += n;
            });
            assert!(published, "live owner: publish must take the mask path");
            assert_eq!(total, 1, "exactly-once delivery");
            unsafe { std::alloc::dealloc(base as *mut u8, layout) };
        });
    }

    /// Two publishers into the same block, racing the owner: both bins
    /// arrive, no duplicates, regardless of dirty-flag interleaving.
    #[test]
    fn loom_remote_mask_two_publishers() {
        bounded(2).check(|| {
            let (table, base, layout) = mask_fixture();
            let (slot, generation) = table.claim_slot().unwrap();
            let pr_ptr = std::ptr::NonNull::new(base as *mut u8).unwrap();
            table
                .lookup(pr_ptr)
                .unwrap()
                .claim_block(0, slot, generation, link);

            let spawn_pub = |addr: usize| {
                let table = table.clone();
                loom::thread::spawn(move || {
                    let bin = std::ptr::NonNull::new(addr as *mut u8).unwrap();
                    assert!(table.publish_bin(bin, u16::MAX, 0));
                })
            };
            let t1 = spawn_pub(base + 64);
            let t2 = spawn_pub(base + 128);
            t1.join().unwrap();
            t2.join().unwrap();

            let mut total = 0u32;
            table.reconcile(slot, generation, link, |_, h, _, n| {
                assert_eq!(chain_len(h), n);
                total += n;
            });
            assert_eq!(total, 2);
            unsafe { std::alloc::dealloc(base as *mut u8, layout) };
        });
    }

    /// A publish racing an ownership handoff (claim_block by a new owner):
    /// the bin is recovered exactly once across the claim harvest, either
    /// owner's reconcile, and the trim sweep backstop.
    #[test]
    fn loom_remote_mask_publish_vs_claim() {
        bounded(2).check(|| {
            let (table, base, layout) = mask_fixture();
            let (slot_a, gen_a) = table.claim_slot().unwrap();
            let (slot_b, gen_b) = table.claim_slot().unwrap();
            let pr_ptr = std::ptr::NonNull::new(base as *mut u8).unwrap();
            table
                .lookup(pr_ptr)
                .unwrap()
                .claim_block(0, slot_a, gen_a, link);

            let t_free = {
                let table = table.clone();
                let addr = base + 64;
                loom::thread::spawn(move || {
                    let bin = std::ptr::NonNull::new(addr as *mut u8).unwrap();
                    table.publish_bin(bin, u16::MAX, 0)
                })
            };
            let t_claim = {
                let table = table.clone();
                loom::thread::spawn(move || {
                    let pr_ptr = std::ptr::NonNull::new(base as *mut u8).unwrap();
                    table
                        .lookup(pr_ptr)
                        .unwrap()
                        .claim_block(0, slot_b, gen_b, link)
                        .map_or(0, |(_, _, n)| n)
                })
            };

            let published = t_free.join().unwrap();
            let mut total = t_claim.join().unwrap();
            table.reconcile(slot_a, gen_a, link, |_, _, _, n| total += n);
            table.reconcile(slot_b, gen_b, link, |_, _, _, n| total += n);
            table.sweep_pool(base, link, |_, _, n| total += n);
            assert_eq!(total, u32::from(published), "exactly-once across handoff");
            unsafe { std::alloc::dealloc(base as *mut u8, layout) };
        });
    }

    /// A publish racing the owner's slot release (thread death): a
    /// successful publish is recovered by the release drain or the trim
    /// sweep; a declined publish leaves nothing behind.
    #[test]
    fn loom_remote_mask_publish_vs_release() {
        bounded(2).check(|| {
            let (table, base, layout) = mask_fixture();
            let (slot, generation) = table.claim_slot().unwrap();
            let pr_ptr = std::ptr::NonNull::new(base as *mut u8).unwrap();
            table
                .lookup(pr_ptr)
                .unwrap()
                .claim_block(0, slot, generation, link);

            let t_free = {
                let table = table.clone();
                let addr = base + 64;
                loom::thread::spawn(move || {
                    let bin = std::ptr::NonNull::new(addr as *mut u8).unwrap();
                    table.publish_bin(bin, u16::MAX, 0)
                })
            };
            let t_release = {
                let table = table.clone();
                loom::thread::spawn(move || {
                    let mut drained = 0u32;
                    table.release_slot(slot, generation, link, |_, _, _, n| drained += n);
                    drained
                })
            };

            let published = t_free.join().unwrap();
            let mut total = t_release.join().unwrap();
            table.sweep_pool(base, link, |_, _, n| total += n);
            assert_eq!(total, u32::from(published), "no bin lost at owner death");
            unsafe { std::alloc::dealloc(base as *mut u8, layout) };
        });
    }
}
