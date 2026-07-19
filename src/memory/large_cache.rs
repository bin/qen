use super::stats;
use super::vm::{PlatformVmOps, VmError, VmOps};
use crate::sync::atomic::Ordering;
use crate::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, AtomicU128, AtomicUsize};
use std::alloc::Layout;
use std::collections::HashMap;
use std::ptr::NonNull;

// ---------------------------------------------------------------------------
// Lock-free segregated bucket cache for page-aligned large allocations
// ---------------------------------------------------------------------------

/// Default cache byte limit (64 MB); overridable via
/// `BinnedAllocatorConfig::large_cache_bytes`.
pub(crate) const LARGE_CACHE_DEFAULT_LIMIT: usize = 64 * 1024 * 1024;

/// Number of exact page-count buckets: bucket `i` holds blocks of exactly
/// `(i + 1) * page_size` bytes. Exact buckets make every reservation size
/// a pure function of the request (`size.next_multiple_of(page_size)`), so
/// alloc and free always agree on it — the old power-of-two canonical
/// scheme could hand a 128 KiB block to a 112 KiB request, whose free
/// (sized by layout) then partially unmapped the reservation and leaked
/// the tail. Blocks above `MAX_BUCKET_COUNT` pages (1 MiB at 4 KiB pages,
/// 4 MiB at 16 KiB) bypass the cache and release straight to the OS; on
/// Linux the huge-page path intercepts most of those first.
#[cfg(not(loom))]
const MAX_BUCKET_COUNT: usize = 256;
/// Loom builds shrink the bucket table: 256 modeled `AtomicU128`s exceed
/// loom's branch budget, and the Treiber push/pop algorithm the models
/// verify is independent of the table width (loom tests use blocks of a
/// few pages at most).
#[cfg(loom)]
const MAX_BUCKET_COUNT: usize = 8;

/// 128-bit tagged pointer for ABA-safe bucket Treiber stacks.
/// Same encoding as the recycler: `[127:64] = generation, [63:0] = pointer`.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
struct LargeTaggedPtr(u128);

impl LargeTaggedPtr {
    const NULL: Self = Self(0);

    #[inline]
    fn new(ptr: *mut LargeCacheNode, generation: u64) -> Self {
        // Provenance: packing a pointer into an integer word is inherent to
        // the DWCAS design; expose_provenance() makes the round-trip
        // explicit (see `ptr()`).
        Self(u128::from(generation) << 64 | (ptr.expose_provenance() as u128))
    }

    #[inline]
    fn ptr(self) -> *mut LargeCacheNode {
        std::ptr::with_exposed_provenance_mut(self.0 as usize)
    }

    #[inline]
    fn generation(self) -> u64 {
        (self.0 >> 64) as u64
    }

    #[inline]
    fn is_null(self) -> bool {
        self.ptr().is_null()
    }
}

/// A node in the bucket free-list. Allocated from a fixed-capacity node pool.
/// Stored as side metadata — never written into decommitted pages.
///
/// `next` is atomic because the Treiber-stack pop reads it speculatively
/// (before CAS) while another thread may be writing it after the node was
/// recycled. `ptr` and `size` are only accessed after a successful CAS
/// establishes exclusive ownership.
struct LargeCacheNode {
    /// The cached (decommitted) memory region.
    ptr: NonNull<u8>,
    /// Actual mapped size of the region.
    size: usize,
    /// Next node in the Treiber stack (bucket or free-list).
    /// Atomic to avoid data-race UB on speculative reads during pop.
    /// `Relaxed` suffices everywhere: the value is only trusted when the
    /// subsequent tagged CAS on the bucket slot succeeds (pop), or is
    /// published by the `Release` CAS that pushes the node (push).
    next: AtomicPtr<LargeCacheNode>,
}

/// Fixed-capacity pool of `LargeCacheNode` structs.
///
/// The backing storage is a raw allocation accessed only via raw pointers,
/// avoiding Stacked Borrows conflicts when nodes are concurrently accessed
/// from Treiber stacks.
///
/// The free list is a lock-free index-based Treiber stack whose head packs
/// `(generation, index)` into one `AtomicU64` (see [`TaggedIndex`]). The
/// generation increments on every successful CAS, which defeats ABA: a
/// stalled `alloc_node` whose observed head index was popped, reused, and
/// pushed back will see a different generation and retry, instead of
/// installing a stale successor and corrupting the list. The same algorithm
/// runs under loom — no mutex substitute — so the model checker exercises
/// exactly the code that ships.
struct NodePool {
    /// Raw pointer to the backing storage, obtained via `Box::into_raw`.
    storage_raw: *mut LargeCacheNode,
    /// Capacity (number of nodes).
    capacity: usize,
    /// Successor index per node (`u32::MAX` terminates). Only meaningful
    /// for nodes currently on the free list; written by the exclusive owner
    /// before the publishing CAS, read speculatively under CAS validation.
    free_next: Box<[AtomicU32]>,
    /// Tagged head of the free list.
    free_head: AtomicU64,
}

/// `(generation, index)` pair packed into a `u64` for the node free list:
/// bits `[63:32]` generation, bits `[31:0]` node index (`u32::MAX` = empty).
///
/// 32 generation bits wrap after 2^32 successful operations on the head; a
/// false CAS match additionally requires a thread stalled across an exact
/// multiple of 2^32 operations landing on the same index — accepted risk,
/// consistent with the crate's other tagged stacks.
#[derive(Clone, Copy)]
struct TaggedIndex(u64);

impl TaggedIndex {
    const EMPTY_INDEX: u32 = u32::MAX;

    #[inline]
    fn new(index: u32, generation: u32) -> Self {
        Self(u64::from(generation) << 32 | u64::from(index))
    }

    #[inline]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "intentional: extracts the 32-bit index lane from the packed word"
    )]
    fn index(self) -> u32 {
        self.0 as u32
    }

    #[inline]
    fn generation(self) -> u32 {
        // (Not flagged by clippy: `u64 >> 32` leaves at most 32 significant
        // bits, so the narrowing is provably lossless.)
        (self.0 >> 32) as u32
    }
}

// Safety: NodePool manages raw pointers to stable heap storage.
// Concurrent access is mediated by the tagged-head atomic free list.
unsafe impl Send for NodePool {}
// Safety: see Send above — all shared mutation goes through atomics.
unsafe impl Sync for NodePool {}

/// Minimum / maximum node pool sizes.
const NODE_POOL_MIN: usize = 64;
const NODE_POOL_MAX: usize = 4096;

impl NodePool {
    /// Create a pool sized to hold at most `cache_limit / page_size` nodes
    /// (one node per smallest cacheable block), clamped to [64, 4096].
    fn new(cache_limit: usize, page_size: usize) -> Self {
        let ideal = cache_limit / page_size;
        let cap = ideal.clamp(NODE_POOL_MIN, NODE_POOL_MAX);

        let storage: Vec<LargeCacheNode> = (0..cap)
            .map(|_| LargeCacheNode {
                ptr: NonNull::dangling(),
                size: 0,
                next: AtomicPtr::new(std::ptr::null_mut()),
            })
            .collect();

        // Convert to raw pointer via Box::into_raw to avoid Unique retag issues.
        let storage_boxed = storage.into_boxed_slice();
        let storage_raw = Box::into_raw(storage_boxed).cast::<LargeCacheNode>();

        let free_next: Vec<AtomicU32> = (0..cap)
            .map(|i| {
                AtomicU32::new(if i + 1 < cap {
                    // cap is clamped to NODE_POOL_MAX (4096) above.
                    u32::try_from(i + 1).expect("node pool capacity fits u32")
                } else {
                    TaggedIndex::EMPTY_INDEX
                })
            })
            .collect();

        Self {
            storage_raw,
            capacity: cap,
            free_next: free_next.into_boxed_slice(),
            free_head: AtomicU64::new(TaggedIndex::new(0, 0).0),
        }
    }

    /// Get a raw pointer to node at index `i`.
    #[inline]
    fn node_ptr(&self, i: usize) -> *mut LargeCacheNode {
        crate::qen_debug_assert!(i < self.capacity);
        // Safety: callers pass indices from the free list, which only
        // holds values < capacity — in-bounds of the storage allocation.
        unsafe { self.storage_raw.add(i) }
    }

    /// Compute the index of a node pointer.
    #[inline]
    fn node_index(&self, node: *mut LargeCacheNode) -> usize {
        let offset = (node as usize) - (self.storage_raw as usize);
        let idx = offset / std::mem::size_of::<LargeCacheNode>();
        crate::qen_debug_assert!(idx < self.capacity);
        idx
    }

    fn alloc_node(&self) -> Option<*mut LargeCacheNode> {
        loop {
            let head = TaggedIndex(self.free_head.load(Ordering::Acquire));
            if head.index() == TaggedIndex::EMPTY_INDEX {
                return None;
            }
            // Speculative read: only trusted if the tagged CAS below
            // succeeds, which proves the head (index AND generation) did not
            // change since the load — so `next` is still this node's
            // successor.
            let next = self.free_next[head.index() as usize].load(Ordering::Relaxed);
            let new = TaggedIndex::new(next, head.generation().wrapping_add(1));
            if self
                .free_head
                .compare_exchange_weak(head.0, new.0, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                return Some(self.node_ptr(head.index() as usize));
            }
        }
    }

    fn free_node(&self, node: *mut LargeCacheNode) {
        // node_index debug_asserts idx < capacity (<= 4096), so the
        // narrowing is lossless for any in-contract node pointer.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "bounded by pool capacity <= 4096; free path"
        )]
        let idx = self.node_index(node) as u32;
        loop {
            let head = TaggedIndex(self.free_head.load(Ordering::Relaxed));
            self.free_next[idx as usize].store(head.index(), Ordering::Relaxed);
            let new = TaggedIndex::new(idx, head.generation().wrapping_add(1));
            if self
                .free_head
                .compare_exchange_weak(head.0, new.0, Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                return;
            }
        }
    }
}

impl Drop for NodePool {
    fn drop(&mut self) {
        // Safety: storage_raw came from Box::into_raw with this exact
        // length; Drop has exclusive access, so no node is in use.
        unsafe {
            let slice = std::slice::from_raw_parts_mut(self.storage_raw, self.capacity);
            drop(Box::from_raw(slice));
        }
    }
}

/// Compute the bucket index for a given size: `ceil(size / page_size) - 1`.
/// Bucket i holds blocks of exactly `(i + 1) * page_size` bytes. Callers
/// bounds-check the result against the bucket count.
#[inline]
fn bucket_for_size(size: usize, page_size: usize) -> usize {
    size.div_ceil(page_size).saturating_sub(1)
}

/// Size of blocks in bucket `idx`.
#[inline]
#[cfg(all(test, not(loom)))]
fn bucket_size(idx: usize, page_size: usize) -> usize {
    (idx + 1) * page_size
}

/// Release a reservation, subtracting `reserved` from `TOTAL_RESERVED` and
/// `committed` from the committed gauges **only when the syscall succeeds**.
/// A failed `munmap`/`VirtualFree` leaves the mapping (and any committed
/// pages) in place, so decrementing the counters anyway would hide the
/// leaked memory from stats-based monitoring.
///
/// Returns whether the release succeeded.
///
/// # Safety
/// `ptr` must be the base of a live reservation of `reserved` bytes owned by
/// the caller, with no outstanding references into it.
unsafe fn release_and_untrack(ptr: NonNull<u8>, reserved: usize, committed: usize) -> bool {
    // Safety: contract forwarded from the caller.
    let ok = unsafe { PlatformVmOps::release(ptr, reserved) }.is_ok();
    if ok {
        stats::TOTAL_RESERVED.sub(reserved);
        if committed > 0 {
            stats::TOTAL_COMMITTED.sub(committed);
            stats::LARGE_ALLOC_CACHE_COMMITTED.sub(committed);
        }
    }
    ok
}

/// Lock-free segregated bucket cache for page-aligned large allocations.
///
/// Each bucket is an ABA-safe Treiber stack of `LargeCacheNode` pointers.
/// Alloc pops, free pushes. No mutex anywhere.
///
/// Buckets are EXACT page counts: bucket i caches blocks of exactly
/// `(i + 1) * page_size` bytes, so every request whose page-rounded size
/// recurs hits, every pop fits exactly (no undersized- or oversized-pop
/// waste), and the reservation size is always derivable from the request
/// alone (see `MAX_BUCKET_COUNT` for why that last property is
/// load-bearing).
pub(crate) struct LargeBucketCache {
    /// Per-bucket Treiber stacks. Bucket i caches blocks of exactly
    /// `(i + 1) * page_size` bytes.
    buckets: [AtomicU128; MAX_BUCKET_COUNT],
    /// Lock-free side-metadata node pool.
    node_pool: NodePool,
    /// Approximate total cached bytes. Best-effort, not linearizable.
    cached_bytes: AtomicUsize,
    cache_limit: usize,
    page_size: usize,
    num_buckets: usize,
}

// Safety: All fields are either atomic or use internal atomic synchronization.
unsafe impl Send for LargeBucketCache {}
// Safety: see Send above — all shared mutation goes through atomics.
unsafe impl Sync for LargeBucketCache {}

impl LargeBucketCache {
    pub fn new(cache_limit: usize) -> Self {
        let page_size = PlatformVmOps::page_size();
        Self {
            buckets: std::array::from_fn(|_| AtomicU128::new(LargeTaggedPtr::NULL.0)),
            node_pool: NodePool::new(cache_limit, page_size),
            cached_bytes: AtomicUsize::new(0),
            cache_limit,
            page_size,
            num_buckets: MAX_BUCKET_COUNT,
        }
    }

    /// The exact block size cached by a given bucket index.
    #[inline]
    fn bucket_block_size(&self, bucket: usize) -> usize {
        (bucket + 1) * self.page_size
    }

    /// Try to pop a cached block from the appropriate bucket.
    /// Returns `(ptr, block_size)` on hit, None on miss.
    ///
    /// Every cached block exactly matches its bucket's block size, which
    /// equals the page-rounded request — an exact fit by construction.
    pub fn try_alloc(&self, size: usize) -> Option<(NonNull<u8>, usize)> {
        let rounded = size.next_multiple_of(self.page_size);
        let bucket = bucket_for_size(rounded, self.page_size);
        if bucket >= self.num_buckets {
            return None;
        }

        let slot = &self.buckets[bucket];
        let bsize = self.bucket_block_size(bucket);
        crate::qen_debug_assert_eq!(bsize, rounded);

        loop {
            let old = LargeTaggedPtr(slot.load(Ordering::Acquire));
            if old.is_null() {
                return None;
            }

            let node = old.ptr();
            // Safety: node pool storage is never freed while the cache is
            // live, and `next` is atomic, so this speculative read is
            // race-free; the value is only meaningful if the CAS below
            // succeeds (the generation tag proves the slot — and thus this
            // node's position as head — was unchanged across the read).
            let next = unsafe { (*node).next.load(Ordering::Relaxed) };
            let next_tagged = LargeTaggedPtr::new(next, old.generation().wrapping_add(2));

            if slot
                .compare_exchange_weak(old.0, next_tagged.0, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                // Safety: the successful tagged CAS above transferred
                // exclusive ownership of `node` to this thread.
                let ptr = unsafe { (*node).ptr };
                // Safety: as above — node is exclusively owned here.
                crate::qen_debug_assert_eq!(unsafe { (*node).size }, bsize);

                // Return node to pool + update bytes (both lock-free)
                self.node_pool.free_node(node);
                self.cached_bytes.fetch_sub(bsize, Ordering::Relaxed);

                // Recommit the memory.
                // Safety: FFI commit of a region this cache owns.
                if unsafe { PlatformVmOps::commit(ptr, bsize) }.is_err() {
                    // Cached regions are decommitted, so only the
                    // reservation is still tracked.
                    // Safety: the popped region is exclusively ours.
                    unsafe { release_and_untrack(ptr, bsize, 0) };
                    return None;
                }

                #[cfg(any(debug_assertions, feature = "hardened"))]
                // Safety: region was just committed and is exclusively ours.
                unsafe {
                    std::ptr::write_bytes(ptr.as_ptr(), 0, bsize);
                }

                stats::TOTAL_COMMITTED.fetch_add(bsize, Ordering::Relaxed);
                stats::LARGE_ALLOC_CACHE_COMMITTED.fetch_add(bsize, Ordering::Relaxed);

                return Some((ptr, bsize));
            }
        }
    }

    /// Cache a freed block by pushing it onto the appropriate bucket.
    /// `size` must be the block's page-rounded reservation size; blocks
    /// larger than the biggest bucket are declined (returns `false`) so
    /// the caller releases to OS.
    pub fn try_cache(&self, ptr: NonNull<u8>, size: usize) -> bool {
        crate::qen_debug_assert!(size > 0 && size.is_multiple_of(self.page_size));
        let bucket = bucket_for_size(size, self.page_size);
        if bucket >= self.num_buckets {
            return false;
        }
        crate::qen_debug_assert_eq!(self.bucket_block_size(bucket), size);

        // Check cache limit (racy but conservative)
        if self
            .cached_bytes
            .load(Ordering::Relaxed)
            .saturating_add(size)
            > self.cache_limit
        {
            return false;
        }

        // Allocate a node (lock-free)
        let Some(node) = self.node_pool.alloc_node() else {
            stats::LARGE_CACHE_NODE_POOL_EXHAUSTED.fetch_add(1, Ordering::Relaxed);
            return false;
        };

        // Decommit the memory (release physical pages, keep VA).
        // Safety: FFI decommit of a region the caller relinquished to us.
        if unsafe { PlatformVmOps::decommit(ptr, size) }.is_err() {
            self.node_pool.free_node(node);
            return false;
        }

        self.cached_bytes.fetch_add(size, Ordering::Relaxed);
        stats::TOTAL_COMMITTED.sub(size);
        stats::LARGE_ALLOC_CACHE_COMMITTED.sub(size);

        // Initialize node.
        // Safety: node was just handed out by the pool — exclusively ours.
        unsafe {
            (*node).ptr = ptr;
            (*node).size = size;
        }

        // CAS-push onto bucket stack
        let slot = &self.buckets[bucket];
        loop {
            let old = LargeTaggedPtr(slot.load(Ordering::Acquire));
            // Safety: node is exclusively ours until the CAS below
            // publishes it; the store is published by that Release CAS.
            unsafe {
                (*node).next.store(old.ptr(), Ordering::Relaxed);
            }
            let new = LargeTaggedPtr::new(node, old.generation().wrapping_add(2));
            if slot
                .compare_exchange_weak(old.0, new.0, Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                return true;
            }
        }
    }

    /// Trim: drain all buckets, release cached memory to OS.
    pub fn trim(&self) {
        self.trim_to(0);
    }

    /// Trim until cached bytes <= target. Drains largest buckets first.
    /// Within a bucket, always drains the entire detached chain (no partial
    /// reattach — avoids the reattach race entirely).
    pub fn trim_to(&self, target: usize) {
        for bucket in (0..self.num_buckets).rev() {
            if self.cached_bytes.load(Ordering::Relaxed) <= target {
                return;
            }

            // CAS-detach: atomically replace head with null, deriving
            // generation from the value actually replaced (not a stale peek).
            let slot = &self.buckets[bucket];
            loop {
                let current = LargeTaggedPtr(slot.load(Ordering::Acquire));
                if current.is_null() {
                    break;
                }
                let null_tagged =
                    LargeTaggedPtr::new(std::ptr::null_mut(), current.generation().wrapping_add(2));
                if slot
                    .compare_exchange_weak(
                        current.0,
                        null_tagged.0,
                        Ordering::AcqRel,
                        Ordering::Relaxed,
                    )
                    .is_ok()
                {
                    // Walk the detached chain and release each entry.
                    // The AcqRel detach CAS gives us exclusive ownership of
                    // the whole chain, so Relaxed reads are sufficient.
                    let mut node = current.ptr();
                    while !node.is_null() {
                        // Safety: the AcqRel detach CAS gave us exclusive
                        // ownership of the whole chain (all three reads).
                        let next = unsafe { (*node).next.load(Ordering::Relaxed) };
                        // Safety: as above.
                        let ptr = unsafe { (*node).ptr };
                        // Safety: as above.
                        let size = unsafe { (*node).size };

                        // Cached regions are decommitted: only the
                        // reservation is still tracked.
                        // Safety: the detached chain is exclusively ours.
                        unsafe { release_and_untrack(ptr, size, 0) };

                        self.cached_bytes.fetch_sub(size, Ordering::Relaxed);
                        self.node_pool.free_node(node);

                        node = next;
                    }
                    break;
                }
            }
        }
    }

    #[cfg(test)]
    pub fn total_cached_bytes(&self) -> usize {
        self.cached_bytes.load(Ordering::Relaxed)
    }
}

/// Bookkeeping for an over-aligned allocation (align > `page_size`).
/// We over-reserve to guarantee the requested alignment, then return an
/// aligned sub-pointer. On free we need the original base and total size
/// to release the full reservation.
struct OverAlignedEntry {
    /// Original base returned by `PlatformVmOps::reserve`.
    original_base: NonNull<u8>,
    /// Total bytes reserved (size + align padding).
    total_reserved: usize,
    /// Committed size within the reservation (the user-visible part).
    committed_size: usize,
}

/// Cached probe results for runtime huge page availability.
///
/// Initialized from [`PlatformVmOps::supported_page_sizes()`]: any size
/// larger than the base page is a candidate. On the first large allocation
/// the allocator tries the *largest* candidate first; on failure the size
/// is marked unavailable (with a debug-mode log) and is never
/// retried. This gives zero-overhead detection:
///
/// - **Apple Silicon**: `supported_page_sizes()` → `[16384]` → no
///   candidates → nothing is ever attempted.
/// - **Linux, hugetlb pool empty**: first alloc tries `MAP_HUGETLB`,
///   `mmap` returns `ENOMEM`, size is marked unavailable, all subsequent
///   allocs go straight to regular pages.
/// - **Linux, hugetlb pool configured**: first alloc succeeds, every
///   subsequent large alloc silently uses huge pages.
struct HugePageProbe {
    /// `(page_size, should_try)` sorted descending (largest first).
    /// Only contains sizes from `supported_page_sizes()` that exceed the
    /// base page size. `should_try` starts `true` and is set to `false`
    /// after the first failed runtime allocation.
    sizes: Vec<(usize, bool)>,
}

impl HugePageProbe {
    /// Auto-detect from the platform's supported page sizes.
    fn new() -> Self {
        let base = PlatformVmOps::page_size();
        let mut sizes: Vec<(usize, bool)> = PlatformVmOps::supported_page_sizes()
            .into_iter()
            .filter(|&s| s > base)
            .map(|s| (s, true))
            .collect();
        sizes.sort_by_key(|b| std::cmp::Reverse(b.0)); // largest first
        Self { sizes }
    }

    /// Explicitly disabled (no huge pages attempted regardless of platform).
    fn disabled() -> Self {
        Self { sizes: Vec::new() }
    }

    /// Mark a page size as runtime-unavailable after a failed `alloc_huge`.
    fn mark_unavailable(&mut self, page_size: usize) {
        if let Some(entry) = self.sizes.iter_mut().find(|(s, _)| *s == page_size) {
            entry.1 = false;
        }
        #[cfg(any(debug_assertions, feature = "hardened"))]
        eprintln!(
            "[memory] {}MB huge pages probed unavailable at runtime; \
             falling back to smaller pages",
            page_size / (1024 * 1024),
        );
    }

    /// True when there are no candidate sizes left to try.
    fn exhausted(&self) -> bool {
        self.sizes.iter().all(|&(_, try_it)| !try_it)
    }
}

/// Cache for large allocations (larger than max small size class).
///
/// **Common path (page-aligned, standard alignment):** Lock-free segregated
/// bucket cache (`LargeBucketCache`). Each power-of-two size class has its
/// own ABA-safe Treiber stack. No mutex on alloc or free.
///
/// **Rare paths (over-aligned, huge pages):** Mutex-protected `HashMaps`,
/// same as before. Over-aligned allocations bypass the cache on free
/// (caching the padding is wasteful).
///
/// Huge pages are attempted automatically for allocations large enough,
/// cascading from the largest supported size down to regular pages.
/// See [`HugePageProbe`] for the detection/caching strategy.
pub(crate) struct LargeAllocCache {
    /// Lock-free bucket cache for standard page-aligned allocs.
    bucket_cache: LargeBucketCache,
    /// Mutex-protected state for rare paths.
    special: crate::sync::Mutex<LargeSpecialState>,
}

/// Mutex-protected state for rare large-alloc paths.
struct LargeSpecialState {
    /// Tracking for over-aligned allocations (align > `page_size`).
    over_aligned: HashMap<usize, OverAlignedEntry>,
    /// Runtime probe for huge page availability.
    huge_probe: HugePageProbe,
    /// Tracking for huge-page-backed allocations, keyed by address. The
    /// value keeps the original pointer (with its provenance, for the
    /// release in `Drop`) alongside the allocation size.
    huge_allocs: HashMap<usize, (NonNull<u8>, usize)>,
}

// Safety: LargeAllocCache is Send+Sync via internal synchronization.
unsafe impl Send for LargeAllocCache {}
// Safety: see Send above — lock-free buckets plus the `special` mutex.
unsafe impl Sync for LargeAllocCache {}

impl LargeAllocCache {
    /// Create a cache with automatic huge page detection.
    pub fn new(limit: usize) -> Self {
        Self::build(limit, HugePageProbe::new())
    }

    /// Create a cache with huge pages explicitly disabled.
    pub fn without_huge_pages(limit: usize) -> Self {
        Self::build(limit, HugePageProbe::disabled())
    }

    fn build(limit: usize, huge_probe: HugePageProbe) -> Self {
        Self {
            bucket_cache: LargeBucketCache::new(limit),
            special: crate::sync::Mutex::new(LargeSpecialState {
                over_aligned: HashMap::new(),
                huge_probe,
                huge_allocs: HashMap::new(),
            }),
        }
    }

    /// Returns true if the given alignment exceeds the system page size.
    #[inline]
    fn needs_over_align(align: usize) -> bool {
        align > PlatformVmOps::page_size()
    }

    pub fn alloc(&self, layout: Layout) -> Result<(NonNull<u8>, usize), VmError> {
        let align = layout.align();
        if Self::needs_over_align(align) {
            return self.alloc_over_aligned(layout);
        }
        self.alloc_page_aligned(layout.size())
    }

    /// Standard path: alignment <= `page_size`.
    fn alloc_page_aligned(&self, size: usize) -> Result<(NonNull<u8>, usize), VmError> {
        let page_size = PlatformVmOps::page_size();
        let size = size.next_multiple_of(page_size);

        // Fast path: try the lock-free bucket cache
        if let Some((ptr, actual_size)) = self.bucket_cache.try_alloc(size) {
            return Ok((ptr, actual_size));
        }

        // Try huge pages (rare path, needs mutex for probe state)
        {
            let mut special = self
                .special
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if !special.huge_probe.exhausted() {
                for i in 0..special.huge_probe.sizes.len() {
                    let (hp_size, should_try) = special.huge_probe.sizes[i];
                    if !should_try || hp_size > size {
                        continue;
                    }
                    let alloc_size = size.next_multiple_of(hp_size);
                    // Safety: FFI call to allocate huge pages.
                    match unsafe { PlatformVmOps::alloc_huge(alloc_size, hp_size) } {
                        Ok(ptr) => {
                            #[cfg(any(debug_assertions, feature = "hardened"))]
                            // Safety: freshly mapped, committed, exclusive.
                            unsafe {
                                std::ptr::write_bytes(ptr.as_ptr(), 0, alloc_size);
                            }

                            special
                                .huge_allocs
                                .insert(ptr.as_ptr() as usize, (ptr, alloc_size));
                            stats::TOTAL_RESERVED.fetch_add(alloc_size, Ordering::Relaxed);
                            stats::TOTAL_COMMITTED.fetch_add(alloc_size, Ordering::Relaxed);
                            stats::LARGE_ALLOC_CACHE_COMMITTED
                                .fetch_add(alloc_size, Ordering::Relaxed);
                            return Ok((ptr, alloc_size));
                        }
                        Err(_) => {
                            special.huge_probe.mark_unavailable(hp_size);
                        }
                    }
                }
            }
        }

        // Cold path: fresh reserve + commit from OS.
        // Safety: FFI reserve/commit/release of a region we own end-to-end;
        // the debug zeroing writes only to the just-committed range.
        unsafe {
            let ptr = PlatformVmOps::reserve(size)?;
            if let Err(e) = PlatformVmOps::commit(ptr, size) {
                // Stats were not incremented yet. If the cleanup release
                // also fails, the reservation persists — surface it in the
                // gauge rather than leaking silently.
                if PlatformVmOps::release(ptr, size).is_err() {
                    stats::TOTAL_RESERVED.fetch_add(size, Ordering::Relaxed);
                }
                return Err(e);
            }

            #[cfg(any(debug_assertions, feature = "hardened"))]
            std::ptr::write_bytes(ptr.as_ptr(), 0, size);

            stats::TOTAL_RESERVED.fetch_add(size, Ordering::Relaxed);
            stats::TOTAL_COMMITTED.fetch_add(size, Ordering::Relaxed);
            stats::LARGE_ALLOC_CACHE_COMMITTED.fetch_add(size, Ordering::Relaxed);

            Ok((ptr, size))
        }
    }

    /// Over-aligned path: alignment > `page_size`.
    fn alloc_over_aligned(&self, layout: Layout) -> Result<(NonNull<u8>, usize), VmError> {
        let page_size = PlatformVmOps::page_size();
        let size = layout.size().next_multiple_of(page_size);
        let align = layout.align();

        let total_reserve = (size + align - page_size).next_multiple_of(page_size);

        // Safety: FFI call to reserve memory.
        let base = unsafe { PlatformVmOps::reserve(total_reserve)? };

        let base_addr = base.as_ptr() as usize;
        let aligned_addr = (base_addr + align - 1) & !(align - 1);
        // Safety: the alignment offset stays within the padded reservation;
        // deriving from base (not the bare address) preserves provenance.
        let aligned_ptr =
            unsafe { NonNull::new_unchecked(base.as_ptr().add(aligned_addr - base_addr)) };

        crate::qen_debug_assert!(
            aligned_addr.is_multiple_of(page_size),
            "over-aligned pointer {aligned_addr:#x} is not page-aligned (page_size={page_size:#x})",
        );

        // Safety: FFI call to commit memory.
        if let Err(e) = unsafe { PlatformVmOps::commit(aligned_ptr, size) } {
            // Stats were not incremented yet. If the cleanup release also
            // fails, the reservation persists — surface it in the gauge
            // rather than leaking silently.
            // Safety: base is the reservation we just made.
            if unsafe { PlatformVmOps::release(base, total_reserve) }.is_err() {
                stats::TOTAL_RESERVED.fetch_add(total_reserve, Ordering::Relaxed);
            }
            return Err(e);
        }

        #[cfg(any(debug_assertions, feature = "hardened"))]
        // Safety: [aligned_ptr, aligned_ptr+size) was just committed and is
        // exclusively ours.
        unsafe {
            std::ptr::write_bytes(aligned_ptr.as_ptr(), 0, size);
        }

        stats::TOTAL_RESERVED.fetch_add(total_reserve, Ordering::Relaxed);
        stats::TOTAL_COMMITTED.fetch_add(size, Ordering::Relaxed);
        stats::LARGE_ALLOC_CACHE_COMMITTED.fetch_add(size, Ordering::Relaxed);

        {
            let mut special = self
                .special
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            special.over_aligned.insert(
                aligned_addr,
                OverAlignedEntry {
                    original_base: base,
                    total_reserved: total_reserve,
                    committed_size: size,
                },
            );
        }

        Ok((aligned_ptr, size))
    }

    pub fn free(&self, ptr: NonNull<u8>, layout: Layout) {
        let align = layout.align();
        if Self::needs_over_align(align) {
            self.free_over_aligned(ptr);
            return;
        }
        self.free_page_aligned(ptr, layout.size());
    }

    /// Standard free path: try to cache in lock-free buckets, else release to OS.
    fn free_page_aligned(&self, ptr: NonNull<u8>, requested_size: usize) {
        let page_size = PlatformVmOps::page_size();
        let size = requested_size.next_multiple_of(page_size);

        // Check if this is a huge-page-backed allocation
        {
            let mut special = self
                .special
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let addr = ptr.as_ptr() as usize;
            if let Some((hp_ptr, hp_size)) = special.huge_allocs.remove(&addr) {
                // Release directly (can't decommit/cache huge pages portably).
                // Safety: entry ownership was just removed from the map.
                if !unsafe { release_and_untrack(ptr, hp_size, hp_size) } {
                    // Keep the entry so Drop retries the release, and the
                    // counters keep reporting the still-mapped memory.
                    special.huge_allocs.insert(addr, (hp_ptr, hp_size));
                }
                return;
            }
        }

        // Try to cache in the lock-free bucket cache
        if self.bucket_cache.try_cache(ptr, size) {
            return;
        }

        // Cache full or no free nodes — release to OS
        // Safety: the caller relinquished ownership of the region.
        unsafe { release_and_untrack(ptr, size, size) };
    }

    /// Over-aligned free: look up the original reservation and release it.
    fn free_over_aligned(&self, ptr: NonNull<u8>) {
        let addr = ptr.as_ptr() as usize;
        let entry = {
            let mut special = self
                .special
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let Some(entry) = special.over_aligned.remove(&addr) else {
                #[cfg(any(debug_assertions, feature = "hardened"))]
                panic!("free_over_aligned: pointer {ptr:p} not found in over_aligned map");
                #[cfg(not(any(debug_assertions, feature = "hardened")))]
                return;
            };
            entry
        };

        // Safety: entry ownership was removed from the map above.
        let released = unsafe {
            release_and_untrack(
                entry.original_base,
                entry.total_reserved,
                entry.committed_size,
            )
        };
        if !released {
            // Keep the entry so Drop retries the release, and the counters
            // keep reporting the still-mapped memory.
            let mut special = self
                .special
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            special.over_aligned.insert(addr, entry);
        }
    }

    pub fn trim(&self) {
        self.bucket_cache.trim();
    }

    /// Partial trim to a target cached-byte budget. Not currently wired to
    /// a public API (`trim` drains fully); kept for callers that manage the
    /// cache incrementally, and exercised by (non-loom) tests.
    #[allow(dead_code)]
    pub fn trim_to(&self, target: usize) {
        self.bucket_cache.trim_to(target);
    }

    #[cfg(test)]
    pub fn total_cached_bytes(&self) -> usize {
        self.bucket_cache.total_cached_bytes()
    }
}

impl Drop for LargeAllocCache {
    fn drop(&mut self) {
        self.trim();
        // Release any remaining over-aligned allocations
        let mut special = self
            .special
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for (_, entry) in special.over_aligned.drain() {
            // Safety: teardown — no outstanding references remain.
            // On failure the counters keep reporting the leaked mapping.
            unsafe {
                release_and_untrack(
                    entry.original_base,
                    entry.total_reserved,
                    entry.committed_size,
                );
            }
        }
        // Release any remaining huge page allocations
        for (_, (ptr, hp_size)) in special.huge_allocs.drain() {
            // Safety: teardown — no outstanding references remain.
            unsafe {
                release_and_untrack(ptr, hp_size, hp_size);
            }
        }
    }
}

/// Loom model checks for the `NodePool` free list. These run the exact
/// production algorithm (there is deliberately no loom substitute), so the
/// ABA-defeating tagged head is what gets verified.
#[cfg(all(test, loom))]
mod loom_node_pool_tests {
    use super::*;
    use crate::sync::Arc;

    #[test]
    fn loom_node_pool_alloc_free_race() {
        let mut builder = loom::model::Builder::new();
        builder.preemption_bound = Some(3);
        builder.check(|| {
            // Minimum capacity (64); two threads race alloc→free.
            let pool = Arc::new(NodePool::new(64 * 4096, 4096));
            let p1 = pool.clone();
            let p2 = pool.clone();

            let t1 = loom::thread::spawn(move || {
                if let Some(n) = p1.alloc_node() {
                    p1.free_node(n);
                }
            });
            let t2 = loom::thread::spawn(move || {
                if let Some(n) = p2.alloc_node() {
                    p2.free_node(n);
                }
            });
            t1.join().unwrap();
            t2.join().unwrap();

            // Every node must still be reachable exactly once: a lost or
            // double-handed node here is the classic ABA corruption.
            let mut nodes = Vec::new();
            while let Some(n) = pool.alloc_node() {
                nodes.push(n as usize);
            }
            assert_eq!(nodes.len(), pool.capacity, "free list lost nodes");
            nodes.sort_unstable();
            nodes.dedup();
            assert_eq!(nodes.len(), pool.capacity, "free list double-handed a node");
            for n in nodes {
                pool.free_node(n as *mut LargeCacheNode);
            }
        });
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use crate::sync::Arc;

    /// Convenience: build a Layout with align=1 for the given size.
    fn lay(size: usize) -> Layout {
        Layout::from_size_align(size, 1).unwrap()
    }

    #[test]
    fn test_large_cache_reuse_exact() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let page_size = PlatformVmOps::page_size();
        let cache = LargeAllocCache::new(page_size * 10);

        let (ptr1, s1) = cache.alloc(lay(page_size)).unwrap();
        let addr1 = ptr1.as_ptr() as usize;

        cache.free(ptr1, lay(s1));

        let (ptr2, s2) = cache.alloc(lay(page_size)).unwrap();
        let addr2 = ptr2.as_ptr() as usize;

        assert_eq!(addr1, addr2);
        assert_eq!(s1, s2);

        // Return the live allocation so Drop releases it (miri leak check).
        cache.free(ptr2, lay(s2));
    }

    #[test]
    fn test_large_cache_same_bucket_reuse() {
        // With segregated buckets, a 2-page alloc goes to bucket 1.
        // Freeing and re-allocating the same size should reuse it.
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let page_size = PlatformVmOps::page_size();
        let cache = LargeAllocCache::new(page_size * 10);

        let (ptr1, s1) = cache.alloc(lay(page_size * 2)).unwrap();
        let addr1 = ptr1.as_ptr() as usize;

        cache.free(ptr1, lay(s1));

        // Same size re-alloc should hit cache
        let (ptr2, s2) = cache.alloc(lay(page_size * 2)).unwrap();
        assert_eq!(ptr2.as_ptr() as usize, addr1);
        assert_eq!(s2, page_size * 2);

        // Return the live allocation so Drop releases it (miri leak check).
        cache.free(ptr2, lay(s2));
    }

    #[test]
    fn test_large_cache_no_cross_bucket_reuse() {
        // A 2-page block is cached in bucket 1; a 1-page alloc uses bucket 0.
        // No cross-bucket fallback — 1-page alloc should get a fresh block.
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let page_size = PlatformVmOps::page_size();
        let cache = LargeAllocCache::new(page_size * 10);

        let (ptr1, s1) = cache.alloc(lay(page_size * 2)).unwrap();
        let addr1 = ptr1.as_ptr() as usize;

        cache.free(ptr1, lay(s1));

        // 1-page alloc should NOT get the 2-page block
        let (ptr2, s2) = cache.alloc(lay(page_size)).unwrap();
        assert_ne!(ptr2.as_ptr() as usize, addr1);

        // Return the live allocation so Drop releases it (miri leak check).
        cache.free(ptr2, lay(s2));
    }

    #[test]
    fn test_large_cache_limit() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let page_size = PlatformVmOps::page_size();
        let cache = LargeAllocCache::new(page_size);

        let (p1, s1) = cache.alloc(lay(page_size)).unwrap();
        let (p2, s2) = cache.alloc(lay(page_size)).unwrap();

        cache.free(p1, lay(s1));
        assert_eq!(cache.total_cached_bytes(), page_size);

        // Second free should exceed limit and go to OS
        cache.free(p2, lay(s2));
        assert_eq!(cache.total_cached_bytes(), page_size);
    }

    #[test]
    fn test_large_cache_drop_releases() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let page_size = PlatformVmOps::page_size();
        let cache = LargeAllocCache::new(page_size * 10);
        let (p1, s1) = cache.alloc(lay(page_size)).unwrap();
        cache.free(p1, lay(s1));
        assert_eq!(cache.total_cached_bytes(), page_size);
    }

    #[test]
    fn test_large_cache_multiple_sizes() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let page_size = PlatformVmOps::page_size();
        let cache = LargeAllocCache::new(page_size * 10);

        let (p1, s1) = cache.alloc(lay(page_size)).unwrap();
        let (p2, s2) = cache.alloc(lay(page_size * 2)).unwrap();

        cache.free(p1, lay(s1));
        cache.free(p2, lay(s2));

        // Re-alloc same sizes; each should come from its own bucket
        let (r1, sr1) = cache.alloc(lay(page_size * 2)).unwrap();
        let (r2, sr2) = cache.alloc(lay(page_size)).unwrap();

        assert_eq!(r1.as_ptr() as usize, p2.as_ptr() as usize);
        assert_eq!(r2.as_ptr() as usize, p1.as_ptr() as usize);
        assert_eq!(sr1, page_size * 2);
        assert_eq!(sr2, page_size);

        // Return the live allocations so Drop releases them (miri leak check).
        cache.free(r1, lay(sr1));
        cache.free(r2, lay(sr2));
    }

    #[test]
    fn test_large_cache_decommit_on_reuse() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let page_size = PlatformVmOps::page_size();
        let cache = LargeAllocCache::new(page_size * 10);

        let (p1, s1) = cache.alloc(lay(page_size)).unwrap();
        cache.free(p1, lay(s1));

        let (p2, s2) = cache.alloc(lay(page_size)).unwrap();
        // Safety: p2 is a live allocation from this cache.
        unsafe {
            *p2.as_ptr() = 0xFF;
        }

        // Return the live allocation so Drop releases it (miri leak check).
        cache.free(p2, lay(s2));
    }

    /// Exact-fit invariant — regression test for the canonical-bucket
    /// leak: the old power-of-two scheme could serve a cached 4-page block
    /// to a 3-page request; the caller then freed by layout (3 pages),
    /// which partially unmapped the 4-page reservation and leaked the
    /// mapped, committed tail page. Exact buckets make
    /// `actual_size == page-rounded request` unconditionally, and
    /// non-power-of-two page counts (declined outright by the old scheme)
    /// now cache and hit.
    #[test]
    fn test_large_cache_exact_fit_no_oversized_serve() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let page_size = PlatformVmOps::page_size();
        let cache = LargeAllocCache::new(page_size * 100);

        // Cache a 4-page block.
        let (p4, s4) = cache.alloc(lay(page_size * 4)).unwrap();
        assert_eq!(s4, page_size * 4);
        cache.free(p4, lay(s4));
        assert_eq!(cache.total_cached_bytes(), page_size * 4);

        // A 3-page request must NOT be served by the 4-page block, and its
        // actual size must equal its own rounded size.
        let (p3, s3) = cache.alloc(lay(page_size * 3)).unwrap();
        assert_eq!(
            s3,
            page_size * 3,
            "actual size must match the rounded request"
        );
        assert_ne!(p3.as_ptr(), p4.as_ptr());
        assert_eq!(
            cache.total_cached_bytes(),
            page_size * 4,
            "the 4-page block stays cached for 4-page requests"
        );

        // Same-size churn on a non-power-of-two page count now hits.
        cache.free(p3, lay(s3));
        let (p3b, s3b) = cache.alloc(lay(page_size * 3)).unwrap();
        assert_eq!(
            p3b.as_ptr(),
            p3.as_ptr(),
            "3-page block must be reused exactly"
        );
        assert_eq!(s3b, page_size * 3);

        // Drain the cache through live handles so Drop releases everything
        // (miri leak check).
        cache.free(p3b, lay(s3b));
        let (p4b, s4b) = cache.alloc(lay(page_size * 4)).unwrap();
        assert_eq!(p4b.as_ptr(), p4.as_ptr());
        cache.free(p4b, lay(s4b));
    }

    #[test]
    fn test_large_cache_page_alignment() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let cache = LargeAllocCache::new(1024 * 1024);
        let (ptr, s) = cache.alloc(lay(123)).unwrap();
        assert_eq!(ptr.as_ptr() as usize % PlatformVmOps::page_size(), 0);

        // Return the live allocation so Drop releases it (miri leak check).
        cache.free(ptr, lay(s));
    }

    #[test]
    fn test_large_cache_trim_empty() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let cache = LargeAllocCache::new(4096 * 10);
        cache.trim();
        assert_eq!(cache.total_cached_bytes(), 0);
    }

    #[test]
    fn test_large_cache_alloc_after_trim() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let page_size = PlatformVmOps::page_size();
        let cache = LargeAllocCache::new(page_size * 10);

        let (p1, s1) = cache.alloc(lay(page_size)).unwrap();
        cache.free(p1, lay(s1));
        assert_eq!(cache.total_cached_bytes(), page_size);

        cache.trim();
        assert_eq!(cache.total_cached_bytes(), 0);

        let (p2, s2) = cache.alloc(lay(page_size)).unwrap();
        // Safety: p2 is a live allocation from this cache.
        unsafe {
            *p2.as_ptr() = 0xAA;
        }

        // Return the live allocation so Drop releases it (miri leak check).
        cache.free(p2, lay(s2));
    }

    #[test]
    fn test_release_failure_keeps_reserved_stat() {
        // Exclusive: failure injection is global. Assertions are scoped to
        // LARGE_ALLOC_CACHE_COMMITTED — a subsystem gauge no other
        // subsystem touches. (Cross-subsystem gauges like TOTAL_RESERVED
        // also move when other tests' thread-local caches drop at thread
        // death, which happens outside TEST_MUTEX.)
        let _guard = crate::memory::TEST_MUTEX.write().unwrap();
        let cache = LargeAllocCache::new(0); // cache disabled: free() releases to OS
        let (ptr, size) = cache.alloc(lay(256 * 1024)).unwrap();

        let committed_before = stats::LARGE_ALLOC_CACHE_COMMITTED.get();
        assert!(committed_before >= size);

        crate::memory::vm::failure_injection::fail_next_releases(1);
        cache.free(ptr, Layout::from_size_align(size, 1).unwrap());
        crate::memory::vm::failure_injection::reset();

        // The failed munmap leaves the mapping (and its pages) in place:
        // the gauge must keep reporting it, not pretend it was returned.
        assert_eq!(
            stats::LARGE_ALLOC_CACHE_COMMITTED.get(),
            committed_before,
            "failed release must not decrement the committed gauge"
        );

        // Clean up the real mapping (the cache no longer tracks it) and
        // balance the gauges we verified above.
        // Safety: the region is live (its release was the injected failure)
        // and no longer referenced by the cache.
        unsafe {
            PlatformVmOps::release(ptr, size).expect("cleanup release");
        }
        stats::TOTAL_RESERVED.sub(size);
        stats::TOTAL_COMMITTED.sub(size);
        stats::LARGE_ALLOC_CACHE_COMMITTED.sub(size);
    }

    #[test]
    fn test_alloc_commit_failure_returns_error() {
        // Exclusive: failure injection is global. See the note above about
        // asserting only the subsystem gauge.
        let _guard = crate::memory::TEST_MUTEX.write().unwrap();
        let cache = LargeAllocCache::new(0);
        let committed_before = stats::LARGE_ALLOC_CACHE_COMMITTED.get();

        crate::memory::vm::failure_injection::fail_next_commits(1);
        let result = cache.alloc(lay(256 * 1024));
        crate::memory::vm::failure_injection::reset();

        assert!(
            matches!(result, Err(VmError::CommitFailed(_))),
            "commit failure must propagate as CommitFailed"
        );
        // The failed alloc never incremented the gauge (its cleanup path
        // released the fresh reservation before stats were added).
        assert_eq!(stats::LARGE_ALLOC_CACHE_COMMITTED.get(), committed_before);
    }

    #[test]
    fn test_large_cache_stats_lifecycle() {
        let _guard = crate::memory::TEST_MUTEX.write().unwrap();
        let page_size = PlatformVmOps::page_size();
        let cache = LargeAllocCache::new(page_size * 10);

        let initial_cp = stats::LARGE_ALLOC_CACHE_COMMITTED.load(Ordering::Relaxed);
        let (p1, s1) = cache.alloc(lay(page_size)).unwrap();
        let inter_cp = stats::LARGE_ALLOC_CACHE_COMMITTED.load(Ordering::Relaxed);

        assert!(inter_cp >= initial_cp + page_size);

        cache.free(p1, lay(s1));
        assert_eq!(cache.total_cached_bytes(), page_size);
    }

    #[test]
    fn test_large_cache_partial_trim() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let page_size = PlatformVmOps::page_size();
        let cache = LargeAllocCache::new(page_size * 10);

        let (p1, s1) = cache.alloc(lay(page_size)).unwrap();
        let (p2, s2) = cache.alloc(lay(page_size)).unwrap();
        let (p3, s3) = cache.alloc(lay(page_size)).unwrap();
        let (p4, s4) = cache.alloc(lay(page_size)).unwrap();

        cache.free(p1, lay(s1));
        cache.free(p2, lay(s2));
        cache.free(p3, lay(s3));
        cache.free(p4, lay(s4));

        assert_eq!(cache.total_cached_bytes(), page_size * 4);

        cache.trim_to(page_size * 2);
        assert!(cache.total_cached_bytes() <= page_size * 2);

        cache.trim();
        assert_eq!(cache.total_cached_bytes(), 0);
    }

    #[test]
    fn test_large_cache_concurrent() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();

        let limit = 10 * 1024 * 1024;
        // LargeAllocCache is now internally thread-safe (Send + Sync), no Mutex needed
        let cache = Arc::new(LargeAllocCache::new(limit));
        let mut handles = vec![];

        for t in 0u8..4 {
            let c = cache.clone();
            handles.push(crate::sync::thread::spawn(move || {
                let page_size = PlatformVmOps::page_size();
                let mut ptrs = vec![];

                for i in 0..50 {
                    let size = page_size * (1 + (i % 4));
                    let (ptr, actual_size) = c.alloc(lay(size)).unwrap();
                    // Safety: ptr is a live allocation from this cache.
                    unsafe {
                        ptr.as_ptr().write(t);
                        assert_eq!(ptr.as_ptr().read(), t);
                    }
                    ptrs.push((ptr, actual_size));

                    if i % 3 == 0 && !ptrs.is_empty() {
                        let (p, s) = ptrs.pop().unwrap();
                        c.free(p, lay(s));
                    }
                }

                for (p, s) in ptrs {
                    c.free(p, lay(s));
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        cache.trim();
        assert_eq!(cache.total_cached_bytes(), 0);
    }

    #[test]
    fn test_large_cache_no_leak_same_bucket() {
        let _guard = crate::memory::TEST_MUTEX.write().unwrap();
        let page_size = PlatformVmOps::page_size();
        let cache = LargeAllocCache::new(page_size * 10);

        let initial_reserved = stats::TOTAL_RESERVED.load(Ordering::Relaxed);

        let (p1, s1) = cache.alloc(lay(page_size * 2)).unwrap();
        assert_eq!(s1, page_size * 2);

        cache.free(p1, lay(s1));
        assert_eq!(cache.total_cached_bytes(), page_size * 2);

        // Re-alloc same size — should reuse
        let (p2, s2) = cache.alloc(lay(page_size * 2)).unwrap();
        assert_eq!(p1, p2);
        assert_eq!(s2, page_size * 2);

        cache.free(p2, lay(s2));
        cache.trim();
        assert_eq!(cache.total_cached_bytes(), 0);

        let final_reserved = stats::TOTAL_RESERVED.load(Ordering::Relaxed);
        assert_eq!(
            final_reserved, initial_reserved,
            "Address space leak detected in TOTAL_RESERVED!"
        );
    }

    #[test]
    fn test_large_cache_trim_largest_first() {
        // Trim drains largest buckets first
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let page_size = PlatformVmOps::page_size();
        let cache = LargeAllocCache::new(page_size * 20);

        // Alloc: 1-page (bucket 0), 2-page (bucket 1), 4-page (bucket 2)
        let (p1, s1) = cache.alloc(lay(page_size)).unwrap();
        let (p2, s2) = cache.alloc(lay(page_size * 2)).unwrap();
        let (p4, s4) = cache.alloc(lay(page_size * 4)).unwrap();

        cache.free(p1, lay(s1));
        cache.free(p2, lay(s2));
        cache.free(p4, lay(s4));
        assert_eq!(cache.total_cached_bytes(), page_size * 7);

        // Trim to 3 pages — should evict 4-page block first
        cache.trim_to(page_size * 3);
        assert!(cache.total_cached_bytes() <= page_size * 3);

        // 1-page and 2-page should still be cached
        let (r1, rs1) = cache.alloc(lay(page_size)).unwrap();
        assert_eq!(r1.as_ptr() as usize, p1.as_ptr() as usize);
        assert_eq!(rs1, page_size);

        let (r2, rs2) = cache.alloc(lay(page_size * 2)).unwrap();
        assert_eq!(r2.as_ptr() as usize, p2.as_ptr() as usize);
        assert_eq!(rs2, page_size * 2);

        assert_eq!(cache.total_cached_bytes(), 0);

        // Return the live allocations so the cache's Drop releases them
        // (miri runs with the leak checker enabled).
        cache.free(r1, lay(rs1));
        cache.free(r2, lay(rs2));
    }

    #[test]
    fn test_large_cache_trim_to_partial_within_bucket() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let page_size = PlatformVmOps::page_size();
        let cache = LargeAllocCache::new(page_size * 20);

        let mut ptrs = Vec::new();
        for _ in 0..5 {
            let (p, s) = cache.alloc(lay(page_size * 2)).unwrap();
            ptrs.push((p, s));
        }
        for (p, s) in ptrs {
            cache.free(p, lay(s));
        }
        assert_eq!(cache.total_cached_bytes(), page_size * 10);

        cache.trim_to(page_size * 4);
        assert!(cache.total_cached_bytes() <= page_size * 4);

        cache.trim();
        assert_eq!(cache.total_cached_bytes(), 0);
    }

    #[test]
    fn test_large_cache_over_aligned() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let page_size = PlatformVmOps::page_size();
        let big_align = page_size * 4;
        let cache = LargeAllocCache::new(page_size * 10);

        let layout = Layout::from_size_align(page_size, big_align).unwrap();
        let (ptr, actual_size) = cache.alloc(layout).unwrap();

        assert_eq!(
            ptr.as_ptr() as usize % big_align,
            0,
            "Over-aligned allocation should be aligned to {big_align}"
        );
        // Safety: ptr is a live allocation from this cache.
        unsafe {
            *ptr.as_ptr() = 0xCC;
        }
        assert_eq!(actual_size, page_size);

        cache.free(ptr, layout);
        // Over-aligned allocs bypass cache
        assert_eq!(cache.total_cached_bytes(), 0);
    }

    #[test]
    fn test_bucket_for_size_basic() {
        let page_size = PlatformVmOps::page_size();
        // Exact page-count buckets: n pages → bucket n-1.
        assert_eq!(bucket_for_size(page_size, page_size), 0);
        assert_eq!(bucket_for_size(page_size * 2, page_size), 1);
        assert_eq!(bucket_for_size(page_size * 3, page_size), 2);
        assert_eq!(bucket_for_size(page_size * 4, page_size), 3);
        assert_eq!(bucket_for_size(page_size * 5, page_size), 4);
        // Partial pages round up to the next full page's bucket.
        assert_eq!(bucket_for_size(page_size + 1, page_size), 1);
        assert_eq!(bucket_for_size(page_size * 2 - 1, page_size), 1);
    }

    #[test]
    fn test_bucket_size_roundtrip() {
        let page_size = PlatformVmOps::page_size();
        for i in 0..MAX_BUCKET_COUNT {
            let bs = bucket_size(i, page_size);
            assert_eq!(bucket_for_size(bs, page_size), i);
        }
    }
}
