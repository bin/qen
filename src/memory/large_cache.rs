use super::stats;
use super::vm::{PlatformVmOps, VmError, VmOps};
use std::alloc::Layout;
use std::collections::{BTreeMap, HashMap};
use std::ptr::NonNull;
use crate::sync::atomic::Ordering;

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
        #[cfg(debug_assertions)]
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
/// Caches freed pages to avoid kernel calls.
///
/// For standard alignments (<= `page_size`), allocations are page-aligned by the
/// OS and cached in a `BTreeMap` for best-fit reuse. For over-aligned allocations
/// (align > `page_size`), we over-reserve, align internally, and track the
/// original mapping in a separate `HashMap`. Over-aligned allocations bypass the
/// cache on free (they are rare and caching the padding is wasteful).
///
/// Huge pages are attempted automatically for allocations large enough,
/// cascading from the largest supported size down to regular pages.
/// See [`HugePageProbe`] for the detection/caching strategy.
pub(crate) struct LargeAllocCache {
    /// Cached freed pages, organized by size.
    /// Values are vectors of pointers.
    cached: BTreeMap<usize, Vec<NonNull<u8>>>,
    cached_bytes: usize,
    cache_limit: usize,
    /// Tracking for over-aligned allocations (align > `page_size`).
    /// Keyed by the aligned pointer address returned to the caller.
    over_aligned: HashMap<usize, OverAlignedEntry>,
    /// Tracking for live page-aligned allocations.
    /// Keyed by pointer address, value is actual mapped size.
    /// Needed because best-fit reuse can return a larger block than requested.
    live_page_allocs: HashMap<usize, usize>,
    /// Runtime probe for huge page availability. Auto-populated from
    /// `supported_page_sizes()`; sizes are struck off after the first
    /// failed allocation attempt.
    huge_probe: HugePageProbe,
    /// Tracking for huge-page-backed allocations.
    /// Keyed by pointer address, value is allocated size.
    /// These bypass the decommit/recommit cache on free because huge page
    /// decommit semantics differ across platforms (no partial decommit on
    /// Windows, no standard recommit on macOS superpages).
    huge_allocs: HashMap<usize, usize>,
}

// Safety: LargeAllocCache owns memory.
unsafe impl Send for LargeAllocCache {}
// Not Sync, intended for use behind a lock or per-thread (but usually global with lock).

impl LargeAllocCache {
    /// Create a cache with automatic huge page detection.
    ///
    /// On the first large allocation (>= 2MB), the cache probes the system
    /// for huge page support by trying `alloc_huge` with the largest
    /// supported size. Failed sizes are cached and never retried.
    /// On platforms with no huge page support (Apple Silicon, etc.) the
    /// probe set is empty and no attempt is ever made.
    pub fn new(limit: usize) -> Self {
        Self::build(limit, HugePageProbe::new())
    }

    /// Create a cache with huge pages explicitly disabled.
    pub fn without_huge_pages(limit: usize) -> Self {
        Self::build(limit, HugePageProbe::disabled())
    }

    fn build(limit: usize, huge_probe: HugePageProbe) -> Self {
        Self {
            cached: BTreeMap::new(),
            cached_bytes: 0,
            cache_limit: limit,
            over_aligned: HashMap::new(),
            live_page_allocs: HashMap::new(),
            huge_probe,
            huge_allocs: HashMap::new(),
        }
    }

    /// Returns true if the given alignment exceeds the system page size and
    /// therefore requires the over-aligned allocation path.
    #[inline]
    fn needs_over_align(align: usize) -> bool {
        align > PlatformVmOps::page_size()
    }

    pub fn alloc(&mut self, layout: Layout) -> Result<(NonNull<u8>, usize), VmError> {
        let align = layout.align();
        if Self::needs_over_align(align) {
            return self.alloc_over_aligned(layout);
        }
        self.alloc_page_aligned(layout.size())
    }

    /// Standard path: alignment <= `page_size`. mmap returns page-aligned memory.
    fn alloc_page_aligned(&mut self, size: usize) -> Result<(NonNull<u8>, usize), VmError> {
        let size = size.next_multiple_of(PlatformVmOps::page_size());

        // Best-fit match check: find the smallest cached entry >= size
        let found_size = self.cached.range_mut(size..).next().map(|(&s, _)| s);

        if let Some(s) = found_size {
            let Some(list) = self.cached.get_mut(&s) else {
                debug_assert!(
                    false,
                    "cached.range_mut returned key {s}, but get_mut failed"
                );
                // Safety: BTreeMap range search and get_mut logic ensure this is reachable only if key exists.
                unsafe { std::hint::unreachable_unchecked() }
            };
            if let Some(ptr) = list.pop() {
                self.cached_bytes -= s;
                if list.is_empty() {
                    self.cached.remove(&s);
                }
                // We decommitted on free, so we must commit here.
                // We commit the whole block we are giving back.
                // Safety: FFI call to commit memory.
                if let Err(e) = unsafe { PlatformVmOps::commit(ptr, s) } {
                    // Recommit failed. We cannot easily return it to the cache (it's uncommitted/invalid state).
                    // We must release the VA reservation to avoid leaking it.
                    // Safety: FFI call to release memory.
                    unsafe {
                        drop(PlatformVmOps::release(ptr, s));
                        stats::sub_saturating(&stats::TOTAL_RESERVED, s);
                    }
                    return Err(e);
                }

                // Debug mode: zero recommitted memory to guarantee deterministic
                // behavior (macOS MADV_FREE may retain stale data).
                #[cfg(debug_assertions)]
                // Safety: ptr is valid and s is correct.
                unsafe {
                    std::ptr::write_bytes(ptr.as_ptr(), 0, s);
                }

                stats::TOTAL_COMMITTED.fetch_add(s, Ordering::Relaxed);
                stats::LARGE_ALLOC_CACHE_COMMITTED.fetch_add(s, Ordering::Relaxed);
                self.live_page_allocs.insert(ptr.as_ptr() as usize, s);

                return Ok((ptr, s));
            }
        }

        // Huge pages: cascade from largest supported size downward.
        // The probe is pre-populated from supported_page_sizes() and
        // sizes are struck off after the first runtime failure. Once all
        // sizes are exhausted (or the probe was empty to begin with,
        // e.g. Apple Silicon) we skip straight to regular pages.
        if !self.huge_probe.exhausted() {
            for i in 0..self.huge_probe.sizes.len() {
                let (hp_size, should_try) = self.huge_probe.sizes[i];
                if !should_try || hp_size > size {
                    continue;
                }
                let alloc_size = size.next_multiple_of(hp_size);
                // Safety: FFI call to allocate huge pages.
                match unsafe { PlatformVmOps::alloc_huge(alloc_size, hp_size) } {
                    Ok(ptr) => {
                        #[cfg(debug_assertions)]
                        // Safety: ptr is valid and alloc_size is correct.
                        unsafe {
                            std::ptr::write_bytes(ptr.as_ptr(), 0, alloc_size);
                        }

                        self.huge_allocs.insert(ptr.as_ptr() as usize, alloc_size);
                        stats::TOTAL_RESERVED.fetch_add(alloc_size, Ordering::Relaxed);
                        stats::TOTAL_COMMITTED.fetch_add(alloc_size, Ordering::Relaxed);
                        stats::LARGE_ALLOC_CACHE_COMMITTED.fetch_add(alloc_size, Ordering::Relaxed);
                        return Ok((ptr, alloc_size));
                    }
                    Err(_) => {
                        self.huge_probe.mark_unavailable(hp_size);
                    }
                }
            }
        }

        // No cached page, alloc new (regular pages)
        // Safety: FFI calls to reserve and commit memory.
        unsafe {
            let ptr = PlatformVmOps::reserve(size)?;
            if let Err(e) = PlatformVmOps::commit(ptr, size) {
                drop(PlatformVmOps::release(ptr, size));
                return Err(e);
            }

            #[cfg(debug_assertions)]
            std::ptr::write_bytes(ptr.as_ptr(), 0, size);

            stats::TOTAL_RESERVED.fetch_add(size, Ordering::Relaxed);
            stats::TOTAL_COMMITTED.fetch_add(size, Ordering::Relaxed);
            stats::LARGE_ALLOC_CACHE_COMMITTED.fetch_add(size, Ordering::Relaxed);
            self.live_page_allocs.insert(ptr.as_ptr() as usize, size);

            Ok((ptr, size))
        }
    }

    /// Over-aligned path: alignment > `page_size`. We over-reserve by `align`
    /// extra bytes, find the aligned offset, commit only the needed portion,
    /// and record the original base for release on free.
    fn alloc_over_aligned(&mut self, layout: Layout) -> Result<(NonNull<u8>, usize), VmError> {
        let page_size = PlatformVmOps::page_size();
        let size = layout.size().next_multiple_of(page_size);
        let align = layout.align();

        // Over-reserve: we need `size` usable bytes at an `align`-aligned address.
        // In the worst case the base is (align - page_size) bytes before the next
        // aligned boundary, so reserving size + align - page_size guarantees we
        // can find an aligned start. We round up to page_size for the OS.
        let total_reserve = (size + align - page_size).next_multiple_of(page_size);

        // Safety: FFI call to reserve memory.
        let base = unsafe { PlatformVmOps::reserve(total_reserve)? };

        let base_addr = base.as_ptr() as usize;
        // Round up to the next align boundary
        let aligned_addr = (base_addr + align - 1) & !(align - 1);
        // Safety: addr is valid.
        let aligned_ptr = unsafe { NonNull::new_unchecked(aligned_addr as *mut u8) };

        // Invariant: since align > page_size and both are powers of 2,
        // aligned_addr is guaranteed to also be page-aligned.
        debug_assert!(
            aligned_addr.is_multiple_of(page_size),
            "over-aligned pointer {aligned_addr:#x} is not page-aligned (page_size={page_size:#x})",
        );

        // Commit only the user-visible region (page-aligned size at the aligned start)
        // Safety: FFI call to commit memory.
        if let Err(e) = unsafe { PlatformVmOps::commit(aligned_ptr, size) } {
            // Safety: FFI call to release memory.
            unsafe {
                drop(PlatformVmOps::release(base, total_reserve));
            }
            return Err(e);
        }

        #[cfg(debug_assertions)]
        // Safety: ptr is valid and size is correct.
        unsafe {
            std::ptr::write_bytes(aligned_ptr.as_ptr(), 0, size);
        }

        stats::TOTAL_RESERVED.fetch_add(total_reserve, Ordering::Relaxed);
        stats::TOTAL_COMMITTED.fetch_add(size, Ordering::Relaxed);
        stats::LARGE_ALLOC_CACHE_COMMITTED.fetch_add(size, Ordering::Relaxed);

        self.over_aligned.insert(
            aligned_addr,
            OverAlignedEntry {
                original_base: base,
                total_reserved: total_reserve,
                committed_size: size,
            },
        );

        Ok((aligned_ptr, size))
    }

    pub fn free(&mut self, ptr: NonNull<u8>, layout: Layout) {
        let align = layout.align();
        if Self::needs_over_align(align) {
            self.free_over_aligned(ptr);
            return;
        }
        self.free_page_aligned(ptr, layout.size());
    }

    /// Standard free path: page-aligned allocations may be cached.
    fn free_page_aligned(&mut self, ptr: NonNull<u8>, _requested_size: usize) {
        // Huge-page-backed: release directly (can't decommit/cache portably).
        let addr = ptr.as_ptr() as usize;
        if let Some(hp_size) = self.huge_allocs.remove(&addr) {
            // Safety: FFI call to release memory.
            unsafe {
                drop(PlatformVmOps::release(ptr, hp_size));
            }
            stats::sub_saturating(&stats::TOTAL_COMMITTED, hp_size);
            stats::sub_saturating(&stats::LARGE_ALLOC_CACHE_COMMITTED, hp_size);
            stats::sub_saturating(&stats::TOTAL_RESERVED, hp_size);
            return;
        }

        let Some(&size) = self.live_page_allocs.get(&addr) else {
            #[cfg(debug_assertions)]
            panic!(
                "free_page_aligned: pointer {ptr:p} not found in live_page_allocs \
                  (double-free or invalid pointer)"
            );
            #[cfg(not(debug_assertions))]
            return;
        };

        if self
            .cached_bytes
            .checked_add(size)
            .is_some_and(|next| next <= self.cache_limit)
        {
            // Decommit to release physical memory but keep reservation
            // Safety: FFI call to decommit memory.
            if unsafe { PlatformVmOps::decommit(ptr, size) }.is_ok() {
                self.live_page_allocs.remove(&addr);
                stats::sub_saturating(&stats::TOTAL_COMMITTED, size);
                stats::sub_saturating(&stats::LARGE_ALLOC_CACHE_COMMITTED, size);

                self.cached.entry(size).or_default().push(ptr);
                self.cached_bytes += size;
            } else if 
                // Safety: FFI call to release memory.
                unsafe { PlatformVmOps::release(ptr, size) }.is_ok() {
                // Decommit failed; release directly instead of caching.
                // Safety: FFI call to release memory.
                self.live_page_allocs.remove(&addr);
                stats::sub_saturating(&stats::TOTAL_COMMITTED, size);
                stats::sub_saturating(&stats::LARGE_ALLOC_CACHE_COMMITTED, size);
                stats::sub_saturating(&stats::TOTAL_RESERVED, size);
            } else {
                // Both failed; keep tracked so we don't prefer leaking over losing track
                #[cfg(debug_assertions)]
                panic!(
                    "free_page_aligned: both decommit and release failed for {ptr:p} (size={size})"
                );
            }
        } else {
            // Too large or cache full, return to OS
            // Safety: FFI call to release memory.
            if unsafe { PlatformVmOps::release(ptr, size) }.is_ok() {
                self.live_page_allocs.remove(&addr);
                stats::sub_saturating(&stats::TOTAL_COMMITTED, size);
                stats::sub_saturating(&stats::LARGE_ALLOC_CACHE_COMMITTED, size);
                stats::sub_saturating(&stats::TOTAL_RESERVED, size);
             }
             // If release fails, we leave it in live_page_allocs (leak prevention by tracking)
        }
    }

    /// Over-aligned free: look up the original reservation and release it.
    /// Over-aligned allocs bypass the cache (rare; caching padding is wasteful).
    fn free_over_aligned(&mut self, ptr: NonNull<u8>) {
        let addr = ptr.as_ptr() as usize;
        let Some(entry) = self.over_aligned.remove(&addr) else {
            #[cfg(debug_assertions)]
            panic!(
                "free_over_aligned: pointer {ptr:p} not found in over_aligned map (double-free or misrouted pointer)"
            );
            #[cfg(not(debug_assertions))]
            return; // Silent no-op in release: nothing to release
        };

        // Safety: FFI call to release memory.
        unsafe {
            drop(PlatformVmOps::release(entry.original_base, entry.total_reserved));
        }
        stats::sub_saturating(&stats::TOTAL_COMMITTED, entry.committed_size);
        stats::sub_saturating(&stats::LARGE_ALLOC_CACHE_COMMITTED, entry.committed_size);
        stats::sub_saturating(&stats::TOTAL_RESERVED, entry.total_reserved);
    }

    pub fn trim(&mut self) {
        self.trim_to(0);
    }

    /// Partially trim the cache until total cached bytes <= target.
    /// Releases largest blocks first.
    pub fn trim_to(&mut self, target: usize) {
        // Iterate sizes in descending order to release largest chunks first.
        // BTreeMap keys are already in ascending order; reverse for descending.
        let mut sizes: Vec<usize> = self.cached.keys().copied().collect();
        sizes.reverse();

        for size in sizes {
            if self.cached_bytes <= target {
                break;
            }

            if let Some(mut list) = self.cached.remove(&size) {
                while let Some(ptr) = list.pop() {
                    // Safety: FFI call to release memory.
                    unsafe {
                        drop(PlatformVmOps::release(ptr, size));
                        stats::sub_saturating(&stats::TOTAL_RESERVED, size);
                    }
                    self.cached_bytes -= size;

                    if self.cached_bytes <= target {
                        break;
                    }
                }

                if !list.is_empty() {
                    self.cached.insert(size, list);
                }
            }
        }
    }

    #[cfg(test)]
    pub fn total_cached_bytes(&self) -> usize {
        self.cached_bytes
    }
}

impl Drop for LargeAllocCache {
    fn drop(&mut self) {
        self.trim();
        // Release any remaining over-aligned allocations
        for (_, entry) in self.over_aligned.drain() {
            // Safety: FFI call to release memory.
            unsafe {
                drop(PlatformVmOps::release(entry.original_base, entry.total_reserved));
            }
            stats::sub_saturating(&stats::TOTAL_COMMITTED, entry.committed_size);
            stats::sub_saturating(&stats::LARGE_ALLOC_CACHE_COMMITTED, entry.committed_size);
            stats::sub_saturating(&stats::TOTAL_RESERVED, entry.total_reserved);
        }
        // Release any remaining huge page allocations
        for (addr, hp_size) in self.huge_allocs.drain() {
            // Safety: FFI call to release memory.
            unsafe {
                let ptr = NonNull::new_unchecked(addr as *mut u8);
                drop(PlatformVmOps::release(ptr, hp_size));
            }
            stats::sub_saturating(&stats::TOTAL_COMMITTED, hp_size);
            stats::sub_saturating(&stats::LARGE_ALLOC_CACHE_COMMITTED, hp_size);
            stats::sub_saturating(&stats::TOTAL_RESERVED, hp_size);
        }
        // Release any remaining live page-aligned allocations.
        for (addr, size) in self.live_page_allocs.drain() {
            // Safety: FFI call to release memory.
            unsafe {
                let ptr = NonNull::new_unchecked(addr as *mut u8);
                drop(PlatformVmOps::release(ptr, size));
            }
            stats::sub_saturating(&stats::TOTAL_COMMITTED, size);
            stats::sub_saturating(&stats::LARGE_ALLOC_CACHE_COMMITTED, size);
            stats::sub_saturating(&stats::TOTAL_RESERVED, size);
        }
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use crate::sync::{Arc, Mutex};

    /// Convenience: build a Layout with align=1 for the given size.
    fn lay(size: usize) -> Layout {
        Layout::from_size_align(size, 1).unwrap()
    }

    #[test]
    fn test_large_cache_reuse_exact() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let page_size = PlatformVmOps::page_size();
        let mut cache = LargeAllocCache::new(page_size * 10);

        let (ptr1, s1) = cache.alloc(lay(page_size)).unwrap();
        let addr1 = ptr1.as_ptr() as usize;

        cache.free(ptr1, lay(s1));

        let (ptr2, s2) = cache.alloc(lay(page_size)).unwrap();
        let addr2 = ptr2.as_ptr() as usize;

        assert_eq!(addr1, addr2);
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_large_cache_reuse_larger() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let page_size = PlatformVmOps::page_size();
        let mut cache = LargeAllocCache::new(page_size * 10);

        let (ptr1, s1) = cache.alloc(lay(page_size * 2)).unwrap();
        let addr1 = ptr1.as_ptr() as usize;

        cache.free(ptr1, lay(s1));

        let (ptr2, s2) = cache.alloc(lay(page_size)).unwrap();
        let addr2 = ptr2.as_ptr() as usize;

        assert_eq!(addr1, addr2);
        assert_eq!(s2, page_size * 2);
    }

    #[test]
    fn test_large_cache_limit() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let page_size = PlatformVmOps::page_size();
        let mut cache = LargeAllocCache::new(page_size);

        let (p1, s1) = cache.alloc(lay(page_size)).unwrap();
        let (p2, s2) = cache.alloc(lay(page_size)).unwrap();

        cache.free(p1, lay(s1));
        assert_eq!(cache.cached_bytes, page_size);

        cache.free(p2, lay(s2));
        assert_eq!(cache.cached_bytes, page_size);
    }

    #[test]
    fn test_large_cache_drop_releases() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let page_size = PlatformVmOps::page_size();
        let mut cache = LargeAllocCache::new(page_size * 10);
        let (p1, s1) = cache.alloc(lay(page_size)).unwrap();
        cache.free(p1, lay(s1));
        assert_eq!(cache.total_cached_bytes(), page_size);
    }

    #[test]
    fn test_large_cache_multiple_sizes() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let page_size = PlatformVmOps::page_size();
        let mut cache = LargeAllocCache::new(page_size * 10);

        let (p1, s1) = cache.alloc(lay(page_size)).unwrap();
        let (p2, s2) = cache.alloc(lay(page_size * 2)).unwrap();

        cache.free(p1, lay(s1));
        cache.free(p2, lay(s2));

        let (r1, sr1) = cache.alloc(lay(page_size * 2)).unwrap();
        let (r2, sr2) = cache.alloc(lay(page_size)).unwrap();

        assert_eq!(r1.as_ptr() as usize, p2.as_ptr() as usize);
        assert_eq!(r2.as_ptr() as usize, p1.as_ptr() as usize);
        assert_eq!(sr1, page_size * 2);
        assert_eq!(sr2, page_size);
    }

    #[test]
    fn test_large_cache_decommit_on_reuse() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let page_size = PlatformVmOps::page_size();
        let mut cache = LargeAllocCache::new(page_size * 10);

        let (p1, s1) = cache.alloc(lay(page_size)).unwrap();
        cache.free(p1, lay(s1));

        let (p2, _s2) = cache.alloc(lay(page_size)).unwrap();
        // Safety: Test code.
        unsafe {
            *p2.as_ptr() = 0xFF;
        }
    }

    #[test]
    fn test_large_cache_page_alignment() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let mut cache = LargeAllocCache::new(1024 * 1024);
        let (ptr, _s) = cache.alloc(lay(123)).unwrap();
        assert_eq!(ptr.as_ptr() as usize % PlatformVmOps::page_size(), 0);
    }

    #[test]
    fn test_large_cache_trim_empty() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let mut cache = LargeAllocCache::new(4096 * 10);
        cache.trim();
        assert_eq!(cache.total_cached_bytes(), 0);
    }

    #[test]
    fn test_large_cache_alloc_after_trim() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let page_size = PlatformVmOps::page_size();
        let mut cache = LargeAllocCache::new(page_size * 10);

        let (p1, s1) = cache.alloc(lay(page_size)).unwrap();
        cache.free(p1, lay(s1));
        assert_eq!(cache.total_cached_bytes(), page_size);

        cache.trim();
        assert_eq!(cache.total_cached_bytes(), 0);

        let (p2, _s2) = cache.alloc(lay(page_size)).unwrap();
        // Safety: Test code.
        unsafe {
            *p2.as_ptr() = 0xAA;
        }
    }

    #[test]
    fn test_large_cache_stats_lifecycle() {
        let _guard = crate::memory::TEST_MUTEX.write().unwrap();
        let page_size = PlatformVmOps::page_size();
        let mut cache = LargeAllocCache::new(page_size * 10);

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
        let mut cache = LargeAllocCache::new(page_size * 10);

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
        assert_eq!(cache.total_cached_bytes(), page_size * 2);

        cache.trim();
        assert_eq!(cache.total_cached_bytes(), 0);
    }

    #[test]
    fn test_large_cache_concurrent() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();

        let limit = 10 * 1024 * 1024;
        let cache = Arc::new(Mutex::new(LargeAllocCache::new(limit)));
        let mut handles = vec![];

        for t in 0u8..4 {
            let c = cache.clone();
            handles.push(crate::sync::thread::spawn(move || {
                let page_size = PlatformVmOps::page_size();
                let mut ptrs = vec![];

                for i in 0..50 {
                    let size = page_size * (1 + (i % 4));
                    let (ptr, actual_size): (NonNull<u8>, usize) = c.lock().unwrap().alloc(lay(size)).unwrap();
                    // Safety: Test code.
                    unsafe {
                        ptr.as_ptr().write(t);
                        assert_eq!(ptr.as_ptr().read(), t);
                    }
                    ptrs.push((ptr, actual_size));

                    if i % 3 == 0 && !ptrs.is_empty() {
                        let (p, s) = ptrs.pop().unwrap();
                        c.lock().unwrap().free(p, lay(s));
                    }
                }

                for (p, s) in ptrs {
                    c.lock().unwrap().free(p, lay(s));
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        cache.lock().unwrap().trim();
        assert_eq!(cache.lock().unwrap().total_cached_bytes(), 0);
    }

    #[test]
    fn test_large_cache_best_fit_no_leak() {
        let _guard = crate::memory::TEST_MUTEX.write().unwrap();
        let page_size = PlatformVmOps::page_size();
        let mut cache = LargeAllocCache::new(page_size * 10);

        let initial_reserved = stats::TOTAL_RESERVED.load(Ordering::Relaxed);

        let (p1, s1) = cache.alloc(lay(page_size * 2)).unwrap();
        assert_eq!(s1, page_size * 2);
        let reserved_after_alloc1 = stats::TOTAL_RESERVED.load(Ordering::Relaxed);
        assert!(reserved_after_alloc1 >= initial_reserved + page_size * 2);

        cache.free(p1, lay(s1));
        assert_eq!(cache.total_cached_bytes(), page_size * 2);

        let (p2, s2) = cache.alloc(lay(page_size)).unwrap();
        assert_eq!(p1, p2);
        assert_eq!(s2, page_size * 2);

        cache.free(p2, lay(s2));
        assert_eq!(cache.total_cached_bytes(), page_size * 2);

        cache.trim();
        assert_eq!(cache.total_cached_bytes(), 0);

        let final_reserved = stats::TOTAL_RESERVED.load(Ordering::Relaxed);
        assert_eq!(
            final_reserved, initial_reserved,
            "Address space leak detected in TOTAL_RESERVED!"
        );
    }

    #[test]
    fn test_large_cache_trim_to_eviction_order() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let page_size = PlatformVmOps::page_size();
        let mut cache = LargeAllocCache::new(page_size * 20);

        let sizes = [page_size, page_size * 2, page_size * 4, page_size * 3];
        let mut ptrs = Vec::new();
        for &s in &sizes {
            let (p, actual) = cache.alloc(lay(s)).unwrap();
            ptrs.push((p, actual));
        }
        for (p, s) in ptrs {
            cache.free(p, lay(s));
        }
        assert_eq!(cache.total_cached_bytes(), page_size * 10);

        cache.trim_to(page_size * 3);
        assert_eq!(cache.total_cached_bytes(), page_size * 3);

        let (_, s1) = cache.alloc(lay(page_size * 2)).unwrap();
        assert_eq!(s1, page_size * 2, "2-page entry should survive trim");

        let (_, s2) = cache.alloc(lay(page_size)).unwrap();
        assert_eq!(s2, page_size, "1-page entry should survive trim");

        assert_eq!(cache.total_cached_bytes(), 0);

        let (_, s3) = cache.alloc(lay(page_size * 4)).unwrap();
        assert_eq!(s3, page_size * 4);
    }

    #[test]
    fn test_large_cache_trim_to_partial_within_size_class() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let page_size = PlatformVmOps::page_size();
        let mut cache = LargeAllocCache::new(page_size * 20);

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
        assert_eq!(cache.total_cached_bytes(), page_size * 4);

        let (_, s1) = cache.alloc(lay(page_size * 2)).unwrap();
        assert_eq!(s1, page_size * 2);
        let (_, s2) = cache.alloc(lay(page_size * 2)).unwrap();
        assert_eq!(s2, page_size * 2);
        assert_eq!(cache.total_cached_bytes(), 0);
    }

    #[test]
    fn test_large_cache_over_aligned() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Test allocation with alignment > page_size
        let page_size = PlatformVmOps::page_size();
        let big_align = page_size * 4; // e.g. 16KB on a 4KB page system
        let mut cache = LargeAllocCache::new(page_size * 10);

        let layout = Layout::from_size_align(page_size, big_align).unwrap();
        let (ptr, actual_size) = cache.alloc(layout).unwrap();

        // Verify alignment
        assert_eq!(
            ptr.as_ptr() as usize % big_align,
            0,
            "Over-aligned allocation should be aligned to {big_align}"
        );
        // Verify we can write
        // Safety: Test code.
        unsafe {
            *ptr.as_ptr() = 0xCC;
        }
        assert_eq!(actual_size, page_size);

        // Free with the same layout
        cache.free(ptr, layout);
        // Over-aligned allocs bypass cache, so cached_bytes should be 0
        assert_eq!(cache.total_cached_bytes(), 0);
    }
}
