use super::stats;
use super::vm::{PlatformVmOps, VmError, VmOps};
use crate::sync::atomic::Ordering;
use crate::sync::{Arc, Mutex, OnceLock};
use std::ptr::NonNull;

/// A thread-safe pool of pages to reduce kernel overhead.
pub struct SharedPagePoolState {
    pages: std::collections::BTreeMap<usize, Vec<NonNull<u8>>>,
    bytes: usize,
}

// Safety: SharedPagePoolState owns the pages (pointers).
// It is protected by a Mutex in SharedPagePool.
// The pointers are just raw addresses of allocated pages.
unsafe impl Send for SharedPagePoolState {}

pub struct SharedPagePool {
    state: Mutex<SharedPagePoolState>,
    capacity_bytes: usize,
}

impl SharedPagePool {
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            state: Mutex::new(SharedPagePoolState {
                pages: std::collections::BTreeMap::new(),
                bytes: 0,
            }),
            capacity_bytes: capacity,
        }
    }

    /// Allocate a page from the shared pool.
    ///
    /// Pages are aligned to the OS page granularity (`mmap`/`VirtualAlloc`)
    /// and no further — callers must NOT assume `size`-alignment.
    ///
    /// # Panics
    ///
    /// Panics if the internal lock is poisoned.
    ///
    /// # Errors
    ///
    /// Returns `VmError` if allocation fails (e.g. OOM).
    pub fn alloc(&self, size: usize) -> Result<NonNull<u8>, VmError> {
        // Try to pop from cache
        let mut guard = self.state.lock().unwrap();
        let state = &mut *guard;
        if let Some(list) = state.pages.get_mut(&size)
            && let Some(ptr) = list.pop()
        {
            state.bytes -= size;
            // If list empty, remove? Not strictly necessary for functionality but cleaner
            if list.is_empty() {
                state.pages.remove(&size);
            }
            #[cfg(any(debug_assertions, feature = "hardened"))]
            // Safety: ptr is valid and size is correct.
            unsafe {
                std::ptr::write_bytes(ptr.as_ptr(), 0, size);
            }
            return Ok(ptr);
        }
        drop(guard);

        // Safety: FFI calls to reserve and commit memory.
        unsafe {
            let p = PlatformVmOps::reserve(size)?;
            if let Err(e) = PlatformVmOps::commit(p, size) {
                drop(PlatformVmOps::release(p, size));
                return Err(e);
            }

            stats::TOTAL_RESERVED.fetch_add(size, Ordering::Relaxed);
            stats::TOTAL_COMMITTED.fetch_add(size, Ordering::Relaxed);
            stats::COMMAND_ARENA_COMMITTED.fetch_add(size, Ordering::Relaxed);

            #[cfg(any(debug_assertions, feature = "hardened"))]
            std::ptr::write_bytes(p.as_ptr(), 0, size);

            Ok(p)
        }
    }

    /// Return a page previously allocated by this pool.
    ///
    /// # Safety
    /// - `ptr` must have been returned by [`alloc`](Self::alloc) on this exact pool.
    /// - `size` must exactly match the size used to allocate `ptr`.
    /// - `ptr` must not have been freed already.
    ///
    /// # Panics
    ///
    /// Panics if the internal lock is poisoned.
    pub unsafe fn free(&self, ptr: NonNull<u8>, size: usize) {
        let mut state = self.state.lock().unwrap();

        // Note: We keep pages committed to avoid commit/decommit overhead on every reuse.
        // This is a trade-off: higher memory usage for better performance.
        if state
            .bytes
            .checked_add(size)
            .is_some_and(|next| next <= self.capacity_bytes)
        {
            state.pages.entry(size).or_default().push(ptr);
            state.bytes += size;
        } else {
            // Cache full, release to OS
            // Safety: FFI call to release memory.
            unsafe {
                drop(PlatformVmOps::release(ptr, size));
                stats::TOTAL_RESERVED.sub(size);
                stats::TOTAL_COMMITTED.sub(size);
                stats::COMMAND_ARENA_COMMITTED.sub(size);
            }
        }
    }

    /// Release all cached pages to the OS.
    ///
    /// # Panics
    ///
    /// Panics if the internal lock is poisoned.
    pub fn trim(&self) {
        let mut state = self.state.lock().unwrap();
        for (size, list) in &state.pages {
            for ptr in list {
                // Safety: FFI call to release memory.
                unsafe {
                    drop(PlatformVmOps::release(*ptr, *size));
                    stats::TOTAL_RESERVED.sub(*size);
                    stats::TOTAL_COMMITTED.sub(*size);
                    stats::COMMAND_ARENA_COMMITTED.sub(*size);
                }
            }
        }
        state.pages.clear();
        state.bytes = 0;
    }
}

static GLOBAL_PAGE_POOL: OnceLock<Arc<SharedPagePool>> = OnceLock::new();

pub struct GlobalSharedPagePool;

impl GlobalSharedPagePool {
    pub fn get() -> Arc<SharedPagePool> {
        GLOBAL_PAGE_POOL
            .get_or_init(|| {
                // Default 64MB shared pool for command pages
                Arc::new(SharedPagePool::new(64 * 1024 * 1024))
            })
            .clone()
    }

    pub fn trim() {
        if let Some(pool) = GLOBAL_PAGE_POOL.get() {
            pool.trim();
        }
    }
}

impl Drop for SharedPagePool {
    fn drop(&mut self) {
        // Since we have `&mut self`, we can bypass the lock.
        let state = match self.state.get_mut() {
            Ok(s) => s,
            Err(e) => e.into_inner(),
        };

        for (size, list) in &state.pages {
            for ptr in list {
                // Safety: FFI call to release memory.
                unsafe {
                    drop(PlatformVmOps::release(*ptr, *size));
                    stats::TOTAL_RESERVED.sub(*size);
                    stats::TOTAL_COMMITTED.sub(*size);
                    stats::COMMAND_ARENA_COMMITTED.sub(*size);
                }
            }
        }
        state.pages.clear();
        state.bytes = 0;
    }
}

struct PageInfo {
    ptr: *mut u8,
    capacity: usize,
    used: usize,
}

// Safety: PageInfo owns the memory pointer.
unsafe impl Send for PageInfo {}

/// A paged linear allocator for command buffers.
/// Pages are allocated from a shared pool (or VM in this simple implementation).
pub struct CommandArena {
    original_pages: Vec<PageInfo>,
    current_page: usize,
    cursor: usize,
    page_size: usize,
    /// Maximum alignment satisfiable on every page. Pool pages are only
    /// guaranteed OS-page-aligned (see [`SharedPagePool::alloc`]), so an
    /// alignment above `min(page_size, os_page_size)` could fail to fit
    /// even in a fresh page; [`push`](Self::push) rejects it up front.
    max_align: usize,
    pool: Arc<SharedPagePool>,
}

// Safety: CommandArena owns its pages.
unsafe impl Send for CommandArena {}

impl CommandArena {
    /// Create an arena that carves objects out of `page_size`-byte pages
    /// from `pool`.
    ///
    /// `page_size` bounds the largest object (see [`push`](Self::push));
    /// sizes below the OS page granularity still consume a full OS page of
    /// physical memory per arena page.
    ///
    /// # Panics
    ///
    /// Panics if `page_size` is zero.
    pub fn new(page_size: usize, pool: Arc<SharedPagePool>) -> Self {
        assert!(page_size > 0, "CommandArena page_size must be non-zero");
        Self {
            original_pages: Vec::new(),
            current_page: 0,
            cursor: 0,
            page_size,
            max_align: page_size.min(PlatformVmOps::page_size()),
            pool,
        }
    }

    /// Allocate a new page.
    fn add_page(&mut self) -> Result<(), VmError> {
        let ptr = self.pool.alloc(self.page_size)?;
        self.original_pages.push(PageInfo {
            ptr: ptr.as_ptr(),
            capacity: self.page_size,
            used: 0,
        });
        Ok(())
    }

    /// Push a command object into the arena.
    ///
    /// A pushed object must fit entirely within a single page; objects are not
    /// split across pages.
    ///
    /// # Errors
    ///
    /// Returns `VmError` if the object is too large or if the arena fails to allocate a new page.
    pub fn push<T: Copy>(&mut self, val: T) -> Result<*mut T, VmError> {
        let layout = std::alloc::Layout::new::<T>();
        let size = layout.size();
        let align = layout.align();

        // A single object MUST fit in one page; objects are not split.
        // Alignment is capped at min(page_size, OS page size): pool pages
        // are only guaranteed OS-page-aligned, so a larger alignment could
        // fail to fit even in a fresh page (the loop below would then
        // reserve new pages forever). Under this cap, a fresh page (cursor
        // 0, OS-page-aligned base ⇒ zero padding) always fits any accepted
        // (size, align), so the loop terminates after at most one add_page.
        if size > self.page_size || align > self.max_align {
            return Err(VmError::ObjectTooLarge {
                size: std::cmp::max(size, align),
                page_size: self.page_size,
            });
        }

        // Ensure we have at least one page
        if self.original_pages.is_empty() {
            self.add_page()?;
        }

        loop {
            // Check if we ran out of pages
            if self.current_page >= self.original_pages.len() {
                self.add_page()?;
            }

            let page_info = &mut self.original_pages[self.current_page];
            let page_ptr = page_info.ptr;
            let page_cap = page_info.capacity;

            // The fit check happens entirely in offsets; a pointer is only
            // materialized (via `add`) once the range is proven in-bounds.
            // Speculatively computing an out-of-bounds `end` pointer — as an
            // earlier revision did — is undefined behaviour even if it is
            // never dereferenced.
            //
            // `cursor` is the offset into the current page (resets to 0 on
            // page switch); padding is computed from the real address so the
            // returned pointer is aligned.
            let current_addr = (page_ptr as usize) + self.cursor;
            let padding = (align - (current_addr % align)) % align;
            let start_offset = self.cursor + padding;
            let end_offset = start_offset.saturating_add(size);

            if end_offset <= page_cap {
                // Fits in current page
                self.cursor = end_offset;
                // Update used
                if self.cursor > page_info.used {
                    page_info.used = self.cursor;
                }

                // Safety: start_offset + size <= page_cap, so the range is
                // in-bounds of the page allocation; padding makes it aligned.
                let ptr = unsafe { page_ptr.add(start_offset) }.cast::<T>();
                // Safety: ptr is valid for writes of T (in-bounds, aligned).
                unsafe { ptr.write(val) };
                return Ok(ptr);
            }

            self.current_page += 1;
            self.cursor = 0;
        }
    }

    /// Documenting that this method panics on OOM.
    /// This is a temporary design choice for the game engine hot path.
    /// Future improvement: propagate Result.
    ///
    /// # Panics
    ///
    /// Panics if allocation fails (e.g. out of memory).
    pub fn push_or_panic<T: Copy>(&mut self, val: T) -> *mut T {
        match self.push(val) {
            Ok(p) => p,
            Err(e) => panic!("CommandArena::push_or_panic failed: {e:?}"),
        }
    }

    /// Reset for reuse. Retains pages.
    ///
    /// # Invalidation contract
    ///
    /// Logically invalidates every pointer previously returned by
    /// [`push`](Self::push)/[`push_or_panic`](Self::push_or_panic):
    /// subsequent pushes hand out the same memory again. The returned raw
    /// pointers are not lifetime-tracked, so the borrow checker cannot
    /// enforce this — reading through a pre-reset pointer after new pushes
    /// observes overwritten bytes (undefined behaviour once they no longer
    /// form a valid `T`).
    pub fn reset(&mut self) {
        self.current_page = 0;
        self.cursor = 0;
        for page in &mut self.original_pages {
            page.used = 0;
        }
    }
    /// Iterate raw used-byte prefixes for each currently used page.
    ///
    /// Contract and intended use:
    /// - Each yielded slice is `&page[0..used]` for one arena page.
    /// - Slices are page-scoped; there is no cross-page packing layer.
    /// - Bytes represent raw arena memory, not framed command records.
    /// - Alignment gaps between consecutive `push<T>()` calls are included.
    /// - Bytes past `used` in each page are never yielded.
    ///
    /// This API is for low-level tooling (diagnostics, dumps, page-prefix hashing,
    /// transport of raw page payloads). It is **not** a typed command-stream API.
    ///
    /// Do not assume:
    /// - every byte corresponds to command payload,
    /// - object boundaries are encoded,
    /// - a single slice contains all pushed commands.
    ///
    /// If you need a structured command stream, track record boundaries separately
    /// (for example, an out-of-band descriptor list).
    #[inline]
    #[must_use]
    pub fn iter_pages(&self) -> CommandIter<'_> {
        CommandIter {
            arena: self,
            page_idx: 0,
        }
    }
}

pub struct CommandIter<'a> {
    arena: &'a CommandArena,
    page_idx: usize,
}

impl<'a> Iterator for CommandIter<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        while self.page_idx <= self.arena.current_page
            && self.page_idx < self.arena.original_pages.len()
        {
            let page_info = &self.arena.original_pages[self.page_idx];
            let len = page_info.used;
            let ptr = page_info.ptr;
            self.page_idx += 1;

            if len > 0 {
                // Safety: ptr and len are tracked by the arena and guaranteed valid.
                return Some(unsafe { std::slice::from_raw_parts(ptr, len) });
            }
        }
        None
    }
}

impl Drop for CommandArena {
    fn drop(&mut self) {
        // Return all pages to pool
        for page in &self.original_pages {
            if let Some(p) = NonNull::new(page.ptr) {
                // Safety: p was allocated from self.pool with capacity.
                unsafe {
                    self.pool.free(p, page.capacity);
                }
            }
        }
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use crate::memory::stats;
    use crate::sync::atomic::Ordering;

    #[test]
    fn test_command_arena_push() {
        // Push enough to cross page boundary

        #[derive(Clone, Copy)]
        struct LargeData {
            _data: [u8; 1024],
        }
        let data = LargeData { _data: [0; 1024] };

        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let page_size = PlatformVmOps::page_size();
        let pool = Arc::new(SharedPagePool::new(1024 * 1024)); // 1MB limit
        let mut arena = CommandArena::new(page_size, pool);

        // Push small item
        let p1 = arena.push(42u32).unwrap();
        // Safety: Test code.
        unsafe {
            assert_eq!(*p1, 42);
        }

        // Push another
        let p2 = arena.push(123u64).unwrap();
        // Safety: Test code.
        unsafe {
            assert_eq!(*p2, 123);
        }

        // Push until page fills
        let mut pushes = 0;
        loop {
            arena.push(data).unwrap();
            pushes += 1;
            if pushes * 1024 > page_size {
                break;
            }
        }

        arena.reset();

        // Should reuse pages
        let p_new = arena.push(999u32).unwrap();
        // Safety: Test code.
        unsafe {
            assert_eq!(*p_new, 999);
        }
    }

    #[cfg(any(debug_assertions, feature = "hardened"))]
    #[test]
    #[should_panic(expected = "ObjectTooLarge")]
    fn test_command_arena_panic_on_too_large() {
        #[derive(Clone, Copy)]
        struct Huge {
            _data: [u8; 5000],
        }

        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let page_size = 4096;
        let pool = Arc::new(SharedPagePool::new(1024 * 1024));
        let mut arena = CommandArena::new(page_size, pool);

        arena.push(Huge { _data: [0; 5000] }).unwrap();
    }
    #[test]
    fn test_command_arena_alignment() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // D1: Verify alignment of pushed objects
        let page_size = 4096;
        let pool = Arc::new(SharedPagePool::new(1024 * 1024));
        let mut arena = CommandArena::new(page_size, pool);

        let p1 = arena.push(1u8).unwrap();
        let p2 = arena.push(1u32).unwrap();
        let p3 = arena.push(1u64).unwrap();

        {
            assert_eq!(p1 as usize % std::mem::align_of::<u8>(), 0);
            assert_eq!(p2 as usize % std::mem::align_of::<u32>(), 0);
            assert_eq!(p3 as usize % std::mem::align_of::<u64>(), 0);
        }
    }

    #[test]
    fn test_command_arena_rejects_high_alignment() {
        use crate::sync::Arc;

        // Unused field, but recall ZSTs are UB on allocator API, cf. nomicon
        #[derive(Clone, Copy)]
        #[repr(align(131072))]
        #[allow(dead_code)]
        struct HugeAlign(u8); // 128KB alignment, exceeds any base page size we support.

        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Align greater than page size should be rejected with ObjectTooLarge.
        let page_size = PlatformVmOps::page_size();
        let pool = Arc::new(SharedPagePool::new(page_size * 2));
        let mut arena = CommandArena::new(page_size, pool);

        assert!(std::mem::align_of::<HugeAlign>() > page_size);

        match arena.push(HugeAlign(1)) {
            Err(VmError::ObjectTooLarge {
                size,
                page_size: ps,
            }) => {
                assert_eq!(ps, page_size);
                assert_eq!(size, std::cmp::max(1, std::mem::align_of::<HugeAlign>()));
            }
            other => panic!("expected ObjectTooLarge, got {other:?}"),
        }
    }

    #[test]
    fn test_command_arena_rejects_alignment_beyond_os_page_granularity() {
        use crate::sync::Arc;

        // Pool pages are only OS-page-aligned, so an alignment above the OS
        // page (even one below the arena's page_size) cannot be guaranteed.
        // The old code accepted it and, when a page happened to be
        // misaligned, looped forever reserving pages that could never fit.
        #[derive(Clone, Copy)]
        #[repr(align(131072))]
        #[allow(dead_code)]
        struct WideAlign(u8);

        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let os_page = PlatformVmOps::page_size();
        assert!(std::mem::align_of::<WideAlign>() > os_page);

        // Arena pages big enough that the OLD `align > page_size` check
        // would have accepted this alignment.
        let arena_page = std::mem::align_of::<WideAlign>() * 2;
        let pool = Arc::new(SharedPagePool::new(arena_page * 2));
        let mut arena = CommandArena::new(arena_page, pool);

        assert!(
            matches!(
                arena.push(WideAlign(1)),
                Err(VmError::ObjectTooLarge { .. })
            ),
            "alignment above OS page granularity must be rejected, not looped on"
        );
    }

    #[test]
    fn test_command_arena_os_page_alignment_supported() {
        use crate::sync::Arc;

        // 4096 is <= the OS page size on every supported platform, so this
        // alignment must be satisfiable on every page, including across
        // page boundaries (loop termination).
        #[derive(Clone, Copy)]
        #[repr(C, align(4096))]
        struct PageAligned([u8; 64]);

        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let os_page = PlatformVmOps::page_size();
        let pool = Arc::new(SharedPagePool::new(os_page * 8));
        let mut arena = CommandArena::new(os_page * 2, pool);

        for i in 0..8 {
            let p = arena.push(PageAligned([i; 64])).unwrap();
            assert_eq!(p as usize % 4096, 0, "push {i} misaligned");
        }
    }

    #[test]
    fn test_command_arena_growth() {
        #[derive(Clone, Copy)]
        struct PageData {
            _d: [u8; 4096],
        }

        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // D2: Shared pool grows beyond capacity (capacity is for caching)
        // Shared pool capacity 8192 (2 pages)
        let page_size = 4096;
        let pool = Arc::new(SharedPagePool::new(page_size * 2));
        let mut arena = CommandArena::new(page_size, pool);

        // Push 1 byte -> 1 page
        arena.push(1u8).unwrap();

        // Force new pages
        arena.push(PageData { _d: [0; 4096] }).unwrap(); // Page 2
        arena.push(PageData { _d: [0; 4096] }).unwrap(); // Page 3 (exceeds pool cache capacity, but should succeed alloc from OS)

        // Should succeed
    }

    #[test]
    fn test_command_arena_reset_reuses_pages() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // D4: Push, Reset, Push -> Same pointer (if same size/align sequence)
        let page_size = 4096;
        let pool = Arc::new(SharedPagePool::new(page_size * 4));
        let mut arena = CommandArena::new(page_size, pool);

        let p1 = arena.push(123u64).unwrap();
        let addr1 = p1 as usize;

        arena.reset();

        let p2 = arena.push(456u64).unwrap();
        let addr2 = p2 as usize;

        assert_eq!(addr1, addr2);
        // Safety: Test code.
        unsafe {
            assert_eq!(*p2, 456);
        }
    }

    #[test]
    fn test_command_arena_drop_returns_to_pool() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // D5: create arena, push (alloc page), drop arena.
        // SharedPagePool should have the page in cache.
        let page_size = 4096;
        let pool = Arc::new(SharedPagePool::new(page_size * 10));

        {
            let mut arena = CommandArena::new(page_size, pool.clone());
            arena.push(1u8).unwrap(); // Alloc 1 page
        } // Drop arena
    }

    #[test]
    fn test_command_arena_many_large_objects() {
        #[derive(Clone, Copy)]
        struct Big {
            _d: [u8; 4000],
        }

        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // D8: Push many large objects (near page size)
        let page_size = 4096;
        let pool = Arc::new(SharedPagePool::new(page_size * 100));
        let mut arena = CommandArena::new(page_size, pool);

        for _ in 0..50 {
            arena.push(Big { _d: [0; 4000] }).unwrap();
        }
        // Should succeed
    }

    #[test]
    fn test_command_arena_zst() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // D7: Zero sized type
        let page_size = 4096;
        let pool = Arc::new(SharedPagePool::new(page_size));
        let mut arena = CommandArena::new(page_size, pool);

        let p = arena.push(()).unwrap();
        // ZST usually has dangling pointer or non-null.
        // Should not crash.
        // Safety: Test code.
        unsafe {
            assert_eq!(*p, ());
        }
    }

    #[test]
    fn test_command_arena_mixed_sizes() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // D9: Interleave small and large
        let page_size = 1024;
        let pool = Arc::new(SharedPagePool::new(page_size * 10));
        let mut arena = CommandArena::new(page_size, pool);

        for i in 0..100 {
            if i % 2 == 0 {
                arena.push(1u8).unwrap();
            } else {
                arena.push(100u64).unwrap();
            }
        }
    }

    // Test D3, D6 implied by others or simple variants
    // D6 cross page: covered by `test_command_arena_push` loop.
    // D3 iter: Not supported directly.

    #[test]
    fn test_command_arena_stats() {
        // D10: verify the command-arena gauge grows when a page is
        // committed and returns exactly to baseline after trim.
        //
        // Only COMMAND_ARENA_COMMITTED is asserted: cross-subsystem gauges
        // (TOTAL_COMMITTED) also move when other tests' thread-local caches
        // are dropped at thread death, which happens AFTER those tests
        // release TEST_MUTEX — even a write lock can't exclude that.
        let _guard = crate::memory::TEST_MUTEX.write().unwrap();
        let initial_arena = stats::COMMAND_ARENA_COMMITTED.load(Ordering::Relaxed);

        let page_size = 4096;
        let pool = Arc::new(SharedPagePool::new(page_size * 2));
        {
            let mut arena = CommandArena::new(page_size, pool.clone());
            arena.push(1u8).unwrap();

            assert!(
                stats::COMMAND_ARENA_COMMITTED.load(Ordering::Relaxed) >= initial_arena + page_size,
                "COMMAND_ARENA_COMMITTED did not grow by at least one page"
            );
        }

        // Dropping the arena returns the page to the pool (still committed);
        // trimming the pool releases it and restores the gauge exactly.
        pool.trim();
        assert_eq!(
            stats::COMMAND_ARENA_COMMITTED.load(Ordering::Relaxed),
            initial_arena
        );
    }

    #[test]
    fn test_command_arena_iter() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // D3: Iterate
        let page_size = 4096;
        let pool = Arc::new(SharedPagePool::new(page_size * 10));
        let mut arena = CommandArena::new(page_size, pool);

        arena.push(1u8).unwrap();
        let _ = arena.push(2u32).unwrap();

        let chunks: Vec<_> = arena.iter_pages().collect();
        assert_eq!(chunks.len(), 1); // 1 page
        // Size should be 1 + padding + 4.
        // align of u32 is 4.
        // 1u8 is at 0.
        // 2u32 is at 4. (padding 3 bytes).
        // Total 8 bytes used.
        assert_eq!(chunks[0].len(), 8);
    }

    #[test]
    fn test_command_arena_iter_empty() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // I1: Iter on fresh arena — yields nothing
        let page_size = 4096;
        let pool = Arc::new(SharedPagePool::new(page_size));
        let arena = CommandArena::new(page_size, pool);

        let count = arena.iter_pages().count();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_command_arena_iter_after_push() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // I2: Push N items, iter — yields items in order (well, yields pages)
        // CommandArena iter yields slices of pages.
        let page_size = 4096;
        let pool = Arc::new(SharedPagePool::new(page_size));
        let mut arena = CommandArena::new(page_size, pool);

        arena.push(1u8).unwrap();
        arena.push(2u8).unwrap();

        let chunks: Vec<_> = arena.iter_pages().collect();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 2);
    }

    #[test]
    fn test_command_arena_iter_cross_page() {
        #[derive(Clone, Copy)]
        struct Item([u8; 100]);

        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // I3: Push enough to span pages, iter — yields all in order
        let page_size = 128; // Small page
        let pool = Arc::new(SharedPagePool::new(page_size * 4));
        let mut arena = CommandArena::new(page_size, pool);

        let item = Item([0; 100]);
        let _ = item.0; // Mark field as read

        arena.push(Item([0; 100])).unwrap(); // Page 1
        arena.push(Item([1; 100])).unwrap(); // Page 2
        arena.push(Item([2; 100])).unwrap(); // Page 3

        let chunks: Vec<_> = arena.iter_pages().collect();
        assert_eq!(chunks.len(), 3);

        assert_eq!(chunks[0].len(), 100);
        assert_eq!(chunks[1].len(), 100);
        assert_eq!(chunks[2].len(), 100);
    }

    #[test]
    fn test_command_arena_iter_after_reset() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // I4: Push, reset, iter — yields nothing
        let page_size = 4096;
        let pool = Arc::new(SharedPagePool::new(page_size));
        let mut arena = CommandArena::new(page_size, pool);

        arena.push(1u8).unwrap();
        arena.reset();

        let count = arena.iter_pages().count();
        assert_eq!(count, 0);

        // Push again
        arena.push(2u8).unwrap();
        let chunks: Vec<_> = arena.iter_pages().collect();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 1);
    }

    #[test]
    fn test_shared_page_pool_concurrent_alloc_free() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // I5: Multiple threads alloc/free from same pool
        let pool = Arc::new(SharedPagePool::new(1024 * 1024));
        let mut handles = vec![];

        for _ in 0..4 {
            let p = pool.clone();
            handles.push(crate::sync::thread::spawn(move || {
                let size = 4096;
                for _ in 0..50 {
                    let ptr = p.alloc(size).unwrap();
                    // Safety: Test code.
                    unsafe {
                        p.free(ptr, size);
                    }
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_shared_page_pool_capacity_exact_boundary() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // I6: cache capacity is exactly 2 pages; freeing a third page while
        // the cache is full must release it to the OS, not grow the cache.
        let page_size = 4096;
        let pool = Arc::new(SharedPagePool::new(page_size * 2));

        let p1 = pool.alloc(page_size).unwrap();
        let p2 = pool.alloc(page_size).unwrap();
        let p3 = pool.alloc(page_size).unwrap();
        let addr1 = p1.as_ptr() as usize;
        let addr2 = p2.as_ptr() as usize;
        let addr3 = p3.as_ptr() as usize;

        // Return p1, p2: fills the cache to exactly capacity_bytes.
        // Safety: pages were allocated from this pool with this size.
        unsafe {
            pool.free(p1, page_size);
            pool.free(p2, page_size);
        }
        {
            let state = pool.state.lock().unwrap();
            assert_eq!(
                state.bytes,
                page_size * 2,
                "cache must hold exactly capacity"
            );
            assert_eq!(state.pages.get(&page_size).map(Vec::len), Some(2));
        }

        // Cache is at capacity: freeing p3 must evict to the OS.
        // Safety: p3 was allocated from this pool with this size.
        unsafe { pool.free(p3, page_size) };
        {
            let state = pool.state.lock().unwrap();
            assert_eq!(
                state.bytes,
                page_size * 2,
                "over-capacity free must not grow the cache"
            );
            let cached: Vec<usize> = state.pages[&page_size]
                .iter()
                .map(|p| p.as_ptr() as usize)
                .collect();
            assert!(cached.contains(&addr1) && cached.contains(&addr2));
            assert!(
                !cached.contains(&addr3),
                "the evicted page must not appear in the cache"
            );
        }

        // Reallocation drains the cache LIFO, then maps a fresh page.
        let r1 = pool.alloc(page_size).unwrap();
        let r2 = pool.alloc(page_size).unwrap();
        assert_eq!(
            r1.as_ptr() as usize,
            addr2,
            "expected LIFO reuse of cached pages"
        );
        assert_eq!(
            r2.as_ptr() as usize,
            addr1,
            "expected LIFO reuse of cached pages"
        );
        assert_eq!(pool.state.lock().unwrap().bytes, 0, "cache must be drained");
        let r3 = pool.alloc(page_size).unwrap();

        // Return everything and trim so nothing outlives the test (miri
        // runs with the leak checker enabled).
        // Safety: pages were allocated from this pool with this size.
        unsafe {
            pool.free(r1, page_size);
            pool.free(r2, page_size);
            pool.free(r3, page_size);
        }
        pool.trim();
    }

    #[test]
    fn test_shared_page_pool_trim() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let pool = SharedPagePool::new(1024 * 1024);
        let ptr = pool.alloc(4096).unwrap();
        // Safety: Test code.
        unsafe {
            pool.free(ptr, 4096);
        }

        // Should be cached
        {
            let state = pool.state.lock().unwrap();
            assert!(state.bytes > 0);
        }

        pool.trim();

        // Should be empty
        {
            let state = pool.state.lock().unwrap();
            assert_eq!(state.bytes, 0);
            assert!(state.pages.is_empty());
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Miri has limited heap; this test deliberately exhausts memory
    fn test_command_arena_oom_propagation() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // We want to force an OOM by using a ridiculous page size.
        // SharedPagePool will try to reserve this and fail.
        let huge_page_size = 1usize << 60;
        let pool = Arc::new(SharedPagePool::new(0));
        let mut arena = CommandArena::new(huge_page_size, pool);

        // This should fail to allocate the first page
        let res = arena.push(1u8);
        assert!(res.is_err());

        match res.unwrap_err() {
            VmError::ReservationFailed(_) => { /* Good */ }
            other => panic!("Expected ReservationFailed, got {other:?}"),
        }
    }

    #[test]
    fn test_shared_page_pool_stats_tracking() {
        let _guard = crate::memory::TEST_MUTEX.write().unwrap();
        let page_size = 4096;
        let pool = SharedPagePool::new(page_size); // Only room for 1 page in cache

        let initial_command = stats::COMMAND_ARENA_COMMITTED.load(Ordering::Relaxed);

        // 1. Alloc 2 pages
        let p1 = pool.alloc(page_size).unwrap();
        let p2 = pool.alloc(page_size).unwrap();

        let after_alloc = stats::COMMAND_ARENA_COMMITTED.load(Ordering::Relaxed);
        assert!(after_alloc >= initial_command + 2 * page_size);

        // 2. Free p1 (within capacity -> cached)
        // Safety: Test code.
        unsafe {
            pool.free(p1, page_size);
        }
        let after_free1 = stats::COMMAND_ARENA_COMMITTED.load(Ordering::Relaxed);
        // Note: other tests might be running, but our page should NOT be released.
        // So global count should NOT decrease by page_size.
        assert!(after_free1 >= after_alloc);

        // 3. Free p2 (exceeds capacity -> released)
        // Safety: Test code.
        unsafe {
            pool.free(p2, page_size);
        }
        let after_free2 = stats::COMMAND_ARENA_COMMITTED.load(Ordering::Relaxed);
        // This page MUST be released, so global count should decrease relative to what it WOULD be.
        // Since we can't control other threads if we don't use --test-threads=1, we check that it's less than after_free1.
        // If it fails, it's likely noise from concurrent tests.
        assert!(
            after_free2 < after_free1 + page_size,
            "Expected stats to decrease after release, but {after_free2} >= {after_free1} + {page_size}"
        );

        // 4. Trim
        pool.trim();
        let after_trim = stats::COMMAND_ARENA_COMMITTED.load(Ordering::Relaxed);
        assert!(after_trim < after_free2);
    }

    // --- T8: Internal fragmentation measurement ---
    #[test]
    fn test_command_arena_internal_fragmentation() {
        #[derive(Clone, Copy)]
        #[repr(C)]
        struct NearPageSize {
            _data: [u8; 4095], // page_size - 1
        }

        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Push items of size (page_size - 1). Each item wastes ~1 byte per page,
        // but the remainder after the first item can't fit a second → 1 item per page.
        // This documents the P9 fragmentation issue.
        let page_size = 4096;
        let pool = Arc::new(SharedPagePool::new(page_size * 100));
        let mut arena = CommandArena::new(page_size, pool);

        let num_items = 10;
        for _ in 0..num_items {
            arena.push(NearPageSize { _data: [0; 4095] }).unwrap();
        }

        // Count pages used
        let pages_used = arena.iter_pages().count();

        // With 4095-byte items in 4096-byte pages, each item takes a full page.
        // Fragmentation ratio = wasted / total = 1/4096 per page = ~0.02%.
        // But if alignment padding pushes it over, we get 1 item per page.
        assert_eq!(
            pages_used, num_items,
            "Near-page-size items should each consume a full page (P9 fragmentation)"
        );

        // Document: effective utilization is 4095/4096 = 99.97% per page.
        // The real issue is with items of size ~page_size/2, which waste ~50%.
    }

    #[test]
    fn test_command_arena_half_page_fragmentation() {
        #[derive(Clone, Copy)]
        #[repr(C)]
        struct HalfPlusOne {
            _data: [u8; 2049], // page_size/2 + 1
        }

        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Items slightly larger than half-page can't fit 2-per-page.
        let page_size = 4096;
        let pool = Arc::new(SharedPagePool::new(page_size * 100));
        let mut arena = CommandArena::new(page_size, pool);

        let num_items = 10;
        for _ in 0..num_items {
            arena.push(HalfPlusOne { _data: [0; 2049] }).unwrap();
        }

        let pages_used = arena.iter_pages().count();
        // Each item > half page → 1 item per page → 50% waste
        assert_eq!(
            pages_used, num_items,
            "Items > half-page should each waste ~50% (P9 fragmentation)"
        );
    }
}
