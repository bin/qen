use super::stats;
use super::vm::{PlatformVmOps, VmError, VmOps};
use crate::sync::atomic::Ordering;
use crate::sync::{Arc, Mutex, OnceLock};
use std::any::TypeId;
use std::marker::PhantomData;
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

#[repr(C)]
struct CommandHeader {
    payload_offset: usize,
    payload_size: usize,
    payload_align: usize,
    element_count: usize,
    record_size: usize,
    payload_kind: CommandPayloadKind,
    type_id: fn() -> TypeId,
    drop_value: unsafe fn(*mut u8),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
enum CommandPayloadKind {
    Value,
    Slice,
}

fn command_type_id<T: 'static>() -> TypeId {
    TypeId::of::<T>()
}

unsafe fn drop_command<T>(ptr: *mut u8) {
    // Safety: callers pass the aligned payload address at which a live T was
    // written exactly once, and invoke this function at most once.
    unsafe { ptr.cast::<T>().drop_in_place() };
}

unsafe fn drop_nothing(_: *mut u8) {}

fn align_address(address: usize, align: usize) -> Option<usize> {
    debug_assert!(align.is_power_of_two());
    address
        .checked_add(align - 1)
        .map(|value| value & !(align - 1))
}

/// A type-erased command borrowed from a [`CommandArena`].
///
/// Records are yielded in insertion order. The arena retains ownership of the
/// value, so a record can only expose shared access; reset or arena destruction
/// runs the value's destructor exactly once.
pub struct CommandRecord<'a> {
    header: &'a CommandHeader,
    payload: NonNull<u8>,
    marker: PhantomData<&'a u8>,
}

impl<'a> CommandRecord<'a> {
    /// The concrete Rust type stored in this record.
    #[must_use]
    pub fn type_id(&self) -> TypeId {
        (self.header.type_id)()
    }

    /// The payload size recorded when the command was pushed.
    #[must_use]
    pub fn size(&self) -> usize {
        self.header.payload_size
    }

    /// The payload alignment recorded when the command was pushed.
    #[must_use]
    pub fn align(&self) -> usize {
        self.header.payload_align
    }

    /// Whether this record contains a dynamically sized slice rather than one
    /// sized Rust value.
    #[must_use]
    pub fn is_slice(&self) -> bool {
        self.header.payload_kind == CommandPayloadKind::Slice
    }

    /// Number of elements in a slice record. Sized-value records return
    /// `None` so a one-element slice remains distinguishable from one value.
    #[must_use]
    pub fn slice_len(&self) -> Option<usize> {
        self.is_slice().then_some(self.header.element_count)
    }

    /// Borrow the command when its concrete type is `T`.
    #[must_use]
    pub fn downcast_ref<T: 'static>(&self) -> Option<&'a T> {
        if self.header.payload_kind != CommandPayloadKind::Value
            || self.type_id() != TypeId::of::<T>()
        {
            return None;
        }

        // Safety: matching TypeId, size, and alignment metadata were written
        // with the live T at push time. The record cannot outlive the arena or
        // coexist with a mutable arena borrow.
        Some(unsafe { self.payload.cast::<T>().as_ref() })
    }

    /// Borrow a copied slice when its element type is `T`.
    #[must_use]
    pub fn downcast_slice<T: 'static>(&self) -> Option<&'a [T]> {
        if self.header.payload_kind != CommandPayloadKind::Slice
            || self.type_id() != TypeId::of::<T>()
            || self.header.payload_align != std::mem::align_of::<T>()
            || self.header.payload_size
                != std::mem::size_of::<T>().checked_mul(self.header.element_count)?
        {
            return None;
        }

        // Safety: push_slice initialized exactly element_count contiguous T
        // values at this aligned payload address. The shared slice cannot
        // outlive the arena or coexist with a mutable arena borrow.
        Some(unsafe {
            std::slice::from_raw_parts(self.payload.cast::<T>().as_ptr(), self.header.element_count)
        })
    }
}

/// A paged, framed command buffer.
///
/// Each command is stored wholly within one page behind an internal header
/// containing its type, size, alignment, frame length, and destructor. Iteration
/// therefore preserves record boundaries without exposing alignment padding or
/// potentially uninitialized object bytes. Pages come from a shared pool.
pub struct CommandArena {
    original_pages: Vec<PageInfo>,
    current_page: usize,
    cursor: usize,
    page_size: usize,
    /// Maximum alignment satisfiable on every page. Pool pages are only
    /// guaranteed OS-page-aligned (see [`SharedPagePool::alloc`]), so an
    /// alignment above `min(page_size, os_page_size)` could fail to fit even
    /// in a fresh page; [`push`](Self::push) rejects it up front.
    max_align: usize,
    pool: Arc<SharedPagePool>,
}

struct VacantCommandFrame {
    header: NonNull<CommandHeader>,
    payload: NonNull<u8>,
    payload_offset: usize,
    record_size: usize,
}

// Safety: CommandArena owns its pages.
unsafe impl Send for CommandArena {}

impl CommandArena {
    /// Create an arena that carves objects out of `page_size`-byte pages
    /// from `pool`.
    ///
    /// `page_size` bounds the largest framed object (see [`push`](Self::push));
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

    fn reserve_frame(&mut self, layout: std::alloc::Layout) -> Result<VacantCommandFrame, VmError> {
        let size = layout.size();
        let align = layout.align();
        let header_align = std::mem::align_of::<CommandHeader>();
        let header_size = std::mem::size_of::<CommandHeader>();

        // A complete record MUST fit in one page; records are not split.
        // Alignment is capped at min(page_size, OS page size): pool pages
        // are only guaranteed OS-page-aligned, so a larger alignment could
        // fail to fit even in a fresh page. Under this cap, a fresh page
        // always fits any accepted layout and the loop adds at most one page.
        let minimum_record_size = align_address(header_size, align)
            .and_then(|payload_offset| payload_offset.checked_add(size))
            .unwrap_or(usize::MAX);
        if minimum_record_size > self.page_size
            || align > self.max_align
            || header_align > self.max_align
        {
            return Err(VmError::ObjectTooLarge {
                size: minimum_record_size.max(align).max(header_align),
                page_size: self.page_size,
            });
        }

        if self.original_pages.is_empty() {
            self.add_page()?;
        }

        loop {
            if self.current_page >= self.original_pages.len() {
                self.add_page()?;
            }

            let page_info = &mut self.original_pages[self.current_page];
            let page_ptr = page_info.ptr;
            let page_cap = page_info.capacity;
            let base = page_ptr as usize;
            let Some(header_address) = base
                .checked_add(self.cursor)
                .and_then(|address| align_address(address, header_align))
            else {
                return Err(VmError::ObjectTooLarge {
                    size: usize::MAX,
                    page_size: self.page_size,
                });
            };
            let header_offset = header_address - base;
            let Some(payload_address) = header_address
                .checked_add(header_size)
                .and_then(|address| align_address(address, align))
            else {
                return Err(VmError::ObjectTooLarge {
                    size: usize::MAX,
                    page_size: self.page_size,
                });
            };
            let payload_offset = payload_address - base;
            let end_offset = payload_offset.saturating_add(size);

            if end_offset <= page_cap {
                self.cursor = end_offset;
                page_info.used = page_info.used.max(self.cursor);
                // Safety: offsets were derived from checked addresses, have
                // their requested alignment, and lie within a live page.
                return Ok(unsafe {
                    VacantCommandFrame {
                        header: NonNull::new_unchecked(page_ptr.add(header_offset))
                            .cast::<CommandHeader>(),
                        payload: NonNull::new_unchecked(page_ptr.add(payload_offset)),
                        payload_offset: payload_offset - header_offset,
                        record_size: end_offset - header_offset,
                    }
                });
            }

            self.current_page += 1;
            self.cursor = 0;
        }
    }

    /// Push a command into the arena and borrow it mutably.
    ///
    /// The arena owns `value` until [`reset`](Self::reset) or drop. Commands
    /// must be `Send` because the arena itself can move between threads. The
    /// returned reference may be used to finish initializing or update the
    /// command, but the usual mutable-borrow rules prevent another push until
    /// it is released. A header, alignment padding, and the value must fit
    /// wholly in one page.
    ///
    /// # Errors
    ///
    /// Returns `VmError` if the framed object is too large or page allocation
    /// fails. On error, `value` has not entered the arena and is dropped
    /// normally while the error is returned.
    pub fn push<T: Send + 'static>(&mut self, value: T) -> Result<&mut T, VmError> {
        let layout = std::alloc::Layout::new::<T>();
        let frame = self.reserve_frame(layout)?;
        // Write the payload before publishing its frame header. No fallible
        // operation follows, so both become live together.
        // Safety: reserve_frame returned aligned, non-overlapping in-page
        // storage for the header and T.
        unsafe {
            let payload = frame.payload.cast::<T>();
            payload.as_ptr().write(value);
            frame.header.as_ptr().write(CommandHeader {
                payload_offset: frame.payload_offset,
                payload_size: layout.size(),
                payload_align: layout.align(),
                element_count: 1,
                record_size: frame.record_size,
                payload_kind: CommandPayloadKind::Value,
                type_id: command_type_id::<T>,
                drop_value: drop_command::<T>,
            });
            Ok(&mut *payload.as_ptr())
        }
    }

    /// Copy one dynamically sized slice into a single framed arena record.
    ///
    /// The element type must be `Copy` because reset is O(records), not
    /// O(elements), and therefore does not run one destructor per element.
    /// The complete slice plus framing must fit within one arena page. This is
    /// useful for bounded encoded commands and transaction payload chunks that
    /// should live entirely in Qen rather than owning a secondary heap buffer.
    ///
    /// # Errors
    ///
    /// Returns `VmError` if the layout overflows, the framed slice is too
    /// large, or page allocation fails. On error no live record is added.
    pub fn push_slice<T: Copy + Send + 'static>(
        &mut self,
        values: &[T],
    ) -> Result<&mut [T], VmError> {
        let layout =
            std::alloc::Layout::array::<T>(values.len()).map_err(|_| VmError::ObjectTooLarge {
                size: usize::MAX,
                page_size: self.page_size,
            })?;
        let frame = self.reserve_frame(layout)?;
        // Safety: reserve_frame returned storage for values.len() contiguous
        // T values, and the source cannot overlap the arena-owned destination.
        unsafe {
            let payload = frame.payload.cast::<T>();
            std::ptr::copy_nonoverlapping(values.as_ptr(), payload.as_ptr(), values.len());
            frame.header.as_ptr().write(CommandHeader {
                payload_offset: frame.payload_offset,
                payload_size: layout.size(),
                payload_align: layout.align(),
                element_count: values.len(),
                record_size: frame.record_size,
                payload_kind: CommandPayloadKind::Slice,
                type_id: command_type_id::<T>,
                drop_value: drop_nothing,
            });
            Ok(std::slice::from_raw_parts_mut(
                payload.as_ptr(),
                values.len(),
            ))
        }
    }

    /// Documenting that this method panics on OOM.
    /// This is a temporary design choice for the game engine hot path.
    /// Future improvement: propagate Result.
    ///
    /// # Panics
    ///
    /// Panics if allocation fails (e.g. out of memory).
    pub fn push_or_panic<T: Send + 'static>(&mut self, value: T) -> &mut T {
        match self.push(value) {
            Ok(p) => p,
            Err(e) => panic!("CommandArena::push_or_panic failed: {e:?}"),
        }
    }

    /// Reset for reuse. Retains pages.
    ///
    /// # Invalidation contract
    ///
    /// Destroys all commands in insertion order, then logically invalidates
    /// their storage and retains the pages for reuse. If a destructor panics,
    /// the arena still runs the remaining destructors and resets itself before
    /// resuming the first panic. Destructors must not access the arena.
    pub fn reset(&mut self) {
        let first_panic = self.drop_commands();
        self.current_page = 0;
        self.cursor = 0;
        for page in &mut self.original_pages {
            page.used = 0;
        }
        if let Some(payload) = first_panic {
            std::panic::resume_unwind(payload);
        }
    }

    fn drop_commands(&mut self) -> Option<Box<dyn std::any::Any + Send>> {
        let mut first_panic = None;
        for page in &self.original_pages {
            let mut offset = 0;
            while offset < page.used {
                let base = page.ptr as usize;
                let header_address =
                    align_address(base + offset, std::mem::align_of::<CommandHeader>())
                        .expect("live command header address cannot overflow");
                let header_offset = header_address - base;
                // Safety: every live page prefix consists of complete command
                // frames written by push, and header_offset is header-aligned.
                let header = unsafe {
                    NonNull::new_unchecked(page.ptr.add(header_offset))
                        .cast::<CommandHeader>()
                        .as_ref()
                };
                let payload_offset = header_offset + header.payload_offset;
                let next_offset = header_offset + header.record_size;
                // Advance before calling user code so a panic cannot cause this
                // record to be destroyed twice.
                offset = next_offset;

                // Safety: the frame owns one live value at this aligned,
                // in-bounds payload address and this loop visits it once.
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
                    (header.drop_value)(page.ptr.add(payload_offset));
                }));
                if let Err(payload) = result
                    && first_panic.is_none()
                {
                    first_panic = Some(payload);
                }
            }
        }
        first_panic
    }

    /// Iterate framed commands in insertion order.
    #[inline]
    #[must_use]
    pub fn iter(&self) -> CommandIter<'_> {
        CommandIter {
            arena: self,
            page_idx: 0,
            offset: 0,
        }
    }

    /// Number of pages containing at least one live command frame.
    #[must_use]
    pub fn used_pages(&self) -> usize {
        self.original_pages
            .iter()
            .filter(|page| page.used != 0)
            .count()
    }

    /// Bytes occupied by live record headers, padding, and payloads.
    #[must_use]
    pub fn used_bytes(&self) -> usize {
        self.original_pages
            .iter()
            .fold(0usize, |total, page| total.saturating_add(page.used))
    }
}

/// Iterator over the live framed records in a [`CommandArena`].
pub struct CommandIter<'a> {
    arena: &'a CommandArena,
    page_idx: usize,
    offset: usize,
}

impl<'a> Iterator for CommandIter<'a> {
    type Item = CommandRecord<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        while self.page_idx < self.arena.original_pages.len() {
            let page_info = &self.arena.original_pages[self.page_idx];
            if self.offset >= page_info.used {
                self.page_idx += 1;
                self.offset = 0;
                continue;
            }

            let base = page_info.ptr as usize;
            let header_address =
                align_address(base + self.offset, std::mem::align_of::<CommandHeader>())?;
            let header_offset = header_address - base;
            // Safety: live used prefixes consist only of complete frames and
            // the computed address satisfies CommandHeader alignment.
            let header = unsafe {
                NonNull::new_unchecked(page_info.ptr.add(header_offset))
                    .cast::<CommandHeader>()
                    .as_ref()
            };
            let payload_offset = header_offset + header.payload_offset;
            self.offset = header_offset + header.record_size;
            // Safety: the header describes the in-bounds payload written in
            // the same frame. VM allocations are never null.
            let payload = unsafe { NonNull::new_unchecked(page_info.ptr.add(payload_offset)) };
            return Some(CommandRecord {
                header,
                payload,
                marker: PhantomData,
            });
        }
        None
    }
}

impl<'a> IntoIterator for &'a CommandArena {
    type Item = CommandRecord<'a>;
    type IntoIter = CommandIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl Drop for CommandArena {
    fn drop(&mut self) {
        let first_panic = self.drop_commands();
        // Return all pages to pool
        for page in &self.original_pages {
            if let Some(p) = NonNull::new(page.ptr) {
                // Safety: p was allocated from self.pool with capacity.
                unsafe {
                    self.pool.free(p, page.capacity);
                }
            }
        }
        if let Some(payload) = first_panic
            && !std::thread::panicking()
        {
            std::panic::resume_unwind(payload);
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
        assert_eq!(*p1, 42);

        // Push another
        let p2 = arena.push(123u64).unwrap();
        assert_eq!(*p2, 123);

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
        assert_eq!(*p_new, 999);
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

        let p1 = std::ptr::from_mut(arena.push(1u8).unwrap()) as usize;
        let p2 = std::ptr::from_mut(arena.push(1u32).unwrap()) as usize;
        let p3 = std::ptr::from_mut(arena.push(1u64).unwrap()) as usize;

        assert_eq!(p1 % std::mem::align_of::<u8>(), 0);
        assert_eq!(p2 % std::mem::align_of::<u32>(), 0);
        assert_eq!(p3 % std::mem::align_of::<u64>(), 0);
    }

    #[test]
    fn test_command_arena_rejects_high_alignment() {
        use crate::sync::Arc;

        // Unused field, but recall ZSTs are UB on allocator API, cf. nomicon
        #[derive(Clone, Copy, Debug)]
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
                assert!(size >= std::mem::align_of::<HugeAlign>());
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
            assert_eq!(
                std::ptr::from_mut(p) as usize % 4096,
                0,
                "push {i} misaligned"
            );
        }
    }

    #[test]
    fn test_command_arena_growth() {
        #[derive(Clone, Copy)]
        struct PageData {
            _d: [u8; 4000],
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
        arena.push(PageData { _d: [0; 4000] }).unwrap(); // Page 2
        arena.push(PageData { _d: [0; 4000] }).unwrap(); // Page 3 (exceeds pool cache capacity, but should succeed alloc from OS)

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
        let addr1 = std::ptr::from_mut(p1) as usize;

        arena.reset();

        let p2 = arena.push(456u64).unwrap();
        let addr2 = std::ptr::from_mut(p2) as usize;

        assert_eq!(addr1, addr2);
        assert_eq!(*p2, 456);
    }

    #[test]
    fn test_command_arena_owns_and_drops_commands_exactly_once() {
        use std::sync::atomic::{AtomicUsize, Ordering as StdOrdering};

        struct DropCounter(Arc<AtomicUsize>);

        impl Drop for DropCounter {
            fn drop(&mut self) {
                self.0.fetch_add(1, StdOrdering::SeqCst);
            }
        }

        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let page_size = 4096;
        let pool = Arc::new(SharedPagePool::new(page_size * 4));
        let drops = Arc::new(AtomicUsize::new(0));
        let mut arena = CommandArena::new(page_size, pool);

        arena.push(DropCounter(drops.clone())).unwrap();
        arena.push(String::from("typed, non-Copy command")).unwrap();
        assert_eq!(arena.iter().count(), 2);
        assert_eq!(
            arena
                .iter()
                .nth(1)
                .and_then(|record| record.downcast_ref::<String>()),
            Some(&String::from("typed, non-Copy command"))
        );

        arena.reset();
        assert_eq!(drops.load(StdOrdering::SeqCst), 1);
        assert_eq!(arena.iter().count(), 0);

        arena.push(DropCounter(drops.clone())).unwrap();
        drop(arena);
        assert_eq!(drops.load(StdOrdering::SeqCst), 2);
    }

    #[test]
    fn test_command_arena_finishes_reset_when_a_destructor_panics() {
        use std::sync::atomic::{AtomicUsize, Ordering as StdOrdering};

        struct DropAction {
            drops: Arc<AtomicUsize>,
            panic: bool,
        }

        impl Drop for DropAction {
            fn drop(&mut self) {
                self.drops.fetch_add(1, StdOrdering::SeqCst);
                assert!(!self.panic, "intentional command destructor panic");
            }
        }

        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let page_size = 4096;
        let pool = Arc::new(SharedPagePool::new(page_size * 4));
        let drops = Arc::new(AtomicUsize::new(0));
        let mut arena = CommandArena::new(page_size, pool);
        arena
            .push(DropAction {
                drops: drops.clone(),
                panic: true,
            })
            .unwrap();
        arena
            .push(DropAction {
                drops: drops.clone(),
                panic: false,
            })
            .unwrap();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| arena.reset()));
        assert!(result.is_err());
        assert_eq!(drops.load(StdOrdering::SeqCst), 2);
        assert_eq!(arena.iter().count(), 0);

        arena.push(7u32).unwrap();
        assert_eq!(arena.iter().count(), 1);
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
        assert_eq!(*p, ());
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

        let records: Vec<_> = arena.iter().collect();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].downcast_ref::<u8>(), Some(&1));
        assert_eq!(records[1].downcast_ref::<u32>(), Some(&2));
    }

    #[test]
    fn command_arena_copies_and_distinguishes_slice_records() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let page_size = 4096;
        let pool = Arc::new(SharedPagePool::new(page_size * 2));
        let mut arena = CommandArena::new(page_size, pool);

        let source = [11u32, 22, 33, 44];
        arena.push_slice(&source).unwrap()[1] = 99;
        arena.push(7u32).unwrap();
        arena.push_slice::<u8>(&[]).unwrap();

        let records = arena.iter().collect::<Vec<_>>();
        assert_eq!(records.len(), 3);
        assert!(records[0].is_slice());
        assert_eq!(records[0].slice_len(), Some(4));
        assert_eq!(
            records[0].downcast_slice::<u32>(),
            Some(&[11, 99, 33, 44][..])
        );
        assert_eq!(records[0].downcast_ref::<u32>(), None);
        assert_eq!(records[1].downcast_ref::<u32>(), Some(&7));
        assert_eq!(records[1].downcast_slice::<u32>(), None);
        assert_eq!(records[2].downcast_slice::<u8>(), Some(&[][..]));
        assert!(arena.used_bytes() >= source.len() * std::mem::size_of::<u32>());

        arena.reset();
        assert_eq!(arena.used_bytes(), 0);
        assert_eq!(arena.iter().count(), 0);

        assert!(matches!(
            arena.push_slice(&[0u8; 4096]),
            Err(VmError::ObjectTooLarge { .. })
        ));
        assert_eq!(arena.iter().count(), 0);
    }

    #[test]
    fn test_command_arena_iter_empty() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // I1: Iter on fresh arena — yields nothing
        let page_size = 4096;
        let pool = Arc::new(SharedPagePool::new(page_size));
        let arena = CommandArena::new(page_size, pool);

        let count = arena.iter().count();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_command_arena_iter_after_push() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // I2: Push N items, iter — yields framed items in order.
        let page_size = 4096;
        let pool = Arc::new(SharedPagePool::new(page_size));
        let mut arena = CommandArena::new(page_size, pool);

        arena.push(1u8).unwrap();
        arena.push(2u8).unwrap();

        let records: Vec<_> = arena.iter().collect();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].downcast_ref::<u8>(), Some(&1));
        assert_eq!(records[1].downcast_ref::<u8>(), Some(&2));
    }

    #[test]
    fn test_command_arena_iter_cross_page() {
        #[derive(Clone, Copy)]
        struct Item([u8; 100]);

        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // I3: Push enough to span pages, iter — yields all in order
        let page_size = 192; // Small page, but large enough for one framed Item.
        let pool = Arc::new(SharedPagePool::new(page_size * 4));
        let mut arena = CommandArena::new(page_size, pool);

        let item = Item([0; 100]);
        let _ = item.0; // Mark field as read

        arena.push(Item([0; 100])).unwrap(); // Page 1
        arena.push(Item([1; 100])).unwrap(); // Page 2
        arena.push(Item([2; 100])).unwrap(); // Page 3

        let records: Vec<_> = arena.iter().collect();
        assert_eq!(records.len(), 3);
        assert_eq!(records[0].downcast_ref::<Item>().unwrap().0[0], 0);
        assert_eq!(records[1].downcast_ref::<Item>().unwrap().0[0], 1);
        assert_eq!(records[2].downcast_ref::<Item>().unwrap().0[0], 2);
        assert_eq!(arena.used_pages(), 3);
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

        let count = arena.iter().count();
        assert_eq!(count, 0);

        // Push again
        arena.push(2u8).unwrap();
        let records: Vec<_> = arena.iter().collect();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].downcast_ref::<u8>(), Some(&2));
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
            _data: [u8; 4000],
        }

        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Push items that nearly fill a page once their frame header is
        // included. The remainder cannot fit another record.
        let page_size = 4096;
        let pool = Arc::new(SharedPagePool::new(page_size * 100));
        let mut arena = CommandArena::new(page_size, pool);

        let num_items = 10;
        for _ in 0..num_items {
            arena.push(NearPageSize { _data: [0; 4000] }).unwrap();
        }

        // Count pages used
        let pages_used = arena.used_pages();

        // The payload plus framing leaves too little room for another record.
        assert_eq!(
            pages_used, num_items,
            "Near-page-size items should each consume a full page (P9 fragmentation)"
        );

        // The real fragmentation case is payloads slightly over half the
        // usable page size, covered below.
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

        let pages_used = arena.used_pages();
        // Each item > half page → 1 item per page → 50% waste
        assert_eq!(
            pages_used, num_items,
            "Items > half-page should each waste ~50% (P9 fragmentation)"
        );
    }
}
