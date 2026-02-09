use std::fmt;
use std::ptr::NonNull;

/// Huge page size constants.
#[allow(dead_code)]
const PAGE_SIZE_2MB: usize = 2 * 1024 * 1024;
#[allow(dead_code)]
const PAGE_SIZE_1GB: usize = 1024 * 1024 * 1024;

#[derive(Debug)]
pub enum VmError {
    ReservationFailed(std::io::Error),
    CommitFailed(std::io::Error),
    DecommitFailed(std::io::Error),
    ReleaseFailed(std::io::Error),
    InitializationFailed(String),
    ObjectTooLarge { size: usize, page_size: usize },
}

impl fmt::Display for VmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VmError::ReservationFailed(e) => write!(f, "VM reservation failed: {e}"),
            VmError::CommitFailed(e) => write!(f, "VM commit failed: {e}"),
            VmError::DecommitFailed(e) => write!(f, "VM decommit failed: {e}"),
            VmError::ReleaseFailed(e) => write!(f, "VM release failed: {e}"),
            VmError::InitializationFailed(msg) => write!(f, "VM initialization failed: {msg}"),
            VmError::ObjectTooLarge { size, page_size } => write!(
                f,
                "Object too large for page: size {size} exceeds page size {page_size}"
            ),
        }
    }
}

impl std::error::Error for VmError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            VmError::ReservationFailed(e) | VmError::CommitFailed(e) | VmError::DecommitFailed(e) | VmError::ReleaseFailed(e) => Some(e),
            VmError::InitializationFailed(_) | VmError::ObjectTooLarge { .. } => None,
        }
    }
}

/// Abstract interface for virtual memory operations.
pub(crate) trait VmOps {
    /// Reserve address space without committing physical pages.
    /// Returns a pointer to the start of the reserved range.
    unsafe fn reserve(size: usize) -> Result<NonNull<u8>, VmError>;

    /// Commit (back with physical pages) a range within a reservation.
    unsafe fn commit(ptr: NonNull<u8>, size: usize) -> Result<(), VmError>;

    /// Decommit (return physical pages, keep address range reserved).
    unsafe fn decommit(ptr: NonNull<u8>, size: usize) -> Result<(), VmError>;

    /// Release address space entirely (after which pointers are invalid).
    unsafe fn release(ptr: NonNull<u8>, size: usize) -> Result<(), VmError>;

    /// OS page size (default/minimum).
    fn page_size() -> usize;

    /// Returns a list of supported page sizes (e.g. [4096, 2097152]).
    fn supported_page_sizes() -> Vec<usize>;

    /// Allocate memory backed by explicit huge pages (reserve + commit).
    ///
    /// Unlike regular [`reserve`] + [`commit`], huge page allocations are
    /// physically backed immediately. The entire region is committed at
    /// allocation time and cannot be partially decommitted.
    ///
    /// # Arguments
    /// * `size` — Must be a non-zero multiple of `huge_page_size`.
    /// * `huge_page_size` — Requested page granularity:
    ///   [`PAGE_SIZE_2MB`] or [`PAGE_SIZE_1GB`].
    ///
    /// # Platform Notes
    /// - **Linux**: `MAP_HUGETLB | MAP_HUGE_2MB/1GB`. Requires pre-allocated
    ///   hugetlb pages (2MB: `/proc/sys/vm/nr_hugepages`; 1GB: boot-time
    ///   kernel param `hugepagesz=1G hugepages=N`).
    /// - **macOS `x86_64`**: XNU superpages (2MB only). No 1GB support.
    /// - **macOS `aarch64`**: Not supported. Apple Silicon has no superpage
    ///   mechanism; any attempt returns `KERN_INVALID_ARGUMENT`.
    /// - **Windows**: `MEM_LARGE_PAGES`. Requires `SeLockMemoryPrivilege`.
    ///   Typically 2MB (`GetLargePageMinimum()`); 1GB not available.
    ///
    /// Free with [`release`] (same as regular allocations).
    unsafe fn alloc_huge(size: usize, huge_page_size: usize) -> Result<NonNull<u8>, VmError>;
}

pub(crate) struct PlatformVmOps;

#[cfg(all(any(target_os = "macos", target_os = "linux"), not(any(loom, miri))))]
mod unix {
    use super::{NonNull, VmError, PlatformVmOps, VmOps};
    use libc;
    use std::io;

    // ----------------------------------------------------------------
    // Huge page allocation — platform-specific helpers
    // ----------------------------------------------------------------

    /// Linux: MAP_HUGETLB with the page-size encoded in the upper bits of flags.
    /// Requires pre-allocated hugetlb pages:
    ///   2MB:  echo N > /proc/sys/vm/nr_hugepages
    ///   1GB:  boot param `hugepagesz=1G hugepages=N` (boot-time only)
    #[cfg(target_os = "linux")]
    unsafe fn alloc_huge_impl(size: usize, huge_page_size: usize) -> Result<NonNull<u8>, VmError> {
        // MAP_HUGE_SHIFT is 26; the log₂ of the page size goes in bits [31:26].
        const MAP_HUGE_SHIFT: libc::c_int = 26;
        const MAP_HUGE_2MB: libc::c_int = 21 << MAP_HUGE_SHIFT;
        const MAP_HUGE_1GB: libc::c_int = 30 << MAP_HUGE_SHIFT;

        let huge_flag = match huge_page_size {
            super::PAGE_SIZE_2MB => libc::MAP_HUGETLB | MAP_HUGE_2MB,
            super::PAGE_SIZE_1GB => libc::MAP_HUGETLB | MAP_HUGE_1GB,
            _ => {
                return Err(VmError::ReservationFailed(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "alloc_huge: unsupported huge page size {} on Linux \
                     (supported: 2MB = {}, 1GB = {})",
                        huge_page_size,
                        super::PAGE_SIZE_2MB,
                        super::PAGE_SIZE_1GB,
                    ),
                )));
            }
        };

        // Safety: FFI call to mmap.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON | huge_flag,
                -1,
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err(VmError::ReservationFailed(io::Error::last_os_error()));
        }

        NonNull::new(ptr as *mut u8).ok_or_else(|| {
            VmError::ReservationFailed(io::Error::new(
                io::ErrorKind::Other,
                "mmap returned null for huge page allocation",
            ))
        })
    }

    /// macOS Intel (x86_64): XNU superpages via mmap flag.
    ///
    /// The superpage size is encoded in the upper 16 bits of the `flags`
    /// argument when `MAP_ANON` is set. XNU's `kern_mman.c` extracts
    /// `flags & 0xFFFF0000` as `vm_alloc_flags`.
    ///
    /// `VM_FLAGS_SUPERPAGE_SIZE_2MB` (1) << `VM_FLAGS_SUPERPAGE_SHIFT` (16)
    /// = 0x10000.
    ///
    /// Only 2MB superpages are available on macOS; no 1GB support.
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    unsafe fn alloc_huge_impl(size: usize, huge_page_size: usize) -> Result<NonNull<u8>, VmError> {
        const SUPERPAGE_2MB: libc::c_int = 1 << 16;

        debug_assert!(
            huge_page_size == super::PAGE_SIZE_2MB,
            "macOS x86_64 only supports 2MB superpages, requested {}",
            huge_page_size
        );

        // Safety: FFI call to mmap.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON | SUPERPAGE_2MB,
                -1,
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err(VmError::ReservationFailed(io::Error::last_os_error()));
        }

        NonNull::new(ptr as *mut u8).ok_or_else(|| {
            VmError::ReservationFailed(io::Error::new(
                io::ErrorKind::Other,
                "mmap returned null for superpage allocation",
            ))
        })
    }

    /// Apple Silicon (aarch64): no superpage support whatsoever.
    /// The hardware only supports 16KB pages; any superpage flag to mmap
    /// results in `KERN_INVALID_ARGUMENT` from the Mach VM layer.
    #[cfg(all(target_os = "macos", not(target_arch = "x86_64")))]
    unsafe fn alloc_huge_impl(
        _size: usize,
        _huge_page_size: usize,
    ) -> Result<NonNull<u8>, VmError> {
        Err(VmError::ReservationFailed(io::Error::new(
            io::ErrorKind::Unsupported,
            "Apple Silicon does not support superpages (only 16KB pages); \
             attempting superpage flags returns KERN_INVALID_ARGUMENT",
        )))
    }

    // ----------------------------------------------------------------
    // Page size probing — platform-specific helpers
    // ----------------------------------------------------------------

    /// Linux: probe /sys/kernel/mm/hugepages/ for kernel-supported huge page
    /// sizes. Directory names are "hugepages-NkB" where N is the size in KiB.
    ///
    /// This reports what sizes the kernel *supports*, not what's currently
    /// allocated; `alloc_huge` may still fail if `nr_hugepages` is 0.
    #[cfg(target_os = "linux")]
    fn probe_supported_page_sizes() -> Vec<usize> {
        let base = PlatformVmOps::page_size();
        let mut sizes = vec![base];

        if let Ok(entries) = std::fs::read_dir("/sys/kernel/mm/hugepages") {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if let Some(kb_str) = name
                    .strip_prefix("hugepages-")
                    .and_then(|s| s.strip_suffix("kB"))
                {
                    if let Ok(kb) = kb_str.parse::<usize>() {
                        sizes.push(kb * 1024);
                    }
                }
            }
        }

        sizes.sort();
        sizes.dedup();
        sizes
    }

    /// macOS: Intel supports 2MB superpages; Apple Silicon has only 16KB pages.
    #[cfg(target_os = "macos")]
    fn probe_supported_page_sizes() -> Vec<usize> {
        let base = PlatformVmOps::page_size();
        // Only Intel Macs support superpages (2MB). Apple Silicon (aarch64)
        // is limited to 16KB pages with no superpage mechanism.
        #[cfg(target_arch = "x86_64")]
        {
            vec![base, super::PAGE_SIZE_2MB]
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            vec![base]
        }
    }

    // ----------------------------------------------------------------

    impl VmOps for PlatformVmOps {
        unsafe fn reserve(size: usize) -> Result<NonNull<u8>, VmError> {
            // Safety: FFI call to mmap.
            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    size,
                    libc::PROT_NONE,
                    libc::MAP_PRIVATE | libc::MAP_ANON,
                    -1,
                    0,
                )
            };

            if ptr == libc::MAP_FAILED {
                return Err(VmError::ReservationFailed(io::Error::last_os_error()));
            }

            match NonNull::new(ptr.cast::<u8>()) {
                Some(p) => Ok(p),
                None => Err(VmError::ReservationFailed(io::Error::other(
                    "mmap returned null",
                ))),
            }
        }

        unsafe fn commit(ptr: NonNull<u8>, size: usize) -> Result<(), VmError> {
            // Safety: FFI call to mprotect.
            if unsafe {
                libc::mprotect(
                    ptr.as_ptr().cast::<libc::c_void>(),
                    size,
                    libc::PROT_READ | libc::PROT_WRITE,
                )
            } != 0
            {
                return Err(VmError::CommitFailed(io::Error::last_os_error()));
            }

            #[cfg(target_os = "linux")]
            {
                // Transparent Huge Pages: ask the kernel to back this region
                // with 2MB pages when possible.
                // Only advise HUGEPAGE if the size is at least 2MB, otherwise
                // it's likely noise/overhead for the kernel.
                if size >= super::PAGE_SIZE_2MB {
                    // Safety: FFI call to madvise.
                    unsafe {
                        libc::madvise(ptr.as_ptr() as *mut libc::c_void, size, libc::MADV_HUGEPAGE)
                    };
                }
                // Safety: FFI call to madvise.
                unsafe {
                    // BinnedAllocator and ChunkPool commit memory in chunks
                    // largely when they're needed so we want immediate physical
                    // backing.  Avoid a bunch of minor page faults.
                    libc::madvise(ptr.as_ptr() as *mut libc::c_void, size, libc::MADV_WILLNEED)
                };
            }

            // NOTE: Zeroing is NOT done here. commit() may be called
            // speculatively outside a lock (e.g. BinnedAllocator pre-commit).
            // Callers that need zero-fill (debug assertions) must zero at the
            // allocator level, under their own lock, after confirming the
            // commit is integrated. See Pool::alloc() and integrate_precommit().

            Ok(())
        }

        unsafe fn decommit(ptr: NonNull<u8>, size: usize) -> Result<(), VmError> {
            // Unified path for macOS and Linux: MADV_FREE + mprotect(PROT_NONE).
            //
            // MADV_FREE marks pages for lazy reclamation — the cheapest decommit
            // on both platforms. The kernel reclaims physical pages when under
            // pressure; if it doesn't, old data may persist. No zeroing guarantee.
            //
            // mprotect(PROT_NONE) removes access. On recommit (mprotect RW), pages
            // may contain stale data (kernel kept them) or be zero-filled (kernel
            // reclaimed). We don't rely on either: debug assertions zeroes explicitly at
            // the allocator layer, release doesn't care.
            //
            // MADV_FREE: macOS (all versions), Linux >= 4.5 (March 2016).
            // Safety: FFI call to madvise.
            if unsafe { libc::madvise(ptr.as_ptr().cast::<libc::c_void>(), size, libc::MADV_FREE) }
                != 0
            {
                return Err(VmError::DecommitFailed(io::Error::last_os_error()));
            }
            // Safety: FFI call to mprotect.
            if unsafe { libc::mprotect(ptr.as_ptr().cast::<libc::c_void>(), size, libc::PROT_NONE) }
                != 0
            {
                return Err(VmError::DecommitFailed(io::Error::last_os_error()));
            }
            Ok(())
        }

        unsafe fn release(ptr: NonNull<u8>, size: usize) -> Result<(), VmError> {
            // Safety: FFI call to munmap.
            if unsafe { libc::munmap(ptr.as_ptr().cast::<libc::c_void>(), size) } != 0 {
                return Err(VmError::ReleaseFailed(io::Error::last_os_error()));
            }
            Ok(())
        }

        fn page_size() -> usize {
            use crate::sync::OnceLock;
            static CACHED: OnceLock<usize> = OnceLock::new();
            *CACHED.get_or_init(|| {
                // Safety: FFI call to sysconf.
                let raw = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
                assert!(
                    raw > 0,
                    "sysconf(_SC_PAGESIZE) failed: {}",
                    io::Error::last_os_error()
                );
                // SAFETY/PORTABILITY: this crate supports only 64-bit targets; page size fits in
                // usize there.
                #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                {
                    raw as usize
                }
            })
        }

        fn supported_page_sizes() -> Vec<usize> {
            use crate::sync::OnceLock;
            static CACHED: OnceLock<Vec<usize>> = OnceLock::new();
            CACHED.get_or_init(probe_supported_page_sizes).clone()
        }

        unsafe fn alloc_huge(size: usize, huge_page_size: usize) -> Result<NonNull<u8>, VmError> {
            debug_assert!(
                size != 0 && huge_page_size != 0 && size.is_multiple_of(huge_page_size),
                "alloc_huge: size ({size}) must be a non-zero multiple of huge_page_size ({huge_page_size})"
            );
            debug_assert!(
                huge_page_size.is_power_of_two(),
                "alloc_huge: huge_page_size ({huge_page_size}) must be a power of two"
            );

            // Safety: alloc_huge_impl is unsafe because it performs FFI.
            // We have verified preconditions above.
            unsafe { alloc_huge_impl(size, huge_page_size) }
        }
    }
}

#[cfg(all(target_os = "windows", not(any(loom, miri))))]
mod windows {
    use super::*;
    use libc;
    use std::io;

    /// `MEM_LARGE_PAGES` flag for VirtualAlloc.
    /// Allocates using large pages (typically 2MB on x86_64).
    /// Requires the process to hold `SeLockMemoryPrivilege`.
    const MEM_LARGE_PAGES: u32 = 0x20000000;

    extern "system" {
        /// Returns the minimum large page size supported by the system,
        /// or 0 if large pages are not supported.
        fn GetLargePageMinimum() -> usize;
    }

    impl VmOps for PlatformVmOps {
        unsafe fn reserve(size: usize) -> Result<NonNull<u8>, VmError> {
            // Safety: FFI call to VirtualAlloc.
            let ptr = unsafe {
                libc::VirtualAlloc(
                    std::ptr::null_mut(),
                    size,
                    libc::MEM_RESERVE,
                    libc::PAGE_NOACCESS,
                )
            };

            match NonNull::new(ptr as *mut u8) {
                Some(p) => Ok(p),
                None => Err(VmError::ReservationFailed(io::Error::last_os_error())),
            }
        }

        unsafe fn commit(ptr: NonNull<u8>, size: usize) -> Result<(), VmError> {
            // Safety: FFI call to VirtualAlloc.
            let result = unsafe {
                libc::VirtualAlloc(
                    ptr.as_ptr() as *mut libc::c_void,
                    size,
                    libc::MEM_COMMIT,
                    libc::PAGE_READWRITE,
                )
            };

            if result.is_null() {
                return Err(VmError::CommitFailed(io::Error::last_os_error()));
            }

            Ok(())
        }

        unsafe fn decommit(ptr: NonNull<u8>, size: usize) -> Result<(), VmError> {
            // Safety: FFI call to VirtualFree.
            if unsafe {
                libc::VirtualFree(ptr.as_ptr() as *mut libc::c_void, size, libc::MEM_DECOMMIT)
            } == 0
            {
                return Err(VmError::DecommitFailed(io::Error::last_os_error()));
            }

            Ok(())
        }

        unsafe fn release(ptr: NonNull<u8>, _size: usize) -> Result<(), VmError> {
            // Windows VirtualFree with MEM_RELEASE must have size 0 and the base address of the region.
            // Safety: FFI call to VirtualFree.
            if unsafe { libc::VirtualFree(ptr.as_ptr() as *mut libc::c_void, 0, libc::MEM_RELEASE) }
                == 0
            {
                return Err(VmError::ReleaseFailed(io::Error::last_os_error()));
            }
            Ok(())
        }

        fn page_size() -> usize {
            use crate::sync::OnceLock;
            static PAGE_SIZE: OnceLock<usize> = OnceLock::new();
            // Safety: FFI call to GetSystemInfo.
            *PAGE_SIZE.get_or_init(|| unsafe {
                let mut info: libc::SYSTEM_INFO = std::mem::zeroed();
                libc::GetSystemInfo(&mut info);
                info.dwPageSize as usize
            })
        }

        fn supported_page_sizes() -> Vec<usize> {
            use crate::sync::OnceLock;
            static CACHED: OnceLock<Vec<usize>> = OnceLock::new();
            CACHED
                .get_or_init(|| {
                    let base = Self::page_size();
                    let mut sizes = vec![base];
                    let large_page = unsafe { GetLargePageMinimum() };
                    if large_page > 0 && large_page != base {
                        sizes.push(large_page);
                    }
                    sizes.sort();
                    sizes.dedup();
                    sizes
                })
                .clone()
        }

        unsafe fn alloc_huge(size: usize, huge_page_size: usize) -> Result<NonNull<u8>, VmError> {
            debug_assert!(
                size != 0 && huge_page_size != 0 && size % huge_page_size == 0,
                "alloc_huge: size ({}) must be a non-zero multiple of huge_page_size ({})",
                size,
                huge_page_size
            );

            let system_large_page = unsafe { GetLargePageMinimum() };
            debug_assert!(
                system_large_page != 0,
                "large pages not available (GetLargePageMinimum returned 0); ensure SeLockMemoryPrivilege is granted"
            );
            // Windows only supports one large page size (returned by
            // GetLargePageMinimum). Typically 2MB on x86_64.
            debug_assert!(
                huge_page_size == system_large_page,
                "Windows large page size is {} bytes, requested {}",
                system_large_page,
                huge_page_size
            );

            // MEM_LARGE_PAGES must be combined with MEM_RESERVE | MEM_COMMIT.
            // The allocation is fully backed from the start (no partial commit).
            // Safety: FFI call to VirtualAlloc.
            let ptr = unsafe {
                libc::VirtualAlloc(
                    std::ptr::null_mut(),
                    size,
                    libc::MEM_RESERVE | libc::MEM_COMMIT | MEM_LARGE_PAGES,
                    libc::PAGE_READWRITE,
                )
            };

            match NonNull::new(ptr as *mut u8) {
                Some(p) => Ok(p),
                None => Err(VmError::ReservationFailed(io::Error::last_os_error())),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Loom mock: heap-backed VmOps (no real mmap/VirtualAlloc)
//
// Under `cfg(loom)` we cannot issue real VM syscalls — loom runs inside a
// single OS process with its own scheduler. Instead we back every "reservation"
// with a plain heap allocation (via `std::alloc::alloc` / `dealloc`).
//
// `commit` / `decommit` are intentional no-ops: the memory is always
// accessible once reserved.  `release` frees the heap block.
//
// This is sufficient for testing the *synchronization* logic of the allocators
// (loom) and detecting undefined behaviour in unsafe pointer code (Miri);
// actual page-fault and huge-page behaviour is tested by the real platform
// implementation in normal builds.
// ---------------------------------------------------------------------------
#[cfg(any(loom, miri))]
impl VmOps for PlatformVmOps {
    unsafe fn reserve(size: usize) -> Result<NonNull<u8>, VmError> {
        if size == 0 {
            return Err(VmError::ReservationFailed(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "zero-size reservation",
            )));
        }
        let layout = std::alloc::Layout::from_size_align(size, 4096)
            .map_err(|e| VmError::ReservationFailed(std::io::Error::other(e)))?;
        // Safety: layout has non-zero size.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        NonNull::new(ptr).ok_or_else(|| {
            VmError::ReservationFailed(std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                "alloc returned null",
            ))
        })
    }

    unsafe fn commit(_ptr: NonNull<u8>, _size: usize) -> Result<(), VmError> {
        Ok(()) // heap memory is always accessible
    }

    unsafe fn decommit(_ptr: NonNull<u8>, _size: usize) -> Result<(), VmError> {
        Ok(()) // no-op; memory remains accessible
    }

    unsafe fn release(ptr: NonNull<u8>, size: usize) -> Result<(), VmError> {
        let layout = std::alloc::Layout::from_size_align(size, 4096)
            .map_err(|e| VmError::ReleaseFailed(std::io::Error::other(e)))?;
        // Safety: ptr was allocated with the same layout via `reserve`.
        unsafe { std::alloc::dealloc(ptr.as_ptr(), layout) };
        Ok(())
    }

    fn page_size() -> usize {
        4096
    }

    fn supported_page_sizes() -> Vec<usize> {
        vec![4096]
    }

    unsafe fn alloc_huge(size: usize, _huge_page_size: usize) -> Result<NonNull<u8>, VmError> {
        // Under loom, just forward to reserve (no real huge pages).
        // Safety: caller guarantees size > 0 and alignment requirements.
        unsafe { Self::reserve(size) }
    }
}

#[cfg(all(test, not(any(loom, miri))))]
mod tests {
    use super::*;

    #[test]
    fn test_reserve_commit_release() {
        let size = PlatformVmOps::page_size();
        // Safety: Test code.
        unsafe {
            let ptr = PlatformVmOps::reserve(size).expect("Reserve failed");

            // Should fail to write if PROT_NONE
            // This would segfault in a normal run, preventing test completion.
            // We skip verifying segfaults in unit tests usually.

            PlatformVmOps::commit(ptr, size).expect("Commit failed");

            // Write to memory
            let slice = std::slice::from_raw_parts_mut(ptr.as_ptr(), size);
            slice[0] = 42;
            assert_eq!(slice[0], 42);

            PlatformVmOps::decommit(ptr, size).expect("Decommit failed");

            // Release
            PlatformVmOps::release(ptr, size).expect("Release failed");
        }
    }
    #[test]
    fn test_reserve_zero_size() {
        // V1: Verify behavior when reserving 0 bytes
        // mmap with 0 size usually fails with EINVAL.
        // We expect an error.
        // Safety: Test code.
        let result = unsafe { PlatformVmOps::reserve(0) };
        assert!(result.is_err(), "Reserving 0 bytes should fail");
    }

    #[test]
    fn test_commit_idempotent() {
        // V2: Commit same range twice — should succeed without error
        let size = PlatformVmOps::page_size();
        // Safety: Test code.
        unsafe {
            let ptr = PlatformVmOps::reserve(size).expect("Reserve failed");

            PlatformVmOps::commit(ptr, size).expect("First commit failed");

            // Second commit on same range
            PlatformVmOps::commit(ptr, size).expect("Second commit failed (idempotency check)");

            // Verify write
            *(ptr.as_ptr()) = 123;

            PlatformVmOps::release(ptr, size).expect("Release failed");
        }
    }

    #[test]
    fn test_decommit_then_recommit() {
        // V3: Full cycle: reserve → commit → write → decommit → recommit → write → release
        let size = PlatformVmOps::page_size();
        // Safety: Test code.
        unsafe {
            let ptr = PlatformVmOps::reserve(size).expect("Reserve failed");

            // 1. Commit & Write
            PlatformVmOps::commit(ptr, size).expect("Commit failed");
            *(ptr.as_ptr()) = 42;
            assert_eq!(*(ptr.as_ptr().cast_const()), 42);

            // 2. Decommit
            PlatformVmOps::decommit(ptr, size).expect("Decommit failed");

            // 3. Recommit
            PlatformVmOps::commit(ptr, size).expect("Recommit failed");

            // 4. Write again (memory content is undefined after decommit, so we just write new)
            *(ptr.as_ptr()) = 84;
            assert_eq!(*(ptr.as_ptr().cast_const()), 84);

            PlatformVmOps::release(ptr, size).expect("Release failed");
        }
    }

    #[test]
    fn test_partial_commit() {
        // V4: Reserve large range, commit only a sub-range
        let page_size = PlatformVmOps::page_size();
        let total_size = page_size * 4;
        let commit_size = page_size * 2;
        let offset = page_size;

        // Safety: Test code.
        unsafe {
            let ptr = PlatformVmOps::reserve(total_size).expect("Reserve failed");
            let commit_ptr = NonNull::new(ptr.as_ptr().add(offset)).unwrap();

            // Commit middle pages
            PlatformVmOps::commit(commit_ptr, commit_size).expect("Partial commit failed");

            // Write to committed region
            let slice = std::slice::from_raw_parts_mut(commit_ptr.as_ptr(), commit_size);
            slice[0] = 10;
            slice[commit_size - 1] = 20;

            assert_eq!(slice[0], 10);
            assert_eq!(slice[commit_size - 1], 20);

            // Clean up
            PlatformVmOps::release(ptr, total_size).expect("Release failed");
        }
    }

    #[test]
    fn test_release_then_access_is_invalid() {
        // V5: Document that released memory must not be accessed
        // We cannot safely test access (segfault), but we verify release API succeeds.
        let size = PlatformVmOps::page_size();
        // Safety: Test code.
        unsafe {
            let ptr = PlatformVmOps::reserve(size).expect("Reserve failed");
            PlatformVmOps::commit(ptr, size).expect("Commit failed");
            PlatformVmOps::release(ptr, size).expect("Release failed");
            // DO NOT ACCESS ptr here.
        }
    }

    #[test]
    fn test_page_size_is_power_of_two() {
        // V6: page_size() returns a power of 2
        let size = PlatformVmOps::page_size();
        assert!(size > 0);
        assert_eq!(
            size & (size - 1),
            0,
            "Page size {size} is not power of two"
        );
    }

    #[test]
    fn test_reserve_very_large() {
        // V8: Reserve a large range (e.g. 1GB) — verify succeeds
        // 1GB is usually fine on 64-bit systems.
        let size = 1024 * 1024 * 1024;
        // Safety: Test code.
        unsafe {
            let ptr = PlatformVmOps::reserve(size).expect("Failed to reserve 1GB");
            PlatformVmOps::release(ptr, size).expect("Release failed");
        }
    }

    #[test]
    fn test_commit_unaligned_size() {
        // V9: Commit a non-page-aligned size within a reservation
        let page_size = PlatformVmOps::page_size();
        let size = page_size * 2;
        // Safety: Test code.
        unsafe {
            let ptr = PlatformVmOps::reserve(size).expect("Reserve failed");

            // Commit 1.5 pages (should round up to 2 pages usually, or failure if strict API?)
            // Implementation uses mprotect. mprotect usually requires page-aligned start,
            // but length? man mprotect: "The length argument ... is rounded up to a multiple of the system page size" (Linux).
            // macOS? "len is the length of the region ... rounding up to the next page boundary".
            // So it should work and cover 2 pages.
            let unaligned_size = page_size + 1;
            PlatformVmOps::commit(ptr, unaligned_size).expect("Commit unaligned failed");

            // Verify access to second page
            let slice = std::slice::from_raw_parts_mut(ptr.as_ptr(), size);
            slice[page_size] = 42; // Should not segfault

            PlatformVmOps::release(ptr, size).expect("Release failed");
        }
    }

    #[test]
    fn test_multiple_reservations() {
        // V10: Multiple independent reserve/commit/release cycles — no interference
        let page_size = PlatformVmOps::page_size();
        // Safety: Test code.
        unsafe {
            let ptr1 = PlatformVmOps::reserve(page_size).expect("Reserve 1 failed");
            let ptr2 = PlatformVmOps::reserve(page_size).expect("Reserve 2 failed");

            assert_ne!(ptr1, ptr2);

            PlatformVmOps::commit(ptr1, page_size).expect("Commit 1 failed");
            PlatformVmOps::commit(ptr2, page_size).expect("Commit 2 failed");

            *(ptr1.as_ptr()) = 1;
            *(ptr2.as_ptr()) = 2;

            assert_eq!(*(ptr1.as_ptr()), 1);
            assert_eq!(*(ptr2.as_ptr()), 2);

            PlatformVmOps::release(ptr1, page_size).expect("Release 1 failed");

            // ptr2 should still be valid
            assert_eq!(*(ptr2.as_ptr()), 2);

            PlatformVmOps::release(ptr2, page_size).expect("Release 2 failed");
        }
    }

    #[test]
    fn test_decommit_recommit_accessible() {
        // Verify decommit + commit cycle produces accessible memory.
        // NOTE: The VM layer does NOT guarantee zero-fill. On macOS, MADV_FREE
        // may retain stale data after recommit. Zeroing is the allocator
        // layer's responsibility (Pool::alloc, integrate_precommit, etc.)
        // under the `debug assertions` feature.
        let size = PlatformVmOps::page_size();
        // Safety: Test code.
        unsafe {
            let ptr = PlatformVmOps::reserve(size).expect("Reserve failed");
            PlatformVmOps::commit(ptr, size).expect("Commit failed");

            // Write "dirty" data
            let slice = std::slice::from_raw_parts_mut(ptr.as_ptr(), size);
            for item in slice.iter_mut().take(size) {
                *item = 0xAA;
            }

            PlatformVmOps::decommit(ptr, size).expect("Decommit failed");
            PlatformVmOps::commit(ptr, size).expect("Recommit failed");

            // Pages must be accessible after recommit (read + write)
            let slice = std::slice::from_raw_parts_mut(ptr.as_ptr(), size);
            slice[0] = 0x42;
            assert_eq!(slice[0], 0x42, "Recommitted memory is not writable");

            PlatformVmOps::release(ptr, size).expect("Release failed");
        }
    }

    // ----------------------------------------------------------------
    // Huge page tests
    // ----------------------------------------------------------------

    #[test]
    fn test_supported_page_sizes_includes_base() {
        // Re-validate that supported_page_sizes always includes the base.
        let base = PlatformVmOps::page_size();
        let supported = PlatformVmOps::supported_page_sizes();
        assert!(
            supported.contains(&base),
            "supported_page_sizes {supported:?} must include base page size {base}"
        );
        // All sizes must be powers of two
        for &s in &supported {
            assert!(s.is_power_of_two(), "Page size {s} is not a power of two");
        }
        // Must be sorted ascending
        for w in supported.windows(2) {
            assert!(
                w[0] < w[1],
                "supported_page_sizes not sorted: {supported:?}"
            );
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "must be a non-zero multiple")]
    fn test_alloc_huge_bad_args_size_zero_panics() {
        // Safety: Test code.
        unsafe {
            drop(PlatformVmOps::alloc_huge(0, PAGE_SIZE_2MB));
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "must be a non-zero multiple")]
    fn test_alloc_huge_bad_args_huge_page_size_zero_panics() {
        // Safety: Test code.
        unsafe {
            drop(PlatformVmOps::alloc_huge(PAGE_SIZE_2MB, 0));
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "must be a non-zero multiple")]
    fn test_alloc_huge_bad_args_not_multiple_panics() {
        // Safety: Test code.
        unsafe {
            drop(PlatformVmOps::alloc_huge(PAGE_SIZE_2MB + 1, PAGE_SIZE_2MB));
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "must be a power of two")]
    fn test_alloc_huge_bad_args_non_power_of_two_panics() {
        // Safety: Test code.
        unsafe {
            drop(PlatformVmOps::alloc_huge(3 * 1024 * 1024, 3 * 1024 * 1024));
        }
    }

    #[test]
    fn test_alloc_huge_2mb() {
        // Attempt a 2MB huge page allocation. This may fail gracefully
        // if the system doesn't have huge pages configured (Apple Silicon,
        // hugetlb pool empty on Linux, no SeLockMemoryPrivilege on Windows).
        let size = PAGE_SIZE_2MB;
        // Safety: Test code.
        let result = unsafe { PlatformVmOps::alloc_huge(size, PAGE_SIZE_2MB) };

        // Apple Silicon: must fail (no superpage support)
        #[cfg(all(target_os = "macos", not(target_arch = "x86_64")))]
        {
            assert!(result.is_err(), "alloc_huge must fail on Apple Silicon");
        }

        // Other platforms: may succeed or fail depending on system config.
        #[cfg(not(all(target_os = "macos", not(target_arch = "x86_64"))))]
        match result {
            Ok(ptr) => unsafe {
                // Verify alignment
                assert_eq!(
                    ptr.as_ptr() as usize % PAGE_SIZE_2MB,
                    0,
                    "Huge page allocation not aligned to 2MB: {:p}",
                    ptr,
                );
                // Verify read/write
                let slice = std::slice::from_raw_parts_mut(ptr.as_ptr(), size);
                slice[0] = 0xDE;
                slice[size - 1] = 0xAD;
                assert_eq!(slice[0], 0xDE);
                assert_eq!(slice[size - 1], 0xAD);
                // Release
                PlatformVmOps::release(ptr, size).expect("release after alloc_huge failed");
            },
            Err(e) => {
                // Acceptable: system doesn't have huge pages configured.
                eprintln!("test_alloc_huge_2mb: not available on this system: {}", e);
            }
        }
    }

    #[test]
    fn test_alloc_huge_1gb() {
        // 1GB pages: only on Linux (boot-time config) and maybe Windows.
        // macOS doesn't support them at all.
        let size = PAGE_SIZE_1GB;
        // Safety: Test code.
        let result = unsafe { PlatformVmOps::alloc_huge(size, PAGE_SIZE_1GB) };

        // macOS: must fail (no 1GB support on any architecture)
        #[cfg(target_os = "macos")]
        {
            assert!(result.is_err(), "alloc_huge(1GB) must fail on macOS");
        }

        // Linux/Windows: almost certainly fails (1GB pages need boot-time
        // reservation on Linux, and Windows typically only has 2MB).
        // But if it succeeds, verify correctness.
        #[cfg(not(target_os = "macos"))]
        match result {
            Ok(ptr) => unsafe {
                assert_eq!(
                    ptr.as_ptr() as usize % PAGE_SIZE_1GB,
                    0,
                    "1GB allocation not aligned: {:p}",
                    ptr,
                );
                let slice = std::slice::from_raw_parts_mut(ptr.as_ptr(), size);
                slice[0] = 0xBE;
                slice[size - 1] = 0xEF;
                assert_eq!(slice[0], 0xBE);
                assert_eq!(slice[size - 1], 0xEF);
                PlatformVmOps::release(ptr, size).expect("release after alloc_huge(1GB) failed");
            },
            Err(e) => {
                eprintln!("test_alloc_huge_1gb: not available on this system: {}", e);
            }
        }
    }

    #[test]
    fn test_alloc_huge_multi_page() {
        // Allocate multiple huge pages at once (4MB = 2 × 2MB).
        let size = PAGE_SIZE_2MB * 2;
        // Safety: Test code.
        let result = unsafe { PlatformVmOps::alloc_huge(size, PAGE_SIZE_2MB) };

        #[cfg(all(target_os = "macos", not(target_arch = "x86_64")))]
        {
            assert!(result.is_err());
        }

        #[cfg(not(all(target_os = "macos", not(target_arch = "x86_64"))))]
        match result {
            Ok(ptr) => unsafe {
                assert_eq!(ptr.as_ptr() as usize % PAGE_SIZE_2MB, 0);
                // Write to both huge page boundaries
                *ptr.as_ptr() = 0x11;
                *ptr.as_ptr().add(PAGE_SIZE_2MB) = 0x22;
                assert_eq!(*ptr.as_ptr(), 0x11);
                assert_eq!(*ptr.as_ptr().add(PAGE_SIZE_2MB), 0x22);
                PlatformVmOps::release(ptr, size).expect("release multi-page failed");
            },
            Err(e) => {
                eprintln!("test_alloc_huge_multi_page: not available: {}", e);
            }
        }
    }

    #[test]
    fn test_alloc_huge_unsupported_size() {
        // A page size that no platform supports (e.g. 4MB = not a standard
        // huge page size). Should fail with InvalidInput.
        // Safety: Test code.
        let result = unsafe { PlatformVmOps::alloc_huge(4 * 1024 * 1024, 4 * 1024 * 1024) };

        // macOS aarch64: fails because no superpages at all
        // macOS x86_64: fails because only 2MB is supported
        // Linux: fails because MAP_HUGETLB doesn't support 4MB
        // Windows: fails because 4MB != GetLargePageMinimum()
        assert!(
            result.is_err(),
            "alloc_huge with 4MB page size should fail on all platforms",
        );
    }
}
