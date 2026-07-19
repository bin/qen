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
    ObjectTooLarge {
        size: usize,
        page_size: usize,
    },
    /// A fixed-size pool ran out of reserved address space. Distinct from
    /// `CommitFailed` (an OS syscall refusing physical pages): callers such
    /// as `PoolChain` respond to exhaustion by reusing or adding pools, but
    /// must propagate genuine commit failures.
    PoolExhausted,
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
            VmError::PoolExhausted => write!(f, "Pool reserved address space exhausted"),
        }
    }
}

impl std::error::Error for VmError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            VmError::ReservationFailed(e)
            | VmError::CommitFailed(e)
            | VmError::DecommitFailed(e)
            | VmError::ReleaseFailed(e) => Some(e),
            VmError::InitializationFailed(_)
            | VmError::ObjectTooLarge { .. }
            | VmError::PoolExhausted => None,
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

/// Result of an aligned reservation. Stores both the aligned user pointer
/// and the original base/total for correct release.
pub(crate) struct AlignedReservation {
    /// Aligned pointer for use by the caller.
    pub aligned: NonNull<u8>,
    /// Original mmap base (may differ from `aligned` due to slop).
    pub original_base: NonNull<u8>,
    /// Total reserved size (>= requested size due to over-reservation).
    pub total_reserved: usize,
}

/// Reserve `size` bytes of address space aligned to `align`.
///
/// Over-reserves `size + align - page_size`, then returns the aligned sub-range.
/// Caller must release using `original_base` and `total_reserved`, not `aligned`.
pub(crate) unsafe fn reserve_aligned<V: VmOps>(
    size: usize,
    align: usize,
) -> Result<AlignedReservation, VmError> {
    crate::qen_debug_assert!(size > 0);
    crate::qen_debug_assert!(align.is_power_of_two());
    crate::qen_debug_assert!(
        size.is_multiple_of(V::page_size()),
        "size must be page-aligned"
    );

    let page_size = V::page_size();

    // If align <= page_size, regular reserve is sufficient (mmap returns page-aligned)
    if align <= page_size {
        // Safety: FFI reserve; caller contract forwarded.
        let ptr = unsafe { V::reserve(size)? };
        return Ok(AlignedReservation {
            aligned: ptr,
            original_base: ptr,
            total_reserved: size,
        });
    }

    let total_reserve = size + align - page_size;
    // Safety: FFI reserve; caller contract forwarded.
    let base = unsafe { V::reserve(total_reserve)? };
    let base_addr = base.as_ptr() as usize;
    let aligned_addr = (base_addr + align - 1) & !(align - 1);
    // Safety: the alignment offset stays within the padded reservation;
    // deriving from base (not the bare address) preserves provenance.
    let aligned = unsafe { NonNull::new_unchecked(base.as_ptr().add(aligned_addr - base_addr)) };

    Ok(AlignedReservation {
        aligned,
        original_base: base,
        total_reserved: total_reserve,
    })
}

pub(crate) struct PlatformVmOps;

/// Test-only failure injection for VM operations.
///
/// Arms the next `n` calls of an operation to fail with an injected error,
/// letting tests exercise the error-handling paths (`Pool::trim` retry,
/// `LargeAllocCache` release accounting, alloc commit failures) that can
/// never be triggered on a healthy machine.
///
/// The counters are global, but injections fire only on the thread that
/// armed them: async cleanups on other threads (TLS cache drops at thread
/// death, allocator drops on detached test threads) run outside
/// `TEST_MUTEX` and must not steal an armed failure out from under the
/// arming test. Tests that arm failures MUST still hold
/// `TEST_MUTEX.write()` (gauge assertions need exclusivity), and must
/// disarm (drain) them before finishing.
///
/// Excluded under loom — loom explores interleavings, not fault injection,
/// and the extra atomics would pollute every modeled execution.
#[cfg(all(test, not(loom)))]
pub(crate) mod failure_injection {
    use super::VmError;
    use std::io;
    use std::sync::atomic::{AtomicU32, Ordering};

    pub static FAIL_RESERVE: AtomicU32 = AtomicU32::new(0);
    pub static FAIL_COMMIT: AtomicU32 = AtomicU32::new(0);
    pub static FAIL_DECOMMIT: AtomicU32 = AtomicU32::new(0);
    pub static FAIL_RELEASE: AtomicU32 = AtomicU32::new(0);

    /// The thread that armed the injection (see the module docs for why
    /// injections are thread-scoped).
    static ARMED_THREAD: std::sync::Mutex<Option<std::thread::ThreadId>> =
        std::sync::Mutex::new(None);

    fn arm(counter: &AtomicU32, n: u32) {
        *ARMED_THREAD
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(std::thread::current().id());
        counter.store(n, Ordering::SeqCst);
    }

    /// Arm the next `n` calls of the operation (on this thread) to fail.
    pub fn fail_next_reserves(n: u32) {
        arm(&FAIL_RESERVE, n);
    }
    pub fn fail_next_commits(n: u32) {
        arm(&FAIL_COMMIT, n);
    }
    pub fn fail_next_decommits(n: u32) {
        arm(&FAIL_DECOMMIT, n);
    }
    pub fn fail_next_releases(n: u32) {
        arm(&FAIL_RELEASE, n);
    }

    /// Disarm all injection counters (call before releasing `TEST_MUTEX`).
    pub fn reset() {
        FAIL_RESERVE.store(0, Ordering::SeqCst);
        FAIL_COMMIT.store(0, Ordering::SeqCst);
        FAIL_DECOMMIT.store(0, Ordering::SeqCst);
        FAIL_RELEASE.store(0, Ordering::SeqCst);
        *ARMED_THREAD
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
    }

    fn should_fail(counter: &AtomicU32) -> bool {
        // Cheap disarmed path first: only consult the armed-thread lock
        // when an injection is actually pending.
        if counter.load(Ordering::SeqCst) == 0 {
            return false;
        }
        let armed = *ARMED_THREAD
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if armed != Some(std::thread::current().id()) {
            return false;
        }
        counter
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| v.checked_sub(1))
            .is_ok()
    }

    fn injected(kind: &str) -> io::Error {
        io::Error::other(format!("injected {kind} failure"))
    }

    pub fn maybe_fail_reserve() -> Result<(), VmError> {
        if should_fail(&FAIL_RESERVE) {
            return Err(VmError::ReservationFailed(injected("reserve")));
        }
        Ok(())
    }
    pub fn maybe_fail_commit() -> Result<(), VmError> {
        if should_fail(&FAIL_COMMIT) {
            return Err(VmError::CommitFailed(injected("commit")));
        }
        Ok(())
    }
    pub fn maybe_fail_decommit() -> Result<(), VmError> {
        if should_fail(&FAIL_DECOMMIT) {
            return Err(VmError::DecommitFailed(injected("decommit")));
        }
        Ok(())
    }
    pub fn maybe_fail_release() -> Result<(), VmError> {
        if should_fail(&FAIL_RELEASE) {
            return Err(VmError::ReleaseFailed(injected("release")));
        }
        Ok(())
    }
}

#[cfg(all(any(target_os = "macos", target_os = "linux"), not(any(loom, miri))))]
mod unix {
    use super::{NonNull, PlatformVmOps, VmError, VmOps};
    use libc;
    use std::io;

    // ----------------------------------------------------------------
    // Huge page allocation — platform-specific helpers
    // ----------------------------------------------------------------

    /// Linux: `MAP_HUGETLB` with the page-size encoded in the upper bits of flags.
    /// Requires pre-allocated hugetlb pages:
    ///   2MB:  `echo N > /proc/sys/vm/nr_hugepages`
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

        NonNull::new(ptr.cast::<u8>()).ok_or_else(|| {
            VmError::ReservationFailed(io::Error::other(
                "mmap returned null for huge page allocation",
            ))
        })
    }

    /// macOS Intel (`x86_64`): XNU superpages via mmap flag.
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

        // Runtime check in all build modes (like the Linux path): a
        // debug_assert here would let release builds silently map 2MB
        // superpages for a caller that asked for a different size.
        if huge_page_size != super::PAGE_SIZE_2MB {
            return Err(VmError::ReservationFailed(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "alloc_huge: unsupported huge page size {} on macOS x86_64 \
                     (only 2MB = {} superpages)",
                    huge_page_size,
                    super::PAGE_SIZE_2MB,
                ),
            )));
        }

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

        NonNull::new(ptr.cast::<u8>()).ok_or_else(|| {
            VmError::ReservationFailed(io::Error::other(
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
                    && let Ok(kb) = kb_str.parse::<usize>()
                {
                    sizes.push(kb * 1024);
                }
            }
        }

        sizes.sort_unstable();
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
            #[cfg(test)]
            super::failure_injection::maybe_fail_reserve()?;

            // MAP_NORESERVE (Linux): a PROT_NONE reservation must not be
            // charged against overcommit accounting at mmap time — under
            // vm.overcommit_memory=2 a multi-GB sparse reservation would
            // otherwise fail with ENOMEM despite requesting no physical
            // pages. Commit charging happens at mprotect(RW) time instead.
            // macOS has no equivalent flag (and no strict-overcommit mode).
            #[cfg(target_os = "linux")]
            let flags = libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_NORESERVE;
            #[cfg(not(target_os = "linux"))]
            let flags = libc::MAP_PRIVATE | libc::MAP_ANON;

            // Safety: FFI call to mmap.
            let ptr =
                unsafe { libc::mmap(std::ptr::null_mut(), size, libc::PROT_NONE, flags, -1, 0) };

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
            #[cfg(test)]
            super::failure_injection::maybe_fail_commit()?;

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
                        libc::madvise(
                            ptr.as_ptr().cast::<libc::c_void>(),
                            size,
                            libc::MADV_HUGEPAGE,
                        )
                    };
                }
                // Safety: FFI call to madvise.
                unsafe {
                    // BinnedAllocator and ChunkPool commit memory in chunks
                    // largely when they're needed so we want immediate physical
                    // backing.  Avoid a bunch of minor page faults.
                    libc::madvise(
                        ptr.as_ptr().cast::<libc::c_void>(),
                        size,
                        libc::MADV_WILLNEED,
                    )
                };
            }

            // NOTE: Zeroing is NOT done here. commit() may be called
            // speculatively outside a lock (e.g. BinnedAllocator pre-commit).
            // Callers that need zero-fill (debug/hardened mode) must zero at the
            // allocator level, under their own lock, after confirming the
            // commit is integrated. See Pool::alloc() and integrate_precommit().

            Ok(())
        }

        unsafe fn decommit(ptr: NonNull<u8>, size: usize) -> Result<(), VmError> {
            #[cfg(test)]
            super::failure_injection::maybe_fail_decommit()?;

            // Unified path for macOS and Linux: MADV_FREE + mprotect(PROT_NONE).
            //
            // MADV_FREE marks pages for lazy reclamation — the cheapest decommit
            // on both platforms. The kernel reclaims physical pages when under
            // pressure; if it doesn't, old data may persist. No zeroing guarantee.
            //
            // mprotect(PROT_NONE) removes access. On recommit (mprotect RW), pages
            // may contain stale data (kernel kept them) or be zero-filled (kernel
            // reclaimed). We don't rely on either: debug/hardened mode zeroes
            // explicitly at the allocator layer; ordinary release does not care.
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
            #[cfg(test)]
            super::failure_injection::maybe_fail_release()?;

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
            crate::qen_debug_assert!(
                size != 0 && huge_page_size != 0 && size.is_multiple_of(huge_page_size),
                "alloc_huge: size ({size}) must be a non-zero multiple of huge_page_size ({huge_page_size})"
            );
            crate::qen_debug_assert!(
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
    use super::{NonNull, PlatformVmOps, VmError, VmOps};
    use std::ffi::c_void;
    use std::io;

    // Self-contained Win32 bindings. The `libc` crate exposes the C runtime
    // on Windows, NOT the Win32 API — `libc::VirtualAlloc` does not exist
    // (this module previously failed to compile at all because of that).
    // Signatures per Microsoft's `memoryapi.h` / `sysinfoapi.h`.

    const MEM_COMMIT: u32 = 0x0000_1000;
    const MEM_RESERVE: u32 = 0x0000_2000;
    const MEM_DECOMMIT: u32 = 0x0000_4000;
    const MEM_RELEASE: u32 = 0x0000_8000;
    /// `MEM_LARGE_PAGES` flag for `VirtualAlloc`.
    /// Allocates using large pages (typically 2MB on `x86_64`).
    /// Requires the process to hold `SeLockMemoryPrivilege`.
    const MEM_LARGE_PAGES: u32 = 0x2000_0000;
    const PAGE_NOACCESS: u32 = 0x01;
    const PAGE_READWRITE: u32 = 0x04;

    /// `SYSTEM_INFO` from sysinfoapi.h (the leading union of `dwOemId` /
    /// (`wProcessorArchitecture`, `wReserved`) is modelled as its two
    /// 16-bit fields; sizes and offsets are identical).
    #[repr(C)]
    struct SystemInfo {
        processor_architecture: u16,
        reserved: u16,
        page_size: u32,
        minimum_application_address: *mut c_void,
        maximum_application_address: *mut c_void,
        active_processor_mask: usize,
        number_of_processors: u32,
        processor_type: u32,
        allocation_granularity: u32,
        processor_level: u16,
        processor_revision: u16,
    }

    #[allow(non_snake_case)]
    unsafe extern "system" {
        fn VirtualAlloc(
            lpAddress: *mut c_void,
            dwSize: usize,
            flAllocationType: u32,
            flProtect: u32,
        ) -> *mut c_void;
        fn VirtualFree(lpAddress: *mut c_void, dwSize: usize, dwFreeType: u32) -> i32;
        fn GetSystemInfo(lpSystemInfo: *mut SystemInfo);
        /// Returns the minimum large page size supported by the system,
        /// or 0 if large pages are not supported.
        fn GetLargePageMinimum() -> usize;
    }

    impl VmOps for PlatformVmOps {
        unsafe fn reserve(size: usize) -> Result<NonNull<u8>, VmError> {
            #[cfg(test)]
            super::failure_injection::maybe_fail_reserve()?;

            // Safety: FFI call to VirtualAlloc.
            let ptr =
                unsafe { VirtualAlloc(std::ptr::null_mut(), size, MEM_RESERVE, PAGE_NOACCESS) };

            match NonNull::new(ptr.cast::<u8>()) {
                Some(p) => Ok(p),
                None => Err(VmError::ReservationFailed(io::Error::last_os_error())),
            }
        }

        unsafe fn commit(ptr: NonNull<u8>, size: usize) -> Result<(), VmError> {
            #[cfg(test)]
            super::failure_injection::maybe_fail_commit()?;

            // Safety: FFI call to VirtualAlloc.
            let result = unsafe {
                VirtualAlloc(
                    ptr.as_ptr().cast::<c_void>(),
                    size,
                    MEM_COMMIT,
                    PAGE_READWRITE,
                )
            };

            if result.is_null() {
                return Err(VmError::CommitFailed(io::Error::last_os_error()));
            }

            Ok(())
        }

        unsafe fn decommit(ptr: NonNull<u8>, size: usize) -> Result<(), VmError> {
            #[cfg(test)]
            super::failure_injection::maybe_fail_decommit()?;

            // Safety: FFI call to VirtualFree.
            if unsafe { VirtualFree(ptr.as_ptr().cast::<c_void>(), size, MEM_DECOMMIT) } == 0 {
                return Err(VmError::DecommitFailed(io::Error::last_os_error()));
            }

            Ok(())
        }

        unsafe fn release(ptr: NonNull<u8>, _size: usize) -> Result<(), VmError> {
            #[cfg(test)]
            super::failure_injection::maybe_fail_release()?;

            // Windows VirtualFree with MEM_RELEASE must have size 0 and the base address of the region.
            // Safety: FFI call to VirtualFree.
            if unsafe { VirtualFree(ptr.as_ptr().cast::<c_void>(), 0, MEM_RELEASE) } == 0 {
                return Err(VmError::ReleaseFailed(io::Error::last_os_error()));
            }
            Ok(())
        }

        fn page_size() -> usize {
            use crate::sync::OnceLock;
            static PAGE_SIZE: OnceLock<usize> = OnceLock::new();
            // Safety: FFI call to GetSystemInfo with a zeroed, correctly
            // laid-out SystemInfo it fully initialises.
            *PAGE_SIZE.get_or_init(|| unsafe {
                let mut info: SystemInfo = std::mem::zeroed();
                GetSystemInfo(&raw mut info);
                info.page_size as usize
            })
        }

        fn supported_page_sizes() -> Vec<usize> {
            use crate::sync::OnceLock;
            static CACHED: OnceLock<Vec<usize>> = OnceLock::new();
            CACHED
                .get_or_init(|| {
                    let base = Self::page_size();
                    let mut sizes = vec![base];
                    // Safety: trivial FFI query with no arguments.
                    let large_page = unsafe { GetLargePageMinimum() };
                    if large_page > 0 && large_page != base {
                        sizes.push(large_page);
                    }
                    sizes.sort_unstable();
                    sizes.dedup();
                    sizes
                })
                .clone()
        }

        unsafe fn alloc_huge(size: usize, huge_page_size: usize) -> Result<NonNull<u8>, VmError> {
            crate::qen_debug_assert!(
                size != 0 && huge_page_size != 0 && size.is_multiple_of(huge_page_size),
                "alloc_huge: size ({size}) must be a non-zero multiple of huge_page_size ({huge_page_size})",
            );

            // Runtime checks in all build modes (like the Linux path): a
            // debug_assert here would let release builds silently substitute
            // the system's large page size for the one requested.
            // Safety: trivial FFI query with no arguments.
            let system_large_page = unsafe { GetLargePageMinimum() };
            if system_large_page == 0 {
                return Err(VmError::ReservationFailed(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "large pages not available (GetLargePageMinimum returned 0); \
                     ensure SeLockMemoryPrivilege is granted",
                )));
            }
            // Windows only supports one large page size (returned by
            // GetLargePageMinimum). Typically 2MB on x86_64.
            if huge_page_size != system_large_page {
                return Err(VmError::ReservationFailed(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "alloc_huge: Windows large page size is {system_large_page} bytes, \
                         requested {huge_page_size}",
                    ),
                )));
            }

            // MEM_LARGE_PAGES must be combined with MEM_RESERVE | MEM_COMMIT.
            // The allocation is fully backed from the start (no partial commit).
            // Safety: FFI call to VirtualAlloc.
            let ptr = unsafe {
                VirtualAlloc(
                    std::ptr::null_mut(),
                    size,
                    MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                    PAGE_READWRITE,
                )
            };

            match NonNull::new(ptr.cast::<u8>()) {
                Some(p) => Ok(p),
                None => Err(VmError::ReservationFailed(io::Error::last_os_error())),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Loom/Miri mock: heap-backed VmOps (no real mmap/VirtualAlloc)
//
// Under `cfg(loom)` we cannot issue real VM syscalls — loom runs inside a
// single OS process with its own scheduler. Instead we back every "reservation"
// with a plain heap allocation (via `std::alloc::alloc` / `dealloc`).
//
// `commit` / `decommit` cannot change page protection (the backing is plain
// heap memory), so access-after-decommit is NOT detectable in this
// configuration — that boundary is only enforced by the real platform
// implementations (and would fault, not error, there). What the mock DOES
// verify, via `mock_registry`, is the commit/decommit *protocol*:
//   - commit/decommit must target a range inside a live reservation,
//   - decommit must target a fully-committed range,
//   - release must match a reservation's exact base and size.
// Violations panic, so loom and miri test runs catch allocator bookkeeping
// bugs that the previous no-op mock silently ignored.
// ---------------------------------------------------------------------------
#[cfg(any(loom, miri))]
mod mock_registry {
    use std::collections::HashMap;
    use std::sync::Mutex;

    struct Reservation {
        size: usize,
        /// Committed intervals as (offset, len), disjoint and sorted lazily.
        committed: Vec<(usize, usize)>,
    }

    // Plain std Mutex, NOT the loom shim: this is checker bookkeeping, not
    // modeled synchronization. It is never held across a loom yield point
    // (no loom-tracked operation happens inside the critical sections), so
    // it cannot deadlock the model.
    static REGISTRY: Mutex<Option<HashMap<usize, Reservation>>> = Mutex::new(None);

    fn with<R>(f: impl FnOnce(&mut HashMap<usize, Reservation>) -> R) -> R {
        let mut guard = REGISTRY
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        f(guard.get_or_insert_with(HashMap::new))
    }

    pub fn on_reserve(base: usize, size: usize) {
        with(|reg| {
            let prev = reg.insert(
                base,
                Reservation {
                    size,
                    committed: Vec::new(),
                },
            );
            assert!(
                prev.is_none(),
                "mock reserve returned an address that is already live"
            );
        });
    }

    /// Locate the reservation containing [addr, addr+len). Panics if none.
    fn containing<'a>(
        reg: &'a mut HashMap<usize, Reservation>,
        addr: usize,
        len: usize,
        op: &str,
    ) -> (usize, &'a mut Reservation) {
        for (&base, res) in reg.iter_mut() {
            if addr >= base && addr + len <= base + res.size {
                return (base, res);
            }
        }
        panic!("{op} of range {addr:#x}+{len:#x} outside any live mock reservation");
    }

    pub fn on_commit(addr: usize, len: usize) {
        with(|reg| {
            let (base, res) = containing(reg, addr, len, "commit");
            let (start, end) = (addr - base, addr - base + len);
            // Union-insert: merge with any overlapping/adjacent intervals.
            // Re-committing an already-committed range is allowed (the
            // allocator commits speculatively outside its lock).
            let mut new_start = start;
            let mut new_end = end;
            res.committed.retain(|&(s, l)| {
                let e = s + l;
                if e < new_start || s > new_end {
                    true // disjoint, keep
                } else {
                    new_start = new_start.min(s);
                    new_end = new_end.max(e);
                    false // absorbed
                }
            });
            res.committed.push((new_start, new_end - new_start));
        });
    }

    pub fn on_decommit(addr: usize, len: usize) {
        with(|reg| {
            let (base, res) = containing(reg, addr, len, "decommit");
            let (start, end) = (addr - base, addr - base + len);
            // The range must be fully committed: decommitting uncommitted
            // memory indicates allocator bookkeeping out of sync.
            let within = res
                .committed
                .iter()
                .find(|&&(s, l)| s <= start && end <= s + l)
                .copied();
            let Some((s, l)) = within else {
                panic!(
                    "decommit of not-fully-committed range {addr:#x}+{len:#x} \
                     (allocator commit bookkeeping out of sync)"
                );
            };
            // Subtract [start, end) from (s, l), keeping up to two remnants.
            res.committed.retain(|&iv| iv != (s, l));
            if s < start {
                res.committed.push((s, start - s));
            }
            if end < s + l {
                res.committed.push((end, s + l - end));
            }
        });
    }

    pub fn on_release(addr: usize, len: usize) {
        with(|reg| {
            let Some(res) = reg.get(&addr) else {
                panic!(
                    "release of {addr:#x} which is not the base of any live mock \
                     reservation (double release or interior pointer?)"
                );
            };
            assert_eq!(
                res.size, len,
                "release size {len:#x} does not match reservation size {:#x} at {addr:#x}",
                res.size,
            );
            reg.remove(&addr);
        });
    }
}

#[cfg(any(loom, miri))]
impl VmOps for PlatformVmOps {
    unsafe fn reserve(size: usize) -> Result<NonNull<u8>, VmError> {
        #[cfg(all(test, not(loom)))]
        failure_injection::maybe_fail_reserve()?;

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
        let ptr = NonNull::new(ptr).ok_or_else(|| {
            VmError::ReservationFailed(std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                "alloc returned null",
            ))
        })?;
        mock_registry::on_reserve(ptr.as_ptr() as usize, size);
        Ok(ptr)
    }

    unsafe fn commit(ptr: NonNull<u8>, size: usize) -> Result<(), VmError> {
        #[cfg(all(test, not(loom)))]
        failure_injection::maybe_fail_commit()?;

        mock_registry::on_commit(ptr.as_ptr() as usize, size);
        Ok(()) // heap memory is always accessible; protocol checked above
    }

    unsafe fn decommit(ptr: NonNull<u8>, size: usize) -> Result<(), VmError> {
        #[cfg(all(test, not(loom)))]
        failure_injection::maybe_fail_decommit()?;

        mock_registry::on_decommit(ptr.as_ptr() as usize, size);
        Ok(()) // memory remains accessible; protocol checked above
    }

    unsafe fn release(ptr: NonNull<u8>, size: usize) -> Result<(), VmError> {
        #[cfg(all(test, not(loom)))]
        failure_injection::maybe_fail_release()?;

        mock_registry::on_release(ptr.as_ptr() as usize, size);
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
        // Under loom/miri, forward to reserve + commit (no real huge pages);
        // huge allocations are committed from the start.
        // Safety: caller guarantees size > 0 and alignment requirements.
        let ptr = unsafe { Self::reserve(size)? };
        // Safety: freshly reserved above.
        unsafe { Self::commit(ptr, size)? };
        Ok(ptr)
    }
}

#[cfg(all(test, not(any(loom, miri))))]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // Fault tests: prove the commit/decommit protection boundary is real.
    //
    // A faulting access kills the whole process, so these run the fault
    // sequence in a CHILD process (the test binary re-executes itself,
    // filtered to the crash helper, with QEN_FAULT_SCENARIO set) and the
    // parent asserts the child died of an access violation. This is the
    // only configuration that can verify access-after-decommit/release
    // actually faults: the loom/miri mock is heap-backed and cannot
    // change page protection.
    // -----------------------------------------------------------------

    /// Executes a faulting scenario when `QEN_FAULT_SCENARIO` is set;
    /// a no-op in normal test runs. Named `crash_helper_*` (not `test_*`)
    /// to signal it is driven by the fault tests below.
    #[test]
    fn crash_helper_vm_fault_scenarios() {
        let Some(scenario) = std::env::var_os("QEN_FAULT_SCENARIO") else {
            return; // normal test run: nothing to do
        };
        let page = PlatformVmOps::page_size();
        // Safety: intentionally violates the access rules — in a child
        // process that the parent expects to die.
        unsafe {
            let ptr = PlatformVmOps::reserve(page).expect("reserve");
            PlatformVmOps::commit(ptr, page).expect("commit");
            ptr.as_ptr().write_volatile(0xAB);

            match scenario.to_str() {
                Some("decommit_read") => {
                    PlatformVmOps::decommit(ptr, page).expect("decommit");
                    let v = ptr.as_ptr().read_volatile(); // must fault
                    eprintln!("read {v:#x} from DECOMMITTED memory without faulting");
                }
                Some("decommit_write") => {
                    PlatformVmOps::decommit(ptr, page).expect("decommit");
                    ptr.as_ptr().write_volatile(0xCD); // must fault
                    eprintln!("wrote to DECOMMITTED memory without faulting");
                }
                Some("release_read") => {
                    PlatformVmOps::release(ptr, page).expect("release");
                    let v = ptr.as_ptr().read_volatile(); // must fault
                    eprintln!("read {v:#x} from RELEASED memory without faulting");
                }
                other => panic!("unknown QEN_FAULT_SCENARIO {other:?}"),
            }
        }
        // Reaching this point means the platform failed to revoke access.
        std::process::exit(42);
    }

    /// Spawn the crash helper with the given scenario and assert the
    /// child died of an access violation (not a clean exit, not a panic).
    fn assert_scenario_faults(scenario: &str) {
        let exe = std::env::current_exe().expect("current_exe");
        let mut command = std::process::Command::new(exe);
        command
            .args([
                "--exact",
                "memory::vm::tests::crash_helper_vm_fault_scenarios",
                "--test-threads=1",
                "--nocapture",
            ])
            .env("QEN_FAULT_SCENARIO", scenario)
            .env_remove("RUST_BACKTRACE")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        // macOS may spend minutes writing a core for an intentional fault,
        // leaving the parent test blocked in `Command::output`. Disable core
        // dumps in the child; the terminating signal remains observable.
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;

            // Safety: `setrlimit` is async-signal-safe and this closure does
            // not allocate or touch shared Rust state between fork and exec.
            unsafe {
                command.pre_exec(|| {
                    let no_core = libc::rlimit {
                        rlim_cur: 0,
                        rlim_max: 0,
                    };
                    if libc::setrlimit(libc::RLIMIT_CORE, &raw const no_core) == 0 {
                        Ok(())
                    } else {
                        Err(std::io::Error::last_os_error())
                    }
                });
            }
        }

        let mut child = command.spawn().expect("failed to spawn crash helper");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            if child
                .try_wait()
                .expect("failed to poll crash helper")
                .is_some()
            {
                break;
            }
            if std::time::Instant::now() >= deadline {
                drop(child.kill());
                let output = child
                    .wait_with_output()
                    .expect("failed to collect timed-out crash helper");
                panic!(
                    "fault scenario timed out ({scenario})\nstdout: {}\nstderr: {}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr),
                );
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let output = child
            .wait_with_output()
            .expect("failed to collect crash helper output");

        assert!(
            !output.status.success(),
            "child accessed protected memory without faulting ({scenario}): {output:?}"
        );

        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            let sig = output.status.signal();
            assert!(
                sig == Some(libc::SIGSEGV) || sig == Some(libc::SIGBUS),
                "expected SIGSEGV/SIGBUS for {scenario}, got {:?}\nstdout: {}\nstderr: {}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
        }
        #[cfg(windows)]
        {
            // 0xC0000005 = STATUS_ACCESS_VIOLATION (as a signed i32).
            assert_eq!(
                output.status.code(),
                Some(-1073741819i32),
                "expected STATUS_ACCESS_VIOLATION for {scenario}, got {:?}",
                output.status,
            );
        }
    }

    #[test]
    #[cfg_attr(
        target_os = "macos",
        ignore = "intentional access faults can block indefinitely in macOS crash handling"
    )]
    fn test_decommitted_memory_faults_on_read() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        assert_scenario_faults("decommit_read");
    }

    #[test]
    #[cfg_attr(
        target_os = "macos",
        ignore = "intentional access faults can block indefinitely in macOS crash handling"
    )]
    fn test_decommitted_memory_faults_on_write() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        assert_scenario_faults("decommit_write");
    }

    #[test]
    #[cfg_attr(
        target_os = "macos",
        ignore = "intentional access faults can block indefinitely in macOS crash handling"
    )]
    fn test_released_memory_faults_on_read() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        assert_scenario_faults("release_read");
    }

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
        assert_eq!(size & (size - 1), 0, "Page size {size} is not power of two");
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
        // in debug or hardened builds.
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

    #[cfg(any(debug_assertions, feature = "hardened"))]
    #[test]
    #[should_panic(expected = "must be a non-zero multiple")]
    fn test_alloc_huge_bad_args_size_zero_panics() {
        // Safety: Test code.
        unsafe {
            drop(PlatformVmOps::alloc_huge(0, PAGE_SIZE_2MB));
        }
    }

    #[cfg(any(debug_assertions, feature = "hardened"))]
    #[test]
    #[should_panic(expected = "must be a non-zero multiple")]
    fn test_alloc_huge_bad_args_huge_page_size_zero_panics() {
        // Safety: Test code.
        unsafe {
            drop(PlatformVmOps::alloc_huge(PAGE_SIZE_2MB, 0));
        }
    }

    #[cfg(any(debug_assertions, feature = "hardened"))]
    #[test]
    #[should_panic(expected = "must be a non-zero multiple")]
    fn test_alloc_huge_bad_args_not_multiple_panics() {
        // Safety: Test code.
        unsafe {
            drop(PlatformVmOps::alloc_huge(PAGE_SIZE_2MB + 1, PAGE_SIZE_2MB));
        }
    }

    #[cfg(any(debug_assertions, feature = "hardened"))]
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
