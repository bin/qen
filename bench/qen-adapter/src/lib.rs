//! C-ABI adapter exposing qen through rpmalloc-benchmark's `benchmark.h`.
//!
//! The suite drives the C `malloc/free` shape, so `benchmark_free(ptr)`
//! carries no size. qen's own unsized path handles that for everything
//! pool-backed: `GlobalBinnedAllocator::try_free_ptr` masks the pointer to
//! its pool's aligned base and reads the size class from a lock-free,
//! L1-resident table — the address is the metadata. This adapter therefore
//! adds NO small-path bookkeeping at all (an earlier revision kept its own
//! span table here; its per-alloc store once serialized the whole
//! 16-thread benchmark — see RESULTS.md — and qen's internal lookup has
//! since replaced it outright).
//!
//! **Large allocations** (`size > MAX_SMALL_SIZE` or over-aligned) are
//! standalone mappings the class table cannot describe, so an exact
//! `ptr -> layout` side map behind a mutex recovers their sized `free`.
//! Large allocations are page-granular and syscall-adjacent; the map cost
//! is noise there.

use std::alloc::Layout;
use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void};
use std::ptr::NonNull;
use std::sync::{Mutex, OnceLock};

use qen::{GlobalBinnedAllocator, MAX_SMALL_SIZE};

/// Alignment every binned allocation satisfies (bin sizes are multiples of
/// 16 and blocks are 16-aligned). Requests up to this go through the
/// binned path.
const SMALL_ALIGN: usize = 16;

static LARGE: OnceLock<Mutex<HashMap<usize, Layout>>> = OnceLock::new();

fn large() -> &'static Mutex<HashMap<usize, Layout>> {
    LARGE.get_or_init(|| Mutex::new(HashMap::new()))
}

#[unsafe(no_mangle)]
pub extern "C" fn benchmark_initialize() -> c_int {
    // May already be initialized on repeat calls; both outcomes are fine.
    drop(GlobalBinnedAllocator::init());
    let _ = large();
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn benchmark_finalize() -> c_int {
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn benchmark_thread_initialize() -> c_int {
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn benchmark_thread_finalize() -> c_int {
    // qen's thread cache flushes via TLS destructor on thread exit.
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn benchmark_thread_collect() {}

#[unsafe(no_mangle)]
pub extern "C" fn benchmark_name() -> *const c_char {
    c"qen".as_ptr()
}

#[unsafe(no_mangle)]
pub extern "C" fn benchmark_malloc(alignment: usize, size: usize) -> *mut c_void {
    let size = size.max(1);

    if size <= MAX_SMALL_SIZE && alignment <= SMALL_ALIGN {
        match GlobalBinnedAllocator::alloc_bytes(size) {
            Ok(ptr) => ptr.as_ptr().cast(),
            Err(_) => std::ptr::null_mut(),
        }
    } else {
        let layout = Layout::from_size_align(size, alignment.max(SMALL_ALIGN))
            .expect("invalid layout request");
        match GlobalBinnedAllocator::alloc(layout) {
            Ok(ptr) => {
                large().lock().unwrap().insert(ptr.as_ptr() as usize, layout);
                ptr.as_ptr().cast()
            }
            Err(_) => std::ptr::null_mut(),
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn benchmark_free(ptr: *mut c_void) {
    let Some(ptr) = NonNull::new(ptr.cast::<u8>()) else {
        return;
    };

    // Pool-backed pointers free themselves from the address alone.
    // Safety: ptr came from benchmark_malloc and is freed once.
    if unsafe { GlobalBinnedAllocator::try_free_ptr(ptr) } {
        return;
    }

    let layout = large()
        .lock()
        .unwrap()
        .remove(&(ptr.as_ptr() as usize))
        .expect("benchmark_free: pointer unknown to the qen adapter");
    // Safety: ptr/layout pair recorded at allocation time.
    unsafe { GlobalBinnedAllocator::free(ptr, layout) };
}
