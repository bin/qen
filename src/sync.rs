// Unified synchronization primitive shim.
//
// Under `cfg(loom)`, re-exports from the patched `loom` crate (which includes
// `AtomicU128`/`AtomicI128`).  Otherwise, re-exports from `std` and
// `portable_atomic`.
//
// **Every** file in the crate must import sync primitives through this module.
// A single direct `use std::sync::atomic::*` would bypass loom's scheduler and
// silently break exhaustive testing.
#![allow(unused_imports, unused_macros)]

// ---------------------------------------------------------------------------
// atomic
// ---------------------------------------------------------------------------
pub(crate) mod atomic {
    #[cfg(loom)]
    pub(crate) use loom::sync::atomic::{
        AtomicU32, AtomicU64, AtomicUsize, AtomicIsize, Ordering, fence,
        AtomicU128, AtomicI128,
    };

    #[cfg(not(loom))]
    pub(crate) use std::sync::atomic::{
        AtomicU32, AtomicU64, AtomicUsize, AtomicIsize, Ordering, fence,
    };

    #[cfg(not(loom))]
    pub(crate) use portable_atomic::{AtomicU128, AtomicI128};
}

// ---------------------------------------------------------------------------
// sync (Mutex, Arc, RwLock)
// ---------------------------------------------------------------------------
#[cfg(loom)]
pub(crate) use loom::sync::{Mutex, Arc, RwLock};

#[cfg(not(loom))]
pub(crate) use std::sync::{Mutex, Arc, RwLock};

// ---------------------------------------------------------------------------
// cell (UnsafeCell, Cell)
//
// loom's UnsafeCell differs from std: `.get()` returns a `ConstPtr<T>` wrapper
// instead of `*mut T`.  To write code that compiles under both, use the
// `unsafe_cell_get!` and `unsafe_cell_get_mut!` helper macros.
// ---------------------------------------------------------------------------
pub(crate) mod cell {
    #[cfg(loom)]
    pub(crate) use loom::cell::{UnsafeCell, Cell};

    #[cfg(not(loom))]
    pub(crate) use std::cell::{UnsafeCell, Cell};
}

/// Access the contents of an `UnsafeCell` as `&mut T`.
///
/// Under std: `&mut *cell.get()`
/// Under loom: `cell.with_mut(|p| &mut *p)`
///
/// # Safety
/// Caller must guarantee exclusive access (same as `UnsafeCell::get`).
macro_rules! unsafe_cell_get_mut {
    ($cell:expr) => {{
        #[cfg(not(loom))]
        {
            // Safety: upheld by caller.
            unsafe { &mut *$cell.get() }
        }
        #[cfg(loom)]
        {
            // Safety: upheld by caller.
            unsafe { $cell.with_mut(|p| &mut *p) }
        }
    }};
}
pub(crate) use unsafe_cell_get_mut;

// ---------------------------------------------------------------------------
// hint
// ---------------------------------------------------------------------------
pub(crate) mod hint {
    #[cfg(loom)]
    pub(crate) use loom::hint::spin_loop;

    #[cfg(not(loom))]
    pub(crate) use std::hint::spin_loop;
}

// ---------------------------------------------------------------------------
// thread
// ---------------------------------------------------------------------------
pub(crate) mod thread {
    #[cfg(loom)]
    pub(crate) use loom::thread::{spawn, yield_now, current, JoinHandle};

    #[cfg(not(loom))]
    pub(crate) use std::thread::{spawn, yield_now, current, JoinHandle};
}

// ---------------------------------------------------------------------------
// Barrier — loom does not provide Barrier; we shim an atomic countdown.
// Standard tests keep std::sync::Barrier.
// ---------------------------------------------------------------------------
pub(crate) mod barrier {
    #[cfg(not(loom))]
    pub(crate) use std::sync::Barrier;

    /// Under loom, Barrier is not available.  We provide a minimal spin-barrier
    /// built on loom atomics so that existing tests compile unmodified.
    #[cfg(loom)]
    #[allow(dead_code)]
    pub(crate) struct Barrier {
        total: usize,
        count: super::atomic::AtomicUsize,
    }

    #[cfg(loom)]
    #[allow(dead_code)]
    impl Barrier {
        pub(crate) fn new(n: usize) -> Self {
            Self {
                total: n,
                count: super::atomic::AtomicUsize::new(0),
            }
        }

        pub(crate) fn wait(&self) {
            use super::atomic::Ordering;
            let arrived = self.count.fetch_add(1, Ordering::AcqRel) + 1;
            if arrived < self.total {
                while self.count.load(Ordering::Acquire) < self.total {
                    loom::thread::yield_now();
                }
            }
            // All threads have arrived.
        }
    }
}

// ---------------------------------------------------------------------------
// OnceLock shim
//
// loom does not provide OnceLock.  Under cfg(loom) we use a std Mutex<Option<T>>
// (not a loom Mutex) because OnceLock is used in `static` items and loom's
// Mutex::new() is not const.  Since OnceLock is init-once, the inner Mutex is
// not a synchronization point that loom needs to explore — it only serialises
// the one-shot initialisation.
// ---------------------------------------------------------------------------
#[cfg(not(loom))]
pub(crate) use std::sync::OnceLock;

#[cfg(loom)]
pub(crate) struct OnceLock<T> {
    inner: std::sync::Mutex<Option<T>>,
}

#[cfg(loom)]
impl<T> OnceLock<T> {
    pub(crate) const fn new() -> Self {
        Self {
            inner: std::sync::Mutex::new(None),
        }
    }

    pub(crate) fn get(&self) -> Option<&T> {
        let guard = self.inner.lock().unwrap();
        if guard.is_some() {
            let ptr: *const T = guard.as_ref().unwrap();
            // Safety: the value is never moved or dropped while &self is live.
            Some(unsafe { &*ptr })
        } else {
            None
        }
    }

    pub(crate) fn get_or_init<F: FnOnce() -> T>(&self, f: F) -> &T {
        let mut guard = self.inner.lock().unwrap();
        if guard.is_none() {
            *guard = Some(f());
        }
        let ptr: *const T = guard.as_ref().unwrap();
        // Safety: same as above — value lives as long as &self.
        unsafe { &*ptr }
    }

    pub(crate) fn set(&self, value: T) -> Result<(), T> {
        let mut guard = self.inner.lock().unwrap();
        if guard.is_some() {
            return Err(value);
        }
        *guard = Some(value);
        Ok(())
    }
}

#[cfg(loom)]
// Safety: access is serialised by the inner Mutex.
unsafe impl<T: Send> Sync for OnceLock<T> {}

// ---------------------------------------------------------------------------
// Static initialisation helpers
//
// loom atomics/Mutex/RwLock are not const-constructible.  These macros
// create statics that work under both loom and std.
// ---------------------------------------------------------------------------

/// Declare a `static` atomic.  Under std, uses `const` init.  Under loom,
/// uses `loom::lazy_static!` so the value is re-created for each model run.
///
/// Usage: `static_atomic! { [pub] static NAME: Type = init_expr; }`
#[allow(unused_macro_rules)]
macro_rules! static_atomic {
    (pub static $NAME:ident : $Ty:ty = $init:expr ;) => {
        #[cfg(not(loom))]
        pub static $NAME: $Ty = $init;

        #[cfg(loom)]
        loom::lazy_static! {
            pub static ref $NAME: $Ty = $init;
        }
    };
    (static $NAME:ident : $Ty:ty = $init:expr ;) => {
        #[cfg(not(loom))]
        static $NAME: $Ty = $init;

        #[cfg(loom)]
        loom::lazy_static! {
            static ref $NAME: $Ty = $init;
        }
    };
}
pub(crate) use static_atomic;

/// Declare a `static` `RwLock`.  Under std, uses `const` init.  Under loom,
/// uses `loom::lazy_static!`.
#[allow(unused_macro_rules)]
macro_rules! static_rwlock {
    (pub static $NAME:ident : $Ty:ty = $init:expr ;) => {
        #[cfg(not(loom))]
        pub static $NAME: $Ty = $init;

        #[cfg(loom)]
        loom::lazy_static! {
            pub static ref $NAME: $Ty = $init;
        }
    };
    (static $NAME:ident : $Ty:ty = $init:expr ;) => {
        #[cfg(not(loom))]
        static $NAME: $Ty = $init;

        #[cfg(loom)]
        loom::lazy_static! {
            static ref $NAME: $Ty = $init;
        }
    };
}
pub(crate) use static_rwlock;

