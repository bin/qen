//! All counters use `Relaxed` ordering. Individual counter values are
//! eventually consistent. Cross-counter snapshots may be transiently
//! inconsistent (e.g., total may briefly disagree with the sum of
//! per-subsystem counters). This is acceptable for diagnostic display.
//! Do NOT use these values for allocation decisions.

use crate::sync::atomic::{AtomicIsize, Ordering};

/// Diagnostic-only gauge counter.
///
/// Under contention, subtract-before-add races are tolerated and the raw value
/// may transiently dip below zero. Readers should always use `load()`/`get()`,
/// which clamp negative values to zero.
pub struct Counter(AtomicIsize);

impl Counter {
    #[cfg(not(loom))]
    pub const fn new() -> Self {
        Self(AtomicIsize::new(0))
    }

    #[cfg(loom)]
    pub fn new() -> Self {
        Self(AtomicIsize::new(0))
    }

    #[inline]
    fn delta(val: usize) -> isize {
        // Diagnostic counters only: clamp absurd deltas instead of panicking.
        std::cmp::min(val, isize::MAX as usize).cast_signed()
    }

    #[inline]
    #[allow(dead_code)]
    pub fn add(&self, val: usize) {
        self.0.fetch_add(Self::delta(val), Ordering::Relaxed);
    }

    #[inline]
    pub fn sub(&self, val: usize) {
        self.0.fetch_sub(Self::delta(val), Ordering::Relaxed);
    }

    #[inline]
    #[allow(dead_code)]
    pub fn get(&self) -> usize {
        self.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn load(&self, ordering: Ordering) -> usize {
        self.0.load(ordering).max(0).cast_unsigned()
    }

    #[inline]
    pub fn fetch_add(&self, val: usize, ordering: Ordering) -> usize {
        self.0.fetch_add(Self::delta(val), ordering).max(0).cast_unsigned()
    }

    #[inline]
    #[allow(dead_code)]
    pub fn fetch_sub(&self, val: usize, ordering: Ordering) -> usize {
        self.0.fetch_sub(Self::delta(val), ordering).max(0).cast_unsigned()
    }
}

// Total address space reserved by the application engine allocators
crate::sync::static_atomic! {
    pub static TOTAL_RESERVED: Counter = Counter::new();
}
// Total physical memory committed by the application engine allocators
crate::sync::static_atomic! {
    pub static TOTAL_COMMITTED: Counter = Counter::new();
}

// Breakdown by subsystem
crate::sync::static_atomic! {
    pub static CHUNK_POOL_COMMITTED: Counter = Counter::new();
}
crate::sync::static_atomic! {
    pub static CHUNK_POOL_LIVE: Counter = Counter::new();
}

crate::sync::static_atomic! {
    pub static FRAME_ARENA_COMMITTED: Counter = Counter::new();
}

crate::sync::static_atomic! {
    pub static BINNED_ALLOCATOR_COMMITTED: Counter = Counter::new();
}
crate::sync::static_atomic! {
    pub static LARGE_ALLOC_CACHE_COMMITTED: Counter = Counter::new();
}

crate::sync::static_atomic! {
    pub static COMMAND_ARENA_COMMITTED: Counter = Counter::new();
}

/// Best-effort subtract from a diagnostic atomic counter.
///
/// Uses a single atomic subtraction (no TOCTOU load-then-subtract race).
/// Readers clamp negative transients via `Counter::load`.
pub fn sub_saturating(counter: &Counter, val: usize) {
    counter.sub(val);
}
