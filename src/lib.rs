// Raw initial-exec TLS for the allocator fast path: `thread_local!`'s
// `LocalKey::with` machinery measured 1.58 ns/pair — a third of the hot
// alloc/free pair (see `fastpath_decomposition` in binned.rs).
#![feature(thread_local)]

#[cfg(not(target_pointer_width = "64"))]
compile_error!("qen supports only 64-bit targets.");

/// Debug assertions that remain enabled in optimized builds when the
/// `hardened` feature is selected. Using `cfg!` keeps expressions type-checked
/// and variables considered used in every build while allowing LLVM to remove
/// the branch entirely from ordinary release builds.
macro_rules! qen_debug_assert {
    ($($arg:tt)*) => {
        if cfg!(any(debug_assertions, feature = "hardened")) {
            assert!($($arg)*);
        }
    };
}

macro_rules! qen_debug_assert_eq {
    ($($arg:tt)*) => {
        if cfg!(any(debug_assertions, feature = "hardened")) {
            assert_eq!($($arg)*);
        }
    };
}

macro_rules! qen_debug_assert_ne {
    ($($arg:tt)*) => {
        if cfg!(any(debug_assertions, feature = "hardened")) {
            assert_ne!($($arg)*);
        }
    };
}

pub(crate) use qen_debug_assert;
pub(crate) use qen_debug_assert_eq;
pub(crate) use qen_debug_assert_ne;

pub(crate) mod sync;

// public module: contains implementation details (hidden via pub(crate))
// and TEST_MUTEX (public for tests)
pub mod memory;

// allocators/arenas
pub use memory::binned::{
    BinnedAllocator, BinnedAllocatorConfig, GlobalBinnedAllocator, MAX_SMALL_SIZE,
};
pub use memory::chunk_pool::{CHUNK_ALIGN, CHUNK_SIZE, ChunkPool, GlobalChunkPool};
pub use memory::command_arena::{CommandArena, GlobalSharedPagePool, SharedPagePool};
pub use memory::entity_alloc::{EntityAllocator, EntityLocation};
pub use memory::frame_arena::{FrameArena, with_frame_arena};

// mgmt/stats
pub use memory::manager::{MemoryManager, MemoryStats};

// errors
pub use memory::vm::VmError;

// integration tests
// #[cfg(test)]
// pub use memory::integration::*;
