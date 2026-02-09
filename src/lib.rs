#[cfg(not(target_pointer_width = "64"))]
compile_error!("qen supports only 64-bit targets.");

pub(crate) mod sync;

// public module: contains implementation details (hidden via pub(crate))
// and TEST_MUTEX (public for tests)
pub mod memory;

// allocators/arenas
pub use memory::binned::{BinnedAllocator, BinnedAllocatorConfig, GlobalBinnedAllocator};
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
