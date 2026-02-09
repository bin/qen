pub(crate) mod binned;
pub(crate) mod chunk_pool;
pub(crate) mod command_arena;
pub(crate) mod entity_alloc;
pub(crate) mod frame_arena;
pub(crate) mod integration;
pub(crate) mod large_cache;
pub(crate) mod loom_tests;
pub(crate) mod manager;
pub(crate) mod stats;
pub(crate) mod vm;

#[cfg(test)]
crate::sync::static_rwlock! {
    pub static TEST_MUTEX: crate::sync::RwLock<()> = crate::sync::RwLock::new(());
}
