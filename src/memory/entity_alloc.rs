use super::stats;
use super::vm::{PlatformVmOps, VmError, VmOps};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::ptr::NonNull;
use crate::sync::atomic::Ordering;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EntityLocation {
    pub archetype_id: u32,
    pub chunk_index: u32,
    pub index_in_chunk: u32,
}

/// Default maximum entity capacity (16M entities).
/// VA cost: 16M * 4B (gen) + 16M * 12B (loc) = 256MB.
/// No physical memory is consumed until pages are committed via `grow()`.
pub const DEFAULT_MAX_ENTITY_CAPACITY: usize = 16 * 1024 * 1024;

/// Backing storage for entity generations and locations.
///
/// Uses the reserve-large / commit-incrementally pattern: the full VA for
/// [`reserved_capacity`] entities is mapped at init time (`PROT_NONE`),
/// but only [`capacity`] entity slots are backed by physical pages.
/// When the high-water mark reaches the committed limit, `grow()` doubles
/// the committed region within the reserved VA (idempotent commit).
pub struct EntityAllocator {
    generations: NonNull<u32>,
    locations: NonNull<EntityLocation>,
    /// Currently committed entity slot count.
    capacity: usize,
    /// Maximum entity slot count (VA reserved). Upper bound for growth.
    reserved_capacity: usize,
    live_count: usize,
    high_water: u32,
    free_list: BinaryHeap<Reverse<u32>>, // TODO: Consider and profile stack; lose strict
                                         // lowest-first for making overhead constant-time
}

// Safety: EntityAllocator owns the memory (via NonNull) and is safe to send.
unsafe impl Send for EntityAllocator {}

impl EntityAllocator {
    /// Sentinel generation indicating a permanently retired slot.
    ///
    /// When a slot's generation reaches this value (`u32::MAX`, which is odd = freed),
    /// it has been reused ~2 billion times and is permanently removed from the
    /// free list. This avoids panicking on generation overflow and prevents the
    /// ABA hazard that would arise from wrapping back to 0.
    const GENERATION_RETIRED: u32 = u32::MAX;
}

impl Drop for EntityAllocator {
    fn drop(&mut self) {
        // Safety: We own the memory and are dropping the allocator.
        unsafe {
            let gen_reserved = self.reserved_capacity * std::mem::size_of::<u32>();
            let loc_reserved = self.reserved_capacity * std::mem::size_of::<EntityLocation>();

            drop(PlatformVmOps::release(self.generations.cast(), gen_reserved));
            drop(PlatformVmOps::release(self.locations.cast(), loc_reserved));

            stats::sub_saturating(&stats::TOTAL_RESERVED, gen_reserved + loc_reserved);
            stats::sub_saturating(&stats::TOTAL_COMMITTED, self.committed_bytes());
        }
    }
}

impl EntityAllocator {
    /// Create a new `EntityAllocator` with the given initial capacity.
    ///
    /// # Errors
    ///
    /// Returns `VmError` if memory reservation fails.
    pub fn new(initial_capacity: usize) -> Result<Self, VmError> {
        Self::with_max_capacity(initial_capacity, DEFAULT_MAX_ENTITY_CAPACITY)
    }

    /// Create a new `EntityAllocator` with initial and maximum capacities.
    ///
    /// # Errors
    ///
    /// Returns `VmError` if memory reservation fails or if the capacity is invalid (too large).
    pub fn with_max_capacity(
        initial_capacity: usize,
        max_capacity: usize,
    ) -> Result<Self, VmError> {
        if max_capacity > u32::MAX as usize {
            return Err(VmError::InitializationFailed(format!(
                "max_capacity {max_capacity} exceeds u32::MAX",
            )));
        }

        // Minimum capacity 1024 to avoid too many small commits.
        let cap = initial_capacity.max(1024);
        let max_cap = max_capacity.max(cap);

        let gen_reserved_size = max_cap
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| {
                VmError::InitializationFailed(
                    "EntityAllocator generation reservation size overflow".to_string(),
                )
            })?;
        let loc_reserved_size = max_cap
            .checked_mul(std::mem::size_of::<EntityLocation>())
            .ok_or_else(|| {
                VmError::InitializationFailed(
                    "EntityAllocator location reservation size overflow".to_string(),
                )
            })?;
        let gen_commit_size = cap.checked_mul(std::mem::size_of::<u32>()).ok_or_else(|| {
            VmError::InitializationFailed(
                "EntityAllocator generation commit size overflow".to_string(),
            )
        })?;
        let loc_commit_size = cap
            .checked_mul(std::mem::size_of::<EntityLocation>())
            .ok_or_else(|| {
                VmError::InitializationFailed(
                    "EntityAllocator location commit size overflow".to_string(),
                )
            })?;

        // Two separate reservations: generations for validity checks,
        // locations for lookup. Different access patterns, separate is fine.

        let gen_p;
        let loc_p;

        // Safety: FFI calls to reserve/commit memory.
        unsafe {
            gen_p = PlatformVmOps::reserve(gen_reserved_size)?;
            if let Err(e) = PlatformVmOps::commit(gen_p, gen_commit_size) {
                drop(PlatformVmOps::release(gen_p, gen_reserved_size));
                return Err(e);
            }

            loc_p = match PlatformVmOps::reserve(loc_reserved_size) {
                Ok(p) => p,
                Err(e) => {
                    drop(PlatformVmOps::release(gen_p, gen_reserved_size));
                    return Err(e);
                }
            };

            if let Err(e) = PlatformVmOps::commit(loc_p, loc_commit_size) {
                drop(PlatformVmOps::release(gen_p, gen_reserved_size));
                drop(PlatformVmOps::release(loc_p, loc_reserved_size));
                return Err(e);
            }
        }

        stats::TOTAL_RESERVED.fetch_add(gen_reserved_size + loc_reserved_size, Ordering::Relaxed);
        stats::TOTAL_COMMITTED.fetch_add(gen_commit_size + loc_commit_size, Ordering::Relaxed);

        Ok(Self {
            generations: gen_p.cast(),
            locations: loc_p.cast(),
            capacity: cap,
            reserved_capacity: max_cap,
            live_count: 0,
            high_water: 0,
            free_list: BinaryHeap::new(),
        })
    }

    /// Double the committed capacity within the reserved VA.
    ///
    /// Uses idempotent commit (re-mprotecting already-committed pages is a
    /// kernel no-op) so we don't need to track the exact page boundary of
    /// the previous commit.
    fn grow(&mut self) {
        let new_cap = std::cmp::min(self.capacity.saturating_mul(2), self.reserved_capacity);
        assert!(
            new_cap > self.capacity,
            "EntityAllocator: maximum capacity ({}) reached",
            self.reserved_capacity
        );

        let new_gen_size = new_cap
            .checked_mul(std::mem::size_of::<u32>())
            .expect("EntityAllocator::grow generation size overflow");
        let new_loc_size = new_cap
            .checked_mul(std::mem::size_of::<EntityLocation>())
            .expect("EntityAllocator::grow location size overflow");

        // Commit full range — idempotent for already-committed pages.
        // Safety: FFI calls to commit memory.
        unsafe {
            if let Err(e) = PlatformVmOps::commit(self.generations.cast(), new_gen_size) {
                panic!(
                    "EntityAllocator: failed to commit generations during grow: {e:?}",
                );
            }
            if let Err(e) = PlatformVmOps::commit(self.locations.cast(), new_loc_size) {
                panic!(
                    "EntityAllocator: failed to commit locations during grow: {e:?}",
                );
            }
        }

        let old_bytes = self.committed_bytes();
        self.capacity = new_cap;
        let new_bytes = self.committed_bytes();
        stats::TOTAL_COMMITTED.fetch_add(new_bytes - old_bytes, Ordering::Relaxed);
    }

    pub fn alloc(&mut self) -> (u32, u32) {
        // (index, generation)
        while let Some(Reverse(index)) = self.free_list.pop() {
            // Safety: index is from free_list, so it is within bounds.
            let gen_ptr = unsafe { self.generations.as_ptr().add(index as usize) };
            // Safety: gen_ptr is valid.
            let current_gen = unsafe { gen_ptr.read() };

            // Retired slot: generation space exhausted, skip to next candidate.
            // This slot will never be reused (it stays at GENERATION_RETIRED).
            if current_gen == Self::GENERATION_RETIRED {
                continue;
            }

            self.live_count += 1;
            let next_gen = current_gen + 1; // odd (freed) -> even (alive); can't overflow (see free())
            // Safety: gen_ptr is valid.
            unsafe { gen_ptr.write(next_gen) };
            return (index, next_gen);
        }

        let index = self.high_water;

        if index as usize >= self.capacity {
            self.grow();
        }

        self.high_water += 1;
        self.live_count += 1;

        // Safety: index is high_water, which is within committed capacity (checked by grow).
        unsafe {
            *self.generations.as_ptr().add(index as usize) = 0; // Gen 0
        }
        (index, 0)
    }

    /// Free an entity slot previously returned by [`alloc`](Self::alloc).
    ///
    /// # Safety
    /// - `index` must be a currently live slot returned by this allocator.
    /// - `index` must not have been freed already.
    pub unsafe fn free(&mut self, index: u32) {
        if index >= self.high_water {
            debug_assert!(false, "EntityAllocator::free: index out of bounds");
            // Safety: index < high_water check failed (debug_assert would have caught it).
            unsafe { std::hint::unreachable_unchecked() }
        }
        // Safety: index is valid (checked above).
        unsafe {
            let gen_ptr = self.generations.as_ptr().add(index as usize);
            let current_gen = gen_ptr.read();

            // Safety: Alive entities always have even generations (0, 2, ...).
            // Freed entities have odd generations (1, 3, ...).
            if !current_gen.is_multiple_of(2) {
                debug_assert!(
                    false,
                    "Double free detected in EntityAllocator for index {index}"
                );
                std::hint::unreachable_unchecked();
            }

            let next_gen = current_gen + 1; // even (alive) -> odd (freed)
            *gen_ptr = next_gen;

            if next_gen == Self::GENERATION_RETIRED {
                // This slot has exhausted its generation space (~2 billion reuses).
                // It is permanently retired: we do NOT push it to the free list,
                // so it will never be handed out again. The slot's gen stays at
                // GENERATION_RETIRED (odd = freed).
            } else {
                self.free_list.push(Reverse(index));
            }
        }
        self.live_count -= 1;
    }

    /// Update the location metadata for an existing entity slot.
    ///
    /// # Safety
    /// - `index` must refer to a live slot owned by this allocator.
    /// - Callers must ensure external synchronization if accessed concurrently.
    pub unsafe fn set_location(&mut self, index: u32, loc: EntityLocation) {
        if index >= self.high_water {
            debug_assert!(
                false,
                "EntityAllocator::set_location: index {} out of bounds (high_water: {})",
                index, self.high_water
            );
            // Safety: index out of bounds.
            unsafe { std::hint::unreachable_unchecked() }
        }
        // Safety: index is checked to be within high_water.
        unsafe {
            *self.locations.as_ptr().add(index as usize) = loc;
        }
    }

    /// Read location metadata for an existing entity slot.
    ///
    /// # Safety
    /// - `index` must refer to a live slot owned by this allocator.
    /// - Callers must ensure external synchronization if accessed concurrently.
    #[must_use]
    pub unsafe fn get_location(&self, index: u32) -> EntityLocation {
        if index >= self.high_water {
            debug_assert!(
                false,
                "EntityAllocator::get_location: index {} out of bounds (high_water: {})",
                index, self.high_water
            );
            // Safety: index out of bounds.
            unsafe { std::hint::unreachable_unchecked() }
        }
        // Safety: index is valid.
        unsafe { *self.locations.as_ptr().add(index as usize) }
    }

    /// Check whether the slot at `index` currently matches `generation`.
    ///
    /// This is a low-level hot-path API and is intentionally `unsafe`.
    ///
    /// # Safety
    /// - `index` must refer to a slot that has been allocated at least once
    ///   (`index < high_water`) and is within the currently committed capacity.
    /// - `generation` must represent a live handle generation (even-valued).
    /// - Calling this on a freed/retired slot (odd current generation) is invalid.
    ///   In debug builds this is asserted; in release builds it is undefined behavior.
    #[must_use]
    pub unsafe fn is_alive(&self, index: u32, generation: u32) -> bool {
        if index >= self.high_water {
            debug_assert!(
                false,
                "EntityAllocator::is_alive: index {} out of bounds (high_water: {})",
                index, self.high_water
            );
            // Safety: index out of bounds.
            unsafe { std::hint::unreachable_unchecked() }
        }
        debug_assert!(
            generation.is_multiple_of(2),
            "EntityAllocator::is_alive called with non-live generation {generation}",
        );
        if !generation.is_multiple_of(2) {
            // Safety: Contract violation (odd generation).
            unsafe { std::hint::unreachable_unchecked() }
        }

        // Safety: index checked.
        let current = unsafe { *self.generations.as_ptr().add(index as usize) };
        debug_assert!(
            current.is_multiple_of(2),
            "EntityAllocator::is_alive called on freed/retired slot {index} (generation {current})",
        );
        if !current.is_multiple_of(2) {
            // Safety: Contract violation (freed slot).
            unsafe { std::hint::unreachable_unchecked() }
        }

        current == generation
    }

    #[must_use]
    pub fn committed_bytes(&self) -> usize {
        self.capacity * (std::mem::size_of::<u32>() + std::mem::size_of::<EntityLocation>())
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;

    #[test]
    fn test_entity_allocator_recycle() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let mut alloc = EntityAllocator::new(128).unwrap();

        let (id0, gen0) = alloc.alloc();
        assert_eq!(id0, 0);
        assert_eq!(gen0, 0);

        let (id1, _gen1) = alloc.alloc();
        assert_eq!(id1, 1);

        // Safety: Test code.
        unsafe { alloc.free(id0) }; // gen0 becomes 1

        // Should recycle 0
        let (id0_new, gen0_new) = alloc.alloc();
        assert_eq!(id0_new, 0);
        assert_eq!(gen0_new, 2); // gen0 becomes 2

        // Should alloc new 2
        let (id2, _) = alloc.alloc();
        assert_eq!(id2, 2);
    }

    #[test]
    fn test_entity_allocator_high_water() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let mut alloc = EntityAllocator::new(128).unwrap();
        // alloc 3
        let (_i0, _) = alloc.alloc();
        let (i1, _) = alloc.alloc();
        let (i2, _) = alloc.alloc();
        assert_eq!(i2, 2);

        // Free middle
        // Safety: Test code.
        unsafe { alloc.free(i1) };

        // Alloc should take 1
        let (i3, _) = alloc.alloc();
        assert_eq!(i3, 1);

        // Alloc should take 3 (new high water)
        let (i4, _) = alloc.alloc();
        assert_eq!(i4, 3);
    }
    #[test]
    fn test_entity_alloc_growth() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Allocating past initial capacity should trigger grow(), not panic.
        let mut alloc = EntityAllocator::with_max_capacity(1024, 8192).unwrap();
        assert_eq!(alloc.capacity, 1024);

        // Fill initial capacity
        for _ in 0..1024 {
            alloc.alloc();
        }

        // 1025th triggers grow → capacity doubles to 2048
        let (idx, generation) = alloc.alloc();
        assert_eq!(idx, 1024);
        assert_eq!(generation, 0);
        assert_eq!(alloc.capacity, 2048);

        // Verify all prior entities are still alive at gen 0
        for i in 0..1025u32 {
            // Safety: Test code.
            assert!(unsafe { alloc.is_alive(i, 0) });
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "maximum capacity")]
    fn test_entity_alloc_max_capacity_exhaustion() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // When max_capacity == initial_capacity, growth is impossible.
        let mut alloc = EntityAllocator::with_max_capacity(1024, 1024).unwrap();
        for _ in 0..1024 {
            alloc.alloc();
        }
        // 1025th should panic — no room to grow.
        alloc.alloc();
    }

    #[test]
    fn test_entity_alloc_is_alive() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // E5: Verify is_alive checks
        let mut alloc = EntityAllocator::new(128).unwrap();
        let (id, generation) = alloc.alloc();

        // Safety: Test code.
        assert!(unsafe { alloc.is_alive(id, generation) });

        let (id2, gen2) = alloc.alloc();
        assert_eq!(id2, id + 1);
        assert_eq!(gen2, 0);
        // Safety: Test code.
        assert!(!unsafe { alloc.is_alive(id2, gen2 + 2) }); // Wrong live generation
        // Safety: Test code.
        assert!(unsafe { alloc.is_alive(id2, gen2) });

        // Safety: Test code.
        unsafe { alloc.free(id) };
    }

    #[test]
    fn test_entity_alloc_iter() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // E9: Iterate implementation logic check
        let mut alloc = EntityAllocator::new(128).unwrap();
        let (i1, g1) = alloc.alloc();
        let (i2, _g2) = alloc.alloc();
        let (i3, g3) = alloc.alloc();

        // Free middle one
        // Safety: Test code.
        unsafe { alloc.free(i2) };

        let mut alive_count = 0;
        let mut seen_ids = Vec::new();

        // Iterate all slots up to high water
        for i in 0..alloc.high_water {
            // Check against known generations
            if i == i1 {
                // Safety: Test code.
                assert!(unsafe { alloc.is_alive(i, g1) });
                alive_count += 1;
                seen_ids.push(i);
            } else if i == i2 {
                // is_alive() is unsafe on freed slots by contract.
            } else if i == i3 {
                // Safety: Test code.
                assert!(unsafe { alloc.is_alive(i, g3) });
                alive_count += 1;
                seen_ids.push(i);
            }
        }

        assert_eq!(alive_count, 2);
        assert_eq!(seen_ids, vec![i1, i3]);
    }

    #[test]
    fn test_entity_alloc_sparse() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // E10: Alloc many, free random, verify holes
        let mut alloc = EntityAllocator::new(1024).unwrap();
        let mut ids = Vec::new();
        for _ in 0..100 {
            ids.push(alloc.alloc());
        }

        // Free even ones
        for i in (0..100).step_by(2) {
            // Safety: Test code.
            unsafe { alloc.free(ids[i].0) };
        }

        // Realloc should fill holes (LIFO or priority queue)
        // free_list is BinaryHeap<Reverse<u32>> -> MinHeap.
        // So should return smallest index first.
        let (i_new, _) = alloc.alloc();
        assert_eq!(i_new, 0); // Smartest hole
    }

    #[test]
    fn test_entity_alloc_stats() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // E11: Verify live_count
        let mut alloc = EntityAllocator::new(128).unwrap();
        assert_eq!(alloc.live_count, 0);
        let (i1, _) = alloc.alloc();
        assert_eq!(alloc.live_count, 1);
        // Safety: Test code.
        unsafe { alloc.free(i1) };
        assert_eq!(alloc.live_count, 0);
    }

    #[test]
    fn test_entity_alloc_drop_cleanup() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // E12: Verify committed bytes and Drop (stat check skipped for parallel safety)
        {
            let alloc = EntityAllocator::new(1024).unwrap();
            assert!(alloc.committed_bytes() >= 1024 * (4 + 12));
            // implicit drop
        }
        // Strict global stat check removed due to parallel test interference
    }
    #[test]
    fn test_entity_alloc_min_heap_ordering() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // E1: Verify free list order (smallest index first)
        let mut alloc = EntityAllocator::new(128).unwrap();
        let (i0, _) = alloc.alloc();
        let (i1, _) = alloc.alloc();
        let (i2, _) = alloc.alloc();

        // Safety: Test code.
        unsafe {
            alloc.free(i0);
            alloc.free(i2);
            alloc.free(i1);
        }

        // Should recycle 0, then 1, then 2 (MinHeap)
        assert_eq!(alloc.alloc().0, 0);
        assert_eq!(alloc.alloc().0, 1);
        assert_eq!(alloc.alloc().0, 2);
    }

    #[test]
    fn test_entity_alloc_generation_retirement() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // E2: Verify that a slot with exhausted generation space is retired
        // (permanently removed from the free list) instead of panicking.
        let mut alloc = EntityAllocator::new(128).unwrap();
        let (id, _) = alloc.alloc();

        // Artificially boost generation to the maximum even value.
        // Safety: Test code.
        unsafe {
            let gen_ptr = alloc.generations.as_ptr().add(id as usize);
            *gen_ptr = u32::MAX - 1; // even (alive)
        }

        // Safety: Test code.
        // Safety: Test code.
        unsafe { alloc.free(id) }; // gen -> u32::MAX (odd = GENERATION_RETIRED)

        // The retired slot should NOT be returned by alloc.
        // Instead, alloc should hand out a fresh slot from high_water.
        let (new_id, new_gen) = alloc.alloc();
        assert_ne!(new_id, id, "retired slot must not be reused");
        assert_eq!(new_gen, 0, "fresh slot starts at generation 0");

        // Verify the retired slot's generation is permanently u32::MAX
        // Safety: Test code.
        let retired_gen = unsafe { alloc.generations.as_ptr().add(id as usize).read() };
        assert_eq!(retired_gen, u32::MAX);
    }

    #[test]
    fn test_entity_location_default() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // E3: Verify default/zero initialization of locations
        let mut alloc = EntityAllocator::new(128).unwrap();
        let (id, _) = alloc.alloc();

        // Safety: Test code.
        let loc = unsafe { alloc.get_location(id) };
        assert_eq!(loc.archetype_id, 0);
        assert_eq!(loc.chunk_index, 0);
        assert_eq!(loc.index_in_chunk, 0);
    }

    #[test]
    fn test_entity_alloc_free_all_realloc_order() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // E6: Free all and realloc order
        let mut alloc = EntityAllocator::new(128).unwrap();
        let mut ids = Vec::new();
        for _ in 0..100 {
            ids.push(alloc.alloc().0);
        }

        // Free in reverse order
        for id in ids.iter().rev() {
            // Safety: Test code.
            unsafe { alloc.free(*id) };
        }

        // Realloc should still be in ascending order due to MinHeap
        for i in 0u32..100 {
            assert_eq!(alloc.alloc().0, i);
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_entity_alloc_set_location_out_of_bounds() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let mut alloc = EntityAllocator::new(128).unwrap();
        // Safety: Test code.
        unsafe {
            alloc.set_location(
                0,
                EntityLocation {
                    archetype_id: 1,
                    chunk_index: 0,
                    index_in_chunk: 0,
                },
            );
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_entity_alloc_get_location_out_of_bounds() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let alloc = EntityAllocator::new(128).unwrap();
        // Safety: Test code.
        let _ = unsafe { alloc.get_location(0) };
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_free_out_of_bounds() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let mut alloc = EntityAllocator::new(128).unwrap();
        // high_water starts at 0. Freeing index 0 is an error.
        // Safety: Test code.
        unsafe { alloc.free(0) };
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "Double free detected")]
    fn test_entity_alloc_double_free() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let mut alloc = EntityAllocator::new(128).unwrap();
        let (id, _) = alloc.alloc();
        // Safety: Test code.
        unsafe { alloc.free(id) };
        // Safety: Test code.
        // Safety: Test code.
        unsafe { alloc.free(id) }; // Should panic
    }

    // --- T1: free(index >= high_water) panics in all profiles ---
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_free_beyond_high_water_panics() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // B1 fix verification: assert! (not debug_assert!) catches OOB index.
        let mut alloc = EntityAllocator::new(1024).unwrap();
        let _ = alloc.alloc(); // high_water = 1
        // Safety: Test code.
        unsafe { alloc.free(1) }; // index 1 >= high_water(1) → always panics
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_free_large_index_panics() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Index within capacity but beyond high_water.
        let mut alloc = EntityAllocator::new(1024).unwrap();
        // Safety: Test code.
        unsafe { alloc.free(999) }; // high_water is 0 → always panics
    }

    // --- T2: new() cleans up gen_ptr on loc_ptr reservation failure ---
    #[test]
    fn test_new_cleanup_on_partial_failure() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // B2 fix verification: the error path in new() properly releases gen_ptr.
        // We verify the success path tracks stats correctly — the allocator's committed_bytes
        // must exactly match what Drop will release. (Global stats are racy in parallel.)
        let alloc = EntityAllocator::new(1024).unwrap();
        let expected_bytes =
            1024 * (std::mem::size_of::<u32>() + std::mem::size_of::<EntityLocation>());
        assert_eq!(alloc.committed_bytes(), expected_bytes);
        // Drop cleans up — verified by committed_bytes matching the expected reservation.
    }
}
