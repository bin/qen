//! Flux-proportional remote-free channel (roadmap item 4A).
//!
//! Frees always land in the freeing thread's local cache (that fast path is
//! untouched). What this module replaces is the *transfer channel* for the
//! capacity that must flow between threads — previously the globally-shared
//! recycler shards every thread CASes, now commutative per-block bitmasks:
//!
//! - A releasing thread returns a bin to its home block by `fetch_or`-ing
//!   one bit in the block's mask. OR is commutative: arbitrary fan-in
//!   cannot livelock, and repeated releases into the same block keep the
//!   mask line in the releasing core's cache, so the amortized coherence
//!   cost is below one miss per bin wherever there is any block locality.
//! - The block's owner reconciles with one `swap(0)` per mask word covering
//!   up to 64 bins, then *computes* the bin addresses from the bits —
//!   sequential and prefetchable, unlike walking a cold intrusive chain.
//! - Owners learn which blocks have pending returns through a per-owner
//!   intrusive Treiber list of *block* entries (`next_dirty`), so the
//!   notification rate is per-block-transition, not per-free.
//!
//! From the pool's perspective, bins parked in masks are still allocated
//! (exactly like bins parked in thread caches or the recycler), so block
//! `free_count`/decommit accounting is untouched: a block cannot decommit
//! while mask bits reference it because those bins were never freed to the
//! pool.
//!
//! The recycler remains the escape hatch for every case the masks decline:
//! unowned blocks, dead owners, self-owned overflow, and (under loom) the
//! capped model tables.
//!
//! # Ordering
//!
//! Two invariants carry the protocol, both checked by loom models in
//! `loom_tests.rs`:
//!
//! 1. **No stranded publishes.** Both sides touch the dirty flag with RMWs
//!    (`fetch_or` to set, `fetch_and` to clear), never plain loads. Atomic
//!    RMWs read the latest value in the owner word's modification order,
//!    so a publisher whose mask bit lands after the owner's mask swap
//!    necessarily sees the owner's earlier dirty-clear and re-publishes —
//!    the load-based version of this check would strand bins under exactly
//!    that interleaving.
//! 2. **Bin memory handoff.** The freeing thread's last writes to the bin
//!    must happen-before the next user's reads. That edge is the mask
//!    `fetch_or` (release side) → mask `swap` (acquire side) pair; the
//!    `SeqCst` used on these (and the dirty RMWs) is a superset chosen for
//!    cheap insurance on a cold path, but Release/Acquire is the
//!    load-bearing part.

use super::binned::ReciprocalDiv;
use crate::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, Ordering};
use std::ptr::NonNull;

/// Owner word per block: `(slot+1) << 48 | generation << 16 | flags`. Zero means
/// unowned. Storing `slot + 1` keeps the all-zero word as the unowned
/// sentinel without reserving registry slot 0.
const OWNER_DIRTY: u64 = 1;

#[inline]
fn pack_owner(slot: u16, generation: u32) -> u64 {
    (u64::from(slot) + 1) << 48 | u64::from(generation) << 16
}

#[inline]
fn owner_matches(word: u64, slot: u16, generation: u32) -> bool {
    word & !OWNER_DIRTY == pack_owner(slot, generation)
}

#[inline]
fn unpack_owner(word: u64) -> Option<(u16, u32)> {
    let slot_plus1 = (word >> 48) as u16;
    if slot_plus1 == 0 {
        return None;
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "extracting the packed 32-bit generation field"
    )]
    Some((slot_plus1 - 1, (word >> 16) as u32))
}

/// Registered owner slots. A `ThreadCache` claims a slot at bind and
/// releases it at flush-on-death; the generation counter makes stale
/// (slot, generation) pairs on block owner words detectably dead.
#[cfg(not(loom))]
pub(crate) const OWNER_SLOTS: usize = 512;
/// Loom models ≤ 3 threads; the tables are protocol-invariant.
#[cfg(loom)]
pub(crate) const OWNER_SLOTS: usize = 4;

/// Terminator / "not on a list" sentinel for `next_dirty` links and heads.
const DIRTY_END: u32 = u32::MAX;

pub(crate) struct OwnerSlot {
    /// `generation << 1 | live`.
    state: AtomicU64,
    /// Head of this owner's dirty-block Treiber list (packed
    /// `gid << 16 | block`, or `DIRTY_END`).
    dirty_head: AtomicU32,
}

/// Under loom, model-sized tables: the protocol is independent of table
/// width, and loom's object budget is not.
#[cfg(loom)]
pub(crate) const LOOM_BLOCK_CAP: usize = 2;
#[cfg(loom)]
pub(crate) const LOOM_WORDS_PER_BLOCK: usize = 1;

/// Lock-free side metadata for one `Pool`, published at pool creation and
/// immutable (except through its atomics) thereafter. Lives on the heap at
/// a stable address; the owning `RemoteMaskTable` drops it.
pub(crate) struct PoolRemote {
    pub class: u16,
    pub gid: u16,
    pub base: usize,
    pub bin_size: usize,
    block_recip: ReciprocalDiv,
    bin_recip: ReciprocalDiv,
    block_size: usize,
    words_per_block: usize,
    /// Covered block count (may be smaller than the pool's max under loom).
    blocks: usize,
    owners: Box<[AtomicU64]>,
    masks: Box<[AtomicU64]>,
    next_dirty: Box<[AtomicU32]>,
}

impl PoolRemote {
    pub fn new(
        class: u16,
        base: usize,
        bin_size: usize,
        block_size: usize,
        bins_per_block: usize,
        reserved_size: usize,
        (block_recip, bin_recip): (ReciprocalDiv, ReciprocalDiv),
    ) -> Self {
        let gid = 0; // assigned by `RemoteMaskTable::publish`
        let max_blocks = reserved_size / block_size;
        let words_per_block = bins_per_block.div_ceil(64);
        #[cfg(loom)]
        let (blocks, words_per_block) = (
            max_blocks.min(LOOM_BLOCK_CAP),
            words_per_block.min(LOOM_WORDS_PER_BLOCK),
        );
        #[cfg(not(loom))]
        let blocks = max_blocks;

        Self {
            class,
            gid,
            base,
            bin_size,
            block_recip,
            bin_recip,
            block_size,
            words_per_block,
            blocks,
            owners: zeroed_atomics_u64(blocks),
            masks: zeroed_atomics_u64(blocks * words_per_block),
            next_dirty: dirty_links(blocks),
        }
    }

    #[inline]
    pub(crate) fn locate(&self, ptr: NonNull<u8>) -> (usize, usize) {
        let offset = ptr.as_ptr() as usize - self.base;
        let block = self.block_recip.div(offset);
        let bin = self.bin_recip.div(offset - block * self.block_size);
        (block, bin)
    }

    /// Claim ownership of `block` for `(slot, generation)`, harvesting any bins
    /// already parked in its masks (they belong to whoever owns the block;
    /// leaving them would strand bins published toward the previous owner).
    /// Returns the harvested chain, if any.
    ///
    /// Racing publishes that read the old owner word may still set bits
    /// after this harvest and notify the old owner, who skips them on the
    /// ownership check; those bins wait for the next harvest or the trim
    /// sweep — bounded, and only during ownership handoff.
    pub fn claim_block(
        &self,
        block: usize,
        slot: u16,
        generation: u32,
        bin_links: impl Fn(NonNull<u8>, *mut u8),
    ) -> Option<(NonNull<u8>, NonNull<u8>, u32)> {
        if block >= self.blocks {
            return None;
        }
        let word = self.owners[block].load(Ordering::Acquire);
        if owner_matches(word, slot, generation) {
            return None; // already ours; masks flow through reconcile
        }
        // Take ownership (plain store is insufficient: the dirty bit is
        // concurrently RMW'd — preserve nothing, a fresh claim resets it;
        // publishes racing this store re-read the word afterwards or land
        // on the old list and are skipped there).
        self.owners[block].store(pack_owner(slot, generation), Ordering::SeqCst);
        self.drain_block_masks(block, &bin_links)
    }

    /// Swap out every mask word of `block`, building a plain chain of the
    /// bins found. Caller must have ownership (or hold the pool lock during
    /// trim, when caches are quiescent for this block's stragglers).
    fn drain_block_masks(
        &self,
        block: usize,
        bin_links: &impl Fn(NonNull<u8>, *mut u8),
    ) -> Option<(NonNull<u8>, NonNull<u8>, u32)> {
        let mut head: Option<NonNull<u8>> = None;
        let mut tail: Option<NonNull<u8>> = None;
        let mut count = 0u32;
        let words = &self.masks[block * self.words_per_block..][..self.words_per_block];
        for (w, word) in words.iter().enumerate() {
            if word.load(Ordering::Relaxed) == 0 {
                continue;
            }
            let mut bits = word.swap(0, Ordering::SeqCst);
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let bin = w * 64 + bit;
                let addr = self.base + block * self.block_size + bin * self.bin_size;
                // Safety: the bit was set by a publish for exactly this
                // bin, which was a live cache-held allocation; the base
                // arithmetic stays inside the pool reservation.
                let nn = unsafe { NonNull::new_unchecked(addr as *mut u8) };
                bin_links(nn, tail.map_or(std::ptr::null_mut(), NonNull::as_ptr));
                if tail.is_none() {
                    head = Some(nn);
                }
                tail = Some(nn);
                count += 1;
            }
        }
        // Chain was built newest-link-first: `bin_links(nn, prev)` writes
        // nn -> prev, so `tail` above is actually the chain head. Swap.
        head.map(|h| (tail.unwrap(), h, count))
    }
}

/// Zero-initialized atomic array without touching the pages (masks for the
/// smallest class reach ~2 MiB per pool; committing them eagerly would
/// defeat the point of a lazily-paged side table).
#[cfg(not(loom))]
fn zeroed_atomics_u64(n: usize) -> Box<[AtomicU64]> {
    // Safety: AtomicU64 is layout-compatible with u64 and the all-zero bit
    // pattern is a valid initialized value; Box<[u64]> -> Box<[AtomicU64]>
    // preserves length and allocation layout.
    unsafe {
        let z: Box<[u64]> = vec![0u64; n].into_boxed_slice();
        Box::from_raw(Box::into_raw(z) as *mut [AtomicU64])
    }
}

#[cfg(loom)]
fn zeroed_atomics_u64(n: usize) -> Box<[AtomicU64]> {
    (0..n).map(|_| AtomicU64::new(0)).collect()
}

fn dirty_links(n: usize) -> Box<[AtomicU32]> {
    (0..n).map(|_| AtomicU32::new(DIRTY_END)).collect()
}

/// The allocator-wide table: lock-free base→`PoolRemote` map, the global
/// pool id table, and the owner registry.
pub(crate) struct RemoteMaskTable {
    /// Open-addressed map keyed by pool base (reserved-size aligned).
    /// Insert-only under each class's pool lock; reads lock-free.
    map_keys: Box<[AtomicU64]>,
    map_vals: Box<[AtomicPtr<PoolRemote>]>,
    /// `gid` → `PoolRemote`, append-only.
    pool_table: Box<[AtomicPtr<PoolRemote>]>,
    slots: Box<[OwnerSlot]>,
    /// `!(pool_reserved_size - 1)` — masks a bin pointer to its pool base.
    base_mask: usize,
}

/// Map capacity: total pools across all classes is bounded by address
/// space (each pool reserves `pool_reserved_size`); 1024 is far beyond any
/// real configuration and keeps probes short.
#[cfg(not(loom))]
const MAP_CAP: usize = 1024;
/// Loom: enough for the pools whole-allocator models actually touch.
#[cfg(loom)]
const MAP_CAP: usize = 128;

impl RemoteMaskTable {
    pub fn new(pool_reserved_size: usize) -> Self {
        Self {
            map_keys: zeroed_atomics_u64_eager(MAP_CAP),
            map_vals: (0..MAP_CAP)
                .map(|_| AtomicPtr::new(std::ptr::null_mut()))
                .collect(),
            pool_table: (0..MAP_CAP)
                .map(|_| AtomicPtr::new(std::ptr::null_mut()))
                .collect(),
            slots: (0..OWNER_SLOTS)
                .map(|_| OwnerSlot {
                    state: AtomicU64::new(0),
                    dirty_head: AtomicU32::new(DIRTY_END),
                })
                .collect(),
            base_mask: !(pool_reserved_size - 1),
        }
    }

    /// Publish a pool's side table. Called under that class's pool lock
    /// (creation site), so duplicate publishes for one base cannot race.
    /// Returns the assigned gid.
    pub fn publish(&self, mut remote: PoolRemote, gid_counter: &AtomicU64) -> u16 {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "pool count is bounded by MAP_CAP, far below u16::MAX"
        )]
        let gid = gid_counter.fetch_add(1, Ordering::Relaxed) as u16;
        assert!((gid as usize) < MAP_CAP, "pool table exhausted");
        remote.gid = gid;
        let base = remote.base;
        let ptr = Box::into_raw(Box::new(remote));
        self.pool_table[gid as usize].store(ptr, Ordering::Release);

        let mut i = self.probe_start(base);
        loop {
            let k = &self.map_keys[i];
            if k.load(Ordering::Acquire) == 0
                && k.compare_exchange(0, base as u64, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
            {
                self.map_vals[i].store(ptr, Ordering::Release);
                return gid;
            }
            i = (i + 1) % MAP_CAP;
        }
    }

    #[inline]
    #[expect(
        clippy::unused_self,
        reason = "conceptually per-table (keyed by its map)"
    )]
    fn probe_start(&self, base: usize) -> usize {
        // Fibonacci scramble of the aligned base.
        (base.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 52) % MAP_CAP
    }

    #[inline]
    pub fn lookup(&self, ptr: NonNull<u8>) -> Option<&PoolRemote> {
        let base = ptr.as_ptr() as usize & self.base_mask;
        let mut i = self.probe_start(base);
        loop {
            let k = self.map_keys[i].load(Ordering::Acquire);
            if k == base as u64 {
                let v = self.map_vals[i].load(Ordering::Acquire);
                if v.is_null() {
                    return None; // key published, value racing in: decline
                }
                // Safety: values are published once and live until the
                // table drops (with the allocator, after all use).
                return Some(unsafe { &*v });
            }
            if k == 0 {
                return None;
            }
            i = (i + 1) % MAP_CAP;
        }
    }

    /// Claim an owner slot; returns `(slot, generation)`.
    pub fn claim_slot(&self) -> Option<(u16, u32)> {
        for (i, s) in self.slots.iter().enumerate() {
            let st = s.state.load(Ordering::Relaxed);
            if st & 1 == 0
                && s.state
                    .compare_exchange(st, st + 2 + 1, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
            {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "OWNER_SLOTS and the 32-bit generation both fit their fields"
                )]
                return Some((i as u16, ((st >> 1) as u32).wrapping_add(1)));
            }
        }
        None
    }

    /// Release a slot: bump the generation (kills the (slot, generation) pairs on
    /// block owner words) and hand any dirty entries that raced in to
    /// `orphan` (bins go back through the recycler/pool path).
    pub fn release_slot(
        &self,
        slot: u16,
        generation: u32,
        bin_links: impl Fn(NonNull<u8>, *mut u8),
        mut orphan: impl FnMut(u16, NonNull<u8>, NonNull<u8>, u32),
    ) {
        let s = &self.slots[slot as usize];
        crate::qen_debug_assert_eq!(s.state.load(Ordering::Relaxed) >> 1, u64::from(generation));
        s.state.fetch_add(1, Ordering::AcqRel); // live 1 -> 0, generation half-bumped
        // Publishes checked liveness before pushing; drain what landed.
        self.drain_dirty_list(slot, generation, &bin_links, &mut orphan);
    }

    /// Push a dirty-block entry onto `slot`'s list. `entry` packs
    /// `gid << 16 | block`.
    fn push_dirty(&self, slot: u16, pr: &PoolRemote, block: usize) {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "block < blocks <= reserved/block_size <= u16 range by pool construction"
        )]
        let entry = u32::from(pr.gid) << 16 | block as u32;
        let head = &self.slots[slot as usize].dirty_head;
        let mut cur = head.load(Ordering::Relaxed);
        loop {
            pr.next_dirty[block].store(cur, Ordering::Release);
            match head.compare_exchange_weak(cur, entry, Ordering::Release, Ordering::Relaxed) {
                Ok(_) => return,
                Err(actual) => cur = actual,
            }
        }
    }

    /// Try to return one bin through the mask channel. Returns `false` if
    /// the caller must fall back to the recycler (unowned block, dead or
    /// self owner, or outside the covered/model range).
    pub fn publish_bin(&self, ptr: NonNull<u8>, my_slot: u16, my_gen: u32) -> bool {
        let Some(pr) = self.lookup(ptr) else {
            return false;
        };
        let (block, bin) = pr.locate(ptr);
        if block >= pr.blocks || bin / 64 >= pr.words_per_block {
            return false; // loom-capped model range
        }
        let word = pr.owners[block].load(Ordering::Acquire);
        let Some((slot, generation)) = unpack_owner(word) else {
            return false;
        };
        if slot == my_slot && generation == my_gen {
            return false; // self-owned overflow: recycler, as before
        }
        // Owner must be live with the same generation, else the bins would
        // strand behind a dead slot until the trim sweep.
        let st = self.slots[slot as usize].state.load(Ordering::Acquire);
        if st & 1 == 0 || (st >> 1) != u64::from(generation) {
            return false;
        }

        let midx = block * pr.words_per_block + bin / 64;
        pr.masks[midx].fetch_or(1 << (bin % 64), Ordering::SeqCst);

        // Publish the block if it is not already on the owner's list. The
        // SeqCst pairing with the owner's dirty-clear + mask-swap is what
        // prevents a stranded set-after-swap (see module docs).
        let prev = pr.owners[block].fetch_or(OWNER_DIRTY, Ordering::SeqCst);
        if prev & OWNER_DIRTY == 0 {
            // Re-derive the owner from the word we RMW'd: ownership may
            // have moved between our load and the fetch_or; the entry must
            // go to whoever the word said at dirty-set time.
            if let Some((cur_slot, _)) = unpack_owner(prev) {
                self.push_dirty(cur_slot, pr, block);
            }
        }
        true
    }

    /// Reconcile all pending dirty blocks for `(slot, generation)`. For each block
    /// still owned by the caller, drains its masks into a chain handed to
    /// `sink(class, head, tail, count)`.
    pub fn reconcile(
        &self,
        slot: u16,
        generation: u32,
        bin_links: impl Fn(NonNull<u8>, *mut u8),
        mut sink: impl FnMut(u16, NonNull<u8>, NonNull<u8>, u32),
    ) {
        self.drain_dirty_list(slot, generation, &bin_links, &mut sink);
    }

    fn drain_dirty_list(
        &self,
        slot: u16,
        generation: u32,
        bin_links: &impl Fn(NonNull<u8>, *mut u8),
        sink: &mut impl FnMut(u16, NonNull<u8>, NonNull<u8>, u32),
    ) {
        let mut cur = self.slots[slot as usize]
            .dirty_head
            .swap(DIRTY_END, Ordering::Acquire);
        while cur != DIRTY_END {
            let gid = (cur >> 16) as usize;
            let block = (cur & 0xFFFF) as usize;
            let pr_ptr = self.pool_table[gid].load(Ordering::Acquire);
            crate::qen_debug_assert!(!pr_ptr.is_null(), "dirty entry for unpublished pool");
            // Safety: pool_table entries live until the allocator drops.
            let pr = unsafe { &*pr_ptr };
            // Capture the continuation BEFORE processing: clearing the
            // dirty flag lets a racing publish re-push this block, which
            // overwrites next_dirty — the old list's tail must not be lost.
            let next = pr.next_dirty[block].load(Ordering::Acquire);

            let word = pr.owners[block].load(Ordering::Acquire);
            if owner_matches(word, slot, generation) {
                // Clear dirty BEFORE swapping masks: a publish that lands
                // after our swap must find dirty == 0 so it re-notifies.
                pr.owners[block].fetch_and(!OWNER_DIRTY, Ordering::SeqCst);
                if let Some((h, t, n)) = pr.drain_block_masks(block, bin_links) {
                    sink(pr.class, h, t, n);
                }
            }
            // Not ours (ownership moved): skip. The new owner harvested at
            // claim time or will get its own notification.
            cur = next;
        }
    }

    /// Trim-time sweep of one pool's masks (caller holds that pool's lock).
    /// Returns bins to `f` so the caller can `pool.free()` them; only owned
    /// blocks can carry bits, so unowned blocks are skipped for free.
    pub fn sweep_pool(
        &self,
        base: usize,
        bin_links: impl Fn(NonNull<u8>, *mut u8),
        mut f: impl FnMut(NonNull<u8>, NonNull<u8>, u32),
    ) {
        // Safety-free lookup by exact base.
        let Some(pr) = self.lookup(
            // Safety: base is a live pool's non-null base address.
            unsafe { NonNull::new_unchecked(base as *mut u8) },
        ) else {
            return;
        };
        for block in 0..pr.blocks {
            if pr.owners[block].load(Ordering::Relaxed) == 0 {
                continue;
            }
            if let Some((h, t, n)) = pr.drain_block_masks(block, &bin_links) {
                f(h, t, n);
            }
        }
    }
}

/// Eagerly-touched variant for the small fixed tables (map keys).
fn zeroed_atomics_u64_eager(n: usize) -> Box<[AtomicU64]> {
    (0..n).map(|_| AtomicU64::new(0)).collect()
}

impl Drop for RemoteMaskTable {
    fn drop(&mut self) {
        for slot in &self.pool_table {
            let p = slot.load(Ordering::Relaxed);
            if !p.is_null() {
                // Safety: published exactly once via Box::into_raw; the
                // allocator (and all users) are gone when the table drops.
                drop(unsafe { Box::from_raw(p) });
            }
        }
    }
}
