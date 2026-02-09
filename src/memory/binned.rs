use super::stats;
use super::vm::{PlatformVmOps, VmError, VmOps};
#[cfg(all(test, not(loom)))]
use super::binned;
#[cfg(debug_assertions)]
use fixedbitset::FixedBitSet;
use std::ptr::NonNull;
use crate::sync::atomic::{AtomicU64, Ordering};
use crate::sync::hint;

pub(crate) const POOL_RESERVED_SIZE: usize = 256 * 1024 * 1024;

pub(crate) const MAX_SMALL_SIZE: usize = 65536;

/// Configuration for `BinnedAllocator`. All fields have sensible defaults.
/// Set at init time via `BinnedAllocator::with_config()`.
#[derive(Clone, Debug)]
pub struct BinnedAllocatorConfig {
    /// Max items in a TLS cache before flushing. Per bin-size tier:
    /// `[<=1KB, <=8KB, <=32KB, >32KB]`. Default: `[64, 32, 8, 4]`.
    pub cache_count_limits: [u32; 4],

    /// Number of extra bins to prefill into TLS cache on slow-path alloc.
    /// Per bin-size tier. Default: `[16, 8, 4, 2]`.
    pub alloc_extra: [u32; 4],

    /// Pool reserved VA size per size class. Default: 256 MB.
    pub pool_reserved_size: usize,

    /// Block size in bytes. Default: `max(64KB, system page size)`.
    pub block_size: usize,

    /// Enable immediate decommit of fully-empty blocks. Default: true.
    pub immediate_decommit: bool,

    /// Maximum bundles per size class in the lock-free `GlobalRecycler`.
    /// Higher values absorb more cross-thread traffic before falling through
    /// to the pool mutex, at the cost of holding more memory in the recycler.
    /// Default: 16.
    pub recycler_max_bundles: u32,

    /// Enable huge-page-backed large allocations in `LargeAllocCache`.
    /// When true, allocations >= 2MB attempt explicit huge pages
    /// (`alloc_huge`) with graceful fallback to regular pages.
    /// Default: true.
    pub use_huge_pages: bool,
}

impl Default for BinnedAllocatorConfig {
    fn default() -> Self {
        Self {
            cache_count_limits: [64, 32, 8, 4],
            alloc_extra: [16, 8, 4, 2],
            pool_reserved_size: POOL_RESERVED_SIZE,
            block_size: 0, // 0 = auto-detect (max(64KB, page_size))
            immediate_decommit: true,
            recycler_max_bundles: 16,
            use_huge_pages: true,
        }
    }
}

impl BinnedAllocatorConfig {
    /// Return tier index (0..3) for given bin size.
    fn tier(bin_size: usize) -> usize {
        if bin_size <= 1024 {
            0
        } else if bin_size <= 8192 {
            1
        } else if bin_size <= 32768 {
            2
        } else {
            3
        }
    }

    /// Max TLS cache items for given bin size.
    #[must_use]
    pub fn max_cache_for(&self, bin_size: usize) -> u32 {
        self.cache_count_limits[Self::tier(bin_size)]
    }

    /// Batch refill count for the given bin size.
    #[must_use]
    pub fn batch_size_for(&self, bin_size: usize) -> u32 {
        self.alloc_extra[Self::tier(bin_size)]
    }
}

// Each BitTree segment covers 16384 blocks. BitTreeChain chains multiple
// segments for pools that exceed the single-segment capacity.

const BIN_SENTINEL: u16 = 0xFFFF;

/// 8-bit canary value written into every `BlockMeta` on creation.
/// Checked on alloc and free to detect corruption.
/// Only active when debug assertions are enabled.
#[cfg(debug_assertions)]
const BLOCK_CANARY: u8 = 0xA5;

/// 32-bit canary written at offset 4 of every freed bin (freelist path only).
/// Checked on alloc to detect use-after-free and double-alloc corruption.
/// Only active when debug assertions are enabled.
#[cfg(debug_assertions)]
const FREE_CANARY: u32 = 0xAB_AD_BA_BE;

/// Hierarchical bitset segment for finding free blocks.
/// Each segment tracks up to [`BITTREE_CAPACITY`] (16384) blocks.
/// Multiple segments are chained via [`BitTreeChain`] to support larger pools.
pub(crate) struct BitTree {
    l0: u64,        // Covers 4 L1 words (256 bits total for L1)
    l1: [u64; 4],   // 256 bits. Each bit covers 64 blocks.
    l2: [u64; 256], // 16384 bits. Each bit covers 1 block.
}

impl BitTree {
    pub fn new() -> Self {
        Self {
            l0: 0,
            l1: [0; 4],
            l2: [0; 256],
        }
    }

    /// Mark block as having free space.
    pub fn mark_free(&mut self, block_index: usize) {
        let l2_word = block_index / 64;
        let l2_bit = block_index % 64;

        let l1_word = l2_word / 64;
        let l1_bit = l2_word % 64;

        let mask2 = 1u64 << l2_bit;
        if (self.l2[l2_word] & mask2) == 0 {
            self.l2[l2_word] |= mask2;

            let mask1 = 1u64 << l1_bit;
            if (self.l1[l1_word] & mask1) == 0 {
                self.l1[l1_word] |= mask1;

                let mask0 = 1u64 << l1_word;
                self.l0 |= mask0;
            }
        }
    }

    /// Mark block as full.
    pub fn mark_full(&mut self, block_index: usize) {
        let l2_word = block_index / 64;
        let l2_bit = block_index % 64;

        // Clear L2 bit
        self.l2[l2_word] &= !(1u64 << l2_bit);

        if self.l2[l2_word] == 0 {
            let l1_word = l2_word / 64;
            let l1_bit = l2_word % 64;

            // Clear L1 bit
            self.l1[l1_word] &= !(1u64 << l1_bit);

            if self.l1[l1_word] == 0 {
                let mask0 = 1u64 << l1_word;
                self.l0 &= !mask0;
            }
        }
    }

    /// Find first block with free space.
    pub fn find_free(&self) -> Option<usize> {
        if self.l0 == 0 {
            return None;
        }

        let l1_word = self.l0.trailing_zeros() as usize;
        let l1_bit = self.l1[l1_word].trailing_zeros() as usize;
        let l2_word = (l1_word * 64) + l1_bit;
        let l2_bit = self.l2[l2_word].trailing_zeros() as usize;

        Some((l2_word * 64) + l2_bit)
    }
}

/// Number of blocks tracked by single [`BitTree`] segment.
pub(crate) const BITTREE_CAPACITY: usize = 16384;

/// Growable chain of [`BitTree`] segments, each covering [`BITTREE_CAPACITY`] blocks.
///
/// Block indices are global: segment `i` covers blocks `[i*16384 .. (i+1)*16384)`.
/// New segments are allocated lazily when [`mark_free`](BitTreeChain::mark_free) is
/// called for an index beyond the current capacity.
/// [`find_free`](BitTreeChain::find_free) scans segments in order, preserving the
/// low-address-first allocation preference of a single `BitTree`.
pub(crate) struct BitTreeChain {
    trees: Vec<BitTree>,
    search_cursor: usize,
}

impl BitTreeChain {
    pub fn new() -> Self {
        Self { 
            trees: Vec::new(),
            search_cursor: 0,
        }
    }

    /// Ensure chain has enough segments to cover `block_index`.
    #[inline]
    fn ensure_capacity(&mut self, block_index: usize) {
        let tree_idx = block_index / BITTREE_CAPACITY;
        while self.trees.len() <= tree_idx {
            self.trees.push(BitTree::new());
        }
    }

    /// Mark block as having free space.
    pub fn mark_free(&mut self, block_index: usize) {
        self.ensure_capacity(block_index);
        let tree_idx = block_index / BITTREE_CAPACITY;
        let local_idx = block_index % BITTREE_CAPACITY;
        self.trees[tree_idx].mark_free(local_idx);

        if tree_idx < self.search_cursor {
            self.search_cursor = tree_idx;
        }
    }

    /// Mark block as full.
    pub fn mark_full(&mut self, block_index: usize) {
        let tree_idx = block_index / BITTREE_CAPACITY;
        debug_assert!(
            tree_idx < self.trees.len(),
            "mark_full on block {} but only {} segments exist",
            block_index,
            self.trees.len()
        );
        if tree_idx < self.trees.len() {
            let local_idx = block_index % BITTREE_CAPACITY;
            self.trees[tree_idx].mark_full(local_idx);
        }
    }

    /// Find first block with free space across all segments.
    /// Scans segments in order starting from `search_cursor`.
    pub fn find_free(&mut self) -> Option<usize> {
        // Optimization: start from search_cursor. If a tree is full,
        // it returns None and we advance the cursor to skip it next time.
        // This makes finding the first free block amortized O(1) even with many segments.
        let start = self.search_cursor;
        for i in start..self.trees.len() {
            if let Some(local_idx) = self.trees[i].find_free() {
                self.search_cursor = i;
                return Some(i * BITTREE_CAPACITY + local_idx);
            }
            // If tree i is full, next search can skip it.
            // Safe because we hold the lock (external pool lock).
            self.search_cursor = i + 1;
        }
        
        // No free blocks found.
        // Cursor remains at len() so future calls return immediately
        // until mark_free pulls it back.
        None
    }
}

/// Per-block metadata packed into 8 bytes (+ debug-only `FixedBitSet`).
///
/// Bit layout of `packed: u64`:
/// ```text
///   [63..56] canary      (8 bits)  — always BLOCK_CANARY (0xA5)
///   [55..49] reserved    (7 bits)
///   [48]     committed   (1 bit)   — 1 if block is backed by physical pages
///   [47..32] bump_cursor (16 bits) — next virgin slot for bump allocation
///   [31..16] free_head   (16 bits) — index of first free bin, or 0xFFFF sentinel
///   [15..0]  free_count  (16 bits) — number of free bins in this block
/// ```
///
/// 16-bit limits are sufficient: max `bins_per_block` = 65536/16 = 4096 for the
/// smallest bin size, well within u16 range. `BIN_SENTINEL` is 0xFFFF.
pub(crate) struct BlockMeta {
    packed: u64,
    #[cfg(debug_assertions)]
    pub free_map: FixedBitSet,
}

impl BlockMeta {
    /// Create new `BlockMeta`. `fresh` indicates if block is newly allocated
    /// from OS (guaranteed zeroed) or recycled (potentially dirty).
    pub fn new(free_count: u16, #[allow(unused)] bins_per_block: usize, _fresh: bool) -> Self {
        let mut packed: u64 = 0;
        // canary (debug builds only)
        #[cfg(debug_assertions)]
        {
            packed |= u64::from(BLOCK_CANARY) << 56;
        }
        // fresh = 1 if fresh

        // committed = 1
        packed |= 1u64 << 48;
        // bump_cursor = 0 (implicit)
        // free_head = SENTINEL
        packed |= u64::from(BIN_SENTINEL) << 16;
        // free_count
        packed |= u64::from(free_count);

        Self {
            packed,
            #[cfg(debug_assertions)]
            free_map: FixedBitSet::with_capacity(bins_per_block),
        }
    }

    #[inline]
    pub fn free_count(&self) -> u16 {
        (self.packed & 0xFFFF) as u16
    }

    #[inline]
    pub fn set_free_count(&mut self, v: u16) {
        self.packed = (self.packed & !0xFFFF) | u64::from(v);
    }

    #[inline]
    pub fn free_head(&self) -> u16 {
        ((self.packed >> 16) & 0xFFFF) as u16
    }

    #[inline]
    pub fn set_free_head(&mut self, v: u16) {
        self.packed = (self.packed & !(0xFFFF << 16)) | (u64::from(v) << 16);
    }

    #[inline]
    pub fn bump_cursor(&self) -> u16 {
        ((self.packed >> 32) & 0xFFFF) as u16
    }

    #[inline]
    pub fn set_bump_cursor(&mut self, v: u16) {
        self.packed = (self.packed & !(0xFFFF << 32)) | (u64::from(v) << 32);
    }

    #[inline]
    pub fn is_committed(&self) -> bool {
        (self.packed >> 48) & 1 == 1
    }

    #[inline]
    pub fn set_committed(&mut self, v: bool) {
        if v {
            self.packed |= 1u64 << 48;
        } else {
            self.packed &= !(1u64 << 48);
        }
    }

    /// Verify canary byte. Panics on corruption.
    /// No-op when debug assertions disabled.
    #[inline]
    pub fn check_canary(&self) {
        #[cfg(debug_assertions)]
        {
            let canary = ((self.packed >> 56) & 0xFF) as u8;
            assert!(
                canary == BLOCK_CANARY,
                "BinnedAllocator corruption: block canary was 0x{canary:02x}, expected 0x{BLOCK_CANARY:02x}",
            );
        }
    }
}

/// Information needed to pre-commit block outside the pool lock.
/// Returned by `Pool::probe_commit_needed()`; consumed by `Pool::integrate_precommit()`.
pub(crate) struct PreCommitRequest {
    pub ptr: NonNull<u8>,
    pub size: usize,
    pub block_idx: usize,
    pub is_new_block: bool,
}

// Safety: PreCommitRequest owns the pointer and is safe to send between threads.
unsafe impl Send for PreCommitRequest {}

pub(crate) struct Pool {
    pub bin_size: usize,
    pub block_size: usize,
    pub bins_per_block: u16,
    pub base: NonNull<u8>,
    pub committed: usize,
    pub reserved_size: usize,
    pub immediate_decommit: bool,
    pub bit_tree: BitTreeChain,
    pub blocks: Vec<BlockMeta>,
    /// Block indices pending decommit. Populated by `free()` when block
    /// becomes fully empty (instead of decommitting inline). Drained by
    /// `process_pending_decommits()` which is called from `trim()`.
    /// Keeps decommit syscalls out of hot free path.
    decommit_pending: Vec<usize>,
}

// Safety: Pool owns the memory region and is safe to send between threads.
unsafe impl Send for Pool {}

impl Drop for Pool {
    fn drop(&mut self) {
        // Release the entire VM reservation
        // This implicitly assumes all committed pages are also released by the OS
        // when the mapping is removed.
        // Safety: We are dropping the pool, so we can release the memory.
        unsafe {
            drop(PlatformVmOps::release(self.base, self.reserved_size));
            stats::sub_saturating(&stats::TOTAL_RESERVED, self.reserved_size);
            // self.committed is kept accurate by alloc and trim operations.
            stats::sub_saturating(&stats::TOTAL_COMMITTED, self.committed);
            stats::sub_saturating(&stats::BINNED_ALLOCATOR_COMMITTED, self.committed);
        };
    }
}

impl Pool {
    /// Create new Pool with the given configuration.
    ///
    /// # Safety / Constraints
    /// - `bin_size` must be at least 16 bytes to support the `GlobalRecycler` link field.
    /// - `block_size` must be page-aligned.
    pub fn with_config(
        bin_size: usize,
        block_size: usize,
        config: &BinnedAllocatorConfig,
    ) -> Result<Self, VmError> {
        let bins_per_block = block_size / bin_size;
        if bins_per_block > u16::MAX as usize {
            return Err(VmError::InitializationFailed(format!(
                "block_size/bin_size overflow: {} bins per block exceeds u16::MAX ({})",
                bins_per_block,
                u16::MAX
            )));
        }

        let reserved_size = config.pool_reserved_size;

        // Validation: bin_size must be at least usize to hold linked list pointers
        // AND at least 16 to hold the GlobalRecycler link at offset 8.
        debug_assert!(
            bin_size >= 16,
            "bin_size {bin_size} is smaller than minimum required 16 (for recycler links)",
        );
        debug_assert!(
            bin_size >= std::mem::size_of::<usize>(),
            "bin_size {bin_size} is smaller than minimum required {}",
            std::mem::size_of::<usize>()
        );

        // Safety: FFI call to reserve memory.
        let ptr = unsafe { PlatformVmOps::reserve(reserved_size)? };

        stats::TOTAL_RESERVED.fetch_add(reserved_size, Ordering::Relaxed);

        Ok(Self {
            bin_size,
            block_size,
            bins_per_block: u16::from_le_bytes(bins_per_block.to_le_bytes()[..2].try_into().unwrap()),
            base: ptr,
            committed: 0,
            reserved_size,
            immediate_decommit: config.immediate_decommit,
            bit_tree: BitTreeChain::new(),
            blocks: Vec::new(),
            decommit_pending: Vec::new(),
        })
    }

    /// Check whether the next alloc would require a VM commit syscall.
    /// If so, returns `PreCommitRequest` that the caller can use to
    /// perform the commit *outside* the pool lock, then integrate via
    /// `integrate_precommit()` after re-acquiring the lock.
    ///
    /// Returns `None` if committed free block is available (no syscall needed).
    pub fn probe_commit_needed(&mut self) -> Option<PreCommitRequest> {
        if let Some(idx) = self.bit_tree.find_free() {
            if !self.blocks[idx].is_committed() {
                // Decommitted block needs recommit
                // Safety: idx is checked to be within bounds.
                let ptr = unsafe {
                    NonNull::new_unchecked(self.base.as_ptr().add(idx * self.block_size))
                };
                return Some(PreCommitRequest {
                    ptr,
                    size: self.block_size,
                    block_idx: idx,
                    is_new_block: false,
                });
            }
            None // Free committed block available
        } else {
            // No free blocks — need a new one
            if self.committed + self.block_size > self.reserved_size {
                return None; // OOM — alloc will return the error, no commit helps
            }
            let block_idx = self.blocks.len();
            // Safety: block_idx is valid.
            let ptr = unsafe {
                NonNull::new_unchecked(self.base.as_ptr().add(block_idx * self.block_size))
            };
            Some(PreCommitRequest {
                ptr,
                size: self.block_size,
                block_idx,
                is_new_block: true,
            })
        }
    }

    /// Integrate block that was pre-committed outside the pool lock.
    ///
    /// Returns `true` if the pre-commit was integrated into the pool metadata
    /// (the common case). Returns `false` if the pool state changed while the
    /// lock was released (another thread handled it). In the `false` case,
    /// the commit was harmless — the pages are still within the pool's reserved
    /// VA and will be used eventually or released on pool drop.
    pub fn integrate_precommit(&mut self, req: &PreCommitRequest) -> bool {
        if req.is_new_block {
            // Only integrate if no other thread added blocks in the interim
            if self.blocks.len() == req.block_idx {
                // Dev mode: zero the block under the lock. The commit()
                // outside the lock was just mprotect — zeroing must happen
                // here to avoid racing with concurrent allocations.
                // Safety: ptr is valid and size is correct.
                #[cfg(debug_assertions)]
                unsafe {
                    std::ptr::write_bytes(req.ptr.as_ptr(), 0, req.size);
                }

                self.blocks
                    .push(BlockMeta::new(self.bins_per_block, self.bins_per_block as usize, true));
                self.committed += self.block_size;
                stats::TOTAL_COMMITTED.fetch_add(self.block_size, Ordering::Relaxed);
                stats::BINNED_ALLOCATOR_COMMITTED.fetch_add(self.block_size, Ordering::Relaxed);
                self.bit_tree.mark_free(req.block_idx);
                return true;
            }
            false
        } else {
            // Recommit: only integrate if block is still decommitted
            if req.block_idx < self.blocks.len() && !self.blocks[req.block_idx].is_committed() {
                // Dev mode: zero under the lock (same rationale as new_block).
                // Safety: ptr is valid and size is correct.
                #[cfg(debug_assertions)]
                unsafe {
                    std::ptr::write_bytes(req.ptr.as_ptr(), 0, req.size);
                }

                let block = &mut self.blocks[req.block_idx];
                *block = BlockMeta::new(self.bins_per_block, self.bins_per_block as usize, false);
                self.committed += self.block_size;
                stats::TOTAL_COMMITTED.fetch_add(self.block_size, Ordering::Relaxed);
                stats::BINNED_ALLOCATOR_COMMITTED.fetch_add(self.block_size, Ordering::Relaxed);
                return true;
            }
            false
        }
    }

    /// Process pending decommit requests. Each queued block is re-validated
    /// (still fully empty and committed) before the decommit syscall fires.
    /// Blocks that were re-allocated between queueing and processing are
    /// silently skipped.
    pub fn process_pending_decommits(&mut self) {
        if self.decommit_pending.is_empty() {
            return;
        }
        let pending = std::mem::take(&mut self.decommit_pending);
        let bins_per_block = self.bins_per_block;

        for block_idx in pending {
            if block_idx >= self.blocks.len() {
                continue;
            }
            let block = &self.blocks[block_idx];
            // Only decommit if block is still fully empty and committed
            if block.free_count() == bins_per_block && block.is_committed() {
                // Safety: block_idx is valid.
                let block_ptr = unsafe {
                    NonNull::new_unchecked(self.base.as_ptr().add(block_idx * self.block_size))
                };
                // Safety: FFI call to decommit memory.
                if unsafe { PlatformVmOps::decommit(block_ptr, self.block_size) }.is_ok() {
                    self.committed -= self.block_size;
                    stats::sub_saturating(&stats::TOTAL_COMMITTED, self.block_size);
                    stats::sub_saturating(&stats::BINNED_ALLOCATOR_COMMITTED, self.block_size);
                    let block = &mut self.blocks[block_idx];
                    block.set_committed(false);
                }
            }
        }
    }

    /// Allocate a bin from the pool (thread-safe lock required by caller).
    ///
    /// Returns `pointer`.
    ///
    /// Zeroing behavior:
    /// - **Debug**: all allocations are guaranteed zeroed.
    /// - **Release**: undefined content.
    pub fn alloc(&mut self) -> Result<NonNull<u8>, VmError> {
        let block_idx = if let Some(idx) = self.bit_tree.find_free() {
            let block = &self.blocks[idx];
            // If the block was decommitted (sparse decommit), recommit it.
            if !block.is_committed() {
                let block_offset = idx * self.block_size;
                // Safety: block_offset is within reserved range.
                // Safety: block_offset is within reserved range.
            let ptr = unsafe { NonNull::new_unchecked(self.base.as_ptr().add(block_offset)) };
                // Safety: FFI call to commit memory.
                // Safety: FFI call to commit memory.
            unsafe { PlatformVmOps::commit(ptr, self.block_size)? };
                // Debug mode: zero the block under the pool lock to guarantee
                // deterministic behavior. This is safe because we hold the
                // lock and the block has no live allocations (was fully empty).
                // Safety: ptr is valid.
                #[cfg(debug_assertions)]
                unsafe {
                    std::ptr::write_bytes(ptr.as_ptr(), 0, self.block_size);
                }
                self.committed += self.block_size;
                stats::TOTAL_COMMITTED.fetch_add(self.block_size, Ordering::Relaxed);
                stats::BINNED_ALLOCATOR_COMMITTED.fetch_add(self.block_size, Ordering::Relaxed);

                let block = &mut self.blocks[idx];
                // Re-initialize: block with bump allocation
                *block = BlockMeta::new(self.bins_per_block, self.bins_per_block as usize, false);
            }
            idx
        } else {
            // No free blocks, commit new one
            if self.committed + self.block_size > self.reserved_size {
                return Err(VmError::CommitFailed(std::io::Error::new(
                    std::io::ErrorKind::OutOfMemory,
                    "Pool exhausted",
                )));
            }

            let block_idx = self.blocks.len();
            let block_offset = block_idx * self.block_size;
            // Safety: block_offset is within reserved range.
            let ptr = unsafe { NonNull::new_unchecked(self.base.as_ptr().add(block_offset)) };

            // Safety: FFI call to commit memory.
            unsafe { PlatformVmOps::commit(ptr, self.block_size)? };
            // Debug mode: zero the block under the pool lock.
            // Safety: ptr is valid.
            #[cfg(debug_assertions)]
            unsafe {
                std::ptr::write_bytes(ptr.as_ptr(), 0, self.block_size);
            }
            self.committed += self.block_size;

            stats::TOTAL_COMMITTED.fetch_add(self.block_size, Ordering::Relaxed);
            stats::BINNED_ALLOCATOR_COMMITTED.fetch_add(self.block_size, Ordering::Relaxed);

            // No eager freelist initialization — we use a bump pointer for virgin blocks.
            // This avoids touching every cache line on block commit.
            self.blocks
                .push(BlockMeta::new(self.bins_per_block, self.bins_per_block as usize, true));

            self.bit_tree.mark_free(block_idx);
            block_idx
        };

        let block = &mut self.blocks[block_idx];
        block.check_canary();
        let bins_per_block = self.bins_per_block;

        let bin_idx: u16;
        let bin_ptr: *mut u8;

        if block.free_head() == BIN_SENTINEL {
            // Bump path: allocate from space not yet touched by the allocator.
            debug_assert!(block.bump_cursor() < bins_per_block);
            bin_idx = block.bump_cursor();
            block.set_bump_cursor(bin_idx + 1);
            let block_offset = block_idx * self.block_size;
            let bin_offset = block_offset + (bin_idx as usize * self.bin_size);
            // Safety: bin_offset is within valid block.
            bin_ptr = unsafe { self.base.as_ptr().add(bin_offset) };
        } else {
            // Freelist path: recycle a previously-freed bin
            bin_idx = block.free_head();
            let block_offset = block_idx * self.block_size;
            let bin_offset = block_offset + (bin_idx as usize * self.bin_size);
            // Safety: bin_offset is within valid block.
            bin_ptr = unsafe { self.base.as_ptr().add(bin_offset) };

            // Verify free-bin canary at offset 4 (debug builds only)
            #[cfg(debug_assertions)]
            if self.bin_size >= 8 {
                // Safety: bin_ptr+4 is valid for reading canary.
                let canary = unsafe { *bin_ptr.add(4).cast::<()>().cast::<u32>() };
                assert!(
                    canary == FREE_CANARY,
                    "BinnedAllocator corruption: free-bin canary at {bin_ptr:p}+4 was 0x{canary:08x}, expected 0x{FREE_CANARY:08x}",
                );
            }

            // Follow the freelist: first 2 bytes of the bin hold the u16 next-free index
            // Safety: bin_ptr points to valid memory.
            let next_free = unsafe { *bin_ptr.cast::<()>().cast::<u16>() };
            block.set_free_head(next_free);

            // Safety: bin_ptr is valid.
            #[cfg(debug_assertions)]
            unsafe {
                std::ptr::write_bytes(bin_ptr, 0, self.bin_size);
            }
        }

        let fc = block.free_count() - 1;
        block.set_free_count(fc);
        if fc == 0 {
            self.bit_tree.mark_full(block_idx);
        }

        #[cfg(debug_assertions)]
        block.free_map.set(bin_idx as usize, false);

        // Safety: bin_ptr is non-null.
        Ok(unsafe { NonNull::new_unchecked(bin_ptr) })
    }

    pub fn free(&mut self, ptr: NonNull<u8>) {
        let ptr_addr = ptr.as_ptr() as usize;
        let base_addr = self.base.as_ptr() as usize;

        // Dev-mode validation checks (range, commitment, alignment, canary).
        // Disabled in release builds for maximum performance.
        #[cfg(debug_assertions)]
        {
            // Range check (must come first to prevent underflow in offset calc)
            assert!(ptr_addr >= base_addr && ptr_addr < base_addr + self.reserved_size, "Pointer {ptr:p} does not belong to this Pool");
        }

        let offset = ptr_addr - base_addr;
        let block_idx = offset / self.block_size;
        let offset_in_block = offset % self.block_size;
        let bin_idx = u16::from_le_bytes((offset_in_block / self.bin_size).to_le_bytes()[..2].try_into().unwrap());

        #[cfg(debug_assertions)]
        {
            // Commitment check
            assert!(
                block_idx < self.blocks.len(),
                "Pointer {ptr:p} belongs to an uncommitted block in this Pool",
            );
            assert!(
                self.blocks[block_idx].is_committed(),
                "Pointer {ptr:p} belongs to a decommitted block in this Pool (double-free after full block release?)",
            );

            // Alignment check
            assert!(
                offset_in_block.is_multiple_of(self.bin_size),
                "Pointer {ptr:p} is not aligned to bin size {}",
                self.bin_size
            );
        }

        let block = &mut self.blocks[block_idx];
        block.check_canary();

        // Double-free check (debug-only; production uses free-bin canary)
        #[cfg(debug_assertions)]
        {
            assert!(
                !block.free_map.contains(bin_idx as usize),
                "Double free detected in BinnedAllocator: block {block_idx}, bin {bin_idx}",
            );
            block.free_map.insert(bin_idx as usize);
        }

        // Write u16 next-free index at offset 0
        // Safety: ptr is valid.
        unsafe {
            *ptr.as_ptr().cast::<()>().cast::<u16>() = block.free_head();
        }
        // Write free-bin canary at offset 4 (debug builds only)
        #[cfg(debug_assertions)]
        if self.bin_size >= 8 {
            // Safety: ptr+4 is valid.
            unsafe {
                *ptr.as_ptr().add(4).cast::<()>().cast::<u32>() = FREE_CANARY;
            }
        }

        block.set_free_head(bin_idx);
        let fc = block.free_count() + 1;
        block.set_free_count(fc);

        let bins_per_block = self.bins_per_block;
        if fc == 1 {
            // Was full, now has free space
            self.bit_tree.mark_free(block_idx);
        }

        if fc == bins_per_block && self.immediate_decommit {
            // Block is completely empty — queue for deferred decommit.
            // The actual decommit syscall is performed later by
            // process_pending_decommits() (called from trim()), keeping
            // the pool lock free of syscalls during the hot free path.
            self.decommit_pending.push(block_idx);
        }
    }

    pub fn trim(&mut self) {
        // First, process any pending decommits from free().
        // This turns queued empty blocks into actually-decommitted blocks
        // so the trailing-block trim below can pop them.
        self.process_pending_decommits();

        let bins_per_block = self.bins_per_block;

        // Trim trailing empty blocks (pop from Vec).
        while let Some(last_idx) = self.blocks.len().checked_sub(1) {
            let block = &self.blocks[last_idx];

            if block.free_count() == bins_per_block {
                if block.is_committed() {
                    // Still committed — decommit it.
                    // Safety: index is valid.
                    let block_ptr = unsafe {
                        NonNull::new_unchecked(self.base.as_ptr().add(last_idx * self.block_size))
                    };
                    // Safety: FFI call to decommit memory.
                    // Safety: FFI call to decommit memory.
                if unsafe { PlatformVmOps::decommit(block_ptr, self.block_size) }.is_ok() {
                        self.committed -= self.block_size;
                        stats::sub_saturating(&stats::TOTAL_COMMITTED, self.block_size);
                        stats::sub_saturating(&stats::BINNED_ALLOCATOR_COMMITTED, self.block_size);
                    } else {
                        break;
                    }
                }
                // Already decommitted (sparse) or just decommitted above — remove.
                self.blocks.pop();
                self.bit_tree.mark_full(last_idx);
            } else {
                break;
            }
        }
    }
}

pub(crate) struct PoolChain {
    pub pools: Vec<Pool>,
    pub active_index: usize,
    pub bin_size: usize,
    pub block_size: usize,
    pub config: BinnedAllocatorConfig,
}

// Safety: PoolChain owns the pools and is Send.
unsafe impl Send for PoolChain {}

impl PoolChain {
    pub fn new(bin_size: usize, block_size: usize, config: BinnedAllocatorConfig) -> Self {
        Self {
            pools: Vec::new(),
            active_index: 0,
            bin_size,
            block_size,
            config,
        }
    }

    pub fn alloc(&mut self) -> Result<NonNull<u8>, VmError> {
        if self.pools.is_empty() {
            self.add_pool()?;
        }

        match self.pools[self.active_index].alloc() {
            Ok(ptr) => Ok(ptr),
            Err(VmError::CommitFailed(_)) => {
                // Active pool exhausted. Add new pool.
                self.add_pool()?;
                self.active_index += 1;
                self.pools[self.active_index].alloc()
            }
            Err(e) => Err(e),
        }
    }

    fn add_pool(&mut self) -> Result<(), VmError> {
        let pool = Pool::with_config(self.bin_size, self.block_size, &self.config)?;
        self.pools.push(pool);
        Ok(())
    }

    pub fn free(&mut self, ptr: NonNull<u8>) {
        let ptr_addr = ptr.as_ptr() as usize;
        // Optimization: check active pool first.
        // If not found, iterate backwards (assuming LIFO/recent allocs).
        if !self.pools.is_empty() {
             let pool = &mut self.pools[self.active_index];
             let base = pool.base.as_ptr() as usize;
             if ptr_addr >= base && ptr_addr < base + pool.reserved_size {
                 pool.free(ptr);
                 return;
             }
        }

        for pool in self.pools.iter_mut().rev() {
            let base = pool.base.as_ptr() as usize;
            if ptr_addr >= base && ptr_addr < base + pool.reserved_size {
                pool.free(ptr);
                return;
            }
        }
        
        // In debug mode, this panic helps catch bugs.
        // In release mode, if we can't find the pool, we have a stray pointer or corruption.
        // Given we are inside an unsafe free, UB is expected if ptr is invalid.
        // But panicking is safer.
        panic!("Pointer {:p} does not belong to any pool in this chain (bin_size={})", ptr, self.bin_size);
    }

    pub fn trim(&mut self) {
        for pool in &mut self.pools {
            pool.trim();
        }
    }

    pub fn probe_commit_needed(&mut self) -> Option<PreCommitRequest> {
        if self.pools.is_empty() {
            return None;
        }
        self.pools[self.active_index].probe_commit_needed()
    }

    pub fn integrate_precommit(&mut self, req: &PreCommitRequest) -> bool {
        if self.pools.is_empty() {
             return false;
        }
        let ptr_addr = req.ptr.as_ptr() as usize;

        // Check active pool first
        let pool = &mut self.pools[self.active_index];
        let base = pool.base.as_ptr() as usize;
        if ptr_addr >= base && ptr_addr < base + pool.reserved_size {
            return pool.integrate_precommit(req);
        }

        // Fallback to searching all pools
        for pool in &mut self.pools {
             let base = pool.base.as_ptr() as usize;
             if ptr_addr >= base && ptr_addr < base + pool.reserved_size {
                 return pool.integrate_precommit(req);
             }
        }
        false
    }
}

// ----------------------------------------------------------------------------
// Global Recycler — lock-free cross-thread bundle recycling (ABA-safe)
// ----------------------------------------------------------------------------

use crate::sync::atomic::AtomicU128;
use crate::sync::atomic::{AtomicU32, AtomicUsize};

/// Offset within a freed bin where the recycler stores its inter-bundle link.
/// Bytes 0..8 hold the intra-bundle (`LocalFreeList`) next pointer.
/// Bytes 8..16 hold the recycler stack link (next bundle head in Treiber stack).
/// Requires min bin size >= 16, which is always true (smallest `SIZE_CLASS` = 16).
const RECYCLER_LINK_OFFSET: usize = std::mem::size_of::<usize>();

/// 128-bit tagged pointer for ABA-safe Treiber stack operations.
///
/// Packed into a single `u128` for double-width compare-and-swap (DWCAS):
///
/// ```text
///   bits [127:64]  generation counter  (64 bits)
///   bits [63:0]    pointer             (64 bits, full virtual address)
/// ```
///
/// The generation counter is incremented on every successful CAS, preventing
/// ABA: even if a node is popped, reused, freed, and pushed back at the same
/// address, the generation will differ and the CAS will correctly fail.
///
/// Hardware backing:
///   - `x86_64`: `cmpxchg16b` (available since Core 2 / Athlon 64 X2, ~2005)
///   - `ARM64` < v8.1: `ldxp`/`stxp` (double-word LL/SC)
///   - `ARM64` >= v8.1: `casp` (LSE compare-and-swap pair)
///
/// Using the full 64-bit address avoids any assumptions about VA width
/// (48-bit with 4-level paging vs 57-bit with LA57/5-level paging on `x86_64`,
/// or varying VA widths on `ARM64`). No pointer tagging, no stolen bits.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
struct TaggedPtr(u128);

impl TaggedPtr {
    const NULL: Self = Self(0);

    #[inline]
    fn new(ptr: *mut u8, generation: u64) -> Self {
        Self(u128::from(generation) << 64 | (ptr as usize as u128))
    }

    #[inline]
    fn ptr(self) -> *mut u8 {
        // Truncate to lower 64 bits — the full virtual address.
        (self.0 as usize) as *mut u8
    }

    #[inline]
    fn generation(self) -> u64 {
        (self.0 >> 64) as u64
    }

    #[inline]
    fn is_null(self) -> bool {
        self.ptr().is_null()
    }
}

impl std::fmt::Debug for TaggedPtr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TaggedPtr({:p}, gen={})", self.ptr(), self.generation())
    }
}

/// Lock-free per-pool recycler for cross-thread memory recycling.
///
/// Each slot is an ABA-safe Treiber stack of "bundle heads", using 128-bit
/// atomics (DWCAS) to pair each stack-top pointer with a monotonic generation
/// counter. A bundle is a complete `LocalFreeList` chain (linked via first
/// `usize` bytes of each freed bin). The inter-bundle link uses bytes 8..15
/// of the bundle head node.
///
/// Push: detach TLS list → write recycler link at offset 8 → DWCAS onto stack.
/// Pop: DWCAS off stack → read recycler link → transfer entire bundle to TLS.
/// RAII Guard to ensure recycler slot is restored if a panic occurs during pop.
struct PopReservation<'a> {
    slot: &'a AtomicU128,
    old: TaggedPtr,
    committed: bool,
}

impl Drop for PopReservation<'_> {
    fn drop(&mut self) {
        if !self.committed {
            // Panic or return occurred during critical section.
            // Restore the old pointer, bumping generation to effectively "unlock" it.
            // We own the lock (Odd generation), so a simple store (or RMW) is safe.
            // We increment generation by 2 to maintain monotonicity and ensure it is Even.
            let restored_gen = self.old.generation().wrapping_add(2);
            let restored = TaggedPtr::new(self.old.ptr(), restored_gen);
            
            // We use Release to ensure the restoration is visible.
            self.slot.store(restored.0, Ordering::Release);
        }
    }
}

pub(crate) struct GlobalRecycler {
    slots: [AtomicU128; 44],
    counts: [AtomicU32; 44],
    max_bundles: u32,
}

impl GlobalRecycler {
    /// Pointer to the recycler inter-bundle link field at offset 8.
    #[inline]
    unsafe fn recycler_link_atomic_ptr(node: *mut u8) -> *mut AtomicUsize {
        // Safety: node is valid and offset is within bin.
        let p = unsafe { node.add(RECYCLER_LINK_OFFSET) }.cast::<()>().cast::<AtomicUsize>();
        #[cfg(debug_assertions)]
        debug_assert!(
            (p as usize).is_multiple_of(std::mem::align_of::<AtomicUsize>()),
            "recycler link field is not atomically aligned: {p:p}",
        );
        p
    }

    pub fn new(max_bundles: u32) -> Self {
        Self {
            slots: std::array::from_fn(|_| AtomicU128::new(TaggedPtr::NULL.0)),
            max_bundles,
            counts: std::array::from_fn(|_| AtomicU32::new(0)),
        }
    }

    /// Push a bundle onto the recycler. The bundle is a linked list whose head
    /// is `bundle_head` with `count` items linked via first-usize bytes.
    ///
    /// Returns `Some(bundle_head)` if the recycler is full (caller must flush to pool).
    pub fn push(
        &self,
        pool_idx: usize,
        bundle_head: NonNull<u8>,
        _count: u32,
    ) -> Option<NonNull<u8>> {
        // Optimistically reserve a slot.
        // We use Relaxed ordering because strict consistency of the count isn't required for safety,
        // just for approximate resource limiting.
        let prev_count = self.counts[pool_idx].fetch_add(1, Ordering::Relaxed);
        if prev_count >= self.max_bundles {
            // Limit reached/exceeded: back out and return the bundle.
            self.counts[pool_idx].fetch_sub(1, Ordering::Relaxed);
            return Some(bundle_head);
        }

        let slot = &self.slots[pool_idx];
        let new_ptr = bundle_head.as_ptr();

        loop {
            let old = TaggedPtr(slot.load(Ordering::Acquire));

            // Check for RESERVED state (Odd generation).
            // If reserved, another thread is in the middle of a pop (reading the pointer).
            // We must wait for it to stabilize (Even) before we can push.
            if !old.generation().is_multiple_of(2) {
                hint::spin_loop();
                continue;
            }

            // Write recycler-stack-next atomically at offset 8
            // (intra-bundle link at offset 0 is preserved).
            // Safety: new_ptr is valid.
            unsafe {
                (*Self::recycler_link_atomic_ptr(new_ptr))
                    .store(old.ptr() as usize, Ordering::Relaxed);
            }

            // Bump generation by 2: even -> even.
            // Maintains stability invariant and prevents ABA.
            let new = TaggedPtr::new(new_ptr, old.generation().wrapping_add(2));

            if slot.compare_exchange_weak(old.0, new.0, Ordering::Release, Ordering::Relaxed).is_ok() {
                // Count was already incremented at the start.
                return None;
            }
        }
    }

    /// Pop a bundle from the recycler. Returns the bundle head (a linked list
    /// through first-usize bytes) or None if empty.
    pub fn pop(&self, pool_idx: usize) -> Option<NonNull<u8>> {
        let slot = &self.slots[pool_idx];

        loop {
            let old = TaggedPtr(slot.load(Ordering::Acquire));
            if old.is_null() {
                return None;
            }

            // Check for RESERVED state (Odd generation).
            if !old.generation().is_multiple_of(2) {
                hint::spin_loop();
                continue;
            }

            // Step 1: Reserve the node (Even -> Odd).
            // This prevents other threads from popping (they spin on Odd) or pushing (they spin on Odd),
            // effectively pinning 'old.ptr()' so we can safely dereference it.
            let reserved = TaggedPtr::new(old.ptr(), old.generation().wrapping_add(1));

            if slot
                .compare_exchange_weak(old.0, reserved.0, Ordering::Acquire, Ordering::Relaxed)
                .is_err()
            {
                continue;
            }

            // Create RAII guard to unlock (restore Even generation) on panic
            let mut guard = PopReservation {
                slot,
                old,
                committed: false,
            };

            // Step 2: Safe Read.
            // We own the reservation, so 'old.ptr()' is stable.
            // Read recycler-stack-next atomically from offset 8.
            // Safety: old.ptr() is pinned by reservation.
            let next = unsafe {
                (*Self::recycler_link_atomic_ptr(old.ptr())).load(Ordering::Relaxed) as *mut u8
            };

            // Step 3: Commit (Odd -> Even).
            // Restore the stack head to the 'next' pointer and an Even generation.
            // We increment by 2 relative to 'old' (gen+1 -> gen+2) to ensure strict monotonicity.
            let next_stable = TaggedPtr::new(next, old.generation().wrapping_add(2));

            // We use compare_exchange (strong) to ensure completion. Since we hold the reservation,
            // this should only fail if we have a logic bug or memory corruption.
            let res = slot.compare_exchange(
                reserved.0,
                next_stable.0,
                Ordering::Release,
                Ordering::Relaxed,
            );
            debug_assert!(res.is_ok(), "GlobalRecycler::pop reservation broken!");
            
            // Success: mark guard as committed so it doesn't rollback on drop
            guard.committed = true;

            self.counts[pool_idx].fetch_sub(1, Ordering::Relaxed);
            return NonNull::new(old.ptr());
        }
    }
}

// Safety: GlobalRecycler only uses atomics (128-bit DWCAS + 32-bit counters).
unsafe impl Send for GlobalRecycler {}
// Safety: GlobalRecycler handles synchronization internally.
unsafe impl Sync for GlobalRecycler {}

use crate::sync::cell::{Cell, UnsafeCell};
use crate::sync::{Mutex, OnceLock};

static GLOBAL_BINNED_INSTANCE: OnceLock<BinnedAllocator> = OnceLock::new();

// Global trim epoch for cooperative cache flushing.
// `trim()` increments this; each thread cache flushes when it observes
// its local epoch lagging behind.
crate::sync::static_atomic! {
    static CACHE_TRIM_EPOCH: AtomicU64 = AtomicU64::new(0);
}

thread_local! {
    static GLOBAL_THREAD_CACHE: ThreadCacheHandle = ThreadCacheHandle::new();
}

/// Per-thread handle owning the thread-local cache.
///
/// # Safety
///
/// `cache` is wrapped in `UnsafeCell` because it is only ever accessed by the
/// owning thread (via TLS). Neither `alloc_with_cache` nor `free_with_cache`
/// re-enter the TLS access point — they interact with pools and the recycler
/// directly. `flush()` locks pool mutexes but never re-enters TLS.
///
struct ThreadCacheHandle {
    cache: UnsafeCell<ThreadCache>,
    last_seen_trim_epoch: Cell<u64>,
}

// Safety: ThreadCacheHandle is confined to a single thread via thread_local!.
// The UnsafeCell<ThreadCache> and Cell<u64> epoch are thread-local only.
unsafe impl Sync for ThreadCacheHandle {}

impl ThreadCacheHandle {
    fn new() -> Self {
        Self {
            cache: UnsafeCell::new(ThreadCache::new()),
            last_seen_trim_epoch: Cell::new(CACHE_TRIM_EPOCH.load(Ordering::Relaxed)),
        }
    }

    /// Check the cooperative trim epoch and flush if signalled.
    /// Called at the top of every alloc/free hot path.
    #[inline]
    fn check_flush(&self) {
        let global_epoch = CACHE_TRIM_EPOCH.load(Ordering::Acquire);
        if self.last_seen_trim_epoch.get() != global_epoch {
            self.last_seen_trim_epoch.set(global_epoch);
            // Safety: single-threaded TLS access (see struct-level safety comment)
            let cache = crate::sync::unsafe_cell_get_mut!(self.cache);
            cache.flush();
        }
    }
}

impl Drop for ThreadCacheHandle {
    fn drop(&mut self) {
        // Flush cached pointers back to their pools on thread exit.
        // This prevents pointer leaks when threads are destroyed.
        // Safety: Drop provides &mut self, guaranteeing exclusive access.
        let cache = crate::sync::unsafe_cell_get_mut!(self.cache);
        cache.flush();
    }
}

pub struct GlobalBinnedAllocator;

impl GlobalBinnedAllocator {
    /// Initialize the global allocator.
    ///
    /// # Errors
    ///
    /// Returns `VmError::InitializationFailed` if the allocator is already initialized
    /// or if the underlying memory pool creation fails.
    pub fn init() -> Result<(), VmError> {
        GLOBAL_BINNED_INSTANCE
            .set(BinnedAllocator::new()?)
            .map_err(|_| VmError::InitializationFailed("Already initialized".to_string()))
    }

    /// Returns a reference to the initialized global allocator.
    ///
    /// # Panics
    ///
    /// Panics if the global allocator has not been initialized via [`init`](Self::init).
    pub fn get() -> &'static BinnedAllocator {
        GLOBAL_BINNED_INSTANCE
            .get()
            .expect("GlobalBinnedAllocator not initialized")
    }

    /// Allocate memory for the given [`Layout`](std::alloc::Layout).
    ///
    /// This is the primary allocation API. It honours both size and alignment.
    /// For raw byte buffers where alignment does not matter, see
    /// [`alloc_bytes`](Self::alloc_bytes).
    ///
    /// # Errors
    ///
    /// Returns `VmError` if allocation fails (e.g. OOM or invalid layout).
    pub fn alloc(layout: std::alloc::Layout) -> Result<NonNull<u8>, VmError> {
        let allocator = Self::get();
        GLOBAL_THREAD_CACHE.with(|handle| {
            handle.check_flush();
            // Safety: single-threaded TLS access; no re-entrancy possible
            // (alloc_with_cache accesses pools/recycler, never TLS)
            let cache = crate::sync::unsafe_cell_get_mut!(handle.cache);
            if cache.allocator.is_none() {
                cache.bind(allocator);
            }
            allocator.alloc_with_cache(cache, layout)
        })
    }

    /// Allocate `size` bytes with no alignment guarantee beyond the bin's
    /// natural alignment (>= 16). Suitable for raw byte buffers only.
    /// For typed allocations, use [`alloc`](Self::alloc).
    ///
    /// # Errors
    ///
    /// Returns `VmError` if allocation fails (e.g. OOM).
    pub fn alloc_bytes(size: usize) -> Result<NonNull<u8>, VmError> {
        Self::alloc(BinnedAllocator::layout_for_bytes(size)?)
    }

    /// Free a pointer previously obtained from [`alloc`](Self::alloc).
    ///
    /// The `layout` must match the layout used to allocate the pointer.
    /// For raw byte buffers, see [`free_bytes`](Self::free_bytes).
    ///
    /// # Safety
    /// - `ptr` must have been returned by [`alloc`](Self::alloc) from this
    ///   global allocator instance.
    /// - `layout` must exactly match the layout used for allocation.
    /// - `ptr` must not have been freed already.
    pub unsafe fn free(ptr: NonNull<u8>, layout: std::alloc::Layout) {
        let allocator = Self::get();
        GLOBAL_THREAD_CACHE.with(|handle| {
            handle.check_flush();
            // Safety: single-threaded TLS access; no re-entrancy possible
            let cache = crate::sync::unsafe_cell_get_mut!(handle.cache);
            if cache.allocator.is_none() {
                cache.bind(allocator);
            }
            allocator.free_with_cache(cache, ptr, layout);
        });
    }

    /// Free a pointer previously obtained from [`alloc_bytes`](Self::alloc_bytes).
    /// For typed allocations, use [`free`](Self::free).
    ///
    /// # Safety
    /// - `ptr` must have been returned by [`alloc_bytes`](Self::alloc_bytes)
    ///   from this global allocator instance.
    /// - `size` must exactly match the size used for allocation.
    /// - `ptr` must not have been freed already.
    ///
    /// # Panics
    ///
    /// Panics if a layout cannot be created for the given size (e.g., size is too large).
    pub unsafe fn free_bytes(ptr: NonNull<u8>, size: usize) {
        // Safety: ptr and size match allocation.
        unsafe { Self::free(ptr, std::alloc::Layout::from_size_align(size, 1).unwrap()) }
    }

    /// Signal all thread caches to flush and trim global pools.
    ///
    /// Flushing is cooperative: the calling thread's cache is flushed
    /// immediately, while other threads flush on their next alloc/free.
    /// This is the standard approach used by jemalloc and mimalloc —
    /// sleeping threads flush when they wake up and allocate.
    pub fn trim() {
        // Signal all thread caches to flush on their next alloc/free.
        CACHE_TRIM_EPOCH.fetch_add(1, Ordering::AcqRel);

        // Immediately flush the calling thread's own cache
        GLOBAL_THREAD_CACHE.with(|handle| {
            handle.check_flush();
        });

        // Trim global pools
        if let Some(allocator) = GLOBAL_BINNED_INSTANCE.get() {
            allocator.trim();
        }
    }
}

// Safety: Implementation follows GlobalAlloc contract.
unsafe impl std::alloc::GlobalAlloc for GlobalBinnedAllocator {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        match GlobalBinnedAllocator::alloc(layout) {
            Ok(ptr) => ptr.as_ptr(),
            Err(_) => std::ptr::null_mut(),
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        if let Some(ptr) = std::ptr::NonNull::new(ptr) {
            // Safety: Deallocating ptr with correct layout.
            unsafe {
                GlobalBinnedAllocator::free(ptr, layout);
            }
        }
    }
}

pub struct BinnedAllocator {
    pools: Vec<Mutex<PoolChain>>, // One chain per size class
    block_size: usize,
    config: BinnedAllocatorConfig,
    recycler: GlobalRecycler,
    large_cache: Mutex<super::large_cache::LargeAllocCache>,
}

impl BinnedAllocator {
    #[inline]
    fn layout_for_bytes(size: usize) -> Result<std::alloc::Layout, VmError> {
        std::alloc::Layout::from_size_align(size, 1).map_err(|_| {
            VmError::CommitFailed(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Invalid allocation size {size}"),
            ))
        })
    }

    #[inline]
    fn dangling_for_align(align: usize) -> NonNull<u8> {
        debug_assert!(align.is_power_of_two() && align > 0);
        // Non-dereferenceable pointer used for zero-sized allocations.
        // Safety: align is non-zero (power of two).
        unsafe { NonNull::new_unchecked(align as *mut u8) }
    }

    /// Create a new `BinnedAllocator` with default configuration.
    ///
    /// # Errors
    ///
    /// Returns `VmError` if memory reservation fails or configuration is invalid.
    pub fn new() -> Result<Self, VmError> {
        Self::with_config(BinnedAllocatorConfig::default())
    }

    /// Create a new `BinnedAllocator` with custom configuration.
    ///
    /// # Errors
    ///
    /// Returns `VmError` if memory reservation fails or configuration is invalid
    /// (e.g., block size not page-aligned).
    pub fn with_config(mut config: BinnedAllocatorConfig) -> Result<Self, VmError> {
        // Detect page sizes
        let supported_sizes = PlatformVmOps::supported_page_sizes();
        let min_page = supported_sizes.first().copied().unwrap_or(4096);

        // Resolve block_size: 0 means auto-detect
        if config.block_size == 0 {
            config.block_size = std::cmp::max(64 * 1024, min_page);
        }
        let block_size = config.block_size;
        debug_assert!(
            block_size.is_multiple_of(min_page) && block_size >= min_page,
            "block_size ({block_size}) must be page-size aligned (page_size = {min_page})",
        );
        let max_bins_per_block = block_size / SIZE_CLASSES[0];
        if max_bins_per_block > u16::MAX as usize {
            return Err(VmError::InitializationFailed(format!(
                "Invalid block_size {}: smallest size class ({}) yields {} bins per block, exceeding u16::MAX ({})",
                block_size,
                SIZE_CLASSES[0],
                max_bins_per_block,
                u16::MAX
            )));
        }

        let mut pools = Vec::with_capacity(SIZE_CLASSES.len());
        for &bin_size in SIZE_CLASSES {
            pools.push(Mutex::new(PoolChain::new(bin_size, block_size, config.clone())));
        }

        let recycler = GlobalRecycler::new(config.recycler_max_bundles);
        // Large cache: 16 MB limit (holds decommitted OS pages for reuse).
        // Huge page support is auto-detected from supported_page_sizes()
        // unless explicitly disabled via config.
        let large_cache = Mutex::new(if config.use_huge_pages {
            super::large_cache::LargeAllocCache::new(16 * 1024 * 1024)
        } else {
            super::large_cache::LargeAllocCache::without_huge_pages(16 * 1024 * 1024)
        });

        Ok(Self {
            pools,
            block_size,
            config,
            recycler,
            large_cache,
        })
    }

    /// Allocate memory for the given [`Layout`](std::alloc::Layout).
    ///
    /// This is the primary allocation API. It honours both size and alignment.
    /// For raw byte buffers where alignment does not matter, see
    /// [`alloc_bytes`](Self::alloc_bytes).
    ///
    /// # Errors
    ///
    /// Returns `VmError` if allocation fails (e.g. OOM).
    pub fn alloc(&self, layout: std::alloc::Layout) -> Result<NonNull<u8>, VmError> {
        let size = layout.size();
        if size == 0 {
            return Ok(Self::dangling_for_align(layout.align()));
        }
        if size > MAX_SMALL_SIZE || layout.align() > PlatformVmOps::page_size() {
            // Transparent large-alloc routing (covers both oversized allocations
            // and small allocations with alignment too large for any size class)
            let mut cache = self.large_cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            let (ptr, _actual_size) = cache.alloc(layout)?;
            return Ok(ptr);
        }
        let pool_idx = Self::size_class(size, layout.align());
        let mut guard = self.pools[pool_idx]
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        let ptr = guard.alloc()?;
        Ok(ptr)
    }

    /// Allocate `size` bytes with no alignment guarantee beyond the bin's
    /// natural alignment (>= 16). Suitable for raw byte buffers only.
    /// For typed allocations, use [`alloc`](Self::alloc).
    ///
    /// # Errors
    ///
    /// Returns `VmError` if allocation fails (e.g. OOM).
    pub fn alloc_bytes(&self, size: usize) -> Result<NonNull<u8>, VmError> {
        self.alloc(Self::layout_for_bytes(size)?)
    }

    /// Free a pointer previously obtained from [`alloc`](Self::alloc).
    ///
    /// The `layout` must match the layout used to allocate the pointer.
    /// For raw byte buffers, see [`free_bytes`](Self::free_bytes).
    ///
    /// # Safety
    /// - `ptr` must have been returned by [`alloc`](Self::alloc) on this
    ///   allocator instance.
    /// - `layout` must exactly match the layout used for allocation.
    /// - `ptr` must not have been freed already.
    pub unsafe fn free(&self, ptr: NonNull<u8>, layout: std::alloc::Layout) {
        let size = layout.size();
        if size == 0 {
            let _ = ptr;
            return;
        }
        if size > MAX_SMALL_SIZE || layout.align() > PlatformVmOps::page_size() {
            let mut cache = self.large_cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            cache.free(ptr, layout);
            return;
        }
        let pool_idx = Self::size_class(size, layout.align());
        let mut guard = self.pools[pool_idx]
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.free(ptr);
    }

    /// Free a pointer previously obtained from [`alloc_bytes`](Self::alloc_bytes).
    /// For typed allocations, use [`free`](Self::free).
    ///
    /// # Safety
    /// - `ptr` must have been returned by [`alloc_bytes`](Self::alloc_bytes)
    ///   on this allocator instance.
    /// - `size` must exactly match the size used for allocation.
    /// - `ptr` must not have been freed already.
    ///
    /// # Panics
    ///
    /// Panics if a layout cannot be created for the given size (e.g., size is too large).
    pub unsafe fn free_bytes(&self, ptr: NonNull<u8>, size: usize) {
        // Safety: ptr and size match allocation.
        unsafe { self.free(ptr, std::alloc::Layout::from_size_align(size, 1).unwrap()) }
    }

    pub(crate) fn alloc_with_cache(
        &self,
        cache: &mut ThreadCache,
        layout: std::alloc::Layout,
    ) -> Result<NonNull<u8>, VmError> {
        if let Some(owner) = cache.allocator
            && !std::ptr::eq(owner, self)
        {
            debug_assert!(false, "ThreadCache is bound to a different allocator");
            // Safety: Unreachable logic.
            unsafe { std::hint::unreachable_unchecked() }
        }

        let size = layout.size();
        if size == 0 {
            return Ok(Self::dangling_for_align(layout.align()));
        }
        if size > MAX_SMALL_SIZE || layout.align() > PlatformVmOps::page_size() {
            let mut lc = self.large_cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            let (ptr, _) = lc.alloc(layout)?;
            return Ok(ptr);
        }
        let pool_idx = Self::size_class(size, layout.align());

        // Fast path: thread cache
        if let Some(ptr) = cache.bins[pool_idx].pop() {
            return Ok(ptr);
        }

        // If cache is unbound, we cannot safely batch-refill.
        if cache.allocator.is_none() {
            return self.alloc(layout);
        }

        // Medium path: try the lock-free GlobalRecycler before taking the pool lock
        if let Some(bundle_head) = self.recycler.pop(pool_idx) {
            // Single-pass: walk the chain to count + find tail, then prepend to TLS cache
            cache.bins[pool_idx].receive_bundle_walk(bundle_head);
            if let Some(ptr) = cache.bins[pool_idx].pop() {
                return Ok(ptr);
            }
        }

        // Slow path: lock pool and batch-refill.
        // We try to keep VM syscalls (reserve, commit) outside the lock.
        let mut guard = self.pools[pool_idx]
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        // --- Pool initialization outside lock ---
        // For the very first pool in the chain, we try to initialize it outside the lock
        // to avoid holding the lock during the heavy `reserve` syscall.
        if guard.pools.is_empty() {
            drop(guard);
            // We use the same config as the chain.
            let bin_size = SIZE_CLASSES[pool_idx];
            let new_pool = Pool::with_config(bin_size, self.block_size, &self.config)?;
            
            guard = self.pools[pool_idx]
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            
            // Push only if still empty (race check)
            if guard.pools.is_empty() {
                guard.pools.push(new_pool);
            }
        }

        // --- Pre-commit outside lock ---
        // Check if the pool needs a VM commit before alloc can proceed.
        {
            if let Some(req) = guard.probe_commit_needed() {
                drop(guard);
                // commit() is mprotect/mmap — do it unlocked.
                // Safety: FFI call to commit memory.
                unsafe { PlatformVmOps::commit(req.ptr, req.size)? };
                guard = self.pools[pool_idx]
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                guard.integrate_precommit(&req);
            }
        }

        // --- Batch refill (pure pointer math under lock) ---
        let bin_size = SIZE_CLASSES[pool_idx];
        let batch_size = self.config.batch_size_for(bin_size);

        for _ in 0..batch_size {
            // Alloc directly from chain. 
            // If chain needs to grow (new pool), it will do so here (under lock).
            match guard.alloc() {
                Ok(ptr) => cache.bins[pool_idx].push(ptr),
                Err(_) => break,
            }
        }

        if let Some(ptr) = cache.bins[pool_idx].pop() {
            return Ok(ptr);
        }

        // Fallback: single alloc
        let ptr = guard.alloc()?;
        Ok(ptr)
    }

    pub(crate) fn free_with_cache(
        &self,
        cache: &mut ThreadCache,
        ptr: NonNull<u8>,
        layout: std::alloc::Layout,
    ) {
        if let Some(owner) = cache.allocator {
            if !std::ptr::eq(owner, self) {
                debug_assert!(false, "ThreadCache is bound to a different allocator");
                // Safety: Unreachable logic.
                // Safety: Unreachable logic.
            unsafe { std::hint::unreachable_unchecked() }
            }
        } else {
            // Unbound cache: direct free to avoid leaks
            // Safety: ptr is valid.
            unsafe {
                return self.free(ptr, layout);
            }
        }

        let size = layout.size();
        if size == 0 {
            return;
        }
        if size > MAX_SMALL_SIZE || layout.align() > MAX_SMALL_SIZE {
            let mut lc = self.large_cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            lc.free(ptr, layout);
            return;
        }
        let pool_idx = Self::size_class(size, layout.align());

        // Fast path: push to cache
        cache.bins[pool_idx].push(ptr);

        // Flush if over limit.
        let bin_size = SIZE_CLASSES[pool_idx];
        let max_cache_size = self.config.max_cache_for(bin_size);

        if cache.bins[pool_idx].count > max_cache_size {
            // Detach the entire list as a bundle
            if let Some((bundle_head, bundle_count)) = cache.bins[pool_idx].take_bundle() {
                // Try to push the bundle to the lock-free GlobalRecycler
                if let Some(rejected_head) = self.recycler.push(pool_idx, bundle_head, bundle_count)
                {
                    // Recycler is full — flush the bundle directly to the pool under lock
                    let mut guard = self.pools[pool_idx]
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    
                    let mut node = Some(rejected_head);
                    while let Some(n) = node {
                        // Safety: n is valid.
                        let next = unsafe { *n.cast::<usize>().as_ptr() } as *mut u8;
                        guard.free(n);
                        node = NonNull::new(next);
                    }
                }
                // else: bundle accepted by recycler, TLS cache is now empty
            }
        }
    }

    pub fn trim(&self) {
        for (pool_idx, pool_mutex) in self.pools.iter().enumerate() {
            let mut guard = pool_mutex.lock().unwrap_or_else(std::sync::PoisonError::into_inner);

            // First, drain any bundles from the recycler so we can decommit them.
            // Snapshot the count to avoid infinite loops if other threads are pushing.
            let limit = self.recycler.counts[pool_idx].load(Ordering::Relaxed);
            for _ in 0..limit {
                if let Some(bundle_head) = self.recycler.pop(pool_idx) {
                    // Free the bundle directly to the pool
                    let mut node = Some(bundle_head);
                    while let Some(n) = node {
                        // Safety: n is valid.
                        let next = unsafe { *n.cast::<usize>().as_ptr() } as *mut u8;
                        guard.free(n);
                        node = NonNull::new(next);
                    }
                } else {
                    break;
                }
            }

            guard.trim();
        }
        // Also trim the large alloc cache
        let mut lc = self.large_cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        lc.trim();
    }

    pub(crate) fn size_class(size: usize, align: usize) -> usize {
        if size == 0 {
            debug_assert!(false, "Size 0 not supported by BinnedAllocator");
            // Safety: Unreachable logic.
            unsafe { std::hint::unreachable_unchecked() }
        }
        if size > MAX_SMALL_SIZE {
            debug_assert!(false, "Size {size} too large for size classes");
            // Safety: Unreachable logic.
            unsafe { std::hint::unreachable_unchecked() }
        }

        // Start with the class for the requested size
        let mut idx = SIZE_CLASS_LUT[(size + 15) >> 4] as usize;

        // Bump up if the size class itself isn't aligned enough.
        // We iterate because size classes are dense.
        // This is efficient because alignment is usually small power of two.
        while idx < SIZE_CLASSES.len() {
            let sc = SIZE_CLASSES[idx];
            if sc.is_multiple_of(align) {
                return idx;
            }
            idx += 1;
        }

        debug_assert!(
            false,
            "No size class satisfies size {size} and alignment {align}",
        );
        // Safety: Logic ensures all valid inputs are handled by the loop above.
        unsafe { std::hint::unreachable_unchecked() }
    }
}

// 44 Size classes: 16B..128B (step 16), then doubling steps up to 64KB
pub(crate) const SIZE_CLASSES: &[usize] = &[
    16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024,
    1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192, 10240, 12288, 14336,
    16384, 20480, 24576, 28672, 32768, 40960, 49152, 57344, 65536,
];

/// O(1) size-to-class lookup table. Index by `ceil(size / 16)`.
/// Table has 4097 entries covering sizes 1..65536 in 16-byte quanta.
/// Each entry is the size class index (0..43).
static SIZE_CLASS_LUT: [u8; 4097] = build_size_class_lut();

const fn build_size_class_lut() -> [u8; 4097] {
    // Duplicate SIZE_CLASSES as a fixed array for const evaluation
    const CLASSES: [usize; 44] = [
        16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896,
        1024, 1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192, 10240, 12288,
        14336, 16384, 20480, 24576, 28672, 32768, 40960, 49152, 57344, 65536,
    ];
    let mut table = [0u8; 4097];
    // table[0] unused (size 0 is invalid)
    let mut q: usize = 1;
    let mut sc: u8 = 0;
    while sc < 44 {
        let class_quanta = CLASSES[sc as usize] / 16;
        while q <= class_quanta {
            {
                table[q] = sc;
            }
            q += 1;
        }
        sc += 1;
    }
    table
}

// ----------------------------------------------------------------------------
// Thread Cache
// ----------------------------------------------------------------------------

pub(crate) struct LocalFreeList {
    pub head: Option<NonNull<u8>>,
    pub count: u32,
}

impl LocalFreeList {
    pub fn new() -> Self {
        Self {
            head: None,
            count: 0,
        }
    }

    pub fn push(&mut self, ptr: NonNull<u8>) {
        // For >= 32 byte blocks, could write tail and count to head + 16 and
        // head + 24, respectively; would change O(N) walk in recycler to O(1)
        // but overall not efficient.  Keep every size class same in-object
        // layout to reduce branching on hot path -> reduce mispredicts.
        // Bundles should arrive in L1 or at worst L2 anyway while those two
        // writes would dirty cache lines likely hot in another core's priv
        // cache.  Also this would halve effective payload density considering
        // 32-byte alignment.
        // SAFETY: ptr is guaranteed to be valid and aligned to at least align_of::<usize>().
        unsafe {
            // Link via first usize bytes. Pool::free uses u16 + canary, but
            // TLS cache owns the memory and uses full usize pointers.
            *ptr.cast::<usize>().as_ptr() = self.head.map_or(0, |p| p.as_ptr() as usize);
        }
        self.head = Some(ptr);
        self.count += 1;
    }

    pub fn pop(&mut self) -> Option<NonNull<u8>> {
        if let Some(ptr) = self.head {
            // SAFETY: ptr is valid and aligned (see push).
            unsafe {
                let next = *ptr.cast::<usize>().as_ptr();
                self.head = NonNull::new(next as *mut u8);
            }
            self.count -= 1;
            Some(ptr)
        } else {
            None
        }
    }

    /// Detach the entire list as a bundle. Returns `(head, count)` or None if empty.
    pub fn take_bundle(&mut self) -> Option<(NonNull<u8>, u32)> {
        if let Some(head) = self.head.take() {
            let count = self.count;
            self.count = 0;
            Some((head, count))
        } else {
            None
        }
    }

    /// Adopt a bundle from the recycler. The bundle is a null-terminated
    /// linked list through the first usize bytes. Walks the chain to count
    /// items and find the tail, then prepends it to the current list.
    pub fn receive_bundle_walk(&mut self, head: NonNull<u8>) {
        // Single-pass: count items and find tail
        let mut tail = head;
        let mut count = 1u32;
        loop {
            // SAFETY: tail is valid and aligned (from Pool/Recycler).
            let next = unsafe { *tail.cast::<usize>().as_ptr() } as *mut u8;
            if let Some(nn) = NonNull::new(next) {
                tail = nn;
                count += 1;
            } else {
                break;
            }
        }

        // Link tail to our current head
        // SAFETY: tail is valid and aligned.
        unsafe {
            *tail.cast::<usize>().as_ptr() = self.head.map_or(0, |p| p.as_ptr() as usize);
        }
        self.head = Some(head);
        self.count += count;
    }
}

pub(crate) struct ThreadCache {
    // Arrays of free lists per size class
    bins: Vec<LocalFreeList>,
    // Optional reference to the allocator that owns this cache.
    // Must be 'static to ensure it outlives the thread.
    allocator: Option<&'static BinnedAllocator>,
}

// Safety: ThreadCache is only accessed by the owning thread (via TLS). Flushing
// is cooperative — trim() bumps an epoch, and each thread flushes on its
// next alloc/free. The content (pointers) can be sent between threads via flush.
unsafe impl Send for ThreadCache {}

impl ThreadCache {
    pub fn new() -> Self {
        let mut bins = Vec::with_capacity(SIZE_CLASSES.len());
        for _ in 0..SIZE_CLASSES.len() {
            bins.push(LocalFreeList::new());
        }
        Self {
            bins,
            allocator: None,
        }
    }

    pub fn bind(&mut self, allocator: &'static BinnedAllocator) {
        self.allocator = Some(allocator);
    }

    pub fn flush(&mut self) {
        if let Some(allocator) = self.allocator {
            for (idx, bin) in self.bins.iter_mut().enumerate() {
                if bin.count > 0 {
                    // Recover from poisoned mutex to avoid leaking pointers (P8)
                    let mut guard = allocator.pools[idx]
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    
                    while let Some(ptr) = bin.pop() {
                        guard.free(ptr);
                    }
                }
            }
        }
    }
}

impl Drop for ThreadCache {
    fn drop(&mut self) {
        // Return all pointers to the bound allocator if it exists.
        // We must recover from poisoned mutexes to avoid permanent leaks (P8).
        if let Some(allocator) = self.allocator {
            for (idx, bin) in self.bins.iter_mut().enumerate() {
                if bin.count > 0 {
                    let mut guard = allocator.pools[idx]
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    
                    while let Some(ptr) = bin.pop() {
                        guard.free(ptr);
                    }
                }
            }
        }
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use crate::sync::Arc;
    use crate::sync::thread;

    #[test]
    fn test_binned_allocator_basic() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let allocator = BinnedAllocator::new().unwrap();

        // Alloc 16 bytes
        let ptr1 = allocator.alloc_bytes(16).unwrap();
        // Safety: Test code.
        unsafe { ptr1.as_ptr().write(0xAA) };

        // Alloc 32 bytes
        let ptr2 = allocator.alloc_bytes(32).unwrap();
        // Safety: Test code.
        unsafe { ptr2.as_ptr().write(0xBB) };

        assert_ne!(ptr1, ptr2);

        // Safety: Test code.
        unsafe { allocator.free_bytes(ptr1, 16); }
        // Safety: Test code.
        unsafe { allocator.free_bytes(ptr2, 32); }
    }

    #[test]
    fn test_alloc_bytes_huge_size_returns_error_not_panic() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let allocator = BinnedAllocator::new().unwrap();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            allocator.alloc_bytes(usize::MAX)
        }));

        assert!(result.is_ok(), "alloc_bytes(usize::MAX) must not panic");
        assert!(
            result.unwrap().is_err(),
            "alloc_bytes(usize::MAX) must return an error"
        );
    }

    #[test]
    fn test_binned_allocator_thread_safety() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let allocator = Arc::new(BinnedAllocator::new().unwrap());
        let mut handles = vec![];
        let num_threads = 8;
        let allocs_per_thread = 200;

        for t in 0..num_threads {
            let alloc = allocator.clone();
            handles.push(thread::spawn(move || {
                let mut ptrs = Vec::with_capacity(allocs_per_thread);
                let sizes = [16, 64, 256, 1024, 4096, 16384, 65536];

                // 1. Sustained concurrent holding: Allocate all first
                for i in 0..allocs_per_thread {
                    let size = sizes[i % sizes.len()];
                    // Use different paths (with/without cache) intermittently
                    let ptr = if i % 2 == 0 {
                        alloc.alloc_bytes(size).unwrap()
                    } else {
                        // Manual cache for this thread
                        let mut cache = ThreadCache::new();
                        let layout = std::alloc::Layout::from_size_align(size, 1).unwrap();
                        alloc.alloc_with_cache(&mut cache, layout).unwrap()
                        // Note: cache will be dropped here, flushes pointers back to pool.
                        // Wait, if I drop the cache immediately, it flushes ptr back.
                        // That's fine, tests refill logic.
                    };

                    // Write unique data
                    // Safety: Test code.
                    unsafe {
                        let val = (t * 1000 + i).to_le_bytes()[0];
                        ptr.as_ptr().write(val);
                    }
                    ptrs.push((ptr, size));
                }

                // 2. Verify all held pointers still have correct data
                for (i, (ptr, _size)) in ptrs.iter().enumerate() {
                    // Safety: Test code.
                    unsafe {
                        let expected = (t * 1000 + i).to_le_bytes()[0];
                        assert_eq!(ptr.as_ptr().read(), expected, "Memory corruption detected!");
                    }
                }

                // 3. Free everything
                for (ptr, size) in ptrs {
                    // Safety: Test code.
                    // Safety: Test code.
            unsafe { alloc.free_bytes(ptr, size); }
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_bit_tree() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let mut tree = BitTree::new();
        assert_eq!(tree.find_free(), None);

        tree.mark_free(0);
        assert_eq!(tree.find_free(), Some(0));

        tree.mark_full(0);
        assert_eq!(tree.find_free(), None);

        tree.mark_free(1);
        assert_eq!(tree.find_free(), Some(1));

        tree.mark_free(0);
        assert_eq!(tree.find_free(), Some(0));
    }

    #[test]
    fn test_thread_cache() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let allocator = BinnedAllocator::new().unwrap();
        let mut cache = ThreadCache::new();
        // Unsafe bind to test caching logic with local allocator
        // Safety: Test code.
        unsafe {
            cache.bind(std::mem::transmute::<&binned::BinnedAllocator, &binned::BinnedAllocator>(&allocator));
        }
        let layout = std::alloc::Layout::from_size_align(32, 1).unwrap();

        // Cache is initially empty, should alloc from pool
        let ptr = allocator.alloc_with_cache(&mut cache, layout).unwrap();
        // Safety: Test code.
        unsafe { ptr.as_ptr().write(0xCC) };

        allocator.free_with_cache(&mut cache, ptr, layout);

        // Should be in cache now
        // Alloc again, should be same ptr (LIFO usually)
        let ptr2 = allocator.alloc_with_cache(&mut cache, layout).unwrap();
        assert_eq!(ptr, ptr2);

        allocator.free_with_cache(&mut cache, ptr2, layout);
    }

    #[test]
    fn test_global_allocator() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Initialize if not already (might fail if run in parallel with other tests that init, so ignore result)
        drop(GlobalBinnedAllocator::init());

        let ptr = GlobalBinnedAllocator::alloc_bytes(128).unwrap();
        // Safety: Test code.
        unsafe { ptr.as_ptr().write(0xDD) };

        // Safety: Test code.
        unsafe { GlobalBinnedAllocator::free_bytes(ptr, 128); }
    }

    // --- BitTree Tests (B1-B8) ---

    #[test]
    fn test_bit_tree_high_indices() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // B1: mark_free/mark_full/find_free for indices near 16383
        let mut tree = BitTree::new();
        let max_idx = 16383;

        tree.mark_free(max_idx);
        assert_eq!(tree.find_free(), Some(max_idx));

        tree.mark_full(max_idx);
        assert_eq!(tree.find_free(), None);
    }

    #[test]
    fn test_bit_tree_all_free() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // B2: Mark all 16384 blocks free
        // This is slow if we do it one by one, but let's test a subset or the concept.
        // Actually, let's just test that we can find them in order.
        let mut tree = BitTree::new();

        // Mark chunks of them free
        for i in 0..100 {
            tree.mark_free(i);
        }

        // Should find 0
        assert_eq!(tree.find_free(), Some(0));

        // If we mark 0 full
        tree.mark_full(0);
        assert_eq!(tree.find_free(), Some(1));
    }

    #[test]
    fn test_bit_tree_all_full() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // B3: Mark all free then all full
        let mut tree = BitTree::new();
        tree.mark_free(0);
        tree.mark_free(100);

        tree.mark_full(0);
        tree.mark_full(100);

        assert_eq!(tree.find_free(), None);
    }

    #[test]
    fn test_bit_tree_sparse() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // B4: Free only blocks 0, 4095, 8191, 16383
        let mut tree = BitTree::new();
        let indices = [0, 4095, 8191, 16383];

        for &idx in &indices {
            tree.mark_free(idx);
        }

        // Should return lowest first
        assert_eq!(tree.find_free(), Some(0));
        tree.mark_full(0);

        assert_eq!(tree.find_free(), Some(4095));
        tree.mark_full(4095);

        assert_eq!(tree.find_free(), Some(8191));
        tree.mark_full(8191);

        assert_eq!(tree.find_free(), Some(16383));
        tree.mark_full(16383);

        assert_eq!(tree.find_free(), None);
    }

    #[test]
    fn test_bit_tree_boundary_64() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // B5: Blocks at indices 63, 64 (L2 word boundary)
        let mut tree = BitTree::new();

        tree.mark_free(63);
        tree.mark_free(64);

        assert_eq!(tree.find_free(), Some(63));
        tree.mark_full(63);
        assert_eq!(tree.find_free(), Some(64));
    }

    #[test]
    fn test_bit_tree_boundary_4096() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // B6: Blocks at indices 4095, 4096 (L1 word boundary)
        // L1 word 0 covers blocks 0..4095 (64 bits * 64 blocks/bit = 4096)
        let mut tree = BitTree::new();

        tree.mark_free(4095);
        tree.mark_free(4096);

        assert_eq!(tree.find_free(), Some(4095));
        tree.mark_full(4095);
        assert_eq!(tree.find_free(), Some(4096));
    }

    #[test]
    fn test_bit_tree_double_mark_free() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // B7: mark_free same index twice
        let mut tree = BitTree::new();
        tree.mark_free(10);
        tree.mark_free(10);
        assert_eq!(tree.find_free(), Some(10));
    }

    #[test]
    fn test_bit_tree_double_mark_full() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // B8: mark_full same index twice
        let mut tree = BitTree::new();
        tree.mark_free(10);
        tree.mark_full(10);
        tree.mark_full(10); // Should be no-op
        assert_eq!(tree.find_free(), None);
    }

    // --- BitTreeChain Tests (BC1-BC4) ---

    #[test]
    fn test_bit_tree_chain_basic() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // BC1: Basic operations within a single segment
        let mut chain = BitTreeChain::new();
        assert_eq!(chain.find_free(), None);

        chain.mark_free(0);
        assert_eq!(chain.find_free(), Some(0));

        chain.mark_full(0);
        assert_eq!(chain.find_free(), None);

        chain.mark_free(100);
        assert_eq!(chain.find_free(), Some(100));
    }

    #[test]
    fn test_bit_tree_chain_across_boundaries() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // BC2: Operations spanning multiple segments
        let mut chain = BitTreeChain::new();

        chain.mark_free(0); // segment 0, local 0
        chain.mark_free(BITTREE_CAPACITY - 1); // segment 0, last slot
        chain.mark_free(BITTREE_CAPACITY); // segment 1, first slot
        chain.mark_free(BITTREE_CAPACITY * 2 + 42); // segment 2, local 42

        // Lowest-first ordering across segments
        assert_eq!(chain.find_free(), Some(0));
        chain.mark_full(0);
        assert_eq!(chain.find_free(), Some(BITTREE_CAPACITY - 1));
        chain.mark_full(BITTREE_CAPACITY - 1);
        assert_eq!(chain.find_free(), Some(BITTREE_CAPACITY));
        chain.mark_full(BITTREE_CAPACITY);
        assert_eq!(chain.find_free(), Some(BITTREE_CAPACITY * 2 + 42));
        chain.mark_full(BITTREE_CAPACITY * 2 + 42);
        assert_eq!(chain.find_free(), None);
    }

    #[test]
    fn test_bit_tree_chain_lazy_growth() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // BC3: Marking a high index lazily creates intermediate segments
        let mut chain = BitTreeChain::new();

        chain.mark_free(BITTREE_CAPACITY * 3 + 42);
        assert_eq!(chain.find_free(), Some(BITTREE_CAPACITY * 3 + 42));

        // Earlier segments exist but are empty
        chain.mark_free(5);
        assert_eq!(chain.find_free(), Some(5)); // Lower index preferred
    }

    #[test]
    fn test_bit_tree_chain_segment_boundary_toggle() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // BC4: Rapid mark/clear at segment boundary
        let mut chain = BitTreeChain::new();
        let boundary = BITTREE_CAPACITY;

        // Toggle the boundary index
        chain.mark_free(boundary);
        assert_eq!(chain.find_free(), Some(boundary));
        chain.mark_full(boundary);
        assert_eq!(chain.find_free(), None);

        // Adjacent slots across boundary
        chain.mark_free(boundary - 1);
        chain.mark_free(boundary);
        assert_eq!(chain.find_free(), Some(boundary - 1));
        chain.mark_full(boundary - 1);
        assert_eq!(chain.find_free(), Some(boundary));
    }

    // --- Pool Tests (P1-P8) ---

    // Constants for test
    // const TEST_BIN_SIZE: usize = 16;
    // const TEST_BLOCK_SIZE: usize = 64; // 4 bins per block

    #[test]
    fn test_pool_alloc_all_bins_in_block() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // P1: Alloc every bin in a block
        let block_size = 65536;
        let bin_size = 16;
        let bins_per_block = block_size / bin_size;

        let mut pool =
            Pool::with_config(bin_size, block_size, &BinnedAllocatorConfig::default()).unwrap();

        // Fill block 0
        for _ in 0..bins_per_block {
            let _ = pool.alloc().unwrap();
        }

        // Block 0 should be full now
        assert_eq!(pool.bit_tree.find_free(), None);

        // Next alloc triggers new block
        let _p = pool.alloc().unwrap();

        // Should be in block 1
        assert_eq!(pool.blocks.len(), 2);
    }

    #[test]
    fn test_pool_alloc_then_free_all() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // P2: Alloc all, free all
        let block_size = 65536;
        let bin_size = 16;
        let bins_per_block = block_size / bin_size;

        let mut pool =
            Pool::with_config(bin_size, block_size, &BinnedAllocatorConfig::default()).unwrap();
        let mut ptrs = Vec::new();

        for _ in 0..bins_per_block {
            ptrs.push(pool.alloc().unwrap());
        }

        assert_eq!(pool.bit_tree.find_free(), None);

        for p in ptrs {
            pool.free(p);
        }

        // Block 0 should be free now
        assert_eq!(pool.bit_tree.find_free(), Some(0));
    }

    #[test]
    fn test_pool_free_and_realloc_order() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Free bins in random order, re-alloc — freelist is LIFO.
        // Keep one bin (p5) allocated to prevent sparse decommit of the block.
        let block_size = 65536;
        let mut pool =
            Pool::with_config(16, block_size, &BinnedAllocatorConfig::default()).unwrap();
        let p1 = pool.alloc().unwrap();
        let p2 = pool.alloc().unwrap();
        let p3 = pool.alloc().unwrap();
        let p4 = pool.alloc().unwrap();
        let _p5 = pool.alloc().unwrap(); // anchor — prevents full-block decommit

        // Free order: 2, 4, 1, 3 → freelist head chain: 3→1→4→2
        pool.free(p2);
        pool.free(p4);
        pool.free(p1);
        pool.free(p3);

        // Realloc pops from LIFO freelist
        // Realloc pops from LIFO freelist
        let r1 = pool.alloc().unwrap();
        let r2 = pool.alloc().unwrap();
        let r3 = pool.alloc().unwrap();
        let r4 = pool.alloc().unwrap();

        assert_eq!(r1, p3);
        assert_eq!(r2, p1);
        assert_eq!(r3, p4);
        assert_eq!(r4, p2);
    }

    #[test]
    fn test_pool_multiple_blocks() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // P4: Alloc enough to force 2+ block commits
        let block_size = 65536;
        let bin_size = 16;
        let bins_per_block = block_size / bin_size;

        let mut pool =
            Pool::with_config(bin_size, block_size, &BinnedAllocatorConfig::default()).unwrap();

        // Fill 2 blocks + 2 items
        let total = bins_per_block * 2 + 2;

        for _ in 0..total {
            pool.alloc().unwrap();
        }

        assert_eq!(pool.blocks.len(), 3);
    }

    #[test]
    fn test_pool_exhaustion() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // P5: Fill pool to reservation limit
        let block_size = 65536;
        let mut pool =
            Pool::with_config(16, block_size, &BinnedAllocatorConfig::default()).unwrap();

        // Manually set committed to limit
        pool.committed = POOL_RESERVED_SIZE;

        // Next alloc should fail (assuming no free bins)
        let result = pool.alloc();
        assert!(result.is_err());
    }

    #[test]
    fn test_pool_beyond_single_bittree() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Verify that pools requiring > 16384 blocks are now accepted
        // thanks to BitTreeChain. 256MB / 4KB = 65536 blocks → 4 segments.
        let config = BinnedAllocatorConfig {
            pool_reserved_size: 256 * 1024 * 1024,
            ..Default::default()
        };
        let block_size = 4 * 1024;
        let res = Pool::with_config(16, block_size, &config);
        assert!(
            res.is_ok(),
            "Pool creation should succeed with > 16384 blocks"
        );

        let mut pool = res.unwrap();
        // Basic alloc/free should work
        let ptr = pool.alloc().unwrap();
        pool.free(ptr);
    }

    #[test]
    fn test_pool_free_then_alloc_fills_same_bin() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // P7: Free one bin, alloc again — verify returned pointer matches freed bin (LIFO)
        let mut pool = Pool::with_config(16, 65536, &BinnedAllocatorConfig::default()).unwrap();
        let p1 = pool.alloc().unwrap();
        let addr1 = p1.as_ptr() as usize;

        pool.free(p1);

        let p2 = pool.alloc().unwrap();
        let addr2 = p2.as_ptr() as usize;

        assert_eq!(addr1, addr2, "Should recycle immediately freed bin");
    }

    #[test]
    fn test_pool_alloc_after_all_freed() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // P8: Alloc all bins in block, free all, alloc one — verify block reuse
        let block_size = 65536;
        let bin_size = 64; // Fewer bins to iter
        let bins_per_block = block_size / bin_size;

        let mut pool =
            Pool::with_config(bin_size, block_size, &BinnedAllocatorConfig::default()).unwrap();
        let mut ptrs = Vec::new();

        for _ in 0..bins_per_block {
            ptrs.push(pool.alloc().unwrap());
        }

        // Block full
        assert!(pool.bit_tree.find_free().is_none());

        // Free all
        for p in ptrs {
            pool.free(p);
        }

        // Should be free now
        assert!(pool.bit_tree.find_free().is_some());

        // Alloc one
        let new_p = pool.alloc().unwrap();
        // Should succeed and be in the same block/region
        // We can check if it's within block 0 range.
        let base_addr = pool.base.as_ptr() as usize;
        let new_addr = new_p.as_ptr() as usize;
        assert!(new_addr >= base_addr && new_addr < base_addr + block_size);
    }

    #[test]
    fn test_pool_multiple_blocks_no_duplicate_ptrs() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // P9: Alloc across 3+ blocks, collect all ptrs — verify uniqueness
        let block_size = 65536;
        let bin_size = 4096; // 16 bins per block
        let mut pool =
            Pool::with_config(bin_size, block_size, &BinnedAllocatorConfig::default()).unwrap();
        let count = 16 * 3 + 5; // 3 blocks full + 5 items

        let mut ptrs = std::collections::HashSet::new();
        for _ in 0..count {
            let p = pool.alloc().unwrap().as_ptr() as usize;
            assert!(ptrs.insert(p), "Duplicate pointer returned: {p:x}");
        }
    }

    // --- ThreadCache Tests (TC1-TC3) ---

    #[test]
    fn test_thread_cache_empty_pop() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // TC1: Pop from empty cache — returns None
        let mut cache = ThreadCache::new();
        // bin 0 (size 16)
        assert!(cache.bins[0].pop().is_none());
    }

    #[test]
    fn test_thread_cache_flush_returns_to_correct_pool() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // TC2: After flush, verify pool's freelist grew (or we can alloc from it)
        let allocator = BinnedAllocator::new().unwrap();
        let mut cache = ThreadCache::new();

        // Pull items into cache
        let size = 16;
        let layout = std::alloc::Layout::from_size_align(size, 1).unwrap();
        let ptr = allocator.alloc_with_cache(&mut cache, layout).unwrap();

        // Free to cache
        allocator.free_with_cache(&mut cache, ptr, layout);

        // Fill cache to trigger flush (MAX_CACHE_SIZE = 64)
        // We need to push 64 items to cache.
        // We can just manually push to cache bin if we had access, but we don't.
        // So we alloc and free 65 times.
        let mut ptrs = Vec::new();
        for _ in 0..70 {
            ptrs.push(allocator.alloc_with_cache(&mut cache, layout).unwrap());
        }

        // Now free them all to cache
        for p in ptrs {
            allocator.free_with_cache(&mut cache, p, layout);
        }

        // The cache should have flushed some to pool.
        // We can verify by allocating from a FRESH cache or direct from pool (if exposed)
        // thread `cache` now has some items.
        // pool has some items returned.

        // Let's create a secondary cache
        let mut cache2 = ThreadCache::new();
        // Alloc from it. It will go to pool. Pool should have items.
        let p_new = allocator.alloc_with_cache(&mut cache2, layout).unwrap();
        // Safety: Test code.
// Safety: Test code.
        unsafe {
            *p_new.as_ptr() = 0xFF;
        }
    }

    #[test]
    fn test_thread_cache_mixed_size_classes() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // TC3: Alloc/free different sizes through cache — verify no cross-contamination
        let allocator = BinnedAllocator::new().unwrap();
        let mut cache = ThreadCache::new();
// Safety: Test code.
        unsafe {
            cache.bind(std::mem::transmute::<&binned::BinnedAllocator, &binned::BinnedAllocator>(&allocator));
        }

        let size_a = 16;
        let size_b = 32;
        let layout_a = std::alloc::Layout::from_size_align(size_a, 1).unwrap();
        let layout_b = std::alloc::Layout::from_size_align(size_b, 1).unwrap();

        let p_a = allocator.alloc_with_cache(&mut cache, layout_a).unwrap();
        let p_b = allocator.alloc_with_cache(&mut cache, layout_b).unwrap();

        // Safety: Test code.
// Safety: Test code.
        unsafe {
            *p_a.as_ptr() = 0xAA;
            *p_b.as_ptr() = 0xBB;
        }

        allocator.free_with_cache(&mut cache, p_a, layout_a);
        allocator.free_with_cache(&mut cache, p_b, layout_b);

        // Realloc A
        let p_a_first_realloc = allocator.alloc_with_cache(&mut cache, layout_a).unwrap();
        assert_eq!(p_a_first_realloc, p_a); // Should reuse A
        // Verify content hasn't been overwritten by B logic
        // (Memory reuse might have dirty data, but we just check pointer identity here mostly)

        // Realloc B
        let p_b_second_realloc = allocator.alloc_with_cache(&mut cache, layout_b).unwrap();
        assert_eq!(p_b_second_realloc, p_b);
    }

    #[test]
    fn test_pool_smallest_bin_size() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // P7: Smallest bin size (must be at least 16 bytes for recycler links)
        let block_size = 65536;
        let mut pool = Pool::with_config(
            16,
            block_size,
            &BinnedAllocatorConfig::default(),
        )
        .unwrap();

        // Keep an anchor allocated so sparse decommit doesn't fire
        let p1 = pool.alloc().unwrap();
        let p2 = pool.alloc().unwrap();
        let _anchor = pool.alloc().unwrap();

        pool.free(p1);
        pool.free(p2);

        // Also test through LocalFreeList path manually using the (still committed) ptrs
        let mut list = LocalFreeList::new();
        list.push(p1); // Writes 8 bytes (usize)
        list.push(p2);

        assert_eq!(list.pop(), Some(p2));
        assert_eq!(list.pop(), Some(p1));
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "smaller than minimum required 16")]
    fn test_pool_bin_size_too_small() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Should panic due to assert bin_size >= size_of::<usize>()
        drop(Pool::with_config(4, 65536, &BinnedAllocatorConfig::default()));
    }

    #[test]
    fn test_pool_alloc_writes_dont_corrupt_freelist() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // P6: Alloc/Write/Free/Realloc
        let mut pool = Pool::with_config(16, 65536, &BinnedAllocatorConfig::default()).unwrap(); // 64KB block size to satisfy BitTree limit

        let p1 = pool.alloc().unwrap();
        // Safety: Test code.
// Safety: Test code.
        unsafe {
            let slice = std::slice::from_raw_parts_mut(p1.as_ptr(), 16);
            slice.fill(0xAA);
        }

        pool.free(p1);

        // p1 memory is now used for next pointer.
        // Validating that realloc returns it and it's usable.
        let p2 = pool.alloc().unwrap();
        assert_eq!(p1, p2);

        // Safety: Test code.
// Safety: Test code.
        unsafe {
            // Should be overwritable again
            let slice = std::slice::from_raw_parts_mut(p2.as_ptr(), 16);
            slice.fill(0xBB);
        }
    }

    #[test]
    fn test_pool_largest_bin_size() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // P8: 65536-byte bin
        let block_size = 65536; // 1 bin per block
        let mut pool =
            Pool::with_config(65536, block_size, &BinnedAllocatorConfig::default()).unwrap();
        let p = pool.alloc().unwrap();
        // Safety: Test code.
// Safety: Test code.
        unsafe {
            let slice = std::slice::from_raw_parts_mut(p.as_ptr(), 65536);
            slice[0] = 1;
            slice[65535] = 2;
        }
        pool.free(p);
    }

    #[test]
    fn test_aligned_alloc() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let alloc = BinnedAllocator::new().unwrap();

        // Case 1: Size 48, Align 32
        // 48 is not 32-aligned. Next size class is 64.
        let layout1 = std::alloc::Layout::from_size_align(48, 32).unwrap();
        let ptr1 = alloc.alloc(layout1).unwrap();
        let addr1 = ptr1.as_ptr() as usize;
        assert_eq!(addr1 % 32, 0, "Ptr {ptr1:p} should be 32-byte aligned");

        // Check size class logic: size_class(48, 32) should be index for 64.
        let idx1 = BinnedAllocator::size_class(48, 32);
        assert_eq!(SIZE_CLASSES[idx1], 64);

        // Safety: Test code.
// Safety: Test code.
        unsafe { alloc.free(ptr1, layout1); }

        // Case 2: Size 16, Align 64
        // Must pick size class multiple of 64. Smallest is 64.
        let layout2 = std::alloc::Layout::from_size_align(16, 64).unwrap();
        let ptr2 = alloc.alloc(layout2).unwrap();
        let addr2 = ptr2.as_ptr() as usize;
        assert_eq!(addr2 % 64, 0, "Ptr {ptr2:p} should be 64-byte aligned");

        let idx2 = BinnedAllocator::size_class(16, 64);
        assert_eq!(SIZE_CLASSES[idx2], 64);

        // Safety: Test code.
// Safety: Test code.
        unsafe { alloc.free(ptr2, layout2); }
    }

    // --- BinnedAllocator Tests (A1-A11) ---

    #[test]
    fn test_size_class_boundaries() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // A1: Verify size_class mapping

        // 16 -> 0
        assert_eq!(BinnedAllocator::size_class(1, 1), 0);
        assert_eq!(BinnedAllocator::size_class(16, 1), 0);

        // 17 -> 1 (32)
        assert_eq!(BinnedAllocator::size_class(17, 1), 1);
        assert_eq!(BinnedAllocator::size_class(32, 1), 1);

        // 65536 -> last
        let last_idx = SIZE_CLASSES.len() - 1;
        assert_eq!(BinnedAllocator::size_class(65536, 1), last_idx);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "Size 0 not supported")]
    fn test_size_class_zero() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // A2: size_class(0) — now panics
        BinnedAllocator::size_class(0, 1);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "too large")]
    fn test_size_class_too_large() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // A3: size_class(65537) panics
        BinnedAllocator::size_class(65537, 1);
    }

    #[test]
    fn test_alloc_all_size_classes() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // A4: Alloc+free from every size class
        let alloc = BinnedAllocator::new().unwrap();

        for &size in SIZE_CLASSES {
            let ptr = alloc.alloc_bytes(size).unwrap();
            // Safety: Test code.
// Safety: Test code.
            unsafe {
                ptr.as_ptr().write_bytes(0xCC, size);
            }
            // Safety: Test code.
// Safety: Test code.
            unsafe { alloc.free_bytes(ptr, size); }
        }
    }

    #[test]
    fn test_alloc_alignment() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // A5: Alloc various sizes, verify alignment
        let alloc = BinnedAllocator::new().unwrap();

        for &size in &[16, 64, 256, 4096] {
            let ptr = alloc.alloc_bytes(size).unwrap();
            let addr = ptr.as_ptr() as usize;
            // Alignment should be at least 16 (min bin)
            // But usually we align to bin size up to page size?
            // Current `Pool` impl allocates at `block_offset + bin_idx * bin_size`.
            // `base` is page aligned. `bin_size` is multiple of 16.
            // So addr % 16 == 0.
            assert_eq!(addr % 16, 0);
            // Safety: Test code.
// Safety: Test code.
            unsafe { alloc.free_bytes(ptr, size); }
        }
    }

    #[test]
    fn test_alloc_no_overlap() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // A6: Alloc 100 items of same size
        let alloc = BinnedAllocator::new().unwrap();
        let mut ptrs = Vec::new();

        for _ in 0..100 {
            ptrs.push(alloc.alloc_bytes(32).unwrap());
        }

        for i in 0..ptrs.len() {
            for j in i + 1..ptrs.len() {
                assert_ne!(ptrs[i], ptrs[j]);
            }
        }

        for p in ptrs {
            // Safety: Test code.
// Safety: Test code.
            unsafe { alloc.free_bytes(p, 32); }
        }
    }

    #[test]
    fn test_thread_cache_refill_batch() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // A7: Exhaust cache, trigger refill
        let alloc = BinnedAllocator::new().unwrap();
        let mut cache = ThreadCache::new();
// Safety: Test code.
        unsafe {
            cache.bind(std::mem::transmute::<&binned::BinnedAllocator, &binned::BinnedAllocator>(&alloc));
        }

        // 1. Alloc small item (size 32) -> batch size 16
        let layout1 = std::alloc::Layout::from_size_align(32, 1).unwrap();
        let _p1 = alloc.alloc_with_cache(&mut cache, layout1).unwrap();
        let idx1 = BinnedAllocator::size_class(32, 1);
        assert_eq!(cache.bins[idx1].count, 15); // Popped 1, refilled 16

        // 2. Alloc large item (size 64KB) -> batch size 2
        let layout2 = std::alloc::Layout::from_size_align(65536, 1).unwrap();
        let _p2 = alloc.alloc_with_cache(&mut cache, layout2).unwrap();
        let idx2 = BinnedAllocator::size_class(65536, 1);
        assert_eq!(cache.bins[idx2].count, 1); // Popped 1, refilled 2
    }

    #[test]
    fn test_thread_cache_flush() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // A8: Verify bundle-based flush for different sizes.
        // When cache exceeds limit, the entire list is detached and pushed to
        // the GlobalRecycler (or flushed to pool if recycler is full).
        // After flush, the TLS cache is empty.
        let alloc = BinnedAllocator::new().unwrap();
        let mut cache = ThreadCache::new();
// Safety: Test code.
        unsafe {
            cache.bind(std::mem::transmute::<&binned::BinnedAllocator, &binned::BinnedAllocator>(&alloc));
        }

        // 1. Small size (32 bytes) -> limit 64
        let size1 = 32;
        let layout1 = std::alloc::Layout::from_size_align(size1, 1).unwrap();
        let idx1 = BinnedAllocator::size_class(size1, 1);
        let mut ptrs1 = Vec::new();
        for _ in 0..65 {
            ptrs1.push(alloc.alloc_bytes(size1).unwrap());
        }
        for p in ptrs1 {
            alloc.free_with_cache(&mut cache, p, layout1);
        }
        // After exceeding limit 64 at count=65, entire bundle flushed.
        // The 65th free triggers flush, then cache is empty.
        assert_eq!(cache.bins[idx1].count, 0);

        // 2. Large size (64KB) -> limit 4
        let size2 = 65536;
        let layout2 = std::alloc::Layout::from_size_align(size2, 1).unwrap();
        let idx2 = BinnedAllocator::size_class(size2, 1);
        let mut ptrs2 = Vec::new();
        for _ in 0..5 {
            ptrs2.push(alloc.alloc_bytes(size2).unwrap());
        }
        for p in ptrs2 {
            alloc.free_with_cache(&mut cache, p, layout2);
        }
        // After exceeding limit 4 at count=5, entire bundle flushed.
        assert_eq!(cache.bins[idx2].count, 0);
    }

    #[test]
    fn test_thread_cache_cross_thread() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // A9: Alloc on thread A, free on thread B
        let alloc = Arc::new(BinnedAllocator::new().unwrap());

        let alloc2 = alloc.clone();
        let t = thread::spawn(move || alloc2.alloc_bytes(64).unwrap().as_ptr() as usize);

        let ptr_addr = t.join().unwrap();
        let ptr = NonNull::new(ptr_addr as *mut u8).unwrap();

        // Free on this thread (using direct free implies no cache, or use cache)
        // Test asked for "without cache — direct path" for Free?
        // `BinnedAllocator::free` is direct.
        // Safety: Test code.
// Safety: Test code.
        unsafe { alloc.free_bytes(ptr, 64); }
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "bound to a different allocator")]
    fn test_thread_cache_mismatch() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let alloc1 = Box::leak(Box::new(BinnedAllocator::new().unwrap()));
        let alloc2 = BinnedAllocator::new().unwrap();

        let mut cache = ThreadCache::new();
        cache.bind(alloc1);

        let layout = std::alloc::Layout::from_size_align(32, 1).unwrap();
        // Should panic
        drop(alloc2.alloc_with_cache(&mut cache, layout));
    }

    #[test]
    fn test_global_binned_double_init() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // A10: Call GlobalBinnedAllocator::init() twice
        // Ignore result of first init (might be from other tests)
        drop(GlobalBinnedAllocator::init());

        // Second call should definitely fail
        assert!(GlobalBinnedAllocator::init().is_err());
    }

    #[test]
    fn test_global_binned_alloc_free_multithread() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // A11: Multiple threads using GlobalBinnedAllocator
        // Init
        drop(GlobalBinnedAllocator::init());

        let mut handles = vec![];
        for _ in 0..4 {
            handles.push(thread::spawn(|| {
                for _ in 0..50 {
                    let ptr = GlobalBinnedAllocator::alloc_bytes(32).unwrap();
                    // Safety: Test code.
// Safety: Test code.
                    unsafe {
                        *ptr.as_ptr() = 1;
                    }
                    // Safety: Test code.
// Safety: Test code.
                    unsafe { GlobalBinnedAllocator::free_bytes(ptr, 32); }
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_thread_cache_drop_flushes() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Ensure global allocator is initialized
        drop(GlobalBinnedAllocator::init());
        let allocator = GlobalBinnedAllocator::get();
        let size = 128; // Use 128 to avoid contention with other tests using 32/48
        let pool_idx = BinnedAllocator::size_class(size, 1);

        // 0. Pre-trim to ensure baseline is clean from previous tests in this thread
        GlobalBinnedAllocator::trim();

        // 1. Get initial free count in pool
        let initial_free = {
            let guard = allocator.pools[pool_idx].lock().unwrap();
            guard.pools
                .iter()
                .flat_map(|p| p.blocks.iter())
                .map(|b| u32::from(b.free_count()))
                .sum::<u32>()
        };

        // 2. Spawn thread, alloc and free to cache, then exit
        let handle = crate::sync::thread::spawn(move || {
            let ptr = GlobalBinnedAllocator::alloc_bytes(size).unwrap();
            // ptr is now removed from pool (along with refill batch of 15 others)
            // Safety: Test code.
// Safety: Test code.
            unsafe { GlobalBinnedAllocator::free_bytes(ptr, size); }
            // ptr (and others) are now in this thread's cache
        });
        handle.join().unwrap();

        // 3. Check free count again.
        // If Drop worked, all items (including refill batch) should be back in the pool.
        let final_free = {
            let guard = allocator.pools[pool_idx].lock().unwrap();
            guard.pools
                .iter()
                .flat_map(|p| p.blocks.iter())
                .map(|b| u32::from(b.free_count()))
                .sum::<u32>()
        };

        // If a new block was allocated, it should be fully free now.
        // If no new block was allocated, it should be back to initial_free.
        assert!(final_free >= initial_free);

        // More specifically, if we know the batch size is 16 for small bins (128 bytes):
        // Before thread: N free
        // In thread: alloc(128) -> pool.alloc() called 16 times. Pool free count = N - 16
        // In thread: free(ptr) -> cache count = 1.
        // Thread exit -> cache flushes 1 item + 15 in refill batch. Pool free count = (N - 16) + 16 = N.
        assert_eq!(final_free % 16, initial_free % 16);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "Double free detected")]
    fn test_pool_double_free() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // P10: Detect double free. Keep a second bin to prevent sparse decommit.
        let mut pool = Pool::with_config(16, 65536, &BinnedAllocatorConfig::default()).unwrap();
        let ptr = pool.alloc().unwrap();
        let _anchor = pool.alloc().unwrap();

        pool.free(ptr);
        pool.free(ptr); // Should panic with "Double free detected"
    }

    #[test]
    fn test_pool_trim() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let block_size = 65536;
        let mut pool =
            Pool::with_config(16, block_size, &BinnedAllocatorConfig::default()).unwrap();

        // Fill 2 blocks
        let bins_per_block = block_size / 16;
        let mut ptrs = Vec::new();
        for _ in 0..bins_per_block * 2 {
            ptrs.push(pool.alloc().unwrap());
        }

        assert_eq!(pool.blocks.len(), 2);
        assert_eq!(pool.committed, block_size * 2);

        // Free all items in block 1 (the last one).
        // With deferred decommit, the block is queued but still committed.
        for &ptr in &ptrs[bins_per_block .. bins_per_block * 2] {
            pool.free(ptr);
        }
        // Block 1 is still committed (decommit deferred)
        assert_eq!(pool.committed, block_size * 2);

        // trim() processes pending decommits first, then pops trailing empty blocks
        pool.trim();

        assert_eq!(pool.blocks.len(), 1);
        assert_eq!(pool.committed, block_size);

        // Free block 0
        for &ptr in &ptrs[0 .. bins_per_block] {
            pool.free(ptr);
        }
        // Block 0 still committed (decommit deferred)
        assert_eq!(pool.committed, block_size);

        pool.trim();
        assert_eq!(pool.blocks.len(), 0);
        assert_eq!(pool.committed, 0);
    }

    #[test]
    fn test_pool_trim_mixed_blocks() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Scenario with sparse decommit:
        // Block 0: fully used
        // Block 1: empty (intermediate) — sparse-decommitted on free
        // Block 2: fully used
        // Block 3: empty (trailing) — sparse-decommitted on free
        //
        // After sparse decommit: committed = 2 * block_size (only blocks 0, 2)
        // trim() removes trailing decommitted block 3 from the Vec.
        // Block 1 stays (intermediate, can't be popped).

        let block_size = 65536;
        let bin_size = 4096; // 16 bins per block
        let mut pool =
            Pool::with_config(bin_size, block_size, &BinnedAllocatorConfig::default()).unwrap();

        // 1. Fill 4 blocks completely
        let mut ptrs = Vec::new();
        for _ in 0..16 * 4 {
            ptrs.push(pool.alloc().unwrap());
        }
        assert_eq!(pool.blocks.len(), 4);
        assert_eq!(pool.committed, block_size * 4);

        // 2. Free all in Block 1 (intermediate) — queued for deferred decommit
        for &ptr in &ptrs[16..32] {
            pool.free(ptr);
        }
        // Decommit deferred — block still committed
        assert!(pool.blocks[1].is_committed());
        assert_eq!(pool.committed, block_size * 4);

        // 3. Free all in Block 3 (trailing) — queued for deferred decommit
        for &ptr in &ptrs[48..64] {
            pool.free(ptr);
        }
        assert!(pool.blocks[3].is_committed());
        assert_eq!(pool.committed, block_size * 4);

        // 4. Trim — processes pending decommits first, then removes trailing block 3
        pool.trim();

        assert_eq!(
            pool.blocks.len(),
            3,
            "Trim should have removed trailing block"
        );
        assert_eq!(pool.committed, block_size * 2);

        // Block 1 is decommitted-empty but stays (intermediate)
        assert_eq!(pool.blocks[1].free_count(), 16);
        assert!(!pool.blocks[1].is_committed());
        // Block 2 is still fully used
        assert_eq!(pool.blocks[2].free_count(), 0);
    }

    #[cfg(debug_assertions)]
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "does not belong to this Pool")]
    fn test_pool_free_out_of_range() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let mut pool = Pool::with_config(16, 65536, &BinnedAllocatorConfig::default()).unwrap();
        // Safety: Test code.
        let invalid_ptr = unsafe { NonNull::new_unchecked(std::ptr::dangling_mut::<u8>()) };
        pool.free(invalid_ptr);
    }

    #[cfg(debug_assertions)]
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "belongs to an uncommitted block")]
    fn test_pool_free_uncommitted() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let mut pool = Pool::with_config(16, 65536, &BinnedAllocatorConfig::default()).unwrap();
        // Skip block 0, point to block 1
        // Safety: Test code.
        let uncommitted_ptr = unsafe { NonNull::new_unchecked(pool.base.as_ptr().add(65536)) };
        pool.free(uncommitted_ptr);
    }

    #[cfg(debug_assertions)]
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "is not aligned to bin size")]
    fn test_pool_free_misaligned() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let mut pool = Pool::with_config(16, 65536, &BinnedAllocatorConfig::default()).unwrap();
        let ptr = pool.alloc().unwrap();
        // Safety: Test code.
        let misaligned_ptr = unsafe { NonNull::new_unchecked(ptr.as_ptr().add(1)) };
        pool.free(misaligned_ptr);
    }

    #[cfg(debug_assertions)]
    #[test]
    fn test_double_free_caught_through_cache() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Verify that a double free through cache is caught by pool.free_map.
        // Keep an anchor allocation so the block doesn't get sparse-decommitted.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let allocator = BinnedAllocator::new().unwrap();
            let allocator_static: &'static BinnedAllocator =
// Safety: Test code.
                unsafe { &*((&raw const allocator).cast::<BinnedAllocator>()) };
            let mut cache = ThreadCache::new();
            cache.bind(allocator_static);
            let size = 32;
            let layout = std::alloc::Layout::from_size_align(size, 1).unwrap();

            // Allocate an anchor to prevent sparse decommit
            let _anchor = allocator.alloc_with_cache(&mut cache, layout).unwrap();

            let ptr = allocator.alloc_with_cache(&mut cache, layout).unwrap();
            allocator.free_with_cache(&mut cache, ptr, layout);
            cache.flush(); // ptr returned to pool

            allocator.free_with_cache(&mut cache, ptr, layout);
            cache.flush(); // pool.free_map detects double free → panic
        }));
        assert!(result.is_err(), "Expected double-free panic");
    }

    #[cfg(debug_assertions)]
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "does not belong to any pool in this chain")]
    fn test_binned_allocator_cross_pool_free() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let alloc = BinnedAllocator::new().unwrap();

        // 1. Alloc 16 bytes (Pool 0)
        let ptr1 = alloc.alloc_bytes(16).unwrap();

        // 2. Init Pool 1 (32 bytes) so the free path enters Pool::free
        let ptr2 = alloc.alloc_bytes(32).unwrap();
// Safety: Test code.
        unsafe { alloc.free_bytes(ptr2, 32); }

        // 3. Try to free ptr1 (from Pool 0) as 32 bytes (Pool 1)
        // Since ptr1 is in pool 0's VA range, pool 1's range check will catch it.
// Safety: Test code.
        unsafe { alloc.free_bytes(ptr1, 32); }
    }

    #[test]
    fn test_sustained_pressure() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        drop(GlobalBinnedAllocator::init());
        let num_threads = 16;
        let allocs_per_thread = 500;
        let barrier = Arc::new(crate::sync::barrier::Barrier::new(num_threads));
        let mut handles = vec![];

        for t in 0..num_threads {
            let b = barrier.clone();
            handles.push(crate::sync::thread::spawn(move || {
                let mut ptrs = Vec::with_capacity(allocs_per_thread);

                // 1. Concurrent allocation phase
                for i in 0..allocs_per_thread {
                    let size = 256 + (i % 128); // Use larger sizes to avoid contention
                    let ptr = GlobalBinnedAllocator::alloc_bytes(size).unwrap();
// Safety: Test code.
                    unsafe {
                        ptr.as_ptr().write((t ^ i).to_le_bytes()[0]);
                    }
                    ptrs.push((ptr, size));
                }

                // 2. Sustained holding phase: all threads wait here while holding all memory
                b.wait();

                // 3. Verification phase
                for (i, (ptr, _)) in ptrs.iter().enumerate() {
// Safety: Test code.
                    unsafe {
                        assert_eq!(ptr.as_ptr().read(), (t ^ i).to_le_bytes()[0]);
                    }
                }

                // 4. Cleanup
                for (ptr, size) in ptrs {
                    // Safety: Test code.
// Safety: Test code.
            unsafe { GlobalBinnedAllocator::free_bytes(ptr, size); }
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_binned_allocator_chains_correctly() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let allocator = BinnedAllocator::new().unwrap();
        let pool_idx = BinnedAllocator::size_class(16, 1);

        // Manually create a pool with committed set to the limit (no free blocks)
        // and insert it into the chain.
        {
            let mut guard = allocator.pools[pool_idx].lock().unwrap();
            let bin_size = SIZE_CLASSES[pool_idx];
            let mut pool = Pool::with_config(
                bin_size,
                allocator.block_size,
                &BinnedAllocatorConfig::default(),
            )
            .unwrap();
            pool.committed = POOL_RESERVED_SIZE;
            
            // Because committed == reserved_size, probe_commit_needed will return None (or fail to alloc)
            // But we didn't mark bit_tree full. 
            // The Pool::alloc logic checks: if bit_tree has free -> alloc.
            // if not free -> if committed + block > reserved -> Err.
            // So we need to ensure bit_tree is empty (it is, new pool)
            
            guard.pools.clear();
            guard.pools.push(pool);
            guard.active_index = 0;
        }

        // Next alloc should SUCCEED by adding a new pool to the chain
        let result = allocator.alloc_bytes(16);
        assert!(result.is_ok(), "Allocator should chain new pool when first is exhausted");
        
        // Verify chain grew
        {
             let guard = allocator.pools[pool_idx].lock().unwrap();
             assert_eq!(guard.pools.len(), 2, "Chain should have 2 pools now");
             assert_eq!(guard.active_index, 1, "Active index should be 1");
        }
    }

    #[test]
    fn test_local_free_list_integrity() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Demonstrate that LocalFreeList writes a full usize (8 bytes).
        // If we had 4-byte bins, this would cause a buffer overflow.
        // Use u64 array to ensure 8-byte alignment for the pointer.
        let mut fake_memory = [0xAAAA_AAAA_AAAA_AAAAu64; 2];
        let base_ptr = fake_memory.as_mut_ptr().cast::<u8>();

        // "bin" start at offset 0.
        // Safety: Test code.
        let bin_ptr = unsafe { NonNull::new_unchecked(base_ptr) };

        let mut list = LocalFreeList::new();
        list.push(bin_ptr); // Writes 8 bytes to index 0..8 (replaces first u64)

        // Index 8..16 WOULD be the next bin if bin_size was 8.
        // It should be UNTOUCHED by the push to offset 0.
        assert_eq!(fake_memory[1], 0xAAAA_AAAA_AAAA_AAAAu64);

        // But if we "simulate" a 4-byte bin by pointing to the middle...
        // Wait, the demonstration is about writing 8 bytes.
        // If we write at offset 0, we affect 0..8.
        // If bins were 4 bytes, bin 0 is 0..4, bin 1 is 4..8.
        // Writing 8 bytes to bin 0 (0..8) overwrites bin 1 (4..8).
        let first_u64_after = fake_memory[0];
        assert_ne!(first_u64_after, 0xAAAA_AAAA_AAAA_AAAAu64);

        // Let's explicitly check the 4..8 range (the "next bin")
        let second_half_of_first_u64 = (first_u64_after >> 32) as u32;
        // If it only wrote 4 bytes, this half would still be 0xAAAAAAAA.
        assert_ne!(second_half_of_first_u64, 0xAAAA_AAAA);
    }

    // --- T4: Pool::trim no longer panics on decommit failure (B3 fix) ---
    #[test]
    fn test_pool_trim_is_best_effort() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // B3 fix: Pool::trim checks decommit result. If it fails, it stops trimming but doesn't panic.
        // Verify that trim runs to completion (when decommit succeeds) and leaves consistent state.
        let block_size = 65536;
        let mut pool =
            Pool::with_config(16, block_size, &BinnedAllocatorConfig::default()).unwrap();
        let bins = block_size / 16;

        let mut ptrs = Vec::new();
        for _ in 0..bins {
            ptrs.push(pool.alloc().unwrap());
        }
        for p in ptrs {
            pool.free(p);
        }

        // Trim should succeed (decommit is best-effort)
        pool.trim();
        assert_eq!(pool.blocks.len(), 0);
        assert_eq!(pool.committed, 0);

        // Can re-allocate after trim
        let _p = pool.alloc().unwrap();
        assert_eq!(pool.blocks.len(), 1);
        assert_eq!(pool.committed, block_size);
    }

    // --- T5: size_class boundary correctness ---
    #[test]
    fn test_size_class_all_boundaries() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let _alloc = BinnedAllocator::new().unwrap();

        for i in 0..SIZE_CLASSES.len() {
            let sc = SIZE_CLASSES[i];
            // Exact boundary maps to its own class
            assert_eq!(
                BinnedAllocator::size_class(sc, 1),
                i,
                "size_class({sc}) should be {i}",
            );

            if i > 0 {
                // One byte below: still maps to class i (gap between classes ≥ 16)
                assert_eq!(
                    BinnedAllocator::size_class(sc - 1, 1),
                    i,
                    "size_class({}) should be {} (same class, gap≥16)",
                    sc - 1,
                    i
                );

                // Previous class's exact boundary maps to i-1
                assert_eq!(
                    BinnedAllocator::size_class(SIZE_CLASSES[i - 1], 1),
                    i - 1,
                    "size_class({}) should be {}",
                    SIZE_CLASSES[i - 1],
                    i - 1
                );
            }

            // One byte above should map to next class (unless last)
            if i < SIZE_CLASSES.len() - 1 {
                assert_eq!(
                    BinnedAllocator::size_class(sc + 1, 1),
                    i + 1,
                    "size_class({}) should be {}",
                    sc + 1,
                    i + 1
                );
            }
        }
    }

    #[test]
    fn test_size_class_lut_matches_binary_search() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Verify the O(1) LUT produces identical results to the O(log n) search
        for size in 1..=MAX_SMALL_SIZE {
            let lut_result = SIZE_CLASS_LUT[(size + 15) >> 4] as usize;
            let search_result = SIZE_CLASSES.partition_point(|&c| c < size);
            assert_eq!(
                lut_result, search_result,
                "LUT and binary search disagree for size {size}",
            );
        }
    }

    // --- T6: ThreadCache Drop with poisoned mutex ---
    #[test]
    fn test_thread_cache_drop_with_poisoned_mutex() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // P8 fix: ThreadCache::Drop recovers from poisoned mutexes.
        // Spawn a thread that panics while holding a pool lock, poisoning it.
        // Then verify ThreadCache Drop on main thread handles it gracefully.
        let allocator: &'static BinnedAllocator =
            Box::leak(Box::new(BinnedAllocator::new().unwrap()));

        let pool_idx = BinnedAllocator::size_class(32, 1);

        // Poison the mutex by panicking while holding the lock
        drop(thread::spawn(move || {
            let _guard = allocator.pools[pool_idx].lock().unwrap();
            panic!("intentional panic to poison mutex");
        })
        .join());

        // The pool mutex for size class 32 is now poisoned.
        // Verify ThreadCache Drop handles it (doesn't crash).
        let mut cache = ThreadCache::new();
        cache.bind(allocator);
        // Allocate some items that go through a different pool (size 64, not 32)
        // to avoid the poisoned pool on the alloc path.
        let layout64 = std::alloc::Layout::from_size_align(64, 1).unwrap();
        let ptr = allocator.alloc_with_cache(&mut cache, layout64).unwrap();
        allocator.free_with_cache(&mut cache, ptr, layout64);
        // Also manually push a fake-valid pointer into the poisoned bin
        // to exercise the recovery path during Drop.
        let ptr32 = allocator.alloc_bytes(32).unwrap(); // Uses poisoned mutex — recovers
        cache.bins[pool_idx].push(ptr32);

        // Drop cache — must not crash despite poisoned mutex for bin 32
        drop(cache);
    }

    // --- T9: 64+ concurrent threads stress test ---
    #[test]
    fn test_64_thread_stress() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let allocator = Arc::new(BinnedAllocator::new().unwrap());
        let num_threads = 64;
        let allocs_per_thread = 200;
        let barrier = Arc::new(crate::sync::barrier::Barrier::new(num_threads));
        let mut handles = vec![];

        for t in 0..num_threads {
            let alloc = allocator.clone();
            let b = barrier.clone();
            handles.push(thread::spawn(move || {
                let sizes = [16, 32, 64, 128, 256, 512, 1024, 4096];
                let mut ptrs = Vec::with_capacity(allocs_per_thread);

                b.wait(); // Synchronize all threads

                for i in 0..allocs_per_thread {
                    let size = sizes[i % sizes.len()];
                    let ptr = alloc.alloc_bytes(size).unwrap();
// Safety: Test code.
                    unsafe {
                        ptr.as_ptr().write((t ^ i).to_le_bytes()[0]);
                    }
                    ptrs.push((ptr, size));
                }

                // Verify all held pointers
                for (i, (ptr, _)) in ptrs.iter().enumerate() {
// Safety: Test code.
                    unsafe {
                        assert_eq!(
                            ptr.as_ptr().read(),
                            (t ^ i).to_le_bytes()[0],
                            "Memory corruption detected: thread {t}, alloc {i}",
                        );
                    }
                }

                // Free all
                for (ptr, size) in ptrs {
                    // Safety: Test code.
// Safety: Test code.
            unsafe { alloc.free_bytes(ptr, size); }
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }

    // ========================================================================
    // Phase 5: New tests for improved allocator features
    // ========================================================================

    // --- GlobalRecycler tests ---

    #[test]
    fn test_recycler_push_pop_single() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let recycler = GlobalRecycler::new(4);
        let alloc = BinnedAllocator::new().unwrap();
        let ptr = alloc.alloc_bytes(32).unwrap();

        // Push one bundle (single item)
        assert!(recycler.push(1, ptr, 1).is_none()); // accepted
        // Pop it back
        let popped = recycler.pop(1);
        assert!(popped.is_some());
        assert_eq!(popped.unwrap(), ptr);
        // Pop from empty
        assert!(recycler.pop(1).is_none());

        // Safety: Test code.
// Safety: Test code.
        unsafe { alloc.free_bytes(ptr, 32); }
    }

    #[test]
    fn test_recycler_push_until_full() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let recycler = GlobalRecycler::new(2);
        let alloc = BinnedAllocator::new().unwrap();
        let p1 = alloc.alloc_bytes(32).unwrap();
        let p2 = alloc.alloc_bytes(32).unwrap();
        let p3 = alloc.alloc_bytes(32).unwrap();

        assert!(recycler.push(0, p1, 1).is_none()); // accepted
        assert!(recycler.push(0, p2, 1).is_none()); // accepted
        // Third push should be rejected (max_bundles=2)
        let rejected = recycler.push(0, p3, 1);
        assert!(rejected.is_some());
        assert_eq!(rejected.unwrap(), p3);

// Safety: Test code.
        unsafe { alloc.free_bytes(p1, 32); }
// Safety: Test code.
        unsafe { alloc.free_bytes(p2, 32); }
// Safety: Test code.
        unsafe { alloc.free_bytes(p3, 32); }
    }

    #[test]
    fn test_recycler_cross_thread() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Thread A frees, Thread B allocs from recycler
        let alloc = Arc::new(BinnedAllocator::new().unwrap());
        let alloc2 = alloc.clone();

        // Thread A: alloc and free enough to trigger cache flush → recycler
        let handle = thread::spawn(move || {
            let mut cache = ThreadCache::new();
// Safety: Test code.
            unsafe {
                cache.bind(std::mem::transmute::<
                    &BinnedAllocator,
                    &'static BinnedAllocator,
                >(&*alloc2));
            }
            let layout = std::alloc::Layout::from_size_align(32, 1).unwrap();

            // Fill cache beyond limit to trigger flush to recycler
            let mut ptrs = Vec::new();
            for _ in 0..70 {
                ptrs.push(alloc2.alloc_bytes(32).unwrap());
            }
            for p in ptrs {
                alloc2.free_with_cache(&mut cache, p, layout);
            }
            // Cache flushed to recycler
        });
        handle.join().unwrap();

        // Thread B (main): alloc should pull from recycler
        let mut cache = ThreadCache::new();
// Safety: Test code.
        unsafe {
            cache.bind(std::mem::transmute::<
                &BinnedAllocator,
                &'static BinnedAllocator,
            >(&*alloc));
        }
        let layout = std::alloc::Layout::from_size_align(32, 1).unwrap();
        let ptr = alloc.alloc_with_cache(&mut cache, layout).unwrap();
        // Safety: Test code.
// Safety: Test code.
        unsafe {
            *ptr.as_ptr() = 0xCC;
        }
        alloc.free_with_cache(&mut cache, ptr, layout);
    }

    // --- Large-alloc routing tests ---

    #[test]
    fn test_large_alloc_through_binned() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let alloc = BinnedAllocator::new().unwrap();
        // Previously this would panic; now transparently routes to LargeAllocCache
        let ptr = alloc.alloc_bytes(100_000).unwrap();
        // Safety: Test code.
// Safety: Test code.
        unsafe {
            *ptr.as_ptr() = 0xAA;
        }
// Safety: Test code.
        unsafe { alloc.free_bytes(ptr, 100_000); }
    }

    #[test]
    fn test_large_alloc_with_cache() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let alloc = BinnedAllocator::new().unwrap();
        let mut cache = ThreadCache::new();
// Safety: Test code.
        unsafe {
            cache.bind(std::mem::transmute::<&binned::BinnedAllocator, &binned::BinnedAllocator>(&alloc));
        }
        let layout = std::alloc::Layout::from_size_align(200_000, 1).unwrap();

        let ptr = alloc.alloc_with_cache(&mut cache, layout).unwrap();
        // Safety: Test code.
// Safety: Test code.
        unsafe {
            ptr.as_ptr().write_bytes(0xBB, 200_000);
        }
        alloc.free_with_cache(&mut cache, ptr, layout);
    }

    #[test]
    fn test_mixed_small_large_alloc() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let alloc = BinnedAllocator::new().unwrap();
        let small = alloc.alloc_bytes(64).unwrap();
        let large = alloc.alloc_bytes(128_000).unwrap();

        // Safety: Test code.
// Safety: Test code.
        unsafe {
            *small.as_ptr() = 1;
            *large.as_ptr() = 2;
        }

// Safety: Test code.
        unsafe { alloc.free_bytes(small, 64); }
// Safety: Test code.
        unsafe { alloc.free_bytes(large, 128_000); }
    }

    // --- Canary tests (debug builds only) ---

    #[cfg(debug_assertions)]
    #[test]
    fn test_block_canary_checked() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Verify canary is set on new blocks (they don't panic on access)
        let mut pool = Pool::with_config(16, 65536, &BinnedAllocatorConfig::default()).unwrap();
        let ptr = pool.alloc().unwrap();
        pool.blocks[0].check_canary(); // should not panic
        pool.free(ptr);
    }

    #[cfg(debug_assertions)]
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "block canary")]
    fn test_block_canary_corruption_detected() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let mut pool = Pool::with_config(16, 65536, &BinnedAllocatorConfig::default()).unwrap();
        let _ = pool.alloc().unwrap();
        let _anchor = pool.alloc().unwrap(); // keep block committed
        // Corrupt the canary
        pool.blocks[0].packed ^= 0xFF << 56;
        pool.blocks[0].check_canary(); // should panic
    }

    #[test]
    fn test_free_bin_canary_on_freelist_alloc() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // After free + re-alloc via freelist, canary should be verified
        let mut pool = Pool::with_config(16, 65536, &BinnedAllocatorConfig::default()).unwrap();
        let p1 = pool.alloc().unwrap();
        let _anchor = pool.alloc().unwrap();
        pool.free(p1);
        // Re-alloc should succeed (canary is valid)
        let p2 = pool.alloc().unwrap();
        assert_eq!(p1, p2);
    }

    // --- Sparse decommit tests ---

    #[test]
    fn test_sparse_decommit_on_full_block_free() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let block_size = 65536;
        let bin_size = 4096; // 16 bins per block
        let mut pool =
            Pool::with_config(bin_size, block_size, &BinnedAllocatorConfig::default()).unwrap();

        // Fill 2 blocks
        let mut ptrs = Vec::new();
        for _ in 0..32 {
            ptrs.push(pool.alloc().unwrap());
        }
        assert_eq!(pool.blocks.len(), 2);
        assert_eq!(pool.committed, block_size * 2);

        // Free all bins in block 1
        for &ptr in &ptrs[16..32] {
            pool.free(ptr);
        }

        // Block 1 is queued for decommit — process the queue
        pool.process_pending_decommits();

        // Block 1 should be decommitted (sparse)
        assert!(!pool.blocks[1].is_committed());
        assert_eq!(pool.committed, block_size);

        // Alloc should recommit block 1 (via bit_tree finding it as free)
        let _new_ptr = pool.alloc().unwrap();
        assert!(pool.blocks[1].is_committed());
        assert_eq!(pool.committed, block_size * 2);
    }

    #[test]
    fn test_sparse_decommit_preserves_non_trailing_blocks() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let block_size = 65536;
        let bin_size = 4096; // 16 bins per block
        let mut pool =
            Pool::with_config(bin_size, block_size, &BinnedAllocatorConfig::default()).unwrap();

        // Fill 3 blocks
        let mut ptrs = Vec::new();
        for _ in 0..48 {
            ptrs.push(pool.alloc().unwrap());
        }

        // Free all of block 1 (middle)
        for &ptr in &ptrs[16..32] {
            pool.free(ptr);
        }

        // Process pending decommits
        pool.process_pending_decommits();

        // Block 1 decommitted, but blocks.len() still 3
        assert!(!pool.blocks[1].is_committed());
        assert_eq!(pool.blocks.len(), 3);
        assert!(pool.blocks[0].is_committed());
        assert!(pool.blocks[2].is_committed());
    }

    #[cfg(debug_assertions)]
    #[test]
    fn test_decommitted_block_free_panics() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let block_size = 65536;
        let bin_size = 4096;
        let mut pool =
            Pool::with_config(bin_size, block_size, &BinnedAllocatorConfig::default()).unwrap();

        let mut ptrs = Vec::new();
        for _ in 0..16 {
            ptrs.push(pool.alloc().unwrap());
        }

        // Free all → queues decommit
        for p in &ptrs {
            pool.free(*p);
        }

        // Process pending decommits to actually decommit the block
        pool.process_pending_decommits();

        // Trying to free again should panic (decommitted block check)
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.free(ptrs[0]);
        }));
        assert!(result.is_err());
    }

    // --- Config tests ---

    #[test]
    fn test_custom_config() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let config = BinnedAllocatorConfig {
            cache_count_limits: [32, 16, 4, 2],
            alloc_extra: [8, 4, 2, 1],
            ..Default::default()
        };
        let alloc = BinnedAllocator::with_config(config).unwrap();
        let ptr = alloc.alloc_bytes(64).unwrap();
        // Safety: Test code.
// Safety: Test code.
        unsafe { alloc.free_bytes(ptr, 64); }
    }

    #[test]
    fn test_config_rejects_block_size_exceeding_u16_metadata_capacity() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let config = BinnedAllocatorConfig {
            block_size: 2 * 1024 * 1024,
            ..Default::default()
        };
        match BinnedAllocator::with_config(config) {
            Err(VmError::InitializationFailed(msg)) => {
                assert!(msg.contains("u16::MAX"));
            }
            Err(other) => panic!("Expected InitializationFailed, got {other:?}"),
            Ok(_) => panic!("Expected config validation to fail"),
        }
    }

    // --- BlockMeta packed layout tests ---

    #[test]
    fn test_block_meta_packed_roundtrip() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let mut meta = BlockMeta::new(4096, 4096, true);
        assert_eq!(meta.free_count(), 4096);
        assert_eq!(meta.free_head(), BIN_SENTINEL);
        assert_eq!(meta.bump_cursor(), 0);
        assert!(meta.is_committed());
        meta.check_canary();

        meta.set_free_count(100);
        assert_eq!(meta.free_count(), 100);

        meta.set_free_head(42);
        assert_eq!(meta.free_head(), 42);

        meta.set_bump_cursor(999);
        assert_eq!(meta.bump_cursor(), 999);

        meta.set_committed(false);
        assert!(!meta.is_committed());
        meta.set_committed(true);
        assert!(meta.is_committed());

        // Other fields unaffected
        assert_eq!(meta.free_count(), 100);
        assert_eq!(meta.free_head(), 42);
        assert_eq!(meta.bump_cursor(), 999);
        meta.check_canary(); // still valid
    }

    #[test]
    fn test_block_meta_size() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Verify packed BlockMeta is 8 bytes (in release; debug has FixedBitSet)
        assert_eq!(std::mem::size_of::<u64>(), 8);
        // The struct has a u64 field, so minimum 8 bytes
        #[cfg(not(debug_assertions))]
        assert_eq!(std::mem::size_of::<BlockMeta>(), 8);
    }

    // ========================================================================
    // Phase 5 — Comprehensive testing suite
    // ========================================================================

    // --- 5b: GlobalRecycler multi-threaded stress ---

    #[test]
    fn test_recycler_multithread_stress() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Many threads concurrently push and pop bundles from the recycler.
        // Verifies lock-free Treiber stack correctness under contention.
        let alloc = Arc::new(BinnedAllocator::new().unwrap());
        let recycler = Arc::new(GlobalRecycler::new(64)); // high limit
        let num_threads = 16;
        let ops_per_thread = 200;
        let barrier = Arc::new(crate::sync::barrier::Barrier::new(num_threads));
        let mut handles = vec![];

        for _ in 0..num_threads {
            let alloc = alloc.clone();
            let recycler = recycler.clone();
            let b = barrier.clone();
            handles.push(thread::spawn(move || {
                b.wait();
                let mut owned_ptrs = Vec::new();

                for i in 0..ops_per_thread {
                    if i % 3 != 0 {
                        // Push: allocate a bin and push as a 1-item bundle
                        let ptr = alloc.alloc_bytes(32).unwrap();
                        // Write a null next-pointer (single-item bundle)
// Safety: Test code.
                        unsafe {
                            *ptr.as_ptr().cast::<()>().cast::<usize>() = 0;
                        }
                        if recycler.push(1, ptr, 1).is_some() {
                            // Rejected — keep ownership
                            owned_ptrs.push(ptr);
                        }
                    } else {
                        // Pop: try to get a bundle
                        if let Some(ptr) = recycler.pop(1) {
                            // Verify it's usable
// Safety: Test code.
                            unsafe {
                                *ptr.as_ptr() = 0xCC;
                            }
                            owned_ptrs.push(ptr);
                        }
                    }
                }

                // Drain the recycler for this thread's leftover pushes
                while let Some(ptr) = recycler.pop(1) {
                    owned_ptrs.push(ptr);
                }

                // Free all owned pointers
                for ptr in owned_ptrs {
                    // Safety: Test code.
// Safety: Test code.
        unsafe { alloc.free_bytes(ptr, 32); }
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // Recycler should be empty after all threads finish and drain
        assert!(recycler.pop(1).is_none());
    }

    #[test]
    fn test_recycler_slot_isolation() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Pushing to slot 0 doesn't affect slot 1
        let alloc = BinnedAllocator::new().unwrap();
        let recycler = GlobalRecycler::new(4);

        let p1 = alloc.alloc_bytes(32).unwrap();
        let p2 = alloc.alloc_bytes(32).unwrap();
// Safety: Test code.
        unsafe {
            *p1.as_ptr().cast::<()>().cast::<usize>() = 0;
            *p2.as_ptr().cast::<()>().cast::<usize>() = 0;
        }

        recycler.push(0, p1, 1);
        recycler.push(1, p2, 1);

        // Pop from slot 0 should get p1, not p2
        let got0 = recycler.pop(0).unwrap();
        assert_eq!(got0, p1);
        assert!(recycler.pop(0).is_none());

        let got1 = recycler.pop(1).unwrap();
        assert_eq!(got1, p2);
        assert!(recycler.pop(1).is_none());

// Safety: Test code.
        unsafe { alloc.free_bytes(p1, 32); }
// Safety: Test code.
        unsafe { alloc.free_bytes(p2, 32); }
    }

    // --- 5d: Free-bin canary corruption detection ---

    #[cfg(debug_assertions)]
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "free-bin canary")]
    fn test_free_bin_canary_corruption_detected() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Corrupt the free-bin canary (bytes 4-7) of a freed bin, then
        // verify Pool::alloc panics on the next freelist allocation.
        let mut pool = Pool::with_config(16, 65536, &BinnedAllocatorConfig::default()).unwrap();
        let p1 = pool.alloc().unwrap();
        let _anchor = pool.alloc().unwrap(); // keep block committed

        pool.free(p1);

        // Corrupt the free-bin canary at offset 4
// Safety: Test code.
        unsafe {
            *p1.as_ptr().add(4).cast::<()>().cast::<u32>() = 0xDEAD_BEEF;
        }

        // Next alloc from freelist should detect corruption
        drop(pool.alloc()); // should panic
    }

    #[test]
    fn test_free_bin_canary_not_checked_on_bump_path() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Bump path allocations should NOT check free-bin canary
        // (virgin memory has no canary to validate).
        let mut pool = Pool::with_config(16, 65536, &BinnedAllocatorConfig::default()).unwrap();

        // All these come from bump path — no canary check, no panic
        for _ in 0..100 {
            let ptr = pool.alloc().unwrap();
// Safety: Test code.
            unsafe {
                *ptr.as_ptr() = 0xFF;
            }
        }
    }

    #[test]
    fn test_canary_survives_alloc_free_cycle() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Multiple alloc/free cycles on the same bin — canary must be
        // valid on every re-allocation from the freelist.
        let mut pool = Pool::with_config(32, 65536, &BinnedAllocatorConfig::default()).unwrap();
        let _anchor = pool.alloc().unwrap(); // keep block alive

        for _ in 0..50 {
            let ptr = pool.alloc().unwrap();
            // Write user data (overwrites canary area — fine, it's allocated)
// Safety: Test code.
            unsafe {
                ptr.as_ptr().write_bytes(0xAA, 32);
            }
            pool.free(ptr); // rewrites canary at offset 4
            // Next alloc will verify canary
        }
    }

    // --- 5c + 5e: Large alloc and decommit edge cases ---

    #[test]
    fn test_large_alloc_through_global() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        drop(GlobalBinnedAllocator::init());

        let ptr = GlobalBinnedAllocator::alloc_bytes(100_000).unwrap();
// Safety: Test code.
        unsafe {
            ptr.as_ptr().write_bytes(0xDD, 100_000);
        }
// Safety: Test code.
        unsafe { GlobalBinnedAllocator::free_bytes(ptr, 100_000); }
    }

    #[test]
    fn test_large_alloc_various_sizes() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let alloc = BinnedAllocator::new().unwrap();
        let sizes = [65537, 100_000, 256_000, 1_000_000, 4_000_000];

        for &size in &sizes {
            let ptr = alloc.alloc_bytes(size).unwrap();
// Safety: Test code.
            unsafe {
                // Write first and last byte
                *ptr.as_ptr() = 0xAA;
                *ptr.as_ptr().add(size - 1) = 0xBB;
            }
            // Safety: Test code.
// Safety: Test code.
            unsafe { alloc.free_bytes(ptr, size); }
        }
    }

    #[test]
    fn test_large_alloc_best_fit_free_uses_actual_mapping_size() {
        let _guard = crate::memory::TEST_MUTEX.write().unwrap();
        // Regression: best-fit may return a larger cached mapping than requested.
        // free_bytes(requested_size) must still free/decommit the full mapping.
        let baseline_reserved = stats::TOTAL_RESERVED.load(Ordering::Relaxed);

        {
            let alloc = BinnedAllocator::new().unwrap();
            let p1 = alloc.alloc_bytes(200_000).unwrap();
// Safety: Test code.
            unsafe { alloc.free_bytes(p1, 200_000); }

            // Reuses the larger cached mapping from the previous allocation.
            let p2 = alloc.alloc_bytes(100_000).unwrap();
// Safety: Test code.
            unsafe { alloc.free_bytes(p2, 100_000); }
        } // Drop must release all cached/live large mappings.

        let final_reserved = stats::TOTAL_RESERVED.load(Ordering::Relaxed);
        assert_eq!(
            final_reserved, baseline_reserved,
            "TOTAL_RESERVED leaked after best-fit large alloc reuse/free"
        );
    }

    #[test]
    fn test_recommit_cycle() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Repeatedly decommit and recommit the same block.
        // Verifies bump_cursor resets correctly and fresh memory is zeroed.
        let block_size = 65536;
        let bin_size = 4096; // 16 bins per block
        let mut pool =
            Pool::with_config(bin_size, block_size, &BinnedAllocatorConfig::default()).unwrap();

        for cycle in 0..5 {
            // Fill the block
            let mut ptrs = Vec::new();
            if cycle > 0 {
                assert_eq!(
                    pool.blocks.len(),
                    1,
                    "Cycle {cycle}: should reuse block 0",
                );
            }
            for _ in 0..16 {
                let ptr = pool.alloc().unwrap();
                // Safety: Test code.
// Safety: Test code.
                unsafe {
                    ptr.as_ptr().write_bytes(0xCC, bin_size);
                }
                ptrs.push(ptr);
            }
            if cycle > 0 {
                assert_eq!(
                    pool.blocks.len(),
                    1,
                    "Cycle {cycle}: leaked blocks after alloc loop",
                );
            }

            // Free all → queues sparse decommit
            for p in ptrs {
                pool.free(p);
            }

            // Process pending decommits
            pool.process_pending_decommits();

            // Block should be decommitted
            assert!(
                !pool.blocks[0].is_committed(),
                "Cycle {cycle}: block should be decommitted after freeing all bins",
            );
            assert_eq!(pool.committed, 0);
        }
    }

    #[test]
    fn test_sparse_decommit_disabled_by_config() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let config = BinnedAllocatorConfig {
            immediate_decommit: false,
            ..Default::default()
        };
        let mut pool = Pool::with_config(4096, 65536, &config).unwrap();

        let mut ptrs = Vec::new();
        for _ in 0..16 {
            ptrs.push(pool.alloc().unwrap());
        }
        for p in ptrs {
            pool.free(p);
        }

        // With decommit disabled, block stays committed
        assert!(pool.blocks[0].is_committed());
        assert_eq!(pool.committed, 65536);

        // Trim still decommits
        pool.trim();
        assert_eq!(pool.blocks.len(), 0);
        assert_eq!(pool.committed, 0);
    }

    #[test]
    fn test_trim_after_sparse_decommit_multiple_trailing() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Multiple trailing empty blocks, some already sparse-decommitted
        let block_size = 65536;
        let bin_size = 4096; // 16 bins per block
        let mut pool =
            Pool::with_config(bin_size, block_size, &BinnedAllocatorConfig::default()).unwrap();

        // Fill 4 blocks
        let mut ptrs = Vec::new();
        for _ in 0..64 {
            ptrs.push(pool.alloc().unwrap());
        }
        assert_eq!(pool.blocks.len(), 4);

        // Free blocks 2 and 3 (trailing)
        for &ptr in &ptrs[32..64] {
            pool.free(ptr);
        }
        // Process pending decommits
        pool.process_pending_decommits();
        // Both should be sparse-decommitted
        assert!(!pool.blocks[2].is_committed());
        assert!(!pool.blocks[3].is_committed());

        // Trim should pop both
        pool.trim();
        assert_eq!(pool.blocks.len(), 2);

        // Now free blocks 0 and 1
        for &ptr in &ptrs[0..32] {
            pool.free(ptr);
        }
        pool.trim();
        assert_eq!(pool.blocks.len(), 0);
        assert_eq!(pool.committed, 0);
    }

    // --- Concurrent stress with cache + recycler ---

    #[test]
    fn test_producer_consumer_cross_thread() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Classic pathology: Thread A allocates, Thread B frees.
        // Exercises cross-thread recycler path.
        let alloc = Arc::new(BinnedAllocator::new().unwrap());
        let (tx, rx) = std::sync::mpsc::channel::<(usize, usize)>(); // (ptr_addr, size)
        let num_items = 500;

        // Producer thread: allocates items and sends to consumer
        let alloc_p = alloc.clone();
        let producer = thread::spawn(move || {
            let mut cache = ThreadCache::new();
// Safety: Test code.
            unsafe {
                cache.bind(std::mem::transmute::<
                    &BinnedAllocator,
                    &'static BinnedAllocator,
                >(&*alloc_p));
            }
            let sizes = [16, 32, 64, 128, 256, 512];

            for i in 0..num_items {
                let size = sizes[i % sizes.len()];
                let layout = std::alloc::Layout::from_size_align(size, 1).unwrap();
                let ptr = alloc_p.alloc_with_cache(&mut cache, layout).unwrap();
                // Safety: Test code.
// Safety: Test code.
                unsafe {
                    *ptr.as_ptr() = i.to_le_bytes()[0];
                }
                tx.send((ptr.as_ptr() as usize, size)).unwrap();
            }
        });

        // Consumer thread: receives items and frees them
        let alloc_c = alloc.clone();
        let consumer = thread::spawn(move || {
            let mut cache = ThreadCache::new();
// Safety: Test code.
            unsafe {
                cache.bind(std::mem::transmute::<
                    &BinnedAllocator,
                    &'static BinnedAllocator,
                >(&*alloc_c));
            }

            for (ptr_addr, size) in rx {
                let ptr = NonNull::new(ptr_addr as *mut u8).unwrap();
                let layout = std::alloc::Layout::from_size_align(size, 1).unwrap();
                alloc_c.free_with_cache(&mut cache, ptr, layout);
            }
        });

        producer.join().unwrap();
        consumer.join().unwrap();
    }

    #[test]
    fn test_mixed_small_large_concurrent() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Multiple threads doing mixed small + large allocations concurrently
        let alloc = Arc::new(BinnedAllocator::new().unwrap());
        let num_threads = 8u8;
        let barrier = Arc::new(crate::sync::barrier::Barrier::new(num_threads as usize));
        let mut handles = vec![];

        for t in 0..num_threads {
            let alloc = alloc.clone();
            let b = barrier.clone();
            handles.push(thread::spawn(move || {
                b.wait();
                let mut ptrs = Vec::new();
                // Alternate between small and large
                let sizes = [16, 100_000, 64, 200_000, 256, 500_000, 1024, 1_000_000];

                for i in 0u8..40 {
                    let size = sizes[usize::from(i) % sizes.len()];
                    let ptr = alloc.alloc_bytes(size).unwrap();
                    // Safety: Test code.
// Safety: Test code.
                    unsafe {
                        *ptr.as_ptr() = t.wrapping_mul(100).wrapping_add(i);
                    }
                    ptrs.push((ptr, size));
                }

                // Verify
                for (i, (ptr, _)) in ptrs.iter().enumerate() {
// Safety: Test code.
                    unsafe {
                        assert_eq!(ptr.as_ptr().read(), t.wrapping_mul(100).wrapping_add(i.to_le_bytes()[0]));
                    }
                }

                // Free in reverse
                for (ptr, size) in ptrs.into_iter().rev() {
                    // Safety: Test code.
// Safety: Test code.
            unsafe { alloc.free_bytes(ptr, size); }
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_cache_recycler_contention() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // High-contention scenario: many threads rapidly alloc/free the same
        // size class through caches, triggering frequent recycler push/pop.
        let alloc = Arc::new(BinnedAllocator::new().unwrap());
        let num_threads = 16;
        let rounds = 5;
        let batch = 80; // > cache limit (64) to trigger flush each round
        let barrier = Arc::new(crate::sync::barrier::Barrier::new(num_threads));
        let mut handles = vec![];

        for _ in 0..num_threads {
            let alloc = alloc.clone();
            let b = barrier.clone();
            handles.push(thread::spawn(move || {
                let mut cache = ThreadCache::new();
// Safety: Test code.
                unsafe {
                    cache.bind(std::mem::transmute::<
                        &BinnedAllocator,
                        &'static BinnedAllocator,
                    >(&*alloc));
                }
                let layout = std::alloc::Layout::from_size_align(32, 1).unwrap();

                b.wait();

                for _ in 0..rounds {
                    // Alloc batch items
                    let mut ptrs = Vec::with_capacity(batch);
                    for _ in 0..batch {
                        ptrs.push(alloc.alloc_with_cache(&mut cache, layout).unwrap());
                    }
                    // Free them all → triggers cache flush → recycler push/pop
                    for p in ptrs {
                        alloc.free_with_cache(&mut cache, p, layout);
                    }
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_alloc_free_interleaved_stress() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Interleaved alloc/free pattern — allocate a few, free a few,
        // repeat. Tests freelist correctness under churn.
        let mut pool = Pool::with_config(64, 65536, &BinnedAllocatorConfig::default()).unwrap();
        //let _bins_per_block = 65536 / 64; // 1024
        let mut live: Vec<NonNull<u8>> = Vec::new();

        for round in 0u8..20 {
            // Allocate 100
            for i in 0u8..100 {
                let ptr = pool.alloc().unwrap();
                // Safety: Test code.
// Safety: Test code.
                unsafe {
                    *ptr.as_ptr() = round.wrapping_mul(100).wrapping_add(i);
                }
                live.push(ptr);
            }

            // Free every other one
            let mut kept = Vec::new();
            for (i, ptr) in live.drain(..).enumerate() {
                if i % 2 == 0 {
                    pool.free(ptr);
                } else {
                    kept.push(ptr);
                }
            }
            live = kept;
        }

        // Verify remaining live pointers are valid
        for ptr in &live {
// Safety: Test code.
            unsafe {
                let _ = ptr.as_ptr().read();
            }
        }

        // Free all remaining
        for ptr in live {
            pool.free(ptr);
        }

        // Pool should have lots of free blocks now
        let total_free: u32 = pool.blocks.iter().map(|b| u32::from(b.free_count())).sum();
        assert!(total_free > 0);
    }

    #[test]
    fn test_bit_tree_full_sweep() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Mark all 16384 blocks free, then mark all full, verify find_free at each step
        let mut tree = BitTree::new();

        // Mark every 64th block free (one per L2 word)
        for i in (0..16384).step_by(64) {
            tree.mark_free(i);
        }
        assert_eq!(tree.find_free(), Some(0));

        // Mark them full in reverse
        for i in (0..16384).step_by(64).rev() {
            tree.mark_full(i);
        }
        assert_eq!(tree.find_free(), None);
    }

    #[test]
    fn test_receive_bundle_walk_accuracy() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Verify receive_bundle_walk correctly counts and links the bundle.
        let alloc = BinnedAllocator::new().unwrap();

        // Build a chain of 5 items
        let mut ptrs = Vec::new();
        for _ in 0..5 {
            ptrs.push(alloc.alloc_bytes(32).unwrap());
        }

        // Link them as a null-terminated chain via first usize
        for i in 0..4 {
            // Safety: Test code.
// Safety: Test code.
            unsafe {
                *ptrs[i].cast::<usize>().as_ptr() = ptrs[i + 1].as_ptr() as usize;
            }
        }
        // Last item is null-terminated
        // Safety: Test code.
// Safety: Test code.
        unsafe {
            *ptrs[4].cast::<usize>().as_ptr() = 0;
        }

        // Feed the chain to a LocalFreeList via receive_bundle_walk
        let mut list = LocalFreeList::new();
        list.receive_bundle_walk(ptrs[0]);
        assert_eq!(list.count, 5);

        // Pop should return items in order: ptrs[0], ptrs[1], ... ptrs[4]
        for (i, &ptr) in ptrs.iter().take(5).enumerate(){
            let popped = list.pop().unwrap();
            assert_eq!(popped, ptr, "Item {i} mismatch");
        }
        assert!(list.pop().is_none());
        assert_eq!(list.count, 0);

        // Free all
        for p in ptrs {
            // Safety: Test code.
// Safety: Test code.
            unsafe { alloc.free_bytes(p, 32); }
        }
    }

    #[test]
    fn test_receive_bundle_walk_prepends_to_existing() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Verify receive_bundle_walk correctly prepends to a non-empty list.
        let alloc = BinnedAllocator::new().unwrap();

        let existing = alloc.alloc_bytes(32).unwrap();
        let bundle_a = alloc.alloc_bytes(32).unwrap();
        let bundle_b = alloc.alloc_bytes(32).unwrap();

        // Build 2-item bundle: A → B → null
        // Safety: Test code.
// Safety: Test code.
        unsafe {
            *bundle_a.cast::<usize>().as_ptr() = bundle_b.as_ptr() as usize;
            *bundle_b.cast::<usize>().as_ptr() = 0;
        }

        let mut list = LocalFreeList::new();
        list.push(existing); // list: [existing]
        assert_eq!(list.count, 1);

        list.receive_bundle_walk(bundle_a); // list: [A, B, existing]
        assert_eq!(list.count, 3);

        assert_eq!(list.pop().unwrap(), bundle_a);
        assert_eq!(list.pop().unwrap(), bundle_b);
        assert_eq!(list.pop().unwrap(), existing);
        assert!(list.pop().is_none());

        // Safety: Test code.
// Safety: Test code.
        unsafe { alloc.free_bytes(existing, 32); }
        // Safety: Test code.
// Safety: Test code.
        unsafe { alloc.free_bytes(bundle_a, 32); }
        // Safety: Test code.
// Safety: Test code.
        unsafe { alloc.free_bytes(bundle_b, 32); }
    }

    #[test]
    fn test_config_affects_batch_size() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Verify custom alloc_extra is respected during batch refill
        let config = BinnedAllocatorConfig {
            alloc_extra: [4, 2, 1, 1],
            ..Default::default()
        };
        let alloc = BinnedAllocator::with_config(config).unwrap();
        let mut cache = ThreadCache::new();
// Safety: Test code.
        unsafe {
            cache.bind(std::mem::transmute::<&binned::BinnedAllocator, &binned::BinnedAllocator>(&alloc));
        }

        let layout = std::alloc::Layout::from_size_align(32, 1).unwrap();
        let _ = alloc.alloc_with_cache(&mut cache, layout).unwrap();
        let idx = BinnedAllocator::size_class(32, 1);
        // With alloc_extra[0]=4 for <=1KB: batch refill = 4, pop 1 = 3 remaining
        assert_eq!(cache.bins[idx].count, 3);
    }

    #[test]
    fn test_config_affects_cache_limit() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Verify custom cache_count_limits trigger flush at the right threshold
        let config = BinnedAllocatorConfig {
            cache_count_limits: [8, 4, 2, 1],
            ..Default::default()
        };
        let alloc = BinnedAllocator::with_config(config).unwrap();
        let mut cache = ThreadCache::new();
// Safety: Test code.
        unsafe {
            cache.bind(std::mem::transmute::<&binned::BinnedAllocator, &binned::BinnedAllocator>(&alloc));
        }

        let size = 32;
        let layout = std::alloc::Layout::from_size_align(size, 1).unwrap();
        let idx = BinnedAllocator::size_class(size, 1);

        // Free 8 items to cache — should NOT trigger flush (limit is 8, check is >)
        let mut ptrs = Vec::new();
        for _ in 0..8 {
            ptrs.push(alloc.alloc_bytes(size).unwrap());
        }
        for p in ptrs {
            alloc.free_with_cache(&mut cache, p, layout);
        }
        assert!(cache.bins[idx].count <= 8);

        // One more should trigger flush
        let extra = alloc.alloc_bytes(size).unwrap();
        alloc.free_with_cache(&mut cache, extra, layout);
        // After flush, cache is empty (entire bundle detached)
        assert_eq!(cache.bins[idx].count, 0);
    }

    #[test]
    fn test_global_trim_flushes_all_caches() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        drop(GlobalBinnedAllocator::init());
        let barrier = Arc::new(crate::sync::barrier::Barrier::new(4));
        let mut handles = vec![];

        // Spawn 4 threads that alloc+free, leaving items in their caches
        for _ in 0..4 {
            let b = barrier.clone();
            handles.push(thread::spawn(move || {
                let mut ptrs = Vec::new();
                for _ in 0..20 {
                    ptrs.push(GlobalBinnedAllocator::alloc_bytes(256).unwrap());
                }
                for p in ptrs {
                    // Safety: Test code.
// Safety: Test code.
                    unsafe { GlobalBinnedAllocator::free_bytes(p, 256); }
                }
                b.wait(); // sync before trim
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // Trim should flush all caches and trim pools
        GlobalBinnedAllocator::trim();
    }

    #[test]
    fn test_block_meta_edge_values() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Test BlockMeta with extreme values for all fields
        let mut meta = BlockMeta::new(0xFFFF, 0xFFFF_usize, true);
        assert_eq!(meta.free_count(), 0xFFFF);
        meta.check_canary();

        meta.set_free_count(0);
        assert_eq!(meta.free_count(), 0);

        meta.set_free_head(0);
        assert_eq!(meta.free_head(), 0);
        meta.set_free_head(BIN_SENTINEL);
        assert_eq!(meta.free_head(), BIN_SENTINEL);

        meta.set_bump_cursor(0xFFFE);
        assert_eq!(meta.bump_cursor(), 0xFFFE);

        // All other fields should be unaffected
        assert_eq!(meta.free_count(), 0);
        assert_eq!(meta.free_head(), BIN_SENTINEL);
        assert!(meta.is_committed());
        meta.check_canary();
    }

    #[test]
    fn test_pool_alloc_max_blocks_single_bin() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // 65536-byte bin, 1 bin per block. Allocate several blocks.
        let mut pool = Pool::with_config(65536, 65536, &BinnedAllocatorConfig::default()).unwrap();

        let mut ptrs = Vec::new();
        for _ in 0..10 {
            let ptr = pool.alloc().unwrap();
            ptrs.push(ptr);
        }
        assert_eq!(pool.blocks.len(), 10);

        // Free all — each queues individual block decommit
        for p in ptrs {
            pool.free(p);
        }

        // Process deferred decommits
        pool.process_pending_decommits();

        for block in &pool.blocks {
            assert!(!block.is_committed());
        }
    }

    #[test]
    fn test_high_thread_count_with_cache() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // 128 threads using cache + recycler path
        let alloc = Arc::new(BinnedAllocator::new().unwrap());
        let num_threads = 128;
        let barrier = Arc::new(crate::sync::barrier::Barrier::new(num_threads));
        let mut handles = vec![];

        for _t in 0..num_threads {
            let alloc = alloc.clone();
            let b = barrier.clone();
            handles.push(thread::spawn(move || {
                let mut cache = ThreadCache::new();
// Safety: Test code.
                unsafe {
                    cache.bind(std::mem::transmute::<
                        &BinnedAllocator,
                        &'static BinnedAllocator,
                    >(&*alloc));
                }
                let layout = std::alloc::Layout::from_size_align(64, 1).unwrap();

                b.wait();

                let mut ptrs = Vec::new();
                for _ in 0..50 {
                    ptrs.push(alloc.alloc_with_cache(&mut cache, layout).unwrap());
                }
                for p in ptrs {
                    alloc.free_with_cache(&mut cache, p, layout);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_global_alloc_trait_impl() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Just ensure GlobalBinnedAllocator implements GlobalAlloc
        // and we can call alloc/dealloc on it.
        let allocator = GlobalBinnedAllocator;
        let layout = std::alloc::Layout::from_size_align(64, 16).unwrap();

        // Safety: Test code.
// Safety: Test code.
        unsafe {
            // Initialize if not already
            drop(GlobalBinnedAllocator::init());

            let ptr = std::alloc::GlobalAlloc::alloc(&allocator, layout);
            assert!(!ptr.is_null());

            // Check write/read
            std::ptr::write_volatile(ptr, 0xCC);
            assert_eq!(std::ptr::read_volatile(ptr), 0xCC);

            std::alloc::GlobalAlloc::dealloc(&allocator, ptr, layout);
        }
    }

    #[test]
    fn test_small_size_large_alignment() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let alloc = BinnedAllocator::new().unwrap();
        let size = 16;
        let align = 32 * 1024; 
        
        let page_size = PlatformVmOps::page_size();
        if align <= page_size {
             return; 
        }

        let layout = std::alloc::Layout::from_size_align(size, align).unwrap();
        
        let ptr = alloc.alloc(layout).unwrap();
        assert_eq!(ptr.as_ptr() as usize % align, 0, "Pointer should be aligned to {align}");
        
        // Safety: Test code.
// Safety: Test code.
        unsafe {
            *ptr.as_ptr() = 0xAA;
        }
        
        // Safety: Test code.
// Safety: Test code.
        unsafe {
            alloc.free(ptr, layout);
        }
    }
}
