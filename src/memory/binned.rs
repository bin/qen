#[cfg(all(test, not(loom)))]
use super::binned;
use super::stats;
use super::vm::{self, PlatformVmOps, VmError, VmOps};
use crate::sync::atomic::{AtomicU64, Ordering};
#[cfg(any(debug_assertions, feature = "hardened"))]
use fixedbitset::FixedBitSet;
use std::ptr::NonNull;

pub(crate) const POOL_RESERVED_SIZE: usize = 256 * 1024 * 1024;

/// Largest size served by the binned size classes; larger allocations go
/// through the large-allocation cache. Exposed so embedders (e.g. the
/// benchmark adapter) can route bookkeeping by the same threshold.
/// 256 KiB matches internal tcmalloc's `kMaxSize`: mid-size allocations
/// are common enough in real workloads that pushing them through the
/// page-granular large path costs syscalls and cache locality.
pub const MAX_SMALL_SIZE: usize = 262_144;

/// Configuration for `BinnedAllocator`. All fields have sensible defaults.
/// Set at init time via `BinnedAllocator::with_config()`.
#[derive(Clone, Debug)]
pub struct BinnedAllocatorConfig {
    /// INITIAL per-class TLS cache length limit, per bin-size tier:
    /// `[<=1KB, <=8KB, <=32KB, >32KB]`. Default: `[64, 32, 8, 4]`.
    /// Values below the L0 micro-cache capacity (8) are floored to it.
    ///
    /// Caches are adaptive (tcmalloc-style slow start): each class's limit
    /// grows by one transfer batch per refill, and on free-side overflow by
    /// one until it covers a batch and then by whole batches, up to the
    /// per-class ceiling (`CLASS_CAP`, ~256KiB of cached bytes per class,
    /// at most 8192 objects). This field only sets the starting point.
    pub cache_count_limits: [u32; 4],

    /// Per-thread cache byte budget across all classes. When the cached
    /// total exceeds this, the cache scavenges each class down toward its
    /// recent low-water mark and shrinks over-provisioned class limits
    /// (tcmalloc's Scavenge policy), force-releasing from the largest
    /// classes down to half the budget if the low-water pass alone is not
    /// enough. Default: 16 MiB (measured knee for mixed-size churn; below
    /// ~8 MiB the scavenger fights workloads whose per-thread working set
    /// legitimately spans many classes).
    pub max_thread_cache_bytes: usize,

    /// Pool reserved VA size per size class. Default: 256 MB.
    pub pool_reserved_size: usize,

    /// Block size in bytes. Default: `max(256KB, system page size)`
    /// (the largest size class must fit one bin per block).
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

    /// Byte limit for the large-allocation bucket cache: decommitted
    /// reservations above `MAX_SMALL_SIZE` held for reuse instead of being
    /// released to the OS. Default: 64 MiB.
    pub large_cache_bytes: usize,

    /// Number of trim passes a fully-empty block must survive before
    /// decommit. Prevents syscall thrashing on bursty workloads.
    /// Default: 3. Set to 0 for immediate decommit after first trim.
    pub decommit_cooldown: u8,

    /// EXPERIMENTAL: route remote frees through the per-block return-mask
    /// channel (`remote_mask`) instead of the recycler. Default: false —
    /// measured on shuffled cross-thread workloads it is throughput-flat
    /// and strands capacity behind reconcile latency (peak-RSS regression);
    /// its win case is stable producer→consumer flows, pending the block-
    /// adoption phase and pipeline-shaped benchmarks (larson).
    pub remote_mask_channel: bool,
}

impl Default for BinnedAllocatorConfig {
    fn default() -> Self {
        Self {
            cache_count_limits: [64, 32, 8, 4],
            max_thread_cache_bytes: 16 * 1024 * 1024,
            pool_reserved_size: POOL_RESERVED_SIZE,
            block_size: 0, // 0 = auto-detect (max(MAX_SMALL_SIZE, page_size))
            immediate_decommit: true,
            recycler_max_bundles: 16,
            use_huge_pages: true,
            large_cache_bytes: super::large_cache::LARGE_CACHE_DEFAULT_LIMIT,
            decommit_cooldown: DECOMMIT_COOLDOWN,
            remote_mask_channel: false,
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

    /// INITIAL per-class TLS cache length limit for the given bin size
    /// (adaptive growth raises it; see `cache_count_limits`).
    #[must_use]
    pub fn max_cache_for(&self, bin_size: usize) -> u32 {
        self.cache_count_limits[Self::tier(bin_size)]
    }
}

// Each BitTree segment covers 16384 blocks. BitTreeChain chains multiple
// segments for pools that exceed the single-segment capacity.

const BIN_SENTINEL: u16 = 0xFFFF;

/// Default number of trim passes a fully-empty block must survive before decommit.
const DECOMMIT_COOLDOWN: u8 = 3;

/// 8-bit canary value written into every `BlockMeta` on creation.
/// Checked on alloc and free to detect corruption.
/// Only active when debug assertions or hardened mode are enabled.
#[cfg(any(debug_assertions, feature = "hardened"))]
const BLOCK_CANARY: u8 = 0xA5;

/// 32-bit canary written at offset 4 of every freed bin (freelist path only).
/// Checked on alloc to detect use-after-free and double-alloc corruption.
/// Only active when debug assertions or hardened mode are enabled.
#[cfg(any(debug_assertions, feature = "hardened"))]
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
        crate::qen_debug_assert!(
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

    /// Non-mutating probe: does any block have free space?
    ///
    /// Segments below `search_cursor` are known-full (the cursor only skips
    /// a segment once it is full, and `mark_free` pulls it back), so scanning
    /// from the cursor is exact. Cold path — does not advance the cursor.
    pub fn has_free(&self) -> bool {
        let start = self.search_cursor.min(self.trees.len());
        self.trees[start..].iter().any(|t| t.find_free().is_some())
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

/// Per-block metadata packed into 8 bytes (+ debug/hardened-only `FixedBitSet`).
///
/// Bit layout of `packed: u64`:
/// ```text
///   [63..56] canary       (8 bits)  — always BLOCK_CANARY (0xA5)
///   [55..50] reserved     (6 bits)
///   [49]     decommitting (1 bit)   — 1 while a trim is decommitting this
///                                     block outside the pool lock; the block
///                                     is hidden from the bit tree and must
///                                     not be selected by another trim pass
///   [48]     committed    (1 bit)   — 1 if block is backed by physical pages
///   [47..32] bump_cursor  (16 bits) — next virgin slot for bump allocation
///   [31..16] free_head    (16 bits) — index of first free bin, or 0xFFFF sentinel
///   [15..0]  free_count   (16 bits) — number of free bins in this block
/// ```
///
/// 16-bit limits are sufficient: max `bins_per_block` = 65536/16 = 4096 for the
/// smallest bin size, well within u16 range. `BIN_SENTINEL` is 0xFFFF.
pub(crate) struct BlockMeta {
    packed: u64,
    #[cfg(any(debug_assertions, feature = "hardened"))]
    pub free_map: FixedBitSet,
}

impl BlockMeta {
    /// Create new `BlockMeta` for a committed block with all bins free
    /// (bump allocation starts at slot 0).
    pub fn new(free_count: u16, #[allow(unused)] bins_per_block: usize) -> Self {
        let mut packed: u64 = 0;
        // canary (debug and hardened builds only)
        #[cfg(any(debug_assertions, feature = "hardened"))]
        {
            packed |= u64::from(BLOCK_CANARY) << 56;
        }
        // committed = 1
        packed |= 1u64 << 48;
        // bump_cursor = 0 (implicit)
        // free_head = SENTINEL
        packed |= u64::from(BIN_SENTINEL) << 16;
        // free_count
        packed |= u64::from(free_count);

        Self {
            packed,
            #[cfg(any(debug_assertions, feature = "hardened"))]
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

    /// True while a trim pass has detached this block for an out-of-lock
    /// decommit syscall (see `Pool::begin_trim`/`finish_trim`).
    #[inline]
    pub fn is_decommitting(&self) -> bool {
        (self.packed >> 49) & 1 == 1
    }

    #[inline]
    pub fn set_decommitting(&mut self, v: bool) {
        if v {
            self.packed |= 1u64 << 49;
        } else {
            self.packed &= !(1u64 << 49);
        }
    }

    /// Verify canary byte. Panics on corruption.
    /// No-op when debug assertions and hardened mode are disabled.
    #[inline]
    pub fn check_canary(&self) {
        #[cfg(any(debug_assertions, feature = "hardened"))]
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

/// One contiguous run of fully-empty blocks selected for decommit.
///
/// Produced under the pool lock by `begin_trim()` (which marks the blocks
/// `decommitting` and hides them from the bit tree), executed *outside* the
/// lock (`ok` records the syscall result), and integrated back under the
/// lock by `finish_trim()`. This mirrors the commit path's
/// `probe_commit_needed`/`integrate_precommit` protocol so that trim never
/// issues syscalls while holding the pool mutex.
pub(crate) struct TrimDecommit {
    /// Index of the owning pool within its `PoolChain` (0 for direct
    /// `Pool`-level use).
    pool_idx: usize,
    /// First block index of the run.
    run_start: usize,
    /// Number of contiguous blocks in the run.
    run_len: usize,
    /// Base address of the run.
    ptr: NonNull<u8>,
    /// Total bytes to decommit.
    size: usize,
    /// Result of the decommit syscall, filled in by the caller.
    pub ok: bool,
}

pub(crate) struct Pool {
    pub bin_size: usize,
    pub block_size: usize,
    pub bins_per_block: u16,
    /// Precomputed reciprocal for O(1) division by `bin_size` on the free path.
    bin_reciprocal: ReciprocalDiv,
    /// Precomputed reciprocal for O(1) division by `block_size` on the free path.
    block_reciprocal: ReciprocalDiv,
    /// Aligned base pointer (aligned to `reserved_size` for O(1) pool lookup).
    pub base: NonNull<u8>,
    pub committed: usize,
    pub reserved_size: usize,
    /// Original mmap base (may differ from `base` due to over-reservation alignment).
    original_base: NonNull<u8>,
    /// Total reserved size including alignment slop.
    total_reserved: usize,
    pub immediate_decommit: bool,
    pub bit_tree: BitTreeChain,
    pub blocks: Vec<BlockMeta>,
    /// Block indices pending decommit with cooldown counter. Populated by
    /// `free()` when block becomes fully empty. Each trim pass decrements the
    /// counter; the block is only decommitted when it reaches 0.
    decommit_pending: Vec<(usize, u8)>,
    /// Cooldown value for new decommit entries (from config).
    decommit_cooldown: u8,
}

// Safety: Pool owns the memory region and is safe to send between threads.
unsafe impl Send for Pool {}

impl Drop for Pool {
    fn drop(&mut self) {
        // Release the entire VM reservation using the original (pre-alignment) base
        // Safety: We are dropping the pool, so we can release the memory.
        unsafe {
            drop(PlatformVmOps::release(
                self.original_base,
                self.total_reserved,
            ));
            stats::TOTAL_RESERVED.sub(self.total_reserved);
            stats::TOTAL_COMMITTED.sub(self.committed);
            stats::BINNED_ALLOCATOR_COMMITTED.sub(self.committed);
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
        if bins_per_block == 0 {
            // Defense in depth: BinnedAllocator::with_config validates
            // block_size >= MAX_SMALL_SIZE, but a zero-bin pool must never be
            // constructible — free_count arithmetic would wrap and alloc would
            // return out-of-bounds pointers.
            return Err(VmError::InitializationFailed(format!(
                "block_size ({block_size}) is smaller than bin_size ({bin_size}): zero bins per block",
            )));
        }
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
        crate::qen_debug_assert!(
            bin_size >= 16,
            "bin_size {bin_size} is smaller than minimum required 16 (for recycler links)",
        );
        crate::qen_debug_assert!(
            bin_size >= std::mem::size_of::<usize>(),
            "bin_size {bin_size} is smaller than minimum required {}",
            std::mem::size_of::<usize>()
        );

        // Reserve aligned address space for O(1) pool lookup on free path.
        assert!(
            reserved_size.is_power_of_two() && reserved_size > 0,
            "reserved_size ({reserved_size}) must be a non-zero power of two for mask-based lookup"
        );
        // Safety: FFI call to reserve memory.
        let reservation =
            unsafe { vm::reserve_aligned::<PlatformVmOps>(reserved_size, reserved_size)? };

        crate::qen_debug_assert!(
            (reservation.aligned.as_ptr() as usize).is_multiple_of(reserved_size),
            "reserve_aligned returned unaligned base: {:p} (expected {reserved_size}-aligned)",
            reservation.aligned.as_ptr()
        );

        stats::TOTAL_RESERVED.fetch_add(reservation.total_reserved, Ordering::Relaxed);

        Ok(Self {
            bin_size,
            block_size,
            // Cold constructor: the range was checked above, so this never
            // fails; try_from documents the invariant at zero hot-path cost.
            bins_per_block: u16::try_from(bins_per_block)
                .expect("bins_per_block validated against u16::MAX above"),
            bin_reciprocal: compute_reciprocal(bin_size),
            block_reciprocal: compute_reciprocal(block_size),
            base: reservation.aligned,
            committed: 0,
            reserved_size,
            original_base: reservation.original_base,
            total_reserved: reservation.total_reserved,
            immediate_decommit: config.immediate_decommit,
            bit_tree: BitTreeChain::new(),
            blocks: Vec::new(),
            decommit_pending: Vec::new(),
            decommit_cooldown: config.decommit_cooldown,
        })
    }

    /// Cheap capacity probe used by `PoolChain` to decide whether a retired
    /// pool can serve allocations again: true if a free bin exists, a
    /// decommitted block can be recommitted, or a new block still fits in
    /// the reservation. Never syscalls. Mirrors the exhaustion check in
    /// [`Pool::alloc`] exactly.
    pub fn has_free_capacity(&self) -> bool {
        self.bit_tree.has_free() || self.committed + self.block_size <= self.reserved_size
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
                #[cfg(any(debug_assertions, feature = "hardened"))]
                unsafe {
                    std::ptr::write_bytes(req.ptr.as_ptr(), 0, req.size);
                }

                self.blocks.push(BlockMeta::new(
                    self.bins_per_block,
                    self.bins_per_block as usize,
                ));
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
                #[cfg(any(debug_assertions, feature = "hardened"))]
                unsafe {
                    std::ptr::write_bytes(req.ptr.as_ptr(), 0, req.size);
                }

                let block = &mut self.blocks[req.block_idx];
                *block = BlockMeta::new(self.bins_per_block, self.bins_per_block as usize);
                self.committed += self.block_size;
                stats::TOTAL_COMMITTED.fetch_add(self.block_size, Ordering::Relaxed);
                stats::BINNED_ALLOCATOR_COMMITTED.fetch_add(self.block_size, Ordering::Relaxed);
                return true;
            }
            false
        }
    }

    /// Select pending decommits whose cooldown has expired. Blocks must
    /// survive `decommit_cooldown` trim passes before being decommitted.
    /// Each pass decrements the counter; blocks at 0 are eligible.
    /// Re-allocated blocks are silently removed. Contiguous eligible blocks
    /// are merged into single decommit requests.
    ///
    /// Selected blocks are marked `decommitting` and hidden from the bit
    /// tree, so the caller may drop the pool lock, perform the syscalls, and
    /// integrate the results via [`finish_trim`](Self::finish_trim) — no
    /// other thread (allocation or a concurrent trim) can touch them in
    /// between.
    fn begin_pending_decommits(&mut self) -> Vec<TrimDecommit> {
        if self.decommit_pending.is_empty() {
            return Vec::new();
        }
        let mut pending = std::mem::take(&mut self.decommit_pending);
        let bins_per_block = self.bins_per_block;

        // Remove blocks that are no longer fully empty or committed, or are
        // already detached by a concurrent trim pass.
        pending.retain(|&(idx, _)| {
            idx < self.blocks.len()
                && self.blocks[idx].free_count() == bins_per_block
                && self.blocks[idx].is_committed()
                && !self.blocks[idx].is_decommitting()
        });

        // Partition: ready (cooldown == 0) vs still cooling
        let mut ready = Vec::new();
        let mut still_cooling = Vec::new();

        for (idx, cooldown) in pending {
            if cooldown == 0 {
                ready.push(idx);
            } else {
                still_cooling.push((idx, cooldown - 1));
            }
        }

        // Re-queue blocks still cooling down
        self.decommit_pending = still_cooling;

        if ready.is_empty() {
            return Vec::new();
        }

        ready.sort_unstable();
        ready.dedup();

        // Detach the selected blocks: unallocatable while the syscall runs
        // without the lock. They hold no live bins (fully empty), so no
        // free() can touch them either.
        for &idx in &ready {
            self.blocks[idx].set_decommitting(true);
            self.bit_tree.mark_full(idx);
        }

        // Merge contiguous runs into single decommit requests.
        let mut requests = Vec::new();
        let mut run_start = ready[0];
        let mut run_len: usize = 1;

        for i in 1..=ready.len() {
            let contiguous = i < ready.len() && ready[i] == run_start + run_len;

            if contiguous {
                run_len += 1;
            } else {
                // Safety: run blocks lie within the reserved range.
                let ptr = unsafe {
                    NonNull::new_unchecked(self.base.as_ptr().add(run_start * self.block_size))
                };
                requests.push(TrimDecommit {
                    pool_idx: 0,
                    run_start,
                    run_len,
                    ptr,
                    size: run_len * self.block_size,
                    ok: false,
                });

                if i < ready.len() {
                    run_start = ready[i];
                    run_len = 1;
                }
            }
        }
        requests
    }

    /// With `immediate_decommit` disabled, `free()` never queues empty
    /// blocks, so trim itself queues the trailing empty run (the only
    /// candidates in that mode) — subject to the same cooldown.
    fn queue_trailing_candidates(&mut self) {
        if self.immediate_decommit {
            return;
        }
        let bins_per_block = self.bins_per_block;
        for idx in (0..self.blocks.len()).rev() {
            let block = &self.blocks[idx];
            if block.free_count() != bins_per_block {
                break;
            }
            if block.is_committed()
                && !block.is_decommitting()
                && !self.decommit_pending.iter().any(|&(p, _)| p == idx)
            {
                self.decommit_pending.push((idx, self.decommit_cooldown));
            }
        }
    }

    /// Phase 1 of trim: select and detach cooled-down empty blocks.
    /// Returns decommit requests to execute *without* holding the pool lock.
    pub fn begin_trim(&mut self) -> Vec<TrimDecommit> {
        self.queue_trailing_candidates();
        self.begin_pending_decommits()
    }

    /// Integrate one executed decommit request (phase 3, under the lock).
    fn finish_trim_one(&mut self, req: &TrimDecommit) {
        for idx in req.run_start..req.run_start + req.run_len {
            let block = &mut self.blocks[idx];
            crate::qen_debug_assert!(
                block.is_decommitting(),
                "finish_trim on undetached block {idx}"
            );
            block.set_decommitting(false);
            if req.ok {
                block.set_committed(false);
            }
            // Restore visibility: the block is empty and (re)allocatable.
            self.bit_tree.mark_free(idx);
        }
        if req.ok {
            self.committed -= req.size;
            stats::TOTAL_COMMITTED.sub(req.size);
            stats::BINNED_ALLOCATOR_COMMITTED.sub(req.size);
        } else {
            // Best-effort with retry: failed blocks return to the queue and
            // are retried on the next trim pass.
            for idx in req.run_start..req.run_start + req.run_len {
                self.decommit_pending.push((idx, 0));
            }
        }
    }

    /// Phase 3 of trim: integrate executed decommits, then shrink the block
    /// vector by popping trailing decommitted blocks (metadata only — no
    /// syscalls).
    // Exclusive-&mut convenience path: exercised by (non-loom) tests;
    // production goes through PoolChain::begin_trim/finish_trim.
    #[allow(dead_code)]
    pub fn finish_trim(&mut self, batch: &[TrimDecommit]) {
        for req in batch {
            self.finish_trim_one(req);
        }
        self.pop_trailing_decommitted();
    }

    /// Pop trailing fully-empty *decommitted* blocks from the metadata
    /// vector. Committed empty blocks are left in place until their
    /// decommit cooldown expires — popping them here would bypass the
    /// cooldown and reintroduce syscall thrashing on bursty workloads.
    pub fn pop_trailing_decommitted(&mut self) {
        let bins_per_block = self.bins_per_block;
        while let Some(last_idx) = self.blocks.len().checked_sub(1) {
            let block = &self.blocks[last_idx];
            if block.free_count() == bins_per_block
                && !block.is_committed()
                && !block.is_decommitting()
            {
                self.blocks.pop();
                self.bit_tree.mark_full(last_idx);
            } else {
                break;
            }
        }
    }

    /// Process pending decommit requests with cooldown, executing the
    /// syscalls inline. Convenience wrapper over
    /// `begin_pending_decommits`/`finish_trim_one` for exclusive-`&mut`
    /// callers (tests, single-threaded use); the lock-aware three-phase
    /// protocol in `BinnedAllocator::trim` is preferred under contention.
    #[allow(dead_code)]
    pub fn process_pending_decommits(&mut self) {
        let mut batch = self.begin_pending_decommits();
        for req in &mut batch {
            // Safety: FFI call to decommit memory owned by this pool.
            req.ok = unsafe { PlatformVmOps::decommit(req.ptr, req.size) }.is_ok();
        }
        for req in &batch {
            self.finish_trim_one(req);
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
                let ptr = unsafe { NonNull::new_unchecked(self.base.as_ptr().add(block_offset)) };
                // Safety: FFI call to commit memory.
                unsafe { PlatformVmOps::commit(ptr, self.block_size)? };
                // Debug mode: zero the block under the pool lock to guarantee
                // deterministic behavior. This is safe because we hold the
                // lock and the block has no live allocations (was fully empty).
                // Safety: ptr is valid.
                #[cfg(any(debug_assertions, feature = "hardened"))]
                unsafe {
                    std::ptr::write_bytes(ptr.as_ptr(), 0, self.block_size);
                }
                self.committed += self.block_size;
                stats::TOTAL_COMMITTED.fetch_add(self.block_size, Ordering::Relaxed);
                stats::BINNED_ALLOCATOR_COMMITTED.fetch_add(self.block_size, Ordering::Relaxed);

                let block = &mut self.blocks[idx];
                // Re-initialize: block with bump allocation
                *block = BlockMeta::new(self.bins_per_block, self.bins_per_block as usize);
            }
            idx
        } else {
            // No free blocks, commit new one
            if self.committed + self.block_size > self.reserved_size {
                return Err(VmError::PoolExhausted);
            }

            let block_idx = self.blocks.len();
            let block_offset = block_idx * self.block_size;
            // Safety: block_offset is within reserved range.
            let ptr = unsafe { NonNull::new_unchecked(self.base.as_ptr().add(block_offset)) };

            // Safety: FFI call to commit memory.
            unsafe { PlatformVmOps::commit(ptr, self.block_size)? };
            // Debug mode: zero the block under the pool lock.
            // Safety: ptr is valid.
            #[cfg(any(debug_assertions, feature = "hardened"))]
            unsafe {
                std::ptr::write_bytes(ptr.as_ptr(), 0, self.block_size);
            }
            self.committed += self.block_size;

            stats::TOTAL_COMMITTED.fetch_add(self.block_size, Ordering::Relaxed);
            stats::BINNED_ALLOCATOR_COMMITTED.fetch_add(self.block_size, Ordering::Relaxed);

            // No eager freelist initialization — we use a bump pointer for virgin blocks.
            // This avoids touching every cache line on block commit.
            self.blocks.push(BlockMeta::new(
                self.bins_per_block,
                self.bins_per_block as usize,
            ));

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
            crate::qen_debug_assert!(block.bump_cursor() < bins_per_block);
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

            // Verify free-bin canary at offset 4 (debug and hardened builds only)
            #[cfg(any(debug_assertions, feature = "hardened"))]
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
            #[cfg(any(debug_assertions, feature = "hardened"))]
            unsafe {
                std::ptr::write_bytes(bin_ptr, 0, self.bin_size);
            }
        }

        let fc = block.free_count() - 1;
        block.set_free_count(fc);
        if fc == 0 {
            self.bit_tree.mark_full(block_idx);
        }

        #[cfg(any(debug_assertions, feature = "hardened"))]
        block.free_map.set(bin_idx as usize, false);

        // Safety: bin_ptr is non-null.
        Ok(unsafe { NonNull::new_unchecked(bin_ptr) })
    }

    /// Reciprocal pair `(block, bin)` for lock-free remote lookups
    /// (`PoolRemote` keeps copies so the mask channel never touches the
    /// pool lock).
    pub(crate) fn reciprocals(&self) -> (ReciprocalDiv, ReciprocalDiv) {
        (self.block_reciprocal, self.bin_reciprocal)
    }

    /// Find a block with free bins, committing (or recommitting) one if
    /// needed. `Ok(Some(idx))` — block ready; `Ok(None)` — the pool's
    /// reservation is exhausted; `Err` — the commit syscall failed.
    fn find_or_commit_block(&mut self) -> Result<Option<usize>, VmError> {
        if let Some(idx) = self.bit_tree.find_free() {
            if !self.blocks[idx].is_committed() {
                let block_offset = idx * self.block_size;
                // Safety: block_offset is within the reserved range.
                let ptr = unsafe { NonNull::new_unchecked(self.base.as_ptr().add(block_offset)) };
                // Safety: FFI commit of a block within our reservation.
                unsafe { PlatformVmOps::commit(ptr, self.block_size)? };
                #[cfg(any(debug_assertions, feature = "hardened"))]
                // Safety: block was just committed; no live bins in it.
                unsafe {
                    std::ptr::write_bytes(ptr.as_ptr(), 0, self.block_size);
                }
                self.committed += self.block_size;
                stats::TOTAL_COMMITTED.fetch_add(self.block_size, Ordering::Relaxed);
                stats::BINNED_ALLOCATOR_COMMITTED.fetch_add(self.block_size, Ordering::Relaxed);
                self.blocks[idx] =
                    BlockMeta::new(self.bins_per_block, self.bins_per_block as usize);
            }
            return Ok(Some(idx));
        }

        // No free blocks — commit a new one
        if self.committed + self.block_size > self.reserved_size {
            return Ok(None);
        }
        let idx = self.blocks.len();
        let block_offset = idx * self.block_size;
        // Safety: block_offset is within the reserved range (checked
        // against reserved_size above).
        let ptr = unsafe { NonNull::new_unchecked(self.base.as_ptr().add(block_offset)) };
        // Safety: FFI commit of a block within our reservation.
        unsafe { PlatformVmOps::commit(ptr, self.block_size)? };
        #[cfg(any(debug_assertions, feature = "hardened"))]
        // Safety: block was just committed; no live bins in it.
        unsafe {
            std::ptr::write_bytes(ptr.as_ptr(), 0, self.block_size);
        }
        self.committed += self.block_size;
        stats::TOTAL_COMMITTED.fetch_add(self.block_size, Ordering::Relaxed);
        stats::BINNED_ALLOCATOR_COMMITTED.fetch_add(self.block_size, Ordering::Relaxed);
        self.blocks.push(BlockMeta::new(
            self.bins_per_block,
            self.bins_per_block as usize,
        ));
        self.bit_tree.mark_free(idx);
        Ok(Some(idx))
    }

    /// Batch-allocate up to `count` bins directly into `out` (a pointer
    /// array with room for `count` entries). The bump fast path only
    /// COMPUTES addresses — freshly committed bins are never read or
    /// written, so a refill adds zero object-memory traffic (the
    /// chain-based predecessor wrote a link into every bin and the
    /// receiver walked them back: 2×N touches of about-to-be-user
    /// memory). The freelist path still pops per bin — its links live in
    /// the pool's freed-bin words by design. A mid-batch commit failure
    /// returns the partial batch (nothing leaks: the pointers are already
    /// in `out`).
    ///
    /// # Safety
    /// `out` must be valid for `count` pointer writes.
    pub unsafe fn alloc_batch_array(
        &mut self,
        out: *mut *mut u8,
        count: usize,
    ) -> Result<u32, VmError> {
        let mut allocated = 0usize;

        while allocated < count {
            let block_idx = match self.find_or_commit_block() {
                Ok(Some(idx)) => idx,
                // Reservation exhausted: return what we have.
                Ok(None) => break,
                // Genuine commit failure: return the partial batch rather
                // than losing it (mirrors the freelist fallback below).
                Err(e) => {
                    if allocated == 0 {
                        return Err(e);
                    }
                    break;
                }
            };

            let block = &mut self.blocks[block_idx];
            block.check_canary();

            // Batch-bump fast path: block has untouched bins via bump cursor
            if block.free_head() == BIN_SENTINEL {
                let cursor = block.bump_cursor();
                let remaining_in_block = self.bins_per_block - cursor;
                // Saturate rather than truncate: a request above u16::MAX
                // must clamp; the min with remaining_in_block bounds the
                // result to bins_per_block.
                let want = u16::try_from(count - allocated).unwrap_or(u16::MAX);
                let n = remaining_in_block.min(want);

                let block_offset = block_idx * self.block_size;
                // Safety: block_idx is a tracked block within the reservation.
                let base_ptr = unsafe { self.base.as_ptr().add(block_offset) };

                for i in 0..n {
                    let bin_offset = (cursor + i) as usize * self.bin_size;
                    #[cfg(any(debug_assertions, feature = "hardened"))]
                    {
                        block.free_map.set((cursor + i) as usize, false);
                    }
                    // Safety: the slot is within `out` per the safety
                    // contract; the address stays inside this committed
                    // block (cursor + n <= bins_per_block).
                    unsafe {
                        *out.add(allocated + i as usize) = base_ptr.add(bin_offset);
                    }
                }

                block.set_bump_cursor(cursor + n);
                let fc = block.free_count() - n;
                block.set_free_count(fc);
                if fc == 0 {
                    self.bit_tree.mark_full(block_idx);
                }
                allocated += n as usize;
            } else {
                // Freelist path: fall back to single alloc per bin
                match self.alloc() {
                    Ok(ptr) => {
                        // Safety: slot within `out` per the contract.
                        unsafe { *out.add(allocated) = ptr.as_ptr() };
                        allocated += 1;
                    }
                    Err(VmError::PoolExhausted) => break,
                    // Genuine failure: report it rather than mislabelling
                    // as exhaustion, unless part of the batch succeeded.
                    Err(e) => {
                        if allocated == 0 {
                            return Err(e);
                        }
                        break;
                    }
                }
            }
        }

        if allocated > 0 {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "allocated <= count, bounded by pool capacity, far below u32::MAX"
            )]
            Ok(allocated as u32)
        } else {
            Err(VmError::PoolExhausted)
        }
    }

    pub fn free(&mut self, ptr: NonNull<u8>) {
        let ptr_addr = ptr.as_ptr() as usize;
        let base_addr = self.base.as_ptr() as usize;

        // Dev-mode validation checks (range, commitment, alignment, canary).
        // Disabled in release builds for maximum performance.
        #[cfg(any(debug_assertions, feature = "hardened"))]
        {
            // Range check (must come first to prevent underflow in offset calc)
            assert!(
                ptr_addr >= base_addr && ptr_addr < base_addr + self.reserved_size,
                "Pointer {ptr:p} does not belong to this Pool"
            );
        }

        let offset = ptr_addr - base_addr;
        let block_idx = self.block_reciprocal.div(offset);
        let offset_in_block = offset - block_idx * self.block_size;
        // Free hot path: no checked conversion here by design. The result is
        // < bins_per_block (≤ u16::MAX, enforced at construction) for any
        // pointer that satisfies free()'s contract; a violating pointer is
        // already UB per that contract.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "bounded by bins_per_block <= u16::MAX for in-contract pointers; hot path"
        )]
        let bin_idx = self.bin_reciprocal.div(offset_in_block) as u16;

        #[cfg(any(debug_assertions, feature = "hardened"))]
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

            // Alignment check (Granlund-Montgomery divisibility test)
            assert!(
                self.bin_reciprocal.is_multiple(offset_in_block),
                "Pointer {ptr:p} is not aligned to bin size {}",
                self.bin_size
            );
        }

        let block = &mut self.blocks[block_idx];
        block.check_canary();

        // Double-free check (debug/hardened-only; production uses free-bin canary)
        #[cfg(any(debug_assertions, feature = "hardened"))]
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
        // Write free-bin canary at offset 4 (debug and hardened builds only)
        #[cfg(any(debug_assertions, feature = "hardened"))]
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
            self.decommit_pending
                .push((block_idx, self.decommit_cooldown));
        }
    }

    /// Full trim with inline syscalls, for exclusive-`&mut` callers (tests,
    /// single-threaded use). `BinnedAllocator::trim` uses the three-phase
    /// `begin_trim`/`finish_trim` protocol instead so the syscalls run
    /// without holding the shared pool mutex.
    ///
    /// Every decommit — including trailing blocks — honours
    /// `decommit_cooldown`: a fully-empty block survives that many trim
    /// passes before its pages are returned to the OS.
    #[allow(dead_code)]
    pub fn trim(&mut self) {
        let mut batch = self.begin_trim();
        for req in &mut batch {
            // Safety: FFI call to decommit memory owned by this pool.
            req.ok = unsafe { PlatformVmOps::decommit(req.ptr, req.size) }.is_ok();
        }
        self.finish_trim(&batch);
    }
}

/// Growable open-addressing map from pool base address to pool index.
///
/// Keyed by `base >> reserved_size_shift` (unique per pool since bases are
/// `reserved_size`-aligned). Linear probing, power-of-two capacity, doubling
/// growth on insert to keep load factor ≤ 1/2. Entries are never removed
/// (pools live for the allocator's lifetime), so `get` always terminates at
/// an empty slot. Designed for the free-path hot loop: one shift, one mask,
/// typically zero probes.
struct PoolFlatMap {
    /// (key, `pool_index`) pairs. key == EMPTY means slot is vacant.
    slots: Box<[(usize, u32)]>,
    /// `reserved_size.trailing_zeros()` — shift to convert base addr to key.
    shift: u32,
    /// Number of occupied slots.
    len: usize,
}

impl PoolFlatMap {
    const INITIAL_CAPACITY: usize = 8;
    const EMPTY: usize = usize::MAX;

    fn with_shift(shift: u32) -> Self {
        Self {
            slots: vec![(Self::EMPTY, 0); Self::INITIAL_CAPACITY].into_boxed_slice(),
            shift,
            len: 0,
        }
    }

    #[inline]
    fn key(&self, base: usize) -> usize {
        base >> self.shift
    }

    fn insert(&mut self, base: usize, pool_idx: usize) {
        // Grow before insert so load factor stays ≤ 1/2: probe chains stay
        // short and an EMPTY slot always exists to terminate `get`.
        if (self.len + 1) * 2 > self.slots.len() {
            self.grow();
        }
        let k = self.key(base);
        crate::qen_debug_assert_ne!(k, Self::EMPTY, "base address maps to reserved sentinel");
        let mask = self.slots.len() - 1;
        let mut i = k & mask;
        loop {
            if self.slots[i].0 == Self::EMPTY {
                // Cold path (runs once per pool reservation). Pool counts
                // are bounded far below u32::MAX by address-space limits.
                let idx = u32::try_from(pool_idx).expect("pool index exceeds u32::MAX");
                self.slots[i] = (k, idx);
                self.len += 1;
                return;
            }
            crate::qen_debug_assert_ne!(self.slots[i].0, k, "duplicate base in PoolFlatMap");
            i = (i + 1) & mask;
        }
    }

    /// Double capacity and rehash. Cold path: runs only when a new pool is
    /// reserved (a syscall-heavy event), never on alloc/free.
    fn grow(&mut self) {
        let new_cap = self.slots.len() * 2;
        let old = std::mem::replace(
            &mut self.slots,
            vec![(Self::EMPTY, 0); new_cap].into_boxed_slice(),
        );
        let mask = new_cap - 1;
        for &(k, idx) in &old {
            if k == Self::EMPTY {
                continue;
            }
            let mut i = k & mask;
            while self.slots[i].0 != Self::EMPTY {
                i = (i + 1) & mask;
            }
            self.slots[i] = (k, idx);
        }
    }

    #[inline]
    fn get(&self, base: usize) -> Option<usize> {
        let k = self.key(base);
        let mask = self.slots.len() - 1;
        let mut i = k & mask;
        // Terminates: load factor ≤ 1/2 and no deletions guarantee an EMPTY
        // slot on every probe chain.
        loop {
            let (slot_key, slot_idx) = self.slots[i];
            if slot_key == k {
                return Some(slot_idx as usize);
            }
            if slot_key == Self::EMPTY {
                return None;
            }
            i = (i + 1) & mask;
        }
    }
}

pub(crate) struct PoolChain {
    pub pools: Vec<Pool>,
    /// O(1) base-address → pool-index lookup for the free path.
    pool_map: PoolFlatMap,
    pub active_index: usize,
    pub bin_size: usize,
    pub block_size: usize,
    pub config: BinnedAllocatorConfig,
    /// How many of `pools` have a published `PoolRemote` (see
    /// `BinnedAllocator::publish_pool_remotes`).
    pub remotes_published: usize,
}

// Safety: PoolChain owns the pools and is Send.
unsafe impl Send for PoolChain {}

impl PoolChain {
    pub fn new(bin_size: usize, block_size: usize, config: BinnedAllocatorConfig) -> Self {
        let shift = config.pool_reserved_size.trailing_zeros();
        Self {
            pools: Vec::new(),
            pool_map: PoolFlatMap::with_shift(shift),
            active_index: 0,
            bin_size,
            block_size,
            config,
            remotes_published: 0,
        }
    }

    pub fn alloc(&mut self) -> Result<NonNull<u8>, VmError> {
        if self.pools.is_empty() {
            self.add_pool()?;
        }

        match self.pools[self.active_index].alloc() {
            Ok(ptr) => Ok(ptr),
            Err(VmError::PoolExhausted) => {
                // Active pool's reservation is exhausted. Frees may have
                // opened capacity in a retired pool — prefer reusing one over
                // reserving fresh address space.
                for idx in 0..self.pools.len() {
                    if idx == self.active_index || !self.pools[idx].has_free_capacity() {
                        continue;
                    }
                    let ptr = self.pools[idx].alloc()?;
                    self.active_index = idx;
                    return Ok(ptr);
                }
                self.add_pool()?;
                self.active_index = self.pools.len() - 1;
                self.pools[self.active_index].alloc()
            }
            // Genuine failure (commit syscall refused): adding pools would
            // only burn address space — propagate.
            Err(e) => Err(e),
        }
    }

    /// Batch-allocate up to `count` bins into `out`, spreading across
    /// pools as needed. See [`Pool::alloc_batch_array`] for why this
    /// touches no bin memory on the bump path.
    ///
    /// # Safety
    /// `out` must be valid for `count` pointer writes.
    pub unsafe fn alloc_batch_array(
        &mut self,
        out: *mut *mut u8,
        count: usize,
    ) -> Result<u32, VmError> {
        if self.pools.is_empty() {
            self.add_pool()?;
        }

        let mut total = 0u32;
        while (total as usize) < count {
            let remaining = count - total as usize;
            // Safety: forwarded contract, offset by what's already written.
            match unsafe {
                self.pools[self.active_index].alloc_batch_array(out.add(total as usize), remaining)
            } {
                Ok(got) => total += got,
                Err(VmError::PoolExhausted) => {
                    self.activate_pool_with_capacity()?;
                }
                Err(e) => {
                    if total == 0 {
                        return Err(e);
                    }
                    break;
                }
            }
        }

        if total > 0 {
            Ok(total)
        } else {
            Err(VmError::PoolExhausted)
        }
    }

    /// Point `active_index` at a pool that can allocate: prefer an existing
    /// pool with spare capacity, otherwise reserve a new pool.
    fn activate_pool_with_capacity(&mut self) -> Result<(), VmError> {
        for idx in 0..self.pools.len() {
            if idx != self.active_index && self.pools[idx].has_free_capacity() {
                self.active_index = idx;
                return Ok(());
            }
        }
        self.add_pool()?;
        self.active_index = self.pools.len() - 1;
        Ok(())
    }

    fn add_pool(&mut self) -> Result<(), VmError> {
        let pool = Pool::with_config(self.bin_size, self.block_size, &self.config)?;
        let base = pool.base.as_ptr() as usize;
        let idx = self.pools.len();
        self.pools.push(pool);
        self.pool_map.insert(base, idx);
        Ok(())
    }

    pub fn free(&mut self, ptr: NonNull<u8>) {
        crate::qen_debug_assert!(
            !self.pools.is_empty(),
            "PoolChain::free called with no pools (bin_size={})",
            self.bin_size
        );
        assert!(
            !self.pools.is_empty(),
            "Pointer {:p} does not belong to any pool in this chain (bin_size={})",
            ptr,
            self.bin_size
        );

        let ptr_addr = ptr.as_ptr() as usize;
        // O(1) pool lookup via span-aligned base masking.
        // Pool bases are aligned to `reserved_size` (power of 2), so
        // `ptr & !(reserved_size - 1)` gives the pool base directly.
        let reserved_size = self.pools[0].reserved_size;
        crate::qen_debug_assert!(reserved_size.is_power_of_two());
        let mask = !(reserved_size - 1);
        let masked_base = ptr_addr & mask;

        if let Some(pool_idx) = self.pool_map.get(masked_base) {
            self.pools[pool_idx].free(ptr);
            return;
        }

        panic!(
            "Pointer {:p} does not belong to any pool in this chain (bin_size={})",
            ptr, self.bin_size
        );
    }

    /// Full trim with inline syscalls (exclusive-`&mut` convenience;
    /// `BinnedAllocator::trim` uses the three-phase protocol instead).
    #[allow(dead_code)]
    pub fn trim(&mut self) {
        let mut batch = self.begin_trim();
        for req in &mut batch {
            // Safety: FFI call to decommit memory owned by this chain's pools.
            req.ok = unsafe { PlatformVmOps::decommit(req.ptr, req.size) }.is_ok();
        }
        self.finish_trim(&batch);
    }

    /// Phase 1 of trim across all pools: detach cooled-down empty blocks and
    /// return the decommit requests to execute without the chain lock held.
    pub fn begin_trim(&mut self) -> Vec<TrimDecommit> {
        let mut all = Vec::new();
        for (idx, pool) in self.pools.iter_mut().enumerate() {
            let mut reqs = pool.begin_trim();
            for req in &mut reqs {
                req.pool_idx = idx;
            }
            all.append(&mut reqs);
        }
        all
    }

    /// Phase 3 of trim: integrate executed decommits into their pools, then
    /// pop trailing decommitted blocks (metadata only).
    pub fn finish_trim(&mut self, batch: &[TrimDecommit]) {
        for req in batch {
            self.pools[req.pool_idx].finish_trim_one(req);
        }
        for pool in &mut self.pools {
            pool.pop_trailing_decommitted();
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
        let reserved_size = self.pools[0].reserved_size;
        crate::qen_debug_assert!(reserved_size.is_power_of_two());
        let mask = !(reserved_size - 1);
        let masked_base = ptr_addr & mask;

        if let Some(pool_idx) = self.pool_map.get(masked_base) {
            return self.pools[pool_idx].integrate_precommit(req);
        }
        false
    }
}

// ----------------------------------------------------------------------------
// Global Recycler — sharded exchange-detach, lock-free (ABA-safe)
// ----------------------------------------------------------------------------

use crate::sync::atomic::AtomicPtr;
use crate::sync::atomic::AtomicU128;
// Only the non-loom shard-index counter uses AtomicUsize.
#[cfg(not(loom))]
use crate::sync::atomic::AtomicUsize;

/// Offset within a freed bin where the recycler stores its inter-bundle link.
/// Bytes 0..8 hold the intra-bundle (`LocalFreeList`) next pointer.
/// Bytes 8..16 hold the recycler stack link (next bundle head in Treiber stack).
/// Requires min bin size >= 16, which is always true (smallest `SIZE_CLASS` = 16).
const RECYCLER_LINK_OFFSET: usize = std::mem::size_of::<usize>();

/// Number of shards per size class. Reduces contention by spreading
/// producers and consumers across independent stacks.
/// Under Loom, reduced to 2: enough to model every cross-shard behaviour
/// (pop's probe of alternate shards, `drain_all`'s multi-shard chain
/// stitching) while keeping the state space tractable. A single shard —
/// as used previously — made all sharding logic architecturally
/// unreachable by the model checker.
#[cfg(not(loom))]
const RECYCLER_SHARD_COUNT: usize = 4;
#[cfg(loom)]
pub(crate) const RECYCLER_SHARD_COUNT: usize = 2;

/// 128-bit tagged pointer for ABA-safe Treiber stack operations.
///
/// Packed into a single `u128` for double-width compare-and-swap (DWCAS):
///
/// ```text
///   bits [127:96]  bundle count        (32 bits)
///   bits [95:64]   generation counter  (32 bits)
///   bits [63:0]    pointer             (64 bits, full virtual address)
/// ```
///
/// The generation counter is incremented on every successful CAS, preventing
/// ABA: even if a node is popped, reused, freed, and pushed back at the same
/// address, the generation will differ and the CAS will correctly fail.
/// 32 generation bits wrap after 2^31 operations on one shard; a false CAS
/// success additionally requires a thread to stall across an exact multiple
/// of 2^32 generations that lands on an identical head pointer — an accepted
/// risk, strictly stronger than common 16-bit-tag schemes.
///
/// The shard's bundle count rides in the same word so occupancy is updated
/// atomically with every push/detach. (A separate counter would race with
/// detach — reset after the slot is already open to new pushes — silently
/// unbinding the `recycler_max_bundles` memory cap.)
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
    fn new(ptr: *mut u8, generation: u32, count: u32) -> Self {
        // Provenance: packing a pointer into an integer word is inherent to
        // the DWCAS design; expose_provenance() makes the round-trip
        // explicit (see `ptr()`).
        Self(
            u128::from(count) << 96
                | u128::from(generation) << 64
                | (ptr.expose_provenance() as u128),
        )
    }

    #[inline]
    fn ptr(self) -> *mut u8 {
        std::ptr::with_exposed_provenance_mut(self.0 as usize)
    }

    #[inline]
    fn count(self) -> u32 {
        (self.0 >> 96) as u32
    }

    #[inline]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "intentional: extracts the 32-bit generation lane from the packed word"
    )]
    fn generation(self) -> u32 {
        (self.0 >> 64) as u32
    }

    #[inline]
    fn is_null(self) -> bool {
        self.ptr().is_null()
    }
}

impl std::fmt::Debug for TaggedPtr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TaggedPtr({:p}, generation={}, count={})",
            self.ptr(),
            self.generation(),
            self.count()
        )
    }
}

/// Loom-only shard selection override, so model checks can place threads
/// on specific shards and drive the cross-shard paths deterministically.
#[cfg(loom)]
pub(crate) mod loom_shard {
    loom::thread_local! {
        static SHARD_OVERRIDE: std::cell::Cell<usize> = std::cell::Cell::new(0);
    }

    /// Set the calling loom-thread's recycler shard (defaults to 0).
    pub fn set_shard(idx: usize) {
        SHARD_OVERRIDE.with(|c| c.set(idx % super::RECYCLER_SHARD_COUNT));
    }

    pub(super) fn get_shard() -> usize {
        SHARD_OVERRIDE.with(std::cell::Cell::get)
    }
}

/// Thread-local shard index for recycler shard selection.
/// Under Loom, reads the per-thread override (default shard 0) so tests
/// control placement.
#[cfg(loom)]
fn recycler_shard_index() -> usize {
    loom_shard::get_shard()
}

/// Round-robin shard assignment, one per thread.
#[cfg(not(loom))]
fn recycler_shard_index() -> usize {
    thread_local! {
        static SHARD_IDX: usize = {
            crate::sync::static_atomic! {
                static COUNTER: AtomicUsize = AtomicUsize::new(0);
            }
            COUNTER.fetch_add(1, Ordering::Relaxed) % RECYCLER_SHARD_COUNT
        };
    }
    SHARD_IDX.with(|&idx| idx)
}

/// A single shard of the recycler, padded to a cache line to prevent false sharing.
/// Each shard is one ABA-safe Treiber stack slot; the bundle count lives in
/// the tagged word itself (see [`TaggedPtr`]).
///
/// 64-byte alignment = one shard per cache line on both `x86_64` and ARM64.
/// Under Loom we relax to 16 bytes to avoid stack overflow in the
/// model-checker's constrained coroutine stacks (correctness-only, not perf).
#[cfg_attr(not(loom), repr(align(64)))]
#[cfg_attr(loom, repr(align(16)))]
struct RecyclerShard {
    slot: AtomicU128,
}

/// Sharded lock-free recycler using exchange-detach pop.
///
/// Each size class has `RECYCLER_SHARD_COUNT` independent Treiber stacks.
/// Threads are assigned a primary shard and probe alternates on empty pop.
///
/// Push: simple CAS loop (no reservation, no odd-generation spin).
/// Pop: `swap(tagged_null)` detaches entire shard, then bounded-greedy walk.
pub(crate) struct GlobalRecycler {
    shards: [[RecyclerShard; RECYCLER_SHARD_COUNT]; NUM_SIZE_CLASSES],
    max_bundles_per_shard: u32,
}

impl GlobalRecycler {
    /// Pointer to the recycler inter-bundle link field at offset 8.
    /// `AtomicPtr` (not `AtomicUsize`) so the stored links keep their
    /// provenance — miri can then track them like any other pointer.
    #[inline]
    pub(crate) unsafe fn recycler_link_atomic_ptr(node: *mut u8) -> *mut AtomicPtr<u8> {
        // Safety: caller guarantees `node` is a live bin of >= 16 bytes, so
        // offset 8 is in-bounds and aligned for AtomicPtr.
        let p = unsafe { node.add(RECYCLER_LINK_OFFSET) }
            .cast::<()>()
            .cast::<AtomicPtr<u8>>();
        #[cfg(any(debug_assertions, feature = "hardened"))]
        crate::qen_debug_assert!(
            (p as usize).is_multiple_of(std::mem::align_of::<AtomicPtr<u8>>()),
            "recycler link field is not atomically aligned: {p:p}",
        );
        p
    }

    pub fn new(max_bundles: u32) -> Self {
        Self {
            shards: std::array::from_fn(|_| {
                std::array::from_fn(|_| RecyclerShard {
                    slot: AtomicU128::new(TaggedPtr::NULL.0),
                })
            }),
            max_bundles_per_shard: max_bundles
                .div_ceil(u32::try_from(RECYCLER_SHARD_COUNT).expect("shard count fits u32")),
        }
    }

    /// Push a bundle onto the recycler. No reservation protocol — just CAS.
    ///
    /// Returns `Some(bundle_head)` if the recycler is full (caller must flush to pool).
    pub fn push(&self, pool_idx: usize, bundle_head: NonNull<u8>) -> Option<NonNull<u8>> {
        let shard = recycler_shard_index();
        let slot = &self.shards[pool_idx][shard].slot;
        let new_ptr = bundle_head.as_ptr();

        loop {
            let old = TaggedPtr(slot.load(Ordering::Acquire));

            // Occupancy cap: the count rides in the same atomic word as the
            // stack head, so this check is linearized with the CAS below and
            // cannot race with a concurrent detach.
            if old.count() >= self.max_bundles_per_shard {
                return Some(bundle_head);
            }

            // Write inter-bundle link (offset 8) to point to current stack top.
            // Safety: bundle_head is a live bin (>= 16 bytes) we own until
            // the CAS below publishes it; the link field is atomic.
            unsafe {
                (*Self::recycler_link_atomic_ptr(new_ptr)).store(old.ptr(), Ordering::Relaxed);
            }

            // Bump generation by 2 (even → even)
            let new = TaggedPtr::new(new_ptr, old.generation().wrapping_add(2), old.count() + 1);

            if slot
                .compare_exchange_weak(old.0, new.0, Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                return None;
            }
        }
    }

    /// Pop a bundle from the recycler via exchange-detach.
    ///
    /// Swaps the shard to null atomically, takes the first bundle, and
    /// pushes remaining bundles back via a bounded walk (at most
    /// `POP_WALK_BUDGET` inter-bundle links). Any excess beyond the
    /// budget is written to `overflow` for the caller to flush directly
    /// to the pool, bounding worst-case pop latency.
    ///
    /// Probes alternate shards if primary is empty.
    /// CAS-detach: atomically replace a shard's head with null, taking the
    /// entire chain. The generation in the installed null is derived from
    /// the value actually replaced (not a stale peek), ensuring monotonicity.
    /// Returns `None` if the shard is empty.
    fn cas_detach(&self, pool_idx: usize, shard: usize) -> Option<TaggedPtr> {
        let rs = &self.shards[pool_idx][shard];
        loop {
            let current = TaggedPtr(rs.slot.load(Ordering::Acquire));
            if current.is_null() {
                return None;
            }
            // Count resets to 0 in the same CAS that empties the slot —
            // concurrent pushes landing after the detach start from the new
            // word and are never miscounted.
            let null_tagged = TaggedPtr::new(
                std::ptr::null_mut(),
                current.generation().wrapping_add(2),
                0,
            );
            if rs
                .slot
                .compare_exchange_weak(
                    current.0,
                    null_tagged.0,
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                return Some(current);
            }
        }
    }

    pub fn pop(&self, pool_idx: usize, overflow: &mut Option<NonNull<u8>>) -> Option<NonNull<u8>> {
        let primary = recycler_shard_index();

        for probe in 0..RECYCLER_SHARD_COUNT {
            let shard = (primary + probe) % RECYCLER_SHARD_COUNT;

            let Some(old) = self.cas_detach(pool_idx, shard) else {
                continue;
            };

            let head = old.ptr();

            // Read inter-bundle link to get remaining bundles.
            // Safety: the detach CAS gave us exclusive ownership of the
            // chain; head is a live bin with an atomic link field.
            let next_bundle =
                unsafe { (*Self::recycler_link_atomic_ptr(head)).load(Ordering::Relaxed) };

            // Clear the inter-bundle link on the returned head.
            // Safety: as above — head is exclusively ours after the detach.
            unsafe {
                (*Self::recycler_link_atomic_ptr(head))
                    .store(std::ptr::null_mut(), Ordering::Relaxed);
            }

            // Push remaining bundles back with bounded walk
            if let Some(remainder) = NonNull::new(next_bundle) {
                // Walk up to POP_WALK_BUDGET links to find tail of prefix
                let mut count = 1u32;
                let mut tail = remainder.as_ptr();
                while count < Self::POP_WALK_BUDGET {
                    // Safety: walking the exclusively-owned detached chain.
                    let next =
                        unsafe { (*Self::recycler_link_atomic_ptr(tail)).load(Ordering::Relaxed) };
                    if next.is_null() {
                        break;
                    }
                    count += 1;
                    tail = next;
                }

                // Snip at budget: anything beyond is overflow
                if count == Self::POP_WALK_BUDGET {
                    // Safety: walking the exclusively-owned detached chain.
                    let beyond =
                        unsafe { (*Self::recycler_link_atomic_ptr(tail)).load(Ordering::Relaxed) };
                    if !beyond.is_null() {
                        *overflow = NonNull::new(beyond);
                        // Safety: snipping the exclusively-owned chain.
                        unsafe {
                            (*Self::recycler_link_atomic_ptr(tail))
                                .store(std::ptr::null_mut(), Ordering::Relaxed);
                        }
                    }
                }

                self.push_chain_back(pool_idx, shard, remainder, tail, count);
            }

            return NonNull::new(head);
        }

        None
    }

    /// Maximum inter-bundle links walked when pushing remainder back after
    /// exchange-detach pop. Bounds worst-case pointer-chase cost on the hot
    /// path. Any excess bundles are returned to the caller for direct pool
    /// flush rather than recycling through the contended shard.
    const POP_WALK_BUDGET: u32 = 4;

    /// Push a chain segment (head..=tail, `count` bundles) onto a shard.
    fn push_chain_back(
        &self,
        pool_idx: usize,
        shard: usize,
        chain_head: NonNull<u8>,
        tail: *mut u8,
        count: u32,
    ) {
        let rs = &self.shards[pool_idx][shard];

        loop {
            let old = TaggedPtr(rs.slot.load(Ordering::Acquire));

            // Safety: the chain segment is exclusively ours until the CAS
            // below publishes it; tail is a live bin with an atomic link.
            unsafe {
                (*Self::recycler_link_atomic_ptr(tail)).store(old.ptr(), Ordering::Relaxed);
            }

            // Deliberately no occupancy check: this returns bundles we just
            // detached, so it never grows the recycler beyond what pop found
            // (plus the cap-bounded concurrent pushes).
            let new = TaggedPtr::new(
                chain_head.as_ptr(),
                old.generation().wrapping_add(2),
                old.count().saturating_add(count),
            );

            if rs
                .slot
                .compare_exchange_weak(old.0, new.0, Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                return;
            }
        }
    }

    /// Drain all bundles from all shards for a given pool index.
    /// Used during trim to return recycled memory to pools.
    pub fn drain_all(&self, pool_idx: usize) -> Option<NonNull<u8>> {
        let mut chain_head: *mut u8 = std::ptr::null_mut();

        for shard in 0..RECYCLER_SHARD_COUNT {
            let Some(old) = self.cas_detach(pool_idx, shard) else {
                continue;
            };

            if chain_head.is_null() {
                chain_head = old.ptr();
            } else {
                // Walk to end of existing chain and link
                let mut tail = chain_head;
                loop {
                    // Safety: walking chains detached by cas_detach — all
                    // exclusively owned by this call.
                    let next =
                        unsafe { (*Self::recycler_link_atomic_ptr(tail)).load(Ordering::Relaxed) };
                    if next.is_null() {
                        break;
                    }
                    tail = next;
                }
                // Safety: stitching exclusively-owned detached chains.
                unsafe {
                    (*Self::recycler_link_atomic_ptr(tail)).store(old.ptr(), Ordering::Relaxed);
                }
            }
        }

        NonNull::new(chain_head)
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

// Raw `#[thread_local]` static in non-loom builds: access is an offset
// from the TLS base register — no `LocalKey` state checks, no closure
// indirection, no lazy-init branch (measured: the `LocalKey` path cost
// 1.58 ns of the 4.78 ns hot alloc/free pair). Two consequences, both
// deliberate:
//
// - No destructor: flush-at-thread-exit is preserved by `TLS_EXIT_GUARD`
//   below, registered once per thread from `bind_slow`.
// - Operations arriving AFTER the guard has flushed (later TLS
//   destructors that allocate) hit plain memory and simply repopulate the
//   cache, whose bins then leak at thread death. The `LocalKey` design
//   would panic there instead; mimalloc/tcmalloc make the same
//   leak-not-panic trade.
//
// loom builds keep the `thread_local!` path: loom neither supports
// `#[thread_local]` nor models the TLS access itself (see loom_tests.rs
// design notes).
#[cfg(not(loom))]
#[thread_local]
static RAW_THREAD_CACHE: ThreadCacheHandle = ThreadCacheHandle::new_const();

/// Registers the flush-at-exit destructor for `RAW_THREAD_CACHE`. Touched
/// exactly once per thread, from `bind_slow` (cold): const-init +
/// first-access registration keeps it off the hot path entirely.
#[cfg(not(loom))]
struct TlsExitGuard;

#[cfg(not(loom))]
impl Drop for TlsExitGuard {
    fn drop(&mut self) {
        // Safety: destructors run on the owning thread; the handle's
        // single-thread access contract holds.
        let cache = crate::sync::unsafe_cell_get_mut!(RAW_THREAD_CACHE.cache);
        cache.teardown();
    }
}

#[cfg(not(loom))]
thread_local! {
    static TLS_EXIT_GUARD: TlsExitGuard = const { TlsExitGuard };
}

#[cfg(loom)]
thread_local! {
    static GLOBAL_THREAD_CACHE: ThreadCacheHandle = ThreadCacheHandle::new();
}

/// Run `f` with this thread's cache handle. Non-loom: one TLS-relative
/// address computation. Loom: the modeled `thread_local!` path.
#[inline]
#[cfg(not(loom))]
fn with_tls<R>(f: impl FnOnce(&ThreadCacheHandle) -> R) -> R {
    f(&RAW_THREAD_CACHE)
}

#[inline]
#[cfg(loom)]
fn with_tls<R>(f: impl FnOnce(&ThreadCacheHandle) -> R) -> R {
    GLOBAL_THREAD_CACHE.with(f)
}

/// How many fast-path operations may elapse between checks of the global
/// trim epoch. The old design paid an `Acquire` load of a shared cache line
/// on EVERY alloc/free; a thread-local countdown bounds the staleness of
/// cooperative flushing at `EPOCH_CHECK_INTERVAL` operations instead —
/// the same deferred-observation model jemalloc/mimalloc use for purging.
const EPOCH_CHECK_INTERVAL: u32 = 1024;

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
    /// Countdown to the next trim-epoch check (see `EPOCH_CHECK_INTERVAL`).
    epoch_credit: Cell<u32>,
}

// Safety: ThreadCacheHandle is confined to a single thread via thread_local!.
// The UnsafeCell<ThreadCache> and Cells are thread-local only.
unsafe impl Sync for ThreadCacheHandle {}

impl ThreadCacheHandle {
    #[cfg(not(loom))]
    const fn new_const() -> Self {
        Self {
            cache: UnsafeCell::new(ThreadCache::new_const()),
            // The epoch counter starts at 0 globally; a fresh cache is empty
            // so a spurious first flush would be a no-op anyway.
            last_seen_trim_epoch: Cell::new(0),
            epoch_credit: Cell::new(0),
        }
    }

    #[cfg(loom)]
    fn new() -> Self {
        Self {
            cache: UnsafeCell::new(ThreadCache::new()),
            last_seen_trim_epoch: Cell::new(0),
            epoch_credit: Cell::new(0),
        }
    }

    /// Hot-path epoch bookkeeping: one thread-local load/branch/store.
    /// Every `EPOCH_CHECK_INTERVAL` operations it drops into
    /// [`check_flush_now`](Self::check_flush_now), which does the real
    /// (shared) epoch load.
    #[inline]
    fn tick_epoch(&self, cache: &mut ThreadCache) {
        let credit = self.epoch_credit.get();
        if credit == 0 {
            self.check_flush_now(cache);
        } else {
            self.epoch_credit.set(credit - 1);
        }
    }

    /// Check the cooperative trim epoch and flush if signalled. Also the
    /// periodic pull for the mask channel: reconcile only on alloc miss
    /// strands returns behind well-stocked owners (deep caches rarely
    /// miss), unboundedly growing the pools — measured as a 38 GiB peak
    /// before this tick-driven drain bounded stranding to one interval.
    #[cold]
    fn check_flush_now(&self, cache: &mut ThreadCache) {
        self.epoch_credit.set(EPOCH_CHECK_INTERVAL - 1);
        cache.reconcile_remote();
        let global_epoch = CACHE_TRIM_EPOCH.load(Ordering::Acquire);
        if self.last_seen_trim_epoch.get() != global_epoch {
            self.last_seen_trim_epoch.set(global_epoch);
            cache.flush();
        } else if let Some(allocator) = cache.allocator
            && cache.cached_bytes > allocator.config.max_thread_cache_bytes
        {
            // Reconciled returns count against the byte budget; without
            // this the budget only re-binds on the next free.
            allocator.scavenge_cache(cache);
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

/// Process-wide Qen allocator instance for explicit engine allocations.
///
/// This type deliberately does **not** implement [`std::alloc::GlobalAlloc`].
/// Qen's allocator metadata uses ordinary Rust collections, so installing Qen
/// as `#[global_allocator]` would make initialization and metadata growth
/// recursively allocate through the allocator being initialized. Keep a
/// bootstrap-safe allocator such as jemalloc or the system allocator as the
/// Rust global allocator, call [`init`](Self::init) during application startup,
/// and route engine-owned allocations through this explicit API.
pub struct GlobalBinnedAllocator;

impl GlobalBinnedAllocator {
    /// Initialize the process-wide Qen allocator instance.
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

    /// Returns a reference to the initialized process-wide Qen instance.
    ///
    /// # Panics
    ///
    /// Panics if the Qen instance has not been initialized via [`init`](Self::init).
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
        with_tls(|handle| {
            // Safety: single-threaded TLS access; no re-entrancy possible
            // (alloc_with_cache accesses pools/recycler, never TLS)
            let cache = crate::sync::unsafe_cell_get_mut!(handle.cache);
            let allocator = Self::cached_allocator(cache);
            handle.tick_epoch(cache);
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
        if size == 0 || size > MAX_SMALL_SIZE {
            // Cold: zero-size and large requests take the validated
            // layout path.
            return Self::alloc(BinnedAllocator::layout_for_bytes(size)?);
        }
        with_tls(|handle| {
            // Safety: single-threaded TLS access; no re-entrancy possible.
            let cache = crate::sync::unsafe_cell_get_mut!(handle.cache);
            let allocator = Self::cached_allocator(cache);
            handle.tick_epoch(cache);
            // Every bin is 16-aligned, so byte requests skip Layout
            // construction and the alignment loop entirely.
            let pool_idx = BinnedAllocator::size_class_min_align(size);
            allocator.alloc_small_with_cache(cache, pool_idx)
        })
    }

    /// Resolve the allocator through the thread cache's binding — the
    /// `OnceLock` is only touched on a thread's first operation.
    #[inline]
    fn cached_allocator(cache: &mut ThreadCache) -> &'static BinnedAllocator {
        match cache.allocator {
            Some(allocator) => allocator,
            None => Self::bind_slow(cache),
        }
    }

    #[cold]
    fn bind_slow(cache: &mut ThreadCache) -> &'static BinnedAllocator {
        let allocator = Self::get();
        cache.bind(allocator);
        // Register the flush-at-exit destructor for the raw TLS cache
        // (once per thread; `TLS_EXIT_GUARD` is const-init so this is the
        // registration point, not an allocation).
        #[cfg(not(loom))]
        // try_with: a bind AFTER this thread's TLS destructors have run
        // (another destructor allocating) must not panic — the rebound
        // cache simply leaks at thread death, as documented on
        // RAW_THREAD_CACHE.
        let _ = TLS_EXIT_GUARD.try_with(|_| {});
        allocator
    }

    /// Free a pointer previously obtained from [`alloc`](Self::alloc).
    ///
    /// The `layout` must match the layout used to allocate the pointer.
    /// For raw byte buffers, see [`free_bytes`](Self::free_bytes).
    ///
    /// # Safety
    /// - `ptr` must have been returned by [`alloc`](Self::alloc) from this
    ///   process-wide Qen instance.
    /// - `layout` must exactly match the layout used for allocation.
    /// - `ptr` must not have been freed already.
    pub unsafe fn free(ptr: NonNull<u8>, layout: std::alloc::Layout) {
        with_tls(|handle| {
            // Safety: single-threaded TLS access; no re-entrancy possible
            let cache = crate::sync::unsafe_cell_get_mut!(handle.cache);
            let allocator = Self::cached_allocator(cache);
            handle.tick_epoch(cache);
            allocator.free_with_cache(cache, ptr, layout);
        });
    }

    /// Free a pointer previously obtained from [`alloc_bytes`](Self::alloc_bytes).
    /// For typed allocations, use [`free`](Self::free).
    ///
    /// # Safety
    /// - `ptr` must have been returned by [`alloc_bytes`](Self::alloc_bytes)
    ///   from this process-wide Qen instance.
    /// - `size` must exactly match the size used for allocation.
    /// - `ptr` must not have been freed already.
    ///
    /// # Panics
    ///
    /// Panics if a layout cannot be created for the given size (e.g., size is too large).
    pub unsafe fn free_bytes(ptr: NonNull<u8>, size: usize) {
        if size == 0 || size > MAX_SMALL_SIZE {
            // Cold: zero-size and large frees take the validated layout path.
            // Safety: ptr and size match allocation.
            unsafe {
                return Self::free(ptr, std::alloc::Layout::from_size_align(size, 1).unwrap());
            }
        }
        with_tls(|handle| {
            // Safety: single-threaded TLS access; no re-entrancy possible.
            let cache = crate::sync::unsafe_cell_get_mut!(handle.cache);
            let allocator = Self::cached_allocator(cache);
            handle.tick_epoch(cache);
            let pool_idx = BinnedAllocator::size_class_min_align(size);
            allocator.free_small_with_cache(cache, ptr, pool_idx);
        });
    }

    /// Free a pool-backed pointer WITHOUT its size. The pointer's masked
    /// base identifies its pool — and therefore its size class — in one
    /// lock-free lookup of a ~pool-count-sized, L1-resident table (the
    /// address is the metadata; other allocators answer the same question
    /// with a pagemap radix walk or an object header). Returns `false`
    /// without freeing if the pointer is not pool-backed (a large
    /// allocation or an over-aligned one): those require the sized
    /// [`free`](Self::free), because their reservations are standalone
    /// mappings the class table cannot describe.
    ///
    /// # Safety
    /// - `ptr` must be a live allocation from this process-wide Qen instance.
    /// - `ptr` must not have been freed already.
    #[must_use = "false means the pointer was NOT freed (large/foreign: use the sized API)"]
    pub unsafe fn try_free_ptr(ptr: NonNull<u8>) -> bool {
        with_tls(|handle| {
            // Safety: single-threaded TLS access; no re-entrancy possible.
            let cache = crate::sync::unsafe_cell_get_mut!(handle.cache);
            let allocator = Self::cached_allocator(cache);
            let Some(pool_idx) = allocator.class_of_ptr(ptr) else {
                return false;
            };
            handle.tick_epoch(cache);
            allocator.free_small_with_cache(cache, ptr, pool_idx);
            true
        })
    }

    /// Signal all thread caches to flush and trim global pools.
    ///
    /// Flushing is cooperative: the calling thread's cache is flushed
    /// immediately, while other threads flush on their next alloc/free.
    /// This is the standard approach used by jemalloc and mimalloc —
    /// sleeping threads flush when they wake up and allocate.
    pub fn trim() {
        // Signal all thread caches to flush within EPOCH_CHECK_INTERVAL
        // operations of their next alloc/free.
        CACHE_TRIM_EPOCH.fetch_add(1, Ordering::AcqRel);

        // Immediately flush the calling thread's own cache
        with_tls(|handle| {
            // Safety: single-threaded TLS access.
            let cache = crate::sync::unsafe_cell_get_mut!(handle.cache);
            handle.check_flush_now(cache);
        });

        // Trim global pools
        if let Some(allocator) = GLOBAL_BINNED_INSTANCE.get() {
            allocator.trim();
        }
    }
}

/// Write a plain link into a freed bin's first word (the wire format the
/// recycler, caches, and mask channel all share).
#[inline]
fn write_link(nn: NonNull<u8>, prev: *mut u8) {
    // Safety: bins are >= 16 bytes; the first word carries the chain link,
    // and the bin is exclusively held by the code building the chain.
    unsafe { *nn.cast::<*mut u8>().as_ptr() = prev }
}

/// Batch capacity of one depot slot (covers every `CLASS_BATCH`).
const DEPOT_BATCH: usize = 32;
/// Batches per class in the transfer depot. Small on purpose: the depot
/// is a hand-off buffer, not storage — capacity beyond a few batches per
/// class just hides memory from the trimmer.
#[cfg(not(loom))]
const DEPOT_SLOTS: usize = 16;
#[cfg(loom)]
const DEPOT_SLOTS: usize = 2;

/// Per-class dense transfer depot: releasing threads memcpy a batch of
/// slot entries in; refilling threads memcpy a batch out. Neither side
/// ever reads or writes the bins themselves — the recycler's intrusive
/// chains (which cost a link write per bin on release and a cold
/// dependent-load walk per bin on refill) remain only as the overflow /
/// orphan / trim path.
struct Depot {
    used: u8,
    counts: [u8; DEPOT_SLOTS],
    slots: [[*mut u8; DEPOT_BATCH]; DEPOT_SLOTS],
}

impl Depot {
    const fn new() -> Self {
        Self {
            used: 0,
            counts: [0; DEPOT_SLOTS],
            slots: [[std::ptr::null_mut(); DEPOT_BATCH]; DEPOT_SLOTS],
        }
    }
}

// Safety: Depot is plain data; all access is under its Mutex.
unsafe impl Send for Depot {}

/// A depot plus a lock-free occupancy hint: the common starved/flooded
/// cases (pop on empty, push on full) must not touch the mutex at all —
/// measured without the hint, miss-heavy 16-thread workloads halved on
/// mutex parking storms. The hint is advisory (Relaxed): a stale read
/// only costs a harmless fallback to the recycler path.
struct DepotShard {
    hint: crate::sync::atomic::AtomicU8,
    inner: Mutex<Depot>,
}

impl DepotShard {
    fn new() -> Self {
        Self {
            hint: crate::sync::atomic::AtomicU8::new(0),
            inner: Mutex::new(Depot::new()),
        }
    }
}

pub struct BinnedAllocator {
    pools: Vec<Mutex<PoolChain>>, // One chain per size class
    /// Lock-free remote-free channel: per-block return masks + owner
    /// registry (see `remote_mask`). Replaces recycler round-trips for
    /// bins whose home block has a live owner.
    remote: super::remote_mask::RemoteMaskTable,
    /// Monotonic pool gid for `remote` publishes.
    pool_gid_counter: crate::sync::atomic::AtomicU64,
    block_size: usize,
    /// OS page size, cached at construction: the alloc/free hot paths test
    /// alignment against it on every call, and reading it through
    /// `PlatformVmOps::page_size()`'s `OnceLock` cost an atomic load per op.
    page_size: usize,
    config: BinnedAllocatorConfig,
    recycler: GlobalRecycler,
    /// Per-class dense transfer depots (see [`Depot`]/[`DepotShard`]).
    depots: Box<[DepotShard]>,
    large_cache: super::large_cache::LargeAllocCache,
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
        crate::qen_debug_assert!(align.is_power_of_two() && align > 0);
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

        // Resolve block_size: 0 means auto-detect. Every size class must
        // fit at least one bin per block, so the floor is the largest class.
        if config.block_size == 0 {
            config.block_size = std::cmp::max(MAX_SMALL_SIZE, min_page);
        }
        let block_size = config.block_size;
        // Config validation is a one-time init cost: enforce at runtime in all
        // build modes. A misconfigured block_size accepted here would later
        // produce out-of-bounds pointers from safe code.
        if !block_size.is_multiple_of(min_page) || block_size < min_page {
            return Err(VmError::InitializationFailed(format!(
                "Invalid block_size {block_size}: must be a non-zero multiple of the page size ({min_page})",
            )));
        }
        // Every size class must fit at least one bin per block, otherwise the
        // corresponding pool would compute bins_per_block == 0 and hand out
        // pointers past the committed block.
        if block_size < MAX_SMALL_SIZE {
            return Err(VmError::InitializationFailed(format!(
                "Invalid block_size {block_size}: smaller than the largest size class ({MAX_SMALL_SIZE}); \
                 every size class must fit at least one bin per block",
            )));
        }
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
            pools.push(Mutex::new(PoolChain::new(
                bin_size,
                block_size,
                config.clone(),
            )));
        }

        let recycler = GlobalRecycler::new(config.recycler_max_bundles);
        // Large cache holds decommitted OS reservations for reuse, capped
        // by config.large_cache_bytes. Huge page support is auto-detected
        // from supported_page_sizes() unless explicitly disabled via config.
        let large_cache = if config.use_huge_pages {
            super::large_cache::LargeAllocCache::new(config.large_cache_bytes)
        } else {
            super::large_cache::LargeAllocCache::without_huge_pages(config.large_cache_bytes)
        };

        Ok(Self {
            pools,
            remote: super::remote_mask::RemoteMaskTable::new(config.pool_reserved_size),
            pool_gid_counter: crate::sync::atomic::AtomicU64::new(0),
            block_size,
            page_size: min_page,
            config,
            recycler,
            depots: (0..NUM_SIZE_CLASSES).map(|_| DepotShard::new()).collect(),
            large_cache,
        })
    }

    /// Push one dense batch from the cache's class stack into the depot:
    /// a single memcpy of slot entries, touching no bin memory. Returns
    /// `false` (cache untouched) when the depot is full — the caller
    /// falls back to the recycler's chain path.
    fn depot_push(&self, pool_idx: usize, cache: &mut ThreadCache) -> bool {
        let count = cache.bins[pool_idx].count();
        // One class-tuned transfer batch, same release policy the chain
        // path applies (DEPOT_BATCH only bounds the slot's storage).
        let n = count.min(u32::from(CLASS_BATCH[pool_idx]));
        if n == 0 {
            return true; // nothing to release
        }
        let shard = &self.depots[pool_idx];
        // Flooded fast-out without the lock (stale full is a harmless
        // recycler fallback).
        if usize::from(shard.hint.load(Ordering::Relaxed)) >= DEPOT_SLOTS {
            return false;
        }
        // NEVER park: a contended depot degrades to the lock-free
        // recycler instead of a futex wait (measured: blocking locks here
        // put ~60% of 16-thread time into psynch_mutexwait on miss-heavy
        // workloads).
        let Ok(mut depot) = shard.inner.try_lock() else {
            return false;
        };
        if usize::from(depot.used) == DEPOT_SLOTS {
            return false;
        }
        let slot = usize::from(depot.used);
        let src_top = (count - n) as usize;
        // Safety: the cache's slot region holds `count` written entries;
        // the depot slot holds DEPOT_BATCH >= n; regions don't overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(
                cache.slots(pool_idx).add(src_top),
                depot.slots[slot].as_mut_ptr(),
                n as usize,
            );
        }
        #[expect(clippy::cast_possible_truncation, reason = "n <= DEPOT_BATCH = 32")]
        {
            depot.counts[slot] = n as u8;
        }
        depot.used += 1;
        shard.hint.store(depot.used, Ordering::Relaxed);
        drop(depot);
        cache.bins[pool_idx].top -= n;
        cache.bins[pool_idx].note_low_water();
        cache.cached_bytes -= SIZE_CLASSES[pool_idx] * n as usize;
        true
    }

    /// Pop one dense batch from the depot into the cache's (empty) class
    /// stack: a single memcpy, touching no bin memory. Returns the count
    /// adopted (0 = depot empty).
    fn depot_pop(&self, pool_idx: usize, cache: &mut ThreadCache) -> u32 {
        crate::qen_debug_assert_eq!(
            cache.bins[pool_idx].count(),
            0,
            "depot pop on non-empty stack"
        );
        let shard = &self.depots[pool_idx];
        // Starved fast-out without the lock (stale empty is a harmless
        // recycler fallback).
        if shard.hint.load(Ordering::Relaxed) == 0 {
            return 0;
        }
        // NEVER park (see depot_push): contended pops fall through to the
        // recycler.
        let Ok(mut depot) = shard.inner.try_lock() else {
            return 0;
        };
        if depot.used == 0 {
            return 0;
        }
        depot.used -= 1;
        let slot = usize::from(depot.used);
        let n = u32::from(depot.counts[slot]);
        // Safety: the depot slot holds `n` written entries; the cache's
        // slot region holds CLASS_CAP + 1 >= n; regions don't overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(
                depot.slots[slot].as_ptr(),
                cache.slots(pool_idx),
                n as usize,
            );
        }
        shard.hint.store(depot.used, Ordering::Relaxed);
        drop(depot);
        cache.bins[pool_idx].assume_filled(n);
        cache.cached_bytes += SIZE_CLASSES[pool_idx] * n as usize;
        n
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
        if size > MAX_SMALL_SIZE || layout.align() > self.page_size {
            // Transparent large-alloc routing (covers both oversized allocations
            // and small allocations with alignment too large for any size class)
            let (ptr, _actual_size) = self.large_cache.alloc(layout)?;
            return Ok(ptr);
        }
        let pool_idx = Self::size_class(size, layout.align());
        let mut guard = self.pools[pool_idx]
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        let ptr = guard.alloc()?;
        // Keep the pointer→class lookup total over pool-backed memory:
        // every creation site publishes (see `class_of_ptr`).
        self.publish_pool_remotes(pool_idx, &mut guard);
        Ok(ptr)
    }

    /// Size class of a pool-backed pointer, or `None` if the pointer is
    /// not from this allocator's size-class pools (e.g. a large
    /// allocation). Lock-free: masks to the pool's aligned base and hits
    /// the published side table — sound because a non-pool pointer can
    /// never mask to a pool base (reservations own their whole aligned
    /// span). This is what lets `free` run without a size: the address
    /// itself is the metadata, where other allocators walk a pagemap.
    #[inline]
    pub(crate) fn class_of_ptr(&self, ptr: NonNull<u8>) -> Option<usize> {
        self.remote.lookup(ptr).map(|pr| pr.class as usize)
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
        if size > MAX_SMALL_SIZE || layout.align() > self.page_size {
            self.large_cache.free(ptr, layout);
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
            crate::qen_debug_assert!(false, "ThreadCache is bound to a different allocator");
            // Safety: Unreachable logic.
            unsafe { std::hint::unreachable_unchecked() }
        }

        let size = layout.size();
        if size == 0 {
            return Ok(Self::dangling_for_align(layout.align()));
        }
        if size > MAX_SMALL_SIZE || layout.align() > self.page_size {
            let (ptr, _) = self.large_cache.alloc(layout)?;
            return Ok(ptr);
        }
        let pool_idx = Self::size_class(size, layout.align());
        self.alloc_small_with_cache(cache, pool_idx)
    }

    /// Small-class allocation through the thread cache, routed by size
    /// class. The byte-oriented entry points call this directly (skipping
    /// `Layout` construction and the alignment loop entirely); `alloc_with_cache`
    /// routes here after its zero/large checks.
    pub(crate) fn alloc_small_with_cache(
        &self,
        cache: &mut ThreadCache,
        pool_idx: usize,
    ) -> Result<NonNull<u8>, VmError> {
        // Fast path: thread cache
        if let Some(ptr) = cache.pop_bin(pool_idx) {
            cache.cached_bytes -= SIZE_CLASSES[pool_idx];
            #[cfg(feature = "stats")]
            {
                class_stats::record_alloc(pool_idx);
                class_stats::record_cache_hit(pool_idx);
            }
            return Ok(ptr);
        }

        // If cache is unbound, we cannot safely batch-refill.
        if cache.allocator.is_none() {
            // Cold: reconstruct a class-exact layout (any size in the class
            // routes identically).
            let layout = std::alloc::Layout::from_size_align(SIZE_CLASSES[pool_idx], 1).unwrap();
            return self.alloc(layout);
        }

        // Owner reconcile: collect bins other threads returned to blocks
        // this cache owns — one dirty-list swap plus one mask swap per
        // pending block. Cheap when idle (a single load of our own head).
        if let Some(ptr) = self.reconcile_for_alloc(cache, pool_idx) {
            #[cfg(feature = "stats")]
            {
                class_stats::record_alloc(pool_idx);
                class_stats::record_cache_miss(pool_idx);
            }
            return Ok(ptr);
        }

        // Medium path 1: the dense depot — one short lock + one memcpy,
        // no bin-memory touches.
        if self.depot_pop(pool_idx, cache) > 0
            && let Some(ptr) = cache.pop_bin(pool_idx)
        {
            cache.cached_bytes -= SIZE_CLASSES[pool_idx];
            #[cfg(feature = "stats")]
            {
                class_stats::record_alloc(pool_idx);
                class_stats::record_cache_miss(pool_idx);
            }
            return Ok(ptr);
        }

        // Medium path 2: the lock-free GlobalRecycler before the pool lock.
        if let Some(ptr) = self.refill_from_recycler(cache, pool_idx) {
            #[cfg(feature = "stats")]
            {
                class_stats::record_alloc(pool_idx);
                class_stats::record_cache_miss(pool_idx);
            }
            return Ok(ptr);
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
                let base = new_pool.base.as_ptr() as usize;
                let idx = guard.pools.len();
                guard.pools.push(new_pool);
                guard.pool_map.insert(base, idx);
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

        // --- Batch refill: pool bins land directly in the class's slot
        // array; the bump path touches no bin memory at all ---
        let (bin_size, batch) = (SIZE_CLASSES[pool_idx], u32::from(CLASS_BATCH[pool_idx]));

        crate::qen_debug_assert_eq!(cache.bins[pool_idx].count(), 0, "refill on non-empty stack");
        // Safety: the class slot region holds CLASS_CAP + 1 >= batch
        // entries and the stack is empty (top == 0), so the region base is
        // the write cursor; the cache is bound (unbound was handled above).
        let refill = unsafe { guard.alloc_batch_array(cache.slots(pool_idx), batch as usize) };
        // Publish new pools' side tables (lock orders us against creators).
        self.publish_pool_remotes(pool_idx, &mut guard);
        if let Ok(got) = refill {
            drop(guard);
            // Adopt first (top = got), THEN claim: block-claim harvests
            // append at top and must not clobber the refilled entries.
            cache.bins[pool_idx].assume_filled(got);
            cache.cached_bytes += bin_size * got as usize;
            self.claim_refilled_blocks(cache, pool_idx, got);
            // Slow-start growth: one batch per refill, up to the class cap.
            let bin = &mut cache.bins[pool_idx];
            bin.max_length = (bin.max_length + batch).min(u32::from(CLASS_CAP[pool_idx]));

            if let Some(ptr) = cache.pop_bin(pool_idx) {
                cache.cached_bytes -= bin_size;
                #[cfg(feature = "stats")]
                {
                    class_stats::record_alloc(pool_idx);
                    class_stats::record_cache_miss(pool_idx);
                }
                return Ok(ptr);
            }
            // Batch adopted but empty pop is impossible (got > 0); fall
            // through defensively.
            guard = self.pools[pool_idx]
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }

        // Fallback: single alloc
        let ptr = guard.alloc()?;
        #[cfg(feature = "stats")]
        {
            class_stats::record_alloc(pool_idx);
            class_stats::record_cache_miss(pool_idx);
        }
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
                crate::qen_debug_assert!(false, "ThreadCache is bound to a different allocator");
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
        if size > MAX_SMALL_SIZE || layout.align() > self.page_size {
            self.large_cache.free(ptr, layout);
            return;
        }
        let pool_idx = Self::size_class(size, layout.align());
        self.free_small_with_cache(cache, ptr, pool_idx);
    }

    /// Small-class free through the thread cache, routed by size class.
    /// Counterpart of [`alloc_small_with_cache`](Self::alloc_small_with_cache).
    pub(crate) fn free_small_with_cache(
        &self,
        cache: &mut ThreadCache,
        ptr: NonNull<u8>,
        pool_idx: usize,
    ) {
        // Fast path: push to cache (one indexed store; the object's own
        // memory is untouched).
        cache.push_bin(pool_idx, ptr);
        #[cfg(feature = "stats")]
        class_stats::record_free(pool_idx);

        let bin_size = SIZE_CLASSES[pool_idx];
        cache.cached_bytes += bin_size;

        // Over the adaptive limit: release ONE transfer batch and keep the
        // rest cached. The dense depot takes a memcpy of the stack TOP
        // (most recently freed) with zero bin-memory touches; only a full
        // depot falls back to the recycler's intrusive chains.
        if cache.bins[pool_idx].count() > cache.bins[pool_idx].max_length {
            let batch = u32::from(CLASS_BATCH[pool_idx]);
            // The experimental mask channel routes through chains; the
            // depot would intercept its traffic (see remote_mask docs).
            if (self.config.remote_mask_channel || !self.depot_push(pool_idx, cache))
                && let Some((seg_head, seg_count)) = cache.take_top_bin(pool_idx, batch)
            {
                cache.cached_bytes -= bin_size * seg_count as usize;
                self.release_segment(pool_idx, seg_head, cache.owner_slot, cache.owner_gen);
            }
            // Free-side slow start (tcmalloc's ListTooLong policy): grow by
            // one until the limit covers a transfer batch, then by whole
            // batches, so free-heavy phases stop overflowing every batch.
            let bin = &mut cache.bins[pool_idx];
            if bin.max_length < batch {
                bin.max_length += 1;
            } else {
                bin.max_length = (bin.max_length + batch).min(u32::from(CLASS_CAP[pool_idx]));
            }
        }

        // Byte budget across all classes: scavenge toward low-water marks.
        if cache.cached_bytes > self.config.max_thread_cache_bytes {
            self.scavenge_cache(cache);
        }
    }

    /// Recycler medium path of the alloc miss ladder: walk a popped bundle
    /// (and any overflow bundle) into the class's slot array — the
    /// boundary walk C4 will replace with dense batches — then pop.
    fn refill_from_recycler(
        &self,
        cache: &mut ThreadCache,
        pool_idx: usize,
    ) -> Option<NonNull<u8>> {
        let mut overflow = None;
        let bundle_head = self.recycler.pop(pool_idx, &mut overflow)?;
        let (mut received, spill) = cache.receive_walk_bin(pool_idx, bundle_head);
        self.release_spill(pool_idx, spill, cache);
        // The cache's spill threshold naturally flushes any excess from
        // the overflow bundle back to the recycler on the next free.
        if let Some(extra) = overflow {
            let (r2, spill2) = cache.receive_walk_bin(pool_idx, extra);
            received += r2;
            self.release_spill(pool_idx, spill2, cache);
        }
        cache.cached_bytes += SIZE_CLASSES[pool_idx] * received as usize;
        let ptr = cache.pop_bin(pool_idx)?;
        cache.cached_bytes -= SIZE_CLASSES[pool_idx];
        Some(ptr)
    }

    /// Route a receive spill (chain beyond a class's slot capacity) back
    /// through the release path; returns the spilled count for accounting.
    fn release_spill(
        &self,
        pool_idx: usize,
        spill: Option<(NonNull<u8>, u32)>,
        cache: &ThreadCache,
    ) -> u32 {
        match spill {
            Some((head, n)) => {
                self.release_segment(pool_idx, head, cache.owner_slot, cache.owner_gen);
                n
            }
            None => 0,
        }
    }

    /// Publish side tables for any pools created since the last call.
    /// Callers hold this class's pool lock, so creation and publish are
    /// ordered; a reader that races a not-yet-published pool simply falls
    /// back to the recycler.
    fn publish_pool_remotes(&self, pool_idx: usize, guard: &mut PoolChain) {
        while guard.remotes_published < guard.pools.len() {
            let p = &guard.pools[guard.remotes_published];
            let (block_recip, bin_recip) = p.reciprocals();
            #[expect(
                clippy::cast_possible_truncation,
                reason = "pool_idx < NUM_SIZE_CLASSES (96)"
            )]
            let remote = super::remote_mask::PoolRemote::new(
                pool_idx as u16,
                p.base.as_ptr() as usize,
                p.bin_size,
                p.block_size,
                usize::from(p.bins_per_block),
                p.reserved_size,
                (block_recip, bin_recip),
            );
            self.remote.publish(remote, &self.pool_gid_counter);
            guard.remotes_published += 1;
        }
    }

    /// Owner-reconcile step of the alloc miss ladder: pull pending mask
    /// returns home across all classes; if the wanted class received
    /// bins, pop one.
    fn reconcile_for_alloc(&self, cache: &mut ThreadCache, pool_idx: usize) -> Option<NonNull<u8>> {
        if cache.owner_slot == u16::MAX {
            return None;
        }
        let (slot, generation) = (cache.owner_slot, cache.owner_gen);
        let mut hit = false;
        self.remote
            .reconcile(slot, generation, write_link, |class, head, _tail, count| {
                let c = class as usize;
                let spill = cache.receive_chain_bin(c, head, count);
                let adopted = count - spill.as_ref().map_or(0, |s| s.1);
                cache.cached_bytes += SIZE_CLASSES[c] * adopted as usize;
                if let Some((sh, _)) = spill {
                    // Beyond the class's slot capacity: our own blocks, so
                    // publish declines (self-owned) and it lands in the
                    // recycler where any thread can use it.
                    self.release_segment(c, sh, slot, generation);
                }
                if c == pool_idx {
                    hit = true;
                }
            });
        if hit && let Some(ptr) = cache.pop_bin(pool_idx) {
            cache.cached_bytes -= SIZE_CLASSES[pool_idx];
            return Some(ptr);
        }
        None
    }

    /// Take ownership of the blocks a refill batch came from, harvesting
    /// any bins already parked in their masks (published toward the
    /// previous owner — leaving them would strand capacity). Ownership is
    /// what routes future remote frees of these bins back to this cache.
    /// Reads the freshly-refilled entries `slots[top - got .. top]`;
    /// harvests append at `top`, so the walk's window is stable.
    fn claim_refilled_blocks(&self, cache: &mut ThreadCache, pool_idx: usize, got: u32) {
        if cache.owner_slot == u16::MAX || got == 0 {
            return;
        }
        let (slot, generation) = (cache.owner_slot, cache.owner_gen);
        let slots = cache.slots(pool_idx);
        let start = (cache.bins[pool_idx].count() - got) as usize;
        let base_mask = !(self.config.pool_reserved_size - 1);
        let mut pr: Option<&super::remote_mask::PoolRemote> = None;
        let mut last_block = usize::MAX;
        for i in start..start + got as usize {
            // Safety: slots[start..start+got] were written by the refill.
            let p = unsafe { *slots.add(i) };
            // Safety: refill pointers are non-null bins.
            let nn = unsafe { NonNull::new_unchecked(p) };
            if pr.is_none_or(|r| p as usize & base_mask != r.base) {
                // Batch crossed into another pool (pool stitching).
                pr = self.remote.lookup(nn);
                last_block = usize::MAX;
            }
            let Some(r) = pr else { continue };
            let (block, _) = r.locate(nn);
            if block != last_block {
                last_block = block;
                if let Some((h, _t, cnt)) = r.claim_block(block, slot, generation, write_link) {
                    let spill = cache.receive_chain_bin(pool_idx, h, cnt);
                    let adopted = cnt - spill.as_ref().map_or(0, |s| s.1);
                    cache.cached_bytes += SIZE_CLASSES[pool_idx] * adopted as usize;
                    if let Some((sh, _)) = spill {
                        self.release_segment(pool_idx, sh, slot, generation);
                    }
                }
            }
        }
    }

    /// Return a plain-linked segment. Bins whose home block has a live
    /// foreign owner go through the mask channel (coherence cost scales
    /// with blocks touched, not bins moved); the rest — unowned blocks,
    /// dead or self owners — go to the recycler as one bundle, falling
    /// back to the pool lock if the recycler is full.
    fn release_segment(&self, pool_idx: usize, seg_head: NonNull<u8>, slot: u16, generation: u32) {
        // Channel disabled: no block can have an owner, so skip the
        // per-bin walk and take the recycler path directly (the exact
        // pre-mask code path).
        if !self.config.remote_mask_channel {
            if let Some(rejected_head) = self.recycler.push(pool_idx, seg_head) {
                let mut guard = self.pools[pool_idx]
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let mut node = Some(rejected_head);
                while let Some(n) = node {
                    // Safety: n is a freed bin whose first word holds the link.
                    let next = unsafe { *n.cast::<*mut u8>().as_ptr() };
                    guard.free(n);
                    node = NonNull::new(next);
                }
            }
            return;
        }
        let mut leftover: Option<NonNull<u8>> = None;
        let mut node = Some(seg_head);
        while let Some(n) = node {
            // Safety: segment bins are plain-linked via the first word.
            let next = unsafe { *n.cast::<*mut u8>().as_ptr() };
            if !self.remote.publish_bin(n, slot, generation) {
                write_link(n, leftover.map_or(std::ptr::null_mut(), NonNull::as_ptr));
                leftover = Some(n);
            }
            node = NonNull::new(next);
        }
        let Some(seg_head) = leftover else {
            return;
        };
        if let Some(rejected_head) = self.recycler.push(pool_idx, seg_head) {
            let mut guard = self.pools[pool_idx]
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);

            let mut node = Some(rejected_head);
            while let Some(n) = node {
                // Safety: n is a freed bin whose first word holds the link.
                let next = unsafe { *n.cast::<*mut u8>().as_ptr() };
                guard.free(n);
                node = NonNull::new(next);
            }
        }
    }

    /// Bring the cache back under its byte budget.
    ///
    /// Pass 1 is tcmalloc's Scavenge: release half of each class's
    /// untouched surplus (the low-water mark) in batch-sized segments, and
    /// shrink idle lists' adaptive limits so chronically deep classes stop
    /// re-crossing the budget. Pass 2 guarantees progress: when every list
    /// is hot (low-water 0 everywhere) pass 1 releases nothing, and
    /// without a forced release the very next free would re-enter here —
    /// an all-classes walk per free. It releases from the largest bins
    /// down to half the budget (hysteresis), draining L0s if it must;
    /// with every list empty the cache holds zero bytes, so termination
    /// under budget is unconditional.
    #[cold]
    fn scavenge_cache(&self, cache: &mut ThreadCache) {
        for pool_idx in 0..NUM_SIZE_CLASSES {
            let release = cache.bins[pool_idx].low_water / 2;
            let bin_size = SIZE_CLASSES[pool_idx];
            let batch = u32::from(CLASS_BATCH[pool_idx]);
            if release > 0 {
                let mut remaining = release;
                while remaining > 0 {
                    let n = remaining.min(batch);
                    // Bottom of the stack = the depth the interval never
                    // reached: exactly the bins the low-water policy says
                    // to release (the shift is bounded and this path is
                    // cold).
                    let Some((seg_head, seg_count)) = cache.take_bottom_bin(pool_idx, n) else {
                        break;
                    };
                    cache.cached_bytes -= bin_size * seg_count as usize;
                    self.release_segment(pool_idx, seg_head, cache.owner_slot, cache.owner_gen);
                    remaining = remaining.saturating_sub(seg_count);
                }
                // The stack carried untouched stock through the whole
                // observation interval — its limit is too generous.
                // Floor at one transfer batch (tcmalloc's floor): under
                // byte-budget pressure a class may shrink below the
                // growth floor, trading recycler trips for held memory.
                let bin = &mut cache.bins[pool_idx];
                bin.max_length = bin.max_length.saturating_sub(batch).max(batch);
            }
            cache.bins[pool_idx].reset_low_water();
        }

        let target = self.config.max_thread_cache_bytes / 2;
        if cache.cached_bytes <= target {
            return;
        }
        for pool_idx in (0..NUM_SIZE_CLASSES).rev() {
            if cache.bins[pool_idx].count() == 0 {
                continue;
            }
            let bin_size = SIZE_CLASSES[pool_idx];
            let batch = u32::from(CLASS_BATCH[pool_idx]);
            // A force-released class is over-provisioned for the budget:
            // without cutting its limit the stock rebuilds immediately and
            // the budget re-crosses, making this walk chronic.
            {
                let bin = &mut cache.bins[pool_idx];
                bin.max_length = bin.max_length.saturating_sub(batch).max(batch);
            }
            while let Some((seg_head, seg_count)) = cache.take_bottom_bin(pool_idx, batch) {
                cache.cached_bytes -= bin_size * seg_count as usize;
                self.release_segment(pool_idx, seg_head, cache.owner_slot, cache.owner_gen);
                if cache.cached_bytes <= target {
                    return;
                }
            }
        }
    }

    fn trim_size_class(&self, pool_idx: usize) {
        let pool_mutex = &self.pools[pool_idx];
        let mut guard = pool_mutex
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        // Drain the dense depot: bins parked there are pool-allocated
        // capacity that must be free to decommit. (Lock order pool →
        // depot is safe: the release/refill paths take the depot lock
        // with no pool lock held; only trim holds both.)
        {
            let shard = &self.depots[pool_idx];
            let mut depot = shard
                .inner
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            for slot in 0..usize::from(depot.used) {
                for i in 0..usize::from(depot.counts[slot]) {
                    let p = depot.slots[slot][i];
                    // Safety: depot entries are freed bins the depot
                    // exclusively holds.
                    guard.free(unsafe { NonNull::new_unchecked(p) });
                }
            }
            depot.used = 0;
            shard.hint.store(0, Ordering::Relaxed);
        }

        // Drain all shards of the recycler for this size class in one shot.
        if let Some(chain_head) = self.recycler.drain_all(pool_idx) {
            // Walk the inter-bundle chain, freeing each bundle's objects
            let mut bundle: Option<NonNull<u8>> = Some(chain_head);
            while let Some(bh) = bundle {
                // Read inter-bundle link before freeing objects in this bundle.
                // Safety: drain_all detached the chain — exclusively ours.
                let next_bundle = unsafe {
                    (*GlobalRecycler::recycler_link_atomic_ptr(bh.as_ptr())).load(Ordering::Relaxed)
                };

                // Walk intra-bundle chain (offset 0) and free each object
                let mut node = Some(bh);
                while let Some(n) = node {
                    // Safety: n is a freed bin whose first word holds the
                    // intra-bundle link; the chain is exclusively ours.
                    let next = unsafe { *n.cast::<*mut u8>().as_ptr() };
                    guard.free(n);
                    node = NonNull::new(next);
                }

                bundle = NonNull::new(next_bundle);
            }
        }

        // Sweep pending mask returns into the pools so bins parked
        // behind idle or slow owners can decommit this pass. The
        // swap-based drain races benignly with owner reconciles and
        // refill claims: each bit is taken exactly once.
        for pi in 0..guard.pools.len() {
            let base = guard.pools[pi].base.as_ptr() as usize;
            let mut heads: Vec<NonNull<u8>> = Vec::new();
            self.remote
                .sweep_pool(base, write_link, |head, _tail, _count| heads.push(head));
            for head in heads {
                let mut node = Some(head);
                while let Some(n) = node {
                    // Safety: swept chains are plain-linked.
                    let next = unsafe { *n.cast::<*mut u8>().as_ptr() };
                    guard.pools[pi].free(n);
                    node = NonNull::new(next);
                }
            }
        }

        // Three-phase trim: select+detach under the lock, decommit
        // WITHOUT the lock (allocations on this size class proceed
        // meanwhile), then re-lock to integrate. Detached blocks are
        // invisible to alloc (bit tree) and hold no live bins, so
        // nothing can touch them while the lock is released.
        let mut batch = guard.begin_trim();
        if batch.is_empty() {
            // Nothing to decommit — just shrink trailing metadata.
            guard.finish_trim(&[]);
            return;
        }
        drop(guard);

        for req in &mut batch {
            // Safety: FFI call to decommit memory owned by the pool;
            // the blocks are detached (marked decommitting) so no other
            // thread can allocate from them while unmapped.
            req.ok = unsafe { PlatformVmOps::decommit(req.ptr, req.size) }.is_ok();
        }

        let mut guard = pool_mutex
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.finish_trim(&batch);
    }

    #[cfg(loom)]
    pub(crate) fn trim_size_class_for_loom(&self, pool_idx: usize) {
        self.trim_size_class(pool_idx);
    }

    pub fn trim(&self) {
        for pool_idx in 0..self.pools.len() {
            self.trim_size_class(pool_idx);
        }
        // Also trim the large alloc cache
        self.large_cache.trim();
    }

    /// Size class for a byte request with alignment <= 16 (which every bin
    /// satisfies: classes are multiples of 16 in 16-aligned blocks). One
    /// table load — no alignment loop, no `Layout`.
    #[inline]
    pub(crate) fn size_class_min_align(size: usize) -> usize {
        crate::qen_debug_assert!(size > 0 && size <= MAX_SMALL_SIZE);
        SIZE_CLASS_LUT[(size + 15) >> 4] as usize
    }

    pub(crate) fn size_class(size: usize, align: usize) -> usize {
        if size == 0 {
            crate::qen_debug_assert!(false, "Size 0 not supported by BinnedAllocator");
            // Safety: Unreachable logic.
            unsafe { std::hint::unreachable_unchecked() }
        }
        if size > MAX_SMALL_SIZE {
            crate::qen_debug_assert!(false, "Size {size} too large for size classes");
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

        crate::qen_debug_assert!(
            false,
            "No size class satisfies size {size} and alignment {align}",
        );
        // Safety: Logic ensures all valid inputs are handled by the loop above.
        unsafe { std::hint::unreachable_unchecked() }
    }
}

// 96 size classes: 16B..128B (step 16), then 12.5% geometric spacing up to 256KB
pub(crate) const NUM_SIZE_CLASSES: usize = 96;

/// Fixed-size array form for use in const contexts.
const SIZE_CLASSES_ARRAY: [usize; NUM_SIZE_CLASSES] = [
    16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384,
    416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1152, 1280, 1408, 1536, 1664,
    1792, 1920, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656,
    7168, 7680, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384, 18432, 20480, 22528,
    24576, 26624, 28672, 30720, 32768, 36864, 40960, 45056, 49152, 53248, 57344, 61440, 65536,
    73_728, 81_920, 90_112, 98_304, 106_496, 114_688, 122_880, 131_072, 147_456, 163_840, 180_224,
    196_608, 212_992, 229_376, 245_760, 262_144,
];

pub(crate) const SIZE_CLASSES: &[usize] = &SIZE_CLASSES_ARRAY;

/// Transfer batch per class for cache refills and releases:
/// `clamp(64KiB / bin_size, 2, 32)` — tcmalloc's `num_objects_to_move`
/// policy. Batches amortize recycler/pool crossings without moving
/// unboundedly large chains.
static CLASS_BATCH: [u16; NUM_SIZE_CLASSES] = build_class_batch();

/// Adaptive cache growth ceiling per class: about 256KiB of cached bytes
/// per class, floored at two batches and at the L0 micro-cache capacity
/// (limits below L0 would thrash: the hot L0 array would be drained to the
/// recycler every other free), capped at 8192 objects (tcmalloc's
/// `kMaxDynamicFreeListLength`).
static CLASS_CAP: [u16; NUM_SIZE_CLASSES] = build_class_cap();

const fn build_class_batch() -> [u16; NUM_SIZE_CLASSES] {
    let mut table = [0u16; NUM_SIZE_CLASSES];
    let mut i = 0;
    while i < NUM_SIZE_CLASSES {
        let mut batch = 64 * 1024 / SIZE_CLASSES_ARRAY[i];
        if batch < 2 {
            batch = 2;
        }
        if batch > 32 {
            batch = 32;
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "batch is clamped to 2..=32 above"
        )]
        {
            table[i] = batch as u16;
        }
        i += 1;
    }
    table
}

const fn build_class_cap() -> [u16; NUM_SIZE_CLASSES] {
    let batches = build_class_batch();
    let mut table = [0u16; NUM_SIZE_CLASSES];
    let mut i = 0;
    while i < NUM_SIZE_CLASSES {
        let mut cap = 256 * 1024 / SIZE_CLASSES_ARRAY[i];
        let mut floor = 2 * (batches[i] as usize);
        // Floor at the adaptive-limit floor (`CACHE_FLOOR`): limits below
        // it thrash the transfer boundary.
        if floor < 8 {
            floor = 8;
        }
        if cap < floor {
            cap = floor;
        }
        if cap > 8192 {
            cap = 8192;
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "cap is clamped to 2*batch..=8192 above"
        )]
        {
            table[i] = cap as u16;
        }
        i += 1;
    }
    table
}

// ----------------------------------------------------------------------------
// Reciprocal Division — O(1) division by bin_size via multiply-shift
// ----------------------------------------------------------------------------

/// Precomputed reciprocal multiplier for a size class.
///
/// Replaces runtime variable-divisor division (~20-35 cycles on x86) with a
/// fixed multiply + shift (3-5 cycles).
///
/// For divisor `d`, `div_mult = ceil(2^64 / d)`. Then:
///   `floor(n / d) = ((n as u128 * div_mult as u128) >> 64) as usize`
/// for all `n` up to at least `block_size` (tested exhaustively below).
///
/// For the divisibility test `n % d == 0`:
///   `mod_mult = u64::MAX / d + 1` (Granlund-Montgomery trick)
///   `n.wrapping_mul(mod_mult) < mod_mult` iff `n % d == 0`
#[derive(Clone, Copy)]
pub(crate) struct ReciprocalDiv {
    pub div_mult: u64,
    pub mod_mult: u64,
}

// The reciprocal kernels are width-exact 64/128-bit arithmetic on a
// 64-bit-only crate (enforced by the compile_error in lib.rs); the
// narrowing casts are the intended extraction of the low word, proven
// exhaustively by the const block below.
#[expect(
    clippy::cast_possible_truncation,
    reason = "width-exact reciprocal math; 64-bit-only crate; verified by the const proof below"
)]
impl ReciprocalDiv {
    /// Compute quotient `n / d` via reciprocal multiplication.
    #[expect(
        clippy::inline_always,
        reason = "single multiply-shift on the free hot path; must not become a call"
    )]
    #[inline(always)]
    pub const fn div(self, n: usize) -> usize {
        ((n as u128 * self.div_mult as u128) >> 64) as usize
    }

    /// Test `n % d == 0` via Granlund-Montgomery divisibility.
    // Referenced only from debug/hardened assertions, so ordinary release
    // builds see it dead.
    #[allow(dead_code)]
    #[expect(
        clippy::inline_always,
        reason = "single multiply-compare used in hot-path debug/hardened assertions"
    )]
    #[inline(always)]
    pub const fn is_multiple(self, n: usize) -> bool {
        (n as u64).wrapping_mul(self.mod_mult) < self.mod_mult
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "width-exact reciprocal math; 64-bit-only crate; verified by the const proof below"
)]
pub(crate) const fn compute_reciprocal(d: usize) -> ReciprocalDiv {
    // div_mult = ceil(2^64 / d) = (2^64 - 1) / d + 1
    let div_mult = (u64::MAX as u128 / d as u128 + 1) as u64;
    // mod_mult = floor(u64::MAX / d) + 1
    let mod_mult = u64::MAX / d as u64 + 1;
    ReciprocalDiv { div_mult, mod_mult }
}

/// Compile-time exhaustive correctness proof for reciprocal division.
/// Verifies `reciprocal_div(n, d) == n / d` for all dividends up to
/// `max_block_size` and `reciprocal_is_multiple(n, d) == (n % d == 0)`.
#[allow(long_running_const_eval)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "mirrors the width-exact reciprocal kernel it proves correct"
)]
const _: () = {
    const MAX_BLOCK: usize = 1_048_576; // 1 MB — upper bound for block_size
    const CLASSES: [usize; NUM_SIZE_CLASSES] = SIZE_CLASSES_ARRAY;

    let mut sc = 0;
    while sc < NUM_SIZE_CLASSES {
        let d = CLASSES[sc];
        let r = compute_reciprocal(d);
        // Check all multiples of d up to MAX_BLOCK (division correctness)
        let mut n = 0;
        while n <= MAX_BLOCK {
            let expected = n / d;
            let got = ((n as u128 * r.div_mult as u128) >> 64) as usize;
            assert!(expected == got, "reciprocal division mismatch");
            n += d;
        }
        // Check divisibility at every boundary in range: k*d - 1, k*d,
        // k*d + 1 for all multiples up to MAX_BLOCK (the only places the
        // Lemire divisibility test can flip). Checking boundaries across
        // the whole range is both stronger and far cheaper in const eval
        // than exhaustively scanning small n.
        let mut k = 0;
        while k * d <= MAX_BLOCK {
            let base = k * d;
            let mut off = 0;
            while off < 3 {
                // Wrapping at 0 - 1 is fine: u64::MAX is not a multiple.
                let n = (base + off).wrapping_sub(1);
                let expected = n.is_multiple_of(d);
                let got = (n as u64).wrapping_mul(r.mod_mult) < r.mod_mult;
                assert!(expected == got, "reciprocal divisibility mismatch");
                off += 1;
            }
            k += 1;
        }
        sc += 1;
    }
};

/// O(1) size-to-class lookup table. Index by `ceil(size / 16)`.
/// Table has 16385 entries covering sizes 1..262144 in 16-byte quanta.
/// Each entry is the size class index (0..95).
static SIZE_CLASS_LUT: [u8; MAX_SMALL_SIZE / 16 + 1] = build_size_class_lut();

#[expect(
    clippy::large_stack_arrays,
    reason = "const-evaluated only; the table lives in static memory, not on any runtime stack"
)]
const fn build_size_class_lut() -> [u8; MAX_SMALL_SIZE / 16 + 1] {
    const CLASSES: [usize; NUM_SIZE_CLASSES] = SIZE_CLASSES_ARRAY;
    let mut table = [0u8; MAX_SMALL_SIZE / 16 + 1];
    // table[0] unused (size 0 is invalid)
    let mut q: usize = 1;
    let mut sc: u8 = 0;
    while (sc as usize) < NUM_SIZE_CLASSES {
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
// Per-Size-Class Statistics (feature = "stats")
// ----------------------------------------------------------------------------

/// Per-size-class diagnostic counters. Gated behind `--features stats`.
/// All counters are `Relaxed`-ordered, same as the global `stats::Counter`.
#[cfg(feature = "stats")]
#[allow(dead_code)]
pub mod class_stats {
    use super::NUM_SIZE_CLASSES;
    use crate::sync::atomic::{AtomicU64, Ordering};

    struct ClassCounters {
        allocs: AtomicU64,
        frees: AtomicU64,
        cache_hits: AtomicU64,
        cache_misses: AtomicU64,
    }

    impl ClassCounters {
        #[cfg(not(loom))]
        const fn new() -> Self {
            Self {
                allocs: AtomicU64::new(0),
                frees: AtomicU64::new(0),
                cache_hits: AtomicU64::new(0),
                cache_misses: AtomicU64::new(0),
            }
        }

        // loom atomics are not const-constructible; COUNTERS uses
        // loom::lazy_static! below, which initialises at runtime (and
        // re-creates the value for each model run).
        #[cfg(loom)]
        fn new() -> Self {
            Self {
                allocs: AtomicU64::new(0),
                frees: AtomicU64::new(0),
                cache_hits: AtomicU64::new(0),
                cache_misses: AtomicU64::new(0),
            }
        }
    }

    /// Snapshot of a single size class's counters.
    #[derive(Debug, Clone, Copy)]
    pub struct ClassSnapshot {
        pub size_class: usize,
        pub allocs: u64,
        pub frees: u64,
        pub cache_hits: u64,
        pub cache_misses: u64,
    }

    // One set of counters per size class.
    #[cfg(not(loom))]
    static COUNTERS: [ClassCounters; NUM_SIZE_CLASSES] = {
        // The interior-mutable const is the standard idiom for initializing
        // an array of atomics; INIT is only ever used as an initializer.
        #[allow(clippy::declare_interior_mutable_const)]
        const INIT: ClassCounters = ClassCounters::new();
        [INIT; NUM_SIZE_CLASSES]
    };

    #[cfg(loom)]
    loom::lazy_static! {
        static ref COUNTERS: [ClassCounters; NUM_SIZE_CLASSES] =
            std::array::from_fn(|_| ClassCounters::new());
    }

    #[inline]
    pub fn record_alloc(class_idx: usize) {
        if let Some(c) = COUNTERS.get(class_idx) {
            c.allocs.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[inline]
    pub fn record_free(class_idx: usize) {
        if let Some(c) = COUNTERS.get(class_idx) {
            c.frees.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[inline]
    pub fn record_cache_hit(class_idx: usize) {
        if let Some(c) = COUNTERS.get(class_idx) {
            c.cache_hits.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[inline]
    pub fn record_cache_miss(class_idx: usize) {
        if let Some(c) = COUNTERS.get(class_idx) {
            c.cache_misses.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Snapshot all size classes. Returns array of `ClassSnapshot`.
    pub fn snapshot() -> Vec<ClassSnapshot> {
        let classes = super::SIZE_CLASSES;
        COUNTERS
            .iter()
            .enumerate()
            .map(|(i, c)| ClassSnapshot {
                size_class: classes[i],
                allocs: c.allocs.load(Ordering::Relaxed),
                frees: c.frees.load(Ordering::Relaxed),
                cache_hits: c.cache_hits.load(Ordering::Relaxed),
                cache_misses: c.cache_misses.load(Ordering::Relaxed),
            })
            .collect()
    }

    /// Reset all counters to zero.
    pub fn reset() {
        // .iter() (not &COUNTERS): under loom, COUNTERS is a lazy_static
        // wrapper that only exposes iteration through Deref.
        #[allow(clippy::explicit_iter_loop)]
        for c in COUNTERS.iter() {
            c.allocs.store(0, Ordering::Relaxed);
            c.frees.store(0, Ordering::Relaxed);
            c.cache_hits.store(0, Ordering::Relaxed);
            c.cache_misses.store(0, Ordering::Relaxed);
        }
    }
}

// ----------------------------------------------------------------------------
// Thread Cache — dense per-class pointer stacks (front-end rebuild C2)
// ----------------------------------------------------------------------------

/// Floor for adaptive per-class limits (replaces the old L0 micro-cache
/// capacity in that role: limits below it thrash the transfer boundary).
const CACHE_FLOOR: u32 = 8;

/// Per-class slot capacities in the thread-cache slab: `CLASS_CAP + 1`
/// (the free path pushes before its overflow check, so the stack may
/// transiently hold `max_length + 1 <= CLASS_CAP + 1` entries).
const fn build_slab_offsets() -> [u32; NUM_SIZE_CLASSES + 1] {
    let caps = build_class_cap();
    let mut table = [0u32; NUM_SIZE_CLASSES + 1];
    let mut i = 0;
    while i < NUM_SIZE_CLASSES {
        table[i + 1] = table[i] + caps[i] as u32 + 1;
        i += 1;
    }
    table
}

/// Prefix-sum offsets of each class's slot region in the slab;
/// `SLAB_OFFSETS[NUM_SIZE_CLASSES]` is the total slot count.
static SLAB_OFFSETS: [u32; NUM_SIZE_CLASSES + 1] = build_slab_offsets();

/// Per-class free stack. The pointer storage lives in the owning
/// `ThreadCache`'s slab (`slots` parameters below); this struct is only
/// the bookkeeping, so the whole cache stays const-constructible for
/// const-initialized TLS.
///
/// Replaces the old L0-array + intrusive-linked-list design: push/pop are
/// one indexed store/load in allocator-owned memory and the allocator
/// **never touches the object's own cache lines on the fast path** — no
/// link words, no dependent-load pop chain, no hardened XOR needed in
/// this tier (chains crossing the recycler/transfer boundary remain the
/// wire format there).
pub(crate) struct LocalFreeList {
    /// Stack depth (index of the next free slot).
    top: u32,
    /// Adaptive length limit: grows by one batch per refill (slow start)
    /// up to `CLASS_CAP`. Set from config at bind time.
    pub max_length: u32,
    /// Minimum depth observed since the last scavenge/reset — slots below
    /// it were never touched in the interval, so scavenging releases
    /// `low_water / 2` of them (tcmalloc's policy).
    low_water: u32,
}

impl LocalFreeList {
    pub const fn new_const() -> Self {
        Self {
            top: 0,
            max_length: 0,
            low_water: 0,
        }
    }

    #[inline]
    pub fn count(&self) -> u32 {
        self.top
    }

    /// # Safety
    /// `slots` must point at this class's region in the owning cache's
    /// slab with capacity `CLASS_CAP + 1`, and `self.max_length` must not
    /// exceed `CLASS_CAP` (enforced at bind/growth), so the post-push
    /// overflow check bounds `top`.
    #[inline]
    pub unsafe fn push(&mut self, slots: *mut *mut u8, ptr: NonNull<u8>) {
        // Safety: top <= max_length <= CLASS_CAP by the caller's overflow
        // discipline; the slot is in-bounds per the safety contract.
        unsafe { *slots.add(self.top as usize) = ptr.as_ptr() };
        self.top += 1;
    }

    /// # Safety
    /// `slots` as in [`push`](Self::push).
    #[inline]
    pub unsafe fn pop(&mut self, slots: *mut *mut u8) -> Option<NonNull<u8>> {
        if self.top == 0 {
            return None;
        }
        self.top -= 1;
        // Safety: slots[0..top) were written by push/receive.
        let p = unsafe { *slots.add(self.top as usize) };
        self.note_low_water();
        // Safety: only non-null pointers are ever stored.
        Some(unsafe { NonNull::new_unchecked(p) })
    }

    #[inline]
    fn note_low_water(&mut self) {
        if self.top < self.low_water {
            self.low_water = self.top;
        }
    }

    /// Reset the low-water mark to the current depth (start of a new
    /// scavenge observation interval).
    fn reset_low_water(&mut self) {
        self.low_water = self.top;
    }

    /// Adopt `n` entries a batch refill wrote directly into this class's
    /// slot region starting at the current top (see
    /// `Pool::alloc_batch_array`: the bins land in place, so adoption is
    /// pure bookkeeping).
    #[inline]
    pub fn assume_filled(&mut self, n: u32) {
        self.top += n;
    }

    /// Detach up to `n` of the HOTTEST entries (stack top) as a
    /// plain-linked, null-terminated chain for the recycler. Used on the
    /// free-side overflow path where shifting the array is too expensive;
    /// the released bins are the most recently freed.
    ///
    /// # Safety
    /// `slots` as in [`push`](Self::push).
    pub unsafe fn take_top(&mut self, slots: *mut *mut u8, n: u32) -> Option<(NonNull<u8>, u32)> {
        let take = n.min(self.top);
        if take == 0 {
            return None;
        }
        let mut prev = std::ptr::null_mut();
        for _ in 0..take {
            self.top -= 1;
            // Safety: in-bounds reads of written slots; the bins are
            // freed memory we exclusively hold — writing the link word
            // here is the accepted transfer-boundary touch.
            unsafe {
                let p = *slots.add(self.top as usize);
                *NonNull::new_unchecked(p).cast::<*mut u8>().as_ptr() = prev;
                prev = p;
            }
        }
        self.note_low_water();
        // Safety: take >= 1, so prev is a stored non-null pointer.
        Some((unsafe { NonNull::new_unchecked(prev) }, take))
    }

    /// Detach up to `n` of the COLDEST entries (stack bottom) as a chain,
    /// shifting the survivors down. Used by the scavenger, where "release
    /// what the interval never touched" is the whole policy; the memmove
    /// is bounded by the retained depth and the path is cold.
    ///
    /// # Safety
    /// `slots` as in [`push`](Self::push).
    pub unsafe fn take_bottom(
        &mut self,
        slots: *mut *mut u8,
        n: u32,
    ) -> Option<(NonNull<u8>, u32)> {
        let take = n.min(self.top);
        if take == 0 {
            return None;
        }
        let mut prev = std::ptr::null_mut();
        for i in 0..take as usize {
            // Safety: in-bounds; boundary link write as in take_top.
            unsafe {
                let p = *slots.add(i);
                *NonNull::new_unchecked(p).cast::<*mut u8>().as_ptr() = prev;
                prev = p;
            }
        }
        let remaining = (self.top - take) as usize;
        // Safety: shift survivors down; both ranges in-bounds.
        unsafe {
            std::ptr::copy(slots.add(take as usize), slots, remaining);
        }
        self.top -= take;
        self.note_low_water();
        // Safety: take >= 1.
        Some((unsafe { NonNull::new_unchecked(prev) }, take))
    }

    /// Adopt a pool-built, plain-linked chain of exactly `count` items.
    /// Entries beyond `cap` are NOT adopted: the untaken remainder is
    /// returned as `(head, count)` for the caller to route to the
    /// recycler (mask reconciles can deliver more than a class's region
    /// holds — the old intrusive list had no capacity to respect).
    ///
    /// # Safety
    /// `slots`/`cap` must describe this class's slab region; `head` must
    /// be a plain-linked, null-terminated chain of `count` freed bins.
    pub unsafe fn receive_chain(
        &mut self,
        slots: *mut *mut u8,
        cap: u32,
        head: NonNull<u8>,
        count: u32,
    ) -> Option<(NonNull<u8>, u32)> {
        let room = cap.saturating_sub(self.top);
        let take = count.min(room);
        let mut node = Some(head);
        for _ in 0..take {
            let n = node.expect("chain shorter than its stated count");
            // Safety: chain nodes carry a plain link in the first word;
            // the slot write is in-bounds (take <= room).
            unsafe {
                let next = *n.cast::<*mut u8>().as_ptr();
                *slots.add(self.top as usize) = n.as_ptr();
                node = NonNull::new(next);
            }
            self.top += 1;
        }
        node.map(|spill| (spill, count - take))
    }

    /// Adopt a plain-linked chain of UNKNOWN length (recycler bundles),
    /// walking to count. Returns `(received, spill)`.
    ///
    /// # Safety
    /// As [`receive_chain`](Self::receive_chain), minus the count claim.
    pub unsafe fn receive_walk(
        &mut self,
        slots: *mut *mut u8,
        cap: u32,
        head: NonNull<u8>,
    ) -> (u32, Option<(NonNull<u8>, u32)>) {
        let mut received = 0u32;
        let mut node = Some(head);
        while let Some(n) = node {
            if self.top >= cap {
                // Count the spill so the caller can account it.
                let mut spill_len = 0u32;
                let mut cur = Some(n);
                while let Some(c) = cur {
                    spill_len += 1;
                    // Safety: plain chain link.
                    cur = NonNull::new(unsafe { *c.cast::<*mut u8>().as_ptr() });
                }
                return (received, Some((n, spill_len)));
            }
            // Safety: plain chain link; in-bounds slot write.
            unsafe {
                let next = *n.cast::<*mut u8>().as_ptr();
                *slots.add(self.top as usize) = n.as_ptr();
                node = NonNull::new(next);
            }
            self.top += 1;
            received += 1;
        }
        (received, None)
    }
}

pub(crate) struct ThreadCache {
    // Per-class stack bookkeeping. A fixed inline array (not a Vec): bin
    // state access on the hot path is a direct offset from the cache, and
    // the whole struct is const-constructible for const-initialized TLS.
    bins: [LocalFreeList; NUM_SIZE_CLASSES],
    /// One heap allocation holding every class's pointer slots at
    /// `SLAB_OFFSETS` (null until `bind`; lazily paged).
    slab: *mut *mut u8,
    /// Precomputed `slab + SLAB_OFFSETS[c]` per class, so the hot path's
    /// slot access is one load + index instead of an offset-table load
    /// plus pointer arithmetic (measured ~1 ns/pair).
    slot_bases: [*mut *mut u8; NUM_SIZE_CLASSES],
    /// Total bytes currently cached across all classes (budget accounting).
    cached_bytes: usize,
    // Optional reference to the allocator that owns this cache.
    // Must be 'static to ensure it outlives the thread.
    allocator: Option<&'static BinnedAllocator>,
    /// Owner-registry slot for the mask channel (`u16::MAX` = none: all
    /// remote traffic falls back to the recycler).
    owner_slot: u16,
    owner_gen: u32,
}

// Safety: ThreadCache is only accessed by the owning thread (via TLS). Flushing
// is cooperative — trim() bumps an epoch, and each thread flushes on its
// next alloc/free. The content (pointers) can be sent between threads via flush.
unsafe impl Send for ThreadCache {}

impl ThreadCache {
    pub const fn new_const() -> Self {
        const INIT: LocalFreeList = LocalFreeList::new_const();
        Self {
            bins: [INIT; NUM_SIZE_CLASSES],
            slab: std::ptr::null_mut(),
            slot_bases: [std::ptr::null_mut(); NUM_SIZE_CLASSES],
            cached_bytes: 0,
            allocator: None,
            owner_slot: u16::MAX,
            owner_gen: 0,
        }
    }

    // Used by tests and loom builds; the lib target uses new_const.
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self::new_const()
    }

    /// Layout of the slab: every class's slot region back to back.
    fn slab_layout() -> std::alloc::Layout {
        std::alloc::Layout::array::<*mut u8>(SLAB_OFFSETS[NUM_SIZE_CLASSES] as usize)
            .expect("slab layout")
    }

    /// Slot region for class `idx`. Caller must know the cache is bound
    /// (slab non-null); depth-0 fast outs never reach here unbound.
    #[inline]
    fn slots(&self, idx: usize) -> *mut *mut u8 {
        crate::qen_debug_assert!(!self.slab.is_null(), "slot access on unbound cache");
        self.slot_bases[idx]
    }

    #[inline]
    fn pop_bin(&mut self, idx: usize) -> Option<NonNull<u8>> {
        if self.bins[idx].count() == 0 {
            // Also the unbound case: an empty stack never touches the slab.
            return None;
        }
        let slots = self.slots(idx);
        // Safety: slots describes class idx's region; depth > 0 implies bound.
        unsafe { self.bins[idx].pop(slots) }
    }

    /// Push onto a bound cache (all free paths bind before caching).
    #[inline]
    fn push_bin(&mut self, idx: usize, ptr: NonNull<u8>) {
        let slots = self.slots(idx);
        // Safety: class region; the caller's overflow discipline (release
        // a batch whenever count exceeds max_length) bounds the depth.
        unsafe { self.bins[idx].push(slots, ptr) }
    }

    #[inline]
    fn take_top_bin(&mut self, idx: usize, n: u32) -> Option<(NonNull<u8>, u32)> {
        if self.bins[idx].count() == 0 {
            return None;
        }
        let slots = self.slots(idx);
        // Safety: class region; depth > 0 implies bound.
        unsafe { self.bins[idx].take_top(slots, n) }
    }

    #[inline]
    fn take_bottom_bin(&mut self, idx: usize, n: u32) -> Option<(NonNull<u8>, u32)> {
        if self.bins[idx].count() == 0 {
            return None;
        }
        let slots = self.slots(idx);
        // Safety: class region; depth > 0 implies bound.
        unsafe { self.bins[idx].take_bottom(slots, n) }
    }

    /// Adopt a counted plain chain; returns the capacity spill (chain the
    /// caller must route to the recycler), if any.
    #[inline]
    fn receive_chain_bin(
        &mut self,
        idx: usize,
        head: NonNull<u8>,
        count: u32,
    ) -> Option<(NonNull<u8>, u32)> {
        let slots = self.slots(idx);
        // Safety: class region of a bound cache (receives happen on the
        // alloc/reconcile paths, after bind); cap leaves the push slot.
        unsafe { self.bins[idx].receive_chain(slots, u32::from(CLASS_CAP[idx]), head, count) }
    }

    /// Adopt an uncounted plain chain (recycler bundle); returns
    /// `(received, spill)`.
    #[inline]
    fn receive_walk_bin(
        &mut self,
        idx: usize,
        head: NonNull<u8>,
    ) -> (u32, Option<(NonNull<u8>, u32)>) {
        let slots = self.slots(idx);
        // Safety: as receive_chain_bin.
        unsafe { self.bins[idx].receive_walk(slots, u32::from(CLASS_CAP[idx]), head) }
    }

    pub fn bind(&mut self, allocator: &'static BinnedAllocator) {
        crate::qen_debug_assert!(
            self.bins.iter().all(|b| b.count() == 0),
            "ThreadCache::bind on a non-empty cache"
        );
        if self.slab.is_null() {
            // One lazily-paged allocation holds every class's slots
            // (~0.5 MiB of VA; physical pages only where depth reaches).
            #[expect(
                clippy::cast_ptr_alignment,
                reason = "Layout::array::<*mut u8> guarantees pointer alignment"
            )]
            {
                // Safety: the layout is valid and non-zero-sized.
                self.slab = unsafe { std::alloc::alloc(Self::slab_layout()) }.cast::<*mut u8>();
            }
            assert!(!self.slab.is_null(), "thread-cache slab allocation failed");
            let slab = self.slab;
            for (base, off) in self.slot_bases.iter_mut().zip(SLAB_OFFSETS.iter()) {
                // Safety: offsets are within the slab allocation.
                *base = unsafe { slab.add(*off as usize) };
            }
        }
        // Seed each class's adaptive limit from the config's initial
        // value, clamped between the growth floor and the class ceiling.
        for (idx, bin) in self.bins.iter_mut().enumerate() {
            let initial = allocator.config.max_cache_for(SIZE_CLASSES[idx]);
            bin.max_length = initial.clamp(CACHE_FLOOR, u32::from(CLASS_CAP[idx]));
        }
        // Claim an owner-registry slot so blocks this cache refills from
        // can route remote frees back here. Gated by config (see the
        // `remote_mask_channel` doc): with no slot claimed, every mask
        // path degenerates to a no-op and traffic takes the recycler.
        // Registry exhaustion is not an error either: recycler-only.
        if allocator.config.remote_mask_channel
            && self.owner_slot == u16::MAX
            && let Some((slot, generation)) = allocator.remote.claim_slot()
        {
            self.owner_slot = slot;
            self.owner_gen = generation;
        }
        self.allocator = Some(allocator);
    }

    /// Drain pending mask returns for blocks this cache owns into `bins`
    /// so a subsequent flush hands them to the pool rather than stranding
    /// them behind a flushed (but still live) owner.
    fn reconcile_remote(&mut self) {
        if self.owner_slot == u16::MAX {
            return;
        }
        let Some(allocator) = self.allocator else {
            return;
        };
        let (slot, generation) = (self.owner_slot, self.owner_gen);
        let this = &mut *self;
        allocator
            .remote
            .reconcile(slot, generation, write_link, |class, head, _tail, count| {
                let c = class as usize;
                let spill = this.receive_chain_bin(c, head, count);
                let adopted = count - spill.as_ref().map_or(0, |s| s.1);
                this.cached_bytes += SIZE_CLASSES[c] * adopted as usize;
                if let Some((sh, _)) = spill {
                    allocator.release_segment(c, sh, slot, generation);
                }
            });
    }

    pub fn flush(&mut self) {
        self.reconcile_remote();
        let Some(allocator) = self.allocator else {
            return;
        };
        for idx in 0..NUM_SIZE_CLASSES {
            if self.bins[idx].count() > 0 {
                // Recover from poisoned mutex to avoid leaking pointers (P8)
                let mut guard = allocator.pools[idx]
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);

                while let Some(ptr) = self.pop_bin(idx) {
                    guard.free(ptr);
                }
            }
            self.bins[idx].reset_low_water();
        }
        self.cached_bytes = 0;
    }

    /// Full teardown at thread death: drain everything, release the owner
    /// slot, and free the slab. Late operations (other TLS destructors
    /// that allocate) find an unbound cache: frees take the direct pool
    /// path; allocs rebind a fresh slab that then leaks at process exit —
    /// never a dangling slab access.
    pub fn teardown(&mut self) {
        self.flush();
        self.release_owner_slot();
        self.allocator = None;
        if !self.slab.is_null() {
            // Safety: allocated in `bind` with `slab_layout()`; all slots
            // are logically empty after flush.
            unsafe { std::alloc::dealloc(self.slab.cast::<u8>(), Self::slab_layout()) };
            self.slab = std::ptr::null_mut();
            self.slot_bases = [std::ptr::null_mut(); NUM_SIZE_CLASSES];
        }
    }

    /// Release the owner slot (thread death / cache teardown). Late
    /// publishes that raced the release are drained to the recycler;
    /// anything later still is caught by the trim sweep.
    fn release_owner_slot(&mut self) {
        if self.owner_slot == u16::MAX {
            return;
        }
        let Some(allocator) = self.allocator else {
            self.owner_slot = u16::MAX;
            return;
        };
        let (slot, generation) = (self.owner_slot, self.owner_gen);
        self.owner_slot = u16::MAX;
        allocator.remote.release_slot(
            slot,
            generation,
            write_link,
            |class, head, _tail, _count| {
                let c = class as usize;
                if let Some(rejected) = allocator.recycler.push(c, head) {
                    let mut guard = allocator.pools[c]
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    let mut node = Some(rejected);
                    while let Some(n) = node {
                        // Safety: orphan chains are plain-linked.
                        let next = unsafe { *n.cast::<*mut u8>().as_ptr() };
                        guard.free(n);
                        node = NonNull::new(next);
                    }
                }
            },
        );
    }
}

impl Drop for ThreadCache {
    fn drop(&mut self) {
        // Full teardown: drain to pools (recovering poisoned mutexes to
        // avoid leaks, P8), release the mask-channel owner slot, free the
        // slab.
        self.teardown();
    }
}

#[cfg(all(test, not(loom)))]
// Tests routinely narrow small, obviously-bounded values (sizes, loop
// indices) where conversion ceremony would only obscure the assertions.
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use super::*;
    use crate::sync::Arc;
    use crate::sync::thread;

    /// Send wrapper for raw pointers so cross-thread tests can move them
    /// without laundering through `usize` (which would strip provenance
    /// and blunt miri's pointer tracking).
    struct SendPtr(*mut u8);
    // Safety: test-only; the referenced allocation outlives both threads
    // and access is externally synchronized by the test structure.
    unsafe impl Send for SendPtr {}

    /// Config with immediate decommit (cooldown=0) for tests that assert
    /// blocks are decommitted after a single `process_pending_decommits()`.
    fn config_immediate_decommit() -> BinnedAllocatorConfig {
        BinnedAllocatorConfig {
            decommit_cooldown: 0,
            ..BinnedAllocatorConfig::default()
        }
    }

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
        unsafe {
            allocator.free_bytes(ptr1, 16);
        }
        // Safety: Test code.
        unsafe {
            allocator.free_bytes(ptr2, 32);
        }
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
    #[cfg_attr(miri, ignore = "native stress test; focused paths run under Miri")]
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
                let sizes = [16, 64, 256, 1024, 4096, 16384, 65536, 262_144];

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
                    unsafe {
                        alloc.free_bytes(ptr, size);
                    }
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
            cache.bind(std::mem::transmute::<
                &binned::BinnedAllocator,
                &binned::BinnedAllocator,
            >(&allocator));
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
    fn test_global_instance() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Initialize if not already (might fail if run in parallel with other tests that init, so ignore result)
        drop(GlobalBinnedAllocator::init());

        let ptr = GlobalBinnedAllocator::alloc_bytes(128).unwrap();
        // Safety: Test code.
        unsafe { ptr.as_ptr().write(0xDD) };

        // Safety: Test code.
        unsafe {
            GlobalBinnedAllocator::free_bytes(ptr, 128);
        }
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
        // bin 0 (size 16); an unbound cache never touches the null slab.
        assert!(cache.pop_bin(0).is_none());
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
            cache.bind(std::mem::transmute::<
                &binned::BinnedAllocator,
                &binned::BinnedAllocator,
            >(&allocator));
        }

        let size_a = 16;
        let size_b = 32;
        let layout_a = std::alloc::Layout::from_size_align(size_a, 1).unwrap();
        let layout_b = std::alloc::Layout::from_size_align(size_b, 1).unwrap();

        let p_a = allocator.alloc_with_cache(&mut cache, layout_a).unwrap();
        let p_b = allocator.alloc_with_cache(&mut cache, layout_b).unwrap();

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
        let mut pool =
            Pool::with_config(16, block_size, &BinnedAllocatorConfig::default()).unwrap();

        // Keep an anchor allocated so sparse decommit doesn't fire
        let p1 = pool.alloc().unwrap();
        let p2 = pool.alloc().unwrap();
        let _anchor = pool.alloc().unwrap();

        pool.free(p1);
        pool.free(p2);

        // Also test the stack path manually with the (still committed) ptrs
        let mut stack = LocalFreeList::new_const();
        stack.max_length = 16;
        let mut slots = [std::ptr::null_mut::<u8>(); 16];
        // Safety: local slots buffer with adequate capacity.
        unsafe {
            let sp = slots.as_mut_ptr();
            stack.push(sp, p1);
            stack.push(sp, p2);
            assert_eq!(stack.pop(sp), Some(p2));
            assert_eq!(stack.pop(sp), Some(p1));
        }
    }

    #[cfg(any(debug_assertions, feature = "hardened"))]
    #[test]
    #[should_panic(expected = "smaller than minimum required 16")]
    fn test_pool_bin_size_too_small() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Should panic due to assert bin_size >= size_of::<usize>()
        drop(Pool::with_config(
            4,
            65536,
            &BinnedAllocatorConfig::default(),
        ));
    }

    #[test]
    fn test_pool_alloc_writes_dont_corrupt_freelist() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // P6: Alloc/Write/Free/Realloc
        let mut pool = Pool::with_config(16, 65536, &BinnedAllocatorConfig::default()).unwrap(); // 64KB block size to satisfy BitTree limit

        let p1 = pool.alloc().unwrap();
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
        unsafe {
            alloc.free(ptr1, layout1);
        }

        // Case 2: Size 16, Align 64
        // Must pick size class multiple of 64. Smallest is 64.
        let layout2 = std::alloc::Layout::from_size_align(16, 64).unwrap();
        let ptr2 = alloc.alloc(layout2).unwrap();
        let addr2 = ptr2.as_ptr() as usize;
        assert_eq!(addr2 % 64, 0, "Ptr {ptr2:p} should be 64-byte aligned");

        let idx2 = BinnedAllocator::size_class(16, 64);
        assert_eq!(SIZE_CLASSES[idx2], 64);

        // Safety: Test code.
        unsafe {
            alloc.free(ptr2, layout2);
        }
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

        // 65536 is exactly class 79 (end of the pre-extension table)
        assert_eq!(BinnedAllocator::size_class(65536, 1), 79);

        // MAX_SMALL_SIZE -> last
        let last_idx = SIZE_CLASSES.len() - 1;
        assert_eq!(BinnedAllocator::size_class(MAX_SMALL_SIZE, 1), last_idx);
    }

    #[cfg(any(debug_assertions, feature = "hardened"))]
    #[test]
    #[should_panic(expected = "Size 0 not supported")]
    fn test_size_class_zero() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // A2: size_class(0) — now panics
        BinnedAllocator::size_class(0, 1);
    }

    #[cfg(any(debug_assertions, feature = "hardened"))]
    #[test]
    #[should_panic(expected = "too large")]
    fn test_size_class_too_large() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // A3: size_class(MAX_SMALL_SIZE + 1) panics
        BinnedAllocator::size_class(MAX_SMALL_SIZE + 1, 1);
    }

    #[test]
    fn test_alloc_all_size_classes() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // A4: Alloc+free from every size class
        let alloc = BinnedAllocator::new().unwrap();

        for &size in SIZE_CLASSES {
            let ptr = alloc.alloc_bytes(size).unwrap();
            // Safety: Test code.
            unsafe {
                ptr.as_ptr().write_bytes(0xCC, size);
            }
            // Safety: Test code.
            unsafe {
                alloc.free_bytes(ptr, size);
            }
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
            unsafe {
                alloc.free_bytes(ptr, size);
            }
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
            unsafe {
                alloc.free_bytes(p, 32);
            }
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
            cache.bind(std::mem::transmute::<
                &binned::BinnedAllocator,
                &binned::BinnedAllocator,
            >(&alloc));
        }

        // 1. Alloc small item (size 32) -> refill = CLASS_BATCH for the class
        let layout1 = std::alloc::Layout::from_size_align(32, 1).unwrap();
        let _p1 = alloc.alloc_with_cache(&mut cache, layout1).unwrap();
        let idx1 = BinnedAllocator::size_class(32, 1);
        // Popped 1 of the freshly refilled batch
        assert_eq!(cache.bins[idx1].count(), u32::from(CLASS_BATCH[idx1]) - 1);

        // 2. Alloc large item (size 64KB) -> the byte-tuned batch shrinks
        // with bin size (floor of 2)
        let layout2 = std::alloc::Layout::from_size_align(65536, 1).unwrap();
        let _p2 = alloc.alloc_with_cache(&mut cache, layout2).unwrap();
        let idx2 = BinnedAllocator::size_class(65536, 1);
        assert_eq!(u32::from(CLASS_BATCH[idx2]), 2, "64KB bins move in pairs");
        assert_eq!(cache.bins[idx2].count(), 1); // Popped 1, refilled 2
    }

    #[test]
    fn test_thread_cache_flush() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // A8: Verify overflow release for different sizes.
        // When the cache exceeds its adaptive limit, ONE transfer batch is
        // detached from the linked-list portion and pushed to the
        // GlobalRecycler (or freed to the pool if the recycler is full);
        // the rest stays cached for locality.
        let alloc = BinnedAllocator::new().unwrap();
        let mut cache = ThreadCache::new();
        // Safety: Test code.
        unsafe {
            cache.bind(std::mem::transmute::<
                &binned::BinnedAllocator,
                &binned::BinnedAllocator,
            >(&alloc));
        }

        // 1. Small size (32 bytes) -> initial limit 64
        let size1 = 32;
        let layout1 = std::alloc::Layout::from_size_align(size1, 1).unwrap();
        let idx1 = BinnedAllocator::size_class(size1, 1);
        let limit1 = cache.bins[idx1].max_length;
        let mut ptrs1 = Vec::new();
        for _ in 0..=limit1 {
            ptrs1.push(alloc.alloc_bytes(size1).unwrap());
        }
        for p in ptrs1 {
            alloc.free_with_cache(&mut cache, p, layout1);
        }
        // Crossing the limit at count = limit + 1 released one batch.
        assert_eq!(
            cache.bins[idx1].count(),
            limit1 + 1 - u32::from(CLASS_BATCH[idx1])
        );

        // 2. Large size (64KB) -> initial limit 4
        let size2 = 65536;
        let layout2 = std::alloc::Layout::from_size_align(size2, 1).unwrap();
        let idx2 = BinnedAllocator::size_class(size2, 1);
        let limit2 = cache.bins[idx2].max_length;
        let mut ptrs2 = Vec::new();
        for _ in 0..=limit2 {
            ptrs2.push(alloc.alloc_bytes(size2).unwrap());
        }
        for p in ptrs2 {
            alloc.free_with_cache(&mut cache, p, layout2);
        }
        assert_eq!(
            cache.bins[idx2].count(),
            limit2 + 1 - u32::from(CLASS_BATCH[idx2])
        );
    }

    #[test]
    fn test_thread_cache_cross_thread() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // A9: Alloc on thread A, free on thread B
        let alloc = Arc::new(BinnedAllocator::new().unwrap());

        let alloc2 = alloc.clone();
        let t = thread::spawn(move || SendPtr(alloc2.alloc_bytes(64).unwrap().as_ptr()));

        let ptr = NonNull::new(t.join().unwrap().0).unwrap();

        // Free on this thread (using direct free implies no cache, or use cache)
        // Test asked for "without cache — direct path" for Free?
        // `BinnedAllocator::free` is direct.
        // Safety: Test code.
        unsafe {
            alloc.free_bytes(ptr, 64);
        }
    }

    #[cfg(any(debug_assertions, feature = "hardened"))]
    #[test]
    #[should_panic(expected = "bound to a different allocator")]
    fn test_thread_cache_mismatch() {
        // Reclaims the 'static-laundered allocator during the should_panic
        // unwind so this test does not trip miri's leak checker. Bound
        // BEFORE `cache` so that `cache` (whose Drop flushes to the
        // allocator) drops first during unwind.
        struct Reclaim(*mut BinnedAllocator);
        impl Drop for Reclaim {
            fn drop(&mut self) {
                // Safety: pointer came from Box::into_raw below; every
                // borrow (the ThreadCache binding) is dropped before this.
                unsafe { drop(Box::from_raw(self.0)) };
            }
        }

        let _guard = crate::memory::TEST_MUTEX.read().unwrap();

        let alloc1_ptr = Box::into_raw(Box::new(BinnedAllocator::new().unwrap()));
        let _reclaim = Reclaim(alloc1_ptr);
        // Safety: alloc1_ptr stays valid until _reclaim drops, which is
        // after every use of this reference.
        let alloc1: &'static BinnedAllocator = unsafe { &*alloc1_ptr };
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
                    unsafe {
                        *ptr.as_ptr() = 1;
                    }
                    // Safety: Test code.
                    unsafe {
                        GlobalBinnedAllocator::free_bytes(ptr, 32);
                    }
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
        // Ensure the process-wide Qen instance is initialized.
        drop(GlobalBinnedAllocator::init());
        let allocator = GlobalBinnedAllocator::get();
        let size = 128; // Use 128 to avoid contention with other tests using 32/48
        let pool_idx = BinnedAllocator::size_class(size, 1);

        // 0. Pre-trim to ensure baseline is clean from previous tests in this thread
        GlobalBinnedAllocator::trim();

        // 1. Get initial free count in pool
        let initial_free = {
            let guard = allocator.pools[pool_idx].lock().unwrap();
            guard
                .pools
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
            unsafe {
                GlobalBinnedAllocator::free_bytes(ptr, size);
            }
            // ptr (and others) are now in this thread's cache
        });
        handle.join().unwrap();

        // 3. Check free count again.
        // If Drop worked, all items (including refill batch) should be back in the pool.
        let final_free = {
            let guard = allocator.pools[pool_idx].lock().unwrap();
            guard
                .pools
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

    #[cfg(any(debug_assertions, feature = "hardened"))]
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
        let config = BinnedAllocatorConfig::default();
        let cooldown = usize::from(config.decommit_cooldown);
        let mut pool = Pool::with_config(16, block_size, &config).unwrap();

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
        for &ptr in &ptrs[bins_per_block..bins_per_block * 2] {
            pool.free(ptr);
        }
        // Block 1 is still committed (decommit deferred)
        assert_eq!(pool.committed, block_size * 2);

        // The cooldown applies to trailing blocks like any other: the block
        // must survive `decommit_cooldown` trim passes untouched.
        for pass in 0..cooldown {
            pool.trim();
            assert_eq!(
                pool.committed,
                block_size * 2,
                "block decommitted on pass {pass}, before its cooldown expired"
            );
            assert_eq!(pool.blocks.len(), 2);
        }

        // Cooldown expired: this pass decommits and pops the trailing block.
        pool.trim();
        assert_eq!(pool.blocks.len(), 1);
        assert_eq!(pool.committed, block_size);

        // Free block 0
        for &ptr in &ptrs[0..bins_per_block] {
            pool.free(ptr);
        }
        // Block 0 still committed (decommit deferred)
        assert_eq!(pool.committed, block_size);

        for _ in 0..cooldown {
            pool.trim();
        }
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
            Pool::with_config(bin_size, block_size, &config_immediate_decommit()).unwrap();

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

    #[cfg(any(debug_assertions, feature = "hardened"))]
    #[cfg(any(debug_assertions, feature = "hardened"))]
    #[test]
    #[should_panic(expected = "does not belong to this Pool")]
    fn test_pool_free_out_of_range() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let mut pool = Pool::with_config(16, 65536, &BinnedAllocatorConfig::default()).unwrap();
        // Safety: Test code.
        let invalid_ptr = unsafe { NonNull::new_unchecked(std::ptr::dangling_mut::<u8>()) };
        pool.free(invalid_ptr);
    }

    #[cfg(any(debug_assertions, feature = "hardened"))]
    #[cfg(any(debug_assertions, feature = "hardened"))]
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

    #[cfg(any(debug_assertions, feature = "hardened"))]
    #[cfg(any(debug_assertions, feature = "hardened"))]
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

    #[cfg(any(debug_assertions, feature = "hardened"))]
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

    #[cfg(any(debug_assertions, feature = "hardened"))]
    #[cfg(any(debug_assertions, feature = "hardened"))]
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
        unsafe {
            alloc.free_bytes(ptr2, 32);
        }

        // 3. Try to free ptr1 (from Pool 0) as 32 bytes (Pool 1)
        // Since ptr1 is in pool 0's VA range, pool 1's range check will catch it.
        // Safety: Test code.
        unsafe {
            alloc.free_bytes(ptr1, 32);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore = "native stress test; focused paths run under Miri")]
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
                    unsafe {
                        GlobalBinnedAllocator::free_bytes(ptr, size);
                    }
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
            guard.pool_map =
                PoolFlatMap::with_shift(guard.config.pool_reserved_size.trailing_zeros());
            let base = pool.base.as_ptr() as usize;
            guard.pools.push(pool);
            guard.pool_map.insert(base, 0);
            guard.active_index = 0;
        }

        // Next alloc should SUCCEED by adding a new pool to the chain
        let result = allocator.alloc_bytes(16);
        assert!(
            result.is_ok(),
            "Allocator should chain new pool when first is exhausted"
        );

        // Verify chain grew
        {
            let guard = allocator.pools[pool_idx].lock().unwrap();
            assert_eq!(guard.pools.len(), 2, "Chain should have 2 pools now");
            assert_eq!(guard.active_index, 1, "Active index should be 1");
        }
    }

    // --- Dense per-class stack correctness under Miri ---

    /// LIFO push/pop over the slot array; count tracks; empty pops None.
    #[test]
    fn test_stack_push_pop_lifo() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let mut stack = LocalFreeList::new_const();
        stack.max_length = 4096;
        let mut slots = [std::ptr::null_mut::<u8>(); 64];
        let sp = slots.as_mut_ptr();
        let mut backing = [[0usize; 2]; 12];
        let ptrs: Vec<NonNull<u8>> = backing
            .iter_mut()
            .map(|b| NonNull::new(b.as_mut_ptr().cast::<u8>()).unwrap())
            .collect();

        // Safety: sp/backing outlive the stack ops; slots capacity 64.
        unsafe {
            assert!(stack.pop(sp).is_none());
            for &p in &ptrs {
                stack.push(sp, p);
            }
            assert_eq!(stack.count(), 12);
            for i in (0..12).rev() {
                assert_eq!(stack.pop(sp).unwrap().as_ptr(), ptrs[i].as_ptr());
            }
            assert!(stack.pop(sp).is_none());
            assert_eq!(stack.count(), 0);
        }
        // The stack never wrote into the bins themselves.
        assert!(backing.iter().all(|b| b[0] == 0 && b[1] == 0));
    }

    /// `take_top` detaches the hottest entries as a plain chain in stack
    /// order, leaving colder entries poppable.
    #[test]
    fn test_stack_take_top_order_and_chain() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let mut stack = LocalFreeList::new_const();
        stack.max_length = 4096;
        let mut slots = [std::ptr::null_mut::<u8>(); 64];
        let sp = slots.as_mut_ptr();
        let mut backing = [[0usize; 2]; 12];
        let ptrs: Vec<NonNull<u8>> = backing
            .iter_mut()
            .map(|b| NonNull::new(b.as_mut_ptr().cast::<u8>()).unwrap())
            .collect();
        // Safety: as above.
        unsafe {
            for &p in &ptrs {
                stack.push(sp, p);
            }
            let (head, n) = stack.take_top(sp, 3).unwrap();
            assert_eq!(n, 3);
            let seg: Vec<*mut u8> = walk_chain(head).into_iter().map(NonNull::as_ptr).collect();
            assert_eq!(
                seg,
                [ptrs[9].as_ptr(), ptrs[10].as_ptr(), ptrs[11].as_ptr()]
            );
            assert_eq!(stack.count(), 9);
            assert_eq!(stack.pop(sp).unwrap().as_ptr(), ptrs[8].as_ptr());
            // Over-ask drains what's left.
            let (_, n) = stack.take_top(sp, 99).unwrap();
            assert_eq!(n, 8);
            assert!(stack.take_top(sp, 1).is_none());
        }
    }

    /// `take_bottom` detaches the coldest entries (never touched this
    /// interval) and shifts survivors down intact.
    #[test]
    fn test_stack_take_bottom_cold_end() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let mut stack = LocalFreeList::new_const();
        stack.max_length = 4096;
        let mut slots = [std::ptr::null_mut::<u8>(); 64];
        let sp = slots.as_mut_ptr();
        let mut backing = [[0usize; 2]; 12];
        let ptrs: Vec<NonNull<u8>> = backing
            .iter_mut()
            .map(|b| NonNull::new(b.as_mut_ptr().cast::<u8>()).unwrap())
            .collect();
        // Safety: as above.
        unsafe {
            for &p in &ptrs {
                stack.push(sp, p);
            }
            let (head, n) = stack.take_bottom(sp, 3).unwrap();
            assert_eq!(n, 3);
            let seg: Vec<*mut u8> = walk_chain(head).into_iter().map(NonNull::as_ptr).collect();
            assert_eq!(seg, [ptrs[2].as_ptr(), ptrs[1].as_ptr(), ptrs[0].as_ptr()]);
            assert_eq!(stack.count(), 9);
            // Survivors intact, LIFO from the original top.
            for i in (3..12).rev() {
                assert_eq!(stack.pop(sp).unwrap().as_ptr(), ptrs[i].as_ptr());
            }
            assert!(stack.pop(sp).is_none());
        }
    }

    /// `receive_chain` adopts up to capacity and returns the spill chain
    /// with its residual count.
    #[test]
    fn test_stack_receive_chain_and_spill() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let mut stack = LocalFreeList::new_const();
        stack.max_length = 4096;
        let mut slots = [std::ptr::null_mut::<u8>(); 64];
        let sp = slots.as_mut_ptr();
        let mut backing = [[0usize; 2]; 5];
        let ptrs: Vec<NonNull<u8>> = backing
            .iter_mut()
            .map(|b| NonNull::new(b.as_mut_ptr().cast::<u8>()).unwrap())
            .collect();
        // Build chain ptrs[0] -> 1 -> 2 -> 3 -> 4 -> null.
        // Safety: writing link words into live, exclusively-owned buffers.
        unsafe {
            for i in 0..4 {
                *ptrs[i].cast::<*mut u8>().as_ptr() = ptrs[i + 1].as_ptr();
            }
            *ptrs[4].cast::<*mut u8>().as_ptr() = std::ptr::null_mut();

            let spill = stack.receive_chain(sp, 3, ptrs[0], 5);
            let (sh, sn) = spill.unwrap();
            assert_eq!(sn, 2);
            assert_eq!(sh.as_ptr(), ptrs[3].as_ptr());
            assert_eq!(stack.count(), 3);
            // Adopted in chain order; popped LIFO.
            assert_eq!(stack.pop(sp).unwrap().as_ptr(), ptrs[2].as_ptr());
            assert_eq!(stack.pop(sp).unwrap().as_ptr(), ptrs[1].as_ptr());
            assert_eq!(stack.pop(sp).unwrap().as_ptr(), ptrs[0].as_ptr());

            // Fits-entirely case: no spill.
            for i in 0..2 {
                *ptrs[i].cast::<*mut u8>().as_ptr() = ptrs[i + 1].as_ptr();
            }
            *ptrs[2].cast::<*mut u8>().as_ptr() = std::ptr::null_mut();
            assert!(stack.receive_chain(sp, 3, ptrs[0], 3).is_none());
            assert_eq!(stack.count(), 3);
        }
    }

    /// `receive_walk` adopts an uncounted bundle, reporting both the
    /// adopted and spilled counts.
    #[test]
    fn test_stack_receive_walk_spill() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let mut stack = LocalFreeList::new_const();
        stack.max_length = 4096;
        let mut slots = [std::ptr::null_mut::<u8>(); 64];
        let sp = slots.as_mut_ptr();
        let mut backing = [[0usize; 2]; 4];
        let ptrs: Vec<NonNull<u8>> = backing
            .iter_mut()
            .map(|b| NonNull::new(b.as_mut_ptr().cast::<u8>()).unwrap())
            .collect();
        // Safety: as above.
        unsafe {
            for i in 0..3 {
                *ptrs[i].cast::<*mut u8>().as_ptr() = ptrs[i + 1].as_ptr();
            }
            *ptrs[3].cast::<*mut u8>().as_ptr() = std::ptr::null_mut();

            let (received, spill) = stack.receive_walk(sp, 2, ptrs[0]);
            assert_eq!(received, 2);
            let (sh, sn) = spill.unwrap();
            assert_eq!(sn, 2);
            assert_eq!(sh.as_ptr(), ptrs[2].as_ptr());
            assert_eq!(stack.count(), 2);

            // No spill when it fits.
            let (received, spill) = stack.receive_walk(sp, 64, sh);
            assert_eq!(received, 2);
            assert!(spill.is_none());
            assert_eq!(stack.count(), 4);
        }
    }

    /// Low-water tracks the minimum depth over the observation interval.
    #[test]
    fn test_stack_low_water() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let mut stack = LocalFreeList::new_const();
        stack.max_length = 4096;
        let mut slots = [std::ptr::null_mut::<u8>(); 64];
        let sp = slots.as_mut_ptr();
        let mut backing = [[0usize; 2]; 10];
        let ptrs: Vec<NonNull<u8>> = backing
            .iter_mut()
            .map(|b| NonNull::new(b.as_mut_ptr().cast::<u8>()).unwrap())
            .collect();
        // Safety: as above.
        unsafe {
            for &p in &ptrs {
                stack.push(sp, p);
            }
            stack.reset_low_water();
            assert_eq!(stack.low_water, 10);
            for _ in 0..4 {
                stack.pop(sp);
            }
            assert_eq!(stack.low_water, 6);
            stack.push(sp, ptrs[0]);
            stack.push(sp, ptrs[1]);
            assert_eq!(stack.low_water, 6, "pushes must not raise low water");
        }
    }

    // --- Batch alloc under Miri ---

    /// Walk a plain-linked, null-terminated chain into a Vec.
    fn walk_chain(head: NonNull<u8>) -> Vec<NonNull<u8>> {
        let mut nodes = Vec::new();
        let mut cur = Some(head);
        while let Some(node) = cur {
            nodes.push(node);
            // Safety: chain nodes hold a plain link in the first word.
            let next = unsafe { *node.cast::<*mut u8>().as_ptr() };
            cur = NonNull::new(next);
        }
        nodes
    }

    /// `Pool::alloc_batch_array` fills a pointer array in a single call
    /// without touching the bins themselves.
    #[test]
    fn test_pool_alloc_batch_array_correctness() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let mut pool = Pool::with_config(64, 65536, &BinnedAllocatorConfig::default()).unwrap();

        let mut out = [std::ptr::null_mut::<u8>(); 16];
        // Safety: out holds 16 slots.
        let count = unsafe { pool.alloc_batch_array(out.as_mut_ptr(), 16) }.unwrap();
        assert_eq!(count, 16);

        // All pointers distinct, aligned, non-null.
        let mut addrs: Vec<usize> = out.iter().map(|p| *p as usize).collect();
        addrs.sort_unstable();
        addrs.dedup();
        assert_eq!(addrs.len(), 16, "batch alloc must return distinct pointers");
        for &p in &out {
            assert!(!p.is_null());
            assert!((p as usize).is_multiple_of(64), "aligned to bin_size");
        }

        for p in out {
            pool.free(NonNull::new(p).unwrap());
        }
    }

    /// `Pool::alloc_batch_array` across block boundaries.
    #[test]
    fn test_pool_alloc_batch_array_cross_block() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let bin_size = 4096;
        let block_size = 65536;
        let bins_per_block = block_size / bin_size; // 16
        let mut pool =
            Pool::with_config(bin_size, block_size, &BinnedAllocatorConfig::default()).unwrap();

        let mut out = vec![std::ptr::null_mut::<u8>(); bins_per_block + 4];
        // Safety: out holds bins_per_block + 4 slots.
        let count =
            unsafe { pool.alloc_batch_array(out.as_mut_ptr(), bins_per_block + 4) }.unwrap();
        assert_eq!(count as usize, bins_per_block + 4);
        assert_eq!(pool.blocks.len(), 2, "should span 2 blocks");

        for p in out {
            pool.free(NonNull::new(p).unwrap());
        }
    }

    /// `Pool::alloc_batch_array` serves freelist blocks (recycled bins)
    /// via the per-bin fallback.
    #[test]
    fn test_pool_alloc_batch_array_freelist_fallback() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let mut pool = Pool::with_config(64, 65536, &BinnedAllocatorConfig::default()).unwrap();

        // Put the block on the freelist path: alloc some bins singly, free
        // them (free_head != sentinel), then batch out of the freelist.
        let singles: Vec<NonNull<u8>> = (0..8).map(|_| pool.alloc().unwrap()).collect();
        for &ptr in &singles {
            pool.free(ptr);
        }

        let mut out = [std::ptr::null_mut::<u8>(); 8];
        // Safety: out holds 8 slots.
        let count = unsafe { pool.alloc_batch_array(out.as_mut_ptr(), 8) }.unwrap();
        assert_eq!(count, 8);
        let mut addrs: Vec<usize> = out.iter().map(|p| *p as usize).collect();
        addrs.sort_unstable();
        addrs.dedup();
        assert_eq!(
            addrs.len(),
            8,
            "freelist batch must return distinct pointers"
        );

        for p in out {
            pool.free(NonNull::new(p).unwrap());
        }
    }

    // --- Cooldown lifecycle under Miri ---

    /// Full lifecycle: alloc → free → process multiple times → decommit → recommit.
    #[test]
    fn test_cooldown_full_lifecycle() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let config = BinnedAllocatorConfig {
            decommit_cooldown: 3,
            ..Default::default()
        };
        let block_size = 65536;
        let bin_size = 4096;
        let mut pool = Pool::with_config(bin_size, block_size, &config).unwrap();

        // Fill and free a block
        let mut ptrs = Vec::new();
        for _ in 0..16 {
            ptrs.push(pool.alloc().unwrap());
        }
        // Write to each pointer (Miri checks validity)
        for p in &ptrs {
            // Safety: p is a live allocation returned by alloc.
            unsafe {
                p.as_ptr().write(0xAA);
            }
        }
        for p in &ptrs {
            pool.free(*p);
        }

        // Process 3 times: block stays committed (cooldown 3→2→1→0)
        for _ in 0..3 {
            pool.process_pending_decommits();
            assert!(pool.blocks[0].is_committed());
        }

        // 4th process: block is decommitted (cooldown reached 0 on prior pass)
        pool.process_pending_decommits();
        assert!(!pool.blocks[0].is_committed());

        // Re-allocate — triggers recommit
        let new_ptr = pool.alloc().unwrap();
        assert!(pool.blocks[0].is_committed());
        // Should be zeroed after recommit (debug or hardened mode)
        #[cfg(any(debug_assertions, feature = "hardened"))]
        // Safety: new_ptr is a live allocation from this pool.
        unsafe {
            assert_eq!(new_ptr.as_ptr().read(), 0);
        }
        pool.free(new_ptr);
    }

    // (The hardened XOR link-encoding tests lived here. The dense-stack
    // cache keeps pointers in allocator-owned slot arrays, so freed-bin
    // link words no longer exist in the cache tier for the `hardened`
    // feature to protect; transfer-boundary chains remain plain by the
    // recycler's contract, as before.)

    // --- Failure injection: decommit failure during trim is retried ---
    #[test]
    fn test_pool_trim_decommit_failure_is_retried() {
        // Exclusive: failure injection is global.
        let _guard = crate::memory::TEST_MUTEX.write().unwrap();
        let block_size = 65536;
        let config = BinnedAllocatorConfig {
            decommit_cooldown: 0,
            ..Default::default()
        };
        let mut pool = Pool::with_config(16, block_size, &config).unwrap();
        let bins = block_size / 16;
        let mut ptrs = Vec::new();
        for _ in 0..bins {
            ptrs.push(pool.alloc().unwrap());
        }
        for p in ptrs {
            pool.free(p);
        }

        // First trim: the decommit syscall fails. Trim must not panic, the
        // block must stay committed and tracked, and it must be re-queued.
        crate::memory::vm::failure_injection::fail_next_decommits(1);
        pool.trim();
        crate::memory::vm::failure_injection::reset();
        assert_eq!(
            pool.blocks.len(),
            1,
            "failed decommit must not pop the block"
        );
        assert!(pool.blocks[0].is_committed());
        assert_eq!(pool.committed, block_size);

        // The retry succeeds on the next pass and the block is popped.
        pool.trim();
        assert_eq!(pool.blocks.len(), 0);
        assert_eq!(pool.committed, 0);

        // The pool remains fully usable.
        let p = pool.alloc().unwrap();
        pool.free(p);
    }

    // --- Failure injection: reservation failure propagates cleanly ---
    #[test]
    fn test_binned_reserve_failure_propagates() {
        let _guard = crate::memory::TEST_MUTEX.write().unwrap();
        let alloc = BinnedAllocator::new().unwrap();

        crate::memory::vm::failure_injection::fail_next_reserves(1);
        let result = alloc.alloc_bytes(16);
        crate::memory::vm::failure_injection::reset();
        assert!(
            matches!(result, Err(VmError::ReservationFailed(_))),
            "reserve failure must propagate, got {result:?}"
        );

        // The allocator stays usable once the transient failure clears.
        let p = alloc.alloc_bytes(16).unwrap();
        // Safety: p was allocated with this size just above.
        unsafe { alloc.free_bytes(p, 16) };
    }

    // --- T4: trim success path leaves consistent, reusable state ---
    // (The decommit-FAILURE branch is covered by
    // test_pool_trim_decommit_failure_is_retried, which injects a syscall
    // failure — see vm::failure_injection.)
    #[test]
    fn test_pool_trim_is_best_effort() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let block_size = 65536;
        let config = BinnedAllocatorConfig {
            decommit_cooldown: 0, // trim mechanics under test, not the cooldown
            ..Default::default()
        };
        let mut pool = Pool::with_config(16, block_size, &config).unwrap();
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
    #[cfg_attr(miri, ignore = "exhaustive pure lookup-table equivalence test")]
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
        //
        // The allocator is laundered to 'static via a raw pointer (rather
        // than Box::leak) so it can be reclaimed at the end — miri runs
        // with the leak checker enabled.
        let allocator_ptr = Box::into_raw(Box::new(BinnedAllocator::new().unwrap()));
        // Safety: allocator_ptr stays valid until the Box::from_raw at the
        // end of this test, after every borrow has ended.
        let allocator: &'static BinnedAllocator = unsafe { &*allocator_ptr };

        let pool_idx = BinnedAllocator::size_class(32, 1);

        // Poison the mutex by panicking while holding the lock
        drop(
            thread::spawn(move || {
                let _guard = allocator.pools[pool_idx].lock().unwrap();
                panic!("intentional panic to poison mutex");
            })
            .join(),
        );

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
        cache.push_bin(pool_idx, ptr32);

        // Drop cache — must not crash despite poisoned mutex for bin 32
        drop(cache);

        // Reclaim the allocator: the poisoning thread was joined and the
        // cache is dropped, so no borrow remains.
        // Safety: see above; allocator_ptr came from Box::into_raw.
        unsafe { drop(Box::from_raw(allocator_ptr)) };
    }

    // --- T9: 64+ concurrent threads stress test ---
    #[test]
    #[cfg_attr(miri, ignore = "64-thread native stress test")]
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
                    unsafe {
                        alloc.free_bytes(ptr, size);
                    }
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
        assert!(recycler.push(1, ptr).is_none()); // accepted
        // Pop it back
        let popped = recycler.pop(1, &mut None);
        assert!(popped.is_some());
        assert_eq!(popped.unwrap(), ptr);
        // Pop from empty
        assert!(recycler.pop(1, &mut None).is_none());

        // Safety: Test code.
        unsafe {
            alloc.free_bytes(ptr, 32);
        }
    }

    #[test]
    fn test_recycler_push_until_full() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // With sharding, capacity per shard = ceil(max_bundles / SHARD_COUNT).
        // Total capacity ≥ max_bundles. Use a count that fills one shard.
        let max_bundles = RECYCLER_SHARD_COUNT as u32;
        let recycler = GlobalRecycler::new(max_bundles);
        let alloc = BinnedAllocator::new().unwrap();

        // All pushes from one thread go to the same shard.
        // per-shard limit = ceil(max_bundles / SHARD_COUNT) = 1
        let p1 = alloc.alloc_bytes(32).unwrap();
        let p2 = alloc.alloc_bytes(32).unwrap();

        assert!(recycler.push(0, p1).is_none()); // accepted (count 0 < 1)
        // Second push should be rejected (per-shard limit = 1)
        let rejected = recycler.push(0, p2);
        assert!(rejected.is_some());
        assert_eq!(rejected.unwrap(), p2);

        // Safety: Test code.
        unsafe {
            alloc.free_bytes(p1, 32);
        }
        // Safety: Test code.
        unsafe {
            alloc.free_bytes(p2, 32);
        }
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
        // One byte past the largest size class transparently routes to
        // LargeAllocCache (previously this would panic).
        let size = MAX_SMALL_SIZE + 1;
        let ptr = alloc.alloc_bytes(size).unwrap();
        // Safety: Test code.
        unsafe {
            *ptr.as_ptr() = 0xAA;
        }
        // Safety: Test code.
        unsafe {
            alloc.free_bytes(ptr, size);
        }
    }

    #[test]
    fn test_large_alloc_with_cache() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let alloc = BinnedAllocator::new().unwrap();
        let mut cache = ThreadCache::new();
        // Safety: Test code.
        unsafe {
            cache.bind(std::mem::transmute::<
                &binned::BinnedAllocator,
                &binned::BinnedAllocator,
            >(&alloc));
        }
        let layout = std::alloc::Layout::from_size_align(500_000, 1).unwrap();

        let ptr = alloc.alloc_with_cache(&mut cache, layout).unwrap();
        // Safety: Test code.
        unsafe {
            ptr.as_ptr().write_bytes(0xBB, 500_000);
        }
        alloc.free_with_cache(&mut cache, ptr, layout);
    }

    #[test]
    fn test_mixed_small_large_alloc() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let alloc = BinnedAllocator::new().unwrap();
        let small = alloc.alloc_bytes(64).unwrap();
        let large = alloc.alloc_bytes(400_000).unwrap();

        // Safety: Test code.
        unsafe {
            *small.as_ptr() = 1;
            *large.as_ptr() = 2;
        }

        // Safety: Test code.
        unsafe {
            alloc.free_bytes(small, 64);
        }
        // Safety: Test code.
        unsafe {
            alloc.free_bytes(large, 400_000);
        }
    }

    // --- Canary tests (debug and hardened builds only) ---

    #[cfg(any(debug_assertions, feature = "hardened"))]
    #[test]
    fn test_block_canary_checked() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Verify canary is set on new blocks (they don't panic on access)
        let mut pool = Pool::with_config(16, 65536, &BinnedAllocatorConfig::default()).unwrap();
        let ptr = pool.alloc().unwrap();
        pool.blocks[0].check_canary(); // should not panic
        pool.free(ptr);
    }

    #[cfg(any(debug_assertions, feature = "hardened"))]
    #[cfg(any(debug_assertions, feature = "hardened"))]
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
            Pool::with_config(bin_size, block_size, &config_immediate_decommit()).unwrap();

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
    fn test_decommit_cooldown_delays_decommit() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let config = BinnedAllocatorConfig {
            decommit_cooldown: 2,
            ..Default::default()
        };
        let block_size = 65536;
        let bin_size = 4096; // 16 bins per block
        let mut pool = Pool::with_config(bin_size, block_size, &config).unwrap();

        let mut ptrs = Vec::new();
        for _ in 0..16 {
            ptrs.push(pool.alloc().unwrap());
        }
        // Free all — queues block 0 with cooldown=2
        for p in &ptrs {
            pool.free(*p);
        }
        assert!(pool.blocks[0].is_committed());

        // First process: cooldown 2→1, still committed
        pool.process_pending_decommits();
        assert!(pool.blocks[0].is_committed());

        // Second process: cooldown 1→0, still committed (moved to ready next pass)
        pool.process_pending_decommits();
        assert!(pool.blocks[0].is_committed());

        // Third process: cooldown reached 0, now decommits
        pool.process_pending_decommits();
        assert!(!pool.blocks[0].is_committed());
    }

    #[test]
    fn test_decommit_cooldown_realloc_cancels() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let config = BinnedAllocatorConfig {
            decommit_cooldown: 2,
            ..Default::default()
        };
        let block_size = 65536;
        let bin_size = 4096;
        let mut pool = Pool::with_config(bin_size, block_size, &config).unwrap();

        let mut ptrs = Vec::new();
        for _ in 0..16 {
            ptrs.push(pool.alloc().unwrap());
        }
        for p in &ptrs {
            pool.free(*p);
        }

        // Tick once (cooldown 2→1)
        pool.process_pending_decommits();
        assert!(pool.blocks[0].is_committed());

        // Re-alloc into the block — it's no longer fully empty
        let _new = pool.alloc().unwrap();

        // Remaining ticks should not decommit (block is not fully empty)
        pool.process_pending_decommits();
        pool.process_pending_decommits();
        assert!(pool.blocks[0].is_committed());
    }

    #[test]
    fn test_sparse_decommit_preserves_non_trailing_blocks() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let block_size = 65536;
        let bin_size = 4096; // 16 bins per block
        let mut pool =
            Pool::with_config(bin_size, block_size, &config_immediate_decommit()).unwrap();

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

    #[cfg(any(debug_assertions, feature = "hardened"))]
    #[test]
    fn test_decommitted_block_free_panics() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let block_size = 65536;
        let bin_size = 4096;
        let mut pool =
            Pool::with_config(bin_size, block_size, &config_immediate_decommit()).unwrap();

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
            max_thread_cache_bytes: 1024 * 1024,
            ..Default::default()
        };
        let alloc = BinnedAllocator::with_config(config).unwrap();
        let ptr = alloc.alloc_bytes(64).unwrap();
        // Safety: Test code.
        unsafe {
            alloc.free_bytes(ptr, 64);
        }
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

    #[test]
    fn test_config_rejects_block_size_smaller_than_largest_class() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let page_size = PlatformVmOps::page_size();
        if page_size >= MAX_SMALL_SIZE {
            // A single page already fits the largest class; the misconfig
            // under test is unreachable on this platform.
            return;
        }
        // One page: page-aligned (passes the alignment check) but smaller
        // than the largest size class — must be rejected, not accepted with
        // bins_per_block == 0.
        let config = BinnedAllocatorConfig {
            block_size: page_size,
            ..Default::default()
        };
        match BinnedAllocator::with_config(config) {
            Err(VmError::InitializationFailed(msg)) => {
                assert!(
                    msg.contains("largest size class"),
                    "unexpected message: {msg}"
                );
            }
            Err(other) => panic!("Expected InitializationFailed, got {other:?}"),
            Ok(_) => panic!("Expected config validation to fail"),
        }
    }

    #[test]
    fn test_config_rejects_unaligned_block_size() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Odd size: cannot be a multiple of any power-of-two page size.
        // Must be rejected at runtime in all build modes, not just debug.
        let config = BinnedAllocatorConfig {
            block_size: MAX_SMALL_SIZE + 1,
            ..Default::default()
        };
        match BinnedAllocator::with_config(config) {
            Err(VmError::InitializationFailed(msg)) => {
                assert!(
                    msg.contains("multiple of the page size"),
                    "unexpected message: {msg}"
                );
            }
            Err(other) => panic!("Expected InitializationFailed, got {other:?}"),
            Ok(_) => panic!("Expected config validation to fail"),
        }
    }

    #[test]
    fn test_pool_flat_map_grows_beyond_initial_capacity() {
        let shift = 20; // 1 MiB pool spacing
        let mut map = PoolFlatMap::with_shift(shift);
        for i in 0..100usize {
            map.insert((i + 1) << shift, i);
        }
        for i in 0..100usize {
            assert_eq!(
                map.get((i + 1) << shift),
                Some(i),
                "lost entry {i} after growth"
            );
        }
        assert_eq!(map.get(101 << shift), None);
    }

    #[test]
    fn test_pool_chain_grows_past_eight_pools() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // One block per pool, one bin per block for the largest class:
        // every allocation of MAX_SMALL_SIZE exhausts a pool, so 12 live
        // allocations force 12 pools — past the old 8-slot map limit that
        // used to panic (and abort, with panic = "abort") on the 9th pool.
        let config = BinnedAllocatorConfig {
            pool_reserved_size: MAX_SMALL_SIZE,
            block_size: MAX_SMALL_SIZE,
            ..Default::default()
        };
        let alloc = BinnedAllocator::with_config(config).unwrap();
        let mut ptrs = Vec::new();
        for _ in 0..12 {
            ptrs.push(alloc.alloc_bytes(MAX_SMALL_SIZE).unwrap());
        }
        let pool_idx = BinnedAllocator::size_class(MAX_SMALL_SIZE, 1);
        assert!(
            alloc.pools[pool_idx].lock().unwrap().pools.len() >= 12,
            "expected at least 12 pools in the chain"
        );
        for ptr in ptrs {
            // Safety: ptr was allocated by `alloc` with this exact size.
            unsafe { alloc.free_bytes(ptr, MAX_SMALL_SIZE) };
        }
    }

    #[test]
    fn test_pool_chain_reuses_retired_pools() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let config = BinnedAllocatorConfig {
            pool_reserved_size: MAX_SMALL_SIZE,
            block_size: MAX_SMALL_SIZE,
            ..Default::default()
        };
        let alloc = BinnedAllocator::with_config(config).unwrap();
        let pool_idx = BinnedAllocator::size_class(MAX_SMALL_SIZE, 1);

        let mut ptrs = Vec::new();
        for _ in 0..4 {
            ptrs.push(alloc.alloc_bytes(MAX_SMALL_SIZE).unwrap());
        }
        let pools_after_fill = alloc.pools[pool_idx].lock().unwrap().pools.len();
        assert!(pools_after_fill >= 4);

        // Free everything, then allocate the same amount again: the chain
        // must reuse retired pools rather than reserving new ones.
        for ptr in ptrs.drain(..) {
            // Safety: ptr was allocated by `alloc` with this exact size.
            unsafe { alloc.free_bytes(ptr, MAX_SMALL_SIZE) };
        }
        for _ in 0..4 {
            ptrs.push(alloc.alloc_bytes(MAX_SMALL_SIZE).unwrap());
        }
        let pools_after_reuse = alloc.pools[pool_idx].lock().unwrap().pools.len();
        assert_eq!(
            pools_after_fill, pools_after_reuse,
            "chain reserved new pools instead of reusing retired ones"
        );
        for ptr in ptrs {
            // Safety: ptr was allocated by `alloc` with this exact size.
            unsafe { alloc.free_bytes(ptr, MAX_SMALL_SIZE) };
        }
    }

    #[test]
    fn test_pool_rejects_zero_bins_per_block() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Defense in depth at the Pool level: bin_size > block_size must be
        // rejected before any reservation is made.
        let config = BinnedAllocatorConfig::default();
        match Pool::with_config(MAX_SMALL_SIZE, MAX_SMALL_SIZE / 2, &config) {
            Err(VmError::InitializationFailed(msg)) => {
                assert!(
                    msg.contains("zero bins per block"),
                    "unexpected message: {msg}"
                );
            }
            Err(other) => panic!("Expected InitializationFailed, got {other:?}"),
            Ok(_) => panic!("Expected pool validation to fail"),
        }
    }

    // --- BlockMeta packed layout tests ---

    #[test]
    fn test_block_meta_packed_roundtrip() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let mut meta = BlockMeta::new(4096, 4096);
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
        #[cfg(not(any(debug_assertions, feature = "hardened")))]
        assert_eq!(std::mem::size_of::<BlockMeta>(), 8);
    }

    // ========================================================================
    // Phase 5 — Comprehensive testing suite
    // ========================================================================

    // --- 5b: GlobalRecycler multi-threaded stress ---

    #[test]
    #[cfg_attr(
        miri,
        ignore = "native stress test; recycler is covered by focused Miri tests"
    )]
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
                        if recycler.push(1, ptr).is_some() {
                            // Rejected — keep ownership
                            owned_ptrs.push(ptr);
                        }
                    } else {
                        // Pop: try to get a bundle
                        if let Some(ptr) = recycler.pop(1, &mut None) {
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
                while let Some(ptr) = recycler.pop(1, &mut None) {
                    owned_ptrs.push(ptr);
                }

                // Free all owned pointers
                for ptr in owned_ptrs {
                    // Safety: Test code.
                    unsafe {
                        alloc.free_bytes(ptr, 32);
                    }
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // Recycler should be empty after all threads finish and drain
        assert!(recycler.pop(1, &mut None).is_none());
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

        recycler.push(0, p1);
        recycler.push(1, p2);

        // Pop from slot 0 should get p1, not p2
        let got0 = recycler.pop(0, &mut None).unwrap();
        assert_eq!(got0, p1);
        assert!(recycler.pop(0, &mut None).is_none());

        let got1 = recycler.pop(1, &mut None).unwrap();
        assert_eq!(got1, p2);
        assert!(recycler.pop(1, &mut None).is_none());

        // Safety: Test code.
        unsafe {
            alloc.free_bytes(p1, 32);
        }
        // Safety: Test code.
        unsafe {
            alloc.free_bytes(p2, 32);
        }
    }

    #[test]
    fn test_recycler_overflow_all_consumed() {
        // Stress test: push enough bundles to trigger POP_WALK_BUDGET overflow,
        // then verify every pushed pointer is eventually popped (no leaks).
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let alloc = BinnedAllocator::new().unwrap();
        // Large max_bundles so the recycler accepts everything
        let recycler = GlobalRecycler::new(1024);

        let num_bundles = 20; // well above POP_WALK_BUDGET (4)
        let mut pushed: Vec<NonNull<u8>> = Vec::new();

        for _ in 0..num_bundles {
            let ptr = alloc.alloc_bytes(32).unwrap();
            pushed.push(ptr);
            recycler.push(1, ptr);
        }

        // Pop all bundles, consuming overflow chains
        let mut recovered: Vec<NonNull<u8>> = Vec::new();
        loop {
            let mut overflow = None;
            match recycler.pop(1, &mut overflow) {
                Some(head) => {
                    recovered.push(head);
                    // Walk any overflow chain returned
                    if let Some(extra) = overflow {
                        let mut cur = extra.as_ptr();
                        while !cur.is_null() {
                            recovered.push(NonNull::new(cur).unwrap());
                            // Safety: walking an overflow chain returned to
                            // this thread — exclusively owned.
                            let next = unsafe {
                                (*GlobalRecycler::recycler_link_atomic_ptr(cur))
                                    .load(Ordering::Relaxed)
                            };
                            cur = next;
                        }
                    }
                }
                None => break,
            }
        }

        // Every pushed pointer must be recovered exactly once
        let mut pushed_addrs: Vec<usize> = pushed.iter().map(|p| p.as_ptr() as usize).collect();
        let mut recovered_addrs: Vec<usize> =
            recovered.iter().map(|p| p.as_ptr() as usize).collect();
        pushed_addrs.sort_unstable();
        recovered_addrs.sort_unstable();
        assert_eq!(
            pushed_addrs, recovered_addrs,
            "leaked or duplicated pointers in recycler overflow path"
        );

        // Clean up
        for p in pushed {
            // Safety: Test code.
            unsafe {
                alloc.free_bytes(p, 32);
            }
        }
    }

    // --- 5d: Free-bin canary corruption detection ---

    #[cfg(any(debug_assertions, feature = "hardened"))]
    #[cfg(any(debug_assertions, feature = "hardened"))]
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

        let ptr = GlobalBinnedAllocator::alloc_bytes(400_000).unwrap();
        // Safety: Test code.
        unsafe {
            ptr.as_ptr().write_bytes(0xDD, 400_000);
        }
        // Safety: Test code.
        unsafe {
            GlobalBinnedAllocator::free_bytes(ptr, 400_000);
        }
    }

    #[test]
    fn test_large_alloc_various_sizes() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let alloc = BinnedAllocator::new().unwrap();
        let sizes = [MAX_SMALL_SIZE + 1, 300_000, 500_000, 1_000_000, 4_000_000];

        for &size in &sizes {
            let ptr = alloc.alloc_bytes(size).unwrap();
            // Safety: Test code.
            unsafe {
                // Write first and last byte
                *ptr.as_ptr() = 0xAA;
                *ptr.as_ptr().add(size - 1) = 0xBB;
            }
            // Safety: Test code.
            unsafe {
                alloc.free_bytes(ptr, size);
            }
        }
    }

    #[test]
    fn test_large_alloc_mixed_size_churn_balances_reserved() {
        let _guard = crate::memory::TEST_MUTEX.write().unwrap();
        // The large cache's exact page-count buckets guarantee every
        // allocation's reservation size is derivable from the request, so
        // free_bytes(requested_size) always releases/caches the full
        // mapping. Mixed-size churn through the cache must leave
        // TOTAL_RESERVED balanced once the allocator drops.
        let baseline_reserved = stats::TOTAL_RESERVED.load(Ordering::Relaxed);

        {
            let alloc = BinnedAllocator::new().unwrap();
            let p1 = alloc.alloc_bytes(800_000).unwrap();
            // Safety: Test code.
            unsafe {
                alloc.free_bytes(p1, 800_000);
            }

            // A smaller request gets its own exact mapping (never the
            // larger cached one); the cached 800_000 stays for its size.
            let p2 = alloc.alloc_bytes(400_000).unwrap();
            assert_ne!(p2, p1, "exact buckets must not serve an oversized block");
            // Safety: Test code.
            unsafe {
                alloc.free_bytes(p2, 400_000);
            }

            // Same-size churn reuses the exact cached mapping.
            let p3 = alloc.alloc_bytes(800_000).unwrap();
            assert_eq!(p3, p1, "same-size churn must hit the exact bucket");
            // Safety: Test code.
            unsafe {
                alloc.free_bytes(p3, 800_000);
            }
        } // Drop must release all cached/live large mappings.

        let final_reserved = stats::TOTAL_RESERVED.load(Ordering::Relaxed);
        assert_eq!(
            final_reserved, baseline_reserved,
            "TOTAL_RESERVED leaked after mixed-size large alloc churn"
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
            Pool::with_config(bin_size, block_size, &config_immediate_decommit()).unwrap();

        for cycle in 0..5 {
            // Fill the block
            let mut ptrs = Vec::new();
            if cycle > 0 {
                assert_eq!(pool.blocks.len(), 1, "Cycle {cycle}: should reuse block 0",);
            }
            for _ in 0..16 {
                let ptr = pool.alloc().unwrap();
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
        let cooldown = usize::from(config.decommit_cooldown);
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

        // Trim still decommits trailing empty blocks, but only after the
        // block survives `decommit_cooldown` trim passes (the first pass
        // queues the candidate and starts cooling it down).
        for pass in 0..cooldown {
            pool.trim();
            assert_eq!(
                pool.committed, 65536,
                "block decommitted on pass {pass}, before its cooldown expired"
            );
        }
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
            Pool::with_config(bin_size, block_size, &config_immediate_decommit()).unwrap();

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
    #[cfg_attr(
        miri,
        ignore = "native stress test; cross-thread paths are covered by Loom"
    )]
    fn test_producer_consumer_cross_thread() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Classic pathology: Thread A allocates, Thread B frees.
        // Exercises cross-thread recycler path.
        let alloc = Arc::new(BinnedAllocator::new().unwrap());
        let (tx, rx) = std::sync::mpsc::channel::<(SendPtr, usize)>(); // (ptr, size)
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
                unsafe {
                    *ptr.as_ptr() = i.to_le_bytes()[0];
                }
                tx.send((SendPtr(ptr.as_ptr()), size)).unwrap();
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

            for (ptr, size) in rx {
                let ptr = NonNull::new(ptr.0).unwrap();
                let layout = std::alloc::Layout::from_size_align(size, 1).unwrap();
                alloc_c.free_with_cache(&mut cache, ptr, layout);
            }
        });

        producer.join().unwrap();
        consumer.join().unwrap();
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "native stress test; mixed paths are covered separately under Miri"
    )]
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
                    unsafe {
                        *ptr.as_ptr() = t.wrapping_mul(100).wrapping_add(i);
                    }
                    ptrs.push((ptr, size));
                }

                // Verify
                for (i, (ptr, _)) in ptrs.iter().enumerate() {
                    // Safety: Test code.
                    unsafe {
                        assert_eq!(
                            ptr.as_ptr().read(),
                            t.wrapping_mul(100).wrapping_add(i.to_le_bytes()[0])
                        );
                    }
                }

                // Free in reverse
                for (ptr, size) in ptrs.into_iter().rev() {
                    // Safety: Test code.
                    unsafe {
                        alloc.free_bytes(ptr, size);
                    }
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "native contention test; recycler races are covered by Loom"
    )]
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
            unsafe {
                *ptrs[i].cast::<*mut u8>().as_ptr() = ptrs[i + 1].as_ptr();
            }
        }
        // Last item is null-terminated
        // Safety: Test code.
        unsafe {
            *ptrs[4].cast::<*mut u8>().as_ptr() = std::ptr::null_mut();
        }

        // Feed the chain to a stack via receive_walk
        let mut stack = LocalFreeList::new_const();
        stack.max_length = 64;
        let mut slots = [std::ptr::null_mut::<u8>(); 64];
        let sp = slots.as_mut_ptr();
        // Safety: local slots buffer; chain built above.
        unsafe {
            let (received, spill) = stack.receive_walk(sp, 64, ptrs[0]);
            assert_eq!(received, 5);
            assert!(spill.is_none());
            assert_eq!(stack.count(), 5);

            // Adopted in chain order; pops are LIFO: ptrs[4] .. ptrs[0].
            for i in (0..5).rev() {
                let popped = stack.pop(sp).unwrap();
                assert_eq!(popped, ptrs[i], "Item {i} mismatch");
            }
            assert!(stack.pop(sp).is_none());
        }
        assert_eq!(stack.count(), 0);

        // Free all
        for p in ptrs {
            // Safety: Test code.
            unsafe {
                alloc.free_bytes(p, 32);
            }
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
        unsafe {
            *bundle_a.cast::<*mut u8>().as_ptr() = bundle_b.as_ptr();
            *bundle_b.cast::<*mut u8>().as_ptr() = std::ptr::null_mut();
        }

        let mut stack = LocalFreeList::new_const();
        stack.max_length = 64;
        let mut slots = [std::ptr::null_mut::<u8>(); 64];
        let sp = slots.as_mut_ptr();
        // Safety: local slots buffer; bundle built above.
        unsafe {
            stack.push(sp, existing);
            assert_eq!(stack.count(), 1);

            let (received, spill) = stack.receive_walk(sp, 64, bundle_a);
            assert_eq!(received, 2);
            assert!(spill.is_none());
            assert_eq!(stack.count(), 3);

            // LIFO over [existing, A, B].
            assert_eq!(stack.pop(sp).unwrap(), bundle_b);
            assert_eq!(stack.pop(sp).unwrap(), bundle_a);
            assert_eq!(stack.pop(sp).unwrap(), existing);
            assert!(stack.pop(sp).is_none());
        }

        // Safety: Test code.
        unsafe {
            alloc.free_bytes(existing, 32);
        }
        // Safety: Test code.
        unsafe {
            alloc.free_bytes(bundle_a, 32);
        }
        // Safety: Test code.
        unsafe {
            alloc.free_bytes(bundle_b, 32);
        }
    }

    #[test]
    fn test_refill_moves_one_class_batch() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // A cache-miss refill moves exactly CLASS_BATCH[idx] bins as one
        // chain and grows the adaptive per-class limit by one batch
        // (slow start), capped at CLASS_CAP[idx].
        let alloc = BinnedAllocator::with_config(BinnedAllocatorConfig::default()).unwrap();
        let mut cache = ThreadCache::new();
        // Safety: Test code.
        unsafe {
            cache.bind(std::mem::transmute::<
                &binned::BinnedAllocator,
                &binned::BinnedAllocator,
            >(&alloc));
        }

        let idx = BinnedAllocator::size_class(32, 1);
        let batch = u32::from(CLASS_BATCH[idx]);
        let cap = u32::from(CLASS_CAP[idx]);
        let initial_limit = cache.bins[idx].max_length;

        let layout = std::alloc::Layout::from_size_align(32, 1).unwrap();
        let ptr = alloc.alloc_with_cache(&mut cache, layout).unwrap();
        assert_eq!(
            cache.bins[idx].count(),
            batch - 1,
            "refill = one class batch minus the bin handed out"
        );
        assert_eq!(cache.cached_bytes, SIZE_CLASSES[idx] * (batch - 1) as usize);
        assert_eq!(
            cache.bins[idx].max_length,
            (initial_limit + batch).min(cap),
            "adaptive limit grows by one batch per refill"
        );
        alloc.free_with_cache(&mut cache, ptr, layout);
    }

    #[test]
    fn test_config_affects_cache_limit() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        // Verify custom cache_count_limits seed the initial adaptive limit:
        // exceeding it releases one transfer batch (not the whole list).
        let config = BinnedAllocatorConfig {
            cache_count_limits: [8, 4, 2, 1],
            ..Default::default()
        };
        let alloc = BinnedAllocator::with_config(config).unwrap();
        let mut cache = ThreadCache::new();
        // Safety: Test code.
        unsafe {
            cache.bind(std::mem::transmute::<
                &binned::BinnedAllocator,
                &binned::BinnedAllocator,
            >(&alloc));
        }

        let size = 32;
        let layout = std::alloc::Layout::from_size_align(size, 1).unwrap();
        let idx = BinnedAllocator::size_class(size, 1);
        assert_eq!(
            cache.bins[idx].max_length, 8,
            "bind seeds the initial limit"
        );

        // Free 8 items to cache — at the limit (check is >), nothing released
        let mut ptrs = Vec::new();
        for _ in 0..8 {
            ptrs.push(alloc.alloc_bytes(size).unwrap());
        }
        for p in ptrs {
            alloc.free_with_cache(&mut cache, p, layout);
        }
        assert_eq!(cache.bins[idx].count(), 8);

        // One more crosses the limit and releases up to one transfer batch;
        // with the limit still under the batch size that drains the cache,
        // and the free-side slow start nudges the limit up by one.
        let extra = alloc.alloc_bytes(size).unwrap();
        alloc.free_with_cache(&mut cache, extra, layout);
        assert_eq!(cache.bins[idx].count(), 0);
        assert_eq!(
            cache.bins[idx].max_length, 9,
            "slow-start growth on overflow"
        );
        assert_eq!(cache.cached_bytes, 0);
    }

    /// Remote frees of bins whose home blocks have a live owner must flow
    /// through the mask channel back to that owner, not the recycler.
    #[test]
    fn test_remote_mask_returns_to_owner() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let alloc = BinnedAllocator::with_config(BinnedAllocatorConfig {
            remote_mask_channel: true,
            ..Default::default()
        })
        .unwrap();
        let mut a = ThreadCache::new();
        let mut b = ThreadCache::new();
        // Safety: Test code; caches are dropped before `alloc`.
        unsafe {
            a.bind(std::mem::transmute::<
                &binned::BinnedAllocator,
                &binned::BinnedAllocator,
            >(&alloc));
            b.bind(std::mem::transmute::<
                &binned::BinnedAllocator,
                &binned::BinnedAllocator,
            >(&alloc));
        }
        assert_ne!(a.owner_slot, u16::MAX, "bind must claim an owner slot");
        assert_ne!(b.owner_slot, a.owner_slot);

        // A's refills claim ownership of the blocks they draw from.
        let layout = std::alloc::Layout::from_size_align(64, 1).unwrap();
        let n = 200;
        let ptrs: Vec<NonNull<u8>> = (0..n)
            .map(|_| alloc.alloc_with_cache(&mut a, layout).unwrap())
            .collect();

        // B frees them all: overflow releases must publish into A's
        // blocks' masks rather than the recycler.
        for p in ptrs {
            alloc.free_with_cache(&mut b, p, layout);
        }

        let mut reconciled = 0u32;
        {
            let (slot, generation) = (a.owner_slot, a.owner_gen);
            let cache = &mut a;
            alloc
                .remote
                .reconcile(slot, generation, write_link, |class, head, _tail, count| {
                    let c = class as usize;
                    let spill = cache.receive_chain_bin(c, head, count);
                    let adopted = count - spill.as_ref().map_or(0, |s| s.1);
                    cache.cached_bytes += SIZE_CLASSES[c] * adopted as usize;
                    if let Some((sh, _)) = spill {
                        alloc.release_segment(c, sh, slot, generation);
                    }
                    reconciled += adopted;
                });
        }
        assert!(
            reconciled > 0,
            "cross-cache frees must return through the mask channel"
        );

        // Reconciled bins are allocatable again and everything drains
        // cleanly at cache drop (miri leak check covers the accounting).
        let p = alloc.alloc_with_cache(&mut a, layout).unwrap();
        alloc.free_with_cache(&mut a, p, layout);
    }

    /// After the owner dies, remote frees must fall back to the recycler
    /// (never a dead slot), and its still-masked bins must be recoverable
    /// by the trim sweep.
    #[test]
    fn test_remote_mask_dead_owner_falls_back() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        let alloc = BinnedAllocator::with_config(BinnedAllocatorConfig {
            remote_mask_channel: true,
            ..Default::default()
        })
        .unwrap();
        let layout = std::alloc::Layout::from_size_align(64, 1).unwrap();
        let mut b = ThreadCache::new();
        // Safety: Test code; caches are dropped before `alloc`.
        unsafe {
            b.bind(std::mem::transmute::<
                &binned::BinnedAllocator,
                &binned::BinnedAllocator,
            >(&alloc));
        }

        let ptrs: Vec<NonNull<u8>> = {
            let mut a = ThreadCache::new();
            // Safety: Test code; `a` drops inside this scope.
            unsafe {
                a.bind(std::mem::transmute::<
                    &binned::BinnedAllocator,
                    &binned::BinnedAllocator,
                >(&alloc));
            }
            (0..200)
                .map(|_| alloc.alloc_with_cache(&mut a, layout).unwrap())
                .collect()
            // `a` drops here: slot released, generation bumped.
        };

        for p in ptrs {
            alloc.free_with_cache(&mut b, p, layout);
        }
        drop(b);
        // Everything must be recoverable: trim sweeps any bins that raced
        // into masks around the owner's death.
        alloc.trim();
        let p = alloc.alloc_bytes(64).unwrap();
        // Safety: Test code.
        unsafe { alloc.free_bytes(p, 64) };
    }

    /// Fast-path decomposition for the front-end rebuild (roadmap):
    /// empirically prices each architectural layer of the ~8 ns/op
    /// single-thread path. Run manually with
    /// `cargo test --release fastpath_decomposition -- --ignored --nocapture`.
    /// Not a regression test — it prints a table for design decisions.
    #[test]
    #[ignore = "manual measurement harness, run with --release --nocapture"]
    #[expect(
        clippy::too_many_lines,
        clippy::items_after_statements,
        clippy::cast_precision_loss,
        reason = "measurement harness: linear script structure and ns-scale floats are the point"
    )]
    fn fastpath_decomposition() {
        use std::hint::black_box;
        use std::time::Instant;

        let _guard = crate::memory::TEST_MUTEX.read().unwrap();

        fn time(label: &str, pairs: u64, f: impl FnOnce()) -> f64 {
            let t = Instant::now();
            f();
            let ns = t.elapsed().as_nanos() as f64 / pairs as f64;
            println!("{label:44} {ns:6.2} ns/pair");
            ns
        }

        // Pre-generated benchmark-like size stream (even [16,1000]).
        let mut seed = 0x9E37_79B9u32;
        let sizes: Vec<usize> = (0..4096)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                16 + (seed as usize % 985)
            })
            .collect();

        const N: u64 = 20_000_000;

        // (a) Full stack, fixed 64 B LIFO pairs: TLS ceremony + wrappers +
        // LUT + L0 array hit. The best case of the current architecture.
        drop(GlobalBinnedAllocator::init());
        let full_hot = time("full stack, hot 64B pair", N, || {
            for _ in 0..N {
                let p = GlobalBinnedAllocator::alloc_bytes(black_box(64)).unwrap();
                // Safety: just allocated with this size.
                unsafe { GlobalBinnedAllocator::free_bytes(black_box(p), 64) };
            }
        });

        // (b) Full stack, benchmark-like random sizes (still LIFO pairs, so
        // cache-hot: isolates size-stream effects from working-set misses).
        let full_rand = time("full stack, random [16,1000] pair", N, || {
            for i in 0..N {
                let s = sizes[(i & 4095) as usize];
                let p = GlobalBinnedAllocator::alloc_bytes(black_box(s)).unwrap();
                // Safety: just allocated with this size.
                unsafe { GlobalBinnedAllocator::free_bytes(black_box(p), s) };
            }
        });

        // (c) Same ops minus the TLS/global ceremony: direct bound cache.
        let alloc = BinnedAllocator::new().unwrap();
        let mut cache = ThreadCache::new();
        // Safety: cache is dropped before `alloc` at scope end.
        unsafe {
            cache.bind(std::mem::transmute::<
                &binned::BinnedAllocator,
                &binned::BinnedAllocator,
            >(&alloc));
        }
        let direct_hot = time("minus TLS/global (direct cache), 64B", N, || {
            for _ in 0..N {
                let idx = BinnedAllocator::size_class_min_align(black_box(64));
                let p = alloc.alloc_small_with_cache(&mut cache, idx).unwrap();
                alloc.free_small_with_cache(&mut cache, black_box(p), idx);
            }
        });

        // (d) Dense stack (the current cache structure) burst-16 — was
        // the intrusive-list measurement before the C2 rebuild; kept so
        // the decomposition table stays comparable across revisions.
        let mut list = LocalFreeList::new_const();
        list.max_length = 4096;
        let mut stack_slots = vec![std::ptr::null_mut::<u8>(); 256];
        let stack_sp = stack_slots.as_mut_ptr();
        let block = vec![0u8; 64 * 1024];
        let bins: Vec<NonNull<u8>> = (0..512)
            .map(|i| NonNull::new(block.as_ptr().wrapping_add(i * 64).cast_mut()).unwrap())
            .collect();
        // Safety: local slots buffer, capacity 256 >= depth 64.
        unsafe {
            for &b in &bins[..64] {
                list.push(stack_sp, b);
            }
        }
        let mut scratch = [std::ptr::null_mut::<u8>(); 16];
        let list_burst = time("cache stack, burst-16 pop/push", N, || {
            for _ in 0..N / 16 {
                // Safety: same buffer; balanced pops/pushes keep depth <= 64.
                unsafe {
                    for s in &mut scratch {
                        *s = list.pop(stack_sp).unwrap().as_ptr();
                    }
                    for s in &scratch {
                        list.push(stack_sp, NonNull::new(black_box(*s)).unwrap());
                    }
                }
            }
        });

        // (e) Dense array-stack prototype, same burst pattern: the
        // rebuild's proposed cache structure (no object touches).
        let mut top = 64usize;
        let mut slots = vec![std::ptr::null_mut::<u8>(); 4096];
        for (i, &b) in bins[..64].iter().enumerate() {
            slots[i] = b.as_ptr();
        }
        let array_burst = time("array stack, burst-16 pop/push", N, || {
            for _ in 0..N / 16 {
                for s in &mut scratch {
                    top -= 1;
                    *s = black_box(slots[top]);
                }
                for s in &scratch {
                    slots[top] = black_box(*s);
                    top += 1;
                }
            }
        });

        // (f) Size→class LUT alone.
        let lut = time("size_class LUT lookup", N, || {
            let mut acc = 0usize;
            for i in 0..N {
                let s = sizes[(i & 4095) as usize];
                acc = acc.wrapping_add(BinnedAllocator::size_class_min_align(black_box(s)));
            }
            black_box(acc);
        });

        println!("---- decomposition ----");
        println!("TLS/global ceremony  ≈ {:5.2} ns", full_hot - direct_hot);
        println!(
            "intrusive − array    ≈ {:5.2} ns (per pair, burst regime)",
            list_burst - array_burst
        );
        println!("size-stream effect   ≈ {:5.2} ns", full_rand - full_hot);
        println!("LUT                  ≈ {lut:5.2} ns");
        println!(
            "projected rebuilt hot pair ≈ {:5.2} ns (direct − intrusive delta)",
            direct_hot - (list_burst - array_burst).max(0.0)
        );
    }

    /// Unsized free: pool-backed pointers free from the address alone;
    /// large and foreign pointers are refused untouched.
    #[test]
    fn test_try_free_ptr_routes_by_address() {
        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        drop(GlobalBinnedAllocator::init());

        // Every size-class tier, including pools created mid-test.
        for size in [16usize, 48, 1000, 8192, 65536, 262_144] {
            let p = GlobalBinnedAllocator::alloc_bytes(size).unwrap();
            // Safety: live allocation, freed once.
            let freed = unsafe { GlobalBinnedAllocator::try_free_ptr(p) };
            assert!(freed, "pool-backed {size}B pointer must free unsized");
        }

        // Large allocations are not pool-backed: refused, then freed sized.
        let big = GlobalBinnedAllocator::alloc_bytes(400_000).unwrap();
        // Safety: live allocation; try_free_ptr does not free on false.
        unsafe {
            assert!(!GlobalBinnedAllocator::try_free_ptr(big));
            GlobalBinnedAllocator::free_bytes(big, 400_000);
        }

        // A pointer from a DIFFERENT allocator instance is foreign to the
        // global table: refused untouched.
        let other = BinnedAllocator::new().unwrap();
        let foreign = other.alloc_bytes(64).unwrap();
        // Safety: live allocation from `other`; freed there afterwards.
        unsafe {
            assert!(!GlobalBinnedAllocator::try_free_ptr(foreign));
            other.free_bytes(foreign, 64);
        }
    }

    /// Native-API workload benchmark: the rpmalloc-benchmark random churn
    /// shape (even [16,1000], 50 K live slots), driven twice — once
    /// through the sized API (`free_bytes`, what Rust callers get: Layout
    /// always carries the size) and once through the unsized
    /// `try_free_ptr` (what the C shim uses). The pair prices the sized
    /// advantage with everything else held equal. Run with
    /// `cargo test --release sized_api_workload -- --ignored --nocapture`.
    #[test]
    #[ignore = "manual measurement harness, run with --release --nocapture"]
    fn sized_api_workload() {
        const SLOTS: usize = 50_000;
        const OPS: u64 = 30_000_000;

        let _guard = crate::memory::TEST_MUTEX.read().unwrap();
        drop(GlobalBinnedAllocator::init());

        let run = |label: &str, sized: bool| {
            let mut slots: Vec<(*mut u8, usize)> = vec![(std::ptr::null_mut(), 0); SLOTS];
            let mut seed = 0x1234_5678u32;
            let mut rng = move || {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                seed
            };
            let t = std::time::Instant::now();
            for _ in 0..OPS {
                let i = (rng() as usize) % SLOTS;
                let (p, sz) = slots[i];
                if p.is_null() {
                    let size = 16 + (rng() as usize % 985);
                    let ptr = GlobalBinnedAllocator::alloc_bytes(size).unwrap();
                    // Touch the allocation, as the C harness does.
                    // Safety: fresh allocation of at least 16 bytes.
                    unsafe { ptr.as_ptr().write(1) };
                    slots[i] = (ptr.as_ptr(), size);
                } else {
                    // Safety: (p, sz) recorded at allocation; freed once.
                    unsafe {
                        let nn = NonNull::new_unchecked(p);
                        if sized {
                            GlobalBinnedAllocator::free_bytes(nn, sz);
                        } else {
                            assert!(GlobalBinnedAllocator::try_free_ptr(nn));
                        }
                    }
                    slots[i] = (std::ptr::null_mut(), 0);
                }
            }
            let el = t.elapsed().as_secs_f64();
            #[expect(clippy::cast_precision_loss, reason = "display math")]
            let mops = OPS as f64 / el / 1e6;
            println!("{label:32} {mops:7.1} M ops/s");
            for (p, sz) in slots {
                if !p.is_null() {
                    // Safety: recorded live allocation.
                    unsafe { GlobalBinnedAllocator::free_bytes(NonNull::new_unchecked(p), sz) };
                }
            }
        };
        run("sized free (native Rust API)", true);
        run("unsized free (try_free_ptr)", false);
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
                    unsafe {
                        GlobalBinnedAllocator::free_bytes(p, 256);
                    }
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
        let mut meta = BlockMeta::new(0xFFFF, 0xFFFF_usize);
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
        let mut pool = Pool::with_config(65536, 65536, &config_immediate_decommit()).unwrap();

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
    #[cfg_attr(miri, ignore = "128-thread native stress test")]
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
        assert_eq!(
            ptr.as_ptr() as usize % align,
            0,
            "Pointer should be aligned to {align}"
        );

        // Safety: Test code.
        unsafe {
            *ptr.as_ptr() = 0xAA;
        }

        // Safety: Test code.
        unsafe {
            alloc.free(ptr, layout);
        }
    }
}
