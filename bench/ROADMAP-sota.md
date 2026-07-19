# Roadmap: matching internal tcmalloc (SoTA scoping)

**Target.** "Internal tcmalloc" is open source: [google/tcmalloc]
(https://github.com/google/tcmalloc) — the allocator Google runs fleet-wide,
distinct from and far ahead of the gperftools-era `libtcmalloc_minimal`
benchmarked in [RESULTS.md](./RESULTS.md). This document scopes what qen
needs to match it, based on a source-level study of a clone at
`bench/google-tcmalloc` (HEAD `3f490cc`; key files cited inline). Where the
benchmarked machine (macOS/arm64) cannot run its Linux-only mechanisms, the
scoping says so explicitly and gives the portable equivalent.

Status of prerequisite items (from the feasibility analysis):

| item | status |
|---|---|
| 1. Flatten the hot path | done (measured: ~+1% alone; enabler for 2) |
| 2. Adaptive thread caches + chain refills + byte-budget scavenge | done |
| 3. 256 KiB size classes + large-cache policy fix | done |
| 4. Flux-proportional remote frees (return masks + block adoption) | **scoped here** |
| 5+. Per-CPU caches, hugepage-aware backend, tuning | **scoped here** |

Measured before/after for items 1–3 is in
[RESULTS.md](./RESULTS.md) (see "Item 1–3 re-benchmark").

---

## 1. What internal tcmalloc actually is (measured facts, not folklore)

The default production build (`TCMALLOC_PAGE_SHIFT=13`, 8 KiB pages) differs
from gperftools tcmalloc in five load-bearing ways:

### 1.1 Per-CPU rseq caches replace per-thread caches

`tcmalloc/internal/percpu_tcmalloc.h`, `tcmalloc/cpu_cache.h`. One mmap'd
slab region per CPU (dynamically sized 16 KiB → 256 KiB per CPU;
`kInitialBasePerCpuShift=14`, `kMaxBasePerCpuShift=18`). Each per-CPU region
holds, per size class, a 4-byte `{current, end}` header and an array of
`void*` slots. The thread-local slab pointer overlaps `__rseq_abi
.cpu_id_start` (top 4 bytes), so a kernel reschedule *automatically
invalidates* the cached pointer: the fast path is "load one word, test bit
63". Push is ~8 instructions, Pop ~9, **zero atomics, zero locks**, with the
kernel guaranteeing restart-on-preemption (restartable sequences). Both
prefetch the next slot/object. Footprint scales with CPUs (≤ 256 KiB × 2
copies × nCPU), not threads. Per-CPU byte cap: `kMaxCpuCacheSize = 1.5 MiB`.
Hot classes (the 10 smallest) get 2000-object depth; larger classes 144.

A background thread (1 s cadence, `background.cc`) *resizes everything by
measured misses*: per-class capacity steal within a CPU (every 2 s),
cross-CPU steal of up to 5% capacity from low-miss CPUs (every 5 s), full
drain of idle CPUs (every 30 s), slab growth when
`overflows > 0.9 × underflows` (every 29 s).

**Linux/x86-64/aarch64 only.** On any other platform the front end falls
back to gperftools-style per-thread `ThreadCache` (`kMaxThreadCacheSize =
4 MiB`, overall budget 32 MiB with stealing).

### 1.2 A transfer cache between front end and central lists

`tcmalloc/transfer_cache_internals.h`. Per size class: a spinlocked LIFO
array of `void*` moved in `num_objects_to_move` batches (one `memcpy` per
op). Initial capacity 16 batches, max 64 batches, ≤ 1 MiB per class.
Capacity is re-balanced in the background (grow the top-miss classes by
shrinking the least-missed), and a 5 s "plunder" pass returns everything
below the interval low-water mark to the central list. Optionally sharded
per L3 domain. This layer is why central-list contention rarely shows in
its profiles — CPUs exchange whole batches, not objects.

### 1.3 Size classes to 256 KiB, batch-tuned per class

`kMaxSize = 256 KiB`; ~48 populated classes (8-byte granularity ≤ 1 KiB,
128-byte granularity above). `num_objects_to_move` per class: 32 for small
sizes, tapering to 2 at large sizes (`size_classes.cc`). Span sizes chosen
to keep (leftover + 64 B span metadata) / span small. qen's item 3 mirrors
this shape (96 classes to 256 KiB; `CLASS_BATCH = clamp(64 KiB/bin, 2, 32)`).

### 1.4 Hugepage-aware backend ("Temeraire")

`huge_page_aware_allocator.h`, `huge_page_filler.h`, `huge_region.h`,
`docs/temeraire.md`. All OS memory is managed in 2 MiB hugepages:

- **Filler** for ≤ 1 MiB requests: packs allocations into partially-used
  hugepages, ordered by *longest free range* (not total free), so
  fragmentation-resistant hugepages are used first and whole hugepages
  drain empty.
- **Regions** (1 GiB reservations) for mid-size requests that would
  otherwise strand slack.
- **Donation**: the tail of a hugepage half-used by a large allocation is
  donated to the filler rather than wasted.
- **Subrelease discipline**: freed hugepages are returned to the OS only if
  demand history over 60 s / 300 s windows says they won't be needed —
  breaking a hugepage is treated as expensive and semi-permanent.
- Background release at a constant configurable rate; cold allocations get
  `MADV_NOHUGEPAGE`'d native pages so they never fragment hot hugepages.

The *data structures* are portable; the madvise flavors and 2 MiB THP
assumption are Linux-specific.

### 1.5 Free-path economics

Pointer → size class is a 2-level radix pagemap (`pagemap.h`) with the size
class stored in a dedicated parallel array (2–3 dependent loads, lock-free);
span pointer and size class are packed in one word (class in the top 16
bits). With sized delete (`operator delete(p, size)`), the pagemap is
**skipped entirely** — size → class is pure arithmetic, no memory access.
qen's sized-free API is already this design point; the rpmalloc-benchmark
adapter pays a span-table lookup to *recover* sizes, which internal
tcmalloc would also have to pay under that harness.

---

## 2. Where qen stands after items 1–3 (corrected 2026-07-07)

Full matrix in RESULTS.md — including the disclosure that every earlier
qen number at ≥ 4 threads was capped by a benchmark-adapter bug (a
per-alloc store to shared span-table lines; fixed to first-touch). The
clean shape that matters for scoping:

- **Single-threaded** (unaffected by the bug): second only to gperftools
  tcmalloc in 4 of 5 scenarios, 1.6–1.9× behind it. The residual gap is
  the per-op fast-path constant — qen's TLS-handle push/pop vs tcmalloc's
  ~12-instruction hit and internal tcmalloc's strictly cheaper rseq
  per-CPU hit. Items 1–3 bought +18–45% at 1 thread and 46× on the former
  large-path collapse.
- **16 threads, clean**: qen is second or third of the field in every
  scenario past 1 KiB (30.2/24.1/17.3/9.4 M vs rpmalloc's
  39.8/33.4/24.9/11.2), ahead of gperftools tcmalloc everywhere except
  the smallest sizes. The remaining scaling items are: the smallest-size
  cell (33.9 vs the pack's 47–56 M) and a per-CPU-efficiency drop of
  3.7× from 1t→16t vs rpmalloc's 2.3× — ordinary contention-tuning
  territory, **cause currently unprofiled on clean data**. The old "6×
  recycler cliff" never existed.
- **RSS**: lowest of every allocator except gperftools tcmalloc in all
  five 16-thread scenarios, smallest outright at 1 thread. The decommit
  discipline must survive all future items.

---

## 3. Item 4: flux-proportional remote frees — EXECUTED AND FALSIFIED
## (kept as the design record; verdict first)

> **Verdict (2026-07-07).** Item 4A was implemented in full
> (`src/memory/remote_mask.rs`, loom-verified, 274 tests) and failed its
> own pre-committed falsification: 16-thread throughput flat, capacity
> stranded behind reconcile latency (peak-RSS regression). It is gated
> off (`remote_mask_channel: false`) pending pipeline-shaped benchmarks.
> The investigation that followed — placement/line-sharing sweep
> (negative), transfer-density profile (negative), cross-rate-0 and
> working-set sweeps — traced the "16-thread collapse" this item was
> scoped against to a **benchmark-adapter bug**, not qen (see RESULTS.md
> disclosure). The problem statement below therefore overstates the gap
> by ~4×; the design's genuine win case (stable producer→consumer flows
> + block adoption) remains untested until larson-class benchmarks
> exist. Everything below is preserved as written on 2026-07-06.

**Problem.** At 16 threads with 1-in-2 cross-thread frees, the recycler
saturates: shard caps (`recycler_max_bundles/shards`) overflow to the pool
mutex, and every operation's shared-line loads become cross-core misses
(measured in tcmalloc-analysis.md: ~55% of samples on-CPU in wrappers, ~3%
in mutex wait — coherence, not locking).

**The cost floor, derived (design against this, not against the current
leaderboard).** The application already moves a transferred object's own
cache lines between cores; the allocator's *added* cross-core traffic is
metadata — publishing frees and returning capacity to where allocation
pressure lives. The irreducible part is the return flow, which is
proportional to the **net memory flux between threads**, not to the
operation rate. So the floor is: allocator coherence traffic
∝ flux / transfer-granularity, with zero per-op shared-line touches in
steady state. Ranked against that floor:

- qen today (and gperftools tcmalloc structurally): traffic ∝ op-rate /
  batch-size on globally shared lines — wrong scaling variable, worst
  constant. (Measured: item 2's deeper caches moved the small-size
  16-thread cells only +6%.)
- Owner-drained MPSC inboxes (rpmalloc/mimalloc/snmalloc): traffic ∝
  remote-free rate — one CAS per free on per-owner lines, and the owner's
  drain pointer-chases lines that are cold by construction. Better
  constant, still per-op scaling. Adopting it converges qen *to* the
  24–56 M ops/CPU-s of that class; it cannot go past it.
- Block-granularity capacity transfer with an all-local steady state:
  traffic ∝ flux / block_size — three to four orders below per-op. The
  16-thread ceiling then becomes local-cache throughput (the 1-thread
  number), not the current leaders' numbers.

**Design** (derived from the floor; individual pieces have prior art —
mimalloc's delayed-free lists, internal tcmalloc's span bitmaps,
rpmalloc's span adoption — the composition is chosen by the floor):

- **Commutative per-block return masks.** A remote free sets one bit in a
  per-block bitmask: `fetch_or(release)` — no CAS retry loop; OR is
  commutative, so arbitrary fan-in cannot livelock. Consecutive remote
  frees into one block keep the mask line in the freeing core's cache, so
  amortized coherence cost is below one miss per free wherever producers
  allocate with any spatial locality. The owner reconciles with one
  `xchg(acquire)` per block covering up to 64 frees, and reclaimed bins
  are *computed* from bits (contiguous, prefetchable) instead of
  pointer-chased. Fits the existing machinery: free already derives the
  block from the masked base and the bin index via reciprocal division;
  `BlockMeta` carries a has-pending-remote summary bit; masks live in
  pool-side arrays (1 bit per bin, ≤ ~0.8% overhead at 16 B bins). A
  per-owner MPSC list of *dirty block indices* (deduped by the summary
  bit) tells owners what to reconcile — its event rate is per-block, not
  per-free. Trade named honestly: freed capacity becomes visible at
  reconcile points, not instantly.
- **Block ownership migration.** When a thread's frees dominate a block
  and its owner hasn't allocated from it for an epoch, the freeing thread
  adopts the block (one CAS on the block's ownership word, with
  hysteresis against ping-pong). In stable producer→consumer flows the
  consumer ends up owning what it frees into: frees go fully local, the
  producer draws fresh blocks from the pool, and allocator traffic
  reaches the floor (~O(1) metadata ops per block of flux). Random
  fan-in (this suite's shuffled cross-frees) fails the adoption threshold
  and degrades gracefully to mask-reconcile — still cheaper per free than
  an inbox CAS plus cold-chain drain.
- **What stays**: the GlobalRecycler remains the escape hatch for
  orphaned blocks (owner exited — `flush()` + `CACHE_TRIM_EPOCH` extend
  as today), unbound caches, and trim, so correctness never depends on a
  live owner.

**Verification plan**: loom models for mask-publish vs reconcile vs
adoption races (fetch_or/xchg protocols are small, ideal loom targets)
and adoption vs old-owner refill; miri for mask-index provenance; an
interleaved-ownership adversary test to prove the hysteresis holds;
fault-injection for commit failures during reconcile-to-pool.

**Effort**: ~1–1.5 weeks including loom/miri/CI — roughly 1.5–2× the
inbox design, the premium buying the flux-proportional asymptote rather
than parity with the current leaders. Risk concentrates in the adoption
lifecycle (registry of owners, epoch policy), not the fast paths.

**Expected effect, stated as falsifiable predictions**: mask-only should
multiply the small-size 16-thread cells (currently 8.9–11.5 M) rather
than add percent; with adoption engaged on pipeline-shaped workloads
(larson in the planned mimalloc-bench matrix is the canonical case),
throughput should approach single-thread rates and land *above*
rpmalloc's 24–56 M, not at it. If it only matches the inbox class, the
design has failed its own premise — find out in week one, not after a
tuning campaign. Microbenchmarks then refine constants the floor cannot
supply: adoption threshold/epoch, mask-group granularity, reconcile
trigger, and whether the tiniest classes want a hybrid (mask density is
worst at 16 B bins). Does not by itself close the 1-thread gap to
internal tcmalloc.

---

## 4. Per-CPU caches (scoped)

The single biggest remaining fast-path lever on Linux. qen equivalent:

- **Linux/x86-64/aarch64**: adopt rseq per-CPU slabs for the L0/list tier:
  per-CPU regions addressed off a TLS word overlapping `cpu_id_start`,
  17-instruction push/pop critical sections in inline asm, batch spill/
  refill through the item-4 reconcile path or a transfer-cache tier. This is a
  substantial, platform-gated subsystem: rseq registration, asm for two
  architectures, remote-stop protocol for resize/drain
  (`ScopedSlabCpuStop` equivalent), and a fallback detection path. qen's
  Rust context adds FFI/asm and miri/loom exclusions (model the algorithm,
  not the asm).
- **Everywhere else** (incl. this benchmark machine): the per-thread
  adaptive caches from item 2 *are* the fallback tier — same position
  gperftools-mode internal tcmalloc takes on non-Linux.
- **Effort**: ~2–3 weeks to production quality on Linux, with the payoff
  visible only in Linux benchmarks. Recommended only after item 4 and the
  Linux benchmark matrix exist, since its win cannot be measured on macOS.

## 5. Transfer-cache tier (scoped)

Cheap, portable, and high-value: a per-class spinlocked batch LIFO between
thread caches and pools (16→64 batches, ≤ 1 MiB per class), replacing
direct pool-mutex refills when the recycler misses. qen's refill already
moves `CLASS_BATCH` chains; the transfer cache turns "walk the recycler,
else lock the pool and run alloc_batch_chain" into "memcpy a batch under a
short per-class spinlock". Effort: ~2–3 days. Expected: cuts refill
latency variance and central contention at high thread counts; this is
what keeps internal tcmalloc's central lists off its 16-thread profile.

## 6. Hugepage-aware backend (scoped)

For qen's positioning (games/frame workloads, RSS discipline), full
Temeraire is over-scoped. The high-value subset:

1. **Hugepage-aligned pool blocks on Linux** (`MADV_HUGEPAGE` already
   exists in vm.rs): size pool commit units so hot small-class blocks tile
   2 MiB hugepages; keep cold/large traffic off them (qen's large path is
   already separate mappings — the analogue of the hot/cold bifurcation).
2. **Subrelease hysteresis**: qen's decommit-cooldown (count-based) becomes
   time-window-based (60 s/300 s demand peaks), preventing hugepage
   breakage under bursty load. Small change to the existing three-phase
   trim.
3. Skip regions/filler heaps until fragmentation data from a long-running
   workload says otherwise; qen's 256 MiB per-class reservations already
   give strong locality.

Effort: ~1 week for (1)+(2) with Linux measurement.

## 7. Class/batch/budget tuning (continuous)

Item 2/3 constants were chosen by formula (`64 KiB/bin` batches, 256 KiB
class caps, 16 MiB thread budget — the measured knee on this machine).
Internal tcmalloc's equivalents are per-class curated tables refreshed by
fleet telemetry. Once the mimalloc-bench matrix (below) runs in CI, revisit:
batch taper for 2–16 KiB classes, hot-class deepening (their
2000-vs-144-object split), and budget-vs-thread-count policy.

---

## 8. Measurement plan (how we'll know)

1. **Breadth**: add [mimalloc-bench](https://github.com/daanx/mimalloc-bench)
   (cfrac, espresso, larson, mstress, xmalloc-test, rptest, sh6bench…) next
   to rpmalloc-benchmark — single-suite conclusions overfit; larson is the
   canonical producer/consumer test that item 4 targets.
2. **Linux runs**: an x86-64 and an arm64 Linux box (or CI runners) for the
   per-CPU/hugepage items; macOS numbers cannot see either mechanism.
3. **Against the real target**: build google/tcmalloc via its CMake path on
   the Linux boxes and put it *in* the matrix (the brew gperftools numbers
   in RESULTS.md systematically understate the target).
4. **Beyond throughput**: peak RSS (already tracked), fragmentation over
   long runs (mstress), and tail latency (p99.9 alloc/free under load) —
   internal tcmalloc's wins are partly tail-latency wins; qen must not
   regress its RSS advantage to buy throughput.

## 9. Sequencing and expected position

Priorities re-derived on clean data (2026-07-07). The 16-thread
"structural gap" is gone with the adapter fix; the largest remaining
absolute gap is the **single-thread fast-path constant vs tcmalloc
(1.6–1.9×)**, followed by the smallest-size 16-thread cell and the
1t→16t efficiency drop (3.7× vs rpmalloc's 2.3×), both currently
unprofiled on clean data.

| step | effort | platform | expected effect (evidence-based) |
|---|---|---|---|
| clean 16t profile + smallest-size cell diagnosis | 0.5 d | all | names the real residual scaling term before anything is built against it |
| mimalloc-bench (larson, xmalloc-test, mstress) | 1 d | all | breadth; the evidence gate for un-shelving item 4's masks + adoption |
| fast-path constant (measure → shave toward tcmalloc's hit path) | 1–2 w | all | the 1.6–1.9× single-thread gap — now the biggest prize |
| 5. transfer cache | 2–3 d | all | only if the clean profile names refill costs; no longer presumed |
| 6. hugepage subset | ~1 w | Linux | RSS parity + throughput on Linux; keeps qen's decommit edge |
| per-CPU rseq | 2–3 w | Linux only | the final constant-factor step to internal tcmalloc on its home turf |

Honest bottom line: items 1–3 bought real single-thread ground, and the
adapter fix revealed the multi-thread position was already mid-pack —
second or third past 1 KiB sizes. The road to internal tcmalloc now runs
through the per-op constant (portable work first, rseq on Linux for the
rest), with every step gated on a clean profile naming its target and a
pre-committed falsification bar — the discipline that caught three wrong
designs cheaply and eventually the measurement bug itself.
