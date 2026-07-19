# Why tcmalloc consistently outperforms qen — source + profile analysis

Evidence base: gperftools source (vendored in `rpmalloc-benchmark/benchmark/
gperftools/src`, same architecture as the benchmarked brew 2.18.1),
qen source in this tree, and `sample` profiles of both benchmark binaries at
1 and 16 threads on scenario 1 (even [16,1000]).

## The shape of the gap, precisely

| scenario | tcmalloc/qen @1t | @4t | @16t |
|---|---|---|---|
| even [16,1000] | 2.6× | 3.1× | 5.4× |
| linear [16,8000] | 2.1× | 2.3× | 1.8× |
| linear [16,16000] | 2.0× | 1.8× | 1.1× |
| exp [128,64000] | 2.4× | 1.1× | **0.7× (qen wins)** |
| exp [512,160000] | 73× | — | — (qen large-path bug, diagnosed separately) |

tcmalloc's advantage is a **uniform ~2.0–2.6× at 1–2 threads across every
size regime the binned engine serves**. At 16 threads tcmalloc has its own
collapse (central free-list contention) and falls back to qen's level for
mid/large classes. So the question decomposes into: (A) why is tcmalloc's
per-op cost ~2.5× lower, and (B) why does qen's scaling curve sag anyway.

## A. The single-thread constant factor

At 235 M ops/CPU-s, tcmalloc spends ~4.2 ns per op. At that budget the
entire explanation fits in instruction-level bookkeeping.

**tcmalloc's hit path is ~a dozen instructions with zero shared-state
loads.** From `thread_cache.h` / `tcmalloc.cc`: malloc = size→class via one
shift + `class_array_` load, then `FreeList::Pop` on a bare `void* list_`
LIFO (load head, load next, store head), plus a subtraction on the local
byte budget. Free = pagemap radix lookup (2–3 dependent loads) + `Push` +
two `PREDICT_FALSE` guard branches. No atomics, no globals, no layout
validation, no Result plumbing.

**qen's hit path does the same core work buried under per-op layers.** Per
alloc *and* per free (from `binned.rs`):

1. `GlobalBinnedAllocator::get()` — OnceLock atomic load + branch;
2. `GLOBAL_THREAD_CACHE.with` — Rust `thread_local!` via **dynamic TLS**
   (`_tlv_get_addr` is 4% of on-CPU time in the 1-thread profile) + lazy-init
   check;
3. `check_flush()` — an **Acquire load of the global `CACHE_TRIM_EPOCH`**
   on every operation;
4. the large-size guard calls `PlatformVmOps::page_size()` — **another
   OnceLock atomic load per op**;
5. `Layout::from_size_align` validation, `cache.allocator.is_none()` check,
   size-class LUT + alignment loop, `Result` wrap/unwrap.

Each item is a few L1-hot instructions; at a 4-ns budget they sum to the
observed 2–2.6×. The 1-thread profile confirms the locus: **571 top-of-stack
samples inside the `GlobalBinnedAllocator::alloc/free` wrappers vs. 46 in
all pool slow paths combined** — the fast path itself is the cost, not
slow-path frequency alone. (Counterpoint that sharpens it: brew's tcmalloc
runs TLS through `pthread_getspecific` — visible in its own profile — and
still wins. The layer stack, not TLS alone, is the story.)

**Cache depth multiplies slow-path frequency on top of that.** tcmalloc's
per-class thread cache grows adaptively (slow-start) to
`kMaxDynamicFreeListLength = 8192` objects, inside a per-thread budget
starting at 512 KiB (`kMinThreadCacheSize`) that *steals* from a global pool
as demand grows; refills/releases move class-tuned batches
(`num_objects_to_move`) and the central list serves whole batches from a
transfer cache. qen's equivalents are fixed at
`cache_count_limits = [64, 32, 8, 4]` objects and
`alloc_extra = [16, 8, 4, 2]` refill batches. This benchmark keeps 50 000
slots per thread in random churn across ~30 size classes: tcmalloc absorbs
nearly all of it thread-locally; qen structurally cannot, so it crosses
into `alloc_batch` under the pool mutex an order of magnitude more often —
and each refill **allocates a fresh `Vec` for the batch** (a system-malloc
call inside qen's own slow path; `binned.rs:2488`).

## B. The 16-thread sag

The 16-thread qen profile shows ~55% of samples on-CPU inside the
alloc/free wrappers and only ~3% blocked in `psynch_mutexwait` — the
scaling loss is **coherence traffic, not lock waiting**: the recycler cap is
`max_bundles / shards = 16/4 = 4 bundles per shard`, so at a 1-in-2
cross-thread free rate with 16 threads the recycler saturates continuously;
every rejected bundle walks bin-by-bin into a pool mutex, and every
operation's shared-line loads (trim epoch, OnceLock) become cross-core
misses. (The suite's per-CPU-second metric also charges CAS retry spin as
CPU time, which is fair — it is real work the machine does.)

tcmalloc's 16-thread profile shows the mirror-image classical failure —
`SpinLock::SlowLock`, `swtch_pri`, and
`CentralFreeList::FetchFromOneSpans/ReleaseToSpans` dominating — which is
why it falls back to qen's level at 16 threads for mid/large classes. The
allocators that hold 24–52 M ops/CPU-s at 16 threads (rpmalloc, snmalloc,
mimalloc) share the design both qen and tcmalloc lack: **per-thread heaps
with owner-drained remote-free queues** — cross-thread frees are enqueued
to the owning heap and batch-processed by the owner, so no shared structure
is CAS-contended per operation. That is exactly the "full snmalloc message
passing" the README declines; this data quantifies the price of declining
it.

## What holds up for qen

- Lowest or near-lowest peak RSS in every scenario it handles (the
  decommit-with-cooldown discipline is a real, measured advantage).
- jemalloc-class throughput at 1–4 threads; beats tcmalloc outright at 16
  threads on exp [128,64000].
- The adapter's size-recovery tax is measured at ~18% of on-CPU time
  (1-thread) — real, disclosed, and far too small to explain the gap.

## Ranked causes (evidence-backed, no fixes proposed here)

1. **Shallow, static thread caches** (64-object cap, 16-object refill vs.
   adaptive 8192-object lists with batched central transfers) — drives
   slow-path frequency at every thread count.
2. **Per-op constant overhead in the global-allocator wrapper** (trim-epoch
   load, two OnceLock loads — one of them `page_size()` on the size guard —
   dynamic TLS, layout validation) — the single-thread 2× directly.
3. **Recycler capacity (4 bundles/shard) + pool-mutex fallback** — the
   16-thread coherence storm.
4. **`Vec` allocation inside every cache refill** — a malloc inside malloc.
5. **Large-path canonical-size cache policy** — scenario 5, diagnosed in
   RESULTS.md.

Raw profiles: `/tmp/qen-1t.sample`, `/tmp/qen-16t.sample`,
`/tmp/tc-16t.sample` (regenerate via the commands in git history; they are
not committed).

## Addendum (2026-07-06): what fixing causes 1–5 actually bought

This document is the first-run study; it motivated three changes, and the
re-run in [RESULTS.md](./RESULTS.md) is the verdict. Cause by cause:

1. **Shallow static caches** (the headline cause) — replaced with
   slow-start adaptive limits, chain refills, and a byte-budget scavenge.
   Verdict: the big lever, as predicted: +21–44% single-threaded across
   scenarios 1–4, +17–34% at 4 threads.
2. **Per-op wrapper constants** — flattened first, and measured honestly:
   ~+1% alone. The first-run profile overstated this cause because fat LTO
   attributes the entire inlined path (including refill work) to the
   wrapper frames in `sample` output; the layer list was real but its cost
   was mostly cause 1 wearing cause 2's name.
3. **Recycler capacity / coherence at 16 threads** — deliberately not
   addressed (that is roadmap item 4). Verdict confirms the diagnosis: the
   16-thread small-size cells barely moved (+6%) while everything the
   thread cache governs improved.
4. **`Vec` allocation inside refills** — eliminated (intrusive chain
   refills, no heap allocation in the slow path).
5. **Large-path canonical-size policy** — replaced with exact page-count
   buckets + 256 KiB size classes. Verdict: scenario 5 went from 46× behind
   its field to second place at 1 thread (0.9 → 40.9 M ops/CPU-s).

Post-change position vs gperftools tcmalloc: 1.4–1.9× behind at 1 thread
(was 2.0–2.6×), ahead of it at 16 threads in the three mid/large-size
scenarios where its central free-list collapses. The remaining structural
gaps — its 12-instruction hit path, our recycler coherence — are the
per-CPU-cache and remote-free-inbox items in
[ROADMAP-sota.md](./ROADMAP-sota.md).

## Correction (2026-07-07): §B's 16-thread diagnosis was an artifact

Section B attributed qen's 16-thread sag to recycler saturation and
shared-line coherence inside qen. That diagnosis was wrong — the profile
it rested on was taken through a **benchmark-adapter bug**: our
`benchmark_malloc` shim stored to the span table on every allocation,
putting 2–4 cache lines on all sixteen threads' per-op write path. That
storm capped qen at ~9 M ops/CPU-s at 16 threads *independent of
workload* (flat across a 50× working-set range, all size classes, and
with cross-thread frees disabled — the falsification chain is documented
in RESULTS.md). With a first-touch-only store, qen's 16-thread numbers
rose 1.2–3.8× with no allocator change at all.

What survives from §B: tcmalloc's own 16-thread collapse (its profile was
native, no adapter) and the description of the leaders' owner-return
architecture. What does not: every quantitative claim about qen's
recycler being the 16-thread bottleneck, and the sizing of the "price of
declining snmalloc message passing" — on clean data that price is
1.2–1.6× vs rpmalloc past 1 KiB sizes, not 3–6×. The methodological
lesson is recorded where it belongs: an allocator-specific measurement
shim is part of the system under test, and using rival allocators as
controls validates only the shared harness, not the shim.
