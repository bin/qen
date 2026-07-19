#!/usr/bin/env bash
# Run the rpmalloc-benchmark matrix (the suite's runall.sh scenarios with
# loop counts scaled down 4x uniformly, and thread counts fitted to this
# machine) across all built allocators. Emits results/raw.log plus the
# per-run CSVs the harness writes.
set -euo pipefail
cd "$(dirname "$0")"

BIN=$PWD/build/bin
OUT=$PWD/results
ALLOCATORS="crt jemalloc tcmalloc mimalloc snmalloc rpmalloc qen"
THREADS="1 2 4 8 16"

mkdir -p "$OUT/runfiles"
cd "$OUT/runfiles"
: > "$OUT/raw.log"

run_scenario() {
    # args: mode size_mode cross_rate loops allocs ops min max
    for alloc in $ALLOCATORS; do
        for t in $THREADS; do
            "$BIN/benchmark-$alloc" "$t" "$@" 2>/dev/null | tee -a "$OUT/raw.log"
        done
    done
}

# Scenarios from rpmalloc-benchmark/runall.sh, loops scaled 4x down:
echo "=== scenario 1: random even [16,1000] ==="        | tee -a "$OUT/raw.log"
run_scenario 0 0 2 5000 50000 5000 16 1000
echo "=== scenario 2: random linear [16,8000] ==="      | tee -a "$OUT/raw.log"
run_scenario 0 1 2 5000 50000 5000 16 8000
echo "=== scenario 3: random linear [16,16000] ==="     | tee -a "$OUT/raw.log"
run_scenario 0 1 2 2500 50000 5000 16 16000
echo "=== scenario 4: random exp [128,64000] ==="       | tee -a "$OUT/raw.log"
run_scenario 0 2 2 2500 30000 3000 128 64000
echo "=== scenario 5: random exp [512,160000] ==="      | tee -a "$OUT/raw.log"
run_scenario 0 2 2 2500 20000 2000 512 160000

echo "=== all scenarios complete ==="                   | tee -a "$OUT/raw.log"
