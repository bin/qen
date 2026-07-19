#!/usr/bin/env bash
# Build rpmalloc-benchmark harness binaries for each allocator under test.
#
# Layout expected (see RESULTS.md):
#   bench/rpmalloc-benchmark/   — the suite (github.com/mjansson/rpmalloc-benchmark)
#   bench/snmalloc/             — snmalloc source (tag 0.7.1)
#   Homebrew: mimalloc, jemalloc, gperftools
set -euo pipefail
cd "$(dirname "$0")"

BREW="$(brew --prefix)"
RB=rpmalloc-benchmark
OUT=build
mkdir -p "$OUT/bin" "$OUT/obj"

CFLAGS="-O3 -DNDEBUG -I$RB/benchmark -I$RB/test"
LDLIBS="-lpthread"

echo "== harness =="
cc $CFLAGS -c "$RB/benchmark/main.c" -o "$OUT/obj/main.o"
cc $CFLAGS -c "$RB/test/thread.c"    -o "$OUT/obj/thread.o"
cc $CFLAGS -c "$RB/test/timer.c"     -o "$OUT/obj/timer.o"
HARNESS="$OUT/obj/main.o $OUT/obj/thread.o $OUT/obj/timer.o"

echo "== qen =="
# The staticlib links against the prebuilt (unwind) std, so force the
# profile back to unwind even if a user-level cargo config says abort.
(cd qen-adapter && CARGO_PROFILE_RELEASE_PANIC=unwind cargo build --release)
cc -o "$OUT/bin/benchmark-qen" $HARNESS \
    qen-adapter/target/release/libqen_adapter.a $LDLIBS

echo "== crt (system malloc) =="
cc $CFLAGS -c "$RB/benchmark/crt/benchmark.c" -o "$OUT/obj/crt.o"
cc -o "$OUT/bin/benchmark-crt" $HARNESS "$OUT/obj/crt.o" $LDLIBS

echo "== jemalloc (brew) =="
cc $CFLAGS -c adapters/jemalloc/benchmark.c -o "$OUT/obj/jemalloc.o"
# libjemalloc before libSystem so two-level namespacing binds malloc/free
# to jemalloc; the adapter verifies via mallctl at startup.
cc -o "$OUT/bin/benchmark-jemalloc" $HARNESS "$OUT/obj/jemalloc.o" \
    -L"$BREW/lib" -ljemalloc $LDLIBS

echo "== mimalloc (brew) =="
cc $CFLAGS -I"$BREW/include" -c adapters/mimalloc/benchmark.c -o "$OUT/obj/mimalloc.o"
cc -o "$OUT/bin/benchmark-mimalloc" $HARNESS "$OUT/obj/mimalloc.o" \
    -L"$BREW/lib" -lmimalloc $LDLIBS

echo "== tcmalloc (brew gperftools) =="
cc $CFLAGS -c "$RB/benchmark/gperftools/benchmark.c" -o "$OUT/obj/tcmalloc.o"
cc -o "$OUT/bin/benchmark-tcmalloc" $HARNESS "$OUT/obj/tcmalloc.o" \
    -L"$BREW/lib" -ltcmalloc_minimal $LDLIBS

echo "== rpmalloc (vendored in suite) =="
cc $CFLAGS -I"$RB/benchmark/rpmalloc" -c "$RB/benchmark/rpmalloc/benchmark.c" -o "$OUT/obj/rpm_bench.o"
cc $CFLAGS -I"$RB/benchmark/rpmalloc" -c "$RB/benchmark/rpmalloc/rpmalloc.c" -o "$OUT/obj/rpm.o"
cc -o "$OUT/bin/benchmark-rpmalloc" $HARNESS "$OUT/obj/rpm_bench.o" "$OUT/obj/rpm.o" $LDLIBS

echo "== snmalloc (source, static shim, sn_ prefix) =="
if [ ! -f "$OUT/snmalloc/libsnmallocshim-static.a" ]; then
    cmake -S snmalloc -B "$OUT/snmalloc" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DSNMALLOC_BUILD_TESTING=OFF >/dev/null
    ninja -C "$OUT/snmalloc" snmallocshim-static
fi
cc $CFLAGS -c adapters/snmalloc/benchmark.c -o "$OUT/obj/snmalloc.o"
c++ -o "$OUT/bin/benchmark-snmalloc" $HARNESS "$OUT/obj/snmalloc.o" \
    "$OUT/snmalloc/libsnmallocshim-static.a" $LDLIBS

echo "== done =="
ls -la "$OUT/bin"
