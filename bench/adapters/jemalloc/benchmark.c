/* jemalloc adapter (Homebrew libjemalloc, unprefixed API).
 *
 * Homebrew's jemalloc exports plain malloc/free/posix_memalign; linking
 * libjemalloc ahead of libSystem binds these calls to jemalloc under
 * macOS two-level namespacing. benchmark_initialize verifies the binding
 * by querying jemalloc's mallctl("version"). */
#include <benchmark.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

extern int mallctl(const char* name, void* oldp, size_t* oldlenp, void* newp, size_t newlen);

int
benchmark_initialize() {
	const char* version = 0;
	size_t len = sizeof(version);
	if (mallctl("version", &version, &len, 0, 0) != 0 || !version) {
		fprintf(stderr, "jemalloc mallctl(version) failed: not linked against jemalloc?\n");
		return -1;
	}
	fprintf(stderr, "# jemalloc %s\n", version);
	return 0;
}

int
benchmark_finalize(void) {
	return 0;
}

int
benchmark_thread_initialize(void) {
	return 0;
}

int
benchmark_thread_finalize(void) {
	return 0;
}

void
benchmark_thread_collect(void) {
}

void*
benchmark_malloc(size_t alignment, size_t size) {
	if (alignment) {
		void* ptr = 0;
		if (posix_memalign(&ptr, alignment < sizeof(void*) ? sizeof(void*) : alignment, size))
			return 0;
		return ptr;
	}
	return malloc(size);
}

void
benchmark_free(void* ptr) {
	free(ptr);
}

const char*
benchmark_name(void) {
	return "jemalloc";
}
