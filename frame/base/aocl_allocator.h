
#ifndef BLIS_AOCL_ALLOCATOR_H
#define BLIS_AOCL_ALLOCATOR_H

/*
 * AOCL Allocator
 *
   Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 * ===========================================================================
 *          AOCL Allocator - header-tagged, thread-cached allocator
 * ===========================================================================
 *
 * Scope & threat model: this is a BLIS-internal allocator, not a general-purpose
 * one.  It backs BLIS_MALLOC_{POOL,INTL,USER} and is reached only through BLIS's
 * own memory layers (bli_pba/bli_sba/bli_pool) and object creation, always as
 * paired aocl_malloc/calloc/realloc + aocl_free on pointers this allocator
 * produced -- never with caller-mutated headers or foreign pointers (beyond the
 * documented libc fallback).  Header corruption, double-free, cross-allocator
 * frees, and out-of-range direct calls are therefore not part of correct
 * operation: the header seal (leak on mismatch), free-list link validation, and
 * index bounds below are defense-in-depth that FAILS CLOSED should a memory-safety
 * bug elsewhere reach the allocator -- not a claim such inputs occur in normal use,
 * nor an attempt to be a hardened general-purpose/adversarial allocator.
 *
 * Design (header-based, no slab metadata, no address directory):
 *
 *   - Every allocation carries an 8-byte HEADER immediately before the
 *     pointer handed to the caller: { magic, info }. free() reads ptr[-8],
 *     validates the magic (which also detects double-free and foreign/libc
 *     pointers), and recovers the size class straight from `info`. There is
 *     no radix page map and no per-page slab bookkeeping - the object
 *     describes itself.
 *
 *   - Size regimes:
 *       small  (<= 4 KiB) : power-of-two classes 2^3..2^12, carved from
 *                           mmap'd pallets into header+block slots.
 *       large  (> 4 KiB,
 *               <= 4 MiB) : sized to the exact PAGE MULTIPLE of the request
 *                           (not rounded up to a power of two) and carved from
 *                           a per-thread CONTIGUOUS arena rather than one mmap
 *                           per object, so a thread's large buffers pack into
 *                           shared chunks (dense page tables, compact RSS).
 *                           Cached per page-count.
 *       huge   (> 4 MiB)  : one mmap'd region per object (2 MiB-aligned for a
 *                           transparent hugepage), never cached; its resident
 *                           pages are dropped to the OS on free (MADV_DONTNEED)
 *                           while the mapping is retained, so a stale/double
 *                           free reads a zero header and fails closed rather
 *                           than faulting on an unmapped hole (os_release_huge).
 *
 *   - Per-thread bins (LIFO free lists) make the hot path lock-free: malloc
 *     pops, free pushes. Bins are bounded (per-class count + a per-thread byte
 *     cap). Each pop validates the popped node's link (node_link_ok); its
 *     ownership test is a pair of acquire loads -- lock-free and fence-free on
 *     x86-64, though not strictly atomic-free.
 *
 *   - A shared POOL of per-class free lists is the backstop: thread bins spill
 *     into it on overflow or when a different thread frees an object, and
 *     refill from it on a miss. It is partitioned PER CPU CORE (getcpu__(),
 *     each partition under its own mutex) so recycled memory stays local to the
 *     core -- hence the NUMA node -- that first-touched it, and cores stay off
 *     one another's critical section. This recycles memory across threads and
 *     bounds RSS. Each core's large partition is bounded by a RESIDENT-byte cap
 *     (AOCL_POOL_LARGE_MAX_BYTES): a freed large region is retained for reuse
 *     only while the partition is under the cap; past it the region's pages are
 *     returned to the OS (MADV_DONTNEED, arena VA kept mapped) and it is dropped
 *     from recycling. Without this bound a cross-thread / producer-consumer free
 *     pattern (one core only allocates, carving fresh arena; another only frees,
 *     piling regions onto its own partition, which the allocating core never
 *     draws from) grows RSS without limit, since arena chunks are never
 *     unmapped. The residual cost is address space: an abandoned region's VA is
 *     not reused (its pages are non-resident), so heavy sustained cross-core
 *     churn grows the arena's VA while holding RSS flat -- see the cross-thread
 *     note at large_flush(). Small blocks are likewise retained (they cannot be
 *     sub-page-unmapped without slab metadata, and their footprint is tiny). On
 *     thread exit the bins are drained into the pool so nothing is stranded
 *     under thread churn.
 *
 * Concurrency: the fast path is thread-local only. Pool spill/refill runs under
 * the running core's global_mutex[] entry; fresh mmap growth (pallets, arena
 * chunks, huge regions) happens outside any lock.
 *
 * Ownership (VA watermark): the size of a pointer is recovered by reading its
 * header, which is only safe for memory we produced. Every region we mmap
 * updates a global [global_va_lo, global_va_hi) watermark; free()/realloc()
 * test the pointer against it FIRST. Anything outside the range was not
 * produced here - a foreign pointer, or one of our own libc fallbacks - and
 * is handed straight to libc free()/realloc() without ever touching a header.
 * In-range pointers carry a header whose magic confirms ownership and detects
 * double-free.
 *
 * Header-read safety: the header at ptr-AOCL_HDR_SZ is dereferenced only for
 * in-range pointers, and every pointer we hand out sits >= AOCL_HDR_SZ into its
 * own (mapped) page -- small slots are carved off page boundaries (slab_carve),
 * large/huge sit at +AOCL_LARGE_OFF -- so a live pointer's header always shares
 * the pointer's mapped page. Our regions also stay mapped for the process
 * lifetime -- pallets and large arena chunks are never unmapped, and a freed
 * huge region keeps its mapping (only its pages are dropped; see
 * os_release_huge) -- so an in-range address is never an unmapped hole and a
 * stale/double free reads a (zeroed) header and fails closed rather than
 * faulting. free()/realloc() additionally reject any in-range
 * pointer within AOCL_HDR_SZ of a page start WITHOUT dereferencing it, so the
 * read can never touch an unmapped preceding page (the "pointer at the start of
 * a mapping" case). The only residual effect of the coarse range is that a
 * FOREIGN pointer landing in-range has its (mapped) header read to be rejected
 * by the magic check -- memory-safe, though a tool such as Valgrind/ASan will
 * note the read of a neighbouring allocation's bytes; eliminating even that
 * would require a precise region map rather than a min/max watermark.
 *
 * libc fallback: on true OOM (mmap refused - e.g. address space or
 * vm.max_map_count exhausted) aocl_malloc() falls back to an alignment-
 * preserving libc allocation (posix_memalign to AOCL_MAX_ALIGN), so a fallback
 * pointer carries the same alignment guarantee as a native one. It almost always
 * maps OUTSIDE our watermark, so free()/realloc() route it to libc via the range
 * test above and it stays transparent to the caller. The lone exception is a
 * fallback that happens to map INSIDE the watermark: a page-aligned one (the
 * common shape of an mmap'd libc chunk) still routes to libc via the page-start
 * guard below, but a non-page-aligned one reaches the header read, fails the seal
 * and -- being indistinguishable from a corrupted in-range native pointer -- is
 * LEAKED rather than freed (an in-range seal mismatch never calls libc free(),
 * see the seal note below). Leaking such a rare OOM-time fallback is the
 * accepted, safer trade against munmap'ing a corruption-derived native address.
 *
 * Header integrity (seal): the two header words are not independent - `magic` is
 * `base ^ hdr_mix(info)` (see hdr_seal), binding the live/freed sentinel to the
 * size class. A stray write that flips `info` (e.g. a size-class bit) therefore
 * also breaks `magic`, so free()/realloc() detect the inconsistency instead of
 * reusing a mis-sized block. On any such mismatch the allocator FAILS CLOSED: it
 * LEAKS the block (or traps, under AOCL_CORRUPT_ABORT) rather than trust a size it
 * cannot verify, and in no case hands a corrupted, still-in-range NATIVE pointer
 * to libc free() - doing so could
 * reach glibc's IS_MMAPPED path and munmap a corruption-derived address. It does
 * NOT try to recover a class from `magic` and recycle the block: the seal is
 * address-independent, so a coincidental magic match is indistinguishable from a
 * genuine object, and re-arming a suspect pointer onto a live free list is
 * strictly more dangerous than leaking it. Every hand-out re-stamps both words
 * together (hdr_stamp) so a reuse path can never leave them disagreeing. This is
 * defense-in-depth, not a guarantee - a corruption self-consistent across both
 * words still passes.
 *
 * Free-list integrity: a cached object threads its next-free link through its
 * own payload, so a use-after-free (or a neighbour overflow into a parked
 * header) can corrupt the list. Every pop validates the node before trusting
 * the link (node_link_ok): its FREED seal must hold and the link must be NULL
 * or a plausible successor (link_valid). On a mismatch the allocator fails
 * closed - truncate the bad node's chain (bounded leak) and refill fresh, or
 * trap under AOCL_CORRUPT_ABORT. Like the seal, this detects/contains
 * corruption rather than preventing it.
 * ===========================================================================
 */

#ifdef BLIS_ENABLE_AOCL_ALLOC

/*
 * The allocator relies on operating-system facilities that a --disable-system
 * build strips out: pthreads (per-core pool mutexes and the thread-exit flush
 * hook) and mmap/madvise/munmap (the entire backing store). Enabling the
 * allocator without system support is therefore an invalid configuration. The
 * configure and CMake gates already refuse to turn it on when system support
 * is off; this compile-time guard fails fast if such a combination is forced
 * through anyway (e.g. a hand-defined BLIS_ENABLE_AOCL_ALLOC), rather than
 * compiling partway and breaking later or silently violating the
 * BLIS_DISABLE_SYSTEM contract.
 */
#if defined(BLIS_DISABLE_SYSTEM)
  #error "BLIS_ENABLE_AOCL_ALLOC requires operating system support (BLIS_ENABLE_SYSTEM): pthreads + mmap."
#endif

#if defined(__GNUC__) || defined(__clang__)
  #define AOCL_ALLOC_LOCAL __attribute__((visibility("hidden")))
#else
  #define AOCL_ALLOC_LOCAL
#endif

#if !defined(BLIS_OS_LINUX)

/*
 * Non-Linux passthrough. The same public symbols are exposed everywhere; off
 * Linux they delegate to libc so BLIS_MALLOC_POOL/_INTL/_USER resolve
 * uniformly. Bodies are emitted only in the EXPORT_AOCL_ALLOCATOR TU.
 */
#include <stdlib.h>

AOCL_ALLOC_LOCAL void * aocl_malloc(size_t size);
AOCL_ALLOC_LOCAL void   aocl_free(void *ptr);
AOCL_ALLOC_LOCAL void * aocl_calloc(size_t nmemb, size_t size);
AOCL_ALLOC_LOCAL void * aocl_realloc(void *ptr, size_t newsize);
AOCL_ALLOC_LOCAL void   aocl_alloc_ini(void);
AOCL_ALLOC_LOCAL void   aocl_alloc_fini(void);

#ifdef EXPORT_AOCL_ALLOCATOR

AOCL_ALLOC_LOCAL void *aocl_malloc(size_t size)
{
    return malloc( size );
}

AOCL_ALLOC_LOCAL void aocl_free(void *ptr)
{
    free( ptr );
}

AOCL_ALLOC_LOCAL void *aocl_calloc(size_t nmemb, size_t size)
{
    return calloc( nmemb, size );
}

AOCL_ALLOC_LOCAL void *aocl_realloc(void *ptr, size_t newsize)
{
    return realloc( ptr, newsize );
}

AOCL_ALLOC_LOCAL void aocl_alloc_ini(void)
{
    /* nothing to initialise */
}

AOCL_ALLOC_LOCAL void aocl_alloc_fini(void)
{
    /* nothing to tear down */
}

#endif /* EXPORT_AOCL_ALLOCATOR */

#else /* BLIS_OS_LINUX */

#include <stddef.h>     /* size_t for the prototypes below */

AOCL_ALLOC_LOCAL void * aocl_malloc(size_t size);
AOCL_ALLOC_LOCAL void   aocl_free(void *ptr);
AOCL_ALLOC_LOCAL void * aocl_calloc(size_t nmemb, size_t size);
AOCL_ALLOC_LOCAL void * aocl_realloc(void *ptr, size_t newsize);
AOCL_ALLOC_LOCAL void   aocl_alloc_ini(void);
AOCL_ALLOC_LOCAL void   aocl_alloc_fini(void);

#ifdef EXPORT_AOCL_ALLOCATOR

#include <stdlib.h>     /* libc malloc/free: OOM / foreign fallback */
#include <string.h>     /* memcpy, memset                          */
#include <stdint.h>     /* uintN_t, SIZE_MAX                        */
#include <limits.h>     /* CHAR_BIT                                 */
#include <pthread.h>    /* per-core mutexes + thread-exit key       */
#include <sys/mman.h>   /* mmap, munmap, madvise                    */

/* ------------------------------------------------------------------------- *
 *                          Compile-time knobs                               *
 * ------------------------------------------------------------------------- */

/* The allocator assumes a 4 KiB base page.  BLIS gates it to x86_64 at
   configure time (configure: "aocl allocator is only supported on x86_64"),
   where the base page is architecturally 4 KiB.  Enforce that here so a
   mis-gated build fails at compile time instead of silently mismatching the
   kernel's mmap granularity at runtime. */
#if !defined(__x86_64__) && !defined(__amd64__)
  #error "aocl_allocator.h supports only x86_64 (4 KiB base page)."
#endif

/* The implementation relies on GNU/Clang extensions (__thread TLS, __atomic
   builtins, __builtin_*, and _GNU_SOURCE mmap flags).  BLIS gates the feature
   to gcc/clang at configure/CMake time; enforce the same here so a build that
   force-defines BLIS_ENABLE_AOCL_ALLOC on an unsupported compiler fails fast
   with a clear message instead of a cascade of missing-symbol errors.  Note
   classic Intel ICC defines __GNUC__, so it is rejected explicitly. */
#if defined(__INTEL_COMPILER) && !defined(__INTEL_LLVM_COMPILER)
  #error "aocl_allocator.h requires GCC or Clang; Intel classic ICC is not supported."
#endif
#if !defined(__GNUC__) && !defined(__clang__)
  #error "aocl_allocator.h requires GCC or Clang (TLS + __atomic builtins)."
#endif

#define PAGE_BYTES        ((size_t)4096)
#define PAGE_SHIFT        12
#define AOCL_MIN_IDX      3           /* smallest class = 2^3 = 8 bytes        */
#define AOCL_SMALL_IDX    12          /* <= 2^12 (4 KiB) served from pallets   */
#define AOCL_SMALL_MAX    ((size_t)1 << AOCL_SMALL_IDX)      /* 4 KiB          */
#define AOCL_MAX_PAGES    1024        /* cache large regions up to 4 MiB       */
#define AOCL_LARGE_OFF    ((size_t)64)/* alignment reserve before large user   */
#define AOCL_MAX_ALIGN    ((size_t)64)/* max alignment we hand out (cache line) */

#define AOCL_BATCH_MAX    64          /* refill/drain batch bound              */
#define AOCL_LARGE_BATCH  16          /* large refill/drain batch              */
/* Upper bound on hardware threads the recycle pool partitions across.  Reused
   from the initial #358 allocator (getcpu__ folds the CPU id modulo this).
   Override from the build system on machines with > 512 logical CPUs. */
#ifndef MAX_CPU_CORES
#define MAX_CPU_CORES     512
#endif
#ifndef AOCL_TC_CAP_LARGE
#define AOCL_TC_CAP_LARGE 32          /* cached large regions per class/thread */
#endif
#ifndef AOCL_TC_MAX_BYTES
#define AOCL_TC_MAX_BYTES ((size_t)64 << 20)  /* per-thread cache cap: 64 MiB  */
#endif
/* Cap on RESIDENT bytes retained in each per-core shared LARGE recycle pool.
   The per-thread bins are already bounded (AOCL_TC_CAP_LARGE / AOCL_TC_MAX_BYTES);
   this bounds the shared backstop they spill into.  It exists to contain the
   cross-thread / producer-consumer free pattern: when one core only allocates
   and another core only frees, freed regions pile onto the freeing core's
   partition -- which the allocating core never draws from -- and, because arena
   chunks are never unmapped, their resident pages would otherwise accumulate
   without bound.  Past the cap a spilled region's pages are dropped
   (MADV_DONTNEED) and the region is removed from recycling, so retained RSS per
   core is bounded.  Set generously so the common (bounded-churn) case never
   trips it and same-core reuse is unaffected; override from the build system to
   tune. */
#ifndef AOCL_POOL_LARGE_MAX_BYTES
#define AOCL_POOL_LARGE_MAX_BYTES ((size_t)128 << 20)  /* 128 MiB per core */
#endif
/* Master switch for transparent-hugepage (THP) advice on the large/huge backing
   store.  Default OFF.  Advising MADV_HUGEPAGE on buffers > 4 MiB -- notably the
   user matrices routed through BLIS_MALLOC_USER by bli_obj_create() -- gives them
   2 MiB of physically contiguous backing.  With a power-of-two leading dimension
   (e.g. m=n=k = 1024/2048) that contiguity aliases successive tiles onto the same
   L2 cache sets, so demand loads suffer associativity/conflict misses (serviced
   from L3, not DRAM) and mid-size DGEMM regresses 30-50%.  The TLB upside is
   marginal at sizes that are already bandwidth-bound (e.g. 4096).  Measured and
   root-caused (perf + AMDuProf IBS-op) in designnotes_aoclAllocator.md.  Define
   AOCL_ENABLE_THP=1 to opt back in to the previous hugepage behavior. */
#ifndef AOCL_ENABLE_THP
#define AOCL_ENABLE_THP   0
#endif
#ifndef AOCL_HUGE_ADVISE
#define AOCL_HUGE_ADVISE  ((size_t)2 << 20)   /* MADV_HUGEPAGE at/above 2 MiB  */
#endif
#ifndef AOCL_HUGE_ALIGN
#define AOCL_HUGE_ALIGN   ((size_t)2 << 20)   /* 2 MiB THP boundary, huge path  */
#endif
/* Governs only the uncached huge path (allocations too big to recycle, i.e.
   np > AOCL_MAX_PAGES).  When enabled, each such region is backed by a 2 MiB-
   aligned map advised MADV_HUGEPAGE (one TLB entry, fewer page-table walks);
   otherwise it falls back to an exact 4 KiB-paged map.  Defaults to
   AOCL_ENABLE_THP (i.e. OFF) -- see the THP note above for why.  Cacheable large
   regions do NOT use this -- they pack into the per-thread arena described below. */
#ifndef AOCL_LARGE_HUGE
#define AOCL_LARGE_HUGE   AOCL_ENABLE_THP
#endif
/* Cacheable large regions are carved from a per-thread CONTIGUOUS arena instead
   of one mmap per region.  Packing a thread's large buffers into shared chunks
   keeps its page tables dense (few TLB entries, no per-buffer VA scatter) and
   its resident set compact -- matching libc's arena and removing the high-core-
   count straggler tail that per-region maps produce.  This is the VA size of one
   arena chunk; only touched pages are resident (4 KiB, lazily faulted). */
#ifndef AOCL_ARENA_CHUNK
#define AOCL_ARENA_CHUNK  ((size_t)8 << 20)
#endif

/* small-class carving policy (consumed by global_setup) */
#ifndef AOCL_PALLET_MAX
#define AOCL_PALLET_MAX    ((size_t)256 << 10)  /* max bytes per pallet: 256 KiB */
#endif
#define AOCL_SMALL_CAP_MIN 64    /* per-thread small count cap: floor   */
#define AOCL_SMALL_CAP_MAX 4096  /* per-thread small count cap: ceiling */

/* Header sentinels.  Stored `magic` is a base XORed with a check digit over
   `info` (see hdr_seal), so the two are bound; they also flag foreign pointers
   and double-free. */
#define AOCL_MAGIC_ALLOC  0xA10CA10Cu         /* live-object base sentinel  */
#define AOCL_MAGIC_FREED  0xF7EEF7EEu         /* freed-object base sentinel */
#define AOCL_LARGE_BIT    0x80000000u         /* set in info for large/huge    */

/* Counter-measure when corruption is detected -- a header-seal mismatch
   (aocl_free/aocl_realloc) or a corrupt free-list link (see node_link_ok).  By
   default fail closed (leak the suspect block); define AOCL_CORRUPT_ABORT to trap
   instead (hardened/debug builds).  AOCL_LINK_CHECK_ABORT is accepted as a
   back-compat alias for the same knob. */
#if defined(AOCL_CORRUPT_ABORT) || defined(AOCL_LINK_CHECK_ABORT)
  #define AOCL_ON_CORRUPT() __builtin_trap()
#else
  #define AOCL_ON_CORRUPT() ((void)0)
#endif

#if defined(__GNUC__) || defined(__clang__)
  #define aocl_likely(x)   __builtin_expect(!!(x), 1)
  #define aocl_unlikely(x) __builtin_expect(!!(x), 0)
#else
  #define aocl_likely(x)   (x)
  #define aocl_unlikely(x) (x)
#endif

#define AOCL_AUP(x, a)    (((x) + ((a) - 1)) & ~((size_t)(a) - 1))

/* ------------------------------------------------------------------------- *
 *                            Per-object header                              *
 * ------------------------------------------------------------------------- *
 * Sits in the 8 bytes immediately before the returned pointer. `info` is
 * either a small size-class index (3..12) or, with AOCL_LARGE_BIT set, the
 * page count of the backing region. The free list `next` pointer for a
 * cached object lives in the object's own payload (>= 8 bytes), so the
 * header is untouched while an object sits in a bin/pool.
 */
typedef struct
{
    uint32_t magic;
    uint32_t info;
} aocl_hdr_t;

#define AOCL_HDR_SZ ((size_t)sizeof(aocl_hdr_t))   /* 8 */

static inline aocl_hdr_t *hdr_of(void *user)
{
    return (aocl_hdr_t *)( (char *)user - AOCL_HDR_SZ );
}

/* Header seal: bind `magic` to `info` (magic = base ^ hdr_mix(info)) so a stray
   write flipping the size class makes the pair inconsistent and is rejected at
   free()/realloc() instead of returning a mis-sized block.  Cheap bijection
   (odd multiply + xorshift); defense-in-depth, not a guarantee (a self-
   consistent corruption still passes). */
static inline uint32_t hdr_mix(uint32_t info)
{
    uint32_t x = info * 0x9E3779B1u;   /* odd constant -> invertible mod 2^32 */
    return x ^ (x >> 16);
}
static inline uint32_t hdr_seal(uint32_t base, uint32_t info)
{
    return base ^ hdr_mix(info);
}

/* Stamp a header consistently: write `info` and its sealed `magic` together.
   Every hand-out (fresh carve and cache/pool reuse) stamps through here; free
   rewrites only `magic` (against the object's unchanged `info`, so the pair stays
   sealed).  No path can refresh one word against the assumed class while leaving
   the other stale/possibly-corrupted (which would only detonate on that object's
   own later free). */
static inline void hdr_stamp(void *user, uint32_t base, uint32_t info)
{
    aocl_hdr_t *h = hdr_of(user);
    h->info  = info;
    h->magic = hdr_seal(base, info);
}

/* ------------------------------------------------------------------------- *
 *                              Shared state                                 *
 * ------------------------------------------------------------------------- */

/* One shared recycle pool per CPU core so memory stays local to the core (and
   therefore the NUMA node) whose thread first-touched it -- the core-local
   scheme from the initial #358 allocator.  A thread drains to / refills from
   the pool of the core it is currently running on (getcpu__()); the per-core
   lock also keeps threads on other cores off each other's critical section.
   The per-thread TLS bins remain the lock-free hot path; this pool is only the
   slow-path spill/refill between a core's bins and fresh mmap.  Mutexes are
   statically initialised via a GNU range designator so every lock is valid
   before global_setup runs (the large path locks without pthread_once). */
static pthread_mutex_t global_mutex[MAX_CPU_CORES] =
    { [0 ... MAX_CPU_CORES - 1] = PTHREAD_MUTEX_INITIALIZER };

/* Shared recycle pool: per-core, per-class intrusive free lists (the link is
   threaded through each freed object's own payload).  The SMALL partitions are
   uncapped (blocks are sub-page and share pallets that cannot be partially
   unmapped; their footprint is tiny).  The LARGE partitions are bounded by a
   per-core resident-byte cap (AOCL_POOL_LARGE_MAX_BYTES, enforced in
   large_flush) so a cross-thread free pattern cannot grow RSS without bound;
   global_pool_large_bytes[] tracks the bytes currently retained per core (only
   ever touched under that core's global_mutex[]). */
static void  *global_pool_small[MAX_CPU_CORES][AOCL_SMALL_IDX + 1];
static void  *global_pool_large[MAX_CPU_CORES][AOCL_MAX_PAGES + 1];
static size_t global_pool_large_bytes[MAX_CPU_CORES];  /* resident bytes retained */

/* VA watermark: min/max over every region we mmap. A pointer outside
 * [global_va_lo, global_va_hi) was not produced here (foreign, or one of our
 * own libc fallbacks) and is routed to libc without dereferencing an absent
 * header. */
static uintptr_t global_va_lo = UINTPTR_MAX;
static uintptr_t global_va_hi = 0;

/* per-class derived tables (built once in global_setup) */
static size_t global_block[AOCL_SMALL_IDX + 1];   /* usable object size 2^idx */
static size_t global_align[AOCL_SMALL_IDX + 1];   /* min(2^idx, 64) */
static size_t global_stride[AOCL_SMALL_IDX + 1];  /* hdr+block+pad, aligned */
static size_t global_pallet[AOCL_SMALL_IDX + 1];  /* pallet size to carve */
static int    global_small_cap[AOCL_SMALL_IDX + 1];   /* per-thread count cap */
static int    global_small_batch[AOCL_SMALL_IDX + 1]; /* refill/drain batch */

static pthread_once_t global_setup_once = PTHREAD_ONCE_INIT;

/* thread-exit hook */
static pthread_key_t  thread_cache_key;
static pthread_once_t thread_cache_key_once = PTHREAD_ONCE_INIT;

/* ------------------------------------------------------------------------- *
 *                               Thread bins                                 *
 * ------------------------------------------------------------------------- */

static __thread void  *thread_cache_small[AOCL_SMALL_IDX + 1];
static __thread int    thread_cache_small_cnt[AOCL_SMALL_IDX + 1];
static __thread void  *thread_cache_large[AOCL_MAX_PAGES + 1];
static __thread int    thread_cache_large_cnt[AOCL_MAX_PAGES + 1];
static __thread size_t thread_cache_bytes;
static __thread int    thread_cache_reg;

/* ------------------------------------------------------------------------- *
 *                                OS layer                                   *
 * ------------------------------------------------------------------------- */

/* Extend the ownership watermark to cover [p, p+bytes). Lock-free (atomic
 * min/max) so it is safe from every mmap site regardless of global_mutex. */
static void va_note(void *p, size_t bytes)
{
    uintptr_t a = (uintptr_t)p, e = a + bytes, old;

    old = __atomic_load_n(&global_va_lo, __ATOMIC_RELAXED);
    while ( a < old && !__atomic_compare_exchange_n(&global_va_lo, &old, a, 0,
                            __ATOMIC_RELEASE, __ATOMIC_RELAXED) )
    { }
    old = __atomic_load_n(&global_va_hi, __ATOMIC_RELAXED);
    while ( e > old && !__atomic_compare_exchange_n(&global_va_hi, &old, e, 0,
                            __ATOMIC_RELEASE, __ATOMIC_RELAXED) )
    { }
}

/* True only if ptr falls inside memory this allocator has mapped. */
static inline int va_ours(const void *ptr)
{
    uintptr_t p = (uintptr_t)ptr;
    return p >= __atomic_load_n(&global_va_lo, __ATOMIC_ACQUIRE)
        && p <  __atomic_load_n(&global_va_hi, __ATOMIC_ACQUIRE);
}

/* getcpu__ - cheap, approximate index of the CPU the caller runs on.  Reused
   from the initial #358 allocator.  On x86_64 Linux the kernel programs
   IA32_TSC_AUX so RDTSCP returns (numa_node << 12 | cpu_id) in ECX; the low 12
   bits are the CPU id.  RDTSCP is a single instruction (no syscall, no vDSO
   call), so this is cheap enough for the slow pool path.  The value only
   selects a per-core recycle pool: a stale id after a migration costs at most
   one remote reuse and never affects correctness (any core may reuse any
   object; the header is self-describing). */
static inline int getcpu__(void)
{
    /* x86_64 guaranteed by the #error gate above; no non-x86 fallback needed. */
    unsigned int ax, cx, dx;
    __asm__ __volatile__("rdtscp" : "=a"(ax), "=d"(dx), "=c"(cx) ::);
    (void)ax; (void)dx;
    return (int)(cx & 0xfffu) % MAX_CPU_CORES;
}

static void *os_map(size_t bytes)
{
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if ( p == MAP_FAILED )
    {
        return NULL;
    }
#if AOCL_ENABLE_THP
#ifdef MADV_HUGEPAGE
    if ( bytes >= AOCL_HUGE_ADVISE )
    {
        (void)madvise(p, bytes, MADV_HUGEPAGE);
    }
#endif
#endif
    va_note(p, bytes);
    return p;
}

static void os_unmap(void *p, size_t bytes)
{
    (void)munmap(p, bytes);
}

/* Release an uncached HUGE region on free.  munmap() would return the VA to the
   OS, leaving a HOLE inside our monotonic [global_va_lo, global_va_hi) ownership
   watermark: a later double-free or dangling free/realloc of the same pointer
   would then pass va_ours(), read a header on the now-unmapped page, and SEGV --
   the one path in this allocator that faults instead of failing closed.  Drop
   only the RESIDENT pages with MADV_DONTNEED and KEEP the mapping: RSS is
   returned to the OS immediately (the whole point of the uncached huge path),
   while the VA stays mapped and zero-filled, so a stale free reads a zero header,
   misses the seal, and leaks -- consistent with the fail-closed posture used
   everywhere else.  The VA (one VMA per live-or-retired huge region) is retained
   for the process lifetime; the count of DISTINCT huge regions is bounded in
   practice by BLIS's own buffer pooling above this allocator.  If MADV_DONTNEED
   is unavailable we fall back to munmap (restoring the prior, faulting behaviour
   only on that exotic configuration). */
static void os_release_huge(void *p, size_t bytes)
{
#ifdef MADV_DONTNEED
    if ( aocl_likely(madvise(p, bytes, MADV_DONTNEED) == 0) )
    {
        return;
    }
#endif
    os_unmap(p, bytes);
}

/* Return the RESIDENT pages of an arena-backed large region to the OS while
   KEEPING its VA mapped.  The per-thread large arena chunks are shared and never
   unmapped (a region may be recycled by another thread), so an individual region
   cannot be munmap'd; MADV_DONTNEED drops its pages (RSS returned, zero-filled on
   any later fault) without disturbing neighbouring regions in the same chunk or
   creating a new VMA.  Returns 1 iff RSS was released.  Used by large_flush() to
   bound the shared pool under cross-thread free; if MADV_DONTNEED is unavailable
   the caller retains the region (no reclaim) rather than orphan it. */
static int os_purge(void *p, size_t bytes)
{
#ifdef MADV_DONTNEED
    return madvise(p, bytes, MADV_DONTNEED) == 0;
#else
    (void)p; (void)bytes;
    return 0;
#endif
}

/* Backing byte size of a large/huge region of np pages: the exact 4 KiB-page
   span, never rounded up.  Cacheable large regions pack into the per-thread
   arena, where rounding (e.g. a 150 KiB buffer up to 2 MiB) would inflate RSS
   and scatter the working set out of cache.  The uncached huge path gets its
   2 MiB THP alignment separately, inside os_map_large(). */
static inline size_t large_map_bytes(uint32_t np)
{
    return (size_t)np << PAGE_SHIFT;
}

/* ------------------------------------------------------------------------- *
 *                        Free-list link integrity                           *
 * ------------------------------------------------------------------------- */

/* A next-free link must be NULL (list end) or a plausible successor: in our mmap
   watermark, AOCL_HDR_SZ-aligned, and >= AOCL_HDR_SZ into its page (every real
   node is; a stray write almost never satisfies all three). */
static inline int link_valid(const void *next)
{
    uintptr_t p = (uintptr_t)next;

    if ( next == NULL )                            return 1;
    if ( !va_ours(next) )                          return 0;
    if ( (p & (AOCL_HDR_SZ - 1)) != 0 )            return 0;
    if ( (p & (PAGE_BYTES - 1)) < AOCL_HDR_SZ )    return 0;
    return 1;
}

/* Validate a freed node of class `info` before its link is trusted, reading the
   link into *next.  Returns 1 iff the node still carries this class's FREED seal
   (header intact) and the link passes link_valid(). */
static inline int node_link_ok(void *node, uint32_t info, void **next)
{
    aocl_hdr_t *h;

    *next = NULL;
    /* Validate the node pointer itself before dereferencing it.  A bin/pool head
       slot can be scribbled directly (not only via an in-payload link), so guard
       the head read the same way link_valid() guards a successor -- otherwise a
       corrupted head would be an unguarded wild read of hdr_of(node) or *node.
       Every genuine node satisfies link_valid (in-watermark, 8-aligned, off its
       page start); callers pass a non-NULL head. */
    if ( aocl_unlikely(!link_valid(node)) )
    {
        return 0;
    }
    h     = hdr_of(node);
    *next = *(void **)node;
    if ( aocl_unlikely(h->magic != hdr_seal(AOCL_MAGIC_FREED, info)) )
    {
        return 0;
    }
    return link_valid(*next);
}

/* Pop one validated node from a thread bin (NULL if empty).  On detected
   corruption, fail closed: abandon the whole bin (report empty so the caller
   refills fresh) rather than follow the bad link.  `info` is the class seal to
   check; `bytesz` is the per-object size for the byte accounting. */
static inline void *tc_pop(void **head, int *cnt, uint32_t info, size_t bytesz)
{
    void *u = *head;
    void *next;

    if ( u == NULL )
    {
        return NULL;
    }
    if ( aocl_unlikely(!node_link_ok(u, info, &next)) )
    {
        AOCL_ON_CORRUPT();
        thread_cache_bytes -= (size_t)(*cnt) * bytesz;
        *head = NULL;
        *cnt  = 0;
        return NULL;
    }
    *head = next;
    (*cnt)--;
    thread_cache_bytes -= bytesz;
    return u;
}

/* Pop one validated node from this thread's small bin `idx` (NULL if empty). */
static inline void *tc_small_pop(int idx)
{
    return tc_pop(&thread_cache_small[idx], &thread_cache_small_cnt[idx],
                  (uint32_t)idx, global_block[idx]);
}

/* Large-bin analogue of tc_small_pop(). */
static inline void *tc_large_pop(uint32_t np)
{
    return tc_pop(&thread_cache_large[np], &thread_cache_large_cnt[np],
                  AOCL_LARGE_BIT | np, large_map_bytes(np));
}

/* mmap the backing store for one uncached huge region (the np > AOCL_MAX_PAGES
   path; cacheable large regions come from the arena instead).  With
   AOCL_LARGE_HUGE the mapping is forced onto a 2 MiB boundary (over-map one
   alignment unit and trim the head/tail) and advised MADV_HUGEPAGE, so it lands
   on contiguous hugepages -- one TLB entry and far fewer page-table walks than
   a 4 KiB-paged region.  bytes must come from large_map_bytes(). */
static void *os_map_large(size_t bytes)
{
#if AOCL_LARGE_HUGE
    size_t    over = bytes + AOCL_HUGE_ALIGN;
    char     *p    = (char *)mmap(NULL, over, PROT_READ | PROT_WRITE,
                                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    uintptr_t base, aligned;
    size_t    head, tail;

    if ( aocl_unlikely(p == MAP_FAILED) )
    {
        return NULL;
    }
    base    = (uintptr_t)p;
    aligned = AOCL_AUP(base, AOCL_HUGE_ALIGN);
    head    = (size_t)(aligned - base);
    if ( head )
    {
        (void)munmap(p, head);                 /* drop slack before the region */
    }
    tail = over - head - bytes;
    if ( tail )
    {
        (void)munmap((char *)(aligned + bytes), tail);  /* and after it */
    }
#ifdef MADV_HUGEPAGE
    (void)madvise((void *)aligned, bytes, MADV_HUGEPAGE);
#endif
    va_note((void *)aligned, bytes);
    return (void *)aligned;
#else
    return os_map(bytes);
#endif
}

/* Plain anonymous map: 4 KiB paged, lazily faulted, NO madvise/THP.  Backs the
   per-thread large arena so a thread's large buffers pack contiguously (dense
   page tables, compact RSS) rather than scattering one mmap per buffer. */
static void *os_map_plain(size_t bytes)
{
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if ( aocl_unlikely(p == MAP_FAILED) )
    {
        return NULL;
    }
    va_note(p, bytes);
    return p;
}

/* Per-thread bump arena for cacheable large regions.  Chunks are never
   unmapped during the run: a large region may be recycled by another thread via
   the shared pool, so its backing chunk must stay mapped for ownership to
   remain valid.  The OS reclaims all chunks at process exit.  RSS is bounded by
   the actual large-buffer high-water mark (only touched 4 KiB pages fault in),
   not by the VA reserved. */
static __thread char  *arena_cur;   /* bump pointer within current chunk       */
static __thread size_t arena_left;  /* bytes remaining in current chunk        */

/* Reserve `bytes` (always a page multiple: large_map_bytes(np)) of contiguous
   arena space.  os_map_plain() returns a page-aligned base and every region is
   a running sum of page-multiple sizes, so each carved base is page-aligned --
   hence 64B-aligned -- and region + AOCL_LARGE_OFF is the 64B-aligned user
   pointer the large path hands out. */
static char *arena_carve(size_t bytes)
{
    if ( arena_left < bytes )
    {
        size_t csz = AOCL_ARENA_CHUNK;
        char  *c;

        if ( bytes > csz )                    /* region larger than a chunk */
        {
            csz = AOCL_AUP(bytes, PAGE_BYTES);
        }
        c = (char *)os_map_plain(csz);
        if ( aocl_unlikely(c == NULL) )
        {
            return NULL;
        }
        arena_cur  = c;
        arena_left = csz;
    }
    {
        char *r = arena_cur;
        arena_cur  += bytes;
        arena_left -= bytes;
        return r;
    }
}

/* Alignment-preserving libc fallback.  A native allocation is aligned to at
   least AOCL_MAX_ALIGN; on OOM we must honor that too, because a caller cannot
   distinguish a fallback pointer from a native one.  free()/realloc() route
   these back to libc: they almost always fall outside the VA watermark (the rare
   in-range one is caught by the header-read guards -- see the top-of-file
   libc-fallback note). */
static void *os_fallback(size_t bytes)
{
    void *p = NULL;
    return posix_memalign(&p, AOCL_MAX_ALIGN, bytes) == 0 ? p : NULL;
}

/* fork() safety.  If one thread forks while another holds global_mutex, the
   child inherits the lock held forever and deadlocks on its next pool access.
   These handlers serialize fork against the pool lock and release it in both
   the parent and the child after fork().  The per-thread TLS bins need no
   handling: only the forking thread survives in the child and its bins are
   self-consistent.  Registered once (global_setup). */
static void aocl_atfork_prepare(void)
{
    int c;
    for ( c = 0; c < MAX_CPU_CORES; c++ )      /* ascending: consistent order */
    {
        pthread_mutex_lock(&global_mutex[c]);
    }
}

static void aocl_atfork_parent(void)
{
    int c;
    for ( c = 0; c < MAX_CPU_CORES; c++ )
    {
        pthread_mutex_unlock(&global_mutex[c]);
    }
}

static void aocl_atfork_child(void)
{
    int c;
    for ( c = 0; c < MAX_CPU_CORES; c++ )
    {
        pthread_mutex_unlock(&global_mutex[c]);
    }
}

static void global_setup(void)
{
    int idx;

    for ( idx = AOCL_MIN_IDX; idx <= AOCL_SMALL_IDX; idx++ )
    {
        size_t block  = (size_t)1 << idx;
        size_t align  = block < AOCL_MAX_ALIGN ? block : AOCL_MAX_ALIGN;
        size_t stride = AOCL_AUP(block + AOCL_HDR_SZ, align);
        size_t pallet = AOCL_AUP((size_t)64 * stride, PAGE_BYTES);
        size_t slots;

        if ( pallet > AOCL_PALLET_MAX )
        {
            pallet = AOCL_PALLET_MAX;
        }
        if ( pallet < PAGE_BYTES )
        {
            pallet = PAGE_BYTES;
        }

        global_block[idx]  = block;
        global_align[idx]  = align;
        global_stride[idx] = stride;
        global_pallet[idx] = pallet;

        slots = (pallet - align) / stride;
        if ( slots < 1 )
        {
            slots = 1;
        }

        {
            int bat = (int)(slots / 2);
            if ( bat < 1 )
            {
                bat = 1;
            }
            if ( bat > AOCL_BATCH_MAX )
            {
                bat = AOCL_BATCH_MAX;
            }
            global_small_batch[idx] = bat;
        }
        {
            int cap = (int)(slots * 2);
            if ( cap < AOCL_SMALL_CAP_MIN )
            {
                cap = AOCL_SMALL_CAP_MIN;
            }
            if ( cap > AOCL_SMALL_CAP_MAX )
            {
                cap = AOCL_SMALL_CAP_MAX;
            }
            global_small_cap[idx] = cap;
        }
    }

    /* Register fork handlers once, after the tables are live. */
    (void)pthread_atfork(aocl_atfork_prepare, aocl_atfork_parent,
                         aocl_atfork_child);
}

/* ceil(log2), floored at the smallest class and capped at the largest small
   class.  malloc_int() only calls this after a size <= AOCL_SMALL_MAX guard, but
   the cap keeps the result a valid index for any input (it is also used directly
   by tests): thread_cache_small[]/global_pool_small[]/global_block[] are sized
   [AOCL_SMALL_IDX + 1], so an uncapped ceil(log2) of a > 4 KiB size would index
   them out of bounds. */
static int size_to_index(size_t size)
{
    int idx;

    if ( size <= ((size_t)1 << AOCL_MIN_IDX) )
    {
        return AOCL_MIN_IDX;
    }
    /* GCC/Clang guaranteed by the #error gate above. */
    idx = (int)(sizeof(unsigned long long) * CHAR_BIT)
        - __builtin_clzll((unsigned long long)(size - 1));
    return idx > AOCL_SMALL_IDX ? AOCL_SMALL_IDX : idx;
}

/* Pop up to want objects of small class idx from core cpu's pool. Caller holds
   that core's lock. */
static int pool_small_take(int cpu, int idx, void **out, int want)
{
    int got = 0;

    while ( got < want && global_pool_small[cpu][idx] )
    {
        void *o = global_pool_small[cpu][idx];
        void *next;

        if ( aocl_unlikely(!node_link_ok(o, (uint32_t)idx, &next)) )
        {
            /* Corrupt link: drop this node and its tail (fail closed) rather
               than follow a wild pointer; return the valid nodes taken so far. */
            AOCL_ON_CORRUPT();
            global_pool_small[cpu][idx] = NULL;
            break;
        }
        global_pool_small[cpu][idx] = next;
        out[got++] = o;
    }
    return got;
}

/* Push n small objects of class idx into core cpu's pool. Caller holds lock. */
static void pool_small_add(int cpu, int idx, void **in, int n)
{
    int i;

    for ( i = 0; i < n; i++ )
    {
        *(void **)in[i] = global_pool_small[cpu][idx];
        global_pool_small[cpu][idx] = in[i];
    }
}

/* mmap a pallet and thread its slots onto this thread's bin. Returns count. */
static int slab_carve(int idx)
{
    size_t stride = global_stride[idx];
    size_t block  = global_block[idx];
    size_t pallet = global_pallet[idx];
    size_t off    = global_align[idx];  /* first user offset (header fits) */
    char  *pal    = (char *)os_map(pallet);
    int    n      = 0;

    if ( aocl_unlikely(pal == NULL) )
    {
        return 0;
    }

    while ( off + block <= pallet )
    {
        void *u = pal + off;

        /* Never hand out a page-aligned user pointer.  Its header sits at
           u - AOCL_HDR_SZ, i.e. in the previous page; aocl_free()/aocl_realloc()
           reject any pointer within AOCL_HDR_SZ of a page start WITHOUT reading
           its header (that preceding page can be unmapped for a foreign pointer),
           so a page-aligned block of ours would be misrouted to libc.  Skipping
           the slot costs at most one slot per page and keeps every live pointer's
           header in the same mapped page as the pointer.  (large/huge already sit
           at +AOCL_LARGE_OFF, never page-aligned.) */
        if ( aocl_unlikely(((uintptr_t)u & (PAGE_BYTES - 1)) == 0) )
        {
            off += stride;
            continue;
        }
        hdr_stamp(u, AOCL_MAGIC_FREED, (uint32_t)idx);   /* carved: parked free */
        *(void **)u = thread_cache_small[idx];
        thread_cache_small[idx] = u;
        thread_cache_small_cnt[idx]++;
        thread_cache_bytes += block;
        n++;
        off += stride;
    }
    return n;
}

static void thread_cache_register(void);

/* Return one ready (magic=ALLOC) object of small class idx, or NULL. */
static void *small_refill(int idx)
{
    void *buf[AOCL_BATCH_MAX];
    int   want, n, i, cpu;
    void *u;

    /* Lazy safety net.  The per-class tables are normally built eagerly by
       aocl_alloc_ini() during BLIS startup; this guarantees they are also
       ready if the allocator is driven directly before init.  small_refill
       is the sole entry that first touches those tables, so guarding it here
       keeps pthread_once off the allocation hot path entirely. */
    (void)pthread_once(&global_setup_once, global_setup);

    /* Arm the thread-exit flush here, not only in aocl_free(): small_refill and
       slab_carve() stash surplus blocks into this thread's bins during
       allocation.  A thread that only allocates (or whose blocks are freed by
       other threads) would otherwise exit without its bins ever being drained
       to the global pool -- an internal leak under thread churn. */
    thread_cache_register();

    want = global_small_batch[idx];
    if ( want > AOCL_BATCH_MAX )
    {
        want = AOCL_BATCH_MAX;
    }

    cpu = getcpu__();
    pthread_mutex_lock(&global_mutex[cpu]);
    n = pool_small_take(cpu, idx, buf, want);
    pthread_mutex_unlock(&global_mutex[cpu]);

    if ( n > 0 )
    {
        for ( i = 1; i < n; i++ )           /* stash the surplus locally */
        {
            *(void **)buf[i] = thread_cache_small[idx];
            thread_cache_small[idx] = buf[i];
            thread_cache_small_cnt[idx]++;
            thread_cache_bytes += global_block[idx];
        }
        hdr_stamp(buf[0], AOCL_MAGIC_ALLOC, (uint32_t)idx);   /* re-stamp both words */
        return buf[0];
    }

    if ( slab_carve(idx) == 0 )
    {
        return NULL;
    }

    u = tc_small_pop(idx);                                    /* freshly carved */
    if ( aocl_unlikely(u == NULL) )
    {
        return NULL;
    }
    hdr_stamp(u, AOCL_MAGIC_ALLOC, (uint32_t)idx);            /* re-stamp both words */
    return u;
}

/* Move up to m small objects of class idx from this thread's bin to the pool. */
static void thread_cache_drain_small(int idx, int m)
{
    void *buf[AOCL_BATCH_MAX];
    int   got = 0;
    int   cpu = getcpu__();

    while ( m > 0 && thread_cache_small[idx] )
    {
        void *o = tc_small_pop(idx);
        if ( aocl_unlikely(o == NULL) )      /* corrupt link: bin abandoned */
        {
            break;
        }
        buf[got++] = o;
        m--;
        if ( got == AOCL_BATCH_MAX )
        {
            pthread_mutex_lock(&global_mutex[cpu]);
            pool_small_add(cpu, idx, buf, got);
            pthread_mutex_unlock(&global_mutex[cpu]);
            got = 0;
        }
    }
    if ( got )
    {
        pthread_mutex_lock(&global_mutex[cpu]);
        pool_small_add(cpu, idx, buf, got);
        pthread_mutex_unlock(&global_mutex[cpu]);
    }
}

/* Pop up to want np-page regions from core cpu's pool. Caller holds that core's
   lock.  Large analogue of pool_small_take(). */
static int pool_large_take(int cpu, uint32_t np, void **out, int want)
{
    int got = 0;

    while ( got < want && global_pool_large[cpu][np] )
    {
        void *o = global_pool_large[cpu][np];
        void *next;

        if ( aocl_unlikely(!node_link_ok(o, AOCL_LARGE_BIT | np, &next)) )
        {
            /* Corrupt link: drop this node and its tail (fail closed) rather
               than follow a wild pointer; return the valid nodes taken so far. */
            AOCL_ON_CORRUPT();
            global_pool_large[cpu][np] = NULL;
            break;
        }
        global_pool_large[cpu][np] = next;
        out[got++] = o;
    }
    if ( got )
    {
        /* Regions taken leave the pool; drop their bytes from the resident tally
           (see large_flush / AOCL_POOL_LARGE_MAX_BYTES).  A corrupt-link
           truncation above can strand the counter high, which only makes the cap
           purge sooner -- the safe (RSS-bounding) direction -- so clamp at 0
           rather than underflow. */
        size_t dec = (size_t)got * large_map_bytes(np);
        global_pool_large_bytes[cpu] =
            global_pool_large_bytes[cpu] > dec
                ? global_pool_large_bytes[cpu] - dec : 0;
    }
    return got;
}

/* Return one ready (magic=ALLOC) large region of np pages, or NULL. A batch
 * is pulled from the pool so the lock is amortised over many allocations. */
static void *large_refill(uint32_t np)
{
    void       *buf[AOCL_LARGE_BATCH];
    size_t      rb = large_map_bytes(np);
    int         n, i;
    int         cpu = getcpu__();
    char       *region;
    void       *u;

    (void)pthread_once(&global_setup_once, global_setup);
    pthread_mutex_lock(&global_mutex[cpu]);
    n = pool_large_take(cpu, np, buf, AOCL_LARGE_BATCH);
    pthread_mutex_unlock(&global_mutex[cpu]);

    if ( n > 0 )
    {
        thread_cache_register();            /* stashing surplus below: arm exit flush */
        for ( i = 1; i < n; i++ )           /* stash the surplus locally */
        {
            *(void **)buf[i] = thread_cache_large[np];
            thread_cache_large[np] = buf[i];
            thread_cache_large_cnt[np]++;
            thread_cache_bytes += rb;
        }
        hdr_stamp(buf[0], AOCL_MAGIC_ALLOC, AOCL_LARGE_BIT | np);   /* re-stamp both words */
        return buf[0];
    }

    region = arena_carve(rb);           /* contiguous per-thread arena, not 1 mmap/buf */
    if ( aocl_unlikely(region == NULL) )
    {
        return NULL;
    }
    u = region + AOCL_LARGE_OFF;
    hdr_stamp(u, AOCL_MAGIC_ALLOC, AOCL_LARGE_BIT | np);
    return u;
}

/* Push a batch of np-page regions to the pool under one lock, bounding the
 * per-core partition to AOCL_POOL_LARGE_MAX_BYTES of RESIDENT memory.  Arena
 * chunks are shared and never unmapped, so a region cannot be handed back with
 * munmap; instead, once the partition is at its cap, a spilled region's pages
 * are dropped with MADV_DONTNEED (os_purge) and the region is DROPPED from
 * recycling.  This is what bounds RSS under a cross-thread / producer-consumer
 * free (one core allocates, another frees): without it the freeing core's
 * partition -- which the allocating core never draws from -- would retain every
 * freed region's pages forever.  Below the cap, regions are kept fully resident
 * for fast same-core reuse, so the common bounded-churn case is unchanged.
 * Residual cost: an abandoned region's arena VA is not reused (its pages are
 * non-resident), trading address space for bounded RSS. */
static void large_flush(uint32_t np, void **buf, int n)
{
    int    i;
    int    cpu = getcpu__();
    size_t rb  = large_map_bytes(np);

    pthread_mutex_lock(&global_mutex[cpu]);
    for ( i = 0; i < n; i++ )
    {
        if ( aocl_likely(global_pool_large_bytes[cpu] + rb
                         <= AOCL_POOL_LARGE_MAX_BYTES) )
        {
            *(void **)buf[i] = global_pool_large[cpu][np];
            global_pool_large[cpu][np] = buf[i];
            global_pool_large_bytes[cpu] += rb;
        }
        else if ( os_purge((char *)buf[i] - AOCL_LARGE_OFF, rb) )
        {
            /* Over cap and pages returned to the OS: abandon the region.  Its VA
               stays mapped (part of a shared arena chunk) but non-resident, and
               it is not re-linked, so it is never reused -- the residual address-
               space cost noted above, in exchange for bounding RSS. */
        }
        else
        {
            /* MADV_DONTNEED unavailable: fall back to the prior behaviour and
               retain the region rather than leave it unreachable AND resident. */
            *(void **)buf[i] = global_pool_large[cpu][np];
            global_pool_large[cpu][np] = buf[i];
            global_pool_large_bytes[cpu] += rb;
        }
    }
    pthread_mutex_unlock(&global_mutex[cpu]);
}

/* Move up to m large regions of np pages from this thread's bin to the pool. */
static void thread_cache_drain_large(uint32_t np, int m)
{
    void  *buf[AOCL_LARGE_BATCH];
    int    got = 0;

    while ( m > 0 && thread_cache_large[np] )
    {
        void *o = tc_large_pop(np);
        if ( aocl_unlikely(o == NULL) )      /* corrupt link: bin abandoned */
        {
            break;
        }
        buf[got++] = o;
        m--;
        if ( got == AOCL_LARGE_BATCH )
        {
            large_flush(np, buf, got);
            got = 0;
        }
    }
    if ( got )
    {
        large_flush(np, buf, got);
    }
}

static void thread_cache_flush_all(void)
{
    int      i;
    uint32_t np;

    for ( i = AOCL_MIN_IDX; i <= AOCL_SMALL_IDX; i++ )
    {
        if ( thread_cache_small_cnt[i] )
        {
            thread_cache_drain_small(i, thread_cache_small_cnt[i]);
        }
    }

    for ( np = 1; np <= AOCL_MAX_PAGES; np++ )
    {
        if ( thread_cache_large_cnt[np] )
        {
            thread_cache_drain_large(np, thread_cache_large_cnt[np]);
        }
    }
}

static void thread_cache_dtor(void *arg)
{
    (void)arg;
    thread_cache_flush_all();
    thread_cache_reg = 0;
}

static void thread_cache_key_make(void)
{
    (void)pthread_key_create(&thread_cache_key, thread_cache_dtor);
}

static void thread_cache_register(void)
{
#ifndef AOCL_NO_EXIT_FLUSH
    if ( aocl_likely(thread_cache_reg) )
    {
        return;
    }
    (void)pthread_once(&thread_cache_key_once, thread_cache_key_make);
    (void)pthread_setspecific(thread_cache_key, (void *)1);
    thread_cache_reg = 1;
#endif
}

static void *malloc_int(size_t size)
{
    if ( aocl_unlikely(size == 0) )
    {
        size = 1;
    }

    /* ---- small: power-of-two classes served from pallets ---- */
    if ( aocl_likely(size <= AOCL_SMALL_MAX) )
    {
        int   idx = size_to_index(size);
        void *u   = tc_small_pop(idx);             /* lock-free fast path */
        if ( aocl_likely(u != NULL) )
        {
            hdr_stamp(u, AOCL_MAGIC_ALLOC, (uint32_t)idx);   /* re-stamp both words */
            return u;
        }
        u = small_refill(idx);
        return aocl_likely(u != NULL) ? u : os_fallback( size ); /* libc fallback */
    }

    /* ---- large / huge: exact page-multiple regions ---- */
    {
        size_t need, np;

        if ( aocl_unlikely(size > SIZE_MAX - AOCL_LARGE_OFF - PAGE_BYTES) )
        {
            return NULL;
        }
        need = AOCL_LARGE_OFF + size;
        np   = (need + PAGE_BYTES - 1) >> PAGE_SHIFT;

        if ( aocl_likely(np <= AOCL_MAX_PAGES) )   /* cacheable large */
        {
            void *u = tc_large_pop((uint32_t)np);  /* lock-free fast path */
            if ( aocl_likely(u != NULL) )
            {
                hdr_stamp(u, AOCL_MAGIC_ALLOC,                /* re-stamp both words */
                          AOCL_LARGE_BIT | (uint32_t)np);
                return u;
            }
            u = large_refill((uint32_t)np);
            return aocl_likely(u != NULL) ? u : os_fallback( size );
        }
        else                                       /* huge: uncached */
        {
            char *region;
            void *u;

            if ( aocl_unlikely(np > (AOCL_LARGE_BIT - 1)) )
            {
                return os_fallback( size );
            }
            region = (char *)os_map_large(large_map_bytes((uint32_t)np));
            if ( aocl_unlikely(region == NULL) )
            {
                return os_fallback( size );
            }
            u = region + AOCL_LARGE_OFF;
            hdr_stamp(u, AOCL_MAGIC_ALLOC, AOCL_LARGE_BIT | (uint32_t)np);
            return u;
        }
    }
}

AOCL_ALLOC_LOCAL void *aocl_malloc(size_t size)
{
    return malloc_int(size);
}

AOCL_ALLOC_LOCAL void aocl_free(void *ptr)
{
    aocl_hdr_t *h;
    uint32_t    magic, info, chk;

    if ( aocl_unlikely(ptr == NULL) )
    {
        return;
    }

    if ( aocl_unlikely(!va_ours(ptr)) )            /* foreign / libc fallback */
    {
        free( ptr );
        return;
    }

    /* The header sits at ptr - AOCL_HDR_SZ.  Every pointer this allocator hands
       out is >= AOCL_HDR_SZ into its own page (large/huge land at +AOCL_LARGE_OFF;
       small slots are carved to never be page-aligned), so its header shares the
       pointer's page -- which is mapped.  The watermark test above is only a
       coarse [lo,hi) range, so an in-range FOREIGN pointer can still reach here;
       if it lies within AOCL_HDR_SZ of a page start, reading its header could
       touch an unmapped preceding page.  It cannot be ours, so hand it to libc
       without dereferencing. */
    if ( aocl_unlikely(((uintptr_t)ptr & (PAGE_BYTES - 1)) < AOCL_HDR_SZ) )
    {
        free( ptr );
        return;
    }

    h     = hdr_of(ptr);
    magic = h->magic;
    info  = h->info;
    chk   = hdr_mix(info);

    /* The seal binds magic to info (magic = base ^ hdr_mix(info)); a mismatch
       means the two words disagree.  ptr is inside our mmap watermark, so it is
       either ours (with a corrupted header) or a rare in-range foreign/libc-
       fallback pointer -- in NO case is it something libc's malloc produced in a
       way that is safe to free().  Handing an mmap'd native pointer to libc
       free() could reach glibc's IS_MMAPPED path and munmap a corruption-derived
       address, so this path never calls free(). */
    if ( aocl_unlikely(magic != (AOCL_MAGIC_ALLOC ^ chk)) )
    {
        /* The seal disagrees: a corrupted native header, a double-free, or a
           rare in-range foreign/libc-fallback pointer.  The size class is not
           trustworthy in any of these, and handing an in-range NATIVE pointer to
           libc free() could reach glibc's IS_MMAPPED path and munmap a
           corruption-derived address.  FAIL CLOSED and LEAK (a double-free is the
           benign sub-case: the object is already parked, so dropping the
           redundant free leaks nothing).  We deliberately do NOT recover a class
           from magic and recycle the block -- the seal is address-independent, so
           a coincidental magic match cannot be told from a genuine object, and
           threading a suspect pointer back onto a live free list would be
           strictly more dangerous than leaking (large regions stay mapped for the
           process lifetime anyway). */
        AOCL_ON_CORRUPT();
        return;                                    /* fail closed: leak */
    }

    /* ---- small ---- */
    if ( aocl_likely(!(info & AOCL_LARGE_BIT)) )
    {
        int idx = (int)info;
        /* info is confirmed by an intact ALLOC seal, so idx is in range.  Keep
           the bound as a defensive guard against a self-consistent ~2^-32 magic
           collision on a foreign pointer; if it ever trips, leak (never index the
           class tables out of bounds, and never libc-free an in-range pointer --
           see the mismatch note above). */
        if ( aocl_unlikely(idx < AOCL_MIN_IDX || idx > AOCL_SMALL_IDX) )
        {
            AOCL_ON_CORRUPT();
            return;                                /* unrecoverable: leak */
        }
        h->magic = AOCL_MAGIC_FREED ^ chk;
        thread_cache_register();
        *(void **)ptr = thread_cache_small[idx];
        thread_cache_small[idx] = ptr;
        thread_cache_small_cnt[idx]++;
        thread_cache_bytes += global_block[idx];
        if ( aocl_unlikely(thread_cache_small_cnt[idx] > global_small_cap[idx] ||
                           thread_cache_bytes > AOCL_TC_MAX_BYTES) )
        {
            thread_cache_drain_small(idx, global_small_batch[idx]);
        }
        return;
    }

    /* ---- large / huge ---- */
    {
        uint32_t np = info & ~AOCL_LARGE_BIT;
        /* malloc_int() never encodes a zero page count, so np == 0 means a ~2^-32
           magic collision on a foreign/corrupt header.  Leak rather than index
           thread_cache_large[0] (which would poison the allocator) or libc-free
           an in-range pointer. */
        if ( aocl_unlikely(np == 0) )
        {
            AOCL_ON_CORRUPT();
            return;                                /* unrecoverable: leak */
        }
        h->magic = AOCL_MAGIC_FREED ^ chk;
        if ( aocl_unlikely(np > AOCL_MAX_PAGES) )  /* huge: drop RSS, keep VA */
        {
            os_release_huge((char *)ptr - AOCL_LARGE_OFF, large_map_bytes(np));
            return;
        }
        thread_cache_register();
        *(void **)ptr = thread_cache_large[np];
        thread_cache_large[np] = ptr;
        thread_cache_large_cnt[np]++;
        thread_cache_bytes += large_map_bytes(np);
        if ( aocl_unlikely(thread_cache_large_cnt[np] > AOCL_TC_CAP_LARGE ||
                           thread_cache_bytes > AOCL_TC_MAX_BYTES) )
        {
            thread_cache_drain_large(np, AOCL_LARGE_BATCH);
        }
    }
}

AOCL_ALLOC_LOCAL void *aocl_calloc(size_t nmemb, size_t size)
{
    size_t total;
    void  *ptr;

    if ( aocl_unlikely(size != 0 && nmemb > SIZE_MAX / size) )
    {
        return NULL;
    }
    total = nmemb * size;
    ptr = malloc_int(total);
    if ( aocl_unlikely(ptr == NULL) )
    {
        return NULL;
    }
    memset(ptr, 0, total);                          /* reused chunks are dirty */
    return ptr;
}

/* NOTE (currently unwired): BLIS routes only malloc and free through this
   allocator -- the BLIS_MALLOC_* and BLIS_FREE_* families resolve to aocl_malloc
   and aocl_free.  There is no BLIS_REALLOC_* or BLIS_CALLOC_* macro, so
   aocl_realloc() and aocl_calloc() exist for libc-API completeness but are not
   reached in-tree.  If a future caller is wired up, note that aocl_realloc() does
   NOT follow libc realloc() semantics on a corrupted/foreign header: it FAILS
   CLOSED -- returns a fresh block of the requested size and LEAKS the original
   (its contents are dropped, not copied) -- rather than preserving the data or
   returning an error.  That is intentional (see the header-seal note at the top
   of the file: an in-range native pointer whose seal fails is never trusted for a
   size or handed to libc), but a caller expecting libc behaviour must account
   for it. */
AOCL_ALLOC_LOCAL void *aocl_realloc(void *ptr, size_t newsize)
{
    aocl_hdr_t *h;
    uint32_t    info;
    size_t      usable;
    void       *newp;

    if ( ptr == NULL )
    {
        return malloc_int(newsize);
    }

    if ( aocl_unlikely(!va_ours(ptr)) )
    {
        return realloc( ptr, newsize );             /* foreign / libc fallback */
    }

    /* See aocl_free(): a pointer within AOCL_HDR_SZ of a page start cannot be
       ours, and reading its header could fault on an unmapped preceding page. */
    if ( aocl_unlikely(((uintptr_t)ptr & (PAGE_BYTES - 1)) < AOCL_HDR_SZ) )
    {
        return realloc( ptr, newsize );
    }

    h    = hdr_of(ptr);
    info = h->info;
    if ( aocl_unlikely(h->magic != hdr_seal(AOCL_MAGIC_ALLOC, info)) )
    {
        /* Header inconsistent (corrupted native header, or a rare in-range
           foreign/libc-fallback pointer).  The old size is not trustworthy, and
           neither copying from nor libc-realloc'ing an in-range native pointer is
           safe (the free step could munmap a corruption-derived address).  FAIL
           CLOSED: return a fresh block of the requested size and LEAK the
           original.  As in aocl_free(), we do NOT recover a class from magic and
           copy the block: the address-independent seal cannot distinguish a
           coincidental magic match from a genuine object, so trusting it would
           size an out-of-bounds memcpy below. */
        AOCL_ON_CORRUPT();
        return malloc_int(newsize);                 /* fail closed: fresh block, leak old */
    }

    /* Decode the (now trusted) class/page count to size the existing block.  The
       bounds below still fail closed on a ~2^-32 magic collision: allocate a
       fresh block and leak the original -- never libc-realloc an in-range ptr. */
    if ( info & AOCL_LARGE_BIT )
    {
        uint32_t np = info & ~AOCL_LARGE_BIT;
        if ( aocl_unlikely(np == 0) )
        {
            AOCL_ON_CORRUPT();
            return malloc_int(newsize);             /* fail closed: fresh block, leak old */
        }
        usable = ((size_t)np << PAGE_SHIFT) - AOCL_LARGE_OFF;
    }
    else
    {
        int idx = (int)info;
        if ( aocl_unlikely(idx < AOCL_MIN_IDX || idx > AOCL_SMALL_IDX) )
        {
            AOCL_ON_CORRUPT();
            return malloc_int(newsize);             /* fail closed: fresh block, leak old */
        }
        usable = global_block[idx];
    }
    if ( newsize <= usable )
    {
        return ptr;
    }

    newp = malloc_int(newsize);
    if ( aocl_unlikely(newp == NULL) )
    {
        return NULL;
    }
    memcpy(newp, ptr, usable < newsize ? usable : newsize);
    aocl_free(ptr);
    return newp;
}

AOCL_ALLOC_LOCAL void aocl_alloc_ini(void)
{
    (void)pthread_once(&global_setup_once, global_setup);
}

/* Drains only the calling thread's TLS bins to the pool.  Pools, arena chunks,
   and pallets stay mapped for the process lifetime (needed for cross-thread
   reuse), so RSS is not returned at bli_finalize() -- memory is retained for
   reuse and reclaimed by the OS at process exit. */
AOCL_ALLOC_LOCAL void aocl_alloc_fini(void)
{
    thread_cache_flush_all();  /* drain the calling thread's bins to the pool */
}

#endif /* EXPORT_AOCL_ALLOCATOR */

#endif /* !defined(BLIS_OS_LINUX) */
#endif /* BLIS_ENABLE_AOCL_ALLOC */
#endif /* BLIS_AOCL_ALLOCATOR_H */
