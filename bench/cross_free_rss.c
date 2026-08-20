/* Cross-thread free RSS demonstrator for the AOCL allocator.
 *
 * A producer thread pinned to one core allocates large buffers and hands them
 * to a consumer thread pinned to a DIFFERENT core, which frees them. Because
 * the recycle pool is partitioned per core, the producer's partition never
 * receives the frees (they land on the consumer's partition) and the producer
 * keeps carving fresh arena. With the shared large pool uncapped, every freed
 * region's pages stay resident on the consumer's partition forever -> RSS grows
 * without bound. With the per-core resident cap (AOCL_POOL_LARGE_MAX_BYTES),
 * the surplus is MADV_DONTNEED'd and RSS plateaus.
 *
 * Build twice from the same header:
 *   baseline (uncapped): -DAOCL_POOL_LARGE_MAX_BYTES='(~(size_t)0)'
 *   patched  (capped)  : (default cap)
 *
 * Usage: ./cross_free_rss <buf_bytes> <niters> <prod_core> <cons_core>
 */
#define _GNU_SOURCE
#define BLIS_ENABLE_AOCL_ALLOC
#define BLIS_OS_LINUX
#define EXPORT_AOCL_ALLOCATOR
#include "frame/base/aocl_allocator.h"

#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>

#define QCAP 2048

static void          *ring[QCAP];
static long           q_head, q_tail;   /* head==tail => empty */
static int            q_done;
static pthread_mutex_t q_m   = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  q_ne  = PTHREAD_COND_INITIALIZER;  /* not empty */
static pthread_cond_t  q_nf  = PTHREAD_COND_INITIALIZER;  /* not full  */

static size_t BUFSZ;
static long   NITERS;
static int    PROD_CORE, CONS_CORE;
static volatile long prod_count;   /* progress, sampled by monitor */

static void pin(int core)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

static void q_push(void *p)
{
    pthread_mutex_lock(&q_m);
    while ( q_head - q_tail >= QCAP )
        pthread_cond_wait(&q_nf, &q_m);
    ring[q_head % QCAP] = p;
    q_head++;
    pthread_cond_signal(&q_ne);
    pthread_mutex_unlock(&q_m);
}

static void *q_pop(void)
{
    void *p;
    pthread_mutex_lock(&q_m);
    while ( q_head == q_tail && !q_done )
        pthread_cond_wait(&q_ne, &q_m);
    if ( q_head == q_tail && q_done )
    {
        pthread_mutex_unlock(&q_m);
        return NULL;
    }
    p = ring[q_tail % QCAP];
    q_tail++;
    pthread_cond_signal(&q_nf);
    pthread_mutex_unlock(&q_m);
    return p;
}

static void *producer(void *arg)
{
    long i;
    (void)arg;
    pin(PROD_CORE);
    for ( i = 0; i < NITERS; i++ )
    {
        void *p = aocl_malloc(BUFSZ);
        if ( p == NULL ) { fprintf(stderr, "malloc failed at %ld\n", i); exit(1); }
        memset(p, 1, BUFSZ);          /* touch: make pages resident */
        q_push(p);
        __atomic_store_n(&prod_count, i + 1, __ATOMIC_RELAXED);
    }
    pthread_mutex_lock(&q_m);
    q_done = 1;
    pthread_cond_broadcast(&q_ne);
    pthread_mutex_unlock(&q_m);
    return NULL;
}

static void *consumer(void *arg)
{
    (void)arg;
    pin(CONS_CORE);
    for ( ;; )
    {
        void *p = q_pop();
        if ( p == NULL ) break;
        {
            volatile char c = ((char *)p)[0];   /* read: use it */
            (void)c;
        }
        aocl_free(p);
    }
    return NULL;
}

static long rss_kb(void)
{
    long pages = 0, resident = 0;
    FILE *f = fopen("/proc/self/statm", "r");
    if ( f == NULL ) return -1;
    if ( fscanf(f, "%ld %ld", &pages, &resident) != 2 ) resident = -1;
    fclose(f);
    return resident * (sysconf(_SC_PAGESIZE) / 1024);
}

int main(int argc, char **argv)
{
    pthread_t pt, ct;
    long peak = 0;

    BUFSZ     = ( argc > 1 ) ? (size_t)strtoull(argv[1], NULL, 0) : (256u << 10);
    NITERS    = ( argc > 2 ) ? strtol(argv[2], NULL, 0) : 40000;
    PROD_CORE = ( argc > 3 ) ? atoi(argv[3]) : 0;
    CONS_CORE = ( argc > 4 ) ? atoi(argv[4]) : 1;

    aocl_alloc_ini();

    printf("# buf=%zuKiB niters=%ld prod_core=%d cons_core=%d\n",
           BUFSZ >> 10, NITERS, PROD_CORE, CONS_CORE);
    printf("# progress%%   rss_MiB\n");

    pthread_create(&pt, NULL, producer, NULL);
    pthread_create(&ct, NULL, consumer, NULL);

    /* Monitor: sample RSS vs producer progress until the run completes. */
    for ( ;; )
    {
        long done = __atomic_load_n(&prod_count, __ATOMIC_RELAXED);
        long r    = rss_kb();
        long rmib = r / 1024;
        if ( rmib > peak ) peak = rmib;
        printf("%8.1f   %8ld\n", 100.0 * (double)done / (double)NITERS, rmib);
        fflush(stdout);
        if ( done >= NITERS ) break;
        usleep(100000);   /* 100 ms */
    }

    pthread_join(pt, NULL);
    pthread_join(ct, NULL);

    printf("# PEAK_RSS_MiB %ld\n", peak);
    return 0;
}
