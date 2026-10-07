/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2019, Advanced Micro Devices, Inc.

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

*/

#include "blis.h"

//#define BLIS_ENABLE_MEM_TRACING

// -----------------------------------------------------------------------------
// Huge-page-backed pool allocator.
//
// When BLIS_ENABLE_HUGEPAGE_POOL is defined, BLIS's internal memory pools
// (used for the packed A/B/C buffers, which are the hot, repeatedly-streamed
// operands of level-3 operations) are backed by explicit huge pages instead
// of ordinary 4 KiB pages. This reduces dTLB pressure and page walks when
// streaming large packed panels through the cache hierarchy.
//
// Allocation strategy, per block, largest page first (as requested):
//   1 GiB explicit huge pages (MAP_HUGETLB | MAP_HUGE_1GB) when the request is
//     large enough to justify them and the OS has 1 GiB pages reserved;
//   2 MiB explicit huge pages (MAP_HUGETLB | MAP_HUGE_2MB) otherwise, when the
//     OS has 2 MiB pages reserved;
//   ordinary malloc() as the always-succeeding fallback (identical to BLIS's
//   default behaviour, so there is never a regression when no hugetlb pool is
//   configured -- and on a THP=always system malloc's large mmap-backed
//   allocations are transparently promoted to huge pages by the kernel anyway).
//
// hugetlb blocks are obtained with mmap() and released with munmap(); malloc
// blocks with malloc()/free(). A small header just before the returned pointer
// records which path was taken (and the mmap base/length) so the matching
// deallocator can be used. The signatures match malloc()/free(), so these plug
// directly into BLIS_MALLOC_POOL / BLIS_FREE_POOL.
#ifdef BLIS_ENABLE_HUGEPAGE_POOL

#include <sys/mman.h>
#include <stdint.h>

#ifndef MAP_HUGETLB
#define MAP_HUGETLB 0x40000
#endif
#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#endif
#ifndef MAP_HUGE_1GB
#define MAP_HUGE_1GB (30 << MAP_HUGE_SHIFT)
#endif

#define BLI_HP_2MB   ( (size_t)2  * 1024 * 1024 )
#define BLI_HP_1GB   ( (size_t)1024 * 1024 * 1024 )
#define BLI_HP_HDR   ( (size_t)64 )            // >= sizeof(bli_hp_hdr_t)
#define BLI_HP_MAGIC ( (size_t)0x48554745504f4f4cULL ) // "HUGEPOOL"

enum { BLI_HP_MALLOC = 0, BLI_HP_MMAP = 1 };

// Header stored in the BLI_HP_HDR bytes immediately before the returned
// pointer: how the block was obtained, the raw base to hand back to the
// deallocator, and the mapping length (mmap only), plus a magic sentinel.
typedef struct { void* base; size_t len; int mode; size_t magic; } bli_hp_hdr_t;

static size_t bli_hp_round_up( size_t x, size_t a ) { return ( x + a - 1 ) & ~( a - 1 ); }

// Try an explicit-hugetlb mmap of a whole huge pages worth. Returns NULL if the
// OS has no such pages reserved (mmap fails with ENOMEM).
static void* bli_hp_try_hugetlb( size_t len, int huge_flag )
{
	void* p = mmap( NULL, len, PROT_READ | PROT_WRITE,
	                MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | huge_flag, -1, 0 );
	return ( p == MAP_FAILED ) ? NULL : p;
}

static void* bli_hp_finish( void* base, size_t len, int mode )
{
	// The header occupies the first BLI_HP_HDR bytes of the block; the payload
	// begins right after it. The payload need not be huge-page aligned: an
	// entire MAP_HUGETLB mapping is backed by huge pages regardless, and the
	// caller (bli_fmalloc_align) applies its own alignment on top.
	bli_hp_hdr_t* h = ( bli_hp_hdr_t* )base;
	h->base  = base;
	h->len   = len;
	h->mode  = mode;
	h->magic = BLI_HP_MAGIC;
	return ( void* )( ( int8_t* )base + BLI_HP_HDR );
}

void* bli_hugepage_malloc( size_t size )
{
	if ( size == 0 ) return NULL;

	// Resolve environment controls once per process (see coding note: never
	// call getenv() repeatedly on a hot path).
	//   BLI_HP_DEBUG   - trace each allocation and the page size chosen.
	//   BLI_HP_DISABLE - force the plain malloc() path (huge pages off).
	static int dbg = -1, off = -1;
	if ( dbg < 0 ) dbg = ( getenv( "BLI_HP_DEBUG"   ) != NULL );
	if ( off < 0 ) off = ( getenv( "BLI_HP_DISABLE" ) != NULL );

	size_t need = size + BLI_HP_HDR;

	// 1 GiB explicit huge pages for large requests.
	if ( !off && size >= BLI_HP_1GB )
	{
		size_t len  = bli_hp_round_up( need, BLI_HP_1GB );
		void*  base = bli_hp_try_hugetlb( len, MAP_HUGE_1GB );
		if ( dbg ) fprintf( stderr, "[hp] size=%zu -> 1GB %s\n", size, base?"OK":"FAIL" );
		if ( base ) return bli_hp_finish( base, len, BLI_HP_MMAP );
	}
	// 2 MiB explicit huge pages otherwise.
	if ( !off && size >= BLI_HP_2MB )
	{
		size_t len  = bli_hp_round_up( need, BLI_HP_2MB );
		void*  base = bli_hp_try_hugetlb( len, MAP_HUGE_2MB );
		if ( dbg ) fprintf( stderr, "[hp] size=%zu -> 2MB %s\n", size, base?"OK":"FAIL" );
		if ( base ) return bli_hp_finish( base, len, BLI_HP_MMAP );
	}
	// Fallback: ordinary malloc (no regression vs. BLIS's default pool).
	{
		if ( dbg ) fprintf( stderr, "[hp] size=%zu -> malloc\n", size );
		void* base = malloc( need );
		if ( base == NULL ) return NULL;
		return bli_hp_finish( base, 0, BLI_HP_MALLOC );
	}
}

void bli_hugepage_free( void* p )
{
	if ( p == NULL ) return;

	// The header sits in [user - BLI_HP_HDR, user); its start is not fixed
	// relative to the raw base (alignment slack varies), so scan is avoided by
	// reading the header at the known offset used by bli_hp_finish().
	bli_hp_hdr_t* h = ( bli_hp_hdr_t* )( ( int8_t* )p - BLI_HP_HDR );

	// Defensive: if the header magic is wrong, this pointer did not come from
	// bli_hugepage_malloc(); fall back to free() rather than corrupting state.
	if ( h->magic != BLI_HP_MAGIC ) { free( p ); return; }

	if ( h->mode == BLI_HP_MMAP ) munmap( h->base, h->len );
	else                          free( h->base );
}

#endif // BLIS_ENABLE_HUGEPAGE_POOL

// -----------------------------------------------------------------------------

// NOTE: These functions are no longer used. Instead, the relevant sections
// of code call bli_fmalloc_align() and pass in the desired malloc()-like
// function, such as BLIS_MALLOC_POOL.

#if 0
void* bli_malloc_pool( size_t size )
{
	const malloc_ft malloc_fp  = BLIS_MALLOC_POOL;
	const size_t    align_size = BLIS_POOL_ADDR_ALIGN_SIZE;

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_malloc_pool(): size %ld, align size %ld\n",
	        ( long )size, ( long )align_size );
	fflush( stdout );
	#endif

	return bli_fmalloc_align( malloc_fp, size, align_size );
}

void bli_free_pool( void* p )
{
	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_free_pool(): freeing block\n" );
	fflush( stdout );
	#endif

	bli_ffree_align( BLIS_FREE_POOL, p );
}
#endif

// -----------------------------------------------------------------------------

void* bli_malloc_user( size_t size, err_t* r_val )
{
	const malloc_ft malloc_fp  = BLIS_MALLOC_USER;
	const size_t    align_size = BLIS_HEAP_ADDR_ALIGN_SIZE;

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_malloc_user(): size %ld, align size %ld\n",
	        ( long )size, ( long )align_size );
	fflush( stdout );
	#endif

	void* p = bli_fmalloc_align( malloc_fp, size, align_size, r_val );

	return p;
}

void bli_free_user( void* p )
{
	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_free_user(): freeing block\n" );
	fflush( stdout );
	#endif

	bli_ffree_align( BLIS_FREE_USER, p );
}

// -----------------------------------------------------------------------------

void* bli_malloc_intl( size_t size, err_t* r_val )
{
	const malloc_ft malloc_fp = BLIS_MALLOC_INTL;

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_malloc_intl(): size %ld\n", ( long )size );
	fflush( stdout );
	#endif

	void* p = bli_fmalloc_noalign( malloc_fp, size, r_val );

	return p;
}

void* bli_calloc_intl( size_t size, err_t* r_val )
{
	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_calloc_intl(): " );
	#endif

	void* p = bli_malloc_intl( size, r_val );

	if ( bli_is_success( *r_val ) )
		memset( p, 0, size );

	return p;
}

void bli_free_intl( void* p )
{
	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_free_intl(): freeing block\n" );
	fflush( stdout );
	#endif

	bli_ffree_noalign( BLIS_FREE_INTL, p );
}

// -----------------------------------------------------------------------------

void* bli_fmalloc_align
     (
       malloc_ft f,
       size_t    size,
       size_t    align_size,
       err_t*    r_val
     )
{
	const size_t ptr_size     = sizeof( void* );
	size_t       align_offset = 0;
	void*        p_orig;
	int8_t*      p_byte;
	void**       p_addr;

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_fmalloc_align_check( f, size, align_size );

	// Return early if zero bytes were requested.
	if ( size == 0 ) return NULL;

	// Add the alignment size and the size of a pointer to the number
	// of bytes to allocate.
	size += align_size + ptr_size;

	// Call the allocation function.
	p_orig = f( size );

	// Check the pointer returned by malloc().
	if ( bli_error_checking_is_enabled() )
		bli_fmalloc_post_check( p_orig );

	// The pseudo-return value isn't used yet.
	*r_val = BLIS_SUCCESS;

	// Advance the pointer by one pointer element.
	p_byte = p_orig;
	p_byte += ptr_size;

	// Compute the offset to the desired alignment.
	if ( bli_is_unaligned_to( ( siz_t )p_byte, ( siz_t )align_size ) )
	{
		align_offset = align_size -
		               bli_offset_past_alignment( ( siz_t )p_byte,
		                                          ( siz_t )align_size );
	}

	// Advance the pointer using the difference between the alignment
	// size and the alignment offset.
	p_byte += align_offset;

	// Compute the address of the pointer element just before the start
	// of the aligned address, and store the original address there.
	p_addr = ( void** )(p_byte - ptr_size);
	*p_addr = p_orig;

	// Return the aligned pointer.
	return p_byte;
}

void bli_ffree_align
     (
       free_ft f,
       void*   p
     )
{
	const size_t ptr_size = sizeof( void* );
	void*        p_orig;
	int8_t*      p_byte;
	void**       p_addr;

	// If the pointer to free is NULL, it was obviously not aligned and
	// does not need to be freed.
	if ( p == NULL ) return;

	// Since the bli_fmalloc_align() function returned the aligned pointer,
	// we have to first recover the original pointer before we can free the
	// memory.

	// Start by casting the pointer to a byte pointer.
	p_byte = p;

	// Compute the address of the pointer element just before the start
	// of the aligned address, and recover the original address.
	p_addr = ( void** )( p_byte - ptr_size );
	p_orig = *p_addr;

	// Free the original pointer.
	f( p_orig );
}

// -----------------------------------------------------------------------------

void* bli_fmalloc_noalign
     (
       malloc_ft f,
       size_t    size,
       err_t*    r_val
     )
{
	void* p = f( size );

	// Check the pointer returned by malloc().
	if ( bli_error_checking_is_enabled() )
		bli_fmalloc_post_check( p );

	// The pseudo-return value isn't used yet.
	*r_val = BLIS_SUCCESS;

	return p;
}

void bli_ffree_noalign
     (
       free_ft f,
       void*   p
     )
{
	f( p );
}

// -----------------------------------------------------------------------------

void bli_fmalloc_align_check
     (
       malloc_ft f,
       size_t    size,
       size_t    align_size
     )
{
	err_t e_val;

	// Check for valid alignment.

	e_val = bli_check_alignment_is_power_of_two( align_size );
	bli_check_error_code( e_val );

	e_val = bli_check_alignment_is_mult_of_ptr_size( align_size );
	bli_check_error_code( e_val );
}

void bli_fmalloc_post_check
     (
       void* p
     )
{
	err_t e_val;

	// Check for valid values from malloc().

	e_val = bli_check_valid_malloc_buf( p );
	bli_check_error_code( e_val );
}

