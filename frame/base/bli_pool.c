/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018, Advanced Micro Devices, Inc.

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

//#define ENABLE_MEM_DEBUG

void bli_pool_init
     (
       dim_t     num_blocks,
       dim_t     block_ptrs_len,
       siz_t     block_size,
       siz_t     align_size,
       malloc_ft malloc_fp,
       free_ft   free_fp,
       pool_t*   pool
     )
{
	pblk_t* block_ptrs;
	dim_t   i;

	// Make sure that num_block_ptrs is at least num_blocks.
	if ( block_ptrs_len < num_blocks ) block_ptrs_len = num_blocks;

	#ifdef ENABLE_MEM_DEBUG
	printf( "bli_pool_init(): allocating block_ptrs array of size %ld\n",
	        ( long )( block_ptrs_len * sizeof( pblk_t ) ) );
	#endif

	// Allocate the block_ptrs array.
	block_ptrs = bli_malloc_intl( block_ptrs_len * sizeof( pblk_t ) );

	// Allocate and initialize each entry in the block_ptrs array.
	for ( i = 0; i < num_blocks; ++i )
	{
		#ifdef ENABLE_MEM_DEBUG
		printf( "bli_pool_init(): allocating block %d of size %ld (align %ld)\n",
		        ( int )i, ( long )block_size, ( long )align_size );
		#endif

		bli_pool_alloc_block( block_size, align_size,
		                      &(block_ptrs[i]), pool );
	}

	// Initialize the pool_t structure.
	bli_pool_set_block_ptrs( block_ptrs, pool );
	bli_pool_set_block_ptrs_len( block_ptrs_len, pool );
	bli_pool_set_num_blocks( num_blocks, pool );
	bli_pool_set_top_index( 0, pool );
	bli_pool_set_block_size( block_size, pool );
	bli_pool_set_align_size( align_size, pool );
	bli_pool_set_malloc_fp( malloc_fp, pool );
	bli_pool_set_free_fp( free_fp, pool );
}

void bli_pool_finalize
     (
       pool_t* pool
     )
{
	pblk_t* block_ptrs;
	dim_t   num_blocks;
	dim_t   top_index;
	dim_t   i;

	// NOTE: This implementation assumes that either:
	// - all blocks have been checked in by all threads, or
	// - some subset of blocks have been checked in and the caller
	//   is bli_pool_reinit().

	// Query the current block_ptrs array.
	block_ptrs = bli_pool_block_ptrs( pool );

	// Query the total number of blocks presently allocated.
	num_blocks = bli_pool_num_blocks( pool );

	// Query the top_index of the pool.
	top_index  = bli_pool_top_index( pool );

	// Free the individual blocks currently in the pool.
	for ( i = top_index; i < num_blocks; ++i )
	{
		#ifdef ENABLE_MEM_DEBUG
		printf( "bli_pool_finalize(): freeing block %d of size %ld (align %ld)\n",
		        ( int )i, ( long )bli_pool_block_size( pool ),
		                  ( long )bli_pool_align_size( pool ) );
		#endif

		bli_pool_free_block( &(block_ptrs[i]), pool );
	}

	#ifdef ENABLE_MEM_DEBUG
	printf( "bli_pool_finalize(): freeing block_ptrs array of size %ld\n",
	        ( long )( bli_pool_block_ptrs_len( pool ) * sizeof( pblk_t ) ) );
	#endif

	// Free the block_ptrs array.
	bli_free_intl( block_ptrs );

	// This explicit clearing of the pool_t struct is not strictly
	// necessary and so it has been commented out.
#if 0
	// Clear the contents of the pool_t struct.
	bli_pool_set_block_ptrs( NULL, pool );
	bli_pool_set_block_ptrs_len( 0, pool );
	bli_pool_set_num_blocks( 0, pool );
	bli_pool_set_top_index( 0, pool );
	bli_pool_set_block_size( 0, pool );
	bli_pool_set_align_size( 0, pool );
#endif
}

void bli_pool_reinit
     (
       dim_t   num_blocks_new,
       dim_t   block_ptrs_len_new,
       siz_t   block_size_new,
       siz_t   align_size_new,
       pool_t* pool
     )
{
	// Preserve the pointers to malloc() and free() provided when the pool
	// was first initialized.
	malloc_ft malloc_fp = bli_pool_malloc_fp( pool );
	free_ft   free_fp   = bli_pool_free_fp( pool );

	// Finalize the pool as it is currently configured. If some blocks
	// are still checked out to threads, those blocks are not freed
	// here, and instead will be freed when the threads are ready to
	// release the blocks. (This will happen because the threads will
	// notice that the block size of the pool has changed.)
	bli_pool_finalize( pool );

	// Reinitialize the pool with the new parameters, in particular,
	// the new block size.
	bli_pool_init( num_blocks_new,
	               block_ptrs_len_new,
	               block_size_new,
	               align_size_new,
	               malloc_fp,
	               free_fp,
	               pool );
}

void bli_pool_checkout_block
     (
       siz_t   req_size,
       pblk_t* block,
       pool_t* pool
     )
{
	pblk_t* block_ptrs;
	dim_t   top_index;

	if ( bli_pool_block_size( pool ) < req_size )
	{
		const dim_t num_blocks_new     = bli_pool_num_blocks( pool );
		const dim_t block_ptrs_len_new = bli_pool_block_ptrs_len( pool );
		const siz_t align_size_new     = bli_pool_align_size( pool );

		#ifdef ENABLE_MEM_DEBUG
		printf( "bli_pool_checkout_block(): old block size %ld < req size %ld; "
		        "reiniting",
		        ( long )bli_pool_block_size( pool ), ( long )req_size );
		#endif

		// If the requested block size is smaller than what the pool
		// was initialized with, reinitialize the pool to contain blocks
		// of the requested size.
		bli_pool_reinit( num_blocks_new,
		                 block_ptrs_len_new,
		                 req_size,
		                 align_size_new,
		                 pool );
	}

	// If the pool is exhausted, add a block.
	if ( bli_pool_is_exhausted( pool ) )
	{
		#ifdef ENABLE_MEM_DEBUG
		printf( "bli_pool_checkout_block(): pool is exhausted (block size %d); "
		        "growing by 1.\n", ( int )bli_pool_block_size( pool ) );
		#endif

		bli_pool_grow( 1, pool );
	}

	// At this point, at least one block is guaranteed to be available.

	// Query the current block_ptrs array.
	block_ptrs = bli_pool_block_ptrs( pool );

	// Query the top_index of the pool.
	top_index = bli_pool_top_index( pool );

	#ifdef ENABLE_MEM_DEBUG
	printf( "bli_pool_checkout_block(): checking out block %d of size %ld (align %ld)\n",
	        ( int )top_index, ( long )bli_pool_block_size( pool ),
	                          ( long )bli_pool_align_size( pool ) );
	#endif

	// Copy the block at top_index to the caller's pblk_t struct.
	//bli_pblk_copy( *(block_ptrs[top_index]), *block );
	*block = block_ptrs[top_index];

	// Notice that we don't actually need to clear the contents of
	// block_ptrs[top_index]. It will get overwritten eventually when
	// the block is checked back in.
	bli_pblk_clear( &block_ptrs[top_index] );

	// Increment the pool's top_index.
	bli_pool_set_top_index( top_index + 1, pool );
}

void bli_pool_checkin_block
     (
       pblk_t* block,
       pool_t* pool
     )
{
	pblk_t* block_ptrs;
	dim_t   top_index;

	// Query the current block_ptrs array.
	block_ptrs = bli_pool_block_ptrs( pool );

	// Query the top_index of the pool.
	top_index = bli_pool_top_index( pool );

	#ifdef ENABLE_MEM_DEBUG
	printf( "bli_pool_checkin_block(): checking in block %d of size %ld (align %ld)\n",
	        ( int )top_index - 1, ( long )bli_pool_block_size( pool ),
	                              ( long )bli_pool_align_size( pool ) );
	#endif

	// Copy the caller's pblk_t struct to the block at top_index - 1.
	//bli_pblk_copy( *(block_ptrs[top_index-1]), *block );
	block_ptrs[top_index-1] = *block;

	// Decrement the pool's top_index.
	bli_pool_set_top_index( top_index - 1, pool );
}

void bli_pool_grow
     (
       dim_t   num_blocks_add,
       pool_t* pool
     )
{
	pblk_t* block_ptrs_cur;
	dim_t   block_ptrs_len_cur;
	dim_t   num_blocks_cur;

	pblk_t* block_ptrs_new;
	dim_t   num_blocks_new;

	siz_t   block_size;
	siz_t   align_size;
	dim_t   top_index;

	dim_t   i;

	// If the requested increase is zero (or negative), return early.
	if ( num_blocks_add < 1 ) return;

	// Query the allocated length of the block_ptrs array and also the
	// total number of blocks allocated.
	block_ptrs_len_cur = bli_pool_block_ptrs_len( pool );
	num_blocks_cur     = bli_pool_num_blocks( pool );

	// Compute the total number of allocated blocks that will exist
	// after we grow the pool.
	num_blocks_new = num_blocks_cur + num_blocks_add;

	// If the new total number of allocated blocks is larger than the
	// allocated length of the block_ptrs array, we need to allocate
	// a new (larger) block_ptrs array.
	if ( block_ptrs_len_cur < num_blocks_new )
	{
		#ifdef ENABLE_MEM_DEBUG
		printf( "bli_pool_grow(): growing block_ptrs_len from %d to %d.\n",
		        ( int )block_ptrs_len_cur, ( int )num_blocks_new );
		#endif

		// Query the current block_ptrs array.
		block_ptrs_cur = bli_pool_block_ptrs( pool );

		// Allocate a new block_ptrs array of length num_blocks_new.
		block_ptrs_new = bli_malloc_intl( num_blocks_new * sizeof( pblk_t ) );

		// Query the top_index of the pool.
		top_index = bli_pool_top_index( pool );

		// Copy the contents of the old block_ptrs array to the new/resized
		// array. Notice that we can begin with top_index since all entries
		// from 0 to top_index-1 have been checked out to threads.
		for ( i = top_index; i < num_blocks_cur; ++i ) 
		{
			block_ptrs_new[i] = block_ptrs_cur[i];
		}

		#ifdef ENABLE_MEM_DEBUG
		printf( "bli_pool_grow(): freeing previous block_ptrs array.\n" );
		#endif

		// Free the old block_ptrs array.
		bli_free_intl( block_ptrs_cur );

		// Update the pool_t struct with the new block_ptrs array and
		// record its allocated length.
		bli_pool_set_block_ptrs( block_ptrs_new, pool );
		bli_pool_set_block_ptrs_len( num_blocks_new, pool );
	}

	// At this point, we are guaranteed to have enough unused elements
	// in the block_ptrs array to accommodate an additional num_blocks_add
	// blocks.

	// Query the current block_ptrs array (which was possibly just resized).
	block_ptrs_cur = bli_pool_block_ptrs( pool );

	// Query the block size and alignment size of the pool.
	block_size = bli_pool_block_size( pool );
	align_size = bli_pool_align_size( pool );

	#ifdef ENABLE_MEM_DEBUG
	printf( "bli_pool_grow(): growing pool from from %d to %d.\n",
	        ( int )num_blocks_cur, ( int )num_blocks_new );
	#endif

	// Allocate the requested additional blocks in the resized array.
	for ( i = num_blocks_cur; i < num_blocks_new; ++i ) 
	{
		bli_pool_alloc_block( block_size, align_size,
		                      &(block_ptrs_cur[i]), pool );
	}

	// Update the pool_t struct with the new number of allocated blocks.
	// Notice that top_index remains unchanged, as do the block_size and
	// align_size fields.
	bli_pool_set_num_blocks( num_blocks_new, pool );
}

void bli_pool_shrink
     (
       dim_t   num_blocks_sub,
       pool_t* pool
     )
{
	pblk_t* block_ptrs;
	dim_t   num_blocks;
	dim_t   num_blocks_avail;
	dim_t   num_blocks_new;

	dim_t   top_index;

	dim_t   i;

	// Query the total number of blocks presently allocated.
	num_blocks = bli_pool_num_blocks( pool );

	// Query the top_index of the pool.
	top_index = bli_pool_top_index( pool );

	// Compute the number of blocks available to be checked out
	// (and thus available for removal).
	num_blocks_avail = num_blocks - top_index;

	// If the requested decrease is more than the number of available
	// blocks in the pool, only remove the number of blocks available.
	if ( num_blocks_avail < num_blocks_sub )
		num_blocks_sub = num_blocks_avail;

	// If the effective requested decrease is zero (or the requested
	// decrease was negative), return early.
	if ( num_blocks_sub < 1 ) return;

	// Query the current block_ptrs array.
	block_ptrs = bli_pool_block_ptrs( pool );

	// Compute the new total number of blocks.
	num_blocks_new = num_blocks - num_blocks_sub;

	// Free the individual blocks.
	for ( i = num_blocks_new; i < num_blocks; ++i )
	{
		bli_pool_free_block( &(block_ptrs[i]), pool );
	}

	// Update the pool_t struct.
	bli_pool_set_num_blocks( num_blocks_new, pool );

	// Note that after shrinking the pool, num_blocks < block_ptrs_len.
	// This means the pool can grow again by num_blocks_sub before
	// a re-allocation of block_ptrs is triggered.
}

void bli_pool_alloc_block
     (
       siz_t   block_size,
       siz_t   align_size,
       pblk_t* block,
       pool_t* pool
     )
{
	#ifdef ENABLE_MEM_DEBUG
	//printf( "bli_pool_alloc_block(): allocating block of size %ld (align %ld)\n",
	//        ( long )block_size, ( long )align_size );
	#endif

	// Query the malloc() function pointer from the pool.
	malloc_ft malloc_fp = bli_pool_malloc_fp( pool );

	// Allocate the block via the bli_fmalloc_align() wrapper, which performs
	// alignment logic and opaquely saves the original pointer so that it can
	// be recovered when it's time to free the block.
	void* buf = bli_fmalloc_align( malloc_fp, block_size, align_size );

#if 0
	// NOTE: This code is disabled because it is not needed, since
	// bli_fmalloc_align() is guaranteed to return an aligned address.

	// Advance the pointer to achieve the necessary alignment, if it is not
	// already aligned.
	if ( bli_is_unaligned_to( ( siz_t )buf_sys, ( siz_t )align_size ) )
	{
		// C99's stdint.h guarantees that a void* can be safely cast to a
		// uintptr_t and then back to a void*, hence the casting of buf_sys
		// and align_size to uintptr_t. buf_align is initially cast to char*
		// to allow pointer arithmetic in units of bytes, and then advanced
		// to the next nearest alignment boundary, and finally cast back to
		// void* before being stored. Notice that the arithmetic works even
		// if the alignment value is not a power of two.
		buf_align = ( void* )(   ( char*     )buf_align +
		                       ( ( uintptr_t )align_size -
		                         ( uintptr_t )buf_sys %
		                         ( uintptr_t )align_size )
		                     );
	}
#endif

	//printf( "bli_pool_alloc_block(): bsize = %d; asize = %d\n", (int)block_size, (int)align_size );
	//printf( "                        sys   = %p; align = %p\n", buf_sys, buf_align );
	
	// Save the results in the pblk_t structure.
	bli_pblk_set_buf( buf, block );
}

void bli_pool_free_block
     (
       pblk_t* block,
       pool_t* pool
     )
{
	void* buf;

	#ifdef ENABLE_MEM_DEBUG
	printf( "bli_pool_free_block(): freeing block.\n" );
	#endif

	// Query the free() function pointer from the pool.
	free_ft free_fp = bli_pool_free_fp( pool );

	// Extract the pblk_t buffer, which is the aligned address returned from
	// bli_fmalloc_align() when the block was allocated.
	buf = bli_pblk_buf( block );

	// Free the block via the bli_ffree_align() wrapper, which recovers the
	// original pointer that was returned by the pool's malloc() function when
	// the block was allocated.
	bli_ffree_align( free_fp, buf );
}

void bli_pool_print
     (
       pool_t* pool
     )
{
	pblk_t* block_ptrs     = bli_pool_block_ptrs( pool );
	dim_t   block_ptrs_len = bli_pool_block_ptrs_len( pool );
	dim_t   top_index      = bli_pool_top_index( pool );
	dim_t   num_blocks     = bli_pool_num_blocks( pool );
	dim_t   block_size     = bli_pool_block_size( pool );
	dim_t   align_size     = bli_pool_align_size( pool );
	dim_t   i;

	printf( "pool struct ---------------\n" );
	printf( "  block_ptrs:      %p\n", block_ptrs );
	printf( "  block_ptrs_len:  %ld\n", ( long )block_ptrs_len );
	printf( "  top_index:       %ld\n", ( long )top_index );
	printf( "  num_blocks:      %ld\n", ( long )num_blocks );
	printf( "  block_size:      %ld\n", ( long )block_size );
	printf( "  align_size:      %ld\n", ( long )align_size );
	printf( "  pblks   sys    align\n" );
	for ( i = 0; i < num_blocks; ++i )
	{
		printf( "  %ld: %p\n", ( long )i, bli_pblk_buf( &block_ptrs[i] ) );
	}
}

void bli_pblk_print
     (
       pblk_t* pblk
     )
{
	void* buf = bli_pblk_buf( pblk );

	printf( "pblk struct ---------------\n" );
	printf( "  block address (aligned): %p\n", buf );
}

