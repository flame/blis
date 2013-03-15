/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2013, The University of Texas

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
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

#include "blis2.h"


// Define the size of pool blocks. These may be adjusted so that they can
// handle inflated blocksizes at edge cases.
#define BLIS_POOL_MC_Z     BLIS_DEFAULT_MC_Z
#define BLIS_POOL_KC_Z     BLIS_DEFAULT_KC_Z
#define BLIS_POOL_NC_Z     BLIS_DEFAULT_NC_Z

// Define each pool's block size.
#define BLIS_MK_BLOCK_SIZE ( BLIS_POOL_MC_Z * \
                             BLIS_POOL_KC_Z * \
                             sizeof( dcomplex ) \
                           )
#define BLIS_KN_BLOCK_SIZE ( BLIS_POOL_KC_Z * \
                             BLIS_POOL_NC_Z * \
                             sizeof( dcomplex ) \
                           )
#define BLIS_MN_BLOCK_SIZE ( BLIS_POOL_MC_Z * \
                             BLIS_POOL_NC_Z * \
                             sizeof( dcomplex ) \
                           )

// Define each pool's total size.
#define BLIS_MK_POOL_SIZE  ( \
                             BLIS_NUM_MC_X_KC_BLOCKS * \
                             ( BLIS_MK_BLOCK_SIZE + \
                               BLIS_PAGE_SIZE \
                             ) + \
                             BLIS_MAX_PREFETCH_BYTE_OFFSET \
                           )

#define BLIS_KN_POOL_SIZE  ( \
                             BLIS_NUM_KC_X_NC_BLOCKS * \
                             ( BLIS_KN_BLOCK_SIZE + \
                               BLIS_PAGE_SIZE \
                             ) + \
                             BLIS_MAX_PREFETCH_BYTE_OFFSET \
                           )

#define BLIS_MN_POOL_SIZE  ( \
                             BLIS_NUM_MC_X_NC_BLOCKS * \
                             ( BLIS_MN_BLOCK_SIZE + \
                               BLIS_PAGE_SIZE \
                             ) + \
                             BLIS_MAX_PREFETCH_BYTE_OFFSET \
                           )

// Declare one memory pool structure for each block size/shape we want to
// be able to allocate.

static pool_t pools[3];


// Physically contiguous memory for each pool.

static void*  pool_mk_blk_ptrs[ BLIS_NUM_MC_X_KC_BLOCKS ];
static char   pool_mk_mem[ BLIS_MK_POOL_SIZE ];

static void*  pool_kn_blk_ptrs[ BLIS_NUM_KC_X_NC_BLOCKS ];
static char   pool_kn_mem[ BLIS_KN_POOL_SIZE ];

static void*  pool_mn_blk_ptrs[ BLIS_NUM_MC_X_NC_BLOCKS ];
static char   pool_mn_mem[ BLIS_MN_POOL_SIZE ];





void bl2_mem_acquire_m( dim_t     m_req,
                        dim_t     n_req,
                        siz_t     elem_size,
                        packbuf_t buf_type,
                        mem_t*    mem )
{
	siz_t   req_size;
	siz_t   block_size;
	dim_t   pool_index;
	pool_t* pool;
	void**  block_ptrs;
	void*   block;
	int     i;


	// Compute the size of the requested contiguous memory region.
	req_size = m_req * n_req * elem_size;

	if ( buf_type == BLIS_BUFFER_FOR_GEN_USE )
	{
		// For general-use buffer requests, such as those used by level-2
		// operations, using bl2_malloc() is sufficient, since using
		// physically contiguous memory is not as important there.
		block = bl2_malloc( req_size );

		// Initialize the mem_t object with:
		// - the address of the memory block,
		// - the buffer type (a packbuf_t value),
		// - the size of the requested region, and
		// - the requested dimensions, which are presumably already aligned to
		//   dimension multiples (typically equal to register blocksizes).
		// NOTE: We do not initialize the pool field since this block did not
		// come from a contiguous memory pool.
		bl2_mem_set_buffer( block, mem );
		bl2_mem_set_buf_type( buf_type, mem );
		bl2_mem_set_size( req_size, mem );
		bl2_mem_set_dims( m_req, n_req, mem );
		bl2_mem_set_elem_size( elem_size, mem );
	}
	else
	{
		// This branch handles cases where the memory block needs to come
		// from one of the contiguous memory pools.

		// Map the requested packed buffer type to a zero-based index, which
		// we then use to select the corresponding memory pool.
		pool_index = bl2_packbuf_index( buf_type );
		pool       = &pools[ pool_index ];

		// Perform error checking, if enabled.
		if ( bl2_error_checking_is_enabled() )
		{
			err_t e_val;

			// Make sure that the requested matrix size fits inside of a block
			// of the corresponding pool.
			e_val = bl2_check_requested_block_size_for_pool( req_size, pool );
			bl2_check_error_code( e_val );

			// Make sure that the pool contains at least one block to check out
			// to the thread.
			e_val = bl2_check_if_exhausted_pool( pool );
			bl2_check_error_code( e_val );
		}

		// Access the block pointer array from the memory pool data structure.
		block_ptrs = bl2_pool_block_ptrs( pool );


		// BEGIN CRITICAL SECTION


		// Query the index of the contiguous memory block that resides at the
		// "top" of the pool.
		i = bl2_pool_top_index( pool );
	
		// Extract the address of the top block from the block pointer array.
		block = block_ptrs[i];

		// Clear the entry from the block pointer array. (This is actually not
		// necessary.)
		//block_ptrs[i] = NULL; 

		// Decrement the top of the memory pool.
		bl2_pool_dec_top_index( pool );


		// END CRITICAL SECTION

		// Query the size of the blocks in the pool so we can store it in the
		// mem_t object.
		block_size = bl2_pool_block_size( pool );

		// Initialize the mem_t object with:
		// - the address of the memory block,
		// - the buffer type (a packbuf_t value),
		// - the address of the memory pool to which it belongs,
		// - the size of the contiguous memory block (NOT the size of the
		//   requested region), and
		// - the requested dimensions, which are presumably already aligned to
		//   dimension multiples (typically equal to register blocksizes).
		bl2_mem_set_buffer( block, mem );
		bl2_mem_set_buf_type( buf_type, mem );
		bl2_mem_set_pool( pool, mem );
		bl2_mem_set_size( block_size, mem );
		bl2_mem_set_dims( m_req, n_req, mem );
		bl2_mem_set_elem_size( elem_size, mem );
	}
}


void bl2_mem_release( mem_t* mem )
{
	packbuf_t buf_type;
	pool_t*   pool;
	void**    block_ptrs;
	void*     block;
	int       i;

	// Extract the address of the memory block we are trying to
	// release.
	block = bl2_mem_buffer( mem );

	// Extract the buffer type so we know what kind of memory was allocated.
	buf_type = bl2_mem_buf_type( mem );

	if ( buf_type == BLIS_BUFFER_FOR_GEN_USE )
	{
		// For general-use buffers, we allocate with bl2_malloc(), and so
		// here we need to call bl2_free().
		bl2_free( block );
	}
	else
	{
		// This branch handles cases where the memory block came from one
		// of the contiguous memory pools.

		// Extract the pool from which the block was allocated.
		pool = bl2_mem_pool( mem );

		// Extract the block pointer array associated with the pool.
		block_ptrs = bl2_pool_block_ptrs( pool );


		// BEGIN CRITICAL SECTION


		// Increment the top of the memory pool.
		bl2_pool_inc_top_index( pool );

		// Query the newly incremented top index.
		i = bl2_pool_top_index( pool );

		// Place the address of the block back onto the top of the memory pool.
		block_ptrs[i] = block;


		// END CRITICAL SECTION
	}


	// Clear the mem_t object so that it appears unallocated. We clear:
	// - the buffer field,
	// - the pool field,
	// - the size field, and
	// - the dimension fields.
	// NOTE: We do not clear the buf_type field since there is no
	// "uninitialized" value for packbuf_t.
	bl2_mem_set_buffer( NULL, mem );
	bl2_mem_set_pool( NULL, mem );
	bl2_mem_set_size( 0, mem );
	bl2_mem_set_dims( 0, 0, mem );
	bl2_mem_set_elem_size( 0, mem );
}


void bl2_mem_acquire_v( dim_t     m_req,
                        siz_t     elem_size,
                        mem_t*    mem )
{
	bl2_mem_acquire_m( m_req,
	                   1,
	                   elem_size,
	                   BLIS_BUFFER_FOR_GEN_USE,
	                   mem );
}



void bl2_mem_init()
{
	dim_t index_a;
	dim_t index_b;
	dim_t index_c;

	// Map each of the packbuf_t values to an index starting at zero.
	index_a = bl2_packbuf_index( BLIS_BUFFER_FOR_A_BLOCK );
	index_b = bl2_packbuf_index( BLIS_BUFFER_FOR_B_PANEL );
	index_c = bl2_packbuf_index( BLIS_BUFFER_FOR_C_PANEL );

	// Initialize contiguous memory pool for MC x KC blocks.
	bl2_mem_init_pool( pool_mk_mem,
	                   BLIS_MK_BLOCK_SIZE,
	                   BLIS_NUM_MC_X_KC_BLOCKS,
	                   pool_mk_blk_ptrs,
	                   &pools[ index_a ] );

	// Initialize contiguous memory pool for KC x NC blocks.
	bl2_mem_init_pool( pool_kn_mem,
	                   BLIS_KN_BLOCK_SIZE,
	                   BLIS_NUM_KC_X_NC_BLOCKS,
	                   pool_kn_blk_ptrs,
	                   &pools[ index_b ] );

	// Initialize contiguous memory pool for MC x NC blocks.
	bl2_mem_init_pool( pool_mn_mem,
	                   BLIS_MN_BLOCK_SIZE,
	                   BLIS_NUM_MC_X_NC_BLOCKS,
	                   pool_mn_blk_ptrs,
	                   &pools[ index_c ] );
}


void bl2_mem_init_pool( char*   pool_mem,
                        siz_t   block_size,
                        dim_t   num_blocks,
                        void**  block_ptrs,
                        pool_t* pool )
{
	dim_t i;

	// If the pool starting address is not already aligned to the page size,
	// advance it to the beginning of the next page. (Here, we assign that
	// the page size is a multiple of the memory alignment boundary.)
	if ( bl2_is_unaligned_to( pool_mem, BLIS_PAGE_SIZE ) )
	{
		// Notice that this works even if the page size is not a power of two.
		pool_mem += ( BLIS_PAGE_SIZE - ( ( siz_t )pool_mem % BLIS_PAGE_SIZE ) );
	}

	// Step through the memory pool, beginning with the page-aligned address
	// determined above, assigning pointers to the beginning of each m x n
	// block to the ith element of the block_ptrs array.
	for ( i = 0; i < num_blocks; ++i )
	{
		// Save the address of pool, which is guaranteed to be page-aligned.
		block_ptrs[i] = pool_mem;

		// Advance pool by one block.
		pool_mem += block_size;

		// Advance pool a bit further if needed in order to get to the
		// beginning of a page.
		if ( bl2_is_unaligned_to( pool_mem, BLIS_PAGE_SIZE ) )
		{
			pool_mem += ( BLIS_PAGE_SIZE - ( ( siz_t )pool_mem % BLIS_PAGE_SIZE ) );
		}
	}

	// Now that we have initialized the array of pointers to the individual
	// blocks in the pool, we initialize a pool_t data structure so that we
	// can easily manage this pool.
	bl2_pool_init( num_blocks,
	               block_size,
	               block_ptrs,
	               pool );
}



void bl2_mem_finalize()
{
	// Nothing to do.
}

