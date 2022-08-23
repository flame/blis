/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

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

// Statically initialize the mutex within the small block allocator.
// Note that the sba is an apool_t of array_t of pool_t.
static apool_t sba = { .mutex = BLIS_PTHREAD_MUTEX_INITIALIZER };

// A boolean that tracks whether bli_sba_init() has completed successfully.
static bool sba_is_init = FALSE;

// -----------------------------------------------------------------------------

bool bli_sba_is_init( void )
{
	return sba_is_init;
}

void bli_sba_mark_init( void )
{
	sba_is_init = TRUE;
}

void bli_sba_mark_uninit( void )
{
	sba_is_init = FALSE;
}

// -----------------------------------------------------------------------------

apool_t* bli_sba_query( void )
{
	return &sba;
}

// -----------------------------------------------------------------------------

err_t bli_sba_init( void )
{
	err_t r_val;

	// Sanity check: Return early if the API is already initialized.
	if ( bli_sba_is_init() ) return BLIS_SUCCESS;

	// Initialize the small block allocator.
	r_val = bli_apool_init( &sba );
	bli_check_return_if_failure( r_val );

	// Mark the API as initialized.
	bli_sba_mark_init();

	return BLIS_SUCCESS;
}

err_t bli_sba_finalize( void )
{
	err_t r_val;

	// Sanity check: Return early if the API is uninitialized.
	if ( !bli_sba_is_init() ) return BLIS_SUCCESS;

	// Finalize the small block allocator.
	r_val = bli_apool_finalize( &sba );
	bli_check_return_if_failure( r_val );

	// Mark the API as uninitialized.
	bli_sba_mark_uninit();

	return BLIS_SUCCESS;
}

// -----------------------------------------------------------------------------

err_t bli_sba_acquire
     (
       rntm_t* rntm,
       siz_t   req_size,
       void**  block
     )
{
	err_t r_val;

#ifdef BLIS_ENABLE_SBA_POOLS
	if ( rntm == NULL )
	{
		*block = bli_malloc_intl( req_size, &r_val );
		bli_check_return_if_failure( r_val );
	}
	else
	{
		pblk_t pblk;

		// Query the small block pool from the rntm.
		pool_t* pool = bli_rntm_sba_pool( rntm );

		// We don't expect NULL sba_pool pointers in the normal course of BLIS
		// operation. However, there are rare instances where it is convenient
		// to support use of bli_sba_acquire() without having to pass in a valid
		// sba pool data structure. The case that inspired this branch was the
		// gemm_ukr and related test modules in the BLIS testsuite. (There, it
		// is convenient to not have to checkout an array_t from the sba, and it
		// does no harm since the malloc() happens outside of the region that
		// would be timed.)
		if ( pool == NULL )
		{
		    *block = bli_malloc_intl( req_size, &r_val );
			bli_check_return_if_failure( r_val );
		}
		else
		{
			// Query the block_size of the pool_t so that we can request the exact
			// size present.
			const siz_t block_size = bli_pool_block_size( pool );

			// Sanity check: Make sure the requested size is no larger than the
			// block_size field of the pool.
			if ( block_size < req_size )
			{
				printf( "bli_sba_acquire(): ** pool block_size is %d but req_size is %d.\n",
				        ( int )block_size, ( int )req_size );
				bli_abort();
			}

			// Check out a block using the block_size queried above.
			r_val = bli_pool_checkout_block( block_size, &pblk, pool );
			bli_check_return_if_failure( r_val );

			// The block address is stored within the pblk_t.
			*block = bli_pblk_buf( &pblk );
		}
	}
#else

	*block = bli_malloc_intl( req_size, &r_val );
	bli_check_return_if_failure( r_val );

#endif

	// Return the address obtained from the pblk_t.
	return BLIS_SUCCESS;
}

void bli_sba_release
     (
       rntm_t* rntm,
       void*   block
     )
{
#ifdef BLIS_ENABLE_SBA_POOLS
	if ( rntm == NULL )
	{
		bli_free_intl( block );
	}
	else
	{
		// Query the small block pool from the rntm.
		pool_t* pool = bli_rntm_sba_pool( rntm );

		if ( pool == NULL )
		{
		    bli_free_intl( block );
		}
		else
		{
			pblk_t pblk;

			// Query the block_size field from the pool. This is not super-important
			// for this particular application of the pool_t (that is, the "leaf"
			// component of the sba), but it seems like good housekeeping to maintain
			// the block_size field of the pblk_t in case its ever needed/read.
			const siz_t block_size = bli_pool_block_size( pool );

			// Embed the block's memory address into a pblk_t, along with the
			// block_size queried from the pool.
			bli_pblk_set_buf( block, &pblk );
			bli_pblk_set_block_size( block_size, &pblk );

			// Check the pblk_t back into the pool_t. (It's okay that the pblk_t is
			// a local variable since its contents are copied into the pool's internal
			// data structure--an array of pblk_t.)
			bli_pool_checkin_block( &pblk, pool );
		}
	}
#else

	bli_free_intl( block );

#endif
}

// -----------------------------------------------------------------------------

err_t bli_sba_checkout_array
     (
             siz_t     n_threads,
       const array_t** array
     )
{
	err_t r_val;

	#ifndef BLIS_ENABLE_SBA_POOLS
	*array = NULL; return BLIS_SUCCESS;
	#endif

	r_val = bli_apool_checkout_array( n_threads, array, &sba );
	bli_check_return_if_failure( r_val );

	return BLIS_SUCCESS;
}

void bli_sba_checkin_array
     (
       array_t* array
     )
{
	#ifndef BLIS_ENABLE_SBA_POOLS
	return;
	#endif

	bli_apool_checkin_array( array, &sba );
}

// -----------------------------------------------------------------------------

err_t bli_sba_rntm_set_pool
     (
       siz_t    index,
       array_t* array,
       rntm_t*  rntm
     )
{
	#ifndef BLIS_ENABLE_SBA_POOLS
	bli_rntm_set_sba_pool( NULL, rntm );
	return;
	#endif

	pool_t* pool;

	// Query the pool_t* in the array_t corresponding to index.
	err_t r_val = bli_apool_array_elem( index, array, &pool );
	bli_check_return_if_failure( r_val );

	// Embed the pool_t* into the rntm_t.
	bli_rntm_set_sba_pool( pool, rntm );

	return BLIS_SUCCESS;
}


