/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2016 Hewlett Packard Enterprise Development LP

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas at Austin nor the names
      of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

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

#ifdef BLIS_ENABLE_PTHREADS
pthread_mutex_t mem_manager_mutex = PTHREAD_MUTEX_INITIALIZER;
#endif

static membrk_t global_membrk;

// -----------------------------------------------------------------------------

membrk_t* bli_mem_global_membrk( void )
{
	return &global_membrk;
}

siz_t bli_mem_pool_size( packbuf_t buf_type )
{
	siz_t r_val;

	if ( buf_type == BLIS_BUFFER_FOR_GEN_USE )
	{
		// We don't (yet) track the amount of general-purpose
		// memory that is currently allocated.
		r_val = 0;
	}
	else
	{
		dim_t   pool_index;
		pool_t* pool;

		// Acquire the pointer to the pool corresponding to the buf_type
		// provided.
		pool_index = bli_packbuf_index( buf_type );
		pool       = bli_membrk_pool( pool_index, &global_membrk );

		// Compute the pool "size" as the product of the block size
		// and the number of blocks in the pool.
		r_val = bli_pool_block_size( pool ) *
		        bli_pool_num_blocks( pool );
	}

	return r_val;
}

// -----------------------------------------------------------------------------

static bool_t bli_mem_is_init = FALSE;

void bli_mem_init( void )
{
	cntx_t cntx;

	// If the initialization flag is TRUE, we know the API is already
	// initialized, so we can return early.
	if ( bli_mem_is_init == TRUE ) return;

	// Create and initialize a context for gemm so we have something
	// to pass into bli_mem_init_pools().
	bli_gemm_cntx_init( &cntx );

#ifdef BLIS_ENABLE_OPENMP
	_Pragma( "omp critical (mem)" )
#endif
#ifdef BLIS_ENABLE_PTHREADS
	pthread_mutex_lock( &mem_manager_mutex );
#endif

	// BEGIN CRITICAL SECTION
	{
		// Here, we test the initialization flag again. NOTE: THIS IS NOT
		// REDUNDANT. This additional test is needed so that other threads
		// that may be waiting to acquire the lock do not perform any
		// initialization actions once they are finally allowed into this
		// critical section.
		if ( bli_mem_is_init == FALSE )
		{
			// Initialize the global membrk_t object and its memory pools.
			bli_membrk_init( &cntx, &global_membrk );

			// After initialization, mark the API as initialized.
			bli_mem_is_init = TRUE;
		}
	}
	// END CRITICAL SECTION

#ifdef BLIS_ENABLE_PTHREADS
	pthread_mutex_unlock( &mem_manager_mutex );
#endif

	// Finalize the temporary gemm context.
	bli_gemm_cntx_finalize( &cntx );
}

void bli_mem_reinit( cntx_t* cntx )
{
#ifdef BLIS_ENABLE_OPENMP
	_Pragma( "omp critical (mem)" )
#endif
#ifdef BLIS_ENABLE_PTHREADS
	pthread_mutex_lock( &mem_manager_mutex );
#endif

	// BEGIN CRITICAL SECTION
	{
		// If for some reason the memory pools have not yet been
		// initialized (unlikely), we emulate the body of bli_mem_init().
		if ( bli_mem_is_init == FALSE )
		{
			// Initialize the global membrk_t object and its memory pools.
			bli_membrk_init( cntx, &global_membrk );

			// After initialization, mark the API as initialized.
			bli_mem_is_init = TRUE;
		}
		else
		{
			// Reinitialize the global membrk_t object's memory pools.
			bli_membrk_reinit_pools( cntx, &global_membrk );
		}
	}
	// END CRITICAL SECTION

#ifdef BLIS_ENABLE_PTHREADS
	pthread_mutex_unlock( &mem_manager_mutex );
#endif
}

void bli_mem_finalize( void )
{
	// If the initialization flag is FALSE, we know the API is already
	// uninitialized, so we can return early.
	if ( bli_mem_is_init == FALSE ) return;

#ifdef BLIS_ENABLE_OPENMP
	_Pragma( "omp critical (mem)" )
#endif
#ifdef BLIS_ENABLE_PTHREADS
	pthread_mutex_lock( &mem_manager_mutex );
#endif

	// BEGIN CRITICAL SECTION
	{
		// Here, we test the initialization flag again. NOTE: THIS IS NOT
		// REDUNDANT. This additional test is needed so that other threads
		// that may be waiting to acquire the lock do not perform any
		// finalization actions once they are finally allowed into this
		// critical section.
		if ( bli_mem_is_init == TRUE )
		{
			// Finalize the global membrk_t object and its memory pools.
			bli_membrk_finalize( &global_membrk );

			// After finalization, mark the API as uninitialized.
			bli_mem_is_init = FALSE;
		}
	}
	// END CRITICAL SECTION

#ifdef BLIS_ENABLE_PTHREADS
	pthread_mutex_unlock( &mem_manager_mutex );
#endif
}

bool_t bli_mem_is_initialized( void )
{
	return bli_mem_is_init;
}

