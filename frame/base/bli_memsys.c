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
static pthread_mutex_t mem_manager_mutex = PTHREAD_MUTEX_INITIALIZER;
#endif

static membrk_t global_membrk;

// -----------------------------------------------------------------------------

membrk_t* bli_memsys_global_membrk( void )
{
	return &global_membrk;
}

// -----------------------------------------------------------------------------

static bool_t bli_memsys_is_init = FALSE;

void bli_memsys_init( void )
{
	cntx_t cntx;

	// If the initialization flag is TRUE, we know the API is already
	// initialized, so we can return early.
	if ( bli_memsys_is_init == TRUE ) return;

	// Create and initialize a context for gemm so we have something
	// to pass into bli_membrk_init_pools(). We use BLIS_DOUBLE for
	// the datatype, but the dt argument is actually only used when
	// initializing contexts for induced methods.
	bli_gemm_cntx_init( BLIS_DOUBLE, &cntx );

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
		if ( bli_memsys_is_init == FALSE )
		{
			// Initialize the global membrk_t object and its memory pools.
			bli_membrk_init( &cntx, &global_membrk );

			// After initialization, mark the API as initialized.
			bli_memsys_is_init = TRUE;
		}
	}
	// END CRITICAL SECTION

#ifdef BLIS_ENABLE_PTHREADS
	pthread_mutex_unlock( &mem_manager_mutex );
#endif

	// Finalize the temporary gemm context.
	bli_gemm_cntx_finalize( &cntx );
}

void bli_memsys_reinit( cntx_t* cntx )
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
		// initialized (unlikely), we emulate the body of bli_memsys_init().
		if ( bli_memsys_is_init == FALSE )
		{
			// Initialize the global membrk_t object and its memory pools.
			bli_membrk_init( cntx, &global_membrk );

			// After initialization, mark the API as initialized.
			bli_memsys_is_init = TRUE;
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

void bli_memsys_finalize( void )
{
	// If the initialization flag is FALSE, we know the API is already
	// uninitialized, so we can return early.
	if ( bli_memsys_is_init == FALSE ) return;

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
		if ( bli_memsys_is_init == TRUE )
		{
			// Finalize the global membrk_t object and its memory pools.
			bli_membrk_finalize( &global_membrk );

			// After finalization, mark the API as uninitialized.
			bli_memsys_is_init = FALSE;
		}
	}
	// END CRITICAL SECTION

#ifdef BLIS_ENABLE_PTHREADS
	pthread_mutex_unlock( &mem_manager_mutex );
#endif
}

bool_t bli_memsys_is_initialized( void )
{
	return bli_memsys_is_init;
}

