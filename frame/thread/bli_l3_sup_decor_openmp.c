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

#ifdef BLIS_ENABLE_OPENMP

// Define a dummy function bli_l3_sup_thread_entry(), which is needed in the
// pthreads version, so that when building Windows DLLs (with OpenMP enabled
// or no multithreading) we don't risk having an unresolved symbol.
void* bli_l3_sup_thread_entry( void* data_void ) { return NULL; }

//#define PRINT_THRINFO

err_t bli_l3_sup_thread_decorator
     (
             l3supint_t func,
             opid_t     family,
       const obj_t*     alpha,
       const obj_t*     a,
       const obj_t*     b,
       const obj_t*     beta,
       const obj_t*     c,
       const cntx_t*    cntx,
       const rntm_t*    rntm
     )
{
	// Query the total number of threads from the rntm_t object.
	const dim_t n_threads = bli_rntm_num_threads( rntm );

	// NOTE: The sba was initialized in bli_init().

	// Check out an array_t from the small block allocator. This is done
	// with an internal lock to ensure only one application thread accesses
	// the sba at a time. bli_sba_checkout_array() will also automatically
	// resize the array_t, if necessary.
	array_t* array = bli_sba_checkout_array( n_threads );

	// Allcoate a global communicator for the root thrinfo_t structures.
	thrcomm_t* gl_comm = bli_thrcomm_create( NULL, n_threads );

	_Pragma( "omp parallel num_threads(n_threads)" )
	{
		// Create a thread-local copy of the master thread's rntm_t. This is
		// necessary since we want each thread to be able to track its own
		// small block pool_t as it executes down the function stack.
		rntm_t  rntm_l = *rntm;
		rntm_t* rntm_p = &rntm_l;

		// Query the thread's id from OpenMP.
		const dim_t   tid  = omp_get_thread_num();
              pool_t* pool = bli_apool_array_elem( tid, array );

		// Check for a somewhat obscure OpenMP thread-mistmatch issue.
		// NOTE: This calls the same function used for the conventional/large
		// code path.
		bli_l3_thread_decorator_thread_check( n_threads, tid, gl_comm, rntm_p );

		// Create the root node of the thread's thrinfo_t structure.
		thrinfo_t* thread = bli_l3_sup_thrinfo_create( tid, gl_comm, pool, rntm_p );

		func
		(
		  alpha,
		  a,
		  b,
		  beta,
		  c,
		  cntx,
		  rntm_p,
		  bli_thrinfo_sub_node( thread )
		);

		// Free the current thread's thrinfo_t structure.
		bli_thrinfo_free( thread );
	}

	// Free the global communicator, because the root thrinfo_t node
    // never frees its communicator.
    bli_thrcomm_free( NULL, gl_comm );

	// Check the array_t back into the small block allocator. Similar to the
	// check-out, this is done using a lock embedded within the sba to ensure
	// mutual exclusion.
	bli_sba_checkin_array( array );

	return BLIS_SUCCESS;
}

#endif

