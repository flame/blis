/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2026, Advanced Micro Devices, Inc. All rights reserved.

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

#include <omp.h>

// Define a dummy function bli_l3_sup_thread_entry(), which is needed in the
// pthreads version, so that when building Windows DLLs (with OpenMP enabled
// or no multithreading) we don't risk having an unresolved symbol.
void* bli_l3_sup_thread_entry( void* data_void ) { return NULL; }

//#define PRINT_THRINFO

// Per-thread body of the sup decorator, factored out so it can be invoked
// either inside an OpenMP parallel region (multi-threaded) or directly inline
// (single-threaded fast path that avoids GOMP_parallel entirely).
static void bli_l3_sup_decor_body
     (
       dim_t       tid,
       l3supint_t  func,
       obj_t*      alpha,
       obj_t*      a,
       obj_t*      b,
       obj_t*      beta,
       obj_t*      c,
       cntx_t*     cntx,
       rntm_t*     rntm,
       array_t*    array,
       thrcomm_t*  gl_comm,
       dim_t       n_threads
     )
{
	// Create a thread-local copy of the master thread's rntm_t so each thread
	// can track its own small block pool_t as it executes down the stack.
	rntm_t           rntm_l = *rntm;
	rntm_t* restrict rntm_p = &rntm_l;

	// Check for a somewhat obscure OpenMP thread-mismatch issue. NOTE: This
	// calls the same function used for the conventional/large code path, and is
	// a no-op when n_threads == 1 (see the early return in the callee), which
	// is what makes it safe to call from the single-threaded fast path below,
	// where no OpenMP parallel region exists.
	bli_l3_thread_decorator_thread_check( n_threads, tid, gl_comm, rntm_p );

	// Use the thread id to access the appropriate pool_t* within the array_t,
	// and use it to set the sba_pool field within the rntm_t.
	bli_sba_rntm_set_pool( tid, array, rntm_p );

	thrinfo_t* thread = NULL;

	// Create the root node of the thread's thrinfo_t structure.
	bli_l3_sup_thrinfo_create_root( tid, gl_comm, rntm_p, &thread );

	func( alpha, a, b, beta, c, cntx, rntm_p, thread );

	// NOTE: Unlike the conventional path, no barrier is needed here before
	// freeing the thrinfo_t tree (sup pack buffers are stack-local and freed
	// inside func(); gl_comm is freed outside the region; sub-communicators
	// are freed only by their ochief). See the original PR #702 discussion.
	//
	// Free the current thread's thrinfo_t structure.
	bli_l3_sup_thrinfo_free( rntm_p, thread );
}

err_t bli_l3_sup_thread_decorator
     (
       l3supint_t func,
       opid_t     family,
       obj_t*     alpha,
       obj_t*     a,
       obj_t*     b,
       obj_t*     beta,
       obj_t*     c,
       cntx_t*    cntx,
       rntm_t*    rntm
     )
{
	// Query the total number of threads from the rntm_t object.
	const dim_t n_threads = bli_rntm_num_threads( rntm );

	// NOTE: The sba was initialized in bli_init().

	// Check out an array_t from the small block allocator. This is done
	// with an internal lock to ensure only one application thread accesses
	// the sba at a time. bli_sba_checkout_array() will also automatically
	// resize the array_t, if necessary.
	array_t* restrict array = bli_sba_checkout_array( n_threads );

	// Access the pool_t* for thread 0 and embed it into the rntm. We do
	// this up-front only so that we have the rntm_t.sba_pool field
	// initialized and ready for the global communicator creation below.
	bli_sba_rntm_set_pool( 0, array, rntm );

	// Set the packing block allocator field of the rntm. This will be
	// inherited by all of the child threads when they make local copies of
	// the rntm below.
	bli_pba_rntm_set_pba( rntm );

	// Allcoate a global communicator for the root thrinfo_t structures.
	thrcomm_t* restrict gl_comm = bli_thrcomm_create( rntm, n_threads );


	// LOCK 2 guard (sup path): when only one thread is requested, run the
	// operation inline WITHOUT entering an OpenMP parallel region at all.
	// A plain `omp parallel ... if(n_threads>1)` clause does NOT help here:
	// GCC still emits an unconditional GOMP_parallel() call, and libgomp's
	// global per-region bookkeeping serializes highly-concurrent single-
	// threaded gemm callers (each foreign application thread issuing its own
	// 1-thread gemm). Skipping the GOMP_parallel() call entirely removes that
	// serialization (measured ~34x aggregate throughput at 48 concurrent
	// single-threaded DGEMM callers on a 192-core EPYC).
	if ( n_threads == 1 )
	{
		bli_l3_sup_decor_body( 0, func, alpha, a, b, beta, c,
		                       cntx, rntm, array, gl_comm, n_threads );
	}
	else
	{
		_Pragma( "omp parallel num_threads(n_threads)" )
		{
			bli_l3_sup_decor_body( omp_get_thread_num(), func, alpha, a, b,
			                       beta, c, cntx, rntm, array, gl_comm, n_threads );
		}
	}


	// Now global communicator is not freed in bli_l3_sup_thrinfo_free().
	// Free the global communicator after the parallel region completes.
	// This ensures that no thread can be using gl_comm when it is freed,
	// avoiding a potential data race where the chief thread would free
	// gl_comm inside bli_l3_sup_thrinfo_free() while non-chief threads might
	// still hold pointers to it.
	bli_thrcomm_free(rntm, gl_comm);

	// Check the array_t back into the small block allocator. Similar to the
	// check-out, this is done using a lock embedded within the sba to ensure
	// mutual exclusion.
	bli_sba_checkin_array( array );

	return BLIS_SUCCESS;
}

#endif

