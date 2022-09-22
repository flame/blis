/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022, The University of Texas at Austin

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

// Initialize a function pointer array containing function addresses for
// each of the threading-specific level-3 sup thread decorators.

static l3_sup_decor_ft l3_sup_decor_fpa[ BLIS_NUM_THREAD_IMPLS ] =
{
	[BLIS_SINGLE] = bli_l3_sup_thread_decorator_single,
	[BLIS_OPENMP] =
#if   defined(BLIS_ENABLE_OPENMP)
	                bli_l3_sup_thread_decorator_openmp,
#elif defined(BLIS_ENABLE_PTHREADS)
	                NULL,
#else
	                NULL,
#endif
	[BLIS_POSIX]  =
#if   defined(BLIS_ENABLE_PTHREADS)
	                bli_l3_sup_thread_decorator_pthreads,
#elif defined(BLIS_ENABLE_OPENMP)
	                NULL,
#else
	                NULL,
#endif
};

// Define a dispatcher that chooses a threading-specific function from the
// above function pointer array.

err_t bli_l3_sup_thread_decorator
     (
             l3supint_ft func,
             opid_t      family,
       const obj_t*      alpha,
       const obj_t*      a,
       const obj_t*      b,
       const obj_t*      beta,
       const obj_t*      c,
       const cntx_t*     cntx,
             rntm_t*     rntm
     )
{
	rntm_t rntm_l;

	// Query the threading implementation and the number of threads requested.
	timpl_t ti = bli_rntm_thread_impl( rntm );
	dim_t   nt = bli_rntm_num_threads( rntm );

#ifdef BLIS_ENABLE_NT1_VIA_SINGLE
	if ( nt == 1 )
	{
		// An optimization. If the caller requests only one thread, force
		// the sequential level-3 thread decorator even if that means
		// overriding the caller's preferred threading implementation (as
		// communicated via the rntm_t).
		rntm_l = *rntm;
		ti = BLIS_SINGLE;
		bli_rntm_set_thread_impl( BLIS_SINGLE, &rntm_l );
		rntm = &rntm_l;
	}
#endif

	if ( 1 < nt && ti == BLIS_SINGLE )
	{
		// Here, we resolve conflicting information. The caller requested
		// a sequential threading implementation, but also requested more
		// than one thread. Here, we choose to favor the requested threading
		// implementation over the number of threads, and so reset all
		// parallelism parameters to 1.
		rntm_l = *rntm;
		nt = 1;
		bli_rntm_set_ways_only( 1, 1, 1, 1, 1, &rntm_l );
		bli_rntm_set_num_threads_only( 1, &rntm_l );
		rntm = &rntm_l;
	}

	// Use the timpl_t value to index into the corresponding function address
	// from the function pointer array.
	const l3_sup_decor_ft fp = l3_sup_decor_fpa[ ti ];

	// Call the threading-specific decorator function.
	return fp
	(
	  func,
	  family,
	  alpha,
	  a,
	  b,
	  beta,
	  c,
	  cntx,
	  rntm
	);
}

void bli_l3_sup_thread_decorator_check
     (
       rntm_t* rntm
     )
{
	bli_l3_sup_thread_decorator_check( rntm );
}

