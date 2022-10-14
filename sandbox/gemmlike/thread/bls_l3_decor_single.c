/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2021, The University of Texas at Austin

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

#define SKIP_THRINFO_TREE

void bls_l3_thread_decorator_single
     (
       l3sbxint_ft func,
       opid_t      family,
       //pack_t      schema_a,
       //pack_t      schema_b,
       obj_t*      alpha,
       obj_t*      a,
       obj_t*      b,
       obj_t*      beta,
       obj_t*      c,
       cntx_t*     cntx,
       rntm_t*     rntm
     )
{
	// For sequential execution, we use only one thread.
	const dim_t n_threads = 1;

	// NOTE: The sba was initialized in bli_init().

	// Check out an array_t from the small block allocator. This is done
	// with an internal lock to ensure only one application thread accesses
	// the sba at a time. bli_sba_checkout_array() will also automatically
	// resize the array_t, if necessary.
	array_t* array = bli_sba_checkout_array( n_threads );

	// Allcoate a global communicator for the root thrinfo_t structures.
	thrcomm_t* gl_comm = &BLIS_SINGLE_COMM;

	{
		// There is only one thread id (for the thief thread).
		const dim_t tid = 0;

    	// Create the root node of the thread's thrinfo_t structure.
        pool_t*    pool   = bli_apool_array_elem( tid, array );
    	thrinfo_t* thread = bli_l3_sup_thrinfo_create( tid, gl_comm, pool, rntm );

		func
		(
		  alpha,
		  a,
		  b,
		  beta,
		  c,
		  cntx,
		  rntm,
	      bli_thrinfo_sub_node( thread )
		);

		// Free the current thread's thrinfo_t structure.
		bli_thrinfo_free( thread );
	}

	// We shouldn't free the global communicator since it was already freed
	// by the global communicator's chief thread in bli_l3_thrinfo_free()
	// (called above).

	// Check the array_t back into the small block allocator. Similar to the
	// check-out, this is done using a lock embedded within the sba to ensure
	// mutual exclusion.
	bli_sba_checkin_array( array );
}

