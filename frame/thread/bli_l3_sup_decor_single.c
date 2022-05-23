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

#ifndef BLIS_ENABLE_MULTITHREADING

void bli_l3_sup_thrinfo_init_single
     (
       pool_t*    sba_pool,
       pba_t*     pba,
       thrinfo_t* thread
     )
{
    bli_thrinfo_set_comm( &BLIS_SINGLE_COMM, thread );
    bli_thrinfo_set_thread_id( 0, thread );
    bli_thrinfo_set_n_way( 1, thread );
    bli_thrinfo_set_work_id( 0, thread );
    bli_thrinfo_set_free_comm( FALSE, thread );
    bli_thrinfo_set_sba_pool( sba_pool, thread );
    bli_thrinfo_set_pba( pba, thread );
    bli_thrinfo_set_sub_prenode( thread, thread );
    bli_thrinfo_set_sub_node( thread, thread );
}

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
	// For sequential execution, we use only one thread.
	const dim_t n_threads = 1;

	// NOTE: The sba was initialized in bli_init().

	// Check out an array_t from the small block allocator. This is done
	// with an internal lock to ensure only one application thread accesses
	// the sba at a time. bli_sba_checkout_array() will also automatically
	// resize the array_t, if necessary.
	array_t* array = bli_sba_checkout_array( n_threads );

	{
		// Create a special thrinfo_t structure which indicates
        // single-threaded execution for all nodes.
		thrinfo_t thread;
        pool_t*   sba_pool = bli_apool_array_elem( 0, array );
        pba_t*    pba      = bli_pba_query();
		bli_l3_sup_thrinfo_init_single( sba_pool, pba, &thread );

		func
		(
		  alpha,
		  a,
		  b,
		  beta,
		  c,
		  cntx,
		  rntm,
		  &thread
		);
	}

	// Check the array_t back into the small block allocator. Similar to the
	// check-out, this is done using a lock embedded within the sba to ensure
	// mutual exclusion.
	bli_sba_checkin_array( array );

	return BLIS_SUCCESS;
}

#endif

