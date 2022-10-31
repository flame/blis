/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

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

void bli_packm_int
     (
       const obj_t*     a,
             obj_t*     p,
       const cntx_t*    cntx,
       const cntl_t*    cntl,
             thrinfo_t* thread_par
     )
{
	bli_init_once();

	// Extract the function pointer from the object.
	packm_var_oft f = bli_obj_pack_fn( a );

	// Barrier so that we know threads are done with previous computation
	// with the same packing buffer before starting to pack.
	thrinfo_t* thread = bli_thrinfo_sub_node( thread_par );
	bli_thrinfo_barrier( thread );

	// Invoke the packm variant.
	// NOTE: The packing kernel uses two communicators: one which represents a
	// single workgroup of many threads, and one which represents a group of
	// many single-member workgroups. The former communicator is used for
	// barriers and thread communication (i.e. broadcasting the pack buffer
	// pointer), while the latter communicator is used for partitioning work.
	// This is because all of the thread range functions rely on the work_id
	// and number of workgroups (n_way). Thus, we pass along the parent
	// thrinfo_t node which has these two communicators as the sub-node and
	// sub-prenode, respectively.
	f
	(
	  a,
	  p,
	  cntx,
	  cntl,
	  thread_par
	);

	// Barrier so that packing is done before computation.
	bli_thrinfo_barrier( thread );
}

