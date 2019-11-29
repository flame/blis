/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2019, Advanced Micro Devices, Inc.

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

err_t bli_gemmsup_ref
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx,
       rntm_t* rntm
     )
{
	// This function implements the default gemmsup handler. If you are a
	// BLIS developer and wish to use a different gemmsup handler, please
	// register a different function pointer in the context in your
	// sub-configuration's bli_cntx_init_*() function.

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_gemm_check( alpha, a, b, beta, c, cntx );

#if 0
	// FGVZ: Will this be needed for constructing thrinfo_t's (recall: the
	// sba needs to be attached to the rntm; see below)? Or will those nodes
	// just be created "locally," in an exposed manner?

	// Parse and interpret the contents of the rntm_t object to properly
	// set the ways of parallelism for each loop, and then make any
	// additional modifications necessary for the current operation.
	bli_rntm_set_ways_for_op
	(
	  BLIS_GEMM,
	  BLIS_LEFT, // ignored for gemm/hemm/symm
	  bli_obj_length( &c_local ),
	  bli_obj_width( &c_local ),
	  bli_obj_width( &a_local ),
	  rntm
	);

	// FGVZ: the sba needs to be attached to the rntm. But it needs
	// to be done in the thread region, since it needs a thread id.
	//bli_sba_rntm_set_pool( tid, array, rntm_p );
#endif

#if 0
	printf( "rntm.pack_a = %d\n", ( int )bli_rntm_pack_a( rntm ) );
	printf( "rntm.pack_b = %d\n", ( int )bli_rntm_pack_b( rntm ) );
#endif
	//bli_rntm_set_pack_a( 0, rntm );
	//bli_rntm_set_pack_b( 0, rntm );

	// May not need these here since packm_sup infers the schemas based
	// on the stor3_t id. (This would also mean that they don't need to
	// be passed into the thread decorator below.)
	//pack_t schema_a = BLIS_PACKED_ROW_PANELS;
	//pack_t schema_b = BLIS_PACKED_COL_PANELS;


	return
	bli_l3_sup_thread_decorator
	(
	  bli_gemmsup_int,
	  BLIS_GEMM, // operation family id
	  //schema_a,
	  //schema_b,
	  alpha,
	  a,
	  b,
	  beta,
	  c,
	  cntx,
	  rntm
	);
}

