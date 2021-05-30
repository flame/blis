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

#include "bls_gemm.hpp"

#include "bls_gemm_var.hpp"

//
// -- Define the gemm-like operation's object API ------------------------------
//

void bls_gemm_ex
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
	bli_init_once();

	// -- bli_gemmnat() --------------------------------------------------------

	// Obtain a valid (native) context from the gks if necessary.
	// NOTE: This must be done before calling the _check() function, since
	// that function assumes the context pointer is valid.
	if ( cntx == NULL ) cntx = bli_gks_query_cntx();

	// Initialize a local runtime with global settings if necessary. Note
	// that in the case that a runtime is passed in, we make a local copy.
	rntm_t rntm_l;
	if ( rntm == NULL ) { bli_rntm_init_from_global( &rntm_l ); rntm = &rntm_l; }
	else                { rntm_l = *rntm;                       rntm = &rntm_l; }

    // Set the packing block allocator field of the rntm. This will be
    // inherited by all of the child threads when they make local copies of
    // the rntm below.
    bli_membrk_rntm_set_membrk( rntm );

	// -- bli_gemm_front() -----------------------------------------------------

	obj_t a_local;
	obj_t b_local;
	obj_t c_local;

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
	{
		bli_gemm_check( alpha, a, b, beta, c, cntx );
	}

	// If C has a zero dimension, return early.
	if ( bli_obj_has_zero_dim( c ) )
	{
		return;
	}

	// If alpha is zero, or if A or B has a zero dimension, scale C by beta
	// and return early.
	if ( bli_obj_equals( alpha, &BLIS_ZERO ) ||
	     bli_obj_has_zero_dim( a ) ||
	     bli_obj_has_zero_dim( b ) )
	{
		bli_scalm( beta, c );
		return;
	}

	// Alias A, B, and C in case we need to apply transformations.
	bli_obj_alias_to( a, &a_local );
	bli_obj_alias_to( b, &b_local );
	bli_obj_alias_to( c, &c_local );

	// Induce a transposition of A if it has its transposition property set.
	// Then clear the transposition bit in the object.
	if ( bli_obj_has_trans( &a_local ) )
	{
		bli_obj_induce_trans( &a_local );
		bli_obj_set_onlytrans( BLIS_NO_TRANSPOSE, &a_local );
	}

	// Induce a transposition of B if it has its transposition property set.
	// Then clear the transposition bit in the object.
	if ( bli_obj_has_trans( &b_local ) )
	{
		bli_obj_induce_trans( &b_local );
		bli_obj_set_onlytrans( BLIS_NO_TRANSPOSE, &b_local );
	}

	// An optimization: If C is stored by rows and the micro-kernel prefers
	// contiguous columns, or if C is stored by columns and the micro-kernel
	// prefers contiguous rows, transpose the entire operation to allow the
	// micro-kernel to access elements of C in its preferred manner.
	if ( bli_cntx_l3_vir_ukr_dislikes_storage_of( &c_local, BLIS_GEMM_UKR, cntx ) )
	{
		bli_obj_swap( &a_local, &b_local );

		bli_obj_induce_trans( &a_local );
		bli_obj_induce_trans( &b_local );
		bli_obj_induce_trans( &c_local );

		// NOTE: This is probably not needed within the sandbox.
		// We must also swap the pack schemas, which were set by bli_gemm_md()
		// or the inlined code above.
		//bli_obj_swap_pack_schemas( &a_local, &b_local );
	}

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

	tci::parallelize(
	[&](const communicator& thread)
	{
	    num_t dt = bli_obj_dt( &c_local );

	    bls_gemm_bp_var2
	    (
          dt,
          bli_obj_conj_status( &a_local ),
          bli_obj_conj_status( &b_local ),
          bli_obj_length( &c_local ),
          bli_obj_width( &c_local ),
          bli_obj_width( &a_local ),
          (char*)bli_obj_buffer_for_1x1( dt, alpha ),
          (char*)bli_obj_buffer_at_off( &a_local ),
          bli_obj_row_stride( &a_local ), bli_obj_col_stride( &a_local ),
          (char*)bli_obj_buffer_at_off( &b_local ),
          bli_obj_row_stride( &b_local ), bli_obj_col_stride( &b_local ),
          (char*)bli_obj_buffer_for_1x1( dt, beta ),
          (char*)bli_obj_buffer_at_off( &c_local ),
          bli_obj_row_stride( &c_local ), bli_obj_col_stride( &c_local ),
          cntx,
          rntm,
          thread
        );
	},
	bli_rntm_num_threads( rntm ) );
}

