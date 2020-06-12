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

err_t bli_gemmsup
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
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2);
	AOCL_DTL_LOG_GEMM_INPUTS(AOCL_DTL_LEVEL_TRACE_2, alpha, a, b, beta, c);

	// Return early if small matrix handling is disabled at configure-time.
	#ifdef BLIS_DISABLE_SUP_HANDLING
	AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "SUP is Disabled.");
	return BLIS_FAILURE;
	#endif

	// Return early if this is a mixed-datatype computation.
	if ( bli_obj_dt( c ) != bli_obj_dt( a ) ||
	     bli_obj_dt( c ) != bli_obj_dt( b ) ||
	     bli_obj_comp_prec( c ) != bli_obj_prec( c ) ) {
		AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "SUP doesn't support Mixed datatypes.");
		return BLIS_FAILURE;
	}


	const stor3_t stor_id = bli_obj_stor3_from_strides( c, a, b );

	/*General stride is not yet supported in sup*/
	if(BLIS_XXX==stor_id) {
		AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "SUP doesn't support general stride.");
		return BLIS_FAILURE;
	}

	const dim_t m  = bli_obj_length( c );
	const dim_t n  = bli_obj_width( c );
	trans_t transa = bli_obj_conjtrans_status( a );
	trans_t transb = bli_obj_conjtrans_status( b );

	//Don't use sup for currently unsupported storage types in cgemmsup
	if(bli_obj_is_scomplex(c) &&
	(((stor_id == BLIS_RRC)||(stor_id == BLIS_CRC))
	|| ((transa == BLIS_CONJ_NO_TRANSPOSE) || (transa == BLIS_CONJ_TRANSPOSE))
	|| ((transb == BLIS_CONJ_NO_TRANSPOSE) || (transb == BLIS_CONJ_TRANSPOSE))
	)){
		//printf(" gemmsup: Returning with for un-supported storage types and conjugate property in cgemmsup \n");
		AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "SUP - Unsuppported storage type for cgemm");
		return BLIS_FAILURE;
	}

	//Don't use sup for currently unsupported storage types  in zgemmsup
	if(bli_obj_is_dcomplex(c) &&
	(((stor_id == BLIS_RRC)||(stor_id == BLIS_CRC))
	|| ((transa == BLIS_CONJ_NO_TRANSPOSE) || (transa == BLIS_CONJ_TRANSPOSE))
	|| ((transb == BLIS_CONJ_NO_TRANSPOSE) || (transb == BLIS_CONJ_TRANSPOSE))
	)){
		//printf(" gemmsup: Returning with for un-supported storage types and conjugate property in zgemmsup \n");
		AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "SUP - Unsuppported storage type for zgemm.");
		return BLIS_FAILURE;
	}


	// Obtain a valid (native) context from the gks if necessary.
	// NOTE: This must be done before calling the _check() function, since
	// that function assumes the context pointer is valid.
	if ( cntx == NULL ) cntx = bli_gks_query_cntx();

	// Return early if a microkernel preference-induced transposition would
	// have been performed and shifted the dimensions outside of the space
	// of sup-handled problems.
	if ( bli_cntx_l3_vir_ukr_dislikes_storage_of( c, BLIS_GEMM_UKR, cntx ) )
	{
		const num_t dt = bli_obj_dt( c );
		const dim_t k  = bli_obj_width_after_trans( a );

		// Pass in m and n reversed, which simulates a transposition of the
		// entire operation pursuant to the microkernel storage preference.
		if ( !bli_cntx_l3_sup_thresh_is_met( dt, n, m, k, cntx ) ) {
			AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "SUP - Traspostion results in unsupported storage for matrix C.");
			return BLIS_FAILURE;
		}
	}
	else // ukr_prefers_storage_of( c, ... )
	{
		const num_t dt = bli_obj_dt( c );
		const dim_t k  = bli_obj_width_after_trans( a );

		if ( !bli_cntx_l3_sup_thresh_is_met( dt, m, n, k, cntx ) ) {
			AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "SUP - Unsupported storage for matrix C.");
			return BLIS_FAILURE;
		}
	}

	// Initialize a local runtime with global settings if necessary. Note
	// that in the case that a runtime is passed in, we make a local copy.
	rntm_t rntm_l;
	if ( rntm == NULL ) { bli_rntm_init_from_global( &rntm_l ); rntm = &rntm_l; }
	else                { rntm_l = *rntm;                       rntm = &rntm_l; }

#if 0
const num_t dt = bli_obj_dt( c );
const dim_t m  = bli_obj_length( c );
const dim_t n  = bli_obj_width( c );
const dim_t k  = bli_obj_width_after_trans( a );
const dim_t tm = bli_cntx_get_l3_sup_thresh_dt( dt, BLIS_MT, cntx );
const dim_t tn = bli_cntx_get_l3_sup_thresh_dt( dt, BLIS_NT, cntx );
const dim_t tk = bli_cntx_get_l3_sup_thresh_dt( dt, BLIS_KT, cntx );

printf( "dims: %d %d %d (threshs: %d %d %d)\n",
        (int)m, (int)n, (int)k, (int)tm, (int)tn, (int)tk );
#endif

	// We've now ruled out the following two possibilities:
	// - the ukernel prefers the operation as-is, and the sup thresholds are
	//   unsatisfied.
	// - the ukernel prefers a transposed operation, and the sup thresholds are
	//   unsatisfied after taking into account the transposition.
	// This implies that the sup thresholds (at least one of them) are met.
	// and the small/unpacked handler should be called.
	// NOTE: The sup handler is free to enforce a stricter threshold regime
	// if it so chooses, in which case it can/should return BLIS_FAILURE.

	// Query the small/unpacked handler from the context and invoke it.
	gemmsup_oft gemmsup_fp = bli_cntx_get_l3_sup_handler( BLIS_GEMM, cntx );

	return
	gemmsup_fp
	(
	  alpha,
	  a,
	  b,
	  beta,
	  c,
	  cntx,
	  rntm
	);
	
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2);
}


