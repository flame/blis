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
	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_gemm_check( alpha, a, b, beta, c, cntx );

#if 0
	// FGVZ: The datatype-specific variant is now responsible for checking for
	// alpha == 0.0.

	// If alpha is zero, scale by beta and return.
	if ( bli_obj_equals( alpha, &BLIS_ZERO ) )
	{
		bli_scalm( beta, c );
		return BLIS_SUCCESS;
	}
#endif

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
	// FGVZ: The datatype-specific variant is now responsible for inducing a
	// transposition, if needed.

	// Induce transpositions on A and/or B if either object is marked for
	// transposition. We can induce "fast" transpositions since they objects
	// are guaranteed to not have structure or be packed.
	if ( bli_obj_has_trans( a ) )
	{
		bli_obj_induce_fast_trans( a );
		bli_obj_toggle_trans( a );
	}
	if ( bli_obj_has_trans( b ) )
	{
		bli_obj_induce_fast_trans( b );
		bli_obj_toggle_trans( b );
	}
#endif

#if 0
	//bli_gemmsup_ref_var2
	//bli_gemmsup_ref_var1
	#if 0
	bli_gemmsup_ref_var1n
	#else
	#endif
	const stor3_t stor_id = bli_obj_stor3_from_strides( c, a, b );
	const bool_t  is_rrr_rrc_rcr_crr = ( stor_id == BLIS_RRR ||
	                                     stor_id == BLIS_RRC ||
	                                     stor_id == BLIS_RCR ||
	                                     stor_id == BLIS_CRR );
	if ( is_rrr_rrc_rcr_crr )
	{
		bli_gemmsup_ref_var2m
		(
		  BLIS_NO_TRANSPOSE, alpha, a, b, beta, c, stor_id, cntx, rntm
		);
	}
	else
	{
		bli_gemmsup_ref_var2m
		(
		  BLIS_TRANSPOSE, alpha, a, b, beta, c, stor_id, cntx, rntm
		);
	}
#else
	const stor3_t stor_id = bli_obj_stor3_from_strides( c, a, b );

	// Don't use the small/unpacked implementation if one of the matrices
	// uses general stride.
	if ( stor_id == BLIS_XXX ) return BLIS_FAILURE;

	const bool_t  is_rrr_rrc_rcr_crr = ( stor_id == BLIS_RRR ||
	                                     stor_id == BLIS_RRC ||
	                                     stor_id == BLIS_RCR ||
	                                     stor_id == BLIS_CRR );
	const bool_t  is_rcc_crc_ccr_ccc = !is_rrr_rrc_rcr_crr;

	const num_t   dt       = bli_obj_dt( c );
	const bool_t  row_pref = bli_cntx_l3_sup_ker_prefers_rows_dt( dt, stor_id, cntx );

	const bool_t  is_primary = ( row_pref ? is_rrr_rrc_rcr_crr
	                                      : is_rcc_crc_ccr_ccc );

	if ( is_primary )
	{
		// This branch handles:
		//  - rrr rrc rcr crr for row-preferential kernels
		//  - rcc crc ccr ccc for column-preferential kernels

		const dim_t m  = bli_obj_length( c );
		const dim_t n  = bli_obj_width( c );
		const dim_t NR = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx ); \
		const dim_t MR = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx ); \
		const dim_t mu = m / MR;
		const dim_t nu = n / NR;

		if ( mu >= nu )
		{
			// block-panel macrokernel; m -> mc, mr; n -> nc, nr: var2()
			bli_gemmsup_ref_var2m( BLIS_NO_TRANSPOSE,
			                       alpha, a, b, beta, c, stor_id, cntx, rntm );
		}
		else // if ( mu < nu )
		{
			// panel-block macrokernel; m -> nc*,mr; n -> mc*,nr: var1()
			bli_gemmsup_ref_var1n( BLIS_NO_TRANSPOSE,
			                       alpha, a, b, beta, c, stor_id, cntx, rntm );
		}
	}
	else
	{
		// This branch handles:
		//  - rrr rrc rcr crr for column-preferential kernels
		//  - rcc crc ccr ccc for row-preferential kernels

		const dim_t mt = bli_obj_width( c );
		const dim_t nt = bli_obj_length( c );
		const dim_t NR = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx ); \
		const dim_t MR = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx ); \
		const dim_t mu = mt / MR;
		const dim_t nu = nt / NR;

		if ( mu >= nu )
		{
			// panel-block macrokernel; m -> nc, nr; n -> mc, mr: var2() + trans
			bli_gemmsup_ref_var2m( BLIS_TRANSPOSE,
			                       alpha, a, b, beta, c, stor_id, cntx, rntm );
		}
		else // if ( mu < nu )
		{
			// block-panel macrokernel; m -> mc*,nr; n -> nc*,mr: var1() + trans
			bli_gemmsup_ref_var1n( BLIS_TRANSPOSE,
			                       alpha, a, b, beta, c, stor_id, cntx, rntm );
		}
		// *requires nudging of mc,nc up to be a multiple of nr,mr.
	}
#endif

	// Return success so that the caller knows that we computed the solution.
	return BLIS_SUCCESS;
}

