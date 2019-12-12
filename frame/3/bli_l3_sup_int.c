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

err_t bli_gemmsup_int
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx,
       rntm_t* rntm,
       cntl_t* cntl,
       thrinfo_t* thread
     )
{
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

	return BLIS_SUCCESS;
#endif

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
		//if ( m % 2 == 1 && n % 2 == 1 )
		{
			#ifdef TRACEVAR
			printf( "bli_l3_sup_int(): var2m primary\n" );
			#endif
			// block-panel macrokernel; m -> mc, mr; n -> nc, nr: var2()
			bli_gemmsup_ref_var2m( BLIS_NO_TRANSPOSE,
			                       alpha, a, b, beta, c,
			                       stor_id, cntx, rntm, cntl, thread );
		}
		else // if ( mu < nu )
		{
			#ifdef TRACEVAR
			printf( "bli_l3_sup_int(): var1n primary\n" );
			#endif
			// panel-block macrokernel; m -> nc*,mr; n -> mc*,nr: var1()
			bli_gemmsup_ref_var1n( BLIS_NO_TRANSPOSE,
			                       alpha, a, b, beta, c,
			                       stor_id, cntx, rntm, cntl, thread );
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
		//if ( mt % 2 == 1 && nt % 2 == 1 )
		{
			#ifdef TRACEVAR
			printf( "bli_l3_sup_int(): var2m non-primary\n" );
			#endif
			// panel-block macrokernel; m -> nc, nr; n -> mc, mr: var2() + trans
			bli_gemmsup_ref_var2m( BLIS_TRANSPOSE,
			                       alpha, a, b, beta, c,
			                       stor_id, cntx, rntm, cntl, thread );
		}
		else // if ( mu < nu )
		{
			#ifdef TRACEVAR
			printf( "bli_l3_sup_int(): var1n non-primary\n" );
			#endif
			// block-panel macrokernel; m -> mc*,nr; n -> nc*,mr: var1() + trans
			bli_gemmsup_ref_var1n( BLIS_TRANSPOSE,
			                       alpha, a, b, beta, c,
			                       stor_id, cntx, rntm, cntl, thread );
		}
		// *requires nudging of mc,nc up to be a multiple of nr,mr.
	}

	// Return success so that the caller knows that we computed the solution.
	return BLIS_SUCCESS;
}

