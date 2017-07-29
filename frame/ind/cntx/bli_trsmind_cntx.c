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
    - Neither the name of The University of Texas at Austin nor the names
      of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

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

// -----------------------------------------------------------------------------

void bli_trsm3m1_cntx_init( num_t dt, cntx_t* cntx )
{
	const ind_t method = BLIS_3M1;

	// Clear the context fields.
	bli_cntx_clear( cntx );

	// Initialize the context with the current architecture's native
	// level-3 gemm micro-kernel, and its output preferences.
	bli_gks_cntx_set_l3_nat_ukr( BLIS_GEMM_UKR, cntx );
	bli_gks_cntx_set_l3_nat_ukr_prefs( BLIS_GEMM_UKR, cntx );

	// Initialize the context with the virtual micro-kernels associated with
	// the current induced method.
	bli_gks_cntx_set_l3_vir_ukr( method, BLIS_GEMM_UKR, cntx );
	bli_gks_cntx_set_l3_vir_ukr( method, BLIS_GEMMTRSM_L_UKR, cntx );
	bli_gks_cntx_set_l3_vir_ukr( method, BLIS_GEMMTRSM_U_UKR, cntx );
	bli_gks_cntx_set_l3_vir_ukr( method, BLIS_TRSM_L_UKR, cntx );
	bli_gks_cntx_set_l3_vir_ukr( method, BLIS_TRSM_U_UKR, cntx );

	// Initialize the context with packm-related kernels.
	bli_packm_cntx_init( dt, cntx );

	// Initialize the context with the current architecture's register
	// and cache blocksizes (and multiples), and the induced method.
	bli_gks_cntx_set_blkszs
	(
	  method, 6,
	  BLIS_NC, BLIS_NR, 1.0, 1.0,
	  BLIS_KC, BLIS_KR, 3.0, 3.0,
	  BLIS_MC, BLIS_MR, 1.0, 1.0,
	  BLIS_NR, BLIS_NR, 1.0, 1.0,
	  BLIS_MR, BLIS_MR, 1.0, 1.0,
	  BLIS_KR, BLIS_KR, 1.0, 1.0,
	  cntx
	);

	// Set the pack_t schemas for the current induced method.
	bli_cntx_set_pack_schema_a_block( BLIS_PACKED_ROW_PANELS_3MI, cntx );
	bli_cntx_set_pack_schema_b_panel( BLIS_PACKED_COL_PANELS_3MI, cntx );
}

void bli_trsm3m1_cntx_finalize( cntx_t* cntx )
{
}

// -----------------------------------------------------------------------------

void bli_trsm4m1_cntx_init( num_t dt, cntx_t* cntx )
{
	const ind_t method = BLIS_4M1A;

	// Clear the context fields.
	bli_cntx_clear( cntx );

	// Initialize the context with the current architecture's native
	// level-3 gemm micro-kernel, and its output preferences.
	bli_gks_cntx_set_l3_nat_ukr( BLIS_GEMM_UKR, cntx );
	bli_gks_cntx_set_l3_nat_ukr_prefs( BLIS_GEMM_UKR, cntx );

	// Initialize the context with the virtual micro-kernels associated with
	// the current induced method.
	bli_gks_cntx_set_l3_vir_ukr( method, BLIS_GEMM_UKR, cntx );
	bli_gks_cntx_set_l3_vir_ukr( method, BLIS_GEMMTRSM_L_UKR, cntx );
	bli_gks_cntx_set_l3_vir_ukr( method, BLIS_GEMMTRSM_U_UKR, cntx );
	bli_gks_cntx_set_l3_vir_ukr( method, BLIS_TRSM_L_UKR, cntx );
	bli_gks_cntx_set_l3_vir_ukr( method, BLIS_TRSM_U_UKR, cntx );

	// Initialize the context with packm-related kernels.
	bli_packm_cntx_init( dt, cntx );

	// Initialize the context with the current architecture's register
	// and cache blocksizes (and multiples), and the induced method.
	bli_gks_cntx_set_blkszs
	(
	  method, 6,
	  BLIS_NC, BLIS_NR, 1.0, 1.0,
	  BLIS_KC, BLIS_KR, 2.0, 2.0,
	  BLIS_MC, BLIS_MR, 1.0, 1.0,
	  BLIS_NR, BLIS_NR, 1.0, 1.0,
	  BLIS_MR, BLIS_MR, 1.0, 1.0,
	  BLIS_KR, BLIS_KR, 1.0, 1.0,
	  cntx
	);

	// Set the pack_t schemas for the current induced method.
	bli_cntx_set_pack_schema_a_block( BLIS_PACKED_ROW_PANELS_4MI, cntx );
	bli_cntx_set_pack_schema_b_panel( BLIS_PACKED_COL_PANELS_4MI, cntx );
}

void bli_trsm4m1_cntx_finalize( cntx_t* cntx )
{
}

// -----------------------------------------------------------------------------

void bli_trsm1m_cntx_init( num_t dt, cntx_t* cntx )
{
	const ind_t method = BLIS_1M;

	// Clear the context fields.
	bli_cntx_clear( cntx );

	// Initialize the context with the current architecture's native
	// level-3 gemm micro-kernel, and its output preferences.
	bli_gks_cntx_set_l3_nat_ukr( BLIS_GEMM_UKR, cntx );
	bli_gks_cntx_set_l3_nat_ukr_prefs( BLIS_GEMM_UKR, cntx );

	// Initialize the context with the virtual micro-kernels associated with
	// the current induced method.
	bli_gks_cntx_set_l3_vir_ukr( method, BLIS_GEMM_UKR, cntx );
	bli_gks_cntx_set_l3_vir_ukr( method, BLIS_GEMMTRSM_L_UKR, cntx );
	bli_gks_cntx_set_l3_vir_ukr( method, BLIS_GEMMTRSM_U_UKR, cntx );
	bli_gks_cntx_set_l3_vir_ukr( method, BLIS_TRSM_L_UKR, cntx );
	bli_gks_cntx_set_l3_vir_ukr( method, BLIS_TRSM_U_UKR, cntx );

	// Initialize the context with packm-related kernels.
	bli_packm_cntx_init( dt, cntx );

	if ( bli_cntx_l3_ukr_prefers_cols_dt( dt, BLIS_GEMM_UKR, cntx ) )
	{
		// Initialize the context with the current architecture's register
		// and cache blocksizes (and multiples), and the induced method.
		bli_gks_cntx_set_blkszs
		(
		  method, 6,
		  BLIS_NC, BLIS_NR, 1.0, 1.0,
		  BLIS_KC, BLIS_KR, 2.0, 2.0, // halve kc...
		  BLIS_MC, BLIS_MR, 2.0, 2.0, // halve mc...
		  BLIS_NR, BLIS_NR, 1.0, 1.0,
		  BLIS_MR, BLIS_MR, 2.0, 1.0, // ...and mr (but NOT packmr)
		  BLIS_KR, BLIS_KR, 1.0, 1.0,
		  cntx
		);

		// Set the pack_t schemas for the current induced method.
		//bli_cntx_set_pack_schema_ab_blockpanel( BLIS_PACKED_ROW_PANELS_1E,
		//                                        BLIS_PACKED_COL_PANELS_1R,
		//                                        cntx );
		bli_cntx_set_pack_schema_a_block( BLIS_PACKED_ROW_PANELS_1E, cntx );
		bli_cntx_set_pack_schema_b_panel( BLIS_PACKED_COL_PANELS_1R, cntx );
	}
	else // if ( bli_cntx_l3_ukr_prefers_rows_dt( dt, BLIS_GEMM_UKR, cntx ) )
	{
		// Initialize the context with the current architecture's register
		// and cache blocksizes (and multiples), and the induced method.
		bli_gks_cntx_set_blkszs
		(
		  method, 6,
		  BLIS_NC, BLIS_NR, 2.0, 2.0, // halve nc...
		  BLIS_KC, BLIS_KR, 2.0, 2.0, // halve kc...
		  BLIS_MC, BLIS_MR, 1.0, 1.0,
		  BLIS_NR, BLIS_NR, 2.0, 1.0, // ...and nr (but NOT packnr)
		  BLIS_MR, BLIS_MR, 1.0, 1.0,
		  BLIS_KR, BLIS_KR, 1.0, 1.0,
		  cntx
		);

		// Set the pack_t schemas for the current induced method.
		//bli_cntx_set_pack_schema_ab_blockpanel( BLIS_PACKED_ROW_PANELS_1R,
		//                                        BLIS_PACKED_COL_PANELS_1E,
		//                                        cntx );
		bli_cntx_set_pack_schema_a_block( BLIS_PACKED_ROW_PANELS_1R, cntx );
		bli_cntx_set_pack_schema_b_panel( BLIS_PACKED_COL_PANELS_1E, cntx );
	}
}

void bli_trsm1m_cntx_stage( dim_t stage, cntx_t* cntx )
{
}

void bli_trsm1m_cntx_finalize( cntx_t* cntx )
{
}

// -----------------------------------------------------------------------------

void bli_trsmnat_cntx_init( num_t dt, cntx_t* cntx )
{
	bli_trsm_cntx_init( dt, cntx );
}

void bli_trsmnat_cntx_finalize( cntx_t* cntx )
{
	bli_trsm_cntx_finalize( cntx );
}

