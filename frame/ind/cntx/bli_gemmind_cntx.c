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

typedef void (*cntx_ft)( cntx_t* cntx );

static void* bli_gemmind_cntx_fp[BLIS_NUM_IND_METHODS][2] = 
{
        /*              _cntx_init             _cntx_finalize   */
/* 3mh  */ { bli_gemm3mh_cntx_init, bli_gemm3mh_cntx_finalize },
/* 3m3  */ { bli_gemm3m3_cntx_init, bli_gemm3m3_cntx_finalize },
/* 3m2  */ { bli_gemm3m2_cntx_init, bli_gemm3m2_cntx_finalize },
/* 3m1  */ { bli_gemm3m1_cntx_init, bli_gemm3m1_cntx_finalize },
/* 4mh  */ { bli_gemm4mh_cntx_init, bli_gemm4mh_cntx_finalize },
/* 4mb  */ { bli_gemm4mb_cntx_init, bli_gemm4mb_cntx_finalize },
/* 4m1  */ { bli_gemm4m1_cntx_init, bli_gemm4m1_cntx_finalize },
/* nat  */ { bli_gemmnat_cntx_init, bli_gemmnat_cntx_finalize }
};

#define BLIS_CNTX_INIT_INDEX     0
#define BLIS_CNTX_FINALIZE_INDEX 1

// -----------------------------------------------------------------------------

// Use a datatype to find the highest priority available (ie: implemented
// and enabled) induced method, and then execute the context initialization/
// finalization function associated with that induced method.

void bli_gemmind_cntx_init_avail( num_t dt, cntx_t* cntx )
{
	ind_t method = bli_ind_oper_find_avail( BLIS_GEMM, dt );

	bli_gemmind_cntx_init( method, cntx );
}

void bli_gemmind_cntx_finalize_avail( num_t dt, cntx_t* cntx )
{
	ind_t method = bli_ind_oper_find_avail( BLIS_GEMM, dt );

	bli_gemmind_cntx_finalize( method, cntx );
}

// -----------------------------------------------------------------------------

// Execute the context initialization/finalization function associated
// with a given induced method.

void bli_gemmind_cntx_init( ind_t method, cntx_t* cntx )
{
	cntx_ft func = bli_gemmind_cntx_init_get_func( method );

	func( cntx );
}

void bli_gemmind_cntx_finalize( ind_t method, cntx_t* cntx )
{
	cntx_ft func = bli_gemmind_cntx_finalize_get_func( method );

	func( cntx );
}

// -----------------------------------------------------------------------------

void* bli_gemmind_cntx_init_get_func( ind_t method )
{
	return bli_gemmind_cntx_fp[ method ][ BLIS_CNTX_INIT_INDEX ];
}

void* bli_gemmind_cntx_finalize_get_func( ind_t method )
{
	return bli_gemmind_cntx_fp[ method ][ BLIS_CNTX_FINALIZE_INDEX ];
}

// -----------------------------------------------------------------------------

void bli_gemm3m1_cntx_init( cntx_t* cntx )
{
	const ind_t method = BLIS_3M1;

	// Clear the context fields.
	bli_cntx_obj_clear( cntx );

	// Initialize the context with the current architecture's native
	// level-3 gemm micro-kernel, and its output preferences.
	bli_gks_cntx_set_l3_nat_ukr( BLIS_GEMM_UKR, cntx );
	bli_gks_cntx_set_l3_nat_ukr_prefs( BLIS_GEMM_UKR, cntx );

	// Initialize the context with the virtual micro-kernel associated with
	// the current induced method.
	bli_gks_cntx_set_l3_vir_ukr( method, BLIS_GEMM_UKR, cntx );

	// Initialize the context with packm-related kernels.
	bli_packm_cntx_init( cntx );

	// Initialize the context with the current architecture's register
	// and cache blocksizes (and multiples), and the induced method.
	bli_gks_cntx_set_blkszs( method, 6,
	                         BLIS_NC, BLIS_NR, 1.0,
	                         BLIS_KC, BLIS_KR, 3.0,
	                         BLIS_MC, BLIS_MR, 1.0,
	                         BLIS_NR, BLIS_NR, 1.0,
	                         BLIS_MR, BLIS_MR, 1.0,
	                         BLIS_KR, BLIS_KR, 1.0,
	                         cntx );

	// Set the pack_t schemas for the current induced method.
	bli_cntx_set_pack_schema_ab( BLIS_PACKED_ROW_PANELS_3MI,
	                             BLIS_PACKED_COL_PANELS_3MI,
	                             cntx );
}

void bli_gemm3m1_cntx_stage( dim_t stage, cntx_t* cntx )
{
}

void bli_gemm3m1_cntx_finalize( cntx_t* cntx )
{
}

// -----------------------------------------------------------------------------

void bli_gemm3m2_cntx_init( cntx_t* cntx )
{
	const ind_t method = BLIS_3M2;

	// Clear the context fields.
	bli_cntx_obj_clear( cntx );

	// Initialize the context with the current architecture's native
	// level-3 gemm micro-kernel, and its output preferences.
	bli_gks_cntx_set_l3_nat_ukr( BLIS_GEMM_UKR, cntx );
	bli_gks_cntx_set_l3_nat_ukr_prefs( BLIS_GEMM_UKR, cntx );

	// Initialize the context with the virtual micro-kernel associated with
	// the current induced method.
	bli_gks_cntx_set_l3_vir_ukr( method, BLIS_GEMM_UKR, cntx );

	// Initialize the context with packm-related kernels.
	bli_packm_cntx_init( cntx );

	// Initialize the context with the current architecture's register
	// and cache blocksizes (and multiples), and the induced method.
	bli_gks_cntx_set_blkszs( method, 6,
	                         BLIS_NC, BLIS_NR, 3.0,
	                         BLIS_KC, BLIS_KR, 1.0,
	                         BLIS_MC, BLIS_MR, 3.0,
	                         BLIS_NR, BLIS_NR, 1.0,
	                         BLIS_MR, BLIS_MR, 1.0,
	                         BLIS_KR, BLIS_KR, 1.0,
	                         cntx );

	// Set the pack_t schemas for the current induced method.
	bli_cntx_set_pack_schema_ab( BLIS_PACKED_ROW_PANELS_3MS,
	                             BLIS_PACKED_COL_PANELS_3MI,
	                             cntx );
}

void bli_gemm3m2_cntx_stage( dim_t stage, cntx_t* cntx )
{
}

void bli_gemm3m2_cntx_finalize( cntx_t* cntx )
{
}

// -----------------------------------------------------------------------------

void bli_gemm3m3_cntx_init( cntx_t* cntx )
{
	const ind_t method = BLIS_3M3;

	// Clear the context fields.
	bli_cntx_obj_clear( cntx );

	// Initialize the context with the current architecture's native
	// level-3 gemm micro-kernel, and its output preferences.
	bli_gks_cntx_set_l3_nat_ukr( BLIS_GEMM_UKR, cntx );
	bli_gks_cntx_set_l3_nat_ukr_prefs( BLIS_GEMM_UKR, cntx );

	// Initialize the context with the virtual micro-kernel associated with
	// the current induced method.
	bli_gks_cntx_set_l3_vir_ukr( method, BLIS_GEMM_UKR, cntx );

	// Initialize the context with packm-related kernels.
	bli_packm_cntx_init( cntx );

	// Initialize the context with the current architecture's register
	// and cache blocksizes (and multiples), and the induced method.
	bli_gks_cntx_set_blkszs( method, 6,
	                         BLIS_NC, BLIS_NR, 3.0,
	                         BLIS_KC, BLIS_KR, 1.0,
	                         BLIS_MC, BLIS_MR, 1.0,
	                         BLIS_NR, BLIS_NR, 1.0,
	                         BLIS_MR, BLIS_MR, 1.0,
	                         BLIS_KR, BLIS_KR, 1.0,
	                         cntx );

	// Set the pack_t schemas for the current induced method.
	bli_cntx_set_pack_schema_ab( 0, // not yet needed; varies with _stage()
	                             BLIS_PACKED_COL_PANELS_3MS,
	                             cntx );
}

void bli_gemm3m3_cntx_stage( dim_t stage, cntx_t* cntx )
{
	// Set the pack_t schemas as a function of the stage of execution.
	if ( stage == 0 )
	{
		bli_cntx_set_pack_schema_a( BLIS_PACKED_ROW_PANELS_RO, cntx );
	}
	else if ( stage == 1 )
	{
		bli_cntx_set_pack_schema_a( BLIS_PACKED_ROW_PANELS_IO, cntx );
	}
	else // if ( stage == 2 )
	{
		bli_cntx_set_pack_schema_a( BLIS_PACKED_ROW_PANELS_RPI, cntx );
	}
}

void bli_gemm3m3_cntx_finalize( cntx_t* cntx )
{
}

// -----------------------------------------------------------------------------

void bli_gemm3mh_cntx_init( cntx_t* cntx )
{
	const ind_t method = BLIS_3MH;

	// Clear the context fields.
	bli_cntx_obj_clear( cntx );

	// Initialize the context with the current architecture's native
	// level-3 gemm micro-kernel, and its output preferences.
	bli_gks_cntx_set_l3_nat_ukr( BLIS_GEMM_UKR, cntx );
	bli_gks_cntx_set_l3_nat_ukr_prefs( BLIS_GEMM_UKR, cntx );

	// Initialize the context with the virtual micro-kernel associated with
	// the current induced method.
	bli_gks_cntx_set_l3_vir_ukr( method, BLIS_GEMM_UKR, cntx );

	// Initialize the context with packm-related kernels.
	bli_packm_cntx_init( cntx );

	// Initialize the context with the current architecture's register
	// and cache blocksizes (and multiples), and the induced method.
	bli_gks_cntx_set_blkszs( method, 6,
	                         BLIS_NC, BLIS_NR, 1.0,
	                         BLIS_KC, BLIS_KR, 1.0,
	                         BLIS_MC, BLIS_MR, 1.0,
	                         BLIS_NR, BLIS_NR, 1.0,
	                         BLIS_MR, BLIS_MR, 1.0,
	                         BLIS_KR, BLIS_KR, 1.0,
	                         cntx );

	// Set the pack_t schemas for the current induced method.
	bli_cntx_set_pack_schema_ab( 0, // not yet needed; varies with _stage()
	                             0, // not yet needed; varies with _stage()
	                             cntx );
}

void bli_gemm3mh_cntx_stage( dim_t stage, cntx_t* cntx )
{
	// Set the pack_t schemas as a function of the stage of execution.
	if ( stage == 0 )
	{
		bli_cntx_set_pack_schema_ab( BLIS_PACKED_ROW_PANELS_RO,
		                             BLIS_PACKED_COL_PANELS_RO, cntx );
	}
	else if ( stage == 1 )
	{
		bli_cntx_set_pack_schema_ab( BLIS_PACKED_ROW_PANELS_IO,
		                             BLIS_PACKED_COL_PANELS_IO, cntx );
	}
	else // if ( stage == 2 )
	{
		bli_cntx_set_pack_schema_ab( BLIS_PACKED_ROW_PANELS_RPI,
		                             BLIS_PACKED_COL_PANELS_RPI, cntx );
	}
}

void bli_gemm3mh_cntx_finalize( cntx_t* cntx )
{
}

// -----------------------------------------------------------------------------

void bli_gemm4m1_cntx_init( cntx_t* cntx )
{
	const ind_t method = BLIS_4M1A;

	// Clear the context fields.
	bli_cntx_obj_clear( cntx );

	// Initialize the context with the current architecture's native
	// level-3 gemm micro-kernel, and its output preferences.
	bli_gks_cntx_set_l3_nat_ukr( BLIS_GEMM_UKR, cntx );
	bli_gks_cntx_set_l3_nat_ukr_prefs( BLIS_GEMM_UKR, cntx );

	// Initialize the context with the virtual micro-kernel associated with
	// the current induced method.
	bli_gks_cntx_set_l3_vir_ukr( method, BLIS_GEMM_UKR, cntx );

	// Initialize the context with packm-related kernels.
	bli_packm_cntx_init( cntx );

	// Initialize the context with the current architecture's register
	// and cache blocksizes (and multiples), and the induced method.
	bli_gks_cntx_set_blkszs( method, 6,
	                         BLIS_NC, BLIS_NR, 1.0,
	                         BLIS_KC, BLIS_KR, 2.0,
	                         BLIS_MC, BLIS_MR, 1.0,
	                         BLIS_NR, BLIS_NR, 1.0,
	                         BLIS_MR, BLIS_MR, 1.0,
	                         BLIS_KR, BLIS_KR, 1.0,
	                         cntx );

	// Set the pack_t schemas for the current induced method.
	bli_cntx_set_pack_schema_ab( BLIS_PACKED_ROW_PANELS_4MI,
	                             BLIS_PACKED_COL_PANELS_4MI,
	                             cntx );
}

void bli_gemm4m1_cntx_stage( dim_t stage, cntx_t* cntx )
{
}

void bli_gemm4m1_cntx_finalize( cntx_t* cntx )
{
}

// -----------------------------------------------------------------------------

void bli_gemm4mb_cntx_init( cntx_t* cntx )
{
	const ind_t method = BLIS_4M1B;

	// Clear the context fields.
	bli_cntx_obj_clear( cntx );

	// Initialize the context with the current architecture's native
	// level-3 gemm micro-kernel, and its output preferences.
	bli_gks_cntx_set_l3_nat_ukr( BLIS_GEMM_UKR, cntx );
	bli_gks_cntx_set_l3_nat_ukr_prefs( BLIS_GEMM_UKR, cntx );

	// Initialize the context with the virtual micro-kernel associated with
	// the current induced method.
	bli_gks_cntx_set_l3_vir_ukr( method, BLIS_GEMM_UKR, cntx );

	// Initialize the context with packm-related kernels.
	bli_packm_cntx_init( cntx );

	// Initialize the context with the current architecture's register
	// and cache blocksizes (and multiples), and the induced method.
	bli_gks_cntx_set_blkszs( method, 6,
	                         BLIS_NC, BLIS_NR, 2.0,
	                         BLIS_KC, BLIS_KR, 1.0,
	                         BLIS_MC, BLIS_MR, 2.0,
	                         BLIS_NR, BLIS_NR, 1.0,
	                         BLIS_MR, BLIS_MR, 1.0,
	                         BLIS_KR, BLIS_KR, 1.0,
	                         cntx );

	// Set the pack_t schemas for the current induced method.
	bli_cntx_set_pack_schema_ab( BLIS_PACKED_ROW_PANELS_4MI,
	                             BLIS_PACKED_COL_PANELS_4MI,
	                             cntx );
}

void bli_gemm4mb_cntx_stage( dim_t stage, cntx_t* cntx )
{
}

void bli_gemm4mb_cntx_finalize( cntx_t* cntx )
{
}

// -----------------------------------------------------------------------------

void bli_gemm4mh_cntx_init( cntx_t* cntx )
{
	const ind_t method = BLIS_4MH;

	// Clear the context fields.
	bli_cntx_obj_clear( cntx );

	// Initialize the context with the current architecture's native
	// level-3 gemm micro-kernel, and its output preferences.
	bli_gks_cntx_set_l3_nat_ukr( BLIS_GEMM_UKR, cntx );
	bli_gks_cntx_set_l3_nat_ukr_prefs( BLIS_GEMM_UKR, cntx );

	// Initialize the context with the virtual micro-kernel associated with
	// the current induced method.
	bli_gks_cntx_set_l3_vir_ukr( method, BLIS_GEMM_UKR, cntx );

	// Initialize the context with packm-related kernels.
	bli_packm_cntx_init( cntx );

	// Initialize the context with the current architecture's register
	// and cache blocksizes (and multiples), and the induced method.
	bli_gks_cntx_set_blkszs( method, 6,
	                         BLIS_NC, BLIS_NR, 1.0,
	                         BLIS_KC, BLIS_KR, 1.0,
	                         BLIS_MC, BLIS_MR, 1.0,
	                         BLIS_NR, BLIS_NR, 1.0,
	                         BLIS_MR, BLIS_MR, 1.0,
	                         BLIS_KR, BLIS_KR, 1.0,
	                         cntx );

	// Set the pack_t schemas for the current induced method.
	bli_cntx_set_pack_schema_ab( 0, // not yet needed; varies with _stage()
	                             0, // not yet needed; varies with _stage()
	                             cntx );
}

void bli_gemm4mh_cntx_stage( dim_t stage, cntx_t* cntx )
{
	// Set the pack_t schemas as a function of the stage of execution.
	if ( stage == 0 )
	{
		bli_cntx_set_pack_schema_ab( BLIS_PACKED_ROW_PANELS_RO,
		                             BLIS_PACKED_COL_PANELS_RO, cntx );
	}
	else if ( stage == 1 )
	{
		bli_cntx_set_pack_schema_ab( BLIS_PACKED_ROW_PANELS_IO,
		                             BLIS_PACKED_COL_PANELS_IO, cntx );
	}
	else if ( stage == 2 )
	{
		bli_cntx_set_pack_schema_ab( BLIS_PACKED_ROW_PANELS_RO,
		                             BLIS_PACKED_COL_PANELS_IO, cntx );
	}
	else // if ( stage == 3 )
	{
		bli_cntx_set_pack_schema_ab( BLIS_PACKED_ROW_PANELS_IO,
		                             BLIS_PACKED_COL_PANELS_RO, cntx );
	}
}

void bli_gemm4mh_cntx_finalize( cntx_t* cntx )
{
}

// -----------------------------------------------------------------------------

void bli_gemmnat_cntx_init( cntx_t* cntx )
{
	bli_gemm_cntx_init( cntx );
}

void bli_gemmnat_cntx_stage( dim_t stage, cntx_t* cntx )
{
}

void bli_gemmnat_cntx_finalize( cntx_t* cntx )
{
	bli_gemm_cntx_finalize( cntx );
}

