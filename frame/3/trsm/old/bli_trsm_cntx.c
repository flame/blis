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

void bli_trsm_cntx_init( cntx_t* cntx )
{
	// Perform basic setup on the context.
	bli_cntx_obj_create( cntx );

	// Initialize the context with the current architecture's native
	// level-3 gemm micro-kernel, and its output preferences.
	bli_gks_cntx_set_l3_nat_ukr( BLIS_GEMM_UKR, cntx );
	bli_gks_cntx_set_l3_nat_ukr_prefs( BLIS_GEMM_UKR, cntx );

	// Initialize the context with the current architecture's native
	// level-3 trsm micro-kernels.
	bli_gks_cntx_set_l3_nat_ukr( BLIS_GEMMTRSM_L_UKR, cntx );
	bli_gks_cntx_set_l3_nat_ukr( BLIS_GEMMTRSM_U_UKR, cntx );
	bli_gks_cntx_set_l3_nat_ukr( BLIS_TRSM_L_UKR, cntx );
	bli_gks_cntx_set_l3_nat_ukr( BLIS_TRSM_U_UKR, cntx );

	// Initialize the context with the current architecture's register
	// and cache blocksizes (and multiples), given the execution method.
	bli_gks_cntx_set_blkszs( BLIS_NAT, 6,
	                         BLIS_NC, BLIS_NR,
	                         BLIS_KC, BLIS_KR,
	                         BLIS_MC, BLIS_MR,
	                         BLIS_NR, BLIS_NR,
	                         BLIS_MR, BLIS_MR,
	                         BLIS_KR, BLIS_KR,
	                         cntx );

	// Set the pack_t schemas for native execution.
	bli_cntx_set_pack_schema_ab( BLIS_PACKED_ROW_PANELS,
	                             BLIS_PACKED_COL_PANELS,
	                             cntx );
}

void bli_trsm_cntx_finalize( cntx_t* cntx )
{
	// Free the context and all memory allocated to it.
	bli_cntx_obj_free( cntx );
}

