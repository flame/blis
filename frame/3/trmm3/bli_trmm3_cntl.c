/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2013, The University of Texas

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
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

extern scalm_t*   scalm_cntl;
extern gemm_t*    gemm_cntl_bp_ke;

trmm_t*           trmm3_cntl;

trmm_t*           trmm3_cntl_bp_ke;
trmm_t*           trmm3_cntl_op_bp;
trmm_t*           trmm3_cntl_mm_op;
trmm_t*           trmm3_cntl_vl_mm;

packm_t*          trmm3_packa_cntl;
packm_t*          trmm3_packb_cntl;
packm_t*          trmm3_packc_cntl;
unpackm_t*        trmm3_unpackc_cntl;

blksz_t*          trmm3_mc;
blksz_t*          trmm3_nc;
blksz_t*          trmm3_kc;
blksz_t*          trmm3_mr;
blksz_t*          trmm3_nr;
blksz_t*          trmm3_kr;
blksz_t*          trmm3_extmr;
blksz_t*          trmm3_extnr;
blksz_t*          trmm3_extkr;
blksz_t*          trmm3_ni;


void bli_trmm3_cntl_init()
{
	// Create blocksize objects for each dimension.
	trmm3_mc = bli_blksz_obj_create( BLIS_DEFAULT_MC_S,
	                                 BLIS_DEFAULT_MC_D,
	                                 BLIS_DEFAULT_MC_C,
	                                 BLIS_DEFAULT_MC_Z );

	trmm3_nc = bli_blksz_obj_create( BLIS_DEFAULT_NC_S,
	                                 BLIS_DEFAULT_NC_D,
	                                 BLIS_DEFAULT_NC_C,
	                                 BLIS_DEFAULT_NC_Z );

	trmm3_kc = bli_blksz_obj_create( BLIS_DEFAULT_KC_S,
	                                 BLIS_DEFAULT_KC_D,
	                                 BLIS_DEFAULT_KC_C,
	                                 BLIS_DEFAULT_KC_Z );

	trmm3_mr = bli_blksz_obj_create( BLIS_DEFAULT_MR_S,
	                                 BLIS_DEFAULT_MR_D,
	                                 BLIS_DEFAULT_MR_C,
	                                 BLIS_DEFAULT_MR_Z );

	trmm3_nr = bli_blksz_obj_create( BLIS_DEFAULT_NR_S,
	                                 BLIS_DEFAULT_NR_D,
	                                 BLIS_DEFAULT_NR_C,
	                                 BLIS_DEFAULT_NR_Z );

	trmm3_kr = bli_blksz_obj_create( BLIS_DEFAULT_KR_S,
	                                 BLIS_DEFAULT_KR_D,
	                                 BLIS_DEFAULT_KR_C,
	                                 BLIS_DEFAULT_KR_Z );

	trmm3_extmr = bli_blksz_obj_create( BLIS_EXTEND_MR_S,
	                                    BLIS_EXTEND_MR_D,
	                                    BLIS_EXTEND_MR_C,
	                                    BLIS_EXTEND_MR_Z );

	trmm3_extnr = bli_blksz_obj_create( BLIS_EXTEND_NR_S,
	                                    BLIS_EXTEND_NR_D,
	                                    BLIS_EXTEND_NR_C,
	                                    BLIS_EXTEND_NR_Z );

	trmm3_extkr = bli_blksz_obj_create( BLIS_EXTEND_KR_S,
	                                    BLIS_EXTEND_KR_D,
	                                    BLIS_EXTEND_KR_C,
	                                    BLIS_EXTEND_KR_Z );

	trmm3_ni = bli_blksz_obj_create( BLIS_DEFAULT_NI_S,
	                                 BLIS_DEFAULT_NI_D,
	                                 BLIS_DEFAULT_NI_C,
	                                 BLIS_DEFAULT_NI_Z );


	// Create control tree objects for packm operations on a, b, and c.
	trmm3_packa_cntl
	=
	bli_packm_cntl_obj_create( BLIS_BLOCKED,
	                           BLIS_VARIANT3, // pack panels of A compactly
	                           // IMPORTANT: for consistency with trsm, "k" dim
	                           // multiple is set to mr.
	                           trmm3_mr, trmm3_extmr,
	                           trmm3_mr, trmm3_extmr,
	                           FALSE, // do NOT scale by alpha
	                           TRUE,  // densify
	                           FALSE, // do NOT invert diagonal
	                           FALSE, // reverse iteration if upper?
	                           FALSE, // reverse iteration if lower?
	                           BLIS_PACKED_ROW_PANELS,
	                           BLIS_BUFFER_FOR_A_BLOCK );

	trmm3_packb_cntl
	=
	bli_packm_cntl_obj_create( BLIS_BLOCKED,
	                           BLIS_VARIANT2,
	                           // IMPORTANT: m dim multiple here must be mr
	                           // since "k" dim multiple is set to mr above.
	                           trmm3_mr, trmm3_extmr,
	                           trmm3_nr, trmm3_extnr,
	                           FALSE, // do NOT scale by alpha
	                           FALSE, // already dense; densify not necessary
	                           FALSE, // do NOT invert diagonal
	                           FALSE, // reverse iteration if upper?
	                           FALSE, // reverse iteration if lower?
	                           BLIS_PACKED_COL_PANELS,
	                           BLIS_BUFFER_FOR_B_PANEL );

	trmm3_packc_cntl
	=
	bli_packm_cntl_obj_create( BLIS_UNBLOCKED,
	                           BLIS_VARIANT1,
	                           trmm3_mr, trmm3_extmr,
	                           trmm3_nr, trmm3_extnr,
	                           FALSE, // do NOT scale by beta
	                           FALSE, // already dense; densify not necessary
	                           FALSE, // do NOT invert diagonal
	                           FALSE, // reverse iteration if upper?
	                           FALSE, // reverse iteration if lower?
	                           BLIS_PACKED_COLUMNS,
	                           BLIS_BUFFER_FOR_GEN_USE );

	trmm3_unpackc_cntl
	=
	bli_unpackm_cntl_obj_create( BLIS_UNBLOCKED,
	                             BLIS_VARIANT1,
	                             NULL ); // no blocksize needed


	// Create control tree object for lowest-level block-panel kernel.
	trmm3_cntl_bp_ke
	=
	bli_trmm_cntl_obj_create( BLIS_UNB_OPT,
	                          BLIS_VARIANT2,
	                          NULL, NULL, NULL, NULL, NULL,
	                          NULL, NULL, NULL, NULL );

	// Create control tree object for outer panel (to block-panel)
	// problem, packing a and b.
	trmm3_cntl_op_bp
	=
	bli_trmm_cntl_obj_create( BLIS_BLOCKED,
	                          //BLIS_VARIANT4,  // var1 with incremental pack in iter 0
	                          BLIS_VARIANT1,
	                          trmm3_mc,
	                          trmm3_ni,
	                          NULL,
	                          trmm3_packa_cntl,
	                          trmm3_packb_cntl,
	                          NULL,
	                          trmm3_cntl_bp_ke,
	                          gemm_cntl_bp_ke,
	                          NULL );

	// Create control tree object for general problem via multiple
	// rank-k (outer panel) updates, packing a and b.
	trmm3_cntl_mm_op
	=
	bli_trmm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT3,
	                          trmm3_kc,
	                          NULL,
	                          NULL,
	                          NULL, 
	                          NULL,
	                          NULL,
	                          trmm3_cntl_op_bp,
	                          NULL,
	                          NULL );

	// Create control tree object for very large problem via multiple
	// general problems, packing a and b.
	trmm3_cntl_vl_mm
	=
	bli_trmm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT2,
	                          trmm3_nc,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          trmm3_cntl_mm_op,
	                          NULL,
	                          NULL );

	// Alias the "master" trmm3 control tree to a shorter name.
	//trmm3_cntl = trmm3_cntl_mm_op;
	trmm3_cntl = trmm3_cntl_vl_mm;
}

void bli_trmm3_cntl_finalize()
{
	bli_blksz_obj_free( trmm3_mc );
	bli_blksz_obj_free( trmm3_nc );
	bli_blksz_obj_free( trmm3_kc );
	bli_blksz_obj_free( trmm3_mr );
	bli_blksz_obj_free( trmm3_nr );
	bli_blksz_obj_free( trmm3_kr );
	bli_blksz_obj_free( trmm3_ni );

	bli_cntl_obj_free( trmm3_packa_cntl );
	bli_cntl_obj_free( trmm3_packb_cntl );
	bli_cntl_obj_free( trmm3_packc_cntl );
	bli_cntl_obj_free( trmm3_unpackc_cntl );

	bli_cntl_obj_free( trmm3_cntl_bp_ke );
	bli_cntl_obj_free( trmm3_cntl_op_bp );
	bli_cntl_obj_free( trmm3_cntl_mm_op );
	bli_cntl_obj_free( trmm3_cntl_vl_mm );
}

