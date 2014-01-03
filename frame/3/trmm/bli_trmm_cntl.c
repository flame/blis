/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas

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

trmm_t*           trmm_l_cntl;
trmm_t*           trmm_r_cntl;

trmm_t*           trmm_cntl_bp_ke;

trmm_t*           trmm_l_cntl_op_bp;
trmm_t*           trmm_l_cntl_mm_op;
trmm_t*           trmm_l_cntl_vl_mm;

trmm_t*           trmm_r_cntl_op_bp;
trmm_t*           trmm_r_cntl_mm_op;
trmm_t*           trmm_r_cntl_vl_mm;

packm_t*          trmm_l_packa_cntl;
packm_t*          trmm_l_packb_cntl;

packm_t*          trmm_r_packa_cntl;
packm_t*          trmm_r_packb_cntl;

packm_t*          trmm_packc_cntl;
unpackm_t*        trmm_unpackc_cntl;

blksz_t*          trmm_mc;
blksz_t*          trmm_nc;
blksz_t*          trmm_kc;
blksz_t*          trmm_mr;
blksz_t*          trmm_nr;
blksz_t*          trmm_kr;
blksz_t*          trmm_ni;


void bli_trmm_cntl_init()
{
	// Create blocksize objects for each dimension.
	trmm_mc = bli_blksz_obj_create( BLIS_DEFAULT_MC_S, BLIS_EXTEND_MC_S,
	                                BLIS_DEFAULT_MC_D, BLIS_EXTEND_MC_D,
	                                BLIS_DEFAULT_MC_C, BLIS_EXTEND_MC_C,
	                                BLIS_DEFAULT_MC_Z, BLIS_EXTEND_MC_Z );

	trmm_nc = bli_blksz_obj_create( BLIS_DEFAULT_NC_S, BLIS_EXTEND_NC_S,
	                                BLIS_DEFAULT_NC_D, BLIS_EXTEND_NC_D,
	                                BLIS_DEFAULT_NC_C, BLIS_EXTEND_NC_C,
	                                BLIS_DEFAULT_NC_Z, BLIS_EXTEND_NC_Z );

	trmm_kc = bli_blksz_obj_create( BLIS_DEFAULT_KC_S, BLIS_EXTEND_KC_S,
	                                BLIS_DEFAULT_KC_D, BLIS_EXTEND_KC_D,
	                                BLIS_DEFAULT_KC_C, BLIS_EXTEND_KC_C,
	                                BLIS_DEFAULT_KC_Z, BLIS_EXTEND_KC_Z );

	trmm_mr = bli_blksz_obj_create( BLIS_DEFAULT_MR_S, BLIS_EXTEND_MR_S,
	                                BLIS_DEFAULT_MR_D, BLIS_EXTEND_MR_D,
	                                BLIS_DEFAULT_MR_C, BLIS_EXTEND_MR_C,
	                                BLIS_DEFAULT_MR_Z, BLIS_EXTEND_MR_Z );

	trmm_nr = bli_blksz_obj_create( BLIS_DEFAULT_NR_S, BLIS_EXTEND_NR_S,
	                                BLIS_DEFAULT_NR_D, BLIS_EXTEND_NR_D,
	                                BLIS_DEFAULT_NR_C, BLIS_EXTEND_NR_C,
	                                BLIS_DEFAULT_NR_Z, BLIS_EXTEND_NR_Z );

	trmm_kr = bli_blksz_obj_create( BLIS_DEFAULT_KR_S, BLIS_EXTEND_KR_S,
	                                BLIS_DEFAULT_KR_D, BLIS_EXTEND_KR_D,
	                                BLIS_DEFAULT_KR_C, BLIS_EXTEND_KR_C,
	                                BLIS_DEFAULT_KR_Z, BLIS_EXTEND_KR_Z );

	trmm_ni = bli_blksz_obj_create( BLIS_DEFAULT_NI_S, 0,
	                                BLIS_DEFAULT_NI_D, 0,
	                                BLIS_DEFAULT_NI_C, 0,
	                                BLIS_DEFAULT_NI_Z, 0 );


	// Create control tree objects for packm operations (left side).
	trmm_l_packa_cntl
	=
	bli_packm_cntl_obj_create( BLIS_BLOCKED,
	                           BLIS_VARIANT3, // pack panels of A compactly
	                           // IMPORTANT: for consistency with trsm, "k" dim
	                           // multiple is set to mr.
	                           trmm_mr,
	                           trmm_mr,
	                           TRUE,  // densify
	                           FALSE, // do NOT invert diagonal
	                           FALSE, // reverse iteration if upper?
	                           FALSE, // reverse iteration if lower?
	                           BLIS_PACKED_ROW_PANELS,
	                           BLIS_BUFFER_FOR_A_BLOCK );

	trmm_l_packb_cntl
	=
	bli_packm_cntl_obj_create( BLIS_BLOCKED,
	                           BLIS_VARIANT2,
	                           // IMPORTANT: m dim multiple here must be mr
	                           // since "k" dim multiple is set to mr above.
	                           trmm_mr,
	                           trmm_nr,
	                           FALSE, // already dense
	                           FALSE, // do NOT invert diagonal
	                           FALSE, // reverse iteration if upper?
	                           FALSE, // reverse iteration if lower?
	                           BLIS_PACKED_COL_PANELS,
	                           BLIS_BUFFER_FOR_B_PANEL );

	// Create control tree objects for packm operations (right side).
	trmm_r_packa_cntl
	=
	bli_packm_cntl_obj_create( BLIS_BLOCKED,
	                           BLIS_VARIANT2,
	                           // IMPORTANT: for consistency with trsm, "k" dim
	                           // multiple is set to nr.
	                           trmm_mr,
	                           trmm_nr,
	                           FALSE, // already dense
	                           FALSE, // do NOT invert diagonal
	                           FALSE, // reverse iteration if upper?
	                           FALSE, // reverse iteration if lower?
	                           BLIS_PACKED_ROW_PANELS,
	                           BLIS_BUFFER_FOR_A_BLOCK );

	trmm_r_packb_cntl
	=
	bli_packm_cntl_obj_create( BLIS_BLOCKED,
	                           BLIS_VARIANT3, // pack panels of B compactly
	                           // IMPORTANT: m dim multiple here must be nr
	                           // since "k" dim multiple is set to nr above.
	                           trmm_nr,
	                           trmm_nr,
	                           TRUE,  // densify
	                           FALSE, // do NOT invert diagonal
	                           FALSE, // reverse iteration if upper?
	                           FALSE, // reverse iteration if lower?
	                           BLIS_PACKED_COL_PANELS,
	                           BLIS_BUFFER_FOR_B_PANEL );

	// Create control tree objects for packm/unpackm operations on C.
	trmm_packc_cntl
	=
	bli_packm_cntl_obj_create( BLIS_UNBLOCKED,
	                           BLIS_VARIANT1,
	                           trmm_mr,
	                           trmm_nr,
	                           FALSE, // already dense; densify not necessary
	                           FALSE, // do NOT invert diagonal
	                           FALSE, // reverse iteration if upper?
	                           FALSE, // reverse iteration if lower?
	                           BLIS_PACKED_COLUMNS,
	                           BLIS_BUFFER_FOR_GEN_USE );

	trmm_unpackc_cntl
	=
	bli_unpackm_cntl_obj_create( BLIS_UNBLOCKED,
	                             BLIS_VARIANT1,
	                             NULL ); // no blocksize needed


	// Create control tree object for lowest-level block-panel kernel.
	trmm_cntl_bp_ke
	=
	bli_trmm_cntl_obj_create( BLIS_UNB_OPT,
	                          BLIS_VARIANT2,
	                          NULL, NULL, NULL, NULL, NULL,
	                          NULL, NULL, NULL, NULL );

	// Create control tree object for outer panel (to block-panel)
	// problem (left side).
	trmm_l_cntl_op_bp
	=
	bli_trmm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT1,
	                          trmm_mc,
	                          trmm_ni,
	                          NULL,
	                          trmm_l_packa_cntl,
	                          trmm_l_packb_cntl,
	                          NULL,
	                          trmm_cntl_bp_ke,
	                          gemm_cntl_bp_ke,
	                          NULL );

	// Create control tree object for general problem via multiple
	// rank-k (outer panel) updates (left side).
	trmm_l_cntl_mm_op
	=
	bli_trmm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT3,
	                          trmm_kc,
	                          NULL,
	                          NULL,
	                          NULL, 
	                          NULL,
	                          NULL,
	                          trmm_l_cntl_op_bp,
	                          NULL,
	                          NULL );

	// Create control tree object for very large problem via multiple
	// general problems (left side).
	trmm_l_cntl_vl_mm
	=
	bli_trmm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT2,
	                          trmm_nc,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          trmm_l_cntl_mm_op,
	                          NULL,
	                          NULL );

	// Create control tree object for outer panel (to block-panel)
	// problem (right side).
	trmm_r_cntl_op_bp
	=
	bli_trmm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT1,
	                          trmm_mc,
	                          trmm_ni,
	                          NULL,
	                          trmm_r_packa_cntl,
	                          trmm_r_packb_cntl,
	                          NULL,
	                          trmm_cntl_bp_ke,
	                          gemm_cntl_bp_ke,
	                          NULL );

	// Create control tree object for general problem via multiple
	// rank-k (outer panel) updates (right side).
	trmm_r_cntl_mm_op
	=
	bli_trmm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT3,
	                          trmm_kc,
	                          NULL,
	                          NULL,
	                          NULL, 
	                          NULL,
	                          NULL,
	                          trmm_r_cntl_op_bp,
	                          NULL,
	                          NULL );

	// Create control tree object for very large problem via multiple
	// general problems (right side).
	trmm_r_cntl_vl_mm
	=
	bli_trmm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT2,
	                          trmm_nc,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          trmm_r_cntl_mm_op,
	                          NULL,
	                          NULL );

	// Alias the "master" trmm control trees to shorter names.
	trmm_l_cntl = trmm_l_cntl_vl_mm;
	trmm_r_cntl = trmm_r_cntl_vl_mm;
}

void bli_trmm_cntl_finalize()
{
	bli_blksz_obj_free( trmm_mc );
	bli_blksz_obj_free( trmm_nc );
	bli_blksz_obj_free( trmm_kc );
	bli_blksz_obj_free( trmm_mr );
	bli_blksz_obj_free( trmm_nr );
	bli_blksz_obj_free( trmm_kr );
	bli_blksz_obj_free( trmm_ni );

	bli_cntl_obj_free( trmm_l_packa_cntl );
	bli_cntl_obj_free( trmm_l_packb_cntl );
	bli_cntl_obj_free( trmm_r_packa_cntl );
	bli_cntl_obj_free( trmm_r_packb_cntl );
	bli_cntl_obj_free( trmm_packc_cntl );
	bli_cntl_obj_free( trmm_unpackc_cntl );

	bli_cntl_obj_free( trmm_cntl_bp_ke );
	bli_cntl_obj_free( trmm_l_cntl_op_bp );
	bli_cntl_obj_free( trmm_l_cntl_mm_op );
	bli_cntl_obj_free( trmm_l_cntl_vl_mm );
	bli_cntl_obj_free( trmm_r_cntl_op_bp );
	bli_cntl_obj_free( trmm_r_cntl_mm_op );
	bli_cntl_obj_free( trmm_r_cntl_vl_mm );
}

trmm_t* bli_trmm_cntl_obj_create( impl_t     impl_type,
                                  varnum_t   var_num,
                                  blksz_t*   b,
                                  blksz_t*   b_aux,
                                  scalm_t*   sub_scalm,
                                  packm_t*   sub_packm_a,
                                  packm_t*   sub_packm_b,
                                  packm_t*   sub_packm_c,
                                  trmm_t*    sub_trmm,
                                  gemm_t*    sub_gemm,
                                  unpackm_t* sub_unpackm_c )
{
	trmm_t* cntl;

	cntl = ( trmm_t* ) bli_malloc( sizeof(trmm_t) );	

	cntl->impl_type     = impl_type;
	cntl->var_num       = var_num;
	cntl->b             = b;
	cntl->b_aux         = b_aux;
	cntl->sub_scalm     = sub_scalm;
	cntl->sub_packm_a   = sub_packm_a;
	cntl->sub_packm_b   = sub_packm_b;
	cntl->sub_packm_c   = sub_packm_c;
	cntl->sub_trmm      = sub_trmm;
	cntl->sub_gemm      = sub_gemm;
	cntl->sub_unpackm_c = sub_unpackm_c;

	return cntl;
}

