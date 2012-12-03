/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2012, The University of Texas

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

#include "blis2.h"

extern scalm_t*   scalm_cntl;
extern gemm_t*    gemm_cntl_bp_ke;

trmm_t*           trmm_cntl;

trmm_t*           trmm_cntl_bp_ke;
trmm_t*           trmm_cntl_op_bp;
trmm_t*           trmm_cntl_mm_op;
trmm_t*           trmm_cntl_vl_mm;

packm_t*          trmm_packa_cntl;
packm_t*          trmm_packb_cntl;
packm_t*          trmm_packc_cntl;
unpackm_t*        trmm_unpackc_cntl;

blksz_t*          trmm_mc;
blksz_t*          trmm_nc;
blksz_t*          trmm_kc;
blksz_t*          trmm_mr;
blksz_t*          trmm_nr;
blksz_t*          trmm_kr;
blksz_t*          trmm_ni;

// Cache blocksizes.

#define BLIS_TRMM_KC_S BLIS_DEFAULT_KC_S
#define BLIS_TRMM_KC_D BLIS_DEFAULT_KC_D
#define BLIS_TRMM_KC_C BLIS_DEFAULT_KC_C
#define BLIS_TRMM_KC_Z BLIS_DEFAULT_KC_Z

#define BLIS_TRMM_MC_S BLIS_DEFAULT_MC_S
#define BLIS_TRMM_MC_D BLIS_DEFAULT_MC_D
#define BLIS_TRMM_MC_C BLIS_DEFAULT_MC_C
#define BLIS_TRMM_MC_Z BLIS_DEFAULT_MC_Z

#define BLIS_TRMM_NC_S BLIS_DEFAULT_NC_S
#define BLIS_TRMM_NC_D BLIS_DEFAULT_NC_D
#define BLIS_TRMM_NC_C BLIS_DEFAULT_NC_C
#define BLIS_TRMM_NC_Z BLIS_DEFAULT_NC_Z

// Register blocking 

#define BLIS_TRMM_KR_S BLIS_DEFAULT_KR_S
#define BLIS_TRMM_KR_D BLIS_DEFAULT_KR_D
#define BLIS_TRMM_KR_C BLIS_DEFAULT_KR_C
#define BLIS_TRMM_KR_Z BLIS_DEFAULT_KR_Z

#define BLIS_TRMM_MR_S BLIS_DEFAULT_MR_S
#define BLIS_TRMM_MR_D BLIS_DEFAULT_MR_D
#define BLIS_TRMM_MR_C BLIS_DEFAULT_MR_C
#define BLIS_TRMM_MR_Z BLIS_DEFAULT_MR_Z

#define BLIS_TRMM_NR_S BLIS_DEFAULT_NR_S
#define BLIS_TRMM_NR_D BLIS_DEFAULT_NR_D
#define BLIS_TRMM_NR_C BLIS_DEFAULT_NR_C
#define BLIS_TRMM_NR_Z BLIS_DEFAULT_NR_Z

// Incremental pack blocking

#define BLIS_TRMM_NI_S BLIS_DEFAULT_NI_S
#define BLIS_TRMM_NI_D BLIS_DEFAULT_NI_D
#define BLIS_TRMM_NI_C BLIS_DEFAULT_NI_C
#define BLIS_TRMM_NI_Z BLIS_DEFAULT_NI_Z


void bl2_trmm_cntl_init()
{
	// Create blocksize objects for each dimension.
	trmm_mc = bl2_blksz_obj_create( BLIS_TRMM_MC_S,
	                                BLIS_TRMM_MC_D,
	                                BLIS_TRMM_MC_C,
	                                BLIS_TRMM_MC_Z );

	trmm_nc = bl2_blksz_obj_create( BLIS_TRMM_NC_S,
	                                BLIS_TRMM_NC_D,
	                                BLIS_TRMM_NC_C,
	                                BLIS_TRMM_NC_Z );

	trmm_kc = bl2_blksz_obj_create( BLIS_TRMM_KC_S,
	                                BLIS_TRMM_KC_D,
	                                BLIS_TRMM_KC_C,
	                                BLIS_TRMM_KC_Z );

	trmm_mr = bl2_blksz_obj_create( BLIS_TRMM_MR_S,
	                                BLIS_TRMM_MR_D,
	                                BLIS_TRMM_MR_C,
	                                BLIS_TRMM_MR_Z );

	trmm_nr = bl2_blksz_obj_create( BLIS_TRMM_NR_S,
	                                BLIS_TRMM_NR_D,
	                                BLIS_TRMM_NR_C,
	                                BLIS_TRMM_NR_Z );

	trmm_kr = bl2_blksz_obj_create( BLIS_TRMM_KR_S,
	                                BLIS_TRMM_KR_D,
	                                BLIS_TRMM_KR_C,
	                                BLIS_TRMM_KR_Z );

	trmm_ni = bl2_blksz_obj_create( BLIS_TRMM_NI_S,
	                                BLIS_TRMM_NI_D,
	                                BLIS_TRMM_NI_C,
	                                BLIS_TRMM_NI_Z );


	// Create control tree objects for packm operations on a, b, and c.
	trmm_packa_cntl
	=
	bl2_packm_cntl_obj_create( BLIS_BLOCKED,
	                           BLIS_VARIANT3, // pack panels of A compactly
	                           trmm_mr, // IMPORTANT: for consistency with trsm, "k" dim
	                           trmm_mr, // multiple is set to mr.
	                           FALSE, // do NOT scale by alpha
	                           TRUE,  // densify
	                           FALSE, // do NOT invert diagonal
	                           FALSE, // reverse iteration if upper?
	                           FALSE, // reverse iteration if lower?
	                           BLIS_PACKED_ROW_PANELS );

	trmm_packb_cntl
	=
	bl2_packm_cntl_obj_create( BLIS_BLOCKED,
	                           BLIS_VARIANT2,
	                           trmm_mr, // IMPORTANT: m dim multiple here must be mr
	                           trmm_nr, // since "k" dim multiple is set to mr above.
	                           FALSE, // do NOT scale by alpha
	                           FALSE, // already dense; densify not necessary
	                           FALSE, // do NOT invert diagonal
	                           FALSE, // reverse iteration if upper?
	                           FALSE, // reverse iteration if lower?
	                           BLIS_PACKED_COL_PANELS );

	trmm_packc_cntl
	=
	bl2_packm_cntl_obj_create( BLIS_UNBLOCKED,
	                           BLIS_VARIANT1,
	                           trmm_mr,
	                           trmm_nr,
	                           FALSE, // do NOT scale by beta
	                           FALSE, // already dense; densify not necessary
	                           FALSE, // do NOT invert diagonal
	                           FALSE, // reverse iteration if upper?
	                           FALSE, // reverse iteration if lower?
	                           BLIS_PACKED_COLUMNS );

	trmm_unpackc_cntl
	=
	bl2_unpackm_cntl_obj_create( BLIS_UNBLOCKED,
	                             BLIS_VARIANT1,
	                             NULL ); // no blocksize needed


	// Create control tree object for lowest-level block-panel kernel.
	trmm_cntl_bp_ke
	=
	bl2_trmm_cntl_obj_create( BLIS_UNB_OPT,
	                          BLIS_VARIANT2,
	                          NULL, NULL, NULL, NULL, NULL,
	                          NULL, NULL, NULL, NULL );

	// Create control tree object for outer panel (to block-panel)
	// problem, packing a and b.
	trmm_cntl_op_bp
	=
	bl2_trmm_cntl_obj_create( BLIS_BLOCKED,
	                          //BLIS_VARIANT4,  // var1 with incremental pack in iter 0
	                          BLIS_VARIANT1,
	                          trmm_mc,
	                          trmm_ni,
	                          NULL,
	                          trmm_packa_cntl,
	                          trmm_packb_cntl,
	                          NULL,
	                          trmm_cntl_bp_ke,
	                          gemm_cntl_bp_ke,
	                          NULL );

	// Create control tree object for general problem via multiple
	// rank-k (outer panel) updates, packing a and b.
	trmm_cntl_mm_op
	=
	bl2_trmm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT3,
	                          trmm_kc,
	                          NULL,
	                          NULL,
	                          NULL, 
	                          NULL,
	                          NULL,
	                          trmm_cntl_op_bp,
	                          NULL,
	                          NULL );

	// Create control tree object for very large problem via multiple
	// general problems, packing a and b.
	trmm_cntl_vl_mm
	=
	bl2_trmm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT2,
	                          trmm_nc,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          trmm_cntl_mm_op,
	                          NULL,
	                          NULL );

	// Alias the "master" trmm control tree to a shorter name.
	trmm_cntl = trmm_cntl_mm_op;

}

void bl2_trmm_cntl_finalize()
{
	bl2_blksz_obj_free( trmm_mc );
	bl2_blksz_obj_free( trmm_nc );
	bl2_blksz_obj_free( trmm_kc );
	bl2_blksz_obj_free( trmm_mr );
	bl2_blksz_obj_free( trmm_nr );
	bl2_blksz_obj_free( trmm_kr );
	bl2_blksz_obj_free( trmm_ni );

	bl2_cntl_obj_free( trmm_packa_cntl );
	bl2_cntl_obj_free( trmm_packb_cntl );
	bl2_cntl_obj_free( trmm_packc_cntl );
	bl2_cntl_obj_free( trmm_unpackc_cntl );

	bl2_cntl_obj_free( trmm_cntl_bp_ke );
	bl2_cntl_obj_free( trmm_cntl_op_bp );
	bl2_cntl_obj_free( trmm_cntl_mm_op );
	bl2_cntl_obj_free( trmm_cntl_vl_mm );
}

trmm_t* bl2_trmm_cntl_obj_create( impl_t     impl_type,
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

	cntl = ( trmm_t* ) bl2_malloc( sizeof(trmm_t) );	

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

