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

#include "blis2.h"

extern scalm_t*   scalm_cntl;

gemm_t*           hemm_cntl;

gemm_t*           hemm_cntl_bp_ke;
gemm_t*           hemm_cntl_op_bp;
gemm_t*           hemm_cntl_mm_op;
gemm_t*           hemm_cntl_vl_mm;

packm_t*          hemm_packa_cntl;
packm_t*          hemm_packb_cntl;
packm_t*          hemm_packc_cntl;
unpackm_t*        hemm_unpackc_cntl;

blksz_t*          hemm_mc;
blksz_t*          hemm_nc;
blksz_t*          hemm_kc;
blksz_t*          hemm_mr;
blksz_t*          hemm_nr;
blksz_t*          hemm_kr;
blksz_t*          hemm_ni;

// Cache blocksizes.

#define BLIS_HEMM_KC_S BLIS_DEFAULT_KC_S
#define BLIS_HEMM_KC_D BLIS_DEFAULT_KC_D
#define BLIS_HEMM_KC_C BLIS_DEFAULT_KC_C
#define BLIS_HEMM_KC_Z BLIS_DEFAULT_KC_Z

#define BLIS_HEMM_MC_S BLIS_DEFAULT_MC_S
#define BLIS_HEMM_MC_D BLIS_DEFAULT_MC_D
#define BLIS_HEMM_MC_C BLIS_DEFAULT_MC_C
#define BLIS_HEMM_MC_Z BLIS_DEFAULT_MC_Z

#define BLIS_HEMM_NC_S BLIS_DEFAULT_NC_S
#define BLIS_HEMM_NC_D BLIS_DEFAULT_NC_D
#define BLIS_HEMM_NC_C BLIS_DEFAULT_NC_C
#define BLIS_HEMM_NC_Z BLIS_DEFAULT_NC_Z

// Register blocking 

#define BLIS_HEMM_KR_S BLIS_DEFAULT_KR_S
#define BLIS_HEMM_KR_D BLIS_DEFAULT_KR_D
#define BLIS_HEMM_KR_C BLIS_DEFAULT_KR_C
#define BLIS_HEMM_KR_Z BLIS_DEFAULT_KR_Z

#define BLIS_HEMM_MR_S BLIS_DEFAULT_MR_S
#define BLIS_HEMM_MR_D BLIS_DEFAULT_MR_D
#define BLIS_HEMM_MR_C BLIS_DEFAULT_MR_C
#define BLIS_HEMM_MR_Z BLIS_DEFAULT_MR_Z

#define BLIS_HEMM_NR_S BLIS_DEFAULT_NR_S
#define BLIS_HEMM_NR_D BLIS_DEFAULT_NR_D
#define BLIS_HEMM_NR_C BLIS_DEFAULT_NR_C
#define BLIS_HEMM_NR_Z BLIS_DEFAULT_NR_Z

// Incremental pack blocking

#define BLIS_HEMM_NI_S BLIS_DEFAULT_NI_S
#define BLIS_HEMM_NI_D BLIS_DEFAULT_NI_D
#define BLIS_HEMM_NI_C BLIS_DEFAULT_NI_C
#define BLIS_HEMM_NI_Z BLIS_DEFAULT_NI_Z


void bl2_hemm_cntl_init()
{
	// Create blocksize objects for each dimension.
	hemm_mc = bl2_blksz_obj_create( BLIS_HEMM_MC_S,
	                                BLIS_HEMM_MC_D,
	                                BLIS_HEMM_MC_C,
	                                BLIS_HEMM_MC_Z );

	hemm_nc = bl2_blksz_obj_create( BLIS_HEMM_NC_S,
	                                BLIS_HEMM_NC_D,
	                                BLIS_HEMM_NC_C,
	                                BLIS_HEMM_NC_Z );

	hemm_kc = bl2_blksz_obj_create( BLIS_HEMM_KC_S,
	                                BLIS_HEMM_KC_D,
	                                BLIS_HEMM_KC_C,
	                                BLIS_HEMM_KC_Z );

	hemm_mr = bl2_blksz_obj_create( BLIS_HEMM_MR_S,
	                                BLIS_HEMM_MR_D,
	                                BLIS_HEMM_MR_C,
	                                BLIS_HEMM_MR_Z );

	hemm_nr = bl2_blksz_obj_create( BLIS_HEMM_NR_S,
	                                BLIS_HEMM_NR_D,
	                                BLIS_HEMM_NR_C,
	                                BLIS_HEMM_NR_Z );

	hemm_kr = bl2_blksz_obj_create( BLIS_HEMM_KR_S,
	                                BLIS_HEMM_KR_D,
	                                BLIS_HEMM_KR_C,
	                                BLIS_HEMM_KR_Z );

	hemm_ni = bl2_blksz_obj_create( BLIS_HEMM_NI_S,
	                                BLIS_HEMM_NI_D,
	                                BLIS_HEMM_NI_C,
	                                BLIS_HEMM_NI_Z );


	// Create control tree objects for packm operations on a, b, and c.
	hemm_packa_cntl
	=
	bl2_packm_cntl_obj_create( BLIS_BLOCKED,
	                           BLIS_VARIANT2,
	                           hemm_mr,
	                           hemm_kr,
	                           FALSE, // do NOT scale by alpha
	                           TRUE,  // densify
	                           FALSE, // do NOT invert diagonal
	                           FALSE, // reverse iteration if upper?
	                           FALSE, // reverse iteration if lower?
	                           BLIS_PACKED_ROW_PANELS,
	                           BLIS_BUFFER_FOR_A_BLOCK );

	hemm_packb_cntl
	=
	bl2_packm_cntl_obj_create( BLIS_BLOCKED,
	                           BLIS_VARIANT2,
	                           hemm_kr,
	                           hemm_nr,
	                           FALSE, // do NOT scale by alpha
	                           FALSE, // already dense; densify not necessary
	                           FALSE, // do NOT invert diagonal
	                           FALSE, // reverse iteration if upper?
	                           FALSE, // reverse iteration if lower?
	                           BLIS_PACKED_COL_PANELS,
	                           BLIS_BUFFER_FOR_B_PANEL );

	hemm_packc_cntl
	=
	bl2_packm_cntl_obj_create( BLIS_UNBLOCKED,
	                           BLIS_VARIANT1,
	                           hemm_mr,
	                           hemm_nr,
	                           FALSE, // do NOT scale by beta
	                           FALSE, // already dense; densify not necessary
	                           FALSE, // do NOT invert diagonal
	                           FALSE, // reverse iteration if upper?
	                           FALSE, // reverse iteration if lower?
	                           BLIS_PACKED_COLUMNS,
	                           BLIS_BUFFER_FOR_GEN_USE );

	hemm_unpackc_cntl
	=
	bl2_unpackm_cntl_obj_create( BLIS_UNBLOCKED,
	                             BLIS_VARIANT1,
	                             NULL ); // no blocksize needed


	// Create control tree object for lowest-level block-panel kernel.
	hemm_cntl_bp_ke
	=
	bl2_gemm_cntl_obj_create( BLIS_UNB_OPT,
	                          BLIS_VARIANT2,
	                          NULL, NULL, NULL, NULL,
	                          NULL, NULL, NULL, NULL );

	// Create control tree object for outer panel (to block-panel)
	// problem, packing a and b only.
	hemm_cntl_op_bp
	=
	bl2_gemm_cntl_obj_create( BLIS_BLOCKED,
	                          //BLIS_VARIANT4,  // var1 with incremental pack in iter 0
	                          BLIS_VARIANT1,
	                          hemm_mc,
	                          hemm_ni,
	                          NULL,
	                          hemm_packa_cntl,
	                          hemm_packb_cntl,
	                          NULL,
	                          hemm_cntl_bp_ke,
	                          NULL );

	// Create control tree object for general problem via multiple
	// rank-k (outer panel) updates, packing a and b only.
	hemm_cntl_mm_op
	=
	bl2_gemm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT3,
	                          hemm_kc,
	                          NULL,
	                          NULL,
	                          NULL, 
	                          NULL,
	                          NULL,
	                          hemm_cntl_op_bp,
	                          NULL );

	// Create control tree object for very large problem via multiple
	// general problems, packing a and b only.
	hemm_cntl_vl_mm
	=
	bl2_gemm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT2,
	                          hemm_nc,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          hemm_cntl_mm_op,
	                          NULL );

	// Alias the "master" hemm control tree to a shorter name.
	hemm_cntl = hemm_cntl_mm_op;

}

void bl2_hemm_cntl_finalize()
{
	bl2_blksz_obj_free( hemm_mc );
	bl2_blksz_obj_free( hemm_nc );
	bl2_blksz_obj_free( hemm_kc );
	bl2_blksz_obj_free( hemm_mr );
	bl2_blksz_obj_free( hemm_nr );
	bl2_blksz_obj_free( hemm_kr );
	bl2_blksz_obj_free( hemm_ni );

	bl2_cntl_obj_free( hemm_packa_cntl );
	bl2_cntl_obj_free( hemm_packb_cntl );
	bl2_cntl_obj_free( hemm_packc_cntl );
	bl2_cntl_obj_free( hemm_unpackc_cntl );

	bl2_cntl_obj_free( hemm_cntl_bp_ke );
	bl2_cntl_obj_free( hemm_cntl_op_bp );
	bl2_cntl_obj_free( hemm_cntl_mm_op );
	bl2_cntl_obj_free( hemm_cntl_vl_mm );
}

