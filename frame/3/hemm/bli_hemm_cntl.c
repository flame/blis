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


void bli_hemm_cntl_init()
{
	// Create blocksize objects for each dimension.
	hemm_mc = bli_blksz_obj_create( BLIS_DEFAULT_MC_S, BLIS_EXTEND_MC_S,
	                                BLIS_DEFAULT_MC_D, BLIS_EXTEND_MC_D,
	                                BLIS_DEFAULT_MC_C, BLIS_EXTEND_MC_C,
	                                BLIS_DEFAULT_MC_Z, BLIS_EXTEND_MC_Z );

	hemm_nc = bli_blksz_obj_create( BLIS_DEFAULT_NC_S, BLIS_EXTEND_NC_S,
	                                BLIS_DEFAULT_NC_D, BLIS_EXTEND_NC_D,
	                                BLIS_DEFAULT_NC_C, BLIS_EXTEND_NC_C,
	                                BLIS_DEFAULT_NC_Z, BLIS_EXTEND_NC_Z );

	hemm_kc = bli_blksz_obj_create( BLIS_DEFAULT_KC_S, BLIS_EXTEND_KC_S,
	                                BLIS_DEFAULT_KC_D, BLIS_EXTEND_KC_D,
	                                BLIS_DEFAULT_KC_C, BLIS_EXTEND_KC_C,
	                                BLIS_DEFAULT_KC_Z, BLIS_EXTEND_KC_Z );

	hemm_mr = bli_blksz_obj_create( BLIS_DEFAULT_MR_S, BLIS_EXTEND_MR_S,
	                                BLIS_DEFAULT_MR_D, BLIS_EXTEND_MR_D,
	                                BLIS_DEFAULT_MR_C, BLIS_EXTEND_MR_C,
	                                BLIS_DEFAULT_MR_Z, BLIS_EXTEND_MR_Z );

	hemm_nr = bli_blksz_obj_create( BLIS_DEFAULT_NR_S, BLIS_EXTEND_NR_S,
	                                BLIS_DEFAULT_NR_D, BLIS_EXTEND_NR_D,
	                                BLIS_DEFAULT_NR_C, BLIS_EXTEND_NR_C,
	                                BLIS_DEFAULT_NR_Z, BLIS_EXTEND_NR_Z );

	hemm_kr = bli_blksz_obj_create( BLIS_DEFAULT_KR_S, BLIS_EXTEND_KR_S,
	                                BLIS_DEFAULT_KR_D, BLIS_EXTEND_KR_D,
	                                BLIS_DEFAULT_KR_C, BLIS_EXTEND_KR_C,
	                                BLIS_DEFAULT_KR_Z, BLIS_EXTEND_KR_Z );

	hemm_ni = bli_blksz_obj_create( BLIS_DEFAULT_NI_S, 0,
	                                BLIS_DEFAULT_NI_D, 0,
	                                BLIS_DEFAULT_NI_C, 0,
	                                BLIS_DEFAULT_NI_Z, 0 );


	// Create control tree objects for packm operations.
	hemm_packa_cntl
	=
	bli_packm_cntl_obj_create( BLIS_BLOCKED,
	                           BLIS_VARIANT2,
	                           hemm_mr,
	                           hemm_kr,
	                           TRUE,  // densify
	                           FALSE, // do NOT invert diagonal
	                           FALSE, // reverse iteration if upper?
	                           FALSE, // reverse iteration if lower?
	                           BLIS_PACKED_ROW_PANELS,
	                           BLIS_BUFFER_FOR_A_BLOCK );

	hemm_packb_cntl
	=
	bli_packm_cntl_obj_create( BLIS_BLOCKED,
	                           BLIS_VARIANT2,
	                           hemm_kr,
	                           hemm_nr,
	                           TRUE,  // densify
	                           FALSE, // do NOT invert diagonal
	                           FALSE, // reverse iteration if upper?
	                           FALSE, // reverse iteration if lower?
	                           BLIS_PACKED_COL_PANELS,
	                           BLIS_BUFFER_FOR_B_PANEL );

	// Create control tree objects for packm/unpackm operations on C.
	hemm_packc_cntl
	=
	bli_packm_cntl_obj_create( BLIS_UNBLOCKED,
	                           BLIS_VARIANT1,
	                           hemm_mr,
	                           hemm_nr,
	                           FALSE, // already dense; densify not necessary
	                           FALSE, // do NOT invert diagonal
	                           FALSE, // reverse iteration if upper?
	                           FALSE, // reverse iteration if lower?
	                           BLIS_PACKED_COLUMNS,
	                           BLIS_BUFFER_FOR_GEN_USE );

	hemm_unpackc_cntl
	=
	bli_unpackm_cntl_obj_create( BLIS_UNBLOCKED,
	                             BLIS_VARIANT1,
	                             NULL ); // no blocksize needed


	// Create control tree object for lowest-level block-panel kernel.
	hemm_cntl_bp_ke
	=
	bli_gemm_cntl_obj_create( BLIS_UNB_OPT,
	                          BLIS_VARIANT2,
	                          NULL, NULL, NULL, NULL,
	                          NULL, NULL, NULL, NULL );

	// Create control tree object for outer panel (to block-panel)
	// problem.
	hemm_cntl_op_bp
	=
	bli_gemm_cntl_obj_create( BLIS_BLOCKED,
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
	// rank-k (outer panel) updates.
	hemm_cntl_mm_op
	=
	bli_gemm_cntl_obj_create( BLIS_BLOCKED,
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
	// general problems.
	hemm_cntl_vl_mm
	=
	bli_gemm_cntl_obj_create( BLIS_BLOCKED,
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
	hemm_cntl = hemm_cntl_vl_mm;
}

void bli_hemm_cntl_finalize()
{
	bli_blksz_obj_free( hemm_mc );
	bli_blksz_obj_free( hemm_nc );
	bli_blksz_obj_free( hemm_kc );
	bli_blksz_obj_free( hemm_mr );
	bli_blksz_obj_free( hemm_nr );
	bli_blksz_obj_free( hemm_kr );
	bli_blksz_obj_free( hemm_ni );

	bli_cntl_obj_free( hemm_packa_cntl );
	bli_cntl_obj_free( hemm_packb_cntl );
	bli_cntl_obj_free( hemm_packc_cntl );
	bli_cntl_obj_free( hemm_unpackc_cntl );

	bli_cntl_obj_free( hemm_cntl_bp_ke );
	bli_cntl_obj_free( hemm_cntl_op_bp );
	bli_cntl_obj_free( hemm_cntl_mm_op );
	bli_cntl_obj_free( hemm_cntl_vl_mm );
}

