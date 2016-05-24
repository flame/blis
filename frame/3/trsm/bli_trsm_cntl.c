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

extern scalm_t*   scalm_cntl;

extern gemm_t*    gemm_cntl_bp_ke;

packm_t*          trsm_l_packa_cntl = NULL;
packm_t*          trsm_l_packb_cntl = NULL;

packm_t*          trsm_r_packa_cntl = NULL;
packm_t*          trsm_r_packb_cntl = NULL;

trsm_t*           trsm_cntl_bp_ke = NULL;

trsm_t*           trsm_l_cntl_op_bp = NULL;
trsm_t*           trsm_l_cntl_mm_op = NULL;
trsm_t*           trsm_l_cntl_vl_mm = NULL;

trsm_t*           trsm_r_cntl_op_bp = NULL;
trsm_t*           trsm_r_cntl_mm_op = NULL;
trsm_t*           trsm_r_cntl_vl_mm = NULL;

trsm_t*           trsm_l_cntl = NULL;
trsm_t*           trsm_r_cntl = NULL;


void bli_trsm_cntl_init()
{

	// Create control tree objects for packm operations (left side).
	trsm_l_packa_cntl
	=
	bli_packm_cntl_obj_create( BLIS_BLOCKED,
	                           BLIS_VARIANT1,
	                           // IMPORTANT: n dim multiple must be mr to
	                           // support right and bottom-right edge cases
	                           BLIS_MR,
	                           BLIS_MR,
	                           TRUE,  // invert diagonal
	                           TRUE,  // reverse iteration if upper?
	                           FALSE, // reverse iteration if lower?
	                           BLIS_PACKED_ROW_PANELS,
	                           BLIS_BUFFER_FOR_A_BLOCK );

	trsm_l_packb_cntl
	=
	bli_packm_cntl_obj_create( BLIS_BLOCKED,
	                           BLIS_VARIANT1,
	                           // IMPORTANT: m dim multiple must be mr since
	                           // B_pack is updated (ie: serves as C) in trsm
	                           BLIS_MR,
	                           BLIS_NR,
	                           FALSE, // do NOT invert diagonal
	                           FALSE, // reverse iteration if upper?
	                           FALSE, // reverse iteration if lower?
	                           BLIS_PACKED_COL_PANELS,
	                           BLIS_BUFFER_FOR_B_PANEL );

	// Create control tree objects for packm operations (right side).
	trsm_r_packa_cntl
	=
	bli_packm_cntl_obj_create( BLIS_BLOCKED,
	                           BLIS_VARIANT1,
	                           BLIS_NR,
	                           BLIS_MR,
	                           FALSE, // do NOT invert diagonal
	                           FALSE, // reverse iteration if upper?
	                           FALSE, // reverse iteration if lower?
	                           BLIS_PACKED_ROW_PANELS,
	                           BLIS_BUFFER_FOR_A_BLOCK );

	trsm_r_packb_cntl
	=
	bli_packm_cntl_obj_create( BLIS_BLOCKED,
	                           BLIS_VARIANT1, // pack panels of B compactly
	                           BLIS_MR,
	                           BLIS_MR,
	                           TRUE,  // invert diagonal
	                           FALSE, // reverse iteration if upper?
	                           TRUE,  // reverse iteration if lower?
	                           BLIS_PACKED_COL_PANELS,
	                           BLIS_BUFFER_FOR_B_PANEL );


	// Create control tree object for lowest-level block-panel kernel.
	trsm_cntl_bp_ke
	=
	bli_trsm_cntl_obj_create( BLIS_UNB_OPT,
	                          BLIS_VARIANT2,
	                          0, // bszid_t not used by macro-kernel
	                          NULL, NULL, NULL, NULL,
	                          NULL, NULL, NULL );

	// Create control tree object for outer panel (to block-panel)
	// problem (left side).
	trsm_l_cntl_op_bp
	=
	bli_trsm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT1,
	                          BLIS_MC,
	                          NULL,
	                          trsm_l_packa_cntl,
	                          trsm_l_packb_cntl,
	                          NULL,
	                          trsm_cntl_bp_ke,
	                          gemm_cntl_bp_ke,
	                          NULL );

	// Create control tree object for general problem via multiple
	// rank-k (outer panel) updates (left side).
	trsm_l_cntl_mm_op
	=
	bli_trsm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT3,
	                          BLIS_KC,
	                          NULL,
	                          NULL, 
	                          NULL,
	                          NULL,
	                          trsm_l_cntl_op_bp,
	                          NULL,
	                          NULL );

	// Create control tree object for very large problem via multiple
	// general problems (left side).
	trsm_l_cntl_vl_mm
	=
	bli_trsm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT2,
	                          BLIS_NC,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          trsm_l_cntl_mm_op,
	                          NULL,
	                          NULL );

	// Create control tree object for outer panel (to block-panel)
	// problem (right side).
	trsm_r_cntl_op_bp
	=
	bli_trsm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT1,
	                          BLIS_MC,
	                          NULL,
	                          trsm_r_packa_cntl,
	                          trsm_r_packb_cntl,
	                          NULL,
	                          trsm_cntl_bp_ke,
	                          gemm_cntl_bp_ke,
	                          NULL );

	// Create control tree object for general problem via multiple
	// rank-k (outer panel) updates (right side).
	trsm_r_cntl_mm_op
	=
	bli_trsm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT3,
	                          BLIS_KC,
	                          NULL,
	                          NULL, 
	                          NULL,
	                          NULL,
	                          trsm_r_cntl_op_bp,
	                          NULL,
	                          NULL );

	// Create control tree object for very large problem via multiple
	// general problems (right side).
	trsm_r_cntl_vl_mm
	=
	bli_trsm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT2,
	                          BLIS_NC,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          trsm_r_cntl_mm_op,
	                          NULL,
	                          NULL );

	// Alias the "master" trsm control trees to shorter names.
	trsm_l_cntl = trsm_l_cntl_vl_mm;
	trsm_r_cntl = trsm_r_cntl_vl_mm;
}

void bli_trsm_cntl_finalize()
{
	bli_cntl_obj_free( trsm_l_packa_cntl );
	bli_cntl_obj_free( trsm_l_packb_cntl );
	bli_cntl_obj_free( trsm_r_packa_cntl );
	bli_cntl_obj_free( trsm_r_packb_cntl );

	bli_cntl_obj_free( trsm_cntl_bp_ke );

	bli_cntl_obj_free( trsm_l_cntl_op_bp );
	bli_cntl_obj_free( trsm_l_cntl_mm_op );
	bli_cntl_obj_free( trsm_l_cntl_vl_mm );
	bli_cntl_obj_free( trsm_r_cntl_op_bp );
	bli_cntl_obj_free( trsm_r_cntl_mm_op );
	bli_cntl_obj_free( trsm_r_cntl_vl_mm );
}

trsm_t* bli_trsm_cntl_obj_create( impl_t     impl_type,
                                  varnum_t   var_num,
                                  bszid_t    bszid,
                                  scalm_t*   sub_scalm,
                                  packm_t*   sub_packm_a,
                                  packm_t*   sub_packm_b,
                                  packm_t*   sub_packm_c,
                                  trsm_t*    sub_trsm,
                                  gemm_t*    sub_gemm,
                                  unpackm_t* sub_unpackm_c )
{
	trsm_t* cntl;

	cntl = ( trsm_t* ) bli_malloc_intl( sizeof(trsm_t) );

	cntl->impl_type       = impl_type;
	cntl->var_num         = var_num;
	cntl->bszid           = bszid;
	cntl->sub_scalm       = sub_scalm;
	cntl->sub_packm_a     = sub_packm_a;
	cntl->sub_packm_b     = sub_packm_b;
	cntl->sub_packm_c     = sub_packm_c;
	cntl->sub_trsm        = sub_trsm;
	cntl->sub_gemm        = sub_gemm;
	cntl->sub_unpackm_c   = sub_unpackm_c;

	return cntl;
}

