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

extern blksz_t*   gemm4m1_mc;
extern blksz_t*   gemm4m1_nc;
extern blksz_t*   gemm4m1_kc;
extern blksz_t*   gemm4m1_mr;
extern blksz_t*   gemm4m1_nr;
extern blksz_t*   gemm4m1_kr;

extern func_t*    gemm4m1_ukrs;

func_t*           gemmtrsm4m1_l_ukrs;
func_t*           gemmtrsm4m1_u_ukrs;

func_t*           trsm4m1_l_ukrs;
func_t*           trsm4m1_u_ukrs;

packm_t*          trsm4m1_l_packa_cntl;
packm_t*          trsm4m1_l_packb_cntl;

packm_t*          trsm4m1_r_packa_cntl;
packm_t*          trsm4m1_r_packb_cntl;

trsm_t*           trsm4m1_cntl_bp_ke;

trsm_t*           trsm4m1_l_cntl_op_bp;
trsm_t*           trsm4m1_l_cntl_mm_op;
trsm_t*           trsm4m1_l_cntl_vl_mm;

trsm_t*           trsm4m1_r_cntl_op_bp;
trsm_t*           trsm4m1_r_cntl_mm_op;
trsm_t*           trsm4m1_r_cntl_vl_mm;

trsm_t*           trsm4m1_l_cntl;
trsm_t*           trsm4m1_r_cntl;


void bli_trsm4m1_cntl_init()
{

	// Create function pointer objects for each datatype-specific
	// gemmtrsm4m1_l and gemmtrsm4m1_u micro-kernel.
	gemmtrsm4m1_l_ukrs
	=
	bli_func_obj_create( NULL,                       FALSE,
	                     NULL,                       FALSE,
	                     BLIS_CGEMMTRSM4M1_L_UKERNEL, FALSE,
	                     BLIS_ZGEMMTRSM4M1_L_UKERNEL, FALSE );

	gemmtrsm4m1_u_ukrs
	=
	bli_func_obj_create( NULL,                       FALSE,
	                     NULL,                       FALSE,
	                     BLIS_CGEMMTRSM4M1_U_UKERNEL, FALSE,
	                     BLIS_ZGEMMTRSM4M1_U_UKERNEL, FALSE );


	// Create function pointer objects for each datatype-specific
	// trsm4m1_l and trsm4m1_u micro-kernel.
	trsm4m1_l_ukrs
	=
	bli_func_obj_create( NULL,                   FALSE,
	                     NULL,                   FALSE,
	                     BLIS_CTRSM4M1_L_UKERNEL, FALSE,
	                     BLIS_ZTRSM4M1_L_UKERNEL, FALSE );

	trsm4m1_u_ukrs
	=
	bli_func_obj_create( NULL,                   FALSE,
	                     NULL,                   FALSE,
	                     BLIS_CTRSM4M1_U_UKERNEL, FALSE,
	                     BLIS_ZTRSM4M1_U_UKERNEL, FALSE );


	// Create control tree objects for packm operations (left side).
	trsm4m1_l_packa_cntl
	=
	bli_packm_cntl_obj_create( BLIS_BLOCKED,
	                           BLIS_VARIANT2,
	                           // IMPORTANT: n dim multiple must be mr to
	                           // support right and bottom-right edge cases
	                           gemm4m1_mr,
	                           gemm4m1_mr,
	                           TRUE,  // invert diagonal
	                           TRUE,  // reverse iteration if upper?
	                           FALSE, // reverse iteration if lower?
	                           BLIS_PACKED_ROW_PANELS_4MI,
	                           BLIS_BUFFER_FOR_A_BLOCK );

	trsm4m1_l_packb_cntl
	=
	bli_packm_cntl_obj_create( BLIS_BLOCKED,
	                           BLIS_VARIANT2,
	                           // IMPORTANT: m dim multiple must be mr since
	                           // B_pack is updated (ie: serves as C) in trsm
	                           gemm4m1_mr,
	                           gemm4m1_nr,
	                           FALSE, // do NOT invert diagonal
	                           FALSE, // reverse iteration if upper?
	                           FALSE, // reverse iteration if lower?
	                           BLIS_PACKED_COL_PANELS_4MI,
	                           BLIS_BUFFER_FOR_B_PANEL );

	// Create control tree objects for packm operations (right side).
	trsm4m1_r_packa_cntl
	=
	bli_packm_cntl_obj_create( BLIS_BLOCKED,
	                           BLIS_VARIANT2,
	                           gemm4m1_nr,
	                           gemm4m1_mr,
	                           FALSE, // do NOT invert diagonal
	                           FALSE, // reverse iteration if upper?
	                           FALSE, // reverse iteration if lower?
	                           BLIS_PACKED_ROW_PANELS_4MI,
	                           BLIS_BUFFER_FOR_A_BLOCK );

	trsm4m1_r_packb_cntl
	=
	bli_packm_cntl_obj_create( BLIS_BLOCKED,
	                           BLIS_VARIANT2,
	                           gemm4m1_mr,
	                           gemm4m1_mr,
	                           TRUE,  // invert diagonal
	                           FALSE, // reverse iteration if upper?
	                           TRUE,  // reverse iteration if lower?
	                           BLIS_PACKED_COL_PANELS_4MI,
	                           BLIS_BUFFER_FOR_B_PANEL );


	// Create control tree object for lowest-level block-panel kernel.
	trsm4m1_cntl_bp_ke
	=
	bli_trsm_cntl_obj_create( BLIS_UNB_OPT,
	                          BLIS_VARIANT2,
	                          NULL,
	                          gemm4m1_ukrs,
	                          gemmtrsm4m1_l_ukrs,
	                          gemmtrsm4m1_u_ukrs,
	                          NULL, NULL, NULL, NULL,
	                          NULL, NULL, NULL );

	// Create control tree object for outer panel (to block-panel)
	// problem (left side).
	trsm4m1_l_cntl_op_bp
	=
	bli_trsm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT1,
	                          gemm4m1_mc,
	                          NULL, NULL, NULL,
	                          NULL,
	                          trsm4m1_l_packa_cntl,
	                          trsm4m1_l_packb_cntl,
	                          NULL,
	                          trsm4m1_cntl_bp_ke,
	                          NULL,
	                          NULL );

	// Create control tree object for general problem via multiple
	// rank-k (outer panel) updates (left side).
	trsm4m1_l_cntl_mm_op
	=
	bli_trsm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT3,
	                          gemm4m1_kc,
	                          NULL, NULL, NULL,
	                          NULL,
	                          NULL, 
	                          NULL,
	                          NULL,
	                          trsm4m1_l_cntl_op_bp,
	                          NULL,
	                          NULL );

	// Create control tree object for very large problem via multiple
	// general problems (left side).
	trsm4m1_l_cntl_vl_mm
	=
	bli_trsm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT2,
	                          gemm4m1_nc,
	                          NULL, NULL, NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          trsm4m1_l_cntl_mm_op,
	                          NULL,
	                          NULL );

	// Create control tree object for outer panel (to block-panel)
	// problem (right side).
	trsm4m1_r_cntl_op_bp
	=
	bli_trsm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT1,
	                          gemm4m1_mc,
	                          NULL, NULL, NULL,
	                          NULL,
	                          trsm4m1_r_packa_cntl,
	                          trsm4m1_r_packb_cntl,
	                          NULL,
	                          trsm4m1_cntl_bp_ke,
	                          NULL,
	                          NULL );

	// Create control tree object for general problem via multiple
	// rank-k (outer panel) updates (right side).
	trsm4m1_r_cntl_mm_op
	=
	bli_trsm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT3,
	                          gemm4m1_kc,
	                          NULL, NULL, NULL,
	                          NULL,
	                          NULL, 
	                          NULL,
	                          NULL,
	                          trsm4m1_r_cntl_op_bp,
	                          NULL,
	                          NULL );

	// Create control tree object for very large problem via multiple
	// general problems (right side).
	trsm4m1_r_cntl_vl_mm
	=
	bli_trsm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT2,
	                          gemm4m1_nc,
	                          NULL, NULL, NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          trsm4m1_r_cntl_mm_op,
	                          NULL,
	                          NULL );

	// Alias the "master" trsm control trees to shorter names.
	trsm4m1_l_cntl = trsm4m1_l_cntl_vl_mm;
	trsm4m1_r_cntl = trsm4m1_r_cntl_vl_mm;
}

void bli_trsm4m1_cntl_finalize()
{
	bli_func_obj_free( gemmtrsm4m1_l_ukrs );
	bli_func_obj_free( gemmtrsm4m1_u_ukrs );
	bli_func_obj_free( trsm4m1_l_ukrs );
	bli_func_obj_free( trsm4m1_u_ukrs );

	bli_cntl_obj_free( trsm4m1_l_packa_cntl );
	bli_cntl_obj_free( trsm4m1_l_packb_cntl );
	bli_cntl_obj_free( trsm4m1_r_packa_cntl );
	bli_cntl_obj_free( trsm4m1_r_packb_cntl );

	bli_cntl_obj_free( trsm4m1_cntl_bp_ke );

	bli_cntl_obj_free( trsm4m1_l_cntl_op_bp );
	bli_cntl_obj_free( trsm4m1_l_cntl_mm_op );
	bli_cntl_obj_free( trsm4m1_l_cntl_vl_mm );
	bli_cntl_obj_free( trsm4m1_r_cntl_op_bp );
	bli_cntl_obj_free( trsm4m1_r_cntl_mm_op );
	bli_cntl_obj_free( trsm4m1_r_cntl_vl_mm );
}

