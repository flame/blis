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

extern blksz_t*   gemm_mc;
extern blksz_t*   gemm_nc;
extern blksz_t*   gemm_kc;
extern blksz_t*   gemm_mr;
extern blksz_t*   gemm_nr;
extern blksz_t*   gemm_kr;

extern func_t*    gemm_ukrs;

packm_t*          herk_packa_cntl;
packm_t*          herk_packb_cntl;

herk_t*           herk_cntl_bp_ke;
herk_t*           herk_cntl_op_bp;
herk_t*           herk_cntl_mm_op;
herk_t*           herk_cntl_vl_mm;

herk_t*           herk_cntl;


void bli_herk_cntl_init()
{
	// Create control tree objects for packm operations.
	herk_packa_cntl
	=
	bli_packm_cntl_obj_create( BLIS_BLOCKED,
	                           BLIS_VARIANT1,
	                           gemm_mr,
	                           gemm_kr,
	                           FALSE, // do NOT invert diagonal
	                           FALSE, // reverse iteration if upper?
	                           FALSE, // reverse iteration if lower?
	                           BLIS_PACKED_ROW_PANELS,
	                           BLIS_BUFFER_FOR_A_BLOCK );

	herk_packb_cntl
	=
	bli_packm_cntl_obj_create( BLIS_BLOCKED,
	                           BLIS_VARIANT1,
	                           gemm_kr,
	                           gemm_nr,
	                           FALSE, // do NOT invert diagonal
	                           FALSE, // reverse iteration if upper?
	                           FALSE, // reverse iteration if lower?
	                           BLIS_PACKED_COL_PANELS,
	                           BLIS_BUFFER_FOR_B_PANEL );


	// Create control tree object for lowest-level block-panel kernel.
	herk_cntl_bp_ke
	=
	bli_herk_cntl_obj_create( BLIS_UNB_OPT,
	                          BLIS_VARIANT2,
	                          NULL,
	                          gemm_ukrs,
	                          NULL, NULL, NULL,
	                          NULL, NULL, NULL );

	// Create control tree object for outer panel (to block-panel)
	// problem.
	herk_cntl_op_bp
	=
	bli_herk_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT1,
	                          gemm_mc,
	                          gemm_ukrs,
	                          NULL,
	                          herk_packa_cntl,
	                          herk_packb_cntl,
	                          NULL,
	                          herk_cntl_bp_ke,
	                          NULL );

	// Create control tree object for general problem via multiple
	// rank-k (outer panel) updates.
	herk_cntl_mm_op
	=
	bli_herk_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT3,
	                          gemm_kc,
	                          gemm_ukrs,
	                          NULL,
	                          NULL, 
	                          NULL,
	                          NULL,
	                          herk_cntl_op_bp,
	                          NULL );

	// Create control tree object for very large problem via multiple
	// general problems.
	herk_cntl_vl_mm
	=
	bli_herk_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT2,
	                          gemm_nc,
	                          gemm_ukrs,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          herk_cntl_mm_op,
	                          NULL );

	// Alias the "master" herk control tree to a shorter name.
	herk_cntl = herk_cntl_vl_mm;
}

void bli_herk_cntl_finalize()
{
	bli_cntl_obj_free( herk_packa_cntl );
	bli_cntl_obj_free( herk_packb_cntl );

	bli_cntl_obj_free( herk_cntl_bp_ke );
	bli_cntl_obj_free( herk_cntl_op_bp );
	bli_cntl_obj_free( herk_cntl_mm_op );
	bli_cntl_obj_free( herk_cntl_vl_mm );
}

herk_t* bli_herk_cntl_obj_create( impl_t     impl_type,
                                  varnum_t   var_num,
                                  blksz_t*   b,
                                  func_t*    gemm_ukrs_,
                                  scalm_t*   sub_scalm,
                                  packm_t*   sub_packm_a,
                                  packm_t*   sub_packm_b,
                                  packm_t*   sub_packm_c,
                                  herk_t*    sub_herk,
                                  unpackm_t* sub_unpackm_c )
{
	herk_t* cntl;

	cntl = ( herk_t* ) bli_malloc( sizeof(herk_t) );

	cntl->impl_type     = impl_type;
	cntl->var_num       = var_num;
	cntl->b             = b;
	cntl->gemm_ukrs     = gemm_ukrs_; // avoid name conflict with global symbol
	cntl->sub_scalm     = sub_scalm;
	cntl->sub_packm_a   = sub_packm_a;
	cntl->sub_packm_b   = sub_packm_b;
	cntl->sub_packm_c   = sub_packm_c;
	cntl->sub_herk      = sub_herk;
	cntl->sub_unpackm_c = sub_unpackm_c;

	return cntl;
}

