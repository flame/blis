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

extern herk_t*    herk_cntl_bp_ke;

packm_t*          her2k_packa_cntl;
packm_t*          her2k_packb_cntl;

her2k_t*          her2k_cntl_bp_ke;
her2k_t*          her2k_cntl_op_bp;
her2k_t*          her2k_cntl_mm_op;
her2k_t*          her2k_cntl_vl_mm;

her2k_t*          her2k_cntl;


void bli_her2k_cntl_init()
{

	// Create control tree objects for packm operations.
	her2k_packa_cntl
	=
	bli_packm_cntl_obj_create( BLIS_BLOCKED,
	                           BLIS_VARIANT2,
	                           her2k_mr,
	                           her2k_kr,
	                           FALSE, // do NOT invert diagonal
	                           FALSE, // reverse iteration if upper?
	                           FALSE, // reverse iteration if lower?
	                           BLIS_PACKED_ROW_PANELS,
	                           BLIS_BUFFER_FOR_A_BLOCK );

	her2k_packb_cntl
	=
	bli_packm_cntl_obj_create( BLIS_BLOCKED,
	                           BLIS_VARIANT2,
	                           her2k_kr,
	                           her2k_nr,
	                           FALSE, // do NOT invert diagonal
	                           FALSE, // reverse iteration if upper?
	                           FALSE, // reverse iteration if lower?
	                           BLIS_PACKED_COL_PANELS,
	                           BLIS_BUFFER_FOR_B_PANEL );


	// Create control tree object for lowest-level block-panel kernel.
	her2k_cntl_bp_ke
	=
	bli_her2k_cntl_obj_create( BLIS_UNB_OPT,
	                           BLIS_VARIANT2,
	                           NULL,
	                           gemm_ukrs,
	                           NULL, NULL, NULL, NULL,
	                           NULL, NULL, NULL );

	// Create control tree object for outer panel (to block-panel)
	// problem.
	her2k_cntl_op_bp
	=
	bli_her2k_cntl_obj_create( BLIS_BLOCKED,
	                           BLIS_VARIANT1,
	                           her2k_mc,
	                           NULL,
	                           NULL,
	                           her2k_packa_cntl,
	                           her2k_packb_cntl,
	                           NULL,
	                           her2k_cntl_bp_ke,
	                           herk_cntl_bp_ke,
	                           NULL );

	// Create control tree object for general problem via multiple
	// rank-k (outer panel) updates.
	her2k_cntl_mm_op
	=
	bli_her2k_cntl_obj_create( BLIS_BLOCKED,
	                           BLIS_VARIANT3,
	                           her2k_kc,
	                           NULL,
	                           NULL,
	                           NULL, 
	                           NULL,
	                           NULL,
	                           her2k_cntl_op_bp,
	                           NULL,
	                           NULL );

	// Create control tree object for very large problem via multiple
	// general problems.
	her2k_cntl_vl_mm
	=
	bli_her2k_cntl_obj_create( BLIS_BLOCKED,
	                           BLIS_VARIANT2,
	                           her2k_nc,
	                           NULL,
	                           NULL,
	                           NULL, 
	                           NULL,
	                           NULL,
	                           her2k_cntl_mm_op,
	                           NULL,
	                           NULL );

	// Alias the "master" her2k control tree to a shorter name.
	her2k_cntl = her2k_cntl_vl_mm;
}

void bli_her2k_cntl_finalize()
{
	bli_cntl_obj_free( her2k_packa_cntl );
	bli_cntl_obj_free( her2k_packb_cntl );

	bli_cntl_obj_free( her2k_cntl_bp_ke );
	bli_cntl_obj_free( her2k_cntl_op_bp );
	bli_cntl_obj_free( her2k_cntl_mm_op );
	bli_cntl_obj_free( her2k_cntl_vl_mm );
}

her2k_t* bli_her2k_cntl_obj_create( impl_t     impl_type,
                                    varnum_t   var_num,
                                    blksz_t*   b,
                                    func_t*    gemm_ukrs_,
                                    scalm_t*   sub_scalm,
                                    packm_t*   sub_packm_a,
                                    packm_t*   sub_packm_b,
                                    packm_t*   sub_packm_c,
                                    her2k_t*   sub_her2k,
                                    herk_t*    sub_herk,
                                    unpackm_t* sub_unpackm_c )
{
	her2k_t* cntl;

	cntl = ( her2k_t* ) bli_malloc( sizeof(her2k_t) );

	cntl->impl_type     = impl_type;
	cntl->var_num       = var_num;
	cntl->b             = b;
	cntl->gemm_ukrs     = gemm_ukrs_; // avoid name conflict with global symbol
	cntl->sub_scalm     = sub_scalm;
	cntl->sub_packm_a   = sub_packm_a;
	cntl->sub_packm_b   = sub_packm_b;
	cntl->sub_packm_c   = sub_packm_c;
	cntl->sub_her2k     = sub_her2k;
	cntl->sub_herk      = sub_herk;
	cntl->sub_unpackm_c = sub_unpackm_c;

	return cntl;
}

