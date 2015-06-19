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

blksz_t*          gemm_mc;
blksz_t*          gemm_nc;
blksz_t*          gemm_kc;
blksz_t*          gemm_mr;
blksz_t*          gemm_nr;
blksz_t*          gemm_kr;

func_t*           gemm_ukrs;
func_t*           gemm_ref_ukrs;

packm_t*          gemm_packa_cntl;
packm_t*          gemm_packb_cntl;

gemm_t*           gemm_cntl_bp_ke;
gemm_t*           gemm_cntl_op_bp;
gemm_t*           gemm_cntl_mm_op;
gemm_t*           gemm_cntl_vl_mm;

gemm_t*           gemm_cntl;

void bli_gemm_cntl_init()
{
	// Create blocksize objects for each dimension.
	gemm_mc
	=
	bli_blksz_obj_create( BLIS_DEFAULT_MC_S, BLIS_MAXIMUM_MC_S,
	                      BLIS_DEFAULT_MC_D, BLIS_MAXIMUM_MC_D,
	                      BLIS_DEFAULT_MC_C, BLIS_MAXIMUM_MC_C,
	                      BLIS_DEFAULT_MC_Z, BLIS_MAXIMUM_MC_Z );
	gemm_nc
	=
	bli_blksz_obj_create( BLIS_DEFAULT_NC_S, BLIS_MAXIMUM_NC_S,
	                      BLIS_DEFAULT_NC_D, BLIS_MAXIMUM_NC_D,
	                      BLIS_DEFAULT_NC_C, BLIS_MAXIMUM_NC_C,
	                      BLIS_DEFAULT_NC_Z, BLIS_MAXIMUM_NC_Z );
	gemm_kc
	=
	bli_blksz_obj_create( BLIS_DEFAULT_KC_S, BLIS_MAXIMUM_KC_S,
	                      BLIS_DEFAULT_KC_D, BLIS_MAXIMUM_KC_D,
	                      BLIS_DEFAULT_KC_C, BLIS_MAXIMUM_KC_C,
	                      BLIS_DEFAULT_KC_Z, BLIS_MAXIMUM_KC_Z );
	gemm_mr
	=
	bli_blksz_obj_create( BLIS_DEFAULT_MR_S, BLIS_PACKDIM_MR_S,
	                      BLIS_DEFAULT_MR_D, BLIS_PACKDIM_MR_D,
	                      BLIS_DEFAULT_MR_C, BLIS_PACKDIM_MR_C,
	                      BLIS_DEFAULT_MR_Z, BLIS_PACKDIM_MR_Z );
	gemm_nr
	=
	bli_blksz_obj_create( BLIS_DEFAULT_NR_S, BLIS_PACKDIM_NR_S,
	                      BLIS_DEFAULT_NR_D, BLIS_PACKDIM_NR_D,
	                      BLIS_DEFAULT_NR_C, BLIS_PACKDIM_NR_C,
	                      BLIS_DEFAULT_NR_Z, BLIS_PACKDIM_NR_Z );
	gemm_kr
	=
	bli_blksz_obj_create( BLIS_DEFAULT_KR_S, BLIS_PACKDIM_KR_S,
	                      BLIS_DEFAULT_KR_D, BLIS_PACKDIM_KR_D,
	                      BLIS_DEFAULT_KR_C, BLIS_PACKDIM_KR_C,
	                      BLIS_DEFAULT_KR_Z, BLIS_PACKDIM_KR_Z );


	// Attach the register blksz_t objects as blocksize multiples to the cache
	// blksz_t objects.
	bli_blksz_obj_attach_mult_to( gemm_mr, gemm_mc );
	bli_blksz_obj_attach_mult_to( gemm_nr, gemm_nc );
	bli_blksz_obj_attach_mult_to( gemm_kr, gemm_kc );


	// Attach the mr and nr blksz_t objects to each cache blksz_t object.
	// The primary example of why this is needed relates to nudging kc.
	// In hemm, symm, trmm, or trmm3, we need to know both mr and nr,
	// since the multiple we target in nudging depends on whether the
	// structured matrix is on the left or the right.
	bli_blksz_obj_attach_mr_nr_to( gemm_mr, gemm_nr, gemm_mc );
	bli_blksz_obj_attach_mr_nr_to( gemm_mr, gemm_nr, gemm_nc );
	bli_blksz_obj_attach_mr_nr_to( gemm_mr, gemm_nr, gemm_kc );


	// Create function pointer object for each datatype-specific gemm
	// micro-kernel.
	gemm_ukrs
	=
	bli_func_obj_create( BLIS_SGEMM_UKERNEL, BLIS_SGEMM_UKERNEL_PREFERS_CONTIG_ROWS,
	                     BLIS_DGEMM_UKERNEL, BLIS_DGEMM_UKERNEL_PREFERS_CONTIG_ROWS,
	                     BLIS_CGEMM_UKERNEL, BLIS_CGEMM_UKERNEL_PREFERS_CONTIG_ROWS,
	                     BLIS_ZGEMM_UKERNEL, BLIS_ZGEMM_UKERNEL_PREFERS_CONTIG_ROWS );


	// Create function pointer object for reference micro-kernels.
	gemm_ref_ukrs
	=
	bli_func_obj_create( BLIS_SGEMM_UKERNEL_REF, FALSE,
	                     BLIS_DGEMM_UKERNEL_REF, FALSE,
	                     BLIS_CGEMM_UKERNEL_REF, FALSE,
	                     BLIS_ZGEMM_UKERNEL_REF, FALSE );


	// Create control tree objects for packm operations.
	gemm_packa_cntl
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

	gemm_packb_cntl
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


	//
	// Create a control tree for packing A and B, and streaming C.
	//

	// Create control tree object for lowest-level block-panel kernel.
	gemm_cntl_bp_ke
	=
	bli_gemm_cntl_obj_create( BLIS_UNB_OPT,
	                          BLIS_VARIANT2,
	                          NULL,
	                          gemm_ukrs,
	                          NULL, NULL, NULL,
	                          NULL, NULL, NULL );

	// Create control tree object for outer panel (to block-panel)
	// problem.
	gemm_cntl_op_bp
	=
	bli_gemm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT1,
	                          gemm_mc,
	                          NULL,
	                          NULL,
	                          gemm_packa_cntl,
	                          gemm_packb_cntl,
	                          NULL,
	                          gemm_cntl_bp_ke,
	                          NULL );

	// Create control tree object for general problem via multiple
	// rank-k (outer panel) updates.
	gemm_cntl_mm_op
	=
	bli_gemm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT3,
	                          gemm_kc,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          gemm_cntl_op_bp,
	                          NULL );

	// Create control tree object for very large problem via multiple
	// general problems.
	gemm_cntl_vl_mm
	=
	bli_gemm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT2,
	                          gemm_nc,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          gemm_cntl_mm_op,
	                          NULL );

	// Alias the "master" gemm control tree to a shorter name.
	gemm_cntl = gemm_cntl_vl_mm;
}

void bli_gemm_cntl_finalize()
{
	bli_blksz_obj_free( gemm_mc );
	bli_blksz_obj_free( gemm_nc );
	bli_blksz_obj_free( gemm_kc );
	bli_blksz_obj_free( gemm_mr );
	bli_blksz_obj_free( gemm_nr );
	bli_blksz_obj_free( gemm_kr );

	bli_func_obj_free( gemm_ukrs );
	bli_func_obj_free( gemm_ref_ukrs );

	bli_cntl_obj_free( gemm_packa_cntl );
	bli_cntl_obj_free( gemm_packb_cntl );

	bli_cntl_obj_free( gemm_cntl_bp_ke );
	bli_cntl_obj_free( gemm_cntl_op_bp );
	bli_cntl_obj_free( gemm_cntl_mm_op );
	bli_cntl_obj_free( gemm_cntl_vl_mm );
}

gemm_t* bli_gemm_cntl_obj_create( impl_t     impl_type,
                                  varnum_t   var_num,
                                  blksz_t*   b,
                                  func_t*    gemm_ukrs_,
                                  scalm_t*   sub_scalm,
                                  packm_t*   sub_packm_a,
                                  packm_t*   sub_packm_b,
                                  packm_t*   sub_packm_c,
                                  gemm_t*    sub_gemm,
                                  unpackm_t* sub_unpackm_c )
{
	gemm_t* cntl;

	cntl = ( gemm_t* ) bli_malloc( sizeof(gemm_t) );

	cntl->impl_type     = impl_type;
	cntl->var_num       = var_num;
	cntl->b             = b;
	cntl->gemm_ukrs     = gemm_ukrs_; // avoid name conflict with global symbol
	cntl->sub_scalm     = sub_scalm;
	cntl->sub_packm_a   = sub_packm_a;
	cntl->sub_packm_b   = sub_packm_b;
	cntl->sub_packm_c   = sub_packm_c;
	cntl->sub_gemm      = sub_gemm;
	cntl->sub_unpackm_c = sub_unpackm_c;

	return cntl;
}

func_t* bli_gemm_cntl_ukrs( gemm_t* cntl )
{
	dim_t max_depth = 10;
	dim_t i;

	for ( i = 0; ; ++i )
	{
		// If the gemm sub-tree is NULL, we are at the leaf.
		if ( cntl_sub_gemm( cntl ) == NULL ) break;

		// If the above branch was not taken, we can assume the gemm
		// sub-tree is valid. Here, we step down into that sub-tree.
		cntl = cntl_sub_gemm( cntl );

		// Safeguard against infinite loops due to bad control tree
		// configuration.
		if ( i == max_depth ) bli_abort();
	}

	return cntl_gemm_ukrs( cntl );
}

