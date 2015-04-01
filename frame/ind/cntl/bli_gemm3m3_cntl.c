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

blksz_t*          gemm3m3_mc;
blksz_t*          gemm3m3_nc;
blksz_t*          gemm3m3_kc;
blksz_t*          gemm3m3_mr;
blksz_t*          gemm3m3_nr;
blksz_t*          gemm3m3_kr;

func_t*           gemm3m3_ukrs;

packm_t*          gemm3m3_packb_cntl;

gemm_t*           gemm3m3_cntl_bp_ke;
gemm_t*           gemm3m3_cntl_op_bp;
gemm_t*           gemm3m3_cntl_mm_op;
gemm_t*           gemm3m3_cntl_vl_mm;

gemm_t*           gemm3m3_cntl;


void bli_gemm3m3_cntl_init()
{
	// Create blocksize objects for each dimension.
	// NOTE: the complex blocksizes for 3m3 are generally equal to their
	// corresponding real domain counterparts. However, we want to promote
	// similar cache footprints for the micro-panels of A and B (when
	// compared to executing in the real domain), and since the complex
	// micro-panels are three times as "fat" (due to storing real, imaginary
	// and real+imaginary parts), we reduce KC by a factor of 2 to
	// compensate. Ideally, we would reduce by a factor of 3, but that
	// could get messy vis-a-vis keeping KC a multiple of the register
	// blocksizes.
	gemm3m3_mc
	=
	bli_blksz_obj_create( 0,                 0,
	                      0,                 0,
	                      BLIS_DEFAULT_MC_S, BLIS_MAXIMUM_MC_S,
	                      BLIS_DEFAULT_MC_D, BLIS_MAXIMUM_MC_D );
	gemm3m3_nc
	=
	bli_blksz_obj_create( 0,                   0,
	                      0,                   0,
	                      BLIS_DEFAULT_NC_S/3, BLIS_MAXIMUM_NC_S/3,
	                      BLIS_DEFAULT_NC_D/3, BLIS_MAXIMUM_NC_D/3 );
	gemm3m3_kc
	=
	bli_blksz_obj_create( 0,                 0,
	                      0,                 0,
	                      BLIS_DEFAULT_KC_S, BLIS_MAXIMUM_KC_S,
	                      BLIS_DEFAULT_KC_D, BLIS_MAXIMUM_KC_D );
	gemm3m3_mr
	=
	bli_blksz_obj_create( 0,                 0,
	                      0,                 0,
	                      BLIS_DEFAULT_MR_S, BLIS_PACKDIM_MR_S,
	                      BLIS_DEFAULT_MR_D, BLIS_PACKDIM_MR_D );
	gemm3m3_nr
	=
	bli_blksz_obj_create( 0,                 0,
	                      0,                 0,
	                      BLIS_DEFAULT_NR_S, BLIS_PACKDIM_NR_S,
	                      BLIS_DEFAULT_NR_D, BLIS_PACKDIM_NR_D );
	gemm3m3_kr
	=
	bli_blksz_obj_create( 0,                 0,
	                      0,                 0,
	                      BLIS_DEFAULT_KR_S, BLIS_PACKDIM_KR_S,
	                      BLIS_DEFAULT_KR_D, BLIS_PACKDIM_KR_D );


	// Attach the register blksz_t objects as blocksize multiples to the cache
	// blksz_t objects.
	bli_blksz_obj_attach_mult_to( gemm3m3_mr, gemm3m3_mc );
	bli_blksz_obj_attach_mult_to( gemm3m3_nr, gemm3m3_nc );
	bli_blksz_obj_attach_mult_to( gemm3m3_kr, gemm3m3_kc );


	// The cache blocksizes that were scaled above need to be rounded down
	// to their respective nearest register blocksize multiples. Note that
	// this can only happen after the appropriate register blocksize is
	// actually attached as a multiple.
	bli_blksz_reduce_to_mult( gemm3m3_nc );


	// Attach the mr and nr blksz_t objects to each cache blksz_t object.
	// The primary example of why this is needed relates to nudging kc.
	// In hemm, symm, trmm, or trmm3, we need to know both mr and nr,
	// since the multiple we target in nudging depends on whether the
	// structured matrix is on the left or the right.
	bli_blksz_obj_attach_mr_nr_to( gemm3m3_mr, gemm3m3_nr, gemm3m3_mc );
	bli_blksz_obj_attach_mr_nr_to( gemm3m3_mr, gemm3m3_nr, gemm3m3_nc );
	bli_blksz_obj_attach_mr_nr_to( gemm3m3_mr, gemm3m3_nr, gemm3m3_kc );


	// Create function pointer object for each datatype-specific gemm
	// micro-kernel.
	gemm3m3_ukrs
	=
	bli_func_obj_create(
	    NULL,                  FALSE,
	    NULL,                  FALSE,
	    BLIS_CGEMM3M3_UKERNEL, BLIS_CGEMM3M3_UKERNEL_PREFERS_CONTIG_ROWS,
	    BLIS_ZGEMM3M3_UKERNEL, BLIS_ZGEMM3M3_UKERNEL_PREFERS_CONTIG_ROWS );


	// Create control tree objects for packm operations.
	gemm3m3_packb_cntl
	=
	bli_packm_cntl_obj_create( BLIS_BLOCKED,
	                           BLIS_VARIANT2,
	                           gemm3m3_kr,
	                           gemm3m3_nr,
	                           FALSE, // do NOT invert diagonal
	                           FALSE, // reverse iteration if upper?
	                           FALSE, // reverse iteration if lower?
	                           BLIS_PACKED_COL_PANELS_3MS,
	                           BLIS_BUFFER_FOR_B_PANEL );


	//
	// Create a control tree for packing A and B, and streaming C.
	//

	// Create control tree object for lowest-level block-panel kernel.
	gemm3m3_cntl_bp_ke
	=
	bli_gemm_cntl_obj_create( BLIS_UNB_OPT,
	                          BLIS_VARIANT2,
	                          NULL,
	                          gemm3m3_ukrs,
	                          NULL, NULL, NULL,
	                          NULL, NULL, NULL );

	// Create control tree object for outer panel (to block-panel)
	// problem.
	gemm3m3_cntl_op_bp
	=
	bli_gemm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT4,
	                          gemm3m3_mc,
	                          NULL,
	                          NULL,
	                          NULL, // packm cntl nodes accessed directly from blk_var4
	                          gemm3m3_packb_cntl,
	                          NULL,
	                          gemm3m3_cntl_bp_ke,
	                          NULL );

	// Create control tree object for general problem via multiple
	// rank-k (outer panel) updates.
	gemm3m3_cntl_mm_op
	=
	bli_gemm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT3,
	                          gemm3m3_kc,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          gemm3m3_cntl_op_bp,
	                          NULL );

	// Create control tree object for very large problem via multiple
	// general problems.
	gemm3m3_cntl_vl_mm
	=
	bli_gemm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT2,
	                          gemm3m3_nc,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          gemm3m3_cntl_mm_op,
	                          NULL );

	// Alias the "master" gemm control tree to a shorter name.
	gemm3m3_cntl = gemm3m3_cntl_vl_mm;

}

void bli_gemm3m3_cntl_finalize()
{
	bli_blksz_obj_free( gemm3m3_mc );
	bli_blksz_obj_free( gemm3m3_nc );
	bli_blksz_obj_free( gemm3m3_kc );
	bli_blksz_obj_free( gemm3m3_mr );
	bli_blksz_obj_free( gemm3m3_nr );
	bli_blksz_obj_free( gemm3m3_kr );

	bli_func_obj_free( gemm3m3_ukrs );

	bli_cntl_obj_free( gemm3m3_packb_cntl );

	bli_cntl_obj_free( gemm3m3_cntl_bp_ke );
	bli_cntl_obj_free( gemm3m3_cntl_op_bp );
	bli_cntl_obj_free( gemm3m3_cntl_mm_op );
	bli_cntl_obj_free( gemm3m3_cntl_vl_mm );

}

