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

blksz_t*          gemm3mh_mc;
blksz_t*          gemm3mh_nc;
blksz_t*          gemm3mh_kc;
blksz_t*          gemm3mh_mr;
blksz_t*          gemm3mh_nr;
blksz_t*          gemm3mh_kr;

func_t*           gemm3mh_ukrs;

packm_t*          gemm3mh_packa_cntl_ro;
packm_t*          gemm3mh_packb_cntl_ro;
packm_t*          gemm3mh_packa_cntl_io;
packm_t*          gemm3mh_packb_cntl_io;
packm_t*          gemm3mh_packa_cntl_rpi;
packm_t*          gemm3mh_packb_cntl_rpi;

gemm_t*           gemm3mh_cntl_bp_ke;
gemm_t*           gemm3mh_cntl_op_bp_ro;
gemm_t*           gemm3mh_cntl_mm_op_ro;
gemm_t*           gemm3mh_cntl_vl_mm_ro;
gemm_t*           gemm3mh_cntl_op_bp_io;
gemm_t*           gemm3mh_cntl_mm_op_io;
gemm_t*           gemm3mh_cntl_vl_mm_io;
gemm_t*           gemm3mh_cntl_op_bp_rpi;
gemm_t*           gemm3mh_cntl_mm_op_rpi;
gemm_t*           gemm3mh_cntl_vl_mm_rpi;

gemm_t*           gemm3mh_cntl_ro;
gemm_t*           gemm3mh_cntl_io;
gemm_t*           gemm3mh_cntl_rpi;


void bli_gemm3mh_cntl_init()
{
	// Create blocksize objects for each dimension.
	// NOTE: the complex blocksizes for 3mh are equal to their
	// corresponding real domain counterparts.
	gemm3mh_mc
	=
	bli_blksz_obj_create( 0,                 0,
	                      0,                 0,
	                      BLIS_DEFAULT_MC_S, BLIS_MAXIMUM_MC_S,
	                      BLIS_DEFAULT_MC_D, BLIS_MAXIMUM_MC_D );
	gemm3mh_nc
	=
	bli_blksz_obj_create( 0,                 0,
	                      0,                 0,
	                      BLIS_DEFAULT_NC_S, BLIS_MAXIMUM_NC_S,
	                      BLIS_DEFAULT_NC_D, BLIS_MAXIMUM_NC_D );
	gemm3mh_kc
	=
	bli_blksz_obj_create( 0,                 0,
	                      0,                 0,
	                      BLIS_DEFAULT_KC_S, BLIS_MAXIMUM_KC_S,
	                      BLIS_DEFAULT_KC_D, BLIS_MAXIMUM_KC_D );
	gemm3mh_mr
	=
	bli_blksz_obj_create( 0,                 0,
	                      0,                 0,
	                      BLIS_DEFAULT_MR_S, BLIS_PACKDIM_MR_S,
	                      BLIS_DEFAULT_MR_D, BLIS_PACKDIM_MR_D );
	gemm3mh_nr
	=
	bli_blksz_obj_create( 0,                 0,
	                      0,                 0,
	                      BLIS_DEFAULT_NR_S, BLIS_PACKDIM_NR_S,
	                      BLIS_DEFAULT_NR_D, BLIS_PACKDIM_NR_D );
	gemm3mh_kr
	=
	bli_blksz_obj_create( 0,                 0,
	                      0,                 0,
	                      BLIS_DEFAULT_KR_S, BLIS_PACKDIM_KR_S,
	                      BLIS_DEFAULT_KR_D, BLIS_PACKDIM_KR_D );


	// Attach the register blksz_t objects as blocksize multiples to the cache
	// blksz_t objects.
	bli_blksz_obj_attach_mult_to( gemm3mh_mr, gemm3mh_mc );
	bli_blksz_obj_attach_mult_to( gemm3mh_nr, gemm3mh_nc );
	bli_blksz_obj_attach_mult_to( gemm3mh_kr, gemm3mh_kc );


	// Attach the mr and nr blksz_t objects to each cache blksz_t object.
	// The primary example of why this is needed relates to nudging kc.
	// In hemm, symm, trmm, or trmm3, we need to know both mr and nr,
	// since the multiple we target in nudging depends on whether the
	// structured matrix is on the left or the right.
	bli_blksz_obj_attach_mr_nr_to( gemm3mh_mr, gemm3mh_nr, gemm3mh_mc );
	bli_blksz_obj_attach_mr_nr_to( gemm3mh_mr, gemm3mh_nr, gemm3mh_nc );
	bli_blksz_obj_attach_mr_nr_to( gemm3mh_mr, gemm3mh_nr, gemm3mh_kc );


	// Create function pointer object for each datatype-specific gemm
	// micro-kernel.
	gemm3mh_ukrs
	=
	bli_func_obj_create(
	    NULL,                  FALSE,
	    NULL,                  FALSE,
	    BLIS_CGEMM3MH_UKERNEL, BLIS_CGEMM3MH_UKERNEL_PREFERS_CONTIG_ROWS,
	    BLIS_ZGEMM3MH_UKERNEL, BLIS_ZGEMM3MH_UKERNEL_PREFERS_CONTIG_ROWS );


	// Create control tree objects for packm operations (real only).
	gemm3mh_packa_cntl_ro
	=
	bli_packm_cntl_obj_create( BLIS_BLOCKED,
	                           BLIS_VARIANT2,
	                           gemm3mh_mr,
	                           gemm3mh_kr,
	                           FALSE, // do NOT invert diagonal
	                           FALSE, // reverse iteration if upper?
	                           FALSE, // reverse iteration if lower?
	                           BLIS_PACKED_ROW_PANELS_RO,
	                           BLIS_BUFFER_FOR_A_BLOCK );

	gemm3mh_packb_cntl_ro
	=
	bli_packm_cntl_obj_create( BLIS_BLOCKED,
	                           BLIS_VARIANT2,
	                           gemm3mh_kr,
	                           gemm3mh_nr,
	                           FALSE, // do NOT invert diagonal
	                           FALSE, // reverse iteration if upper?
	                           FALSE, // reverse iteration if lower?
	                           BLIS_PACKED_COL_PANELS_RO,
	                           BLIS_BUFFER_FOR_B_PANEL );

	// Create control tree objects for packm operations (imag only).
	gemm3mh_packa_cntl_io
	=
	bli_packm_cntl_obj_create( BLIS_BLOCKED,
	                           BLIS_VARIANT2,
	                           gemm3mh_mr,
	                           gemm3mh_kr,
	                           FALSE, // do NOT invert diagonal
	                           FALSE, // reverse iteration if upper?
	                           FALSE, // reverse iteration if lower?
	                           BLIS_PACKED_ROW_PANELS_IO,
	                           BLIS_BUFFER_FOR_A_BLOCK );

	gemm3mh_packb_cntl_io
	=
	bli_packm_cntl_obj_create( BLIS_BLOCKED,
	                           BLIS_VARIANT2,
	                           gemm3mh_kr,
	                           gemm3mh_nr,
	                           FALSE, // do NOT invert diagonal
	                           FALSE, // reverse iteration if upper?
	                           FALSE, // reverse iteration if lower?
	                           BLIS_PACKED_COL_PANELS_IO,
	                           BLIS_BUFFER_FOR_B_PANEL );

	// Create control tree objects for packm operations (real+imag).
	gemm3mh_packa_cntl_rpi
	=
	bli_packm_cntl_obj_create( BLIS_BLOCKED,
	                           BLIS_VARIANT2,
	                           gemm3mh_mr,
	                           gemm3mh_kr,
	                           FALSE, // do NOT invert diagonal
	                           FALSE, // reverse iteration if upper?
	                           FALSE, // reverse iteration if lower?
	                           BLIS_PACKED_ROW_PANELS_RPI,
	                           BLIS_BUFFER_FOR_A_BLOCK );

	gemm3mh_packb_cntl_rpi
	=
	bli_packm_cntl_obj_create( BLIS_BLOCKED,
	                           BLIS_VARIANT2,
	                           gemm3mh_kr,
	                           gemm3mh_nr,
	                           FALSE, // do NOT invert diagonal
	                           FALSE, // reverse iteration if upper?
	                           FALSE, // reverse iteration if lower?
	                           BLIS_PACKED_COL_PANELS_RPI,
	                           BLIS_BUFFER_FOR_B_PANEL );


	// Create control tree object for lowest-level block-panel kernel.
	gemm3mh_cntl_bp_ke
	=
	bli_gemm_cntl_obj_create( BLIS_UNB_OPT,
	                          BLIS_VARIANT2,
	                          NULL,
	                          gemm3mh_ukrs,
	                          NULL, NULL, NULL,
	                          NULL, NULL, NULL );

	//
	// Create control tree for A.real * B.real.
	//

	// Create control tree object for outer panel (to block-panel)
	// problem. (real x real)
	gemm3mh_cntl_op_bp_ro
	=
	bli_gemm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT1,
	                          gemm3mh_mc,
	                          NULL,
	                          NULL,
	                          gemm3mh_packa_cntl_ro,
	                          gemm3mh_packb_cntl_ro,
	                          NULL,
	                          gemm3mh_cntl_bp_ke,
	                          NULL );

	// Create control tree object for general problem via multiple
	// rank-k (outer panel) updates. (real x real)
	gemm3mh_cntl_mm_op_ro
	=
	bli_gemm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT3,
	                          gemm3mh_kc,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          gemm3mh_cntl_op_bp_ro,
	                          NULL );

	// Create control tree object for very large problem via multiple
	// general problems. (real x real)
	gemm3mh_cntl_vl_mm_ro
	=
	bli_gemm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT2,
	                          gemm3mh_nc,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          gemm3mh_cntl_mm_op_ro,
	                          NULL );

	//
	// Create control tree for A.imag * B.imag.
	//

	// Create control tree object for outer panel (to block-panel)
	// problem. (imag x imag)
	gemm3mh_cntl_op_bp_io
	=
	bli_gemm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT1,
	                          gemm3mh_mc,
	                          NULL,
	                          NULL,
	                          gemm3mh_packa_cntl_io,
	                          gemm3mh_packb_cntl_io,
	                          NULL,
	                          gemm3mh_cntl_bp_ke,
	                          NULL );

	// Create control tree object for general problem via multiple
	// rank-k (outer panel) updates. (imag x imag)
	gemm3mh_cntl_mm_op_io
	=
	bli_gemm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT3,
	                          gemm3mh_kc,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          gemm3mh_cntl_op_bp_io,
	                          NULL );

	// Create control tree object for very large problem via multiple
	// general problems. (imag x imag)
	gemm3mh_cntl_vl_mm_io
	=
	bli_gemm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT2,
	                          gemm3mh_nc,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          gemm3mh_cntl_mm_op_io,
	                          NULL );

	//
	// Create control tree for (A.real + A.imag) * (B.real + B.imag).
	//

	// Create control tree object for outer panel (to block-panel)
	// problem. (real+imag x real+imag)
	gemm3mh_cntl_op_bp_rpi
	=
	bli_gemm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT1,
	                          gemm3mh_mc,
	                          NULL,
	                          NULL,
	                          gemm3mh_packa_cntl_rpi,
	                          gemm3mh_packb_cntl_rpi,
	                          NULL,
	                          gemm3mh_cntl_bp_ke,
	                          NULL );

	// Create control tree object for general problem via multiple
	// rank-k (outer panel) updates. (real+imag x real+imag)
	gemm3mh_cntl_mm_op_rpi
	=
	bli_gemm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT3,
	                          gemm3mh_kc,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          gemm3mh_cntl_op_bp_rpi,
	                          NULL );

	// Create control tree object for very large problem via multiple
	// general problems. (real+imag x real+imag)
	gemm3mh_cntl_vl_mm_rpi
	=
	bli_gemm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT2,
	                          gemm3mh_nc,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          gemm3mh_cntl_mm_op_rpi,
	                          NULL );

	// Alias the "master" gemm control tree to a shorter name.
	gemm3mh_cntl_ro = gemm3mh_cntl_vl_mm_ro;
	gemm3mh_cntl_io = gemm3mh_cntl_vl_mm_io;
	gemm3mh_cntl_rpi = gemm3mh_cntl_vl_mm_rpi;

}

void bli_gemm3mh_cntl_finalize()
{
	bli_blksz_obj_free( gemm3mh_mc );
	bli_blksz_obj_free( gemm3mh_nc );
	bli_blksz_obj_free( gemm3mh_kc );
	bli_blksz_obj_free( gemm3mh_mr );
	bli_blksz_obj_free( gemm3mh_nr );
	bli_blksz_obj_free( gemm3mh_kr );

	bli_func_obj_free( gemm3mh_ukrs );

	bli_cntl_obj_free( gemm3mh_packa_cntl_ro );
	bli_cntl_obj_free( gemm3mh_packb_cntl_ro );
	bli_cntl_obj_free( gemm3mh_packa_cntl_io );
	bli_cntl_obj_free( gemm3mh_packb_cntl_io );
	bli_cntl_obj_free( gemm3mh_packa_cntl_rpi );
	bli_cntl_obj_free( gemm3mh_packb_cntl_rpi );

	bli_cntl_obj_free( gemm3mh_cntl_bp_ke );
	bli_cntl_obj_free( gemm3mh_cntl_op_bp_ro );
	bli_cntl_obj_free( gemm3mh_cntl_mm_op_ro );
	bli_cntl_obj_free( gemm3mh_cntl_vl_mm_ro );
	bli_cntl_obj_free( gemm3mh_cntl_op_bp_io );
	bli_cntl_obj_free( gemm3mh_cntl_mm_op_io );
	bli_cntl_obj_free( gemm3mh_cntl_vl_mm_io );
	bli_cntl_obj_free( gemm3mh_cntl_op_bp_rpi );
	bli_cntl_obj_free( gemm3mh_cntl_mm_op_rpi );
	bli_cntl_obj_free( gemm3mh_cntl_vl_mm_rpi );

}

