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

extern blksz_t*   gemm3m_mc;
extern blksz_t*   gemm3m_nc;
extern blksz_t*   gemm3m_kc;
extern blksz_t*   gemm3m_mr;
extern blksz_t*   gemm3m_nr;
extern blksz_t*   gemm3m_kr;

extern func_t*    gemm3m_ukrs;

packm_t*          herk3m_packa_cntl;
packm_t*          herk3m_packb_cntl;

herk_t*           herk3m_cntl_bp_ke;
herk_t*           herk3m_cntl_op_bp;
herk_t*           herk3m_cntl_mm_op;
herk_t*           herk3m_cntl_vl_mm;

herk_t*           herk3m_cntl;


void bli_herk3m_cntl_init()
{
	// Create control tree objects for packm operations.
	herk3m_packa_cntl
	=
	bli_packm_cntl_obj_create( BLIS_BLOCKED,
	                           BLIS_VARIANT2,
	                           gemm3m_mr,
	                           gemm3m_kr,
	                           FALSE, // do NOT invert diagonal
	                           FALSE, // reverse iteration if upper?
	                           FALSE, // reverse iteration if lower?
	                           BLIS_PACKED_ROW_PANELS_3M,
	                           BLIS_BUFFER_FOR_A_BLOCK );

	herk3m_packb_cntl
	=
	bli_packm_cntl_obj_create( BLIS_BLOCKED,
	                           BLIS_VARIANT2,
	                           gemm3m_kr,
	                           gemm3m_nr,
	                           FALSE, // do NOT invert diagonal
	                           FALSE, // reverse iteration if upper?
	                           FALSE, // reverse iteration if lower?
	                           BLIS_PACKED_COL_PANELS_3M,
	                           BLIS_BUFFER_FOR_B_PANEL );


	// Create control tree object for lowest-level block-panel kernel.
	herk3m_cntl_bp_ke
	=
	bli_herk_cntl_obj_create( BLIS_UNB_OPT,
	                          BLIS_VARIANT2,
	                          NULL,
	                          gemm3m_ukrs,
	                          NULL, NULL, NULL,
	                          NULL, NULL, NULL );

	// Create control tree object for outer panel (to block-panel)
	// problem.
	herk3m_cntl_op_bp
	=
	bli_herk_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT1,
	                          gemm3m_mc,
	                          gemm3m_ukrs,
	                          NULL,
	                          herk3m_packa_cntl,
	                          herk3m_packb_cntl,
	                          NULL,
	                          herk3m_cntl_bp_ke,
	                          NULL );

	// Create control tree object for general problem via multiple
	// rank-k (outer panel) updates.
	herk3m_cntl_mm_op
	=
	bli_herk_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT3,
	                          gemm3m_kc,
	                          gemm3m_ukrs,
	                          NULL,
	                          NULL, 
	                          NULL,
	                          NULL,
	                          herk3m_cntl_op_bp,
	                          NULL );

	// Create control tree object for very large problem via multiple
	// general problems.
	herk3m_cntl_vl_mm
	=
	bli_herk_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT2,
	                          gemm3m_nc,
	                          gemm3m_ukrs,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          herk3m_cntl_mm_op,
	                          NULL );

	// Alias the "master" herk control tree to a shorter name.
	herk3m_cntl = herk3m_cntl_vl_mm;
}

void bli_herk3m_cntl_finalize()
{
	bli_cntl_obj_free( herk3m_packa_cntl );
	bli_cntl_obj_free( herk3m_packb_cntl );

	bli_cntl_obj_free( herk3m_cntl_bp_ke );
	bli_cntl_obj_free( herk3m_cntl_op_bp );
	bli_cntl_obj_free( herk3m_cntl_mm_op );
	bli_cntl_obj_free( herk3m_cntl_vl_mm );
}

