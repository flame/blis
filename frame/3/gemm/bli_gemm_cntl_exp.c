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

extern packm_t*   gemm_packa_cntl;
extern packm_t*   gemm_packb_cntl;

gemm_t*           gemm_cntl5;

gemm_t*           gemm_cntl_bp_ke5;
gemm_t*           gemm_cntl_pm_bp;
gemm_t*           gemm_cntl_mm_pm;
gemm_t*           gemm_cntl_vl_mm5;


void bli_gemm_cntl_init_exp()
{

	//
	// Create a control tree for packing A, and streaming B and C.
	//

	gemm_cntl_bp_ke5
	=
	bli_gemm_cntl_obj_create( BLIS_UNB_OPT,
	                          BLIS_VARIANT5,
	                          NULL,
	                          gemm_ukrs,
	                          NULL, NULL, NULL,
	                          NULL, NULL, NULL );
	gemm_cntl_pm_bp
	=
	bli_gemm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT3,
	                          gemm_kc,
	                          NULL,
	                          NULL,
	                          gemm_packa_cntl,
	                          NULL,
	                          NULL,
	                          gemm_cntl_bp_ke5,
	                          NULL );

	gemm_cntl_mm_pm
	=
	bli_gemm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT1,
	                          gemm_mc,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          gemm_cntl_pm_bp,
	                          NULL );

	gemm_cntl_vl_mm5
	=
	bli_gemm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT2,
	                          gemm_nc,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          gemm_cntl_mm_pm,
	                          NULL );

	gemm_cntl5 = gemm_cntl_vl_mm5;
}

void bli_gemm_cntl_finalize_exp()
{
	bli_cntl_obj_free( gemm_cntl_bp_ke5 );
	bli_cntl_obj_free( gemm_cntl_pm_bp );
	bli_cntl_obj_free( gemm_cntl_mm_pm );
	bli_cntl_obj_free( gemm_cntl_vl_mm5 );
}

