/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022, The University of Texas at Austin
   Copyright (C) 2022, Oracle Labs, Oracle Corporation

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
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

#ifdef BLIS_ENABLE_LEVEL4

static void_fp unb_vars[2] = { bli_chol_l_opt_var2, bli_chol_u_opt_var2 };
static void_fp blk_vars[2] = { bli_chol_l_blk_var3, bli_chol_u_blk_var3 };

l4_cntl_t* bli_chol_cntl_create
     (
       uplo_t  uplo,
       pool_t* pool
     )
{
	dim_t uplo_i;

	if ( bli_is_lower( uplo ) ) uplo_i = 0;
	else                        uplo_i = 1;

	void_fp unb_fp = unb_vars[uplo_i];
	void_fp blk_fp = blk_vars[uplo_i];

	l4_cntl_t* chol_leaf = bli_l4_cntl_create_node
	(
	  pool,
	  BLIS_NO_PART,
	  1, 1,
	  2,
	  unb_fp,
	  NULL
	);

	l4_cntl_t* chol_blk_inner = bli_l4_cntl_create_node
	(
	  pool,
	  BLIS_KC,
	  1, 1,
	  1,
	  blk_fp,
	  chol_leaf
	);

	l4_cntl_t* chol_blk_outer = bli_l4_cntl_create_node
	(
	  pool,
	  BLIS_KC,
	  4, 1,
	  0,
	  blk_fp,
	  chol_blk_inner
	);

	return chol_blk_outer;
}

#endif
