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

static void_fp unb_vars[2] = { bli_trinv_l_opt_var3, bli_trinv_u_opt_var3 };
static void_fp blk_vars[2] = { bli_trinv_l_blk_var3, bli_trinv_u_blk_var3 };

cntl_t* bli_trinv_cntl_create
     (
       uplo_t  uploa,
       diag_t  diaga,
       rntm_t* rntm
     )
{
	dim_t uplo_i;

	if ( bli_is_lower( uploa ) ) uplo_i = 0;
	else                         uplo_i = 1;

	trinv_oft unb_fp = unb_vars[uplo_i];
	trinv_oft blk_fp = blk_vars[uplo_i];

	cntl_t* trinv_xx_leaf = bli_trinv_cntl_create_node
	(
	  rntm,
	  BLIS_NO_PART,
	  1, 1,
	  2,
	  unb_fp,
	  NULL
	);

	cntl_t* trinv_xx_inner = bli_trinv_cntl_create_node
	(
	  rntm,
	  BLIS_KC,
	  1, 1,
	  1,
	  blk_fp,
	  trinv_xx_leaf
	);

	cntl_t *trinv_xx_outer = bli_trinv_cntl_create_node
	(
	  rntm,
	  BLIS_KC,
	  4, 1,
	  0,
	  blk_fp,
	  trinv_xx_inner
	);

	return trinv_xx_outer;
}

// -----------------------------------------------------------------------------

void bli_trinv_cntl_free
     (
       rntm_t*    rntm,
       cntl_t*    cntl,
       thrinfo_t* thread
     )
{
	bli_cntl_free( rntm, cntl, thread );
}

// -----------------------------------------------------------------------------

cntl_t* bli_trinv_cntl_create_node
     (
       rntm_t* rntm,
       bszid_t bszid,
       dim_t   scale_num,
       dim_t   scale_den,
       dim_t   depth,
       void_fp var_func,
       cntl_t* sub_node
     )
{
	trinv_params_t* params = bli_sba_acquire( rntm, sizeof( trinv_params_t ) );

	params->size      = sizeof( trinv_params_t );
	params->scale_num = scale_num;
	params->scale_den = scale_den;
	params->depth     = depth;

	return bli_cntl_create_node
	(
	  rntm,
	  BLIS_NOID,
	  bszid,
	  var_func,
	  params,
	  sub_node
	);
}

#endif
