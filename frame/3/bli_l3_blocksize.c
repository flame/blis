/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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

dim_t bli_l3_determine_kc
      (
        const obj_t*  a,
        const obj_t*  b,
              dim_t*  b_alg,
              dim_t*  b_max,
        const cntx_t* cntx,
        const cntl_t* cntl
      )
{
	const opid_t family = bli_cntl_family( cntl );
	const num_t  dt     = bli_obj_exec_dt( a );
	      dim_t  mnr    = 1;

	// Nudge the default and maximum kc blocksizes up to the nearest
	// multiple of MR if A is Hermitian, symmetric, or triangular or
	// NR if B is Hermitian, symmetric, or triangular. If neither case
	// applies, then we leave the blocksizes unchanged. For trsm we
	// always use MR (rather than sometimes using NR) because even
	// when the triangle is on the right, packing of that matrix uses
	// MR, since only left-side trsm micro-kernels are supported.
	if ( !bli_obj_root_is_general( a ) || family == BLIS_TRSM )
	{
		mnr = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx );
	}
	else if ( !bli_obj_root_is_general( b ) )
	{
		mnr = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx );
	}

	*b_alg = bli_align_dim_to_mult( *b_alg, mnr );
	*b_max = bli_align_dim_to_mult( *b_max, mnr );
}

GENFRONT( trmm_determine_kc_f, f )
GENFRONT( trmm_determine_kc_b, b )

// -----------------------------------------------------------------------------

#undef  GENFRONT
#define GENFRONT( opname, chdir ) \
\
dim_t PASTEMAC0(opname) \
      ( \
        dim_t   i, \
        dim_t   dim, \
        obj_t*  a, \
        obj_t*  b, \
        bszid_t bszid, \
        cntx_t* cntx  \
      ) \
{ \
	num_t    dt; \
	blksz_t* bsize; \
	dim_t    mnr; \
	dim_t    b_alg = 0, b_max; \
	dim_t    b_use; \
 \
	/* bli_*_determine_kc_f():

	bli_l3_adjust_kc( a, b, &b_alg, &b_max, cntx, cntl );

	   We assume that this function is being called from an algorithm that
	   is moving "backward" (ie: bottom to top, right to left, bottom-right
	   to top-left). */ \
\
	/* Extract the execution datatype and use it to query the corresponding
	   blocksize and blocksize maximum values from the blksz_t object. */ \
	dt    = bli_obj_exec_dt( a ); \
	bsize = bli_cntx_get_trsm_blksz( bszid, cntx ); \
	b_alg = bli_blksz_get_def( dt, bsize ); \
	b_max = bli_blksz_get_max( dt, bsize ); \
\
	/* If b_alg != 0, this means that trsm blocksizes are set and
	 * we continue with trsm-specific blocksizes.
	 * Else, we query L3 blocksizes and use them for TRSM execution. */ \
\
	if( b_alg == 0) \
	{ \
		bsize = bli_cntx_get_blksz( bszid, cntx ); \
		b_alg = bli_blksz_get_def( dt, bsize ); \
		b_max = bli_blksz_get_max( dt, bsize ); \
	} \
	/* Nudge the default and maximum kc blocksizes up to the nearest
	   multiple of MR. We always use MR (rather than sometimes using NR)
	   because even when the triangle is on the right, packing of that
	   matrix uses MR, since only left-side trsm micro-kernels are
	   supported. */ \
	mnr   = bli_cntx_get_trsm_blksz_def_dt( dt, BLIS_MR, cntx ); \
	if( mnr == 0 ) \
	{ \
		mnr   = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx ); \
	} \
	b_alg = bli_align_dim_to_mult( b_alg, mnr ); \
	b_max = bli_align_dim_to_mult( b_max, mnr ); \
\
	/* Call the bli_determine_blocksize_[fb]_sub() helper routine defined
	   in bli_blksz.c */ \
	b_use = PASTEMAC2(determine_blocksize_,chdir,_sub)( i, dim, b_alg, b_max ); \
\
	return b_use; \
\
}

