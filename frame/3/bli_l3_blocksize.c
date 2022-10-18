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


void bli_l3_adjust_kc
      (
        const obj_t* a,
        const obj_t* b,
              dim_t  mr,
              dim_t  nr,
              dim_t* bsize,
              dim_t* bsize_max,
              opid_t family
      )
{
	if      ( family == BLIS_GEMM )
		bli_gemm_adjust_kc( a, b, mr, nr, bsize, bsize_max );
	else if ( family == BLIS_GEMMT )
		bli_gemmt_adjust_kc( a, b, mr, nr, bsize, bsize_max );
	else if ( family == BLIS_TRMM )
		bli_trmm_adjust_kc( a, b, mr, nr, bsize, bsize_max );
	else if ( family == BLIS_TRSM )
		bli_trsm_adjust_kc( a, b, mr, nr, bsize, bsize_max );
}

// -----------------------------------------------------------------------------

void bli_gemm_adjust_kc
      (
        const obj_t* a,
        const obj_t* b,
              dim_t  mr,
              dim_t  nr,
              dim_t* bsize,
              dim_t* bsize_max
      )
{
	/* Nudge the default and maximum kc blocksizes up to the nearest
	   multiple of MR if A is Hermitian or symmetric, or NR if B is
	   Hermitian or symmetric. If neither case applies, then we leave
	   the blocksizes unchanged. */
	if      ( bli_obj_root_is_herm_or_symm( a ) )
	{
        *bsize     = bli_align_dim_to_mult( *bsize, mr );
        *bsize_max = bli_align_dim_to_mult( *bsize_max, mr );
	}
	else if ( bli_obj_root_is_herm_or_symm( b ) )
	{
        *bsize     = bli_align_dim_to_mult( *bsize, nr );
        *bsize_max = bli_align_dim_to_mult( *bsize_max, nr );
	}
}

// -----------------------------------------------------------------------------

void bli_gemmt_adjust_kc
      (
        const obj_t* a,
        const obj_t* b,
              dim_t  mr,
              dim_t  nr,
              dim_t* bsize,
              dim_t* bsize_max
      )
{
	/* Notice that for gemmt, we do not need to perform any special handling
	   for the default and maximum kc blocksizes vis-a-vis MR or NR. */
}

// -----------------------------------------------------------------------------

void bli_trmm_adjust_kc
      (
        const obj_t* a,
        const obj_t* b,
              dim_t  mr,
              dim_t  nr,
              dim_t* bsize,
              dim_t* bsize_max
      )
{
	/* Nudge the default and maximum kc blocksizes up to the nearest
	   multiple of MR if the triangular matrix is on the left, or NR
	   if the triangular matrix is one the right. */
	dim_t mnr;
	if ( bli_obj_root_is_triangular( a ) )
		mnr = mr;
	else
		mnr = nr;

    *bsize     = bli_align_dim_to_mult( *bsize, mnr );
    *bsize_max = bli_align_dim_to_mult( *bsize_max, mnr );
}

// -----------------------------------------------------------------------------

void bli_trsm_adjust_kc
      (
        const obj_t* a,
        const obj_t* b,
              dim_t  mr,
              dim_t  nr,
              dim_t* bsize,
              dim_t* bsize_max
      )
{
	/* Nudge the default and maximum kc blocksizes up to the nearest
	   multiple of MR. We always use MR (rather than sometimes using NR)
	   because even when the triangle is on the right, packing of that
	   matrix uses MR, since only left-side trsm micro-kernels are
	   supported. */
    *bsize     = bli_align_dim_to_mult( *bsize, mr );
    *bsize_max = bli_align_dim_to_mult( *bsize_max, mr );
}

