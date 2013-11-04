/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2013, The University of Texas

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
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



void bli_strsm_l_opt_mxn(
                          float*    restrict a,
                          float*    restrict b,
                          float*    restrict bd,
                          float*    restrict c, inc_t rs_c, inc_t cs_c
                        )
{
	/* Just call the reference implementation. */
	bli_strsm_l_ref_mxn( a,
	                     b,
	                     bd,
	                     c, rs_c, cs_c );
}



void bli_dtrsm_l_opt_mxn(
                          double*   restrict a,
                          double*   restrict b,
                          double*   restrict bd,
                          double*   restrict c, inc_t rs_c, inc_t cs_c
                        )
{
/*
  Template trsm_l micro-kernel implementation

  This function contains a template implementation for a double-precision
  real trsm micro-kernel, coded in C, which can serve as the starting point
  for one to write an optimized micro-kernel on an arbitrary architecture.
  (We show a template implementation for only double-precision real because
  the templates for the other three floating-point types would be nearly
  identical.)

  This micro-kernel performs a triangular solve with NR right-hand sides:

    C11 := inv(A11) * B11

  where A11 is MR x MR and lower triangular, B11 is MR x NR, and C11 is
  MR x NR.

  NOTE: Here, this trsm micro-kernel supports element "duplication", a
  feature that is enabled or disabled in bli_kernel.h. Duplication factors
  are also defined in the aforementioned header. Duplication is NOT
  commonly used and most developers may assume it is disabled.

  Parameters:

  - a11:    The address of A11, which is the MR x MR lower triangular block
            within the packed (column-stored) micro-panel of A. By the time
            this trsm micro-kernel is called, the diagonal of A11 has already
            been inverted and the strictly upper triangle contains zeros.
  - b11:    The address of B11, which is the MR x NR subpartition of the
            current packed (row-stored) micro-panel of B.
  - bd11:   The address of the duplicated copy of B11. If duplication is
            disabled, then bd11 == b11.
  - c11:    The address of C11, which is the MR x NR block of the output
            matrix (ie: the matrix provided by the user to the highest-level
            trsm API call). C11 corresponds to the elements that exist in
            packed form in B11, and is stored according to rs_c and cs_c.
  - rs_c:   The row stride of C11 (ie: the distance to the next row of C11,
            in units of matrix elements).
  - cs_c:   The column stride of C11 (ie: the distance to the next column of
            C11, in units of matrix elements).

  Please see the comments in bli_gemmtrsm_l_opt_mxn.c for a diagram of the
  trsm operation and where it fits in with the preceding gemm subproblem.

  Here are a few things to consider:
  - While all three loops are exposed in this template micro-kernel, all
    three loops typically disappear in an optimized code because they are
    fully unrolled.
  - Note that the diagonal of the triangular matrix A11 contains the INVERSE
    of those elements. This is done during packing so that we can avoid
    expensive division instructions within this micro-kernel.
  - This micro-kernel assumes duplication is NOT enabled. If it IS enabled,
    then the result must be written to three places: the sub-block within the
    duplicated copy of the current micro-panel of B, the sub-block within the
    current packed micro-panel of B, and the sub-block of the output matrix C.
    When duplication is not used, the micro-kernel should update only the
    latter two locations.

  For more info, please refer to the BLIS website and/or contact the
  blis-devel mailing list.

  -FGVZ
*/
	const dim_t        m     = bli_dmr;
	const dim_t        n     = bli_dnr;

	const inc_t        rs_a  = 1;
	const inc_t        cs_a  = bli_dpackmr;

	const inc_t        rs_b  = bli_dpacknr;
	const inc_t        cs_b  = 1;

	dim_t              iter, i, j, l;
	dim_t              n_behind;

	double*   restrict alpha11;
	double*   restrict a10t;
	double*   restrict alpha10;
	double*   restrict X0;
	double*   restrict x1;
	double*   restrict x01;
	double*   restrict chi01;
	double*   restrict chi11;
	double*   restrict gamma11;
	double             rho11;

	for ( iter = 0; iter < m; ++iter )
	{
		i        = iter;
		n_behind = i;
		alpha11  = a + (i  )*rs_a + (i  )*cs_a;
		a10t     = a + (i  )*rs_a + (0  )*cs_a;
		X0       = b + (0  )*rs_b + (0  )*cs_b;
		x1       = b + (i  )*rs_b + (0  )*cs_b;

		/* x1 = x1 - a10t * X0; */
		/* x1 = x1 / alpha11; */
		for ( j = 0; j < n; ++j )
		{
			x01     = X0 + (0  )*rs_b + (j  )*cs_b;
			chi11   = x1 + (0  )*rs_b + (j  )*cs_b;
			gamma11 = c  + (i  )*rs_c + (j  )*cs_c;

			/* chi11 = chi11 - a10t * x01; */
			bli_dset0s( rho11 );
			for ( l = 0; l < n_behind; ++l )
			{
				alpha10 = a10t + (l  )*cs_a;
				chi01   = x01  + (l  )*rs_b;

				bli_daxpys( *alpha10, *chi01, rho11 );
			}
			bli_dsubs( rho11, *chi11 );

			/* chi11 = chi11 / alpha11; */
			/* NOTE: The INVERSE of alpha11 (1.0/alpha11) is stored instead
			   of alpha11, so we can multiply rather than divide. We store
			   the inverse of alpha11 intentionally to avoid expensive
			   division instructions within the micro-kernel. */
			bli_dscals( *alpha11, *chi11 );

			/* Output final result to matrix C. */
			bli_dcopys( *chi11, *gamma11 );
		}
	}
}



void bli_ctrsm_l_opt_mxn(
                          scomplex* restrict a,
                          scomplex* restrict b,
                          scomplex* restrict bd,
                          scomplex* restrict c, inc_t rs_c, inc_t cs_c
                        )
{
	/* Just call the reference implementation. */
	bli_ctrsm_l_ref_mxn( a,
	                     b,
	                     bd,
	                     c, rs_c, cs_c );
}



void bli_ztrsm_l_opt_mxn(
                          dcomplex* restrict a,
                          dcomplex* restrict b,
                          dcomplex* restrict bd,
                          dcomplex* restrict c, inc_t rs_c, inc_t cs_c
                        )
{
	/* Just call the reference implementation. */
	bli_ztrsm_l_ref_mxn( a,
	                     b,
	                     bd,
	                     c, rs_c, cs_c );
}

