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



void bli_sgemm_opt_mxn
     (
       dim_t               k,
       float*     restrict alpha,
       float*     restrict a1,
       float*     restrict b1,
       float*     restrict beta,
       float*     restrict c11, inc_t rs_c, inc_t cs_c,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	/* Just call the reference implementation. */
	BLIS_SGEMM_UKERNEL_REF
	(
	  k,
	  alpha,
	  a1,
	  b1,
	  beta,
	  c11, rs_c, cs_c,
	  data,
	  cntx
	);
}



void bli_dgemm_opt_mxn
     (
       dim_t               k,
       double*    restrict alpha,
       double*    restrict a1,
       double*    restrict b1,
       double*    restrict beta,
       double*    restrict c11, inc_t rs_c, inc_t cs_c,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
/*
  Template gemm micro-kernel implementation

  This function contains a template implementation for a double-precision
  real micro-kernel, coded in C, which can serve as the starting point for
  one to write an optimized micro-kernel on an arbitrary architecture. (We
  show a template implementation for only double-precision real because
  the templates for the other three floating-point types would be nearly
  identical.)

  This micro-kernel performs a matrix-matrix multiplication of the form:

    C11 := beta * C11 + alpha * A1 * B1

  where A1 is MR x k, B1 is k x NR, C11 is MR x NR, and alpha and beta are
  scalars.

  For more info, please refer to the BLIS website's wiki on kernels:

    https://github.com/flame/blis/wiki/KernelsHowTo

  and/or contact the blis-devel mailing list.

  -FGVZ
*/
	const num_t        dt     = BLIS_DOUBLE;

	const dim_t        mr     = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx );
	const dim_t        nr     = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx );

	const inc_t        packmr = bli_cntx_get_blksz_max_dt( dt, BLIS_MR, cntx );
	const inc_t        packnr = bli_cntx_get_blksz_max_dt( dt, BLIS_NR, cntx );

	const inc_t        cs_a   = packmr;
	const inc_t        rs_b   = packnr;

	const inc_t        rs_ab  = 1;
	const inc_t        cs_ab  = mr;

	dim_t              l, j, i;

	double             ab[ bli_dmr *
	                       bli_dnr ];
	double*            abij;
	double             ai, bj;


	/* Initialize the accumulator elements in ab to zero. */
	for ( i = 0; i < mr * nr; ++i )
	{
		bli_dset0s( *(ab + i) );
	}

	/* Perform a series of k rank-1 updates into ab. */
	for ( l = 0; l < k; ++l )
	{
		abij = ab;

		/* In an optimized implementation, these two loops over MR and NR
		   are typically fully unrolled. */
		for ( j = 0; j < nr; ++j )
		{
			bj = *(b1 + j);

			for ( i = 0; i < mr; ++i )
			{
				ai = *(a1 + i);

				bli_ddots( ai, bj, *abij );

				abij += rs_ab;
			}
		}

		a1 += cs_a;
		b1 += rs_b;
	}

	/* Scale each element of ab by alpha. */
	for ( i = 0; i < mr * nr; ++i )
	{
		bli_dscals( *alpha, *(ab + i) );
	}

	/* If beta is zero, overwrite c11 with the scaled result in ab.
	   Otherwise, scale c11 by beta and then add the scaled result in
	   ab. */
	if ( bli_deq0( *beta ) )
	{
		/* c11 := ab */
		bli_dcopys_mxn( mr,
		                nr,
		                ab,  rs_ab, cs_ab,
		                c11, rs_c,  cs_c );
	}
	else
	{
		/* c11 := beta * c11 + ab */
		bli_dxpbys_mxn( mr,
		                nr,
		                ab,  rs_ab, cs_ab,
		                beta,
		                c11, rs_c,  cs_c );
	}
}



void bli_cgemm_opt_mxn(
                        dim_t              k,
                        scomplex* restrict alpha,
                        scomplex* restrict a1,
                        scomplex* restrict b1,
                        scomplex* restrict beta,
                        scomplex* restrict c11, inc_t rs_c, inc_t cs_c,
                        auxinfo_t*         data
                      )
     (
       dim_t               k,
       scomplex*  restrict alpha,
       scomplex*  restrict a1,
       scomplex*  restrict b1,
       scomplex*  restrict beta,
       scomplex*  restrict c11, inc_t rs_c, inc_t cs_c,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	/* Just call the reference implementation. */
	BLIS_CGEMM_UKERNEL_REF
	(
	  k,
	  alpha,
	  a1,
	  b1,
	  beta,
	  c11, rs_c, cs_c,
	  data,
	  cntx
	);
}



void bli_zgemm_opt_mxn
     (
       dim_t               k,
       dcomplex*  restrict alpha,
       dcomplex*  restrict a1,
       dcomplex*  restrict b1,
       dcomplex*  restrict beta,
       dcomplex*  restrict c11, inc_t rs_c, inc_t cs_c,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	/* Just call the reference implementation. */
	BLIS_ZGEMM_UKERNEL_REF
	(
	  k,
	  alpha,
	  a1,
	  b1,
	  beta,
	  c11, rs_c, cs_c,
	  data,
	  cntx
	);
}

