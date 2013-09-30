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



void bli_sgemm_opt_mxn(
                        dim_t              k,
                        float*    restrict alpha,
                        float*    restrict a,
                        float*    restrict b,
                        float*    restrict beta,
                        float*    restrict c, inc_t rs_c, inc_t cs_c,
                        float*    restrict a_next,
                        float*    restrict b_next 
                      )
{
	/* Just call the reference implementation. */
	bli_sgemm_ref_mxn( k,
	                   alpha,
	                   a,
	                   b,
	                   beta,
	                   c, rs_c, cs_c,
	                   a_next,
	                   b_next );
}



void bli_dgemm_opt_mxn(
                        dim_t              k,
                        double*   restrict alpha,
                        double*   restrict a,
                        double*   restrict b,
                        double*   restrict beta,
                        double*   restrict c, inc_t rs_c, inc_t cs_c,
                        double*   restrict a_next,
                        double*   restrict b_next 
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

    C := beta * C + alpha * A * B

  where A is MR x k, B is k x NR, C is MR x NR, and alpha and beta are
  scalars.

  Parameters:

  - k:      The number of columns of A and rows of B.
  - alpha:  The address of a scalar to the A*B product.
  - a:      The address of a micro-panel of matrix A of dimension MR x k,
            stored by columns.
  - b:      The address of a micro-panel of matrix B of dimension k x NR,
            stored by rows.
  - beta:   The address of a scalar to the input value of matrix C.
  - c:      The address of a block of matrix C of dimension MR x NR,
            stored according to rs_c and cs_c.
  - rs_c:   The row stride of matrix C (ie: the distance to the next row,
            in units of matrix elements).
  - cs_c:   The column stride of matrix C (ie: the distance to the next
            column, in units of matrix elements).
  - a_next: The address of the micro-panel of A that will be used the next
            time the gemm micro-kernel will be called.
  - b_next: The address of the micro-panel of B that will be used the next
            time the gemm micro-kernel will be called.

  The diagram below shows the packed micro-panel operands and how elements
  of each would be stored when MR == NR == 4. (The hex digits indicate the
  order of the elements in memory.) Note that the storage of C is not shown
  since it is determined by the row and column strides of C.

         c:             a:                         b:                   
         _______        ______________________     _______              
        |       |      |0 4 8 C               |   |0 1 2 3|             
    MR  |       |      |1 5 9 D . . .         |   |4 5 6 7|             
        |       |  +=  |2 6 A E               |   |8 9 A B|             
        |_______|      |3_7_B_F_______________|   |C D E F|             
                                                  |   .   |             
            NR                    k               |   .   |             
                                                  |   .   |             
                                                  |       |             
                                                  |       |             
                                                  |_______|             
                                                                        
                                                      NR                
  Here are a few things to consider:

  - bli_?mr and bli_?nr give the MR and NR register blocksizes for the
    datatype corresponding to the '?' character.
  - bli_?packmr and bli_?packnr are usually equal to bli_?mr and bli_?nr,
    respectively. (They are only not equal if the register blocksize
    extensions are non-zero. See bli_config.h for more details.)
  - You may assume that the addresses a and b are aligned according to
    the alignment value BLIS_CONTIG_STRIDE_ALIGN_SIZE, as defined in
    bli_config.h.
  - Here, we use a local array, ab, as temporary accumulator elements as
    we compute the a*b product. In an optimized micro-kernel, ab is held
    in registers rather than memory.
  - In column-major storage (or column storage), the "leading dimension"
    of a matrix is equivalent to its column stride, and the row stride is
    unit. In row-major storage (row storage), the "leading dimension" is
    equivalent to the row stride and the column stride is unit.
  - While all three loops are exposed in this template micro-kernel, the
    loops over MR and NR typically disappear in an optimized code because
    they are fully unrolled, leaving only the loop over k.
  - Some optimized micro-kernels will need the loop over k to be unrolled
    a few times (4x seems to be a common unrolling factor).
  - a_next and b_next can be used to perform prefetching, if prefetching
    is supported by the architecture. They may be safely ignored by the
    micro-kernel implementation, though.
  - If beta == 0.0 (or 0.0 + 0.0i for complex), then the micro-kernel
    should NOT use it explicitly, as C may contain uninitialized memory
    (including NaNs). This case should be detected and handled separately,
    preferably by simply overwriting C with the alpha*A*B product. An
    example of how to perform this "beta is zero" handling is included in
    this template implementation.

  For more info, please refer to the BLIS website and/or contact the
  blis-devel mailing list.

  -FGVZ
*/
	const dim_t        mr    = bli_dmr;
	const dim_t        nr    = bli_dnr;

	const inc_t        cs_a  = bli_dpackmr;

	const inc_t        rs_b  = bli_dpacknr;

	const inc_t        rs_ab = 1;
	const inc_t        cs_ab = bli_dmr;

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
			bj = *(b + j);

			for ( i = 0; i < mr; ++i )
			{
				ai = *(a + i);

				bli_ddots( ai, bj, *abij );

				abij += rs_ab;
			}
		}

		a += cs_a;
		b += rs_b;
	}

	/* Scale each element of ab by alpha. */
	for ( i = 0; i < mr * nr; ++i )
	{
		bli_dscals( *alpha, *(ab + i) );
	}

	/* If beta is zero, overwrite c with the scaled result in ab. Otherwise,
	   scale c by beta and then add the scaled result in ab. */
	if ( bli_deq0( *beta ) )
	{
		/* c := ab */
		bli_dcopys_mxn( mr,
		                nr,
		                ab, rs_ab, cs_ab,
		                c,  rs_c,  cs_c );
	}
	else
	{
		/* c := beta * c + ab */
		bli_dxpbys_mxn( mr,
		                nr,
		                ab, rs_ab, cs_ab,
		                beta,
		                c,  rs_c,  cs_c );
	}
}



void bli_cgemm_opt_mxn(
                        dim_t              k,
                        scomplex* restrict alpha,
                        scomplex* restrict a,
                        scomplex* restrict b,
                        scomplex* restrict beta,
                        scomplex* restrict c, inc_t rs_c, inc_t cs_c,
                        scomplex* restrict a_next,
                        scomplex* restrict b_next 
                      )
{
	/* Just call the reference implementation. */
	bli_cgemm_ref_mxn( k,
	                   alpha,
	                   a,
	                   b,
	                   beta,
	                   c, rs_c, cs_c,
	                   a_next,
	                   b_next );
}



void bli_zgemm_opt_mxn(
                        dim_t              k,
                        dcomplex* restrict alpha,
                        dcomplex* restrict a,
                        dcomplex* restrict b,
                        dcomplex* restrict beta,
                        dcomplex* restrict c, inc_t rs_c, inc_t cs_c,
                        dcomplex* restrict a_next,
                        dcomplex* restrict b_next 
                      )
{
	/* Just call the reference implementation. */
	bli_zgemm_ref_mxn( k,
	                   alpha,
	                   a,
	                   b,
	                   beta,
	                   c, rs_c, cs_c,
	                   a_next,
	                   b_next );
}

