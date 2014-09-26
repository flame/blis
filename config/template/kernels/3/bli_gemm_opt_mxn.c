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



void bli_sgemm_opt_mxn(
                        dim_t              k,
                        float*    restrict alpha,
                        float*    restrict a1,
                        float*    restrict b1,
                        float*    restrict beta,
                        float*    restrict c11, inc_t rs_c, inc_t cs_c,
                        auxinfo_t*         data
                      )
{
	/* Just call the reference implementation. */
	BLIS_SGEMM_UKERNEL_REF( k,
	                   alpha,
	                   a1,
	                   b1,
	                   beta,
	                   c11, rs_c, cs_c,
	                   data );
}



void bli_dgemm_opt_mxn(
                        dim_t              k,
                        double*   restrict alpha,
                        double*   restrict a1,
                        double*   restrict b1,
                        double*   restrict beta,
                        double*   restrict c11, inc_t rs_c, inc_t cs_c,
                        auxinfo_t*         data
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

  Parameters:

  - k:      The number of columns of A1 and rows of B1.
  - alpha:  The address of a scalar to the A1 * B1 product.
  - a1:     The address of a micro-panel of matrix A of dimension MR x k,
            stored by columns with leading dimension PACKMR, where
            typically PACKMR = MR.
  - b1:     The address of a micro-panel of matrix B of dimension k x NR,
            stored by rows with leading dimension PACKNR, where typically
            PACKNR = NR.
  - beta:   The address of a scalar to the input value of matrix C11.
  - c11:    The address of a submatrix C11 of dimension MR x NR, stored
            according to rs_c and cs_c.
  - rs_c:   The row stride of matrix C11 (ie: the distance to the next row,
            in units of matrix elements).
  - cs_c:   The column stride of matrix C11 (ie: the distance to the next
            column, in units of matrix elements).
  - data:   The address of an auxinfo_t object that contains auxiliary
            information that may be useful when optimizing the gemm
            micro-kernel implementation. (See BLIS KernelsHowTo wiki for
            more info.)

  Diagram for gemm

  The diagram below shows the packed micro-panel operands and how elements
  of each would be stored when MR = NR = 4. The hex digits indicate the
  layout and order (but NOT the numeric contents) of the elements in
  memory. Note that the storage of C11 is not shown since it is determined
  by the row and column strides of C11.

         c11:           a1:                        b1:       
         _______        ______________________     _______   
        |       |      |0 4 8 C               |   |0 1 2 3|  
    MR  |       |      |1 5 9 D . . .         |   |4 5 6 7|  
        |       |  +=  |2 6 A E               |   |8 9 A B|  
        |_______|      |3_7_B_F_______________|   |C D E F|  
                                                  |   .   |  
            NR                    k               |   .   | k
                                                  |   .   |  
                                                  |       |  
                                                  |       |  
                                                  |_______|  
                                                             
                                                      NR     
  Implementation Notes for gemm

  - Register blocksizes. The C preprocessor macros bli_?mr and bli_?nr
    evaluate to the MR and NR register blocksizes for the datatype
    corresponding to the '?' character. These values are abbreviations
    of the macro constants BLIS_DEFAULT_MR_? and BLIS_DEFAULT_NR_?,
    which are defined in the bli_kernel.h header file of the BLIS
    configuration.
  - Leading dimensions of a1 and b1: PACKMR and PACKNR. The packed
    micro-panels a1 and b1 are simply stored in column-major and row-major
    order, respectively. Usually, the width of either micro-panel (ie:
    the number of rows of A1, or MR, and the number of columns of B1, or
    NR) is equal to that micro-panel's so-called "leading dimension."
    Sometimes, it may be beneficial to specify a leading dimension that
    is larger than the panel width. This may be desirable because it
    allows each column of A1 or row of B1 to maintain a certain alignment
    in memory that would not otherwise be maintained by MR and/or NR. In
    this case, you should index through a1 and b1 using the values PACKMR
    and PACKNR, respectively, as defined by bli_?packmr and bli_?packnr.
    These values are defined as BLIS_PACKDIM_MR_? and BLIS_PACKDIM_NR_?,
    respectively, in the bli_kernel.h header file of the BLIS
    configuration.
  - Storage preference of c11: Sometimes, an optimized micro-kernel will
    have a preferred storage format for C11--typically either contiguous
    row-storage or contiguous column-storage. This preference comes from
    how the micro-kernel is most efficiently able to load/store elements
    of C11 from/to memory. Most micro-kernels use vector instructions to
    load and store contigous columns (or column segments) of C11. However,
    the developer may decide that loading contiguous rows (or row
    segments) is desirable. If this is the case, this preference should be
    noted in bli_kernel.h by defining the macro
    BLIS_?GEMM_UKERNEL_PREFERS_CONTIG_ROWS. Leaving the macro undefined
    leaves the default assumption (contiguous column preference) in
    place. Setting this macro allows the framework to perform a minor
    optimization at run-time that will ensure the micro-kernel preference
    is honored, if at all possible.
  - Edge cases in MR, NR dimensions. Sometimes the micro-kernel will be
    called with micro-panels a1 and b1 that correspond to edge cases,
    where only partial results are needed. Zero-padding is handled
    automatically by the packing function to facilitate reuse of the same
    micro-kernel. Similarly, the logic for computing to temporary storage
    and then saving only the elements that correspond to elements of C11
    that exist (at the edges) is handled automatically within the
    macro-kernel.
  - Alignment of a1 and b1. The addresses a1 and b1 are aligned according
    to PACKMR*sizeof(type) and PACKNR*sizeof(type), respectively.
  - Unrolling loops. As a general rule of thumb, the loop over k is
    sometimes moderately unrolled; for example, in our experience, an
    unrolling factor of u = 4 is fairly common. If unrolling is applied
    in the k dimension, edge cases must be handled to support values of k
    that are not multiples of u. It is nearly universally true that there
    should be no loops in the MR or NR directions; in other words,
    iteration over these dimensions should always be fully unrolled
    (within the loop over k).
  - Zero beta. If beta = 0.0 (or 0.0 + 0.0i for complex datatypes), then
    the micro-kernel should NOT use it explicitly, as C11 may contain
    uninitialized memory (including NaNs). This case should be detected
    and handled separately, preferably by simply overwriting C11 with the
    alpha * A1 * B1 product. An example of how to perform this "beta equals
    zero" handling is included in the gemm micro-kernel associated with
    the template configuration.

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
{
	/* Just call the reference implementation. */
	BLIS_CGEMM_UKERNEL_REF( k,
	                   alpha,
	                   a1,
	                   b1,
	                   beta,
	                   c11, rs_c, cs_c,
	                   data );
}



void bli_zgemm_opt_mxn(
                        dim_t              k,
                        dcomplex* restrict alpha,
                        dcomplex* restrict a1,
                        dcomplex* restrict b1,
                        dcomplex* restrict beta,
                        dcomplex* restrict c11, inc_t rs_c, inc_t cs_c,
                        auxinfo_t*         data
                      )
{
	/* Just call the reference implementation. */
	BLIS_ZGEMM_UKERNEL_REF( k,
	                   alpha,
	                   a1,
	                   b1,
	                   beta,
	                   c11, rs_c, cs_c,
	                   data );
}

