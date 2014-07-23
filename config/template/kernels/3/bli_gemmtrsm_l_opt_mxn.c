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



void bli_sgemmtrsm_l_opt_mxn(
                              dim_t              k,
                              float*    restrict alpha,
                              float*    restrict a10,
                              float*    restrict a11,
                              float*    restrict b01,
                              float*    restrict b11,
                              float*    restrict c11, inc_t rs_c, inc_t cs_c,
                              auxinfo_t*         data
                            )
{
	const inc_t        rs_b      = bli_spacknr;
	const inc_t        cs_b      = 1;

	float*    restrict minus_one = bli_sm1;


	bli_sgemm_opt_mxn( k,
	                   minus_one,
	                   a10,
	                   b01,
	                   alpha,
	                   b11, rs_b, cs_b,
	                   data );

	bli_strsm_l_opt_mxn( a11,
	                     b11,
	                     c11, rs_c, cs_c,
	                     data );
}



void bli_dgemmtrsm_l_opt_mxn(
                              dim_t              k,
                              double*   restrict alpha,
                              double*   restrict a10,
                              double*   restrict a11,
                              double*   restrict b01,
                              double*   restrict b11,
                              double*   restrict c11, inc_t rs_c, inc_t cs_c,
                              auxinfo_t*         data
                            )
{
/*
  Template gemmtrsm_l micro-kernel implementation

  This function contains a template implementation for a double-precision
  real micro-kernel that fuses a gemm with a trsm_l subproblem.

  This micro-kernel performs the following compound operation:

    B11 := alpha * B11 - A10 * B01    (gemm)
    B11 := inv(A11) * B11             (trsm)
    C11 := B11

  where A11 is MR x MR and lower triangular, A10 is MR x k, B01 is k x NR,
  B11 is MR x NR, and alpha is a scalar. Here, inv() denotes matrix
  inverse.

  Parameters:

  - k:      The number of columns of A10 and rows of B01.
  - alpha:  The address of a scalar to be applied to B11.
  - a10:    The address of A10, which is the MR x k submatrix of the packed
            micro-panel of A that is situated to the left of the MR x MR
            triangular submatrix A11. A10 is stored by columns with leading
            dimension PACKMR, where typically PACKMR = MR.
  - a11:    The address of A11, which is the MR x MR lower triangular
            submatrix within the packed micro-panel of matrix A that is
            situated to the right of A10. A11 is stored by columns with
            leading dimension PACKMR, where typically PACKMR = MR. Note
            that A11 contains elements in both triangles, though elements
            in the unstored triangle are not guaranteed to be zero and
            thus should not be referenced.
  - b01:    The address of B01, which is the k x NR submatrix of the packed
            micro-panel of B that is situated above the MR x NR submatrix
            B11. B01 is stored by rows with leading dimension PACKNR, where
            typically PACKNR = NR.
  - b11:    The address B11, which is the MR x NR submatrix of the packed
            micro-panel of B, situated below B01. B11 is stored by rows
            with leading dimension PACKNR, where typically PACKNR = NR.
  - c11:    The address of C11, which is the MR x NR submatrix of matrix
            C, stored according to rs_c and cs_c. C11 is the submatrix
            within C that corresponds to the elements which were packed
            into B11. Thus, C is the original input matrix B to the overall
            trsm operation.
  - rs_c:   The row stride of C11 (ie: the distance to the next row of C11,
            in units of matrix elements).
  - cs_c:   The column stride of C11 (ie: the distance to the next column of
            C11, in units of matrix elements).
  - data:   The address of an auxinfo_t object that contains auxiliary
            information that may be useful when optimizing the gemmtrsm
            micro-kernel implementation. (See BLIS KernelsHowTo wiki for
            more info.)

  Diagram for gemmtrsm_l

  The diagram below shows the packed micro-panel operands for trsm_l and
  how elements of each would be stored when MR = NR = 4. (The hex digits
  indicate the layout and order (but NOT the numeric contents) in memory.
  Here, matrix A11 (referenced by a11) is lower triangular. Matrix A11
  does contain elements corresponding to the strictly upper triangle,
  however, they are not guaranteed to contain zeros and thus these elements
  should not be referenced.

                                                NR    
                                              _______ 
                                         b01:|0 1 2 3|
                                             |4 5 6 7|
                                             |8 9 A B|
                                             |C D E F|
                                           k |   .   |
                                             |   .   |
         a10:                a11:            |   .   |
         ___________________  _______        |_______|
        |0 4 8 C            |`.      |   b11:|       |
    MR  |1 5 9 D . . .      |  `.    |       |       |
        |2 6 A E            |    `.  |    MR |       |
        |3_7_B_F____________|______`.|       |_______|
                                                      
                  k             MR                    


  Implementation Notes for gemmtrsm

  - Register blocksizes. See Implementation Notes for gemm.
  - Leading dimensions of a1 and b1: PACKMR and PACKNR. See Implementation
    Notes for gemm.
  - Edge cases in MR, NR dimensions. See Implementation Notes for gemm.
  - Alignment of a1 and b1. The addresses a1 and b1 are aligned according
    to PACKMR*sizeof(type) and PACKNR*sizeof(type), respectively.
  - Unrolling loops. Most optimized implementations should unroll all
    three loops within the trsm subproblem of gemmtrsm. See Implementation
    Notes for gemm for remarks on unrolling the gemm subproblem.
  - Prefetching next micro-panels of A and B. When invoked from within a
    gemmtrsm_l micro-kernel, the addresses accessible via
    bli_auxinfo_next_a() and bli_auxinfo_next_b() refer to the next
    invocation's a10 and b01, respectively, while in gemmtrsm_u, the
    _next_a() and _next_b() macros return the addresses of the next
    invocation's a11 and b11 (since those submatrices precede a12 and b21).
    (See BLIS KernelsHowTo wiki for more info.)
  - Zero alpha. The micro-kernel can safely assume that alpha is non-zero;
    "alpha equals zero" handling is performed at a much higher level,
    which means that, in such a scenario, the micro-kernel will never get
    called.
  - Diagonal elements of A11. See Implementation Notes for trsm.
  - Zero elements of A11. See Implementation Notes for trsm.
  - Output. See Implementation Notes for trsm.
  - Optimization. Let's assume that the gemm micro-kernel has already been
    optimized. You have two options with regard to optimizing the fused
    gemmtrsm micro-kernels:
    (1) Optimize only the trsm micro-kernels. This will result in the gemm
        and trsm_l micro-kernels being called in sequence. (Likewise for
        gemm and trsm_u.)
    (2) Fuse the implementation of the gemm micro-kernel with that of the
        trsm micro-kernels by inlining both into the gemmtrsm_l and
        gemmtrsm_u micro-kernel definitions. This option is more labor-
        intensive, but also more likely to yield higher performance because
        it avoids redundant memory operations on the packed MR x NR
        submatrix B11.

  For more info, please refer to the BLIS website and/or contact the
  blis-devel mailing list.

  -FGVZ
*/
	const inc_t        rs_b      = bli_dpacknr;
	const inc_t        cs_b      = 1;

	double*   restrict minus_one = bli_dm1;

	/* b11 = alpha * b11 - a10 * b01; */
	bli_dgemm_opt_mxn( k,
	                   minus_one,
	                   a10,
	                   b01,
	                   alpha,
	                   b11, rs_b, cs_b,
	                   data );

	/* b11 = inv(a11) * b11;
	   c11 = b11; */
	bli_dtrsm_l_opt_mxn( a11,
	                     b11,
	                     c11, rs_c, cs_c,
	                     data );
}



void bli_cgemmtrsm_l_opt_mxn(
                              dim_t              k,
                              scomplex* restrict alpha,
                              scomplex* restrict a10,
                              scomplex* restrict a11,
                              scomplex* restrict b01,
                              scomplex* restrict b11,
                              scomplex* restrict c11, inc_t rs_c, inc_t cs_c,
                              auxinfo_t*         data
                            )
{
	const inc_t        rs_b      = bli_cpacknr;
	const inc_t        cs_b      = 1;

	scomplex* restrict minus_one = bli_cm1;


	bli_cgemm_opt_mxn( k,
	                   minus_one,
	                   a10,
	                   b01,
	                   alpha,
	                   b11, rs_b, cs_b,
	                   data );

	bli_ctrsm_l_opt_mxn( a11,
	                     b11,
	                     c11, rs_c, cs_c,
	                     data );
}



void bli_zgemmtrsm_l_opt_mxn(
                              dim_t              k,
                              dcomplex* restrict alpha,
                              dcomplex* restrict a10,
                              dcomplex* restrict a11,
                              dcomplex* restrict b01,
                              dcomplex* restrict b11,
                              dcomplex* restrict c11, inc_t rs_c, inc_t cs_c,
                              auxinfo_t*         data
                            )
{
	const inc_t        rs_b      = bli_zpacknr;
	const inc_t        cs_b      = 1;

	dcomplex* restrict minus_one = bli_zm1;


	bli_zgemm_opt_mxn( k,
	                   minus_one,
	                   a10,
	                   b01,
	                   alpha,
	                   b11, rs_b, cs_b,
	                   data );

	bli_ztrsm_l_opt_mxn( a11,
	                     b11,
	                     c11, rs_c, cs_c,
	                     data );
}

