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



void bli_sgemmtrsm_l_opt_mxn(
                              dim_t              k,
                              float*    restrict alpha,
                              float*    restrict a10,
                              float*    restrict a11,
                              float*    restrict bd01,
                              float*    restrict bd11,
                              float*    restrict b11,
                              float*    restrict c11, inc_t rs_c, inc_t cs_c,
                              float*    restrict a_next,
                              float*    restrict b_next 
                            )
{
	const inc_t        rs_b      = bli_spacknr;
	const inc_t        cs_b      = 1;

	float*    restrict minus_one = bli_sm1;


	bli_sgemm_opt_mxn( k,
	                   minus_one,
	                   a10,
	                   bd01,
	                   alpha,
	                   b11, rs_b, cs_b,
	                   a_next,
	                   b_next );

	bli_strsm_l_opt_mxn( a11,
	                     b11,
	                     bd11,
	                     c11, rs_c, cs_c );
}



void bli_dgemmtrsm_l_opt_mxn(
                              dim_t              k,
                              double*   restrict alpha,
                              double*   restrict a10,
                              double*   restrict a11,
                              double*   restrict bd01,
                              double*   restrict bd11,
                              double*   restrict b11,
                              double*   restrict c11, inc_t rs_c, inc_t cs_c,
                              double*   restrict a_next,
                              double*   restrict b_next 
                            )
{
/*
  Template gemmtrsm_l micro-kernel implementation

  This function contains a template implementation for a double-precision
  real micro-kernel that fuses a gemm with a trsm_l subproblem. 

  This micro-kernel implements the following sequence of operations:

    B11 := alpha * B11 - A10 * B01    (gemm)
    B11 := inv(A11) * B11             (trsm)

  where B11 is MR x NR, A10 is MR x k, B01 is k x NR, A11 is MR x MR and
  lower triangular, and alpha is a scalar. Here, inv() denotes matrix
  inverse.

  NOTE: Here, this gemmtrsm micro-kernel supports element "duplication", a
  feature that is enabled or disabled in bli_kernel.h. Duplication factors
  are also defined in the aforementioned header. Duplication is NOT
  commonly used and most developers may assume it is disabled.

  Parameters:

  - k:      The number of columns of A10 and rows of B01.
  - alpha:  The address of a scalar to be applied to B11.
  - a10:    The address of A10, which is the MR x k subpartition of the
            packed (column-stored) micro-panel of A that is situated to the
            left of the MR x MR lower triangular block.
  - a11:    The address of A11, which is the MR x MR lower triangular block
            within the packed micro-panel of A that is situated to the
            right of A10. By the time this gemmtrsm kernel is called, the
            diagonal of A11 has already been inverted and the strictly upper
            triangle contains zeros.
  - bd01:   The address of B01, which is the k x NR subpartition situated
            above the current MR x NR block B11. bd01 is row-stored. If
            duplication is enabled, then each element occurs d times,
            effectively increasing the dimension to k x d*NR. If duplication
            is disabled, then bd01 is simply the address of the top part of
            the current packed (row-stored) micro-panel of B (labeled b01
            in the diagram below).
  - bd11:   The address of B11, which is the MR x NR subpartition situated
            below B01. If duplication is enabled, then each element occurs
            d times, effectively increasing the dimension to MR x d*NR. If
            duplication is disabled, then bd11 is simply the address of the
            current MR x NR block within the packed (row-stored) micro-panel
            of B.
  - b11:    The address of the current MR x NR block within the packed
            micro-panel of B. It exists in duplicated form as bd11. If
            duplication is disabled, then b11 and bd11 refer to the same
            MR x NR block within the packed (row-stored) micro-panel of B.
  - c11:    The address of C11, which is the MR x NR block of the output
            matrix (ie: the matrix provided by the user to the highest-level
            trsm API call). C11 corresponds to the elements that exist in
            packed form in B11, and is stored according to rs_c and cs_c.
  - rs_c:   The row stride of C11 (ie: the distance to the next row of C11,
            in units of matrix elements).
  - cs_c:   The column stride of C11 (ie: the distance to the next column of
            C11, in units of matrix elements).
  - a_next: The address of the packed micro-panel of A that will be used the
            next time the gemmtrsm micro-kernel will be called.
  - b_next: The address of the packed micro-panel of B that will be used the
            next time the gemmtrsm micro-kernel will be called.

  The diagram below shows the packed micro-panel operands and how elements
  of each would be stored when MR == NR == 4. (The hex digits indicate the
  order of the elements in memory.) We also show a B duplication buffer (bd)
  that contains a copy of the packed micro-panel of B with a duplication
  factor of 2. If duplication is disabled (as is commonly the case), then
  bd01 == b01 and bd11 == b11.

                                             NR                 2*NR       
    NOTE: If duplication is disabled       _______         _______________ 
    then bd01 and bd11 simply refer   b01:|0 1 2 3|  bd01:|0 0 1 1 2 2 3 3|
    to b01 and b11, respectively.         |4 5 6 7|       |4 4 5 5 6 6 7 7|
                                          |8 9 A B|       |8 8 9 9 A A B B|
                                          |C D E F|       |C C D D E E F F|
                                        k |   .   |       |       .       |
                                          |   .   |       |       .       |
       a10:                a11:           |   .   |       |       .       |
       ___________________  _______       |_______|       |_______________|
      |0 4 8 C            |`.      |  b11:|       |  bd11:|               |
  MR  |1 5 9 D . . .      |  `.    |      |       |       |               |
      |2 6 A E            |    `.  |   MR |       |       |               |
      |3_7_B_F____________|______`.|      |_______|       |_______________|
                                                                           
                k             MR                                           

  Thus, with duplication enabled, the operation takes the form of:

    b11  = alpha * b11 - a10 * bd01;
    b11  = inv(a11) * b11;
    bd11 = b11;  (skipped if duplication is disabled)
    c11  = b11;
                                                                        
  And if duplication is disabled, the operation reduces to:

    b11 = alpha * b11 - a10 * b01;  (Note: Here, b01 == bd01.)
    b11 = inv(a11) * b11;
    c11 = b11;

  A note on optimization:
  - This implementation simply calls the gemm micro-kernel and then the
    trsm micro-kernel. Let's assume that the gemm micro-kernel has already
    been optimized. You have two options with regards to optimizing the
    fused gemmtrsm kernel.
    (1) Optimize only the trsm kernel and continue to call the gemm and
        trsm micro-kernels in sequence, as is done in this template
        implementation.
    (2) Fuse the implementation of the gemm micro-kernel with that of the
        trsm micro-kernel by inlining both into this gemmtrsm function.
    The latter option is more labor-intensive, but also more likely to
    yield higher performance because it allows you to eliminate redundant
    memory operations on the packed MR x NR block B11.

  For more info, please refer to the BLIS website and/or contact the
  blis-devel mailing list.

  -FGVZ
*/
	const inc_t        rs_b      = bli_dpacknr;
	const inc_t        cs_b      = 1;

	double*   restrict minus_one = bli_dm1;

	/* Reminder: if duplication is disabled, then bd01 == b01, bd11 == b11. */

	/* b11 = alpha * b11 - a10 * bd01; */
	bli_dgemm_opt_mxn( k,
	                   minus_one,
	                   a10,
	                   bd01,
	                   alpha,
	                   b11, rs_b, cs_b,
	                   a_next,
	                   b_next );

	/* b11  = inv(a11) * b11;
	   bd11 = b11; (skipped if duplication is disabled)
	   c11  = b11; */
	bli_dtrsm_l_opt_mxn( a11,
	                     b11,
	                     bd11,
	                     c11, rs_c, cs_c );
}



void bli_cgemmtrsm_l_opt_mxn(
                              dim_t              k,
                              scomplex* restrict alpha,
                              scomplex* restrict a10,
                              scomplex* restrict a11,
                              scomplex* restrict bd01,
                              scomplex* restrict bd11,
                              scomplex* restrict b11,
                              scomplex* restrict c11, inc_t rs_c, inc_t cs_c,
                              scomplex* restrict a_next,
                              scomplex* restrict b_next 
                            )
{
	const inc_t        rs_b      = bli_cpacknr;
	const inc_t        cs_b      = 1;

	scomplex* restrict minus_one = bli_cm1;


	bli_cgemm_opt_mxn( k,
	                   minus_one,
	                   a10,
	                   bd01,
	                   alpha,
	                   b11, rs_b, cs_b,
	                   a_next,
	                   b_next );

	bli_ctrsm_l_opt_mxn( a11,
	                     b11,
	                     bd11,
	                     c11, rs_c, cs_c );
}



void bli_zgemmtrsm_l_opt_mxn(
                              dim_t              k,
                              dcomplex* restrict alpha,
                              dcomplex* restrict a10,
                              dcomplex* restrict a11,
                              dcomplex* restrict bd01,
                              dcomplex* restrict bd11,
                              dcomplex* restrict b11,
                              dcomplex* restrict c11, inc_t rs_c, inc_t cs_c,
                              dcomplex* restrict a_next,
                              dcomplex* restrict b_next 
                            )
{
	const inc_t        rs_b      = bli_zpacknr;
	const inc_t        cs_b      = 1;

	dcomplex* restrict minus_one = bli_zm1;


	bli_zgemm_opt_mxn( k,
	                   minus_one,
	                   a10,
	                   bd01,
	                   alpha,
	                   b11, rs_b, cs_b,
	                   a_next,
	                   b_next );

	bli_ztrsm_l_opt_mxn( a11,
	                     b11,
	                     bd11,
	                     c11, rs_c, cs_c );
}

