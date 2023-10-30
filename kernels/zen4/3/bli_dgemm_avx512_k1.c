/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.

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

#include <stdio.h>
#include "blis.h"
#include "immintrin.h"


#define D_MR  24
#define D_NR  8

err_t bli_dgemm_24x8_avx512_k1_nn
(
    dim_t  m,
    dim_t  n,
    dim_t  k,
    double*    alpha,
    double*    a, const inc_t lda,
    double*    b, const inc_t ldb,
    double*    beta,
    double*    c, const inc_t ldc
)
{
    err_t ret_status = BLIS_FAILURE;
    double alpha_val, beta_val;

    beta_val = *beta;
    alpha_val = *alpha;

    dim_t m_remainder = (m % D_MR);
    dim_t n_remainder = (n % D_NR);

    //scratch registers
    __m512d zmm0, zmm1, zmm2, zmm3;
    __m512d zmm4, zmm5, zmm6, zmm7;
    __m512d zmm8, zmm9, zmm10, zmm11;
    __m512d zmm12, zmm13, zmm14, zmm15;
    __m512d zmm16, zmm17, zmm18, zmm19;
    __m512d zmm20, zmm21, zmm22, zmm23;
    __m512d zmm24, zmm25, zmm26, zmm27;
    __m512d zmm28, zmm29, zmm30, zmm31;

    if(alpha_val != 0.0 && beta_val != 0.0)
    {
        /* Compute C = alpha*A*B + beta*c */
        for(dim_t j = 0; (j + (D_NR-1) < n ); j += D_NR)
        {
            double* temp_b = b + j*ldb;
            double* temp_a = a;
            double* temp_c = c + j*ldc;

            for(dim_t i = 0; i < ( m - D_MR+1); i += D_MR)
            {
                //Clear out vector registers to hold fma result.
                //zmm6 to zmm29 holds fma result.
                //zmm0, zmm1, zmm2 are used to load 24 elements from
                //A matrix.
                //zmm30 and zmm31 are alternatively used to broadcast element
                //from B matrix.
                zmm6 = _mm512_setzero_pd();
                zmm7 = _mm512_setzero_pd();
                zmm8 = _mm512_setzero_pd();
                zmm9 = _mm512_setzero_pd();
                zmm10 = _mm512_setzero_pd();
                zmm11 = _mm512_setzero_pd();
                zmm12 = _mm512_setzero_pd();
                zmm13 = _mm512_setzero_pd();
                zmm14 = _mm512_setzero_pd();
                zmm15 = _mm512_setzero_pd();
                zmm16 = _mm512_setzero_pd();
                zmm17 = _mm512_setzero_pd();
                zmm18 = _mm512_setzero_pd();
                zmm19 = _mm512_setzero_pd();
                zmm20 = _mm512_setzero_pd();
                zmm21 = _mm512_setzero_pd();
                zmm22 = _mm512_setzero_pd();
                zmm23 = _mm512_setzero_pd();
                zmm24 = _mm512_setzero_pd();
                zmm25 = _mm512_setzero_pd();
                zmm26 = _mm512_setzero_pd();
                zmm27 = _mm512_setzero_pd();
                zmm28 = _mm512_setzero_pd();
                zmm29 = _mm512_setzero_pd();
                /*
                    a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                    where alpha_val is not zero.
                    b. This loop operates with 24x8 block size
                    along n dimension for every D_NR columns of temp_b where
                    computing all D_MR rows of temp_a.
                    c. Same approach is used in remaining fringe cases.
                */
                zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                zmm1 = _mm512_loadu_pd((double const *)(temp_a + 8));
                zmm2 = _mm512_loadu_pd((double const *)(temp_a + 16));

                _mm_prefetch((char*)( temp_a + 192), _MM_HINT_T0);
                //Broadcast element from B matrix in zmm30
                zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                //Broadcast element from next column of B matrix in zmm31
                zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));
                //Compute A*B.
                zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);
                zmm8 = _mm512_fmadd_pd(zmm2, zmm30, zmm8);
                //Broadcast element from B matrix in zmm30
                zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));
                //Compute A*B.
                zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);
                //Broadcast element from B matrix in zmm31
                zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 3));
                //Compute A*B.
                zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);
                zmm13 = _mm512_fmadd_pd(zmm1, zmm30, zmm13);
                zmm14 = _mm512_fmadd_pd(zmm2, zmm30, zmm14);
                //Broadcast element from B matrix in zmm30
                zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 4));
                //Compute A*B.
                zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);
                zmm17 = _mm512_fmadd_pd(zmm2, zmm31, zmm17);
                //Broadcast element from B matrix in zmm31
                zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 5));
                //Compute A*B.
                zmm18 = _mm512_fmadd_pd(zmm0, zmm30, zmm18);
                zmm19 = _mm512_fmadd_pd(zmm1, zmm30, zmm19);
                zmm20 = _mm512_fmadd_pd(zmm2, zmm30, zmm20);
                //Broadcast element from B matrix in zmm30
                zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 6));
                //Compute A*B.
                zmm21 = _mm512_fmadd_pd(zmm0, zmm31, zmm21);
                zmm22 = _mm512_fmadd_pd(zmm1, zmm31, zmm22);
                zmm23 = _mm512_fmadd_pd(zmm2, zmm31, zmm23);
                //Broadcast element from B matrix in zmm31
                zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 7));
                //Compute A*B.
                zmm24 = _mm512_fmadd_pd(zmm0, zmm30, zmm24);
                zmm25 = _mm512_fmadd_pd(zmm1, zmm30, zmm25);
                zmm26 = _mm512_fmadd_pd(zmm2, zmm30, zmm26);
                //Compute A*B.
                zmm27 = _mm512_fmadd_pd(zmm0, zmm31, zmm27);
                zmm28 = _mm512_fmadd_pd(zmm1, zmm31, zmm28);
                zmm29 = _mm512_fmadd_pd(zmm2, zmm31, zmm29);

                //Broadcast Alpha into zmm0
                zmm0 = _mm512_set1_pd(alpha_val);
                //Scale fma result with Alpha.
                //Alpha * AB
                zmm6 = _mm512_mul_pd(zmm0, zmm6);
                zmm7 = _mm512_mul_pd(zmm0, zmm7);
                zmm8 = _mm512_mul_pd(zmm0, zmm8);
                zmm9 = _mm512_mul_pd(zmm0, zmm9);
                zmm10 = _mm512_mul_pd(zmm0, zmm10);
                zmm11 = _mm512_mul_pd(zmm0, zmm11);
                zmm12 = _mm512_mul_pd(zmm0, zmm12);
                zmm13 = _mm512_mul_pd(zmm0, zmm13);
                zmm14 = _mm512_mul_pd(zmm0, zmm14);
                zmm15 = _mm512_mul_pd(zmm0, zmm15);
                zmm16 = _mm512_mul_pd(zmm0, zmm16);
                zmm17 = _mm512_mul_pd(zmm0, zmm17);
                zmm18 = _mm512_mul_pd(zmm0, zmm18);
                zmm19 = _mm512_mul_pd(zmm0, zmm19);
                zmm20 = _mm512_mul_pd(zmm0, zmm20);
                zmm21 = _mm512_mul_pd(zmm0, zmm21);
                zmm22 = _mm512_mul_pd(zmm0, zmm22);
                zmm23 = _mm512_mul_pd(zmm0, zmm23);
                zmm24 = _mm512_mul_pd(zmm0, zmm24);
                zmm25 = _mm512_mul_pd(zmm0, zmm25);
                zmm26 = _mm512_mul_pd(zmm0, zmm26);
                zmm27 = _mm512_mul_pd(zmm0, zmm27);
                zmm28 = _mm512_mul_pd(zmm0, zmm28);
                zmm29 = _mm512_mul_pd(zmm0, zmm29);

                //Broadcast Beta into zmm31
                zmm31 = _mm512_set1_pd(beta_val);

                //zmm0, zmm1, zmm2 are used to load 24 elements from
                //matrix C.
                zmm0 = _mm512_loadu_pd((double const *)(temp_c));
                zmm1 = _mm512_loadu_pd((double const *)(temp_c + 8));
                zmm2 = _mm512_loadu_pd((double const *)(temp_c + 16));
                //Compute C * Beta + fma result(AB*Alpha)
                zmm6 = _mm512_fmadd_pd(zmm0, zmm31, zmm6);
                zmm7 = _mm512_fmadd_pd(zmm1, zmm31, zmm7);
                zmm8 = _mm512_fmadd_pd(zmm2, zmm31, zmm8);

                //zmm0, zmm1, zmm2 are used to load 24 elements from
                //matrix C.
                zmm3 = _mm512_loadu_pd((double const *)(temp_c + ldc ));
                zmm4 = _mm512_loadu_pd((double const *)(temp_c + ldc + 8));
                zmm5 = _mm512_loadu_pd((double const *)(temp_c + ldc + 16));
                //Compute C * Beta + fma result(AB*Alpha)
                zmm9 = _mm512_fmadd_pd(zmm3, zmm31, zmm9);
                zmm10 = _mm512_fmadd_pd(zmm4, zmm31, zmm10);
                zmm11 = _mm512_fmadd_pd(zmm5, zmm31, zmm11);

                //zmm0, zmm1, zmm2 are used to load 24 elements from
                //matrix C.
                zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 2));
                zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc * 2 + 8));
                zmm2 = _mm512_loadu_pd((double const *)(temp_c + ldc * 2 + 16));
                //Compute C * Beta + fma result(AB*Alpha)
                zmm12 = _mm512_fmadd_pd(zmm0, zmm31, zmm12);
                zmm13 = _mm512_fmadd_pd(zmm1, zmm31, zmm13);
                zmm14 = _mm512_fmadd_pd(zmm2, zmm31, zmm14);

                //zmm0, zmm1, zmm2 are used to load 24 elements from
                //matrix C.
                zmm3 = _mm512_loadu_pd((double const *)(temp_c + ldc * 3));
                zmm4 = _mm512_loadu_pd((double const *)(temp_c + ldc * 3 + 8));
                zmm5 = _mm512_loadu_pd((double const *)(temp_c + ldc * 3 + 16));
                //Compute C * Beta + fma result(AB*Alpha)
                zmm15 = _mm512_fmadd_pd(zmm3, zmm31, zmm15);
                zmm16 = _mm512_fmadd_pd(zmm4, zmm31, zmm16);
                zmm17 = _mm512_fmadd_pd(zmm5, zmm31, zmm17);

                //zmm0, zmm1, zmm2 are used to load 24 elements from
                //matrix C.
                zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 4));
                zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc * 4 + 8));
                zmm2 = _mm512_loadu_pd((double const *)(temp_c + ldc * 4 + 16));
                //Compute C * Beta + fma result(AB*Alpha)
                zmm18 = _mm512_fmadd_pd(zmm0, zmm31, zmm18);
                zmm19 = _mm512_fmadd_pd(zmm1, zmm31, zmm19);
                zmm20 = _mm512_fmadd_pd(zmm2, zmm31, zmm20);

                //zmm0, zmm1, zmm2 are used to load 24 elements from
                //matrix C.
                zmm3 = _mm512_loadu_pd((double const *)(temp_c + ldc * 5));
                zmm4 = _mm512_loadu_pd((double const *)(temp_c + ldc * 5 + 8));
                zmm5 = _mm512_loadu_pd((double const *)(temp_c + ldc * 5 + 16));
                //Compute C * Beta + fma result(AB*Alpha)
                zmm21 = _mm512_fmadd_pd(zmm3, zmm31, zmm21);
                zmm22 = _mm512_fmadd_pd(zmm4, zmm31, zmm22);
                zmm23 = _mm512_fmadd_pd(zmm5, zmm31, zmm23);

                //zmm0, zmm1, zmm2 are used to load 24 elements from
                //matrix C.
                zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 6));
                zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc * 6 + 8));
                zmm2 = _mm512_loadu_pd((double const *)(temp_c + ldc * 6 + 16));
                //Compute C * Beta + fma result(AB*Alpha)
                zmm24 = _mm512_fmadd_pd(zmm0, zmm31, zmm24);
                zmm25 = _mm512_fmadd_pd(zmm1, zmm31, zmm25);
                zmm26 = _mm512_fmadd_pd(zmm2, zmm31, zmm26);

                //zmm0, zmm1, zmm2 are used to load 24 elements from
                //matrix C.
                zmm3 = _mm512_loadu_pd((double const *)(temp_c + ldc * 7));
                zmm4 = _mm512_loadu_pd((double const *)(temp_c + ldc * 7 + 8));
                zmm5 = _mm512_loadu_pd((double const *)(temp_c + ldc * 7 + 16));
                //Compute C * Beta + fma result(AB*Alpha)
                zmm27 = _mm512_fmadd_pd(zmm3, zmm31, zmm27);
                zmm28 = _mm512_fmadd_pd(zmm4, zmm31, zmm28);
                zmm29 = _mm512_fmadd_pd(zmm5, zmm31, zmm29);

                //Store the result back to Matrix C.
                //Result is available in zmm6 to zmm29.
                _mm512_storeu_pd((double *)(temp_c), zmm6);
                _mm512_storeu_pd((double *)(temp_c + 8), zmm7);
                _mm512_storeu_pd((double *)(temp_c + 16), zmm8);
                //C matrix 2nd column
                _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                _mm512_storeu_pd((double *)(temp_c + ldc + 8), zmm10);
                _mm512_storeu_pd((double *)(temp_c + ldc + 16), zmm11);
                //C matrix 3rd column
                _mm512_storeu_pd((double *)(temp_c + ldc * 2), zmm12);
                _mm512_storeu_pd((double *)(temp_c + ldc * 2 + 8), zmm13);
                _mm512_storeu_pd((double *)(temp_c + ldc * 2 + 16), zmm14);
                //C matrix 4th column
                _mm512_storeu_pd((double *)(temp_c + ldc*3), zmm15);
                _mm512_storeu_pd((double *)(temp_c + ldc * 3 + 8), zmm16);
                _mm512_storeu_pd((double *)(temp_c + ldc * 3 + 16), zmm17);
                //C matrix 5th column
                _mm512_storeu_pd((double *)(temp_c + ldc*4), zmm18);
                _mm512_storeu_pd((double *)(temp_c + ldc*4 + 8), zmm19);
                _mm512_storeu_pd((double *)(temp_c + ldc*4 + 16), zmm20);
                //C matrix 6th column
                _mm512_storeu_pd((double *)(temp_c + ldc*5), zmm21);
                _mm512_storeu_pd((double *)(temp_c + ldc*5 + 8), zmm22);
                _mm512_storeu_pd((double *)(temp_c + ldc*5 + 16), zmm23);
                //C matrix 7th column
                _mm512_storeu_pd((double *)(temp_c + ldc*6), zmm24);
                _mm512_storeu_pd((double *)(temp_c + ldc*6 + 8), zmm25);
                _mm512_storeu_pd((double *)(temp_c + ldc*6 + 16), zmm26);
                //C matrix 8th column
                _mm512_storeu_pd((double *)(temp_c + ldc*7), zmm27);
                _mm512_storeu_pd((double *)(temp_c + ldc*7 + 8), zmm28);
                _mm512_storeu_pd((double *)(temp_c + ldc*7 + 16), zmm29);

                //Update temp_c and temp_a pointer to
                //respective offset.
                temp_c += D_MR;
                temp_a += D_MR;
            }

            dim_t m_rem = m_remainder;
            //Handles the edge case for m_remainder from 17 to 23.
            if(m_rem > 16)
            {
                uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                __mmask8 k0 = _load_mask8(&mask);
                //Clear out vector registers to hold fma result.
                //zmm6 to zmm29 holds fma result.
                //zmm0, zmm1, zmm2 are used to load elements from
                //A matrix.
                //zmm30 and zmm31 are alternatively used to broadcast element
                //from B matrix.
                zmm6 = _mm512_setzero_pd();
                zmm7 = _mm512_setzero_pd();
                zmm8 = _mm512_setzero_pd();
                zmm9 = _mm512_setzero_pd();
                zmm10 = _mm512_setzero_pd();
                zmm11 = _mm512_setzero_pd();
                zmm12 = _mm512_setzero_pd();
                zmm13 = _mm512_setzero_pd();
                zmm14 = _mm512_setzero_pd();
                zmm15 = _mm512_setzero_pd();
                zmm16 = _mm512_setzero_pd();
                zmm17 = _mm512_setzero_pd();
                zmm18 = _mm512_setzero_pd();
                zmm19 = _mm512_setzero_pd();
                zmm20 = _mm512_setzero_pd();
                zmm21 = _mm512_setzero_pd();
                zmm22 = _mm512_setzero_pd();
                zmm23 = _mm512_setzero_pd();
                zmm24 = _mm512_setzero_pd();
                zmm25 = _mm512_setzero_pd();
                zmm26 = _mm512_setzero_pd();
                zmm27 = _mm512_setzero_pd();
                zmm28 = _mm512_setzero_pd();
                zmm29 = _mm512_setzero_pd();
                zmm2 = _mm512_setzero_pd();
                /*
                    a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                    where alpha_val is not zero.
                    b. This loop operates with >16x8 block size
                    along n dimension for every D_NR columns of temp_b where
                    computing all D_MR rows of temp_a.
                    c. Same approach is used in remaining fringe cases.
                */
                zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                zmm1 = _mm512_loadu_pd((double const *)(temp_a + 8));
                zmm2 = _mm512_mask_loadu_pd (zmm2, k0, (double const *)(temp_a + 16));

                //Broadcast element from B matrix in zmm30
                zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                //Broadcast element from next column of B matrix in zmm31
                zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));
                //Compute A*B.
                zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);
                zmm8 = _mm512_fmadd_pd(zmm2, zmm30, zmm8);

                zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));
                //Compute A*B.
                zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);

                zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 3));
                //Compute A*B.
                zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);
                zmm13 = _mm512_fmadd_pd(zmm1, zmm30, zmm13);
                zmm14 = _mm512_fmadd_pd(zmm2, zmm30, zmm14);

                zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 4));
                //Compute A*B.
                zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);
                zmm17 = _mm512_fmadd_pd(zmm2, zmm31, zmm17);

                zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 5));
                //Compute A*B.
                zmm18 = _mm512_fmadd_pd(zmm0, zmm30, zmm18);
                zmm19 = _mm512_fmadd_pd(zmm1, zmm30, zmm19);
                zmm20 = _mm512_fmadd_pd(zmm2, zmm30, zmm20);

                zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 6));
                //Compute A*B.
                zmm21 = _mm512_fmadd_pd(zmm0, zmm31, zmm21);
                zmm22 = _mm512_fmadd_pd(zmm1, zmm31, zmm22);
                zmm23 = _mm512_fmadd_pd(zmm2, zmm31, zmm23);

                zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 7));
                //Compute A*B.
                zmm24 = _mm512_fmadd_pd(zmm0, zmm30, zmm24);
                zmm25 = _mm512_fmadd_pd(zmm1, zmm30, zmm25);
                zmm26 = _mm512_fmadd_pd(zmm2, zmm30, zmm26);
                //Compute A*B.
                zmm27 = _mm512_fmadd_pd(zmm0, zmm31, zmm27);
                zmm28 = _mm512_fmadd_pd(zmm1, zmm31, zmm28);
                zmm29 = _mm512_fmadd_pd(zmm2, zmm31, zmm29);

                //Broadcast Alpha into zmm0
                zmm0 = _mm512_set1_pd(alpha_val);
                //Scale fma result with Alpha.
                //Alpha * AB
                zmm6 = _mm512_mul_pd(zmm0, zmm6);
                zmm7 = _mm512_mul_pd(zmm0, zmm7);
                zmm8 = _mm512_mul_pd(zmm0, zmm8);
                zmm9 = _mm512_mul_pd(zmm0, zmm9);
                zmm10 = _mm512_mul_pd(zmm0, zmm10);
                zmm11 = _mm512_mul_pd(zmm0, zmm11);
                zmm12 = _mm512_mul_pd(zmm0, zmm12);
                zmm13 = _mm512_mul_pd(zmm0, zmm13);
                zmm14 = _mm512_mul_pd(zmm0, zmm14);
                zmm15 = _mm512_mul_pd(zmm0, zmm15);
                zmm16 = _mm512_mul_pd(zmm0, zmm16);
                zmm17 = _mm512_mul_pd(zmm0, zmm17);
                zmm18 = _mm512_mul_pd(zmm0, zmm18);
                zmm19 = _mm512_mul_pd(zmm0, zmm19);
                zmm20 = _mm512_mul_pd(zmm0, zmm20);
                zmm21 = _mm512_mul_pd(zmm0, zmm21);
                zmm22 = _mm512_mul_pd(zmm0, zmm22);
                zmm23 = _mm512_mul_pd(zmm0, zmm23);
                zmm24 = _mm512_mul_pd(zmm0, zmm24);
                zmm25 = _mm512_mul_pd(zmm0, zmm25);
                zmm26 = _mm512_mul_pd(zmm0, zmm26);
                zmm27 = _mm512_mul_pd(zmm0, zmm27);
                zmm28 = _mm512_mul_pd(zmm0, zmm28);
                zmm29 = _mm512_mul_pd(zmm0, zmm29);

                //Broadcast Beta into zmm31
                zmm31 = _mm512_set1_pd(beta_val);
                //zmm0, zmm1, zmm2 are used to load elements from
                //matrix C.
                zmm0 = _mm512_loadu_pd((double const *)(temp_c));
                zmm1 = _mm512_loadu_pd((double const *)(temp_c + 8));
                zmm2 = _mm512_mask_loadu_pd(zmm2, k0, (double const *)(temp_c + 16));
                //Compute C * Beta + fma result(AB*Alpha)
                zmm6 = _mm512_fmadd_pd(zmm0, zmm31, zmm6);
                zmm7 = _mm512_fmadd_pd(zmm1, zmm31, zmm7);
                zmm8 = _mm512_fmadd_pd(zmm2, zmm31, zmm8);

                //zmm0, zmm1, zmm2 are used to load elements from
                //matrix C.
                zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc ));
                zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc + 8));
                zmm2 = _mm512_mask_loadu_pd(zmm2, k0, (double const *)(temp_c + ldc + 16));
                //Compute C * Beta + fma result(AB*Alpha)
                zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);

                //zmm0, zmm1, zmm2 are used to load elements from
                //matrix C.
                zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 2));
                zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc * 2 + 8));
                zmm2 = _mm512_mask_loadu_pd(zmm2, k0, (double const *)(temp_c + ldc * 2 + 16));
                //Compute C * Beta + fma result(AB*Alpha)
                zmm12 = _mm512_fmadd_pd(zmm0, zmm31, zmm12);
                zmm13 = _mm512_fmadd_pd(zmm1, zmm31, zmm13);
                zmm14 = _mm512_fmadd_pd(zmm2, zmm31, zmm14);

                //zmm0, zmm1, zmm2 are used to load elements from
                //matrix C.
                zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 3));
                zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc * 3 + 8));
                zmm2 = _mm512_mask_loadu_pd(zmm2, k0, (double const *)(temp_c + ldc * 3 + 16));
                //Compute C * Beta + fma result(AB*Alpha)
                zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);
                zmm17 = _mm512_fmadd_pd(zmm2, zmm31, zmm17);

                //zmm0, zmm1, zmm2 are used to load elements from
                //matrix C.
                zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 4));
                zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc * 4 + 8));
                zmm2 = _mm512_mask_loadu_pd(zmm2, k0, (double const *)(temp_c + ldc * 4 + 16));
                //Compute C * Beta + fma result(AB*Alpha)
                zmm18 = _mm512_fmadd_pd(zmm0, zmm31, zmm18);
                zmm19 = _mm512_fmadd_pd(zmm1, zmm31, zmm19);
                zmm20 = _mm512_fmadd_pd(zmm2, zmm31, zmm20);

                //zmm0, zmm1, zmm2 are used to load elements from
                //matrix C.
                zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 5));
                zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc * 5 + 8));
                zmm2 = _mm512_mask_loadu_pd(zmm2, k0, (double const *)(temp_c + ldc * 5 + 16));
                //Compute C * Beta + fma result(AB*Alpha)
                zmm21 = _mm512_fmadd_pd(zmm0, zmm31, zmm21);
                zmm22 = _mm512_fmadd_pd(zmm1, zmm31, zmm22);
                zmm23 = _mm512_fmadd_pd(zmm2, zmm31, zmm23);

                //zmm0, zmm1, zmm2 are used to load elements from
                //matrix C.
                zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 6));
                zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc * 6 + 8));
                zmm2 = _mm512_mask_loadu_pd(zmm2, k0, (double const *)(temp_c + ldc * 6 + 16));
                //Compute C * Beta + fma result(AB*Alpha)
                zmm24 = _mm512_fmadd_pd(zmm0, zmm31, zmm24);
                zmm25 = _mm512_fmadd_pd(zmm1, zmm31, zmm25);
                zmm26 = _mm512_fmadd_pd(zmm2, zmm31, zmm26);

                //zmm0, zmm1, zmm2 are used to load elements from
                //matrix C.
                zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 7));
                zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc * 7 + 8));
                zmm2 = _mm512_mask_loadu_pd(zmm2, k0, (double const *)(temp_c + ldc * 7 + 16));
                //Compute C * Beta + fma result(AB*Alpha)
                zmm27 = _mm512_fmadd_pd(zmm0, zmm31, zmm27);
                zmm28 = _mm512_fmadd_pd(zmm1, zmm31, zmm28);
                zmm29 = _mm512_fmadd_pd(zmm2, zmm31, zmm29);

                //Store the result back to Matrix C.
                //Result is available in zmm6 to zmm29.
                _mm512_storeu_pd((double *)(temp_c), zmm6);
                _mm512_storeu_pd((double *)(temp_c + 8), zmm7);
                _mm512_mask_storeu_pd ((double *)(temp_c + 16), k0, zmm8);
                //C matrix 2nd column
                _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                _mm512_storeu_pd((double *)(temp_c + ldc + 8), zmm10);
                _mm512_mask_storeu_pd ((double *)(temp_c + ldc + 16), k0, zmm11);
                //C matrix 3rd column
                _mm512_storeu_pd((double *)(temp_c + ldc * 2), zmm12);
                _mm512_storeu_pd((double *)(temp_c + ldc * 2 + 8), zmm13);
                _mm512_mask_storeu_pd ((double *)(temp_c + ldc * 2 + 16), k0, zmm14);
				//C matrix 4th column
                _mm512_storeu_pd((double *)(temp_c + ldc*3), zmm15);
                _mm512_storeu_pd((double *)(temp_c + ldc * 3 + 8), zmm16);
                _mm512_mask_storeu_pd ((double *)(temp_c + ldc * 3 + 16), k0, zmm17);
				//C matrix 5th column
                _mm512_storeu_pd((double *)(temp_c + ldc*4), zmm18);
                _mm512_storeu_pd((double *)(temp_c + ldc*4 + 8), zmm19);
                _mm512_mask_storeu_pd ((double *)(temp_c + ldc*4 + 16), k0, zmm20);
				//C matrix 6th column
                _mm512_storeu_pd((double *)(temp_c + ldc*5), zmm21);
                _mm512_storeu_pd((double *)(temp_c + ldc*5 + 8), zmm22);
                _mm512_mask_storeu_pd ((double *)(temp_c + ldc*5 + 16), k0, zmm23);
				//C matrix 7th column
                _mm512_storeu_pd((double *)(temp_c + ldc*6), zmm24);
                _mm512_storeu_pd((double *)(temp_c + ldc*6 + 8), zmm25);
                _mm512_mask_storeu_pd ((double *)(temp_c + ldc*6 + 16), k0, zmm26);
				//C matrix 8th column
                _mm512_storeu_pd((double *)(temp_c + ldc*7), zmm27);
                _mm512_storeu_pd((double *)(temp_c + ldc*7 + 8), zmm28);
                _mm512_mask_storeu_pd ((double *)(temp_c + ldc*7 + 16), k0, zmm29);
            }
            //Handles the edge cases where m_remainder is from 9 to 16
            else if(m_rem > 8)
            {
                uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                if (mask == 0) mask = 0xff;
                __mmask8 k0 = _load_mask8(&mask);
                //Clear out vector registers to hold fma result.
                //zmm6 to zmm28 holds fma result.
                //zmm0, zmm1 are used to load elements from
                //A matrix.
                //zmm30 and zmm31 are alternatively used to broadcast element
                //from B matrix.
                zmm6 = _mm512_setzero_pd();
                zmm7 = _mm512_setzero_pd();
                zmm9 = _mm512_setzero_pd();
                zmm10 = _mm512_setzero_pd();
                zmm12 = _mm512_setzero_pd();
                zmm13 = _mm512_setzero_pd();
                zmm15 = _mm512_setzero_pd();
                zmm16 = _mm512_setzero_pd();
                zmm18 = _mm512_setzero_pd();
                zmm19 = _mm512_setzero_pd();
                zmm21 = _mm512_setzero_pd();
                zmm22 = _mm512_setzero_pd();
                zmm24 = _mm512_setzero_pd();
                zmm25 = _mm512_setzero_pd();
                zmm27 = _mm512_setzero_pd();
                zmm28 = _mm512_setzero_pd();
                zmm1 = _mm512_setzero_pd();
                /*
                    a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                    where alpha_val is not zero.
                    b. This loop operates with >8x8 block size
                    along n dimension for every D_NR columns of temp_b where
                    computing all D_MR rows of temp_a.
                    c. Same approach is used in remaining fringe cases.
                */
                zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_a + 8));

                //Broadcast element from B matrix in zmm30
                zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                //Broadcast element from next column of B matrix in zmm31
                zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));
                //Compute A*B.
                zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);

                zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));

                zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);

                zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 3));

                zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);
                zmm13 = _mm512_fmadd_pd(zmm1, zmm30, zmm13);

                zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 4));

                zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);

                zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 5));

                zmm18 = _mm512_fmadd_pd(zmm0, zmm30, zmm18);
                zmm19 = _mm512_fmadd_pd(zmm1, zmm30, zmm19);

                zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 6));

                zmm21 = _mm512_fmadd_pd(zmm0, zmm31, zmm21);
                zmm22 = _mm512_fmadd_pd(zmm1, zmm31, zmm22);

                zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 7));

                zmm24 = _mm512_fmadd_pd(zmm0, zmm30, zmm24);
                zmm25 = _mm512_fmadd_pd(zmm1, zmm30, zmm25);

                zmm27 = _mm512_fmadd_pd(zmm0, zmm31, zmm27);
                zmm28 = _mm512_fmadd_pd(zmm1, zmm31, zmm28);

                //Broadcast Alpha into zmm0
                zmm0 = _mm512_set1_pd(alpha_val);
                //Scale fma result with Alpha.
                //Alpha * AB
                zmm6 = _mm512_mul_pd(zmm0, zmm6);
                zmm7 = _mm512_mul_pd(zmm0, zmm7);
                zmm9 = _mm512_mul_pd(zmm0, zmm9);
                zmm10 = _mm512_mul_pd(zmm0, zmm10);
                zmm12 = _mm512_mul_pd(zmm0, zmm12);
                zmm13 = _mm512_mul_pd(zmm0, zmm13);
                zmm15 = _mm512_mul_pd(zmm0, zmm15);
                zmm16 = _mm512_mul_pd(zmm0, zmm16);
                zmm18 = _mm512_mul_pd(zmm0, zmm18);
                zmm19 = _mm512_mul_pd(zmm0, zmm19);
                zmm21 = _mm512_mul_pd(zmm0, zmm21);
                zmm22 = _mm512_mul_pd(zmm0, zmm22);
                zmm24 = _mm512_mul_pd(zmm0, zmm24);
                zmm25 = _mm512_mul_pd(zmm0, zmm25);
                zmm27 = _mm512_mul_pd(zmm0, zmm27);
                zmm28 = _mm512_mul_pd(zmm0, zmm28);

                //Broadcast Beta into zmm31
                zmm31 = _mm512_set1_pd(beta_val);
                //zmm0, zmm1 are used to load 24 elements from
                //matrix C.
                zmm0 = _mm512_loadu_pd((double const *)(temp_c));
                zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_c + 8));
                //Compute C * Beta + fma result(AB*Alpha)
                zmm6 = _mm512_fmadd_pd(zmm0, zmm31, zmm6);
                zmm7 = _mm512_fmadd_pd(zmm1, zmm31, zmm7);

                //zmm0, zmm1 are used to load 24 elements from
                //matrix C.
                zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc ));
                zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_c + ldc + 8));
                //Compute C * Beta + fma result(AB*Alpha)
                zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);

                //zmm0, zmm1 are used to load 24 elements from
                //matrix C.
                zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 2));
                zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_c + ldc * 2 + 8));
                //Compute C * Beta + fma result(AB*Alpha)
                zmm12 = _mm512_fmadd_pd(zmm0, zmm31, zmm12);
                zmm13 = _mm512_fmadd_pd(zmm1, zmm31, zmm13);

                //zmm0, zmm1 are used to load 24 elements from
                //matrix C.
                zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 3));
                zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_c + ldc * 3 + 8));
                //Compute C * Beta + fma result(AB*Alpha)
                zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);

                //zmm0, zmm1 are used to load 24 elements from
                //matrix C.
                zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 4));
                zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_c + ldc * 4 + 8));
                //Compute C * Beta + fma result(AB*Alpha)
                zmm18 = _mm512_fmadd_pd(zmm0, zmm31, zmm18);
                zmm19 = _mm512_fmadd_pd(zmm1, zmm31, zmm19);

                //zmm0, zmm1 are used to load 24 elements from
                //matrix C.
                zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 5));
                zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_c + ldc * 5 + 8));
                //Compute C * Beta + fma result(AB*Alpha)
                zmm21 = _mm512_fmadd_pd(zmm0, zmm31, zmm21);
                zmm22 = _mm512_fmadd_pd(zmm1, zmm31, zmm22);

                //zmm0, zmm1 are used to load 24 elements from
                //matrix C.
                zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 6));
                zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_c + ldc * 6 + 8));
                //Compute C * Beta + fma result(AB*Alpha)
                zmm24 = _mm512_fmadd_pd(zmm0, zmm31, zmm24);
                zmm25 = _mm512_fmadd_pd(zmm1, zmm31, zmm25);

                //zmm0, zmm1 are used to load 24 elements from
                //matrix C.
                zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 7));
                zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_c + ldc * 7 + 8));
                //Compute C * Beta + fma result(AB*Alpha)
                zmm27 = _mm512_fmadd_pd(zmm0, zmm31, zmm27);
                zmm28 = _mm512_fmadd_pd(zmm1, zmm31, zmm28);

                //Store the result back to Matrix C.
                //Result is available in zmm6 to zmm28.
                _mm512_storeu_pd((double *)(temp_c), zmm6);
                _mm512_mask_storeu_pd((double *)(temp_c + 8), k0, zmm7);
                //C matrix 2nd column
                _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                _mm512_mask_storeu_pd((double *)(temp_c + ldc + 8), k0, zmm10);
                //C matrix 3rd column
                _mm512_storeu_pd((double *)(temp_c + ldc * 2), zmm12);
                _mm512_mask_storeu_pd((double *)(temp_c + ldc * 2 + 8), k0, zmm13);
                //C matrix 4th column
                _mm512_storeu_pd((double *)(temp_c + ldc*3), zmm15);
                _mm512_mask_storeu_pd((double *)(temp_c + ldc * 3 + 8), k0, zmm16);
                //C matrix 5th column
                _mm512_storeu_pd((double *)(temp_c + ldc*4), zmm18);
                _mm512_mask_storeu_pd((double *)(temp_c + ldc*4 + 8), k0, zmm19);
                //C matrix 6th column
                _mm512_storeu_pd((double *)(temp_c + ldc*5), zmm21);
                _mm512_mask_storeu_pd((double *)(temp_c + ldc*5 + 8), k0, zmm22);
                //C matrix 7th column
                _mm512_storeu_pd((double *)(temp_c + ldc*6), zmm24);
                _mm512_mask_storeu_pd((double *)(temp_c + ldc*6 + 8), k0, zmm25);
                //C matrix 8th column
                _mm512_storeu_pd((double *)(temp_c + ldc*7), zmm27);
                _mm512_mask_storeu_pd((double *)(temp_c + ldc*7 + 8), k0, zmm28);
            }
            //Handles the edge case where m_remainder is from 1 to 8
            else if(m_rem > 0)
            {
                uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                if (mask == 0) mask = 0xff;
                __mmask8 k0 = _load_mask8(&mask);
                //Clear out vector registers to hold fma result.
                //zmm6 to zmm27 holds fma result.
                //zmm0 are used to load 8 elements from
                //A matrix.
                //zmm30 and zmm31 are alternatively used to broadcast element
                //from B matrix.
                zmm6 = _mm512_setzero_pd();
                zmm9 = _mm512_setzero_pd();
                zmm12 = _mm512_setzero_pd();
                zmm15 = _mm512_setzero_pd();
                zmm18 = _mm512_setzero_pd();
                zmm21 = _mm512_setzero_pd();
                zmm24 = _mm512_setzero_pd();
                zmm27 = _mm512_setzero_pd();
                zmm0 = _mm512_setzero_pd();
                /*
                    a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                    where alpha_val is not zero.
                    b. This loop operates with >1x8 block size
                    along n dimension for every D_NR columns of temp_b where
                    computing all D_MR rows of temp_a.
                    c. Same approach is used in remaining fringe cases.
                */
                zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_a));

                //Broadcast element from B matrix in zmm30
                zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                //Broadcast element from next column of B matrix in zmm31
                zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));
                //Compute A*B.
                zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);

                zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));

                zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);

                zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 3));

                zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);

                zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 4));

                zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);

                zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 5));

                zmm18 = _mm512_fmadd_pd(zmm0, zmm30, zmm18);

                zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 6));

                zmm21 = _mm512_fmadd_pd(zmm0, zmm31, zmm21);

                zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 7));

                zmm24 = _mm512_fmadd_pd(zmm0, zmm30, zmm24);
                zmm27 = _mm512_fmadd_pd(zmm0, zmm31, zmm27);

                //Broadcast Alpha into zmm0
                zmm0 = _mm512_set1_pd(alpha_val);
                //Scale fma result with Alpha.
                //Alpha * AB
                zmm6 = _mm512_mul_pd(zmm0, zmm6);
                zmm9 = _mm512_mul_pd(zmm0, zmm9);
                zmm12 = _mm512_mul_pd(zmm0, zmm12);
                zmm15 = _mm512_mul_pd(zmm0, zmm15);
                zmm18 = _mm512_mul_pd(zmm0, zmm18);
                zmm21 = _mm512_mul_pd(zmm0, zmm21);
                zmm24 = _mm512_mul_pd(zmm0, zmm24);
                zmm27 = _mm512_mul_pd(zmm0, zmm27);

                //Broadcast Beta into zmm31
                zmm31 = _mm512_set1_pd(beta_val);
                //zmm0 used to load 8 elements from
                //matrix C.
                zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_c));
                //Compute C * Beta + fma result(AB*Alpha)
                zmm6 = _mm512_fmadd_pd(zmm0, zmm31, zmm6);

                //zmm0 used to load 8 elements from
                //matrix C.
                zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_c + ldc ));
                //Compute C * Beta + fma result(AB*Alpha)
                zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);

                //zmm0 used to load 8 elements from
                //matrix C.
                zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_c + ldc * 2));
                //Compute C * Beta + fma result(AB*Alpha)
                zmm12 = _mm512_fmadd_pd(zmm0, zmm31, zmm12);

                //zmm0 used to load 8 elements from
                //matrix C.
                zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_c + ldc * 3));
                //Compute C * Beta + fma result(AB*Alpha)
                zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);

                //zmm0 used to load 8 elements from
                //matrix C.
                zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_c + ldc * 4));
                //Compute C * Beta + fma result(AB*Alpha)
                zmm18 = _mm512_fmadd_pd(zmm0, zmm31, zmm18);

                //zmm0 used to load 8 elements from
                //matrix C.
                zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_c + ldc * 5));
                //Compute C * Beta + fma result(AB*Alpha)
                zmm21 = _mm512_fmadd_pd(zmm0, zmm31, zmm21);

                //zmm0 used to load 8 elements from
                //matrix C.
                zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_c + ldc * 6));
                //Compute C * Beta + fma result(AB*Alpha)
                zmm24 = _mm512_fmadd_pd(zmm0, zmm31, zmm24);

                //zmm0 used to load 8 elements from
                //matrix C.
                zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_c + ldc * 7));
                //Compute C * Beta + fma result(AB*Alpha)
                zmm27 = _mm512_fmadd_pd(zmm0, zmm31, zmm27);

                //Store the result back to Matrix C.
                _mm512_mask_storeu_pd((double *)(temp_c), k0, zmm6);
                //C matrix 2nd column
                _mm512_mask_storeu_pd((double *)(temp_c + ldc), k0, zmm9);
                //C matrix 3rd column
                _mm512_mask_storeu_pd((double *)(temp_c + ldc * 2), k0, zmm12);
				//C matrix 4th column
                _mm512_mask_storeu_pd((double *)(temp_c + ldc*3), k0, zmm15);
				//C matrix 5th column
                _mm512_mask_storeu_pd((double *)(temp_c + ldc*4), k0, zmm18);
				//C matrix 6th column
                _mm512_mask_storeu_pd((double *)(temp_c + ldc*5), k0, zmm21);
				//C matrix 7th column
                _mm512_mask_storeu_pd((double *)(temp_c + ldc*6), k0, zmm24);
				//C matrix 8th column
                _mm512_mask_storeu_pd((double *)(temp_c + ldc*7), k0, zmm27);
            }
        }

        switch(n_remainder)
        {
            case 7:
            {
                double* temp_b = b + (n - n_remainder)*ldb;
                double* temp_a = a;
                double* temp_c = c + (n - n_remainder)*ldc;
                for(dim_t i = 0;i < (m-D_MR+1);i=i+D_MR)
                {
                    //Clear out vector registers to hold fma result.
                    //zmm6 to zmm26 holds fma result.
                    //zmm0, zmm1, zmm2 are used to load 24 elements from
                    //A matrix.
                    //zmm30 and zmm31 are alternatively used to broadcast element
                    //from B matrix.
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm8 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm10 = _mm512_setzero_pd();
                    zmm11 = _mm512_setzero_pd();
                    zmm12 = _mm512_setzero_pd();
                    zmm13 = _mm512_setzero_pd();
                    zmm14 = _mm512_setzero_pd();
                    zmm15 = _mm512_setzero_pd();
                    zmm16 = _mm512_setzero_pd();
                    zmm17 = _mm512_setzero_pd();
                    zmm18 = _mm512_setzero_pd();
                    zmm19 = _mm512_setzero_pd();
                    zmm20 = _mm512_setzero_pd();
                    zmm21 = _mm512_setzero_pd();
                    zmm22 = _mm512_setzero_pd();
                    zmm23 = _mm512_setzero_pd();
                    zmm24 = _mm512_setzero_pd();
                    zmm25 = _mm512_setzero_pd();
                    zmm26 = _mm512_setzero_pd();
                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with 24x7 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_a + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_a + 16));

                    _mm_prefetch((char*)( temp_a + 192), _MM_HINT_T0);
                    //Broadcast element from B matrix in zmm30
                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    //Broadcast element from B matrix in zmm31
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));
                    //Compute A*B.
                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm30, zmm8);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));
                    //Compute A*B.
                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                    zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 3));
                    //Compute A*B.
                    zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm30, zmm13);
                    zmm14 = _mm512_fmadd_pd(zmm2, zmm30, zmm14);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 4));
                    //Compute A*B.
                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                    zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);
                    zmm17 = _mm512_fmadd_pd(zmm2, zmm31, zmm17);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 5));
                    //Compute A*B.
                    zmm18 = _mm512_fmadd_pd(zmm0, zmm30, zmm18);
                    zmm19 = _mm512_fmadd_pd(zmm1, zmm30, zmm19);
                    zmm20 = _mm512_fmadd_pd(zmm2, zmm30, zmm20);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 6));
                    //Compute A*B.
                    zmm21 = _mm512_fmadd_pd(zmm0, zmm31, zmm21);
                    zmm22 = _mm512_fmadd_pd(zmm1, zmm31, zmm22);
                    zmm23 = _mm512_fmadd_pd(zmm2, zmm31, zmm23);

                    zmm24 = _mm512_fmadd_pd(zmm0, zmm30, zmm24);
                    zmm25 = _mm512_fmadd_pd(zmm1, zmm30, zmm25);
                    zmm26 = _mm512_fmadd_pd(zmm2, zmm30, zmm26);

                    //Broadcast Alpha into zmm0
                    zmm0 = _mm512_set1_pd(alpha_val);
                    //Scale fma result with Alpha.
                    //Alpha * AB
                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);
                    zmm8 = _mm512_mul_pd(zmm0, zmm8);
                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm10 = _mm512_mul_pd(zmm0, zmm10);
                    zmm11 = _mm512_mul_pd(zmm0, zmm11);
                    zmm12 = _mm512_mul_pd(zmm0, zmm12);
                    zmm13 = _mm512_mul_pd(zmm0, zmm13);
                    zmm14 = _mm512_mul_pd(zmm0, zmm14);
                    zmm15 = _mm512_mul_pd(zmm0, zmm15);
                    zmm16 = _mm512_mul_pd(zmm0, zmm16);
                    zmm17 = _mm512_mul_pd(zmm0, zmm17);
                    zmm18 = _mm512_mul_pd(zmm0, zmm18);
                    zmm19 = _mm512_mul_pd(zmm0, zmm19);
                    zmm20 = _mm512_mul_pd(zmm0, zmm20);
                    zmm21 = _mm512_mul_pd(zmm0, zmm21);
                    zmm22 = _mm512_mul_pd(zmm0, zmm22);
                    zmm23 = _mm512_mul_pd(zmm0, zmm23);
                    zmm24 = _mm512_mul_pd(zmm0, zmm24);
                    zmm25 = _mm512_mul_pd(zmm0, zmm25);
                    zmm26 = _mm512_mul_pd(zmm0, zmm26);

                    //Broadcast Beta into zmm31
                    zmm31 = _mm512_set1_pd(beta_val);
                    //zmm0, zmm1, zmm2 are used to load elements from
                    //matrix C.
                    zmm0 = _mm512_loadu_pd((double const *)(temp_c));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_c + 16));
                    //Compute C * Beta + fma result(AB*Alpha)
                    zmm6 = _mm512_fmadd_pd(zmm0, zmm31, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm31, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm31, zmm8);
                    //zmm0, zmm1, zmm2 are used to load elements from
                    //matrix C.
                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc ));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_c + ldc + 16));
                    //Compute C * Beta + fma result(AB*Alpha)
                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                    zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);
                    //zmm0, zmm1, zmm2 are used to load elements from
                    //matrix C.
                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 2));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc * 2 + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_c + ldc * 2 + 16));
                    //Compute C * Beta + fma result(AB*Alpha)
                    zmm12 = _mm512_fmadd_pd(zmm0, zmm31, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm31, zmm13);
                    zmm14 = _mm512_fmadd_pd(zmm2, zmm31, zmm14);
                    //zmm0, zmm1, zmm2 are used to load elements from
                    //matrix C.
                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 3));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc * 3 + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_c + ldc * 3 + 16));
                    //Compute C * Beta + fma result(AB*Alpha)
                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                    zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);
                    zmm17 = _mm512_fmadd_pd(zmm2, zmm31, zmm17);
                    //zmm0, zmm1, zmm2 are used to load elements from
                    //matrix C.
                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 4));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc * 4 + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_c + ldc * 4 + 16));
                    //Compute C * Beta + fma result(AB*Alpha)
                    zmm18 = _mm512_fmadd_pd(zmm0, zmm31, zmm18);
                    zmm19 = _mm512_fmadd_pd(zmm1, zmm31, zmm19);
                    zmm20 = _mm512_fmadd_pd(zmm2, zmm31, zmm20);
                    //zmm0, zmm1, zmm2 are used to load elements from
                    //matrix C.
                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 5));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc * 5 + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_c + ldc * 5 + 16));
                    //Compute C * Beta + fma result(AB*Alpha)
                    zmm21 = _mm512_fmadd_pd(zmm0, zmm31, zmm21);
                    zmm22 = _mm512_fmadd_pd(zmm1, zmm31, zmm22);
                    zmm23 = _mm512_fmadd_pd(zmm2, zmm31, zmm23);
                    //zmm0, zmm1, zmm2 are used to load elements from
                    //matrix C.
                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 6));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc * 6 + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_c + ldc * 6 + 16));
                    //Compute C * Beta + fma result(AB*Alpha)
                    zmm24 = _mm512_fmadd_pd(zmm0, zmm31, zmm24);
                    zmm25 = _mm512_fmadd_pd(zmm1, zmm31, zmm25);
                    zmm26 = _mm512_fmadd_pd(zmm2, zmm31, zmm26);

                    //Store the result back to Matrix C.
                    //Result is available in zmm6 to zmm26.
                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_storeu_pd((double *)(temp_c + 8), zmm7);
                    _mm512_storeu_pd((double *)(temp_c + 16), zmm8);
                    //C matrix 2nd column
                    _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                    _mm512_storeu_pd((double *)(temp_c + ldc + 8), zmm10);
                    _mm512_storeu_pd((double *)(temp_c + ldc + 16), zmm11);
                    //C matrix 3rd column
                    _mm512_storeu_pd((double *)(temp_c + ldc * 2), zmm12);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 2 + 8), zmm13);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 2 + 16), zmm14);
                    //C matrix 4th column
                    _mm512_storeu_pd((double *)(temp_c + ldc*3), zmm15);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 3 + 8), zmm16);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 3 + 16), zmm17);
                    //C matrix 5th column
                    _mm512_storeu_pd((double *)(temp_c + ldc*4), zmm18);
                    _mm512_storeu_pd((double *)(temp_c + ldc*4 + 8), zmm19);
                    _mm512_storeu_pd((double *)(temp_c + ldc*4 + 16), zmm20);
                    //C matrix 6th column
                    _mm512_storeu_pd((double *)(temp_c + ldc*5), zmm21);
                    _mm512_storeu_pd((double *)(temp_c + ldc*5 + 8), zmm22);
                    _mm512_storeu_pd((double *)(temp_c + ldc*5 + 16), zmm23);
                    //C matrix 7th column
                    _mm512_storeu_pd((double *)(temp_c + ldc*6), zmm24);
                    _mm512_storeu_pd((double *)(temp_c + ldc*6 + 8), zmm25);
                    _mm512_storeu_pd((double *)(temp_c + ldc*6 + 16), zmm26);

                    temp_c += D_MR;
                    temp_a += D_MR;
                }
                dim_t m_rem = m_remainder;
                //Handles the edge case where m_remainder is from 17 to 23
                if(m_rem > 16)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    __mmask8 k0 = _load_mask8(&mask);
                    //Clear out vector registers to hold fma result.
                    //zmm6 to zmm26 holds fma result.
                    //zmm0, zmm1, zmm2 are used to load elements from
                    //A matrix.
                    //zmm30 and zmm31 are alternatively used to broadcast element
                    //from B matrix.
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm8 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm10 = _mm512_setzero_pd();
                    zmm11 = _mm512_setzero_pd();
                    zmm12 = _mm512_setzero_pd();
                    zmm13 = _mm512_setzero_pd();
                    zmm14 = _mm512_setzero_pd();
                    zmm15 = _mm512_setzero_pd();
                    zmm16 = _mm512_setzero_pd();
                    zmm17 = _mm512_setzero_pd();
                    zmm18 = _mm512_setzero_pd();
                    zmm19 = _mm512_setzero_pd();
                    zmm20 = _mm512_setzero_pd();
                    zmm21 = _mm512_setzero_pd();
                    zmm22 = _mm512_setzero_pd();
                    zmm23 = _mm512_setzero_pd();
                    zmm24 = _mm512_setzero_pd();
                    zmm25 = _mm512_setzero_pd();
                    zmm26 = _mm512_setzero_pd();
                    zmm2 = _mm512_setzero_pd();

                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with (>16)x7 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_a + 8));
                    zmm2 = _mm512_mask_loadu_pd (zmm2, k0, (double const *)(temp_a + 16));

                    //Broadcast element from B matrix in zmm30
                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    //Broadcast element from next column of B matrix in zmm31
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));
                    //Compute A*B.
                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm30, zmm8);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));
                    //Compute A*B.
                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                    zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 3));
                    //Compute A*B.
                    zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm30, zmm13);
                    zmm14 = _mm512_fmadd_pd(zmm2, zmm30, zmm14);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 4));
                    //Compute A*B.
                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                    zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);
                    zmm17 = _mm512_fmadd_pd(zmm2, zmm31, zmm17);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 5));
                    //Compute A*B.
                    zmm18 = _mm512_fmadd_pd(zmm0, zmm30, zmm18);
                    zmm19 = _mm512_fmadd_pd(zmm1, zmm30, zmm19);
                    zmm20 = _mm512_fmadd_pd(zmm2, zmm30, zmm20);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 6));
                    //Compute A*B.
                    zmm21 = _mm512_fmadd_pd(zmm0, zmm31, zmm21);
                    zmm22 = _mm512_fmadd_pd(zmm1, zmm31, zmm22);
                    zmm23 = _mm512_fmadd_pd(zmm2, zmm31, zmm23);

                    zmm24 = _mm512_fmadd_pd(zmm0, zmm30, zmm24);
                    zmm25 = _mm512_fmadd_pd(zmm1, zmm30, zmm25);
                    zmm26 = _mm512_fmadd_pd(zmm2, zmm30, zmm26);

                    //Broadcast Alpha into zmm0
                    zmm0 = _mm512_set1_pd(alpha_val);
                    //Scale fma result with Alpha.
                    //Alpha * AB
                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);
                    zmm8 = _mm512_mul_pd(zmm0, zmm8);
                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm10 = _mm512_mul_pd(zmm0, zmm10);
                    zmm11 = _mm512_mul_pd(zmm0, zmm11);
                    zmm12 = _mm512_mul_pd(zmm0, zmm12);
                    zmm13 = _mm512_mul_pd(zmm0, zmm13);
                    zmm14 = _mm512_mul_pd(zmm0, zmm14);
                    zmm15 = _mm512_mul_pd(zmm0, zmm15);
                    zmm16 = _mm512_mul_pd(zmm0, zmm16);
                    zmm17 = _mm512_mul_pd(zmm0, zmm17);
                    zmm18 = _mm512_mul_pd(zmm0, zmm18);
                    zmm19 = _mm512_mul_pd(zmm0, zmm19);
                    zmm20 = _mm512_mul_pd(zmm0, zmm20);
                    zmm21 = _mm512_mul_pd(zmm0, zmm21);
                    zmm22 = _mm512_mul_pd(zmm0, zmm22);
                    zmm23 = _mm512_mul_pd(zmm0, zmm23);
                    zmm24 = _mm512_mul_pd(zmm0, zmm24);
                    zmm25 = _mm512_mul_pd(zmm0, zmm25);
                    zmm26 = _mm512_mul_pd(zmm0, zmm26);

                    //Broadcast Beta into zmm31
                    zmm31 = _mm512_set1_pd(beta_val);

                    //zmm0, zmm1, zmm2 are used to load elements from
                    //matrix C.
                    //Compute C * Beta + fma result(AB*Alpha)
                    zmm0 = _mm512_loadu_pd((double const *)(temp_c));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + 8));
                    zmm2 = _mm512_mask_loadu_pd(zmm2, k0, (double const *)(temp_c + 16));
                    zmm6 = _mm512_fmadd_pd(zmm0, zmm31, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm31, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm31, zmm8);

                    //zmm0, zmm1, zmm2 are used to load elements from
                    //matrix C.
                    //Compute C * Beta + fma result(AB*Alpha)
                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc ));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc + 8));
                    zmm2 = _mm512_mask_loadu_pd(zmm2, k0, (double const *)(temp_c + ldc + 16));
                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                    zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);

                    //zmm0, zmm1, zmm2 are used to load elements from
                    //matrix C.
                    //Compute C * Beta + fma result(AB*Alpha)
                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 2));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc * 2 + 8));
                    zmm2 = _mm512_mask_loadu_pd(zmm2, k0, (double const *)(temp_c + ldc * 2 + 16));
                    zmm12 = _mm512_fmadd_pd(zmm0, zmm31, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm31, zmm13);
                    zmm14 = _mm512_fmadd_pd(zmm2, zmm31, zmm14);

                    //zmm0, zmm1, zmm2 are used to load elements from
                    //matrix C.
                    //Compute C * Beta + fma result(AB*Alpha)
                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 3));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc * 3 + 8));
                    zmm2 = _mm512_mask_loadu_pd(zmm2, k0, (double const *)(temp_c + ldc * 3 + 16));
                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                    zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);
                    zmm17 = _mm512_fmadd_pd(zmm2, zmm31, zmm17);

                    //zmm0, zmm1, zmm2 are used to load elements from
                    //matrix C.
                    //Compute C * Beta + fma result(AB*Alpha)
                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 4));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc * 4 + 8));
                    zmm2 = _mm512_mask_loadu_pd(zmm2, k0, (double const *)(temp_c + ldc * 4 + 16));
                    zmm18 = _mm512_fmadd_pd(zmm0, zmm31, zmm18);
                    zmm19 = _mm512_fmadd_pd(zmm1, zmm31, zmm19);
                    zmm20 = _mm512_fmadd_pd(zmm2, zmm31, zmm20);

                    //zmm0, zmm1, zmm2 are used to load elements from
                    //matrix C.
                    //Compute C * Beta + fma result(AB*Alpha)
                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 5));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc * 5 + 8));
                    zmm2 = _mm512_mask_loadu_pd(zmm2, k0, (double const *)(temp_c + ldc * 5 + 16));
                    zmm21 = _mm512_fmadd_pd(zmm0, zmm31, zmm21);
                    zmm22 = _mm512_fmadd_pd(zmm1, zmm31, zmm22);
                    zmm23 = _mm512_fmadd_pd(zmm2, zmm31, zmm23);

                    //zmm0, zmm1, zmm2 are used to load elements from
                    //matrix C.
                    //Compute C * Beta + fma result(AB*Alpha)
                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 6));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc * 6 + 8));
                    zmm2 = _mm512_mask_loadu_pd(zmm2, k0, (double const *)(temp_c + ldc * 6 + 16));
                    zmm24 = _mm512_fmadd_pd(zmm0, zmm31, zmm24);
                    zmm25 = _mm512_fmadd_pd(zmm1, zmm31, zmm25);
                    zmm26 = _mm512_fmadd_pd(zmm2, zmm31, zmm26);

                    //Store the result back to Matrix C.
                    //Result is available in zmm6 to zmm26.
                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_storeu_pd((double *)(temp_c + 8), zmm7);
                    _mm512_mask_storeu_pd ((double *)(temp_c + 16), k0, zmm8);
                    //C matrix 2nd column
                    _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                    _mm512_storeu_pd((double *)(temp_c + ldc + 8), zmm10);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc + 16), k0, zmm11);
                    //C matrix 3rd column
                    _mm512_storeu_pd((double *)(temp_c + ldc * 2), zmm12);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 2 + 8), zmm13);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc * 2 + 16), k0, zmm14);
                    //C matrix 4th column
                    _mm512_storeu_pd((double *)(temp_c + ldc*3), zmm15);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 3 + 8), zmm16);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc * 3 + 16), k0, zmm17);
                    //C matrix 5th column
                    _mm512_storeu_pd((double *)(temp_c + ldc*4), zmm18);
                    _mm512_storeu_pd((double *)(temp_c + ldc*4 + 8), zmm19);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc*4 + 16), k0, zmm20);
                    //C matrix 6th column
                    _mm512_storeu_pd((double *)(temp_c + ldc*5), zmm21);
                    _mm512_storeu_pd((double *)(temp_c + ldc*5 + 8), zmm22);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc*5 + 16), k0, zmm23);
                    //C matrix 7th column
                    _mm512_storeu_pd((double *)(temp_c + ldc*6), zmm24);
                    _mm512_storeu_pd((double *)(temp_c + ldc*6 + 8), zmm25);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc*6 + 16), k0, zmm26);

                }
                //Handles the edge case where m_remadiner is from 9 to 16.
                else if(m_rem > 8)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    if (mask == 0) mask = 0xff;
                    __mmask8 k0 = _load_mask8(&mask);
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm10 = _mm512_setzero_pd();
                    zmm12 = _mm512_setzero_pd();
                    zmm13 = _mm512_setzero_pd();
                    zmm15 = _mm512_setzero_pd();
                    zmm16 = _mm512_setzero_pd();
                    zmm18 = _mm512_setzero_pd();
                    zmm19 = _mm512_setzero_pd();
                    zmm21 = _mm512_setzero_pd();
                    zmm22 = _mm512_setzero_pd();
                    zmm24 = _mm512_setzero_pd();
                    zmm25 = _mm512_setzero_pd();
                    zmm1 = _mm512_setzero_pd();

                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with (>8)x7 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_a + 8));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));
                    //Compute A*B.
                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));
                    //Compute A*B.
                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 3));
                    //Compute A*B.
                    zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm30, zmm13);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 4));
                    //Compute A*B.
                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                    zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 5));
                    //Compute A*B.
                    zmm18 = _mm512_fmadd_pd(zmm0, zmm30, zmm18);
                    zmm19 = _mm512_fmadd_pd(zmm1, zmm30, zmm19);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 6));
                    //Compute A*B.
                    zmm21 = _mm512_fmadd_pd(zmm0, zmm31, zmm21);
                    zmm22 = _mm512_fmadd_pd(zmm1, zmm31, zmm22);
                    //Compute A*B.
                    zmm24 = _mm512_fmadd_pd(zmm0, zmm30, zmm24);
                    zmm25 = _mm512_fmadd_pd(zmm1, zmm30, zmm25);

                    //Broadcast Alpha into zmm0
                    zmm0 = _mm512_set1_pd(alpha_val);
                    //Scale fma result with Alpha.
                    //Alpha * AB
                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);
                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm10 = _mm512_mul_pd(zmm0, zmm10);
                    zmm12 = _mm512_mul_pd(zmm0, zmm12);
                    zmm13 = _mm512_mul_pd(zmm0, zmm13);
                    zmm15 = _mm512_mul_pd(zmm0, zmm15);
                    zmm16 = _mm512_mul_pd(zmm0, zmm16);
                    zmm18 = _mm512_mul_pd(zmm0, zmm18);
                    zmm19 = _mm512_mul_pd(zmm0, zmm19);
                    zmm21 = _mm512_mul_pd(zmm0, zmm21);
                    zmm22 = _mm512_mul_pd(zmm0, zmm22);
                    zmm24 = _mm512_mul_pd(zmm0, zmm24);
                    zmm25 = _mm512_mul_pd(zmm0, zmm25);

                    //Broadcast Beta into zmm31
                    zmm31 = _mm512_set1_pd(beta_val);
                    //zmm0, zmm1 are used to load elements from
                    //matrix C.
                    //Compute C * Beta + fma result(AB*Alpha)
                    zmm0 = _mm512_loadu_pd((double const *)(temp_c));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_c + 8));
                    zmm6 = _mm512_fmadd_pd(zmm0, zmm31, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm31, zmm7);

                    //zmm0, zmm1 are used to load elements from
                    //matrix C.
                    //Compute C * Beta + fma result(AB*Alpha)
                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc ));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_c + ldc + 8));
                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);

                    //zmm0, zmm1 are used to load elements from
                    //matrix C.
                    //Compute C * Beta + fma result(AB*Alpha)
                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 2));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_c + ldc * 2 + 8));
                    zmm12 = _mm512_fmadd_pd(zmm0, zmm31, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm31, zmm13);

                    //zmm0, zmm1 are used to load elements from
                    //matrix C.
                    //Compute C * Beta + fma result(AB*Alpha)
                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 3));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_c + ldc * 3 + 8));
                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                    zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);

                    //zmm0, zmm1 are used to load elements from
                    //matrix C.
                    //Compute C * Beta + fma result(AB*Alpha)
                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 4));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_c + ldc * 4 + 8));
                    zmm18 = _mm512_fmadd_pd(zmm0, zmm31, zmm18);
                    zmm19 = _mm512_fmadd_pd(zmm1, zmm31, zmm19);

                    //zmm0, zmm1 are used to load elements from
                    //matrix C.
                    //Compute C * Beta + fma result(AB*Alpha)
                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 5));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_c + ldc * 5 + 8));
                    zmm21 = _mm512_fmadd_pd(zmm0, zmm31, zmm21);
                    zmm22 = _mm512_fmadd_pd(zmm1, zmm31, zmm22);

                    //zmm0, zmm1 are used to load elements from
                    //matrix C.
                    //Compute C * Beta + fma result(AB*Alpha)
                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 6));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_c + ldc * 6 + 8));
                    zmm24 = _mm512_fmadd_pd(zmm0, zmm31, zmm24);
                    zmm25 = _mm512_fmadd_pd(zmm1, zmm31, zmm25);

                    //Store the result back to Matrix C.
                    //Result is available in zmm6 to zmm25.
                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_mask_storeu_pd((double *)(temp_c + 8), k0, zmm7);
				    //C matrix 2nd column
                    _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc + 8), k0, zmm10);
				    //C matrix 3rd column
                    _mm512_storeu_pd((double *)(temp_c + ldc * 2), zmm12);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc * 2 + 8), k0, zmm13);
				    //C matrix 4th column
                    _mm512_storeu_pd((double *)(temp_c + ldc*3), zmm15);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc * 3 + 8), k0, zmm16);
				    //C matrix 5th column
                    _mm512_storeu_pd((double *)(temp_c + ldc*4), zmm18);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc*4 + 8), k0, zmm19);
				    //C matrix 6th column
                    _mm512_storeu_pd((double *)(temp_c + ldc*5), zmm21);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc*5 + 8), k0, zmm22);
				    //C matrix 7th column
                    _mm512_storeu_pd((double *)(temp_c + ldc*6), zmm24);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc*6 + 8), k0, zmm25);
                }
                else if(m_rem > 0)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    if (mask == 0) mask = 0xff;
                    __mmask8 k0 = _load_mask8(&mask);
                    zmm6 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm12 = _mm512_setzero_pd();
                    zmm15 = _mm512_setzero_pd();
                    zmm18 = _mm512_setzero_pd();
                    zmm21 = _mm512_setzero_pd();
                    zmm24 = _mm512_setzero_pd();
                    zmm0 = _mm512_setzero_pd();
                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with (>1)x7 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_a));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 3));

                    zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 4));

                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 5));

                    zmm18 = _mm512_fmadd_pd(zmm0, zmm30, zmm18);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 6));

                    zmm21 = _mm512_fmadd_pd(zmm0, zmm31, zmm21);

                    zmm24 = _mm512_fmadd_pd(zmm0, zmm30, zmm24);
                    //Broadcast Alpha into zmm0
                    zmm0 = _mm512_set1_pd(alpha_val);
                    //Scale fma result with Alpha.
                    //Alpha * AB
                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm12 = _mm512_mul_pd(zmm0, zmm12);
                    zmm15 = _mm512_mul_pd(zmm0, zmm15);
                    zmm18 = _mm512_mul_pd(zmm0, zmm18);
                    zmm21 = _mm512_mul_pd(zmm0, zmm21);
                    zmm24 = _mm512_mul_pd(zmm0, zmm24);
                    //Broadcast Beta into zmm31
                    zmm31 = _mm512_set1_pd(beta_val);
                    //zmm0 are used to load elements from
                    //matrix C.
                    //Compute C * Beta + fma result(AB*Alpha)
                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_c));
                    zmm6 = _mm512_fmadd_pd(zmm0, zmm31, zmm6);

                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_c + ldc ));
                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);

                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_c + ldc * 2));
                    zmm12 = _mm512_fmadd_pd(zmm0, zmm31, zmm12);

                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_c + ldc * 3));
                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);

                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_c + ldc * 4));
                    zmm18 = _mm512_fmadd_pd(zmm0, zmm31, zmm18);

                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_c + ldc * 5));
                    zmm21 = _mm512_fmadd_pd(zmm0, zmm31, zmm21);

                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_c + ldc * 6));
                    zmm24 = _mm512_fmadd_pd(zmm0, zmm31, zmm24);

                    //Store the result back to Matrix C.
                    //Result is available in zmm6 to zmm24.
                    _mm512_mask_storeu_pd((double *)(temp_c), k0, zmm6);
                    //C matrix 2nd column
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc), k0, zmm9);
                    //C matrix 3rd column
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc * 2), k0, zmm12);
                    //C matrix 4th column
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc*3), k0, zmm15);
                    //C matrix 5th column
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc*4), k0, zmm18);
                    //C matrix 6th column
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc*5), k0, zmm21);
                    //C matrix 7th column
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc*6), k0, zmm24);
                }
                break;
            }
            case 6:
            {
                double* temp_b = b + (n - n_remainder)*ldb;
                double* temp_a = a;
                double* temp_c = c + (n - n_remainder)*ldc;
                for(dim_t i = 0;i < (m-D_MR+1);i=i+D_MR)
                {
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm8 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm10 = _mm512_setzero_pd();
                    zmm11 = _mm512_setzero_pd();
                    zmm12 = _mm512_setzero_pd();
                    zmm13 = _mm512_setzero_pd();
                    zmm14 = _mm512_setzero_pd();
                    zmm15 = _mm512_setzero_pd();
                    zmm16 = _mm512_setzero_pd();
                    zmm17 = _mm512_setzero_pd();
                    zmm18 = _mm512_setzero_pd();
                    zmm19 = _mm512_setzero_pd();
                    zmm20 = _mm512_setzero_pd();
                    zmm21 = _mm512_setzero_pd();
                    zmm22 = _mm512_setzero_pd();
                    zmm23 = _mm512_setzero_pd();
                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with 24x6 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_a + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_a + 16));

                    _mm_prefetch((char*)( temp_a + 192), _MM_HINT_T0);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm30, zmm8);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                    zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 3));

                    zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm30, zmm13);
                    zmm14 = _mm512_fmadd_pd(zmm2, zmm30, zmm14);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 4));

                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                    zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);
                    zmm17 = _mm512_fmadd_pd(zmm2, zmm31, zmm17);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 5));

                    zmm18 = _mm512_fmadd_pd(zmm0, zmm30, zmm18);
                    zmm19 = _mm512_fmadd_pd(zmm1, zmm30, zmm19);
                    zmm20 = _mm512_fmadd_pd(zmm2, zmm30, zmm20);

                    zmm21 = _mm512_fmadd_pd(zmm0, zmm31, zmm21);
                    zmm22 = _mm512_fmadd_pd(zmm1, zmm31, zmm22);
                    zmm23 = _mm512_fmadd_pd(zmm2, zmm31, zmm23);

                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);
                    zmm8 = _mm512_mul_pd(zmm0, zmm8);

                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm10 = _mm512_mul_pd(zmm0, zmm10);
                    zmm11 = _mm512_mul_pd(zmm0, zmm11);

                    zmm12 = _mm512_mul_pd(zmm0, zmm12);
                    zmm13 = _mm512_mul_pd(zmm0, zmm13);
                    zmm14 = _mm512_mul_pd(zmm0, zmm14);

                    zmm15 = _mm512_mul_pd(zmm0, zmm15);
                    zmm16 = _mm512_mul_pd(zmm0, zmm16);
                    zmm17 = _mm512_mul_pd(zmm0, zmm17);

                    zmm18 = _mm512_mul_pd(zmm0, zmm18);
                    zmm19 = _mm512_mul_pd(zmm0, zmm19);
                    zmm20 = _mm512_mul_pd(zmm0, zmm20);

                    zmm21 = _mm512_mul_pd(zmm0, zmm21);
                    zmm22 = _mm512_mul_pd(zmm0, zmm22);
                    zmm23 = _mm512_mul_pd(zmm0, zmm23);

                    zmm31 = _mm512_set1_pd(beta_val);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_c + 16));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm31, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm31, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm31, zmm8);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc ));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_c + ldc + 16));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                    zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 2));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc * 2 + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_c + ldc * 2 + 16));

                    zmm12 = _mm512_fmadd_pd(zmm0, zmm31, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm31, zmm13);
                    zmm14 = _mm512_fmadd_pd(zmm2, zmm31, zmm14);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 3));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc * 3 + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_c + ldc * 3 + 16));

                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                    zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);
                    zmm17 = _mm512_fmadd_pd(zmm2, zmm31, zmm17);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 4));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc * 4 + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_c + ldc * 4 + 16));

                    zmm18 = _mm512_fmadd_pd(zmm0, zmm31, zmm18);
                    zmm19 = _mm512_fmadd_pd(zmm1, zmm31, zmm19);
                    zmm20 = _mm512_fmadd_pd(zmm2, zmm31, zmm20);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 5));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc * 5 + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_c + ldc * 5 + 16));

                    zmm21 = _mm512_fmadd_pd(zmm0, zmm31, zmm21);
                    zmm22 = _mm512_fmadd_pd(zmm1, zmm31, zmm22);
                    zmm23 = _mm512_fmadd_pd(zmm2, zmm31, zmm23);

                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_storeu_pd((double *)(temp_c + 8), zmm7);
                    _mm512_storeu_pd((double *)(temp_c + 16), zmm8);

                    _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                    _mm512_storeu_pd((double *)(temp_c + ldc + 8), zmm10);
                    _mm512_storeu_pd((double *)(temp_c + ldc + 16), zmm11);

                    _mm512_storeu_pd((double *)(temp_c + ldc * 2), zmm12);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 2 + 8), zmm13);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 2 + 16), zmm14);

                    _mm512_storeu_pd((double *)(temp_c + ldc*3), zmm15);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 3 + 8), zmm16);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 3 + 16), zmm17);

                    _mm512_storeu_pd((double *)(temp_c + ldc*4), zmm18);
                    _mm512_storeu_pd((double *)(temp_c + ldc*4 + 8), zmm19);
                    _mm512_storeu_pd((double *)(temp_c + ldc*4 + 16), zmm20);

                    _mm512_storeu_pd((double *)(temp_c + ldc*5), zmm21);
                    _mm512_storeu_pd((double *)(temp_c + ldc*5 + 8), zmm22);
                    _mm512_storeu_pd((double *)(temp_c + ldc*5 + 16), zmm23);

                    temp_c += D_MR;
                    temp_a += D_MR;
                }
                dim_t m_rem = m_remainder;
                if(m_rem > 16)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    __mmask8 k0 = _load_mask8(&mask);
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm8 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm10 = _mm512_setzero_pd();
                    zmm11 = _mm512_setzero_pd();
                    zmm12 = _mm512_setzero_pd();
                    zmm13 = _mm512_setzero_pd();
                    zmm14 = _mm512_setzero_pd();
                    zmm15 = _mm512_setzero_pd();
                    zmm16 = _mm512_setzero_pd();
                    zmm17 = _mm512_setzero_pd();
                    zmm18 = _mm512_setzero_pd();
                    zmm19 = _mm512_setzero_pd();
                    zmm20 = _mm512_setzero_pd();
                    zmm21 = _mm512_setzero_pd();
                    zmm22 = _mm512_setzero_pd();
                    zmm23 = _mm512_setzero_pd();
                    zmm2 = _mm512_setzero_pd();

                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with >16x6 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_a + 8));
                    zmm2 = _mm512_mask_loadu_pd (zmm2, k0, (double const *)(temp_a + 16));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm30, zmm8);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                    zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 3));

                    zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm30, zmm13);
                    zmm14 = _mm512_fmadd_pd(zmm2, zmm30, zmm14);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 4));

                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                    zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);
                    zmm17 = _mm512_fmadd_pd(zmm2, zmm31, zmm17);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 5));

                    zmm18 = _mm512_fmadd_pd(zmm0, zmm30, zmm18);
                    zmm19 = _mm512_fmadd_pd(zmm1, zmm30, zmm19);
                    zmm20 = _mm512_fmadd_pd(zmm2, zmm30, zmm20);

                    zmm21 = _mm512_fmadd_pd(zmm0, zmm31, zmm21);
                    zmm22 = _mm512_fmadd_pd(zmm1, zmm31, zmm22);
                    zmm23 = _mm512_fmadd_pd(zmm2, zmm31, zmm23);

                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);
                    zmm8 = _mm512_mul_pd(zmm0, zmm8);

                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm10 = _mm512_mul_pd(zmm0, zmm10);
                    zmm11 = _mm512_mul_pd(zmm0, zmm11);

                    zmm12 = _mm512_mul_pd(zmm0, zmm12);
                    zmm13 = _mm512_mul_pd(zmm0, zmm13);
                    zmm14 = _mm512_mul_pd(zmm0, zmm14);

                    zmm15 = _mm512_mul_pd(zmm0, zmm15);
                    zmm16 = _mm512_mul_pd(zmm0, zmm16);
                    zmm17 = _mm512_mul_pd(zmm0, zmm17);

                    zmm18 = _mm512_mul_pd(zmm0, zmm18);
                    zmm19 = _mm512_mul_pd(zmm0, zmm19);
                    zmm20 = _mm512_mul_pd(zmm0, zmm20);

                    zmm21 = _mm512_mul_pd(zmm0, zmm21);
                    zmm22 = _mm512_mul_pd(zmm0, zmm22);
                    zmm23 = _mm512_mul_pd(zmm0, zmm23);


                    zmm31 = _mm512_set1_pd(beta_val);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + 8));
                    zmm2 = _mm512_mask_loadu_pd(zmm2, k0, (double const *)(temp_c + 16));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm31, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm31, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm31, zmm8);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc ));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc + 8));
                    zmm2 = _mm512_mask_loadu_pd(zmm2, k0, (double const *)(temp_c + ldc + 16));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                    zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 2));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc * 2 + 8));
                    zmm2 = _mm512_mask_loadu_pd(zmm2, k0, (double const *)(temp_c + ldc * 2 + 16));

                    zmm12 = _mm512_fmadd_pd(zmm0, zmm31, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm31, zmm13);
                    zmm14 = _mm512_fmadd_pd(zmm2, zmm31, zmm14);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 3));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc * 3 + 8));
                    zmm2 = _mm512_mask_loadu_pd(zmm2, k0, (double const *)(temp_c + ldc * 3 + 16));

                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                    zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);
                    zmm17 = _mm512_fmadd_pd(zmm2, zmm31, zmm17);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 4));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc * 4 + 8));
                    zmm2 = _mm512_mask_loadu_pd(zmm2, k0, (double const *)(temp_c + ldc * 4 + 16));

                    zmm18 = _mm512_fmadd_pd(zmm0, zmm31, zmm18);
                    zmm19 = _mm512_fmadd_pd(zmm1, zmm31, zmm19);
                    zmm20 = _mm512_fmadd_pd(zmm2, zmm31, zmm20);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 5));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc * 5 + 8));
                    zmm2 = _mm512_mask_loadu_pd(zmm2, k0, (double const *)(temp_c + ldc * 5 + 16));

                    zmm21 = _mm512_fmadd_pd(zmm0, zmm31, zmm21);
                    zmm22 = _mm512_fmadd_pd(zmm1, zmm31, zmm22);
                    zmm23 = _mm512_fmadd_pd(zmm2, zmm31, zmm23);

                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_storeu_pd((double *)(temp_c + 8), zmm7);
                    _mm512_mask_storeu_pd ((double *)(temp_c + 16), k0, zmm8);

                    _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                    _mm512_storeu_pd((double *)(temp_c + ldc + 8), zmm10);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc + 16), k0, zmm11);

                    _mm512_storeu_pd((double *)(temp_c + ldc * 2), zmm12);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 2 + 8), zmm13);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc * 2 + 16), k0, zmm14);

                    _mm512_storeu_pd((double *)(temp_c + ldc*3), zmm15);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 3 + 8), zmm16);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc * 3 + 16), k0, zmm17);

                    _mm512_storeu_pd((double *)(temp_c + ldc*4), zmm18);
                    _mm512_storeu_pd((double *)(temp_c + ldc*4 + 8), zmm19);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc*4 + 16), k0, zmm20);

                    _mm512_storeu_pd((double *)(temp_c + ldc*5), zmm21);
                    _mm512_storeu_pd((double *)(temp_c + ldc*5 + 8), zmm22);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc*5 + 16), k0, zmm23);

                }
                else if(m_rem > 8)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    if (mask == 0) mask = 0xff;
                    __mmask8 k0 = _load_mask8(&mask);
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm10 = _mm512_setzero_pd();
                    zmm12 = _mm512_setzero_pd();
                    zmm13 = _mm512_setzero_pd();
                    zmm15 = _mm512_setzero_pd();
                    zmm16 = _mm512_setzero_pd();
                    zmm18 = _mm512_setzero_pd();
                    zmm19 = _mm512_setzero_pd();
                    zmm21 = _mm512_setzero_pd();
                    zmm22 = _mm512_setzero_pd();
                    zmm1 = _mm512_setzero_pd();

                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with >8x6 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_a + 8));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 3));

                    zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm30, zmm13);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 4));

                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                    zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 5));

                    zmm18 = _mm512_fmadd_pd(zmm0, zmm30, zmm18);
                    zmm19 = _mm512_fmadd_pd(zmm1, zmm30, zmm19);

                    zmm21 = _mm512_fmadd_pd(zmm0, zmm31, zmm21);
                    zmm22 = _mm512_fmadd_pd(zmm1, zmm31, zmm22);

                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);

                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm10 = _mm512_mul_pd(zmm0, zmm10);

                    zmm12 = _mm512_mul_pd(zmm0, zmm12);
                    zmm13 = _mm512_mul_pd(zmm0, zmm13);

                    zmm15 = _mm512_mul_pd(zmm0, zmm15);
                    zmm16 = _mm512_mul_pd(zmm0, zmm16);

                    zmm18 = _mm512_mul_pd(zmm0, zmm18);
                    zmm19 = _mm512_mul_pd(zmm0, zmm19);

                    zmm21 = _mm512_mul_pd(zmm0, zmm21);
                    zmm22 = _mm512_mul_pd(zmm0, zmm22);

                    zmm31 = _mm512_set1_pd(beta_val);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_c + 8));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm31, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm31, zmm7);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc ));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_c + ldc + 8));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 2));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_c + ldc * 2 + 8));

                    zmm12 = _mm512_fmadd_pd(zmm0, zmm31, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm31, zmm13);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 3));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_c + ldc * 3 + 8));

                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                    zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 4));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_c + ldc * 4 + 8));

                    zmm18 = _mm512_fmadd_pd(zmm0, zmm31, zmm18);
                    zmm19 = _mm512_fmadd_pd(zmm1, zmm31, zmm19);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 5));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_c + ldc * 5 + 8));

                    zmm21 = _mm512_fmadd_pd(zmm0, zmm31, zmm21);
                    zmm22 = _mm512_fmadd_pd(zmm1, zmm31, zmm22);

                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_mask_storeu_pd((double *)(temp_c + 8), k0, zmm7);

                    _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc + 8), k0, zmm10);

                    _mm512_storeu_pd((double *)(temp_c + ldc * 2), zmm12);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc * 2 + 8), k0, zmm13);

                    _mm512_storeu_pd((double *)(temp_c + ldc*3), zmm15);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc * 3 + 8), k0, zmm16);

                    _mm512_storeu_pd((double *)(temp_c + ldc*4), zmm18);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc*4 + 8), k0, zmm19);

                    _mm512_storeu_pd((double *)(temp_c + ldc*5), zmm21);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc*5 + 8), k0, zmm22);
                }
                else if(m_rem > 0)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    if (mask == 0) mask = 0xff;
                    __mmask8 k0 = _load_mask8(&mask);
                    zmm6 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm12 = _mm512_setzero_pd();
                    zmm15 = _mm512_setzero_pd();
                    zmm18 = _mm512_setzero_pd();
                    zmm21 = _mm512_setzero_pd();
                    zmm0 = _mm512_setzero_pd();


                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with >1x6 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_a));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));
                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));
                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 3));
                    zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 4));
                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 5));
                    zmm18 = _mm512_fmadd_pd(zmm0, zmm30, zmm18);

                    zmm21 = _mm512_fmadd_pd(zmm0, zmm31, zmm21);

                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm12 = _mm512_mul_pd(zmm0, zmm12);
                    zmm15 = _mm512_mul_pd(zmm0, zmm15);
                    zmm18 = _mm512_mul_pd(zmm0, zmm18);
                    zmm21 = _mm512_mul_pd(zmm0, zmm21);

                    zmm31 = _mm512_set1_pd(beta_val);

                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_c));
                    zmm6 = _mm512_fmadd_pd(zmm0, zmm31, zmm6);

                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_c + ldc ));
                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);

                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_c + ldc * 2));
                    zmm12 = _mm512_fmadd_pd(zmm0, zmm31, zmm12);

                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_c + ldc * 3));
                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);

                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_c + ldc * 4));
                    zmm18 = _mm512_fmadd_pd(zmm0, zmm31, zmm18);

                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_c + ldc * 5));
                    zmm21 = _mm512_fmadd_pd(zmm0, zmm31, zmm21);

                    _mm512_mask_storeu_pd((double *)(temp_c), k0, zmm6);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc), k0, zmm9);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc * 2), k0, zmm12);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc*3), k0, zmm15);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc*4), k0, zmm18);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc*5), k0, zmm21);
                }
                break;
            }
            case 5:
            {
                double* temp_b = b + (n - n_remainder)*ldb;
                double* temp_a = a;
                double* temp_c = c + (n - n_remainder)*ldc;
                for(dim_t i = 0;i < (m-D_MR+1);i=i+D_MR)
                {
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm8 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm10 = _mm512_setzero_pd();
                    zmm11 = _mm512_setzero_pd();
                    zmm12 = _mm512_setzero_pd();
                    zmm13 = _mm512_setzero_pd();
                    zmm14 = _mm512_setzero_pd();
                    zmm15 = _mm512_setzero_pd();
                    zmm16 = _mm512_setzero_pd();
                    zmm17 = _mm512_setzero_pd();
                    zmm18 = _mm512_setzero_pd();
                    zmm19 = _mm512_setzero_pd();
                    zmm20 = _mm512_setzero_pd();

                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with 24x5 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_a + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_a + 16));

                    _mm_prefetch((char*)( temp_a + 192), _MM_HINT_T0);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm30, zmm8);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                    zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 3));

                    zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm30, zmm13);
                    zmm14 = _mm512_fmadd_pd(zmm2, zmm30, zmm14);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 4));

                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                    zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);
                    zmm17 = _mm512_fmadd_pd(zmm2, zmm31, zmm17);

                    zmm18 = _mm512_fmadd_pd(zmm0, zmm30, zmm18);
                    zmm19 = _mm512_fmadd_pd(zmm1, zmm30, zmm19);
                    zmm20 = _mm512_fmadd_pd(zmm2, zmm30, zmm20);


                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);
                    zmm8 = _mm512_mul_pd(zmm0, zmm8);

                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm10 = _mm512_mul_pd(zmm0, zmm10);
                    zmm11 = _mm512_mul_pd(zmm0, zmm11);

                    zmm12 = _mm512_mul_pd(zmm0, zmm12);
                    zmm13 = _mm512_mul_pd(zmm0, zmm13);
                    zmm14 = _mm512_mul_pd(zmm0, zmm14);

                    zmm15 = _mm512_mul_pd(zmm0, zmm15);
                    zmm16 = _mm512_mul_pd(zmm0, zmm16);
                    zmm17 = _mm512_mul_pd(zmm0, zmm17);

                    zmm18 = _mm512_mul_pd(zmm0, zmm18);
                    zmm19 = _mm512_mul_pd(zmm0, zmm19);
                    zmm20 = _mm512_mul_pd(zmm0, zmm20);

                    zmm31 = _mm512_set1_pd(beta_val);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_c + 16));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm31, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm31, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm31, zmm8);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc ));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_c + ldc + 16));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                    zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 2));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc * 2 + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_c + ldc * 2 + 16));

                    zmm12 = _mm512_fmadd_pd(zmm0, zmm31, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm31, zmm13);
                    zmm14 = _mm512_fmadd_pd(zmm2, zmm31, zmm14);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 3));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc * 3 + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_c + ldc * 3 + 16));

                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                    zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);
                    zmm17 = _mm512_fmadd_pd(zmm2, zmm31, zmm17);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 4));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc * 4 + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_c + ldc * 4 + 16));

                    zmm18 = _mm512_fmadd_pd(zmm0, zmm31, zmm18);
                    zmm19 = _mm512_fmadd_pd(zmm1, zmm31, zmm19);
                    zmm20 = _mm512_fmadd_pd(zmm2, zmm31, zmm20);

                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_storeu_pd((double *)(temp_c + 8), zmm7);
                    _mm512_storeu_pd((double *)(temp_c + 16), zmm8);

                    _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                    _mm512_storeu_pd((double *)(temp_c + ldc + 8), zmm10);
                    _mm512_storeu_pd((double *)(temp_c + ldc + 16), zmm11);

                    _mm512_storeu_pd((double *)(temp_c + ldc * 2), zmm12);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 2 + 8), zmm13);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 2 + 16), zmm14);

                    _mm512_storeu_pd((double *)(temp_c + ldc*3), zmm15);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 3 + 8), zmm16);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 3 + 16), zmm17);

                    _mm512_storeu_pd((double *)(temp_c + ldc*4), zmm18);
                    _mm512_storeu_pd((double *)(temp_c + ldc*4 + 8), zmm19);
                    _mm512_storeu_pd((double *)(temp_c + ldc*4 + 16), zmm20);

                    temp_c += D_MR;
                    temp_a += D_MR;
                }
                dim_t m_rem = m_remainder;
                if(m_rem > 16)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    __mmask8 k0 = _load_mask8(&mask);
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm8 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm10 = _mm512_setzero_pd();
                    zmm11 = _mm512_setzero_pd();
                    zmm12 = _mm512_setzero_pd();
                    zmm13 = _mm512_setzero_pd();
                    zmm14 = _mm512_setzero_pd();
                    zmm15 = _mm512_setzero_pd();
                    zmm16 = _mm512_setzero_pd();
                    zmm17 = _mm512_setzero_pd();
                    zmm18 = _mm512_setzero_pd();
                    zmm19 = _mm512_setzero_pd();
                    zmm20 = _mm512_setzero_pd();
                    zmm2 = _mm512_setzero_pd();

                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with 8x6 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_a + 8));
                    zmm2 = _mm512_mask_loadu_pd (zmm2, k0, (double const *)(temp_a + 16));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm30, zmm8);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                    zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 3));

                    zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm30, zmm13);
                    zmm14 = _mm512_fmadd_pd(zmm2, zmm30, zmm14);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 4));

                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                    zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);
                    zmm17 = _mm512_fmadd_pd(zmm2, zmm31, zmm17);

                    zmm18 = _mm512_fmadd_pd(zmm0, zmm30, zmm18);
                    zmm19 = _mm512_fmadd_pd(zmm1, zmm30, zmm19);
                    zmm20 = _mm512_fmadd_pd(zmm2, zmm30, zmm20);

                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);
                    zmm8 = _mm512_mul_pd(zmm0, zmm8);

                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm10 = _mm512_mul_pd(zmm0, zmm10);
                    zmm11 = _mm512_mul_pd(zmm0, zmm11);

                    zmm12 = _mm512_mul_pd(zmm0, zmm12);
                    zmm13 = _mm512_mul_pd(zmm0, zmm13);
                    zmm14 = _mm512_mul_pd(zmm0, zmm14);

                    zmm15 = _mm512_mul_pd(zmm0, zmm15);
                    zmm16 = _mm512_mul_pd(zmm0, zmm16);
                    zmm17 = _mm512_mul_pd(zmm0, zmm17);

                    zmm18 = _mm512_mul_pd(zmm0, zmm18);
                    zmm19 = _mm512_mul_pd(zmm0, zmm19);
                    zmm20 = _mm512_mul_pd(zmm0, zmm20);

                    zmm31 = _mm512_set1_pd(beta_val);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + 8));
                    zmm2 = _mm512_mask_loadu_pd(zmm2, k0, (double const *)(temp_c + 16));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm31, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm31, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm31, zmm8);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc ));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc + 8));
                    zmm2 = _mm512_mask_loadu_pd(zmm2, k0, (double const *)(temp_c + ldc + 16));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                    zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 2));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc * 2 + 8));
                    zmm2 = _mm512_mask_loadu_pd(zmm2, k0, (double const *)(temp_c + ldc * 2 + 16));

                    zmm12 = _mm512_fmadd_pd(zmm0, zmm31, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm31, zmm13);
                    zmm14 = _mm512_fmadd_pd(zmm2, zmm31, zmm14);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 3));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc * 3 + 8));
                    zmm2 = _mm512_mask_loadu_pd(zmm2, k0, (double const *)(temp_c + ldc * 3 + 16));

                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                    zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);
                    zmm17 = _mm512_fmadd_pd(zmm2, zmm31, zmm17);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 4));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc * 4 + 8));
                    zmm2 = _mm512_mask_loadu_pd(zmm2, k0, (double const *)(temp_c + ldc * 4 + 16));

                    zmm18 = _mm512_fmadd_pd(zmm0, zmm31, zmm18);
                    zmm19 = _mm512_fmadd_pd(zmm1, zmm31, zmm19);
                    zmm20 = _mm512_fmadd_pd(zmm2, zmm31, zmm20);

                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_storeu_pd((double *)(temp_c + 8), zmm7);
                    _mm512_mask_storeu_pd ((double *)(temp_c + 16), k0, zmm8);

                    _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                    _mm512_storeu_pd((double *)(temp_c + ldc + 8), zmm10);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc + 16), k0, zmm11);

                    _mm512_storeu_pd((double *)(temp_c + ldc * 2), zmm12);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 2 + 8), zmm13);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc * 2 + 16), k0, zmm14);

                    _mm512_storeu_pd((double *)(temp_c + ldc*3), zmm15);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 3 + 8), zmm16);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc * 3 + 16), k0, zmm17);

                    _mm512_storeu_pd((double *)(temp_c + ldc*4), zmm18);
                    _mm512_storeu_pd((double *)(temp_c + ldc*4 + 8), zmm19);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc*4 + 16), k0, zmm20);

                }
                else if(m_rem > 8)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    if (mask == 0) mask = 0xff;
                    __mmask8 k0 = _load_mask8(&mask);
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm10 = _mm512_setzero_pd();
                    zmm12 = _mm512_setzero_pd();
                    zmm13 = _mm512_setzero_pd();
                    zmm15 = _mm512_setzero_pd();
                    zmm16 = _mm512_setzero_pd();
                    zmm18 = _mm512_setzero_pd();
                    zmm19 = _mm512_setzero_pd();
                    zmm1 = _mm512_setzero_pd();

                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with >8x6 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_a + 8));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 3));

                    zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm30, zmm13);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 4));

                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                    zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);

                    zmm18 = _mm512_fmadd_pd(zmm0, zmm30, zmm18);
                    zmm19 = _mm512_fmadd_pd(zmm1, zmm30, zmm19);

                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);

                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm10 = _mm512_mul_pd(zmm0, zmm10);

                    zmm12 = _mm512_mul_pd(zmm0, zmm12);
                    zmm13 = _mm512_mul_pd(zmm0, zmm13);

                    zmm15 = _mm512_mul_pd(zmm0, zmm15);
                    zmm16 = _mm512_mul_pd(zmm0, zmm16);

                    zmm18 = _mm512_mul_pd(zmm0, zmm18);
                    zmm19 = _mm512_mul_pd(zmm0, zmm19);

                    zmm31 = _mm512_set1_pd(beta_val);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_c + 8));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm31, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm31, zmm7);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc ));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_c + ldc + 8));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 2));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_c + ldc * 2 + 8));

                    zmm12 = _mm512_fmadd_pd(zmm0, zmm31, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm31, zmm13);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 3));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_c + ldc * 3 + 8));

                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                    zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 4));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_c + ldc * 4 + 8));

                    zmm18 = _mm512_fmadd_pd(zmm0, zmm31, zmm18);
                    zmm19 = _mm512_fmadd_pd(zmm1, zmm31, zmm19);

                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_mask_storeu_pd((double *)(temp_c + 8), k0, zmm7);

                    _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc + 8), k0, zmm10);

                    _mm512_storeu_pd((double *)(temp_c + ldc * 2), zmm12);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc * 2 + 8), k0, zmm13);

                    _mm512_storeu_pd((double *)(temp_c + ldc*3), zmm15);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc * 3 + 8), k0, zmm16);

                    _mm512_storeu_pd((double *)(temp_c + ldc*4), zmm18);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc*4 + 8), k0, zmm19);

                }
                else if(m_rem > 0)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    if (mask == 0) mask = 0xff;
                    __mmask8 k0 = _load_mask8(&mask);
                    zmm6 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm12 = _mm512_setzero_pd();
                    zmm15 = _mm512_setzero_pd();
                    zmm18 = _mm512_setzero_pd();
                    zmm0 = _mm512_setzero_pd();

                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with >1x6 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_a));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));
                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));
                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 3));
                    zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 4));
                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);

                    zmm18 = _mm512_fmadd_pd(zmm0, zmm30, zmm18);

                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm12 = _mm512_mul_pd(zmm0, zmm12);
                    zmm15 = _mm512_mul_pd(zmm0, zmm15);
                    zmm18 = _mm512_mul_pd(zmm0, zmm18);

                    zmm31 = _mm512_set1_pd(beta_val);

                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_c));
                    zmm6 = _mm512_fmadd_pd(zmm0, zmm31, zmm6);

                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_c + ldc ));
                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);

                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_c + ldc * 2));
                    zmm12 = _mm512_fmadd_pd(zmm0, zmm31, zmm12);

                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_c + ldc * 3));
                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);

                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_c + ldc * 4));
                    zmm18 = _mm512_fmadd_pd(zmm0, zmm31, zmm18);

                    _mm512_mask_storeu_pd((double *)(temp_c), k0, zmm6);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc), k0, zmm9);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc * 2), k0, zmm12);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc*3), k0, zmm15);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc*4), k0, zmm18);
                }
                break;
            }
            case 4:
            {
                double* temp_b = b + (n - n_remainder)*ldb;
                double* temp_a = a;
                double* temp_c = c + (n - n_remainder)*ldc;
                for(dim_t i = 0;i < (m-D_MR+1);i=i+D_MR)
                {
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm8 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm10 = _mm512_setzero_pd();
                    zmm11 = _mm512_setzero_pd();
                    zmm12 = _mm512_setzero_pd();
                    zmm13 = _mm512_setzero_pd();
                    zmm14 = _mm512_setzero_pd();
                    zmm15 = _mm512_setzero_pd();
                    zmm16 = _mm512_setzero_pd();
                    zmm17 = _mm512_setzero_pd();
                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with 24x4 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_a + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_a + 16));

                    _mm_prefetch((char*)( temp_a + 192), _MM_HINT_T0);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm30, zmm8);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                    zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 3));

                    zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm30, zmm13);
                    zmm14 = _mm512_fmadd_pd(zmm2, zmm30, zmm14);

                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                    zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);
                    zmm17 = _mm512_fmadd_pd(zmm2, zmm31, zmm17);


                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);
                    zmm8 = _mm512_mul_pd(zmm0, zmm8);

                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm10 = _mm512_mul_pd(zmm0, zmm10);
                    zmm11 = _mm512_mul_pd(zmm0, zmm11);

                    zmm12 = _mm512_mul_pd(zmm0, zmm12);
                    zmm13 = _mm512_mul_pd(zmm0, zmm13);
                    zmm14 = _mm512_mul_pd(zmm0, zmm14);

                    zmm15 = _mm512_mul_pd(zmm0, zmm15);
                    zmm16 = _mm512_mul_pd(zmm0, zmm16);
                    zmm17 = _mm512_mul_pd(zmm0, zmm17);

                    zmm31 = _mm512_set1_pd(beta_val);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_c + 16));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm31, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm31, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm31, zmm8);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc ));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_c + ldc + 16));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                    zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 2));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc * 2 + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_c + ldc * 2 + 16));

                    zmm12 = _mm512_fmadd_pd(zmm0, zmm31, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm31, zmm13);
                    zmm14 = _mm512_fmadd_pd(zmm2, zmm31, zmm14);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 3));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc * 3 + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_c + ldc * 3 + 16));

                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                    zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);
                    zmm17 = _mm512_fmadd_pd(zmm2, zmm31, zmm17);

                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_storeu_pd((double *)(temp_c + 8), zmm7);
                    _mm512_storeu_pd((double *)(temp_c + 16), zmm8);

                    _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                    _mm512_storeu_pd((double *)(temp_c + ldc + 8), zmm10);
                    _mm512_storeu_pd((double *)(temp_c + ldc + 16), zmm11);

                    _mm512_storeu_pd((double *)(temp_c + ldc * 2), zmm12);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 2 + 8), zmm13);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 2 + 16), zmm14);

                    _mm512_storeu_pd((double *)(temp_c + ldc*3), zmm15);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 3 + 8), zmm16);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 3 + 16), zmm17);

                    temp_c += D_MR;
                    temp_a += D_MR;
                }
                dim_t m_rem = m_remainder;
                if(m_rem > 16)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    __mmask8 k0 = _load_mask8(&mask);
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm8 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm10 = _mm512_setzero_pd();
                    zmm11 = _mm512_setzero_pd();
                    zmm12 = _mm512_setzero_pd();
                    zmm13 = _mm512_setzero_pd();
                    zmm14 = _mm512_setzero_pd();
                    zmm15 = _mm512_setzero_pd();
                    zmm16 = _mm512_setzero_pd();
                    zmm17 = _mm512_setzero_pd();
                    zmm2 = _mm512_setzero_pd();
                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with >16x4 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_a + 8));
                    zmm2 = _mm512_mask_loadu_pd (zmm2, k0, (double const *)(temp_a + 16));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm30, zmm8);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                    zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 3));

                    zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm30, zmm13);
                    zmm14 = _mm512_fmadd_pd(zmm2, zmm30, zmm14);

                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                    zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);
                    zmm17 = _mm512_fmadd_pd(zmm2, zmm31, zmm17);

                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);
                    zmm8 = _mm512_mul_pd(zmm0, zmm8);

                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm10 = _mm512_mul_pd(zmm0, zmm10);
                    zmm11 = _mm512_mul_pd(zmm0, zmm11);

                    zmm12 = _mm512_mul_pd(zmm0, zmm12);
                    zmm13 = _mm512_mul_pd(zmm0, zmm13);
                    zmm14 = _mm512_mul_pd(zmm0, zmm14);

                    zmm15 = _mm512_mul_pd(zmm0, zmm15);
                    zmm16 = _mm512_mul_pd(zmm0, zmm16);
                    zmm17 = _mm512_mul_pd(zmm0, zmm17);

                    zmm31 = _mm512_set1_pd(beta_val);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + 8));
                    zmm2 = _mm512_mask_loadu_pd(zmm2, k0, (double const *)(temp_c + 16));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm31, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm31, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm31, zmm8);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc ));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc + 8));
                    zmm2 = _mm512_mask_loadu_pd(zmm2, k0, (double const *)(temp_c + ldc + 16));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                    zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 2));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc * 2 + 8));
                    zmm2 = _mm512_mask_loadu_pd(zmm2, k0, (double const *)(temp_c + ldc * 2 + 16));

                    zmm12 = _mm512_fmadd_pd(zmm0, zmm31, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm31, zmm13);
                    zmm14 = _mm512_fmadd_pd(zmm2, zmm31, zmm14);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 3));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc * 3 + 8));
                    zmm2 = _mm512_mask_loadu_pd(zmm2, k0, (double const *)(temp_c + ldc * 3 + 16));

                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                    zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);
                    zmm17 = _mm512_fmadd_pd(zmm2, zmm31, zmm17);


                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_storeu_pd((double *)(temp_c + 8), zmm7);
                    _mm512_mask_storeu_pd ((double *)(temp_c + 16), k0, zmm8);

                    _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                    _mm512_storeu_pd((double *)(temp_c + ldc + 8), zmm10);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc + 16), k0, zmm11);

                    _mm512_storeu_pd((double *)(temp_c + ldc * 2), zmm12);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 2 + 8), zmm13);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc * 2 + 16), k0, zmm14);

                    _mm512_storeu_pd((double *)(temp_c + ldc*3), zmm15);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 3 + 8), zmm16);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc * 3 + 16), k0, zmm17);

                }
                else if(m_rem > 8)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    if (mask == 0) mask = 0xff;
                    __mmask8 k0 = _load_mask8(&mask);
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm10 = _mm512_setzero_pd();
                    zmm12 = _mm512_setzero_pd();
                    zmm13 = _mm512_setzero_pd();
                    zmm15 = _mm512_setzero_pd();
                    zmm16 = _mm512_setzero_pd();
                    zmm1 = _mm512_setzero_pd();

                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with >8x4 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_a + 8));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 3));

                    zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm30, zmm13);

                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                    zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);

                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);

                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm10 = _mm512_mul_pd(zmm0, zmm10);

                    zmm12 = _mm512_mul_pd(zmm0, zmm12);
                    zmm13 = _mm512_mul_pd(zmm0, zmm13);

                    zmm15 = _mm512_mul_pd(zmm0, zmm15);
                    zmm16 = _mm512_mul_pd(zmm0, zmm16);

                    zmm31 = _mm512_set1_pd(beta_val);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_c + 8));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm31, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm31, zmm7);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc ));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_c + ldc + 8));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 2));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_c + ldc * 2 + 8));

                    zmm12 = _mm512_fmadd_pd(zmm0, zmm31, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm31, zmm13);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 3));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_c + ldc * 3 + 8));

                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                    zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);

                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_mask_storeu_pd((double *)(temp_c + 8), k0, zmm7);

                    _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc + 8), k0, zmm10);

                    _mm512_storeu_pd((double *)(temp_c + ldc * 2), zmm12);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc * 2 + 8), k0, zmm13);

                    _mm512_storeu_pd((double *)(temp_c + ldc*3), zmm15);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc * 3 + 8), k0, zmm16);

                }
                else if(m_rem > 0)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    if (mask == 0) mask = 0xff;
                    __mmask8 k0 = _load_mask8(&mask);
                    zmm6 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm12 = _mm512_setzero_pd();
                    zmm15 = _mm512_setzero_pd();
                    zmm0 = _mm512_setzero_pd();

                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with >1x4 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_a));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));
                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));
                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 3));
                    zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);

                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);

                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm12 = _mm512_mul_pd(zmm0, zmm12);
                    zmm15 = _mm512_mul_pd(zmm0, zmm15);

                    zmm31 = _mm512_set1_pd(beta_val);

                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_c));
                    zmm6 = _mm512_fmadd_pd(zmm0, zmm31, zmm6);

                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_c + ldc ));
                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);

                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_c + ldc * 2));
                    zmm12 = _mm512_fmadd_pd(zmm0, zmm31, zmm12);

                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_c + ldc * 3));
                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);

                    _mm512_mask_storeu_pd((double *)(temp_c), k0, zmm6);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc), k0, zmm9);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc * 2), k0, zmm12);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc*3), k0, zmm15);
                }
                break;
            }
            case 3:
            {
                double* temp_b = b + (n - n_remainder)*ldb;
                double* temp_a = a;
                double* temp_c = c + (n - n_remainder)*ldc;
                for(dim_t i = 0;i < (m-D_MR+1);i=i+D_MR)
                {
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm8 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm10 = _mm512_setzero_pd();
                    zmm11 = _mm512_setzero_pd();
                    zmm12 = _mm512_setzero_pd();
                    zmm13 = _mm512_setzero_pd();
                    zmm14 = _mm512_setzero_pd();
                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with 8x6 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_a + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_a + 16));

                    _mm_prefetch((char*)( temp_a + 192), _MM_HINT_T0);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm30, zmm8);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                    zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);

                    zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm30, zmm13);
                    zmm14 = _mm512_fmadd_pd(zmm2, zmm30, zmm14);


                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);
                    zmm8 = _mm512_mul_pd(zmm0, zmm8);

                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm10 = _mm512_mul_pd(zmm0, zmm10);
                    zmm11 = _mm512_mul_pd(zmm0, zmm11);

                    zmm12 = _mm512_mul_pd(zmm0, zmm12);
                    zmm13 = _mm512_mul_pd(zmm0, zmm13);
                    zmm14 = _mm512_mul_pd(zmm0, zmm14);

                    zmm31 = _mm512_set1_pd(beta_val);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_c + 16));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm31, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm31, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm31, zmm8);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc ));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_c + ldc + 16));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                    zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 2));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc * 2 + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_c + ldc * 2 + 16));

                    zmm12 = _mm512_fmadd_pd(zmm0, zmm31, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm31, zmm13);
                    zmm14 = _mm512_fmadd_pd(zmm2, zmm31, zmm14);

                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_storeu_pd((double *)(temp_c + 8), zmm7);
                    _mm512_storeu_pd((double *)(temp_c + 16), zmm8);

                    _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                    _mm512_storeu_pd((double *)(temp_c + ldc + 8), zmm10);
                    _mm512_storeu_pd((double *)(temp_c + ldc + 16), zmm11);

                    _mm512_storeu_pd((double *)(temp_c + ldc * 2), zmm12);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 2 + 8), zmm13);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 2 + 16), zmm14);

                    temp_c += D_MR;
                    temp_a += D_MR;
                }
                dim_t m_rem = m_remainder;
                if(m_rem > 16)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    __mmask8 k0 = _load_mask8(&mask);
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm8 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm10 = _mm512_setzero_pd();
                    zmm11 = _mm512_setzero_pd();
                    zmm12 = _mm512_setzero_pd();
                    zmm13 = _mm512_setzero_pd();
                    zmm14 = _mm512_setzero_pd();
                    zmm2 = _mm512_setzero_pd();
                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with 8x6 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_a + 8));
                    zmm2 = _mm512_mask_loadu_pd (zmm2, k0, (double const *)(temp_a + 16));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm30, zmm8);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                    zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);

                    zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm30, zmm13);
                    zmm14 = _mm512_fmadd_pd(zmm2, zmm30, zmm14);

                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);
                    zmm8 = _mm512_mul_pd(zmm0, zmm8);

                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm10 = _mm512_mul_pd(zmm0, zmm10);
                    zmm11 = _mm512_mul_pd(zmm0, zmm11);

                    zmm12 = _mm512_mul_pd(zmm0, zmm12);
                    zmm13 = _mm512_mul_pd(zmm0, zmm13);
                    zmm14 = _mm512_mul_pd(zmm0, zmm14);

                    zmm31 = _mm512_set1_pd(beta_val);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + 8));
                    zmm2 = _mm512_mask_loadu_pd(zmm2, k0, (double const *)(temp_c + 16));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm31, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm31, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm31, zmm8);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc ));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc + 8));
                    zmm2 = _mm512_mask_loadu_pd(zmm2, k0, (double const *)(temp_c + ldc + 16));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                    zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 2));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc * 2 + 8));
                    zmm2 = _mm512_mask_loadu_pd(zmm2, k0, (double const *)(temp_c + ldc * 2 + 16));

                    zmm12 = _mm512_fmadd_pd(zmm0, zmm31, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm31, zmm13);
                    zmm14 = _mm512_fmadd_pd(zmm2, zmm31, zmm14);


                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_storeu_pd((double *)(temp_c + 8), zmm7);
                    _mm512_mask_storeu_pd ((double *)(temp_c + 16), k0, zmm8);

                    _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                    _mm512_storeu_pd((double *)(temp_c + ldc + 8), zmm10);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc + 16), k0, zmm11);

                    _mm512_storeu_pd((double *)(temp_c + ldc * 2), zmm12);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 2 + 8), zmm13);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc * 2 + 16), k0, zmm14);

                }
                else if(m_rem > 8)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    if (mask == 0) mask = 0xff;
                    __mmask8 k0 = _load_mask8(&mask);
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm10 = _mm512_setzero_pd();
                    zmm12 = _mm512_setzero_pd();
                    zmm13 = _mm512_setzero_pd();
                    zmm1 = _mm512_setzero_pd();

                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with >8x3 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_a + 8));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);

                    zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm30, zmm13);

                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);

                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm10 = _mm512_mul_pd(zmm0, zmm10);

                    zmm12 = _mm512_mul_pd(zmm0, zmm12);
                    zmm13 = _mm512_mul_pd(zmm0, zmm13);

                    zmm31 = _mm512_set1_pd(beta_val);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_c + 8));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm31, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm31, zmm7);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc ));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_c + ldc + 8));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc * 2));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_c + ldc * 2 + 8));

                    zmm12 = _mm512_fmadd_pd(zmm0, zmm31, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm31, zmm13);

                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_mask_storeu_pd((double *)(temp_c + 8), k0, zmm7);

                    _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc + 8), k0, zmm10);

                    _mm512_storeu_pd((double *)(temp_c + ldc * 2), zmm12);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc * 2 + 8), k0, zmm13);

                }
                else if(m_rem > 0)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    if (mask == 0) mask = 0xff;
                    __mmask8 k0 = _load_mask8(&mask);
                    zmm6 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm12 = _mm512_setzero_pd();
                    zmm0 = _mm512_setzero_pd();

                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with >1x3 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_a));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));
                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));
                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);

                    zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);

                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm12 = _mm512_mul_pd(zmm0, zmm12);

                    zmm31 = _mm512_set1_pd(beta_val);

                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_c));
                    zmm6 = _mm512_fmadd_pd(zmm0, zmm31, zmm6);

                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_c + ldc ));
                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);

                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_c + ldc * 2));
                    zmm12 = _mm512_fmadd_pd(zmm0, zmm31, zmm12);

                    _mm512_mask_storeu_pd((double *)(temp_c), k0, zmm6);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc), k0, zmm9);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc * 2), k0, zmm12);
                }
                break;
            }
            case 2:
            {
                double* temp_b = b + (n - n_remainder)*ldb;
                double* temp_a = a;
                double* temp_c = c + (n - n_remainder)*ldc;
                for(dim_t i = 0;i < (m-D_MR+1);i=i+D_MR)
                {
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm8 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm10 = _mm512_setzero_pd();
                    zmm11 = _mm512_setzero_pd();

                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with 24x2 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_a + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_a + 16));

                    _mm_prefetch((char*)( temp_a + 192), _MM_HINT_T0);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm30, zmm8);

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                    zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);

                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);
                    zmm8 = _mm512_mul_pd(zmm0, zmm8);

                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm10 = _mm512_mul_pd(zmm0, zmm10);
                    zmm11 = _mm512_mul_pd(zmm0, zmm11);

                    zmm31 = _mm512_set1_pd(beta_val);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_c + 16));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm31, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm31, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm31, zmm8);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc ));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_c + ldc + 16));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                    zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);

                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_storeu_pd((double *)(temp_c + 8), zmm7);
                    _mm512_storeu_pd((double *)(temp_c + 16), zmm8);

                    _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                    _mm512_storeu_pd((double *)(temp_c + ldc + 8), zmm10);
                    _mm512_storeu_pd((double *)(temp_c + ldc + 16), zmm11);

                    temp_c += D_MR;
                    temp_a += D_MR;
                }
                dim_t m_rem = m_remainder;
                if(m_rem > 16)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    __mmask8 k0 = _load_mask8(&mask);
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm8 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm10 = _mm512_setzero_pd();
                    zmm11 = _mm512_setzero_pd();
                    zmm2 = _mm512_setzero_pd();

                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with >16x2 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_a + 8));
                    zmm2 = _mm512_mask_loadu_pd (zmm2, k0, (double const *)(temp_a + 16));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm30, zmm8);

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                    zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);

                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);
                    zmm8 = _mm512_mul_pd(zmm0, zmm8);

                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm10 = _mm512_mul_pd(zmm0, zmm10);
                    zmm11 = _mm512_mul_pd(zmm0, zmm11);

                    zmm31 = _mm512_set1_pd(beta_val);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + 8));
                    zmm2 = _mm512_mask_loadu_pd(zmm2, k0, (double const *)(temp_c + 16));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm31, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm31, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm31, zmm8);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc ));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + ldc + 8));
                    zmm2 = _mm512_mask_loadu_pd(zmm2, k0, (double const *)(temp_c + ldc + 16));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                    zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);


                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_storeu_pd((double *)(temp_c + 8), zmm7);
                    _mm512_mask_storeu_pd ((double *)(temp_c + 16), k0, zmm8);

                    _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                    _mm512_storeu_pd((double *)(temp_c + ldc + 8), zmm10);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc + 16), k0, zmm11);

                }
                else if(m_rem > 8)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    if (mask == 0) mask = 0xff;
                    __mmask8 k0 = _load_mask8(&mask);
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm10 = _mm512_setzero_pd();
                    zmm1 = _mm512_setzero_pd();

                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with >8x2 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_a + 8));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);

                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);

                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm10 = _mm512_mul_pd(zmm0, zmm10);

                    zmm31 = _mm512_set1_pd(beta_val);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_c + 8));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm31, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm31, zmm7);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c + ldc ));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_c + ldc + 8));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);


                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_mask_storeu_pd((double *)(temp_c + 8), k0, zmm7);

                    _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc + 8), k0, zmm10);

                }
                else if(m_rem > 0)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    if (mask == 0) mask = 0xff;
                    __mmask8 k0 = _load_mask8(&mask);
                    zmm6 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm0 = _mm512_setzero_pd();

                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with >1x2 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_a));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));
                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);

                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm9 = _mm512_mul_pd(zmm0, zmm9);

                    zmm31 = _mm512_set1_pd(beta_val);

                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_c));
                    zmm6 = _mm512_fmadd_pd(zmm0, zmm31, zmm6);

                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_c + ldc ));
                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);

                    _mm512_mask_storeu_pd((double *)(temp_c), k0, zmm6);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc), k0, zmm9);
                }
                break;
            }
            case 1:
            {
                double* temp_b = b + (n - n_remainder)*ldb;
                double* temp_a = a;
                double* temp_c = c + (n - n_remainder)*ldc;
                for(dim_t i = 0;i < (m-D_MR+1);i=i+D_MR)
                {
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm8 = _mm512_setzero_pd();

                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with 24x1 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_a + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_a + 16));

                    _mm_prefetch((char*)( temp_a + 192), _MM_HINT_T0);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm30, zmm8);

                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);
                    zmm8 = _mm512_mul_pd(zmm0, zmm8);

                    zmm31 = _mm512_set1_pd(beta_val);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_c + 16));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm31, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm31, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm31, zmm8);

                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_storeu_pd((double *)(temp_c + 8), zmm7);
                    _mm512_storeu_pd((double *)(temp_c + 16), zmm8);

                    temp_c += D_MR;
                    temp_a += D_MR;
                }
                dim_t m_rem = m_remainder;
                if(m_rem > 16)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    __mmask8 k0 = _load_mask8(&mask);
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm8 = _mm512_setzero_pd();
                    zmm2 = _mm512_setzero_pd();

                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with >16x1 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_a + 8));
                    zmm2 = _mm512_mask_loadu_pd (zmm2, k0, (double const *)(temp_a + 16));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm30, zmm8);

                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);
                    zmm8 = _mm512_mul_pd(zmm0, zmm8);

                    zmm31 = _mm512_set1_pd(beta_val);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_c + 8));
                    zmm2 = _mm512_mask_loadu_pd(zmm2, k0, (double const *)(temp_c + 16));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm31, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm31, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm31, zmm8);


                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_storeu_pd((double *)(temp_c + 8), zmm7);
                    _mm512_mask_storeu_pd ((double *)(temp_c + 16), k0, zmm8);

                }
                else if(m_rem > 8)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    if (mask == 0) mask = 0xff;
                    __mmask8 k0 = _load_mask8(&mask);
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm1 = _mm512_setzero_pd();
                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with >8x1 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_a + 8));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);

                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);

                    zmm31 = _mm512_set1_pd(beta_val);

                    zmm0 = _mm512_loadu_pd((double const *)(temp_c));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_c + 8));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm31, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm31, zmm7);


                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_mask_storeu_pd((double *)(temp_c + 8), k0, zmm7);
                }
                else if(m_rem > 0)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    if (mask == 0) mask = 0xff;
                    __mmask8 k0 = _load_mask8(&mask);
                    zmm6 = _mm512_setzero_pd();
                    zmm0 = _mm512_setzero_pd();
                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with >1x1 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_a));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);

                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);

                    zmm31 = _mm512_set1_pd(beta_val);

                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_c));
                    zmm6 = _mm512_fmadd_pd(zmm0, zmm31, zmm6);

                    _mm512_mask_storeu_pd((double *)(temp_c), k0, zmm6);
                }
                break;
            }
            default:
            {
                break;
            }
        }
	    ret_status = BLIS_SUCCESS;
    }
    else if(alpha_val != 0.0 && beta_val == 0.0)
    {
        /* Compute C = alpha*A*B + beta*c */
        for(dim_t j = 0; (j + (D_NR-1) < n ); j += D_NR)
        {
            double* temp_b = b + j*ldb;
            double* temp_a = a;
            double* temp_c = c + j*ldc;

            for(dim_t i = 0; i < ( m - D_MR+1); i += D_MR)
            {
                //Clear out vector registers to hold fma result.
                //zmm6 to zmm29 holds fma result.
                //zmm0, zmm1, zmm2 are used to load 24 elements from
                //A matrix.
                //zmm30 and zmm31 are alternatively used to broadcast element
                //from B matrix.
                zmm6 = _mm512_setzero_pd();
                zmm7 = _mm512_setzero_pd();
                zmm8 = _mm512_setzero_pd();
                zmm9 = _mm512_setzero_pd();
                zmm10 = _mm512_setzero_pd();
                zmm11 = _mm512_setzero_pd();
                zmm12 = _mm512_setzero_pd();
                zmm13 = _mm512_setzero_pd();
                zmm14 = _mm512_setzero_pd();
                zmm15 = _mm512_setzero_pd();
                zmm16 = _mm512_setzero_pd();
                zmm17 = _mm512_setzero_pd();
                zmm18 = _mm512_setzero_pd();
                zmm19 = _mm512_setzero_pd();
                zmm20 = _mm512_setzero_pd();
                zmm21 = _mm512_setzero_pd();
                zmm22 = _mm512_setzero_pd();
                zmm23 = _mm512_setzero_pd();
                zmm24 = _mm512_setzero_pd();
                zmm25 = _mm512_setzero_pd();
                zmm26 = _mm512_setzero_pd();
                zmm27 = _mm512_setzero_pd();
                zmm28 = _mm512_setzero_pd();
                zmm29 = _mm512_setzero_pd();
                /*
                    a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                    where alpha_val is not zero.
                    b. This loop operates with 24x8 block size
                    along n dimension for every D_NR columns of temp_b where
                    computing all D_MR rows of temp_a.
                    c. Same approach is used in remaining fringe cases.
                */
                zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                zmm1 = _mm512_loadu_pd((double const *)(temp_a + 8));
                zmm2 = _mm512_loadu_pd((double const *)(temp_a + 16));

                _mm_prefetch((char*)( temp_a + 192), _MM_HINT_T0);
                //Broadcast element from B matrix in zmm30
                zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                //Broadcast element from next column of B matrix in zmm31
                zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));
                //Compute A*B.
                zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);
                zmm8 = _mm512_fmadd_pd(zmm2, zmm30, zmm8);
                //Broadcast element from B matrix in zmm30
                zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));
                //Compute A*B.
                zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);
                //Broadcast element from B matrix in zmm31
                zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 3));
                //Compute A*B.
                zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);
                zmm13 = _mm512_fmadd_pd(zmm1, zmm30, zmm13);
                zmm14 = _mm512_fmadd_pd(zmm2, zmm30, zmm14);
                //Broadcast element from B matrix in zmm30
                zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 4));
                //Compute A*B.
                zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);
                zmm17 = _mm512_fmadd_pd(zmm2, zmm31, zmm17);
                //Broadcast element from B matrix in zmm31
                zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 5));
                //Compute A*B.
                zmm18 = _mm512_fmadd_pd(zmm0, zmm30, zmm18);
                zmm19 = _mm512_fmadd_pd(zmm1, zmm30, zmm19);
                zmm20 = _mm512_fmadd_pd(zmm2, zmm30, zmm20);
                //Broadcast element from B matrix in zmm30
                zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 6));
                //Compute A*B.
                zmm21 = _mm512_fmadd_pd(zmm0, zmm31, zmm21);
                zmm22 = _mm512_fmadd_pd(zmm1, zmm31, zmm22);
                zmm23 = _mm512_fmadd_pd(zmm2, zmm31, zmm23);
                //Broadcast element from B matrix in zmm31
                zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 7));
                //Compute A*B.
                zmm24 = _mm512_fmadd_pd(zmm0, zmm30, zmm24);
                zmm25 = _mm512_fmadd_pd(zmm1, zmm30, zmm25);
                zmm26 = _mm512_fmadd_pd(zmm2, zmm30, zmm26);
                //Compute A*B.
                zmm27 = _mm512_fmadd_pd(zmm0, zmm31, zmm27);
                zmm28 = _mm512_fmadd_pd(zmm1, zmm31, zmm28);
                zmm29 = _mm512_fmadd_pd(zmm2, zmm31, zmm29);

                //Broadcast Alpha into zmm0
                zmm0 = _mm512_set1_pd(alpha_val);
                //Scale fma result with Alpha.
                //Alpha * AB
                zmm6 = _mm512_mul_pd(zmm0, zmm6);
                zmm7 = _mm512_mul_pd(zmm0, zmm7);
                zmm8 = _mm512_mul_pd(zmm0, zmm8);
                zmm9 = _mm512_mul_pd(zmm0, zmm9);
                zmm10 = _mm512_mul_pd(zmm0, zmm10);
                zmm11 = _mm512_mul_pd(zmm0, zmm11);
                zmm12 = _mm512_mul_pd(zmm0, zmm12);
                zmm13 = _mm512_mul_pd(zmm0, zmm13);
                zmm14 = _mm512_mul_pd(zmm0, zmm14);
                zmm15 = _mm512_mul_pd(zmm0, zmm15);
                zmm16 = _mm512_mul_pd(zmm0, zmm16);
                zmm17 = _mm512_mul_pd(zmm0, zmm17);
                zmm18 = _mm512_mul_pd(zmm0, zmm18);
                zmm19 = _mm512_mul_pd(zmm0, zmm19);
                zmm20 = _mm512_mul_pd(zmm0, zmm20);
                zmm21 = _mm512_mul_pd(zmm0, zmm21);
                zmm22 = _mm512_mul_pd(zmm0, zmm22);
                zmm23 = _mm512_mul_pd(zmm0, zmm23);
                zmm24 = _mm512_mul_pd(zmm0, zmm24);
                zmm25 = _mm512_mul_pd(zmm0, zmm25);
                zmm26 = _mm512_mul_pd(zmm0, zmm26);
                zmm27 = _mm512_mul_pd(zmm0, zmm27);
                zmm28 = _mm512_mul_pd(zmm0, zmm28);
                zmm29 = _mm512_mul_pd(zmm0, zmm29);

                //Store the result back to Matrix C.
                //Result is available in zmm6 to zmm29.
                _mm512_storeu_pd((double *)(temp_c), zmm6);
                _mm512_storeu_pd((double *)(temp_c + 8), zmm7);
                _mm512_storeu_pd((double *)(temp_c + 16), zmm8);
                //C matrix 2nd column
                _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                _mm512_storeu_pd((double *)(temp_c + ldc + 8), zmm10);
                _mm512_storeu_pd((double *)(temp_c + ldc + 16), zmm11);
                //C matrix 3rd column
                _mm512_storeu_pd((double *)(temp_c + ldc * 2), zmm12);
                _mm512_storeu_pd((double *)(temp_c + ldc * 2 + 8), zmm13);
                _mm512_storeu_pd((double *)(temp_c + ldc * 2 + 16), zmm14);
                //C matrix 4th column
                _mm512_storeu_pd((double *)(temp_c + ldc*3), zmm15);
                _mm512_storeu_pd((double *)(temp_c + ldc * 3 + 8), zmm16);
                _mm512_storeu_pd((double *)(temp_c + ldc * 3 + 16), zmm17);
                //C matrix 5th column
                _mm512_storeu_pd((double *)(temp_c + ldc*4), zmm18);
                _mm512_storeu_pd((double *)(temp_c + ldc*4 + 8), zmm19);
                _mm512_storeu_pd((double *)(temp_c + ldc*4 + 16), zmm20);
                //C matrix 6th column
                _mm512_storeu_pd((double *)(temp_c + ldc*5), zmm21);
                _mm512_storeu_pd((double *)(temp_c + ldc*5 + 8), zmm22);
                _mm512_storeu_pd((double *)(temp_c + ldc*5 + 16), zmm23);
                //C matrix 7th column
                _mm512_storeu_pd((double *)(temp_c + ldc*6), zmm24);
                _mm512_storeu_pd((double *)(temp_c + ldc*6 + 8), zmm25);
                _mm512_storeu_pd((double *)(temp_c + ldc*6 + 16), zmm26);
                //C matrix 8th column
                _mm512_storeu_pd((double *)(temp_c + ldc*7), zmm27);
                _mm512_storeu_pd((double *)(temp_c + ldc*7 + 8), zmm28);
                _mm512_storeu_pd((double *)(temp_c + ldc*7 + 16), zmm29);

                //Update temp_c and temp_a pointer to
                //respective offset.
                temp_c += D_MR;
                temp_a += D_MR;
            }

            dim_t m_rem = m_remainder;
            //Handles the edge case for m_remainder from 17 to 23.
            if(m_rem > 16)
            {
                uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                __mmask8 k0 = _load_mask8(&mask);
                //Clear out vector registers to hold fma result.
                //zmm6 to zmm29 holds fma result.
                //zmm0, zmm1, zmm2 are used to load elements from
                //A matrix.
                //zmm30 and zmm31 are alternatively used to broadcast element
                //from B matrix.
                zmm6 = _mm512_setzero_pd();
                zmm7 = _mm512_setzero_pd();
                zmm8 = _mm512_setzero_pd();
                zmm9 = _mm512_setzero_pd();
                zmm10 = _mm512_setzero_pd();
                zmm11 = _mm512_setzero_pd();
                zmm12 = _mm512_setzero_pd();
                zmm13 = _mm512_setzero_pd();
                zmm14 = _mm512_setzero_pd();
                zmm15 = _mm512_setzero_pd();
                zmm16 = _mm512_setzero_pd();
                zmm17 = _mm512_setzero_pd();
                zmm18 = _mm512_setzero_pd();
                zmm19 = _mm512_setzero_pd();
                zmm20 = _mm512_setzero_pd();
                zmm21 = _mm512_setzero_pd();
                zmm22 = _mm512_setzero_pd();
                zmm23 = _mm512_setzero_pd();
                zmm24 = _mm512_setzero_pd();
                zmm25 = _mm512_setzero_pd();
                zmm26 = _mm512_setzero_pd();
                zmm27 = _mm512_setzero_pd();
                zmm28 = _mm512_setzero_pd();
                zmm29 = _mm512_setzero_pd();
                zmm2 = _mm512_setzero_pd();
                /*
                    a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                    where alpha_val is not zero.
                    b. This loop operates with >16x8 block size
                    along n dimension for every D_NR columns of temp_b where
                    computing all D_MR rows of temp_a.
                    c. Same approach is used in remaining fringe cases.
                */
                zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                zmm1 = _mm512_loadu_pd((double const *)(temp_a + 8));
                zmm2 = _mm512_mask_loadu_pd (zmm2, k0, (double const *)(temp_a + 16));

                //Broadcast element from B matrix in zmm30
                zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                //Broadcast element from next column of B matrix in zmm31
                zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));
                //Compute A*B.
                zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);
                zmm8 = _mm512_fmadd_pd(zmm2, zmm30, zmm8);

                zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));
                //Compute A*B.
                zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);

                zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 3));
                //Compute A*B.
                zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);
                zmm13 = _mm512_fmadd_pd(zmm1, zmm30, zmm13);
                zmm14 = _mm512_fmadd_pd(zmm2, zmm30, zmm14);

                zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 4));
                //Compute A*B.
                zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);
                zmm17 = _mm512_fmadd_pd(zmm2, zmm31, zmm17);

                zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 5));
                //Compute A*B.
                zmm18 = _mm512_fmadd_pd(zmm0, zmm30, zmm18);
                zmm19 = _mm512_fmadd_pd(zmm1, zmm30, zmm19);
                zmm20 = _mm512_fmadd_pd(zmm2, zmm30, zmm20);

                zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 6));
                //Compute A*B.
                zmm21 = _mm512_fmadd_pd(zmm0, zmm31, zmm21);
                zmm22 = _mm512_fmadd_pd(zmm1, zmm31, zmm22);
                zmm23 = _mm512_fmadd_pd(zmm2, zmm31, zmm23);

                zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 7));
                //Compute A*B.
                zmm24 = _mm512_fmadd_pd(zmm0, zmm30, zmm24);
                zmm25 = _mm512_fmadd_pd(zmm1, zmm30, zmm25);
                zmm26 = _mm512_fmadd_pd(zmm2, zmm30, zmm26);
                //Compute A*B.
                zmm27 = _mm512_fmadd_pd(zmm0, zmm31, zmm27);
                zmm28 = _mm512_fmadd_pd(zmm1, zmm31, zmm28);
                zmm29 = _mm512_fmadd_pd(zmm2, zmm31, zmm29);

                //Broadcast Alpha into zmm0
                zmm0 = _mm512_set1_pd(alpha_val);
                //Scale fma result with Alpha.
                //Alpha * AB
                zmm6 = _mm512_mul_pd(zmm0, zmm6);
                zmm7 = _mm512_mul_pd(zmm0, zmm7);
                zmm8 = _mm512_mul_pd(zmm0, zmm8);
                zmm9 = _mm512_mul_pd(zmm0, zmm9);
                zmm10 = _mm512_mul_pd(zmm0, zmm10);
                zmm11 = _mm512_mul_pd(zmm0, zmm11);
                zmm12 = _mm512_mul_pd(zmm0, zmm12);
                zmm13 = _mm512_mul_pd(zmm0, zmm13);
                zmm14 = _mm512_mul_pd(zmm0, zmm14);
                zmm15 = _mm512_mul_pd(zmm0, zmm15);
                zmm16 = _mm512_mul_pd(zmm0, zmm16);
                zmm17 = _mm512_mul_pd(zmm0, zmm17);
                zmm18 = _mm512_mul_pd(zmm0, zmm18);
                zmm19 = _mm512_mul_pd(zmm0, zmm19);
                zmm20 = _mm512_mul_pd(zmm0, zmm20);
                zmm21 = _mm512_mul_pd(zmm0, zmm21);
                zmm22 = _mm512_mul_pd(zmm0, zmm22);
                zmm23 = _mm512_mul_pd(zmm0, zmm23);
                zmm24 = _mm512_mul_pd(zmm0, zmm24);
                zmm25 = _mm512_mul_pd(zmm0, zmm25);
                zmm26 = _mm512_mul_pd(zmm0, zmm26);
                zmm27 = _mm512_mul_pd(zmm0, zmm27);
                zmm28 = _mm512_mul_pd(zmm0, zmm28);
                zmm29 = _mm512_mul_pd(zmm0, zmm29);

                //Store the result back to Matrix C.
                //Result is available in zmm6 to zmm29.
                _mm512_storeu_pd((double *)(temp_c), zmm6);
                _mm512_storeu_pd((double *)(temp_c + 8), zmm7);
                _mm512_mask_storeu_pd ((double *)(temp_c + 16), k0, zmm8);
                //C matrix 2nd column
                _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                _mm512_storeu_pd((double *)(temp_c + ldc + 8), zmm10);
                _mm512_mask_storeu_pd ((double *)(temp_c + ldc + 16), k0, zmm11);
                //C matrix 3rd column
                _mm512_storeu_pd((double *)(temp_c + ldc * 2), zmm12);
                _mm512_storeu_pd((double *)(temp_c + ldc * 2 + 8), zmm13);
                _mm512_mask_storeu_pd ((double *)(temp_c + ldc * 2 + 16), k0, zmm14);
				//C matrix 4th column
                _mm512_storeu_pd((double *)(temp_c + ldc*3), zmm15);
                _mm512_storeu_pd((double *)(temp_c + ldc * 3 + 8), zmm16);
                _mm512_mask_storeu_pd ((double *)(temp_c + ldc * 3 + 16), k0, zmm17);
				//C matrix 5th column
                _mm512_storeu_pd((double *)(temp_c + ldc*4), zmm18);
                _mm512_storeu_pd((double *)(temp_c + ldc*4 + 8), zmm19);
                _mm512_mask_storeu_pd ((double *)(temp_c + ldc*4 + 16), k0, zmm20);
				//C matrix 6th column
                _mm512_storeu_pd((double *)(temp_c + ldc*5), zmm21);
                _mm512_storeu_pd((double *)(temp_c + ldc*5 + 8), zmm22);
                _mm512_mask_storeu_pd ((double *)(temp_c + ldc*5 + 16), k0, zmm23);
				//C matrix 7th column
                _mm512_storeu_pd((double *)(temp_c + ldc*6), zmm24);
                _mm512_storeu_pd((double *)(temp_c + ldc*6 + 8), zmm25);
                _mm512_mask_storeu_pd ((double *)(temp_c + ldc*6 + 16), k0, zmm26);
				//C matrix 8th column
                _mm512_storeu_pd((double *)(temp_c + ldc*7), zmm27);
                _mm512_storeu_pd((double *)(temp_c + ldc*7 + 8), zmm28);
                _mm512_mask_storeu_pd ((double *)(temp_c + ldc*7 + 16), k0, zmm29);
            }
            //Handles the edge cases where m_remainder is from 9 to 16
            else if(m_rem > 8)
            {
                uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                if (mask == 0) mask = 0xff;
                __mmask8 k0 = _load_mask8(&mask);
                //Clear out vector registers to hold fma result.
                //zmm6 to zmm28 holds fma result.
                //zmm0, zmm1 are used to load elements from
                //A matrix.
                //zmm30 and zmm31 are alternatively used to broadcast element
                //from B matrix.
                zmm6 = _mm512_setzero_pd();
                zmm7 = _mm512_setzero_pd();
                zmm9 = _mm512_setzero_pd();
                zmm10 = _mm512_setzero_pd();
                zmm12 = _mm512_setzero_pd();
                zmm13 = _mm512_setzero_pd();
                zmm15 = _mm512_setzero_pd();
                zmm16 = _mm512_setzero_pd();
                zmm18 = _mm512_setzero_pd();
                zmm19 = _mm512_setzero_pd();
                zmm21 = _mm512_setzero_pd();
                zmm22 = _mm512_setzero_pd();
                zmm24 = _mm512_setzero_pd();
                zmm25 = _mm512_setzero_pd();
                zmm27 = _mm512_setzero_pd();
                zmm28 = _mm512_setzero_pd();
                zmm1 = _mm512_setzero_pd();
                /*
                    a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                    where alpha_val is not zero.
                    b. This loop operates with >8x8 block size
                    along n dimension for every D_NR columns of temp_b where
                    computing all D_MR rows of temp_a.
                    c. Same approach is used in remaining fringe cases.
                */
                zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_a + 8));

                //Broadcast element from B matrix in zmm30
                zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                //Broadcast element from next column of B matrix in zmm31
                zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));
                //Compute A*B.
                zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);

                zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));

                zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);

                zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 3));

                zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);
                zmm13 = _mm512_fmadd_pd(zmm1, zmm30, zmm13);

                zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 4));

                zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);

                zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 5));

                zmm18 = _mm512_fmadd_pd(zmm0, zmm30, zmm18);
                zmm19 = _mm512_fmadd_pd(zmm1, zmm30, zmm19);

                zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 6));

                zmm21 = _mm512_fmadd_pd(zmm0, zmm31, zmm21);
                zmm22 = _mm512_fmadd_pd(zmm1, zmm31, zmm22);

                zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 7));

                zmm24 = _mm512_fmadd_pd(zmm0, zmm30, zmm24);
                zmm25 = _mm512_fmadd_pd(zmm1, zmm30, zmm25);

                zmm27 = _mm512_fmadd_pd(zmm0, zmm31, zmm27);
                zmm28 = _mm512_fmadd_pd(zmm1, zmm31, zmm28);

                //Broadcast Alpha into zmm0
                zmm0 = _mm512_set1_pd(alpha_val);
                //Scale fma result with Alpha.
                //Alpha * AB
                zmm6 = _mm512_mul_pd(zmm0, zmm6);
                zmm7 = _mm512_mul_pd(zmm0, zmm7);
                zmm9 = _mm512_mul_pd(zmm0, zmm9);
                zmm10 = _mm512_mul_pd(zmm0, zmm10);
                zmm12 = _mm512_mul_pd(zmm0, zmm12);
                zmm13 = _mm512_mul_pd(zmm0, zmm13);
                zmm15 = _mm512_mul_pd(zmm0, zmm15);
                zmm16 = _mm512_mul_pd(zmm0, zmm16);
                zmm18 = _mm512_mul_pd(zmm0, zmm18);
                zmm19 = _mm512_mul_pd(zmm0, zmm19);
                zmm21 = _mm512_mul_pd(zmm0, zmm21);
                zmm22 = _mm512_mul_pd(zmm0, zmm22);
                zmm24 = _mm512_mul_pd(zmm0, zmm24);
                zmm25 = _mm512_mul_pd(zmm0, zmm25);
                zmm27 = _mm512_mul_pd(zmm0, zmm27);
                zmm28 = _mm512_mul_pd(zmm0, zmm28);

                //Store the result back to Matrix C.
                //Result is available in zmm6 to zmm28.
                _mm512_storeu_pd((double *)(temp_c), zmm6);
                _mm512_mask_storeu_pd((double *)(temp_c + 8), k0, zmm7);
                //C matrix 2nd column
                _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                _mm512_mask_storeu_pd((double *)(temp_c + ldc + 8), k0, zmm10);
                //C matrix 3rd column
                _mm512_storeu_pd((double *)(temp_c + ldc * 2), zmm12);
                _mm512_mask_storeu_pd((double *)(temp_c + ldc * 2 + 8), k0, zmm13);
                //C matrix 4th column
                _mm512_storeu_pd((double *)(temp_c + ldc*3), zmm15);
                _mm512_mask_storeu_pd((double *)(temp_c + ldc * 3 + 8), k0, zmm16);
                //C matrix 5th column
                _mm512_storeu_pd((double *)(temp_c + ldc*4), zmm18);
                _mm512_mask_storeu_pd((double *)(temp_c + ldc*4 + 8), k0, zmm19);
                //C matrix 6th column
                _mm512_storeu_pd((double *)(temp_c + ldc*5), zmm21);
                _mm512_mask_storeu_pd((double *)(temp_c + ldc*5 + 8), k0, zmm22);
                //C matrix 7th column
                _mm512_storeu_pd((double *)(temp_c + ldc*6), zmm24);
                _mm512_mask_storeu_pd((double *)(temp_c + ldc*6 + 8), k0, zmm25);
                //C matrix 8th column
                _mm512_storeu_pd((double *)(temp_c + ldc*7), zmm27);
                _mm512_mask_storeu_pd((double *)(temp_c + ldc*7 + 8), k0, zmm28);
            }
            //Handles the edge case where m_remainder is from 1 to 8
            else if(m_rem > 0)
            {
                uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                if (mask == 0) mask = 0xff;
                __mmask8 k0 = _load_mask8(&mask);
                //Clear out vector registers to hold fma result.
                //zmm6 to zmm27 holds fma result.
                //zmm0 are used to load 8 elements from
                //A matrix.
                //zmm30 and zmm31 are alternatively used to broadcast element
                //from B matrix.
                zmm6 = _mm512_setzero_pd();
                zmm9 = _mm512_setzero_pd();
                zmm12 = _mm512_setzero_pd();
                zmm15 = _mm512_setzero_pd();
                zmm18 = _mm512_setzero_pd();
                zmm21 = _mm512_setzero_pd();
                zmm24 = _mm512_setzero_pd();
                zmm27 = _mm512_setzero_pd();
                zmm0 = _mm512_setzero_pd();
                /*
                    a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                    where alpha_val is not zero.
                    b. This loop operates with >1x8 block size
                    along n dimension for every D_NR columns of temp_b where
                    computing all D_MR rows of temp_a.
                    c. Same approach is used in remaining fringe cases.
                */
                zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_a));

                //Broadcast element from B matrix in zmm30
                zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                //Broadcast element from next column of B matrix in zmm31
                zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));
                //Compute A*B.
                zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);

                zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));

                zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);

                zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 3));

                zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);

                zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 4));

                zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);

                zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 5));

                zmm18 = _mm512_fmadd_pd(zmm0, zmm30, zmm18);

                zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 6));

                zmm21 = _mm512_fmadd_pd(zmm0, zmm31, zmm21);

                zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 7));

                zmm24 = _mm512_fmadd_pd(zmm0, zmm30, zmm24);
                zmm27 = _mm512_fmadd_pd(zmm0, zmm31, zmm27);

                //Broadcast Alpha into zmm0
                zmm0 = _mm512_set1_pd(alpha_val);
                //Scale fma result with Alpha.
                //Alpha * AB
                zmm6 = _mm512_mul_pd(zmm0, zmm6);
                zmm9 = _mm512_mul_pd(zmm0, zmm9);
                zmm12 = _mm512_mul_pd(zmm0, zmm12);
                zmm15 = _mm512_mul_pd(zmm0, zmm15);
                zmm18 = _mm512_mul_pd(zmm0, zmm18);
                zmm21 = _mm512_mul_pd(zmm0, zmm21);
                zmm24 = _mm512_mul_pd(zmm0, zmm24);
                zmm27 = _mm512_mul_pd(zmm0, zmm27);

                //Store the result back to Matrix C.
                _mm512_mask_storeu_pd((double *)(temp_c), k0, zmm6);
                //C matrix 2nd column
                _mm512_mask_storeu_pd((double *)(temp_c + ldc), k0, zmm9);
                //C matrix 3rd column
                _mm512_mask_storeu_pd((double *)(temp_c + ldc * 2), k0, zmm12);
				//C matrix 4th column
                _mm512_mask_storeu_pd((double *)(temp_c + ldc*3), k0, zmm15);
				//C matrix 5th column
                _mm512_mask_storeu_pd((double *)(temp_c + ldc*4), k0, zmm18);
				//C matrix 6th column
                _mm512_mask_storeu_pd((double *)(temp_c + ldc*5), k0, zmm21);
				//C matrix 7th column
                _mm512_mask_storeu_pd((double *)(temp_c + ldc*6), k0, zmm24);
				//C matrix 8th column
                _mm512_mask_storeu_pd((double *)(temp_c + ldc*7), k0, zmm27);
            }
        }

        switch(n_remainder)
        {
            case 7:
            {
                double* temp_b = b + (n - n_remainder)*ldb;
                double* temp_a = a;
                double* temp_c = c + (n - n_remainder)*ldc;
                for(dim_t i = 0;i < (m-D_MR+1);i=i+D_MR)
                {
                    //Clear out vector registers to hold fma result.
                    //zmm6 to zmm26 holds fma result.
                    //zmm0, zmm1, zmm2 are used to load 24 elements from
                    //A matrix.
                    //zmm30 and zmm31 are alternatively used to broadcast element
                    //from B matrix.
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm8 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm10 = _mm512_setzero_pd();
                    zmm11 = _mm512_setzero_pd();
                    zmm12 = _mm512_setzero_pd();
                    zmm13 = _mm512_setzero_pd();
                    zmm14 = _mm512_setzero_pd();
                    zmm15 = _mm512_setzero_pd();
                    zmm16 = _mm512_setzero_pd();
                    zmm17 = _mm512_setzero_pd();
                    zmm18 = _mm512_setzero_pd();
                    zmm19 = _mm512_setzero_pd();
                    zmm20 = _mm512_setzero_pd();
                    zmm21 = _mm512_setzero_pd();
                    zmm22 = _mm512_setzero_pd();
                    zmm23 = _mm512_setzero_pd();
                    zmm24 = _mm512_setzero_pd();
                    zmm25 = _mm512_setzero_pd();
                    zmm26 = _mm512_setzero_pd();
                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with 24x7 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_a + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_a + 16));

                    _mm_prefetch((char*)( temp_a + 192), _MM_HINT_T0);
                    //Broadcast element from B matrix in zmm30
                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    //Broadcast element from B matrix in zmm31
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));
                    //Compute A*B.
                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm30, zmm8);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));
                    //Compute A*B.
                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                    zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 3));
                    //Compute A*B.
                    zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm30, zmm13);
                    zmm14 = _mm512_fmadd_pd(zmm2, zmm30, zmm14);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 4));
                    //Compute A*B.
                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                    zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);
                    zmm17 = _mm512_fmadd_pd(zmm2, zmm31, zmm17);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 5));
                    //Compute A*B.
                    zmm18 = _mm512_fmadd_pd(zmm0, zmm30, zmm18);
                    zmm19 = _mm512_fmadd_pd(zmm1, zmm30, zmm19);
                    zmm20 = _mm512_fmadd_pd(zmm2, zmm30, zmm20);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 6));
                    //Compute A*B.
                    zmm21 = _mm512_fmadd_pd(zmm0, zmm31, zmm21);
                    zmm22 = _mm512_fmadd_pd(zmm1, zmm31, zmm22);
                    zmm23 = _mm512_fmadd_pd(zmm2, zmm31, zmm23);

                    zmm24 = _mm512_fmadd_pd(zmm0, zmm30, zmm24);
                    zmm25 = _mm512_fmadd_pd(zmm1, zmm30, zmm25);
                    zmm26 = _mm512_fmadd_pd(zmm2, zmm30, zmm26);

                    //Broadcast Alpha into zmm0
                    zmm0 = _mm512_set1_pd(alpha_val);
                    //Scale fma result with Alpha.
                    //Alpha * AB
                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);
                    zmm8 = _mm512_mul_pd(zmm0, zmm8);
                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm10 = _mm512_mul_pd(zmm0, zmm10);
                    zmm11 = _mm512_mul_pd(zmm0, zmm11);
                    zmm12 = _mm512_mul_pd(zmm0, zmm12);
                    zmm13 = _mm512_mul_pd(zmm0, zmm13);
                    zmm14 = _mm512_mul_pd(zmm0, zmm14);
                    zmm15 = _mm512_mul_pd(zmm0, zmm15);
                    zmm16 = _mm512_mul_pd(zmm0, zmm16);
                    zmm17 = _mm512_mul_pd(zmm0, zmm17);
                    zmm18 = _mm512_mul_pd(zmm0, zmm18);
                    zmm19 = _mm512_mul_pd(zmm0, zmm19);
                    zmm20 = _mm512_mul_pd(zmm0, zmm20);
                    zmm21 = _mm512_mul_pd(zmm0, zmm21);
                    zmm22 = _mm512_mul_pd(zmm0, zmm22);
                    zmm23 = _mm512_mul_pd(zmm0, zmm23);
                    zmm24 = _mm512_mul_pd(zmm0, zmm24);
                    zmm25 = _mm512_mul_pd(zmm0, zmm25);
                    zmm26 = _mm512_mul_pd(zmm0, zmm26);

                    //Store the result back to Matrix C.
                    //Result is available in zmm6 to zmm26.
                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_storeu_pd((double *)(temp_c + 8), zmm7);
                    _mm512_storeu_pd((double *)(temp_c + 16), zmm8);
                    //C matrix 2nd column
                    _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                    _mm512_storeu_pd((double *)(temp_c + ldc + 8), zmm10);
                    _mm512_storeu_pd((double *)(temp_c + ldc + 16), zmm11);
                    //C matrix 3rd column
                    _mm512_storeu_pd((double *)(temp_c + ldc * 2), zmm12);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 2 + 8), zmm13);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 2 + 16), zmm14);
                    //C matrix 4th column
                    _mm512_storeu_pd((double *)(temp_c + ldc*3), zmm15);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 3 + 8), zmm16);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 3 + 16), zmm17);
                    //C matrix 5th column
                    _mm512_storeu_pd((double *)(temp_c + ldc*4), zmm18);
                    _mm512_storeu_pd((double *)(temp_c + ldc*4 + 8), zmm19);
                    _mm512_storeu_pd((double *)(temp_c + ldc*4 + 16), zmm20);
                    //C matrix 6th column
                    _mm512_storeu_pd((double *)(temp_c + ldc*5), zmm21);
                    _mm512_storeu_pd((double *)(temp_c + ldc*5 + 8), zmm22);
                    _mm512_storeu_pd((double *)(temp_c + ldc*5 + 16), zmm23);
                    //C matrix 7th column
                    _mm512_storeu_pd((double *)(temp_c + ldc*6), zmm24);
                    _mm512_storeu_pd((double *)(temp_c + ldc*6 + 8), zmm25);
                    _mm512_storeu_pd((double *)(temp_c + ldc*6 + 16), zmm26);

                    temp_c += D_MR;
                    temp_a += D_MR;
                }
                dim_t m_rem = m_remainder;
                //Handles the edge case where m_remainder is from 17 to 23
                if(m_rem > 16)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    __mmask8 k0 = _load_mask8(&mask);
                    //Clear out vector registers to hold fma result.
                    //zmm6 to zmm26 holds fma result.
                    //zmm0, zmm1, zmm2 are used to load elements from
                    //A matrix.
                    //zmm30 and zmm31 are alternatively used to broadcast element
                    //from B matrix.
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm8 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm10 = _mm512_setzero_pd();
                    zmm11 = _mm512_setzero_pd();
                    zmm12 = _mm512_setzero_pd();
                    zmm13 = _mm512_setzero_pd();
                    zmm14 = _mm512_setzero_pd();
                    zmm15 = _mm512_setzero_pd();
                    zmm16 = _mm512_setzero_pd();
                    zmm17 = _mm512_setzero_pd();
                    zmm18 = _mm512_setzero_pd();
                    zmm19 = _mm512_setzero_pd();
                    zmm20 = _mm512_setzero_pd();
                    zmm21 = _mm512_setzero_pd();
                    zmm22 = _mm512_setzero_pd();
                    zmm23 = _mm512_setzero_pd();
                    zmm24 = _mm512_setzero_pd();
                    zmm25 = _mm512_setzero_pd();
                    zmm26 = _mm512_setzero_pd();
                    zmm2 = _mm512_setzero_pd();

                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with (>16)x7 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_a + 8));
                    zmm2 = _mm512_mask_loadu_pd (zmm2, k0, (double const *)(temp_a + 16));

                    //Broadcast element from B matrix in zmm30
                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    //Broadcast element from next column of B matrix in zmm31
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));
                    //Compute A*B.
                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm30, zmm8);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));
                    //Compute A*B.
                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                    zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 3));
                    //Compute A*B.
                    zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm30, zmm13);
                    zmm14 = _mm512_fmadd_pd(zmm2, zmm30, zmm14);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 4));
                    //Compute A*B.
                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                    zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);
                    zmm17 = _mm512_fmadd_pd(zmm2, zmm31, zmm17);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 5));
                    //Compute A*B.
                    zmm18 = _mm512_fmadd_pd(zmm0, zmm30, zmm18);
                    zmm19 = _mm512_fmadd_pd(zmm1, zmm30, zmm19);
                    zmm20 = _mm512_fmadd_pd(zmm2, zmm30, zmm20);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 6));
                    //Compute A*B.
                    zmm21 = _mm512_fmadd_pd(zmm0, zmm31, zmm21);
                    zmm22 = _mm512_fmadd_pd(zmm1, zmm31, zmm22);
                    zmm23 = _mm512_fmadd_pd(zmm2, zmm31, zmm23);

                    zmm24 = _mm512_fmadd_pd(zmm0, zmm30, zmm24);
                    zmm25 = _mm512_fmadd_pd(zmm1, zmm30, zmm25);
                    zmm26 = _mm512_fmadd_pd(zmm2, zmm30, zmm26);

                    //Broadcast Alpha into zmm0
                    zmm0 = _mm512_set1_pd(alpha_val);
                    //Scale fma result with Alpha.
                    //Alpha * AB
                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);
                    zmm8 = _mm512_mul_pd(zmm0, zmm8);
                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm10 = _mm512_mul_pd(zmm0, zmm10);
                    zmm11 = _mm512_mul_pd(zmm0, zmm11);
                    zmm12 = _mm512_mul_pd(zmm0, zmm12);
                    zmm13 = _mm512_mul_pd(zmm0, zmm13);
                    zmm14 = _mm512_mul_pd(zmm0, zmm14);
                    zmm15 = _mm512_mul_pd(zmm0, zmm15);
                    zmm16 = _mm512_mul_pd(zmm0, zmm16);
                    zmm17 = _mm512_mul_pd(zmm0, zmm17);
                    zmm18 = _mm512_mul_pd(zmm0, zmm18);
                    zmm19 = _mm512_mul_pd(zmm0, zmm19);
                    zmm20 = _mm512_mul_pd(zmm0, zmm20);
                    zmm21 = _mm512_mul_pd(zmm0, zmm21);
                    zmm22 = _mm512_mul_pd(zmm0, zmm22);
                    zmm23 = _mm512_mul_pd(zmm0, zmm23);
                    zmm24 = _mm512_mul_pd(zmm0, zmm24);
                    zmm25 = _mm512_mul_pd(zmm0, zmm25);
                    zmm26 = _mm512_mul_pd(zmm0, zmm26);

                    //Store the result back to Matrix C.
                    //Result is available in zmm6 to zmm26.
                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_storeu_pd((double *)(temp_c + 8), zmm7);
                    _mm512_mask_storeu_pd ((double *)(temp_c + 16), k0, zmm8);
                    //C matrix 2nd column
                    _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                    _mm512_storeu_pd((double *)(temp_c + ldc + 8), zmm10);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc + 16), k0, zmm11);
                    //C matrix 3rd column
                    _mm512_storeu_pd((double *)(temp_c + ldc * 2), zmm12);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 2 + 8), zmm13);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc * 2 + 16), k0, zmm14);
                    //C matrix 4th column
                    _mm512_storeu_pd((double *)(temp_c + ldc*3), zmm15);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 3 + 8), zmm16);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc * 3 + 16), k0, zmm17);
                    //C matrix 5th column
                    _mm512_storeu_pd((double *)(temp_c + ldc*4), zmm18);
                    _mm512_storeu_pd((double *)(temp_c + ldc*4 + 8), zmm19);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc*4 + 16), k0, zmm20);
                    //C matrix 6th column
                    _mm512_storeu_pd((double *)(temp_c + ldc*5), zmm21);
                    _mm512_storeu_pd((double *)(temp_c + ldc*5 + 8), zmm22);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc*5 + 16), k0, zmm23);
                    //C matrix 7th column
                    _mm512_storeu_pd((double *)(temp_c + ldc*6), zmm24);
                    _mm512_storeu_pd((double *)(temp_c + ldc*6 + 8), zmm25);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc*6 + 16), k0, zmm26);

                }
                //Handles the edge case where m_remadiner is from 9 to 16.
                else if(m_rem > 8)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    if (mask == 0) mask = 0xff;
                    __mmask8 k0 = _load_mask8(&mask);
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm10 = _mm512_setzero_pd();
                    zmm12 = _mm512_setzero_pd();
                    zmm13 = _mm512_setzero_pd();
                    zmm15 = _mm512_setzero_pd();
                    zmm16 = _mm512_setzero_pd();
                    zmm18 = _mm512_setzero_pd();
                    zmm19 = _mm512_setzero_pd();
                    zmm21 = _mm512_setzero_pd();
                    zmm22 = _mm512_setzero_pd();
                    zmm24 = _mm512_setzero_pd();
                    zmm25 = _mm512_setzero_pd();
                    zmm1 = _mm512_setzero_pd();

                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with (>8)x7 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_a + 8));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));
                    //Compute A*B.
                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));
                    //Compute A*B.
                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 3));
                    //Compute A*B.
                    zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm30, zmm13);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 4));
                    //Compute A*B.
                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                    zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 5));
                    //Compute A*B.
                    zmm18 = _mm512_fmadd_pd(zmm0, zmm30, zmm18);
                    zmm19 = _mm512_fmadd_pd(zmm1, zmm30, zmm19);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 6));
                    //Compute A*B.
                    zmm21 = _mm512_fmadd_pd(zmm0, zmm31, zmm21);
                    zmm22 = _mm512_fmadd_pd(zmm1, zmm31, zmm22);
                    //Compute A*B.
                    zmm24 = _mm512_fmadd_pd(zmm0, zmm30, zmm24);
                    zmm25 = _mm512_fmadd_pd(zmm1, zmm30, zmm25);

                    //Broadcast Alpha into zmm0
                    zmm0 = _mm512_set1_pd(alpha_val);
                    //Scale fma result with Alpha.
                    //Alpha * AB
                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);
                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm10 = _mm512_mul_pd(zmm0, zmm10);
                    zmm12 = _mm512_mul_pd(zmm0, zmm12);
                    zmm13 = _mm512_mul_pd(zmm0, zmm13);
                    zmm15 = _mm512_mul_pd(zmm0, zmm15);
                    zmm16 = _mm512_mul_pd(zmm0, zmm16);
                    zmm18 = _mm512_mul_pd(zmm0, zmm18);
                    zmm19 = _mm512_mul_pd(zmm0, zmm19);
                    zmm21 = _mm512_mul_pd(zmm0, zmm21);
                    zmm22 = _mm512_mul_pd(zmm0, zmm22);
                    zmm24 = _mm512_mul_pd(zmm0, zmm24);
                    zmm25 = _mm512_mul_pd(zmm0, zmm25);

                    //Store the result back to Matrix C.
                    //Result is available in zmm6 to zmm25.
                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_mask_storeu_pd((double *)(temp_c + 8), k0, zmm7);
				    //C matrix 2nd column
                    _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc + 8), k0, zmm10);
				    //C matrix 3rd column
                    _mm512_storeu_pd((double *)(temp_c + ldc * 2), zmm12);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc * 2 + 8), k0, zmm13);
				    //C matrix 4th column
                    _mm512_storeu_pd((double *)(temp_c + ldc*3), zmm15);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc * 3 + 8), k0, zmm16);
				    //C matrix 5th column
                    _mm512_storeu_pd((double *)(temp_c + ldc*4), zmm18);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc*4 + 8), k0, zmm19);
				    //C matrix 6th column
                    _mm512_storeu_pd((double *)(temp_c + ldc*5), zmm21);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc*5 + 8), k0, zmm22);
				    //C matrix 7th column
                    _mm512_storeu_pd((double *)(temp_c + ldc*6), zmm24);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc*6 + 8), k0, zmm25);
                }
                else if(m_rem > 0)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    if (mask == 0) mask = 0xff;
                    __mmask8 k0 = _load_mask8(&mask);
                    zmm6 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm12 = _mm512_setzero_pd();
                    zmm15 = _mm512_setzero_pd();
                    zmm18 = _mm512_setzero_pd();
                    zmm21 = _mm512_setzero_pd();
                    zmm24 = _mm512_setzero_pd();
                    zmm0 = _mm512_setzero_pd();
                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with (>1)x7 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_a));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 3));

                    zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 4));

                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 5));

                    zmm18 = _mm512_fmadd_pd(zmm0, zmm30, zmm18);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 6));

                    zmm21 = _mm512_fmadd_pd(zmm0, zmm31, zmm21);

                    zmm24 = _mm512_fmadd_pd(zmm0, zmm30, zmm24);
                    //Broadcast Alpha into zmm0
                    zmm0 = _mm512_set1_pd(alpha_val);
                    //Scale fma result with Alpha.
                    //Alpha * AB
                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm12 = _mm512_mul_pd(zmm0, zmm12);
                    zmm15 = _mm512_mul_pd(zmm0, zmm15);
                    zmm18 = _mm512_mul_pd(zmm0, zmm18);
                    zmm21 = _mm512_mul_pd(zmm0, zmm21);
                    zmm24 = _mm512_mul_pd(zmm0, zmm24);

                    //Store the result back to Matrix C.
                    //Result is available in zmm6 to zmm24.
                    _mm512_mask_storeu_pd((double *)(temp_c), k0, zmm6);
                    //C matrix 2nd column
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc), k0, zmm9);
                    //C matrix 3rd column
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc * 2), k0, zmm12);
                    //C matrix 4th column
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc*3), k0, zmm15);
                    //C matrix 5th column
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc*4), k0, zmm18);
                    //C matrix 6th column
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc*5), k0, zmm21);
                    //C matrix 7th column
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc*6), k0, zmm24);
                }
                break;
            }
            case 6:
            {
                double* temp_b = b + (n - n_remainder)*ldb;
                double* temp_a = a;
                double* temp_c = c + (n - n_remainder)*ldc;
                for(dim_t i = 0;i < (m-D_MR+1);i=i+D_MR)
                {
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm8 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm10 = _mm512_setzero_pd();
                    zmm11 = _mm512_setzero_pd();
                    zmm12 = _mm512_setzero_pd();
                    zmm13 = _mm512_setzero_pd();
                    zmm14 = _mm512_setzero_pd();
                    zmm15 = _mm512_setzero_pd();
                    zmm16 = _mm512_setzero_pd();
                    zmm17 = _mm512_setzero_pd();
                    zmm18 = _mm512_setzero_pd();
                    zmm19 = _mm512_setzero_pd();
                    zmm20 = _mm512_setzero_pd();
                    zmm21 = _mm512_setzero_pd();
                    zmm22 = _mm512_setzero_pd();
                    zmm23 = _mm512_setzero_pd();
                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with 24x6 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_a + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_a + 16));

                    _mm_prefetch((char*)( temp_a + 192), _MM_HINT_T0);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm30, zmm8);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                    zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 3));

                    zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm30, zmm13);
                    zmm14 = _mm512_fmadd_pd(zmm2, zmm30, zmm14);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 4));

                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                    zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);
                    zmm17 = _mm512_fmadd_pd(zmm2, zmm31, zmm17);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 5));

                    zmm18 = _mm512_fmadd_pd(zmm0, zmm30, zmm18);
                    zmm19 = _mm512_fmadd_pd(zmm1, zmm30, zmm19);
                    zmm20 = _mm512_fmadd_pd(zmm2, zmm30, zmm20);

                    zmm21 = _mm512_fmadd_pd(zmm0, zmm31, zmm21);
                    zmm22 = _mm512_fmadd_pd(zmm1, zmm31, zmm22);
                    zmm23 = _mm512_fmadd_pd(zmm2, zmm31, zmm23);

                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);
                    zmm8 = _mm512_mul_pd(zmm0, zmm8);

                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm10 = _mm512_mul_pd(zmm0, zmm10);
                    zmm11 = _mm512_mul_pd(zmm0, zmm11);

                    zmm12 = _mm512_mul_pd(zmm0, zmm12);
                    zmm13 = _mm512_mul_pd(zmm0, zmm13);
                    zmm14 = _mm512_mul_pd(zmm0, zmm14);

                    zmm15 = _mm512_mul_pd(zmm0, zmm15);
                    zmm16 = _mm512_mul_pd(zmm0, zmm16);
                    zmm17 = _mm512_mul_pd(zmm0, zmm17);

                    zmm18 = _mm512_mul_pd(zmm0, zmm18);
                    zmm19 = _mm512_mul_pd(zmm0, zmm19);
                    zmm20 = _mm512_mul_pd(zmm0, zmm20);

                    zmm21 = _mm512_mul_pd(zmm0, zmm21);
                    zmm22 = _mm512_mul_pd(zmm0, zmm22);
                    zmm23 = _mm512_mul_pd(zmm0, zmm23);

                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_storeu_pd((double *)(temp_c + 8), zmm7);
                    _mm512_storeu_pd((double *)(temp_c + 16), zmm8);

                    _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                    _mm512_storeu_pd((double *)(temp_c + ldc + 8), zmm10);
                    _mm512_storeu_pd((double *)(temp_c + ldc + 16), zmm11);

                    _mm512_storeu_pd((double *)(temp_c + ldc * 2), zmm12);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 2 + 8), zmm13);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 2 + 16), zmm14);

                    _mm512_storeu_pd((double *)(temp_c + ldc*3), zmm15);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 3 + 8), zmm16);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 3 + 16), zmm17);

                    _mm512_storeu_pd((double *)(temp_c + ldc*4), zmm18);
                    _mm512_storeu_pd((double *)(temp_c + ldc*4 + 8), zmm19);
                    _mm512_storeu_pd((double *)(temp_c + ldc*4 + 16), zmm20);

                    _mm512_storeu_pd((double *)(temp_c + ldc*5), zmm21);
                    _mm512_storeu_pd((double *)(temp_c + ldc*5 + 8), zmm22);
                    _mm512_storeu_pd((double *)(temp_c + ldc*5 + 16), zmm23);

                    temp_c += D_MR;
                    temp_a += D_MR;
                }
                dim_t m_rem = m_remainder;
                if(m_rem > 16)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    __mmask8 k0 = _load_mask8(&mask);
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm8 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm10 = _mm512_setzero_pd();
                    zmm11 = _mm512_setzero_pd();
                    zmm12 = _mm512_setzero_pd();
                    zmm13 = _mm512_setzero_pd();
                    zmm14 = _mm512_setzero_pd();
                    zmm15 = _mm512_setzero_pd();
                    zmm16 = _mm512_setzero_pd();
                    zmm17 = _mm512_setzero_pd();
                    zmm18 = _mm512_setzero_pd();
                    zmm19 = _mm512_setzero_pd();
                    zmm20 = _mm512_setzero_pd();
                    zmm21 = _mm512_setzero_pd();
                    zmm22 = _mm512_setzero_pd();
                    zmm23 = _mm512_setzero_pd();
                    zmm2 = _mm512_setzero_pd();

                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with >16x6 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_a + 8));
                    zmm2 = _mm512_mask_loadu_pd (zmm2, k0, (double const *)(temp_a + 16));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm30, zmm8);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                    zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 3));

                    zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm30, zmm13);
                    zmm14 = _mm512_fmadd_pd(zmm2, zmm30, zmm14);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 4));

                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                    zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);
                    zmm17 = _mm512_fmadd_pd(zmm2, zmm31, zmm17);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 5));

                    zmm18 = _mm512_fmadd_pd(zmm0, zmm30, zmm18);
                    zmm19 = _mm512_fmadd_pd(zmm1, zmm30, zmm19);
                    zmm20 = _mm512_fmadd_pd(zmm2, zmm30, zmm20);

                    zmm21 = _mm512_fmadd_pd(zmm0, zmm31, zmm21);
                    zmm22 = _mm512_fmadd_pd(zmm1, zmm31, zmm22);
                    zmm23 = _mm512_fmadd_pd(zmm2, zmm31, zmm23);

                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);
                    zmm8 = _mm512_mul_pd(zmm0, zmm8);

                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm10 = _mm512_mul_pd(zmm0, zmm10);
                    zmm11 = _mm512_mul_pd(zmm0, zmm11);

                    zmm12 = _mm512_mul_pd(zmm0, zmm12);
                    zmm13 = _mm512_mul_pd(zmm0, zmm13);
                    zmm14 = _mm512_mul_pd(zmm0, zmm14);

                    zmm15 = _mm512_mul_pd(zmm0, zmm15);
                    zmm16 = _mm512_mul_pd(zmm0, zmm16);
                    zmm17 = _mm512_mul_pd(zmm0, zmm17);

                    zmm18 = _mm512_mul_pd(zmm0, zmm18);
                    zmm19 = _mm512_mul_pd(zmm0, zmm19);
                    zmm20 = _mm512_mul_pd(zmm0, zmm20);

                    zmm21 = _mm512_mul_pd(zmm0, zmm21);
                    zmm22 = _mm512_mul_pd(zmm0, zmm22);
                    zmm23 = _mm512_mul_pd(zmm0, zmm23);

                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_storeu_pd((double *)(temp_c + 8), zmm7);
                    _mm512_mask_storeu_pd ((double *)(temp_c + 16), k0, zmm8);

                    _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                    _mm512_storeu_pd((double *)(temp_c + ldc + 8), zmm10);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc + 16), k0, zmm11);

                    _mm512_storeu_pd((double *)(temp_c + ldc * 2), zmm12);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 2 + 8), zmm13);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc * 2 + 16), k0, zmm14);

                    _mm512_storeu_pd((double *)(temp_c + ldc*3), zmm15);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 3 + 8), zmm16);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc * 3 + 16), k0, zmm17);

                    _mm512_storeu_pd((double *)(temp_c + ldc*4), zmm18);
                    _mm512_storeu_pd((double *)(temp_c + ldc*4 + 8), zmm19);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc*4 + 16), k0, zmm20);

                    _mm512_storeu_pd((double *)(temp_c + ldc*5), zmm21);
                    _mm512_storeu_pd((double *)(temp_c + ldc*5 + 8), zmm22);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc*5 + 16), k0, zmm23);

                }
                else if(m_rem > 8)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    if (mask == 0) mask = 0xff;
                    __mmask8 k0 = _load_mask8(&mask);
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm10 = _mm512_setzero_pd();
                    zmm12 = _mm512_setzero_pd();
                    zmm13 = _mm512_setzero_pd();
                    zmm15 = _mm512_setzero_pd();
                    zmm16 = _mm512_setzero_pd();
                    zmm18 = _mm512_setzero_pd();
                    zmm19 = _mm512_setzero_pd();
                    zmm21 = _mm512_setzero_pd();
                    zmm22 = _mm512_setzero_pd();
                    zmm1 = _mm512_setzero_pd();

                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with >8x6 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_a + 8));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 3));

                    zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm30, zmm13);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 4));

                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                    zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 5));

                    zmm18 = _mm512_fmadd_pd(zmm0, zmm30, zmm18);
                    zmm19 = _mm512_fmadd_pd(zmm1, zmm30, zmm19);

                    zmm21 = _mm512_fmadd_pd(zmm0, zmm31, zmm21);
                    zmm22 = _mm512_fmadd_pd(zmm1, zmm31, zmm22);

                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);

                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm10 = _mm512_mul_pd(zmm0, zmm10);

                    zmm12 = _mm512_mul_pd(zmm0, zmm12);
                    zmm13 = _mm512_mul_pd(zmm0, zmm13);

                    zmm15 = _mm512_mul_pd(zmm0, zmm15);
                    zmm16 = _mm512_mul_pd(zmm0, zmm16);

                    zmm18 = _mm512_mul_pd(zmm0, zmm18);
                    zmm19 = _mm512_mul_pd(zmm0, zmm19);

                    zmm21 = _mm512_mul_pd(zmm0, zmm21);
                    zmm22 = _mm512_mul_pd(zmm0, zmm22);

                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_mask_storeu_pd((double *)(temp_c + 8), k0, zmm7);

                    _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc + 8), k0, zmm10);

                    _mm512_storeu_pd((double *)(temp_c + ldc * 2), zmm12);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc * 2 + 8), k0, zmm13);

                    _mm512_storeu_pd((double *)(temp_c + ldc*3), zmm15);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc * 3 + 8), k0, zmm16);

                    _mm512_storeu_pd((double *)(temp_c + ldc*4), zmm18);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc*4 + 8), k0, zmm19);

                    _mm512_storeu_pd((double *)(temp_c + ldc*5), zmm21);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc*5 + 8), k0, zmm22);
                }
                else if(m_rem > 0)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    if (mask == 0) mask = 0xff;
                    __mmask8 k0 = _load_mask8(&mask);
                    zmm6 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm12 = _mm512_setzero_pd();
                    zmm15 = _mm512_setzero_pd();
                    zmm18 = _mm512_setzero_pd();
                    zmm21 = _mm512_setzero_pd();
                    zmm0 = _mm512_setzero_pd();

                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with >1x6 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_a));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));
                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));
                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 3));
                    zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 4));
                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 5));
                    zmm18 = _mm512_fmadd_pd(zmm0, zmm30, zmm18);

                    zmm21 = _mm512_fmadd_pd(zmm0, zmm31, zmm21);

                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm12 = _mm512_mul_pd(zmm0, zmm12);
                    zmm15 = _mm512_mul_pd(zmm0, zmm15);
                    zmm18 = _mm512_mul_pd(zmm0, zmm18);
                    zmm21 = _mm512_mul_pd(zmm0, zmm21);

                    _mm512_mask_storeu_pd((double *)(temp_c), k0, zmm6);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc), k0, zmm9);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc * 2), k0, zmm12);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc*3), k0, zmm15);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc*4), k0, zmm18);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc*5), k0, zmm21);
                }
                break;
            }
            case 5:
            {
                double* temp_b = b + (n - n_remainder)*ldb;
                double* temp_a = a;
                double* temp_c = c + (n - n_remainder)*ldc;
                for(dim_t i = 0;i < (m-D_MR+1);i=i+D_MR)
                {
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm8 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm10 = _mm512_setzero_pd();
                    zmm11 = _mm512_setzero_pd();
                    zmm12 = _mm512_setzero_pd();
                    zmm13 = _mm512_setzero_pd();
                    zmm14 = _mm512_setzero_pd();
                    zmm15 = _mm512_setzero_pd();
                    zmm16 = _mm512_setzero_pd();
                    zmm17 = _mm512_setzero_pd();
                    zmm18 = _mm512_setzero_pd();
                    zmm19 = _mm512_setzero_pd();
                    zmm20 = _mm512_setzero_pd();

                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with 24x5 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_a + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_a + 16));

                    _mm_prefetch((char*)( temp_a + 192), _MM_HINT_T0);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm30, zmm8);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                    zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 3));

                    zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm30, zmm13);
                    zmm14 = _mm512_fmadd_pd(zmm2, zmm30, zmm14);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 4));

                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                    zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);
                    zmm17 = _mm512_fmadd_pd(zmm2, zmm31, zmm17);

                    zmm18 = _mm512_fmadd_pd(zmm0, zmm30, zmm18);
                    zmm19 = _mm512_fmadd_pd(zmm1, zmm30, zmm19);
                    zmm20 = _mm512_fmadd_pd(zmm2, zmm30, zmm20);


                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);
                    zmm8 = _mm512_mul_pd(zmm0, zmm8);

                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm10 = _mm512_mul_pd(zmm0, zmm10);
                    zmm11 = _mm512_mul_pd(zmm0, zmm11);

                    zmm12 = _mm512_mul_pd(zmm0, zmm12);
                    zmm13 = _mm512_mul_pd(zmm0, zmm13);
                    zmm14 = _mm512_mul_pd(zmm0, zmm14);

                    zmm15 = _mm512_mul_pd(zmm0, zmm15);
                    zmm16 = _mm512_mul_pd(zmm0, zmm16);
                    zmm17 = _mm512_mul_pd(zmm0, zmm17);

                    zmm18 = _mm512_mul_pd(zmm0, zmm18);
                    zmm19 = _mm512_mul_pd(zmm0, zmm19);
                    zmm20 = _mm512_mul_pd(zmm0, zmm20);

                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_storeu_pd((double *)(temp_c + 8), zmm7);
                    _mm512_storeu_pd((double *)(temp_c + 16), zmm8);

                    _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                    _mm512_storeu_pd((double *)(temp_c + ldc + 8), zmm10);
                    _mm512_storeu_pd((double *)(temp_c + ldc + 16), zmm11);

                    _mm512_storeu_pd((double *)(temp_c + ldc * 2), zmm12);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 2 + 8), zmm13);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 2 + 16), zmm14);

                    _mm512_storeu_pd((double *)(temp_c + ldc*3), zmm15);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 3 + 8), zmm16);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 3 + 16), zmm17);

                    _mm512_storeu_pd((double *)(temp_c + ldc*4), zmm18);
                    _mm512_storeu_pd((double *)(temp_c + ldc*4 + 8), zmm19);
                    _mm512_storeu_pd((double *)(temp_c + ldc*4 + 16), zmm20);

                    temp_c += D_MR;
                    temp_a += D_MR;
                }
                dim_t m_rem = m_remainder;
                if(m_rem > 16)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    __mmask8 k0 = _load_mask8(&mask);
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm8 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm10 = _mm512_setzero_pd();
                    zmm11 = _mm512_setzero_pd();
                    zmm12 = _mm512_setzero_pd();
                    zmm13 = _mm512_setzero_pd();
                    zmm14 = _mm512_setzero_pd();
                    zmm15 = _mm512_setzero_pd();
                    zmm16 = _mm512_setzero_pd();
                    zmm17 = _mm512_setzero_pd();
                    zmm18 = _mm512_setzero_pd();
                    zmm19 = _mm512_setzero_pd();
                    zmm20 = _mm512_setzero_pd();
                    zmm2 = _mm512_setzero_pd();

                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with 8x6 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_a + 8));
                    zmm2 = _mm512_mask_loadu_pd (zmm2, k0, (double const *)(temp_a + 16));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm30, zmm8);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                    zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 3));

                    zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm30, zmm13);
                    zmm14 = _mm512_fmadd_pd(zmm2, zmm30, zmm14);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 4));

                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                    zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);
                    zmm17 = _mm512_fmadd_pd(zmm2, zmm31, zmm17);

                    zmm18 = _mm512_fmadd_pd(zmm0, zmm30, zmm18);
                    zmm19 = _mm512_fmadd_pd(zmm1, zmm30, zmm19);
                    zmm20 = _mm512_fmadd_pd(zmm2, zmm30, zmm20);

                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);
                    zmm8 = _mm512_mul_pd(zmm0, zmm8);

                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm10 = _mm512_mul_pd(zmm0, zmm10);
                    zmm11 = _mm512_mul_pd(zmm0, zmm11);

                    zmm12 = _mm512_mul_pd(zmm0, zmm12);
                    zmm13 = _mm512_mul_pd(zmm0, zmm13);
                    zmm14 = _mm512_mul_pd(zmm0, zmm14);

                    zmm15 = _mm512_mul_pd(zmm0, zmm15);
                    zmm16 = _mm512_mul_pd(zmm0, zmm16);
                    zmm17 = _mm512_mul_pd(zmm0, zmm17);

                    zmm18 = _mm512_mul_pd(zmm0, zmm18);
                    zmm19 = _mm512_mul_pd(zmm0, zmm19);
                    zmm20 = _mm512_mul_pd(zmm0, zmm20);

                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_storeu_pd((double *)(temp_c + 8), zmm7);
                    _mm512_mask_storeu_pd ((double *)(temp_c + 16), k0, zmm8);

                    _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                    _mm512_storeu_pd((double *)(temp_c + ldc + 8), zmm10);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc + 16), k0, zmm11);

                    _mm512_storeu_pd((double *)(temp_c + ldc * 2), zmm12);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 2 + 8), zmm13);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc * 2 + 16), k0, zmm14);

                    _mm512_storeu_pd((double *)(temp_c + ldc*3), zmm15);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 3 + 8), zmm16);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc * 3 + 16), k0, zmm17);

                    _mm512_storeu_pd((double *)(temp_c + ldc*4), zmm18);
                    _mm512_storeu_pd((double *)(temp_c + ldc*4 + 8), zmm19);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc*4 + 16), k0, zmm20);

                }
                else if(m_rem > 8)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    if (mask == 0) mask = 0xff;
                    __mmask8 k0 = _load_mask8(&mask);
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm10 = _mm512_setzero_pd();
                    zmm12 = _mm512_setzero_pd();
                    zmm13 = _mm512_setzero_pd();
                    zmm15 = _mm512_setzero_pd();
                    zmm16 = _mm512_setzero_pd();
                    zmm18 = _mm512_setzero_pd();
                    zmm19 = _mm512_setzero_pd();
                    zmm1 = _mm512_setzero_pd();

                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with >8x6 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_a + 8));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 3));

                    zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm30, zmm13);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 4));

                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                    zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);

                    zmm18 = _mm512_fmadd_pd(zmm0, zmm30, zmm18);
                    zmm19 = _mm512_fmadd_pd(zmm1, zmm30, zmm19);

                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);

                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm10 = _mm512_mul_pd(zmm0, zmm10);

                    zmm12 = _mm512_mul_pd(zmm0, zmm12);
                    zmm13 = _mm512_mul_pd(zmm0, zmm13);

                    zmm15 = _mm512_mul_pd(zmm0, zmm15);
                    zmm16 = _mm512_mul_pd(zmm0, zmm16);

                    zmm18 = _mm512_mul_pd(zmm0, zmm18);
                    zmm19 = _mm512_mul_pd(zmm0, zmm19);

                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_mask_storeu_pd((double *)(temp_c + 8), k0, zmm7);

                    _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc + 8), k0, zmm10);

                    _mm512_storeu_pd((double *)(temp_c + ldc * 2), zmm12);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc * 2 + 8), k0, zmm13);

                    _mm512_storeu_pd((double *)(temp_c + ldc*3), zmm15);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc * 3 + 8), k0, zmm16);

                    _mm512_storeu_pd((double *)(temp_c + ldc*4), zmm18);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc*4 + 8), k0, zmm19);

                }
                else if(m_rem > 0)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    if (mask == 0) mask = 0xff;
                    __mmask8 k0 = _load_mask8(&mask);
                    zmm6 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm12 = _mm512_setzero_pd();
                    zmm15 = _mm512_setzero_pd();
                    zmm18 = _mm512_setzero_pd();
                    zmm0 = _mm512_setzero_pd();

                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with >1x6 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_a));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));
                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));
                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 3));
                    zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 4));
                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);

                    zmm18 = _mm512_fmadd_pd(zmm0, zmm30, zmm18);

                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm12 = _mm512_mul_pd(zmm0, zmm12);
                    zmm15 = _mm512_mul_pd(zmm0, zmm15);
                    zmm18 = _mm512_mul_pd(zmm0, zmm18);

                    _mm512_mask_storeu_pd((double *)(temp_c), k0, zmm6);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc), k0, zmm9);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc * 2), k0, zmm12);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc*3), k0, zmm15);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc*4), k0, zmm18);
                }
                break;
            }
            case 4:
            {
                double* temp_b = b + (n - n_remainder)*ldb;
                double* temp_a = a;
                double* temp_c = c + (n - n_remainder)*ldc;
                for(dim_t i = 0;i < (m-D_MR+1);i=i+D_MR)
                {
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm8 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm10 = _mm512_setzero_pd();
                    zmm11 = _mm512_setzero_pd();
                    zmm12 = _mm512_setzero_pd();
                    zmm13 = _mm512_setzero_pd();
                    zmm14 = _mm512_setzero_pd();
                    zmm15 = _mm512_setzero_pd();
                    zmm16 = _mm512_setzero_pd();
                    zmm17 = _mm512_setzero_pd();
                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with 24x4 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_a + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_a + 16));

                    _mm_prefetch((char*)( temp_a + 192), _MM_HINT_T0);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm30, zmm8);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                    zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 3));

                    zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm30, zmm13);
                    zmm14 = _mm512_fmadd_pd(zmm2, zmm30, zmm14);

                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                    zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);
                    zmm17 = _mm512_fmadd_pd(zmm2, zmm31, zmm17);


                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);
                    zmm8 = _mm512_mul_pd(zmm0, zmm8);

                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm10 = _mm512_mul_pd(zmm0, zmm10);
                    zmm11 = _mm512_mul_pd(zmm0, zmm11);

                    zmm12 = _mm512_mul_pd(zmm0, zmm12);
                    zmm13 = _mm512_mul_pd(zmm0, zmm13);
                    zmm14 = _mm512_mul_pd(zmm0, zmm14);

                    zmm15 = _mm512_mul_pd(zmm0, zmm15);
                    zmm16 = _mm512_mul_pd(zmm0, zmm16);
                    zmm17 = _mm512_mul_pd(zmm0, zmm17);

                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_storeu_pd((double *)(temp_c + 8), zmm7);
                    _mm512_storeu_pd((double *)(temp_c + 16), zmm8);

                    _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                    _mm512_storeu_pd((double *)(temp_c + ldc + 8), zmm10);
                    _mm512_storeu_pd((double *)(temp_c + ldc + 16), zmm11);

                    _mm512_storeu_pd((double *)(temp_c + ldc * 2), zmm12);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 2 + 8), zmm13);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 2 + 16), zmm14);

                    _mm512_storeu_pd((double *)(temp_c + ldc*3), zmm15);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 3 + 8), zmm16);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 3 + 16), zmm17);

                    temp_c += D_MR;
                    temp_a += D_MR;
                }
                dim_t m_rem = m_remainder;
                if(m_rem > 16)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    __mmask8 k0 = _load_mask8(&mask);
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm8 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm10 = _mm512_setzero_pd();
                    zmm11 = _mm512_setzero_pd();
                    zmm12 = _mm512_setzero_pd();
                    zmm13 = _mm512_setzero_pd();
                    zmm14 = _mm512_setzero_pd();
                    zmm15 = _mm512_setzero_pd();
                    zmm16 = _mm512_setzero_pd();
                    zmm17 = _mm512_setzero_pd();
                    zmm2 = _mm512_setzero_pd();
                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with >16x4 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_a + 8));
                    zmm2 = _mm512_mask_loadu_pd (zmm2, k0, (double const *)(temp_a + 16));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm30, zmm8);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                    zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 3));

                    zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm30, zmm13);
                    zmm14 = _mm512_fmadd_pd(zmm2, zmm30, zmm14);

                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                    zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);
                    zmm17 = _mm512_fmadd_pd(zmm2, zmm31, zmm17);

                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);
                    zmm8 = _mm512_mul_pd(zmm0, zmm8);

                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm10 = _mm512_mul_pd(zmm0, zmm10);
                    zmm11 = _mm512_mul_pd(zmm0, zmm11);

                    zmm12 = _mm512_mul_pd(zmm0, zmm12);
                    zmm13 = _mm512_mul_pd(zmm0, zmm13);
                    zmm14 = _mm512_mul_pd(zmm0, zmm14);

                    zmm15 = _mm512_mul_pd(zmm0, zmm15);
                    zmm16 = _mm512_mul_pd(zmm0, zmm16);
                    zmm17 = _mm512_mul_pd(zmm0, zmm17);

                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_storeu_pd((double *)(temp_c + 8), zmm7);
                    _mm512_mask_storeu_pd ((double *)(temp_c + 16), k0, zmm8);

                    _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                    _mm512_storeu_pd((double *)(temp_c + ldc + 8), zmm10);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc + 16), k0, zmm11);

                    _mm512_storeu_pd((double *)(temp_c + ldc * 2), zmm12);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 2 + 8), zmm13);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc * 2 + 16), k0, zmm14);

                    _mm512_storeu_pd((double *)(temp_c + ldc*3), zmm15);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 3 + 8), zmm16);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc * 3 + 16), k0, zmm17);

                }
                else if(m_rem > 8)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    if (mask == 0) mask = 0xff;
                    __mmask8 k0 = _load_mask8(&mask);
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm10 = _mm512_setzero_pd();
                    zmm12 = _mm512_setzero_pd();
                    zmm13 = _mm512_setzero_pd();
                    zmm15 = _mm512_setzero_pd();
                    zmm16 = _mm512_setzero_pd();
                    zmm1 = _mm512_setzero_pd();

                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with >8x4 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_a + 8));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 3));

                    zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm30, zmm13);

                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);
                    zmm16 = _mm512_fmadd_pd(zmm1, zmm31, zmm16);

                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);

                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm10 = _mm512_mul_pd(zmm0, zmm10);

                    zmm12 = _mm512_mul_pd(zmm0, zmm12);
                    zmm13 = _mm512_mul_pd(zmm0, zmm13);

                    zmm15 = _mm512_mul_pd(zmm0, zmm15);
                    zmm16 = _mm512_mul_pd(zmm0, zmm16);

                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_mask_storeu_pd((double *)(temp_c + 8), k0, zmm7);

                    _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc + 8), k0, zmm10);

                    _mm512_storeu_pd((double *)(temp_c + ldc * 2), zmm12);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc * 2 + 8), k0, zmm13);

                    _mm512_storeu_pd((double *)(temp_c + ldc*3), zmm15);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc * 3 + 8), k0, zmm16);

                }
                else if(m_rem > 0)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    if (mask == 0) mask = 0xff;
                    __mmask8 k0 = _load_mask8(&mask);
                    zmm6 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm12 = _mm512_setzero_pd();
                    zmm15 = _mm512_setzero_pd();
                    zmm0 = _mm512_setzero_pd();

                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with >1x4 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_a));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));
                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));
                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);

                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 3));
                    zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);

                    zmm15 = _mm512_fmadd_pd(zmm0, zmm31, zmm15);

                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm12 = _mm512_mul_pd(zmm0, zmm12);
                    zmm15 = _mm512_mul_pd(zmm0, zmm15);

                    _mm512_mask_storeu_pd((double *)(temp_c), k0, zmm6);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc), k0, zmm9);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc * 2), k0, zmm12);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc*3), k0, zmm15);
                }
                break;
            }
            case 3:
            {
                double* temp_b = b + (n - n_remainder)*ldb;
                double* temp_a = a;
                double* temp_c = c + (n - n_remainder)*ldc;
                for(dim_t i = 0;i < (m-D_MR+1);i=i+D_MR)
                {
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm8 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm10 = _mm512_setzero_pd();
                    zmm11 = _mm512_setzero_pd();
                    zmm12 = _mm512_setzero_pd();
                    zmm13 = _mm512_setzero_pd();
                    zmm14 = _mm512_setzero_pd();
                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with 8x6 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_a + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_a + 16));

                    _mm_prefetch((char*)( temp_a + 192), _MM_HINT_T0);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm30, zmm8);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                    zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);

                    zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm30, zmm13);
                    zmm14 = _mm512_fmadd_pd(zmm2, zmm30, zmm14);


                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);
                    zmm8 = _mm512_mul_pd(zmm0, zmm8);

                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm10 = _mm512_mul_pd(zmm0, zmm10);
                    zmm11 = _mm512_mul_pd(zmm0, zmm11);

                    zmm12 = _mm512_mul_pd(zmm0, zmm12);
                    zmm13 = _mm512_mul_pd(zmm0, zmm13);
                    zmm14 = _mm512_mul_pd(zmm0, zmm14);

                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_storeu_pd((double *)(temp_c + 8), zmm7);
                    _mm512_storeu_pd((double *)(temp_c + 16), zmm8);

                    _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                    _mm512_storeu_pd((double *)(temp_c + ldc + 8), zmm10);
                    _mm512_storeu_pd((double *)(temp_c + ldc + 16), zmm11);

                    _mm512_storeu_pd((double *)(temp_c + ldc * 2), zmm12);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 2 + 8), zmm13);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 2 + 16), zmm14);

                    temp_c += D_MR;
                    temp_a += D_MR;
                }
                dim_t m_rem = m_remainder;
                if(m_rem > 16)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    __mmask8 k0 = _load_mask8(&mask);
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm8 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm10 = _mm512_setzero_pd();
                    zmm11 = _mm512_setzero_pd();
                    zmm12 = _mm512_setzero_pd();
                    zmm13 = _mm512_setzero_pd();
                    zmm14 = _mm512_setzero_pd();
                    zmm2 = _mm512_setzero_pd();
                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with 8x6 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_a + 8));
                    zmm2 = _mm512_mask_loadu_pd (zmm2, k0, (double const *)(temp_a + 16));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm30, zmm8);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                    zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);

                    zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm30, zmm13);
                    zmm14 = _mm512_fmadd_pd(zmm2, zmm30, zmm14);

                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);
                    zmm8 = _mm512_mul_pd(zmm0, zmm8);

                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm10 = _mm512_mul_pd(zmm0, zmm10);
                    zmm11 = _mm512_mul_pd(zmm0, zmm11);

                    zmm12 = _mm512_mul_pd(zmm0, zmm12);
                    zmm13 = _mm512_mul_pd(zmm0, zmm13);
                    zmm14 = _mm512_mul_pd(zmm0, zmm14);

                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_storeu_pd((double *)(temp_c + 8), zmm7);
                    _mm512_mask_storeu_pd ((double *)(temp_c + 16), k0, zmm8);

                    _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                    _mm512_storeu_pd((double *)(temp_c + ldc + 8), zmm10);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc + 16), k0, zmm11);

                    _mm512_storeu_pd((double *)(temp_c + ldc * 2), zmm12);
                    _mm512_storeu_pd((double *)(temp_c + ldc * 2 + 8), zmm13);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc * 2 + 16), k0, zmm14);

                }
                else if(m_rem > 8)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    if (mask == 0) mask = 0xff;
                    __mmask8 k0 = _load_mask8(&mask);
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm10 = _mm512_setzero_pd();
                    zmm12 = _mm512_setzero_pd();
                    zmm13 = _mm512_setzero_pd();
                    zmm1 = _mm512_setzero_pd();

                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with >8x3 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_a + 8));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);

                    zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);
                    zmm13 = _mm512_fmadd_pd(zmm1, zmm30, zmm13);

                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);

                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm10 = _mm512_mul_pd(zmm0, zmm10);

                    zmm12 = _mm512_mul_pd(zmm0, zmm12);
                    zmm13 = _mm512_mul_pd(zmm0, zmm13);

                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_mask_storeu_pd((double *)(temp_c + 8), k0, zmm7);

                    _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc + 8), k0, zmm10);

                    _mm512_storeu_pd((double *)(temp_c + ldc * 2), zmm12);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc * 2 + 8), k0, zmm13);

                }
                else if(m_rem > 0)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    if (mask == 0) mask = 0xff;
                    __mmask8 k0 = _load_mask8(&mask);
                    zmm6 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm12 = _mm512_setzero_pd();
                    zmm0 = _mm512_setzero_pd();

                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with >1x3 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_a));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));
                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 2));
                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);

                    zmm12 = _mm512_fmadd_pd(zmm0, zmm30, zmm12);

                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm12 = _mm512_mul_pd(zmm0, zmm12);

                    _mm512_mask_storeu_pd((double *)(temp_c), k0, zmm6);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc), k0, zmm9);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc * 2), k0, zmm12);
                }
                break;
            }
            case 2:
            {
                double* temp_b = b + (n - n_remainder)*ldb;
                double* temp_a = a;
                double* temp_c = c + (n - n_remainder)*ldc;
                for(dim_t i = 0;i < (m-D_MR+1);i=i+D_MR)
                {
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm8 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm10 = _mm512_setzero_pd();
                    zmm11 = _mm512_setzero_pd();

                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with 24x2 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_a + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_a + 16));

                    _mm_prefetch((char*)( temp_a + 192), _MM_HINT_T0);

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm30, zmm8);

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                    zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);

                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);
                    zmm8 = _mm512_mul_pd(zmm0, zmm8);

                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm10 = _mm512_mul_pd(zmm0, zmm10);
                    zmm11 = _mm512_mul_pd(zmm0, zmm11);

                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_storeu_pd((double *)(temp_c + 8), zmm7);
                    _mm512_storeu_pd((double *)(temp_c + 16), zmm8);

                    _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                    _mm512_storeu_pd((double *)(temp_c + ldc + 8), zmm10);
                    _mm512_storeu_pd((double *)(temp_c + ldc + 16), zmm11);

                    temp_c += D_MR;
                    temp_a += D_MR;
                }
                dim_t m_rem = m_remainder;
                if(m_rem > 16)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    __mmask8 k0 = _load_mask8(&mask);
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm8 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm10 = _mm512_setzero_pd();
                    zmm11 = _mm512_setzero_pd();
                    zmm2 = _mm512_setzero_pd();

                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with >16x2 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_a + 8));
                    zmm2 = _mm512_mask_loadu_pd (zmm2, k0, (double const *)(temp_a + 16));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm30, zmm8);

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);
                    zmm11 = _mm512_fmadd_pd(zmm2, zmm31, zmm11);

                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);
                    zmm8 = _mm512_mul_pd(zmm0, zmm8);

                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm10 = _mm512_mul_pd(zmm0, zmm10);
                    zmm11 = _mm512_mul_pd(zmm0, zmm11);

                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_storeu_pd((double *)(temp_c + 8), zmm7);
                    _mm512_mask_storeu_pd ((double *)(temp_c + 16), k0, zmm8);

                    _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                    _mm512_storeu_pd((double *)(temp_c + ldc + 8), zmm10);
                    _mm512_mask_storeu_pd ((double *)(temp_c + ldc + 16), k0, zmm11);

                }
                else if(m_rem > 8)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    if (mask == 0) mask = 0xff;
                    __mmask8 k0 = _load_mask8(&mask);
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm10 = _mm512_setzero_pd();
                    zmm1 = _mm512_setzero_pd();

                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with >8x2 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_a + 8));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);
                    zmm10 = _mm512_fmadd_pd(zmm1, zmm31, zmm10);

                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);

                    zmm9 = _mm512_mul_pd(zmm0, zmm9);
                    zmm10 = _mm512_mul_pd(zmm0, zmm10);

                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_mask_storeu_pd((double *)(temp_c + 8), k0, zmm7);

                    _mm512_storeu_pd((double *)(temp_c + ldc), zmm9);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc + 8), k0, zmm10);

                }
                else if(m_rem > 0)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    if (mask == 0) mask = 0xff;
                    __mmask8 k0 = _load_mask8(&mask);
                    zmm6 = _mm512_setzero_pd();
                    zmm9 = _mm512_setzero_pd();
                    zmm0 = _mm512_setzero_pd();

                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with >1x2 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_a));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm31 = _mm512_set1_pd(*(double const *)(temp_b + ldb * 1));
                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);

                    zmm9 = _mm512_fmadd_pd(zmm0, zmm31, zmm9);

                    zmm0 = _mm512_set1_pd(alpha_val);

                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm9 = _mm512_mul_pd(zmm0, zmm9);

                    _mm512_mask_storeu_pd((double *)(temp_c), k0, zmm6);
                    _mm512_mask_storeu_pd((double *)(temp_c + ldc), k0, zmm9);
                }
                break;
            }
            case 1:
            {
                double* temp_b = b + (n - n_remainder)*ldb;
                double* temp_a = a;
                double* temp_c = c + (n - n_remainder)*ldc;
                for(dim_t i = 0;i < (m-D_MR+1);i=i+D_MR)
                {
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm8 = _mm512_setzero_pd();

                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with 24x1 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_a + 8));
                    zmm2 = _mm512_loadu_pd((double const *)(temp_a + 16));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm30, zmm8);

                    zmm0 = _mm512_set1_pd(alpha_val);
                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);
                    zmm8 = _mm512_mul_pd(zmm0, zmm8);

                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_storeu_pd((double *)(temp_c + 8), zmm7);
                    _mm512_storeu_pd((double *)(temp_c + 16), zmm8);

                    temp_c += D_MR;
                    temp_a += D_MR;
                }
                dim_t m_rem = m_remainder;
                if(m_rem > 16)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    __mmask8 k0 = _load_mask8(&mask);
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm8 = _mm512_setzero_pd();
                    zmm2 = _mm512_setzero_pd();

                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with >16x1 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_loadu_pd((double const *)(temp_a + 8));
                    zmm2 = _mm512_mask_loadu_pd (zmm2, k0, (double const *)(temp_a + 16));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);
                    zmm8 = _mm512_fmadd_pd(zmm2, zmm30, zmm8);

                    zmm0 = _mm512_set1_pd(alpha_val);
                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);
                    zmm8 = _mm512_mul_pd(zmm0, zmm8);

                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_storeu_pd((double *)(temp_c + 8), zmm7);
                    _mm512_mask_storeu_pd ((double *)(temp_c + 16), k0, zmm8);

                }
                else if(m_rem > 8)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    if (mask == 0) mask = 0xff;
                    __mmask8 k0 = _load_mask8(&mask);
                    zmm6 = _mm512_setzero_pd();
                    zmm7 = _mm512_setzero_pd();
                    zmm1 = _mm512_setzero_pd();
                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with >8x1 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_loadu_pd((double const *)(temp_a));
                    zmm1 = _mm512_mask_loadu_pd(zmm1, k0, (double const *)(temp_a + 8));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));

                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);
                    zmm7 = _mm512_fmadd_pd(zmm1, zmm30, zmm7);

                    zmm0 = _mm512_set1_pd(alpha_val);
                    zmm6 = _mm512_mul_pd(zmm0, zmm6);
                    zmm7 = _mm512_mul_pd(zmm0, zmm7);

                    _mm512_storeu_pd((double *)(temp_c), zmm6);
                    _mm512_mask_storeu_pd((double *)(temp_c + 8), k0, zmm7);
                }
                else if(m_rem > 0)
                {
                    uint8_t mask = (0xff >> (0x8 - (m & 7))); // calculate mask based on m_remainder
                    if (mask == 0) mask = 0xff;
                    __mmask8 k0 = _load_mask8(&mask);
                    zmm6 = _mm512_setzero_pd();
                    zmm0 = _mm512_setzero_pd();
                    /*
                        a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                        where alpha_val is not zero.
                        b. This loop operates with >1x1 block size
                        along n dimension for every D_NR columns of temp_b where
                        computing all D_MR rows of temp_a.
                        c. Same approach is used in remaining fringe cases.
                    */
                    zmm0 = _mm512_mask_loadu_pd(zmm0, k0, (double const *)(temp_a));

                    zmm30 = _mm512_set1_pd(*(double const *)(temp_b));
                    zmm6 = _mm512_fmadd_pd(zmm0, zmm30, zmm6);

                    zmm0 = _mm512_set1_pd(alpha_val);
                    zmm6 = _mm512_mul_pd(zmm0, zmm6);

                    _mm512_mask_storeu_pd((double *)(temp_c), k0, zmm6);
                }
                break;
            }
            default:
            {
                break;
            }
        }
	    ret_status = BLIS_SUCCESS;
    }
    else
    {
	    ;//return failure;
    }
    return ret_status;

}
