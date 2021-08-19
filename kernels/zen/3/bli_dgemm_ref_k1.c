/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2021, Advanced Micro Devices, Inc. All rights reserved.

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

#define D_MR  8
#define D_NR  6

void bli_dgemm_ref_k1_nn
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
    double alpha_val, beta_val;

    beta_val = *beta;
    alpha_val = *alpha;

    if((m == 0) || (n == 0) || (((alpha_val == 0.0) || (k == 0)) && (beta_val == 1.0))){
        return;
    }

    dim_t m_remainder = (m % D_MR);
    dim_t n_remainder = (n % D_NR);

    //scratch registers
    __m256d ymm0, ymm1, ymm2, ymm3;
    __m256d ymm4, ymm5, ymm6, ymm7;
    __m256d ymm8, ymm9, ymm10, ymm11;
    __m256d ymm12, ymm13, ymm14, ymm15;
    __m128d xmm5;

    /* Form C = alpha*A*B + beta*c */
    for(dim_t j = 0;j < (n-D_NR+1);j=j+D_NR)
    {
        double* temp_b = b + j*ldb;
        double* temp_a = a;
        double* temp_c = c + j*ldc;

        for(dim_t i = 0;i < (m-D_MR+1);i=i+D_MR)
        {
            ymm3 = _mm256_setzero_pd();
            ymm4 = _mm256_setzero_pd();
            ymm5 = _mm256_setzero_pd();
            ymm6 = _mm256_setzero_pd();
            ymm7 = _mm256_setzero_pd();
            ymm8 = _mm256_setzero_pd();
            ymm9 = _mm256_setzero_pd();
            ymm10 = _mm256_setzero_pd();
            ymm11 = _mm256_setzero_pd();
            ymm12 = _mm256_setzero_pd();
            ymm13 = _mm256_setzero_pd();
            ymm14 = _mm256_setzero_pd();
            ymm15 = _mm256_setzero_pd();

            if(alpha_val != 0.0)
            {
                /*
                    a. Perform alpha*A*B using temp_a, temp_b and alpha_val,
                       where alpha_val is not zero.
                    b. This loop operates with 8x6 block size
                       along n dimension for every D_NR columns of temp_b where
                       computing all D_MR rows of temp_a.
                    c. Same approach is used in remaining fringe cases.
                */
                ymm0 = _mm256_loadu_pd((double const *)(temp_a));     //a[0][0] a[1][0] a[2][0] a[3][0]
                ymm1 = _mm256_loadu_pd((double const *)(temp_a + 4)); //a[4][0] a[5][0] a[6][0] a[7][0]
                _mm_prefetch((char*)( temp_a + 64), _MM_HINT_T0);

                ymm15 = _mm256_broadcast_sd((double const *)(&alpha_val));

                ymm0 = _mm256_mul_pd(ymm0,ymm15); //ymm0 = (alpha_val*a[0][0] alpha_val*a[1][0] alpha_val*a[2][0] alpha_val*a[3][0])
                ymm1 = _mm256_mul_pd(ymm1,ymm15); //ymm1 = (alpha_val*a[4][0] alpha_val*a[5][0] alpha_val*a[6][0] alpha_val*a[7][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b));  //b[0][0]
                ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3); //ymm3 += (b[0][0]*a[0][0] b[0][0]*a[1][0] b[0][0]*a[2][0] b[0][0]*a[3][0])
                ymm4 = _mm256_fmadd_pd(ymm2, ymm1, ymm4); //ymm4 += (b[0][0]*a[4][0] b[0][0]*a[5][0] b[0][0]*a[6][0] b[0][0]*a[7][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b + ldb * 1)); //b[0][1]
                ymm5 = _mm256_fmadd_pd(ymm2, ymm0, ymm5); //ymm5 += (b[0][1]*a[0][0] b[0][1]*a[1][0] b[0][1]*a[2][0] b[0][1]*a[3][0])
                ymm6 = _mm256_fmadd_pd(ymm2, ymm1, ymm6); //ymm6 += (b[0][1]*a[4][0] b[0][1]*a[5][0] b[0][1]*a[6][0] b[0][1]*a[7][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b + ldb * 2)); //b[0][2]
                ymm7 = _mm256_fmadd_pd(ymm2, ymm0, ymm7); //ymm7 += (b[0][2]*a[0][0] b[0][2]*a[1][0] b[0][2]*a[2][0] b[0][2]*a[3][0])
                ymm8 = _mm256_fmadd_pd(ymm2, ymm1, ymm8); //ymm8 += (b[0][2]*a[4][0] b[0][2]*a[5][0] b[0][2]*a[6][0] b[0][2]*a[7][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b + ldb * 3)); //b[0][3]
                ymm9 = _mm256_fmadd_pd(ymm2, ymm0, ymm9);   //ymm9 += (b[0][3]*a[0][0] b[0][3]*a[1][0] b[0][3]*a[2][0] b[0][3]*a[3][0])
                ymm10 = _mm256_fmadd_pd(ymm2, ymm1, ymm10); //ymm10 += (b[0][3]*a[4][0] b[0][3]*a[5][0] b[0][3]*a[6][0] b[0][3]*a[7][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b + ldb * 4)); //b[0][4]
                ymm11 = _mm256_fmadd_pd(ymm2, ymm0, ymm11); //ymm11 += (b[0][4]*a[0][0] b[0][4]*a[1][0] b[0][4]*a[2][0] b[0][4]*a[3][0])
                ymm12 = _mm256_fmadd_pd(ymm2, ymm1, ymm12); //ymm12 += (b[0][4]*a[4][0] b[0][4]*a[5][0] b[0][4]*a[6][0] b[0][4]*a[7][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b + ldb * 5)); //b[0][5]
                ymm13 = _mm256_fmadd_pd(ymm2, ymm0, ymm13); //ymm13 += (b[0][5]*a[0][0] b[0][5]*a[1][0] b[0][5]*a[2][0] b[0][5]*a[3][0])
                ymm14 = _mm256_fmadd_pd(ymm2, ymm1, ymm14); //ymm14 += (b[0][5]*a[4][0] b[0][5]*a[5][0] b[0][5]*a[6][0] b[0][5]*a[7][0])
            }

            if(beta_val != 0.0)
            {
                /*
                    a. Perform beta*C using temp_c, beta,
                       where beta_val is not zero.
                    b. This loop operates with 8x6 block size
                       along n dimension for every D_NR columns of temp_c where
                       computing all D_MR rows of temp_c.
                    c. Accumulated alpha*A*B into registers will be added to beta*C
                    d. Same approach is used in remaining fringe cases.
                */
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_val));

                ymm0 = _mm256_loadu_pd((double const *)(temp_c));     //c[0][0] c[1][0] c[2][0] c[3][0]
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + 4)); //c[4][0] c[5][0] c[6][0] c[7][0]

                ymm3 = _mm256_fmadd_pd(ymm15, ymm0, ymm3); //ymm3 += (beta_val*c[0][0] beta_val*c[1][0] beta_val*c[2][0] beta_val*c[3][0])
                ymm4 = _mm256_fmadd_pd(ymm15, ymm1, ymm4); //ymm4 += (beta_val*c[4][0] beta_val*c[5][0] beta_val*c[6][0] beta_val*c[7][0])

                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc));     //c[0][1] c[1][1] c[2][1] c[3][1]
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc + 4)); //c[4][1] c[5][1] c[6][1] c[7][1]

                ymm5 = _mm256_fmadd_pd(ymm15, ymm0, ymm5); //ymm5 += (beta_val*c[0][1] beta_val*c[1][1] beta_val*c[2][1] beta_val*c[3][1])
                ymm6 = _mm256_fmadd_pd(ymm15, ymm1, ymm6); //ymm6 += (beta_val*c[4][1] beta_val*c[5][1] beta_val*c[6][1] beta_val*c[7][1])

                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*2));     //c[0][2] c[1][2] c[2][2] c[3][2]
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc*2 + 4)); //c[4][2] c[5][2] c[6][2] c[7][2]

                ymm7 = _mm256_fmadd_pd(ymm15, ymm0, ymm7); //ymm7 += (beta_val*c[0][2] beta_val*c[1][2] beta_val*c[2][2] beta_val*c[3][2])
                ymm8 = _mm256_fmadd_pd(ymm15, ymm1, ymm8); //ymm8 += (beta_val*c[4][2] beta_val*c[5][2] beta_val*c[6][2] beta_val*c[7][2])

                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*3));     //c[0][3] c[1][3] c[2][3] c[3][3]
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc*3 + 4)); //c[4][3] c[5][3] c[6][3] c[7][3]

                ymm9 = _mm256_fmadd_pd(ymm15, ymm0, ymm9);   //ymm9 += (beta_val*c[0][3] beta_val*c[1][3] beta_val*c[2][3] beta_val*c[3][3])
                ymm10 = _mm256_fmadd_pd(ymm15, ymm1, ymm10); //ymm10 += (beta_val*c[4][3] beta_val*c[5][3] beta_val*c[6][3] beta_val*c[7][3])

                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*4));     //c[0][4] c[1][4] c[2][4] c[3][4]
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc*4 + 4)); //c[4][4] c[5][4] c[6][4] c[7][4]

                ymm11 = _mm256_fmadd_pd(ymm15, ymm0, ymm11); //ymm11 += (beta_val*c[0][4] beta_val*c[1][4] beta_val*c[2][4] beta_val*c[3][4])
                ymm12 = _mm256_fmadd_pd(ymm15, ymm1, ymm12); //ymm12 += (beta_val*c[4][4] beta_val*c[5][4] beta_val*c[6][4] beta_val*c[7][4])

                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*5));     //c[0][5] c[1][5] c[2][5] c[3][5]
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc*5 + 4)); //c[4][5] c[5][5] c[6][5] c[7][5]

                ymm13 = _mm256_fmadd_pd(ymm15, ymm0, ymm13); //ymm13 += (beta_val*c[0][5] beta_val*c[1][5] beta_val*c[2][5] beta_val*c[3][5])
                ymm14 = _mm256_fmadd_pd(ymm15, ymm1, ymm14); //ymm14 += (beta_val*c[4][5] beta_val*c[5][5] beta_val*c[6][5] beta_val*c[7][5])
            }

            /*
                a. If both alpha_val & beta_val are zeros,
                   C matix will be filled with all zeros.
                b. If only alpha_val is zero,
                   accumulated alpha*A*B will be stored into C.
                c. If only beta_val is zero,
                   accumulated beta*C will be stored into C.
                d. If both alpha_val & beta_val are not zeros,
                   accumulated alpha*A*B + beta*C will be stored into C.
                e. Same approach is used in remaining fringe cases.
            */

            _mm256_storeu_pd((double *)(temp_c), ymm3);              //c[0][0] c[1][0] c[2][0] c[3][0]
            _mm256_storeu_pd((double *)(temp_c + 4), ymm4);          //c[4][0] c[5][0] c[6][0] c[7][0]

            _mm256_storeu_pd((double *)(temp_c + ldc), ymm5);        //c[0][1] c[1][1] c[2][1] c[3][1]
            _mm256_storeu_pd((double *)(temp_c + ldc + 4), ymm6);    //c[4][1] c[5][1] c[6][1] c[7][1]

            _mm256_storeu_pd((double *)(temp_c + ldc*2), ymm7);      //c[0][2] c[1][2] c[2][2] c[3][2]
            _mm256_storeu_pd((double *)(temp_c + ldc*2 + 4), ymm8);  //c[4][2] c[5][2] c[6][2] c[7][2]

            _mm256_storeu_pd((double *)(temp_c + ldc*3), ymm9);      //c[0][3] c[1][3] c[2][3] c[3][3]
            _mm256_storeu_pd((double *)(temp_c + ldc*3 +4), ymm10);  //c[4][3] c[5][3] c[6][3] c[7][3]

            _mm256_storeu_pd((double *)(temp_c + ldc*4), ymm11);     //c[0][4] c[1][4] c[2][4] c[3][4]
            _mm256_storeu_pd((double *)(temp_c + ldc*4 + 4), ymm12); //c[4][4] c[5][4] c[6][4] c[7][4]

            _mm256_storeu_pd((double *)(temp_c + ldc*5), ymm13);     //c[0][5] c[1][5] c[2][5] c[3][5]
            _mm256_storeu_pd((double *)(temp_c + ldc*5 + 4), ymm14); //c[4][5] c[5][5] c[6][5] c[7][5]

            temp_c += D_MR;
            temp_a += D_MR;
        }

        dim_t m_rem = m_remainder;
        if(m_remainder >= 4)
        {
            ymm3 = _mm256_setzero_pd();
            ymm5 = _mm256_setzero_pd();
            ymm7 = _mm256_setzero_pd();
            ymm9 = _mm256_setzero_pd();
            ymm11 = _mm256_setzero_pd();
            ymm13 = _mm256_setzero_pd();
            ymm15 = _mm256_setzero_pd();

            if(alpha_val != 0.0)
            {
                ymm0 = _mm256_loadu_pd((double const *)(temp_a));  //a[0][0] a[1][0] a[2][0] a[3][0]

                ymm15 = _mm256_broadcast_sd((double const *)(&alpha_val));
                ymm0 = _mm256_mul_pd(ymm0,ymm15);  //ymm0 = (alpha_val*a[0][0] alpha_val*a[1][0] alpha_val*a[2][0] alpha_val*a[3][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b));  //b[0][0]
                ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3); //ymm3 += (b[0][0]*a[0][0] b[0][0]*a[1][0] b[0][0]*a[2][0] b[0][0]*a[3][0]

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b + ldb * 1));  //b[0][1]
                ymm5 = _mm256_fmadd_pd(ymm2, ymm0, ymm5); //ymm5 += (b[0][1]*a[0][0] b[0][1]*a[1][0] b[0][1]*a[2][0] b[0][1]*a[3][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b + ldb * 2));  //b[0][2]
                ymm7 = _mm256_fmadd_pd(ymm2, ymm0, ymm7); //ymm7 += (b[0][2]*a[0][0] b[0][2]*a[1][0] b[0][2]*a[2][0] b[0][2]*a[3][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b + ldb * 3));  //b[0][3]
                ymm9 = _mm256_fmadd_pd(ymm2, ymm0, ymm9); //ymm9 += (b[0][3]*a[0][0] b[0][3]*a[1][0] b[0][3]*a[2][0] b[0][3]*a[3][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b + ldb * 4));  //b[0][4]
                ymm11 = _mm256_fmadd_pd(ymm2, ymm0, ymm11); //ymm11 += (b[0][4]*a[0][0] b[0][4]*a[1][0] b[0][4]*a[2][0] b[0][4]*a[3][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b + ldb * 5));  //b[0][5]
                ymm13 = _mm256_fmadd_pd(ymm2, ymm0, ymm13); //ymm13 += (b[0][5]*a[0][0] b[0][5]*a[1][0] b[0][5]*a[2][0] b[0][5]*a[3][0])
            }

            if(beta_val != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_val));

                ymm0 = _mm256_loadu_pd((double const *)(temp_c));  //c[0][0] c[1][0] c[2][0] c[3][0]
                ymm3 = _mm256_fmadd_pd(ymm15, ymm0, ymm3);   //ymm3 += (beta_val*c[0][0] beta_val*c[1][0] beta_val*c[2][0] beta_val*c[3][0])

                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc));    //c[0][1] c[1][1] c[2][1] c[3][1]
                ymm5 = _mm256_fmadd_pd(ymm15, ymm0, ymm5);   //ymm5 += (beta_val*c[0][1] beta_val*c[1][1] beta_val*c[2][1] beta_val*c[3][1])

                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*2));  //c[0][2] c[1][2] c[2][2] c[3][2]
                ymm7 = _mm256_fmadd_pd(ymm15, ymm0, ymm7);   //ymm7 += (beta_val*c[0][2] beta_val*c[1][2] beta_val*c[2][2] beta_val*c[3][2])

                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*3));  //c[0][3] c[1][3] c[2][3] c[3][3]
                ymm9 = _mm256_fmadd_pd(ymm15, ymm0, ymm9);   //ymm9 += (beta_val*c[0][3] beta_val*c[1][3] beta_val*c[2][3] beta_val*c[3][3])

                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*4));  //c[0][4] c[1][4] c[2][4] c[3][4]
                ymm11 = _mm256_fmadd_pd(ymm15, ymm0, ymm11); //ymm11 += (beta_val*c[0][4] beta_val*c[1][4] beta_val*c[2][4] beta_val*c[3][4])

                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*5));  //c[0][5] c[1][5] c[2][5] c[3][5]
                ymm13 = _mm256_fmadd_pd(ymm15, ymm0, ymm13); //ymm13 += (beta_val*c[0][5] beta_val*c[1][5] beta_val*c[2][5] beta_val*c[3][5])

            }
            _mm256_storeu_pd((double *)(temp_c), ymm3);          //c[0][0] c[1][0] c[2][0] c[3][0]
            _mm256_storeu_pd((double *)(temp_c + ldc), ymm5);    //c[0][1] c[1][1] c[2][1] c[3][1]
            _mm256_storeu_pd((double *)(temp_c + ldc*2), ymm7);  //c[0][2] c[1][2] c[2][2] c[3][2]
            _mm256_storeu_pd((double *)(temp_c + ldc*3), ymm9);  //c[0][3] c[1][3] c[2][3] c[3][3]
            _mm256_storeu_pd((double *)(temp_c + ldc*4), ymm11); //c[0][4] c[1][4] c[2][4] c[3][4]
            _mm256_storeu_pd((double *)(temp_c + ldc*5), ymm13); //c[0][5] c[1][5] c[2][5] c[3][5]

            temp_c += 4;
            temp_a += 4;
            m_rem = m_remainder - 4;
        }

        if(m_rem >= 2)
        {
            ymm3 = _mm256_setzero_pd();
            ymm5 = _mm256_setzero_pd();
            ymm7 = _mm256_setzero_pd();
            ymm9 = _mm256_setzero_pd();
            ymm11 = _mm256_setzero_pd();
            ymm13 = _mm256_setzero_pd();
            ymm15 = _mm256_setzero_pd();

            if(alpha_val != 0.0)
            {
                __m128d xmm5;
                xmm5 = _mm_loadu_pd((double const*)(temp_a));         //a[0][0] a[1][0]
                ymm0 = _mm256_broadcast_sd((double const*)(temp_a));  //a[0][0] a[0][0] a[0][0] a[0][0]
                ymm0 = _mm256_insertf128_pd(ymm1, xmm5, 0);           //a[0][0] a[1][0] a[0][0] a[1][0]

                ymm15 = _mm256_broadcast_sd((double const *)(&alpha_val));

                ymm0 = _mm256_mul_pd(ymm0,ymm15);    //ymm0 = (alpha_val*a[0][0] alpha_val*a[1][0] alpha_val*a[0][0] alpha_val*a[1][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b));  //b[0][0]
                ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3);   //ymm3 += (b[0][0]*a[0][0] b[0][0]*a[1][0] b[0][0]*a[0][0] b[0][0]*a[1][0]

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b + ldb * 1));  //b[0][1]
                ymm5 = _mm256_fmadd_pd(ymm2, ymm0, ymm5);   //ymm5 += (b[0][1]*a[0][0] b[0][1]*a[1][0] b[0][1]*a[0][0] b[0][1]*a[1][0]

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b + ldb * 2));  //b[0][2]
                ymm7 = _mm256_fmadd_pd(ymm2, ymm0, ymm7);   //ymm7 += (b[0][2]*a[0][0] b[0][2]*a[1][0] b[0][2]*a[0][0] b[0][2]*a[1][0]

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b + ldb * 3));  //b[0][3]
                ymm9 = _mm256_fmadd_pd(ymm2, ymm0, ymm9);  //ymm9 += (b[0][3]*a[0][0] b[0][3]*a[1][0] b[0][3]*a[0][0] b[0][3]*a[1][0]

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b + ldb * 4));  //b[0][4]
                ymm11 = _mm256_fmadd_pd(ymm2, ymm0, ymm11);//ymm11 += (b[0][4]*a[0][0] b[0][4]*a[1][0] b[0][4]*a[0][0] b[0][4]*a[1][0]

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b + ldb * 5));  //b[0][5]
                ymm13 = _mm256_fmadd_pd(ymm2, ymm0, ymm13);//ymm13 += (b[0][5]*a[0][0] b[0][5]*a[1][0] b[0][5]*a[0][0] b[0][5]*a[1][0]
            }

            if(beta_val != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_val));

                xmm5 = _mm_loadu_pd((double const*)(temp_c));          //c[0][0] c[1][0]
                ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0);            //c[0][0] c[1][0] c[0][0] c[1][0]

                ymm3 = _mm256_fmadd_pd(ymm15, ymm0, ymm3);             //ymm3 += (beta_val*c[0][0] beta_val*c[1][0] beta_val*c[0][0] beta_val*c[1][0])

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc));    //c[0][1] c[1][1]
                ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0);            //c[0][1] c[1][1] c[0][1] c[1][1]

                ymm5 = _mm256_fmadd_pd(ymm15, ymm0, ymm5);             //ymm5 += (beta_val*c[0][1] beta_val*c[1][1] beta_val*c[0][1] beta_val*c[1][1])

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc*2));  //c[0][2] c[1][2]
                ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0);            //c[0][2] c[1][2] c[0][2] c[1][2]

                ymm7 = _mm256_fmadd_pd(ymm15, ymm0, ymm7);             //ymm7 += (beta_val*c[0][2] beta_val*c[1][2] beta_val*c[0][2] beta_val*c[1][2])

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc*3));  //c[0][3] c[1][3]
                ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0);            //c[0][3] c[1][3] c[0][3] c[1][3]

                ymm9 = _mm256_fmadd_pd(ymm15, ymm0, ymm9);             //ymm7 += (beta_val*c[0][3] beta_val*c[1][3] beta_val*c[0][3] beta_val*c[1][3])

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc*4));  //c[0][4] c[1][4]
                ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0);            //c[0][4] c[1][4] c[0][4] c[1][4]

                ymm11 = _mm256_fmadd_pd(ymm15, ymm0, ymm11);           //ymm11 += (beta_val*c[0][4] beta_val*c[1][4] beta_val*c[0][4] beta_val*c[1][4])

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc*5));  //c[0][5] c[1][5]
                ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0);            //c[0][5] c[1][5] c[0][5] c[1][5]

                ymm13 = _mm256_fmadd_pd(ymm15, ymm0, ymm13);           //ymm13 += (beta_val*c[0][5] beta_val*c[1][5] beta_val*c[0][5] beta_val*c[1][5])
            }

            xmm5 = _mm256_extractf128_pd(ymm3, 0);            // xmm5 = ymm3[0] ymm3[1]
            _mm_storeu_pd((double *)(temp_c), xmm5);          //c[0][0] c[1][0]

            xmm5 = _mm256_extractf128_pd(ymm5, 0);            // xmm5 = ymm5[0] ymm5[1]
            _mm_storeu_pd((double *)(temp_c + ldc), xmm5);    //c[0][1] c[1][1]

            xmm5 = _mm256_extractf128_pd(ymm7, 0);            // xmm5 = ymm7[0] ymm7[1]
            _mm_storeu_pd((double *)(temp_c + ldc*2), xmm5);  //c[0][2] c[1][2]

            xmm5 = _mm256_extractf128_pd(ymm9, 0);            // xmm5 = ymm9[0] ymm9[1]
            _mm_storeu_pd((double *)(temp_c + ldc*3), xmm5);  //c[0][3] c[1][3]

            xmm5 = _mm256_extractf128_pd(ymm11, 0);           // xmm5 = ymm11[0] ymm11[1]
            _mm_storeu_pd((double *)(temp_c + ldc*4), xmm5);  //c[0][4] c[1][4]

            xmm5 = _mm256_extractf128_pd(ymm13, 0);           // xmm5 = ymm13[0] ymm13[1]
            _mm_storeu_pd((double *)(temp_c + ldc*5), xmm5);  //c[0][5] c[1][5]

            temp_c += 2;
            temp_a += 2;
            m_rem = m_rem - 2;
        }

        if(m_rem == 1)
        {
            ymm3 = _mm256_setzero_pd();
            ymm5 = _mm256_setzero_pd();
            ymm7 = _mm256_setzero_pd();
            ymm9 = _mm256_setzero_pd();
            ymm11 = _mm256_setzero_pd();
            ymm13 = _mm256_setzero_pd();
            ymm15 = _mm256_setzero_pd();

            if(alpha_val != 0.0)
            {
                ymm0 = _mm256_broadcast_sd((double const *)(temp_a));      //a[0][0] a[0][0] a[0][0] a[0][0]

                ymm15 = _mm256_broadcast_sd((double const *)(&alpha_val));
                ymm0 = _mm256_mul_pd(ymm0,ymm15);           //ymm0 = (alpha_val*a[0][0] alpha_val*a[0][0] alpha_val*a[0][0] alpha_val*a[0][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b));            //b[0][0]
                ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3);   //ymm3 += (b[0][0]*a[0][0] b[0][0]*a[0][0] b[0][0]*a[0][0] b[0][0]*a[0][0]

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b + ldb * 1));  //
                ymm5 = _mm256_fmadd_pd(ymm2, ymm0, ymm5);   //ymm5 += (b[0][1]*a[0][0] b[0][1]*a[0][0] b[0][1]*a[0][0] b[0][1]*a[0][0]

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b + ldb * 2));  //b[0][2]
                ymm7 = _mm256_fmadd_pd(ymm2, ymm0, ymm7);   //ymm7 += (b[0][2]*a[0][0] b[0][2]*a[0][0] b[0][2]*a[0][0] b[0][2]*a[0][0]

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b + ldb * 3));  //b[0][3]
                ymm9 = _mm256_fmadd_pd(ymm2, ymm0, ymm9);   //ymm9 += (b[0][3]*a[0][0] b[0][3]*a[0][0] b[0][3]*a[0][0] b[0][3]*a[0][0]

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b + ldb * 4));  //b[0][4]
                ymm11 = _mm256_fmadd_pd(ymm2, ymm0, ymm11); //ymm11 += (b[0][4]*a[0][0] b[0][4]*a[0][0] b[0][4]*a[0][0] b[0][4]*a[0][0]

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b + ldb * 5));  //b[0][5]
                ymm13 = _mm256_fmadd_pd(ymm2, ymm0, ymm13); //ymm13 += (b[0][5]*a[0][0] b[0][5]*a[0][0] b[0][5]*a[0][0] b[0][5]*a[0][0]
            }

            if(beta_val != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_val));

                ymm0 = _mm256_broadcast_sd((double const *)(temp_c));         //c[0][0] c[0][0] c[0][0] c[0][0]
                ymm3 = _mm256_fmadd_pd(ymm15, ymm0, ymm3);   //ymm3 += (beta_val*c[0][0] beta_val*c[0][0] beta_val*c[0][0] beta_val*c[0][0])

                ymm0 = _mm256_broadcast_sd((double const *)(temp_c + ldc));   //c[0][1] c[0][1] c[0][1] c[0][1]
                ymm5 = _mm256_fmadd_pd(ymm15, ymm0, ymm5);   //ymm5 += (beta_val*c[0][1] beta_val*c[0][1] beta_val*c[0][1] beta_val*c[0][1])

                ymm0 = _mm256_broadcast_sd((double const *)(temp_c + ldc*2)); //c[0][2] c[0][2] c[0][2] c[0][2]
                ymm7 = _mm256_fmadd_pd(ymm15, ymm0, ymm7);   //ymm7 += (beta_val*c[0][2] beta_val*c[0][2] beta_val*c[0][2] beta_val*c[0][2])

                ymm0 = _mm256_broadcast_sd((double const *)(temp_c + ldc*3)); //c[0][3] c[0][3] c[0][3] c[0][3]
                ymm9 = _mm256_fmadd_pd(ymm15, ymm0, ymm9);   //ymm9 += (beta_val*c[0][3] beta_val*c[0][3] beta_val*c[0][3] beta_val*c[0][3])

                ymm0 = _mm256_broadcast_sd((double const *)(temp_c + ldc*4)); //c[0][4] c[0][4] c[0][4] c[0][4]
                ymm11 = _mm256_fmadd_pd(ymm15, ymm0, ymm11); //ymm11 += (beta_val*c[0][4] beta_val*c[0][4] beta_val*c[0][4] beta_val*c[0][4])

                ymm0 = _mm256_broadcast_sd((double const *)(temp_c + ldc*5)); //c[0][5] c[0][5] c[0][5] c[0][5]
                ymm13 = _mm256_fmadd_pd(ymm15, ymm0, ymm13); //ymm13 += (beta_val*c[0][5] beta_val*c[0][5] beta_val*c[0][5] beta_val*c[0][5])
            }
            ymm0 = _mm256_blend_pd(ymm3, ymm0, 0x0E);                         // ymm0 = ymm3[0] ymm0[1] ymm0[2] ymm0[3]
            _mm_storel_pd((temp_c), _mm256_extractf128_pd(ymm0, 0));          //c[0][0]

            ymm0 = _mm256_blend_pd(ymm5, ymm0, 0x0E);                         // ymm0 = ymm5[0] ymm0[1] ymm0[2] ymm0[3]
            _mm_storel_pd((temp_c + ldc), _mm256_extractf128_pd(ymm0, 0));    //c[0][1]

            ymm0 = _mm256_blend_pd(ymm7, ymm0, 0x0E);                         // ymm0 = ymm7[0] ymm0[1] ymm0[2] ymm0[3]
            _mm_storel_pd((temp_c + ldc*2), _mm256_extractf128_pd(ymm0, 0));  //c[0][2]

            ymm0 = _mm256_blend_pd(ymm9, ymm0, 0x0E);                         // ymm0 = ymm9[0] ymm0[1] ymm0[2] ymm0[3]
            _mm_storel_pd((temp_c + ldc*3), _mm256_extractf128_pd(ymm0, 0));  //c[0][3]

            ymm0 = _mm256_blend_pd(ymm11, ymm0, 0x0E);                        // ymm0 = ymm11[0] ymm0[1] ymm0[2] ymm0[3]
            _mm_storel_pd((temp_c + ldc*4), _mm256_extractf128_pd(ymm0, 0));  //c[0][4]

            ymm0 = _mm256_blend_pd(ymm13, ymm0, 0x0E);                        // ymm0 = ymm13[0] ymm0[1] ymm0[2] ymm0[3]
            _mm_storel_pd((temp_c + ldc*5), _mm256_extractf128_pd(ymm0, 0));  //c[0][5]
        }
    }

    if(n_remainder >=4)
    {
        double* temp_b = b + (n - n_remainder)*ldb;
        double* temp_a = a;
        double* temp_c = c + (n - n_remainder)*ldc;

        for(dim_t i = 0;i < (m-D_MR+1);i=i+D_MR)
        {
            ymm3 = _mm256_setzero_pd();
            ymm4 = _mm256_setzero_pd();
            ymm5 = _mm256_setzero_pd();
            ymm6 = _mm256_setzero_pd();
            ymm7 = _mm256_setzero_pd();
            ymm8 = _mm256_setzero_pd();
            ymm9 = _mm256_setzero_pd();
            ymm10 = _mm256_setzero_pd();
            ymm15 = _mm256_setzero_pd();

            if(alpha_val != 0.0)
            {
                ymm0 = _mm256_loadu_pd((double const *)(temp_a));       //a[0][0] a[1][0] a[2][0] a[3][0]
                ymm1 = _mm256_loadu_pd((double const *)(temp_a + 4));   //a[4][0] a[5][0] a[6][0] a[7][0]

                ymm15 = _mm256_broadcast_sd((double const *)(&alpha_val));

                ymm0 = _mm256_mul_pd(ymm0,ymm15);      //ymm0 = (alpha_val*a[0][0] alpha_val*a[1][0] alpha_val*a[2][0] alpha_val*a[3][0])
                ymm1 = _mm256_mul_pd(ymm1,ymm15);      //ymm1 = (alpha_val*a[4][0] alpha_val*a[5][0] alpha_val*a[6][0] alpha_val*a[7][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b));            //b[0][0]
                ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3);   //ymm3 += (b[0][0]*a[0][0] b[0][0]*a[1][0] b[0][0]*a[2][0] b[0][0]*a[3][0])
                ymm4 = _mm256_fmadd_pd(ymm2, ymm1, ymm4);   //ymm4 += (b[0][0]*a[4][0] b[0][0]*a[5][0] b[0][0]*a[6][0] b[0][0]*a[7][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b + ldb * 1));  //b[0][1]
                ymm5 = _mm256_fmadd_pd(ymm2, ymm0, ymm5);   //ymm5 += (b[0][1]*a[0][0] b[0][1]*a[1][0] b[0][1]*a[2][0] b[0][1]*a[3][0])
                ymm6 = _mm256_fmadd_pd(ymm2, ymm1, ymm6);   //ymm6 += (b[0][1]*a[4][0] b[0][1]*a[5][0] b[0][1]*a[6][0] b[0][1]*a[7][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b + ldb * 2));  //b[0][2]
                ymm7 = _mm256_fmadd_pd(ymm2, ymm0, ymm7);   //ymm7 += (b[0][2]*a[0][0] b[0][2]*a[1][0] b[0][2]*a[2][0] b[0][2]*a[3][0])
                ymm8 = _mm256_fmadd_pd(ymm2, ymm1, ymm8);   //ymm8 += (b[0][2]*a[4][0] b[0][2]*a[5][0] b[0][2]*a[6][0] b[0][2]*a[7][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b + ldb * 3));  //b[0][3]
                ymm9 = _mm256_fmadd_pd(ymm2, ymm0, ymm9);   //ymm9 +=  (b[0][3]*a[0][0] b[0][3]*a[1][0] b[0][3]*a[2][0] b[0][3]*a[3][0])
                ymm10 = _mm256_fmadd_pd(ymm2, ymm1, ymm10); //ymm10 += (b[0][3]*a[4][0] b[0][3]*a[5][0] b[0][3]*a[6][0] b[0][3]*a[7][0])
            }

            if(beta_val != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_val));

                ymm0 = _mm256_loadu_pd((double const *)(temp_c));         //c[0][0] c[1][0] c[2][0] c[3][0]
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + 4));     //c[4][0] c[5][0] c[6][0] c[7][0]

                ymm3 = _mm256_fmadd_pd(ymm15, ymm0, ymm3);   //ymm3 += (beta_val*c[0][0] beta_val*c[1][0] beta_val*c[2][0] beta_val*c[3][0])
                ymm4 = _mm256_fmadd_pd(ymm15, ymm1, ymm4);   //ymm4 += (beta_val*c[4][0] beta_val*c[5][0] beta_val*c[6][0] beta_val*c[7][0])

                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc));    //c[0][1] c[1][1] c[2][1] c[3][1]
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc + 4));//c[4][1] c[5][1] c[6][1] c[7][1]

                ymm5 = _mm256_fmadd_pd(ymm15, ymm0, ymm5);   //ymm5 += (beta_val*c[0][1] beta_val*c[1][1] beta_val*c[2][1] beta_val*c[3][1])
                ymm6 = _mm256_fmadd_pd(ymm15, ymm1, ymm6);   //ymm6 += (beta_val*c[4][1] beta_val*c[5][1] beta_val*c[6][1] beta_val*c[7][1])

                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*2));    //c[0][2] c[1][2] c[2][2] c[3][2]
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc*2 + 4));//c[4][2] c[5][2] c[6][2] c[7][2]

                ymm7 = _mm256_fmadd_pd(ymm15, ymm0, ymm7);   //ymm7 += (beta_val*c[0][2] beta_val*c[1][2] beta_val*c[2][2] beta_val*c[3][2])
                ymm8 = _mm256_fmadd_pd(ymm15, ymm1, ymm8);   //ymm8 += (beta_val*c[4][2] beta_val*c[5][2] beta_val*c[6][2] beta_val*c[7][2])

                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*3));    //c[0][3] c[1][3] c[2][3] c[3][3]
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc*3 + 4));//c[4][3] c[5][3] c[6][3] c[7][3]

                ymm9 = _mm256_fmadd_pd(ymm15, ymm0, ymm9);   //ymm9 += (beta_val*c[0][3] beta_val*c[1][3] beta_val*c[2][3] beta_val*c[3][3])
                ymm10 = _mm256_fmadd_pd(ymm15, ymm1, ymm10); //ymm10 += (beta_val*c[4][3] beta_val*c[5][3] beta_val*c[6][3] beta_val*c[7][3])
            }

            _mm256_storeu_pd((double *)(temp_c), ymm3);             //c[0][0] c[1][0] c[2][0] c[3][0]
            _mm256_storeu_pd((double *)(temp_c + 4), ymm4);         //c[4][0] c[5][0] c[6][0] c[7][0]

            _mm256_storeu_pd((double *)(temp_c + ldc), ymm5);       //c[0][1] c[1][1] c[2][1] c[3][1]
            _mm256_storeu_pd((double *)(temp_c + ldc + 4), ymm6);   //c[4][1] c[5][1] c[6][1] c[7][1]

            _mm256_storeu_pd((double *)(temp_c + ldc*2), ymm7);     //c[0][2] c[1][2] c[2][2] c[3][2]
            _mm256_storeu_pd((double *)(temp_c + ldc*2 + 4), ymm8); //c[4][2] c[5][2] c[6][2] c[7][2]

            _mm256_storeu_pd((double *)(temp_c + ldc*3), ymm9);     //c[0][3] c[1][3] c[2][3] c[3][3]
            _mm256_storeu_pd((double *)(temp_c + ldc*3 +4), ymm10); //c[4][3] c[5][3] c[6][3] c[7][3]

            temp_c += D_MR;
            temp_a += D_MR;
        }

        dim_t m_rem = m_remainder;
        if(m_remainder >= 4)
        {
            ymm3 = _mm256_setzero_pd();
            ymm5 = _mm256_setzero_pd();
            ymm7 = _mm256_setzero_pd();
            ymm9 = _mm256_setzero_pd();
            ymm15 = _mm256_setzero_pd();

            if(alpha_val != 0.0)
            {
                ymm0 = _mm256_loadu_pd((double const *)(temp_a));                //a[0][0] a[1][0] a[2][0] a[3][0]

                ymm15 = _mm256_broadcast_sd((double const *)(&alpha_val));
                ymm0 = _mm256_mul_pd(ymm0,ymm15);       //ymm0 = (alpha_val*a[0][0] alpha_val*a[1][0] alpha_val*a[2][0] alpha_val*a[3][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b));            //b[0][0]
                ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3);  //ymm3 += (b[0][0]*a[0][0] b[0][0]*a[1][0] b[0][0]*a[2][0] b[0][0]*a[3][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b + ldb * 1));  //b[0][1]
                ymm5 = _mm256_fmadd_pd(ymm2, ymm0, ymm5);  //ymm5 += (b[0][1]*a[0][0] b[0][1]*a[1][0] b[0][1]*a[2][0] b[0][1]*a[3][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b + ldb * 2));  //b[0][2]
                ymm7 = _mm256_fmadd_pd(ymm2, ymm0, ymm7);  //ymm7 += (b[0][2]*a[0][0] b[0][2]*a[1][0] b[0][2]*a[2][0] b[0][2]*a[3][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b + ldb * 3));  //b[0][3]
                ymm9 = _mm256_fmadd_pd(ymm2, ymm0, ymm9);  //ymm9 += (b[0][3]*a[0][0] b[0][3]*a[1][0] b[0][3]*a[2][0] b[0][3]*a[3][0])
            }

            if(beta_val != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_val));

                ymm0 = _mm256_loadu_pd((double const *)(temp_c));          //c[0][0] c[1][0] c[2][0] c[3][0]
                ymm3 = _mm256_fmadd_pd(ymm15, ymm0, ymm3);  //ymm3 += (beta_val*c[0][0] beta_val*c[1][0] beta_val*c[2][0] beta_val*c[3][0])

                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc));    //c[0][1] c[1][1] c[2][1] c[3][1]
                ymm5 = _mm256_fmadd_pd(ymm15, ymm0, ymm5);  //ymm5 += (beta_val*c[0][1] beta_val*c[1][1] beta_val*c[2][1] beta_val*c[3][1])

                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*2));  //c[0][2] c[1][2] c[2][2] c[3][2]
                ymm7 = _mm256_fmadd_pd(ymm15, ymm0, ymm7);  //ymm7 += (beta_val*c[0][2] beta_val*c[1][2] beta_val*c[2][2] beta_val*c[3][2])

                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*3));  //c[0][3] c[1][3] c[2][3] c[3][3]
                ymm9 = _mm256_fmadd_pd(ymm15, ymm0, ymm9);  //ymm9 += (beta_val*c[0][3] beta_val*c[1][3] beta_val*c[2][3] beta_val*c[3][3])
            }
            _mm256_storeu_pd((double *)(temp_c), ymm3);          //c[0][0] c[1][0] c[2][0] c[3][0]
            _mm256_storeu_pd((double *)(temp_c + ldc), ymm5);    //c[0][1] c[1][1] c[2][1] c[3][1]
            _mm256_storeu_pd((double *)(temp_c + ldc*2), ymm7);  //c[0][2] c[1][2] c[2][2] c[3][2]
            _mm256_storeu_pd((double *)(temp_c + ldc*3), ymm9);  //c[0][3] c[1][3] c[2][3] c[3][3]

            temp_c += 4;
            temp_a += 4;
            m_rem = m_remainder - 4;
        }

        if(m_rem >= 2)
        {
            ymm3 = _mm256_setzero_pd();
            ymm5 = _mm256_setzero_pd();
            ymm7 = _mm256_setzero_pd();
            ymm9 = _mm256_setzero_pd();
            ymm15 = _mm256_setzero_pd();

            if(alpha_val != 0.0)
            {
                __m128d xmm5;
                xmm5 = _mm_loadu_pd((double const*)(temp_a));                   //a[0][0] a[1][0]
                ymm0 = _mm256_broadcast_sd((double const*)(temp_a));
                ymm0 = _mm256_insertf128_pd(ymm1, xmm5, 0);                     //a[0][0] a[1][0] a[0][0] a[1][0]

                ymm15 = _mm256_broadcast_sd((double const *)(&alpha_val));
                ymm0 = _mm256_mul_pd(ymm0,ymm15);    //ymm0 = (alpha_val*a[0][0] alpha_val*a[1][0] alpha_val*a[0][0] alpha_val*a[1][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b));           //b[0][0]
                ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3);  //ymm3 += (b[0][0]*a[0][0] b[0][0]*a[1][0] b[0][0]*a[0][0] b[0][0]*a[1][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b + ldb * 1)); //b[0][1]
                ymm5 = _mm256_fmadd_pd(ymm2, ymm0, ymm5);  //ymm5 += (b[0][1]*a[0][0] b[0][1]*a[1][0] b[0][1]*a[0][0] b[0][1]*a[1][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b + ldb * 2)); //b[0][2]
                ymm7 = _mm256_fmadd_pd(ymm2, ymm0, ymm7);  //ymm7 += (b[0][2]*a[0][0] b[0][2]*a[1][0] b[0][2]*a[0][0] b[0][2]*a[1][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b + ldb * 3)); //b[0][3]
                ymm9 = _mm256_fmadd_pd(ymm2, ymm0, ymm9);  //ymm9 += (b[0][3]*a[0][0] b[0][3]*a[1][0] b[0][3]*a[0][0] b[0][3]*a[1][0])
            }

            if(beta_val != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_val));

                xmm5 = _mm_loadu_pd((double const*)(temp_c));                 //c[0][0] c[1][0]
                ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0);                   //c[0][0] c[1][0] c[0][0] c[1][0]

                ymm3 = _mm256_fmadd_pd(ymm15, ymm0, ymm3);   //ymm3 += (beta_val*c[0][0] beta_val*c[1][0] beta_val*c[0][0] beta_val*c[1][0])

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc));           //c[0][1] c[1][1]
                ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0);                   //c[0][1] c[1][1] c[0][1] c[1][1]

                ymm5 = _mm256_fmadd_pd(ymm15, ymm0, ymm5);   //ymm5 += (beta_val*c[0][1] beta_val*c[1][1] beta_val*c[0][1] beta_val*c[1][1])

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc*2));         //c[0][2] c[1][2]
                ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0);                   //c[0][2] c[1][2] c[0][2] c[1][2]

                ymm7 = _mm256_fmadd_pd(ymm15, ymm0, ymm7);  //ymm7 += (beta_val*c[0][2] beta_val*c[1][2] beta_val*c[0][2] beta_val*c[1][2])

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc*3));         //c[0][3] c[1][3]
                ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0);                   //c[0][3] c[1][3] c[0][3] c[1][3]

                ymm9 = _mm256_fmadd_pd(ymm15, ymm0, ymm9);  //ymm9 += (beta_val*c[0][3] beta_val*c[1][3] beta_val*c[0][3] beta_val*c[1][3])
            }

            xmm5 = _mm256_extractf128_pd(ymm3, 0);              // xmm5 = ymm3[0] ymm3[1]
            _mm_storeu_pd((double *)(temp_c), xmm5);            //c[0][0] c[1][0]

            xmm5 = _mm256_extractf128_pd(ymm5, 0);              // xmm5 = ymm5[0] ymm5[1]
            _mm_storeu_pd((double *)(temp_c + ldc), xmm5);      //c[0][1] c[1][1]

            xmm5 = _mm256_extractf128_pd(ymm7, 0);              // xmm5 = ymm7[0] ymm7[1]
            _mm_storeu_pd((double *)(temp_c + ldc*2), xmm5);    //c[0][2] c[1][2]

            xmm5 = _mm256_extractf128_pd(ymm9, 0);              // xmm5 = ymm9[0] ymm9[1]
            _mm_storeu_pd((double *)(temp_c + ldc*3), xmm5);    //c[0][3] c[1][3]

            temp_c += 2;
            temp_a += 2;
            m_rem = m_rem - 2;
        }

        if(m_rem == 1)
        {
            ymm3 = _mm256_setzero_pd();
            ymm5 = _mm256_setzero_pd();
            ymm7 = _mm256_setzero_pd();
            ymm9 = _mm256_setzero_pd();
            ymm15 = _mm256_setzero_pd();

            if(alpha_val != 0.0)
            {
                ymm0 = _mm256_broadcast_sd((double const *)(temp_a));            //a[0][0]

                ymm15 = _mm256_broadcast_sd((double const *)(&alpha_val));
                ymm0 = _mm256_mul_pd(ymm0,ymm15);   //ymm0 = (alpha_val*a[0][0] alpha_val*a[0][0] alpha_val*a[0][0] alpha_val*a[0][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b));            //b[0][0]
                ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3);  //ymm3 += (b[0][0]*a[0][0] b[0][0]*a[0][0] b[0][0]*a[0][0] b[0][0]*a[0][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b + ldb * 1));  //b[0][1]
                ymm5 = _mm256_fmadd_pd(ymm2, ymm0, ymm5);  //ymm5 += (b[0][1]*a[0][0] b[0][1]*a[0][0] b[0][1]*a[0][0] b[0][1]*a[0][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b + ldb * 2));  //b[0][2]
                ymm7 = _mm256_fmadd_pd(ymm2, ymm0, ymm7);  //ymm7 += (b[0][2]*a[0][0] b[0][2]*a[0][0] b[0][2]*a[0][0] b[0][2]*a[0][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b + ldb * 3));  //b[0][3]
                ymm9 = _mm256_fmadd_pd(ymm2, ymm0, ymm9);  //ymm9 += (b[0][3]*a[0][0] b[0][3]*a[0][0] b[0][3]*a[0][0] b[0][3]*a[0][0])
            }

            if(beta_val != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_val));

                ymm0 = _mm256_broadcast_sd((double const *)(temp_c));          //c[0][0]
                ymm3 = _mm256_fmadd_pd(ymm15, ymm0, ymm3);      //ymm3 += (beta_val*c[0][0] beta_val*c[0][0] beta_val*c[0][0] beta_val*c[0][0])

                ymm0 = _mm256_broadcast_sd((double const *)(temp_c + ldc));    //c[0][1]
                ymm5 = _mm256_fmadd_pd(ymm15, ymm0, ymm5);      //ymm5 += (beta_val*c[0][1] beta_val*c[0][1] beta_val*c[0][1] beta_val*c[0][1])

                ymm0 = _mm256_broadcast_sd((double const *)(temp_c + ldc*2));  //c[0][2]
                ymm7 = _mm256_fmadd_pd(ymm15, ymm0, ymm7);      //ymm7 += (beta_val*c[0][2] beta_val*c[0][2] beta_val*c[0][2] beta_val*c[0][2])

                ymm0 = _mm256_broadcast_sd((double const *)(temp_c + ldc*3));  //c[0][3]
                ymm9 = _mm256_fmadd_pd(ymm15, ymm0, ymm9);      //ymm9 += (beta_val*c[0][3] beta_val*c[0][3] beta_val*c[0][3] beta_val*c[0][3])
            }
            ymm0 = _mm256_blend_pd(ymm3, ymm0, 0x0E);                          // ymm0 = ymm3[0] ymm0[1] ymm0[2] ymm0[3]
            _mm_storel_pd((temp_c), _mm256_extractf128_pd(ymm0, 0));           //c[0][0]

            ymm0 = _mm256_blend_pd(ymm5, ymm0, 0x0E);                          // ymm0 = ymm5[0] ymm0[1] ymm0[2] ymm0[3]
            _mm_storel_pd((temp_c + ldc), _mm256_extractf128_pd(ymm0, 0));     //c[0][1]

            ymm0 = _mm256_blend_pd(ymm7, ymm0, 0x0E);                          // ymm0 = ymm7[0] ymm0[1] ymm0[2] ymm0[3]
            _mm_storel_pd((temp_c + ldc*2), _mm256_extractf128_pd(ymm0, 0));   //c[0][2]

            ymm0 = _mm256_blend_pd(ymm9, ymm0, 0x0E);                          // ymm0 = ymm9[0] ymm0[1] ymm0[2] ymm0[3]
            _mm_storel_pd((temp_c + ldc*3), _mm256_extractf128_pd(ymm0, 0));   //c[0][3]
        }
        n_remainder = n_remainder - 4;
    }

    if(n_remainder >=2)
    {
        double* temp_b = b + (n - n_remainder)*ldb;
        double* temp_a = a;
        double* temp_c = c + (n - n_remainder)*ldc;

        for(dim_t i = 0;i < (m-D_MR+1);i=i+D_MR)
        {
            ymm3 = _mm256_setzero_pd();
            ymm4 = _mm256_setzero_pd();
            ymm5 = _mm256_setzero_pd();
            ymm6 = _mm256_setzero_pd();
            ymm15 = _mm256_setzero_pd();

            if(alpha_val != 0.0)
            {
                ymm0 = _mm256_loadu_pd((double const *)(temp_a));       //a[0][0] a[1][0] a[2][0] a[3][0]
                ymm1 = _mm256_loadu_pd((double const *)(temp_a + 4));   //a[4][0] a[5][0] a[6][0] a[7][0]

                ymm15 = _mm256_broadcast_sd((double const *)(&alpha_val));

                ymm0 = _mm256_mul_pd(ymm0,ymm15);   //ymm0 = (alpha_val*a[0][0] alpha_val*a[1][0] alpha_val*a[2][0] alpha_val*a[3][0])
                ymm1 = _mm256_mul_pd(ymm1,ymm15);   //ymm1 = (alpha_val*a[4][0] alpha_val*a[5][0] alpha_val*a[6][0] alpha_val*a[7][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b));  //b[0][0]
                ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3);  //ymm3 += (b[0][0]*a[0][0] b[0][0]*a[1][0] b[0][0]*a[2][0] b[0][0]*a[3][0])
                ymm4 = _mm256_fmadd_pd(ymm2, ymm1, ymm4);  //ymm4 += (b[0][0]*a[4][0] b[0][0]*a[5][0] b[0][0]*a[6][0] b[0][0]*a[7][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b + ldb));  //b[0][1]
                ymm5 = _mm256_fmadd_pd(ymm2, ymm0, ymm5);  //ymm5 += (b[0][1]*a[0][0] b[0][1]*a[1][0] b[0][1]*a[2][0] b[0][1]*a[3][0])
                ymm6 = _mm256_fmadd_pd(ymm2, ymm1, ymm6);  //ymm6 += (b[0][1]*a[4][0] b[0][1]*a[5][0] b[0][1]*a[6][0] b[0][1]*a[7][0])
            }

            if(beta_val != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_val));

                ymm0 = _mm256_loadu_pd((double const *)(temp_c));          //c[0][0] c[1][0] c[2][0] c[3][0]
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + 4));      //c[4][0] c[5][0] c[6][0] c[7][0]

                ymm3 = _mm256_fmadd_pd(ymm15, ymm0, ymm3);  //ymm3 += (beta_val*c[0][0] beta_val*c[1][0] beta_val*c[2][0] beta_val*c[3][0])
                ymm4 = _mm256_fmadd_pd(ymm15, ymm1, ymm4);  //ymm4 += (beta_val*c[4][0] beta_val*c[5][0] beta_val*c[6][0] beta_val*c[7][0])

                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc));    //c[0][1] c[1][1] c[2][1] c[3][1]
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc + 4));//c[4][1] c[5][1] c[6][1] c[7][1]

                ymm5 = _mm256_fmadd_pd(ymm15, ymm0, ymm5);  //ymm5 += (beta_val*c[0][1] beta_val*c[1][1] beta_val*c[2][1] beta_val*c[3][1])
                ymm6 = _mm256_fmadd_pd(ymm15, ymm1, ymm6);  //ymm6 += (beta_val*c[4][1] beta_val*c[5][1] beta_val*c[6][1] beta_val*c[7][1])
            }

            _mm256_storeu_pd((double *)(temp_c), ymm3);                    //c[0][0] c[1][0] c[2][0] c[3][0]
            _mm256_storeu_pd((double *)(temp_c + 4), ymm4);                //c[4][0] c[5][0] c[6][0] c[7][0]

            _mm256_storeu_pd((double *)(temp_c + ldc), ymm5);              //c[0][1] c[1][1] c[2][1] c[3][1]
            _mm256_storeu_pd((double *)(temp_c + ldc + 4), ymm6);          //c[4][1] c[5][1] c[6][1] c[7][1]

            temp_c += D_MR;
            temp_a += D_MR;
        }

        dim_t m_rem = m_remainder;
        if(m_remainder >= 4)
        {
            ymm3 = _mm256_setzero_pd();
            ymm5 = _mm256_setzero_pd();
            ymm15 = _mm256_setzero_pd();

            if(alpha_val != 0.0)
            {
                ymm0 = _mm256_loadu_pd((double const *)(temp_a));            //a[0][0] a[1][0] a[2][0] a[3][0]

                ymm15 = _mm256_broadcast_sd((double const *)(&alpha_val));
                ymm0 = _mm256_mul_pd(ymm0,ymm15);  //ymm0 = (alpha_val*a[0][0] alpha_val*a[1][0] alpha_val*a[2][0] alpha_val*a[3][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b));  //b[0][0]
                ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3);    //ymm3 += (b[0][0]*a[0][0] b[0][0]*a[1][0] b[0][0]*a[2][0] b[0][0]*a[3][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b + ldb)); //b[0][1]
                ymm5 = _mm256_fmadd_pd(ymm2, ymm0, ymm5);    //ymm5 += (b[0][1]*a[0][0] b[0][1]*a[1][0] b[0][1]*a[2][0] b[0][1]*a[3][0])
            }

            if(beta_val != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_val));

                ymm0 = _mm256_loadu_pd((double const *)(temp_c));           //c[0][0] c[1][0] c[2][0] c[3][0]
                ymm3 = _mm256_fmadd_pd(ymm15, ymm0, ymm3);  //ymm3 += (beta_val*c[0][0] beta_val*c[1][0] beta_val*c[2][0] beta_val*c[3][0])

                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc));     //c[0][1] c[1][1] c[2][1] c[3][1]
                ymm5 = _mm256_fmadd_pd(ymm15, ymm0, ymm5);  //ymm5 += (beta_val*c[0][1] beta_val*c[1][1] beta_val*c[2][1] beta_val*c[3][1])
            }

            _mm256_storeu_pd((double *)(temp_c), ymm3);        //c[0][0] c[1][0] c[2][0] c[3][0]
            _mm256_storeu_pd((double *)(temp_c + ldc), ymm5);  //c[0][1] c[1][1] c[2][1] c[3][1]

            temp_c += 4;
            temp_a += 4;
            m_rem = m_remainder - 4;
        }

        if(m_rem >= 2)
        {
            ymm3 = _mm256_setzero_pd();
            ymm5 = _mm256_setzero_pd();
            ymm15 = _mm256_setzero_pd();

            if(alpha_val != 0.0)
            {
                __m128d xmm5;
                xmm5 = _mm_loadu_pd((double const*)(temp_a));                    //a[0][0] a[1][0]
                ymm0 = _mm256_broadcast_sd((double const*)(temp_a));
                ymm0 = _mm256_insertf128_pd(ymm1, xmm5, 0);                      //a[0][0] a[1][0] a[0][0] a[1][0]

                ymm15 = _mm256_broadcast_sd((double const *)(&alpha_val));
                ymm0 = _mm256_mul_pd(ymm0,ymm15);   //ymm0 = (alpha_val*a[0][0] alpha_val*a[1][0] alpha_val*a[0][0] alpha_val*a[1][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b));            //b[0][0]
                ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3);   //ymm3 += (b[0][0]*a[0][0] b[0][0]*a[1][0] b[0][0]*a[0][0] b[0][0]*a[1][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b + ldb * 1));  //b[0][1]
                ymm5 = _mm256_fmadd_pd(ymm2, ymm0, ymm5);   //ymm5 += (b[0][1]*a[0][0] b[0][1]*a[1][0] b[0][1]*a[0][0] b[0][1]*a[1][0])
            }

            if(beta_val != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_val));

                xmm5 = _mm_loadu_pd((double const*)(temp_c));                   //c[0][0] c[1][0]
                ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0);                     //c[0][0] c[1][0] c[0][0] c[1][0]

                ymm3 = _mm256_fmadd_pd(ymm15, ymm0, ymm3);    //ymm3 += (beta_val*c[0][0] beta_val*c[1][0] beta_val*c[0][0] beta_val*c[1][0])

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc));             //c[0][1] c[1][1]
                ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0);                     //c[0][1] c[1][1] c[0][1] c[1][1]

                ymm5 = _mm256_fmadd_pd(ymm15, ymm0, ymm5);   //ymm5 += (beta_val*c[0][1] beta_val*c[1][1] beta_val*c[0][1] beta_val*c[1][1])
            }
            xmm5 = _mm256_extractf128_pd(ymm3, 0);           // xmm5 = ymm3[0] ymm3[1]
            _mm_storeu_pd((double *)(temp_c), xmm5);         //c[0][0] c[1][0]

            xmm5 = _mm256_extractf128_pd(ymm5, 0);           // xmm5 = ymm5[0] ymm5[1]
            _mm_storeu_pd((double *)(temp_c + ldc), xmm5);   //c[0][1] c[1][1]

            temp_c += 2;
            temp_a += 2;
            m_rem = m_rem - 2;
        }

        if(m_rem == 1)
        {
            ymm3 = _mm256_setzero_pd();
            ymm5 = _mm256_setzero_pd();
            ymm15 = _mm256_setzero_pd();

            if(alpha_val != 0.0)
            {
                ymm0 = _mm256_broadcast_sd((double const *)(temp_a));         //a[0][0] a[0][0] a[0][0] a[0][0]

                ymm15 = _mm256_broadcast_sd((double const *)(&alpha_val));
                ymm0 = _mm256_mul_pd(ymm0,ymm15);          //ymm0 = (alpha_val*a[0][0] alpha_val*a[0][0] alpha_val*a[0][0] alpha_val*a[0][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b)); //b[0][0]
                ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3);  //ymm3 += (b[0][0]*a[0][0] b[0][0]*a[0][0] b[0][0]*a[0][0] b[0][0]*a[0][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b + ldb)); //b[0][1]
                ymm5 = _mm256_fmadd_pd(ymm2, ymm0, ymm5);  //ymm5 += (b[0][1]*a[0][0] b[0][1]*a[0][0] b[0][1]*a[0][0] b[0][1]*a[0][0])
            }

            if(beta_val != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_val));

                ymm0 = _mm256_broadcast_sd((double const *)(temp_c));       //c[0][0] c[0][0] c[0][0] c[0][0]
                ymm3 = _mm256_fmadd_pd(ymm15, ymm0, ymm3);   //ymm3 += (beta_val*c[0][0] beta_val*c[0][0] beta_val*c[0][0] beta_val*c[0][0])

                ymm0 = _mm256_broadcast_sd((double const *)(temp_c + ldc)); //c[0][1] c[0][1] c[0][1] c[0][1]
                ymm5 = _mm256_fmadd_pd(ymm15, ymm0, ymm5);   //ymm5 += (beta_val*c[0][1] beta_val*c[0][1] beta_val*c[0][1] beta_val*c[0][1])
            }

            ymm0 = _mm256_blend_pd(ymm3, ymm0, 0x0E);                      // ymm0 = ymm3[0] ymm0[1] ymm0[2] ymm0[3]
            _mm_storel_pd((temp_c), _mm256_extractf128_pd(ymm0, 0));       // c[0][0]

            ymm0 = _mm256_blend_pd(ymm5, ymm0, 0x0E);                      //ymm0 = ymm5[0] ymm0[1] ymm0[2] ymm0[3]
            _mm_storel_pd((temp_c + ldc), _mm256_extractf128_pd(ymm0, 0)); // c[0][1]
        }
        n_remainder = n_remainder - 2;
    }

    if(n_remainder == 1)
    {
        double* temp_b = b + (n - n_remainder)*ldb;
        double* temp_a = a;
        double* temp_c = c + (n - n_remainder)*ldc;

        for(dim_t i = 0;i < (m-D_MR+1);i=i+D_MR)
        {
            ymm3 = _mm256_setzero_pd();
            ymm4 = _mm256_setzero_pd();
            ymm15 = _mm256_setzero_pd();

            if(alpha_val != 0.0)
            {
                ymm0 = _mm256_loadu_pd((double const *)(temp_a));           //a[0][0] a[1][0] a[2][0] a[3][0]
                ymm1 = _mm256_loadu_pd((double const *)(temp_a + 4));       //a[4][0] a[5][0] a[6][0] a[7][0]

                ymm15 = _mm256_broadcast_sd((double const *)(&alpha_val));

                ymm0 = _mm256_mul_pd(ymm0,ymm15);        //ymm0 = (alpha_val*a[0][0] alpha_val*a[1][0] alpha_val*a[2][0] alpha_val*a[3][0])
                ymm1 = _mm256_mul_pd(ymm1,ymm15);        //ymm1 = (alpha_val*a[4][0] alpha_val*a[5][0] alpha_val*a[6][0] alpha_val*a[7][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b));      //b[0][0]
                ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3);//ymm3 += (b[0][0]*a[0][0] b[0][0]*a[1][0] b[0][0]*a[2][0] b[0][0]*a[3][0])
                ymm4 = _mm256_fmadd_pd(ymm2, ymm1, ymm4);//ymm4 += (b[0][0]*a[4][0] b[0][0]*a[5][0] b[0][0]*a[6][0] b[0][0]*a[7][0])
            }

            if(beta_val != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_val));

                ymm0 = _mm256_loadu_pd((double const *)(temp_c));       //c[0][0] c[1][0] c[2][0] c[3][0]
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + 4));   //c[4][0] c[5][0] c[6][0] c[7][0]

                ymm3 = _mm256_fmadd_pd(ymm15, ymm0, ymm3);//ymm3 += (beta_val*c[0][0] beta_val*c[1][0] beta_val*c[2][0] beta_val*c[3][0])
                ymm4 = _mm256_fmadd_pd(ymm15, ymm1, ymm4);//ymm4 += (beta_val*c[4][0] beta_val*c[5][0] beta_val*c[6][0] beta_val*c[7][0])
            }

            _mm256_storeu_pd((double *)(temp_c), ymm3);                //c[0][0] c[1][0] c[2][0] c[3][0]
            _mm256_storeu_pd((double *)(temp_c + 4), ymm4);            //c[4][0] c[5][0] c[6][0] c[7][0]

            temp_c += D_MR;
            temp_a += D_MR;
        }

        dim_t m_rem = m_remainder;
        if(m_remainder >= 4)
        {
            ymm3 = _mm256_setzero_pd();
            ymm15 = _mm256_setzero_pd();

            if(alpha_val != 0.0)
            {
                ymm0 = _mm256_loadu_pd((double const *)(temp_a));          //a[0][0] a[1][0] a[2][0] a[3][0]

                ymm15 = _mm256_broadcast_sd((double const *)(&alpha_val));
                ymm0 = _mm256_mul_pd(ymm0,ymm15);   //ymm0 = (alpha_val*a[0][0] alpha_val*a[1][0] alpha_val*a[2][0] alpha_val*a[3][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b));      //b[0][0]
                ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3);//ymm3 += (b[0][0]*a[0][0] b[0][0]*a[1][0] b[0][0]*a[2][0] b[0][0]*a[3][0])
            }

            if(beta_val != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_val));

                ymm0 = _mm256_loadu_pd((double const *)(temp_c));         //c[0][0] c[1][0] c[2][0] c[3][0]
                ymm3 = _mm256_fmadd_pd(ymm15, ymm0, ymm3);//ymm3 += (beta_val*c[0][0] beta_val*c[1][0] beta_val*c[2][0] beta_val*c[3][0])
            }

            _mm256_storeu_pd((double *)(temp_c), ymm3);                   //c[0][0] c[1][0] c[2][0] c[3][0]

            temp_c += 4;
            temp_a += 4;
            m_rem = m_remainder - 4;
        }

        if(m_rem >= 2)
        {
            ymm3 = _mm256_setzero_pd();
            ymm15 = _mm256_setzero_pd();

            if(alpha_val != 0.0)
            {
                __m128d xmm5;
                xmm5 = _mm_loadu_pd((double const*)(temp_a));               //a[0][0] a[1][0]
                ymm0 = _mm256_broadcast_sd((double const*)(temp_a));
                ymm0 = _mm256_insertf128_pd(ymm1, xmm5, 0);                 //a[0][0] a[1][0] a[0][0] a[1][0]

                ymm15 = _mm256_broadcast_sd((double const *)(&alpha_val));

                ymm0 = _mm256_mul_pd(ymm0,ymm15);   //ymm0 = (alpha_val*a[0][0] alpha_val*a[1][0] alpha_val*a[0][0] alpha_val*a[1][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b));      //b[0][0]
                ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3);//ymm3 += (b[0][0]*a[0][0] b[0][0]*a[1][0] b[0][0]*a[0][0] b[0][0]*a[1][0])
            }

            if(beta_val != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_val));

                xmm5 = _mm_loadu_pd((double const*)(temp_c));              //c[0][0] c[1][0]
                ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0);                //c[0][0] c[1][0] c[0][0] c[1][0]

                ymm3 = _mm256_fmadd_pd(ymm15, ymm0, ymm3);//ymm3 += (beta_val*c[0][0] beta_val*c[1][0] beta_val*c[0][0] beta_val*c[1][0])
            }

            xmm5 = _mm256_extractf128_pd(ymm3, 0);                        // xmm5 = ymm3[0] ymm3[1]
            _mm_storeu_pd((double *)(temp_c), xmm5);                      //c[0][0] c[1][0]

            temp_c += 2;
            temp_a += 2;
            m_rem = m_rem - 2;
        }

        if(m_rem == 1)
        {
            ymm3 = _mm256_setzero_pd();
            ymm15 = _mm256_setzero_pd();

            if(alpha_val != 0.0)
            {
                ymm0 = _mm256_broadcast_sd((double const *)(temp_a));       //a[0][0] a[0][0] a[0][0] a[0][0]

                ymm15 = _mm256_broadcast_sd((double const *)(&alpha_val));
                ymm0 = _mm256_mul_pd(ymm0,ymm15);  //ymm0 = (alpha_val*a[0][0] alpha_val*a[0][0] alpha_val*a[0][0] alpha_val*a[0][0])

                ymm2 = _mm256_broadcast_sd((double const *)(temp_b));      //b[0][0]
                ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3);//ymm3 += (b[0][0]*a[0][0] b[0][0]*a[0][0] b[0][0]*a[0][0] b[0][0]*a[0][0])
            }

            if(beta_val != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_val));

                ymm0 = _mm256_broadcast_sd((double const *)(temp_c));       //c[0][0] c[0][0] c[0][0] c[0][0]
                ymm3 = _mm256_fmadd_pd(ymm15, ymm0, ymm3);//ymm3 += (beta_val*c[0][0] beta_val*c[0][0] beta_val*c[0][0] beta_val*c[0][0])
            }

            ymm0 = _mm256_blend_pd(ymm3, ymm0, 0x0E);                      // ymm0 = ymm3[0] ymm0[1] ymm0[2] ymm0[3]

            _mm_storel_pd((temp_c), _mm256_extractf128_pd(ymm0, 0));       //c[0][0]
        }
        n_remainder = n_remainder - 2;
    }
    return;
}
