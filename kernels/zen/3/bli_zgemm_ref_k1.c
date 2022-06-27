/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.

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
#include<complex.h>
#include "blis.h"

#include "immintrin.h"

#define Z_MR  4
#define Z_NR  5

// Macros for the main loop for M
#define SCALE_ALPHA_REAL_M_LOOP(r0,r1,r2,valr)  \
    if(valr != 0.0)  \
    {   \
        r2 = _mm256_broadcast_sd((double const *)(&valr)); \
        r0 = _mm256_mul_pd(r0,r2); \
        r1 = _mm256_mul_pd(r1,r2); \
    }   \

#define SCALE_ALPHA_IMAG_M_LOOP(r0,r1,r2,r3,rc1,rc2,vali)  \
    if(vali != 0.0)  \
    {   \
        r3 = _mm256_permute4x64_pd(rc1,0b10110001);  \
        r2 = _mm256_set_pd(1.0,-1.0,1.0,-1.0);  \
        r3 = _mm256_mul_pd(r3, r2); \
        r2 = _mm256_broadcast_sd((double const *)(&vali)); \
        r0 = _mm256_fmadd_pd(r3,r2,r0); \
        r3 = _mm256_permute4x64_pd(rc2,0b10110001);  \
        r2 = _mm256_set_pd(1.0,-1.0,1.0,-1.0);  \
        r3 = _mm256_mul_pd(r3, r2); \
        r2 = _mm256_broadcast_sd((double const *)(&vali)); \
        r1 = _mm256_fmadd_pd(r3,r2,r1); \
    }   \

#define NEG_PERM_M_LOOP(r0,r1,r2)    \
    r0 = _mm256_permute4x64_pd(r0,0b10110001);    \
    r1 = _mm256_permute4x64_pd(r1,0b10110001);    \
    r2 = _mm256_set_pd(1.0,-1.0,1.0,-1.0);  \
    r0 = _mm256_mul_pd(r2, r0); \
    r1 = _mm256_mul_pd(r2, r1); \

#define FMA_M_LOOP(rin_0,rin_1,rout_0,rout_1,rbc,loc)   \
    rbc = _mm256_broadcast_sd(loc);    \
    rout_0 = _mm256_fmadd_pd(rbc, rin_0, rout_0);   \
    rout_1 = _mm256_fmadd_pd(rbc, rin_1, rout_1);   \

#define SCALE_BETA_REAL_M_LOOP(rin_0,rin_1,rout_0,rout_1,rbc)   \
    rout_0 = _mm256_fmadd_pd(rbc, rin_0, rout_0);  \
    rout_1 = _mm256_fmadd_pd(rbc, rin_1, rout_1);  \

#define SCALE_BETA_IMAG_M_LOOP(rin_0,rin_1,rout_0,rout_1,rbc,rn)    \
    NEG_PERM_M_LOOP(rin_0,rin_1,rn);    \
    rout_0 = _mm256_fmadd_pd(rbc, rin_0, rout_0);    \
    rout_1 = _mm256_fmadd_pd(rbc, rin_1, rout_1);   \


// Macros for fringe cases with M
#define SCALE_ALPHA_REAL_M_FRINGE(r0,r2,val)     \
    if(val != 0.0)  \
    {   \
        r2 =  _mm256_broadcast_sd((double const *)(&val)); \
        r0 = _mm256_mul_pd(r0,r2); \
    }   \

#define SCALE_ALPHA_IMAG_M_FRINGE(r0,r2,r3,r4,val)  \
    if(val != 0.0)  \
    {   \
        r3 = _mm256_permute4x64_pd(r4,0b10110001);  \
        r2 = _mm256_set_pd(1.0,-1.0,1.0,-1.0);  \
        r3 = _mm256_mul_pd(r3, r2); \
        r2 = _mm256_broadcast_sd((double const *)(&val)); \
        r0 = _mm256_fmadd_pd(r3,r2,r0); \
    }   \

#define NEG_PERM_M_FRINGE(r0,r2)    \
    r0 = _mm256_permute4x64_pd(r0,0b10110001);    \
    r2 = _mm256_set_pd(1.0,-1.0,1.0,-1.0);  \
    r0 = _mm256_mul_pd(r2, r0); \

#define FMA_M_FRINGE(r_in,r_out,r_bc,loc)   \
    r_bc = _mm256_broadcast_sd(loc);    \
    r_out = _mm256_fmadd_pd(r_bc, r_in, r_out);   \

#define SCALE_BETA_REAL_M_FRINGE(rin_0,rout_0,rbc)   \
    rout_0 = _mm256_fmadd_pd(rbc, rin_0, rout_0);  \

#define SCALE_BETA_IMAG_M_FRINGE(rin_0,rout_0,rbc,rn)    \
    NEG_PERM_M_FRINGE(rin_0,rn);    \
    rout_0 = _mm256_fmadd_pd(rbc, rin_0, rout_0);    \


void bli_zgemm_ref_k1_nn
(
    dim_t  m,
    dim_t  n,
    dim_t  k,
    dcomplex*    alpha,
    dcomplex*    a, const inc_t lda,
    dcomplex*    b, const inc_t ldb,
    dcomplex*    beta,
    dcomplex*    c, const inc_t ldc
    )
{

    double alpha_valr, beta_valr;
    double alpha_vali, beta_vali;

    alpha_valr = alpha->real;
    beta_valr = beta->real;
    alpha_vali = alpha->imag;
    beta_vali = beta->imag;

    if((m == 0) || (n == 0) || (((alpha_valr == 0.0 && alpha_vali == 0.0) || (k == 0))
        && (beta_valr == 1.0 && beta_vali == 0.0)))
    {
        return;
    }
    dim_t m_remainder = (m % Z_MR);
    dim_t n_remainder = (n % Z_NR);

    //scratch registers
    __m256d ymm0, ymm1, ymm2, ymm3;
    __m256d ymm4, ymm5, ymm6, ymm7;
    __m256d ymm8, ymm9, ymm10, ymm11;
    __m256d ymm12, ymm13, ymm14, ymm15;
    __m128d xmm5;

    /* Form C = alpha*A*B + beta*c */
    for(dim_t j = 0;j < (n-Z_NR+1);j=j+Z_NR)
    {
        dcomplex* temp_b = b + j*ldb;
        dcomplex* temp_a = a;
        dcomplex* temp_c = c + j*ldc;

        for(dim_t i = 0;i < (m-Z_MR+1);i=i+Z_MR)
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

            if(alpha_valr != 0.0 || alpha_vali != 0.0)
            {
                /*
                    a. Perform alpha*A*B using temp_a, temp_b and alpha_valr, alpha_vali
                       where alpha_valr and/or alpha_vali is not zero.
                    b. This loop operates with 4x5 block size
                       along n dimension for every Z_NR columns of temp_b where
                       computing all Z_MR rows of temp_a.
                    c. Same approach is used in remaining fringe cases.
                */
                //R(a[0][0]) I(a[0][0]) R(a[1][0]) I(a[1][0])
                ymm0 = _mm256_loadu_pd((double const *)(temp_a));
                //R(a[2][0]) I(a[2][0]) R(a[3][0]) I(a[3][0])
                ymm1 = _mm256_loadu_pd((double const *)(temp_a + 2));

                ymm13 = ymm0;
                ymm14 = ymm1;

                _mm_prefetch((char*)(temp_a) + 64, _MM_HINT_T0);

                SCALE_ALPHA_REAL_M_LOOP(ymm0,ymm1,ymm15,alpha_valr);
                SCALE_ALPHA_IMAG_M_LOOP(ymm0,ymm1,ymm15,ymm2,ymm13,ymm14,alpha_vali);

                /*
                The result after scaling with alpha_valr and/or alpha_vali is as follows:
                For ymm0 :
                R(a[0][0]) = alpha_valr*R(a[0][0])-alpha_vali*I(a[0][0])
                I(a[0][0]) = alpha_valr*I(a[0][0])+alpha_vali*R[0][0]
                R(a[1][0]) = alpha_valr*R(a[1][0])-alpha_vali*I(a[1][0])
                I(a[1][0]) = alpha_valr*I(a[1][0])+alpha_vali*(R[1][0])

                For ymm1 :
                R(a[2][0]) = alpha_valr*R(a[2][0])-alpha_vali*I(a[2][0])
                I(a[2][0]) = alpha_valr*I(a[2][0])+alpha_vali*R[2][0]
                R(a[3][0]) = alpha_valr*R(a[3][0])-alpha_vali*I(a[3][0])
                I(a[3][0]) = alpha_valr*I(a[3][0])+alpha_vali*(R[3][0])
                */

                //Calculating using real part of complex number in B matrix
                //ymm3+=R(b[0][0])*R(a[0][0]) R(b[0][0])*I(a[0][0])
                //      R(b[0][0])*R(a[1][0]) R(b[0][0])*I(a[1][0])
                //ymm4+=R(b[0][0])*R(a[2][0]) R(b[0][0])*I(a[2][0])
                //      R(b[0][0])*R(a[3][0]) R(b[0][0])*I(a[3][0])
                FMA_M_LOOP(ymm0,ymm1,ymm3,ymm4,ymm2,(double const *)(temp_b));
                //ymm5+=R(b[0][1])*R(a[0][0]) R(b[0][1])*I(a[0][0])
                //      R(b[0][1])*R(a[1][0]) R(b[0][1])*I(a[1][0])
                //ymm6+=R(b[0][1])*R(a[0][0]) R(b[0][1])*I(a[0][0])
                //      R(b[0][1])*R(a[1][0]) R(b[0][1])*I(a[1][0])
                FMA_M_LOOP(ymm0,ymm1,ymm5,ymm6,ymm2,(double const *)(temp_b+ldb));
                //ymm7+=R(b[0][2])*R(a[0][0]) R(b[0][2])*I(a[0][0])
                //      R(b[0][2])*R(a[1][0]) R(b[0][2])*I(a[1][0])
                //ymm8+=R(b[0][2])*R(a[0][0]) R(b[0][2])*I(a[0][0])
                //      R(b[0][2])*R(a[1][0]) R(b[0][2])*I(a[1][0])
                FMA_M_LOOP(ymm0,ymm1,ymm7,ymm8,ymm2,(double const *)(temp_b+ldb*2));
                //ymm9+=R(b[0][3])*R(a[0][0]) R(b[0][3])*I(a[0][0])
                //      R(b[0][3])*R(a[1][0]) R(b[0][3])*I(a[1][0])
                //ymm10+=R(b[0][3])*R(a[0][0]) R(b[0][3])*I(a[0][0])
                //      R(b[0][3])*R(a[1][0]) R(b[0][3])*I(a[1][0])
                FMA_M_LOOP(ymm0,ymm1,ymm9,ymm10,ymm2,(double const *)(temp_b+ldb*3));
                //ymm11+=R(b[0][4])*R(a[0][0]) R(b[0][4])*I(a[0][0])
                //      R(b[0][4])*R(a[1][0]) R(b[0][4])*I(a[1][0])
                //ymm12+=R(b[0][4])*R(a[0][0]) R(b[0][4])*I(a[0][0])
                //      R(b[0][4])*R(a[1][0]) R(b[0][4])*I(a[1][0])
                FMA_M_LOOP(ymm0,ymm1,ymm11,ymm12,ymm2,(double const *)(temp_b+ldb*4));

                //Calculating using imaginary part of complex numbers in B matrix
                //Shuffling ymm0 and ymm1 in accordance to the requirement
                NEG_PERM_M_LOOP(ymm0,ymm1,ymm2);
                //ymm3+=I(b[0][0])*R(a[0][0]) I(b[0][0])*I(a[0][0])
                //      I(b[0][0])*R(a[1][0]) I(b[0][0])*I(a[1][0])
                //ymm4+=R(b[0][0])*R(a[2][0]) I(b[0][0])*I(a[2][0])
                //      I(b[0][0])*R(a[3][0]) I(b[0][0])*I(a[3][0])
                FMA_M_LOOP(ymm0,ymm1,ymm3,ymm4,ymm2,(double const *)(temp_b)+1);
                //ymm5+=I(b[0][1])*R(a[0][0]) I(b[0][1])*I(a[0][0])
                //      I(b[0][1])*R(a[1][0]) I(b[0][1])*I(a[1][0])
                //ymm6+=R(b[0][1])*R(a[0][0]) I(b[0][1])*I(a[0][0])
                //      I(b[0][1])*R(a[1][0]) I(b[0][1])*I(a[1][0])
                FMA_M_LOOP(ymm0,ymm1,ymm5,ymm6,ymm2,(double const *)(temp_b+ldb)+1);
                //ymm7+=I(b[0][2])*R(a[0][0]) I(b[0][2])*I(a[0][0])
                //      I(b[0][2])*R(a[1][0]) I(b[0][2])*I(a[1][0])
                //ymm8+=I(b[0][2])*R(a[0][0]) I(b[0][2])*I(a[0][0])
                //      I(b[0][2])*R(a[1][0]) I(b[0][2])*I(a[1][0])
                FMA_M_LOOP(ymm0,ymm1,ymm7,ymm8,ymm2,(double const *)(temp_b+ldb*2)+1);
                //ymm9+=I(b[0][3])*R(a[0][0]) I(b[0][3])*I(a[0][0])
                //      I(b[0][3])*R(a[1][0]) I(b[0][3])*I(a[1][0])
                //ymm10+=I(b[0][3])*R(a[0][0]) I(b[0][3])*I(a[0][0])
                //      I(b[0][3])*R(a[1][0]) I(b[0][3])*I(a[1][0])
                FMA_M_LOOP(ymm0,ymm1,ymm9,ymm10,ymm2,(double const *)(temp_b+ldb*3)+1);
                //ymm11+=I(b[0][4])*R(a[0][0]) I(b[0][4])*I(a[0][0])
                //      I(b[0][4])*R(a[1][0]) I(b[0][4])*I(a[1][0])
                //ymm12+=I(b[0][4])*R(a[0][0]) I(b[0][4])*I(a[0][0])
                //      I(b[0][4])*R(a[1][0]) I(b[0][4])*I(a[1][0])
                FMA_M_LOOP(ymm0,ymm1,ymm11,ymm12,ymm2,(double const *)(temp_b+ldb*4)+1);
            }
            if(beta_valr != 0.0)
            {
                /*
                    a. Perform beta*C using temp_c, beta_valr,
                       where beta_valr is not zero.
                    b. This loop operates with 4x5 block size
                       along n dimension for every Z_NR columns of temp_c where
                       computing all Z_MR rows of temp_c.
                    c. Accumulated alpha*A*B into registers will be added to beta*C
                    d. Same approach is used in remaining fringe cases.
                */
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_valr));

                //R(c[0][0]) I(c[0][0]) R(c[1][0]) I(c[1][0])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c));
                //R(c[2][0]) I(c[2][0]) R(c[3][0]) I(c[3][0])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + 2));
                //ymm3+=beta_valr*R(c[0][0]) beta_valr*I(c[0][0])
                //      beta_valr*R(c[1][0]) beta_valr*I(c[1][0])
                //ymm4+=beta_valr*R(c[2][0]) beta_valr*I(c[2][0])
                //      beta_valr*R(c[3][0]) beta_valr*I(c[3][0])
                SCALE_BETA_REAL_M_LOOP(ymm0,ymm1,ymm3,ymm4,ymm15);

                //R(c[0][1]) I(c[0][1]) R(c[1][1]) I(c[1][1])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc));
                //R(c[2][1]) I(c[2][1]) R(c[3][1]) I(c[3][1])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc + 2));
                //ymm5+=beta_valr*R(c[0][1]) beta_valr*I(c[0][1])
                //      beta_valr*R(c[1][1]) beta_valr*I(c[1][1])
                //ymm6+=beta_valr*R(c[2][1]) beta_valr*I(c[2][1])
                //      beta_valr*R(c[3][1]) beta_valr*I(c[3][1])
                SCALE_BETA_REAL_M_LOOP(ymm0,ymm1,ymm5,ymm6,ymm15);

                //R(c[0][2]) I(c[0][2]) R(c[1][2]) I(c[1][2])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*2));
                //R(c[2][2]) I(c[2][2]) R(c[3][2]) I(c[3][2])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc*2 + 2));
                //ymm7+=beta_valr*R(c[0][2]) beta_valr*I(c[0][2])
                //      beta_valr*R(c[1][2]) beta_valr*I(c[1][2])
                //ymm8+=beta_valr*R(c[2][2]) beta_valr*I(c[2][2])
                        //beta_valr*R(c[3][2]) beta_valr*I(c[3][2])
                SCALE_BETA_REAL_M_LOOP(ymm0,ymm1,ymm7,ymm8,ymm15);

                //R(c[0][3]) I(c[0][3]) R(c[1][3]) I(c[1][3])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*3));
                //R(c[2][3]) I(c[2][3]) R(c[3][3]) I(c[3][3])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc*3 + 2));
                //ymm9+=beta_valr*R(c[0][3]) beta_valr*I(c[0][3])
                //      beta_valr*R(c[1][3]) beta_valr*I(c[1][3])
                //ymm10+=beta_valr*R(c[2][3]) beta_valr*I(c[2][3])
                //      beta_valr*R(c[3][3]) beta_valr*I(c[3][3])
                SCALE_BETA_REAL_M_LOOP(ymm0,ymm1,ymm9,ymm10,ymm15);

                //R(c[0][4]) I(c[0][4]) R(c[1][4]) I(c[1][4])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*4));
                //R(c[2][4]) I(c[2][4]) R(c[3][4]) I(c[3][4])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc*4 + 2));
                //ymm11+=beta_valr*R(c[0][4]) beta_valr*I(c[0][4])
                //      beta_valr*R(c[1][4]) beta_valr*I(c[1][4])
                //ymm12+=beta_valr*R(c[2][4]) beta_valr*I(c[2][4])
                //      beta_valr*R(c[3][4]) beta_valr*I(c[3][4])
                SCALE_BETA_REAL_M_LOOP(ymm0,ymm1,ymm11,ymm12,ymm15);

            }
            if(beta_vali != 0.0)
            {
                /*
                    a. Perform beta*C using temp_c, beta_vali,
                       where beta_vali is not zero.
                    b. This loop operates with 4x5 block size
                       along n dimension for every Z_NR columns of temp_c where
                       computing all Z_MR rows of temp_c.
                    c. Accumulated alpha*A*B into registers will be added to beta*C
                    d. Same approach is used in remaining fringe cases.
                */

                ymm15 = _mm256_broadcast_sd((double const *)(&beta_vali));

                //R(c[0][0]) I(c[0][0]) R(c[1][0]) I(c[1][0])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c));
                //R(c[2][0]) I(c[2][0]) R(c[3][0]) I(c[3][0])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + 2));
                //ymm3+=beta_vali*(-I(c[0][0])) beta_vali*R(c[0][0])
                //      beta_vali*(-I(c[1][0])) beta_vali*R(c[1][0])
                //ymm4+=beta_vali*(-I(c[2][0])) beta_vali*R(c[2][0])
                //      beta_vali*(-I(c[3][0])) beta_vali*R(c[3][0])
                SCALE_BETA_IMAG_M_LOOP(ymm0,ymm1,ymm3,ymm4,ymm15,ymm2);

                //R(c[0][1]) I(c[0][1]) R(c[1][1]) I(c[1][1])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc));
                //R(c[2][1]) I(c[2][1]) R(c[3][1]) I(c[3][1])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc + 2));
                //ymm5+=beta_vali*(-I(c[0][1])) beta_vali*R(c[0][1])
                //      beta_vali*(-I(c[1][1])) beta_vali*R(c[1][1])
                //ymm6+=beta_vali*(-I(c[2][1])) beta_vali*R(c[2][1])
                //      beta_vali*(-I(c[3][1])) beta_vali*R(c[3][1])
                SCALE_BETA_IMAG_M_LOOP(ymm0,ymm1,ymm5,ymm6,ymm15,ymm2);

                //R(c[0][2]) I(c[0][2]) R(c[1][2]) I(c[1][2])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*2));
                //R(c[2][2]) I(c[2][2]) R(c[3][2]) I(c[3][2])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc*2 + 2));
                //ymm7+=beta_vali*(-I(c[0][2])) beta_vali*R(c[0][2])
                //      beta_vali*(-I(c[1][2])) beta_vali*R(c[1][2])
                //ymm8+=beta_vali*(-I(c[2][2])) beta_vali*R(c[2][2])
                //      beta_vali*(-I(c[3][2])) beta_vali*R(c[3][2])
                SCALE_BETA_IMAG_M_LOOP(ymm0,ymm1,ymm7,ymm8,ymm15,ymm2);

                //R(c[0][3]) I(c[0][3]) R(c[1][3]) I(c[1][3])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*3));
                //R(c[2][3]) I(c[2][3]) R(c[3][3]) I(c[3][3])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc*3 + 2));
                //ymm9+=beta_vali*(-I(c[0][3])) beta_vali*R(c[0][3])
                //      beta_vali*(-I(c[1][3])) beta_vali*R(c[1][3])
                //ymm10+=beta_vali*(-I(c[2][3])) beta_vali*R(c[2][3])
                //      beta_vali*(-I(c[3][3])) beta_vali*R(c[3][3])
                SCALE_BETA_IMAG_M_LOOP(ymm0,ymm1,ymm9,ymm10,ymm15,ymm2);

                //R(c[0][4]) I(c[0][4]) R(c[1][4]) I(c[1][4])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*4));
                //R(c[2][4]) I(c[2][4]) R(c[3][4]) I(c[3][4])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc*4 + 2));
                //ymm11+=beta_vali*(-I(c[0][4])) beta_vali*R(c[0][4])
                //      beta_vali*(-I(c[1][4])) beta_vali*R(c[1][4])
                //ymm12+=beta_vali*(-I(c[2][4])) beta_vali*R(c[2][4])
                //      beta_vali*(-I(c[3][4])) beta_vali*R(c[3][4])
                SCALE_BETA_IMAG_M_LOOP(ymm0,ymm1,ymm11,ymm12,ymm15,ymm2);
            }
            /*
            The scaling has been done sequentially as follows:
            - If alpha_valr is not 0, it is used for scaling A
            - If alpha_vali is not 0, it is used for scaling A using permutation
              and selective negation, after loading
            - If beta_valr is not 0, is is used for scaling C
            - If beta_vali is not 0, it is used for scaling C using permutation
              and selective negation, after loading

            The results are accumalated in accordance to the non zero scalar values,
            and similar approach is followed in fringe cases
            */

            _mm256_storeu_pd((double *)(temp_c), ymm3);
            _mm256_storeu_pd((double *)(temp_c + 2), ymm4);

            _mm256_storeu_pd((double *)(temp_c + ldc), ymm5);
            _mm256_storeu_pd((double *)(temp_c + ldc + 2), ymm6);

            _mm256_storeu_pd((double *)(temp_c + ldc*2), ymm7);
            _mm256_storeu_pd((double *)(temp_c + ldc*2 + 2), ymm8);

            _mm256_storeu_pd((double *)(temp_c + ldc*3), ymm9);
            _mm256_storeu_pd((double *)(temp_c + ldc*3 + 2), ymm10);

            _mm256_storeu_pd((double *)(temp_c + ldc*4), ymm11);
            _mm256_storeu_pd((double *)(temp_c + ldc*4 + 2), ymm12);

            temp_c+=Z_MR;
            temp_a+=Z_MR;
        }

        dim_t m_rem=m_remainder;
        if(m_rem>=2)
        {
            ymm3 = _mm256_setzero_pd();
            ymm5 = _mm256_setzero_pd();
            ymm7 = _mm256_setzero_pd();
            ymm9 = _mm256_setzero_pd();
            ymm11 = _mm256_setzero_pd();

            if(alpha_valr != 0.0 || alpha_vali != 0.0)
            {

                //R(a[0][0]) I(a[0][0]) R(a[1][0]) I(a[1][0])
                ymm0 = _mm256_loadu_pd((double const *)(temp_a));
                ymm13 = ymm0;
                SCALE_ALPHA_REAL_M_FRINGE(ymm0,ymm15,alpha_valr);
                SCALE_ALPHA_IMAG_M_FRINGE(ymm0,ymm15,ymm2,ymm13,alpha_vali);
                /*
                The result after scaling with alpha_valr and/or alpha_vali is as follows:
                For ymm0 :
                R(a[0][0]) = alpha_valr*R(a[0][0])-alpha_vali*I(a[0][0])
                I(a[0][0]) = alpha_valr*I(a[0][0])+alpha_vali*R[0][0]
                R(a[1][0]) = alpha_valr*R(a[1][0])-alpha_vali*I(a[1][0])
                I(a[1][0]) = alpha_valr*I(a[1][0])+alpha_vali*(R[1][0])
                */

                //Calculating using real part of complex number in B matrix
                //ymm3+=R(b[0][0])*R(a[0][0]) R(b[0][0])*I(a[0][0])
                //      R(b[0][0])*R(a[1][0]) R(b[0][0])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm3,ymm2,(double const *)(temp_b));
                //ymm5+=R(b[0][1])*R(a[0][0]) R(b[0][1])*I(a[0][0])
                //      R(b[0][1])*R(a[1][0]) R(b[0][1])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm5,ymm2,(double const *)(temp_b+ldb));
                //ymm7+=R(b[0][2])*R(a[0][0]) R(b[0][2])*I(a[0][0])
                //      R(b[0][2])*R(a[1][0]) R(b[0][2])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm7,ymm2,(double const *)(temp_b+ldb*2));
                //ymm9+=R(b[0][3])*R(a[0][0]) R(b[0][3])*I(a[0][0])
                //      R(b[0][3])*R(a[1][0]) R(b[0][3])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm9,ymm2,(double const *)(temp_b+ldb*3));
                //ymm11+=R(b[0][4])*R(a[0][0]) R(b[0][4])*I(a[0][0])
                //      R(b[0][4])*R(a[1][0]) R(b[0][4])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm11,ymm2,(double const *)(temp_b+ldb*4));

                //Calculating using imaginary part of complex numbers in B matrix
                //Shuffling ymm0 in accordance to the requirement
                NEG_PERM_M_FRINGE(ymm0,ymm2);

                // ymm3+=I(b[0][0])*R(a[0][0]) I(b[0][0])*I(a[0][0])
                //      I(b[0][0])*R(a[1][0]) I(b[0][0])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm3,ymm2,(double const *)(temp_b)+1);
                //ymm5+=I(b[0][1])*R(a[0][0]) I(b[0][1])*I(a[0][0])
                //      I(b[0][1])*R(a[1][0]) I(b[0][1])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm5,ymm2,(double const *)(temp_b+ldb)+1);
                //ymm7+=I(b[0][2])*R(a[0][0]) I(b[0][2])*I(a[0][0])
                //      I(b[0][2])*R(a[1][0]) I(b[0][2])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm7,ymm2,(double const *)(temp_b+ldb*2)+1);
                //ymm9+=I(b[0][3])*R(a[0][0]) I(b[0][3])*I(a[0][0])
                //      I(b[0][3])*R(a[1][0]) I(b[0][3])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm9,ymm2,(double const *)(temp_b+ldb*3)+1);
                //ymm11+=I(b[0][4])*R(a[0][0]) I(b[0][4])*I(a[0][0])
                //      I(b[0][4])*R(a[1][0]) I(b[0][4])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm11,ymm2,(double const *)(temp_b+ldb*4)+1);

            }

            if(beta_valr != 0.0)
            {

                ymm15 = _mm256_broadcast_sd((double const *)(&beta_valr));

                //R(c[0][0]) I(c[0][0]) R(c[1][0]) I(c[1][0])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c));
                //ymm3+=beta_valr*R(c[0][0]) beta_valr*I(c[0][0])
                //      beta_valr*R(c[1][0]) beta_valr*I(c[1][0])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm3,ymm15);

                //R(c[0][1]) I(c[0][1]) R(c[1][1]) I(c[1][1])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc));
                //ymm5+=beta_valr*R(c[0][1]) beta_valr*I(c[0][1])
                //      beta_valr*R(c[1][1]) beta_valr*I(c[1][1])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm5,ymm15);

                //R(c[0][2]) I(c[0][2]) R(c[1][2]) I(c[1][2])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*2));
                //ymm7+=beta_valr*R(c[0][2]) beta_valr*I(c[0][2])
                //      beta_valr*R(c[1][2]) beta_valr*I(c[1][2])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm7,ymm15);

                //R(c[0][3]) I(c[0][3]) R(c[1][3]) I(c[1][3])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*3));
                //ymm9+=beta_valr*R(c[0][3]) beta_valr*I(c[0][3])
                //      beta_valr*R(c[1][3]) beta_valr*I(c[1][3])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm9,ymm15);

                //R(c[0][4]) I(c[0][4]) R(c[1][4]) I(c[1][4])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*4));
                //ymm11+=beta_valr*R(c[0][4]) beta_valr*I(c[0][4])
                //      beta_valr*R(c[1][4]) beta_valr*I(c[1][4])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm11,ymm15);

            }

            if(beta_vali != 0.0)
            {

                ymm15 = _mm256_broadcast_sd((double const *)(&beta_vali));

                //R(c[0][0]) I(c[0][0]) R(c[1][0]) I(c[1][0])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c));
                //ymm3+=beta_vali*(-I(c[0][0])) beta_vali*R(c[0][0])
                //      beta_vali*(-I(c[1][0])) beta_vali*R(c[1][0])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm3,ymm15,ymm2);

                //R(c[0][1]) I(c[0][1]) R(c[1][1]) I(c[1][1])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc));
                //ymm5+=beta_vali*(-I(c[0][1])) beta_vali*R(c[0][1])
                //      beta_vali*(-I(c[1][1])) beta_vali*R(c[1][1])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm5,ymm15,ymm2);

                //R(c[0][2]) I(c[0][2]) R(c[1][2]) I(c[1][2])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*2));
                //ymm7+=beta_vali*(-I(c[0][2])) beta_vali*R(c[0][2])
                //      beta_vali*(-I(c[1][2])) beta_vali*R(c[1][2])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm7,ymm15,ymm2);

                //R(c[0][3]) I(c[0][3]) R(c[1][3]) I(c[1][3])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*3));
                //ymm9+=beta_vali*(-I(c[0][3])) beta_vali*R(c[0][3])
                //      beta_vali*(-I(c[1][3])) beta_vali*R(c[1][3])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm9,ymm15,ymm2);

                //R(c[0][4]) I(c[0][4]) R(c[1][4]) I(c[1][4])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*4));
                //ymm11+=beta_vali*(-I(c[0][4])) beta_vali*R(c[0][4])
                //      beta_vali*(-I(c[1][4])) beta_vali*R(c[1][4])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm11,ymm15,ymm2);
            }

            /*
            The scaling has been done sequentially as follows:
            - If alpha_valr is not 0, it is used for scaling A
            - If alpha_vali is not 0, it is used for scaling A using permutation
              and selective negation, after loading
            - If beta_valr is not 0, is is used for scaling C
            - If beta_vali is not 0, it is used for scaling C using permutation
              and selective negation, after loading

            The results are accumalated in accordance to the non zero scalar values,
            and similar approach is followed in fringe cases
            */

            _mm256_storeu_pd((double *)(temp_c), ymm3);
            _mm256_storeu_pd((double *)(temp_c + ldc), ymm5);
            _mm256_storeu_pd((double *)(temp_c + ldc*2), ymm7);
            _mm256_storeu_pd((double *)(temp_c + ldc*3), ymm9);
            _mm256_storeu_pd((double *)(temp_c + ldc*4), ymm11);

            temp_c+=2;
            temp_a+=2;

            m_rem -= 2;
        }

        if(m_rem==1)
        {

            xmm5 = _mm_setzero_pd();
            ymm3 = _mm256_setzero_pd();
            ymm5 = _mm256_setzero_pd();
            ymm7 = _mm256_setzero_pd();
            ymm9 = _mm256_setzero_pd();
            ymm11 = _mm256_setzero_pd();

            if(alpha_valr != 0.0 || alpha_vali != 0.0)
            {
                xmm5 = _mm_loadu_pd((double const*)(temp_a));//R(a[0][0]) I(a[0][0])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(a[0][0]) I(a[0][0])
                ymm13 = ymm0;

                SCALE_ALPHA_REAL_M_FRINGE(ymm0,ymm15,alpha_valr);
                SCALE_ALPHA_IMAG_M_FRINGE(ymm0,ymm15,ymm2,ymm13,alpha_vali);

                //Calculating using real part of complex number in B matrix
                //ymm3+=R(b[0][0])*R(a[0][0]) R(b[0][0])*I(a[0][0])
                //      R(b[0][0])*R(a[1][0]) R(b[0][0])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm3,ymm2,(double const *)(temp_b));
                //ymm5+=R(b[0][1])*R(a[0][0]) R(b[0][1])*I(a[0][0])
                //      R(b[0][1])*R(a[1][0]) R(b[0][1])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm5,ymm2,(double const *)(temp_b+ldb));
                //ymm7+=R(b[0][2])*R(a[0][0]) R(b[0][2])*I(a[0][0])
                //      R(b[0][2])*R(a[1][0]) R(b[0][2])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm7,ymm2,(double const *)(temp_b+ldb*2));
                //ymm9+=R(b[0][3])*R(a[0][0]) R(b[0][3])*I(a[0][0])
                //      R(b[0][3])*R(a[1][0]) R(b[0][3])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm9,ymm2,(double const *)(temp_b+ldb*3));
                //ymm11+=R(b[0][4])*R(a[0][0]) R(b[0][4])*I(a[0][0])
                //      R(b[0][4])*R(a[1][0]) R(b[0][4])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm11,ymm2,(double const *)(temp_b+ldb*4));

                //Calculating using imaginary part of complex numbers in B matrix
                //Shuffling ymm0 in accordance to the requirement
                NEG_PERM_M_FRINGE(ymm0,ymm2);

                // ymm3+=I(b[0][0])*R(a[0][0]) I(b[0][0])*I(a[0][0])
                //      I(b[0][0])*R(a[1][0]) I(b[0][0])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm3,ymm2,(double const *)(temp_b)+1);
                //ymm5+=I(b[0][1])*R(a[0][0]) I(b[0][1])*I(a[0][0])
                //      I(b[0][1])*R(a[1][0]) I(b[0][1])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm5,ymm2,(double const *)(temp_b+ldb)+1);
                //ymm7+=I(b[0][2])*R(a[0][0]) I(b[0][2])*I(a[0][0])
                //      I(b[0][2])*R(a[1][0]) I(b[0][2])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm7,ymm2,(double const *)(temp_b+ldb*2)+1);
                //ymm9+=I(b[0][3])*R(a[0][0]) I(b[0][3])*I(a[0][0])
                //      I(b[0][3])*R(a[1][0]) I(b[0][3])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm9,ymm2,(double const *)(temp_b+ldb*3)+1);
                //ymm11+=I(b[0][4])*R(a[0][0]) I(b[0][4])*I(a[0][0])
                //      I(b[0][4])*R(a[1][0]) I(b[0][4])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm11,ymm2,(double const *)(temp_b+ldb*4)+1);

            }
            if(beta_valr != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_valr));

                xmm5 = _mm_loadu_pd((double const*)(temp_c));//R(c[0][0]) I(c[0][0])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][0]) I(c[0][0])
                //ymm3+=beta_valr*R(c[0][0]) beta_valr*I(c[0][0])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm3,ymm15);

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc));//R(c[0][1]) I(c[0][1])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][1]) I(c[0][1])
                //ymm5+=beta_valr*R(c[0][1]) beta_valr*I(c[0][1])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm5,ymm15);

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc * 2));//R(c[0][2]) I(c[0][2])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][2]) I(c[0][2])
                //ymm7+=beta_valr*R(c[0][2]) beta_valr*I(c[0][2])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm7,ymm15);

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc * 3));//R(c[0][3]) I(c[0][3])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][3]) I(c[0][3])
                //ymm9+=beta_valr*R(c[0][3]) beta_valr*I(c[0][3])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm9,ymm15);

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc * 4));//R(c[0][4]) I(c[0][4])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][4]) I(c[0][4])
                //ymm11+=beta_valr*R(c[0][4]) beta_valr*I(c[0][4])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm11,ymm15);
            }
            if(beta_vali != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_vali));

                xmm5 = _mm_loadu_pd((double const*)(temp_c));//R(c[0][0]) I(c[0][0])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][0]) I(c[0][0])
                //ymm3+=beta_vali*(-I(c[0][0])) beta_vali*R(c[0][0])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm3,ymm15,ymm2);

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc));//R(c[0][1]) I(c[0][1])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][1]) I(c[0][1])
                //ymm5+=beta_vali*(-I(c[0][1])) beta_vali*R(c[0][1])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm5,ymm15,ymm2);

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc * 2));//R(c[0][2]) I(c[0][2])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][2]) I(c[0][2])
                //ymm7+=beta_vali*(-I(c[0][2])) beta_vali*R(c[0][2])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm7,ymm15,ymm2);

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc * 3));//R(c[0][3]) I(c[0][3])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][3]) I(c[0][3])
                //ymm9+=beta_vali*(-I(c[0][3])) beta_vali*R(c[0][3])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm9,ymm15,ymm2);

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc * 4));//R(c[0][4]) I(c[0][4])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][4]) I(c[0][4])
                //ymm11+=beta_vali*(-I(c[0][4])) beta_vali*R(c[0][4])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm11,ymm15,ymm2);

            }

            xmm5 = _mm256_extractf128_pd(ymm3, 0);
            _mm_storeu_pd((double *)(temp_c), xmm5);

            xmm5 = _mm256_extractf128_pd(ymm5, 0);
            _mm_storeu_pd((double *)(temp_c + ldc), xmm5);

            xmm5 = _mm256_extractf128_pd(ymm7, 0);
            _mm_storeu_pd((double *)(temp_c + ldc*2), xmm5);

            xmm5 = _mm256_extractf128_pd(ymm9, 0);
            _mm_storeu_pd((double *)(temp_c + ldc*3), xmm5);

            xmm5 = _mm256_extractf128_pd(ymm11, 0);
            _mm_storeu_pd((double *)(temp_c + ldc*4), xmm5);

        }

    }
    if(n_remainder==4)
    {
        dcomplex* temp_b = b + (n - n_remainder)*ldb;
        dcomplex* temp_a = a;
        dcomplex* temp_c = c + (n - n_remainder)*ldc;
        for(dim_t i = 0;i < (m-Z_MR+1);i=i+Z_MR)
        {
            ymm3 = _mm256_setzero_pd();
            ymm4 = _mm256_setzero_pd();
            ymm5 = _mm256_setzero_pd();
            ymm6 = _mm256_setzero_pd();
            ymm7 = _mm256_setzero_pd();
            ymm8 = _mm256_setzero_pd();
            ymm9 = _mm256_setzero_pd();
            ymm10 = _mm256_setzero_pd();

            if(alpha_valr != 0.0 || alpha_vali != 0.0)
            {
                /*
                    a. Perform alpha*A*B using temp_a, temp_b and alpha_valr, alpha_vali
                       where alpha_valr and/or alpha_vali is not zero.
                    b. This loop operates with 4x5 block size
                       along n dimension for every Z_NR columns of temp_b where
                       computing all Z_MR rows of temp_a.
                    c. Same approach is used in remaining fringe cases.
                */

                //R(a[0][0]) I(a[0][0]) R(a[1][0]) I(a[1][0])
                ymm0 = _mm256_loadu_pd((double const *)(temp_a));
                //R(a[2][0]) I(a[2][0]) R(a[3][0]) I(a[3][0])
                ymm1 = _mm256_loadu_pd((double const *)(temp_a + 2));

                ymm13 = ymm0;
                ymm14 = ymm1;

                _mm_prefetch((char*)(temp_a) + 64, _MM_HINT_T0);

                SCALE_ALPHA_REAL_M_LOOP(ymm0,ymm1,ymm15,alpha_valr);
                SCALE_ALPHA_IMAG_M_LOOP(ymm0,ymm1,ymm15,ymm2,ymm13,ymm14,alpha_vali);

                /*
                The result after scaling with alpha_valr and/or alpha_vali is as follows:
                For ymm0 :
                R(a[0][0]) = alpha_valr*R(a[0][0])-alpha_vali*I(a[0][0])
                I(a[0][0]) = alpha_valr*I(a[0][0])+alpha_vali*R[0][0]
                R(a[1][0]) = alpha_valr*R(a[1][0])-alpha_vali*I(a[1][0])
                I(a[1][0]) = alpha_valr*I(a[1][0])+alpha_vali*(R[1][0])

                For ymm1 :
                R(a[2][0]) = alpha_valr*R(a[2][0])-alpha_vali*I(a[2][0])
                I(a[2][0]) = alpha_valr*I(a[2][0])+alpha_vali*R[2][0]
                R(a[3][0]) = alpha_valr*R(a[3][0])-alpha_vali*I(a[3][0])
                I(a[3][0]) = alpha_valr*I(a[3][0])+alpha_vali*(R[3][0])
                */

                //Calculating using real part of complex number in B matrix
                FMA_M_LOOP(ymm0,ymm1,ymm3,ymm4,ymm2,(double const *)(temp_b));
                FMA_M_LOOP(ymm0,ymm1,ymm5,ymm6,ymm2,(double const *)(temp_b+ldb));
                FMA_M_LOOP(ymm0,ymm1,ymm7,ymm8,ymm2,(double const *)(temp_b+ldb*2));
                FMA_M_LOOP(ymm0,ymm1,ymm9,ymm10,ymm2,(double const *)(temp_b+ldb*3));

                //Calculating using imaginary part of complex numbers in B matrix
                //Shuffling ymm0 and ymm1 in accordance to the requirement
                NEG_PERM_M_LOOP(ymm0,ymm1,ymm2);
                FMA_M_LOOP(ymm0,ymm1,ymm3,ymm4,ymm2,(double const *)(temp_b)+1);
                FMA_M_LOOP(ymm0,ymm1,ymm5,ymm6,ymm2,(double const *)(temp_b+ldb)+1);
                FMA_M_LOOP(ymm0,ymm1,ymm7,ymm8,ymm2,(double const *)(temp_b+ldb*2)+1);
                FMA_M_LOOP(ymm0,ymm1,ymm9,ymm10,ymm2,(double const *)(temp_b+ldb*3)+1);
            }
            if(beta_valr != 0.0)
            {
                /*
                    a. Perform beta*C using temp_c, beta_valr,
                       where beta_valr is not zero.
                    b. This loop operates with 4x5 block size
                       along n dimension for every Z_NR columns of temp_c where
                       computing all Z_MR rows of temp_c.
                    c. Accumulated alpha*A*B into registers will be added to beta*C
                    d. Same approach is used in remaining fringe cases.
                */
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_valr));

                //R(c[0][0]) I(c[0][0]) R(c[1][0]) I(c[1][0])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c));
                //R(c[2][0]) I(c[2][0]) R(c[3][0]) I(c[3][0])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + 2));
                //ymm3+=beta_valr*R(c[0][0]) beta_valr*I(c[0][0])
                //      beta_valr*R(c[1][0]) beta_valr*I(c[1][0])
                //ymm4+=beta_valr*R(c[2][0]) beta_valr*I(c[2][0])
                //      beta_valr*R(c[3][0]) beta_valr*I(c[3][0])
                SCALE_BETA_REAL_M_LOOP(ymm0,ymm1,ymm3,ymm4,ymm15);

                //R(c[0][1]) I(c[0][1]) R(c[1][1]) I(c[1][1])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc));
                //R(c[2][1]) I(c[2][1]) R(c[3][1]) I(c[3][1])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc + 2));
                //ymm5+=beta_valr*R(c[0][1]) beta_valr*I(c[0][1])
                //      beta_valr*R(c[1][1]) beta_valr*I(c[1][1])
                //ymm6+=beta_valr*R(c[2][1]) beta_valr*I(c[2][1])
                //      beta_valr*R(c[3][1]) beta_valr*I(c[3][1])
                SCALE_BETA_REAL_M_LOOP(ymm0,ymm1,ymm5,ymm6,ymm15);

                //R(c[0][2]) I(c[0][2]) R(c[1][2]) I(c[1][2])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*2));
                //R(c[2][2]) I(c[2][2]) R(c[3][2]) I(c[3][2])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc*2 + 2));
                //ymm7+=beta_valr*R(c[0][2]) beta_valr*I(c[0][2])
                //      beta_valr*R(c[1][2]) beta_valr*I(c[1][2])
                //ymm8+=beta_valr*R(c[2][2]) beta_valr*I(c[2][2])
                //      beta_valr*R(c[3][2]) beta_valr*I(c[3][2])
                SCALE_BETA_REAL_M_LOOP(ymm0,ymm1,ymm7,ymm8,ymm15);

                //R(c[0][3]) I(c[0][3]) R(c[1][3]) I(c[1][3])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*3));
                //R(c[2][3]) I(c[2][3]) R(c[3][3]) I(c[3][3])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc*3 + 2));
                //ymm9+=beta_valr*R(c[0][3]) beta_valr*I(c[0][3])
                //      beta_valr*R(c[1][3]) beta_valr*I(c[1][3])
                //ymm10+=beta_valr*R(c[2][3]) beta_valr*I(c[2][3])
                //      beta_valr*R(c[3][3]) beta_valr*I(c[3][3])
                SCALE_BETA_REAL_M_LOOP(ymm0,ymm1,ymm9,ymm10,ymm15);

            }
            if(beta_vali != 0.0)
            {
                /*
                    a. Perform beta*C using temp_c, beta_vali,
                       where beta_vali is not zero.
                    b. This loop operates with 4x5 block size
                       along n dimension for every Z_NR columns of temp_c where
                       computing all Z_MR rows of temp_c.
                    c. Accumulated alpha*A*B into registers will be added to beta*C
                    d. Same approach is used in remaining fringe cases.
                */

                ymm15 = _mm256_broadcast_sd((double const *)(&beta_vali));

                //R(c[0][0]) I(c[0][0]) R(c[1][0]) I(c[1][0])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c));
                //R(c[2][0]) I(c[2][0]) R(c[3][0]) I(c[3][0])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + 2));
                //ymm3+=beta_vali*(-I(c[0][0])) beta_vali*R(c[0][0])
                //      beta_vali*(-I(c[1][0])) beta_vali*R(c[1][0])
                //ymm4+=beta_vali*(-I(c[2][0])) beta_vali*R(c[2][0])
                //      beta_vali*(-I(c[3][0])) beta_vali*R(c[3][0])
                SCALE_BETA_IMAG_M_LOOP(ymm0,ymm1,ymm3,ymm4,ymm15,ymm2);

                //R(c[0][1]) I(c[0][1]) R(c[1][1]) I(c[1][1])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc));
                //R(c[2][1]) I(c[2][1]) R(c[3][1]) I(c[3][1])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc + 2));
                //ymm5+=beta_vali*(-I(c[0][1])) beta_vali*R(c[0][1])
                //      beta_vali*(-I(c[1][1])) beta_vali*R(c[1][1])
                //ymm6+=beta_vali*(-I(c[2][1])) beta_vali*R(c[2][1])
                //      beta_vali*(-I(c[3][1])) beta_vali*R(c[3][1])
                SCALE_BETA_IMAG_M_LOOP(ymm0,ymm1,ymm5,ymm6,ymm15,ymm2);

                //R(c[0][2]) I(c[0][2]) R(c[1][2]) I(c[1][2])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*2));
                //R(c[2][2]) I(c[2][2]) R(c[3][2]) I(c[3][2])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc*2 + 2));
                //ymm7+=beta_vali*(-I(c[0][2])) beta_vali*R(c[0][2])
                //      beta_vali*(-I(c[1][2])) beta_vali*R(c[1][2])
                //ymm8+=beta_vali*(-I(c[2][2])) beta_vali*R(c[2][2])
                //      beta_vali*(-I(c[3][2])) beta_vali*R(c[3][2])
                SCALE_BETA_IMAG_M_LOOP(ymm0,ymm1,ymm7,ymm8,ymm15,ymm2);

                //R(c[0][3]) I(c[0][3]) R(c[1][3]) I(c[1][3])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*3));
                //R(c[2][3]) I(c[2][3]) R(c[3][3]) I(c[3][3])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc*3 + 2));
                //ymm9+=beta_vali*(-I(c[0][3])) beta_vali*R(c[0][3])
                //      beta_vali*(-I(c[1][3])) beta_vali*R(c[1][3])
                //ymm10+=beta_vali*(-I(c[2][3])) beta_vali*R(c[2][3])
                //      beta_vali*(-I(c[3][3])) beta_vali*R(c[3][3])
                SCALE_BETA_IMAG_M_LOOP(ymm0,ymm1,ymm9,ymm10,ymm15,ymm2);
            }
            /*
            The scaling has been done sequentially as follows:
            - If alpha_valr is not 0, it is used for scaling A
            - If alpha_vali is not 0, it is used for scaling A using permutation
              and selective negation, after loading
            - If beta_valr is not 0, is is used for scaling C
            - If beta_vali is not 0, it is used for scaling C using permutation
              and selective negation, after loading

            The results are accumalated in accordance to the non zero scalar values,
            and similar approach is followed in fringe cases
            */

            _mm256_storeu_pd((double *)(temp_c), ymm3);
            _mm256_storeu_pd((double *)(temp_c + 2), ymm4);

            _mm256_storeu_pd((double *)(temp_c + ldc), ymm5);
            _mm256_storeu_pd((double *)(temp_c + ldc + 2), ymm6);

            _mm256_storeu_pd((double *)(temp_c + ldc*2), ymm7);
            _mm256_storeu_pd((double *)(temp_c + ldc*2 + 2), ymm8);

            _mm256_storeu_pd((double *)(temp_c + ldc*3), ymm9);
            _mm256_storeu_pd((double *)(temp_c + ldc*3 + 2), ymm10);

            temp_c+=Z_MR;
            temp_a+=Z_MR;
        }

        dim_t m_rem=m_remainder;
        if(m_rem>=2)
        {
            ymm3 = _mm256_setzero_pd();
            ymm5 = _mm256_setzero_pd();
            ymm7 = _mm256_setzero_pd();
            ymm9 = _mm256_setzero_pd();

            if(alpha_valr != 0.0 || alpha_vali != 0.0)
            {

                //R(a[0][0]) I(a[0][0]) R(a[1][0]) I(a[1][0])
                ymm0 = _mm256_loadu_pd((double const *)(temp_a));
                ymm13 = ymm0;
                SCALE_ALPHA_REAL_M_FRINGE(ymm0,ymm15,alpha_valr);
                SCALE_ALPHA_IMAG_M_FRINGE(ymm0,ymm15,ymm2,ymm13,alpha_vali);
                /*
                The result after scaling with alpha_valr and/or alpha_vali is as follows:
                For ymm0 :
                R(a[0][0]) = alpha_valr*R(a[0][0])-alpha_vali*I(a[0][0])
                I(a[0][0]) = alpha_valr*I(a[0][0])+alpha_vali*R[0][0]
                R(a[1][0]) = alpha_valr*R(a[1][0])-alpha_vali*I(a[1][0])
                I(a[1][0]) = alpha_valr*I(a[1][0])+alpha_vali*(R[1][0])
                */

                //Calculating using real part of complex number in B matrix
                //ymm3+=R(b[0][0])*R(a[0][0]) R(b[0][0])*I(a[0][0])
                //      R(b[0][0])*R(a[1][0]) R(b[0][0])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm3,ymm2,(double const *)(temp_b));
                //ymm5+=R(b[0][1])*R(a[0][0]) R(b[0][1])*I(a[0][0])
                //      R(b[0][1])*R(a[1][0]) R(b[0][1])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm5,ymm2,(double const *)(temp_b+ldb));
                //ymm7+=R(b[0][2])*R(a[0][0]) R(b[0][2])*I(a[0][0])
                //      R(b[0][2])*R(a[1][0]) R(b[0][2])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm7,ymm2,(double const *)(temp_b+ldb*2));
                //ymm9+=R(b[0][3])*R(a[0][0]) R(b[0][3])*I(a[0][0])
                //      R(b[0][3])*R(a[1][0]) R(b[0][3])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm9,ymm2,(double const *)(temp_b+ldb*3));

                //Calculating using imaginary part of complex numbers in B matrix
                //Shuffling ymm0 in accordance to the requirement
                NEG_PERM_M_FRINGE(ymm0,ymm2);

                // ymm3+=I(b[0][0])*R(a[0][0]) I(b[0][0])*I(a[0][0])
                //      I(b[0][0])*R(a[1][0]) I(b[0][0])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm3,ymm2,(double const *)(temp_b)+1);
                //ymm5+=I(b[0][1])*R(a[0][0]) I(b[0][1])*I(a[0][0])
                //      I(b[0][1])*R(a[1][0]) I(b[0][1])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm5,ymm2,(double const *)(temp_b+ldb)+1);
                //ymm7+=I(b[0][2])*R(a[0][0]) I(b[0][2])*I(a[0][0])
                //      I(b[0][2])*R(a[1][0]) I(b[0][2])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm7,ymm2,(double const *)(temp_b+ldb*2)+1);
                //ymm9+=I(b[0][3])*R(a[0][0]) I(b[0][3])*I(a[0][0])
                //      I(b[0][3])*R(a[1][0]) I(b[0][3])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm9,ymm2,(double const *)(temp_b+ldb*3)+1);

            }

            if(beta_valr != 0.0)
            {

                ymm15 = _mm256_broadcast_sd((double const *)(&beta_valr));

                //R(c[0][0]) I(c[0][0]) R(c[1][0]) I(c[1][0])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c));
                //ymm3+=beta_valr*R(c[0][0]) beta_valr*I(c[0][0])
                //      beta_valr*R(c[1][0]) beta_valr*I(c[1][0])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm3,ymm15);

                //R(c[0][1]) I(c[0][1]) R(c[1][1]) I(c[1][1])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc));
                //ymm5+=beta_valr*R(c[0][1]) beta_valr*I(c[0][1])
                //      beta_valr*R(c[1][1]) beta_valr*I(c[1][1])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm5,ymm15);

                //R(c[0][2]) I(c[0][2]) R(c[1][2]) I(c[1][2])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*2));
                //ymm7+=beta_valr*R(c[0][2]) beta_valr*I(c[0][2])
                //      beta_valr*R(c[1][2]) beta_valr*I(c[1][2])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm7,ymm15);

                //R(c[0][3]) I(c[0][3]) R(c[1][3]) I(c[1][3])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*3));
                //ymm9+=beta_valr*R(c[0][3]) beta_valr*I(c[0][3])
                //      beta_valr*R(c[1][3]) beta_valr*I(c[1][3])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm9,ymm15);

            }

            if(beta_vali != 0.0)
            {

                ymm15 = _mm256_broadcast_sd((double const *)(&beta_vali));

                //R(c[0][0]) I(c[0][0]) R(c[1][0]) I(c[1][0])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c));
                //ymm3+=beta_vali*(-I(c[0][0])) beta_vali*R(c[0][0])
                //      beta_vali*(-I(c[1][0])) beta_vali*R(c[1][0])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm3,ymm15,ymm2);

                //R(c[0][1]) I(c[0][1]) R(c[1][1]) I(c[1][1])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc));
                //ymm5+=beta_vali*(-I(c[0][1])) beta_vali*R(c[0][1])
                //      beta_vali*(-I(c[1][1])) beta_vali*R(c[1][1])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm5,ymm15,ymm2);

                //R(c[0][2]) I(c[0][2]) R(c[1][2]) I(c[1][2])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*2));
                //ymm7+=beta_vali*(-I(c[0][2])) beta_vali*R(c[0][2])
                //      beta_vali*(-I(c[1][2])) beta_vali*R(c[1][2])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm7,ymm15,ymm2);

                //R(c[0][3]) I(c[0][3]) R(c[1][3]) I(c[1][3])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*3));
                //ymm9+=beta_vali*(-I(c[0][3])) beta_vali*R(c[0][3])
                //      beta_vali*(-I(c[1][3])) beta_vali*R(c[1][3])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm9,ymm15,ymm2);
            }

            /*
            The scaling has been done sequentially as follows:
            - If alpha_valr is not 0, it is used for scaling A
            - If alpha_vali is not 0, it is used for scaling A using permutation
              and selective negation, after loading
            - If beta_valr is not 0, is is used for scaling C
            - If beta_vali is not 0, it is used for scaling C using permutation
              and selective negation, after loading

            The results are accumalated in accordance to the non zero scalar values,
            and similar approach is followed in fringe cases
            */

            _mm256_storeu_pd((double *)(temp_c), ymm3);
            _mm256_storeu_pd((double *)(temp_c + ldc), ymm5);
            _mm256_storeu_pd((double *)(temp_c + ldc*2), ymm7);
            _mm256_storeu_pd((double *)(temp_c + ldc*3), ymm9);

            temp_c+=2;
            temp_a+=2;

            m_rem -= 2;
        }

        if(m_rem==1)
        {

            xmm5 = _mm_setzero_pd();
            ymm3 = _mm256_setzero_pd();
            ymm5 = _mm256_setzero_pd();
            ymm7 = _mm256_setzero_pd();
            ymm9 = _mm256_setzero_pd();

            if(alpha_valr != 0.0 || alpha_vali != 0.0)
            {
                xmm5 = _mm_loadu_pd((double const*)(temp_a));//R(a[0][0]) I(a[0][0])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(a[0][0]) I(a[0][0])
                ymm13 = ymm0;

                SCALE_ALPHA_REAL_M_FRINGE(ymm0,ymm15,alpha_valr);
                SCALE_ALPHA_IMAG_M_FRINGE(ymm0,ymm15,ymm2,ymm13,alpha_vali);

                //Calculating using real part of complex number in B matrix
                //ymm3+=R(b[0][0])*R(a[0][0]) R(b[0][0])*I(a[0][0])
                //      R(b[0][0])*R(a[1][0]) R(b[0][0])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm3,ymm2,(double const *)(temp_b));
                //ymm5+=R(b[0][1])*R(a[0][0]) R(b[0][1])*I(a[0][0])
                //      R(b[0][1])*R(a[1][0]) R(b[0][1])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm5,ymm2,(double const *)(temp_b+ldb));
                //ymm7+=R(b[0][2])*R(a[0][0]) R(b[0][2])*I(a[0][0])
                //      R(b[0][2])*R(a[1][0]) R(b[0][2])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm7,ymm2,(double const *)(temp_b+ldb*2));
                //ymm9+=R(b[0][3])*R(a[0][0]) R(b[0][3])*I(a[0][0])
                //      R(b[0][3])*R(a[1][0]) R(b[0][3])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm9,ymm2,(double const *)(temp_b+ldb*3));

                //Calculating using imaginary part of complex numbers in B matrix
                //Shuffling ymm0 in accordance to the requirement
                NEG_PERM_M_FRINGE(ymm0,ymm2);

                // ymm3+=I(b[0][0])*R(a[0][0]) I(b[0][0])*I(a[0][0])
                //       I(b[0][0])*R(a[1][0]) I(b[0][0])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm3,ymm2,(double const *)(temp_b)+1);
                //ymm5+=I(b[0][1])*R(a[0][0]) I(b[0][1])*I(a[0][0])
                //      I(b[0][1])*R(a[1][0]) I(b[0][1])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm5,ymm2,(double const *)(temp_b+ldb)+1);
                //ymm7+=I(b[0][2])*R(a[0][0]) I(b[0][2])*I(a[0][0])
                //      I(b[0][2])*R(a[1][0]) I(b[0][2])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm7,ymm2,(double const *)(temp_b+ldb*2)+1);
                //ymm9+=I(b[0][3])*R(a[0][0]) I(b[0][3])*I(a[0][0])
                //      I(b[0][3])*R(a[1][0]) I(b[0][3])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm9,ymm2,(double const *)(temp_b+ldb*3)+1);

            }
            if(beta_valr != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_valr));

                xmm5 = _mm_loadu_pd((double const*)(temp_c));//R(c[0][0]) I(c[0][0])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][0]) I(c[0][0])
                //ymm3+=beta_valr*R(c[0][0]) beta_valr*I(c[0][0])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm3,ymm15);

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc));//R(c[0][1]) I(c[0][1])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][1]) I(c[0][1])
                //ymm5+=beta_valr*R(c[0][1]) beta_valr*I(c[0][1])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm5,ymm15);

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc * 2));//R(c[0][2]) I(c[0][2])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][2]) I(c[0][2])
                //ymm7+=beta_valr*R(c[0][2]) beta_valr*I(c[0][2])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm7,ymm15);

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc * 3));//R(c[0][3]) I(c[0][3])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][3]) I(c[0][3])
                //ymm9+=beta_valr*R(c[0][3]) beta_valr*I(c[0][3])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm9,ymm15);
            }
            if(beta_vali != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_vali));

                xmm5 = _mm_loadu_pd((double const*)(temp_c));//R(c[0][0]) I(c[0][0])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][0]) I(c[0][0])
                //ymm3+=beta_vali*(-I(c[0][0])) beta_vali*R(c[0][0])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm3,ymm15,ymm2);

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc));//R(c[0][1]) I(c[0][1])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][1]) I(c[0][1])
                //ymm5+=beta_vali*(-I(c[0][1])) beta_vali*R(c[0][1])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm5,ymm15,ymm2);

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc * 2));//R(c[0][2]) I(c[0][2])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][2]) I(c[0][2])
                //ymm7+=beta_vali*(-I(c[0][2])) beta_vali*R(c[0][2])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm7,ymm15,ymm2);

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc * 3));//R(c[0][3]) I(c[0][3])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][3]) I(c[0][3])
                //ymm9+=beta_vali*(-I(c[0][3])) beta_vali*R(c[0][3])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm9,ymm15,ymm2);

            }

            xmm5 = _mm256_extractf128_pd(ymm3, 0);
            _mm_storeu_pd((double *)(temp_c), xmm5);

            xmm5 = _mm256_extractf128_pd(ymm5, 0);
            _mm_storeu_pd((double *)(temp_c + ldc), xmm5);

            xmm5 = _mm256_extractf128_pd(ymm7, 0);
            _mm_storeu_pd((double *)(temp_c + ldc*2), xmm5);

            xmm5 = _mm256_extractf128_pd(ymm9, 0);
            _mm_storeu_pd((double *)(temp_c + ldc*3), xmm5);

        }
        n_remainder -= 4;

    }
    if(n_remainder>=2)
    {
        dcomplex* temp_b = b + (n - n_remainder)*ldb;
        dcomplex* temp_a = a;
        dcomplex* temp_c = c + (n - n_remainder)*ldc;
        for(dim_t i = 0;i < (m-Z_MR+1);i=i+Z_MR)
        {
            ymm3 = _mm256_setzero_pd();
            ymm4 = _mm256_setzero_pd();
            ymm5 = _mm256_setzero_pd();
            ymm6 = _mm256_setzero_pd();

            if(alpha_valr != 0.0 || alpha_vali != 0.0)
            {
                /*
                    a. Perform alpha*A*B using temp_a, temp_b and alpha_valr, alpha_vali
                       where alpha_valr and/or alpha_vali is not zero.
                    b. This loop operates with 4x5 block size
                       along n dimension for every Z_NR columns of temp_b where
                       computing all Z_MR rows of temp_a.
                    c. Same approach is used in remaining fringe cases.
                */

                //R(a[0][0]) I(a[0][0]) R(a[1][0]) I(a[1][0])
                ymm0 = _mm256_loadu_pd((double const *)(temp_a));
                //R(a[2][0]) I(a[2][0]) R(a[3][0]) I(a[3][0])
                ymm1 = _mm256_loadu_pd((double const *)(temp_a + 2));

                ymm13 = ymm0;
                ymm14 = ymm1;

                _mm_prefetch((char*)(temp_a) + 64, _MM_HINT_T0);

                SCALE_ALPHA_REAL_M_LOOP(ymm0,ymm1,ymm15,alpha_valr);
                SCALE_ALPHA_IMAG_M_LOOP(ymm0,ymm1,ymm15,ymm2,ymm13,ymm14,alpha_vali);

                /*
                The result after scaling with alpha_valr and/or alpha_vali is as follows:
                For ymm0 :
                R(a[0][0]) = alpha_valr*R(a[0][0])-alpha_vali*I(a[0][0])
                I(a[0][0]) = alpha_valr*I(a[0][0])+alpha_vali*R[0][0]
                R(a[1][0]) = alpha_valr*R(a[1][0])-alpha_vali*I(a[1][0])
                I(a[1][0]) = alpha_valr*I(a[1][0])+alpha_vali*(R[1][0])

                For ymm1 :
                R(a[2][0]) = alpha_valr*R(a[2][0])-alpha_vali*I(a[2][0])
                I(a[2][0]) = alpha_valr*I(a[2][0])+alpha_vali*R[2][0]
                R(a[3][0]) = alpha_valr*R(a[3][0])-alpha_vali*I(a[3][0])
                I(a[3][0]) = alpha_valr*I(a[3][0])+alpha_vali*(R[3][0])
                */

                //Calculating using real part of complex number in B matrix
                FMA_M_LOOP(ymm0,ymm1,ymm3,ymm4,ymm2,(double const *)(temp_b));
                FMA_M_LOOP(ymm0,ymm1,ymm5,ymm6,ymm2,(double const *)(temp_b+ldb));

                //Calculating using imaginary part of complex numbers in B matrix
                //Shuffling ymm0 and ymm1 in accordance to the requirement
                NEG_PERM_M_LOOP(ymm0,ymm1,ymm2);
                FMA_M_LOOP(ymm0,ymm1,ymm3,ymm4,ymm2,(double const *)(temp_b)+1);
                FMA_M_LOOP(ymm0,ymm1,ymm5,ymm6,ymm2,(double const *)(temp_b+ldb)+1);
            }
            if(beta_valr != 0.0)
            {
                /*
                    a. Perform beta*C using temp_c, beta_valr,
                       where beta_valr is not zero.
                    b. This loop operates with 4x5 block size
                       along n dimension for every Z_NR columns of temp_c where
                       computing all Z_MR rows of temp_c.
                    c. Accumulated alpha*A*B into registers will be added to beta*C
                    d. Same approach is used in remaining fringe cases.
                */
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_valr));

                //R(c[0][0]) I(c[0][0]) R(c[1][0]) I(c[1][0])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c));
                //R(c[2][0]) I(c[2][0]) R(c[3][0]) I(c[3][0])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + 2));
                //ymm3+=beta_valr*R(c[0][0]) beta_valr*I(c[0][0])
                //      beta_valr*R(c[1][0]) beta_valr*I(c[1][0])
                //ymm4+=beta_valr*R(c[2][0]) beta_valr*I(c[2][0])
                //      beta_valr*R(c[3][0]) beta_valr*I(c[3][0])
                SCALE_BETA_REAL_M_LOOP(ymm0,ymm1,ymm3,ymm4,ymm15);

                //R(c[0][1]) I(c[0][1]) R(c[1][1]) I(c[1][1])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc));
                //R(c[2][1]) I(c[2][1]) R(c[3][1]) I(c[3][1])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc + 2));
                //ymm5+=beta_valr*R(c[0][1]) beta_valr*I(c[0][1])
                //      beta_valr*R(c[1][1]) beta_valr*I(c[1][1])
                //ymm6+=beta_valr*R(c[2][1]) beta_valr*I(c[2][1])
                //      beta_valr*R(c[3][1]) beta_valr*I(c[3][1])
                SCALE_BETA_REAL_M_LOOP(ymm0,ymm1,ymm5,ymm6,ymm15);

            }
            if(beta_vali != 0.0)
            {
                /*
                    a. Perform beta*C using temp_c, beta_vali,
                       where beta_vali is not zero.
                    b. This loop operates with 4x5 block size
                       along n dimension for every Z_NR columns of temp_c where
                       computing all Z_MR rows of temp_c.
                    c. Accumulated alpha*A*B into registers will be added to beta*C
                    d. Same approach is used in remaining fringe cases.
                */

                ymm15 = _mm256_broadcast_sd((double const *)(&beta_vali));

                //R(c[0][0]) I(c[0][0]) R(c[1][0]) I(c[1][0])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c));
                //R(c[2][0]) I(c[2][0]) R(c[3][0]) I(c[3][0])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + 2));
                //ymm3+=beta_vali*(-I(c[0][0])) beta_vali*R(c[0][0])
                //      beta_vali*(-I(c[1][0])) beta_vali*R(c[1][0])
                //ymm4+=beta_vali*(-I(c[2][0])) beta_vali*R(c[2][0])
                //      beta_vali*(-I(c[3][0])) beta_vali*R(c[3][0])
                SCALE_BETA_IMAG_M_LOOP(ymm0,ymm1,ymm3,ymm4,ymm15,ymm2);

                //R(c[0][1]) I(c[0][1]) R(c[1][1]) I(c[1][1])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc));
                //R(c[2][1]) I(c[2][1]) R(c[3][1]) I(c[3][1])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc + 2));
                //ymm5+=beta_vali*(-I(c[0][1])) beta_vali*R(c[0][1])
                //      beta_vali*(-I(c[1][1])) beta_vali*R(c[1][1])
                //ymm6+=beta_vali*(-I(c[2][1])) beta_vali*R(c[2][1])
                //      beta_vali*(-I(c[3][1])) beta_vali*R(c[3][1])
                SCALE_BETA_IMAG_M_LOOP(ymm0,ymm1,ymm5,ymm6,ymm15,ymm2);
            }
            /*
            The scaling has been done sequentially as follows:
            - If alpha_valr is not 0, it is used for scaling A
            - If alpha_vali is not 0, it is used for scaling A using permutation
              and selective negation, after loading
            - If beta_valr is not 0, is is used for scaling C
            - If beta_vali is not 0, it is used for scaling C using permutation
              and selective negation, after loading

            The results are accumalated in accordance to the non zero scalar values,
            and similar approach is followed in fringe cases
            */

            _mm256_storeu_pd((double *)(temp_c), ymm3);
            _mm256_storeu_pd((double *)(temp_c + 2), ymm4);

            _mm256_storeu_pd((double *)(temp_c + ldc), ymm5);
            _mm256_storeu_pd((double *)(temp_c + ldc + 2), ymm6);

            temp_c+=Z_MR;
            temp_a+=Z_MR;
        }

        dim_t m_rem=m_remainder;
        if(m_rem>=2)
        {
            ymm3 = _mm256_setzero_pd();
            ymm5 = _mm256_setzero_pd();

            if(alpha_valr != 0.0 || alpha_vali != 0.0)
            {

                //R(a[0][0]) I(a[0][0]) R(a[1][0]) I(a[1][0])
                ymm0 = _mm256_loadu_pd((double const *)(temp_a));
                ymm13 = ymm0;
                SCALE_ALPHA_REAL_M_FRINGE(ymm0,ymm15,alpha_valr);
                SCALE_ALPHA_IMAG_M_FRINGE(ymm0,ymm15,ymm2,ymm13,alpha_vali);
                /*
                The result after scaling with alpha_valr and/or alpha_vali is as follows:
                For ymm0 :
                R(a[0][0]) = alpha_valr*R(a[0][0])-alpha_vali*I(a[0][0])
                I(a[0][0]) = alpha_valr*I(a[0][0])+alpha_vali*R[0][0]
                R(a[1][0]) = alpha_valr*R(a[1][0])-alpha_vali*I(a[1][0])
                I(a[1][0]) = alpha_valr*I(a[1][0])+alpha_vali*(R[1][0])
                */

                //Calculating using real part of complex number in B matrix
                //ymm3+=R(b[0][0])*R(a[0][0]) R(b[0][0])*I(a[0][0])
                //      R(b[0][0])*R(a[1][0]) R(b[0][0])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm3,ymm2,(double const *)(temp_b));
                //ymm5+=R(b[0][1])*R(a[0][0]) R(b[0][1])*I(a[0][0])
                //      R(b[0][1])*R(a[1][0]) R(b[0][1])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm5,ymm2,(double const *)(temp_b+ldb));

                //Calculating using imaginary part of complex numbers in B matrix
                //Shuffling ymm0 in accordance to the requirement
                NEG_PERM_M_FRINGE(ymm0,ymm2);

                // ymm3+=I(b[0][0])*R(a[0][0]) I(b[0][0])*I(a[0][0])
                //      I(b[0][0])*R(a[1][0]) I(b[0][0])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm3,ymm2,(double const *)(temp_b)+1);
                //ymm5+=I(b[0][1])*R(a[0][0]) I(b[0][1])*I(a[0][0])
                //      I(b[0][1])*R(a[1][0]) I(b[0][1])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm5,ymm2,(double const *)(temp_b+ldb)+1);

            }

            if(beta_valr != 0.0)
            {

                ymm15 = _mm256_broadcast_sd((double const *)(&beta_valr));

                //R(c[0][0]) I(c[0][0]) R(c[1][0]) I(c[1][0])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c));
                //ymm3+=beta_valr*R(c[0][0]) beta_valr*I(c[0][0])
                //      beta_valr*R(c[1][0]) beta_valr*I(c[1][0])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm3,ymm15);

                //R(c[0][1]) I(c[0][1]) R(c[1][1]) I(c[1][1])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc));
                //ymm5+=beta_valr*R(c[0][1]) beta_valr*I(c[0][1])
                //      beta_valr*R(c[1][1]) beta_valr*I(c[1][1])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm5,ymm15);

            }

            if(beta_vali != 0.0)
            {

                ymm15 = _mm256_broadcast_sd((double const *)(&beta_vali));

                //R(c[0][0]) I(c[0][0]) R(c[1][0]) I(c[1][0])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c));
                //ymm3+=beta_vali*(-I(c[0][0])) beta_vali*R(c[0][0])
                //      beta_vali*(-I(c[1][0])) beta_vali*R(c[1][0])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm3,ymm15,ymm2);

                //R(c[0][1]) I(c[0][1]) R(c[1][1]) I(c[1][1])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc));
                //ymm5+=beta_vali*(-I(c[0][1])) beta_vali*R(c[0][1])
                //      beta_vali*(-I(c[1][1])) beta_vali*R(c[1][1])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm5,ymm15,ymm2);
            }

            /*
            The scaling has been done sequentially as follows:
            - If alpha_valr is not 0, it is used for scaling A
            - If alpha_vali is not 0, it is used for scaling A using permutation
              and selective negation, after loading
            - If beta_valr is not 0, is is used for scaling C
            - If beta_vali is not 0, it is used for scaling C using permutation
              and selective negation, after loading

            The results are accumalated in accordance to the non zero scalar values,
            and similar approach is followed in fringe cases
            */

            _mm256_storeu_pd((double *)(temp_c), ymm3);
            _mm256_storeu_pd((double *)(temp_c + ldc), ymm5);

            temp_c+=2;
            temp_a+=2;

            m_rem -= 2;
        }

        if(m_rem==1)
        {

            xmm5 = _mm_setzero_pd();
            ymm3 = _mm256_setzero_pd();
            ymm5 = _mm256_setzero_pd();

            if(alpha_valr != 0.0 || alpha_vali != 0.0)
            {
                xmm5 = _mm_loadu_pd((double const*)(temp_a));//R(a[0][0]) I(a[0][0])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(a[0][0]) I(a[0][0])
                ymm13 = ymm0;

                SCALE_ALPHA_REAL_M_FRINGE(ymm0,ymm15,alpha_valr);
                SCALE_ALPHA_IMAG_M_FRINGE(ymm0,ymm15,ymm2,ymm13,alpha_vali);

                //Calculating using real part of complex number in B matrix
                //ymm3+=R(b[0][0])*R(a[0][0]) R(b[0][0])*I(a[0][0])
                //      R(b[0][0])*R(a[1][0]) R(b[0][0])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm3,ymm2,(double const *)(temp_b));
                //ymm5+=R(b[0][1])*R(a[0][0]) R(b[0][1])*I(a[0][0])
                //      R(b[0][1])*R(a[1][0]) R(b[0][1])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm5,ymm2,(double const *)(temp_b+ldb));

                //Calculating using imaginary part of complex numbers in B matrix
                //Shuffling ymm0 in accordance to the requirement
                NEG_PERM_M_FRINGE(ymm0,ymm2);

                // ymm3+=I(b[0][0])*R(a[0][0]) I(b[0][0])*I(a[0][0])
                //       I(b[0][0])*R(a[1][0]) I(b[0][0])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm3,ymm2,(double const *)(temp_b)+1);
                //ymm5+=I(b[0][1])*R(a[0][0]) I(b[0][1])*I(a[0][0])
                //      I(b[0][1])*R(a[1][0]) I(b[0][1])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm5,ymm2,(double const *)(temp_b+ldb)+1);

            }
            if(beta_valr != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_valr));

                xmm5 = _mm_loadu_pd((double const*)(temp_c));//R(c[0][0]) I(c[0][0])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][0]) I(c[0][0])
                //ymm3+=beta_valr*R(c[0][0]) beta_valr*I(c[0][0])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm3,ymm15);

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc));//R(c[0][1]) I(c[0][1])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][1]) I(c[0][1])
                //ymm5+=beta_valr*R(c[0][1]) beta_valr*I(c[0][1])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm5,ymm15);
            }
            if(beta_vali != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_vali));

                xmm5 = _mm_loadu_pd((double const*)(temp_c));//R(c[0][0]) I(c[0][0])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][0]) I(c[0][0])
                //ymm3+=beta_vali*(-I(c[0][0])) beta_vali*R(c[0][0])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm3,ymm15,ymm2);

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc));//R(c[0][1]) I(c[0][1])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][1]) I(c[0][1])
                //ymm5+=beta_vali*(-I(c[0][1])) beta_vali*R(c[0][1])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm5,ymm15,ymm2);

            }

            xmm5 = _mm256_extractf128_pd(ymm3, 0);
            _mm_storeu_pd((double *)(temp_c), xmm5);

            xmm5 = _mm256_extractf128_pd(ymm5, 0);
            _mm_storeu_pd((double *)(temp_c + ldc), xmm5);

        }
        n_remainder -= 2;
    }
    if(n_remainder==1)
    {
        dcomplex* temp_b = b + (n - n_remainder)*ldb;
        dcomplex* temp_a = a;
        dcomplex* temp_c = c + (n - n_remainder)*ldc;

        for(dim_t i = 0;i < (m-Z_MR+1);i=i+Z_MR)
        {
            ymm3 = _mm256_setzero_pd();
            ymm4 = _mm256_setzero_pd();

            if(alpha_valr != 0.0 || alpha_vali != 0.0)
            {
                /*
                    a. Perform alpha*A*B using temp_a, temp_b and alpha_valr, aplha_vali
                       where alpha_valr and/or alpha_vali is not zero.
                    b. This loop operates with 4x5 block size
                       along n dimension for every Z_NR columns of temp_b where
                       computing all Z_MR rows of temp_a.
                    c. Same approach is used in remaining fringe cases.
                */

                //R(a[0][0]) I(a[0][0]) R(a[1][0]) I(a[1][0])
                ymm0 = _mm256_loadu_pd((double const *)(temp_a));
                //R(a[2][0]) I(a[2][0]) R(a[3][0]) I(a[3][0])
                ymm1 = _mm256_loadu_pd((double const *)(temp_a + 2));

                ymm13 = ymm0;
                ymm14 = ymm1;
                _mm_prefetch((char*)(temp_a) + 64, _MM_HINT_T0);

                SCALE_ALPHA_REAL_M_LOOP(ymm0,ymm1,ymm15,alpha_valr);
                SCALE_ALPHA_IMAG_M_LOOP(ymm0,ymm1,ymm15,ymm2,ymm13,ymm14,alpha_vali);

                /*
                The result after scaling with alpha_valr and/or alpha_vali is as follows:
                For ymm0 :
                R(a[0][0]) = alpha_valr*R(a[0][0])-alpha_vali*I(a[0][0])
                I(a[0][0]) = alpha_valr*I(a[0][0])+alpha_vali*R[0][0]
                R(a[1][0]) = alpha_valr*R(a[1][0])-alpha_vali*I(a[1][0])
                I(a[1][0]) = alpha_valr*I(a[1][0])+alpha_vali*(R[1][0])

                For ymm1 :
                R(a[2][0]) = alpha_valr*R(a[2][0])-alpha_vali*I(a[2][0])
                I(a[2][0]) = alpha_valr*I(a[2][0])+alpha_vali*R[2][0]
                R(a[3][0]) = alpha_valr*R(a[3][0])-alpha_vali*I(a[3][0])
                I(a[3][0]) = alpha_valr*I(a[3][0])+alpha_vali*(R[3][0])
                */

                //Calculating using real part of complex number in B matrix
                FMA_M_LOOP(ymm0,ymm1,ymm3,ymm4,ymm2,(double const *)(temp_b));

                //Calculating using imaginary part of complex numbers in B matrix
                //Shuffling ymm0 and ymm1 in accordance to the requirement
                NEG_PERM_M_LOOP(ymm0,ymm1,ymm2);
                FMA_M_LOOP(ymm0,ymm1,ymm3,ymm4,ymm2,(double const *)(temp_b)+1);

            }
            if(beta_valr != 0.0)
            {
                /*
                    a. Perform beta*C using temp_c, beta_valr,
                       where beta_valr is not zero.
                    b. This loop operates with 4x5 block size
                       along n dimension for every Z_NR columns of temp_c where
                       computing all Z_MR rows of temp_c.
                    c. Accumulated alpha*A*B into registers will be added to beta*C
                    d. Same approach is used in remaining fringe cases.
                */
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_valr));

                //R(c[0][0]) I(c[0][0]) R(c[1][0]) I(c[1][0])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c));
                //R(c[2][0]) I(c[2][0]) R(c[3][0]) I(c[3][0])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + 2));
                    //ymm3+=beta_valr*R(c[0][0]) beta_valr*I(c[0][0])
                    //      beta_valr*R(c[1][0]) beta_valr*I(c[1][0])
                    //ymm4+=beta_valr*R(c[2][0]) beta_valr*I(c[2][0])
                    //      beta_valr*R(c[3][0]) beta_valr*I(c[3][0])
                SCALE_BETA_REAL_M_LOOP(ymm0,ymm1,ymm3,ymm4,ymm15);

            }
            if(beta_vali != 0.0)
            {
                /*
                    a. Perform beta*C using temp_c, beta_vali,
                       where beta_vali is not zero.
                    b. This loop operates with 4x5 block size
                       along n dimension for every Z_NR columns of temp_c where
                       computing all Z_MR rows of temp_c.
                    c. Accumulated alpha*A*B into registers will be added to beta*C
                    d. Same approach is used in remaining fringe cases.
                */

                ymm15 = _mm256_broadcast_sd((double const *)(&beta_vali));

                ymm0 = _mm256_loadu_pd((double const *)(temp_c));
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + 2));
                    //ymm3+=beta_vali*(-I(c[0][0])) beta_vali*R(c[0][0])
                    //      beta_vali*(-I(c[1][0])) beta_vali*R(c[1][0])
                    //ymm4+=beta_vali*(-I(c[2][0])) beta_vali*R(c[2][0])
                    //      beta_vali*(-I(c[3][0])) beta_vali*R(c[3][0])
                SCALE_BETA_IMAG_M_LOOP(ymm0,ymm1,ymm3,ymm4,ymm15,ymm2);
            }
            /*
            The scaling has been done sequentially as follows:
            - If alpha_valr is not 0, it is used for scaling A
            - If alpha_vali is not 0, it is used for scaling A using permutation
              and selective negation, after loading
            - If beta_valr is not 0, is is used for scaling C
            - If beta_vali is not 0, it is used for scaling C using permutation
              and selective negation, after loading

            The results are accumalated in accordance to the non zero scalar values,
            and similar approach is followed in fringe cases
            */

            //R(c[0][0]) I(c[0][0]) R(c[1][0]) I(c[1][0])
            _mm256_storeu_pd((double *)(temp_c), ymm3);
            //R(c[2][0]) I(c[2][0]) R(c[3][0]) I(c[3][0])
            _mm256_storeu_pd((double *)(temp_c + 2), ymm4);

            temp_c+=Z_MR;
            temp_a+=Z_MR;
        }

        dim_t m_rem=m_remainder;
        if(m_rem>=2)
        {
            ymm3 = _mm256_setzero_pd();

            if(alpha_valr != 0.0 || alpha_vali != 0.0)
            {

                //R(a[0][0]) I(a[0][0]) R(a[1][0]) I(a[1][0])
                ymm0 = _mm256_loadu_pd((double const *)(temp_a));
                ymm13 = ymm0;

                SCALE_ALPHA_REAL_M_FRINGE(ymm0,ymm15,alpha_valr);
                SCALE_ALPHA_IMAG_M_FRINGE(ymm0,ymm15,ymm2,ymm13,alpha_vali);

                /*
                The result after scaling with alpha_valr and/or alpha_vali is as follows:
                For ymm0 :
                R(a[0][0]) = alpha_valr*R(a[0][0])-alpha_vali*I(a[0][0])
                I(a[0][0]) = alpha_valr*I(a[0][0])+alpha_vali*R[0][0]
                R(a[1][0]) = alpha_valr*R(a[1][0])-alpha_vali*I(a[1][0])
                I(a[1][0]) = alpha_valr*I(a[1][0])+alpha_vali*(R[1][0])
                */

                //Calculating using real part of complex number in B matrix
                //ymm3+=R(b[0][0])*R(a[0][0]) R(b[0][0])*I(a[0][0])
                //      R(b[0][0])*R(a[1][0]) R(b[0][0])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm3,ymm2,(double const *)(temp_b));

                //Calculating using imaginary part of complex numbers in B matrix
                //Shuffling ymm0 in accordance to the requirement
                NEG_PERM_M_FRINGE(ymm0,ymm2);

                // ymm3+=I(b[0][0])*R(a[0][0]) I(b[0][0])*I(a[0][0])
                //      I(b[0][0])*R(a[1][0]) I(b[0][0])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm3,ymm2,(double const *)(temp_b)+1);
            }

            if(beta_valr != 0.0)
            {

                ymm15 = _mm256_broadcast_sd((double const *)(&beta_valr));

                ymm0 = _mm256_loadu_pd((double const *)(temp_c));
                //ymm3+=beta_valr*R(c[0][0]) beta_valr*I(c[0][0])
                //      beta_valr*R(c[1][0]) beta_valr*I(c[1][0])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm3,ymm15);
            }

            if(beta_vali != 0.0)
            {

                ymm15 = _mm256_broadcast_sd((double const *)(&beta_vali));

                ymm0 = _mm256_loadu_pd((double const *)(temp_c));
                //ymm3+=beta_vali*(-I(c[0][0])) beta_vali*R(c[0][0])
                //      beta_vali*(-I(c[1][0])) beta_vali*R(c[1][0])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm3,ymm15,ymm2);
            }

            /*
            The scaling has been done sequentially as follows:
            - If alpha_valr is not 0, it is used for scaling A
            - If alpha_vali is not 0, it is used for scaling A using permutation
              and selective negation, after loading
            - If beta_valr is not 0, is is used for scaling C
            - If beta_vali is not 0, it is used for scaling C using permutation
              and selective negation, after loading

            The results are accumalated in accordance to the non zero scalar values,
            and similar approach is followed in fringe cases
            */

            _mm256_storeu_pd((double *)(temp_c), ymm3);

            temp_c+=2;
            temp_a+=2;

            m_rem -= 2;
        }

        if(m_rem==1)
        {

            xmm5 = _mm_setzero_pd();
            ymm3 = _mm256_setzero_pd();

            if(alpha_valr != 0.0 || alpha_vali != 0.0)
            {
                xmm5 = _mm_loadu_pd((double const*)(temp_a));
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);
                ymm13 = ymm0;

                SCALE_ALPHA_REAL_M_FRINGE(ymm0,ymm15,alpha_valr);
                SCALE_ALPHA_IMAG_M_FRINGE(ymm0,ymm15,ymm2,ymm13,alpha_vali);

                //Calculating using real part of complex number in B matrix
                //ymm3+=R(b[0][0])*R(a[0][0]) R(b[0][0])*I(a[0][0])
                //      R(b[0][0])*R(a[1][0]) R(b[0][0])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm3,ymm2,(double const *)(temp_b));

                //Calculating using imaginary part of complex numbers in B matrix
                //Shuffling ymm0 in accordance to the requirement
                NEG_PERM_M_FRINGE(ymm0,ymm2);

                // ymm3+=I(b[0][0])*R(a[0][0]) I(b[0][0])*I(a[0][0])
                //      I(b[0][0])*R(a[1][0]) I(b[0][0])*I(a[1][0])
                FMA_M_FRINGE(ymm0,ymm3,ymm2,(double const *)(temp_b)+1);
            }
            if(beta_valr != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_valr));

                xmm5 = _mm_loadu_pd((double const*)(temp_c));
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);
                //ymm3+=beta_valr*R(c[0][0]) beta_valr*I(c[0][0])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm3,ymm15);
            }
            if(beta_vali != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_vali));

                xmm5 = _mm_loadu_pd((double const*)(temp_c));
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);
                //ymm3+=beta_vali*(-I(c[0][0])) beta_vali*R(c[0][0])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm3,ymm15,ymm2);

            }

            xmm5 = _mm256_extractf128_pd(ymm3, 0);
            _mm_storeu_pd((double *)(temp_c), xmm5);

        }

    }

}
