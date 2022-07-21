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
#define Z_NR  6

// Macros for the main loop for M
#define SCALE_ALPHA_REAL_M_LOOP(rin_0,rin_1,r_bcast,real_val)  \
    r_bcast = _mm256_broadcast_sd((double const *)(&real_val)); \
    rin_0 = _mm256_mul_pd(rin_0,r_bcast); \
    rin_1 = _mm256_mul_pd(rin_1,r_bcast); \

#define SCALE_ALPHA_IMAG_M_LOOP(rout_0,rout_1,rin_0,rin_1,r_bcast,r_perm,imag_val)  \
    r_perm = _mm256_permute4x64_pd(rin_0,0b10110001);  \
    r_bcast = _mm256_set_pd(1.0,-1.0,1.0,-1.0);  \
    r_perm = _mm256_mul_pd(r_bcast, r_perm); \
    r_bcast = _mm256_broadcast_sd((double const *)(&imag_val)); \
    rout_0 = _mm256_fmadd_pd(r_perm,r_bcast,rout_0); \
    r_perm = _mm256_permute4x64_pd(rin_1,0b10110001);  \
    r_bcast = _mm256_set_pd(1.0,-1.0,1.0,-1.0);  \
    r_perm = _mm256_mul_pd(r_bcast, r_perm); \
    r_bcast = _mm256_broadcast_sd((double const *)(&imag_val)); \
    rout_1 = _mm256_fmadd_pd(r_perm,r_bcast,rout_1); \

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
#define SCALE_ALPHA_REAL_M_FRINGE(rin_0,r_bcast,real_val)  \
    r_bcast = _mm256_broadcast_sd((double const *)(&real_val)); \
    rin_0 = _mm256_mul_pd(rin_0,r_bcast); \

#define SCALE_ALPHA_IMAG_M_FRINGE(rout_0,rin_0,r_bcast,r_perm,imag_val)  \
    r_perm = _mm256_permute4x64_pd(rin_0,0b10110001);  \
    r_bcast = _mm256_set_pd(1.0,-1.0,1.0,-1.0);  \
    r_perm = _mm256_mul_pd(r_bcast, r_perm); \
    r_bcast = _mm256_broadcast_sd((double const *)(&imag_val)); \
    rout_0 = _mm256_fmadd_pd(r_perm,r_bcast,rout_0); \

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

    double alpha_real, beta_real;
    double alpha_imag, beta_imag;

    alpha_real = alpha->real;
    beta_real = beta->real;
    alpha_imag = alpha->imag;
    beta_imag = beta->imag;

    /* If m or n is zero, return immediately. */
	if ( bli_zero_dim2( m, n ) ) return;
	/* If alpha alone is zero, scale by beta and return. */
	if (bli_zeq0(*(alpha)))
	{
       bli_zscalm(
            BLIS_NO_CONJUGATE,
            0,
            BLIS_NONUNIT_DIAG,
            BLIS_DENSE,
            m, n,
            beta,
            c, 1, ldc
        );
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
    // Main loop along N dimension
    for(dim_t j = 0;j < (n-Z_NR+1);j=j+Z_NR)
    {
        dcomplex* temp_b = b + j*ldb;
        dcomplex* temp_a = a;
        dcomplex* temp_c = c + j*ldc;

        //Main loop along M dimension
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

            /*
                a. Perform alpha*A*B using temp_a, temp_b and alpha_real, alpha_imag
                    where alpha_real and/or alpha_imag is not zero.
                b. This loop operates with 4x6 block size
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
            _mm_prefetch((char*)(temp_a + 32), _MM_HINT_T0);

            SCALE_ALPHA_REAL_M_LOOP(ymm0,ymm1,ymm15,alpha_real);
            SCALE_ALPHA_IMAG_M_LOOP(ymm0,ymm1,ymm13,ymm14,ymm15,ymm2,alpha_imag);

            ymm13 = _mm256_setzero_pd();
            ymm14 = _mm256_setzero_pd();

            /*
            The result after scaling with alpha_real and/or alpha_imag is as follows:
            For ymm0 :
            R(a[0][0]) = alpha_real*R(a[0][0])-alpha_imag*I(a[0][0])
            I(a[0][0]) = alpha_real*I(a[0][0])+alpha_imag*R[0][0]
            R(a[1][0]) = alpha_real*R(a[1][0])-alpha_imag*I(a[1][0])
            I(a[1][0]) = alpha_real*I(a[1][0])+alpha_imag*(R[1][0])

            For ymm1 :
            R(a[2][0]) = alpha_real*R(a[2][0])-alpha_imag*I(a[2][0])
            I(a[2][0]) = alpha_real*I(a[2][0])+alpha_imag*R[2][0]
            R(a[3][0]) = alpha_real*R(a[3][0])-alpha_imag*I(a[3][0])
            I(a[3][0]) = alpha_real*I(a[3][0])+alpha_imag*(R[3][0])
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
            //ymm11+=R(b[0][5])*R(a[0][0]) R(b[0][5])*I(a[0][0])
            //      R(b[0][5])*R(a[1][0]) R(b[0][5])*I(a[1][0])
            //ymm12+=R(b[0][5])*R(a[0][0]) R(b[0][5])*I(a[0][0])
            //      R(b[0][5])*R(a[1][0]) R(b[0][5])*I(a[1][0])
            FMA_M_LOOP(ymm0,ymm1,ymm13,ymm14,ymm2,(double const *)(temp_b+ldb*5));

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
            //ymm13+=I(b[0][5])*R(a[0][0]) I(b[0][5])*I(a[0][0])
            //      I(b[0][5])*R(a[1][0]) I(b[0][5])*I(a[1][0])
            //ymm14+=I(b[0][5])*R(a[0][0]) I(b[0][5])*I(a[0][0])
            //      I(b[0][5])*R(a[1][0]) I(b[0][5])*I(a[1][0])
            FMA_M_LOOP(ymm0,ymm1,ymm13,ymm14,ymm2,(double const *)(temp_b+ldb*5)+1);

            /*
                a. Perform beta*C using temp_c, beta_real,
                    where beta_real is not zero.
                b. This loop operates with 4x6 block size
                    along n dimension for every Z_NR columns of temp_c where
                    computing all Z_MR rows of temp_c.
                c. Accumulated alpha*A*B into registers will be added to beta*C
                d. Same approach is used in remaining fringe cases.
            */
            if(beta_real != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_real));

                //R(c[0][0]) I(c[0][0]) R(c[1][0]) I(c[1][0])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c));
                //R(c[2][0]) I(c[2][0]) R(c[3][0]) I(c[3][0])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + 2));
                //ymm3+=beta_real*R(c[0][0]) beta_real*I(c[0][0])
                //      beta_real*R(c[1][0]) beta_real*I(c[1][0])
                //ymm4+=beta_real*R(c[2][0]) beta_real*I(c[2][0])
                //      beta_real*R(c[3][0]) beta_real*I(c[3][0])
                SCALE_BETA_REAL_M_LOOP(ymm0,ymm1,ymm3,ymm4,ymm15);

                //R(c[0][1]) I(c[0][1]) R(c[1][1]) I(c[1][1])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc));
                //R(c[2][1]) I(c[2][1]) R(c[3][1]) I(c[3][1])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc + 2));
                //ymm5+=beta_real*R(c[0][1]) beta_real*I(c[0][1])
                //      beta_real*R(c[1][1]) beta_real*I(c[1][1])
                //ymm6+=beta_real*R(c[2][1]) beta_real*I(c[2][1])
                //      beta_real*R(c[3][1]) beta_real*I(c[3][1])
                SCALE_BETA_REAL_M_LOOP(ymm0,ymm1,ymm5,ymm6,ymm15);

                //R(c[0][2]) I(c[0][2]) R(c[1][2]) I(c[1][2])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*2));
                //R(c[2][2]) I(c[2][2]) R(c[3][2]) I(c[3][2])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc*2 + 2));
                //ymm7+=beta_real*R(c[0][2]) beta_real*I(c[0][2])
                //      beta_real*R(c[1][2]) beta_real*I(c[1][2])
                //ymm8+=beta_real*R(c[2][2]) beta_real*I(c[2][2])
                        //beta_real*R(c[3][2]) beta_real*I(c[3][2])
                SCALE_BETA_REAL_M_LOOP(ymm0,ymm1,ymm7,ymm8,ymm15);

                //R(c[0][3]) I(c[0][3]) R(c[1][3]) I(c[1][3])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*3));
                //R(c[2][3]) I(c[2][3]) R(c[3][3]) I(c[3][3])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc*3 + 2));
                //ymm9+=beta_real*R(c[0][3]) beta_real*I(c[0][3])
                //      beta_real*R(c[1][3]) beta_real*I(c[1][3])
                //ymm10+=beta_real*R(c[2][3]) beta_real*I(c[2][3])
                //      beta_real*R(c[3][3]) beta_real*I(c[3][3])
                SCALE_BETA_REAL_M_LOOP(ymm0,ymm1,ymm9,ymm10,ymm15);

                //R(c[0][4]) I(c[0][4]) R(c[1][4]) I(c[1][4])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*4));
                //R(c[2][4]) I(c[2][4]) R(c[3][4]) I(c[3][4])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc*4 + 2));
                //ymm11+=beta_real*R(c[0][4]) beta_real*I(c[0][4])
                //      beta_real*R(c[1][4]) beta_real*I(c[1][4])
                //ymm12+=beta_real*R(c[2][4]) beta_real*I(c[2][4])
                //      beta_real*R(c[3][4]) beta_real*I(c[3][4])
                SCALE_BETA_REAL_M_LOOP(ymm0,ymm1,ymm11,ymm12,ymm15);

                //R(c[0][5]) I(c[0][5]) R(c[1][5]) I(c[1][5])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*5));
                //R(c[2][5]) I(c[2][5]) R(c[3][5]) I(c[3][5])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc*5 + 2));
                //ymm13+=beta_real*R(c[0][5]) beta_real*I(c[0][5])
                //      beta_real*R(c[1][5]) beta_real*I(c[1][5])
                //ymm14+=beta_real*R(c[2][5]) beta_real*I(c[2][5])
                //      beta_real*R(c[3][5]) beta_real*I(c[3][5])
                SCALE_BETA_REAL_M_LOOP(ymm0,ymm1,ymm13,ymm14,ymm15);
            }

            /*
                a. Perform beta*C using temp_c, beta_imag,
                    where beta_imag is not zero.
                b. This loop operates with 4x6 block size
                    along n dimension for every Z_NR columns of temp_c where
                    computing all Z_MR rows of temp_c.
                c. Accumulated alpha*A*B into registers will be added to beta*C
                d. Same approach is used in remaining fringe cases.
            */

            if(beta_imag != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_imag));

                //R(c[0][0]) I(c[0][0]) R(c[1][0]) I(c[1][0])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c));
                //R(c[2][0]) I(c[2][0]) R(c[3][0]) I(c[3][0])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + 2));
                //ymm3+=beta_imag*(-I(c[0][0])) beta_imag*R(c[0][0])
                //      beta_imag*(-I(c[1][0])) beta_imag*R(c[1][0])
                //ymm4+=beta_imag*(-I(c[2][0])) beta_imag*R(c[2][0])
                //      beta_imag*(-I(c[3][0])) beta_imag*R(c[3][0])
                SCALE_BETA_IMAG_M_LOOP(ymm0,ymm1,ymm3,ymm4,ymm15,ymm2);

                //R(c[0][1]) I(c[0][1]) R(c[1][1]) I(c[1][1])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc));
                //R(c[2][1]) I(c[2][1]) R(c[3][1]) I(c[3][1])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc + 2));
                //ymm5+=beta_imag*(-I(c[0][1])) beta_imag*R(c[0][1])
                //      beta_imag*(-I(c[1][1])) beta_imag*R(c[1][1])
                //ymm6+=beta_imag*(-I(c[2][1])) beta_imag*R(c[2][1])
                //      beta_imag*(-I(c[3][1])) beta_imag*R(c[3][1])
                SCALE_BETA_IMAG_M_LOOP(ymm0,ymm1,ymm5,ymm6,ymm15,ymm2);

                //R(c[0][2]) I(c[0][2]) R(c[1][2]) I(c[1][2])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*2));
                //R(c[2][2]) I(c[2][2]) R(c[3][2]) I(c[3][2])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc*2 + 2));
                //ymm7+=beta_imag*(-I(c[0][2])) beta_imag*R(c[0][2])
                //      beta_imag*(-I(c[1][2])) beta_imag*R(c[1][2])
                //ymm8+=beta_imag*(-I(c[2][2])) beta_imag*R(c[2][2])
                //      beta_imag*(-I(c[3][2])) beta_imag*R(c[3][2])
                SCALE_BETA_IMAG_M_LOOP(ymm0,ymm1,ymm7,ymm8,ymm15,ymm2);

                //R(c[0][3]) I(c[0][3]) R(c[1][3]) I(c[1][3])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*3));
                //R(c[2][3]) I(c[2][3]) R(c[3][3]) I(c[3][3])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc*3 + 2));
                //ymm9+=beta_imag*(-I(c[0][3])) beta_imag*R(c[0][3])
                //      beta_imag*(-I(c[1][3])) beta_imag*R(c[1][3])
                //ymm10+=beta_imag*(-I(c[2][3])) beta_imag*R(c[2][3])
                //      beta_imag*(-I(c[3][3])) beta_imag*R(c[3][3])
                SCALE_BETA_IMAG_M_LOOP(ymm0,ymm1,ymm9,ymm10,ymm15,ymm2);

                //R(c[0][4]) I(c[0][4]) R(c[1][4]) I(c[1][4])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*4));
                //R(c[2][4]) I(c[2][4]) R(c[3][4]) I(c[3][4])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc*4 + 2));
                //ymm11+=beta_imag*(-I(c[0][4])) beta_imag*R(c[0][4])
                //      beta_imag*(-I(c[1][4])) beta_imag*R(c[1][4])
                //ymm12+=beta_imag*(-I(c[2][4])) beta_imag*R(c[2][4])
                //      beta_imag*(-I(c[3][4])) beta_imag*R(c[3][4])
                SCALE_BETA_IMAG_M_LOOP(ymm0,ymm1,ymm11,ymm12,ymm15,ymm2);

                //R(c[0][5]) I(c[0][5]) R(c[1][5]) I(c[1][5])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*5));
                //R(c[2][5]) I(c[2][5]) R(c[3][5]) I(c[3][5])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc*5 + 2));
                //ymm13+=beta_imag*(-I(c[0][5])) beta_imag*R(c[0][5])
                //      beta_imag*(-I(c[1][5])) beta_imag*R(c[1][5])
                //ymm14+=beta_imag*(-I(c[2][5])) beta_imag*R(c[2][5])
                //      beta_imag*(-I(c[3][5])) beta_imag*R(c[3][5])
                SCALE_BETA_IMAG_M_LOOP(ymm0,ymm1,ymm13,ymm14,ymm15,ymm2);
            }
            /*
            The scaling has been done sequentially as follows:
            - If alpha_real is not 0, it is used for scaling A
            - If alpha_imag is not 0, it is used for scaling A using permutation
              and selective negation, after loading
            - If beta_real is not 0, is is used for scaling C
            - If beta_imag is not 0, it is used for scaling C using permutation
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

            _mm256_storeu_pd((double *)(temp_c + ldc*5), ymm13);
            _mm256_storeu_pd((double *)(temp_c + ldc*5 + 2), ymm14);

            temp_c+=Z_MR;
            temp_a+=Z_MR;
        }

        // Fringe cases for M
        dim_t m_rem=m_remainder;
        if(m_rem>=2)
        {
            ymm3 = _mm256_setzero_pd();
            ymm5 = _mm256_setzero_pd();
            ymm7 = _mm256_setzero_pd();
            ymm9 = _mm256_setzero_pd();
            ymm11 = _mm256_setzero_pd();

            //R(a[0][0]) I(a[0][0]) R(a[1][0]) I(a[1][0])
            ymm0 = _mm256_loadu_pd((double const *)(temp_a));

            ymm13 = ymm0;
            SCALE_ALPHA_REAL_M_FRINGE(ymm0,ymm15,alpha_real);
            SCALE_ALPHA_IMAG_M_FRINGE(ymm0,ymm13,ymm15,ymm2,alpha_imag);

            ymm13 = _mm256_setzero_pd();

            /*
            The result after scaling with alpha_real and/or alpha_imag is as follows:
            For ymm0 :
            R(a[0][0]) = alpha_real*R(a[0][0])-alpha_imag*I(a[0][0])
            I(a[0][0]) = alpha_real*I(a[0][0])+alpha_imag*R[0][0]
            R(a[1][0]) = alpha_real*R(a[1][0])-alpha_imag*I(a[1][0])
            I(a[1][0]) = alpha_real*I(a[1][0])+alpha_imag*(R[1][0])
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
            //ymm13+=R(b[0][5])*R(a[0][0]) R(b[0][5])*I(a[0][0])
            //      R(b[0][5])*R(a[1][0]) R(b[0][5])*I(a[1][0])
            FMA_M_FRINGE(ymm0,ymm13,ymm2,(double const *)(temp_b+ldb*5));

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
            //ymm13+=I(b[0][5])*R(a[0][0]) I(b[0][5])*I(a[0][0])
            //      I(b[0][5])*R(a[1][0]) I(b[0][5])*I(a[1][0])
            FMA_M_FRINGE(ymm0,ymm13,ymm2,(double const *)(temp_b+ldb*5)+1);


            if(beta_real != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_real));

                //R(c[0][0]) I(c[0][0]) R(c[1][0]) I(c[1][0])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c));
                //ymm3+=beta_real*R(c[0][0]) beta_real*I(c[0][0])
                //      beta_real*R(c[1][0]) beta_real*I(c[1][0])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm3,ymm15);

                //R(c[0][1]) I(c[0][1]) R(c[1][1]) I(c[1][1])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc));
                //ymm5+=beta_real*R(c[0][1]) beta_real*I(c[0][1])
                //      beta_real*R(c[1][1]) beta_real*I(c[1][1])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm5,ymm15);

                //R(c[0][2]) I(c[0][2]) R(c[1][2]) I(c[1][2])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*2));
                //ymm7+=beta_real*R(c[0][2]) beta_real*I(c[0][2])
                //      beta_real*R(c[1][2]) beta_real*I(c[1][2])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm7,ymm15);

                //R(c[0][3]) I(c[0][3]) R(c[1][3]) I(c[1][3])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*3));
                //ymm9+=beta_real*R(c[0][3]) beta_real*I(c[0][3])
                //      beta_real*R(c[1][3]) beta_real*I(c[1][3])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm9,ymm15);

                //R(c[0][4]) I(c[0][4]) R(c[1][4]) I(c[1][4])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*4));
                //ymm11+=beta_real*R(c[0][4]) beta_real*I(c[0][4])
                //      beta_real*R(c[1][4]) beta_real*I(c[1][4])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm11,ymm15);

                //R(c[0][5]) I(c[0][5]) R(c[1][5]) I(c[1][5])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*5));
                //ymm13+=beta_real*R(c[0][5]) beta_real*I(c[0][5])
                //      beta_real*R(c[1][5]) beta_real*I(c[1][5])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm13,ymm15);
            }


            if(beta_imag != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_imag));

                //R(c[0][0]) I(c[0][0]) R(c[1][0]) I(c[1][0])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c));
                //ymm3+=beta_imag*(-I(c[0][0])) beta_imag*R(c[0][0])
                //      beta_imag*(-I(c[1][0])) beta_imag*R(c[1][0])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm3,ymm15,ymm2);

                //R(c[0][1]) I(c[0][1]) R(c[1][1]) I(c[1][1])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc));
                //ymm5+=beta_imag*(-I(c[0][1])) beta_imag*R(c[0][1])
                //      beta_imag*(-I(c[1][1])) beta_imag*R(c[1][1])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm5,ymm15,ymm2);

                //R(c[0][2]) I(c[0][2]) R(c[1][2]) I(c[1][2])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*2));
                //ymm7+=beta_imag*(-I(c[0][2])) beta_imag*R(c[0][2])
                //      beta_imag*(-I(c[1][2])) beta_imag*R(c[1][2])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm7,ymm15,ymm2);

                //R(c[0][3]) I(c[0][3]) R(c[1][3]) I(c[1][3])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*3));
                //ymm9+=beta_imag*(-I(c[0][3])) beta_imag*R(c[0][3])
                //      beta_imag*(-I(c[1][3])) beta_imag*R(c[1][3])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm9,ymm15,ymm2);

                //R(c[0][4]) I(c[0][4]) R(c[1][4]) I(c[1][4])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*4));
                //ymm11+=beta_imag*(-I(c[0][4])) beta_imag*R(c[0][4])
                //      beta_imag*(-I(c[1][4])) beta_imag*R(c[1][4])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm11,ymm15,ymm2);

                //R(c[0][5]) I(c[0][5]) R(c[1][5]) I(c[1][5])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*5));
                //ymm13+=beta_imag*(-I(c[0][5])) beta_imag*R(c[0][5])
                //      beta_imag*(-I(c[1][5])) beta_imag*R(c[1][5])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm13,ymm15,ymm2);
            }

            /*
            The scaling has been done sequentially as follows:
            - If alpha_real is not 0, it is used for scaling A
            - If alpha_imag is not 0, it is used for scaling A using permutation
              and selective negation, after loading
            - If beta_real is not 0, is is used for scaling C
            - If beta_imag is not 0, it is used for scaling C using permutation
              and selective negation, after loading

            The results are accumalated in accordance to the non zero scalar values.
            */

            _mm256_storeu_pd((double *)(temp_c), ymm3);
            _mm256_storeu_pd((double *)(temp_c + ldc), ymm5);
            _mm256_storeu_pd((double *)(temp_c + ldc*2), ymm7);
            _mm256_storeu_pd((double *)(temp_c + ldc*3), ymm9);
            _mm256_storeu_pd((double *)(temp_c + ldc*4), ymm11);
            _mm256_storeu_pd((double *)(temp_c + ldc*5), ymm13);

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

            xmm5 = _mm_loadu_pd((double const*)(temp_a));//R(a[0][0]) I(a[0][0])
            ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(a[0][0]) I(a[0][0])

            ymm13 = ymm0;
            SCALE_ALPHA_REAL_M_FRINGE(ymm0,ymm15,alpha_real);
            SCALE_ALPHA_IMAG_M_FRINGE(ymm0,ymm13,ymm15,ymm2,alpha_imag);

            ymm13 = _mm256_setzero_pd();

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
            //ymm13+=R(b[0][5])*R(a[0][0]) R(b[0][5])*I(a[0][0])
            //      R(b[0][5])*R(a[1][0]) R(b[0][5])*I(a[1][0])
            FMA_M_FRINGE(ymm0,ymm13,ymm2,(double const *)(temp_b+ldb*5));

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
            //ymm13+=I(b[0][5])*R(a[0][0]) I(b[0][5])*I(a[0][0])
            //      I(b[0][5])*R(a[1][0]) I(b[0][5])*I(a[1][0])
            FMA_M_FRINGE(ymm0,ymm13,ymm2,(double const *)(temp_b+ldb*5)+1);

            if(beta_real != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_real));

                xmm5 = _mm_loadu_pd((double const*)(temp_c));//R(c[0][0]) I(c[0][0])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][0]) I(c[0][0])
                //ymm3+=beta_real*R(c[0][0]) beta_real*I(c[0][0])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm3,ymm15);

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc));//R(c[0][1]) I(c[0][1])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][1]) I(c[0][1])
                //ymm5+=beta_real*R(c[0][1]) beta_real*I(c[0][1])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm5,ymm15);

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc * 2));//R(c[0][2]) I(c[0][2])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][2]) I(c[0][2])
                //ymm7+=beta_real*R(c[0][2]) beta_real*I(c[0][2])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm7,ymm15);

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc * 3));//R(c[0][3]) I(c[0][3])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][3]) I(c[0][3])
                //ymm9+=beta_real*R(c[0][3]) beta_real*I(c[0][3])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm9,ymm15);

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc * 4));//R(c[0][4]) I(c[0][4])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][4]) I(c[0][4])
                //ymm11+=beta_real*R(c[0][4]) beta_real*I(c[0][4])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm11,ymm15);

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc * 5));//R(c[0][5]) I(c[0][5])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][5]) I(c[0][5])
                //ymm13+=beta_real*R(c[0][5]) beta_real*I(c[0][5])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm13,ymm15);
            }

            if(beta_imag != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_imag));

                xmm5 = _mm_loadu_pd((double const*)(temp_c));//R(c[0][0]) I(c[0][0])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][0]) I(c[0][0])
                //ymm3+=beta_imag*(-I(c[0][0])) beta_imag*R(c[0][0])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm3,ymm15,ymm2);

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc));//R(c[0][1]) I(c[0][1])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][1]) I(c[0][1])
                //ymm5+=beta_imag*(-I(c[0][1])) beta_imag*R(c[0][1])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm5,ymm15,ymm2);

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc * 2));//R(c[0][2]) I(c[0][2])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][2]) I(c[0][2])
                //ymm7+=beta_imag*(-I(c[0][2])) beta_imag*R(c[0][2])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm7,ymm15,ymm2);

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc * 3));//R(c[0][3]) I(c[0][3])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][3]) I(c[0][3])
                //ymm9+=beta_imag*(-I(c[0][3])) beta_imag*R(c[0][3])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm9,ymm15,ymm2);

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc * 4));//R(c[0][4]) I(c[0][4])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][4]) I(c[0][4])
                //ymm11+=beta_imag*(-I(c[0][4])) beta_imag*R(c[0][4])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm11,ymm15,ymm2);

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc * 5));//R(c[0][5]) I(c[0][5])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][5]) I(c[0][5])
                //ymm13+=beta_imag*(-I(c[0][5])) beta_imag*R(c[0][5])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm13,ymm15,ymm2);
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

            xmm5 = _mm256_extractf128_pd(ymm13, 0);
            _mm_storeu_pd((double *)(temp_c + ldc*5), xmm5);

        }

    }

    //Fringe case for N
    if(n_remainder>=4)
    {
        dcomplex* temp_b = b + (n - n_remainder)*ldb;
        dcomplex* temp_a = a;
        dcomplex* temp_c = c + (n - n_remainder)*ldc;

        //Main loop for M
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

            /*
                a. Perform alpha*A*B using temp_a, temp_b and alpha_real, alpha_imag
                    where alpha_real and/or alpha_imag is not zero.
                b. This loop operates with 4x6 block size
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
            SCALE_ALPHA_REAL_M_LOOP(ymm0,ymm1,ymm15,alpha_real);
            SCALE_ALPHA_IMAG_M_LOOP(ymm0,ymm1,ymm13,ymm14,ymm15,ymm2,alpha_imag);

            /*
            The result after scaling with alpha_real and/or alpha_imag is as follows:
            For ymm0 :
            R(a[0][0]) = alpha_real*R(a[0][0])-alpha_imag*I(a[0][0])
            I(a[0][0]) = alpha_real*I(a[0][0])+alpha_imag*R[0][0]
            R(a[1][0]) = alpha_real*R(a[1][0])-alpha_imag*I(a[1][0])
            I(a[1][0]) = alpha_real*I(a[1][0])+alpha_imag*(R[1][0])

            For ymm1 :
            R(a[2][0]) = alpha_real*R(a[2][0])-alpha_imag*I(a[2][0])
            I(a[2][0]) = alpha_real*I(a[2][0])+alpha_imag*R[2][0]
            R(a[3][0]) = alpha_real*R(a[3][0])-alpha_imag*I(a[3][0])
            I(a[3][0]) = alpha_real*I(a[3][0])+alpha_imag*(R[3][0])
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

            /*
                a. Perform beta*C using temp_c, beta_real,
                    where beta_real is not zero.
                b. This loop operates with 4x6 block size
                    along n dimension for every Z_NR columns of temp_c where
                    computing all Z_MR rows of temp_c.
                c. Accumulated alpha*A*B into registers will be added to beta*C
                d. Same approach is used in remaining fringe cases.
            */
            if(beta_real != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_real));

                //R(c[0][0]) I(c[0][0]) R(c[1][0]) I(c[1][0])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c));
                //R(c[2][0]) I(c[2][0]) R(c[3][0]) I(c[3][0])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + 2));
                //ymm3+=beta_real*R(c[0][0]) beta_real*I(c[0][0])
                //      beta_real*R(c[1][0]) beta_real*I(c[1][0])
                //ymm4+=beta_real*R(c[2][0]) beta_real*I(c[2][0])
                //      beta_real*R(c[3][0]) beta_real*I(c[3][0])
                SCALE_BETA_REAL_M_LOOP(ymm0,ymm1,ymm3,ymm4,ymm15);

                //R(c[0][1]) I(c[0][1]) R(c[1][1]) I(c[1][1])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc));
                //R(c[2][1]) I(c[2][1]) R(c[3][1]) I(c[3][1])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc + 2));
                //ymm5+=beta_real*R(c[0][1]) beta_real*I(c[0][1])
                //      beta_real*R(c[1][1]) beta_real*I(c[1][1])
                //ymm6+=beta_real*R(c[2][1]) beta_real*I(c[2][1])
                //      beta_real*R(c[3][1]) beta_real*I(c[3][1])
                SCALE_BETA_REAL_M_LOOP(ymm0,ymm1,ymm5,ymm6,ymm15);

                //R(c[0][2]) I(c[0][2]) R(c[1][2]) I(c[1][2])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*2));
                //R(c[2][2]) I(c[2][2]) R(c[3][2]) I(c[3][2])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc*2 + 2));
                //ymm7+=beta_real*R(c[0][2]) beta_real*I(c[0][2])
                //      beta_real*R(c[1][2]) beta_real*I(c[1][2])
                //ymm8+=beta_real*R(c[2][2]) beta_real*I(c[2][2])
                //      beta_real*R(c[3][2]) beta_real*I(c[3][2])
                SCALE_BETA_REAL_M_LOOP(ymm0,ymm1,ymm7,ymm8,ymm15);

                //R(c[0][3]) I(c[0][3]) R(c[1][3]) I(c[1][3])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*3));
                //R(c[2][3]) I(c[2][3]) R(c[3][3]) I(c[3][3])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc*3 + 2));
                //ymm9+=beta_real*R(c[0][3]) beta_real*I(c[0][3])
                //      beta_real*R(c[1][3]) beta_real*I(c[1][3])
                //ymm10+=beta_real*R(c[2][3]) beta_real*I(c[2][3])
                //      beta_real*R(c[3][3]) beta_real*I(c[3][3])
                SCALE_BETA_REAL_M_LOOP(ymm0,ymm1,ymm9,ymm10,ymm15);
            }
            /*
                a. Perform beta*C using temp_c, beta_imag,
                    where beta_imag is not zero.
                b. This loop operates with 4x6 block size
                    along n dimension for every Z_NR columns of temp_c where
                    computing all Z_MR rows of temp_c.
                c. Accumulated alpha*A*B into registers will be added to beta*C
                d. Same approach is used in remaining fringe cases.
            */

            if(beta_imag != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_imag));

                //R(c[0][0]) I(c[0][0]) R(c[1][0]) I(c[1][0])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c));
                //R(c[2][0]) I(c[2][0]) R(c[3][0]) I(c[3][0])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + 2));
                //ymm3+=beta_imag*(-I(c[0][0])) beta_imag*R(c[0][0])
                //      beta_imag*(-I(c[1][0])) beta_imag*R(c[1][0])
                //ymm4+=beta_imag*(-I(c[2][0])) beta_imag*R(c[2][0])
                //      beta_imag*(-I(c[3][0])) beta_imag*R(c[3][0])
                SCALE_BETA_IMAG_M_LOOP(ymm0,ymm1,ymm3,ymm4,ymm15,ymm2);

                //R(c[0][1]) I(c[0][1]) R(c[1][1]) I(c[1][1])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc));
                //R(c[2][1]) I(c[2][1]) R(c[3][1]) I(c[3][1])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc + 2));
                //ymm5+=beta_imag*(-I(c[0][1])) beta_imag*R(c[0][1])
                //      beta_imag*(-I(c[1][1])) beta_imag*R(c[1][1])
                //ymm6+=beta_imag*(-I(c[2][1])) beta_imag*R(c[2][1])
                //      beta_imag*(-I(c[3][1])) beta_imag*R(c[3][1])
                SCALE_BETA_IMAG_M_LOOP(ymm0,ymm1,ymm5,ymm6,ymm15,ymm2);

                //R(c[0][2]) I(c[0][2]) R(c[1][2]) I(c[1][2])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*2));
                //R(c[2][2]) I(c[2][2]) R(c[3][2]) I(c[3][2])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc*2 + 2));
                //ymm7+=beta_imag*(-I(c[0][2])) beta_imag*R(c[0][2])
                //      beta_imag*(-I(c[1][2])) beta_imag*R(c[1][2])
                //ymm8+=beta_imag*(-I(c[2][2])) beta_imag*R(c[2][2])
                //      beta_imag*(-I(c[3][2])) beta_imag*R(c[3][2])
                SCALE_BETA_IMAG_M_LOOP(ymm0,ymm1,ymm7,ymm8,ymm15,ymm2);

                //R(c[0][3]) I(c[0][3]) R(c[1][3]) I(c[1][3])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*3));
                //R(c[2][3]) I(c[2][3]) R(c[3][3]) I(c[3][3])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc*3 + 2));
                //ymm9+=beta_imag*(-I(c[0][3])) beta_imag*R(c[0][3])
                //      beta_imag*(-I(c[1][3])) beta_imag*R(c[1][3])
                //ymm10+=beta_imag*(-I(c[2][3])) beta_imag*R(c[2][3])
                //      beta_imag*(-I(c[3][3])) beta_imag*R(c[3][3])
                SCALE_BETA_IMAG_M_LOOP(ymm0,ymm1,ymm9,ymm10,ymm15,ymm2);
            }
            /*
            The scaling has been done sequentially as follows:
            - If alpha_real is not 0, it is used for scaling A
            - If alpha_imag is not 0, it is used for scaling A using permutation
              and selective negation, after loading
            - If beta_real is not 0, is is used for scaling C
            - If beta_imag is not 0, it is used for scaling C using permutation
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

        // Fringe cases for M
        dim_t m_rem=m_remainder;
        if(m_rem>=2)
        {
            ymm3 = _mm256_setzero_pd();
            ymm5 = _mm256_setzero_pd();
            ymm7 = _mm256_setzero_pd();
            ymm9 = _mm256_setzero_pd();


            //R(a[0][0]) I(a[0][0]) R(a[1][0]) I(a[1][0])
            ymm0 = _mm256_loadu_pd((double const *)(temp_a));

            ymm13 = ymm0;
            SCALE_ALPHA_REAL_M_FRINGE(ymm0,ymm15,alpha_real);
            SCALE_ALPHA_IMAG_M_FRINGE(ymm0,ymm13,ymm15,ymm2,alpha_imag);
            /*
            The result after scaling with alpha_real and/or alpha_imag is as follows:
            For ymm0 :
            R(a[0][0]) = alpha_real*R(a[0][0])-alpha_imag*I(a[0][0])
            I(a[0][0]) = alpha_real*I(a[0][0])+alpha_imag*R[0][0]
            R(a[1][0]) = alpha_real*R(a[1][0])-alpha_imag*I(a[1][0])
            I(a[1][0]) = alpha_real*I(a[1][0])+alpha_imag*(R[1][0])
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


            if(beta_real != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_real));

                //R(c[0][0]) I(c[0][0]) R(c[1][0]) I(c[1][0])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c));
                //ymm3+=beta_real*R(c[0][0]) beta_real*I(c[0][0])
                //      beta_real*R(c[1][0]) beta_real*I(c[1][0])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm3,ymm15);

                //R(c[0][1]) I(c[0][1]) R(c[1][1]) I(c[1][1])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc));
                //ymm5+=beta_real*R(c[0][1]) beta_real*I(c[0][1])
                //      beta_real*R(c[1][1]) beta_real*I(c[1][1])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm5,ymm15);

                //R(c[0][2]) I(c[0][2]) R(c[1][2]) I(c[1][2])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*2));
                //ymm7+=beta_real*R(c[0][2]) beta_real*I(c[0][2])
                //      beta_real*R(c[1][2]) beta_real*I(c[1][2])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm7,ymm15);

                //R(c[0][3]) I(c[0][3]) R(c[1][3]) I(c[1][3])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*3));
                //ymm9+=beta_real*R(c[0][3]) beta_real*I(c[0][3])
                //      beta_real*R(c[1][3]) beta_real*I(c[1][3])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm9,ymm15);
            }

            if(beta_imag != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_imag));

                //R(c[0][0]) I(c[0][0]) R(c[1][0]) I(c[1][0])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c));
                //ymm3+=beta_imag*(-I(c[0][0])) beta_imag*R(c[0][0])
                //      beta_imag*(-I(c[1][0])) beta_imag*R(c[1][0])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm3,ymm15,ymm2);

                //R(c[0][1]) I(c[0][1]) R(c[1][1]) I(c[1][1])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc));
                //ymm5+=beta_imag*(-I(c[0][1])) beta_imag*R(c[0][1])
                //      beta_imag*(-I(c[1][1])) beta_imag*R(c[1][1])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm5,ymm15,ymm2);

                //R(c[0][2]) I(c[0][2]) R(c[1][2]) I(c[1][2])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*2));
                //ymm7+=beta_imag*(-I(c[0][2])) beta_imag*R(c[0][2])
                //      beta_imag*(-I(c[1][2])) beta_imag*R(c[1][2])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm7,ymm15,ymm2);

                //R(c[0][3]) I(c[0][3]) R(c[1][3]) I(c[1][3])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc*3));
                //ymm9+=beta_imag*(-I(c[0][3])) beta_imag*R(c[0][3])
                //      beta_imag*(-I(c[1][3])) beta_imag*R(c[1][3])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm9,ymm15,ymm2);
            }

            /*
            The scaling has been done sequentially as follows:
            - If alpha_real is not 0, it is used for scaling A
            - If alpha_imag is not 0, it is used for scaling A using permutation
              and selective negation, after loading
            - If beta_real is not 0, is is used for scaling C
            - If beta_imag is not 0, it is used for scaling C using permutation
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

            xmm5 = _mm_loadu_pd((double const*)(temp_a));//R(a[0][0]) I(a[0][0])
            ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(a[0][0]) I(a[0][0])

            ymm13 = ymm0;
            SCALE_ALPHA_REAL_M_FRINGE(ymm0,ymm15,alpha_real);
            SCALE_ALPHA_IMAG_M_FRINGE(ymm0,ymm13,ymm15,ymm2,alpha_imag);

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

            if(beta_real != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_real));

                xmm5 = _mm_loadu_pd((double const*)(temp_c));//R(c[0][0]) I(c[0][0])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][0]) I(c[0][0])
                //ymm3+=beta_real*R(c[0][0]) beta_real*I(c[0][0])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm3,ymm15);

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc));//R(c[0][1]) I(c[0][1])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][1]) I(c[0][1])
                //ymm5+=beta_real*R(c[0][1]) beta_real*I(c[0][1])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm5,ymm15);

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc * 2));//R(c[0][2]) I(c[0][2])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][2]) I(c[0][2])
                //ymm7+=beta_real*R(c[0][2]) beta_real*I(c[0][2])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm7,ymm15);

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc * 3));//R(c[0][3]) I(c[0][3])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][3]) I(c[0][3])
                //ymm9+=beta_real*R(c[0][3]) beta_real*I(c[0][3])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm9,ymm15);
            }

            if(beta_imag != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_imag));

                xmm5 = _mm_loadu_pd((double const*)(temp_c));//R(c[0][0]) I(c[0][0])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][0]) I(c[0][0])
                //ymm3+=beta_imag*(-I(c[0][0])) beta_imag*R(c[0][0])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm3,ymm15,ymm2);

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc));//R(c[0][1]) I(c[0][1])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][1]) I(c[0][1])
                //ymm5+=beta_imag*(-I(c[0][1])) beta_imag*R(c[0][1])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm5,ymm15,ymm2);

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc * 2));//R(c[0][2]) I(c[0][2])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][2]) I(c[0][2])
                //ymm7+=beta_imag*(-I(c[0][2])) beta_imag*R(c[0][2])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm7,ymm15,ymm2);

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc * 3));//R(c[0][3]) I(c[0][3])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][3]) I(c[0][3])
                //ymm9+=beta_imag*(-I(c[0][3])) beta_imag*R(c[0][3])
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

            /*
                a. Perform alpha*A*B using temp_a, temp_b and alpha_real, alpha_imag
                    where alpha_real and/or alpha_imag is not zero.
                b. This loop operates with 4x6 block size
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
            SCALE_ALPHA_REAL_M_LOOP(ymm0,ymm1,ymm15,alpha_real);
            SCALE_ALPHA_IMAG_M_LOOP(ymm0,ymm1,ymm13,ymm14,ymm15,ymm2,alpha_imag);

            /*
            The result after scaling with alpha_real and/or alpha_imag is as follows:
            For ymm0 :
            R(a[0][0]) = alpha_real*R(a[0][0])-alpha_imag*I(a[0][0])
            I(a[0][0]) = alpha_real*I(a[0][0])+alpha_imag*R[0][0]
            R(a[1][0]) = alpha_real*R(a[1][0])-alpha_imag*I(a[1][0])
            I(a[1][0]) = alpha_real*I(a[1][0])+alpha_imag*(R[1][0])

            For ymm1 :
            R(a[2][0]) = alpha_real*R(a[2][0])-alpha_imag*I(a[2][0])
            I(a[2][0]) = alpha_real*I(a[2][0])+alpha_imag*R[2][0]
            R(a[3][0]) = alpha_real*R(a[3][0])-alpha_imag*I(a[3][0])
            I(a[3][0]) = alpha_real*I(a[3][0])+alpha_imag*(R[3][0])
            */

            //Calculating using real part of complex number in B matrix
            FMA_M_LOOP(ymm0,ymm1,ymm3,ymm4,ymm2,(double const *)(temp_b));
            FMA_M_LOOP(ymm0,ymm1,ymm5,ymm6,ymm2,(double const *)(temp_b+ldb));

            //Calculating using imaginary part of complex numbers in B matrix
            //Shuffling ymm0 and ymm1 in accordance to the requirement
            NEG_PERM_M_LOOP(ymm0,ymm1,ymm2);
            FMA_M_LOOP(ymm0,ymm1,ymm3,ymm4,ymm2,(double const *)(temp_b)+1);
            FMA_M_LOOP(ymm0,ymm1,ymm5,ymm6,ymm2,(double const *)(temp_b+ldb)+1);

            /*
                a. Perform beta*C using temp_c, beta_real,
                    where beta_real is not zero.
                b. This loop operates with 4x6 block size
                    along n dimension for every Z_NR columns of temp_c where
                    computing all Z_MR rows of temp_c.
                c. Accumulated alpha*A*B into registers will be added to beta*C
                d. Same approach is used in remaining fringe cases.
            */
            if(beta_real != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_real));

                //R(c[0][0]) I(c[0][0]) R(c[1][0]) I(c[1][0])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c));
                //R(c[2][0]) I(c[2][0]) R(c[3][0]) I(c[3][0])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + 2));
                //ymm3+=beta_real*R(c[0][0]) beta_real*I(c[0][0])
                //      beta_real*R(c[1][0]) beta_real*I(c[1][0])
                //ymm4+=beta_real*R(c[2][0]) beta_real*I(c[2][0])
                //      beta_real*R(c[3][0]) beta_real*I(c[3][0])
                SCALE_BETA_REAL_M_LOOP(ymm0,ymm1,ymm3,ymm4,ymm15);

                //R(c[0][1]) I(c[0][1]) R(c[1][1]) I(c[1][1])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc));
                //R(c[2][1]) I(c[2][1]) R(c[3][1]) I(c[3][1])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc + 2));
                //ymm5+=beta_real*R(c[0][1]) beta_real*I(c[0][1])
                //      beta_real*R(c[1][1]) beta_real*I(c[1][1])
                //ymm6+=beta_real*R(c[2][1]) beta_real*I(c[2][1])
                //      beta_real*R(c[3][1]) beta_real*I(c[3][1])
                SCALE_BETA_REAL_M_LOOP(ymm0,ymm1,ymm5,ymm6,ymm15);
            }

            /*
                a. Perform beta*C using temp_c, beta_imag,
                    where beta_imag is not zero.
                b. This loop operates with 4x6 block size
                    along n dimension for every Z_NR columns of temp_c where
                    computing all Z_MR rows of temp_c.
                c. Accumulated alpha*A*B into registers will be added to beta*C
                d. Same approach is used in remaining fringe cases.
            */

            if(beta_imag != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_imag));

                //R(c[0][0]) I(c[0][0]) R(c[1][0]) I(c[1][0])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c));
                //R(c[2][0]) I(c[2][0]) R(c[3][0]) I(c[3][0])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + 2));
                //ymm3+=beta_imag*(-I(c[0][0])) beta_imag*R(c[0][0])
                //      beta_imag*(-I(c[1][0])) beta_imag*R(c[1][0])
                //ymm4+=beta_imag*(-I(c[2][0])) beta_imag*R(c[2][0])
                //      beta_imag*(-I(c[3][0])) beta_imag*R(c[3][0])
                SCALE_BETA_IMAG_M_LOOP(ymm0,ymm1,ymm3,ymm4,ymm15,ymm2);

                //R(c[0][1]) I(c[0][1]) R(c[1][1]) I(c[1][1])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc));
                //R(c[2][1]) I(c[2][1]) R(c[3][1]) I(c[3][1])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + ldc + 2));
                //ymm5+=beta_imag*(-I(c[0][1])) beta_imag*R(c[0][1])
                //      beta_imag*(-I(c[1][1])) beta_imag*R(c[1][1])
                //ymm6+=beta_imag*(-I(c[2][1])) beta_imag*R(c[2][1])
                //      beta_imag*(-I(c[3][1])) beta_imag*R(c[3][1])
                SCALE_BETA_IMAG_M_LOOP(ymm0,ymm1,ymm5,ymm6,ymm15,ymm2);
            }
            /*
            The scaling has been done sequentially as follows:
            - If alpha_real is not 0, it is used for scaling A
            - If alpha_imag is not 0, it is used for scaling A using permutation
              and selective negation, after loading
            - If beta_real is not 0, is is used for scaling C
            - If beta_imag is not 0, it is used for scaling C using permutation
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


            //R(a[0][0]) I(a[0][0]) R(a[1][0]) I(a[1][0])
            ymm0 = _mm256_loadu_pd((double const *)(temp_a));

            ymm13 = ymm0;
            SCALE_ALPHA_REAL_M_FRINGE(ymm0,ymm15,alpha_real);
            SCALE_ALPHA_IMAG_M_FRINGE(ymm0,ymm13,ymm15,ymm2,alpha_imag);
            /*
            The result after scaling with alpha_real and/or alpha_imag is as follows:
            For ymm0 :
            R(a[0][0]) = alpha_real*R(a[0][0])-alpha_imag*I(a[0][0])
            I(a[0][0]) = alpha_real*I(a[0][0])+alpha_imag*R[0][0]
            R(a[1][0]) = alpha_real*R(a[1][0])-alpha_imag*I(a[1][0])
            I(a[1][0]) = alpha_real*I(a[1][0])+alpha_imag*(R[1][0])
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


            if(beta_real != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_real));

                //R(c[0][0]) I(c[0][0]) R(c[1][0]) I(c[1][0])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c));
                //ymm3+=beta_real*R(c[0][0]) beta_real*I(c[0][0])
                //      beta_real*R(c[1][0]) beta_real*I(c[1][0])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm3,ymm15);

                //R(c[0][1]) I(c[0][1]) R(c[1][1]) I(c[1][1])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc));
                //ymm5+=beta_real*R(c[0][1]) beta_real*I(c[0][1])
                //      beta_real*R(c[1][1]) beta_real*I(c[1][1])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm5,ymm15);
            }

            if(beta_imag != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_imag));

                //R(c[0][0]) I(c[0][0]) R(c[1][0]) I(c[1][0])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c));
                //ymm3+=beta_imag*(-I(c[0][0])) beta_imag*R(c[0][0])
                //      beta_imag*(-I(c[1][0])) beta_imag*R(c[1][0])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm3,ymm15,ymm2);

                //R(c[0][1]) I(c[0][1]) R(c[1][1]) I(c[1][1])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c + ldc));
                //ymm5+=beta_imag*(-I(c[0][1])) beta_imag*R(c[0][1])
                //      beta_imag*(-I(c[1][1])) beta_imag*R(c[1][1])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm5,ymm15,ymm2);
            }

            /*
            The scaling has been done sequentially as follows:
            - If alpha_real is not 0, it is used for scaling A
            - If alpha_imag is not 0, it is used for scaling A using permutation
              and selective negation, after loading
            - If beta_real is not 0, is is used for scaling C
            - If beta_imag is not 0, it is used for scaling C using permutation
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

            xmm5 = _mm_loadu_pd((double const*)(temp_a));//R(a[0][0]) I(a[0][0])
            ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(a[0][0]) I(a[0][0])

            ymm13 = ymm0;
            SCALE_ALPHA_REAL_M_FRINGE(ymm0,ymm15,alpha_real);
            SCALE_ALPHA_IMAG_M_FRINGE(ymm0,ymm13,ymm15,ymm2,alpha_imag);

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

            if(beta_real != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_real));

                xmm5 = _mm_loadu_pd((double const*)(temp_c));//R(c[0][0]) I(c[0][0])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][0]) I(c[0][0])
                //ymm3+=beta_real*R(c[0][0]) beta_real*I(c[0][0])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm3,ymm15);

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc));//R(c[0][1]) I(c[0][1])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][1]) I(c[0][1])
                //ymm5+=beta_real*R(c[0][1]) beta_real*I(c[0][1])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm5,ymm15);
            }

            if(beta_imag != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_imag));

                xmm5 = _mm_loadu_pd((double const*)(temp_c));//R(c[0][0]) I(c[0][0])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][0]) I(c[0][0])
                //ymm3+=beta_imag*(-I(c[0][0])) beta_imag*R(c[0][0])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm3,ymm15,ymm2);

                xmm5 = _mm_loadu_pd((double const*)(temp_c + ldc));//R(c[0][1]) I(c[0][1])
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);//R(c[0][1]) I(c[0][1])
                //ymm5+=beta_imag*(-I(c[0][1])) beta_imag*R(c[0][1])
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

        // Main loop for M
        for(dim_t i = 0;i < (m-Z_MR+1);i=i+Z_MR)
        {
            ymm3 = _mm256_setzero_pd();
            ymm4 = _mm256_setzero_pd();


            /*
                a. Perform alpha*A*B using temp_a, temp_b and alpha_real, aplha_vali
                    where alpha_real and/or alpha_imag is not zero.
                b. This loop operates with 4x6 block size
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
            SCALE_ALPHA_REAL_M_LOOP(ymm0,ymm1,ymm15,alpha_real);
            SCALE_ALPHA_IMAG_M_LOOP(ymm0,ymm1,ymm13,ymm14,ymm15,ymm2,alpha_imag);

            /*
            The result after scaling with alpha_real and/or alpha_imag is as follows:
            For ymm0 :
            R(a[0][0]) = alpha_real*R(a[0][0])-alpha_imag*I(a[0][0])
            I(a[0][0]) = alpha_real*I(a[0][0])+alpha_imag*R[0][0]
            R(a[1][0]) = alpha_real*R(a[1][0])-alpha_imag*I(a[1][0])
            I(a[1][0]) = alpha_real*I(a[1][0])+alpha_imag*(R[1][0])

            For ymm1 :
            R(a[2][0]) = alpha_real*R(a[2][0])-alpha_imag*I(a[2][0])
            I(a[2][0]) = alpha_real*I(a[2][0])+alpha_imag*R[2][0]
            R(a[3][0]) = alpha_real*R(a[3][0])-alpha_imag*I(a[3][0])
            I(a[3][0]) = alpha_real*I(a[3][0])+alpha_imag*(R[3][0])
            */

            //Calculating using real part of complex number in B matrix
            FMA_M_LOOP(ymm0,ymm1,ymm3,ymm4,ymm2,(double const *)(temp_b));

            //Calculating using imaginary part of complex numbers in B matrix
            //Shuffling ymm0 and ymm1 in accordance to the requirement
            NEG_PERM_M_LOOP(ymm0,ymm1,ymm2);
            FMA_M_LOOP(ymm0,ymm1,ymm3,ymm4,ymm2,(double const *)(temp_b)+1);

            /*
                a. Perform beta*C using temp_c, beta_real,
                    where beta_real is not zero.
                b. This loop operates with 4x6 block size
                    along n dimension for every Z_NR columns of temp_c where
                    computing all Z_MR rows of temp_c.
                c. Accumulated alpha*A*B into registers will be added to beta*C
                d. Same approach is used in remaining fringe cases.
            */
            if(beta_real != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_real));

                //R(c[0][0]) I(c[0][0]) R(c[1][0]) I(c[1][0])
                ymm0 = _mm256_loadu_pd((double const *)(temp_c));
                //R(c[2][0]) I(c[2][0]) R(c[3][0]) I(c[3][0])
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + 2));
                    //ymm3+=beta_real*R(c[0][0]) beta_real*I(c[0][0])
                    //      beta_real*R(c[1][0]) beta_real*I(c[1][0])
                    //ymm4+=beta_real*R(c[2][0]) beta_real*I(c[2][0])
                    //      beta_real*R(c[3][0]) beta_real*I(c[3][0])
                SCALE_BETA_REAL_M_LOOP(ymm0,ymm1,ymm3,ymm4,ymm15);
            }

            /*
                a. Perform beta*C using temp_c, beta_imag,
                    where beta_imag is not zero.
                b. This loop operates with 4x6 block size
                    along n dimension for every Z_NR columns of temp_c where
                    computing all Z_MR rows of temp_c.
                c. Accumulated alpha*A*B into registers will be added to beta*C
                d. Same approach is used in remaining fringe cases.
            */

            if(beta_imag != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_imag));

                ymm0 = _mm256_loadu_pd((double const *)(temp_c));
                ymm1 = _mm256_loadu_pd((double const *)(temp_c + 2));
                    //ymm3+=beta_imag*(-I(c[0][0])) beta_imag*R(c[0][0])
                    //      beta_imag*(-I(c[1][0])) beta_imag*R(c[1][0])
                    //ymm4+=beta_imag*(-I(c[2][0])) beta_imag*R(c[2][0])
                    //      beta_imag*(-I(c[3][0])) beta_imag*R(c[3][0])
                SCALE_BETA_IMAG_M_LOOP(ymm0,ymm1,ymm3,ymm4,ymm15,ymm2);
            }
            /*
            The scaling has been done sequentially as follows:
            - If alpha_real is not 0, it is used for scaling A
            - If alpha_imag is not 0, it is used for scaling A using permutation
              and selective negation, after loading
            - If beta_real is not 0, is is used for scaling C
            - If beta_imag is not 0, it is used for scaling C using permutation
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

        // Fringe cases for M
        dim_t m_rem=m_remainder;
        if(m_rem>=2)
        {
            ymm3 = _mm256_setzero_pd();


            //R(a[0][0]) I(a[0][0]) R(a[1][0]) I(a[1][0])
            ymm0 = _mm256_loadu_pd((double const *)(temp_a));

            ymm13 = ymm0;
            SCALE_ALPHA_REAL_M_FRINGE(ymm0,ymm15,alpha_real);
            SCALE_ALPHA_IMAG_M_FRINGE(ymm0,ymm13,ymm15,ymm2,alpha_imag);

            /*
            The result after scaling with alpha_real and/or alpha_imag is as follows:
            For ymm0 :
            R(a[0][0]) = alpha_real*R(a[0][0])-alpha_imag*I(a[0][0])
            I(a[0][0]) = alpha_real*I(a[0][0])+alpha_imag*R[0][0]
            R(a[1][0]) = alpha_real*R(a[1][0])-alpha_imag*I(a[1][0])
            I(a[1][0]) = alpha_real*I(a[1][0])+alpha_imag*(R[1][0])
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


            if(beta_real != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_real));

                ymm0 = _mm256_loadu_pd((double const *)(temp_c));
                //ymm3+=beta_real*R(c[0][0]) beta_real*I(c[0][0])
                //      beta_real*R(c[1][0]) beta_real*I(c[1][0])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm3,ymm15);
            }

            if(beta_imag != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_imag));

                ymm0 = _mm256_loadu_pd((double const *)(temp_c));
                //ymm3+=beta_imag*(-I(c[0][0])) beta_imag*R(c[0][0])
                //      beta_imag*(-I(c[1][0])) beta_imag*R(c[1][0])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm3,ymm15,ymm2);
            }
            /*
            The scaling has been done sequentially as follows:
            - If alpha_real is not 0, it is used for scaling A
            - If alpha_imag is not 0, it is used for scaling A using permutation
              and selective negation, after loading
            - If beta_real is not 0, is is used for scaling C
            - If beta_imag is not 0, it is used for scaling C using permutation
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

            xmm5 = _mm_loadu_pd((double const*)(temp_a));
            ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);

            ymm13 = ymm0;
            SCALE_ALPHA_REAL_M_FRINGE(ymm0,ymm15,alpha_real);
            SCALE_ALPHA_IMAG_M_FRINGE(ymm0,ymm13,ymm15,ymm2,alpha_imag);

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

            if(beta_real != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_real));

                xmm5 = _mm_loadu_pd((double const*)(temp_c));
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);
                //ymm3+=beta_real*R(c[0][0]) beta_real*I(c[0][0])
                SCALE_BETA_REAL_M_FRINGE(ymm0,ymm3,ymm15);
            }

            if(beta_imag != 0.0)
            {
                ymm15 = _mm256_broadcast_sd((double const *)(&beta_imag));

                xmm5 = _mm_loadu_pd((double const*)(temp_c));
                ymm0 = _mm256_insertf128_pd(ymm0,xmm5,0);
                //ymm3+=beta_imag*(-I(c[0][0])) beta_imag*R(c[0][0])
                SCALE_BETA_IMAG_M_FRINGE(ymm0,ymm3,ymm15,ymm2);
            }

            xmm5 = _mm256_extractf128_pd(ymm3, 0);
            _mm_storeu_pd((double *)(temp_c), xmm5);

        }

    }

}