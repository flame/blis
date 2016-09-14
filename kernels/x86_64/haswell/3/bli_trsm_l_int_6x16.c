/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2016, Advanced Micro Devices, Inc

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
#include <immintrin.h>

typedef union
{
    __m256 v;
    float f[8];
}v8ff_t;

void bli_strsm_l_int_6x16 (
                            float*     restrict a11,
                            float*     restrict b11,
                            float*     restrict c11, inc_t rs_c, inc_t cs_c,
                            auxinfo_t* restrict data,
                            cntx_t*    restrict cntx
    )
{

    v8ff_t ymm8;
    v8ff_t ymm12;
    v8ff_t ymm9;
    v8ff_t ymm13;
    v8ff_t ymm10;
    v8ff_t ymm14;
    v8ff_t ymm11;
    v8ff_t ymm15;
    v8ff_t ymm0;
    v8ff_t ymm1;
    v8ff_t ymm2;
    v8ff_t ymm3;
    v8ff_t ymm4;
    v8ff_t ymm5;
    v8ff_t ymm6;
    v8ff_t ymm7;

    v8ff_t ymm16;
    v8ff_t ymm17;
    v8ff_t ymm18;
    v8ff_t ymm19;

    // b11 - row major
    // a11 - column major
    // c11 - rs_c & cs_c

    // a11 = The alpha's stored in a11 are 1/alphas. When we refer to alphas or 1/alphas they are simply
    // elements of a11.

    // Load all the rows of b11
    ymm8.v  = _mm256_load_ps(b11 + 0 * 8);     // ymm8  = beta00 ... beta07
    ymm12.v = _mm256_load_ps(b11 + 1 * 8);     // ymm12 = beta08 ... beta15
    ymm9.v  = _mm256_load_ps(b11 + 2 * 8);     // ymm9  = beta10 ... beta17
    ymm13.v = _mm256_load_ps(b11 + 3 * 8);     // ymm13 = beta18 ... beta1,15
    ymm10.v = _mm256_load_ps(b11 + 4 * 8);     // ymm10 = beta20 ... beta27
    ymm14.v = _mm256_load_ps(b11 + 5 * 8);     // ymm14 = beta28 ... beta2,15
    ymm11.v = _mm256_load_ps(b11 + 6 * 8);     // ymm11 = beta30 ... beta37
    ymm15.v = _mm256_load_ps(b11 + 7 * 8);     // ymm15 = beta38 ... beta3,15

    ymm16.v  = _mm256_load_ps(b11 + 8 * 8);     // ymm6 = beta40 ... beta47
    ymm17.v = _mm256_load_ps(b11 + 9 * 8);      // ymm7 = beta48 ... beta4,15
    ymm18.v = _mm256_load_ps(b11 + 10 * 8);     //  ymm4 = beta50 ... beta57
    ymm19.v = _mm256_load_ps(b11 + 11 * 8);     //  ymm5 = beta58 ... beta5,15

    // iteration 0
    ymm0.v  = _mm256_broadcast_ss(a11 + 0 );     // load ymm0 = (1/alpha00)
    ymm8.v  = _mm256_mul_ps(ymm0.v, ymm8.v);    // ymm8  *= (1/alpha00);
    ymm12.v = _mm256_mul_ps(ymm0.v, ymm12.v);   // ymm12 *= (1/alpha00);
    _mm256_store_ps(b11 + 0 * 8, ymm8.v);   // store ( beta00 ... beta07 ) = ymm8
    _mm256_store_ps(b11 + 1 * 8, ymm12.v);  // store ( beta08 ... beta15 ) = ymm12
    // Store in C11 
    for (int i = 0; i < 8; i++)
    {
        c11[0 * rs_c + i*cs_c] = ymm8.f[i];           // store (gama00 ... gama07) = ymm8[0] ... ymm8[7])
        c11[0 * rs_c + (i + 8) * cs_c] = ymm12.f[i];  // store (gama08 ... gama015) = ymm12[0] ... ymm12[7])
    }

    // iteration 1
    ymm0.v = _mm256_broadcast_ss(a11 + 1);          // ymm0 = (1/alpha10)
    ymm1.v = _mm256_broadcast_ss(a11 + 1 * 6 + 1);  // ymm1 = (1/alpha11)
    ymm4.v = _mm256_loadu_ps(ymm0.f);               // ymm4 = ymm0 = (1/alpha10)
    
    ymm0.v = _mm256_mul_ps(ymm0.v, ymm8.v);        // ymm0 = alpha10 * alpha00* (beta00 .. beta07)
    ymm4.v = _mm256_mul_ps(ymm4.v, ymm12.v);       // ymm4 = alpha10 * alpha00 * (beta08 ... beta015)

    ymm9.v  = _mm256_sub_ps(ymm9.v, ymm0.v);       // ymm9 =  [beta10 ... beta17] - ymm0
    ymm13.v = _mm256_sub_ps(ymm13.v, ymm4.v);      // ymm13 = [beta18 ... beta1,15] - ymm4

    ymm9.v  = _mm256_mul_ps(ymm9.v, ymm1.v);       // ymm9 = 1/alpha11 * ymm9
    ymm13.v = _mm256_mul_ps(ymm13.v, ymm1.v);      // ymm13 = 1/alpha11 * ymm13

    _mm256_store_ps(b11 + 2 * 8, ymm9.v);   // store ( beta10 ... beta17 )  = ymm9
    _mm256_store_ps(b11 + 3 * 8, ymm13.v);  // store ( beta18 ... beta1,15 ) = ymm13

    // Store @ C11 
    for (int i = 0; i < 8; i++)
    {
        c11[1 * rs_c + i*cs_c] = ymm9.f[i];           // store (gama10 ... gama17) = ymm9[0] ... ymm9[7])
        c11[1 * rs_c + (i + 8) * cs_c] = ymm13.f[i];  // store (gama18 ... gama115) = ymm13[0] ... ymm13[7])
    }

    // iteration 2
    ymm0.v = _mm256_broadcast_ss(a11 + 2);  // ymm0 = 1/alpha20
    ymm1.v = _mm256_broadcast_ss(a11 + 1 * 6 + 2); // ymm1 = 1/alpha21
    ymm2.v = _mm256_broadcast_ss(a11 + 2 * 6 + 2); // ymm2 = 1/alpha22

    ymm4.v = _mm256_loadu_ps(ymm0.f);               // ymm4 = ymm0 = 1/alpha20
    ymm5.v = _mm256_loadu_ps(ymm1.f);               // ymm5 = ymm1 = 1/alpha21

    ymm0.v = _mm256_mul_ps(ymm0.v, ymm8.v);        // ymm0 = alpha20 * [alpha00* (beta00 .. beta07)]
    ymm4.v = _mm256_mul_ps(ymm4.v, ymm12.v);       // ymm4 = alpha20 * [alpha00 * (beta08 ... beta015)]
    ymm1.v = _mm256_mul_ps(ymm9.v, ymm1.v);        // ymm1 = alpha21 * ymm9
    ymm5.v = _mm256_mul_ps(ymm13.v, ymm5.v);       // ymm5 = alpha21 * ymm13

    ymm0.v = _mm256_add_ps(ymm1.v, ymm0.v);       // ymm0 += ymm1
    ymm4.v = _mm256_add_ps(ymm5.v, ymm4.v);       // ymm4 += ymm5

    ymm10.v = _mm256_sub_ps(ymm10.v, ymm0.v);    // ymm10 -= ymm0
    ymm14.v = _mm256_sub_ps(ymm14.v, ymm4.v);    // ymm14 -= ymm4

    ymm10.v = _mm256_mul_ps(ymm10.v, ymm2.v);    // ymm10 *= 1/alpha22
    ymm14.v = _mm256_mul_ps(ymm14.v, ymm2.v);    // ymm14 *= 1/alpha22

    // Store the result
    _mm256_store_ps(b11 + 4 * 8, ymm10.v);   // store ( beta20 ... beta27 )  = ymm10
    _mm256_store_ps(b11 + 5 * 8, ymm14.v);  // store ( beta28 ... beta2,15 ) = ymm14

    // Store @ C11 
    for (int i = 0; i < 8; i++)
    {
        c11[2 * rs_c + i*cs_c] = ymm10.f[i];           // store (gama20 ... gama27) = ymm10[0] ... ymm10[7])
        c11[2 * rs_c + (i + 8) * cs_c] = ymm14.f[i];  // store (gama28 ... gama2,15) = ymm14[0] ... ymm14[7])
    }

    // iteration 3
    ymm0.v = _mm256_broadcast_ss(a11 + 3);  // ymm0 = 1/alpha30
    ymm1.v = _mm256_broadcast_ss(a11 + 1 * 6 + 3); // ymm1 = 1/alpha31
    ymm2.v = _mm256_broadcast_ss(a11 + 2 * 6 + 3); // ymm2 = 1/alpha32
    ymm3.v = _mm256_broadcast_ss(a11 + 3 * 6 + 3); // ymm3 = 1/alpha33

    ymm4.v = _mm256_loadu_ps(ymm0.f);               // ymm4 = ymm0 = 1/alpha30
    ymm5.v = _mm256_loadu_ps(ymm1.f);               // ymm5 = ymm1 = 1/alpha31
    ymm6.v = _mm256_loadu_ps(ymm2.f);               // ymm6 = ymm2 = 1/alpha32

    ymm0.v = _mm256_mul_ps(ymm8.v, ymm0.v);       // ymm0 = alpha30 * ymm8
    ymm4.v = _mm256_mul_ps(ymm12.v, ymm4.v);      // ymm4 = alpha30 * ymm12
    ymm1.v = _mm256_mul_ps(ymm9.v, ymm1.v);       // ymm1 = alpha31 * ymm9
    ymm5.v = _mm256_mul_ps(ymm13.v, ymm5.v);      // ymm5 = alpha31 * ymm13
    ymm2.v = _mm256_mul_ps(ymm10.v, ymm2.v);      // ymm2 = alpha32 * ymm10
    ymm6.v = _mm256_mul_ps(ymm14.v, ymm6.v);      // ymm6 = alpha32 * ymm14

    ymm0.v = _mm256_add_ps(ymm0.v, ymm1.v);      // ymm0 += ymm1
    ymm4.v = _mm256_add_ps(ymm5.v, ymm4.v);      // ymm4 += ymm5
    ymm0.v = _mm256_add_ps(ymm2.v, ymm0.v);      // ymm0 += ymm2
    ymm4.v = _mm256_add_ps(ymm6.v, ymm4.v);      // ymm4 += ymm6

    ymm11.v = _mm256_sub_ps(ymm11.v, ymm0.v);    // ymm11 -= ymm0 {[beta30 ... beta37] - ymm0}
    ymm15.v = _mm256_sub_ps(ymm15.v, ymm4.v);    // ymm15 -= ymm4 {[beta38 ... beta3,15] - ymm4}

    ymm11.v = _mm256_mul_ps(ymm11.v, ymm3.v);    // ymm11 *= alpha33
    ymm15.v = _mm256_mul_ps(ymm15.v, ymm3.v);    // ymm15 *= alpha33

    // Store the result
    _mm256_store_ps(b11 + 6 * 8, ymm11.v);   // store ( beta30 ... beta37 )  = ymm11
    _mm256_store_ps(b11 + 7 * 8, ymm15.v);  // store ( beta38 ... beta3,15 ) = ymm15

    // Store @ C11 
    for (int i = 0; i < 8; i++)
    {
        c11[3 * rs_c + i*cs_c] = ymm11.f[i];           // store (gama30 ... gama37) = ymm11[0] ... ymm11[7])
        c11[3 * rs_c + (i + 8) * cs_c] = ymm15.f[i];  // store (gama38 ... gama3,15) = ymm15[0] ... ymm15[7])
    }

        // iteration 4
    v8ff_t ymm21;
    ymm0.v = _mm256_broadcast_ss(a11 + 4);          // ymm0 = 1/alpha40
    ymm1.v = _mm256_broadcast_ss(a11 + 1 * 6 + 4);  // ymm1 = 1/alpha41
    ymm2.v = _mm256_broadcast_ss(a11 + 2 * 6 + 4);  // ymm2 = 1/alpha42
    ymm3.v = _mm256_broadcast_ss(a11 + 3 * 6 + 4);  // ymm3 = 1/alpha43
    ymm4.v = _mm256_broadcast_ss(a11 + 4 * 6 + 4);  // ymm4 = 1/alpha44

    ymm5.v = _mm256_loadu_ps(ymm0.f);   // ymm5 = ymm0 = alpha40
    ymm6.v = _mm256_loadu_ps(ymm1.f);   // ymm6 = ymm1 = alpha41
    ymm7.v = _mm256_loadu_ps(ymm2.f);   // ymm7 = ymm2 = alpha42
    ymm21.v = _mm256_loadu_ps(ymm3.f);  // ymm21 = ymm3 = alpha43

    ymm0.v = _mm256_mul_ps(ymm8.v, ymm0.v); // alpha40 * ymm8
    ymm5.v = _mm256_mul_ps(ymm12.v, ymm5.v); // alpha40 * ymm12

    ymm1.v = _mm256_mul_ps(ymm9.v, ymm1.v); //  ymm1 = alpha41 * ymm9
    ymm6.v = _mm256_mul_ps(ymm13.v, ymm6.v); // ymm6 = alpha41 * ymm13

    ymm2.v = _mm256_mul_ps(ymm10.v, ymm2.v); // ymm2 = alpha42 * ymm10
    ymm7.v = _mm256_mul_ps(ymm14.v, ymm7.v); // ymm7 = alpha42 * ymm14

    ymm3.v = _mm256_mul_ps(ymm11.v, ymm3.v); // ymm3 = alpha43 * ymm11
    ymm21.v = _mm256_mul_ps(ymm15.v, ymm21.v); // ymm21 = alpha43 * ymm21

    ymm0.v = _mm256_add_ps(ymm0.v, ymm1.v); // ymm0 += ymm1
    ymm5.v = _mm256_add_ps(ymm5.v, ymm6.v); // ymm5 += ymm6

    ymm2.v = _mm256_add_ps(ymm2.v, ymm3.v);  // ymm2 += ymm3
    ymm7.v = _mm256_add_ps(ymm7.v, ymm21.v); // ymm7 += ymm21

    ymm0.v = _mm256_add_ps(ymm0.v, ymm2.v); // ymm0 += ymm2 (ymm0 = ymm0 + ymm1 + ymm2 + ymm3)
    ymm5.v = _mm256_add_ps(ymm5.v, ymm7.v); // ymm5 += ymm7 (ymm5 = ymm5 + ymm6 + ymm7 + ymm21)

    ymm16.v = _mm256_sub_ps(ymm16.v, ymm0.v); // ymm16 -= ymm0 {[beta40 ... beta47] -ymm0}
    ymm17.v = _mm256_sub_ps(ymm17.v, ymm5.v); // ymm17 -= ymm5 {[beta48 ... beta4,15] - ymm17}

    ymm16.v = _mm256_mul_ps(ymm16.v, ymm4.v); // ymm16 *= alpha44
    ymm17.v = _mm256_mul_ps(ymm17.v, ymm4.v); // ymm17 *= alpha44

    // Store the result
    _mm256_store_ps(b11 + 8 * 8, ymm16.v);   // store ( beta40 ... beta47 )  = ymm16
    _mm256_store_ps(b11 + 9 * 8, ymm17.v);  // store ( beta48 ... beta4,15 ) = ymm17

    // Store @ C11 
    for (int i = 0; i < 8; i++)
    {
        c11[4 * rs_c + i*cs_c] = ymm16.f[i];           // store (gama40 ... gama47) = ymm16[0] ... ymm16[7])
        c11[4 * rs_c + (i + 8) * cs_c] = ymm17.f[i];  // store (gama48 ... gama4,15) = ymm17[0] ... ymm17[7])
    }

    // iteration 5
    v8ff_t ymm31;
    v8ff_t ymm41;

    ymm0.v = _mm256_broadcast_ss(a11 + 5);         // ymm0 = 1/alpha50
    ymm1.v = _mm256_broadcast_ss(a11 + 1 * 6 + 5); // ymm1 = 1/alpha51
    ymm2.v = _mm256_broadcast_ss(a11 + 2 * 6 + 5); // ymm2 = 1/alpha52
    ymm3.v = _mm256_broadcast_ss(a11 + 3 * 6 + 5); // ymm3 = 1/alpha53
    ymm4.v = _mm256_broadcast_ss(a11 + 4 * 6 + 5); // ymm4 = 1/alpha54
    ymm5.v = _mm256_broadcast_ss(a11 + 5 * 6 + 5); // ymm5 = 1/alpha55

    ymm6.v  = _mm256_loadu_ps(ymm0.f);   // ymm6 = ymm0 = 1/alpha50
    ymm7.v  = _mm256_loadu_ps(ymm1.f);   // ymm7 = ymm1 = 1/alpha51
    ymm21.v = _mm256_loadu_ps(ymm2.f);   // ymm21 = ymm2 = 1/alpha52
    ymm31.v = _mm256_loadu_ps(ymm3.f);   // ymm31 = ymm3 = 1/alpha53
    ymm41.v = _mm256_loadu_ps(ymm4.f);   // ymm41 = ymm4 = 1/alpha54

    ymm0.v = _mm256_mul_ps(ymm8.v, ymm0.v);  // ymm0 = alpha50 * ymm8
    ymm6.v = _mm256_mul_ps(ymm12.v, ymm6.v); // ymm6 = alpha50 * ymm12

    ymm1.v = _mm256_mul_ps(ymm9.v, ymm1.v);  // ymm1 = alpha51 * ymm9
    ymm7.v = _mm256_mul_ps(ymm13.v, ymm7.v); // ymm7 = alpha51 * ymm13

    ymm2.v = _mm256_mul_ps(ymm10.v, ymm2.v); // ymm2 = alpha52 * ymm10
    ymm21.v = _mm256_mul_ps(ymm14.v, ymm21.v); // ymm21 = alpha52 * ymm14

    ymm3.v = _mm256_mul_ps(ymm11.v, ymm3.v); // ymm3 = alpha53 * ymm11
    ymm31.v = _mm256_mul_ps(ymm31.v, ymm15.v); // ymm31 = alpha53 * ymm15

    ymm4.v = _mm256_mul_ps(ymm4.v, ymm16.v);  // ymm4 = alpha54 * ymm16
    ymm41.v = _mm256_mul_ps(ymm41.v, ymm17.v); // ymm41 = alpha54 * ymm17

    ymm0.v = _mm256_add_ps(ymm0.v, ymm1.v);   // ymm0 += ymm1
    ymm6.v = _mm256_add_ps(ymm6.v, ymm7.v);   // ymm6 += ymm7
    ymm2.v = _mm256_add_ps(ymm2.v, ymm3.v);   // ymm2 += ymm3
    ymm21.v = _mm256_add_ps(ymm21.v, ymm31.v); // ymm21 += ymm31
    ymm0.v = _mm256_add_ps(ymm0.v, ymm2.v);   // ymm0 += ymm2 {ymm0 = ymm0 + ymm1 + ymm2 + ymm3}
    ymm6.v = _mm256_add_ps(ymm6.v, ymm21.v);  // ymm6 += ymm21 {ymm6 = ymm6 + ymm7 + ymm21 + ymm31}
    ymm0.v = _mm256_add_ps(ymm0.v, ymm4.v);   // ymm0 += ymm4 {ymm0 += ymm1 + ymm2 + ymm3 + ymm4}
    ymm6.v = _mm256_add_ps(ymm6.v, ymm41.v);  // ymm6 += ymm41 { ymm6 += ymm7 + ymm21 + ymm31 + ymm41}

    ymm18.v = _mm256_sub_ps(ymm18.v, ymm0.v);  // {[beta50 ... beta57] - ymm0}  ymm18 -= ymm0
    ymm19.v = _mm256_sub_ps(ymm19.v, ymm6.v); // {[beta58 ... beta5,15] - ymm6}: ymm19 -= ymm6

    ymm18.v = _mm256_mul_ps(ymm5.v, ymm18.v);  // alpha55 * ymm18
    ymm19.v = _mm256_mul_ps(ymm5.v, ymm19.v);  // alpha55 * ymm19

    // Store the result
    _mm256_store_ps(b11 + 10 * 8, ymm18.v);   // store ( beta50 ... beta57 )  = ymm18
    _mm256_store_ps(b11 + 11 * 8, ymm19.v);  // store ( beta58 ... beta5,15 ) = ymm19

    // Store @ C11 
    for (int i = 0; i < 8; i++)
    {
        c11[5 * rs_c + i*cs_c] = ymm18.f[i];           // store (gama50 ... gama57) = ymm18[0] ... ymm18[7])
        c11[5 * rs_c + (i + 8) * cs_c] = ymm19.f[i];  // store (gama58 ... gama5,15) = ymm19[0] ... ymm19[7])
    }
}// End of the function
