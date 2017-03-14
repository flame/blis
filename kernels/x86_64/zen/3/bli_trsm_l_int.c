/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************ */
#include "blis.h"
#include <immintrin.h>

typedef union
{
  __m256 v;
  float f[8];
}v8ff_t;

typedef union
{
  __m256d v;
  double d[4];
}v4df_t;

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
  ymm8.v  = _mm256_loadu_ps(b11 + 0 * 8);     // ymm8  = beta00 ... beta07
  ymm12.v = _mm256_loadu_ps(b11 + 1 * 8);     // ymm12 = beta08 ... beta15
  ymm9.v  = _mm256_loadu_ps(b11 + 2 * 8);     // ymm9  = beta10 ... beta17
  ymm13.v = _mm256_loadu_ps(b11 + 3 * 8);     // ymm13 = beta18 ... beta1,15
  ymm10.v = _mm256_loadu_ps(b11 + 4 * 8);     // ymm10 = beta20 ... beta27
  ymm14.v = _mm256_loadu_ps(b11 + 5 * 8);     // ymm14 = beta28 ... beta2,15
  ymm11.v = _mm256_loadu_ps(b11 + 6 * 8);     // ymm11 = beta30 ... beta37
  ymm15.v = _mm256_loadu_ps(b11 + 7 * 8);     // ymm15 = beta38 ... beta3,15

  ymm16.v  = _mm256_loadu_ps(b11 + 8 * 8);     // ymm16 = beta40 ... beta47
  ymm17.v  = _mm256_loadu_ps(b11 + 9 * 8);     // ymm17 = beta48 ... beta4,15
  ymm18.v  = _mm256_loadu_ps(b11 + 10 * 8);    // ymm18 = beta50 ... beta57
  ymm19.v  = _mm256_loadu_ps(b11 + 11 * 8);    // ymm19 = beta58 ... beta5,15

  // iteration 0
  ymm0.v  = _mm256_broadcast_ss(a11 + 0 );    // load ymm0 = (1/alpha00)
  ymm8.v  = _mm256_mul_ps(ymm0.v, ymm8.v);    // ymm8  *= (1/alpha00);
  ymm12.v = _mm256_mul_ps(ymm0.v, ymm12.v);   // ymm12 *= (1/alpha00);
  _mm256_storeu_ps(b11 + 0 * 8, ymm8.v);      // store ( beta00 ... beta07 ) = ymm8
  _mm256_storeu_ps(b11 + 1 * 8, ymm12.v);  // store ( beta08 ... beta15 ) = ymm12
  // Store in C11 
  for (int i = 0; i < 8; i++)
    {
      c11[0 * rs_c + i*cs_c] = ymm8.f[i];           // store (gama00 ... gama07) = ymm8[0] ... ymm8[7])
      c11[0 * rs_c + (i + 8) * cs_c] = ymm12.f[i];  // store (gama08 ... gama015) = ymm12[0] ... ymm12[7])
    }

  // iteration 1
  ymm0.v = _mm256_broadcast_ss(a11 + 1);          // ymm0 = alpha10
  ymm1.v = _mm256_broadcast_ss(a11 + 1 * 6 + 1);  // ymm1 = (1/alpha11)
  ymm4.v = _mm256_loadu_ps(ymm0.f);               // ymm4 = ymm0 = alpha10
    
  ymm0.v = _mm256_mul_ps(ymm0.v, ymm8.v);        // ymm0 = alpha10 * alpha00* (beta00 .. beta07)
  ymm4.v = _mm256_mul_ps(ymm4.v, ymm12.v);       // ymm4 = alpha10 * alpha00 * (beta08 ... beta015)

  ymm9.v  = _mm256_sub_ps(ymm9.v, ymm0.v);       // ymm9 =  [beta10 ... beta17] - ymm0
  ymm13.v = _mm256_sub_ps(ymm13.v, ymm4.v);      // ymm13 = [beta18 ... beta1,15] - ymm4

  ymm9.v  = _mm256_mul_ps(ymm9.v, ymm1.v);       // ymm9 = 1/alpha11 * ymm9
  ymm13.v = _mm256_mul_ps(ymm13.v, ymm1.v);      // ymm13 = 1/alpha11 * ymm13

  _mm256_storeu_ps(b11 + 2 * 8, ymm9.v);   // store ( beta10 ... beta17 )  = ymm9
  _mm256_storeu_ps(b11 + 3 * 8, ymm13.v);  // store ( beta18 ... beta1,15 ) = ymm13

  // Store @ C11 
  for (int i = 0; i < 8; i++)
    {
      c11[1 * rs_c + i*cs_c] = ymm9.f[i];           // store (gama10 ... gama17) = ymm9[0] ... ymm9[7])
      c11[1 * rs_c + (i + 8) * cs_c] = ymm13.f[i];  // store (gama18 ... gama115) = ymm13[0] ... ymm13[7])
    }

  // iteration 2
  ymm0.v = _mm256_broadcast_ss(a11 + 2);  // ymm0 = alpha20
  ymm1.v = _mm256_broadcast_ss(a11 + 1 * 6 + 2); // ymm1 = alpha21
  ymm2.v = _mm256_broadcast_ss(a11 + 2 * 6 + 2); // ymm2 = 1/alpha22

  ymm4.v = _mm256_loadu_ps(ymm0.f);               // ymm4 = ymm0 = alpha20
  ymm5.v = _mm256_loadu_ps(ymm1.f);               // ymm5 = ymm1 = alpha21

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
  _mm256_storeu_ps(b11 + 4 * 8, ymm10.v);   // store ( beta20 ... beta27 )  = ymm10
  _mm256_storeu_ps(b11 + 5 * 8, ymm14.v);  // store ( beta28 ... beta2,15 ) = ymm14

  // Store @ C11 
  for (int i = 0; i < 8; i++)
    {
      c11[2 * rs_c + i*cs_c] = ymm10.f[i];           // store (gama20 ... gama27) = ymm10[0] ... ymm10[7])
      c11[2 * rs_c + (i + 8) * cs_c] = ymm14.f[i];  // store (gama28 ... gama2,15) = ymm14[0] ... ymm14[7])
    }

  // iteration 3
  ymm0.v = _mm256_broadcast_ss(a11 + 3);         // ymm0 = alpha30
  ymm1.v = _mm256_broadcast_ss(a11 + 1 * 6 + 3); // ymm1 = alpha31
  ymm2.v = _mm256_broadcast_ss(a11 + 2 * 6 + 3); // ymm2 = alpha32
  ymm3.v = _mm256_broadcast_ss(a11 + 3 * 6 + 3); // ymm3 = 1/alpha33

  ymm4.v = _mm256_loadu_ps(ymm0.f);               // ymm4 = ymm0 = alpha30
  ymm5.v = _mm256_loadu_ps(ymm1.f);               // ymm5 = ymm1 = alpha31
  ymm6.v = _mm256_loadu_ps(ymm2.f);               // ymm6 = ymm2 = alpha32

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

  ymm11.v = _mm256_mul_ps(ymm11.v, ymm3.v);    // ymm11 *= 1/alpha33
  ymm15.v = _mm256_mul_ps(ymm15.v, ymm3.v);    // ymm15 *= 1/alpha33

  // Store the result
  _mm256_storeu_ps(b11 + 6 * 8, ymm11.v);   // store ( beta30 ... beta37 )  = ymm11
  _mm256_storeu_ps(b11 + 7 * 8, ymm15.v);  // store ( beta38 ... beta3,15 ) = ymm15

  // Store @ C11 
  for (int i = 0; i < 8; i++)
    {
      c11[3 * rs_c + i*cs_c] = ymm11.f[i];           // store (gama30 ... gama37) = ymm11[0] ... ymm11[7])
      c11[3 * rs_c + (i + 8) * cs_c] = ymm15.f[i];  // store (gama38 ... gama3,15) = ymm15[0] ... ymm15[7])
    }

  // iteration 4
  v8ff_t ymm21;
  ymm0.v = _mm256_broadcast_ss(a11 + 4);          // ymm0 = alpha40
  ymm1.v = _mm256_broadcast_ss(a11 + 1 * 6 + 4);  // ymm1 = alpha41
  ymm2.v = _mm256_broadcast_ss(a11 + 2 * 6 + 4);  // ymm2 = alpha42
  ymm3.v = _mm256_broadcast_ss(a11 + 3 * 6 + 4);  // ymm3 = alpha43
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

  ymm16.v = _mm256_mul_ps(ymm16.v, ymm4.v); // ymm16 *= 1/alpha44
  ymm17.v = _mm256_mul_ps(ymm17.v, ymm4.v); // ymm17 *= 1/alpha44

  // Store the result
  _mm256_storeu_ps(b11 + 8 * 8, ymm16.v);   // store ( beta40 ... beta47 )  = ymm16
  _mm256_storeu_ps(b11 + 9 * 8, ymm17.v);  // store ( beta48 ... beta4,15 ) = ymm17

  // Store @ C11 
  for (int i = 0; i < 8; i++)
    {
      c11[4 * rs_c + i*cs_c] = ymm16.f[i];           // store (gama40 ... gama47) = ymm16[0] ... ymm16[7])
      c11[4 * rs_c + (i + 8) * cs_c] = ymm17.f[i];  // store (gama48 ... gama4,15) = ymm17[0] ... ymm17[7])
    }

  // iteration 5
  v8ff_t ymm31;
  v8ff_t ymm41;

  ymm0.v = _mm256_broadcast_ss(a11 + 5);         // ymm0 = alpha50
  ymm1.v = _mm256_broadcast_ss(a11 + 1 * 6 + 5); // ymm1 = alpha51
  ymm2.v = _mm256_broadcast_ss(a11 + 2 * 6 + 5); // ymm2 = alpha52
  ymm3.v = _mm256_broadcast_ss(a11 + 3 * 6 + 5); // ymm3 = alpha53
  ymm4.v = _mm256_broadcast_ss(a11 + 4 * 6 + 5); // ymm4 = alpha54
  ymm5.v = _mm256_broadcast_ss(a11 + 5 * 6 + 5); // ymm5 = 1/alpha55

  ymm6.v  = _mm256_loadu_ps(ymm0.f);   // ymm6 = ymm0 = alpha50
  ymm7.v  = _mm256_loadu_ps(ymm1.f);   // ymm7 = ymm1 = alpha51
  ymm21.v = _mm256_loadu_ps(ymm2.f);   // ymm21 = ymm2 = alpha52
  ymm31.v = _mm256_loadu_ps(ymm3.f);   // ymm31 = ymm3 = alpha53
  ymm41.v = _mm256_loadu_ps(ymm4.f);   // ymm41 = ymm4 = alpha54

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

  ymm18.v = _mm256_mul_ps(ymm5.v, ymm18.v);  // 1/alpha55 * ymm18
  ymm19.v = _mm256_mul_ps(ymm5.v, ymm19.v);  // 1/alpha55 * ymm19

  // Store the result
  _mm256_storeu_ps(b11 + 10 * 8, ymm18.v);   // store ( beta50 ... beta57 )  = ymm18
  _mm256_storeu_ps(b11 + 11 * 8, ymm19.v);  // store ( beta58 ... beta5,15 ) = ymm19

  // Store @ C11 
  for (int i = 0; i < 8; i++)
    {
      c11[5 * rs_c + i*cs_c] = ymm18.f[i];           // store (gama50 ... gama57) = ymm18[0] ... ymm18[7])
      c11[5 * rs_c + (i + 8) * cs_c] = ymm19.f[i];  // store (gama58 ... gama5,15) = ymm19[0] ... ymm19[7])
    }
}// End of the function

void bli_dtrsm_l_int_6x8(
               double*     restrict a11,
               double*     restrict b11,
               double*     restrict c11, inc_t rs_c, inc_t cs_c,
               auxinfo_t* restrict data,
               cntx_t*    restrict cntx
             )
{
  v4df_t ymm8a;
  v4df_t ymm8b;


  v4df_t ymm9a;
  v4df_t ymm9b;


  v4df_t ymm10a;
  v4df_t ymm10b;


  v4df_t ymm11a;
  v4df_t ymm11b;


  v4df_t ymm0a;
  v4df_t ymm0b;
  v4df_t ymm1a;
  v4df_t ymm1b;
  v4df_t ymm2a;
  v4df_t ymm2b;
  v4df_t ymm3a;
  v4df_t ymm3b;
  v4df_t ymm4a;
  v4df_t ymm4b;
  v4df_t ymm5a;






  v4df_t ymm16a;
  v4df_t ymm16b;


  v4df_t ymm18a;
  v4df_t ymm18b;



  // b11 - row major
  // a11 - column major
  // c11 - rs_c & cs_c

  // a11 = The diagonal elements of a11 are stored in 1/alpha's to avoid costly division operation in TRSV.

  // Load all the rows of b11
  ymm8a.v  = _mm256_loadu_pd(b11 + 0 * 4);     // ymm8a  = beta00 ... beta03
  ymm8b.v  = _mm256_loadu_pd(b11 + 1 * 4);     // ymm8b  = beta04 ... beta07
  ymm9a.v  = _mm256_loadu_pd(b11 + 2 * 4);     // ymm9a  = beta10 ... beta13
  ymm9b.v  = _mm256_loadu_pd(b11 + 3 * 4);     // ymm9b  = beta14 ... beta17
  ymm10a.v = _mm256_loadu_pd(b11 + 4 * 4);     // ymm10a = beta20 ... beta23
  ymm10b.v = _mm256_loadu_pd(b11 + 5 * 4);     // ymm10b = beta24 ... beta27
  ymm11a.v = _mm256_loadu_pd(b11 + 6 * 4);     // ymm11a = beta30 ... beta33
  ymm11b.v = _mm256_loadu_pd(b11 + 7 * 4);     // ymm11b = beta34 ... beta37
  ymm16a.v = _mm256_loadu_pd(b11 + 8 * 4);     // ymm16a = beta40 ... beta43
  ymm16b.v = _mm256_loadu_pd(b11 + 9 * 4);     // ymm16b = beta44 ... beta47
  ymm18a.v = _mm256_loadu_pd(b11 + 10 * 4);     // ymm18a = beta50 ... beta53
  ymm18b.v = _mm256_loadu_pd(b11 + 11 * 4);     // ymm18b = beta54 ... beta57

  // iteration 0
  ymm0a.v  = _mm256_broadcast_sd(a11 + 0);     // load ymm0 = (1/alpha00)
  ymm8a.v  = _mm256_mul_pd(ymm0a.v, ymm8a.v);    // ymm8a  *= (1/alpha00);
  ymm8b.v  = _mm256_mul_pd(ymm0a.v, ymm8b.v);    // ymm8b  *= (1/alpha00);

  _mm256_storeu_pd(b11 + 0 * 4, ymm8a.v);   // store ( beta00 ... beta03 ) = ymm8a
  _mm256_storeu_pd(b11 + 1 * 4, ymm8b.v);   // store ( beta04 ... beta07 ) = ymm8b
  // Store in C11
  for (int i = 0; i < 4; i++)
    {
      c11[0 * rs_c + i*cs_c]          = ymm8a.d[i];   // store (gama00 ... gama03)     = ymm8[0] ... ymm8[3])
      c11[0 * rs_c + (i + 4)*cs_c]    = ymm8b.d[i];   // store (gama04 ... gama07)     = ymm8[4] ... ymm8[7])
    }

  // iteration 1
  ymm0a.v = _mm256_broadcast_sd(a11 + 1);          // ymm0 = alpha10)
  ymm0b.v = _mm256_broadcast_sd(a11 + 1);          // ymm0 = alpha10)
  ymm1a.v = _mm256_broadcast_sd(a11 + 1 * 6 + 1);  // ymm1 = (1/alpha11)

  ymm0a.v = _mm256_mul_pd(ymm0a.v, ymm8a.v);        // ymm0 = alpha10 * alpha00* (beta00 .. beta03)
  ymm0b.v = _mm256_mul_pd(ymm0b.v, ymm8b.v);        // ymm0 = alpha10 * alpha00* (beta04 .. beta07)

  ymm9a.v = _mm256_sub_pd(ymm9a.v, ymm0a.v);       // ymm9 =  [beta10 ... beta13] - ymm0
  ymm9b.v = _mm256_sub_pd(ymm9b.v, ymm0b.v);       // ymm9 =  [beta14 ... beta17] - ymm0

  ymm9a.v  = _mm256_mul_pd(ymm9a.v,  ymm1a.v);       // ymm9 = 1/alpha11 * ymm9
  ymm9b.v  = _mm256_mul_pd(ymm9b.v,  ymm1a.v);       // ymm9 = 1/alpha11 * ymm9

  _mm256_storeu_pd(b11 + 2 * 4, ymm9a.v);   // store ( beta10 ... beta17 )  = ymm9
  _mm256_storeu_pd(b11 + 3 * 4, ymm9b.v);   // store ( beta10 ... beta17 )  = ymm9

  // Store @ C11
  for (int i = 0; i < 4; i++)
    {
      c11[1 * rs_c + i*cs_c]          = ymm9a.d[i];  // store (gama10 ... gama17) = ymm9[0] ... ymm9[7])
      c11[1 * rs_c + (i + 4)*cs_c]    = ymm9b.d[i];  // store (gama10 ... gama17) = ymm9[0] ... ymm9[7])
    }

  // iteration 2
  ymm0a.v = _mm256_broadcast_sd(a11 + 2);  // ymm0 = alpha20
  ymm0b.v = _mm256_broadcast_sd(a11 + 2);  // ymm0 = alpha20
  ymm1a.v = _mm256_broadcast_sd(a11 + 1 * 6 + 2); // ymm1 = alpha21
  ymm1b.v = _mm256_broadcast_sd(a11 + 1 * 6 + 2); // ymm1 = alpha21

  ymm2a.v = _mm256_broadcast_sd(a11 + 2 * 6 + 2); // ymm2 = 1/alpha22


  ymm0a.v = _mm256_mul_pd(ymm0a.v, ymm8a.v);        // ymm0 = alpha20 * [alpha00* (beta00 .. beta03)]
  ymm0b.v = _mm256_mul_pd(ymm0b.v, ymm8b.v);        // ymm0 = alpha20 * [alpha00* (beta04 .. beta07)]

  ymm1a.v = _mm256_mul_pd(ymm9a.v, ymm1a.v);        // ymm1 = alpha21 * ymm9
  ymm1b.v = _mm256_mul_pd(ymm9b.v, ymm1b.v);        // ymm1 = alpha21 * ymm9


  ymm0a.v = _mm256_add_pd(ymm1a.v, ymm0a.v);       // ymm0 += ymm1
  ymm0b.v = _mm256_add_pd(ymm1b.v, ymm0b.v);       // ymm0 += ymm1

  ymm10a.v = _mm256_sub_pd(ymm10a.v, ymm0a.v);    // ymm10 -= ymm0
  ymm10b.v = _mm256_sub_pd(ymm10b.v, ymm0b.v);    // ymm10 -= ymm0

  ymm10a.v = _mm256_mul_pd(ymm10a.v, ymm2a.v);    // ymm10 *= 1/alpha22
  ymm10b.v = _mm256_mul_pd(ymm10b.v, ymm2a.v);    // ymm10 *= 1/alpha22

  // Store the result
  _mm256_storeu_pd(b11 + 4 * 4,  ymm10a.v);   // store ( beta20 ... beta23 )  = ymm10
  _mm256_storeu_pd(b11 + 5 * 4,  ymm10b.v);   // store ( beta24 ... beta27 )  = ymm10

  // Store @ C11
  for (int i = 0; i < 4; i++)
    {
      c11[2 * rs_c + i*cs_c]          = ymm10a.d[i];  // store (gama20 ... gama27) = ymm10[0] ... ymm10[7])
      c11[2 * rs_c + (i + 4)*cs_c]    = ymm10b.d[i];  // store (gama20 ... gama27) = ymm10[0] ... ymm10[7])
    }

  // iteration 3
  ymm0a.v = _mm256_broadcast_sd(a11 + 3);  // ymm0 = alpha30
  ymm0b.v = _mm256_broadcast_sd(a11 + 3);  // ymm0 = alpha30
  ymm1a.v = _mm256_broadcast_sd(a11 + 1 * 6 + 3); // ymm1 = alpha31
  ymm1b.v = _mm256_broadcast_sd(a11 + 1 * 6 + 3); // ymm1 = alpha31
  ymm2a.v = _mm256_broadcast_sd(a11 + 2 * 6 + 3); // ymm2 = alpha32
  ymm2b.v = _mm256_broadcast_sd(a11 + 2 * 6 + 3); // ymm2 = alpha32

  ymm3a.v = _mm256_broadcast_sd(a11 + 3 * 6 + 3); // ymm3 = 1/alpha33

  ymm0a.v = _mm256_mul_pd(ymm8a.v, ymm0a.v);       // ymm0 = alpha30 * ymm8
  ymm0b.v = _mm256_mul_pd(ymm8b.v, ymm0b.v);       // ymm0 = alpha30 * ymm8
  ymm1a.v = _mm256_mul_pd(ymm9a.v, ymm1a.v);       // ymm1 = alpha31 * ymm9
  ymm1b.v = _mm256_mul_pd(ymm9b.v, ymm1b.v);       // ymm1 = alpha31 * ymm9
  ymm2a.v = _mm256_mul_pd(ymm10a.v, ymm2a.v);      // ymm2 = alpha32 * ymm10
  ymm2b.v = _mm256_mul_pd(ymm10b.v, ymm2b.v);      // ymm2 = alpha32 * ymm10

  ymm0a.v = _mm256_add_pd(ymm0a.v, ymm1a.v);      // ymm0 += ymm1
  ymm0b.v = _mm256_add_pd(ymm0b.v, ymm1b.v);      // ymm0 += ymm1
  ymm0a.v = _mm256_add_pd(ymm2a.v, ymm0a.v);      // ymm0 += ymm2
  ymm0b.v = _mm256_add_pd(ymm2b.v, ymm0b.v);      // ymm0 += ymm2

  ymm11a.v = _mm256_sub_pd(ymm11a.v, ymm0a.v);    // ymm11 -= ymm0 {[beta30 ... beta37] - ymm0}
  ymm11b.v = _mm256_sub_pd(ymm11b.v, ymm0b.v);    // ymm11 -= ymm0 {[beta30 ... beta37] - ymm0}

  ymm11a.v = _mm256_mul_pd(ymm11a.v, ymm3a.v);    // ymm11 *= alpha33
  ymm11b.v = _mm256_mul_pd(ymm11b.v, ymm3a.v);    // ymm11 *= alpha33

  // Store the result
  _mm256_storeu_pd(b11 + 6 * 4, ymm11a.v);   // store ( beta30 ... beta37 )  = ymm11
  _mm256_storeu_pd(b11 + 7 * 4, ymm11b.v);   // store ( beta30 ... beta37 )  = ymm11

  // Store @ C11
  for (int i = 0; i < 4; i++)
    {
      c11[3 * rs_c + i*cs_c]          = ymm11a.d[i];           // store (gama30 ... gama37) = ymm11[0] ... ymm11[7])
      c11[3 * rs_c + (i + 4)*cs_c]    = ymm11b.d[i];           // store (gama30 ... gama37) = ymm11[0] ... ymm11[7])
    }


  // iteration 4
  ymm0a.v = _mm256_broadcast_sd(a11 + 4);          // ymm0 = alpha40
  ymm0b.v = _mm256_broadcast_sd(a11 + 4);          // ymm0 = alpha40
  ymm1a.v = _mm256_broadcast_sd(a11 + 1 * 6 + 4);  // ymm1 = alpha41
  ymm1b.v = _mm256_broadcast_sd(a11 + 1 * 6 + 4);  // ymm1 = alpha41
  ymm2a.v = _mm256_broadcast_sd(a11 + 2 * 6 + 4);  // ymm2 = alpha42
  ymm2b.v = _mm256_broadcast_sd(a11 + 2 * 6 + 4);  // ymm2 = alpha42
  ymm3a.v = _mm256_broadcast_sd(a11 + 3 * 6 + 4);  // ymm3 = alpha43
  ymm3b.v = _mm256_broadcast_sd(a11 + 3 * 6 + 4);  // ymm3 = alpha43

  ymm4a.v = _mm256_broadcast_sd(a11 + 4 * 6 + 4);  // ymm4 = 1/alpha44


  ymm0a.v = _mm256_mul_pd(ymm8a.v, ymm0a.v); // alpha40 * ymm8
  ymm0b.v = _mm256_mul_pd(ymm8b.v, ymm0b.v); // alpha40 * ymm8

  ymm1a.v = _mm256_mul_pd(ymm9a.v, ymm1a.v); //  ymm1 = alpha41 * ymm9
  ymm1b.v = _mm256_mul_pd(ymm9b.v, ymm1b.v); //  ymm1 = alpha41 * ymm9

  ymm2a.v = _mm256_mul_pd(ymm10a.v, ymm2a.v); // ymm2 = alpha42 * ymm10
  ymm2b.v = _mm256_mul_pd(ymm10b.v, ymm2b.v); // ymm2 = alpha42 * ymm10

  ymm3a.v = _mm256_mul_pd(ymm11a.v, ymm3a.v); // ymm3 = alpha43 * ymm11
  ymm3b.v = _mm256_mul_pd(ymm11b.v, ymm3b.v); // ymm3 = alpha43 * ymm11

  ymm0a.v = _mm256_add_pd(ymm0a.v, ymm1a.v); // ymm0 += ymm1
  ymm0b.v = _mm256_add_pd(ymm0b.v, ymm1b.v); // ymm0 += ymm1

  ymm2a.v = _mm256_add_pd(ymm2a.v, ymm3a.v);  // ymm2 += ymm3
  ymm2b.v = _mm256_add_pd(ymm2b.v, ymm3b.v);  // ymm2 += ymm3

  ymm0a.v = _mm256_add_pd(ymm0a.v, ymm2a.v); // ymm0 += ymm2 (ymm0 = ymm0 + ymm1 + ymm2 + ymm3)
  ymm0b.v = _mm256_add_pd(ymm0b.v, ymm2b.v); // ymm0 += ymm2 (ymm0 = ymm0 + ymm1 + ymm2 + ymm3)

  ymm16a.v = _mm256_sub_pd(ymm16a.v, ymm0a.v); // ymm16 -= ymm0 {[beta40 ... beta47] -ymm0}
  ymm16b.v = _mm256_sub_pd(ymm16b.v, ymm0b.v); // ymm16 -= ymm0 {[beta40 ... beta47] -ymm0}

  ymm16a.v = _mm256_mul_pd(ymm16a.v, ymm4a.v); // ymm16 *= 1/alpha44
  ymm16b.v = _mm256_mul_pd(ymm16b.v, ymm4a.v); // ymm16 *= 1/alpha44

  // Store the result
  _mm256_storeu_pd(b11 + 8 * 4, ymm16a.v);   // store ( beta40 ... beta43 )  = ymm16
  _mm256_storeu_pd(b11 + 9 * 4, ymm16b.v);   // store ( beta44 ... beta47 )  = ymm16

  // Store @ C11
  for (int i = 0; i < 4; i++)
  {
      c11[4 * rs_c + i*cs_c] = ymm16a.d[i];  // store (gama40 ... gama43) = ymm16[0] ... ymm16[3])
      c11[4 * rs_c + (i + 4)*cs_c] = ymm16b.d[i];  // store (gama44 ... gama47) = ymm16[0] ... ymm16[3])
  }

  // iteration 5

  ymm0a.v = _mm256_broadcast_sd(a11 + 5);         // ymm0 = alpha50
  ymm0b.v = _mm256_broadcast_sd(a11 + 5);         // ymm0 = alpha50
  ymm1a.v = _mm256_broadcast_sd(a11 + 1 * 6 + 5); // ymm1 = alpha51
  ymm1b.v = _mm256_broadcast_sd(a11 + 1 * 6 + 5); // ymm1 = alpha51
  ymm2a.v = _mm256_broadcast_sd(a11 + 2 * 6 + 5); // ymm2 = alpha52
  ymm2b.v = _mm256_broadcast_sd(a11 + 2 * 6 + 5); // ymm2 = alpha52
  ymm3a.v = _mm256_broadcast_sd(a11 + 3 * 6 + 5); // ymm3 = alpha53
  ymm3b.v = _mm256_broadcast_sd(a11 + 3 * 6 + 5); // ymm3 = alpha53
  ymm4a.v = _mm256_broadcast_sd(a11 + 4 * 6 + 5); // ymm4 = alpha54
  ymm4b.v = _mm256_broadcast_sd(a11 + 4 * 6 + 5); // ymm4 = alpha54

  ymm5a.v = _mm256_broadcast_sd(a11 + 5 * 6 + 5); // ymm5 = 1/alpha55

  ymm0a.v = _mm256_mul_pd(ymm8a.v, ymm0a.v);  // ymm0 = alpha50 * ymm8
  ymm0b.v = _mm256_mul_pd(ymm8b.v, ymm0b.v);  // ymm0 = alpha50 * ymm8

  ymm1a.v = _mm256_mul_pd(ymm9a.v, ymm1a.v);  // ymm1 = alpha51 * ymm9
  ymm1b.v = _mm256_mul_pd(ymm9b.v, ymm1b.v);  // ymm1 = alpha51 * ymm9

  ymm2a.v  = _mm256_mul_pd(ymm10a.v, ymm2a.v); // ymm2 = alpha52 * ymm10
  ymm2b.v  = _mm256_mul_pd(ymm10b.v, ymm2b.v); // ymm2 = alpha52 * ymm10

  ymm3a.v  = _mm256_mul_pd(ymm11a.v, ymm3a.v); // ymm3 = alpha53 * ymm11
  ymm3b.v  = _mm256_mul_pd(ymm11b.v, ymm3b.v); // ymm3 = alpha53 * ymm11

  ymm4a.v = _mm256_mul_pd(ymm4a.v, ymm16a.v);  // ymm4 = alpha54 * ymm16
  ymm4b.v = _mm256_mul_pd(ymm4b.v, ymm16b.v);  // ymm4 = alpha54 * ymm16


  ymm0a.v = _mm256_add_pd(ymm0a.v, ymm1a.v);   // ymm0 += ymm1
  ymm0b.v = _mm256_add_pd(ymm0b.v, ymm1b.v);   // ymm0 += ymm1
  ymm2a.v = _mm256_add_pd(ymm2a.v, ymm3a.v);   // ymm2 += ymm3
  ymm2b.v = _mm256_add_pd(ymm2b.v, ymm3b.v);   // ymm2 += ymm3
  ymm0a.v = _mm256_add_pd(ymm0a.v, ymm2a.v);   // ymm0 += ymm2 {ymm0 = ymm0 + ymm1 + ymm2 + ymm3}
  ymm0b.v = _mm256_add_pd(ymm0b.v, ymm2b.v);   // ymm0 += ymm2 {ymm0 = ymm0 + ymm1 + ymm2 + ymm3}
  ymm0a.v = _mm256_add_pd(ymm0a.v, ymm4a.v);   // ymm0 += ymm4 {ymm0 += ymm1 + ymm2 + ymm3 + ymm4}
  ymm0b.v = _mm256_add_pd(ymm0b.v, ymm4b.v);   // ymm0 += ymm4 {ymm0 += ymm1 + ymm2 + ymm3 + ymm4}

  ymm18a.v = _mm256_sub_pd(ymm18a.v, ymm0a.v);  // {[beta50 ... beta53] - ymm0}  ymm18 -= ymm0
  ymm18b.v = _mm256_sub_pd(ymm18b.v, ymm0b.v);  // {[beta54 ... beta57] - ymm0}  ymm18 -= ymm0

  ymm18a.v = _mm256_mul_pd(ymm5a.v, ymm18a.v);  // 1/alpha55 * ymm18
  ymm18b.v = _mm256_mul_pd(ymm5a.v, ymm18b.v);  // 1/alpha55 * ymm18

  // Store the result
  _mm256_storeu_pd(b11 + 10 * 4, ymm18a.v);   // store ( beta50 ... beta53 )  = ymm18
  _mm256_storeu_pd(b11 + 11 * 4, ymm18b.v);   // store ( beta54 ... beta57 )  = ymm18

  // Store @ C11
  for (int i = 0; i < 4; i++)
    {
      c11[5 * rs_c + i*cs_c]          = ymm18a.d[i];  // store (gama50 ... gama57) = ymm18[0] ... ymm18[7])
      c11[5 * rs_c + (i + 4)*cs_c]    = ymm18b.d[i];  // store (gama50 ... gama57) = ymm18[0] ... ymm18[7])
    }

} // End of the function



void bli_dtrsm_l_int_6x16(
			   double*     restrict a11,
			   double*     restrict b11,
			   double*     restrict c11, inc_t rs_c, inc_t cs_c,
			   auxinfo_t* restrict data,
               cntx_t*    restrict cntx
			 )
{
  v4df_t ymm8a;
  v4df_t ymm8b;
  v4df_t ymm12a;
  v4df_t ymm12b;
  v4df_t ymm9a;
  v4df_t ymm9b;
  v4df_t ymm13a;
  v4df_t ymm13b;
  v4df_t ymm10a;
  v4df_t ymm10b;
  v4df_t ymm14a;
  v4df_t ymm14b;
  v4df_t ymm11a;
  v4df_t ymm11b;
  v4df_t ymm15a;
  v4df_t ymm15b;
  v4df_t ymm0a;
  v4df_t ymm0b;
  v4df_t ymm1a;
  v4df_t ymm1b;
  v4df_t ymm2a;
  v4df_t ymm2b;
  v4df_t ymm3a;
  v4df_t ymm3b;
  v4df_t ymm4a;
  v4df_t ymm4b;
  v4df_t ymm5a;
  v4df_t ymm5b;
  v4df_t ymm6a;
  v4df_t ymm6b;
  v4df_t ymm7a;
  v4df_t ymm7b;

  v4df_t ymm16a;
  v4df_t ymm16b;
  v4df_t ymm17a;
  v4df_t ymm17b;
  v4df_t ymm18a;
  v4df_t ymm18b;
  v4df_t ymm19a;
  v4df_t ymm19b;

  // b11 - row major
  // a11 - column major
  // c11 - rs_c & cs_c

  // a11 = The diagonal elements of a11 are stored in 1/alpha's to avoid costly division operation in TRSV.

  // Load all the rows of b11
  ymm8a.v  = _mm256_loadu_pd(b11 + 0 * 4);     // ymm8a  = beta00 ... beta03
  ymm8b.v  = _mm256_loadu_pd(b11 + 1 * 4);     // ymm8b  = beta04 ... beta07
  ymm12a.v = _mm256_loadu_pd(b11 + 2 * 4);     // ymm12a = beta08 ... beta11
  ymm12b.v = _mm256_loadu_pd(b11 + 3 * 4);     // ymm12b = beta12 ... beta15
  ymm9a.v  = _mm256_loadu_pd(b11 + 4 * 4);     // ymm9a  = beta10 ... beta13
  ymm9b.v  = _mm256_loadu_pd(b11 + 5 * 4);     // ymm9b  = beta1.4 ... beta1.7
  ymm13a.v = _mm256_loadu_pd(b11 + 6 * 4);     // ymm13a = beta1.8 ... beta1.11
  ymm13b.v = _mm256_loadu_pd(b11 + 7 * 4);     // ymm13b = beta1.12 ... beta1.15
  ymm10a.v = _mm256_loadu_pd(b11 + 8 * 4);     // ymm10a = beta20 ... beta23
  ymm10b.v = _mm256_loadu_pd(b11 + 9 * 4);     // ymm10b = beta2.4 ... beta2.7
  ymm14a.v = _mm256_loadu_pd(b11 + 10 * 4);    // ymm14a = beta2.8 ... beta2,11
  ymm14b.v = _mm256_loadu_pd(b11 + 11 * 4);    // ymm14b = beta2.12 ... beta2.15
  ymm11a.v = _mm256_loadu_pd(b11 + 12 * 4);     // ymm11a = beta30 ... beta3.3
  ymm11b.v = _mm256_loadu_pd(b11 + 13 * 4);     // ymm11b = beta3.4 ... beta3.7
  ymm15a.v = _mm256_loadu_pd(b11 + 14 * 4);     // ymm15a = beta3.8 ... beta3.11
  ymm15b.v = _mm256_loadu_pd(b11 + 15 * 4);     // ymm15b = beta3.12 ... beta3.15
  ymm16a.v = _mm256_loadu_pd(b11 + 16 * 4);     // ymm16a = beta40 ... beta43
  ymm16b.v = _mm256_loadu_pd(b11 + 17 * 4);     // ymm16b = beta44 ... beta47
  ymm17a.v = _mm256_loadu_pd(b11 + 18 * 4);     // ymm17a = beta4.8 ... beta4.11
  ymm17b.v = _mm256_loadu_pd(b11 + 19 * 4);     // ymm17a = beta4.12 ... beta4.15
  ymm18a.v = _mm256_loadu_pd(b11 + 20 * 4);     // ymm18a = beta5.0 ... beta5.3
  ymm18b.v = _mm256_loadu_pd(b11 + 21 * 4);     // ymm18b = beta5.4 ... beta5.7
  ymm19a.v = _mm256_loadu_pd(b11 + 22 * 4);     // ymm19a = beta5.8 ... beta5.11
  ymm19b.v = _mm256_loadu_pd(b11 + 23 * 4);     // ymm19b = beta5.12 ... beta5.15

  // iteration 0
  ymm0a.v  = _mm256_broadcast_sd(a11 + 0);     // load ymm0 = (1/alpha00)
  ymm8a.v  = _mm256_mul_pd(ymm0a.v, ymm8a.v);    // ymm8a  *= (1/alpha00);
  ymm8b.v  = _mm256_mul_pd(ymm0a.v, ymm8b.v);    // ymm8b  *= (1/alpha00);
  ymm12a.v = _mm256_mul_pd(ymm0a.v, ymm12a.v);   // ymm12a *= (1/alpha00);
  ymm12b.v = _mm256_mul_pd(ymm0a.v, ymm12b.v);   // ymm12b *= (1/alpha00);

  _mm256_storeu_pd(b11 + 0 * 4, ymm8a.v);   // store ( beta00 ... beta03 ) = ymm8a
  _mm256_storeu_pd(b11 + 1 * 4, ymm8b.v);   // store ( beta04 ... beta07 ) = ymm8b
  _mm256_storeu_pd(b11 + 2 * 4, ymm12a.v);  // store ( beta08 ... beta1.11 ) = ymm12a
  _mm256_storeu_pd(b11 + 3 * 4, ymm12b.v);  // store ( beta0.12 ... beta0.15 ) = ymm12b
  // Store in C11 
  for (int i = 0; i < 4; i++)
    {
      c11[0 * rs_c + i*cs_c]          = ymm8a.d[i];   // store (gama00 ... gama03)     = ymm8[0] ... ymm8[3])
      c11[0 * rs_c + (i + 4)*cs_c]    = ymm8b.d[i];   // store (gama04 ... gama07)     = ymm8[4] ... ymm8[7])
      c11[0 * rs_c + (i + 8) * cs_c]  = ymm12a.d[i];  // store (gama08 ... gama01.11)  = ymm12[8] ... ymm12[11])
      c11[0 * rs_c + (i + 12) * cs_c] = ymm12b.d[i];  // store (gama0.12 ... gama0.15) = ymm12[12] ... ymm12[15])
    }

  // iteration 1
  ymm0a.v = _mm256_broadcast_sd(a11 + 1);          // ymm0 = (1/alpha10)
  ymm0b.v = _mm256_broadcast_sd(a11 + 1);          // ymm0 = (1/alpha10)
  ymm1a.v = _mm256_broadcast_sd(a11 + 1 * 6 + 1);  // ymm1 = (1/alpha11)
  ymm4a.v = _mm256_loadu_pd(ymm0a.d);               // ymm4 = ymm0 = (1/alpha10)
  ymm4b.v = _mm256_loadu_pd(ymm0a.d);               // ymm4 = ymm0 = (1/alpha10)

  ymm0a.v = _mm256_mul_pd(ymm0a.v, ymm8a.v);        // ymm0 = alpha10 * alpha00* (beta00 .. beta03)
  ymm0b.v = _mm256_mul_pd(ymm0b.v, ymm8b.v);        // ymm0 = alpha10 * alpha00* (beta04 .. beta07)
  ymm4a.v = _mm256_mul_pd(ymm4a.v, ymm12a.v);       // ymm4 = alpha10 * alpha00 * (beta08 ... beta0.11)
  ymm4b.v = _mm256_mul_pd(ymm4b.v, ymm12b.v);       // ymm4 = alpha10 * alpha00 * (beta0.12 ... beta0.15)

  ymm9a.v = _mm256_sub_pd(ymm9a.v, ymm0a.v);       // ymm9 =  [beta10 ... beta13] - ymm0
  ymm9b.v = _mm256_sub_pd(ymm9b.v, ymm0b.v);       // ymm9 =  [beta1.4 ... beta17] - ymm0

  ymm13a.v = _mm256_sub_pd(ymm13a.v, ymm4a.v);      // ymm13 = [beta18 ... beta1,11] - ymm4
  ymm13b.v = _mm256_sub_pd(ymm13b.v, ymm4b.v);      // ymm13 = [beta1.12 ... beta1.15] - ymm4

  ymm9a.v  = _mm256_mul_pd(ymm9a.v,  ymm1a.v);       // ymm9 = 1/alpha11 * ymm9
  ymm9b.v  = _mm256_mul_pd(ymm9b.v,  ymm1a.v);       // ymm9 = 1/alpha11 * ymm9
  ymm13a.v = _mm256_mul_pd(ymm13a.v, ymm1a.v);      // ymm13 = 1/alpha11 * ymm13
  ymm13b.v = _mm256_mul_pd(ymm13b.v, ymm1a.v);      // ymm13 = 1/alpha11 * ymm13

  _mm256_storeu_pd(b11 + 4 * 4, ymm9a.v);   // store ( beta10 ... beta17 )  = ymm9
  _mm256_storeu_pd(b11 + 5 * 4, ymm9b.v);   // store ( beta10 ... beta17 )  = ymm9
  _mm256_storeu_pd(b11 + 6 * 4, ymm13a.v);  // store ( beta18 ... beta1,15 ) = ymm13
  _mm256_storeu_pd(b11 + 7 * 4, ymm13b.v);  // store ( beta18 ... beta1,15 ) = ymm13

  // Store @ C11
  for (int i = 0; i < 4; i++)
    {
      c11[1 * rs_c + i*cs_c]          = ymm9a.d[i];  // store (gama10 ... gama17) = ymm9[0] ... ymm9[7])
      c11[1 * rs_c + (i + 4)*cs_c]    = ymm9b.d[i];  // store (gama10 ... gama17) = ymm9[0] ... ymm9[7])
      c11[1 * rs_c + (i + 8) * cs_c]  = ymm13a.d[i]; // store (gama18 ... gama115) = ymm13[0] ... ymm13[7])
      c11[1 * rs_c + (i + 12) * cs_c] = ymm13b.d[i]; // store (gama18 ... gama115) = ymm13[0] ... ymm13[7])
    }

  // iteration 2
  ymm0a.v = _mm256_broadcast_sd(a11 + 2);  // ymm0 = 1/alpha20
  ymm0b.v = _mm256_broadcast_sd(a11 + 2);  // ymm0 = 1/alpha20
  ymm1a.v = _mm256_broadcast_sd(a11 + 1 * 6 + 2); // ymm1 = 1/alpha21
  ymm1b.v = _mm256_broadcast_sd(a11 + 1 * 6 + 2); // ymm1 = 1/alpha21

  ymm2a.v = _mm256_broadcast_sd(a11 + 2 * 6 + 2); // ymm2 = 1/alpha22

  ymm4a.v = _mm256_loadu_pd(ymm0a.d);               // ymm4 = ymm0 = 1/alpha20
  ymm4b.v = _mm256_loadu_pd(ymm0b.d);               // ymm4 = ymm0 = 1/alpha20
  ymm5a.v = _mm256_loadu_pd(ymm1a.d);               // ymm5 = ymm1 = 1/alpha21
  ymm5b.v = _mm256_loadu_pd(ymm1b.d);               // ymm5 = ymm1 = 1/alpha21

  ymm0a.v = _mm256_mul_pd(ymm0a.v, ymm8a.v);        // ymm0 = alpha20 * [alpha00* (beta00 .. beta03)]
  ymm0b.v = _mm256_mul_pd(ymm0b.v, ymm8b.v);        // ymm0 = alpha20 * [alpha00* (beta04 .. beta07)]
  ymm4a.v = _mm256_mul_pd(ymm4a.v, ymm12a.v);       // ymm4 = alpha20 * [alpha00 * (beta08 ... beta011)]
  ymm4b.v = _mm256_mul_pd(ymm4b.v, ymm12b.v);       // ymm4 = alpha20 * [alpha00 * (beta012 ... beta015)]

  ymm1a.v = _mm256_mul_pd(ymm9a.v, ymm1a.v);        // ymm1 = alpha21 * ymm9
  ymm1b.v = _mm256_mul_pd(ymm9b.v, ymm1b.v);        // ymm1 = alpha21 * ymm9

  ymm5a.v = _mm256_mul_pd(ymm13a.v, ymm5a.v);       // ymm5 = alpha21 * ymm13
  ymm5b.v = _mm256_mul_pd(ymm13b.v, ymm5b.v);       // ymm5 = alpha21 * ymm13

  ymm0a.v = _mm256_add_pd(ymm1a.v, ymm0a.v);       // ymm0 += ymm1
  ymm0b.v = _mm256_add_pd(ymm1b.v, ymm0b.v);       // ymm0 += ymm1
  ymm4a.v = _mm256_add_pd(ymm5a.v, ymm4a.v);       // ymm4 += ymm5
  ymm4b.v = _mm256_add_pd(ymm5b.v, ymm4b.v);       // ymm4 += ymm5

  ymm10a.v = _mm256_sub_pd(ymm10a.v, ymm0a.v);    // ymm10 -= ymm0
  ymm10b.v = _mm256_sub_pd(ymm10b.v, ymm0b.v);    // ymm10 -= ymm0
  ymm14a.v = _mm256_sub_pd(ymm14a.v, ymm4a.v);    // ymm14 -= ymm4
  ymm14b.v = _mm256_sub_pd(ymm14b.v, ymm4b.v);    // ymm14 -= ymm4

  ymm10a.v = _mm256_mul_pd(ymm10a.v, ymm2a.v);    // ymm10 *= 1/alpha22
  ymm10b.v = _mm256_mul_pd(ymm10b.v, ymm2a.v);    // ymm10 *= 1/alpha22
  ymm14a.v = _mm256_mul_pd(ymm14a.v, ymm2a.v);    // ymm14 *= 1/alpha22
  ymm14b.v = _mm256_mul_pd(ymm14b.v, ymm2a.v);    // ymm14 *= 1/alpha22

  // Store the result
  _mm256_storeu_pd(b11 + 8 * 4,  ymm10a.v);   // store ( beta20 ... beta27 )  = ymm10
  _mm256_storeu_pd(b11 + 9 * 4,  ymm10b.v);   // store ( beta20 ... beta27 )  = ymm10
  _mm256_storeu_pd(b11 + 10 * 4, ymm14a.v);  // store ( beta28 ... beta2,15 ) = ymm14
  _mm256_storeu_pd(b11 + 11 * 4, ymm14b.v);  // store ( beta28 ... beta2,15 ) = ymm14

  // Store @ C11
  for (int i = 0; i < 4; i++)
    {
      c11[2 * rs_c + i*cs_c]          = ymm10a.d[i];  // store (gama20 ... gama27) = ymm10[0] ... ymm10[7])
      c11[2 * rs_c + (i + 4)*cs_c]    = ymm10b.d[i];  // store (gama20 ... gama27) = ymm10[0] ... ymm10[7])
      c11[2 * rs_c + (i + 8) * cs_c]  = ymm14a.d[i];  // store (gama28 ... gama2,15) = ymm14[0] ... ymm14[7])
      c11[2 * rs_c + (i + 12) * cs_c] = ymm14b.d[i];  // store (gama28 ... gama2,15) = ymm14[0] ... ymm14[7])
    }

  // iteration 3
  ymm0a.v = _mm256_broadcast_sd(a11 + 3);  // ymm0 = 1/alpha30
  ymm0b.v = _mm256_broadcast_sd(a11 + 3);  // ymm0 = 1/alpha30    
  ymm1a.v = _mm256_broadcast_sd(a11 + 1 * 6 + 3); // ymm1 = 1/alpha31
  ymm1b.v = _mm256_broadcast_sd(a11 + 1 * 6 + 3); // ymm1 = 1/alpha31
  ymm2a.v = _mm256_broadcast_sd(a11 + 2 * 6 + 3); // ymm2 = 1/alpha32
  ymm2b.v = _mm256_broadcast_sd(a11 + 2 * 6 + 3); // ymm2 = 1/alpha32

  ymm3a.v = _mm256_broadcast_sd(a11 + 3 * 6 + 3); // ymm3 = 1/alpha33

  ymm4a.v = _mm256_loadu_pd(ymm0a.d);               // ymm4 = ymm0 = 1/alpha30
  ymm4b.v = _mm256_loadu_pd(ymm0b.d);               // ymm4 = ymm0 = 1/alpha30
  ymm5a.v = _mm256_loadu_pd(ymm1a.d);               // ymm5 = ymm1 = 1/alpha31
  ymm5b.v = _mm256_loadu_pd(ymm1b.d);               // ymm5 = ymm1 = 1/alpha31
  ymm6a.v = _mm256_loadu_pd(ymm2a.d);               // ymm6 = ymm2 = 1/alpha32
  ymm6b.v = _mm256_loadu_pd(ymm2b.d);               // ymm6 = ymm2 = 1/alpha32

  ymm0a.v = _mm256_mul_pd(ymm8a.v, ymm0a.v);       // ymm0 = alpha30 * ymm8
  ymm0b.v = _mm256_mul_pd(ymm8b.v, ymm0b.v);       // ymm0 = alpha30 * ymm8
  ymm4a.v = _mm256_mul_pd(ymm12a.v, ymm4a.v);      // ymm4 = alpha30 * ymm12
  ymm4b.v = _mm256_mul_pd(ymm12b.v, ymm4b.v);      // ymm4 = alpha30 * ymm12
  ymm1a.v = _mm256_mul_pd(ymm9a.v, ymm1a.v);       // ymm1 = alpha31 * ymm9
  ymm1b.v = _mm256_mul_pd(ymm9b.v, ymm1b.v);       // ymm1 = alpha31 * ymm9
  ymm5a.v = _mm256_mul_pd(ymm13a.v, ymm5a.v);      // ymm5 = alpha31 * ymm13
  ymm5b.v = _mm256_mul_pd(ymm13b.v, ymm5b.v);      // ymm5 = alpha31 * ymm13
  ymm2a.v = _mm256_mul_pd(ymm10a.v, ymm2a.v);      // ymm2 = alpha32 * ymm10
  ymm2b.v = _mm256_mul_pd(ymm10b.v, ymm2b.v);      // ymm2 = alpha32 * ymm10
  ymm6a.v = _mm256_mul_pd(ymm14a.v, ymm6a.v);      // ymm6 = alpha32 * ymm14
  ymm6b.v = _mm256_mul_pd(ymm14b.v, ymm6b.v);      // ymm6 = alpha32 * ymm14

  ymm0a.v = _mm256_add_pd(ymm0a.v, ymm1a.v);      // ymm0 += ymm1
  ymm0b.v = _mm256_add_pd(ymm0b.v, ymm1b.v);      // ymm0 += ymm1
  ymm4a.v = _mm256_add_pd(ymm5a.v, ymm4a.v);      // ymm4 += ymm5
  ymm4b.v = _mm256_add_pd(ymm5b.v, ymm4b.v);      // ymm4 += ymm5
  ymm0a.v = _mm256_add_pd(ymm2a.v, ymm0a.v);      // ymm0 += ymm2
  ymm0b.v = _mm256_add_pd(ymm2b.v, ymm0b.v);      // ymm0 += ymm2
  ymm4a.v = _mm256_add_pd(ymm6a.v, ymm4a.v);      // ymm4 += ymm6
  ymm4b.v = _mm256_add_pd(ymm6b.v, ymm4b.v);      // ymm4 += ymm6

  ymm11a.v = _mm256_sub_pd(ymm11a.v, ymm0a.v);    // ymm11 -= ymm0 {[beta30 ... beta37] - ymm0}
  ymm11b.v = _mm256_sub_pd(ymm11b.v, ymm0b.v);    // ymm11 -= ymm0 {[beta30 ... beta37] - ymm0}
  ymm15a.v = _mm256_sub_pd(ymm15a.v, ymm4a.v);    // ymm15 -= ymm4 {[beta38 ... beta3,15] - ymm4}
  ymm15b.v = _mm256_sub_pd(ymm15b.v, ymm4b.v);    // ymm15 -= ymm4 {[beta38 ... beta3,15] - ymm4}

  ymm11a.v = _mm256_mul_pd(ymm11a.v, ymm3a.v);    // ymm11 *= alpha33
  ymm11b.v = _mm256_mul_pd(ymm11b.v, ymm3a.v);    // ymm11 *= alpha33
  ymm15a.v = _mm256_mul_pd(ymm15a.v, ymm3a.v);    // ymm15 *= alpha33
  ymm15b.v = _mm256_mul_pd(ymm15b.v, ymm3a.v);    // ymm15 *= alpha33

  // Store the result
  _mm256_storeu_pd(b11 + 12 * 4, ymm11a.v);   // store ( beta30 ... beta37 )  = ymm11
  _mm256_storeu_pd(b11 + 13 * 4, ymm11b.v);   // store ( beta30 ... beta37 )  = ymm11
  _mm256_storeu_pd(b11 + 14 * 4, ymm15a.v);  // store ( beta38 ... beta3,15 ) = ymm15
  _mm256_storeu_pd(b11 + 15 * 4, ymm15b.v);  // store ( beta38 ... beta3,15 ) = ymm15

  // Store @ C11
  for (int i = 0; i < 4; i++)
    {
      c11[3 * rs_c + i*cs_c]          = ymm11a.d[i];           // store (gama30 ... gama37) = ymm11[0] ... ymm11[7])
      c11[3 * rs_c + (i + 4)*cs_c]    = ymm11b.d[i];           // store (gama30 ... gama37) = ymm11[0] ... ymm11[7])
      c11[3 * rs_c + (i + 8) * cs_c]  = ymm15a.d[i];  // store (gama38 ... gama3,15) = ymm15[0] ... ymm15[7])
      c11[3 * rs_c + (i + 12) * cs_c] = ymm15b.d[i];  // store (gama38 ... gama3,15) = ymm15[0] ... ymm15[7])
    }


  // iteration 4
  v4df_t ymm21a;
  v4df_t ymm21b;
  ymm0a.v = _mm256_broadcast_sd(a11 + 4);          // ymm0 = 1/alpha40
  ymm0b.v = _mm256_broadcast_sd(a11 + 4);          // ymm0 = 1/alpha40
  ymm1a.v = _mm256_broadcast_sd(a11 + 1 * 6 + 4);  // ymm1 = 1/alpha41
  ymm1b.v = _mm256_broadcast_sd(a11 + 1 * 6 + 4);  // ymm1 = 1/alpha41
  ymm2a.v = _mm256_broadcast_sd(a11 + 2 * 6 + 4);  // ymm2 = 1/alpha42
  ymm2b.v = _mm256_broadcast_sd(a11 + 2 * 6 + 4);  // ymm2 = 1/alpha42
  ymm3a.v = _mm256_broadcast_sd(a11 + 3 * 6 + 4);  // ymm3 = 1/alpha43
  ymm3b.v = _mm256_broadcast_sd(a11 + 3 * 6 + 4);  // ymm3 = 1/alpha43

  ymm4a.v = _mm256_broadcast_sd(a11 + 4 * 6 + 4);  // ymm4 = 1/alpha44

  ymm5a.v  = _mm256_loadu_pd(ymm0a.d);   // ymm5 = ymm0 = alpha40
  ymm5b.v  = _mm256_loadu_pd(ymm0b.d);   // ymm5 = ymm0 = alpha40
  ymm6a.v  = _mm256_loadu_pd(ymm1a.d);   // ymm6 = ymm1 = alpha41
  ymm6b.v  = _mm256_loadu_pd(ymm1b.d);   // ymm6 = ymm1 = alpha41
  ymm7a.v  = _mm256_loadu_pd(ymm2a.d);   // ymm7 = ymm2 = alpha42
  ymm7b.v  = _mm256_loadu_pd(ymm2b.d);   // ymm7 = ymm2 = alpha42
  ymm21a.v = _mm256_loadu_pd(ymm3a.d);  // ymm21 = ymm3 = alpha43
  ymm21b.v = _mm256_loadu_pd(ymm3b.d);  // ymm21 = ymm3 = alpha43

  ymm0a.v = _mm256_mul_pd(ymm8a.v, ymm0a.v); // alpha40 * ymm8
  ymm0b.v = _mm256_mul_pd(ymm8b.v, ymm0b.v); // alpha40 * ymm8
  ymm5a.v = _mm256_mul_pd(ymm12a.v, ymm5a.v); // alpha40 * ymm12
  ymm5b.v = _mm256_mul_pd(ymm12b.v, ymm5b.v); // alpha40 * ymm12

  ymm1a.v = _mm256_mul_pd(ymm9a.v, ymm1a.v); //  ymm1 = alpha41 * ymm9
  ymm1b.v = _mm256_mul_pd(ymm9b.v, ymm1b.v); //  ymm1 = alpha41 * ymm9
  ymm6a.v = _mm256_mul_pd(ymm13a.v, ymm6a.v); // ymm6 = alpha41 * ymm13
  ymm6b.v = _mm256_mul_pd(ymm13b.v, ymm6b.v); // ymm6 = alpha41 * ymm13

  ymm2a.v = _mm256_mul_pd(ymm10a.v, ymm2a.v); // ymm2 = alpha42 * ymm10
  ymm2b.v = _mm256_mul_pd(ymm10b.v, ymm2b.v); // ymm2 = alpha42 * ymm10
  ymm7a.v = _mm256_mul_pd(ymm14a.v, ymm7a.v); // ymm7 = alpha42 * ymm14
  ymm7b.v = _mm256_mul_pd(ymm14b.v, ymm7b.v); // ymm7 = alpha42 * ymm14

  ymm3a.v = _mm256_mul_pd(ymm11a.v, ymm3a.v); // ymm3 = alpha43 * ymm11
  ymm3b.v = _mm256_mul_pd(ymm11b.v, ymm3b.v); // ymm3 = alpha43 * ymm11
  ymm21a.v = _mm256_mul_pd(ymm15a.v, ymm21a.v); // ymm21 = alpha43 * ymm21
  ymm21b.v = _mm256_mul_pd(ymm15b.v, ymm21b.v); // ymm21 = alpha43 * ymm21

  ymm0a.v = _mm256_add_pd(ymm0a.v, ymm1a.v); // ymm0 += ymm1
  ymm0b.v = _mm256_add_pd(ymm0b.v, ymm1b.v); // ymm0 += ymm1
  ymm5a.v = _mm256_add_pd(ymm5a.v, ymm6a.v); // ymm5 += ymm6
  ymm5b.v = _mm256_add_pd(ymm5b.v, ymm6b.v); // ymm5 += ymm6

  ymm2a.v = _mm256_add_pd(ymm2a.v, ymm3a.v);  // ymm2 += ymm3
  ymm2b.v = _mm256_add_pd(ymm2b.v, ymm3b.v);  // ymm2 += ymm3
  ymm7a.v = _mm256_add_pd(ymm7a.v, ymm21a.v); // ymm7 += ymm21
  ymm7b.v = _mm256_add_pd(ymm7b.v, ymm21b.v); // ymm7 += ymm21

  ymm0a.v = _mm256_add_pd(ymm0a.v, ymm2a.v); // ymm0 += ymm2 (ymm0 = ymm0 + ymm1 + ymm2 + ymm3)
  ymm0b.v = _mm256_add_pd(ymm0b.v, ymm2b.v); // ymm0 += ymm2 (ymm0 = ymm0 + ymm1 + ymm2 + ymm3)
  ymm5a.v = _mm256_add_pd(ymm5a.v, ymm7a.v); // ymm5 += ymm7 (ymm5 = ymm5 + ymm6 + ymm7 + ymm21)
  ymm5b.v = _mm256_add_pd(ymm5b.v, ymm7b.v); // ymm5 += ymm7 (ymm5 = ymm5 + ymm6 + ymm7 + ymm21)

  ymm16a.v = _mm256_sub_pd(ymm16a.v, ymm0a.v); // ymm16 -= ymm0 {[beta40 ... beta47] -ymm0}
  ymm16b.v = _mm256_sub_pd(ymm16b.v, ymm0b.v); // ymm16 -= ymm0 {[beta40 ... beta47] -ymm0}
  ymm17a.v = _mm256_sub_pd(ymm17a.v, ymm5a.v); // ymm17 -= ymm5 {[beta48 ... beta4,15] - ymm17}
  ymm17b.v = _mm256_sub_pd(ymm17b.v, ymm5b.v); // ymm17 -= ymm5 {[beta48 ... beta4,15] - ymm17}

  ymm16a.v = _mm256_mul_pd(ymm16a.v, ymm4a.v); // ymm16 *= alpha44
  ymm16b.v = _mm256_mul_pd(ymm16b.v, ymm4a.v); // ymm16 *= alpha44
  ymm17a.v = _mm256_mul_pd(ymm17a.v, ymm4a.v); // ymm17 *= alpha44
  ymm17b.v = _mm256_mul_pd(ymm17b.v, ymm4a.v); // ymm17 *= alpha44

  // Store the result
  _mm256_storeu_pd(b11 + 16 * 4, ymm16a.v);   // store ( beta40 ... beta43 )  = ymm16
  _mm256_storeu_pd(b11 + 17 * 4, ymm16b.v);   // store ( beta44 ... beta47 )  = ymm16
  _mm256_storeu_pd(b11 + 18 * 4, ymm17a.v);  // store ( beta48 ... beta4,11 ) = ymm17
  _mm256_storeu_pd(b11 + 19 * 4, ymm17b.v);  // store ( beta4,12 ... beta4,15 ) = ymm17

  // Store @ C11
  for (int i = 0; i < 4; i++)
    {
      c11[4 * rs_c + i*cs_c]          = ymm16a.d[i];  // store (gama40 ... gama43) = ymm16[0] ... ymm16[3])
      c11[4 * rs_c + (i + 4)*cs_c]    = ymm16b.d[i];  // store (gama44 ... gama47) = ymm16[0] ... ymm16[3])
      c11[4 * rs_c + (i + 8) * cs_c]  = ymm17a.d[i];  // store (gama48 ... gama4,11) = ymm17[0] ... ymm17[3])
      c11[4 * rs_c + (i + 12) * cs_c] = ymm17b.d[i];  // store (gama4,12 ... gama4,15) = ymm17[0] ... ymm17[3])
    }

  // iteration 5
  v4df_t ymm31a;
  v4df_t ymm31b;
  v4df_t ymm41a;
  v4df_t ymm41b;

  ymm0a.v = _mm256_broadcast_sd(a11 + 5);         // ymm0 = alpha50
  ymm0b.v = _mm256_broadcast_sd(a11 + 5);         // ymm0 = alpha50
  ymm1a.v = _mm256_broadcast_sd(a11 + 1 * 6 + 5); // ymm1 = alpha51
  ymm1b.v = _mm256_broadcast_sd(a11 + 1 * 6 + 5); // ymm1 = alpha51
  ymm2a.v = _mm256_broadcast_sd(a11 + 2 * 6 + 5); // ymm2 = alpha52
  ymm2b.v = _mm256_broadcast_sd(a11 + 2 * 6 + 5); // ymm2 = alpha52
  ymm3a.v = _mm256_broadcast_sd(a11 + 3 * 6 + 5); // ymm3 = alpha53
  ymm3b.v = _mm256_broadcast_sd(a11 + 3 * 6 + 5); // ymm3 = alpha53
  ymm4a.v = _mm256_broadcast_sd(a11 + 4 * 6 + 5); // ymm4 = alpha54
  ymm4b.v = _mm256_broadcast_sd(a11 + 4 * 6 + 5); // ymm4 = alpha54

  ymm5a.v = _mm256_broadcast_sd(a11 + 5 * 6 + 5); // ymm5 = 1/alpha55

  ymm6a.v = _mm256_loadu_pd(ymm0a.d);   // ymm6 = ymm0 = alpha50
  ymm6b.v = _mm256_loadu_pd(ymm0b.d);   // ymm6 = ymm0 = alpha50
  ymm7a.v = _mm256_loadu_pd(ymm1a.d);   // ymm7 = ymm1 = alpha51
  ymm7b.v = _mm256_loadu_pd(ymm1b.d);   // ymm7 = ymm1 = alpha51
  ymm21a.v = _mm256_loadu_pd(ymm2a.d);   // ymm21 = ymm2 = alpha52
  ymm21b.v = _mm256_loadu_pd(ymm2b.d);   // ymm21 = ymm2 = alpha52
  ymm31a.v = _mm256_loadu_pd(ymm3a.d);   // ymm31 = ymm3 = alpha53
  ymm31b.v = _mm256_loadu_pd(ymm3b.d);   // ymm31 = ymm3 = alpha53
  ymm41a.v = _mm256_loadu_pd(ymm4a.d);   // ymm41 = ymm4 = alpha54
  ymm41b.v = _mm256_loadu_pd(ymm4b.d);   // ymm41 = ymm4 = alpha54

  ymm0a.v = _mm256_mul_pd(ymm8a.v, ymm0a.v);  // ymm0 = alpha50 * ymm8
  ymm0b.v = _mm256_mul_pd(ymm8b.v, ymm0b.v);  // ymm0 = alpha50 * ymm8
  ymm6a.v = _mm256_mul_pd(ymm12a.v, ymm6a.v); // ymm6 = alpha50 * ymm12
  ymm6b.v = _mm256_mul_pd(ymm12b.v, ymm6b.v); // ymm6 = alpha50 * ymm12

  ymm1a.v = _mm256_mul_pd(ymm9a.v, ymm1a.v);  // ymm1 = alpha51 * ymm9
  ymm1b.v = _mm256_mul_pd(ymm9b.v, ymm1b.v);  // ymm1 = alpha51 * ymm9
  ymm7a.v = _mm256_mul_pd(ymm13a.v, ymm7a.v); // ymm7 = alpha51 * ymm13
  ymm7b.v = _mm256_mul_pd(ymm13b.v, ymm7b.v); // ymm7 = alpha51 * ymm13

  ymm2a.v  = _mm256_mul_pd(ymm10a.v, ymm2a.v); // ymm2 = alpha52 * ymm10
  ymm2b.v  = _mm256_mul_pd(ymm10b.v, ymm2b.v); // ymm2 = alpha52 * ymm10
  ymm21a.v = _mm256_mul_pd(ymm14a.v, ymm21a.v); // ymm21 = alpha52 * ymm14
  ymm21b.v = _mm256_mul_pd(ymm14b.v, ymm21b.v); // ymm21 = alpha52 * ymm14

  ymm3a.v  = _mm256_mul_pd(ymm11a.v, ymm3a.v); // ymm3 = alpha53 * ymm11
  ymm3b.v  = _mm256_mul_pd(ymm11b.v, ymm3b.v); // ymm3 = alpha53 * ymm11
  ymm31a.v = _mm256_mul_pd(ymm31a.v, ymm15a.v); // ymm31 = alpha53 * ymm15
  ymm31b.v = _mm256_mul_pd(ymm31b.v, ymm15b.v); // ymm31 = alpha53 * ymm15

  ymm4a.v = _mm256_mul_pd(ymm4a.v, ymm16a.v);  // ymm4 = alpha54 * ymm16
  ymm4b.v = _mm256_mul_pd(ymm4b.v, ymm16b.v);  // ymm4 = alpha54 * ymm16

  ymm41a.v = _mm256_mul_pd(ymm41a.v, ymm17a.v); // ymm41 = alpha54 * ymm17
  ymm41b.v = _mm256_mul_pd(ymm41b.v, ymm17b.v); // ymm41 = alpha54 * ymm17

  ymm0a.v = _mm256_add_pd(ymm0a.v, ymm1a.v);   // ymm0 += ymm1
  ymm0b.v = _mm256_add_pd(ymm0b.v, ymm1b.v);   // ymm0 += ymm1
  ymm6a.v = _mm256_add_pd(ymm6a.v, ymm7a.v);   // ymm6 += ymm7
  ymm6b.v = _mm256_add_pd(ymm6b.v, ymm7b.v);   // ymm6 += ymm7
  ymm2a.v = _mm256_add_pd(ymm2a.v, ymm3a.v);   // ymm2 += ymm3
  ymm2b.v = _mm256_add_pd(ymm2b.v, ymm3b.v);   // ymm2 += ymm3
  ymm21a.v = _mm256_add_pd(ymm21a.v, ymm31a.v); // ymm21 += ymm31
  ymm21b.v = _mm256_add_pd(ymm21b.v, ymm31b.v); // ymm21 += ymm31
  ymm0a.v = _mm256_add_pd(ymm0a.v, ymm2a.v);   // ymm0 += ymm2 {ymm0 = ymm0 + ymm1 + ymm2 + ymm3}
  ymm0b.v = _mm256_add_pd(ymm0b.v, ymm2b.v);   // ymm0 += ymm2 {ymm0 = ymm0 + ymm1 + ymm2 + ymm3}
  ymm6a.v = _mm256_add_pd(ymm6a.v, ymm21a.v);  // ymm6 += ymm21 {ymm6 = ymm6 + ymm7 + ymm21 + ymm31}
  ymm6b.v = _mm256_add_pd(ymm6b.v, ymm21b.v);  // ymm6 += ymm21 {ymm6 = ymm6 + ymm7 + ymm21 + ymm31}
  ymm0a.v = _mm256_add_pd(ymm0a.v, ymm4a.v);   // ymm0 += ymm4 {ymm0 += ymm1 + ymm2 + ymm3 + ymm4}
  ymm0b.v = _mm256_add_pd(ymm0b.v, ymm4b.v);   // ymm0 += ymm4 {ymm0 += ymm1 + ymm2 + ymm3 + ymm4}
  ymm6a.v = _mm256_add_pd(ymm6a.v, ymm41a.v);  // ymm6 += ymm41 { ymm6 += ymm7 + ymm21 + ymm31 + ymm41}
  ymm6b.v = _mm256_add_pd(ymm6b.v, ymm41b.v);  // ymm6 += ymm41 { ymm6 += ymm7 + ymm21 + ymm31 + ymm41}

  ymm18a.v = _mm256_sub_pd(ymm18a.v, ymm0a.v);  // {[beta50 ... beta53] - ymm0}  ymm18 -= ymm0
  ymm18b.v = _mm256_sub_pd(ymm18b.v, ymm0b.v);  // {[beta54 ... beta57] - ymm0}  ymm18 -= ymm0
  ymm19a.v = _mm256_sub_pd(ymm19a.v, ymm6a.v); // {[beta58 ... beta5,11] - ymm6}: ymm19 -= ymm6
  ymm19b.v = _mm256_sub_pd(ymm19b.v, ymm6b.v); // {[beta5,12 ... beta5,15] - ymm6}: ymm19 -= ymm6

  ymm18a.v = _mm256_mul_pd(ymm5a.v, ymm18a.v);  // 1/alpha55 * ymm18
  ymm18b.v = _mm256_mul_pd(ymm5a.v, ymm18b.v);  // 1/alpha55 * ymm18
  ymm19a.v = _mm256_mul_pd(ymm5a.v, ymm19a.v);  // 1/alpha55 * ymm19
  ymm19b.v = _mm256_mul_pd(ymm5a.v, ymm19b.v);  // 1/alpha55 * ymm19

  // Store the result
  _mm256_storeu_pd(b11 + 20 * 4, ymm18a.v);   // store ( beta50 ... beta53 )  = ymm18
  _mm256_storeu_pd(b11 + 21 * 4, ymm18b.v);   // store ( beta54 ... beta57 )  = ymm18
  _mm256_storeu_pd(b11 + 22 * 4, ymm19a.v);  // store ( beta58 ... beta5,11 ) = ymm19
  _mm256_storeu_pd(b11 + 23 * 4, ymm19b.v);  // store ( beta5,12 ... beta5,15 ) = ymm19


  //#if 1 // VK
  // Store @ C11
  for (int i = 0; i < 4; i++)
    {
      c11[5 * rs_c + i*cs_c]          = ymm18a.d[i];  // store (gama50 ... gama57) = ymm18[0] ... ymm18[7])
      c11[5 * rs_c + (i + 4)*cs_c]    = ymm18b.d[i];  // store (gama50 ... gama57) = ymm18[0] ... ymm18[7])
      c11[5 * rs_c + (i + 8) * cs_c]  = ymm19a.d[i];  // store (gama58 ... gama5,15) = ymm19[0] ... ymm19[7])
      c11[5 * rs_c + (i + 12) * cs_c] = ymm19b.d[i];  // store (gama58 ... gama5,15) = ymm19[0] ... ymm19[7])
    }
 
  //#endif

} // End of the function
