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
#include "immintrin.h"
#include "xmmintrin.h"
#include "blis.h"

#ifdef BLIS_ADDON_LPGEMM

#include "lpgemm_kernel_macros_f32_avx2.h"

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_5x16)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    /*Declare the registers*/
    __m256 ymm0, ymm1, ymm2, ymm3;
    __m256 ymm4, ymm5, ymm6, ymm7;
    __m256 ymm8, ymm9, ymm10, ymm11;
    __m256 ymm12, ymm13;

    /* zero the accumulator registers */
    ZERO_ACC_YMM_4_REG(ymm4, ymm5, ymm6, ymm7);
    ZERO_ACC_YMM_4_REG(ymm8,  ymm9,  ymm10, ymm11);
    ymm12 = _mm256_setzero_ps();
    ymm13 = _mm256_setzero_ps();

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;

    /*_mm_prefetch( (MR X NR) from C*/
    _mm_prefetch((cbuf + 0*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 1*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 2*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 3*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 4*rs_c), _MM_HINT_T0);

    for(dim_t k = 0; k < k_iter; k++)
    {
      /*Load 16 elements from row0 of B*/
      ymm0 = _mm256_loadu_ps(bbuf );
      ymm1 = _mm256_loadu_ps(bbuf + 8);
      bbuf += rs_b;  //move b pointer to next row

      ymm2 = _mm256_broadcast_ss((abuf + 0*rs_a)); //broadcast c0r0
      ymm3 = _mm256_broadcast_ss((abuf + 1*rs_a)); //broadcast c0r1

      ymm4 = _mm256_fmadd_ps(ymm0, ymm2, ymm4);
      ymm5 = _mm256_fmadd_ps(ymm1, ymm2, ymm5);
      ymm6 = _mm256_fmadd_ps(ymm0, ymm3, ymm6);
      ymm7 = _mm256_fmadd_ps(ymm1, ymm3, ymm7);

      ymm2 = _mm256_broadcast_ss((abuf + 2*rs_a)); //broadcast c0r2 
      ymm3 = _mm256_broadcast_ss((abuf + 3*rs_a)); //broadcast c0r3

      ymm8 = _mm256_fmadd_ps(ymm0, ymm2, ymm8);
      ymm9 = _mm256_fmadd_ps(ymm1, ymm2, ymm9);
      ymm10 = _mm256_fmadd_ps(ymm0, ymm3, ymm10);
      ymm11 = _mm256_fmadd_ps(ymm1, ymm3, ymm11);

      ymm2 = _mm256_broadcast_ss((abuf + 4*rs_a)); //broadcast c0r4
      abuf += cs_a;  //move a pointer to next col
    
      ymm12 = _mm256_fmadd_ps(ymm0, ymm2, ymm12);
      ymm13 = _mm256_fmadd_ps(ymm1, ymm2, ymm13);
    }//kloop

    ymm0 = _mm256_broadcast_ss(&(alpha));
    ALPHA_MUL_ACC_YMM_4_REG(ymm4,ymm5,ymm6,ymm7,ymm0)
    ALPHA_MUL_ACC_YMM_4_REG(ymm8,ymm9,ymm10,ymm11,ymm0)
    ALPHA_MUL_ACC_YMM_4_REG(ymm12,ymm13,ymm2,ymm3,ymm0)

    //store output when beta=0
    if(beta == 0.0)
    {
      _mm256_storeu_ps(cbuf, ymm4); 
      _mm256_storeu_ps(cbuf + 8, ymm5);
      cbuf += rs_c;
      _mm256_storeu_ps(cbuf, ymm6); 
      _mm256_storeu_ps(cbuf + 8, ymm7);
      cbuf += rs_c;
      _mm256_storeu_ps(cbuf, ymm8); 
      _mm256_storeu_ps(cbuf + 8, ymm9);
      cbuf += rs_c;
      _mm256_storeu_ps(cbuf, ymm10); 
      _mm256_storeu_ps(cbuf + 8, ymm11);
      cbuf += rs_c;
      _mm256_storeu_ps(cbuf, ymm12); 
      _mm256_storeu_ps(cbuf + 8, ymm13);
    }else
    {
      //load c and multiply with beta and 
      //add to accumulator and store back
      ymm3 = _mm256_broadcast_ss(&(beta));

      F32_C_STORE_BNZ_8(cbuf,rs_c,ymm0,ymm3,ymm4)
      F32_C_STORE_BNZ_8(cbuf+8,rs_c,ymm1,ymm3,ymm5)
      cbuf += rs_c;
      F32_C_STORE_BNZ_8(cbuf,rs_c,ymm0,ymm3,ymm6)
      F32_C_STORE_BNZ_8(cbuf+8,rs_c,ymm1,ymm3,ymm7)
      cbuf += rs_c;
      F32_C_STORE_BNZ_8(cbuf,rs_c,ymm0,ymm3,ymm8)
      F32_C_STORE_BNZ_8(cbuf+8,rs_c,ymm1,ymm3,ymm9)
      cbuf += rs_c;
      F32_C_STORE_BNZ_8(cbuf,rs_c,ymm0,ymm3,ymm10)
      F32_C_STORE_BNZ_8(cbuf+8,rs_c,ymm1,ymm3,ymm11)
      cbuf += rs_c;
      F32_C_STORE_BNZ_8(cbuf,rs_c,ymm0,ymm3,ymm12)
      F32_C_STORE_BNZ_8(cbuf+8,rs_c,ymm1,ymm3,ymm13)
    }//betazero
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_4x16)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    /*Declare the registers*/
    __m256 ymm0, ymm1, ymm2, ymm3;
    __m256 ymm4, ymm5, ymm6, ymm7;
    __m256 ymm8, ymm9, ymm10, ymm11;

    /* zero the accumulator registers */
    ZERO_ACC_YMM_4_REG(ymm4, ymm5, ymm6, ymm7);
    ZERO_ACC_YMM_4_REG(ymm8, ymm9,  ymm10, ymm11);

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;

    /*_mm_prefetch( (MR X NR) from C*/
    _mm_prefetch((cbuf + 0*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 1*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 2*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 3*rs_c), _MM_HINT_T0);

    for(dim_t k = 0; k < k_iter; k++)
    {
      /*Load 16 elements from row0 of B*/
      ymm0 = _mm256_loadu_ps(bbuf );
      ymm1 = _mm256_loadu_ps(bbuf + 8);
      bbuf += rs_b;  //move b pointer to next row

      ymm2 = _mm256_broadcast_ss((abuf + 0*rs_a)); //broadcast c0r0
      ymm3 = _mm256_broadcast_ss((abuf + 1*rs_a)); //broadcast c0r1

      ymm4 = _mm256_fmadd_ps(ymm0, ymm2, ymm4);
      ymm5 = _mm256_fmadd_ps(ymm1, ymm2, ymm5);
      ymm6 = _mm256_fmadd_ps(ymm0, ymm3, ymm6);
      ymm7 = _mm256_fmadd_ps(ymm1, ymm3, ymm7);

      ymm2 = _mm256_broadcast_ss((abuf + 2*rs_a)); //broadcast c0r2 
      ymm3 = _mm256_broadcast_ss((abuf + 3*rs_a)); //broadcast c0r3

      ymm8 = _mm256_fmadd_ps(ymm0, ymm2, ymm8);
      ymm9 = _mm256_fmadd_ps(ymm1, ymm2, ymm9);
      ymm10 = _mm256_fmadd_ps(ymm0, ymm3, ymm10);
      ymm11 = _mm256_fmadd_ps(ymm1, ymm3, ymm11);

      abuf += cs_a;  //move a pointer to next col
    }//kloop

    ymm0 = _mm256_broadcast_ss(&(alpha));
    ALPHA_MUL_ACC_YMM_4_REG(ymm4,ymm5,ymm6,ymm7,ymm0)
    ALPHA_MUL_ACC_YMM_4_REG(ymm8,ymm9,ymm10,ymm11,ymm0)

    //store output when beta=0
    if(beta == 0.0)
    {
      _mm256_storeu_ps(cbuf, ymm4); 
      _mm256_storeu_ps(cbuf + 8, ymm5);
      cbuf += rs_c;
      _mm256_storeu_ps(cbuf, ymm6); 
      _mm256_storeu_ps(cbuf + 8, ymm7);
      cbuf += rs_c;
      _mm256_storeu_ps(cbuf, ymm8); 
      _mm256_storeu_ps(cbuf + 8, ymm9);
      cbuf += rs_c;
      _mm256_storeu_ps(cbuf, ymm10); 
      _mm256_storeu_ps(cbuf + 8, ymm11);      
    }else
    {
      //load c and multiply with beta and 
      //add to accumulator and store back
      ymm3 = _mm256_broadcast_ss(&(beta));

      F32_C_STORE_BNZ_8(cbuf,rs_c,ymm0,ymm3,ymm4)
      F32_C_STORE_BNZ_8(cbuf+8,rs_c,ymm1,ymm3,ymm5)
      cbuf += rs_c;
      F32_C_STORE_BNZ_8(cbuf,rs_c,ymm0,ymm3,ymm6)
      F32_C_STORE_BNZ_8(cbuf+8,rs_c,ymm1,ymm3,ymm7)
      cbuf += rs_c;
      F32_C_STORE_BNZ_8(cbuf,rs_c,ymm0,ymm3,ymm8)
      F32_C_STORE_BNZ_8(cbuf+8,rs_c,ymm1,ymm3,ymm9)
      cbuf += rs_c;
      F32_C_STORE_BNZ_8(cbuf,rs_c,ymm0,ymm3,ymm10)
      F32_C_STORE_BNZ_8(cbuf+8,rs_c,ymm1,ymm3,ymm11)
    }//betazero
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_3x16)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    /*Declare the registers*/
    __m256 ymm0, ymm1, ymm2, ymm3;
    __m256 ymm4, ymm5, ymm6, ymm7;
    __m256 ymm8, ymm9;

    /* zero the accumulator registers */
    ZERO_ACC_YMM_4_REG(ymm4, ymm5, ymm6, ymm7);
    ymm8 = _mm256_setzero_ps();
    ymm9 = _mm256_setzero_ps();

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;

    /*_mm_prefetch( (MR X NR) from C*/
    _mm_prefetch((cbuf + 0*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 1*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 2*rs_c), _MM_HINT_T0);

    for(dim_t k = 0; k < k_iter; k++)
    {
      /*Load 16 elements from row0 of B*/
      ymm0 = _mm256_loadu_ps(bbuf );
      ymm1 = _mm256_loadu_ps(bbuf + 8);
      bbuf += rs_b;  //move b pointer to next row

      ymm2 = _mm256_broadcast_ss((abuf + 0*rs_a)); //broadcast c0r0
      ymm3 = _mm256_broadcast_ss((abuf + 1*rs_a)); //broadcast c0r1

      ymm4 = _mm256_fmadd_ps(ymm0, ymm2, ymm4);
      ymm5 = _mm256_fmadd_ps(ymm1, ymm2, ymm5);
      ymm6 = _mm256_fmadd_ps(ymm0, ymm3, ymm6);
      ymm7 = _mm256_fmadd_ps(ymm1, ymm3, ymm7);

      ymm2 = _mm256_broadcast_ss((abuf + 2*rs_a)); //broadcast c0r2 
      ymm8 = _mm256_fmadd_ps(ymm0, ymm2, ymm8);
      ymm9 = _mm256_fmadd_ps(ymm1, ymm2, ymm9);

      abuf += cs_a;  //move a pointer to next col
    }//kloop

    ymm0 = _mm256_broadcast_ss(&(alpha));
    ALPHA_MUL_ACC_YMM_4_REG(ymm4,ymm5,ymm6,ymm7,ymm0)
    ALPHA_MUL_ACC_YMM_4_REG(ymm8,ymm9,ymm2,ymm3,ymm0)

    //store output when beta=0
    if(beta == 0.0)
    {
      _mm256_storeu_ps(cbuf, ymm4); 
      _mm256_storeu_ps(cbuf + 8, ymm5);
      cbuf += rs_c;
      _mm256_storeu_ps(cbuf, ymm6); 
      _mm256_storeu_ps(cbuf + 8, ymm7);
      cbuf += rs_c;
      _mm256_storeu_ps(cbuf, ymm8); 
      _mm256_storeu_ps(cbuf + 8, ymm9);
    }else
    {
      //load c and multiply with beta and 
      //add to accumulator and store back
      ymm3 = _mm256_broadcast_ss(&(beta));

      F32_C_STORE_BNZ_8(cbuf,rs_c,ymm0,ymm3,ymm4)
      F32_C_STORE_BNZ_8(cbuf+8,rs_c,ymm1,ymm3,ymm5)
      cbuf += rs_c;
      F32_C_STORE_BNZ_8(cbuf,rs_c,ymm0,ymm3,ymm6)
      F32_C_STORE_BNZ_8(cbuf+8,rs_c,ymm1,ymm3,ymm7)
      cbuf += rs_c;
      F32_C_STORE_BNZ_8(cbuf,rs_c,ymm0,ymm3,ymm8)
      F32_C_STORE_BNZ_8(cbuf+8,rs_c,ymm1,ymm3,ymm9)
    }//betazero
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_2x16)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    /*Declare the registers*/
    __m256 ymm0, ymm1, ymm2, ymm3;
    __m256 ymm4, ymm5, ymm6, ymm7;

    /* zero the accumulator registers */
    ZERO_ACC_YMM_4_REG(ymm4, ymm5, ymm6, ymm7);

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;

    /*_mm_prefetch( (MR X NR) from C*/
    _mm_prefetch((cbuf + 0*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 1*rs_c), _MM_HINT_T0);

    for(dim_t k = 0; k < k_iter; k++)
    {
      /*Load 16 elements from row0 of B*/
      ymm0 = _mm256_loadu_ps(bbuf );
      ymm1 = _mm256_loadu_ps(bbuf + 8);
      bbuf += rs_b;  //move b pointer to next row

      ymm2 = _mm256_broadcast_ss((abuf + 0*rs_a)); //broadcast c0r0
      ymm3 = _mm256_broadcast_ss((abuf + 1*rs_a)); //broadcast c0r1

      ymm4 = _mm256_fmadd_ps(ymm0, ymm2, ymm4);
      ymm5 = _mm256_fmadd_ps(ymm1, ymm2, ymm5);
      ymm6 = _mm256_fmadd_ps(ymm0, ymm3, ymm6);
      ymm7 = _mm256_fmadd_ps(ymm1, ymm3, ymm7);

      abuf += cs_a;  //move a pointer to next col
    }//kloop

    ymm0 = _mm256_broadcast_ss(&(alpha));
    ALPHA_MUL_ACC_YMM_4_REG(ymm4,ymm5,ymm6,ymm7,ymm0)

    //store output when beta=0
    if(beta == 0.0)
    {
      _mm256_storeu_ps(cbuf, ymm4); 
      _mm256_storeu_ps(cbuf + 8, ymm5);
      cbuf += rs_c;
      _mm256_storeu_ps(cbuf, ymm6); 
      _mm256_storeu_ps(cbuf + 8, ymm7);
    }else
    {
      //load c and multiply with beta and 
      //add to accumulator and store back
      ymm3 = _mm256_broadcast_ss(&(beta));

      F32_C_STORE_BNZ_8(cbuf,rs_c,ymm0,ymm3,ymm4)
      F32_C_STORE_BNZ_8(cbuf+8,rs_c,ymm1,ymm3,ymm5)
      cbuf += rs_c;
      F32_C_STORE_BNZ_8(cbuf,rs_c,ymm0,ymm3,ymm6)
      F32_C_STORE_BNZ_8(cbuf+8,rs_c,ymm1,ymm3,ymm7)
    }//betazero
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_1x16)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    /*Declare the registers*/
    __m256 ymm0, ymm1, ymm2, ymm3;
    __m256 ymm4, ymm5;

    /* zero the accumulator registers */
    ymm4 = _mm256_setzero_ps();
    ymm5 = _mm256_setzero_ps();

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;

    /*_mm_prefetch( (MR X NR) from C*/
    _mm_prefetch((cbuf + 0*rs_c), _MM_HINT_T0);

    for(dim_t k = 0; k < k_iter; k++)
    {
      /*Load 16 elements from row0 of B*/
      ymm0 = _mm256_loadu_ps(bbuf );
      ymm1 = _mm256_loadu_ps(bbuf + 8);
      bbuf += rs_b;  //move b pointer to next row

      ymm2 = _mm256_broadcast_ss((abuf + 0*rs_a)); //broadcast c0r0

      ymm4 = _mm256_fmadd_ps(ymm0, ymm2, ymm4);
      ymm5 = _mm256_fmadd_ps(ymm1, ymm2, ymm5);

      abuf += cs_a;  //move a pointer to next col
    }//kloop

    ymm0 = _mm256_broadcast_ss(&(alpha));
    ymm4 = _mm256_mul_ps(ymm4,ymm0);
    ymm5 = _mm256_mul_ps(ymm5,ymm0);

    //store output when beta=0
    if(beta == 0.0)
    {
      _mm256_storeu_ps(cbuf, ymm4); 
      _mm256_storeu_ps(cbuf + 8, ymm5);
      cbuf += rs_c;
    }else
    {
      //load c and multiply with beta and 
      //add to accumulator and store back
      ymm3 = _mm256_broadcast_ss(&(beta));

      F32_C_STORE_BNZ_8(cbuf,rs_c,ymm0,ymm3,ymm4)
      F32_C_STORE_BNZ_8(cbuf+8,rs_c,ymm1,ymm3,ymm5)
      cbuf += rs_c;
    }//betazero
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_5x8)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    /*Declare the registers*/
    __m256 ymm0, ymm2, ymm3;
    __m256 ymm4, ymm6, ymm8, ymm10;
    __m256 ymm12;
    
    /* zero the accumulator registers */
    ZERO_ACC_YMM_4_REG(ymm4, ymm6, ymm2, ymm3);
    ZERO_ACC_YMM_4_REG(ymm8, ymm10, ymm12, ymm0);    
      
    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;
      
    /*_mm_prefetch( (MR X NR) from C*/
    _mm_prefetch((cbuf + 0*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 1*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 2*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 3*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 4*rs_c), _MM_HINT_T0);

    for(dim_t k = 0; k < k_iter; k++)
    {
      /*Load 16 elements from row0 of B*/
      ymm0 = _mm256_loadu_ps(bbuf );
      bbuf += rs_b;  //move b pointer to next row

      ymm2 = _mm256_broadcast_ss((abuf + 0*rs_a)); //broadcast c0r0
      ymm3 = _mm256_broadcast_ss((abuf + 1*rs_a)); //broadcast c0r1  

      ymm4 = _mm256_fmadd_ps(ymm0, ymm2, ymm4);
      ymm6 = _mm256_fmadd_ps(ymm0, ymm3, ymm6);

      ymm2 = _mm256_broadcast_ss((abuf + 2*rs_a)); //broadcast c0r2 
      ymm3 = _mm256_broadcast_ss((abuf + 3*rs_a)); //broadcast c0r3

      ymm8 = _mm256_fmadd_ps(ymm0, ymm2, ymm8);
      ymm2 = _mm256_broadcast_ss((abuf + 4*rs_a)); //broadcast c0r4
      
      ymm10 = _mm256_fmadd_ps(ymm0, ymm3, ymm10);
      ymm12 = _mm256_fmadd_ps(ymm0, ymm2, ymm12);

      abuf += cs_a;  //move a pointer to next col
    }//kloop

    ymm0 = _mm256_broadcast_ss(&(alpha));
    ALPHA_MUL_ACC_YMM_4_REG(ymm4,ymm6,ymm8,ymm10,ymm0)
    ymm12 = _mm256_mul_ps(ymm12,ymm0);

    //store output when beta=0
    if(beta == 0.0)
    {
      _mm256_storeu_ps(cbuf, ymm4); 
      cbuf += rs_c;
      _mm256_storeu_ps(cbuf, ymm6); 
      cbuf += rs_c;
      _mm256_storeu_ps(cbuf, ymm8); 
      cbuf += rs_c;
      _mm256_storeu_ps(cbuf, ymm10); 
      cbuf += rs_c;
      _mm256_storeu_ps(cbuf, ymm12);
    }else
    {
      //load c and multiply with beta and 
      //add to accumulator and store back
      ymm3 = _mm256_broadcast_ss(&(beta));

      F32_C_STORE_BNZ_8(cbuf,rs_c,ymm0,ymm3,ymm4)
      cbuf += rs_c;
      F32_C_STORE_BNZ_8(cbuf,rs_c,ymm0,ymm3,ymm6)
      cbuf += rs_c;
      F32_C_STORE_BNZ_8(cbuf,rs_c,ymm0,ymm3,ymm8)
      cbuf += rs_c;
      F32_C_STORE_BNZ_8(cbuf,rs_c,ymm0,ymm3,ymm10)
      cbuf += rs_c;
      F32_C_STORE_BNZ_8(cbuf,rs_c,ymm0,ymm3,ymm12)
    }//betazero
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_4x8)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    /*Declare the registers*/
    __m256 ymm0, ymm2, ymm3;
    __m256 ymm4, ymm6, ymm8, ymm10;
    
    /* zero the accumulator registers */
    ZERO_ACC_YMM_4_REG(ymm4, ymm6, ymm8, ymm10);
      
    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;
      
    /*_mm_prefetch( (MR X NR) from C*/
    _mm_prefetch((cbuf + 0*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 1*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 2*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 3*rs_c), _MM_HINT_T0);

    for(dim_t k = 0; k < k_iter; k++)
    {
      /*Load 16 elements from row0 of B*/
      ymm0 = _mm256_loadu_ps(bbuf );
      bbuf += rs_b;  //move b pointer to next row

      ymm2 = _mm256_broadcast_ss((abuf + 0*rs_a)); //broadcast c0r0
      ymm3 = _mm256_broadcast_ss((abuf + 1*rs_a)); //broadcast c0r1  

      ymm4 = _mm256_fmadd_ps(ymm0, ymm2, ymm4);
      ymm6 = _mm256_fmadd_ps(ymm0, ymm3, ymm6);

      ymm2 = _mm256_broadcast_ss((abuf + 2*rs_a)); //broadcast c0r2 
      ymm3 = _mm256_broadcast_ss((abuf + 3*rs_a)); //broadcast c0r3

      ymm8 = _mm256_fmadd_ps(ymm0, ymm2, ymm8);
      ymm10 = _mm256_fmadd_ps(ymm0, ymm3, ymm10);

      abuf += cs_a;  //move a pointer to next col
    }//kloop

    ymm0 = _mm256_broadcast_ss(&(alpha));
    ALPHA_MUL_ACC_YMM_4_REG(ymm4,ymm6,ymm8,ymm10,ymm0)

    //store output when beta=0
    if(beta == 0.0)
    {
      _mm256_storeu_ps(cbuf, ymm4); 
      cbuf += rs_c;
      _mm256_storeu_ps(cbuf, ymm6); 
      cbuf += rs_c;
      _mm256_storeu_ps(cbuf, ymm8); 
      cbuf += rs_c;
      _mm256_storeu_ps(cbuf, ymm10);
    }else
    {
      //load c and multiply with beta and 
      //add to accumulator and store back
      ymm3 = _mm256_broadcast_ss(&(beta));

      F32_C_STORE_BNZ_8(cbuf,rs_c,ymm0,ymm3,ymm4)
      cbuf += rs_c;
      F32_C_STORE_BNZ_8(cbuf,rs_c,ymm0,ymm3,ymm6)
      cbuf += rs_c;
      F32_C_STORE_BNZ_8(cbuf,rs_c,ymm0,ymm3,ymm8)
      cbuf += rs_c;
      F32_C_STORE_BNZ_8(cbuf,rs_c,ymm0,ymm3,ymm10)
    }//betazero
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_3x8)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    /*Declare the registers*/
    __m256 ymm0, ymm2, ymm3;
    __m256 ymm4, ymm6, ymm8;

    /* zero the accumulator registers */
    ZERO_ACC_YMM_4_REG(ymm4, ymm6, ymm2, ymm8);

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;

    /*_mm_prefetch( (MR X NR) from C*/
    _mm_prefetch((cbuf + 0*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 1*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 2*rs_c), _MM_HINT_T0);

    for(dim_t k = 0; k < k_iter; k++)
    {
      /*Load 16 elements from row0 of B*/
      ymm0 = _mm256_loadu_ps(bbuf );
      bbuf += rs_b;  //move b pointer to next row

      ymm2 = _mm256_broadcast_ss((abuf + 0*rs_a)); //broadcast c0r0
      ymm3 = _mm256_broadcast_ss((abuf + 1*rs_a)); //broadcast c0r1  

      ymm4 = _mm256_fmadd_ps(ymm0, ymm2, ymm4);
      ymm2 = _mm256_broadcast_ss((abuf + 2*rs_a)); //broadcast c0r2 

      ymm6 = _mm256_fmadd_ps(ymm0, ymm3, ymm6);
      ymm8 = _mm256_fmadd_ps(ymm0, ymm2, ymm8);

      abuf += cs_a;  //move a pointer to next col
    }//kloop

    ymm0 = _mm256_broadcast_ss(&(alpha));
    ymm4 = _mm256_mul_ps(ymm4,ymm0);
    ymm6 = _mm256_mul_ps(ymm6,ymm0);
    ymm8 = _mm256_mul_ps(ymm8,ymm0);    

    //store output when beta=0
    if(beta == 0.0)
    {
      _mm256_storeu_ps(cbuf, ymm4); 
      cbuf += rs_c;
      _mm256_storeu_ps(cbuf, ymm6); 
      cbuf += rs_c;
      _mm256_storeu_ps(cbuf, ymm8);
    }else
    {
      //load c and multiply with beta and 
      //add to accumulator and store back
      ymm3 = _mm256_broadcast_ss(&(beta));
      F32_C_STORE_BNZ_8(cbuf,rs_c,ymm0,ymm3,ymm4)
      cbuf += rs_c;
      F32_C_STORE_BNZ_8(cbuf,rs_c,ymm0,ymm3,ymm6)
      cbuf += rs_c;
      F32_C_STORE_BNZ_8(cbuf,rs_c,ymm0,ymm3,ymm8)
    }//betazero
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_2x8)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    /*Declare the registers*/
    __m256 ymm0, ymm2, ymm3;
    __m256 ymm4, ymm6;

    /* zero the accumulator registers */
    ZERO_ACC_YMM_4_REG(ymm4, ymm6, ymm2, ymm3);

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;

    /*_mm_prefetch( (MR X NR) from C*/
    _mm_prefetch((cbuf + 0*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 1*rs_c), _MM_HINT_T0);

    for(dim_t k = 0; k < k_iter; k++)
    {
      /*Load 16 elements from row0 of B*/
      ymm0 = _mm256_loadu_ps(bbuf );
      bbuf += rs_b;  //move b pointer to next row

      ymm2 = _mm256_broadcast_ss((abuf + 0*rs_a)); //broadcast c0r0
      ymm3 = _mm256_broadcast_ss((abuf + 1*rs_a)); //broadcast c0r1  

      ymm4 = _mm256_fmadd_ps(ymm0, ymm2, ymm4);
      ymm6 = _mm256_fmadd_ps(ymm0, ymm3, ymm6);
        
      abuf += cs_a;  //move a pointer to next col
    }//kloop

    ymm0 = _mm256_broadcast_ss(&(alpha));
    ymm4 = _mm256_mul_ps(ymm4,ymm0);
    ymm6 = _mm256_mul_ps(ymm6,ymm0);

    //store output when beta=0
    if(beta == 0.0)
    {
      _mm256_storeu_ps(cbuf, ymm4); 
      cbuf += rs_c;
      _mm256_storeu_ps(cbuf, ymm6);
    }else
    {
      //load c and multiply with beta and 
      //add to accumulator and store back
      ymm3 = _mm256_broadcast_ss(&(beta));

      F32_C_STORE_BNZ_8(cbuf,rs_c,ymm0,ymm3,ymm4)
      cbuf += rs_c;
      F32_C_STORE_BNZ_8(cbuf,rs_c,ymm0,ymm3,ymm6)
    }//betazero
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_1x8)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    /*Declare the registers*/
    __m256 ymm0, ymm2, ymm3;
    __m256 ymm4;

    /* zero the accumulator registers */
    ymm4 = _mm256_setzero_ps();

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;

    /*_mm_prefetch( (MR X NR) from C*/
    _mm_prefetch((cbuf + 0*rs_c), _MM_HINT_T0);

    for(dim_t k = 0; k < k_iter; k++)
    {
      /*Load 16 elements from row0 of B*/
      ymm0 = _mm256_loadu_ps(bbuf );
      bbuf += rs_b;  //move b pointer to next row

      ymm2 = _mm256_broadcast_ss((abuf + 0*rs_a)); //broadcast c0r0
      ymm4 = _mm256_fmadd_ps(ymm0, ymm2, ymm4);

      abuf += cs_a;  //move a pointer to next col
    }//kloop

    ymm0 = _mm256_broadcast_ss(&(alpha));
    ymm4 = _mm256_mul_ps(ymm4,ymm0);

    //store output when beta=0
    if(beta == 0.0)
    {
      _mm256_storeu_ps(cbuf, ymm4); 
      cbuf += rs_c;
    }else
    {
      //load c and multiply with beta and 
      //add to accumulator and store back
      ymm3 = _mm256_broadcast_ss(&(beta));
      F32_C_STORE_BNZ_8(cbuf,rs_c,ymm0,ymm3,ymm4)
    }//betazero
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_5x4)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;
    
    /*Declare the registers*/
    __m128 xmm0, xmm1, xmm2, xmm3;
    __m128 xmm4, xmm5, xmm6, xmm7;
    __m128 xmm8, xmm9;
    
    /* zero the accumulator registers */
    ZERO_ACC_XMM_4_REG(xmm4,xmm5,xmm6,xmm7) 
    ZERO_ACC_XMM_4_REG(xmm8,xmm9,xmm0,xmm1) 
    
    /*_mm_prefetch( (MR X NR) from C*/
    _mm_prefetch((cbuf + 0*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 1*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 2*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 3*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 4*rs_c), _MM_HINT_T0);

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 16 elements from row0 of B*/
        xmm0 = _mm_loadu_ps(bbuf );
        bbuf += rs_b;  //move b pointer to next row

        xmm1 = _mm_broadcast_ss((abuf + 0*rs_a)); //broadcast c0r0
        xmm2 = _mm_broadcast_ss((abuf + 1*rs_a)); //broadcast c0r0
        xmm3 = _mm_broadcast_ss((abuf + 2*rs_a)); //broadcast c0r0

        xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);
        xmm5 = _mm_fmadd_ps(xmm0, xmm2, xmm5);
        xmm6 = _mm_fmadd_ps(xmm0, xmm3, xmm6);

        xmm1 = _mm_broadcast_ss((abuf + 3*rs_a)); //broadcast c0r0
        xmm2 = _mm_broadcast_ss((abuf + 4*rs_a)); //broadcast c0r0
        abuf += cs_a;  //move a pointer to next col

        xmm7 = _mm_fmadd_ps(xmm0, xmm1, xmm7);
        xmm8 = _mm_fmadd_ps(xmm0, xmm2, xmm8);
    }//kloop

    xmm0 = _mm_broadcast_ss(&(alpha));
    ALPHA_MUL_ACC_XMM_4_REG(xmm4,xmm5,xmm6,xmm7,xmm0) 
    ALPHA_MUL_ACC_XMM_4_REG(xmm8,xmm9,xmm2,xmm3,xmm0)

    //store output when beta=0
    if(beta == 0.0)
    {
        _mm_storeu_ps(cbuf, xmm4);
        cbuf += rs_c;
        _mm_storeu_ps(cbuf, xmm5);
        cbuf += rs_c;
        _mm_storeu_ps(cbuf, xmm6);
        cbuf += rs_c;
        _mm_storeu_ps(cbuf, xmm7);
        cbuf += rs_c;
        _mm_storeu_ps(cbuf, xmm8);
    }else
    {
        //load c and multiply with beta and 
        //add to accumulator and store back
        xmm3 = _mm_broadcast_ss(&(beta));

        F32_C_STORE_BNZ_4(cbuf,rs_c,xmm1,xmm3,xmm4)
        cbuf += rs_c;
        F32_C_STORE_BNZ_4(cbuf,rs_c,xmm1,xmm3,xmm5)
        cbuf += rs_c;
        F32_C_STORE_BNZ_4(cbuf,rs_c,xmm1,xmm3,xmm6)
        cbuf += rs_c;
        F32_C_STORE_BNZ_4(cbuf,rs_c,xmm1,xmm3,xmm7)
        cbuf += rs_c;
        F32_C_STORE_BNZ_4(cbuf,rs_c,xmm1,xmm3,xmm8)
    }//betazero
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_4x4)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;

    /*Declare the registers*/
    __m128 xmm0, xmm1, xmm2, xmm3;
    __m128 xmm4, xmm5, xmm6, xmm7;
    
    /* zero the accumulator registers */
    ZERO_ACC_XMM_4_REG(xmm4,xmm5,xmm6,xmm7) 
    
    /*_mm_prefetch( (MR X NR) from C*/
    _mm_prefetch((cbuf + 0*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 1*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 2*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 3*rs_c), _MM_HINT_T0);

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 16 elements from row0 of B*/
        xmm0 = _mm_loadu_ps(bbuf );
        bbuf += rs_b;  //move b pointer to next row

        xmm1 = _mm_broadcast_ss((abuf + 0*rs_a)); //broadcast c0r0
        xmm2 = _mm_broadcast_ss((abuf + 1*rs_a)); //broadcast c0r0
        xmm3 = _mm_broadcast_ss((abuf + 2*rs_a)); //broadcast c0r0

        xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);
        xmm5 = _mm_fmadd_ps(xmm0, xmm2, xmm5);
        xmm6 = _mm_fmadd_ps(xmm0, xmm3, xmm6);

        xmm1 = _mm_broadcast_ss((abuf + 3*rs_a)); //broadcast c0r0
        abuf += cs_a;  //move a pointer to next col

        xmm7 = _mm_fmadd_ps(xmm0, xmm1, xmm7);
    }//kloop

    xmm0 = _mm_broadcast_ss(&(alpha));
    ALPHA_MUL_ACC_XMM_4_REG(xmm4,xmm5,xmm6,xmm7,xmm0)

    //store output when beta=0
    if(beta == 0.0)
    {
        _mm_storeu_ps(cbuf, xmm4);
        cbuf += rs_c;
        _mm_storeu_ps(cbuf, xmm5);
        cbuf += rs_c;
        _mm_storeu_ps(cbuf, xmm6);
        cbuf += rs_c;
        _mm_storeu_ps(cbuf, xmm7);
    }else
    {
        //load c and multiply with beta and 
        //add to accumulator and store back
        xmm3 = _mm_broadcast_ss(&(beta));

        F32_C_STORE_BNZ_4(cbuf,rs_c,xmm1,xmm3,xmm4)
        cbuf += rs_c;
        F32_C_STORE_BNZ_4(cbuf,rs_c,xmm1,xmm3,xmm5)
        cbuf += rs_c;
        F32_C_STORE_BNZ_4(cbuf,rs_c,xmm1,xmm3,xmm6)
        cbuf += rs_c;
        F32_C_STORE_BNZ_4(cbuf,rs_c,xmm1,xmm3,xmm7)
    }//betazero
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_3x4)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;

    /*Declare the registers*/
    __m128 xmm0, xmm1, xmm2, xmm3;
    __m128 xmm4, xmm5, xmm6, xmm7;
    
    /* zero the accumulator registers */
    ZERO_ACC_XMM_4_REG(xmm4,xmm5,xmm6,xmm7) 
    
    /*_mm_prefetch( (MR X NR) from C*/
    _mm_prefetch((cbuf + 0*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 1*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 2*rs_c), _MM_HINT_T0);

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 16 elements from row0 of B*/
        xmm0 = _mm_loadu_ps(bbuf );
        bbuf += rs_b;  //move b pointer to next row

        xmm1 = _mm_broadcast_ss((abuf + 0*rs_a)); //broadcast c0r0
        xmm2 = _mm_broadcast_ss((abuf + 1*rs_a)); //broadcast c0r0
        xmm3 = _mm_broadcast_ss((abuf + 2*rs_a)); //broadcast c0r0
        abuf += cs_a;  //move a pointer to next col
        xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);
        xmm5 = _mm_fmadd_ps(xmm0, xmm2, xmm5);
        xmm6 = _mm_fmadd_ps(xmm0, xmm3, xmm6);
    }//kloop

    xmm0 = _mm_broadcast_ss(&(alpha));
    ALPHA_MUL_ACC_XMM_4_REG(xmm4,xmm5,xmm6,xmm7,xmm0)

    //store output when beta=0
    if(beta == 0.0)
    {
        _mm_storeu_ps(cbuf, xmm4);
        cbuf += rs_c;
        _mm_storeu_ps(cbuf, xmm5);
        cbuf += rs_c;
        _mm_storeu_ps(cbuf, xmm6);
    }else
    {
        //load c and multiply with beta and 
        //add to accumulator and store back
        xmm3 = _mm_broadcast_ss(&(beta));

        F32_C_STORE_BNZ_4(cbuf,rs_c,xmm1,xmm3,xmm4)
        cbuf += rs_c;
        F32_C_STORE_BNZ_4(cbuf,rs_c,xmm1,xmm3,xmm5)
        cbuf += rs_c;
        F32_C_STORE_BNZ_4(cbuf,rs_c,xmm1,xmm3,xmm6)
    }//betazero
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_2x4)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;

    /*Declare the registers*/
    __m128 xmm0, xmm1, xmm2, xmm3;
    __m128 xmm4, xmm5;

    /* zero the accumulator registers */
    xmm4 = _mm_setzero_ps();
    xmm5 = _mm_setzero_ps();
    
    /*_mm_prefetch( (MR X NR) from C*/
    _mm_prefetch((cbuf + 0*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 1*rs_c), _MM_HINT_T0);

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 16 elements from row0 of B*/
        xmm0 = _mm_loadu_ps(bbuf );
        bbuf += rs_b;  //move b pointer to next row

        xmm1 = _mm_broadcast_ss((abuf + 0*rs_a)); //broadcast c0r0
        xmm2 = _mm_broadcast_ss((abuf + 1*rs_a)); //broadcast c0r0
        abuf += cs_a;  //move a pointer to next col
        xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);
        xmm5 = _mm_fmadd_ps(xmm0, xmm2, xmm5);
    }//kloop

    xmm0 = _mm_broadcast_ss(&(alpha));
    xmm4 = _mm_mul_ps(xmm4,xmm0);
    xmm5 = _mm_mul_ps(xmm5,xmm0);

    //store output when beta=0
    if(beta == 0.0)
    {
        _mm_storeu_ps(cbuf, xmm4);
        cbuf += rs_c;
        _mm_storeu_ps(cbuf, xmm5);
    }else
    {
        //load c and multiply with beta and 
        //add to accumulator and store back
        xmm3 = _mm_broadcast_ss(&(beta));

        F32_C_STORE_BNZ_4(cbuf,rs_c,xmm0,xmm3,xmm4)
        cbuf += rs_c;
        F32_C_STORE_BNZ_4(cbuf,rs_c,xmm0,xmm3,xmm5)
    }//betazero
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_1x4)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;

    /*Declare the registers*/
    __m128 xmm0, xmm1, xmm3, xmm4;
    
    /* zero the accumulator registers */
    xmm4 = _mm_setzero_ps();
    
    /*_mm_prefetch( (MR X NR) from C*/
    _mm_prefetch((cbuf + 0*rs_c), _MM_HINT_T0);

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 16 elements from row0 of B*/
        xmm0 = _mm_loadu_ps(bbuf );
        bbuf += rs_b;  //move b pointer to next row
        xmm1 = _mm_broadcast_ss((abuf + 0*rs_a)); //broadcast c0r0
        abuf += cs_a;  //move a pointer to next col
        xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);
    }//kloop

    xmm0 = _mm_broadcast_ss(&(alpha));
    xmm4 = _mm_mul_ps(xmm4,xmm0);

    //store output when beta=0
    if(beta == 0.0)
    {
        _mm_storeu_ps(cbuf, xmm4);
    }else
    {
        //load c and multiply with beta and 
        //add to accumulator and store back
        xmm3 = _mm_broadcast_ss(&(beta));
        F32_C_STORE_BNZ_4(cbuf,rs_c,xmm0,xmm3,xmm4)
    }//betazero
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_5x2)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;

    /*Declare the registers*/
    __m128 xmm0, xmm1, xmm2, xmm3;
    __m128 xmm4, xmm5, xmm6, xmm7;
    __m128 xmm8, xmm9;
    
    /* zero the accumulator registers */
    ZERO_ACC_XMM_4_REG(xmm4,xmm5,xmm6,xmm7) 
    ZERO_ACC_XMM_4_REG(xmm8,xmm9,xmm0,xmm1) 
    
    /*_mm_prefetch( (MR X NR) from C*/
    _mm_prefetch((cbuf + 0*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 1*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 2*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 3*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 4*rs_c), _MM_HINT_T0);

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 16 elements from row0 of B*/
        xmm0 = _mm_load_sd((const double*)bbuf );
        bbuf += rs_b;  //move b pointer to next row

        xmm1 = _mm_broadcast_ss((abuf + 0*rs_a)); //broadcast c0r0
        xmm2 = _mm_broadcast_ss((abuf + 1*rs_a)); //broadcast c0r0
        xmm3 = _mm_broadcast_ss((abuf + 2*rs_a)); //broadcast c0r0

        xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);
        xmm5 = _mm_fmadd_ps(xmm0, xmm2, xmm5);
        xmm6 = _mm_fmadd_ps(xmm0, xmm3, xmm6);

        xmm1 = _mm_broadcast_ss((abuf + 3*rs_a)); //broadcast c0r0
        xmm2 = _mm_broadcast_ss((abuf + 4*rs_a)); //broadcast c0r0
        abuf += cs_a;  //move a pointer to next col

        xmm7 = _mm_fmadd_ps(xmm0, xmm1, xmm7);
        xmm8 = _mm_fmadd_ps(xmm0, xmm2, xmm8);
    }//kloop

    xmm0 = _mm_broadcast_ss(&(alpha));
    ALPHA_MUL_ACC_XMM_4_REG(xmm4,xmm5,xmm6,xmm7,xmm0) 
    ALPHA_MUL_ACC_XMM_4_REG(xmm8,xmm9,xmm2,xmm3,xmm0)

    //store output when beta=0
    if(beta == 0.0)
    {
        _mm_store_sd((double*)cbuf, xmm4);
        cbuf += rs_c;
        _mm_store_sd((double*)cbuf, xmm5);
        cbuf += rs_c;
        _mm_store_sd((double*)cbuf, xmm6);
        cbuf += rs_c;
        _mm_store_sd((double*)cbuf, xmm7);
        cbuf += rs_c;
        _mm_store_sd((double*)cbuf, xmm8);
    }else
    {
        //load c and multiply with beta and 
        //add to accumulator and store back
        xmm3 = _mm_broadcast_ss(&(beta));

        F32_C_STORE_BNZ_2(cbuf,rs_c,xmm1,xmm3,xmm4)
        cbuf += rs_c;
        F32_C_STORE_BNZ_2(cbuf,rs_c,xmm1,xmm3,xmm5)
        cbuf += rs_c;
        F32_C_STORE_BNZ_2(cbuf,rs_c,xmm1,xmm3,xmm6)
        cbuf += rs_c;
        F32_C_STORE_BNZ_2(cbuf,rs_c,xmm1,xmm3,xmm7)
        cbuf += rs_c;
        F32_C_STORE_BNZ_2(cbuf,rs_c,xmm1,xmm3,xmm8)
    }//betazero
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_4x2)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;

    /*Declare the registers*/
    __m128 xmm0, xmm1, xmm2, xmm3;
    __m128 xmm4, xmm5, xmm6, xmm7;
    
    /* zero the accumulator registers */
    ZERO_ACC_XMM_4_REG(xmm4,xmm5,xmm6,xmm7) 
    
    /*_mm_prefetch( (MR X NR) from C*/
    _mm_prefetch((cbuf + 0*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 1*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 2*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 3*rs_c), _MM_HINT_T0);

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 16 elements from row0 of B*/
        xmm0 = _mm_load_sd((const double*)bbuf );
        bbuf += rs_b;  //move b pointer to next row

        xmm1 = _mm_broadcast_ss((abuf + 0*rs_a)); //broadcast c0r0
        xmm2 = _mm_broadcast_ss((abuf + 1*rs_a)); //broadcast c0r0
        xmm3 = _mm_broadcast_ss((abuf + 2*rs_a)); //broadcast c0r0

        xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);
        xmm5 = _mm_fmadd_ps(xmm0, xmm2, xmm5);
        xmm6 = _mm_fmadd_ps(xmm0, xmm3, xmm6);

        xmm1 = _mm_broadcast_ss((abuf + 3*rs_a)); //broadcast c0r0
        abuf += cs_a;  //move a pointer to next col

        xmm7 = _mm_fmadd_ps(xmm0, xmm1, xmm7);
    }//kloop

    xmm0 = _mm_broadcast_ss(&(alpha));
    ALPHA_MUL_ACC_XMM_4_REG(xmm4,xmm5,xmm6,xmm7,xmm0)

    //store output when beta=0
    if(beta == 0.0)
    {
        _mm_store_sd((double*)cbuf, xmm4);
        cbuf += rs_c;
        _mm_store_sd((double*)cbuf, xmm5);
        cbuf += rs_c;
        _mm_store_sd((double*)cbuf, xmm6);
        cbuf += rs_c;
        _mm_storeu_ps(cbuf, xmm7);
    }else
    {
        //load c and multiply with beta and 
        //add to accumulator and store back
        xmm3 = _mm_broadcast_ss(&(beta));

        F32_C_STORE_BNZ_2(cbuf,rs_c,xmm1,xmm3,xmm4)
        cbuf += rs_c;
        F32_C_STORE_BNZ_2(cbuf,rs_c,xmm1,xmm3,xmm5)
        cbuf += rs_c;
        F32_C_STORE_BNZ_2(cbuf,rs_c,xmm1,xmm3,xmm6)
        cbuf += rs_c;
        F32_C_STORE_BNZ_2(cbuf,rs_c,xmm1,xmm3,xmm7)
    }//betazero
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_3x2)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;

    /*Declare the registers*/
    __m128 xmm0, xmm1, xmm2, xmm3;
    __m128 xmm4, xmm5, xmm6, xmm7;
    
    /* zero the accumulator registers */
    ZERO_ACC_XMM_4_REG(xmm4,xmm5,xmm6,xmm7) 
    
    /*_mm_prefetch( (MR X NR) from C*/
    _mm_prefetch((cbuf + 0*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 1*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 2*rs_c), _MM_HINT_T0);

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 16 elements from row0 of B*/
        xmm0 = _mm_load_sd((const double*)bbuf );
        bbuf += rs_b;  //move b pointer to next row

        xmm1 = _mm_broadcast_ss((abuf + 0*rs_a)); //broadcast c0r0
        xmm2 = _mm_broadcast_ss((abuf + 1*rs_a)); //broadcast c0r0
        xmm3 = _mm_broadcast_ss((abuf + 2*rs_a)); //broadcast c0r0
        abuf += cs_a;  //move a pointer to next col
        xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);
        xmm5 = _mm_fmadd_ps(xmm0, xmm2, xmm5);
        xmm6 = _mm_fmadd_ps(xmm0, xmm3, xmm6);
    }//kloop

    xmm0 = _mm_broadcast_ss(&(alpha));
    ALPHA_MUL_ACC_XMM_4_REG(xmm4,xmm5,xmm6,xmm7,xmm0)

    //store output when beta=0
    if(beta == 0.0)
    {
        _mm_store_sd((double*)cbuf, xmm4);
        cbuf += rs_c;
        _mm_store_sd((double*)cbuf, xmm5);
        cbuf += rs_c;
        _mm_store_sd((double*)cbuf, xmm6);
    }else
    {
        //load c and multiply with beta and 
        //add to accumulator and store back
        xmm3 = _mm_broadcast_ss(&(beta));

        F32_C_STORE_BNZ_2(cbuf,rs_c,xmm1,xmm3,xmm4)
        cbuf += rs_c;
        F32_C_STORE_BNZ_2(cbuf,rs_c,xmm1,xmm3,xmm5)
        cbuf += rs_c;
        F32_C_STORE_BNZ_2(cbuf,rs_c,xmm1,xmm3,xmm6)
    }//betazero
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_2x2)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;

    /*Declare the registers*/
    __m128 xmm0, xmm1, xmm2, xmm3;
    __m128 xmm4, xmm5;

    /* zero the accumulator registers */
    xmm4 = _mm_setzero_ps();
    xmm5 = _mm_setzero_ps();
    
    /*_mm_prefetch( (MR X NR) from C*/
    _mm_prefetch((cbuf + 0*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 1*rs_c), _MM_HINT_T0);

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 16 elements from row0 of B*/
        xmm0 = _mm_load_sd((const double*)bbuf );
        bbuf += rs_b;  //move b pointer to next row

        xmm1 = _mm_broadcast_ss((abuf + 0*rs_a)); //broadcast c0r0
        xmm2 = _mm_broadcast_ss((abuf + 1*rs_a)); //broadcast c0r0
        abuf += cs_a;  //move a pointer to next col
        xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);
        xmm5 = _mm_fmadd_ps(xmm0, xmm2, xmm5);
    }//kloop

    xmm0 = _mm_broadcast_ss(&(alpha));
    xmm4 = _mm_mul_ps(xmm4,xmm0);
    xmm5 = _mm_mul_ps(xmm5,xmm0);

    //store output when beta=0
    if(beta == 0.0)
    {
        _mm_store_sd((double*)cbuf, xmm4);
        cbuf += rs_c;
        _mm_store_sd((double*)cbuf, xmm5);
    }else
    {
        //load c and multiply with beta and 
        //add to accumulator and store back
        xmm3 = _mm_broadcast_ss(&(beta));

        F32_C_STORE_BNZ_2(cbuf,rs_c,xmm0,xmm3,xmm4)
        cbuf += rs_c;
        F32_C_STORE_BNZ_2(cbuf,rs_c,xmm0,xmm3,xmm5)
    }//betazero
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_1x2)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;

    /*Declare the registers*/
    __m128 xmm0, xmm1, xmm3, xmm4;
    
    /* zero the accumulator registers */
    xmm4 = _mm_setzero_ps();
    
    /*_mm_prefetch( (MR X NR) from C*/
    _mm_prefetch((cbuf + 0*rs_c), _MM_HINT_T0);

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 16 elements from row0 of B*/
        xmm0 = _mm_load_sd((const double*)bbuf );
        bbuf += rs_b;  //move b pointer to next row
        xmm1 = _mm_broadcast_ss((abuf + 0*rs_a)); //broadcast c0r0
        abuf += cs_a;  //move a pointer to next col
        xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);
    }//kloop

    xmm0 = _mm_broadcast_ss(&(alpha));
    xmm4 = _mm_mul_ps(xmm4,xmm0);

    //store output when beta=0
    if(beta == 0.0)
    {
        _mm_store_sd((double*)cbuf, xmm4);
    }else
    {
        //load c and multiply with beta and 
        //add to accumulator and store back
        xmm3 = _mm_broadcast_ss(&(beta));
        F32_C_STORE_BNZ_2(cbuf,rs_c,xmm0,xmm3,xmm4)
    }//betazero
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_5x1)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;

    /*Declare the registers*/
    __m128 xmm0, xmm1, xmm2, xmm3;
    __m128 xmm4, xmm5, xmm6, xmm7;
    __m128 xmm8, xmm9;
    
    /* zero the accumulator registers */
    ZERO_ACC_XMM_4_REG(xmm4,xmm5,xmm6,xmm7) 
    ZERO_ACC_XMM_4_REG(xmm8,xmm9,xmm0,xmm1) 
    
    /*_mm_prefetch( (MR X NR) from C*/
    _mm_prefetch((cbuf + 0*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 1*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 2*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 3*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 4*rs_c), _MM_HINT_T0);

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 16 elements from row0 of B*/
        xmm0 = _mm_load_ss( bbuf );
        bbuf += rs_b;  //move b pointer to next row

        xmm1 = _mm_broadcast_ss((abuf + 0*rs_a)); //broadcast c0r0
        xmm2 = _mm_broadcast_ss((abuf + 1*rs_a)); //broadcast c0r0
        xmm3 = _mm_broadcast_ss((abuf + 2*rs_a)); //broadcast c0r0

        xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);
        xmm5 = _mm_fmadd_ps(xmm0, xmm2, xmm5);
        xmm6 = _mm_fmadd_ps(xmm0, xmm3, xmm6);

        xmm1 = _mm_broadcast_ss((abuf + 3*rs_a)); //broadcast c0r0
        xmm2 = _mm_broadcast_ss((abuf + 4*rs_a)); //broadcast c0r0
        abuf += cs_a;  //move a pointer to next col

        xmm7 = _mm_fmadd_ps(xmm0, xmm1, xmm7);
        xmm8 = _mm_fmadd_ps(xmm0, xmm2, xmm8);
    }//kloop

    xmm0 = _mm_broadcast_ss(&(alpha));
    ALPHA_MUL_ACC_XMM_4_REG(xmm4,xmm5,xmm6,xmm7,xmm0) 
    ALPHA_MUL_ACC_XMM_4_REG(xmm8,xmm9,xmm2,xmm3,xmm0)

    //store output when beta=0
    if(beta == 0.0)
    {
        _mm_store_ss(cbuf, xmm4);
        cbuf += rs_c;
        _mm_store_ss(cbuf, xmm5);
        cbuf += rs_c;
        _mm_store_ss(cbuf, xmm6);
        cbuf += rs_c;
        _mm_store_ss(cbuf, xmm7);
        cbuf += rs_c;
        _mm_store_ss(cbuf, xmm8);
    }else
    {
        //load c and multiply with beta and 
        //add to accumulator and store back
        xmm3 = _mm_broadcast_ss(&(beta));

        F32_C_STORE_BNZ_1(cbuf,rs_c,xmm1,xmm3,xmm4)
        cbuf += rs_c;
        F32_C_STORE_BNZ_1(cbuf,rs_c,xmm1,xmm3,xmm5)
        cbuf += rs_c;
        F32_C_STORE_BNZ_1(cbuf,rs_c,xmm1,xmm3,xmm6)
        cbuf += rs_c;
        F32_C_STORE_BNZ_1(cbuf,rs_c,xmm1,xmm3,xmm7)
        cbuf += rs_c;
        F32_C_STORE_BNZ_1(cbuf,rs_c,xmm1,xmm3,xmm8)
    }//betazero
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_4x1)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;

    /*Declare the registers*/
    __m128 xmm0, xmm1, xmm2, xmm3;
    __m128 xmm4, xmm5, xmm6, xmm7;
    
    /* zero the accumulator registers */
    ZERO_ACC_XMM_4_REG(xmm4,xmm5,xmm6,xmm7) 
    
    /*_mm_prefetch( (MR X NR) from C*/
    _mm_prefetch((cbuf + 0*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 1*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 2*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 3*rs_c), _MM_HINT_T0);

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 16 elements from row0 of B*/
        xmm0 = _mm_load_ss( bbuf );
        bbuf += rs_b;  //move b pointer to next row

        xmm1 = _mm_broadcast_ss((abuf + 0*rs_a)); //broadcast c0r0
        xmm2 = _mm_broadcast_ss((abuf + 1*rs_a)); //broadcast c0r0
        xmm3 = _mm_broadcast_ss((abuf + 2*rs_a)); //broadcast c0r0

        xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);
        xmm5 = _mm_fmadd_ps(xmm0, xmm2, xmm5);
        xmm6 = _mm_fmadd_ps(xmm0, xmm3, xmm6);

        xmm1 = _mm_broadcast_ss((abuf + 3*rs_a)); //broadcast c0r0
        abuf += cs_a;  //move a pointer to next col

        xmm7 = _mm_fmadd_ps(xmm0, xmm1, xmm7);
    }//kloop

    xmm0 = _mm_broadcast_ss(&(alpha));
    ALPHA_MUL_ACC_XMM_4_REG(xmm4,xmm5,xmm6,xmm7,xmm0)

    //store output when beta=0
    if(beta == 0.0)
    {
        _mm_store_ss(cbuf, xmm4);
        cbuf += rs_c;
        _mm_store_ss(cbuf, xmm5);
        cbuf += rs_c;
        _mm_store_ss(cbuf, xmm6);
        cbuf += rs_c;
        _mm_store_ss(cbuf, xmm7);
    }else
    {
        //load c and multiply with beta and 
        //add to accumulator and store back
        xmm3 = _mm_broadcast_ss(&(beta));

        F32_C_STORE_BNZ_1(cbuf,rs_c,xmm1,xmm3,xmm4)
        cbuf += rs_c;
        F32_C_STORE_BNZ_1(cbuf,rs_c,xmm1,xmm3,xmm5)
        cbuf += rs_c;
        F32_C_STORE_BNZ_1(cbuf,rs_c,xmm1,xmm3,xmm6)
        cbuf += rs_c;
        F32_C_STORE_BNZ_1(cbuf,rs_c,xmm1,xmm3,xmm7)
    }//betazero
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_3x1)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;

    /*Declare the registers*/
    __m128 xmm0, xmm1, xmm2, xmm3;
    __m128 xmm4, xmm5, xmm6, xmm7;
    
    /* zero the accumulator registers */
    ZERO_ACC_XMM_4_REG(xmm4,xmm5,xmm6,xmm7) 
    
    /*_mm_prefetch( (MR X NR) from C*/
    _mm_prefetch((cbuf + 0*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 1*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 2*rs_c), _MM_HINT_T0);

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 16 elements from row0 of B*/
        xmm0 = _mm_load_ss( bbuf );
        bbuf += rs_b;  //move b pointer to next row

        xmm1 = _mm_broadcast_ss((abuf + 0*rs_a)); //broadcast c0r0
        xmm2 = _mm_broadcast_ss((abuf + 1*rs_a)); //broadcast c0r0
        xmm3 = _mm_broadcast_ss((abuf + 2*rs_a)); //broadcast c0r0
        abuf += cs_a;  //move a pointer to next col
        xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);
        xmm5 = _mm_fmadd_ps(xmm0, xmm2, xmm5);
        xmm6 = _mm_fmadd_ps(xmm0, xmm3, xmm6);
    }//kloop

    xmm0 = _mm_broadcast_ss(&(alpha));
    ALPHA_MUL_ACC_XMM_4_REG(xmm4,xmm5,xmm6,xmm7,xmm0)

    //store output when beta=0
    if(beta == 0.0)
    {
        _mm_store_ss(cbuf, xmm4);
        cbuf += rs_c;
        _mm_store_ss(cbuf, xmm5);
        cbuf += rs_c;
        _mm_store_ss(cbuf, xmm6);
    }else
    {
        //load c and multiply with beta and 
        //add to accumulator and store back
        xmm3 = _mm_broadcast_ss(&(beta));

        F32_C_STORE_BNZ_1(cbuf,rs_c,xmm1,xmm3,xmm4)
        cbuf += rs_c;
        F32_C_STORE_BNZ_1(cbuf,rs_c,xmm1,xmm3,xmm5)
        cbuf += rs_c;
        F32_C_STORE_BNZ_1(cbuf,rs_c,xmm1,xmm3,xmm6)
    }//betazero
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_2x1)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;

    /*Declare the registers*/
    __m128 xmm0, xmm1, xmm2, xmm3;
    __m128 xmm4, xmm5;

    /* zero the accumulator registers */
    xmm4 = _mm_setzero_ps();
    xmm5 = _mm_setzero_ps();
    
    /*_mm_prefetch( (MR X NR) from C*/
    _mm_prefetch((cbuf + 0*rs_c), _MM_HINT_T0);
    _mm_prefetch((cbuf + 1*rs_c), _MM_HINT_T0);

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 16 elements from row0 of B*/
        xmm0 = _mm_load_ss( bbuf );
        bbuf += rs_b;  //move b pointer to next row

        xmm1 = _mm_broadcast_ss((abuf + 0*rs_a)); //broadcast c0r0
        xmm2 = _mm_broadcast_ss((abuf + 1*rs_a)); //broadcast c0r0
        abuf += cs_a;  //move a pointer to next col
        xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);
        xmm5 = _mm_fmadd_ps(xmm0, xmm2, xmm5);
    }//kloop

    xmm0 = _mm_broadcast_ss(&(alpha));
    xmm4 = _mm_mul_ps(xmm4,xmm0);
    xmm5 = _mm_mul_ps(xmm5,xmm0);

    //store output when beta=0
    if(beta == 0.0)
    {
        _mm_store_ss(cbuf, xmm4);
        cbuf += rs_c;
        _mm_store_ss(cbuf, xmm5);
    }else
    {
        //load c and multiply with beta and 
        //add to accumulator and store back
        xmm3 = _mm_broadcast_ss(&(beta));

        F32_C_STORE_BNZ_1(cbuf,rs_c,xmm0,xmm3,xmm4)
        cbuf += rs_c;
        F32_C_STORE_BNZ_1(cbuf,rs_c,xmm0,xmm3,xmm5)
    }//betazero
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_1x1)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;

    /*Declare the registers*/
    __m128 xmm0, xmm1, xmm3, xmm4;
    
    /* zero the accumulator registers */
    xmm4 = _mm_setzero_ps();
    
    /*_mm_prefetch( (MR X NR) from C*/
    _mm_prefetch((cbuf + 0*rs_c), _MM_HINT_T0);

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 16 elements from row0 of B*/
        xmm0 = _mm_load_ss( bbuf );
        bbuf += rs_b;  //move b pointer to next row
        xmm1 = _mm_broadcast_ss((abuf + 0*rs_a)); //broadcast c0r0
        abuf += cs_a;  //move a pointer to next col
        xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);
    }//kloop

    xmm0 = _mm_broadcast_ss(&(alpha));
    xmm4 = _mm_mul_ps(xmm4,xmm0);

    //store output when beta=0
    if(beta == 0.0)
    {
        _mm_store_ss(cbuf, xmm4);
    }else
    {
        //load c and multiply with beta and 
        //add to accumulator and store back
        xmm3 = _mm_broadcast_ss(&(beta));
        F32_C_STORE_BNZ_1(cbuf,rs_c,xmm0,xmm3,xmm4)
    }//betazero
}
#endif
