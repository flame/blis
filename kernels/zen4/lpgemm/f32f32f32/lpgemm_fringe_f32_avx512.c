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

#include "lpgemm_kernel_macros_f32.h"

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_5x64)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k0;

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7;
    __m512 zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15;
    __m512 zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23;
    __m512 zmm24, zmm25, zmm26, zmm27;
    
    /* zero the accumulator registers */
    ZERO_ACC_ZMM_4_REG(zmm8, zmm9, zmm10, zmm11);
    ZERO_ACC_ZMM_4_REG(zmm12, zmm13, zmm14, zmm15);
    ZERO_ACC_ZMM_4_REG(zmm16, zmm17, zmm18, zmm19);
    ZERO_ACC_ZMM_4_REG(zmm20, zmm21, zmm22, zmm23);
    ZERO_ACC_ZMM_4_REG(zmm24, zmm25, zmm26, zmm27);

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 32 elements from row0 of B*/
        zmm0 = _mm512_loadu_ps (bbuf );     //load 0-15 values from current row 
        zmm1 = _mm512_loadu_ps (bbuf + 16); //load 16-31 values from current row

        /*Load Next 32 elements from row0 of B*/
        zmm6 = _mm512_loadu_ps (bbuf + 32); //load 32-47 from current row 
        zmm7 = _mm512_loadu_ps (bbuf + 48); //load 48-63 from current row
        
        /*Broadcast col0 elements of 12 rows of A*/
        zmm2 = _mm512_set1_ps(*(abuf + 0*rs_a)); //broadcast c0r0
        zmm3 = _mm512_set1_ps(*(abuf + 1*rs_a)); //broadcast c0r1  
        zmm4 = _mm512_set1_ps(*(abuf + 2*rs_a)); //broadcast c0r2 
        zmm5 = _mm512_set1_ps(*(abuf + 3*rs_a)); //broadcast c0r3

        zmm8 = _mm512_fmadd_ps(zmm0, zmm2, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm2, zmm9);
        zmm10 = _mm512_fmadd_ps(zmm6, zmm2, zmm10);
        zmm11 = _mm512_fmadd_ps(zmm7, zmm2, zmm11);

        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);
        zmm14 = _mm512_fmadd_ps(zmm6, zmm3, zmm14);
        zmm15 = _mm512_fmadd_ps(zmm7, zmm3, zmm15);

        zmm2 = _mm512_set1_ps(*(abuf + 4*rs_a)); //broadcast c0r4
        
        zmm16 = _mm512_fmadd_ps(zmm0, zmm4, zmm16);
        zmm17 = _mm512_fmadd_ps(zmm1, zmm4, zmm17);
        zmm18 = _mm512_fmadd_ps(zmm6, zmm4, zmm18);
        zmm19 = _mm512_fmadd_ps(zmm7, zmm4, zmm19);
        
        zmm20 = _mm512_fmadd_ps(zmm0, zmm5, zmm20);
        zmm21 = _mm512_fmadd_ps(zmm1, zmm5, zmm21);
        zmm22 = _mm512_fmadd_ps(zmm6, zmm5, zmm22);
        zmm23 = _mm512_fmadd_ps(zmm7, zmm5, zmm23);

        zmm24 = _mm512_fmadd_ps(zmm0, zmm2, zmm24);
        zmm25 = _mm512_fmadd_ps(zmm1, zmm2, zmm25);
        zmm26 = _mm512_fmadd_ps(zmm6, zmm2, zmm26);
        zmm27 = _mm512_fmadd_ps(zmm7, zmm2, zmm27);

        bbuf += rs_b;  //move b pointer to next row
        abuf += cs_a;  //move a pointer to next col

    }//kloop

    zmm0 = _mm512_set1_ps(alpha);
    ALPHA_MUL_ACC_ZMM_4_REG(zmm8,  zmm9,  zmm10, zmm11, zmm0);
    ALPHA_MUL_ACC_ZMM_4_REG(zmm12, zmm13, zmm14, zmm15, zmm0);
    ALPHA_MUL_ACC_ZMM_4_REG(zmm16, zmm17, zmm18, zmm19, zmm0);
    ALPHA_MUL_ACC_ZMM_4_REG(zmm20, zmm21, zmm22, zmm23, zmm0);
    ALPHA_MUL_ACC_ZMM_4_REG(zmm24, zmm25, zmm26, zmm27, zmm0);

      //store output when beta=0
    if(beta == 0.0)
    {
        _mm512_storeu_ps(cbuf, zmm8); 
        _mm512_storeu_ps(cbuf + 16, zmm9);
        _mm512_storeu_ps(cbuf + 32, zmm10);
        _mm512_storeu_ps(cbuf + 48, zmm11);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm12);
        _mm512_storeu_ps(cbuf + 16, zmm13);
        _mm512_storeu_ps(cbuf + 32, zmm14);
        _mm512_storeu_ps(cbuf + 48, zmm15);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm16);
        _mm512_storeu_ps(cbuf + 16, zmm17);
        _mm512_storeu_ps(cbuf + 32, zmm18);
        _mm512_storeu_ps(cbuf + 48, zmm19);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm20);
        _mm512_storeu_ps(cbuf + 16, zmm21);
        _mm512_storeu_ps(cbuf + 32, zmm22);
        _mm512_storeu_ps(cbuf + 48, zmm23);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm24);
        _mm512_storeu_ps(cbuf + 16, zmm25);
        _mm512_storeu_ps(cbuf + 32, zmm26);
        _mm512_storeu_ps(cbuf + 48, zmm27);
        //cbuf += rs_c;
    }else
    { 
        //load c and multiply with beta and 
        //add to accumulator and store back
        zmm3 = _mm512_set1_ps(beta);

        zmm0 = _mm512_load_ps(cbuf);
        zmm1 = _mm512_load_ps(cbuf + 16);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);
        _mm512_storeu_ps(cbuf, zmm8); 
        _mm512_storeu_ps(cbuf + 16, zmm9);
        zmm0 = _mm512_load_ps(cbuf + 32);
        zmm1 = _mm512_load_ps(cbuf + 48);
        zmm10 = _mm512_fmadd_ps(zmm0, zmm3, zmm10);
        zmm11 = _mm512_fmadd_ps(zmm1, zmm3, zmm11);
        _mm512_storeu_ps(cbuf + 32, zmm10);
        _mm512_storeu_ps(cbuf + 48, zmm11);
        cbuf += rs_c;

        zmm0 = _mm512_load_ps(cbuf);
        zmm1 = _mm512_load_ps(cbuf + 16);
        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);
        _mm512_storeu_ps(cbuf, zmm12);
        _mm512_storeu_ps(cbuf + 16, zmm13);
        zmm0 = _mm512_load_ps(cbuf + 32);
        zmm1 = _mm512_load_ps(cbuf + 48);
        zmm14 = _mm512_fmadd_ps(zmm0, zmm3, zmm14);
        zmm15 = _mm512_fmadd_ps(zmm1, zmm3, zmm15);
        _mm512_storeu_ps(cbuf + 32, zmm14);
        _mm512_storeu_ps(cbuf + 48, zmm15);
        cbuf += rs_c;

        zmm0 = _mm512_load_ps(cbuf);
        zmm1 = _mm512_load_ps(cbuf+16);
        zmm16 = _mm512_fmadd_ps(zmm0, zmm3, zmm16);
        zmm17 = _mm512_fmadd_ps(zmm1, zmm3, zmm17);
        _mm512_storeu_ps(cbuf, zmm16);
        _mm512_storeu_ps(cbuf + 16, zmm17);
        zmm0 = _mm512_load_ps(cbuf + 32);
        zmm1 = _mm512_load_ps(cbuf + 48);
        zmm18 = _mm512_fmadd_ps(zmm0, zmm3, zmm18);
        zmm19 = _mm512_fmadd_ps(zmm1, zmm3, zmm19);
        _mm512_storeu_ps(cbuf + 32, zmm18);
        _mm512_storeu_ps(cbuf + 48, zmm19);
        cbuf += rs_c;

        zmm0 = _mm512_load_ps(cbuf);
        zmm1 = _mm512_load_ps(cbuf+16);
        zmm20 = _mm512_fmadd_ps(zmm0, zmm3, zmm20);
        zmm21 = _mm512_fmadd_ps(zmm1, zmm3, zmm21);
        _mm512_storeu_ps(cbuf, zmm20);
        _mm512_storeu_ps(cbuf + 16, zmm21);
        zmm0 = _mm512_load_ps(cbuf + 32);
        zmm1 = _mm512_load_ps(cbuf + 48);
        zmm22 = _mm512_fmadd_ps(zmm0, zmm3, zmm22);
        zmm23 = _mm512_fmadd_ps(zmm1, zmm3, zmm23);
        _mm512_storeu_ps(cbuf + 32, zmm22);
        _mm512_storeu_ps(cbuf + 48, zmm23);
        cbuf += rs_c;

        zmm0 = _mm512_load_ps(cbuf);
        zmm1 = _mm512_load_ps(cbuf+16);
        zmm24 = _mm512_fmadd_ps(zmm0, zmm3, zmm24);
        zmm25 = _mm512_fmadd_ps(zmm1, zmm3, zmm25);
        _mm512_storeu_ps(cbuf, zmm24);
        _mm512_storeu_ps(cbuf + 16, zmm25);
        zmm0 = _mm512_load_ps(cbuf + 32);
        zmm1 = _mm512_load_ps(cbuf + 48);
        zmm26 = _mm512_fmadd_ps(zmm0, zmm3, zmm26);
        zmm27 = _mm512_fmadd_ps(zmm1, zmm3, zmm27);
        _mm512_storeu_ps(cbuf + 32, zmm26);
        _mm512_storeu_ps(cbuf + 48, zmm27);
        //cbuf += rs_c;
    }//betazero
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_4x64)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k0;

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7;
    __m512 zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15;
    __m512 zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23;
    
    /* zero the accumulator registers */
    ZERO_ACC_ZMM_4_REG(zmm8, zmm9, zmm10, zmm11);
    ZERO_ACC_ZMM_4_REG(zmm12, zmm13, zmm14, zmm15);
    ZERO_ACC_ZMM_4_REG(zmm16, zmm17, zmm18, zmm19);
    ZERO_ACC_ZMM_4_REG(zmm20, zmm21, zmm22, zmm23);

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 32 elements from row0 of B*/
        zmm0 = _mm512_loadu_ps (bbuf );     //load 0-15 values from current row 
        zmm1 = _mm512_loadu_ps (bbuf + 16); //load 16-31 values from current row

        /*Load Next 32 elements from row0 of B*/
        zmm6 = _mm512_loadu_ps (bbuf + 32); //load 32-47 from current row 
        zmm7 = _mm512_loadu_ps (bbuf + 48); //load 48-63 from current row
        
        /*Broadcast col0 elements of 12 rows of A*/
        zmm2 = _mm512_set1_ps(*(abuf + 0*rs_a)); //broadcast c0r0
        zmm3 = _mm512_set1_ps(*(abuf + 1*rs_a)); //broadcast c0r1  
        zmm4 = _mm512_set1_ps(*(abuf + 2*rs_a)); //broadcast c0r2 
        zmm5 = _mm512_set1_ps(*(abuf + 3*rs_a)); //broadcast c0r3

        zmm8 = _mm512_fmadd_ps(zmm0, zmm2, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm2, zmm9);
        zmm10 = _mm512_fmadd_ps(zmm6, zmm2, zmm10);
        zmm11 = _mm512_fmadd_ps(zmm7, zmm2, zmm11);

        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);
        zmm14 = _mm512_fmadd_ps(zmm6, zmm3, zmm14);
        zmm15 = _mm512_fmadd_ps(zmm7, zmm3, zmm15);

        zmm16 = _mm512_fmadd_ps(zmm0, zmm4, zmm16);
        zmm17 = _mm512_fmadd_ps(zmm1, zmm4, zmm17);
        zmm18 = _mm512_fmadd_ps(zmm6, zmm4, zmm18);
        zmm19 = _mm512_fmadd_ps(zmm7, zmm4, zmm19);
        
        zmm20 = _mm512_fmadd_ps(zmm0, zmm5, zmm20);
        zmm21 = _mm512_fmadd_ps(zmm1, zmm5, zmm21);
        zmm22 = _mm512_fmadd_ps(zmm6, zmm5, zmm22);
        zmm23 = _mm512_fmadd_ps(zmm7, zmm5, zmm23);

        bbuf += rs_b;  //move b pointer to next row
        abuf += cs_a;  //move a pointer to next col

    }//kloop

    zmm0 = _mm512_set1_ps(alpha);
    ALPHA_MUL_ACC_ZMM_4_REG(zmm8,  zmm9,  zmm10, zmm11, zmm0);
    ALPHA_MUL_ACC_ZMM_4_REG(zmm12, zmm13, zmm14, zmm15, zmm0);
    ALPHA_MUL_ACC_ZMM_4_REG(zmm16, zmm17, zmm18, zmm19, zmm0);
    ALPHA_MUL_ACC_ZMM_4_REG(zmm20, zmm21, zmm22, zmm23, zmm0);

      //store output when beta=0
    if(beta == 0.0)
    {
        _mm512_storeu_ps(cbuf, zmm8); 
        _mm512_storeu_ps(cbuf + 16, zmm9);
        _mm512_storeu_ps(cbuf + 32, zmm10);
        _mm512_storeu_ps(cbuf + 48, zmm11);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm12);
        _mm512_storeu_ps(cbuf + 16, zmm13);
        _mm512_storeu_ps(cbuf + 32, zmm14);
        _mm512_storeu_ps(cbuf + 48, zmm15);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm16);
        _mm512_storeu_ps(cbuf + 16, zmm17);
        _mm512_storeu_ps(cbuf + 32, zmm18);
        _mm512_storeu_ps(cbuf + 48, zmm19);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm20);
        _mm512_storeu_ps(cbuf + 16, zmm21);
        _mm512_storeu_ps(cbuf + 32, zmm22);
        _mm512_storeu_ps(cbuf + 48, zmm23);
        //cbuf += rs_c;
    }else
    { 
        //load c and multiply with beta and 
        //add to accumulator and store back
        zmm3 = _mm512_set1_ps(beta);

        zmm0 = _mm512_load_ps(cbuf);
        zmm1 = _mm512_load_ps(cbuf + 16);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);
        _mm512_storeu_ps(cbuf, zmm8); 
        _mm512_storeu_ps(cbuf + 16, zmm9);
        zmm0 = _mm512_load_ps(cbuf + 32);
        zmm1 = _mm512_load_ps(cbuf + 48);
        zmm10 = _mm512_fmadd_ps(zmm0, zmm3, zmm10);
        zmm11 = _mm512_fmadd_ps(zmm1, zmm3, zmm11);
        _mm512_storeu_ps(cbuf + 32, zmm10);
        _mm512_storeu_ps(cbuf + 48, zmm11);
        cbuf += rs_c;

        zmm0 = _mm512_load_ps(cbuf);
        zmm1 = _mm512_load_ps(cbuf + 16);
        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);
        _mm512_storeu_ps(cbuf, zmm12);
        _mm512_storeu_ps(cbuf + 16, zmm13);
        zmm0 = _mm512_load_ps(cbuf + 32);
        zmm1 = _mm512_load_ps(cbuf + 48);
        zmm14 = _mm512_fmadd_ps(zmm0, zmm3, zmm14);
        zmm15 = _mm512_fmadd_ps(zmm1, zmm3, zmm15);
        _mm512_storeu_ps(cbuf + 32, zmm14);
        _mm512_storeu_ps(cbuf + 48, zmm15);
        cbuf += rs_c;

        zmm0 = _mm512_load_ps(cbuf);
        zmm1 = _mm512_load_ps(cbuf+16);
        zmm16 = _mm512_fmadd_ps(zmm0, zmm3, zmm16);
        zmm17 = _mm512_fmadd_ps(zmm1, zmm3, zmm17);
        _mm512_storeu_ps(cbuf, zmm16);
        _mm512_storeu_ps(cbuf + 16, zmm17);
        zmm0 = _mm512_load_ps(cbuf + 32);
        zmm1 = _mm512_load_ps(cbuf + 48);
        zmm18 = _mm512_fmadd_ps(zmm0, zmm3, zmm18);
        zmm19 = _mm512_fmadd_ps(zmm1, zmm3, zmm19);
        _mm512_storeu_ps(cbuf + 32, zmm18);
        _mm512_storeu_ps(cbuf + 48, zmm19);
        cbuf += rs_c;

        zmm0 = _mm512_load_ps(cbuf);
        zmm1 = _mm512_load_ps(cbuf+16);
        zmm20 = _mm512_fmadd_ps(zmm0, zmm3, zmm20);
        zmm21 = _mm512_fmadd_ps(zmm1, zmm3, zmm21);
        _mm512_storeu_ps(cbuf, zmm20);
        _mm512_storeu_ps(cbuf + 16, zmm21);
        zmm0 = _mm512_load_ps(cbuf + 32);
        zmm1 = _mm512_load_ps(cbuf + 48);
        zmm22 = _mm512_fmadd_ps(zmm0, zmm3, zmm22);
        zmm23 = _mm512_fmadd_ps(zmm1, zmm3, zmm23);
        _mm512_storeu_ps(cbuf + 32, zmm22);
        _mm512_storeu_ps(cbuf + 48, zmm23);
        //cbuf += rs_c;
    }//betazero
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_3x64)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k0;

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm6, zmm7;
    __m512 zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15;
    __m512 zmm16, zmm17, zmm18, zmm19;
    
    /* zero the accumulator registers */
    ZERO_ACC_ZMM_4_REG(zmm8, zmm9, zmm10, zmm11);
    ZERO_ACC_ZMM_4_REG(zmm12, zmm13, zmm14, zmm15);
    ZERO_ACC_ZMM_4_REG(zmm16, zmm17, zmm18, zmm19);

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 32 elements from row0 of B*/
        zmm0 = _mm512_loadu_ps (bbuf );     //load 0-15 values from current row 
        zmm1 = _mm512_loadu_ps (bbuf + 16); //load 16-31 values from current row

        /*Load Next 32 elements from row0 of B*/
        zmm6 = _mm512_loadu_ps (bbuf + 32); //load 32-47 from current row 
        zmm7 = _mm512_loadu_ps (bbuf + 48); //load 48-63 from current row
        
        /*Broadcast col0 elements of 12 rows of A*/
        zmm2 = _mm512_set1_ps(*(abuf + 0*rs_a)); //broadcast c0r0
        zmm3 = _mm512_set1_ps(*(abuf + 1*rs_a)); //broadcast c0r1  
        zmm4 = _mm512_set1_ps(*(abuf + 2*rs_a)); //broadcast c0r2 

        zmm8 = _mm512_fmadd_ps(zmm0, zmm2, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm2, zmm9);
        zmm10 = _mm512_fmadd_ps(zmm6, zmm2, zmm10);
        zmm11 = _mm512_fmadd_ps(zmm7, zmm2, zmm11);

        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);
        zmm14 = _mm512_fmadd_ps(zmm6, zmm3, zmm14);
        zmm15 = _mm512_fmadd_ps(zmm7, zmm3, zmm15);

        zmm16 = _mm512_fmadd_ps(zmm0, zmm4, zmm16);
        zmm17 = _mm512_fmadd_ps(zmm1, zmm4, zmm17);
        zmm18 = _mm512_fmadd_ps(zmm6, zmm4, zmm18);
        zmm19 = _mm512_fmadd_ps(zmm7, zmm4, zmm19);

        bbuf += rs_b;  //move b pointer to next row
        abuf += cs_a;  //move a pointer to next col

    }//kloop

    zmm0 = _mm512_set1_ps(alpha);
    ALPHA_MUL_ACC_ZMM_4_REG(zmm8,  zmm9,  zmm10, zmm11, zmm0);
    ALPHA_MUL_ACC_ZMM_4_REG(zmm12, zmm13, zmm14, zmm15, zmm0);
    ALPHA_MUL_ACC_ZMM_4_REG(zmm16, zmm17, zmm18, zmm19, zmm0);

      //store output when beta=0
    if(beta == 0.0)
    {
        _mm512_storeu_ps(cbuf, zmm8); 
        _mm512_storeu_ps(cbuf + 16, zmm9);
        _mm512_storeu_ps(cbuf + 32, zmm10);
        _mm512_storeu_ps(cbuf + 48, zmm11);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm12);
        _mm512_storeu_ps(cbuf + 16, zmm13);
        _mm512_storeu_ps(cbuf + 32, zmm14);
        _mm512_storeu_ps(cbuf + 48, zmm15);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm16);
        _mm512_storeu_ps(cbuf + 16, zmm17);
        _mm512_storeu_ps(cbuf + 32, zmm18);
        _mm512_storeu_ps(cbuf + 48, zmm19);
    }else
    { 
        //load c and multiply with beta and 
        //add to accumulator and store back
        zmm3 = _mm512_set1_ps(beta);

        zmm0 = _mm512_load_ps(cbuf);
        zmm1 = _mm512_load_ps(cbuf + 16);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);
        _mm512_storeu_ps(cbuf, zmm8); 
        _mm512_storeu_ps(cbuf + 16, zmm9);
        zmm0 = _mm512_load_ps(cbuf + 32);
        zmm1 = _mm512_load_ps(cbuf + 48);
        zmm10 = _mm512_fmadd_ps(zmm0, zmm3, zmm10);
        zmm11 = _mm512_fmadd_ps(zmm1, zmm3, zmm11);
        _mm512_storeu_ps(cbuf + 32, zmm10);
        _mm512_storeu_ps(cbuf + 48, zmm11);
        cbuf += rs_c;

        zmm0 = _mm512_load_ps(cbuf);
        zmm1 = _mm512_load_ps(cbuf + 16);
        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);
        _mm512_storeu_ps(cbuf, zmm12);
        _mm512_storeu_ps(cbuf + 16, zmm13);
        zmm0 = _mm512_load_ps(cbuf + 32);
        zmm1 = _mm512_load_ps(cbuf + 48);
        zmm14 = _mm512_fmadd_ps(zmm0, zmm3, zmm14);
        zmm15 = _mm512_fmadd_ps(zmm1, zmm3, zmm15);
        _mm512_storeu_ps(cbuf + 32, zmm14);
        _mm512_storeu_ps(cbuf + 48, zmm15);
        cbuf += rs_c;

        zmm0 = _mm512_load_ps(cbuf);
        zmm1 = _mm512_load_ps(cbuf+16);
        zmm16 = _mm512_fmadd_ps(zmm0, zmm3, zmm16);
        zmm17 = _mm512_fmadd_ps(zmm1, zmm3, zmm17);
        _mm512_storeu_ps(cbuf, zmm16);
        _mm512_storeu_ps(cbuf + 16, zmm17);
        zmm0 = _mm512_load_ps(cbuf + 32);
        zmm1 = _mm512_load_ps(cbuf + 48);
        zmm18 = _mm512_fmadd_ps(zmm0, zmm3, zmm18);
        zmm19 = _mm512_fmadd_ps(zmm1, zmm3, zmm19);
        _mm512_storeu_ps(cbuf + 32, zmm18);
        _mm512_storeu_ps(cbuf + 48, zmm19);
    }//betazero
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_2x64)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k0;

    __m512 zmm0, zmm1, zmm2, zmm3, zmm6, zmm7;
    __m512 zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15;
    
    /* zero the accumulator registers */
    ZERO_ACC_ZMM_4_REG(zmm8, zmm9, zmm10, zmm11);
    ZERO_ACC_ZMM_4_REG(zmm12, zmm13, zmm14, zmm15);

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float    *cbuf = (float *)c;

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 32 elements from row0 of B*/
        zmm0 = _mm512_loadu_ps (bbuf );     //load 0-15 values from current row 
        zmm1 = _mm512_loadu_ps (bbuf + 16); //load 16-31 values from current row

        /*Load Next 32 elements from row0 of B*/
        zmm6 = _mm512_loadu_ps (bbuf + 32); //load 32-47 from current row 
        zmm7 = _mm512_loadu_ps (bbuf + 48); //load 48-63 from current row
        
        /*Broadcast col0 elements of 12 rows of A*/
        zmm2 = _mm512_set1_ps(*(abuf + 0*rs_a)); //broadcast c0r0
        zmm3 = _mm512_set1_ps(*(abuf + 1*rs_a)); //broadcast c0r1 

        zmm8 = _mm512_fmadd_ps(zmm0, zmm2, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm2, zmm9);
        zmm10 = _mm512_fmadd_ps(zmm6, zmm2, zmm10);
        zmm11 = _mm512_fmadd_ps(zmm7, zmm2, zmm11);

        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);
        zmm14 = _mm512_fmadd_ps(zmm6, zmm3, zmm14);
        zmm15 = _mm512_fmadd_ps(zmm7, zmm3, zmm15);

        bbuf += rs_b;  //move b pointer to next row
        abuf += cs_a;  //move a pointer to next col

    }//kloop

    zmm0 = _mm512_set1_ps(alpha);
    ALPHA_MUL_ACC_ZMM_4_REG(zmm8,  zmm9,  zmm10, zmm11, zmm0);
    ALPHA_MUL_ACC_ZMM_4_REG(zmm12, zmm13, zmm14, zmm15, zmm0);

      //store output when beta=0
    if(beta == 0.0)
    {
        _mm512_storeu_ps(cbuf, zmm8); 
        _mm512_storeu_ps(cbuf + 16, zmm9);
        _mm512_storeu_ps(cbuf + 32, zmm10);
        _mm512_storeu_ps(cbuf + 48, zmm11);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm12);
        _mm512_storeu_ps(cbuf + 16, zmm13);
        _mm512_storeu_ps(cbuf + 32, zmm14);
        _mm512_storeu_ps(cbuf + 48, zmm15);
        //cbuf += rs_c;
    }else
    { 
        //load c and multiply with beta and 
        //add to accumulator and store back
        zmm3 = _mm512_set1_ps(beta);

        zmm0 = _mm512_load_ps(cbuf);
        zmm1 = _mm512_load_ps(cbuf + 16);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);
        _mm512_storeu_ps(cbuf, zmm8); 
        _mm512_storeu_ps(cbuf + 16, zmm9);
        zmm0 = _mm512_load_ps(cbuf + 32);
        zmm1 = _mm512_load_ps(cbuf + 48);
        zmm10 = _mm512_fmadd_ps(zmm0, zmm3, zmm10);
        zmm11 = _mm512_fmadd_ps(zmm1, zmm3, zmm11);
        _mm512_storeu_ps(cbuf + 32, zmm10);
        _mm512_storeu_ps(cbuf + 48, zmm11);
        cbuf += rs_c;

        zmm0 = _mm512_load_ps(cbuf);
        zmm1 = _mm512_load_ps(cbuf + 16);
        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);
        _mm512_storeu_ps(cbuf, zmm12);
        _mm512_storeu_ps(cbuf + 16, zmm13);
        zmm0 = _mm512_load_ps(cbuf + 32);
        zmm1 = _mm512_load_ps(cbuf + 48);
        zmm14 = _mm512_fmadd_ps(zmm0, zmm3, zmm14);
        zmm15 = _mm512_fmadd_ps(zmm1, zmm3, zmm15);
        _mm512_storeu_ps(cbuf + 32, zmm14);
        _mm512_storeu_ps(cbuf + 48, zmm15);
    }//betazero
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_1x64)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k0;

    __m512 zmm0, zmm1, zmm2, zmm3, zmm6, zmm7;
    __m512 zmm8, zmm9, zmm10, zmm11;
    
    /* zero the accumulator registers */
    ZERO_ACC_ZMM_4_REG(zmm8, zmm9, zmm10, zmm11);

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float    *cbuf = (float *)c;

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 32 elements from row0 of B*/
        zmm0 = _mm512_loadu_ps (bbuf );     //load 0-15 values from current row 
        zmm1 = _mm512_loadu_ps (bbuf + 16); //load 16-31 values from current row

        /*Load Next 32 elements from row0 of B*/
        zmm6 = _mm512_loadu_ps (bbuf + 32); //load 32-47 from current row 
        zmm7 = _mm512_loadu_ps (bbuf + 48); //load 48-63 from current row
        
        /*Broadcast col0 elements of 12 rows of A*/
        zmm2 = _mm512_set1_ps(*(abuf + 0*rs_a)); //broadcast c0r0

        zmm8 = _mm512_fmadd_ps(zmm0, zmm2, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm2, zmm9);
        zmm10 = _mm512_fmadd_ps(zmm6, zmm2, zmm10);
        zmm11 = _mm512_fmadd_ps(zmm7, zmm2, zmm11);

        bbuf += rs_b;  //move b pointer to next row
        abuf += cs_a;  //move a pointer to next col

    }//kloop

    zmm0 = _mm512_set1_ps(alpha);

    ALPHA_MUL_ACC_ZMM_4_REG(zmm8,  zmm9,  zmm10, zmm11, zmm0);

    //store output when beta=0
    if(beta == 0.0)
    {
        _mm512_storeu_ps(cbuf, zmm8); 
        _mm512_storeu_ps(cbuf + 16, zmm9);
        _mm512_storeu_ps(cbuf + 32, zmm10);
        _mm512_storeu_ps(cbuf + 48, zmm11);
        //cbuf += rs_c;
    }else
    { 
        //load c and multiply with beta and 
        //add to accumulator and store back
        zmm3 = _mm512_set1_ps(beta);

        zmm0 = _mm512_load_ps(cbuf);
        zmm1 = _mm512_load_ps(cbuf + 16);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);
        _mm512_storeu_ps(cbuf, zmm8); 
        _mm512_storeu_ps(cbuf + 16, zmm9);
        zmm0 = _mm512_load_ps(cbuf + 32);
        zmm1 = _mm512_load_ps(cbuf + 48);
        zmm10 = _mm512_fmadd_ps(zmm0, zmm3, zmm10);
        zmm11 = _mm512_fmadd_ps(zmm1, zmm3, zmm11);
        _mm512_storeu_ps(cbuf + 32, zmm10);
        _mm512_storeu_ps(cbuf + 48, zmm11);
        //cbuf += rs_c;
    }//betazero
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_5x48)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k0;

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6;
    __m512 zmm8, zmm9, zmm10, zmm12, zmm13, zmm14;
    __m512 zmm16, zmm17, zmm18, zmm20, zmm21, zmm22;
    __m512 zmm24, zmm25, zmm26, zmm28;

    /* zero the accumulator registers */
    ZERO_ACC_ZMM_4_REG(zmm8, zmm9, zmm10, zmm12);
    ZERO_ACC_ZMM_4_REG(zmm13, zmm14,zmm16, zmm17);
    ZERO_ACC_ZMM_4_REG(zmm18, zmm20, zmm21, zmm22);
    ZERO_ACC_ZMM_4_REG(zmm24, zmm25, zmm26, zmm28);

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 32 elements from row0 of B*/
        zmm0 = _mm512_loadu_ps (bbuf );     //load 0-15 values from current row 
        zmm1 = _mm512_loadu_ps (bbuf + 16); //load 16-31 values from current row

        /*Load Next 32 elements from row0 of B*/
        zmm6 = _mm512_loadu_ps (bbuf + 32); //load 32-47 from current row 
        
        /*Broadcast col0 elements of 12 rows of A*/
        zmm2 = _mm512_set1_ps(*(abuf + 0*rs_a)); //broadcast c0r0
        zmm3 = _mm512_set1_ps(*(abuf + 1*rs_a)); //broadcast c0r1  
        zmm4 = _mm512_set1_ps(*(abuf + 2*rs_a)); //broadcast c0r2 
        zmm5 = _mm512_set1_ps(*(abuf + 3*rs_a)); //broadcast c0r3

        zmm8 = _mm512_fmadd_ps(zmm0, zmm2, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm2, zmm9);
        zmm10 = _mm512_fmadd_ps(zmm6, zmm2, zmm10);

        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);
        zmm14 = _mm512_fmadd_ps(zmm6, zmm3, zmm14);

        zmm2 = _mm512_set1_ps(*(abuf + 4*rs_a)); //broadcast c0r4
        
        zmm16 = _mm512_fmadd_ps(zmm0, zmm4, zmm16);
        zmm17 = _mm512_fmadd_ps(zmm1, zmm4, zmm17);
        zmm18 = _mm512_fmadd_ps(zmm6, zmm4, zmm18);
        
        zmm20 = _mm512_fmadd_ps(zmm0, zmm5, zmm20);
        zmm21 = _mm512_fmadd_ps(zmm1, zmm5, zmm21);
        zmm22 = _mm512_fmadd_ps(zmm6, zmm5, zmm22);

        zmm24 = _mm512_fmadd_ps(zmm0, zmm2, zmm24);
        zmm25 = _mm512_fmadd_ps(zmm1, zmm2, zmm25);
        zmm26 = _mm512_fmadd_ps(zmm6, zmm2, zmm26);

        bbuf += rs_b;  //move b pointer to next row
        abuf += cs_a;  //move a pointer to next col

    }//kloop

    zmm0 = _mm512_set1_ps(alpha);
    ALPHA_MUL_ACC_ZMM_4_REG(zmm8,zmm9,zmm10,zmm12,zmm0)
    ALPHA_MUL_ACC_ZMM_4_REG(zmm13,zmm14,zmm16,zmm17,zmm0)
    ALPHA_MUL_ACC_ZMM_4_REG(zmm18,zmm20,zmm21,zmm22,zmm0)
    ALPHA_MUL_ACC_ZMM_4_REG(zmm24,zmm25,zmm26,zmm28,zmm0)

    //store output when beta=0
    if(beta == 0.0)
    {
      _mm512_storeu_ps(cbuf, zmm8); 
      _mm512_storeu_ps(cbuf + 16, zmm9);
      _mm512_storeu_ps(cbuf + 32, zmm10);
      cbuf += rs_c;
      _mm512_storeu_ps(cbuf, zmm12);
      _mm512_storeu_ps(cbuf + 16, zmm13);
      _mm512_storeu_ps(cbuf + 32, zmm14);
      cbuf += rs_c;
      _mm512_storeu_ps(cbuf, zmm16);
      _mm512_storeu_ps(cbuf + 16, zmm17);
      _mm512_storeu_ps(cbuf + 32, zmm18);
      cbuf += rs_c;
      _mm512_storeu_ps(cbuf, zmm20);
      _mm512_storeu_ps(cbuf + 16, zmm21);
      _mm512_storeu_ps(cbuf + 32, zmm22);
      cbuf += rs_c;
      _mm512_storeu_ps(cbuf, zmm24);
      _mm512_storeu_ps(cbuf + 16, zmm25);
      _mm512_storeu_ps(cbuf + 32, zmm26);
      //cbuf += rs_c;
    }else
    { 
      //load c and multiply with beta and 
      //add to accumulator and store back
      zmm3 = _mm512_set1_ps(beta);

      zmm0 = _mm512_load_ps(cbuf);
      zmm1 = _mm512_load_ps(cbuf + 16);
      zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
      zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);
      _mm512_storeu_ps(cbuf, zmm8); 
      _mm512_storeu_ps(cbuf + 16, zmm9);
      zmm0 = _mm512_load_ps(cbuf + 32);
      zmm10 = _mm512_fmadd_ps(zmm0, zmm3, zmm10);
      _mm512_storeu_ps(cbuf + 32, zmm10);
      cbuf += rs_c;

      zmm0 = _mm512_load_ps(cbuf);
      zmm1 = _mm512_load_ps(cbuf + 16);
      zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
      zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);
      _mm512_storeu_ps(cbuf, zmm12);
      _mm512_storeu_ps(cbuf + 16, zmm13);
      zmm0 = _mm512_load_ps(cbuf + 32);
      zmm14 = _mm512_fmadd_ps(zmm0, zmm3, zmm14);
      _mm512_storeu_ps(cbuf + 32, zmm14);
      cbuf += rs_c;

      zmm0 = _mm512_load_ps(cbuf);
      zmm1 = _mm512_load_ps(cbuf+16);
      zmm16 = _mm512_fmadd_ps(zmm0, zmm3, zmm16);
      zmm17 = _mm512_fmadd_ps(zmm1, zmm3, zmm17);
      _mm512_storeu_ps(cbuf, zmm16);
      _mm512_storeu_ps(cbuf + 16, zmm17);
      zmm0 = _mm512_load_ps(cbuf + 32);
      zmm18 = _mm512_fmadd_ps(zmm0, zmm3, zmm18);
      _mm512_storeu_ps(cbuf + 32, zmm18);
      cbuf += rs_c;

      zmm0 = _mm512_load_ps(cbuf);
      zmm1 = _mm512_load_ps(cbuf+16);
      zmm20 = _mm512_fmadd_ps(zmm0, zmm3, zmm20);
      zmm21 = _mm512_fmadd_ps(zmm1, zmm3, zmm21);
      _mm512_storeu_ps(cbuf, zmm20);
      _mm512_storeu_ps(cbuf + 16, zmm21);
      zmm0 = _mm512_load_ps(cbuf + 32);
      zmm22 = _mm512_fmadd_ps(zmm0, zmm3, zmm22);
      _mm512_storeu_ps(cbuf + 32, zmm22);
      cbuf += rs_c;

      zmm0 = _mm512_load_ps(cbuf);
      zmm1 = _mm512_load_ps(cbuf+16);
      zmm24 = _mm512_fmadd_ps(zmm0, zmm3, zmm24);
      zmm25 = _mm512_fmadd_ps(zmm1, zmm3, zmm25);
      _mm512_storeu_ps(cbuf, zmm24);
      _mm512_storeu_ps(cbuf + 16, zmm25);
      zmm0 = _mm512_load_ps(cbuf + 32);
      zmm26 = _mm512_fmadd_ps(zmm0, zmm3, zmm26);
      _mm512_storeu_ps(cbuf + 32, zmm26);
    }//betazero
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_4x48)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k0;

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6;
    __m512 zmm8, zmm9, zmm10, zmm12, zmm13, zmm14;
    __m512 zmm16, zmm17, zmm18, zmm20, zmm21, zmm22;

    /* zero the accumulator registers */
    ZERO_ACC_ZMM_4_REG(zmm8, zmm9, zmm10, zmm12);
    ZERO_ACC_ZMM_4_REG(zmm13, zmm14,zmm16, zmm17);
    ZERO_ACC_ZMM_4_REG(zmm18, zmm20, zmm21, zmm22);

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 32 elements from row0 of B*/
        zmm0 = _mm512_loadu_ps (bbuf );     //load 0-15 values from current row 
        zmm1 = _mm512_loadu_ps (bbuf + 16); //load 16-31 values from current row

        /*Load Next 32 elements from row0 of B*/
        zmm6 = _mm512_loadu_ps (bbuf + 32); //load 32-47 from current row 
        
        /*Broadcast col0 elements of 12 rows of A*/
        zmm2 = _mm512_set1_ps(*(abuf + 0*rs_a)); //broadcast c0r0
        zmm3 = _mm512_set1_ps(*(abuf + 1*rs_a)); //broadcast c0r1  
        zmm4 = _mm512_set1_ps(*(abuf + 2*rs_a)); //broadcast c0r2 
        zmm5 = _mm512_set1_ps(*(abuf + 3*rs_a)); //broadcast c0r3

        zmm8 = _mm512_fmadd_ps(zmm0, zmm2, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm2, zmm9);
        zmm10 = _mm512_fmadd_ps(zmm6, zmm2, zmm10);

        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);
        zmm14 = _mm512_fmadd_ps(zmm6, zmm3, zmm14);

        zmm16 = _mm512_fmadd_ps(zmm0, zmm4, zmm16);
        zmm17 = _mm512_fmadd_ps(zmm1, zmm4, zmm17);
        zmm18 = _mm512_fmadd_ps(zmm6, zmm4, zmm18);
        
        zmm20 = _mm512_fmadd_ps(zmm0, zmm5, zmm20);
        zmm21 = _mm512_fmadd_ps(zmm1, zmm5, zmm21);
        zmm22 = _mm512_fmadd_ps(zmm6, zmm5, zmm22);

        bbuf += rs_b;  //move b pointer to next row
        abuf += cs_a;  //move a pointer to next col

    }//kloop

    zmm0 = _mm512_set1_ps(alpha);

    ALPHA_MUL_ACC_ZMM_4_REG(zmm8,zmm9,zmm10,zmm12,zmm0)
    ALPHA_MUL_ACC_ZMM_4_REG(zmm13,zmm14,zmm16,zmm17,zmm0)
    ALPHA_MUL_ACC_ZMM_4_REG(zmm18,zmm20,zmm21,zmm22,zmm0)

    //store output when beta=0
    if(beta == 0.0)
    {
      _mm512_storeu_ps(cbuf, zmm8); 
      _mm512_storeu_ps(cbuf + 16, zmm9);
      _mm512_storeu_ps(cbuf + 32, zmm10);
      cbuf += rs_c;
      _mm512_storeu_ps(cbuf, zmm12);
      _mm512_storeu_ps(cbuf + 16, zmm13);
      _mm512_storeu_ps(cbuf + 32, zmm14);
      cbuf += rs_c;
      _mm512_storeu_ps(cbuf, zmm16);
      _mm512_storeu_ps(cbuf + 16, zmm17);
      _mm512_storeu_ps(cbuf + 32, zmm18);
      cbuf += rs_c;
      _mm512_storeu_ps(cbuf, zmm20);
      _mm512_storeu_ps(cbuf + 16, zmm21);
      _mm512_storeu_ps(cbuf + 32, zmm22);
      //cbuf += rs_c;
    }else
    { 
      //load c and multiply with beta and 
      //add to accumulator and store back
      zmm3 = _mm512_set1_ps(beta);

      zmm0 = _mm512_load_ps(cbuf);
      zmm1 = _mm512_load_ps(cbuf + 16);
      zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
      zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);
      _mm512_storeu_ps(cbuf, zmm8); 
      _mm512_storeu_ps(cbuf + 16, zmm9);
      zmm0 = _mm512_load_ps(cbuf + 32);
      zmm10 = _mm512_fmadd_ps(zmm0, zmm3, zmm10);
      _mm512_storeu_ps(cbuf + 32, zmm10);
      cbuf += rs_c;

      zmm0 = _mm512_load_ps(cbuf);
      zmm1 = _mm512_load_ps(cbuf + 16);
      zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
      zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);
      _mm512_storeu_ps(cbuf, zmm12);
      _mm512_storeu_ps(cbuf + 16, zmm13);
      zmm0 = _mm512_load_ps(cbuf + 32);
      zmm14 = _mm512_fmadd_ps(zmm0, zmm3, zmm14);
      _mm512_storeu_ps(cbuf + 32, zmm14);
      cbuf += rs_c;

      zmm0 = _mm512_load_ps(cbuf);
      zmm1 = _mm512_load_ps(cbuf+16);
      zmm16 = _mm512_fmadd_ps(zmm0, zmm3, zmm16);
      zmm17 = _mm512_fmadd_ps(zmm1, zmm3, zmm17);
      _mm512_storeu_ps(cbuf, zmm16);
      _mm512_storeu_ps(cbuf + 16, zmm17);
      zmm0 = _mm512_load_ps(cbuf + 32);
      zmm18 = _mm512_fmadd_ps(zmm0, zmm3, zmm18);
      _mm512_storeu_ps(cbuf + 32, zmm18);
      cbuf += rs_c;

      zmm0 = _mm512_load_ps(cbuf);
      zmm1 = _mm512_load_ps(cbuf+16);
      zmm20 = _mm512_fmadd_ps(zmm0, zmm3, zmm20);
      zmm21 = _mm512_fmadd_ps(zmm1, zmm3, zmm21);
      _mm512_storeu_ps(cbuf, zmm20);
      _mm512_storeu_ps(cbuf + 16, zmm21);
      zmm0 = _mm512_load_ps(cbuf + 32);
      zmm22 = _mm512_fmadd_ps(zmm0, zmm3, zmm22);
      _mm512_storeu_ps(cbuf + 32, zmm22);
    }//betazero
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_3x48)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k0;

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm6;
    __m512 zmm8, zmm9, zmm10, zmm12, zmm13, zmm14;
    __m512 zmm16, zmm17, zmm18;

    /* zero the accumulator registers */
    ZERO_ACC_ZMM_4_REG(zmm8, zmm9, zmm10, zmm12);
    ZERO_ACC_ZMM_4_REG(zmm13, zmm14,zmm16, zmm17);
    zmm18 = _mm512_setzero_ps();

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 32 elements from row0 of B*/
        zmm0 = _mm512_loadu_ps (bbuf );     //load 0-15 values from current row 
        zmm1 = _mm512_loadu_ps (bbuf + 16); //load 16-31 values from current row

        /*Load Next 32 elements from row0 of B*/
        zmm6 = _mm512_loadu_ps (bbuf + 32); //load 32-47 from current row 
        
        /*Broadcast col0 elements of 12 rows of A*/
        zmm2 = _mm512_set1_ps(*(abuf + 0*rs_a)); //broadcast c0r0
        zmm3 = _mm512_set1_ps(*(abuf + 1*rs_a)); //broadcast c0r1  
        zmm4 = _mm512_set1_ps(*(abuf + 2*rs_a)); //broadcast c0r2 

        zmm8 = _mm512_fmadd_ps(zmm0, zmm2, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm2, zmm9);
        zmm10 = _mm512_fmadd_ps(zmm6, zmm2, zmm10);

        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);
        zmm14 = _mm512_fmadd_ps(zmm6, zmm3, zmm14);

        zmm16 = _mm512_fmadd_ps(zmm0, zmm4, zmm16);
        zmm17 = _mm512_fmadd_ps(zmm1, zmm4, zmm17);
        zmm18 = _mm512_fmadd_ps(zmm6, zmm4, zmm18);        

        bbuf += rs_b;  //move b pointer to next row
        abuf += cs_a;  //move a pointer to next col

    }//kloop

    zmm0 = _mm512_set1_ps(alpha);
    ALPHA_MUL_ACC_ZMM_4_REG(zmm8,zmm9,zmm10,zmm12,zmm0)
    ALPHA_MUL_ACC_ZMM_4_REG(zmm13,zmm14,zmm16,zmm17,zmm0)
    zmm18 = _mm512_mul_ps(zmm18, zmm0);

    //store output when beta=0
    if(beta == 0.0)
    {
      _mm512_storeu_ps(cbuf, zmm8); 
      _mm512_storeu_ps(cbuf + 16, zmm9);
      _mm512_storeu_ps(cbuf + 32, zmm10);
      cbuf += rs_c;
      _mm512_storeu_ps(cbuf, zmm12);
      _mm512_storeu_ps(cbuf + 16, zmm13);
      _mm512_storeu_ps(cbuf + 32, zmm14);
      cbuf += rs_c;
      _mm512_storeu_ps(cbuf, zmm16);
      _mm512_storeu_ps(cbuf + 16, zmm17);
      _mm512_storeu_ps(cbuf + 32, zmm18);
      //cbuf += rs_c;
    }else
    { 
      //load c and multiply with beta and 
      //add to accumulator and store back
      zmm3 = _mm512_set1_ps(beta);

      zmm0 = _mm512_load_ps(cbuf);
      zmm1 = _mm512_load_ps(cbuf + 16);
      zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
      zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);
      _mm512_storeu_ps(cbuf, zmm8); 
      _mm512_storeu_ps(cbuf + 16, zmm9);
      zmm0 = _mm512_load_ps(cbuf + 32);
      zmm10 = _mm512_fmadd_ps(zmm0, zmm3, zmm10);
      _mm512_storeu_ps(cbuf + 32, zmm10);
      cbuf += rs_c;

      zmm0 = _mm512_load_ps(cbuf);
      zmm1 = _mm512_load_ps(cbuf + 16);
      zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
      zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);
      _mm512_storeu_ps(cbuf, zmm12);
      _mm512_storeu_ps(cbuf + 16, zmm13);
      zmm0 = _mm512_load_ps(cbuf + 32);
      zmm14 = _mm512_fmadd_ps(zmm0, zmm3, zmm14);
      _mm512_storeu_ps(cbuf + 32, zmm14);
      cbuf += rs_c;

      zmm0 = _mm512_load_ps(cbuf);
      zmm1 = _mm512_load_ps(cbuf+16);
      zmm16 = _mm512_fmadd_ps(zmm0, zmm3, zmm16);
      zmm17 = _mm512_fmadd_ps(zmm1, zmm3, zmm17);
      _mm512_storeu_ps(cbuf, zmm16);
      _mm512_storeu_ps(cbuf + 16, zmm17);
      zmm0 = _mm512_load_ps(cbuf + 32);
      zmm18 = _mm512_fmadd_ps(zmm0, zmm3, zmm18);
      _mm512_storeu_ps(cbuf + 32, zmm18);
    }//betazero
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_2x48)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k0;

    __m512 zmm0, zmm1, zmm2, zmm3, zmm6;
    __m512 zmm8, zmm9, zmm10, zmm12, zmm13, zmm14, zmm16,zmm17;

    /* zero the accumulator registers */
    ZERO_ACC_ZMM_4_REG(zmm8, zmm9, zmm10, zmm12);
    ZERO_ACC_ZMM_4_REG(zmm13, zmm14,zmm16, zmm17);

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 32 elements from row0 of B*/
        zmm0 = _mm512_loadu_ps (bbuf );     //load 0-15 values from current row 
        zmm1 = _mm512_loadu_ps (bbuf + 16); //load 16-31 values from current row

        /*Load Next 32 elements from row0 of B*/
        zmm6 = _mm512_loadu_ps (bbuf + 32); //load 32-47 from current row 
        
        /*Broadcast col0 elements of 12 rows of A*/
        zmm2 = _mm512_set1_ps(*(abuf + 0*rs_a)); //broadcast c0r0
        zmm3 = _mm512_set1_ps(*(abuf + 1*rs_a)); //broadcast c0r1

        zmm8 = _mm512_fmadd_ps(zmm0, zmm2, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm2, zmm9);
        zmm10 = _mm512_fmadd_ps(zmm6, zmm2, zmm10);

        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);
        zmm14 = _mm512_fmadd_ps(zmm6, zmm3, zmm14);

        bbuf += rs_b;  //move b pointer to next row
        abuf += cs_a;  //move a pointer to next col

    }//kloop

    zmm0 = _mm512_set1_ps(alpha);
    ALPHA_MUL_ACC_ZMM_4_REG(zmm8,zmm9,zmm10,zmm12,zmm0)
    ALPHA_MUL_ACC_ZMM_4_REG(zmm13,zmm14,zmm16,zmm17,zmm0)

    //store output when beta=0
    if(beta == 0.0)
    {
      _mm512_storeu_ps(cbuf, zmm8); 
      _mm512_storeu_ps(cbuf + 16, zmm9);
      _mm512_storeu_ps(cbuf + 32, zmm10);
      cbuf += rs_c;
      _mm512_storeu_ps(cbuf, zmm12);
      _mm512_storeu_ps(cbuf + 16, zmm13);
      _mm512_storeu_ps(cbuf + 32, zmm14);
    }else
    { 
      //load c and multiply with beta and 
      //add to accumulator and store back
      zmm3 = _mm512_set1_ps(beta);

      zmm0 = _mm512_load_ps(cbuf);
      zmm1 = _mm512_load_ps(cbuf + 16);
      zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
      zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);
      _mm512_storeu_ps(cbuf, zmm8); 
      _mm512_storeu_ps(cbuf + 16, zmm9);
      zmm0 = _mm512_load_ps(cbuf + 32);
      zmm10 = _mm512_fmadd_ps(zmm0, zmm3, zmm10);
      _mm512_storeu_ps(cbuf + 32, zmm10);
      cbuf += rs_c;

      zmm0 = _mm512_load_ps(cbuf);
      zmm1 = _mm512_load_ps(cbuf + 16);
      zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
      zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);
      _mm512_storeu_ps(cbuf, zmm12);
      _mm512_storeu_ps(cbuf + 16, zmm13);
      zmm0 = _mm512_load_ps(cbuf + 32);
      zmm14 = _mm512_fmadd_ps(zmm0, zmm3, zmm14);
      _mm512_storeu_ps(cbuf + 32, zmm14);
    }//betazero
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_1x48)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k0;

    __m512 zmm0, zmm1, zmm2, zmm3, zmm6;
    __m512 zmm8, zmm9, zmm10, zmm12;

    /* zero the accumulator registers */
    ZERO_ACC_ZMM_4_REG(zmm8, zmm9, zmm10, zmm12);;

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 32 elements from row0 of B*/
        zmm0 = _mm512_loadu_ps (bbuf );     //load 0-15 values from current row 
        zmm1 = _mm512_loadu_ps (bbuf + 16); //load 16-31 values from current row

        /*Load Next 32 elements from row0 of B*/
        zmm6 = _mm512_loadu_ps (bbuf + 32); //load 32-47 from current row 
        
        /*Broadcast col0 elements of 12 rows of A*/
        zmm2 = _mm512_set1_ps(*(abuf + 0*rs_a)); //broadcast c0r0

        zmm8 = _mm512_fmadd_ps(zmm0, zmm2, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm2, zmm9);
        zmm10 = _mm512_fmadd_ps(zmm6, zmm2, zmm10);

        bbuf += rs_b;  //move b pointer to next row
        abuf += cs_a;  //move a pointer to next col

    }//kloop

    zmm0 = _mm512_set1_ps(alpha);
    ALPHA_MUL_ACC_ZMM_4_REG(zmm8,zmm9,zmm10,zmm12,zmm0)

    //store output when beta=0
    if(beta == 0.0)
    {
      _mm512_storeu_ps(cbuf, zmm8); 
      _mm512_storeu_ps(cbuf + 16, zmm9);
      _mm512_storeu_ps(cbuf + 32, zmm10);
    }else
    { 
      //load c and multiply with beta and 
      //add to accumulator and store back
      zmm3 = _mm512_set1_ps(beta);

      zmm0 = _mm512_load_ps(cbuf);
      zmm1 = _mm512_load_ps(cbuf + 16);
      zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
      zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);
      _mm512_storeu_ps(cbuf, zmm8); 
      _mm512_storeu_ps(cbuf + 16, zmm9);
      zmm0 = _mm512_load_ps(cbuf + 32);
      zmm10 = _mm512_fmadd_ps(zmm0, zmm3, zmm10);
      _mm512_storeu_ps(cbuf + 32, zmm10);
    }//betazero
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_5x32)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k0;

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5;
    __m512 zmm8, zmm9, zmm12, zmm13;
    __m512 zmm16, zmm17, zmm20, zmm21;
    __m512 zmm24, zmm25, zmm28, zmm29;

    /* zero the accumulator registers */
    ZERO_ACC_ZMM_4_REG(zmm8, zmm9, zmm12, zmm13);
    ZERO_ACC_ZMM_4_REG(zmm16, zmm17, zmm20, zmm21);
    ZERO_ACC_ZMM_4_REG(zmm24, zmm25, zmm28, zmm29);

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 32 elements from row0 of B*/
        zmm0 = _mm512_loadu_ps (bbuf );     //load 0-15 values from current row 
        zmm1 = _mm512_loadu_ps (bbuf + 16); //load 16-31 values from current row

       /*Broadcast col0 elements of 12 rows of A*/
        zmm2 = _mm512_set1_ps(*(abuf + 0*rs_a)); //broadcast c0r0
        zmm3 = _mm512_set1_ps(*(abuf + 1*rs_a)); //broadcast c0r1  
        zmm4 = _mm512_set1_ps(*(abuf + 2*rs_a)); //broadcast c0r2 
        zmm5 = _mm512_set1_ps(*(abuf + 3*rs_a)); //broadcast c0r3

        zmm8 = _mm512_fmadd_ps(zmm0, zmm2, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm2, zmm9);

        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);

        zmm2 = _mm512_set1_ps(*(abuf + 4*rs_a)); //broadcast c0r4
        
        zmm16 = _mm512_fmadd_ps(zmm0, zmm4, zmm16);
        zmm17 = _mm512_fmadd_ps(zmm1, zmm4, zmm17);
        
        zmm20 = _mm512_fmadd_ps(zmm0, zmm5, zmm20);
        zmm21 = _mm512_fmadd_ps(zmm1, zmm5, zmm21);

        zmm24 = _mm512_fmadd_ps(zmm0, zmm2, zmm24);
        zmm25 = _mm512_fmadd_ps(zmm1, zmm2, zmm25);

        bbuf += rs_b;  //move b pointer to next row
        abuf += cs_a;  //move a pointer to next col

    }//kloop

    zmm0 = _mm512_set1_ps(alpha);
      
    ALPHA_MUL_ACC_ZMM_4_REG(zmm8,zmm9,zmm12,zmm13,zmm0)
    ALPHA_MUL_ACC_ZMM_4_REG(zmm16,zmm17,zmm20,zmm21,zmm0)
    ALPHA_MUL_ACC_ZMM_4_REG(zmm24,zmm25,zmm28,zmm29,zmm0)

    //store output when beta=0
    if(beta == 0.0)
    {
        _mm512_storeu_ps(cbuf, zmm8); 
        _mm512_storeu_ps(cbuf + 16, zmm9);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm12);
        _mm512_storeu_ps(cbuf + 16, zmm13);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm16);
        _mm512_storeu_ps(cbuf + 16, zmm17);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm20);
        _mm512_storeu_ps(cbuf + 16, zmm21);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm24);
        _mm512_storeu_ps(cbuf + 16, zmm25);
    }else
    { 
        //load c and multiply with beta and 
        //add to accumulator and store back
        zmm3 = _mm512_set1_ps(beta);

        zmm0 = _mm512_load_ps(cbuf);
        zmm1 = _mm512_load_ps(cbuf + 16);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);
        _mm512_storeu_ps(cbuf, zmm8); 
        _mm512_storeu_ps(cbuf + 16, zmm9);
        cbuf += rs_c;

        zmm0 = _mm512_load_ps(cbuf);
        zmm1 = _mm512_load_ps(cbuf + 16);
        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);
        _mm512_storeu_ps(cbuf, zmm12);
        _mm512_storeu_ps(cbuf + 16, zmm13);
        cbuf += rs_c;

        zmm0 = _mm512_load_ps(cbuf);
        zmm1 = _mm512_load_ps(cbuf+16);
        zmm16 = _mm512_fmadd_ps(zmm0, zmm3, zmm16);
        zmm17 = _mm512_fmadd_ps(zmm1, zmm3, zmm17);
        _mm512_storeu_ps(cbuf, zmm16);
        _mm512_storeu_ps(cbuf + 16, zmm17);
        cbuf += rs_c;

        zmm0 = _mm512_load_ps(cbuf);
        zmm1 = _mm512_load_ps(cbuf+16);
        zmm20 = _mm512_fmadd_ps(zmm0, zmm3, zmm20);
        zmm21 = _mm512_fmadd_ps(zmm1, zmm3, zmm21);
        _mm512_storeu_ps(cbuf, zmm20);
        _mm512_storeu_ps(cbuf + 16, zmm21);
        cbuf += rs_c;

        zmm0 = _mm512_load_ps(cbuf);
        zmm1 = _mm512_load_ps(cbuf+16);
        zmm24 = _mm512_fmadd_ps(zmm0, zmm3, zmm24);
        zmm25 = _mm512_fmadd_ps(zmm1, zmm3, zmm25);
        _mm512_storeu_ps(cbuf, zmm24);
        _mm512_storeu_ps(cbuf + 16, zmm25);
    }//betazero
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_4x32)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k0;

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5;
    __m512 zmm8, zmm9, zmm12, zmm13;
    __m512 zmm16, zmm17, zmm20, zmm21;

    /* zero the accumulator registers */
    ZERO_ACC_ZMM_4_REG(zmm8, zmm9, zmm12, zmm13);
    ZERO_ACC_ZMM_4_REG(zmm16, zmm17, zmm20, zmm21);

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 32 elements from row0 of B*/
        zmm0 = _mm512_loadu_ps (bbuf );     //load 0-15 values from current row 
        zmm1 = _mm512_loadu_ps (bbuf + 16); //load 16-31 values from current row

       /*Broadcast col0 elements of 12 rows of A*/
        zmm2 = _mm512_set1_ps(*(abuf + 0*rs_a)); //broadcast c0r0
        zmm3 = _mm512_set1_ps(*(abuf + 1*rs_a)); //broadcast c0r1  
        zmm4 = _mm512_set1_ps(*(abuf + 2*rs_a)); //broadcast c0r2 
        zmm5 = _mm512_set1_ps(*(abuf + 3*rs_a)); //broadcast c0r3

        zmm8 = _mm512_fmadd_ps(zmm0, zmm2, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm2, zmm9);

        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);
    
        zmm16 = _mm512_fmadd_ps(zmm0, zmm4, zmm16);
        zmm17 = _mm512_fmadd_ps(zmm1, zmm4, zmm17);
        
        zmm20 = _mm512_fmadd_ps(zmm0, zmm5, zmm20);
        zmm21 = _mm512_fmadd_ps(zmm1, zmm5, zmm21);

        bbuf += rs_b;  //move b pointer to next row
        abuf += cs_a;  //move a pointer to next col

    }//kloop

    zmm0 = _mm512_set1_ps(alpha);
      
    ALPHA_MUL_ACC_ZMM_4_REG(zmm8,zmm9,zmm12,zmm13,zmm0)
    ALPHA_MUL_ACC_ZMM_4_REG(zmm16,zmm17,zmm20,zmm21,zmm0)

    //store output when beta=0
    if(beta == 0.0)
    {
        _mm512_storeu_ps(cbuf, zmm8); 
        _mm512_storeu_ps(cbuf + 16, zmm9);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm12);
        _mm512_storeu_ps(cbuf + 16, zmm13);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm16);
        _mm512_storeu_ps(cbuf + 16, zmm17);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm20);
        _mm512_storeu_ps(cbuf + 16, zmm21);
    }else
    { 
        //load c and multiply with beta and 
        //add to accumulator and store back
        zmm3 = _mm512_set1_ps(beta);

        zmm0 = _mm512_load_ps(cbuf);
        zmm1 = _mm512_load_ps(cbuf + 16);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);
        _mm512_storeu_ps(cbuf, zmm8); 
        _mm512_storeu_ps(cbuf + 16, zmm9);
        cbuf += rs_c;

        zmm0 = _mm512_load_ps(cbuf);
        zmm1 = _mm512_load_ps(cbuf + 16);
        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);
        _mm512_storeu_ps(cbuf, zmm12);
        _mm512_storeu_ps(cbuf + 16, zmm13);
        cbuf += rs_c;

        zmm0 = _mm512_load_ps(cbuf);
        zmm1 = _mm512_load_ps(cbuf+16);
        zmm16 = _mm512_fmadd_ps(zmm0, zmm3, zmm16);
        zmm17 = _mm512_fmadd_ps(zmm1, zmm3, zmm17);
        _mm512_storeu_ps(cbuf, zmm16);
        _mm512_storeu_ps(cbuf + 16, zmm17);
        cbuf += rs_c;

        zmm0 = _mm512_load_ps(cbuf);
        zmm1 = _mm512_load_ps(cbuf+16);
        zmm20 = _mm512_fmadd_ps(zmm0, zmm3, zmm20);
        zmm21 = _mm512_fmadd_ps(zmm1, zmm3, zmm21);
        _mm512_storeu_ps(cbuf, zmm20);
        _mm512_storeu_ps(cbuf + 16, zmm21);
    }//betazero
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_3x32)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k0;

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4;
    __m512 zmm8, zmm9, zmm12, zmm13;
    __m512 zmm16, zmm17, zmm20, zmm21;

    /* zero the accumulator registers */
    ZERO_ACC_ZMM_4_REG(zmm8, zmm9, zmm12, zmm13);
    ZERO_ACC_ZMM_4_REG(zmm16, zmm17, zmm20, zmm21);

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 32 elements from row0 of B*/
        zmm0 = _mm512_loadu_ps (bbuf );     //load 0-15 values from current row 
        zmm1 = _mm512_loadu_ps (bbuf + 16); //load 16-31 values from current row

       /*Broadcast col0 elements of 12 rows of A*/
        zmm2 = _mm512_set1_ps(*(abuf + 0*rs_a)); //broadcast c0r0
        zmm3 = _mm512_set1_ps(*(abuf + 1*rs_a)); //broadcast c0r1  
        zmm4 = _mm512_set1_ps(*(abuf + 2*rs_a)); //broadcast c0r2 

        zmm8 = _mm512_fmadd_ps(zmm0, zmm2, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm2, zmm9);

        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);
    
        zmm16 = _mm512_fmadd_ps(zmm0, zmm4, zmm16);
        zmm17 = _mm512_fmadd_ps(zmm1, zmm4, zmm17);

        bbuf += rs_b;  //move b pointer to next row
        abuf += cs_a;  //move a pointer to next col

    }//kloop

    zmm0 = _mm512_set1_ps(alpha);
      
    ALPHA_MUL_ACC_ZMM_4_REG(zmm8,zmm9,zmm12,zmm13,zmm0)
    ALPHA_MUL_ACC_ZMM_4_REG(zmm16,zmm17,zmm20,zmm21,zmm0)

    //store output when beta=0
    if(beta == 0.0)
    {
        _mm512_storeu_ps(cbuf, zmm8); 
        _mm512_storeu_ps(cbuf + 16, zmm9);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm12);
        _mm512_storeu_ps(cbuf + 16, zmm13);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm16);
        _mm512_storeu_ps(cbuf + 16, zmm17);
    }else
    { 
        //load c and multiply with beta and 
        //add to accumulator and store back
        zmm3 = _mm512_set1_ps(beta);

        zmm0 = _mm512_load_ps(cbuf);
        zmm1 = _mm512_load_ps(cbuf + 16);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);
        _mm512_storeu_ps(cbuf, zmm8); 
        _mm512_storeu_ps(cbuf + 16, zmm9);
        cbuf += rs_c;

        zmm0 = _mm512_load_ps(cbuf);
        zmm1 = _mm512_load_ps(cbuf + 16);
        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);
        _mm512_storeu_ps(cbuf, zmm12);
        _mm512_storeu_ps(cbuf + 16, zmm13);
        cbuf += rs_c;

        zmm0 = _mm512_load_ps(cbuf);
        zmm1 = _mm512_load_ps(cbuf+16);
        zmm16 = _mm512_fmadd_ps(zmm0, zmm3, zmm16);
        zmm17 = _mm512_fmadd_ps(zmm1, zmm3, zmm17);
        _mm512_storeu_ps(cbuf, zmm16);
        _mm512_storeu_ps(cbuf + 16, zmm17);
        cbuf += rs_c;
    }//betazero
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_2x32)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k0;

    __m512 zmm0, zmm1, zmm2, zmm3;
    __m512 zmm8, zmm9, zmm12, zmm13;

    /* zero the accumulator registers */
    ZERO_ACC_ZMM_4_REG(zmm8, zmm9, zmm12, zmm13);

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 32 elements from row0 of B*/
        zmm0 = _mm512_loadu_ps (bbuf );     //load 0-15 values from current row 
        zmm1 = _mm512_loadu_ps (bbuf + 16); //load 16-31 values from current row

       /*Broadcast col0 elements of 12 rows of A*/
        zmm2 = _mm512_set1_ps(*(abuf + 0*rs_a)); //broadcast c0r0
        zmm3 = _mm512_set1_ps(*(abuf + 1*rs_a)); //broadcast c0r1  

        zmm8 = _mm512_fmadd_ps(zmm0, zmm2, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm2, zmm9);

        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);
    
        bbuf += rs_b;  //move b pointer to next row
        abuf += cs_a;  //move a pointer to next col

    }//kloop

    zmm0 = _mm512_set1_ps(alpha);
      
    ALPHA_MUL_ACC_ZMM_4_REG(zmm8,zmm9,zmm12,zmm13,zmm0)

    //store output when beta=0
    if(beta == 0.0)
    {
        _mm512_storeu_ps(cbuf, zmm8); 
        _mm512_storeu_ps(cbuf + 16, zmm9);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm12);
        _mm512_storeu_ps(cbuf + 16, zmm13);
    }else
    { 
        //load c and multiply with beta and 
        //add to accumulator and store back
        zmm3 = _mm512_set1_ps(beta);

        zmm0 = _mm512_load_ps(cbuf);
        zmm1 = _mm512_load_ps(cbuf + 16);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);
        _mm512_storeu_ps(cbuf, zmm8); 
        _mm512_storeu_ps(cbuf + 16, zmm9);
        cbuf += rs_c;

        zmm0 = _mm512_load_ps(cbuf);
        zmm1 = _mm512_load_ps(cbuf + 16);
        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);
        _mm512_storeu_ps(cbuf, zmm12);
        _mm512_storeu_ps(cbuf + 16, zmm13);
    }//betazero
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_1x32)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k0;

    __m512 zmm0, zmm1, zmm2, zmm3;
    __m512 zmm8, zmm9, zmm12, zmm13;

    /* zero the accumulator registers */
    ZERO_ACC_ZMM_4_REG(zmm8, zmm9, zmm12, zmm13);

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 32 elements from row0 of B*/
        zmm0 = _mm512_loadu_ps (bbuf );     //load 0-15 values from current row 
        zmm1 = _mm512_loadu_ps (bbuf + 16); //load 16-31 values from current row

       /*Broadcast col0 elements of 12 rows of A*/
        zmm2 = _mm512_set1_ps(*(abuf + 0*rs_a)); //broadcast c0r0 

        zmm8 = _mm512_fmadd_ps(zmm0, zmm2, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm2, zmm9);
    
        bbuf += rs_b;  //move b pointer to next row
        abuf += cs_a;  //move a pointer to next col

    }//kloop

    zmm0 = _mm512_set1_ps(alpha);
      
    ALPHA_MUL_ACC_ZMM_4_REG(zmm8,zmm9,zmm12,zmm13,zmm0)
    //store output when beta=0
    if(beta == 0.0)
    {
        _mm512_storeu_ps(cbuf, zmm8); 
        _mm512_storeu_ps(cbuf + 16, zmm9);
    }else
    { 
        //load c and multiply with beta and 
        //add to accumulator and store back
        zmm3 = _mm512_set1_ps(beta);

        zmm0 = _mm512_load_ps(cbuf);
        zmm1 = _mm512_load_ps(cbuf + 16);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);
        _mm512_storeu_ps(cbuf, zmm8); 
        _mm512_storeu_ps(cbuf + 16, zmm9);
    }//betazero
}
#endif
