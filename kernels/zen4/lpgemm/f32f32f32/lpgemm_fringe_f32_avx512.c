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
    static void* post_ops_labels[] =
            {
              &&POST_OPS_5x64F_DISABLE,
              &&POST_OPS_BIAS_5x64F,
              &&POST_OPS_RELU_5x64F,
              &&POST_OPS_RELU_SCALE_5x64F,
              &&POST_OPS_GELU_TANH_5x64F,
              &&POST_OPS_GELU_ERF_5x64F,
              &&POST_OPS_CLIP_5x64F
            };
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
    float *_cbuf = NULL;

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

    if ( beta != 0.0 )
    {
        _cbuf = cbuf; 
        //load c and multiply with beta and 
        //add to accumulator and store back
        zmm3 = _mm512_set1_ps(beta);

        zmm0 = _mm512_load_ps(_cbuf);
        zmm1 = _mm512_load_ps(_cbuf + 16);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);

        zmm0 = _mm512_load_ps(_cbuf + 32);
        zmm1 = _mm512_load_ps(_cbuf + 48);
        zmm10 = _mm512_fmadd_ps(zmm0, zmm3, zmm10);
        zmm11 = _mm512_fmadd_ps(zmm1, zmm3, zmm11);
        _cbuf += rs_c;

        zmm0 = _mm512_load_ps(_cbuf);
        zmm1 = _mm512_load_ps(_cbuf + 16);
        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);

        zmm0 = _mm512_load_ps(_cbuf + 32);
        zmm1 = _mm512_load_ps(_cbuf + 48);
        zmm14 = _mm512_fmadd_ps(zmm0, zmm3, zmm14);
        zmm15 = _mm512_fmadd_ps(zmm1, zmm3, zmm15);
        _cbuf += rs_c;

        zmm0 = _mm512_load_ps(_cbuf);
        zmm1 = _mm512_load_ps(_cbuf+16);
        zmm16 = _mm512_fmadd_ps(zmm0, zmm3, zmm16);
        zmm17 = _mm512_fmadd_ps(zmm1, zmm3, zmm17);

        zmm0 = _mm512_load_ps(_cbuf + 32);
        zmm1 = _mm512_load_ps(_cbuf + 48);
        zmm18 = _mm512_fmadd_ps(zmm0, zmm3, zmm18);
        zmm19 = _mm512_fmadd_ps(zmm1, zmm3, zmm19);
        _cbuf += rs_c;

        zmm0 = _mm512_load_ps(_cbuf);
        zmm1 = _mm512_load_ps(_cbuf+16);
        zmm20 = _mm512_fmadd_ps(zmm0, zmm3, zmm20);
        zmm21 = _mm512_fmadd_ps(zmm1, zmm3, zmm21);

        zmm0 = _mm512_load_ps(_cbuf + 32);
        zmm1 = _mm512_load_ps(_cbuf + 48);
        zmm22 = _mm512_fmadd_ps(zmm0, zmm3, zmm22);
        zmm23 = _mm512_fmadd_ps(zmm1, zmm3, zmm23);
        _cbuf += rs_c;

        zmm0 = _mm512_load_ps(_cbuf);
        zmm1 = _mm512_load_ps(_cbuf+16);
        zmm24 = _mm512_fmadd_ps(zmm0, zmm3, zmm24);
        zmm25 = _mm512_fmadd_ps(zmm1, zmm3, zmm25);

        zmm0 = _mm512_load_ps(_cbuf + 32);
        zmm1 = _mm512_load_ps(_cbuf + 48);
        zmm26 = _mm512_fmadd_ps(zmm0, zmm3, zmm26);
        zmm27 = _mm512_fmadd_ps(zmm1, zmm3, zmm27);
    }

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_5x64F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        zmm1 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        zmm2 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 1 * 16 ) );
        zmm3 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 2 * 16 ) );
        zmm4 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 3 * 16 ) );

        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[0, 16-31]
        zmm9 = _mm512_add_ps( zmm2, zmm9 );

        // c[0,32-47]
        zmm10 = _mm512_add_ps( zmm3, zmm10 );

        // c[0,48-63]
        zmm11 = _mm512_add_ps( zmm4, zmm11 );

        // c[1,0-15]
        zmm12 = _mm512_add_ps( zmm1, zmm12 );

        // c[1, 16-31]
        zmm13 = _mm512_add_ps( zmm2, zmm13 );

        // c[1,32-47]
        zmm14 = _mm512_add_ps( zmm3, zmm14 );

        // c[1,48-63]
        zmm15 = _mm512_add_ps( zmm4, zmm15 );

        // c[2,0-15]
        zmm16 = _mm512_add_ps( zmm1, zmm16 );

        // c[2, 16-31]
        zmm17 = _mm512_add_ps( zmm2, zmm17 );

        // c[2,32-47]
        zmm18 = _mm512_add_ps( zmm3, zmm18 );

        // c[2,48-63]
        zmm19 = _mm512_add_ps( zmm4, zmm19 );

        // c[3,0-15]
        zmm20 = _mm512_add_ps( zmm1, zmm20 );

        // c[3, 16-31]
        zmm21 = _mm512_add_ps( zmm2, zmm21 );

        // c[3,32-47]
        zmm22 = _mm512_add_ps( zmm3, zmm22 );

        // c[3,48-63]
        zmm23 = _mm512_add_ps( zmm4, zmm23 );

        // c[4,0-15]
        zmm24 = _mm512_add_ps( zmm1, zmm24 );

        // c[4, 16-31]
        zmm25 = _mm512_add_ps( zmm2, zmm25 );

        // c[4,32-47]
        zmm26 = _mm512_add_ps( zmm3, zmm26 );

        // c[4,48-63]
        zmm27 = _mm512_add_ps( zmm4, zmm27 );
      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the bias array will be accessed by
        // the ic index, and each bias element corresponds to an
        // entire row of the transposed output array, instead of an
        // entire column.
        zmm1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 0 ) );
        zmm2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 1 ) );
        zmm3 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 2 ) );
        zmm4 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 3 ) );
        zmm5 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 4 ) );

        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[0, 16-31]
        zmm9 = _mm512_add_ps( zmm1, zmm9 );

        // c[0,32-47]
        zmm10 = _mm512_add_ps( zmm1, zmm10 );

        // c[0,48-63]
        zmm11 = _mm512_add_ps( zmm1, zmm11 );

        // c[1,0-15]
        zmm12 = _mm512_add_ps( zmm2, zmm12 );

        // c[1, 16-31]
        zmm13 = _mm512_add_ps( zmm2, zmm13 );

        // c[1,32-47]
        zmm14 = _mm512_add_ps( zmm2, zmm14 );

        // c[1,48-63]
        zmm15 = _mm512_add_ps( zmm2, zmm15 );

        // c[2,0-15]
        zmm16 = _mm512_add_ps( zmm3, zmm16 );

        // c[2, 16-31]
        zmm17 = _mm512_add_ps( zmm3, zmm17 );

        // c[2,32-47]
        zmm18 = _mm512_add_ps( zmm3, zmm18 );

        // c[2,48-63]
        zmm19 = _mm512_add_ps( zmm3, zmm19 );

        // c[3,0-15]
        zmm20 = _mm512_add_ps( zmm4, zmm20 );

        // c[3, 16-31]
        zmm21 = _mm512_add_ps( zmm4, zmm21 );

        // c[3,32-47]
        zmm22 = _mm512_add_ps( zmm4, zmm22 );

        // c[3,48-63]
        zmm23 = _mm512_add_ps( zmm4, zmm23 );

        // c[4,0-15]
        zmm24 = _mm512_add_ps( zmm5, zmm24 );

        // c[4, 16-31]
        zmm25 = _mm512_add_ps( zmm5, zmm25 );

        // c[4,32-47]
        zmm26 = _mm512_add_ps( zmm5, zmm26 );

        // c[4,48-63]
        zmm27 = _mm512_add_ps( zmm5, zmm27 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_5x64F:
    {
      zmm1 = _mm512_setzero_ps();

      // c[0,0-15]
      zmm8 = _mm512_max_ps( zmm1, zmm8 );

      // c[0, 16-31]
      zmm9 = _mm512_max_ps( zmm1, zmm9 );

      // c[0,32-47]
      zmm10 = _mm512_max_ps( zmm1, zmm10 );

      // c[0,48-63]
      zmm11 = _mm512_max_ps( zmm1, zmm11 );

      // c[1,0-15]
      zmm12 = _mm512_max_ps( zmm1, zmm12 );

      // c[1,16-31]
      zmm13 = _mm512_max_ps( zmm1, zmm13 );

      // c[1,32-47]
      zmm14 = _mm512_max_ps( zmm1, zmm14 );

      // c[1,48-63]
      zmm15 = _mm512_max_ps( zmm1, zmm15 );

      // c[2,0-15]
      zmm16 = _mm512_max_ps( zmm1, zmm16 );

      // c[2,16-31]
      zmm17 = _mm512_max_ps( zmm1, zmm17 );

      // c[2,32-47]
      zmm18 = _mm512_max_ps( zmm1, zmm18 );

      // c[2,48-63]
      zmm19 = _mm512_max_ps( zmm1, zmm19 );

      // c[3,0-15]
      zmm20 = _mm512_max_ps( zmm1, zmm20 );

      // c[3,16-31]
      zmm21 = _mm512_max_ps( zmm1, zmm21 );

      // c[3,32-47]
      zmm22 = _mm512_max_ps( zmm1, zmm22 );

      // c[3,48-63]
      zmm23 = _mm512_max_ps( zmm1, zmm23 );

      // c[4,0-15]
      zmm24 = _mm512_max_ps( zmm1, zmm24 );

      // c[4,16-31]
      zmm25 = _mm512_max_ps( zmm1, zmm25 );

      // c[4,32-47]
      zmm26 = _mm512_max_ps( zmm1, zmm26 );

      // c[4,48-63]
      zmm27 = _mm512_max_ps( zmm1, zmm27 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_5x64F:
    {
      zmm1 = _mm512_setzero_ps();
      zmm2 =
        _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

      __mmask16 relu_cmp_mask;

      // c[0, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm8)

      // c[0, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm9)

      // c[0, 32-47]
      RELU_SCALE_OP_F32S_AVX512(zmm10)

      // c[0, 48-63]
      RELU_SCALE_OP_F32S_AVX512(zmm11)

      // c[1, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm12)

      // c[1, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm13)

      // c[1, 32-47]
      RELU_SCALE_OP_F32S_AVX512(zmm14)

      // c[1, 48-63]
      RELU_SCALE_OP_F32S_AVX512(zmm15)

      // c[2, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm16)

      // c[2, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm17)

      // c[2, 32-47]
      RELU_SCALE_OP_F32S_AVX512(zmm18)

      // c[2, 48-63]
      RELU_SCALE_OP_F32S_AVX512(zmm19)

      // c[3, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm20)

      // c[3, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm21)

      // c[3, 32-47]
      RELU_SCALE_OP_F32S_AVX512(zmm22)

      // c[3, 48-63]
      RELU_SCALE_OP_F32S_AVX512(zmm23)

      // c[4, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm24)

      // c[4, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm25)

      // c[4, 32-47]
      RELU_SCALE_OP_F32S_AVX512(zmm26)

      // c[4, 48-63]
      RELU_SCALE_OP_F32S_AVX512(zmm27)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_5x64F:
    {
      __m512i zmm6;
      // c[0, 0-15]
      GELU_TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[0, 16-31]
      GELU_TANH_F32S_AVX512(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[0, 32-47]
      GELU_TANH_F32S_AVX512(zmm10, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[0, 48-63]
      GELU_TANH_F32S_AVX512(zmm11, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 0-15]
      GELU_TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 16-31]
      GELU_TANH_F32S_AVX512(zmm13, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 32-47]
      GELU_TANH_F32S_AVX512(zmm14, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 48-63]
      GELU_TANH_F32S_AVX512(zmm15, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[2, 0-15]
      GELU_TANH_F32S_AVX512(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[2, 16-31]
      GELU_TANH_F32S_AVX512(zmm17, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[2, 32-47]
      GELU_TANH_F32S_AVX512(zmm18, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[2, 48-63]
      GELU_TANH_F32S_AVX512(zmm19, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[3, 0-15]
      GELU_TANH_F32S_AVX512(zmm20, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[3, 16-31]
      GELU_TANH_F32S_AVX512(zmm21, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[3, 32-47]
      GELU_TANH_F32S_AVX512(zmm22, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[3, 48-63]
      GELU_TANH_F32S_AVX512(zmm23, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[4, 0-15]
      GELU_TANH_F32S_AVX512(zmm24, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[4, 16-31]
      GELU_TANH_F32S_AVX512(zmm25, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[4, 32-47]
      GELU_TANH_F32S_AVX512(zmm26, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[4, 48-63]
      GELU_TANH_F32S_AVX512(zmm27, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_5x64F:
    {
      // c[0, 0-15]
      GELU_ERF_F32S_AVX512(zmm8, zmm0, zmm1, zmm2)

      // c[0, 16-31]
      GELU_ERF_F32S_AVX512(zmm9, zmm0, zmm1, zmm2)

      // c[0, 32-47]
      GELU_ERF_F32S_AVX512(zmm10, zmm0, zmm1, zmm2)

      // c[0, 48-63]
      GELU_ERF_F32S_AVX512(zmm11, zmm0, zmm1, zmm2)

      // c[1, 0-15]
      GELU_ERF_F32S_AVX512(zmm12, zmm0, zmm1, zmm2)

      // c[1, 16-31]
      GELU_ERF_F32S_AVX512(zmm13, zmm0, zmm1, zmm2)

      // c[1, 32-47]
      GELU_ERF_F32S_AVX512(zmm14, zmm0, zmm1, zmm2)

      // c[1, 48-63]
      GELU_ERF_F32S_AVX512(zmm15, zmm0, zmm1, zmm2)

      // c[2, 0-15]
      GELU_ERF_F32S_AVX512(zmm16, zmm0, zmm1, zmm2)

      // c[2, 16-31]
      GELU_ERF_F32S_AVX512(zmm17, zmm0, zmm1, zmm2)

      // c[2, 32-47]
      GELU_ERF_F32S_AVX512(zmm18, zmm0, zmm1, zmm2)

      // c[2, 48-63]
      GELU_ERF_F32S_AVX512(zmm19, zmm0, zmm1, zmm2)

      // c[3, 0-15]
      GELU_ERF_F32S_AVX512(zmm20, zmm0, zmm1, zmm2)

      // c[3, 16-31]
      GELU_ERF_F32S_AVX512(zmm21, zmm0, zmm1, zmm2)

      // c[3, 32-47]
      GELU_ERF_F32S_AVX512(zmm22, zmm0, zmm1, zmm2)

      // c[3, 48-63]
      GELU_ERF_F32S_AVX512(zmm23, zmm0, zmm1, zmm2)

      // c[4, 0-15]
      GELU_ERF_F32S_AVX512(zmm24, zmm0, zmm1, zmm2)

      // c[4, 16-31]
      GELU_ERF_F32S_AVX512(zmm25, zmm0, zmm1, zmm2)

      // c[4, 32-47]
      GELU_ERF_F32S_AVX512(zmm26, zmm0, zmm1, zmm2)

      // c[4, 48-63]
      GELU_ERF_F32S_AVX512(zmm27, zmm0, zmm1, zmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_5x64F:
    {
      zmm0 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      zmm1 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0, 0-15]
      CLIP_F32S_AVX512(zmm8, zmm0, zmm1)

      // c[0, 16-31]
      CLIP_F32S_AVX512(zmm9, zmm0, zmm1)

      // c[0, 32-47]
      CLIP_F32S_AVX512(zmm10, zmm0, zmm1)

      // c[0, 48-63]
      CLIP_F32S_AVX512(zmm11, zmm0, zmm1)

      // c[1, 0-15]
      CLIP_F32S_AVX512(zmm12, zmm0, zmm1)

      // c[1, 16-31]
      CLIP_F32S_AVX512(zmm13, zmm0, zmm1)

      // c[1, 32-47]
      CLIP_F32S_AVX512(zmm14, zmm0, zmm1)

      // c[1, 48-63]
      CLIP_F32S_AVX512(zmm15, zmm0, zmm1)

      // c[2, 0-15]
      CLIP_F32S_AVX512(zmm16, zmm0, zmm1)

      // c[2, 16-31]
      CLIP_F32S_AVX512(zmm17, zmm0, zmm1)

      // c[2, 32-47]
      CLIP_F32S_AVX512(zmm18, zmm0, zmm1)

      // c[2, 48-63]
      CLIP_F32S_AVX512(zmm19, zmm0, zmm1)

      // c[3, 0-15]
      CLIP_F32S_AVX512(zmm20, zmm0, zmm1)

      // c[3, 16-31]
      CLIP_F32S_AVX512(zmm21, zmm0, zmm1)

      // c[3, 32-47]
      CLIP_F32S_AVX512(zmm22, zmm0, zmm1)

      // c[3, 48-63]
      CLIP_F32S_AVX512(zmm23, zmm0, zmm1)

      // c[4, 0-15]
      CLIP_F32S_AVX512(zmm24, zmm0, zmm1)

      // c[4, 16-31]
      CLIP_F32S_AVX512(zmm25, zmm0, zmm1)

      // c[4, 32-47]
      CLIP_F32S_AVX512(zmm26, zmm0, zmm1)

      // c[4, 48-63]
      CLIP_F32S_AVX512(zmm27, zmm0, zmm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_5x64F_DISABLE:
    ;

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
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_4x64)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_4x64F_DISABLE,
              &&POST_OPS_BIAS_4x64F,
              &&POST_OPS_RELU_4x64F,
              &&POST_OPS_RELU_SCALE_4x64F,
              &&POST_OPS_GELU_TANH_4x64F,
              &&POST_OPS_GELU_ERF_4x64F,
              &&POST_OPS_CLIP_4x64F
            };
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
    float *_cbuf = NULL;

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

    if ( beta != 0.0 )
    {
        _cbuf = cbuf; 
        //load c and multiply with beta and 
        //add to accumulator and store back
        zmm3 = _mm512_set1_ps(beta);

        zmm0 = _mm512_load_ps(_cbuf);
        zmm1 = _mm512_load_ps(_cbuf + 16);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);

        zmm0 = _mm512_load_ps(_cbuf + 32);
        zmm1 = _mm512_load_ps(_cbuf + 48);
        zmm10 = _mm512_fmadd_ps(zmm0, zmm3, zmm10);
        zmm11 = _mm512_fmadd_ps(zmm1, zmm3, zmm11);
        _cbuf += rs_c;

        zmm0 = _mm512_load_ps(_cbuf);
        zmm1 = _mm512_load_ps(_cbuf + 16);
        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);

        zmm0 = _mm512_load_ps(_cbuf + 32);
        zmm1 = _mm512_load_ps(_cbuf + 48);
        zmm14 = _mm512_fmadd_ps(zmm0, zmm3, zmm14);
        zmm15 = _mm512_fmadd_ps(zmm1, zmm3, zmm15);
        _cbuf += rs_c;

        zmm0 = _mm512_load_ps(_cbuf);
        zmm1 = _mm512_load_ps(_cbuf+16);
        zmm16 = _mm512_fmadd_ps(zmm0, zmm3, zmm16);
        zmm17 = _mm512_fmadd_ps(zmm1, zmm3, zmm17);

        zmm0 = _mm512_load_ps(_cbuf + 32);
        zmm1 = _mm512_load_ps(_cbuf + 48);
        zmm18 = _mm512_fmadd_ps(zmm0, zmm3, zmm18);
        zmm19 = _mm512_fmadd_ps(zmm1, zmm3, zmm19);
        _cbuf += rs_c;

        zmm0 = _mm512_load_ps(_cbuf);
        zmm1 = _mm512_load_ps(_cbuf+16);
        zmm20 = _mm512_fmadd_ps(zmm0, zmm3, zmm20);
        zmm21 = _mm512_fmadd_ps(zmm1, zmm3, zmm21);

        zmm0 = _mm512_load_ps(_cbuf + 32);
        zmm1 = _mm512_load_ps(_cbuf + 48);
        zmm22 = _mm512_fmadd_ps(zmm0, zmm3, zmm22);
        zmm23 = _mm512_fmadd_ps(zmm1, zmm3, zmm23);
    }

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_4x64F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        zmm1 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        zmm2 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 1 * 16 ) );
        zmm3 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 2 * 16 ) );
        zmm4 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 3 * 16 ) );

        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[0, 16-31]
        zmm9 = _mm512_add_ps( zmm2, zmm9 );

        // c[0,32-47]
        zmm10 = _mm512_add_ps( zmm3, zmm10 );

        // c[0,48-63]
        zmm11 = _mm512_add_ps( zmm4, zmm11 );

        // c[1,0-15]
        zmm12 = _mm512_add_ps( zmm1, zmm12 );

        // c[1, 16-31]
        zmm13 = _mm512_add_ps( zmm2, zmm13 );

        // c[1,32-47]
        zmm14 = _mm512_add_ps( zmm3, zmm14 );

        // c[1,48-63]
        zmm15 = _mm512_add_ps( zmm4, zmm15 );

        // c[2,0-15]
        zmm16 = _mm512_add_ps( zmm1, zmm16 );

        // c[2, 16-31]
        zmm17 = _mm512_add_ps( zmm2, zmm17 );

        // c[2,32-47]
        zmm18 = _mm512_add_ps( zmm3, zmm18 );

        // c[2,48-63]
        zmm19 = _mm512_add_ps( zmm4, zmm19 );

        // c[3,0-15]
        zmm20 = _mm512_add_ps( zmm1, zmm20 );

        // c[3, 16-31]
        zmm21 = _mm512_add_ps( zmm2, zmm21 );

        // c[3,32-47]
        zmm22 = _mm512_add_ps( zmm3, zmm22 );

        // c[3,48-63]
        zmm23 = _mm512_add_ps( zmm4, zmm23 );
      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the bias array will be accessed by
        // the ic index, and each bias element corresponds to an
        // entire row of the transposed output array, instead of an
        // entire column.
        zmm1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 0 ) );
        zmm2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 1 ) );
        zmm3 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 2 ) );
        zmm4 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 3 ) );

        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[0, 16-31]
        zmm9 = _mm512_add_ps( zmm1, zmm9 );

        // c[0,32-47]
        zmm10 = _mm512_add_ps( zmm1, zmm10 );

        // c[0,48-63]
        zmm11 = _mm512_add_ps( zmm1, zmm11 );

        // c[1,0-15]
        zmm12 = _mm512_add_ps( zmm2, zmm12 );

        // c[1, 16-31]
        zmm13 = _mm512_add_ps( zmm2, zmm13 );

        // c[1,32-47]
        zmm14 = _mm512_add_ps( zmm2, zmm14 );

        // c[1,48-63]
        zmm15 = _mm512_add_ps( zmm2, zmm15 );

        // c[2,0-15]
        zmm16 = _mm512_add_ps( zmm3, zmm16 );

        // c[2, 16-31]
        zmm17 = _mm512_add_ps( zmm3, zmm17 );

        // c[2,32-47]
        zmm18 = _mm512_add_ps( zmm3, zmm18 );

        // c[2,48-63]
        zmm19 = _mm512_add_ps( zmm3, zmm19 );

        // c[3,0-15]
        zmm20 = _mm512_add_ps( zmm4, zmm20 );

        // c[3, 16-31]
        zmm21 = _mm512_add_ps( zmm4, zmm21 );

        // c[3,32-47]
        zmm22 = _mm512_add_ps( zmm4, zmm22 );

        // c[3,48-63]
        zmm23 = _mm512_add_ps( zmm4, zmm23 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_4x64F:
    {
      zmm1 = _mm512_setzero_ps();

      // c[0,0-15]
      zmm8 = _mm512_max_ps( zmm1, zmm8 );

      // c[0, 16-31]
      zmm9 = _mm512_max_ps( zmm1, zmm9 );

      // c[0,32-47]
      zmm10 = _mm512_max_ps( zmm1, zmm10 );

      // c[0,48-63]
      zmm11 = _mm512_max_ps( zmm1, zmm11 );

      // c[1,0-15]
      zmm12 = _mm512_max_ps( zmm1, zmm12 );

      // c[1,16-31]
      zmm13 = _mm512_max_ps( zmm1, zmm13 );

      // c[1,32-47]
      zmm14 = _mm512_max_ps( zmm1, zmm14 );

      // c[1,48-63]
      zmm15 = _mm512_max_ps( zmm1, zmm15 );

      // c[2,0-15]
      zmm16 = _mm512_max_ps( zmm1, zmm16 );

      // c[2,16-31]
      zmm17 = _mm512_max_ps( zmm1, zmm17 );

      // c[2,32-47]
      zmm18 = _mm512_max_ps( zmm1, zmm18 );

      // c[2,48-63]
      zmm19 = _mm512_max_ps( zmm1, zmm19 );

      // c[3,0-15]
      zmm20 = _mm512_max_ps( zmm1, zmm20 );

      // c[3,16-31]
      zmm21 = _mm512_max_ps( zmm1, zmm21 );

      // c[3,32-47]
      zmm22 = _mm512_max_ps( zmm1, zmm22 );

      // c[3,48-63]
      zmm23 = _mm512_max_ps( zmm1, zmm23 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_4x64F:
    {
      zmm1 = _mm512_setzero_ps();
      zmm2 =
        _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

      __mmask16 relu_cmp_mask;

      // c[0, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm8)

      // c[0, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm9)

      // c[0, 32-47]
      RELU_SCALE_OP_F32S_AVX512(zmm10)

      // c[0, 48-63]
      RELU_SCALE_OP_F32S_AVX512(zmm11)

      // c[1, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm12)

      // c[1, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm13)

      // c[1, 32-47]
      RELU_SCALE_OP_F32S_AVX512(zmm14)

      // c[1, 48-63]
      RELU_SCALE_OP_F32S_AVX512(zmm15)

      // c[2, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm16)

      // c[2, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm17)

      // c[2, 32-47]
      RELU_SCALE_OP_F32S_AVX512(zmm18)

      // c[2, 48-63]
      RELU_SCALE_OP_F32S_AVX512(zmm19)

      // c[3, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm20)

      // c[3, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm21)

      // c[3, 32-47]
      RELU_SCALE_OP_F32S_AVX512(zmm22)

      // c[3, 48-63]
      RELU_SCALE_OP_F32S_AVX512(zmm23)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_4x64F:
    {
      __m512i zmm6;
      // c[0, 0-15]
      GELU_TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[0, 16-31]
      GELU_TANH_F32S_AVX512(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[0, 32-47]
      GELU_TANH_F32S_AVX512(zmm10, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[0, 48-63]
      GELU_TANH_F32S_AVX512(zmm11, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 0-15]
      GELU_TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 16-31]
      GELU_TANH_F32S_AVX512(zmm13, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 32-47]
      GELU_TANH_F32S_AVX512(zmm14, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 48-63]
      GELU_TANH_F32S_AVX512(zmm15, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[2, 0-15]
      GELU_TANH_F32S_AVX512(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[2, 16-31]
      GELU_TANH_F32S_AVX512(zmm17, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[2, 32-47]
      GELU_TANH_F32S_AVX512(zmm18, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[2, 48-63]
      GELU_TANH_F32S_AVX512(zmm19, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[3, 0-15]
      GELU_TANH_F32S_AVX512(zmm20, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[3, 16-31]
      GELU_TANH_F32S_AVX512(zmm21, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[3, 32-47]
      GELU_TANH_F32S_AVX512(zmm22, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[3, 48-63]
      GELU_TANH_F32S_AVX512(zmm23, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_4x64F:
    {
      // c[0, 0-15]
      GELU_ERF_F32S_AVX512(zmm8, zmm0, zmm1, zmm2)

      // c[0, 16-31]
      GELU_ERF_F32S_AVX512(zmm9, zmm0, zmm1, zmm2)

      // c[0, 32-47]
      GELU_ERF_F32S_AVX512(zmm10, zmm0, zmm1, zmm2)

      // c[0, 48-63]
      GELU_ERF_F32S_AVX512(zmm11, zmm0, zmm1, zmm2)

      // c[1, 0-15]
      GELU_ERF_F32S_AVX512(zmm12, zmm0, zmm1, zmm2)

      // c[1, 16-31]
      GELU_ERF_F32S_AVX512(zmm13, zmm0, zmm1, zmm2)

      // c[1, 32-47]
      GELU_ERF_F32S_AVX512(zmm14, zmm0, zmm1, zmm2)

      // c[1, 48-63]
      GELU_ERF_F32S_AVX512(zmm15, zmm0, zmm1, zmm2)

      // c[2, 0-15]
      GELU_ERF_F32S_AVX512(zmm16, zmm0, zmm1, zmm2)

      // c[2, 16-31]
      GELU_ERF_F32S_AVX512(zmm17, zmm0, zmm1, zmm2)

      // c[2, 32-47]
      GELU_ERF_F32S_AVX512(zmm18, zmm0, zmm1, zmm2)

      // c[2, 48-63]
      GELU_ERF_F32S_AVX512(zmm19, zmm0, zmm1, zmm2)

      // c[3, 0-15]
      GELU_ERF_F32S_AVX512(zmm20, zmm0, zmm1, zmm2)

      // c[3, 16-31]
      GELU_ERF_F32S_AVX512(zmm21, zmm0, zmm1, zmm2)

      // c[3, 32-47]
      GELU_ERF_F32S_AVX512(zmm22, zmm0, zmm1, zmm2)

      // c[3, 48-63]
      GELU_ERF_F32S_AVX512(zmm23, zmm0, zmm1, zmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_4x64F:
    {
      zmm0 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      zmm1 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0, 0-15]
      CLIP_F32S_AVX512(zmm8, zmm0, zmm1)

      // c[0, 16-31]
      CLIP_F32S_AVX512(zmm9, zmm0, zmm1)

      // c[0, 32-47]
      CLIP_F32S_AVX512(zmm10, zmm0, zmm1)

      // c[0, 48-63]
      CLIP_F32S_AVX512(zmm11, zmm0, zmm1)

      // c[1, 0-15]
      CLIP_F32S_AVX512(zmm12, zmm0, zmm1)

      // c[1, 16-31]
      CLIP_F32S_AVX512(zmm13, zmm0, zmm1)

      // c[1, 32-47]
      CLIP_F32S_AVX512(zmm14, zmm0, zmm1)

      // c[1, 48-63]
      CLIP_F32S_AVX512(zmm15, zmm0, zmm1)

      // c[2, 0-15]
      CLIP_F32S_AVX512(zmm16, zmm0, zmm1)

      // c[2, 16-31]
      CLIP_F32S_AVX512(zmm17, zmm0, zmm1)

      // c[2, 32-47]
      CLIP_F32S_AVX512(zmm18, zmm0, zmm1)

      // c[2, 48-63]
      CLIP_F32S_AVX512(zmm19, zmm0, zmm1)

      // c[3, 0-15]
      CLIP_F32S_AVX512(zmm20, zmm0, zmm1)

      // c[3, 16-31]
      CLIP_F32S_AVX512(zmm21, zmm0, zmm1)

      // c[3, 32-47]
      CLIP_F32S_AVX512(zmm22, zmm0, zmm1)

      // c[3, 48-63]
      CLIP_F32S_AVX512(zmm23, zmm0, zmm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_4x64F_DISABLE:
    ;

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
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_3x64)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_3x64F_DISABLE,
              &&POST_OPS_BIAS_3x64F,
              &&POST_OPS_RELU_3x64F,
              &&POST_OPS_RELU_SCALE_3x64F,
              &&POST_OPS_GELU_TANH_3x64F,
              &&POST_OPS_GELU_ERF_3x64F,
              &&POST_OPS_CLIP_3x64F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k0;

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7;
    __m512 zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15;
    __m512 zmm16, zmm17, zmm18, zmm19;
    
    /* zero the accumulator registers */
    ZERO_ACC_ZMM_4_REG(zmm8, zmm9, zmm10, zmm11);
    ZERO_ACC_ZMM_4_REG(zmm12, zmm13, zmm14, zmm15);
    ZERO_ACC_ZMM_4_REG(zmm16, zmm17, zmm18, zmm19);

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;
    float *_cbuf = NULL;

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

    if ( beta != 0.0 )
    {
        _cbuf = cbuf;
        //load c and multiply with beta and 
        //add to accumulator and store back
        zmm3 = _mm512_set1_ps(beta);

        zmm0 = _mm512_load_ps(_cbuf);
        zmm1 = _mm512_load_ps(_cbuf + 16);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);

        zmm0 = _mm512_load_ps(_cbuf + 32);
        zmm1 = _mm512_load_ps(_cbuf + 48);
        zmm10 = _mm512_fmadd_ps(zmm0, zmm3, zmm10);
        zmm11 = _mm512_fmadd_ps(zmm1, zmm3, zmm11);
        _cbuf += rs_c;

        zmm0 = _mm512_load_ps(_cbuf);
        zmm1 = _mm512_load_ps(_cbuf + 16);
        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);

        zmm0 = _mm512_load_ps(_cbuf + 32);
        zmm1 = _mm512_load_ps(_cbuf + 48);
        zmm14 = _mm512_fmadd_ps(zmm0, zmm3, zmm14);
        zmm15 = _mm512_fmadd_ps(zmm1, zmm3, zmm15);
        _cbuf += rs_c;

        zmm0 = _mm512_load_ps(_cbuf);
        zmm1 = _mm512_load_ps(_cbuf+16);
        zmm16 = _mm512_fmadd_ps(zmm0, zmm3, zmm16);
        zmm17 = _mm512_fmadd_ps(zmm1, zmm3, zmm17);

        zmm0 = _mm512_load_ps(_cbuf + 32);
        zmm1 = _mm512_load_ps(_cbuf + 48);
        zmm18 = _mm512_fmadd_ps(zmm0, zmm3, zmm18);
        zmm19 = _mm512_fmadd_ps(zmm1, zmm3, zmm19);
    }

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_3x64F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        zmm1 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        zmm2 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 1 * 16 ) );
        zmm3 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 2 * 16 ) );
        zmm4 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 3 * 16 ) );

        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[0, 16-31]
        zmm9 = _mm512_add_ps( zmm2, zmm9 );

        // c[0,32-47]
        zmm10 = _mm512_add_ps( zmm3, zmm10 );

        // c[0,48-63]
        zmm11 = _mm512_add_ps( zmm4, zmm11 );

        // c[1,0-15]
        zmm12 = _mm512_add_ps( zmm1, zmm12 );

        // c[1, 16-31]
        zmm13 = _mm512_add_ps( zmm2, zmm13 );

        // c[1,32-47]
        zmm14 = _mm512_add_ps( zmm3, zmm14 );

        // c[1,48-63]
        zmm15 = _mm512_add_ps( zmm4, zmm15 );

        // c[2,0-15]
        zmm16 = _mm512_add_ps( zmm1, zmm16 );

        // c[2, 16-31]
        zmm17 = _mm512_add_ps( zmm2, zmm17 );

        // c[2,32-47]
        zmm18 = _mm512_add_ps( zmm3, zmm18 );

        // c[2,48-63]
        zmm19 = _mm512_add_ps( zmm4, zmm19 );
      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the bias array will be accessed by
        // the ic index, and each bias element corresponds to an
        // entire row of the transposed output array, instead of an
        // entire column.
        zmm1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 0 ) );
        zmm2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 1 ) );
        zmm3 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 2 ) );

        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[0, 16-31]
        zmm9 = _mm512_add_ps( zmm1, zmm9 );

        // c[0,32-47]
        zmm10 = _mm512_add_ps( zmm1, zmm10 );

        // c[0,48-63]
        zmm11 = _mm512_add_ps( zmm1, zmm11 );

        // c[1,0-15]
        zmm12 = _mm512_add_ps( zmm2, zmm12 );

        // c[1, 16-31]
        zmm13 = _mm512_add_ps( zmm2, zmm13 );

        // c[1,32-47]
        zmm14 = _mm512_add_ps( zmm2, zmm14 );

        // c[1,48-63]
        zmm15 = _mm512_add_ps( zmm2, zmm15 );

        // c[2,0-15]
        zmm16 = _mm512_add_ps( zmm3, zmm16 );

        // c[2, 16-31]
        zmm17 = _mm512_add_ps( zmm3, zmm17 );

        // c[2,32-47]
        zmm18 = _mm512_add_ps( zmm3, zmm18 );

        // c[2,48-63]
        zmm19 = _mm512_add_ps( zmm3, zmm19 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_3x64F:
    {
      zmm1 = _mm512_setzero_ps();

      // c[0,0-15]
      zmm8 = _mm512_max_ps( zmm1, zmm8 );

      // c[0, 16-31]
      zmm9 = _mm512_max_ps( zmm1, zmm9 );

      // c[0,32-47]
      zmm10 = _mm512_max_ps( zmm1, zmm10 );

      // c[0,48-63]
      zmm11 = _mm512_max_ps( zmm1, zmm11 );

      // c[1,0-15]
      zmm12 = _mm512_max_ps( zmm1, zmm12 );

      // c[1,16-31]
      zmm13 = _mm512_max_ps( zmm1, zmm13 );

      // c[1,32-47]
      zmm14 = _mm512_max_ps( zmm1, zmm14 );

      // c[1,48-63]
      zmm15 = _mm512_max_ps( zmm1, zmm15 );

      // c[2,0-15]
      zmm16 = _mm512_max_ps( zmm1, zmm16 );

      // c[2,16-31]
      zmm17 = _mm512_max_ps( zmm1, zmm17 );

      // c[2,32-47]
      zmm18 = _mm512_max_ps( zmm1, zmm18 );

      // c[2,48-63]
      zmm19 = _mm512_max_ps( zmm1, zmm19 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_3x64F:
    {
      zmm1 = _mm512_setzero_ps();
      zmm2 =
        _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

      __mmask16 relu_cmp_mask;

      // c[0, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm8)

      // c[0, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm9)

      // c[0, 32-47]
      RELU_SCALE_OP_F32S_AVX512(zmm10)

      // c[0, 48-63]
      RELU_SCALE_OP_F32S_AVX512(zmm11)

      // c[1, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm12)

      // c[1, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm13)

      // c[1, 32-47]
      RELU_SCALE_OP_F32S_AVX512(zmm14)

      // c[1, 48-63]
      RELU_SCALE_OP_F32S_AVX512(zmm15)

      // c[2, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm16)

      // c[2, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm17)

      // c[2, 32-47]
      RELU_SCALE_OP_F32S_AVX512(zmm18)

      // c[2, 48-63]
      RELU_SCALE_OP_F32S_AVX512(zmm19)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_3x64F:
    {
      __m512i zmm6;
      // c[0, 0-15]
      GELU_TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[0, 16-31]
      GELU_TANH_F32S_AVX512(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[0, 32-47]
      GELU_TANH_F32S_AVX512(zmm10, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[0, 48-63]
      GELU_TANH_F32S_AVX512(zmm11, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 0-15]
      GELU_TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 16-31]
      GELU_TANH_F32S_AVX512(zmm13, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 32-47]
      GELU_TANH_F32S_AVX512(zmm14, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 48-63]
      GELU_TANH_F32S_AVX512(zmm15, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[2, 0-15]
      GELU_TANH_F32S_AVX512(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[2, 16-31]
      GELU_TANH_F32S_AVX512(zmm17, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[2, 32-47]
      GELU_TANH_F32S_AVX512(zmm18, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[2, 48-63]
      GELU_TANH_F32S_AVX512(zmm19, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_3x64F:
    {
      // c[0, 0-15]
      GELU_ERF_F32S_AVX512(zmm8, zmm0, zmm1, zmm2)

      // c[0, 16-31]
      GELU_ERF_F32S_AVX512(zmm9, zmm0, zmm1, zmm2)

      // c[0, 32-47]
      GELU_ERF_F32S_AVX512(zmm10, zmm0, zmm1, zmm2)

      // c[0, 48-63]
      GELU_ERF_F32S_AVX512(zmm11, zmm0, zmm1, zmm2)

      // c[1, 0-15]
      GELU_ERF_F32S_AVX512(zmm12, zmm0, zmm1, zmm2)

      // c[1, 16-31]
      GELU_ERF_F32S_AVX512(zmm13, zmm0, zmm1, zmm2)

      // c[1, 32-47]
      GELU_ERF_F32S_AVX512(zmm14, zmm0, zmm1, zmm2)

      // c[1, 48-63]
      GELU_ERF_F32S_AVX512(zmm15, zmm0, zmm1, zmm2)

      // c[2, 0-15]
      GELU_ERF_F32S_AVX512(zmm16, zmm0, zmm1, zmm2)

      // c[2, 16-31]
      GELU_ERF_F32S_AVX512(zmm17, zmm0, zmm1, zmm2)

      // c[2, 32-47]
      GELU_ERF_F32S_AVX512(zmm18, zmm0, zmm1, zmm2)

      // c[2, 48-63]
      GELU_ERF_F32S_AVX512(zmm19, zmm0, zmm1, zmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_3x64F:
    {
      zmm0 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      zmm1 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0, 0-15]
      CLIP_F32S_AVX512(zmm8, zmm0, zmm1)

      // c[0, 16-31]
      CLIP_F32S_AVX512(zmm9, zmm0, zmm1)

      // c[0, 32-47]
      CLIP_F32S_AVX512(zmm10, zmm0, zmm1)

      // c[0, 48-63]
      CLIP_F32S_AVX512(zmm11, zmm0, zmm1)

      // c[1, 0-15]
      CLIP_F32S_AVX512(zmm12, zmm0, zmm1)

      // c[1, 16-31]
      CLIP_F32S_AVX512(zmm13, zmm0, zmm1)

      // c[1, 32-47]
      CLIP_F32S_AVX512(zmm14, zmm0, zmm1)

      // c[1, 48-63]
      CLIP_F32S_AVX512(zmm15, zmm0, zmm1)

      // c[2, 0-15]
      CLIP_F32S_AVX512(zmm16, zmm0, zmm1)

      // c[2, 16-31]
      CLIP_F32S_AVX512(zmm17, zmm0, zmm1)

      // c[2, 32-47]
      CLIP_F32S_AVX512(zmm18, zmm0, zmm1)

      // c[2, 48-63]
      CLIP_F32S_AVX512(zmm19, zmm0, zmm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_3x64F_DISABLE:
    ;

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
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_2x64)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_2x64F_DISABLE,
              &&POST_OPS_BIAS_2x64F,
              &&POST_OPS_RELU_2x64F,
              &&POST_OPS_RELU_SCALE_2x64F,
              &&POST_OPS_GELU_TANH_2x64F,
              &&POST_OPS_GELU_ERF_2x64F,
              &&POST_OPS_CLIP_2x64F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k0;

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7;
    __m512 zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15;
    
    /* zero the accumulator registers */
    ZERO_ACC_ZMM_4_REG(zmm8, zmm9, zmm10, zmm11);
    ZERO_ACC_ZMM_4_REG(zmm12, zmm13, zmm14, zmm15);

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;
    float *_cbuf = NULL;

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

    if ( beta != 0.0 )
    {
        _cbuf = cbuf;
        //load c and multiply with beta and 
        //add to accumulator and store back
        zmm3 = _mm512_set1_ps(beta);

        zmm0 = _mm512_load_ps(_cbuf);
        zmm1 = _mm512_load_ps(_cbuf + 16);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);

        zmm0 = _mm512_load_ps(_cbuf + 32);
        zmm1 = _mm512_load_ps(_cbuf + 48);
        zmm10 = _mm512_fmadd_ps(zmm0, zmm3, zmm10);
        zmm11 = _mm512_fmadd_ps(zmm1, zmm3, zmm11);
        _cbuf += rs_c;

        zmm0 = _mm512_load_ps(_cbuf);
        zmm1 = _mm512_load_ps(_cbuf + 16);
        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);

        zmm0 = _mm512_load_ps(_cbuf + 32);
        zmm1 = _mm512_load_ps(_cbuf + 48);
        zmm14 = _mm512_fmadd_ps(zmm0, zmm3, zmm14);
        zmm15 = _mm512_fmadd_ps(zmm1, zmm3, zmm15);
    }

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_2x64F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        zmm1 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        zmm2 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 1 * 16 ) );
        zmm3 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 2 * 16 ) );
        zmm4 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 3 * 16 ) );

        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[0, 16-31]
        zmm9 = _mm512_add_ps( zmm2, zmm9 );

        // c[0,32-47]
        zmm10 = _mm512_add_ps( zmm3, zmm10 );

        // c[0,48-63]
        zmm11 = _mm512_add_ps( zmm4, zmm11 );

        // c[1,0-15]
        zmm12 = _mm512_add_ps( zmm1, zmm12 );

        // c[1, 16-31]
        zmm13 = _mm512_add_ps( zmm2, zmm13 );

        // c[1,32-47]
        zmm14 = _mm512_add_ps( zmm3, zmm14 );

        // c[1,48-63]
        zmm15 = _mm512_add_ps( zmm4, zmm15 );
      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the bias array will be accessed by
        // the ic index, and each bias element corresponds to an
        // entire row of the transposed output array, instead of an
        // entire column.
        zmm1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 0 ) );
        zmm2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 1 ) );

        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[0, 16-31]
        zmm9 = _mm512_add_ps( zmm1, zmm9 );

        // c[0,32-47]
        zmm10 = _mm512_add_ps( zmm1, zmm10 );

        // c[0,48-63]
        zmm11 = _mm512_add_ps( zmm1, zmm11 );

        // c[1,0-15]
        zmm12 = _mm512_add_ps( zmm2, zmm12 );

        // c[1, 16-31]
        zmm13 = _mm512_add_ps( zmm2, zmm13 );

        // c[1,32-47]
        zmm14 = _mm512_add_ps( zmm2, zmm14 );

        // c[1,48-63]
        zmm15 = _mm512_add_ps( zmm2, zmm15 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_2x64F:
    {
      zmm1 = _mm512_setzero_ps();

      // c[0,0-15]
      zmm8 = _mm512_max_ps( zmm1, zmm8 );

      // c[0, 16-31]
      zmm9 = _mm512_max_ps( zmm1, zmm9 );

      // c[0,32-47]
      zmm10 = _mm512_max_ps( zmm1, zmm10 );

      // c[0,48-63]
      zmm11 = _mm512_max_ps( zmm1, zmm11 );

      // c[1,0-15]
      zmm12 = _mm512_max_ps( zmm1, zmm12 );

      // c[1,16-31]
      zmm13 = _mm512_max_ps( zmm1, zmm13 );

      // c[1,32-47]
      zmm14 = _mm512_max_ps( zmm1, zmm14 );

      // c[1,48-63]
      zmm15 = _mm512_max_ps( zmm1, zmm15 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_2x64F:
    {
      zmm1 = _mm512_setzero_ps();
      zmm2 =
        _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

      __mmask16 relu_cmp_mask;

      // c[0, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm8)

      // c[0, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm9)

      // c[0, 32-47]
      RELU_SCALE_OP_F32S_AVX512(zmm10)

      // c[0, 48-63]
      RELU_SCALE_OP_F32S_AVX512(zmm11)

      // c[1, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm12)

      // c[1, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm13)

      // c[1, 32-47]
      RELU_SCALE_OP_F32S_AVX512(zmm14)

      // c[1, 48-63]
      RELU_SCALE_OP_F32S_AVX512(zmm15)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_2x64F:
    {
      __m512i zmm6;
      // c[0, 0-15]
      GELU_TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[0, 16-31]
      GELU_TANH_F32S_AVX512(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[0, 32-47]
      GELU_TANH_F32S_AVX512(zmm10, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[0, 48-63]
      GELU_TANH_F32S_AVX512(zmm11, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 0-15]
      GELU_TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 16-31]
      GELU_TANH_F32S_AVX512(zmm13, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 32-47]
      GELU_TANH_F32S_AVX512(zmm14, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 48-63]
      GELU_TANH_F32S_AVX512(zmm15, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_2x64F:
    {
      // c[0, 0-15]
      GELU_ERF_F32S_AVX512(zmm8, zmm0, zmm1, zmm2)

      // c[0, 16-31]
      GELU_ERF_F32S_AVX512(zmm9, zmm0, zmm1, zmm2)

      // c[0, 32-47]
      GELU_ERF_F32S_AVX512(zmm10, zmm0, zmm1, zmm2)

      // c[0, 48-63]
      GELU_ERF_F32S_AVX512(zmm11, zmm0, zmm1, zmm2)

      // c[1, 0-15]
      GELU_ERF_F32S_AVX512(zmm12, zmm0, zmm1, zmm2)

      // c[1, 16-31]
      GELU_ERF_F32S_AVX512(zmm13, zmm0, zmm1, zmm2)

      // c[1, 32-47]
      GELU_ERF_F32S_AVX512(zmm14, zmm0, zmm1, zmm2)

      // c[1, 48-63]
      GELU_ERF_F32S_AVX512(zmm15, zmm0, zmm1, zmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_2x64F:
    {
      zmm0 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      zmm1 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0, 0-15]
      CLIP_F32S_AVX512(zmm8, zmm0, zmm1)

      // c[0, 16-31]
      CLIP_F32S_AVX512(zmm9, zmm0, zmm1)

      // c[0, 32-47]
      CLIP_F32S_AVX512(zmm10, zmm0, zmm1)

      // c[0, 48-63]
      CLIP_F32S_AVX512(zmm11, zmm0, zmm1)

      // c[1, 0-15]
      CLIP_F32S_AVX512(zmm12, zmm0, zmm1)

      // c[1, 16-31]
      CLIP_F32S_AVX512(zmm13, zmm0, zmm1)

      // c[1, 32-47]
      CLIP_F32S_AVX512(zmm14, zmm0, zmm1)

      // c[1, 48-63]
      CLIP_F32S_AVX512(zmm15, zmm0, zmm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_2x64F_DISABLE:
    ;

    _mm512_storeu_ps(cbuf, zmm8); 
    _mm512_storeu_ps(cbuf + 16, zmm9);
    _mm512_storeu_ps(cbuf + 32, zmm10);
    _mm512_storeu_ps(cbuf + 48, zmm11);
    cbuf += rs_c;
    _mm512_storeu_ps(cbuf, zmm12);
    _mm512_storeu_ps(cbuf + 16, zmm13);
    _mm512_storeu_ps(cbuf + 32, zmm14);
    _mm512_storeu_ps(cbuf + 48, zmm15);
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_1x64)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_1x64F_DISABLE,
              &&POST_OPS_BIAS_1x64F,
              &&POST_OPS_RELU_1x64F,
              &&POST_OPS_RELU_SCALE_1x64F,
              &&POST_OPS_GELU_TANH_1x64F,
              &&POST_OPS_GELU_ERF_1x64F,
              &&POST_OPS_CLIP_1x64F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k0;

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7;
    __m512 zmm8, zmm9, zmm10, zmm11;
    
    /* zero the accumulator registers */
    ZERO_ACC_ZMM_4_REG(zmm8, zmm9, zmm10, zmm11);

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

        zmm8 = _mm512_fmadd_ps(zmm0, zmm2, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm2, zmm9);
        zmm10 = _mm512_fmadd_ps(zmm6, zmm2, zmm10);
        zmm11 = _mm512_fmadd_ps(zmm7, zmm2, zmm11);

        bbuf += rs_b;  //move b pointer to next row
        abuf += cs_a;  //move a pointer to next col

    }//kloop

    zmm0 = _mm512_set1_ps(alpha);

    ALPHA_MUL_ACC_ZMM_4_REG(zmm8,  zmm9,  zmm10, zmm11, zmm0);

    if ( beta != 0.0 )
    { 
        //load c and multiply with beta and 
        //add to accumulator and store back
        zmm3 = _mm512_set1_ps(beta);

        zmm0 = _mm512_load_ps(cbuf);
        zmm1 = _mm512_load_ps(cbuf + 16);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);

        zmm0 = _mm512_load_ps(cbuf + 32);
        zmm1 = _mm512_load_ps(cbuf + 48);
        zmm10 = _mm512_fmadd_ps(zmm0, zmm3, zmm10);
        zmm11 = _mm512_fmadd_ps(zmm1, zmm3, zmm11);
    }

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_1x64F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        zmm1 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        zmm2 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 1 * 16 ) );
        zmm3 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 2 * 16 ) );
        zmm4 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 3 * 16 ) );

        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[0, 16-31]
        zmm9 = _mm512_add_ps( zmm2, zmm9 );

        // c[0,32-47]
        zmm10 = _mm512_add_ps( zmm3, zmm10 );

        // c[0,48-63]
        zmm11 = _mm512_add_ps( zmm4, zmm11 );
      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the bias array will be accessed by
        // the ic index, and each bias element corresponds to an
        // entire row of the transposed output array, instead of an
        // entire column.
        zmm1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 0 ) );

        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[0, 16-31]
        zmm9 = _mm512_add_ps( zmm1, zmm9 );

        // c[0,32-47]
        zmm10 = _mm512_add_ps( zmm1, zmm10 );

        // c[0,48-63]
        zmm11 = _mm512_add_ps( zmm1, zmm11 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_1x64F:
    {
      zmm1 = _mm512_setzero_ps();

      // c[0,0-15]
      zmm8 = _mm512_max_ps( zmm1, zmm8 );

      // c[0, 16-31]
      zmm9 = _mm512_max_ps( zmm1, zmm9 );

      // c[0,32-47]
      zmm10 = _mm512_max_ps( zmm1, zmm10 );

      // c[0,48-63]
      zmm11 = _mm512_max_ps( zmm1, zmm11 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_1x64F:
    {
      zmm1 = _mm512_setzero_ps();
      zmm2 =
        _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

      __mmask16 relu_cmp_mask;

      // c[0, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm8)

      // c[0, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm9)

      // c[0, 32-47]
      RELU_SCALE_OP_F32S_AVX512(zmm10)

      // c[0, 48-63]
      RELU_SCALE_OP_F32S_AVX512(zmm11)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_1x64F:
    {
      __m512i zmm6;
      // c[0, 0-15]
      GELU_TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[0, 16-31]
      GELU_TANH_F32S_AVX512(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[0, 32-47]
      GELU_TANH_F32S_AVX512(zmm10, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[0, 48-63]
      GELU_TANH_F32S_AVX512(zmm11, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_1x64F:
    {
      // c[0, 0-15]
      GELU_ERF_F32S_AVX512(zmm8, zmm0, zmm1, zmm2)

      // c[0, 16-31]
      GELU_ERF_F32S_AVX512(zmm9, zmm0, zmm1, zmm2)

      // c[0, 32-47]
      GELU_ERF_F32S_AVX512(zmm10, zmm0, zmm1, zmm2)

      // c[0, 48-63]
      GELU_ERF_F32S_AVX512(zmm11, zmm0, zmm1, zmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_1x64F:
    {
      zmm0 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      zmm1 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0, 0-15]
      CLIP_F32S_AVX512(zmm8, zmm0, zmm1)

      // c[0, 16-31]
      CLIP_F32S_AVX512(zmm9, zmm0, zmm1)

      // c[0, 32-47]
      CLIP_F32S_AVX512(zmm10, zmm0, zmm1)

      // c[0, 48-63]
      CLIP_F32S_AVX512(zmm11, zmm0, zmm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_1x64F_DISABLE:
    ;

    _mm512_storeu_ps(cbuf, zmm8); 
    _mm512_storeu_ps(cbuf + 16, zmm9);
    _mm512_storeu_ps(cbuf + 32, zmm10);
    _mm512_storeu_ps(cbuf + 48, zmm11);
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_5x48)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_5x48F_DISABLE,
              &&POST_OPS_BIAS_5x48F,
              &&POST_OPS_RELU_5x48F,
              &&POST_OPS_RELU_SCALE_5x48F,
              &&POST_OPS_GELU_TANH_5x48F,
              &&POST_OPS_GELU_ERF_5x48F,
              &&POST_OPS_CLIP_5x48F
            };
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
    float *_cbuf = NULL;

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

    if ( beta != 0.0 )
    {
      _cbuf = cbuf;
      //load c and multiply with beta and 
      //add to accumulator and store back
      zmm3 = _mm512_set1_ps(beta);

      zmm0 = _mm512_load_ps(_cbuf);
      zmm1 = _mm512_load_ps(_cbuf + 16);
      zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
      zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);

      zmm0 = _mm512_load_ps(_cbuf + 32);
      zmm10 = _mm512_fmadd_ps(zmm0, zmm3, zmm10);
      _cbuf += rs_c;

      zmm0 = _mm512_load_ps(_cbuf);
      zmm1 = _mm512_load_ps(_cbuf + 16);
      zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
      zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);

      zmm0 = _mm512_load_ps(_cbuf + 32);
      zmm14 = _mm512_fmadd_ps(zmm0, zmm3, zmm14);
      _cbuf += rs_c;

      zmm0 = _mm512_load_ps(_cbuf);
      zmm1 = _mm512_load_ps(_cbuf+16);
      zmm16 = _mm512_fmadd_ps(zmm0, zmm3, zmm16);
      zmm17 = _mm512_fmadd_ps(zmm1, zmm3, zmm17);

      zmm0 = _mm512_load_ps(_cbuf + 32);
      zmm18 = _mm512_fmadd_ps(zmm0, zmm3, zmm18);
      _cbuf += rs_c;

      zmm0 = _mm512_load_ps(_cbuf);
      zmm1 = _mm512_load_ps(_cbuf+16);
      zmm20 = _mm512_fmadd_ps(zmm0, zmm3, zmm20);
      zmm21 = _mm512_fmadd_ps(zmm1, zmm3, zmm21);

      zmm0 = _mm512_load_ps(_cbuf + 32);
      zmm22 = _mm512_fmadd_ps(zmm0, zmm3, zmm22);
      _cbuf += rs_c;

      zmm0 = _mm512_load_ps(_cbuf);
      zmm1 = _mm512_load_ps(_cbuf+16);
      zmm24 = _mm512_fmadd_ps(zmm0, zmm3, zmm24);
      zmm25 = _mm512_fmadd_ps(zmm1, zmm3, zmm25);

      zmm0 = _mm512_load_ps(_cbuf + 32);
      zmm26 = _mm512_fmadd_ps(zmm0, zmm3, zmm26);
    }

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_5x48F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        zmm1 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        zmm2 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 1 * 16 ) );
        zmm3 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 2 * 16 ) );

        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[0, 16-31]
        zmm9 = _mm512_add_ps( zmm2, zmm9 );

        // c[0,32-47]
        zmm10 = _mm512_add_ps( zmm3, zmm10 );

        // c[1,0-15]
        zmm12 = _mm512_add_ps( zmm1, zmm12 );

        // c[1, 16-31]
        zmm13 = _mm512_add_ps( zmm2, zmm13 );

        // c[1,32-47]
        zmm14 = _mm512_add_ps( zmm3, zmm14 );

        // c[2,0-15]
        zmm16 = _mm512_add_ps( zmm1, zmm16 );

        // c[2, 16-31]
        zmm17 = _mm512_add_ps( zmm2, zmm17 );

        // c[2,32-47]
        zmm18 = _mm512_add_ps( zmm3, zmm18 );

        // c[3,0-15]
        zmm20 = _mm512_add_ps( zmm1, zmm20 );

        // c[3, 16-31]
        zmm21 = _mm512_add_ps( zmm2, zmm21 );

        // c[3,32-47]
        zmm22 = _mm512_add_ps( zmm3, zmm22 );

        // c[4,0-15]
        zmm24 = _mm512_add_ps( zmm1, zmm24 );

        // c[4, 16-31]
        zmm25 = _mm512_add_ps( zmm2, zmm25 );

        // c[4,32-47]
        zmm26 = _mm512_add_ps( zmm3, zmm26 );
      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the bias array will be accessed by
        // the ic index, and each bias element corresponds to an
        // entire row of the transposed output array, instead of an
        // entire column.
        zmm1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 0 ) );
        zmm2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 1 ) );
        zmm3 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 2 ) );
        zmm4 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 3 ) );
        zmm5 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 4 ) );

        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[0, 16-31]
        zmm9 = _mm512_add_ps( zmm1, zmm9 );

        // c[0,32-47]
        zmm10 = _mm512_add_ps( zmm1, zmm10 );

        // c[1,0-15]
        zmm12 = _mm512_add_ps( zmm2, zmm12 );

        // c[1, 16-31]
        zmm13 = _mm512_add_ps( zmm2, zmm13 );

        // c[1,32-47]
        zmm14 = _mm512_add_ps( zmm2, zmm14 );

        // c[2,0-15]
        zmm16 = _mm512_add_ps( zmm3, zmm16 );

        // c[2, 16-31]
        zmm17 = _mm512_add_ps( zmm3, zmm17 );

        // c[2,32-47]
        zmm18 = _mm512_add_ps( zmm3, zmm18 );

        // c[3,0-15]
        zmm20 = _mm512_add_ps( zmm4, zmm20 );

        // c[3, 16-31]
        zmm21 = _mm512_add_ps( zmm4, zmm21 );

        // c[3,32-47]
        zmm22 = _mm512_add_ps( zmm4, zmm22 );

        // c[4,0-15]
        zmm24 = _mm512_add_ps( zmm5, zmm24 );

        // c[4, 16-31]
        zmm25 = _mm512_add_ps( zmm5, zmm25 );

        // c[4,32-47]
        zmm26 = _mm512_add_ps( zmm5, zmm26 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_5x48F:
    {
      zmm1 = _mm512_setzero_ps();

      // c[0,0-15]
      zmm8 = _mm512_max_ps( zmm1, zmm8 );

      // c[0, 16-31]
      zmm9 = _mm512_max_ps( zmm1, zmm9 );

      // c[0,32-47]
      zmm10 = _mm512_max_ps( zmm1, zmm10 );

      // c[1,0-15]
      zmm12 = _mm512_max_ps( zmm1, zmm12 );

      // c[1,16-31]
      zmm13 = _mm512_max_ps( zmm1, zmm13 );

      // c[1,32-47]
      zmm14 = _mm512_max_ps( zmm1, zmm14 );

      // c[2,0-15]
      zmm16 = _mm512_max_ps( zmm1, zmm16 );

      // c[2,16-31]
      zmm17 = _mm512_max_ps( zmm1, zmm17 );

      // c[2,32-47]
      zmm18 = _mm512_max_ps( zmm1, zmm18 );

      // c[3,0-15]
      zmm20 = _mm512_max_ps( zmm1, zmm20 );

      // c[3,16-31]
      zmm21 = _mm512_max_ps( zmm1, zmm21 );

      // c[3,32-47]
      zmm22 = _mm512_max_ps( zmm1, zmm22 );

      // c[4,0-15]
      zmm24 = _mm512_max_ps( zmm1, zmm24 );

      // c[4,16-31]
      zmm25 = _mm512_max_ps( zmm1, zmm25 );

      // c[4,32-47]
      zmm26 = _mm512_max_ps( zmm1, zmm26 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_5x48F:
    {
      zmm1 = _mm512_setzero_ps();
      zmm2 =
        _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

      __mmask16 relu_cmp_mask;

      // c[0, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm8)

      // c[0, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm9)

      // c[0, 32-47]
      RELU_SCALE_OP_F32S_AVX512(zmm10)

      // c[1, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm12)

      // c[1, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm13)

      // c[1, 32-47]
      RELU_SCALE_OP_F32S_AVX512(zmm14)

      // c[2, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm16)

      // c[2, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm17)

      // c[2, 32-47]
      RELU_SCALE_OP_F32S_AVX512(zmm18)

      // c[3, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm20)

      // c[3, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm21)

      // c[3, 32-47]
      RELU_SCALE_OP_F32S_AVX512(zmm22)

      // c[4, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm24)

      // c[4, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm25)

      // c[4, 32-47]
      RELU_SCALE_OP_F32S_AVX512(zmm26)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_5x48F:
    {
      __m512i zmm6;
      // c[0, 0-15]
      GELU_TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[0, 16-31]
      GELU_TANH_F32S_AVX512(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[0, 32-47]
      GELU_TANH_F32S_AVX512(zmm10, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 0-15]
      GELU_TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 16-31]
      GELU_TANH_F32S_AVX512(zmm13, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 32-47]
      GELU_TANH_F32S_AVX512(zmm14, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[2, 0-15]
      GELU_TANH_F32S_AVX512(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[2, 16-31]
      GELU_TANH_F32S_AVX512(zmm17, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[2, 32-47]
      GELU_TANH_F32S_AVX512(zmm18, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[3, 0-15]
      GELU_TANH_F32S_AVX512(zmm20, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[3, 16-31]
      GELU_TANH_F32S_AVX512(zmm21, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[3, 32-47]
      GELU_TANH_F32S_AVX512(zmm22, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[4, 0-15]
      GELU_TANH_F32S_AVX512(zmm24, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[4, 16-31]
      GELU_TANH_F32S_AVX512(zmm25, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[4, 32-47]
      GELU_TANH_F32S_AVX512(zmm26, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_5x48F:
    {
      // c[0, 0-15]
      GELU_ERF_F32S_AVX512(zmm8, zmm0, zmm1, zmm2)

      // c[0, 16-31]
      GELU_ERF_F32S_AVX512(zmm9, zmm0, zmm1, zmm2)

      // c[0, 32-47]
      GELU_ERF_F32S_AVX512(zmm10, zmm0, zmm1, zmm2)

      // c[1, 0-15]
      GELU_ERF_F32S_AVX512(zmm12, zmm0, zmm1, zmm2)

      // c[1, 16-31]
      GELU_ERF_F32S_AVX512(zmm13, zmm0, zmm1, zmm2)

      // c[1, 32-47]
      GELU_ERF_F32S_AVX512(zmm14, zmm0, zmm1, zmm2)

      // c[2, 0-15]
      GELU_ERF_F32S_AVX512(zmm16, zmm0, zmm1, zmm2)

      // c[2, 16-31]
      GELU_ERF_F32S_AVX512(zmm17, zmm0, zmm1, zmm2)

      // c[2, 32-47]
      GELU_ERF_F32S_AVX512(zmm18, zmm0, zmm1, zmm2)

      // c[3, 0-15]
      GELU_ERF_F32S_AVX512(zmm20, zmm0, zmm1, zmm2)

      // c[3, 16-31]
      GELU_ERF_F32S_AVX512(zmm21, zmm0, zmm1, zmm2)

      // c[3, 32-47]
      GELU_ERF_F32S_AVX512(zmm22, zmm0, zmm1, zmm2)

      // c[4, 0-15]
      GELU_ERF_F32S_AVX512(zmm24, zmm0, zmm1, zmm2)

      // c[4, 16-31]
      GELU_ERF_F32S_AVX512(zmm25, zmm0, zmm1, zmm2)

      // c[4, 32-47]
      GELU_ERF_F32S_AVX512(zmm26, zmm0, zmm1, zmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_5x48F:
    {
      zmm0 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      zmm1 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0, 0-15]
      CLIP_F32S_AVX512(zmm8, zmm0, zmm1)

      // c[0, 16-31]
      CLIP_F32S_AVX512(zmm9, zmm0, zmm1)

      // c[0, 32-47]
      CLIP_F32S_AVX512(zmm10, zmm0, zmm1)

      // c[1, 0-15]
      CLIP_F32S_AVX512(zmm12, zmm0, zmm1)

      // c[1, 16-31]
      CLIP_F32S_AVX512(zmm13, zmm0, zmm1)

      // c[1, 32-47]
      CLIP_F32S_AVX512(zmm14, zmm0, zmm1)

      // c[2, 0-15]
      CLIP_F32S_AVX512(zmm16, zmm0, zmm1)

      // c[2, 16-31]
      CLIP_F32S_AVX512(zmm17, zmm0, zmm1)

      // c[2, 32-47]
      CLIP_F32S_AVX512(zmm18, zmm0, zmm1)

      // c[3, 0-15]
      CLIP_F32S_AVX512(zmm20, zmm0, zmm1)

      // c[3, 16-31]
      CLIP_F32S_AVX512(zmm21, zmm0, zmm1)

      // c[3, 32-47]
      CLIP_F32S_AVX512(zmm22, zmm0, zmm1)

      // c[4, 0-15]
      CLIP_F32S_AVX512(zmm24, zmm0, zmm1)

      // c[4, 16-31]
      CLIP_F32S_AVX512(zmm25, zmm0, zmm1)

      // c[4, 32-47]
      CLIP_F32S_AVX512(zmm26, zmm0, zmm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_5x48F_DISABLE:
    ;

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
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_4x48)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_4x48F_DISABLE,
              &&POST_OPS_BIAS_4x48F,
              &&POST_OPS_RELU_4x48F,
              &&POST_OPS_RELU_SCALE_4x48F,
              &&POST_OPS_GELU_TANH_4x48F,
              &&POST_OPS_GELU_ERF_4x48F,
              &&POST_OPS_CLIP_4x48F
            };
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
    float *_cbuf = NULL;

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

    if ( beta != 0.0 )
    {
      _cbuf = cbuf;
      //load c and multiply with beta and 
      //add to accumulator and store back
      zmm3 = _mm512_set1_ps(beta);

      zmm0 = _mm512_load_ps(_cbuf);
      zmm1 = _mm512_load_ps(_cbuf + 16);
      zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
      zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);

      zmm0 = _mm512_load_ps(_cbuf + 32);
      zmm10 = _mm512_fmadd_ps(zmm0, zmm3, zmm10);
      _cbuf += rs_c;

      zmm0 = _mm512_load_ps(_cbuf);
      zmm1 = _mm512_load_ps(_cbuf + 16);
      zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
      zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);

      zmm0 = _mm512_load_ps(_cbuf + 32);
      zmm14 = _mm512_fmadd_ps(zmm0, zmm3, zmm14);
      _cbuf += rs_c;

      zmm0 = _mm512_load_ps(_cbuf);
      zmm1 = _mm512_load_ps(_cbuf+16);
      zmm16 = _mm512_fmadd_ps(zmm0, zmm3, zmm16);
      zmm17 = _mm512_fmadd_ps(zmm1, zmm3, zmm17);

      zmm0 = _mm512_load_ps(_cbuf + 32);
      zmm18 = _mm512_fmadd_ps(zmm0, zmm3, zmm18);
      _cbuf += rs_c;

      zmm0 = _mm512_load_ps(_cbuf);
      zmm1 = _mm512_load_ps(_cbuf+16);
      zmm20 = _mm512_fmadd_ps(zmm0, zmm3, zmm20);
      zmm21 = _mm512_fmadd_ps(zmm1, zmm3, zmm21);

      zmm0 = _mm512_load_ps(_cbuf + 32);
      zmm22 = _mm512_fmadd_ps(zmm0, zmm3, zmm22);
    }

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_4x48F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        zmm1 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        zmm2 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 1 * 16 ) );
        zmm3 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 2 * 16 ) );

        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[0, 16-31]
        zmm9 = _mm512_add_ps( zmm2, zmm9 );

        // c[0,32-47]
        zmm10 = _mm512_add_ps( zmm3, zmm10 );

        // c[1,0-15]
        zmm12 = _mm512_add_ps( zmm1, zmm12 );

        // c[1, 16-31]
        zmm13 = _mm512_add_ps( zmm2, zmm13 );

        // c[1,32-47]
        zmm14 = _mm512_add_ps( zmm3, zmm14 );

        // c[2,0-15]
        zmm16 = _mm512_add_ps( zmm1, zmm16 );

        // c[2, 16-31]
        zmm17 = _mm512_add_ps( zmm2, zmm17 );

        // c[2,32-47]
        zmm18 = _mm512_add_ps( zmm3, zmm18 );

        // c[3,0-15]
        zmm20 = _mm512_add_ps( zmm1, zmm20 );

        // c[3, 16-31]
        zmm21 = _mm512_add_ps( zmm2, zmm21 );

        // c[3,32-47]
        zmm22 = _mm512_add_ps( zmm3, zmm22 );
      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the bias array will be accessed by
        // the ic index, and each bias element corresponds to an
        // entire row of the transposed output array, instead of an
        // entire column.
        zmm1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 0 ) );
        zmm2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 1 ) );
        zmm3 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 2 ) );
        zmm4 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 3 ) );

        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[0, 16-31]
        zmm9 = _mm512_add_ps( zmm1, zmm9 );

        // c[0,32-47]
        zmm10 = _mm512_add_ps( zmm1, zmm10 );

        // c[1,0-15]
        zmm12 = _mm512_add_ps( zmm2, zmm12 );

        // c[1, 16-31]
        zmm13 = _mm512_add_ps( zmm2, zmm13 );

        // c[1,32-47]
        zmm14 = _mm512_add_ps( zmm2, zmm14 );

        // c[2,0-15]
        zmm16 = _mm512_add_ps( zmm3, zmm16 );

        // c[2, 16-31]
        zmm17 = _mm512_add_ps( zmm3, zmm17 );

        // c[2,32-47]
        zmm18 = _mm512_add_ps( zmm3, zmm18 );

        // c[3,0-15]
        zmm20 = _mm512_add_ps( zmm4, zmm20 );

        // c[3, 16-31]
        zmm21 = _mm512_add_ps( zmm4, zmm21 );

        // c[3,32-47]
        zmm22 = _mm512_add_ps( zmm4, zmm22 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_4x48F:
    {
      zmm1 = _mm512_setzero_ps();

      // c[0,0-15]
      zmm8 = _mm512_max_ps( zmm1, zmm8 );

      // c[0, 16-31]
      zmm9 = _mm512_max_ps( zmm1, zmm9 );

      // c[0,32-47]
      zmm10 = _mm512_max_ps( zmm1, zmm10 );

      // c[1,0-15]
      zmm12 = _mm512_max_ps( zmm1, zmm12 );

      // c[1,16-31]
      zmm13 = _mm512_max_ps( zmm1, zmm13 );

      // c[1,32-47]
      zmm14 = _mm512_max_ps( zmm1, zmm14 );

      // c[2,0-15]
      zmm16 = _mm512_max_ps( zmm1, zmm16 );

      // c[2,16-31]
      zmm17 = _mm512_max_ps( zmm1, zmm17 );

      // c[2,32-47]
      zmm18 = _mm512_max_ps( zmm1, zmm18 );

      // c[3,0-15]
      zmm20 = _mm512_max_ps( zmm1, zmm20 );

      // c[3,16-31]
      zmm21 = _mm512_max_ps( zmm1, zmm21 );

      // c[3,32-47]
      zmm22 = _mm512_max_ps( zmm1, zmm22 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_4x48F:
    {
      zmm1 = _mm512_setzero_ps();
      zmm2 =
        _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

      __mmask16 relu_cmp_mask;

      // c[0, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm8)

      // c[0, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm9)

      // c[0, 32-47]
      RELU_SCALE_OP_F32S_AVX512(zmm10)

      // c[1, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm12)

      // c[1, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm13)

      // c[1, 32-47]
      RELU_SCALE_OP_F32S_AVX512(zmm14)

      // c[2, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm16)

      // c[2, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm17)

      // c[2, 32-47]
      RELU_SCALE_OP_F32S_AVX512(zmm18)

      // c[3, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm20)

      // c[3, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm21)

      // c[3, 32-47]
      RELU_SCALE_OP_F32S_AVX512(zmm22)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_4x48F:
    {
      __m512i zmm6;
      // c[0, 0-15]
      GELU_TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[0, 16-31]
      GELU_TANH_F32S_AVX512(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[0, 32-47]
      GELU_TANH_F32S_AVX512(zmm10, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 0-15]
      GELU_TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 16-31]
      GELU_TANH_F32S_AVX512(zmm13, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 32-47]
      GELU_TANH_F32S_AVX512(zmm14, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[2, 0-15]
      GELU_TANH_F32S_AVX512(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[2, 16-31]
      GELU_TANH_F32S_AVX512(zmm17, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[2, 32-47]
      GELU_TANH_F32S_AVX512(zmm18, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[3, 0-15]
      GELU_TANH_F32S_AVX512(zmm20, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[3, 16-31]
      GELU_TANH_F32S_AVX512(zmm21, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[3, 32-47]
      GELU_TANH_F32S_AVX512(zmm22, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_4x48F:
    {
      // c[0, 0-15]
      GELU_ERF_F32S_AVX512(zmm8, zmm0, zmm1, zmm2)

      // c[0, 16-31]
      GELU_ERF_F32S_AVX512(zmm9, zmm0, zmm1, zmm2)

      // c[0, 32-47]
      GELU_ERF_F32S_AVX512(zmm10, zmm0, zmm1, zmm2)

      // c[1, 0-15]
      GELU_ERF_F32S_AVX512(zmm12, zmm0, zmm1, zmm2)

      // c[1, 16-31]
      GELU_ERF_F32S_AVX512(zmm13, zmm0, zmm1, zmm2)

      // c[1, 32-47]
      GELU_ERF_F32S_AVX512(zmm14, zmm0, zmm1, zmm2)

      // c[2, 0-15]
      GELU_ERF_F32S_AVX512(zmm16, zmm0, zmm1, zmm2)

      // c[2, 16-31]
      GELU_ERF_F32S_AVX512(zmm17, zmm0, zmm1, zmm2)

      // c[2, 32-47]
      GELU_ERF_F32S_AVX512(zmm18, zmm0, zmm1, zmm2)

      // c[3, 0-15]
      GELU_ERF_F32S_AVX512(zmm20, zmm0, zmm1, zmm2)

      // c[3, 16-31]
      GELU_ERF_F32S_AVX512(zmm21, zmm0, zmm1, zmm2)

      // c[3, 32-47]
      GELU_ERF_F32S_AVX512(zmm22, zmm0, zmm1, zmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_4x48F:
    {
      zmm0 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      zmm1 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0, 0-15]
      CLIP_F32S_AVX512(zmm8, zmm0, zmm1)

      // c[0, 16-31]
      CLIP_F32S_AVX512(zmm9, zmm0, zmm1)

      // c[0, 32-47]
      CLIP_F32S_AVX512(zmm10, zmm0, zmm1)

      // c[1, 0-15]
      CLIP_F32S_AVX512(zmm12, zmm0, zmm1)

      // c[1, 16-31]
      CLIP_F32S_AVX512(zmm13, zmm0, zmm1)

      // c[1, 32-47]
      CLIP_F32S_AVX512(zmm14, zmm0, zmm1)

      // c[2, 0-15]
      CLIP_F32S_AVX512(zmm16, zmm0, zmm1)

      // c[2, 16-31]
      CLIP_F32S_AVX512(zmm17, zmm0, zmm1)

      // c[2, 32-47]
      CLIP_F32S_AVX512(zmm18, zmm0, zmm1)

      // c[3, 0-15]
      CLIP_F32S_AVX512(zmm20, zmm0, zmm1)

      // c[3, 16-31]
      CLIP_F32S_AVX512(zmm21, zmm0, zmm1)

      // c[3, 32-47]
      CLIP_F32S_AVX512(zmm22, zmm0, zmm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_4x48F_DISABLE:
    ;

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
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_3x48)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_3x48F_DISABLE,
              &&POST_OPS_BIAS_3x48F,
              &&POST_OPS_RELU_3x48F,
              &&POST_OPS_RELU_SCALE_3x48F,
              &&POST_OPS_GELU_TANH_3x48F,
              &&POST_OPS_GELU_ERF_3x48F,
              &&POST_OPS_CLIP_3x48F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k0;

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6;
    __m512 zmm8, zmm9, zmm10, zmm12, zmm13, zmm14;
    __m512 zmm16, zmm17, zmm18;

    /* zero the accumulator registers */
    ZERO_ACC_ZMM_4_REG(zmm8, zmm9, zmm10, zmm12);
    ZERO_ACC_ZMM_4_REG(zmm13, zmm14,zmm16, zmm17);
    zmm18 = _mm512_setzero_ps();

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;
    float *_cbuf = NULL;

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

    if ( beta != 0.0 )
    {
      _cbuf = cbuf;
      //load c and multiply with beta and 
      //add to accumulator and store back
      zmm3 = _mm512_set1_ps(beta);

      zmm0 = _mm512_load_ps(_cbuf);
      zmm1 = _mm512_load_ps(_cbuf + 16);
      zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
      zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);

      zmm0 = _mm512_load_ps(_cbuf + 32);
      zmm10 = _mm512_fmadd_ps(zmm0, zmm3, zmm10);
      _cbuf += rs_c;

      zmm0 = _mm512_load_ps(_cbuf);
      zmm1 = _mm512_load_ps(_cbuf + 16);
      zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
      zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);

      zmm0 = _mm512_load_ps(_cbuf + 32);
      zmm14 = _mm512_fmadd_ps(zmm0, zmm3, zmm14);
      _cbuf += rs_c;

      zmm0 = _mm512_load_ps(_cbuf);
      zmm1 = _mm512_load_ps(_cbuf+16);
      zmm16 = _mm512_fmadd_ps(zmm0, zmm3, zmm16);
      zmm17 = _mm512_fmadd_ps(zmm1, zmm3, zmm17);

      zmm0 = _mm512_load_ps(_cbuf + 32);
      zmm18 = _mm512_fmadd_ps(zmm0, zmm3, zmm18);
    }

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_3x48F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        zmm1 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        zmm2 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 1 * 16 ) );
        zmm3 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 2 * 16 ) );

        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[0, 16-31]
        zmm9 = _mm512_add_ps( zmm2, zmm9 );

        // c[0,32-47]
        zmm10 = _mm512_add_ps( zmm3, zmm10 );

        // c[1,0-15]
        zmm12 = _mm512_add_ps( zmm1, zmm12 );

        // c[1, 16-31]
        zmm13 = _mm512_add_ps( zmm2, zmm13 );

        // c[1,32-47]
        zmm14 = _mm512_add_ps( zmm3, zmm14 );

        // c[2,0-15]
        zmm16 = _mm512_add_ps( zmm1, zmm16 );

        // c[2, 16-31]
        zmm17 = _mm512_add_ps( zmm2, zmm17 );

        // c[2,32-47]
        zmm18 = _mm512_add_ps( zmm3, zmm18 );
      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the bias array will be accessed by
        // the ic index, and each bias element corresponds to an
        // entire row of the transposed output array, instead of an
        // entire column.
        zmm1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 0 ) );
        zmm2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 1 ) );
        zmm3 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 2 ) );

        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[0, 16-31]
        zmm9 = _mm512_add_ps( zmm1, zmm9 );

        // c[0,32-47]
        zmm10 = _mm512_add_ps( zmm1, zmm10 );

        // c[1,0-15]
        zmm12 = _mm512_add_ps( zmm2, zmm12 );

        // c[1, 16-31]
        zmm13 = _mm512_add_ps( zmm2, zmm13 );

        // c[1,32-47]
        zmm14 = _mm512_add_ps( zmm2, zmm14 );

        // c[2,0-15]
        zmm16 = _mm512_add_ps( zmm3, zmm16 );

        // c[2, 16-31]
        zmm17 = _mm512_add_ps( zmm3, zmm17 );

        // c[2,32-47]
        zmm18 = _mm512_add_ps( zmm3, zmm18 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_3x48F:
    {
      zmm1 = _mm512_setzero_ps();

      // c[0,0-15]
      zmm8 = _mm512_max_ps( zmm1, zmm8 );

      // c[0, 16-31]
      zmm9 = _mm512_max_ps( zmm1, zmm9 );

      // c[0,32-47]
      zmm10 = _mm512_max_ps( zmm1, zmm10 );

      // c[1,0-15]
      zmm12 = _mm512_max_ps( zmm1, zmm12 );

      // c[1,16-31]
      zmm13 = _mm512_max_ps( zmm1, zmm13 );

      // c[1,32-47]
      zmm14 = _mm512_max_ps( zmm1, zmm14 );

      // c[2,0-15]
      zmm16 = _mm512_max_ps( zmm1, zmm16 );

      // c[2,16-31]
      zmm17 = _mm512_max_ps( zmm1, zmm17 );

      // c[2,32-47]
      zmm18 = _mm512_max_ps( zmm1, zmm18 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_3x48F:
    {
      zmm1 = _mm512_setzero_ps();
      zmm2 =
        _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

      __mmask16 relu_cmp_mask;

      // c[0, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm8)

      // c[0, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm9)

      // c[0, 32-47]
      RELU_SCALE_OP_F32S_AVX512(zmm10)

      // c[1, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm12)

      // c[1, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm13)

      // c[1, 32-47]
      RELU_SCALE_OP_F32S_AVX512(zmm14)

      // c[2, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm16)

      // c[2, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm17)

      // c[2, 32-47]
      RELU_SCALE_OP_F32S_AVX512(zmm18)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_3x48F:
    {
      __m512i zmm6;
      // c[0, 0-15]
      GELU_TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[0, 16-31]
      GELU_TANH_F32S_AVX512(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[0, 32-47]
      GELU_TANH_F32S_AVX512(zmm10, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 0-15]
      GELU_TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 16-31]
      GELU_TANH_F32S_AVX512(zmm13, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 32-47]
      GELU_TANH_F32S_AVX512(zmm14, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[2, 0-15]
      GELU_TANH_F32S_AVX512(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[2, 16-31]
      GELU_TANH_F32S_AVX512(zmm17, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[2, 32-47]
      GELU_TANH_F32S_AVX512(zmm18, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_3x48F:
    {
      // c[0, 0-15]
      GELU_ERF_F32S_AVX512(zmm8, zmm0, zmm1, zmm2)

      // c[0, 16-31]
      GELU_ERF_F32S_AVX512(zmm9, zmm0, zmm1, zmm2)

      // c[0, 32-47]
      GELU_ERF_F32S_AVX512(zmm10, zmm0, zmm1, zmm2)

      // c[1, 0-15]
      GELU_ERF_F32S_AVX512(zmm12, zmm0, zmm1, zmm2)

      // c[1, 16-31]
      GELU_ERF_F32S_AVX512(zmm13, zmm0, zmm1, zmm2)

      // c[1, 32-47]
      GELU_ERF_F32S_AVX512(zmm14, zmm0, zmm1, zmm2)

      // c[2, 0-15]
      GELU_ERF_F32S_AVX512(zmm16, zmm0, zmm1, zmm2)

      // c[2, 16-31]
      GELU_ERF_F32S_AVX512(zmm17, zmm0, zmm1, zmm2)

      // c[2, 32-47]
      GELU_ERF_F32S_AVX512(zmm18, zmm0, zmm1, zmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_3x48F:
    {
      zmm0 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      zmm1 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0, 0-15]
      CLIP_F32S_AVX512(zmm8, zmm0, zmm1)

      // c[0, 16-31]
      CLIP_F32S_AVX512(zmm9, zmm0, zmm1)

      // c[0, 32-47]
      CLIP_F32S_AVX512(zmm10, zmm0, zmm1)

      // c[1, 0-15]
      CLIP_F32S_AVX512(zmm12, zmm0, zmm1)

      // c[1, 16-31]
      CLIP_F32S_AVX512(zmm13, zmm0, zmm1)

      // c[1, 32-47]
      CLIP_F32S_AVX512(zmm14, zmm0, zmm1)

      // c[2, 0-15]
      CLIP_F32S_AVX512(zmm16, zmm0, zmm1)

      // c[2, 16-31]
      CLIP_F32S_AVX512(zmm17, zmm0, zmm1)

      // c[2, 32-47]
      CLIP_F32S_AVX512(zmm18, zmm0, zmm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_3x48F_DISABLE:
    ;

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
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_2x48)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_2x48F_DISABLE,
              &&POST_OPS_BIAS_2x48F,
              &&POST_OPS_RELU_2x48F,
              &&POST_OPS_RELU_SCALE_2x48F,
              &&POST_OPS_GELU_TANH_2x48F,
              &&POST_OPS_GELU_ERF_2x48F,
              &&POST_OPS_CLIP_2x48F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k0;

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6;
    __m512 zmm8, zmm9, zmm10, zmm12, zmm13, zmm14, zmm16,zmm17;

    /* zero the accumulator registers */
    ZERO_ACC_ZMM_4_REG(zmm8, zmm9, zmm10, zmm12);
    ZERO_ACC_ZMM_4_REG(zmm13, zmm14,zmm16, zmm17);

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;
    float *_cbuf = NULL;

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

    if ( beta != 0.0 )
    {
      _cbuf = cbuf;
      //load c and multiply with beta and 
      //add to accumulator and store back
      zmm3 = _mm512_set1_ps(beta);

      zmm0 = _mm512_load_ps(_cbuf);
      zmm1 = _mm512_load_ps(_cbuf + 16);
      zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
      zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);

      zmm0 = _mm512_load_ps(_cbuf + 32);
      zmm10 = _mm512_fmadd_ps(zmm0, zmm3, zmm10);
      _cbuf += rs_c;

      zmm0 = _mm512_load_ps(_cbuf);
      zmm1 = _mm512_load_ps(_cbuf + 16);
      zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
      zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);

      zmm0 = _mm512_load_ps(_cbuf + 32);
      zmm14 = _mm512_fmadd_ps(zmm0, zmm3, zmm14);
    }

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_2x48F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        zmm1 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        zmm2 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 1 * 16 ) );
        zmm3 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 2 * 16 ) );

        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[0, 16-31]
        zmm9 = _mm512_add_ps( zmm2, zmm9 );

        // c[0,32-47]
        zmm10 = _mm512_add_ps( zmm3, zmm10 );

        // c[1,0-15]
        zmm12 = _mm512_add_ps( zmm1, zmm12 );

        // c[1, 16-31]
        zmm13 = _mm512_add_ps( zmm2, zmm13 );

        // c[1,32-47]
        zmm14 = _mm512_add_ps( zmm3, zmm14 );
      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the bias array will be accessed by
        // the ic index, and each bias element corresponds to an
        // entire row of the transposed output array, instead of an
        // entire column.
        zmm1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 0 ) );
        zmm2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 1 ) );

        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[0, 16-31]
        zmm9 = _mm512_add_ps( zmm1, zmm9 );

        // c[0,32-47]
        zmm10 = _mm512_add_ps( zmm1, zmm10 );

        // c[1,0-15]
        zmm12 = _mm512_add_ps( zmm2, zmm12 );

        // c[1, 16-31]
        zmm13 = _mm512_add_ps( zmm2, zmm13 );

        // c[1,32-47]
        zmm14 = _mm512_add_ps( zmm2, zmm14 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_2x48F:
    {
      zmm1 = _mm512_setzero_ps();

      // c[0,0-15]
      zmm8 = _mm512_max_ps( zmm1, zmm8 );

      // c[0, 16-31]
      zmm9 = _mm512_max_ps( zmm1, zmm9 );

      // c[0,32-47]
      zmm10 = _mm512_max_ps( zmm1, zmm10 );

      // c[1,0-15]
      zmm12 = _mm512_max_ps( zmm1, zmm12 );

      // c[1,16-31]
      zmm13 = _mm512_max_ps( zmm1, zmm13 );

      // c[1,32-47]
      zmm14 = _mm512_max_ps( zmm1, zmm14 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_2x48F:
    {
      zmm1 = _mm512_setzero_ps();
      zmm2 =
        _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

      __mmask16 relu_cmp_mask;

      // c[0, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm8)

      // c[0, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm9)

      // c[0, 32-47]
      RELU_SCALE_OP_F32S_AVX512(zmm10)

      // c[1, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm12)

      // c[1, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm13)

      // c[1, 32-47]
      RELU_SCALE_OP_F32S_AVX512(zmm14)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_2x48F:
    {
      __m512i zmm6;
      // c[0, 0-15]
      GELU_TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[0, 16-31]
      GELU_TANH_F32S_AVX512(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[0, 32-47]
      GELU_TANH_F32S_AVX512(zmm10, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 0-15]
      GELU_TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 16-31]
      GELU_TANH_F32S_AVX512(zmm13, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 32-47]
      GELU_TANH_F32S_AVX512(zmm14, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_2x48F:
    {
      // c[0, 0-15]
      GELU_ERF_F32S_AVX512(zmm8, zmm0, zmm1, zmm2)

      // c[0, 16-31]
      GELU_ERF_F32S_AVX512(zmm9, zmm0, zmm1, zmm2)

      // c[0, 32-47]
      GELU_ERF_F32S_AVX512(zmm10, zmm0, zmm1, zmm2)

      // c[1, 0-15]
      GELU_ERF_F32S_AVX512(zmm12, zmm0, zmm1, zmm2)

      // c[1, 16-31]
      GELU_ERF_F32S_AVX512(zmm13, zmm0, zmm1, zmm2)

      // c[1, 32-47]
      GELU_ERF_F32S_AVX512(zmm14, zmm0, zmm1, zmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_2x48F:
    {
      zmm0 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      zmm1 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0, 0-15]
      CLIP_F32S_AVX512(zmm8, zmm0, zmm1)

      // c[0, 16-31]
      CLIP_F32S_AVX512(zmm9, zmm0, zmm1)

      // c[0, 32-47]
      CLIP_F32S_AVX512(zmm10, zmm0, zmm1)

      // c[1, 0-15]
      CLIP_F32S_AVX512(zmm12, zmm0, zmm1)

      // c[1, 16-31]
      CLIP_F32S_AVX512(zmm13, zmm0, zmm1)

      // c[1, 32-47]
      CLIP_F32S_AVX512(zmm14, zmm0, zmm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_2x48F_DISABLE:
    ;

    _mm512_storeu_ps(cbuf, zmm8); 
    _mm512_storeu_ps(cbuf + 16, zmm9);
    _mm512_storeu_ps(cbuf + 32, zmm10);
    cbuf += rs_c;
    _mm512_storeu_ps(cbuf, zmm12);
    _mm512_storeu_ps(cbuf + 16, zmm13);
    _mm512_storeu_ps(cbuf + 32, zmm14);
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_1x48)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_1x48F_DISABLE,
              &&POST_OPS_BIAS_1x48F,
              &&POST_OPS_RELU_1x48F,
              &&POST_OPS_RELU_SCALE_1x48F,
              &&POST_OPS_GELU_TANH_1x48F,
              &&POST_OPS_GELU_ERF_1x48F,
              &&POST_OPS_CLIP_1x48F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k0;

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6;
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

    if ( beta != 0.0 )
    { 
      //load c and multiply with beta and 
      //add to accumulator and store back
      zmm3 = _mm512_set1_ps(beta);

      zmm0 = _mm512_load_ps(cbuf);
      zmm1 = _mm512_load_ps(cbuf + 16);
      zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
      zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);

      zmm0 = _mm512_load_ps(cbuf + 32);
      zmm10 = _mm512_fmadd_ps(zmm0, zmm3, zmm10);
    }

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_1x48F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        zmm1 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        zmm2 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 1 * 16 ) );
        zmm3 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 2 * 16 ) );

        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[0, 16-31]
        zmm9 = _mm512_add_ps( zmm2, zmm9 );

        // c[0,32-47]
        zmm10 = _mm512_add_ps( zmm3, zmm10 );
      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the bias array will be accessed by
        // the ic index, and each bias element corresponds to an
        // entire row of the transposed output array, instead of an
        // entire column.
        zmm1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 0 ) );

        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[0, 16-31]
        zmm9 = _mm512_add_ps( zmm1, zmm9 );

        // c[0,32-47]
        zmm10 = _mm512_add_ps( zmm1, zmm10 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_1x48F:
    {
      zmm1 = _mm512_setzero_ps();

      // c[0,0-15]
      zmm8 = _mm512_max_ps( zmm1, zmm8 );

      // c[0, 16-31]
      zmm9 = _mm512_max_ps( zmm1, zmm9 );

      // c[0,32-47]
      zmm10 = _mm512_max_ps( zmm1, zmm10 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_1x48F:
    {
      zmm1 = _mm512_setzero_ps();
      zmm2 =
        _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

      __mmask16 relu_cmp_mask;

      // c[0, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm8)

      // c[0, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm9)

      // c[0, 32-47]
      RELU_SCALE_OP_F32S_AVX512(zmm10)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_1x48F:
    {
      __m512i zmm6;
      // c[0, 0-15]
      GELU_TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[0, 16-31]
      GELU_TANH_F32S_AVX512(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[0, 32-47]
      GELU_TANH_F32S_AVX512(zmm10, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_1x48F:
    {
      // c[0, 0-15]
      GELU_ERF_F32S_AVX512(zmm8, zmm0, zmm1, zmm2)

      // c[0, 16-31]
      GELU_ERF_F32S_AVX512(zmm9, zmm0, zmm1, zmm2)

      // c[0, 32-47]
      GELU_ERF_F32S_AVX512(zmm10, zmm0, zmm1, zmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_1x48F:
    {
      zmm0 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      zmm1 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0, 0-15]
      CLIP_F32S_AVX512(zmm8, zmm0, zmm1)

      // c[0, 16-31]
      CLIP_F32S_AVX512(zmm9, zmm0, zmm1)

      // c[0, 32-47]
      CLIP_F32S_AVX512(zmm10, zmm0, zmm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_1x48F_DISABLE:
    ;

    _mm512_storeu_ps(cbuf, zmm8); 
    _mm512_storeu_ps(cbuf + 16, zmm9);
    _mm512_storeu_ps(cbuf + 32, zmm10);
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_5x32)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_5x32F_DISABLE,
              &&POST_OPS_BIAS_5x32F,
              &&POST_OPS_RELU_5x32F,
              &&POST_OPS_RELU_SCALE_5x32F,
              &&POST_OPS_GELU_TANH_5x32F,
              &&POST_OPS_GELU_ERF_5x32F,
              &&POST_OPS_CLIP_5x32F
            };
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
    float *_cbuf = NULL;

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

    if ( beta != 0.0 )
    {
        _cbuf = cbuf;
        //load c and multiply with beta and 
        //add to accumulator and store back
        zmm3 = _mm512_set1_ps(beta);

        zmm0 = _mm512_load_ps(_cbuf);
        zmm1 = _mm512_load_ps(_cbuf + 16);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);
        _cbuf += rs_c;

        zmm0 = _mm512_load_ps(_cbuf);
        zmm1 = _mm512_load_ps(_cbuf + 16);
        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);
        _cbuf += rs_c;

        zmm0 = _mm512_load_ps(_cbuf);
        zmm1 = _mm512_load_ps(_cbuf+16);
        zmm16 = _mm512_fmadd_ps(zmm0, zmm3, zmm16);
        zmm17 = _mm512_fmadd_ps(zmm1, zmm3, zmm17);
        _cbuf += rs_c;

        zmm0 = _mm512_load_ps(_cbuf);
        zmm1 = _mm512_load_ps(_cbuf+16);
        zmm20 = _mm512_fmadd_ps(zmm0, zmm3, zmm20);
        zmm21 = _mm512_fmadd_ps(zmm1, zmm3, zmm21);
        _cbuf += rs_c;

        zmm0 = _mm512_load_ps(_cbuf);
        zmm1 = _mm512_load_ps(_cbuf+16);
        zmm24 = _mm512_fmadd_ps(zmm0, zmm3, zmm24);
        zmm25 = _mm512_fmadd_ps(zmm1, zmm3, zmm25);
    }

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_5x32F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        zmm1 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        zmm2 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 1 * 16 ) );

        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[0, 16-31]
        zmm9 = _mm512_add_ps( zmm2, zmm9 );

        // c[1,0-15]
        zmm12 = _mm512_add_ps( zmm1, zmm12 );

        // c[1, 16-31]
        zmm13 = _mm512_add_ps( zmm2, zmm13 );

        // c[2,0-15]
        zmm16 = _mm512_add_ps( zmm1, zmm16 );

        // c[2, 16-31]
        zmm17 = _mm512_add_ps( zmm2, zmm17 );

        // c[3,0-15]
        zmm20 = _mm512_add_ps( zmm1, zmm20 );

        // c[3, 16-31]
        zmm21 = _mm512_add_ps( zmm2, zmm21 );

        // c[4,0-15]
        zmm24 = _mm512_add_ps( zmm1, zmm24 );

        // c[4, 16-31]
        zmm25 = _mm512_add_ps( zmm2, zmm25 );
      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the bias array will be accessed by
        // the ic index, and each bias element corresponds to an
        // entire row of the transposed output array, instead of an
        // entire column.
        zmm1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 0 ) );
        zmm2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 1 ) );
        zmm3 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 2 ) );
        zmm4 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 3 ) );
        zmm5 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 4 ) );

        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[0, 16-31]
        zmm9 = _mm512_add_ps( zmm1, zmm9 );

        // c[1,0-15]
        zmm12 = _mm512_add_ps( zmm2, zmm12 );

        // c[1, 16-31]
        zmm13 = _mm512_add_ps( zmm2, zmm13 );

        // c[2,0-15]
        zmm16 = _mm512_add_ps( zmm3, zmm16 );

        // c[2, 16-31]
        zmm17 = _mm512_add_ps( zmm3, zmm17 );

        // c[3,0-15]
        zmm20 = _mm512_add_ps( zmm4, zmm20 );

        // c[3, 16-31]
        zmm21 = _mm512_add_ps( zmm4, zmm21 );

        // c[4,0-15]
        zmm24 = _mm512_add_ps( zmm5, zmm24 );

        // c[4, 16-31]
        zmm25 = _mm512_add_ps( zmm5, zmm25 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_5x32F:
    {
      zmm1 = _mm512_setzero_ps();

      // c[0,0-15]
      zmm8 = _mm512_max_ps( zmm1, zmm8 );

      // c[0, 16-31]
      zmm9 = _mm512_max_ps( zmm1, zmm9 );

      // c[1,0-15]
      zmm12 = _mm512_max_ps( zmm1, zmm12 );

      // c[1,16-31]
      zmm13 = _mm512_max_ps( zmm1, zmm13 );

      // c[2,0-15]
      zmm16 = _mm512_max_ps( zmm1, zmm16 );

      // c[2,16-31]
      zmm17 = _mm512_max_ps( zmm1, zmm17 );

      // c[3,0-15]
      zmm20 = _mm512_max_ps( zmm1, zmm20 );

      // c[3,16-31]
      zmm21 = _mm512_max_ps( zmm1, zmm21 );

      // c[4,0-15]
      zmm24 = _mm512_max_ps( zmm1, zmm24 );

      // c[4,16-31]
      zmm25 = _mm512_max_ps( zmm1, zmm25 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_5x32F:
    {
      zmm1 = _mm512_setzero_ps();
      zmm2 =
        _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

      __mmask16 relu_cmp_mask;

      // c[0, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm8)

      // c[0, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm9)

      // c[1, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm12)

      // c[1, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm13)

      // c[2, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm16)

      // c[2, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm17)

      // c[3, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm20)

      // c[3, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm21)

      // c[4, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm24)

      // c[4, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm25)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_5x32F:
    {
      __m512i zmm6;
      // c[0, 0-15]
      GELU_TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[0, 16-31]
      GELU_TANH_F32S_AVX512(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 0-15]
      GELU_TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 16-31]
      GELU_TANH_F32S_AVX512(zmm13, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[2, 0-15]
      GELU_TANH_F32S_AVX512(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[2, 16-31]
      GELU_TANH_F32S_AVX512(zmm17, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[3, 0-15]
      GELU_TANH_F32S_AVX512(zmm20, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[3, 16-31]
      GELU_TANH_F32S_AVX512(zmm21, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[4, 0-15]
      GELU_TANH_F32S_AVX512(zmm24, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[4, 16-31]
      GELU_TANH_F32S_AVX512(zmm25, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_5x32F:
    {
      // c[0, 0-15]
      GELU_ERF_F32S_AVX512(zmm8, zmm0, zmm1, zmm2)

      // c[0, 16-31]
      GELU_ERF_F32S_AVX512(zmm9, zmm0, zmm1, zmm2)

      // c[1, 0-15]
      GELU_ERF_F32S_AVX512(zmm12, zmm0, zmm1, zmm2)

      // c[1, 16-31]
      GELU_ERF_F32S_AVX512(zmm13, zmm0, zmm1, zmm2)

      // c[2, 0-15]
      GELU_ERF_F32S_AVX512(zmm16, zmm0, zmm1, zmm2)

      // c[2, 16-31]
      GELU_ERF_F32S_AVX512(zmm17, zmm0, zmm1, zmm2)

      // c[3, 0-15]
      GELU_ERF_F32S_AVX512(zmm20, zmm0, zmm1, zmm2)

      // c[3, 16-31]
      GELU_ERF_F32S_AVX512(zmm21, zmm0, zmm1, zmm2)

      // c[4, 0-15]
      GELU_ERF_F32S_AVX512(zmm24, zmm0, zmm1, zmm2)

      // c[4, 16-31]
      GELU_ERF_F32S_AVX512(zmm25, zmm0, zmm1, zmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_5x32F:
    {
      zmm0 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      zmm1 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0, 0-15]
      CLIP_F32S_AVX512(zmm8, zmm0, zmm1)

      // c[0, 16-31]
      CLIP_F32S_AVX512(zmm9, zmm0, zmm1)

      // c[1, 0-15]
      CLIP_F32S_AVX512(zmm12, zmm0, zmm1)

      // c[1, 16-31]
      CLIP_F32S_AVX512(zmm13, zmm0, zmm1)

      // c[2, 0-15]
      CLIP_F32S_AVX512(zmm16, zmm0, zmm1)

      // c[2, 16-31]
      CLIP_F32S_AVX512(zmm17, zmm0, zmm1)

      // c[3, 0-15]
      CLIP_F32S_AVX512(zmm20, zmm0, zmm1)

      // c[3, 16-31]
      CLIP_F32S_AVX512(zmm21, zmm0, zmm1)

      // c[4, 0-15]
      CLIP_F32S_AVX512(zmm24, zmm0, zmm1)

      // c[4, 16-31]
      CLIP_F32S_AVX512(zmm25, zmm0, zmm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_5x32F_DISABLE:
    ;

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
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_4x32)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_4x32F_DISABLE,
              &&POST_OPS_BIAS_4x32F,
              &&POST_OPS_RELU_4x32F,
              &&POST_OPS_RELU_SCALE_4x32F,
              &&POST_OPS_GELU_TANH_4x32F,
              &&POST_OPS_GELU_ERF_4x32F,
              &&POST_OPS_CLIP_4x32F
            };
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
    float *_cbuf = NULL;

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

    if ( beta != 0.0 )
    {
        _cbuf = cbuf;
        //load c and multiply with beta and 
        //add to accumulator and store back
        zmm3 = _mm512_set1_ps(beta);

        zmm0 = _mm512_load_ps(_cbuf);
        zmm1 = _mm512_load_ps(_cbuf + 16);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);
        _cbuf += rs_c;

        zmm0 = _mm512_load_ps(_cbuf);
        zmm1 = _mm512_load_ps(_cbuf + 16);
        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);
        _cbuf += rs_c;

        zmm0 = _mm512_load_ps(_cbuf);
        zmm1 = _mm512_load_ps(_cbuf+16);
        zmm16 = _mm512_fmadd_ps(zmm0, zmm3, zmm16);
        zmm17 = _mm512_fmadd_ps(zmm1, zmm3, zmm17);
        _cbuf += rs_c;

        zmm0 = _mm512_load_ps(_cbuf);
        zmm1 = _mm512_load_ps(_cbuf+16);
        zmm20 = _mm512_fmadd_ps(zmm0, zmm3, zmm20);
        zmm21 = _mm512_fmadd_ps(zmm1, zmm3, zmm21);
    }

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_4x32F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        zmm1 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        zmm2 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 1 * 16 ) );

        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[0, 16-31]
        zmm9 = _mm512_add_ps( zmm2, zmm9 );

        // c[1,0-15]
        zmm12 = _mm512_add_ps( zmm1, zmm12 );

        // c[1, 16-31]
        zmm13 = _mm512_add_ps( zmm2, zmm13 );

        // c[2,0-15]
        zmm16 = _mm512_add_ps( zmm1, zmm16 );

        // c[2, 16-31]
        zmm17 = _mm512_add_ps( zmm2, zmm17 );

        // c[3,0-15]
        zmm20 = _mm512_add_ps( zmm1, zmm20 );

        // c[3, 16-31]
        zmm21 = _mm512_add_ps( zmm2, zmm21 );
      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the bias array will be accessed by
        // the ic index, and each bias element corresponds to an
        // entire row of the transposed output array, instead of an
        // entire column.
        zmm1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 0 ) );
        zmm2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 1 ) );
        zmm3 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 2 ) );
        zmm4 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 3 ) );

        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[0, 16-31]
        zmm9 = _mm512_add_ps( zmm1, zmm9 );

        // c[1,0-15]
        zmm12 = _mm512_add_ps( zmm2, zmm12 );

        // c[1, 16-31]
        zmm13 = _mm512_add_ps( zmm2, zmm13 );

        // c[2,0-15]
        zmm16 = _mm512_add_ps( zmm3, zmm16 );

        // c[2, 16-31]
        zmm17 = _mm512_add_ps( zmm3, zmm17 );

        // c[3,0-15]
        zmm20 = _mm512_add_ps( zmm4, zmm20 );

        // c[3, 16-31]
        zmm21 = _mm512_add_ps( zmm4, zmm21 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_4x32F:
    {
      zmm1 = _mm512_setzero_ps();

      // c[0,0-15]
      zmm8 = _mm512_max_ps( zmm1, zmm8 );

      // c[0, 16-31]
      zmm9 = _mm512_max_ps( zmm1, zmm9 );

      // c[1,0-15]
      zmm12 = _mm512_max_ps( zmm1, zmm12 );

      // c[1,16-31]
      zmm13 = _mm512_max_ps( zmm1, zmm13 );

      // c[2,0-15]
      zmm16 = _mm512_max_ps( zmm1, zmm16 );

      // c[2,16-31]
      zmm17 = _mm512_max_ps( zmm1, zmm17 );

      // c[3,0-15]
      zmm20 = _mm512_max_ps( zmm1, zmm20 );

      // c[3,16-31]
      zmm21 = _mm512_max_ps( zmm1, zmm21 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_4x32F:
    {
      zmm1 = _mm512_setzero_ps();
      zmm2 =
        _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

      __mmask16 relu_cmp_mask;

      // c[0, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm8)

      // c[0, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm9)

      // c[1, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm12)

      // c[1, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm13)

      // c[2, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm16)

      // c[2, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm17)

      // c[3, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm20)

      // c[3, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm21)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_4x32F:
    {
      __m512i zmm6;
      // c[0, 0-15]
      GELU_TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[0, 16-31]
      GELU_TANH_F32S_AVX512(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 0-15]
      GELU_TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 16-31]
      GELU_TANH_F32S_AVX512(zmm13, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[2, 0-15]
      GELU_TANH_F32S_AVX512(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[2, 16-31]
      GELU_TANH_F32S_AVX512(zmm17, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[3, 0-15]
      GELU_TANH_F32S_AVX512(zmm20, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[3, 16-31]
      GELU_TANH_F32S_AVX512(zmm21, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_4x32F:
    {
      // c[0, 0-15]
      GELU_ERF_F32S_AVX512(zmm8, zmm0, zmm1, zmm2)

      // c[0, 16-31]
      GELU_ERF_F32S_AVX512(zmm9, zmm0, zmm1, zmm2)

      // c[1, 0-15]
      GELU_ERF_F32S_AVX512(zmm12, zmm0, zmm1, zmm2)

      // c[1, 16-31]
      GELU_ERF_F32S_AVX512(zmm13, zmm0, zmm1, zmm2)

      // c[2, 0-15]
      GELU_ERF_F32S_AVX512(zmm16, zmm0, zmm1, zmm2)

      // c[2, 16-31]
      GELU_ERF_F32S_AVX512(zmm17, zmm0, zmm1, zmm2)

      // c[3, 0-15]
      GELU_ERF_F32S_AVX512(zmm20, zmm0, zmm1, zmm2)

      // c[3, 16-31]
      GELU_ERF_F32S_AVX512(zmm21, zmm0, zmm1, zmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_4x32F:
    {
      zmm0 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      zmm1 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0, 0-15]
      CLIP_F32S_AVX512(zmm8, zmm0, zmm1)

      // c[0, 16-31]
      CLIP_F32S_AVX512(zmm9, zmm0, zmm1)

      // c[1, 0-15]
      CLIP_F32S_AVX512(zmm12, zmm0, zmm1)

      // c[1, 16-31]
      CLIP_F32S_AVX512(zmm13, zmm0, zmm1)

      // c[2, 0-15]
      CLIP_F32S_AVX512(zmm16, zmm0, zmm1)

      // c[2, 16-31]
      CLIP_F32S_AVX512(zmm17, zmm0, zmm1)

      // c[3, 0-15]
      CLIP_F32S_AVX512(zmm20, zmm0, zmm1)

      // c[3, 16-31]
      CLIP_F32S_AVX512(zmm21, zmm0, zmm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_4x32F_DISABLE:
    ;

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
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_3x32)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_3x32F_DISABLE,
              &&POST_OPS_BIAS_3x32F,
              &&POST_OPS_RELU_3x32F,
              &&POST_OPS_RELU_SCALE_3x32F,
              &&POST_OPS_GELU_TANH_3x32F,
              &&POST_OPS_GELU_ERF_3x32F,
              &&POST_OPS_CLIP_3x32F
            };
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
    float *_cbuf = NULL;

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

    if ( beta != 0.0 )
    {
        _cbuf = cbuf;
        //load c and multiply with beta and 
        //add to accumulator and store back
        zmm3 = _mm512_set1_ps(beta);

        zmm0 = _mm512_load_ps(_cbuf);
        zmm1 = _mm512_load_ps(_cbuf + 16);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);
        _cbuf += rs_c;

        zmm0 = _mm512_load_ps(_cbuf);
        zmm1 = _mm512_load_ps(_cbuf + 16);
        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);
        _cbuf += rs_c;

        zmm0 = _mm512_load_ps(_cbuf);
        zmm1 = _mm512_load_ps(_cbuf+16);
        zmm16 = _mm512_fmadd_ps(zmm0, zmm3, zmm16);
        zmm17 = _mm512_fmadd_ps(zmm1, zmm3, zmm17);
        _cbuf += rs_c;
    }

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_3x32F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        zmm1 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        zmm2 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 1 * 16 ) );

        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[0, 16-31]
        zmm9 = _mm512_add_ps( zmm2, zmm9 );

        // c[1,0-15]
        zmm12 = _mm512_add_ps( zmm1, zmm12 );

        // c[1, 16-31]
        zmm13 = _mm512_add_ps( zmm2, zmm13 );

        // c[2,0-15]
        zmm16 = _mm512_add_ps( zmm1, zmm16 );

        // c[2, 16-31]
        zmm17 = _mm512_add_ps( zmm2, zmm17 );
      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the bias array will be accessed by
        // the ic index, and each bias element corresponds to an
        // entire row of the transposed output array, instead of an
        // entire column.
        zmm1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 0 ) );
        zmm2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 1 ) );
        zmm3 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 2 ) );

        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[0, 16-31]
        zmm9 = _mm512_add_ps( zmm1, zmm9 );

        // c[1,0-15]
        zmm12 = _mm512_add_ps( zmm2, zmm12 );

        // c[1, 16-31]
        zmm13 = _mm512_add_ps( zmm2, zmm13 );

        // c[2,0-15]
        zmm16 = _mm512_add_ps( zmm3, zmm16 );

        // c[2, 16-31]
        zmm17 = _mm512_add_ps( zmm3, zmm17 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_3x32F:
    {
      zmm1 = _mm512_setzero_ps();

      // c[0,0-15]
      zmm8 = _mm512_max_ps( zmm1, zmm8 );

      // c[0, 16-31]
      zmm9 = _mm512_max_ps( zmm1, zmm9 );

      // c[1,0-15]
      zmm12 = _mm512_max_ps( zmm1, zmm12 );

      // c[1,16-31]
      zmm13 = _mm512_max_ps( zmm1, zmm13 );

      // c[2,0-15]
      zmm16 = _mm512_max_ps( zmm1, zmm16 );

      // c[2,16-31]
      zmm17 = _mm512_max_ps( zmm1, zmm17 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_3x32F:
    {
      zmm1 = _mm512_setzero_ps();
      zmm2 =
        _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

      __mmask16 relu_cmp_mask;

      // c[0, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm8)

      // c[0, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm9)

      // c[1, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm12)

      // c[1, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm13)

      // c[2, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm16)

      // c[2, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm17)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_3x32F:
    {
      __m512i zmm6;
      // c[0, 0-15]
      GELU_TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[0, 16-31]
      GELU_TANH_F32S_AVX512(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 0-15]
      GELU_TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 16-31]
      GELU_TANH_F32S_AVX512(zmm13, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[2, 0-15]
      GELU_TANH_F32S_AVX512(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[2, 16-31]
      GELU_TANH_F32S_AVX512(zmm17, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_3x32F:
    {
      // c[0, 0-15]
      GELU_ERF_F32S_AVX512(zmm8, zmm0, zmm1, zmm2)

      // c[0, 16-31]
      GELU_ERF_F32S_AVX512(zmm9, zmm0, zmm1, zmm2)

      // c[1, 0-15]
      GELU_ERF_F32S_AVX512(zmm12, zmm0, zmm1, zmm2)

      // c[1, 16-31]
      GELU_ERF_F32S_AVX512(zmm13, zmm0, zmm1, zmm2)

      // c[2, 0-15]
      GELU_ERF_F32S_AVX512(zmm16, zmm0, zmm1, zmm2)

      // c[2, 16-31]
      GELU_ERF_F32S_AVX512(zmm17, zmm0, zmm1, zmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_3x32F:
    {
      zmm0 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      zmm1 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0, 0-15]
      CLIP_F32S_AVX512(zmm8, zmm0, zmm1)

      // c[0, 16-31]
      CLIP_F32S_AVX512(zmm9, zmm0, zmm1)

      // c[1, 0-15]
      CLIP_F32S_AVX512(zmm12, zmm0, zmm1)

      // c[1, 16-31]
      CLIP_F32S_AVX512(zmm13, zmm0, zmm1)

      // c[2, 0-15]
      CLIP_F32S_AVX512(zmm16, zmm0, zmm1)

      // c[2, 16-31]
      CLIP_F32S_AVX512(zmm17, zmm0, zmm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_3x32F_DISABLE:
    ;

    _mm512_storeu_ps(cbuf, zmm8); 
    _mm512_storeu_ps(cbuf + 16, zmm9);
    cbuf += rs_c;
    _mm512_storeu_ps(cbuf, zmm12);
    _mm512_storeu_ps(cbuf + 16, zmm13);
    cbuf += rs_c;
    _mm512_storeu_ps(cbuf, zmm16);
    _mm512_storeu_ps(cbuf + 16, zmm17);
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_2x32)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_2x32F_DISABLE,
              &&POST_OPS_BIAS_2x32F,
              &&POST_OPS_RELU_2x32F,
              &&POST_OPS_RELU_SCALE_2x32F,
              &&POST_OPS_GELU_TANH_2x32F,
              &&POST_OPS_GELU_ERF_2x32F,
              &&POST_OPS_CLIP_2x32F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k0;

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5;
    __m512 zmm8, zmm9, zmm12, zmm13;

    /* zero the accumulator registers */
    ZERO_ACC_ZMM_4_REG(zmm8, zmm9, zmm12, zmm13);

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;
    float *_cbuf = NULL;

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

    if ( beta != 0.0 )
    {
        _cbuf = cbuf;
        //load c and multiply with beta and 
        //add to accumulator and store back
        zmm3 = _mm512_set1_ps(beta);

        zmm0 = _mm512_load_ps(_cbuf);
        zmm1 = _mm512_load_ps(_cbuf + 16);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);
        _cbuf += rs_c;

        zmm0 = _mm512_load_ps(_cbuf);
        zmm1 = _mm512_load_ps(_cbuf + 16);
        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);
    }

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_2x32F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        zmm1 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        zmm2 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 1 * 16 ) );

        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[0, 16-31]
        zmm9 = _mm512_add_ps( zmm2, zmm9 );

        // c[1,0-15]
        zmm12 = _mm512_add_ps( zmm1, zmm12 );

        // c[1, 16-31]
        zmm13 = _mm512_add_ps( zmm2, zmm13 );
      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the bias array will be accessed by
        // the ic index, and each bias element corresponds to an
        // entire row of the transposed output array, instead of an
        // entire column.
        zmm1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 0 ) );
        zmm2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 1 ) );

        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[0, 16-31]
        zmm9 = _mm512_add_ps( zmm1, zmm9 );

        // c[1,0-15]
        zmm12 = _mm512_add_ps( zmm2, zmm12 );

        // c[1, 16-31]
        zmm13 = _mm512_add_ps( zmm2, zmm13 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_2x32F:
    {
      zmm1 = _mm512_setzero_ps();

      // c[0,0-15]
      zmm8 = _mm512_max_ps( zmm1, zmm8 );

      // c[0, 16-31]
      zmm9 = _mm512_max_ps( zmm1, zmm9 );

      // c[1,0-15]
      zmm12 = _mm512_max_ps( zmm1, zmm12 );

      // c[1,16-31]
      zmm13 = _mm512_max_ps( zmm1, zmm13 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_2x32F:
    {
      zmm1 = _mm512_setzero_ps();
      zmm2 =
        _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

      __mmask16 relu_cmp_mask;

      // c[0, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm8)

      // c[0, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm9)

      // c[1, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm12)

      // c[1, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm13)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_2x32F:
    {
      __m512i zmm6;
      // c[0, 0-15]
      GELU_TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[0, 16-31]
      GELU_TANH_F32S_AVX512(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 0-15]
      GELU_TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 16-31]
      GELU_TANH_F32S_AVX512(zmm13, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_2x32F:
    {
      // c[0, 0-15]
      GELU_ERF_F32S_AVX512(zmm8, zmm0, zmm1, zmm2)

      // c[0, 16-31]
      GELU_ERF_F32S_AVX512(zmm9, zmm0, zmm1, zmm2)

      // c[1, 0-15]
      GELU_ERF_F32S_AVX512(zmm12, zmm0, zmm1, zmm2)

      // c[1, 16-31]
      GELU_ERF_F32S_AVX512(zmm13, zmm0, zmm1, zmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_2x32F:
    {
      zmm0 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      zmm1 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0, 0-15]
      CLIP_F32S_AVX512(zmm8, zmm0, zmm1)

      // c[0, 16-31]
      CLIP_F32S_AVX512(zmm9, zmm0, zmm1)

      // c[1, 0-15]
      CLIP_F32S_AVX512(zmm12, zmm0, zmm1)

      // c[1, 16-31]
      CLIP_F32S_AVX512(zmm13, zmm0, zmm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_2x32F_DISABLE:
    ;

    _mm512_storeu_ps(cbuf, zmm8); 
    _mm512_storeu_ps(cbuf + 16, zmm9);
    cbuf += rs_c;
    _mm512_storeu_ps(cbuf, zmm12);
    _mm512_storeu_ps(cbuf + 16, zmm13);
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_1x32)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_1x32F_DISABLE,
              &&POST_OPS_BIAS_1x32F,
              &&POST_OPS_RELU_1x32F,
              &&POST_OPS_RELU_SCALE_1x32F,
              &&POST_OPS_GELU_TANH_1x32F,
              &&POST_OPS_GELU_ERF_1x32F,
              &&POST_OPS_CLIP_1x32F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k0;

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5;
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

    if ( beta != 0.0 )
    {
        //load c and multiply with beta and 
        //add to accumulator and store back
        zmm3 = _mm512_set1_ps(beta);

        zmm0 = _mm512_load_ps(cbuf);
        zmm1 = _mm512_load_ps(cbuf + 16);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);
    }

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_1x32F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        zmm1 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        zmm2 =
          _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 1 * 16 ) );

        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[0, 16-31]
        zmm9 = _mm512_add_ps( zmm2, zmm9 );
      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the bias array will be accessed by
        // the ic index, and each bias element corresponds to an
        // entire row of the transposed output array, instead of an
        // entire column.
        zmm1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_i + 0 ) );

        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[0, 16-31]
        zmm9 = _mm512_add_ps( zmm1, zmm9 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_1x32F:
    {
      zmm1 = _mm512_setzero_ps();

      // c[0,0-15]
      zmm8 = _mm512_max_ps( zmm1, zmm8 );

      // c[0, 16-31]
      zmm9 = _mm512_max_ps( zmm1, zmm9 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_1x32F:
    {
      zmm1 = _mm512_setzero_ps();
      zmm2 =
        _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

      __mmask16 relu_cmp_mask;

      // c[0, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm8)

      // c[0, 16-31]
      RELU_SCALE_OP_F32S_AVX512(zmm9)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_1x32F:
    {
      __m512i zmm6;
      // c[0, 0-15]
      GELU_TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[0, 16-31]
      GELU_TANH_F32S_AVX512(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_1x32F:
    {
      // c[0, 0-15]
      GELU_ERF_F32S_AVX512(zmm8, zmm0, zmm1, zmm2)

      // c[0, 16-31]
      GELU_ERF_F32S_AVX512(zmm9, zmm0, zmm1, zmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_1x32F:
    {
      zmm0 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      zmm1 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0, 0-15]
      CLIP_F32S_AVX512(zmm8, zmm0, zmm1)

      // c[0, 16-31]
      CLIP_F32S_AVX512(zmm9, zmm0, zmm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_1x32F_DISABLE:
    ;

    _mm512_storeu_ps(cbuf, zmm8); 
    _mm512_storeu_ps(cbuf + 16, zmm9);
}
#endif
