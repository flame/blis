/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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
              &&POST_OPS_CLIP_5x64F,
              &&POST_OPS_DOWNSCALE_5x64F,
              &&POST_OPS_MATRIX_ADD_5x64F,
              &&POST_OPS_SWISH_5x64F,
              &&POST_OPS_MATRIX_MUL_5x64F,
              &&POST_OPS_TANH_5x64F,
              &&POST_OPS_SIGMOID_5x64F
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
      zmm3 = _mm512_set1_ps(beta);

      //load c and beta, convert to f32
      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_first_k == TRUE ) )
      {
        //c[0, 0-15]
        BF16_F32_BETA_OP(zmm8, m, 0, 0, zmm0,zmm3);
        //c[0, 16-31]
        BF16_F32_BETA_OP(zmm9, m, 0, 1, zmm1,zmm3);
        //c[0, 32-47]
        BF16_F32_BETA_OP(zmm10, m, 0, 2, zmm0,zmm3);
        //c[0, 48-63]
        BF16_F32_BETA_OP(zmm11, m, 0, 3, zmm1,zmm3);
        //c[1, 0-15]
        BF16_F32_BETA_OP(zmm12, m, 1, 0, zmm0,zmm3);
        //c[1, 16-31]
        BF16_F32_BETA_OP(zmm13, m, 1, 1, zmm1,zmm3);
        //c[1, 32-47]
        BF16_F32_BETA_OP(zmm14, m, 1, 2, zmm0,zmm3);
        //c[1, 48-63]
        BF16_F32_BETA_OP(zmm15, m, 1, 3, zmm1,zmm3);
        //c[2, 0-15]
        BF16_F32_BETA_OP(zmm16, m, 2, 0, zmm0,zmm3);
        //c[2,16-31]
        BF16_F32_BETA_OP(zmm17, m, 2, 1, zmm1,zmm3);
        //c[2,32-47]
        BF16_F32_BETA_OP(zmm18, m, 2, 2, zmm0,zmm3);
        //c[2,48-63]
        BF16_F32_BETA_OP(zmm19, m, 2, 3, zmm1,zmm3);
        //c[3, 0-15]
        BF16_F32_BETA_OP(zmm20, m, 3, 0, zmm0,zmm3);
        //c[3,16-31]
        BF16_F32_BETA_OP(zmm21, m, 3, 1, zmm1,zmm3);
        //c[3,32-47]
        BF16_F32_BETA_OP(zmm22, m, 3, 2, zmm0,zmm3);
        //c[3,48-63]
        BF16_F32_BETA_OP(zmm23, m, 3, 3, zmm1,zmm3);
        //c[4, 0-15]
        BF16_F32_BETA_OP(zmm24, m, 4, 0, zmm0,zmm3);
        //c[4, 16-31]
        BF16_F32_BETA_OP(zmm25, m, 4, 1, zmm1,zmm3);
        //c[4, 32-47]
        BF16_F32_BETA_OP(zmm26, m, 4, 2, zmm0,zmm3);
        //c[4, 48-63]
        BF16_F32_BETA_OP(zmm27, m, 4, 3, zmm1,zmm3);
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);

        zmm0 = _mm512_loadu_ps(_cbuf + 32);
        zmm1 = _mm512_loadu_ps(_cbuf + 48);
        zmm10 = _mm512_fmadd_ps(zmm0, zmm3, zmm10);
        zmm11 = _mm512_fmadd_ps(zmm1, zmm3, zmm11);
        _cbuf += rs_c;

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);

        zmm0 = _mm512_loadu_ps(_cbuf + 32);
        zmm1 = _mm512_loadu_ps(_cbuf + 48);
        zmm14 = _mm512_fmadd_ps(zmm0, zmm3, zmm14);
        zmm15 = _mm512_fmadd_ps(zmm1, zmm3, zmm15);
        _cbuf += rs_c;

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm16 = _mm512_fmadd_ps(zmm0, zmm3, zmm16);
        zmm17 = _mm512_fmadd_ps(zmm1, zmm3, zmm17);

        zmm0 = _mm512_loadu_ps(_cbuf + 32);
        zmm1 = _mm512_loadu_ps(_cbuf + 48);
        zmm18 = _mm512_fmadd_ps(zmm0, zmm3, zmm18);
        zmm19 = _mm512_fmadd_ps(zmm1, zmm3, zmm19);
        _cbuf += rs_c;

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm20 = _mm512_fmadd_ps(zmm0, zmm3, zmm20);
        zmm21 = _mm512_fmadd_ps(zmm1, zmm3, zmm21);

        zmm0 = _mm512_loadu_ps(_cbuf + 32);
        zmm1 = _mm512_loadu_ps(_cbuf + 48);
        zmm22 = _mm512_fmadd_ps(zmm0, zmm3, zmm22);
        zmm23 = _mm512_fmadd_ps(zmm1, zmm3, zmm23);
        _cbuf += rs_c;

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm24 = _mm512_fmadd_ps(zmm0, zmm3, zmm24);
        zmm25 = _mm512_fmadd_ps(zmm1, zmm3, zmm25);

        zmm0 = _mm512_loadu_ps(_cbuf + 32);
        zmm1 = _mm512_loadu_ps(_cbuf + 48);
        zmm26 = _mm512_fmadd_ps(zmm0, zmm3, zmm26);
        zmm27 = _mm512_fmadd_ps(zmm1, zmm3, zmm27);
      }
    }

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_5x64F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_BIAS_LOAD(zmm1, bias_mask, 0)
          BF16_F32_BIAS_LOAD(zmm2, bias_mask, 1)
          BF16_F32_BIAS_LOAD(zmm3, bias_mask, 2)
          BF16_F32_BIAS_LOAD(zmm4, bias_mask, 3)
        }
        else
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
        }
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
        if ( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );

          BF16_F32_BIAS_BCAST(zmm1, bias_mask, 0)
          BF16_F32_BIAS_BCAST(zmm2, bias_mask, 1)
          BF16_F32_BIAS_BCAST(zmm3, bias_mask, 2)
          BF16_F32_BIAS_BCAST(zmm4, bias_mask, 3)
          BF16_F32_BIAS_BCAST(zmm5, bias_mask, 4)
        }
        else
        {
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
        }
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
POST_OPS_DOWNSCALE_5x64F:
    {
      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();
      __m512 selector3 = _mm512_setzero_ps();
      __m512 selector4 = _mm512_setzero_ps();

      __m512 zero_point0 = _mm512_setzero_ps();
      __m512 zero_point1 = _mm512_setzero_ps();
      __m512 zero_point2 = _mm512_setzero_ps();
      __m512 zero_point3 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                    ( ( post_ops_list_temp->stor_type == NONE ) &&
                      ( post_ops_attr.c_stor_type == BF16 ) );

      // Need to account for row vs column major swaps. For scalars
      // scale and zero point, no implications.
      // Even though different registers are used for scalar in column
      // and row major downscale path, all those registers will contain
      // the same value.

      if( post_ops_list_temp->scale_factor_len == 1 )
      {
        selector1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector2 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector3 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector4 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if ( is_bf16 == TRUE )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_ZP_BCST(zero_point0,0, zp_mask)
          BF16_F32_ZP_BCST(zero_point1,1, zp_mask)
          BF16_F32_ZP_BCST(zero_point2,2, zp_mask)
          BF16_F32_ZP_BCST(zero_point3,3, zp_mask)
        }
        else
        {
          zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point3 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          selector2 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 1 * 16 ) );
          selector3 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 2 * 16 ) );
          selector4 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 3 * 16 ) );
        }
        if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 load_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_LOAD(zero_point0,load_mask, 0)
            BF16_F32_ZP_LOAD(zero_point1,load_mask, 1)
            BF16_F32_ZP_LOAD(zero_point2,load_mask, 2)
            BF16_F32_ZP_LOAD(zero_point3,load_mask, 3)
          }
          else
          {
            zero_point0 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 0 * 16 ) );
            zero_point1 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 1 * 16 ) );
            zero_point2 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 2 * 16 ) );
            zero_point3 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 3 * 16 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[0, 16-31]
        F32_SCL_MULRND(zmm9, selector2, zero_point1);

        //c[0, 32-47]
        F32_SCL_MULRND(zmm10, selector3, zero_point2);

        //c[0, 48-63]
        F32_SCL_MULRND(zmm11, selector4, zero_point3);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector1, zero_point0);

        //c[1, 16-31]
        F32_SCL_MULRND(zmm13, selector2, zero_point1);

        //c[1, 32-47]
        F32_SCL_MULRND(zmm14, selector3, zero_point2);

        //c[1, 48-63]
        F32_SCL_MULRND(zmm15, selector4, zero_point3);

        //c[2, 0-15]
        F32_SCL_MULRND(zmm16, selector1, zero_point0);

        //c[2, 16-31]
        F32_SCL_MULRND(zmm17, selector2, zero_point1);

        //c[2, 32-47]
        F32_SCL_MULRND(zmm18, selector3, zero_point2);

        //c[2, 48-63]
        F32_SCL_MULRND(zmm19, selector4, zero_point3);

        //c[3, 0-15]
        F32_SCL_MULRND(zmm20, selector1, zero_point0);

        //c[3, 16-31]
        F32_SCL_MULRND(zmm21, selector2, zero_point1);

        //c[3, 32-47]
        F32_SCL_MULRND(zmm22, selector3, zero_point2);

        //c[3, 48-63]
        F32_SCL_MULRND(zmm23, selector4, zero_point3);

        //c[4, 0-15]
        F32_SCL_MULRND(zmm24, selector1, zero_point0);

        //c[4, 16-31]
        F32_SCL_MULRND(zmm25, selector2, zero_point1);

        //c[4, 32-47]
        F32_SCL_MULRND(zmm26, selector3, zero_point2);

        //c[4, 48-63]
        F32_SCL_MULRND(zmm27, selector4, zero_point3);
      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the scale as well as zp array will
        // be accessed by the ic index, and each scale/zp element
        // corresponds to an entire row of the transposed output array,
        // instead of an entire column.
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 0 ) );
          selector2 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 1 ) );
          selector3 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 2 ) );
          selector4 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 3 ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_COL_BCST(zero_point0, 0, zp_mask)
            BF16_F32_ZP_COL_BCST(zero_point1, 1, zp_mask)
            BF16_F32_ZP_COL_BCST(zero_point2, 2, zp_mask)
            BF16_F32_ZP_COL_BCST(zero_point3, 3, zp_mask)
          }
          else
          {
            zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 1 ) );
            zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 2 ) );
            zero_point3 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 3 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[0, 16-31]
        F32_SCL_MULRND(zmm9, selector1, zero_point0);

        //c[0, 32-47]
        F32_SCL_MULRND(zmm10, selector1, zero_point0);

        //c[0, 48-63]
        F32_SCL_MULRND(zmm11, selector1, zero_point0);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector2, zero_point1);

        //c[1, 16-31]
        F32_SCL_MULRND(zmm13, selector2, zero_point1);

        //c[1, 32-47]
        F32_SCL_MULRND(zmm14, selector2, zero_point1);

        //c[1, 48-63]
        F32_SCL_MULRND(zmm15, selector2, zero_point1);

        //c[2, 0-15]
        F32_SCL_MULRND(zmm16, selector3, zero_point2);

        //c[2, 16-31]
        F32_SCL_MULRND(zmm17, selector3, zero_point2);

        //c[2, 32-47]
        F32_SCL_MULRND(zmm18, selector3, zero_point2);

        //c[2, 48-63]
        F32_SCL_MULRND(zmm19, selector3, zero_point2);

        //c[3, 0-15]
        F32_SCL_MULRND(zmm20, selector4, zero_point3);

        //c[3, 16-31]
        F32_SCL_MULRND(zmm21, selector4, zero_point3);

        //c[3, 32-47]
        F32_SCL_MULRND(zmm22, selector4, zero_point3);

        //c[3, 48-63]
        F32_SCL_MULRND(zmm23, selector4, zero_point3);

        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 4 ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_COL_BCST(zero_point0, 4, zp_mask)
          }
          else
          {
            zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 4 ) );
          }
        }
        //c[4, 0-15]
        F32_SCL_MULRND(zmm24, selector1, zero_point0);

        //c[4, 16-31]
        F32_SCL_MULRND(zmm25, selector1, zero_point0);

        //c[4, 32-47]
        F32_SCL_MULRND(zmm26, selector1, zero_point0);

        //c[4, 48-63]
        F32_SCL_MULRND(zmm27, selector1, zero_point0);

      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_5x64F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();
      __m512 selector3 = _mm512_setzero_ps();
      __m512 selector4 = _mm512_setzero_ps();

      __m512 scl_fctr1 = _mm512_setzero_ps();
      __m512 scl_fctr2 = _mm512_setzero_ps();
      __m512 scl_fctr3 = _mm512_setzero_ps();
      __m512 scl_fctr4 = _mm512_setzero_ps();
      __m512 scl_fctr5 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                ( post_ops_attr.c_stor_type == BF16 ) );

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr4 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr5 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          scl_fctr2 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 1 * 16 ) );
          scl_fctr3 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 2 * 16 ) );
          scl_fctr4 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 3 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
          scl_fctr4 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 3 ) );
          scl_fctr5 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 4 ) );
        }
      }

      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31,32-47,48-63]
          BF16_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0,8,9,10,11);

          // c[1:0-15,16-31,32-47,48-63]
          BF16_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1,12,13,14,15);

          // c[2:0-15,16-31,32-47,48-63]
          BF16_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,2,16,17,18,19);

          // c[3:0-15,16-31,32-47,48-63]
          BF16_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,3,20,21,22,23);

          // c[4:0-15,16-31,32-47,48-63]
          BF16_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,4,24,25,26,27);
        }
        else
        {
          // c[0:0-15,16-31,32-47,48-63]
          BF16_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0,8,9,10,11);

          // c[1:0-15,16-31,32-47,48-63]
          BF16_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1,12,13,14,15);

          // c[2:0-15,16-31,32-47,48-63]
          BF16_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,2,16,17,18,19);

          // c[3:0-15,16-31,32-47,48-63]
          BF16_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,3,20,21,22,23);

          // c[4:0-15,16-31,32-47,48-63]
          BF16_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr5,scl_fctr5,scl_fctr5,scl_fctr5,4,24,25,26,27);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31,32-47,48-63]
          F32_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0,8,9,10,11);

          // c[1:0-15,16-31,32-47,48-63]
          F32_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1,12,13,14,15);

          // c[2:0-15,16-31,32-47,48-63]
          F32_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,2,16,17,18,19);

          // c[3:0-15,16-31,32-47,48-63]
          F32_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,3,20,21,22,23);

          // c[4:0-15,16-31,32-47,48-63]
          F32_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,4,24,25,26,27);
        }
        else
        {
          // c[0:0-15,16-31,32-47,48-63]
          F32_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0,8,9,10,11);

          // c[1:0-15,16-31,32-47,48-63]
          F32_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1,12,13,14,15);

          // c[2:0-15,16-31,32-47,48-63]
          F32_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,2,16,17,18,19);

          // c[3:0-15,16-31,32-47,48-63]
          F32_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,3,20,21,22,23);

          // c[4:0-15,16-31,32-47,48-63]
          F32_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr5,scl_fctr5,scl_fctr5,scl_fctr5,4,24,25,26,27);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_5x64F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();
      __m512 selector3 = _mm512_setzero_ps();
      __m512 selector4 = _mm512_setzero_ps();

      __m512 scl_fctr1 = _mm512_setzero_ps();
      __m512 scl_fctr2 = _mm512_setzero_ps();
      __m512 scl_fctr3 = _mm512_setzero_ps();
      __m512 scl_fctr4 = _mm512_setzero_ps();
      __m512 scl_fctr5 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                    ( post_ops_attr.c_stor_type == BF16 ) );

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr4 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr5 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
              ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          scl_fctr2 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 1 * 16 ) );
          scl_fctr3 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 2 * 16 ) );
          scl_fctr4 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 3 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
          scl_fctr4 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 3 ) );
          scl_fctr5 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 4 ) );
        }
      }
      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31,32-47,48-63]
          BF16_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0,8,9,10,11);

          // c[1:0-15,16-31,32-47,48-63]
          BF16_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1,12,13,14,15);

          // c[2:0-15,16-31,32-47,48-63]
          BF16_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,2,16,17,18,19);

          // c[3:0-15,16-31,32-47,48-63]
          BF16_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,3,20,21,22,23);

          // c[4:0-15,16-31,32-47,48-63]
          BF16_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,4,24,25,26,27);
        }
        else
        {

          // c[0:0-15,16-31,32-47,48-63]
          BF16_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0,8,9,10,11);

          // c[1:0-15,16-31,32-47,48-63]
          BF16_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1,12,13,14,15);

          // c[2:0-15,16-31,32-47,48-63]
          BF16_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,2,16,17,18,19);

          // c[3:0-15,16-31,32-47,48-63]
          BF16_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,3,20,21,22,23);

          // c[4:0-15,16-31,32-47,48-63]
          BF16_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr5,scl_fctr5,scl_fctr5,scl_fctr5,4,24,25,26,27);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31,32-47,48-63]
          F32_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0,8,9,10,11);

          // c[1:0-15,16-31,32-47,48-63]
          F32_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1,12,13,14,15);

          // c[2:0-15,16-31,32-47,48-63]
          F32_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,2,16,17,18,19);

          // c[3:0-15,16-31,32-47,48-63]
          F32_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,3,20,21,22,23);

          // c[4:0-15,16-31,32-47,48-63]
          F32_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,4,24,25,26,27);
        }
        else
        {

          // c[0:0-15,16-31,32-47,48-63]
          F32_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0,8,9,10,11);

          // c[1:0-15,16-31,32-47,48-63]
          F32_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1,12,13,14,15);

          // c[2:0-15,16-31,32-47,48-63]
          F32_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,2,16,17,18,19);

          // c[3:0-15,16-31,32-47,48-63]
          F32_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,3,20,21,22,23);

          // c[4:0-15,16-31,32-47,48-63]
          F32_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr5,scl_fctr5,scl_fctr5,scl_fctr5,4,24,25,26,27);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_5x64F:
    {
        zmm7 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );
        __m512i ex_out;

        // c[0, 0-15]
        SWISH_F32_AVX512_DEF(zmm8, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[0, 16-31]
        SWISH_F32_AVX512_DEF(zmm9, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[0, 32-47]
        SWISH_F32_AVX512_DEF(zmm10, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[0, 48-63]
        SWISH_F32_AVX512_DEF(zmm11, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 0-15]
        SWISH_F32_AVX512_DEF(zmm12, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 16-31]
        SWISH_F32_AVX512_DEF(zmm13, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 32-47]
        SWISH_F32_AVX512_DEF(zmm14, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 48-63]
        SWISH_F32_AVX512_DEF(zmm15, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[2, 0-15]
        SWISH_F32_AVX512_DEF(zmm16, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[2, 16-31]
        SWISH_F32_AVX512_DEF(zmm17, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[2, 32-47]
        SWISH_F32_AVX512_DEF(zmm18, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[2, 48-63]
        SWISH_F32_AVX512_DEF(zmm19, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[3, 0-15]
        SWISH_F32_AVX512_DEF(zmm20, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[3, 16-31]
        SWISH_F32_AVX512_DEF(zmm21, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[3, 32-47]
        SWISH_F32_AVX512_DEF(zmm22, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[3, 48-63]
        SWISH_F32_AVX512_DEF(zmm23, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[4, 0-15]
        SWISH_F32_AVX512_DEF(zmm24, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[4, 16-31]
        SWISH_F32_AVX512_DEF(zmm25, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[4, 32-47]
        SWISH_F32_AVX512_DEF(zmm26, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[4, 48-63]
        SWISH_F32_AVX512_DEF(zmm27, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_5x64F:
      {
        __m512i zmm6;
        // c[0, 0-15]
        TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[0, 16-31]
        TANH_F32S_AVX512(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[0, 32-47]
        TANH_F32S_AVX512(zmm10, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[0, 48-63]
        TANH_F32S_AVX512(zmm11, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 0-15]
        TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 16-31]
        TANH_F32S_AVX512(zmm13, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 32-47]
        TANH_F32S_AVX512(zmm14, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 48-63]
        TANH_F32S_AVX512(zmm15, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[2, 0-15]
        TANH_F32S_AVX512(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[2, 16-31]
        TANH_F32S_AVX512(zmm17, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[2, 32-47]
        TANH_F32S_AVX512(zmm18, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[2, 48-63]
        TANH_F32S_AVX512(zmm19, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[3, 0-15]
        TANH_F32S_AVX512(zmm20, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[3, 16-31]
        TANH_F32S_AVX512(zmm21, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[3, 32-47]
        TANH_F32S_AVX512(zmm22, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[3, 48-63]
        TANH_F32S_AVX512(zmm23, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[4, 0-15]
        TANH_F32S_AVX512(zmm24, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[4, 16-31]
        TANH_F32S_AVX512(zmm25, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[4, 32-47]
        TANH_F32S_AVX512(zmm26, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[4, 48-63]
        TANH_F32S_AVX512(zmm27, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_SIGMOID_5x64F:
      {
          __m512i ex_out;

          // c[0, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[0, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[0, 32-47]
          SIGMOID_F32_AVX512_DEF(zmm10, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[0, 48-63]
          SIGMOID_F32_AVX512_DEF(zmm11, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm13, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 32-47]
          SIGMOID_F32_AVX512_DEF(zmm14, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 48-63]
          SIGMOID_F32_AVX512_DEF(zmm15, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm17, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 32-47]
          SIGMOID_F32_AVX512_DEF(zmm18, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 48-63]
          SIGMOID_F32_AVX512_DEF(zmm19, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[3, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm20, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[3, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm21, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[3, 32-47]
          SIGMOID_F32_AVX512_DEF(zmm22, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[3, 48-63]
          SIGMOID_F32_AVX512_DEF(zmm23, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[4, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm24, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[4, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm25, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[4, 32-47]
          SIGMOID_F32_AVX512_DEF(zmm26, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[4, 48-63]
          SIGMOID_F32_AVX512_DEF(zmm27, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }

POST_OPS_5x64F_DISABLE:
      ;

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_last_k == TRUE ) )
      {
        uint32_t tlsb, rounded, temp[16] = {0};
        int i;
        bfloat16* dest;

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm8, 0, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm9, 0, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm10, 0, 2);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm11, 0, 3);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm12, 1, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm13, 1, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm14, 1, 2);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm15, 1, 3);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm16, 2, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm17, 2, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm18, 2, 2);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm19, 2, 3);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm20, 3, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm21, 3, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm22, 3, 2);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm23, 3, 3);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm24, 4, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm25, 4, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm26, 4, 2);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm27, 4, 3);
      }
      else
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
      }
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
              &&POST_OPS_CLIP_4x64F,
              &&POST_OPS_DOWNSCALE_4x64F,
              &&POST_OPS_MATRIX_ADD_4x64F,
              &&POST_OPS_SWISH_4x64F,
              &&POST_OPS_MATRIX_MUL_4x64F,
              &&POST_OPS_TANH_4x64F,
              &&POST_OPS_SIGMOID_4x64F
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
        zmm0 = _mm512_shuffle_ps(zmm0, zmm0, 0xE4); // dummy shuffle
        zmm1 = _mm512_shuffle_ps(zmm1, zmm1, 0xE4); // dummy shuffle
        /*Load Next 32 elements from row0 of B*/
        zmm6 = _mm512_loadu_ps (bbuf + 32); //load 32-47 from current row
        zmm7 = _mm512_loadu_ps (bbuf + 48); //load 48-63 from current row
        zmm6 = _mm512_shuffle_ps(zmm6, zmm6, 0xE4); // dummy shuffle
        zmm7 = _mm512_shuffle_ps(zmm7, zmm7, 0xE4); // dummy shuffle

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
      zmm3 = _mm512_set1_ps(beta);

      //load c and beta, convert to f32
      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_first_k == TRUE ) )
      {
        //c[0, 0-15]
        BF16_F32_BETA_OP(zmm8, m, 0, 0, zmm0,zmm3);
        //c[0, 16-31]
        BF16_F32_BETA_OP(zmm9, m, 0, 1, zmm1,zmm3);
        //c[0, 32-47]
        BF16_F32_BETA_OP(zmm10, m, 0, 2, zmm0,zmm3);
        //c[0, 48-63]
        BF16_F32_BETA_OP(zmm11, m, 0, 3, zmm1,zmm3);
        //c[1, 0-15]
        BF16_F32_BETA_OP(zmm12, m, 1, 0, zmm0,zmm3);
        //c[1, 16-31]
        BF16_F32_BETA_OP(zmm13, m, 1, 1, zmm1,zmm3);
        //c[1, 32-47]
        BF16_F32_BETA_OP(zmm14, m, 1, 2, zmm0,zmm3);
        //c[1, 48-63]
        BF16_F32_BETA_OP(zmm15, m, 1, 3, zmm1,zmm3);
        //c[2, 0-15]
        BF16_F32_BETA_OP(zmm16, m, 2, 0, zmm0,zmm3);
        //c[2,16-31]
        BF16_F32_BETA_OP(zmm17, m, 2, 1, zmm1,zmm3);
        //c[2,32-47]
        BF16_F32_BETA_OP(zmm18, m, 2, 2, zmm0,zmm3);
        //c[2,48-63]
        BF16_F32_BETA_OP(zmm19, m, 2, 3, zmm1,zmm3);
        //c[3, 0-15]
        BF16_F32_BETA_OP(zmm20, m, 3, 0, zmm0,zmm3);
        //c[3,16-31]
        BF16_F32_BETA_OP(zmm21, m, 3, 1, zmm1,zmm3);
        //c[3,32-47]
        BF16_F32_BETA_OP(zmm22, m, 3, 2, zmm0,zmm3);
        //c[3,48-63]
        BF16_F32_BETA_OP(zmm23, m, 3, 3, zmm1,zmm3);
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);

        zmm0 = _mm512_loadu_ps(_cbuf + 32);
        zmm1 = _mm512_loadu_ps(_cbuf + 48);
        zmm10 = _mm512_fmadd_ps(zmm0, zmm3, zmm10);
        zmm11 = _mm512_fmadd_ps(zmm1, zmm3, zmm11);
        _cbuf += rs_c;

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);

        zmm0 = _mm512_loadu_ps(_cbuf + 32);
        zmm1 = _mm512_loadu_ps(_cbuf + 48);
        zmm14 = _mm512_fmadd_ps(zmm0, zmm3, zmm14);
        zmm15 = _mm512_fmadd_ps(zmm1, zmm3, zmm15);
        _cbuf += rs_c;

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm16 = _mm512_fmadd_ps(zmm0, zmm3, zmm16);
        zmm17 = _mm512_fmadd_ps(zmm1, zmm3, zmm17);

        zmm0 = _mm512_loadu_ps(_cbuf + 32);
        zmm1 = _mm512_loadu_ps(_cbuf + 48);
        zmm18 = _mm512_fmadd_ps(zmm0, zmm3, zmm18);
        zmm19 = _mm512_fmadd_ps(zmm1, zmm3, zmm19);
        _cbuf += rs_c;

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm20 = _mm512_fmadd_ps(zmm0, zmm3, zmm20);
        zmm21 = _mm512_fmadd_ps(zmm1, zmm3, zmm21);

        zmm0 = _mm512_loadu_ps(_cbuf + 32);
        zmm1 = _mm512_loadu_ps(_cbuf + 48);
        zmm22 = _mm512_fmadd_ps(zmm0, zmm3, zmm22);
        zmm23 = _mm512_fmadd_ps(zmm1, zmm3, zmm23);
      }
    }

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_4x64F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_BIAS_LOAD(zmm1, bias_mask, 0)
          BF16_F32_BIAS_LOAD(zmm2, bias_mask, 1)
          BF16_F32_BIAS_LOAD(zmm3, bias_mask, 2)
          BF16_F32_BIAS_LOAD(zmm4, bias_mask, 3)
        }
        else
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
        }
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
        if ( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );

          BF16_F32_BIAS_BCAST(zmm1, bias_mask, 0)
          BF16_F32_BIAS_BCAST(zmm2, bias_mask, 1)
          BF16_F32_BIAS_BCAST(zmm3, bias_mask, 2)
          BF16_F32_BIAS_BCAST(zmm4, bias_mask, 3)
        }
        else
        {
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
        }
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
POST_OPS_DOWNSCALE_4x64F:
    {
      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();
      __m512 selector3 = _mm512_setzero_ps();
      __m512 selector4 = _mm512_setzero_ps();

      __m512 zero_point0 = _mm512_setzero_ps();
      __m512 zero_point1 = _mm512_setzero_ps();
      __m512 zero_point2 = _mm512_setzero_ps();
      __m512 zero_point3 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                    ( ( post_ops_list_temp->stor_type == NONE ) &&
                      ( post_ops_attr.c_stor_type == BF16 ) );

      // Need to account for row vs column major swaps. For scalars
      // scale and zero point, no implications.
      // Even though different registers are used for scalar in column
      // and row major downscale path, all those registers will contain
      // the same value.

      if( post_ops_list_temp->scale_factor_len == 1 )
      {
        selector1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector2 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector3 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector4 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if ( is_bf16 == TRUE )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_ZP_BCST(zero_point0,0, zp_mask)
          BF16_F32_ZP_BCST(zero_point1,1, zp_mask)
          BF16_F32_ZP_BCST(zero_point2,2, zp_mask)
          BF16_F32_ZP_BCST(zero_point3,3, zp_mask)
        }
        else
        {
          zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point3 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          selector2 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 1 * 16 ) );
          selector3 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 2 * 16 ) );
          selector4 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 3 * 16 ) );
        }
        if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 load_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_LOAD(zero_point0,load_mask, 0)
            BF16_F32_ZP_LOAD(zero_point1,load_mask, 1)
            BF16_F32_ZP_LOAD(zero_point2,load_mask, 2)
            BF16_F32_ZP_LOAD(zero_point3,load_mask, 3)
          }
          else
          {
            zero_point0 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 0 * 16 ) );
            zero_point1 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 1 * 16 ) );
            zero_point2 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 2 * 16 ) );
            zero_point3 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 3 * 16 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[0, 16-31]
        F32_SCL_MULRND(zmm9, selector2, zero_point1);

        //c[0, 32-47]
        F32_SCL_MULRND(zmm10, selector3, zero_point2);

        //c[0, 48-63]
        F32_SCL_MULRND(zmm11, selector4, zero_point3);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector1, zero_point0);

        //c[1, 16-31]
        F32_SCL_MULRND(zmm13, selector2, zero_point1);

        //c[1, 32-47]
        F32_SCL_MULRND(zmm14, selector3, zero_point2);

        //c[1, 48-63]
        F32_SCL_MULRND(zmm15, selector4, zero_point3);

        //c[2, 0-15]
        F32_SCL_MULRND(zmm16, selector1, zero_point0);

        //c[2, 16-31]
        F32_SCL_MULRND(zmm17, selector2, zero_point1);

        //c[2, 32-47]
        F32_SCL_MULRND(zmm18, selector3, zero_point2);

        //c[2, 48-63]
        F32_SCL_MULRND(zmm19, selector4, zero_point3);

        //c[3, 0-15]
        F32_SCL_MULRND(zmm20, selector1, zero_point0);

        //c[3, 16-31]
        F32_SCL_MULRND(zmm21, selector2, zero_point1);

        //c[3, 32-47]
        F32_SCL_MULRND(zmm22, selector3, zero_point2);

        //c[3, 48-63]
        F32_SCL_MULRND(zmm23, selector4, zero_point3);
      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the scale as well as zp array will
        // be accessed by the ic index, and each scale/zp element
        // corresponds to an entire row of the transposed output array,
        // instead of an entire column.
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 0 ) );
          selector2 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 1 ) );
          selector3 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 2 ) );
          selector4 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 3 ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_COL_BCST(zero_point0, 0, zp_mask)
            BF16_F32_ZP_COL_BCST(zero_point1, 1, zp_mask)
            BF16_F32_ZP_COL_BCST(zero_point2, 2, zp_mask)
            BF16_F32_ZP_COL_BCST(zero_point3, 3, zp_mask)
          }
          else
          {
            zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 1 ) );
            zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 2 ) );
            zero_point3 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 3 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[0, 16-31]
        F32_SCL_MULRND(zmm9, selector1, zero_point0);

        //c[0, 32-47]
        F32_SCL_MULRND(zmm10, selector1, zero_point0);

        //c[0, 48-63]
        F32_SCL_MULRND(zmm11, selector1, zero_point0);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector2, zero_point1);

        //c[1, 16-31]
        F32_SCL_MULRND(zmm13, selector2, zero_point1);

        //c[1, 32-47]
        F32_SCL_MULRND(zmm14, selector2, zero_point1);

        //c[1, 48-63]
        F32_SCL_MULRND(zmm15, selector2, zero_point1);

        //c[2, 0-15]
        F32_SCL_MULRND(zmm16, selector3, zero_point2);

        //c[2, 16-31]
        F32_SCL_MULRND(zmm17, selector3, zero_point2);

        //c[2, 32-47]
        F32_SCL_MULRND(zmm18, selector3, zero_point2);

        //c[2, 48-63]
        F32_SCL_MULRND(zmm19, selector3, zero_point2);

        //c[3, 0-15]
        F32_SCL_MULRND(zmm20, selector4, zero_point3);

        //c[3, 16-31]
        F32_SCL_MULRND(zmm21, selector4, zero_point3);

        //c[3, 32-47]
        F32_SCL_MULRND(zmm22, selector4, zero_point3);

        //c[3, 48-63]
        F32_SCL_MULRND(zmm23, selector4, zero_point3);
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_4x64F:
      {
        dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

        __m512 selector1 = _mm512_setzero_ps();
        __m512 selector2 = _mm512_setzero_ps();
        __m512 selector3 = _mm512_setzero_ps();
        __m512 selector4 = _mm512_setzero_ps();

        __m512 scl_fctr1 = _mm512_setzero_ps();
        __m512 scl_fctr2 = _mm512_setzero_ps();
        __m512 scl_fctr3 = _mm512_setzero_ps();
        __m512 scl_fctr4 = _mm512_setzero_ps();

	bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                ( post_ops_attr.c_stor_type == BF16 ) );

        // Even though different registers are used for scalar in column and
        // row major case, all those registers will contain the same value.
        if ( post_ops_list_temp->scale_factor_len == 1 )
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          scl_fctr3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          scl_fctr4 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        }
        else
        {
          if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
          {
            scl_fctr1 =
              _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 0 * 16 ) );
            scl_fctr2 =
              _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 1 * 16 ) );
            scl_fctr3 =
              _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 2 * 16 ) );
            scl_fctr4 =
              _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 3 * 16 ) );
          }
          else
          {
            scl_fctr1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 0 ) );
            scl_fctr2 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 1 ) );
            scl_fctr3 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 2 ) );
            scl_fctr4 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 3 ) );
          }
        }

        if ( is_bf16 == TRUE )
        {
          bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

          if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
          {
            // c[0:0-15,16-31,32-47,48-63]
            BF16_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0,8,9,10,11);

            // c[1:0-15,16-31,32-47,48-63]
            BF16_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1,12,13,14,15);

            // c[2:0-15,16-31,32-47,48-63]
            BF16_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,2,16,17,18,19);

            // c[3:0-15,16-31,32-47,48-63]
            BF16_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,3,20,21,22,23);
          }
          else
          {
            // c[0:0-15,16-31,32-47,48-63]
            BF16_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0,8,9,10,11);

            // c[1:0-15,16-31,32-47,48-63]
            BF16_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1,12,13,14,15);

            // c[2:0-15,16-31,32-47,48-63]
            BF16_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,2,16,17,18,19);

            // c[3:0-15,16-31,32-47,48-63]
            BF16_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,3,20,21,22,23);
          }
        }
        else
        {
          float* matptr = ( float* )post_ops_list_temp->op_args1;

          if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
          {
            // c[0:0-15,16-31,32-47,48-63]
            F32_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0,8,9,10,11);

            // c[1:0-15,16-31,32-47,48-63]
            F32_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1,12,13,14,15);

            // c[2:0-15,16-31,32-47,48-63]
            F32_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,2,16,17,18,19);

            // c[3:0-15,16-31,32-47,48-63]
            F32_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,3,20,21,22,23);
          }
          else
          {
            // c[0:0-15,16-31,32-47,48-63]
            F32_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0,8,9,10,11);

            // c[1:0-15,16-31,32-47,48-63]
            F32_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1,12,13,14,15);

            // c[2:0-15,16-31,32-47,48-63]
            F32_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,2,16,17,18,19);

            // c[3:0-15,16-31,32-47,48-63]
            F32_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,3,20,21,22,23);
          }
        }
        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_MATRIX_MUL_4x64F:
      {
        dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

        __m512 selector1 = _mm512_setzero_ps();
        __m512 selector2 = _mm512_setzero_ps();
        __m512 selector3 = _mm512_setzero_ps();
        __m512 selector4 = _mm512_setzero_ps();

        __m512 scl_fctr1 = _mm512_setzero_ps();
        __m512 scl_fctr2 = _mm512_setzero_ps();
        __m512 scl_fctr3 = _mm512_setzero_ps();
        __m512 scl_fctr4 = _mm512_setzero_ps();

        bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                    ( post_ops_attr.c_stor_type == BF16 ) );

        // Even though different registers are used for scalar in column and
        // row major case, all those registers will contain the same value.
        if ( post_ops_list_temp->scale_factor_len == 1 )
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          scl_fctr3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          scl_fctr4 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        }
        else
        {
          if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
                ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
          {
            scl_fctr1 =
              _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 0 * 16 ) );
            scl_fctr2 =
              _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 1 * 16 ) );
            scl_fctr3 =
              _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 2 * 16 ) );
            scl_fctr4 =
              _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 3 * 16 ) );
          }
          else
          {
            scl_fctr1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 0 ) );
            scl_fctr2 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 1 ) );
            scl_fctr3 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 2 ) );
            scl_fctr4 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 3 ) );
          }
        }
        if ( is_bf16 == TRUE )
        {
          bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

          if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
          {
            // c[0:0-15,16-31,32-47,48-63]
            BF16_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0,8,9,10,11);

            // c[1:0-15,16-31,32-47,48-63]
            BF16_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1,12,13,14,15);

            // c[2:0-15,16-31,32-47,48-63]
            BF16_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,2,16,17,18,19);

            // c[3:0-15,16-31,32-47,48-63]
            BF16_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,3,20,21,22,23);
          }
          else
          {
            // c[0:0-15,16-31,32-47,48-63]
            BF16_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0,8,9,10,11);

            // c[1:0-15,16-31,32-47,48-63]
            BF16_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1,12,13,14,15);

            // c[2:0-15,16-31,32-47,48-63]
            BF16_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,2,16,17,18,19);

            // c[3:0-15,16-31,32-47,48-63]
            BF16_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,3,20,21,22,23);
          }
        }
        else
        {
          float* matptr = ( float* )post_ops_list_temp->op_args1;

          if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
          {
            // c[0:0-15,16-31,32-47,48-63]
            F32_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0,8,9,10,11);

            // c[1:0-15,16-31,32-47,48-63]
            F32_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1,12,13,14,15);

            // c[2:0-15,16-31,32-47,48-63]
            F32_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,2,16,17,18,19);

            // c[3:0-15,16-31,32-47,48-63]
            F32_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,3,20,21,22,23);
          }
          else
          {
            // c[0:0-15,16-31,32-47,48-63]
            F32_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0,8,9,10,11);

            // c[1:0-15,16-31,32-47,48-63]
            F32_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1,12,13,14,15);

            // c[2:0-15,16-31,32-47,48-63]
            F32_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,2,16,17,18,19);

            // c[3:0-15,16-31,32-47,48-63]
            F32_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,3,20,21,22,23);
          }
        }
        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_SWISH_4x64F:
    {
        zmm7 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );
        __m512i ex_out;

        // c[0, 0-15]
        SWISH_F32_AVX512_DEF(zmm8, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[0, 16-31]
        SWISH_F32_AVX512_DEF(zmm9, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[0, 32-47]
        SWISH_F32_AVX512_DEF(zmm10, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[0, 48-63]
        SWISH_F32_AVX512_DEF(zmm11, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 0-15]
        SWISH_F32_AVX512_DEF(zmm12, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 16-31]
        SWISH_F32_AVX512_DEF(zmm13, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 32-47]
        SWISH_F32_AVX512_DEF(zmm14, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 48-63]
        SWISH_F32_AVX512_DEF(zmm15, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[2, 0-15]
        SWISH_F32_AVX512_DEF(zmm16, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[2, 16-31]
        SWISH_F32_AVX512_DEF(zmm17, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[2, 32-47]
        SWISH_F32_AVX512_DEF(zmm18, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[2, 48-63]
        SWISH_F32_AVX512_DEF(zmm19, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[3, 0-15]
        SWISH_F32_AVX512_DEF(zmm20, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[3, 16-31]
        SWISH_F32_AVX512_DEF(zmm21, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[3, 32-47]
        SWISH_F32_AVX512_DEF(zmm22, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[3, 48-63]
        SWISH_F32_AVX512_DEF(zmm23, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_4x64F:
      {
        __m512i zmm6;
        // c[0, 0-15]
        TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[0, 16-31]
        TANH_F32S_AVX512(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[0, 32-47]
        TANH_F32S_AVX512(zmm10, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[0, 48-63]
        TANH_F32S_AVX512(zmm11, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 0-15]
        TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 16-31]
        TANH_F32S_AVX512(zmm13, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 32-47]
        TANH_F32S_AVX512(zmm14, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 48-63]
        TANH_F32S_AVX512(zmm15, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[2, 0-15]
        TANH_F32S_AVX512(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[2, 16-31]
        TANH_F32S_AVX512(zmm17, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[2, 32-47]
        TANH_F32S_AVX512(zmm18, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[2, 48-63]
        TANH_F32S_AVX512(zmm19, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[3, 0-15]
        TANH_F32S_AVX512(zmm20, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[3, 16-31]
        TANH_F32S_AVX512(zmm21, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[3, 32-47]
        TANH_F32S_AVX512(zmm22, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[3, 48-63]
        TANH_F32S_AVX512(zmm23, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_SIGMOID_4x64F:
      {
          __m512i ex_out;

          // c[0, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[0, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[0, 32-47]
          SIGMOID_F32_AVX512_DEF(zmm10, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[0, 48-63]
          SIGMOID_F32_AVX512_DEF(zmm11, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm13, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 32-47]
          SIGMOID_F32_AVX512_DEF(zmm14, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 48-63]
          SIGMOID_F32_AVX512_DEF(zmm15, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm17, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 32-47]
          SIGMOID_F32_AVX512_DEF(zmm18, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 48-63]
          SIGMOID_F32_AVX512_DEF(zmm19, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[3, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm20, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[3, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm21, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[3, 32-47]
          SIGMOID_F32_AVX512_DEF(zmm22, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[3, 48-63]
          SIGMOID_F32_AVX512_DEF(zmm23, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_4x64F_DISABLE:
      ;

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
      ( post_ops_attr.is_last_k == TRUE ) )
      {
        uint32_t tlsb, rounded, temp[16] = {0};
        int i;
        bfloat16* dest;

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm8, 0, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm9, 0, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm10, 0, 2);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm11, 0, 3);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm12, 1, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm13, 1, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm14, 1, 2);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm15, 1, 3);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm16, 2, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm17, 2, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm18, 2, 2);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm19, 2, 3);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm20, 3, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm21, 3, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm22, 3, 2);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm23, 3, 3);
      }
      else
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
      }
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
              &&POST_OPS_CLIP_3x64F,
              &&POST_OPS_DOWNSCALE_3x64F,
              &&POST_OPS_MATRIX_ADD_3x64F,
              &&POST_OPS_SWISH_3x64F,
              &&POST_OPS_MATRIX_MUL_3x64F,
              &&POST_OPS_TANH_3x64F,
              &&POST_OPS_SIGMOID_3x64F
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
        zmm0 = _mm512_shuffle_ps(zmm0, zmm0, 0xE4); // dummy shuffle
        zmm1 = _mm512_shuffle_ps(zmm1, zmm1, 0xE4); // dummy shuffle

        /*Load Next 32 elements from row0 of B*/
        zmm6 = _mm512_loadu_ps (bbuf + 32); //load 32-47 from current row
        zmm7 = _mm512_loadu_ps (bbuf + 48); //load 48-63 from current row
        zmm6 = _mm512_shuffle_ps(zmm6, zmm6, 0xE4); // dummy shuffle
        zmm7 = _mm512_shuffle_ps(zmm7, zmm7, 0xE4); // dummy shuffle

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
      zmm3 = _mm512_set1_ps(beta);

      //load c and beta, convert to f32
      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_first_k == TRUE ) )
      {
        //c[0, 0-15]
        BF16_F32_BETA_OP(zmm8, m, 0, 0, zmm0,zmm3);
        //c[0, 16-31]
        BF16_F32_BETA_OP(zmm9, m, 0, 1, zmm1,zmm3);
        //c[0, 32-47]
        BF16_F32_BETA_OP(zmm10, m, 0, 2, zmm0,zmm3);
        //c[0, 48-63]
        BF16_F32_BETA_OP(zmm11, m, 0, 3, zmm1,zmm3);
        //c[1, 0-15]
        BF16_F32_BETA_OP(zmm12, m, 1, 0, zmm0,zmm3);
        //c[1, 16-31]
        BF16_F32_BETA_OP(zmm13, m, 1, 1, zmm1,zmm3);
        //c[1, 32-47]
        BF16_F32_BETA_OP(zmm14, m, 1, 2, zmm0,zmm3);
        //c[1, 48-63]
        BF16_F32_BETA_OP(zmm15, m, 1, 3, zmm1,zmm3);
        //c[2, 0-15]
        BF16_F32_BETA_OP(zmm16, m, 2, 0, zmm0,zmm3);
        //c[2,16-31]
        BF16_F32_BETA_OP(zmm17, m, 2, 1, zmm1,zmm3);
        //c[2,32-47]
        BF16_F32_BETA_OP(zmm18, m, 2, 2, zmm0,zmm3);
        //c[2,48-63]
        BF16_F32_BETA_OP(zmm19, m, 2, 3, zmm1,zmm3);
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);

        zmm0 = _mm512_loadu_ps(_cbuf + 32);
        zmm1 = _mm512_loadu_ps(_cbuf + 48);
        zmm10 = _mm512_fmadd_ps(zmm0, zmm3, zmm10);
        zmm11 = _mm512_fmadd_ps(zmm1, zmm3, zmm11);
        _cbuf += rs_c;

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);

        zmm0 = _mm512_loadu_ps(_cbuf + 32);
        zmm1 = _mm512_loadu_ps(_cbuf + 48);
        zmm14 = _mm512_fmadd_ps(zmm0, zmm3, zmm14);
        zmm15 = _mm512_fmadd_ps(zmm1, zmm3, zmm15);
        _cbuf += rs_c;

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm16 = _mm512_fmadd_ps(zmm0, zmm3, zmm16);
        zmm17 = _mm512_fmadd_ps(zmm1, zmm3, zmm17);

        zmm0 = _mm512_loadu_ps(_cbuf + 32);
        zmm1 = _mm512_loadu_ps(_cbuf + 48);
        zmm18 = _mm512_fmadd_ps(zmm0, zmm3, zmm18);
        zmm19 = _mm512_fmadd_ps(zmm1, zmm3, zmm19);
      }
    }

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_3x64F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_BIAS_LOAD(zmm1, bias_mask, 0)
          BF16_F32_BIAS_LOAD(zmm2, bias_mask, 1)
          BF16_F32_BIAS_LOAD(zmm3, bias_mask, 2)
          BF16_F32_BIAS_LOAD(zmm4, bias_mask, 3)
        }
        else
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
        }
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
        if ( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );

          BF16_F32_BIAS_BCAST(zmm1, bias_mask, 0)
          BF16_F32_BIAS_BCAST(zmm2, bias_mask, 1)
          BF16_F32_BIAS_BCAST(zmm3, bias_mask, 2)
        }
        else
        {
          zmm1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 0 ) );
          zmm2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 1 ) );
          zmm3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 2 ) );
        }
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
POST_OPS_DOWNSCALE_3x64F:
{
      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();
      __m512 selector3 = _mm512_setzero_ps();
      __m512 selector4 = _mm512_setzero_ps();

      __m512 zero_point0 = _mm512_setzero_ps();
      __m512 zero_point1 = _mm512_setzero_ps();
      __m512 zero_point2 = _mm512_setzero_ps();
      __m512 zero_point3 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                    ( ( post_ops_list_temp->stor_type == NONE ) &&
                      ( post_ops_attr.c_stor_type == BF16 ) );

      // Need to account for row vs column major swaps. For scalars
      // scale and zero point, no implications.
      // Even though different registers are used for scalar in column
      // and row major downscale path, all those registers will contain
      // the same value.

      if( post_ops_list_temp->scale_factor_len == 1 )
      {
        selector1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector2 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector3 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector4 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
         if ( is_bf16 == TRUE )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_BCST(zero_point0,0, zp_mask)
            BF16_F32_ZP_BCST(zero_point1,1, zp_mask)
            BF16_F32_ZP_BCST(zero_point2,2, zp_mask)
            BF16_F32_ZP_BCST(zero_point3,3, zp_mask)
          }
          else
          {
            zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point3 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          selector2 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 1 * 16 ) );
          selector3 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 2 * 16 ) );
          selector4 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 3 * 16 ) );
        }
        if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 load_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_LOAD(zero_point0,load_mask, 0)
            BF16_F32_ZP_LOAD(zero_point1,load_mask, 1)
            BF16_F32_ZP_LOAD(zero_point2,load_mask, 2)
            BF16_F32_ZP_LOAD(zero_point3,load_mask, 3)
          }
          else
          {
            zero_point0 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 0 * 16 ) );
            zero_point1 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 1 * 16 ) );
            zero_point2 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 2 * 16 ) );
            zero_point3 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 3 * 16 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[0, 16-31]
        F32_SCL_MULRND(zmm9, selector2, zero_point1);

        //c[0, 32-47]
        F32_SCL_MULRND(zmm10, selector3, zero_point2);

        //c[0, 48-63]
        F32_SCL_MULRND(zmm11, selector4, zero_point3);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector1, zero_point0);

        //c[1, 16-31]
        F32_SCL_MULRND(zmm13, selector2, zero_point1);

        //c[1, 32-47]
        F32_SCL_MULRND(zmm14, selector3, zero_point2);

        //c[1, 48-63]
        F32_SCL_MULRND(zmm15, selector4, zero_point3);

        //c[2, 0-15]
        F32_SCL_MULRND(zmm16, selector1, zero_point0);

        //c[2, 16-31]
        F32_SCL_MULRND(zmm17, selector2, zero_point1);

        //c[2, 32-47]
        F32_SCL_MULRND(zmm18, selector3, zero_point2);

        //c[2, 48-63]
        F32_SCL_MULRND(zmm19, selector4, zero_point3);
      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the scale as well as zp array will
        // be accessed by the ic index, and each scale/zp element
        // corresponds to an entire row of the transposed output array,
        // instead of an entire column.
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 0 ) );
          selector2 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 1 ) );
          selector3 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 2 ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_COL_BCST(zero_point0, 0, zp_mask)
            BF16_F32_ZP_COL_BCST(zero_point1, 1, zp_mask)
            BF16_F32_ZP_COL_BCST(zero_point2, 2, zp_mask)
          }
          else
          {
            zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 1 ) );
            zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 2 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[0, 16-31]
        F32_SCL_MULRND(zmm9, selector1, zero_point0);

        //c[0, 32-47]
        F32_SCL_MULRND(zmm10, selector1, zero_point0);

        //c[0, 48-63]
        F32_SCL_MULRND(zmm11, selector1, zero_point0);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector2, zero_point1);

        //c[1, 16-31]
        F32_SCL_MULRND(zmm13, selector2, zero_point1);

        //c[1, 32-47]
        F32_SCL_MULRND(zmm14, selector2, zero_point1);

        //c[1, 48-63]
        F32_SCL_MULRND(zmm15, selector2, zero_point1);

        //c[2, 0-15]
        F32_SCL_MULRND(zmm16, selector3, zero_point2);

        //c[2, 16-31]
        F32_SCL_MULRND(zmm17, selector3, zero_point2);

        //c[2, 32-47]
        F32_SCL_MULRND(zmm18, selector3, zero_point2);

        //c[2, 48-63]
        F32_SCL_MULRND(zmm19, selector3, zero_point2);
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
}
POST_OPS_MATRIX_ADD_3x64F:
      {
        dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

        __m512 selector1 = _mm512_setzero_ps();
        __m512 selector2 = _mm512_setzero_ps();
        __m512 selector3 = _mm512_setzero_ps();
        __m512 selector4 = _mm512_setzero_ps();

        __m512 scl_fctr1 = _mm512_setzero_ps();
        __m512 scl_fctr2 = _mm512_setzero_ps();
        __m512 scl_fctr3 = _mm512_setzero_ps();
        __m512 scl_fctr4 = _mm512_setzero_ps();

        bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                ( post_ops_attr.c_stor_type == BF16 ) );

        // Even though different registers are used for scalar in column and
        // row major case, all those registers will contain the same value.
        if ( post_ops_list_temp->scale_factor_len == 1 )
        {
          scl_fctr1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          scl_fctr3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          scl_fctr4 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        }
        else
        {
          if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
          {
            scl_fctr1 =
              _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 0 * 16 ) );
            scl_fctr2 =
              _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 1 * 16 ) );
            scl_fctr3 =
              _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 2 * 16 ) );
            scl_fctr4 =
              _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 3 * 16 ) );
          }
          else
          {
            scl_fctr1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 0 ) );
            scl_fctr2 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 1 ) );
            scl_fctr3 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 2 ) );
          }
        }

        if ( is_bf16 == TRUE )
        {
          bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

          if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
          {
            // c[0:0-15,16-31,32-47,48-63]
            BF16_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0,8,9,10,11);

            // c[1:0-15,16-31,32-47,48-63]
            BF16_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1,12,13,14,15);

            // c[2:0-15,16-31,32-47,48-63]
            BF16_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,2,16,17,18,19);
          }
          else
          {

            // c[0:0-15,16-31,32-47,48-63]
            BF16_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0,8,9,10,11);

            // c[1:0-15,16-31,32-47,48-63]
            BF16_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1,12,13,14,15);

            // c[2:0-15,16-31,32-47,48-63]
            BF16_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,2,16,17,18,19);
          }
        }
        else
        {
          float* matptr = ( float* )post_ops_list_temp->op_args1;

          if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
          {
            // c[0:0-15,16-31,32-47,48-63]
            F32_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0,8,9,10,11);

            // c[1:0-15,16-31,32-47,48-63]
            F32_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1,12,13,14,15);

            // c[2:0-15,16-31,32-47,48-63]
            F32_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,2,16,17,18,19);
          }
          else
          {

            // c[0:0-15,16-31,32-47,48-63]
            F32_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0,8,9,10,11);

            // c[1:0-15,16-31,32-47,48-63]
            F32_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1,12,13,14,15);

            // c[2:0-15,16-31,32-47,48-63]
            F32_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,2,16,17,18,19);
          }
        }
        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_MATRIX_MUL_3x64F:
      {
        dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

        __m512 selector1 = _mm512_setzero_ps();
        __m512 selector2 = _mm512_setzero_ps();
        __m512 selector3 = _mm512_setzero_ps();
        __m512 selector4 = _mm512_setzero_ps();

        __m512 scl_fctr1 = _mm512_setzero_ps();
        __m512 scl_fctr2 = _mm512_setzero_ps();
        __m512 scl_fctr3 = _mm512_setzero_ps();
        __m512 scl_fctr4 = _mm512_setzero_ps();

        bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                    ( post_ops_attr.c_stor_type == BF16 ) );

        // Even though different registers are used for scalar in column and
        // row major case, all those registers will contain the same value.
        if ( post_ops_list_temp->scale_factor_len == 1 )
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          scl_fctr3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          scl_fctr4 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        }
        else
        {
          if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
                ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
          {
            scl_fctr1 =
              _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 0 * 16 ) );
            scl_fctr2 =
              _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 1 * 16 ) );
            scl_fctr3 =
              _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 2 * 16 ) );
            scl_fctr4 =
              _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 3 * 16 ) );
          }
          else
          {
            scl_fctr1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 0 ) );
            scl_fctr2 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 1 ) );
            scl_fctr3 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 2 ) );
          }
        }
        if ( is_bf16 == TRUE )
        {
          bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

          if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
          {
            // c[0:0-15,16-31,32-47,48-63]
            BF16_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0,8,9,10,11);

            // c[1:0-15,16-31,32-47,48-63]
            BF16_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1,12,13,14,15);

            // c[2:0-15,16-31,32-47,48-63]
            BF16_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,2,16,17,18,19);
          }
          else
          {

            // c[0:0-15,16-31,32-47,48-63]
            BF16_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0,8,9,10,11);

            // c[1:0-15,16-31,32-47,48-63]
            BF16_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1,12,13,14,15);

            // c[2:0-15,16-31,32-47,48-63]
            BF16_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,2,16,17,18,19);
          }
        }
        else
        {
          float* matptr = ( float* )post_ops_list_temp->op_args1;

          if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
          {
            // c[0:0-15,16-31,32-47,48-63]
            F32_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0,8,9,10,11);

            // c[1:0-15,16-31,32-47,48-63]
            F32_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1,12,13,14,15);

            // c[2:0-15,16-31,32-47,48-63]
            F32_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,2,16,17,18,19);
          }
          else
          {
            // c[0:0-15,16-31,32-47,48-63]
            F32_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0,8,9,10,11);

            // c[1:0-15,16-31,32-47,48-63]
            F32_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1,12,13,14,15);

            // c[2:0-15,16-31,32-47,48-63]
            F32_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,2,16,17,18,19);
          }
        }
        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_SWISH_3x64F:
    {
        zmm7 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );
        __m512i ex_out;

        // c[0, 0-15]
        SWISH_F32_AVX512_DEF(zmm8, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[0, 16-31]
        SWISH_F32_AVX512_DEF(zmm9, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[0, 32-47]
        SWISH_F32_AVX512_DEF(zmm10, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[0, 48-63]
        SWISH_F32_AVX512_DEF(zmm11, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 0-15]
        SWISH_F32_AVX512_DEF(zmm12, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 16-31]
        SWISH_F32_AVX512_DEF(zmm13, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 32-47]
        SWISH_F32_AVX512_DEF(zmm14, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 48-63]
        SWISH_F32_AVX512_DEF(zmm15, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[2, 0-15]
        SWISH_F32_AVX512_DEF(zmm16, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[2, 16-31]
        SWISH_F32_AVX512_DEF(zmm17, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[2, 32-47]
        SWISH_F32_AVX512_DEF(zmm18, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[2, 48-63]
        SWISH_F32_AVX512_DEF(zmm19, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_3x64F:
      {
        __m512i zmm6;
        // c[0, 0-15]
        TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[0, 16-31]
        TANH_F32S_AVX512(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[0, 32-47]
        TANH_F32S_AVX512(zmm10, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[0, 48-63]
        TANH_F32S_AVX512(zmm11, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 0-15]
        TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 16-31]
        TANH_F32S_AVX512(zmm13, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 32-47]
        TANH_F32S_AVX512(zmm14, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 48-63]
        TANH_F32S_AVX512(zmm15, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[2, 0-15]
        TANH_F32S_AVX512(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[2, 16-31]
        TANH_F32S_AVX512(zmm17, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[2, 32-47]
        TANH_F32S_AVX512(zmm18, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[2, 48-63]
        TANH_F32S_AVX512(zmm19, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_SIGMOID_3x64F:
      {
          __m512i ex_out;

          // c[0, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[0, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[0, 32-47]
          SIGMOID_F32_AVX512_DEF(zmm10, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[0, 48-63]
          SIGMOID_F32_AVX512_DEF(zmm11, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm13, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 32-47]
          SIGMOID_F32_AVX512_DEF(zmm14, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 48-63]
          SIGMOID_F32_AVX512_DEF(zmm15, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm17, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 32-47]
          SIGMOID_F32_AVX512_DEF(zmm18, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 48-63]
          SIGMOID_F32_AVX512_DEF(zmm19, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_3x64F_DISABLE:
      ;

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_last_k == TRUE ) )
      {
        uint32_t tlsb, rounded, temp[16] = {0};
        int i;
        bfloat16* dest;

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm8, 0, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm9, 0, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm10, 0, 2);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm11, 0, 3);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm12, 1, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm13, 1, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm14, 1, 2);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm15, 1, 3);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm16, 2, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm17, 2, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm18, 2, 2);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm19, 2, 3);
      }
      else
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
      }
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
              &&POST_OPS_CLIP_2x64F,
              &&POST_OPS_DOWNSCALE_2x64F,
              &&POST_OPS_MATRIX_ADD_2x64F,
              &&POST_OPS_SWISH_2x64F,
              &&POST_OPS_MATRIX_MUL_2x64F,
              &&POST_OPS_TANH_2x64F,
              &&POST_OPS_SIGMOID_2x64F
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
        zmm0 = _mm512_shuffle_ps(zmm0, zmm0, 0xE4); // dummy shuffle
        zmm1 = _mm512_shuffle_ps(zmm1, zmm1, 0xE4); // dummy shuffle

        /*Load Next 32 elements from row0 of B*/
        zmm6 = _mm512_loadu_ps (bbuf + 32); //load 32-47 from current row
        zmm7 = _mm512_loadu_ps (bbuf + 48); //load 48-63 from current row
        zmm6 = _mm512_shuffle_ps(zmm6, zmm6, 0xE4); // dummy shuffle
        zmm7 = _mm512_shuffle_ps(zmm7, zmm7, 0xE4); // dummy shuffle

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
      zmm3 = _mm512_set1_ps(beta);

      //load c and beta, convert to f32
      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_first_k == TRUE ) )
      {
        //c[0, 0-15]
        BF16_F32_BETA_OP(zmm8, m, 0, 0, zmm0,zmm3);
        //c[0, 16-31]
        BF16_F32_BETA_OP(zmm9, m, 0, 1, zmm1,zmm3);
        //c[0, 32-47]
        BF16_F32_BETA_OP(zmm10, m, 0, 2, zmm0,zmm3);
        //c[0, 48-63]
        BF16_F32_BETA_OP(zmm11, m, 0, 3, zmm1,zmm3);
        //c[1, 0-15]
        BF16_F32_BETA_OP(zmm12, m, 1, 0, zmm0,zmm3);
        //c[1, 16-31]
        BF16_F32_BETA_OP(zmm13, m, 1, 1, zmm1,zmm3);
        //c[1, 32-47]
        BF16_F32_BETA_OP(zmm14, m, 1, 2, zmm0,zmm3);
        //c[1, 48-63]
        BF16_F32_BETA_OP(zmm15, m, 1, 3, zmm1,zmm3);
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);

        zmm0 = _mm512_loadu_ps(_cbuf + 32);
        zmm1 = _mm512_loadu_ps(_cbuf + 48);
        zmm10 = _mm512_fmadd_ps(zmm0, zmm3, zmm10);
        zmm11 = _mm512_fmadd_ps(zmm1, zmm3, zmm11);
        _cbuf += rs_c;

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);

        zmm0 = _mm512_loadu_ps(_cbuf + 32);
        zmm1 = _mm512_loadu_ps(_cbuf + 48);
        zmm14 = _mm512_fmadd_ps(zmm0, zmm3, zmm14);
        zmm15 = _mm512_fmadd_ps(zmm1, zmm3, zmm15);
      }
    }

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_2x64F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_BIAS_LOAD(zmm1, bias_mask, 0)
          BF16_F32_BIAS_LOAD(zmm2, bias_mask, 1)
          BF16_F32_BIAS_LOAD(zmm3, bias_mask, 2)
          BF16_F32_BIAS_LOAD(zmm4, bias_mask, 3)
        }
        else
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
        }
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
        if ( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );

          BF16_F32_BIAS_BCAST(zmm1, bias_mask, 0)
          BF16_F32_BIAS_BCAST(zmm2, bias_mask, 1)
        }
        else
        {
          zmm1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 0 ) );
          zmm2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 1 ) );
        }
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
POST_OPS_DOWNSCALE_2x64F:
    {
      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();
      __m512 selector3 = _mm512_setzero_ps();
      __m512 selector4 = _mm512_setzero_ps();

      __m512 zero_point0 = _mm512_setzero_ps();
      __m512 zero_point1 = _mm512_setzero_ps();
      __m512 zero_point2 = _mm512_setzero_ps();
      __m512 zero_point3 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                    ( ( post_ops_list_temp->stor_type == NONE ) &&
                      ( post_ops_attr.c_stor_type == BF16 ) );

      // Need to account for row vs column major swaps. For scalars
      // scale and zero point, no implications.
      // Even though different registers are used for scalar in column
      // and row major downscale path, all those registers will contain
      // the same value.

      if( post_ops_list_temp->scale_factor_len == 1 )
      {
        selector1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector2 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector3 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector4 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
         if ( is_bf16 == TRUE )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_BCST(zero_point0,0, zp_mask)
            BF16_F32_ZP_BCST(zero_point1,1, zp_mask)
            BF16_F32_ZP_BCST(zero_point2,2, zp_mask)
            BF16_F32_ZP_BCST(zero_point3,3, zp_mask)
          }
          else
          {
            zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point3 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          selector2 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 1 * 16 ) );
          selector3 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 2 * 16 ) );
          selector4 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 3 * 16 ) );
        }
        if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
            {
              __mmask16 load_mask = _cvtu32_mask16( 0xFFFF );
              BF16_F32_ZP_LOAD(zero_point0,load_mask, 0)
              BF16_F32_ZP_LOAD(zero_point1,load_mask, 1)
              BF16_F32_ZP_LOAD(zero_point2,load_mask, 2)
              BF16_F32_ZP_LOAD(zero_point3,load_mask, 3)
            }
            else
            {
              zero_point0 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_j + ( 0 * 16 ) );
              zero_point1 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_j + ( 1 * 16 ) );
              zero_point2 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_j + ( 2 * 16 ) );
              zero_point3 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_j + ( 3 * 16 ) );
            }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[0, 16-31]
        F32_SCL_MULRND(zmm9, selector2, zero_point1);

        //c[0, 32-47]
        F32_SCL_MULRND(zmm10, selector3, zero_point2);

        //c[0, 48-63]
        F32_SCL_MULRND(zmm11, selector4, zero_point3);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector1, zero_point0);

        //c[1, 16-31]
        F32_SCL_MULRND(zmm13, selector2, zero_point1);

        //c[1, 32-47]
        F32_SCL_MULRND(zmm14, selector3, zero_point2);

        //c[1, 48-63]
        F32_SCL_MULRND(zmm15, selector4, zero_point3);
      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the scale as well as zp array will
        // be accessed by the ic index, and each scale/zp element
        // corresponds to an entire row of the transposed output array,
        // instead of an entire column.
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 0 ) );
          selector2 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 1 ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_COL_BCST(zero_point0, 0, zp_mask)
            BF16_F32_ZP_COL_BCST(zero_point1, 1, zp_mask)
          }
          else
          {
            zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 1 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[0, 16-31]
        F32_SCL_MULRND(zmm9, selector1, zero_point0);

        //c[0, 32-47]
        F32_SCL_MULRND(zmm10, selector1, zero_point0);

        //c[0, 48-63]
        F32_SCL_MULRND(zmm11, selector1, zero_point0);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector2, zero_point1);

        //c[1, 16-31]
        F32_SCL_MULRND(zmm13, selector2, zero_point1);

        //c[1, 32-47]
        F32_SCL_MULRND(zmm14, selector2, zero_point1);

        //c[1, 48-63]
        F32_SCL_MULRND(zmm15, selector2, zero_point1);
      }
    POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
}
POST_OPS_MATRIX_ADD_2x64F:
      {
        dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

        __m512 selector1 = _mm512_setzero_ps();
        __m512 selector2 = _mm512_setzero_ps();
        __m512 selector3 = _mm512_setzero_ps();
        __m512 selector4 = _mm512_setzero_ps();

        __m512 scl_fctr1 = _mm512_setzero_ps();
        __m512 scl_fctr2 = _mm512_setzero_ps();
        __m512 scl_fctr3 = _mm512_setzero_ps();
        __m512 scl_fctr4 = _mm512_setzero_ps();

        bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                ( post_ops_attr.c_stor_type == BF16 ) );

        // Even though different registers are used for scalar in column and
        // row major case, all those registers will contain the same value.
        if ( post_ops_list_temp->scale_factor_len == 1 )
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          scl_fctr3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          scl_fctr4 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        }
        else
        {
          if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
          {
            scl_fctr1 =
              _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 0 * 16 ) );
            scl_fctr2 =
              _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 1 * 16 ) );
            scl_fctr3 =
              _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 2 * 16 ) );
            scl_fctr4 =
              _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 3 * 16 ) );
          }
          else
          {
            scl_fctr1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 0 ) );
            scl_fctr2 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 1 ) );
          }
        }

        if ( is_bf16 == TRUE )
        {
          bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

          if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
          {
            // c[0:0-15,16-31,32-47,48-63]
            BF16_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0,8,9,10,11);

            // c[1:0-15,16-31,32-47,48-63]
            BF16_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1,12,13,14,15);
          }
          else
          {
            // c[0:0-15,16-31,32-47,48-63]
            BF16_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0,8,9,10,11);

            // c[1:0-15,16-31,32-47,48-63]
            BF16_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1,12,13,14,15);
          }
        }
        else
        {
          float* matptr = ( float* )post_ops_list_temp->op_args1;

          if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
          {
            // c[0:0-15,16-31,32-47,48-63]
            F32_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0,8,9,10,11);

            // c[1:0-15,16-31,32-47,48-63]
            F32_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1,12,13,14,15);
          }
          else
          {
            // c[0:0-15,16-31,32-47,48-63]
            F32_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0,8,9,10,11);

            // c[1:0-15,16-31,32-47,48-63]
            F32_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1,12,13,14,15);
          }
        }
        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_MATRIX_MUL_2x64F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();
      __m512 selector3 = _mm512_setzero_ps();
      __m512 selector4 = _mm512_setzero_ps();

      __m512 scl_fctr1 = _mm512_setzero_ps();
      __m512 scl_fctr2 = _mm512_setzero_ps();
      __m512 scl_fctr3 = _mm512_setzero_ps();
      __m512 scl_fctr4 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                    ( post_ops_attr.c_stor_type == BF16 ) );

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr4 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
              ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          scl_fctr2 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 1 * 16 ) );
          scl_fctr3 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 2 * 16 ) );
          scl_fctr4 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 3 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
        }
      }
      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31,32-47,48-63]
          BF16_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0,8,9,10,11);

          // c[1:0-15,16-31,32-47,48-63]
          BF16_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1,12,13,14,15);
        }
        else
        {
          // c[0:0-15,16-31,32-47,48-63]
          BF16_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0,8,9,10,11);

          // c[1:0-15,16-31,32-47,48-63]
          BF16_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1,12,13,14,15);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31,32-47,48-63]
          F32_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0,8,9,10,11);

          // c[1:0-15,16-31,32-47,48-63]
          F32_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1,12,13,14,15);
        }
        else
        {

          // c[0:0-15,16-31,32-47,48-63]
          F32_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0,8,9,10,11);

          // c[1:0-15,16-31,32-47,48-63]
          F32_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
              scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1,12,13,14,15);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_2x64F:
    {
        zmm7 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );
        __m512i ex_out;

        // c[0, 0-15]
        SWISH_F32_AVX512_DEF(zmm8, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[0, 16-31]
        SWISH_F32_AVX512_DEF(zmm9, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[0, 32-47]
        SWISH_F32_AVX512_DEF(zmm10, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[0, 48-63]
        SWISH_F32_AVX512_DEF(zmm11, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 0-15]
        SWISH_F32_AVX512_DEF(zmm12, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 16-31]
        SWISH_F32_AVX512_DEF(zmm13, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 32-47]
        SWISH_F32_AVX512_DEF(zmm14, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 48-63]
        SWISH_F32_AVX512_DEF(zmm15, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_2x64F:
      {
        __m512i zmm6;
        // c[0, 0-15]
        TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[0, 16-31]
        TANH_F32S_AVX512(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[0, 32-47]
        TANH_F32S_AVX512(zmm10, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[0, 48-63]
        TANH_F32S_AVX512(zmm11, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 0-15]
        TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 16-31]
        TANH_F32S_AVX512(zmm13, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 32-47]
        TANH_F32S_AVX512(zmm14, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 48-63]
        TANH_F32S_AVX512(zmm15, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_SIGMOID_2x64F:
      {
          __m512i ex_out;

          // c[0, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[0, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[0, 32-47]
          SIGMOID_F32_AVX512_DEF(zmm10, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[0, 48-63]
          SIGMOID_F32_AVX512_DEF(zmm11, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm13, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 32-47]
          SIGMOID_F32_AVX512_DEF(zmm14, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 48-63]
          SIGMOID_F32_AVX512_DEF(zmm15, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_2x64F_DISABLE:
      ;

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_last_k == TRUE ) )
      {
        uint32_t tlsb, rounded, temp[16] = {0};
        int i;
        bfloat16* dest;

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm8, 0, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm9, 0, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm10, 0, 2);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm11, 0, 3);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm12, 1, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm13, 1, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm14, 1, 2);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm15, 1, 3);
      }
      else
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
      }
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
              &&POST_OPS_CLIP_1x64F,
              &&POST_OPS_DOWNSCALE_1x64F,
              &&POST_OPS_MATRIX_ADD_1x64F,
              &&POST_OPS_SWISH_1x64F,
              &&POST_OPS_MATRIX_MUL_1x64F,
              &&POST_OPS_TANH_1x64F,
              &&POST_OPS_SIGMOID_1x64F
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
      zmm3 = _mm512_set1_ps(beta);

      //load c and beta, convert to f32
      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_first_k == TRUE ) )
      {
        //c[0, 0-15]
        BF16_F32_BETA_OP(zmm8, m, 0, 0, zmm0,zmm3);
        //c[0, 16-31]
        BF16_F32_BETA_OP(zmm9, m, 0, 1, zmm1,zmm3);
        //c[0, 32-47]
        BF16_F32_BETA_OP(zmm10, m, 0, 2, zmm0,zmm3);
        //c[0, 48-63]
        BF16_F32_BETA_OP(zmm11, m, 0, 3, zmm1,zmm3);
      }
      else
      {
        //load c and multiply with beta and
        //add to accumulator and store back

        zmm0 = _mm512_loadu_ps(cbuf);
        zmm1 = _mm512_loadu_ps(cbuf + 16);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);

        zmm0 = _mm512_loadu_ps(cbuf + 32);
        zmm1 = _mm512_loadu_ps(cbuf + 48);
        zmm10 = _mm512_fmadd_ps(zmm0, zmm3, zmm10);
        zmm11 = _mm512_fmadd_ps(zmm1, zmm3, zmm11);
      }
    }

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_1x64F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_BIAS_LOAD(zmm1, bias_mask, 0)
          BF16_F32_BIAS_LOAD(zmm2, bias_mask, 1)
          BF16_F32_BIAS_LOAD(zmm3, bias_mask, 2)
          BF16_F32_BIAS_LOAD(zmm4, bias_mask, 3)
        }
        else
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
        }
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
        if ( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );

          BF16_F32_BIAS_BCAST(zmm1, bias_mask, 0)
        }
        else
        {
          zmm1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 0 ) );
        }
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
POST_OPS_DOWNSCALE_1x64F:
    {
      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();
      __m512 selector3 = _mm512_setzero_ps();
      __m512 selector4 = _mm512_setzero_ps();

      __m512 zero_point0 = _mm512_setzero_ps();
      __m512 zero_point1 = _mm512_setzero_ps();
      __m512 zero_point2 = _mm512_setzero_ps();
      __m512 zero_point3 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                    ( ( post_ops_list_temp->stor_type == NONE ) &&
                      ( post_ops_attr.c_stor_type == BF16 ) );

      // Need to account for row vs column major swaps. For scalars
      // scale and zero point, no implications.
      // Even though different registers are used for scalar in column
      // and row major downscale path, all those registers will contain
      // the same value.

      if( post_ops_list_temp->scale_factor_len == 1 )
      {
        selector1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector2 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector3 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector4 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
         if ( is_bf16 == TRUE )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_BCST(zero_point0,0, zp_mask)
            BF16_F32_ZP_BCST(zero_point1,1, zp_mask)
            BF16_F32_ZP_BCST(zero_point2,2, zp_mask)
            BF16_F32_ZP_BCST(zero_point3,3, zp_mask)
          }
          else
          {
            zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point3 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          selector2 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 1 * 16 ) );
          selector3 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 2 * 16 ) );
          selector4 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 3 * 16 ) );
        }
        if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 load_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_LOAD(zero_point0,load_mask, 0)
            BF16_F32_ZP_LOAD(zero_point1,load_mask, 1)
            BF16_F32_ZP_LOAD(zero_point2,load_mask, 2)
            BF16_F32_ZP_LOAD(zero_point3,load_mask, 3)
          }
          else
          {
            zero_point0 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 0 * 16 ) );
            zero_point1 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 1 * 16 ) );
            zero_point2 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 2 * 16 ) );
            zero_point3 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 3 * 16 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[0, 16-31]
        F32_SCL_MULRND(zmm9, selector2, zero_point1);

        //c[0, 32-47]
        F32_SCL_MULRND(zmm10, selector3, zero_point2);

        //c[0, 48-63]
        F32_SCL_MULRND(zmm11, selector4, zero_point3);
      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the scale as well as zp array will
        // be accessed by the ic index, and each scale/zp element
        // corresponds to an entire row of the transposed output array,
        // instead of an entire column.
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 0 ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_COL_BCST(zero_point0, 0, zp_mask)
          }
          else
          {
            zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 0 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[0, 16-31]
        F32_SCL_MULRND(zmm9, selector1, zero_point0);

        //c[0, 32-47]
        F32_SCL_MULRND(zmm10, selector1, zero_point0);

        //c[0, 48-63]
        F32_SCL_MULRND(zmm11, selector1, zero_point0);
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_1x64F:
      {
        dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

        __m512 selector1 = _mm512_setzero_ps();
        __m512 selector2 = _mm512_setzero_ps();
        __m512 selector3 = _mm512_setzero_ps();
        __m512 selector4 = _mm512_setzero_ps();

        __m512 scl_fctr1 = _mm512_setzero_ps();
        __m512 scl_fctr2 = _mm512_setzero_ps();
        __m512 scl_fctr3 = _mm512_setzero_ps();
        __m512 scl_fctr4 = _mm512_setzero_ps();

        bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                ( post_ops_attr.c_stor_type == BF16 ) );

        // Even though different registers are used for scalar in column and
        // row major case, all those registers will contain the same value.
        if ( post_ops_list_temp->scale_factor_len == 1 )
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          scl_fctr3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          scl_fctr4 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        }
        else
        {
          if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
          {
            scl_fctr1 =
              _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 0 * 16 ) );
            scl_fctr2 =
              _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 1 * 16 ) );
            scl_fctr3 =
              _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 2 * 16 ) );
            scl_fctr4 =
              _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 3 * 16 ) );
          }
          else
          {
            scl_fctr1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 0 ) );
          }
        }

        if ( is_bf16 == TRUE )
        {
          bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

          if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
          {
            // c[0:0-15,16-31,32-47,48-63]
            BF16_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0,8,9,10,11);
          }
          else
          {
            // c[0:0-15,16-31,32-47,48-63]
            BF16_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0,8,9,10,11);
          }
        }
        else
        {
          float* matptr = ( float* )post_ops_list_temp->op_args1;

          if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
          {
            // c[0:0-15,16-31,32-47,48-63]
            F32_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0,8,9,10,11);
          }
          else
          {
            // c[0:0-15,16-31,32-47,48-63]
            F32_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0,8,9,10,11);
          }
        }
        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_MATRIX_MUL_1x64F:
      {
        dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

        __m512 selector1 = _mm512_setzero_ps();
        __m512 selector2 = _mm512_setzero_ps();
        __m512 selector3 = _mm512_setzero_ps();
        __m512 selector4 = _mm512_setzero_ps();

        __m512 scl_fctr1 = _mm512_setzero_ps();
        __m512 scl_fctr2 = _mm512_setzero_ps();
        __m512 scl_fctr3 = _mm512_setzero_ps();
        __m512 scl_fctr4 = _mm512_setzero_ps();

        bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                    ( post_ops_attr.c_stor_type == BF16 ) );

        // Even though different registers are used for scalar in column and
        // row major case, all those registers will contain the same value.
        if ( post_ops_list_temp->scale_factor_len == 1 )
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          scl_fctr3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          scl_fctr4 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        }
        else
        {
          if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
                ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
          {
            scl_fctr1 =
              _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 0 * 16 ) );
            scl_fctr2 =
              _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 1 * 16 ) );
            scl_fctr3 =
              _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 2 * 16 ) );
            scl_fctr4 =
              _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 3 * 16 ) );
          }
          else
          {
            scl_fctr1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 0 ) );
          }
        }
        if ( is_bf16 == TRUE )
        {
          bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

          if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
          {
            // c[0:0-15,16-31,32-47,48-63]
            BF16_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0,8,9,10,11);
          }
          else
          {
            // c[0:0-15,16-31,32-47,48-63]
            BF16_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0,8,9,10,11);
          }
        }
        else
        {
          float* matptr = ( float* )post_ops_list_temp->op_args1;

          if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
          {
            // c[0:0-15,16-31,32-47,48-63]
            F32_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0,8,9,10,11);
          }
          else
          {
            // c[0:0-15,16-31,32-47,48-63]
            F32_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,\
                scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0,8,9,10,11);
          }
        }
        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_SWISH_1x64F:
    {
        zmm7 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );
        __m512i ex_out;

        // c[0, 0-15]
        SWISH_F32_AVX512_DEF(zmm8, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[0, 16-31]
        SWISH_F32_AVX512_DEF(zmm9, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[0, 32-47]
        SWISH_F32_AVX512_DEF(zmm10, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[0, 48-63]
        SWISH_F32_AVX512_DEF(zmm11, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_1x64F:
      {
        __m512i zmm6;
        // c[0, 0-15]
        TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[0, 16-31]
        TANH_F32S_AVX512(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[0, 32-47]
        TANH_F32S_AVX512(zmm10, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[0, 48-63]
        TANH_F32S_AVX512(zmm11, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_SIGMOID_1x64F:
      {
          __m512i ex_out;

          // c[0, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[0, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[0, 32-47]
          SIGMOID_F32_AVX512_DEF(zmm10, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[0, 48-63]
          SIGMOID_F32_AVX512_DEF(zmm11, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_1x64F_DISABLE:
      ;

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_last_k == TRUE ) )
      {
        uint32_t tlsb, rounded, temp[16] = {0};
        int i;
        bfloat16* dest;

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm8, 0, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm9, 0, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm10, 0, 2);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm11, 0, 3);
      }
      else
      {
        _mm512_storeu_ps(cbuf, zmm8);
        _mm512_storeu_ps(cbuf + 16, zmm9);
        _mm512_storeu_ps(cbuf + 32, zmm10);
        _mm512_storeu_ps(cbuf + 48, zmm11);
      }
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
              &&POST_OPS_CLIP_5x48F,
              &&POST_OPS_DOWNSCALE_5x48F,
              &&POST_OPS_MATRIX_ADD_5x48F,
              &&POST_OPS_SWISH_5x48F,
              &&POST_OPS_MATRIX_MUL_5x48F,
              &&POST_OPS_TANH_5x48F,
              &&POST_OPS_SIGMOID_5x48F
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
      zmm3 = _mm512_set1_ps(beta);

      //load c and beta, convert to f32
      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_first_k == TRUE ) )
      {
        //c[0, 0-15]
        BF16_F32_BETA_OP(zmm8, m, 0, 0, zmm0,zmm3);
        //c[0, 16-31]
        BF16_F32_BETA_OP(zmm9, m, 0, 1, zmm1,zmm3);
        //c[0, 32-47]
        BF16_F32_BETA_OP(zmm10, m, 0, 2, zmm0,zmm3);
        //c[1, 0-15]
        BF16_F32_BETA_OP(zmm12, m, 1, 0, zmm0,zmm3);
        //c[1, 16-31]
        BF16_F32_BETA_OP(zmm13, m, 1, 1, zmm1,zmm3);
        //c[1, 32-47]
        BF16_F32_BETA_OP(zmm14, m, 1, 2, zmm0,zmm3);
        //c[2, 0-15]
        BF16_F32_BETA_OP(zmm16, m, 2, 0, zmm0,zmm3);
        //c[2,16-31]
        BF16_F32_BETA_OP(zmm17, m, 2, 1, zmm1,zmm3);
        //c[2,32-47]
        BF16_F32_BETA_OP(zmm18, m, 2, 2, zmm0,zmm3);
        //c[3, 0-15]
        BF16_F32_BETA_OP(zmm20, m, 3, 0, zmm0,zmm3);
        //c[3,16-31]
        BF16_F32_BETA_OP(zmm21, m, 3, 1, zmm1,zmm3);
        //c[3,32-47]
        BF16_F32_BETA_OP(zmm22, m, 3, 2, zmm0,zmm3);
        //c[4, 0-15]
        BF16_F32_BETA_OP(zmm24, m, 4, 0, zmm0,zmm3);
        //c[4, 16-31]
        BF16_F32_BETA_OP(zmm25, m, 4, 1, zmm1,zmm3);
        //c[4, 32-47]
        BF16_F32_BETA_OP(zmm26, m, 4, 2, zmm0,zmm3);
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);

        zmm0 = _mm512_loadu_ps(_cbuf + 32);
        zmm10 = _mm512_fmadd_ps(zmm0, zmm3, zmm10);
        _cbuf += rs_c;

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);

        zmm0 = _mm512_loadu_ps(_cbuf + 32);
        zmm14 = _mm512_fmadd_ps(zmm0, zmm3, zmm14);
        _cbuf += rs_c;

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm16 = _mm512_fmadd_ps(zmm0, zmm3, zmm16);
        zmm17 = _mm512_fmadd_ps(zmm1, zmm3, zmm17);

        zmm0 = _mm512_loadu_ps(_cbuf + 32);
        zmm18 = _mm512_fmadd_ps(zmm0, zmm3, zmm18);
        _cbuf += rs_c;

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm20 = _mm512_fmadd_ps(zmm0, zmm3, zmm20);
        zmm21 = _mm512_fmadd_ps(zmm1, zmm3, zmm21);

        zmm0 = _mm512_loadu_ps(_cbuf + 32);
        zmm22 = _mm512_fmadd_ps(zmm0, zmm3, zmm22);
        _cbuf += rs_c;

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm24 = _mm512_fmadd_ps(zmm0, zmm3, zmm24);
        zmm25 = _mm512_fmadd_ps(zmm1, zmm3, zmm25);

        zmm0 = _mm512_loadu_ps(_cbuf + 32);
        zmm26 = _mm512_fmadd_ps(zmm0, zmm3, zmm26);
      }
    }

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_5x48F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {

        if( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_BIAS_LOAD(zmm1, bias_mask, 0)
          BF16_F32_BIAS_LOAD(zmm2, bias_mask, 1)
          BF16_F32_BIAS_LOAD(zmm3, bias_mask, 2)
        }
        else
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
        }
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
        if ( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );

          BF16_F32_BIAS_BCAST(zmm1, bias_mask, 0)
          BF16_F32_BIAS_BCAST(zmm2, bias_mask, 1)
          BF16_F32_BIAS_BCAST(zmm3, bias_mask, 2)
          BF16_F32_BIAS_BCAST(zmm4, bias_mask, 3)
          BF16_F32_BIAS_BCAST(zmm5, bias_mask, 4)
        }
        else
        {
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
        }
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
POST_OPS_DOWNSCALE_5x48F:
    {
      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();
      __m512 selector3 = _mm512_setzero_ps();
      __m512 selector4 = _mm512_setzero_ps();

      __m512 zero_point0 = _mm512_setzero_ps();
      __m512 zero_point1 = _mm512_setzero_ps();
      __m512 zero_point2 = _mm512_setzero_ps();
      __m512 zero_point3 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                    ( ( post_ops_list_temp->stor_type == NONE ) &&
                      ( post_ops_attr.c_stor_type == BF16 ) );

      // Need to account for row vs column major swaps. For scalars
      // scale and zero point, no implications.
      // Even though different registers are used for scalar in column
      // and row major downscale path, all those registers will contain
      // the same value.

      if( post_ops_list_temp->scale_factor_len == 1 )
      {
        selector1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector2 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector3 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector4 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
         if ( is_bf16 == TRUE )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_BCST(zero_point0,0, zp_mask)
            BF16_F32_ZP_BCST(zero_point1,1, zp_mask)
            BF16_F32_ZP_BCST(zero_point2,2, zp_mask)
            BF16_F32_ZP_BCST(zero_point3,3, zp_mask)
          }
          else
          {
            zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point3 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          selector2 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 1 * 16 ) );
          selector3 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 2 * 16 ) );
        }
        if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 load_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_LOAD(zero_point0,load_mask, 0)
            BF16_F32_ZP_LOAD(zero_point1,load_mask, 1)
            BF16_F32_ZP_LOAD(zero_point2,load_mask, 2)
          }
          else
          {
            zero_point0 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 0 * 16 ) );
            zero_point1 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 1 * 16 ) );
            zero_point2 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 2 * 16 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[0, 16-31]
        F32_SCL_MULRND(zmm9, selector2, zero_point1);

        //c[0, 32-47]
        F32_SCL_MULRND(zmm10, selector3, zero_point2);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector1, zero_point0);

        //c[1, 16-31]
        F32_SCL_MULRND(zmm13, selector2, zero_point1);

        //c[1, 32-47]
        F32_SCL_MULRND(zmm14, selector3, zero_point2);

        //c[2, 0-15]
        F32_SCL_MULRND(zmm16, selector1, zero_point0);

        //c[2, 16-31]
        F32_SCL_MULRND(zmm17, selector2, zero_point1);

        //c[2, 32-47]
        F32_SCL_MULRND(zmm18, selector3, zero_point2);

        //c[3, 0-15]
        F32_SCL_MULRND(zmm20, selector1, zero_point0);

        //c[3, 16-31]
        F32_SCL_MULRND(zmm21, selector2, zero_point1);

        //c[3, 32-47]
        F32_SCL_MULRND(zmm22, selector3, zero_point2);

        //c[4, 0-15]
        F32_SCL_MULRND(zmm24, selector1, zero_point0);

        //c[4, 16-31]
        F32_SCL_MULRND(zmm25, selector2, zero_point1);

        //c[4, 32-47]
        F32_SCL_MULRND(zmm26, selector3, zero_point2);
      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the scale as well as zp array will
        // be accessed by the ic index, and each scale/zp element
        // corresponds to an entire row of the transposed output array,
        // instead of an entire column.
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 0 ) );
          selector2 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 1 ) );
          selector3 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 2 ) );
          selector4 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 3 ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_COL_BCST(zero_point0, 0, zp_mask)
            BF16_F32_ZP_COL_BCST(zero_point1, 1, zp_mask)
            BF16_F32_ZP_COL_BCST(zero_point2, 2, zp_mask)
            BF16_F32_ZP_COL_BCST(zero_point3, 3, zp_mask)
          }
          else
          {
            zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 1 ) );
            zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 2 ) );
            zero_point3 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 3 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[0, 16-31]
        F32_SCL_MULRND(zmm9, selector1, zero_point0);

        //c[0, 32-47]
        F32_SCL_MULRND(zmm10, selector1, zero_point0);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector2, zero_point1);

        //c[1, 16-31]
        F32_SCL_MULRND(zmm13, selector2, zero_point1);

        //c[1, 32-47]
        F32_SCL_MULRND(zmm14, selector2, zero_point1);

        //c[2, 0-15]
        F32_SCL_MULRND(zmm16, selector3, zero_point2);

        //c[2, 16-31]
        F32_SCL_MULRND(zmm17, selector3, zero_point2);

        //c[2, 32-47]
        F32_SCL_MULRND(zmm18, selector3, zero_point2);

        //c[3, 0-15]
        F32_SCL_MULRND(zmm20, selector4, zero_point3);

        //c[3, 16-31]
        F32_SCL_MULRND(zmm21, selector4, zero_point3);

        //c[3, 32-47]
        F32_SCL_MULRND(zmm22, selector4, zero_point3);

        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 4 ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( ( post_ops_attr.buf_downscale != NULL ) &&
                ( post_ops_attr.is_first_k == TRUE ) )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_COL_BCST(zero_point0, 4, zp_mask)
          }
          else
          {
            zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 4 ) );
          }
        }
        //c[4, 0-15]
        F32_SCL_MULRND(zmm24, selector1, zero_point0);

        //c[4, 16-31]
        F32_SCL_MULRND(zmm25, selector1, zero_point0);

        //c[4, 32-47]
        F32_SCL_MULRND(zmm26, selector1, zero_point0);
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_5x48F:
      {
        dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

        __m512 selector1 = _mm512_setzero_ps();
        __m512 selector2 = _mm512_setzero_ps();
        __m512 selector3 = _mm512_setzero_ps();

        __m512 scl_fctr1 = _mm512_setzero_ps();
        __m512 scl_fctr2 = _mm512_setzero_ps();
        __m512 scl_fctr3 = _mm512_setzero_ps();
        __m512 scl_fctr4 = _mm512_setzero_ps();
        __m512 scl_fctr5 = _mm512_setzero_ps();

        bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                ( post_ops_attr.c_stor_type == BF16 ) );

        // Even though different registers are used for scalar in column and
        // row major case, all those registers will contain the same value.
        if ( post_ops_list_temp->scale_factor_len == 1 )
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          scl_fctr3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          scl_fctr4 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          scl_fctr5 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        }
        else
        {
          if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
          {
            scl_fctr1 =
              _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 0 * 16 ) );
            scl_fctr2 =
              _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 1 * 16 ) );
            scl_fctr3 =
              _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 2 * 16 ) );
          }
          else
          {
            scl_fctr1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 0 ) );
            scl_fctr2 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 1 ) );
            scl_fctr3 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 2 ) );
            scl_fctr4 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 3 ) );
            scl_fctr5 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 4 ) );
          }
        }

        if ( is_bf16 == TRUE )
        {
          bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;
          if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
          {
            // c[0:0-15,16-31,32-47]
            BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr2,scl_fctr3,0,8,9,10);

            // c[1:0-15,16-31,32-47]
            BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr2,scl_fctr3,1,12,13,14);

            // c[2:0-15,16-31,32-47]
            BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr2,scl_fctr3,2,16,17,18);

            // c[3:0-15,16-31,32-47]
            BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr2,scl_fctr3,3,20,21,22);

            // c[4:0-15,16-31,32-47]
            BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr2,scl_fctr3,4,24,25,26);
          }
          else
          {
            // c[0:0-15,16-31,32-47]
            BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr1,scl_fctr1,0,8,9,10);

            // c[1:0-15,16-31,32-47]
            BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
                scl_fctr2,scl_fctr2,scl_fctr2,1,12,13,14);

            // c[2:0-15,16-31,32-47]
            BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
                scl_fctr3,scl_fctr3,scl_fctr3,2,16,17,18);

            // c[3:0-15,16-31,32-47]
            BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
                scl_fctr4,scl_fctr4,scl_fctr4,3,20,21,22);

            // c[4:0-15,16-31,32-47]
            BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
                scl_fctr5,scl_fctr5,scl_fctr5,4,24,25,26);
          }
        }
        else
        {
          float* matptr = ( float* )post_ops_list_temp->op_args1;

          if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
          {
            // c[0:0-15,16-31,32-47]
            F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr2,scl_fctr3,0,8,9,10);

            // c[1:0-15,16-31,32-47]
            F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr2,scl_fctr3,1,12,13,14);

            // c[2:0-15,16-31,32-47]
            F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr2,scl_fctr3,2,16,17,18);

            // c[3:0-15,16-31,32-47]
            F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr2,scl_fctr3,3,20,21,22);

            // c[4:0-15,16-31,32-47]
            F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr2,scl_fctr3,4,24,25,26);
          }
          else
          {
            // c[0:0-15,16-31,32-47]
            F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr1,scl_fctr1,0,8,9,10);

            // c[1:0-15,16-31,32-47]
            F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
                scl_fctr2,scl_fctr2,scl_fctr2,1,12,13,14);

            // c[2:0-15,16-31,32-47]
            F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
                scl_fctr3,scl_fctr3,scl_fctr3,2,16,17,18);

            // c[3:0-15,16-31,32-47]
            F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
                scl_fctr4,scl_fctr4,scl_fctr4,3,20,21,22);

            // c[4:0-15,16-31,32-47]
            F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
                scl_fctr5,scl_fctr5,scl_fctr5,4,24,25,26);
          }
        }
        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_MATRIX_MUL_5x48F:
      {
        dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

        __m512 selector1 = _mm512_setzero_ps();
        __m512 selector2 = _mm512_setzero_ps();
        __m512 selector3 = _mm512_setzero_ps();

        __m512 scl_fctr1 = _mm512_setzero_ps();
        __m512 scl_fctr2 = _mm512_setzero_ps();
        __m512 scl_fctr3 = _mm512_setzero_ps();
        __m512 scl_fctr4 = _mm512_setzero_ps();
        __m512 scl_fctr5 = _mm512_setzero_ps();

        bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                    ( post_ops_attr.c_stor_type == BF16 ) );

        // Even though different registers are used for scalar in column and
        // row major case, all those registers will contain the same value.
        if ( post_ops_list_temp->scale_factor_len == 1 )
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          scl_fctr3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          scl_fctr4 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          scl_fctr5 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        }
        else
        {
          if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
                ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
          {
            scl_fctr1 =
              _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 0 * 16 ) );
            scl_fctr2 =
              _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 1 * 16 ) );
            scl_fctr3 =
              _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 2 * 16 ) );
          }
          else
          {
            scl_fctr1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 0 ) );
            scl_fctr2 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 1 ) );
            scl_fctr3 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 2 ) );
            scl_fctr4 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 3 ) );
            scl_fctr5 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 4 ) );
          }
        }
        if ( is_bf16 == TRUE )
        {
          bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

          if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
          {
            // c[0:0-15,16-31,32-47]
            BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr2,scl_fctr3,0,8,9,10);

            // c[1:0-15,16-31,32-47]
            BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr2,scl_fctr3,1,12,13,14);

            // c[2:0-15,16-31,32-47]
            BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr2,scl_fctr3,2,16,17,18);

            // c[3:0-15,16-31,32-47]
            BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr2,scl_fctr3,3,20,21,22);

            // c[4:0-15,16-31,32-47]
            BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr2,scl_fctr3,4,24,25,26);
          }
          else
          {

            // c[0:0-15,16-31,32-47]
            BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr1,scl_fctr1,0,8,9,10);

            // c[1:0-15,16-31,32-47]
            BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
                scl_fctr2,scl_fctr2,scl_fctr2,1,12,13,14);

            // c[2:0-15,16-31,32-47]
            BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
                scl_fctr3,scl_fctr3,scl_fctr3,2,16,17,18);

            // c[3:0-15,16-31,32-47]
            BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
                scl_fctr4,scl_fctr4,scl_fctr4,3,20,21,22);

            // c[4:0-15,16-31,32-47]
            BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
                scl_fctr5,scl_fctr5,scl_fctr5,4,24,25,26);
          }
        }
        else
        {
          float* matptr = ( float* )post_ops_list_temp->op_args1;

          if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
          {
            // c[0:0-15,16-31,32-47]
            F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr2,scl_fctr3,0,8,9,10);

            // c[1:0-15,16-31,32-47]
            F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr2,scl_fctr3,1,12,13,14);

            // c[2:0-15,16-31,32-47]
            F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr2,scl_fctr3,2,16,17,18);

            // c[3:0-15,16-31,32-47]
            F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr2,scl_fctr3,3,20,21,22);

            // c[4:0-15,16-31,32-47]
            F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr2,scl_fctr3,4,24,25,26);
          }
          else
          {
            // c[0:0-15,16-31,32-47]
            F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr1,scl_fctr1,0,8,9,10);

            // c[1:0-15,16-31,32-47]
            F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
                scl_fctr2,scl_fctr2,scl_fctr2,1,12,13,14);

            // c[2:0-15,16-31,32-47]
            F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
                scl_fctr3,scl_fctr3,scl_fctr3,2,16,17,18);

            // c[3:0-15,16-31,32-47]
            F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
                scl_fctr4,scl_fctr4,scl_fctr4,3,20,21,22);

            // c[4:0-15,16-31,32-47]
            F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
                scl_fctr5,scl_fctr5,scl_fctr5,4,24,25,26);
          }
        }
        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_SWISH_5x48F:
    {
        __m512 zmm7 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );
        __m512i ex_out;

        // c[0, 0-15]
        SWISH_F32_AVX512_DEF(zmm8, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[0, 16-31]
        SWISH_F32_AVX512_DEF(zmm9, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[0, 32-47]
        SWISH_F32_AVX512_DEF(zmm10, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 0-15]
        SWISH_F32_AVX512_DEF(zmm12, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 16-31]
        SWISH_F32_AVX512_DEF(zmm13, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 32-47]
        SWISH_F32_AVX512_DEF(zmm14, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[2, 0-15]
        SWISH_F32_AVX512_DEF(zmm16, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[2, 16-31]
        SWISH_F32_AVX512_DEF(zmm17, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[2, 32-47]
        SWISH_F32_AVX512_DEF(zmm18, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[3, 0-15]
        SWISH_F32_AVX512_DEF(zmm20, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[3, 16-31]
        SWISH_F32_AVX512_DEF(zmm21, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[3, 32-47]
        SWISH_F32_AVX512_DEF(zmm22, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[4, 0-15]
        SWISH_F32_AVX512_DEF(zmm24, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[4, 16-31]
        SWISH_F32_AVX512_DEF(zmm25, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[4, 32-47]
        SWISH_F32_AVX512_DEF(zmm26, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_5x48F:
      {
        __m512i zmm6;
        // c[0, 0-15]
        TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[0, 16-31]
        TANH_F32S_AVX512(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[0, 32-47]
        TANH_F32S_AVX512(zmm10, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 0-15]
        TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 16-31]
        TANH_F32S_AVX512(zmm13, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 32-47]
        TANH_F32S_AVX512(zmm14, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[2, 0-15]
        TANH_F32S_AVX512(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[2, 16-31]
        TANH_F32S_AVX512(zmm17, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[2, 32-47]
        TANH_F32S_AVX512(zmm18, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[3, 0-15]
        TANH_F32S_AVX512(zmm20, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[3, 16-31]
        TANH_F32S_AVX512(zmm21, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[3, 32-47]
        TANH_F32S_AVX512(zmm22, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[4, 0-15]
        TANH_F32S_AVX512(zmm24, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[4, 16-31]
        TANH_F32S_AVX512(zmm25, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[4, 32-47]
        TANH_F32S_AVX512(zmm26, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_SIGMOID_5x48F:
      {
          __m512i ex_out;

          // c[0, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[0, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[0, 32-47]
          SIGMOID_F32_AVX512_DEF(zmm10, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm13, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 32-47]
          SIGMOID_F32_AVX512_DEF(zmm14, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm17, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 32-47]
          SIGMOID_F32_AVX512_DEF(zmm18, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[3, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm20, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[3, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm21, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[3, 32-47]
          SIGMOID_F32_AVX512_DEF(zmm22, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[4, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm24, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[4, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm25, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[4, 32-47]
          SIGMOID_F32_AVX512_DEF(zmm26, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_5x48F_DISABLE:
      ;

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_last_k == TRUE ) )
      {
        uint32_t tlsb, rounded, temp[16] = {0};
        int i;
        bfloat16* dest;

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm8, 0, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm9, 0, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm10, 0, 2);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm12, 1, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm13, 1, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm14, 1, 2);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm16, 2, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm17, 2, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm18, 2, 2);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm20, 3, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm21, 3, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm22, 3, 2);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm24, 4, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm25, 4, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm26, 4, 2);
      }
      else
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
      }
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
              &&POST_OPS_CLIP_4x48F,
              &&POST_OPS_DOWNSCALE_4x48F,
              &&POST_OPS_MATRIX_ADD_4x48F,
              &&POST_OPS_SWISH_4x48F,
              &&POST_OPS_MATRIX_MUL_4x48F,
              &&POST_OPS_TANH_4x48F,
              &&POST_OPS_SIGMOID_4x48F
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
      zmm3 = _mm512_set1_ps(beta);

      //load c and beta, convert to f32
      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_first_k == TRUE ) )
      {
        //c[0, 0-15]
        BF16_F32_BETA_OP(zmm8, m, 0, 0, zmm0,zmm3);
        //c[0, 16-31]
        BF16_F32_BETA_OP(zmm9, m, 0, 1, zmm1,zmm3);
        //c[0, 32-47]
        BF16_F32_BETA_OP(zmm10, m, 0, 2, zmm0,zmm3);
        //c[1, 0-15]
        BF16_F32_BETA_OP(zmm12, m, 1, 0, zmm0,zmm3);
        //c[1, 16-31]
        BF16_F32_BETA_OP(zmm13, m, 1, 1, zmm1,zmm3);
        //c[1, 32-47]
        BF16_F32_BETA_OP(zmm14, m, 1, 2, zmm0,zmm3);
        //c[2, 0-15]
        BF16_F32_BETA_OP(zmm16, m, 2, 0, zmm0,zmm3);
        //c[2,16-31]
        BF16_F32_BETA_OP(zmm17, m, 2, 1, zmm1,zmm3);
        //c[2,32-47]
        BF16_F32_BETA_OP(zmm18, m, 2, 2, zmm0,zmm3);
        //c[3, 0-15]
        BF16_F32_BETA_OP(zmm20, m, 3, 0, zmm0,zmm3);
        //c[3,16-31]
        BF16_F32_BETA_OP(zmm21, m, 3, 1, zmm1,zmm3);
        //c[3,32-47]
        BF16_F32_BETA_OP(zmm22, m, 3, 2, zmm0,zmm3);
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);

        zmm0 = _mm512_loadu_ps(_cbuf + 32);
        zmm10 = _mm512_fmadd_ps(zmm0, zmm3, zmm10);
        _cbuf += rs_c;

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);

        zmm0 = _mm512_loadu_ps(_cbuf + 32);
        zmm14 = _mm512_fmadd_ps(zmm0, zmm3, zmm14);
        _cbuf += rs_c;

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm16 = _mm512_fmadd_ps(zmm0, zmm3, zmm16);
        zmm17 = _mm512_fmadd_ps(zmm1, zmm3, zmm17);

        zmm0 = _mm512_loadu_ps(_cbuf + 32);
        zmm18 = _mm512_fmadd_ps(zmm0, zmm3, zmm18);
        _cbuf += rs_c;

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm20 = _mm512_fmadd_ps(zmm0, zmm3, zmm20);
        zmm21 = _mm512_fmadd_ps(zmm1, zmm3, zmm21);

        zmm0 = _mm512_loadu_ps(_cbuf + 32);
        zmm22 = _mm512_fmadd_ps(zmm0, zmm3, zmm22);
      }
    }

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_4x48F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_BIAS_LOAD(zmm1, bias_mask, 0)
          BF16_F32_BIAS_LOAD(zmm2, bias_mask, 1)
          BF16_F32_BIAS_LOAD(zmm3, bias_mask, 2)
        }
        else
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
        }
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
        if ( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );

          BF16_F32_BIAS_BCAST(zmm1, bias_mask, 0)
          BF16_F32_BIAS_BCAST(zmm2, bias_mask, 1)
          BF16_F32_BIAS_BCAST(zmm3, bias_mask, 2)
          BF16_F32_BIAS_BCAST(zmm4, bias_mask, 3)
        }
        else
        {
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
        }
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
POST_OPS_DOWNSCALE_4x48F:
    {
      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();
      __m512 selector3 = _mm512_setzero_ps();
      __m512 selector4 = _mm512_setzero_ps();

      __m512 zero_point0 = _mm512_setzero_ps();
      __m512 zero_point1 = _mm512_setzero_ps();
      __m512 zero_point2 = _mm512_setzero_ps();
      __m512 zero_point3 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                    ( ( post_ops_list_temp->stor_type == NONE ) &&
                      ( post_ops_attr.c_stor_type == BF16 ) );

      // Need to account for row vs column major swaps. For scalars
      // scale and zero point, no implications.
      // Even though different registers are used for scalar in column
      // and row major downscale path, all those registers will contain
      // the same value.

      if( post_ops_list_temp->scale_factor_len == 1 )
      {
        selector1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector2 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector3 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector4 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if ( is_bf16 == TRUE )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_ZP_BCST(zero_point0,0, zp_mask)
          BF16_F32_ZP_BCST(zero_point1,1, zp_mask)
          BF16_F32_ZP_BCST(zero_point2,2, zp_mask)
          BF16_F32_ZP_BCST(zero_point3,3, zp_mask)
        }
        else
        {
          zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point3 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          selector2 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 1 * 16 ) );
          selector3 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 2 * 16 ) );
        }
        if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 load_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_LOAD(zero_point0,load_mask, 0)
            BF16_F32_ZP_LOAD(zero_point1,load_mask, 1)
            BF16_F32_ZP_LOAD(zero_point2,load_mask, 2)
          }
          else
          {
            zero_point0 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 0 * 16 ) );
            zero_point1 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 1 * 16 ) );
            zero_point2 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 2 * 16 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[0, 16-31]
        F32_SCL_MULRND(zmm9, selector2, zero_point1);

        //c[0, 32-47]
        F32_SCL_MULRND(zmm10, selector3, zero_point2);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector1, zero_point0);

        //c[1, 16-31]
        F32_SCL_MULRND(zmm13, selector2, zero_point1);

        //c[1, 32-47]
        F32_SCL_MULRND(zmm14, selector3, zero_point2);

        //c[2, 0-15]
        F32_SCL_MULRND(zmm16, selector1, zero_point0);

        //c[2, 16-31]
        F32_SCL_MULRND(zmm17, selector2, zero_point1);

        //c[2, 32-47]
        F32_SCL_MULRND(zmm18, selector3, zero_point2);

        //c[3, 0-15]
        F32_SCL_MULRND(zmm20, selector1, zero_point0);

        //c[3, 16-31]
        F32_SCL_MULRND(zmm21, selector2, zero_point1);

        //c[3, 32-47]
        F32_SCL_MULRND(zmm22, selector3, zero_point2);
      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the scale as well as zp array will
        // be accessed by the ic index, and each scale/zp element
        // corresponds to an entire row of the transposed output array,
        // instead of an entire column.
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 0 ) );
          selector2 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 1 ) );
          selector3 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 2 ) );
          selector4 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 3 ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_COL_BCST(zero_point0, 0, zp_mask)
            BF16_F32_ZP_COL_BCST(zero_point1, 1, zp_mask)
            BF16_F32_ZP_COL_BCST(zero_point2, 2, zp_mask)
            BF16_F32_ZP_COL_BCST(zero_point3, 3, zp_mask)
          }
          else
          {
            zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 1 ) );
            zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 2 ) );
            zero_point3 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 3 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[0, 16-31]
        F32_SCL_MULRND(zmm9, selector1, zero_point0);

        //c[0, 32-47]
        F32_SCL_MULRND(zmm10, selector1, zero_point0);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector2, zero_point1);

        //c[1, 16-31]
        F32_SCL_MULRND(zmm13, selector2, zero_point1);

        //c[1, 32-47]
        F32_SCL_MULRND(zmm14, selector2, zero_point1);

        //c[2, 0-15]
        F32_SCL_MULRND(zmm16, selector3, zero_point2);

        //c[2, 16-31]
        F32_SCL_MULRND(zmm17, selector3, zero_point2);

        //c[2, 32-47]
        F32_SCL_MULRND(zmm18, selector3, zero_point2);

        //c[3, 0-15]
        F32_SCL_MULRND(zmm20, selector4, zero_point3);

        //c[3, 16-31]
        F32_SCL_MULRND(zmm21, selector4, zero_point3);

        //c[3, 32-47]
        F32_SCL_MULRND(zmm22, selector4, zero_point3);
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_4x48F:
      {
        dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

        __m512 selector1 = _mm512_setzero_ps();
        __m512 selector2 = _mm512_setzero_ps();
        __m512 selector3 = _mm512_setzero_ps();

        __m512 scl_fctr1 = _mm512_setzero_ps();
        __m512 scl_fctr2 = _mm512_setzero_ps();
        __m512 scl_fctr3 = _mm512_setzero_ps();
        __m512 scl_fctr4 = _mm512_setzero_ps();

        bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                ( post_ops_attr.c_stor_type == BF16 ) );

        // Even though different registers are used for scalar in column and
        // row major case, all those registers will contain the same value.
        if ( post_ops_list_temp->scale_factor_len == 1 )
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          scl_fctr3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          scl_fctr4 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        }
        else
        {
          if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
          {
            scl_fctr1 =
              _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 0 * 16 ) );
            scl_fctr2 =
              _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 1 * 16 ) );
            scl_fctr3 =
              _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 2 * 16 ) );
          }
          else
          {
            scl_fctr1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 0 ) );
            scl_fctr2 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 1 ) );
            scl_fctr3 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 2 ) );
            scl_fctr4 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 3 ) );
          }
        }
        if ( is_bf16 == TRUE )
        {
          bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

          if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
          {
            // c[0:0-15,16-31,32-47]
            BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr2,scl_fctr3,0,8,9,10);

            // c[1:0-15,16-31,32-47]
            BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr2,scl_fctr3,1,12,13,14);

            // c[2:0-15,16-31,32-47]
            BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr2,scl_fctr3,2,16,17,18);

            // c[3:0-15,16-31,32-47]
            BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr2,scl_fctr3,3,20,21,22);
          }
          else
          {
            // c[0:0-15,16-31,32-47]
            BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr1,scl_fctr1,0,8,9,10);

            // c[1:0-15,16-31,32-47]
            BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
                scl_fctr2,scl_fctr2,scl_fctr2,1,12,13,14);

            // c[2:0-15,16-31,32-47]
            BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
                scl_fctr3,scl_fctr3,scl_fctr3,2,16,17,18);

            // c[3:0-15,16-31,32-47]
            BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
                scl_fctr4,scl_fctr4,scl_fctr4,3,20,21,22);
          }
        }
        else
        {
          float* matptr = ( float* )post_ops_list_temp->op_args1;

          if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
          {
            // c[0:0-15,16-31,32-47]
            F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr2,scl_fctr3,0,8,9,10);

            // c[1:0-15,16-31,32-47]
            F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr2,scl_fctr3,1,12,13,14);

            // c[2:0-15,16-31,32-47]
            F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr2,scl_fctr3,2,16,17,18);

            // c[3:0-15,16-31,32-47]
            F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr2,scl_fctr3,3,20,21,22);
          }
          else
          {
            // c[0:0-15,16-31,32-47]
            F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr1,scl_fctr1,0,8,9,10);

            // c[1:0-15,16-31,32-47]
            F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
                scl_fctr2,scl_fctr2,scl_fctr2,1,12,13,14);

            // c[2:0-15,16-31,32-47]
            F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
                scl_fctr3,scl_fctr3,scl_fctr3,2,16,17,18);

            // c[3:0-15,16-31,32-47]
            F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
                scl_fctr4,scl_fctr4,scl_fctr4,3,20,21,22);
          }
        }
        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_MATRIX_MUL_4x48F:
      {
        dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

        __m512 selector1 = _mm512_setzero_ps();
        __m512 selector2 = _mm512_setzero_ps();
        __m512 selector3 = _mm512_setzero_ps();

        __m512 scl_fctr1 = _mm512_setzero_ps();
        __m512 scl_fctr2 = _mm512_setzero_ps();
        __m512 scl_fctr3 = _mm512_setzero_ps();
        __m512 scl_fctr4 = _mm512_setzero_ps();

        bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                    ( post_ops_attr.c_stor_type == BF16 ) );

        // Even though different registers are used for scalar in column and
        // row major case, all those registers will contain the same value.
        if ( post_ops_list_temp->scale_factor_len == 1 )
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          scl_fctr3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          scl_fctr4 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        }
        else
        {
          if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
                ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
          {
            scl_fctr1 =
              _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 0 * 16 ) );
            scl_fctr2 =
              _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 1 * 16 ) );
            scl_fctr3 =
              _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 2 * 16 ) );
          }
          else
          {
            scl_fctr1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 0 ) );
            scl_fctr2 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 1 ) );
            scl_fctr3 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 2 ) );
            scl_fctr4 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 3 ) );
          }
        }
        if ( is_bf16 == TRUE )
        {
          bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

          if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
          {
            // c[0:0-15,16-31,32-47]
            BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr2,scl_fctr3,0,8,9,10);

            // c[1:0-15,16-31,32-47]
            BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr2,scl_fctr3,1,12,13,14);

            // c[2:0-15,16-31,32-47]
            BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr2,scl_fctr3,2,16,17,18);

            // c[3:0-15,16-31,32-47]
            BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr2,scl_fctr3,3,20,21,22);

          }
          else
          {
            // c[0:0-15,16-31,32-47]
            BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr1,scl_fctr1,0,8,9,10);

            // c[1:0-15,16-31,32-47]
            BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
                scl_fctr2,scl_fctr2,scl_fctr2,1,12,13,14);

            // c[2:0-15,16-31,32-47]
            BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
                scl_fctr3,scl_fctr3,scl_fctr3,2,16,17,18);

            // c[3:0-15,16-31,32-47]
            BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
                scl_fctr4,scl_fctr4,scl_fctr4,3,20,21,22);
          }
        }
        else
        {
          float* matptr = ( float* )post_ops_list_temp->op_args1;

          if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
          {
            // c[0:0-15,16-31,32-47]
            F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr2,scl_fctr3,0,8,9,10);

            // c[1:0-15,16-31,32-47]
            F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr2,scl_fctr3,1,12,13,14);

            // c[2:0-15,16-31,32-47]
            F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr2,scl_fctr3,2,16,17,18);

            // c[3:0-15,16-31,32-47]
            F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr2,scl_fctr3,3,20,21,22);

          }
          else
          {
            // c[0:0-15,16-31,32-47]
            F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
                scl_fctr1,scl_fctr1,scl_fctr1,0,8,9,10);

            // c[1:0-15,16-31,32-47]
            F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
                scl_fctr2,scl_fctr2,scl_fctr2,1,12,13,14);

            // c[2:0-15,16-31,32-47]
            F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
                scl_fctr3,scl_fctr3,scl_fctr3,2,16,17,18);

            // c[3:0-15,16-31,32-47]
            F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
                scl_fctr4,scl_fctr4,scl_fctr4,3,20,21,22);
          }
        }
        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_SWISH_4x48F:
    {
        __m512 zmm7 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );
        __m512i ex_out;

        // c[0, 0-15]
        SWISH_F32_AVX512_DEF(zmm8, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[0, 16-31]
        SWISH_F32_AVX512_DEF(zmm9, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[0, 32-47]
        SWISH_F32_AVX512_DEF(zmm10, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 0-15]
        SWISH_F32_AVX512_DEF(zmm12, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 16-31]
        SWISH_F32_AVX512_DEF(zmm13, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 32-47]
        SWISH_F32_AVX512_DEF(zmm14, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[2, 0-15]
        SWISH_F32_AVX512_DEF(zmm16, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[2, 16-31]
        SWISH_F32_AVX512_DEF(zmm17, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[2, 32-47]
        SWISH_F32_AVX512_DEF(zmm18, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[3, 0-15]
        SWISH_F32_AVX512_DEF(zmm20, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[3, 16-31]
        SWISH_F32_AVX512_DEF(zmm21, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[3, 32-47]
        SWISH_F32_AVX512_DEF(zmm22, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_4x48F:
      {
        __m512i zmm6;
        // c[0, 0-15]
        TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[0, 16-31]
        TANH_F32S_AVX512(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[0, 32-47]
        TANH_F32S_AVX512(zmm10, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 0-15]
        TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 16-31]
        TANH_F32S_AVX512(zmm13, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 32-47]
        TANH_F32S_AVX512(zmm14, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[2, 0-15]
        TANH_F32S_AVX512(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[2, 16-31]
        TANH_F32S_AVX512(zmm17, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[2, 32-47]
        TANH_F32S_AVX512(zmm18, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[3, 0-15]
        TANH_F32S_AVX512(zmm20, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[3, 16-31]
        TANH_F32S_AVX512(zmm21, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[3, 32-47]
        TANH_F32S_AVX512(zmm22, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_SIGMOID_4x48F:
      {
          __m512i ex_out;

          // c[0, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[0, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[0, 32-47]
          SIGMOID_F32_AVX512_DEF(zmm10, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm13, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 32-47]
          SIGMOID_F32_AVX512_DEF(zmm14, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm17, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 32-47]
          SIGMOID_F32_AVX512_DEF(zmm18, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[3, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm20, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[3, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm21, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[3, 32-47]
          SIGMOID_F32_AVX512_DEF(zmm22, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_4x48F_DISABLE:
      ;

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_last_k == TRUE ) )
      {
        uint32_t tlsb, rounded, temp[16] = {0};
        int i;
        bfloat16* dest;

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm8, 0, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm9, 0, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm10, 0, 2);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm12, 1, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm13, 1, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm14, 1, 2);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm16, 2, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm17, 2, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm18, 2, 2);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm20, 3, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm21, 3, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm22, 3, 2);
      }
      else
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
      }
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
              &&POST_OPS_CLIP_3x48F,
              &&POST_OPS_DOWNSCALE_3x48F,
              &&POST_OPS_MATRIX_ADD_3x48F,
              &&POST_OPS_SWISH_3x48F,
              &&POST_OPS_MATRIX_MUL_3x48F,
              &&POST_OPS_TANH_3x48F,
              &&POST_OPS_SIGMOID_3x48F
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
      zmm3 = _mm512_set1_ps(beta);

      //load c and beta, convert to f32
      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_first_k == TRUE ) )
      {
        //c[0, 0-15]
        BF16_F32_BETA_OP(zmm8, m, 0, 0, zmm0,zmm3);
        //c[0, 16-31]
        BF16_F32_BETA_OP(zmm9, m, 0, 1, zmm1,zmm3);
        //c[0, 32-47]
        BF16_F32_BETA_OP(zmm10, m, 0, 2, zmm0,zmm3);
        //c[1, 0-15]
        BF16_F32_BETA_OP(zmm12, m, 1, 0, zmm0,zmm3);
        //c[1, 16-31]
        BF16_F32_BETA_OP(zmm13, m, 1, 1, zmm1,zmm3);
        //c[1, 32-47]
        BF16_F32_BETA_OP(zmm14, m, 1, 2, zmm0,zmm3);
        //c[2, 0-15]
        BF16_F32_BETA_OP(zmm16, m, 2, 0, zmm0,zmm3);
        //c[2,16-31]
        BF16_F32_BETA_OP(zmm17, m, 2, 1, zmm1,zmm3);
        //c[2,32-47]
        BF16_F32_BETA_OP(zmm18, m, 2, 2, zmm0,zmm3);
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);

        zmm0 = _mm512_loadu_ps(_cbuf + 32);
        zmm10 = _mm512_fmadd_ps(zmm0, zmm3, zmm10);
        _cbuf += rs_c;

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);

        zmm0 = _mm512_loadu_ps(_cbuf + 32);
        zmm14 = _mm512_fmadd_ps(zmm0, zmm3, zmm14);
        _cbuf += rs_c;

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm16 = _mm512_fmadd_ps(zmm0, zmm3, zmm16);
        zmm17 = _mm512_fmadd_ps(zmm1, zmm3, zmm17);

        zmm0 = _mm512_loadu_ps(_cbuf + 32);
        zmm18 = _mm512_fmadd_ps(zmm0, zmm3, zmm18);
      }
    }

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_3x48F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_BIAS_LOAD(zmm1, bias_mask, 0)
          BF16_F32_BIAS_LOAD(zmm2, bias_mask, 1)
          BF16_F32_BIAS_LOAD(zmm3, bias_mask, 2)
        }
        else
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
        }
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
        if ( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );

          BF16_F32_BIAS_BCAST(zmm1, bias_mask, 0)
          BF16_F32_BIAS_BCAST(zmm2, bias_mask, 1)
          BF16_F32_BIAS_BCAST(zmm3, bias_mask, 2)
        }
        else
        {
          zmm1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 0 ) );
          zmm2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 1 ) );
          zmm3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 2 ) );
        }
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
POST_OPS_DOWNSCALE_3x48F:
    {
      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();
      __m512 selector3 = _mm512_setzero_ps();

      __m512 zero_point0 = _mm512_setzero_ps();
      __m512 zero_point1 = _mm512_setzero_ps();
      __m512 zero_point2 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                    ( ( post_ops_list_temp->stor_type == NONE ) &&
                      ( post_ops_attr.c_stor_type == BF16 ) );

      // Need to account for row vs column major swaps. For scalars
      // scale and zero point, no implications.
      // Even though different registers are used for scalar in column
      // and row major downscale path, all those registers will contain
      // the same value.

      if( post_ops_list_temp->scale_factor_len == 1 )
      {
        selector1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector2 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector3 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if ( is_bf16 == TRUE )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_ZP_BCST(zero_point0,0, zp_mask)
          BF16_F32_ZP_BCST(zero_point1,1, zp_mask)
          BF16_F32_ZP_BCST(zero_point2,2, zp_mask)
        }
        else
        {
          zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          selector2 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 1 * 16 ) );
          selector3 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 2 * 16 ) );
        }
        if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 load_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_LOAD(zero_point0,load_mask, 0)
            BF16_F32_ZP_LOAD(zero_point1,load_mask, 1)
            BF16_F32_ZP_LOAD(zero_point2,load_mask, 2)
          }
          else
          {
            zero_point0 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 0 * 16 ) );
            zero_point1 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 1 * 16 ) );
            zero_point2 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 2 * 16 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[0, 16-31]
        F32_SCL_MULRND(zmm9, selector2, zero_point1);

        //c[0, 32-47]
        F32_SCL_MULRND(zmm10, selector3, zero_point2);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector1, zero_point0);

        //c[1, 16-31]
        F32_SCL_MULRND(zmm13, selector2, zero_point1);

        //c[1, 32-47]
        F32_SCL_MULRND(zmm14, selector3, zero_point2);

        //c[2, 0-15]
        F32_SCL_MULRND(zmm16, selector1, zero_point0);

        //c[2, 16-31]
        F32_SCL_MULRND(zmm17, selector2, zero_point1);

        //c[2, 32-47]
        F32_SCL_MULRND(zmm18, selector3, zero_point2);
      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the scale as well as zp array will
        // be accessed by the ic index, and each scale/zp element
        // corresponds to an entire row of the transposed output array,
        // instead of an entire column.
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 0 ) );
          selector2 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 1 ) );
          selector3 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 2 ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_COL_BCST(zero_point0, 0, zp_mask)
            BF16_F32_ZP_COL_BCST(zero_point1, 1, zp_mask)
            BF16_F32_ZP_COL_BCST(zero_point2, 2, zp_mask)
          }
          else
          {
            zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 1 ) );
            zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 2 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[0, 16-31]
        F32_SCL_MULRND(zmm9, selector1, zero_point0);

        //c[0, 32-47]
        F32_SCL_MULRND(zmm10, selector1, zero_point0);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector2, zero_point1);

        //c[1, 16-31]
        F32_SCL_MULRND(zmm13, selector2, zero_point1);

        //c[1, 32-47]
        F32_SCL_MULRND(zmm14, selector2, zero_point1);

        //c[2, 0-15]
        F32_SCL_MULRND(zmm16, selector3, zero_point2);

        //c[2, 16-31]
        F32_SCL_MULRND(zmm17, selector3, zero_point2);

        //c[2, 32-47]
        F32_SCL_MULRND(zmm18, selector3, zero_point2);
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_3x48F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();
      __m512 selector3 = _mm512_setzero_ps();

      __m512 scl_fctr1 = _mm512_setzero_ps();
      __m512 scl_fctr2 = _mm512_setzero_ps();
      __m512 scl_fctr3 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                ( post_ops_attr.c_stor_type == BF16 ) );

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          scl_fctr2 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 1 * 16 ) );
          scl_fctr3 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 2 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
        }
      }
      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31,32-47]
          BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
              scl_fctr1,scl_fctr2,scl_fctr3,0,8,9,10);

          // c[1:0-15,16-31,32-47]
          BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
              scl_fctr1,scl_fctr2,scl_fctr3,1,12,13,14);

          // c[2:0-15,16-31,32-47]
          BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
              scl_fctr1,scl_fctr2,scl_fctr3,2,16,17,18);
        }
        else
        {
          // c[0:0-15,16-31,32-47]
          BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
              scl_fctr1,scl_fctr1,scl_fctr1,0,8,9,10);

          // c[1:0-15,16-31,32-47]
          BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
              scl_fctr2,scl_fctr2,scl_fctr2,1,12,13,14);

          // c[2:0-15,16-31,32-47]
          BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
              scl_fctr3,scl_fctr3,scl_fctr3,2,16,17,18);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31,32-47]
          F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
              scl_fctr1,scl_fctr2,scl_fctr3,0,8,9,10);

          // c[1:0-15,16-31,32-47]
          F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
              scl_fctr1,scl_fctr2,scl_fctr3,1,12,13,14);

          // c[2:0-15,16-31,32-47]
          F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
              scl_fctr1,scl_fctr2,scl_fctr3,2,16,17,18);
        }
        else
        {
          // c[0:0-15,16-31,32-47]
          F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
              scl_fctr1,scl_fctr1,scl_fctr1,0,8,9,10);

          // c[1:0-15,16-31,32-47]
          F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
              scl_fctr2,scl_fctr2,scl_fctr2,1,12,13,14);

          // c[2:0-15,16-31,32-47]
          F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
              scl_fctr3,scl_fctr3,scl_fctr3,2,16,17,18);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_3x48F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();
      __m512 selector3 = _mm512_setzero_ps();

      __m512 scl_fctr1 = _mm512_setzero_ps();
      __m512 scl_fctr2 = _mm512_setzero_ps();
      __m512 scl_fctr3 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                    ( post_ops_attr.c_stor_type == BF16 ) );

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
              ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          scl_fctr2 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 1 * 16 ) );
          scl_fctr3 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 2 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
        }
      }

      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31,32-47]
          BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
              scl_fctr1,scl_fctr2,scl_fctr3,0,8,9,10);

          // c[1:0-15,16-31,32-47]
          BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
              scl_fctr1,scl_fctr2,scl_fctr3,1,12,13,14);

          // c[2:0-15,16-31,32-47]
          BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
              scl_fctr1,scl_fctr2,scl_fctr3,2,16,17,18);
        }
        else
        {
          // c[0:0-15,16-31,32-47]
          BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
              scl_fctr1,scl_fctr1,scl_fctr1,0,8,9,10);

          // c[1:0-15,16-31,32-47]
          BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
              scl_fctr2,scl_fctr2,scl_fctr2,1,12,13,14);

          // c[2:0-15,16-31,32-47]
          BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
              scl_fctr3,scl_fctr3,scl_fctr3,2,16,17,18);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31,32-47]
          F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
              scl_fctr1,scl_fctr2,scl_fctr3,0,8,9,10);

          // c[1:0-15,16-31,32-47]
          F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
              scl_fctr1,scl_fctr2,scl_fctr3,1,12,13,14);

          // c[2:0-15,16-31,32-47]
          F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
              scl_fctr1,scl_fctr2,scl_fctr3,2,16,17,18);
        }
        else
        {
          // c[0:0-15,16-31,32-47]
          F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
              scl_fctr1,scl_fctr1,scl_fctr1,0,8,9,10);

          // c[1:0-15,16-31,32-47]
          F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
              scl_fctr2,scl_fctr2,scl_fctr2,1,12,13,14);

          // c[2:0-15,16-31,32-47]
          F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
              scl_fctr3,scl_fctr3,scl_fctr3,2,16,17,18);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_3x48F:
    {
        __m512 zmm7 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );
        __m512i ex_out;

        // c[0, 0-15]
        SWISH_F32_AVX512_DEF(zmm8, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[0, 16-31]
        SWISH_F32_AVX512_DEF(zmm9, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[0, 32-47]
        SWISH_F32_AVX512_DEF(zmm10, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 0-15]
        SWISH_F32_AVX512_DEF(zmm12, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 16-31]
        SWISH_F32_AVX512_DEF(zmm13, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 32-47]
        SWISH_F32_AVX512_DEF(zmm14, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[2, 0-15]
        SWISH_F32_AVX512_DEF(zmm16, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[2, 16-31]
        SWISH_F32_AVX512_DEF(zmm17, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[2, 32-47]
        SWISH_F32_AVX512_DEF(zmm18, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_3x48F:
      {
        __m512i zmm6;
        // c[0, 0-15]
        TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[0, 16-31]
        TANH_F32S_AVX512(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[0, 32-47]
        TANH_F32S_AVX512(zmm10, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 0-15]
        TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 16-31]
        TANH_F32S_AVX512(zmm13, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 32-47]
        TANH_F32S_AVX512(zmm14, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[2, 0-15]
        TANH_F32S_AVX512(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[2, 16-31]
        TANH_F32S_AVX512(zmm17, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[2, 32-47]
        TANH_F32S_AVX512(zmm18, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_SIGMOID_3x48F:
      {
          __m512i ex_out;

          // c[0, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[0, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[0, 32-47]
          SIGMOID_F32_AVX512_DEF(zmm10, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm13, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 32-47]
          SIGMOID_F32_AVX512_DEF(zmm14, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm17, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 32-47]
          SIGMOID_F32_AVX512_DEF(zmm18, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_3x48F_DISABLE:
      ;

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_last_k == TRUE ) )
      {
        uint32_t tlsb, rounded, temp[16] = {0};
        int i;
        bfloat16* dest;

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm8, 0, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm9, 0, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm10, 0, 2);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm12, 1, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm13, 1, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm14, 1, 2);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm16, 2, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm17, 2, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm18, 2, 2);
      }
      else
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
      }
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
              &&POST_OPS_CLIP_2x48F,
              &&POST_OPS_DOWNSCALE_2x48F,
              &&POST_OPS_MATRIX_ADD_2x48F,
              &&POST_OPS_SWISH_2x48F,
              &&POST_OPS_MATRIX_MUL_2x48F,
              &&POST_OPS_TANH_2x48F,
              &&POST_OPS_SIGMOID_2x48F
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
      zmm3 = _mm512_set1_ps(beta);

      //load c and beta, convert to f32
      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_first_k == TRUE ) )
      {
        //c[0, 0-15]
        BF16_F32_BETA_OP(zmm8, m, 0, 0, zmm0,zmm3);
        //c[0, 16-31]
        BF16_F32_BETA_OP(zmm9, m, 0, 1, zmm1,zmm3);
        //c[0, 32-47]
        BF16_F32_BETA_OP(zmm10, m, 0, 2, zmm0,zmm3);
        //c[1, 0-15]
        BF16_F32_BETA_OP(zmm12, m, 1, 0, zmm0,zmm3);
        //c[1, 16-31]
        BF16_F32_BETA_OP(zmm13, m, 1, 1, zmm1,zmm3);
        //c[1, 32-47]
        BF16_F32_BETA_OP(zmm14, m, 1, 2, zmm0,zmm3);
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);

        zmm0 = _mm512_loadu_ps(_cbuf + 32);
        zmm10 = _mm512_fmadd_ps(zmm0, zmm3, zmm10);
        _cbuf += rs_c;

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);

        zmm0 = _mm512_loadu_ps(_cbuf + 32);
        zmm14 = _mm512_fmadd_ps(zmm0, zmm3, zmm14);
      }
    }

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_2x48F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_BIAS_LOAD(zmm1, bias_mask, 0)
          BF16_F32_BIAS_LOAD(zmm2, bias_mask, 1)
          BF16_F32_BIAS_LOAD(zmm3, bias_mask, 2)
        }
        else
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
        }
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
        if ( post_ops_list_temp->stor_type == BF16 )
          {
            __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );

            BF16_F32_BIAS_BCAST(zmm1, bias_mask, 0)
            BF16_F32_BIAS_BCAST(zmm2, bias_mask, 1)
          }
          else
          {
            zmm1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                post_ops_attr.post_op_c_i + 0 ) );
            zmm2 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                post_ops_attr.post_op_c_i + 1 ) );
          }
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
POST_OPS_DOWNSCALE_2x48F:
    {
      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();
      __m512 selector3 = _mm512_setzero_ps();

      __m512 zero_point0 = _mm512_setzero_ps();
      __m512 zero_point1 = _mm512_setzero_ps();
      __m512 zero_point2 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                    ( ( post_ops_list_temp->stor_type == NONE ) &&
                      ( post_ops_attr.c_stor_type == BF16 ) );

      // Need to account for row vs column major swaps. For scalars
      // scale and zero point, no implications.
      // Even though different registers are used for scalar in column
      // and row major downscale path, all those registers will contain
      // the same value.

      if( post_ops_list_temp->scale_factor_len == 1 )
      {
        selector1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector2 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector3 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
     }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if ( is_bf16 == TRUE )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_ZP_BCST(zero_point0,0, zp_mask)
          BF16_F32_ZP_BCST(zero_point1,1, zp_mask)
          BF16_F32_ZP_BCST(zero_point2,2, zp_mask)
        }
        else
        {
          zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          selector2 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 1 * 16 ) );
          selector3 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 2 * 16 ) );
        }
        if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 load_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_LOAD(zero_point0,load_mask, 0)
            BF16_F32_ZP_LOAD(zero_point1,load_mask, 1)
            BF16_F32_ZP_LOAD(zero_point2,load_mask, 2)
          }
          else
          {
            zero_point0 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 0 * 16 ) );
            zero_point1 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 1 * 16 ) );
            zero_point2 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 2 * 16 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[0, 16-31]
        F32_SCL_MULRND(zmm9, selector2, zero_point1);

        //c[0, 32-47]
        F32_SCL_MULRND(zmm10, selector3, zero_point2);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector1, zero_point0);

        //c[1, 16-31]
        F32_SCL_MULRND(zmm13, selector2, zero_point1);

        //c[1, 32-47]
        F32_SCL_MULRND(zmm14, selector3, zero_point2);
      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the scale as well as zp array will
        // be accessed by the ic index, and each scale/zp element
        // corresponds to an entire row of the transposed output array,
        // instead of an entire column.
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 0 ) );
          selector2 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 1 ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_COL_BCST(zero_point0, 0, zp_mask)
            BF16_F32_ZP_COL_BCST(zero_point1, 1, zp_mask)
          }
          else
          {
            zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 1 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[0, 16-31]
        F32_SCL_MULRND(zmm9, selector1, zero_point0);

        //c[0, 32-47]
        F32_SCL_MULRND(zmm10, selector1, zero_point0);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector2, zero_point1);

        //c[1, 16-31]
        F32_SCL_MULRND(zmm13, selector2, zero_point1);

        //c[1, 32-47]
        F32_SCL_MULRND(zmm14, selector2, zero_point1);
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_2x48F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();
      __m512 selector3 = _mm512_setzero_ps();

      __m512 scl_fctr1 = _mm512_setzero_ps();
      __m512 scl_fctr2 = _mm512_setzero_ps();
      __m512 scl_fctr3 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                ( post_ops_attr.c_stor_type == BF16 ) );

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          scl_fctr2 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 1 * 16 ) );
          scl_fctr3 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 2 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
        }
      }

      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31,32-47]
          BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
              scl_fctr1,scl_fctr2,scl_fctr3,0,8,9,10);

          // c[1:0-15,16-31,32-47]
          BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
              scl_fctr1,scl_fctr2,scl_fctr3,1,12,13,14);
        }
        else
        {
          // c[0:0-15,16-31,32-47]
          BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
              scl_fctr1,scl_fctr1,scl_fctr1,0,8,9,10);

          // c[1:0-15,16-31,32-47]
          BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
              scl_fctr2,scl_fctr2,scl_fctr2,1,12,13,14);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31,32-47]
          F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
              scl_fctr1,scl_fctr2,scl_fctr3,0,8,9,10);

          // c[1:0-15,16-31,32-47]
          F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
              scl_fctr1,scl_fctr2,scl_fctr3,1,12,13,14);
        }
        else
        {
          // c[0:0-15,16-31,32-47]
          F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
              scl_fctr1,scl_fctr1,scl_fctr1,0,8,9,10);

          // c[1:0-15,16-31,32-47]
          F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
              scl_fctr2,scl_fctr2,scl_fctr2,1,12,13,14);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_2x48F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();
      __m512 selector3 = _mm512_setzero_ps();

      __m512 scl_fctr1 = _mm512_setzero_ps();
      __m512 scl_fctr2 = _mm512_setzero_ps();
      __m512 scl_fctr3 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                    ( post_ops_attr.c_stor_type == BF16 ) );

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
              ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          scl_fctr2 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 1 * 16 ) );
          scl_fctr3 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 2 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
        }
      }
      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31,32-47]
          BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
              scl_fctr1,scl_fctr2,scl_fctr3,0,8,9,10);

          // c[1:0-15,16-31,32-47]
          BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
              scl_fctr1,scl_fctr2,scl_fctr3,1,12,13,14);
        }
        else
        {
          // c[0:0-15,16-31,32-47]
          BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
              scl_fctr1,scl_fctr1,scl_fctr1,0,8,9,10);

          // c[1:0-15,16-31,32-47]
          BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
              scl_fctr2,scl_fctr2,scl_fctr2,1,12,13,14);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31,32-47]
          F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
              scl_fctr1,scl_fctr2,scl_fctr3,0,8,9,10);

          // c[1:0-15,16-31,32-47]
          F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
              scl_fctr1,scl_fctr2,scl_fctr3,1,12,13,14);
        }
        else
        {
          // c[0:0-15,16-31,32-47]
          F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
              scl_fctr1,scl_fctr1,scl_fctr1,0,8,9,10);

          // c[1:0-15,16-31,32-47]
          F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
              scl_fctr2,scl_fctr2,scl_fctr2,1,12,13,14);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_2x48F:
    {
        __m512 zmm7 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );
        __m512i ex_out;

        // c[0, 0-15]
        SWISH_F32_AVX512_DEF(zmm8, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[0, 16-31]
        SWISH_F32_AVX512_DEF(zmm9, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[0, 32-47]
        SWISH_F32_AVX512_DEF(zmm10, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 0-15]
        SWISH_F32_AVX512_DEF(zmm12, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 16-31]
        SWISH_F32_AVX512_DEF(zmm13, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 32-47]
        SWISH_F32_AVX512_DEF(zmm14, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_2x48F:
      {
        __m512i zmm6;
        // c[0, 0-15]
        TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[0, 16-31]
        TANH_F32S_AVX512(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[0, 32-47]
        TANH_F32S_AVX512(zmm10, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 0-15]
        TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 16-31]
        TANH_F32S_AVX512(zmm13, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 32-47]
        TANH_F32S_AVX512(zmm14, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_SIGMOID_2x48F:
      {
          __m512i ex_out;

          // c[0, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[0, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[0, 32-47]
          SIGMOID_F32_AVX512_DEF(zmm10, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm13, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 32-47]
          SIGMOID_F32_AVX512_DEF(zmm14, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_2x48F_DISABLE:
      ;

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_last_k == TRUE ) )
      {
        uint32_t tlsb, rounded, temp[16] = {0};
        int i;
        bfloat16* dest;

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm8, 0, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm9, 0, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm10, 0, 2);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm12, 1, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm13, 1, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm14, 1, 2);
      }
      else
      {
        _mm512_storeu_ps(cbuf, zmm8);
        _mm512_storeu_ps(cbuf + 16, zmm9);
        _mm512_storeu_ps(cbuf + 32, zmm10);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm12);
        _mm512_storeu_ps(cbuf + 16, zmm13);
        _mm512_storeu_ps(cbuf + 32, zmm14);
      }
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
              &&POST_OPS_CLIP_1x48F,
              &&POST_OPS_DOWNSCALE_1x48F,
              &&POST_OPS_MATRIX_ADD_1x48F,
              &&POST_OPS_SWISH_1x48F,
              &&POST_OPS_MATRIX_MUL_1x48F,
              &&POST_OPS_TANH_1x48F,
              &&POST_OPS_SIGMOID_1x48F
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
      zmm3 = _mm512_set1_ps(beta);

      //load c and beta, convert to f32
      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_first_k == TRUE ) )
      {
        //c[0, 0-15]
        BF16_F32_BETA_OP(zmm8, m, 0, 0, zmm0,zmm3);
        //c[0, 16-31]
        BF16_F32_BETA_OP(zmm9, m, 0, 1, zmm1,zmm3);
        //c[0, 32-47]
        BF16_F32_BETA_OP(zmm10, m, 0, 2, zmm0,zmm3);
      }
      else
      {
        //load c and multiply with beta and
        //add to accumulator and store back
        zmm0 = _mm512_loadu_ps(cbuf);
        zmm1 = _mm512_loadu_ps(cbuf + 16);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);

        zmm0 = _mm512_loadu_ps(cbuf + 32);
        zmm10 = _mm512_fmadd_ps(zmm0, zmm3, zmm10);
      }
    }

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_1x48F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_BIAS_LOAD(zmm1, bias_mask, 0)
          BF16_F32_BIAS_LOAD(zmm2, bias_mask, 1)
          BF16_F32_BIAS_LOAD(zmm3, bias_mask, 2)
        }
        else
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
        }
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
        if ( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );

          BF16_F32_BIAS_BCAST(zmm1, bias_mask, 0)
        }
        else
        {
          zmm1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 0 ) );
        }
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
POST_OPS_DOWNSCALE_1x48F:
    {
      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();
      __m512 selector3 = _mm512_setzero_ps();

      __m512 zero_point0 = _mm512_setzero_ps();
      __m512 zero_point1 = _mm512_setzero_ps();
      __m512 zero_point2 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                    ( ( post_ops_list_temp->stor_type == NONE ) &&
                      ( post_ops_attr.c_stor_type == BF16 ) );

      // Need to account for row vs column major swaps. For scalars
      // scale and zero point, no implications.
      // Even though different registers are used for scalar in column
      // and row major downscale path, all those registers will contain
      // the same value.

      if( post_ops_list_temp->scale_factor_len == 1 )
      {
        selector1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector2 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector3 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
     }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if ( is_bf16 == TRUE )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_ZP_BCST(zero_point0,0, zp_mask)
          BF16_F32_ZP_BCST(zero_point1,1, zp_mask)
          BF16_F32_ZP_BCST(zero_point2,2, zp_mask)
        }
        else
        {
          zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          selector2 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 1 * 16 ) );
          selector3 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 2 * 16 ) );
        }
        if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 load_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_LOAD(zero_point0,load_mask, 0)
            BF16_F32_ZP_LOAD(zero_point1,load_mask, 1)
            BF16_F32_ZP_LOAD(zero_point2,load_mask, 2)
          }
          else
          {
            zero_point0 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 0 * 16 ) );
            zero_point1 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 1 * 16 ) );
            zero_point2 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 2 * 16 ) );
          }
        }

        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[0, 16-31]
        F32_SCL_MULRND(zmm9, selector2, zero_point1);

        //c[0, 32-47]
        F32_SCL_MULRND(zmm10, selector3, zero_point2);
      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the scale as well as zp array will
        // be accessed by the ic index, and each scale/zp element
        // corresponds to an entire row of the transposed output array,
        // instead of an entire column.
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 0 ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( ( post_ops_attr.buf_downscale != NULL ) &&
               ( post_ops_attr.is_first_k == TRUE ) )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_COL_BCST(zero_point0, 0, zp_mask)
          }
          else
          {
            zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 0 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[0, 16-31]
        F32_SCL_MULRND(zmm9, selector1, zero_point0);

        //c[0, 32-47]
        F32_SCL_MULRND(zmm10, selector1, zero_point0);
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_1x48F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();
      __m512 selector3 = _mm512_setzero_ps();

      __m512 scl_fctr1 = _mm512_setzero_ps();
      __m512 scl_fctr2 = _mm512_setzero_ps();
      __m512 scl_fctr3 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                ( post_ops_attr.c_stor_type == BF16 ) );

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          scl_fctr2 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 1 * 16 ) );
          scl_fctr3 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 2 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
        }
      }

      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31,32-47]
          BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
              scl_fctr1,scl_fctr2,scl_fctr3,0,8,9,10);
        }
        else
        {
          // c[0:0-15,16-31,32-47]
          BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
              scl_fctr1,scl_fctr1,scl_fctr1,0,8,9,10);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31,32-47]
          F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
              scl_fctr1,scl_fctr2,scl_fctr3,0,8,9,10);
        }
        else
        {
          // c[0:0-15,16-31,32-47]
          F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,\
              scl_fctr1,scl_fctr1,scl_fctr1,0,8,9,10);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_1x48F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();
      __m512 selector3 = _mm512_setzero_ps();

      __m512 scl_fctr1 = _mm512_setzero_ps();
      __m512 scl_fctr2 = _mm512_setzero_ps();
      __m512 scl_fctr3 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                    ( post_ops_attr.c_stor_type == BF16 ) );

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
              ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          scl_fctr2 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 1 * 16 ) );
          scl_fctr3 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 2 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
        }
      }

      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31,32-47]
          BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
              scl_fctr1,scl_fctr2,scl_fctr3,0,8,9,10);
        }
        else
        {
          // c[0:0-15,16-31,32-47]
          BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
              scl_fctr1,scl_fctr1,scl_fctr1,0,8,9,10);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31,32-47]
          F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
              scl_fctr1,scl_fctr2,scl_fctr3,0,8,9,10);
        }
        else
        {
          // c[0:0-15,16-31,32-47]
          F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,\
              scl_fctr1,scl_fctr1,scl_fctr1,0,8,9,10);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_1x48F:
    {
        __m512 zmm7 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );
        __m512i ex_out;

        // c[0, 0-15]
        SWISH_F32_AVX512_DEF(zmm8, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[0, 16-31]
        SWISH_F32_AVX512_DEF(zmm9, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[0, 32-47]
        SWISH_F32_AVX512_DEF(zmm10, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_1x48F:
      {
        __m512i zmm6;
        // c[0, 0-15]
        TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[0, 16-31]
        TANH_F32S_AVX512(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[0, 32-47]
        TANH_F32S_AVX512(zmm10, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_SIGMOID_1x48F:
      {
          __m512i ex_out;

          // c[0, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[0, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[0, 32-47]
          SIGMOID_F32_AVX512_DEF(zmm10, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_1x48F_DISABLE:
      ;

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_last_k == TRUE ) )
      {
        uint32_t tlsb, rounded, temp[16] = {0};
        int i;
        bfloat16* dest;

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm8, 0, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm9, 0, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm10, 0, 2);
      }
      else
      {
        _mm512_storeu_ps(cbuf, zmm8);
        _mm512_storeu_ps(cbuf + 16, zmm9);
        _mm512_storeu_ps(cbuf + 32, zmm10);
      }
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
              &&POST_OPS_CLIP_5x32F,
              &&POST_OPS_DOWNSCALE_5x32F,
              &&POST_OPS_MATRIX_ADD_5x32F,
              &&POST_OPS_SWISH_5x32F,
              &&POST_OPS_MATRIX_MUL_5x32F,
              &&POST_OPS_TANH_5x32F,
              &&POST_OPS_SIGMOID_5x32F
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
      zmm3 = _mm512_set1_ps(beta);

      //load c and beta, convert to f32
      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_first_k == TRUE ) )
      {
        //c[0, 0-15]
        BF16_F32_BETA_OP(zmm8, m, 0, 0, zmm0,zmm3);
        //c[0, 16-31]
        BF16_F32_BETA_OP(zmm9, m, 0, 1, zmm1,zmm3);
        //c[1, 0-15]
        BF16_F32_BETA_OP(zmm12, m, 1, 0, zmm0,zmm3);
        //c[1, 16-31]
        BF16_F32_BETA_OP(zmm13, m, 1, 1, zmm1,zmm3);
        //c[2, 0-15]
        BF16_F32_BETA_OP(zmm16, m, 2, 0, zmm0,zmm3);
        //c[2,16-31]
        BF16_F32_BETA_OP(zmm17, m, 2, 1, zmm1,zmm3);
        //c[3, 0-15]
        BF16_F32_BETA_OP(zmm20, m, 3, 0, zmm0,zmm3);
        //c[3,16-31]
        BF16_F32_BETA_OP(zmm21, m, 3, 1, zmm1,zmm3);
        //c[4, 0-15]
        BF16_F32_BETA_OP(zmm24, m, 4, 0, zmm0,zmm3);
        //c[4, 16-31]
        BF16_F32_BETA_OP(zmm25, m, 4, 1, zmm1,zmm3);
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);
        _cbuf += rs_c;

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);
        _cbuf += rs_c;

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm16 = _mm512_fmadd_ps(zmm0, zmm3, zmm16);
        zmm17 = _mm512_fmadd_ps(zmm1, zmm3, zmm17);
        _cbuf += rs_c;

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm20 = _mm512_fmadd_ps(zmm0, zmm3, zmm20);
        zmm21 = _mm512_fmadd_ps(zmm1, zmm3, zmm21);
        _cbuf += rs_c;

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm24 = _mm512_fmadd_ps(zmm0, zmm3, zmm24);
        zmm25 = _mm512_fmadd_ps(zmm1, zmm3, zmm25);
      }
    }

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_5x32F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_BIAS_LOAD(zmm1, bias_mask, 0)
          BF16_F32_BIAS_LOAD(zmm2, bias_mask, 1)
        }
        else
        {
          zmm1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          zmm2 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 1 * 16 ) );
        }
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
        if ( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );

          BF16_F32_BIAS_BCAST(zmm1, bias_mask, 0)
          BF16_F32_BIAS_BCAST(zmm2, bias_mask, 1)
          BF16_F32_BIAS_BCAST(zmm3, bias_mask, 2)
          BF16_F32_BIAS_BCAST(zmm4, bias_mask, 3)
          BF16_F32_BIAS_BCAST(zmm5, bias_mask, 4)
        }
        else
        {
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
        }
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
POST_OPS_DOWNSCALE_5x32F:
    {
      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();
      __m512 selector3 = _mm512_setzero_ps();
      __m512 selector4 = _mm512_setzero_ps();
      __m512 selector5 = _mm512_setzero_ps();

      __m512 zero_point0 = _mm512_setzero_ps();
      __m512 zero_point1 = _mm512_setzero_ps();
      __m512 zero_point2 = _mm512_setzero_ps();
      __m512 zero_point3 = _mm512_setzero_ps();
      __m512 zero_point4 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                    ( ( post_ops_list_temp->stor_type == NONE ) &&
                      ( post_ops_attr.c_stor_type == BF16 ) );

      // Need to account for row vs column major swaps. For scalars
      // scale and zero point, no implications.
      // Even though different registers are used for scalar in column
      // and row major downscale path, all those registers will contain
      // the same value.

      if( post_ops_list_temp->scale_factor_len == 1 )
      {
        selector1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector2 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if ( is_bf16 == TRUE )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_ZP_BCST(zero_point0,0, zp_mask)
          BF16_F32_ZP_BCST(zero_point1,1, zp_mask)
        }
        else
        {
          zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          selector2 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 1 * 16 ) );
        }
        if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 load_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_LOAD(zero_point0,load_mask, 0)
            BF16_F32_ZP_LOAD(zero_point1,load_mask, 1)
          }
          else
          {
            zero_point0 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 0 * 16 ) );
            zero_point1 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 1 * 16 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[0, 16-31]
        F32_SCL_MULRND(zmm9, selector2, zero_point1);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector1, zero_point0);

        //c[1, 16-31]
        F32_SCL_MULRND(zmm13, selector2, zero_point1);

        //c[2, 0-15]
        F32_SCL_MULRND(zmm16, selector1, zero_point0);

        //c[2, 16-31]
        F32_SCL_MULRND(zmm17, selector2, zero_point1);

        //c[3, 0-15]
        F32_SCL_MULRND(zmm20, selector1, zero_point0);

        //c[3, 16-31]
        F32_SCL_MULRND(zmm21, selector2, zero_point1);

        //c[4, 0-15]
        F32_SCL_MULRND(zmm24, selector1, zero_point0);

        //c[4, 16-31]
        F32_SCL_MULRND(zmm25, selector2, zero_point1);
      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the scale as well as zp array will
        // be accessed by the ic index, and each scale/zp element
        // corresponds to an entire row of the transposed output array,
        // instead of an entire column.
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 0 ) );
          selector2 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 1 ) );
          selector3 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 2 ) );
          selector4 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 3 ) );
          selector5 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 4 ) );
        }
        else
        {
          selector3 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          selector4 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          selector5 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
              BF16_F32_ZP_COL_BCST(zero_point0, 0, zp_mask)
              BF16_F32_ZP_COL_BCST(zero_point1, 1, zp_mask)
              BF16_F32_ZP_COL_BCST(zero_point2, 2, zp_mask)
              BF16_F32_ZP_COL_BCST(zero_point3, 3, zp_mask)
              BF16_F32_ZP_COL_BCST(zero_point4, 4, zp_mask)
          }
          else
          {
            zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 1) );
            zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 2 ) );
            zero_point3 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 3 ) );
            zero_point4 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 4 ) );
          }
        }
        else
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_BCST(zero_point2,2, zp_mask)
            BF16_F32_ZP_BCST(zero_point3,3, zp_mask)
            BF16_F32_ZP_BCST(zero_point4,4, zp_mask)
          }
          else
          {
            zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point3 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point4 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[0, 16-31]
        F32_SCL_MULRND(zmm9, selector1, zero_point0);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector2, zero_point1);

        //c[1, 16-31]
        F32_SCL_MULRND(zmm13, selector2, zero_point1);

        //c[2, 0-15]
        F32_SCL_MULRND(zmm16, selector3, zero_point2);

        //c[2, 16-31]
        F32_SCL_MULRND(zmm17, selector3, zero_point2);

        //c[3, 0-15]
        F32_SCL_MULRND(zmm20, selector4, zero_point3);

        //c[3, 16-31]
        F32_SCL_MULRND(zmm21, selector4, zero_point3);

        //c[4, 0-15]
        F32_SCL_MULRND(zmm24, selector5, zero_point4);

        //c[4, 16-31]
        F32_SCL_MULRND(zmm25, selector5, zero_point4);
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_5x32F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();

      __m512 scl_fctr1 = _mm512_setzero_ps();
      __m512 scl_fctr2 = _mm512_setzero_ps();
      __m512 scl_fctr3 = _mm512_setzero_ps();
      __m512 scl_fctr4 = _mm512_setzero_ps();
      __m512 scl_fctr5 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                ( post_ops_attr.c_stor_type == BF16 ) );

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr4 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr5 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          scl_fctr2 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 1 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
          scl_fctr4 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 3 ) );
          scl_fctr5 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 4 ) );
        }
      }

      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,0,8,9);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,1,12,13);

          // c[2:0-15,16-31]
          BF16_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,2,16,17);

          // c[3:0-15,16-31]
          BF16_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,3,20,21);

          // c[4:0-15,16-31]
          BF16_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,4,24,25);
        }
        else
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr1,0,8,9);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr2,scl_fctr2,1,12,13);

          // c[2:0-15,16-31]
          BF16_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr3,scl_fctr3,2,16,17);

          // c[3:0-15,16-31]
          BF16_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr4,scl_fctr4,3,20,21);

          // c[4:0-15,16-31]
          BF16_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr5,scl_fctr5,4,24,25);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,0,8,9);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,1,12,13);

          // c[2:0-15,16-31]
          F32_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,2,16,17);

          // c[3:0-15,16-31]
          F32_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,3,20,21);

          // c[4:0-15,16-31]
          F32_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,4,24,25);
        }
        else
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr1,0,8,9);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr2,scl_fctr2,1,12,13);

          // c[2:0-15,16-31]
          F32_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr3,scl_fctr3,2,16,17);

          // c[3:0-15,16-31]
          F32_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr4,scl_fctr4,3,20,21);

          // c[4:0-15,16-31]
          F32_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr5,scl_fctr5,4,24,25);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_5x32F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();

      __m512 scl_fctr1 = _mm512_setzero_ps();
      __m512 scl_fctr2 = _mm512_setzero_ps();
      __m512 scl_fctr3 = _mm512_setzero_ps();
      __m512 scl_fctr4 = _mm512_setzero_ps();
      __m512 scl_fctr5 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                    ( post_ops_attr.c_stor_type == BF16 ) );

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr4 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr5 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
              ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          scl_fctr2 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 1 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
          scl_fctr4 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 3 ) );
          scl_fctr5 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 4 ) );
        }
      }

      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,0,8,9);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,1,12,13);

          // c[2:0-15,16-31]
          BF16_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,2,16,17);

          // c[3:0-15,16-31]
          BF16_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,3,20,21);

          // c[4:0-15,16-31]
          BF16_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,4,24,25);
        }
        else
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr1,0,8,9);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr2,scl_fctr2,1,12,13);

          // c[2:0-15,16-31]
          BF16_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr3,scl_fctr3,2,16,17);

          // c[3:0-15,16-31]
          BF16_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr4,scl_fctr4,3,20,21);

          // c[4:0-15,16-31]
          BF16_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr5,scl_fctr5,4,24,25);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,0,8,9);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,1,12,13);

          // c[2:0-15,16-31]
          F32_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,2,16,17);

          // c[3:0-15,16-31]
          F32_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,3,20,21);

          // c[4:0-15,16-31]
          F32_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,4,24,25);
        }
        else
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr1,0,8,9);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr2,scl_fctr2,1,12,13);

          // c[2:0-15,16-31]
          F32_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr3,scl_fctr3,2,16,17);

          // c[3:0-15,16-31]
          F32_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr4,scl_fctr4,3,20,21);

          // c[4:0-15,16-31]
          F32_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr5,scl_fctr5,4,24,25);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_5x32F:
    {
        __m512 zmm7 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );
        __m512i ex_out;

        // c[0, 0-15]
        SWISH_F32_AVX512_DEF(zmm8, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[0, 16-31]
        SWISH_F32_AVX512_DEF(zmm9, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 0-15]
        SWISH_F32_AVX512_DEF(zmm12, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 16-31]
        SWISH_F32_AVX512_DEF(zmm13, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[2, 0-15]
        SWISH_F32_AVX512_DEF(zmm16, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[2, 16-31]
        SWISH_F32_AVX512_DEF(zmm17, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[3, 0-15]
        SWISH_F32_AVX512_DEF(zmm20, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[3, 16-31]
        SWISH_F32_AVX512_DEF(zmm21, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[4, 0-15]
        SWISH_F32_AVX512_DEF(zmm24, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[4, 16-31]
        SWISH_F32_AVX512_DEF(zmm25, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_5x32F:
      {
        __m512i zmm6;
        // c[0, 0-15]
        TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[0, 16-31]
        TANH_F32S_AVX512(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 0-15]
        TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 16-31]
        TANH_F32S_AVX512(zmm13, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[2, 0-15]
        TANH_F32S_AVX512(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[2, 16-31]
        TANH_F32S_AVX512(zmm17, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[3, 0-15]
        TANH_F32S_AVX512(zmm20, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[3, 16-31]
        TANH_F32S_AVX512(zmm21, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[4, 0-15]
        TANH_F32S_AVX512(zmm24, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[4, 16-31]
        TANH_F32S_AVX512(zmm25, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_SIGMOID_5x32F:
      {
          __m512i ex_out;

          // c[0, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[0, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm13, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm17, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[3, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm20, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[3, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm21, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[4, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm24, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[4, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm25, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_5x32F_DISABLE:
      ;

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_last_k == TRUE ) )
      {
        uint32_t tlsb, rounded, temp[16] = {0};
        int i;
        bfloat16* dest;

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm8, 0, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm9, 0, 1);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm12, 1, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm13, 1, 1);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm16, 2, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm17, 2, 1);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm20, 3, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm21, 3, 1);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm24, 4, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm25, 4, 1);
      }
      else
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
      }
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
              &&POST_OPS_CLIP_4x32F,
              &&POST_OPS_DOWNSCALE_4x32F,
              &&POST_OPS_MATRIX_ADD_4x32F,
              &&POST_OPS_SWISH_4x32F,
              &&POST_OPS_MATRIX_MUL_4x32F,
              &&POST_OPS_TANH_4x32F,
              &&POST_OPS_SIGMOID_4x32F
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
      zmm3 = _mm512_set1_ps(beta);

      //load c and beta, convert to f32
      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_first_k == TRUE ) )
      {
        //c[0, 0-15]
        BF16_F32_BETA_OP(zmm8, m, 0, 0, zmm0,zmm3);
        //c[0, 16-31]
        BF16_F32_BETA_OP(zmm9, m, 0, 1, zmm1,zmm3);
        //c[1, 0-15]
        BF16_F32_BETA_OP(zmm12, m, 1, 0, zmm0,zmm3);
        //c[1, 16-31]
        BF16_F32_BETA_OP(zmm13, m, 1, 1, zmm1,zmm3);
        //c[2, 0-15]
        BF16_F32_BETA_OP(zmm16, m, 2, 0, zmm0,zmm3);
        //c[2,16-31]
        BF16_F32_BETA_OP(zmm17, m, 2, 1, zmm1,zmm3);
        //c[3, 0-15]
        BF16_F32_BETA_OP(zmm20, m, 3, 0, zmm0,zmm3);
        //c[3,16-31]
        BF16_F32_BETA_OP(zmm21, m, 3, 1, zmm1,zmm3);
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);
        _cbuf += rs_c;

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);
        _cbuf += rs_c;

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm16 = _mm512_fmadd_ps(zmm0, zmm3, zmm16);
        zmm17 = _mm512_fmadd_ps(zmm1, zmm3, zmm17);
        _cbuf += rs_c;

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm20 = _mm512_fmadd_ps(zmm0, zmm3, zmm20);
        zmm21 = _mm512_fmadd_ps(zmm1, zmm3, zmm21);
      }
    }

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_4x32F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_BIAS_LOAD(zmm1, bias_mask, 0)
          BF16_F32_BIAS_LOAD(zmm2, bias_mask, 1)
        }
        else
        {
          zmm1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          zmm2 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 1 * 16 ) );
        }
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
        if ( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );

          BF16_F32_BIAS_BCAST(zmm1, bias_mask, 0)
          BF16_F32_BIAS_BCAST(zmm2, bias_mask, 1)
          BF16_F32_BIAS_BCAST(zmm3, bias_mask, 2)
          BF16_F32_BIAS_BCAST(zmm4, bias_mask, 3)
        }
        else
        {
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
        }
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
POST_OPS_DOWNSCALE_4x32F:
    {
      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();
      __m512 selector3 = _mm512_setzero_ps();
      __m512 selector4 = _mm512_setzero_ps();

      __m512 zero_point0 = _mm512_setzero_ps();
      __m512 zero_point1 = _mm512_setzero_ps();
      __m512 zero_point2 = _mm512_setzero_ps();
      __m512 zero_point3 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                    ( ( post_ops_list_temp->stor_type == NONE ) &&
                      ( post_ops_attr.c_stor_type == BF16 ) );

      // Need to account for row vs column major swaps. For scalars
      // scale and zero point, no implications.
      // Even though different registers are used for scalar in column
      // and row major downscale path, all those registers will contain
      // the same value.

      if( post_ops_list_temp->scale_factor_len == 1 )
      {
        selector1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector2 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if ( is_bf16 == TRUE )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_ZP_BCST(zero_point0,0, zp_mask)
          BF16_F32_ZP_BCST(zero_point1,1, zp_mask)
        }
        else
        {
          zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          selector2 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 1 * 16 ) );
        }
        if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 load_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_LOAD(zero_point0,load_mask, 0)
            BF16_F32_ZP_LOAD(zero_point1,load_mask, 1)
          }
          else
          {
            zero_point0 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 0 * 16 ) );
            zero_point1 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 1 * 16 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[0, 16-31]
        F32_SCL_MULRND(zmm9, selector2, zero_point1);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector1, zero_point0);

        //c[1, 16-31]
        F32_SCL_MULRND(zmm13, selector2, zero_point1);

        //c[2, 0-15]
        F32_SCL_MULRND(zmm16, selector1, zero_point0);

        //c[2, 16-31]
        F32_SCL_MULRND(zmm17, selector2, zero_point1);

        //c[3, 0-15]
        F32_SCL_MULRND(zmm20, selector1, zero_point0);

        //c[3, 16-31]
        F32_SCL_MULRND(zmm21, selector2, zero_point1);
      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the scale as well as zp array will
        // be accessed by the ic index, and each scale/zp element
        // corresponds to an entire row of the transposed output array,
        // instead of an entire column.
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 0 ) );
          selector2 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 1 ) );
          selector3 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 2 ) );
          selector4 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 3 ) );
        }
        else
        {
          selector3 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          selector4 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_COL_BCST(zero_point0, 0, zp_mask)
            BF16_F32_ZP_COL_BCST(zero_point1, 1, zp_mask)
            BF16_F32_ZP_COL_BCST(zero_point2, 2, zp_mask)
            BF16_F32_ZP_COL_BCST(zero_point3, 3, zp_mask)
          }
          else
          {
            zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 1) );
            zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 2 ) );
            zero_point3 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 3 ) );
          }
        }
        else
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_BCST(zero_point2,2, zp_mask)
            BF16_F32_ZP_BCST(zero_point3,3, zp_mask)
          }
          else
          {
            zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point3 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[0, 16-31]
        F32_SCL_MULRND(zmm9, selector1, zero_point0);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector2, zero_point1);

        //c[1, 16-31]
        F32_SCL_MULRND(zmm13, selector2, zero_point1);

        //c[2, 0-15]
        F32_SCL_MULRND(zmm16, selector3, zero_point2);

        //c[2, 16-31]
        F32_SCL_MULRND(zmm17, selector3, zero_point2);

        //c[3, 0-15]
        F32_SCL_MULRND(zmm20, selector4, zero_point3);

        //c[3, 16-31]
        F32_SCL_MULRND(zmm21, selector4, zero_point3);
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_4x32F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();

      __m512 scl_fctr1 = _mm512_setzero_ps();
      __m512 scl_fctr2 = _mm512_setzero_ps();
      __m512 scl_fctr3 = _mm512_setzero_ps();
      __m512 scl_fctr4 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                ( post_ops_attr.c_stor_type == BF16 ) );

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr4 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          scl_fctr2 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 1 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
          scl_fctr4 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 3 ) );
        }
      }

      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,0,8,9);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,1,12,13);

          // c[2:0-15,16-31]
          BF16_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,2,16,17);

          // c[3:0-15,16-31]
          BF16_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,3,20,21);
        }
        else
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr1,0,8,9);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr2,scl_fctr2,1,12,13);

          // c[2:0-15,16-31]
          BF16_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr3,scl_fctr3,2,16,17);

          // c[3:0-15,16-31]
          BF16_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr4,scl_fctr4,3,20,21);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,0,8,9);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,1,12,13);

          // c[2:0-15,16-31]
          F32_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,2,16,17);

          // c[3:0-15,16-31]
          F32_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,3,20,21);
        }
        else
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr1,0,8,9);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr2,scl_fctr2,1,12,13);

          // c[2:0-15,16-31]
          F32_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr3,scl_fctr3,2,16,17);

          // c[3:0-15,16-31]
          F32_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr4,scl_fctr4,3,20,21);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_4x32F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();

      __m512 scl_fctr1 = _mm512_setzero_ps();
      __m512 scl_fctr2 = _mm512_setzero_ps();
      __m512 scl_fctr3 = _mm512_setzero_ps();
      __m512 scl_fctr4 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                    ( post_ops_attr.c_stor_type == BF16 ) );

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr4 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
              ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          scl_fctr2 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 1 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
          scl_fctr4 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 3 ) );
        }
      }

      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,0,8,9);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,1,12,13);

          // c[2:0-15,16-31]
          BF16_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,2,16,17);

          // c[3:0-15,16-31]
          BF16_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,3,20,21);
        }
        else
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr1,0,8,9);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr2,scl_fctr2,1,12,13);

          // c[2:0-15,16-31]
          BF16_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr3,scl_fctr3,2,16,17);

          // c[3:0-15,16-31]
          BF16_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr4,scl_fctr4,3,20,21);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,0,8,9);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,1,12,13);

          // c[2:0-15,16-31]
          F32_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,2,16,17);

          // c[3:0-15,16-31]
          F32_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,3,20,21);
        }
        else
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr1,0,8,9);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr2,scl_fctr2,1,12,13);

          // c[2:0-15,16-31]
          F32_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr3,scl_fctr3,2,16,17);

          // c[3:0-15,16-31]
          F32_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr4,scl_fctr4,3,20,21);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_4x32F:
    {
        __m512 zmm7 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );
        __m512i ex_out;

        // c[0, 0-15]
        SWISH_F32_AVX512_DEF(zmm8, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[0, 16-31]
        SWISH_F32_AVX512_DEF(zmm9, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 0-15]
        SWISH_F32_AVX512_DEF(zmm12, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 16-31]
        SWISH_F32_AVX512_DEF(zmm13, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[2, 0-15]
        SWISH_F32_AVX512_DEF(zmm16, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[2, 16-31]
        SWISH_F32_AVX512_DEF(zmm17, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[3, 0-15]
        SWISH_F32_AVX512_DEF(zmm20, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[3, 16-31]
        SWISH_F32_AVX512_DEF(zmm21, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_4x32F:
      {
        __m512i zmm6;
        // c[0, 0-15]
        TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[0, 16-31]
        TANH_F32S_AVX512(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 0-15]
        TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 16-31]
        TANH_F32S_AVX512(zmm13, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[2, 0-15]
        TANH_F32S_AVX512(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[2, 16-31]
        TANH_F32S_AVX512(zmm17, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[3, 0-15]
        TANH_F32S_AVX512(zmm20, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[3, 16-31]
        TANH_F32S_AVX512(zmm21, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_SIGMOID_4x32F:
      {
          __m512i ex_out;

          // c[0, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[0, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm13, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm17, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[3, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm20, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[3, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm21, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_4x32F_DISABLE:
      ;

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_last_k == TRUE ) )
      {
        uint32_t tlsb, rounded, temp[16] = {0};
        int i;
        bfloat16* dest;

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm8, 0, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm9, 0, 1);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm12, 1, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm13, 1, 1);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm16, 2, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm17, 2, 1);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm20, 3, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm21, 3, 1);
      }
      else
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
      }
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
              &&POST_OPS_CLIP_3x32F,
              &&POST_OPS_DOWNSCALE_3x32F,
              &&POST_OPS_MATRIX_ADD_3x32F,
              &&POST_OPS_SWISH_3x32F,
              &&POST_OPS_MATRIX_MUL_3x32F,
              &&POST_OPS_TANH_3x32F,
              &&POST_OPS_SIGMOID_3x32F
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
      zmm3 = _mm512_set1_ps(beta);

      //load c and beta, convert to f32
      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_first_k == TRUE ) )
      {
        //c[0, 0-15]
        BF16_F32_BETA_OP(zmm8, m, 0, 0, zmm0,zmm3);
        //c[0, 16-31]
        BF16_F32_BETA_OP(zmm9, m, 0, 1, zmm1,zmm3);
        //c[1, 0-15]
        BF16_F32_BETA_OP(zmm12, m, 1, 0, zmm0,zmm3);
        //c[1, 16-31]
        BF16_F32_BETA_OP(zmm13, m, 1, 1, zmm1,zmm3);
        //c[2, 0-15]
        BF16_F32_BETA_OP(zmm16, m, 2, 0, zmm0,zmm3);
        //c[2,16-31]
        BF16_F32_BETA_OP(zmm17, m, 2, 1, zmm1,zmm3);
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);
        _cbuf += rs_c;

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);
        _cbuf += rs_c;

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm16 = _mm512_fmadd_ps(zmm0, zmm3, zmm16);
        zmm17 = _mm512_fmadd_ps(zmm1, zmm3, zmm17);
      }
    }

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_3x32F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_BIAS_LOAD(zmm1, bias_mask, 0)
          BF16_F32_BIAS_LOAD(zmm2, bias_mask, 1)
        }
        else
        {
          zmm1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          zmm2 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 1 * 16 ) );
        }
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
        if ( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );

          BF16_F32_BIAS_BCAST(zmm1, bias_mask, 0)
          BF16_F32_BIAS_BCAST(zmm2, bias_mask, 1)
          BF16_F32_BIAS_BCAST(zmm3, bias_mask, 2)
        }
        else
        {
          zmm1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 0 ) );
          zmm2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 1 ) );
          zmm3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 2 ) );
        }
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
POST_OPS_DOWNSCALE_3x32F:
    {
      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();
      __m512 selector3 = _mm512_setzero_ps();

      __m512 zero_point0 = _mm512_setzero_ps();
      __m512 zero_point1 = _mm512_setzero_ps();
      __m512 zero_point2 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                    ( ( post_ops_list_temp->stor_type == NONE ) &&
                      ( post_ops_attr.c_stor_type == BF16 ) );

      // Need to account for row vs column major swaps. For scalars
      // scale and zero point, no implications.
      // Even though different registers are used for scalar in column
      // and row major downscale path, all those registers will contain
      // the same value.

      if( post_ops_list_temp->scale_factor_len == 1 )
      {
        selector1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector2 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if ( is_bf16 == TRUE )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_ZP_BCST(zero_point0,0, zp_mask)
          BF16_F32_ZP_BCST(zero_point1,1, zp_mask)
        }
        else
        {
          zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          selector2 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 1 * 16 ) );
        }
        if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 load_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_LOAD(zero_point0,load_mask, 0)
            BF16_F32_ZP_LOAD(zero_point1,load_mask, 1)
          }
          else
          {
            zero_point0 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 0 * 16 ) );
            zero_point1 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 1 * 16 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[0, 16-31]
        F32_SCL_MULRND(zmm9, selector2, zero_point1);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector1, zero_point0);

        //c[1, 16-31]
        F32_SCL_MULRND(zmm13, selector2, zero_point1);

        //c[2, 0-15]
        F32_SCL_MULRND(zmm16, selector1, zero_point0);

        //c[2, 16-31]
        F32_SCL_MULRND(zmm17, selector2, zero_point1);
      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the scale as well as zp array will
        // be accessed by the ic index, and each scale/zp element
        // corresponds to an entire row of the transposed output array,
        // instead of an entire column.
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 0 ) );
          selector2 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 1 ) );
          selector3 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 2 ) );
        }
        else
        {
          selector3 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_COL_BCST(zero_point0, 0, zp_mask)
            BF16_F32_ZP_COL_BCST(zero_point1, 1, zp_mask)
            BF16_F32_ZP_COL_BCST(zero_point2, 2, zp_mask)
          }
          else
          {
            zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 1) );
            zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 2 ) );
          }
        }
        else
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_BCST(zero_point2,2, zp_mask)
          }
          else
          {
            zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[0, 16-31]
        F32_SCL_MULRND(zmm9, selector1, zero_point0);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector2, zero_point1);

        //c[1, 16-31]
        F32_SCL_MULRND(zmm13, selector2, zero_point1);

        //c[2, 0-15]
        F32_SCL_MULRND(zmm16, selector3, zero_point2);

        //c[2, 16-31]
        F32_SCL_MULRND(zmm17, selector3, zero_point2);
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_3x32F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();

      __m512 scl_fctr1 = _mm512_setzero_ps();
      __m512 scl_fctr2 = _mm512_setzero_ps();
      __m512 scl_fctr3 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                ( post_ops_attr.c_stor_type == BF16 ) );

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          scl_fctr2 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 1 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
        }
      }

      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,0,8,9);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,1,12,13);

          // c[2:0-15,16-31]
          BF16_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,2,16,17);
        }
        else
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr1,0,8,9);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr2,scl_fctr2,1,12,13);

          // c[2:0-15,16-31]
          BF16_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr3,scl_fctr3,2,16,17);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,0,8,9);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,1,12,13);

          // c[2:0-15,16-31]
          F32_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,2,16,17);
        }
        else
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr1,0,8,9);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr2,scl_fctr2,1,12,13);

          // c[2:0-15,16-31]
          F32_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr3,scl_fctr3,2,16,17);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_3x32F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();

      __m512 scl_fctr1 = _mm512_setzero_ps();
      __m512 scl_fctr2 = _mm512_setzero_ps();
      __m512 scl_fctr3 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                    ( post_ops_attr.c_stor_type == BF16 ) );

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
              ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          scl_fctr2 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 1 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
        }
      }

      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,0,8,9);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,1,12,13);

          // c[2:0-15,16-31]
          BF16_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,2,16,17);
        }
        else
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr1,0,8,9);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr2,scl_fctr2,1,12,13);

          // c[2:0-15,16-31]
          BF16_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr3,scl_fctr3,2,16,17);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,0,8,9);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,1,12,13);

          // c[2:0-15,16-31]
          F32_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,2,16,17);
        }
        else
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr1,0,8,9);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr2,scl_fctr2,1,12,13);

          // c[2:0-15,16-31]
          F32_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr3,scl_fctr3,2,16,17);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_3x32F:
    {
        __m512 zmm7 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );
        __m512i ex_out;

        // c[0, 0-15]
        SWISH_F32_AVX512_DEF(zmm8, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[0, 16-31]
        SWISH_F32_AVX512_DEF(zmm9, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 0-15]
        SWISH_F32_AVX512_DEF(zmm12, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 16-31]
        SWISH_F32_AVX512_DEF(zmm13, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[2, 0-15]
        SWISH_F32_AVX512_DEF(zmm16, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[2, 16-31]
        SWISH_F32_AVX512_DEF(zmm17, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_3x32F:
      {
        __m512i zmm6;
        // c[0, 0-15]
        TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[0, 16-31]
        TANH_F32S_AVX512(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 0-15]
        TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 16-31]
        TANH_F32S_AVX512(zmm13, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[2, 0-15]
        TANH_F32S_AVX512(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[2, 16-31]
        TANH_F32S_AVX512(zmm17, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[3, 0-15]
        TANH_F32S_AVX512(zmm20, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[3, 16-31]
        TANH_F32S_AVX512(zmm21, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_SIGMOID_3x32F:
      {
          __m512i ex_out;

          // c[0, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[0, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm13, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm17, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[3, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm20, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[3, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm21, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_3x32F_DISABLE:
      ;

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_last_k == TRUE ) )
      {
        uint32_t tlsb, rounded, temp[16] = {0};
        int i;
        bfloat16* dest;

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm8, 0, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm9, 0, 1);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm12, 1, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm13, 1, 1);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm16, 2, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm17, 2, 1);
      }
      else
      {
        _mm512_storeu_ps(cbuf, zmm8);
        _mm512_storeu_ps(cbuf + 16, zmm9);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm12);
        _mm512_storeu_ps(cbuf + 16, zmm13);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm16);
        _mm512_storeu_ps(cbuf + 16, zmm17);

      }
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
              &&POST_OPS_CLIP_2x32F,
              &&POST_OPS_DOWNSCALE_2x32F,
              &&POST_OPS_MATRIX_ADD_2x32F,
              &&POST_OPS_SWISH_2x32F,
              &&POST_OPS_MATRIX_MUL_2x32F,
              &&POST_OPS_TANH_2x32F,
              &&POST_OPS_SIGMOID_2x32F
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
      zmm3 = _mm512_set1_ps(beta);

      //load c and beta, convert to f32
      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_first_k == TRUE ) )
      {
        //c[0, 0-15]
        BF16_F32_BETA_OP(zmm8, m, 0, 0, zmm0,zmm3);
        //c[0, 16-31]
        BF16_F32_BETA_OP(zmm9, m, 0, 1, zmm1,zmm3);
        //c[1, 0-15]
        BF16_F32_BETA_OP(zmm12, m, 1, 0, zmm0,zmm3);
        //c[1, 16-31]
        BF16_F32_BETA_OP(zmm13, m, 1, 1, zmm1,zmm3);
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);
        _cbuf += rs_c;

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm1 = _mm512_loadu_ps(_cbuf + 16);
        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);
      }
    }

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_2x32F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_BIAS_LOAD(zmm1, bias_mask, 0)
          BF16_F32_BIAS_LOAD(zmm2, bias_mask, 1)
        }
        else
        {
          zmm1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          zmm2 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 1 * 16 ) );
        }
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
        if ( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );

          BF16_F32_BIAS_BCAST(zmm1, bias_mask, 0)
          BF16_F32_BIAS_BCAST(zmm2, bias_mask, 1)
        }
        else
        {
          zmm1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 0 ) );
          zmm2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 1 ) );
        }
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
POST_OPS_DOWNSCALE_2x32F:
    {
      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();

      __m512 zero_point0 = _mm512_setzero_ps();
      __m512 zero_point1 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                    ( ( post_ops_list_temp->stor_type == NONE ) &&
                      ( post_ops_attr.c_stor_type == BF16 ) );

      // Need to account for row vs column major swaps. For scalars
      // scale and zero point, no implications.
      // Even though different registers are used for scalar in column
      // and row major downscale path, all those registers will contain
      // the same value.

      if( post_ops_list_temp->scale_factor_len == 1 )
      {
        selector1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector2 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if ( is_bf16 == TRUE )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_ZP_BCST(zero_point0,0, zp_mask)
          BF16_F32_ZP_BCST(zero_point1,1, zp_mask)
        }
        else
        {
          zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          selector2 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 1 * 16 ) );
        }
        if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 load_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_LOAD(zero_point0,load_mask, 0)
            BF16_F32_ZP_LOAD(zero_point1,load_mask, 1)
          }
          else
          {
            zero_point0 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 0 * 16 ) );
            zero_point1 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 1 * 16 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[0, 16-31]
        F32_SCL_MULRND(zmm9, selector2, zero_point1);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector1, zero_point0);

        //c[1, 16-31]
        F32_SCL_MULRND(zmm13, selector2, zero_point1);
      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the scale as well as zp array will
        // be accessed by the ic index, and each scale/zp element
        // corresponds to an entire row of the transposed output array,
        // instead of an entire column.
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 0 ) );
          selector2 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 1 ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_COL_BCST(zero_point0, 0, zp_mask)
            BF16_F32_ZP_COL_BCST(zero_point1, 1, zp_mask)
          }
          else
          {
            zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 1) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[0, 16-31]
        F32_SCL_MULRND(zmm9, selector1, zero_point0);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector2, zero_point1);

        //c[1, 16-31]
        F32_SCL_MULRND(zmm13, selector2, zero_point1);
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_2x32F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();

      __m512 scl_fctr1 = _mm512_setzero_ps();
      __m512 scl_fctr2 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                ( post_ops_attr.c_stor_type == BF16 ) );

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          scl_fctr2 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 1 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
        }
      }

      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,0,8,9);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,1,12,13);
        }
        else
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr1,0,8,9);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr2,scl_fctr2,1,12,13);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,0,8,9);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,1,12,13);
        }
        else
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr1,0,8,9);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr2,scl_fctr2,1,12,13);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_2x32F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();

      __m512 scl_fctr1 = _mm512_setzero_ps();
      __m512 scl_fctr2 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                    ( post_ops_attr.c_stor_type == BF16 ) );

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
              ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          scl_fctr2 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 1 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
        }
      }
      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,0,8,9);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,1,12,13);
        }
        else
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr1,0,8,9);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr2,scl_fctr2,1,12,13);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,0,8,9);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,1,12,13);
        }
        else
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr1,0,8,9);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr2,scl_fctr2,1,12,13);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_2x32F:
    {
        __m512 zmm7 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );
        __m512i ex_out;

        // c[0, 0-15]
        SWISH_F32_AVX512_DEF(zmm8, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[0, 16-31]
        SWISH_F32_AVX512_DEF(zmm9, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 0-15]
        SWISH_F32_AVX512_DEF(zmm12, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 16-31]
        SWISH_F32_AVX512_DEF(zmm13, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_2x32F:
      {
        __m512i zmm6;
        // c[0, 0-15]
        TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[0, 16-31]
        TANH_F32S_AVX512(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 0-15]
        TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 16-31]
        TANH_F32S_AVX512(zmm13, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_SIGMOID_2x32F:
      {
          __m512i ex_out;

          // c[0, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[0, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm13, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_2x32F_DISABLE:
      ;

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_last_k == TRUE ) )
      {
        uint32_t tlsb, rounded, temp[16] = {0};
        int i;
        bfloat16* dest;

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm8, 0, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm9, 0, 1);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm12, 1, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm13, 1, 1);
      }
      else
      {
        _mm512_storeu_ps(cbuf, zmm8);
        _mm512_storeu_ps(cbuf + 16, zmm9);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm12);
        _mm512_storeu_ps(cbuf + 16, zmm13);
      }
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
              &&POST_OPS_CLIP_1x32F,
              &&POST_OPS_DOWNSCALE_1x32F,
              &&POST_OPS_MATRIX_ADD_1x32F,
              &&POST_OPS_SWISH_1x32F,
              &&POST_OPS_MATRIX_MUL_1x32F,
              &&POST_OPS_TANH_1x32F,
              &&POST_OPS_SIGMOID_1x32F
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
      zmm3 = _mm512_set1_ps(beta);

      //load c and beta, convert to f32
      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_first_k == TRUE ) )
      {
        //c[0, 0-15]
        BF16_F32_BETA_OP(zmm8, m, 0, 0, zmm0,zmm3);
        //c[0, 16-31]
        BF16_F32_BETA_OP(zmm9, m, 0, 1, zmm1,zmm3);
      }
      else
      {
        //load c and multiply with beta and
        //add to accumulator and store back

        zmm0 = _mm512_loadu_ps(cbuf);
        zmm1 = _mm512_loadu_ps(cbuf + 16);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);
      }
    }

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_1x32F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_BIAS_LOAD(zmm1, bias_mask, 0)
          BF16_F32_BIAS_LOAD(zmm2, bias_mask, 1)
        }
        else
        {
          zmm1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          zmm2 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 1 * 16 ) );
        }
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
        if ( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );

          BF16_F32_BIAS_BCAST(zmm1, bias_mask, 0)
        }
        else
        {
          zmm1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 0 ) );
        }
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
POST_OPS_DOWNSCALE_1x32F:
    {
      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();

      __m512 zero_point0 = _mm512_setzero_ps();
      __m512 zero_point1 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                    ( ( post_ops_list_temp->stor_type == NONE ) &&
                      ( post_ops_attr.c_stor_type == BF16 ) );

      // Need to account for row vs column major swaps. For scalars
      // scale and zero point, no implications.
      // Even though different registers are used for scalar in column
      // and row major downscale path, all those registers will contain
      // the same value.

      if( post_ops_list_temp->scale_factor_len == 1 )
      {
        selector1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector2 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if ( is_bf16 == TRUE )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_ZP_BCST(zero_point0,0, zp_mask)
          BF16_F32_ZP_BCST(zero_point1,1, zp_mask)
        }
        else
        {
          zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          selector2 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 1 * 16 ) );
        }
        if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 load_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_LOAD(zero_point0,load_mask, 0)
            BF16_F32_ZP_LOAD(zero_point1,load_mask, 1)
          }
          else
          {
            zero_point0 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 0 * 16 ) );
            zero_point1 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 1 * 16 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[0, 16-31]
        F32_SCL_MULRND(zmm9, selector2, zero_point1);
      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the scale as well as zp array will
        // be accessed by the ic index, and each scale/zp element
        // corresponds to an entire row of the transposed output array,
        // instead of an entire column.
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 0 ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_COL_BCST(zero_point0, 0, zp_mask)
          }
          else
          {
            zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 0 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[0, 16-31]
        F32_SCL_MULRND(zmm9, selector1, zero_point0);
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_1x32F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();

      __m512 scl_fctr1 = _mm512_setzero_ps();
      __m512 scl_fctr2 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                ( post_ops_attr.c_stor_type == BF16 ) );

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          scl_fctr2 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 1 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
        }
      }

      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,0,8,9);
        }
        else
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr1,0,8,9);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,0,8,9);
        }
        else
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_ADD_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr1,0,8,9);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_1x32F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();

      __m512 scl_fctr1 = _mm512_setzero_ps();
      __m512 scl_fctr2 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                    ( post_ops_attr.c_stor_type == BF16 ) );

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
              ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          scl_fctr2 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 1 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
        }
      }
      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,0,8,9);
        }
        else
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr1,0,8,9);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr2,0,8,9);
        }
        else
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_MUL_2COL(selector1,selector2,\
              scl_fctr1,scl_fctr1,0,8,9);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_1x32F:
    {
        __m512 zmm7 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );
        __m512i ex_out;

        // c[0, 0-15]
        SWISH_F32_AVX512_DEF(zmm8, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[0, 16-31]
        SWISH_F32_AVX512_DEF(zmm9, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_1x32F:
      {
        __m512i zmm6;
        // c[0, 0-15]
        TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[0, 16-31]
        TANH_F32S_AVX512(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_SIGMOID_1x32F:
      {
          __m512i ex_out;

          // c[0, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[0, 16-31]
          SIGMOID_F32_AVX512_DEF(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_1x32F_DISABLE:
      ;

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_last_k == TRUE ) )
      {
        uint32_t tlsb, rounded, temp[16] = {0};
        int i;
        bfloat16* dest;

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm8, 0, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm9, 0, 1);
      }
      else
      {
        _mm512_storeu_ps(cbuf, zmm8);
        _mm512_storeu_ps(cbuf + 16, zmm9);
      }
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_5x16)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_5x16F_DISABLE,
              &&POST_OPS_BIAS_5x16F,
              &&POST_OPS_RELU_5x16F,
              &&POST_OPS_RELU_SCALE_5x16F,
              &&POST_OPS_GELU_TANH_5x16F,
              &&POST_OPS_GELU_ERF_5x16F,
              &&POST_OPS_CLIP_5x16F,
              &&POST_OPS_DOWNSCALE_5x16F,
              &&POST_OPS_MATRIX_ADD_5x16F,
              &&POST_OPS_SWISH_5x16F,
              &&POST_OPS_MATRIX_MUL_5x16F,
              &&POST_OPS_TANH_5x16F,
              &&POST_OPS_SIGMOID_5x16F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k0;

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5;
    __m512 zmm8, zmm12;
    __m512 zmm16, zmm20;
    __m512 zmm24;

    /* zero the accumulator registers */
    ZERO_ACC_ZMM_4_REG(zmm8, zmm12, zmm16, zmm20);
    zmm24 = _mm512_setzero_ps();

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;
    float *_cbuf = NULL;

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 32 elements from row0 of B*/
        zmm0 = _mm512_loadu_ps (bbuf );     //load 0-15 values from current row

       /*Broadcast col0 elements of 12 rows of A*/
        zmm2 = _mm512_set1_ps(*(abuf + 0*rs_a)); //broadcast c0r0
        zmm3 = _mm512_set1_ps(*(abuf + 1*rs_a)); //broadcast c0r1
        zmm4 = _mm512_set1_ps(*(abuf + 2*rs_a)); //broadcast c0r2
        zmm5 = _mm512_set1_ps(*(abuf + 3*rs_a)); //broadcast c0r3

        zmm8 = _mm512_fmadd_ps(zmm0, zmm2, zmm8);

        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);

        zmm2 = _mm512_set1_ps(*(abuf + 4*rs_a)); //broadcast c0r4

        zmm16 = _mm512_fmadd_ps(zmm0, zmm4, zmm16);

        zmm20 = _mm512_fmadd_ps(zmm0, zmm5, zmm20);

        zmm24 = _mm512_fmadd_ps(zmm0, zmm2, zmm24);

        bbuf += rs_b;  //move b pointer to next row
        abuf += cs_a;  //move a pointer to next col

    }//kloop

    zmm0 = _mm512_set1_ps(alpha);

    ALPHA_MUL_ACC_ZMM_4_REG(zmm8,zmm12,zmm16,zmm20,zmm0)
    ALPHA_MUL_ACC_ZMM_1_REG(zmm24,zmm0)

    if ( beta != 0.0 )
    {
      zmm3 = _mm512_set1_ps(beta);

      //load c and beta, convert to f32
      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_first_k == TRUE ) )
      {
        //c[0, 0-15]
        BF16_F32_BETA_OP(zmm8, m, 0, 0, zmm0,zmm3);
        //c[1, 0-15]
        BF16_F32_BETA_OP(zmm12, m, 1, 0, zmm0,zmm3);
        //c[2, 0-15]
        BF16_F32_BETA_OP(zmm16, m, 2, 0, zmm0,zmm3);
        //c[3, 0-15]
        BF16_F32_BETA_OP(zmm20, m, 3, 0, zmm0,zmm3);
        //c[4, 0-15]
        BF16_F32_BETA_OP(zmm24, m, 4, 0, zmm0,zmm3);
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        _cbuf += rs_c;

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        _cbuf += rs_c;

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm16 = _mm512_fmadd_ps(zmm0, zmm3, zmm16);
        _cbuf += rs_c;

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm20 = _mm512_fmadd_ps(zmm0, zmm3, zmm20);
        _cbuf += rs_c;

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm24 = _mm512_fmadd_ps(zmm0, zmm3, zmm24);
      }
    }

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_5x16F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_BIAS_LOAD(zmm1, bias_mask, 0)
        }
        else
        {
          zmm1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[1,0-15]
        zmm12 = _mm512_add_ps( zmm1, zmm12 );

        // c[2,0-15]
        zmm16 = _mm512_add_ps( zmm1, zmm16 );

        // c[3,0-15]
        zmm20 = _mm512_add_ps( zmm1, zmm20 );

        // c[4,0-15]
        zmm24 = _mm512_add_ps( zmm1, zmm24 );

      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the bias array will be accessed by
        // the ic index, and each bias element corresponds to an
        // entire row of the transposed output array, instead of an
        // entire column.
        if ( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );

          BF16_F32_BIAS_BCAST(zmm1, bias_mask, 0)
          BF16_F32_BIAS_BCAST(zmm2, bias_mask, 1)
          BF16_F32_BIAS_BCAST(zmm3, bias_mask, 2)
          BF16_F32_BIAS_BCAST(zmm4, bias_mask, 3)
          BF16_F32_BIAS_BCAST(zmm5, bias_mask, 4)
        }
        else
        {
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
          }
        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[1,0-15]
        zmm12 = _mm512_add_ps( zmm2, zmm12 );

        // c[2,0-15]
        zmm16 = _mm512_add_ps( zmm3, zmm16 );

        // c[3,0-15]
        zmm20 = _mm512_add_ps( zmm4, zmm20 );

        // c[4,0-15]
        zmm24 = _mm512_add_ps( zmm5, zmm24 );

      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_5x16F:
    {
      zmm1 = _mm512_setzero_ps();

      // c[0,0-15]
      zmm8 = _mm512_max_ps( zmm1, zmm8 );

      // c[1,0-15]
      zmm12 = _mm512_max_ps( zmm1, zmm12 );

      // c[2,0-15]
      zmm16 = _mm512_max_ps( zmm1, zmm16 );

      // c[3,0-15]
      zmm20 = _mm512_max_ps( zmm1, zmm20 );

      // c[4,0-15]
      zmm24 = _mm512_max_ps( zmm1, zmm24 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_5x16F:
    {
      zmm1 = _mm512_setzero_ps();
      zmm2 =
        _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

      __mmask16 relu_cmp_mask;

      // c[0, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm8)

      // c[1, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm12)

      // c[2, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm16)

      // c[3, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm20)

      // c[4, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm24)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_5x16F:
    {
      __m512i zmm6;
      // c[0, 0-15]
      GELU_TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 0-15]
      GELU_TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[2, 0-15]
      GELU_TANH_F32S_AVX512(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[3, 0-15]
      GELU_TANH_F32S_AVX512(zmm20, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[4, 0-15]
      GELU_TANH_F32S_AVX512(zmm24, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_5x16F:
    {
      // c[0, 0-15]
      GELU_ERF_F32S_AVX512(zmm8, zmm0, zmm1, zmm2)

      // c[1, 0-15]
      GELU_ERF_F32S_AVX512(zmm12, zmm0, zmm1, zmm2)

      // c[2, 0-15]
      GELU_ERF_F32S_AVX512(zmm16, zmm0, zmm1, zmm2)

      // c[3, 0-15]
      GELU_ERF_F32S_AVX512(zmm20, zmm0, zmm1, zmm2)

      // c[4, 0-15]
      GELU_ERF_F32S_AVX512(zmm24, zmm0, zmm1, zmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_5x16F:
    {
      zmm0 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      zmm1 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0, 0-15]
      CLIP_F32S_AVX512(zmm8, zmm0, zmm1)

      // c[1, 0-15]
      CLIP_F32S_AVX512(zmm12, zmm0, zmm1)

      // c[2, 0-15]
      CLIP_F32S_AVX512(zmm16, zmm0, zmm1)

      // c[3, 0-15]
      CLIP_F32S_AVX512(zmm20, zmm0, zmm1)

      // c[4, 0-15]
      CLIP_F32S_AVX512(zmm24, zmm0, zmm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_DOWNSCALE_5x16F:
    {
      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();
      __m512 selector3 = _mm512_setzero_ps();
      __m512 selector4 = _mm512_setzero_ps();
      __m512 selector5 = _mm512_setzero_ps();

      __m512 zero_point0 = _mm512_setzero_ps();
      __m512 zero_point1 = _mm512_setzero_ps();
      __m512 zero_point2 = _mm512_setzero_ps();
      __m512 zero_point3 = _mm512_setzero_ps();
      __m512 zero_point4 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                    ( ( post_ops_list_temp->stor_type == NONE ) &&
                      ( post_ops_attr.c_stor_type == BF16 ) );

      // Need to account for row vs column major swaps. For scalars
      // scale and zero point, no implications.
      // Even though different registers are used for scalar in column
      // and row major downscale path, all those registers will contain
      // the same value.

      if( post_ops_list_temp->scale_factor_len == 1 )
      {
        selector1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if ( is_bf16 == TRUE )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_ZP_BCST(zero_point0,0, zp_mask)
        }
        else
        {
          zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 load_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_LOAD(zero_point0,load_mask, 0)
          }
          else
          {
            zero_point0 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector1, zero_point0);

        //c[2, 0-15]
        F32_SCL_MULRND(zmm16, selector1, zero_point0);

        //c[3, 0-15]
        F32_SCL_MULRND(zmm20, selector1, zero_point0);

        //c[4, 0-15]
        F32_SCL_MULRND(zmm24, selector1, zero_point0);

      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the scale as well as zp array will
        // be accessed by the ic index, and each scale/zp element
        // corresponds to an entire row of the transposed output array,
        // instead of an entire column.
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 0 ) );
          selector2 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 1 ) );
          selector3 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 2 ) );
          selector4 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 3 ) );
          selector5 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 4 ) );
        }
        else
        {
          selector2 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          selector3 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          selector4 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          selector5 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 load_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_LOAD(zero_point0,load_mask, 0)
            BF16_F32_ZP_LOAD(zero_point1,load_mask, 1)
            BF16_F32_ZP_LOAD(zero_point2,load_mask, 2)
            BF16_F32_ZP_LOAD(zero_point3,load_mask, 3)
            BF16_F32_ZP_LOAD(zero_point4,load_mask, 4)
          }
          else
          {
            zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 1) );
            zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 2 ) );
            zero_point3 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 3 ) );
            zero_point4 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 4 ) );
          }
        }
        else
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_BCST(zero_point1,1, zp_mask)
            BF16_F32_ZP_BCST(zero_point2,2, zp_mask)
            BF16_F32_ZP_BCST(zero_point3,3, zp_mask)
            BF16_F32_ZP_BCST(zero_point4,4, zp_mask)
          }
          else
          {
            zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point3 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point4 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector2, zero_point1);

        //c[2, 0-15]
        F32_SCL_MULRND(zmm16, selector3, zero_point2);

        //c[3, 0-15]
        F32_SCL_MULRND(zmm20, selector4, zero_point3);

        //c[4, 0-15]
        F32_SCL_MULRND(zmm24, selector5, zero_point4);

      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_5x16F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      __m512 selector1 = _mm512_setzero_ps();

      __m512 scl_fctr1 = _mm512_setzero_ps();
      __m512 scl_fctr2 = _mm512_setzero_ps();
      __m512 scl_fctr3 = _mm512_setzero_ps();
      __m512 scl_fctr4 = _mm512_setzero_ps();
      __m512 scl_fctr5 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                ( post_ops_attr.c_stor_type == BF16 ) );

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr4 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr5 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
          scl_fctr4 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 3 ) );
          scl_fctr5 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 4 ) );
        }
      }
      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL(selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL(selector1,\
              scl_fctr1,1,12);

          // c[2:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL(selector1,\
              scl_fctr1,2,16);

          // c[3:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL(selector1,\
              scl_fctr1,3,20);

          // c[4:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL(selector1,\
              scl_fctr1,4,24);
        }
        else
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL(selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL(selector1,\
              scl_fctr2,1,12);

          // c[2:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL(selector1,\
              scl_fctr3,2,16);

          // c[3:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL(selector1,\
              scl_fctr4,3,20);

          // c[4:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL(selector1,\
              scl_fctr5,4,24);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM(selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM(selector1,\
              scl_fctr1,1,12);

          // c[2:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM(selector1,\
              scl_fctr1,2,16);

          // c[3:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM(selector1,\
              scl_fctr1,3,20);

          // c[4:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM(selector1,\
              scl_fctr1,4,24);
        }
        else
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM(selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM(selector1,\
              scl_fctr2,1,12);

          // c[2:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM(selector1,\
              scl_fctr3,2,16);

          // c[3:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM(selector1,\
              scl_fctr4,3,20);

          // c[4:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM(selector1,\
              scl_fctr5,4,24);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_5x16F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      __m512 selector1 = _mm512_setzero_ps();

      __m512 scl_fctr1 = _mm512_setzero_ps();
      __m512 scl_fctr2 = _mm512_setzero_ps();
      __m512 scl_fctr3 = _mm512_setzero_ps();
      __m512 scl_fctr4 = _mm512_setzero_ps();
      __m512 scl_fctr5 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                    ( post_ops_attr.c_stor_type == BF16 ) );

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr4 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr5 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
              ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
          scl_fctr4 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 3 ) );
          scl_fctr5 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 4 ) );
        }
      }
      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL(selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL(selector1,\
              scl_fctr1,1,12);

          // c[2:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL(selector1,\
              scl_fctr1,2,16);

          // c[3:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL(selector1,\
              scl_fctr1,3,20);

          // c[4:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL(selector1,\
              scl_fctr1,4,24);
        }
        else
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL(selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL(selector1,\
              scl_fctr2,1,12);

          // c[2:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL(selector1,\
              scl_fctr3,2,16);

          // c[3:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL(selector1,\
              scl_fctr4,3,20);

          // c[4:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL(selector1,\
              scl_fctr5,4,24);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM(selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM(selector1,\
              scl_fctr1,1,12);

          // c[2:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM(selector1,\
              scl_fctr1,2,16);

          // c[3:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM(selector1,\
              scl_fctr1,3,20);

          // c[4:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM(selector1,\
              scl_fctr1,4,24);
        }
        else
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM(selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM(selector1,\
              scl_fctr2,1,12);

          // c[2:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM(selector1,\
              scl_fctr3,2,16);

          // c[3:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM(selector1,\
              scl_fctr4,3,20);

          // c[4:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM(selector1,\
              scl_fctr5,4,24);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_5x16F:
    {
        __m512 zmm7 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );
        __m512i ex_out;

        // c[0, 0-15]
        SWISH_F32_AVX512_DEF(zmm8, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 0-15]
        SWISH_F32_AVX512_DEF(zmm12, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[2, 0-15]
        SWISH_F32_AVX512_DEF(zmm16, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[3, 0-15]
        SWISH_F32_AVX512_DEF(zmm20, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[4, 0-15]
        SWISH_F32_AVX512_DEF(zmm24, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_5x16F:
      {
        __m512i zmm6;
        // c[0, 0-15]
        TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 0-15]
        TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[2, 0-15]
        TANH_F32S_AVX512(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[3, 0-15]
        TANH_F32S_AVX512(zmm20, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[4, 0-15]
        TANH_F32S_AVX512(zmm24, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_SIGMOID_5x16F:
      {
          __m512i ex_out;

          // c[0, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[3, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm20, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[4, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm24, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_5x16F_DISABLE:
      ;

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_last_k == TRUE ) )
      {
        uint32_t tlsb, rounded, temp[16] = {0};
        int i;
        bfloat16* dest;

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm8, 0, 0);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm12, 1, 0);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm16, 2, 0);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm20, 3, 0);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm24, 4, 0);
      }
      else
      {
        _mm512_storeu_ps(cbuf, zmm8);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm12);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm16);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm20);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm24);
      }
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_4x16)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_4x16F_DISABLE,
              &&POST_OPS_BIAS_4x16F,
              &&POST_OPS_RELU_4x16F,
              &&POST_OPS_RELU_SCALE_4x16F,
              &&POST_OPS_GELU_TANH_4x16F,
              &&POST_OPS_GELU_ERF_4x16F,
              &&POST_OPS_CLIP_4x16F,
              &&POST_OPS_DOWNSCALE_4x16F,
              &&POST_OPS_MATRIX_ADD_4x16F,
              &&POST_OPS_SWISH_4x16F,
              &&POST_OPS_MATRIX_MUL_4x16F,
              &&POST_OPS_TANH_4x16F,
              &&POST_OPS_SIGMOID_4x16F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k0;

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5;
    __m512 zmm8, zmm12;
    __m512 zmm16, zmm20;

    /* zero the accumulator registers */
    ZERO_ACC_ZMM_4_REG(zmm8, zmm12, zmm16, zmm20);

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;
    float *_cbuf = NULL;

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 32 elements from row0 of B*/
        zmm0 = _mm512_loadu_ps (bbuf );     //load 0-15 values from current row

       /*Broadcast col0 elements of 12 rows of A*/
        zmm2 = _mm512_set1_ps(*(abuf + 0*rs_a)); //broadcast c0r0
        zmm3 = _mm512_set1_ps(*(abuf + 1*rs_a)); //broadcast c0r1
        zmm4 = _mm512_set1_ps(*(abuf + 2*rs_a)); //broadcast c0r2
        zmm5 = _mm512_set1_ps(*(abuf + 3*rs_a)); //broadcast c0r3

        zmm8 = _mm512_fmadd_ps(zmm0, zmm2, zmm8);

        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);

        zmm16 = _mm512_fmadd_ps(zmm0, zmm4, zmm16);

        zmm20 = _mm512_fmadd_ps(zmm0, zmm5, zmm20);

        bbuf += rs_b;  //move b pointer to next row
        abuf += cs_a;  //move a pointer to next col

    }//kloop

    zmm0 = _mm512_set1_ps(alpha);

    ALPHA_MUL_ACC_ZMM_4_REG(zmm8,zmm12,zmm16,zmm20,zmm0)

    if ( beta != 0.0 )
    {
      zmm3 = _mm512_set1_ps(beta);

      //load c and beta, convert to f32
      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_first_k == TRUE ) )
      {
        //c[0, 0-15]
        BF16_F32_BETA_OP(zmm8, m, 0, 0, zmm0,zmm3);
        //c[1, 0-15]
        BF16_F32_BETA_OP(zmm12, m, 1, 0, zmm0,zmm3);
        //c[2, 0-15]
        BF16_F32_BETA_OP(zmm16, m, 2, 0, zmm0,zmm3);
        //c[3, 0-15]
        BF16_F32_BETA_OP(zmm20, m, 3, 0, zmm0,zmm3);
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        _cbuf += rs_c;

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        _cbuf += rs_c;

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm16 = _mm512_fmadd_ps(zmm0, zmm3, zmm16);
        _cbuf += rs_c;

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm20 = _mm512_fmadd_ps(zmm0, zmm3, zmm20);
        _cbuf += rs_c;

      }
    }

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_4x16F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_BIAS_LOAD(zmm1, bias_mask, 0)
        }
        else
        {
          zmm1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[1,0-15]
        zmm12 = _mm512_add_ps( zmm1, zmm12 );

        // c[2,0-15]
        zmm16 = _mm512_add_ps( zmm1, zmm16 );

        // c[3,0-15]
        zmm20 = _mm512_add_ps( zmm1, zmm20 );

      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the bias array will be accessed by
        // the ic index, and each bias element corresponds to an
        // entire row of the transposed output array, instead of an
        // entire column.
        if ( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );

          BF16_F32_BIAS_BCAST(zmm1, bias_mask, 0)
          BF16_F32_BIAS_BCAST(zmm2, bias_mask, 1)
          BF16_F32_BIAS_BCAST(zmm3, bias_mask, 2)
          BF16_F32_BIAS_BCAST(zmm4, bias_mask, 3)
        }
        else
        {
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
        }
        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[1,0-15]
        zmm12 = _mm512_add_ps( zmm2, zmm12 );

        // c[2,0-15]
        zmm16 = _mm512_add_ps( zmm3, zmm16 );

        // c[3,0-15]
        zmm20 = _mm512_add_ps( zmm4, zmm20 );

      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_4x16F:
    {
      zmm1 = _mm512_setzero_ps();

      // c[0,0-15]
      zmm8 = _mm512_max_ps( zmm1, zmm8 );

      // c[1,0-15]
      zmm12 = _mm512_max_ps( zmm1, zmm12 );

      // c[2,0-15]
      zmm16 = _mm512_max_ps( zmm1, zmm16 );

      // c[3,0-15]
      zmm20 = _mm512_max_ps( zmm1, zmm20 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_4x16F:
    {
      zmm1 = _mm512_setzero_ps();
      zmm2 =
        _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

      __mmask16 relu_cmp_mask;

      // c[0, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm8)

      // c[1, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm12)

      // c[2, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm16)

      // c[3, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm20)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_4x16F:
    {
      __m512i zmm6;
      // c[0, 0-15]
      GELU_TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 0-15]
      GELU_TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[2, 0-15]
      GELU_TANH_F32S_AVX512(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[3, 0-15]
      GELU_TANH_F32S_AVX512(zmm20, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_4x16F:
    {
      // c[0, 0-15]
      GELU_ERF_F32S_AVX512(zmm8, zmm0, zmm1, zmm2)

      // c[1, 0-15]
      GELU_ERF_F32S_AVX512(zmm12, zmm0, zmm1, zmm2)

      // c[2, 0-15]
      GELU_ERF_F32S_AVX512(zmm16, zmm0, zmm1, zmm2)

      // c[3, 0-15]
      GELU_ERF_F32S_AVX512(zmm20, zmm0, zmm1, zmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_4x16F:
    {
      zmm0 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      zmm1 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0, 0-15]
      CLIP_F32S_AVX512(zmm8, zmm0, zmm1)

      // c[1, 0-15]
      CLIP_F32S_AVX512(zmm12, zmm0, zmm1)

      // c[2, 0-15]
      CLIP_F32S_AVX512(zmm16, zmm0, zmm1)

      // c[3, 0-15]
      CLIP_F32S_AVX512(zmm20, zmm0, zmm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_DOWNSCALE_4x16F:
    {
      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();
      __m512 selector3 = _mm512_setzero_ps();
      __m512 selector4 = _mm512_setzero_ps();

      __m512 zero_point0 = _mm512_setzero_ps();
      __m512 zero_point1 = _mm512_setzero_ps();
      __m512 zero_point2 = _mm512_setzero_ps();
      __m512 zero_point3 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                    ( ( post_ops_list_temp->stor_type == NONE ) &&
                      ( post_ops_attr.c_stor_type == BF16 ) );

      // Need to account for row vs column major swaps. For scalars
      // scale and zero point, no implications.
      // Even though different registers are used for scalar in column
      // and row major downscale path, all those registers will contain
      // the same value.

      if( post_ops_list_temp->scale_factor_len == 1 )
      {
        selector1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if ( is_bf16 == TRUE )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_ZP_BCST(zero_point0,0, zp_mask)
        }
        else
        {
          zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 load_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_LOAD(zero_point0,load_mask, 0)
          }
          else
          {
            zero_point0 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector1, zero_point0);

        //c[2, 0-15]
        F32_SCL_MULRND(zmm16, selector1, zero_point0);

        //c[3, 0-15]
        F32_SCL_MULRND(zmm20, selector1, zero_point0);

      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the scale as well as zp array will
        // be accessed by the ic index, and each scale/zp element
        // corresponds to an entire row of the transposed output array,
        // instead of an entire column.
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 0 ) );
          selector2 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 1 ) );
          selector3 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 2 ) );
          selector4 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 3 ) );
        }
        else
        {
          selector2 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          selector3 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          selector4 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 load_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_LOAD(zero_point0,load_mask, 0)
            BF16_F32_ZP_LOAD(zero_point1,load_mask, 1)
            BF16_F32_ZP_LOAD(zero_point2,load_mask, 2)
            BF16_F32_ZP_LOAD(zero_point3,load_mask, 3)
          }
          else
          {
            zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 1) );
            zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 2 ) );
            zero_point3 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 3 ) );
          }
        }
        else
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_BCST(zero_point1,1, zp_mask)
            BF16_F32_ZP_BCST(zero_point2,2, zp_mask)
            BF16_F32_ZP_BCST(zero_point3,3, zp_mask)
          }
          else
          {
            zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point3 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector2, zero_point1);

        //c[2, 0-15]
        F32_SCL_MULRND(zmm16, selector3, zero_point2);

        //c[3, 0-15]
        F32_SCL_MULRND(zmm20, selector4, zero_point3);

      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_4x16F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      __m512 selector1 = _mm512_setzero_ps();

      __m512 scl_fctr1 = _mm512_setzero_ps();
      __m512 scl_fctr2 = _mm512_setzero_ps();
      __m512 scl_fctr3 = _mm512_setzero_ps();
      __m512 scl_fctr4 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                ( post_ops_attr.c_stor_type == BF16 ) );

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr4 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
          scl_fctr4 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 3 ) );
        }
      }
      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL(selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL(selector1,\
              scl_fctr1,1,12);

          // c[2:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL(selector1,\
              scl_fctr1,2,16);

          // c[3:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL(selector1,\
              scl_fctr1,3,20);
        }
        else
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL(selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL(selector1,\
              scl_fctr2,1,12);

          // c[2:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL(selector1,\
              scl_fctr3,2,16);

          // c[3:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL(selector1,\
              scl_fctr4,3,20);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM(selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM(selector1,\
              scl_fctr1,1,12);

          // c[2:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM(selector1,\
              scl_fctr1,2,16);

          // c[3:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM(selector1,\
              scl_fctr1,3,20);
        }
        else
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM(selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM(selector1,\
              scl_fctr2,1,12);

          // c[2:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM(selector1,\
              scl_fctr3,2,16);

          // c[3:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM(selector1,\
              scl_fctr4,3,20);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_4x16F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      __m512 selector1 = _mm512_setzero_ps();

      __m512 scl_fctr1 = _mm512_setzero_ps();
      __m512 scl_fctr2 = _mm512_setzero_ps();
      __m512 scl_fctr3 = _mm512_setzero_ps();
      __m512 scl_fctr4 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
      ( ( post_ops_list_temp->stor_type == NONE ) &&
          ( post_ops_attr.c_stor_type == BF16 ) );

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr4 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
              ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
          scl_fctr4 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 3 ) );
        }
      }
      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL(selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL(selector1,\
              scl_fctr1,1,12);

          // c[2:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL(selector1,\
              scl_fctr1,2,16);

          // c[3:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL(selector1,\
              scl_fctr1,3,20);
        }
        else
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL(selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL(selector1,\
              scl_fctr2,1,12);

          // c[2:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL(selector1,\
              scl_fctr3,2,16);

          // c[3:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL(selector1,\
              scl_fctr4,3,20);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM(selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM(selector1,\
              scl_fctr1,1,12);

          // c[2:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM(selector1,\
              scl_fctr1,2,16);

          // c[3:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM(selector1,\
              scl_fctr1,3,20);
        }
        else
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM(selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM(selector1,\
              scl_fctr2,1,12);

          // c[2:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM(selector1,\
              scl_fctr3,2,16);

          // c[3:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM(selector1,\
              scl_fctr4,3,20);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_4x16F:
    {
        __m512 zmm7 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );
        __m512i ex_out;

        // c[0, 0-15]
        SWISH_F32_AVX512_DEF(zmm8, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 0-15]
        SWISH_F32_AVX512_DEF(zmm12, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[2, 0-15]
        SWISH_F32_AVX512_DEF(zmm16, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[3, 0-15]
        SWISH_F32_AVX512_DEF(zmm20, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_4x16F:
      {
        __m512i zmm6;
        // c[0, 0-15]
        TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 0-15]
        TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[2, 0-15]
        TANH_F32S_AVX512(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[3, 0-15]
        TANH_F32S_AVX512(zmm20, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_SIGMOID_4x16F:
      {
          __m512i ex_out;

          // c[0, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[3, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm20, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_4x16F_DISABLE:
      ;

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_last_k == TRUE ) )
      {
        uint32_t tlsb, rounded, temp[16] = {0};
        int i;
        bfloat16* dest;

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm8, 0, 0);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm12, 1, 0);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm16, 2, 0);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm20, 3, 0);

      }
      else
      {
        _mm512_storeu_ps(cbuf, zmm8);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm12);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm16);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm20);
      }
}


LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_3x16)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_3x16F_DISABLE,
              &&POST_OPS_BIAS_3x16F,
              &&POST_OPS_RELU_3x16F,
              &&POST_OPS_RELU_SCALE_3x16F,
              &&POST_OPS_GELU_TANH_3x16F,
              &&POST_OPS_GELU_ERF_3x16F,
              &&POST_OPS_CLIP_3x16F,
              &&POST_OPS_DOWNSCALE_3x16F,
              &&POST_OPS_MATRIX_ADD_3x16F,
              &&POST_OPS_SWISH_3x16F,
              &&POST_OPS_MATRIX_MUL_3x16F,
              &&POST_OPS_TANH_3x16F,
              &&POST_OPS_SIGMOID_3x16F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k0;

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5;
    __m512 zmm8, zmm12;
    __m512 zmm16;

    /* zero the accumulator registers */
    ZERO_ACC_ZMM_2_REG(zmm8, zmm12);
    zmm16 = _mm512_setzero_ps();

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;
    float *_cbuf = NULL;

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 32 elements from row0 of B*/
        zmm0 = _mm512_loadu_ps (bbuf );     //load 0-15 values from current row

       /*Broadcast col0 elements of 12 rows of A*/
        zmm2 = _mm512_set1_ps(*(abuf + 0*rs_a)); //broadcast c0r0
        zmm3 = _mm512_set1_ps(*(abuf + 1*rs_a)); //broadcast c0r1
        zmm4 = _mm512_set1_ps(*(abuf + 2*rs_a)); //broadcast c0r2

        zmm8 = _mm512_fmadd_ps(zmm0, zmm2, zmm8);

        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);

        zmm16 = _mm512_fmadd_ps(zmm0, zmm4, zmm16);

        bbuf += rs_b;  //move b pointer to next row
        abuf += cs_a;  //move a pointer to next col

    }//kloop

    zmm0 = _mm512_set1_ps(alpha);

    ALPHA_MUL_ACC_ZMM_2_REG(zmm8,zmm12,zmm0)
    ALPHA_MUL_ACC_ZMM_1_REG(zmm16,zmm0)

    if ( beta != 0.0 )
    {
      zmm3 = _mm512_set1_ps(beta);

      //load c and beta, convert to f32
      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_first_k == TRUE ) )
      {
        //c[0, 0-15]
        BF16_F32_BETA_OP(zmm8, m, 0, 0, zmm0,zmm3);
        //c[1, 0-15]
        BF16_F32_BETA_OP(zmm12, m, 1, 0, zmm0,zmm3);
        //c[2, 0-15]
        BF16_F32_BETA_OP(zmm16, m, 2, 0, zmm0,zmm3);
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        _cbuf += rs_c;

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        _cbuf += rs_c;

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm16 = _mm512_fmadd_ps(zmm0, zmm3, zmm16);
        _cbuf += rs_c;
      }
    }

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_3x16F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_BIAS_LOAD(zmm1, bias_mask, 0)
        }
        else
        {
          zmm1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[1,0-15]
        zmm12 = _mm512_add_ps( zmm1, zmm12 );

        // c[2,0-15]
        zmm16 = _mm512_add_ps( zmm1, zmm16 );

      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the bias array will be accessed by
        // the ic index, and each bias element corresponds to an
        // entire row of the transposed output array, instead of an
        // entire column.
        if ( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );

          BF16_F32_BIAS_BCAST(zmm1, bias_mask, 0)
          BF16_F32_BIAS_BCAST(zmm2, bias_mask, 1)
          BF16_F32_BIAS_BCAST(zmm3, bias_mask, 2)
        }
        else
        {
          zmm1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 0 ) );
          zmm2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 1 ) );
          zmm3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 2 ) );
        }
        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[1,0-15]
        zmm12 = _mm512_add_ps( zmm2, zmm12 );

        // c[2,0-15]
        zmm16 = _mm512_add_ps( zmm3, zmm16 );

      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_3x16F:
    {
      zmm1 = _mm512_setzero_ps();

      // c[0,0-15]
      zmm8 = _mm512_max_ps( zmm1, zmm8 );

      // c[1,0-15]
      zmm12 = _mm512_max_ps( zmm1, zmm12 );

      // c[2,0-15]
      zmm16 = _mm512_max_ps( zmm1, zmm16 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_3x16F:
    {
      zmm1 = _mm512_setzero_ps();
      zmm2 =
        _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

      __mmask16 relu_cmp_mask;

      // c[0, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm8)

      // c[1, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm12)

      // c[2, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm16)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_3x16F:
    {
      __m512i zmm6;
      // c[0, 0-15]
      GELU_TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 0-15]
      GELU_TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[2, 0-15]
      GELU_TANH_F32S_AVX512(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_3x16F:
    {
      // c[0, 0-15]
      GELU_ERF_F32S_AVX512(zmm8, zmm0, zmm1, zmm2)

      // c[1, 0-15]
      GELU_ERF_F32S_AVX512(zmm12, zmm0, zmm1, zmm2)

      // c[2, 0-15]
      GELU_ERF_F32S_AVX512(zmm16, zmm0, zmm1, zmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_3x16F:
    {
      zmm0 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      zmm1 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0, 0-15]
      CLIP_F32S_AVX512(zmm8, zmm0, zmm1)

      // c[1, 0-15]
      CLIP_F32S_AVX512(zmm12, zmm0, zmm1)

      // c[2, 0-15]
      CLIP_F32S_AVX512(zmm16, zmm0, zmm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_DOWNSCALE_3x16F:
    {
      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();
      __m512 selector3 = _mm512_setzero_ps();

      __m512 zero_point0 = _mm512_setzero_ps();
      __m512 zero_point1 = _mm512_setzero_ps();
      __m512 zero_point2 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                    ( ( post_ops_list_temp->stor_type == NONE ) &&
                      ( post_ops_attr.c_stor_type == BF16 ) );

      // Need to account for row vs column major swaps. For scalars
      // scale and zero point, no implications.
      // Even though different registers are used for scalar in column
      // and row major downscale path, all those registers will contain
      // the same value.

      if( post_ops_list_temp->scale_factor_len == 1 )
      {
        selector1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if ( is_bf16 == TRUE )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_ZP_BCST(zero_point0,0, zp_mask)
        }
        else
        {
          zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 load_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_LOAD(zero_point0,load_mask, 0)
          }
          else
          {
            zero_point0 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector1, zero_point0);

        //c[2, 0-15]
        F32_SCL_MULRND(zmm16, selector1, zero_point0);

      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the scale as well as zp array will
        // be accessed by the ic index, and each scale/zp element
        // corresponds to an entire row of the transposed output array,
        // instead of an entire column.
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 0 ) );
          selector2 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 1 ) );
          selector3 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 2 ) );
        }
        else
        {
          selector2 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          selector3 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 load_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_LOAD(zero_point0,load_mask, 0)
            BF16_F32_ZP_LOAD(zero_point1,load_mask, 1)
            BF16_F32_ZP_LOAD(zero_point2,load_mask, 2)
          }
          else
          {
            zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 1) );
            zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 2 ) );
          }
        }
        else
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_BCST(zero_point1,1, zp_mask)
            BF16_F32_ZP_BCST(zero_point2,2, zp_mask)
          }
          else
          {
            zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector2, zero_point1);

        //c[2, 0-15]
        F32_SCL_MULRND(zmm16, selector3, zero_point2);

      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_3x16F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      __m512 selector1 = _mm512_setzero_ps();

      __m512 scl_fctr1 = _mm512_setzero_ps();
      __m512 scl_fctr2 = _mm512_setzero_ps();
      __m512 scl_fctr3 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                ( post_ops_attr.c_stor_type == BF16 ) );

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
        }
      }
      if( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL(selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL(selector1,\
              scl_fctr1,1,12);

          // c[2:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL(selector1,\
              scl_fctr1,2,16);
        }
        else
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL(selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL(selector1,\
              scl_fctr2,1,12);

          // c[2:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL(selector1,\
              scl_fctr3,2,16);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM(selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM(selector1,\
              scl_fctr1,1,12);

          // c[2:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM(selector1,\
              scl_fctr1,2,16);
        }
        else
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM(selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM(selector1,\
              scl_fctr2,1,12);

          // c[2:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM(selector1,\
              scl_fctr3,2,16);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_3x16F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      __m512 selector1 = _mm512_setzero_ps();

      __m512 scl_fctr1 = _mm512_setzero_ps();
      __m512 scl_fctr2 = _mm512_setzero_ps();
      __m512 scl_fctr3 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                    ( post_ops_attr.c_stor_type == BF16 ) );

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
              ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
        }
      }
      if( is_bf16 == TRUE)
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL(selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL(selector1,\
              scl_fctr1,1,12);

          // c[2:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL(selector1,\
              scl_fctr1,2,16);
        }
        else
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL(selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL(selector1,\
              scl_fctr2,1,12);

          // c[2:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL(selector1,\
              scl_fctr3,2,16);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM(selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM(selector1,\
              scl_fctr1,1,12);

          // c[2:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM(selector1,\
              scl_fctr1,2,16);

        }
        else
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM(selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM(selector1,\
              scl_fctr2,1,12);

          // c[2:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM(selector1,\
              scl_fctr3,2,16);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_3x16F:
    {
        __m512 zmm7 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );
        __m512i ex_out;

        // c[0, 0-15]
        SWISH_F32_AVX512_DEF(zmm8, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 0-15]
        SWISH_F32_AVX512_DEF(zmm12, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[2, 0-15]
        SWISH_F32_AVX512_DEF(zmm16, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_3x16F:
      {
        __m512i zmm6;
        // c[0, 0-15]
        TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 0-15]
        TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[2, 0-15]
        TANH_F32S_AVX512(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_SIGMOID_3x16F:
      {
          __m512i ex_out;

          // c[0, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_3x16F_DISABLE:
      ;

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_last_k == TRUE ) )
      {
        uint32_t tlsb, rounded, temp[16] = {0};
        int i;
        bfloat16* dest;

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm8, 0, 0);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm12, 1, 0);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm16, 2, 0);

      }
      else
      {
        _mm512_storeu_ps(cbuf, zmm8);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm12);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm16);
      }
}


LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_2x16)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_2x16F_DISABLE,
              &&POST_OPS_BIAS_2x16F,
              &&POST_OPS_RELU_2x16F,
              &&POST_OPS_RELU_SCALE_2x16F,
              &&POST_OPS_GELU_TANH_2x16F,
              &&POST_OPS_GELU_ERF_2x16F,
              &&POST_OPS_CLIP_2x16F,
              &&POST_OPS_DOWNSCALE_2x16F,
              &&POST_OPS_MATRIX_ADD_2x16F,
              &&POST_OPS_SWISH_2x16F,
              &&POST_OPS_MATRIX_MUL_2x16F,
              &&POST_OPS_TANH_2x16F,
              &&POST_OPS_SIGMOID_2x16F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k0;

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5;
    __m512 zmm8, zmm12;

    /* zero the accumulator registers */
    ZERO_ACC_ZMM_2_REG(zmm8, zmm12);

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;
    float *_cbuf = NULL;

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 32 elements from row0 of B*/
        zmm0 = _mm512_loadu_ps (bbuf );     //load 0-15 values from current row

       /*Broadcast col0 elements of 12 rows of A*/
        zmm2 = _mm512_set1_ps(*(abuf + 0*rs_a)); //broadcast c0r0
        zmm3 = _mm512_set1_ps(*(abuf + 1*rs_a)); //broadcast c0r1

        zmm8 = _mm512_fmadd_ps(zmm0, zmm2, zmm8);

        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);

        bbuf += rs_b;  //move b pointer to next row
        abuf += cs_a;  //move a pointer to next col

    }//kloop

    zmm0 = _mm512_set1_ps(alpha);

    ALPHA_MUL_ACC_ZMM_2_REG(zmm8,zmm12,zmm0)

    if ( beta != 0.0 )
    {
      zmm3 = _mm512_set1_ps(beta);

      //load c and beta, convert to f32
      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_first_k == TRUE ) )
      {
        //c[0, 0-15]
        BF16_F32_BETA_OP(zmm8, m, 0, 0, zmm0,zmm3);
        //c[1, 0-15]
        BF16_F32_BETA_OP(zmm12, m, 1, 0, zmm0,zmm3);
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        _cbuf += rs_c;

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        _cbuf += rs_c;
      }
    }

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_2x16F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_BIAS_LOAD(zmm1, bias_mask, 0)
        }
        else
        {
          zmm1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[1,0-15]
        zmm12 = _mm512_add_ps( zmm1, zmm12 );

      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the bias array will be accessed by
        // the ic index, and each bias element corresponds to an
        // entire row of the transposed output array, instead of an
        // entire column.
        if ( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );

          BF16_F32_BIAS_BCAST(zmm1, bias_mask, 0)
          BF16_F32_BIAS_BCAST(zmm2, bias_mask, 1)
        }
        else
        {
            zmm1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                post_ops_attr.post_op_c_i + 0 ) );
            zmm2 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                post_ops_attr.post_op_c_i + 1 ) );
        }
        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[1,0-15]
        zmm12 = _mm512_add_ps( zmm2, zmm12 );

      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_2x16F:
    {
      zmm1 = _mm512_setzero_ps();

      // c[0,0-15]
      zmm8 = _mm512_max_ps( zmm1, zmm8 );

      // c[1,0-15]
      zmm12 = _mm512_max_ps( zmm1, zmm12 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_2x16F:
    {
      zmm1 = _mm512_setzero_ps();
      zmm2 =
        _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

      __mmask16 relu_cmp_mask;

      // c[0, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm8)

      // c[1, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm12)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_2x16F:
    {
      __m512i zmm6;
      // c[0, 0-15]
      GELU_TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 0-15]
      GELU_TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_2x16F:
    {
      // c[0, 0-15]
      GELU_ERF_F32S_AVX512(zmm8, zmm0, zmm1, zmm2)

      // c[1, 0-15]
      GELU_ERF_F32S_AVX512(zmm12, zmm0, zmm1, zmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_2x16F:
    {
      zmm0 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      zmm1 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0, 0-15]
      CLIP_F32S_AVX512(zmm8, zmm0, zmm1)

      // c[1, 0-15]
      CLIP_F32S_AVX512(zmm12, zmm0, zmm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_DOWNSCALE_2x16F:
    {
      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();

      __m512 zero_point0 = _mm512_setzero_ps();
      __m512 zero_point1 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                    ( ( post_ops_list_temp->stor_type == NONE ) &&
                      ( post_ops_attr.c_stor_type == BF16 ) );

      // Need to account for row vs column major swaps. For scalars
      // scale and zero point, no implications.
      // Even though different registers are used for scalar in column
      // and row major downscale path, all those registers will contain
      // the same value.

      if( post_ops_list_temp->scale_factor_len == 1 )
      {
        selector1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if ( is_bf16 == TRUE )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_ZP_BCST(zero_point0,0, zp_mask)
        }
        else
        {
          zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 load_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_LOAD(zero_point0,load_mask, 0)
          }
          else
          {
            zero_point0 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector1, zero_point0);

      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the scale as well as zp array will
        // be accessed by the ic index, and each scale/zp element
        // corresponds to an entire row of the transposed output array,
        // instead of an entire column.
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 0 ) );
          selector2 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 1 ) );
        }
        else
        {
          selector2 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 load_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_LOAD(zero_point0,load_mask, 0)
            BF16_F32_ZP_LOAD(zero_point1,load_mask, 1)
          }
          else
          {
            zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 1) );
          }
        }
        else
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_BCST(zero_point1,1, zp_mask)
          }
          else
          {
            zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector2, zero_point1);

      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_2x16F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      __m512 selector1 = _mm512_setzero_ps();

      __m512 scl_fctr1 = _mm512_setzero_ps();
      __m512 scl_fctr2 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                ( post_ops_attr.c_stor_type == BF16 ) );

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
        }
      }
      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL(selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL(selector1,\
              scl_fctr1,1,12);

        }
        else
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL(selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL(selector1,\
              scl_fctr2,1,12);

        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM(selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM(selector1,\
              scl_fctr1,1,12);
        }
        else
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM(selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM(selector1,\
              scl_fctr2,1,12);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_2x16F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      __m512 selector1 = _mm512_setzero_ps();

      __m512 scl_fctr1 = _mm512_setzero_ps();
      __m512 scl_fctr2 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                    ( post_ops_attr.c_stor_type == BF16 ) );

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
              ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
        }
      }
      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL(selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL(selector1,\
              scl_fctr1,1,12);
        }
        else
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL(selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL(selector1,\
              scl_fctr2,1,12);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM(selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM(selector1,\
              scl_fctr1,1,12);
        }
        else
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM(selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM(selector1,\
              scl_fctr2,1,12);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_2x16F:
    {
        __m512 zmm7 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );
        __m512i ex_out;

        // c[0, 0-15]
        SWISH_F32_AVX512_DEF(zmm8, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 0-15]
        SWISH_F32_AVX512_DEF(zmm12, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_2x16F:
      {
        __m512i zmm6;
        // c[0, 0-15]
        TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 0-15]
        TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_SIGMOID_2x16F:
      {
          __m512i ex_out;

          // c[0, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_2x16F_DISABLE:
      ;

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_last_k == TRUE ) )
      {
        uint32_t tlsb, rounded, temp[16] = {0};
        int i;
        bfloat16* dest;

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm8, 0, 0);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm12, 1, 0);

      }
      else
      {
        _mm512_storeu_ps(cbuf, zmm8);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm12);
      }
}


LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_1x16)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_1x16F_DISABLE,
              &&POST_OPS_BIAS_1x16F,
              &&POST_OPS_RELU_1x16F,
              &&POST_OPS_RELU_SCALE_1x16F,
              &&POST_OPS_GELU_TANH_1x16F,
              &&POST_OPS_GELU_ERF_1x16F,
              &&POST_OPS_CLIP_1x16F,
              &&POST_OPS_DOWNSCALE_1x16F,
              &&POST_OPS_MATRIX_ADD_1x16F,
              &&POST_OPS_SWISH_1x16F,
              &&POST_OPS_MATRIX_MUL_1x16F,
              &&POST_OPS_TANH_1x16F,
              &&POST_OPS_SIGMOID_1x16F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k0;

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5;
    __m512 zmm8;

    /* zero the accumulator registers */
    zmm8 = _mm512_setzero_ps();

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;
    float *_cbuf = NULL;

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 32 elements from row0 of B*/
        zmm0 = _mm512_loadu_ps (bbuf );     //load 0-15 values from current row

       /*Broadcast col0 elements of 12 rows of A*/
        zmm2 = _mm512_set1_ps(*(abuf + 0*rs_a)); //broadcast c0r0

        zmm8 = _mm512_fmadd_ps(zmm0, zmm2, zmm8);

        bbuf += rs_b;  //move b pointer to next row
        abuf += cs_a;  //move a pointer to next col

    }//kloop

    zmm0 = _mm512_set1_ps(alpha);

    ALPHA_MUL_ACC_ZMM_1_REG(zmm8,zmm0)

    if ( beta != 0.0 )
    {
      zmm3 = _mm512_set1_ps(beta);

      //load c and beta, convert to f32
      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_first_k == TRUE ) )
      {
        //c[0, 0-15]
        BF16_F32_BETA_OP(zmm8, m, 0, 0, zmm0,zmm3);
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back

        zmm0 = _mm512_loadu_ps(_cbuf);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        _cbuf += rs_c;
      }
    }

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_1x16F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_BIAS_LOAD(zmm1, bias_mask, 0)
        }
        else
        {
          zmm1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the bias array will be accessed by
        // the ic index, and each bias element corresponds to an
        // entire row of the transposed output array, instead of an
        // entire column.
        if ( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );

          BF16_F32_BIAS_BCAST(zmm1, bias_mask, 0)
        }
        else
        {
          zmm1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 0 ) );
        }
        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_1x16F:
    {
      zmm1 = _mm512_setzero_ps();

      // c[0,0-15]
      zmm8 = _mm512_max_ps( zmm1, zmm8 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_1x16F:
    {
      zmm1 = _mm512_setzero_ps();
      zmm2 =
        _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

      __mmask16 relu_cmp_mask;

      // c[0, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm8)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_1x16F:
    {
      __m512i zmm6;
      // c[0, 0-15]
      GELU_TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_1x16F:
    {
      // c[0, 0-15]
      GELU_ERF_F32S_AVX512(zmm8, zmm0, zmm1, zmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_1x16F:
    {
      zmm0 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      zmm1 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0, 0-15]
      CLIP_F32S_AVX512(zmm8, zmm0, zmm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_DOWNSCALE_1x16F:
    {
      __m512 selector1 = _mm512_setzero_ps();

      __m512 zero_point0 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                    ( ( post_ops_list_temp->stor_type == NONE ) &&
                      ( post_ops_attr.c_stor_type == BF16 ) );

      // Need to account for row vs column major swaps. For scalars
      // scale and zero point, no implications.
      // Even though different registers are used for scalar in column
      // and row major downscale path, all those registers will contain
      // the same value.

      if( post_ops_list_temp->scale_factor_len == 1 )
      {
        selector1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if ( is_bf16 == TRUE )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_ZP_BCST(zero_point0,0, zp_mask)
        }
        else
        {
          zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 load_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_LOAD(zero_point0,load_mask, 0)
          }
          else
          {
            zero_point0 = _mm512_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the scale as well as zp array will
        // be accessed by the ic index, and each scale/zp element
        // corresponds to an entire row of the transposed output array,
        // instead of an entire column.
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 0 ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 load_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_LOAD(zero_point0,load_mask, 0)
          }
          else
          {
            zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 0 ) );
          }
        }

        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_1x16F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      __m512 selector1 = _mm512_setzero_ps();

      __m512 scl_fctr1 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                ( post_ops_attr.c_stor_type == BF16 ) );

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
        }
      }
      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
        ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL(selector1,\
              scl_fctr1,0,8);
        }
        else
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL(selector1,\
              scl_fctr1,0,8);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM(selector1,\
              scl_fctr1,0,8);
        }
        else
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM(selector1,\
              scl_fctr1,0,8);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_1x16F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      __m512 selector1 = _mm512_setzero_ps();

      __m512 scl_fctr1 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                    ( post_ops_attr.c_stor_type == BF16 ) );

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
              ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
        }
      }
      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL(selector1,\
              scl_fctr1,0,8);
        }
        else
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL(selector1,\
              scl_fctr1,0,8);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM(selector1,\
              scl_fctr1,0,8);
        }
        else
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM(selector1,\
              scl_fctr1,0,8);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_1x16F:
    {
        __m512 zmm7 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );
        __m512i ex_out;

        // c[0, 0-15]
        SWISH_F32_AVX512_DEF(zmm8, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_1x16F:
      {
        __m512i zmm6;
        // c[0, 0-15]
        TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_SIGMOID_1x16F:
      {
          __m512i ex_out;

          // c[0, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_1x16F_DISABLE:
      ;

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_last_k == TRUE ) )
      {
        uint32_t tlsb, rounded, temp[16] = {0};
        int i;
        bfloat16* dest;

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm8, 0, 0);

      }
      else
      {
        _mm512_storeu_ps(cbuf, zmm8);

      }
}


LPGEMM_MN_LT_NR0_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_5xlt16)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_5xlt16F_DISABLE,
              &&POST_OPS_BIAS_5xlt16F,
              &&POST_OPS_RELU_5xlt16F,
              &&POST_OPS_RELU_SCALE_5xlt16F,
              &&POST_OPS_GELU_TANH_5xlt16F,
              &&POST_OPS_GELU_ERF_5xlt16F,
              &&POST_OPS_CLIP_5xlt16F,
              &&POST_OPS_DOWNSCALE_5xlt16F,
              &&POST_OPS_MATRIX_ADD_5xlt16F,
              &&POST_OPS_SWISH_5xlt16F,
              &&POST_OPS_MATRIX_MUL_5xlt16F,
              &&POST_OPS_TANH_5xlt16F,
              &&POST_OPS_SIGMOID_5xlt16F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k0;

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5;
    __m512 zmm8, zmm12;
    __m512 zmm16, zmm20;
    __m512 zmm24;

    /* zero the accumulator registers */
    ZERO_ACC_ZMM_4_REG(zmm8, zmm12, zmm16, zmm20);
    zmm24 = _mm512_setzero_ps();

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;
    float *_cbuf = NULL;

    __mmask16 mask16 = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 32 elements from row0 of B*/
        zmm0 = _mm512_maskz_loadu_ps (mask16, bbuf);     //load 0-15 values from current row

       /*Broadcast col0 elements of 12 rows of A*/
        zmm2 = _mm512_set1_ps(*(abuf + 0*rs_a)); //broadcast c0r0
        zmm3 = _mm512_set1_ps(*(abuf + 1*rs_a)); //broadcast c0r1
        zmm4 = _mm512_set1_ps(*(abuf + 2*rs_a)); //broadcast c0r2
        zmm5 = _mm512_set1_ps(*(abuf + 3*rs_a)); //broadcast c0r3

        zmm8 = _mm512_fmadd_ps(zmm0, zmm2, zmm8);

        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);

        zmm2 = _mm512_set1_ps(*(abuf + 4*rs_a)); //broadcast c0r4

        zmm16 = _mm512_fmadd_ps(zmm0, zmm4, zmm16);

        zmm20 = _mm512_fmadd_ps(zmm0, zmm5, zmm20);

        zmm24 = _mm512_fmadd_ps(zmm0, zmm2, zmm24);

        bbuf += rs_b;  //move b pointer to next row
        abuf += cs_a;  //move a pointer to next col

    }//kloop

    zmm0 = _mm512_set1_ps(alpha);

    ALPHA_MUL_ACC_ZMM_4_REG(zmm8,zmm12,zmm16,zmm20,zmm0)
    ALPHA_MUL_ACC_ZMM_1_REG(zmm24,zmm0)

    if ( beta != 0.0 )
    {
      zmm3 = _mm512_set1_ps(beta);

      //load c and beta, convert to f32
      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_first_k == TRUE ) )
      {
        //c[0, 0-15]
        BF16_F32_BETA_OP_NLT16F_MASK(mask16, zmm8, 0, 0, zmm0,zmm3);
        //c[1, 0-15]
        BF16_F32_BETA_OP_NLT16F_MASK(mask16, zmm12, 1, 0, zmm0,zmm3);
        //c[2, 0-15]
        BF16_F32_BETA_OP_NLT16F_MASK(mask16, zmm16, 2, 0, zmm0,zmm3);
        //c[3, 0-15]
        BF16_F32_BETA_OP_NLT16F_MASK(mask16, zmm20, 3, 0, zmm0,zmm3);
        //c[4, 0-15]
        BF16_F32_BETA_OP_NLT16F_MASK(mask16, zmm24, 4, 0, zmm0,zmm3);
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back

        zmm0 = _mm512_maskz_loadu_ps(mask16, _cbuf);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        _cbuf += rs_c;

        zmm0 = _mm512_maskz_loadu_ps(mask16, _cbuf);
        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        _cbuf += rs_c;

        zmm0 = _mm512_maskz_loadu_ps(mask16, _cbuf);
        zmm16 = _mm512_fmadd_ps(zmm0, zmm3, zmm16);
        _cbuf += rs_c;

        zmm0 = _mm512_maskz_loadu_ps(mask16, _cbuf);
        zmm20 = _mm512_fmadd_ps(zmm0, zmm3, zmm20);
        _cbuf += rs_c;

        zmm0 = _mm512_maskz_loadu_ps(mask16, _cbuf);
        zmm24 = _mm512_fmadd_ps(zmm0, zmm3, zmm24);
      }
    }

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_5xlt16F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          BF16_F32_BIAS_LOAD(zmm1, mask16, 0)
        }
        else
        {
          zmm1 =
            _mm512_maskz_loadu_ps(mask16, ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[1,0-15]
        zmm12 = _mm512_add_ps( zmm1, zmm12 );

        // c[2,0-15]
        zmm16 = _mm512_add_ps( zmm1, zmm16 );

        // c[3,0-15]
        zmm20 = _mm512_add_ps( zmm1, zmm20 );

        // c[4,0-15]
        zmm24 = _mm512_add_ps( zmm1, zmm24 );

      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the bias array will be accessed by
        // the ic index, and each bias element corresponds to an
        // entire row of the transposed output array, instead of an
        // entire column.
        if( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );

          BF16_F32_BIAS_BCAST(zmm1, bias_mask, 0)
          BF16_F32_BIAS_BCAST(zmm2, bias_mask, 1)
          BF16_F32_BIAS_BCAST(zmm3, bias_mask, 2)
          BF16_F32_BIAS_BCAST(zmm4, bias_mask, 3)
          BF16_F32_BIAS_BCAST(zmm5, bias_mask, 4)
        }
        else
        {
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
        }
        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[1,0-15]
        zmm12 = _mm512_add_ps( zmm2, zmm12 );

        // c[2,0-15]
        zmm16 = _mm512_add_ps( zmm3, zmm16 );

        // c[3,0-15]
        zmm20 = _mm512_add_ps( zmm4, zmm20 );

        // c[4,0-15]
        zmm24 = _mm512_add_ps( zmm5, zmm24 );

      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_5xlt16F:
    {
      zmm1 = _mm512_setzero_ps();

      // c[0,0-15]
      zmm8 = _mm512_max_ps( zmm1, zmm8 );

      // c[1,0-15]
      zmm12 = _mm512_max_ps( zmm1, zmm12 );

      // c[2,0-15]
      zmm16 = _mm512_max_ps( zmm1, zmm16 );

      // c[3,0-15]
      zmm20 = _mm512_max_ps( zmm1, zmm20 );

      // c[4,0-15]
      zmm24 = _mm512_max_ps( zmm1, zmm24 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_5xlt16F:
    {
      zmm1 = _mm512_setzero_ps();
      zmm2 =
        _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

      __mmask16 relu_cmp_mask;

      // c[0, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm8)

      // c[1, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm12)

      // c[2, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm16)

      // c[3, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm20)

      // c[4, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm24)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_5xlt16F:
    {
      __m512i zmm6;
      // c[0, 0-15]
      GELU_TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 0-15]
      GELU_TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[2, 0-15]
      GELU_TANH_F32S_AVX512(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[3, 0-15]
      GELU_TANH_F32S_AVX512(zmm20, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[4, 0-15]
      GELU_TANH_F32S_AVX512(zmm24, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_5xlt16F:
    {
      // c[0, 0-15]
      GELU_ERF_F32S_AVX512(zmm8, zmm0, zmm1, zmm2)

      // c[1, 0-15]
      GELU_ERF_F32S_AVX512(zmm12, zmm0, zmm1, zmm2)

      // c[2, 0-15]
      GELU_ERF_F32S_AVX512(zmm16, zmm0, zmm1, zmm2)

      // c[3, 0-15]
      GELU_ERF_F32S_AVX512(zmm20, zmm0, zmm1, zmm2)

      // c[4, 0-15]
      GELU_ERF_F32S_AVX512(zmm24, zmm0, zmm1, zmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_5xlt16F:
    {
      zmm0 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      zmm1 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0, 0-15]
      CLIP_F32S_AVX512(zmm8, zmm0, zmm1)

      // c[1, 0-15]
      CLIP_F32S_AVX512(zmm12, zmm0, zmm1)

      // c[2, 0-15]
      CLIP_F32S_AVX512(zmm16, zmm0, zmm1)

      // c[3, 0-15]
      CLIP_F32S_AVX512(zmm20, zmm0, zmm1)

      // c[4, 0-15]
      CLIP_F32S_AVX512(zmm24, zmm0, zmm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_DOWNSCALE_5xlt16F:
    {
      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();
      __m512 selector3 = _mm512_setzero_ps();
      __m512 selector4 = _mm512_setzero_ps();
      __m512 selector5 = _mm512_setzero_ps();

      __m512 zero_point0 = _mm512_setzero_ps();
      __m512 zero_point1 = _mm512_setzero_ps();
      __m512 zero_point2 = _mm512_setzero_ps();
      __m512 zero_point3 = _mm512_setzero_ps();
      __m512 zero_point4 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                    ( ( post_ops_list_temp->stor_type == NONE ) &&
                      ( post_ops_attr.c_stor_type == BF16 ) );

      // Need to account for row vs column major swaps. For scalars
      // scale and zero point, no implications.
      // Even though different registers are used for scalar in column
      // and row major downscale path, all those registers will contain
      // the same value.

      if( post_ops_list_temp->scale_factor_len == 1 )
      {
        selector1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if ( is_bf16 == TRUE )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_ZP_BCST(zero_point0,0, zp_mask)
        }
        else
        {
          zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm512_maskz_loadu_ps(mask16, ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            BF16_F32_ZP_LOAD(zero_point0,mask16, 0)
          }
          else
          {
            zero_point0 = _mm512_maskz_loadu_ps(mask16, (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector1, zero_point0);

        //c[2, 0-15]
        F32_SCL_MULRND(zmm16, selector1, zero_point0);

        //c[3, 0-15]
        F32_SCL_MULRND(zmm20, selector1, zero_point0);

        //c[4, 0-15]
        F32_SCL_MULRND(zmm24, selector1, zero_point0);

      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the scale as well as zp array will
        // be accessed by the ic index, and each scale/zp element
        // corresponds to an entire row of the transposed output array,
        // instead of an entire column.
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 0 ) );
          selector2 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 1 ) );
          selector3 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 2 ) );
          selector4 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 3 ) );
          selector5 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 4 ) );
        }
        else
        {
          selector2 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          selector3 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          selector4 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          selector5 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            BF16_F32_ZP_LOAD(zero_point0,mask16, 0)
            BF16_F32_ZP_LOAD(zero_point1,mask16, 1)
            BF16_F32_ZP_LOAD(zero_point2,mask16, 2)
            BF16_F32_ZP_LOAD(zero_point3,mask16, 3)
            BF16_F32_ZP_LOAD(zero_point4,mask16, 4)
          }
          else
          {
            zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 1) );
            zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 2 ) );
            zero_point3 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 3 ) );
            zero_point4 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 4 ) );
          }
        }
        else
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_BCST(zero_point1,1, zp_mask)
            BF16_F32_ZP_BCST(zero_point2,2, zp_mask)
            BF16_F32_ZP_BCST(zero_point3,3, zp_mask)
            BF16_F32_ZP_BCST(zero_point4,4, zp_mask)
          }
          else
          {
            zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point3 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point4 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector2, zero_point1);

        //c[2, 0-15]
        F32_SCL_MULRND(zmm16, selector3, zero_point2);

        //c[3, 0-15]
        F32_SCL_MULRND(zmm20, selector4, zero_point3);

        //c[4, 0-15]
        F32_SCL_MULRND(zmm24, selector5, zero_point4);

      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_5xlt16F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      __m512 selector1 = _mm512_setzero_ps();

      __m512 scl_fctr1 = _mm512_setzero_ps();
      __m512 scl_fctr2 = _mm512_setzero_ps();
      __m512 scl_fctr3 = _mm512_setzero_ps();
      __m512 scl_fctr4 = _mm512_setzero_ps();
      __m512 scl_fctr5 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                ( post_ops_attr.c_stor_type == BF16 ) );

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr4 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr5 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm512_maskz_loadu_ps(mask16, ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
          scl_fctr4 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 3 ) );
          scl_fctr5 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 4 ) );
        }
      }
      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL_MASK(mask16, selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL_MASK(mask16, selector1,\
              scl_fctr1,1,12);

          // c[2:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL_MASK(mask16, selector1,\
              scl_fctr1,2,16);

          // c[3:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL_MASK(mask16, selector1,\
              scl_fctr1,3,20);

          // c[4:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL_MASK(mask16, selector1,\
              scl_fctr1,4,24);
        }
        else
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL_MASK(mask16, selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL_MASK(mask16, selector1,\
              scl_fctr2,1,12);

          // c[2:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL_MASK(mask16, selector1,\
              scl_fctr3,2,16);

          // c[3:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL_MASK(mask16, selector1,\
              scl_fctr4,3,20);

          // c[4:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL_MASK(mask16, selector1,\
              scl_fctr5,4,24);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr1,1,12);

          // c[2:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr1,2,16);

          // c[3:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr1,3,20);

          // c[4:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr1,4,24);
        }
        else
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr2,1,12);

          // c[2:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr3,2,16);

          // c[3:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr4,3,20);

          // c[4:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr5,4,24);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_5xlt16F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      __m512 selector1 = _mm512_setzero_ps();

      __m512 scl_fctr1 = _mm512_setzero_ps();
      __m512 scl_fctr2 = _mm512_setzero_ps();
      __m512 scl_fctr3 = _mm512_setzero_ps();
      __m512 scl_fctr4 = _mm512_setzero_ps();
      __m512 scl_fctr5 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                    ( post_ops_attr.c_stor_type == BF16 ) );

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr4 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr5 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
              ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm512_maskz_loadu_ps(mask16, ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
          scl_fctr4 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 3 ) );
          scl_fctr5 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 4 ) );
        }
      }
      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL_MASK(mask16, selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL_MASK(mask16, selector1,\
              scl_fctr1,1,12);

          // c[2:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL_MASK(mask16, selector1,\
              scl_fctr1,2,16);

          // c[3:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL_MASK(mask16, selector1,\
              scl_fctr1,3,20);

          // c[4:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL_MASK(mask16, selector1,\
              scl_fctr1,4,24);
        }
        else
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL_MASK(mask16, selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL_MASK(mask16, selector1,\
              scl_fctr2,1,12);

          // c[2:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL_MASK(mask16, selector1,\
              scl_fctr3,2,16);

          // c[3:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL_MASK(mask16, selector1,\
              scl_fctr4,3,20);

          // c[4:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL_MASK(mask16, selector1,\
              scl_fctr5,4,24);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr1,1,12);

          // c[2:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr1,2,16);

          // c[3:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr1,3,20);

          // c[4:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr1,4,24);
        }
        else
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr2,1,12);

          // c[2:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr3,2,16);

          // c[3:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr4,3,20);

          // c[4:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr5,4,24);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_5xlt16F:
    {
        __m512 zmm7 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );
        __m512i ex_out;

        // c[0, 0-15]
        SWISH_F32_AVX512_DEF(zmm8, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 0-15]
        SWISH_F32_AVX512_DEF(zmm12, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[2, 0-15]
        SWISH_F32_AVX512_DEF(zmm16, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[3, 0-15]
        SWISH_F32_AVX512_DEF(zmm20, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[4, 0-15]
        SWISH_F32_AVX512_DEF(zmm24, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_5xlt16F:
      {
        __m512i zmm6;
        // c[0, 0-15]
        TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 0-15]
        TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[2, 0-15]
        TANH_F32S_AVX512(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[3, 0-15]
        TANH_F32S_AVX512(zmm20, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[4, 0-15]
        TANH_F32S_AVX512(zmm24, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_SIGMOID_5xlt16F:
      {
          __m512i ex_out;

          // c[0, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[3, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm20, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[4, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm24, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_5xlt16F_DISABLE:
      ;

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_last_k == TRUE ) )
      {
        uint32_t tlsb, rounded, temp[16] = {0};
        int i;
        bfloat16* dest;

        CVT_STORE_F32_BF16_MASK_AVX512(n0_rem, zmm8, 0, 0);

        CVT_STORE_F32_BF16_MASK_AVX512(n0_rem, zmm12, 1, 0);

        CVT_STORE_F32_BF16_MASK_AVX512(n0_rem, zmm16, 2, 0);

        CVT_STORE_F32_BF16_MASK_AVX512(n0_rem, zmm20, 3, 0);

        CVT_STORE_F32_BF16_MASK_AVX512(n0_rem, zmm24, 4, 0);
      }
      else
      {
        _mm512_mask_storeu_ps(cbuf, mask16, zmm8);
        cbuf += rs_c;
        _mm512_mask_storeu_ps(cbuf, mask16, zmm12);
        cbuf += rs_c;
        _mm512_mask_storeu_ps(cbuf, mask16, zmm16);
        cbuf += rs_c;
        _mm512_mask_storeu_ps(cbuf, mask16, zmm20);
        cbuf += rs_c;
        _mm512_mask_storeu_ps(cbuf, mask16, zmm24);
      }
}

LPGEMM_MN_LT_NR0_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_4xlt16)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_4xlt16F_DISABLE,
              &&POST_OPS_BIAS_4xlt16F,
              &&POST_OPS_RELU_4xlt16F,
              &&POST_OPS_RELU_SCALE_4xlt16F,
              &&POST_OPS_GELU_TANH_4xlt16F,
              &&POST_OPS_GELU_ERF_4xlt16F,
              &&POST_OPS_CLIP_4xlt16F,
              &&POST_OPS_DOWNSCALE_4xlt16F,
              &&POST_OPS_MATRIX_ADD_4xlt16F,
              &&POST_OPS_SWISH_4xlt16F,
              &&POST_OPS_MATRIX_MUL_4xlt16F,
              &&POST_OPS_TANH_4xlt16F,
              &&POST_OPS_SIGMOID_4xlt16F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k0;

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5;
    __m512 zmm8, zmm12;
    __m512 zmm16, zmm20;

    /* zero the accumulator registers */
    ZERO_ACC_ZMM_4_REG(zmm8, zmm12, zmm16, zmm20);

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;
    float *_cbuf = NULL;

    __mmask16 mask16 = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 32 elements from row0 of B*/
        zmm0 = _mm512_maskz_loadu_ps (mask16, bbuf);     //load 0-15 values from current row

       /*Broadcast col0 elements of 12 rows of A*/
        zmm2 = _mm512_set1_ps(*(abuf + 0*rs_a)); //broadcast c0r0
        zmm3 = _mm512_set1_ps(*(abuf + 1*rs_a)); //broadcast c0r1
        zmm4 = _mm512_set1_ps(*(abuf + 2*rs_a)); //broadcast c0r2
        zmm5 = _mm512_set1_ps(*(abuf + 3*rs_a)); //broadcast c0r3

        zmm8 = _mm512_fmadd_ps(zmm0, zmm2, zmm8);

        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);

        zmm16 = _mm512_fmadd_ps(zmm0, zmm4, zmm16);

        zmm20 = _mm512_fmadd_ps(zmm0, zmm5, zmm20);

        bbuf += rs_b;  //move b pointer to next row
        abuf += cs_a;  //move a pointer to next col

    }//kloop

    zmm0 = _mm512_set1_ps(alpha);

    ALPHA_MUL_ACC_ZMM_4_REG(zmm8,zmm12,zmm16,zmm20,zmm0)

    if ( beta != 0.0 )
    {
      zmm3 = _mm512_set1_ps(beta);

      //load c and beta, convert to f32
      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_first_k == TRUE ) )
      {
        //c[0, 0-15]
        BF16_F32_BETA_OP_NLT16F_MASK(mask16, zmm8, 0, 0, zmm0,zmm3);
        //c[1, 0-15]
        BF16_F32_BETA_OP_NLT16F_MASK(mask16, zmm12, 1, 0, zmm0,zmm3);
        //c[2, 0-15]
        BF16_F32_BETA_OP_NLT16F_MASK(mask16, zmm16, 2, 0, zmm0,zmm3);
        //c[3, 0-15]
        BF16_F32_BETA_OP_NLT16F_MASK(mask16, zmm20, 3, 0, zmm0,zmm3);
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back

        zmm0 = _mm512_maskz_loadu_ps(mask16, _cbuf);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        _cbuf += rs_c;

        zmm0 = _mm512_maskz_loadu_ps(mask16, _cbuf);
        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        _cbuf += rs_c;

        zmm0 = _mm512_maskz_loadu_ps(mask16, _cbuf);
        zmm16 = _mm512_fmadd_ps(zmm0, zmm3, zmm16);
        _cbuf += rs_c;

        zmm0 = _mm512_maskz_loadu_ps(mask16, _cbuf);
        zmm20 = _mm512_fmadd_ps(zmm0, zmm3, zmm20);
      }
    }

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_4xlt16F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          BF16_F32_BIAS_LOAD(zmm1, mask16, 0)
        }
        else
        {
          zmm1 =
            _mm512_maskz_loadu_ps(mask16, ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[1,0-15]
        zmm12 = _mm512_add_ps( zmm1, zmm12 );

        // c[2,0-15]
        zmm16 = _mm512_add_ps( zmm1, zmm16 );

        // c[3,0-15]
        zmm20 = _mm512_add_ps( zmm1, zmm20 );

      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the bias array will be accessed by
        // the ic index, and each bias element corresponds to an
        // entire row of the transposed output array, instead of an
        // entire column.
        if( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );

          BF16_F32_BIAS_BCAST(zmm1, bias_mask, 0)
          BF16_F32_BIAS_BCAST(zmm2, bias_mask, 1)
          BF16_F32_BIAS_BCAST(zmm3, bias_mask, 2)
          BF16_F32_BIAS_BCAST(zmm4, bias_mask, 3)
        }
        else
        {
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
        }

        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[1,0-15]
        zmm12 = _mm512_add_ps( zmm2, zmm12 );

        // c[2,0-15]
        zmm16 = _mm512_add_ps( zmm3, zmm16 );

        // c[3,0-15]
        zmm20 = _mm512_add_ps( zmm4, zmm20 );

      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_4xlt16F:
    {
      zmm1 = _mm512_setzero_ps();

      // c[0,0-15]
      zmm8 = _mm512_max_ps( zmm1, zmm8 );

      // c[1,0-15]
      zmm12 = _mm512_max_ps( zmm1, zmm12 );

      // c[2,0-15]
      zmm16 = _mm512_max_ps( zmm1, zmm16 );

      // c[3,0-15]
      zmm20 = _mm512_max_ps( zmm1, zmm20 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_4xlt16F:
    {
      zmm1 = _mm512_setzero_ps();
      zmm2 =
        _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

      __mmask16 relu_cmp_mask;

      // c[0, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm8)

      // c[1, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm12)

      // c[2, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm16)

      // c[3, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm20)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_4xlt16F:
    {
      __m512i zmm6;
      // c[0, 0-15]
      GELU_TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 0-15]
      GELU_TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[2, 0-15]
      GELU_TANH_F32S_AVX512(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[3, 0-15]
      GELU_TANH_F32S_AVX512(zmm20, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_4xlt16F:
    {
      // c[0, 0-15]
      GELU_ERF_F32S_AVX512(zmm8, zmm0, zmm1, zmm2)

      // c[1, 0-15]
      GELU_ERF_F32S_AVX512(zmm12, zmm0, zmm1, zmm2)

      // c[2, 0-15]
      GELU_ERF_F32S_AVX512(zmm16, zmm0, zmm1, zmm2)

      // c[3, 0-15]
      GELU_ERF_F32S_AVX512(zmm20, zmm0, zmm1, zmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_4xlt16F:
    {
      zmm0 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      zmm1 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0, 0-15]
      CLIP_F32S_AVX512(zmm8, zmm0, zmm1)

      // c[1, 0-15]
      CLIP_F32S_AVX512(zmm12, zmm0, zmm1)

      // c[2, 0-15]
      CLIP_F32S_AVX512(zmm16, zmm0, zmm1)

      // c[3, 0-15]
      CLIP_F32S_AVX512(zmm20, zmm0, zmm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_DOWNSCALE_4xlt16F:
    {
      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();
      __m512 selector3 = _mm512_setzero_ps();
      __m512 selector4 = _mm512_setzero_ps();

      __m512 zero_point0 = _mm512_setzero_ps();
      __m512 zero_point1 = _mm512_setzero_ps();
      __m512 zero_point2 = _mm512_setzero_ps();
      __m512 zero_point3 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                    ( ( post_ops_list_temp->stor_type == NONE ) &&
                      ( post_ops_attr.c_stor_type == BF16 ) );

      // Need to account for row vs column major swaps. For scalars
      // scale and zero point, no implications.
      // Even though different registers are used for scalar in column
      // and row major downscale path, all those registers will contain
      // the same value.

      if( post_ops_list_temp->scale_factor_len == 1 )
      {
        selector1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if ( is_bf16 == TRUE )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_ZP_BCST(zero_point0,0, zp_mask)
        }
        else
        {
          zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm512_maskz_loadu_ps(mask16, ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            BF16_F32_ZP_LOAD(zero_point0,mask16, 0)
          }
          else
          {
            zero_point0 = _mm512_maskz_loadu_ps(mask16, (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector1, zero_point0);

        //c[2, 0-15]
        F32_SCL_MULRND(zmm16, selector1, zero_point0);

        //c[3, 0-15]
        F32_SCL_MULRND(zmm20, selector1, zero_point0);

      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the scale as well as zp array will
        // be accessed by the ic index, and each scale/zp element
        // corresponds to an entire row of the transposed output array,
        // instead of an entire column.
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 0 ) );
          selector2 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 1 ) );
          selector3 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 2 ) );
          selector4 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 3 ) );
        }
        else
        {
          selector2 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          selector3 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          selector4 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            BF16_F32_ZP_LOAD(zero_point0,mask16, 0)
            BF16_F32_ZP_LOAD(zero_point1,mask16, 1)
            BF16_F32_ZP_LOAD(zero_point2,mask16, 2)
            BF16_F32_ZP_LOAD(zero_point3,mask16, 3)
          }
          else
          {
            zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 1) );
            zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 2 ) );
            zero_point3 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 3 ) );
          }
        }
        else
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_BCST(zero_point1,1, zp_mask)
            BF16_F32_ZP_BCST(zero_point2,2, zp_mask)
            BF16_F32_ZP_BCST(zero_point3,3, zp_mask)
          }
          else
          {
            zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point3 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector2, zero_point1);

        //c[2, 0-15]
        F32_SCL_MULRND(zmm16, selector3, zero_point2);

        //c[3, 0-15]
        F32_SCL_MULRND(zmm20, selector4, zero_point3);

      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_4xlt16F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      __m512 selector1 = _mm512_setzero_ps();

      __m512 scl_fctr1 = _mm512_setzero_ps();
      __m512 scl_fctr2 = _mm512_setzero_ps();
      __m512 scl_fctr3 = _mm512_setzero_ps();
      __m512 scl_fctr4 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                ( post_ops_attr.c_stor_type == BF16 ) );

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr4 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm512_maskz_loadu_ps(mask16, ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
          scl_fctr4 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 3 ) );
        }
      }
      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL_MASK(mask16, selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL_MASK(mask16, selector1,\
              scl_fctr1,1,12);

          // c[2:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL_MASK(mask16, selector1,\
              scl_fctr1,2,16);

          // c[3:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL_MASK(mask16, selector1,\
              scl_fctr1,3,20);
        }
        else
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL_MASK(mask16, selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL_MASK(mask16, selector1,\
              scl_fctr2,1,12);

          // c[2:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL_MASK(mask16, selector1,\
              scl_fctr3,2,16);

          // c[3:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL_MASK(mask16, selector1,\
              scl_fctr4,3,20);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr1,1,12);

          // c[2:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr1,2,16);

          // c[3:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr1,3,20);
        }
        else
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr2,1,12);

          // c[2:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr3,2,16);

          // c[3:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr4,3,20);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_4xlt16F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      __m512 selector1 = _mm512_setzero_ps();

      __m512 scl_fctr1 = _mm512_setzero_ps();
      __m512 scl_fctr2 = _mm512_setzero_ps();
      __m512 scl_fctr3 = _mm512_setzero_ps();
      __m512 scl_fctr4 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                    ( post_ops_attr.c_stor_type == BF16 ) );

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr4 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
              ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm512_maskz_loadu_ps(mask16, ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
          scl_fctr4 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 3 ) );
        }
      }
      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL_MASK(mask16, selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL_MASK(mask16, selector1,\
              scl_fctr1,1,12);

          // c[2:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL_MASK(mask16, selector1,\
              scl_fctr1,2,16);

          // c[3:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL_MASK(mask16, selector1,\
              scl_fctr1,3,20);
        }
        else
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL_MASK(mask16, selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL_MASK(mask16, selector1,\
              scl_fctr2,1,12);

          // c[2:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL_MASK(mask16, selector1,\
              scl_fctr3,2,16);

          // c[3:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL_MASK(mask16, selector1,\
              scl_fctr4,3,20);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr1,1,12);

          // c[2:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr1,2,16);

          // c[3:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr1,3,20);

        }
        else
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr2,1,12);

          // c[2:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr3,2,16);

          // c[3:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr4,3,20);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_4xlt16F:
    {
        __m512 zmm7 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );
        __m512i ex_out;

        // c[0, 0-15]
        SWISH_F32_AVX512_DEF(zmm8, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 0-15]
        SWISH_F32_AVX512_DEF(zmm12, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[2, 0-15]
        SWISH_F32_AVX512_DEF(zmm16, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[3, 0-15]
        SWISH_F32_AVX512_DEF(zmm20, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_4xlt16F:
      {
        __m512i zmm6;
        // c[0, 0-15]
        TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 0-15]
        TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[2, 0-15]
        TANH_F32S_AVX512(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[3, 0-15]
        TANH_F32S_AVX512(zmm20, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_SIGMOID_4xlt16F:
      {
          __m512i ex_out;

          // c[0, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[3, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm20, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_4xlt16F_DISABLE:
      ;

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_last_k == TRUE ) )
      {
        uint32_t tlsb, rounded, temp[16] = {0};
        int i;
        bfloat16* dest;

        CVT_STORE_F32_BF16_MASK_AVX512(n0_rem, zmm8, 0, 0);

        CVT_STORE_F32_BF16_MASK_AVX512(n0_rem, zmm12, 1, 0);

        CVT_STORE_F32_BF16_MASK_AVX512(n0_rem, zmm16, 2, 0);

        CVT_STORE_F32_BF16_MASK_AVX512(n0_rem, zmm20, 3, 0);

      }
      else
      {
        _mm512_mask_storeu_ps(cbuf, mask16, zmm8);
        cbuf += rs_c;
        _mm512_mask_storeu_ps(cbuf, mask16, zmm12);
        cbuf += rs_c;
        _mm512_mask_storeu_ps(cbuf, mask16, zmm16);
        cbuf += rs_c;
        _mm512_mask_storeu_ps(cbuf, mask16, zmm20);

      }
}

LPGEMM_MN_LT_NR0_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_3xlt16)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_3xlt16F_DISABLE,
              &&POST_OPS_BIAS_3xlt16F,
              &&POST_OPS_RELU_3xlt16F,
              &&POST_OPS_RELU_SCALE_3xlt16F,
              &&POST_OPS_GELU_TANH_3xlt16F,
              &&POST_OPS_GELU_ERF_3xlt16F,
              &&POST_OPS_CLIP_3xlt16F,
              &&POST_OPS_DOWNSCALE_3xlt16F,
              &&POST_OPS_MATRIX_ADD_3xlt16F,
              &&POST_OPS_SWISH_3xlt16F,
              &&POST_OPS_MATRIX_MUL_3xlt16F,
              &&POST_OPS_TANH_3xlt16F,
              &&POST_OPS_SIGMOID_3xlt16F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k0;

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5;
    __m512 zmm8, zmm12;
    __m512 zmm16;

    /* zero the accumulator registers */
    ZERO_ACC_ZMM_2_REG(zmm8, zmm12);
    zmm16 = _mm512_setzero_ps();

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;
    float *_cbuf = NULL;

    __mmask16 mask16 = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 32 elements from row0 of B*/
        zmm0 = _mm512_maskz_loadu_ps (mask16, bbuf);     //load 0-15 values from current row

       /*Broadcast col0 elements of 12 rows of A*/
        zmm2 = _mm512_set1_ps(*(abuf + 0*rs_a)); //broadcast c0r0
        zmm3 = _mm512_set1_ps(*(abuf + 1*rs_a)); //broadcast c0r1
        zmm4 = _mm512_set1_ps(*(abuf + 2*rs_a)); //broadcast c0r2

        zmm8 = _mm512_fmadd_ps(zmm0, zmm2, zmm8);

        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);

        zmm16 = _mm512_fmadd_ps(zmm0, zmm4, zmm16);

        bbuf += rs_b;  //move b pointer to next row
        abuf += cs_a;  //move a pointer to next col

    }//kloop

    zmm0 = _mm512_set1_ps(alpha);

    ALPHA_MUL_ACC_ZMM_2_REG(zmm8,zmm12,zmm0)
    ALPHA_MUL_ACC_ZMM_1_REG(zmm16,zmm0)

    if ( beta != 0.0 )
    {
      zmm3 = _mm512_set1_ps(beta);

      //load c and beta, convert to f32
      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_first_k == TRUE ) )
      {
        //c[0, 0-15]
        BF16_F32_BETA_OP_NLT16F_MASK(mask16, zmm8, 0, 0, zmm0,zmm3);
        //c[1, 0-15]
        BF16_F32_BETA_OP_NLT16F_MASK(mask16, zmm12, 1, 0, zmm0,zmm3);
        //c[2, 0-15]
        BF16_F32_BETA_OP_NLT16F_MASK(mask16, zmm16, 2, 0, zmm0,zmm3);
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back

        zmm0 = _mm512_maskz_loadu_ps(mask16, _cbuf);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        _cbuf += rs_c;

        zmm0 = _mm512_maskz_loadu_ps(mask16, _cbuf);
        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        _cbuf += rs_c;

        zmm0 = _mm512_maskz_loadu_ps(mask16, _cbuf);
        zmm16 = _mm512_fmadd_ps(zmm0, zmm3, zmm16);

      }
    }

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_3xlt16F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          BF16_F32_BIAS_LOAD(zmm1, mask16, 0)
        }
        else
        {
          zmm1 =
            _mm512_maskz_loadu_ps(mask16, ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[1,0-15]
        zmm12 = _mm512_add_ps( zmm1, zmm12 );

        // c[2,0-15]
        zmm16 = _mm512_add_ps( zmm1, zmm16 );

      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the bias array will be accessed by
        // the ic index, and each bias element corresponds to an
        // entire row of the transposed output array, instead of an
        // entire column.
        if( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );

          BF16_F32_BIAS_BCAST(zmm1, bias_mask, 0)
          BF16_F32_BIAS_BCAST(zmm2, bias_mask, 1)
          BF16_F32_BIAS_BCAST(zmm3, bias_mask, 2)
        }
        else
        {
          zmm1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 0 ) );
          zmm2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 1 ) );
          zmm3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 2 ) );
        }
        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[1,0-15]
        zmm12 = _mm512_add_ps( zmm2, zmm12 );

        // c[2,0-15]
        zmm16 = _mm512_add_ps( zmm3, zmm16 );

      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_3xlt16F:
    {
      zmm1 = _mm512_setzero_ps();

      // c[0,0-15]
      zmm8 = _mm512_max_ps( zmm1, zmm8 );

      // c[1,0-15]
      zmm12 = _mm512_max_ps( zmm1, zmm12 );

      // c[2,0-15]
      zmm16 = _mm512_max_ps( zmm1, zmm16 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_3xlt16F:
    {
      zmm1 = _mm512_setzero_ps();
      zmm2 =
        _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

      __mmask16 relu_cmp_mask;

      // c[0, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm8)

      // c[1, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm12)

      // c[2, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm16)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_3xlt16F:
    {
      __m512i zmm6;
      // c[0, 0-15]
      GELU_TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 0-15]
      GELU_TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[2, 0-15]
      GELU_TANH_F32S_AVX512(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_3xlt16F:
    {
      // c[0, 0-15]
      GELU_ERF_F32S_AVX512(zmm8, zmm0, zmm1, zmm2)

      // c[1, 0-15]
      GELU_ERF_F32S_AVX512(zmm12, zmm0, zmm1, zmm2)

      // c[2, 0-15]
      GELU_ERF_F32S_AVX512(zmm16, zmm0, zmm1, zmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_3xlt16F:
    {
      zmm0 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      zmm1 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0, 0-15]
      CLIP_F32S_AVX512(zmm8, zmm0, zmm1)

      // c[1, 0-15]
      CLIP_F32S_AVX512(zmm12, zmm0, zmm1)

      // c[2, 0-15]
      CLIP_F32S_AVX512(zmm16, zmm0, zmm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_DOWNSCALE_3xlt16F:
    {
      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();
      __m512 selector3 = _mm512_setzero_ps();

      __m512 zero_point0 = _mm512_setzero_ps();
      __m512 zero_point1 = _mm512_setzero_ps();
      __m512 zero_point2 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                    ( ( post_ops_list_temp->stor_type == NONE ) &&
                      ( post_ops_attr.c_stor_type == BF16 ) );

      // Need to account for row vs column major swaps. For scalars
      // scale and zero point, no implications.
      // Even though different registers are used for scalar in column
      // and row major downscale path, all those registers will contain
      // the same value.

      if( post_ops_list_temp->scale_factor_len == 1 )
      {
        selector1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if ( ( post_ops_attr.buf_downscale != NULL ) &&
             ( post_ops_attr.is_first_k == TRUE ) )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_ZP_BCST(zero_point0,0, zp_mask)
        }
        else
        {
          zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm512_maskz_loadu_ps(mask16, ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            BF16_F32_ZP_LOAD(zero_point0,mask16, 0)
          }
          else
          {
            zero_point0 = _mm512_maskz_loadu_ps(mask16, (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector1, zero_point0);

        //c[2, 0-15]
        F32_SCL_MULRND(zmm16, selector1, zero_point0);

      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the scale as well as zp array will
        // be accessed by the ic index, and each scale/zp element
        // corresponds to an entire row of the transposed output array,
        // instead of an entire column.
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 0 ) );
          selector2 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 1 ) );
          selector3 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 2 ) );
        }
        else
        {
          selector2 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          selector3 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            BF16_F32_ZP_LOAD(zero_point0,mask16, 0)
            BF16_F32_ZP_LOAD(zero_point1,mask16, 1)
            BF16_F32_ZP_LOAD(zero_point2,mask16, 2)
          }
          else
          {
            zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 1) );
            zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 2 ) );
          }
        }
        else
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_BCST(zero_point1,1, zp_mask)
            BF16_F32_ZP_BCST(zero_point2,2, zp_mask)
          }
          else
          {
            zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point2 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector2, zero_point1);

        //c[2, 0-15]
        F32_SCL_MULRND(zmm16, selector3, zero_point2);

      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_3xlt16F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      __m512 selector1 = _mm512_setzero_ps();

      __m512 scl_fctr1 = _mm512_setzero_ps();
      __m512 scl_fctr2 = _mm512_setzero_ps();
      __m512 scl_fctr3 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                ( post_ops_attr.c_stor_type == BF16 ) );

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm512_maskz_loadu_ps(mask16, ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
        }
      }
      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL_MASK(mask16, selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL_MASK(mask16, selector1,\
              scl_fctr1,1,12);

          // c[2:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL_MASK(mask16, selector1,\
              scl_fctr1,2,16);

        }
        else
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL_MASK(mask16, selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL_MASK(mask16, selector1,\
              scl_fctr2,1,12);

          // c[2:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL_MASK(mask16, selector1,\
              scl_fctr3,2,16);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr1,1,12);

          // c[2:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr1,2,16);

        }
        else
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr2,1,12);

          // c[2:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr3,2,16);

        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_3xlt16F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      __m512 selector1 = _mm512_setzero_ps();

      __m512 scl_fctr1 = _mm512_setzero_ps();
      __m512 scl_fctr2 = _mm512_setzero_ps();
      __m512 scl_fctr3 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                    ( post_ops_attr.c_stor_type == BF16 ) );

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
              ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm512_maskz_loadu_ps(mask16, ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
        }
      }
      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL_MASK(mask16, selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL_MASK(mask16, selector1,\
              scl_fctr1,1,12);

          // c[2:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL_MASK(mask16, selector1,\
              scl_fctr1,2,16);
        }
        else
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL_MASK(mask16, selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL_MASK(mask16, selector1,\
              scl_fctr2,1,12);

          // c[2:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL_MASK(mask16, selector1,\
              scl_fctr3,2,16);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr1,1,12);

          // c[2:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr1,2,16);
        }
        else
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr2,1,12);

          // c[2:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr3,2,16);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_3xlt16F:
    {
        __m512 zmm7 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );
        __m512i ex_out;

        // c[0, 0-15]
        SWISH_F32_AVX512_DEF(zmm8, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 0-15]
        SWISH_F32_AVX512_DEF(zmm12, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[2, 0-15]
        SWISH_F32_AVX512_DEF(zmm16, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_3xlt16F:
      {
        __m512i zmm6;
        // c[0, 0-15]
        TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 0-15]
        TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[2, 0-15]
        TANH_F32S_AVX512(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_SIGMOID_3xlt16F:
      {
          __m512i ex_out;

          // c[0, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_3xlt16F_DISABLE:
      ;

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_last_k == TRUE ) )
      {
        uint32_t tlsb, rounded, temp[16] = {0};
        int i;
        bfloat16* dest;

        CVT_STORE_F32_BF16_MASK_AVX512(n0_rem, zmm8, 0, 0);

        CVT_STORE_F32_BF16_MASK_AVX512(n0_rem, zmm12, 1, 0);

        CVT_STORE_F32_BF16_MASK_AVX512(n0_rem, zmm16, 2, 0);

      }
      else
      {
        _mm512_mask_storeu_ps(cbuf, mask16, zmm8);
        cbuf += rs_c;
        _mm512_mask_storeu_ps(cbuf, mask16, zmm12);
        cbuf += rs_c;
        _mm512_mask_storeu_ps(cbuf, mask16, zmm16);

      }
}

LPGEMM_MN_LT_NR0_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_2xlt16)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_2xlt16F_DISABLE,
              &&POST_OPS_BIAS_2xlt16F,
              &&POST_OPS_RELU_2xlt16F,
              &&POST_OPS_RELU_SCALE_2xlt16F,
              &&POST_OPS_GELU_TANH_2xlt16F,
              &&POST_OPS_GELU_ERF_2xlt16F,
              &&POST_OPS_CLIP_2xlt16F,
              &&POST_OPS_DOWNSCALE_2xlt16F,
              &&POST_OPS_MATRIX_ADD_2xlt16F,
              &&POST_OPS_SWISH_2xlt16F,
              &&POST_OPS_MATRIX_MUL_2xlt16F,
              &&POST_OPS_TANH_2xlt16F,
              &&POST_OPS_SIGMOID_2xlt16F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k0;

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5;
    __m512 zmm8, zmm12;

    /* zero the accumulator registers */
    ZERO_ACC_ZMM_2_REG(zmm8, zmm12);

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;
    float *_cbuf = NULL;

    __mmask16 mask16 = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 32 elements from row0 of B*/
        zmm0 = _mm512_maskz_loadu_ps (mask16, bbuf);     //load 0-15 values from current row

       /*Broadcast col0 elements of 12 rows of A*/
        zmm2 = _mm512_set1_ps(*(abuf + 0*rs_a)); //broadcast c0r0
        zmm3 = _mm512_set1_ps(*(abuf + 1*rs_a)); //broadcast c0r1

        zmm8 = _mm512_fmadd_ps(zmm0, zmm2, zmm8);

        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);

        bbuf += rs_b;  //move b pointer to next row
        abuf += cs_a;  //move a pointer to next col

    }//kloop

    zmm0 = _mm512_set1_ps(alpha);

    ALPHA_MUL_ACC_ZMM_2_REG(zmm8,zmm12,zmm0)

    if ( beta != 0.0 )
    {
      zmm3 = _mm512_set1_ps(beta);

      //load c and beta, convert to f32
      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_first_k == TRUE ) )
      {
        //c[0, 0-15]
        BF16_F32_BETA_OP_NLT16F_MASK(mask16, zmm8, 0, 0, zmm0,zmm3);
        //c[1, 0-15]
        BF16_F32_BETA_OP_NLT16F_MASK(mask16, zmm12, 1, 0, zmm0,zmm3);
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back

        zmm0 = _mm512_maskz_loadu_ps(mask16, _cbuf);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
        _cbuf += rs_c;

        zmm0 = _mm512_maskz_loadu_ps(mask16, _cbuf);
        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);

      }
    }

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_2xlt16F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          BF16_F32_BIAS_LOAD(zmm1, mask16, 0)
        }
        else
        {
          zmm1 =
            _mm512_maskz_loadu_ps(mask16, ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[1,0-15]
        zmm12 = _mm512_add_ps( zmm1, zmm12 );

      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the bias array will be accessed by
        // the ic index, and each bias element corresponds to an
        // entire row of the transposed output array, instead of an
        // entire column.
        if( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );

          BF16_F32_BIAS_BCAST(zmm1, bias_mask, 0)
          BF16_F32_BIAS_BCAST(zmm2, bias_mask, 1)
        }
        else
        {
          zmm1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 0 ) );
          zmm2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 1 ) );
        }
        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

        // c[1,0-15]
        zmm12 = _mm512_add_ps( zmm2, zmm12 );

      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_2xlt16F:
    {
      zmm1 = _mm512_setzero_ps();

      // c[0,0-15]
      zmm8 = _mm512_max_ps( zmm1, zmm8 );

      // c[1,0-15]
      zmm12 = _mm512_max_ps( zmm1, zmm12 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_2xlt16F:
    {
      zmm1 = _mm512_setzero_ps();
      zmm2 =
        _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

      __mmask16 relu_cmp_mask;

      // c[0, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm8)

      // c[1, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm12)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_2xlt16F:
    {
      __m512i zmm6;
      // c[0, 0-15]
      GELU_TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      // c[1, 0-15]
      GELU_TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_2xlt16F:
    {
      // c[0, 0-15]
      GELU_ERF_F32S_AVX512(zmm8, zmm0, zmm1, zmm2)

      // c[1, 0-15]
      GELU_ERF_F32S_AVX512(zmm12, zmm0, zmm1, zmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_2xlt16F:
    {
      zmm0 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      zmm1 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0, 0-15]
      CLIP_F32S_AVX512(zmm8, zmm0, zmm1)

      // c[1, 0-15]
      CLIP_F32S_AVX512(zmm12, zmm0, zmm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_DOWNSCALE_2xlt16F:
    {
      __m512 selector1 = _mm512_setzero_ps();
      __m512 selector2 = _mm512_setzero_ps();

      __m512 zero_point0 = _mm512_setzero_ps();
      __m512 zero_point1 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                    ( ( post_ops_list_temp->stor_type == NONE ) &&
                      ( post_ops_attr.c_stor_type == BF16 ) );

      // Need to account for row vs column major swaps. For scalars
      // scale and zero point, no implications.
      // Even though different registers are used for scalar in column
      // and row major downscale path, all those registers will contain
      // the same value.

      if( post_ops_list_temp->scale_factor_len == 1 )
      {
        selector1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if ( is_bf16 == TRUE )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_ZP_BCST(zero_point0,0, zp_mask)
        }
        else
        {
          zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm512_maskz_loadu_ps(mask16, ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            BF16_F32_ZP_LOAD(zero_point0,mask16, 0)
          }
          else
          {
            zero_point0 = _mm512_maskz_loadu_ps(mask16, (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector1, zero_point0);

      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the scale as well as zp array will
        // be accessed by the ic index, and each scale/zp element
        // corresponds to an entire row of the transposed output array,
        // instead of an entire column.
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 0 ) );
          selector2 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 1 ) );
        }
        else
        {
          selector2 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE )
          {
            BF16_F32_ZP_LOAD(zero_point0,mask16, 0)
            BF16_F32_ZP_LOAD(zero_point1,mask16, 1)
          }
          else
          {
            zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 1) );
          }
        }
        else
        {
          if ( is_bf16 == TRUE )
          {
            __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
            BF16_F32_ZP_BCST(zero_point1,1, zp_mask)
          }
          else
          {
            zero_point1 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

        //c[1, 0-15]
        F32_SCL_MULRND(zmm12, selector2, zero_point1);

      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_2xlt16F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      __m512 selector1 = _mm512_setzero_ps();

      __m512 scl_fctr1 = _mm512_setzero_ps();
      __m512 scl_fctr2 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                ( post_ops_attr.c_stor_type == BF16 ) );

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm512_maskz_loadu_ps(mask16, ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
        }
      }
      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL_MASK(mask16, selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL_MASK(mask16, selector1,\
              scl_fctr1,1,12);
        }
        else
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL_MASK(mask16, selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL_MASK(mask16, selector1,\
              scl_fctr2,1,12);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr1,1,12);
        }
        else
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr2,1,12);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_2xlt16F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      __m512 selector1 = _mm512_setzero_ps();

      __m512 scl_fctr1 = _mm512_setzero_ps();
      __m512 scl_fctr2 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                    ( post_ops_attr.c_stor_type == BF16 ) );

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
              ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm512_maskz_loadu_ps(mask16, ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
        }
      }
      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL_MASK(mask16, selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL_MASK(mask16, selector1,\
              scl_fctr1,1,12);
        }
        else
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL_MASK(mask16, selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL_MASK(mask16, selector1,\
              scl_fctr2,1,12);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr1,1,12);
        }
        else
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr1,0,8);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr2,1,12);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_2xlt16F:
    {
        __m512 zmm7 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );
        __m512i ex_out;

        // c[0, 0-15]
        SWISH_F32_AVX512_DEF(zmm8, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        // c[1, 0-15]
        SWISH_F32_AVX512_DEF(zmm12, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_2xlt16F:
      {
        __m512i zmm6;
        // c[0, 0-15]
        TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        // c[1, 0-15]
        TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_SIGMOID_2xlt16F:
      {
          __m512i ex_out;

          // c[0, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_2xlt16F_DISABLE:
      ;

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_last_k == TRUE ) )
      {
        uint32_t tlsb, rounded, temp[16] = {0};
        int i;
        bfloat16* dest;

        CVT_STORE_F32_BF16_MASK_AVX512(n0_rem, zmm8, 0, 0);

        CVT_STORE_F32_BF16_MASK_AVX512(n0_rem, zmm12, 1, 0);

      }
      else
      {
        _mm512_mask_storeu_ps(cbuf, mask16, zmm8);
        cbuf += rs_c;
        _mm512_mask_storeu_ps(cbuf, mask16, zmm12);

      }
}

LPGEMM_MN_LT_NR0_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_1xlt16)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_1xlt16F_DISABLE,
              &&POST_OPS_BIAS_1xlt16F,
              &&POST_OPS_RELU_1xlt16F,
              &&POST_OPS_RELU_SCALE_1xlt16F,
              &&POST_OPS_GELU_TANH_1xlt16F,
              &&POST_OPS_GELU_ERF_1xlt16F,
              &&POST_OPS_CLIP_1xlt16F,
              &&POST_OPS_DOWNSCALE_1xlt16F,
              &&POST_OPS_MATRIX_ADD_1xlt16F,
              &&POST_OPS_SWISH_1xlt16F,
              &&POST_OPS_MATRIX_MUL_1xlt16F,
              &&POST_OPS_TANH_1xlt16F,
              &&POST_OPS_SIGMOID_1xlt16F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k0;

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5;
    __m512 zmm8;

    /* zero the accumulator registers */
    zmm8 = _mm512_setzero_ps();

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;
    float *_cbuf = NULL;

    __mmask16 mask16 = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 32 elements from row0 of B*/
        zmm0 = _mm512_maskz_loadu_ps (mask16, bbuf);     //load 0-15 values from current row

       /*Broadcast col0 elements of 12 rows of A*/
        zmm2 = _mm512_set1_ps(*(abuf + 0*rs_a)); //broadcast c0r0

        zmm8 = _mm512_fmadd_ps(zmm0, zmm2, zmm8);

        bbuf += rs_b;  //move b pointer to next row
        abuf += cs_a;  //move a pointer to next col

    }//kloop

    zmm0 = _mm512_set1_ps(alpha);

    ALPHA_MUL_ACC_ZMM_1_REG(zmm8,zmm0)

    if ( beta != 0.0 )
    {
      zmm3 = _mm512_set1_ps(beta);

      //load c and beta, convert to f32
      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_first_k == TRUE ) )
      {
        //c[0, 0-15]
        BF16_F32_BETA_OP_NLT16F_MASK(mask16, zmm8, 0, 0, zmm0,zmm3);
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back

        zmm0 = _mm512_maskz_loadu_ps(mask16, _cbuf);
        zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
      }
    }

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_1xlt16F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          BF16_F32_BIAS_LOAD(zmm1, mask16, 0)
        }
        else
        {
          zmm1 =
            _mm512_maskz_loadu_ps(mask16, ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the bias array will be accessed by
        // the ic index, and each bias element corresponds to an
        // entire row of the transposed output array, instead of an
        // entire column.
        if( post_ops_list_temp->stor_type == BF16 )
        {
          __mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );

          BF16_F32_BIAS_BCAST(zmm1, bias_mask, 0)
        }
        else
        {
          zmm1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 0 ) );
        }
        // c[0,0-15]
        zmm8 = _mm512_add_ps( zmm1, zmm8 );

      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_1xlt16F:
    {
      zmm1 = _mm512_setzero_ps();

      // c[0,0-15]
      zmm8 = _mm512_max_ps( zmm1, zmm8 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_1xlt16F:
    {
      zmm1 = _mm512_setzero_ps();
      zmm2 =
        _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

      __mmask16 relu_cmp_mask;

      // c[0, 0-15]
      RELU_SCALE_OP_F32S_AVX512(zmm8)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_1xlt16F:
    {
      __m512i zmm6;
      // c[0, 0-15]
      GELU_TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_1xlt16F:
    {
      // c[0, 0-15]
      GELU_ERF_F32S_AVX512(zmm8, zmm0, zmm1, zmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_1xlt16F:
    {
      zmm0 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      zmm1 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0, 0-15]
      CLIP_F32S_AVX512(zmm8, zmm0, zmm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_DOWNSCALE_1xlt16F:
    {
      __m512 selector1 = _mm512_setzero_ps();

      __m512 zero_point0 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                    ( ( post_ops_list_temp->stor_type == NONE ) &&
                      ( post_ops_attr.c_stor_type == BF16 ) );

      // Need to account for row vs column major swaps. For scalars
      // scale and zero point, no implications.
      // Even though different registers are used for scalar in column
      // and row major downscale path, all those registers will contain
      // the same value.

      if( post_ops_list_temp->scale_factor_len == 1 )
      {
        selector1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if ( is_bf16 == TRUE )
        {
          __mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );
          BF16_F32_ZP_BCST(zero_point0,0, zp_mask)
        }
        else
        {
          zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm512_maskz_loadu_ps(mask16, ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE  )
          {
            BF16_F32_ZP_LOAD(zero_point0,mask16, 0)
          }
          else
          {
            zero_point0 = _mm512_maskz_loadu_ps(mask16, (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          }
        }
        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

      }
      else
      {
        // If original output was columns major, then by the time
        // kernel sees it, the matrix would be accessed as if it were
        // transposed. Due to this the scale as well as zp array will
        // be accessed by the ic index, and each scale/zp element
        // corresponds to an entire row of the transposed output array,
        // instead of an entire column.
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 =
              _mm512_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 0 ) );
        }
        else
        {
          selector1 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if ( is_bf16 == TRUE  )
          {
            BF16_F32_ZP_LOAD(zero_point0,mask16, 0)
          }
          else
          {
            zero_point0 = _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 0 ) );
          }
        }

        //c[0, 0-15]
        F32_SCL_MULRND(zmm8, selector1, zero_point0);

      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_1xlt16F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      __m512 selector1 = _mm512_setzero_ps();

      __m512 scl_fctr1 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                ( post_ops_attr.c_stor_type == BF16 ) );

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm512_maskz_loadu_ps(mask16, ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
        }
      }
      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL_MASK(mask16, selector1,\
              scl_fctr1,0,8);
        }
        else
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_ADD_1COL_MASK(mask16, selector1,\
              scl_fctr1,0,8);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr1,0,8);
        }
        else
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_ADD_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr1,0,8);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_1xlt16F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      __m512 selector1 = _mm512_setzero_ps();

      __m512 scl_fctr1 = _mm512_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                    ( post_ops_attr.c_stor_type == BF16 ) );

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
              ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm512_maskz_loadu_ps(mask16, ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
        }
      }
      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL_MASK(mask16, selector1,\
              scl_fctr1,0,8);
        }
        else
        {
          // c[0:0-15,16-31]
          BF16_F32_MATRIX_MUL_1COL_MASK(mask16, selector1,\
              scl_fctr1,0,8);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr1,0,8);
        }
        else
        {
          // c[0:0-15,16-31]
          F32_F32_MATRIX_MUL_1COL_ZMM_MASK(mask16, selector1,\
              scl_fctr1,0,8);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_1xlt16F:
    {
        __m512 zmm7 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );
        __m512i ex_out;

        // c[0, 0-15]
        SWISH_F32_AVX512_DEF(zmm8, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_1xlt16F:
      {
        __m512i zmm6;
        // c[0, 0-15]
        TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm6)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_SIGMOID_1xlt16F:
      {
          __m512i ex_out;

          // c[0, 0-15]
          SIGMOID_F32_AVX512_DEF(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_1xlt16F_DISABLE:
      ;

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_last_k == TRUE ) )
      {
        uint32_t tlsb, rounded, temp[16] = {0};
        int i;
        bfloat16* dest;

        CVT_STORE_F32_BF16_MASK_AVX512(n0_rem, zmm8, 0, 0);

      }
      else
      {
        _mm512_mask_storeu_ps(cbuf, mask16, zmm8);

      }
}

#endif
