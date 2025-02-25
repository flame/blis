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

#include "lpgemm_kernel_macros_f32_avx2.h"

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_5x16)
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
    float *_cbuf = NULL;

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

    if ( beta != 0.0 )
    {
      ymm3 = _mm256_broadcast_ss(&(beta));

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
        ( post_ops_attr.is_first_k == TRUE ) )
      {
        BF16_F32_C_BNZ_8(0,0,ymm0,ymm3,ymm4)
        BF16_F32_C_BNZ_8(0,1,ymm1,ymm3,ymm5)
        BF16_F32_C_BNZ_8(1,0,ymm0,ymm3,ymm6)
        BF16_F32_C_BNZ_8(1,1,ymm1,ymm3,ymm7)
        BF16_F32_C_BNZ_8(2,0,ymm0,ymm3,ymm8)
        BF16_F32_C_BNZ_8(2,1,ymm1,ymm3,ymm9)
        BF16_F32_C_BNZ_8(3,0,ymm0,ymm3,ymm10)
        BF16_F32_C_BNZ_8(3,1,ymm1,ymm3,ymm11)
        BF16_F32_C_BNZ_8(4,0,ymm0,ymm3,ymm12)
        BF16_F32_C_BNZ_8(4,1,ymm1,ymm3,ymm13)
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back
        F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm4)
        F32_C_BNZ_8(_cbuf+8,rs_c,ymm1,ymm3,ymm5)
        _cbuf += rs_c;
        F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm6)
        F32_C_BNZ_8(_cbuf+8,rs_c,ymm1,ymm3,ymm7)
        _cbuf += rs_c;
        F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm8)
        F32_C_BNZ_8(_cbuf+8,rs_c,ymm1,ymm3,ymm9)
        _cbuf += rs_c;
        F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm10)
        F32_C_BNZ_8(_cbuf+8,rs_c,ymm1,ymm3,ymm11)
        _cbuf += rs_c;
        F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm12)
        F32_C_BNZ_8(_cbuf+8,rs_c,ymm1,ymm3,ymm13)
      }
    }//betazero

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
          BF16_F32_BIAS_LOAD_AVX2(ymm0, 0);
          BF16_F32_BIAS_LOAD_AVX2( ymm1, 1 );
        }
        else
        {
          ymm0 = _mm256_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 0 * 8 ) );
          ymm1 = _mm256_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 1 * 8 ) );
        }

        // c[0,0-7]
        ymm4 = _mm256_add_ps( ymm4, ymm0 );

        // c[0,8-15]
        ymm5 = _mm256_add_ps( ymm5, ymm1 );

        // c[1,0-7]
        ymm6 = _mm256_add_ps( ymm6, ymm0 );

        // c[1,8-15]
        ymm7 = _mm256_add_ps( ymm7, ymm1 );

        // c[2,0-7]
        ymm8 = _mm256_add_ps( ymm8, ymm0 );

        // c[2,8-15]
        ymm9 = _mm256_add_ps( ymm9, ymm1 );

        // c[3,0-7]
        ymm10 = _mm256_add_ps( ymm10, ymm0 );

        // c[3,8-15]
        ymm11 = _mm256_add_ps( ymm11, ymm1 );

        // c[4,0-7]
        ymm12 = _mm256_add_ps( ymm12, ymm0 );

        // c[4,8-15]
        ymm13 = _mm256_add_ps( ymm13, ymm1 );
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
          BF16_F32_BIAS_BCAST_AVX2(ymm0,0);
          BF16_F32_BIAS_BCAST_AVX2(ymm1,1);
          BF16_F32_BIAS_BCAST_AVX2(ymm2,2);
          BF16_F32_BIAS_BCAST_AVX2(ymm3,3);
        }
        else
        {
          ymm0 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 0 );
          ymm1 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 1 );
          ymm2 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 2 );
          ymm3 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 3 );
        }

        // c[0,0-7]
        ymm4 = _mm256_add_ps( ymm4, ymm0 );

        // c[0,8-15]
        ymm5 = _mm256_add_ps( ymm5, ymm0 );

        // c[1,0-7]
        ymm6 = _mm256_add_ps( ymm6, ymm1 );

        // c[1,8-15]
        ymm7 = _mm256_add_ps( ymm7, ymm1 );

        // c[2,0-7]
        ymm8 = _mm256_add_ps( ymm8, ymm2 );

        // c[2,8-15]
        ymm9 = _mm256_add_ps( ymm9, ymm2 );

        // c[3,0-7]
        ymm10 = _mm256_add_ps( ymm10, ymm3 );

        // c[3,8-15]
        ymm11 = _mm256_add_ps( ymm11, ymm3 );

        if( post_ops_list_temp->stor_type == BF16 )
        {
          BF16_F32_BIAS_BCAST_AVX2(ymm0,4);
        }
        else
        {
          ymm0 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
                post_ops_attr.post_op_c_i + 4 );
        }

        // c[4,0-7]
        ymm12 = _mm256_add_ps( ymm12, ymm0 );

        // c[4,8-15]
        ymm13 = _mm256_add_ps( ymm13, ymm0 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_5x16F:
    {
      ymm0 = _mm256_setzero_ps();

      // c[0,0-7]
      ymm4 = _mm256_max_ps( ymm4, ymm0 );

      // c[0,8-15]
      ymm5 = _mm256_max_ps( ymm5, ymm0 );

      // c[1,0-7]
      ymm6 = _mm256_max_ps( ymm6, ymm0 );

      // c[1,8-15]
      ymm7 = _mm256_max_ps( ymm7, ymm0 );

      // c[2,0-7]
      ymm8 = _mm256_max_ps( ymm8, ymm0 );

      // c[2,8-15]
      ymm9 = _mm256_max_ps( ymm9, ymm0 );

      // c[3,0-7]
      ymm10 = _mm256_max_ps( ymm10, ymm0 );

      // c[3,8-15]
      ymm11 = _mm256_max_ps( ymm11, ymm0 );

      // c[4,0-7]
      ymm12 = _mm256_max_ps( ymm12, ymm0 );

      // c[4,8-15]
      ymm13 = _mm256_max_ps( ymm13, ymm0 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_5x16F:
    {
      ymm0 =
        _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
      ymm1 = _mm256_setzero_ps();

      // c[0,0-7]
      RELU_SCALE_OP_F32S_AVX2(ymm4, ymm0, ymm1, ymm2)

      // c[0,8-15]
      RELU_SCALE_OP_F32S_AVX2(ymm5, ymm0, ymm1, ymm2)

      // c[1,0-7]
      RELU_SCALE_OP_F32S_AVX2(ymm6, ymm0, ymm1, ymm2)

      // c[1,8-15]
      RELU_SCALE_OP_F32S_AVX2(ymm7, ymm0, ymm1, ymm2)

      // c[2,0-7]
      RELU_SCALE_OP_F32S_AVX2(ymm8, ymm0, ymm1, ymm2)

      // c[2,8-15]
      RELU_SCALE_OP_F32S_AVX2(ymm9, ymm0, ymm1, ymm2)

      // c[3,0-7]
      RELU_SCALE_OP_F32S_AVX2(ymm10, ymm0, ymm1, ymm2)

      // c[3,8-15]
      RELU_SCALE_OP_F32S_AVX2(ymm11, ymm0, ymm1, ymm2)

      // c[4,0-7]
      RELU_SCALE_OP_F32S_AVX2(ymm12, ymm0, ymm1, ymm2)

      // c[4,8-15]
      RELU_SCALE_OP_F32S_AVX2(ymm13, ymm0, ymm1, ymm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_5x16F:
    {
      __m256 dn, x_tanh;
      __m256i q;

      // c[0,0-7]
      GELU_TANH_F32S_AVX2(ymm4, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      // c[0,8-15]
      GELU_TANH_F32S_AVX2(ymm5, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      // c[1,0-7]
      GELU_TANH_F32S_AVX2(ymm6, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      // c[1,8-15]
      GELU_TANH_F32S_AVX2(ymm7, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      // c[2,0-7]
      GELU_TANH_F32S_AVX2(ymm8, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      // c[2,8-15]
      GELU_TANH_F32S_AVX2(ymm9, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      // c[3,0-7]
      GELU_TANH_F32S_AVX2(ymm10, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      // c[3,8-15]
      GELU_TANH_F32S_AVX2(ymm11, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      // c[4,0-7]
      GELU_TANH_F32S_AVX2(ymm12, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      // c[4,8-15]
      GELU_TANH_F32S_AVX2(ymm13, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_5x16F:
    {
      // c[0,0-7]
      GELU_ERF_F32S_AVX2(ymm4, ymm0, ymm1, ymm2)

      // c[0,8-15]
      GELU_ERF_F32S_AVX2(ymm5, ymm0, ymm1, ymm2)

      // c[1,0-7]
      GELU_ERF_F32S_AVX2(ymm6, ymm0, ymm1, ymm2)

      // c[1,8-15]
      GELU_ERF_F32S_AVX2(ymm7, ymm0, ymm1, ymm2)

      // c[2,0-7]
      GELU_ERF_F32S_AVX2(ymm8, ymm0, ymm1, ymm2)

      // c[2,8-15]
      GELU_ERF_F32S_AVX2(ymm9, ymm0, ymm1, ymm2)

      // c[3,0-7]
      GELU_ERF_F32S_AVX2(ymm10, ymm0, ymm1, ymm2)

      // c[3,8-15]
      GELU_ERF_F32S_AVX2(ymm11, ymm0, ymm1, ymm2)

      // c[4,0-7]
      GELU_ERF_F32S_AVX2(ymm12, ymm0, ymm1, ymm2)

      // c[4,8-15]
      GELU_ERF_F32S_AVX2(ymm13, ymm0, ymm1, ymm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_5x16F:
    {
      ymm0 = _mm256_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      ymm1 = _mm256_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0,0-7]
      CLIP_F32S_AVX2(ymm4, ymm0, ymm1)

      // c[0,8-15]
      CLIP_F32S_AVX2(ymm5, ymm0, ymm1)

      // c[1,0-7]
      CLIP_F32S_AVX2(ymm6, ymm0, ymm1)

      // c[1,8-15]
      CLIP_F32S_AVX2(ymm7, ymm0, ymm1)

      // c[2,0-7]
      CLIP_F32S_AVX2(ymm8, ymm0, ymm1)

      // c[2,8-15]
      CLIP_F32S_AVX2(ymm9, ymm0, ymm1)

      // c[3,0-7]
      CLIP_F32S_AVX2(ymm10, ymm0, ymm1)

      // c[3,8-15]
      CLIP_F32S_AVX2(ymm11, ymm0, ymm1)

      // c[4,0-7]
      CLIP_F32S_AVX2(ymm12, ymm0, ymm1)

      // c[4,8-15]
      CLIP_F32S_AVX2(ymm13, ymm0, ymm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_DOWNSCALE_5x16F:
    {
      __m256 selector1 = _mm256_setzero_ps();
      __m256 selector2 = _mm256_setzero_ps();
      __m256 selector3 = _mm256_setzero_ps();
      __m256 selector4 = _mm256_setzero_ps();
      __m256 selector5 = _mm256_setzero_ps();

      __m256 zero_point0 = _mm256_setzero_ps();
      __m256 zero_point1 = _mm256_setzero_ps();
      __m256 zero_point2 = _mm256_setzero_ps();
      __m256 zero_point3 = _mm256_setzero_ps();
      __m256 zero_point4 = _mm256_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      if( post_ops_list_temp->scale_factor_len == 1 )
      {
        selector1 =
              _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector2 =
              _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector3 =
              _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector4 =
              _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector5 =
              _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if( is_bf16 == TRUE )
        {
          BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point0);
          BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point1);
          BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point2);
          BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point3);
          BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point4);
        }
        else
        {
          zero_point0 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point1 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point2 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point3 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point4 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }

      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
              ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm256_loadu_ps ( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 8 ) );
          selector2 = _mm256_loadu_ps ( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 1 * 8 ) );
        }
        if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_LOAD_AVX2(zero_point0,0);
            BF16_F32_ZP_VECTOR_LOAD_AVX2(zero_point1,1);
          }
          else
          {
            zero_point0 = _mm256_loadu_ps ( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 0 * 8 ) );
            zero_point1 = _mm256_loadu_ps ( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 1 * 8 ) );
          }
        }
        //c[0, 0-7]
        F32_SCL_MULRND_AVX2(ymm4, selector1, zero_point0);

        //c[0, 8-15]
        F32_SCL_MULRND_AVX2(ymm5, selector2, zero_point1);

        //c[1, 0-7]
        F32_SCL_MULRND_AVX2(ymm6, selector1, zero_point0);

        //c[1, 8-15]
        F32_SCL_MULRND_AVX2(ymm7, selector2, zero_point1);

        //c[2, 0-7]
        F32_SCL_MULRND_AVX2(ymm8, selector1, zero_point0);

        //c[2, 8-15]
        F32_SCL_MULRND_AVX2(ymm9, selector2, zero_point1);

        //c[3, 0-7]
        F32_SCL_MULRND_AVX2(ymm10, selector1, zero_point0);

        //c[3, 8-15]
        F32_SCL_MULRND_AVX2(ymm11, selector2, zero_point1);

        //c[4, 0-7]
        F32_SCL_MULRND_AVX2(ymm12, selector1, zero_point0);

        //c[4, 8-15]
        F32_SCL_MULRND_AVX2(ymm13, selector2, zero_point1);
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
              _mm256_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 0 ) );
          selector2 =
              _mm256_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 1 ) );
          selector3 =
              _mm256_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 2 ) );
          selector4 =
              _mm256_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 3 ) );
          selector5 =
              _mm256_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 4 ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_BCAST_AVX2(zero_point0,0);
            BF16_F32_ZP_VECTOR_BCAST_AVX2(zero_point1,1);
            BF16_F32_ZP_VECTOR_BCAST_AVX2(zero_point2,2);
            BF16_F32_ZP_VECTOR_BCAST_AVX2(zero_point3,3);
            BF16_F32_ZP_VECTOR_BCAST_AVX2(zero_point4,4);
          }
          else
          {
            zero_point0 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 1 ) );
            zero_point2 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 2 ) );
            zero_point3 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 3 ) );
            zero_point4 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                post_ops_attr.post_op_c_i + 4 ) );
          }
        }
        //c[0, 0-7]
        F32_SCL_MULRND_AVX2(ymm4, selector1, zero_point0);

        //c[0, 8-15]
        F32_SCL_MULRND_AVX2(ymm5, selector1, zero_point0);

        //c[1, 0-7]
        F32_SCL_MULRND_AVX2(ymm6, selector2, zero_point1);

        //c[1, 8-15]
        F32_SCL_MULRND_AVX2(ymm7, selector2, zero_point1);

        //c[2, 0-7]
        F32_SCL_MULRND_AVX2(ymm8, selector3, zero_point2);

        //c[2, 8-15]
        F32_SCL_MULRND_AVX2(ymm9, selector3, zero_point2);

        //c[3, 0-7]
        F32_SCL_MULRND_AVX2(ymm10, selector4, zero_point3);

        //c[3, 8-15]
        F32_SCL_MULRND_AVX2(ymm11, selector4, zero_point3);

        //c[4, 0-7]
        F32_SCL_MULRND_AVX2(ymm12, selector5, zero_point4);

        //c[4, 8-15]
        F32_SCL_MULRND_AVX2(ymm13, selector5, zero_point4);
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_5x16F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
        ( ( post_ops_list_temp->stor_type == NONE ) &&
          ( post_ops_attr.c_stor_type == BF16 ) );

      __m256 scl_fctr1 = _mm256_setzero_ps();
      __m256 scl_fctr2 = _mm256_setzero_ps();
      __m256 scl_fctr3 = _mm256_setzero_ps();
      __m256 scl_fctr4 = _mm256_setzero_ps();
      __m256 scl_fctr5 = _mm256_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr4 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr5 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm256_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          scl_fctr2 =
            _mm256_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 1 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
          scl_fctr4 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 3 ) );
          scl_fctr5 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 4 ) );
        }
      }

      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,0,4,5);

          // c[1:0-15]
          BF16_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,1,6,7);

          // c[2:0-15]
          BF16_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,2,8,9);

          // c[3:0-15]
          BF16_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,3,10,11);

          // c[4:0-15]
          BF16_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,4,12,13);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr1,0,4,5);

          // c[1:0-15]
          BF16_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr2,scl_fctr2,1,6,7);

          // c[2:0-15]
          BF16_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr3,scl_fctr3,2,8,9);

          // c[3:0-15]
          BF16_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr4,scl_fctr4,3,10,11);

          // c[4:0-15]
          BF16_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr5,scl_fctr5,4,12,13);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,0,4,5);

          // c[1:0-15]
          F32_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,1,6,7);

          // c[2:0-15]
          F32_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,2,8,9);

          // c[3:0-15]
          F32_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,3,10,11);

          // c[4:0-15]
          F32_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,4,12,13);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr1,0,4,5);

          // c[1:0-15]
          F32_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr2,scl_fctr2,1,6,7);

          // c[2:0-15]
          F32_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr3,scl_fctr3,2,8,9);

          // c[3:0-15]
          F32_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr4,scl_fctr4,3,10,11);

          // c[4:0-15]
          F32_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr5,scl_fctr5,4,12,13);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_5x16F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m256 scl_fctr1 = _mm256_setzero_ps();
      __m256 scl_fctr2 = _mm256_setzero_ps();
      __m256 scl_fctr3 = _mm256_setzero_ps();
      __m256 scl_fctr4 = _mm256_setzero_ps();
      __m256 scl_fctr5 = _mm256_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr4 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr5 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm256_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          scl_fctr2 =
            _mm256_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 1 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
          scl_fctr4 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 3 ) );
          scl_fctr5 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 4 ) );
        }
      }

      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
        ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,0,4,5);

          // c[1:0-15]
          BF16_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,1,6,7);

          // c[2:0-15]
          BF16_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,2,8,9);

          // c[3:0-15]
          BF16_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,3,10,11);

          // c[4:0-15]
          BF16_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,4,12,13);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr1,0,4,5);

          // c[1:0-15]
          BF16_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr2,scl_fctr2,1,6,7);

          // c[2:0-15]
          BF16_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr3,scl_fctr3,2,8,9);

          // c[3:0-15]
          BF16_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr4,scl_fctr4,3,10,11);

          // c[4:0-15]
          BF16_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr5,scl_fctr5,4,12,13);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,0,4,5);

          // c[1:0-15]
          F32_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,1,6,7);

          // c[2:0-15]
          F32_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,2,8,9);

          // c[3:0-15]
          F32_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,3,10,11);

          // c[4:0-15]
          F32_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,4,12,13);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr1,0,4,5);

          // c[1:0-15]
          F32_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr2,scl_fctr2,1,6,7);

          // c[2:0-15]
          F32_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr3,scl_fctr3,2,8,9);

          // c[3:0-15]
          F32_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr4,scl_fctr4,3,10,11);

          // c[4:0-15]
          F32_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr5,scl_fctr5,4,12,13);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_5x16F:
    {
        ymm0 =
          _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
        __m256 z, dn;
        __m256i ex_out;

        // c[0,0-7]
        SWISH_F32_AVX2_DEF(ymm4, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[0,8-15]
        SWISH_F32_AVX2_DEF(ymm5, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[1,0-7]
        SWISH_F32_AVX2_DEF(ymm6, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[1,8-15]
        SWISH_F32_AVX2_DEF(ymm7, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[2,0-7]
        SWISH_F32_AVX2_DEF(ymm8, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[2,8-15]
        SWISH_F32_AVX2_DEF(ymm9, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[3,0-7]
        SWISH_F32_AVX2_DEF(ymm10, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[3,8-15]
        SWISH_F32_AVX2_DEF(ymm11, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[4,0-7]
        SWISH_F32_AVX2_DEF(ymm12, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[4,8-15]
        SWISH_F32_AVX2_DEF(ymm13, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_5x16F:
    {
      __m256 dn;
      __m256i q;

      // c[0,0-7]
      TANH_F32S_AVX2(ymm4, ymm0, ymm1, ymm2, ymm3, dn, q)

      // c[0,8-15]
      TANH_F32S_AVX2(ymm5, ymm0, ymm1, ymm2, ymm3, dn, q)

      // c[1,0-7]
      TANH_F32S_AVX2(ymm6, ymm0, ymm1, ymm2, ymm3, dn, q)

      // c[1,8-15]
      TANH_F32S_AVX2(ymm7, ymm0, ymm1, ymm2, ymm3, dn, q)

      // c[2,0-7]
      TANH_F32S_AVX2(ymm8, ymm0, ymm1, ymm2, ymm3, dn, q)

      // c[2,8-15]
      TANH_F32S_AVX2(ymm9, ymm0, ymm1, ymm2, ymm3, dn, q)

      // c[3,0-7]
      TANH_F32S_AVX2(ymm10, ymm0, ymm1, ymm2, ymm3, dn, q)

      // c[3,8-15]
      TANH_F32S_AVX2(ymm11, ymm0, ymm1, ymm2, ymm3, dn, q)

      // c[4,0-7]
      TANH_F32S_AVX2(ymm12, ymm0, ymm1, ymm2, ymm3, dn, q)

      // c[4,8-15]
      TANH_F32S_AVX2(ymm13, ymm0, ymm1, ymm2, ymm3, dn, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SIGMOID_5x16F:
    {
        __m256 z, dn;
        __m256i ex_out;

        // c[0,0-7]
        SIGMOID_F32_AVX2_DEF(ymm4, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[0,8-15]
        SIGMOID_F32_AVX2_DEF(ymm5, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[1,0-7]
        SIGMOID_F32_AVX2_DEF(ymm6, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[1,8-15]
        SIGMOID_F32_AVX2_DEF(ymm7, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[2,0-7]
        SIGMOID_F32_AVX2_DEF(ymm8, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[2,8-15]
        SIGMOID_F32_AVX2_DEF(ymm9, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[3,0-7]
        SIGMOID_F32_AVX2_DEF(ymm10, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[3,8-15]
        SIGMOID_F32_AVX2_DEF(ymm11, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[4,0-7]
        SIGMOID_F32_AVX2_DEF(ymm12, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[4,8-15]
        SIGMOID_F32_AVX2_DEF(ymm13, ymm1, ymm2, ymm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_5x16F_DISABLE:
    ;

    uint32_t tlsb, rounded, temp[8] = {0};
    int i;
    bfloat16* dest;

    if ( ( post_ops_attr.buf_downscale != NULL ) &&
    ( post_ops_attr.is_last_k == TRUE ) )
    {
      STORE_F32_BF16_YMM(ymm4, 0, 0)
      STORE_F32_BF16_YMM(ymm5, 0, 1)
      STORE_F32_BF16_YMM(ymm6, 1, 0)
      STORE_F32_BF16_YMM(ymm7, 1, 1)
      STORE_F32_BF16_YMM(ymm8, 2, 0)
      STORE_F32_BF16_YMM(ymm9, 2, 1)
      STORE_F32_BF16_YMM(ymm10, 3, 0)
      STORE_F32_BF16_YMM(ymm11, 3, 1)
      STORE_F32_BF16_YMM(ymm12, 4, 0)
      STORE_F32_BF16_YMM(ymm13, 4, 1)
    }
    else
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
    }
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_4x16)
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
    float *_cbuf = NULL;

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


    if ( beta != 0.0 )
    {
      ymm3 = _mm256_broadcast_ss(&(beta));

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
        ( post_ops_attr.is_first_k == TRUE ) )
      {
        BF16_F32_C_BNZ_8(0,0,ymm0,ymm3,ymm4)
        BF16_F32_C_BNZ_8(0,1,ymm1,ymm3,ymm5)
        BF16_F32_C_BNZ_8(1,0,ymm0,ymm3,ymm6)
        BF16_F32_C_BNZ_8(1,1,ymm1,ymm3,ymm7)
        BF16_F32_C_BNZ_8(2,0,ymm0,ymm3,ymm8)
        BF16_F32_C_BNZ_8(2,1,ymm1,ymm3,ymm9)
        BF16_F32_C_BNZ_8(3,0,ymm0,ymm3,ymm10)
        BF16_F32_C_BNZ_8(3,1,ymm1,ymm3,ymm11)
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back
        F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm4)
        F32_C_BNZ_8(_cbuf+8,rs_c,ymm1,ymm3,ymm5)
        _cbuf += rs_c;
        F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm6)
        F32_C_BNZ_8(_cbuf+8,rs_c,ymm1,ymm3,ymm7)
        _cbuf += rs_c;
        F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm8)
        F32_C_BNZ_8(_cbuf+8,rs_c,ymm1,ymm3,ymm9)
        _cbuf += rs_c;
        F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm10)
        F32_C_BNZ_8(_cbuf+8,rs_c,ymm1,ymm3,ymm11)
      }
    }//betazero

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
          BF16_F32_BIAS_LOAD_AVX2( ymm0, 0 );
          BF16_F32_BIAS_LOAD_AVX2( ymm1, 1 );
        }
        else
        {
          ymm0 = _mm256_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 0 * 8 ) );
          ymm1 = _mm256_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 1 * 8 ) );
        }

        // c[0,0-7]
        ymm4 = _mm256_add_ps( ymm4, ymm0 );

        // c[0,8-15]
        ymm5 = _mm256_add_ps( ymm5, ymm1 );

        // c[1,0-7]
        ymm6 = _mm256_add_ps( ymm6, ymm0 );

        // c[1,8-15]
        ymm7 = _mm256_add_ps( ymm7, ymm1 );

        // c[2,0-7]
        ymm8 = _mm256_add_ps( ymm8, ymm0 );

        // c[2,8-15]
        ymm9 = _mm256_add_ps( ymm9, ymm1 );

        // c[3,0-7]
        ymm10 = _mm256_add_ps( ymm10, ymm0 );

        // c[3,8-15]
        ymm11 = _mm256_add_ps( ymm11, ymm1 );
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
          BF16_F32_BIAS_BCAST_AVX2(ymm0,0);
          BF16_F32_BIAS_BCAST_AVX2(ymm1,1);
          BF16_F32_BIAS_BCAST_AVX2(ymm2,2);
          BF16_F32_BIAS_BCAST_AVX2(ymm3,3);
        }
        else
        {
          ymm0 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 0 );
          ymm1 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 1 );
          ymm2 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 2 );
          ymm3 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 3 );
        }

        // c[0,0-7]
        ymm4 = _mm256_add_ps( ymm4, ymm0 );

        // c[0,8-15]
        ymm5 = _mm256_add_ps( ymm5, ymm0 );

        // c[1,0-7]
        ymm6 = _mm256_add_ps( ymm6, ymm1 );

        // c[1,8-15]
        ymm7 = _mm256_add_ps( ymm7, ymm1 );

        // c[2,0-7]
        ymm8 = _mm256_add_ps( ymm8, ymm2 );

        // c[2,8-15]
        ymm9 = _mm256_add_ps( ymm9, ymm2 );

        // c[3,0-7]
        ymm10 = _mm256_add_ps( ymm10, ymm3 );

        // c[3,8-15]
        ymm11 = _mm256_add_ps( ymm11, ymm3 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_4x16F:
    {
      ymm0 = _mm256_setzero_ps();

      // c[0,0-7]
      ymm4 = _mm256_max_ps( ymm4, ymm0 );

      // c[0,8-15]
      ymm5 = _mm256_max_ps( ymm5, ymm0 );

      // c[1,0-7]
      ymm6 = _mm256_max_ps( ymm6, ymm0 );

      // c[1,8-15]
      ymm7 = _mm256_max_ps( ymm7, ymm0 );

      // c[2,0-7]
      ymm8 = _mm256_max_ps( ymm8, ymm0 );

      // c[2,8-15]
      ymm9 = _mm256_max_ps( ymm9, ymm0 );

      // c[3,0-7]
      ymm10 = _mm256_max_ps( ymm10, ymm0 );

      // c[3,8-15]
      ymm11 = _mm256_max_ps( ymm11, ymm0 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_4x16F:
    {
      ymm0 =
        _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
      ymm1 = _mm256_setzero_ps();

      // c[0,0-7]
      RELU_SCALE_OP_F32S_AVX2(ymm4, ymm0, ymm1, ymm2)

      // c[0,8-15]
      RELU_SCALE_OP_F32S_AVX2(ymm5, ymm0, ymm1, ymm2)

      // c[1,0-7]
      RELU_SCALE_OP_F32S_AVX2(ymm6, ymm0, ymm1, ymm2)

      // c[1,8-15]
      RELU_SCALE_OP_F32S_AVX2(ymm7, ymm0, ymm1, ymm2)

      // c[2,0-7]
      RELU_SCALE_OP_F32S_AVX2(ymm8, ymm0, ymm1, ymm2)

      // c[2,8-15]
      RELU_SCALE_OP_F32S_AVX2(ymm9, ymm0, ymm1, ymm2)

      // c[3,0-7]
      RELU_SCALE_OP_F32S_AVX2(ymm10, ymm0, ymm1, ymm2)

      // c[3,8-15]
      RELU_SCALE_OP_F32S_AVX2(ymm11, ymm0, ymm1, ymm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_4x16F:
    {
      __m256 dn, x_tanh;
      __m256i q;

      // c[0,0-7]
      GELU_TANH_F32S_AVX2(ymm4, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      // c[0,8-15]
      GELU_TANH_F32S_AVX2(ymm5, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      // c[1,0-7]
      GELU_TANH_F32S_AVX2(ymm6, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      // c[1,8-15]
      GELU_TANH_F32S_AVX2(ymm7, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      // c[2,0-7]
      GELU_TANH_F32S_AVX2(ymm8, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      // c[2,8-15]
      GELU_TANH_F32S_AVX2(ymm9, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      // c[3,0-7]
      GELU_TANH_F32S_AVX2(ymm10, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      // c[3,8-15]
      GELU_TANH_F32S_AVX2(ymm11, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_4x16F:
    {
      // c[0,0-7]
      GELU_ERF_F32S_AVX2(ymm4, ymm0, ymm1, ymm2)

      // c[0,8-15]
      GELU_ERF_F32S_AVX2(ymm5, ymm0, ymm1, ymm2)

      // c[1,0-7]
      GELU_ERF_F32S_AVX2(ymm6, ymm0, ymm1, ymm2)

      // c[1,8-15]
      GELU_ERF_F32S_AVX2(ymm7, ymm0, ymm1, ymm2)

      // c[2,0-7]
      GELU_ERF_F32S_AVX2(ymm8, ymm0, ymm1, ymm2)

      // c[2,8-15]
      GELU_ERF_F32S_AVX2(ymm9, ymm0, ymm1, ymm2)

      // c[3,0-7]
      GELU_ERF_F32S_AVX2(ymm10, ymm0, ymm1, ymm2)

      // c[3,8-15]
      GELU_ERF_F32S_AVX2(ymm11, ymm0, ymm1, ymm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_4x16F:
    {
      ymm0 = _mm256_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      ymm1 = _mm256_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0,0-7]
      CLIP_F32S_AVX2(ymm4, ymm0, ymm1)

      // c[0,8-15]
      CLIP_F32S_AVX2(ymm5, ymm0, ymm1)

      // c[1,0-7]
      CLIP_F32S_AVX2(ymm6, ymm0, ymm1)

      // c[1,8-15]
      CLIP_F32S_AVX2(ymm7, ymm0, ymm1)

      // c[2,0-7]
      CLIP_F32S_AVX2(ymm8, ymm0, ymm1)

      // c[2,8-15]
      CLIP_F32S_AVX2(ymm9, ymm0, ymm1)

      // c[3,0-7]
      CLIP_F32S_AVX2(ymm10, ymm0, ymm1)

      // c[3,8-15]
      CLIP_F32S_AVX2(ymm11, ymm0, ymm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_DOWNSCALE_4x16F:
    {
      __m256 selector1 = _mm256_setzero_ps();
      __m256 selector2 = _mm256_setzero_ps();
      __m256 selector3 = _mm256_setzero_ps();
      __m256 selector4 = _mm256_setzero_ps();

      __m256 zero_point0 = _mm256_setzero_ps();
      __m256 zero_point1 = _mm256_setzero_ps();
      __m256 zero_point2 = _mm256_setzero_ps();
      __m256 zero_point3 = _mm256_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      if( post_ops_list_temp->scale_factor_len == 1 )
      {
        selector1 =
              _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector2 =
              _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector3 =
              _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector4 =
              _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if( is_bf16 == TRUE )
        {
          BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point0);
          BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point1);
          BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point2);
          BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point3);
        }
        else
        {
          zero_point0 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point1 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point2 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point3 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }

      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
              ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm256_loadu_ps ( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 8 ) );
          selector2 = _mm256_loadu_ps ( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 1 * 8 ) );
        }
        if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_LOAD_AVX2(zero_point0,0);
            BF16_F32_ZP_VECTOR_LOAD_AVX2(zero_point1,1);
          }
          else
          {
            zero_point0 = _mm256_loadu_ps ( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 0 * 8 ) );
            zero_point1 = _mm256_loadu_ps ( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 1 * 8 ) );
          }
        }
        //c[0, 0-7]
        F32_SCL_MULRND_AVX2(ymm4, selector1, zero_point0);

        //c[0, 8-15]
        F32_SCL_MULRND_AVX2(ymm5, selector2, zero_point1);

        //c[1, 0-7]
        F32_SCL_MULRND_AVX2(ymm6, selector1, zero_point0);

        //c[1, 8-15]
        F32_SCL_MULRND_AVX2(ymm7, selector2, zero_point1);

        //c[2, 0-7]
        F32_SCL_MULRND_AVX2(ymm8, selector1, zero_point0);

        //c[2, 8-15]
        F32_SCL_MULRND_AVX2(ymm9, selector2, zero_point1);

        //c[3, 0-7]
        F32_SCL_MULRND_AVX2(ymm10, selector1, zero_point0);

        //c[3, 8-15]
        F32_SCL_MULRND_AVX2(ymm11, selector2, zero_point1);
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
              _mm256_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 0 ) );
          selector2 =
              _mm256_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 1 ) );
          selector3 =
              _mm256_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 2 ) );
          selector4 =
              _mm256_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 3 ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_BCAST_AVX2(zero_point0,0);
            BF16_F32_ZP_VECTOR_BCAST_AVX2(zero_point1,1);
            BF16_F32_ZP_VECTOR_BCAST_AVX2(zero_point2,2);
            BF16_F32_ZP_VECTOR_BCAST_AVX2(zero_point3,3);
          }
          else
          {
            zero_point0 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 1 ) );
            zero_point2 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 2 ) );
            zero_point3 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 3 ) );
          }
        }
        //c[0, 0-7]
        F32_SCL_MULRND_AVX2(ymm4, selector1, zero_point0);

        //c[0, 8-15]
        F32_SCL_MULRND_AVX2(ymm5, selector1, zero_point0);

        //c[1, 0-7]
        F32_SCL_MULRND_AVX2(ymm6, selector2, zero_point1);

        //c[1, 8-15]
        F32_SCL_MULRND_AVX2(ymm7, selector2, zero_point1);

        //c[2, 0-7]
        F32_SCL_MULRND_AVX2(ymm8, selector3, zero_point2);

        //c[2, 8-15]
        F32_SCL_MULRND_AVX2(ymm9, selector3, zero_point2);

        //c[3, 0-7]
        F32_SCL_MULRND_AVX2(ymm10, selector4, zero_point3);

        //c[3, 8-15]
        F32_SCL_MULRND_AVX2(ymm11, selector4, zero_point3);
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_4x16F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m256 scl_fctr1 = _mm256_setzero_ps();
      __m256 scl_fctr2 = _mm256_setzero_ps();
      __m256 scl_fctr3 = _mm256_setzero_ps();
      __m256 scl_fctr4 = _mm256_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr4 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm256_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          scl_fctr2 =
            _mm256_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 1 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
          scl_fctr4 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 3 ) );
        }
      }

      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,0,4,5);

          // c[1:0-15]
          BF16_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,1,6,7);

          // c[2:0-15]
          BF16_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,2,8,9);

          // c[3:0-15]
          BF16_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,3,10,11);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr1,0,4,5);

          // c[1:0-15]
          BF16_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr2,scl_fctr2,1,6,7);

          // c[2:0-15]
          BF16_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr3,scl_fctr3,2,8,9);

          // c[3:0-15]
          BF16_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr4,scl_fctr4,3,10,11);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,0,4,5);

          // c[1:0-15]
          F32_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,1,6,7);

          // c[2:0-15]
          F32_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,2,8,9);

          // c[3:0-15]
          F32_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,3,10,11);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr1,0,4,5);

          // c[1:0-15]
          F32_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr2,scl_fctr2,1,6,7);

          // c[2:0-15]
          F32_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr3,scl_fctr3,2,8,9);

          // c[3:0-15]
          F32_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr4,scl_fctr4,3,10,11);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_4x16F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m256 scl_fctr1 = _mm256_setzero_ps();
      __m256 scl_fctr2 = _mm256_setzero_ps();
      __m256 scl_fctr3 = _mm256_setzero_ps();
      __m256 scl_fctr4 = _mm256_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr4 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm256_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          scl_fctr2 =
            _mm256_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 1 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
          scl_fctr4 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 3 ) );
        }
      }

      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
        ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,0,4,5);

          // c[1:0-15]
          BF16_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,1,6,7);

          // c[2:0-15]
          BF16_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,2,8,9);

          // c[3:0-15]
          BF16_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,3,10,11);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr1,0,4,5);

          // c[1:0-15]
          BF16_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr2,scl_fctr2,1,6,7);

          // c[2:0-15]
          BF16_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr3,scl_fctr3,2,8,9);

          // c[3:0-15]
          BF16_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr4,scl_fctr4,3,10,11);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,0,4,5);

          // c[1:0-15]
          F32_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,1,6,7);

          // c[2:0-15]
          F32_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,2,8,9);

          // c[3:0-15]
          F32_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,3,10,11);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr1,0,4,5);

          // c[1:0-15]
          F32_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr2,scl_fctr2,1,6,7);

          // c[2:0-15]
          F32_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr3,scl_fctr3,2,8,9);

          // c[3:0-15]
          F32_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr4,scl_fctr4,3,10,11);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_4x16F:
    {
        ymm0 =
          _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
        __m256 z, dn;
        __m256i ex_out;

        // c[0,0-7]
        SWISH_F32_AVX2_DEF(ymm4, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[0,8-15]
        SWISH_F32_AVX2_DEF(ymm5, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[1,0-7]
        SWISH_F32_AVX2_DEF(ymm6, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[1,8-15]
        SWISH_F32_AVX2_DEF(ymm7, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[2,0-7]
        SWISH_F32_AVX2_DEF(ymm8, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[2,8-15]
        SWISH_F32_AVX2_DEF(ymm9, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[3,0-7]
        SWISH_F32_AVX2_DEF(ymm10, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[3,8-15]
        SWISH_F32_AVX2_DEF(ymm11, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_4x16F:
    {
      __m256 dn;
      __m256i q;

      // c[0,0-7]
      TANH_F32S_AVX2(ymm4, ymm0, ymm1, ymm2, ymm3, dn, q)

      // c[0,8-15]
      TANH_F32S_AVX2(ymm5, ymm0, ymm1, ymm2, ymm3, dn, q)

      // c[1,0-7]
      TANH_F32S_AVX2(ymm6, ymm0, ymm1, ymm2, ymm3, dn, q)

      // c[1,8-15]
      TANH_F32S_AVX2(ymm7, ymm0, ymm1, ymm2, ymm3, dn, q)

      // c[2,0-7]
      TANH_F32S_AVX2(ymm8, ymm0, ymm1, ymm2, ymm3, dn, q)

      // c[2,8-15]
      TANH_F32S_AVX2(ymm9, ymm0, ymm1, ymm2, ymm3, dn, q)

      // c[3,0-7]
      TANH_F32S_AVX2(ymm10, ymm0, ymm1, ymm2, ymm3, dn, q)

      // c[3,8-15]
      TANH_F32S_AVX2(ymm11, ymm0, ymm1, ymm2, ymm3, dn, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SIGMOID_4x16F:
    {
        __m256 z, dn;
        __m256i ex_out;

        // c[0,0-7]
        SIGMOID_F32_AVX2_DEF(ymm4, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[0,8-15]
        SIGMOID_F32_AVX2_DEF(ymm5, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[1,0-7]
        SIGMOID_F32_AVX2_DEF(ymm6, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[1,8-15]
        SIGMOID_F32_AVX2_DEF(ymm7, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[2,0-7]
        SIGMOID_F32_AVX2_DEF(ymm8, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[2,8-15]
        SIGMOID_F32_AVX2_DEF(ymm9, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[3,0-7]
        SIGMOID_F32_AVX2_DEF(ymm10, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[3,8-15]
        SIGMOID_F32_AVX2_DEF(ymm11, ymm1, ymm2, ymm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_4x16F_DISABLE:
    ;

  uint32_t tlsb, rounded, temp[8] = {0};
  int i;
  bfloat16* dest;

  if ( ( post_ops_attr.buf_downscale != NULL ) &&
    ( post_ops_attr.is_last_k == TRUE ) )
  {
    STORE_F32_BF16_YMM(ymm4, 0, 0);
    STORE_F32_BF16_YMM(ymm5, 0, 1);
    STORE_F32_BF16_YMM(ymm6, 1, 0);
    STORE_F32_BF16_YMM(ymm7, 1, 1);
    STORE_F32_BF16_YMM(ymm8, 2, 0);
    STORE_F32_BF16_YMM(ymm9, 2, 1);
    STORE_F32_BF16_YMM(ymm10, 3, 0);
    STORE_F32_BF16_YMM(ymm11, 3, 1);
  }
  else
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
  }
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_3x16)
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
    float *_cbuf = NULL;

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


    if ( beta != 0.0 )
    {
      ymm3 = _mm256_broadcast_ss(&(beta));

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
        ( post_ops_attr.is_first_k == TRUE ) )
      {
        BF16_F32_C_BNZ_8(0,0,ymm0,ymm3,ymm4)
        BF16_F32_C_BNZ_8(0,1,ymm1,ymm3,ymm5)
        BF16_F32_C_BNZ_8(1,0,ymm0,ymm3,ymm6)
        BF16_F32_C_BNZ_8(1,1,ymm1,ymm3,ymm7)
        BF16_F32_C_BNZ_8(2,0,ymm0,ymm3,ymm8)
        BF16_F32_C_BNZ_8(2,1,ymm1,ymm3,ymm9)
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back
        F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm4)
        F32_C_BNZ_8(_cbuf+8,rs_c,ymm1,ymm3,ymm5)
        _cbuf += rs_c;
        F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm6)
        F32_C_BNZ_8(_cbuf+8,rs_c,ymm1,ymm3,ymm7)
        _cbuf += rs_c;
        F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm8)
        F32_C_BNZ_8(_cbuf+8,rs_c,ymm1,ymm3,ymm9)
      }
    }//betazero

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
          BF16_F32_BIAS_LOAD_AVX2( ymm0, 0 );
          BF16_F32_BIAS_LOAD_AVX2( ymm1, 1 );
        }
        else
        {
          ymm0 = _mm256_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 0 * 8 ) );
          ymm1 = _mm256_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 1 * 8 ) );
        }

        // c[0,0-7]
        ymm4 = _mm256_add_ps( ymm4, ymm0 );

        // c[0,8-15]
        ymm5 = _mm256_add_ps( ymm5, ymm1 );

        // c[1,0-7]
        ymm6 = _mm256_add_ps( ymm6, ymm0 );

        // c[1,8-15]
        ymm7 = _mm256_add_ps( ymm7, ymm1 );

        // c[2,0-7]
        ymm8 = _mm256_add_ps( ymm8, ymm0 );

        // c[2,8-15]
        ymm9 = _mm256_add_ps( ymm9, ymm1 );
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
          BF16_F32_BIAS_BCAST_AVX2(ymm0,0);
          BF16_F32_BIAS_BCAST_AVX2(ymm1,1);
          BF16_F32_BIAS_BCAST_AVX2(ymm2,2);
        }
        else
        {
          ymm0 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 0 );
          ymm1 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 1 );
          ymm2 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 2 );
        }

        // c[0,0-7]
        ymm4 = _mm256_add_ps( ymm4, ymm0 );

        // c[0,8-15]
        ymm5 = _mm256_add_ps( ymm5, ymm0 );

        // c[1,0-7]
        ymm6 = _mm256_add_ps( ymm6, ymm1 );

        // c[1,8-15]
        ymm7 = _mm256_add_ps( ymm7, ymm1 );

        // c[2,0-7]
        ymm8 = _mm256_add_ps( ymm8, ymm2 );

        // c[2,8-15]
        ymm9 = _mm256_add_ps( ymm9, ymm2 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_3x16F:
    {
      ymm0 = _mm256_setzero_ps();

      // c[0,0-7]
      ymm4 = _mm256_max_ps( ymm4, ymm0 );

      // c[0,8-15]
      ymm5 = _mm256_max_ps( ymm5, ymm0 );

      // c[1,0-7]
      ymm6 = _mm256_max_ps( ymm6, ymm0 );

      // c[1,8-15]
      ymm7 = _mm256_max_ps( ymm7, ymm0 );

      // c[2,0-7]
      ymm8 = _mm256_max_ps( ymm8, ymm0 );

      // c[2,8-15]
      ymm9 = _mm256_max_ps( ymm9, ymm0 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_3x16F:
    {
      ymm0 =
        _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
      ymm1 = _mm256_setzero_ps();

      // c[0,0-7]
      RELU_SCALE_OP_F32S_AVX2(ymm4, ymm0, ymm1, ymm2)

      // c[0,8-15]
      RELU_SCALE_OP_F32S_AVX2(ymm5, ymm0, ymm1, ymm2)

      // c[1,0-7]
      RELU_SCALE_OP_F32S_AVX2(ymm6, ymm0, ymm1, ymm2)

      // c[1,8-15]
      RELU_SCALE_OP_F32S_AVX2(ymm7, ymm0, ymm1, ymm2)

      // c[2,0-7]
      RELU_SCALE_OP_F32S_AVX2(ymm8, ymm0, ymm1, ymm2)

      // c[2,8-15]
      RELU_SCALE_OP_F32S_AVX2(ymm9, ymm0, ymm1, ymm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_3x16F:
    {
      __m256 dn, x_tanh;
      __m256i q;

      // c[0,0-7]
      GELU_TANH_F32S_AVX2(ymm4, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      // c[0,8-15]
      GELU_TANH_F32S_AVX2(ymm5, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      // c[1,0-7]
      GELU_TANH_F32S_AVX2(ymm6, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      // c[1,8-15]
      GELU_TANH_F32S_AVX2(ymm7, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      // c[2,0-7]
      GELU_TANH_F32S_AVX2(ymm8, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      // c[2,8-15]
      GELU_TANH_F32S_AVX2(ymm9, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_3x16F:
    {
      // c[0,0-7]
      GELU_ERF_F32S_AVX2(ymm4, ymm0, ymm1, ymm2)

      // c[0,8-15]
      GELU_ERF_F32S_AVX2(ymm5, ymm0, ymm1, ymm2)

      // c[1,0-7]
      GELU_ERF_F32S_AVX2(ymm6, ymm0, ymm1, ymm2)

      // c[1,8-15]
      GELU_ERF_F32S_AVX2(ymm7, ymm0, ymm1, ymm2)

      // c[2,0-7]
      GELU_ERF_F32S_AVX2(ymm8, ymm0, ymm1, ymm2)

      // c[2,8-15]
      GELU_ERF_F32S_AVX2(ymm9, ymm0, ymm1, ymm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_3x16F:
    {
      ymm0 = _mm256_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      ymm1 = _mm256_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0,0-7]
      CLIP_F32S_AVX2(ymm4, ymm0, ymm1)

      // c[0,8-15]
      CLIP_F32S_AVX2(ymm5, ymm0, ymm1)

      // c[1,0-7]
      CLIP_F32S_AVX2(ymm6, ymm0, ymm1)

      // c[1,8-15]
      CLIP_F32S_AVX2(ymm7, ymm0, ymm1)

      // c[2,0-7]
      CLIP_F32S_AVX2(ymm8, ymm0, ymm1)

      // c[2,8-15]
      CLIP_F32S_AVX2(ymm9, ymm0, ymm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_DOWNSCALE_3x16F:
    {
      __m256 selector1 = _mm256_setzero_ps();
      __m256 selector2 = _mm256_setzero_ps();
      __m256 selector3 = _mm256_setzero_ps();

      __m256 zero_point0 = _mm256_setzero_ps();
      __m256 zero_point1 = _mm256_setzero_ps();
      __m256 zero_point2 = _mm256_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      if( post_ops_list_temp->scale_factor_len == 1 )
      {
        selector1 =
              _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector2 =
              _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector3 =
              _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if( is_bf16 == TRUE )
        {
          BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point0);
          BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point1);
          BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point2);
        }
        else
        {
          zero_point0 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point1 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point2 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }

      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
              ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm256_loadu_ps ( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 8 ) );
          selector2 = _mm256_loadu_ps ( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 1 * 8 ) );
        }
        if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_LOAD_AVX2(zero_point0,0);
            BF16_F32_ZP_VECTOR_LOAD_AVX2(zero_point1,1);
          }
          else
          {
            zero_point0 = _mm256_loadu_ps ( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 0 * 8 ) );
            zero_point1 = _mm256_loadu_ps ( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 1 * 8 ) );
          }
        }
        //c[0, 0-7]
        F32_SCL_MULRND_AVX2(ymm4, selector1, zero_point0);

        //c[0, 8-15]
        F32_SCL_MULRND_AVX2(ymm5, selector2, zero_point1);

        //c[1, 0-7]
        F32_SCL_MULRND_AVX2(ymm6, selector1, zero_point0);

        //c[1, 8-15]
        F32_SCL_MULRND_AVX2(ymm7, selector2, zero_point1);

        //c[2, 0-7]
        F32_SCL_MULRND_AVX2(ymm8, selector1, zero_point0);

        //c[2, 8-15]
        F32_SCL_MULRND_AVX2(ymm9, selector2, zero_point1);
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
              _mm256_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 0 ) );
          selector2 =
              _mm256_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 1 ) );
          selector3 =
              _mm256_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 2 ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_BCAST_AVX2(zero_point0,0);
            BF16_F32_ZP_VECTOR_BCAST_AVX2(zero_point1,1);
            BF16_F32_ZP_VECTOR_BCAST_AVX2(zero_point2,2);
          }
          else
          {
            zero_point0 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 1 ) );
            zero_point2 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 2 ) );
           }
        }
        //c[0, 0-7]
        F32_SCL_MULRND_AVX2(ymm4, selector1, zero_point0);

        //c[0, 8-15]
        F32_SCL_MULRND_AVX2(ymm5, selector1, zero_point0);

        //c[1, 0-7]
        F32_SCL_MULRND_AVX2(ymm6, selector2, zero_point1);

        //c[1, 8-15]
        F32_SCL_MULRND_AVX2(ymm7, selector2, zero_point1);

        //c[2, 0-7]
        F32_SCL_MULRND_AVX2(ymm8, selector3, zero_point2);

        //c[2, 8-15]
        F32_SCL_MULRND_AVX2(ymm9, selector3, zero_point2);
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_3x16F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m256 scl_fctr1 = _mm256_setzero_ps();
      __m256 scl_fctr2 = _mm256_setzero_ps();
      __m256 scl_fctr3 = _mm256_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm256_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          scl_fctr2 =
            _mm256_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 1 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
        }
      }

      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,0,4,5);

          // c[1:0-15]
          BF16_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,1,6,7);

          // c[2:0-15]
          BF16_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,2,8,9);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr1,0,4,5);

          // c[1:0-15]
          BF16_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr2,scl_fctr2,1,6,7);

          // c[2:0-15]
          BF16_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr3,scl_fctr3,2,8,9);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,0,4,5);

          // c[1:0-15]
          F32_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,1,6,7);

          // c[2:0-15]
          F32_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,2,8,9);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr1,0,4,5);

          // c[1:0-15]
          F32_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr2,scl_fctr2,1,6,7);

          // c[2:0-15]
          F32_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr3,scl_fctr3,2,8,9);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_3x16F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m256 scl_fctr1 = _mm256_setzero_ps();
      __m256 scl_fctr2 = _mm256_setzero_ps();
      __m256 scl_fctr3 = _mm256_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm256_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          scl_fctr2 =
            _mm256_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 1 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
        }
      }

      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
        ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,0,4,5);

          // c[1:0-15]
          BF16_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,1,6,7);

          // c[2:0-15]
          BF16_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,2,8,9);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr1,0,4,5);

          // c[1:0-15]
          BF16_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr2,scl_fctr2,1,6,7);

          // c[2:0-15]
          BF16_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr3,scl_fctr3,2,8,9);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,0,4,5);

          // c[1:0-15]
          F32_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,1,6,7);

          // c[2:0-15]
          F32_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,2,8,9);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr1,0,4,5);

          // c[1:0-15]
          F32_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr2,scl_fctr2,1,6,7);

          // c[2:0-15]
          F32_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr3,scl_fctr3,2,8,9);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_3x16F:
    {
        ymm0 =
          _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
        __m256 z, dn;
        __m256i ex_out;

        // c[0,0-7]
        SWISH_F32_AVX2_DEF(ymm4, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[0,8-15]
        SWISH_F32_AVX2_DEF(ymm5, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[1,0-7]
        SWISH_F32_AVX2_DEF(ymm6, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[1,8-15]
        SWISH_F32_AVX2_DEF(ymm7, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[2,0-7]
        SWISH_F32_AVX2_DEF(ymm8, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[2,8-15]
        SWISH_F32_AVX2_DEF(ymm9, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_3x16F:
    {
      __m256 dn;
      __m256i q;

      // c[0,0-7]
      TANH_F32S_AVX2(ymm4, ymm0, ymm1, ymm2, ymm3, dn, q)

      // c[0,8-15]
      TANH_F32S_AVX2(ymm5, ymm0, ymm1, ymm2, ymm3, dn, q)

      // c[1,0-7]
      TANH_F32S_AVX2(ymm6, ymm0, ymm1, ymm2, ymm3, dn, q)

      // c[1,8-15]
      TANH_F32S_AVX2(ymm7, ymm0, ymm1, ymm2, ymm3, dn, q)

      // c[2,0-7]
      TANH_F32S_AVX2(ymm8, ymm0, ymm1, ymm2, ymm3, dn, q)

      // c[2,8-15]
      TANH_F32S_AVX2(ymm9, ymm0, ymm1, ymm2, ymm3, dn, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SIGMOID_3x16F:
    {
        __m256 z, dn;
        __m256i ex_out;

        // c[0,0-7]
        SIGMOID_F32_AVX2_DEF(ymm4, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[0,8-15]
        SIGMOID_F32_AVX2_DEF(ymm5, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[1,0-7]
        SIGMOID_F32_AVX2_DEF(ymm6, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[1,8-15]
        SIGMOID_F32_AVX2_DEF(ymm7, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[2,0-7]
        SIGMOID_F32_AVX2_DEF(ymm8, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[2,8-15]
        SIGMOID_F32_AVX2_DEF(ymm9, ymm1, ymm2, ymm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_3x16F_DISABLE:
    ;

    uint32_t tlsb, rounded, temp[8] = {0};
    int i;
    bfloat16* dest;

    if ( ( post_ops_attr.buf_downscale != NULL ) &&
      ( post_ops_attr.is_last_k == TRUE ) )
    {
      STORE_F32_BF16_YMM(ymm4, 0, 0);
      STORE_F32_BF16_YMM(ymm5, 0, 1);
      STORE_F32_BF16_YMM(ymm6, 1, 0);
      STORE_F32_BF16_YMM(ymm7, 1, 1);
      STORE_F32_BF16_YMM(ymm8, 2, 0);
      STORE_F32_BF16_YMM(ymm9, 2, 1);
    }
    else
    {
      _mm256_storeu_ps(cbuf, ymm4);
      _mm256_storeu_ps(cbuf + 8, ymm5);
      cbuf += rs_c;
      _mm256_storeu_ps(cbuf, ymm6);
      _mm256_storeu_ps(cbuf + 8, ymm7);
      cbuf += rs_c;
      _mm256_storeu_ps(cbuf, ymm8);
      _mm256_storeu_ps(cbuf + 8, ymm9);
    }
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_2x16)
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
    uint64_t k_iter = (uint64_t)k0;

    /*Declare the registers*/
    __m256 ymm0, ymm1, ymm2, ymm3;
    __m256 ymm4, ymm5, ymm6, ymm7;

    /* zero the accumulator registers */
    ZERO_ACC_YMM_4_REG(ymm4, ymm5, ymm6, ymm7);

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;
    float *_cbuf = NULL;

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


    if ( beta != 0.0 )
    {
      ymm3 = _mm256_broadcast_ss(&(beta));

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
        ( post_ops_attr.is_first_k == TRUE ) )
      {
        BF16_F32_C_BNZ_8(0,0,ymm0,ymm3,ymm4)
        BF16_F32_C_BNZ_8(0,1,ymm0,ymm3,ymm5)
        BF16_F32_C_BNZ_8(1,0,ymm0,ymm3,ymm6)
        BF16_F32_C_BNZ_8(1,1,ymm0,ymm3,ymm7)
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back
        F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm4)
        F32_C_BNZ_8(_cbuf+8,rs_c,ymm1,ymm3,ymm5)
        _cbuf += rs_c;
        F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm6)
        F32_C_BNZ_8(_cbuf+8,rs_c,ymm1,ymm3,ymm7)
      }
    }//betazero

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
          BF16_F32_BIAS_LOAD_AVX2( ymm0, 0 );
          BF16_F32_BIAS_LOAD_AVX2( ymm1, 1 );
        }
        else
        {
          ymm0 = _mm256_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 0 * 8 ) );
          ymm1 = _mm256_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 1 * 8 ) );
        }

        // c[0,0-7]
        ymm4 = _mm256_add_ps( ymm4, ymm0 );

        // c[0,8-15]
        ymm5 = _mm256_add_ps( ymm5, ymm1 );

        // c[1,0-7]
        ymm6 = _mm256_add_ps( ymm6, ymm0 );

        // c[1,8-15]
        ymm7 = _mm256_add_ps( ymm7, ymm1 );
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
          BF16_F32_BIAS_BCAST_AVX2(ymm0,0);
          BF16_F32_BIAS_BCAST_AVX2(ymm1,1);
        }
        else
        {
          ymm0 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 0 );
          ymm1 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 1 );
        }

        // c[0,0-7]
        ymm4 = _mm256_add_ps( ymm4, ymm0 );

        // c[0,8-15]
        ymm5 = _mm256_add_ps( ymm5, ymm0 );

        // c[1,0-7]
        ymm6 = _mm256_add_ps( ymm6, ymm1 );

        // c[1,8-15]
        ymm7 = _mm256_add_ps( ymm7, ymm1 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_2x16F:
    {
      ymm0 = _mm256_setzero_ps();

      // c[0,0-7]
      ymm4 = _mm256_max_ps( ymm4, ymm0 );

      // c[0,8-15]
      ymm5 = _mm256_max_ps( ymm5, ymm0 );

      // c[1,0-7]
      ymm6 = _mm256_max_ps( ymm6, ymm0 );

      // c[1,8-15]
      ymm7 = _mm256_max_ps( ymm7, ymm0 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_2x16F:
    {
      ymm0 =
        _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
      ymm1 = _mm256_setzero_ps();

      // c[0,0-7]
      RELU_SCALE_OP_F32S_AVX2(ymm4, ymm0, ymm1, ymm2)

      // c[0,8-15]
      RELU_SCALE_OP_F32S_AVX2(ymm5, ymm0, ymm1, ymm2)

      // c[1,0-7]
      RELU_SCALE_OP_F32S_AVX2(ymm6, ymm0, ymm1, ymm2)

      // c[1,8-15]
      RELU_SCALE_OP_F32S_AVX2(ymm7, ymm0, ymm1, ymm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_2x16F:
    {
      __m256 dn, x_tanh;
      __m256i q;

      // c[0,0-7]
      GELU_TANH_F32S_AVX2(ymm4, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      // c[0,8-15]
      GELU_TANH_F32S_AVX2(ymm5, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      // c[1,0-7]
      GELU_TANH_F32S_AVX2(ymm6, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      // c[1,8-15]
      GELU_TANH_F32S_AVX2(ymm7, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_2x16F:
    {
      // c[0,0-7]
      GELU_ERF_F32S_AVX2(ymm4, ymm0, ymm1, ymm2)

      // c[0,8-15]
      GELU_ERF_F32S_AVX2(ymm5, ymm0, ymm1, ymm2)

      // c[1,0-7]
      GELU_ERF_F32S_AVX2(ymm6, ymm0, ymm1, ymm2)

      // c[1,8-15]
      GELU_ERF_F32S_AVX2(ymm7, ymm0, ymm1, ymm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_2x16F:
    {
      ymm0 = _mm256_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      ymm1 = _mm256_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0,0-7]
      CLIP_F32S_AVX2(ymm4, ymm0, ymm1)

      // c[0,8-15]
      CLIP_F32S_AVX2(ymm5, ymm0, ymm1)

      // c[1,0-7]
      CLIP_F32S_AVX2(ymm6, ymm0, ymm1)

      // c[1,8-15]
      CLIP_F32S_AVX2(ymm7, ymm0, ymm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_DOWNSCALE_2x16F:
    {
      __m256 selector1 = _mm256_setzero_ps();
      __m256 selector2 = _mm256_setzero_ps();

      __m256 zero_point0 = _mm256_setzero_ps();
      __m256 zero_point1 = _mm256_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      if( post_ops_list_temp->scale_factor_len == 1 )
      {
        selector1 =
              _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector2 =
              _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if( is_bf16 == TRUE )
        {
          BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point0);
          BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point1);
        }
        else
        {
          zero_point0 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point1 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }

      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
              ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm256_loadu_ps ( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 8 ) );
          selector2 = _mm256_loadu_ps ( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 1 * 8 ) );
        }
        if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_LOAD_AVX2(zero_point0,0);
            BF16_F32_ZP_VECTOR_LOAD_AVX2(zero_point1,1);
          }
          else
          {
            zero_point0 = _mm256_loadu_ps ( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 0 * 8 ) );
            zero_point1 = _mm256_loadu_ps ( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 1 * 8 ) );
          }
        }
        //c[0, 0-7]
        F32_SCL_MULRND_AVX2(ymm4, selector1, zero_point0);

        //c[0, 8-15]
        F32_SCL_MULRND_AVX2(ymm5, selector2, zero_point1);

        //c[1, 0-7]
        F32_SCL_MULRND_AVX2(ymm6, selector1, zero_point0);

        //c[1, 8-15]
        F32_SCL_MULRND_AVX2(ymm7, selector2, zero_point1);
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
              _mm256_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 0 ) );
          selector2 =
              _mm256_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 1 ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_BCAST_AVX2(zero_point0,0);
            BF16_F32_ZP_VECTOR_BCAST_AVX2(zero_point1,1);
          }
          else
          {
            zero_point0 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 1 ) );
          }
        }
        //c[0, 0-7]
        F32_SCL_MULRND_AVX2(ymm4, selector1, zero_point0);

        //c[0, 8-15]
        F32_SCL_MULRND_AVX2(ymm5, selector1, zero_point0);

        //c[1, 0-7]
        F32_SCL_MULRND_AVX2(ymm6, selector2, zero_point1);

        //c[1, 8-15]
        F32_SCL_MULRND_AVX2(ymm7, selector2, zero_point1);
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_2x16F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m256 scl_fctr1 = _mm256_setzero_ps();
      __m256 scl_fctr2 = _mm256_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm256_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          scl_fctr2 =
            _mm256_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 1 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
        }
      }

      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,0,4,5);

          // c[1:0-15]
          BF16_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,1,6,7);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr1,0,4,5);

          // c[1:0-15]
          BF16_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr2,scl_fctr2,1,6,7);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,0,4,5);

          // c[1:0-15]
          F32_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,1,6,7);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr1,0,4,5);

          // c[1:0-15]
          F32_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr2,scl_fctr2,1,6,7);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_2x16F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m256 scl_fctr1 = _mm256_setzero_ps();
      __m256 scl_fctr2 = _mm256_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm256_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          scl_fctr2 =
            _mm256_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 1 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
        }
      }

      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
        ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,0,4,5);

          // c[1:0-15]
          BF16_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,1,6,7);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr1,0,4,5);

          // c[1:0-15]
          BF16_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr2,scl_fctr2,1,6,7);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,0,4,5);

          // c[1:0-15]
          F32_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,1,6,7);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr1,0,4,5);

          // c[1:0-15]
          F32_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr2,scl_fctr2,1,6,7);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_2x16F:
    {
        ymm0 =
          _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
        __m256 z, dn;
        __m256i ex_out;

        // c[0,0-7]
        SWISH_F32_AVX2_DEF(ymm4, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[0,8-15]
        SWISH_F32_AVX2_DEF(ymm5, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[1,0-7]
        SWISH_F32_AVX2_DEF(ymm6, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[1,8-15]
        SWISH_F32_AVX2_DEF(ymm7, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_2x16F:
    {
      __m256 dn;
      __m256i q;

      // c[0,0-7]
      TANH_F32S_AVX2(ymm4, ymm0, ymm1, ymm2, ymm3, dn, q)

      // c[0,8-15]
      TANH_F32S_AVX2(ymm5, ymm0, ymm1, ymm2, ymm3, dn, q)

      // c[1,0-7]
      TANH_F32S_AVX2(ymm6, ymm0, ymm1, ymm2, ymm3, dn, q)

      // c[1,8-15]
      TANH_F32S_AVX2(ymm7, ymm0, ymm1, ymm2, ymm3, dn, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SIGMOID_2x16F:
    {
        __m256 z, dn;
        __m256i ex_out;

        // c[0,0-7]
        SIGMOID_F32_AVX2_DEF(ymm4, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[0,8-15]
        SIGMOID_F32_AVX2_DEF(ymm5, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[1,0-7]
        SIGMOID_F32_AVX2_DEF(ymm6, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[1,8-15]
        SIGMOID_F32_AVX2_DEF(ymm7, ymm1, ymm2, ymm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_2x16F_DISABLE:
    ;

    uint32_t tlsb, rounded, temp[8] = {0};
    int i;
    bfloat16* dest;

    if ( ( post_ops_attr.buf_downscale != NULL ) &&
      ( post_ops_attr.is_last_k == TRUE ) )
    {
      STORE_F32_BF16_YMM(ymm4, 0, 0);
      STORE_F32_BF16_YMM(ymm5, 0, 1);
      STORE_F32_BF16_YMM(ymm6, 1, 0);
      STORE_F32_BF16_YMM(ymm7, 1, 1);
    }
    else
    {
      _mm256_storeu_ps(cbuf, ymm4);
      _mm256_storeu_ps(cbuf + 8, ymm5);
      cbuf += rs_c;
      _mm256_storeu_ps(cbuf, ymm6);
      _mm256_storeu_ps(cbuf + 8, ymm7);
    }
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_1x16)
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
    fflush(stdout);
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


    if ( beta != 0.0 )
    {
      //load c and multiply with beta and
      //add to accumulator and store back
      ymm3 = _mm256_broadcast_ss(&(beta));

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
        ( post_ops_attr.is_first_k == TRUE ) )
      {
        BF16_F32_C_BNZ_8(0,0,ymm0,ymm3,ymm4)
        BF16_F32_C_BNZ_8(0,1,ymm1,ymm3,ymm5)
      }
      else
      {
        F32_C_BNZ_8(cbuf,rs_c,ymm0,ymm3,ymm4)
        F32_C_BNZ_8(cbuf+8,rs_c,ymm1,ymm3,ymm5)
      }
    }//betazero

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
          BF16_F32_BIAS_LOAD_AVX2( ymm0, 0 );
          BF16_F32_BIAS_LOAD_AVX2( ymm1, 1 );
        }
        else
        {
          ymm0 = _mm256_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 0 * 8 ) );
          ymm1 = _mm256_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 1 * 8 ) );
        }

        // c[0,0-7]
        ymm4 = _mm256_add_ps( ymm4, ymm0 );

        // c[0,8-15]
        ymm5 = _mm256_add_ps( ymm5, ymm1 );
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
          BF16_F32_BIAS_BCAST_AVX2(ymm0,0);
        }
        else
        {
          ymm0 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
                  post_ops_attr.post_op_c_i + 0 );
        }

        // c[0,0-7]
        ymm4 = _mm256_add_ps( ymm4, ymm0 );

        // c[0,8-15]
        ymm5 = _mm256_add_ps( ymm5, ymm0 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_1x16F:
    {
      ymm0 = _mm256_setzero_ps();

      // c[0,0-7]
      ymm4 = _mm256_max_ps( ymm4, ymm0 );

      // c[0,8-15]
      ymm5 = _mm256_max_ps( ymm5, ymm0 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_1x16F:
    {
      ymm0 =
        _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
      ymm1 = _mm256_setzero_ps();

      // c[0,0-7]
      RELU_SCALE_OP_F32S_AVX2(ymm4, ymm0, ymm1, ymm2)

      // c[0,8-15]
      RELU_SCALE_OP_F32S_AVX2(ymm5, ymm0, ymm1, ymm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_1x16F:
    {
      __m256 dn, x_tanh;
      __m256i q;

      // c[0,0-7]
      GELU_TANH_F32S_AVX2(ymm4, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      // c[0,8-15]
      GELU_TANH_F32S_AVX2(ymm5, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_1x16F:
    {
      // c[0,0-7]
      GELU_ERF_F32S_AVX2(ymm4, ymm0, ymm1, ymm2)

      // c[0,8-15]
      GELU_ERF_F32S_AVX2(ymm5, ymm0, ymm1, ymm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_1x16F:
    {
      ymm0 = _mm256_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      ymm1 = _mm256_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0,0-7]
      CLIP_F32S_AVX2(ymm4, ymm0, ymm1)

      // c[0,8-15]
      CLIP_F32S_AVX2(ymm5, ymm0, ymm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_DOWNSCALE_1x16F:
    {
      __m256 selector1 = _mm256_setzero_ps();
      __m256 selector2 = _mm256_setzero_ps();

      __m256 zero_point0 = _mm256_setzero_ps();
      __m256 zero_point1 = _mm256_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );


      if( post_ops_list_temp->scale_factor_len == 1 )
      {
        selector1 =
              _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector2 =
              _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if( is_bf16 == TRUE )
        {
          BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point0);
          BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point1);
        }
        else
        {
          zero_point0 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point1 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }

      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
              ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm256_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 8 ) );
          selector2 = _mm256_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 1 * 8 ) );
        }
        if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_LOAD_AVX2(zero_point0,0);
            BF16_F32_ZP_VECTOR_LOAD_AVX2(zero_point1,1);
          }
          else
          {
            zero_point0 = _mm256_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 0 * 8 ) );
            zero_point1 = _mm256_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 1 * 8 ) );
          }
        }
        //c[0, 0-7]
        F32_SCL_MULRND_AVX2(ymm4, selector1, zero_point0);

        //c[0, 8-15]
        F32_SCL_MULRND_AVX2(ymm5, selector2, zero_point1);
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
              _mm256_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 0 ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_BCAST_AVX2(zero_point0,0);
          }
          else
          {
            zero_point0 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                post_ops_attr.post_op_c_i + 0 ) );
          }
        }
        //c[0, 0-7]
        F32_SCL_MULRND_AVX2(ymm4, selector1, zero_point0);

        //c[0, 8-15]
        F32_SCL_MULRND_AVX2(ymm5, selector1, zero_point0);
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_1x16F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m256 scl_fctr1 = _mm256_setzero_ps();
      __m256 scl_fctr2 = _mm256_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm256_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          scl_fctr2 =
            _mm256_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 1 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
        }
      }

      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,0,4,5);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr1,0,4,5);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,0,4,5);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr1,0,4,5);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_1x16F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m256 scl_fctr1 = _mm256_setzero_ps();
      __m256 scl_fctr2 = _mm256_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm256_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          scl_fctr2 =
            _mm256_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 1 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
        }
      }

      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
        ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,0,4,5);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr1,0,4,5);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,0,4,5);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr1,0,4,5);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_1x16F:
    {
        ymm0 =
          _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
        __m256 z, dn;
        __m256i ex_out;

        // c[0,0-7]
        SWISH_F32_AVX2_DEF(ymm4, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[0,8-15]
        SWISH_F32_AVX2_DEF(ymm5, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_1x16F:
    {
      __m256 dn;
      __m256i q;

      // c[0,0-7]
      TANH_F32S_AVX2(ymm4, ymm0, ymm1, ymm2, ymm3, dn, q)

      // c[0,8-15]
      TANH_F32S_AVX2(ymm5, ymm0, ymm1, ymm2, ymm3, dn, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SIGMOID_1x16F:
    {
        __m256 z, dn;
        __m256i ex_out;

        // c[0,0-7]
        SIGMOID_F32_AVX2_DEF(ymm4, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[0,8-15]
        SIGMOID_F32_AVX2_DEF(ymm5, ymm1, ymm2, ymm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_1x16F_DISABLE:
    ;

    uint32_t tlsb, rounded, temp[8] = {0};
    int i;
    bfloat16* dest;

    if ( ( post_ops_attr.buf_downscale != NULL ) &&
      ( post_ops_attr.is_last_k == TRUE ) )
    {
      STORE_F32_BF16_YMM(ymm4, 0, 0);
      STORE_F32_BF16_YMM(ymm5, 0, 1);
    }
    else
    {
      _mm256_storeu_ps(cbuf, ymm4);
      _mm256_storeu_ps(cbuf + 8, ymm5);
    }
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_5x8)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_5x8F_DISABLE,
              &&POST_OPS_BIAS_5x8F,
              &&POST_OPS_RELU_5x8F,
              &&POST_OPS_RELU_SCALE_5x8F,
              &&POST_OPS_GELU_TANH_5x8F,
              &&POST_OPS_GELU_ERF_5x8F,
              &&POST_OPS_CLIP_5x8F,
              &&POST_OPS_DOWNSCALE_5x8F,
              &&POST_OPS_MATRIX_ADD_5x8F,
              &&POST_OPS_SWISH_5x8F,
              &&POST_OPS_MATRIX_MUL_5x8F,
              &&POST_OPS_TANH_5x8F,
              &&POST_OPS_SIGMOID_5x8F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    /*Declare the registers*/
    __m256 ymm0, ymm1, ymm2, ymm3;
    __m256 ymm4, ymm6, ymm8, ymm10;
    __m256 ymm12;

    /* zero the accumulator registers */
    ZERO_ACC_YMM_4_REG(ymm4, ymm6, ymm2, ymm3);
    ZERO_ACC_YMM_4_REG(ymm8, ymm10, ymm12, ymm0);

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;
    float *_cbuf = NULL;

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


    if ( beta != 0.0 )
    {
      ymm3 = _mm256_broadcast_ss(&(beta));

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
        ( post_ops_attr.is_first_k == TRUE ) )
      {
        BF16_F32_C_BNZ_8(0,0,ymm0,ymm3,ymm4)
        BF16_F32_C_BNZ_8(1,0,ymm0,ymm3,ymm6)
        BF16_F32_C_BNZ_8(2,0,ymm0,ymm3,ymm8)
        BF16_F32_C_BNZ_8(3,0,ymm0,ymm3,ymm10)
        BF16_F32_C_BNZ_8(4,0,ymm0,ymm3,ymm12)
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back
        F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm4)
        _cbuf += rs_c;
        F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm6)
        _cbuf += rs_c;
        F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm8)
        _cbuf += rs_c;
        F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm10)
        _cbuf += rs_c;
        F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm12)
      }
    }//betazero

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_5x8F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          BF16_F32_BIAS_LOAD_AVX2( ymm0, 0 );
        }
        else
        {
          ymm0 = _mm256_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 0 * 8 ) );
        }

        // c[0,0-7]
        ymm4 = _mm256_add_ps( ymm4, ymm0 );

        // c[1,0-7]
        ymm6 = _mm256_add_ps( ymm6, ymm0 );

        // c[2,0-7]
        ymm8 = _mm256_add_ps( ymm8, ymm0 );

        // c[3,0-7]
        ymm10 = _mm256_add_ps( ymm10, ymm0 );

        // c[4,0-7]
        ymm12 = _mm256_add_ps( ymm12, ymm0 );
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
          BF16_F32_BIAS_BCAST_AVX2(ymm0,0);
          BF16_F32_BIAS_BCAST_AVX2(ymm1,1);
          BF16_F32_BIAS_BCAST_AVX2(ymm2,2);
          BF16_F32_BIAS_BCAST_AVX2(ymm3,3);
        }
        else
        {
          ymm0 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 0 );
          ymm1 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 1 );
          ymm2 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 2 );
          ymm3 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 3 );
        }

        // c[0,0-7]
        ymm4 = _mm256_add_ps( ymm4, ymm0 );

        // c[1,0-7]
        ymm6 = _mm256_add_ps( ymm6, ymm1 );

        // c[2,0-7]
        ymm8 = _mm256_add_ps( ymm8, ymm2 );

        // c[3,0-7]
        ymm10 = _mm256_add_ps( ymm10, ymm3 );

        if( post_ops_list_temp->stor_type == BF16 )
        {
          BF16_F32_BIAS_BCAST_AVX2(ymm0,4);
        }
        else
        {
          ymm0 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 4 );
        }

        // c[4,0-7]
        ymm12 = _mm256_add_ps( ymm12, ymm0 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_5x8F:
    {
      ymm0 = _mm256_setzero_ps();

      // c[0,0-7]
      ymm4 = _mm256_max_ps( ymm4, ymm0 );

      // c[1,0-7]
      ymm6 = _mm256_max_ps( ymm6, ymm0 );

      // c[2,0-7]
      ymm8 = _mm256_max_ps( ymm8, ymm0 );

      // c[3,0-7]
      ymm10 = _mm256_max_ps( ymm10, ymm0 );

      // c[4,0-7]
      ymm12 = _mm256_max_ps( ymm12, ymm0 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_5x8F:
    {
      ymm0 =
        _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
      ymm1 = _mm256_setzero_ps();

      // c[0,0-7]
      RELU_SCALE_OP_F32S_AVX2(ymm4, ymm0, ymm1, ymm2)

      // c[1,0-7]
      RELU_SCALE_OP_F32S_AVX2(ymm6, ymm0, ymm1, ymm2)

      // c[2,0-7]
      RELU_SCALE_OP_F32S_AVX2(ymm8, ymm0, ymm1, ymm2)

      // c[3,0-7]
      RELU_SCALE_OP_F32S_AVX2(ymm10, ymm0, ymm1, ymm2)

      // c[4,0-7]
      RELU_SCALE_OP_F32S_AVX2(ymm12, ymm0, ymm1, ymm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_5x8F:
    {
      __m256 dn, x_tanh;
      __m256i q;

      // c[0,0-7]
      GELU_TANH_F32S_AVX2(ymm4, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      // c[1,0-7]
      GELU_TANH_F32S_AVX2(ymm6, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      // c[2,0-7]
      GELU_TANH_F32S_AVX2(ymm8, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      // c[3,0-7]
      GELU_TANH_F32S_AVX2(ymm10, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      // c[4,0-7]
      GELU_TANH_F32S_AVX2(ymm12, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_5x8F:
    {
      // c[0,0-7]
      GELU_ERF_F32S_AVX2(ymm4, ymm0, ymm1, ymm2)

      // c[1,0-7]
      GELU_ERF_F32S_AVX2(ymm6, ymm0, ymm1, ymm2)

      // c[2,0-7]
      GELU_ERF_F32S_AVX2(ymm8, ymm0, ymm1, ymm2)

      // c[3,0-7]
      GELU_ERF_F32S_AVX2(ymm10, ymm0, ymm1, ymm2)

      // c[4,0-7]
      GELU_ERF_F32S_AVX2(ymm12, ymm0, ymm1, ymm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_5x8F:
    {
      ymm0 = _mm256_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      ymm1 = _mm256_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0,0-7]
      CLIP_F32S_AVX2(ymm4, ymm0, ymm1)

      // c[1,0-7]
      CLIP_F32S_AVX2(ymm6, ymm0, ymm1)

      // c[2,0-7]
      CLIP_F32S_AVX2(ymm8, ymm0, ymm1)

      // c[3,0-7]
      CLIP_F32S_AVX2(ymm10, ymm0, ymm1)

      // c[4,0-7]
      CLIP_F32S_AVX2(ymm12, ymm0, ymm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_DOWNSCALE_5x8F:
    {
      __m256 selector1 = _mm256_setzero_ps();
      __m256 selector2 = _mm256_setzero_ps();
      __m256 selector3 = _mm256_setzero_ps();
      __m256 selector4 = _mm256_setzero_ps();
      __m256 selector5 = _mm256_setzero_ps();

      __m256 zero_point0 = _mm256_setzero_ps();
      __m256 zero_point1 = _mm256_setzero_ps();
      __m256 zero_point2 = _mm256_setzero_ps();
      __m256 zero_point3 = _mm256_setzero_ps();
      __m256 zero_point4 = _mm256_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                    ( ( post_ops_list_temp->stor_type == NONE ) &&
                      ( post_ops_attr.c_stor_type == BF16 ) );

      if( post_ops_list_temp->scale_factor_len == 1 )
      {
        selector1 =
              _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector2 =
              _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector3 =
              _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector4 =
              _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector5 =
              _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if( is_bf16 == TRUE )
        {
          BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point0);
          BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point1);
          BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point2);
          BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point3);
          BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point4);
        }
        else
        {
          zero_point0 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point1 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point2 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point3 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point4 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }

      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
              ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm256_loadu_ps ( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 8 ) );
        }
        if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_LOAD_AVX2(zero_point0,0);
          }
          else
          {
            zero_point0 = _mm256_loadu_ps ( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 0 * 8 ) );
          }
        }
        //c[0, 0-7]
        F32_SCL_MULRND_AVX2(ymm4, selector1, zero_point0);

        //c[1, 0-7]
        F32_SCL_MULRND_AVX2(ymm6, selector1, zero_point0);

        //c[2, 0-7]
        F32_SCL_MULRND_AVX2(ymm8, selector1, zero_point0);

        //c[3, 0-7]
        F32_SCL_MULRND_AVX2(ymm10, selector1, zero_point0);

        //c[4, 0-7]
        F32_SCL_MULRND_AVX2(ymm12, selector1, zero_point0);
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
              _mm256_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 0 ) );
          selector2 =
              _mm256_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 1 ) );
          selector3 =
              _mm256_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 2 ) );
          selector4 =
              _mm256_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 3 ) );
          selector5 =
              _mm256_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 4 ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_BCAST_AVX2(zero_point0,0);
            BF16_F32_ZP_VECTOR_BCAST_AVX2(zero_point1,1);
            BF16_F32_ZP_VECTOR_BCAST_AVX2(zero_point2,2);
            BF16_F32_ZP_VECTOR_BCAST_AVX2(zero_point3,3);
            BF16_F32_ZP_VECTOR_BCAST_AVX2(zero_point4,4);
          }
          else
          {
            zero_point0 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 1 ) );
            zero_point2 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 2 ) );
            zero_point3 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 3 ) );
            zero_point4 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 4 ) );
          }
        }
        //c[0, 0-7]
        F32_SCL_MULRND_AVX2(ymm4, selector1, zero_point0);

        //c[1, 0-7]
        F32_SCL_MULRND_AVX2(ymm6, selector2, zero_point1);

        //c[2, 0-7]
        F32_SCL_MULRND_AVX2(ymm8, selector3, zero_point2);

        //c[3, 0-7]
        F32_SCL_MULRND_AVX2(ymm10, selector4, zero_point3);

        //c[4, 0-7]
        F32_SCL_MULRND_AVX2(ymm12, selector5, zero_point4);
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_5x8F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

        bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m256 scl_fctr1 = _mm256_setzero_ps();
      __m256 scl_fctr2 = _mm256_setzero_ps();
      __m256 scl_fctr3 = _mm256_setzero_ps();
      __m256 scl_fctr4 = _mm256_setzero_ps();
      __m256 scl_fctr5 = _mm256_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr4 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr5 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm256_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
          scl_fctr4 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 3 ) );
          scl_fctr5 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 4 ) );
        }
      }
      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,1,6);

          // c[2:0-15]
          BF16_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,2,8);

          // c[3:0-15]
          BF16_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,3,10);

          // c[4:0-15]
          BF16_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,4,12);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_ADD_1COL(ymm1,scl_fctr2,1,6);

          // c[2:0-15]
          BF16_F32_MATRIX_ADD_1COL(ymm1,scl_fctr3,2,8);

          // c[3:0-15]
          BF16_F32_MATRIX_ADD_1COL(ymm1,scl_fctr4,3,10);

          // c[4:0-15]
          BF16_F32_MATRIX_ADD_1COL(ymm1,scl_fctr5,4,12);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,1,6);

          // c[2:0-15]
          F32_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,2,8);

          // c[3:0-15]
          F32_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,3,10);

          // c[4:0-15]
          F32_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,4,12);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_ADD_1COL(ymm1,scl_fctr2,1,6);

          // c[2:0-15]
          F32_F32_MATRIX_ADD_1COL(ymm1,scl_fctr3,2,8);

          // c[3:0-15]
          F32_F32_MATRIX_ADD_1COL(ymm1,scl_fctr4,3,10);

          // c[4:0-15]
          F32_F32_MATRIX_ADD_1COL(ymm1,scl_fctr5,4,12);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_5x8F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m256 scl_fctr1 = _mm256_setzero_ps();
      __m256 scl_fctr2 = _mm256_setzero_ps();
      __m256 scl_fctr3 = _mm256_setzero_ps();
      __m256 scl_fctr4 = _mm256_setzero_ps();
      __m256 scl_fctr5 = _mm256_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr4 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr5 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm256_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
          scl_fctr4 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 3 ) );
          scl_fctr5 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 4 ) );
        }
      }

      if( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,1,6);

          // c[2:0-15]
          BF16_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,2,8);

          // c[3:0-15]
          BF16_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,3,10);

          // c[4:0-15]
          BF16_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,4,12);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_MUL_1COL(ymm1,scl_fctr2,1,6);

          // c[2:0-15]
          BF16_F32_MATRIX_MUL_1COL(ymm1,scl_fctr3,2,8);

          // c[3:0-15]
          BF16_F32_MATRIX_MUL_1COL(ymm1,scl_fctr4,3,10);

          // c[4:0-15]
          BF16_F32_MATRIX_MUL_1COL(ymm1,scl_fctr5,4,12);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,1,6);

          // c[2:0-15]
          F32_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,2,8);

          // c[3:0-15]
          F32_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,3,10);

          // c[4:0-15]
          F32_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,4,12);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_MUL_1COL(ymm1,scl_fctr2,1,6);

          // c[2:0-15]
          F32_F32_MATRIX_MUL_1COL(ymm1,scl_fctr3,2,8);

          // c[3:0-15]
          F32_F32_MATRIX_MUL_1COL(ymm1,scl_fctr4,3,10);

          // c[4:0-15]
          F32_F32_MATRIX_MUL_1COL(ymm1,scl_fctr5,4,12);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_5x8F:
    {
        ymm0 =
          _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
        __m256 z, dn;
        __m256i ex_out;

        // c[0,0-7]
        SWISH_F32_AVX2_DEF(ymm4, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[1,0-7]
        SWISH_F32_AVX2_DEF(ymm6, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[2,0-7]
        SWISH_F32_AVX2_DEF(ymm8, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[3,0-7]
        SWISH_F32_AVX2_DEF(ymm10, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[4,0-7]
        SWISH_F32_AVX2_DEF(ymm12, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_5x8F:
    {
      __m256 dn;
      __m256i q;

      // c[0,0-7]
      TANH_F32S_AVX2(ymm4, ymm0, ymm1, ymm2, ymm3, dn, q)

      // c[1,0-7]
      TANH_F32S_AVX2(ymm6, ymm0, ymm1, ymm2, ymm3, dn, q)

      // c[2,0-7]
      TANH_F32S_AVX2(ymm8, ymm0, ymm1, ymm2, ymm3, dn, q)

      // c[3,0-7]
      TANH_F32S_AVX2(ymm10, ymm0, ymm1, ymm2, ymm3, dn, q)

      // c[4,0-7]
      TANH_F32S_AVX2(ymm12, ymm0, ymm1, ymm2, ymm3, dn, q)


      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SIGMOID_5x8F:
    {
        __m256 z, dn;
        __m256i ex_out;

        // c[0,0-7]
        SIGMOID_F32_AVX2_DEF(ymm4, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[1,0-7]
        SIGMOID_F32_AVX2_DEF(ymm6, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[2,0-7]
        SIGMOID_F32_AVX2_DEF(ymm8, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[3,0-7]
        SIGMOID_F32_AVX2_DEF(ymm10, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[4,0-7]
        SIGMOID_F32_AVX2_DEF(ymm12, ymm1, ymm2, ymm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_5x8F_DISABLE:
    ;

    uint32_t tlsb, rounded, temp[8] = {0};
    int i;
    bfloat16* dest;

    if ( ( post_ops_attr.buf_downscale != NULL ) &&
      ( post_ops_attr.is_last_k == TRUE ) )
    {
      STORE_F32_BF16_YMM(ymm4, 0, 0)
      STORE_F32_BF16_YMM(ymm6, 1, 0)
      STORE_F32_BF16_YMM(ymm8, 2, 0)
      STORE_F32_BF16_YMM(ymm10, 3, 0)
      STORE_F32_BF16_YMM(ymm12, 4, 0)
    }
    else
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
    }
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_4x8)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_4x8F_DISABLE,
              &&POST_OPS_BIAS_4x8F,
              &&POST_OPS_RELU_4x8F,
              &&POST_OPS_RELU_SCALE_4x8F,
              &&POST_OPS_GELU_TANH_4x8F,
              &&POST_OPS_GELU_ERF_4x8F,
              &&POST_OPS_CLIP_4x8F,
              &&POST_OPS_DOWNSCALE_4x8F,
              &&POST_OPS_MATRIX_ADD_4x8F,
              &&POST_OPS_SWISH_4x8F,
              &&POST_OPS_MATRIX_MUL_4x8F,
              &&POST_OPS_TANH_4x8F,
              &&POST_OPS_SIGMOID_4x8F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    /*Declare the registers*/
    __m256 ymm0, ymm1, ymm2, ymm3;
    __m256 ymm4, ymm6, ymm8, ymm10;

    /* zero the accumulator registers */
    ZERO_ACC_YMM_4_REG(ymm4, ymm6, ymm8, ymm10);

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;
    float *_cbuf = NULL;

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


    if ( beta != 0.0 )
    {
      ymm3 = _mm256_broadcast_ss(&(beta));

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
        ( post_ops_attr.is_first_k == TRUE ) )
      {
        BF16_F32_C_BNZ_8(0,0,ymm0,ymm3,ymm4)
        BF16_F32_C_BNZ_8(1,0,ymm0,ymm3,ymm6)
        BF16_F32_C_BNZ_8(2,0,ymm0,ymm3,ymm8)
        BF16_F32_C_BNZ_8(3,0,ymm0,ymm3,ymm10)
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back
        F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm4)
        _cbuf += rs_c;
        F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm6)
        _cbuf += rs_c;
        F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm8)
        _cbuf += rs_c;
        F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm10)
      }
    }//betazero

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_4x8F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          BF16_F32_BIAS_LOAD_AVX2( ymm0, 0 );
        }
        else
        {
          ymm0 = _mm256_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
                  post_ops_attr.post_op_c_j + ( 0 * 8 ) );
        }

        // c[0,0-7]
        ymm4 = _mm256_add_ps( ymm4, ymm0 );

        // c[1,0-7]
        ymm6 = _mm256_add_ps( ymm6, ymm0 );

        // c[2,0-7]
        ymm8 = _mm256_add_ps( ymm8, ymm0 );

        // c[3,0-7]
        ymm10 = _mm256_add_ps( ymm10, ymm0 );
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
          BF16_F32_BIAS_BCAST_AVX2(ymm0,0);
          BF16_F32_BIAS_BCAST_AVX2(ymm1,1);
          BF16_F32_BIAS_BCAST_AVX2(ymm2,2);
          BF16_F32_BIAS_BCAST_AVX2(ymm3,3);
        }
        else
        {
          ymm0 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 0 );
          ymm1 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 1 );
          ymm2 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 2 );
          ymm3 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 3 );
        }

        // c[0,0-7]
        ymm4 = _mm256_add_ps( ymm4, ymm0 );

        // c[1,0-7]
        ymm6 = _mm256_add_ps( ymm6, ymm1 );

        // c[2,0-7]
        ymm8 = _mm256_add_ps( ymm8, ymm2 );

        // c[3,0-7]
        ymm10 = _mm256_add_ps( ymm10, ymm3 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_4x8F:
    {
      ymm0 = _mm256_setzero_ps();

      // c[0,0-7]
      ymm4 = _mm256_max_ps( ymm4, ymm0 );

      // c[1,0-7]
      ymm6 = _mm256_max_ps( ymm6, ymm0 );

      // c[2,0-7]
      ymm8 = _mm256_max_ps( ymm8, ymm0 );

      // c[3,0-7]
      ymm10 = _mm256_max_ps( ymm10, ymm0 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_4x8F:
    {
      ymm0 =
        _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
      ymm1 = _mm256_setzero_ps();

      // c[0,0-7]
      RELU_SCALE_OP_F32S_AVX2(ymm4, ymm0, ymm1, ymm2)

      // c[1,0-7]
      RELU_SCALE_OP_F32S_AVX2(ymm6, ymm0, ymm1, ymm2)

      // c[2,0-7]
      RELU_SCALE_OP_F32S_AVX2(ymm8, ymm0, ymm1, ymm2)

      // c[3,0-7]
      RELU_SCALE_OP_F32S_AVX2(ymm10, ymm0, ymm1, ymm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_4x8F:
    {
      __m256 dn, x_tanh;
      __m256i q;

      // c[0,0-7]
      GELU_TANH_F32S_AVX2(ymm4, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      // c[1,0-7]
      GELU_TANH_F32S_AVX2(ymm6, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      // c[2,0-7]
      GELU_TANH_F32S_AVX2(ymm8, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      // c[3,0-7]
      GELU_TANH_F32S_AVX2(ymm10, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_4x8F:
    {
      // c[0,0-7]
      GELU_ERF_F32S_AVX2(ymm4, ymm0, ymm1, ymm2)

      // c[1,0-7]
      GELU_ERF_F32S_AVX2(ymm6, ymm0, ymm1, ymm2)

      // c[2,0-7]
      GELU_ERF_F32S_AVX2(ymm8, ymm0, ymm1, ymm2)

      // c[3,0-7]
      GELU_ERF_F32S_AVX2(ymm10, ymm0, ymm1, ymm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_4x8F:
    {
      ymm0 = _mm256_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      ymm1 = _mm256_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0,0-7]
      CLIP_F32S_AVX2(ymm4, ymm0, ymm1)

      // c[1,0-7]
      CLIP_F32S_AVX2(ymm6, ymm0, ymm1)

      // c[2,0-7]
      CLIP_F32S_AVX2(ymm8, ymm0, ymm1)

      // c[3,0-7]
      CLIP_F32S_AVX2(ymm10, ymm0, ymm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_DOWNSCALE_4x8F:
    {
      __m256 selector1 = _mm256_setzero_ps();
      __m256 selector2 = _mm256_setzero_ps();
      __m256 selector3 = _mm256_setzero_ps();
      __m256 selector4 = _mm256_setzero_ps();

      __m256 zero_point0 = _mm256_setzero_ps();
      __m256 zero_point1 = _mm256_setzero_ps();
      __m256 zero_point2 = _mm256_setzero_ps();
      __m256 zero_point3 = _mm256_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                    ( ( post_ops_list_temp->stor_type == NONE ) &&
                      ( post_ops_attr.c_stor_type == BF16 ) );

      if( post_ops_list_temp->scale_factor_len == 1 )
      {
        selector1 =
              _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector2 =
              _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector3 =
              _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector4 =
              _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if( is_bf16 == TRUE )
        {
          BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point0);
          BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point1);
          BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point2);
          BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point3);
        }
        else
        {
          zero_point0 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point1 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point2 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point3 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }

      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
              ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm256_loadu_ps ( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 8 ) );
        }
        if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_LOAD_AVX2(zero_point0,0);
          }
          else
          {
            zero_point0 = _mm256_loadu_ps ( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 0 * 8 ) );
          }
        }
        //c[0, 0-7]
        F32_SCL_MULRND_AVX2(ymm4, selector1, zero_point0);

        //c[1, 0-7]
        F32_SCL_MULRND_AVX2(ymm6, selector1, zero_point0);

        //c[2, 0-7]
        F32_SCL_MULRND_AVX2(ymm8, selector1, zero_point0);

        //c[3, 0-7]
        F32_SCL_MULRND_AVX2(ymm10, selector1, zero_point0);
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
              _mm256_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 0 ) );
          selector2 =
              _mm256_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 1 ) );
          selector3 =
              _mm256_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 2 ) );
          selector4 =
              _mm256_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 3 ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_BCAST_AVX2(zero_point0,0);
            BF16_F32_ZP_VECTOR_BCAST_AVX2(zero_point1,1);
            BF16_F32_ZP_VECTOR_BCAST_AVX2(zero_point2,2);
            BF16_F32_ZP_VECTOR_BCAST_AVX2(zero_point3,3);
          }
          else
          {
            zero_point0 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 1 ) );
            zero_point2 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 2 ) );
            zero_point3 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 3 ) );
          }
        }
        //c[0, 0-7]
        F32_SCL_MULRND_AVX2(ymm4, selector1, zero_point0);

        //c[1, 0-7]
        F32_SCL_MULRND_AVX2(ymm6, selector2, zero_point1);

        //c[2, 0-7]
        F32_SCL_MULRND_AVX2(ymm8, selector3, zero_point2);

        //c[3, 0-7]
        F32_SCL_MULRND_AVX2(ymm10, selector4, zero_point3);
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_4x8F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m256 scl_fctr1 = _mm256_setzero_ps();
      __m256 scl_fctr2 = _mm256_setzero_ps();
      __m256 scl_fctr3 = _mm256_setzero_ps();
      __m256 scl_fctr4 = _mm256_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr4 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm256_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
          scl_fctr4 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 3 ) );
        }
      }

      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,1,6);

          // c[2:0-15]
          BF16_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,2,8);

          // c[3:0-15]
          BF16_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,3,10);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_ADD_1COL(ymm1,scl_fctr2,1,6);

          // c[2:0-15]
          BF16_F32_MATRIX_ADD_1COL(ymm1,scl_fctr3,2,8);

          // c[3:0-15]
          BF16_F32_MATRIX_ADD_1COL(ymm1,scl_fctr4,3,10);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,1,6);

          // c[2:0-15]
          F32_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,2,8);

          // c[3:0-15]
          F32_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,3,10);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_ADD_1COL(ymm1,scl_fctr2,1,6);

          // c[2:0-15]
          F32_F32_MATRIX_ADD_1COL(ymm1,scl_fctr3,2,8);

          // c[3:0-15]
          F32_F32_MATRIX_ADD_1COL(ymm1,scl_fctr4,3,10);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_4x8F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m256 scl_fctr1 = _mm256_setzero_ps();
      __m256 scl_fctr2 = _mm256_setzero_ps();
      __m256 scl_fctr3 = _mm256_setzero_ps();
      __m256 scl_fctr4 = _mm256_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr4 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm256_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
          scl_fctr4 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 3 ) );
        }
      }
      if( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,1,6);

          // c[2:0-15]
          BF16_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,2,8);

          // c[3:0-15]
          BF16_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,3,10);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_MUL_1COL(ymm1,scl_fctr2,1,6);

          // c[2:0-15]
          BF16_F32_MATRIX_MUL_1COL(ymm1,scl_fctr3,2,8);

          // c[3:0-15]
          BF16_F32_MATRIX_MUL_1COL(ymm1,scl_fctr4,3,10);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,1,6);

          // c[2:0-15]
          F32_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,2,8);

          // c[3:0-15]
          F32_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,3,10);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_MUL_1COL(ymm1,scl_fctr2,1,6);

          // c[2:0-15]
          F32_F32_MATRIX_MUL_1COL(ymm1,scl_fctr3,2,8);

          // c[3:0-15]
          F32_F32_MATRIX_MUL_1COL(ymm1,scl_fctr4,3,10);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_4x8F:
    {
        ymm0 =
          _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
        __m256 z, dn;
        __m256i ex_out;

        // c[0,0-7]
        SWISH_F32_AVX2_DEF(ymm4, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[1,0-7]
        SWISH_F32_AVX2_DEF(ymm6, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[2,0-7]
        SWISH_F32_AVX2_DEF(ymm8, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[3,0-7]
        SWISH_F32_AVX2_DEF(ymm10, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_4x8F:
    {
      __m256 dn;
      __m256i q;

      // c[0,0-7]
      TANH_F32S_AVX2(ymm4, ymm0, ymm1, ymm2, ymm3, dn, q)

      // c[1,0-7]
      TANH_F32S_AVX2(ymm6, ymm0, ymm1, ymm2, ymm3, dn, q)

      // c[2,0-7]
      TANH_F32S_AVX2(ymm8, ymm0, ymm1, ymm2, ymm3, dn, q)

      // c[3,0-7]
      TANH_F32S_AVX2(ymm10, ymm0, ymm1, ymm2, ymm3, dn, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SIGMOID_4x8F:
    {
        __m256 z, dn;
        __m256i ex_out;

        // c[0,0-7]
        SIGMOID_F32_AVX2_DEF(ymm4, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[1,0-7]
        SIGMOID_F32_AVX2_DEF(ymm6, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[2,0-7]
        SIGMOID_F32_AVX2_DEF(ymm8, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[3,0-7]
        SIGMOID_F32_AVX2_DEF(ymm10, ymm1, ymm2, ymm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_4x8F_DISABLE:
    ;

    uint32_t tlsb, rounded, temp[8] = {0};
    int i;
    bfloat16* dest;

    if ( ( post_ops_attr.buf_downscale != NULL ) &&
      ( post_ops_attr.is_last_k == TRUE ) )
    {
      STORE_F32_BF16_YMM(ymm4, 0, 0)
      STORE_F32_BF16_YMM(ymm6, 1, 0)
      STORE_F32_BF16_YMM(ymm8, 2, 0)
      STORE_F32_BF16_YMM(ymm10, 3, 0)
    }
    else
    {
      _mm256_storeu_ps(cbuf, ymm4);
      cbuf += rs_c;
      _mm256_storeu_ps(cbuf, ymm6);
      cbuf += rs_c;
      _mm256_storeu_ps(cbuf, ymm8);
      cbuf += rs_c;
      _mm256_storeu_ps(cbuf, ymm10);
    }
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_3x8)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_3x8F_DISABLE,
              &&POST_OPS_BIAS_3x8F,
              &&POST_OPS_RELU_3x8F,
              &&POST_OPS_RELU_SCALE_3x8F,
              &&POST_OPS_GELU_TANH_3x8F,
              &&POST_OPS_GELU_ERF_3x8F,
              &&POST_OPS_CLIP_3x8F,
              &&POST_OPS_DOWNSCALE_3x8F,
              &&POST_OPS_MATRIX_ADD_3x8F,
              &&POST_OPS_SWISH_3x8F,
              &&POST_OPS_MATRIX_MUL_3x8F,
              &&POST_OPS_TANH_3x8F,
              &&POST_OPS_SIGMOID_3x8F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    /*Declare the registers*/
    __m256 ymm0, ymm1, ymm2, ymm3;
    __m256 ymm4, ymm6, ymm8;

    /* zero the accumulator registers */
    ZERO_ACC_YMM_4_REG(ymm4, ymm6, ymm2, ymm8);

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;
    float *_cbuf = NULL;

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

    if ( beta != 0.0 )
    {
      ymm3 = _mm256_broadcast_ss(&(beta));

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
        ( post_ops_attr.is_first_k == TRUE ) )
      {
        BF16_F32_C_BNZ_8(0,0,ymm0,ymm3,ymm4)
        BF16_F32_C_BNZ_8(1,0,ymm0,ymm3,ymm6)
        BF16_F32_C_BNZ_8(2,0,ymm0,ymm3,ymm8)
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back
        F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm4)
        _cbuf += rs_c;
        F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm6)
        _cbuf += rs_c;
        F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm8)
      }
    }//betazero

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_3x8F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          BF16_F32_BIAS_LOAD_AVX2( ymm0, 0 );
        }
        else
        {
          ymm0 = _mm256_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
            post_ops_attr.post_op_c_j + ( 0 * 8 ) );
        }

        // c[0,0-7]
        ymm4 = _mm256_add_ps( ymm4, ymm0 );

        // c[1,0-7]
        ymm6 = _mm256_add_ps( ymm6, ymm0 );

        // c[2,0-7]
        ymm8 = _mm256_add_ps( ymm8, ymm0 );
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
          BF16_F32_BIAS_BCAST_AVX2(ymm0,0);
          BF16_F32_BIAS_BCAST_AVX2(ymm1,1);
          BF16_F32_BIAS_BCAST_AVX2(ymm2,2);
        }
        else
        {
          ymm0 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 0 );
          ymm1 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 1 );
          ymm2 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 2 );
        }

        // c[0,0-7]
        ymm4 = _mm256_add_ps( ymm4, ymm0 );

        // c[1,0-7]
        ymm6 = _mm256_add_ps( ymm6, ymm1 );

        // c[2,0-7]
        ymm8 = _mm256_add_ps( ymm8, ymm2 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_3x8F:
    {
      ymm0 = _mm256_setzero_ps();

      // c[0,0-7]
      ymm4 = _mm256_max_ps( ymm4, ymm0 );

      // c[1,0-7]
      ymm6 = _mm256_max_ps( ymm6, ymm0 );

      // c[2,0-7]
      ymm8 = _mm256_max_ps( ymm8, ymm0 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_3x8F:
    {
      ymm0 =
        _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
      ymm1 = _mm256_setzero_ps();

      // c[0,0-7]
      RELU_SCALE_OP_F32S_AVX2(ymm4, ymm0, ymm1, ymm2)

      // c[1,0-7]
      RELU_SCALE_OP_F32S_AVX2(ymm6, ymm0, ymm1, ymm2)

      // c[2,0-7]
      RELU_SCALE_OP_F32S_AVX2(ymm8, ymm0, ymm1, ymm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_3x8F:
    {
      __m256 dn, x_tanh;
      __m256i q;

      // c[0,0-7]
      GELU_TANH_F32S_AVX2(ymm4, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      // c[1,0-7]
      GELU_TANH_F32S_AVX2(ymm6, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      // c[2,0-7]
      GELU_TANH_F32S_AVX2(ymm8, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_3x8F:
    {
      // c[0,0-7]
      GELU_ERF_F32S_AVX2(ymm4, ymm0, ymm1, ymm2)

      // c[1,0-7]
      GELU_ERF_F32S_AVX2(ymm6, ymm0, ymm1, ymm2)

      // c[2,0-7]
      GELU_ERF_F32S_AVX2(ymm8, ymm0, ymm1, ymm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_3x8F:
    {
      ymm0 = _mm256_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      ymm1 = _mm256_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0,0-7]
      CLIP_F32S_AVX2(ymm4, ymm0, ymm1)

      // c[1,0-7]
      CLIP_F32S_AVX2(ymm6, ymm0, ymm1)

      // c[2,0-7]
      CLIP_F32S_AVX2(ymm8, ymm0, ymm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_DOWNSCALE_3x8F:
    {
      __m256 selector1 = _mm256_setzero_ps();
      __m256 selector2 = _mm256_setzero_ps();
      __m256 selector3 = _mm256_setzero_ps();

      __m256 zero_point0 = _mm256_setzero_ps();
      __m256 zero_point1 = _mm256_setzero_ps();
      __m256 zero_point2 = _mm256_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                    ( ( post_ops_list_temp->stor_type == NONE ) &&
                      ( post_ops_attr.c_stor_type == BF16 ) );

      if( post_ops_list_temp->scale_factor_len == 1 )
      {
        selector1 =
              _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector2 =
              _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector3 =
              _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if( is_bf16 == TRUE )
        {
          BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point0);
          BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point1);
          BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point2);
        }
        else
        {
          zero_point0 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point1 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point2 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }

      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
              ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm256_loadu_ps ( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 8 ) );
        }
        if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_LOAD_AVX2(zero_point0,0);
          }
          else
          {
            zero_point0 = _mm256_loadu_ps ( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 0 * 8 ) );
          }
        }
        //c[0, 0-7]
        F32_SCL_MULRND_AVX2(ymm4, selector1, zero_point0);

        //c[1, 0-7]
        F32_SCL_MULRND_AVX2(ymm6, selector1, zero_point0);

        //c[2, 0-7]
        F32_SCL_MULRND_AVX2(ymm8, selector1, zero_point0);
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
              _mm256_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 0 ) );
          selector2 =
              _mm256_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 1 ) );
          selector3 =
              _mm256_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 2 ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_BCAST_AVX2(zero_point0,0);
            BF16_F32_ZP_VECTOR_BCAST_AVX2(zero_point1,1);
            BF16_F32_ZP_VECTOR_BCAST_AVX2(zero_point2,2);
          }
          else
          {
            zero_point0 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 1 ) );
            zero_point2 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 2 ) );
          }
        }
        //c[0, 0-7]
        F32_SCL_MULRND_AVX2(ymm4, selector1, zero_point0);

        //c[1, 0-7]
        F32_SCL_MULRND_AVX2(ymm6, selector2, zero_point1);

        //c[2, 0-7]
        F32_SCL_MULRND_AVX2(ymm8, selector3, zero_point2);
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_3x8F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m256 scl_fctr1 = _mm256_setzero_ps();
      __m256 scl_fctr2 = _mm256_setzero_ps();
      __m256 scl_fctr3 = _mm256_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm256_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
        }
      }

      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,1,6);

          // c[2:0-15]
          BF16_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,2,8);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_ADD_1COL(ymm1,scl_fctr2,1,6);

          // c[2:0-15]
          BF16_F32_MATRIX_ADD_1COL(ymm1,scl_fctr3,2,8);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,1,6);

          // c[2:0-15]
          F32_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,2,8);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_ADD_1COL(ymm1,scl_fctr2,1,6);

          // c[2:0-15]
          F32_F32_MATRIX_ADD_1COL(ymm1,scl_fctr3,2,8);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_3x8F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m256 scl_fctr1 = _mm256_setzero_ps();
      __m256 scl_fctr2 = _mm256_setzero_ps();
      __m256 scl_fctr3 = _mm256_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm256_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
        }
      }

      if( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,1,6);

          // c[2:0-15]
          BF16_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,2,8);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_MUL_1COL(ymm1,scl_fctr2,1,6);

          // c[2:0-15]
          BF16_F32_MATRIX_MUL_1COL(ymm1,scl_fctr3,2,8);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,1,6);

          // c[2:0-15]
          F32_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,2,8);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_MUL_1COL(ymm1,scl_fctr2,1,6);

          // c[2:0-15]
          F32_F32_MATRIX_MUL_1COL(ymm1,scl_fctr3,2,8);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_3x8F:
    {
        ymm0 =
          _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
        __m256 z, dn;
        __m256i ex_out;

        // c[0,0-7]
        SWISH_F32_AVX2_DEF(ymm4, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[1,0-7]
        SWISH_F32_AVX2_DEF(ymm6, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[2,0-7]
        SWISH_F32_AVX2_DEF(ymm8, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_3x8F:
    {
      __m256 dn;
      __m256i q;

      // c[0,0-7]
      TANH_F32S_AVX2(ymm4, ymm0, ymm1, ymm2, ymm3, dn, q)

      // c[1,0-7]
      TANH_F32S_AVX2(ymm6, ymm0, ymm1, ymm2, ymm3, dn, q)

      // c[2,0-7]
      TANH_F32S_AVX2(ymm8, ymm0, ymm1, ymm2, ymm3, dn, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SIGMOID_3x8F:
    {
        __m256 z, dn;
        __m256i ex_out;

        // c[0,0-7]
        SIGMOID_F32_AVX2_DEF(ymm4, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[1,0-7]
        SIGMOID_F32_AVX2_DEF(ymm6, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[2,0-7]
        SIGMOID_F32_AVX2_DEF(ymm8, ymm1, ymm2, ymm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_3x8F_DISABLE:
    ;

    uint32_t tlsb, rounded, temp[8] = {0};
    int i;
    bfloat16* dest;

    if ( ( post_ops_attr.buf_downscale != NULL ) &&
      ( post_ops_attr.is_last_k == TRUE ) )
    {
      STORE_F32_BF16_YMM(ymm4, 0, 0)
      STORE_F32_BF16_YMM(ymm6, 1, 0)
      STORE_F32_BF16_YMM(ymm8, 2, 0)
    }
    else
    {
      _mm256_storeu_ps(cbuf, ymm4);
      cbuf += rs_c;
      _mm256_storeu_ps(cbuf, ymm6);
      cbuf += rs_c;
      _mm256_storeu_ps(cbuf, ymm8);
    }
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_2x8)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_2x8F_DISABLE,
              &&POST_OPS_BIAS_2x8F,
              &&POST_OPS_RELU_2x8F,
              &&POST_OPS_RELU_SCALE_2x8F,
              &&POST_OPS_GELU_TANH_2x8F,
              &&POST_OPS_GELU_ERF_2x8F,
              &&POST_OPS_CLIP_2x8F,
              &&POST_OPS_DOWNSCALE_2x8F,
              &&POST_OPS_MATRIX_ADD_2x8F,
              &&POST_OPS_SWISH_2x8F,
              &&POST_OPS_MATRIX_MUL_2x8F,
              &&POST_OPS_TANH_2x8F,
              &&POST_OPS_SIGMOID_2x8F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    /*Declare the registers*/
    __m256 ymm0, ymm1, ymm2, ymm3;
    __m256 ymm4, ymm6;

    /* zero the accumulator registers */
    ZERO_ACC_YMM_4_REG(ymm4, ymm6, ymm2, ymm3);

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;
    float *_cbuf = NULL;

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


    if ( beta != 0.0 )
    {
      ymm3 = _mm256_broadcast_ss(&(beta));

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
        ( post_ops_attr.is_first_k == TRUE ) )
      {
        BF16_F32_C_BNZ_8(0,0,ymm0,ymm3,ymm4)
        BF16_F32_C_BNZ_8(1,0,ymm0,ymm3,ymm6)
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back
        F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm4)
        _cbuf += rs_c;
        F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm6)
      }
    }//betazero

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_2x8F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          BF16_F32_BIAS_LOAD_AVX2( ymm0, 0 );
        }
        else
        {
          ymm0 = _mm256_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 0 * 8 ) );
        }

        // c[0,0-7]
        ymm4 = _mm256_add_ps( ymm4, ymm0 );

        // c[1,0-7]
        ymm6 = _mm256_add_ps( ymm6, ymm0 );
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
          BF16_F32_BIAS_BCAST_AVX2(ymm0,0);
          BF16_F32_BIAS_BCAST_AVX2(ymm1,1);
        }
        else
        {
          ymm0 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 0 );
          ymm1 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 1 );
        }

        // c[0,0-7]
        ymm4 = _mm256_add_ps( ymm4, ymm0 );

        // c[1,0-7]
        ymm6 = _mm256_add_ps( ymm6, ymm1 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_2x8F:
    {
      ymm0 = _mm256_setzero_ps();

      // c[0,0-7]
      ymm4 = _mm256_max_ps( ymm4, ymm0 );

      // c[1,0-7]
      ymm6 = _mm256_max_ps( ymm6, ymm0 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_2x8F:
    {
      ymm0 =
        _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
      ymm1 = _mm256_setzero_ps();

      // c[0,0-7]
      RELU_SCALE_OP_F32S_AVX2(ymm4, ymm0, ymm1, ymm2)

      // c[1,0-7]
      RELU_SCALE_OP_F32S_AVX2(ymm6, ymm0, ymm1, ymm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_2x8F:
    {
      __m256 dn, x_tanh;
      __m256i q;

      // c[0,0-7]
      GELU_TANH_F32S_AVX2(ymm4, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      // c[1,0-7]
      GELU_TANH_F32S_AVX2(ymm6, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_2x8F:
    {
      // c[0,0-7]
      GELU_ERF_F32S_AVX2(ymm4, ymm0, ymm1, ymm2)

      // c[1,0-7]
      GELU_ERF_F32S_AVX2(ymm6, ymm0, ymm1, ymm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_2x8F:
    {
      ymm0 = _mm256_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      ymm1 = _mm256_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0,0-7]
      CLIP_F32S_AVX2(ymm4, ymm0, ymm1)

      // c[1,0-7]
      CLIP_F32S_AVX2(ymm6, ymm0, ymm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_DOWNSCALE_2x8F:
    {
      __m256 selector1 = _mm256_setzero_ps();
      __m256 selector2 = _mm256_setzero_ps();

      __m256 zero_point0 = _mm256_setzero_ps();
      __m256 zero_point1 = _mm256_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                    ( ( post_ops_list_temp->stor_type == NONE ) &&
                      ( post_ops_attr.c_stor_type == BF16 ) );

      if( post_ops_list_temp->scale_factor_len == 1 )
      {
        selector1 =
              _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector2 =
              _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if( is_bf16 == TRUE )
        {
          BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point0);
          BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point1);
        }
        else
        {
          zero_point0 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point1 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }

      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
              ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm256_loadu_ps ( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 8 ) );
        }
        if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_LOAD_AVX2(zero_point0,0);
          }
          else
          {
            zero_point0 = _mm256_loadu_ps ( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 0 * 8 ) );
          }
        }
        //c[0, 0-7]
        F32_SCL_MULRND_AVX2(ymm4, selector1, zero_point0);

        //c[1, 0-7]
        F32_SCL_MULRND_AVX2(ymm6, selector1, zero_point0);
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
              _mm256_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 0 ) );
          selector2 =
              _mm256_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 1 ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_BCAST_AVX2(zero_point0,0);
            BF16_F32_ZP_VECTOR_BCAST_AVX2(zero_point1,1);
          }
          else
          {
            zero_point0 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 1 ) );
          }
        }
        //c[0, 0-7]
        F32_SCL_MULRND_AVX2(ymm4, selector1, zero_point0);

        //c[1, 0-7]
        F32_SCL_MULRND_AVX2(ymm6, selector2, zero_point1);
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_2x8F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m256 scl_fctr1 = _mm256_setzero_ps();
      __m256 scl_fctr2 = _mm256_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm256_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
        }
      }
      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,1,6);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_ADD_1COL(ymm1,scl_fctr2,1,6);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,1,6);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_ADD_1COL(ymm1,scl_fctr2,1,6);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_2x8F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m256 scl_fctr1 = _mm256_setzero_ps();
      __m256 scl_fctr2 = _mm256_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm256_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
        }
      }
      if( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,1,6);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_MUL_1COL(ymm1,scl_fctr2,1,6);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,1,6);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_MUL_1COL(ymm1,scl_fctr2,1,6);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_2x8F:
    {
        ymm0 =
          _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
        __m256 z, dn;
        __m256i ex_out;

        // c[0,0-7]
        SWISH_F32_AVX2_DEF(ymm4, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[1,0-7]
        SWISH_F32_AVX2_DEF(ymm6, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_2x8F:
    {
      __m256 dn;
      __m256i q;

      // c[0,0-7]
      TANH_F32S_AVX2(ymm4, ymm0, ymm1, ymm2, ymm3, dn, q)

      // c[1,0-7]
      TANH_F32S_AVX2(ymm6, ymm0, ymm1, ymm2, ymm3, dn, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SIGMOID_2x8F:
    {
        __m256 z, dn;
        __m256i ex_out;

        // c[0,0-7]
        SIGMOID_F32_AVX2_DEF(ymm4, ymm1, ymm2, ymm3, z, dn, ex_out)

        // c[1,0-7]
        SIGMOID_F32_AVX2_DEF(ymm6, ymm1, ymm2, ymm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_2x8F_DISABLE:
    ;

    uint32_t tlsb, rounded, temp[8] = {0};
    int i;
    bfloat16* dest;

    if ( ( post_ops_attr.buf_downscale != NULL ) &&
      ( post_ops_attr.is_last_k == TRUE ) )
    {
      STORE_F32_BF16_YMM(ymm4, 0, 0)
      STORE_F32_BF16_YMM(ymm6, 1, 0)
    }
    else
    {
      _mm256_storeu_ps(cbuf, ymm4);
      cbuf += rs_c;
      _mm256_storeu_ps(cbuf, ymm6);
    }
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_1x8)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_1x8F_DISABLE,
              &&POST_OPS_BIAS_1x8F,
              &&POST_OPS_RELU_1x8F,
              &&POST_OPS_RELU_SCALE_1x8F,
              &&POST_OPS_GELU_TANH_1x8F,
              &&POST_OPS_GELU_ERF_1x8F,
              &&POST_OPS_CLIP_1x8F,
              &&POST_OPS_DOWNSCALE_1x8F,
              &&POST_OPS_MATRIX_ADD_1x8F,
              &&POST_OPS_SWISH_1x8F,
              &&POST_OPS_MATRIX_MUL_1x8F,
              &&POST_OPS_TANH_1x8F,
              &&POST_OPS_SIGMOID_1x8F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    /*Declare the registers*/
    __m256 ymm0, ymm1, ymm2, ymm3;
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


    if ( beta != 0.0 )
    {
      ymm3 = _mm256_broadcast_ss(&(beta));

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
        ( post_ops_attr.is_first_k == TRUE ) )
      {
        BF16_F32_C_BNZ_8(0,0,ymm0,ymm3,ymm4)
      }
      else
      {
        //load c and multiply with beta and
        //add to accumulator and store back
        F32_C_BNZ_8(cbuf,rs_c,ymm0,ymm3,ymm4)
      }
    }//betazero

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_1x8F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          BF16_F32_BIAS_LOAD_AVX2( ymm0, 0 );
        }
        else
        {
          ymm0 = _mm256_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 0 * 8 ) );
        }

        // c[0,0-7]
        ymm4 = _mm256_add_ps( ymm4, ymm0 );
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
          BF16_F32_BIAS_BCAST_AVX2(ymm0,0);
        }
        else
        {
          ymm0 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 0 );
        }

        // c[0,0-7]
        ymm4 = _mm256_add_ps( ymm4, ymm0 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_1x8F:
    {
      ymm0 = _mm256_setzero_ps();

      // c[0,0-7]
      ymm4 = _mm256_max_ps( ymm4, ymm0 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_1x8F:
    {
      ymm0 =
        _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
      ymm1 = _mm256_setzero_ps();

      // c[0,0-7]
      RELU_SCALE_OP_F32S_AVX2(ymm4, ymm0, ymm1, ymm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_1x8F:
    {
      __m256 dn, x_tanh;
      __m256i q;

      // c[0,0-7]
      GELU_TANH_F32S_AVX2(ymm4, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_1x8F:
    {
      // c[0,0-7]
      GELU_ERF_F32S_AVX2(ymm4, ymm0, ymm1, ymm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_1x8F:
    {
      ymm0 = _mm256_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      ymm1 = _mm256_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0,0-7]
      CLIP_F32S_AVX2(ymm4, ymm0, ymm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_DOWNSCALE_1x8F:
    {
      __m256 selector1 = _mm256_setzero_ps();

      __m256 zero_point0 = _mm256_setzero_ps();

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                    ( ( post_ops_list_temp->stor_type == NONE ) &&
                      ( post_ops_attr.c_stor_type == BF16 ) );

      if( post_ops_list_temp->scale_factor_len == 1 )
      {
        selector1 =
              _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if( is_bf16 == TRUE )
        {
          BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point0);
        }
        else
        {
          zero_point0 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }

      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
              ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector1 = _mm256_loadu_ps ( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 8 ) );
        }
        if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_LOAD_AVX2(zero_point0,0);
          }
          else
          {
            zero_point0 = _mm256_loadu_ps ( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 0 * 8 ) );
          }
        }
        //c[0, 0-7]
        F32_SCL_MULRND_AVX2(ymm4, selector1, zero_point0);
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
              _mm256_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                              post_ops_attr.post_op_c_i + 0 ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_BCAST_AVX2(zero_point0,0);
          }
          else
          {
            zero_point0 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                  post_ops_attr.post_op_c_i + 0 ) );
          }
        }
        //c[0, 0-7]
        F32_SCL_MULRND_AVX2(ymm4, selector1, zero_point0);
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_1x8F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m256 scl_fctr1 = _mm256_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm256_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
        }
      }
      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,0,4);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,0,4);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,0,4);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,0,4);
        }
        }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_1x8F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m256 scl_fctr1 = _mm256_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm256_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
        }
      }

      if( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,0,4);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,0,4);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,0,4);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,0,4);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_1x8F:
    {
        ymm0 =
          _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
        __m256 z, dn;
        __m256i ex_out;

        // c[0,0-7]
        SWISH_F32_AVX2_DEF(ymm4, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_1x8F:
    {
      __m256 dn;
      __m256i q;

      // c[0,0-7]
      TANH_F32S_AVX2(ymm4, ymm0, ymm1, ymm2, ymm3, dn, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SIGMOID_1x8F:
    {
        __m256 z, dn;
        __m256i ex_out;

        // c[0,0-7]
        SIGMOID_F32_AVX2_DEF(ymm4, ymm1, ymm2, ymm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_1x8F_DISABLE:
    ;

    uint32_t tlsb, rounded, temp[8] = {0};
    int i;
    bfloat16* dest;

    if ( ( post_ops_attr.buf_downscale != NULL ) &&
      ( post_ops_attr.is_last_k == TRUE ) )
    {
      STORE_F32_BF16_YMM(ymm4, 0, 0)
    }
    else
    {
      _mm256_storeu_ps(cbuf, ymm4);
    }
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_5x4)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_5x4F_DISABLE,
              &&POST_OPS_BIAS_5x4F,
              &&POST_OPS_RELU_5x4F,
              &&POST_OPS_RELU_SCALE_5x4F,
              &&POST_OPS_GELU_TANH_5x4F,
              &&POST_OPS_GELU_ERF_5x4F,
              &&POST_OPS_CLIP_5x4F,
              &&POST_OPS_DOWNSCALE_5x4F,
              &&POST_OPS_MATRIX_ADD_5x4F,
              &&POST_OPS_SWISH_5x4F,
              &&POST_OPS_MATRIX_MUL_5x4F,
              &&POST_OPS_TANH_5x4F,
              &&POST_OPS_SIGMOID_5x4F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;
    float *_cbuf = NULL;

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


    if ( beta != 0.0 )
    {
      xmm3 = _mm_broadcast_ss(&(beta));

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
        ( post_ops_attr.is_first_k == TRUE ) )
      {
        BF16_F32_C_BNZ_4(0,0,xmm1,xmm3,xmm4)
        BF16_F32_C_BNZ_4(1,0,xmm1,xmm3,xmm5)
        BF16_F32_C_BNZ_4(2,0,xmm1,xmm3,xmm6)
        BF16_F32_C_BNZ_4(3,0,xmm1,xmm3,xmm7)
        BF16_F32_C_BNZ_4(4,0,xmm1,xmm3,xmm8)
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back
        F32_C_BNZ_4(_cbuf,rs_c,xmm1,xmm3,xmm4)
        _cbuf += rs_c;
        F32_C_BNZ_4(_cbuf,rs_c,xmm1,xmm3,xmm5)
        _cbuf += rs_c;
        F32_C_BNZ_4(_cbuf,rs_c,xmm1,xmm3,xmm6)
        _cbuf += rs_c;
        F32_C_BNZ_4(_cbuf,rs_c,xmm1,xmm3,xmm7)
        _cbuf += rs_c;
        F32_C_BNZ_4(_cbuf,rs_c,xmm1,xmm3,xmm8)
      }
    }//betazero

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_5x4F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          BF16_F32_BIAS_LOAD_4BF16_AVX2(xmm0, 0);
        }
        else
        {
          xmm0 = _mm_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 0 * 8 ) );
        }

        // c[0,0-3]
        xmm4 = _mm_add_ps( xmm4, xmm0 );

        // c[1,0-3]
        xmm5 = _mm_add_ps( xmm5, xmm0 );

        // c[2,0-3]
        xmm6 = _mm_add_ps( xmm6, xmm0 );

        // c[3,0-3]
        xmm7 = _mm_add_ps( xmm7, xmm0 );

        // c[4,0-3]
        xmm8 = _mm_add_ps( xmm8, xmm0 );
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
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm0, 0);
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm1, 1);
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm2, 2);
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm3, 3);
        }
        else
        {
          xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 0 );
          xmm1 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 1 );
          xmm2 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 2 );
          xmm3 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 3 );
        }

        // c[0,0-3]
        xmm4 = _mm_add_ps( xmm4, xmm0 );

        // c[1,0-3]
        xmm5 = _mm_add_ps( xmm5, xmm1 );

        // c[2,0-3]
        xmm6 = _mm_add_ps( xmm6, xmm2 );

        // c[3,0-3]
        xmm7 = _mm_add_ps( xmm7, xmm3 );

        if( post_ops_list_temp->stor_type == BF16 )
        {
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm0, 4);
        }
        else
        {
          xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 4 );
        }

        // c[4,0-3]
        xmm8 = _mm_add_ps( xmm8, xmm0 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_5x4F:
    {
      xmm0 = _mm_setzero_ps();

      // c[0,0-3]
      xmm4 = _mm_max_ps( xmm4, xmm0 );

      // c[1,0-3]
      xmm5 = _mm_max_ps( xmm5, xmm0 );

      // c[2,0-3]
      xmm6 = _mm_max_ps( xmm6, xmm0 );

      // c[3,0-3]
      xmm7 = _mm_max_ps( xmm7, xmm0 );

      // c[4,0-3]
      xmm8 = _mm_max_ps( xmm8, xmm0 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_5x4F:
    {
      xmm0 =
        _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
      xmm1 = _mm_setzero_ps();

      // c[0,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

      // c[1,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm5, xmm0, xmm1, xmm2)

      // c[2,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm6, xmm0, xmm1, xmm2)

      // c[3,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm7, xmm0, xmm1, xmm2)

      // c[4,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm8, xmm0, xmm1, xmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_5x4F:
    {
      __m128 dn, x_tanh;
      __m128i q;

      // c[0,0-3]
      GELU_TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      // c[1,0-3]
      GELU_TANH_F32S_SSE(xmm5, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      // c[2,0-3]
      GELU_TANH_F32S_SSE(xmm6, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      // c[3,0-3]
      GELU_TANH_F32S_SSE(xmm7, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      // c[4,0-3]
      GELU_TANH_F32S_SSE(xmm8, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_5x4F:
    {
      // c[0,0-3]
      GELU_ERF_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

      // c[1,0-3]
      GELU_ERF_F32S_SSE(xmm5, xmm0, xmm1, xmm2)

      // c[2,0-3]
      GELU_ERF_F32S_SSE(xmm6, xmm0, xmm1, xmm2)

      // c[3,0-3]
      GELU_ERF_F32S_SSE(xmm7, xmm0, xmm1, xmm2)

      // c[4,0-3]
      GELU_ERF_F32S_SSE(xmm8, xmm0, xmm1, xmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_5x4F:
    {
      xmm0 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      xmm1 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0,0-3]
      CLIP_F32S_SSE(xmm4, xmm0, xmm1)

      // c[1,0-3]
      CLIP_F32S_SSE(xmm5, xmm0, xmm1)

      // c[2,0-3]
      CLIP_F32S_SSE(xmm6, xmm0, xmm1)

      // c[3,0-3]
      CLIP_F32S_SSE(xmm7, xmm0, xmm1)

      // c[4,0-3]
      CLIP_F32S_SSE(xmm8, xmm0, xmm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_DOWNSCALE_5x4F:
    {
      __m128 selector0 = _mm_setzero_ps();
      __m128 selector1 = _mm_setzero_ps();
      __m128 selector2 = _mm_setzero_ps();
      __m128 selector3 = _mm_setzero_ps();
      __m128 selector4 = _mm_setzero_ps();

      __m128 zero_point0 = _mm_setzero_ps();
      __m128 zero_point1 = _mm_setzero_ps();
      __m128 zero_point2 = _mm_setzero_ps();
      __m128 zero_point3 = _mm_setzero_ps();
      __m128 zero_point4 = _mm_setzero_ps();

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
        selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector4 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if( is_bf16 == TRUE )
        {
          BF16_F32_ZP_SCALAR_BCAST_SSE(zero_point0);
          BF16_F32_ZP_SCALAR_BCAST_SSE(zero_point1);
          BF16_F32_ZP_SCALAR_BCAST_SSE(zero_point2);
          BF16_F32_ZP_SCALAR_BCAST_SSE(zero_point3);
          BF16_F32_ZP_SCALAR_BCAST_SSE(zero_point4);
        }
        else
        {
          zero_point0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point4 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector0 = _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_j + ( 0 * 4) );
        }
        if( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_4LOAD_SSE(zero_point0,0)
          }
          else
          {
            zero_point0 = _mm_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_j + ( 0 * 4 ) );
          }
        }
        //c[0, 0-3]
        F32_SCL_MULRND_SSE(xmm4, selector0, zero_point0);

        //c[1, 0-3]
        F32_SCL_MULRND_SSE(xmm5, selector0, zero_point0);

        //c[2, 0-3]
        F32_SCL_MULRND_SSE(xmm6, selector0, zero_point0);

        //c[3, 0-3]
        F32_SCL_MULRND_SSE(xmm7, selector0, zero_point0);

        //c[4, 0-3]
        F32_SCL_MULRND_SSE(xmm8, selector0, zero_point0);
      }
      else
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_i + 0 ) );
          selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_i + 1 ) );
          selector2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_i + 2 ) );
          selector3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_i + 3 ) );
          selector4 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_i + 4 ) );
        }
        if( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point0,0)
            BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point1,1)
            BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point2,2)
            BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point3,3)
            BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point4,4)
          }
          else
          {
            zero_point0 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 1 ) );
            zero_point2 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 2 ) );
            zero_point3 =_mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 3 ) );
            zero_point4 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 4 ) );
          }
        }
        //c[0, 0-3]
        F32_SCL_MULRND_SSE(xmm4, selector0, zero_point0);

        //c[1, 0-3]
        F32_SCL_MULRND_SSE(xmm5, selector1, zero_point1);

        //c[2, 0-3]
        F32_SCL_MULRND_SSE(xmm6, selector2, zero_point2);

        //c[3, 0-3]
        F32_SCL_MULRND_SSE(xmm7, selector3, zero_point3);

        //c[4, 0-3]
        F32_SCL_MULRND_SSE(xmm8, selector4, zero_point4);
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_5x4F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m128 scl_fctr1 = _mm_setzero_ps();
      __m128 scl_fctr2 = _mm_setzero_ps();
      __m128 scl_fctr3 = _mm_setzero_ps();
      __m128 scl_fctr4 = _mm_setzero_ps();
      __m128 scl_fctr5 = _mm_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr4 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr5 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
          scl_fctr4 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 3 ) );
          scl_fctr5 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 4 ) );
        }
      }
      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,1,5);

          // c[2:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,2,6);

          // c[3:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,3,7);

          // c[4:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,4,8);
        }
        else
        {
          BF16_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr2,1,5);

          // c[2:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr3,2,6);

          // c[3:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr4,3,7);

          // c[4:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr5,4,8);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,1,5);

          // c[2:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,2,6);

          // c[3:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,3,7);

          // c[4:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,4,8);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr2,1,5);

          // c[2:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr3,2,6);

          // c[3:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr4,3,7);

          // c[4:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr5,4,8);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_5x4F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );


      __m128 scl_fctr1 = _mm_setzero_ps();
      __m128 scl_fctr2 = _mm_setzero_ps();
      __m128 scl_fctr3 = _mm_setzero_ps();
      __m128 scl_fctr4 = _mm_setzero_ps();
      __m128 scl_fctr5 = _mm_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr4 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr5 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
          scl_fctr4 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 3 ) );
          scl_fctr5 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 4 ) );
        }
      }

      if( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,1,5);

          // c[2:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,2,6);

          // c[3:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,3,7);

          // c[4:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,4,8);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr2,1,5);

          // c[2:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr3,2,6);

          // c[3:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr4,3,7);

          // c[4:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr5,4,8);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,1,5);

          // c[2:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,2,6);

          // c[3:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,3,7);

          // c[4:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,4,8);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr2,1,5);

          // c[2:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr3,2,6);

          // c[3:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr4,3,7);

          // c[4:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr5,4,8);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_5x4F:
    {
        xmm0 =
          _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
        __m128 z, dn;
        __m128i ex_out;

        // c[0,0-3]
        SWISH_F32_SSE_DEF(xmm4, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[1,0-3]
        SWISH_F32_SSE_DEF(xmm5, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[2,0-3]
        SWISH_F32_SSE_DEF(xmm6, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[3,0-3]
        SWISH_F32_SSE_DEF(xmm7, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[4,0-3]
        SWISH_F32_SSE_DEF(xmm8, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_5x4F:
    {
      __m128 dn;
      __m128i q;

      // c[0,0-3]
      TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, q)

      // c[1,0-3]
      TANH_F32S_SSE(xmm5, xmm0, xmm1, xmm2, xmm3, dn, q)

      // c[2,0-3]
      TANH_F32S_SSE(xmm6, xmm0, xmm1, xmm2, xmm3, dn, q)

      // c[3,0-3]
      TANH_F32S_SSE(xmm7, xmm0, xmm1, xmm2, xmm3, dn, q)

      // c[4,0-3]
      TANH_F32S_SSE(xmm8, xmm0, xmm1, xmm2, xmm3, dn, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SIGMOID_5x4F:
    {
        __m128 z, dn;
        __m128i ex_out;

        // c[0,0-3]
        SIGMOID_F32_SSE_DEF(xmm4, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[1,0-3]
        SIGMOID_F32_SSE_DEF(xmm5, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[2,0-3]
        SIGMOID_F32_SSE_DEF(xmm6, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[3,0-3]
        SIGMOID_F32_SSE_DEF(xmm7, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[4,0-3]
        SIGMOID_F32_SSE_DEF(xmm8, xmm1, xmm2, xmm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_5x4F_DISABLE:
    ;

    uint32_t tlsb, rounded, temp[4] = {0};
    int i;
    bfloat16* dest;

    if ( ( post_ops_attr.buf_downscale != NULL ) &&
      ( post_ops_attr.is_last_k == TRUE ) )
    {
      STORE_F32_BF16_4XMM(xmm4, 0, 0)
      STORE_F32_BF16_4XMM(xmm5, 1, 0)
      STORE_F32_BF16_4XMM(xmm6, 2, 0)
      STORE_F32_BF16_4XMM(xmm7, 3, 0)
      STORE_F32_BF16_4XMM(xmm8, 4, 0)
    }
    else
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
    }
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_4x4)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_4x4F_DISABLE,
              &&POST_OPS_BIAS_4x4F,
              &&POST_OPS_RELU_4x4F,
              &&POST_OPS_RELU_SCALE_4x4F,
              &&POST_OPS_GELU_TANH_4x4F,
              &&POST_OPS_GELU_ERF_4x4F,
              &&POST_OPS_CLIP_4x4F,
              &&POST_OPS_DOWNSCALE_4x4F,
              &&POST_OPS_MATRIX_ADD_4x4F,
              &&POST_OPS_SWISH_4x4F,
              &&POST_OPS_MATRIX_MUL_4x4F,
              &&POST_OPS_TANH_4x4F,
              &&POST_OPS_SIGMOID_4x4F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;
    float *_cbuf = NULL;

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


    if ( beta != 0.0 )
    {
      xmm3 = _mm_broadcast_ss(&(beta));

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
        ( post_ops_attr.is_first_k == TRUE ) )
      {
        BF16_F32_C_BNZ_4(0,0,xmm1,xmm3,xmm4)
        BF16_F32_C_BNZ_4(1,0,xmm1,xmm3,xmm5)
        BF16_F32_C_BNZ_4(2,0,xmm1,xmm3,xmm6)
        BF16_F32_C_BNZ_4(3,0,xmm1,xmm3,xmm7)
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back
        F32_C_BNZ_4(_cbuf,rs_c,xmm1,xmm3,xmm4)
        _cbuf += rs_c;
        F32_C_BNZ_4(_cbuf,rs_c,xmm1,xmm3,xmm5)
        _cbuf += rs_c;
        F32_C_BNZ_4(_cbuf,rs_c,xmm1,xmm3,xmm6)
        _cbuf += rs_c;
        F32_C_BNZ_4(_cbuf,rs_c,xmm1,xmm3,xmm7)
      }
    }//betazero

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_4x4F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          BF16_F32_BIAS_LOAD_4BF16_AVX2(xmm0, 0);
        }
        else
        {
          xmm0 = _mm_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 0 * 8 ) );
        }

        // c[0,0-3]
        xmm4 = _mm_add_ps( xmm4, xmm0 );

        // c[1,0-3]
        xmm5 = _mm_add_ps( xmm5, xmm0 );

        // c[2,0-3]
        xmm6 = _mm_add_ps( xmm6, xmm0 );

        // c[3,0-3]
        xmm7 = _mm_add_ps( xmm7, xmm0 );
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
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm0, 0);
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm1, 1);
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm2, 2);
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm3, 3);
        }
        else
        {
          xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 0 );
          xmm1 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 1 );
          xmm2 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 2 );
          xmm3 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 3 );
        }
        // c[0,0-3]
        xmm4 = _mm_add_ps( xmm4, xmm0 );

        // c[1,0-3]
        xmm5 = _mm_add_ps( xmm5, xmm1 );

        // c[2,0-3]
        xmm6 = _mm_add_ps( xmm6, xmm2 );

        // c[3,0-3]
        xmm7 = _mm_add_ps( xmm7, xmm3 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_4x4F:
    {
      xmm0 = _mm_setzero_ps();

      // c[0,0-3]
      xmm4 = _mm_max_ps( xmm4, xmm0 );

      // c[1,0-3]
      xmm5 = _mm_max_ps( xmm5, xmm0 );

      // c[2,0-3]
      xmm6 = _mm_max_ps( xmm6, xmm0 );

      // c[3,0-3]
      xmm7 = _mm_max_ps( xmm7, xmm0 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_4x4F:
    {
      xmm0 =
        _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
      xmm1 = _mm_setzero_ps();

      // c[0,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

      // c[1,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm5, xmm0, xmm1, xmm2)

      // c[2,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm6, xmm0, xmm1, xmm2)

      // c[3,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm7, xmm0, xmm1, xmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_4x4F:
    {
      __m128 dn, x_tanh;
      __m128i q;

      // c[0,0-3]
      GELU_TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      // c[1,0-3]
      GELU_TANH_F32S_SSE(xmm5, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      // c[2,0-3]
      GELU_TANH_F32S_SSE(xmm6, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      // c[3,0-3]
      GELU_TANH_F32S_SSE(xmm7, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_4x4F:
    {
      // c[0,0-3]
      GELU_ERF_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

      // c[1,0-3]
      GELU_ERF_F32S_SSE(xmm5, xmm0, xmm1, xmm2)

      // c[2,0-3]
      GELU_ERF_F32S_SSE(xmm6, xmm0, xmm1, xmm2)

      // c[3,0-3]
      GELU_ERF_F32S_SSE(xmm7, xmm0, xmm1, xmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_4x4F:
    {
      xmm0 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      xmm1 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0,0-3]
      CLIP_F32S_SSE(xmm4, xmm0, xmm1)

      // c[1,0-3]
      CLIP_F32S_SSE(xmm5, xmm0, xmm1)

      // c[2,0-3]
      CLIP_F32S_SSE(xmm6, xmm0, xmm1)

      // c[3,0-3]
      CLIP_F32S_SSE(xmm7, xmm0, xmm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_DOWNSCALE_4x4F:
      {
      __m128 selector0 = _mm_setzero_ps();
      __m128 selector1 = _mm_setzero_ps();
      __m128 selector2 = _mm_setzero_ps();
      __m128 selector3 = _mm_setzero_ps();

      __m128 zero_point0 = _mm_setzero_ps();
      __m128 zero_point1 = _mm_setzero_ps();
      __m128 zero_point2 = _mm_setzero_ps();
      __m128 zero_point3 = _mm_setzero_ps();

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
        selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if( is_bf16 == TRUE )
        {
          BF16_F32_ZP_SCALAR_BCAST_SSE(zero_point0);
          BF16_F32_ZP_SCALAR_BCAST_SSE(zero_point1);
          BF16_F32_ZP_SCALAR_BCAST_SSE(zero_point2);
          BF16_F32_ZP_SCALAR_BCAST_SSE(zero_point3);
        }
        else
        {
          zero_point0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector0 = _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_j + ( 0 * 4) );
        }
        if( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_4LOAD_SSE(zero_point0,0)
          }
          else
          {
            zero_point0 = _mm_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 0 * 4 ) );
          }
        }
        //c[0, 0-3]
        F32_SCL_MULRND_SSE(xmm4, selector0, zero_point0);

        //c[1, 0-3]
        F32_SCL_MULRND_SSE(xmm5, selector0, zero_point0);

        //c[2, 0-3]
        F32_SCL_MULRND_SSE(xmm6, selector0, zero_point0);

        //c[3, 0-3]
        F32_SCL_MULRND_SSE(xmm7, selector0, zero_point0);
      }
      else
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_i + 0 ) );
          selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_i + 1 ) );
          selector2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_i + 2 ) );
          selector3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_i + 3 ) );
        }
        if( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point0,0)
            BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point1,1)
            BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point2,2)
            BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point3,3)
          }
          else
          {
            zero_point0 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 1 ) );
            zero_point2 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 2 ) );
            zero_point3 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 3 ) );
          }
        }
        //c[0, 0-3]
        F32_SCL_MULRND_SSE(xmm4, selector0, zero_point0);

        //c[1, 0-3]
        F32_SCL_MULRND_SSE(xmm5, selector1, zero_point1);

        //c[2, 0-3]
        F32_SCL_MULRND_SSE(xmm6, selector2, zero_point2);

        //c[3, 0-3]
        F32_SCL_MULRND_SSE(xmm7, selector3, zero_point3);
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_4x4F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m128 scl_fctr1 = _mm_setzero_ps();
      __m128 scl_fctr2 = _mm_setzero_ps();
      __m128 scl_fctr3 = _mm_setzero_ps();
      __m128 scl_fctr4 = _mm_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr4 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
          scl_fctr4 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 3 ) );
        }
      }

      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,1,5);

          // c[2:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,2,6);

          // c[3:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,3,7);
        }
        else
        {
          BF16_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr2,1,5);

          // c[2:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr3,2,6);

          // c[3:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr4,3,7);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,1,5);

          // c[2:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,2,6);

          // c[3:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,3,7);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr2,1,5);

          // c[2:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr3,2,6);

          // c[3:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr4,3,7);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_4x4F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );


      __m128 scl_fctr1 = _mm_setzero_ps();
      __m128 scl_fctr2 = _mm_setzero_ps();
      __m128 scl_fctr3 = _mm_setzero_ps();
      __m128 scl_fctr4 = _mm_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr4 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
          scl_fctr4 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 3 ) );
        }
      }

      if( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,1,5);

          // c[2:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,2,6);

          // c[3:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,3,7);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr2,1,5);

          // c[2:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr3,2,6);

          // c[3:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr4,3,7);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,1,5);

          // c[2:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,2,6);

          // c[3:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,3,7);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr2,1,5);

          // c[2:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr3,2,6);

          // c[3:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr4,3,7);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_4x4F:
    {
        xmm0 =
          _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
        __m128 z, dn;
        __m128i ex_out;

        // c[0,0-3]
        SWISH_F32_SSE_DEF(xmm4, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[1,0-3]
        SWISH_F32_SSE_DEF(xmm5, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[2,0-3]
        SWISH_F32_SSE_DEF(xmm6, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[3,0-3]
        SWISH_F32_SSE_DEF(xmm7, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_4x4F:
    {
      __m128 dn;
      __m128i q;

      // c[0,0-3]
      TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, q)

      // c[1,0-3]
      TANH_F32S_SSE(xmm5, xmm0, xmm1, xmm2, xmm3, dn, q)

      // c[2,0-3]
      TANH_F32S_SSE(xmm6, xmm0, xmm1, xmm2, xmm3, dn, q)

      // c[3,0-3]
      TANH_F32S_SSE(xmm7, xmm0, xmm1, xmm2, xmm3, dn, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SIGMOID_4x4F:
    {
        __m128 z, dn;
        __m128i ex_out;

        // c[0,0-3]
        SIGMOID_F32_SSE_DEF(xmm4, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[1,0-3]
        SIGMOID_F32_SSE_DEF(xmm5, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[2,0-3]
        SIGMOID_F32_SSE_DEF(xmm6, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[3,0-3]
        SIGMOID_F32_SSE_DEF(xmm7, xmm1, xmm2, xmm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_4x4F_DISABLE:
    ;

    uint32_t tlsb, rounded, temp[4] = {0};
    int i;
    bfloat16* dest;

    if ( ( post_ops_attr.buf_downscale != NULL ) &&
      ( post_ops_attr.is_last_k == TRUE ) )
    {
      STORE_F32_BF16_4XMM(xmm4, 0, 0)
      STORE_F32_BF16_4XMM(xmm5, 1, 0)
      STORE_F32_BF16_4XMM(xmm6, 2, 0)
      STORE_F32_BF16_4XMM(xmm7, 3, 0)
    }
    else
    {
      _mm_storeu_ps(cbuf, xmm4);
      cbuf += rs_c;
      _mm_storeu_ps(cbuf, xmm5);
      cbuf += rs_c;
      _mm_storeu_ps(cbuf, xmm6);
      cbuf += rs_c;
      _mm_storeu_ps(cbuf, xmm7);
    }
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_3x4)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_3x4F_DISABLE,
              &&POST_OPS_BIAS_3x4F,
              &&POST_OPS_RELU_3x4F,
              &&POST_OPS_RELU_SCALE_3x4F,
              &&POST_OPS_GELU_TANH_3x4F,
              &&POST_OPS_GELU_ERF_3x4F,
              &&POST_OPS_CLIP_3x4F,
              &&POST_OPS_DOWNSCALE_3x4F,
              &&POST_OPS_MATRIX_ADD_3x4F,
              &&POST_OPS_SWISH_3x4F,
              &&POST_OPS_MATRIX_MUL_3x4F,
              &&POST_OPS_TANH_3x4F,
              &&POST_OPS_SIGMOID_3x4F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;
    float *_cbuf = NULL;

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


    if ( beta != 0.0 )
    {
      xmm3 = _mm_broadcast_ss(&(beta));

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
        ( post_ops_attr.is_first_k == TRUE ) )
      {
        BF16_F32_C_BNZ_4(0,0,xmm1,xmm3,xmm4)
        BF16_F32_C_BNZ_4(1,0,xmm1,xmm3,xmm5)
        BF16_F32_C_BNZ_4(2,0,xmm1,xmm3,xmm6)
        BF16_F32_C_BNZ_4(3,0,xmm1,xmm3,xmm7)
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back
        F32_C_BNZ_4(_cbuf,rs_c,xmm1,xmm3,xmm4)
        _cbuf += rs_c;
        F32_C_BNZ_4(_cbuf,rs_c,xmm1,xmm3,xmm5)
        _cbuf += rs_c;
        F32_C_BNZ_4(_cbuf,rs_c,xmm1,xmm3,xmm6)
      }
    }//betazero

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_3x4F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          BF16_F32_BIAS_LOAD_4BF16_AVX2(xmm0, 0);
        }
        else
        {
          xmm0 = _mm_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 0 * 8 ) );
        }

        // c[0,0-3]
        xmm4 = _mm_add_ps( xmm4, xmm0 );

        // c[1,0-3]
        xmm5 = _mm_add_ps( xmm5, xmm0 );

        // c[2,0-3]
        xmm6 = _mm_add_ps( xmm6, xmm0 );
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
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm0, 0);
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm1, 1);
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm2, 2);
        }
        else
        {
          xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 0 );
          xmm1 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 1 );
          xmm2 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 2 );
        }

        // c[0,0-3]
        xmm4 = _mm_add_ps( xmm4, xmm0 );

        // c[1,0-3]
        xmm5 = _mm_add_ps( xmm5, xmm1 );

        // c[2,0-3]
        xmm6 = _mm_add_ps( xmm6, xmm2 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_3x4F:
    {
      xmm0 = _mm_setzero_ps();

      // c[0,0-3]
      xmm4 = _mm_max_ps( xmm4, xmm0 );

      // c[1,0-3]
      xmm5 = _mm_max_ps( xmm5, xmm0 );

      // c[2,0-3]
      xmm6 = _mm_max_ps( xmm6, xmm0 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_3x4F:
    {
      xmm0 =
        _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
      xmm1 = _mm_setzero_ps();

      // c[0,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

      // c[1,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm5, xmm0, xmm1, xmm2)

      // c[2,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm6, xmm0, xmm1, xmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_3x4F:
    {
      __m128 dn, x_tanh;
      __m128i q;

      // c[0,0-3]
      GELU_TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      // c[1,0-3]
      GELU_TANH_F32S_SSE(xmm5, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      // c[2,0-3]
      GELU_TANH_F32S_SSE(xmm6, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_3x4F:
    {
      // c[0,0-3]
      GELU_ERF_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

      // c[1,0-3]
      GELU_ERF_F32S_SSE(xmm5, xmm0, xmm1, xmm2)

      // c[2,0-3]
      GELU_ERF_F32S_SSE(xmm6, xmm0, xmm1, xmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_3x4F:
    {
      xmm0 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      xmm1 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0,0-3]
      CLIP_F32S_SSE(xmm4, xmm0, xmm1)

      // c[1,0-3]
      CLIP_F32S_SSE(xmm5, xmm0, xmm1)

      // c[2,0-3]
      CLIP_F32S_SSE(xmm6, xmm0, xmm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_DOWNSCALE_3x4F:
    {
      __m128 selector0 = _mm_setzero_ps();
      __m128 selector1 = _mm_setzero_ps();
      __m128 selector2 = _mm_setzero_ps();

      __m128 zero_point0 = _mm_setzero_ps();
      __m128 zero_point1 = _mm_setzero_ps();
      __m128 zero_point2 = _mm_setzero_ps();

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
        selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if( is_bf16 == TRUE )
        {
          BF16_F32_ZP_SCALAR_BCAST_SSE(zero_point0);
          BF16_F32_ZP_SCALAR_BCAST_SSE(zero_point1);
          BF16_F32_ZP_SCALAR_BCAST_SSE(zero_point2);
        }
        else
        {
          zero_point0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector0 = _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_j + ( 0 * 4) );
        }
        if( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_4LOAD_SSE(zero_point0,0)
          }
          else
          {
            zero_point0 = _mm_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_j + ( 0 * 4 ) );
          }
        }
        //c[0, 0-3]
        F32_SCL_MULRND_SSE(xmm4, selector0, zero_point0);

        //c[1, 0-3]
        F32_SCL_MULRND_SSE(xmm5, selector0, zero_point0);

        //c[2, 0-3]
        F32_SCL_MULRND_SSE(xmm6, selector0, zero_point0);
      }
      else
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_i + 0 ) );
          selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_i + 1 ) );
          selector2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_i + 2 ) );
        }
        if( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point0,0)
            BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point1,1)
            BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point2,2)
          }
          else
          {
            zero_point0 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 1 ) );
            zero_point2 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 2 ) );
          }
        }
        //c[0, 0-3]
        F32_SCL_MULRND_SSE(xmm4, selector0, zero_point0);

        //c[1, 0-3]
        F32_SCL_MULRND_SSE(xmm5, selector1, zero_point1);

        //c[2, 0-3]
        F32_SCL_MULRND_SSE(xmm6, selector2, zero_point2);
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_3x4F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m128 scl_fctr1 = _mm_setzero_ps();
      __m128 scl_fctr2 = _mm_setzero_ps();
      __m128 scl_fctr3 = _mm_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
        }
      }

      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,1,5);

          // c[2:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,2,6);
        }
        else
        {
          BF16_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr2,1,5);

          // c[2:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr3,2,6);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,1,5);

          // c[2:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,2,6);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr2,1,5);

          // c[2:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr3,2,6);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_3x4F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m128 scl_fctr1 = _mm_setzero_ps();
      __m128 scl_fctr2 = _mm_setzero_ps();
      __m128 scl_fctr3 = _mm_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
        }
      }

      if( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,1,5);

          // c[2:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,2,6);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr2,1,5);

          // c[2:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr3,2,6);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,1,5);

          // c[2:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,2,6);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr2,1,5);

          // c[2:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr3,2,6);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_3x4F:
    {
        xmm0 =
          _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
        __m128 z, dn;
        __m128i ex_out;

        // c[0,0-3]
        SWISH_F32_SSE_DEF(xmm4, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[1,0-3]
        SWISH_F32_SSE_DEF(xmm5, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[2,0-3]
        SWISH_F32_SSE_DEF(xmm6, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_3x4F:
    {
      __m128 dn;
      __m128i q;

      // c[0,0-3]
      TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, q)

      // c[1,0-3]
      TANH_F32S_SSE(xmm5, xmm0, xmm1, xmm2, xmm3, dn, q)

      // c[2,0-3]
      TANH_F32S_SSE(xmm6, xmm0, xmm1, xmm2, xmm3, dn, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SIGMOID_3x4F:
    {
        __m128 z, dn;
        __m128i ex_out;

        // c[0,0-3]
        SIGMOID_F32_SSE_DEF(xmm4, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[1,0-3]
        SIGMOID_F32_SSE_DEF(xmm5, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[2,0-3]
        SIGMOID_F32_SSE_DEF(xmm6, xmm1, xmm2, xmm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }

POST_OPS_3x4F_DISABLE:
    ;

    uint32_t tlsb, rounded, temp[4] = {0};
    int i;
    bfloat16* dest;

    if ( ( post_ops_attr.buf_downscale != NULL ) &&
      ( post_ops_attr.is_last_k == TRUE ) )
    {
      STORE_F32_BF16_4XMM(xmm4, 0, 0)
      STORE_F32_BF16_4XMM(xmm5, 1, 0)
      STORE_F32_BF16_4XMM(xmm6, 2, 0)
    }
    else
    {
      _mm_storeu_ps(cbuf, xmm4);
      cbuf += rs_c;
      _mm_storeu_ps(cbuf, xmm5);
      cbuf += rs_c;
      _mm_storeu_ps(cbuf, xmm6);
    }
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_2x4)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_2x4F_DISABLE,
              &&POST_OPS_BIAS_2x4F,
              &&POST_OPS_RELU_2x4F,
              &&POST_OPS_RELU_SCALE_2x4F,
              &&POST_OPS_GELU_TANH_2x4F,
              &&POST_OPS_GELU_ERF_2x4F,
              &&POST_OPS_CLIP_2x4F,
              &&POST_OPS_DOWNSCALE_2x4F,
              &&POST_OPS_MATRIX_ADD_2x4F,
              &&POST_OPS_SWISH_2x4F,
              &&POST_OPS_MATRIX_MUL_2x4F,
              &&POST_OPS_TANH_2x4F,
              &&POST_OPS_SIGMOID_2x4F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;
    float *_cbuf = NULL;

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


    if ( beta != 0.0 )
    {
      xmm3 = _mm_broadcast_ss(&(beta));

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
        ( post_ops_attr.is_first_k == TRUE ) )
      {
        BF16_F32_C_BNZ_4(0,0,xmm1,xmm3,xmm4)
        BF16_F32_C_BNZ_4(1,0,xmm1,xmm3,xmm5)
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back

        F32_C_BNZ_4(_cbuf,rs_c,xmm0,xmm3,xmm4)
        _cbuf += rs_c;
        F32_C_BNZ_4(_cbuf,rs_c,xmm0,xmm3,xmm5)
      }
    }//betazero

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_2x4F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          BF16_F32_BIAS_LOAD_4BF16_AVX2(xmm0, 0);
        }
        else
        {
          xmm0 = _mm_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 0 * 8 ) );
        }

        // c[0,0-3]
        xmm4 = _mm_add_ps( xmm4, xmm0 );

        // c[1,0-3]
        xmm5 = _mm_add_ps( xmm5, xmm0 );
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
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm0, 0);
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm1, 1);
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm2, 2);
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm3, 3);
        }
        else
        {
          xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 0 );
          xmm1 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 1 );
        }

        // c[0,0-3]
        xmm4 = _mm_add_ps( xmm4, xmm0 );

        // c[1,0-3]
        xmm5 = _mm_add_ps( xmm5, xmm1 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_2x4F:
    {
      xmm0 = _mm_setzero_ps();

      // c[0,0-3]
      xmm4 = _mm_max_ps( xmm4, xmm0 );

      // c[1,0-3]
      xmm5 = _mm_max_ps( xmm5, xmm0 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_2x4F:
    {
      xmm0 =
        _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
      xmm1 = _mm_setzero_ps();

      // c[0,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

      // c[1,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm5, xmm0, xmm1, xmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_2x4F:
    {
      __m128 dn, x_tanh;
      __m128i q;

      // c[0,0-3]
      GELU_TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      // c[1,0-3]
      GELU_TANH_F32S_SSE(xmm5, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_2x4F:
    {
      // c[0,0-3]
      GELU_ERF_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

      // c[1,0-3]
      GELU_ERF_F32S_SSE(xmm5, xmm0, xmm1, xmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_2x4F:
    {
      xmm0 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      xmm1 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0,0-3]
      CLIP_F32S_SSE(xmm4, xmm0, xmm1)

      // c[1,0-3]
      CLIP_F32S_SSE(xmm5, xmm0, xmm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_DOWNSCALE_2x4F:
    {
      __m128 selector0 = _mm_setzero_ps();
      __m128 selector1 = _mm_setzero_ps();

      __m128 zero_point0 = _mm_setzero_ps();
      __m128 zero_point1 = _mm_setzero_ps();

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
        selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if( is_bf16 == TRUE )
        {
          BF16_F32_ZP_SCALAR_BCAST_SSE(zero_point0);
          BF16_F32_ZP_SCALAR_BCAST_SSE(zero_point1);
        }
        else
        {
          zero_point0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector0 = _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_j + ( 0 * 4) );
        }
        if( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_4LOAD_SSE(zero_point0,0)
          }
          else
          {
            zero_point0 = _mm_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_j + ( 0 * 4 ) );
          }
        }
        //c[0, 0-3]
        F32_SCL_MULRND_SSE(xmm4, selector0, zero_point0);

        //c[1, 0-3]
        F32_SCL_MULRND_SSE(xmm5, selector0, zero_point0);
      }
      else
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_i + 0 ) );
          selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_i + 1 ) );
        }
        if( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point0,0)
            BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point1,1)
          }
          else
          {
            zero_point0 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 1 ) );
          }
        }
        //c[0, 0-3]
        F32_SCL_MULRND_SSE(xmm4, selector0, zero_point0);

        //c[1, 0-3]
        F32_SCL_MULRND_SSE(xmm5, selector1, zero_point1);
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_2x4F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m128 scl_fctr1 = _mm_setzero_ps();
      __m128 scl_fctr2 = _mm_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
        }
      }

      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,1,5);
        }
        else
        {
          BF16_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr2,1,5);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,1,5);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr2,1,5);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_2x4F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m128 scl_fctr1 = _mm_setzero_ps();
      __m128 scl_fctr2 = _mm_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
        }
      }

      if( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,1,5);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr2,1,5);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,1,5);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr2,1,5);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_2x4F:
    {
        xmm0 =
          _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
        __m128 z, dn;
        __m128i ex_out;

        // c[0,0-3]
        SWISH_F32_SSE_DEF(xmm4, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[1,0-3]
        SWISH_F32_SSE_DEF(xmm5, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_2x4F:
    {
      __m128 dn;
      __m128i q;

      // c[0,0-3]
      TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, q)

      // c[1,0-3]
      TANH_F32S_SSE(xmm5, xmm0, xmm1, xmm2, xmm3, dn, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SIGMOID_2x4F:
    {
        __m128 z, dn;
        __m128i ex_out;

        // c[0,0-3]
        SIGMOID_F32_SSE_DEF(xmm4, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[1,0-3]
        SIGMOID_F32_SSE_DEF(xmm5, xmm1, xmm2, xmm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }

POST_OPS_2x4F_DISABLE:
    ;

    uint32_t tlsb, rounded, temp[4] = {0};
    int i;
    bfloat16* dest;

    if ( ( post_ops_attr.buf_downscale != NULL ) &&
      ( post_ops_attr.is_last_k == TRUE ) )
    {
      STORE_F32_BF16_4XMM(xmm4, 0, 0)
      STORE_F32_BF16_4XMM(xmm5, 1, 0)
    }
    else
    {
      _mm_storeu_ps(cbuf, xmm4);
      cbuf += rs_c;
      _mm_storeu_ps(cbuf, xmm5);
    }
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_1x4)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_1x4F_DISABLE,
              &&POST_OPS_BIAS_1x4F,
              &&POST_OPS_RELU_1x4F,
              &&POST_OPS_RELU_SCALE_1x4F,
              &&POST_OPS_GELU_TANH_1x4F,
              &&POST_OPS_GELU_ERF_1x4F,
              &&POST_OPS_CLIP_1x4F,
              &&POST_OPS_DOWNSCALE_1x4F,
              &&POST_OPS_MATRIX_ADD_1x4F,
              &&POST_OPS_SWISH_1x4F,
              &&POST_OPS_MATRIX_MUL_1x4F,
              &&POST_OPS_TANH_1x4F,
              &&POST_OPS_SIGMOID_1x4F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;

    /*Declare the registers*/
    __m128 xmm0, xmm1, xmm2, xmm3, xmm4;

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


    if ( beta != 0.0 )
    {
      xmm3 = _mm_broadcast_ss(&(beta));

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
        ( post_ops_attr.is_first_k == TRUE ) )
      {
        BF16_F32_C_BNZ_4(0,0,xmm1,xmm3,xmm4)
      }
      else
      {
        //load c and multiply with beta and
        //add to accumulator and store back
        F32_C_BNZ_4(cbuf,rs_c,xmm0,xmm3,xmm4)
      }
    }//betazero

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_1x4F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          BF16_F32_BIAS_LOAD_4BF16_AVX2(xmm0, 0);
        }
        else
        {
          xmm0 = _mm_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 0 * 8 ) );
        }

        // c[0,0-3]
        xmm4 = _mm_add_ps( xmm4, xmm0 );
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
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm0, 0);
        }
        else
        {
          xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 0 );
        }

        // c[0,0-3]
        xmm4 = _mm_add_ps( xmm4, xmm0 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_1x4F:
    {
      xmm0 = _mm_setzero_ps();

      // c[0,0-3]
      xmm4 = _mm_max_ps( xmm4, xmm0 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_1x4F:
    {
      xmm0 =
        _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
      xmm1 = _mm_setzero_ps();

      // c[0,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_1x4F:
    {
      __m128 dn, x_tanh;
      __m128i q;

      // c[0,0-3]
      GELU_TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_1x4F:
    {
      // c[0,0-3]
      GELU_ERF_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_1x4F:
    {
      xmm0 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      xmm1 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0,0-3]
      CLIP_F32S_SSE(xmm4, xmm0, xmm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_DOWNSCALE_1x4F:
    {
      __m128 selector0 = _mm_setzero_ps();

      __m128 zero_point0 = _mm_setzero_ps();

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
        selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if( is_bf16 == TRUE )
        {
          BF16_F32_ZP_SCALAR_BCAST_SSE(zero_point0);
        }
        else
        {
          zero_point0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector0 = _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_j + ( 0 * 4) );
        }
        if( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_4LOAD_SSE(zero_point0,0)
          }
          else
          {
            zero_point0 = _mm_loadu_ps( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_j + ( 0 * 4 ) );
          }
        }
        //c[0, 0-3]
        F32_SCL_MULRND_SSE(xmm4, selector0, zero_point0);
      }
      else
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector0 = _mm_set1_ps( * ( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_i + 0 ) );
        }
        if( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point0,0)
          }
          else
          {
            zero_point0 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 0 ) );
          }
        }
        //c[0, 0-3]
        F32_SCL_MULRND_SSE(xmm4, selector0, zero_point0);
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_1x4F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m128 scl_fctr1 = _mm_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
        }
      }
      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,0,4);
        }
        else
        {
          BF16_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,0,4);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,0,4);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,0,4);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_1x4F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m128 scl_fctr1 = _mm_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
        }
      }
      if( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,0,4);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,0,4);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,0,4);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,0,4);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_1x4F:
    {
        xmm0 =
          _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
        __m128 z, dn;
        __m128i ex_out;

        // c[0,0-3]
        SWISH_F32_SSE_DEF(xmm4, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_1x4F:
    {
      __m128 dn;
      __m128i q;

      // c[0,0-3]
      TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SIGMOID_1x4F:
    {
        __m128 z, dn;
        __m128i ex_out;

        // c[0,0-3]
        SIGMOID_F32_SSE_DEF(xmm4, xmm1, xmm2, xmm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }

POST_OPS_1x4F_DISABLE:
    ;

    uint32_t tlsb, rounded, temp[4] = {0};
    int i;
    bfloat16* dest;

    if ( ( post_ops_attr.buf_downscale != NULL ) &&
      ( post_ops_attr.is_last_k == TRUE ) )
    {
      STORE_F32_BF16_4XMM(xmm4, 0, 0)
    }
    else
    {
      _mm_storeu_ps(cbuf, xmm4);
    }
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_5x2)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_5x2F_DISABLE,
              &&POST_OPS_BIAS_5x2F,
              &&POST_OPS_RELU_5x2F,
              &&POST_OPS_RELU_SCALE_5x2F,
              &&POST_OPS_GELU_TANH_5x2F,
              &&POST_OPS_GELU_ERF_5x2F,
              &&POST_OPS_CLIP_5x2F,
              &&POST_OPS_DOWNSCALE_5x2F,
              &&POST_OPS_MATRIX_ADD_5x2F,
              &&POST_OPS_SWISH_5x2F,
              &&POST_OPS_MATRIX_MUL_5x2F,
              &&POST_OPS_TANH_5x2F,
              &&POST_OPS_SIGMOID_5x2F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;
    float *_cbuf = NULL;

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
        xmm0 = ( __m128 )_mm_load_sd((const double*)bbuf );
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


    if ( beta != 0.0 )
    {
      xmm3 = _mm_broadcast_ss(&(beta));

        if ( ( post_ops_attr.buf_downscale != NULL ) &&
        ( post_ops_attr.is_first_k == TRUE ) )
      {
        BF16_F32_C_BNZ_2(0,0,xmm1,xmm3,xmm4)
        BF16_F32_C_BNZ_2(1,0,xmm1,xmm3,xmm5)
        BF16_F32_C_BNZ_2(2,0,xmm1,xmm3,xmm6)
        BF16_F32_C_BNZ_2(3,0,xmm1,xmm3,xmm7)
        BF16_F32_C_BNZ_2(4,0,xmm1,xmm3,xmm8)
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back
        F32_C_BNZ_2(_cbuf,rs_c,xmm1,xmm3,xmm4)
        _cbuf += rs_c;
        F32_C_BNZ_2(_cbuf,rs_c,xmm1,xmm3,xmm5)
        _cbuf += rs_c;
        F32_C_BNZ_2(_cbuf,rs_c,xmm1,xmm3,xmm6)
        _cbuf += rs_c;
        F32_C_BNZ_2(_cbuf,rs_c,xmm1,xmm3,xmm7)
        _cbuf += rs_c;
        F32_C_BNZ_2(_cbuf,rs_c,xmm1,xmm3,xmm8)
      }
    }//betazero

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_5x2F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          BF16_F32_BIAS_LOAD_2BF16_AVX2(xmm0, 0);
        }
        else
        {
          xmm0 = (__m128)_mm_load_sd((const double *)
                ((float * )post_ops_list_temp->op_args1 +
                post_ops_attr.post_op_c_j + (0 * 8)));
        }

        // c[0,0-3]
        xmm4 = _mm_add_ps( xmm4, xmm0 );

        // c[1,0-3]
        xmm5 = _mm_add_ps( xmm5, xmm0 );

        // c[2,0-3]
        xmm6 = _mm_add_ps( xmm6, xmm0 );

        // c[3,0-3]
        xmm7 = _mm_add_ps( xmm7, xmm0 );

        // c[4,0-3]
        xmm8 = _mm_add_ps( xmm8, xmm0 );
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
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm0, 0);
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm1, 1);
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm2, 2);
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm3, 3);
        }
        else
        {
          xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 0 );
          xmm1 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 1 );
          xmm2 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 2 );
          xmm3 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 3 );
        }

        // c[0,0-3]
        xmm4 = _mm_add_ps( xmm4, xmm0 );

        // c[1,0-3]
        xmm5 = _mm_add_ps( xmm5, xmm1 );

        // c[2,0-3]
        xmm6 = _mm_add_ps( xmm6, xmm2 );

        // c[3,0-3]
        xmm7 = _mm_add_ps( xmm7, xmm3 );

        if( post_ops_list_temp->stor_type == BF16 )
        {
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm0, 4);
        }
        else
        {
          xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 4 );
        }

        // c[4,0-3]
        xmm8 = _mm_add_ps( xmm8, xmm0 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_5x2F:
    {
      xmm0 = _mm_setzero_ps();

      // c[0,0-3]
      xmm4 = _mm_max_ps( xmm4, xmm0 );

      // c[1,0-3]
      xmm5 = _mm_max_ps( xmm5, xmm0 );

      // c[2,0-3]
      xmm6 = _mm_max_ps( xmm6, xmm0 );

      // c[3,0-3]
      xmm7 = _mm_max_ps( xmm7, xmm0 );

      // c[4,0-3]
      xmm8 = _mm_max_ps( xmm8, xmm0 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_5x2F:
    {
      xmm0 =
        _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
      xmm1 = _mm_setzero_ps();

      // c[0,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

      // c[1,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm5, xmm0, xmm1, xmm2)

      // c[2,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm6, xmm0, xmm1, xmm2)

      // c[3,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm7, xmm0, xmm1, xmm2)

      // c[4,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm8, xmm0, xmm1, xmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_5x2F:
    {
      __m128 dn, x_tanh;
      __m128i q;

      // c[0,0-3]
      GELU_TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      // c[1,0-3]
      GELU_TANH_F32S_SSE(xmm5, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      // c[2,0-3]
      GELU_TANH_F32S_SSE(xmm6, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      // c[3,0-3]
      GELU_TANH_F32S_SSE(xmm7, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      // c[4,0-3]
      GELU_TANH_F32S_SSE(xmm8, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_5x2F:
    {
      // c[0,0-3]
      GELU_ERF_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

      // c[1,0-3]
      GELU_ERF_F32S_SSE(xmm5, xmm0, xmm1, xmm2)

      // c[2,0-3]
      GELU_ERF_F32S_SSE(xmm6, xmm0, xmm1, xmm2)

      // c[3,0-3]
      GELU_ERF_F32S_SSE(xmm7, xmm0, xmm1, xmm2)

      // c[4,0-3]
      GELU_ERF_F32S_SSE(xmm8, xmm0, xmm1, xmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_5x2F:
    {
      xmm0 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      xmm1 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0,0-3]
      CLIP_F32S_SSE(xmm4, xmm0, xmm1)

      // c[1,0-3]
      CLIP_F32S_SSE(xmm5, xmm0, xmm1)

      // c[2,0-3]
      CLIP_F32S_SSE(xmm6, xmm0, xmm1)

      // c[3,0-3]
      CLIP_F32S_SSE(xmm7, xmm0, xmm1)

      // c[4,0-3]
      CLIP_F32S_SSE(xmm8, xmm0, xmm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_DOWNSCALE_5x2F:
    {
        __m128 selector0 = _mm_setzero_ps();
        __m128 selector1 = _mm_setzero_ps();
        __m128 selector2 = _mm_setzero_ps();
        __m128 selector3 = _mm_setzero_ps();
        __m128 selector4 = _mm_setzero_ps();

        __m128 zero_point0 = _mm_setzero_ps();
        __m128 zero_point1 = _mm_setzero_ps();
        __m128 zero_point2 = _mm_setzero_ps();
        __m128 zero_point3 = _mm_setzero_ps();
        __m128 zero_point4 = _mm_setzero_ps();

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
          selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          selector2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          selector3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          selector4 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        }
        if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_SCALAR_BCAST_SSE( zero_point0 );
            BF16_F32_ZP_SCALAR_BCAST_SSE( zero_point1 );
            BF16_F32_ZP_SCALAR_BCAST_SSE( zero_point2 );
            BF16_F32_ZP_SCALAR_BCAST_SSE( zero_point3 );
            BF16_F32_ZP_SCALAR_BCAST_SSE( zero_point4 );
          }
          else
          {
            zero_point0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point4 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          }
        }
        if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
              ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          if( post_ops_list_temp->scale_factor_len > 1 )
          {
            selector0 = ( __m128 )_mm_load_sd( (const double*)( ( float* )post_ops_list_temp->scale_factor +
                            post_ops_attr.post_op_c_j + ( 0 * 8 ) ) );
          }
          if( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
          {
            if( is_bf16 == TRUE )
            {
              BF16_F32_ZP_VECTOR_2LOAD_SSE(zero_point0,0)
            }
            else
            {
              zero_point0 = ( __m128 )_mm_load_sd( (const double*)( ( float* )post_ops_list_temp->op_args1 +
                              post_ops_attr.post_op_c_j + ( 0 * 8 ) ) );
            }
          }
          //c[0, 0-3]
          F32_SCL_MULRND_SSE(xmm4, selector0, zero_point0);

          //c[1, 0-3]
          F32_SCL_MULRND_SSE(xmm5, selector0, zero_point0);

          //c[2, 0-3]
          F32_SCL_MULRND_SSE(xmm6, selector0, zero_point0);

          //c[3, 0-3]
          F32_SCL_MULRND_SSE(xmm7, selector0, zero_point0);

          //c[4, 0-3]
          F32_SCL_MULRND_SSE(xmm8, selector0, zero_point0);
        }
        else
        {
          if( post_ops_list_temp->scale_factor_len > 1 )
          {
            selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                            post_ops_attr.post_op_c_i + 0 ) );
            selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                            post_ops_attr.post_op_c_i + 1 ) );
            selector2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                            post_ops_attr.post_op_c_i + 2 ) );
            selector3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                            post_ops_attr.post_op_c_i + 3 ) );
            selector4 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                            post_ops_attr.post_op_c_i + 4 ) );
          }
          if( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
          {
            if( is_bf16 == TRUE )
            {
              BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point0,0)
              BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point1,1)
              BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point2,2)
              BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point3,3)
              BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point4,4)
            }
            else
            {
              zero_point0 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                              post_ops_attr.post_op_c_i + 0 ) );
              zero_point1 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                              post_ops_attr.post_op_c_i + 1 ) );
              zero_point2 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                              post_ops_attr.post_op_c_i + 2 ) );
              zero_point3 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                              post_ops_attr.post_op_c_i + 3 ) );
              zero_point4 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                              post_ops_attr.post_op_c_i + 4 ) );
            }
          }
          //c[0, 0-3]
          F32_SCL_MULRND_SSE(xmm4, selector0, zero_point0);

          //c[1, 0-3]
          F32_SCL_MULRND_SSE(xmm5, selector1, zero_point1);

          //c[2, 0-3]
          F32_SCL_MULRND_SSE(xmm6, selector2, zero_point2);

          //c[3, 0-3]
          F32_SCL_MULRND_SSE(xmm7, selector3, zero_point3);

          //c[4, 0-3]
          F32_SCL_MULRND_SSE(xmm8, selector4, zero_point4);
        }
        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_MATRIX_ADD_5x2F:
      {
        dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

        bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

        __m128 scl_fctr1 = _mm_setzero_ps();
        __m128 scl_fctr2 = _mm_setzero_ps();
        __m128 scl_fctr3 = _mm_setzero_ps();
        __m128 scl_fctr4 = _mm_setzero_ps();
        __m128 scl_fctr5 = _mm_setzero_ps();

        // Even though different registers are used for scalar in column and
        // row major case, all those registers will contain the same value.
        if ( post_ops_list_temp->scale_factor_len == 1 )
        {
          scl_fctr1 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          scl_fctr2 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          scl_fctr3 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          scl_fctr4 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          scl_fctr5 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        }
        else
        {
          if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
          {
            scl_fctr1 =
              _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          }
          else
          {
            scl_fctr1 =
              _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 0 ) );
            scl_fctr2 =
              _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 1 ) );
            scl_fctr3 =
              _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 2 ) );
            scl_fctr4 =
              _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 3 ) );
            scl_fctr5 =
              _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 4 ) );
          }
        }

        if ( is_bf16 == TRUE )
        {
          bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

          if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
          {
            // c[0:0-15]
            BF16_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,0,4);

            // c[1:0-15]
            BF16_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,1,5);

            // c[2:0-15]
            BF16_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,2,6);

            // c[3:0-15]
            BF16_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,3,7);

            // c[4:0-15]
            BF16_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,4,8);
          }
          else
          {
            // c[0:0-15]
            BF16_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,0,4);

            // c[1:0-15]
            BF16_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr2,1,5);

            // c[2:0-15]
            BF16_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr3,2,6);

            // c[3:0-15]
            BF16_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr4,3,7);

            // c[4:0-15]
            BF16_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr5,4,8);
          }
        }
        else
        {
          float* matptr = ( float* )post_ops_list_temp->op_args1;

          if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
          {
            // c[0:0-15]
            F32_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,0,4);

            // c[1:0-15]
            F32_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,1,5);

            // c[2:0-15]
            F32_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,2,6);

            // c[3:0-15]
            F32_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,3,7);

            // c[4:0-15]
            F32_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,4,8);
          }
          else
          {
            // c[0:0-15]
            F32_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,0,4);

            // c[1:0-15]
            F32_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr2,1,5);

            // c[2:0-15]
            F32_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr3,2,6);

            // c[3:0-15]
            F32_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr4,3,7);

            // c[4:0-15]
            F32_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr5,4,8);
          }
        }
        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_MATRIX_MUL_5x2F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m128 scl_fctr1 = _mm_setzero_ps();
      __m128 scl_fctr2 = _mm_setzero_ps();
      __m128 scl_fctr3 = _mm_setzero_ps();
      __m128 scl_fctr4 = _mm_setzero_ps();
      __m128 scl_fctr5 = _mm_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr4 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr5 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
          scl_fctr4 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 3 ) );
          scl_fctr5 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 4 ) );
        }
      }

      if( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,1,5);

          // c[2:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,2,6);

          // c[3:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,3,7);

          // c[4:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,4,8);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr2,1,5);

          // c[2:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr3,2,6);

          // c[3:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr4,3,7);

          // c[4:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr5,4,8);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,1,5);

          // c[2:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,2,6);

          // c[3:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,3,7);

          // c[4:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,4,8);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr2,1,5);

          // c[2:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr3,2,6);

          // c[3:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr4,3,7);

          // c[4:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr5,4,8);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_5x2F:
    {
        xmm0 =
          _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
        __m128 z, dn;
        __m128i ex_out;

        // c[0,0-1]
        SWISH_F32_SSE_DEF(xmm4, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[1,0-1]
        SWISH_F32_SSE_DEF(xmm5, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[2,0-1]
        SWISH_F32_SSE_DEF(xmm6, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[3,0-1]
        SWISH_F32_SSE_DEF(xmm7, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[4,0-1]
        SWISH_F32_SSE_DEF(xmm8, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_5x2F:
    {
      __m128 dn;
      __m128i q;

      // c[0,0-3]
      TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, q)

      // c[1,0-3]
      TANH_F32S_SSE(xmm5, xmm0, xmm1, xmm2, xmm3, dn, q)

      // c[2,0-3]
      TANH_F32S_SSE(xmm6, xmm0, xmm1, xmm2, xmm3, dn, q)

      // c[3,0-3]
      TANH_F32S_SSE(xmm7, xmm0, xmm1, xmm2, xmm3, dn, q)

      // c[4,0-3]
      TANH_F32S_SSE(xmm8, xmm0, xmm1, xmm2, xmm3, dn, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SIGMOID_5x2F:
    {
        __m128 z, dn;
        __m128i ex_out;

        // c[0,0-1]
        SIGMOID_F32_SSE_DEF(xmm4, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[1,0-1]
        SIGMOID_F32_SSE_DEF(xmm5, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[2,0-1]
        SIGMOID_F32_SSE_DEF(xmm6, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[3,0-1]
        SIGMOID_F32_SSE_DEF(xmm7, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[4,0-1]
        SIGMOID_F32_SSE_DEF(xmm8, xmm1, xmm2, xmm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_5x2F_DISABLE:
    ;

    uint32_t tlsb, rounded, temp[2] = {0};
    int i;
    bfloat16* dest;

    if ( ( post_ops_attr.buf_downscale != NULL ) &&
      ( post_ops_attr.is_last_k == TRUE ) )
    {
      STORE_F32_BF16_2XMM(xmm4, 0, 0)
      STORE_F32_BF16_2XMM(xmm5, 1, 0)
      STORE_F32_BF16_2XMM(xmm6, 2, 0)
      STORE_F32_BF16_2XMM(xmm7, 3, 0)
      STORE_F32_BF16_2XMM(xmm8, 4, 0)
    }
    else
    {
      _mm_store_sd((double*)cbuf, ( __m128d )xmm4);
      cbuf += rs_c;
      _mm_store_sd((double*)cbuf, ( __m128d )xmm5);
      cbuf += rs_c;
      _mm_store_sd((double*)cbuf, ( __m128d )xmm6);
      cbuf += rs_c;
      _mm_store_sd((double*)cbuf, ( __m128d )xmm7);
      cbuf += rs_c;
      _mm_store_sd((double*)cbuf, ( __m128d )xmm8);
    }
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_4x2)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_4x2F_DISABLE,
              &&POST_OPS_BIAS_4x2F,
              &&POST_OPS_RELU_4x2F,
              &&POST_OPS_RELU_SCALE_4x2F,
              &&POST_OPS_GELU_TANH_4x2F,
              &&POST_OPS_GELU_ERF_4x2F,
              &&POST_OPS_CLIP_4x2F,
              &&POST_OPS_DOWNSCALE_4x2F,
              &&POST_OPS_MATRIX_ADD_4x2F,
              &&POST_OPS_SWISH_4x2F,
              &&POST_OPS_MATRIX_MUL_4x2F,
              &&POST_OPS_TANH_4x2F,
              &&POST_OPS_SIGMOID_4x2F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;
    float *_cbuf = NULL;

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
        xmm0 = ( __m128 )_mm_load_sd((const double*)bbuf );
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


    if ( beta != 0.0 )
    {
      xmm3 = _mm_broadcast_ss(&(beta));
      if ( ( post_ops_attr.buf_downscale != NULL ) &&
        ( post_ops_attr.is_first_k == TRUE ) )
      {
        BF16_F32_C_BNZ_2(0,0,xmm1,xmm3,xmm4)
        BF16_F32_C_BNZ_2(1,0,xmm1,xmm3,xmm5)
        BF16_F32_C_BNZ_2(2,0,xmm1,xmm3,xmm6)
        BF16_F32_C_BNZ_2(3,0,xmm1,xmm3,xmm7)
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back
        F32_C_BNZ_2(_cbuf,rs_c,xmm1,xmm3,xmm4)
        _cbuf += rs_c;
        F32_C_BNZ_2(_cbuf,rs_c,xmm1,xmm3,xmm5)
        _cbuf += rs_c;
        F32_C_BNZ_2(_cbuf,rs_c,xmm1,xmm3,xmm6)
        _cbuf += rs_c;
        F32_C_BNZ_2(_cbuf,rs_c,xmm1,xmm3,xmm7)
      }
    }//betazero

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_4x2F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          BF16_F32_BIAS_LOAD_2BF16_AVX2(xmm0, 0);
        }
        else
        {
          xmm0 = (__m128)_mm_load_sd((const double *)
                ((float *)post_ops_list_temp->op_args1 +
                post_ops_attr.post_op_c_j + (0 * 8)));
        }

        // c[0,0-3]
        xmm4 = _mm_add_ps( xmm4, xmm0 );

        // c[1,0-3]
        xmm5 = _mm_add_ps( xmm5, xmm0 );

        // c[2,0-3]
        xmm6 = _mm_add_ps( xmm6, xmm0 );

        // c[3,0-3]
        xmm7 = _mm_add_ps( xmm7, xmm0 );
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
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm0, 0);
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm1, 1);
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm2, 2);
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm3, 3);
        }
        else
        {
          xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 0 );
          xmm1 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 1 );
          xmm2 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 2 );
          xmm3 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 3 );
        }

        // c[0,0-3]
        xmm4 = _mm_add_ps( xmm4, xmm0 );

        // c[1,0-3]
        xmm5 = _mm_add_ps( xmm5, xmm1 );

        // c[2,0-3]
        xmm6 = _mm_add_ps( xmm6, xmm2 );

        // c[3,0-3]
        xmm7 = _mm_add_ps( xmm7, xmm3 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_4x2F:
    {
      xmm0 = _mm_setzero_ps();

      // c[0,0-3]
      xmm4 = _mm_max_ps( xmm4, xmm0 );

      // c[1,0-3]
      xmm5 = _mm_max_ps( xmm5, xmm0 );

      // c[2,0-3]
      xmm6 = _mm_max_ps( xmm6, xmm0 );

      // c[3,0-3]
      xmm7 = _mm_max_ps( xmm7, xmm0 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_4x2F:
    {
      xmm0 =
        _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
      xmm1 = _mm_setzero_ps();

      // c[0,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

      // c[1,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm5, xmm0, xmm1, xmm2)

      // c[2,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm6, xmm0, xmm1, xmm2)

      // c[3,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm7, xmm0, xmm1, xmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_4x2F:
    {
      __m128 dn, x_tanh;
      __m128i q;

      // c[0,0-3]
      GELU_TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      // c[1,0-3]
      GELU_TANH_F32S_SSE(xmm5, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      // c[2,0-3]
      GELU_TANH_F32S_SSE(xmm6, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      // c[3,0-3]
      GELU_TANH_F32S_SSE(xmm7, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_4x2F:
    {
      // c[0,0-3]
      GELU_ERF_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

      // c[1,0-3]
      GELU_ERF_F32S_SSE(xmm5, xmm0, xmm1, xmm2)

      // c[2,0-3]
      GELU_ERF_F32S_SSE(xmm6, xmm0, xmm1, xmm2)

      // c[3,0-3]
      GELU_ERF_F32S_SSE(xmm7, xmm0, xmm1, xmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_4x2F:
    {
      xmm0 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      xmm1 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0,0-3]
      CLIP_F32S_SSE(xmm4, xmm0, xmm1)

      // c[1,0-3]
      CLIP_F32S_SSE(xmm5, xmm0, xmm1)

      // c[2,0-3]
      CLIP_F32S_SSE(xmm6, xmm0, xmm1)

      // c[3,0-3]
      CLIP_F32S_SSE(xmm7, xmm0, xmm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_DOWNSCALE_4x2F:
    {
      __m128 selector0 = _mm_setzero_ps();
      __m128 selector1 = _mm_setzero_ps();
      __m128 selector2 = _mm_setzero_ps();
      __m128 selector3 = _mm_setzero_ps();

      __m128 zero_point0 = _mm_setzero_ps();
      __m128 zero_point1 = _mm_setzero_ps();
      __m128 zero_point2 = _mm_setzero_ps();
      __m128 zero_point3 = _mm_setzero_ps();

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
        selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if( is_bf16 == TRUE )
        {
          BF16_F32_ZP_SCALAR_BCAST_SSE( zero_point0 );
          BF16_F32_ZP_SCALAR_BCAST_SSE( zero_point1 );
          BF16_F32_ZP_SCALAR_BCAST_SSE( zero_point2 );
          BF16_F32_ZP_SCALAR_BCAST_SSE( zero_point3 );
        }
        else
        {
          zero_point0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector0 = ( __m128 )_mm_load_sd( (const double*)( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_j + ( 0 * 8 ) ) );
        }
        if( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_2LOAD_SSE(zero_point0,0)
          }
          else
          {
            zero_point0 = ( __m128 )_mm_load_sd( (const double*)( ( float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_j + ( 0 * 8 ) ) );
          }
        }
        //c[0, 0-3]
        F32_SCL_MULRND_SSE(xmm4, selector0, zero_point0);

        //c[1, 0-3]
        F32_SCL_MULRND_SSE(xmm5, selector0, zero_point0);

        //c[2, 0-3]
        F32_SCL_MULRND_SSE(xmm6, selector0, zero_point0);

        //c[3, 0-3]
        F32_SCL_MULRND_SSE(xmm7, selector0, zero_point0);
      }
      else
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_i + 0 ) );
          selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_i + 1 ) );
          selector2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_i + 2 ) );
          selector3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_i + 3 ) );
        }
        if( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point0,0)
            BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point1,1)
            BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point2,2)
            BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point3,3)
          }
          else
          {
            zero_point0 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 1 ) );
            zero_point2 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 2 ) );
            zero_point3 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 3 ) );
          }
        }
        //c[0, 0-3]
        F32_SCL_MULRND_SSE(xmm4, selector0, zero_point0);

        //c[1, 0-3]
        F32_SCL_MULRND_SSE(xmm5, selector1, zero_point1);

        //c[2, 0-3]
        F32_SCL_MULRND_SSE(xmm6, selector2, zero_point2);

        //c[3, 0-3]
        F32_SCL_MULRND_SSE(xmm7, selector3, zero_point3);
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_MATRIX_ADD_4x2F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
        ( ( post_ops_list_temp->stor_type == NONE ) &&
          ( post_ops_attr.c_stor_type == BF16 ) );

      __m128 scl_fctr1 = _mm_setzero_ps();
      __m128 scl_fctr2 = _mm_setzero_ps();
      __m128 scl_fctr3 = _mm_setzero_ps();
      __m128 scl_fctr4 = _mm_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr4 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
          scl_fctr4 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 3 ) );
        }
      }
      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,1,5);

          // c[2:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,2,6);

          // c[3:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,3,7);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr2,1,5);

          // c[2:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr3,2,6);

          // c[3:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr4,3,7);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,1,5);

          // c[2:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,2,6);

          // c[3:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,3,7);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr2,1,5);

          // c[2:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr3,2,6);

          // c[3:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr4,3,7);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_4x2F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m128 scl_fctr1 = _mm_setzero_ps();
      __m128 scl_fctr2 = _mm_setzero_ps();
      __m128 scl_fctr3 = _mm_setzero_ps();
      __m128 scl_fctr4 = _mm_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr4 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
          scl_fctr4 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 3 ) );
        }
      }

      if( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,1,5);

          // c[2:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,2,6);

          // c[3:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,3,7);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr2,1,5);

          // c[2:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr3,2,6);

          // c[3:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr4,3,7);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,1,5);

          // c[2:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,2,6);

          // c[3:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,3,7);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr2,1,5);

          // c[2:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr3,2,6);

          // c[3:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr4,3,7);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_4x2F:
    {
        xmm0 =
          _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
        __m128 z, dn;
        __m128i ex_out;

        // c[0,0-1]
        SWISH_F32_SSE_DEF(xmm4, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[1,0-1]
        SWISH_F32_SSE_DEF(xmm5, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[2,0-1]
        SWISH_F32_SSE_DEF(xmm6, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[3,0-1]
        SWISH_F32_SSE_DEF(xmm7, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_4x2F:
    {
      __m128 dn;
      __m128i q;

      // c[0,0-3]
      TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, q)

      // c[1,0-3]
      TANH_F32S_SSE(xmm5, xmm0, xmm1, xmm2, xmm3, dn, q)

      // c[2,0-3]
      TANH_F32S_SSE(xmm6, xmm0, xmm1, xmm2, xmm3, dn, q)

      // c[3,0-3]
      TANH_F32S_SSE(xmm7, xmm0, xmm1, xmm2, xmm3, dn, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SIGMOID_4x2F:
    {
        __m128 z, dn;
        __m128i ex_out;

        // c[0,0-1]
        SIGMOID_F32_SSE_DEF(xmm4, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[1,0-1]
        SIGMOID_F32_SSE_DEF(xmm5, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[2,0-1]
        SIGMOID_F32_SSE_DEF(xmm6, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[3,0-1]
        SIGMOID_F32_SSE_DEF(xmm7, xmm1, xmm2, xmm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_4x2F_DISABLE:
    ;

    uint32_t tlsb, rounded, temp[2] = {0};
    int i;
    bfloat16* dest;

    if ( ( post_ops_attr.buf_downscale != NULL ) &&
      ( post_ops_attr.is_last_k == TRUE ) )
    {
      STORE_F32_BF16_2XMM(xmm4, 0, 0)
      STORE_F32_BF16_2XMM(xmm5, 1, 0)
      STORE_F32_BF16_2XMM(xmm6, 2, 0)
      STORE_F32_BF16_2XMM(xmm7, 3, 0)
    }
    else
    {
      _mm_store_sd((double*)cbuf, ( __m128d )xmm4);
      cbuf += rs_c;
      _mm_store_sd((double*)cbuf, ( __m128d )xmm5);
      cbuf += rs_c;
      _mm_store_sd((double*)cbuf, ( __m128d )xmm6);
      cbuf += rs_c;
      _mm_store_sd((double*)cbuf, ( __m128d )xmm7);
    }
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_3x2)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_3x2F_DISABLE,
              &&POST_OPS_BIAS_3x2F,
              &&POST_OPS_RELU_3x2F,
              &&POST_OPS_RELU_SCALE_3x2F,
              &&POST_OPS_GELU_TANH_3x2F,
              &&POST_OPS_GELU_ERF_3x2F,
              &&POST_OPS_CLIP_3x2F,
              &&POST_OPS_DOWNSCALE_3x2F,
              &&POST_OPS_MATRIX_ADD_3x2F,
              &&POST_OPS_SWISH_3x2F,
              &&POST_OPS_MATRIX_MUL_3x2F,
              &&POST_OPS_TANH_3x2F,
              &&POST_OPS_SIGMOID_3x2F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;
    float *_cbuf = NULL;

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
        xmm0 = ( __m128 )_mm_load_sd((const double*)bbuf );
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


    if ( beta != 0.0 )
    {
      xmm3 = _mm_broadcast_ss(&(beta));

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
      ( post_ops_attr.is_first_k == TRUE ) )
      {
        BF16_F32_C_BNZ_2(0,0,xmm1,xmm3,xmm4)
        BF16_F32_C_BNZ_2(1,0,xmm1,xmm3,xmm5)
        BF16_F32_C_BNZ_2(2,0,xmm1,xmm3,xmm6)
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back
        F32_C_BNZ_2(_cbuf,rs_c,xmm1,xmm3,xmm4)
        _cbuf += rs_c;
        F32_C_BNZ_2(_cbuf,rs_c,xmm1,xmm3,xmm5)
        _cbuf += rs_c;
        F32_C_BNZ_2(_cbuf,rs_c,xmm1,xmm3,xmm6)
      }
    }//betazero

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_3x2F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          BF16_F32_BIAS_LOAD_2BF16_AVX2(xmm0, 0);
        }
        else
        {
          xmm0 = (__m128)_mm_load_sd( (const double *)
                ((float *) post_ops_list_temp->op_args1 +
                post_ops_attr.post_op_c_j + (0 * 8)));
        }

        // c[0,0-3]
        xmm4 = _mm_add_ps( xmm4, xmm0 );

        // c[1,0-3]
        xmm5 = _mm_add_ps( xmm5, xmm0 );

        // c[2,0-3]
        xmm6 = _mm_add_ps( xmm6, xmm0 );
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
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm0, 0);
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm1, 1);
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm2, 2);
        }
        else
        {
          xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 0 );
          xmm1 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 1 );
          xmm2 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 2 );
        }

        // c[0,0-3]
        xmm4 = _mm_add_ps( xmm4, xmm0 );

        // c[1,0-3]
        xmm5 = _mm_add_ps( xmm5, xmm1 );

        // c[2,0-3]
        xmm6 = _mm_add_ps( xmm6, xmm2 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_3x2F:
    {
      xmm0 = _mm_setzero_ps();

      // c[0,0-3]
      xmm4 = _mm_max_ps( xmm4, xmm0 );

      // c[1,0-3]
      xmm5 = _mm_max_ps( xmm5, xmm0 );

      // c[2,0-3]
      xmm6 = _mm_max_ps( xmm6, xmm0 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_3x2F:
    {
      xmm0 =
        _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
      xmm1 = _mm_setzero_ps();

      // c[0,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

      // c[1,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm5, xmm0, xmm1, xmm2)

      // c[2,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm6, xmm0, xmm1, xmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_3x2F:
    {
      __m128 dn, x_tanh;
      __m128i q;

      // c[0,0-3]
      GELU_TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      // c[1,0-3]
      GELU_TANH_F32S_SSE(xmm5, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      // c[2,0-3]
      GELU_TANH_F32S_SSE(xmm6, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_3x2F:
    {
      // c[0,0-3]
      GELU_ERF_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

      // c[1,0-3]
      GELU_ERF_F32S_SSE(xmm5, xmm0, xmm1, xmm2)

      // c[2,0-3]
      GELU_ERF_F32S_SSE(xmm6, xmm0, xmm1, xmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_3x2F:
    {
      xmm0 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      xmm1 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0,0-3]
      CLIP_F32S_SSE(xmm4, xmm0, xmm1)

      // c[1,0-3]
      CLIP_F32S_SSE(xmm5, xmm0, xmm1)

      // c[2,0-3]
      CLIP_F32S_SSE(xmm6, xmm0, xmm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_DOWNSCALE_3x2F:
    {
      __m128 selector0 = _mm_setzero_ps();
      __m128 selector1 = _mm_setzero_ps();
      __m128 selector2 = _mm_setzero_ps();

      __m128 zero_point0 = _mm_setzero_ps();
      __m128 zero_point1 = _mm_setzero_ps();
      __m128 zero_point2 = _mm_setzero_ps();

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
        selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if( is_bf16 == TRUE )
        {
          BF16_F32_ZP_SCALAR_BCAST_SSE( zero_point0 )
          BF16_F32_ZP_SCALAR_BCAST_SSE( zero_point1 )
          BF16_F32_ZP_SCALAR_BCAST_SSE( zero_point2 )
        }
        else
        {
          zero_point0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector0 = ( __m128 )_mm_load_sd( (const double*)( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_j + ( 0 * 8 ) ) );
        }
        if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_2LOAD_SSE(zero_point0,0)
          }
          else
          {
            zero_point0 = ( __m128 )_mm_load_sd( (const double*)( ( float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_j + ( 0 * 8 ) ) );
          }
        }
        //c[0, 0-3]
        F32_SCL_MULRND_SSE(xmm4, selector0, zero_point0);

        //c[1, 0-3]
        F32_SCL_MULRND_SSE(xmm5, selector0, zero_point0);

        //c[2, 0-3]
        F32_SCL_MULRND_SSE(xmm6, selector0, zero_point0);
      }
      else
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_i + 0 ) );
          selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_i + 1 ) );
          selector2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_i + 2 ) );
        }
        if( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point0,0)
            BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point1,1)
            BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point2,2)
          }
          else
          {
            zero_point0 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 1 ) );
            zero_point2 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 2 ) );
          }
        }
        //c[0, 0-3]
        F32_SCL_MULRND_SSE(xmm4, selector0, zero_point0);

        //c[1, 0-3]
        F32_SCL_MULRND_SSE(xmm5, selector1, zero_point1);

        //c[2, 0-3]
        F32_SCL_MULRND_SSE(xmm6, selector2, zero_point2);
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_MATRIX_ADD_3x2F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
        ( ( post_ops_list_temp->stor_type == NONE ) &&
          ( post_ops_attr.c_stor_type == BF16 ) );

      __m128 scl_fctr1 = _mm_setzero_ps();
      __m128 scl_fctr2 = _mm_setzero_ps();
      __m128 scl_fctr3 = _mm_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
        }
      }
      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,1,5);

          // c[2:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,2,6);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr2,1,5);

          // c[2:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr3,2,6);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,1,5);

          // c[2:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,2,6);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr2,1,5);

          // c[2:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr3,2,6);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_3x2F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m128 scl_fctr1 = _mm_setzero_ps();
      __m128 scl_fctr2 = _mm_setzero_ps();
      __m128 scl_fctr3 = _mm_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
        }
      }

      if( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,1,5);

          // c[2:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,2,6);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr2,1,5);

          // c[2:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr3,2,6);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,1,5);

          // c[2:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,2,6);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr2,1,5);

          // c[2:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr3,2,6);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_3x2F:
    {
        xmm0 =
          _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
        __m128 z, dn;
        __m128i ex_out;

        // c[0,0-1]
        SWISH_F32_SSE_DEF(xmm4, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[1,0-1]
        SWISH_F32_SSE_DEF(xmm5, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[2,0-1]
        SWISH_F32_SSE_DEF(xmm6, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_3x2F:
    {
      __m128 dn;
      __m128i q;

      // c[0,0-3]
      TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, q)

      // c[1,0-3]
      TANH_F32S_SSE(xmm5, xmm0, xmm1, xmm2, xmm3, dn, q)

      // c[2,0-3]
      TANH_F32S_SSE(xmm6, xmm0, xmm1, xmm2, xmm3, dn, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SIGMOID_3x2F:
    {
        __m128 z, dn;
        __m128i ex_out;

        // c[0,0-1]
        SIGMOID_F32_SSE_DEF(xmm4, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[1,0-1]
        SIGMOID_F32_SSE_DEF(xmm5, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[2,0-1]
        SIGMOID_F32_SSE_DEF(xmm6, xmm1, xmm2, xmm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_3x2F_DISABLE:
    ;

    uint32_t tlsb, rounded, temp[2] = {0};
    int i;
    bfloat16* dest;

    if ( ( post_ops_attr.buf_downscale != NULL ) &&
      ( post_ops_attr.is_last_k == TRUE ) )
    {
      STORE_F32_BF16_2XMM(xmm4, 0, 0)
      STORE_F32_BF16_2XMM(xmm5, 1, 0)
      STORE_F32_BF16_2XMM(xmm6, 2, 0)
    }
    else
    {
      _mm_store_sd((double*)cbuf, ( __m128d )xmm4);
      cbuf += rs_c;
      _mm_store_sd((double*)cbuf, ( __m128d )xmm5);
      cbuf += rs_c;
      _mm_store_sd((double*)cbuf, ( __m128d )xmm6);
    }
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_2x2)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_2x2F_DISABLE,
              &&POST_OPS_BIAS_2x2F,
              &&POST_OPS_RELU_2x2F,
              &&POST_OPS_RELU_SCALE_2x2F,
              &&POST_OPS_GELU_TANH_2x2F,
              &&POST_OPS_GELU_ERF_2x2F,
              &&POST_OPS_CLIP_2x2F,
              &&POST_OPS_DOWNSCALE_2x2F,
              &&POST_OPS_MATRIX_ADD_2x2F,
              &&POST_OPS_SWISH_2x2F,
              &&POST_OPS_MATRIX_MUL_2x2F,
              &&POST_OPS_TANH_2x2F,
              &&POST_OPS_SIGMOID_2x2F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;
    float *_cbuf = NULL;

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
        xmm0 = ( __m128 )_mm_load_sd((const double*)bbuf );
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


    if ( beta != 0.0 )
    {
      xmm3 = _mm_broadcast_ss(&(beta));

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
        ( post_ops_attr.is_first_k == TRUE ) )
      {
        BF16_F32_C_BNZ_2(0,0,xmm1,xmm3,xmm4)
        BF16_F32_C_BNZ_2(1,0,xmm1,xmm3,xmm5)
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back
        F32_C_BNZ_2(_cbuf,rs_c,xmm0,xmm3,xmm4)
        _cbuf += rs_c;
        F32_C_BNZ_2(_cbuf,rs_c,xmm0,xmm3,xmm5)
      }
    }//betazero

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_2x2F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          BF16_F32_BIAS_LOAD_2BF16_AVX2(xmm0, 0);
        }
        else
        {
          xmm0 = (__m128)_mm_load_sd((const double *)
                ((float *)post_ops_list_temp->op_args1 +
                post_ops_attr.post_op_c_j + (0 * 8)));
        }

        // c[0,0-3]
        xmm4 = _mm_add_ps( xmm4, xmm0 );

        // c[1,0-3]
        xmm5 = _mm_add_ps( xmm5, xmm0 );
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
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm0, 0);
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm1, 1);
        }
        else
        {
          xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 0 );
          xmm1 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 1 );
        }

        // c[0,0-3]
        xmm4 = _mm_add_ps( xmm4, xmm0 );

        // c[1,0-3]
        xmm5 = _mm_add_ps( xmm5, xmm1 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_2x2F:
    {
      xmm0 = _mm_setzero_ps();

      // c[0,0-3]
      xmm4 = _mm_max_ps( xmm4, xmm0 );

      // c[1,0-3]
      xmm5 = _mm_max_ps( xmm5, xmm0 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_2x2F:
    {
      xmm0 =
        _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
      xmm1 = _mm_setzero_ps();

      // c[0,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

      // c[1,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm5, xmm0, xmm1, xmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_2x2F:
    {
      __m128 dn, x_tanh;
      __m128i q;

      // c[0,0-3]
      GELU_TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      // c[1,0-3]
      GELU_TANH_F32S_SSE(xmm5, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_2x2F:
    {
      // c[0,0-3]
      GELU_ERF_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

      // c[1,0-3]
      GELU_ERF_F32S_SSE(xmm5, xmm0, xmm1, xmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_2x2F:
    {
      xmm0 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      xmm1 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0,0-3]
      CLIP_F32S_SSE(xmm4, xmm0, xmm1)

      // c[1,0-3]
      CLIP_F32S_SSE(xmm5, xmm0, xmm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_DOWNSCALE_2x2F:
    {
      __m128 selector0 = _mm_setzero_ps();
      __m128 selector1 = _mm_setzero_ps();

      __m128 zero_point0 = _mm_setzero_ps();
      __m128 zero_point1 = _mm_setzero_ps();

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
        selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if( is_bf16 == TRUE )
        {
          BF16_F32_ZP_SCALAR_BCAST_SSE( zero_point0 );
          BF16_F32_ZP_SCALAR_BCAST_SSE( zero_point1 );
        }
        else
        {
          zero_point0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector0 = ( __m128 )_mm_load_sd( (const double*)( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_j + ( 0 * 8 ) ) );
        }
        if( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_2LOAD_SSE(zero_point0,0)
          }
          else
          {
            zero_point0 = ( __m128 )_mm_load_sd( (const double*)( ( float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_j + ( 0 * 8 ) ) );
          }
        }
        //c[0, 0-3]
        F32_SCL_MULRND_SSE(xmm4, selector0, zero_point0);

        //c[1, 0-3]
        F32_SCL_MULRND_SSE(xmm5, selector0, zero_point0);
      }
      else
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_i + 0 ) );
          selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_i + 1 ) );
        }
        if( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point0,0)
            BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point1,1)
          }
          else
          {
            zero_point0 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 1 ) );
          }
        }
        //c[0, 0-3]
        F32_SCL_MULRND_SSE(xmm4, selector0, zero_point0);

        //c[1, 0-3]
        F32_SCL_MULRND_SSE(xmm5, selector1, zero_point1);
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_2x2F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
        ( ( post_ops_list_temp->stor_type == NONE ) &&
          ( post_ops_attr.c_stor_type == BF16 ) );

      __m128 scl_fctr1 = _mm_setzero_ps();
      __m128 scl_fctr2 = _mm_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
        }
      }
      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,1,5);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr2,1,5);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,1,5);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr2,1,5);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_2x2F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m128 scl_fctr1 = _mm_setzero_ps();
      __m128 scl_fctr2 = _mm_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
        }
      }

      if( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,1,5);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr2,1,5);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,1,5);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr2,1,5);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_2x2F:
    {
        xmm0 =
          _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
        __m128 z, dn;
        __m128i ex_out;

        // c[0,0-1]
        SWISH_F32_SSE_DEF(xmm4, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[1,0-1]
        SWISH_F32_SSE_DEF(xmm5, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_2x2F:
    {
      __m128 dn;
      __m128i q;

      // c[0,0-3]
      TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, q)

      // c[1,0-3]
      TANH_F32S_SSE(xmm5, xmm0, xmm1, xmm2, xmm3, dn, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SIGMOID_2x2F:
    {
        __m128 z, dn;
        __m128i ex_out;

        // c[0,0-1]
        SIGMOID_F32_SSE_DEF(xmm4, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[1,0-1]
        SIGMOID_F32_SSE_DEF(xmm5, xmm1, xmm2, xmm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_2x2F_DISABLE:
    ;

    uint32_t tlsb, rounded, temp[2] = {0};
    int i;
    bfloat16* dest;

    if ( ( post_ops_attr.buf_downscale != NULL ) &&
      ( post_ops_attr.is_last_k == TRUE ) )
    {
      STORE_F32_BF16_2XMM(xmm4, 0, 0)
      STORE_F32_BF16_2XMM(xmm5, 1, 0)
    }
    else
    {
      _mm_store_sd((double*)cbuf, ( __m128d )xmm4);
      cbuf += rs_c;
      _mm_store_sd((double*)cbuf, ( __m128d )xmm5);
    }
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_1x2)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_1x2F_DISABLE,
              &&POST_OPS_BIAS_1x2F,
              &&POST_OPS_RELU_1x2F,
              &&POST_OPS_RELU_SCALE_1x2F,
              &&POST_OPS_GELU_TANH_1x2F,
              &&POST_OPS_GELU_ERF_1x2F,
              &&POST_OPS_CLIP_1x2F,
              &&POST_OPS_DOWNSCALE_1x2F,
              &&POST_OPS_MATRIX_ADD_1x2F,
              &&POST_OPS_SWISH_1x2F,
              &&POST_OPS_MATRIX_MUL_1x2F,
              &&POST_OPS_TANH_1x2F,
              &&POST_OPS_SIGMOID_1x2F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;

    /*Declare the registers*/
    __m128 xmm0, xmm1, xmm2, xmm3, xmm4;

    /* zero the accumulator registers */
    xmm4 = _mm_setzero_ps();

    /*_mm_prefetch( (MR X NR) from C*/
    _mm_prefetch((cbuf + 0*rs_c), _MM_HINT_T0);

    for(dim_t k = 0; k < k_iter; k++)
    {
        /*Load 16 elements from row0 of B*/
        xmm0 = ( __m128 )_mm_load_sd((const double*)bbuf );
        bbuf += rs_b;  //move b pointer to next row
        xmm1 = _mm_broadcast_ss((abuf + 0*rs_a)); //broadcast c0r0
        abuf += cs_a;  //move a pointer to next col
        xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);
    }//kloop

    xmm0 = _mm_broadcast_ss(&(alpha));
    xmm4 = _mm_mul_ps(xmm4,xmm0);


    if ( beta != 0.0 )
    {
      xmm3 = _mm_broadcast_ss(&(beta));
      if ( ( post_ops_attr.buf_downscale != NULL ) &&
        ( post_ops_attr.is_first_k == TRUE ) )
      {
        BF16_F32_C_BNZ_2(0,0,xmm1,xmm3,xmm4)
      }
      else
      {
        //load c and multiply with beta and
        //add to accumulator and store back
        F32_C_BNZ_2(cbuf,rs_c,xmm0,xmm3,xmm4)
      }
    }//betazero

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_1x2F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          BF16_F32_BIAS_LOAD_2BF16_AVX2(xmm0, 0);
        }
        else
        {
          xmm0 = (__m128)_mm_load_sd((const double *)
                ((float*)post_ops_list_temp->op_args1 +
                post_ops_attr.post_op_c_j + (0 * 8)));
        }

        // c[0,0-3]
        xmm4 = _mm_add_ps( xmm4, xmm0 );
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
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm0, 0);
        }
        else
        {
          xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 0 );
        }

        // c[0,0-3]
        xmm4 = _mm_add_ps( xmm4, xmm0 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_1x2F:
    {
      xmm0 = _mm_setzero_ps();

      // c[0,0-3]
      xmm4 = _mm_max_ps( xmm4, xmm0 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_1x2F:
    {
      xmm0 =
        _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
      xmm1 = _mm_setzero_ps();

      // c[0,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_1x2F:
    {
      __m128 dn, x_tanh;
      __m128i q;

      // c[0,0-3]
      GELU_TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_1x2F:
    {
      // c[0,0-3]
      GELU_ERF_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_1x2F:
    {
      xmm0 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      xmm1 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0,0-3]
      CLIP_F32S_SSE(xmm4, xmm0, xmm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_DOWNSCALE_1x2F:
    {
      __m128 selector0 = _mm_setzero_ps();

      __m128 zero_point0 = _mm_setzero_ps();

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
        selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if( is_bf16 == TRUE )
        {
          BF16_F32_ZP_SCALAR_BCAST_SSE( zero_point0 );
        }
        else
        {
          zero_point0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector0 = ( __m128 )_mm_load_sd( (const double*)( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_j + ( 0 * 8 ) ) );
        }
        if( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_2LOAD_SSE(zero_point0,0)
          }
          else
          {
            zero_point0 = ( __m128 )_mm_load_sd( (const double*)( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_j + ( 0 * 8 ) ) );
          }
        }
        //c[0, 0-3]
        F32_SCL_MULRND_SSE(xmm4, selector0, zero_point0);
      }
      else
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_i + 0 ) );
        }
        if( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point0,0)
          }
          else
          {
            zero_point0 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                          post_ops_attr.post_op_c_i + 0 ) );
          }
        }
        //c[0, 0-3]
        F32_SCL_MULRND_SSE(xmm4, selector0, zero_point0);
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_1x2F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
        ( ( post_ops_list_temp->stor_type == NONE ) &&
          ( post_ops_attr.c_stor_type == BF16 ) );

      __m128 scl_fctr1 = _mm_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
        }
      }
      if ( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,0,4);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,0,4);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,0,4);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,0,4);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_1x2F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m128 scl_fctr1 = _mm_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
        }
      }

      if( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,0,4);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,0,4);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,0,4);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,0,4);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_1x2F:
    {
        xmm0 =
          _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
        __m128 z, dn;
        __m128i ex_out;

        // c[0,0-1]
        SWISH_F32_SSE_DEF(xmm4, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_1x2F:
    {
      __m128 dn;
      __m128i q;

      // c[0,0-3]
      TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SIGMOID_1x2F:
    {
        __m128 z, dn;
        __m128i ex_out;

        // c[0,0-1]
        SIGMOID_F32_SSE_DEF(xmm4, xmm1, xmm2, xmm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_1x2F_DISABLE:
    ;

    uint32_t tlsb, rounded, temp[2] = {0};
    int i;
    bfloat16* dest;

    if ( ( post_ops_attr.buf_downscale != NULL ) &&
      ( post_ops_attr.is_last_k == TRUE ) )
    {
      STORE_F32_BF16_2XMM(xmm4, 0, 0)
    }
    else
    {
      _mm_store_sd((double*)cbuf, ( __m128d )xmm4);
    }
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_5x1)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_5x1F_DISABLE,
              &&POST_OPS_BIAS_5x1F,
              &&POST_OPS_RELU_5x1F,
              &&POST_OPS_RELU_SCALE_5x1F,
              &&POST_OPS_GELU_TANH_5x1F,
              &&POST_OPS_GELU_ERF_5x1F,
              &&POST_OPS_CLIP_5x1F,
              &&POST_OPS_DOWNSCALE_5x1F,
              &&POST_OPS_MATRIX_ADD_5x1F,
              &&POST_OPS_SWISH_5x1F,
              &&POST_OPS_MATRIX_MUL_5x1F,
              &&POST_OPS_TANH_5x1F,
              &&POST_OPS_SIGMOID_5x1F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;
    float *_cbuf = NULL;

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


    if ( beta != 0.0 )
    {
      xmm3 = _mm_broadcast_ss(&(beta));

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
        ( post_ops_attr.is_first_k == TRUE ) )
      {
        BF16_F32_C_BNZ_1(0,0,xmm1,xmm3,xmm4)
        BF16_F32_C_BNZ_1(1,0,xmm1,xmm3,xmm5)
        BF16_F32_C_BNZ_1(2,0,xmm1,xmm3,xmm6)
        BF16_F32_C_BNZ_1(3,0,xmm1,xmm3,xmm7)
        BF16_F32_C_BNZ_1(4,0,xmm1,xmm3,xmm8)
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back
        F32_C_BNZ_1(_cbuf,rs_c,xmm1,xmm3,xmm4)
        _cbuf += rs_c;
        F32_C_BNZ_1(_cbuf,rs_c,xmm1,xmm3,xmm5)
        _cbuf += rs_c;
        F32_C_BNZ_1(_cbuf,rs_c,xmm1,xmm3,xmm6)
        _cbuf += rs_c;
        F32_C_BNZ_1(_cbuf,rs_c,xmm1,xmm3,xmm7)
        _cbuf += rs_c;
        F32_C_BNZ_1(_cbuf,rs_c,xmm1,xmm3,xmm8)
      }
    }//betazero

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_5x1F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          BF16_F32_BIAS_LOAD_1BF16_AVX2(xmm0, 0);
        }
        else
        {
          xmm0 = ( __m128 )_mm_load_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 0 * 8 ) );
        }

        // c[0,0-3]
        xmm4 = _mm_add_ps( xmm4, xmm0 );

        // c[1,0-3]
        xmm5 = _mm_add_ps( xmm5, xmm0 );

        // c[2,0-3]
        xmm6 = _mm_add_ps( xmm6, xmm0 );

        // c[3,0-3]
        xmm7 = _mm_add_ps( xmm7, xmm0 );

        // c[4,0-3]
        xmm8 = _mm_add_ps( xmm8, xmm0 );
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
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm0, 0);
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm1, 1);
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm2, 2);
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm3, 3);
        }
        else
        {
          xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 0 );
          xmm1 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 1 );
          xmm2 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 2 );
          xmm3 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 3 );
        }

        // c[0,0-3]
        xmm4 = _mm_add_ps( xmm4, xmm0 );

        // c[1,0-3]
        xmm5 = _mm_add_ps( xmm5, xmm1 );

        // c[2,0-3]
        xmm6 = _mm_add_ps( xmm6, xmm2 );

        // c[3,0-3]
        xmm7 = _mm_add_ps( xmm7, xmm3 );

        if( post_ops_list_temp->stor_type == BF16 )
        {
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm0, 4);
        }
        else
        {
          xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 4 );
        }

        // c[4,0-3]
        xmm8 = _mm_add_ps( xmm8, xmm0 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_5x1F:
    {
      xmm0 = _mm_setzero_ps();

      // c[0,0-3]
      xmm4 = _mm_max_ps( xmm4, xmm0 );

      // c[1,0-3]
      xmm5 = _mm_max_ps( xmm5, xmm0 );

      // c[2,0-3]
      xmm6 = _mm_max_ps( xmm6, xmm0 );

      // c[3,0-3]
      xmm7 = _mm_max_ps( xmm7, xmm0 );

      // c[4,0-3]
      xmm8 = _mm_max_ps( xmm8, xmm0 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_5x1F:
    {
      xmm0 =
        _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
      xmm1 = _mm_setzero_ps();

      // c[0,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

      // c[1,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm5, xmm0, xmm1, xmm2)

      // c[2,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm6, xmm0, xmm1, xmm2)

      // c[3,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm7, xmm0, xmm1, xmm2)

      // c[4,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm8, xmm0, xmm1, xmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_5x1F:
    {
      __m128 dn, x_tanh;
      __m128i q;

      // c[0,0-3]
      GELU_TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      // c[1,0-3]
      GELU_TANH_F32S_SSE(xmm5, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      // c[2,0-3]
      GELU_TANH_F32S_SSE(xmm6, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      // c[3,0-3]
      GELU_TANH_F32S_SSE(xmm7, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      // c[4,0-3]
      GELU_TANH_F32S_SSE(xmm8, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_5x1F:
    {
      // c[0,0-3]
      GELU_ERF_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

      // c[1,0-3]
      GELU_ERF_F32S_SSE(xmm5, xmm0, xmm1, xmm2)

      // c[2,0-3]
      GELU_ERF_F32S_SSE(xmm6, xmm0, xmm1, xmm2)

      // c[3,0-3]
      GELU_ERF_F32S_SSE(xmm7, xmm0, xmm1, xmm2)

      // c[4,0-3]
      GELU_ERF_F32S_SSE(xmm8, xmm0, xmm1, xmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_5x1F:
    {
      xmm0 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      xmm1 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0,0-3]
      CLIP_F32S_SSE(xmm4, xmm0, xmm1)

      // c[1,0-3]
      CLIP_F32S_SSE(xmm5, xmm0, xmm1)

      // c[2,0-3]
      CLIP_F32S_SSE(xmm6, xmm0, xmm1)

      // c[3,0-3]
      CLIP_F32S_SSE(xmm7, xmm0, xmm1)

      // c[4,0-3]
      CLIP_F32S_SSE(xmm8, xmm0, xmm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_DOWNSCALE_5x1F:
    {
      __m128 selector0 = _mm_setzero_ps();
      __m128 selector1 = _mm_setzero_ps();
      __m128 selector2 = _mm_setzero_ps();
      __m128 selector3 = _mm_setzero_ps();
      __m128 selector4 = _mm_setzero_ps();

      __m128 zero_point0 = _mm_setzero_ps();
      __m128 zero_point1 = _mm_setzero_ps();
      __m128 zero_point2 = _mm_setzero_ps();
      __m128 zero_point3 = _mm_setzero_ps();
      __m128 zero_point4 = _mm_setzero_ps();

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
        selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector4 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if( is_bf16 == TRUE )
        {
          BF16_F32_ZP_SCALAR_BCAST_SSE(zero_point0);
          BF16_F32_ZP_SCALAR_BCAST_SSE(zero_point1);
          BF16_F32_ZP_SCALAR_BCAST_SSE(zero_point2);
          BF16_F32_ZP_SCALAR_BCAST_SSE(zero_point3);
          BF16_F32_ZP_SCALAR_BCAST_SSE(zero_point4);
        }
        else
        {
          zero_point0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point4 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector0 = ( __m128 )_mm_load_ss( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_j + ( 0 * 4) );
        }
        if( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_1LOAD_SSE(zero_point0,0)
          }
          else
          {
            zero_point0 = ( __m128 )_mm_load_ss( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_j + ( 0 * 4 ) );
          }
        }
        //c[0, 0-3]
        F32_SCL_MULRND_SSE(xmm4, selector0, zero_point0);

        //c[1, 0-3]
        F32_SCL_MULRND_SSE(xmm5, selector0, zero_point0);

        //c[2, 0-3]
        F32_SCL_MULRND_SSE(xmm6, selector0, zero_point0);

        //c[3, 0-3]
        F32_SCL_MULRND_SSE(xmm7, selector0, zero_point0);

        //c[4, 0-3]
        F32_SCL_MULRND_SSE(xmm8, selector0, zero_point0);
      }
      else
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_i + 0 ) );
          selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_i + 1 ) );
          selector2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_i + 2 ) );
          selector3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_i + 3 ) );
          selector4 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_i + 4 ) );
        }
        if( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point0,0)
            BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point1,1)
            BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point2,2)
            BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point3,3)
            BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point4,4)
          }
          else
          {
            zero_point0 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 1 ) );
            zero_point2 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 2 ) );
            zero_point3 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 3 ) );
            zero_point4 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 4 ) );
          }
        }
        //c[0, 0-3]
        F32_SCL_MULRND_SSE(xmm4, selector0, zero_point0);

        //c[1, 0-3]
        F32_SCL_MULRND_SSE(xmm5, selector1, zero_point1);

        //c[2, 0-3]
        F32_SCL_MULRND_SSE(xmm6, selector2, zero_point2);

        //c[3, 0-3]
        F32_SCL_MULRND_SSE(xmm7, selector3, zero_point3);

        //c[4, 0-3]
        F32_SCL_MULRND_SSE(xmm8, selector4, zero_point4);
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_5x1F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m128 scl_fctr1 = _mm_setzero_ps();
      __m128 scl_fctr2 = _mm_setzero_ps();
      __m128 scl_fctr3 = _mm_setzero_ps();
      __m128 scl_fctr4 = _mm_setzero_ps();
      __m128 scl_fctr5 = _mm_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr4 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr5 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
          scl_fctr4 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 3 ) );
          scl_fctr5 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 4 ) );
        }
      }
      if( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,1,5);

          // c[2:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,2,6);

          // c[3:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,3,7);

          // c[4:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,4,8);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr2,1,5);

          // c[2:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr3,2,6);

          // c[3:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr4,3,7);

          // c[4:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr5,4,8);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,1,5);

          // c[2:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,2,6);

          // c[3:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,3,7);

          // c[4:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,4,8);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr2,1,5);

          // c[2:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr3,2,6);

          // c[3:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr4,3,7);

          // c[4:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr5,4,8);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_5x1F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m128 scl_fctr1 = _mm_setzero_ps();
      __m128 scl_fctr2 = _mm_setzero_ps();
      __m128 scl_fctr3 = _mm_setzero_ps();
      __m128 scl_fctr4 = _mm_setzero_ps();
      __m128 scl_fctr5 = _mm_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr4 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr5 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
          scl_fctr4 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 3 ) );
          scl_fctr5 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 4 ) );
        }
      }

      if( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,1,5);

          // c[2:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,2,6);

          // c[3:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,3,7);

          // c[4:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,4,8);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr2,1,5);

          // c[2:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr3,2,6);

          // c[3:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr4,3,7);

          // c[4:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr5,4,8);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,1,5);

          // c[2:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,2,6);

          // c[3:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,3,7);

          // c[4:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,4,8);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr2,1,5);

          // c[2:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr3,2,6);

          // c[3:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr4,3,7);

          // c[4:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr5,4,8);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_5x1F:
    {
        xmm0 =
          _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
        __m128 z, dn;
        __m128i ex_out;

        // c[0,0-0]
        SWISH_F32_SSE_DEF(xmm4, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[1,0-0]
        SWISH_F32_SSE_DEF(xmm5, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[2,0-0]
        SWISH_F32_SSE_DEF(xmm6, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[3,0-0]
        SWISH_F32_SSE_DEF(xmm7, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[4,0-0]
        SWISH_F32_SSE_DEF(xmm8, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_5x1F:
    {
      __m128 dn;
      __m128i q;

      // c[0,0-3]
      TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, q)

      // c[1,0-3]
      TANH_F32S_SSE(xmm5, xmm0, xmm1, xmm2, xmm3, dn, q)

      // c[2,0-3]
      TANH_F32S_SSE(xmm6, xmm0, xmm1, xmm2, xmm3, dn, q)

      // c[3,0-3]
      TANH_F32S_SSE(xmm7, xmm0, xmm1, xmm2, xmm3, dn, q)

      // c[4,0-3]
      TANH_F32S_SSE(xmm8, xmm0, xmm1, xmm2, xmm3, dn, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SIGMOID_5x1F:
    {
        __m128 z, dn;
        __m128i ex_out;

        // c[0,0-0]
        SIGMOID_F32_SSE_DEF(xmm4, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[1,0-0]
        SIGMOID_F32_SSE_DEF(xmm5, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[2,0-0]
        SIGMOID_F32_SSE_DEF(xmm6, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[3,0-0]
        SIGMOID_F32_SSE_DEF(xmm7, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[4,0-0]
        SIGMOID_F32_SSE_DEF(xmm8, xmm1, xmm2, xmm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_5x1F_DISABLE:
    ;

    uint32_t tlsb, rounded, temp[1] = {0};
    int i;
    bfloat16* dest;

    if ( ( post_ops_attr.buf_downscale != NULL ) &&
      ( post_ops_attr.is_last_k == TRUE ) )
    {
      STORE_F32_BF16_1XMM(xmm4, 0, 0)
      STORE_F32_BF16_1XMM(xmm5, 1, 0)
      STORE_F32_BF16_1XMM(xmm6, 2, 0)
      STORE_F32_BF16_1XMM(xmm7, 3, 0)
      STORE_F32_BF16_1XMM(xmm8, 4, 0)
    }
    else
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
    }
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_4x1)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_4x1F_DISABLE,
              &&POST_OPS_BIAS_4x1F,
              &&POST_OPS_RELU_4x1F,
              &&POST_OPS_RELU_SCALE_4x1F,
              &&POST_OPS_GELU_TANH_4x1F,
              &&POST_OPS_GELU_ERF_4x1F,
              &&POST_OPS_CLIP_4x1F,
              &&POST_OPS_DOWNSCALE_4x1F,
              &&POST_OPS_MATRIX_ADD_4x1F,
              &&POST_OPS_SWISH_4x1F,
              &&POST_OPS_MATRIX_MUL_4x1F,
              &&POST_OPS_TANH_4x1F,
              &&POST_OPS_SIGMOID_4x1F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;
    float *_cbuf = NULL;

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


    if ( beta != 0.0 )
    {
      xmm3 = _mm_broadcast_ss(&(beta));

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
        ( post_ops_attr.is_first_k == TRUE ) )
      {
        BF16_F32_C_BNZ_1(0,0,xmm1,xmm3,xmm4)
        BF16_F32_C_BNZ_1(1,0,xmm1,xmm3,xmm5)
        BF16_F32_C_BNZ_1(2,0,xmm1,xmm3,xmm6)
        BF16_F32_C_BNZ_1(3,0,xmm1,xmm3,xmm7)
      }
      else
      {
         _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back
        F32_C_BNZ_1(_cbuf,rs_c,xmm1,xmm3,xmm4)
        _cbuf += rs_c;
        F32_C_BNZ_1(_cbuf,rs_c,xmm1,xmm3,xmm5)
        _cbuf += rs_c;
        F32_C_BNZ_1(_cbuf,rs_c,xmm1,xmm3,xmm6)
        _cbuf += rs_c;
        F32_C_BNZ_1(_cbuf,rs_c,xmm1,xmm3,xmm7)
      }
    }//betazero

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_4x1F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          BF16_F32_BIAS_LOAD_1BF16_AVX2(xmm0, 0);
        }
        else
        {
          xmm0 = ( __m128 )_mm_load_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 0 * 8 ) );
        }

        // c[0,0-3]
        xmm4 = _mm_add_ps( xmm4, xmm0 );

        // c[1,0-3]
        xmm5 = _mm_add_ps( xmm5, xmm0 );

        // c[2,0-3]
        xmm6 = _mm_add_ps( xmm6, xmm0 );

        // c[3,0-3]
        xmm7 = _mm_add_ps( xmm7, xmm0 );
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
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm0, 0);
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm1, 1);
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm2, 2);
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm3, 3);
        }
        else
        {
          xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 0 );
          xmm1 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 1 );
          xmm2 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 2 );
          xmm3 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 3 );
        }

        // c[0,0-3]
        xmm4 = _mm_add_ps( xmm4, xmm0 );

        // c[1,0-3]
        xmm5 = _mm_add_ps( xmm5, xmm1 );

        // c[2,0-3]
        xmm6 = _mm_add_ps( xmm6, xmm2 );

        // c[3,0-3]
        xmm7 = _mm_add_ps( xmm7, xmm3 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_4x1F:
    {
      xmm0 = _mm_setzero_ps();

      // c[0,0-3]
      xmm4 = _mm_max_ps( xmm4, xmm0 );

      // c[1,0-3]
      xmm5 = _mm_max_ps( xmm5, xmm0 );

      // c[2,0-3]
      xmm6 = _mm_max_ps( xmm6, xmm0 );

      // c[3,0-3]
      xmm7 = _mm_max_ps( xmm7, xmm0 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_4x1F:
    {
      xmm0 =
        _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
      xmm1 = _mm_setzero_ps();

      // c[0,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

      // c[1,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm5, xmm0, xmm1, xmm2)

      // c[2,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm6, xmm0, xmm1, xmm2)

      // c[3,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm7, xmm0, xmm1, xmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_4x1F:
    {
      __m128 dn, x_tanh;
      __m128i q;

      // c[0,0-3]
      GELU_TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      // c[1,0-3]
      GELU_TANH_F32S_SSE(xmm5, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      // c[2,0-3]
      GELU_TANH_F32S_SSE(xmm6, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      // c[3,0-3]
      GELU_TANH_F32S_SSE(xmm7, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_4x1F:
    {
      // c[0,0-3]
      GELU_ERF_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

      // c[1,0-3]
      GELU_ERF_F32S_SSE(xmm5, xmm0, xmm1, xmm2)

      // c[2,0-3]
      GELU_ERF_F32S_SSE(xmm6, xmm0, xmm1, xmm2)

      // c[3,0-3]
      GELU_ERF_F32S_SSE(xmm7, xmm0, xmm1, xmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_4x1F:
    {
      xmm0 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      xmm1 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0,0-3]
      CLIP_F32S_SSE(xmm4, xmm0, xmm1)

      // c[1,0-3]
      CLIP_F32S_SSE(xmm5, xmm0, xmm1)

      // c[2,0-3]
      CLIP_F32S_SSE(xmm6, xmm0, xmm1)

      // c[3,0-3]
      CLIP_F32S_SSE(xmm7, xmm0, xmm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_DOWNSCALE_4x1F:
    {
      __m128 selector0 = _mm_setzero_ps();
      __m128 selector1 = _mm_setzero_ps();
      __m128 selector2 = _mm_setzero_ps();
      __m128 selector3 = _mm_setzero_ps();

      __m128 zero_point0 = _mm_setzero_ps();
      __m128 zero_point1 = _mm_setzero_ps();
      __m128 zero_point2 = _mm_setzero_ps();
      __m128 zero_point3 = _mm_setzero_ps();

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
        selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if( is_bf16 == TRUE )
        {
          BF16_F32_ZP_SCALAR_BCAST_SSE(zero_point0);
          BF16_F32_ZP_SCALAR_BCAST_SSE(zero_point1);
          BF16_F32_ZP_SCALAR_BCAST_SSE(zero_point2);
          BF16_F32_ZP_SCALAR_BCAST_SSE(zero_point3);
        }
        else
        {
          zero_point0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector0 = ( __m128 )_mm_load_ss( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_j + ( 0 * 4) );
        }
        if( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_1LOAD_SSE(zero_point0,0)
          }
          else
          {
            zero_point0 = ( __m128 )_mm_load_ss( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_j + ( 0 * 4 ) );
          }
        }
        //c[0, 0-3]
        F32_SCL_MULRND_SSE(xmm4, selector0, zero_point0);

        //c[1, 0-3]
        F32_SCL_MULRND_SSE(xmm5, selector0, zero_point0);

        //c[2, 0-3]
        F32_SCL_MULRND_SSE(xmm6, selector0, zero_point0);

        //c[3, 0-3]
        F32_SCL_MULRND_SSE(xmm7, selector0, zero_point0);;
      }
      else
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_i + 0 ) );
          selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_i + 1 ) );
          selector2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_i + 2 ) );
          selector3 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_i + 3 ) );
        }
        if( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point0,0)
            BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point1,1)
            BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point2,2)
            BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point3,3)
          }
          else
          {
            zero_point0 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 1 ) );
            zero_point2 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 2 ) );
            zero_point3 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 3 ) );
          }
        }
        //c[0, 0-3]
        F32_SCL_MULRND_SSE(xmm4, selector0, zero_point0);

        //c[1, 0-3]
        F32_SCL_MULRND_SSE(xmm5, selector1, zero_point1);

        //c[2, 0-3]
        F32_SCL_MULRND_SSE(xmm6, selector2, zero_point2);

        //c[3, 0-3]
        F32_SCL_MULRND_SSE(xmm7, selector3, zero_point3);
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_4x1F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m128 scl_fctr1 = _mm_setzero_ps();
      __m128 scl_fctr2 = _mm_setzero_ps();
      __m128 scl_fctr3 = _mm_setzero_ps();
      __m128 scl_fctr4 = _mm_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr4 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
          scl_fctr4 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 3 ) );
        }
      }
      if( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,1,5);

          // c[2:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,2,6);

          // c[3:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,3,7);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr2,1,5);

          // c[2:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr3,2,6);

          // c[3:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr4,3,7);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,1,5);

          // c[2:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,2,6);

          // c[3:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,3,7);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr2,1,5);

          // c[2:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr3,2,6);

          // c[3:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr4,3,7);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_4x1F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m128 scl_fctr1 = _mm_setzero_ps();
      __m128 scl_fctr2 = _mm_setzero_ps();
      __m128 scl_fctr3 = _mm_setzero_ps();
      __m128 scl_fctr4 = _mm_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr4 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
          scl_fctr4 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 3 ) );
        }
      }

      if( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,1,5);

          // c[2:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,2,6);

          // c[3:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,3,7);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr2,1,5);

          // c[2:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr3,2,6);

          // c[3:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr4,3,7);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,1,5);

          // c[2:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,2,6);

          // c[3:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,3,7);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr2,1,5);

          // c[2:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr3,2,6);

          // c[3:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr4,3,7);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_4x1F:
    {
        xmm0 =
          _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
        __m128 z, dn;
        __m128i ex_out;

        // c[0,0-0]
        SWISH_F32_SSE_DEF(xmm4, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[1,0-0]
        SWISH_F32_SSE_DEF(xmm5, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[2,0-0]
        SWISH_F32_SSE_DEF(xmm6, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[3,0-0]
        SWISH_F32_SSE_DEF(xmm7, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_4x1F:
    {
      __m128 dn;
      __m128i q;

      // c[0,0-3]
      TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, q)

      // c[1,0-3]
      TANH_F32S_SSE(xmm5, xmm0, xmm1, xmm2, xmm3, dn, q)

      // c[2,0-3]
      TANH_F32S_SSE(xmm6, xmm0, xmm1, xmm2, xmm3, dn, q)

      // c[3,0-3]
      TANH_F32S_SSE(xmm7, xmm0, xmm1, xmm2, xmm3, dn, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SIGMOID_4x1F:
    {
        __m128 z, dn;
        __m128i ex_out;

        // c[0,0-0]
        SIGMOID_F32_SSE_DEF(xmm4, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[1,0-0]
        SIGMOID_F32_SSE_DEF(xmm5, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[2,0-0]
        SIGMOID_F32_SSE_DEF(xmm6, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[3,0-0]
        SIGMOID_F32_SSE_DEF(xmm7, xmm1, xmm2, xmm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_4x1F_DISABLE:
    ;

    uint32_t tlsb, rounded, temp[1] = {0};
    int i;
    bfloat16* dest;

    if ( ( post_ops_attr.buf_downscale != NULL ) &&
      ( post_ops_attr.is_last_k == TRUE ) )
    {
      STORE_F32_BF16_1XMM(xmm4, 0, 0)
      STORE_F32_BF16_1XMM(xmm5, 1, 0)
      STORE_F32_BF16_1XMM(xmm6, 2, 0)
      STORE_F32_BF16_1XMM(xmm7, 3, 0)
    }
    else
    {
      _mm_store_ss(cbuf, xmm4);
      cbuf += rs_c;
      _mm_store_ss(cbuf, xmm5);
      cbuf += rs_c;
      _mm_store_ss(cbuf, xmm6);
      cbuf += rs_c;
      _mm_store_ss(cbuf, xmm7);
    }
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_3x1)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_3x1F_DISABLE,
              &&POST_OPS_BIAS_3x1F,
              &&POST_OPS_RELU_3x1F,
              &&POST_OPS_RELU_SCALE_3x1F,
              &&POST_OPS_GELU_TANH_3x1F,
              &&POST_OPS_GELU_ERF_3x1F,
              &&POST_OPS_CLIP_3x1F,
              &&POST_OPS_DOWNSCALE_3x1F,
              &&POST_OPS_MATRIX_ADD_3x1F,
              &&POST_OPS_SWISH_3x1F,
              &&POST_OPS_MATRIX_MUL_3x1F,
              &&POST_OPS_TANH_3x1F,
              &&POST_OPS_SIGMOID_3x1F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;
    float *_cbuf = NULL;

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


    if ( beta != 0.0 )
    {
      xmm3 = _mm_broadcast_ss(&(beta));
      if ( ( post_ops_attr.buf_downscale != NULL ) &&
        ( post_ops_attr.is_first_k == TRUE ) )
      {
        BF16_F32_C_BNZ_1(0,0,xmm1,xmm3,xmm4)
        BF16_F32_C_BNZ_1(1,0,xmm1,xmm3,xmm5)
        BF16_F32_C_BNZ_1(2,0,xmm1,xmm3,xmm6)
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back
        F32_C_BNZ_1(_cbuf,rs_c,xmm1,xmm3,xmm4)
        _cbuf += rs_c;
        F32_C_BNZ_1(_cbuf,rs_c,xmm1,xmm3,xmm5)
        _cbuf += rs_c;
        F32_C_BNZ_1(_cbuf,rs_c,xmm1,xmm3,xmm6)
      }
    }//betazero

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_3x1F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          BF16_F32_BIAS_LOAD_1BF16_AVX2(xmm0, 0);
        }
        else
        {
          xmm0 = ( __m128 )_mm_load_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 0 * 8 ) );
        }

        // c[0,0-3]
        xmm4 = _mm_add_ps( xmm4, xmm0 );

        // c[1,0-3]
        xmm5 = _mm_add_ps( xmm5, xmm0 );

        // c[2,0-3]
        xmm6 = _mm_add_ps( xmm6, xmm0 );
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
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm0, 0);
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm1, 1);
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm2, 2);
        }
        else
        {
          xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 0 );
          xmm1 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 1 );
          xmm2 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 2 );
        }

        // c[0,0-3]
        xmm4 = _mm_add_ps( xmm4, xmm0 );

        // c[1,0-3]
        xmm5 = _mm_add_ps( xmm5, xmm1 );

        // c[2,0-3]
        xmm6 = _mm_add_ps( xmm6, xmm2 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_3x1F:
    {
      xmm0 = _mm_setzero_ps();

      // c[0,0-3]
      xmm4 = _mm_max_ps( xmm4, xmm0 );

      // c[1,0-3]
      xmm5 = _mm_max_ps( xmm5, xmm0 );

      // c[2,0-3]
      xmm6 = _mm_max_ps( xmm6, xmm0 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_3x1F:
    {
      xmm0 =
        _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
      xmm1 = _mm_setzero_ps();

      // c[0,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

      // c[1,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm5, xmm0, xmm1, xmm2)

      // c[2,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm6, xmm0, xmm1, xmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_3x1F:
    {
      __m128 dn, x_tanh;
      __m128i q;

      // c[0,0-3]
      GELU_TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      // c[1,0-3]
      GELU_TANH_F32S_SSE(xmm5, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      // c[2,0-3]
      GELU_TANH_F32S_SSE(xmm6, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_3x1F:
    {
      // c[0,0-3]
      GELU_ERF_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

      // c[1,0-3]
      GELU_ERF_F32S_SSE(xmm5, xmm0, xmm1, xmm2)

      // c[2,0-3]
      GELU_ERF_F32S_SSE(xmm6, xmm0, xmm1, xmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_3x1F:
    {
      xmm0 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      xmm1 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0,0-3]
      CLIP_F32S_SSE(xmm4, xmm0, xmm1)

      // c[1,0-3]
      CLIP_F32S_SSE(xmm5, xmm0, xmm1)

      // c[2,0-3]
      CLIP_F32S_SSE(xmm6, xmm0, xmm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_DOWNSCALE_3x1F:
    {
      __m128 selector0 = _mm_setzero_ps();
      __m128 selector1 = _mm_setzero_ps();
      __m128 selector2 = _mm_setzero_ps();

      __m128 zero_point0 = _mm_setzero_ps();
      __m128 zero_point1 = _mm_setzero_ps();
      __m128 zero_point2 = _mm_setzero_ps();

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
        selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if( is_bf16 == TRUE )
        {
          BF16_F32_ZP_SCALAR_BCAST_SSE(zero_point0);
          BF16_F32_ZP_SCALAR_BCAST_SSE(zero_point1);
          BF16_F32_ZP_SCALAR_BCAST_SSE(zero_point2);
        }
        else
        {
          zero_point0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector0 = ( __m128 )_mm_load_ss( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_j + ( 0 * 4) );
        }
        if( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_1LOAD_SSE(zero_point0,0)
          }
          else
          {
            zero_point0 = ( __m128 )_mm_load_ss( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_j + ( 0 * 4 ) );
          }
        }
        //c[0, 0-3]
        F32_SCL_MULRND_SSE(xmm4, selector0, zero_point0);

        //c[1, 0-3]
        F32_SCL_MULRND_SSE(xmm5, selector0, zero_point0);

        //c[2, 0-3]
        F32_SCL_MULRND_SSE(xmm6, selector0, zero_point0);
      }
      else
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_i + 0 ) );
          selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_i + 1 ) );
          selector2 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_i + 2 ) );
        }
        if( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point0,0)
            BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point1,1)
            BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point2,2)
          }
          else
          {
            zero_point0 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 1 ) );
            zero_point2 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 2 ) );
          }
        }
        //c[0, 0-3]
        F32_SCL_MULRND_SSE(xmm4, selector0, zero_point0);

        //c[1, 0-3]
        F32_SCL_MULRND_SSE(xmm5, selector1, zero_point1);

        //c[2, 0-3]
        F32_SCL_MULRND_SSE(xmm6, selector2, zero_point2);
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_3x1F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m128 scl_fctr1 = _mm_setzero_ps();
      __m128 scl_fctr2 = _mm_setzero_ps();
      __m128 scl_fctr3 = _mm_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
        }
      }
      if( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,1,5);

          // c[2:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,2,6);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr2,1,5);

          // c[2:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr3,2,6);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,1,5);

          // c[2:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,2,6);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr2,1,5);

          // c[2:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr3,2,6);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_3x1F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m128 scl_fctr1 = _mm_setzero_ps();
      __m128 scl_fctr2 = _mm_setzero_ps();
      __m128 scl_fctr3 = _mm_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr3 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
          scl_fctr3 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 2 ) );
        }
      }

      if( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,1,5);

          // c[2:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,2,6);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr2,1,5);

          // c[2:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr3,2,6);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,1,5);

          // c[2:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,2,6);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr2,1,5);

          // c[2:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr3,2,6);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_3x1F:
    {
        xmm0 =
          _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
        __m128 z, dn;
        __m128i ex_out;

        // c[0,0-0]
        SWISH_F32_SSE_DEF(xmm4, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[1,0-0]
        SWISH_F32_SSE_DEF(xmm5, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[2,0-0]
        SWISH_F32_SSE_DEF(xmm6, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_3x1F:
    {
      __m128 dn;
      __m128i q;

      // c[0,0-3]
      TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, q)

      // c[1,0-3]
      TANH_F32S_SSE(xmm5, xmm0, xmm1, xmm2, xmm3, dn, q)

      // c[2,0-3]
      TANH_F32S_SSE(xmm6, xmm0, xmm1, xmm2, xmm3, dn, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SIGMOID_3x1F:
    {
        __m128 z, dn;
        __m128i ex_out;

        // c[0,0-0]
        SIGMOID_F32_SSE_DEF(xmm4, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[1,0-0]
        SIGMOID_F32_SSE_DEF(xmm5, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[2,0-0]
        SIGMOID_F32_SSE_DEF(xmm6, xmm1, xmm2, xmm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_3x1F_DISABLE:
    ;

    uint32_t tlsb, rounded, temp[1] = {0};
    int i;
    bfloat16* dest;

    if ( ( post_ops_attr.buf_downscale != NULL ) &&
      ( post_ops_attr.is_last_k == TRUE ) )
    {
      STORE_F32_BF16_1XMM(xmm4, 0, 0)
      STORE_F32_BF16_1XMM(xmm5, 1, 0)
      STORE_F32_BF16_1XMM(xmm6, 2, 0)
    }
    else
    {
      _mm_store_ss(cbuf, xmm4);
      cbuf += rs_c;
      _mm_store_ss(cbuf, xmm5);
      cbuf += rs_c;
      _mm_store_ss(cbuf, xmm6);
    }
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_2x1)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_2x1F_DISABLE,
              &&POST_OPS_BIAS_2x1F,
              &&POST_OPS_RELU_2x1F,
              &&POST_OPS_RELU_SCALE_2x1F,
              &&POST_OPS_GELU_TANH_2x1F,
              &&POST_OPS_GELU_ERF_2x1F,
              &&POST_OPS_CLIP_2x1F,
              &&POST_OPS_DOWNSCALE_2x1F,
              &&POST_OPS_MATRIX_ADD_2x1F,
              &&POST_OPS_SWISH_2x1F,
              &&POST_OPS_MATRIX_MUL_2x1F,
              &&POST_OPS_TANH_2x1F,
              &&POST_OPS_SIGMOID_2x1F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;
    float *_cbuf = NULL;

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


    if ( beta != 0.0 )
    {
      xmm3 = _mm_broadcast_ss(&(beta));

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
        ( post_ops_attr.is_first_k == TRUE ) )
      {
        BF16_F32_C_BNZ_1(0,0,xmm1,xmm3,xmm4)
        BF16_F32_C_BNZ_1(1,0,xmm1,xmm3,xmm5)
      }
      else
      {
        _cbuf = cbuf;
        //load c and multiply with beta and
        //add to accumulator and store back
        F32_C_BNZ_1(_cbuf,rs_c,xmm0,xmm3,xmm4)
        _cbuf += rs_c;
        F32_C_BNZ_1(_cbuf,rs_c,xmm0,xmm3,xmm5)
      }
    }//betazero

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_2x1F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          BF16_F32_BIAS_LOAD_1BF16_AVX2(xmm0, 0);
        }
        else
        {
          xmm0 = ( __m128 )_mm_load_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 0 * 8 ) );
        }

        // c[0,0-3]
        xmm4 = _mm_add_ps( xmm4, xmm0 );

        // c[1,0-3]
        xmm5 = _mm_add_ps( xmm5, xmm0 );
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
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm0, 0);
          BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm1, 1);
        }
        else
        {
          xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 0 );
          xmm1 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 1 );
        }

        // c[0,0-3]
        xmm4 = _mm_add_ps( xmm4, xmm0 );

        // c[1,0-3]
        xmm5 = _mm_add_ps( xmm5, xmm1 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_2x1F:
    {
      xmm0 = _mm_setzero_ps();

      // c[0,0-3]
      xmm4 = _mm_max_ps( xmm4, xmm0 );

      // c[1,0-3]
      xmm5 = _mm_max_ps( xmm5, xmm0 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_2x1F:
    {
      xmm0 =
        _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
      xmm1 = _mm_setzero_ps();

      // c[0,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

      // c[1,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm5, xmm0, xmm1, xmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_2x1F:
    {
      __m128 dn, x_tanh;
      __m128i q;

      // c[0,0-3]
      GELU_TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      // c[1,0-3]
      GELU_TANH_F32S_SSE(xmm5, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_2x1F:
    {
      // c[0,0-3]
      GELU_ERF_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

      // c[1,0-3]
      GELU_ERF_F32S_SSE(xmm5, xmm0, xmm1, xmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_2x1F:
    {
      xmm0 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      xmm1 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0,0-3]
      CLIP_F32S_SSE(xmm4, xmm0, xmm1)

      // c[1,0-3]
      CLIP_F32S_SSE(xmm5, xmm0, xmm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_DOWNSCALE_2x1F:
    {
      __m128 selector0 = _mm_setzero_ps();
      __m128 selector1 = _mm_setzero_ps();


      __m128 zero_point0 = _mm_setzero_ps();
      __m128 zero_point1 = _mm_setzero_ps();

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
        selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if( is_bf16 == TRUE )
        {
          BF16_F32_ZP_SCALAR_BCAST_SSE(zero_point0);
          BF16_F32_ZP_SCALAR_BCAST_SSE(zero_point1);
        }
        else
        {
          zero_point0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
          zero_point1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector0 = ( __m128 )_mm_load_ss( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_j + ( 0 * 4) );
        }
        if( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_1LOAD_SSE(zero_point0,0)
          }
          else
          {
            zero_point0 = ( __m128 )_mm_load_ss( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_j + ( 0 * 4 ) );
          }
        }
        //c[0, 0-3]
        F32_SCL_MULRND_SSE(xmm4, selector0, zero_point0);

        //c[1, 0-3]
        F32_SCL_MULRND_SSE(xmm5, selector0, zero_point0);
      }
      else
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_i + 0 ) );
          selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_i + 1 ) );
        }
        if( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point0,0)
            BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point1,1)
          }
          else
          {
            zero_point0 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 0 ) );
            zero_point1 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 1 ) );
          }
        }
        //c[0, 0-3]
        F32_SCL_MULRND_SSE(xmm4, selector0, zero_point0);

        //c[1, 0-3]
        F32_SCL_MULRND_SSE(xmm5, selector1, zero_point1);
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_2x1F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m128 scl_fctr1 = _mm_setzero_ps();
      __m128 scl_fctr2 = _mm_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
        }
      }
      if( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,1,5);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr2,1,5);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,1,5);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr2,1,5);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_2x1F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m128 scl_fctr1 = _mm_setzero_ps();
      __m128 scl_fctr2 = _mm_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
        scl_fctr2 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
          scl_fctr2 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 1 ) );
        }
      }

      if( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,1,5);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr2,1,5);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,1,5);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,0,4);

          // c[1:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr2,1,5);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_2x1F:
    {
        xmm0 =
          _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
        __m128 z, dn;
        __m128i ex_out;

        // c[0,0-0]
        SWISH_F32_SSE_DEF(xmm4, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[1,0-0]
        SWISH_F32_SSE_DEF(xmm5, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_2x1F:
    {
      __m128 dn;
      __m128i q;

      // c[0,0-3]
      TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, q)

      // c[1,0-3]
      TANH_F32S_SSE(xmm5, xmm0, xmm1, xmm2, xmm3, dn, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SIGMOID_2x1F:
    {
        __m128 z, dn;
        __m128i ex_out;

        // c[0,0-0]
        SIGMOID_F32_SSE_DEF(xmm4, xmm1, xmm2, xmm3, z, dn, ex_out)

        // c[1,0-0]
        SIGMOID_F32_SSE_DEF(xmm5, xmm1, xmm2, xmm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_2x1F_DISABLE:
    ;

    uint32_t tlsb, rounded, temp[1] = {0};
    int i;
    bfloat16* dest;

    if ( ( post_ops_attr.buf_downscale != NULL ) &&
      ( post_ops_attr.is_last_k == TRUE ) )
    {
      STORE_F32_BF16_1XMM(xmm4, 0, 0)
      STORE_F32_BF16_1XMM(xmm5, 1, 0)
    }
    else
    {
      _mm_store_ss(cbuf, xmm4);
      cbuf += rs_c;
      _mm_store_ss(cbuf, xmm5);
    }
}

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_1x1)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_1x1F_DISABLE,
              &&POST_OPS_BIAS_1x1F,
              &&POST_OPS_RELU_1x1F,
              &&POST_OPS_RELU_SCALE_1x1F,
              &&POST_OPS_GELU_TANH_1x1F,
              &&POST_OPS_GELU_ERF_1x1F,
              &&POST_OPS_CLIP_1x1F,
              &&POST_OPS_DOWNSCALE_1x1F,
              &&POST_OPS_MATRIX_ADD_1x1F,
              &&POST_OPS_SWISH_1x1F,
              &&POST_OPS_MATRIX_MUL_1x1F,
              &&POST_OPS_TANH_1x1F,
              &&POST_OPS_SIGMOID_1x1F

            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    float *abuf = (float *)a;
    float *bbuf = (float *)b;
    float *cbuf = (float *)c;

    /*Declare the registers*/
    __m128 xmm0, xmm1, xmm2, xmm3, xmm4;

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


    if ( beta != 0.0 )
    {
      xmm3 = _mm_broadcast_ss(&(beta));
      if ( ( post_ops_attr.buf_downscale != NULL ) &&
        ( post_ops_attr.is_first_k == TRUE ) )
      {
        BF16_F32_C_BNZ_1(0,0,xmm1,xmm3,xmm4)
      }
      else
      {
        //load c and multiply with beta and
        //add to accumulator and store back
        F32_C_BNZ_1(cbuf,rs_c,xmm0,xmm3,xmm4)
      }
    }//betazero

    // Post Ops
    lpgemm_post_op* post_ops_list_temp = post_ops_list;
    POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_1x1F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          BF16_F32_BIAS_LOAD_1BF16_AVX2(xmm0, 0);
        }
        else
        {
          xmm0 = ( __m128 )_mm_load_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 0 * 8 ) );
        }

        // c[0,0-3]
        xmm4 = _mm_add_ps( xmm4, xmm0 );
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
          BF16_F32_ZP_VECTOR_BCAST_SSE(xmm0, 0);
        }
        else
        {
          xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 0 );
        }

        // c[0,0-3]
        xmm4 = _mm_add_ps( xmm4, xmm0 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_1x1F:
    {
      xmm0 = _mm_setzero_ps();

      // c[0,0-3]
      xmm4 = _mm_max_ps( xmm4, xmm0 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_1x1F:
    {
      xmm0 =
        _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
      xmm1 = _mm_setzero_ps();

      // c[0,0-3]
      RELU_SCALE_OP_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_1x1F:
    {
      __m128 dn, x_tanh;
      __m128i q;

      // c[0,0-3]
      GELU_TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_1x1F:
    {
      // c[0,0-3]
      GELU_ERF_F32S_SSE(xmm4, xmm0, xmm1, xmm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_1x1F:
    {
      xmm0 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      xmm1 = _mm_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0,0-3]
      CLIP_F32S_SSE(xmm4, xmm0, xmm1)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_DOWNSCALE_1x1F:
    {
      __m128 selector0 = _mm_setzero_ps();

      __m128 zero_point0 = _mm_setzero_ps();

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
        selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
      {
        if( is_bf16 == TRUE )
        {
          BF16_F32_ZP_SCALAR_BCAST_SSE(zero_point0);
        }
        else
        {
          zero_point0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
        }
      }
      if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector0 = ( __m128 )_mm_load_ss( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_j + ( 0 * 4) );
        }
        if( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_1LOAD_SSE(zero_point0,0)
          }
          else
          {
            zero_point0 = ( __m128 )_mm_load_ss( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_j + ( 0 * 4 ) );
          }
        }
        //c[0, 0-3]
        F32_SCL_MULRND_SSE(xmm4, selector0, zero_point0);
      }
      else
      {
        if( post_ops_list_temp->scale_factor_len > 1 )
        {
          selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                          post_ops_attr.post_op_c_i + 0 ) );
        }
        if( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
        {
          if( is_bf16 == TRUE )
          {
            BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point0,0)
          }
          else
          {
            zero_point0 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 0 ) );
          }
        }
        //c[0, 0-3]
        F32_SCL_MULRND_SSE(xmm4, selector0, zero_point0);
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_ADD_1x1F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m128 scl_fctr1 = _mm_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
        }
      }
      if( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,0,4);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,0,4);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,0,4);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,0,4);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_MATRIX_MUL_1x1F:
    {
      dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

      bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
          ( ( post_ops_list_temp->stor_type == NONE ) &&
            ( post_ops_attr.c_stor_type == BF16 ) );

      __m128 scl_fctr1 = _mm_setzero_ps();

      // Even though different registers are used for scalar in column and
      // row major case, all those registers will contain the same value.
      if ( post_ops_list_temp->scale_factor_len == 1 )
      {
        scl_fctr1 =
          _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
      }
      else
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          scl_fctr1 =
            _mm_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_j + ( 0 * 16 ) );
        }
        else
        {
          scl_fctr1 =
            _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                post_ops_attr.post_op_c_i + 0 ) );
        }
      }

      if( is_bf16 == TRUE )
      {
        bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,0,4);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,0,4);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,0,4);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,0,4);
        }
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SWISH_1x1F:
    {
        xmm0 =
          _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
        __m128 z, dn;
        __m128i ex_out;

        // c[0,0-0]
        SWISH_F32_SSE_DEF(xmm4, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_1x1F:
    {
      __m128 dn;
      __m128i q;

      // c[0,0-3]
      TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SIGMOID_1x1F:
    {
        __m128 z, dn;
        __m128i ex_out;

        // c[0,0-0]
        SIGMOID_F32_SSE_DEF(xmm4, xmm1, xmm2, xmm3, z, dn, ex_out)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_1x1F_DISABLE:
    ;

    uint32_t tlsb, rounded, temp[1] = {0};
    int i;
    bfloat16* dest;

    if ( ( post_ops_attr.buf_downscale != NULL ) &&
      ( post_ops_attr.is_last_k == TRUE ) )
    {
      STORE_F32_BF16_1XMM(xmm4, 0, 0)
    }
    else
    {
      _mm_store_ss(cbuf, xmm4);
    }
}
#endif
