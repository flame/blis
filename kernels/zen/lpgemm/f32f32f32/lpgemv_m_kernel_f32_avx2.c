/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

  Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

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

void lpgemv_m_one_f32f32f32of32_avx2_LT16
(
	const dim_t           n0,
	const dim_t           k,
	const float          *a,
	const dim_t           rs_a,
	const dim_t           cs_a,
	const AOCL_MEMORY_TAG mtag_a,
	const float          *b,
	dim_t                 rs_b,
	const dim_t           cs_b,
	const AOCL_MEMORY_TAG mtag_b,
	float                *c,
	const dim_t           rs_c,
	const dim_t           jr,
	const float          alpha,
	const float          beta,
	dim_t                 NR,
	const dim_t           KC,
	const dim_t           n_sub_updated,
	const dim_t           jc_cur_loop_rem,
	lpgemm_post_op        *post_op,
	lpgemm_post_op_attr   *post_op_attr
  )
{
  static void *post_ops_labels[] =
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
  const float *a_use = NULL;
  const float *b_use = NULL;
  float *c_use = NULL;

  lpgemm_post_op_attr post_ops_attr = *( post_op_attr );
  __m256i masks[9] = {
    _mm256_set_epi32(0,  0,  0,  0,  0,  0,  0,  0),    // 0 elements (all zeros)
    _mm256_set_epi32(0,  0,  0,  0,  0,  0,  0, -1),    // 1 element
    _mm256_set_epi32(0,  0,  0,  0,  0,  0, -1, -1),    // 2 elements
    _mm256_set_epi32(0,  0,  0,  0,  0, -1, -1, -1),    // 3 elements
    _mm256_set_epi32(0,  0,  0,  0, -1, -1, -1, -1),    // 4 elements
    _mm256_set_epi32(0,  0,  0, -1, -1, -1, -1, -1),    // 5 elements
    _mm256_set_epi32(0,  0, -1, -1, -1, -1, -1, -1),    // 6 elements
    _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1),    // 7 elements
    _mm256_set_epi32(-1, -1, -1, -1, -1, -1, -1, -1),   // 8 elements
  };

    dim_t nr0 = n0;
    c_use = c;
    __m256i k1 = masks[8], k2 = masks[8];

    //Declare the registers
    __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
    __m256 ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;

    //zero the accumulator registers
    ZERO_ACC_YMM_4_REG(ymm4, ymm5, ymm6, ymm7);
    ZERO_ACC_YMM_4_REG(ymm8,  ymm9,  ymm10, ymm11);
    ZERO_ACC_YMM_4_REG(ymm12, ymm13, ymm14, ymm15);

    dim_t n_left = n0 % 8;
    // n1, n2 holds the n_elems values.
    dim_t n1 = 0, n2 = 0;

    if (nr0 < 8)
    {
      k1 = masks[n_left];
      n1 = n_left;
      k2 =masks[0];
      n2 = 0;
    }
    else
    {
      k1 = masks[8];
      n1 = 8;
      k2 = masks[n_left];
      n2 = n_left;
    }

    _mm_prefetch((c_use + 0 * rs_c), _MM_HINT_T0);
    _mm_prefetch((c_use + 8 * rs_c), _MM_HINT_T0);

    for(dim_t pc = 0; pc < k; pc += KC)
    {
      dim_t kc0 = bli_min((k - pc), KC);
      uint64_t k_iter = kc0 / 4;
      uint64_t k_rem = kc0 % 4;
      dim_t ps_b_use = 0;
      dim_t rs_b_use = NR;

      // No parallelization in k dim, k always starts at 0.
      if (mtag_b == REORDERED||mtag_b == PACK)
      {
        // In multi-threaded scenarios, an extra offset into a given
        // packed B panel is required, since the jc loop split can
        // result in per thread start offset inside the panel, instead
        // of panel boundaries.
        b_use = b + (n_sub_updated * pc) + (jc_cur_loop_rem * kc0);
        ps_b_use = kc0;
      }
      else
      {
        b_use = b + (pc * rs_b);
        ps_b_use = 1;
        rs_b_use = rs_b;
      }

      a_use = a + pc;

      b_use = b_use + jr * ps_b_use;

      for (dim_t k = 0; k < k_iter; k++)
      {
        _mm_prefetch((b_use + 4 * rs_b_use), _MM_HINT_T0);
        _mm_prefetch((b_use + 5 * rs_b_use), _MM_HINT_T0);
        _mm_prefetch((b_use + 6 * rs_b_use), _MM_HINT_T0);
        _mm_prefetch((b_use + 7 * rs_b_use), _MM_HINT_T0);
        //Using mask loads to avoid writing fringe kernels
        //Load first 4x8 tile from row 0-3
        //float arr[8];
        ymm0 = _mm256_maskload_ps(b_use, k1);
        ymm1 = _mm256_maskload_ps(b_use + rs_b_use, k1);
        b_use += 8;
        ymm2 = _mm256_maskload_ps(b_use, k2);
        ymm3 = _mm256_maskload_ps(b_use + rs_b_use, k2);
        b_use -= 8;


        //Broadcast col0 - col3 element of A
        ymm4 = _mm256_broadcast_ss( a_use );
        ymm5 = _mm256_broadcast_ss( a_use + 1 );

        ymm8 = _mm256_fmadd_ps(ymm0, ymm4, ymm8);
        ymm9 = _mm256_fmadd_ps(ymm1, ymm5, ymm9);
        ymm12 = _mm256_fmadd_ps(ymm2, ymm4, ymm12);
        ymm13 = _mm256_fmadd_ps(ymm3, ymm5, ymm13);

        ymm6 = _mm256_broadcast_ss( a_use + 2 );
        ymm7 = _mm256_broadcast_ss( a_use + 3 );

        ymm0 = _mm256_maskload_ps(b_use + 2 * rs_b_use, k1);
        ymm1 = _mm256_maskload_ps(b_use + 3 * rs_b_use, k1);
        b_use += 8;
        ymm2 = _mm256_maskload_ps(b_use + 2 * rs_b_use, k2);
        ymm3 = _mm256_maskload_ps(b_use + 3 * rs_b_use, k2);


        ymm10 = _mm256_fmadd_ps(ymm0, ymm6, ymm10);
        ymm11 = _mm256_fmadd_ps(ymm1, ymm7, ymm11);
        ymm14 = _mm256_fmadd_ps(ymm2, ymm6, ymm14);
        ymm15 = _mm256_fmadd_ps(ymm3, ymm7, ymm15);

        b_use -= 8;// move b point back to start of KCXNR
        b_use += (4 * rs_b_use);
        a_use += 4;// move a pointer to next col
      }

      for (dim_t kr = 0; kr < k_rem; kr++)
      {
        //Load 16 elements from a row of B
        ymm0 = _mm256_maskload_ps(b_use, k1);
        ymm1 = _mm256_maskload_ps(b_use + 8, k2);

        //Broadcast Element of A
        ymm4 = _mm256_broadcast_ss( a_use ); // broadcast c0r0

        ymm8 = _mm256_fmadd_ps(ymm0, ymm4, ymm8);
        ymm12 = _mm256_fmadd_ps(ymm1, ymm4, ymm12);

        b_use += rs_b_use; // move b pointer to next row
        a_use++;     // move a pointer to next col
      }
    }

    //SUMUP K untoll output
    ymm8 = _mm256_add_ps(ymm9, ymm8);
    ymm10 = _mm256_add_ps(ymm11, ymm10);
    ymm8 = _mm256_add_ps(ymm10, ymm8);// 8 outputs

    ymm12 = _mm256_add_ps(ymm13, ymm12);
    ymm14 = _mm256_add_ps(ymm15, ymm14);
    ymm12 = _mm256_add_ps(ymm14, ymm12);// 8 outputs

    //Mulitply A*B output with alpha
    ymm0 =_mm256_set1_ps( alpha );
    ymm8 = _mm256_mul_ps( ymm0, ymm8 );
    ymm12 = _mm256_mul_ps( ymm0, ymm12 );

    if(beta != 0)
    {
      const float *_cbuf = c_use;
      // load c and multiply with beta and
      // add to accumulator and store back
      ymm3 = _mm256_set1_ps( beta );
      if( post_ops_attr.buf_downscale != NULL )
			{
        BF16_F32_C_BNZ_GEMV_MASK(0, ymm0, ymm3, ymm8, n1)
        BF16_F32_C_BNZ_GEMV_MASK(1, ymm1, ymm3, ymm12, n2)
      }
      else
      {
        ymm0 = _mm256_maskload_ps( _cbuf, k1 );
        ymm8 = _mm256_fmadd_ps( ymm0, ymm3, ymm8 );

        ymm1 = _mm256_maskload_ps( _cbuf + 8, k2 );
        ymm12 = _mm256_fmadd_ps( ymm1, ymm3, ymm12 );
      }
    }

    // Post Ops
    post_ops_attr.is_last_k = TRUE;
    lpgemm_post_op *post_ops_list_temp = post_op;
    POST_OP_LABEL_LASTK_SAFE_JUMP

    POST_OPS_BIAS_1x16F:
    {
      if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
           ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
      {
        if( post_ops_list_temp->stor_type == BF16 )
        {
          BF16_F32_BIAS_AVX2_GEMV_MASK(0, ymm0, n1 )
          BF16_F32_BIAS_AVX2_GEMV_MASK(1, ymm1, n2 )
          // BF16_F32_BIAS_LOAD_AVX2( ymm0, 0 );
          // BF16_F32_BIAS_LOAD_AVX2( ymm1, 1 );
        }
        else
        {
          ymm0 = _mm256_maskload_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 0 * 8 ), k1 );
          ymm1 = _mm256_maskload_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 1 * 8 ), k2 );
        }

        // c[0,0-7]
        ymm8 = _mm256_add_ps( ymm8, ymm0 );

        // c[0,8-15]
        ymm12 = _mm256_add_ps( ymm12, ymm1 );
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
        ymm8 = _mm256_add_ps( ymm8, ymm0 );

        // c[0,8-15]
        ymm12 = _mm256_add_ps( ymm12, ymm0 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_1x16F:
    {
      ymm0 = _mm256_setzero_ps();

      // c[0,0-7]
      ymm8 = _mm256_max_ps( ymm8, ymm0 );

      // c[0,8-15]
      ymm12 = _mm256_max_ps( ymm12, ymm0 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_1x16F:
    {
      ymm0 =
        _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
      ymm1 = _mm256_setzero_ps();

      // c[0,0-7]
      RELU_SCALE_OP_F32S_AVX2(ymm8, ymm0, ymm1, ymm2)

      // c[0,8-15]
      RELU_SCALE_OP_F32S_AVX2(ymm12, ymm0, ymm1, ymm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_1x16F:
    {
      __m256 dn, x_tanh;
      __m256i q;

      // c[0,0-7]
      GELU_TANH_F32S_AVX2(ymm8, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      // c[0,8-15]
      GELU_TANH_F32S_AVX2(ymm12, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_1x16F:
    {
      // c[0,0-7]
      GELU_ERF_F32S_AVX2(ymm8, ymm0, ymm1, ymm2)

      // c[0,8-15]
      GELU_ERF_F32S_AVX2(ymm12, ymm0, ymm1, ymm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_1x16F:
    {
      ymm0 = _mm256_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      ymm1 = _mm256_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0,0-7]
      CLIP_F32S_AVX2(ymm8, ymm0, ymm1)

      // c[0,8-15]
      CLIP_F32S_AVX2(ymm12, ymm0, ymm1)

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
        F32_SCL_MULRND_AVX2(ymm8, selector1, zero_point0);

        //c[0, 8-15]
        F32_SCL_MULRND_AVX2(ymm12, selector2, zero_point1);
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
        F32_SCL_MULRND_AVX2(ymm8, selector1, zero_point0);

        //c[0, 8-15]
        F32_SCL_MULRND_AVX2(ymm12, selector1, zero_point0);
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
          BF16_F32_MATRIX_ADD_2COL_YMM(ymm1,ymm2,scl_fctr1,scl_fctr2,0,8,12);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_2COL_YMM(ymm1,ymm2,scl_fctr1,scl_fctr1,0,8,12);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_2COL_YMM(ymm1,ymm2,scl_fctr1,scl_fctr2,0,8,12);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_2COL_YMM(ymm1,ymm2,scl_fctr1,scl_fctr1,0,8,12);
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
          BF16_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,0,8,12);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr1,0,8,12);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_2COL_YMM(ymm1,ymm2,scl_fctr1,scl_fctr2,0,8,12);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_2COL_YMM(ymm1,ymm2,scl_fctr1,scl_fctr1,0,8,12);
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
      SWISH_F32_AVX2_DEF(ymm8, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

      // c[0,8-15]
      SWISH_F32_AVX2_DEF(ymm12, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_1x16F:
    {
      __m256 dn;
      __m256i q;

      // c[0,0-7]
      TANH_F32S_AVX2(ymm8, ymm0, ymm1, ymm2, ymm3, dn, q)

      // c[0,8-15]
      TANH_F32S_AVX2(ymm12, ymm0, ymm1, ymm2, ymm3, dn, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SIGMOID_1x16F:
    {
      __m256 z, dn;
      __m256i ex_out;

      // c[0,0-7]
      SIGMOID_F32_AVX2_DEF(ymm8, ymm1, ymm2, ymm3, z, dn, ex_out)

      // c[0,8-15]
      SIGMOID_F32_AVX2_DEF(ymm12, ymm1, ymm2, ymm3, z, dn, ex_out)
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
      STORE_F32_BF16_YMM(ymm8, 0, 0, 8);
      STORE_F32_BF16_YMM(ymm12, 0, 1, 8);
    }
    else
    {
      _mm256_maskstore_ps (c_use, k1, ymm8);
      _mm256_maskstore_ps (c_use + 8, k2, ymm12);
    }

}



LPGEMV_M_EQ1_KERN( float, float, float, f32f32f32of32_avx2 )
{
  static void *post_ops_labels[] =
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
  const float *a_use = NULL;
  const float *b_use = NULL;
  float *c_use = NULL;

  lpgemm_post_op_attr post_ops_attr = *( post_op_attr );
  for( dim_t jr = 0; jr < n0; jr += NR )
  {
    dim_t nr0 = bli_min((n0 - jr), NR);
    c_use = c + jr;

    if (nr0 < NR)
    {
      b_use = b;
      lpgemv_m_one_f32f32f32of32_avx2_LT16
      (
        nr0, k,
        a, rs_a, cs_a, mtag_a,
        b_use, rs_b, cs_b, mtag_b,
        c_use, rs_c, jr,
        alpha, beta,
        NR, KC,
        n_sub_updated, jc_cur_loop_rem,
        post_op, &post_ops_attr
      );
      return;
    }

    //Declare the registers
    __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
    __m256 ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;

    //zero the accumulator registers
    ZERO_ACC_YMM_4_REG(ymm4, ymm5, ymm6, ymm7);
    ZERO_ACC_YMM_4_REG(ymm8,  ymm9,  ymm10, ymm11);
    ZERO_ACC_YMM_4_REG(ymm12, ymm13, ymm14, ymm15);

    _mm_prefetch((c_use + 0 * rs_c), _MM_HINT_T0);
    _mm_prefetch((c_use + 8 * rs_c), _MM_HINT_T0);

    for(dim_t pc = 0; pc < k; pc += KC)
    {
      dim_t kc0 = bli_min((k - pc), KC);
      uint64_t k_iter = kc0 / 4;
      uint64_t k_rem = kc0 % 4;
      dim_t ps_b_use = 0;
      dim_t rs_b_use = NR;

      // No parallelization in k dim, k always starts at 0.
      if (mtag_b == REORDERED||mtag_b == PACK)
      {
        // In multi-threaded scenarios, an extra offset into a given
        // packed B panel is required, since the jc loop split can
        // result in per thread start offset inside the panel, instead
        // of panel boundaries.
        b_use = b + (n_sub_updated * pc) + (jc_cur_loop_rem * kc0);
        ps_b_use = kc0;
      }
      else
      {
        b_use = b + (pc * rs_b);
        ps_b_use = 1;
        rs_b_use = rs_b;
      }

      a_use = a + pc;
      b_use = b_use + jr * ps_b_use;

      for (dim_t k = 0; k < k_iter; k++)
      {
        _mm_prefetch((b_use + 4 * rs_b_use), _MM_HINT_T0);
        _mm_prefetch((b_use + 5 * rs_b_use), _MM_HINT_T0);
        _mm_prefetch((b_use + 6 * rs_b_use), _MM_HINT_T0);
        _mm_prefetch((b_use + 7 * rs_b_use), _MM_HINT_T0);
        //Using mask loads to avoid writing fringe kernels

        //Load first 4x8 tile from row 0-3
        ymm0 = _mm256_loadu_ps(b_use);
        ymm1 = _mm256_loadu_ps(b_use + rs_b_use);
        ymm2 = _mm256_loadu_ps(b_use + 2 * rs_b_use);
        ymm3 = _mm256_loadu_ps(b_use + 3 * rs_b_use);
        b_use += 8;

        //Broadcast col0 - col3 element of A
        ymm4 = _mm256_broadcast_ss( a_use );
        ymm5 = _mm256_broadcast_ss( a_use + 1 );
        ymm6 = _mm256_broadcast_ss( a_use + 2 );
        ymm7 = _mm256_broadcast_ss( a_use + 3 );

        ymm8 = _mm256_fmadd_ps(ymm0, ymm4, ymm8);
        ymm9 = _mm256_fmadd_ps(ymm1, ymm5, ymm9);
        ymm10 = _mm256_fmadd_ps(ymm2, ymm6, ymm10);
        ymm11 = _mm256_fmadd_ps(ymm3, ymm7, ymm11);

        //Load second 4x8 tile from row 0-3
        ymm0 = _mm256_loadu_ps(b_use);
        ymm1 = _mm256_loadu_ps(b_use + rs_b_use);
        ymm2 = _mm256_loadu_ps(b_use + 2 * rs_b_use);
        ymm3 = _mm256_loadu_ps(b_use + 3 * rs_b_use);

        ymm12 = _mm256_fmadd_ps(ymm0, ymm4, ymm12);
        ymm13 = _mm256_fmadd_ps(ymm1, ymm5, ymm13);
        ymm14 = _mm256_fmadd_ps(ymm2, ymm6, ymm14);
        ymm15 = _mm256_fmadd_ps(ymm3, ymm7, ymm15);

        b_use -= 8;// move b point back to start of KCXNR
        b_use += (4 * rs_b_use);
        a_use += 4;// move a pointer to next col
      }

      for (dim_t kr = 0; kr < k_rem; kr++)
      {
        //Load 16 elements from a row of B
        ymm0 = _mm256_loadu_ps(b_use);
        ymm1 = _mm256_loadu_ps(b_use + 8);

        //Broadcast Element of A
        ymm4 = _mm256_broadcast_ss( a_use ); // broadcast c0r0

        ymm8 = _mm256_fmadd_ps(ymm0, ymm4, ymm8);
        ymm12 = _mm256_fmadd_ps(ymm1, ymm4, ymm12);

        b_use += rs_b_use; // move b pointer to next row
        a_use++;     // move a pointer to next col
      }
    }

    //SUMUP K untoll output
    ymm8 = _mm256_add_ps(ymm9, ymm8);
    ymm10 = _mm256_add_ps(ymm11, ymm10);
    ymm8 = _mm256_add_ps(ymm10, ymm8);// 8 outputs

    ymm12 = _mm256_add_ps(ymm13, ymm12);
    ymm14 = _mm256_add_ps(ymm15, ymm14);
    ymm12 = _mm256_add_ps(ymm14, ymm12);// 8 outputs

    //Mulitply A*B output with alpha
    ymm0 =_mm256_set1_ps( alpha );
    ymm8 = _mm256_mul_ps( ymm0, ymm8 );
    ymm12 = _mm256_mul_ps( ymm0, ymm12 );

    if(beta != 0)
    {
      const float *_cbuf = c_use;
      // load c and multiply with beta and
      // add to accumulator and store back
      ymm3 = _mm256_set1_ps( beta );

      if ( ( post_ops_attr.buf_downscale != NULL ) )
      {
        BF16_F32_C_BNZ_8(0,0,ymm0,ymm3,ymm8)
        BF16_F32_C_BNZ_8(0,1,ymm1,ymm3,ymm12)
      }
      else
      {
        ymm0 = _mm256_loadu_ps( _cbuf );
        ymm8 = _mm256_fmadd_ps( ymm0, ymm3, ymm8 );

        ymm1 = _mm256_loadu_ps( _cbuf + 8 );
        ymm12 = _mm256_fmadd_ps( ymm1, ymm3, ymm12 );

      }
    }

    // Post Ops
    post_ops_attr.is_last_k = TRUE;
    lpgemm_post_op *post_ops_list_temp = post_op;
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
        ymm8 = _mm256_add_ps( ymm8, ymm0 );

        // c[0,8-15]
        ymm12 = _mm256_add_ps( ymm12, ymm1 );
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
        ymm8 = _mm256_add_ps( ymm8, ymm0 );

        // c[0,8-15]
        ymm12 = _mm256_add_ps( ymm12, ymm0 );
      }
      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_1x16F:
    {
      ymm0 = _mm256_setzero_ps();

      // c[0,0-7]
      ymm8 = _mm256_max_ps( ymm8, ymm0 );

      // c[0,8-15]
      ymm12 = _mm256_max_ps( ymm12, ymm0 );

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_RELU_SCALE_1x16F:
    {
      ymm0 =
        _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
      ymm1 = _mm256_setzero_ps();

      // c[0,0-7]
      RELU_SCALE_OP_F32S_AVX2(ymm8, ymm0, ymm1, ymm2)

      // c[0,8-15]
      RELU_SCALE_OP_F32S_AVX2(ymm12, ymm0, ymm1, ymm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_TANH_1x16F:
    {
      __m256 dn, x_tanh;
      __m256i q;

      // c[0,0-7]
      GELU_TANH_F32S_AVX2(ymm8, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      // c[0,8-15]
      GELU_TANH_F32S_AVX2(ymm12, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_GELU_ERF_1x16F:
    {
      // c[0,0-7]
      GELU_ERF_F32S_AVX2(ymm8, ymm0, ymm1, ymm2)

      // c[0,8-15]
      GELU_ERF_F32S_AVX2(ymm12, ymm0, ymm1, ymm2)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_CLIP_1x16F:
    {
      ymm0 = _mm256_set1_ps( *( float* )post_ops_list_temp->op_args2 );
      ymm1 = _mm256_set1_ps( *( float* )post_ops_list_temp->op_args3 );

      // c[0,0-7]
      CLIP_F32S_AVX2(ymm8, ymm0, ymm1)

      // c[0,8-15]
      CLIP_F32S_AVX2(ymm12, ymm0, ymm1)

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
        F32_SCL_MULRND_AVX2(ymm8, selector1, zero_point0);

        //c[0, 8-15]
        F32_SCL_MULRND_AVX2(ymm12, selector2, zero_point1);
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
        F32_SCL_MULRND_AVX2(ymm8, selector1, zero_point0);

        //c[0, 8-15]
        F32_SCL_MULRND_AVX2(ymm12, selector1, zero_point0);
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
          BF16_F32_MATRIX_ADD_2COL_YMM(ymm1,ymm2,scl_fctr1,scl_fctr2,0,8,12);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_ADD_2COL_YMM(ymm1,ymm2,scl_fctr1,scl_fctr1,0,8,12);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_2COL_YMM(ymm1,ymm2,scl_fctr1,scl_fctr2,0,8,12);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_ADD_2COL_YMM(ymm1,ymm2,scl_fctr1,scl_fctr1,0,8,12);
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
          BF16_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,0,8,12);
        }
        else
        {
          // c[0:0-15]
          BF16_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr1,0,8,12);
        }
      }
      else
      {
        float* matptr = ( float* )post_ops_list_temp->op_args1;

        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
          ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_2COL_YMM(ymm1,ymm2,scl_fctr1,scl_fctr2,0,8,12);
        }
        else
        {
          // c[0:0-15]
          F32_F32_MATRIX_MUL_2COL_YMM(ymm1,ymm2,scl_fctr1,scl_fctr1,0,8,12);
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
      SWISH_F32_AVX2_DEF(ymm8, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

      // c[0,8-15]
      SWISH_F32_AVX2_DEF(ymm12, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_TANH_1x16F:
    {
      __m256 dn;
      __m256i q;

      // c[0,0-7]
      TANH_F32S_AVX2(ymm8, ymm0, ymm1, ymm2, ymm3, dn, q)

      // c[0,8-15]
      TANH_F32S_AVX2(ymm12, ymm0, ymm1, ymm2, ymm3, dn, q)

      POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
    }
POST_OPS_SIGMOID_1x16F:
    {
      __m256i ex_out;
      __m256 z, dn;

      // c[0,0-7]
      SIGMOID_F32_AVX2_DEF(ymm8, ymm1, ymm2, ymm3, z, dn, ex_out)

      // c[0,8-15]
      SIGMOID_F32_AVX2_DEF(ymm12, ymm1, ymm2, ymm3, z, dn, ex_out)

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
      STORE_F32_BF16_YMM(ymm8, 0, 0, 8);
      STORE_F32_BF16_YMM(ymm12, 0, 1, 8);
    }
    else
    {
      _mm256_storeu_ps (c_use, ymm8);
      _mm256_storeu_ps (c_use + 8, ymm12);
    }
    post_ops_attr.post_op_c_j += NR;
  }
}

#endif // BLIS_ADDON_LPGEMM

