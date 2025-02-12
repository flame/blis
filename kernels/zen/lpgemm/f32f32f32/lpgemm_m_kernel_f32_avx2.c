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

#define MR 6
#define NR 16

LPGEMM_MAIN_KERN(float,float,float,f32f32f32of32_6x16m)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_6x16F_DISABLE,
              &&POST_OPS_BIAS_6x16F,
              &&POST_OPS_RELU_6x16F,
              &&POST_OPS_RELU_SCALE_6x16F,
              &&POST_OPS_GELU_TANH_6x16F,
              &&POST_OPS_GELU_ERF_6x16F,
              &&POST_OPS_CLIP_6x16F,
              &&POST_OPS_DOWNSCALE_6x16F,
              &&POST_OPS_MATRIX_ADD_6x16F,
              &&POST_OPS_SWISH_6x16F,
              &&POST_OPS_MATRIX_MUL_6x16F,
              &&POST_OPS_TANH_6x16F,
              &&POST_OPS_SIGMOID_6x16F
            };
    uint64_t n_left = n0 % NR;  //n0 is expected to be n0<=NR
    // First check whether this is a edge case in the n dimension.
    // If so, dispatch other 6x?m kernels, as needed.
    if (n_left )
    {
        float*  cij = (float* )c;
        float*  bj  = (float* )b;
        float*  ai  = (float* )a;

        if ( 8 <= n_left )
        {
            const dim_t nr_cur = 8;

            lpgemm_rowvar_f32f32f32of32_6x8m
            (
              m0, k0,
              ai,  rs_a, cs_a, ps_a,
              bj,  rs_b, cs_b,
              cij, rs_c,
              alpha, beta,
              post_ops_list, post_ops_attr
            );

            cij += nr_cur*cs_c; bj += nr_cur*cs_b; n_left -= nr_cur;
            post_ops_attr.post_op_c_j += 8;
        }

        if ( 4 <= n_left )
        {
            const dim_t nr_cur = 4;

            lpgemm_rowvar_f32f32f32of32_6x4m
            (
              m0, k0,
              ai,  rs_a, cs_a, ps_a,
              bj,  rs_b, cs_b,
              cij, rs_c,
              alpha, beta,
              post_ops_list, post_ops_attr
            );
            cij += nr_cur*cs_c; bj += nr_cur*cs_b; n_left -= nr_cur;
            post_ops_attr.post_op_c_j += 4;
        }

        if ( 2 <= n_left )
        {
            const dim_t nr_cur = 2;

            lpgemm_rowvar_f32f32f32of32_6x2m
            (
              m0, k0,
              ai,  rs_a, cs_a, ps_a,
              bj,  rs_b, cs_b,
              cij, rs_c,
              alpha, beta,
              post_ops_list, post_ops_attr
            );
            cij += nr_cur*cs_c; bj += nr_cur*cs_b; n_left -= nr_cur;
            post_ops_attr.post_op_c_j += 2;
        }

        if ( 1 == n_left )
        {
            lpgemm_rowvar_f32f32f32of32_6x1m
            (
              m0, k0,
              ai,  rs_a, cs_a, ps_a,
              bj,  rs_b, cs_b,
              cij, rs_c,
              alpha, beta,
              post_ops_list, post_ops_attr
            );
        }

        return;
    }

    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    uint64_t m_iter = (uint64_t)m0 / 6;
    uint64_t m_left = (uint64_t)m0 % 6;

    if ( m_iter == 0 ){    goto consider_edge_cases; }

    /*Declare the registers*/
    __m256 ymm0, ymm1, ymm2, ymm3;
    __m256 ymm4, ymm5, ymm6, ymm7;
    __m256 ymm8, ymm9, ymm10, ymm11;
    __m256 ymm12, ymm13, ymm14, ymm15;

    /*Produce MRxNR outputs */
    for(dim_t m=0; m < m_iter; m++)
    {
      /* zero the accumulator registers */
      ZERO_ACC_YMM_4_REG(ymm4, ymm5, ymm6, ymm7);
      ZERO_ACC_YMM_4_REG(ymm8,  ymm9,  ymm10, ymm11);
      ZERO_ACC_YMM_4_REG(ymm12, ymm13, ymm14, ymm15);

      float *abuf, *bbuf, *cbuf, *_cbuf;

      abuf = (float *)a + m * ps_a; // Move to next MRxKC in MCxKC (where MC>=MR)
      bbuf = (float *)b;  //Same KCxNR panel is used across MCxKC block
      cbuf = (float *)c + m * MR * rs_c; // Move to next MRXNR in output

      /*_mm_prefetch( (MR X NR) from C*/
      _mm_prefetch((cbuf + 0*rs_c), _MM_HINT_T0);
      _mm_prefetch((cbuf + 1*rs_c), _MM_HINT_T0);
      _mm_prefetch((cbuf + 2*rs_c), _MM_HINT_T0);
      _mm_prefetch((cbuf + 3*rs_c), _MM_HINT_T0);
      _mm_prefetch((cbuf + 4*rs_c), _MM_HINT_T0);
      _mm_prefetch((cbuf + 5*rs_c), _MM_HINT_T0);

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
        ymm3 = _mm256_broadcast_ss((abuf + 5*rs_a)); //broadcast c0r5
        abuf += cs_a;  //move a pointer to next col

        ymm12 = _mm256_fmadd_ps(ymm0, ymm2, ymm12);
        ymm13 = _mm256_fmadd_ps(ymm1, ymm2, ymm13);
        ymm14 = _mm256_fmadd_ps(ymm0, ymm3, ymm14);
        ymm15 = _mm256_fmadd_ps(ymm1, ymm3, ymm15);
      }//kloop

      ymm0 = _mm256_broadcast_ss(&(alpha));
      ALPHA_MUL_ACC_YMM_4_REG(ymm4,ymm5,ymm6,ymm7,ymm0)
      ALPHA_MUL_ACC_YMM_4_REG(ymm8,ymm9,ymm10,ymm11,ymm0)
      ALPHA_MUL_ACC_YMM_4_REG(ymm12,ymm13,ymm14,ymm15,ymm0)

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
          BF16_F32_C_BNZ_8(5,0,ymm0,ymm3,ymm14)
          BF16_F32_C_BNZ_8(5,1,ymm1,ymm3,ymm15)
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
          _cbuf += rs_c;
          F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm14)
          F32_C_BNZ_8(_cbuf+8,rs_c,ymm1,ymm3,ymm15)
        }
      }

      // Post Ops
      lpgemm_post_op* post_ops_list_temp = post_ops_list;
      POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_6x16F:
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

          // c[4,0-7]
          ymm12 = _mm256_add_ps( ymm12, ymm0 );

          // c[4,8-15]
          ymm13 = _mm256_add_ps( ymm13, ymm1 );

          // c[5,0-7]
          ymm14 = _mm256_add_ps( ymm14, ymm0 );

          // c[5,8-15]
          ymm15 = _mm256_add_ps( ymm15, ymm1 );
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
            BF16_F32_BIAS_BCAST_AVX2(ymm1,5);
          }
          else
          {
            ymm0 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
                post_ops_attr.post_op_c_i + 4 );
            ymm1 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 5 );
          }

          // c[4,0-7]
          ymm12 = _mm256_add_ps( ymm12, ymm0 );

          // c[4,8-15]
          ymm13 = _mm256_add_ps( ymm13, ymm0 );

          // c[5,0-7]
          ymm14 = _mm256_add_ps( ymm14, ymm1 );

          // c[5,8-15]
          ymm15 = _mm256_add_ps( ymm15, ymm1 );
        }
        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_RELU_6x16F:
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

        // c[5,0-7]
        ymm14 = _mm256_max_ps( ymm14, ymm0 );

        // c[5,8-15]
        ymm15 = _mm256_max_ps( ymm15, ymm0 );

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_RELU_SCALE_6x16F:
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

        // c[5,0-7]
        RELU_SCALE_OP_F32S_AVX2(ymm14, ymm0, ymm1, ymm2)

        // c[5,8-15]
        RELU_SCALE_OP_F32S_AVX2(ymm15, ymm0, ymm1, ymm2)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_GELU_TANH_6x16F:
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

        // c[5,0-7]
        GELU_TANH_F32S_AVX2(ymm14, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

        // c[5,8-15]
        GELU_TANH_F32S_AVX2(ymm15, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_GELU_ERF_6x16F:
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

        // c[5,0-7]
        GELU_ERF_F32S_AVX2(ymm14, ymm0, ymm1, ymm2)

        // c[5,8-15]
        GELU_ERF_F32S_AVX2(ymm15, ymm0, ymm1, ymm2)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_CLIP_6x16F:
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

        // c[5,0-7]
        CLIP_F32S_AVX2(ymm14, ymm0, ymm1)

        // c[5,8-15]
        CLIP_F32S_AVX2(ymm15, ymm0, ymm1)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_DOWNSCALE_6x16F:
      {
        __m256 selector1 = _mm256_setzero_ps();
        __m256 selector2 = _mm256_setzero_ps();
        __m256 selector3 = _mm256_setzero_ps();
        __m256 selector4 = _mm256_setzero_ps();
        __m256 selector5 = _mm256_setzero_ps();
        __m256 selector6 = _mm256_setzero_ps();

        __m256 zero_point0 = _mm256_setzero_ps();
        __m256 zero_point1 = _mm256_setzero_ps();
        __m256 zero_point2 = _mm256_setzero_ps();
        __m256 zero_point3 = _mm256_setzero_ps();
        __m256 zero_point4 = _mm256_setzero_ps();
        __m256 zero_point5 = _mm256_setzero_ps();

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
                _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          selector2 =
                _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          selector3 =
                _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          selector4 =
                _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          selector5 =
                _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          selector6 =
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
            BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point5);
          }
          else
          {
            zero_point0 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point1 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point2 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point3 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point4 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point5 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
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

          //c[5, 0-7]
          F32_SCL_MULRND_AVX2(ymm14, selector1, zero_point0);

          //c[5, 8-15]
          F32_SCL_MULRND_AVX2(ymm15, selector2, zero_point1);
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
            selector6 =
                _mm256_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                                post_ops_attr.post_op_c_i + 5 ) );
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
              BF16_F32_ZP_VECTOR_BCAST_AVX2(zero_point5,5);
            }
            else
            {
              zero_point0 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 0 ) );
              zero_point1 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 1) );
              zero_point2 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 2 ) );
              zero_point3 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 3 ) );
              zero_point4 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 4 ) );
              zero_point5 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_i + 5 ) );
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

          //c[5, 0-7]
          F32_SCL_MULRND_AVX2(ymm14, selector6, zero_point5);

          //c[5, 8-15]
          F32_SCL_MULRND_AVX2(ymm15, selector6, zero_point5);
        }
        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_MATRIX_ADD_6x16F:
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
        __m256 scl_fctr6 = _mm256_setzero_ps();

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
          scl_fctr6 =
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
            scl_fctr6 =
              _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 5 ) );
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

          // c[5:0-15]
          BF16_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,5,14,15);
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

          // c[5:0-15]
          BF16_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr6,scl_fctr6,5,14,15);
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

          // c[5:0-15]
          F32_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,5,14,15);
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

          // c[5:0-15]
          F32_F32_MATRIX_ADD_2COL(ymm1,ymm2,scl_fctr6,scl_fctr6,5,14,15);
        }
      }
        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_MATRIX_MUL_6x16F:
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
        __m256 scl_fctr6 = _mm256_setzero_ps();

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
          scl_fctr6 =
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
            scl_fctr6 =
              _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 5 ) );
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

            // c[5:0-15]
            BF16_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,5,14,15);
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

            // c[5:0-15]
            BF16_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr6,scl_fctr6,5,14,15);
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

            // c[5:0-15]
            F32_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr1,scl_fctr2,5,14,15);
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

            // c[5:0-15]
            F32_F32_MATRIX_MUL_2COL(ymm1,ymm2,scl_fctr6,scl_fctr6,5,14,15);
          }
        }
        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_SWISH_6x16F:
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

          // c[5,0-7]
          SWISH_F32_AVX2_DEF(ymm14, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

          // c[5,8-15]
          SWISH_F32_AVX2_DEF(ymm15, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_TANH_6x16F:
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

        // c[5,0-7]
        TANH_F32S_AVX2(ymm14, ymm0, ymm1, ymm2, ymm3, dn, q)

        // c[5,8-15]
        TANH_F32S_AVX2(ymm15, ymm0, ymm1, ymm2, ymm3, dn, q)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_SIGMOID_6x16F:
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

          // c[5,0-7]
          SIGMOID_F32_AVX2_DEF(ymm14, ymm1, ymm2, ymm3, z, dn, ex_out)

          // c[5,8-15]
          SIGMOID_F32_AVX2_DEF(ymm15, ymm1, ymm2, ymm3, z, dn, ex_out)

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_6x16F_DISABLE:
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
        STORE_F32_BF16_YMM(ymm12, 4, 0);
        STORE_F32_BF16_YMM(ymm13, 4, 1);
        STORE_F32_BF16_YMM(ymm14, 5, 0);
        STORE_F32_BF16_YMM(ymm15, 5, 1);
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
        cbuf += rs_c;
        _mm256_storeu_ps(cbuf, ymm14);
        _mm256_storeu_ps(cbuf + 8, ymm15);
      }

      post_ops_attr.post_op_c_i += MR;
    }//mloop

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        float*  restrict cij = (float *) c + i_edge*rs_c;
        float*  restrict ai  = (float *) a + m_iter*ps_a;
        float*  restrict bj  = (float *) b;

        lpgemm_m_fringe_f32_ker_ft ker_fps[6] =
        {
          NULL,
          lpgemm_rowvar_f32f32f32of32_1x16,
          lpgemm_rowvar_f32f32f32of32_2x16,
          lpgemm_rowvar_f32f32f32of32_3x16,
          lpgemm_rowvar_f32f32f32of32_4x16,
          lpgemm_rowvar_f32f32f32of32_5x16
        };

        lpgemm_m_fringe_f32_ker_ft ker_fp = ker_fps[ m_left ];

        ker_fp
        (
          k0,
          ai, rs_a, cs_a,
          bj, rs_b, cs_b,
          cij, rs_c,
          alpha, beta,
          post_ops_list, post_ops_attr
        );
        return;
    }
}

LPGEMM_N_FRINGE_KERN(float,float,float,f32f32f32of32_6x8m)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_6x8F_DISABLE,
              &&POST_OPS_BIAS_6x8F,
              &&POST_OPS_RELU_6x8F,
              &&POST_OPS_RELU_SCALE_6x8F,
              &&POST_OPS_GELU_TANH_6x8F,
              &&POST_OPS_GELU_ERF_6x8F,
              &&POST_OPS_CLIP_6x8F,
              &&POST_OPS_DOWNSCALE_6x8F,
              &&POST_OPS_MATRIX_ADD_6x8F,
              &&POST_OPS_SWISH_6x8F,
              &&POST_OPS_MATRIX_MUL_6x8F,
              &&POST_OPS_TANH_6x8F,
              &&POST_OPS_SIGMOID_6x8F
            };

    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    uint64_t m_iter = (uint64_t)m0 / 6;
    uint64_t m_left = (uint64_t)m0 % 6;

    if ( m_iter == 0 ){    goto consider_edge_cases; }

    /*Declare the registers*/
    __m256 ymm0, ymm1, ymm2, ymm3;
    __m256 ymm4, ymm6, ymm8, ymm10;
    __m256 ymm12, ymm13, ymm14, ymm15;

    /*Produce MRxNR outputs */
    for(dim_t m=0; m < m_iter; m++)
    {
      /* zero the accumulator registers */
      ZERO_ACC_YMM_4_REG(ymm4, ymm6, ymm8, ymm10);
      ZERO_ACC_YMM_4_REG(ymm12, ymm13, ymm14, ymm15);

      float *abuf, *bbuf, *cbuf, *_cbuf;

      abuf = (float *)a + m * ps_a; // Move to next MRxKC in MCxKC (where MC>=MR)
      bbuf = (float *)b;  //Same KCxNR panel is used across MCxKC block
      cbuf = (float *)c + m * MR * rs_c; // Move to next MRXNR in output

      /*_mm_prefetch( (MR X NR) from C*/
      _mm_prefetch((cbuf + 0*rs_c), _MM_HINT_T0);
      _mm_prefetch((cbuf + 1*rs_c), _MM_HINT_T0);
      _mm_prefetch((cbuf + 2*rs_c), _MM_HINT_T0);
      _mm_prefetch((cbuf + 3*rs_c), _MM_HINT_T0);
      _mm_prefetch((cbuf + 4*rs_c), _MM_HINT_T0);
      _mm_prefetch((cbuf + 5*rs_c), _MM_HINT_T0);

      for(dim_t k = 0; k < k_iter; k++)
      {
        /*Load 8 elements from row0 of B*/
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

        ymm2 = _mm256_broadcast_ss((abuf + 4*rs_a)); //broadcast c0r4
        ymm3 = _mm256_broadcast_ss((abuf + 5*rs_a)); //broadcast c0r5
        abuf += cs_a;  //move a pointer to next col

        ymm12 = _mm256_fmadd_ps(ymm0, ymm2, ymm12);
        ymm14 = _mm256_fmadd_ps(ymm0, ymm3, ymm14);
      }//kloop

      ymm0 = _mm256_broadcast_ss(&(alpha));
      ALPHA_MUL_ACC_YMM_4_REG(ymm4,ymm6,ymm8,ymm10,ymm0)
      ALPHA_MUL_ACC_YMM_4_REG(ymm12,ymm13,ymm14,ymm15,ymm0)

      if ( beta != 0.0 )
      {
        //load c and multiply with beta and
        //add to accumulator and store back
        ymm3 = _mm256_broadcast_ss(&(beta));
        if ( ( post_ops_attr.buf_downscale != NULL ) &&
          ( post_ops_attr.is_first_k == TRUE ) )
        {
          BF16_F32_C_BNZ_8(0,0,ymm0,ymm3,ymm4)
          BF16_F32_C_BNZ_8(1,0,ymm0,ymm3,ymm6)
          BF16_F32_C_BNZ_8(2,0,ymm0,ymm3,ymm8)
          BF16_F32_C_BNZ_8(3,0,ymm0,ymm3,ymm10)
          BF16_F32_C_BNZ_8(4,0,ymm0,ymm3,ymm12)
          BF16_F32_C_BNZ_8(5,0,ymm0,ymm3,ymm14)
        }
        else
        {
          _cbuf = cbuf;
          F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm4)
          _cbuf += rs_c;
          F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm6)
          _cbuf += rs_c;
          F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm8)
          _cbuf += rs_c;
          F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm10)
          _cbuf += rs_c;
          F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm12)
          _cbuf += rs_c;
          F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm14)
        }
      }

      // Post Ops
      lpgemm_post_op* post_ops_list_temp = post_ops_list;
      POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_6x8F:
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

          // c[5,0-7]
          ymm14 = _mm256_add_ps( ymm14, ymm0 );
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
            BF16_F32_BIAS_BCAST_AVX2(ymm1,5);
          }
          else
          {
            ymm0 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
                    post_ops_attr.post_op_c_i + 4 );
            ymm1 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
                    post_ops_attr.post_op_c_i + 5 );
          }

          // c[4,0-7]
          ymm12 = _mm256_add_ps( ymm12, ymm0 );

          // c[5,0-7]
          ymm14 = _mm256_add_ps( ymm14, ymm1 );
        }
        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_RELU_6x8F:
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

        // c[5,0-7]
        ymm14 = _mm256_max_ps( ymm14, ymm0 );

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_RELU_SCALE_6x8F:
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

        // c[5,0-7]
        RELU_SCALE_OP_F32S_AVX2(ymm14, ymm0, ymm1, ymm2)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_GELU_TANH_6x8F:
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

        // c[5,0-7]
        GELU_TANH_F32S_AVX2(ymm14, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_GELU_ERF_6x8F:
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

        // c[5,0-7]
        GELU_ERF_F32S_AVX2(ymm14, ymm0, ymm1, ymm2)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_CLIP_6x8F:
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

        // c[5,0-7]
        CLIP_F32S_AVX2(ymm14, ymm0, ymm1)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_DOWNSCALE_6x8F:
      {
        __m256 selector1 = _mm256_setzero_ps();
        __m256 selector2 = _mm256_setzero_ps();
        __m256 selector3 = _mm256_setzero_ps();
        __m256 selector4 = _mm256_setzero_ps();
        __m256 selector5 = _mm256_setzero_ps();
        __m256 selector6 = _mm256_setzero_ps();

        __m256 zero_point0 = _mm256_setzero_ps();
        __m256 zero_point1 = _mm256_setzero_ps();
        __m256 zero_point2 = _mm256_setzero_ps();
        __m256 zero_point3 = _mm256_setzero_ps();
        __m256 zero_point4 = _mm256_setzero_ps();
        __m256 zero_point5 = _mm256_setzero_ps();

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
                _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          selector2 =
                _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          selector3 =
                _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          selector4 =
                _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          selector5 =
                _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
          selector6 =
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
            BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point5);
          }
          else
          {
            zero_point0 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point1 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point2 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point3 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point4 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
            zero_point5 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
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

          //c[5, 0-7]
          F32_SCL_MULRND_AVX2(ymm14, selector1, zero_point0);
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
            selector6 =
                _mm256_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                                post_ops_attr.post_op_c_i + 5 ) );
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
              BF16_F32_ZP_VECTOR_BCAST_AVX2(zero_point5,5);
            }
            else
            {
              zero_point0 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                    post_ops_attr.post_op_c_i + 0 ) );
              zero_point1 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                    post_ops_attr.post_op_c_i + 1) );
              zero_point2 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                    post_ops_attr.post_op_c_i + 2 ) );
              zero_point3 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                    post_ops_attr.post_op_c_i + 3 ) );
              zero_point4 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                    post_ops_attr.post_op_c_i + 4 ) );
              zero_point5 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                    post_ops_attr.post_op_c_i + 5 ) );
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

          //c[5, 0-7]
          F32_SCL_MULRND_AVX2(ymm14, selector6, zero_point5);
        }
        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_MATRIX_ADD_6x8F:
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
        __m256 scl_fctr6 = _mm256_setzero_ps();

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
          scl_fctr6 =
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
            scl_fctr6 =
              _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 5 ) );
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

            // c[5:0-15]
            BF16_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,5,14);
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

            // c[5:0-15]
            BF16_F32_MATRIX_ADD_1COL(ymm1,scl_fctr6,5,14);
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

            // c[5:0-15]
            F32_F32_MATRIX_ADD_1COL(ymm1,scl_fctr1,5,14);
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

            // c[5:0-15]
            F32_F32_MATRIX_ADD_1COL(ymm1,scl_fctr6,5,14);
          }
        }
        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_MATRIX_MUL_6x8F:
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
        __m256 scl_fctr6 = _mm256_setzero_ps();

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
          scl_fctr6 =
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
            scl_fctr6 =
              _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 5 ) );
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

            // c[5:0-15]
            BF16_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,5,14);
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

            // c[5:0-15]
            BF16_F32_MATRIX_MUL_1COL(ymm1,scl_fctr6,5,14);
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

            // c[5:0-15]
            F32_F32_MATRIX_MUL_1COL(ymm1,scl_fctr1,5,14);
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

            // c[5:0-15]
            F32_F32_MATRIX_MUL_1COL(ymm1,scl_fctr6,5,14);
          }
        }
        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_SWISH_6x8F:
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

          // c[5,0-7]
          SWISH_F32_AVX2_DEF(ymm14, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_TANH_6x8F:
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

        // c[5,0-7]
        TANH_F32S_AVX2(ymm14, ymm0, ymm1, ymm2, ymm3, dn, q)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_SIGMOID_6x8F:
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

          // c[5,0-7]
          SIGMOID_F32_AVX2_DEF(ymm14, ymm1, ymm2, ymm3, z, dn, ex_out)

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_6x8F_DISABLE:
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
        STORE_F32_BF16_YMM(ymm14, 5, 0)
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
        cbuf += rs_c;
        _mm256_storeu_ps(cbuf, ymm14);
      }

      post_ops_attr.post_op_c_i += MR;
    }//mloop

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        float*  restrict cij = (float *) c + i_edge*rs_c;
        float*  restrict ai  = (float *) a + m_iter*ps_a;
        float*  restrict bj  = (float *) b;

        lpgemm_m_fringe_f32_ker_ft ker_fps[6] =
        {
          NULL,
          lpgemm_rowvar_f32f32f32of32_1x8,
          lpgemm_rowvar_f32f32f32of32_2x8,
          lpgemm_rowvar_f32f32f32of32_3x8,
          lpgemm_rowvar_f32f32f32of32_4x8,
          lpgemm_rowvar_f32f32f32of32_5x8
        };

        lpgemm_m_fringe_f32_ker_ft ker_fp = ker_fps[ m_left ];

        ker_fp
        (
          k0,
          ai, rs_a, cs_a,
          bj, rs_b, cs_b,
          cij,rs_c,
          alpha, beta,
          post_ops_list, post_ops_attr
        );
        return;
    }
}

LPGEMM_N_FRINGE_KERN(float,float,float,f32f32f32of32_6x4m)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_6x4F_DISABLE,
              &&POST_OPS_BIAS_6x4F,
              &&POST_OPS_RELU_6x4F,
              &&POST_OPS_RELU_SCALE_6x4F,
              &&POST_OPS_GELU_TANH_6x4F,
              &&POST_OPS_GELU_ERF_6x4F,
              &&POST_OPS_CLIP_6x4F,
              &&POST_OPS_DOWNSCALE_6x4F,
              &&POST_OPS_MATRIX_ADD_6x4F,
              &&POST_OPS_SWISH_6x4F,
              &&POST_OPS_MATRIX_MUL_6x4F,
              &&POST_OPS_TANH_6x4F,
              &&POST_OPS_SIGMOID_6x4F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    uint64_t m_iter = (uint64_t)m0 / 6;
    uint64_t m_left = (uint64_t)m0 % 6;

    if ( m_iter == 0 ){    goto consider_edge_cases; }

    /*Declare the registers*/
    __m128 xmm0, xmm1, xmm2, xmm3;
    __m128 xmm4, xmm5, xmm6, xmm7;
    __m128 xmm8, xmm9;

    /*Produce MRxNR outputs */
    for(dim_t m=0; m < m_iter; m++)
    {
      /* zero the accumulator registers */
      ZERO_ACC_XMM_4_REG(xmm4,xmm5,xmm6,xmm7)
      ZERO_ACC_XMM_4_REG(xmm8,xmm9,xmm0,xmm1)

      float *abuf, *bbuf, *cbuf, *_cbuf;

      abuf = (float *)a + m * ps_a; // Move to next MRxKC in MCxKC (where MC>=MR)
      bbuf = (float *)b;  //Same KCxNR panel is used across MCxKC block
      cbuf = (float *)c + m * MR * rs_c; // Move to next MRXNR in output

      /*_mm_prefetch( (MR X NR) from C*/
      _mm_prefetch((cbuf + 0*rs_c), _MM_HINT_T0);
      _mm_prefetch((cbuf + 1*rs_c), _MM_HINT_T0);
      _mm_prefetch((cbuf + 2*rs_c), _MM_HINT_T0);
      _mm_prefetch((cbuf + 3*rs_c), _MM_HINT_T0);
      _mm_prefetch((cbuf + 4*rs_c), _MM_HINT_T0);
      _mm_prefetch((cbuf + 5*rs_c), _MM_HINT_T0);

      for(dim_t k = 0; k < k_iter; k++)
      {
        /*Load 4 elements from row0 of B*/
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
        xmm3 = _mm_broadcast_ss((abuf + 5*rs_a)); //broadcast c0r0
        abuf += cs_a;  //move a pointer to next col

        xmm7 = _mm_fmadd_ps(xmm0, xmm1, xmm7);
        xmm8 = _mm_fmadd_ps(xmm0, xmm2, xmm8);
        xmm9 = _mm_fmadd_ps(xmm0, xmm3, xmm9);
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
          BF16_F32_C_BNZ_4(5,0,xmm1,xmm3,xmm9)
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
          _cbuf += rs_c;
          F32_C_BNZ_4(_cbuf,rs_c,xmm1,xmm3,xmm9)
        }
      }

      // Post Ops
      lpgemm_post_op* post_ops_list_temp = post_ops_list;
      POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_6x4F:
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

          // c[5,0-3]
          xmm9 = _mm_add_ps( xmm9, xmm0 );
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
            BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm1, 5);
          }
          else
          {
            xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 4 );
            xmm1 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 5 );
          }

          // c[4,0-3]
          xmm8 = _mm_add_ps( xmm8, xmm0 );

          // c[5,0-3]
          xmm9 = _mm_add_ps( xmm9, xmm1 );
        }
        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_RELU_6x4F:
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

        // c[5,0-3]
        xmm9 = _mm_max_ps( xmm9, xmm0 );

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_RELU_SCALE_6x4F:
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

        // c[5,0-3]
        RELU_SCALE_OP_F32S_SSE(xmm9, xmm0, xmm1, xmm2)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_GELU_TANH_6x4F:
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

        // c[5,0-3]
        GELU_TANH_F32S_SSE(xmm9, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_GELU_ERF_6x4F:
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

        // c[5,0-3]
        GELU_ERF_F32S_SSE(xmm9, xmm0, xmm1, xmm2)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_CLIP_6x4F:
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

        // c[5,0-3]
        CLIP_F32S_SSE(xmm9, xmm0, xmm1)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_DOWNSCALE_6x4F:
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

          //c[4, 0-3]
          F32_SCL_MULRND_SSE(xmm8, selector0, zero_point0);

          //c[5, 0-3]
          F32_SCL_MULRND_SSE(xmm9, selector0, zero_point0);
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
              zero_point3 =_mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
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

          if( post_ops_list_temp->scale_factor_len > 1 )
          {
            selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                            post_ops_attr.post_op_c_i + 4 ) );
            selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                            post_ops_attr.post_op_c_i + 5 ) );
          }
          if( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
          {
            if( is_bf16 == TRUE )
            {
              BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point0,4)
              BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point1,5)
            }
            else
            {
              zero_point0 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                              post_ops_attr.post_op_c_i + 4 ) );
              zero_point1 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                              post_ops_attr.post_op_c_i + 5 ) );
            }
          }

          //c[4, 0-3]
          F32_SCL_MULRND_SSE(xmm8, selector0, zero_point0);

          //c[5, 0-3]
          F32_SCL_MULRND_SSE(xmm9, selector1, zero_point1);
        }
        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_MATRIX_ADD_6x4F:
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
        __m128 scl_fctr6 = _mm_setzero_ps();

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
          scl_fctr6 =
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
            scl_fctr6 =
              _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 5 ) );
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

            // c[5:0-15]
            BF16_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,5,9);
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

            // c[5:0-15]
            BF16_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr6,5,9);
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

            // c[5:0-15]
            F32_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr1,5,9);
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

            // c[5:0-15]
            F32_F32_MATRIX_ADD_1COL_XMM(xmm1,scl_fctr6,5,9);
          }
        }
        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_MATRIX_MUL_6x4F:
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
        __m128 scl_fctr6 = _mm_setzero_ps();

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
          scl_fctr6 =
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
            scl_fctr6 =
              _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 5 ) );
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

            // c[5:0-15]
            BF16_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,5,9);
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

            // c[5:0-15]
            BF16_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr6,5,9);
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

            // c[5:0-15]
            F32_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr1,5,9);
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

            // c[5:0-15]
            F32_F32_MATRIX_MUL_1COL_XMM(xmm1,scl_fctr6,5,9);
          }
        }
        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_SWISH_6x4F:
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

          // c[5,0-3]
          SWISH_F32_SSE_DEF(xmm9, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_TANH_6x4F:
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

        // c[5,0-3]
        TANH_F32S_SSE(xmm9, xmm0, xmm1, xmm2, xmm3, dn, q)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_SIGMOID_6x4F:
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

          // c[5,0-3]
          SIGMOID_F32_SSE_DEF(xmm9, xmm1, xmm2, xmm3, z, dn, ex_out)

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_6x4F_DISABLE:
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
        STORE_F32_BF16_4XMM(xmm9, 5, 0)
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
        cbuf += rs_c;
        _mm_storeu_ps(cbuf, xmm9);
      }

      post_ops_attr.post_op_c_i += MR;
    }//mloop

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        float*  restrict cij = (float *) c + i_edge*rs_c;
        float*  restrict ai  = (float *) a + m_iter*ps_a;
        float*  restrict bj  = (float *) b;

        lpgemm_m_fringe_f32_ker_ft ker_fps[6] =
        {
          NULL,
          lpgemm_rowvar_f32f32f32of32_1x4,
          lpgemm_rowvar_f32f32f32of32_2x4,
          lpgemm_rowvar_f32f32f32of32_3x4,
          lpgemm_rowvar_f32f32f32of32_4x4,
          lpgemm_rowvar_f32f32f32of32_5x4
        };

        lpgemm_m_fringe_f32_ker_ft ker_fp = ker_fps[ m_left ];

        ker_fp
        (
          k0,
          ai, rs_a, cs_a,
          bj, rs_b, cs_b,
          cij, rs_c,
          alpha, beta,
          post_ops_list, post_ops_attr
        );
        return;
    }
}

LPGEMM_N_FRINGE_KERN(float,float,float,f32f32f32of32_6x2m)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_6x2F_DISABLE,
              &&POST_OPS_BIAS_6x2F,
              &&POST_OPS_RELU_6x2F,
              &&POST_OPS_RELU_SCALE_6x2F,
              &&POST_OPS_GELU_TANH_6x2F,
              &&POST_OPS_GELU_ERF_6x2F,
              &&POST_OPS_CLIP_6x2F,
              &&POST_OPS_DOWNSCALE_6x2F,
              &&POST_OPS_MATRIX_ADD_6x2F,
              &&POST_OPS_SWISH_6x2F,
              &&POST_OPS_MATRIX_MUL_6x2F,
              &&POST_OPS_TANH_6x2F,
              &&POST_OPS_SIGMOID_6x2F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    uint64_t m_iter = (uint64_t)m0 / 6;
    uint64_t m_left = (uint64_t)m0 % 6;

    if ( m_iter == 0 ){    goto consider_edge_cases; }

    /*Declare the registers*/
    __m128 xmm0, xmm1, xmm2, xmm3;
    __m128 xmm4, xmm5, xmm6, xmm7;
    __m128 xmm8, xmm9;

    /*Produce MRxNR outputs */
    for(dim_t m=0; m < m_iter; m++)
    {
      /* zero the accumulator registers */
      ZERO_ACC_XMM_4_REG(xmm4,xmm5,xmm6,xmm7)
      ZERO_ACC_XMM_4_REG(xmm8,xmm9,xmm0,xmm1)

      float *abuf, *bbuf, *cbuf, *_cbuf;

      abuf = (float *)a + m * ps_a; // Move to next MRxKC in MCxKC (where MC>=MR)
      bbuf = (float *)b;  //Same KCxNR panel is used across MCxKC block
      cbuf = (float *)c + m * MR * rs_c; // Move to next MRXNR in output

      /*_mm_prefetch( (MR X NR) from C*/
      _mm_prefetch((cbuf + 0*rs_c), _MM_HINT_T0);
      _mm_prefetch((cbuf + 1*rs_c), _MM_HINT_T0);
      _mm_prefetch((cbuf + 2*rs_c), _MM_HINT_T0);
      _mm_prefetch((cbuf + 3*rs_c), _MM_HINT_T0);
      _mm_prefetch((cbuf + 4*rs_c), _MM_HINT_T0);
      _mm_prefetch((cbuf + 5*rs_c), _MM_HINT_T0);

      for(dim_t k = 0; k < k_iter; k++)
      {
        /*Load 2 elements from row0 of B*/
        xmm0 = ( __m128 )_mm_load_sd((const double*) bbuf );
        bbuf += rs_b;  //move b pointer to next row

        xmm1 = _mm_broadcast_ss((abuf + 0*rs_a)); //broadcast c0r0
        xmm2 = _mm_broadcast_ss((abuf + 1*rs_a)); //broadcast c0r0
        xmm3 = _mm_broadcast_ss((abuf + 2*rs_a)); //broadcast c0r0

        xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);
        xmm5 = _mm_fmadd_ps(xmm0, xmm2, xmm5);
        xmm6 = _mm_fmadd_ps(xmm0, xmm3, xmm6);

        xmm1 = _mm_broadcast_ss((abuf + 3*rs_a)); //broadcast c0r0
        xmm2 = _mm_broadcast_ss((abuf + 4*rs_a)); //broadcast c0r0
        xmm3 = _mm_broadcast_ss((abuf + 5*rs_a)); //broadcast c0r0
        abuf += cs_a;  //move a pointer to next col

        xmm7 = _mm_fmadd_ps(xmm0, xmm1, xmm7);
        xmm8 = _mm_fmadd_ps(xmm0, xmm2, xmm8);
        xmm9 = _mm_fmadd_ps(xmm0, xmm3, xmm9);
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
          BF16_F32_C_BNZ_2(5,0,xmm1,xmm3,xmm9)
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
          _cbuf += rs_c;
          F32_C_BNZ_2(_cbuf,rs_c,xmm1,xmm3,xmm9)
        }
      }

      // Post Ops
      lpgemm_post_op* post_ops_list_temp = post_ops_list;
      POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_6x2F:
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
            xmm0 = ( __m128 )_mm_load_sd( (const double*)
                  (( float* )post_ops_list_temp->op_args1 +
                  post_ops_attr.post_op_c_j + ( 0 * 8 ) ));
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

          // c[5,0-3]
          xmm9 = _mm_add_ps( xmm9, xmm0 );
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
            BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm1, 5);
          }
          else
          {
            xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
                post_ops_attr.post_op_c_i + 4 );
            xmm1 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
                post_ops_attr.post_op_c_i + 5 );
          }

          // c[4,0-3]
          xmm8 = _mm_add_ps( xmm8, xmm0 );

          // c[5,0-3]
          xmm9 = _mm_add_ps( xmm9, xmm1 );
        }
        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_RELU_6x2F:
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

        // c[5,0-3]
        xmm9 = _mm_max_ps( xmm9, xmm0 );

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_RELU_SCALE_6x2F:
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

        // c[5,0-3]
        RELU_SCALE_OP_F32S_SSE(xmm9, xmm0, xmm1, xmm2)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_GELU_TANH_6x2F:
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

        // c[5,0-3]
        GELU_TANH_F32S_SSE(xmm9, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_GELU_ERF_6x2F:
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

        // c[5,0-3]
        GELU_ERF_F32S_SSE(xmm9, xmm0, xmm1, xmm2)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_CLIP_6x2F:
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

        // c[5,0-3]
        CLIP_F32S_SSE(xmm9, xmm0, xmm1)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_DOWNSCALE_6x2F:
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
            selector0 = ( __m128 )_mm_load_sd( (const double*)
                            ( ( float* )post_ops_list_temp->scale_factor +
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
              zero_point0 = ( __m128 )_mm_load_sd( (const double*)
                            ( (float* )post_ops_list_temp->op_args1 +
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

          //c[5, 0-3]
          F32_SCL_MULRND_SSE(xmm9, selector0, zero_point0);
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

          if( post_ops_list_temp->scale_factor_len > 1 )
          {
            selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                            post_ops_attr.post_op_c_i + 4 ) );
            selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                            post_ops_attr.post_op_c_i + 5 ) );
          }
          if( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
          {
            if( is_bf16 == TRUE )
            {
              BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point0,4)
              BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point1,5)
            }
            else
            {
              zero_point0 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                              post_ops_attr.post_op_c_i + 4 ) );
              zero_point1 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                              post_ops_attr.post_op_c_i + 5 ) );
            }
          }
          //c[4, 0-3]
          F32_SCL_MULRND_SSE(xmm8, selector0, zero_point0);

          //c[5, 0-3]
          F32_SCL_MULRND_SSE(xmm9, selector1, zero_point1);
        }
        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_MATRIX_ADD_6x2F:
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
        __m128 scl_fctr6 = _mm_setzero_ps();

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
          scl_fctr6 =
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
            scl_fctr6 =
              _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 5 ) );
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

            // c[5:0-15]
            BF16_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,5,9);
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

            // c[5:0-15]
            BF16_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr6,5,9);
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

            // c[5:0-15]
            F32_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr1,5,9);
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

            // c[5:0-15]
            F32_F32_MATRIX_ADD_1COL_XMM_2ELE(xmm1,scl_fctr6,5,9);
          }
        }
        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_MATRIX_MUL_6x2F:
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
        __m128 scl_fctr6 = _mm_setzero_ps();

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
          scl_fctr6 =
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
            scl_fctr6 =
              _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 5 ) );
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

            // c[5:0-15]
            BF16_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,5,9);
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

            // c[5:0-15]
            BF16_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr6,5,9);
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

            // c[5:0-15]
            F32_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr1,5,9);
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

            // c[5:0-15]
            F32_F32_MATRIX_MUL_1COL_XMM_2ELE(xmm1,scl_fctr6,5,9);
          }
        }
        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_SWISH_6x2F:
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

          // c[5,0-1]
          SWISH_F32_SSE_DEF(xmm9, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_TANH_6x2F:
      {
        __m128 dn;
        __m128i q;

        // c[0,0-1]
        TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, q)

        // c[1,0-1]
        TANH_F32S_SSE(xmm5, xmm0, xmm1, xmm2, xmm3, dn, q)

        // c[2,0-1]
        TANH_F32S_SSE(xmm6, xmm0, xmm1, xmm2, xmm3, dn, q)

        // c[3,0-1]
        TANH_F32S_SSE(xmm7, xmm0, xmm1, xmm2, xmm3, dn, q)

        // c[4,0-1]
        TANH_F32S_SSE(xmm8, xmm0, xmm1, xmm2, xmm3, dn, q)

        // c[5,0-1]
        TANH_F32S_SSE(xmm9, xmm0, xmm1, xmm2, xmm3, dn, q)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_SIGMOID_6x2F:
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

          // c[5,0-1]
          SIGMOID_F32_SSE_DEF(xmm9, xmm1, xmm2, xmm3, z, dn, ex_out)

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_6x2F_DISABLE:
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
        STORE_F32_BF16_2XMM(xmm9, 5, 0)
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
        cbuf += rs_c;
        _mm_store_sd((double*)cbuf, ( __m128d )xmm9);
      }

      post_ops_attr.post_op_c_i += MR;
    }//mloop

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        float*  restrict cij = (float *) c + i_edge*rs_c;
        float*  restrict ai  = (float *) a + m_iter*ps_a;
        float*  restrict bj  = (float *) b;

        lpgemm_m_fringe_f32_ker_ft ker_fps[6] =
        {
          NULL,
          lpgemm_rowvar_f32f32f32of32_1x2,
          lpgemm_rowvar_f32f32f32of32_2x2,
          lpgemm_rowvar_f32f32f32of32_3x2,
          lpgemm_rowvar_f32f32f32of32_4x2,
          lpgemm_rowvar_f32f32f32of32_5x2
        };

        lpgemm_m_fringe_f32_ker_ft ker_fp = ker_fps[ m_left ];

        ker_fp
        (
          k0,
          ai, rs_a, cs_a,
          bj, rs_b, cs_b,
          cij,rs_c,
          alpha, beta,
          post_ops_list, post_ops_attr
        );
        return;
    }
}

LPGEMM_N_FRINGE_KERN(float,float,float,f32f32f32of32_6x1m)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_6x1F_DISABLE,
              &&POST_OPS_BIAS_6x1F,
              &&POST_OPS_RELU_6x1F,
              &&POST_OPS_RELU_SCALE_6x1F,
              &&POST_OPS_GELU_TANH_6x1F,
              &&POST_OPS_GELU_ERF_6x1F,
              &&POST_OPS_CLIP_6x1F,
              &&POST_OPS_DOWNSCALE_6x1F,
              &&POST_OPS_MATRIX_ADD_6x1F,
              &&POST_OPS_SWISH_6x1F,
              &&POST_OPS_MATRIX_MUL_6x1F,
              &&POST_OPS_TANH_6x1F,
              &&POST_OPS_SIGMOID_6x1F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    uint64_t m_iter = (uint64_t)m0 / 6;
    uint64_t m_left = (uint64_t)m0 % 6;

    if ( m_iter == 0 ){    goto consider_edge_cases; }

    /*Declare the registers*/
    __m128 xmm0, xmm1, xmm2, xmm3;
    __m128 xmm4, xmm5, xmm6, xmm7;
    __m128 xmm8, xmm9;

    /*Produce MRxNR outputs */
    for(dim_t m=0; m < m_iter; m++)
    {
      /* zero the accumulator registers */
      ZERO_ACC_XMM_4_REG(xmm4,xmm5,xmm6,xmm7)
      ZERO_ACC_XMM_4_REG(xmm8,xmm9,xmm0,xmm1)

      float *abuf, *bbuf, *cbuf, *_cbuf;

      abuf = (float *)a + m * ps_a; // Move to next MRxKC in MCxKC (where MC>=MR)
      bbuf = (float *)b;  //Same KCxNR panel is used across MCxKC block
      cbuf = (float *)c + m * MR * rs_c; // Move to next MRXNR in output

      /*_mm_prefetch( (MR X NR) from C*/
      _mm_prefetch((cbuf + 0*rs_c), _MM_HINT_T0);
      _mm_prefetch((cbuf + 1*rs_c), _MM_HINT_T0);
      _mm_prefetch((cbuf + 2*rs_c), _MM_HINT_T0);
      _mm_prefetch((cbuf + 3*rs_c), _MM_HINT_T0);
      _mm_prefetch((cbuf + 4*rs_c), _MM_HINT_T0);
      _mm_prefetch((cbuf + 5*rs_c), _MM_HINT_T0);

      for(dim_t k = 0; k < k_iter; k++)
      {
        /*Load 1 elements from row0 of B*/
        xmm0 = _mm_load_ss(bbuf );
        bbuf += rs_b;  //move b pointer to next row

        xmm1 = _mm_broadcast_ss((abuf + 0*rs_a)); //broadcast c0r0
        xmm2 = _mm_broadcast_ss((abuf + 1*rs_a)); //broadcast c0r0
        xmm3 = _mm_broadcast_ss((abuf + 2*rs_a)); //broadcast c0r0

        xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);
        xmm5 = _mm_fmadd_ps(xmm0, xmm2, xmm5);
        xmm6 = _mm_fmadd_ps(xmm0, xmm3, xmm6);

        xmm1 = _mm_broadcast_ss((abuf + 3*rs_a)); //broadcast c0r0
        xmm2 = _mm_broadcast_ss((abuf + 4*rs_a)); //broadcast c0r0
        xmm3 = _mm_broadcast_ss((abuf + 5*rs_a)); //broadcast c0r0
        abuf += cs_a;  //move a pointer to next col

        xmm7 = _mm_fmadd_ps(xmm0, xmm1, xmm7);
        xmm8 = _mm_fmadd_ps(xmm0, xmm2, xmm8);
        xmm9 = _mm_fmadd_ps(xmm0, xmm3, xmm9);
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
          BF16_F32_C_BNZ_1(5,0,xmm1,xmm3,xmm9)
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
          _cbuf += rs_c;
          F32_C_BNZ_1(_cbuf,rs_c,xmm1,xmm3,xmm9)
        }
      }

      // Post Ops
      lpgemm_post_op* post_ops_list_temp = post_ops_list;
      POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_6x1F:
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

          // c[5,0-3]
          xmm9 = _mm_add_ps( xmm9, xmm0 );
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
            BF16_F32_BIAS_BCAST_LT4BF16_AVX2(xmm1, 5);
          }
          else
          {
            xmm0 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
                post_ops_attr.post_op_c_i + 4 );
            xmm1 = _mm_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
                post_ops_attr.post_op_c_i + 5 );
          }

          // c[4,0-3]
          xmm8 = _mm_add_ps( xmm8, xmm0 );

          // c[5,0-3]
          xmm9 = _mm_add_ps( xmm9, xmm1 );
        }
        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_RELU_6x1F:
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

        // c[5,0-3]
        xmm9 = _mm_max_ps( xmm9, xmm0 );

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_RELU_SCALE_6x1F:
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

        // c[5,0-3]
        RELU_SCALE_OP_F32S_SSE(xmm9, xmm0, xmm1, xmm2)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_GELU_TANH_6x1F:
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

        // c[5,0-3]
        GELU_TANH_F32S_SSE(xmm9, xmm0, xmm1, xmm2, xmm3, dn, x_tanh, q)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_GELU_ERF_6x1F:
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

        // c[5,0-3]
        GELU_ERF_F32S_SSE(xmm9, xmm0, xmm1, xmm2)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_CLIP_6x1F:
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

        // c[5,0-3]
        CLIP_F32S_SSE(xmm9, xmm0, xmm1)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_DOWNSCALE_6x1F:
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
              BF16_F32_ZP_VECTOR_1LOAD_SSE(zero_point0,0)
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

          //c[5, 0-3]
          F32_SCL_MULRND_SSE(xmm9, selector0, zero_point0);
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

          if( post_ops_list_temp->scale_factor_len > 1 )
          {
            selector0 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                            post_ops_attr.post_op_c_i + 4 ) );
            selector1 = _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                            post_ops_attr.post_op_c_i + 5 ) );
          }
          if( *( (dim_t* )post_ops_list_temp->op_args3 ) > 1 )
          {
            if( is_bf16 == TRUE )
            {
              BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point0,4)
              BF16_F32_ZP_VECTOR_BCAST_SSE(zero_point1,5)
            }
            else
            {
              zero_point0 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                              post_ops_attr.post_op_c_i + 4 ) );
              zero_point1 = _mm_set1_ps( *( (float* )post_ops_list_temp->op_args1 +
                              post_ops_attr.post_op_c_i + 5) );
            }
          }
          //c[4, 0-3]
          F32_SCL_MULRND_SSE(xmm8, selector0, zero_point0);

          //c[5, 0-3]
          F32_SCL_MULRND_SSE(xmm9, selector1, zero_point1);
        }
        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_MATRIX_ADD_6x1F:
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
        __m128 scl_fctr6 = _mm_setzero_ps();

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
          scl_fctr6 =
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
            scl_fctr6 =
              _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 5 ) );
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

            // c[5:0-15]
            BF16_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,5,9);
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

            // c[5:0-15]
            BF16_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr6,5,9);
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

            // c[5:0-15]
            F32_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr1,5,9);
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

            // c[5:0-15]
            F32_F32_MATRIX_ADD_1COL_XMM_1ELE(xmm1,scl_fctr6,5,9);
          }
        }
        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_MATRIX_MUL_6x1F:
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
        __m128 scl_fctr6 = _mm_setzero_ps();

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
          scl_fctr6 =
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
            scl_fctr6 =
              _mm_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                  post_ops_attr.post_op_c_i + 5 ) );
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

            // c[5:0-15]
            BF16_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,5,9);
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

            // c[5:0-15]
            BF16_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr6,5,9);
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

            // c[5:0-15]
            F32_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr1,5,9);
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

            // c[5:0-15]
            F32_F32_MATRIX_MUL_1COL_XMM_1ELE(xmm1,scl_fctr6,5,9);
          }
        }
        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_SWISH_6x1F:
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

          // c[5,0-0]
          SWISH_F32_SSE_DEF(xmm9, xmm0, xmm1, xmm2, xmm3, z, dn, ex_out)

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_TANH_6x1F:
      {
        __m128 dn;
        __m128i q;

        // c[0,0-0]
        TANH_F32S_SSE(xmm4, xmm0, xmm1, xmm2, xmm3, dn, q)

        // c[1,0-0]
        TANH_F32S_SSE(xmm5, xmm0, xmm1, xmm2, xmm3, dn, q)

        // c[2,0-0]
        TANH_F32S_SSE(xmm6, xmm0, xmm1, xmm2, xmm3, dn, q)

        // c[3,0-0]
        TANH_F32S_SSE(xmm7, xmm0, xmm1, xmm2, xmm3, dn, q)

        // c[4,0-0]
        TANH_F32S_SSE(xmm8, xmm0, xmm1, xmm2, xmm3, dn, q)

        // c[5,0-0]
        TANH_F32S_SSE(xmm9, xmm0, xmm1, xmm2, xmm3, dn, q)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_SIGMOID_6x1F:
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

          // c[5,0-0]
          SIGMOID_F32_SSE_DEF(xmm9, xmm1, xmm2, xmm3, z, dn, ex_out)

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_6x1F_DISABLE:
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
        STORE_F32_BF16_1XMM(xmm9, 5, 0)
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
        cbuf += rs_c;
        _mm_store_ss(cbuf, xmm9);
      }

      post_ops_attr.post_op_c_i += MR;
    }//mloop

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        float*  restrict cij = (float *) c + i_edge*rs_c;
        float*  restrict ai  = (float *) a + m_iter*ps_a;
        float*  restrict bj  = (float *) b;

        lpgemm_m_fringe_f32_ker_ft ker_fps[6] =
        {
          NULL,
          lpgemm_rowvar_f32f32f32of32_1x1,
          lpgemm_rowvar_f32f32f32of32_2x1,
          lpgemm_rowvar_f32f32f32of32_3x1,
          lpgemm_rowvar_f32f32f32of32_4x1,
          lpgemm_rowvar_f32f32f32of32_5x1
        };

        lpgemm_m_fringe_f32_ker_ft ker_fp = ker_fps[ m_left ];

        ker_fp
        (
          k0,
          ai, rs_a, cs_a,
          bj, rs_b, cs_b,
          cij,rs_c,
          alpha, beta,
          post_ops_list, post_ops_attr
        );
        return;
    }
}
#endif
